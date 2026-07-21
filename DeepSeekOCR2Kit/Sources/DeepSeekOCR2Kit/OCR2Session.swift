// The public streaming facade over the `DeepSeekOCR2Model` + `DeepSeekOCR2Processor`
// building blocks: one `load` to bring up model + tokenizer, one `ocr` that
// streams decoded text token-by-token. This is the surface sub-project ② (the
// app's `ModelManager`) consumes.

import CoreGraphics
import Foundation
import MLX
import MLXLMCommon

/// What to ask the model for.
///
/// The prompt templates are the reference's exact strings (see
/// `OCR2Session.prompt(for:)`): `.freeOCR` reproduces the fixtures' verified
/// `"<image>\nFree OCR. "`; `.grounding` uses the reference README's
/// "Text Localization (Grounding)" template, the only documented prompt with a
/// user-supplied slot.
public enum OCRTask: Sendable {
    /// Plain text extraction, no layout markers (reference `"Free OCR."`).
    case freeOCR
    /// Locate `query` in the image and return its bounding box(es); the response
    /// carries `<|ref|>…<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>` markers, parse it
    /// with `OCR2Session.parseGrounding`.
    case grounding(query: String)
}

/// Streaming DeepSeek-OCR-2 inference on Apple Silicon (mlx-swift). Combines the
/// image processor and the quantized model behind a two-call API.
///
/// `@unchecked Sendable`: the model (an `MLXNN.Module`) and its `MLXArray`
/// weights are not `Sendable`, but a session is immutable after `load` and each
/// `ocr` call allocates its own KV cache, so the only shared state read across
/// the internal generation `Task` is the read-only weight set. Concurrent
/// `ocr` calls on one session are NOT supported (MLX compute is not reentrant
/// here) -- drive one image at a time, or use one session per concurrent stream.
public final class OCR2Session: @unchecked Sendable {
    private let model: DeepSeekOCR2Model
    private let processor: DeepSeekOCR2Processor
    private let config: DeepSeekOCR2Configuration

    /// Generous default generation cap. A full page of dense OCR is well under
    /// this; grounding responses are far shorter. Callers can override per call.
    public static let defaultMaxTokens = 4096

    private init(
        model: DeepSeekOCR2Model, processor: DeepSeekOCR2Processor,
        config: DeepSeekOCR2Configuration
    ) {
        self.model = model
        self.processor = processor
        self.config = config
    }

    /// Loads the quantized checkpoint and tokenizer from `dir` (a local
    /// snapshot of `mlx-community/DeepSeek-OCR-2-8bit`). `progress` reports
    /// weight-shard load progress in `0...1`.
    public static func load(
        from dir: URL, progress: @Sendable (Double) -> Void = { _ in }
    ) async throws -> OCR2Session {
        let (model, config) = try await DeepSeekOCR2Model.load(from: dir, progress: progress)
        let processor = try await DeepSeekOCR2Processor(modelDir: dir)
        return OCR2Session(model: model, processor: processor, config: config)
    }

    /// Runs OCR on `image`, streaming decoded text as it is generated. Emission
    /// is CJK/multibyte-safe (buffers partial UTF-8 until a full character is
    /// available). The stream finishes when the model emits EOS, when
    /// `maxTokens` is reached, or when the consuming task is cancelled.
    ///
    /// - Note: `.grounding` output is streamed with its `<|ref|>`/`<|det|>`
    ///   markers intact; collect the full text and feed it to `parseGrounding`.
    public func ocr(
        image: CGImage, task: OCRTask = .freeOCR,
        maxTokens: Int = OCR2Session.defaultMaxTokens
    ) -> AsyncThrowingStream<String, Error> {
        // CGImage is immutable/Sendable; capturing it into the generation Task
        // is safe.
        return AsyncThrowingStream { continuation in
            let work = Task {
                do {
                    try self.generate(
                        image: image, task: task, maxTokens: maxTokens,
                        isCancelled: { Task.isCancelled },
                        emit: { continuation.yield($0) })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in work.cancel() }
        }
    }

    /// Parses `<|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>` pairs from a
    /// grounding response into normalized `[0,1]` boxes. Malformed/partial pairs
    /// are dropped. See `GroundingParser`.
    public static func parseGrounding(_ text: String) -> [GroundingBox] {
        GroundingParser.parse(text)
    }

    // MARK: - Generation

    /// Synchronous greedy decode with a KV cache, mirroring `FixtureSupport
    /// .greedyDecode` (the parity-verified loop) but emitting text incrementally
    /// through `emit`. Checks `isCancelled` each step and returns cleanly.
    private func generate(
        image: CGImage, task: OCRTask, maxTokens: Int,
        isCancelled: () -> Bool, emit: (String) -> Void
    ) throws {
        let input = try processor.prepare(image: image, prompt: Self.prompt(for: task))
        let embeds = model.inputEmbeddings(
            tokens: input.tokens, pixelsGlobal: input.pixelsGlobal,
            pixelsPatches: input.pixelsPatches, seqMask: input.seqMask
        ).asType(.bfloat16)

        // Free OCR matches the reference's skip-specials decode; grounding keeps
        // the markers (they are special:true -- see decodeKeepingMarkers).
        let skipSpecials: Bool = { if case .grounding = task { return false } else { return true } }()
        var detok = StreamingDetokenizer(processor: processor, skipSpecialTokens: skipSpecials)

        let cache = model.newCache()
        var logits = model(inputEmbeds: embeds, cache: cache)

        // Optional throughput diagnostic (stderr, off by default): set
        // OCR2_STATS=1 to print `[N tokens, Ts, R tok/s]` after generation.
        let wantStats = ProcessInfo.processInfo.environment["OCR2_STATS"] != nil
        let start = Date()
        var generated = 0

        for _ in 0..<maxTokens {
            if isCancelled() { break }
            let next = MLX.argMax(logits[0..., -1, 0...], axis: -1)
            let nextID = next.item(Int.self)
            if nextID == config.eosTokenID { break }
            generated += 1
            if let chunk = detok.append(nextID) { emit(chunk) }
            logits = model(inputEmbeds: model.embed(next.reshaped(1, 1)), cache: cache)
        }

        if wantStats {
            let dt = Date().timeIntervalSince(start)
            let tps = dt > 0 ? Double(generated) / dt : 0
            let line = String(format: "\n[%d tokens, %.2fs, %.1f tok/s]\n", generated, dt, tps)
            FileHandle.standardError.write(line.data(using: .utf8)!)
        }
    }

    /// Reference-exact prompt for each task.
    ///
    /// - `.freeOCR` -> `"<image>\nFree OCR. "` (the verified fixture prompt; the
    ///   trailing space is the reference chat template's `content + " "`).
    /// - `.grounding(q)` -> `"<image>\nLocate <|ref|>\(q)<|/ref|> in the image. "`,
    ///   the reference README's "Text Localization (Grounding)" template.
    static func prompt(for task: OCRTask) -> String {
        switch task {
        case .freeOCR:
            return "<image>\nFree OCR. "
        case .grounding(let query):
            return "<image>\nLocate <|ref|>\(query)<|/ref|> in the image. "
        }
    }
}

/// Incremental, CJK/multibyte-safe detokenizer, a port of MLXLMCommon's
/// `NaiveStreamingDetokenizer` onto the processor's swift-transformers
/// tokenizer with an explicit `skipSpecialTokens` (the library type hardcodes
/// its own `Tokenizer` protocol and a fixed decode mode, so it can't preserve
/// grounding markers).
///
/// It re-decodes the running token buffer each step and returns only the newly
/// completed suffix; a trailing U+FFFD (replacement char) means the last token
/// did not finish a Unicode scalar, so nothing is emitted until it does. The
/// buffer resets after each newline (keeping the last token as a decode seed) to
/// bound re-decode cost -- identical to the reference algorithm.
private struct StreamingDetokenizer {
    private let processor: DeepSeekOCR2Processor
    private let skipSpecialTokens: Bool
    private var segmentTokens: [Int] = []
    private var segment = ""

    init(processor: DeepSeekOCR2Processor, skipSpecialTokens: Bool) {
        self.processor = processor
        self.skipSpecialTokens = skipSpecialTokens
    }

    private func decode(_ ids: [Int]) -> String {
        processor.decode(ids, skipSpecialTokens: skipSpecialTokens)
    }

    /// Appends `token` and returns any newly-complete text, or `nil` if the
    /// token only extended an in-progress multibyte character.
    mutating func append(_ token: Int) -> String? {
        segmentTokens.append(token)
        let newSegment = decode(segmentTokens)
        let new = newSegment.suffix(max(0, newSegment.count - segment.count))

        if new.last == "\u{fffd}" { return nil }  // incomplete Unicode scalar

        if new.hasSuffix("\n") {
            // Reset the segment; keep the last token so the next decode has the
            // byte context to continue correctly.
            let last = segmentTokens.removeLast()
            segmentTokens = [last]
            segment = decode(segmentTokens)
        } else {
            segment = newSegment
        }
        return new.isEmpty ? nil : String(new)
    }
}
