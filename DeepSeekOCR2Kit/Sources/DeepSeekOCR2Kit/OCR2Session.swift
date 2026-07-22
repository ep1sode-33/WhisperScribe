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

/// Errors surfaced through an `OCR2Session.ocr` stream.
public enum OCR2SessionError: Error, Equatable {
    /// `maxTokens` was < 1; generation needs at least one step. The stream
    /// finishes with this error before any work is done.
    case invalidMaxTokens(Int)
}

/// Streaming DeepSeek-OCR-2 inference on Apple Silicon (mlx-swift). Combines the
/// image processor and the quantized model behind a two-call API.
///
/// `@unchecked Sendable`: the model (an `MLXNN.Module`) and its `MLXArray`
/// weights are not `Sendable`, but a session is immutable after `load` and each
/// `ocr` call allocates its own KV cache, so the only shared state read across
/// the internal generation `Task` is the read-only weight set.
///
/// **Concurrency (serialized):** MLX compute on one session is not reentrant, so
/// generation is *serialized* by an internal single-flight gate. Overlapping
/// `ocr` calls do not run at the same time -- they queue, and each waits for the
/// previous generation `Task` to fully exit before starting, including that
/// task's cancellation cleanup (so a cancelled request can never overlap the
/// next one still finishing an MLX op). This makes concurrent `ocr` calls *safe*
/// (they no longer corrupt each other) but not parallel; use one session per
/// stream if you need true parallelism.
public final class OCR2Session: @unchecked Sendable {
    private let model: DeepSeekOCR2Model
    private let processor: DeepSeekOCR2Processor
    private let config: DeepSeekOCR2Configuration
    /// Serializes generation across concurrent `ocr` calls (see the type doc).
    private let gate = GenerationGate()

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
    /// - Parameter maxTokens: generation cap (default ``defaultMaxTokens`` ==
    ///   4096; a dense full page is well under it). Must be >= 1 -- a value < 1
    ///   finishes the stream immediately with
    ///   ``OCR2SessionError/invalidMaxTokens(_:)``. Reaching the cap ends the
    ///   stream *indistinguishably* from an EOS stop: there is no
    ///   termination-reason signal (that API is deferred to app integration).
    ///   If you must know whether output was truncated, compare the token/char
    ///   count you received against your cap.
    ///
    /// - Note: `.grounding` output is streamed with its `<|ref|>`/`<|det|>`
    ///   markers intact; collect the full text and feed it to `parseGrounding`.
    ///
    /// - Note: **Eager producer, unbounded buffer.** Generation starts as soon
    ///   as the returned stream is created (subject to the serialization gate --
    ///   see the type doc), not lazily on first `next()`. Chunks are buffered
    ///   with `AsyncThrowingStream`'s default `.unbounded` policy, so a slow
    ///   consumer does NOT backpressure generation -- memory is bounded only by
    ///   the produced text. Cancelling the consuming task stops generation
    ///   (checked before preprocessing, before prefill, and each decode step)
    ///   and releases the gate. A pull-based `unfolding` variant was considered
    ///   and rejected: it would force the synchronous, KV-cache-stateful decode
    ///   loop into a resumable state machine, contorting the parity-verified
    ///   path for a bound (max-tokens) that is already small and caller-set.
    public func ocr(
        image: CGImage, task: OCRTask = .freeOCR,
        maxTokens: Int = OCR2Session.defaultMaxTokens
    ) -> AsyncThrowingStream<String, Error> {
        // CGImage is immutable/Sendable; capturing it into the generation Task
        // is safe.
        return AsyncThrowingStream { continuation in
            guard maxTokens >= 1 else {
                continuation.finish(throwing: OCR2SessionError.invalidMaxTokens(maxTokens))
                return
            }
            let work = Task {
                // Serialize against any other in-flight generation on this
                // session (MLX is not reentrant). The permit is released only
                // after generation has fully finished -- on every path: success,
                // a thrown error, or cancellation (generation returns cleanly on
                // cancel) -- so the next request can never overlap a cancelled
                // one still finishing its final MLX op. `catch` swallows every
                // error, so the trailing `release()` is always reached.
                await self.gate.acquire()
                do {
                    try Task.checkCancellation()
                    try self.generate(
                        image: image, task: task, maxTokens: maxTokens,
                        isCancelled: { Task.isCancelled },
                        emit: { continuation.yield($0) })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                await self.gate.release()
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
        // N4: bail before the (expensive) preprocessing if already cancelled.
        if isCancelled() { return }
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
        // N4: bail before the prefill pass (the single most expensive MLX op).
        if isCancelled() { return }
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

/// A FIFO async single-flight gate (an `AsyncSemaphore(value: 1)`), implemented
/// as an actor so its state is serialized without a manual lock (the Swift-6
/// `noasync` locks are awkward to hold across the continuation append anyway).
/// `acquire()` suspends the caller until the gate is free; the holder MUST call
/// `release()` exactly once when its generation work has fully finished.
/// `OCR2Session.ocr` releases on every exit path (success, thrown error, and
/// cancellation -- generation returns cleanly on cancel), so the gate is always
/// freed for the next request and never held past the generation task's exit.
///
/// A waiter cancelled while suspended is not dequeued eagerly -- it is resumed
/// in turn by the previous holder's `release()`, then bails out of the
/// (cancellation-checked) generation immediately, so no continuation leaks and
/// no request ever overlaps another.
actor GenerationGate {
    private var locked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !locked {
            locked = true
            return
        }
        await withCheckedContinuation { cont in
            waiters.append(cont)
        }
    }

    func release() {
        if waiters.isEmpty {
            locked = false
        } else {
            // Hand the permit straight to the next waiter (`locked` stays true).
            waiters.removeFirst().resume()
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
/// bound re-decode cost -- identical to the reference algorithm -- plus a
/// newline-independent checkpoint flush (N5, see `append`) so a long line does
/// not degrade to O(n^2).
///
/// `internal` (not `private`) so the flush behavior can be unit-tested with an
/// injected `flushThreshold`.
struct StreamingDetokenizer {
    private let processor: DeepSeekOCR2Processor
    private let skipSpecialTokens: Bool
    /// Once `segmentTokens` grows past this many tokens, a clean already-emitted
    /// prefix is flushed to bound the per-step re-decode cost. `.max` disables
    /// flushing (used by tests to get the un-checkpointed reference output).
    private let flushThreshold: Int
    /// Tokens kept as a decode seed after a checkpoint flush (byte-level BPE
    /// needs a little context to keep decoding a multibyte run correctly).
    private let overlapTail: Int
    private var segmentTokens: [Int] = []
    private var segment = ""

    init(
        processor: DeepSeekOCR2Processor, skipSpecialTokens: Bool,
        flushThreshold: Int = 256, overlapTail: Int = 32
    ) {
        self.processor = processor
        self.skipSpecialTokens = skipSpecialTokens
        self.flushThreshold = flushThreshold
        self.overlapTail = overlapTail
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
            // N5: a long line with no newline would otherwise grow
            // `segmentTokens` without bound, making each re-decode O(n) (so the
            // line as a whole is O(n^2)). Once past the threshold, drop the
            // already-emitted prefix and keep a short overlap tail as a decode
            // seed -- but ONLY when that tail re-decodes to a clean suffix of the
            // current segment. That guarantees the drop boundary sits on a
            // character boundary, so the flush cannot change any emitted text.
            // A byte-level straddle leaves a leading U+FFFD in `tailText`,
            // failing `hasSuffix`, so we just keep growing until a clean split
            // appears (correctness always beats the cost bound).
            if segmentTokens.count > flushThreshold {
                let tail = Array(segmentTokens.suffix(overlapTail))
                let tailText = decode(tail)
                if !tailText.isEmpty, segment.hasSuffix(tailText) {
                    segmentTokens = tail
                    segment = tailText
                }
            }
        }
        return new.isEmpty ? nil : String(new)
    }
}
