import Testing
import Foundation
import CoreGraphics
import ImageIO
import MLX
@testable import DeepSeekOCR2Kit

/// Drains an `ocr` stream to a single String. Free function (captures no test
/// `self`) so it is safe to run from an `async let` child task.
private func collectStream(_ stream: AsyncThrowingStream<String, Error>) async throws -> String {
    var out = ""
    for try await chunk in stream { out += chunk }
    return out
}

/// Milestone M6: the streaming facade end to end. Fixture-gated (needs the real
/// checkpoint): drives a real `CGImage` through `OCR2Session.ocr` and requires
/// the streamed, incrementally-detokenized text to reproduce the reference's
/// known doc_page prefix -- proving the streaming greedy loop + CJK-safe
/// detokenizer match the parity-verified batch path.
@Suite struct OCR2SessionTests {
    static func cgImage(_ name: String) throws -> CGImage {
        let url = FixtureSupport.root!.appending(path: "images/\(name).png")
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "cgimage-load", code: 1) }
        return img
    }

    /// The prompt-template helper is pure logic (no model): assert the two
    /// reference-exact strings so a template regression is caught in CI too.
    @Test func promptTemplates() {
        #expect(OCR2Session.prompt(for: .freeOCR) == "<image>\nFree OCR. ")
        #expect(
            OCR2Session.prompt(for: .grounding(query: "Total"))
                == "<image>\nLocate <|ref|>Total<|/ref|> in the image. ")
    }

    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func streamsDocPagePrefix() async throws {
        let session = try await OCR2Session.load(from: FixtureSupport.modelDir!)
        let cg = try Self.cgImage("doc_page")

        var out = ""
        for try await chunk in session.ocr(image: cg, task: .freeOCR, maxTokens: 40) {
            out += chunk
        }

        let trimmed = out.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(!trimmed.isEmpty, "stream produced no text")
        #expect(
            trimmed.hasPrefix("DeepSeekOCR2Kit Parity Test Document"),
            "stream prefix mismatch:\n\(trimmed)")
    }

    /// N1: two overlapping `ocr` calls on ONE session must both complete with the
    /// correct prefix -- the single-flight gate serializes their (non-reentrant)
    /// MLX compute so neither corrupts the other.
    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func concurrentOCRCallsBothComplete() async throws {
        let session = try await OCR2Session.load(from: FixtureSupport.modelDir!)
        let cg = try Self.cgImage("doc_page")

        async let a = collectStream(session.ocr(image: cg, task: .freeOCR, maxTokens: 40))
        async let b = collectStream(session.ocr(image: cg, task: .freeOCR, maxTokens: 40))
        let (ra, rb) = try await (a, b)

        for r in [ra, rb] {
            let t = r.trimmingCharacters(in: .whitespacesAndNewlines)
            #expect(
                t.hasPrefix("DeepSeekOCR2Kit Parity Test Document"),
                "concurrent stream prefix mismatch:\n\(t)")
        }
    }

    /// A3: `maxTokens < 1` finishes the stream immediately with a typed error,
    /// before any generation work. Needs a session (tokenizer + weights) but the
    /// image is never processed.
    @Test(.enabled(if: FixtureSupport.modelDir != nil))
    func invalidMaxTokensYieldsError() async throws {
        let session = try await OCR2Session.load(from: FixtureSupport.modelDir!)
        let img = ProcessorTests.rgbaImage(width: 8, height: 8, pixel: (255, 255, 255, 255))
        await #expect(throws: OCR2SessionError.self) {
            for try await _ in session.ocr(image: img, maxTokens: -1) {}
        }
    }

    /// N5: the streaming detokenizer's periodic checkpoint flush must not change
    /// emitted text. Feed a long newline-free run of a real multibyte token (so
    /// the segment buffer grows past the 256-token threshold and actually
    /// flushes) into a checkpointing detok and a never-flushing one; the outputs
    /// must be byte-identical. Uses the tokenizer only (no model weights).
    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func detokenizerCheckpointFlushPreservesText() async throws {
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)

        // A real content token that decodes to a non-empty, newline-free,
        // complete character -- repeating it forces the no-newline growth path.
        let base = try FixtureSupport.load("cjk_dense", "greedy_tokens")["ids"]!
        let baseIDs = (0..<base.shape[0]).map { Int(base[$0].item(Int32.self)) }
        let safe = baseIDs.first { id in
            let s = p.decode([id], skipSpecialTokens: true)
            return !s.isEmpty && !s.contains("\n") && !s.contains("\u{fffd}")
        }
        let token = try #require(safe, "no newline-free content token in fixture")
        let ids = Array(repeating: token, count: 400)  // > 256 threshold

        var flushing = StreamingDetokenizer(
            processor: p, skipSpecialTokens: true, flushThreshold: 256, overlapTail: 32)
        var never = StreamingDetokenizer(
            processor: p, skipSpecialTokens: true, flushThreshold: .max, overlapTail: 32)
        var a = "", b = ""
        for id in ids {
            if let c = flushing.append(id) { a += c }
            if let c = never.append(id) { b += c }
        }
        #expect(a == b, "checkpoint flush changed emitted text")
        #expect(a.count > 256, "expected a long newline-free run (got \(a.count) chars)")
    }
}
