import Testing
import Foundation
import CoreGraphics
import ImageIO
@testable import DeepSeekOCR2Kit

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
}
