import Testing
import Foundation
@testable import DeepSeekOCR2Kit

/// Milestone M6: the grounding parser + marker-preserving decode.
///
/// The parser tests are pure logic (regex over synthetic strings) and run
/// everywhere, including CI without fixtures -- there is intentionally NO golden
/// fixture for grounding output (all 5 fixtures were generated with the
/// `"<image>\nFree OCR. "` prompt, so the model never emitted `<|det|>` markers
/// under them). The decode test is fixture-gated because it needs the real
/// tokenizer to prove the marker ids survive a `skipSpecialTokens:false` decode.
@Suite struct GroundingParserTests {
    @Test func parsesRefDetPairs() {
        let s = "<|ref|>Mocha<|/ref|><|det|>[[120, 340, 560, 400]]<|/det|>"
        let boxes = OCR2Session.parseGrounding(s)
        #expect(boxes.count == 1 && boxes[0].label == "Mocha")
        #expect(abs(boxes[0].box.minX - 0.120) < 1e-9)   // 0–1000 → /1000 归一化
        #expect(abs(boxes[0].box.maxY - 0.400) < 1e-9)
    }

    @Test func ignoresMalformed() {
        #expect(OCR2Session.parseGrounding("<|det|>[[1,2,3]]<|/det|>").isEmpty)
        #expect(OCR2Session.parseGrounding("no markers").isEmpty)
    }

    /// C4: coordinate sanity. Reversed corners (`x2 < x1` / `y2 < y1`) and
    /// out-of-range (`> 1000`) boxes are dropped; in-range ordered boxes parse.
    @Test func dropsOutOfRangeAndReversedBoxes() {
        // reversed x (x2 < x1)
        #expect(OCR2Session.parseGrounding("<|ref|>A<|/ref|><|det|>[[500, 10, 100, 800]]<|/det|>").isEmpty)
        // reversed y (y2 < y1)
        #expect(OCR2Session.parseGrounding("<|ref|>A<|/ref|><|det|>[[10, 800, 500, 100]]<|/det|>").isEmpty)
        // x2 > 1000
        #expect(OCR2Session.parseGrounding("<|ref|>A<|/ref|><|det|>[[10, 10, 1001, 800]]<|/det|>").isEmpty)
        // y2 > 1000
        #expect(OCR2Session.parseGrounding("<|ref|>A<|/ref|><|det|>[[10, 10, 500, 1001]]<|/det|>").isEmpty)

        // a valid box still parses (and boundary value 1000 is allowed)
        let boxes = OCR2Session.parseGrounding("<|ref|>A<|/ref|><|det|>[[0, 0, 1000, 1000]]<|/det|>")
        #expect(boxes.count == 1)
        #expect(abs(boxes[0].box.width - 1.0) < 1e-9)
        #expect(abs(boxes[0].box.height - 1.0) < 1e-9)

        // one malformed + one valid: only the valid one survives.
        let mixed = OCR2Session.parseGrounding(
            "<|ref|>bad<|/ref|><|det|>[[900, 10, 100, 20]]<|/det|> "
                + "<|ref|>good<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>")
        #expect(mixed.count == 1 && mixed[0].label == "good")
    }

    /// Multiple ref/det pairs interleaved with prose (the shape a real
    /// `<|grounding|>` / Locate response takes) all parse; the CGRect geometry
    /// is `(x1, y1, x2-x1, y2-y1)` normalized by 1000.
    @Test func parsesMultiplePairs() {
        let s = "prefix <|ref|>A<|/ref|><|det|>[[0, 0, 500, 1000]]<|/det|> mid "
            + "<|ref|>B B<|/ref|><|det|>[[250,250,750,750]]<|/det|> tail"
        let boxes = OCR2Session.parseGrounding(s)
        #expect(boxes.count == 2)
        #expect(boxes[0].label == "A")
        #expect(abs(boxes[0].box.width - 0.5) < 1e-9)
        #expect(abs(boxes[0].box.height - 1.0) < 1e-9)
        #expect(boxes[1].label == "B B")
        #expect(abs(boxes[1].box.minX - 0.25) < 1e-9)
        #expect(abs(boxes[1].box.maxX - 0.75) < 1e-9)
    }

    /// Marker-preserving decode (Task 9 review finding): the grounding markers
    /// are `special:true` in tokenizer.json, so the default `skipSpecialTokens:
    /// true` decode STRIPS them; the grounding path must decode with
    /// `skipSpecialTokens:false` and drop only BOS(0)/EOS(1). Proves both halves
    /// on the real tokenizer.
    @Test(.enabled(if: FixtureSupport.modelDir != nil))
    func groundingDecodePreservesMarkers() async throws {
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)
        // 0=BOS, 1=EOS, 128816..128819 = <|ref|> <|/ref|> <|det|> <|/det|>.
        let ids = [0, 128816, 128817, 128818, 128819, 1]

        let kept = p.decodeKeepingMarkers(ids)
        #expect(kept.contains("<|ref|>"))
        #expect(kept.contains("<|/ref|>"))
        #expect(kept.contains("<|det|>"))
        #expect(kept.contains("<|/det|>"))
        // BOS/EOS literal surface forms must be gone.
        #expect(!kept.contains("begin"))
        #expect(!kept.contains("end"))

        // Contrast: the default skip-specials decode drops the markers entirely.
        let stripped = p.decode([128816, 128817, 128818, 128819])
        #expect(!stripped.contains("<|ref|>"))
    }
}
