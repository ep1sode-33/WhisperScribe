import Testing
import Foundation
import MLX
@testable import DeepSeekOCR2Kit

/// Mask-first TDD per task-6-brief.md risk #1: a wrong attention mask does
/// not crash the encoder, it silently produces garbage OCR. The pure-logic
/// quadrant tests and the Python-fixture comparison tests are checked BEFORE
/// the full encoder parity test is trusted -- if the fixture comparison
/// fails, the mask is fixed, never the test (brief Step 4).
@Suite struct VisionEncoderTests {
    @Test(arguments: [144, 256])
    func maskQuadrants(n: Int) throws {
        let m = Qwen2VisionEncoder.attentionMask(imageTokens: n, queries: n)
        #expect(m.shape == [2 * n, 2 * n])
        #expect(m[0, n - 1].item(Float.self) == 0)          // img→img bidirectional
        #expect(m[0, n].item(Float.self) <= -1e8)            // img→query blocked
        #expect(m[n, 0].item(Float.self) == 0)               // query→img fully open
        #expect(m[n, n].item(Float.self) == 0)               // query diagonal
        #expect(m[n, n + 1].item(Float.self) <= -1e8)         // query upper-triangle blocked
        #expect(m[2 * n - 1, n].item(Float.self) == 0)        // query lower-triangle visible
    }

    @Test(.enabled(if: FixtureSupport.root != nil), arguments: [144, 256])
    func maskMatchesPythonFixture(n: Int) throws {
        let ref = try FixtureSupport.load("masks", "mask_\(n)")["mask"]!
        let m = Qwen2VisionEncoder.attentionMask(imageTokens: n, queries: n)
        #expect(allClose(m, ref).item(Bool.self))
    }

    // Config correction (binding, per Task 6 controller notes / task-5
    // precedent): DeepSeekOCR2Configuration.default mirrors the Python
    // dataclass defaults, not necessarily the real downloaded checkpoint, so
    // parity tests build the config via `mergingJSON:` over the real
    // checkpoint's config.json (project convention since Task 3), not
    // `.default`.
    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func encoderOutputParity() throws {
        let modelDir = try #require(FixtureSupport.modelDir)
        let config = try DeepSeekOCR2Configuration(
            mergingJSON: Data(contentsOf: modelDir.appending(path: "config.json")))

        let sam = try FixtureSupport.load("doc_page", "sam_out")["x"]!
        let ref = try FixtureSupport.load("doc_page", "encoder_out")["x"]!
        let enc = Qwen2VisionEncoder(config.qwen2Encoder)
        try TestWeights.load(into: enc, prefix: "vision_model.qwen2_encoder.")
        let out = enc(samFeatures: sam.asType(.bfloat16)).asType(.float32)
        #expect(out.shape == ref.shape)
        let relErr = ((out - ref).abs().mean() / (ref.abs().mean() + 1e-6)).item(Float.self)
        #expect(relErr < 2e-2, "rel err \(relErr)")
    }
}
