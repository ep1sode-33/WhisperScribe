import Testing
import Foundation
import MLX
@testable import DeepSeekOCR2Kit

/// NOTE on the brief vs. reality (see task-5-report.md for the full
/// rationale): `DeepSeekOCR2Configuration.default` mirrors the Python
/// dataclass defaults, not necessarily the real downloaded model, so the
/// config used here comes from `DeepSeekOCR2Configuration(mergingJSON:)`
/// over the real checkpoint's `config.json` (its `.sam` field). The real
/// checkpoint's `sam_model.*` weights sit at the TOP LEVEL of the namespace
/// (a sibling of `vision_model.*`, verified against `model.safetensors.
/// index.json` and `deepseekocr_2.py:Model.__init__`'s `self.sam_model =
/// SAMEncoder(...)`), so the loader prefix is `"sam_model."`, not
/// `"vision_model.sam_model."`.
@Suite struct SAMParityTests {
    static func realConfig() throws -> DeepSeekOCR2Configuration {
        let dir = try #require(FixtureSupport.modelDir)
        return try DeepSeekOCR2Configuration(
            mergingJSON: Data(contentsOf: dir.appending(path: "config.json")))
    }

    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func globalPathMatchesReference() throws {
        let pix = try FixtureSupport.load("doc_page", "pixels_global")["x"]!
        let ref = try FixtureSupport.load("doc_page", "sam_out")["x"]!

        // pixels_global.x is stored exactly as the processor returns it:
        // channel-first (1, 3, 1024, 1024) float32 (see gen_fixtures.py's
        // "deviations from the original brief draft" note). The Python
        // reference transposes to channel-last right before calling
        // sam_model (`global_hwc = pix_g.transpose(0, 2, 3, 1)` --
        // gen_fixtures.py ~line 168, mirroring
        // `deepseekocr_2.py:Model.get_input_embeddings`'s own
        // `.transpose(0, 2, 3, 1)`), so the test must apply the identical
        // NCHW -> NHWC transform to feed the encoder exactly what the
        // reference fed it.
        let pixHWC = pix.transposed(0, 2, 3, 1)

        let config = try Self.realConfig()
        let sam = SAMEncoder(config.sam)
        try TestWeights.load(into: sam, prefix: "sam_model.")

        // The reference itself casts pixel tensors to bf16 before feeding
        // them through sam_model (`processing_deepseekocr.py`:
        // `global_tensor = self.image_transform(global_view)
        // .astype(mx.bfloat16)`), so bf16 here reproduces the reference's
        // own precision path -- NOT just "the checkpoint's storage dtype".
        // (Feeding float32 pixels through float32-upcast weights instead
        // was tried during development and gave a *larger* rel err, since
        // it stops being an apples-to-apples comparison against a
        // bf16-computed reference.)
        let out = sam(pixHWC.asType(.bfloat16)).asType(.float32)
        #expect(out.shape == ref.shape)

        let relErr = ((out - ref).abs().mean() / (ref.abs().mean() + 1e-6)).item(Float.self)
        #expect(relErr < 2e-2, "rel err \(relErr)")
    }

    /// The 768px "local tile" path shares the same SAM weights as the
    /// 1024px global view but has no golden intermediate fixture (only the
    /// 1024px global view was dumped by `gen_fixtures.py`; the 768px path
    /// is only ever driven end-to-end through Qwen2Encoder/projector in the
    /// per-image fixtures' `pixels_patches` array). This exercises the path
    /// structurally with the loaded weights -- window partition pads 48 ->
    /// 56 (48 % 14 != 0) the same way the 1024px path pads 64 -> 70, and
    /// the global-attention blocks' absolute/relative position embeddings
    /// get bicubically/linearly resampled down from their 1024px-pretrain
    /// sizes. Full numeric parity for this path is covered end-to-end by
    /// Task 8's higher-level fixtures.
    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func tilePathProducesExpectedShape() throws {
        let config = try Self.realConfig()
        let sam = SAMEncoder(config.sam)
        try TestWeights.load(into: sam, prefix: "sam_model.")

        let pix = MLXArray.zeros([1, 768, 768, 3]).asType(.bfloat16)
        let out = sam(pix)
        #expect(out.shape == [1, 12, 12, 896])
    }
}
