import Testing
import Foundation
import MLX
@testable import DeepSeekOCR2Kit

/// Milestone M4: injected-pixel end-to-end greedy parity. Loads the real
/// quantized checkpoint, feeds each fixture's exact preprocessed pixels +
/// prompt token ids through `DeepSeekOCR2Model.inputEmbeddings` + the greedy
/// decoder, and requires the first 100 sampled token ids to match the Python
/// reference's `greedy_tokens` bit-for-bit -- the whole port (SAM + Qwen2
/// encoder + projector + scatter + MoE LM) either proves out here or reveals an
/// integration bug.
///
/// Pixel contract (see `DeepSeekOCR2Model.inputEmbeddings`): the model takes
/// NHWC pixels; the fixtures are stored channel-first (NCHW) bf16 exactly as
/// the processor emits them, so this test applies the `.transposed(0, 2, 3, 1)`
/// NCHW->NHWC flip itself -- the same transform `gen_fixtures.py` (and the
/// Python reference, internally) applies right before SAM.
extension DeepSeekOCR2Model: GreedyDecodable {}

// `.serialized`: each case loads the full quantized model; since load now
// eagerly materializes all weights (N2), running the parameterized cases in
// parallel would hold several full models resident at once and exhaust memory.
// Serializing caps this suite to one resident model at a time.
@Suite(.serialized) struct EndToEndParityTests {
    @Test(
        .enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil),
        arguments: ["doc_page", "cjk_dense", "tall_scroll"]
    )
    func injectedPixelGreedyParity(name: String) async throws {
        let meta = try FixtureSupport.meta(name)
        let promptIDs = meta["prompt_token_ids"] as! [Int]
        let refIDs = try FixtureSupport.load(name, "greedy_tokens")["ids"]!
        let pixG = try FixtureSupport.load(name, "pixels_global")["x"]!
        // Patches are only dumped when the image is actually cropped; absent ->
        // no local tiles.
        let pixP: MLXArray? = (try? FixtureSupport.load(name, "pixels_patches"))?["x"]

        let (model, cfg) = try await DeepSeekOCR2Model.load(
            from: FixtureSupport.modelDir!, progress: { _ in })

        let tokens = MLXArray(promptIDs.map(Int32.init)).reshaped(1, -1)
        let seqMask = MLX.equal(tokens, MLXArray(Int32(cfg.imageTokenID)))

        let embeds = model.inputEmbeddings(
            tokens: tokens,
            pixelsGlobal: pixG.transposed(0, 2, 3, 1).asType(.bfloat16),
            pixelsPatches: pixP.map { $0.transposed(0, 2, 3, 1).asType(.bfloat16) },
            seqMask: seqMask)

        let ids = FixtureSupport.greedyDecode(
            model: model, embeds: embeds, steps: min(100, refIDs.shape[0]),
            eos: cfg.eosTokenID)

        let ref = (0..<ids.count).map { Int(refIDs[$0].item(Int32.self)) }
        let divergedAt = zip(ids, ref).enumerated().first { $1.0 != $1.1 }?.offset
        #expect(ids == ref, "\(name) diverged at \(divergedAt.map(String.init) ?? "n/a")")
    }
}
