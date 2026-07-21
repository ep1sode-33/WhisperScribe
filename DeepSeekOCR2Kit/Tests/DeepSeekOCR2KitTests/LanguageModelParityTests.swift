import Testing
import Foundation
import MLX
@testable import DeepSeekOCR2Kit

/// NOTE on the brief vs. reality (see task-7-report.md): the brief's
/// skeleton builds `MoELanguageModel(DeepSeekOCR2Configuration.default.text)`
/// and loads via `TestWeights.load`, but `.default` is the Python dataclass
/// default (30 layers / 32 heads / 102,400 vocab), not the real downloaded
/// checkpoint (12 layers / 10 heads / 129,280 vocab) -- so, matching
/// `SAMParityTests`/`VisionEncoderTests`, the config here comes from
/// `DeepSeekOCR2Configuration(mergingJSON:)` over the real `config.json`, and
/// loading uses `TestWeights.loadQuantized` (the `language_model.*` subtree
/// ships 8-bit quantized -- see `FixtureSupport.swift`).
@Suite struct LanguageModelParityTests {
    static func realConfig() throws -> DeepSeekOCR2Configuration {
        let dir = try #require(FixtureSupport.modelDir)
        return try DeepSeekOCR2Configuration(
            mergingJSON: Data(contentsOf: dir.appending(path: "config.json")))
    }

    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func prefillLogitsParity() throws {
        let embeds = try FixtureSupport.load("doc_page", "input_embeds")["x"]!
        let refLogits = try FixtureSupport.load("doc_page", "prefill_logits")["x"]!

        let config = try Self.realConfig()
        let lm = MoELanguageModel(config.text)
        try TestWeights.loadQuantized(into: lm, prefix: "language_model.")

        let logits = lm(inputEmbeds: embeds.asType(.bfloat16), cache: nil).asType(.float32)
        let last = logits[0..., -1, 0...]

        // Quantized-model logits drift from the (presumably higher-precision)
        // reference, so the pass criterion is argmax match plus a top-5
        // index-set overlap of >= 4 -- not bitwise/rel-err closeness.
        let predicted = MLX.argMax(last, axis: -1).item(Int.self)
        let expected = MLX.argMax(refLogits, axis: -1).item(Int.self)
        #expect(predicted == expected, "argmax \(predicted) != reference argmax \(expected)")

        let top5Predicted = Set(MLX.argSort(last, axis: -1)[0..., (-5)...].asArray(Int32.self))
        let top5Expected = Set(MLX.argSort(refLogits, axis: -1)[0..., (-5)...].asArray(Int32.self))
        let overlap = top5Predicted.intersection(top5Expected).count
        #expect(overlap >= 4, "top-5 overlap \(overlap) < 4 (\(top5Predicted) vs \(top5Expected))")
    }

    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func greedyContinuationParity() throws {
        let embeds = try FixtureSupport.load("doc_page", "input_embeds")["x"]!
        let refIDs = try FixtureSupport.load("doc_page", "greedy_tokens")["ids"]!

        let config = try Self.realConfig()
        let lm = MoELanguageModel(config.text)
        try TestWeights.loadQuantized(into: lm, prefix: "language_model.")

        let cache = lm.newCache()
        var logits = lm(inputEmbeds: embeds.asType(.bfloat16), cache: cache)
        var ids: [Int] = []
        for _ in 0..<min(100, refIDs.shape[0]) {
            let next = MLX.argMax(logits[0..., -1, 0...], axis: -1)
            let nextID = next.item(Int.self)
            ids.append(nextID)
            if nextID == config.eosTokenID { break }
            logits = lm(inputEmbeds: lm.embed(next.reshaped(1, 1)), cache: cache)
        }

        let ref = (0..<ids.count).map { refIDs[$0].item(Int32.self) }
        let gotAsRef = ids.map { Int32($0) }
        let divergedAt = zip(gotAsRef, ref).enumerated().first { $1.0 != $1.1 }?.offset
        #expect(gotAsRef == ref, "diverged at step \(divergedAt.map(String.init) ?? "n/a")")
    }
}
