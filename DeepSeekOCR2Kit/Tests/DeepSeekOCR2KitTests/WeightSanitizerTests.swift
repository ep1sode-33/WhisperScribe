import Testing
import Foundation
import MLX
@testable import DeepSeekOCR2Kit

/// NOTE on the brief vs. reality (see task-4-report.md "Brief vs reality" section):
/// the downloaded checkpoint (`mlx-community/DeepSeek-OCR-2-8bit`) is *already*
/// in mlx-vlm's post-sanitize namespace (renamed, expert-stacked, conv-transposed) —
/// unlike the brief's assumed near-original-HF key shapes. In particular the
/// `view_separator` key is already spelled correctly in this checkpoint (no
/// `view_seperator` source key exists), so the brief's `hasSuffix("view_seperator")`
/// spot-check is adjusted to the real spelling below.
@Suite struct WeightSanitizerTests {
    // 无模型则整套 skip
    static let modelDir = FixtureSupport.modelDir

    static func realConfig() throws -> DeepSeekOCR2Configuration {
        let dir = try #require(modelDir)
        return try DeepSeekOCR2Configuration(mergingJSON: Data(contentsOf: dir.appending(path: "config.json")))
    }

    @Test(.enabled(if: modelDir != nil))
    func remapsAllCheckpointKeys() throws {
        let raw = try loadAllShards()
        let out = WeightSanitizer.sanitize(raw, config: try Self.realConfig())
        // On this pre-sanitized checkpoint, sanitize is an identity on key names, so set equality guards against future edits dropping or colliding keys.
        #expect(Set(out.keys) == Set(raw.keys))
        // 1) 不残留 checkpoint 前缀
        #expect(!out.keys.contains { $0.hasPrefix("model.") })
        // 2) 关键目标键存在
        #expect(out["vision_model.qwen2_encoder.query_1024"] != nil)
        #expect(out["vision_model.qwen2_encoder.query_768"] != nil)
        #expect(out["view_separator"] != nil)
        // 3) expert 堆叠:64 专家折叠进 switch_mlp,层内不再有 experts.N
        #expect(!out.keys.contains { $0.contains("experts.0.") })
        #expect(!out.keys.contains { $0.contains(".experts.") })
        // 4) 堆叠后的专家张量在专家维(axis 0)长度为 64,匹配 SwitchGLU/QuantizedSwitchLinear 的
        //    `[numExperts, outputDims, inputDims]` 约定
        let stacked = try #require(out["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"])
        #expect(stacked.shape.first == 64)
    }

    @Test(.enabled(if: modelDir != nil))
    func spotChecksNumerics() throws {
        let raw = try loadAllShards()
        let out = WeightSanitizer.sanitize(raw, config: try Self.realConfig())
        // view_separator 数值与源张量一致(仅改名不动值;该 checkpoint 源键拼写已是 view_separator)
        let src = try #require(raw.first { $0.key.hasSuffix("view_separator") }?.value)
        #expect(allClose(out["view_separator"]!, src).item(Bool.self))

        // SAM patch_embed 卷积核数值不变(该 checkpoint 已是 OHWI,sanitize 不应改动数值)
        let convKey = try #require(raw.keys.first { $0.hasSuffix("sam_model.patch_embed.proj.weight") })
        #expect(allClose(out["sam_model.patch_embed.proj.weight"]!, raw[convKey]!).item(Bool.self))
    }

    @Test
    func stacksSyntheticExpertsInOrder() throws {
        // Exercises the expert-stacking branch directly with tiny synthetic
        // per-expert tensors (the real checkpoint arrives pre-stacked, so this
        // branch is otherwise untested by the fixture-driven tests above).
        var config = DeepSeekOCR2Configuration.default
        config.text.layers = 1
        config.text.numExperts = 3

        var raw: [String: MLXArray] = [:]
        for e in 0..<3 {
            raw["language_model.model.layers.0.mlp.experts.\(e).gate_proj.weight"] = MLXArray.ones([2, 2]) * Float(e)
            raw["language_model.model.layers.0.mlp.experts.\(e).gate_proj.scales"] = MLXArray.ones([2]) * Float(e)
        }
        // an unrelated key should pass through untouched
        raw["language_model.model.norm.weight"] = MLXArray.ones([4])

        let out = WeightSanitizer.sanitize(raw, config: config)

        #expect(!out.keys.contains { $0.contains(".experts.") })
        let stacked = try #require(out["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"])
        #expect(stacked.shape == [3, 2, 2])
        for e in 0..<3 {
            #expect(allClose(stacked[e], MLXArray.ones([2, 2]) * Float(e)).item(Bool.self))
        }
        #expect(out["language_model.model.norm.weight"] != nil)
        // scales stacked the same way
        let stackedScales = try #require(out["language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales"])
        #expect(stackedScales.shape == [3, 2])
    }

    @Test
    func transposesRawHFPrefixedKeysAndSAMConv() throws {
        // Exercises the Stage-1 prefix-rename branches and the Stage-2 SAM
        // conv OIHW->OHWI transpose using synthetic keys shaped like the
        // near-original HuggingFace checkpoint (the real fixture is already
        // renamed/transposed, so these branches are otherwise untested).
        var raw: [String: MLXArray] = [:]
        // Per the Python docstring in deepseekocr_2.py Model.sanitize: "HuggingFace:
        // model.qwen2_model.model.model.layers.X..." -- top-level `model.`, NOT
        // nested under `model.vision_model.` (that prefix is for the *other*
        // vision_model.* keys, e.g. a CLIP-style tower in sibling models).
        raw["model.qwen2_model.model.model.layers.0.input_layernorm.weight"] = MLXArray.ones([4])
        raw["model.qwen2_model.query_1024.weight"] = MLXArray.ones([256, 4])
        raw["model.qwen2_model.query_768"] = MLXArray.ones([144, 4])
        raw["model.layers.0.input_layernorm.weight"] = MLXArray.ones([4])
        raw["model.embed_tokens.weight"] = MLXArray.ones([4, 4])
        raw["model.norm.weight"] = MLXArray.ones([4])
        raw["model.sam_model.patch_embed.proj.weight"] = MLXArray.ones([8, 3, 2, 2])  // OIHW
        raw["model.projector.layers.weight"] = MLXArray.ones([4, 4])
        raw["model.view_seperator"] = MLXArray.ones([4])
        raw["lm_head.weight"] = MLXArray.ones([4, 4])
        raw["model.qwen2_model.model.model.norm.weight"] = MLXArray.ones([4])

        let out = WeightSanitizer.sanitize(raw, config: .default)

        #expect(!out.keys.contains { $0.hasPrefix("model.") })
        #expect(out["vision_model.qwen2_encoder.layers.0.input_layernorm.weight"] != nil)
        #expect(out["vision_model.qwen2_encoder.norm.weight"] != nil)
        #expect(out["vision_model.qwen2_encoder.query_1024"] != nil)
        #expect(out["vision_model.qwen2_encoder.query_768"] != nil)
        #expect(out["language_model.model.layers.0.input_layernorm.weight"] != nil)
        #expect(out["language_model.model.embed_tokens.weight"] != nil)
        #expect(out["language_model.model.norm.weight"] != nil)
        #expect(out["projector.layers.weight"] != nil)
        #expect(out["view_separator"] != nil)
        #expect(out["language_model.lm_head.weight"] != nil)

        // OIHW [8,3,2,2] -> OHWI [8,2,2,3]
        let conv = try #require(out["sam_model.patch_embed.proj.weight"])
        #expect(conv.shape == [8, 2, 2, 3])
    }

    /// Loads every tensor from the checkpoint's `model.safetensors.index.json`
    /// shard set (this checkpoint happens to have exactly one shard, but the
    /// helper handles the general multi-shard case).
    func loadAllShards() throws -> [String: MLXArray] {
        let dir = try #require(FixtureSupport.modelDir)
        let indexData = try Data(contentsOf: dir.appending(path: "model.safetensors.index.json"))
        let index = try JSONSerialization.jsonObject(with: indexData) as! [String: Any]
        let weightMap = index["weight_map"] as! [String: String]
        let shardNames = Set(weightMap.values)

        var merged: [String: MLXArray] = [:]
        for shard in shardNames {
            let shardWeights = try MLX.loadArrays(url: dir.appending(path: shard))
            merged.merge(shardWeights) { _, new in new }
        }
        return merged
    }
}
