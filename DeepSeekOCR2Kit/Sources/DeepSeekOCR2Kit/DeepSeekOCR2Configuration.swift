import Foundation

/// Configuration for the DeepSeek-OCR-2 model family.
///
/// Most hyperparameters below are NOT present in the real model's `config.json`
/// (in particular everything about the SAM ViT and the Qwen2-as-encoder vision
/// tower) and are hardcoded here, transcribed from the Python reference
/// (`mlx_vlm/models/deepseekocr_2/config.py`, dataclasses `SAMViTConfig` /
/// `Qwen2EncoderConfig` / `TextConfig` / `ModelConfig`) or, where a value has no
/// dataclass default at all (e.g. SAM output channel count, tile image size,
/// query-token counts), from the corresponding model implementation files
/// (`vision.py`, `processing_deepseekocr.py`).
///
/// `init(mergingJSON:)` overrides only the fields that are actually present in
/// a real `config.json`; everything else keeps its hardcoded default.
public struct DeepSeekOCR2Configuration: Codable, Sendable {
    public struct SAMConfig: Codable, Sendable {
        public var layers = 12, width = 768, windowSize = 14, heads = 12
        public var globalAttnIndexes = [2, 5, 8, 11]
        public var outputChannels = 896          // v2: net_3 出 896(v1 为 1024)
        public var patchSize = 16, imageSizeGlobal = 1024, imageSizeTile = 768
    }
    public struct Qwen2EncoderConfig: Codable, Sendable {
        public var layers = 24, dim = 896, heads = 14, kvHeads = 2
        public var intermediate = 4864
        public var ropeTheta: Double = 1_000_000
        public var rmsNormEps: Double = 1e-6
        public var queryTokens1024 = 256, queryTokens768 = 144
    }
    public struct TextConfig: Codable, Sendable {
        // hiddenSize/intermediate/moeIntermediate/numExperts/topK/sharedExperts/
        // qkNopeHeadDim/nGroup/topkGroup/rmsNormEps/ropeTheta match the Python
        // dataclass default AND the real model's config.json (both agree).
        // layers/heads/firstKDenseReplace/vocabSize below are the Python
        // dataclass *defaults* (30/32/0/102_400) — the real downloaded model's
        // config.json overrides all four to 12/10/1/129_280 via `mergingJSON:`.
        public var hiddenSize = 1280, layers = 30, heads = 32
        public var intermediate = 6848, moeIntermediate = 896
        public var numExperts = 64, topK = 6, sharedExperts = 2
        public var firstKDenseReplace = 0, vocabSize = 102_400
        public var qkNopeHeadDim = 0            // ==0 → 非 MLA(LlamaAttention 路径)
        public var nGroup = 1, topkGroup = 1    // 加载时断言,防未实现的 group routing
        public var rmsNormEps: Double = 1e-6
        public var ropeTheta: Double = 10_000
    }
    public var modelType = "deepseekocr_2"
    public var sam = SAMConfig(), qwen2Encoder = Qwen2EncoderConfig(), text = TextConfig()
    // projectorInput: Python dataclass default is 2048; real config.json merges to 896.
    public var projectorInput = 2048, projectorOutput = 1280
    public var bosTokenID = 0, eosTokenID = 1, imageTokenID = 128_815

    public static let `default` = DeepSeekOCR2Configuration()
    public init() {}

    /// Raw shape of the real `config.json` (only the keys we need to merge).
    private struct RawJSON: Decodable {
        struct LanguageConfig: Decodable {
            var hidden_size: Int?
            var num_hidden_layers: Int?
            var num_attention_heads: Int?
            var intermediate_size: Int?
            var moe_intermediate_size: Int?
            var n_routed_experts: Int?
            var num_experts_per_tok: Int?
            var n_shared_experts: Int?
            var first_k_dense_replace: Int?
            var vocab_size: Int?
            var qk_nope_head_dim: Int?
            var n_group: Int?
            var topk_group: Int?
        }
        struct VisionConfig: Decodable {
            struct Width: Decodable {
                struct SAMWidth: Decodable {
                    var layers: Int?
                    var width: Int?
                    var global_attn_indexes: [Int]?
                    var heads: Int?
                }
                struct Qwen2Width: Decodable {
                    var dim: Int?
                }
                var samVitB: SAMWidth?
                var qwen2: Qwen2Width?
                enum CodingKeys: String, CodingKey {
                    case samVitB = "sam_vit_b"
                    case qwen2 = "qwen2-0-5b"
                }
            }
            var width: Width?
        }
        struct ProjectorConfig: Decodable {
            var input_dim: Int?
            var n_embed: Int?
        }
        var model_type: String?
        var bos_token_id: Int?
        var eos_token_id: Int?
        var language_config: LanguageConfig?
        var vision_config: VisionConfig?
        var projector_config: ProjectorConfig?
    }

    /// Merges the fields present in a real model `config.json` on top of the
    /// hardcoded defaults. Fields absent from the JSON keep their default.
    public init(mergingJSON data: Data) throws {
        self.init()
        let raw = try JSONDecoder().decode(RawJSON.self, from: data)

        if let v = raw.model_type { modelType = v }
        if let v = raw.bos_token_id { bosTokenID = v }
        if let v = raw.eos_token_id { eosTokenID = v }

        if let lc = raw.language_config {
            if let v = lc.hidden_size { text.hiddenSize = v }
            if let v = lc.num_hidden_layers { text.layers = v }
            if let v = lc.num_attention_heads { text.heads = v }
            if let v = lc.intermediate_size { text.intermediate = v }
            if let v = lc.moe_intermediate_size { text.moeIntermediate = v }
            if let v = lc.n_routed_experts { text.numExperts = v }
            if let v = lc.num_experts_per_tok { text.topK = v }
            if let v = lc.n_shared_experts { text.sharedExperts = v }
            if let v = lc.first_k_dense_replace { text.firstKDenseReplace = v }
            if let v = lc.vocab_size { text.vocabSize = v }
            if let v = lc.qk_nope_head_dim { text.qkNopeHeadDim = v }
            if let v = lc.n_group { text.nGroup = v }
            if let v = lc.topk_group { text.topkGroup = v }
        }

        if let width = raw.vision_config?.width {
            if let samWidth = width.samVitB {
                if let v = samWidth.layers { sam.layers = v }
                if let v = samWidth.width { sam.width = v }
                if let v = samWidth.global_attn_indexes { sam.globalAttnIndexes = v }
                if let v = samWidth.heads { sam.heads = v }
            }
            if let v = width.qwen2?.dim { qwen2Encoder.dim = v }
        }

        if let pc = raw.projector_config {
            if let v = pc.input_dim { projectorInput = v }
            if let v = pc.n_embed { projectorOutput = v }
        }
    }
}
