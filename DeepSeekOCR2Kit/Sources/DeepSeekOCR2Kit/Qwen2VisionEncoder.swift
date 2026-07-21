// Qwen2-as-encoder "causal visual flow" tower: takes SAM ViT-B features and
// runs them through a Qwen2-style transformer stack alongside a block of
// learnable query tokens, using a mixed bidirectional/causal attention mask
// (image tokens attend to each other and are attended by queries;  queries
// additionally see themselves causally). The final `numQueries` hidden
// states become the fixed-size visual token sequence fed to the language
// model's projector.
//
// Python reference: mlx_vlm/models/deepseekocr_2/vision.py, lines 25-367
// (Qwen2RMSNorm / Qwen2RotaryEmbedding / rotate_half / apply_rotary_pos_emb /
// Qwen2MLP / Qwen2Attention / Qwen2DecoderLayer / Qwen2Decoder2Encoder).
// Layer-structure template (q/k/v bias, GQA head reshape, SwiGLU MLP,
// RMSNorm) borrowed from mlx-swift-lm's `MLXVLM/Models/FastVLM.swift`
// `Language` section, stripped of KV-cache and lm_head (this tower is never
// autoregressive -- every call processes the full image+query sequence in
// one shot) -- every numeric detail (RoPE base/position convention, mask
// values, GQA repeat, output slice) is ported from vision.py instead, per
// task-6-brief.md's "where the brief's skeleton and vision.py disagree,
// vision.py wins" rule.

import Foundation
import MLX
import MLXNN

// MARK: - MLP (vision.py: Qwen2MLP)

final class Qwen2EncoderMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dim: Int, intermediate: Int) {
        self._gate.wrappedValue = Linear(dim, intermediate, bias: false)
        self._up.wrappedValue = Linear(dim, intermediate, bias: false)
        self._down.wrappedValue = Linear(intermediate, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - GQA attention with RoPE (vision.py: Qwen2Attention)

final class Qwen2EncoderAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    // `rotary_emb` in vision.py's Qwen2Attention has no learnable
    // parameters (inv_freq is recomputed on the fly, never stored -- see
    // the Python class's own comment), so it isn't a checkpoint key and
    // doesn't need a @ModuleInfo/@ParameterInfo wrapper here; a plain
    // MLXNN.RoPE instance (traditional: false, matching rotate_half's
    // split-half convention) reproduces its math exactly.
    let rotaryEmbedding: RoPE

    init(config: DeepSeekOCR2Configuration.Qwen2EncoderConfig) {
        self.heads = config.heads
        self.kvHeads = config.kvHeads
        self.headDim = config.dim / config.heads
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(config.dim, heads * headDim, bias: true)
        self._kProj.wrappedValue = Linear(config.dim, kvHeads * headDim, bias: true)
        self._vProj.wrappedValue = Linear(config.dim, kvHeads * headDim, bias: true)
        self._oProj.wrappedValue = Linear(heads * headDim, config.dim, bias: false)

        self.rotaryEmbedding = RoPE(
            dimensions: headDim, traditional: false, base: Float(config.ropeTheta))

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))

        var q = qProj(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)

        // position_ids = arange(seq_len) in vision.py covers the full
        // concatenated [image; query] sequence starting at 0 -- exactly
        // what MLXNN.RoPE applies with offset: 0 over the whole L axis.
        q = rotaryEmbedding(q, offset: 0)
        k = rotaryEmbedding(k, offset: 0)

        // GQA: kProj/vProj already have fewer heads than qProj (kvHeads <
        // heads); MLXFast.scaledDotProductAttention broadcasts key/value
        // heads across each query-head group internally (its docs
        // explicitly say NOT to pre-tile them), which is the same
        // consecutive-block head-grouping convention as vision.py's
        // explicit `mx.repeat(..., num_key_value_groups, axis=1)` before
        // its call to `mx.fast.scaled_dot_product_attention` -- so this
        // reproduces the same math without materializing the repeat.
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(b, l, -1)

        return oProj(out)
    }
}

// MARK: - Transformer layer (vision.py: Qwen2DecoderLayer)

final class Qwen2EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen2EncoderAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen2EncoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: DeepSeekOCR2Configuration.Qwen2EncoderConfig) {
        self._selfAttn.wrappedValue = Qwen2EncoderAttention(config: config)
        self._mlp.wrappedValue = Qwen2EncoderMLP(dim: config.dim, intermediate: config.intermediate)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.dim, eps: Float(config.rmsNormEps))
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.dim, eps: Float(config.rmsNormEps))
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x + selfAttn(inputLayerNorm(x), mask: mask)
        h = h + mlp(postAttentionLayerNorm(h))
        return h
    }
}

// MARK: - Encoder tower (vision.py: Qwen2Decoder2Encoder)

/// Qwen2-based "causal visual flow" encoder: SAM features (B, h, w, 896) in,
/// (B, numQueries, 896) out, numQueries ∈ {144, 256} depending on how many
/// image tokens SAM produced (768px tile -> 144, 1024px global view -> 256).
final class Qwen2VisionEncoder: Module {
    let config: DeepSeekOCR2Configuration.Qwen2EncoderConfig

    // Checkpoint keys `query_1024`/`query_768` are raw top-level MLXArrays
    // under the qwen2_encoder prefix (Python assigns them as plain
    // `mx.zeros(...)` attributes, not submodules), matching sam_model's
    // `pos_embed`/`rel_pos_h` pattern in SAMEncoder.swift.
    @ParameterInfo(key: "query_1024") var query1024: MLXArray
    @ParameterInfo(key: "query_768") var query768: MLXArray
    @ModuleInfo(key: "layers") var layers: [Qwen2EncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: DeepSeekOCR2Configuration.Qwen2EncoderConfig) {
        self.config = config
        self._query1024.wrappedValue = MLXArray.zeros([config.queryTokens1024, config.dim])
        self._query768.wrappedValue = MLXArray.zeros([config.queryTokens768, config.dim])
        self._layers.wrappedValue = (0..<config.layers).map { _ in Qwen2EncoderLayer(config) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: Float(config.rmsNormEps))
        super.init()
    }

    /// Port of vision.py's inline mask construction (lines 296-348),
    /// restructured as four quadrant blocks that concatenate to the same
    /// result: image tokens attend to all image tokens (bidirectional,
    /// top-left zeros), image tokens never attend to query tokens
    /// (top-right blocked), query tokens attend to all image tokens
    /// (bottom-left zeros), and query tokens attend to themselves causally
    /// (bottom-right lower-triangular). Exposed as `internal` (not
    /// `private`) so `VisionEncoderTests` can unit-test it directly, per
    /// the brief's risk #1: a wrong mask silently degrades OCR instead of
    /// crashing, so it gets its own fixture-verified test before the full
    /// encoder is trusted.
    static func attentionMask(imageTokens n: Int, queries q: Int) -> MLXArray {
        let negInf = MLXArray(Float(-1e9))
        let topLeft = MLXArray.zeros([n, n])
        let topRight = MLXArray.full([n, q], values: negInf)
        let bottomLeft = MLXArray.zeros([q, n])
        let bottomRight = MLX.triu(MLXArray.full([q, q], values: negInf), k: 1)
        return MLX.concatenated(
            [
                MLX.concatenated([topLeft, topRight], axis: 1),
                MLX.concatenated([bottomLeft, bottomRight], axis: 1),
            ], axis: 0)
    }

    func callAsFunction(samFeatures: MLXArray) -> MLXArray {
        let b = samFeatures.dim(0)
        let flat = samFeatures.reshaped(b, -1, config.dim)
        let n = flat.dim(1)

        // vision.py selects query_1024 unless num_image_tokens == 144
        // exactly (144 -> query_768, everything else including 256 and any
        // unexpected size -> query_1024, its own "default" branch).
        let query = n == config.queryTokens768 ? query768 : query1024
        let q = query.dim(0)

        var h = MLX.concatenated(
            [flat, MLX.broadcast(query[.newAxis], to: [b, q, config.dim])], axis: 1)
        let mask = Self.attentionMask(imageTokens: n, queries: q)
            .asType(h.dtype)[.newAxis, .newAxis, 0..., 0...]

        for layer in layers {
            h = layer(h, mask: mask)
        }
        h = norm(h)

        return h[0..., (-q)..., 0...]
    }
}
