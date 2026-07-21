// Adapted from mzbac/deepseek-ocr.swift (MIT)
//
// DeepSeek-V2-style MoE decoder (12 layers, layer 0 dense / layers 1-11
// routed-MoE + 2 shared experts, plain MHA -- `qk_nope_head_dim == 0` means
// this checkpoint never takes the MLA (`DeepseekV2Attention`) branch, only
// the `LlamaAttention` one).
//
// v1's hand-rolled `RoPE`/`KVCache`/softmax-attention are replaced with
// MLXLMCommon's `MLXNN.RoPE` + `KVCacheSimple` + `attentionWithCacheUpdate` +
// `createAttentionMask`, and the per-expert MLP loop is replaced with
// MLXLMCommon's `SwitchGLU` (see `WeightSanitizer.stackExperts` for the
// matching `mlp.switch_mlp.*` checkpoint key stacking).
//
// Python reference (authoritative over v1 Swift where the two disagree):
// mlx_vlm/models/deepseekocr/language.py -- shared verbatim by
// deepseekocr_2.py (`from ..deepseekocr.language import LanguageModel`), so
// this file is a v1/v2-shared port, not v2-specific.
//
// Real checkpoint facts this file relies on (verified against
// `model.safetensors.index.json` / `config.json`, see task-7-report.md):
//   - `language_config.topk_method == "greedy"` and `scoring_func` is absent
//     (Python dataclass default `"softmax"`) -- so `MoEGate` only implements
//     that one branch of Python's `MoEGate.__call__` (no `noaux_tc` grouped
//     routing, no sigmoid scoring).
//   - `num_key_value_heads == num_attention_heads == 10` -- plain MHA, no
//     GQA head-repeat needed (the fast attention kernel used by
//     `attentionWithCacheUpdate` supports GQA/MQA natively anyway).
//   - `language_model.model.embed_tokens.*`, `language_model.lm_head.*`,
//     every `self_attn.*_proj`/`mlp.*_proj` Linear, and `mlp.switch_mlp.*` /
//     `mlp.shared_experts.*` all ship as 8-bit affine quantized (`.weight`
//     packed + `.scales`/`.biases` companions, group_size 64); only
//     `mlp.gate.weight` (the router) and the two RMSNorm weights per layer
//     are plain bf16. `TestWeights.loadQuantized` (FixtureSupport.swift)
//     handles this at load time.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Attention (LlamaAttention path; qk_nope_head_dim == 0 disables MLA)

final class MoEAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: DeepSeekOCR2Configuration.TextConfig) {
        numHeads = config.heads
        numKVHeads = config.kvHeads
        headDim = config.hiddenSize / config.heads
        scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)

        // rope_scaling is absent from the real config.json (no YaRN), and
        // rope_traditional's Python default is false -- plain non-scaled RoPE.
        rope = RoPE(dimensions: headDim, traditional: false, base: Float(config.ropeTheta))

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output =
            attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Dense SwiGLU MLP (layer 0, and each MoE layer's 2 shared experts)

final class MoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediate: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediate, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediate, bias: false)
        _downProj.wrappedValue = Linear(intermediate, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE gate (language.py: MoEGate, `topk_method == "greedy"` branch only)

/// The real checkpoint's `topk_method` is `"greedy"` and `scoring_func`
/// defaults to `"softmax"` (absent from `config.json`, so the Python
/// dataclass default applies) -- this only implements that combination.
/// `n_group`/`topk_group` grouped routing (Python's `"noaux_tc"` branch) is
/// asserted away in `MoELanguageModel.init`.
final class MoEGate: Module {
    let topK: Int
    /// Python default `routed_scaling_factor = 1.0`; not present in the real
    /// `config.json`, so the default applies -- hardcoded here rather than
    /// added as a `TextConfig` field with nothing to ever merge into it.
    let routedScalingFactor: Float = 1.0

    @ModuleInfo(key: "weight") var weight: MLXArray

    init(_ config: DeepSeekOCR2Configuration.TextConfig) {
        topK = config.topK
        _weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize])
        super.init()
    }

    /// Returns `(indices, scores)`, both `[..., topK]`.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = matmul(x, weight.T)
        let scores = MLX.softmax(gates, axis: -1, precise: true)

        // Python: `mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]`. The
        // weighted sum below is invariant to the order of the selected
        // top-k, so an ascending argSort's tail-k selects the identical
        // index SET (and thus the identical result) without depending on
        // argPartition's negative-kth convention.
        let nExperts = scores.dim(-1)
        let sortedIdx = MLX.argSort(scores, axis: -1)
        let indices = sortedIdx[.ellipsis, (nExperts - topK)...]
        let selected = MLX.takeAlong(scores, indices, axis: -1) * routedScalingFactor

        return (indices, selected)
    }
}

// MARK: - MoE block (language.py: DeepseekV2MoE)

final class MoEBlock: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: MoEMLP?

    init(_ config: DeepSeekOCR2Configuration.TextConfig) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediate,
            numExperts: config.numExperts, activation: silu)
        _gate.wrappedValue = MoEGate(config)
        if config.sharedExperts > 0 {
            _sharedExperts.wrappedValue = MoEMLP(
                hiddenSize: config.hiddenSize,
                intermediate: config.moeIntermediate * config.sharedExperts)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        let y = weightedExpertSum(switchMLP(x, indices), scores)
        if let sharedExperts {
            return y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Decoder layer (dense for layer < firstKDenseReplace, else MoE)

final class MoEDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MoEAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    // Untyped so the property can hold either a dense `MoEMLP` or a routed
    // `MoEBlock`; the checkpoint key is `mlp` either way. No `@ModuleInfo`
    // annotation needed -- reflection keys child modules by property name,
    // which already matches ("mlp").
    let mlp: Module

    init(_ config: DeepSeekOCR2Configuration.TextConfig, layerIndex: Int) {
        _selfAttn.wrappedValue = MoEAttention(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))

        // language.py: `layer_idx >= first_k_dense_replace and layer_idx %
        // moe_layer_freq == 0`; moe_layer_freq's Python default is 1, so the
        // modulo term is always true and drops out.
        mlp =
            layerIndex >= config.firstKDenseReplace
            ? MoEBlock(config)
            : MoEMLP(hiddenSize: config.hiddenSize, intermediate: config.intermediate)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let normed = postAttentionLayerNorm(h)
        let mlpOut: MLXArray
        if let dense = mlp as? MoEMLP {
            mlpOut = dense(normed)
        } else if let moe = mlp as? MoEBlock {
            mlpOut = moe(normed)
        } else {
            fatalError("MoEDecoderLayer.mlp must be MoEMLP or MoEBlock")
        }
        return h + mlpOut
    }
}

// MARK: - Model inner (language.py: DeepseekV2Model)

final class MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [MoEDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: DeepSeekOCR2Configuration.TextConfig) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0..<config.layers).map { MoEDecoderLayer(config, layerIndex: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        super.init()
    }

    func callAsFunction(inputEmbeds: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = inputEmbeds
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Top-level language model (language.py: LanguageModel)

/// DeepSeek-3B-MoE decoder: 12 layers (layer 0 dense, layers 1-11 routed-MoE
/// with 6-of-64 experts + 2 always-on shared experts), plain 10-head MHA,
/// vocab 129,280. See the file header for the checkpoint facts this
/// implementation is pinned to.
public final class MoELanguageModel: Module {
    @ModuleInfo(key: "model") var model: MoEModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: DeepSeekOCR2Configuration.TextConfig

    public init(_ config: DeepSeekOCR2Configuration.TextConfig) {
        // This implementation only covers the MoE-routing/attention path the
        // real checkpoint actually uses: no grouped routing (Python's
        // "noaux_tc" topk_method) and no MLA attention. Fail loudly rather
        // than silently mis-route/mis-attend if a future config disagrees.
        guard config.nGroup == 1, config.topkGroup == 1, config.qkNopeHeadDim == 0 else {
            fatalError(
                "MoELanguageModel requires nGroup == 1, topkGroup == 1, qkNopeHeadDim == 0 "
                    + "(got nGroup=\(config.nGroup), topkGroup=\(config.topkGroup), "
                    + "qkNopeHeadDim=\(config.qkNopeHeadDim)) -- grouped MoE routing and MLA "
                    + "attention are not implemented.")
        }

        self.config = config
        _model.wrappedValue = MoEModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)

        super.init()
    }

    public func callAsFunction(inputEmbeds: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputEmbeds: inputEmbeds, cache: cache))
    }

    /// Exposes the token embedding lookup so callers can feed greedy-decoded
    /// token ids back in as `inputEmbeds` for the next step (see
    /// `LanguageModelParityTests.greedyContinuationParity`).
    public func embed(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func newCache() -> [KVCache] {
        (0..<config.layers).map { _ in KVCacheSimple() }
    }
}
