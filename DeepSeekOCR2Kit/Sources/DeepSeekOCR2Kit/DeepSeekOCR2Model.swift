// Top-level DeepSeek-OCR-2 model assembly: the projector, the per-view vision
// pipeline (SAM ViT-B -> Qwen2 causal-visual-flow encoder -> linear
// projector), the scatter of visual features into the language model's token
// embeddings, and the quantized full-model checkpoint loader.
//
// Python reference (authoritative where it disagrees with the brief):
// mlx_vlm/models/deepseekocr_2/deepseekocr_2.py --
//   * `MlpProjector` (lines 14-30): the real structure is a SINGLE
//     `nn.Linear(input_dim, n_embed)` stored under `self.layers` (the
//     `projector_type == "linear"` branch, the only one this checkpoint
//     uses); NOT a two-layer MLP with an activation. Ported verbatim below.
//   * `Model.get_input_embeddings` (lines 65-178): per-view order, the
//     `[local_patches..., global, view_separator]` feature sequence, and the
//     seq-mask scatter -- reproduced in `inputEmbeddings(...)` below.
//
// Module tree / checkpoint-key facts (established Tasks 4-7, verified against
// this checkpoint's `model.safetensors.index.json`): the sanitized namespace
// has five top-level subtrees -- `sam_model.*`, `vision_model.qwen2_encoder.*`,
// `projector.layers.*`, `language_model.*`, and a bare `view_separator` -- so
// this Module's property keys are chosen to match, letting the whole sanitized
// weight dict load in ONE `update(parameters:verify:)`.
//
// Quantization: only `language_model.*` (119 leaves) and `projector.layers`
// carry `.scales`/`.biases` companions (8-bit affine, group_size 64 per the
// real `config.json` `quantization` section); SAM and the Qwen2 encoder are
// bf16. `QuantizedWeightLoader.load` applies the same scales-presence predicate
// used by `TestWeights.loadQuantized` (Task 7), factored here so the load site
// and the tests share one implementation.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Projector (deepseekocr_2.py: MlpProjector)

/// `projector_type == "linear"` -> a single `Linear(input_dim, n_embed)` under
/// the `layers` key (matching the real checkpoint's `projector.layers.{weight,
/// bias,scales,biases}`). The checkpoint ships this Linear 8-bit quantized WITH
/// a plain bf16 `bias`; `quantize(...)` at load time swaps it for a
/// `QuantizedLinear` that keeps the bias, so the bias defaults to `true` here.
final class MlpProjector: Module {
    @ModuleInfo(key: "layers") var layers: Linear

    init(inputDim: Int, nEmbed: Int) {
        self._layers.wrappedValue = Linear(inputDim, nEmbed)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers(x)
    }
}

// MARK: - Vision wrapper (holds the Qwen2 encoder under `qwen2_encoder`)

/// Thin container matching the checkpoint's `vision_model.qwen2_encoder.*` key
/// nesting. The Python `VisionModel` also owns the SAM encoder, but this port
/// keeps `sam_model` a top-level sibling (see `WeightSanitizer` /
/// `deepseekocr_2.py:Model.__init__`'s `self.sam_model = SAMEncoder(...)`), so
/// this wrapper carries only the Qwen2 tower.
final class DeepSeekVisionModel: Module {
    @ModuleInfo(key: "qwen2_encoder") var qwen2Encoder: Qwen2VisionEncoder

    init(_ config: DeepSeekOCR2Configuration.Qwen2EncoderConfig) {
        self._qwen2Encoder.wrappedValue = Qwen2VisionEncoder(config)
        super.init()
    }
}

// MARK: - Shared quantized-weight loader

/// Swaps every leaf whose sanitized `weights` dict carries a `<path>.scales`
/// companion for its quantized counterpart, then loads the full (now
/// shape-matching) `weights` set into `module` in one verified update. Shared
/// by `DeepSeekOCR2Model.load(from:)` (full model, full paths) and
/// `TestWeights.loadQuantized` (a single subtree, prefix-stripped paths) so the
/// quantize+update logic lives in exactly one place.
enum QuantizedWeightLoader {
    static func load(
        into module: Module, weights: [String: MLXArray],
        groupSize: Int, bits: Int, mode: QuantizationMode,
        verify: Module.VerifyUpdate
    ) throws {
        // `quantize`'s filter receives each leaf's flattened path from
        // `module`; a matching `<path>.scales` in `weights` marks it quantized.
        quantize(model: module, groupSize: groupSize, bits: bits, mode: mode) { path, _ in
            weights["\(path).scales"] != nil
        }
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: verify)
    }
}

// MARK: - Top-level model (deepseekocr_2.py: Model)

/// DeepSeek-OCR-2 end to end: image pixels + a token sequence (with N image
/// placeholder tokens) in, language-model logits out.
///
/// ## `inputEmbeddings` pixel contract (NHWC-in)
/// `pixelsGlobal` / `pixelsPatches` are **channel-last** (`(V,1024,1024,3)` /
/// `(P,768,768,3)`), MLX's native conv layout, fed straight to `SAMEncoder`.
/// This deliberately differs from the on-disk fixtures and the processor
/// output, which are channel-first (NCHW) bf16 -- the Python reference
/// transposes NCHW->NHWC internally right before SAM
/// (`global_image.transpose(0, 2, 3, 1)`). Here that transpose is the CALLER's
/// responsibility: the end-to-end test does it exactly as `gen_fixtures.py`
/// did, and Task 9's processor must emit NHWC to match this contract.
public final class DeepSeekOCR2Model: Module {
    @ModuleInfo(key: "sam_model") var samModel: SAMEncoder
    @ModuleInfo(key: "vision_model") var visionModel: DeepSeekVisionModel
    @ModuleInfo(key: "projector") var projector: MlpProjector
    @ModuleInfo(key: "language_model") var languageModel: MoELanguageModel
    // Bare top-level parameter (Python assigns `self.view_separator =
    // mx.zeros((n_embed,))`); keyed to the checkpoint's `view_separator`.
    @ParameterInfo(key: "view_separator") var viewSeparator: MLXArray

    let config: DeepSeekOCR2Configuration

    public init(_ config: DeepSeekOCR2Configuration) {
        self.config = config
        self._samModel.wrappedValue = SAMEncoder(config.sam)
        self._visionModel.wrappedValue = DeepSeekVisionModel(config.qwen2Encoder)
        self._projector.wrappedValue = MlpProjector(
            inputDim: config.projectorInput, nEmbed: config.projectorOutput)
        self._languageModel.wrappedValue = MoELanguageModel(config.text)
        self._viewSeparator.wrappedValue = MLXArray.zeros([config.projectorOutput])
        super.init()
    }

    /// Runs one image view (`(1, H, W, 3)` NHWC) through the full vision
    /// pipeline: SAM -> Qwen2 encoder -> projector, returning `(numTokens,
    /// projectorOutput)` (144 for a 768px tile, 256 for the 1024px global
    /// view). Mirrors the reference's per-view forward exactly (batch 1).
    private func viewFeatures(_ view: MLXArray) -> MLXArray {
        let sam = samModel(view)                                   // (1, h, w, 896)
        let enc = visionModel.qwen2Encoder(samFeatures: sam)       // (1, tokens, 896)
        let proj = projector(enc)                                  // (1, tokens, 1280)
        return proj[0]                                             // (tokens, 1280)
    }

    /// Builds the language-model input embeddings: text tokens embedded
    /// normally, image-token positions overwritten by the projected visual
    /// features in the reference's `[local_patches..., global, view_separator]`
    /// order.
    ///
    /// - Parameters:
    ///   - tokens: `(1, L)` int32 token ids (BOS + N image tokens + prompt).
    ///   - pixelsGlobal: `(V, 1024, 1024, 3)` NHWC (one global view per image;
    ///     V == 1 in practice).
    ///   - pixelsPatches: `(P, 768, 768, 3)` NHWC local tiles, or `nil` for a
    ///     no-crop image.
    ///   - seqMask: `(1, L)` Bool; `true` exactly at image-token positions.
    public func inputEmbeddings(
        tokens: MLXArray, pixelsGlobal: MLXArray,
        pixelsPatches: MLXArray?, seqMask: MLXArray
    ) -> MLXArray {
        // Text embeddings (exact -- same quantized embed_tokens the reference
        // uses); image positions are overwritten below.
        let inputEmbeds = languageModel.embed(tokens)              // (1, L, 1280)

        var allFeatures: [MLXArray] = []

        // Local tiles first, one at a time (matches the reference's per-patch
        // loop -- each SAM/encoder call is batch 1, no cross-view mixing).
        if let patches = pixelsPatches {
            for p in 0..<patches.dim(0) {
                allFeatures.append(viewFeatures(patches[p ..< (p + 1)]))  // (144, 1280)
            }
        }

        // Then the single global view. Exactly one is supported: the view
        // separator below is appended once, so a V>1 batch would desync the
        // feature/separator layout relative to the image-token positions.
        precondition(
            pixelsGlobal.dim(0) == 1,
            "expected exactly one global view, got \(pixelsGlobal.dim(0))"
        )
        for v in 0..<pixelsGlobal.dim(0) {
            allFeatures.append(viewFeatures(pixelsGlobal[v ..< (v + 1)]))  // (256, 1280)
        }

        // Then the single view separator (config.json's global_view_pos:"head"
        // is misleading -- the code appends it at the TAIL; trust the code).
        allFeatures.append(viewSeparator[.newAxis])                 // (1, 1280)

        let visionFeatures = MLX.concatenated(allFeatures, axis: 0)  // (P*144+256+1, 1280)

        // Scatter into the image-token positions. The positions need NOT be a
        // contiguous block, so they are gathered explicitly from the mask and
        // assigned by advanced indexing (MLX Swift has no boolean
        // scatter-assign). Only this index-gather+assign *shape* mirrors
        // mlx-swift-lm's `QwenVL.mergeInputIdsWithImageFeatures`; the
        // token/feature count contract is the Python reference's, which
        // truncates-or-pads the features to the image-token count
        // (deepseekocr_2.py:168-176). We instead require an exact match and
        // fail loudly -- a mismatch here means an upstream processor bug, not
        // something to silently paper over.
        let maskRow = seqMask[0].asArray(Bool.self)
        var imageIndices: [Int] = []
        imageIndices.reserveCapacity(maskRow.count)
        for (i, on) in maskRow.enumerated() where on { imageIndices.append(i) }

        guard !imageIndices.isEmpty else { return inputEmbeds }
        precondition(
            imageIndices.count == visionFeatures.dim(0),
            "image-token count (\(imageIndices.count)) must equal "
                + "vision-feature count (\(visionFeatures.dim(0)))"
        )
        let result = inputEmbeds
        result[0..., MLXArray(imageIndices), 0...] =
            visionFeatures[.newAxis, 0..., 0...]
        return result
    }

    // MARK: LM delegation (so the greedy-decode helper treats this like the LM)

    public func callAsFunction(inputEmbeds: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputEmbeds: inputEmbeds, cache: cache)
    }

    public func embed(_ tokens: MLXArray) -> MLXArray {
        languageModel.embed(tokens)
    }

    public func newCache() -> [KVCache] {
        languageModel.newCache()
    }

    // MARK: - Loading

    /// Loads config + all safetensors shards from `dir`, sanitizes the keys,
    /// applies the config's affine quantization to the `.scales`-bearing leaves,
    /// and loads everything in one verified update. `progress` is reported by
    /// shard count (`i/n`).
    public static func load(
        from dir: URL, progress: @Sendable (Double) -> Void
    ) async throws -> (model: DeepSeekOCR2Model, config: DeepSeekOCR2Configuration) {
        let config = try DeepSeekOCR2Configuration(
            mergingJSON: Data(contentsOf: dir.appending(path: "config.json")))
        let model = DeepSeekOCR2Model(config)

        // Resolve the shard set from the index (falling back to a lone
        // `model.safetensors` if there is no index).
        let shards = try shardNames(in: dir)

        var raw: [String: MLXArray] = [:]
        for (i, shard) in shards.enumerated() {
            let shardWeights = try MLX.loadArrays(url: dir.appending(path: shard))
            raw.merge(shardWeights) { _, new in new }
            progress(Double(i + 1) / Double(shards.count))
        }

        let sanitized = WeightSanitizer.sanitize(raw, config: config)

        if let q = config.quantization {
            let mode = QuantizationMode(rawValue: q.mode) ?? .affine
            try QuantizedWeightLoader.load(
                into: model, weights: sanitized,
                groupSize: q.groupSize, bits: q.bits, mode: mode, verify: .all)
        } else {
            try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        }

        return (model, config)
    }

    private static func shardNames(in dir: URL) throws -> [String] {
        let indexURL = dir.appending(path: "model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL))
                as! [String: Any]
            let weightMap = index["weight_map"] as! [String: String]
            return Set(weightMap.values).sorted()
        }
        return ["model.safetensors"]
    }
}
