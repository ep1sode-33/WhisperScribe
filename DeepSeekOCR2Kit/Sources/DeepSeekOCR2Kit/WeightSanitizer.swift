import Foundation
import MLX

/// Pure checkpoint-key remapping for the DeepSeek-OCR-2 checkpoint.
///
/// `WeightSanitizer.sanitize` turns whatever key namespace a checkpoint on
/// disk happens to use into the namespace the Swift module tree (Tasks 5-8)
/// expects: top-level `vision_model.qwen2_encoder.*` (incl. `query_1024` /
/// `query_768`), top-level `sam_model.*` (a *sibling* of `vision_model`, not
/// nested under it — see `sam_model = SAMEncoder(...)` in
/// `deepseekocr_2.py:Model.__init__`), `projector.*`, `language_model.*`, and
/// a bare `view_separator`.
///
/// This is a faithful 1:1 port of the three sanitize stages mlx-vlm chains
/// together when loading this model (`mlx_vlm/utils.py` calls, in order):
///   1. `Model.sanitize`         — prefix renames (`deepseekocr_2.py:206-293`)
///   2. `VisionModel.sanitize`   — drops `position_ids`, transposes SAM conv
///                                 kernels OIHW -> OHWI (`vision.py:407-439`)
///   3. `LanguageModel.sanitize` — stacks the 64 per-expert MoE weights into
///                                 `switch_mlp` (`deepseekocr/language.py:499-510`)
///
/// Every stage guards on the *presence* of its expected source key, exactly
/// like the Python original, which makes the whole function idempotent:
/// checkpoints that already ship fully renamed/stacked/transposed (as
/// `mlx-community/DeepSeek-OCR-2-8bit` does — see task-4-report.md) pass
/// through unchanged, while checkpoints closer to the original HuggingFace
/// layout get fully remapped.
public enum WeightSanitizer {
    public static func sanitize(
        _ weights: [String: MLXArray], config: DeepSeekOCR2Configuration
    ) throws -> [String: MLXArray] {
        var renamed: [String: MLXArray] = [:]
        renamed.reserveCapacity(weights.count)
        for (key, value) in weights {
            renamed[renameKey(key)] = value
        }

        let transposed = transposeConvWeights(renamed)
        return try stackExperts(transposed, config: config)
    }

    // MARK: - Stage 1: prefix renames (Model.sanitize)

    /// Ported line-for-line from `deepseekocr_2.py Model.sanitize.transform_key`.
    private static func renameKey(_ key: String) -> String {
        var key = key

        if key.contains("qwen2_model.model.model.layers") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.model.model.layers",
                with: "vision_model.qwen2_encoder.layers")
        }
        if key.contains("qwen2_model.model.model.norm") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.model.model.norm",
                with: "vision_model.qwen2_encoder.norm")
        }

        if key.contains("model.qwen2_model.query_1024.weight") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.query_1024.weight",
                with: "vision_model.qwen2_encoder.query_1024")
        } else if key.contains("model.qwen2_model.query_1024") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.query_1024",
                with: "vision_model.qwen2_encoder.query_1024")
        }
        if key.contains("model.qwen2_model.query_768.weight") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.query_768.weight",
                with: "vision_model.qwen2_encoder.query_768")
        } else if key.contains("model.qwen2_model.query_768") {
            key = key.replacingOccurrences(
                of: "model.qwen2_model.query_768",
                with: "vision_model.qwen2_encoder.query_768")
        }

        if key.contains("model.layers"), !key.contains("language_model"), !key.contains("qwen2") {
            key = key.replacingOccurrences(of: "model.layers", with: "language_model.model.layers")
        }
        if key.contains("model.embed_tokens"), !key.contains("language_model"), !key.contains("qwen2") {
            key = key.replacingOccurrences(
                of: "model.embed_tokens", with: "language_model.model.embed_tokens")
        }
        if key.contains("model.norm"), !key.contains("language_model"), !key.contains("qwen2") {
            key = key.replacingOccurrences(of: "model.norm", with: "language_model.model.norm")
        }

        if key.contains("model.vision_model") {
            key = key.replacingOccurrences(of: "model.vision_model", with: "vision_model")
        }
        // sam_model stays a TOP-LEVEL sibling of vision_model (matches
        // `self.sam_model = SAMEncoder(...)` in Model.__init__, and the real
        // checkpoint's own top-level key prefixes) — it is NOT nested under
        // vision_model.
        if key.contains("model.sam_model") {
            key = key.replacingOccurrences(of: "model.sam_model", with: "sam_model")
        }
        if key.contains("model.projector") {
            key = key.replacingOccurrences(of: "model.projector", with: "projector")
        }
        // HuggingFace has the typo "view_seperator" (e instead of a).
        if key.contains("model.view_seperator") {
            key = key.replacingOccurrences(of: "model.view_seperator", with: "view_separator")
        }
        if key.contains("lm_head.weight"), !key.contains("language_model") {
            key = key.replacingOccurrences(of: "lm_head.weight", with: "language_model.lm_head.weight")
        }

        return key
    }

    // MARK: - Stage 2: SAM conv OIHW -> OHWI (VisionModel.sanitize)

    /// Last-three-dot-segment key suffixes that name a PyTorch `Conv2d`
    /// weight tensor, ported verbatim from `vision.py VisionModel.sanitize`'s
    /// `weight_keys` set.
    private static let convWeightKeySuffixes: Set<String> = [
        "neck.0.weight", "neck.2.weight",
        "neck_hd.0.weight", "neck_hd.2.weight",
        "sam_model.net_2.weight", "sam_model.net_3.weight",
        "downsamples.0.weight", "downsamples.1.weight",
        "patch_embed.proj.weight", "embeddings.patch_embedding.weight",
    ]

    private static func transposeConvWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.contains("position_ids") { continue }  // unused buffer, dropped

            if convWeightKeySuffixes.contains(lastThreeDotSegments(of: key)) {
                out[key] = isAlreadyOHWI(value) ? value : value.transposed(0, 2, 3, 1)
            } else {
                out[key] = value
            }
        }
        return out
    }

    private static func lastThreeDotSegments(of key: String) -> String {
        let parts = key.split(separator: ".")
        guard parts.count >= 3 else { return key }
        return parts.suffix(3).joined(separator: ".")
    }

    /// Port of `vision.py check_array_shape`: a PyTorch Conv2d weight is
    /// `[out, in, kH, kW]`; MLX's `nn.Conv2d` wants `[out, kH, kW, in]`. This
    /// heuristically detects tensors already in the MLX (OHWI) layout so the
    /// transpose is skipped for checkpoints that ship pre-converted.
    private static func isAlreadyOHWI(_ array: MLXArray) -> Bool {
        guard array.ndim == 4 else { return false }
        let shape = array.shape
        let outChannels = shape[0], kH = shape[1], kW = shape[2]
        return outChannels >= kH && outChannels >= kW && kH == kW
    }

    // MARK: - Stage 3: MoE expert stacking (LanguageModel.sanitize)

    /// Port of `deepseekocr/language.py LanguageModel.sanitize`: stacks each
    /// layer's `mlp.experts.{0..<numExperts}.{gate,up,down}_proj.{weight,
    /// scales,biases}` into `mlp.switch_mlp.{...}.{...}` along a new leading
    /// axis (`MLX.stacked(_:axis:0)`), matching MLXLMCommon's
    /// `SwitchLinear`/`QuantizedSwitchLinear` convention of
    /// `[numExperts, outputDims, inputDims]` (weight) and
    /// `[numExperts, outputDims, numGroups]` (scales/biases).
    private static func stackExperts(
        _ weights: [String: MLXArray], config: DeepSeekOCR2Configuration
    ) throws -> [String: MLXArray] {
        var out = weights
        for layer in 0..<config.text.layers {
            let prefix = "language_model.model.layers.\(layer)"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for attribute in ["weight", "scales", "biases"] {
                    let firstExpertKey = "\(prefix).mlp.experts.0.\(projection).\(attribute)"
                    guard out[firstExpertKey] != nil else { continue }

                    var perExpert: [MLXArray] = []
                    perExpert.reserveCapacity(config.text.numExperts)
                    for expert in 0..<config.text.numExperts {
                        let key = "\(prefix).mlp.experts.\(expert).\(projection).\(attribute)"
                        guard let value = out.removeValue(forKey: key) else {
                            // A partial expert set is a corrupt/incompatible
                            // checkpoint, not a programmer error -- surface it as
                            // a typed load error rather than trapping. (A2)
                            throw OCR2LoadError.missingTensor(
                                "missing expert \(expert) for "
                                    + "\(prefix).mlp.\(projection).\(attribute) "
                                    + "(expected \(config.text.numExperts) experts)")
                        }
                        perExpert.append(value)
                    }
                    out["\(prefix).mlp.switch_mlp.\(projection).\(attribute)"] = MLX.stacked(perExpert)
                }
            }
        }
        return out
    }
}
