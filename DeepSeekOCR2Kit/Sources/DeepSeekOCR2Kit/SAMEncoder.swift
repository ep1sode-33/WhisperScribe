// Adapted from mzbac/deepseek-ocr.swift (MIT)
// Source: https://github.com/mzbac/deepseek-ocr.swift
//         Sources/DeepSeekOCR/VisionEncoder.swift
// (the upstream repository's README declares the project MIT-licensed; no
// separate LICENSE file/header was present in the source checked out for
// this port, so this notice stands in for it)
//
// Ported to DeepSeek-OCR-2's SAM ViT-B encoder:
//   - checkpoint key layout matches the real `sam_model.*` namespace
//     (`blocks.N.*`, `neck.{0,1,2,3}.*`, `pos_embed`, `attn.rel_pos_{h,w}`)
//     instead of v1's renamed (`layers`, `neck.conv1`, ...) keys, so that
//     `Module.update(parameters:verify:)` can load the checkpoint directly
//     without a manual key-rewrite pass.
//   - `net_3`'s output channel count is `config.outputChannels` (896 for
//     OCR-2) instead of v1's hardcoded 1024.
//   - added the 768px tile path: same weights, same code paths (window
//     partition pads 48 -> 56 exactly like 1024 pads 64 -> 70 for
//     window_size=14), but the absolute position embedding — always
//     initialized at the 1024px pretrain grid (64x64) — must be bicubically
//     resampled down to 48x48 (`get_abs_pos_sam` in the Python reference).
//     v1 used `MLXNN.Upsample(mode: .cubic)` for this; ported here to
//     `MLXLMCommon.bicubicInterpolate` (a dedicated Metal kernel with
//     `antialias: true`, matching the Python reference's own
//     `bicubic_interpolate(..., antialias=True)` call) since NHWC/NCHW
//     semantics and the antialias flag matter for reproducing the
//     reference numerically -- see `interpolatedPosEmbed(targetSize:)`
//     below.
//
// Python reference: mlx_vlm/models/deepseekocr/sam.py (489 lines, shared
// verbatim by v1 and v2), wired up as `Model.sam_model` in
// mlx_vlm/models/deepseekocr_2/deepseekocr_2.py with `final_out_chans=896`.

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - MLP block (sam.py: MLPBlock)

final class SAMMLP: Module {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    init(embedDim: Int, mlpDim: Int) {
        self._lin1.wrappedValue = Linear(embedDim, mlpDim)
        self._lin2.wrappedValue = Linear(mlpDim, embedDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lin2(gelu(lin1(x)))
    }
}

// MARK: - Attention with decomposed relative position embeddings (sam.py: Attention)

final class SAMAttention: Module {
    let numHeads: Int
    let scale: Float
    let windowSize: Int
    let useRelPos: Bool

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    // Checkpoint keys are `rel_pos_h`/`rel_pos_w` (snake_case, from the
    // PyTorch/mlx-vlm original); @ParameterInfo maps the camelCase Swift
    // property to that exact key so `update(parameters:verify:
    // .noUnusedKeys)` matches them without a manual rename pass. Plain
    // (non-wrapped) `MLXArray?` properties would instead be keyed by their
    // literal Swift property name ("relPosH"), which would never match.
    @ParameterInfo(key: "rel_pos_h") var relPosH: MLXArray?
    @ParameterInfo(key: "rel_pos_w") var relPosW: MLXArray?

    /// - Parameters:
    ///   - globalInputSize: `imageSizeGlobal / patchSize` (64) -- the SAM
    ///     encoder's *construction-time* pretrain grid size. Global-attention
    ///     blocks (`windowSize == 0`) size their rel-pos tables against this
    ///     constant, matching Python's `input_size=(img_size // patch_size,
    ///     img_size // patch_size)` where `img_size` is `SAMEncoder`'s
    ///     constructor argument (always 1024) -- NOT whatever resolution is
    ///     actually fed through `callAsFunction` at inference (768 or 1024
    ///     both reuse this same encoder instance and its fixed-size weights).
    init(
        width: Int, heads: Int, windowSize: Int, globalInputSize: Int, qkvBias: Bool,
        useRelPos: Bool
    ) {
        self.numHeads = heads
        let headDim = width / heads
        self.scale = pow(Float(headDim), -0.5)
        self.windowSize = windowSize
        self.useRelPos = useRelPos

        self._qkv.wrappedValue = Linear(width, width * 3, bias: qkvBias)
        self._proj.wrappedValue = Linear(width, width)

        if useRelPos {
            let inputSize = windowSize > 0 ? windowSize : globalInputSize
            self._relPosH.wrappedValue = MLXArray.zeros([2 * inputSize - 1, headDim])
            self._relPosW.wrappedValue = MLXArray.zeros([2 * inputSize - 1, headDim])
        }

        super.init()
    }

    /// Port of `sam.py get_rel_pos`. Extracts relative-position embeddings
    /// for a given query/key size pair, linearly resampling `relPos` along
    /// its length first if it doesn't already match `2 * max(qSize, kSize)
    /// - 1` -- ported as the manual floor/ceil weighted-average PyTorch/mlx
    /// itself uses (NOT `MLXNN.Upsample`, whose `align_corners`/kernel
    /// semantics aren't guaranteed to match this specific algorithm) so the
    /// 768px tile path's global-attention blocks (the only blocks that ever
    /// hit this branch -- see `SAMEncoder`'s doc comment) reproduce the
    /// reference exactly. Not exercised by the 1024px golden fixture: every
    /// block's checkpoint rel-pos length already matches its q/k size at
    /// that resolution (27 for window blocks, 127 for global-attention
    /// blocks), so this branch is untested by `SAMParityTests` and instead
    /// only shape-checked via the 768px path.
    private func getRelPos(qSize: Int, kSize: Int, relPos: MLXArray) -> MLXArray {
        let maxRelDist = 2 * max(qSize, kSize) - 1

        var relPosResized = relPos
        if relPos.dim(0) != maxRelDist {
            let dtype = relPos.dtype
            let length = relPos.dim(0)
            let channels = relPos.dim(1)

            // (L, C) -> (1, L, C) -> (1, C, L), matching Python's
            // `rel_pos.reshape(1, L, -1).transpose(0, 2, 1)`.
            var resized = relPos.asType(.float32).reshaped(1, length, channels).transposed(
                0, 2, 1)

            let scale = Float(length) / Float(maxRelDist)
            let indices = MLXArray((0..<maxRelDist).map { Float($0) * scale })
            let idxFloor = floor(indices).asType(.int32)
            let idxCeil = minimum(idxFloor + 1, MLXArray(Int32(length - 1)))
            let weight = indices - idxFloor.asType(.float32)

            let floorVals = resized.take(idxFloor, axis: 2)
            let ceilVals = resized.take(idxCeil, axis: 2)
            resized = (floorVals * (1 - weight) + ceilVals * weight).asType(dtype)

            relPosResized = resized.reshaped(channels, maxRelDist).transposed(1, 0)
        }

        let qCoords = MLXArray((0..<qSize).map { Float($0) * max(Float(kSize) / Float(qSize), 1.0) })
        let kCoords = MLXArray((0..<kSize).map { Float($0) * max(Float(qSize) / Float(kSize), 1.0) })

        let qCoordsExpanded = qCoords.reshaped(qSize, 1)
        let kCoordsExpanded = kCoords.reshaped(1, kSize)
        let offset = Float(kSize - 1) * max(Float(qSize) / Float(kSize), 1.0)
        let relativeCoords = qCoordsExpanded - kCoordsExpanded + offset

        let indices = relativeCoords.asType(.int32)
        return relPosResized[indices]
    }

    /// Port of `sam.py add_decomposed_rel_pos`, algebraically restructured
    /// to use `matmul` with a transpose trick instead of `mx.einsum`
    /// (`"bhwc,hkc->bhwk"` / `"bhwc,wkc->bhwk"`) -- MLX's batched matmul
    /// broadcasts the shared `h`/`w` index the same way einsum's repeated
    /// subscript would, so the two are numerically equivalent.
    private func computeRelPosBias(
        query: MLXArray,
        height: Int,
        width: Int
    ) -> (MLXArray, MLXArray)? {
        guard useRelPos, let relPosH = relPosH, let relPosW = relPosW else {
            return nil
        }

        let batchTimesHeads = query.dim(0)
        let dim = query.dim(-1)

        let rQ = query.reshaped(batchTimesHeads, height, width, dim)

        let relH = getRelPos(qSize: height, kSize: height, relPos: relPosH)
        let relW = getRelPos(qSize: width, kSize: width, relPos: relPosW)

        let relHResult = matmul(rQ, relH.swappedAxes(-2, -1))

        let rQTransposed = rQ.transposed(0, 2, 1, 3)
        var relWResult = matmul(rQTransposed, relW.swappedAxes(-2, -1))
        relWResult = relWResult.transposed(0, 2, 1, 3)

        let relHBias = relHResult.reshaped(batchTimesHeads, height * width, height, 1)
        let relWBias = relWResult.reshaped(batchTimesHeads, height * width, 1, width)

        return (relHBias, relWBias)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (batchSize, height, width, _) = (
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            hiddenStates.dim(2),
            hiddenStates.dim(3)
        )

        var qkvOut = qkv(hiddenStates)

        qkvOut = qkvOut
            .reshaped(batchSize, height * width, 3, numHeads, -1)
            .transposed(2, 0, 3, 1, 4)

        let qkvFlat = qkvOut.reshaped(3, batchSize * numHeads, height * width, -1)
        let qFlat = qkvFlat[0]

        var attnBias: MLXArray? = nil
        if useRelPos, let relPos = computeRelPosBias(query: qFlat, height: height, width: width) {
            let (relHBias, relWBias) = relPos
            let combined = relHBias + relWBias
            attnBias = combined.reshaped(batchSize, numHeads, height * width, height * width)
        }

        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]

        let attnOutput =
            MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: scale, mask: attnBias
            )
            .reshaped(batchSize, numHeads, height, width, -1)
            .transposed(0, 2, 3, 1, 4)
            .reshaped(batchSize, height, width, -1)

        return proj(attnOutput)
    }
}

// MARK: - Transformer block with optional window attention (sam.py: Block)

final class SAMBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: SAMAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SAMMLP

    let windowSize: Int

    init(
        width: Int, heads: Int, windowSize: Int, globalInputSize: Int, qkvBias: Bool,
        useRelPos: Bool, mlpDim: Int, eps: Float
    ) {
        self.windowSize = windowSize
        self._norm1.wrappedValue = LayerNorm(dimensions: width, eps: eps)
        self._attn.wrappedValue = SAMAttention(
            width: width, heads: heads, windowSize: windowSize, globalInputSize: globalInputSize,
            qkvBias: qkvBias, useRelPos: useRelPos)
        self._norm2.wrappedValue = LayerNorm(dimensions: width, eps: eps)
        self._mlp.wrappedValue = SAMMLP(embedDim: width, mlpDim: mlpDim)
        super.init()
    }

    /// Port of `sam.py window_partition`.
    private func windowPartition(_ hiddenStates: MLXArray, windowSize: Int) -> (
        MLXArray, (Int, Int), Int
    ) {
        let (batchSize, height, width, channels) = (
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            hiddenStates.dim(2),
            hiddenStates.dim(3)
        )

        let padH = (windowSize - height % windowSize) % windowSize
        let padW = (windowSize - width % windowSize) % windowSize

        var h = hiddenStates
        if padH > 0 || padW > 0 {
            h = padded(h, widths: [.init((0, 0)), .init((0, padH)), .init((0, padW)), .init((0, 0))])
        }

        let padHeight = height + padH
        let padWidth = width + padW

        h = h.reshaped(
            batchSize,
            padHeight / windowSize, windowSize,
            padWidth / windowSize, windowSize,
            channels
        )
        h = h.transposed(0, 1, 3, 2, 4, 5)
        let windows = h.reshaped(-1, windowSize, windowSize, channels)

        return (windows, (padHeight, padWidth), batchSize)
    }

    /// Port of `sam.py window_unpartition`.
    private func windowUnpartition(
        _ windows: MLXArray,
        windowSize: Int,
        paddingShape: (Int, Int),
        originalShape: (Int, Int),
        batchSize: Int
    ) -> MLXArray {
        let (padHeight, padWidth) = paddingShape
        let (height, width) = originalShape
        let channels = windows.dim(-1)

        var h = windows.reshaped(
            batchSize,
            padHeight / windowSize,
            padWidth / windowSize,
            windowSize, windowSize,
            channels
        )
        h = h.transposed(0, 1, 3, 2, 4, 5)
        h = h.reshaped(batchSize, padHeight, padWidth, channels)

        if padHeight > height || padWidth > width {
            h = h[0..., 0..<height, 0..<width, 0...]
        }

        return h
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let residual = hiddenStates
        var h = norm1(hiddenStates)

        let (height, width) = (h.dim(1), h.dim(2))

        var paddingShape: (Int, Int)?
        var originalBatchSize: Int?
        if windowSize > 0 {
            let result = windowPartition(h, windowSize: windowSize)
            h = result.0
            paddingShape = result.1
            originalBatchSize = result.2
        }

        h = attn(h)

        if windowSize > 0, let padding = paddingShape, let batchSize = originalBatchSize {
            h = windowUnpartition(
                h, windowSize: windowSize, paddingShape: padding, originalShape: (height, width),
                batchSize: batchSize)
        }

        h = residual + h
        h = h + mlp(norm2(h))
        return h
    }
}

// MARK: - Patch embedding (sam.py: PatchEmbed)

final class SAMPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(inChannels: Int, embedDim: Int, patchSize: Int) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: IntOrPair(patchSize),
            stride: IntOrPair(patchSize)
        )
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        proj(pixelValues)
    }
}

// MARK: - SAM ViT-B encoder (sam.py: SAMEncoder)

/// SAM ViT-B image encoder shared, with identical weights, by DeepSeek-OCR-2's
/// two input resolutions: the 1024px "global view" tile and the 768px
/// per-crop "local view" tiles (`deepseekocr_2.py:Model.get_input_embeddings`
/// calls `self.sam_model(...)` on both). Input is `(B, H, W, 3)` (NHWC, H=W
/// in {1024, 768}); output is `(B, H/64, W/64, config.outputChannels)`
/// (16x16x896 for 1024, 12x12x896 for 768) after two stride-2 downsampling
/// convs (`net_2`, `net_3`) following the ViT backbone + neck.
///
/// Hyperparameters not present in `SAMConfig` / the real `config.json` (SAM's
/// `vision_config.width.sam_vit_b` section is only partially consulted by
/// the checked-in `DeepSeekOCR2Configuration.SAMConfig` merge -- see
/// `DeepSeekOCR2Configuration`'s own doc comment) are hardcoded below,
/// transcribed from `sam.py
/// SAMEncoder.__init__`'s own defaults, none of which
/// `deepseekocr_2.py:Model.__init__`'s `SAMEncoder(...)` call overrides
/// except `final_out_chans` (-> `config.outputChannels`).
public final class SAMEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: SAMPatchEmbed
    // Checkpoint key is `pos_embed` (Python assigns `self.pos_embed = mx.zeros(...)`
    // directly as a raw array attribute, not a named submodule).
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray?
    @ModuleInfo(key: "blocks") var blocks: [SAMBlock]
    // Checkpoint keys are `neck.0`/`neck.1`/`neck.2`/`neck.3` (the Python
    // reference assigns `self.neck = [Conv2d, LayerNorm, Conv2d, LayerNorm]`
    // as a plain list, not named submodules). `NestedDictionary.unflattened`
    // treats an all-numeric-key subtree as an `.array`, which only matches
    // a genuine `[UnaryLayer]` array property here (a wrapper `Module` with
    // 4 named-but-numerically-keyed properties instead produces a
    // `.dictionary` and fails `update(parameters:verify:)` with
    // `incompatibleItems`) -- so, like `blocks`, `neck` is a plain
    // heterogeneous array applied in sequence, matching Python exactly.
    @ModuleInfo(key: "neck") var neck: [UnaryLayer]
    @ModuleInfo(key: "net_2") var net2: Conv2d
    @ModuleInfo(key: "net_3") var net3: Conv2d

    /// `img_size / patch_size` at construction time (64 for the 1024px
    /// pretrain grid) -- the grid the absolute/relative position embeddings
    /// are permanently sized against, independent of what resolution is fed
    /// through `callAsFunction` at inference.
    private let originalGridSize: Int

    // sam.py SAMEncoder.__init__ defaults, never overridden by
    // deepseekocr_2.py's `SAMEncoder(...)` call site or by config.json.
    private static let inChannels = 3
    private static let neckOutChannels = 256  // out_chans
    private static let net2MidChannels = 512  // hardcoded literal in sam.py __call__
    private static let mlpRatio = 4.0
    private static let layerNormEps: Float = 1e-6
    private static let qkvBias = true
    private static let useAbsPos = true
    private static let useRelPos = true

    public init(_ config: DeepSeekOCR2Configuration.SAMConfig) {
        self.originalGridSize = config.imageSizeGlobal / config.patchSize

        self._patchEmbed.wrappedValue = SAMPatchEmbed(
            inChannels: Self.inChannels, embedDim: config.width, patchSize: config.patchSize)

        if Self.useAbsPos {
            self._posEmbed.wrappedValue = MLXArray.zeros([
                1, originalGridSize, originalGridSize, config.width,
            ])
        }

        let mlpDim = Int(Double(config.width) * Self.mlpRatio)
        var blocks: [SAMBlock] = []
        blocks.reserveCapacity(config.layers)
        for i in 0..<config.layers {
            let windowSize = config.globalAttnIndexes.contains(i) ? 0 : config.windowSize
            blocks.append(
                SAMBlock(
                    width: config.width, heads: config.heads, windowSize: windowSize,
                    globalInputSize: originalGridSize, qkvBias: Self.qkvBias,
                    useRelPos: Self.useRelPos, mlpDim: mlpDim, eps: Self.layerNormEps))
        }
        self._blocks.wrappedValue = blocks

        self._neck.wrappedValue = [
            Conv2d(
                inputChannels: config.width, outputChannels: Self.neckOutChannels,
                kernelSize: IntOrPair(1), bias: false),
            LayerNorm(dimensions: Self.neckOutChannels, eps: Self.layerNormEps),
            Conv2d(
                inputChannels: Self.neckOutChannels, outputChannels: Self.neckOutChannels,
                kernelSize: IntOrPair(3), padding: IntOrPair(1), bias: false),
            LayerNorm(dimensions: Self.neckOutChannels, eps: Self.layerNormEps),
        ]

        self._net2.wrappedValue = Conv2d(
            inputChannels: Self.neckOutChannels,
            outputChannels: Self.net2MidChannels,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1),
            bias: false
        )
        self._net3.wrappedValue = Conv2d(
            inputChannels: Self.net2MidChannels,
            outputChannels: config.outputChannels,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1),
            bias: false
        )

        super.init()
    }

    /// Port of `sam.py get_abs_pos_sam`. Bicubically resamples the absolute
    /// position embedding from `originalGridSize` (64) to `targetSize` --
    /// the grid `patchEmbed` actually produced for whatever resolution was
    /// fed to `callAsFunction`. A no-op for the 1024px global view
    /// (targetSize == 64 == originalGridSize, so the golden-fixture parity
    /// test never exercises the interpolation path below); does interpolate
    /// for the 768px tile view (targetSize == 48).
    private func interpolatedPosEmbed(targetSize: Int) -> MLXArray {
        guard let posEmbed else {
            fatalError("SAMEncoder: pos_embed missing (use_abs_pos requires it)")
        }
        guard posEmbed.dim(1) != targetSize else { return posEmbed }

        let dtype = posEmbed.dtype
        // (1, H, W, C) -> (1, C, H, W) for the kernel's NCHW convention.
        let nchw = posEmbed.transposed(0, 3, 1, 2).asType(.float32)
        let resized = bicubicInterpolate(nchw, size: (targetSize, targetSize), antialias: true)
        // (1, C, H, W) -> (1, H, W, C)
        return resized.transposed(0, 2, 3, 1).asType(dtype)
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hiddenStates = patchEmbed(pixelValues)

        if Self.useAbsPos {
            hiddenStates = hiddenStates + interpolatedPosEmbed(targetSize: hiddenStates.dim(1))
        }

        for block in blocks {
            hiddenStates = block(hiddenStates)
        }

        for layer in neck {
            hiddenStates = layer(hiddenStates)
        }
        hiddenStates = net2(hiddenStates)
        hiddenStates = net3(hiddenStates)

        return hiddenStates
    }
}
