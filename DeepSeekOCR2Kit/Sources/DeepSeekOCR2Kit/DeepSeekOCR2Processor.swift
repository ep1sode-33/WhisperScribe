// Adapted from mzbac/deepseek-ocr.swift (MIT) -- tiling scaffolding
// (dynamicPreprocess / findClosestAspectRatio / pad geometry); parameters and
// resampling rewritten to match mlx-vlm's DeepSeek-OCR-2 reference
// (processing_deepseekocr.py) bit-for-bit.
// Source: https://github.com/mzbac/deepseek-ocr.swift
//         Sources/DeepSeekOCR/ImageProcessor.swift

import CoreGraphics
import Foundation
import ImageIO
import MLX
import Tokenizers

/// Preprocessed inputs for `DeepSeekOCR2Model.inputEmbeddings`. Pixel tensors
/// are **NHWC bf16** (the model's documented contract, Task 8): the reference
/// processor emits NCHW, but the model transposes NCHW->NHWC right before SAM,
/// so we produce NHWC directly and skip the round-trip.
///
/// `@unchecked Sendable`: `MLXArray` is a reference type that mlx-swift does not
/// mark `Sendable`, but these tensors are produced once and only read
/// downstream, so crossing an isolation boundary is safe in practice.
public struct OCR2Input: @unchecked Sendable {
    /// `(1, L)` int32 ids: BOS(0) + `<image>`-expanded image tokens + prompt
    /// text, with the final `inference_mode` token stripped.
    public let tokens: MLXArray
    /// `(V, 1024, 1024, 3)` bf16 global view(s), `(x/255-0.5)/0.5` normalized.
    public let pixelsGlobal: MLXArray
    /// `(P, 768, 768, 3)` bf16 local tiles, or `nil` when the image is not
    /// cropped (never `nil` under the reference's default `cropping`/`min=1`).
    public let pixelsPatches: MLXArray?
    /// `(1, L)` Bool; `true` exactly at image-token positions.
    public let seqMask: MLXArray
    /// One `[rows, cols]` grid per image (the reference's `images_spatial_crop`).
    public let spatialCrop: [[Int]]
}

/// Image preprocessing (dynamic tiling + normalization) and prompt
/// tokenization for DeepSeek-OCR-2 -- "OCR 质量第一决定因素". A reference-parity
/// port of `processing_deepseekocr.py` (`process_one` / `tokenize_with_images`
/// / `dynamic_preprocess` / `find_closest_aspect_ratio`).
///
/// **Resampling**: a hand-rolled PIL-equivalent separable bicubic (the plan's
/// pre-authorized fallback), not `MediaProcessing.resampleBicubic`. PIL
/// interpolates directly on the sRGB byte values; CoreImage's bicubic
/// interpolates in a color-managed space, so it cannot reproduce PIL's pixels.
/// This port replicates PIL's `precompute_coeffs` (bicubic a=-0.5) + two-pass
/// (horizontal then vertical) resample with intermediate uint8 rounding, which
/// matches the golden fixtures to ~1e-5 in the normalized domain.
public final class DeepSeekOCR2Processor: Sendable {
    // Sizes / counts (reference `process_one` defaults + Qwen2 encoder token
    // counts from `tokenize_with_images`).
    private let baseSize = 1024       // global view
    private let tileSize = 768        // local patch
    private let minPatches = 1
    private let maxPatches = 6
    private let tokensPerPatch = 144  // 12x12 SAM features (768px tile)
    private let tokensPerGlobal = 256 // 16x16 SAM features (1024px view)
    private let tokensViewSep = 1
    private let padGray: Float = 127  // ImageOps.pad color = int(0.5*255)
    private let imageToken = "<image>"

    private let imageTokenID: Int
    private let bosTokenID: Int
    private let eosTokenID: Int
    private let tokenizer: any Tokenizers.Tokenizer

    /// Loads the tokenizer from `modelDir/tokenizer.json` and reads
    /// `bos`/`eos`/`image` token ids from `config.json`. `strict: false` so the
    /// checkpoint's byte-level BPE `tokenizer.json` (custom Split pre-tokenizer)
    /// loads verbatim -- see Task 2's note on the Llama-fast decoder-glue trap.
    public init(modelDir: URL) async throws {
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir, strict: false)
        let cfg = try DeepSeekOCR2Configuration(
            mergingJSON: Data(contentsOf: modelDir.appending(path: "config.json")))
        self.imageTokenID = cfg.imageTokenID
        self.bosTokenID = cfg.bosTokenID
        self.eosTokenID = cfg.eosTokenID
    }

    /// `CGImage` + prompt -> model-ready inputs. `prompt` must contain exactly
    /// one `<image>` (single-image inference, e.g. `"<image>\nFree OCR. "`).
    public func prepare(image: CGImage, prompt: String) throws -> OCR2Input {
        let splits = prompt.components(separatedBy: imageToken)
        precondition(
            splits.count == 2,
            "prompt must contain exactly one \(imageToken), got \(splits.count - 1)")

        let rgb = Self.cgImageToRGB(image)  // (H, W, 3) f32 in [0, 255]

        // Global view: ImageOps.pad to 1024x1024 (aspect-preserving contain,
        // then center-pad with mean gray).
        let globalHWC = padToSquareContain(rgb)                        // (1024, 1024, 3) f32
        let pixelsGlobal = normalize(globalHWC).expandedDimensions(axis: 0)  // (1, 1024, 1024, 3) bf16

        // Local tiles: dynamic_preprocess.
        let (patchViews, crop) = dynamicPreprocess(rgb)                // [(768,768,3) f32], (rows, cols)
        let pixelsPatches = MLX.stacked(patchViews.map { normalize($0) }, axis: 0)  // (P,768,768,3) bf16
        let numPatches = patchViews.count

        // Tokens (structure + tokenizer).
        let (tokenIDs, maskBools) = buildTokens(splits: splits, numPatches: numPatches)
        let tokens = MLXArray(tokenIDs.map { Int32($0) }).reshaped(1, -1)
        let seqMask = MLXArray(maskBools).reshaped(1, -1)

        return OCR2Input(
            tokens: tokens, pixelsGlobal: pixelsGlobal, pixelsPatches: pixelsPatches,
            seqMask: seqMask, spatialCrop: [[crop.rows, crop.cols]])
    }

    /// Decodes token ids to text via the loaded byte-level BPE tokenizer's own
    /// ByteLevel decoder (correct CJK/multibyte), matching how `meta.json`'s
    /// reference text was produced (`skip_special_tokens=True`).
    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: true)
    }

    /// Decode with explicit control over whether special-token surface forms are
    /// kept. `skipSpecialTokens: false` preserves the grounding markers
    /// (`<|ref|>`/`<|det|>`/…, all `special:true` in tokenizer.json). Used by
    /// `OCR2Session`'s streaming detokenizer so free-OCR and grounding share one
    /// decode path with a per-task flag.
    public func decode(_ ids: [Int], skipSpecialTokens: Bool) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: skipSpecialTokens)
    }

    /// Marker-preserving decode for grounding output (Task 9 review finding):
    /// the `<|ref|>/<|/ref|>/<|det|>/<|/det|>` markers are `special:true`, so the
    /// plain `decode(_:)` (skip specials) STRIPS them. This mirrors the Python
    /// reference's grounding decode (`skip_special_tokens=False`) which keeps the
    /// markers and drops only the sentinel BOS(0)/EOS(1) ids manually.
    public func decodeKeepingMarkers(_ ids: [Int]) -> String {
        let filtered = ids.filter { $0 != bosTokenID && $0 != eosTokenID }
        return tokenizer.decode(tokens: filtered, skipSpecialTokens: false)
    }

    // MARK: - Tokenization

    private func buildTokens(splits: [String], numPatches: Int) -> (ids: [Int], mask: [Bool]) {
        var ids: [Int] = []
        var mask: [Bool] = []

        // Single image: text before it, then the expanded image block.
        let pre = tokenizer.encode(text: splits[0], addSpecialTokens: false)
        ids += pre
        mask += Array(repeating: false, count: pre.count)

        let n = numPatches * tokensPerPatch + tokensPerGlobal + tokensViewSep
        ids += Array(repeating: imageTokenID, count: n)
        mask += Array(repeating: true, count: n)

        // Text after the last image.
        let post = tokenizer.encode(text: splits[splits.count - 1], addSpecialTokens: false)
        ids += post
        mask += Array(repeating: false, count: post.count)

        // Prepend BOS (hardcoded 0 in the reference), then inference_mode strips
        // the final token.
        ids = [bosTokenID] + ids
        mask = [false] + mask
        ids.removeLast()
        mask.removeLast()

        return (ids, mask)
    }

    // MARK: - Global view (ImageOps.pad: contain + center-pad)

    private func padToSquareContain(_ rgb: MLXArray) -> MLXArray {
        let h = rgb.dim(0), w = rgb.dim(1)
        let imRatio = Double(w) / Double(h)

        // contain(): fit within (base, base) preserving aspect (Python round()
        // == round-half-to-even).
        let nw: Int, nh: Int
        if imRatio > 1.0 {
            nw = baseSize
            nh = Int((Double(baseSize) / imRatio).rounded(.toNearestOrEven))
        } else if imRatio < 1.0 {
            nh = baseSize
            nw = Int((Double(baseSize) * imRatio).rounded(.toNearestOrEven))
        } else {
            nw = baseSize; nh = baseSize
        }

        let resized = (nw == w && nh == h) ? rgb : resizeBicubic(rgb, outW: nw, outH: nh)

        let canvas = MLXArray.full([baseSize, baseSize, 3], values: MLXArray(padGray))  // f32
        // pad(): exactly one axis is short of `base`; center it (round-half-even).
        var x = 0, y = 0
        if nw != baseSize {
            x = Int((Double(baseSize - nw) * 0.5).rounded(.toNearestOrEven))
        } else {
            y = Int((Double(baseSize - nh) * 0.5).rounded(.toNearestOrEven))
        }
        canvas[y ..< (y + nh), x ..< (x + nw), 0...] = resized
        return canvas
    }

    // MARK: - Local tiles (dynamic_preprocess)

    private func dynamicPreprocess(_ rgb: MLXArray) -> (patches: [MLXArray], crop: (rows: Int, cols: Int)) {
        let h = rgb.dim(0), w = rgb.dim(1)
        let aspect = Double(w) / Double(h)

        // Candidate (i, j) grids with min <= i*j <= max, ordered by area (i*j).
        var ratios = Set<[Int]>()
        for n in minPatches ... maxPatches {
            for i in 1 ... n {
                for j in 1 ... n where (i * j) <= maxPatches && (i * j) >= minPatches {
                    ratios.insert([i, j])
                }
            }
        }
        // Deterministic order: area asc, then i asc, then j asc. (Reference sorts
        // a Python set by area; among our fixtures no exact ratio_diff tie occurs,
        // so the intra-area order is immaterial. The area-override tie-break below
        // is ported verbatim regardless.)
        let ordered = ratios.sorted {
            ($0[0] * $0[1], $0[0], $0[1]) < ($1[0] * $1[1], $1[0], $1[1])
        }
        let best = findClosestAspectRatio(aspect: aspect, ratios: ordered, w: w, h: h)
        let (gi, gj) = (best[0], best[1])

        let tw = tileSize * gi, th = tileSize * gj
        let resized = resizeBicubic(rgb, outW: tw, outH: th)  // (th, tw, 3)

        let cols = tw / tileSize  // == gi
        var patches: [MLXArray] = []
        patches.reserveCapacity(gi * gj)
        for b in 0 ..< (gi * gj) {
            let cx = (b % cols) * tileSize
            let cy = (b / cols) * tileSize
            patches.append(resized[cy ..< (cy + tileSize), cx ..< (cx + tileSize), 0...])
        }
        return (patches, (rows: gi, cols: gj))
    }

    /// Verbatim port of `find_closest_aspect_ratio`: min |aspect - i/j|, ties
    /// broken toward the larger grid when `area > 0.5 * tile^2 * i * j`.
    private func findClosestAspectRatio(aspect: Double, ratios: [[Int]], w: Int, h: Int) -> [Int] {
        var bestDiff = Double.infinity
        var best = [1, 1]
        let area = Double(w * h)
        for r in ratios {
            let target = Double(r[0]) / Double(r[1])
            let diff = abs(aspect - target)
            if diff < bestDiff {
                bestDiff = diff
                best = r
            } else if diff == bestDiff {
                if area > 0.5 * Double(tileSize) * Double(tileSize) * Double(r[0]) * Double(r[1]) {
                    best = r
                }
            }
        }
        return best
    }

    // MARK: - Normalization

    /// `(v/255 - 0.5)/0.5` in f32, then bf16 (reference ImageTransform).
    private func normalize(_ hwc: MLXArray) -> MLXArray {
        ((hwc / 255.0 - 0.5) / 0.5).asType(.bfloat16)
    }

    // MARK: - PIL-equivalent separable bicubic

    /// Resample `img` (H, W, C, f32 in [0,255]) to `(outH, outW, C)`.
    /// Horizontal (width) pass first, then vertical (height), each clamped and
    /// rounded to uint8 between passes -- exactly PIL's `ImagingResample`.
    private func resizeBicubic(_ img: MLXArray, outW: Int, outH: Int) -> MLXArray {
        let h = img.dim(0), w = img.dim(1), c = img.dim(2)
        var cur = img

        if outW != w {
            let wx = Self.bicubicMatrix(inSize: w, outSize: outW)   // (outW, W)
            // contract W: (H, C, W) @ (W, outW) -> (H, C, outW) -> (H, outW, C)
            let t = matmul(cur.transposed(0, 2, 1), wx.T)
            cur = Self.clip8(t.transposed(0, 2, 1))
        }
        if outH != h {
            let curW = cur.dim(1)
            let wy = Self.bicubicMatrix(inSize: h, outSize: outH)   // (outH, H)
            // contract H: (outH, H) @ (H, outW*C) -> (outH, outW*C) -> (outH, outW, C)
            let flat = cur.reshaped(h, curW * c)
            cur = Self.clip8(matmul(wy, flat).reshaped(outH, curW, c))
        }
        return cur
    }

    private static func clip8(_ a: MLXArray) -> MLXArray {
        MLX.clip(a.round(), min: 0, max: 255)
    }

    /// PIL `precompute_coeffs` for a single axis: a dense `(outSize, inSize)`
    /// weight matrix (each output row has bicubic support over the input).
    static func bicubicMatrix(inSize: Int, outSize: Int) -> MLXArray {
        let support = 2.0
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)        // >1 when downsampling (anti-alias)
        let fsupport = support * filterscale
        let ss = 1.0 / filterscale

        var weights = [Float](repeating: 0, count: outSize * inSize)
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            var xmin = Int(center - fsupport + 0.5)  // C (int) cast: truncate toward zero
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + fsupport + 0.5)
            if xmax > inSize { xmax = inSize }

            var ks = [Double]()
            ks.reserveCapacity(xmax - xmin)
            var ww = 0.0
            for xi in xmin ..< xmax {
                let wv = bicubicKernel((Double(xi) - center + 0.5) * ss)
                ks.append(wv)
                ww += wv
            }
            var k = 0
            for xi in xmin ..< xmax {
                weights[xx * inSize + xi] = Float(ww != 0.0 ? ks[k] / ww : 0.0)
                k += 1
            }
        }
        return MLXArray(weights, [outSize, inSize])
    }

    /// Bicubic (Keys) kernel with a = -0.5, support 2.0 -- PIL's `bicubic_filter`.
    private static func bicubicKernel(_ x0: Double) -> Double {
        let a = -0.5
        let x = abs(x0)
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0 }
        if x < 2.0 { return (((x - 5.0) * x + 8.0) * x - 4.0) * a }
        return 0.0
    }

    // MARK: - CGImage -> raw RGB

    /// Renders `cg` into an sRGB RGBA8 bitmap and returns `(H, W, 3)` f32 in
    /// [0,255], top-left origin -- matching `np.array(Image.open(png).convert
    /// ("RGB"))`. The PNGs carry no ICC/gamma, so sRGB rendering reproduces the
    /// raw stored bytes (verified: darkest row + values match PIL exactly).
    static func cgImageToRGB(_ cg: CGImage) -> MLXArray {
        let w = cg.width, h = cg.height
        let bytesPerRow = w * 4
        var buffer = [UInt8](repeating: 0, count: h * bytesPerRow)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        buffer.withUnsafeMutableBytes { ptr in
            let ctx = CGContext(
                data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: bytesPerRow, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        let rgba = MLXArray(Data(buffer), [h, w, 4], type: UInt8.self).asType(.float32)
        return rgba[0..., 0..., ..<3]  // drop alpha -> (H, W, 3)
    }
}
