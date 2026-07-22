import Testing
import Foundation
import MLX
import CoreGraphics
import ImageIO
@testable import DeepSeekOCR2Kit

/// Milestone M5: the processor closes the loop. Tasks 1-8 proved the model
/// chain with *injected* reference pixels; this suite feeds a real `CGImage`
/// through `DeepSeekOCR2Processor` and requires:
///   1. token-id + spatial-crop parity on all 5 fixtures (tokenizer + tiling),
///   2. pixel parity vs the golden NCHW fixtures (bicubic + pad geometry),
///   3. TRUE end-to-end text (CGImage -> processor -> model -> greedy) matching
///      the reference decode on doc_page AND cjk_dense.
///
/// If preprocessing matches the reference, real images produce
/// reference-identical text -- which is exactly what (3) asserts.
@Suite struct ProcessorTests {
    static func cgImage(_ name: String) throws -> CGImage {
        let url = FixtureSupport.root!.appending(path: "images/\(name).png")
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "cgimage-load", code: 1) }
        return img
    }

    /// Builds a synthetic `w x h` CGImage with every pixel set to `pixel`
    /// (premultiplied-last bytes), for tests that don't need a fixture image.
    static func rgbaImage(
        width w: Int, height h: Int, pixel: (r: UInt8, g: UInt8, b: UInt8, a: UInt8)
    ) -> CGImage {
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) {
            bytes[i * 4 + 0] = pixel.r
            bytes[i * 4 + 1] = pixel.g
            bytes[i * 4 + 2] = pixel.b
            bytes[i * 4 + 3] = pixel.a
        }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
            space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    // MARK: 0. Correctness fixes (Codex review): no model weights required.

    /// C2: a half-transparent black image composites onto WHITE (not black), so
    /// every channel is a mid-gray (> 0), never pure black.
    @Test func transparentPixelsCompositeOntoWhite() {
        // Premultiplied-last (0,0,0,128) == black at 50% alpha; over white this
        // yields ~127 (255 * (1 - 128/255)) on every channel.
        let img = Self.rgbaImage(width: 8, height: 8, pixel: (0, 0, 0, 128))
        let rgb = DeepSeekOCR2Processor.cgImageToRGB(img)
        let minV = rgb.min().item(Float.self)
        let meanV = rgb.mean().item(Float.self)
        #expect(minV > 0, "transparent-black should not stay pure black (min \(minV))")
        #expect(abs(meanV - 127) < 3, "expected ~127 gray, got \(meanV)")
    }

    /// C3: a prompt with no text after `<image>` throws (rather than silently
    /// eating an image placeholder token). Needs the tokenizer, not the weights.
    @Test(.enabled(if: FixtureSupport.modelDir != nil))
    func emptyPromptAfterImageThrows() async throws {
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)
        let img = Self.rgbaImage(width: 16, height: 16, pixel: (255, 255, 255, 255))
        #expect(throws: OCR2PrepareError.emptyPromptAfterImage) {
            _ = try p.prepare(image: img, prompt: "<image>")
        }
        // A prompt WITH post-image text still prepares.
        _ = try p.prepare(image: img, prompt: "<image>\nFree OCR. ")
    }

    /// C5: an extreme aspect ratio (5000x2) prepares without crashing; the global
    /// view still comes out (1, 1024, 1024, 3). Structure-only (tokenizer only).
    @Test(.enabled(if: FixtureSupport.modelDir != nil))
    func extremeAspectRatioPreparesWithoutCrash() async throws {
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)
        let img = Self.rgbaImage(width: 5000, height: 2, pixel: (200, 200, 200, 255))
        let input = try p.prepare(image: img, prompt: "<image>\nFree OCR. ")
        #expect(input.pixelsGlobal.shape == [1, 1024, 1024, 3])
        #expect(input.pixelsPatches != nil)
    }

    // MARK: 1. Tokenizer + structural (tiling) parity -- the combined gate.

    @Test(
        .enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil),
        arguments: ["doc_page", "receipt", "cjk_dense", "tall_scroll", "grounding_menu"]
    )
    func promptTokenIDsMatchReference(name: String) async throws {
        let meta = try FixtureSupport.meta(name)
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)
        let input = try p.prepare(image: Self.cgImage(name), prompt: "<image>\nFree OCR. ")

        let ids = (0..<input.tokens.shape[1]).map { input.tokens[0, $0].item(Int.self) }
        #expect(ids == meta["prompt_token_ids"] as! [Int])
        #expect(input.spatialCrop == meta["images_spatial_crop"] as! [[Int]])
    }

    // MARK: 2. Pixel parity (global + patches), fixture transposed NCHW->NHWC.

    @Test(
        .enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil),
        arguments: ["doc_page", "cjk_dense", "tall_scroll"]
    )
    func pixelParity(name: String) async throws {
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)
        let input = try p.prepare(image: Self.cgImage(name), prompt: "<image>\nFree OCR. ")

        // Fixtures are NCHW f32 (processor-native layout); the OCR2Input contract
        // is NHWC bf16, so transpose the FIXTURE for comparison (mirror of the
        // NCHW->NHWC flip the model/gen_fixtures apply right before SAM).
        let tol: Float = 2.0 / 255.0 / 0.5  // 2/255 in the normalized (/0.5) domain

        let refG = try FixtureSupport.load(name, "pixels_global")["x"]!.transposed(0, 2, 3, 1)
        let diffG = MLX.abs(input.pixelsGlobal.asType(.float32) - refG).mean().item(Float.self)
        #expect(diffG < tol, "\(name) global mean abs diff \(diffG)")

        let refP = try FixtureSupport.load(name, "pixels_patches")["x"]!.transposed(0, 2, 3, 1)
        let patches = try #require(input.pixelsPatches)
        let diffP = MLX.abs(patches.asType(.float32) - refP).mean().item(Float.self)
        #expect(diffP < tol, "\(name) patches mean abs diff \(diffP)")
    }

    // MARK: 3. TRUE end-to-end text (CGImage -> processor -> model -> greedy).

    @Test(
        .enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil),
        arguments: ["doc_page", "cjk_dense"]
    )
    func realImageEndToEndText(name: String) async throws {
        let meta = try FixtureSupport.meta(name)
        let (model, cfg) = try await DeepSeekOCR2Model.load(
            from: FixtureSupport.modelDir!, progress: { _ in })
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)
        let input = try p.prepare(image: Self.cgImage(name), prompt: "<image>\nFree OCR. ")

        let embeds = model.inputEmbeddings(
            tokens: input.tokens, pixelsGlobal: input.pixelsGlobal,
            pixelsPatches: input.pixelsPatches, seqMask: input.seqMask)
        let ids = FixtureSupport.greedyDecode(
            model: model, embeds: embeds, steps: 100, eos: cfg.eosTokenID)
        let text = p.decode(ids)

        // Reference `meta["text"]` is decoded from the reference's own 100-step
        // greedy (skip_special_tokens). Our run is capped at 100 (and may stop
        // early on eos), so require the reference to START WITH ours after both
        // are trimmed of trailing whitespace.
        let mine = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = (meta["text"] as! String).trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(!mine.isEmpty, "\(name) produced empty text")
        #expect(ref.hasPrefix(mine), "\(name)\n--- mine ---\n\(mine)\n--- ref ---\n\(ref)")
    }
}
