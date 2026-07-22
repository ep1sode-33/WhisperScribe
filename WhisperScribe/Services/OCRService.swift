import CoreGraphics
import DeepSeekOCR2Kit
import Foundation
import ImageIO

/// Protocol seam over DeepSeek-OCR-2 so the batch pipeline can be tested with a
/// fake (`FakeOCR` in the test target) without loading the ~3GB checkpoint.
protocol OCRProviding: Sendable {
    /// Loads the model from `modelDir`. Idempotent: a second call after a
    /// successful load reports `progress(1.0)` and returns immediately.
    func prepare(modelDir: URL, progress: @escaping @Sendable (Double) -> Void) async throws
    /// Runs OCR on the image at `url`, streaming decoded text via `onChunk` and
    /// returning the full transcript. Throws `AppError.ocrModelMissing` if the
    /// model was never prepared, `AppError.ocrFailed` on a generation error, and
    /// re-throws `CancellationError` on cancellation.
    func recognize(imageAt url: URL, onChunk: @escaping @Sendable (String) -> Void) async throws -> String
    /// Drops the loaded session, freeing model memory.
    func unload() async
}

/// App-side actor wrapping the kit's `OCR2Session`. Serializes access to the
/// (non-reentrant) session and normalizes EXIF orientation before OCR.
actor OCRService: OCRProviding {
    private var session: OCR2Session?

    func prepare(modelDir: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        guard session == nil else { progress(1.0); return }
        session = try await OCR2Session.load(from: modelDir, progress: progress)
    }

    func recognize(imageAt url: URL, onChunk: @escaping @Sendable (String) -> Void) async throws -> String {
        guard let session else { throw AppError.ocrModelMissing }
        let image = try Self.loadOrientedImage(at: url)
        var full = ""
        do {
            for try await chunk in session.ocr(image: image, task: .freeOCR) {
                full += chunk; onChunk(chunk)
            }
        } catch is CancellationError { throw CancellationError() }
        catch { throw AppError.ocrFailed(url.lastPathComponent) }
        return full
    }

    func unload() { session = nil }

    /// Loads `url` as an orientation-normalized `CGImage`. Cameras/phones store
    /// pixels in the sensor's native layout and record the display rotation as an
    /// EXIF `Orientation` tag; a plain `CGImageSourceCreateImageAtIndex` returns
    /// those un-rotated pixels. When the tag is not `.up` (1) we route through
    /// ImageIO's thumbnail path with `kCGImageSourceCreateThumbnailWithTransform:
    /// true`, which bakes the orientation transform into the pixels;
    /// `kCGImageSourceThumbnailMaxPixelSize` is pinned to the larger dimension so
    /// nothing is downscaled (the value is orientation-invariant). Upright images
    /// take the plain decode path unchanged. Any failure surfaces as
    /// `AppError.unsupportedFile`. Ported from
    /// `DeepSeekOCR2Kit/Sources/ocr2-cli/OCR2CLI.swift`'s `loadOrientedImage`.
    nonisolated static func loadOrientedImage(at url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil)
        else { throw AppError.unsupportedFile(url.lastPathComponent) }

        let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
        let orientation = (props?[kCGImagePropertyOrientation] as? UInt32) ?? 1

        if orientation == 1 {
            guard let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
            else { throw AppError.unsupportedFile(url.lastPathComponent) }
            return cg
        }

        var options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceCreateThumbnailFromImageAlways: true,
        ]
        let w = (props?[kCGImagePropertyPixelWidth] as? Int) ?? 0
        let h = (props?[kCGImagePropertyPixelHeight] as? Int) ?? 0
        if max(w, h) > 0 {
            options[kCGImageSourceThumbnailMaxPixelSize] = max(w, h)
        }
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary)
        else { throw AppError.unsupportedFile(url.lastPathComponent) }
        return cg
    }
}
