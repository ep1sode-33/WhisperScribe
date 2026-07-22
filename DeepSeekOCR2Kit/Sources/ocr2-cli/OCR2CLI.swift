import ArgumentParser
import CoreGraphics
import DeepSeekOCR2Kit
import Foundation
import ImageIO

/// `ocr2-cli <image> [--model-dir DIR] [--grounding [--query TEXT]]`
///
/// Streams DeepSeek-OCR-2 output to stdout as it is generated. Progress and (in
/// grounding mode) the parsed boxes go to stderr, so stdout stays a clean
/// transcript. In grounding mode the raw marker text is streamed AND the parsed
/// `label + normalized rect` list is printed, so a manual smoke check is
/// meaningful.
@main
struct OCR2CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ocr2-cli",
        abstract: "DeepSeek-OCR-2 command-line interface (streaming OCR + grounding)"
    )

    @Argument(help: "image path") var image: String
    @Option(help: "model dir (default: $OCR2_MODEL_DIR)") var modelDir: String?
    @Flag(help: "grounding mode; pass the text to locate via --query") var grounding = false
    @Option(help: "grounding query (text to locate)") var query: String = "all the text"

    mutating func run() async throws {
        guard let dir = modelDir ?? ProcessInfo.processInfo.environment["OCR2_MODEL_DIR"]
        else { throw ValidationError("--model-dir or OCR2_MODEL_DIR required") }
        let cg = try Self.loadOrientedImage(path: image)

        let session = try await OCR2Session.load(from: URL(fileURLWithPath: dir)) {
            FileHandle.standardError.write("\rloading \(Int($0 * 100))%".data(using: .utf8)!)
        }
        FileHandle.standardError.write("\n".data(using: .utf8)!)

        let task: OCRTask = grounding ? .grounding(query: query) : .freeOCR

        var full = ""
        for try await chunk in session.ocr(image: cg, task: task) {
            print(chunk, terminator: "")
            fflush(stdout)
            full += chunk
        }
        print()

        if grounding {
            let boxes = OCR2Session.parseGrounding(full)
            let header = "\n--- parsed \(boxes.count) box(es) ---\n"
            FileHandle.standardError.write(header.data(using: .utf8)!)
            for b in boxes {
                let r = b.box
                let line = String(
                    format: "  %@  x=[%.3f,%.3f] y=[%.3f,%.3f]\n",
                    b.label, r.minX, r.maxX, r.minY, r.maxY)
                FileHandle.standardError.write(line.data(using: .utf8)!)
            }
        }
    }

    /// Loads `path` as an orientation-normalized `CGImage`. Cameras and phones
    /// store pixels in the sensor's native layout and record the display
    /// rotation as an EXIF `Orientation` tag; a plain
    /// `CGImageSourceCreateImageAtIndex` returns those un-rotated pixels, so a
    /// portrait photo shot sideways would reach the processor rotated. When the
    /// tag is not `.up` (1) we route through ImageIO's thumbnail path with
    /// `kCGImageSourceCreateThumbnailWithTransform: true`, which bakes the
    /// orientation transform into the pixels. `kCGImageSourceThumbnailMaxPixelSize`
    /// is pinned to the image's larger dimension so nothing is downscaled (the
    /// value is orientation-invariant: a 90 degree tag swaps width/height but not
    /// their max). Upright images (the common case, and every parity fixture)
    /// take the plain decode path unchanged.
    static func loadOrientedImage(path: String) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil)
        else { throw ValidationError("cannot read image: \(path)") }

        let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
        let orientation = (props?[kCGImagePropertyOrientation] as? UInt32) ?? 1

        if orientation == 1 {
            guard let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
            else { throw ValidationError("cannot decode image: \(path)") }
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
        else { throw ValidationError("cannot decode image: \(path)") }
        return cg
    }
}

// The `@main` attribute is the entry point (it resolves unambiguously to
// `AsyncParsableCommand.main() async`). It lives in `OCR2CLI.swift` -- NOT
// `main.swift` -- because a file literally named `main.swift` permits top-level
// code, which Xcode's build system (required to run Metal-backed MLX tests via
// `xcodebuild test`) rejects alongside an `@main` type. Renaming the file
// sidesteps that entirely.
