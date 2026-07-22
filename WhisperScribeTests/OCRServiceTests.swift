import Testing
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Foundation
@testable import WhisperScribe

/// 供 BatchPipelineTests 复用的 fake（本文件内 internal）
final class FakeOCR: OCRProviding, @unchecked Sendable {
    var prepared = false
    var unloadCalled = false                        // records model-memory release
    var results: [String: String] = [:]           // fileName → text
    var failOn: Set<String> = []
    func prepare(modelDir: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        prepared = true; progress(1.0)
    }
    func recognize(imageAt url: URL, onChunk: @escaping @Sendable (String) -> Void) async throws -> String {
        let name = url.lastPathComponent
        if failOn.contains(name) { throw AppError.ocrFailed(name) }
        let text = results[name, default: "text-of-\(name)"]
        onChunk(text)
        return text
    }
    func unload() async { prepared = false; unloadCalled = true }
}

struct OCRServiceTests {
    @Test func orientedImageLoaderHonorsEXIF() throws {
        // 生成一张 2x1 图，写入 orientation=6（右转90°），加载后应为 1x2
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appending(path: "o.jpg")
        let ctx = CGContext(data: nil, width: 2, height: 1, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let img = ctx.makeImage()!
        let dest = CGImageDestinationCreateWithURL(path as CFURL, UTType.jpeg.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, img, [kCGImagePropertyOrientation: 6] as CFDictionary)
        #expect(CGImageDestinationFinalize(dest))
        let loaded = try OCRService.loadOrientedImage(at: path)
        #expect(loaded.width == 1 && loaded.height == 2)
    }
    @Test func fakeConformsAndFails() async {
        let fake = FakeOCR(); fake.failOn = ["bad.png"]
        await #expect(throws: AppError.self) {
            _ = try await fake.recognize(imageAt: URL(fileURLWithPath: "/tmp/bad.png"), onChunk: { _ in })
        }
    }
}
