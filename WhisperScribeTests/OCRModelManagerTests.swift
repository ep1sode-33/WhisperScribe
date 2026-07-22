import Testing
import Foundation
@testable import WhisperScribe

private struct FakeOCRDownloader: ModelDownloading {
    var fail = false
    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        if fail { throw URLError(.networkConnectionLost) }
        progress(1.0)
        return URL(fileURLWithPath: "/tmp/unused")
    }
}

@MainActor private func makeOCRManager(_ d: ModelDownloading, dir: URL) -> OCRModelManager {
    OCRModelManager(downloader: d, baseDir: dir)
}
@MainActor private func plantModel(in dir: URL) throws {
    let m = dir.appending(path: OCRModelManager.repoID)
    try FileManager.default.createDirectory(at: m, withIntermediateDirectories: true)
    for f in ["config.json", "tokenizer.json", "model.safetensors"] {
        FileManager.default.createFile(atPath: m.appending(path: f).path, contents: Data("x".utf8))
    }
}

struct OCRModelManagerTests {
    @Test @MainActor func downloadSuccessMarksInstalled() async throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let m = makeOCRManager(FakeOCRDownloader(), dir: dir)
        try plantModel(in: dir)          // fake 下载器不真写盘——就绪由磁盘态判定
        await m.performDownload(token: 0)
        #expect(m.installed && m.isReady)
    }
    @Test @MainActor func downloadFailureSetsFailed() async {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let m = makeOCRManager(FakeOCRDownloader(fail: true), dir: dir)
        await m.performDownload(token: 0)
        if case .failed = m.state {} else { Issue.record("expected .failed, got \(m.state)") }
    }
    @Test @MainActor func readinessNeedsAllThreeFiles() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let m = makeOCRManager(FakeOCRDownloader(), dir: dir)
        let folder = dir.appending(path: OCRModelManager.repoID)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: folder.appending(path: "config.json").path, contents: Data())
        m.refreshInstalled()
        #expect(!m.installed)            // 缺 tokenizer/safetensors
    }
    @Test @MainActor func deleteRemovesAndResets() async throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let m = makeOCRManager(FakeOCRDownloader(), dir: dir)
        try plantModel(in: dir); m.refreshInstalled(); #expect(m.installed)
        m.delete()
        #expect(!m.installed)
    }
}
