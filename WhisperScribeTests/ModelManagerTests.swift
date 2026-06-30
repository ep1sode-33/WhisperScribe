import Foundation
import Testing
@testable import WhisperScribe

/// Fake downloader: optionally fails; otherwise reports the given progress values
/// and creates a `<baseDir>/<variant>/Model.mlmodelc` folder so the installed scan sees it.
private struct FakeDownloader: ModelDownloading {
    let baseDir: URL
    var fail: Bool = false
    var progressValues: [Double] = [0.5, 1.0]
    enum Boom: Error { case fail }

    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        if fail { throw Boom.fail }
        for v in progressValues { progress(v) }
        let folder = baseDir.appendingPathComponent(variant, isDirectory: true)
        try FileManager.default.createDirectory(
            at: folder.appendingPathComponent("Model.mlmodelc", isDirectory: true),
            withIntermediateDirectories: true)
        return folder
    }
}

@MainActor
private func makeManager(_ downloader: ModelDownloading, baseDir: URL) -> ModelManager {
    let defaults = UserDefaults(suiteName: "test.\(UUID().uuidString)")!
    return ModelManager(downloader: downloader, defaults: defaults, fileManager: .default, baseDir: baseDir)
}

private func tempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

struct ModelManagerTests {

    @Test @MainActor func freshManagerIsNotReadyAndDefaultsToLargeV3() throws {
        let base = try tempDir()
        let m = makeManager(FakeDownloader(baseDir: base), baseDir: base)
        #expect(m.selectedModel.id == "largeV3")
        #expect(m.isReady == false)
        #expect(m.installedIDs.isEmpty)
    }

    @Test @MainActor func modelFolderResolvesUnderBase() throws {
        let base = try tempDir()
        let m = makeManager(FakeDownloader(baseDir: base), baseDir: base)
        let model = WhisperModel.default
        #expect(m.modelFolder(model) == base.appendingPathComponent(model.variant, isDirectory: true))
    }

    @Test @MainActor func successfulDownloadInstallsAndAutoSelects() async throws {
        let base = try tempDir()
        let m = makeManager(FakeDownloader(baseDir: base), baseDir: base)
        let model = WhisperModel.with(id: "distilV3")!
        await m.performDownload(model)
        #expect(m.isInstalled(model))
        #expect(m.state(for: model) == .idle)
        // largeV3 (the default selection) is NOT installed, so the freshly installed model wins.
        #expect(m.selectedModel.id == "distilV3")
        #expect(m.isReady == true)
    }

    @Test @MainActor func failedDownloadSurfacesFailedState() async throws {
        let base = try tempDir()
        let m = makeManager(FakeDownloader(baseDir: base, fail: true), baseDir: base)
        let model = WhisperModel.default
        await m.performDownload(model)
        if case .failed = m.state(for: model) {} else { Issue.record("expected .failed, got \(m.state(for: model))") }
        #expect(m.isInstalled(model) == false)
    }

    @Test @MainActor func installedScanRequiresMlmodelc() throws {
        let base = try tempDir()
        let model = WhisperModel.default
        // Empty variant folder → not installed.
        try FileManager.default.createDirectory(at: base.appendingPathComponent(model.variant), withIntermediateDirectories: true)
        let m = makeManager(FakeDownloader(baseDir: base), baseDir: base)
        #expect(m.isInstalled(model) == false)
    }

    @Test func clampedProgressIsMonotonicAndFinite() {
        #expect(ModelManager.clampedProgress(current: 0.4, raw: 0.6) == 0.6)
        #expect(ModelManager.clampedProgress(current: 0.7, raw: 0.6) == 0.7) // never goes backward
        #expect(ModelManager.clampedProgress(current: 0.0, raw: .nan) == 0.0)
        #expect(ModelManager.clampedProgress(current: 0.0, raw: 2.0) == 1.0)
    }
}
