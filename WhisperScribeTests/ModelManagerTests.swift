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

/// A FIFO gate: each `wait()` parks until the test resumes it (one waiter at a time via
/// `resumeNext()`). Lets a test suspend two overlapping downloads and unwind them in a chosen
/// order, deterministically reproducing the cancel/re-download race. Thread-safe (continuations
/// may be resumed off the main actor before performDownload hops back).
private final class Gate: @unchecked Sendable {
    private let lock = NSLock()
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        await withCheckedContinuation { cont in
            lock.lock()
            waiters.append(cont)
            lock.unlock()
        }
    }

    /// Number of downloads currently parked on the gate. Used by tests to wait until a task has
    /// actually reached (and registered at) its suspension point before resuming it.
    var waiterCount: Int {
        lock.lock(); defer { lock.unlock() }
        return waiters.count
    }

    /// Resume the oldest still-parked waiter (FIFO). No-op if none is parked yet.
    func resumeNext() {
        lock.lock()
        let next = waiters.isEmpty ? nil : waiters.removeFirst()
        lock.unlock()
        next?.resume()
    }
}

/// Like `FakeDownloader`, but each `download` suspends on a shared `Gate` until the test resumes
/// it, then creates the `<baseDir>/<variant>/Model.mlmodelc` folder — so the test controls exactly
/// when each in-flight download unwinds.
private struct SuspendingDownloader: ModelDownloading {
    let baseDir: URL
    let gate: Gate

    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        await gate.wait()
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

/// Cooperatively yield the main actor (letting spawned download tasks run) until `cond` holds.
/// Bounded so a never-true condition fails an assertion instead of hanging the suite.
@MainActor
private func yieldUntil(_ cond: () -> Bool) async {
    for _ in 0..<10_000 {
        if cond() { return }
        await Task.yield()
    }
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
        await m.performDownload(model, token: 0)
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
        await m.performDownload(model, token: 0)
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

    @Test @MainActor func failedDownloadUsesLocalizedAppErrorText() async throws {
        let base = try tempDir()
        let m = makeManager(FakeDownloader(baseDir: base, fail: true), baseDir: base)
        let model = WhisperModel.default
        await m.performDownload(model, token: 0)
        let expected = AppError.modelDownloadFailed(FakeDownloader.Boom.fail.localizedDescription).localizedDescription
        #expect(m.state(for: model) == .failed(expected))
    }

    /// Fix 1: a superseded in-flight download must not clobber the latest operation's state.
    ///
    /// Deterministic interleaving of the exact race: start X (TaskA suspends inside `download`),
    /// cancel X (bumps the generation, cancels TaskA), re-download X (TaskB suspends). We then
    /// unwind TaskA FIRST — while TaskB is still live — and assert TaskA's stale catch leaves
    /// TaskB's `.downloading` state and tracked task intact, then unwind TaskB and assert it is
    /// the only operation that installs/selects. Without the token guard, TaskA's
    /// `catch is CancellationError` would reset `downloads` to `.idle` and nil `tasks`, leaving
    /// TaskB an untracked live download — which this test would catch at the first expectation.
    @Test @MainActor func supersededDownloadDoesNotClobberLatestOperation() async throws {
        let base = try tempDir()
        let gate = Gate()
        let m = makeManager(SuspendingDownloader(baseDir: base, gate: gate), baseDir: base)
        let model = WhisperModel.with(id: "distilV3")!

        // 1. Start X → TaskA sets .downloading(0) then suspends, parking on the gate.
        m.download(model)
        await yieldUntil { gate.waiterCount == 1 }
        #expect(m.state(for: model) == .downloading(0))

        // 2. Cancel X → generation bumped, TaskA cancelled (still parked on the gate), state .idle.
        m.cancelDownload(model)
        #expect(m.state(for: model) == .idle)

        // 3. Re-download X → TaskB (current generation) sets .downloading(0) then parks too.
        m.download(model)
        await yieldUntil { gate.waiterCount == 2 }
        #expect(m.state(for: model) == .downloading(0))

        // 4. Unwind TaskA first (FIFO). It was cancelled → throws → stale catch must NOT mutate.
        gate.resumeNext()
        await yieldUntil { m.isInstalled(model) }   // TaskA's refreshInstalled() registers the folder
        #expect(m.state(for: model) == .downloading(0))  // TaskB NOT clobbered to .idle by stale TaskA

        // 5. Unwind TaskB (current) → it is the sole writer that completes the install + auto-select.
        gate.resumeNext()
        await yieldUntil { m.state(for: model) == .idle }
        #expect(m.isInstalled(model))
        #expect(m.selectedModel.id == "distilV3")
        #expect(m.isReady)
    }
}
