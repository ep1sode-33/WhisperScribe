import Foundation
import SwiftUI

/// Progress state for a single model download. Promoted from a nested `ModelManager`
/// type to file scope (behaviour-preserving — no external code referenced
/// `ModelManager.DownloadState`) so the sibling `OCRModelManager` publishes the same
/// `DownloadState` without duplicating it.
enum DownloadState: Equatable {
    case idle
    case downloading(Double)   // 0...1
    case failed(String)
}

/// Single source of truth for Whisper model state: which model is selected, which are
/// installed on disk, and per-model download progress. Injected as an `EnvironmentObject`.
@MainActor
final class ModelManager: ObservableObject {

    @Published var selectedModelID: String {
        didSet { defaults.set(selectedModelID, forKey: Self.selectedKey) }
    }
    @Published private(set) var installedIDs: Set<String> = []
    @Published private(set) var downloads: [String: DownloadState] = [:]

    static let selectedKey = "selectedModelID"

    private let downloader: ModelDownloading
    private let defaults: UserDefaults
    private let fileManager: FileManager
    private let baseDir: URL
    private var tasks: [String: Task<Void, Never>] = [:]
    /// Per-id monotonic generation. Bumped on cancel/re-download so a superseded in-flight
    /// operation, unwinding after its `await`, can detect it is stale and skip mutating UI state.
    private var downloadSeq: [String: Int] = [:]

    init(downloader: ModelDownloading,
         defaults: UserDefaults = .standard,
         fileManager: FileManager = .default,
         baseDir: URL? = nil) {
        self.downloader = downloader
        self.defaults = defaults
        self.fileManager = fileManager
        let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true).appendingPathComponent("Documents")
        self.baseDir = baseDir
            ?? documents.appendingPathComponent("huggingface/models/argmaxinc/whisperkit-coreml", isDirectory: true)
        let saved = defaults.string(forKey: Self.selectedKey) ?? ""
        self.selectedModelID = WhisperModel.with(id: saved)?.id ?? WhisperModel.default.id
        refreshInstalled()
    }

    // MARK: - Derived state

    var selectedModel: WhisperModel { WhisperModel.with(id: selectedModelID) ?? .default }

    func isInstalled(_ m: WhisperModel) -> Bool { installedIDs.contains(m.id) }

    func state(for m: WhisperModel) -> DownloadState { downloads[m.id] ?? .idle }

    func modelFolder(_ m: WhisperModel) -> URL { baseDir.appendingPathComponent(m.variant, isDirectory: true) }

    func modelFolderPath(_ m: WhisperModel) -> String { modelFolder(m).path }

    /// Selected model is installed and not currently downloading.
    var isReady: Bool {
        guard installedIDs.contains(selectedModel.id) else { return false }
        if case .downloading = downloads[selectedModel.id] { return false }
        return true
    }

    // MARK: - Installed scan

    func refreshInstalled() {
        installedIDs = Set(WhisperModel.all.filter { isInstalledOnDisk($0) }.map(\.id))
    }

    private func isInstalledOnDisk(_ m: WhisperModel) -> Bool {
        let folder = modelFolder(m)
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: folder.path, isDirectory: &isDir), isDir.boolValue else { return false }
        let entries = (try? fileManager.contentsOfDirectory(atPath: folder.path)) ?? []
        return entries.contains { $0.hasSuffix(".mlmodelc") }
    }

    // MARK: - Download

    func download(_ m: WhisperModel) {
        guard tasks[m.id] == nil else { return }
        let token = (downloadSeq[m.id] ?? 0) + 1
        downloadSeq[m.id] = token
        tasks[m.id] = Task { [weak self] in await self?.performDownload(m, token: token) }
    }

    func cancelDownload(_ m: WhisperModel) {
        tasks[m.id]?.cancel()
        // Invalidate the in-flight generation so its suspended task, when it unwinds,
        // recognises it is superseded and leaves the latest operation's state intact.
        downloadSeq[m.id] = (downloadSeq[m.id] ?? 0) + 1
        tasks[m.id] = nil
        downloads[m.id] = .idle
    }

    /// The actual download flow. `internal` so tests can await it deterministically.
    /// `token` pins this invocation to a generation; only the CURRENT generation may mutate
    /// `downloads`/`tasks`/`selectedModelID`. Refreshing disk truth is never gated.
    func performDownload(_ m: WhisperModel, token: Int) async {
        if case .downloading = downloads[m.id] { return }
        downloads[m.id] = .downloading(0)
        do {
            _ = try await downloader.download(variant: m.variant) { [weak self] frac in
                Task { @MainActor in self?.updateProgress(m.id, token: token, frac) }
            }
            try Task.checkCancellation()
            refreshInstalled()
            guard downloadSeq[m.id, default: 0] == token else { return }
            downloads[m.id] = .idle
            tasks[m.id] = nil
            // If the previously-selected model isn't usable, adopt this freshly installed one.
            if !installedIDs.contains(selectedModel.id), installedIDs.contains(m.id) {
                selectedModelID = m.id
            }
        } catch is CancellationError {
            // A download may have landed on disk just before cancellation threw; register it.
            refreshInstalled()
            guard downloadSeq[m.id, default: 0] == token else { return }
            downloads[m.id] = .idle
            tasks[m.id] = nil
        } catch {
            guard downloadSeq[m.id, default: 0] == token else { return }
            downloads[m.id] = .failed(AppError.modelDownloadFailed(error.localizedDescription).localizedDescription)
            tasks[m.id] = nil
        }
    }

    func updateProgress(_ id: String, token: Int, _ raw: Double) {
        // Stale callbacks from a superseded (cancelled + re-issued) generation must not write the
        // NEW task's fraction, even if the id is again `.downloading`.
        guard downloadSeq[id, default: 0] == token else { return }
        // Late callbacks after completion/cancel see a non-downloading state → ignored.
        guard case .downloading(let current) = downloads[id] else { return }
        downloads[id] = .downloading(Self.clampedProgress(current: current, raw: raw))
    }

    /// Monotonic, finite-clamped progress in 0...1. Pure — unit tested directly.
    nonisolated static func clampedProgress(current: Double, raw: Double) -> Double {
        let clamped = raw.isFinite ? min(max(raw, 0), 1) : 0
        return max(current, clamped)
    }

    // MARK: - Delete (reclaim disk)

    func delete(_ m: WhisperModel) {
        try? fileManager.removeItem(at: modelFolder(m))
        refreshInstalled()
        if selectedModel.id == m.id, !installedIDs.contains(m.id) {
            selectedModelID = WhisperModel.all.first(where: { installedIDs.contains($0.id) })?.id ?? WhisperModel.default.id
        }
    }
}
