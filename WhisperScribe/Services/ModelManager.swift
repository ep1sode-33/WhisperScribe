import Foundation
import SwiftUI

/// Single source of truth for Whisper model state: which model is selected, which are
/// installed on disk, and per-model download progress. Injected as an `EnvironmentObject`.
@MainActor
final class ModelManager: ObservableObject {

    enum DownloadState: Equatable {
        case idle
        case downloading(Double)   // 0...1
        case failed(String)
    }

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
        tasks[m.id] = Task { [weak self] in await self?.performDownload(m) }
    }

    func cancelDownload(_ m: WhisperModel) {
        tasks[m.id]?.cancel()
        tasks[m.id] = nil
        downloads[m.id] = .idle
    }

    /// The actual download flow. `internal` so tests can await it deterministically.
    func performDownload(_ m: WhisperModel) async {
        if case .downloading = downloads[m.id] { return }
        downloads[m.id] = .downloading(0)
        do {
            _ = try await downloader.download(variant: m.variant) { [weak self] frac in
                Task { @MainActor in self?.updateProgress(m.id, frac) }
            }
            try Task.checkCancellation()
            downloads[m.id] = .idle
            tasks[m.id] = nil
            refreshInstalled()
            // If the previously-selected model isn't usable, adopt this freshly installed one.
            if !installedIDs.contains(selectedModel.id), installedIDs.contains(m.id) {
                selectedModelID = m.id
            }
        } catch is CancellationError {
            downloads[m.id] = .idle
            tasks[m.id] = nil
        } catch {
            downloads[m.id] = .failed(error.localizedDescription)
            tasks[m.id] = nil
        }
    }

    private func updateProgress(_ id: String, _ raw: Double) {
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
            selectedModelID = installedIDs.sorted().first ?? WhisperModel.default.id
        }
    }
}
