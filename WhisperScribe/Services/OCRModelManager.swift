import Foundation
import Hub
import SwiftUI

/// Single source of truth for the DeepSeek-OCR-2 model: whether the ~3GB checkpoint is
/// installed on disk and its download progress. Sibling to `ModelManager` (Whisper) — the
/// OCR model is a single HuggingFace repo with no variant catalogue, so state is scalar
/// rather than keyed by id. The generation-token / progress-clamping / single-task-guard
/// machinery mirrors `ModelManager` line-for-line. Injected as an `EnvironmentObject`.
@MainActor
final class OCRModelManager: ObservableObject {

    static let repoID = "mlx-community/DeepSeek-OCR-2-8bit"

    @Published var state: DownloadState = .idle
    @Published private(set) var installed: Bool = false

    /// The on-disk model directory (`<baseDir>/<repoID>`). Readiness is judged here, and
    /// `delete()` reclaims it. In production it coincides with where `OCRHubDownloader`
    /// (HubApi) lands the snapshot: `~/Documents/huggingface/models/<repoID>`.
    let modelDir: URL

    private let downloader: ModelDownloading
    private let fileManager: FileManager
    private var task: Task<Void, Never>?
    /// Monotonic generation. Bumped on cancel/re-download so a superseded in-flight
    /// operation, unwinding after its `await`, can detect it is stale and skip mutating state.
    private var downloadSeq: Int = 0

    init(downloader: ModelDownloading = OCRHubDownloader(), baseDir: URL? = nil) {
        self.downloader = downloader
        let fm = FileManager.default
        self.fileManager = fm
        let base = baseDir ?? {
            let documents = fm.urls(for: .documentDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true).appendingPathComponent("Documents")
            return documents.appendingPathComponent("huggingface/models", isDirectory: true)
        }()
        self.modelDir = base.appendingPathComponent(Self.repoID, isDirectory: true)
        refreshInstalled()
    }

    // MARK: - Derived state

    /// Installed on disk and not currently downloading.
    var isReady: Bool {
        guard installed else { return false }
        if case .downloading = state { return false }
        return true
    }

    // MARK: - Installed scan

    /// Ready iff the model dir holds `config.json`, `tokenizer.json`, and at least one
    /// `*.safetensors` weight shard.
    func refreshInstalled() {
        installed = isInstalledOnDisk()
    }

    private func isInstalledOnDisk() -> Bool {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: modelDir.path, isDirectory: &isDir), isDir.boolValue else { return false }
        let entries = (try? fileManager.contentsOfDirectory(atPath: modelDir.path)) ?? []
        return entries.contains("config.json")
            && entries.contains("tokenizer.json")
            && entries.contains { $0.hasSuffix(".safetensors") }
    }

    // MARK: - Download

    func download() {
        guard task == nil else { return }
        let token = downloadSeq + 1
        downloadSeq = token
        task = Task { [weak self] in await self?.performDownload(token: token) }
    }

    func cancelDownload() {
        task?.cancel()
        // Invalidate the in-flight generation so its suspended task, when it unwinds,
        // recognises it is superseded and leaves the latest operation's state intact.
        downloadSeq += 1
        task = nil
        state = .idle
    }

    /// The actual download flow. `internal` so tests can await it deterministically.
    /// `token` pins this invocation to a generation; only the CURRENT generation may mutate
    /// `state`/`task`. Refreshing disk truth is never gated.
    func performDownload(token: Int) async {
        if case .downloading = state { return }
        state = .downloading(0)
        do {
            _ = try await downloader.download(variant: Self.repoID) { [weak self] frac in
                Task { @MainActor in self?.updateProgress(token: token, frac) }
            }
            try Task.checkCancellation()
            refreshInstalled()
            guard downloadSeq == token else { return }
            state = .idle
            task = nil
        } catch is CancellationError {
            // A download may have landed on disk just before cancellation threw; register it.
            refreshInstalled()
            guard downloadSeq == token else { return }
            state = .idle
            task = nil
        } catch {
            guard downloadSeq == token else { return }
            state = .failed(AppError.modelDownloadFailed(error.localizedDescription).localizedDescription)
            task = nil
        }
    }

    func updateProgress(token: Int, _ raw: Double) {
        // Stale callbacks from a superseded (cancelled + re-issued) generation must not write the
        // NEW task's fraction, even if the state is again `.downloading`.
        guard downloadSeq == token else { return }
        // Late callbacks after completion/cancel see a non-downloading state → ignored.
        guard case .downloading(let current) = state else { return }
        state = .downloading(Self.clampedProgress(current: current, raw: raw))
    }

    /// Monotonic, finite-clamped progress in 0...1. Pure — mirrors `ModelManager.clampedProgress`.
    nonisolated static func clampedProgress(current: Double, raw: Double) -> Double {
        let clamped = raw.isFinite ? min(max(raw, 0), 1) : 0
        return max(current, clamped)
    }

    // MARK: - Delete (reclaim disk)

    func delete() {
        try? fileManager.removeItem(at: modelDir)
        refreshInstalled()
    }
}

/// Production `ModelDownloading` for the OCR model: pulls the DeepSeek-OCR-2 repo via
/// swift-transformers' `HubApi` (in the graph transitively through DeepSeekOCR2Kit) into
/// HubApi's default base (`~/Documents/huggingface`), where the snapshot lands at
/// `models/<repoID>`. `variant` is ignored — the OCR model is a single repo.
struct OCRHubDownloader: ModelDownloading {
    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true).appendingPathComponent("Documents")
        let hub = HubApi(downloadBase: documents.appendingPathComponent("huggingface", isDirectory: true))
        return try await hub.snapshot(from: Hub.Repo(id: OCRModelManager.repoID), matching: ["*"]) { p in
            progress(p.fractionCompleted)
        }
    }
}
