import Foundation
import AppKit

@MainActor
final class TranscriptionViewModel: ObservableObject {
    @Published var state: JobState = .idle
    /// Progress across a multi-file batch; `nil` for single-file / non-batch jobs.
    @Published var batch: BatchProgress?
    /// Source-file count of the batch that produced the current `.done` outputs. `batch` is
    /// cleared on every terminal state, so this survives to let the done view render the
    /// `done.batchSummary` ("N files → M outputs") line. Reset on a new job / `reset()`.
    @Published var lastBatchFileCount: Int?

    private let settings: SettingsStore
    private let transcriber: Transcribing
    private let cleaner: LLMCleaner
    private let modelManager: ModelManager
    private let ocr: OCRProviding
    private let ocrModels: OCRModelManager
    private let merger: MergeService

    private var task: Task<Void, Never>?
    private var lastOutputs: [URL] = []

    private var jobToken = 0
    private var lastTranscribeFraction = 0.0
    private var lastCleanFraction = 0.0

    init(settings: SettingsStore,
         transcriber: Transcribing,
         cleaner: LLMCleaner,
         modelManager: ModelManager,
         ocr: OCRProviding,
         ocrModels: OCRModelManager,
         merger: MergeService) {
        self.settings = settings
        self.transcriber = transcriber
        self.cleaner = cleaner
        self.modelManager = modelManager
        self.ocr = ocr
        self.ocrModels = ocrModels
        self.merger = merger
    }

    // MARK: - Entry points

    /// Single-file entry point retained for the drag/drop and file-picker call sites;
    /// forwards to the batch pipeline.
    func start(url: URL) { start(urls: [url]) }

    /// Classify the dropped URLs into a homogeneous audio or image batch and route to the
    /// matching orchestration. A mixed / unsupported / empty batch is an immediate error.
    func start(urls: [URL]) {
        // Ignore new requests while busy. Mark busy SYNCHRONOUSLY (before the Task is
        // scheduled) so two rapid calls cannot both pass the guard.
        guard !state.isBusy else { return }

        let kind: BatchKind
        do {
            kind = try BatchClassifier.classify(urls)
        } catch {
            state = .error((error as? AppError) ?? .cancelled)
            return
        }

        jobToken &+= 1
        let token = jobToken
        batch = nil
        lastBatchFileCount = nil
        state = .loadingModel

        task = Task { [weak self] in
            guard let self else { return }
            switch kind {
            case .audio(let files):  await self.runAudioBatch(files, token: token)
            case .images(let files): await self.runImageBatch(files, token: token)
            }
        }
    }

    func cancel() {
        task?.cancel()
    }

    func reset() {
        guard !state.isBusy else { return }
        state = .idle
        batch = nil
        lastBatchFileCount = nil
    }

    func revealInFinder() {
        guard !lastOutputs.isEmpty else { return }
        NSWorkspace.shared.activateFileViewerSelecting(lastOutputs)
    }

    // MARK: - Audio batch

    /// Runs each audio file through the existing decode→transcribe→clean pipeline. A
    /// single-file batch keeps today's behavior exactly (srt + txt, no merge). A
    /// multi-file batch writes per-file srt (timeline artifact) and one merged .txt
    /// (which replaces the per-file txt). A single file's failure is skipped with a
    /// warning; only an all-failed batch is an error.
    private func runAudioBatch(_ files: [URL], token: Int) async {
        await ocr.unload()   // memory exclusivity: release the OCR model before loading Whisper
        let isBatch = files.count > 1
        do {
            // Load the Whisper model once for the whole batch.
            let model = modelManager.selectedModel
            guard modelManager.isReady else { throw AppError.modelNotInstalled }
            try await transcriber.prepare(modelFolder: modelManager.modelFolderPath(model))
            try Task.checkCancellation()

            var srtURLs: [URL] = []
            var singleTxt: URL?
            var parts: [(name: String, text: String)] = []
            var warnings: [String] = []
            var lastError: AppError?

            for (i, url) in files.enumerated() {
                guard jobToken == token else { return }
                try Task.checkCancellation()
                batch = isBatch ? BatchProgress(index: i + 1, count: files.count, fileName: url.lastPathComponent) : nil
                do {
                    let outcome = try await transcribeOneAudio(url, token: token)
                    if isBatch {
                        srtURLs.append(try writeSRTOnly(segments: outcome.segments, source: url))
                    } else {
                        let urls = try SubtitleWriter.write(
                            segments: outcome.segments, txt: outcome.txt, source: url,
                            outputDir: settings.resolvedOutputDir, overwrite: settings.overwritePolicy)
                        srtURLs.append(urls.srt)
                        singleTxt = urls.txt
                    }
                    parts.append((name: url.lastPathComponent, text: outcome.txt))
                    warnings.append(contentsOf: outcome.warnings)
                } catch is CancellationError {
                    throw CancellationError()
                } catch let e as AppError {
                    lastError = e
                    warnings.append(skipWarning(url, e))
                } catch {
                    let e = AppError.transcriptionFailed(String(describing: error))
                    lastError = e
                    warnings.append(skipWarning(url, e))
                }
            }

            guard jobToken == token else { return }
            if parts.isEmpty {
                // Every file failed — surface an error rather than an empty success.
                state = .error(lastError ?? .transcriptionFailed(""))
                batch = nil
                return
            }

            var outputs = srtURLs
            if isBatch {
                let language = settings.language.isEmpty ? nil : settings.language
                state = .merging(progress: 0, note: "")
                let merged = await merger.merge(
                    parts: parts, language: language, config: settings.llmConfig,
                    progress: mergeProgress(token: token))
                try Task.checkCancellation()
                let mergedURL = try SubtitleWriter.writeMergedText(
                    merged.text, firstSource: files[0],
                    outputDir: settings.resolvedOutputDir, overwrite: settings.overwritePolicy)
                outputs.append(mergedURL)
                warnings.append(contentsOf: merged.warnings)
            } else if let singleTxt {
                outputs.append(singleTxt)
            }

            guard jobToken == token else { return }
            batch = nil
            lastOutputs = outputs
            lastBatchFileCount = files.count
            state = .done(outputs: outputs, warnings: warnings)
        } catch is CancellationError {
            state = .idle
            batch = nil
        } catch let e as AppError {
            state = .error(e)
            batch = nil
        } catch {
            state = .error(.transcriptionFailed(String(describing: error)))
            batch = nil
        }
    }

    /// The per-file decode→transcribe→clean pipeline (no writing). Reused by single-file
    /// and multi-file audio batches so the two paths stay in lock-step.
    private func transcribeOneAudio(_ url: URL, token: Int) async throws -> CleanupOutcome {
        state = .extractingAudio
        let samples = try await AudioExtractor.decodeMono16k(url: url)
        try Task.checkCancellation()

        state = .transcribing(progress: 0)
        lastTranscribeFraction = 0
        let language = settings.language.isEmpty ? nil : settings.language
        let segs = try await transcriber.transcribe(
            samples: samples,
            language: language,
            progress: { [weak self] p in
                Task { @MainActor in
                    guard let self, self.jobToken == token,
                          case .transcribing = self.state, p >= self.lastTranscribeFraction else { return }
                    self.lastTranscribeFraction = p
                    self.state = .transcribing(progress: p)
                }
            })
        try Task.checkCancellation()

        let level = settings.cleanupLevel
        if level.usesLLM {
            state = .cleaning(progress: 0, note: "")
            lastCleanFraction = 0
        }
        let outcome = try await cleaner.process(
            segments: segs, level: level, language: language, config: settings.llmConfig,
            progress: { [weak self] p, note in
                Task { @MainActor in
                    guard let self, self.jobToken == token,
                          case .cleaning = self.state, p >= self.lastCleanFraction else { return }
                    self.lastCleanFraction = p
                    self.state = .cleaning(progress: p, note: note)
                }
            })
        try Task.checkCancellation()
        return outcome
    }

    // MARK: - Image batch

    /// OCRs each image and merges the per-image transcripts into a single .txt. A single
    /// image's failure is skipped with a warning; an all-failed batch is an error.
    private func runImageBatch(_ files: [URL], token: Int) async {
        // Precondition: the OCR checkpoint must be on disk before any work.
        guard ocrModels.isReady else {
            state = .error(.ocrModelMissing)
            batch = nil
            return
        }
        // Memory exclusivity: drop the Whisper pipeline before loading the OCR model.
        await transcriber.unload()
        do {
            state = .loadingModel
            try await ocr.prepare(modelDir: ocrModels.modelDir, progress: { _ in })
            try Task.checkCancellation()

            var parts: [(name: String, text: String)] = []
            var warnings: [String] = []
            var lastError: AppError?

            for (i, url) in files.enumerated() {
                guard jobToken == token else { return }
                try Task.checkCancellation()
                batch = BatchProgress(index: i + 1, count: files.count, fileName: url.lastPathComponent)
                state = .recognizing(progress: Double(i) / Double(files.count))
                do {
                    let text = try await ocr.recognize(imageAt: url, onChunk: { _ in })
                    // Kit semantics: mid-generation cancellation returns PARTIAL text WITHOUT
                    // throwing, so probe cancellation explicitly after every recognize.
                    try Task.checkCancellation()
                    parts.append((name: url.lastPathComponent, text: text))
                } catch is CancellationError {
                    throw CancellationError()
                } catch let e as AppError {
                    lastError = e
                    warnings.append(skipWarning(url, e))
                } catch {
                    let e = AppError.ocrFailed(url.lastPathComponent)
                    lastError = e
                    warnings.append(skipWarning(url, e))
                }
            }

            guard jobToken == token else { return }
            if parts.isEmpty {
                state = .error(lastError ?? .ocrFailed(""))
                batch = nil
                return
            }

            let language = settings.language.isEmpty ? nil : settings.language
            state = .merging(progress: 0, note: "")
            let merged = await merger.merge(
                parts: parts, language: language, config: settings.llmConfig,
                progress: mergeProgress(token: token))
            try Task.checkCancellation()
            let mergedURL = try SubtitleWriter.writeMergedText(
                merged.text, firstSource: files[0],
                outputDir: settings.resolvedOutputDir, overwrite: settings.overwritePolicy)
            warnings.append(contentsOf: merged.warnings)

            guard jobToken == token else { return }
            batch = nil
            lastOutputs = [mergedURL]
            lastBatchFileCount = files.count
            state = .done(outputs: [mergedURL], warnings: warnings)
        } catch is CancellationError {
            state = .idle
            batch = nil
        } catch let e as AppError {
            state = .error(e)
            batch = nil
        } catch {
            state = .error(.ocrFailed(String(describing: error)))
            batch = nil
        }
    }

    // MARK: - Shared helpers

    /// Progress sink for the merge phase, gated on the current job so a stale merge cannot
    /// write state after a newer job started.
    private func mergeProgress(token: Int) -> @Sendable (Double, String) -> Void {
        { [weak self] p, note in
            Task { @MainActor in
                guard let self, self.jobToken == token, case .merging = self.state else { return }
                self.state = .merging(progress: p, note: note)
            }
        }
    }

    /// Writes just the timeline `.srt` for a file in a multi-file batch (the merged `.txt`
    /// replaces the per-file prose). Mirrors `SubtitleWriter.write`'s directory-creation and
    /// error-wrapping, minus the `.txt`.
    private func writeSRTOnly(segments: [TimedSegment], source: URL) throws -> URL {
        let urls = FileNaming.outputURLs(source: source, outputDir: settings.resolvedOutputDir, overwrite: settings.overwritePolicy)
        let srt = SRTFormatter.srt(from: segments)
        try? FileManager.default.createDirectory(at: urls.srt.deletingLastPathComponent(), withIntermediateDirectories: true)
        do {
            try srt.write(to: urls.srt, atomically: true, encoding: .utf8)
        } catch {
            throw AppError.writeFailed(String(describing: error))
        }
        return urls.srt
    }

    /// Skip-and-warn text for a single failed file within a batch.
    private func skipWarning(_ url: URL, _ error: AppError) -> String {
        String.localizedStringWithFormat(
            NSLocalizedString("cleanup.warning.fileSkipped", comment: ""),
            url.lastPathComponent, error.localizedDescription)
    }
}
