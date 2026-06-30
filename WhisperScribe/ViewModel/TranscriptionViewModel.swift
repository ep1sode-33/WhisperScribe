import Foundation
import AppKit

@MainActor
final class TranscriptionViewModel: ObservableObject {
    @Published var state: JobState = .idle

    private let settings: SettingsStore
    private let transcriber: TranscriberService
    private let cleaner: LLMCleaner
    private let modelManager: ModelManager

    private var task: Task<Void, Never>?
    private var lastOutputs: (srt: URL, txt: URL)?

    private var jobToken = 0
    private var lastTranscribeFraction = 0.0
    private var lastCleanFraction = 0.0

    init(settings: SettingsStore, transcriber: TranscriberService, cleaner: LLMCleaner, modelManager: ModelManager) {
        self.settings = settings
        self.transcriber = transcriber
        self.cleaner = cleaner
        self.modelManager = modelManager
    }

    func start(url: URL) {
        // Ignore new requests while busy. Mark busy SYNCHRONOUSLY (before the Task is
        // scheduled) so two rapid calls cannot both pass the guard.
        guard !state.isBusy else { return }
        jobToken &+= 1
        let token = jobToken
        lastTranscribeFraction = 0
        lastCleanFraction = 0
        state = .loadingModel

        task = Task { [weak self] in
            guard let self else { return }
            do {
                // 1. Load model (resolve the selected model's folder; gate on readiness)
                let model = self.modelManager.selectedModel
                guard self.modelManager.isReady else { throw AppError.modelNotInstalled }
                try await self.transcriber.prepare(modelFolder: self.modelManager.modelFolderPath(model))
                try Task.checkCancellation()

                // 2. Extract audio
                self.state = .extractingAudio
                let samples = try await AudioExtractor.decodeMono16k(url: url)
                try Task.checkCancellation()

                // 3. Transcribe
                self.state = .transcribing(progress: 0)
                let language = self.settings.language.isEmpty ? nil : self.settings.language
                let segs = try await self.transcriber.transcribe(
                    samples: samples,
                    language: language,
                    progress: { p in
                        Task { @MainActor in
                            guard self.jobToken == token, case .transcribing = self.state, p >= self.lastTranscribeFraction else { return }
                            self.lastTranscribeFraction = p
                            self.state = .transcribing(progress: p)
                        }
                    }
                )
                try Task.checkCancellation()

                // 4. Cleanup (optional LLM)
                let level = self.settings.cleanupLevel
                if level.usesLLM {
                    self.state = .cleaning(progress: 0, note: "")
                }
                let outcome = try await self.cleaner.process(
                    segments: segs,
                    level: level,
                    language: language,
                    config: self.settings.llmConfig,
                    progress: { p, note in
                        Task { @MainActor in
                            guard self.jobToken == token, case .cleaning = self.state, p >= self.lastCleanFraction else { return }
                            self.lastCleanFraction = p
                            self.state = .cleaning(progress: p, note: note)
                        }
                    }
                )
                try Task.checkCancellation()

                // 5. Write outputs
                let urls = try SubtitleWriter.write(
                    segments: outcome.segments,
                    txt: outcome.txt,
                    source: url,
                    outputDir: self.settings.resolvedOutputDir,
                    overwrite: self.settings.overwritePolicy
                )

                // 6. Done
                self.lastOutputs = urls
                self.state = .done(srt: urls.srt, txt: urls.txt, warnings: outcome.warnings)
            } catch is CancellationError {
                self.state = .idle
            } catch let e as AppError {
                self.state = .error(e)
            } catch {
                self.state = .error(.transcriptionFailed(String(describing: error)))
            }
        }
    }

    func cancel() {
        task?.cancel()
    }

    func reset() {
        guard !state.isBusy else { return }
        state = .idle
    }

    func revealInFinder() {
        guard let urls = lastOutputs else { return }
        NSWorkspace.shared.activateFileViewerSelecting([urls.srt, urls.txt])
    }
}
