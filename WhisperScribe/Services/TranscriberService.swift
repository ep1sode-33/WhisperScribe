import Foundation
import AVFoundation
import WhisperKit

/// Wraps a lazily-loaded WhisperKit pipeline and produces `TimedSegment`s.
actor TranscriberService {

    private var pipe: WhisperKit?
    private var loadedFolder: String?

    init() {}

    /// Idempotently load the WhisperKit pipeline for `modelFolder`. Reloads if the folder
    /// changed since the last load (so switching models in Settings takes effect next job).
    func prepare(modelFolder: String) async throws {
        if pipe != nil, loadedFolder == modelFolder { return }

        guard FileManager.default.fileExists(atPath: modelFolder) else {
            throw AppError.modelNotInstalled
        }

        // Drop any previously-loaded pipeline before loading a different folder.
        pipe = nil
        loadedFolder = nil

        let compute = ModelComputeOptions(
            melCompute: .cpuAndGPU,
            audioEncoderCompute: .cpuAndGPU,
            textDecoderCompute: .cpuAndNeuralEngine
        )

        let config = WhisperKitConfig(
            modelFolder: modelFolder,
            computeOptions: compute,
            verbose: true,
            logLevel: .info,
            prewarm: true,
            load: true,
            download: false
        )

        do {
            pipe = try await WhisperKit(config)
            loadedFolder = modelFolder
        } catch {
            throw AppError.transcriptionFailed(String(describing: error))
        }
    }

    /// Transcribe 16 kHz mono Float32 samples into globally-indexed timed segments.
    func transcribe(
        samples: [Float],
        language: String?,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> [TimedSegment] {
        guard let pipe else {
            throw AppError.modelNotInstalled
        }

        if Task.isCancelled { throw CancellationError() }

        var o = DecodingOptions()
        o.task = .transcribe
        if let language {
            o.language = language
            o.detectLanguage = false
        } else {
            o.detectLanguage = true
        }
        o.temperature = 0.0
        o.wordTimestamps = false
        o.chunkingStrategy = .vad

        progress(0)

        // WhisperKit drives a Foundation `Progress` object monotonically while decoding
        // (WhisperKit.swift:45). In the `.vad` path it is `completed chunks / total chunks`
        // (WhisperKit.swift:919 + 1008-1009); in the single-window path it is
        // `processed samples / total samples` (TranscribeTask.swift:111/280/285). This is the
        // exact signal the official WhisperAX demo binds its progress bar to.
        //
        // We do NOT use `segmentCallback` for progress: under `.vad` the segment `start/end`
        // delivered to the callback are block-relative (each chunk restarts at 0), so they
        // cannot be normalized against total duration — that was the cause of the stuck-at-0%
        // bar. The absolute offsets are only applied to the final results after transcription.
        //
        // The progress object is swapped for a fresh `Progress()` once it finishes or is
        // cancelled (WhisperKit.swift:1036/1044), so we must capture *this* instance up front
        // and observe it via KVO. We retain a strong reference to the captured object, so the
        // later swap does not affect our observation.
        let liveProgress = pipe.progress
        let monotonic = MonotonicFraction()
        let token = liveProgress.observe(\.fractionCompleted, options: [.initial, .new]) { obj, _ in
            // KVO fires on a background thread; the caller's `progress` closure is responsible
            // for hopping to the main actor for UI updates.
            progress(monotonic.update(obj.fractionCompleted))
        }

        let results: [TranscriptionResult]
        do {
            results = try await pipe.transcribe(
                audioArray: samples,
                decodeOptions: o,
                callback: { _ in !Task.isCancelled }            // cancellation only
            )
        } catch {
            token.invalidate()
            if Task.isCancelled { throw CancellationError() }
            throw AppError.transcriptionFailed(String(describing: error))
        }
        token.invalidate()

        if Task.isCancelled { throw CancellationError() }

        // Flatten all segments in order, assigning a fresh global index.
        let flatSegments = results.flatMap { $0.segments }
        let timed: [TimedSegment] = flatSegments.enumerated().map { (i, seg) in
            TimedSegment(
                index: i,
                start: TimeInterval(seg.start),
                end: TimeInterval(seg.end),
                text: seg.text.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }

        progress(monotonic.update(1))
        return timed
    }
}

/// Thread-safe, monotonically non-decreasing progress fraction clamped to 0...1.
/// WhisperKit's `Progress.fractionCompleted` KVO callbacks may arrive on concurrent
/// worker threads; this guarantees the bar never jumps backwards.
private final class MonotonicFraction: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Double = 0
    func update(_ raw: Double) -> Double {
        // Progress with totalUnitCount == 0 can momentarily yield a non-finite value.
        let clamped = raw.isFinite ? min(max(raw, 0), 1) : 0
        lock.lock(); defer { lock.unlock() }
        if clamped > value { value = clamped }
        return value
    }
}
