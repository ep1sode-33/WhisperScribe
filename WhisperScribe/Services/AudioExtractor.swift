import Foundation
import AVFoundation

/// Decodes arbitrary audio/video containers into 16 kHz mono Float32 PCM samples.
///
/// Primary path uses AVFoundation's async `AVAssetReader`. If that fails (for a
/// reason other than a genuinely missing audio track) or yields no samples, it
/// falls back to a Homebrew `ffmpeg` invocation, reading the resulting WAV back
/// through `AVAudioFile`.
enum AudioExtractor {

    private static let targetSampleRate: Double = 16_000

    /// Decode `url` (any of .mp4/.mov/.m4a/.mp3/.wav/.aac/.caf …) to 16 kHz mono Float32 samples.
    static func decodeMono16k(url: URL) async throws -> [Float] {
        // Primary: AVFoundation. A genuinely missing audio track is authoritative;
        // cancellation must propagate. Any other failure (or an empty result) falls
        // through to a SINGLE ffmpeg attempt.
        do {
            let samples = try await decodeViaAVFoundation(url: url)
            if !samples.isEmpty {
                return samples
            }
            // Decoded OK but produced no samples — fall through to ffmpeg once.
        } catch let error as AppError where error == .noAudioTrack {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // Any other AVFoundation failure — fall through to ffmpeg once.
        }
        return try await decodeViaFFmpeg(url: url)
    }

    // MARK: - AVFoundation path

    private static func decodeViaAVFoundation(url: URL) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        guard let track = tracks.first else { throw AppError.noAudioTrack }

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw AppError.audioDecodeFailed(error.localizedDescription)
        }

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
            AVSampleRateKey: targetSampleRate,
            AVNumberOfChannelsKey: 1
        ]

        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false

        guard reader.canAdd(output) else {
            throw AppError.audioDecodeFailed(String(localized: "error.audioOutputAddFailed"))
        }
        reader.add(output)

        guard reader.startReading() else {
            throw AppError.audioDecodeFailed(reader.error?.localizedDescription ?? String(localized: "error.audioReadStartFailed"))
        }

        var samples: [Float] = []

        while let sampleBuffer = output.copyNextSampleBuffer() {
            if Task.isCancelled {
                reader.cancelReading()
                throw CancellationError()
            }
            if let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) {
                let length = CMBlockBufferGetDataLength(blockBuffer)
                let floatCount = length / MemoryLayout<Float>.size
                if floatCount > 0 {
                    var chunk = [Float](repeating: 0, count: floatCount)
                    let status = chunk.withUnsafeMutableBytes { rawBuffer -> OSStatus in
                        CMBlockBufferCopyDataBytes(
                            blockBuffer,
                            atOffset: 0,
                            dataLength: floatCount * MemoryLayout<Float>.size,
                            destination: rawBuffer.baseAddress!
                        )
                    }
                    if status == kCMBlockBufferNoErr {
                        samples.append(contentsOf: chunk)
                    }
                }
            }
            CMSampleBufferInvalidate(sampleBuffer)
        }

        guard reader.status == .completed else {
            throw AppError.audioDecodeFailed(reader.error?.localizedDescription ?? "unknown")
        }

        return samples
    }

    // MARK: - ffmpeg fallback

    private static func decodeViaFFmpeg(url: URL) async throws -> [Float] {
        let ffmpegPath = "/opt/homebrew/bin/ffmpeg"
        guard FileManager.default.fileExists(atPath: ffmpegPath) else {
            throw AppError.audioDecodeFailed(String(localized: "error.audioDecodeNoFFmpeg"))
        }

        let tmpWav = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: tmpWav) }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpegPath)
        process.arguments = [
            "-nostdin",
            "-i", url.path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_f32le",
            "-f", "wav",
            tmpWav.path
        ]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice

        // Await the process via its terminationHandler so Task cancellation can
        // terminate ffmpeg instead of blocking on waitUntilExit(). A lock-guarded
        // one-shot box hands off the continuation safely between the calling task
        // and the (background-thread) termination handler, and guarantees `resume`
        // is invoked exactly once even under races.
        final class ProcessWaiter: @unchecked Sendable {
            private let lock = NSLock()
            private var continuation: CheckedContinuation<Void, Never>?
            private var finished = false

            /// Called from terminationHandler when the process exits.
            func complete() {
                lock.lock()
                let cont = continuation
                continuation = nil
                finished = true
                lock.unlock()
                cont?.resume()
            }

            /// Register the continuation; resume immediately if already finished.
            func register(_ cont: CheckedContinuation<Void, Never>) {
                lock.lock()
                if finished {
                    lock.unlock()
                    cont.resume()
                } else {
                    continuation = cont
                    lock.unlock()
                }
            }
        }

        let waiter = ProcessWaiter()
        process.terminationHandler = { _ in waiter.complete() }

        do {
            try process.run()
        } catch {
            throw AppError.audioDecodeFailed(String.localizedStringWithFormat(NSLocalizedString("error.ffmpegLaunchFailed", comment: ""), error.localizedDescription))
        }

        await withTaskCancellationHandler {
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                waiter.register(cont)
            }
        } onCancel: {
            process.terminate()
        }

        if Task.isCancelled {
            throw CancellationError()
        }

        guard process.terminationStatus == 0,
              FileManager.default.fileExists(atPath: tmpWav.path) else {
            throw AppError.audioDecodeFailed(String.localizedStringWithFormat(NSLocalizedString("error.ffmpegDecodeFailed", comment: ""), Int(process.terminationStatus)))
        }

        return try readWavSamples(at: tmpWav)
    }

    private static func readWavSamples(at url: URL) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AppError.audioDecodeFailed(String.localizedStringWithFormat(NSLocalizedString("error.wavReadFailed", comment: ""), error.localizedDescription))
        }

        guard file.length > 0, file.length <= Int64(AVAudioFrameCount.max) else {
            if file.length <= 0 {
                throw AppError.audioDecodeFailed(String(localized: "error.decodeEmpty"))
            }
            throw AppError.audioDecodeFailed(String(localized: "error.audioTooLong"))
        }
        let frameCount = AVAudioFrameCount(file.length)

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: file.processingFormat.sampleRate,
            channels: 1,
            interleaved: false
        ), let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AppError.audioDecodeFailed(String(localized: "error.audioBufferCreateFailed"))
        }

        do {
            try file.read(into: buffer)
        } catch {
            throw AppError.audioDecodeFailed(String.localizedStringWithFormat(NSLocalizedString("error.wavDataReadFailed", comment: ""), error.localizedDescription))
        }

        guard let channelData = buffer.floatChannelData else {
            throw AppError.audioDecodeFailed(String(localized: "error.wavNoFloatChannel"))
        }

        let count = Int(buffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData[0], count: count))
    }
}
