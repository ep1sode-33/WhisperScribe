import Testing
import Foundation
import AVFoundation
import Combine
@testable import WhisperScribe

// MARK: - Fakes (own to this file; FakeOCR is reused from OCRServiceTests, internal)

/// Minimal `Transcribing` fake: returns fixed segments, ignores the samples.
/// Records `unload()` so a batch can assert model-memory exclusivity.
private final class FakeTranscriber: Transcribing, @unchecked Sendable {
    let segments: [TimedSegment]
    private let lock = NSLock()
    private var _unloadCalled = false
    var unloadCalled: Bool { lock.lock(); defer { lock.unlock() }; return _unloadCalled }
    init(segments: [TimedSegment] = []) { self.segments = segments }
    func prepare(modelFolder: String) async throws {}
    func transcribe(samples: [Float], language: String?,
                    progress: @escaping @Sendable (Double) -> Void) async throws -> [TimedSegment] {
        progress(1.0)
        return segments
    }
    func unload() async { lock.lock(); _unloadCalled = true; lock.unlock() }
}

/// Own `ChatStreaming` fake for MergeService (MergeServiceTests' FakeChat is private).
/// Records every assembled `user` payload and returns a fixed reply.
private final class RecordingChat: ChatStreaming, @unchecked Sendable {
    private let lock = NSLock()
    private var _users: [String] = []
    let reply: String
    init(reply: String) { self.reply = reply }
    var users: [String] { lock.lock(); defer { lock.unlock() }; return _users }
    var callCount: Int { lock.lock(); defer { lock.unlock() }; return _users.count }
    func streamChat(config: LLMConfig, system: String, user: String,
                    onDelta: @escaping @Sendable (String) -> Void) async throws -> String {
        lock.lock(); _users.append(user); lock.unlock()
        onDelta(reply)
        return reply
    }
}

/// Suspending `ChatStreaming` fake for MergeService: `streamChat` parks on a `Gate` so a
/// test can cancel the job while it is stalled mid-merge, then resume it deterministically.
private final class GatedChat: ChatStreaming, @unchecked Sendable {
    let gate: Gate
    let reply: String
    init(gate: Gate, reply: String) { self.gate = gate; self.reply = reply }
    func streamChat(config: LLMConfig, system: String, user: String,
                    onDelta: @escaping @Sendable (String) -> Void) async throws -> String {
        await gate.wait()
        onDelta(reply)
        return reply
    }
}

/// FIFO gate — copied from ModelManagerTests' pattern (that one is private). Lets a test
/// park an in-flight `recognize` and resume it deterministically to reproduce a cancel.
private final class Gate: @unchecked Sendable {
    private let lock = NSLock()
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        await withCheckedContinuation { cont in
            lock.lock(); waiters.append(cont); lock.unlock()
        }
    }
    var waiterCount: Int { lock.lock(); defer { lock.unlock() }; return waiters.count }
    func resumeNext() {
        lock.lock()
        let next = waiters.isEmpty ? nil : waiters.removeFirst()
        lock.unlock()
        next?.resume()
    }
}

/// Suspending OCR fake: `recognize` parks on a `Gate`, then — per DeepSeek-OCR-2 kit
/// semantics — returns PARTIAL text WITHOUT throwing on cancellation.
private final class GatedOCR: OCRProviding, @unchecked Sendable {
    let gate: Gate
    var prepared = false
    init(gate: Gate) { self.gate = gate }
    func prepare(modelDir: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        prepared = true; progress(1.0)
    }
    func recognize(imageAt url: URL, onChunk: @escaping @Sendable (String) -> Void) async throws -> String {
        await gate.wait()
        return ""   // partial-without-throw on cancel; caller must probe Task.isCancelled
    }
    func unload() async { prepared = false }
}

/// OCR fake that parks inside `prepare` on a `Gate`, giving a test a deterministic
/// mid-prepare observation point to assert `isLoadingOCRModel == true` while the OCR model
/// is loading. `recognize` returns per-file text immediately (no parking) so the batch can
/// run to completion once prepare is resumed.
private final class GatedPrepareOCR: OCRProviding, @unchecked Sendable {
    let gate: Gate
    var results: [String: String] = [:]
    var prepared = false
    init(gate: Gate) { self.gate = gate }
    func prepare(modelDir: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        await gate.wait()
        prepared = true; progress(1.0)
    }
    func recognize(imageAt url: URL, onChunk: @escaping @Sendable (String) -> Void) async throws -> String {
        let text = results[url.lastPathComponent, default: "text-of-\(url.lastPathComponent)"]
        onChunk(text)
        return text
    }
    func unload() async { prepared = false }
}

private struct NoopDownloader: ModelDownloading {
    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        URL(fileURLWithPath: "/dev/null")
    }
}

// MARK: - Helpers

private func tempDir() throws -> URL {
    let d = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
    return d
}

private func seg(_ text: String, _ i: Int = 0) -> TimedSegment {
    TimedSegment(index: i, start: Double(i), end: Double(i) + 1, text: text)
}

/// Writes a tiny valid 16 kHz mono WAV so `AudioExtractor.decodeMono16k` yields non-empty samples.
private func makeWav(at url: URL) throws {
    let format = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1)!
    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    let frames: AVAudioFrameCount = 1_600
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
    buffer.frameLength = frames
    try file.write(from: buffer)   // silence decodes fine
}

/// OCRModelManager whose on-disk checkpoint LOOKS installed (config/tokenizer/weights present).
@MainActor
private func installedOCRModels() throws -> OCRModelManager {
    let base = try tempDir()
    let dir = base.appendingPathComponent(OCRModelManager.repoID, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    for f in ["config.json", "tokenizer.json", "model.safetensors"] {
        FileManager.default.createFile(atPath: dir.appendingPathComponent(f).path, contents: Data())
    }
    return OCRModelManager(baseDir: base)   // init's refreshInstalled() -> installed == true
}

@MainActor
private func missingOCRModels() throws -> OCRModelManager {
    OCRModelManager(baseDir: try tempDir())   // empty dir -> not installed
}

/// ModelManager whose selected (default) Whisper model appears installed -> isReady.
@MainActor
private func readyModelManager() throws -> ModelManager {
    let base = try tempDir()
    try FileManager.default.createDirectory(
        at: base.appendingPathComponent(WhisperModel.default.variant).appendingPathComponent("Model.mlmodelc"),
        withIntermediateDirectories: true)
    let defaults = UserDefaults(suiteName: "test.\(UUID().uuidString)")!
    return ModelManager(downloader: NoopDownloader(), defaults: defaults, fileManager: .default, baseDir: base)
}

@MainActor
private func freshModelManager() throws -> ModelManager {
    let defaults = UserDefaults(suiteName: "test.\(UUID().uuidString)")!
    return ModelManager(downloader: NoopDownloader(), defaults: defaults, fileManager: .default, baseDir: try tempDir())
}

@MainActor
private func makeVM(settings: SettingsStore,
                    transcriber: Transcribing = FakeTranscriber(),
                    ocr: OCRProviding,
                    ocrModels: OCRModelManager,
                    merger: MergeService,
                    modelManager: ModelManager) -> TranscriptionViewModel {
    TranscriptionViewModel(settings: settings, transcriber: transcriber, cleaner: LLMCleaner(),
                           modelManager: modelManager, ocr: ocr, ocrModels: ocrModels, merger: merger)
}

/// Cooperatively yield the main actor until `cond` holds (bounded so a stuck job fails, not hangs).
@MainActor
private func waitUntil(_ cond: () -> Bool) async {
    for _ in 0..<200_000 {
        if cond() { return }
        await Task.yield()
    }
}

@MainActor
private func configured(_ s: SettingsStore) {
    s.llmBaseURL = "https://api.example.com/v1"
    s.llmModel = "m"
}

/// The test bundle is app-hosted, so `SettingsStore`'s `@AppStorage` writes land in the
/// app's real `UserDefaults` domain. This snapshots the keys these tests mutate and returns
/// a `defer`-able restore so a run never pollutes the developer's actual preferences.
@MainActor
private func isolatedSettings() -> (SettingsStore, () -> Void) {
    let defaults = UserDefaults.standard
    let keys = ["llmBaseURL", "llmModel", "cleanupLevel"]
    let saved = keys.map { ($0, defaults.object(forKey: $0)) }
    let restore = {
        for (key, value) in saved {
            if let value { defaults.set(value, forKey: key) } else { defaults.removeObject(forKey: key) }
        }
    }
    return (SettingsStore(), restore)
}

// MARK: - Tests

@Suite(.serialized)
struct BatchPipelineTests {

    /// 1. Three images -> one merged .txt; batch progress reaches (3,3).
    @Test @MainActor func imageBatchProducesMergedTxt() async throws {
        let dir = try tempDir()
        let imgs = ["1.png", "2.png", "3.png"].map { dir.appendingPathComponent($0) }
        let (settings, restore) = isolatedSettings(); defer { restore() }
        configured(settings)
        let ocr = FakeOCR(); ocr.results = ["1.png": "A", "2.png": "B", "3.png": "C"]
        let chat = RecordingChat(reply: "ABC-merged")
        let vm = makeVM(settings: settings, ocr: ocr, ocrModels: try installedOCRModels(),
                        merger: MergeService(chat: chat), modelManager: try freshModelManager())

        var recorded: [BatchProgress] = []
        let sub = vm.$batch.compactMap { $0 }.sink { recorded.append($0) }
        vm.start(urls: imgs)
        await waitUntil { !vm.state.isBusy }
        sub.cancel()

        guard case .done(let outputs, let warnings) = vm.state else {
            Issue.record("expected .done, got \(vm.state)"); return
        }
        #expect(outputs.count == 1)
        #expect(try String(contentsOf: outputs[0], encoding: .utf8) == "ABC-merged")
        #expect(warnings.isEmpty)
        #expect(recorded.contains { $0.index == 3 && $0.count == 3 })
    }

    /// 2. Middle image fails -> skipped with a warning; merge input has only 1 & 3.
    @Test @MainActor func imageBatchSkipsFailedFile() async throws {
        let dir = try tempDir()
        let imgs = ["1.png", "2.png", "3.png"].map { dir.appendingPathComponent($0) }
        let (settings, restore) = isolatedSettings(); defer { restore() }
        configured(settings)
        let ocr = FakeOCR()
        ocr.results = ["1.png": "A", "2.png": "B", "3.png": "C"]
        ocr.failOn = ["2.png"]
        let chat = RecordingChat(reply: "MERGED")
        let vm = makeVM(settings: settings, ocr: ocr, ocrModels: try installedOCRModels(),
                        merger: MergeService(chat: chat), modelManager: try freshModelManager())

        vm.start(urls: imgs)
        await waitUntil { !vm.state.isBusy }

        guard case .done(let outputs, let warnings) = vm.state else {
            Issue.record("expected .done, got \(vm.state)"); return
        }
        #expect(outputs.count == 1)
        #expect(warnings.contains { $0.contains("2.png") })
        #expect(chat.callCount == 1)
        let user = chat.users.first ?? ""
        #expect(user.contains("1.png") && user.contains("3.png"))
        #expect(!user.contains("2.png"))
    }

    /// 3. Every image fails -> error.
    @Test @MainActor func imageBatchAllFailedIsError() async throws {
        let dir = try tempDir()
        let imgs = ["1.png", "2.png", "3.png"].map { dir.appendingPathComponent($0) }
        let (settings, restore) = isolatedSettings(); defer { restore() }
        configured(settings)
        let ocr = FakeOCR(); ocr.failOn = ["1.png", "2.png", "3.png"]
        let chat = RecordingChat(reply: "X")
        let vm = makeVM(settings: settings, ocr: ocr, ocrModels: try installedOCRModels(),
                        merger: MergeService(chat: chat), modelManager: try freshModelManager())

        vm.start(urls: imgs)
        await waitUntil { !vm.state.isBusy }

        guard case .error = vm.state else { Issue.record("expected .error, got \(vm.state)"); return }
        #expect(chat.callCount == 0)
    }

    /// 4. OCR model not installed -> .error(.ocrModelMissing) BEFORE any OCR work.
    @Test @MainActor func imageBatchRequiresOCRModel() async throws {
        let dir = try tempDir()
        let imgs = ["1.png", "2.png"].map { dir.appendingPathComponent($0) }
        let (settings, restore) = isolatedSettings(); defer { restore() }
        let ocr = FakeOCR()
        let vm = makeVM(settings: settings, ocr: ocr, ocrModels: try missingOCRModels(),
                        merger: MergeService(chat: RecordingChat(reply: "X")), modelManager: try freshModelManager())

        vm.start(urls: imgs)
        await waitUntil { !vm.state.isBusy }

        #expect(vm.state == .error(.ocrModelMissing))
        #expect(ocr.prepared == false)
    }

    /// 5. Mixed audio + image -> immediate .error(.mixedBatch) (synchronous).
    @Test @MainActor func mixedBatchIsImmediateError() async throws {
        let (settings, restore) = isolatedSettings(); defer { restore() }
        let vm = makeVM(settings: settings, ocr: FakeOCR(), ocrModels: try missingOCRModels(),
                        merger: MergeService(chat: RecordingChat(reply: "X")), modelManager: try freshModelManager())
        vm.start(urls: [URL(fileURLWithPath: "/tmp/a.mp3"), URL(fileURLWithPath: "/tmp/b.png")])
        #expect(vm.state == .error(.mixedBatch))
    }

    /// 6. Cancel while OCR is parked mid-batch -> state returns to .idle.
    @Test @MainActor func cancelMidBatchReturnsIdle() async throws {
        let dir = try tempDir()
        let imgs = ["1.png", "2.png"].map { dir.appendingPathComponent($0) }
        let (settings, restore) = isolatedSettings(); defer { restore() }
        configured(settings)
        let gate = Gate()
        let ocr = GatedOCR(gate: gate)
        let vm = makeVM(settings: settings, ocr: ocr, ocrModels: try installedOCRModels(),
                        merger: MergeService(chat: RecordingChat(reply: "X")), modelManager: try freshModelManager())

        vm.start(urls: imgs)
        await waitUntil { gate.waiterCount == 1 }   // parked inside recognize of the first image
        vm.cancel()
        gate.resumeNext()                            // recognize returns partial, no throw
        await waitUntil { vm.state == .idle }
        #expect(vm.state == .idle)
    }

    /// 7. Single audio via start(urls:) -> today's srt + txt pair; merge is skipped.
    @Test @MainActor func singleAudioStillWorks() async throws {
        let dir = try tempDir()
        let audio = dir.appendingPathComponent("clip.wav"); try makeWav(at: audio)
        let (settings, restore) = isolatedSettings(); defer { restore() }
        settings.cleanupLevel = .raw
        let chat = RecordingChat(reply: "SHOULD-NOT-BE-CALLED")
        let vm = makeVM(settings: settings, transcriber: FakeTranscriber(segments: [seg("hello")]),
                        ocr: FakeOCR(), ocrModels: try missingOCRModels(),
                        merger: MergeService(chat: chat), modelManager: try readyModelManager())

        vm.start(urls: [audio])
        await waitUntil { !vm.state.isBusy }

        guard case .done(let outputs, _) = vm.state else { Issue.record("expected .done, got \(vm.state)"); return }
        #expect(outputs.count == 2)
        #expect(outputs.filter { $0.pathExtension == "srt" }.count == 1)
        #expect(outputs.filter { $0.pathExtension == "txt" }.count == 1)
        #expect(chat.callCount == 0)   // single-file batch does NOT merge
    }

    /// 8. Two audio -> 2 srt + 1 merged txt; merged content comes from the (fake) LLM.
    @Test @MainActor func multiAudioProducesPerFileSrtPlusMerged() async throws {
        let dir = try tempDir()
        let a = dir.appendingPathComponent("1.wav"); try makeWav(at: a)
        let b = dir.appendingPathComponent("2.wav"); try makeWav(at: b)
        let (settings, restore) = isolatedSettings(); defer { restore() }
        settings.cleanupLevel = .raw; configured(settings)
        let chat = RecordingChat(reply: "MERGED-AUDIO")
        let vm = makeVM(settings: settings, transcriber: FakeTranscriber(segments: [seg("hello")]),
                        ocr: FakeOCR(), ocrModels: try missingOCRModels(),
                        merger: MergeService(chat: chat), modelManager: try readyModelManager())

        vm.start(urls: [a, b])
        await waitUntil { !vm.state.isBusy }

        guard case .done(let outputs, _) = vm.state else { Issue.record("expected .done, got \(vm.state)"); return }
        #expect(outputs.count == 3)
        #expect(outputs.filter { $0.pathExtension == "srt" }.count == 2)
        let txts = outputs.filter { $0.pathExtension == "txt" }
        #expect(txts.count == 1)
        #expect(try String(contentsOf: txts[0], encoding: .utf8) == "MERGED-AUDIO")
        #expect(chat.callCount == 1)
    }

    /// 9. Model-memory exclusivity: an image batch must release the Whisper model before
    /// loading the OCR model, so the transcriber's `unload()` is invoked.
    @Test @MainActor func imageBatchUnloadsTranscriber() async throws {
        let dir = try tempDir()
        let imgs = ["1.png", "2.png"].map { dir.appendingPathComponent($0) }
        let (settings, restore) = isolatedSettings(); defer { restore() }
        configured(settings)
        let transcriber = FakeTranscriber()
        let ocr = FakeOCR(); ocr.results = ["1.png": "A", "2.png": "B"]
        let vm = makeVM(settings: settings, transcriber: transcriber, ocr: ocr,
                        ocrModels: try installedOCRModels(),
                        merger: MergeService(chat: RecordingChat(reply: "M")),
                        modelManager: try freshModelManager())

        vm.start(urls: imgs)
        await waitUntil { !vm.state.isBusy }

        #expect(transcriber.unloadCalled)
    }

    /// 10. Model-memory exclusivity: an audio batch must release the OCR model before
    /// loading Whisper, so the OCR provider's `unload()` is invoked.
    @Test @MainActor func audioBatchUnloadsOCR() async throws {
        let dir = try tempDir()
        let a = dir.appendingPathComponent("1.wav"); try makeWav(at: a)
        let b = dir.appendingPathComponent("2.wav"); try makeWav(at: b)
        let (settings, restore) = isolatedSettings(); defer { restore() }
        settings.cleanupLevel = .raw; configured(settings)
        let ocr = FakeOCR()
        let vm = makeVM(settings: settings, transcriber: FakeTranscriber(segments: [seg("hello")]),
                        ocr: ocr, ocrModels: try missingOCRModels(),
                        merger: MergeService(chat: RecordingChat(reply: "M")),
                        modelManager: try readyModelManager())

        vm.start(urls: [a, b])
        await waitUntil { !vm.state.isBusy }

        #expect(ocr.unloadCalled)
    }

    /// 11. Cancel while the merge phase is parked mid-stream -> state returns to `.idle`
    /// and NO merged `.txt` is written (the write is guarded by a post-merge cancel check).
    @Test @MainActor func cancelDuringMergeReturnsIdle() async throws {
        let dir = try tempDir()
        let imgs = ["1.png", "2.png"].map { dir.appendingPathComponent($0) }
        let (settings, restore) = isolatedSettings(); defer { restore() }
        configured(settings)
        let ocr = FakeOCR(); ocr.results = ["1.png": "A", "2.png": "B"]
        let gate = Gate()
        let vm = makeVM(settings: settings, ocr: ocr, ocrModels: try installedOCRModels(),
                        merger: MergeService(chat: GatedChat(gate: gate, reply: "MERGED")),
                        modelManager: try freshModelManager())

        vm.start(urls: imgs)
        await waitUntil { gate.waiterCount == 1 }     // parked inside the merge's streamChat
        guard case .merging = vm.state else { Issue.record("expected .merging, got \(vm.state)"); return }
        vm.cancel()
        gate.resumeNext()                              // streamChat returns; merge completes
        await waitUntil { vm.state == .idle }

        #expect(vm.state == .idle)
        // The merged file must NOT have been written — the only .txt a done image batch
        // produces is the merged output, so an empty .txt scan proves the write was skipped.
        let mergedURL = dir.appendingPathComponent("1 merged.txt")
        #expect(!FileManager.default.fileExists(atPath: mergedURL.path))
        let leftovers = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        #expect(!leftovers.contains { $0.pathExtension == "txt" })
    }

    /// 12. `isLoadingOCRModel` gates StatusView's `.loadingModel` label: it must be true while
    /// the OCR model is loading (parked in `prepare`, state `.loadingModel`) and false once the
    /// batch completes. Uses a prepare-parking OCR fake for a deterministic mid-prepare probe.
    @Test @MainActor func isLoadingOCRModelTrueDuringPrepareFalseAfter() async throws {
        let dir = try tempDir()
        let imgs = ["1.png", "2.png"].map { dir.appendingPathComponent($0) }
        let (settings, restore) = isolatedSettings(); defer { restore() }
        configured(settings)
        let gate = Gate()
        let ocr = GatedPrepareOCR(gate: gate)
        ocr.results = ["1.png": "A", "2.png": "B"]
        let vm = makeVM(settings: settings, ocr: ocr, ocrModels: try installedOCRModels(),
                        merger: MergeService(chat: RecordingChat(reply: "M")),
                        modelManager: try freshModelManager())

        #expect(vm.isLoadingOCRModel == false)          // idle: flag clear
        vm.start(urls: imgs)
        await waitUntil { gate.waiterCount == 1 }        // parked inside prepare → OCR model loading
        #expect(vm.isLoadingOCRModel == true)            // flag set during the load
        guard case .loadingModel = vm.state else { Issue.record("expected .loadingModel, got \(vm.state)"); return }
        gate.resumeNext()                                 // prepare returns; batch runs to done
        await waitUntil { !vm.state.isBusy }
        #expect(vm.isLoadingOCRModel == false)           // cleared once prepare completed
        guard case .done = vm.state else { Issue.record("expected .done, got \(vm.state)"); return }
    }
}
