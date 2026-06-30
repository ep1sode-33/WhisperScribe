# Automatic Whisper Model Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the manual `huggingface-cli` model setup with an in-app curated model picker that auto-downloads via WhisperKit, with a first-run gate and backward-compatible reuse of already-installed models.

**Architecture:** A new `ModelManager` (`@MainActor ObservableObject`) is the single source of truth for model state (selection, installed-on-disk, per-model download progress). It downloads through a `ModelDownloading` seam whose production implementation wraps `WhisperKit.download`; tests inject a fake. `TranscriberService` is rewired to load whatever folder `ModelManager` resolves and to reload when it changes. The UI gates the drop zone behind `isReady` and exposes the picker in Settings.

**Tech Stack:** Swift 5.0, SwiftUI, WhisperKit 0.18.0, Swift Testing (new test target), Xcode 26, `xcodeproj` Ruby gem for project-file edits.

## Global Constraints

- Platform: macOS 14.0+, Apple Silicon. `MACOSX_DEPLOYMENT_TARGET = 14.0`.
- `SWIFT_VERSION = 5.0`; ad-hoc signing `CODE_SIGN_IDENTITY = "-"`.
- App bundle id: `com.william.WhisperScribe`. Test bundle id: `com.william.WhisperScribeTests`.
- The Xcode project (objectVersion 77) uses `PBXFileSystemSynchronizedRootGroup` for `WhisperScribe/`: **new `.swift` files placed anywhere under `WhisperScribe/` join the app target automatically — no pbxproj edit needed.** Only the *test* target and *test* files require pbxproj edits (done via `xcodeproj` scripts).
- WhisperKit repo is fixed at `argmaxinc/whisperkit-coreml`; default download base is `~/Documents/huggingface` (so a variant lands at `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/<variant>/`).
- The three model variants are exactly: `openai_whisper-large-v3-v20240930`, `openai_whisper-large-v3-v20240930_turbo`, `distil-whisper_distil-large-v3`.
- Localization: every user-facing string exists in all five languages — `en`, `zh-Hans`, `zh-Hant`, `ja`, `ko` — in `WhisperScribe/Localizable.xcstrings`.
- All `ruby` commands assume the `xcodeproj` gem is installed for the user (1.27.0, already present via `gem install --user-install xcodeproj`).
- Run all commands from the repo root `/Users/william/Desktop/WhisperScribe`.

---

## File structure

| File | Responsibility |
|---|---|
| `WhisperScribe/Models/WhisperModel.swift` (new) | The 3-entry curated catalog: id, display name, variant, tagline/size keys. |
| `WhisperScribe/Services/ModelDownloading.swift` (new) | `protocol ModelDownloading` — the download seam (Foundation only). |
| `WhisperScribe/Services/ModelManager.swift` (new) | `@MainActor` state machine: selection, installed scan, download/cancel/delete, `isReady`. No WhisperKit import. |
| `WhisperScribe/Services/WhisperKitDownloader.swift` (new) | Production `ModelDownloading` wrapping `WhisperKit.download` (only file importing WhisperKit for download). |
| `WhisperScribe/Services/TranscriberService.swift` (modify) | `prepare(modelFolder:)` reload-on-change; `transcribe` guards `pipe`; drop hard-coded path. |
| `WhisperScribe/Models/AppError.swift` (modify) | Replace `.modelMissing` → `.modelNotInstalled`; add `.modelDownloadFailed`. |
| `WhisperScribe/ViewModel/TranscriptionViewModel.swift` (modify) | Resolve folder from `ModelManager`, fail fast if not ready. |
| `WhisperScribe/App/AppModel.swift` (modify) | Own `modelManager`; pass into `viewModel`. |
| `WhisperScribe/App/WhisperScribeApp.swift` (modify) | Inject `modelManager` into both scenes. |
| `WhisperScribe/Views/ContentView.swift` (modify) | First-run gate: drop zone only when `isReady`. |
| `WhisperScribe/Views/SettingsView.swift` (modify) | New "Model" section. |
| `WhisperScribe/Localizable.xcstrings` (modify) | New keys ×5 languages; remove `error.modelMissing`. |
| `README.md` (modify) | Delete the manual model-setup ritual. |
| `WhisperScribeTests/` (new target) | Swift Testing logic tests for `WhisperModel` and `ModelManager`. |
| `scripts/add_test_target.rb`, `scripts/add_test_file.rb`, `scripts/update_strings.rb` (new) | Reliable project/catalog edits. |

## Subagent execution map (task dependencies)

Only Task 1 and the test-file registrations touch `project.pbxproj`; only Task 5 touches `AppError`. So the conflict surface is tiny.

```
Task 1 (test target)  ─┬─> Task 2 (WhisperModel + tests)
                       │        └─> Task 3 (ModelManager + tests) ─> Task 4 (WhisperKitDownloader)
Task 5 (AppError + TranscriberService)  ── independent of 2–4 ──┐
                                                                 ├─> Task 6 (AppModel/App/ViewModel wiring)
Task 4 ──────────────────────────────────────────────────────┘
Task 6 ─┬─> Task 7 (ContentView gate)      ─┐  (7 & 8 independent of each other)
        └─> Task 8 (SettingsView section)  ─┘
Task 9 (strings)  ── independent (UI compiles without it; keys fall back) ──> ok any time after Task 5
Task 10 (README)  ── independent ──> any time
Task 11 (integration build + manual run) ── last
```

For subagent-driven execution: run **1→2→3→4** in order (each modifies the project file or builds on the prior type), run **5** any time after 1, then **6**, then **7 and 8 in parallel**, with **9 and 10** as independent parallel tasks, and **11** last. Each task below ends in an independently reviewable, testable deliverable; serialize the pbxproj-touching steps (1, 2's & 3's `add_test_file.rb` calls) — never run two `xcodeproj` saves concurrently.

---

### Task 1: Add the Swift Testing target + project-edit helpers

**Files:**
- Create: `scripts/add_test_target.rb`
- Create: `scripts/add_test_file.rb`
- Create: `WhisperScribeTests/SmokeTests.swift`
- Modify: `WhisperScribe.xcodeproj/project.pbxproj` (via script)
- Modify: `WhisperScribe.xcodeproj/xcshareddata/xcschemes/WhisperScribe.xcscheme` (via script)

**Interfaces:**
- Produces: a `WhisperScribeTests` host-based unit-test bundle (TEST_HOST = the app), wired into the shared `WhisperScribe` scheme so `xcodebuild test -scheme WhisperScribe` runs it. Helper `scripts/add_test_file.rb <path>` registers a test source file into the target (idempotent).

- [ ] **Step 1: Write `scripts/add_test_target.rb`**

```ruby
#!/usr/bin/env ruby
# Adds a host-based Swift Testing unit-test bundle "WhisperScribeTests" and
# wires it into the shared "WhisperScribe" scheme. Idempotent.
require 'xcodeproj'

PROJECT = 'WhisperScribe.xcodeproj'
proj = Xcodeproj::Project.open(PROJECT)

app = proj.targets.find { |t| t.name == 'WhisperScribe' } or abort 'app target not found'

test = proj.targets.find { |t| t.name == 'WhisperScribeTests' }
unless test
  test = proj.new_target(:unit_test_bundle, 'WhisperScribeTests', :osx, '14.0')
end

test.build_configurations.each do |c|
  s = c.build_settings
  s['PRODUCT_BUNDLE_IDENTIFIER'] = 'com.william.WhisperScribeTests'
  s['PRODUCT_NAME']              = '$(TARGET_NAME)'
  s['SWIFT_VERSION']            = '5.0'
  s['MACOSX_DEPLOYMENT_TARGET'] = '14.0'
  s['GENERATE_INFOPLIST_FILE']  = 'YES'
  s['CODE_SIGN_IDENTITY']       = '-'
  s['CODE_SIGN_STYLE']          = 'Automatic'
  s['SWIFT_EMIT_LOC_STRINGS']   = 'NO'
  s['TEST_HOST']    = '$(BUILT_PRODUCTS_DIR)/WhisperScribe.app/Contents/MacOS/WhisperScribe'
  s['BUNDLE_LOADER'] = '$(TEST_HOST)'
end

# Depend on the app so it builds + hosts the tests.
test.add_dependency(app) unless test.dependencies.any? { |d| d.target == app }

# A plain (non-synchronized) group for test sources.
proj.main_group['WhisperScribeTests'] || proj.main_group.new_group('WhisperScribeTests', 'WhisperScribeTests')

proj.save

# Wire the test target into the shared scheme.
scheme_path = File.join(Xcodeproj::XCScheme.shared_data_dir(PROJECT).to_s, 'WhisperScribe.xcscheme')
scheme = Xcodeproj::XCScheme.new(scheme_path)
already = scheme.test_action.testables.any? { |t| t.buildable_references.first&.target_name == 'WhisperScribeTests' }
unless already
  scheme.test_action.add_testable(Xcodeproj::XCScheme::TestAction::TestableReference.new(test))
  scheme.save_as(PROJECT, 'WhisperScribe', true)
end

puts 'WhisperScribeTests target ready.'
```

- [ ] **Step 2: Write `scripts/add_test_file.rb`**

```ruby
#!/usr/bin/env ruby
# Registers a test source file (already created on disk under WhisperScribeTests/)
# into the WhisperScribeTests target. Idempotent. Usage:
#   ruby scripts/add_test_file.rb WhisperScribeTests/WhisperModelTests.swift
require 'xcodeproj'

rel = ARGV[0] or abort 'usage: add_test_file.rb WhisperScribeTests/<File>.swift'
abort "file not found: #{rel}" unless File.exist?(rel)

PROJECT = 'WhisperScribe.xcodeproj'
proj = Xcodeproj::Project.open(PROJECT)
test = proj.targets.find { |t| t.name == 'WhisperScribeTests' } or abort 'test target not found'
group = proj.main_group['WhisperScribeTests'] || proj.main_group.new_group('WhisperScribeTests', 'WhisperScribeTests')

abs = File.expand_path(rel)
ref = proj.files.find { |f| f.real_path.to_s == abs } || group.new_file(File.basename(rel))
test.add_file_references([ref]) unless test.source_build_phase.files_references.include?(ref)

proj.save
puts "registered #{rel}"
```

- [ ] **Step 3: Run the target script**

Run: `ruby scripts/add_test_target.rb`
Expected: prints `WhisperScribeTests target ready.`

- [ ] **Step 4: Write the smoke test `WhisperScribeTests/SmokeTests.swift`**

```swift
import Testing
@testable import WhisperScribe

struct SmokeTests {
    @Test func harnessRuns() {
        #expect(Bool(true))
    }
}
```

- [ ] **Step 5: Register the smoke test**

Run: `ruby scripts/add_test_file.rb WhisperScribeTests/SmokeTests.swift`
Expected: prints `registered WhisperScribeTests/SmokeTests.swift`

- [ ] **Step 6: Run the test bundle to verify the whole harness works**

Run: `xcodebuild test -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS' 2>&1 | tail -25`
Expected: builds, launches the host app, `SmokeTests.harnessRuns` passes — `** TEST SUCCEEDED **`.

- [ ] **Step 7: Commit**

```bash
git add scripts/add_test_target.rb scripts/add_test_file.rb WhisperScribeTests/SmokeTests.swift WhisperScribe.xcodeproj
git commit -m "test: add Swift Testing target (WhisperScribeTests) + project helpers"
```

---

### Task 2: `WhisperModel` catalog

**Files:**
- Create: `WhisperScribe/Models/WhisperModel.swift`
- Create: `WhisperScribeTests/WhisperModelTests.swift`

**Interfaces:**
- Produces: `struct WhisperModel: Identifiable, Hashable` with `let id, name, variant, taglineKey, sizeKey: String`; statics `WhisperModel.all: [WhisperModel]` (3 entries), `WhisperModel.default`, `WhisperModel.with(id:) -> WhisperModel?`.

- [ ] **Step 1: Write the failing test `WhisperScribeTests/WhisperModelTests.swift`**

```swift
import Testing
@testable import WhisperScribe

struct WhisperModelTests {
    @Test func catalogHasThreeUniqueModels() {
        #expect(WhisperModel.all.count == 3)
        #expect(Set(WhisperModel.all.map(\.id)).count == 3)
        #expect(Set(WhisperModel.all.map(\.variant)).count == 3)
    }

    @Test func defaultIsLargeV3WithInstalledVariant() {
        #expect(WhisperModel.default.id == "largeV3")
        #expect(WhisperModel.default.variant == "openai_whisper-large-v3-v20240930")
    }

    @Test func lookupRoundTrips() {
        for m in WhisperModel.all {
            #expect(WhisperModel.with(id: m.id) == m)
        }
        #expect(WhisperModel.with(id: "nope") == nil)
    }
}
```

- [ ] **Step 2: Register the test file and run to verify it fails**

Run:
```bash
ruby scripts/add_test_file.rb WhisperScribeTests/WhisperModelTests.swift
xcodebuild test -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS' 2>&1 | tail -25
```
Expected: build failure — `cannot find 'WhisperModel' in scope`.

- [ ] **Step 3: Write `WhisperScribe/Models/WhisperModel.swift`**

```swift
import Foundation

/// One entry in the curated model catalog. Pure value type — no I/O, no WhisperKit.
struct WhisperModel: Identifiable, Hashable {
    let id: String         // stable, persisted as the selection key
    let name: String       // verbatim display name, e.g. "large-v3-turbo"
    let variant: String    // WhisperKit/HF folder name; the download() argument
    let taglineKey: String // localized one-line description
    let sizeKey: String    // localized approximate size

    static let all: [WhisperModel] = [
        WhisperModel(id: "largeV3",
                     name: "large-v3",
                     variant: "openai_whisper-large-v3-v20240930",
                     taglineKey: "model.tagline.bestQuality",
                     sizeKey: "model.size.large"),
        WhisperModel(id: "largeV3Turbo",
                     name: "large-v3-turbo",
                     variant: "openai_whisper-large-v3-v20240930_turbo",
                     taglineKey: "model.tagline.fast",
                     sizeKey: "model.size.large"),
        WhisperModel(id: "distilV3",
                     name: "distil-large-v3",
                     variant: "distil-whisper_distil-large-v3",
                     taglineKey: "model.tagline.smallFast",
                     sizeKey: "model.size.distil"),
    ]

    static let `default`: WhisperModel = all[0]

    static func with(id: String) -> WhisperModel? { all.first { $0.id == id } }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `xcodebuild test -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS' 2>&1 | tail -25`
Expected: `** TEST SUCCEEDED **`, `WhisperModelTests` green.

- [ ] **Step 5: Commit**

```bash
git add WhisperScribe/Models/WhisperModel.swift WhisperScribeTests/WhisperModelTests.swift WhisperScribe.xcodeproj
git commit -m "feat: add curated WhisperModel catalog"
```

---

### Task 3: `ModelDownloading` seam + `ModelManager` state machine

**Files:**
- Create: `WhisperScribe/Services/ModelDownloading.swift`
- Create: `WhisperScribe/Services/ModelManager.swift`
- Create: `WhisperScribeTests/ModelManagerTests.swift`

**Interfaces:**
- Consumes: `WhisperModel` (Task 2).
- Produces:
  - `protocol ModelDownloading: Sendable { func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL }`
  - `@MainActor final class ModelManager: ObservableObject` with:
    - `enum DownloadState: Equatable { case idle; case downloading(Double); case failed(String) }`
    - `init(downloader: ModelDownloading, defaults: UserDefaults = .standard, fileManager: FileManager = .default, baseDir: URL? = nil)`
    - `@Published var selectedModelID: String`, `@Published private(set) var installedIDs: Set<String>`, `@Published private(set) var downloads: [String: DownloadState]`
    - `var selectedModel: WhisperModel`, `var isReady: Bool`
    - `func isInstalled(_:) -> Bool`, `func state(for:) -> DownloadState`, `func modelFolder(_:) -> URL`, `func modelFolderPath(_:) -> String`, `func refreshInstalled()`
    - `func download(_:)`, `func cancelDownload(_:)`, `func performDownload(_:) async`, `func delete(_:)`
    - `static func clampedProgress(current: Double, raw: Double) -> Double`

- [ ] **Step 1: Write the failing test `WhisperScribeTests/ModelManagerTests.swift`**

```swift
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
```

- [ ] **Step 2: Register the test file and run to verify it fails**

Run:
```bash
ruby scripts/add_test_file.rb WhisperScribeTests/ModelManagerTests.swift
xcodebuild test -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS' 2>&1 | tail -25
```
Expected: build failure — `cannot find 'ModelManager' / 'ModelDownloading' in scope`.

- [ ] **Step 3: Write `WhisperScribe/Services/ModelDownloading.swift`**

```swift
import Foundation

/// The model-download seam. Production code wraps `WhisperKit.download`; tests inject a fake.
/// Kept WhisperKit-free so `ModelManager` (which depends on this) stays dependency-light.
protocol ModelDownloading: Sendable {
    /// Download `variant` from the WhisperKit repo into the default base, reporting fractional
    /// progress (0...1) — which may be called off the main actor. Returns the model folder URL.
    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL
}
```

- [ ] **Step 4: Write `WhisperScribe/Services/ModelManager.swift`**

```swift
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
    static func clampedProgress(current: Double, raw: Double) -> Double {
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `xcodebuild test -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS' 2>&1 | tail -30`
Expected: `** TEST SUCCEEDED **`, all `ModelManagerTests` green.

- [ ] **Step 6: Commit**

```bash
git add WhisperScribe/Services/ModelDownloading.swift WhisperScribe/Services/ModelManager.swift WhisperScribeTests/ModelManagerTests.swift WhisperScribe.xcodeproj
git commit -m "feat: add ModelManager state machine + download seam"
```

---

### Task 4: Production `WhisperKitDownloader`

**Files:**
- Create: `WhisperScribe/Services/WhisperKitDownloader.swift`

**Interfaces:**
- Consumes: `ModelDownloading` (Task 3), `WhisperKit.download` (`WhisperKit.swift:244`).
- Produces: `struct WhisperKitDownloader: ModelDownloading`.

This wraps a network call, so it is verified by compilation + the end-to-end run in Task 11, not a unit test.

- [ ] **Step 1: Write `WhisperScribe/Services/WhisperKitDownloader.swift`**

```swift
import Foundation
import WhisperKit

/// Production `ModelDownloading`: downloads from `argmaxinc/whisperkit-coreml` into
/// WhisperKit's default base (`~/Documents/huggingface/...`). The only model-download
/// path that imports WhisperKit.
struct WhisperKitDownloader: ModelDownloading {
    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        try await WhisperKit.download(variant: variant) { p in
            progress(p.fractionCompleted)
        }
    }
}
```

- [ ] **Step 2: Build to verify it compiles**

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -configuration Debug -destination 'platform=macOS' build 2>&1 | tail -15`
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Commit**

```bash
git add WhisperScribe/Services/WhisperKitDownloader.swift
git commit -m "feat: add WhisperKitDownloader (production download seam)"
```

---

### Task 5: `AppError` cases + `TranscriberService` rewiring

**Files:**
- Modify: `WhisperScribe/Models/AppError.swift`
- Modify: `WhisperScribe/Services/TranscriberService.swift`

**Interfaces:**
- Produces: `AppError.modelNotInstalled`, `AppError.modelDownloadFailed(String)` (replacing `.modelMissing`); `TranscriberService.prepare(modelFolder: String)` and a `transcribe(...)` that no longer self-prepares and guards `pipe`.
- Consumes: nothing from Tasks 2–4 (kept independent so it can run in parallel).

- [ ] **Step 1: Edit `WhisperScribe/Models/AppError.swift`** — replace the `modelMissing` case and its description.

Replace:
```swift
    case noAudioTrack
    case modelMissing(path: String)
    case audioDecodeFailed(String)
```
with:
```swift
    case noAudioTrack
    case modelNotInstalled
    case modelDownloadFailed(String)
    case audioDecodeFailed(String)
```

Replace:
```swift
        case .modelMissing(let path):
            return String.localizedStringWithFormat(NSLocalizedString("error.modelMissing", comment: ""), path)
```
with:
```swift
        case .modelNotInstalled:
            return String(localized: "error.modelNotInstalled")
        case .modelDownloadFailed(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.modelDownloadFailed", comment: ""), m)
```

- [ ] **Step 2: Edit `WhisperScribe/Services/TranscriberService.swift`** — drop the hard-coded path, add reload-on-change, and stop self-preparing in `transcribe`.

Replace the property + `prepare()`:
```swift
    private var pipe: WhisperKit?

    private let modelFolder: String = {
        ("~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3-v20240930" as NSString)
            .expandingTildeInPath
    }()

    init() {}

    /// Idempotently load the WhisperKit pipeline.
    func prepare() async throws {
        if pipe != nil { return }

        guard FileManager.default.fileExists(atPath: modelFolder) else {
            throw AppError.modelMissing(path: modelFolder)
        }

        let compute = ModelComputeOptions(
```
with:
```swift
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
```

In the same `prepare`, replace the load/assignment:
```swift
        do {
            pipe = try await WhisperKit(config)
        } catch {
            throw AppError.transcriptionFailed(String(describing: error))
        }
    }
```
with:
```swift
        do {
            pipe = try await WhisperKit(config)
            loadedFolder = modelFolder
        } catch {
            throw AppError.transcriptionFailed(String(describing: error))
        }
    }
```

In `transcribe(...)`, replace the self-prepare + guard:
```swift
        try await prepare()
        guard let pipe else {
            throw AppError.transcriptionFailed(String(localized: "error.whisperKitNotInitialized"))
        }
```
with:
```swift
        guard let pipe else {
            throw AppError.modelNotInstalled
        }
```

- [ ] **Step 3: Build to verify it compiles**

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -configuration Debug -destination 'platform=macOS' build 2>&1 | tail -15`
Expected: `** BUILD SUCCEEDED **`. (No remaining reference to `.modelMissing` or `prepare()` with no args — the only `prepare()` caller is `TranscriptionViewModel`, updated in Task 6; building Task 5 alone will surface that call site as an error, which Task 6 fixes. If running Task 5 standalone, expect the build to fail only at `TranscriptionViewModel.swift:39` — that is acceptable and fixed in Task 6.)

> Note for sequencing: Task 5 and Task 6 together must build green. If your workflow requires every commit to build, combine Step 3's verification with Task 6. Otherwise commit Task 5 and proceed immediately to Task 6.

- [ ] **Step 4: Commit**

```bash
git add WhisperScribe/Models/AppError.swift WhisperScribe/Services/TranscriberService.swift
git commit -m "refactor: TranscriberService loads a passed model folder; AppError model cases"
```

---

### Task 6: Wire `ModelManager` through `AppModel`, the app scenes, and the view model

**Files:**
- Modify: `WhisperScribe/App/AppModel.swift`
- Modify: `WhisperScribe/App/WhisperScribeApp.swift`
- Modify: `WhisperScribe/ViewModel/TranscriptionViewModel.swift`

**Interfaces:**
- Consumes: `ModelManager(downloader:)` (Task 3), `WhisperKitDownloader` (Task 4), `TranscriberService.prepare(modelFolder:)` + `AppError.modelNotInstalled` (Task 5).
- Produces: `AppModel.modelManager`; `TranscriptionViewModel.init(settings:transcriber:cleaner:modelManager:)`.

- [ ] **Step 1: Edit `WhisperScribe/App/AppModel.swift`**

Replace the whole body:
```swift
@MainActor
final class AppModel: ObservableObject {
    let settings = SettingsStore()
    let transcriber = TranscriberService()
    let cleaner = LLMCleaner()
    lazy var viewModel = TranscriptionViewModel(
        settings: settings,
        transcriber: transcriber,
        cleaner: cleaner
    )
}
```
with:
```swift
@MainActor
final class AppModel: ObservableObject {
    let settings = SettingsStore()
    let transcriber = TranscriberService()
    let cleaner = LLMCleaner()
    let modelManager = ModelManager(downloader: WhisperKitDownloader())
    lazy var viewModel = TranscriptionViewModel(
        settings: settings,
        transcriber: transcriber,
        cleaner: cleaner,
        modelManager: modelManager
    )
}
```

- [ ] **Step 2: Edit `WhisperScribe/App/WhisperScribeApp.swift`** — inject `modelManager` into both scenes.

Replace:
```swift
            ContentView()
                .environmentObject(appModel)
                .environmentObject(appModel.settings)
                .environmentObject(appModel.viewModel)
                .frame(minWidth: 560, minHeight: 420)
```
with:
```swift
            ContentView()
                .environmentObject(appModel)
                .environmentObject(appModel.settings)
                .environmentObject(appModel.viewModel)
                .environmentObject(appModel.modelManager)
                .frame(minWidth: 560, minHeight: 420)
```

Replace:
```swift
            SettingsView()
                .environmentObject(appModel.settings)
                .environmentObject(appModel)
```
with:
```swift
            SettingsView()
                .environmentObject(appModel.settings)
                .environmentObject(appModel)
                .environmentObject(appModel.modelManager)
```

- [ ] **Step 3: Edit `WhisperScribe/ViewModel/TranscriptionViewModel.swift`** — add the dependency and resolve the folder.

Replace:
```swift
    private let settings: SettingsStore
    private let transcriber: TranscriberService
    private let cleaner: LLMCleaner
```
with:
```swift
    private let settings: SettingsStore
    private let transcriber: TranscriberService
    private let cleaner: LLMCleaner
    private let modelManager: ModelManager
```

Replace:
```swift
    init(settings: SettingsStore, transcriber: TranscriberService, cleaner: LLMCleaner) {
        self.settings = settings
        self.transcriber = transcriber
        self.cleaner = cleaner
    }
```
with:
```swift
    init(settings: SettingsStore, transcriber: TranscriberService, cleaner: LLMCleaner, modelManager: ModelManager) {
        self.settings = settings
        self.transcriber = transcriber
        self.cleaner = cleaner
        self.modelManager = modelManager
    }
```

Replace step 1 of the pipeline:
```swift
                // 1. Load model
                try await self.transcriber.prepare()
                try Task.checkCancellation()
```
with:
```swift
                // 1. Load model (resolve the selected model's folder; gate on readiness)
                let model = self.modelManager.selectedModel
                guard self.modelManager.isReady else { throw AppError.modelNotInstalled }
                try await self.transcriber.prepare(modelFolder: self.modelManager.modelFolderPath(model))
                try Task.checkCancellation()
```

- [ ] **Step 4: Build to verify the app compiles**

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -configuration Debug -destination 'platform=macOS' build 2>&1 | tail -15`
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 5: Commit**

```bash
git add WhisperScribe/App/AppModel.swift WhisperScribe/App/WhisperScribeApp.swift WhisperScribe/ViewModel/TranscriptionViewModel.swift
git commit -m "feat: wire ModelManager through app + view model"
```

---

### Task 7: First-run gate in `ContentView`

**Files:**
- Modify: `WhisperScribe/Views/ContentView.swift`

**Interfaces:**
- Consumes: `ModelManager.isReady` (Task 3), `modelManager` EnvironmentObject (Task 6). Localized keys `content.noModel.title`, `content.noModel.openSettings` (Task 9; UI compiles and falls back to the key if Task 9 hasn't run yet).

- [ ] **Step 1: Edit `WhisperScribe/Views/ContentView.swift`** — observe the manager and gate the drop zone.

Add after the existing `@EnvironmentObject` lines:
```swift
    @EnvironmentObject var modelManager: ModelManager
```

Replace:
```swift
    private var showDropZone: Bool {
        if case .idle = viewModel.state { return true }
        return false
    }
```
with:
```swift
    private var isIdle: Bool {
        if case .idle = viewModel.state { return true }
        return false
    }
```

Replace the `Group { ... }` content:
```swift
            Group {
                if showDropZone {
                    DropZone { url in
                        viewModel.start(url: url)
                    }
                    .padding(20)
                } else {
                    StatusView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
```
with:
```swift
            Group {
                if isIdle {
                    if modelManager.isReady {
                        DropZone { url in
                            viewModel.start(url: url)
                        }
                        .padding(20)
                    } else {
                        noModelView
                    }
                } else {
                    StatusView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
```

Add this computed view (e.g. after `header`):
```swift
    private var noModelView: some View {
        VStack(spacing: 14) {
            Image(systemName: "arrow.down.circle.dotted")
                .font(.system(size: 44))
                .foregroundStyle(.secondary)
            Text("content.noModel.title")
                .font(.headline)
            SettingsLink {
                Label("content.noModel.openSettings", systemImage: "gearshape")
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: 420, maxHeight: .infinity)
        .padding(24)
    }
```

- [ ] **Step 2: Build to verify it compiles**

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -configuration Debug -destination 'platform=macOS' build 2>&1 | tail -15`
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Commit**

```bash
git add WhisperScribe/Views/ContentView.swift
git commit -m "feat: gate drop zone behind installed model"
```

---

### Task 8: "Model" section in `SettingsView`

**Files:**
- Modify: `WhisperScribe/Views/SettingsView.swift`

**Interfaces:**
- Consumes: `ModelManager` API (Task 3), `modelManager` EnvironmentObject (Task 6). Localized keys from Task 9 (compiles without them).

- [ ] **Step 1: Edit `WhisperScribe/Views/SettingsView.swift`** — add the manager and the section.

Add after `@EnvironmentObject var appModel: AppModel`:
```swift
    @EnvironmentObject var modelManager: ModelManager
```

In `body`, add `modelSection` as the first section:
```swift
        Form {
            modelSection
            transcriptionSection
            cleanupSection
            llmSection
            outputSection
        }
```

Add the section + row builder (e.g. after `transcriptionSection`):
```swift
    // MARK: - 模型

    private var modelSection: some View {
        Section("settings.model") {
            ForEach(WhisperModel.all) { model in
                modelRow(model)
            }
            Text("model.storageFootnote")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private func modelRow(_ model: WhisperModel) -> some View {
        let installed = modelManager.isInstalled(model)
        let selected = modelManager.selectedModel.id == model.id
        HStack(spacing: 10) {
            Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                .foregroundStyle(selected ? Color.accentColor : Color.secondary)
                .onTapGesture { if installed { modelManager.selectedModelID = model.id } }
                .opacity(installed ? 1 : 0.4)

            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: model.name).font(.body)
                HStack(spacing: 6) {
                    Text(model.taglineKey)
                    Text(verbatim: "·")
                    Text(model.sizeKey)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            modelAccessory(model, installed: installed)
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func modelAccessory(_ model: WhisperModel, installed: Bool) -> some View {
        switch modelManager.state(for: model) {
        case .downloading(let f):
            HStack(spacing: 8) {
                ProgressView(value: f).progressViewStyle(.linear).frame(width: 90)
                Text("\(Int(f * 100))%").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Button("common.cancel") { modelManager.cancelDownload(model) }
            }
        case .failed(let msg):
            HStack(spacing: 8) {
                Label(msg, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange).lineLimit(1)
                Button("model.retry") { modelManager.download(model) }
            }
        case .idle:
            if installed {
                HStack(spacing: 10) {
                    Label("model.installed", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.green)
                    Button("model.delete", role: .destructive) { modelManager.delete(model) }
                        .font(.caption)
                }
            } else {
                Button("model.download") { modelManager.download(model) }
            }
        }
    }
```

- [ ] **Step 2: Build to verify it compiles**

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -configuration Debug -destination 'platform=macOS' build 2>&1 | tail -15`
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 3: Commit**

```bash
git add WhisperScribe/Views/SettingsView.swift
git commit -m "feat: add Model picker/download section to Settings"
```

---

### Task 9: Localized strings

**Files:**
- Create: `scripts/update_strings.rb`
- Modify: `WhisperScribe/Localizable.xcstrings`

**Interfaces:**
- Produces: all new `model.*`, `content.noModel.*`, `error.modelNotInstalled`, `error.modelDownloadFailed` keys in 5 languages; removes `error.modelMissing`.

- [ ] **Step 1: Write `scripts/update_strings.rb`**

```ruby
#!/usr/bin/env ruby
# Adds the model-management strings (5 languages) to Localizable.xcstrings and removes
# the obsolete error.modelMissing key. Idempotent (overwrites these keys, leaves others).
require 'json'

PATH = 'WhisperScribe/Localizable.xcstrings'
doc = JSON.parse(File.read(PATH))
strings = doc['strings']

def entry(en:, zh_hans:, zh_hant:, ja:, ko:)
  {
    'extractionState' => 'manual',
    'localizations' => {
      'en'      => { 'stringUnit' => { 'state' => 'translated', 'value' => en } },
      'zh-Hans' => { 'stringUnit' => { 'state' => 'translated', 'value' => zh_hans } },
      'zh-Hant' => { 'stringUnit' => { 'state' => 'translated', 'value' => zh_hant } },
      'ja'      => { 'stringUnit' => { 'state' => 'translated', 'value' => ja } },
      'ko'      => { 'stringUnit' => { 'state' => 'translated', 'value' => ko } },
    }
  }
end

adds = {
  'settings.model'              => entry(en: 'Model',              zh_hans: '模型',         zh_hant: '模型',         ja: 'モデル',            ko: '모델'),
  'model.tagline.bestQuality'   => entry(en: 'Best quality',       zh_hans: '质量最佳',     zh_hant: '品質最佳',     ja: '最高品質',          ko: '최고 품질'),
  'model.tagline.fast'          => entry(en: 'Fast',               zh_hans: '快速',         zh_hant: '快速',         ja: '高速',              ko: '빠름'),
  'model.tagline.smallFast'     => entry(en: 'Small & fast',       zh_hans: '小而快',       zh_hant: '小而快',       ja: '小型・高速',        ko: '작고 빠름'),
  'model.size.large'            => entry(en: '~1.5 GB',            zh_hans: '约 1.5 GB',    zh_hant: '約 1.5 GB',    ja: '約 1.5 GB',         ko: '약 1.5 GB'),
  'model.size.distil'           => entry(en: '~0.6 GB',            zh_hans: '约 0.6 GB',    zh_hant: '約 0.6 GB',    ja: '約 0.6 GB',         ko: '약 0.6 GB'),
  'model.download'              => entry(en: 'Download',           zh_hans: '下载',         zh_hant: '下載',         ja: 'ダウンロード',      ko: '다운로드'),
  'model.installed'             => entry(en: 'Installed',          zh_hans: '已安装',       zh_hant: '已安裝',       ja: 'インストール済み',  ko: '설치됨'),
  'model.retry'                 => entry(en: 'Retry',              zh_hans: '重试',         zh_hant: '重試',         ja: '再試行',            ko: '다시 시도'),
  'model.delete'                => entry(en: 'Delete',             zh_hans: '删除',         zh_hant: '刪除',         ja: '削除',              ko: '삭제'),
  'model.storageFootnote'       => entry(en: 'Models are stored locally and never re-downloaded.',
                                          zh_hans: '模型保存在本地，不会重复下载。',
                                          zh_hant: '模型儲存在本機，不會重複下載。',
                                          ja: 'モデルはローカルに保存され、再ダウンロードされません。',
                                          ko: '모델은 로컬에 저장되며 다시 다운로드되지 않습니다.'),
  'content.noModel.title'       => entry(en: 'No transcription model installed yet',
                                          zh_hans: '还没有安装转录模型',
                                          zh_hant: '尚未安裝轉錄模型',
                                          ja: '文字起こしモデルがまだインストールされていません',
                                          ko: '아직 설치된 받아쓰기 모델이 없습니다'),
  'content.noModel.openSettings'=> entry(en: 'Open Settings to download',
                                          zh_hans: '打开设置下载',
                                          zh_hant: '開啟設定下載',
                                          ja: '設定を開いてダウンロード',
                                          ko: '설정을 열어 다운로드'),
  'error.modelNotInstalled'     => entry(en: 'No transcription model installed. Open Settings (⌘,) to download one.',
                                          zh_hans: '尚未安装转录模型。请在设置（⌘,）中下载一个。',
                                          zh_hant: '尚未安裝轉錄模型。請在設定（⌘,）中下載一個。',
                                          ja: '文字起こしモデルがインストールされていません。設定（⌘,）からダウンロードしてください。',
                                          ko: '설치된 받아쓰기 모델이 없습니다. 설정(⌘,)에서 다운로드하세요.'),
  'error.modelDownloadFailed'   => entry(en: 'Model download failed: %@',
                                          zh_hans: '模型下载失败：%@',
                                          zh_hant: '模型下載失敗：%@',
                                          ja: 'モデルのダウンロードに失敗しました：%@',
                                          ko: '모델 다운로드 실패: %@'),
}

adds.each { |k, v| strings[k] = v }
strings.delete('error.modelMissing')

File.write(PATH, JSON.pretty_generate(doc) + "\n")
puts "updated #{adds.size} keys; removed error.modelMissing"
```

- [ ] **Step 2: Run it**

Run: `ruby scripts/update_strings.rb`
Expected: prints `updated 14 keys; removed error.modelMissing`.

- [ ] **Step 3: Verify the catalog is valid JSON and keys are present**

Run:
```bash
ruby -e 'require "json"; d=JSON.parse(File.read("WhisperScribe/Localizable.xcstrings")); %w[settings.model error.modelNotInstalled content.noModel.title].each{|k| abort("missing "+k) unless d["strings"][k]}; abort("modelMissing still present") if d["strings"]["error.modelMissing"]; puts "ok: "+d["strings"].length.to_s+" keys, all 5 langs"'
```
Expected: `ok: 90 keys, all 5 langs`.

- [ ] **Step 4: Build to confirm the catalog still compiles**

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -configuration Debug -destination 'platform=macOS' build 2>&1 | tail -8`
Expected: `** BUILD SUCCEEDED **`.

- [ ] **Step 5: Commit**

```bash
git add scripts/update_strings.rb WhisperScribe/Localizable.xcstrings
git commit -m "i18n: add model-management strings; drop error.modelMissing"
```

---

### Task 10: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Delete the "Model setup" section** — remove the entire block from the `## Model setup` heading through the blockquote that ends `...in \`WhisperScribe/Services/TranscriberService.swift\`.` (README.md lines ~29–48 in the current file: the heading, the path code block, the `huggingface-cli` block, and the "Want a different model…" note).

- [ ] **Step 2: Update the Requirements bullet** — replace:
```markdown
- The **WhisperKit CoreML model**, cached locally (see below).
```
with:
```markdown
- A **WhisperKit CoreML model** — the app downloads one for you on first run (pick it in Settings); no manual setup.
```

- [ ] **Step 3: Update the Usage list** — replace the step 2 bullet:
```markdown
2. *(optional)* Open **Settings** (⌘,) to set the cleanup level, language, output
   location, and your BYOK LLM endpoint (then **Test Connection**).
```
with:
```markdown
2. **First run:** open **Settings** (⌘,) → **Model**, pick one (large-v3, large-v3-turbo,
   or distil-large-v3) and it downloads itself with a progress bar. Also here: cleanup
   level, language, output location, and your BYOK LLM endpoint (then **Test Connection**).
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: drop manual model setup; document in-app model picker"
```

---

### Task 11: End-to-end integration

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run: `xcodebuild test -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS' 2>&1 | tail -30`
Expected: `** TEST SUCCEEDED **` — `WhisperModelTests`, `ModelManagerTests`, `SmokeTests` all green.

- [ ] **Step 2: Manual run-through (build + launch)**

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -configuration Debug -destination 'platform=macOS' build 2>&1 | tail -5` then `open` the built app (path printed under `Build/Products/Debug/WhisperScribe.app`), or press ⌘R in Xcode.

Verify by hand:
- Because `openai_whisper-large-v3-v20240930` is already on disk, the app opens **ready** (drop zone shown, no gate) with **large-v3 · best quality** marked Installed + selected in Settings.
- Settings → Model shows three rows; the two not-installed show **Download**; clicking one shows a progress bar then ✓ Installed; selecting an installed model updates the radio.
- Drop a media file → transcribes as before (now using the selected model's folder).
- *(optional negative check)* temporarily rename the installed variant folder → app shows the "No transcription model installed yet → Open Settings" gate; restore it → drop zone returns.

- [ ] **Step 3: Final commit (if any manual fixups were needed)**

```bash
git add -A
git commit -m "chore: finalize automatic model management" || echo "nothing to finalize"
```

---

## Self-review notes

- **Spec coverage:** catalog → Task 2; `ModelManager` (selection/installed/download/isReady/delete) → Task 3; download seam prod impl → Task 4; `TranscriberService.prepare(modelFolder:)` reload + `transcribe` guard + `AppError` → Task 5; app/viewmodel wiring → Task 6; main-window gate → Task 7; Settings section → Task 8; strings (5 langs) + remove `modelMissing` → Task 9; README → Task 10; testing strategy (Swift Testing target) → Task 1 + tests in 2–3; backward-compat (reuse `openai_whisper-large-v3-v20240930`) → asserted in Task 2 and Task 11 manual check.
- **Type consistency:** `prepare(modelFolder:)`, `modelFolderPath(_:)`, `selectedModel`, `isReady`, `state(for:)`, `download(_:)`, `cancelDownload(_:)`, `performDownload(_:)`, `clampedProgress(current:raw:)`, `AppError.modelNotInstalled` / `.modelDownloadFailed(String)` are used identically across tasks.
- **No placeholders:** every code/edit step shows the exact content.
