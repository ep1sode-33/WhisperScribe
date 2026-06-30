# Design: Automatic Whisper model management

**Date:** 2026-06-30
**Status:** Approved (design)
**Topic:** Replace manual `huggingface-cli` model setup with an in-app model picker that auto-downloads.

## Problem

Today the app forces the user to manage the Whisper model by hand:

- `TranscriberService.swift` hard-codes a model path under `~/Documents/huggingface/...`
  and constructs `WhisperKitConfig(..., download: false)` — auto-download is explicitly
  disabled.
- If the folder is absent, `prepare()` throws `AppError.modelMissing` and the README's
  entire "Model setup" section asks the user to run `huggingface-cli download` first and
  to edit `modelFolder` in source to change models.

For a drag-a-file-in consumer app this is backwards: WhisperKit already ships model
downloading. We are opting out of it and pushing the work onto the user.

## Goal

The user never touches the command line. On first run they pick one of a few curated
models in Settings and it downloads itself with a progress bar. They can switch models
later. No model = a clear in-app prompt, not a CLI ritual.

Non-goals (explicitly out of scope): a full "show all variants" catalog, quantized
variants, and background/resumable download sessions. We use the three full-precision
variants below and a normal foreground download with a Cancel button.

## Key facts established from source

- `WhisperKit.download(variant:downloadBase:useBackgroundSession:from:token:endpoint:progressCallback:) async throws -> URL`
  (`WhisperKit.swift:244`) downloads a variant and returns its on-disk folder URL. The
  `progressCallback: ((Progress) -> Void)?` gives a Foundation `Progress` we bind to a
  determinate download bar.
- `HubApi` defaults `downloadBase` to `~/Documents/huggingface` (`HubApi.swift:121`), so a
  downloaded variant lands at `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/<variant>/`
  — **exactly** the path the README documents for manual setup. The dev machine already
  has `openai_whisper-large-v3-v20240930` there.
- **Therefore auto-download is fully backward-compatible:** an already-present variant
  folder is reused and nothing re-downloads. WhisperKit's `download` uses
  `hubApi.snapshot`, which skips files already present.
- All three chosen variants are valid WhisperKit variants (present in the
  `Models.swift` fallback `ModelSupportConfig`, lines ~1642–1690).

## Curated model catalog

Three models, full-precision, matching what the dev machine already has:

| id (stable)   | Display name              | Variant folder                              | Approx size |
|---------------|---------------------------|---------------------------------------------|-------------|
| `largeV3`     | large-v3 · best quality   | `openai_whisper-large-v3-v20240930`         | ~1.5 GB     |
| `largeV3Turbo`| large-v3-turbo · fast     | `openai_whisper-large-v3-v20240930_turbo`   | ~1.5 GB     |
| `distilV3`    | distil-large-v3 · small   | `distil-whisper_distil-large-v3`            | ~0.6 GB     |

`largeV3` maps to the variant already installed on disk, so existing users re-download
nothing.

## Components

### `Models/WhisperModel.swift` (new)

A value type describing one catalog entry:

```
struct WhisperModel: Identifiable, Hashable {
    let id: String            // "largeV3" — stable, persisted as the selection key
    let variant: String       // "openai_whisper-large-v3-v20240930" — HF folder / download arg
    let displayNameKey: String // localized key for the display name
    let approxSizeText: String // "~1.5 GB" (localized)
    static let all: [WhisperModel]   // the three above, in display order
    static let `default`: WhisperModel // .all.first (largeV3)
    static func with(id: String) -> WhisperModel?
}
```

The repo is fixed at `argmaxinc/whisperkit-coreml` (WhisperKit's default) and is not part
of this type.

### `Services/ModelManager.swift` (new, `@MainActor final class ModelManager: ObservableObject`)

Single source of truth for model state; injected as an `EnvironmentObject`.

State:
- `@AppStorage("selectedModelID")` private backing + a published computed `selectedModel`
  (defaults to `WhisperModel.default` when unset/unknown), mirroring `SettingsStore`'s
  pattern of `objectWillChange.send()` on a private `@AppStorage` backing.
- `@Published var downloads: [String: DownloadState]` keyed by model id, where
  `enum DownloadState { case idle, downloading(Double), failed(String) }`.
- `@Published private(set) var installedIDs: Set<String>` — recomputed on init, after a
  download completes, and after a delete.

Derived:
- `func isInstalled(_ m: WhisperModel) -> Bool` — the variant folder exists **and**
  contains the model packages (check for `*.mlmodelc` entries / `MelSpectrogram.mlmodelc`),
  which is more robust than the current bare folder-exists check.
- `var isReady: Bool` — `selectedModel` is installed and not currently downloading.
- `func modelFolder(_ m: WhisperModel) -> String` — resolves
  `<downloadBase>/models/argmaxinc/whisperkit-coreml/<variant>` using the same default
  base WhisperKit uses (`FileManager` documents dir + `huggingface`). Used by the
  transcription path to load.

Actions:
- `func download(_ m: WhisperModel)` — sets `.downloading(0)`, calls
  `WhisperKit.download(variant: m.variant, progressCallback:)`, marshals progress to the
  main actor (callback may fire off-main), monotonically clamps to 0…1, on success adds to
  `installedIDs` and — if no model was selected/installed before — sets it as
  `selectedModel`; on failure sets `.failed(message)`. Guards against double-starting an
  in-flight download for the same id.
- `func cancelDownload(_ m: WhisperModel)` — cancels the in-flight task, resets to `.idle`.
- `func delete(_ m: WhisperModel)` *(optional, low priority)* — removes the variant folder,
  updates `installedIDs`; if it was selected, fall back to another installed model or none.

Error mapping: a failed download surfaces `AppError.modelDownloadFailed(message)` text in
the row; it never crashes or blocks other models.

### `Services/TranscriberService.swift` (changed)

- `prepare()` becomes `prepare(modelFolder: String)`. It tracks the currently-loaded folder
  and **reloads the pipeline if the folder changed** (switching models in Settings takes
  effect on the next job). Remove the hard-coded `modelFolder` constant and the
  `AppError.modelMissing` throw.
- Keep `WhisperKitConfig(..., download: false)` at load time — by the time we transcribe the
  variant is already on disk, so loading stays deterministic and offline. If the folder is
  somehow missing at load (race / deleted), throw `AppError.modelNotInstalled`.
- `transcribe(...)` no longer calls `prepare()` internally and takes no folder argument.
  The caller (`TranscriptionViewModel.start`) calls `prepare(modelFolder:)` first; `transcribe`
  guards `pipe != nil` and throws `AppError.modelNotInstalled` otherwise. This removes the
  ambiguity of who resolves the folder — the ViewModel owns resolution, `prepare` owns
  loading, `transcribe` owns decoding.

### `ViewModel/TranscriptionViewModel.swift` (changed)

- Holds a reference to `ModelManager` (added to its init).
- In `start(url:)` step 1, resolve `let folder = modelManager.modelFolder(modelManager.selectedModel)`
  and call `transcriber.prepare(modelFolder: folder)`. If `!modelManager.isReady`, fail fast
  with `AppError.modelNotInstalled` (the UI gate normally prevents reaching here).

### `App/AppModel.swift` + `App/WhisperScribeApp.swift` (changed)

- `AppModel` gains `let modelManager = ModelManager()` and passes it into the `viewModel`
  initializer.
- Both scenes in `WhisperScribeApp` add `.environmentObject(appModel.modelManager)` (the
  `WindowGroup` for `ContentView`'s gate, and `Settings` for the model section).

### `Views/ContentView.swift` (changed) — main-window gate

- Add `@EnvironmentObject var modelManager: ModelManager`.
- When state is `.idle` **and** `!modelManager.isReady`, render a "no model" view instead of
  the drop zone: an icon, the message *"No transcription model installed yet — open Settings
  (⌘,) to download one,"* and a `SettingsLink` button. When `isReady`, show the normal
  `DropZone`. (Busy/done/error states are unchanged and still show `StatusView`.)

### `Views/SettingsView.swift` (changed) — new "Model" section

A new `Section("settings.model")`, placed first (before Transcription). One row per
`WhisperModel.all`:

- A leading selection control (radio-style) binding the active model to
  `modelManager.selectedModel`. A model is selectable once installed; non-installed rows
  show the selection control disabled.
- Name (`displayNameKey`) + secondary size text (`approxSizeText`).
- Trailing accessory driven by `downloads[id]` / installed state:
  - not installed, idle → `[Download]` button → `modelManager.download(m)`.
  - downloading(f) → determinate `ProgressView(value: f)` + percent + a Cancel button.
  - failed(msg) → small red label + a Retry button.
  - installed → a ✓ "Installed" label, and *(optional)* a Delete button.
- Section footer: models are stored locally under `~/Documents/huggingface/...` and are
  never re-downloaded.

### `Models/AppError.swift` (changed)

- Remove `case modelMissing(path:)`.
- Add `case modelNotInstalled` → localized "No transcription model installed. Open Settings
  (⌘,) to download one."
- Add `case modelDownloadFailed(String)` → localized "Model download failed: %@".

### `Localizable.xcstrings` (changed)

Add keys, translated for all five languages (en, zh-Hans, zh-Hant, ja, ko):
- `settings.model` (section title)
- `model.largeV3.name`, `model.largeV3Turbo.name`, `model.distilV3.name`
- `model.size.large` (~1.5 GB, reused by `largeV3` and `largeV3Turbo`),
  `model.size.distil` (~0.6 GB)
- `model.download`, `model.installed`, `model.downloading`, `model.retry`,
  `model.delete`, `model.cancel`
- `model.storageFootnote`
- `content.noModel.title`, `content.noModel.openSettings`
- `error.modelNotInstalled`, `error.modelDownloadFailed`
- Remove the now-unused `error.modelMissing`.

### `README.md` (changed)

- Delete the "Model setup" section (the `huggingface-cli` block and the "change
  `modelFolder` in source" note).
- In Requirements, change "The WhisperKit CoreML model, cached locally (see below)" to a
  line noting the app downloads the model on first run.
- In Usage, add: on first run, open Settings and pick a model; it downloads itself.

## Data flow

```
First run (no model):
  ContentView sees !modelManager.isReady → shows "no model" gate → SettingsLink
  SettingsView Model section → user taps [Download] on a model
  ModelManager.download → WhisperKit.download(variant:, progressCallback:)
    → progress bar in the row → on success: installedIDs += id, selectedModel = it
  ContentView now isReady → DropZone

Transcribe (model present):
  drop file → ViewModel.start
    → folder = modelManager.modelFolder(selectedModel)
    → transcriber.prepare(modelFolder: folder)   // reloads if changed
    → extract → transcribe → cleanup → write     // unchanged downstream

Switch model in Settings:
  selectedModel changes → next job's prepare(modelFolder:) sees a new folder → reload
```

## Error handling

- Download failure → `.failed(msg)` in the row + Retry; other models unaffected; never
  blocks the app.
- Selected model deleted/missing at load time → `AppError.modelNotInstalled`, app returns to
  the gate.
- Off-main progress callbacks → marshalled to `@MainActor`, monotonic-clamped (reuse the
  existing `MonotonicFraction` idea) so the bar never jumps backward.
- Cancel mid-download → task cancelled, row returns to idle; a partially-written folder is
  treated as not-installed by the `.mlmodelc` presence check.

## Testing

- `WhisperModel`: `all` has 3 entries, ids/variants unique, `with(id:)` round-trips,
  `default == largeV3`.
- `ModelManager.modelFolder` resolves to the expected path for each variant and matches
  WhisperKit's default base (so existing installs are detected).
- `isInstalled` true only when folder exists and contains `.mlmodelc` packages (use a temp
  dir with/without a fake `.mlmodelc`).
- `isReady` reflects selected+installed and flips false during download.
- Download state machine via an injected/faked downloader: idle→downloading→installed,
  failure→failed, cancel→idle, progress is monotonic and clamped.
- `TranscriberService` reloads when `modelFolder` changes (folder-change detection unit
  test around the tracked-folder logic).

UI wiring (gate visibility, Settings rows) is verified manually by building and running.

## Implementation decomposition (for the plan)

The work splits into mostly-independent units suitable for parallel subagents; the plan
will spell out exact task boundaries, dependencies, and review checkpoints:

- **Task A — model types + manager (core, no UI):** `WhisperModel`, `ModelManager`
  (with a small injectable downloader seam for tests), unit tests. Pure logic; no
  dependency on UI tasks. This is the foundation other tasks import.
- **Task B — transcription wiring:** `TranscriberService.prepare(modelFolder:)` reload
  logic, `TranscriptionViewModel` + `AppModel` + `WhisperScribeApp` injection, `AppError`
  changes. Depends on A's public surface (the `modelFolder`/`selectedModel` API).
- **Task C — UI:** `ContentView` gate + `SettingsView` Model section. Depends on A's
  published state and B's `AppError` cases.
- **Task D — strings + docs:** `Localizable.xcstrings` keys (5 languages) + README edits.
  Largely independent; the string keys are agreed in this spec so D can proceed against the
  key list while A–C reference the same keys.

Dependency order: A first (defines the API surface), then B and C in parallel, D alongside.
A final integration + build/run pass verifies the gate→download→transcribe flow end to end.
Each task gets its own implementation step with a review checkpoint per the
subagent-driven-development / dispatching-parallel-agents workflow.
