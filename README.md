# WhisperScribe

A native macOS app that turns any audio/video file into clean subtitles. Drop a file
in, it transcribes **locally on the Apple Neural Engine** with WhisperKit, optionally
polishes the text with a **bring-your-own-key** OpenAI-compatible LLM, and exports
`.srt` + `.txt`.

<p align="center">
  <img src="WhisperScribe/Assets.xcassets/AppIcon.appiconset/icon_256x256.png" width="128" alt="WhisperScribe icon">
</p>

## Features

- 🎙️ **Local transcription** via [WhisperKit](https://github.com/argmaxinc/WhisperKit) (Whisper large-v3) running on the **Neural Engine** — low power, no audio leaves your Mac.
- 📦 **Any container** — drag-drop `.mp4/.mov/.m4a/.mp3/.wav/.aac/.caf …`; audio is decoded in-app with AVFoundation (with an `ffmpeg` fallback for exotic codecs).
- 🧹 **Optional LLM cleanup** (BYOK, OpenAI-compatible) in **4 levels**, with a timestamp-preserving two-pass design so subtitle timing is never corrupted.
- 📝 **SRT + TXT** output next to the source file (or a folder you choose).
- 🌐 **5 languages** — English, 简体中文, 繁體中文, 日本語, 한국어.
- 📊 **Honest progress** — a determinate bar for transcription (bound to WhisperKit's own progress) and a live "characters generated" counter during streaming LLM cleanup.

## Requirements

- **macOS 14** (Sonoma) or later, **Apple Silicon** (M-series).
- **Xcode 16+** to build (developed on Xcode 26).
- A **WhisperKit CoreML model** — the app downloads one for you on first run (pick it in Settings); no manual setup.
- *(optional)* [`ffmpeg`](https://ffmpeg.org) on `PATH` at `/opt/homebrew/bin/ffmpeg` — only used as a fallback for containers AVFoundation can't open.
- *(optional)* a BYOK OpenAI-compatible chat endpoint for the cleanup feature.

## Build & run

```bash
open WhisperScribe.xcodeproj    # then press ⌘R
```

or from the command line:

```bash
xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe \
  -configuration Debug -destination 'platform=macOS' build
```

The app is unsandboxed and ad-hoc signed ("Sign to Run Locally") — no developer
account needed to run it on your own Mac. For a distributable `.app`, use
**Product ▸ Archive**.

## Usage

1. Drag a media file onto the window (or click **Choose File…**).
2. **First run:** open **Settings** (⌘,) → **Model**, pick one (large-v3, large-v3-turbo,
   or distil-large-v3) and it downloads itself with a progress bar. Also here: cleanup
   level, language, output location, and your BYOK LLM endpoint (then **Test Connection**).
3. Watch it transcribe → clean → export. Outputs land next to the source by default.

### Cleanup levels

| Level | What it does |
|-------|--------------|
| **L0 Raw** | Whisper output verbatim — no LLM. |
| **L1 Fix-only** | Punctuation, casing, homophones, proper nouns/terminology. Every word kept. |
| **L2 Clean + polish** | L1 + remove filler/stutters, fix grammar, add paragraph breaks. Meaning preserved. |
| **L3 Polish + light edit** | L2 + light condensing/reordering for readability (TXT only). |

> The **SRT is always capped at L2 semantics** (segment-local edits) so timestamps stay
> exactly aligned to the audio; L3's reflow applies only to the flowing-prose `.txt`.
> If the LLM is unreachable or misconfigured, cleanup **degrades gracefully to raw**
> output and warns you — it never fails the job or corrupts timing.

### Bring-your-own-key (LLM)

Any OpenAI-compatible `/chat/completions` endpoint works — enter `base_url`, `api_key`,
`model` in Settings (all blank by default; the key is stored in the **Keychain**). The
client streams responses (SSE) and reads both `content` and `reasoning_content`.

> **Reasoning models** (e.g. DeepSeek V4) "think" a lot, so batched cleanup of a whole
> transcript is slow. For fast bulk cleanup, point it at a non-thinking model or use L1.

## Architecture

```
ContentView ─ DropZone ─ StatusView ─ SettingsView      (SwiftUI)
        │
TranscriptionViewModel  (@MainActor state machine, cancellable pipeline)
        │
        ├─ AudioExtractor      AVFoundation → 16 kHz mono Float (ffmpeg fallback)
        ├─ TranscriberService  actor; WhisperKit on ANE; progress via Foundation Progress KVO
        ├─ LLMCleaner          actor; BYOK SSE streaming; two-pass, indexed 1:1 JSON, fail-closed
        └─ SubtitleWriter      SRTFormatter + CJK-aware TextJoiner + FileNaming → .srt/.txt
```

Notable choices: transcription progress is read from WhisperKit's own `Progress`
object (the segment callback gives block-relative timestamps under VAD chunking).
The LLM SRT pass sends an indexed JSON array and requires the same indices back
(strict 1:1), validating set-equality + length ratios and falling back to raw text
per-batch on any mismatch. UI strings use an Xcode **String Catalog**
(`Localizable.xcstrings`) with English semantic keys.

## Project layout

```
WhisperScribe/
├─ App/          WhisperScribeApp, AppModel
├─ Views/        ContentView, DropZone, StatusView, SettingsView
├─ ViewModel/    TranscriptionViewModel
├─ Services/     TranscriberService, AudioExtractor, LLMCleaner, LLMPrompts, SubtitleWriter
├─ Persistence/  SettingsStore, KeychainStore
├─ Support/      SRTFormatter, TextJoiner, FileNaming
├─ Models/       TimedSegment, CleanupLevel, JobState, AppError, LLMConfig, Preferences
└─ Localizable.xcstrings
```

## License

[MIT](LICENSE) © 2026 William.
