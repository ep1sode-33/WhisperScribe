# DeepSeekOCR2Kit

Local **DeepSeek-OCR-2** inference on Apple Silicon, built on
[mlx-swift](https://github.com/ml-explore/mlx-swift) /
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm). No network, no Python
runtime — the SAM ViT-B + Qwen2 vision encoder, the linear projector, and the
DeepSeek-3B-MoE language model all run on-device from a quantized checkpoint.

The kit is a bit-for-bit Swift port of the
[`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) `deepseekocr_2` reference; every
stage (image preprocessing, SAM, the Qwen2 encoder, the MoE LM, greedy decode)
is parity-tested against golden tensors dumped from that reference (see
[Parity](#parity)).

## API

Two calls: `load` brings up the model + tokenizer; `ocr` streams decoded text.

```swift
import DeepSeekOCR2Kit

// 1. Load a local snapshot of the checkpoint (see Model download).
let session = try await OCR2Session.load(from: modelDir) { progress in
    print("loading \(Int(progress * 100))%")
}

// 2a. Free OCR — stream text as it is generated (CJK/multibyte-safe).
for try await chunk in session.ocr(image: cgImage) {
    print(chunk, terminator: "")
}

// 2b. Grounding — locate text and get normalized bounding boxes.
var raw = ""
for try await chunk in session.ocr(image: cgImage, task: .grounding(query: "Total")) {
    raw += chunk        // raw markers: <|ref|>Total<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
}
for box in OCR2Session.parseGrounding(raw) {
    print(box.label, box.box)   // box.box is a normalized [0,1] CGRect, top-left origin
}
```

`ocr(image:task:maxTokens:)` returns an `AsyncThrowingStream<String, Error>`.
Generation stops on EOS, at `maxTokens` (default **4096**), or when the consuming
task is cancelled (the stream finishes cleanly). Emission buffers partial UTF-8
until a full Unicode scalar is available, so CJK text never streams as garbled
bytes.

### Prompt templates

`OCRTask` maps to the reference's exact prompt strings:

| Task | Prompt sent to the model |
|---|---|
| `.freeOCR` | `<image>\nFree OCR. ` |
| `.grounding(query:)` | `<image>\nLocate <\|ref\|>{query}<\|/ref\|> in the image. ` |

The grounding template is the reference README's *"Text Localization
(Grounding)"* format — the only documented prompt with a user-supplied slot.
Its response carries `<|ref|>…<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>` markers
(coordinates in a 0–1000 range, top-left origin); `parseGrounding` divides by
1000 into normalized boxes. Grounding output is decoded with
`skipSpecialTokens: false` so those markers survive (they are `special:true` in
`tokenizer.json`, so a skip-specials decode would strip them).

## CLI

```bash
# Free OCR (streams to stdout)
ocr2-cli page.png --model-dir /path/to/DeepSeek-OCR-2-8bit

# Grounding (raw markers to stdout; parsed boxes + stats to stderr)
ocr2-cli receipt.png --model-dir DIR --grounding --query "price of Croissant"
```

`--model-dir` defaults to `$OCR2_MODEL_DIR`. Set `OCR2_STATS=1` for a
`[N tokens, Ts, R tok/s]` throughput line on stderr. Reference throughput on an
M-series Mac is **~28 tok/s** (steady-state generation).

## Model download

The checkpoint is `mlx-community/DeepSeek-OCR-2-8bit` (8-bit affine-quantized
language model + projector; bf16 SAM + Qwen2 encoder). Fetch a local snapshot:

```bash
huggingface-cli download mlx-community/DeepSeek-OCR-2-8bit \
    --local-dir ~/models/DeepSeek-OCR-2-8bit
```

Point `OCR2Session.load(from:)` at that directory. In the WhisperScribe app
(sub-project ②) the download + on-disk snapshot are managed by the app's
`ModelManager`; this kit only consumes a ready local directory.

## Parity

The port is validated **fixture-first**: golden per-stage tensors are dumped
from the `mlx-vlm` Python reference (`scripts/gen_fixtures.py`), and each Swift
stage is asserted against them. No stage advances until its gate is green.

| Stage | Gate | Result |
|---|---|---|
| Image preprocessing | pixel mean-abs diff vs golden NCHW (tol 2/255/0.5 ≈ 1.6e-2) | ~2e-8 (≈bit-exact), 3 fixtures |
| Prompt tokenization | token ids + `images_spatial_crop` | 5/5 fixtures exact |
| SAM ViT-B encoder | output rel err | ~1.5e-2 |
| Qwen2 vision encoder | output rel err (`< 2e-2`) | 1.4e-2 |
| Full model (injected pixels) | first-100 greedy token ids exact | 3/3 fixtures |
| End-to-end (CGImage→text) | decoded text vs reference | 2/2 fixtures |

The small (~1–2%) vision-tower drift is absorbed by greedy decode — token
embeddings use the same quantized `embed_tokens` as the reference, so only the
scattered visual features carry drift, and 100 greedy tokens match bit-for-bit
regardless.

### Regenerating fixtures

Golden tensors live under `Fixtures/`. To regenerate (needs the reference
`mlx-vlm` install and the model snapshot):

```bash
python3 -m venv .venv && .venv/bin/pip install mlx-vlm==0.6.6
.venv/bin/python scripts/make_test_images.py   # deterministic PIL test images
.venv/bin/python scripts/gen_fixtures.py        # per-stage golden tensors
```

`gen_fixtures.py` reads `Fixtures/model_dir.txt` (the local snapshot path) and
`Fixtures/images/`, and writes per-fixture `meta.json` + `*.safetensors`.

## Testing

**Run tests with `xcodebuild`, not `swift test`.** MLX's Metal shader library
(`default.metallib`) is compiled by Xcode's build system; the SwiftPM CLI does
not produce it, so `swift test` (and a `swift run`-built binary) fail at runtime
with *"Failed to load the default metallib"*. Use:

```bash
xcodebuild test \
    -scheme DeepSeekOCR2Kit-Package \
    -destination 'platform=macOS' \
    -skipPackagePluginValidation -skipMacroValidation
```

The parity tests are fixture-gated (`.enabled(if: FixtureSupport.root != nil)`)
and skip cleanly when `Fixtures/` and the model snapshot are absent — so CI runs
only the pure-logic tests (mask geometry, grounding parser, config merge, weight
sanitizer, prompt templates) without needing the multi-GB checkpoint. To run the
`ocr2-cli` locally, build it with `xcodebuild` too (that step produces the
metallib next to the binary).
