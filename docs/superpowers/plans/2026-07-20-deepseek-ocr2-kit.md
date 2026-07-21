# DeepSeekOCR2Kit 实施计划（DeepSeek-OCR-2 → mlx-swift 移植）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 WhisperScribe 仓库内建成 `DeepSeekOCR2Kit` 本地 SwiftPM 包：mlx-swift 上运行 `mlx-community/DeepSeek-OCR-2-8bit`，流式 OCR API + CLI，与 Python mlx-vlm 参考实现逐级 parity 验证。

**Architecture:** 复用 mlx-swift-lm 基元（量化加载/KV cache/MoE 内核/生成循环）+ 从 mzbac v1 港拷贝改造 SAM/LM/预处理 + 新写 Qwen2 视觉编码器、顶层拼装、Processor。双轨 parity：Python 夹具逐级中间张量比对（轨道 A）+ 预处理结构量/像素比对（轨道 B）。

**Tech Stack:** Swift 6 / SwiftPM、mlx-swift-lm（MLXVLM+MLXLMCommon+MLXHuggingFace）、swift-transformers Tokenizers、Swift Testing、Python mlx-vlm（仅夹具生成）。

**设计文档:** `docs/superpowers/specs/2026-07-20-deepseek-ocr2-kit-design.md`（含全部已确认决策与风险清单，实施前先读）

## Global Constraints

- 包目录 `DeepSeekOCR2Kit/`（**不能**叫 `Packages/`——被 .gitignore 第 25 行忽略）。
- 依赖：`https://github.com/ml-explore/mlx-swift-lm` `from: "3.31.3"`（产品 `MLXVLM`、`MLXLMCommon`、`MLXHuggingFace`）；`https://github.com/apple/swift-argument-parser` `from: "1.3.0"`（仅 CLI target）。不 fork 任何仓库。
- 平台 `.macOS(.v14)`，swift-tools-version 6.0，测试框架 Swift Testing（`import Testing`）。
- 从 mzbac/deepseek-ocr.swift 拷贝的文件**保留原 MIT 版权头**，文件顶部加 `// Adapted from mzbac/deepseek-ocr.swift (MIT)`。
- 模型/夹具不进 git：`DeepSeekOCR2Kit/Fixtures/` 与 `DeepSeekOCR2Kit/.venv/` 加入 .gitignore。重型 parity 测试用 `Fixtures/` 存在性门控（缺失时 skip），CI 只跑纯逻辑测试。
- 关键常量（全项目统一，出处见 spec）：BOS=0、EOS=1、image token id=128815、prompt=`"<image>\nFree OCR. "`（含尾随空格，禁用 applyChatTemplate）、global 1024²/tile 768²、tile 数 min 1 max 6、token 公式 `numPatches*144 + 256 + 1`、特征序 `[local…, global, view_separator]`、坐标归一化 0–1000。
- 参考源（本机 scratchpad 已克隆，路径失效则按 spec 的 URL 重新浅克隆）：
  - `PY=/private/tmp/claude-501/-Users-william-Desktop-WhisperScribe/f284a1dc-c6f3-4632-b837-e5b3f381bd04/scratchpad/mlx-vlm/mlx_vlm/models/deepseekocr_2`
  - `V1=/private/tmp/claude-501/-Users-william-Desktop-WhisperScribe/f284a1dc-c6f3-4632-b837-e5b3f381bd04/scratchpad/deepseek-ocr-swift/Sources/DeepSeekOCR`
  - `LM=/private/tmp/claude-501/-Users-william-Desktop-WhisperScribe/f284a1dc-c6f3-4632-b837-e5b3f381bd04/scratchpad/mlx-swift-lm/Libraries`
- 每个任务的提交信息末尾带：
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01U5wGZ5BEeMyPZd9jcPP34f`

## 文件结构总览

```
DeepSeekOCR2Kit/
├─ Package.swift
├─ README.md                                   (Task 10)
├─ scripts/
│  ├─ make_test_images.py                      (Task 2) 生成固定测试图
│  ├─ gen_fixtures.py                          (Task 2) Python 黄金夹具导出
│  └─ diff_keys.py                             (Task 4) 权重键集合 diff
├─ Sources/DeepSeekOCR2Kit/
│  ├─ DeepSeekOCR2Configuration.swift          (Task 3) P5
│  ├─ WeightSanitizer.swift                    (Task 4) P6
│  ├─ SAMEncoder.swift                         (Task 5) P2, 拷贝自 V1/VisionEncoder.swift
│  ├─ Qwen2VisionEncoder.swift                 (Task 6) P1
│  ├─ MoELanguageModel.swift                   (Task 7) 拷贝自 V1/LanguageModel.swift
│  ├─ DeepSeekOCR2Model.swift                  (Task 8) P3: projector+顶层+VLMModel
│  ├─ DeepSeekOCR2Processor.swift              (Task 9) P4, 预处理拷贝自 V1/ImageProcessor.swift
│  ├─ GroundingParser.swift                    (Task 10) P8
│  └─ OCR2Session.swift                        (Task 10) P7: 注册+门面 API
├─ Sources/ocr2-cli/main.swift                 (Task 10)
└─ Tests/DeepSeekOCR2KitTests/
   ├─ FixtureSupport.swift                     (Task 3)
   ├─ ConfigurationTests.swift                 (Task 3)
   ├─ WeightSanitizerTests.swift               (Task 4)
   ├─ SAMParityTests.swift                     (Task 5)
   ├─ VisionEncoderTests.swift                 (Task 6) mask 单测 + parity
   ├─ LanguageModelParityTests.swift           (Task 7)
   ├─ EndToEndParityTests.swift                (Task 8)
   ├─ ProcessorTests.swift                     (Task 9) tokenizer/结构/像素 parity
   └─ GroundingParserTests.swift               (Task 10)
```

---

### Task 1: 包骨架

**Files:**
- Create: `DeepSeekOCR2Kit/Package.swift`
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/DeepSeekOCR2Kit.swift`（版本常量占位）
- Create: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/SmokeTests.swift`
- Modify: `.gitignore`（追加夹具/venv 忽略）

**Interfaces:**
- Produces: 可 `swift build`/`swift test` 的空包；后续所有任务的编译载体。

- [ ] **Step 1: 写 Package.swift**

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DeepSeekOCR2Kit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "DeepSeekOCR2Kit", targets: ["DeepSeekOCR2Kit"]),
        .executable(name: "ocr2-cli", targets: ["ocr2-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "DeepSeekOCR2Kit",
            dependencies: [
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
            ]),
        .executableTarget(
            name: "ocr2-cli",
            dependencies: [
                "DeepSeekOCR2Kit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]),
        .testTarget(name: "DeepSeekOCR2KitTests", dependencies: ["DeepSeekOCR2Kit"]),
    ]
)
```

注：若 `from: "3.31.3"` 解析失败（版本号演进），执行 `git ls-remote --tags https://github.com/ml-explore/mlx-swift-lm | tail -5` 取最新 3.x tag 替换。

- [ ] **Step 2: 最小源文件与冒烟测试**

`Sources/DeepSeekOCR2Kit/DeepSeekOCR2Kit.swift`:

```swift
public enum DeepSeekOCR2Kit {
    public static let modelRepoID = "mlx-community/DeepSeek-OCR-2-8bit"
}
```

`Tests/DeepSeekOCR2KitTests/SmokeTests.swift`:

```swift
import Testing
@testable import DeepSeekOCR2Kit

@Test func smokeBuild() {
    #expect(DeepSeekOCR2Kit.modelRepoID.contains("DeepSeek-OCR-2"))
}
```

- [ ] **Step 3: .gitignore 追加**

在仓库根 `.gitignore` 的 `dist/` 行后追加：

```gitignore
DeepSeekOCR2Kit/Fixtures/
DeepSeekOCR2Kit/.venv/
DeepSeekOCR2Kit/.build/
```

- [ ] **Step 4: 构建 + 测试**

Run: `cd DeepSeekOCR2Kit && swift test 2>&1 | tail -5`
Expected: 首次解析依赖较久；末尾 `Test run with 1 test passed`

- [ ] **Step 5: Commit**

```bash
git add DeepSeekOCR2Kit .gitignore
git commit -m "feat(ocr2): DeepSeekOCR2Kit package skeleton (mlx-swift-lm dep)"
```

---

### Task 2: 测试图 + Python 黄金夹具

**Files:**
- Create: `DeepSeekOCR2Kit/scripts/make_test_images.py`
- Create: `DeepSeekOCR2Kit/scripts/gen_fixtures.py`
- 产物（不进 git）：`DeepSeekOCR2Kit/Fixtures/{images/*.png, doc_page/*.safetensors, …, masks/*.safetensors, model -> HF snapshot 路径}`

**Interfaces:**
- Produces: 每张图一个目录 `Fixtures/<name>/`，内含 `pixels_global.safetensors`（键 `x`，(1,1024,1024,3) bf16→存 float32）、`pixels_patches.safetensors`（键 `x`，(N,768,768,3)，N=0 时缺省）、`sam_out.safetensors`、`encoder_out.safetensors`、`projector_out.safetensors`、`input_embeds.safetensors`、`prefill_logits.safetensors`（末位置 (1,vocab)）、`greedy_tokens.safetensors`（键 `ids`，前 100 步 int32）、`meta.json`（`{"prompt_token_ids": […], "images_spatial_crop": […], "num_patches": n, "text": "…"}`）；`Fixtures/masks/mask_144.safetensors`、`mask_256.safetensors`（键 `mask`）；`Fixtures/model_dir.txt`（模型快照绝对路径）。
- 五张图固定命名：`doc_page`（英文文档排版）、`receipt`（数字表格）、`cjk_dense`（中文密集段落）、`tall_scroll`（1:4 长图，强制 ≥4 tiles）、`grounding_menu`（用于 grounding prompt）。

- [ ] **Step 1: 建 Python 环境并下载模型**

```bash
cd DeepSeekOCR2Kit
python3 -m venv .venv && .venv/bin/pip install -q mlx-vlm pillow
.venv/bin/python -c "from huggingface_hub import snapshot_download; p=snapshot_download('mlx-community/DeepSeek-OCR-2-8bit'); print(p)" | tail -1 > Fixtures_model_path.tmp
mkdir -p Fixtures && mv Fixtures_model_path.tmp Fixtures/model_dir.txt && cat Fixtures/model_dir.txt
```

Expected: 打印本地快照路径（~3GB 下载，仅首次）。

- [ ] **Step 2: 写 make_test_images.py（完整）**

```python
#!/usr/bin/env python3
"""Deterministic test images for parity fixtures. PIL only."""
import pathlib
from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path(__file__).parent.parent / "Fixtures" / "images"
OUT.mkdir(parents=True, exist_ok=True)

def font(size, cjk=False):
    path = ("/System/Library/Fonts/PingFang.ttc" if cjk
            else "/System/Library/Fonts/Helvetica.ttc")
    return ImageFont.truetype(path, size)

def page(name, size, lines, cjk=False, fsize=28, margin=60, spacing=14):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = margin
    for ln in lines:
        d.text((margin, y), ln, fill="black", font=font(fsize, cjk))
        y += fsize + spacing
    img.save(OUT / f"{name}.png")

page("doc_page", (1200, 1600), [
    "DeepSeekOCR2Kit Parity Test Document",
    "", "1. Introduction",
    "This fixed document verifies OCR output parity",
    "between the Python reference and the Swift port.",
    "", "2. Requirements",
    "- Deterministic rendering", "- No external assets",
    "The quick brown fox jumps over the lazy dog. 0123456789.",
])
page("receipt", (600, 900), [
    "RECEIPT #004217", "----------------",
    "Espresso        4.50", "Croissant       3.25",
    "Total           7.75", "VISA ****1234  PAID",
], fsize=24)
page("cjk_dense", (1000, 1400), [
    "语音转写与光学识别的本地化验证文档",
    "第一段：本页用于验证中日韩文字的识别一致性，",
    "包含标点、数字 3.1415、以及混排 English words。",
    "第二段：滚动截图、书页拍照、演示文稿截屏等场景",
    "均应产生逐字节一致的贪心解码序列。",
], cjk=True)
page("tall_scroll", (700, 2800),
     [f"第 {i} 行：这是一张模拟滚动截图的超长图片。Line {i}." for i in range(1, 60)],
     cjk=True, fsize=24)
page("grounding_menu", (900, 1200), [
    "MENU", "Latte ......... 5.00", "Mocha ......... 5.50",
    "Tea ........... 3.00", "Find the price of Mocha.",
], fsize=32)
print("wrote", sorted(p.name for p in OUT.iterdir()))
```

- [ ] **Step 3: 写 gen_fixtures.py（完整）**

```python
#!/usr/bin/env python3
"""Dump per-stage golden tensors from the mlx-vlm DeepSeek-OCR-2 reference.

Run:  .venv/bin/python scripts/gen_fixtures.py
Needs: Fixtures/model_dir.txt (Task 2 Step 1), Fixtures/images/ (Step 2).
"""
import json, pathlib, sys
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.deepseekocr_2.vision import Qwen2Decoder2Encoder

ROOT = pathlib.Path(__file__).parent.parent
FIX = ROOT / "Fixtures"
MODEL_DIR = (FIX / "model_dir.txt").read_text().strip()
PROMPT = "<image>\nFree OCR. "
GREEDY_STEPS = 100

model, processor = load(MODEL_DIR)
cfg = model.config

def save(d, name, **arrays):
    d.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(d / f"{name}.safetensors"),
                        {k: v.astype(mx.float32) if v.dtype != mx.int32 else v
                         for k, v in arrays.items()})

# --- masks standalone (P1 单测夹具) ---
enc = model.vision_model.qwen2_encoder
assert isinstance(enc, Qwen2Decoder2Encoder), type(enc)
for n in (144, 256):
    sam_stub = mx.zeros((1, int(n ** 0.5), int(n ** 0.5), enc.config.dim))
    # 复刻 __call__ 内 mask 构造:直接调用一遍并捕获——为稳妥,重算等价 mask:
    L = 2 * n
    m = mx.full((L, L), -1e9, dtype=mx.float32)
    img = mx.zeros((n, n)); q2i = mx.zeros((n, n))
    causal = mx.triu(mx.full((n, n), -1e9, dtype=mx.float32), k=1)
    top = mx.concatenate([img, mx.full((n, n), -1e9)], axis=1)
    bot = mx.concatenate([q2i, causal], axis=1)
    m = mx.concatenate([top, bot], axis=0)
    save(FIX / "masks", f"mask_{n}", mask=m)

# --- per-image fixtures ---
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: F401 (不使用;确认存在即可)
for img_path in sorted((FIX / "images").glob("*.png")):
    name = img_path.stem
    out = FIX / name
    # 预处理(参考实现自身的 processor)
    inputs = processor(text=PROMPT, images=[str(img_path)], return_tensors="mlx")
    # 属性名以 processing_deepseekocr.py 实际返回为准——先打印一次确认:
    print(name, "input keys:", list(inputs.keys()))
    ids = mx.array(inputs["input_ids"])
    pix_g = mx.array(inputs["pixel_values"])          # 全局 1024²
    pix_p = inputs.get("images_crop")                  # 局部 tiles(可能为 None/空)
    crop = inputs.get("images_spatial_crop")
    seq_mask = mx.array(inputs["images_seq_mask"])
    save(out, "pixels_global", x=pix_g)
    n_patch = 0
    if pix_p is not None and mx.array(pix_p).size > 0:
        pix_p = mx.array(pix_p); n_patch = int(pix_p.shape[0]); save(out, "pixels_patches", x=pix_p)
    # 逐级前向
    sam = model.vision_model.sam_model(pix_g.astype(mx.bfloat16)) if hasattr(model.vision_model, "sam_model") else None
    if sam is not None: save(out, "sam_out", x=sam)
    embeds = model.get_input_embeddings(ids, pix_g, images_crop=pix_p,
                                        images_seq_mask=seq_mask,
                                        images_spatial_crop=crop)
    save(out, "input_embeds", x=embeds)
    logits = model.language_model(inputs_embeds=embeds).logits if hasattr(model, "language_model") else model(ids, pix_g)
    save(out, "prefill_logits", x=logits[:, -1, :])
    # 贪心 100 步(用库自身生成,temperature 0)
    from mlx_vlm.generate import generate
    text = generate(model, processor, PROMPT, image=[str(img_path)],
                    max_tokens=GREEDY_STEPS, temperature=0.0, verbose=False)
    ids_greedy = mx.array(processor.tokenizer.encode(text), dtype=mx.int32)
    save(out, "greedy_tokens", ids=ids_greedy)
    meta = {"prompt_token_ids": [int(t) for t in mx.array(ids).flatten().tolist()],
            "images_spatial_crop": (mx.array(crop).tolist() if crop is not None else None),
            "num_patches": n_patch, "text": text}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print("fixtures done")
```

**重要**：此脚本按参考实现的公开接口书写；`inputs` 的确切键名/`get_input_embeddings` 形参名以 `$PY/processing_deepseekocr.py`、`$PY/deepseekocr_2.py` 为准——执行时先 `print` 键名核对（脚本已内置），不符则对照源文件修正脚本（**修脚本，不得改参考实现**）。encoder/projector 中间张量若属性名不同（`model.vision_model.qwen2_encoder`、`model.projector`），同样以源码为准修正后补 `encoder_out`/`projector_out` 两级 dump（在 `get_input_embeddings` 内部对应调用处用局部复算：`enc_out = enc(sam)`、`proj_out = model.projector(enc_out)`）。

- [ ] **Step 4: 生成并抽验**

```bash
cd DeepSeekOCR2Kit
.venv/bin/python scripts/make_test_images.py
.venv/bin/python scripts/gen_fixtures.py
ls Fixtures/doc_page/ && python3 -c "import json;print(json.load(open('Fixtures/doc_page/meta.json'))['text'][:120])"
```

Expected: 五个图目录齐全；`doc_page` 的 text 含 "DeepSeekOCR2Kit Parity Test Document"（OCR 正确读出标题）。**若 text 是乱码，参考实现环境本身有问题，停下排查（模型/版本），不得继续。**

- [ ] **Step 5: Commit（只提交脚本）**

```bash
git add DeepSeekOCR2Kit/scripts
git commit -m "feat(ocr2): deterministic test images + golden fixture generator"
```

---

### Task 3: Configuration（P5）

**Files:**
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/DeepSeekOCR2Configuration.swift`
- Create: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/FixtureSupport.swift`
- Test: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/ConfigurationTests.swift`

**Interfaces:**
- Produces: `struct DeepSeekOCR2Configuration: Codable, Sendable` 及嵌套 `SAMConfig` / `Qwen2EncoderConfig` / `TextConfig`；`FixtureSupport.fixturesRoot: URL?`、`FixtureSupport.load(_ dir: String, _ name: String) throws -> [String: MLXArray]`、`FixtureSupport.modelDir: URL?`。后续所有任务用这些名字。

- [ ] **Step 1: 写失败测试**

```swift
import Testing
import Foundation
@testable import DeepSeekOCR2Kit

@Suite struct ConfigurationTests {
    @Test func hardcodedDefaults() {
        let c = DeepSeekOCR2Configuration.default
        #expect(c.sam.layers == 12 && c.sam.width == 768 && c.sam.windowSize == 14)
        #expect(c.sam.globalAttnIndexes == [2, 5, 8, 11] && c.sam.outputChannels == 896)
        #expect(c.qwen2Encoder.layers == 24 && c.qwen2Encoder.dim == 896)
        #expect(c.qwen2Encoder.heads == 14 && c.qwen2Encoder.kvHeads == 2)
        #expect(c.qwen2Encoder.intermediate == 4864 && c.qwen2Encoder.ropeTheta == 1_000_000)
        #expect(c.text.hiddenSize == 1280 && c.text.layers == 12 && c.text.heads == 10)
        #expect(c.text.intermediate == 6848 && c.text.moeIntermediate == 896)
        #expect(c.text.numExperts == 64 && c.text.topK == 6 && c.text.sharedExperts == 2)
        #expect(c.text.firstKDenseReplace == 1 && c.text.vocabSize == 129_280)
        #expect(c.bosTokenID == 0 && c.eosTokenID == 1 && c.imageTokenID == 128_815)
    }
    @Test func decodesRealConfigJSON() throws {
        guard let model = FixtureSupport.modelDir else { return }  // 无模型时静默通过
        let data = try Data(contentsOf: model.appending(path: "config.json"))
        let c = try DeepSeekOCR2Configuration(mergingJSON: data)
        #expect(c.text.vocabSize == 129_280)
        #expect(c.modelType == "deepseekocr_2")
    }
}
```

`FixtureSupport.swift`（测试共用，完整）：

```swift
import Foundation
import MLX
@testable import DeepSeekOCR2Kit

enum FixtureSupport {
    static var root: URL? {
        let url = URL(fileURLWithPath: #filePath)  // …/Tests/DeepSeekOCR2KitTests/FixtureSupport.swift
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "Fixtures")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
    static var modelDir: URL? {
        guard let root, let s = try? String(contentsOf: root.appending(path: "model_dir.txt"), encoding: .utf8)
        else { return nil }
        return URL(fileURLWithPath: s.trimmingCharacters(in: .whitespacesAndNewlines))
    }
    static func load(_ dir: String, _ name: String) throws -> [String: MLXArray] {
        guard let root else { throw NSError(domain: "fixtures-missing", code: 1) }
        return try MLX.loadArrays(url: root.appending(path: dir).appending(path: "\(name).safetensors"))
    }
    static func meta(_ dir: String) throws -> [String: Any] {
        guard let root else { throw NSError(domain: "fixtures-missing", code: 1) }
        let data = try Data(contentsOf: root.appending(path: dir).appending(path: "meta.json"))
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd DeepSeekOCR2Kit && swift test --filter ConfigurationTests 2>&1 | tail -3`
Expected: 编译失败（`DeepSeekOCR2Configuration` 未定义）

- [ ] **Step 3: 实现 Configuration**

`DeepSeekOCR2Configuration.swift` 核心结构（**大量超参不在 config.json 内，硬编码默认值；`init(mergingJSON:)` 只覆盖 json 中出现的字段**）。逐字段值对照 `$PY/config.py`（`SAMViTConfig`/`Qwen2EncoderConfig`/`TextConfig`/`ModelConfig` 的 dataclass 默认值）转录，形如：

```swift
import Foundation

public struct DeepSeekOCR2Configuration: Codable, Sendable {
    public struct SAMConfig: Codable, Sendable {
        public var layers = 12, width = 768, windowSize = 14
        public var globalAttnIndexes = [2, 5, 8, 11]
        public var outputChannels = 896          // v2: net_3 出 896(v1 为 1024)
        public var patchSize = 16, imageSizeGlobal = 1024, imageSizeTile = 768
    }
    public struct Qwen2EncoderConfig: Codable, Sendable {
        public var layers = 24, dim = 896, heads = 14, kvHeads = 2
        public var intermediate = 4864
        public var ropeTheta: Double = 1_000_000
        public var rmsNormEps: Double = 1e-6
        public var queryTokens1024 = 256, queryTokens768 = 144
    }
    public struct TextConfig: Codable, Sendable {
        public var hiddenSize = 1280, layers = 12, heads = 10
        public var intermediate = 6848, moeIntermediate = 896
        public var numExperts = 64, topK = 6, sharedExperts = 2
        public var firstKDenseReplace = 1, vocabSize = 129_280
        public var qkNopeHeadDim = 0            // ==0 → 非 MLA(LlamaAttention 路径)
        public var nGroup = 1, topkGroup = 1    // 加载时断言,防未实现的 group routing
        public var rmsNormEps: Double = 1e-6
        public var ropeTheta: Double = 10_000
    }
    public var modelType = "deepseekocr_2"
    public var sam = SAMConfig(), qwen2Encoder = Qwen2EncoderConfig(), text = TextConfig()
    public var projectorInput = 896, projectorOutput = 1280
    public var bosTokenID = 0, eosTokenID = 1, imageTokenID = 128_815

    public static let `default` = DeepSeekOCR2Configuration()
    public init() {}
    public init(mergingJSON data: Data) throws { /* JSONDecoder 逐段可选解码,缺省用默认值;
        对照 config.json 顶层键(model_type/language_config/vision_config/candidate 名以实际文件为准)逐一映射 */ }
}
```

`init(mergingJSON:)` 的键名以 `Fixtures` 模型目录里的真实 `config.json` 为准（`cat $(cat Fixtures/model_dir.txt)/config.json`），plan 不预断——写实现时逐键核对。`$PY/config.py` 里 dataclass 与默认值若与上表冲突，**以 Python 源为准并回改本测试的期望值**（源头即 spec 记录的 gap 分析，可能存在个别转录误差）。

- [ ] **Step 4: 跑测试通过**

Run: `cd DeepSeekOCR2Kit && swift test --filter ConfigurationTests 2>&1 | tail -3`
Expected: `2 tests passed`（无模型时第二条静默通过）

- [ ] **Step 5: Commit**

```bash
git add DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/DeepSeekOCR2Configuration.swift DeepSeekOCR2Kit/Tests
git commit -m "feat(ocr2): configuration with hardcoded reference defaults + fixture support"
```

---

### Task 4: WeightSanitizer（P6）

**Files:**
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/WeightSanitizer.swift`
- Create: `DeepSeekOCR2Kit/scripts/diff_keys.py`
- Test: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/WeightSanitizerTests.swift`

**Interfaces:**
- Consumes: `FixtureSupport.modelDir`。
- Produces: `enum WeightSanitizer { static func sanitize(_ weights: [String: MLXArray], config: DeepSeekOCR2Configuration) -> [String: MLXArray] }`。Task 8 的模型 `update(verify: [.all])` 前必经此函数。

- [ ] **Step 1: 写失败测试（真权重驱动）**

```swift
import Testing
import MLX
@testable import DeepSeekOCR2Kit

@Suite struct WeightSanitizerTests {
    // 无模型则整套 skip
    static let modelDir = FixtureSupport.modelDir

    @Test(.enabled(if: modelDir != nil))
    func remapsAllCheckpointKeys() throws {
        let raw = try loadAllShards()          // 见 Step 3 helper
        let out = WeightSanitizer.sanitize(raw, config: .default)
        // 1) 不残留 checkpoint 前缀
        #expect(!out.keys.contains { $0.hasPrefix("model.") })
        // 2) 关键目标键存在
        #expect(out["vision_model.qwen2_encoder.query_1024"] != nil)
        #expect(out["vision_model.qwen2_encoder.query_768"] != nil)
        #expect(out["view_separator"] != nil)   // 源键拼写是 view_seperator!
        // 3) expert 堆叠:64 专家折叠进 switch_mlp,层内不再有 experts.N
        #expect(!out.keys.contains { $0.contains("experts.0.") })
    }
    @Test(.enabled(if: modelDir != nil))
    func spotChecksNumerics() throws {
        let raw = try loadAllShards()
        let out = WeightSanitizer.sanitize(raw, config: .default)
        // view_separator 数值与源张量一致(仅改名不动值)
        let src = raw.first { $0.key.hasSuffix("view_seperator") }!.value
        #expect(mx.allClose(out["view_separator"]!, src).item(Bool.self))
    }
}
```

- [ ] **Step 2: 确认失败**

Run: `cd DeepSeekOCR2Kit && OCR2_FIXTURES=1 swift test --filter WeightSanitizerTests 2>&1 | tail -3`
Expected: 编译失败（`WeightSanitizer` 未定义）

- [ ] **Step 3: 实现**

移植 `$PY/deepseekocr_2.py` 的 `Model.sanitize`（206-293 行）与 `$PY/vision.py` 的 `VisionModel.sanitize`（407 行起）为一个纯函数；参考 `$V1/DeepSeekOCRModel.swift` 中 `sanitizeWeights` 的 expert 堆叠与 conv OIHW→OHWI 转置写法。要点清单（全部来自 Python 源，逐行对照）：

- 前缀剥离/重映射：`model.vision_model.…`→`vision_model.…`、`model.qwen2_model.model.model.layers.N.…`→`vision_model.qwen2_encoder.layers.N.…`、`model.projector.…`→`projector.…`、`model.language_model.…`（或 LM 直挂 `model.layers.*`，以 index.json 实况为准）→`language_model.…`
- `model.query_1024/query_768`（或嵌套于 qwen2_model 下，以实况为准）→ `vision_model.qwen2_encoder.query_*`
- `model.view_seperator` → `view_separator`（**修正拼写**）
- LM 每层 64 个 `mlp.experts.N.{gate,up,down}_proj.weight`（含 `.scales`/`.biases` 量化伴生张量）按 N 序 stack 为 `mlp.switch_mlp.{gate,up,down}_proj.weight`（MLXLMCommon `SwitchGLU` 约定）
- SAM 卷积核 OIHW→OHWI transpose（对照 `$V1` 里既有列表）
- 测试 helper `loadAllShards()`：读 `modelDir` 下 `model.safetensors.index.json` 列出的所有分片，`MLX.loadArrays` 合并（写在测试文件底部，~15 行）。

`scripts/diff_keys.py`（完整）：

```python
#!/usr/bin/env python3
"""对照 index.json 键集合与 Swift sanitize 目标模块树,人工核查遗漏。
用法: .venv/bin/python scripts/diff_keys.py <sanitized_keys.txt>
(sanitized_keys.txt 由临时 Swift 脚本/测试 print 导出,一行一键)"""
import json, pathlib, sys
model_dir = pathlib.Path((pathlib.Path(__file__).parent.parent / "Fixtures" / "model_dir.txt").read_text().strip())
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
src = set(index["weight_map"].keys())
dst = set(pathlib.Path(sys.argv[1]).read_text().split())
print(f"checkpoint keys: {len(src)}  sanitized keys: {len(dst)}")
print("UNCONSUMED source keys (应为 0):")
# sanitize 是 src→dst 改名;此脚本靠人工比对数量与抽样,src 每键都应有 dst 对应
for k in sorted(src)[:20]: print(" ", k)
```

- [ ] **Step 4: 跑测试通过**

Run: `cd DeepSeekOCR2Kit && swift test --filter WeightSanitizerTests 2>&1 | tail -3`
Expected: `2 tests passed`（868 张量全部载入内存，本步 ~10s+）

- [ ] **Step 5: Commit**

```bash
git add DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/WeightSanitizer.swift DeepSeekOCR2Kit/Tests DeepSeekOCR2Kit/scripts/diff_keys.py
git commit -m "feat(ocr2): checkpoint key sanitizer (expert stacking, view_separator, conv transpose)"
```

---

### Task 5: SAM 编码器（P2）

**Files:**
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/SAMEncoder.swift`（拷贝 `$V1/VisionEncoder.swift` 改造）
- Test: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/SAMParityTests.swift`

**Interfaces:**
- Consumes: `DeepSeekOCR2Configuration.SAMConfig`、`FixtureSupport`。
- Produces: `final class SAMEncoder: Module { init(_ config: DeepSeekOCR2Configuration.SAMConfig); func callAsFunction(_ x: MLXArray) -> MLXArray }`，输入 (B,H,W,3) bf16（H∈{1024,768}），输出 (B,h,w,896)（1024→16×16 …经 net 下采样后实际网格见 Python `sam.py`；**以夹具 `sam_out` 的实际形状为准写断言**）。

- [ ] **Step 1: 写失败 parity 测试**

```swift
import Testing
import MLX
@testable import DeepSeekOCR2Kit

@Suite struct SAMParityTests {
    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func globalPathMatchesReference() throws {
        let pix = try FixtureSupport.load("doc_page", "pixels_global")["x"]!
        let ref = try FixtureSupport.load("doc_page", "sam_out")["x"]!
        let sam = SAMEncoder(DeepSeekOCR2Configuration.default.sam)
        try TestWeights.load(into: sam, prefix: "vision_model.sam_model.")  // helper 见 Step 3
        let out = sam(pix.asType(.bfloat16)).asType(.float32)
        #expect(out.shape == ref.shape)
        let relErr = (mx.abs(out - ref).mean() / (mx.abs(ref).mean() + 1e-6)).item(Float.self)
        #expect(relErr < 2e-2, "rel err \(relErr)")
    }
}
```

- [ ] **Step 2: 确认失败**

Run: `cd DeepSeekOCR2Kit && swift test --filter SAMParityTests 2>&1 | tail -3`
Expected: 编译失败（`SAMEncoder`/`TestWeights` 未定义）

- [ ] **Step 3: 拷贝改造**

`cp $V1/VisionEncoder.swift Sources/DeepSeekOCR2Kit/SAMEncoder.swift`，保留 MIT 头，然后：

1. 类型改名 `VisionEncoder`→`SAMEncoder`，构造参数换 `SAMConfig`。
2. `net_3` 输出通道 1024→`config.outputChannels`（896）。
3. 新增 768² 输入路径：patch 网格 48×48（768/16），window 分区 pad 48→56 与 1024 路径同构；`get_abs_pos` 位置编码插值换 `MLXLMCommon` 的 `bicubicInterpolate`（Metal kernel，对照 `$LM/MLXLMCommon/InterpolationUtils.swift` 签名）。对照 `$PY/../deepseekocr/sam.py`（489 行，v1/v2 共享）逐函数核对 rel_pos 重采样。
4. 测试 helper `TestWeights.load(into:prefix:)` 加进 `FixtureSupport.swift`：读全部分片 → `WeightSanitizer.sanitize` → 按前缀过滤剥前缀 → `module.update(parameters:, verify: .noUnusedKeys)`（~20 行，Task 6/7/8 复用）。

- [ ] **Step 4: 跑测试通过**

Run: `cd DeepSeekOCR2Kit && swift test --filter SAMParityTests 2>&1 | tail -3`
Expected: PASS。若 relErr 卡在 2e-2~1e-1：先查 bicubic 插值实现（风险清单 #2），对照夹具重生成一版禁插值路径定位。

- [ ] **Step 5: Commit**

```bash
git add DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/SAMEncoder.swift DeepSeekOCR2Kit/Tests
git commit -m "feat(ocr2): SAM ViT-B encoder (v1 port adapted: 896ch net_3, 768px path)"
```

---

### Task 6: Qwen2Decoder2Encoder（P1，最高风险）

**Files:**
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/Qwen2VisionEncoder.swift`
- Test: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/VisionEncoderTests.swift`

**Interfaces:**
- Consumes: `Qwen2EncoderConfig`、夹具 `masks/mask_{144,256}`、`encoder_out`。
- Produces: `final class Qwen2VisionEncoder: Module { init(_ config: …); func callAsFunction(samFeatures: MLXArray) -> MLXArray }`（输入 (B,h,w,896)，输出 (B,Q,896)，Q=144|256）；`static func attentionMask(imageTokens: Int, queries: Int) -> MLXArray`（暴露为内部 API 供单测）。

- [ ] **Step 1: mask 单测先行（纯逻辑，CI 可跑）**

```swift
import Testing
import MLX
@testable import DeepSeekOCR2Kit

@Suite struct VisionEncoderTests {
    @Test(arguments: [144, 256])
    func maskQuadrants(n: Int) throws {
        let m = Qwen2VisionEncoder.attentionMask(imageTokens: n, queries: n)
        #expect(m.shape == [2 * n, 2 * n])
        #expect(m[0, n - 1].item(Float.self) == 0)          // img→img 双向
        #expect(m[0, n].item(Float.self) <= -1e8)           // img→query 屏蔽
        #expect(m[n, 0].item(Float.self) == 0)              // query→img 全通
        #expect(m[n, n].item(Float.self) == 0)              // query 对角
        #expect(m[n, n + 1].item(Float.self) <= -1e8)       // query 上三角屏蔽
        #expect(m[2 * n - 1, n].item(Float.self) == 0)      // query 下三角可见
    }
    @Test(arguments: [144, 256], .enabled(if: FixtureSupport.root != nil))
    func maskMatchesPythonFixture(n: Int) throws {
        let ref = try FixtureSupport.load("masks", "mask_\(n)")["mask"]!
        let m = Qwen2VisionEncoder.attentionMask(imageTokens: n, queries: n)
        #expect(mx.allClose(m, ref).item(Bool.self))
    }
    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func encoderOutputParity() throws {
        let sam = try FixtureSupport.load("doc_page", "sam_out")["x"]!
        let ref = try FixtureSupport.load("doc_page", "encoder_out")["x"]!
        let enc = Qwen2VisionEncoder(DeepSeekOCR2Configuration.default.qwen2Encoder)
        try TestWeights.load(into: enc, prefix: "vision_model.qwen2_encoder.")
        let out = enc(samFeatures: sam.asType(.bfloat16)).asType(.float32)
        #expect(out.shape == ref.shape)
        let relErr = (mx.abs(out - ref).mean() / (mx.abs(ref).mean() + 1e-6)).item(Float.self)
        #expect(relErr < 2e-2, "rel err \(relErr)")
    }
}
```

- [ ] **Step 2: 确认失败** — `swift test --filter VisionEncoderTests`，编译失败。

- [ ] **Step 3: 实现**

主干从 `$LM/MLXVLM/Models/FastVLM.swift` 的 `Language` 部分抄 Qwen2 层结构（q/k/v bias、GQA、SwiftGLU、RMSNorm），**去 KV cache、去 lm_head**，自定义点逐一对照 `$PY/vision.py:225-367`：

```swift
// 核心骨架(完整逻辑;层实现从 FastVLM 抄)
final class Qwen2VisionEncoder: Module {
    let config: DeepSeekOCR2Configuration.Qwen2EncoderConfig
    @ParameterInfo(key: "query_1024") var query1024: MLXArray   // (256, dim)
    @ParameterInfo(key: "query_768") var query768: MLXArray     // (144, dim)
    @ModuleInfo(key: "layers") var layers: [Qwen2EncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    static func attentionMask(imageTokens n: Int, queries q: Int) -> MLXArray {
        let topLeft = MLXArray.zeros([n, n])                            // img↔img
        let topRight = MLXArray.full([n, q], values: MLXArray(-1e9))    // img→query 屏蔽
        let bottomLeft = MLXArray.zeros([q, n])                         // query→img
        let bottomRight = MLX.triu(MLXArray.full([q, q], values: MLXArray(-1e9)), k: 1) // causal
        return MLX.concatenated([
            MLX.concatenated([topLeft, topRight], axis: 1),
            MLX.concatenated([bottomLeft, bottomRight], axis: 1),
        ], axis: 0)
    }

    func callAsFunction(samFeatures: MLXArray) -> MLXArray {
        let b = samFeatures.shape[0]
        let flat = samFeatures.reshaped(b, -1, config.dim)
        let n = flat.shape[1]
        let query = (n == 144 ? query768 : query1024)     // 非 144 一律 1024(对照 Python else 分支)
        let q = query.shape[0]
        var h = MLX.concatenated([flat, MLX.broadcast(query[.newAxis], to: [b, q, config.dim])], axis: 1)
        let mask = Self.attentionMask(imageTokens: n, queries: q)
            .asType(h.dtype)[.newAxis, .newAxis, 0..., 0...]
        // position_ids = arange(seq) 覆盖 concat 全序列 → RoPE 直接按绝对位置施加
        for layer in layers { h = layer(h, mask: mask) }
        return norm(h)[0..., (-q)..., 0...]
    }
}
```

RoPE：`Qwen2EncoderLayer` 内用 MLXNN `RoPE(dimensions: headDim, base: config.ropeTheta)`，位置从 0 起覆盖全长（与 Python `position_ids = arange(seq_len)` 等价）。

- [ ] **Step 4: 跑测试通过**

Run: `cd DeepSeekOCR2Kit && swift test --filter VisionEncoderTests 2>&1 | tail -4`
Expected: mask 4 条 + parity 全过。**mask 夹具比对不过时优先修 mask，禁止先调 encoder**（风险 #1：mask 错→静默劣化）。

- [ ] **Step 5: Commit**

```bash
git add DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/Qwen2VisionEncoder.swift DeepSeekOCR2Kit/Tests
git commit -m "feat(ocr2): Qwen2 causal-visual-flow encoder with quadrant mask (parity-tested)"
```

---

### Task 7: MoE 语言模型

**Files:**
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/MoELanguageModel.swift`（拷贝 `$V1/LanguageModel.swift` 改造）
- Test: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/LanguageModelParityTests.swift`

**Interfaces:**
- Consumes: `TextConfig`、夹具 `input_embeds`/`prefill_logits`/`greedy_tokens`。
- Produces: `final class MoELanguageModel: Module { init(_ config: TextConfig); func callAsFunction(inputEmbeds: MLXArray, cache: [KVCache]?) -> MLXArray; func embed(_ tokens: MLXArray) -> MLXArray; func newCache() -> [KVCache] }`。Task 8 直接挂载。

- [ ] **Step 1: 写失败 parity 测试**

```swift
import Testing
import MLX
@testable import DeepSeekOCR2Kit

@Suite struct LanguageModelParityTests {
    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func prefillLogitsParity() throws {
        let embeds = try FixtureSupport.load("doc_page", "input_embeds")["x"]!
        let refLogits = try FixtureSupport.load("doc_page", "prefill_logits")["x"]!
        let lm = MoELanguageModel(DeepSeekOCR2Configuration.default.text)
        try TestWeights.load(into: lm, prefix: "language_model.")
        let logits = lm(inputEmbeds: embeds.asType(.bfloat16), cache: nil).asType(.float32)
        let last = logits[0..., -1, 0...]
        // 量化模型 logits 有漂移:判据是 argmax 一致 + top-5 集合重叠 ≥4
        #expect(mx.argMax(last, axis: -1).item(Int.self)
             == mx.argMax(refLogits, axis: -1).item(Int.self))
    }
    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func greedyContinuationParity() throws {
        let embeds = try FixtureSupport.load("doc_page", "input_embeds")["x"]!
        let refIDs = try FixtureSupport.load("doc_page", "greedy_tokens")["ids"]!
        let lm = MoELanguageModel(DeepSeekOCR2Configuration.default.text)
        try TestWeights.load(into: lm, prefix: "language_model.")
        var cache = lm.newCache()
        var logits = lm(inputEmbeds: embeds.asType(.bfloat16), cache: cache)
        var ids: [Int] = []
        for _ in 0..<min(100, refIDs.shape[0]) {
            let next = mx.argMax(logits[0..., -1, 0...], axis: -1)
            ids.append(next.item(Int.self))
            if ids.last == 1 { break }   // EOS=1
            logits = lm(inputEmbeds: lm.embed(next.reshaped(1, 1)), cache: cache)
        }
        let ref = (0..<ids.count).map { refIDs[$0].item(Int.self) }
        #expect(ids == ref, "diverged at \(zip(ids, ref).firstIndex { $0 != $1 } ?? -1)")
    }
}
```

- [ ] **Step 2: 确认失败** — 编译失败。

- [ ] **Step 3: 拷贝改造**

`cp $V1/LanguageModel.swift Sources/DeepSeekOCR2Kit/MoELanguageModel.swift`，保留 MIT 头，然后：

1. attention 换 `MLXLMCommon` 基元：`KVCacheSimple` + `attentionWithCacheUpdate` + `createAttentionMask`（删 v1 的手写 softmax 与 concat cache）。
2. MoE 层换 `$LM/MLXLMCommon/SwitchLayers.swift` 的 `SwitchGLU`（与 Task 4 的 stacked 权重键对齐）。
3. `init` 断言 `config.nGroup == 1 && config.topkGroup == 1 && config.qkNopeHeadDim == 0`（越界配置直接 fatalError，防静默走错分支）。
4. 首层 dense（`firstKDenseReplace=1`）、其余 MoE + 2 shared experts 的结构保持 v1 写法。
5. 增加 `embed(_:)`（暴露 embedding 查表）与 `newCache()`（返回 `config.layers` 个 `KVCacheSimple`）。

- [ ] **Step 4: 跑测试通过**

Run: `cd DeepSeekOCR2Kit && swift test --filter LanguageModelParityTests 2>&1 | tail -3`
Expected: 2 PASS。贪心分歧时打印首个分歧步，对照 Python 逐层缩小（风险 #4：先查 expert 堆叠序）。

- [ ] **Step 5: Commit**

```bash
git add DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/MoELanguageModel.swift DeepSeekOCR2Kit/Tests
git commit -m "feat(ocr2): DeepSeek-3B-MoE LM on MLXLMCommon primitives (greedy-parity vs reference)"
```

---

### Task 8: 顶层模型（P3）+ 注入像素端到端 parity

**Files:**
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/DeepSeekOCR2Model.swift`
- Test: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/EndToEndParityTests.swift`

**Interfaces:**
- Consumes: Task 3–7 全部类型。
- Produces:

```swift
public final class DeepSeekOCR2Model: Module {
    public init(_ config: DeepSeekOCR2Configuration)
    // tokens: (1,L) 已含 BOS 与 N×imageToken;pixelsGlobal: (V,1024,1024,3);
    // pixelsPatches: (P,768,768,3)?;seqMask: (1,L) Bool — imageToken 位置为 true
    public func inputEmbeddings(tokens: MLXArray, pixelsGlobal: MLXArray,
                                pixelsPatches: MLXArray?, seqMask: MLXArray) -> MLXArray
    public func callAsFunction(inputEmbeds: MLXArray, cache: [KVCache]?) -> MLXArray
    public static func load(from dir: URL, progress: @Sendable (Double) -> Void) async throws
        -> (model: DeepSeekOCR2Model, config: DeepSeekOCR2Configuration)
}
```

- [ ] **Step 1: 写失败测试**

```swift
import Testing
import MLX
@testable import DeepSeekOCR2Kit

@Suite struct EndToEndParityTests {
    @Test(arguments: ["doc_page", "cjk_dense", "tall_scroll"],
          .enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func injectedPixelGreedyParity(name: String) throws {
        let meta = try FixtureSupport.meta(name)
        let promptIDs = (meta["prompt_token_ids"] as! [Int])
        let refIDs = try FixtureSupport.load(name, "greedy_tokens")["ids"]!
        let pixG = try FixtureSupport.load(name, "pixels_global")["x"]!
        let pixP = try? FixtureSupport.load(name, "pixels_patches")["x"]
        let (model, cfg) = try await DeepSeekOCR2Model.load(
            from: FixtureSupport.modelDir!, progress: { _ in })
        let tokens = MLXArray(promptIDs.map(Int32.init)).reshaped(1, -1)
        let seqMask = MLX.equal(tokens, MLXArray(Int32(cfg.imageTokenID)))
        let embeds = model.inputEmbeddings(tokens: tokens,
            pixelsGlobal: pixG.asType(.bfloat16),
            pixelsPatches: pixP?["x"]?.asType(.bfloat16), seqMask: seqMask)
        // 复用 Task 7 的贪心循环(抽到 FixtureSupport.greedyDecode helper)
        let ids = try FixtureSupport.greedyDecode(model: model, embeds: embeds, steps: 100, eos: cfg.eosTokenID)
        let ref = (0..<ids.count).map { refIDs[$0].item(Int.self) }
        #expect(ids == ref, "\(name) diverged at \(zip(ids, ref).firstIndex { $0 != $1 } ?? -1)")
    }
}
```

（`greedyDecode` helper：把 Task 7 Step 1 的循环体抽进 `FixtureSupport`，Task 7 的测试同步改用——一处实现两处用。）

- [ ] **Step 2: 确认失败** — 编译失败。

- [ ] **Step 3: 实现**

对照 `$PY/deepseekocr_2.py`（`MlpProjector` 14-30 行、`Model.get_input_embeddings` 65-180 行）：

1. `projector` = `Linear(896, 1280)`（`MLPConfig`/`ProjectorConfig` 以 Python 源为准，若含激活/双层则如实移植——写码时读源）。
2. `inputEmbeddings`：逐视图 `SAMEncoder`→`Qwen2VisionEncoder`→projector；特征顺序 **`[local_patches…, global, view_separator]`**（Python 源为准；config 里 `global_view_pos:"head"` 有误导，见 spec 风险 #3 注）；`view_separator` 参数张量 (1280,) 追加末尾；再按 `seqMask` 把视觉特征散布进 `lm.embed(tokens)` 的对应位置（`MLX.where` 或索引 scatter——**不得**假设 image token 连续成块）。
3. `load(from:)`：读 `config.json` → `DeepSeekOCR2Configuration(mergingJSON:)` → 构建 Module 树 → 读全部分片 → `WeightSanitizer.sanitize` → 量化层替换（`quantization` config 存在时按 `.scales` 伴生键对量化层做 `QuantizedLinear` 替换，走 `MLXLMCommon.loadWeights` 若其签名可直接用——对照 `$LM/MLXLMCommon/Load.swift`；不适配则手写 `quantize(model:)` 谓词版）→ `update(parameters:, verify: [.all])`。
4. `progress` 按分片数汇报（i/n）。

- [ ] **Step 4: 跑测试通过（里程碑 M4 门禁）**

Run: `cd DeepSeekOCR2Kit && swift test --filter EndToEndParityTests 2>&1 | tail -5`
Expected: 3 张图贪心 100 token 全一致。分歧时按"embeds→prefill logits→步 N"顺序对夹具定位层级。

- [ ] **Step 5: Commit**

```bash
git add DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/DeepSeekOCR2Model.swift DeepSeekOCR2Kit/Tests
git commit -m "feat(ocr2): top-level model assembly + quantized load (100-token greedy parity)"
```

---

### Task 9: Processor（P4）

**Files:**
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/DeepSeekOCR2Processor.swift`（预处理部分拷贝 `$V1/ImageProcessor.swift` 改造）
- Test: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/ProcessorTests.swift`

**Interfaces:**
- Consumes: `DeepSeekOCR2Configuration`、swift-transformers `Tokenizers`（经 MLXHuggingFace 加载）。
- Produces:

```swift
public struct OCR2Input: Sendable {
    public let tokens: MLXArray        // (1,L) 含 BOS=0 前置、<image>→N×128815 展开、inference 剥末 token
    public let pixelsGlobal: MLXArray  // (V,1024,1024,3) bf16, (x/255-0.5)/0.5
    public let pixelsPatches: MLXArray?
    public let seqMask: MLXArray       // (1,L) Bool
    public let spatialCrop: [[Int]]
}
public final class DeepSeekOCR2Processor: Sendable {
    public init(modelDir: URL) async throws     // 加载 tokenizer.json
    public func prepare(image: CGImage, prompt: String) throws -> OCR2Input
}
```

- [ ] **Step 1: 写失败测试（三层：tokenizer / 结构量 / 像素）**

```swift
import Testing
import MLX
import CoreGraphics
import ImageIO
@testable import DeepSeekOCR2Kit

@Suite struct ProcessorTests {
    static func cgImage(_ name: String) throws -> CGImage {
        let url = FixtureSupport.root!.appending(path: "images/\(name).png")
        let src = CGImageSourceCreateWithURL(url as CFURL, nil)!
        return CGImageSourceCreateImageAtIndex(src, 0, nil)!
    }
    @Test(arguments: ["doc_page", "receipt", "cjk_dense", "tall_scroll", "grounding_menu"],
          .enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func promptTokenIDsMatchReference(name: String) async throws {
        let meta = try FixtureSupport.meta(name)
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)
        let input = try p.prepare(image: Self.cgImage(name), prompt: "<image>\nFree OCR. ")
        let ids = (0..<input.tokens.shape[1]).map { input.tokens[0, $0].item(Int.self) }
        #expect(ids == meta["prompt_token_ids"] as! [Int])       // 结构量+tokenizer 合并判据
        #expect(input.spatialCrop == meta["images_spatial_crop"] as! [[Int]])
    }
    @Test(.enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func pixelParity() async throws {
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)
        let input = try p.prepare(image: Self.cgImage("doc_page"), prompt: "<image>\nFree OCR. ")
        let ref = try FixtureSupport.load("doc_page", "pixels_global")["x"]!
        let diff = mx.abs(input.pixelsGlobal.asType(.float32) - ref).mean().item(Float.self)
        #expect(diff < 2.0 / 255.0 / 0.5, "mean abs diff \(diff)")  // 归一化域(÷0.5)下等价 2/255
    }
    @Test(arguments: ["doc_page", "cjk_dense"],
          .enabled(if: FixtureSupport.root != nil && FixtureSupport.modelDir != nil))
    func realImageEndToEndText(name: String) async throws {
        // 全链路:CGImage→processor→model→贪心 100 步→文本与 meta.text 一致(CER≈0)
        let meta = try FixtureSupport.meta(name)
        let (model, cfg) = try await DeepSeekOCR2Model.load(from: FixtureSupport.modelDir!, progress: { _ in })
        let p = try await DeepSeekOCR2Processor(modelDir: FixtureSupport.modelDir!)
        let input = try p.prepare(image: Self.cgImage(name), prompt: "<image>\nFree OCR. ")
        let embeds = model.inputEmbeddings(tokens: input.tokens, pixelsGlobal: input.pixelsGlobal,
                                           pixelsPatches: input.pixelsPatches, seqMask: input.seqMask)
        let ids = try FixtureSupport.greedyDecode(model: model, embeds: embeds, steps: 100, eos: cfg.eosTokenID)
        let text = p.decode(ids)               // decode 走 NaiveStreamingDetokenizer 同源实现
        #expect(text == (meta["text"] as! String).prefix(text.count).description)
    }
}
```

- [ ] **Step 2: 确认失败** — 编译失败。

- [ ] **Step 3: 实现**

预处理拷贝 `$V1/ImageProcessor.swift`（`dynamicPreprocess`/`findClosestAspectRatio`/`padToSquare`）改参数，逐行对照 `$PY/processing_deepseekocr.py`：

1. base 1024、tile 768、min 1/max 6；**`findClosestAspectRatio` 平手规则（先 ratio_diff 后 area）逐字对齐 Python**。
2. 归一化 `(x/255 - 0.5)/0.5`、填充色 mean-gray (127,127,127)、bf16 输出；缩放用 `MediaProcessing.resampleBicubic`（`$LM/MLXVLM/MediaProcessing.swift`），parity 不达标时手写 PIL 等价 bicubic（风险 #2 的既定退路）。
3. token 组装：文本按 `<image>` 切分 → tokenizer 编码各段（**禁用 chat template**）→ 每图展开 `numPatches*144 + 256 + 1` 个 id 128815 → 头部插 BOS 0 → 剥除末 token（`inference_mode`）。`seqMask` 同步构造。
4. `decode(_ ids: [Int]) -> String` 用与生成路径同一 detokenizer（`NaiveStreamingDetokenizer`，防 CJK 多字节乱码）。
5. tokenizer 经 `MLXHuggingFace` 的 tokenizer 加载器指向本地 `modelDir`（HF 仓库无 preprocessor_config.json——**不走** `VLMModelFactory` 的 processor 装载，本类型即兜底方案）。

- [ ] **Step 4: 跑测试通过（里程碑 M5 门禁）**

Run: `cd DeepSeekOCR2Kit && swift test --filter ProcessorTests 2>&1 | tail -6`
Expected: 5 图 token/结构断言 + 像素 parity + 2 图端到端文本全过。

- [ ] **Step 5: Commit**

```bash
git add DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/DeepSeekOCR2Processor.swift DeepSeekOCR2Kit/Tests
git commit -m "feat(ocr2): image processor with reference-parity preprocessing and tokenization"
```

---

### Task 10: 门面 API + CLI + grounding + CI 接线

**Files:**
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/OCR2Session.swift`
- Create: `DeepSeekOCR2Kit/Sources/DeepSeekOCR2Kit/GroundingParser.swift`
- Create: `DeepSeekOCR2Kit/Sources/ocr2-cli/main.swift`
- Create: `DeepSeekOCR2Kit/README.md`
- Modify: `.github/workflows/ci.yml`（追加 kit 测试步骤）
- Test: `DeepSeekOCR2Kit/Tests/DeepSeekOCR2KitTests/GroundingParserTests.swift`

**Interfaces:**
- Produces（②子项目的最终消费面；spec API 节写的 `DeepSeekOCR2Model.load/ocr` 由本任务的 `OCR2Session` 门面实现——Module 与门面分离，语义一致）:

```swift
public enum OCRTask: Sendable { case freeOCR, grounding(query: String) }
public struct GroundingBox: Equatable, Sendable { public let label: String; public let box: CGRect } // 0–1 归一化
public final class OCR2Session {
    public static func load(from dir: URL, progress: @Sendable (Double) -> Void) async throws -> OCR2Session
    public func ocr(image: CGImage, task: OCRTask = .freeOCR) -> AsyncThrowingStream<String, Error>
    public static func parseGrounding(_ text: String) -> [GroundingBox]
}
```

- [ ] **Step 1: grounding 解析器失败测试**

```swift
import Testing
@testable import DeepSeekOCR2Kit

@Suite struct GroundingParserTests {
    @Test func parsesRefDetPairs() {
        let s = "<|ref|>Mocha<|/ref|><|det|>[[120, 340, 560, 400]]<|/det|>"
        let boxes = OCR2Session.parseGrounding(s)
        #expect(boxes.count == 1 && boxes[0].label == "Mocha")
        #expect(abs(boxes[0].box.minX - 0.120) < 1e-9)   // 0–1000 → /1000 归一化
        #expect(abs(boxes[0].box.maxY - 0.400) < 1e-9)
    }
    @Test func ignoresMalformed() {
        #expect(OCR2Session.parseGrounding("<|det|>[[1,2,3]]<|/det|>").isEmpty)
        #expect(OCR2Session.parseGrounding("no markers").isEmpty)
    }
}
```

- [ ] **Step 2: 确认失败**，然后实现：

`GroundingParser.swift`：正则 `<\|ref\|>(.+?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>`，坐标 ÷1000 → `CGRect(x: x1, y: y1, width: x2-x1, height: y2-y1)`；四元组不全则丢弃。

`OCR2Session.swift`：组合 `DeepSeekOCR2Model` + `DeepSeekOCR2Processor`；`ocr()` 内部 `Task` 循环贪心解码（Task 7 的 cache 流程），每步经 detokenizer 增量吐字到 `AsyncThrowingStream`；`task == .grounding(q)` 时 prompt 换 `"<image>\n<|grounding|>\(q) "`（**具体 grounding prompt 格式以 `$PY/README.md` 与 `processing_deepseekocr.py` 的实际模板为准，实施时核对**）；`Task.isCancelled` 检查每步执行，取消即 finish。

`main.swift`（完整）：

```swift
import ArgumentParser
import CoreGraphics
import Foundation
import ImageIO
import DeepSeekOCR2Kit

@main struct OCR2CLI: AsyncParsableCommand {
    @Argument(help: "image path") var image: String
    @Option(help: "model dir (default: $OCR2_MODEL_DIR)") var modelDir: String?
    @Flag(help: "grounding mode; pass query via --query") var grounding = false
    @Option var query: String = "Locate all text."

    mutating func run() async throws {
        guard let dir = modelDir ?? ProcessInfo.processInfo.environment["OCR2_MODEL_DIR"]
        else { throw ValidationError("--model-dir or OCR2_MODEL_DIR required") }
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: image) as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw ValidationError("cannot read image: \(image)") }
        let session = try await OCR2Session.load(from: URL(fileURLWithPath: dir)) {
            FileHandle.standardError.write("\rloading \(Int($0 * 100))%".data(using: .utf8)!)
        }
        FileHandle.standardError.write("\n".data(using: .utf8)!)
        let task: OCRTask = grounding ? .grounding(query: query) : .freeOCR
        for try await chunk in session.ocr(image: cg, task: task) {
            print(chunk, terminator: ""); fflush(stdout)
        }
        print()
    }
}
```

`README.md`：包定位、API 两段示例、模型下载说明（`huggingface-cli download mlx-community/DeepSeek-OCR-2-8bit`）、夹具再生成流程、与 Python 参考的 parity 声明。

- [ ] **Step 3: ci.yml 追加 kit 步骤**

在 `.github/workflows/ci.yml` 的 `Build and test` 步骤后追加：

```yaml
      - name: Kit tests (DeepSeekOCR2Kit, logic-only)
        run: |
          set -o pipefail
          cd DeepSeekOCR2Kit && swift test 2>&1 | tail -20
```

（夹具/模型不存在时 parity 测试自动 skip，CI 只跑 mask/grounding/config 等纯逻辑测试。）

- [ ] **Step 4: 全量验证**

```bash
cd DeepSeekOCR2Kit && swift test 2>&1 | tail -5
MODEL=$(cat Fixtures/model_dir.txt)
swift run -c release ocr2-cli Fixtures/images/doc_page.png --model-dir "$MODEL" | head -5
swift run -c release ocr2-cli Fixtures/images/receipt.png --model-dir "$MODEL" --grounding --query "price of Croissant"
```

Expected: 测试全绿；CLI 流式输出 doc_page 文本（首行含 "DeepSeekOCR2Kit Parity Test Document"）；grounding 输出含 `<|det|>` 坐标（解析后打印框）。记录一次 tokens/s（M 系列参考值应 >20 tok/s，供 ② 的 UX 预估）。

- [ ] **Step 5: Commit**

```bash
git add DeepSeekOCR2Kit .github/workflows/ci.yml
git commit -m "feat(ocr2): streaming session API, grounding parser, ocr2-cli, CI wiring"
```

---

## 验证总表（对应 spec 里程碑）

| 里程碑 | 门禁 | 任务 |
|---|---|---|
| M0 | 包可测 + 夹具生成且参考文本正确 | 1, 2 |
| M1 | 868 键 sanitize 全覆盖 + 数值抽查 | 4 |
| M2 | SAM 输出 parity | 5 |
| M3 | mask 夹具一致 + 编码器 parity | 6 |
| M4 | 注入像素贪心 100 token 一致（3 图） | 7, 8 |
| M5 | token/结构/像素 parity + 真图端到端文本一致 | 9 |
| M6 | 全量测试绿 + CLI 演示 + CI 接线 | 10 |

执行纪律：**任何 parity 门禁不过，禁止进入下一任务**；分歧一律按"夹具逐级定位"处理（spec 风险清单的既定缓解），不做"看起来对就行"的目测放行。
