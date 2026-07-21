# DeepSeekOCR2Kit 设计：DeepSeek-OCR-2 的 mlx-swift 移植

日期：2026-07-20 ｜ 状态：已确认 ｜ 子项目 ①/②（先 ① 后 ②）

## 背景与目标

WhisperScribe 产品定位从「音频→字幕」扩展为「音频或多张图片→文字」：多文件一批丢入，
音频走 WhisperKit、图片走本地 OCR，产物统一交给 BYOK LLM 拼接去重清洗。用户明确要求
OCR 用 **DeepSeek-OCR-2**（2026-01-27 发布，3B VLM，MIT）**本地运行**，不用云端 VLM。

pivot 拆为两个子项目：
- **①（本篇）**：`DeepSeekOCR2Kit` —— 把 DeepSeek-OCR-2 移植到 mlx-swift，产出可被 app
  消费的 SwiftPM 包。
- **②（另立 spec）**：多文件批次流水线 + LLM 合并去重 + UI/设置/文案改造。②只依赖 ①
  的 protocol 接口。已确认的 ② 关键决策（记录备用）：同类批次（音频只配音频、图片只配
  图片）；图片批 → 合并 `.txt`；多音频批 → 每文件 `.srt` + 合并 `.txt`；四类图片场景
  （滚动截图/书页/PPT/混合）全部走通用 LLM 去重，不做场景特化。

### 硬件路径的现实约束（已与用户确认）

3B 自回归 VLM 目前**没有可用的 CoreML/ANE 路径**；本地推理走 **MLX（Metal GPU）**。
最终形态：WhisperKit 继续跑 ANE，OCR 跑 GPU，识别全程不出 Mac。

## 已确认决策

| 决策点 | 结论 |
|---|---|
| 路线 | 方案 A：自研移植 v2（不用 v1 顶包、不用 Python sidecar） |
| 包形态 | WhisperScribe 仓库内本地 SwiftPM 包，目录 `DeepSeekOCR2Kit/`（避开 .gitignore 的 `Packages/` 规则）；将来可拆独立仓库 |
| 模型 | `mlx-community/DeepSeek-OCR-2-8bit`（~3GB，混合精度：LM+projector 8-bit，视觉侧 bf16），不自行量化 |
| 依赖 | `mlx-swift-lm ≥ 3.31.3`（MLXVLM + MLXLMCommon + MLXHuggingFace）、swift-transformers ≥1.3.0 Tokenizers（app 依赖树已有）；不 fork 任何仓库 |
| v1 复用 | 从 mzbac/deepseek-ocr.swift（MIT）**拷贝文件**进包（保留版权声明），不作为依赖（其锁 mlx-swift 0.29.1 与生态 0.31.4 冲突） |

## 对外 API（② 的消费面）

```swift
public final class DeepSeekOCR2Model {
    public static func load(from dir: URL, progress: @Sendable (Double) -> Void) async throws -> DeepSeekOCR2Model
    public func ocr(image: CGImage, task: OCRTask = .freeOCR) -> AsyncThrowingStream<String, Error>
}
public enum OCRTask { case freeOCR, grounding }   // grounding 输出 <|ref|>/<|det|> 坐标（0–1000 归一化）
```

- 模型下载不在包内实现：包只接收本地目录（`factory.load(from:using:)` 纯本地路径）；
  下载/进度 UX 由 app 现有 `ModelManager` 模式在 ② 中扩展。
- 附 `ocr2-cli` 可执行目标：`ocr2-cli <image> [--grounding] [--model-dir <dir>]`，独立调试用。

## 组件清单

### 复用（拷贝改造 / 直接依赖）

| 组件 | 来源 | 改造点 |
|---|---|---|
| SAM ViT-B 编码器 | v1 港 `VisionEncoder.swift`（409 行）——**v2 未换 SAM** | `net_3` 输出 1024→896；新增 768² 输入路径；pos-embed 插值换 MLXLMCommon 的 Metal `bicubicInterpolate` |
| DeepSeek-3B-MoE LM | v1 港 `LanguageModel.swift`（464 行），v2 config 几乎相同 | attention/KV cache 换 `MLXLMCommon` 基元（`KVCacheSimple`/`attentionWithCacheUpdate`）；断言 `n_group=1/topk_group=1`；`qk_nope_head_dim=0` → 走非 MLA 分支 |
| 动态裁剪预处理 | v1 港 `ImageProcessor.swift`（397 行） | base 1024、tile **768**、min 1/max 6；token 公式 `num_patches*144 + 256 + 1`；删 imageNewline 逻辑；输出双张量 + `images_spatial_crop` |
| expert 堆叠/conv 转置 | v1 港 `sanitizeWeights` | 键前缀按 v2 checkpoint 重做 |
| 工厂注册/量化加载/MoE 内核/RoPE/生成循环/流式 detokenizer/tokenizer 加载 | mlx-swift-lm | 原样；运行时 `VLMTypeRegistry.shared.registerModelType("deepseekocr_2", …)` |

不复用：v1 的 CLIP 编码器（v2 已弃）、旧 projector、自定义量化 manifest、朴素 KV cache/生成器。

### 新写（~2,500–3,200 行）

| # | 组件 | 参考 | 规模 | 首要风险 |
|---|---|---|---|---|
| P1 | `Qwen2Decoder2Encoder` 视觉编码器：Qwen2-0.5B 主干（24 层/896 宽/GQA 14:2/theta 1e6）+ query bank `query_1024`(256×896)/`query_768`(144×896) + 四象限自定义 mask（image↔image 双向、image→query -1e9、query→image 全通、query↔query causal） | mlx-vlm `deepseekocr_2/vision.py` | M ~400 行 | **全项目最高**：mask 写错不崩溃只出垃圾；query bank 按 SAM token 数选择；RoPE 覆盖 concat 全序列 |
| P2 | SAM 适配层（见上复用行的改造） | `deepseekocr/sam.py` + v1 港 | S–M | bicubic 数值差；rel_pos 48×48 重采样 |
| P3 | 顶层模型：逐视图 SAM→Qwen2→`Linear(896,1280)`，特征序 **`[local…, global, view_separator]`**（config 的 `global_view_pos:"head"` 有误导），按 `images_seq_mask` 散布进 embedding | `deepseekocr_2.py`（293 行） | M ~350 行 | 特征顺序；mask 散布；vision 仅 prefill 跑一次 |
| P4 | Processor：全局 1024²（127 灰填充）+ InternVL 平铺 768²、`(x/255-0.5)/0.5` bf16、`<image>`→N×id **128815**、**硬编码 BOS 0 前置**、`inference_mode` 剥末 token；prompt 精确为 `"<image>\nFree OCR. "`（**含尾随空格**，chat template 是 no-op，禁用 `applyChatTemplate`）；双分辨率张量 flatten+frames 约定打包 | `processing_deepseekocr.py`（624 行） | L ~500 行 | **OCR 质量第一决定因素**：PIL vs CoreImage 重采样数值差；`find_closest_aspect_ratio` 平手规则逐字对齐 |
| P5 | Configuration：大量超参**不在 config.json**（SAM/Qwen2-encoder/query 尺寸皆代码内默认），逐一硬编码复刻；bos 0/eos 1/image 128815 | `config.py`（216 行） | S ~200 行 | 默认值抄漏 |
| P6 | v2 sanitize 键映射：`model.qwen2_model.model.model.layers.*`→内部命名、`query_1024/768` 重映射、**`model.view_seperator` 拼写陷阱**、expert 堆叠、conv OIHW→OHWI；868 张量全覆盖 | `deepseekocr_2.py:206-293` | M ~150 行 | 键漏配 |
| P7 | 注册 + 门面 + CLI；HF 仓库**无 preprocessor_config.json**，工厂 processor 装载不适配时绕开工厂手动组 `ModelContext` | — | S | 工厂行为需实测 |
| P8 | grounding 解析（0–1000 坐标）——可选，随 CLI 交付 | README 格式 | S | — |

## 验证策略（成败关键）

双轨 parity，夹具由脚本生成、**不进 git**（`DeepSeekOCR2Kit/Fixtures/` 入 .gitignore）：

1. **黄金夹具**：Python mlx-vlm（scratchpad 已克隆，pip 环境本地建）对 3–5 张固定图
   （文档页/票据/CJK 密集/≥4 tiles 长图/grounding 各一）dump 逐级中间张量为 safetensors：
   预处理像素 → SAM 输出 → 编码器输出(144|256×896) → projector → merge 后 embeds →
   prefill 末位 logits → 贪心前 100 token id → 最终文本。mask 矩阵本身也 dump 作 P1 单测夹具。
2. **轨道 A（模型 parity）**：Swift 注入参考像素逐级比对（相对误差 ~1e-2 级）；
   **最终判据 = temperature 0 贪心 token 序列前 100 步 100% 一致**（量化模型 logits 必有
   漂移，argmax 序列才是行为等价）。
3. **轨道 B（预处理 parity）**：同一原图走 Swift 管线，先断言结构量（tile 数、
   `images_spatial_crop`、token 总数、prompt token id 序列）再比像素（mean abs diff < ~2/255）；
   分歧过大时手写 PIL 等价 bicubic。
4. **Tokenizer parity**：全部 prompt 变体 Python/Swift 编码 id 逐一相等；CJK decode 往返
   （防 v1 港的逐 token UTF-8 乱码缺陷，用 `NaiveStreamingDetokenizer`）。
5. **权重加载校验**：`update(verify: [.all])` + 对 `model.safetensors.index.json` 868 键集合
   diff + 抽查张量数值（含 stacked expert、转置 conv、`view_seperator`）。

CI（现有 ci.yml 会自动覆盖包的单元测试）只跑 tokenizer/结构/mask 单测；重型 parity 测试
本地跑，用环境变量门控（夹具缺失时 skip）。

## 里程碑（每个都有硬性门禁）

- **M0** 包骨架 + Python 夹具生成脚本（`DeepSeekOCR2Kit/scripts/gen_fixtures.py`）
- **M1** 权重加载：868 键全覆盖 + 数值抽查通过
- **M2** SAM 适配：SAM 输出 parity
- **M3** Qwen2 编码器：mask 单测 + 编码器输出 parity（144 与 256 两条路径）
- **M4** 顶层拼装：注入像素端到端贪心 100 token 一致
- **M5** Processor：预处理 parity + 真原图端到端文本一致（CER≈0）
- **M6** 流式 API + CLI + 文档；ci.yml 覆盖包单测

## Top 5 风险与缓解

1. P1 自定义 mask 写错（静默劣化）→ mask 矩阵夹具单测 + 144/256 双路径逐级比对
2. 预处理数值/结构分歧 → 双轨分离、结构量先行断言、必要时手写 PIL 等价重采样
3. 双分辨率张量穿 `LMInput` 抽象 + 工厂缺 preprocessor_config.json → flatten+frames 约定；兜底绕开工厂手动组 `ModelContext`
4. 权重键映射/混合精度加载错 → `verify:[.all]` + 键集合 diff + 数值抽查
5. prompt/tokenizer 细节漂移（尾随空格、BOS 0、EOS 1、no-op 模板）→ 手工复刻 prompt 组装 + token-id 级 parity + `extraEOSTokens` 显式配 1

## 参考材料

- Python 参考实现：`Blaizzy/mlx-vlm` → `mlx_vlm/models/deepseekocr_2/`（+ 共享 `deepseekocr/sam.py`）
- v1 Swift 港（拷贝源）：`mzbac/deepseek-ocr.swift` → `Sources/DeepSeekOCR/{VisionEncoder,ImageProcessor,LanguageModel,DeepSeekOCRModel}.swift`
- 新 VLM 模板：`ml-explore/mlx-swift-lm` → `Libraries/MLXVLM/Models/FastVLM.swift`
- 本会话 scratchpad 已有三仓浅克隆（`mlx-vlm/`、`deepseek-ocr-swift/`、`mlx-swift-examples/`、`mlx-swift-lm/`），实施时若已失效按上述 URL 重克隆
