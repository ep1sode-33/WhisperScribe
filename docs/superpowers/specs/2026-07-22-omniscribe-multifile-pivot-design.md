# OmniScribe 多文件 pivot 设计（子项目 ②）

日期：2026-07-22 ｜ 状态：已确认（方案 A）｜ 前置：子项目 ①（DeepSeekOCR2Kit）已合并

## 背景与目标

产品定位从「音频→字幕」扩展为「**音频或多张图片→文字**」，并改名 **OmniScribe**。
① 已交付本地 OCR 引擎（`DeepSeekOCR2Kit` 的 `OCR2Session`，~28 tok/s，流式）；
② 把它接进 app，并把单文件流水线改造成同类多文件批次 + LLM 拼接去重。

## 已确认决策（含 ① brainstorm 期间的记录）

| 决策点 | 结论 |
|---|---|
| 批次组合 | **同类批次**：全音频或全图片；混投报错。图片支持 png/jpg/heic/webp/tiff |
| 图片批输出 | 一份合并去重 `.txt`（无 SRT——图片无时间轴） |
| 多音频批输出 | **每文件 `.srt`**（时间轴各自从 0 起）+ 一份合并 `.txt` |
| 去重策略 | 四类场景（滚动截图/书页/PPT/混合）全走**通用 LLM 合并**，不做场景特化 |
| 批内排序 | 文件名自然排序（`localizedStandardCompare`） |
| OCR 引擎 | `OCR2Session`（`.freeOCR`），模型 `mlx-community/DeepSeek-OCR-2-8bit`（~3GB） |
| 降级 | LLM 未配置/失败 → **按序拼接 + 分隔线**，绝不失败任务（沿用 fail-closed 哲学） |
| 改名 | 仅 `INFOPLIST_KEY_CFBundleDisplayName = OmniScribe` + README；**工程文件/bundle id/仓库名不动** |
| 合并 TXT 命名 | `<批内首文件名> merged.txt`，沿用 `FileNaming` uniquify |
| 内存策略 | 按批类型 lazy 加载；**加载新类型模型前释放另一类**（避免 Whisper+OCR 双 3GB 常驻） |

## 架构（方案 A：最小侵入扩展）

```
DropZone / NSOpenPanel（多选）
      │  BatchClassifier: [URL] → .audio([URL]) | .images([URL]) | 混投错误
      ▼
TranscriptionViewModel（批次状态机，沿用 jobToken 取消机制）
      ├─ 音频批: for 每文件 → AudioExtractor → TranscriberService → LLMCleaner → SubtitleWriter(.srt)
      │          全部转写文本 → MergeService → 合并 .txt
      ├─ 图片批: for 每文件 → OCRService(OCR2Session, 流式) 
      │          全部 OCR 文本 → MergeService → 合并 .txt
      └─ 进度: 批级 (i/N + 文件名) × 文件内阶段进度
```

### 新增组件

1. **`BatchClassifier`**（`Support/`，纯函数）：URL 列表 → 批类型；UTType 判定（音频沿用现有判定；图片 `.image` 及子类）；自然排序；混投/空列表返回 typed 错误。
2. **`OCRService`**（`Services/`，actor）：包装 `OCR2Session.load(from:progress:)` + `ocr(image:task:)`；输入 URL → CGImage（**应用 EXIF 方向**，复用 ① CLI 的 orientation 处理模式）；流式 chunk 计数上报进度（对齐 LLM 清洗的"已生成字符"UX）；协议 `OCRProviding` 供测试注入 fake。加载遵循内存策略：图片批开始时若 Whisper 模型驻留则先卸载，反之亦然（由 ViewModel 编排）。
3. **`MergeService`**（`Services/`，actor）：输入 `[(name: String, text: String)]` → 合并去重文本。走 `LLMCleaner` 同款 BYOK SSE 通道（复用其 HTTP/重试/流式基建——具体复用方式实施时依 `LLMCleaner` 现有内部结构定：抽公共 client 或经由新增公共方法）；分批策略沿用 ≤6000 字符/批思路，跨批携带上文尾部摘要；**提示词新增** `LLMPrompts.mergeSystem(language:)` / `mergeUser(…)`：识别相邻重叠段、去页眉页脚/页码、整张级重复丢弃、按输入序合并，输出纯文本；失败/未配置 → `"\n\n----- <文件名> -----\n\n"` 分隔的按序拼接 + 警告（`cleanup.warning.*` 模式）。
4. **`ModelManager` 扩展**：OCR 模型作为新的可下载条目（区别于 Whisper variants——新增一个 `OCRModelManager` 或在现有 manager 加类型维度，实施时依现有 `ModelManager`/`ModelDownloading` 结构选最小改动路径）；下载走 HuggingFace（与 WhisperKit 下载同源的 Hub API）；Settings ▸ Model 段加 OCR 小节（下载/进度/删除/占用空间）。图片批开始时若 OCR 模型未下载 → 引导到设置（复用现有 `noModelView` 模式）。

### 改造组件

5. **`DropZone`**：`onPick: (URL) -> Void` → `onPick: ([URL]) -> Void`；处理全部 providers（现在只取 `.first`）；类型过滤扩展图片；拖入即分类，混投时 UI 提示错误（不静默吞）。
6. **`ContentView`**：`NSOpenPanel.allowsMultipleSelection = true`，`allowedContentTypes` 加 `.image`；空态文案改双模态。
7. **`TranscriptionViewModel`**：`start(url:)` → `start(urls: [URL])`（单文件 = 单元素批，语义不变）；批循环 + 逐文件阶段推进；`lastOutputs` 改产物列表；`revealInFinder` 揭示全部产物所在目录。
8. **`JobState`**：扩展批上下文（`fileIndex/fileCount/currentFileName`）进各阶段 case（或包一层 `BatchProgress`——实施时选对 SwiftUI 观察最友好的形态）；`.done` 携带 `[URL]`。新增 `.recognizing(progress:)`（OCR 阶段）与 `.merging(progress:note:)`。
9. **`AppError`**：新增 `.mixedBatch`、`.unsupportedImage`、`.ocrModelMissing`、`.ocrFailed(String)`，各配五语言文案。
10. **`SubtitleWriter`/`FileNaming`**：新增合并 TXT 写出入口（`writeMergedText(name:dir:overwrite:)`）；SRT 路径不动。

### 改名与文案

11. `INFOPLIST_KEY_CFBundleDisplayName = OmniScribe`（两处 target 配置只改 app target）；窗口标题/关于沿用显示名。
12. README 重写：定位「本地音频/图片→文字」、双引擎（WhisperKit ANE + DeepSeek-OCR-2 GPU，全程不出 Mac）、图片批玩法示例、OCR 模型下载说明；保留 CI 徽章/Install/xattr 说明；DeepSeekOCR2Kit 链到包 README。
13. `Localizable.xcstrings`：新增 `drop.*`（双模态）、`status.recognizing/merging`、`batch.*`（i/N 进度）、`settings.ocrModel.*`、新 error 键——全部五语言。

## 错误处理

- 混投/不支持格式：拖放层即时报错，不进入任务。
- 批中单文件失败（音频解码失败/图片解码失败）：**跳过并计入警告**，批继续；全部失败才置 error（与"绝不失败任务"哲学一致）。
- OCR 模型未下载：任务前置检查 → 引导设置。
- 取消：任意阶段生效，释放 OCR session；沿用 jobToken 防旧任务写状态。
- LLM merge 失败：降级拼接（见上），警告可见。

## 测试

- 纯逻辑单测：`BatchClassifier`（分类/排序/混投/空）、`FileNaming` merged 命名、merge 降级拼接输出、`JobState` 批进度展示逻辑。
- 协议注入单测：`OCRProviding` fake（沿用 `FakeDownloader` 模式）驱动图片批状态机全流程（含单文件失败跳过、取消）；fake LLM 驱动 merge 成功/失败两路。
- ModelManager OCR 条目：复用现有 `ModelManagerTests` 的 fake-downloader 套路。
- 手动清单：真实拖放（多音频/多截图/混投）、OCR 模型下载 UX、端到端出稿、双语 UI 抽查。

## 验证

1. `xcodebuild test`（app）+ kit 套件全绿；CI 全绿。
2. 手动：4 张滚动截图 → 一份去重合并 txt；2 段音频 → 2 srt + 1 合并 txt；混投报错；取消中途生效。
3. About/Dock 显示 OmniScribe；README 定位一致。
