# OmniScribe 多文件 pivot 实施计划（子项目 ②）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把单文件音频→字幕流水线改造为同类多文件批次（音频批/图片批），图片走本地 DeepSeekOCR2Kit OCR，产物经 BYOK LLM 拼接去重，app 显示名改 OmniScribe。

**Architecture:** 方案 A 最小侵入：新增 `BatchClassifier`（纯函数分类）、`MergeService`（复用 LLM 传输层的拼接去重 actor）、`OCRService`（包装 `OCR2Session`）、`OCRModelManager`（兄弟模型管理器）；`TranscriptionViewModel` 扩展为批次编排（批上下文用独立 `@Published`，`JobState` 只加两个新 case）；UI 层多选 + OCR 模型设置段。

**Tech Stack:** SwiftUI/Swift Testing（现状）、DeepSeekOCR2Kit（本地包）、swift-transformers `HubApi`（OCR 模型下载）。

**设计文档:** `docs/superpowers/specs/2026-07-22-omniscribe-multifile-pivot-design.md`（实施前先读——所有已确认决策在内）

## Global Constraints

- 同类批次：全音频或全图片（png/jpg/jpeg/heic/webp/tiff）；混投 → `AppError.mixedBatch`。批内排序 `localizedStandardCompare`（文件名自然排序）。
- 图片批 → 一份合并 `.txt`；多音频批 → 每文件 `.srt` + 一份合并 `.txt`；合并名 `<首文件名> merged.txt`（uniquify）。
- LLM 未配置/失败 → 按序拼接 + `"\n\n----- <文件名> -----\n\n"` 分隔 + 警告，**任务绝不失败**。
- 批中单文件失败 → 跳过 + 警告；全部失败才 `.error`。
- 内存互斥：图片批前 `transcriber.unload()`，音频批前 `ocr.unload()`。
- 改名仅 `INFOPLIST_KEY_CFBundleDisplayName = OmniScribe`；bundle id/工程名/仓库名不动。
- OCR 模型：`mlx-community/DeepSeek-OCR-2-8bit`，落盘 `~/Documents/huggingface/models/mlx-community/DeepSeek-OCR-2-8bit`（HubApi 默认布局）；就绪判定 = 目录含 `config.json` + `tokenizer.json` + 任一 `*.safetensors`。
- **工程注册规则**：app 源文件放 `WhisperScribe/` 下任意位置即自动入 target（synchronized group）；**每个新测试文件必须** `ruby scripts/add_test_file.rb WhisperScribeTests/<File>.swift`。
- 测试命令：`xcodebuild test -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS' -skipPackagePluginValidation -skipMacroValidation`（下文简写 `xcodebuild test`）。
- 新 UI 字符串一律先在 Task 1 入 xcstrings（五语言），组件只引用键名。
- 提交信息末尾带：
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01U5wGZ5BEeMyPZd9jcPP34f`

## 文件结构总览

```
WhisperScribe/
├─ Support/BatchClassifier.swift        (T2) 新
├─ Support/FileNaming.swift             (T3) 改
├─ Services/SubtitleWriter.swift        (T3) 改
├─ Services/LLMTransport.swift          (T4) 新——从 LLMCleaner 抽出
├─ Services/LLMCleaner.swift            (T4) 改——改用 LLMTransport
├─ Services/LLMPrompts.swift            (T4) 改——merge 提示词
├─ Services/MergeService.swift          (T4) 新
├─ Services/OCRService.swift            (T6) 新
├─ Services/OCRModelManager.swift       (T7) 新
├─ Services/TranscriberService.swift    (T8) 改——加 unload()
├─ Models/JobState.swift                (T8) 改
├─ Models/AppError.swift                (T2) 改
├─ ViewModel/TranscriptionViewModel.swift (T8/T9) 改
├─ Views/DropZone.swift                 (T10) 改
├─ Views/ContentView.swift              (T10) 改
├─ Views/StatusView.swift               (T10) 改
├─ Views/SettingsView.swift             (T10) 改
├─ App/AppModel.swift                   (T9) 改
└─ Localizable.xcstrings                (T1) 改
WhisperScribeTests/
├─ BatchClassifierTests.swift           (T2) 新
├─ MergedNamingTests.swift              (T3) 新
├─ MergeServiceTests.swift              (T4) 新
├─ OCRServiceTests.swift                (T6) 新
├─ OCRModelManagerTests.swift           (T7) 新
└─ BatchPipelineTests.swift             (T8/T9) 新
scripts/add_pivot_strings.rb            (T1) 新（一次性，仿 update_strings.rb）
WhisperScribe.xcodeproj/project.pbxproj (T5 kit 依赖 / T11 显示名)
README.md                               (T11) 改
```

---

### Task 1: i18n 键批量新增

**Files:**
- Create: `scripts/add_pivot_strings.rb`
- Modify: `WhisperScribe/Localizable.xcstrings`（脚本产出）

**Interfaces:**
- Produces: 下列键（后续任务只引用，不再新增）：`drop.subtitleMulti`、`status.recognizing`、`status.merging`、`status.loadingOCRModel`、`batch.fileProgress`、`done.batchSummary`、`error.mixedBatch`、`error.unsupportedImage`、`error.ocrModelMissing`、`error.ocrFailed`、`settings.ocrModel.title`、`settings.ocrModel.tagline`、`settings.ocrModel.size`、`cleanup.warning.mergeFallback`、`cleanup.warning.fileSkipped`。

- [ ] **Step 1: 写一次性脚本**（仿 `scripts/update_strings.rb` 的 `entry` helper 结构——先读该文件抄其 helper，再填下表内容；`extractionState: "manual"`、全部 `state: "translated"`）

| 键 | en | zh-Hans | zh-Hant | ja | ko |
|---|---|---|---|---|---|
| drop.subtitleMulti | Audio files or images — drop several at once | 音频或图片，可一次拖入多个 | 音訊或圖片，可一次拖入多個 | 音声または画像 — 複数まとめてドロップ可 | 오디오 또는 이미지 — 여러 개 동시에 드롭 가능 |
| status.recognizing | Recognizing text… | 正在识别文字… | 正在識別文字… | 文字を認識中… | 텍스트 인식 중… |
| status.merging | Merging & deduplicating… | 正在拼接去重… | 正在拼接去重… | 結合・重複除去中… | 병합 및 중복 제거 중… |
| status.loadingOCRModel | Loading OCR model… | 正在加载 OCR 模型… | 正在載入 OCR 模型… | OCR モデルを読み込み中… | OCR 모델 로드 중… |
| batch.fileProgress | File %1$d of %2$d — %3$@ | 第 %1$d/%2$d 个文件 — %3$@ | 第 %1$d/%2$d 個檔案 — %3$@ | ファイル %1$d/%2$d — %3$@ | 파일 %1$d/%2$d — %3$@ |
| done.batchSummary | %1$d files → %2$d outputs | %1$d 个文件 → %2$d 个产物 | %1$d 個檔案 → %2$d 個產物 | %1$d ファイル → %2$d 出力 | 파일 %1$d개 → 출력 %2$d개 |
| error.mixedBatch | Mixing audio and images isn't supported — drop one kind at a time. | 不支持音频和图片混合投放——一次只能拖同一类文件。 | 不支援音訊與圖片混合投放——一次只能拖同一類檔案。 | 音声と画像の混在には対応していません。同じ種類のみドロップしてください。 | 오디오와 이미지를 섞을 수 없습니다. 한 번에 한 종류만 드롭하세요. |
| error.unsupportedImage | Unsupported file: %@ | 不支持的文件：%@ | 不支援的檔案：%@ | 対応していないファイル: %@ | 지원하지 않는 파일: %@ |
| error.ocrModelMissing | The OCR model isn't downloaded yet. Get it in Settings ▸ Model. | OCR 模型尚未下载，请到 设置 ▸ 模型 下载。 | OCR 模型尚未下載，請到 設定 ▸ 模型 下載。 | OCR モデルが未ダウンロードです。設定 ▸ モデルから取得してください。 | OCR 모델이 아직 없습니다. 설정 ▸ 모델에서 받으세요. |
| error.ocrFailed | Text recognition failed: %@ | 文字识别失败：%@ | 文字識別失敗：%@ | 文字認識に失敗しました: %@ | 텍스트 인식 실패: %@ |
| settings.ocrModel.title | OCR model (images → text) | OCR 模型（图片→文字） | OCR 模型（圖片→文字） | OCR モデル（画像→テキスト） | OCR 모델 (이미지→텍스트) |
| settings.ocrModel.tagline | DeepSeek-OCR-2 · runs locally on GPU | DeepSeek-OCR-2 · 本地 GPU 运行 | DeepSeek-OCR-2 · 本地 GPU 執行 | DeepSeek-OCR-2 · ローカル GPU で実行 | DeepSeek-OCR-2 · 로컬 GPU 실행 |
| settings.ocrModel.size | ~3 GB | ~3 GB | ~3 GB | ~3 GB | ~3 GB |
| cleanup.warning.mergeFallback | LLM merge unavailable — files were joined in order without deduplication. | LLM 拼接不可用——已按顺序直接拼接，未去重。 | LLM 拼接不可用——已按順序直接拼接，未去重。 | LLM 結合が使えないため順番どおり連結しました（重複除去なし）。 | LLM 병합 불가 — 순서대로 이어붙였습니다(중복 제거 없음). |
| cleanup.warning.fileSkipped | Skipped %@: %@ | 已跳过 %@：%@ | 已跳過 %@：%@ | %@ をスキップ: %@ | %@ 건너뜀: %@ |

- [ ] **Step 2: 执行并抽验**

Run: `ruby scripts/add_pivot_strings.rb && python3 -c "import json;d=json.load(open('WhisperScribe/Localizable.xcstrings'));ks=[k for k in d['strings'] if k in ('batch.fileProgress','error.mixedBatch','settings.ocrModel.title')];print(ks, all(len(d['strings'][k]['localizations'])==5 for k in ks))"`
Expected: 三键齐全且各 5 语言 → `True`

- [ ] **Step 3: 构建冒烟**（xcstrings 编译入构建）

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS' -configuration Debug build 2>&1 | tail -3`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 4: Commit** — `git add scripts/add_pivot_strings.rb WhisperScribe/Localizable.xcstrings && git commit -m "i18n: add multi-file pivot strings (batch, OCR, merge, rename prep)"`

---

### Task 2: BatchClassifier + AppError 新 case

**Files:**
- Create: `WhisperScribe/Support/BatchClassifier.swift`
- Modify: `WhisperScribe/Models/AppError.swift`（新 case + errorDescription）
- Test: `WhisperScribeTests/BatchClassifierTests.swift`（**注册**：`ruby scripts/add_test_file.rb WhisperScribeTests/BatchClassifierTests.swift`）

**Interfaces:**
- Produces:

```swift
enum BatchKind: Equatable {
    case audio([URL])    // 已按文件名自然排序
    case images([URL])   // 已按文件名自然排序
}
enum BatchClassifier {
    static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "heic", "webp", "tiff", "tif"]
    static func classify(_ urls: [URL]) throws -> BatchKind   // throws AppError
    static func isAcceptedURL(_ url: URL) -> Bool             // 拖放高亮判定用（音频或图片）
}
// AppError 新 case: .mixedBatch, .unsupportedFile(String), .ocrModelMissing, .ocrFailed(String)
```

- [ ] **Step 1: 写失败测试**

```swift
import Testing
import Foundation
@testable import WhisperScribe

struct BatchClassifierTests {
    private func url(_ name: String) -> URL { URL(fileURLWithPath: "/tmp/\(name)") }

    @Test func classifiesAudioBatchSorted() throws {
        let kind = try BatchClassifier.classify([url("b10.mp3"), url("b2.wav"), url("a.m4a")])
        guard case .audio(let sorted) = kind else { Issue.record("expected audio"); return }
        #expect(sorted.map(\.lastPathComponent) == ["a.m4a", "b2.wav", "b10.mp3"])  // 自然排序: 2 < 10
    }
    @Test func classifiesImageBatchSorted() throws {
        let kind = try BatchClassifier.classify([url("IMG_10.PNG"), url("IMG_2.jpg")])
        guard case .images(let sorted) = kind else { Issue.record("expected images"); return }
        #expect(sorted.map(\.lastPathComponent) == ["IMG_2.jpg", "IMG_10.PNG"])     // 扩展名大小写不敏感
    }
    @Test func rejectsMixedBatch() {
        #expect(throws: AppError.mixedBatch) { try BatchClassifier.classify([url("a.mp3"), url("b.png")]) }
    }
    @Test func rejectsUnsupportedFile() {
        #expect((try? BatchClassifier.classify([url("x.pdf")])) == nil)
    }
    @Test func rejectsEmpty() {
        #expect((try? BatchClassifier.classify([])) == nil)
    }
    @Test func acceptedURLCoversBothKinds() {
        #expect(BatchClassifier.isAcceptedURL(url("a.mp3")))
        #expect(BatchClassifier.isAcceptedURL(url("a.heic")))
        #expect(!BatchClassifier.isAcceptedURL(url("a.pdf")))
    }
}
```

- [ ] **Step 2: 注册测试文件并确认编译失败**

Run: `ruby scripts/add_test_file.rb WhisperScribeTests/BatchClassifierTests.swift && xcodebuild test 2>&1 | tail -5`
Expected: 编译失败（`BatchClassifier` 未定义）

- [ ] **Step 3: 实现**

`BatchClassifier.swift`：音频判定复用 UTType（对照 `Views/DropZone.swift:51-60` 的 `isMediaURL` 逻辑：`UTType(filenameExtension:)` conforms `.audio` 或 `.audiovisualContent`）；图片判定用 `imageExtensions`（小写比较）；分类规则：逐个判 kind，出现两种 → `throw AppError.mixedBatch`；出现 other → `throw AppError.unsupportedFile(name)`；空 → `throw AppError.unsupportedFile("")`；排序 `sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }`。

`AppError.swift`：加 4 个 case（`mixedBatch`、`unsupportedFile(String)`、`ocrModelMissing`、`ocrFailed(String)`），`errorDescription` 按现有模式用 `NSLocalizedString`（键：`error.mixedBatch`/`error.unsupportedImage`/`error.ocrModelMissing`/`error.ocrFailed`，带参的用 `String.localizedStringWithFormat`，对照 `AppError.swift:19` 现有写法）。同步补 `Equatable` 分支（若现有实现是 synthesized 则自动覆盖）。

- [ ] **Step 4: 测试通过** — Run: `xcodebuild test 2>&1 | tail -3`，Expected: 全绿（原有 14 + 新 6）
- [ ] **Step 5: Commit** — `git add WhisperScribe/Support/BatchClassifier.swift WhisperScribe/Models/AppError.swift WhisperScribeTests/BatchClassifierTests.swift WhisperScribe.xcodeproj && git commit -m "feat: batch classifier with natural sort + new error cases"`

---

### Task 3: 合并 TXT 命名与写出

**Files:**
- Modify: `WhisperScribe/Support/FileNaming.swift`
- Modify: `WhisperScribe/Services/SubtitleWriter.swift`
- Test: `WhisperScribeTests/MergedNamingTests.swift`（注册脚本同前）

**Interfaces:**
- Consumes: `FileNaming.outputURLs` 现有 uniquify 逻辑（`FileNaming.swift:28-35`）。
- Produces:

```swift
// FileNaming 新增
static func mergedTextURL(firstSource: URL, outputDir: URL?, overwrite: OverwritePolicy) -> URL
// 基名 = firstSource 去扩展名 + " merged"，目录规则与 outputURLs 相同（outputDir ?? 源目录），.uniquify 追加 " 2"/" 3"…
// SubtitleWriter 新增
static func writeMergedText(_ text: String, firstSource: URL, outputDir: URL?, overwrite: OverwritePolicy) throws -> URL
// 建父目录 + 原子写 UTF-8，失败包 AppError.writeFailed（对照现有 write 的模式）
```

- [ ] **Step 1: 写失败测试**

```swift
import Testing
import Foundation
@testable import WhisperScribe

struct MergedNamingTests {
    @Test func mergedNameNextToSource() {
        let u = FileNaming.mergedTextURL(firstSource: URL(fileURLWithPath: "/tmp/shot 1.png"),
                                         outputDir: nil, overwrite: .overwrite)
        #expect(u.path == "/tmp/shot 1 merged.txt")
    }
    @Test func mergedNameUniquifies() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let src = dir.appending(path: "a.png")
        FileManager.default.createFile(atPath: dir.appending(path: "a merged.txt").path, contents: Data())
        let u = FileNaming.mergedTextURL(firstSource: src, outputDir: nil, overwrite: .uniquify)
        #expect(u.lastPathComponent == "a merged 2.txt")
    }
    @Test func writeMergedTextRoundTrip() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let out = try SubtitleWriter.writeMergedText("你好\nworld",
                                                     firstSource: dir.appending(path: "x.wav"),
                                                     outputDir: dir, overwrite: .overwrite)
        #expect(try String(contentsOf: out, encoding: .utf8) == "你好\nworld")
        #expect(out.lastPathComponent == "x merged.txt")
    }
}
```

- [ ] **Step 2: 注册 + 确认编译失败**（命令模式同 Task 2 Step 2）
- [ ] **Step 3: 实现**（uniquify 循环仿 `FileNaming.swift:28-35`，但只探测单个 txt 文件而非 srt/txt 对）
- [ ] **Step 4: 测试通过** — `xcodebuild test 2>&1 | tail -3` 全绿
- [ ] **Step 5: Commit** — `git commit -m "feat: merged-txt naming and writer"`（含 pbxproj）

---

### Task 4: LLMTransport 抽取 + MergeService + merge 提示词

**Files:**
- Create: `WhisperScribe/Services/LLMTransport.swift`
- Modify: `WhisperScribe/Services/LLMCleaner.swift`（改用 LLMTransport，行为不变）
- Modify: `WhisperScribe/Services/LLMPrompts.swift`（新增 merge 提示词）
- Create: `WhisperScribe/Services/MergeService.swift`
- Test: `WhisperScribeTests/MergeServiceTests.swift`（注册）

**Interfaces:**
- Consumes: `LLMConfig`（`Models/LLMConfig.swift:4-21`，`isConfigured`）。
- Produces（后续任务按此签名使用）:

```swift
enum LLMTransport {
    struct ChatMessage: Codable { let role: String; let content: String }
    static func endpointURL(_ base: String) -> URL?
    static func sleepBackoff(attempt: Int, retryAfter: Double?) async throws
    /// SSE 流式 chat。onDelta 每收到一段增量文本回调一次。返回完整文本。
    static func streamChat(config: LLMConfig, system: String, user: String,
                           maxTokens: Int?, onDelta: @escaping @Sendable (String) -> Void) async throws -> String
}
protocol ChatStreaming: Sendable {
    func streamChat(config: LLMConfig, system: String, user: String,
                    onDelta: @escaping @Sendable (String) -> Void) async throws -> String
}
struct LLMTransportChat: ChatStreaming {}   // 默认实现，转调 LLMTransport（maxTokens nil）
actor MergeService {
    init(chat: ChatStreaming = LLMTransportChat())
    /// 永不 throw。LLM 不可用/失败 → fallback 拼接 + warning。
    func merge(parts: [(name: String, text: String)], language: String?, config: LLMConfig,
               progress: @escaping @Sendable (Double, String) -> Void) async -> (text: String, warnings: [String])
}
// LLMPrompts 新增
static func mergeSystem(language: String?) -> String
static func mergeUser(parts: String) -> String
```

- [ ] **Step 1: 抽取 LLMTransport（重构，不改行为）**

从 `LLMCleaner.swift` 迁出：`endpointURL`（:504-513）、`sleepBackoff`（:608-615）、`ChatMessage`/`ChatRequestBody`/`StreamChunk`（:515-532）、`LLMRequestError`（:26-38）→ 全部移入 `LLMTransport.swift`（`LLMRequestError` 保持 internal）。`performChat`（:537-605）改造成 `LLMTransport.streamChat(...)`（`nonisolated static`，其对 actor 隔离状态 `generatedChars` 的副作用改为 `onDelta` 回调外置）。

**实施者裁量点（明确授权）**：`LLMCleaner` 的 live-progress（`generatedChars` 逐 delta 更新）如何在 actor 隔离下接回调——推荐两选一：(1) `streamChat` 提供 `AsyncThrowingStream<String, Error>` 形态，`LLMCleaner.performChat` 在自身 actor 上下文 `for try await` 消费并更新计数；(2) `streamChat` 的 onDelta 加 `isolated` 调用方参数。无论选哪种，验收标准固定：(a) LLMCleaner 现有行为不变（清洗/进度/重试语义一致，构建零新警告）；(b) MergeService 与其测试只依赖 `ChatStreaming` 协议签名。所选方案写进报告。

- [ ] **Step 2: 写 MergeService 失败测试**

```swift
import Testing
import Foundation
@testable import WhisperScribe

private struct FakeChat: ChatStreaming, @unchecked Sendable {
    let result: Result<String, Error>
    var received: (@Sendable (String, String) -> Void)? = nil   // (system, user)
    func streamChat(config: LLMConfig, system: String, user: String,
                    onDelta: @escaping @Sendable (String) -> Void) async throws -> String {
        received?(system, user)
        switch result {
        case .success(let s): onDelta(s); return s
        case .failure(let e): throw e
        }
    }
}
private let cfg = LLMConfig(baseURL: "https://api.example.com/v1", apiKey: "k", model: "m")

struct MergeServiceTests {
    @Test func mergesViaLLM() async {
        let svc = MergeService(chat: FakeChat(result: .success("合并后的全文")))
        let out = await svc.merge(parts: [("a.png", "第一段"), ("b.png", "第二段")],
                                  language: "zh", config: cfg, progress: { _, _ in })
        #expect(out.text == "合并后的全文")
        #expect(out.warnings.isEmpty)
    }
    @Test func fallsBackWhenLLMFails() async {
        struct E: Error {}
        let svc = MergeService(chat: FakeChat(result: .failure(E())))
        let out = await svc.merge(parts: [("a.png", "甲"), ("b.png", "乙")],
                                  language: nil, config: cfg, progress: { _, _ in })
        #expect(out.text.contains("甲") && out.text.contains("乙"))
        #expect(out.text.contains("----- b.png -----"))
        #expect(out.warnings.count == 1)
    }
    @Test func fallsBackWhenUnconfigured() async {
        let svc = MergeService(chat: FakeChat(result: .success("不应被调用")))
        let out = await svc.merge(parts: [("a.png", "甲")], language: nil,
                                  config: LLMConfig(baseURL: "", apiKey: "", model: ""),
                                  progress: { _, _ in })
        #expect(out.text == "甲")           // 单文件直接原文（无需分隔头）
        #expect(out.warnings.count == 1)
    }
    @Test func promptCarriesPartsInOrder() async {
        nonisolated(unsafe) var captured = ""
        var fake = FakeChat(result: .success("ok"))
        fake.received = { _, user in captured = user }
        _ = await MergeService(chat: fake).merge(parts: [("1.png", "AAA"), ("2.png", "BBB")],
                                                 language: nil, config: cfg, progress: { _, _ in })
        #expect(captured.range(of: "AAA")!.lowerBound < captured.range(of: "BBB")!.lowerBound)
    }
}
```

- [ ] **Step 3: 注册 + 确认编译失败**（同前模式）
- [ ] **Step 4: 实现 MergeService + 提示词**

`LLMPrompts.mergeSystem(language:)`（中文指令风格对照现有 `srtL2System` 写法，复用 `multilingualRules`/`languageNote`）：

```
你是文本合并引擎。用户提供多份按顺序排列的文字片段（来自连续截图或分段录音的识别结果）。任务：
1. 按给定顺序合并为一篇连贯文本；
2. 相邻片段若有重叠区域（前一份结尾与后一份开头重复），只保留一份；
3. 删除页眉、页脚、页码、状态栏时间等与正文无关的重复元素；
4. 整份内容与前一份基本相同的片段，整体丢弃；
5. 不改写、不总结、不增删正文语义；保持原语言。
只输出合并后的正文，不要任何解释。
```

`mergeUser(parts:)`：`"以下是按顺序编号的片段：\n\n" + parts`（parts 由 MergeService 组装为 `【片段 i · 文件名】\n正文` 序列）。

`MergeService.merge` 逻辑：
1. `parts.isEmpty` → 返回 `("", [])`；`config.isConfigured == false` → fallback。
2. 组装 user 文本；总字符 ≤ 12000 → 单次 `chat.streamChat`；否则按 ≤ 12000 字符分组（不拆单个 part），逐组调用，第 2 组起 user 前缀携带上一组结果的尾部 600 字符（标注"【已合并文本的结尾，供衔接】"），结果逐段拼接。
3. progress：完成组数/总组数 + `String(localized: "status.merging")`；onDelta 时可细化到字符计数（`0.x + 字符注记`，对照 LLMCleaner 的 note 风格，简单实现即可）。
4. 任何 throw → fallback：`parts.map` 以 `"\n\n----- \(name) -----\n\n"` 连接（首段不加头），warning 加 `String(localized: "cleanup.warning.mergeFallback")`；单 part 直接原文。

- [ ] **Step 5: 测试通过**（`xcodebuild test 2>&1 | tail -3` 全绿；LLMCleaner 相关：全项目构建通过即可）
- [ ] **Step 6: Commit** — `git commit -m "feat: shared LLM transport + merge/dedup service with fail-open fallback"`

---

### Task 5: app 接入 DeepSeekOCR2Kit 本地包

**Files:**
- Modify: `WhisperScribe.xcodeproj/project.pbxproj`

**Interfaces:**
- Produces: app target 可 `import DeepSeekOCR2Kit`（T6 依赖）。

- [ ] **Step 1: pbxproj 手工接线**（objectVersion 77；生成 3 个互不重复的 24 位大写十六进制 ID，下文以 `AAAA…`/`BBBB…`/`CCCC…` 指代）

1. `objects` 尾部（`XCConfigurationList` 节之后）新增两节：

```
/* Begin XCLocalSwiftPackageReference section */
		AAAAAAAAAAAAAAAAAAAAAAAA /* XCLocalSwiftPackageReference "DeepSeekOCR2Kit" */ = {
			isa = XCLocalSwiftPackageReference;
			relativePath = DeepSeekOCR2Kit;
		};
/* End XCLocalSwiftPackageReference section */

/* Begin XCSwiftPackageProductDependency section */
		BBBBBBBBBBBBBBBBBBBBBBBB /* DeepSeekOCR2Kit */ = {
			isa = XCSwiftPackageProductDependency;
			productName = DeepSeekOCR2Kit;
		};
/* End XCSwiftPackageProductDependency section */
```

2. `PBXProject` 对象（含 `targets = (...)` 的那个）加属性：`packageReferences = (AAAAAAAAAAAAAAAAAAAAAAAA /* XCLocalSwiftPackageReference "DeepSeekOCR2Kit" */, );`
3. app target（`WhisperScribe` 的 `PBXNativeTarget`）加：`packageProductDependencies = (BBBBBBBBBBBBBBBBBBBBBBBB /* DeepSeekOCR2Kit */, );`
4. app target 的 `PBXFrameworksBuildPhase` 的 `files` 里加一个 `PBXBuildFile`：objects 的 PBXBuildFile 节新增 `CCCCCCCCCCCCCCCCCCCCCCCC /* DeepSeekOCR2Kit in Frameworks */ = {isa = PBXBuildFile; productRef = BBBBBBBBBBBBBBBBBBBBBBBB /* DeepSeekOCR2Kit */; };` 并把 `CCCC…` 加进该 build phase 的 files 列表。

- [ ] **Step 2: 冒烟验证**（在 `WhisperScribe/App/AppModel.swift` 顶部临时加 `import DeepSeekOCR2Kit` 再删除？——不，直接构建即验证解析；import 冒烟随 T6 一起）

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS' build 2>&1 | tail -3`
Expected: `** BUILD SUCCEEDED **`（首次会解析包依赖）。若 pbxproj 解析报错，逐条对照 Step 1 的四处改动。

- [ ] **Step 3: 测试套件不回归** — `xcodebuild test 2>&1 | tail -3` 全绿
- [ ] **Step 4: Commit** — `git commit -m "build: link DeepSeekOCR2Kit local package into app target"`

---

### Task 6: OCRProviding + OCRService

**Files:**
- Create: `WhisperScribe/Services/OCRService.swift`
- Test: `WhisperScribeTests/OCRServiceTests.swift`（注册）

**Interfaces:**
- Consumes: `OCR2Session.load(from:progress:)`、`session.ocr(image:task:) -> AsyncThrowingStream<String, Error>`（DeepSeekOCR2Kit 公共 API）。
- Produces:

```swift
protocol OCRProviding: Sendable {
    func prepare(modelDir: URL, progress: @escaping @Sendable (Double) -> Void) async throws
    func recognize(imageAt url: URL, onChunk: @escaping @Sendable (String) -> Void) async throws -> String
    func unload() async
}
actor OCRService: OCRProviding {}
```

- [ ] **Step 1: 写失败测试**（协议层——OCRService 真身依赖 3GB 模型无法单测，测试聚焦：协议存在、EXIF 图像加载 helper 行为、以及供 T8/9 使用的 fake）

```swift
import Testing
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Foundation
@testable import WhisperScribe

/// 供 BatchPipelineTests 复用的 fake（本文件内 internal）
final class FakeOCR: OCRProviding, @unchecked Sendable {
    var prepared = false
    var results: [String: String] = [:]           // fileName → text
    var failOn: Set<String> = []
    func prepare(modelDir: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        prepared = true; progress(1.0)
    }
    func recognize(imageAt url: URL, onChunk: @escaping @Sendable (String) -> Void) async throws -> String {
        let name = url.lastPathComponent
        if failOn.contains(name) { throw AppError.ocrFailed(name) }
        let text = results[name, default: "text-of-\(name)"]
        onChunk(text)
        return text
    }
    func unload() async { prepared = false }
}

struct OCRServiceTests {
    @Test func orientedImageLoaderHonorsEXIF() throws {
        // 生成一张 2x1 图，写入 orientation=6（右转90°），加载后应为 1x2
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appending(path: "o.jpg")
        let ctx = CGContext(data: nil, width: 2, height: 1, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let img = ctx.makeImage()!
        let dest = CGImageDestinationCreateWithURL(path as CFURL, UTType.jpeg.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, img, [kCGImagePropertyOrientation: 6] as CFDictionary)
        #expect(CGImageDestinationFinalize(dest))
        let loaded = try OCRService.loadOrientedImage(at: path)
        #expect(loaded.width == 1 && loaded.height == 2)
    }
    @Test func fakeConformsAndFails() async {
        let fake = FakeOCR(); fake.failOn = ["bad.png"]
        await #expect(throws: AppError.self) {
            _ = try await fake.recognize(imageAt: URL(fileURLWithPath: "/tmp/bad.png"), onChunk: { _ in })
        }
    }
}
```

- [ ] **Step 2: 注册 + 确认编译失败**
- [ ] **Step 3: 实现 OCRService**

```swift
import CoreGraphics
import ImageIO
import DeepSeekOCR2Kit

actor OCRService: OCRProviding {
    private var session: OCR2Session?

    func prepare(modelDir: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        guard session == nil else { progress(1.0); return }
        session = try await OCR2Session.load(from: modelDir, progress: progress)
    }
    func recognize(imageAt url: URL, onChunk: @escaping @Sendable (String) -> Void) async throws -> String {
        guard let session else { throw AppError.ocrModelMissing }
        let image = try Self.loadOrientedImage(at: url)
        var full = ""
        do {
            for try await chunk in session.ocr(image: image, task: .freeOCR) {
                full += chunk; onChunk(chunk)
            }
        } catch is CancellationError { throw CancellationError() }
        catch { throw AppError.ocrFailed(url.lastPathComponent) }
        return full
    }
    func unload() { session = nil }

    /// EXIF 方向归一化加载（模式取自 DeepSeekOCR2Kit/Sources/ocr2-cli/OCR2CLI.swift 的 loadOrientedImage，改造为 throws AppError.unsupportedFile）
    nonisolated static func loadOrientedImage(at url: URL) throws -> CGImage { /* 对照 CLI 实现移植 */ }
}
```

`loadOrientedImage`：读 `CGImageSourceCopyPropertiesAtIndex` 的 orientation，≠1 时用 `CGImageSourceCreateThumbnailAtIndex` + `kCGImageSourceCreateThumbnailWithTransform: true` + `kCGImageSourceThumbnailMaxPixelSize: max(w,h)`；=1 时直接 `CGImageSourceCreateImageAtIndex`。失败 → `AppError.unsupportedFile(url.lastPathComponent)`。**先读 CLI 的实现再移植**（路径见注释）。

- [ ] **Step 4: 测试通过**；**Step 5: Commit** — `git commit -m "feat: OCR service actor wrapping OCR2Session (EXIF-aware, protocol-injectable)"`

---

### Task 7: OCRModelManager + Hub 下载器

**Files:**
- Create: `WhisperScribe/Services/OCRModelManager.swift`
- Test: `WhisperScribeTests/OCRModelManagerTests.swift`（注册）

**Interfaces:**
- Consumes: `ModelDownloading` 协议（`Services/ModelDownloading.swift:5-9`）；`DownloadState`（`ModelManager.swift:9-13`——若为文件内非嵌套 enum 直接复用，若嵌套则提为共享，实施时按实际访问性选择并在报告说明）。
- Produces:

```swift
@MainActor final class OCRModelManager: ObservableObject {
    static let repoID = "mlx-community/DeepSeek-OCR-2-8bit"
    @Published var state: DownloadState = .idle
    @Published var installed: Bool = false
    let modelDir: URL          // 默认 ~/Documents/huggingface/models/<repoID>；测试可注入 baseDir
    init(downloader: ModelDownloading = OCRHubDownloader(), baseDir: URL? = nil)
    var isReady: Bool          // installed && state 非 downloading
    func refreshInstalled()    // config.json + tokenizer.json + 任一 .safetensors
    func download(); func cancelDownload(); func delete()
    func performDownload(token: Int) async   // 测试可 await（对照 ModelManager.swift:102 模式）
}
struct OCRHubDownloader: ModelDownloading { }  // HubApi.snapshot(from: Repo(id: repoID))
```

- [ ] **Step 1: 写失败测试**（完全仿 `ModelManagerTests.swift` 的 `FakeDownloader`/`makeManager` 模式——先读该文件，测试覆盖：下载成功→installed、失败→.failed、取消→回 idle、delete→未安装、就绪判定需三类文件齐全）

```swift
import Testing
import Foundation
@testable import WhisperScribe

private struct FakeOCRDownloader: ModelDownloading {
    var fail = false
    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        if fail { throw URLError(.networkConnectionLost) }
        progress(1.0)
        return URL(fileURLWithPath: "/tmp/unused")
    }
}

@MainActor private func makeOCRManager(_ d: ModelDownloading, dir: URL) -> OCRModelManager {
    OCRModelManager(downloader: d, baseDir: dir)
}
@MainActor private func plantModel(in dir: URL) throws {
    let m = dir.appending(path: OCRModelManager.repoID)
    try FileManager.default.createDirectory(at: m, withIntermediateDirectories: true)
    for f in ["config.json", "tokenizer.json", "model.safetensors"] {
        FileManager.default.createFile(atPath: m.appending(path: f).path, contents: Data("x".utf8))
    }
}

struct OCRModelManagerTests {
    @Test @MainActor func downloadSuccessMarksInstalled() async throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let m = makeOCRManager(FakeOCRDownloader(), dir: dir)
        try plantModel(in: dir)          // fake 下载器不真写盘——就绪由磁盘态判定
        await m.performDownload(token: 0)
        #expect(m.installed && m.isReady)
    }
    @Test @MainActor func downloadFailureSetsFailed() async {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let m = makeOCRManager(FakeOCRDownloader(fail: true), dir: dir)
        await m.performDownload(token: 0)
        if case .failed = m.state {} else { Issue.record("expected .failed, got \(m.state)") }
    }
    @Test @MainActor func readinessNeedsAllThreeFiles() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let m = makeOCRManager(FakeOCRDownloader(), dir: dir)
        let folder = dir.appending(path: OCRModelManager.repoID)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: folder.appending(path: "config.json").path, contents: Data())
        m.refreshInstalled()
        #expect(!m.installed)            // 缺 tokenizer/safetensors
    }
    @Test @MainActor func deleteRemovesAndResets() async throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let m = makeOCRManager(FakeOCRDownloader(), dir: dir)
        try plantModel(in: dir); m.refreshInstalled(); #expect(m.installed)
        m.delete()
        #expect(!m.installed)
    }
}
```

- [ ] **Step 2: 注册 + 确认编译失败**
- [ ] **Step 3: 实现**：生成令牌/进度钳制/单任务守卫机制**逐行对照** `ModelManager.swift:83-144` 移植（downloadSeq token、`clampedProgress`）；`OCRHubDownloader` 用 `import Hub`（swift-transformers 经 WhisperKit 已在依赖树）：`let hub = HubApi(downloadBase: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appending(path: "huggingface")); _ = try await hub.snapshot(from: Hub.Repo(id: Self.repoID), matching: ["*"]) { p in progress(p.fractionCompleted) }`（对照 `.spm/checkouts/WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift:244-295` 的用法）。
- [ ] **Step 4: 测试通过**；**Step 5: Commit** — `git commit -m "feat: OCR model manager (HubApi download, disk readiness, cancel/delete)"`

---

### Task 8: JobState 批扩展 + 音频批编排

**Files:**
- Modify: `WhisperScribe/Models/JobState.swift`
- Modify: `WhisperScribe/Services/TranscriberService.swift`（加 `func unload()`：置内部 WhisperKit 实例为 nil）
- Modify: `WhisperScribe/ViewModel/TranscriptionViewModel.swift`
- Modify: `WhisperScribe/App/AppModel.swift`（新服务实例化与注入）
- Test: `WhisperScribeTests/BatchPipelineTests.swift`（注册）

**Interfaces:**
- Consumes: T2 `BatchClassifier`、T3 `writeMergedText`、T4 `MergeService`、T6 `OCRProviding`/`FakeOCR`、T7 `OCRModelManager`。
- Produces:

```swift
// JobState 新 case（其余 case 形状不动）:
case recognizing(progress: Double)          // OCR 阶段
case merging(progress: Double, note: String)
// .done 改为: case done(outputs: [URL], warnings: [String])   ← 触及全部 switch 站点
struct BatchProgress: Equatable { let index: Int; let count: Int; let fileName: String }
// TranscriptionViewModel:
@Published var batch: BatchProgress?        // nil = 非批次/单文件
func start(urls: [URL])                     // 分类→路由; 原 start(url:) 改为转调 start(urls: [url])
// init 注入: transcriber, cleaner, modelManager(现有) + ocr: OCRProviding, ocrModels: OCRModelManager, merger: MergeService
```

- [ ] **Step 1: 写失败测试（音频批部分；图片批用例本任务先写好、以 fake 全链路一起过）**

```swift
import Testing
import Foundation
@testable import WhisperScribe

// 依赖注入用最小 fake（转写侧沿用现有测试基建思路; 若 TranscriberService 无协议缝, 先为其抽
// protocol Transcribing（transcribe+prepare+unload）——实施时按现有 TranscriberService 实际接口抽取,
// TranscriptionViewModel 改依赖协议, 生产代码注入真身。这是本任务授权的既定重构。）
```

测试用例清单（完整代码由实施者按上述 fake 基建写出，**断言逐条固定**）：
1. `imageBatchProducesMergedTxt`：3 张图（FakeOCR 返回 "A"/"B"/"C"，FakeChat 返回 "ABC-merged"）→ `.done(outputs:)` 恰 1 个 URL，内容 "ABC-merged"，`batch` 进度曾达 `(3,3)`。
2. `imageBatchSkipsFailedFile`：3 张图第 2 张 `failOn` → done 输出 1 个，warnings 含 fileSkipped 文案，merge 输入只含 1、3 两段。
3. `imageBatchAllFailedIsError`：全部 failOn → `.error`。
4. `imageBatchRequiresOCRModel`：`ocrModels.installed == false` → `.error(.ocrModelMissing)`，不触发 FakeOCR。
5. `mixedBatchIsImmediateError`：`start(urls: [a.mp3, b.png])` → `.error(.mixedBatch)`。
6. `cancelMidBatchReturnsIdle`：FakeOCR 挂起版（仿 `ModelManagerTests` 的 `SuspendingDownloader` Gate 模式）→ start 后 cancel → 状态回 `.idle`（沿用 jobToken 防旧任务写状态——对照 `TranscriptionViewModel.swift:27-45` 现有机制）。
7. `singleAudioStillWorks`：单音频 URL 走 `start(urls:)` → 行为与现状一致（fake 转写返回固定 segments → 出 1 srt + 1 txt；注意单音频批**不产** merged txt——`parts.count == 1` 时跳过 merge，输出沿用现有 srt/txt 对）。
8. `multiAudioProducesPerFileSrtPlusMerged`：2 段音频 → done 输出 3 个 URL（2 srt + 1 merged txt）；merged 内容来自 FakeChat。

- [ ] **Step 2: 注册 + 确认编译失败**
- [ ] **Step 3: 实现**

`TranscriptionViewModel.start(urls:)` 骨架（对照现有 `start(url:)` `:27-107` 的 jobToken/阶段推进模式扩展）：

```swift
func start(urls: [URL]) {
    guard !state.isBusy else { return }
    let kind: BatchKind
    do { kind = try BatchClassifier.classify(urls) }
    catch { state = .error(error as? AppError ?? .cancelled); return }
    jobToken += 1; let token = jobToken
    task = Task { [weak self] in
        switch kind {
        case .audio(let files):  await self?.runAudioBatch(files, token: token)
        case .images(let files): await self?.runImageBatch(files, token: token)
        }
    }
}
```

`runAudioBatch`：for (i, url) in files：更新 `batch = BatchProgress(index: i+1, count: files.count, fileName: url.lastPathComponent)` → 现有单文件五步流水线（decode→transcribe→clean→write srt/txt）逐文件执行，收集 `(name, cleanedTxt)` 与警告；单文件 throw → 警告 `cleanup.warning.fileSkipped` 继续；全失败 → `.error`。`files.count > 1` 时：`.merging` 状态 → `merger.merge(...)` → `writeMergedText(firstSource: files[0]...)`；outputs = 各 srt + 各 txt? **设计定案**：多音频批每文件仍写 srt（时间轴产物），**逐文件 txt 不再单独写**（合并 txt 取代），outputs = N 个 srt + 1 个 merged txt。单文件批完全走现状（srt + txt）。
`runImageBatch`：前置 `guard ocrModels.isReady else { .error(.ocrModelMissing) }` → `transcriber.unload()` → `.loadingModel`?（用新键 `status.loadingOCRModel` 的展示放 StatusView，状态可复用 `.loadingModel`）→ `ocr.prepare(modelDir:progress:)` → for 每图 `.recognizing(progress:)`（progress = 已完成文件数/总数 + 当前文件 chunk 计数注记）→ 收集 → `.merging` → merge → writeMergedText → `.done(outputs: [merged], warnings:)`。
音频批开头对称地 `await ocr.unload()`。
`.done` 形状变更：更新 `StatusView` 的 done 分支与 `revealInFinder`（outputs.first 所在目录）——**本任务只改到编译通过 + 测试绿**，视觉细化留 T10。
`AppModel`：实例化 `ocrModels/ocr/merger` 并注入 VM；`WhisperScribeApp` 若需 environmentObject 传递 `ocrModels` 给 Settings，T10 处理。

- [ ] **Step 4: 全套测试通过**（`xcodebuild test 2>&1 | tail -3`）
- [ ] **Step 5: Commit** — `git commit -m "feat: batch state machine — audio/image batch orchestration with merge"`

---

### Task 9: 内存互斥与取消收尾（编排细节）

**Files:**
- Modify: `WhisperScribe/ViewModel/TranscriptionViewModel.swift`（若 T8 已覆盖则本任务为验证性收尾）
- Test: `WhisperScribeTests/BatchPipelineTests.swift`（追加）

**Interfaces:** 无新接口；固化两条不变量。

- [ ] **Step 1: 追加失败测试**

1. `imageBatchUnloadsTranscriber`：fake Transcribing 记录 `unloadCalled` → 图片批开跑后为 true。
2. `audioBatchUnloadsOCR`：FakeOCR 记录 unload → 音频批开跑后为 true。
3. `cancelDuringMergeReturnsIdle`：FakeChat 挂起版（Gate）→ merge 中 cancel → `.idle` 且无产物写盘。

- [ ] **Step 2: 确认失败（若 T8 实现已顺带满足则记录为即时绿并说明）**
- [ ] **Step 3: 实现/补齐**（merge 调用点包 `Task.isCancelled` 检查；写盘前最后检查一次 token）
- [ ] **Step 4: 全绿**；**Step 5: Commit** — `git commit -m "feat: model-memory exclusivity + cancellation hardening for batches"`

---

### Task 10: UI 层——多选、拖放、状态展示、设置 OCR 段

**Files:**
- Modify: `WhisperScribe/Views/DropZone.swift`
- Modify: `WhisperScribe/Views/ContentView.swift`
- Modify: `WhisperScribe/Views/StatusView.swift`
- Modify: `WhisperScribe/Views/SettingsView.swift`
- Modify: `WhisperScribe/App/WhisperScribeApp.swift`（environmentObject 注入 `OCRModelManager`）

**Interfaces:**
- Consumes: `BatchClassifier.isAcceptedURL`、`viewModel.start(urls:)`、`viewModel.batch`、`OCRModelManager`（`state/installed/download/cancelDownload/delete`）、T1 全部键。

- [ ] **Step 1: DropZone 多文件**：`var onPick: ([URL]) -> Void`；`handleDrop` 收集**全部** providers 的 fileURL（`withTaskGroup`/DispatchGroup 聚齐后一次回调，对照现有 `:40-49` 改造）；类型高亮判定改 `BatchClassifier.isAcceptedURL`；副标题文案换 `drop.subtitleMulti`。
- [ ] **Step 2: ContentView**：`chooseFile()` 的 NSOpenPanel `allowsMultipleSelection = true`、`allowedContentTypes = [.audio, .audiovisualContent, .image]`（`:88-99`）；回调 `viewModel.start(urls: panel.urls)`；DropZone 闭包同步改。
- [ ] **Step 3: StatusView**：顶部若 `viewModel.batch != nil` 显示 `batch.fileProgress` 格式行（`String.localizedStringWithFormat(NSLocalizedString("batch.fileProgress", comment: ""), b.index, b.count, b.fileName)`）；新增 `.recognizing` 分支（determinate bar + `status.recognizing`）与 `.merging` 分支（`status.merging` + note）；`.done` 分支改产物列表（逐行文件名 + `done.batchSummary` 摘要 + 现有 Reveal 按钮语义改指向 outputs）。
- [ ] **Step 4: SettingsView**：`modelSection` 之后新增 `ocrModelSection`——单行条目：`settings.ocrModel.title`/`tagline`/`size` + accessory 复用现有 `modelAccessory` 的三态形状（downloading→进度+取消、failed→重试、idle→已装(勾+删除)/下载按钮），绑定 `OCRModelManager`（`@EnvironmentObject`）。**对照 `SettingsView.swift:45-99` 现有行结构实现，风格一致**。
- [ ] **Step 5: 构建 + 全测试绿 + 手动冒烟**（打开 app：拖 2 张图（无 OCR 模型时提示去设置）；设置页看到 OCR 模型条目）

Run: `xcodebuild test 2>&1 | tail -3` && `open build 产物或 ⌘R 手测`（手测结果记录在提交信息 body）
- [ ] **Step 6: Commit** — `git commit -m "feat: multi-file UI — drop/picker multi-select, batch status, OCR model settings"`

---

### Task 11: 改名 OmniScribe + README 重写 + 端到端验证

**Files:**
- Modify: `WhisperScribe.xcodeproj/project.pbxproj`（app target 两个 configuration 的 buildSettings 各加 `INFOPLIST_KEY_CFBundleDisplayName = OmniScribe;`——只加 app target，不动 tests target）
- Modify: `README.md`

**Interfaces:** 无。

- [ ] **Step 1: 显示名**：pbxproj 两处 buildSettings 添加；构建后验证：

Run: `xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -configuration Debug -destination 'platform=macOS' build 2>&1 | tail -2 && defaults read "$(xcodebuild -project WhisperScribe.xcodeproj -scheme WhisperScribe -configuration Debug -destination 'platform=macOS' -showBuildSettings 2>/dev/null | awk '/ BUILT_PRODUCTS_DIR/{print $3}')/WhisperScribe.app/Contents/Info.plist" CFBundleDisplayName`
Expected: `OmniScribe`

- [ ] **Step 2: README 重写**（保留结构：徽章/Install/Requirements/Build&run/License；重写定位段与 Features）：

标题改 `# OmniScribe`；副题定位句：`A native macOS app that turns audio files or stacks of images into clean text. Drop files in — audio is transcribed locally on the Neural Engine (WhisperKit), images are OCR'd locally on the GPU (DeepSeek-OCR-2 via mlx-swift), and an optional BYOK LLM merges, deduplicates and polishes the result.`；Features 增：多文件批次玩法两例（滚动截图→合并去重 txt；分段录音→逐段 srt + 合并 txt）、OCR 模型在 Settings ▸ Model 下载（~3GB）；Architecture 图补 `BatchClassifier/OCRService(DeepSeekOCR2Kit)/MergeService` 三个节点；说明仓库名仍为 WhisperScribe（历史沿革一句话）；链接 `DeepSeekOCR2Kit/README.md`。

- [ ] **Step 3: 全量验证**

1. `xcodebuild test 2>&1 | tail -3` 全绿；`cd DeepSeekOCR2Kit && xcodebuild test -scheme DeepSeekOCR2Kit-Package -destination 'platform=macOS' -skipPackagePluginValidation -skipMacroValidation 2>&1 | tail -3` 全绿。
2. 手动端到端（记录在提交 body）：设置页下载 OCR 模型（真 3GB）→ 拖 4 张滚动截图 → 得合并去重 txt；拖 2 段音频 → 2 srt + 1 合并 txt；混投报错；中途取消回 idle；Dock/关于显示 OmniScribe。
3. 推分支开 PR，CI 全绿后合并（沿用 rebase 流程）。

- [ ] **Step 4: Commit** — `git commit -m "feat: rename display name to OmniScribe + reposition README for audio/images→text"`

---

## 验证总表

| 关卡 | 内容 | 任务 |
|---|---|---|
| 单元 | 分类/排序/命名/merge 降级/OCR 管理器/EXIF 加载 | 2,3,4,6,7 |
| 状态机 | 8 条批次用例 + 3 条互斥/取消用例（全 fake 注入） | 8,9 |
| 构建 | 每任务 xcodebuild test 全绿；kit 套件不回归 | 全部 |
| 手动 | 拖放/下载/端到端/改名 四清单 | 10,11 |
| CI | PR 全绿后合并 | 11 |

执行纪律：`.done` 形状变更（T8）触及全部 switch 站点——**编译错误清单就是改造地图，禁止用 default 分支糊过**；所有新增用户可见字符串必须来自 T1 的键，发现缺键回 T1 补齐而非硬编码。
