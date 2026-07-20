# WhisperScribe CI/CD + 自动 Release 设计

日期：2026-07-20 ｜ 状态：已确认（方案 A）

## 背景与目标

仓库目前没有任何 CI/CD：推代码后没有自动构建/测试，发版全靠本地手动 Archive。
目标：

1. **CI** — 每次 push / PR 自动构建并跑测试（`WhisperScribeTests`，纯逻辑、不碰网络，适合 CI）。
2. **CD + 自动 Release** — 推 `v*` tag 后自动构建 Release 版、打 `.dmg`、创建 GitHub Release。

已确认的决策：

| 决策点 | 结论 |
|---|---|
| 签名 | **ad-hoc**（用户有开发者账号但不用于本项目）；workflow 内留注释版 Developer ID + 公证扩展位 |
| 触发 | **推 `v*` tag**；另留 `workflow_dispatch` 手动入口 |
| 产物 | **`.dmg`**（hdiutil 打包，含 Applications 快捷方式）+ SHA256 校验文件 |
| 版本选型 | `macos-26` runner（GA，默认 Xcode 26.4.1）、`actions/checkout@v7`、`actions/cache@v6`，发布用预装 `gh` CLI —— 零第三方 action |

## 架构：两个独立 workflow

### 1. `.github/workflows/ci.yml`

- 触发：`push`（main）+ `pull_request`（main）。
- `concurrency`：同分支新 run 取消旧 run。
- `permissions: contents: read`；`timeout-minutes: 30`。
- 步骤：
  1. `actions/checkout@v7`
  2. `actions/cache@v6` 缓存 `.spm`（`-clonedSourcePackagesDirPath`），key = `Package.resolved` 哈希（WhisperKit / swift-transformers 等依赖较大，缓存显著提速）
  3. `xcodebuild test -project WhisperScribe.xcodeproj -scheme WhisperScribe -destination 'platform=macOS,arch=arm64' -configuration Debug -clonedSourcePackagesDirPath .spm -skipPackagePluginValidation -skipMacroValidation`（`set -o pipefail`）

共享 scheme（含 TestAction）与 `Package.resolved` 均已提交，无需改动工程。

### 2. `.github/workflows/release.yml`

- 触发：`push` tags `v*`；`workflow_dispatch`（输入已有 tag 名，checkout 该 ref 重发）。
- `permissions: contents: write`。
- 步骤：
  1. checkout + SPM 缓存（同 CI）
  2. **测试门禁**：同 CI 的 `xcodebuild test`——tag 打在未测提交上时不发布
  3. Release 构建：`xcodebuild build -configuration Release -derivedDataPath build`，命令行注入版本号（不改 pbxproj）：
     - `MARKETING_VERSION` = tag 去掉 `v` 前缀（如 `v1.2.3` → `1.2.3`）
     - `CURRENT_PROJECT_VERSION` = `$GITHUB_RUN_NUMBER`
  4. `codesign --verify --deep --strict` 校验产物（ad-hoc 签名由构建自动完成）
  5. 打 dmg：staging 目录放 `WhisperScribe.app` + `Applications` 软链 → `hdiutil create -format UDZO` → `WhisperScribe-<版本>.dmg`；`shasum -a 256` 生成 `checksums.txt`
  6. `gh release create "$TAG" <dmg> <checksums> --title "WhisperScribe <版本>" --generate-notes`（`GH_TOKEN` 用内置 `github.token`）
- 文件末尾以注释保留 Developer ID 签名 + `notarytool submit --wait` + `stapler staple` 步骤及所需 Secrets 说明，日后配好 Secrets 即可启用。

## README 更新

- 顶部加 CI 徽章。
- 新增 **Install** 一节：从 Releases 下载 dmg → 拖入 Applications → **首次运行前必须执行**
  `xattr -dr com.apple.quarantine /Applications/WhisperScribe.app`（`-r` 递归清掉 bundle 内所有文件的 quarantine）
  （ad-hoc 签名的 app 带 quarantine 属性时 Gatekeeper 直接报「已损坏」，右键打开也无效，必须写清楚这条命令。）

## 其他改动

- `.gitignore` 追加 `.spm/` 与 `build/`（本地复现 CI 命令时的产物目录）。

## 错误处理

- 测试失败 → release job 直接失败，不产生 Release。
- `workflow_dispatch` 输入的 tag 不存在 → checkout 失败即中止。
- dmg/签名校验失败 → 中止，不发布半成品。

## 验证

1. 推 `ci-cd` 分支开 PR → 确认 `ci.yml` 触发并通过（首跑无缓存较慢，二跑验证缓存命中）。
2. 合并后推测试 tag `v0.9.9` → 确认 release workflow 全链路通过、Release 出现、dmg 可下载可挂载、app 可启动、「关于」里版本号为 0.9.9。
3. 验证后删除测试 Release 与 tag。
