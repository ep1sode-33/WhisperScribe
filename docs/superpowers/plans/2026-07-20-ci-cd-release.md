# WhisperScribe CI/CD + 自动 Release 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 WhisperScribe 添加 GitHub Actions CI（push/PR 构建+测试）与推 `v*` tag 自动构建 `.dmg` 并发布 GitHub Release 的 CD 流水线。

**Architecture:** 两个独立 workflow：`ci.yml`（push/PR → SPM 缓存 → `xcodebuild test`）与 `release.yml`（tag → 测试门禁 → Release 构建（命令行注入版本号）→ codesign 校验 → hdiutil 打 dmg → `gh release create`）。零第三方 action：仅官方 `actions/checkout` + `actions/cache`，发布用 runner 预装的 `gh` CLI，打包用系统 `hdiutil`。

**Tech Stack:** GitHub Actions（`macos-26` runner，默认 Xcode 26.4.1）、xcodebuild、hdiutil、gh CLI。

**设计文档:** `docs/superpowers/specs/2026-07-20-ci-cd-release-design.md`

## Global Constraints

- Runner 一律 `macos-26`；action 版本固定 `actions/checkout@v7`、`actions/cache@v6`；不引入任何第三方 action。
- 签名方式为 **ad-hoc**（工程默认 `CODE_SIGN_IDENTITY = "-"`），不配置证书/Secrets；Developer ID + 公证仅以注释形式留在 `release.yml` 末尾。
- 版本号注入只走 xcodebuild 命令行参数（`MARKETING_VERSION` / `CURRENT_PROJECT_VERSION`），**不得修改 pbxproj**。
- 本地验证命令与 CI 命令保持逐字一致（同样的 `-clonedSourcePackagesDirPath .spm` 等参数），确保「本地能跑 ⇒ CI 能跑」。
- 所有含 xcodebuild 管道的 run 步骤开头必须 `set -o pipefail`。
- 提交信息末尾带:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01U5wGZ5BEeMyPZd9jcPP34f`

---

### Task 1: `.gitignore` 追加 + `ci.yml`

**Files:**
- Modify: `.gitignore`（文件末尾追加）
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `.spm` 目录约定（`-clonedSourcePackagesDirPath .spm`）与 SPM 缓存 key 模式 `spm-${{ runner.os }}-${{ hashFiles('**/Package.resolved') }}`，Task 2 的 release.yml 必须复用一致的路径与 key。

- [ ] **Step 1: 追加 .gitignore**

在 `.gitignore` 末尾追加：

```gitignore

# CI-equivalent local runs (xcodebuild -clonedSourcePackagesDirPath / -derivedDataPath / dmg staging)
.spm/
build/
dist/
```

- [ ] **Step 2: 写 ci.yml**

创建 `.github/workflows/ci.yml`，内容如下（完整文件）：

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  test:
    name: Build & Test
    runs-on: macos-26
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v7

      - name: Cache Swift packages
        uses: actions/cache@v6
        with:
          path: .spm
          key: spm-${{ runner.os }}-${{ hashFiles('**/Package.resolved') }}
          restore-keys: |
            spm-${{ runner.os }}-

      - name: Build and test
        run: |
          set -o pipefail
          xcodebuild test \
            -project WhisperScribe.xcodeproj \
            -scheme WhisperScribe \
            -configuration Debug \
            -destination 'platform=macOS,arch=arm64' \
            -clonedSourcePackagesDirPath .spm \
            -skipPackagePluginValidation \
            -skipMacroValidation
```

- [ ] **Step 3: 校验 YAML 语法**

Run: `ruby -ryaml -e "YAML.load_file('.github/workflows/ci.yml'); puts 'ci.yml OK'"`
Expected: 输出 `ci.yml OK`（macOS 自带 ruby，无需安装）

- [ ] **Step 4: 本地跑与 CI 逐字一致的测试命令（这是本任务的"失败测试→通过"验证）**

Run:

```bash
set -o pipefail
xcodebuild test \
  -project WhisperScribe.xcodeproj \
  -scheme WhisperScribe \
  -configuration Debug \
  -destination 'platform=macOS,arch=arm64' \
  -clonedSourcePackagesDirPath .spm \
  -skipPackagePluginValidation \
  -skipMacroValidation
```

Expected: 末尾出现 `** TEST SUCCEEDED **`（首跑会把 SPM 依赖克隆进 `.spm/`，需几分钟）

- [ ] **Step 5: 确认 `.spm/`、`build/` 未被 git 追踪**

Run: `git status --porcelain | grep -E "\.spm|^\?\? build" || echo "ignore OK"`
Expected: 输出 `ignore OK`（`.spm/` 因 .gitignore 生效不出现在 untracked 里）

- [ ] **Step 6: Commit**

```bash
git add .gitignore .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions build+test workflow (macos-26, checkout@v7, cache@v6)"
```

---

### Task 2: `release.yml`

**Files:**
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: Task 1 的 `.spm` 路径与缓存 key 模式（必须逐字一致，release 才能命中 CI 建好的缓存）。
- Produces: Release 资产命名 `WhisperScribe-<version>.dmg` + `checksums.txt`；README（Task 3）的下载说明依赖该命名。

- [ ] **Step 1: 写 release.yml**

创建 `.github/workflows/release.yml`，内容如下（完整文件）：

```yaml
name: Release

on:
  push:
    tags: ['v*']
  workflow_dispatch:
    inputs:
      tag:
        description: 'Existing tag to (re-)release, e.g. v1.2.3'
        required: true
        type: string

permissions:
  contents: write

concurrency:
  group: release-${{ inputs.tag || github.ref_name }}
  cancel-in-progress: false

jobs:
  release:
    name: Test, build, package, publish
    runs-on: macos-26
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v7
        with:
          ref: ${{ inputs.tag || github.ref }}

      - name: Derive version from tag
        id: version
        run: |
          TAG="${{ inputs.tag || github.ref_name }}"
          if [[ ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.-]+)?$ ]]; then
            echo "::error::Tag '$TAG' is not v<major>.<minor>.<patch>[-suffix]"
            exit 1
          fi
          echo "tag=$TAG" >> "$GITHUB_OUTPUT"
          echo "version=${TAG#v}" >> "$GITHUB_OUTPUT"

      - name: Cache Swift packages
        uses: actions/cache@v6
        with:
          path: .spm
          key: spm-${{ runner.os }}-${{ hashFiles('**/Package.resolved') }}
          restore-keys: |
            spm-${{ runner.os }}-

      - name: Test gate
        run: |
          set -o pipefail
          xcodebuild test \
            -project WhisperScribe.xcodeproj \
            -scheme WhisperScribe \
            -configuration Debug \
            -destination 'platform=macOS,arch=arm64' \
            -clonedSourcePackagesDirPath .spm \
            -skipPackagePluginValidation \
            -skipMacroValidation

      - name: Build Release
        run: |
          set -o pipefail
          xcodebuild build \
            -project WhisperScribe.xcodeproj \
            -scheme WhisperScribe \
            -configuration Release \
            -destination 'platform=macOS,arch=arm64' \
            -derivedDataPath build \
            -clonedSourcePackagesDirPath .spm \
            -skipPackagePluginValidation \
            -skipMacroValidation \
            MARKETING_VERSION="${{ steps.version.outputs.version }}" \
            CURRENT_PROJECT_VERSION="${{ github.run_number }}"

      - name: Verify code signature (ad-hoc)
        run: |
          codesign --verify --deep --strict --verbose=2 \
            "build/Build/Products/Release/WhisperScribe.app"

      - name: Package DMG
        run: |
          VERSION="${{ steps.version.outputs.version }}"
          mkdir -p dist/dmg
          cp -R "build/Build/Products/Release/WhisperScribe.app" dist/dmg/
          ln -s /Applications dist/dmg/Applications
          hdiutil create \
            -volname "WhisperScribe $VERSION" \
            -srcfolder dist/dmg \
            -ov -format UDZO \
            "dist/WhisperScribe-$VERSION.dmg"
          (cd dist && shasum -a 256 "WhisperScribe-$VERSION.dmg" > checksums.txt)

      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          VERSION="${{ steps.version.outputs.version }}"
          gh release create "${{ steps.version.outputs.tag }}" \
            "dist/WhisperScribe-$VERSION.dmg" \
            "dist/checksums.txt" \
            --title "WhisperScribe $VERSION" \
            --verify-tag \
            --generate-notes

# ── Developer ID signing + notarization (currently disabled) ──────────────
# 想改成正式签名+公证发布时：
#   1. 配置 repo Secrets:
#        MACOS_CERT_P12_BASE64   Developer ID Application 证书 .p12 的 base64
#        MACOS_CERT_PASSWORD     .p12 密码
#        APPLE_TEAM_ID           团队 ID
#        NOTARY_KEY_ID / NOTARY_ISSUER_ID / NOTARY_KEY_BASE64
#                                App Store Connect API key (.p8)
#   2. 在 "Package DMG" 之前插入步骤：导入证书到临时 keychain 并重签：
#        security create-keychain -p ci build.keychain
#        security import cert.p12 -k build.keychain -P "$MACOS_CERT_PASSWORD" \
#          -T /usr/bin/codesign
#        security set-key-partition-list -S apple-tool:,apple: -s -k ci build.keychain
#        codesign --force --deep --options runtime \
#          --sign "Developer ID Application: <name> ($APPLE_TEAM_ID)" \
#          build/Build/Products/Release/WhisperScribe.app
#   3. 在 "Package DMG" 之后、"Create GitHub Release" 之前插入公证：
#        echo "$NOTARY_KEY_BASE64" | base64 -d > key.p8
#        xcrun notarytool submit "dist/WhisperScribe-$VERSION.dmg" \
#          --key key.p8 --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID" --wait
#        xcrun stapler staple "dist/WhisperScribe-$VERSION.dmg"
#   4. README 的 Install 一节删掉 xattr 说明。
```

- [ ] **Step 2: 校验 YAML 语法**

Run: `ruby -ryaml -e "YAML.load_file('.github/workflows/release.yml'); puts 'release.yml OK'"`
Expected: 输出 `release.yml OK`

- [ ] **Step 3: 本地验证发布链路的构建/校验/打包三步（版本号用 0.0.0）**

Run:

```bash
set -o pipefail
xcodebuild build \
  -project WhisperScribe.xcodeproj \
  -scheme WhisperScribe \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath build \
  -clonedSourcePackagesDirPath .spm \
  -skipPackagePluginValidation \
  -skipMacroValidation \
  MARKETING_VERSION="0.0.0" \
  CURRENT_PROJECT_VERSION="999" \
&& codesign --verify --deep --strict --verbose=2 build/Build/Products/Release/WhisperScribe.app \
&& rm -rf dist && mkdir -p dist/dmg \
&& cp -R build/Build/Products/Release/WhisperScribe.app dist/dmg/ \
&& ln -s /Applications dist/dmg/Applications \
&& hdiutil create -volname "WhisperScribe 0.0.0" -srcfolder dist/dmg -ov -format UDZO dist/WhisperScribe-0.0.0.dmg \
&& (cd dist && shasum -a 256 WhisperScribe-0.0.0.dmg > checksums.txt) \
&& hdiutil attach -nobrowse -readonly dist/WhisperScribe-0.0.0.dmg \
&& defaults read "/Volumes/WhisperScribe 0.0.0/WhisperScribe.app/Contents/Info.plist" CFBundleShortVersionString \
&& hdiutil detach "/Volumes/WhisperScribe 0.0.0"
```

Expected: codesign 输出 `valid on disk` / `satisfies its Designated Requirement`；`defaults read` 输出 `0.0.0`（证明命令行注入版本号生效）；dmg 挂载/卸载成功

- [ ] **Step 4: 清理本地产物**

Run: `rm -rf dist build`
Expected: 无输出；`git status` 不含 dist/build

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add tag-triggered release workflow (dmg + gh release, ad-hoc signed)"
```

---

### Task 3: README 徽章 + Install 一节

**Files:**
- Modify: `README.md`（标题下加徽章；`## Requirements` 之前插入 `## Install`；`## Build & run` 末段提一句 Releases）

**Interfaces:**
- Consumes: Task 2 的资产命名 `WhisperScribe-<version>.dmg` 与 Releases 页面。

- [ ] **Step 1: 标题下加 CI 徽章**

在 `# WhisperScribe` 标题行之后、简介段之前插入（空行隔开）：

```markdown
[![CI](https://github.com/ep1sode-33/WhisperScribe/actions/workflows/ci.yml/badge.svg)](https://github.com/ep1sode-33/WhisperScribe/actions/workflows/ci.yml)
```

- [ ] **Step 2: 插入 Install 一节**

在 `## Requirements` 之前插入：

````markdown
## Install

Download the latest `WhisperScribe-<version>.dmg` from
[Releases](https://github.com/ep1sode-33/WhisperScribe/releases), open it and drag
**WhisperScribe** into **Applications**.

The app is **ad-hoc signed** (not notarized), so on first launch Gatekeeper will
claim it is "damaged". Clear the quarantine flag once and it opens normally:

```bash
xattr -dr com.apple.quarantine /Applications/WhisperScribe.app
```

Prefer building from source? See [Build & run](#build--run).

````

- [ ] **Step 3: Build & run 一节补一句**

把现有段落：

```markdown
The app is unsandboxed and ad-hoc signed ("Sign to Run Locally") — no developer
account needed to run it on your own Mac. For a distributable `.app`, use
**Product ▸ Archive**.
```

改为：

```markdown
The app is unsandboxed and ad-hoc signed ("Sign to Run Locally") — no developer
account needed to run it on your own Mac. Distributable `.dmg`s are built
automatically by [the release workflow](.github/workflows/release.yml) when a
`v*` tag is pushed; for a one-off local build use **Product ▸ Archive**.
```

- [ ] **Step 4: 校验 markdown 链接锚点**

Run: `grep -n "build--run\|badge.svg\|releases" README.md`
Expected: 三处都能 grep 到（徽章、Releases 链接、`#build--run` 锚点）

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add CI badge and Install section (dmg + quarantine note)"
```

---

### Task 4: 推分支开 PR，实测 CI

**Files:** 无新文件（远端验证）

**Interfaces:**
- Consumes: Task 1 的 `ci.yml`。

- [ ] **Step 1: 推分支并开 PR**

```bash
git push -u origin ci-cd
gh pr create --title "CI/CD: build+test workflow and tag-triggered dmg release" \
  --body "$(cat <<'EOF'
Adds GitHub Actions CI (push/PR → xcodebuild test on macos-26) and a tag-triggered
release workflow (test gate → Release build with version injected from tag →
ad-hoc codesign verify → hdiutil dmg → gh release create --generate-notes).
Zero third-party actions: checkout@v7 + cache@v6 + preinstalled gh CLI only.

Design: docs/superpowers/specs/2026-07-20-ci-cd-release-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01U5wGZ5BEeMyPZd9jcPP34f
EOF
)"
```

Expected: PR URL 输出

- [ ] **Step 2: 等 CI 跑完**

Run: `gh run watch --exit-status $(gh run list --workflow=ci.yml --branch=ci-cd --limit 1 --json databaseId -q '.[0].databaseId')`
Expected: 退出码 0，job `Build & Test` 绿

- [ ] **Step 3: 验证缓存生效（第二跑）**

推一个空提交触发二跑，确认 `Cache Swift packages` 步骤显示 `Cache restored from key: spm-...`：

```bash
git commit --allow-empty -m "ci: verify SPM cache hit"
git push
gh run watch --exit-status $(gh run list --workflow=ci.yml --branch=ci-cd --limit 1 --json databaseId -q '.[0].databaseId')
```

Expected: 二跑明显快于首跑；日志含 `Cache restored`

---

### Task 5: 合并后全链路发布演练

**Files:** 无（远端验证；需要用户确认合并 PR 后进行）

**Interfaces:**
- Consumes: Task 2 的 `release.yml`。

- [ ] **Step 1: 合并 PR 后，打测试 tag**

```bash
git checkout main && git pull
git tag v0.9.9
git push origin v0.9.9
```

- [ ] **Step 2: 等 release workflow 跑完**

Run: `gh run watch --exit-status $(gh run list --workflow=release.yml --limit 1 --json databaseId -q '.[0].databaseId')`
Expected: 退出码 0

- [ ] **Step 3: 验证 Release 与产物**

```bash
gh release view v0.9.9 --json assets,name -q '{name: .name, assets: [.assets[].name]}'
gh release download v0.9.9 --dir /tmp/ws-release-test
hdiutil attach -nobrowse -readonly /tmp/ws-release-test/WhisperScribe-0.9.9.dmg
defaults read "/Volumes/WhisperScribe 0.9.9/WhisperScribe.app/Contents/Info.plist" CFBundleShortVersionString
cp -R "/Volumes/WhisperScribe 0.9.9/WhisperScribe.app" /tmp/ws-release-test/
hdiutil detach "/Volumes/WhisperScribe 0.9.9"
xattr -dr com.apple.quarantine /tmp/ws-release-test/WhisperScribe.app
open /tmp/ws-release-test/WhisperScribe.app
```

Expected: assets 含 `WhisperScribe-0.9.9.dmg` 与 `checksums.txt`；版本号输出 `0.9.9`；app 正常启动

- [ ] **Step 4: 清理测试 Release**

```bash
gh release delete v0.9.9 --yes
git push origin :refs/tags/v0.9.9
git tag -d v0.9.9
rm -rf /tmp/ws-release-test
```

Expected: Releases 页面无 v0.9.9

- [ ] **Step 5: （由用户决定）发首个正式版**

```bash
git tag v1.0.0 && git push origin v1.0.0
```
