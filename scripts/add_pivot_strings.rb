#!/usr/bin/env ruby
# Adds the multi-file pivot strings (batch progress, OCR, merge fallback, rename prep)
# to Localizable.xcstrings in all 5 languages (en/zh-Hans/zh-Hant/ja/ko). Idempotent
# upsert of exactly the 15 keys below — overwrites these keys to their canonical values,
# leaves every other key untouched. Mirrors scripts/update_strings.rb's entry helper
# (extractionState: "manual", every language state: "translated").
#
# Format-string keys (batch.fileProgress, done.batchSummary, error.unsupportedImage,
# error.ocrFailed, cleanup.warning.fileSkipped) use printf-style positional/plain
# placeholders (%1$d/%2$d/%3$@/%@) consumed via String.localizedStringWithFormat.
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
  'drop.subtitleMulti'          => entry(en: 'Audio files or images — drop several at once',
                                          zh_hans: '音频或图片，可一次拖入多个',
                                          zh_hant: '音訊或圖片，可一次拖入多個',
                                          ja: '音声または画像 — 複数まとめてドロップ可',
                                          ko: '오디오 또는 이미지 — 여러 개 동시에 드롭 가능'),
  'status.recognizing'          => entry(en: 'Recognizing text…',
                                          zh_hans: '正在识别文字…',
                                          zh_hant: '正在識別文字…',
                                          ja: '文字を認識中…',
                                          ko: '텍스트 인식 중…'),
  'status.merging'              => entry(en: 'Merging & deduplicating…',
                                          zh_hans: '正在拼接去重…',
                                          zh_hant: '正在拼接去重…',
                                          ja: '結合・重複除去中…',
                                          ko: '병합 및 중복 제거 중…'),
  'status.loadingOCRModel'      => entry(en: 'Loading OCR model…',
                                          zh_hans: '正在加载 OCR 模型…',
                                          zh_hant: '正在載入 OCR 模型…',
                                          ja: 'OCR モデルを読み込み中…',
                                          ko: 'OCR 모델 로드 중…'),
  'batch.fileProgress'          => entry(en: 'File %1$d of %2$d — %3$@',
                                          zh_hans: '第 %1$d/%2$d 个文件 — %3$@',
                                          zh_hant: '第 %1$d/%2$d 個檔案 — %3$@',
                                          ja: 'ファイル %1$d/%2$d — %3$@',
                                          ko: '파일 %1$d/%2$d — %3$@'),
  'done.batchSummary'           => entry(en: '%1$d files → %2$d outputs',
                                          zh_hans: '%1$d 个文件 → %2$d 个产物',
                                          zh_hant: '%1$d 個檔案 → %2$d 個產物',
                                          ja: '%1$d ファイル → %2$d 出力',
                                          ko: '파일 %1$d개 → 출력 %2$d개'),
  'error.mixedBatch'            => entry(en: "Mixing audio and images isn't supported — drop one kind at a time.",
                                          zh_hans: '不支持音频和图片混合投放——一次只能拖同一类文件。',
                                          zh_hant: '不支援音訊與圖片混合投放——一次只能拖同一類檔案。',
                                          ja: '音声と画像の混在には対応していません。同じ種類のみドロップしてください。',
                                          ko: '오디오와 이미지를 섞을 수 없습니다. 한 번에 한 종류만 드롭하세요.'),
  'error.unsupportedImage'      => entry(en: 'Unsupported file: %@',
                                          zh_hans: '不支持的文件：%@',
                                          zh_hant: '不支援的檔案：%@',
                                          ja: '対応していないファイル: %@',
                                          ko: '지원하지 않는 파일: %@'),
  'error.ocrModelMissing'       => entry(en: "The OCR model isn't downloaded yet. Get it in Settings ▸ Model.",
                                          zh_hans: 'OCR 模型尚未下载，请到 设置 ▸ 模型 下载。',
                                          zh_hant: 'OCR 模型尚未下載，請到 設定 ▸ 模型 下載。',
                                          ja: 'OCR モデルが未ダウンロードです。設定 ▸ モデルから取得してください。',
                                          ko: 'OCR 모델이 아직 없습니다. 설정 ▸ 모델에서 받으세요.'),
  'error.ocrFailed'             => entry(en: 'Text recognition failed: %@',
                                          zh_hans: '文字识别失败：%@',
                                          zh_hant: '文字識別失敗：%@',
                                          ja: '文字認識に失敗しました: %@',
                                          ko: '텍스트 인식 실패: %@'),
  'settings.ocrModel.title'     => entry(en: 'OCR model (images → text)',
                                          zh_hans: 'OCR 模型（图片→文字）',
                                          zh_hant: 'OCR 模型（圖片→文字）',
                                          ja: 'OCR モデル（画像→テキスト）',
                                          ko: 'OCR 모델 (이미지→텍스트)'),
  'settings.ocrModel.tagline'   => entry(en: 'DeepSeek-OCR-2 · runs locally on GPU',
                                          zh_hans: 'DeepSeek-OCR-2 · 本地 GPU 运行',
                                          zh_hant: 'DeepSeek-OCR-2 · 本地 GPU 執行',
                                          ja: 'DeepSeek-OCR-2 · ローカル GPU で実行',
                                          ko: 'DeepSeek-OCR-2 · 로컬 GPU 실행'),
  'settings.ocrModel.size'      => entry(en: '~3 GB',
                                          zh_hans: '~3 GB',
                                          zh_hant: '~3 GB',
                                          ja: '~3 GB',
                                          ko: '~3 GB'),
  'cleanup.warning.mergeFallback' => entry(en: 'LLM merge unavailable — files were joined in order without deduplication.',
                                          zh_hans: 'LLM 拼接不可用——已按顺序直接拼接，未去重。',
                                          zh_hant: 'LLM 拼接不可用——已按順序直接拼接，未去重。',
                                          ja: 'LLM 結合が使えないため順番どおり連結しました（重複除去なし）。',
                                          ko: 'LLM 병합 불가 — 순서대로 이어붙였습니다(중복 제거 없음).'),
  'cleanup.warning.fileSkipped' => entry(en: 'Skipped %@: %@',
                                          zh_hans: '已跳过 %@：%@',
                                          zh_hant: '已跳過 %@：%@',
                                          ja: '%@ をスキップ: %@',
                                          ko: '%@ 건너뜀: %@'),
}

adds.each { |k, v| strings[k] = v }

File.write(PATH, JSON.pretty_generate(doc) + "\n")
puts "upserted #{adds.size} pivot keys"
