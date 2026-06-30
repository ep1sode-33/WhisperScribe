#!/usr/bin/env ruby
# Adds the model-management strings (5 languages) to Localizable.xcstrings and removes
# the obsolete error.modelMissing and error.whisperKitNotInitialized keys (now orphaned —
# their only code references were removed in Task 5). Idempotent (overwrites these keys,
# leaves others untouched).
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
  'settings.model'              => entry(en: 'Model',              zh_hans: '模型',         zh_hant: '模型',         ja: 'モデル',            ko: '모델'),
  'model.tagline.bestQuality'   => entry(en: 'Best quality',       zh_hans: '质量最佳',     zh_hant: '品質最佳',     ja: '最高品質',          ko: '최고 품질'),
  'model.tagline.fast'          => entry(en: 'Fast',               zh_hans: '快速',         zh_hant: '快速',         ja: '高速',              ko: '빠름'),
  'model.tagline.smallFast'     => entry(en: 'Small & fast',       zh_hans: '小而快',       zh_hant: '小而快',       ja: '小型・高速',        ko: '작고 빠름'),
  'model.size.large'            => entry(en: '~1.5 GB',            zh_hans: '约 1.5 GB',    zh_hant: '約 1.5 GB',    ja: '約 1.5 GB',         ko: '약 1.5 GB'),
  'model.size.distil'           => entry(en: '~0.6 GB',            zh_hans: '约 0.6 GB',    zh_hant: '約 0.6 GB',    ja: '約 0.6 GB',         ko: '약 0.6 GB'),
  'model.download'              => entry(en: 'Download',           zh_hans: '下载',         zh_hant: '下載',         ja: 'ダウンロード',      ko: '다운로드'),
  'model.installed'             => entry(en: 'Installed',          zh_hans: '已安装',       zh_hant: '已安裝',       ja: 'インストール済み',  ko: '설치됨'),
  'model.retry'                 => entry(en: 'Retry',              zh_hans: '重试',         zh_hant: '重試',         ja: '再試行',            ko: '다시 시도'),
  'model.delete'                => entry(en: 'Delete',             zh_hans: '删除',         zh_hant: '刪除',         ja: '削除',              ko: '삭제'),
  'model.storageFootnote'       => entry(en: 'Models are stored locally and never re-downloaded.',
                                          zh_hans: '模型保存在本地，不会重复下载。',
                                          zh_hant: '模型儲存在本機，不會重複下載。',
                                          ja: 'モデルはローカルに保存され、再ダウンロードされません。',
                                          ko: '모델은 로컬에 저장되며 다시 다운로드되지 않습니다.'),
  'content.noModel.title'       => entry(en: 'No transcription model installed yet',
                                          zh_hans: '还没有安装转录模型',
                                          zh_hant: '尚未安裝轉錄模型',
                                          ja: '文字起こしモデルがまだインストールされていません',
                                          ko: '아직 설치된 받아쓰기 모델이 없습니다'),
  'content.noModel.openSettings'=> entry(en: 'Open Settings to download',
                                          zh_hans: '打开设置下载',
                                          zh_hant: '開啟設定下載',
                                          ja: '設定を開いてダウンロード',
                                          ko: '설정을 열어 다운로드'),
  'error.modelNotInstalled'     => entry(en: 'No transcription model installed. Open Settings (⌘,) to download one.',
                                          zh_hans: '尚未安装转录模型。请在设置（⌘,）中下载一个。',
                                          zh_hant: '尚未安裝轉錄模型。請在設定（⌘,）中下載一個。',
                                          ja: '文字起こしモデルがインストールされていません。設定（⌘,）からダウンロードしてください。',
                                          ko: '설치된 받아쓰기 모델이 없습니다. 설정(⌘,)에서 다운로드하세요.'),
  'error.modelDownloadFailed'   => entry(en: 'Model download failed: %@',
                                          zh_hans: '模型下载失败：%@',
                                          zh_hant: '模型下載失敗：%@',
                                          ja: 'モデルのダウンロードに失敗しました：%@',
                                          ko: '모델 다운로드 실패: %@'),
}

adds.each { |k, v| strings[k] = v }
strings.delete('error.modelMissing')
strings.delete('error.whisperKitNotInitialized')

File.write(PATH, JSON.pretty_generate(doc) + "\n")
puts "updated #{adds.size} keys; removed error.modelMissing and error.whisperKitNotInitialized"
