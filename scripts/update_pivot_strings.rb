#!/usr/bin/env ruby
# One-shot upsert of EXACTLY the 4 dual-modal copy keys below to their pivot values in
# all 5 languages (en/zh-Hans/zh-Hant/ja/ko). These keys already exist with audio-only
# personality wording; this rewrites their values only. Every other key is left untouched,
# so the catalog diff shows nothing but these four keys' value changes. Mirrors the entry
# helper of scripts/add_pivot_strings.rb (extractionState "manual"; every language state
# "translated").
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

updates = {
  'common.appSubtitle'        => entry(en: 'Local audio & images to text',
                                       zh_hans: '本地音频/图片转文字',
                                       zh_hant: '本地音訊/圖片轉文字',
                                       ja: '音声・画像をローカルでテキストに',
                                       ko: '오디오·이미지를 로컬에서 텍스트로'),
  'drop.title'                => entry(en: 'Drop audio files or images',
                                       zh_hans: '拖入音频或图片',
                                       zh_hant: '拖入音訊或圖片',
                                       ja: '音声または画像をドロップ',
                                       ko: '오디오 또는 이미지를 드롭'),
  'common.chooseFile'         => entry(en: 'Choose Files…',
                                       zh_hans: '选择文件…',
                                       zh_hant: '選擇檔案…',
                                       ja: 'ファイルを選択…',
                                       ko: '파일 선택…'),
  'common.chooseMediaMessage' => entry(en: 'Choose audio files or images to transcribe',
                                       zh_hans: '选择要转写的音频或图片',
                                       zh_hant: '選擇要轉寫的音訊或圖片',
                                       ja: '文字起こしする音声または画像を選択',
                                       ko: '변환할 오디오 또는 이미지를 선택'),
}

updates.each { |k, v| strings[k] = v }

File.write(PATH, JSON.pretty_generate(doc) + "\n")
puts "updated #{updates.size} pivot copy keys"
