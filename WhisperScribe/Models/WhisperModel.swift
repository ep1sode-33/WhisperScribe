import Foundation

/// One entry in the curated model catalog. Pure value type — no I/O, no WhisperKit.
struct WhisperModel: Identifiable, Hashable {
    let id: String         // stable, persisted as the selection key
    let name: String       // verbatim display name, e.g. "large-v3-turbo"
    let variant: String    // WhisperKit/HF folder name; the download() argument
    let taglineKey: String // localized one-line description
    let sizeKey: String    // localized approximate size

    static let all: [WhisperModel] = [
        WhisperModel(id: "largeV3",
                     name: "large-v3",
                     variant: "openai_whisper-large-v3-v20240930",
                     taglineKey: "model.tagline.bestQuality",
                     sizeKey: "model.size.large"),
        WhisperModel(id: "largeV3Turbo",
                     name: "large-v3-turbo",
                     variant: "openai_whisper-large-v3-v20240930_turbo",
                     taglineKey: "model.tagline.fast",
                     sizeKey: "model.size.large"),
        WhisperModel(id: "distilV3",
                     name: "distil-large-v3",
                     variant: "distil-whisper_distil-large-v3",
                     taglineKey: "model.tagline.smallFast",
                     sizeKey: "model.size.distil"),
    ]

    static let `default`: WhisperModel = all[0]

    static func with(id: String) -> WhisperModel? { all.first { $0.id == id } }
}
