import Foundation

/// The single source of truth for the UI. Drives StatusView.
enum JobState: Equatable {
    case idle
    case loadingModel
    case extractingAudio
    case transcribing(progress: Double)   // 0...1, determinate when WhisperKit reports it
    case recognizing(progress: Double)    // 0...1, OCR stage (images → text)
    case cleaning(progress: Double, note: String)        // 0...1, by segments/batches processed
    case merging(progress: Double, note: String)         // 0...1, multi-file LLM merge/dedup
    case done(outputs: [URL], warnings: [String])
    case error(AppError)

    var isBusy: Bool {
        switch self {
        case .idle, .done, .error:
            return false
        case .loadingModel, .extractingAudio, .transcribing, .recognizing, .cleaning, .merging:
            return true
        }
    }
}

/// Progress across a multi-file batch. `nil` on the ViewModel means a single-file /
/// non-batch job. `index` is 1-based; `count` is the total files in the batch.
struct BatchProgress: Equatable {
    let index: Int
    let count: Int
    let fileName: String
}
