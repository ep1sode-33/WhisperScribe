import Foundation

/// The single source of truth for the UI. Drives StatusView.
enum JobState: Equatable {
    case idle
    case loadingModel
    case extractingAudio
    case transcribing(progress: Double)   // 0...1, determinate when WhisperKit reports it
    case cleaning(progress: Double, note: String)        // 0...1, by segments/batches processed
    case done(srt: URL, txt: URL, warnings: [String])
    case error(AppError)

    var isBusy: Bool {
        switch self {
        case .idle, .done, .error: return false
        default: return true
        }
    }
}
