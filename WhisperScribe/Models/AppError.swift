import Foundation

enum AppError: LocalizedError, Equatable {
    case noAudioTrack
    case modelMissing(path: String)
    case audioDecodeFailed(String)
    case transcriptionFailed(String)
    case writeFailed(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .noAudioTrack:
            return String(localized: "error.noAudioTrack")
        case .modelMissing(let path):
            return String.localizedStringWithFormat(NSLocalizedString("error.modelMissing", comment: ""), path)
        case .audioDecodeFailed(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.audioDecodeFailed", comment: ""), m)
        case .transcriptionFailed(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.transcriptionFailed", comment: ""), m)
        case .writeFailed(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.writeFailed", comment: ""), m)
        case .cancelled:
            return String(localized: "error.cancelled")
        }
    }
}
