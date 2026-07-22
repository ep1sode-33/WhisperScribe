import Foundation

enum AppError: LocalizedError, Equatable {
    case noAudioTrack
    case modelNotInstalled
    case modelDownloadFailed(String)
    case audioDecodeFailed(String)
    case transcriptionFailed(String)
    case writeFailed(String)
    case cancelled
    case mixedBatch
    case unsupportedFile(String)
    case ocrModelMissing
    case ocrFailed(String)

    var errorDescription: String? {
        switch self {
        case .noAudioTrack:
            return String(localized: "error.noAudioTrack")
        case .modelNotInstalled:
            return String(localized: "error.modelNotInstalled")
        case .modelDownloadFailed(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.modelDownloadFailed", comment: ""), m)
        case .audioDecodeFailed(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.audioDecodeFailed", comment: ""), m)
        case .transcriptionFailed(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.transcriptionFailed", comment: ""), m)
        case .writeFailed(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.writeFailed", comment: ""), m)
        case .cancelled:
            return String(localized: "error.cancelled")
        case .mixedBatch:
            return String(localized: "error.mixedBatch")
        case .unsupportedFile(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.unsupportedImage", comment: ""), m)
        case .ocrModelMissing:
            return String(localized: "error.ocrModelMissing")
        case .ocrFailed(let m):
            return String.localizedStringWithFormat(NSLocalizedString("error.ocrFailed", comment: ""), m)
        }
    }
}
