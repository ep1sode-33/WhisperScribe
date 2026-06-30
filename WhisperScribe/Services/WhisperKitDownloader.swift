import Foundation
import WhisperKit

/// Production `ModelDownloading`: downloads from `argmaxinc/whisperkit-coreml` into
/// WhisperKit's default base (`~/Documents/huggingface/...`). The only model-download
/// path that imports WhisperKit.
struct WhisperKitDownloader: ModelDownloading {
    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        try await WhisperKit.download(variant: variant) { p in
            progress(p.fractionCompleted)
        }
    }
}
