import Foundation

/// The model-download seam. Production code wraps `WhisperKit.download`; tests inject a fake.
/// Kept WhisperKit-free so `ModelManager` (which depends on this) stays dependency-light.
protocol ModelDownloading: Sendable {
    /// Download `variant` from the WhisperKit repo into the default base, reporting fractional
    /// progress (0...1) — which may be called off the main actor. Returns the model folder URL.
    func download(variant: String, progress: @escaping @Sendable (Double) -> Void) async throws -> URL
}
