import Foundation
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    let settings = SettingsStore()
    let transcriber = TranscriberService()
    let cleaner = LLMCleaner()
    let modelManager = ModelManager(downloader: WhisperKitDownloader())
    lazy var viewModel = TranscriptionViewModel(
        settings: settings,
        transcriber: transcriber,
        cleaner: cleaner,
        modelManager: modelManager
    )
}
