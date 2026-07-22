import Foundation
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    let settings = SettingsStore()
    let transcriber = TranscriberService()
    let cleaner = LLMCleaner()
    let modelManager = ModelManager(downloader: WhisperKitDownloader())
    let ocrModels = OCRModelManager()
    let ocr = OCRService()
    let merger = MergeService()
    lazy var viewModel = TranscriptionViewModel(
        settings: settings,
        transcriber: transcriber,
        cleaner: cleaner,
        modelManager: modelManager,
        ocr: ocr,
        ocrModels: ocrModels,
        merger: merger
    )
}
