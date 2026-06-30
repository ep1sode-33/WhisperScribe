import Foundation
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    let settings = SettingsStore()
    let transcriber = TranscriberService()
    let cleaner = LLMCleaner()
    lazy var viewModel = TranscriptionViewModel(
        settings: settings,
        transcriber: transcriber,
        cleaner: cleaner
    )
}
