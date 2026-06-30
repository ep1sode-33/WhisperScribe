import SwiftUI

@main
struct WhisperScribeApp: App {
    @StateObject private var appModel = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appModel)
                .environmentObject(appModel.settings)
                .environmentObject(appModel.viewModel)
                .frame(minWidth: 560, minHeight: 420)
        }
        .windowResizability(.contentMinSize)

        Settings {
            SettingsView()
                .environmentObject(appModel.settings)
                .environmentObject(appModel)
        }
    }
}
