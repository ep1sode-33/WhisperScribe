import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var viewModel: TranscriptionViewModel
    @EnvironmentObject var settings: SettingsStore

    private var showDropZone: Bool {
        if case .idle = viewModel.state { return true }
        return false
    }

    var body: some View {
        VStack(spacing: 0) {
            header

            Divider()

            Group {
                if showDropZone {
                    DropZone { url in
                        viewModel.start(url: url)
                    }
                    .padding(20)
                } else {
                    StatusView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    chooseFile()
                } label: {
                    Label("common.chooseFile", systemImage: "doc.badge.plus")
                }
                .disabled(viewModel.state.isBusy)
            }
            ToolbarItem(placement: .automatic) {
                SettingsLink {
                    Label("settings.title", systemImage: "gearshape")
                }
            }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "waveform")
                .font(.title2)
                .foregroundStyle(Color.accentColor)
            VStack(alignment: .leading, spacing: 2) {
                Text("WhisperScribe")
                    .font(.headline)
                Text("common.appSubtitle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private func chooseFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.audio, .audiovisualContent]
        panel.prompt = String(localized: "common.choose")
        panel.message = String(localized: "common.chooseMediaMessage")
        if panel.runModal() == .OK, let url = panel.url {
            viewModel.start(url: url)
        }
    }
}
