import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var viewModel: TranscriptionViewModel
    @EnvironmentObject var settings: SettingsStore
    @EnvironmentObject var modelManager: ModelManager

    private var isIdle: Bool {
        if case .idle = viewModel.state { return true }
        return false
    }

    var body: some View {
        VStack(spacing: 0) {
            header

            Divider()

            Group {
                if isIdle {
                    if modelManager.isReady {
                        DropZone { urls in
                            viewModel.start(urls: urls)
                        }
                        .padding(20)
                    } else {
                        noModelView
                    }
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
                .disabled(viewModel.state.isBusy || !modelManager.isReady)
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

    private var noModelView: some View {
        VStack(spacing: 14) {
            Image(systemName: "arrow.down.circle.dotted")
                .font(.system(size: 44))
                .foregroundStyle(.secondary)
            Text("content.noModel.title")
                .font(.headline)
            SettingsLink {
                Label("content.noModel.openSettings", systemImage: "gearshape")
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: 420, maxHeight: .infinity)
        .padding(24)
    }

    private func chooseFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.audio, .audiovisualContent, .image]
        panel.prompt = String(localized: "common.choose")
        panel.message = String(localized: "common.chooseMediaMessage")
        if panel.runModal() == .OK, !panel.urls.isEmpty {
            viewModel.start(urls: panel.urls)
        }
    }
}
