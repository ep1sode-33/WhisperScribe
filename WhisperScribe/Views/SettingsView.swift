import SwiftUI
import AppKit

struct SettingsView: View {
    @EnvironmentObject var settings: SettingsStore
    @EnvironmentObject var appModel: AppModel
    @EnvironmentObject var modelManager: ModelManager

    @State private var testing = false
    @State private var testResult: TestResult?

    private enum TestResult: Equatable {
        case success(String)
        case failure(String)
    }

    var body: some View {
        Form {
            modelSection
            transcriptionSection
            cleanupSection
            llmSection
            outputSection
        }
        .formStyle(.grouped)
        .frame(width: 480)
        .frame(minHeight: 540)
    }

    // MARK: - 模型

    private var modelSection: some View {
        Section("settings.model") {
            ForEach(WhisperModel.all) { model in
                modelRow(model)
            }
            Text("model.storageFootnote")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private func modelRow(_ model: WhisperModel) -> some View {
        let installed = modelManager.isInstalled(model)
        let selected = modelManager.selectedModel.id == model.id
        HStack(spacing: 10) {
            Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                .foregroundStyle(selected ? Color.accentColor : Color.secondary)
                .onTapGesture { if installed { modelManager.selectedModelID = model.id } }
                .opacity(installed ? 1 : 0.4)

            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: model.name).font(.body)
                HStack(spacing: 6) {
                    Text(LocalizedStringKey(model.taglineKey))
                    Text(verbatim: "·")
                    Text(LocalizedStringKey(model.sizeKey))
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            modelAccessory(model, installed: installed)
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func modelAccessory(_ model: WhisperModel, installed: Bool) -> some View {
        switch modelManager.state(for: model) {
        case .downloading(let f):
            HStack(spacing: 8) {
                ProgressView(value: f).progressViewStyle(.linear).frame(width: 90)
                Text("\(Int(f * 100))%").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Button("common.cancel") { modelManager.cancelDownload(model) }
            }
        case .failed(let msg):
            HStack(spacing: 8) {
                Label(msg, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange).lineLimit(1)
                Button("model.retry") { modelManager.download(model) }
            }
        case .idle:
            if installed {
                HStack(spacing: 10) {
                    Label("model.installed", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.green)
                    Button("model.delete", role: .destructive) { modelManager.delete(model) }
                        .font(.caption)
                }
            } else {
                Button("model.download") { modelManager.download(model) }
            }
        }
    }

    // MARK: - 转录

    private var transcriptionSection: some View {
        Section("settings.transcription") {
            Picker("settings.language", selection: $settings.language) {
                Text("settings.language.auto").tag("")
                Text(verbatim: "中文").tag("zh")
                Text(verbatim: "English").tag("en")
                Text(verbatim: "日本語").tag("ja")
            }
        }
    }

    // MARK: - 清理级别

    private var cleanupSection: some View {
        Section("settings.cleanupLevel") {
            Picker("settings.level", selection: $settings.cleanupLevel) {
                ForEach(CleanupLevel.allCases) { level in
                    Text(level.title).tag(level)
                }
            }
            .pickerStyle(.segmented)

            Text(settings.cleanupLevel.detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - LLM

    private var llmSection: some View {
        Section("settings.llmSection") {
            TextField("settings.baseURL", text: $settings.llmBaseURL)
                .textContentType(.URL)
            TextField("settings.modelName", text: $settings.llmModel)
            SecureField("settings.apiKey", text: $settings.apiKey)

            HStack {
                Button {
                    runTest()
                } label: {
                    if testing {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("settings.testConnection")
                    }
                }
                .disabled(testing)

                Spacer()

                switch testResult {
                case .success(let msg):
                    Label(msg, systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                        .lineLimit(1)
                case .failure(let msg):
                    Label(msg, systemImage: "xmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(1)
                case .none:
                    EmptyView()
                }
            }

            if settings.cleanupLevel.usesLLM && !settings.llmConfig.isConfigured {
                Label("settings.byokWarning", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    // MARK: - 输出

    private var outputSection: some View {
        Section("settings.output") {
            Picker("settings.outputLocation", selection: $settings.outputMode) {
                ForEach(OutputMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }

            if settings.outputMode == .customFolder {
                HStack {
                    Text(settings.customOutputFolderPath.isEmpty ? String(localized: "settings.noFolderSelected") : settings.customOutputFolderPath)
                        .font(.caption)
                        .foregroundStyle(settings.customOutputFolderPath.isEmpty ? .secondary : .primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button("settings.chooseFolder") {
                        chooseFolder()
                    }
                }
            }

            Picker("settings.sameNameFile", selection: $settings.overwritePolicy) {
                ForEach(OverwritePolicy.allCases) { policy in
                    Text(policy.title).tag(policy)
                }
            }
        }
    }

    // MARK: - Actions

    private func runTest() {
        testing = true
        testResult = nil
        let config = settings.llmConfig
        Task {
            let result = await appModel.cleaner.testConnection(config: config)
            await MainActor.run {
                testing = false
                switch result {
                case .success(let msg):
                    testResult = .success(msg)
                case .failure(let error):
                    testResult = .failure(error.localizedDescription)
                }
            }
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = String(localized: "common.choose")
        panel.message = String(localized: "settings.chooseFolderMessage")
        if panel.runModal() == .OK, let url = panel.url {
            settings.customOutputFolderPath = url.path
        }
    }
}
