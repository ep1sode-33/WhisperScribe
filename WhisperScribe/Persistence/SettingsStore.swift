import Foundation
import SwiftUI

@MainActor
final class SettingsStore: ObservableObject {
    @AppStorage("cleanupLevel") private var cleanupLevelRaw: Int = CleanupLevel.fixOnly.rawValue
    @AppStorage("language") private var languageRaw: String = ""
    @AppStorage("llmBaseURL") private var llmBaseURLRaw: String = ""
    @AppStorage("llmModel") private var llmModelRaw: String = ""
    @AppStorage("outputMode") private var outputModeRaw: Int = OutputMode.nextToSource.rawValue
    @AppStorage("customOutputFolderPath") private var customOutputFolderPathRaw: String = ""
    @AppStorage("overwritePolicy") private var overwritePolicyRaw: Int = OverwritePolicy.uniquify.rawValue

    // Computed over PRIVATE @AppStorage backing so assignments publish via
    // objectWillChange (raw @AppStorage on the ObservableObject does not), keeping
    // @EnvironmentObject views in sync — e.g. the folder label after chooseFolder().
    var language: String {
        get { languageRaw }
        set {
            objectWillChange.send()
            languageRaw = newValue
        }
    }

    var llmBaseURL: String {
        get { llmBaseURLRaw }
        set {
            objectWillChange.send()
            llmBaseURLRaw = newValue
        }
    }

    var llmModel: String {
        get { llmModelRaw }
        set {
            objectWillChange.send()
            llmModelRaw = newValue
        }
    }

    var customOutputFolderPath: String {
        get { customOutputFolderPathRaw }
        set {
            objectWillChange.send()
            customOutputFolderPathRaw = newValue
        }
    }

    var cleanupLevel: CleanupLevel {
        get { CleanupLevel(rawValue: cleanupLevelRaw) ?? .fixOnly }
        set {
            objectWillChange.send()
            cleanupLevelRaw = newValue.rawValue
        }
    }

    var outputMode: OutputMode {
        get { OutputMode(rawValue: outputModeRaw) ?? .nextToSource }
        set {
            objectWillChange.send()
            outputModeRaw = newValue.rawValue
        }
    }

    var overwritePolicy: OverwritePolicy {
        get { OverwritePolicy(rawValue: overwritePolicyRaw) ?? .uniquify }
        set {
            objectWillChange.send()
            overwritePolicyRaw = newValue.rawValue
        }
    }

    var apiKey: String {
        get { KeychainStore.get() ?? "" }
        set {
            objectWillChange.send()
            if newValue.isEmpty {
                KeychainStore.delete()
            } else {
                KeychainStore.set(newValue)
            }
        }
    }

    var llmConfig: LLMConfig {
        LLMConfig(baseURL: llmBaseURL, apiKey: apiKey, model: llmModel)
    }

    var resolvedOutputDir: URL? {
        outputMode == .customFolder && !customOutputFolderPath.isEmpty
            ? URL(fileURLWithPath: customOutputFolderPath)
            : nil
    }
}
