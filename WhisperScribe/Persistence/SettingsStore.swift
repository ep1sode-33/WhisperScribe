import Foundation
import SwiftUI

@MainActor
final class SettingsStore: ObservableObject {
    @AppStorage("cleanupLevel") private var cleanupLevelRaw: Int = CleanupLevel.fixOnly.rawValue
    @AppStorage("language") var language: String = ""
    @AppStorage("llmBaseURL") var llmBaseURL: String = ""
    @AppStorage("llmModel") var llmModel: String = ""
    @AppStorage("outputMode") private var outputModeRaw: Int = OutputMode.nextToSource.rawValue
    @AppStorage("customOutputFolderPath") var customOutputFolderPath: String = ""
    @AppStorage("overwritePolicy") private var overwritePolicyRaw: Int = OverwritePolicy.uniquify.rawValue

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
