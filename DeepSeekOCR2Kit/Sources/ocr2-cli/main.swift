import Foundation
import ArgumentParser
import DeepSeekOCR2Kit

@main
struct OCR2CLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ocr2-cli",
        abstract: "DeepSeek-OCR-2 command-line interface"
    )

    func run() throws {
        print("DeepSeekOCR2Kit CLI - Model: \(DeepSeekOCR2Kit.modelRepoID)")
    }
}
