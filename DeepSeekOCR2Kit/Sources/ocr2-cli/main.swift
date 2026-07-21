import Foundation
import ArgumentParser
import DeepSeekOCR2Kit

struct OCR2CLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ocr2-cli",
        abstract: "DeepSeek-OCR-2 command-line interface"
    )

    func run() throws {
        print("DeepSeekOCR2Kit CLI - Model: \(DeepSeekOCR2Kit.modelRepoID)")
    }
}

// NOTE: this file is named `main.swift`, which implicitly permits top-level
// code and is incompatible with an `@main`-attributed type in the same file
// (`swift build` tolerates it, but Xcode's build system — required to run
// Metal-backed MLX tests via `xcodebuild test`, since SwiftPM's command-line
// build cannot compile the Metal shader library — rejects it).
OCR2CLI.main()
