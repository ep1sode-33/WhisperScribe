import Foundation
import MLX
@testable import DeepSeekOCR2Kit

enum FixtureSupport {
    static var root: URL? {
        let url = URL(fileURLWithPath: #filePath)  // …/Tests/DeepSeekOCR2KitTests/FixtureSupport.swift
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "Fixtures")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
    static var modelDir: URL? {
        guard let root, let s = try? String(contentsOf: root.appending(path: "model_dir.txt"), encoding: .utf8)
        else { return nil }
        return URL(fileURLWithPath: s.trimmingCharacters(in: .whitespacesAndNewlines))
    }
    static func load(_ dir: String, _ name: String) throws -> [String: MLXArray] {
        guard let root else { throw NSError(domain: "fixtures-missing", code: 1) }
        return try MLX.loadArrays(url: root.appending(path: dir).appending(path: "\(name).safetensors"))
    }
    static func meta(_ dir: String) throws -> [String: Any] {
        guard let root else { throw NSError(domain: "fixtures-missing", code: 1) }
        let data = try Data(contentsOf: root.appending(path: dir).appending(path: "meta.json"))
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }
}
