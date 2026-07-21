import Foundation
import MLX
import MLXNN
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

    /// Loads every tensor from the real checkpoint's
    /// `model.safetensors.index.json` shard set (this checkpoint happens to
    /// have exactly one shard, but the helper handles the general
    /// multi-shard case). Shared by `WeightSanitizerTests` and `TestWeights`
    /// so shard-reading logic lives in exactly one place.
    static func loadAllShards() throws -> [String: MLXArray] {
        guard let dir = modelDir else { throw NSError(domain: "fixtures-missing", code: 1) }
        let indexData = try Data(contentsOf: dir.appending(path: "model.safetensors.index.json"))
        let index = try JSONSerialization.jsonObject(with: indexData) as! [String: Any]
        let weightMap = index["weight_map"] as! [String: String]
        let shardNames = Set(weightMap.values)

        var merged: [String: MLXArray] = [:]
        for shard in shardNames {
            let shardWeights = try MLX.loadArrays(url: dir.appending(path: shard))
            merged.merge(shardWeights) { _, new in new }
        }
        return merged
    }
}

/// Loads a slice of the real checkpoint's sanitized weights into a `Module`
/// subtree, for parity tests (Tasks 5-8) that instantiate one component
/// (e.g. `SAMEncoder`) in isolation rather than the whole model.
///
/// Real checkpoint keys for the components these tests target are top-level
/// (e.g. `sam_model.*`, a *sibling* of `vision_model.*` -- see
/// `WeightSanitizer`'s doc comment) and plain bf16, with quantized
/// `.scales`/`.biases` companions appearing only under `language_model.*`
/// and `projector.*` keys -- so a straight `update(parameters:verify:)` on
/// the prefix-filtered/stripped subtree (no quantization bookkeeping) is
/// sufficient here.
enum TestWeights {
    static func load(into module: Module, prefix: String) throws {
        let raw = try FixtureSupport.loadAllShards()
        guard let modelDir = FixtureSupport.modelDir else {
            throw NSError(domain: "fixtures-missing", code: 1)
        }
        let config = try DeepSeekOCR2Configuration(
            mergingJSON: Data(contentsOf: modelDir.appending(path: "config.json")))
        let sanitized = WeightSanitizer.sanitize(raw, config: config)

        var stripped: [String: MLXArray] = [:]
        for (key, value) in sanitized where key.hasPrefix(prefix) {
            stripped[String(key.dropFirst(prefix.count))] = value
        }

        let parameters = ModuleParameters.unflattened(stripped)
        try module.update(parameters: parameters, verify: .noUnusedKeys)
    }
}
