import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import DeepSeekOCR2Kit

/// Anything that can be greedy-decoded token-by-token from a starting set of
/// input embeddings: the language model on its own (Task 7) and the full
/// `DeepSeekOCR2Model` (Task 8) both satisfy this, so `greedyDecode` has one
/// implementation and two call sites.
protocol GreedyDecodable {
    func newCache() -> [KVCache]
    func embed(_ tokens: MLXArray) -> MLXArray
    func callAsFunction(inputEmbeds: MLXArray, cache: [KVCache]?) -> MLXArray
}

extension MoELanguageModel: GreedyDecodable {}

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

    /// Greedy (argmax) decode of up to `steps` tokens, prefilling with `embeds`
    /// then feeding each sampled token's embedding back in through the shared
    /// KV cache. One implementation used by both `LanguageModelParityTests`
    /// (Task 7) and `EndToEndParityTests` (Task 8). Stops early on `eos`.
    /// `embeds` is cast to bf16 to match the checkpoint's compute precision.
    static func greedyDecode<M: GreedyDecodable>(
        model: M, embeds: MLXArray, steps: Int, eos: Int
    ) -> [Int] {
        let cache = model.newCache()
        var logits = model(inputEmbeds: embeds.asType(.bfloat16), cache: cache)
        var ids: [Int] = []
        for _ in 0..<steps {
            let next = MLX.argMax(logits[0..., -1, 0...], axis: -1)
            let nextID = next.item(Int.self)
            ids.append(nextID)
            if nextID == eos { break }
            logits = model(inputEmbeds: model.embed(next.reshaped(1, 1)), cache: cache)
        }
        return ids
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

    /// Quantization-aware variant for `language_model.*` (Task 7): unlike the
    /// `sam_model.*`/`vision_model.*` subtrees `load(into:prefix:)` targets,
    /// the language model's `Linear`/`Embedding`/`SwitchLinear` leaves ship
    /// with `.scales`/`.biases` companions -- 8-bit affine, group_size 64,
    /// per the real checkpoint's `config.json` `quantization` section (see
    /// `MoELanguageModel.swift`'s file header for the exact key inventory).
    /// A plain `update(parameters:)` would either throw (unconsumed `.scales`
    /// / `.biases` keys, since a stock `Linear` has no such parameters) or,
    /// worse, silently fail to raise if `verify` were relaxed.
    ///
    /// This mirrors mlx-swift-lm's own loader
    /// (`Libraries/MLXLMCommon/Load.swift: loadWeights`): sanitize the raw
    /// weights, then `MLXNN.quantize(model:groupSize:bits:mode:filter:)` to
    /// swap every leaf module that has a `.scales` companion in the loaded
    /// weights (and only those) for its quantized counterpart, and only then
    /// `update(parameters:)` with the full (now shape-matching) weight set.
    static func loadQuantized(
        into module: Module, prefix: String,
        groupSize: Int = 64, bits: Int = 8, mode: QuantizationMode = .affine
    ) throws {
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

        // Shared with `DeepSeekOCR2Model.load(from:)` -- one quantize+update
        // implementation. `.noUnusedKeys` (not `.all`) because this loads a
        // single subtree in isolation, so the module has no "missing" keys to
        // check against the full checkpoint.
        try QuantizedWeightLoader.load(
            into: module, weights: stripped,
            groupSize: groupSize, bits: bits, mode: mode, verify: .noUnusedKeys)
    }
}
