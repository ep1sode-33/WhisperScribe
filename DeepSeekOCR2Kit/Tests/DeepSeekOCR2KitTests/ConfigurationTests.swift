import Testing
import Foundation
@testable import DeepSeekOCR2Kit

@Suite struct ConfigurationTests {
    @Test func hardcodedDefaults() {
        let c = DeepSeekOCR2Configuration.default
        #expect(c.sam.layers == 12 && c.sam.width == 768 && c.sam.windowSize == 14)
        #expect(c.sam.globalAttnIndexes == [2, 5, 8, 11] && c.sam.outputChannels == 896)
        #expect(c.qwen2Encoder.layers == 24 && c.qwen2Encoder.dim == 896)
        #expect(c.qwen2Encoder.heads == 14 && c.qwen2Encoder.kvHeads == 2)
        #expect(c.qwen2Encoder.intermediate == 4864 && c.qwen2Encoder.ropeTheta == 1_000_000)
        #expect(c.text.hiddenSize == 1280 && c.text.layers == 30 && c.text.heads == 32)
        #expect(c.text.intermediate == 6848 && c.text.moeIntermediate == 896)
        #expect(c.text.numExperts == 64 && c.text.topK == 6 && c.text.sharedExperts == 2)
        #expect(c.text.firstKDenseReplace == 0 && c.text.vocabSize == 102_400)
        #expect(c.bosTokenID == 0 && c.eosTokenID == 1 && c.imageTokenID == 128_815)
    }
    @Test func decodesRealConfigJSON() throws {
        guard let model = FixtureSupport.modelDir else { return }  // 无模型时静默通过
        let data = try Data(contentsOf: model.appending(path: "config.json"))
        let c = try DeepSeekOCR2Configuration(mergingJSON: data)
        #expect(c.text.vocabSize == 129_280)
        #expect(c.modelType == "deepseekocr_2")
    }
}
