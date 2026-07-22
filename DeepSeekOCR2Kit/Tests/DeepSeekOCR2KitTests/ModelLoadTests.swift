import Testing
import Foundation
import MLX
@testable import DeepSeekOCR2Kit

/// Codex review load-path findings (A2 config validation, A5 quantization mode).
/// Pure-logic tests over the factored `DeepSeekOCR2Model` load helpers -- no
/// checkpoint required, so they run everywhere (including CI without fixtures).
@Suite struct ModelLoadTests {
    /// A2: a config that selects an unimplemented code path (MLA attention via
    /// `qkNopeHeadDim != 0`, or grouped MoE routing via `nGroup`/`topkGroup`)
    /// is rejected with `OCR2LoadError.unsupportedConfiguration` BEFORE the
    /// model is constructed -- so the public `load` path never reaches
    /// `MoELanguageModel.init`'s fatalError.
    @Test func unsupportedConfigurationRejected() {
        // The real-checkpoint-shaped default validates fine.
        #expect(throws: Never.self) {
            try DeepSeekOCR2Model.validateSupported(.default)
        }

        var mla = DeepSeekOCR2Configuration.default
        mla.text.qkNopeHeadDim = 64  // MLA attention path -> unsupported
        #expect(throws: OCR2LoadError.self) {
            try DeepSeekOCR2Model.validateSupported(mla)
        }

        var grouped = DeepSeekOCR2Configuration.default
        grouped.text.nGroup = 8
        grouped.text.topkGroup = 4  // grouped ("noaux_tc") routing -> unsupported
        #expect(throws: OCR2LoadError.self) {
            try DeepSeekOCR2Model.validateSupported(grouped)
        }
    }

    /// A5: a known quantization mode maps through; an unknown one throws
    /// `OCR2LoadError.unsupportedQuantization` (rather than silently defaulting
    /// to `.affine`, which would mis-dequantize a genuinely different format).
    @Test func quantizationModeMappingAndRejection() throws {
        #expect(try DeepSeekOCR2Model.quantizationMode("affine") == .affine)
        #expect(throws: OCR2LoadError.self) {
            _ = try DeepSeekOCR2Model.quantizationMode("bogus")
        }
    }
}
