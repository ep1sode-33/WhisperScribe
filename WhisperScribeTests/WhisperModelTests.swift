import Testing
@testable import WhisperScribe

struct WhisperModelTests {
    @Test func catalogHasThreeUniqueModels() {
        #expect(WhisperModel.all.count == 3)
        #expect(Set(WhisperModel.all.map(\.id)).count == 3)
        #expect(Set(WhisperModel.all.map(\.variant)).count == 3)
    }

    @Test func defaultIsLargeV3WithInstalledVariant() {
        #expect(WhisperModel.default.id == "largeV3")
        #expect(WhisperModel.default.variant == "openai_whisper-large-v3-v20240930")
    }

    @Test func lookupRoundTrips() {
        for m in WhisperModel.all {
            #expect(WhisperModel.with(id: m.id) == m)
        }
        #expect(WhisperModel.with(id: "nope") == nil)
    }

    @Test func variantsArePinned() {
        #expect(WhisperModel.with(id: "largeV3")?.variant == "openai_whisper-large-v3-v20240930")
        #expect(WhisperModel.with(id: "largeV3Turbo")?.variant == "openai_whisper-large-v3-v20240930_turbo")
        #expect(WhisperModel.with(id: "distilV3")?.variant == "distil-whisper_distil-large-v3")
    }
}
