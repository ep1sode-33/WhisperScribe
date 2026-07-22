import Testing
import Foundation
@testable import WhisperScribe

struct BatchClassifierTests {
    private func url(_ name: String) -> URL { URL(fileURLWithPath: "/tmp/\(name)") }

    @Test func classifiesAudioBatchSorted() throws {
        let kind = try BatchClassifier.classify([url("b10.mp3"), url("b2.wav"), url("a.m4a")])
        guard case .audio(let sorted) = kind else { Issue.record("expected audio"); return }
        #expect(sorted.map(\.lastPathComponent) == ["a.m4a", "b2.wav", "b10.mp3"])  // 自然排序: 2 < 10
    }
    @Test func classifiesImageBatchSorted() throws {
        let kind = try BatchClassifier.classify([url("IMG_10.PNG"), url("IMG_2.jpg")])
        guard case .images(let sorted) = kind else { Issue.record("expected images"); return }
        #expect(sorted.map(\.lastPathComponent) == ["IMG_2.jpg", "IMG_10.PNG"])     // 扩展名大小写不敏感
    }
    @Test func rejectsMixedBatch() {
        #expect(throws: AppError.mixedBatch) { try BatchClassifier.classify([url("a.mp3"), url("b.png")]) }
    }
    @Test func rejectsUnsupportedFile() {
        #expect((try? BatchClassifier.classify([url("x.pdf")])) == nil)
    }
    @Test func rejectsEmpty() {
        #expect((try? BatchClassifier.classify([])) == nil)
    }
    @Test func acceptedURLCoversBothKinds() {
        #expect(BatchClassifier.isAcceptedURL(url("a.mp3")))
        #expect(BatchClassifier.isAcceptedURL(url("a.heic")))
        #expect(!BatchClassifier.isAcceptedURL(url("a.pdf")))
    }
}
