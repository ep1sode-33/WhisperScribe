import Testing
import Foundation
@testable import WhisperScribe

private struct FakeChat: ChatStreaming, @unchecked Sendable {
    let result: Result<String, Error>
    var received: (@Sendable (String, String) -> Void)? = nil   // (system, user)
    func streamChat(config: LLMConfig, system: String, user: String,
                    onDelta: @escaping @Sendable (String) -> Void) async throws -> String {
        received?(system, user)
        switch result {
        case .success(let s): onDelta(s); return s
        case .failure(let e): throw e
        }
    }
}
private let cfg = LLMConfig(baseURL: "https://api.example.com/v1", apiKey: "k", model: "m")

struct MergeServiceTests {
    @Test func mergesViaLLM() async {
        let svc = MergeService(chat: FakeChat(result: .success("合并后的全文")))
        let out = await svc.merge(parts: [("a.png", "第一段"), ("b.png", "第二段")],
                                  language: "zh", config: cfg, progress: { _, _ in })
        #expect(out.text == "合并后的全文")
        #expect(out.warnings.isEmpty)
    }
    @Test func fallsBackWhenLLMFails() async {
        struct E: Error {}
        let svc = MergeService(chat: FakeChat(result: .failure(E())))
        let out = await svc.merge(parts: [("a.png", "甲"), ("b.png", "乙")],
                                  language: nil, config: cfg, progress: { _, _ in })
        #expect(out.text.contains("甲") && out.text.contains("乙"))
        #expect(out.text.contains("----- b.png -----"))
        #expect(out.warnings.count == 1)
    }
    @Test func fallsBackWhenUnconfigured() async {
        let svc = MergeService(chat: FakeChat(result: .success("不应被调用")))
        let out = await svc.merge(parts: [("a.png", "甲")], language: nil,
                                  config: LLMConfig(baseURL: "", apiKey: "", model: ""),
                                  progress: { _, _ in })
        #expect(out.text == "甲")           // 单文件直接原文（无需分隔头）
        #expect(out.warnings.count == 1)
    }
    @Test func promptCarriesPartsInOrder() async {
        nonisolated(unsafe) var captured = ""
        var fake = FakeChat(result: .success("ok"))
        fake.received = { _, user in captured = user }
        _ = await MergeService(chat: fake).merge(parts: [("1.png", "AAA"), ("2.png", "BBB")],
                                                 language: nil, config: cfg, progress: { _, _ in })
        #expect(captured.range(of: "AAA")!.lowerBound < captured.range(of: "BBB")!.lowerBound)
    }
}
