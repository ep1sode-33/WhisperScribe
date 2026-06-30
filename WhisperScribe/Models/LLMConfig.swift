import Foundation

/// BYOK OpenAI-compatible endpoint config. User-supplied; all fields blank by default.
struct LLMConfig: Equatable {
    var baseURL: String   // may or may not include /v1
    var apiKey: String    // may legitimately be empty for some local gateways
    var model: String

    init(baseURL: String = "", apiKey: String = "", model: String = "") {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.model = model
    }

    /// Enough to attempt a request. apiKey is intentionally NOT required
    /// (some local OpenAI-compatible gateways accept no key).
    var isConfigured: Bool {
        !baseURL.trimmingCharacters(in: .whitespaces).isEmpty &&
        !model.trimmingCharacters(in: .whitespaces).isEmpty
    }
}
