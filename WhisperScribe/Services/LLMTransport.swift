import Foundation

/// Low-level transport / HTTP errors used internally to drive retry decisions.
/// Extracted verbatim from `LLMCleaner` so both the cleanup pipeline and the
/// merge service share one retry-classification vocabulary. Kept `internal`.
enum LLMRequestError: LocalizedError {
    case badURL
    case httpStatus(Int, retryAfter: TimeInterval?)
    case emptyContent

    var errorDescription: String? {
        switch self {
        case .badURL:                  return String(localized: "error.badBaseURL")
        case .httpStatus(let code, _): return String.localizedStringWithFormat(NSLocalizedString("error.httpStatus", comment: ""), code)
        case .emptyContent:            return String(localized: "error.emptyResponse")
        }
    }
}

/// Shared, actor-agnostic LLM transport for BYOK OpenAI-compatible endpoints.
///
/// Extracted from `LLMCleaner` WITHOUT behavior change: the endpoint resolution,
/// request body shape, SSE decoding, retry backoff and error classification are
/// byte-for-byte the same requests the two-pass cleanup has always sent.
///
/// The one deliberate refactor is how live progress is surfaced: `LLMCleaner`
/// used to mutate its actor-isolated `generatedChars` counter directly inside the
/// SSE loop. Transport is now isolation-free, so it exposes the stream two ways:
/// - `chatDeltas(...)` yields one `StreamDelta` per SSE chunk (content +
///   reasoning kept separate). `LLMCleaner` consumes this on its own actor and
///   updates the live counter there — parity with the old inline mutation.
/// - `streamChat(...onDelta:)` is the closure form other callers (MergeService via
///   `ChatStreaming`) use: it fires `onDelta` for each answer delta and returns the
///   concatenated answer text.
enum LLMTransport {

    struct ChatMessage: Codable { let role: String; let content: String }

    private struct ChatRequestBody: Encodable {
        let model: String
        let messages: [ChatMessage]
        let temperature: Int
        let top_p: Int
        let max_tokens: Int?   // omitted from JSON when nil (so reasoning models aren't starved)
        let stream: Bool
    }

    /// One streamed chat-completions chunk (OpenAI-compatible SSE).
    private struct StreamChunk: Decodable {
        struct Choice: Decodable {
            struct Delta: Decodable { let content: String?; let reasoning_content: String? }
            let delta: Delta?
            let finish_reason: String?
        }
        let choices: [Choice]?
    }

    /// One decoded SSE chunk. `content` is answer text (accumulated into the result);
    /// `reasoning` is chain-of-thought (counted toward live progress only, never
    /// returned). Kept as a single per-chunk value so a consumer can reproduce the
    /// original "count content + reasoning, then throttle once per chunk" cadence.
    struct StreamDelta: Sendable {
        let content: String?
        let reasoning: String?
    }

    private static let requestTimeout: TimeInterval = 120

    // MARK: - Endpoint

    static func endpointURL(_ base: String) -> URL? {
        var s = base.trimmingCharacters(in: .whitespacesAndNewlines)
        while s.hasSuffix("/") { s.removeLast() }
        guard !s.isEmpty else { return nil }
        if s.hasSuffix("/chat/completions") { return URL(string: s) }
        // Cloudflare AI Gateway unified endpoint: ".../compat" -> ".../compat/chat/completions"
        if s.hasSuffix("/compat") { return URL(string: s + "/chat/completions") }
        if s.hasSuffix("/v1") { return URL(string: s + "/chat/completions") }
        return URL(string: s + "/v1/chat/completions")
    }

    // MARK: - Streaming core

    /// One streamed chat-completions call, surfaced as a delta stream.
    /// Yields one `StreamDelta` per SSE chunk that carries content and/or reasoning,
    /// in arrival order. Finishes (throwing) with `LLMRequestError` or the underlying
    /// transport error, exactly as the old inline loop did.
    ///
    /// An empty `system` string means "no system message" (the old `system == nil`
    /// path); it is never sent as an empty system message.
    static func chatDeltas(config: LLMConfig,
                           system: String,
                           user: String,
                           maxTokens: Int?) -> AsyncThrowingStream<StreamDelta, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let url = endpointURL(config.baseURL) else { throw LLMRequestError.badURL }

                    var messages: [ChatMessage] = []
                    if !system.isEmpty { messages.append(ChatMessage(role: "system", content: system)) }
                    messages.append(ChatMessage(role: "user", content: user))

                    let body = ChatRequestBody(model: config.model,
                                               messages: messages,
                                               temperature: 0,
                                               top_p: 1,
                                               max_tokens: maxTokens,
                                               stream: true)

                    var req = URLRequest(url: url)
                    req.httpMethod = "POST"
                    req.timeoutInterval = requestTimeout
                    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    // Some gateways (e.g. Cloudflare AI Gateway) reject non-browser User-Agents.
                    req.setValue("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                                 forHTTPHeaderField: "User-Agent")
                    let key = config.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !key.isEmpty { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
                    req.httpBody = try JSONEncoder().encode(body)

                    let (bytes, response) = try await URLSession.shared.bytes(for: req)

                    guard let http = response as? HTTPURLResponse else { throw LLMRequestError.emptyContent }
                    if !(200...299).contains(http.statusCode) {
                        let retryAfter = http.value(forHTTPHeaderField: "Retry-After").flatMap { TimeInterval($0) }
                        throw LLMRequestError.httpStatus(http.statusCode, retryAfter: retryAfter)
                    }

                    let decoder = JSONDecoder()
                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        guard line.hasPrefix("data:") else { continue }
                        let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
                        if payload == "[DONE]" { break }
                        guard let d = payload.data(using: .utf8),
                              let chunk = try? decoder.decode(StreamChunk.self, from: d) else {
                            continue   // ignore a single malformed line; don't abort the stream
                        }
                        guard let delta = chunk.choices?.first?.delta else { continue }
                        if delta.content != nil || delta.reasoning_content != nil {
                            continuation.yield(StreamDelta(content: delta.content,
                                                           reasoning: delta.reasoning_content))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Closure form of `chatDeltas`. Fires `onDelta` for each answer delta and
    /// returns the concatenated answer text (`reasoning_content` is ignored here).
    /// Throws `LLMRequestError` or the underlying transport error, and
    /// `LLMRequestError.emptyContent` when no answer text arrived.
    static func streamChat(config: LLMConfig,
                           system: String,
                           user: String,
                           maxTokens: Int?,
                           onDelta: @escaping @Sendable (String) -> Void) async throws -> String {
        var finalContent = ""
        for try await delta in chatDeltas(config: config, system: system, user: user, maxTokens: maxTokens) {
            try Task.checkCancellation()
            if let c = delta.content {
                finalContent += c
                onDelta(c)
            }
        }
        try Task.checkCancellation()
        if finalContent.isEmpty { throw LLMRequestError.emptyContent }
        return finalContent
    }

    // MARK: - Backoff

    /// Exponential backoff with per-attempt jitter; honors Retry-After when present.
    static func sleepBackoff(attempt: Int, retryAfter: Double?) async throws {
        let base: [TimeInterval] = [0.5, 2.0, 8.0]
        let b = attempt < base.count ? base[attempt] : 8.0
        let jitter = Double(attempt) * 0.13 + 0.07
        let wait = (retryAfter ?? b) + jitter
        let safe = wait.isFinite ? min(max(wait, 0), 30) : 0   // clamp: no negative/NaN/huge Retry-After
        try await Task.sleep(nanoseconds: UInt64(safe * 1_000_000_000))
    }
}

/// Abstraction over the streaming transport so services can be unit-tested with a
/// fake. The default implementation forwards to `LLMTransport` with no token cap.
protocol ChatStreaming: Sendable {
    func streamChat(config: LLMConfig, system: String, user: String,
                    onDelta: @escaping @Sendable (String) -> Void) async throws -> String
}

struct LLMTransportChat: ChatStreaming {
    func streamChat(config: LLMConfig, system: String, user: String,
                    onDelta: @escaping @Sendable (String) -> Void) async throws -> String {
        try await LLMTransport.streamChat(config: config, system: system, user: user,
                                          maxTokens: nil, onDelta: onDelta)
    }
}
