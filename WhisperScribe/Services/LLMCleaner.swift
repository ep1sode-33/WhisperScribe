import Foundation

/// Result of the two-pass cleanup.
/// `segments` carry the (possibly cleaned) per-segment text on the ORIGINAL
/// indices/timings — safe to hand to SRTFormatter. `txt` is the prose transcript.
struct CleanupOutcome {
    let segments: [TimedSegment]
    let txt: String
    let warnings: [String]
}

/// Errors surfaced ONLY by `testConnection`. The `process` pipeline never throws
/// for LLM/network problems — it degrades to raw output instead.
enum LLMCleanerError: LocalizedError {
    case notConfigured

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return String(localized: "error.llmNotConfigured")
        }
    }
}

/// Low-level transport / HTTP errors used internally to drive retry decisions.
private enum LLMRequestError: LocalizedError {
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

/// BYOK OpenAI-compatible cleanup. Timestamp-preserving two-pass design.
actor LLMCleaner {

    init() {}

    // Tunables
    private let maxSegmentsPerBatch = 40
    private let maxCharsPerBatch = 6000
    private let maxConcurrentBatches = 6
    private let maxPassBChunkChars = 6000
    private let requestTimeout: TimeInterval = 120

    // MARK: - Live progress state (actor-isolated → data-race-free across awaits)

    /// The closure passed into `process()`; nil outside a `process()` call.
    private var progressSink: (@Sendable (Double, String) -> Void)?
    /// GLOBAL char count across all concurrent batches; only grows during a process() call.
    private var generatedChars = 0
    /// Throttle marker for `emit()`.
    private var lastEmittedChars = 0
    /// Current overall completion fraction (0...1).
    private var curFraction: Double = 0
    /// Current phase note, e.g. "已修复字幕 1/2 批" or "整理正文 1/3 块".
    private var curPhaseNote: String = ""

    private func emit() {
        let n = generatedChars
        let note: String
        if curPhaseNote.isEmpty {
            note = String.localizedStringWithFormat(NSLocalizedString("cleanup.status.generatedChars", comment: ""), n)
        } else {
            let suffix = String.localizedStringWithFormat(NSLocalizedString("cleanup.status.generatedCharsSuffix", comment: ""), n)
            note = curPhaseNote + suffix
        }
        progressSink?(curFraction, note)
    }

    // MARK: - Public API

    func process(segments: [TimedSegment],
                 level: CleanupLevel,
                 language: String?,
                 config: LLMConfig,
                 progress: @escaping @Sendable (Double, String) -> Void) async throws -> CleanupOutcome {

        // L0: no network at all.
        if level == .raw {
            return CleanupOutcome(segments: segments,
                                  txt: TextJoiner.join(segments.map { $0.text }),
                                  warnings: [])
        }

        // Misconfigured BYOK -> degrade to raw-equivalent (≈L0).
        if !config.isConfigured {
            return CleanupOutcome(segments: segments,
                                  txt: TextJoiner.join(segments.map { $0.text }),
                                  warnings: [String(localized: "cleanup.warning.notConfiguredDegrade")])
        }

        // Nothing to do.
        if segments.isEmpty {
            return CleanupOutcome(segments: segments, txt: "", warnings: [])
        }

        // Wire the live counter for this run. Cleared on exit so stale closures
        // never fire after process() returns.
        self.progressSink = progress
        generatedChars = 0
        lastEmittedChars = 0
        curFraction = 0
        curPhaseNote = ""
        defer { self.progressSink = nil }

        progress(0.0, "")

        let passA = try await runPassA(segments: segments,
                                       level: level,
                                       language: language,
                                       config: config,
                                       progress: progress)
        try Task.checkCancellation()

        let passB = try await runPassB(cleanedSegments: passA.segments,
                                       level: level,
                                       language: language,
                                       config: config,
                                       progress: progress)

        var warnings: [String] = []
        if passA.fallbackBatches > 0 {
            let cleaned = passA.totalBatches - passA.fallbackBatches
            warnings.append(String.localizedStringWithFormat(NSLocalizedString("cleanup.warning.partialFallback", comment: ""), cleaned, passA.totalBatches, passA.fallbackBatches))
            if passA.fallbackBatches == passA.totalBatches {
                warnings.append(String(localized: "cleanup.warning.allFallback"))
            }
        }
        if passB.fellBack {
            warnings.append(String(localized: "cleanup.warning.txtFallback"))
        }

        progress(1.0, "")
        return CleanupOutcome(segments: passA.segments, txt: passB.txt, warnings: warnings)
    }

    func testConnection(config: LLMConfig) async -> Result<String, Error> {
        if !config.isConfigured {
            return .failure(LLMCleanerError.notConfigured)
        }
        do {
            let content = try await performChat(config: config,
                                                system: nil,
                                                user: "ping, reply with OK",
                                                maxTokens: nil,
                                                reportLiveProgress: false)
            let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
            return .success(trimmed.isEmpty ? config.model : trimmed)
        } catch {
            return .failure(error)
        }
    }

    // MARK: - Pass A (SRT, indexed 1:1)

    private func runPassA(segments: [TimedSegment],
                          level: CleanupLevel,
                          language: String?,
                          config: LLMConfig,
                          progress: @escaping @Sendable (Double, String) -> Void)
        async throws -> (segments: [TimedSegment], totalBatches: Int, fallbackBatches: Int) {

        // Cap SRT semantics at L2 even when the user picked L3.
        let effectiveLevel: CleanupLevel = (level == .lightEdit) ? .cleanPolish : level
        let system = (effectiveLevel == .fixOnly)
            ? LLMPrompts.srtL1System(language: language)
            : LLMPrompts.srtL2System(language: language)

        let batches = makeBatches(segments)
        let total = batches.count
        guard total > 0 else { return (segments, 0, 0) }

        // Report the total up front so the UI shows "已修复字幕 0/N 批" immediately
        // (instead of a blank 0%) while the first — slow — batch is still in flight.
        curFraction = 0
        curPhaseNote = String.localizedStringWithFormat(NSLocalizedString("cleanup.status.fixSubtitles", comment: ""), 0, total)
        emit()

        var cleanedTextByIndex: [Int: String] = [:]
        var fallbackCount = 0
        var completed = 0

        try await withThrowingTaskGroup(of: (Int, [Int: String]?).self) { group in
            var nextBatch = 0

            func addTask(_ idx: Int) {
                let batch = batches[idx]
                group.addTask { [self] in
                    try Task.checkCancellation()
                    let result = try await self.processBatch(batch: batch,
                                                             level: effectiveLevel,
                                                             system: system,
                                                             config: config)
                    return (idx, result)
                }
            }

            while nextBatch < total && nextBatch < maxConcurrentBatches {
                addTask(nextBatch)
                nextBatch += 1
            }

            while let (idx, result) = try await group.next() {
                if let result {
                    for (k, v) in result { cleanedTextByIndex[k] = v }
                } else {
                    // Fail closed: keep RAW text for the WHOLE batch.
                    fallbackCount += 1
                    for seg in batches[idx] { cleanedTextByIndex[seg.index] = seg.text }
                }
                completed += 1
                curFraction = 0.7 * Double(completed) / Double(total)
                curPhaseNote = String.localizedStringWithFormat(NSLocalizedString("cleanup.status.fixSubtitles", comment: ""), completed, total)
                emit()

                try Task.checkCancellation()
                if nextBatch < total {
                    addTask(nextBatch)
                    nextBatch += 1
                }
            }
        }

        let cleaned = segments.map { seg -> TimedSegment in
            var s = seg
            if let t = cleanedTextByIndex[seg.index] { s.text = t }
            return s
        }
        return (cleaned, total, fallbackCount)
    }

    /// Processes one batch. Returns the cleaned text keyed by global index, or
    /// nil to signal a fail-closed fallback. Throws ONLY CancellationError.
    private func processBatch(batch: [TimedSegment],
                              level: CleanupLevel,
                              system: String,
                              config: LLMConfig) async throws -> [Int: String]? {

        struct InItem: Encodable { let i: Int; let t: String }
        let expected = Set(batch.map { $0.index })
        let inItems = batch.map { InItem(i: $0.index, t: $0.text) }
        guard let jsonData = try? JSONEncoder().encode(inItems),
              let jsonStr = String(data: jsonData, encoding: .utf8) else {
            return nil
        }

        let baseUser = LLMPrompts.srtUser(json: jsonStr)
        var nudge = ""

        for attempt in 0..<3 {
            try Task.checkCancellation()
            let userMsg = baseUser + nudge
            do {
                let content = try await performChat(config: config,
                                                    system: system,
                                                    user: userMsg,
                                                    maxTokens: nil)
                if let parsed = parseItems(content),
                   let map = validate(items: parsed,
                                      expectedIndices: expected,
                                      batch: batch,
                                      level: level) {
                    return map
                }
                // Parse / validation failure -> retry with a corrective nudge.
                nudge = LLMPrompts.correctiveNudge(count: batch.count, indices: batch.map { $0.index })
                if attempt < 2 { try await sleepBackoff(attempt: attempt, retryAfter: nil) }
            } catch is CancellationError {
                throw CancellationError()
            } catch LLMRequestError.badURL {
                return nil
            } catch LLMRequestError.httpStatus(let code, let ra) {
                if code == 429 || (500...599).contains(code) {
                    if attempt < 2 { try await sleepBackoff(attempt: attempt, retryAfter: ra) }
                } else {
                    return nil   // non-retryable client error (401/400/404…)
                }
            } catch {
                // Transport error / timeout / empty content -> retryable.
                if attempt < 2 { try await sleepBackoff(attempt: attempt, retryAfter: nil) }
            }
        }
        return nil
    }

    private func makeBatches(_ segments: [TimedSegment]) -> [[TimedSegment]] {
        var batches: [[TimedSegment]] = []
        var current: [TimedSegment] = []
        var currentChars = 0
        for seg in segments {
            let segChars = seg.text.count + 16  // rough JSON-wrapping overhead
            if !current.isEmpty &&
                (current.count >= maxSegmentsPerBatch || currentChars + segChars > maxCharsPerBatch) {
                batches.append(current)
                current = []
                currentChars = 0
            }
            current.append(seg)
            currentChars += segChars
        }
        if !current.isEmpty { batches.append(current) }
        return batches
    }

    // MARK: - Pass B (TXT, full level)

    private func runPassB(cleanedSegments: [TimedSegment],
                          level: CleanupLevel,
                          language: String?,
                          config: LLMConfig,
                          progress: @escaping @Sendable (Double, String) -> Void)
        async throws -> (txt: String, fellBack: Bool) {

        // Start from the already-corrected Pass-A text.
        let joined = TextJoiner.join(cleanedSegments.map { $0.text })

        // L1 keeps wording — no Pass B needed.
        if level == .fixOnly {
            curFraction = 1.0
            emit()
            return (joined, false)
        }
        if joined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            curFraction = 1.0
            emit()
            return (joined, false)
        }

        let system = (level == .lightEdit)
            ? LLMPrompts.txtL3System(language: language)
            : LLMPrompts.txtL2System(language: language)

        let chunks = splitForPassB(joined, maxChars: maxPassBChunkChars)
        let total = chunks.count

        var outputByIndex: [Int: String] = [:]
        var fellBack = false
        var done = 0

        try await withThrowingTaskGroup(of: (Int, String?).self) { group in
            var nextChunk = 0

            func addTask(_ idx: Int) {
                let chunk = chunks[idx]
                group.addTask { [self] in
                    try Task.checkCancellation()
                    let result = try await self.passBChunk(text: chunk, system: system, config: config)
                    return (idx, result)
                }
            }

            while nextChunk < total && nextChunk < maxConcurrentBatches {
                addTask(nextChunk)
                nextChunk += 1
            }

            while let (idx, result) = try await group.next() {
                if let result {
                    outputByIndex[idx] = result
                } else {
                    outputByIndex[idx] = chunks[idx]   // fall back to the (already cleaned) joined chunk
                    fellBack = true
                }
                done += 1
                curFraction = 0.7 + 0.3 * Double(done) / Double(total)
                curPhaseNote = String.localizedStringWithFormat(NSLocalizedString("cleanup.status.polishText", comment: ""), done, total)
                emit()

                try Task.checkCancellation()
                if nextChunk < total {
                    addTask(nextChunk)
                    nextChunk += 1
                }
            }
        }

        let outputs = (0..<total).map { outputByIndex[$0] ?? chunks[$0] }
        return (outputs.joined(separator: "\n\n"), fellBack)
    }

    /// Runs one Pass-B chunk. Returns cleaned prose, or nil to fall back.
    /// Throws ONLY CancellationError.
    private func passBChunk(text: String,
                            system: String,
                            config: LLMConfig) async throws -> String? {
        let user = LLMPrompts.txtUser(text: text)

        for attempt in 0..<3 {
            try Task.checkCancellation()
            do {
                let content = try await performChat(config: config,
                                                    system: system,
                                                    user: user,
                                                    maxTokens: nil)
                let cleaned = stripCodeFences(content)
                let trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
                // Accept only when the cleaned length is within a both-sided RELATIVE
                // range of the input: rejects an "OK"-style truncation AND an oversized
                // hallucinated appendix, while letting valid short cleanups through
                // ("um I agree" -> "I agree." no longer trips an absolute-20 floor).
                let lower = Int(0.3 * Double(text.count))
                let upper = Int(2.0 * Double(text.count)) + 40
                if (lower...upper).contains(trimmed.count) {
                    return cleaned
                }
                if attempt < 2 { try await sleepBackoff(attempt: attempt, retryAfter: nil) }
            } catch is CancellationError {
                throw CancellationError()
            } catch LLMRequestError.badURL {
                return nil
            } catch LLMRequestError.httpStatus(let code, let ra) {
                if code == 429 || (500...599).contains(code) {
                    if attempt < 2 { try await sleepBackoff(attempt: attempt, retryAfter: ra) }
                } else {
                    return nil
                }
            } catch {
                if attempt < 2 { try await sleepBackoff(attempt: attempt, retryAfter: nil) }
            }
        }
        return nil
    }

    private func splitForPassB(_ text: String, maxChars: Int) -> [String] {
        if text.count <= maxChars { return [text] }
        let paras = text
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if paras.count > 1 {
            return packUnits(paras, separator: "\n\n", maxChars: maxChars)
        }
        return packUnits(splitSentences(text), separator: "", maxChars: maxChars)
    }

    /// Hard-splits any single unit longer than `maxChars` into <= maxChars pieces
    /// by character count, so no chunk can ever exceed the budget.
    private func hardSplit(_ unit: String, maxChars: Int) -> [String] {
        guard maxChars > 0, unit.count > maxChars else { return [unit] }
        var pieces: [String] = []
        let chars = Array(unit)
        var i = 0
        while i < chars.count {
            let end = min(i + maxChars, chars.count)
            pieces.append(String(chars[i..<end]))
            i = end
        }
        return pieces
    }

    private func packUnits(_ rawUnits: [String], separator: String, maxChars: Int) -> [String] {
        // Explode any oversize unit first so no chunk exceeds the budget.
        let units = rawUnits.flatMap { hardSplit($0, maxChars: maxChars) }
        var chunks: [String] = []
        var current = ""
        for unit in units {
            if current.isEmpty {
                current = unit
            } else if current.count + separator.count + unit.count <= maxChars {
                current += separator + unit
            } else {
                chunks.append(current)
                current = unit
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks.isEmpty ? [units.joined(separator: separator)] : chunks
    }

    private func splitSentences(_ text: String) -> [String] {
        var result: [String] = []
        var current = ""
        let chars = Array(text)
        for (i, ch) in chars.enumerated() {
            current.append(ch)
            let isCJKEnd = "。！？".contains(ch)
            // English sentence end: . ! ? followed by whitespace or end-of-string.
            let isASCIIEnd: Bool = {
                guard ".!?".contains(ch) else { return false }
                let next = i + 1 < chars.count ? chars[i + 1] : nil
                return next == nil || next!.isWhitespace
            }()
            let isNewline = ch.isNewline
            if isCJKEnd || isASCIIEnd || isNewline {
                result.append(current)
                current = ""
            }
        }
        if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            result.append(current)
        }
        return result.isEmpty ? [text] : result
    }

    // MARK: - HTTP

    private func endpointURL(_ base: String) -> URL? {
        var s = base.trimmingCharacters(in: .whitespacesAndNewlines)
        while s.hasSuffix("/") { s.removeLast() }
        guard !s.isEmpty else { return nil }
        if s.hasSuffix("/chat/completions") { return URL(string: s) }
        // Cloudflare AI Gateway unified endpoint: ".../compat" -> ".../compat/chat/completions"
        if s.hasSuffix("/compat") { return URL(string: s + "/chat/completions") }
        if s.hasSuffix("/v1") { return URL(string: s + "/chat/completions") }
        return URL(string: s + "/v1/chat/completions")
    }

    private struct ChatMessage: Encodable { let role: String; let content: String }
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

    /// One streamed chat-completions call. Returns the concatenated `delta.content`
    /// (the answer). `reasoning_content` is chain-of-thought: counted toward the live
    /// counter but NOT returned. Throws LLMRequestError or the underlying transport error.
    private func performChat(config: LLMConfig,
                             system: String?,
                             user: String,
                             maxTokens: Int?,
                             reportLiveProgress: Bool = true) async throws -> String {
        guard let url = endpointURL(config.baseURL) else { throw LLMRequestError.badURL }

        var messages: [ChatMessage] = []
        if let system { messages.append(ChatMessage(role: "system", content: system)) }
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

        var finalContent = ""
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
            if let c = delta.content {
                finalContent += c
                if reportLiveProgress { generatedChars += c.count }
            }
            if reportLiveProgress, let r = delta.reasoning_content {
                // Chain-of-thought: counts toward the live counter only.
                generatedChars += r.count
            }
            if reportLiveProgress && generatedChars - lastEmittedChars >= 12 {
                lastEmittedChars = generatedChars
                emit()
            }
        }

        if reportLiveProgress { emit() }   // final count for this stream

        if finalContent.isEmpty { throw LLMRequestError.emptyContent }
        return finalContent
    }

    /// Exponential backoff with per-attempt jitter; honors Retry-After when present.
    private func sleepBackoff(attempt: Int, retryAfter: TimeInterval?) async throws {
        let base: [TimeInterval] = [0.5, 2.0, 8.0]
        let b = attempt < base.count ? base[attempt] : 8.0
        let jitter = Double(attempt) * 0.13 + 0.07
        let wait = (retryAfter ?? b) + jitter
        let safe = wait.isFinite ? min(max(wait, 0), 30) : 0   // clamp: no negative/NaN/huge Retry-After
        try await Task.sleep(nanoseconds: UInt64(safe * 1_000_000_000))
    }

    // MARK: - Parsing (defensive, stop at first success)

    private struct OutItem: Decodable {
        let i: Int
        let t: String
        enum CodingKeys: String, CodingKey { case i, t }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            if let intI = try? c.decode(Int.self, forKey: .i) {
                i = intI
            } else if let strI = try? c.decode(String.self, forKey: .i), let parsed = Int(strI) {
                i = parsed
            } else {
                throw DecodingError.dataCorruptedError(forKey: .i, in: c,
                    debugDescription: "i is not an integer")
            }
            if let strT = try? c.decode(String.self, forKey: .t) {
                t = strT
            } else {
                throw DecodingError.dataCorruptedError(forKey: .t, in: c,
                    debugDescription: "t is not a string")
            }
        }
    }

    private func parseItems(_ content: String) -> [OutItem]? {
        let dec = JSONDecoder()
        // 1) decode the whole content
        if let d = content.data(using: .utf8), let arr = try? dec.decode([OutItem].self, from: d) {
            return arr
        }
        // 2) strip Markdown fences then retry
        let stripped = stripCodeFences(content)
        if stripped != content,
           let d = stripped.data(using: .utf8),
           let arr = try? dec.decode([OutItem].self, from: d) {
            return arr
        }
        // 3) locate the outermost balanced [...] and parse it
        if let sub = outermostJSONArray(stripped) ?? outermostJSONArray(content),
           let d = sub.data(using: .utf8),
           let arr = try? dec.decode([OutItem].self, from: d) {
            return arr
        }
        return nil
    }

    private func stripCodeFences(_ s: String) -> String {
        var t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard t.hasPrefix("```") else { return t }
        if let firstNL = t.firstIndex(of: "\n") {
            t = String(t[t.index(after: firstNL)...])
        } else {
            t = String(t.dropFirst(3))
        }
        if let range = t.range(of: "```", options: .backwards) {
            t = String(t[..<range.lowerBound])
        }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Bracket scan that respects string literals and escapes.
    private func outermostJSONArray(_ s: String) -> String? {
        let chars = Array(s)
        guard let start = chars.firstIndex(of: "[") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var i = start
        while i < chars.count {
            let ch = chars[i]
            if inString {
                if escaped { escaped = false }
                else if ch == "\\" { escaped = true }
                else if ch == "\"" { inString = false }
            } else {
                if ch == "\"" { inString = true }
                else if ch == "[" { depth += 1 }
                else if ch == "]" {
                    depth -= 1
                    if depth == 0 { return String(chars[start...i]) }
                }
            }
            i += 1
        }
        return nil
    }

    // MARK: - Validation (fail closed)

    /// Returns cleaned text keyed by global index, or nil if the batch must fall back.
    private func validate(items: [OutItem],
                          expectedIndices: Set<Int>,
                          batch: [TimedSegment],
                          level: CleanupLevel) -> [Int: String]? {
        // Count + exact set match together guarantee no missing/extra/duplicate.
        guard items.count == expectedIndices.count else { return nil }
        guard Set(items.map { $0.i }) == expectedIndices else { return nil }

        let rawByIndex = Dictionary(batch.map { ($0.index, $0.text) }, uniquingKeysWith: { _, new in new })
        let lower = (level == .fixOnly) ? 0.3 : 0.2
        let upper = (level == .fixOnly) ? 1.8 : 1.5

        var map: [Int: String] = [:]
        var nonEmptyRawCount = 0
        var emptiedCount = 0
        for item in items {
            let raw = (rawByIndex[item.i] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let cleanLen = item.t.trimmingCharacters(in: .whitespacesAndNewlines).count
            let rawLen = raw.count
            if rawLen > 0 {
                nonEmptyRawCount += 1
                if cleanLen == 0 {
                    // L1 must keep words; an emptied segment is suspect -> fail.
                    if level == .fixOnly { return nil }
                    emptiedCount += 1
                    // L2 allows emptying a filler-only segment.
                } else {
                    let ratio = Double(cleanLen) / Double(rawLen)
                    // Lower (shrink) bound always applies.
                    if ratio < lower { return nil }
                    // Upper (growth) bound with BOUNDED slack, applied to ALL segments:
                    // short ones get rawLen+6 slack, longer ones get rawLen*upper. This
                    // stops a short raw ("OK") validating against a paragraph-length
                    // hallucination, while still letting "好" -> "好。" pass.
                    let maxLen = max(Int(Double(rawLen) * upper), rawLen + 6)
                    if cleanLen > maxLen { return nil }
                }
            }
            map[item.i] = item.t
        }
        // Batch-level mass-emptying guard: if the model emptied most of the
        // originally-non-empty segments, fail closed (-> raw fallback) instead of
        // emitting a blank SRT. Only applied when there are >= 4 non-empty raw
        // segments; on smaller batches a lone filler-only segment ("um" -> "") may
        // legitimately be emptied, so the guard is skipped.
        if nonEmptyRawCount >= 4 && Double(emptiedCount) / Double(nonEmptyRawCount) > 0.6 {
            return nil
        }
        return map
    }
}
