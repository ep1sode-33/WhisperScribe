import Foundation

/// LLM-powered merge / dedup for multi-file transcripts (consecutive screenshots or
/// split-recording segments). Fail-open: any transport/LLM problem — or a
/// misconfigured endpoint — degrades to an ordered, headed concatenation plus a
/// warning, so the pipeline always produces output. Never throws.
actor MergeService {

    private let chat: ChatStreaming
    init(chat: ChatStreaming = LLMTransportChat()) { self.chat = chat }

    /// Assembled user-payload budget per LLM call. A single part is never split
    /// across calls (an oversize part still goes out whole, in its own group).
    private let maxCharsPerGroup = 12000
    /// Tail of the previous group's result carried into the next call for continuity.
    private let tailContextChars = 600

    func merge(parts: [(name: String, text: String)],
               language: String?,
               config: LLMConfig,
               progress: @escaping @Sendable (Double, String) -> Void) async -> (text: String, warnings: [String]) {

        if parts.isEmpty { return ("", []) }
        if !config.isConfigured {
            return (fallbackJoin(parts), [String(localized: "cleanup.warning.mergeFallback")])
        }

        do {
            let text = try await mergeViaLLM(parts: parts, language: language,
                                             config: config, progress: progress)
            return (text, [])
        } catch {
            // Any throw (network, HTTP, empty content, cancellation, …) -> fail open.
            return (fallbackJoin(parts), [String(localized: "cleanup.warning.mergeFallback")])
        }
    }

    // MARK: - LLM path

    private func mergeViaLLM(parts: [(name: String, text: String)],
                             language: String?,
                             config: LLMConfig,
                             progress: @escaping @Sendable (Double, String) -> Void) async throws -> String {
        let system = LLMPrompts.mergeSystem(language: language)
        // One assembled segment per part: 【片段 i · 文件名】\n正文 (1-based, global order).
        let segments = parts.enumerated().map { (i, p) in
            "【片段 \(i + 1) · \(p.name)】\n\(p.text)"
        }
        let groups = groupSegments(segments)
        let total = groups.count

        let merging = String(localized: "status.merging")
        progress(0, merging)

        var results: [String] = []
        var prevTail = ""
        for (gi, group) in groups.enumerated() {
            try Task.checkCancellation()
            var payload = group.joined(separator: "\n\n")
            // From the 2nd group on, prepend the previous group's tail for continuity.
            if gi > 0 && !prevTail.isEmpty {
                payload = "【已合并文本的结尾，供衔接】\n\(prevTail)\n\n" + payload
            }
            let base = Double(gi) / Double(total)
            let counter = CharCounter()
            let out = try await chat.streamChat(config: config,
                                                system: system,
                                                user: LLMPrompts.mergeUser(parts: payload)) { delta in
                // Refine progress within a group by streamed char count (LLMCleaner note style).
                let n = counter.add(delta.count)
                let note = merging + String.localizedStringWithFormat(
                    NSLocalizedString("cleanup.status.generatedCharsSuffix", comment: ""), n)
                progress(base, note)
            }
            results.append(out)
            prevTail = String(out.suffix(tailContextChars))
            progress(Double(gi + 1) / Double(total), merging)
        }
        return results.joined(separator: "\n\n")
    }

    /// Packs assembled segment strings into groups whose joined length stays within
    /// `maxCharsPerGroup`. Parts are never split: an oversize part gets its own group.
    private func groupSegments(_ segments: [String]) -> [[String]] {
        var groups: [[String]] = []
        var current: [String] = []
        var currentChars = 0
        for seg in segments {
            let sep = current.isEmpty ? 0 : 2   // "\n\n" joiner
            if !current.isEmpty && currentChars + sep + seg.count > maxCharsPerGroup {
                groups.append(current)
                current = [seg]
                currentChars = seg.count
            } else {
                current.append(seg)
                currentChars += sep + seg.count
            }
        }
        if !current.isEmpty { groups.append(current) }
        return groups
    }

    // MARK: - Fallback

    /// Ordered concatenation with per-part headers (first part unheaded); a single
    /// part is returned verbatim. The separator format is load-bearing for callers.
    private func fallbackJoin(_ parts: [(name: String, text: String)]) -> String {
        if parts.count == 1 { return parts[0].text }
        var out = ""
        for (i, p) in parts.enumerated() {
            if i == 0 { out = p.text }
            else { out += "\n\n----- \(p.name) -----\n\n" + p.text }
        }
        return out
    }
}

/// Small locked counter so the `@Sendable` onDelta closure can accumulate a running
/// character total (off the actor) without a data race.
private final class CharCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func add(_ n: Int) -> Int {
        lock.lock(); defer { lock.unlock() }
        count += n
        return count
    }
}
