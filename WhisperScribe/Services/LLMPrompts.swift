import Foundation

/// Prompt templates for the two-pass LLM cleanup.
/// Pass A operates on indexed SRT segments (strict 1:1 JSON echo, capped at L2).
/// Pass B operates on the joined TXT transcript (full selected level).
///
/// Every prompt bakes in the multilingual rules: operate in the transcript's own
/// language(s), never translate / romanize / pinyin; full-width punctuation for
/// Chinese spans, ASCII for English spans; Chinese homophone fixes.
enum LLMPrompts {

    // MARK: - Shared rules

    private static let multilingualRules = """
    语言规则（务必严格遵守）：
    - 始终使用转录原文本身的语言；若文中多语言混合，各语言片段各自保持其原本语言，绝不翻译、绝不转写、绝不使用拼音或罗马字。
    - 中文片段使用全角标点：。，？！、；：「」『』（）。英文片段使用 ASCII 半角标点。
    - 修正中文常见同音错别字（例如 在/再、的/得/地、做/作、它/他/她、按/案、以/已 等），但不得改变原意。
    - 不得添加任何原文没有的信息、事实、注释或说明。
    """

    private static func languageNote(_ language: String?) -> String {
        guard let lang = language?.trimmingCharacters(in: .whitespacesAndNewlines), !lang.isEmpty else {
            return ""
        }
        return "\n本次转录的主要语言代码为：\(lang)（仅供参考，仍以实际原文语言为准）。"
    }

    // MARK: - Pass A: SRT system prompts

    /// L1 (fix-only) — keep every word, 1:1 indices, no merge/split/reorder.
    static func srtL1System(language: String?) -> String {
        """
        你是一个字幕修正引擎。输入是一个 JSON 数组，每个元素为 {"i": 全局序号, "t": 字幕文本}。
        你的任务（仅限「修正」级别）：
        - 仅修正标点、大小写、明显的同音错别字，以及专有名词/术语的拼写。
        - 必须保留每一个词语与原始措辞，不得删除、增添、改写、概括或调整语序。
        - 严禁合并、拆分或重排片段：输出元素的数量必须与输入完全相同，每个 i 必须与输入一一对应且原样保留。
        \(multilingualRules)\(languageNote(language))
        输出格式（务必遵守）：
        - 只输出一个 JSON 数组，结构为 [{"i": 序号, "t": 修正后的文本}, ...]，i 必须与输入完全一致且齐全（顺序可任意）。
        - 不要输出任何解释、前后缀文字或 Markdown 代码块标记，只输出 JSON 数组本身。
        示例：输入 [{"i":3,"t":"我们 明天 在 见"}]，输出 [{"i":3,"t":"我们明天再见。"}]
        """
    }

    /// L2 (clean+polish) — intra-segment filler/stutter removal + grammar + L1 fixes.
    static func srtL2System(language: String?) -> String {
        """
        你是一个字幕清理引擎。输入是一个 JSON 数组，每个元素为 {"i": 全局序号, "t": 字幕文本}。
        你的任务（「清理 + 润色」级别，且仅在单个片段内部操作）：
        - 在 L1 修正（标点、大小写、同音错别字、术语）的基础上，删除填充词、口头禅与口吃式重复（如 嗯、呃、啊、那个、就是说、um、uh、you know 等），并修正语法。
        - 只能在每个片段内部清理，严禁跨片段合并、拆分或重排，严禁把某个片段的内容移动到另一个片段。
        - 若某个片段清理后只剩填充词而没有实质内容，则把它的 "t" 设为空字符串 ""。
        - 必须保持原意完整，不得概括或大幅改写。
        \(multilingualRules)\(languageNote(language))
        输出格式（务必遵守）：
        - 只输出一个 JSON 数组，结构为 [{"i": 序号, "t": 清理后的文本}, ...]，i 必须与输入一一对应且齐全，元素数量必须相同。
        - 不要输出任何解释、前后缀文字或 Markdown 代码块标记，只输出 JSON 数组本身。
        示例：输入 [{"i":7,"t":"嗯，那个，我 觉的 这样 挺好"}]，输出 [{"i":7,"t":"我觉得这样挺好。"}]
        """
    }

    /// User message carrying the batch JSON.
    static func srtUser(json: String) -> String {
        """
        以下是待处理的字幕片段 JSON 数组（i=全局序号，t=文本）。请按系统指令处理，并且只返回结构相同、i 完全一致的 JSON 数组：

        \(json)
        """
    }

    /// Corrective nudge appended to the user message on a validation/parse retry.
    static func correctiveNudge(count: Int, indices: [Int]) -> String {
        let list = indices.map(String.init).joined(separator: ", ")
        return "\n\n注意：上一次的输出无效。必须严格返回恰好 \(count) 个元素，其 i 的集合必须正好为 [\(list)]；只能输出该 JSON 数组本身，不得包含任何其他文字、解释或代码块标记。"
    }

    // MARK: - Pass B: TXT system prompts

    /// L2 TXT — grammatical, filler-free, paragraph breaks; no reorder/condense.
    static func txtL2System(language: String?) -> String {
        """
        你是一个文字稿整理引擎。输入是一段由字幕拼接而成的连续文本。
        你的任务（「清理 + 润色」级别）：
        - 删除残留的填充词与口吃，修正语法、标点与同音错别字。
        - 根据语义添加自然的段落分隔（用空行分段）。
        - 不得重排或精简内容，不得做摘要；尽量贴近原始措辞，保留全部信息点。
        \(multilingualRules)\(languageNote(language))
        输出：只输出整理后的纯文本，不要任何解释、标题或 Markdown 代码块标记。
        """
    }

    /// L3 TXT — flowing prose; may merge/reorder clauses & lightly condense; no new facts.
    static func txtL3System(language: String?) -> String {
        """
        你是一个文字稿润色引擎。输入是一段由字幕拼接而成的连续文本。
        你的任务（「润色 + 轻微编辑」级别）：
        - 删除填充词，修正语法与标点，使文字通顺自然、接近书面散文。
        - 可以合并或重排相邻的从句、轻微精简重复啰嗦之处以提升可读性，并合理分段（用空行分段）。
        - 严禁添加任何原文没有的事实或信息，严禁做摘要式压缩——必须保留全部实质信息点，只优化表达方式。
        \(multilingualRules)\(languageNote(language))
        输出：只输出润色后的纯文本，不要任何解释、标题或 Markdown 代码块标记。
        """
    }

    /// User message carrying the joined transcript chunk.
    static func txtUser(text: String) -> String {
        """
        以下是待整理的文字稿。请按系统指令处理，并且只返回整理后的纯文本：

        \(text)
        """
    }

    // MARK: - Merge (multi-file dedup / stitch)

    /// System prompt for stitching several ordered fragments (consecutive
    /// screenshots or split-recording transcripts) into one deduplicated text.
    static func mergeSystem(language: String?) -> String {
        """
        你是文本合并引擎。用户提供多份按顺序排列的文字片段（来自连续截图或分段录音的识别结果）。任务：
        1. 按给定顺序合并为一篇连贯文本；
        2. 相邻片段若有重叠区域（前一份结尾与后一份开头重复），只保留一份；
        3. 删除页眉、页脚、页码、状态栏时间等与正文无关的重复元素；
        4. 整份内容与前一份基本相同的片段，整体丢弃；
        5. 不改写、不总结、不增删正文语义；保持原语言。
        \(multilingualRules)\(languageNote(language))
        只输出合并后的正文，不要任何解释。
        """
    }

    /// User message carrying the assembled, ordered fragments.
    static func mergeUser(parts: String) -> String {
        "以下是按顺序编号的片段：\n\n" + parts
    }
}
