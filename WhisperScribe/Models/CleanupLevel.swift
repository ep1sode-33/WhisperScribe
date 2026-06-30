import Foundation

/// LLM post-processing intensity. Raw value persists in @AppStorage.
enum CleanupLevel: Int, CaseIterable, Identifiable, Codable {
    case raw = 0         // L0: no LLM at all
    case fixOnly = 1     // L1: punctuation/casing/homophones/terms; keep every word
    case cleanPolish = 2 // L2: + remove filler/stutters, fix grammar, paragraph breaks
    case lightEdit = 3   // L3: + condense/reorder (TXT only); SRT stays capped at L2

    var id: Int { rawValue }

    /// Whether this level needs the BYOK LLM at all.
    var usesLLM: Bool { self != .raw }

    var title: String {
        switch self {
        case .raw:         return String(localized: "cleanup.level.raw.title")
        case .fixOnly:     return String(localized: "cleanup.level.fixOnly.title")
        case .cleanPolish: return String(localized: "cleanup.level.cleanPolish.title")
        case .lightEdit:   return String(localized: "cleanup.level.lightEdit.title")
        }
    }

    var detail: String {
        switch self {
        case .raw:
            return String(localized: "cleanup.level.raw.detail")
        case .fixOnly:
            return String(localized: "cleanup.level.fixOnly.detail")
        case .cleanPolish:
            return String(localized: "cleanup.level.cleanPolish.detail")
        case .lightEdit:
            return String(localized: "cleanup.level.lightEdit.detail")
        }
    }
}
