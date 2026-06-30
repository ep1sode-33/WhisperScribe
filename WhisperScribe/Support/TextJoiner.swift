import Foundation

enum TextJoiner {
    static func join(_ texts: [String]) -> String {
        let pieces = texts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard var result = pieces.first else { return "" }
        for piece in pieces.dropFirst() {
            if let left = result.unicodeScalars.last,
               let right = piece.unicodeScalars.first,
               isCJK(left), isCJK(right) {
                result += piece
            } else {
                result += " " + piece
            }
        }
        return result
    }

    static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        let value = scalar.value
        switch value {
        case 0x4E00...0x9FFF,   // CJK Unified Ideographs
             0x3400...0x4DBF,   // CJK Extension-A
             0x3040...0x309F,   // Hiragana
             0x30A0...0x30FF,   // Katakana
             0x3000...0x303F,   // CJK Symbols and Punctuation
             0xFF00...0xFFEF:   // Halfwidth and Fullwidth Forms
            return true
        default:
            return false
        }
    }
}
