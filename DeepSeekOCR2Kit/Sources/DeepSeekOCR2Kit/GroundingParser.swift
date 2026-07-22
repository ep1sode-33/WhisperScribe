import CoreGraphics
import Foundation

/// One localized text span from a DeepSeek-OCR-2 grounding response: the
/// referenced text (`label`) and its bounding `box`, normalized to `[0, 1]`.
///
/// The reference emits box coordinates in a `0–1000` integer range
/// (`<|det|>[[x1, y1, x2, y2]]<|/det|>`, top-left origin); `parseGrounding`
/// divides by 1000 so `box` is resolution-independent and ready to scale onto
/// any rendered image.
public struct GroundingBox: Equatable, Sendable {
    public let label: String
    /// Normalized `[0, 1]` rect, top-left origin: `x/width == x1/1000`,
    /// `y/height == y1/1000`.
    public let box: CGRect

    public init(label: String, box: CGRect) {
        self.label = label
        self.box = box
    }
}

/// Extracts `<|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>` pairs from a
/// grounding response's raw (marker-preserving) text.
///
/// Only fully-formed pairs with all four integer coordinates are kept; a stray
/// `<|det|>` without a preceding `<|ref|>`, a 3-tuple, or a `<|ref|>` with no
/// `<|det|>` are silently dropped (a partial/garbled emission is not a box).
enum GroundingParser {
    // <|ref|>(label)<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
    //   * label is non-greedy so adjacent pairs don't merge.
    //   * `[\s\S]` (not `.`) so a label that spans a newline still matches.
    //   * digits are unsigned ints in the reference's 0–1000 range.
    private static let pattern =
        #"<\|ref\|>([\s\S]+?)<\|/ref\|><\|det\|>\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]<\|/det\|>"#

    private static let regex = try! NSRegularExpression(pattern: pattern)

    static func parse(_ text: String) -> [GroundingBox] {
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))

        var boxes: [GroundingBox] = []
        boxes.reserveCapacity(matches.count)
        for m in matches {
            guard m.numberOfRanges == 6 else { continue }
            let label = ns.substring(with: m.range(at: 1))
            guard
                let x1 = Double(ns.substring(with: m.range(at: 2))),
                let y1 = Double(ns.substring(with: m.range(at: 3))),
                let x2 = Double(ns.substring(with: m.range(at: 4))),
                let y2 = Double(ns.substring(with: m.range(at: 5)))
            else { continue }

            // Coordinate sanity (consistent with the drop-malformed policy): the
            // reference emits ordered corners in `0...1000`. Drop reversed
            // (`x2 < x1` / `y2 < y1`, which would yield a negative-size rect) or
            // out-of-range boxes -- a garbled emission is not a box. `\d+` already
            // guarantees the lower bound is >= 0.
            guard x1 <= x2, y1 <= y2, x2 <= 1000, y2 <= 1000 else { continue }

            let box = CGRect(
                x: x1 / 1000, y: y1 / 1000,
                width: (x2 - x1) / 1000, height: (y2 - y1) / 1000)
            boxes.append(GroundingBox(label: label, box: box))
        }
        return boxes
    }
}
