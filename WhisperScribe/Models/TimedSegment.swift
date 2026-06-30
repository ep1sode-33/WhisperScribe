import Foundation

/// One timed transcript segment. Times are in seconds.
/// `index` is the GLOBAL segment index, stable across the whole transcript and
/// used by the LLM cleanup pass to re-attach cleaned text to its timestamps.
struct TimedSegment: Identifiable, Codable, Equatable {
    let index: Int
    let start: TimeInterval
    let end: TimeInterval
    var text: String

    var id: Int { index }

    init(index: Int, start: TimeInterval, end: TimeInterval, text: String) {
        self.index = index
        self.start = start
        self.end = end
        self.text = text
    }
}
