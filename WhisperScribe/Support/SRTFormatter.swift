import Foundation

enum SRTFormatter {
    static func srt(from segments: [TimedSegment]) -> String {
        var cues: [String] = []
        var number = 1
        for segment in segments {
            let text = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }
            let start = timestamp(from: segment.start)
            let end = timestamp(from: segment.end)
            cues.append("\(number)\n\(start) --> \(end)\n\(text)\n")
            number += 1
        }
        return cues.joined(separator: "\n")
    }

    private static func timestamp(from seconds: TimeInterval) -> String {
        let totalMillis = max(0, Int((seconds * 1000).rounded()))
        let millis = totalMillis % 1000
        let totalSeconds = totalMillis / 1000
        let secs = totalSeconds % 60
        let totalMinutes = totalSeconds / 60
        let mins = totalMinutes % 60
        let hours = totalMinutes / 60
        return String(format: "%02d:%02d:%02d,%03d", hours, mins, secs, millis)
    }
}
