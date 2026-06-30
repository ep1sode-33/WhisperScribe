import Foundation

enum SubtitleWriter {
    static func write(segments: [TimedSegment], txt: String, source: URL, outputDir: URL?, overwrite: OverwritePolicy) throws -> (srt: URL, txt: URL) {
        let urls = FileNaming.outputURLs(source: source, outputDir: outputDir, overwrite: overwrite)
        let srtContent = SRTFormatter.srt(from: segments)
        do {
            try srtContent.write(to: urls.srt, atomically: true, encoding: .utf8)
            try txt.write(to: urls.txt, atomically: true, encoding: .utf8)
        } catch {
            throw AppError.writeFailed(String(describing: error))
        }
        return (urls.srt, urls.txt)
    }
}
