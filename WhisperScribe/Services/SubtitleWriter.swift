import Foundation

enum SubtitleWriter {
    static func write(segments: [TimedSegment], txt: String, source: URL, outputDir: URL?, overwrite: OverwritePolicy) throws -> (srt: URL, txt: URL) {
        let urls = FileNaming.outputURLs(source: source, outputDir: outputDir, overwrite: overwrite)
        let srtContent = SRTFormatter.srt(from: segments)
        // Ensure the destination directory exists (e.g. a custom output folder that
        // was deleted/moved) so a missing folder doesn't fail the job. srt and txt
        // share a parent directory.
        try? FileManager.default.createDirectory(at: urls.srt.deletingLastPathComponent(), withIntermediateDirectories: true)
        do {
            try srtContent.write(to: urls.srt, atomically: true, encoding: .utf8)
            try txt.write(to: urls.txt, atomically: true, encoding: .utf8)
        } catch {
            throw AppError.writeFailed(String(describing: error))
        }
        return (urls.srt, urls.txt)
    }
}
