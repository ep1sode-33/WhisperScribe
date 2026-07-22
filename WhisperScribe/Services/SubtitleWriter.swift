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

    /// Writes the single merged `.txt` for a batch job. Creates the parent
    /// directory if missing, writes UTF-8 atomically, and wraps any failure in
    /// `AppError.writeFailed` (mirrors `write` above).
    static func writeMergedText(_ text: String, firstSource: URL, outputDir: URL?, overwrite: OverwritePolicy) throws -> URL {
        let url = FileNaming.mergedTextURL(firstSource: firstSource, outputDir: outputDir, overwrite: overwrite)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        do {
            try text.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            throw AppError.writeFailed(String(describing: error))
        }
        return url
    }
}
