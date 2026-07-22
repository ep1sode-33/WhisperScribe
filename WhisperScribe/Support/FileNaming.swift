import Foundation

enum FileNaming {
    static func outputURLs(source: URL, outputDir: URL?, overwrite: OverwritePolicy) -> (srt: URL, txt: URL) {
        let base = source.deletingPathExtension().lastPathComponent
        let dir = outputDir ?? source.deletingLastPathComponent()

        func pair(_ suffix: String) -> (srt: URL, txt: URL) {
            (
                dir.appendingPathComponent("\(base)\(suffix).srt"),
                dir.appendingPathComponent("\(base)\(suffix).txt")
            )
        }

        if overwrite == .overwrite {
            return pair("")
        }

        func isFree(_ suffix: String) -> Bool {
            let urls = pair(suffix)
            return !FileManager.default.fileExists(atPath: urls.srt.path)
                && !FileManager.default.fileExists(atPath: urls.txt.path)
        }

        if isFree("") {
            return pair("")
        }
        var counter = 2
        while true {
            let suffix = " \(counter)"
            if isFree(suffix) {
                return pair(suffix)
            }
            counter += 1
        }
    }

    /// URL for the single merged `.txt` produced by a batch job.
    /// Base name = firstSource without extension + " merged"; directory rule
    /// matches `outputURLs` (outputDir ?? source directory). Under `.uniquify`,
    /// probes a single txt file and appends " 2"/" 3"… to avoid overwriting.
    static func mergedTextURL(firstSource: URL, outputDir: URL?, overwrite: OverwritePolicy) -> URL {
        let base = "\(firstSource.deletingPathExtension().lastPathComponent) merged"
        let dir = outputDir ?? firstSource.deletingLastPathComponent()

        func txt(_ suffix: String) -> URL {
            dir.appendingPathComponent("\(base)\(suffix).txt")
        }

        if overwrite == .overwrite {
            return txt("")
        }

        func isFree(_ suffix: String) -> Bool {
            !FileManager.default.fileExists(atPath: txt(suffix).path)
        }

        if isFree("") {
            return txt("")
        }
        var counter = 2
        while true {
            let suffix = " \(counter)"
            if isFree(suffix) {
                return txt(suffix)
            }
            counter += 1
        }
    }
}
