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
}
