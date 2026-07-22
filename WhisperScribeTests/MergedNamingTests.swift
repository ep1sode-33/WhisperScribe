import Testing
import Foundation
@testable import WhisperScribe

struct MergedNamingTests {
    @Test func mergedNameNextToSource() {
        let u = FileNaming.mergedTextURL(firstSource: URL(fileURLWithPath: "/tmp/shot 1.png"),
                                         outputDir: nil, overwrite: .overwrite)
        #expect(u.path == "/tmp/shot 1 merged.txt")
    }
    @Test func mergedNameUniquifies() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let src = dir.appending(path: "a.png")
        FileManager.default.createFile(atPath: dir.appending(path: "a merged.txt").path, contents: Data())
        let u = FileNaming.mergedTextURL(firstSource: src, outputDir: nil, overwrite: .uniquify)
        #expect(u.lastPathComponent == "a merged 2.txt")
    }
    @Test func writeMergedTextRoundTrip() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: UUID().uuidString)
        let out = try SubtitleWriter.writeMergedText("你好\nworld",
                                                     firstSource: dir.appending(path: "x.wav"),
                                                     outputDir: dir, overwrite: .overwrite)
        #expect(try String(contentsOf: out, encoding: .utf8) == "你好\nworld")
        #expect(out.lastPathComponent == "x merged.txt")
    }
}
