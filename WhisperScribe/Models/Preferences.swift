import Foundation

/// Where to write the .srt / .txt outputs.
enum OutputMode: Int, CaseIterable, Identifiable, Codable {
    case nextToSource = 0
    case customFolder = 1

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .nextToSource: return String(localized: "output.mode.nextToSource.title")
        case .customFolder: return String(localized: "output.mode.customFolder.title")
        }
    }
}

/// What to do when an output file with the same name already exists.
enum OverwritePolicy: Int, CaseIterable, Identifiable, Codable {
    case uniquify = 0   // write "name 2.srt" to preserve the old file
    case overwrite = 1

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .uniquify:  return String(localized: "output.overwrite.uniquify.title")
        case .overwrite: return String(localized: "output.overwrite.overwrite.title")
        }
    }
}
