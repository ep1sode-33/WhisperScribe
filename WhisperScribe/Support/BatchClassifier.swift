import Foundation
import UniformTypeIdentifiers

/// A homogeneous batch of dropped URLs, already sorted by filename (natural order).
enum BatchKind: Equatable {
    case audio([URL])    // 已按文件名自然排序
    case images([URL])   // 已按文件名自然排序
}

/// Pure classification/sorting of dropped URLs into an audio batch or an image batch.
///
/// Mixing kinds, unsupported files, and empty input all throw `AppError`.
enum BatchClassifier {
    static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "heic", "webp", "tiff", "tif"]

    private enum Item {
        case audio
        case image
        case other
    }

    static func classify(_ urls: [URL]) throws -> BatchKind {
        var audio: [URL] = []
        var images: [URL] = []
        for url in urls {
            switch kind(of: url) {
            case .audio:
                audio.append(url)
            case .image:
                images.append(url)
            case .other:
                throw AppError.unsupportedFile(url.lastPathComponent)
            }
        }
        switch (audio.isEmpty, images.isEmpty) {
        case (false, false):
            throw AppError.mixedBatch
        case (false, true):
            return .audio(naturallySorted(audio))
        case (true, false):
            return .images(naturallySorted(images))
        case (true, true):
            // Empty input — UI never sends this; defensive contract.
            throw AppError.unsupportedFile("")
        }
    }

    /// Whether a single URL is accepted (audio or image) — used for drop highlighting.
    static func isAcceptedURL(_ url: URL) -> Bool {
        switch kind(of: url) {
        case .audio, .image:
            return true
        case .other:
            return false
        }
    }

    // MARK: - Private

    private static func kind(of url: URL) -> Item {
        if isImageURL(url) {
            return .image
        }
        if isAudioURL(url) {
            return .audio
        }
        return .other
    }

    private static func isImageURL(_ url: URL) -> Bool {
        imageExtensions.contains(url.pathExtension.lowercased())
    }

    /// Resolve the audio/video content type, falling back to the filename extension.
    private static func isAudioURL(_ url: URL) -> Bool {
        if let type = (try? url.resourceValues(forKeys: [.contentTypeKey]))?.contentType {
            return type.conforms(to: .audio) || type.conforms(to: .audiovisualContent)
        }
        if let type = UTType(filenameExtension: url.pathExtension) {
            return type.conforms(to: .audio) || type.conforms(to: .audiovisualContent)
        }
        return false
    }

    private static func naturallySorted(_ urls: [URL]) -> [URL] {
        urls.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }
}
