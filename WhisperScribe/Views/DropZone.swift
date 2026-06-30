import SwiftUI
import UniformTypeIdentifiers

struct DropZone: View {
    var onPick: (URL) -> Void

    @State private var isTargeted = false

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "waveform.badge.plus")
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(isTargeted ? Color.accentColor : .secondary)
            Text("drop.title")
                .font(.title3)
                .foregroundStyle(.secondary)
            Text("drop.subtitle")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(32)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(isTargeted ? Color.accentColor.opacity(0.08) : Color.clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(
                    isTargeted ? Color.accentColor : Color.secondary.opacity(0.4),
                    style: StrokeStyle(lineWidth: 2, dash: [8, 6])
                )
        )
        .animation(.easeInOut(duration: 0.15), value: isTargeted)
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            handleDrop(providers)
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            guard let url, Self.isMediaURL(url) else { return }
            DispatchQueue.main.async {
                onPick(url)
            }
        }
        return true
    }

    static func isMediaURL(_ url: URL) -> Bool {
        // Prefer the resolved content type; fall back to the extension.
        if let type = (try? url.resourceValues(forKeys: [.contentTypeKey]))?.contentType {
            return type.conforms(to: .audio) || type.conforms(to: .audiovisualContent)
        }
        if let type = UTType(filenameExtension: url.pathExtension) {
            return type.conforms(to: .audio) || type.conforms(to: .audiovisualContent)
        }
        return false
    }
}
