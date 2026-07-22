import SwiftUI
import UniformTypeIdentifiers

struct DropZone: View {
    var onPick: ([URL]) -> Void

    @State private var isTargeted = false

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "waveform.badge.plus")
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(isTargeted ? Color.accentColor : .secondary)
            Text("drop.title")
                .font(.title3)
                .foregroundStyle(.secondary)
            Text("drop.subtitleMulti")
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

    /// Resolve the file URL from *every* dropped provider, then deliver them in ONE `onPick`
    /// call. Provider order is preserved (a fixed slot array indexed by provider position);
    /// `BatchClassifier` re-sorts into natural order downstream. EVERY resolved file URL is
    /// delivered — unsupported files are NOT filtered here so the ViewModel's classifier can
    /// throw `AppError.unsupportedFile` and surface a visible error (即时报错), rather than the
    /// drop being silently swallowed. Writes are serialised on `lock` because provider
    /// completions fire on arbitrary queues.
    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        let group = DispatchGroup()
        let lock = DispatchQueue(label: "DropZone.collect")
        var slots = [URL?](repeating: nil, count: providers.count)
        for (index, provider) in providers.enumerated() {
            group.enter()
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url {
                    lock.sync { slots[index] = url }
                }
                group.leave()
            }
        }
        group.notify(queue: .main) {
            let urls = lock.sync { slots.compactMap { $0 } }
            guard !urls.isEmpty else { return }
            onPick(urls)
        }
        return true
    }
}
