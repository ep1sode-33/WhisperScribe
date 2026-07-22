import SwiftUI

struct StatusView: View {
    @EnvironmentObject var viewModel: TranscriptionViewModel

    var body: some View {
        VStack(spacing: 16) {
            switch viewModel.state {
            case .idle:
                EmptyView()

            case .loadingModel:
                indeterminate(String(localized: "status.loadingModel"))

            case .extractingAudio:
                indeterminate(String(localized: "status.extractingAudio"))

            case .transcribing(let p):
                transcribing(String(localized: "status.transcribing"), fraction: p)

            case .recognizing(let p):
                transcribing(String(localized: "status.recognizing"), fraction: p)

            case .cleaning(let p, let note):
                cleaning(String(localized: "status.cleaning"), fraction: p, note: note)

            case .merging(let p, let note):
                cleaning(String(localized: "status.merging"), fraction: p, note: note)

            case .done(let outputs, let warnings):
                doneView(outputs: outputs, warnings: warnings)

            case .error(let e):
                errorView(e)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    // MARK: - Subviews

    private func indeterminate(_ label: String) -> some View {
        VStack(spacing: 14) {
            ProgressView()
                .controlSize(.large)
            Text(label)
                .foregroundStyle(.secondary)
        }
    }

    /// Transcription phase. WhisperKit drives a Foundation `Progress` object that climbs
    /// monotonically to 1.0 (VAD: per-chunk; single window: per-sample), which we surface
    /// here as a determinate linear bar plus a percentage. Cancellable.
    private func transcribing(_ label: String, fraction: Double) -> some View {
        let f = max(0, min(1, fraction))
        return VStack(spacing: 14) {
            Text(label)
                .font(.headline)
            ProgressView(value: f)
                .progressViewStyle(.linear)
                .frame(maxWidth: 320)
            Text("\(Int(f * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            Button("common.cancel", role: .cancel) {
                viewModel.cancel()
            }
            .padding(.top, 4)
        }
    }

    /// LLM cleanup phase. Advances per batch/chunk, surfaced as a determinate linear
    /// bar plus a Chinese status note ("已完成 X/Y 批/块"). Cancellable.
    private func cleaning(_ label: String, fraction: Double, note: String) -> some View {
        let f = max(0, min(1, fraction))
        return VStack(spacing: 14) {
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)          // keeps spinning during a slow batch → clearly alive
                Text(label)
                    .font(.headline)
            }
            ProgressView(value: f)
                .progressViewStyle(.linear)
                .frame(maxWidth: 320)
            Text(note.isEmpty ? String(localized: "status.preparing") : note)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("\(Int(f * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            Text("status.slowModelHint")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Button("common.cancel", role: .cancel) {
                viewModel.cancel()
            }
            .padding(.top, 4)
        }
    }

    private func doneView(outputs: [URL], warnings: [String]) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 44))
                .foregroundStyle(.green)
            Text("status.done")
                .font(.title3.weight(.semibold))

            VStack(alignment: .leading, spacing: 6) {
                ForEach(outputs, id: \.self) { url in
                    Label(url.lastPathComponent,
                          systemImage: url.pathExtension == "srt" ? "doc.text" : "doc.plaintext")
                }
            }
            .font(.callout)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

            if !warnings.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(warnings.enumerated()), id: \.offset) { _, w in
                        Label(w, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                    }
                }
                .foregroundStyle(.orange)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
            }

            HStack(spacing: 12) {
                Button {
                    viewModel.revealInFinder()
                } label: {
                    Label("common.revealInFinder", systemImage: "folder")
                }
                .buttonStyle(.borderedProminent)

                Button("status.transcribeAnother") {
                    viewModel.reset()
                }
            }
        }
        .frame(maxWidth: 420)
    }

    private func errorView(_ error: AppError) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "xmark.octagon.fill")
                .font(.system(size: 44))
                .foregroundStyle(.red)
            Text(error.localizedDescription)
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(12)
                .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
                .foregroundStyle(.red)

            Button("common.ok") {
                viewModel.reset()
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: 420)
    }
}
