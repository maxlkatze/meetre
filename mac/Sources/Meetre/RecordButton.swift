import SwiftUI

/// The hero control (DESIGN.md §9): one click toggles recording.
struct RecordButton: View {
    @Environment(AppModel.self) private var model
    var size: CGFloat = Theme.Layout.recordButton

    var body: some View {
        let s = model.state.state
        Button(action: model.toggleRecording) {
            ZStack {
                switch s {
                case .idle:
                    Image(systemName: "record.circle")
                        .font(.system(size: size * 0.5, weight: .regular))
                        .foregroundStyle(Theme.record)
                        .symbolRenderingMode(.hierarchical)
                case .recording:
                    RoundedRectangle(cornerRadius: Theme.Radius.chip, style: .continuous)
                        .fill(Theme.record)
                        .frame(width: size * 0.32, height: size * 0.32)
                case .processing:
                    ProgressView().controlSize(.small)
                }
            }
            .frame(width: size, height: size)
            .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .glassEffect(.regular.interactive(), in: Circle())
        .disabled(s == .processing || model.connection != .ready)
        .animation(Theme.bouncy, value: s)
        .help(s == .recording ? "Stop recording" : "Record")
        .accessibilityLabel(s == .recording ? "Stop recording" : "Record")
        .keyboardShortcut("r", modifiers: .command)
    }
}

/// Pulsing red dot shown while recording (stops under Reduce Motion).
struct RecordingDot: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var dim = false
    var size: CGFloat = 8

    var body: some View {
        Circle()
            .fill(Theme.record)
            .frame(width: size, height: size)
            .opacity(dim ? 0.35 : 1)
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(Theme.pulse) { dim = true }
            }
            .accessibilityLabel("Recording")
    }
}

/// Thin determinate/indeterminate progress (DESIGN.md §9 status line).
struct ThinProgress: View {
    var value: Double?

    var body: some View {
        Group {
            if let value {
                ProgressView(value: min(max(value, 0), 1))
            } else {
                ProgressView()
            }
        }
        .progressViewStyle(.linear)
        .tint(Theme.accent)
        .controlSize(.mini)
        .frame(height: Theme.Layout.progressHeight)
    }
}

#Preview {
    RecordButton()
        .padding(Theme.Space.xl)
        .environment(AppModel(fixtures: true))
}
