import AppKit
import SwiftUI

/// A borderless, non-activating floating panel in the top-right corner that
/// hosts SwiftUI. Click-through everywhere except `interactiveRect` (the header
/// controls): a 15 Hz tracker toggles `ignoresMouseEvents` while the pointer is
/// over them — the same approach as the Python overlay.
@MainActor
final class HUDPanel {
    let panel: NSPanel
    private let host: NSHostingView<AnyView>
    private var shown = false
    var clickThrough: Bool
    /// Rect (in panel content coordinates, origin bottom-left) that takes clicks.
    var interactiveRect: () -> CGRect = { .zero }

    init<Content: View>(height: CGFloat, clickThrough: Bool = true, @ViewBuilder content: () -> Content) {
        self.clickThrough = clickThrough
        host = NSHostingView(rootView: AnyView(content()))
        host.sizingOptions = []
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: Theme.Layout.hudWidth, height: height),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        panel.isReleasedWhenClosed = false
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.level = .statusBar
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle, .fullScreenAuxiliary]
        panel.ignoresMouseEvents = clickThrough
        panel.alphaValue = 0
        panel.contentView = host
        place(height: height, offset: 0)
    }

    /// Pin to the top-right of the main screen's visible frame; `offset` pushes
    /// it down (to stack under another HUD).
    func place(height: CGFloat, offset: CGFloat) {
        guard let screen = NSScreen.main else { return }
        let vf = screen.visibleFrame
        let m = Theme.Layout.hudMargin
        let frame = NSRect(x: vf.maxX - Theme.Layout.hudWidth - m, y: vf.maxY - height - m - offset,
                           width: Theme.Layout.hudWidth, height: height)
        panel.setFrame(frame, display: true)
    }

    /// Resize keeping the top edge pinned.
    func resize(height: CGFloat) {
        var f = panel.frame
        guard abs(f.height - height) > 0.5 else { return }
        let top = f.maxY
        f.size.height = height
        f.origin.y = top - height
        panel.setFrame(f, display: true, animate: false)
    }

    func setVisible(_ visible: Bool) {
        guard visible != shown else { return }
        shown = visible
        if visible {
            panel.orderFrontRegardless()
            NSAnimationContext.runAnimationGroup { $0.duration = 0.35; panel.animator().alphaValue = 1 }
        } else {
            NSAnimationContext.runAnimationGroup({ $0.duration = 0.6; panel.animator().alphaValue = 0 }) {
                MainActor.assumeIsolated { if !self.shown { self.panel.orderOut(nil) } }
            }
        }
    }

    var isVisible: Bool { shown }

    func trackMouse() {
        guard shown, clickThrough else { return }
        let r = interactiveRect()
        let screenRect = panel.convertToScreen(r).insetBy(dx: -4, dy: -4)
        let hover = screenRect.contains(NSEvent.mouseLocation)
        if panel.ignoresMouseEvents == hover { panel.ignoresMouseEvents = !hover }
    }
}

/// Owns the live-transcript HUD, the summary HUD and the call prompt, and
/// shows/hides them from `AppModel` state.
@MainActor
final class OverlayController {
    private let model: AppModel
    private var transcript: HUDPanel!
    private var summary: HUDPanel!
    private var call: HUDPanel!
    private var timer: Timer?
    private var summaryDoneAt: Date?
    private var stoppedAt: Date?
    /// `--demo-hud`: keep the transcript HUD visible regardless of state.
    var demo = false

    init(model: AppModel) {
        self.model = model
        transcript = HUDPanel(height: Theme.Layout.hudHeight) {
            TranscriptHUD().environment(model)
        }
        transcript.interactiveRect = { [weak transcript] in
            guard let t = transcript else { return .zero }
            let h = t.panel.frame.height
            return CGRect(x: Theme.Layout.hudWidth - 44, y: h - 40, width: 40, height: 36)
        }
        summary = HUDPanel(height: Theme.Layout.hudHeight) {
            SummaryHUD().environment(model)
        }
        summary.interactiveRect = { [weak summary] in
            guard let s = summary else { return .zero }
            let h = s.panel.frame.height
            return CGRect(x: Theme.Layout.hudWidth - 44, y: h - 40, width: 40, height: 36)
        }
        call = HUDPanel(height: 112, clickThrough: false) {
            CallPromptView().environment(model)
        }
    }

    func start() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 15, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
        RunLoop.main.add(timer!, forMode: .common)
    }

    private func tick() {
        let cfg = model.config
        let recording = model.state.state == .recording
        if recording { stoppedAt = nil } else if stoppedAt == nil { stoppedAt = .now }

        // Live transcript: while recording, plus 1.5 s after stop.
        // Both HUDs start collapsed to their header strip; the chevron
        // expands them (remembered in config).
        let minimized = !(cfg["live_transcript_expanded"]?.bool ?? false)
        let liveOn = cfg["live_transcript"]?.bool ?? true
        let lingering = stoppedAt.map { Date.now.timeIntervalSince($0) < 1.5 } ?? false
        let showTranscript = demo || (liveOn && (recording || (lingering && !model.captions.isEmpty)))
        transcript.resize(height: minimized ? Theme.Layout.hudMinimized : Theme.Layout.hudHeight)
        transcript.setVisible(showTranscript)
        transcript.trackMouse()

        // Summary: while streaming, held 8 s after done.
        let s = model.summaryStream
        if s.active { summaryDoneAt = nil } else if summaryDoneAt == nil, !s.text.isEmpty { summaryDoneAt = .now }
        let hold = summaryDoneAt.map { Date.now.timeIntervalSince($0) < 8 } ?? false
        let showSummary = (cfg["summary_overlay"]?.bool ?? true) && (s.active || hold) && !showTranscript
        summary.resize(height: (cfg["summary_overlay_expanded"]?.bool ?? false)
                       ? Theme.Layout.hudHeight : Theme.Layout.hudMinimized)
        summary.setVisible(showSummary)
        summary.trackMouse()

        // Call prompt: under any visible HUD; auto-dismiss rules as before.
        if case .started(let at) = model.callPrompt,
           model.state.state != .idle || Date.now.timeIntervalSince(at) > 120 {
            model.callPrompt = nil
        }
        if case .ended(let at) = model.callPrompt {
            // Unanswered: stop by itself once the countdown runs out.
            let limit = cfg["call_end_autostop"]?.double ?? 60
            if model.state.state != .recording {
                model.callPrompt = nil
            } else if limit > 0, Date.now.timeIntervalSince(at) >= limit {
                model.stopRecording()
            }
        }
        let offset: CGFloat = showTranscript ? transcript.panel.frame.height + Theme.Space.s
            : (showSummary ? summary.panel.frame.height + Theme.Space.s : 0)
        if model.callPrompt != nil && !call.isVisible { call.place(height: 112, offset: offset) }
        call.setVisible(model.callPrompt != nil)
    }
}

// MARK: - HUD views (always dark, DESIGN.md §3/§6)

private struct HUDChrome<Content: View>: View {
    var title: String
    var status: String
    var statusColor: Color
    var minimized = false
    var trailing: AnyView? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: Theme.Space.s) {
                Text(title)
                    .font(.system(size: 10, weight: .semibold))
                    .tracking(1.2)
                    .foregroundStyle(.white.opacity(0.62))
                Spacer()
                Text(status)
                    .font(.system(size: 10, weight: .semibold))
                    .tracking(1.2)
                    .foregroundStyle(statusColor)
                if let trailing { trailing }
            }
            .frame(height: 22)
            .padding(.horizontal, Theme.Space.l)
            .padding(.top, Theme.Space.m)
            Rectangle()
                .fill(.white.opacity(0.08))
                .frame(height: 1)
                .padding(.horizontal, Theme.Space.l)
                .padding(.top, Theme.Space.s - 2)
            if !minimized {
                content
                    .padding(.horizontal, Theme.Space.l)
                    .padding(.top, Theme.Space.m)
                    .padding(.bottom, Theme.Space.m)
                    .transition(.opacity)
            }
            Spacer(minLength: 0)
        }
        .frame(width: Theme.Layout.hudWidth)
        .frame(maxHeight: .infinity, alignment: .top)
        .background {
            ZStack {
                VisualEffect(material: .hudWindow)
                Theme.hudTint
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
            .strokeBorder(Theme.hudBorder, lineWidth: 1))
        .environment(\.colorScheme, .dark)
    }
}

private struct TranscriptHUD: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let minimized = !(model.config["live_transcript_expanded"]?.bool ?? false)
        let recording = model.state.state == .recording
        HUDChrome(title: "MEETRE  ·  LIVE TRANSCRIPT", status: statusText,
                  statusColor: recording ? Theme.record : .white.opacity(0.45),
                  minimized: minimized,
                  trailing: AnyView(HUDExpandButton(key: "live_transcript_expanded", what: "live transcript"))) {
            VStack(alignment: .leading, spacing: Theme.Space.s) {
                ScrollViewReader { proxy in
                    ScrollView(showsIndicators: false) {
                        captionText
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id("end")
                    }
                    .onChange(of: model.captions) { proxy.scrollTo("end", anchor: .bottom) }
                    .onChange(of: model.tentative) { proxy.scrollTo("end", anchor: .bottom) }
                }
                Text(footer)
                    .font(.system(size: 9.5))
                    .foregroundStyle(.white.opacity(0.45))
            }
        }
    }

    private var statusText: String {
        switch model.state.state {
        case .recording:
            let st = model.captionStatus
            return st.isEmpty || st == "listening" ? "● LIVE" : st.uppercased()
        default:
            return "■ STOPPED"
        }
    }

    private var footer: String {
        let words = model.captions.reduce(0) { $0 + $1.text.split(separator: " ").count }
        let m = model.config["model"]?.string ?? ""
        return "\(Formatting.timer(model.state.elapsed))  ·  \(words) words  ·  \(m)  ·  on-device"
    }

    private var captionText: some View {
        let lines = model.captions.suffix(40)
        var out = AttributedString()
        for (i, line) in lines.enumerated() {
            if i > 0 { out += AttributedString("\n") }
            if let s = line.speaker, !s.isEmpty {
                var who = AttributedString(s.uppercased() + "  ")
                who.font = .system(size: 9.5, weight: .semibold)
                who.foregroundColor = .white.opacity(0.45)
                who.kern = 1
                out += who
            }
            var said = AttributedString(line.text)
            said.font = .system(size: 13)
            said.foregroundColor = .white.opacity(0.92)
            out += said
        }
        if !model.tentative.isEmpty {
            var t = AttributedString((lines.isEmpty ? "" : "  ") + model.tentative)
            t.font = .system(size: 13)
            t.foregroundColor = .white.opacity(0.55)
            out += t
        }
        if out.characters.isEmpty {
            var t = AttributedString(model.captionStatus.isEmpty || model.captionStatus == "listening"
                                     ? "Listening…" : "Preparing live captions…")
            t.font = .system(size: 12)
            t.foregroundColor = .white.opacity(0.45)
            out = t
        }
        return Text(out).lineSpacing(3)
    }
}

/// Chevron in a HUD header: collapses the HUD to its header strip or expands
/// it. The state lives in config (`key`, default collapsed).
private struct HUDExpandButton: View {
    @Environment(AppModel.self) private var model
    let key: String
    let what: String
    @State private var hover = false

    var body: some View {
        let expanded = model.config[key]?.bool ?? false
        Button {
            model.set(key, .bool(!expanded))
        } label: {
            Image(systemName: expanded ? "chevron.up" : "chevron.down")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(hover ? 0.92 : 0.45))
                .frame(width: 22, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .help(expanded ? "Minimize \(what)" : "Expand \(what)")
        .accessibilityLabel(expanded ? "Minimize \(what)" : "Expand \(what)")
    }
}

private struct SummaryHUD: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let s = model.summaryStream
        HUDChrome(title: "MEETRE  ·  LIVE SUMMARY", status: status(s),
                  statusColor: .white.opacity(s.active ? 0.62 : 0.45),
                  minimized: !(model.config["summary_overlay_expanded"]?.bool ?? false),
                  trailing: AnyView(HUDExpandButton(key: "summary_overlay_expanded", what: "live summary"))) {
            ScrollViewReader { proxy in
                ScrollView(showsIndicators: false) {
                    Group {
                        if s.phase == "thinking" || s.text.isEmpty {
                            Text("Thinking…").foregroundStyle(.white.opacity(0.45))
                        } else {
                            MarkdownText(s.text)
                                .foregroundStyle(.white.opacity(0.92))
                                .font(.system(size: 12.5))
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .id("end")
                }
                .onChange(of: s.text) { proxy.scrollTo("end", anchor: .bottom) }
            }
        }
    }

    private func status(_ s: SummaryStream) -> String {
        var tag = s.active ? (s.phase == "thinking" ? "● THINKING" : "● WRITING") : "✓ DONE"
        if s.active && !s.stage.isEmpty { tag = s.stage.uppercased() + " · " + tag }
        return tag
    }
}

/// "Teams call detected — record?" / "Call ended — stop?" (DESIGN.md §10).
private struct CallPromptView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let endedAt: Date? = if case .ended(let at) = model.callPrompt { at } else { nil }
        let ended = endedAt != nil
        let limit = model.config["call_end_autostop"]?.double ?? 60
        VStack(alignment: .leading, spacing: Theme.Space.m) {
            HStack(spacing: Theme.Space.s) {
                Image(systemName: ended ? "phone.down" : "phone.arrow.up.right")
                    .foregroundStyle(ended ? Theme.record : Theme.live)
                VStack(alignment: .leading, spacing: Theme.Space.xxs) {
                    Text(ended ? "Call ended" : "Teams call detected")
                        .font(.headline)
                    Group {
                        if let endedAt, limit > 0 {
                            TimelineView(.periodic(from: endedAt, by: 1)) { ctx in
                                let left = max(0, Int((limit - ctx.date.timeIntervalSince(endedAt)).rounded(.up)))
                                Text("Stopping and transcribing in \(left) s.")
                                    .monospacedDigit()
                            }
                        } else {
                            Text(ended ? "Stop the recording and transcribe it?" : "Record and transcribe this call?")
                        }
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                }
            }
            HStack {
                Spacer()
                Button(ended ? "Keep recording" : "Not now") { model.callPrompt = nil }
                    .keyboardShortcut(.cancelAction)
                Button(ended ? "Stop" : "Record") {
                    ended ? model.stopRecording() : model.startRecording(title: "")
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.record)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(Theme.Space.l)
        .frame(width: Theme.Layout.hudWidth, height: 112, alignment: .top)
        .background {
            ZStack {
                VisualEffect(material: .hudWindow)
                Theme.hudTint
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
            .strokeBorder(Theme.hudBorder, lineWidth: 1))
        .environment(\.colorScheme, .dark)
        // The prompt never takes key focus (it mustn't steal typing from the
        // call), so render controls as active anyway.
        .environment(\.controlActiveState, .key)
    }
}

/// NSVisualEffectView bridge (SwiftUI has no HUD material).
struct VisualEffect: NSViewRepresentable {
    var material: NSVisualEffectView.Material

    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = material
        v.blendingMode = .behindWindow
        v.state = .active
        return v
    }

    func updateNSView(_ v: NSVisualEffectView, context: Context) { v.material = material }
}

#Preview("Transcript HUD") {
    TranscriptHUD()
        .environment(AppModel(fixtures: true))
        .frame(height: Theme.Layout.hudHeight)
}
