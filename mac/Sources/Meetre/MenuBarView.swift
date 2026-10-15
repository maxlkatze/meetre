import SwiftUI

/// Menu bar popover (DESIGN.md §10): header · hero · quick options · recent · footer.
struct MenuBarView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.horizontal, Theme.Space.l)
                .padding(.top, Theme.Space.m)
                .padding(.bottom, Theme.Space.s)

            hero
                .padding(.horizontal, Theme.Space.l)
                .padding(.vertical, Theme.Space.m)

            if model.state.state == .processing || model.badge == .done {
                PipelineStepsView()
                    .padding(.horizontal, Theme.Space.l)
                    .padding(.bottom, Theme.Space.m)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            if let banner = model.banner {
                Banner(text: banner) { model.banner = nil }
                    .padding(.horizontal, Theme.Space.l)
                    .padding(.bottom, Theme.Space.s)
            }

            if model.state.state == .idle {
                Divider().padding(.horizontal, Theme.Space.l)
                QuickOptions()
                    .padding(.horizontal, Theme.Space.l)
                    .padding(.vertical, Theme.Space.m)
            }

            if !model.recentMeetings.isEmpty {
                Divider().padding(.horizontal, Theme.Space.l)
                recent
                    .padding(.horizontal, Theme.Space.s)
                    .padding(.vertical, Theme.Space.s)
            }

            Divider().padding(.horizontal, Theme.Space.l)
            footer
                .padding(.horizontal, Theme.Space.s)
                .padding(.vertical, Theme.Space.s)
        }
        .frame(width: Theme.Layout.popoverWidth)
        .onAppear { model.pipelineFailed = false }  // seen — clear the ⚠
        .frame(maxHeight: Theme.Layout.popoverMaxHeight)
        .tint(Theme.accent)
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: Theme.Space.s) {
            Text("meetre")
                .font(.headline)
                .foregroundStyle(.secondary)
            EngineDot()
            Spacer()
            Button {
                NSApp.activate()
                openSettings()
            } label: {
                Image(systemName: "gearshape")
            }
            .buttonStyle(.borderless)
            .help("Settings")
            .accessibilityLabel("Settings")
        }
    }

    // MARK: Hero

    private var hero: some View {
        VStack(alignment: .leading, spacing: Theme.Space.m) {
            HStack(spacing: Theme.Space.m) {
                RecordButton()
                VStack(alignment: .leading, spacing: Theme.Space.xs) {
                    heroTitle
                    HStack(spacing: Theme.Space.xs) {
                        if model.state.state == .recording { RecordingDot(size: 6) }
                        Text(model.statusLine)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .contentTransition(.opacity)
                    }
                }
                Spacer(minLength: 0)
            }
            if model.state.state == .processing {
                ThinProgress(value: model.state.progress)
            }
            if model.state.state == .recording {
                // Always two lines tall while recording: captions arriving,
                // growing or wrapping must not resize (jiggle) the popover.
                Label {
                    Text(lastCaption ?? "Listening…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(2, reservesSpace: true)
                        .truncationMode(.head)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } icon: {
                    Image(systemName: "captions.bubble")
                        .foregroundStyle(Theme.live)
                }
                .transition(.opacity)
            }
            if model.state.state == .idle {
                TextField("Meeting title (optional)", text: Bindable(model).pendingTitle)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { model.startRecording() }
                    .help("Leave empty and meetre names the meeting from its summary.")
            }
        }
        .animation(Theme.smooth, value: model.state.state)
    }

    @ViewBuilder private var heroTitle: some View {
        switch model.state.state {
        case .recording:
            Text(Formatting.timer(model.state.elapsed))
                .font(.system(.title3, design: .rounded).monospacedDigit().weight(.semibold))
                .foregroundStyle(Theme.record)
                .contentTransition(.numericText())
        case .processing:
            Text(model.state.title?.isEmpty == false ? model.state.title! : "Processing")
                .font(.title3.weight(.semibold))
                .lineLimit(1)
        case .idle:
            Text("Record a meeting")
                .font(.title3.weight(.semibold))
        }
    }

    private var lastCaption: String? {
        if !model.tentative.isEmpty { return model.tentative }
        return model.captions.last?.text
    }

    // MARK: Recent

    private var recent: some View {
        VStack(alignment: .leading, spacing: Theme.Space.xxs) {
            Text("Recent")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, Theme.Space.s)
                .padding(.bottom, Theme.Space.xxs)
            ForEach(model.recentMeetings) { m in
                MenuRow {
                    model.selectedID = m.id
                    openLibrary()
                } label: {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(m.title).lineLimit(1)
                        Text("\(Formatting.time(m.date)) · \(Formatting.duration(m.duration))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer(minLength: Theme.Space.s)
                    if m.hasSummary == true {
                        Image(systemName: "text.badge.star")
                            .foregroundStyle(.secondary)
                            .help("Summary ready")
                    }
                }
            }
        }
    }

    // MARK: Footer

    private var footer: some View {
        VStack(spacing: Theme.Space.xxs) {
            MenuRow(action: openLibrary) {
                Label("Open meetre", systemImage: "rectangle.stack")
                Spacer()
                Text("⌘O").foregroundStyle(.tertiary)
            }
            .keyboardShortcut("o", modifiers: .command)
            MenuRow { NSApp.terminate(nil) } label: {
                Label("Quit meetre", systemImage: "power")
                Spacer()
                Text("⌘Q").foregroundStyle(.tertiary)
            }
            .keyboardShortcut("q", modifiers: .command)
        }
    }

    private func openLibrary() {
        openWindow(id: "library")
        NSApp.activate()
    }
}

/// Small engine status dot: green when ready, amber otherwise.
struct EngineDot: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let (color, help): (Color, String) = switch model.connection {
        case .ready: (Theme.live, "Engine ready · on-device")
        case .connecting: (Color.secondary, "Starting engine…")
        case .failed(let why): (Theme.warn, why)
        }
        Circle()
            .fill(color)
            .frame(width: 6, height: 6)
            .help(help)
            .accessibilityLabel(help)
    }
}

/// Full-width row with menu-like hover highlight.
struct MenuRow<Label: View>: View {
    var action: () -> Void
    @ViewBuilder var label: Label
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: Theme.Space.s) { label }
                .padding(.horizontal, Theme.Space.s)
                .padding(.vertical, Theme.Space.xs + 1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.chip, style: .continuous)
                        .fill(hover ? Color.primary.opacity(0.08) : .clear)
                )
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
    }
}

/// Compact pickers for the next recording (idle only).
struct QuickOptions: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: Theme.Space.s) {
            languageMenu
            speakersMenu
            Spacer(minLength: 0)
            Toggle(isOn: model.bool("capture_system", default: true)) {
                Image(systemName: "speaker.wave.2")
            }
            .toggleStyle(.button)
            .help("Record system audio (other participants)")
            .accessibilityLabel("System audio")
            Toggle(isOn: model.bool("live_transcript", default: true)) {
                Image(systemName: "captions.bubble")
            }
            .toggleStyle(.button)
            .help("Live transcript overlay")
            .accessibilityLabel("Live transcript")
        }
        .controlSize(.small)
    }

    private var languageMenu: some View {
        let current = model.config["language"]?.string
        let label = model.catalog.languages.first { $0.code == current }?.label
            ?? (current?.uppercased() ?? "Auto")
        return Menu {
            Picker("Language", selection: model.optionalString("language")) {
                ForEach(model.catalog.languages) { l in
                    Text(l.label).tag(l.code)
                }
            }
            .pickerStyle(.inline)
        } label: {
            Label(label, systemImage: "globe")
        }
        .menuStyle(.button)
        .fixedSize()
        .help("Meeting language")
    }

    private var speakersMenu: some View {
        let n = model.config["num_speakers"]?.int
        return Menu {
            Picker("Speakers", selection: model.optionalInt("num_speakers")) {
                Text("Auto-detect").tag(Int?.none)
                ForEach(1...8, id: \.self) { i in Text("\(i)").tag(Int?.some(i)) }
            }
            .pickerStyle(.inline)
            Divider()
            Toggle("Detect speakers", isOn: model.bool("person_detection"))
        } label: {
            Label(n.map { "\($0)" } ?? "Auto", systemImage: "person.2")
        }
        .menuStyle(.button)
        .fixedSize()
        .help("Speakers")
    }
}

/// Inline problem banner (DESIGN.md §9 Errors).
struct Banner: View {
    var text: String
    var actionTitle: String = "Dismiss"
    var action: () -> Void
    @Environment(AppModel.self) private var model
    @State private var copied = false

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Theme.warn)
            Text(text)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Button(copied ? "Copied ✓" : "Copy log") {
                ErrorLog.shared.copyReport(model: model)
                copied = true
                Task { try? await Task.sleep(for: .seconds(2)); copied = false }
            }
            .buttonStyle(.borderless)
            .controlSize(.small)
            .help("Copy a diagnostic report (no transcripts) to paste into a bug report")
            Button(actionTitle, action: action)
                .buttonStyle(.borderless)
                .controlSize(.small)
        }
        .padding(Theme.Space.s)
        .background(Theme.warn.opacity(0.12), in: RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous))
    }
}

/// Menu bar label: the sparkle, or a red dot + timer while recording.
#Preview {
    MenuBarView()
        .environment(AppModel(fixtures: true))
}
