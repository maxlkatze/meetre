import SwiftUI

/// Library window (DESIGN.md §10): grouped sidebar + meeting detail.
struct LibraryView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 240, ideal: 280, max: 320)
        } detail: {
            if let meeting = model.meeting {
                MeetingDetail(meeting: meeting)
                    .id(meeting.id)
            } else if model.loadingMeeting {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                EmptyState(symbol: "waveform", text: model.meetings.isEmpty ? "No meetings yet." : "Select a meeting.",
                           actionTitle: model.meetings.isEmpty ? "Record" : nil, action: model.toggleRecording)
            }
        }
        .searchable(text: $model.query, placement: .sidebar, prompt: "Search meetings")
        .toolbar(removing: .title)
        .toolbar {
            ToolbarItem(placement: .navigation) {
                RecordToolbarButton()
            }
        }
        .frame(minWidth: 780, minHeight: 480)
        .tint(Theme.accent)
        .onAppear { NSApp.setActivationPolicy(.regular); NSApp.activate() }
        .onDisappear { NSApp.setActivationPolicy(.accessory) }
    }
}

// MARK: - Sidebar

private struct Sidebar: View {
    @Environment(AppModel.self) private var model
    @State private var pendingDelete: MeetingSummary?
    @AppStorage("showBackups") private var showBackups = true

    var body: some View {
        @Bindable var model = model
        let groups = Dictionary(grouping: model.meetings) { Formatting.group(for: $0.date) }
        List(selection: $model.selectedID) {
            if let banner = model.banner {
                Banner(text: banner) { model.banner = nil }
                    .listRowSeparator(.hidden)
            }
            ForEach(groups.keys.sorted(), id: \.self) { g in
                Section(g.title) {
                    ForEach(groups[g] ?? []) { m in
                        MeetingRow(meeting: m).tag(m.id)
                            .contextMenu {
                                Button("Reveal in Finder") { model.reveal(m.id) }
                                Button("Open in Notes") { model.openNote(m.id) }
                                Divider()
                                // Destructive: always confirm (DESIGN.md §9 Errors).
                                Button("Move to Trash…", role: .destructive) { pendingDelete = m }
                            }
                    }
                }
            }
            if !model.backups.isEmpty && model.query.isEmpty {
                Section("Not transcribed", isExpanded: $showBackups) {
                    ForEach(model.backups) { b in
                        BackupRow(backup: b)
                            .contextMenu {
                                Button("Transcribe") { model.importAudio(b.path) }
                                    .disabled(!model.canImport)
                                Button("Reveal in Finder") { model.revealFile(b.path) }
                            }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .confirmationDialog("Move “\(pendingDelete?.title ?? "")” to the Trash?",
                            isPresented: Binding(get: { pendingDelete != nil },
                                                 set: { if !$0 { pendingDelete = nil } }),
                            presenting: pendingDelete) { m in
            Button("Move to Trash", role: .destructive) { model.delete(m.id) }
        } message: { _ in
            Text("You can restore it from the Trash in Finder.")
        }
        .overlay {
            if model.meetings.isEmpty && !model.query.isEmpty {
                ContentUnavailableView.search(text: model.query)
            }
        }
    }
}

private struct MeetingRow: View {
    let meeting: MeetingSummary
    @Environment(AppModel.self) private var model

    var body: some View {
        let isProcessing = model.state.state == .processing && model.state.title == meeting.title
        HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s) {
            VStack(alignment: .leading, spacing: Theme.Space.xxs) {
                Text(meeting.title)
                    .font(.body)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Text("\(Formatting.time(meeting.date)) · \(Formatting.duration(meeting.duration))")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            if isProcessing {
                ProgressView().controlSize(.mini)
            } else if meeting.hasSummary == true {
                Image(systemName: "text.badge.star")
                    .foregroundStyle(.secondary)
                    .help("Summary ready")
            }
        }
        .padding(.vertical, Theme.Space.xxs)
    }
}

/// An audio backup without a transcript: "Transcribe" turns it into a meeting.
private struct BackupRow: View {
    let backup: AudioBackup
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(alignment: .center, spacing: Theme.Space.s) {
            VStack(alignment: .leading, spacing: Theme.Space.xxs) {
                Text(Formatting.shortDayAndTime(backup.date))
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Text(Formatting.size(backup.size))
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
            }
            Spacer(minLength: 0)
            Button("Transcribe") { model.importAudio(backup.path) }
                .buttonStyle(.borderless)
                .controlSize(.small)
                .disabled(!model.canImport)
                .help("Transcribe this recording into a meeting")
        }
        .padding(.vertical, Theme.Space.xxs)
    }
}

private struct RecordToolbarButton: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let s = model.state.state
        Button(action: model.toggleRecording) {
            switch s {
            case .recording:
                Label {
                    Text("Stop · \(Formatting.timer(model.state.elapsed))").monospacedDigit()
                } icon: {
                    Image(systemName: "stop.fill").foregroundStyle(Theme.record)
                }
            case .processing:
                Label(Formatting.stage(model.state.stage).isEmpty ? "Processing" : Formatting.stage(model.state.stage),
                      systemImage: "hourglass")
            case .idle:
                Label {
                    Text("Record")
                } icon: {
                    Image(systemName: "record.circle").foregroundStyle(Theme.record)
                }
            }
        }
        .labelStyle(.titleAndIcon)
        .disabled(s == .processing || model.connection != .ready)
        .help(s == .recording ? "Stop recording (⌘R)" : "Record (⌘R)")
        .keyboardShortcut("r", modifiers: .command)
    }
}

// MARK: - Detail

enum DetailTab: String, CaseIterable, Identifiable {
    case summary = "Summary", transcript = "Transcript", chat = "Chat"
    var id: Self { self }
    var symbol: String {
        switch self {
        case .summary: "text.badge.star"
        case .transcript: "text.quote"
        case .chat: "bubble.left.and.text.bubble.right"
        }
    }
}

struct MeetingDetail: View {
    let meeting: Meeting
    @Environment(AppModel.self) private var model
    @AppStorage("detailTab") private var tabRaw = DetailTab.summary.rawValue
    @State private var title = ""
    @State private var editingTitle = false
    @FocusState private var titleFocused: Bool

    private var tab: Binding<DetailTab> {
        Binding(get: { DetailTab(rawValue: tabRaw) ?? .summary }, set: { tabRaw = $0.rawValue })
    }
    @State private var confirmDelete = false
    @State private var confirmRetranscribe = false

    var body: some View {
        let owners = meeting.sections?.tasks?.compactMap(\.owner) ?? []
        let voices = meeting.turns?.compactMap(\.speaker) ?? []
        let palette = SpeakerPalette(speakers: (meeting.speakers ?? []) + voices + owners,
                                     localNames: model.localSpeakerNames)
        VStack(spacing: 0) {
            header(palette)
                .padding(.horizontal, Theme.Space.xl)
                .padding(.top, Theme.Space.l)
                .padding(.bottom, Theme.Space.m)
            Divider()
            Group {
                switch tab.wrappedValue {
                case .summary: SummaryTab(meeting: meeting, palette: palette)
                case .transcript: TranscriptTab(meeting: meeting, palette: palette)
                case .chat: ChatView(meeting: meeting)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(.background)
        }
        .navigationTitle(meeting.title)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("View", selection: tab) {
                    ForEach(DetailTab.allCases) { t in
                        Label(t.rawValue, systemImage: t.symbol).tag(t)
                    }
                }
                .pickerStyle(.segmented)
                .labelStyle(.titleOnly)
                .fixedSize()
            }
            ToolbarItemGroup(placement: .primaryAction) {
                Button(action: model.summarizeAgain) {
                    Label("Summarize again", systemImage: "arrow.clockwise")
                }
                .help("Summarize again (⌘⇧S)")
                .keyboardShortcut("s", modifiers: [.command, .shift])
                .disabled(model.summaryStream.active || model.connection != .ready)
                Button { confirmRetranscribe = true } label: {
                    Label("Transcribe again", systemImage: "waveform.badge.magnifyingglass")
                }
                .help("Transcribe again from the audio backup")
                .disabled(model.state.state != .idle || model.connection != .ready)
                Menu {
                    Button("Copy summary as Markdown", action: model.copySummary)
                    Button("Open in Apple Notes") { model.openNote(meeting.id) }
                    Button("Reveal in Finder") { model.reveal(meeting.id) }
                    Divider()
                    Button("Move to Trash…", role: .destructive) { confirmDelete = true }
                } label: {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .help("Share")
            }
        }
        .confirmationDialog("Move “\(meeting.title)” to the Trash?", isPresented: $confirmDelete) {
            Button("Move to Trash", role: .destructive) { model.delete(meeting.id) }
        }
        .confirmationDialog("Transcribe “\(meeting.title)” again?", isPresented: $confirmRetranscribe) {
            Button("Transcribe again", action: model.transcribeAgain)
        } message: {
            Text("The transcript and summary are replaced with a fresh pass over the audio backup.")
        }
        .onAppear { title = meeting.title }
    }

    private func commitTitle() {
        editingTitle = false
        if title != meeting.title { model.rename(meeting.id, to: title) }
    }

    private func header(_ palette: SpeakerPalette) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.s) {
            if editingTitle {
                TextField("Title", text: $title)
                    .textFieldStyle(.plain)
                    .font(.title2.weight(.semibold))
                    .focused($titleFocused)
                    .onSubmit(commitTitle)
                    .onExitCommand { title = meeting.title; editingTitle = false }
                    .onChange(of: titleFocused) { if !titleFocused { commitTitle() } }
            } else {
                Text(meeting.title)
                    .font(.title2.weight(.semibold))
                    .textSelection(.disabled)
                    .onTapGesture(count: 2) { editingTitle = true; titleFocused = true }
                    .help("Double-click to rename")
            }
            Text(Formatting.meta(date: meeting.date, duration: meeting.duration,
                                 speakers: meeting.speakers?.count ?? 0))
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if let speakers = meeting.speakers, !speakers.isEmpty {
                HStack(spacing: Theme.Space.xs) {
                    ForEach(speakers, id: \.self) { s in
                        SpeakerChip(name: s, color: palette.color(for: s))
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct SpeakerChip: View {
    var name: String
    var color: Color

    var body: some View {
        Text(name)
            .font(.caption.weight(.medium))
            .foregroundStyle(color == .secondary ? Color.primary.opacity(0.75) : color)
            .padding(.horizontal, Theme.Space.s)
            .padding(.vertical, Theme.Space.xxs)
            .background(color.opacity(0.15), in: Capsule())
    }
}

// MARK: - Summary tab

private struct SummaryTab: View {
    let meeting: Meeting
    let palette: SpeakerPalette
    @Environment(AppModel.self) private var model

    var body: some View {
        let stream = model.summaryStream
        let streaming = stream.active && stream.meetingID == meeting.id
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.xl) {
                if streaming {
                    StreamingSummary(stream: stream)
                } else if let s = meeting.sections, hasContent(s) {
                    if let text = s.summary, !text.isEmpty {
                        Text(text)
                            .font(.body)
                            .lineSpacing(2)
                            .textSelection(.enabled)
                    }
                    section("Decisions", symbol: "checkmark.seal", items: s.decisions ?? [])
                    tasks(s.tasks ?? [])
                    section("Open questions", symbol: "questionmark.circle", items: s.questions ?? [])
                } else if let md = meeting.summaryMd, !md.isEmpty {
                    MarkdownText(md)
                } else {
                    EmptyState(symbol: "text.badge.star", text: "No summary yet.",
                               actionTitle: "Summarize", action: model.summarizeAgain)
                        .padding(.top, Theme.Space.xxl)
                }
            }
            .frame(maxWidth: Theme.Layout.readingWidth, alignment: .leading)
            .padding(Theme.Space.xl)
            .frame(maxWidth: .infinity)
        }
    }

    private func hasContent(_ s: MeetingSections) -> Bool {
        !(s.summary ?? "").isEmpty || !(s.decisions ?? []).isEmpty || !(s.tasks ?? []).isEmpty
    }

    @ViewBuilder
    private func section(_ title: String, symbol: String, items: [String]) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.s) {
            Label(title, systemImage: symbol)
                .font(.headline)
                .symbolRenderingMode(.hierarchical)
            if items.isEmpty {
                Text("None").foregroundStyle(.tertiary)
            } else {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s) {
                        Text("•").foregroundStyle(.tertiary)
                        Text(item).lineSpacing(2).textSelection(.enabled)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func tasks(_ tasks: [MeetingTask]) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.s) {
            HStack {
                Label("Action items", systemImage: "checkmark.circle")
                    .font(.headline)
                    .symbolRenderingMode(.hierarchical)
                if !tasks.isEmpty {
                    let open = tasks.indices.filter { !model.isDone($0) }.count
                    Text(open == 0 ? "All done" : "\(open) open")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            if tasks.isEmpty {
                Text("None").foregroundStyle(.tertiary)
            } else {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(tasks.enumerated()), id: \.offset) { i, t in
                        TaskRow(task: t, done: model.isDone(i),
                                ownerColor: palette.color(for: t.owner)) { model.toggleDone(i) }
                        if i < tasks.count - 1 { Divider().padding(.leading, 28) }
                    }
                }
                .padding(.vertical, Theme.Space.xs)
                .padding(.horizontal, Theme.Space.m)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous))
            }
        }
    }
}

private struct TaskRow: View {
    let task: MeetingTask
    let done: Bool
    let ownerColor: Color
    let toggle: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s) {
            Button(action: toggle) {
                Image(systemName: done ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(done ? Theme.accent : .secondary)
                    .font(.body)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(done ? "Mark as open" : "Mark as done")
            VStack(alignment: .leading, spacing: Theme.Space.xxs) {
                Text(task.text)
                    .strikethrough(done, color: .secondary)
                    .foregroundStyle(done ? .secondary : .primary)
                    .textSelection(.enabled)
                if let due = task.due, !due.isEmpty {
                    Text("Due \(due)").font(.subheadline).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: Theme.Space.s)
            if let owner = task.owner, !owner.isEmpty {
                SpeakerChip(name: owner, color: ownerColor)
            }
        }
        .padding(.vertical, Theme.Space.s)
        .animation(Theme.smooth, value: done)
    }
}

/// The summary while K2 writes it (thinking → writing), in the reading column.
private struct StreamingSummary: View {
    let stream: SummaryStream

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.m) {
            HStack(spacing: Theme.Space.s) {
                ProgressView().controlSize(.small)
                Text(stream.phase == "thinking" ? "Thinking…" : "Writing summary…")
                    .foregroundStyle(.secondary)
                if !stream.stage.isEmpty {
                    Text("· \(stream.stage)").foregroundStyle(.tertiary)
                }
            }
            if stream.phase != "thinking" && !stream.text.isEmpty {
                MarkdownText(stream.text)
            }
        }
    }
}

// MARK: - Transcript tab

private struct TranscriptTab: View {
    let meeting: Meeting
    let palette: SpeakerPalette
    @State private var find = ""
    @State private var showFind = false
    @FocusState private var findFocused: Bool

    private struct Group: Identifiable {
        let id: Int
        let speaker: String?
        let start: Double?
        var lines: [String]
    }

    private var groups: [Group] {
        var out: [Group] = []
        for t in meeting.turns ?? [] {
            if let last = out.last, last.speaker == t.speaker {
                out[out.count - 1].lines.append(t.text)
            } else {
                out.append(Group(id: out.count, speaker: t.speaker, start: t.start, lines: [t.text]))
            }
        }
        if find.isEmpty { return out }
        return out.filter { $0.lines.joined(separator: " ").localizedCaseInsensitiveContains(find) }
    }

    var body: some View {
        VStack(spacing: 0) {
            if showFind {
                HStack(spacing: Theme.Space.s) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("Find in transcript", text: $find)
                        .textFieldStyle(.plain)
                        .focused($findFocused)
                    Button("Done") { find = ""; showFind = false }
                        .buttonStyle(.borderless)
                }
                .padding(.horizontal, Theme.Space.xl)
                .padding(.vertical, Theme.Space.s)
                Divider()
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: Theme.Space.l) {
                    if groups.isEmpty {
                        EmptyState(symbol: "text.quote",
                                   text: find.isEmpty ? "No transcript." : "No matches for “\(find)”.")
                            .padding(.top, Theme.Space.xxl)
                    }
                    ForEach(groups) { g in
                        TurnView(speaker: g.speaker, start: g.start, text: g.lines.joined(separator: " "),
                                 color: palette.color(for: g.speaker), highlight: find)
                    }
                }
                .frame(maxWidth: Theme.Layout.readingWidth, alignment: .leading)
                .padding(Theme.Space.xl)
                .frame(maxWidth: .infinity)
            }
        }
        .background {
            Button("") { showFind = true; findFocused = true }
                .keyboardShortcut("f", modifiers: .command)
                .hidden()
        }
    }
}

private struct TurnView: View {
    let speaker: String?
    let start: Double?
    let text: String
    let color: Color
    let highlight: String
    @State private var hover = false

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Space.m) {
            RoundedRectangle(cornerRadius: 1.5)
                .fill(color == .secondary ? Color.secondary.opacity(0.4) : color)
                .frame(width: 3)
            VStack(alignment: .leading, spacing: Theme.Space.xs) {
                HStack(spacing: Theme.Space.s) {
                    Text(speaker ?? "Speaker")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(color == .secondary ? Color.primary.opacity(0.7) : color)
                    Text(Formatting.offset(start))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.tertiary)
                        .opacity(hover ? 1 : 0)
                }
                Text(attributed)
                    .lineSpacing(2)
                    .textSelection(.enabled)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .onHover { hover = $0 }
    }

    private var attributed: AttributedString {
        var a = AttributedString(text)
        guard !highlight.isEmpty else { return a }
        var range = a.startIndex..<a.endIndex
        while let r = a[range].range(of: highlight, options: .caseInsensitive) {
            a[r].backgroundColor = Theme.warn.opacity(0.35)
            range = r.upperBound..<a.endIndex
        }
        return a
    }
}

// MARK: - Shared

struct EmptyState: View {
    var symbol: String
    var text: String
    var actionTitle: String? = nil
    var action: () -> Void = {}

    var body: some View {
        VStack(spacing: Theme.Space.m) {
            Image(systemName: symbol)
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text(text).foregroundStyle(.secondary)
            if let actionTitle {
                Button(actionTitle, action: action)
                    .buttonStyle(.glassProminent)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Renders summary-style markdown: `##` headings, `-` bullets, inline bold.
struct MarkdownText: View {
    let markdown: String
    init(_ markdown: String) { self.markdown = markdown }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.s) {
            ForEach(Array(markdown.components(separatedBy: "\n").enumerated()), id: \.offset) { _, raw in
                let line = raw.trimmingCharacters(in: .whitespaces)
                if line.hasPrefix("#") {
                    Text(inline(line.drop { $0 == "#" }.trimmingCharacters(in: .whitespaces)))
                        .font(.headline)
                        .padding(.top, Theme.Space.s)
                } else if line.hasPrefix("- ") || line.hasPrefix("* ") {
                    HStack(alignment: .firstTextBaseline, spacing: Theme.Space.s) {
                        Text("•").foregroundStyle(.tertiary)
                        Text(inline(String(line.dropFirst(2)))).lineSpacing(2)
                    }
                } else if !line.isEmpty {
                    Text(inline(line)).lineSpacing(2)
                }
            }
        }
        .textSelection(.enabled)
    }

    private func inline(_ s: String) -> AttributedString {
        (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(s)
    }
}

#Preview {
    LibraryView()
        .environment(AppModel(fixtures: true))
        .frame(width: 1100, height: 720)
}
