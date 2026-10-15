import SwiftUI

/// Chat with the local model about one meeting (DESIGN.md §10 Library ▸ Chat).
struct ChatView: View {
    let meeting: Meeting
    @Environment(AppModel.self) private var model
    @FocusState private var focused: Bool

    private let suggestions = [
        "What did we decide?",
        "List all action items with owners.",
        "Draft a follow-up email.",
        "What is still open?",
    ]

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: Theme.Space.l) {
                        if model.chat.isEmpty && model.chatStreaming == nil {
                            emptyState
                        }
                        ForEach(model.chat) { m in
                            Bubble(role: m.role, text: m.text)
                        }
                        if let streaming = model.chatStreaming {
                            Bubble(role: .assistant, text: streaming, streaming: true)
                                .id("streaming")
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .frame(maxWidth: Theme.Layout.readingWidth, alignment: .leading)
                    .padding(Theme.Space.xl)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: model.chatStreaming) { proxy.scrollTo("bottom", anchor: .bottom) }
                .onChange(of: model.chat.count) { withAnimation(Theme.smooth) { proxy.scrollTo("bottom", anchor: .bottom) } }
            }
            composer
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: Theme.Space.m) {
            Label("Ask about this meeting", systemImage: "bubble.left.and.text.bubble.right")
                .font(.headline)
                .symbolRenderingMode(.hierarchical)
            Text("Answers come from the local model and this transcript — nothing leaves your Mac.")
                .foregroundStyle(.secondary)
            FlowChips(items: suggestions) { s in
                model.chatDraft = s
                model.sendChat()
            }
        }
        .padding(.top, Theme.Space.l)
    }

    private var composer: some View {
        @Bindable var model = model
        return HStack(alignment: .bottom, spacing: Theme.Space.s) {
            TextField("Ask about “\(meeting.title)”", text: $model.chatDraft, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .focused($focused)
                .onSubmit(model.sendChat)
                .padding(.horizontal, Theme.Space.m)
                .padding(.vertical, Theme.Space.s)
                .background(.background, in: RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                    .strokeBorder(.separator))
            if model.chatStreaming != nil {
                Button(action: model.stopChat) {
                    Image(systemName: "stop.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.glass)
                .help("Stop")
            } else {
                Button(action: model.sendChat) {
                    Image(systemName: "arrow.up")
                        .fontWeight(.semibold)
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.glassProminent)
                .disabled(model.chatDraft.trimmingCharacters(in: .whitespaces).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
                .help("Send (⌘↩)")
            }
        }
        .padding(.horizontal, Theme.Space.xl)
        .padding(.vertical, Theme.Space.m)
        .frame(maxWidth: Theme.Layout.readingWidth + 2 * Theme.Space.xl)
        .frame(maxWidth: .infinity)
        .background(.bar)
        .onAppear { focused = true }
    }
}

private struct Bubble: View {
    let role: ChatMessage.Role
    let text: String
    var streaming = false

    var body: some View {
        if role == .user {
            HStack {
                Spacer(minLength: Theme.Space.xxl * 2)
                Text(text)
                    .textSelection(.enabled)
                    .padding(.horizontal, Theme.Space.m)
                    .padding(.vertical, Theme.Space.s)
                    .background(Theme.accent.opacity(0.15),
                                in: RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
            }
        } else {
            VStack(alignment: .leading, spacing: Theme.Space.xs) {
                if text.isEmpty && streaming {
                    HStack(spacing: Theme.Space.s) {
                        ProgressView().controlSize(.small)
                        Text("Thinking…").foregroundStyle(.secondary)
                    }
                } else {
                    MarkdownText(text)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Wrapping row of suggestion chips.
struct FlowChips: View {
    let items: [String]
    let tap: (String) -> Void

    var body: some View {
        FlowLayout(spacing: Theme.Space.s) {
            ForEach(items, id: \.self) { s in
                Button(s) { tap(s) }
                    .buttonStyle(.glass)
                    .controlSize(.regular)
            }
        }
    }
}

struct FlowLayout: Layout {
    var spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0, maxX: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x > 0 && x + s.width > width { x = 0; y += rowH + spacing; rowH = 0 }
            x += s.width + spacing
            rowH = max(rowH, s.height)
            maxX = max(maxX, x - spacing)
        }
        return CGSize(width: maxX, height: y + rowH)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowH: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x > bounds.minX && x + s.width > bounds.maxX { x = bounds.minX; y += rowH + spacing; rowH = 0 }
            v.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(s))
            x += s.width + spacing
            rowH = max(rowH, s.height)
        }
    }
}

#Preview {
    ChatView(meeting: Fixtures.meeting(id: "prio"))
        .environment(AppModel(fixtures: true))
        .frame(width: 700, height: 500)
}
