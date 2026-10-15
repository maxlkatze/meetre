import AppKit
import Observation
import SwiftUI
@preconcurrency import UserNotifications

/// The single source of truth for every view. Views never talk to the engine
/// directly (docs/DESIGN.md §12).
@Observable
@MainActor
final class AppModel {
    enum Connection: Equatable {
        case connecting
        case ready
        case failed(String)
    }

    // Engine
    private(set) var connection: Connection = .connecting
    private(set) var engineVersion: String?
    var state = EngineState()
    var banner: String? {
        didSet { if let banner, banner != oldValue { ErrorLog.shared.record(banner) } }
    }

    // Library
    var meetings: [MeetingSummary] = []
    var selectedID: String? {
        didSet { if selectedID != oldValue { loadSelected() } }
    }
    private(set) var meeting: Meeting?
    private(set) var loadingMeeting = false
    var query = "" { didSet { if query != oldValue { refreshMeetings() } } }

    // Settings
    var config: [String: JSONValue] = [:]
    var catalog = Catalog()
    var startAtLogin = false

    // Live
    var captions: [CaptionLine] = []
    var tentative = ""
    var captionStatus = ""
    var summaryStream = SummaryStream()
    var callPrompt: CallPrompt?

    // Chat
    var chat: [ChatMessage] = []
    var chatDraft = ""
    var chatStreaming: String?
    private var chatMeetingID: String?

    // Record options (popover)
    var pendingTitle = ""

    /// Action-item check state per meeting (local until storage v2 syncs it).
    private var done: [String: Set<Int>] = [:]

    let engine: EngineClient
    private let fixtures: Bool

    init(engine: EngineClient = EngineClient(), fixtures: Bool = false) {
        self.engine = engine
        self.fixtures = fixtures
        if let saved = UserDefaults.standard.dictionary(forKey: "doneTasks") as? [String: [Int]] {
            done = saved.mapValues(Set.init)
        }
        if fixtures { Fixtures.apply(to: self) }
    }

    // MARK: Lifecycle

    /// When processing last finished (drives the ✓ flash in the menu bar).
    var finishedAt: Date?
    /// The last pipeline run reported a failure (⚠ until the popover opens).
    var pipelineFailed = false
    /// Furthest pipeline step reached in the current run (steps can repeat
    /// their wording; the checklist never goes backwards).
    var reachedStep: PipelineStep?

    /// Whether `start()` ran (onboarding may start the engine itself).
    private(set) var started = false

    func start() {
        guard !fixtures, !started else { return }
        started = true
        Task { [weak self] in
            guard let self else { return }
            for await event in engine.events {
                handle(event)
            }
        }
        engine.start()
    }

    private func handle(_ event: EngineEvent) {
        switch event {
        case .connected:
            connection = .connecting
            Task { await handshake() }
        case .disconnected(let why):
            connection = .failed(why)
            ErrorLog.shared.record("Engine stopped: \(why)", source: "engine")
        case .notification(let method, let params):
            handleNotification(method, params)
        }
    }

    private func handshake() async {
        do {
            let hello = try await engine.call("hello", ["client_version": .string(Self.version)])
            engineVersion = hello["engine_version"]?.string
            connection = .ready
            banner = nil
            async let s: EngineState = engine.call("state.get", as: EngineState.self)
            async let c: JSONValue = engine.call("config.get")
            async let k: Catalog = engine.call("catalog.get", as: Catalog.self)
            state = try await s
            config = try await c.object ?? [:]
            catalog = (try? await k) ?? Catalog()
            startAtLogin = config["start_at_login"]?.bool ?? false
            refreshMeetings()
        } catch {
            connection = .failed(error.localizedDescription)
        }
    }

    static var version: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev"
    }

    // MARK: Notifications from the engine

    private func handleNotification(_ method: String, _ p: JSONValue) {
        switch method {
        case "state":
            do {
                let s = try p.decode(EngineState.self)
                let was = state.state
                state = s
                if s.state == .recording && was != .recording {
                    captions = []; tentative = ""
                    pipelineFailed = false; reachedStep = nil; finishedAt = nil
                }
                if s.state == .processing {
                    let step = PipelineStep.from(stage: s.stage, label: s.progressLabel)
                    reachedStep = max(step, reachedStep ?? step)
                }
                if s.state == .idle && was == .processing {
                    let at = Date.now
                    finishedAt = at
                    reachedStep = nil
                    // The "Done ✓" flash: cleared here, not by a timer in the
                    // menu bar label (that re-render loop pinned the CPU).
                    Task { [weak self] in
                        try? await Task.sleep(for: .seconds(5))
                        if self?.finishedAt == at { self?.finishedAt = nil }
                    }
                }
            } catch {
                // Never silently freeze the UI on a format mismatch.
                FileHandle.standardError.write(Data("meetre: undecodable state event: \(error)\n".utf8))
            }
        case "live.caption":
            captions = (p["lines"]?.array ?? []).compactMap { line in
                guard let pair = line.array, pair.count >= 2 else { return nil }
                return CaptionLine(speaker: pair[0].string, text: pair[1].string ?? "")
            }
            tentative = p["tentative"]?.string ?? ""
            captionStatus = p["status"]?.string ?? ""
        case "summary.stream":
            let phase = p["phase"]?.string ?? "writing"
            summaryStream = SummaryStream(meetingID: p["meeting_id"]?.string, phase: phase,
                                          text: p["text"]?.string ?? summaryStream.text,
                                          stage: p["stage"]?.string ?? "", active: phase != "done")
        case "chat.delta":
            if chatStreaming != nil { chatStreaming = p["text"]?.string ?? chatStreaming }
        case "meeting.updated":
            if let s = try? p["summary"]?.decode(MeetingSummary.self) {
                upsert(s)
                if selectedID == s.id || selectedID == nil { selectedID = s.id; loadSelected() }
            } else {
                refreshMeetings()
            }
        case "call":
            switch p["event"]?.string {
            case "started" where state.state == .idle: callPrompt = .started(.now)
            case "ended" where state.state == .recording: callPrompt = .ended
            default: break
            }
        case "notify":
            let subtitle = p["subtitle"]?.string ?? ""
            if subtitle.localizedCaseInsensitiveContains("failed") {
                pipelineFailed = true
                // Also in the popover (with "Copy log"), not only as a notification.
                banner = [subtitle, p["body"]?.string].compactMap { $0 }.filter { !$0.isEmpty }
                    .joined(separator: " — ")
            }
            post(title: p["title"]?.string ?? "meetre", subtitle: p["subtitle"]?.string, body: p["body"]?.string)
        case "error":
            banner = [p["message"]?.string, p["hint"]?.string].compactMap { $0 }.joined(separator: " — ")
        default:
            break
        }
    }

    private func upsert(_ s: MeetingSummary) {
        if let i = meetings.firstIndex(where: { $0.id == s.id }) {
            meetings[i] = s
        } else {
            meetings.insert(s, at: 0)
        }
    }

    // MARK: Recording

    func toggleRecording() {
        switch state.state {
        case .idle: startRecording()
        case .recording: stopRecording()
        case .processing: break
        }
    }

    func startRecording(title: String? = nil) {
        let t = (title ?? pendingTitle).trimmingCharacters(in: .whitespaces)
        callPrompt = nil
        withAnimation(Theme.bouncy) { state.state = .recording; state.elapsed = 0 }
        perform("record.start", ["title": .string(t)]) { [weak self] in self?.state.state = .idle }
        pendingTitle = ""
    }

    func stopRecording() {
        callPrompt = nil
        withAnimation(Theme.bouncy) { state.state = .processing; state.stage = "Finishing recording…" }
        perform("record.stop") { [weak self] in self?.resyncState() }
    }

    /// Replace optimistic UI state with the engine's truth (after a failed
    /// request, the UI must never keep showing a state the engine isn't in).
    func resyncState() {
        Task {
            if let s = try? await engine.call("state.get", as: EngineState.self) { state = s }
        }
    }

    func cancelRecording() {
        perform("record.cancel")
    }

    // MARK: Library

    func refreshMeetings() {
        guard !fixtures else { return }
        Task {
            do {
                meetings = try await engine.call("meetings.list", query.isEmpty ? [:] : ["query": .string(query)],
                                                 as: [MeetingSummary].self)
                if selectedID == nil { selectedID = meetings.first?.id }
            } catch {
                banner = error.localizedDescription
            }
        }
    }

    private func loadSelected() {
        guard let id = selectedID else { meeting = nil; return }
        if chatMeetingID != id { chat = []; chatStreaming = nil; chatMeetingID = id }
        guard !fixtures else { meeting = Fixtures.meeting(id: id); return }
        loadingMeeting = meeting?.id != id
        Task {
            defer { loadingMeeting = false }
            do {
                let m = try await engine.call("meetings.get", ["id": .string(id)], as: Meeting.self)
                if selectedID == id { meeting = m }
            } catch {
                banner = error.localizedDescription
            }
        }
    }

    func summarizeAgain() {
        guard let id = selectedID else { return }
        summaryStream = SummaryStream(meetingID: id, active: true)
        perform("meetings.summarize", ["id": .string(id)])
    }

    func rename(_ id: String, to title: String) {
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        Task {
            do {
                let s = try await engine.call("meetings.rename", ["id": .string(id), "title": .string(t)],
                                              as: MeetingSummary.self)
                if let i = meetings.firstIndex(where: { $0.id == id }) { meetings[i] = s }
                if selectedID == id { selectedID = s.id; loadSelected() }
            } catch { banner = error.localizedDescription }
        }
    }

    func delete(_ id: String) {
        Task {
            do {
                try await engine.call("meetings.delete", ["id": .string(id)])
                meetings.removeAll { $0.id == id }
                if selectedID == id { selectedID = meetings.first?.id }
            } catch { banner = error.localizedDescription }
        }
    }

    func reveal(_ id: String) { perform("meetings.reveal", ["id": .string(id)]) }
    func openNote(_ id: String) { perform("meetings.open_note", ["id": .string(id)]) }

    func copySummary() {
        guard let md = meeting?.summaryMd, !md.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("# \(meeting?.title ?? "")\n\n\(md)", forType: .string)
    }

    // MARK: Action items

    func isDone(_ index: Int) -> Bool {
        guard let id = meeting?.id else { return false }
        return done[id]?.contains(index) ?? false
    }

    func toggleDone(_ index: Int) {
        guard let id = meeting?.id else { return }
        var set = done[id] ?? []
        if set.contains(index) { set.remove(index) } else { set.insert(index) }
        done[id] = set
        UserDefaults.standard.set(done.mapValues { Array($0) }, forKey: "doneTasks")
    }

    // MARK: Chat

    func sendChat() {
        let text = chatDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, chatStreaming == nil else { return }
        let history = chat.map { JSONValue.object(["role": .string($0.role.rawValue), "content": .string($0.text)]) }
        chat.append(ChatMessage(role: .user, text: text))
        chatDraft = ""
        chatStreaming = ""
        let meetingID = selectedID
        Task {
            do {
                var params: [String: JSONValue] = ["history": .array(history), "message": .string(text)]
                if let meetingID { params["meeting_id"] = .string(meetingID) }
                let r = try await engine.call("chat.send", params)
                chat.append(ChatMessage(role: .assistant, text: r["text"]?.string ?? chatStreaming ?? ""))
            } catch {
                chat.append(ChatMessage(role: .assistant, text: "⚠︎ \(error.localizedDescription)"))
            }
            chatStreaming = nil
        }
    }

    func stopChat() { perform("chat.stop") }

    func newChat() { chat = []; chatStreaming = nil }

    // MARK: Config

    func bool(_ key: String, default value: Bool = false) -> Binding<Bool> {
        Binding(get: { self.config[key]?.bool ?? value },
                set: { self.set(key, .bool($0)) })
    }

    func string(_ key: String, default value: String = "") -> Binding<String> {
        Binding(get: { self.config[key]?.string ?? value },
                set: { self.set(key, .string($0)) })
    }

    /// Optional string (e.g. language: nil = auto-detect).
    func optionalString(_ key: String) -> Binding<String?> {
        Binding(get: { self.config[key]?.string },
                set: { self.set(key, $0.map(JSONValue.string) ?? .null) })
    }

    /// Optional int (e.g. num_speakers: nil = auto).
    func optionalInt(_ key: String) -> Binding<Int?> {
        Binding(get: { self.config[key]?.int },
                set: { self.set(key, $0.map { .number(Double($0)) } ?? .null) })
    }

    func set(_ key: String, _ value: JSONValue) {
        config[key] = value
        guard !fixtures else { return }
        Task {
            do {
                config = try await engine.call("config.set", [key: value]).object ?? config
            } catch { banner = error.localizedDescription }
        }
    }

    func setStartAtLogin(_ on: Bool) {
        startAtLogin = on
        perform("autostart.set", ["enabled": .bool(on)])
    }

    func ensureModel(kind: String, id: String) {
        perform("models.ensure", ["kind": .string(kind), "id": .string(id)])
    }

    func restartEngine() {
        connection = .connecting
        engine.restart()
    }

    // MARK: Helpers

    private func perform(_ method: String, _ params: [String: JSONValue] = [:],
                         onError: (@MainActor () -> Void)? = nil) {
        guard !fixtures else { return }
        Task {
            do {
                try await engine.call(method, params)
            } catch {
                banner = error.localizedDescription
                onError?()
            }
        }
    }

    private func post(title: String, subtitle: String?, body: String?) {
        // UNUserNotificationCenter traps outside an app bundle (`swift run`).
        guard Bundle.main.bundleIdentifier != nil else { return }
        let content = UNMutableNotificationContent()
        content.title = title
        if let subtitle { content.subtitle = subtitle }
        if let body { content.body = body }
        let center = UNUserNotificationCenter.current()
        center.requestAuthorization(options: [.alert, .sound]) { granted, _ in
            guard granted else { return }
            center.add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil))
        }
    }

    // MARK: Derived

    var recentMeetings: [MeetingSummary] { Array(meetings.prefix(3)) }

    var localSpeakerNames: Set<String> {
        var s: Set<String> = ["Me", "Ich"]
        if let mine = config["my_name"]?.string, !mine.isEmpty { s.insert(mine) }
        return s
    }

    var statusLine: String {
        switch connection {
        case .connecting: return "Starting engine…"
        case .failed: return "Engine unavailable"
        case .ready: break
        }
        switch state.state {
        case .idle: return "Ready · on-device"
        case .recording:
            return captionStatus.isEmpty || captionStatus == "listening"
                ? "Recording" : "Recording · \(Formatting.stage(captionStatus))"
        case .processing:
            let stage = Formatting.stage(state.progressLabel ?? state.stage)
            if let p = state.progress { return "\(stage.isEmpty ? "Working" : stage) · \(Int(p * 100))%" }
            return stage.isEmpty ? "Working…" : stage
        }
    }
}
