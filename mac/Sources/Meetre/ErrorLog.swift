import AppKit
import Foundation
import Observation

/// Collects what went wrong and turns it into one copyable report.
///
/// Sources: the app's own errors (banners, engine crashes, failed recordings
/// — also appended to `~/Library/Logs/meetre/app.log`), the error-relevant
/// part of the engine's stderr log (`engine.log`, tracebacks and PortAudio
/// errors), and the newest crash logs the engine writes to `<repo>/crashlogs`.
/// The report never includes transcripts or summaries.
@MainActor
@Observable
final class ErrorLog {
    static let shared = ErrorLog()

    struct Entry: Identifiable, Hashable {
        let id = UUID()
        let date: Date
        let source: String
        let message: String
    }

    /// Newest last; capped so a flapping engine can't grow it forever.
    private(set) var entries: [Entry] = []

    static var folder: URL { EngineClient.logURL.deletingLastPathComponent() }
    private static var appLog: URL { folder.appendingPathComponent("app.log") }

    func record(_ message: String, source: String = "app") {
        let text = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        // The same message twice in a row (a retrying engine) is one entry.
        if let last = entries.last, last.message == text, Date.now.timeIntervalSince(last.date) < 30 { return }
        let entry = Entry(date: .now, source: source, message: text)
        entries.append(entry)
        if entries.count > 200 { entries.removeFirst(entries.count - 200) }
        let line = "\(Self.stamp(entry.date)) [\(source)] \(text)\n"
        if let h = try? FileHandle(forWritingTo: Self.appLog) {
            _ = try? h.seekToEnd()
            h.write(Data(line.utf8))
            try? h.close()
        } else {
            try? Data(line.utf8).write(to: Self.appLog)
        }
    }

    /// Copies `report(model:)` to the clipboard.
    func copyReport(model: AppModel) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(report(model: model), forType: .string)
    }

    // MARK: Report

    func report(model: AppModel) -> String {
        var out = "## meetre error log\n\n"
        out += "- App: \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev")"
        out += " · engine \(model.engineVersion ?? "–") · \(Self.connection(model.connection))\n"
        out += "- macOS \(ProcessInfo.processInfo.operatingSystemVersionString) · \(Self.chip()) · "
        out += "\(ProcessInfo.processInfo.physicalMemory / 1_073_741_824) GB\n"
        out += "- Models: \(model.config["model"]?.string ?? "?") / \(model.config["summary_model"]?.string ?? "?")"
        out += " · language \(model.config["language"]?.string ?? "auto")\n"
        out += "- State: \(model.state.state.rawValue)\(model.state.stage.map { " · \($0)" } ?? "")\n"
        out += "- Install: \(EngineClient.repoRoot.path)\n"
        out += "- Generated: \(Self.stamp(.now))\n"

        out += "\n### App errors\n\n"
        let recent = entries.suffix(40)
        out += recent.isEmpty ? "(none this session)\n"
            : recent.map { "\(Self.stamp($0.date)) [\($0.source)] \($0.message)" }.joined(separator: "\n") + "\n"

        out += "\n### Engine log (errors)\n\n```\n\(Self.engineErrors())\n```\n"

        for crash in Self.newestCrashLogs(limit: 2) {
            let text = (try? String(contentsOf: crash, encoding: .utf8)) ?? ""
            out += "\n### \(crash.lastPathComponent)\n\n```\n\(String(text.suffix(6000)))\n```\n"
        }
        return out
    }

    /// Lines from the engine log that matter: tracebacks (with their
    /// context), errors and warnings — from the last few sessions only.
    private static func engineErrors() -> String {
        guard let data = try? Data(contentsOf: EngineClient.logURL),
              let text = String(data: data.suffix(400_000), encoding: .utf8) else { return "(no engine log yet)" }
        let noise = ["Warning: You are sending unauthenticated requests", "[transformers] You are using a model of type"]
        let keys = ["Traceback", "Error", "error", "Exception", "failed", "Failed", "PaMacCore", "WARNING", "=== engine"]
        var keep: [String] = []
        var inTraceback = false
        for line in text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init) {
            if noise.contains(where: line.contains) { continue }
            if line.hasPrefix("Traceback") { inTraceback = true }
            if inTraceback || keys.contains(where: line.contains) { keep.append(line) }
            // A traceback ends with its exception line (no leading space).
            if inTraceback, !line.hasPrefix(" "), !line.hasPrefix("Traceback") { inTraceback = false }
        }
        let tail = keep.suffix(200)
        return tail.isEmpty ? "(no errors logged)" : tail.joined(separator: "\n")
    }

    private static func newestCrashLogs(limit: Int) -> [URL] {
        let dirs = [EngineClient.repoRoot.appendingPathComponent("crashlogs"),
                    FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".config/meetre/crashlogs")]
        let files = dirs.flatMap {
            (try? FileManager.default.contentsOfDirectory(at: $0, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        }.filter { $0.lastPathComponent.hasPrefix("crash-") }
        let dated = files.map { ($0, (try? $0.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast) }
        return dated.sorted { $0.1 > $1.1 }.prefix(limit).map(\.0)
    }

    private static func connection(_ c: AppModel.Connection) -> String {
        switch c {
        case .ready: "engine ready"
        case .connecting: "engine starting"
        case .failed(let why): "engine failed: \(why)"
        }
    }

    private static func chip() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    static func stamp(_ d: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return f.string(from: d)
    }
}
