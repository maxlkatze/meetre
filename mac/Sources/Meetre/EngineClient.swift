import Foundation

/// Notifications from the engine plus connection lifecycle.
enum EngineEvent: Sendable {
    case connected
    case disconnected(String)
    case notification(method: String, params: JSONValue)
}

struct EngineError: Error, LocalizedError, Sendable {
    var message: String
    var hint: String?
    var errorDescription: String? { hint.map { "\(message) — \($0)" } ?? message }
}

/// Runs `python -m meetre engine` as a child process and speaks newline-
/// delimited JSON-RPC 2.0 over its stdin/stdout (docs/ROADMAP.md).
///
/// Launching the engine from the app makes macOS attribute microphone and
/// screen-recording access to Meetre.app. The process is restarted with
/// exponential backoff if it exits unexpectedly.
final class EngineClient: @unchecked Sendable {
    let events: AsyncStream<EngineEvent>
    private let eventSink: AsyncStream<EngineEvent>.Continuation

    private let lock = NSLock()
    private var process: Process?
    private var stdin: FileHandle?
    private var buffer = Data()
    private var nextID = 1
    private var pending: [Int: CheckedContinuation<JSONValue, Error>] = [:]
    private var stopping = false
    private var backoff: TimeInterval = 1

    init() {
        (events, eventSink) = AsyncStream.makeStream(of: EngineEvent.self, bufferingPolicy: .unbounded)
    }

    // MARK: Locating the engine

    /// Where onboarding remembers the install it created.
    static let repoRootDefaultsKey = "MeetreRepoRoot"

    /// The meetre install (repo + .venv) to run the engine from: env override,
    /// then the install onboarding created, then the build-time location, then
    /// the usual `~/meetre` / `~/meetre-main`.
    static var repoRoot: URL {
        if let env = ProcessInfo.processInfo.environment["MEETRE_REPO"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        var candidates: [URL] = []
        if let saved = UserDefaults.standard.string(forKey: repoRootDefaultsKey), !saved.isEmpty {
            candidates.append(URL(fileURLWithPath: saved))
        }
        if let plist = Bundle.main.object(forInfoDictionaryKey: "MeetreRepoRoot") as? String, !plist.isEmpty {
            candidates.append(URL(fileURLWithPath: plist))
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        candidates += ["meetre", "meetre-main", "meetre-app"].map { home.appendingPathComponent($0) }
        if let found = candidates.first(where: { isInstalled(at: $0) }) { return found }
        // `swift run` from mac/: the repo is the parent directory.
        return candidates.first
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath).deletingLastPathComponent()
    }

    /// A usable install: the venv interpreter and the meetre sources exist.
    static func isInstalled(at root: URL) -> Bool {
        let fm = FileManager.default
        return fm.isExecutableFile(atPath: root.appendingPathComponent(".venv/bin/python").path)
            && fm.fileExists(atPath: root.appendingPathComponent("src/meetre/engine.py").path)
    }

    static var usesFakeEngine: Bool {
        ProcessInfo.processInfo.environment["MEETRE_FAKE_ENGINE"] == "1"
            || CommandLine.arguments.contains("--fake-engine")
    }

    private func command() -> (URL, [String], [String: String]) {
        let root = Self.repoRoot
        let venvPython = root.appendingPathComponent(".venv/bin/python")
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"
        if Self.usesFakeEngine {
            let python = FileManager.default.isExecutableFile(atPath: venvPython.path)
                ? venvPython : URL(fileURLWithPath: "/usr/bin/python3")
            let script = root.appendingPathComponent("mac/Scripts/fake_engine.py").path
            return (python, ["-I", script], env)
        }
        let site = (try? FileManager.default.contentsOfDirectory(
            at: root.appendingPathComponent(".venv/lib"), includingPropertiesForKeys: nil))?
            .filter { $0.lastPathComponent.hasPrefix("python") }
            .map { $0.appendingPathComponent("site-packages").path } ?? []
        env["PYTHONPATH"] = (site + [root.appendingPathComponent("src").path]).joined(separator: ":")
        // Lets the engine point "start at login" at this app instead of the
        // legacy Python menu bar.
        if Bundle.main.bundlePath.hasSuffix(".app") {
            env["MEETRE_APP"] = Bundle.main.bundlePath
        }
        return (venvPython, ["-m", "meetre", "engine"], env)
    }

    // MARK: Lifecycle

    func start() {
        lock.lock(); stopping = false; lock.unlock()
        launch()
    }

    func stop() {
        lock.lock()
        stopping = true
        let p = process
        lock.unlock()
        if let p, p.isRunning {
            send(["jsonrpc": .string("2.0"), "method": .string("shutdown"), "id": .number(0)])
            DispatchQueue.global().asyncAfter(deadline: .now() + 1.5) { if p.isRunning { p.terminate() } }
        }
    }

    /// Kill and relaunch (Settings ▸ Advanced ▸ Restart engine).
    func restart() {
        lock.lock()
        let p = process
        backoff = 1
        lock.unlock()
        if let p, p.isRunning { p.terminate() } else { launch() }
    }

    private func launch() {
        let (exe, args, env) = command()
        let p = Process()
        p.executableURL = exe
        p.arguments = args
        p.environment = env
        p.currentDirectoryURL = Self.repoRoot
        let inPipe = Pipe(), outPipe = Pipe(), errPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            if data.isEmpty { h.readabilityHandler = nil; return }
            self?.ingest(data)
        }
        errPipe.fileHandleForReading.readabilityHandler = { h in
            let data = h.availableData
            if data.isEmpty { h.readabilityHandler = nil; return }
            Self.log(data)
        }
        p.terminationHandler = { [weak self] proc in
            self?.terminated(status: proc.terminationStatus)
        }

        do {
            try p.run()
        } catch {
            eventSink.yield(.disconnected("Could not start the engine: \(error.localizedDescription)"))
            scheduleRestart()
            return
        }
        lock.lock()
        process = p
        stdin = inPipe.fileHandleForWriting
        buffer.removeAll()
        lock.unlock()
        eventSink.yield(.connected)
    }

    private func terminated(status: Int32) {
        lock.lock()
        let waiting = pending
        pending.removeAll()
        process = nil
        stdin = nil
        let wasStopping = stopping
        lock.unlock()
        for (_, c) in waiting {
            c.resume(throwing: EngineError(message: "The engine stopped", hint: "It restarts automatically."))
        }
        guard !wasStopping else { return }
        eventSink.yield(.disconnected("The engine exited (status \(status))."))
        scheduleRestart()
    }

    private func scheduleRestart() {
        lock.lock()
        let delay = backoff
        backoff = min(backoff * 2, 30)
        let wasStopping = stopping
        lock.unlock()
        guard !wasStopping else { return }
        DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            self.lock.lock(); let s = self.stopping; self.lock.unlock()
            if !s { self.launch() }
        }
    }

    // MARK: RPC

    @discardableResult
    func call(_ method: String, _ params: [String: JSONValue] = [:]) async throws -> JSONValue {
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<JSONValue, Error>) in
            lock.lock()
            guard stdin != nil else {
                lock.unlock()
                c.resume(throwing: EngineError(message: "The engine isn't running", hint: "Retrying…"))
                return
            }
            let id = nextID
            nextID += 1
            pending[id] = c
            lock.unlock()
            send(["jsonrpc": .string("2.0"), "id": .number(Double(id)),
                  "method": .string(method), "params": .object(params)])
        }
    }

    func call<T: Decodable>(_ method: String, _ params: [String: JSONValue] = [:], as type: T.Type) async throws -> T {
        try await call(method, params).decode(T.self)
    }

    private func send(_ message: [String: JSONValue]) {
        guard var data = try? JSONEncoder().encode(JSONValue.object(message)) else { return }
        data.append(0x0A)
        lock.lock()
        let h = stdin
        lock.unlock()
        try? h?.write(contentsOf: data)
    }

    private func ingest(_ data: Data) {
        lock.lock()
        buffer.append(data)
        var lines: [Data] = []
        while let nl = buffer.firstIndex(of: 0x0A) {
            lines.append(buffer.subdata(in: buffer.startIndex..<nl))
            buffer.removeSubrange(buffer.startIndex...nl)
        }
        lock.unlock()
        for line in lines where !line.isEmpty {
            handle(line)
        }
    }

    private func handle(_ line: Data) {
        guard let msg = try? JSONDecoder().decode(JSONValue.self, from: line), let obj = msg.object else {
            Self.log(line)  // stray print() from a dependency
            return
        }
        if let id = obj["id"]?.int, obj["method"] == nil {
            lock.lock()
            backoff = 1
            let c = pending.removeValue(forKey: id)
            lock.unlock()
            if let err = obj["error"], !err.isNull {
                c?.resume(throwing: EngineError(message: err["message"]?.string ?? "Engine error",
                                                hint: err["data"]?["hint"]?.string))
            } else {
                c?.resume(returning: obj["result"] ?? .null)
            }
        } else if let method = obj["method"]?.string {
            eventSink.yield(.notification(method: method, params: obj["params"] ?? .null))
        }
    }

    /// Engine stderr (tracebacks, PortAudio errors …) goes to
    /// ~/Library/Logs/meetre/engine.log — visible in Console.app — and to our
    /// own stderr when run from a terminal. Rotated at 5 MB.
    static let logURL: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/meetre", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("engine.log")
    }()

    private static let logLock = NSLock()
    nonisolated(unsafe) private static var logHandle: FileHandle?

    private static func log(_ data: Data) {
        FileHandle.standardError.write(data)
        logLock.lock(); defer { logLock.unlock() }
        if logHandle == nil {
            let fm = FileManager.default
            if let size = (try? fm.attributesOfItem(atPath: logURL.path))?[.size] as? Int, size > 5_000_000 {
                try? fm.removeItem(at: logURL.appendingPathExtension("1"))
                try? fm.moveItem(at: logURL, to: logURL.appendingPathExtension("1"))
            }
            if !fm.fileExists(atPath: logURL.path) { fm.createFile(atPath: logURL.path, contents: nil) }
            logHandle = try? FileHandle(forWritingTo: logURL)
            _ = try? logHandle?.seekToEnd()
            let stamp = ISO8601DateFormatter().string(from: Date())
            logHandle?.write(Data("\n=== engine \(stamp) ===\n".utf8))
        }
        logHandle?.write(data)
    }
}
