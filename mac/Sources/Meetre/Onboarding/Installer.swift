import Foundation
import Observation

/// What the user picked on the install card.
struct InstallOptions: Sendable {
    var startAtLogin = true
    var askForCalls = true
    var recognizeSpeakers = false
    var huggingFaceToken = ""
    var downloadModels = true
}

/// Sets meetre up from the onboarding screen.
///
/// Fresh Mac: download the source tarball into `~/meetre`, run
/// `install.sh` (progress via `::meetre-step` lines), remember the
/// location. Existing install: skip straight to configuring. Then start the
/// engine, apply the options and (optionally) download the models.
@MainActor
@Observable
final class Installer {
    enum Phase: Equatable {
        case idle
        case working
        case done
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    /// One short line for the current step ("Installing the engine").
    private(set) var label = ""
    /// 0…1 across the whole install, or nil while indeterminate.
    private(set) var progress: Double?
    /// Steps completed so far — onboarding plays a note for each.
    private(set) var stepsDone = 0

    private let tarball = URL(string: "https://github.com/maxlkatze/meetre/archive/refs/heads/main.tar.gz")!

    static var isInstalled: Bool { EngineClient.isInstalled(at: EngineClient.repoRoot) }

    func run(_ options: InstallOptions, model: AppModel) async {
        phase = .working
        progress = 0
        do {
            if !Self.isInstalled {
                let root = try await fetchSource()
                try await runInstallScript(in: root, options: options)
                UserDefaults.standard.set(root.path, forKey: EngineClient.repoRootDefaultsKey)
            }
            advance("Waking up the engine", to: 0.72)
            model.start()  // no-op if already running
            try await waitForEngine(model)

            advance("Applying your choices", to: 0.76)
            model.set("detect_calls", .bool(options.askForCalls))
            model.set("person_detection", .bool(options.recognizeSpeakers))
            let token = options.huggingFaceToken.trimmingCharacters(in: .whitespacesAndNewlines)
            if !token.isEmpty { model.set("hf_token", .string(token)) }
            model.setStartAtLogin(options.startAtLogin)

            if options.downloadModels {
                try await downloadModels(model)
            }
            advance("Ready", to: 1)
            phase = .done
        } catch {
            phase = .failed((error as? LocalizedError)?.errorDescription ?? "\(error)")
        }
    }

    // MARK: Steps

    private func advance(_ text: String, to value: Double?) {
        label = text
        progress = value
        stepsDone += 1
    }

    /// Downloads and unpacks the source into ~/meetre (or ~/meetre-app if a
    /// different ~/meetre already exists). Not ~/Documents etc.: the login
    /// item can't read TCC-protected folders.
    private func fetchSource() async throws -> URL {
        advance("Downloading meetre", to: 0.02)
        let home = FileManager.default.homeDirectoryForCurrentUser
        var target = home.appendingPathComponent("meetre")
        if FileManager.default.fileExists(atPath: target.path) {
            target = home.appendingPathComponent("meetre-app")
        }
        let (bytes, response) = try await URLSession.shared.bytes(from: tarball)
        let expected = max(response.expectedContentLength, 1)
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("meetre-src-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let archive = tmp.appendingPathComponent("meetre.tar.gz")
        FileManager.default.createFile(atPath: archive.path, contents: nil)
        let out = try FileHandle(forWritingTo: archive)
        var chunk = Data()
        var received: Int64 = 0
        for try await byte in bytes {
            chunk.append(byte)
            if chunk.count >= 64 * 1024 {
                try out.write(contentsOf: chunk)
                received += Int64(chunk.count)
                chunk.removeAll(keepingCapacity: true)
                if response.expectedContentLength > 0 {
                    progress = 0.02 + 0.08 * Double(received) / Double(expected)
                }
            }
        }
        try out.write(contentsOf: chunk)
        try out.close()

        try await Self.run("/usr/bin/tar", ["-xzf", archive.path, "-C", tmp.path])
        let unpacked = tmp.appendingPathComponent("meetre-main")
        try FileManager.default.moveItem(at: unpacked, to: target)
        try? FileManager.default.removeItem(at: tmp)
        return target
    }

    private func runInstallScript(in root: URL, options: InstallOptions) async throws {
        advance("Preparing", to: 0.1)
        var args = [root.appendingPathComponent("install.sh").path]
        if options.recognizeSpeakers { args.append("--persons") }
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"
        env["MEETRE_HF_TOKEN"] = options.huggingFaceToken
        env["MEETRE_NO_LAUNCH"] = "1"  // we're the app; don't build/open another
        try await Self.run("/bin/bash", args, cwd: root, env: env) { [weak self] line in
            // "::meetre-step 3/6 Creating a home for meetre"
            guard line.hasPrefix("::meetre-step ") else { return }
            let rest = line.dropFirst("::meetre-step ".count)
            let parts = rest.split(separator: " ", maxSplits: 1)
            let nums = parts.first?.split(separator: "/").compactMap { Double($0) } ?? []
            Task { @MainActor in
                guard let self, nums.count == 2 else { return }
                self.advance(parts.count > 1 ? String(parts[1]) : "", to: 0.1 + 0.6 * (nums[0] - 1) / nums[1])
            }
        }
    }

    private func waitForEngine(_ model: AppModel) async throws {
        for _ in 0..<300 where model.connection != .ready {
            if case .failed(let why) = model.connection, !why.isEmpty {
                try await Task.sleep(for: .milliseconds(200))  // it retries; give it a moment
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        guard model.connection == .ready else {
            throw EngineError(message: "The engine didn't start.", hint: "Open Settings ▸ Advanced ▸ Restart engine.")
        }
    }

    /// Transcription model first (small, needed for every meeting), then the
    /// summary model. Progress comes from the engine's `state` events.
    private func downloadModels(_ model: AppModel) async throws {
        let asr = model.config["model"]?.string ?? "parakeet-tdt-v3"
        let summary = model.catalog.defaultSummaryModel ?? model.config["summary_model"]?.string ?? "k2-horizon-7b"
        for (i, (kind, id, label)) in [("asr", asr, "Downloading the ears"),
                                       ("summary", summary, "Downloading the brain")].enumerated() {
            advance(label, to: 0.78 + 0.1 * Double(i))
            let watcher = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    if let p = model.state.progress { self?.progress = 0.78 + 0.1 * Double(i) + 0.1 * p }
                    try? await Task.sleep(for: .milliseconds(200))
                }
            }
            defer { watcher.cancel() }
            _ = try await model.engine.call("models.ensure", ["kind": .string(kind), "id": .string(id)])
        }
    }

    // MARK: Process helper

    private static func run(_ exe: String, _ args: [String], cwd: URL? = nil,
                            env: [String: String]? = nil,
                            onLine: (@Sendable (String) -> Void)? = nil) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let p = Process()
            p.executableURL = URL(fileURLWithPath: exe)
            p.arguments = args
            if let cwd { p.currentDirectoryURL = cwd }
            if let env { p.environment = env }
            let pipe = Pipe()
            p.standardOutput = pipe
            p.standardError = pipe
            let tail = TailBuffer()
            pipe.fileHandleForReading.readabilityHandler = { h in
                let data = h.availableData
                guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
                for line in text.split(separator: "\n") {
                    tail.append(String(line))
                    onLine?(String(line))
                }
            }
            p.terminationHandler = { proc in
                pipe.fileHandleForReading.readabilityHandler = nil
                if proc.terminationStatus == 0 {
                    cont.resume()
                } else {
                    cont.resume(throwing: EngineError(
                        message: "Setup stopped (\(URL(fileURLWithPath: exe).lastPathComponent) exited \(proc.terminationStatus)).",
                        hint: tail.last))
                }
            }
            do { try p.run() } catch { cont.resume(throwing: error) }
        }
    }
}

/// Last meaningful output line, for error messages.
private final class TailBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var line: String?
    func append(_ s: String) {
        let t = s.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return }
        lock.lock(); line = t; lock.unlock()
    }
    var last: String? { lock.lock(); defer { lock.unlock() }; return line }
}
