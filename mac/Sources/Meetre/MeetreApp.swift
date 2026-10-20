import AppKit
import SwiftUI

@main
struct MeetreApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        MenuBarExtra {
            MenuBarView()
                .environment(delegate.model)
        } label: {
            MenuBarLabel()
                .environment(delegate.model)
                .modifier(OpenWindowCapture())
        }
        .menuBarExtraStyle(.window)

        Window("meetre", id: "library") {
            LibraryView()
                .environment(delegate.model)
        }
        .defaultSize(width: 1100, height: 720)
        .windowToolbarStyle(.unifiedCompact)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Import Audio…", action: delegate.model.chooseAudioToImport)
                    .keyboardShortcut("i", modifiers: [.command, .shift])
                    .disabled(!delegate.model.canImport)
            }
        }

        Settings {
            SettingsView()
                .environment(delegate.model)
        }

        // Debug: the popover in a normal window for screenshots/UI work.
        Window("meetre — popover preview", id: "popover-preview") {
            MenuBarView()
                .environment(delegate.model)
                .background(.regularMaterial)
        }
        .windowResizability(.contentSize)
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model: AppModel
    private var overlays: OverlayController?

    override init() {
        model = AppModel(fixtures: CommandLine.arguments.contains("--fixtures"))
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        if let i = CommandLine.arguments.firstIndex(of: "--render-onboarding"), i + 1 < CommandLine.arguments.count {
            OnboardingSnapshots.render(to: URL(fileURLWithPath: CommandLine.arguments[i + 1]), model: model)
            exit(0)
        }
        // A fresh Mac has no engine yet: onboarding installs it, then starts it.
        if Installer.isInstalled || EngineClient.usesFakeEngine { model.start() }
        let overlays = OverlayController(model: model)
        overlays.demo = CommandLine.arguments.contains("--demo-hud")
        overlays.start()
        self.overlays = overlays
        let debugging = CommandLine.arguments.contains { $0.hasPrefix("--open") || $0.hasPrefix("--preview") || $0.hasPrefix("--demo") }
        if OnboardingController.needed && (!debugging || CommandLine.arguments.contains("--onboarding")) {
            onboarding.show(model: model) { openLibrary in
                if openLibrary {
                    NSApp.activate()
                    OpenWindowBridge.shared.open?("library")
                }
            }
        }
        DebugLaunch.run(model: model)
    }

    private let onboarding = OnboardingController()

    func applicationWillTerminate(_ notification: Notification) {
        model.engine.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}

/// Debug launch arguments for screenshots and UI work:
/// `--open-library`, `--open-settings`, `--preview-popover`, `--demo-record`,
/// `--demo-hud`, `--fixtures`, `--fake-engine`, `--print-windows`.
@MainActor
enum DebugLaunch {
    static func run(model: AppModel) {
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--appearance"), i + 1 < args.count {
            NSApp.appearance = NSAppearance(named: args[i + 1] == "light" ? .aqua : .darkAqua)
        }
        guard args.contains(where: { $0.hasPrefix("--open") || $0.hasPrefix("--preview") || $0.hasPrefix("--demo") || $0 == "--print-windows" }) else { return }
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(800))
            if args.contains("--open-library") { open(id: "library") }
            if args.contains("--preview-popover") { open(id: "popover-preview") }
            if args.contains("--open-settings") {
                NSApp.activate()
                OpenWindowBridge.shared.openSettings?()
            }
            if args.contains("--demo-record") {
                for _ in 0..<50 where model.connection != .ready { try? await Task.sleep(for: .milliseconds(100)) }
                model.startRecording(title: "")
                if let i = args.firstIndex(of: "--demo-stop"), i + 1 < args.count, let secs = Double(args[i + 1]) {
                    try? await Task.sleep(for: .seconds(secs))
                    model.stopRecording()
                }
            }
            if args.contains("--demo-chat") {
                for _ in 0..<50 where model.meeting == nil { try? await Task.sleep(for: .milliseconds(100)) }
                model.chatDraft = "What did we decide?"
                model.sendChat()
            }
            if args.contains("--print-windows") {
                try? await Task.sleep(for: .seconds(2))
                for w in NSApp.windows where w.isVisible {
                    print("WINDOW \(w.windowNumber) \(w.title.isEmpty ? String(describing: type(of: w)) : w.title)")
                }
                fflush(stdout)
            }
        }
    }

    /// Opens a SwiftUI `Window` scene by id without a view context.
    private static func open(id: String) {
        NSApp.activate()
        OpenWindowBridge.shared.open?(id)
    }
}

/// Lets non-view code open SwiftUI windows: a hidden view captures
/// `openWindow` once the menu bar label renders.
@MainActor
final class OpenWindowBridge {
    static let shared = OpenWindowBridge()
    var open: ((String) -> Void)?
    var openSettings: (() -> Void)?
}

struct OpenWindowCapture: ViewModifier {
    @Environment(\.openWindow) private var openWindow
    @Environment(\.openSettings) private var openSettings

    func body(content: Content) -> some View {
        content.onAppear {
            OpenWindowBridge.shared.open = { openWindow(id: $0) }
            OpenWindowBridge.shared.openSettings = { openSettings() }
        }
    }
}
