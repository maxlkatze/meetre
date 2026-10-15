import AppKit
import SwiftUI

/// Hosts `OnboardingView` in a borderless, translucent window that covers the
/// main screen below the menu bar (so the final hint can point at the status
/// item). The desktop shows through a full-screen blur.
@MainActor
final class OnboardingController {
    static let doneKey = "onboardingDone"
    /// Set before a permission relaunch: reopen onboarding at that step.
    static let resumeKey = "onboardingResume"

    private var window: NSWindow?

    /// First launch, a fresh Mac without the engine, or `--onboarding`.
    static var needed: Bool {
        CommandLine.arguments.contains("--onboarding")
            || UserDefaults.standard.string(forKey: resumeKey) != nil
            || !UserDefaults.standard.bool(forKey: doneKey)
            || !Installer.isInstalled
    }

    func show(model: AppModel, onFinish: @escaping (_ openLibrary: Bool) -> Void) {
        guard let screen = NSScreen.main else { return }
        let frame = screen.frame

        let window = KeyableWindow(contentRect: frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .normal
        window.collectionBehavior = [.fullScreenNone, .canJoinAllSpaces]
        window.isReleasedWhenClosed = false
        window.setFrame(frame, display: true)

        let blur = NSVisualEffectView(frame: NSRect(origin: .zero, size: frame.size))
        blur.material = .fullScreenUI
        blur.blendingMode = .behindWindow
        blur.state = .active
        blur.autoresizingMask = [.width, .height]

        // Back from a permission relaunch: skip the intro, continue there.
        let resume = UserDefaults.standard.string(forKey: Self.resumeKey)
        UserDefaults.standard.removeObject(forKey: Self.resumeKey)
        let start: OnboardingView.Step = resume == "permissions" ? .permissions : .hello
        let root = OnboardingView(alreadyInstalled: Installer.isInstalled, start: start,
                                  autoplay: start == .hello) { [weak self] openLibrary in
            UserDefaults.standard.set(true, forKey: Self.doneKey)
            self?.close()
            onFinish(openLibrary)
        }
        .environment(model)
        let host = NSHostingView(rootView: root)
        host.frame = blur.bounds
        host.autoresizingMask = [.width, .height]
        blur.addSubview(host)
        window.contentView = blur

        window.alphaValue = 0
        NSApp.activate()
        window.makeKeyAndOrderFront(nil)
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.6
            window.animator().alphaValue = 1
        }
        self.window = window
    }

    func close() {
        guard let window else { return }
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = 0.5
            window.animator().alphaValue = 0
        }, completionHandler: {
            Task { @MainActor in
                window.orderOut(nil)
                window.close()
            }
        })
        self.window = nil
    }
}

/// Borderless windows can't become key by default; onboarding needs the
/// keyboard (Return to continue, the token field).
private final class KeyableWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

/// `--render-onboarding <dir>`: draws every scene off-screen to PNGs (muted,
/// no window) — for design review without taking over the screen.
@MainActor
enum OnboardingSnapshots {
    static func render(to dir: URL, model: AppModel) {
        Chimes.shared.muted = true
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for step in OnboardingView.Step.allCases {
            let view = OnboardingView(alreadyInstalled: false, start: step, autoplay: false) { _ in }
                .environment(model)
                .frame(width: 1512, height: 945)
                .background(Color(nsColor: NSColor(hex: 0x1B1C22)))
            let renderer = ImageRenderer(content: view)
            renderer.scale = 1
            guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
                  let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else { continue }
            try? png.write(to: dir.appendingPathComponent("\(step.rawValue)-\(step).png"))
        }
    }
}
