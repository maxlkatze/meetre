import AppKit
import AVFoundation
import CoreGraphics

/// The two privacy permissions meetre needs. Both are granted to Meetre.app;
/// the engine and the system-audio helper it launches inherit them (macOS
/// attributes child processes to their "responsible" app).
@MainActor
enum Permissions {
    enum Status: Equatable { case granted, denied, notAsked }

    static var microphone: Status {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: .granted
        case .denied, .restricted: .denied
        default: .notAsked
        }
    }

    /// System audio is captured via ScreenCaptureKit, which macOS gates behind
    /// Screen Recording. Preflight stays false until meetre restarts after
    /// the user enables it.
    static var screen: Status { CGPreflightScreenCaptureAccess() ? .granted : .notAsked }

    static var allGranted: Bool { microphone == .granted && screen == .granted }

    static func requestMicrophone() async -> Status {
        _ = await AVCaptureDevice.requestAccess(for: .audio)
        return microphone
    }

    /// Shows the system prompt the first time; afterwards macOS only offers
    /// System Settings, so open it right at the pane.
    static func requestScreen() {
        if !CGRequestScreenCaptureAccess() {
            NSWorkspace.shared.open(URL(string:
                "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!)
        }
    }

    static func openMicrophoneSettings() {
        NSWorkspace.shared.open(URL(string:
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!)
    }

    /// Quit and reopen meetre (Screen Recording only applies after a
    /// restart); onboarding resumes at `resumeAt`.
    static func relaunch(resumeAt step: String) {
        UserDefaults.standard.set(step, forKey: OnboardingController.resumeKey)
        let path = Bundle.main.bundlePath
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/sh")
        p.arguments = ["-c", "sleep 1; /usr/bin/open \"$0\"", path]
        try? p.run()
        NSApp.terminate(nil)
    }
}
