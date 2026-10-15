import AppKit
import SwiftUI

/// Design tokens from docs/DESIGN.md. Every color, spacing, radius and motion
/// value used by the UI comes from here — add new ones to the rule book first.
enum Theme {
    // MARK: Color (§3)

    static let accent = Color(light: 0x5B5BD6, dark: 0x8B8BF5)
    /// Recording state ONLY.
    static let record = Color(light: 0xE5484D, dark: 0xFF6369)
    static let live = Color(light: 0x30A46C, dark: 0x3DD68C)
    static let warn = Color(light: 0xF5A524, dark: 0xFFC53D)
    /// `record` for AppKit (colored menu bar symbols).
    static var recordNS: NSColor { NSColor(record) }

    /// Pipeline step colors (§3): each processing step has its own hue so the
    /// menu bar shows at a glance where a meeting is.
    static func step(_ s: PipelineStep) -> Color {
        switch s {
        case .preparing: Color(light: 0x8B8D98, dark: 0xB0B4BA)   // neutral gray
        case .transcribing: Color(light: 0x0090FF, dark: 0x3B9EFF) // blue
        case .speakers: Color(light: 0xF76B15, dark: 0xFF801F)     // orange
        case .summary: accent                                       // indigo
        case .saving: live                                          // green
        }
    }

    /// Speaker palette in order of first appearance. "Me" is neutral.
    static let speakers: [Color] = [.blue, .orange, .pink, .teal, .purple, .brown, .mint, .indigo]

    /// HUD graphite tint (always dark).
    static let hudTint = Color(red: 0.105, green: 0.105, blue: 0.115).opacity(0.94)
    static let hudBorder = Color.white.opacity(0.14)

    // MARK: Spacing (§5) — 2, 4, 8, 12, 16, 24, 32

    enum Space {
        static let xxs: CGFloat = 2
        static let xs: CGFloat = 4
        static let s: CGFloat = 8
        static let m: CGFloat = 12
        static let l: CGFloat = 16
        static let xl: CGFloat = 24
        static let xxl: CGFloat = 32
    }

    // MARK: Radii (§5)

    enum Radius {
        static let chip: CGFloat = 6
        static let card: CGFloat = 10
        static let panel: CGFloat = 14
        static let hero: CGFloat = 20
    }

    // MARK: Layout (§5)

    enum Layout {
        static let popoverWidth: CGFloat = 360
        static let popoverMaxHeight: CGFloat = 560
        static let readingWidth: CGFloat = 680
        static let hudWidth: CGFloat = 440
        static let hudHeight: CGFloat = 300
        static let hudMinimized: CGFloat = 46
        static let hudMargin: CGFloat = 14
        static let recordButton: CGFloat = 56
        static let progressHeight: CGFloat = 4
    }

    // MARK: Motion (§8)

    static let smooth = Animation.smooth(duration: 0.3)
    static let bouncy = Animation.bouncy(duration: 0.35)
    static let pulse = Animation.easeInOut(duration: 1.2).repeatForever(autoreverses: true)
}

extension Color {
    /// Dynamic light/dark color from two sRGB hex values.
    init(light: UInt32, dark: UInt32) {
        self.init(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .vibrantDark, .accessibilityHighContrastDarkAqua]) != nil
            return NSColor(hex: isDark ? dark : light)
        })
    }
}

extension NSColor {
    convenience init(hex: UInt32) {
        self.init(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
                  green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue: CGFloat(hex & 0xFF) / 255,
                  alpha: 1)
    }
}

/// Speaker → color, in order of first appearance; the local speaker is neutral.
struct SpeakerPalette {
    private var order: [String] = []
    let localNames: Set<String>

    init(speakers: [String], localNames: Set<String> = ["Me", "Ich"]) {
        self.localNames = localNames
        for s in speakers where !localNames.contains(s) && !order.contains(s) {
            order.append(s)
        }
    }

    func color(for speaker: String?) -> Color {
        guard let speaker, !localNames.contains(speaker) else { return .secondary }
        // Unknown speaker: stable across launches (hashValue is seeded per process).
        guard let i = order.firstIndex(of: speaker) else {
            let h = speaker.unicodeScalars.reduce(0) { $0 &+ Int($1.value) }
            return Theme.speakers[h % Theme.speakers.count]
        }
        return Theme.speakers[i % Theme.speakers.count]
    }
}
