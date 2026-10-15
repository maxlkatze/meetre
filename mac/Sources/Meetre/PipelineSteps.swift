import AppKit
import SwiftUI

/// The post-recording pipeline as five visible steps (DESIGN.md §3 step
/// colors, §9 "Pipeline steps"). Derived from the engine's free-text `stage`
/// / `progress_label`, so new engine wording only needs a mapping here.
enum PipelineStep: Int, CaseIterable, Comparable, Sendable {
    case preparing, transcribing, speakers, summary, saving

    static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }

    var title: String {
        switch self {
        case .preparing: "Preparing"
        case .transcribing: "Transcribing"
        case .speakers: "Speakers"
        case .summary: "Summary"
        case .saving: "Saving"
        }
    }

    var symbol: String {
        switch self {
        case .preparing: "waveform"
        case .transcribing: "text.quote"
        case .speakers: "person.2.fill"
        case .summary: "text.badge.star"
        case .saving: "square.and.arrow.down.fill"
        }
    }

    var color: Color { Theme.step(self) }

    /// Maps engine wording ("Detecting speakers…", "Speakers: analyzing
    /// voices", "Loading summary model…", …) to a step.
    static func from(stage: String?, label: String?) -> PipelineStep {
        let text = "\(label ?? "") \(stage ?? "")".lowercased()
        if text.contains("saving") || text.contains("notes") { return .saving }
        if text.contains("summar") || text.contains("naming meeting") || text.contains("title") { return .summary }
        if text.contains("speaker") { return .speakers }
        if text.contains("transcri") { return .transcribing }
        return .preparing
    }
}

/// What the menu bar item shows right now.
enum StatusBadge: Equatable {
    case idle
    case recording(elapsed: Double)
    case working(PipelineStep, progress: Double?)
    case done
    case failed
}

extension AppModel {
    var badge: StatusBadge {
        switch state.state {
        case .recording: return .recording(elapsed: state.elapsed)
        case .processing:
            return .working(PipelineStep.from(stage: state.stage, label: state.progressLabel),
                            progress: state.progress)
        case .idle:
            if pipelineFailed { return .failed }
            return finishedAt != nil ? .done : .idle  // AppModel clears it after 5 s
        }
    }
}

// MARK: - Menu bar label

/// Menu bar labels are template-rendered (monochrome); colored symbols must be
/// non-template images. Text stays a real `Text` so it follows the menu bar's
/// light/dark appearance.
///
/// Keep this view cheap and stable: SwiftUI re-renders the status item on
/// every change it sees here. A TimelineView or a freshly built NSImage per
/// render made it re-render endlessly (100% CPU, GBs of images) — so no
/// timers here, and images come from a cache (same instance every time).
struct MenuBarLabel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        content(model.badge)
    }

    @ViewBuilder private func content(_ badge: StatusBadge) -> some View {
        switch badge {
        case .idle:
            Image(systemName: "sparkles")
        // One image per state (symbol + text): the item keeps a constant
        // width while digits tick, so the popover anchored to it can't jiggle.
        case .recording(let elapsed):
            Image(nsImage: .statusLabel("record.circle.fill", Theme.recordNS, Formatting.timer(elapsed)))
        case .working(let step, let progress):
            Image(nsImage: .statusLabel(step.symbol, NSColor(step.color),
                                        progress.map { "\(step.title) \(String(format: "%3d", Int($0 * 100)))%" }
                                            ?? "\(step.title)…"))
        case .done:
            Image(nsImage: .statusLabel("checkmark.circle.fill", NSColor(Theme.live), "Done"))
        case .failed:
            Image(nsImage: .colored("exclamationmark.triangle.fill", NSColor(Theme.warn)))
        }
    }
}

@MainActor
private var coloredImageCache: [String: NSImage] = [:]

extension NSImage {
    /// A full-color (non-template) SF Symbol sized for the menu bar. Cached:
    /// the menu bar label must get the identical image on every render.
    @MainActor
    static func colored(_ symbol: String, _ color: NSColor, pointSize: CGFloat = 14) -> NSImage {
        let rgb = color.usingColorSpace(.sRGB)
        let key = "\(symbol)|\(pointSize)|\(rgb?.redComponent ?? 0),\(rgb?.greenComponent ?? 0),\(rgb?.blueComponent ?? 0)"
        if let cached = coloredImageCache[key] { return cached }
        let config = NSImage.SymbolConfiguration(pointSize: pointSize, weight: .semibold)
            .applying(.init(paletteColors: [color]))
        let image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)?
            .withSymbolConfiguration(config) ?? NSImage()
        image.isTemplate = false
        coloredImageCache[key] = image
        return image
    }
}

@MainActor
private var statusLabelCache: [String: NSImage] = [:]

extension NSImage {
    /// Colored symbol + text drawn into a single non-template image. Digits
    /// use a fixed-width font so "0:59" and "1:00" are the same width; the
    /// text color is resolved at draw time (follows the menu bar's
    /// appearance). A few recent labels are cached so re-renders reuse them.
    @MainActor
    static func statusLabel(_ symbol: String, _ color: NSColor, _ text: String) -> NSImage {
        let key = "\(symbol)|\(color.description)|\(text)"
        if let cached = statusLabelCache[key] { return cached }
        let icon = colored(symbol, color, pointSize: 13)
        let font = NSFont.monospacedDigitSystemFont(ofSize: 13, weight: .medium)
        let textSize = (text as NSString).size(withAttributes: [.font: font])
        let gap: CGFloat = 4
        let height: CGFloat = 18
        let size = NSSize(width: ceil(icon.size.width + gap + textSize.width), height: height)
        let image = NSImage(size: size, flipped: false) { _ in
            icon.draw(in: NSRect(x: 0, y: (height - icon.size.height) / 2,
                                 width: icon.size.width, height: icon.size.height))
            (text as NSString).draw(at: NSPoint(x: icon.size.width + gap, y: (height - textSize.height) / 2),
                                    withAttributes: [.font: font, .foregroundColor: NSColor.labelColor])
            return true
        }
        image.isTemplate = false
        if statusLabelCache.count > 8 { statusLabelCache.removeAll() }
        statusLabelCache[key] = image
        return image
    }
}

// MARK: - Popover step list

/// The pipeline as a short checklist: done ✓, the active step with its
/// progress, pending steps dimmed.
struct PipelineStepsView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let badge = model.badge
        VStack(alignment: .leading, spacing: Theme.Space.s) {
            ForEach(PipelineStep.allCases, id: \.self) { step in
                row(step, badge: badge)
            }
        }
        .padding(Theme.Space.m)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: Theme.Radius.card))
        .animation(Theme.smooth, value: badge)
    }

    private func row(_ step: PipelineStep, badge: StatusBadge) -> some View {
        let status = status(of: step, badge: badge)
        return HStack(spacing: Theme.Space.s) {
            ZStack {
                Circle().fill(step.color.opacity(status == .pending ? 0.08 : 0.18))
                Image(systemName: status == .done ? "checkmark" : step.symbol)
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(status == .pending ? Color.secondary : step.color)
                    .symbolEffect(.pulse, isActive: status == .active)
            }
            .frame(width: 22, height: 22)
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(step.title)
                        .font(.callout.weight(status == .active ? .semibold : .regular))
                        .foregroundStyle(status == .pending ? .tertiary : .primary)
                    Spacer()
                    if status == .active, let p = model.state.progress {
                        Text("\(Int(p * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    } else if status == .active {
                        ProgressView().controlSize(.mini)
                    }
                }
                if status == .active {
                    if let p = model.state.progress {
                        ProgressView(value: p).tint(step.color).controlSize(.mini)
                    }
                    if let detail = model.state.progressLabel ?? model.state.stage {
                        Text(detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
        }
    }

    private enum RowStatus { case done, active, pending }

    private func status(of step: PipelineStep, badge: StatusBadge) -> RowStatus {
        switch badge {
        case .done: return .done
        case .working(let current, _):
            let reached = max(current, model.reachedStep ?? current)
            if step < reached { return .done }
            return step == reached ? .active : .pending
        default: return .pending
        }
    }
}
