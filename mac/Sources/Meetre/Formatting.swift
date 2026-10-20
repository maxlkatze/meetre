import Foundation

enum Formatting {
    private static let iso: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return f
    }()

    nonisolated(unsafe) private static let isoFrac: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    static func parseISO(_ s: String) -> Date? {
        if let d = iso.date(from: String(s.prefix(19))) { return d }
        return isoFrac.date(from: s) ?? (try? Date(s, strategy: .iso8601))
    }

    /// "42 s", "31 min", "1 h 05 min".
    static func duration(_ seconds: Double?) -> String {
        guard let seconds, seconds > 0 else { return "–" }
        let s = Int(seconds.rounded())
        if s < 60 { return "\(s) s" }
        let m = s / 60
        if m < 60 { return "\(m) min" }
        return String(format: "%d h %02d min", m / 60, m % 60)
    }

    /// Running timer "02:14" / "1:02:14".
    static func timer(_ seconds: Double) -> String {
        let s = max(0, Int(seconds))
        if s >= 3600 { return String(format: "%d:%02d:%02d", s / 3600, (s / 60) % 60, s % 60) }
        return String(format: "%02d:%02d", s / 60, s % 60)
    }

    /// Transcript offset "12:05".
    static func offset(_ seconds: Double?) -> String {
        guard let seconds else { return "" }
        return timer(seconds)
    }

    static func time(_ date: Date?) -> String {
        guard let date else { return "" }
        return date.formatted(date: .omitted, time: .shortened)
    }

    static func dayAndTime(_ date: Date?) -> String {
        guard let date else { return "" }
        return date.formatted(.dateTime.weekday(.wide).day().month(.wide).hour().minute())
    }

    /// Metadata line: "Wednesday, 15 October 10:46 · 31 min · 3 speakers".
    /// "20 Oct, 11:45" — fits a sidebar row.
    static func shortDayAndTime(_ date: Date?) -> String {
        guard let date else { return "" }
        return date.formatted(.dateTime.day().month(.abbreviated).hour().minute())
    }

    /// "14 MB"
    static func size(_ bytes: Int?) -> String {
        guard let bytes else { return "" }
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }

    static func meta(date: Date?, duration: Double?, speakers: Int) -> String {
        var parts = [dayAndTime(date), Self.duration(duration)].filter { !$0.isEmpty && $0 != "–" }
        if speakers > 0 { parts.append(speakers == 1 ? "1 speaker" : "\(speakers) speakers") }
        return parts.joined(separator: " · ")
    }

    // MARK: Sidebar grouping

    enum Group: Int, CaseIterable, Comparable {
        case today, yesterday, thisWeek, earlier

        var title: String {
            switch self {
            case .today: "Today"
            case .yesterday: "Yesterday"
            case .thisWeek: "This week"
            case .earlier: "Earlier"
            }
        }

        static func < (a: Group, b: Group) -> Bool { a.rawValue < b.rawValue }
    }

    static func group(for date: Date?, now: Date = .now) -> Group {
        guard let date else { return .earlier }
        let cal = Calendar.current
        if cal.isDateInToday(date) { return .today }
        if cal.isDateInYesterday(date) { return .yesterday }
        if let week = cal.dateInterval(of: .weekOfYear, for: now), week.contains(date) { return .thisWeek }
        return .earlier
    }

    /// "Detecting speakers…" style stage text, capitalised.
    static func stage(_ s: String?) -> String {
        guard let s, !s.isEmpty else { return "" }
        return s.prefix(1).uppercased() + s.dropFirst()
    }
}
