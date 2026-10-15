import Foundation

/// Sample data for previews (and `--fixtures`, which runs the UI without any
/// engine). Mirrors the shapes the engine returns.
@MainActor
enum Fixtures {
    static let meetings: [MeetingSummary] = {
        let cal = Calendar.current
        func iso(_ daysAgo: Int, _ h: Int, _ m: Int) -> String {
            let d = cal.date(bySettingHour: h, minute: m, second: 0,
                             of: cal.date(byAdding: .day, value: -daysAgo, to: .now)!)!
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
            return f.string(from: d)
        }
        // Invented demo data (previews, `--fixtures`, public screenshots) —
        // never real meetings.
        return [
            MeetingSummary(id: "onboarding", title: "Onboarding flow — design crit",
                           startedAt: iso(0, 11, 5), duration: 846, speakers: ["Me", "Speaker 1"],
                           hasSummary: true),
            MeetingSummary(id: "roadmap", title: "Q4 roadmap review",
                           startedAt: iso(0, 9, 30), duration: 1712, speakers: ["Me", "Speaker 1", "Speaker 2"],
                           hasSummary: true),
            MeetingSummary(id: "weekly", title: "Weekly product sync",
                           startedAt: iso(1, 10, 0), duration: 1885, speakers: ["Me", "Speaker 1", "Speaker 2"],
                           hasSummary: true),
            MeetingSummary(id: "interview", title: "Customer interview — Lumen Labs",
                           startedAt: iso(4, 14, 30), duration: 2402, speakers: ["Me", "Speaker 1"],
                           hasSummary: true),
            MeetingSummary(id: "hiring", title: "Hiring debrief",
                           startedAt: iso(9, 16, 15), duration: 540, speakers: ["Me", "Speaker 1", "Speaker 2"],
                           hasSummary: false),
        ]
    }()

    static func meeting(id: String) -> Meeting {
        let s = meetings.first { $0.id == id } ?? meetings[1]
        return Meeting(
            id: s.id, title: s.title, startedAt: s.startedAt, duration: s.duration, speakers: s.speakers,
            hasSummary: s.hasSummary, path: nil,
            summaryMd: "## Summary\n…",
            sections: s.hasSummary == true ? sections : nil,
            turns: turns, model: "parakeet-tdt-v3", backend: "parakeet-tdt-v3")
    }

    static let sections = MeetingSections(
        summary: "The Q4 plan is too wide for the team: three bets compete for the same two engineers. "
            + "Offline mode moves to Q1 so search and the new onboarding can ship before the December freeze.",
        decisions: [
            "Offline mode moves to Q1 — it needs the sync rewrite first.",
            "Search ships behind a flag on Nov 14, then rolls out to 100% after a week.",
            "Onboarding v2 replaces the tour; the old tour is removed, not hidden.",
        ],
        tasks: [
            MeetingTask(text: "Write the Q1 brief for offline mode", owner: "Me", due: "Friday"),
            MeetingTask(text: "Set up the search rollout flag and dashboard", owner: "Speaker 2"),
            MeetingTask(text: "Share onboarding v2 prototype with support", owner: "Speaker 1", due: "Nov 7"),
            MeetingTask(text: "Book the December freeze review", owner: nil),
        ],
        questions: [
            "Do we need legal review for the new data export before the freeze?",
        ])

    static let turns: [Turn] = [
        Turn(speaker: "Me", start: 0, text: "Let's be honest about Q4: we have three big bets and two engineers."),
        Turn(speaker: "Me", start: 9, text: "Something has to move, and I think it's offline mode."),
        Turn(speaker: "Speaker 1", start: 21, text: "Agreed. Offline needs the sync rewrite anyway, that's a Q1 thing."),
        Turn(speaker: "Speaker 2", start: 34, text: "Then search can go out on the 14th behind a flag. I'd watch it for a week."),
        Turn(speaker: "Me", start: 46, text: "Good. And onboarding v2 replaces the tour completely?"),
        Turn(speaker: "Speaker 1", start: 52, text: "Yes. I'll share the prototype with support by the 7th."),
        Turn(speaker: "Speaker 2", start: 61, text: "Open point: does the new data export need a legal review before the freeze?"),
    ]

    static let captions: [CaptionLine] = [
        CaptionLine(speaker: "Me", text: "Okay, let's start — I sent the agenda this morning."),
        CaptionLine(speaker: "Speaker", text: "Got it. Can we begin with search? The flag is almost ready."),
        CaptionLine(speaker: "Me", text: "Sure. Then onboarding, and offline mode last."),
    ]

    static let config: [String: JSONValue] = [
        "model": .string("parakeet-tdt-v3"), "language": .string("en"),
        "person_detection": .bool(true), "capture_system": .bool(true),
        "summary_model": .string("k2-horizon-7b"), "auto_summarize": .bool(true),
        "auto_notes": .bool(true), "detect_calls": .bool(true), "live_transcript": .bool(true),
        "summary_overlay": .bool(true), "name_speakers": .bool(true), "my_name": .string(""),
        "vad": .bool(true), "word_timestamps": .bool(true), "num_speakers": .null,
        "summary_prompt": .string(""), "transcripts_dir": .string("~/Documents/meetre/transcripts"),
    ]

    static let catalog = Catalog(
        asrModels: [
            ASRModel(id: "parakeet-tdt-v3", label: "Parakeet TDT v3", installed: true),
            ASRModel(id: "large-v3-turbo", label: "Whisper large-v3 turbo", installed: false),
            ASRModel(id: "small", label: "Whisper small", installed: false),
        ],
        summaryModels: [
            SummaryModel(id: "k2-horizon-7b", label: "K2 Horizon 7B", note: "reasoning — excellent summary writing",
                         sizeGb: 7.2, fits: true, installed: true, recommended: true),
            SummaryModel(id: "qwen3.5-35b", label: "Qwen3.5 35B", note: "MoE, ~3B active — fast",
                         sizeGb: 19.6, fits: true, installed: false),
            SummaryModel(id: "qwen3.5-397b", label: "Qwen3.5 397B", note: "flagship MoE — Mac Studio Ultra only",
                         sizeGb: 210, fits: false, installed: false),
        ],
        languages: [Language(code: "de", label: "Deutsch"), Language(code: "en", label: "English"),
                    Language(code: nil, label: "Auto-detect")])

    static func apply(to model: AppModel) {
        model.meetings = meetings
        model.config = config
        model.catalog = catalog
        model.selectedID = "prio"
        model.captions = captions
        model.tentative = "Wir wollen ja einzigartig sein, kein Me-Too-Produkt"
        model.captionStatus = "listening"
    }
}
