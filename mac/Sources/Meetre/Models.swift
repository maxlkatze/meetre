import Foundation

/// Any JSON value — used for config and raw RPC payloads so the app tolerates
/// keys the engine adds later.
enum JSONValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let n = try? c.decode(Double.self) { self = .number(n) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let n): try c.encode(n)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    var bool: Bool? { if case .bool(let b) = self { b } else { nil } }
    var string: String? { if case .string(let s) = self { s } else { nil } }
    var double: Double? { if case .number(let n) = self { n } else { nil } }
    var int: Int? { double.map { Int($0) } }
    var array: [JSONValue]? { if case .array(let a) = self { a } else { nil } }
    var object: [String: JSONValue]? { if case .object(let o) = self { o } else { nil } }
    var isNull: Bool { self == .null }

    subscript(key: String) -> JSONValue? { object?[key] }

    /// Decode into a Codable type (snake_case keys → camelCase properties).
    func decode<T: Decodable>(_ type: T.Type = T.self) throws -> T {
        let data = try JSONEncoder().encode(self)
        return try JSONValue.decoder.decode(T.self, from: data)
    }

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()
}

// MARK: - Engine state

enum RecorderState: String, Codable, Sendable {
    case idle, recording, processing
}

struct EngineState: Codable, Sendable, Equatable {
    var state: RecorderState = .idle
    var elapsed: Double = 0
    var stage: String? = nil
    var progress: Double? = nil
    var progressLabel: String? = nil
    var title: String? = nil
}

// MARK: - Meetings

struct MeetingSummary: Codable, Sendable, Identifiable, Hashable {
    var id: String
    var title: String
    var startedAt: String?
    var duration: Double?
    var speakers: [String]?
    var hasSummary: Bool?
    var path: String?

    var date: Date? { startedAt.flatMap(Formatting.parseISO) }
}

/// An audio backup with no transcript (the meeting failed or was deleted).
struct AudioBackup: Codable, Sendable, Identifiable, Hashable {
    var path: String
    var name: String
    var startedAt: String?
    var size: Int?
    var stems: [String]?

    var id: String { path }
    var date: Date? { startedAt.flatMap(Formatting.parseISO) }
}

struct MeetingTask: Codable, Sendable, Hashable {
    var text: String
    var owner: String?
    var due: String?
}

struct MeetingSections: Codable, Sendable, Hashable {
    var summary: String?
    var decisions: [String]?
    var tasks: [MeetingTask]?
    var questions: [String]?
}

struct Turn: Codable, Sendable, Hashable {
    var speaker: String?
    var start: Double?
    var text: String
    /// Timed words for playback highlighting (estimated for old transcripts).
    var words: [Word]? = nil
}

struct Word: Codable, Sendable, Hashable {
    var w: String
    var s: Double
    var e: Double
}

struct Meeting: Codable, Sendable, Identifiable, Hashable {
    var id: String
    var title: String
    var startedAt: String?
    var duration: Double?
    var speakers: [String]?
    var hasSummary: Bool?
    var path: String?
    var summaryMd: String?
    var sections: MeetingSections?
    var turns: [Turn]?
    /// The MP3 backup, for listening back in the transcript.
    var audio: String? = nil
    var model: String?
    var backend: String?

    var date: Date? { startedAt.flatMap(Formatting.parseISO) }
    var summary: MeetingSummary {
        MeetingSummary(id: id, title: title, startedAt: startedAt, duration: duration,
                       speakers: speakers, hasSummary: hasSummary, path: path)
    }
}

// MARK: - Catalog

struct ASRModel: Codable, Sendable, Identifiable, Hashable {
    var id: String
    var label: String?
    var installed: Bool?
}

struct SummaryModel: Codable, Sendable, Identifiable, Hashable {
    var id: String
    var label: String?
    var note: String?
    var sizeGb: Double?
    var fits: Bool?
    var installed: Bool?
    var recommended: Bool?
    /// A GGUF file from the local models folder (run with llama.cpp).
    var local: Bool?
}

struct DownloadedModel: Codable, Sendable, Identifiable, Hashable {
    var id: String
    var label: String
    var sizeBytes: Double
    var kind: String
    var inUse: Bool?
}

struct Language: Codable, Sendable, Identifiable, Hashable {
    var code: String?
    var label: String
    var id: String { code ?? "auto" }
}

struct Catalog: Codable, Sendable {
    var asrModels: [ASRModel] = []
    var summaryModels: [SummaryModel] = []
    var languages: [Language] = []
    var defaultPrompt: String? = nil
    var defaultSummaryModel: String? = nil
}

// MARK: - Chat

struct ChatMessage: Identifiable, Hashable, Sendable {
    enum Role: String, Sendable { case user, assistant }
    let id = UUID()
    var role: Role
    var text: String
}

// MARK: - Live

struct CaptionLine: Hashable, Sendable {
    var speaker: String?
    var text: String
}

struct SummaryStream: Sendable, Equatable {
    var meetingID: String?
    var phase: String = "thinking"
    var text: String = ""
    var stage: String = ""
    var active = false
}

enum CallPrompt: Equatable, Sendable {
    case started(Date)
    case ended(Date)
}
