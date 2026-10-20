# meetre — Native app roadmap

## Architecture

```
┌──────────────── Meetre.app (SwiftUI, macOS 26) ────────────────┐
│ MenuBarExtra popover · Library window · Settings · HUD panels  │
│ AppModel (@Observable)  ←→  EngineClient (JSON-RPC over stdio) │
└───────────────────────────────┬────────────────────────────────┘
                                │ newline-delimited JSON-RPC 2.0
┌───────────────────────────────▼────────────────────────────────┐
│ meetre engine  (python -m meetre engine — existing Python)     │
│ Recorder · LiveTranscriber · Parakeet/Whisper · pyannote ·     │
│ K2 summaries/chat · Apple Notes · call detection · library     │
└────────────────────────────────────────────────────────────────┘
```

- The app launches the engine as a child process, so macOS attributes
  microphone / screen-recording permission to **Meetre.app**.
- The engine is headless (no AppKit UI). Everything visual is Swift.
- Pieces may move to Swift later (e.g. capture via AVAudioEngine/SCK, Parakeet
  via CoreML) behind the same protocol, one method at a time.

## Phases

1. **Desktop app (now)** — engine protocol, SwiftUI app: menu bar popover,
   library (summary/transcript/chat), settings, live HUDs, call prompt.
   Python menu bar stays as fallback until the Swift app reaches parity; then
   the login item switches to Meetre.app.
2. **Storage v2 + sync** — meetings become folders (below) in a user-chosen
   root, default `iCloud Drive/meetre`. Engine migrates legacy `.md` files.
   File coordination (`NSFileCoordinator`) on the Swift side; engine writes
   atomically (temp file + rename). Conflicts: last-writer-wins per file;
   action-item checks merge by id.
3. **Companion app (planned)** — iPhone/iPad, reads the same iCloud folder.
   v1: browse/search meetings, read summary & transcript, check off action
   items, share. Recording stays on the Mac. Later: record in-person meetings
   on the phone, transcribe on the Mac when synced (or on-device CoreML).
4. **Native engine** — move capture and ASR to Swift; LLM via MLX Swift once
   K2 (or a successor) runs there.

## Storage v2 (sync format)

```
<root>/2026-10-15 0930 Q4 roadmap review/
  meeting.json     # id (UUID), title, started_at, duration, language,
                   # models, speakers [{label, name, color}],
                   # tasks [{id, text, owner, due, done}], schema_version
  transcript.md    # human-readable, same format as today
  summary.md       # summary sections only
  audio.m4a        # optional (setting), AAC 64 kbit/s mono
```

Plain files on purpose: readable in Finder/Obsidian, sync through iCloud
Drive with no developer entitlement, trivially backed up.

## Engine protocol (v1)

Transport: engine stdin/stdout, one JSON object per line (UTF-8). stderr is
free-form logging. Requests carry `id`; notifications don't.

```json
→ {"jsonrpc":"2.0","id":1,"method":"record.start","params":{"title":"Weekly"}}
← {"jsonrpc":"2.0","id":1,"result":{"started_at":"2026-10-15T10:46:00"}}
← {"jsonrpc":"2.0","method":"state","params":{"state":"recording","elapsed":3.0}}
```

Errors: `{"id":1,"error":{"code":-32000,"message":"…","data":{"hint":"…"}}}`.

### Methods (app → engine)

| Method | Params | Result |
|---|---|---|
| `hello` | `{client_version}` | `{engine_version, protocol: 1, python, platform}` |
| `state.get` | – | `State` (see `state` event) |
| `config.get` | – | full config object (keys as in `config.py`) |
| `config.set` | `{key: value, …}` | full config object |
| `catalog.get` | – | `{asr_models:[{id,label,installed}], summary_models:[{id,label,note,size_gb,fits,installed,recommended}], languages:[{code,label}]}` |
| `record.start` | `{title?: str, options?: {…config overrides}}` | `{started_at}` — empty title ⇒ auto title from summary |
| `record.stop` | – | `{}` (processing continues; watch events) |
| `record.cancel` | – | `{}` discards audio |
| `meetings.list` | `{query?: str}` | `[MeetingSummary]` newest first |
| `meetings.get` | `{id}` | `Meeting` |
| `meetings.summarize` | `{id}` | `{}`; streams `summary.stream`, ends with `meeting.updated` |
| `meetings.retranscribe` | `{id}` | `{}`; re-runs the pipeline on the audio backup, ends with `meeting.updated` |
| `meetings.import` | `{path}` | `{}`; transcribes an audio file (orphaned backup or any MP3/WAV/…) into a new meeting, ends with `meeting.updated` + `backups.changed` |
| `backups.list` | – | `[{path, name, started_at, size, stems:[str]}]` audio backups without a transcript, newest first |
| `meetings.rename` | `{id, title}` | `MeetingSummary` |
| `meetings.delete` | `{id}` | `{}` (moves file to Trash) |
| `meetings.reveal` | `{id}` | `{}` (Finder) |
| `meetings.open_note` | `{id}` | `{}` (Apple Notes, if a note exists) |
| `chat.send` | `{meeting_id?, history:[{role,content}], message}` | `{text}` final; streams `chat.delta` |
| `chat.stop` | – | `{}` |
| `models.ensure` | `{kind:"asr"|"summary", id}` | `{}`; streams `state` download progress |
| `models.uninstall` | `{id}` | `{freed_bytes}` |
| `autostart.set` | `{enabled}` | `{enabled}` |
| `shutdown` | – | `{}` then exit |

`MeetingSummary` = `{id, title, started_at, duration, speakers:[str], has_summary, path}`
(`id` = file name stem for now).
`Meeting` = `MeetingSummary + {summary_md, sections:{summary, decisions[], tasks[{text, owner}], questions[]}, turns:[{speaker, start, text}], model, backend}`.

### Events (engine → app)

| Event | Params |
|---|---|
| `state` | `{state: "idle"|"recording"|"processing", elapsed, stage, progress (0–1 or null), progress_label, title}` — sent on change and every 0.5 s while recording/processing |
| `live.caption` | `{lines:[[speaker, text]], tentative, status}` |
| `summary.stream` | `{meeting_id?, phase: "thinking"|"writing"|"done", text, stage}` |
| `chat.delta` | `{text}` (full text so far) |
| `meeting.updated` | `{id, stage: "transcript"|"speakers"|"summary"|"done", summary: MeetingSummary}` |
| `backups.changed` | `{}` — refetch `backups.list` |
| `call` | `{event: "started"|"ended"}` (Teams call detection) |
| `notify` | `{title, subtitle, body}` (app decides whether to post a notification) |
| `error` | `{message, hint}` |

## Open questions

- Distribute Meetre.app signed/notarized (needs Developer ID) or keep ad-hoc
  signed local builds via `mac/build.sh`?
- Audio retention default for synced folders (size vs. re-transcription).
