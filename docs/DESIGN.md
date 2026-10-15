# meetre — Design & Rule Book

The rules every meetre surface follows: the Mac app (menu bar, library window,
HUDs, settings) today, the companion app later. When a decision isn't covered
here, follow Apple's Human Interface Guidelines for macOS 26 and the
**principles** below — then add the rule here.

---

## 1. Principles

1. **Quiet until needed.** meetre lives next to a meeting, not in it. Nothing
   moves, blinks or asks for attention unless the user must act. Recording is
   the only state allowed to be loud (the red dot).
2. **Content over chrome.** The meeting — what was decided, who does what — is
   the interface. Toolbars, labels and borders are as light as they can be.
3. **Local and honest.** Never imply cloud processing. Say "on-device" where it
   reassures, show real progress (no fake spinners), say plainly when something
   failed and what to do.
4. **One obvious action per surface.** Menu bar: record/stop. Meeting: read the
   summary. Settings: change one thing. Secondary actions live in menus.
5. **Native first.** SwiftUI + system controls, SF Symbols, system materials
   (Liquid Glass), system type sizes. Custom drawing only where the system has
   nothing (HUD caption flow, waveform).
6. **Feels instant.** Every interaction gives feedback within 100 ms
   (state change, pressed state, or a progress affordance), even if the work
   takes minutes.

---

## 2. Brand

- **Name:** always lowercase `meetre`, also at sentence start in UI copy.
- **Mark:** the `✦` sparkle (status item uses the template-rendered app icon;
  `✦` is the text fallback).
- **Voice:** calm, short, concrete. Verbs on buttons ("Record", "Stop",
  "Summarize again"), never "OK" when a verb fits. No exclamation marks, no
  emoji in UI chrome (emoji are allowed inside user content only).
- **Languages:** UI is English with German localization; meeting content
  follows the meeting language setting. Never mix languages within one control.

---

## 3. Color

Use **semantic system colors**; never hard-code hex for text or backgrounds.
The only custom colors are the tokens below, defined in the asset catalog with
light/dark variants.

| Token | Light | Dark | Use |
|---|---|---|---|
| `accent` | `#5B5BD6` (indigo) | `#8B8BF5` | Selection, links, primary buttons, focus. App tint. |
| `record` | `#E5484D` | `#FF6369` | Recording state ONLY: record button fill, red dot, elapsed time. |
| `live` | `#30A46C` | `#3DD68C` | "Live"/connected indicators (captions running, engine ready). |
| `warn` | `#F5A524` | `#FFC53D` | Recoverable problems (model missing, permission needed). |
| `step.preparing` | `#8B8D98` | `#B0B4BA` | Pipeline step: finishing recording, loading models. |
| `step.transcribing` | `#0090FF` | `#3B9EFF` | Pipeline step: speech → text. |
| `step.speakers` | `#F76B15` | `#FF801F` | Pipeline step: who spoke (diarization, naming). |
| `step.summary` | = `accent` | = `accent` | Pipeline step: summary + title. |
| `step.saving` | = `live` | = `live` | Pipeline step: writing files / Apple Notes; also the "Done ✓" flash. |
| `speaker.1…8` | system palette | system palette | Speaker chips + transcript gutters: `.blue .orange .pink .teal .purple .brown .mint .indigo`, in order of first appearance. "Me" is always `.secondary` (neutral). |

Rules:
- `record` red appears nowhere except recording. Destructive actions use the
  system `.red` role styling (which is different in intent), never `record`.
- Text: `.primary`, `.secondary`, `.tertiary` only. Placeholder/empty states use
  `.tertiary`.
- Never convey state by color alone — pair with a symbol or a word
  (● Recording, ✓ Done, ⚠ Needs permission).
- HUD panels are always dark (`.environment(\.colorScheme, .dark)`), so they
  read over any wallpaper or app.

---

## 4. Typography

System font (SF Pro / SF Pro Rounded for numerals in timers), Dynamic Type
styles only — no fixed point sizes except in HUDs.

| Role | Style | Notes |
|---|---|---|
| Window title / meeting title | `.title2.weight(.semibold)` | One per view. |
| Section header (Summary, Decisions …) | `.headline` | Sentence case, no colon. |
| Body / summary bullets | `.body` | Line spacing +2. Max line length ~72 ch (`frame(maxWidth: 680)`). |
| Metadata (date · duration · speakers) | `.subheadline` + `.secondary` | Separated by ` · `. |
| Labels, chips | `.caption.weight(.medium)` | Speaker chips uppercase **only** in HUD. |
| Timers | `.title3.monospacedDigit()` rounded design | Never jitters while counting. |
| HUD caption text | 13 pt regular | HUD header 10 pt semibold, tracking +1.2, uppercase. |

---

## 5. Layout, spacing, shape

- **Spacing scale (pt):** 2, 4, 8, 12, 16, 24, 32. Nothing else.
  Default stack spacing 8; section spacing 24; window content padding 24
  (16 in popovers).
- **Corner radii:** 6 (chips, small buttons), 10 (cards, text fields),
  14 (HUD panels, popover cards), 20 (large hero cards). Use
  `.containerRelative`/concentric shapes inside glass containers.
- **Window:** `NavigationSplitView`; sidebar 240–320 pt, detail min 520 pt;
  default window 1100×720. Toolbar uses unified compact style.
- **Popover (menu bar):** fixed width 360 pt, height fits content (max 560).
- **HUDs:** 440 pt wide, top-right of the main screen's visible frame with a
  14 pt margin; live transcript 300 pt tall (46 pt minimized), summary 300 pt.

---

## 6. Materials (Liquid Glass)

- Use system glass via `.glassEffect()` / `GlassEffectContainer` for floating
  controls: the record button, popover action rows, HUD header controls.
- Content areas (transcript, summary text) sit on **solid** backgrounds
  (`.background(.background)`) — never glass behind long text.
- Don't stack glass on glass. One glass layer per region.
- HUD panels: dark HUD material + 94% graphite tint, 1 pt white border at 14%
  opacity, 14 pt radius, system shadow.

---

## 7. Iconography

SF Symbols only, `.hierarchical` rendering by default.

| Concept | Symbol |
|---|---|
| Record / recording | `record.circle` / `record.circle.fill` (in `record` color) |
| Stop | `stop.fill` |
| Live captions | `captions.bubble` |
| Summary | `text.badge.star` |
| Transcript | `text.quote` |
| Chat | `bubble.left.and.text.bubble.right` |
| Action item | `checkmark.circle` / `circle` |
| Decision | `checkmark.seal` |
| Open question | `questionmark.circle` |
| Speakers | `person.2` |
| Apple Notes | `note.text` |
| Settings | `gearshape` |
| On-device | `lock.shield` |

---

## 8. Motion

- Default animation: `.smooth(duration: 0.3)`; state changes of the record
  button: `.bouncy(duration: 0.35)`.
- Lists insert/remove with `.opacity` + slight `.move(edge: .top)`.
- The recording dot pulses (opacity 1 → 0.35, 1.2 s, ease-in-out, repeat).
  Nothing else loops forever.
- Respect **Reduce Motion**: replace movement with cross-fades; stop pulsing.
- HUDs fade in 0.35 s, fade out 0.6 s; never slide.

---

## 9. Components

**Record button** — the hero control. Circle 56 pt in the popover, glass,
`record.circle` idle → filled red square-in-circle while recording, with the
elapsed timer next to it. One click toggles. Long-press / menu offers "Record
with previous settings" and "Record with options…".

**Menu bar item** — the one place color carries state at a glance:
idle `sparkles` (template, monochrome) · recording red `record.circle.fill`
+ timer · processing the current step's colored symbol + "Transcribing 42%"
· done green `checkmark.circle.fill` + "Done" for 5 s · failed amber
`exclamationmark.triangle.fill` until the popover is opened. Colored symbols
are non-template images; text stays template so it follows the menu bar.

**Pipeline steps** — while processing, the popover shows the five steps
(Preparing · Transcribing · Speakers · Summary · Saving) as a checklist:
done ✓ in the step color, active with pulse + thin progress bar + detail
line, pending dimmed. Never go backwards.

**Status line** — one line under the record button: state + detail
(`Transcribing · 42%`, `Detecting speakers…`, `Ready · on-device`). Progress
bars are thin (4 pt), rounded, `accent` tint, determinate whenever possible.

**Meeting row (sidebar)** — title (1 line, `.body`), secondary line with time
and duration, trailing small badges: summary ready (`text.badge.star`), tasks
count. Grouped by Today / Yesterday / This week / Earlier.

**Speaker chip** — capsule, speaker color at 15% fill + full-color text,
`.caption.weight(.medium)`. "Me" uses neutral.

**Action item row** — checkbox (`circle`/`checkmark.circle.fill`), task text,
owner chip, optional due text in `.secondary`. Checking is local state stored
with the meeting (synced later).

**Empty states** — symbol (48 pt, `.tertiary`), one sentence, one button.
Example: "No meetings yet." [Record].

**Errors** — inline banner at the top of the affected view (`warn` symbol, one
sentence, one fix button). Use alerts only to confirm destructive actions.

---

## 10. Surfaces

### Menu bar popover
1. Header: `meetre` wordmark (secondary) · engine status dot (`live`) · settings button.
2. Hero: record button + timer + status line; live caption preview (last line,
   `.secondary`) while recording.
3. Quick options (only when idle): language, speakers, system audio — as
   compact menus, not a long list.
4. Recent: last 3 meetings (tap opens the library at that meeting).
5. Footer: "Open meetre" (⌘O), "Quit".

### Library window
- Sidebar: search field, grouped meetings.
- Detail: title (editable inline), metadata line, speaker chips, then a
  segmented control **Summary · Transcript · Chat**.
  - Summary: Decisions, Action items (checklist), Open questions as distinct
    sections; "Summarize again" in the toolbar.
  - Transcript: speaker-grouped turns with colored gutter, timestamps in
    `.tertiary` on hover, find (⌘F).
  - Chat: K2 conversation about this meeting, suggestion chips when empty.
- Toolbar: Record (mirrors menu bar), Share (Markdown/Notes), Reveal in Finder.

### HUDs
- Live transcript and live summary as described in §5/§6. Click-through
  everywhere except their header buttons (minimize, close).

### Settings
Tabs: **General** (start at login, call detection, HUD toggles, storage
folder) · **Transcription** (model, language, system audio, VAD) ·
**Speakers** (detection, count, my name, name from conversation) ·
**Summary** (model with fit badges, auto-summarize, prompt editor with reset) ·
**Integrations** (Apple Notes, HF token) · **Advanced** (logs, engine restart).

### Onboarding (first launch)
A borderless, full-screen window *below* the menu bar over the blurred
desktop, with a slowly drifting mesh-gradient aurora (indigo, violet, a hint
of green) at ≤55% opacity. Always dark.
- **Words:** one short line per scene, never a paragraph. "Hi." → "It
  listens." → "It writes it down." → "And it stays on this Mac." → "Let's set
  you up." → "You're set." Display type is SF Pro Rounded, semibold, 40–96 pt —
  the only place rounded display type is allowed.
- **Pace:** scenes auto-advance (1.7–2.4 s); click, Return or Space skips.
  Transitions are soft scale + fade (0.7 s); Reduce Motion → fades only.
- **Install card:** one glass card with at most five toggles (icon + few
  words), one prominent capsule button ("Install" / "Set up"), and the
  on-device line. Options that need more input reveal it inline.
- **Progress:** a ring with a slowly turning sparkle and one changing line
  ("Downloading the brain"). Never show logs; on failure, one sentence, the
  last error line, and "Try again".
- **Done:** "You're set." with a pointer toward the status item.

---

## 11. Sound

Sound is reserved for onboarding — the everyday app is silent (system
notification sounds only).
- Synthesized soft bells (`Chimes.swift`), major pentatonic, no assets.
- Cues: hello (rising arpeggio), step (single note), hover (barely audible
  tick), toggle (up/down note), install progress (one ascending note per
  step), success (bright flourish), failure (two descending notes).
- Always provide a mute button (bottom-right). Never loop sounds. Keep the
  mixer around −5 dB; nothing should startle during a call.

---

## 12. Accessibility

- Every control has an accessibility label; icon-only buttons get `.help()`
  tooltips with the same text.
- All functionality is keyboard reachable. Shortcuts: ⌘R record/stop (in-app),
  ⌘O open library, ⌘F find, ⌘, settings, ⌘⇧S summarize again.
- Minimum contrast 4.5:1 for text (HUD text verified against the graphite
  tint).
- Honor Reduce Motion, Reduce Transparency (glass → solid), Increase Contrast.

---

## 13. Engineering rules for UI

- SwiftUI + Observation (`@Observable`), Swift 6 strict concurrency, macOS 26
  deployment target.
- Views never talk to the engine directly — only through `AppModel`.
- No business logic in views; formatting helpers live in `Formatting.swift`.
- Every new color, radius or spacing value must come from the tokens above
  (`Theme.swift`); a PR that adds a literal adds it to this book first.
- Previews for every view with fixture data (`Fixtures.swift`).
