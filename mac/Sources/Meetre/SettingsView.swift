import AppKit
import SwiftUI

/// Settings (DESIGN.md §10): General · Transcription · Speakers · Summary ·
/// Integrations · Advanced.
struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("settingsTab") private var tab = "general"

    var body: some View {
        TabView(selection: $tab) {
            Tab("General", systemImage: "gearshape", value: "general") { GeneralPane() }
            Tab("Transcription", systemImage: "waveform", value: "transcription") { TranscriptionPane() }
            Tab("Speakers", systemImage: "person.2", value: "speakers") { SpeakersPane() }
            Tab("Summary", systemImage: "text.badge.star", value: "summary") { SummaryPane() }
            Tab("Integrations", systemImage: "note.text", value: "integrations") { IntegrationsPane() }
            Tab("Advanced", systemImage: "wrench.and.screwdriver", value: "advanced") { AdvancedPane() }
        }
        .frame(width: 560)
        .tint(Theme.accent)
    }
}

private struct Pane<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        Form { content }
            .formStyle(.grouped)
            .frame(minHeight: 380)
            .scrollContentBackground(.hidden)
    }
}

private struct GeneralPane: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Pane {
            Section {
                Toggle("Start meetre at login", isOn: Binding(get: { model.startAtLogin },
                                                               set: { model.setStartAtLogin($0) }))
                Toggle("Ask to record Teams calls", isOn: model.bool("detect_calls", default: true))
            }
            Section("Overlays") {
                Toggle("Live transcript while recording", isOn: model.bool("live_transcript", default: true))
                Toggle("Live summary while it's written", isOn: model.bool("summary_overlay", default: true))
            }
            Section {
                LabeledContent("Transcripts") {
                    HStack(spacing: Theme.Space.s) {
                        Text(model.config["transcripts_dir"]?.string ?? "–")
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(.secondary)
                        Button("Choose…") { chooseFolder("transcripts_dir") }
                    }
                }
            } header: {
                Text("Storage")
            } footer: {
                Text("iCloud Drive sync arrives with storage v2.")
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func chooseFolder(_ key: String) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            model.set(key, .string(url.path))
        }
    }
}

private struct TranscriptionPane: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Pane {
            Section {
                Picker("Model", selection: model.string("model", default: "parakeet-tdt-v3")) {
                    ForEach(model.catalog.asrModels) { m in
                        HStack {
                            Text(m.label ?? m.id)
                            if m.installed != true { Text("· download").foregroundStyle(.secondary) }
                        }
                        .tag(m.id)
                    }
                }
                Picker("Language", selection: model.optionalString("language")) {
                    ForEach(model.catalog.languages) { l in Text(l.label).tag(l.code) }
                }
            }
            Section {
                Toggle("Record system audio", isOn: model.bool("capture_system", default: true))
            } footer: {
                Text("Captures the other participants via ScreenCaptureKit. Needs Screen Recording permission.")
                    .foregroundStyle(.secondary)
            }
            Section("Accuracy") {
                Toggle("Drop silence (voice activity detection)", isOn: model.bool("vad", default: true))
                Toggle("Word-level timestamps", isOn: model.bool("word_timestamps", default: true))
            }
        }
    }
}

private struct SpeakersPane: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Pane {
            Section {
                Toggle("Detect speakers", isOn: model.bool("person_detection"))
                Picker("Number of speakers", selection: model.optionalInt("num_speakers")) {
                    Text("Auto-detect").tag(Int?.none)
                    ForEach(1...8, id: \.self) { i in Text("\(i)").tag(Int?.some(i)) }
                }
                .disabled(!(model.config["person_detection"]?.bool ?? false))
            }
            Section {
                TextField("My name", text: model.string("my_name"), prompt: Text("Me"))
                Toggle("Name speakers from the conversation", isOn: model.bool("name_speakers", default: true))
            } footer: {
                Text("Replaces “Speaker 2” with a name only when the transcript clearly says it.")
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct SummaryPane: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Pane {
            Section {
                Toggle("Summarize after each recording", isOn: model.bool("auto_summarize", default: true))
                Picker("Model", selection: model.string("summary_model", default: "auto")) {
                    Text("Automatic (best that fits)").tag("auto")
                    ForEach(model.catalog.summaryModels) { m in
                        SummaryModelLabel(model: m).tag(m.id)
                            .selectionDisabled(m.fits == false)
                    }
                }
                .disabled(!(model.config["auto_summarize"]?.bool ?? true))
            }
            Section {
                PromptEditor()
            } header: {
                Text("Prompt")
            } footer: {
                Text("Leave unchanged to use the built-in prompt, tuned for K2 Horizon.")
                    .foregroundStyle(.secondary)
            }
            Section {
                LabeledContent("Folder") {
                    HStack(spacing: Theme.Space.s) {
                        Text(model.config["gguf_folder"]?.string ?? "~/models")
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(.secondary)
                        Button("Choose…", action: chooseGGUFFolder)
                    }
                }
            } header: {
                Text("Local GGUF models")
            } footer: {
                Text("GGUF files in this folder (and its subfolders) appear in the model list and run with llama.cpp's llama-server.")
                    .foregroundStyle(.secondary)
            }
            DownloadedModelsSection()
        }
    }

    private func chooseGGUFFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url {
            model.set("gguf_folder", .string(url.path))
        }
    }
}

/// Models meetre downloaded, with sizes and a delete button. Other apps'
/// models in the shared Hugging Face cache and local GGUF files aren't listed.
private struct DownloadedModelsSection: View {
    @Environment(AppModel.self) private var model
    @State private var models: [DownloadedModel] = []
    @State private var confirm: DownloadedModel?

    var body: some View {
        Section {
            if models.isEmpty {
                Text("No downloaded models.").foregroundStyle(.secondary)
            }
            ForEach(models) { m in
                HStack {
                    VStack(alignment: .leading, spacing: Theme.Space.xxs) {
                        Text(m.label)
                        Text(m.kind == "asr" ? "Transcription"
                             : m.kind == "legacy" ? "MLX — no longer used" : "Summary (GGUF)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(ByteCountFormatter.string(fromByteCount: Int64(m.sizeBytes), countStyle: .file))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                    Button("Delete", role: .destructive) { confirm = m }
                        .disabled(m.inUse == true)
                        .help(m.inUse == true ? "In use — pick another model first" : "Delete this download")
                }
            }
        } header: {
            Text("Downloaded models")
        } footer: {
            Text("A deleted model downloads again if you pick it later.")
                .foregroundStyle(.secondary)
        }
        .task { models = await model.downloadedModels() }
        .confirmationDialog("Delete \(confirm?.label ?? "")?", isPresented: Binding(
            get: { confirm != nil }, set: { if !$0 { confirm = nil } }
        )) {
            Button("Delete", role: .destructive) {
                guard let m = confirm else { return }
                Task {
                    await model.deleteModel(m.id)
                    models = await model.downloadedModels()
                }
            }
        } message: {
            Text("Frees \(ByteCountFormatter.string(fromByteCount: Int64(confirm?.sizeBytes ?? 0), countStyle: .file)).")
        }
    }
}

private struct SummaryModelLabel: View {
    let model: SummaryModel

    var body: some View {
        HStack {
            Text(model.label ?? model.id)
            if let gb = model.sizeGb { Text(String(format: "%.1f GB", gb)).foregroundStyle(.secondary) }
            if model.recommended == true { Text("· recommended").foregroundStyle(.secondary) }
            if model.local == true { Text("· GGUF").foregroundStyle(.secondary) }
            if model.fits == false { Text("· too large").foregroundStyle(.secondary) }
        }
    }
}

private struct PromptEditor: View {
    @Environment(AppModel.self) private var model
    @State private var text = ""
    @State private var loaded = false

    var body: some View {
        VStack(alignment: .trailing, spacing: Theme.Space.s) {
            TextEditor(text: $text)
                .font(.callout)
                .frame(minHeight: 140, maxHeight: 220)
                .scrollContentBackground(.hidden)
                .padding(Theme.Space.xs)
                .background(.background, in: RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous))
            HStack {
                Button("Reset to default") {
                    text = model.catalog.defaultPrompt ?? ""
                    model.set("summary_prompt", .string(""))
                }
                .disabled((model.config["summary_prompt"]?.string ?? "").isEmpty)
                Spacer()
                Button("Save") {
                    let isDefault = text.trimmingCharacters(in: .whitespacesAndNewlines)
                        == (model.catalog.defaultPrompt ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                    model.set("summary_prompt", .string(isDefault ? "" : text))
                }
                .buttonStyle(.glassProminent)
            }
        }
        .onAppear {
            guard !loaded else { return }
            loaded = true
            let custom = model.config["summary_prompt"]?.string ?? ""
            text = custom.isEmpty ? (model.catalog.defaultPrompt ?? "") : custom
        }
    }
}

private struct IntegrationsPane: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Pane {
            Section {
                Toggle("Save each meeting to Apple Notes", isOn: model.bool("auto_notes", default: true))
            } header: {
                Label("Apple Notes", systemImage: "note.text")
            }
            Section {
                SecureField("Hugging Face token", text: model.string("hf_token"), prompt: Text("hf_…"))
            } header: {
                Label("Speaker detection model", systemImage: "person.2")
            } footer: {
                Text("pyannote needs a free Hugging Face token to download its model once. It stays on this Mac.")
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct AdvancedPane: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Pane {
            Section("Engine") {
                LabeledContent("Status") {
                    HStack(spacing: Theme.Space.s) {
                        EngineDot()
                        Text(status).foregroundStyle(.secondary)
                    }
                }
                LabeledContent("Version", value: model.engineVersion ?? "–")
                LabeledContent("Install", value: EngineClient.repoRoot.path)
                Button("Restart engine", action: model.restartEngine)
            }
            ErrorLogSection()
            Section {
                LabeledContent("Privacy") {
                    Label("Audio and text never leave this Mac", systemImage: "lock.shield")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var status: String {
        switch model.connection {
        case .ready: "Ready"
        case .connecting: "Starting…"
        case .failed(let why): why
        }
    }
}

/// Settings ▸ Advanced ▸ Error log: recent problems + one-click report.
private struct ErrorLogSection: View {
    @Environment(AppModel.self) private var model
    @State private var copied = false

    var body: some View {
        let log = ErrorLog.shared
        Section {
            if log.entries.isEmpty {
                Label("No errors this session", systemImage: "checkmark.circle")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(log.entries.suffix(5).reversed()) { e in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(e.message)
                            .font(.callout)
                            .lineLimit(3)
                            .textSelection(.enabled)
                        Text("\(e.date.formatted(date: .omitted, time: .standard)) · \(e.source)")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            HStack {
                Button {
                    log.copyReport(model: model)
                    withAnimation(Theme.smooth) { copied = true }
                    Task { try? await Task.sleep(for: .seconds(2)); withAnimation(Theme.smooth) { copied = false } }
                } label: {
                    Label(copied ? "Copied" : "Copy error log",
                          systemImage: copied ? "checkmark" : "doc.on.doc")
                        .contentTransition(.symbolEffect(.replace))
                }
                Button("Open log folder") {
                    NSWorkspace.shared.open(ErrorLog.folder)
                }
            }
        } header: {
            Text("Error log")
        } footer: {
            Text("The report has versions, settings and error messages — never transcripts or summaries. Paste it into a bug report.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

#Preview {
    SettingsView()
        .environment(AppModel(fixtures: true))
}
