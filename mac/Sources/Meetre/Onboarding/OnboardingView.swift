import SwiftUI

/// First-launch experience: a few words, soft sounds, one Install button.
/// See docs/DESIGN.md §10 (Onboarding) and §11 (Sound).
struct OnboardingView: View {
    enum Step: Int, CaseIterable { case hello, listens, writes, stays, setup, installing, permissions, done }

    @Environment(AppModel.self) private var model
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var step: Step
    @State private var options = InstallOptions()
    @State private var installer = Installer()
    @State private var muted = false
    @State private var heard = 0  // install steps already chimed
    let alreadyInstalled: Bool
    let autoplay: Bool
    let finish: (_ openLibrary: Bool) -> Void

    init(alreadyInstalled: Bool, start: Step = .hello, autoplay: Bool = true,
         finish: @escaping (_ openLibrary: Bool) -> Void) {
        self.alreadyInstalled = alreadyInstalled
        self.autoplay = autoplay
        self.finish = finish
        _step = State(initialValue: start)
    }

    var body: some View {
        ZStack {
            Aurora(animated: !reduceMotion)
                .ignoresSafeArea()

            Group {
                switch step {
                case .hello: Hello()
                case .listens: Beat(symbol: "waveform", words: "It listens.", effect: .variable)
                case .writes: Beat(symbol: "text.quote", words: "It writes it down.", effect: .bounce)
                case .stays: Beat(symbol: "lock.shield", words: "And it stays on this Mac.", effect: .pulse)
                case .setup: SetupCard(options: $options, alreadyInstalled: alreadyInstalled, install: install)
                case .installing: Installing(installer: installer, retry: install)
                case .permissions: PermissionsCard { go(.done); Chimes.shared.success() }
                case .done: Done(finish: finish)
                }
            }
            .id(step)
            .transition(reduceMotion ? .opacity
                        : .asymmetric(insertion: .opacity.combined(with: .scale(scale: 0.96)).combined(with: .offset(y: 12)),
                                      removal: .opacity.combined(with: .scale(scale: 1.03))))

            Corner(muted: $muted, showsClose: alreadyInstalled && step.rawValue < Step.installing.rawValue,
                   close: { finish(false) })
        }
        .contentShape(Rectangle())
        .onTapGesture { if step.rawValue < Step.setup.rawValue { advance() } }
        .onKeyPress(.return) { if step.rawValue < Step.setup.rawValue { advance(); return .handled }; return .ignored }
        .onKeyPress(.space) { if step.rawValue < Step.setup.rawValue { advance(); return .handled }; return .ignored }
        .onChange(of: muted) { Chimes.shared.muted = muted }
        .onChange(of: installer.stepsDone) { _, n in
            while heard < n { Chimes.shared.progress(heard); heard += 1 }
        }
        .onChange(of: installer.phase) { _, phase in
            switch phase {
            case .done:
                if Permissions.allGranted {
                    Chimes.shared.success()
                    go(.done)
                } else {
                    Chimes.shared.reveal()
                    go(.permissions)
                }
                Chimes.shared.shimmer(count: 10, over: 1.2, velocity: 0.14)
            case .failed: Chimes.shared.failure()
            default: break
            }
        }
        .task { if autoplay { await intro() } }
        // Always dark, independent of the system appearance: the scenes sit
        // on a dimmed, blurred desktop (preferredColorScheme alone doesn't
        // reach a hosting view in a borderless window).
        .foregroundStyle(.white)
        .environment(\.colorScheme, .dark)
        .preferredColorScheme(.dark)
    }

    // MARK: Flow

    /// Auto-plays hello → three beats; a click or Return skips ahead.
    private func intro() async {
        try? await Task.sleep(for: .milliseconds(450))
        Chimes.shared.pad()
        Chimes.shared.hello()
        try? await Task.sleep(for: .milliseconds(650))
        Chimes.shared.shimmer()
        let holds: [Step: Double] = [.hello: 2.4, .listens: 1.7, .writes: 1.7, .stays: 2.1]
        while step.rawValue < Step.setup.rawValue {
            let current = step
            try? await Task.sleep(for: .seconds(holds[current] ?? 2))
            if step == current { advance() }  // not skipped meanwhile
        }
    }

    private func advance() {
        guard let next = Step(rawValue: step.rawValue + 1) else { return }
        cue(for: next)
        go(next)
    }

    /// Every scene has its own little sound.
    private func cue(for next: Step) {
        let c = Chimes.shared
        switch next {
        case .listens: c.listen()
        case .writes: c.typing()
        case .stays: c.lock()
        case .setup: c.reveal()
        default: c.step()
        }
    }

    private func go(_ next: Step) {
        withAnimation(reduceMotion ? .easeInOut(duration: 0.3) : .smooth(duration: 0.7)) { step = next }
    }

    private func install() {
        Chimes.shared.begin()
        heard = installer.stepsDone
        go(.installing)
        Task { await installer.run(options, model: model) }
    }
}

// MARK: - Scenes

private struct Hello: View {
    @State private var shown = false

    var body: some View {
        VStack(spacing: 28) {
            Sparkle(size: 120)
            Text("Hi.")
                .font(.system(size: 96, weight: .semibold, design: .rounded))
                .opacity(shown ? 1 : 0)
                .blur(radius: shown ? 0 : 12)
        }
        .onAppear { withAnimation(.smooth(duration: 1.1).delay(0.25)) { shown = true } }
    }
}

private struct Beat: View {
    enum Effect { case variable, bounce, pulse }
    let symbol: String
    let words: String
    let effect: Effect
    @State private var shown = false

    var body: some View {
        VStack(spacing: 30) {
            icon
                .font(.system(size: 88, weight: .light))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(.white)
                .frame(height: 110)
            Text(words)
                .font(.system(size: 52, weight: .semibold, design: .rounded))
                .opacity(shown ? 1 : 0)
                .offset(y: shown ? 0 : 8)
        }
        .onAppear { withAnimation(.smooth(duration: 0.8).delay(0.15)) { shown = true } }
    }

    @ViewBuilder private var icon: some View {
        let image = Image(systemName: symbol)
        switch effect {
        case .variable: image.symbolEffect(.variableColor.iterative.reversing, options: .repeating)
        case .bounce: image.symbolEffect(.bounce, options: .repeat(2), value: shown)
        case .pulse: image.symbolEffect(.pulse, options: .repeating)
        }
    }
}

private struct SetupCard: View {
    @Binding var options: InstallOptions
    let alreadyInstalled: Bool
    let install: () -> Void
    @FocusState private var tokenFocused: Bool

    var body: some View {
        VStack(spacing: 34) {
            VStack(spacing: 10) {
                Sparkle(size: 56)
                Text("Let's set you up.")
                    .font(.system(size: 40, weight: .semibold, design: .rounded))
            }

            VStack(spacing: 0) {
                Option(symbol: "power", title: "Start at login", isOn: $options.startAtLogin)
                Divider().opacity(0.4)
                Option(symbol: "phone.badge.waveform", title: "Offer to record Teams calls", isOn: $options.askForCalls)
                Divider().opacity(0.4)
                Option(symbol: "person.2", title: "Recognize who speaks", isOn: $options.recognizeSpeakers)
                if options.recognizeSpeakers {
                    HStack(spacing: 8) {
                        TextField("Hugging Face token (optional)", text: $options.huggingFaceToken)
                            .textFieldStyle(.plain)
                            .focused($tokenFocused)
                            .padding(.horizontal, 12).padding(.vertical, 8)
                            .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: Theme.Radius.card))
                        Link(destination: URL(string: "https://huggingface.co/settings/tokens")!) {
                            Image(systemName: "questionmark.circle")
                        }
                        .help("Free token — accept the pyannote model terms first")
                    }
                    .padding(.horizontal, 18).padding(.bottom, 12)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
                Divider().opacity(0.4)
                Option(symbol: "arrow.down.circle", title: "Download the AI now", detail: "≈ 8 GB",
                       isOn: $options.downloadModels)
            }
            .frame(width: 420)
            .glassEffect(.regular, in: RoundedRectangle(cornerRadius: Theme.Radius.hero))
            .animation(.smooth(duration: 0.3), value: options.recognizeSpeakers)

            InstallButton(title: alreadyInstalled ? "Set up" : "Install", action: install)

            Label("Everything runs on-device", systemImage: "lock.shield")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }
}

private struct Option: View {
    let symbol: String
    let title: String
    var detail: String? = nil
    @Binding var isOn: Bool

    var body: some View {
        Toggle(isOn: $isOn.animation(.smooth(duration: 0.25))) {
            HStack(spacing: 12) {
                Image(systemName: symbol)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 22)
                    .foregroundStyle(isOn ? Theme.accent : .secondary)
                Text(title)
                if let detail { Text(detail).foregroundStyle(.tertiary) }
            }
            .font(.system(size: 15))
        }
        .toggleStyle(.switch)
        .tint(Theme.accent)
        .padding(.horizontal, 18).padding(.vertical, 13)
        .onChange(of: isOn) { _, on in Chimes.shared.toggle(on) }
    }
}

private struct InstallButton: View {
    let title: String
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 20, weight: .semibold, design: .rounded))
                .frame(width: 220, height: 58)
                .contentShape(Capsule())
        }
        .buttonStyle(.glassProminent)
        .tint(Theme.accent)
        .keyboardShortcut(.defaultAction)
        .scaleEffect(hovering ? 1.04 : 1)
        .shadow(color: Theme.accent.opacity(hovering ? 0.55 : 0.3), radius: hovering ? 28 : 18)
        .animation(.bouncy(duration: 0.35), value: hovering)
        .onHover { h in
            hovering = h
            if h { Chimes.shared.tick() }
        }
    }
}

private struct Installing: View {
    let installer: Installer
    let retry: () -> Void

    var body: some View {
        VStack(spacing: 30) {
            if case .failed(let why) = installer.phase {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 64, weight: .light))
                    .foregroundStyle(Theme.warn)
                Text("That didn't work.")
                    .font(.system(size: 40, weight: .semibold, design: .rounded))
                Text(why)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 460)
                    .textSelection(.enabled)
                InstallButton(title: "Try again", action: retry)
            } else {
                ProgressRing(progress: installer.progress)
                Text(installer.label)
                    .font(.system(size: 26, weight: .medium, design: .rounded))
                    .contentTransition(.opacity)
                    .animation(.smooth, value: installer.label)
                    .frame(height: 34)
            }
        }
    }
}

private struct Done: View {
    let finish: (_ openLibrary: Bool) -> Void
    @State private var bob = false

    var body: some View {
        ZStack(alignment: .topTrailing) {
            VStack(spacing: 26) {
                Sparkle(size: 110)
                Text("You're set.")
                    .font(.system(size: 64, weight: .semibold, design: .rounded))
                Text("Find meetre in your menu bar.")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                HStack(spacing: 14) {
                    InstallButton(title: "Start") { finish(false) }
                    Button("Open library") { finish(true) }
                        .buttonStyle(.glass)
                        .controlSize(.large)
                }
                .padding(.top, 8)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // A gentle pointer toward the status item, top-right.
            Image(systemName: "arrow.up.right")
                .font(.system(size: 34, weight: .semibold))
                .foregroundStyle(.white.opacity(0.85))
                .offset(x: bob ? 0 : -8, y: bob ? 0 : 8)
                .padding(.top, 18).padding(.trailing, 150)
                .onAppear {
                    withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) { bob = true }
                }
        }
    }
}

/// Microphone (your voice) and Screen Recording (the others — macOS gates
/// system audio behind it). Granted to Meetre.app, which the engine and the
/// capture helper inherit as their "responsible" app.
private struct PermissionsCard: View {
    let next: () -> Void
    @State private var mic = Permissions.microphone
    @State private var screen = Permissions.screen
    @State private var askedScreen = false

    var body: some View {
        VStack(spacing: 30) {
            VStack(spacing: 10) {
                Image(systemName: "hand.raised")
                    .font(.system(size: 52, weight: .light))
                    .symbolRenderingMode(.hierarchical)
                Text("Two quick permissions.")
                    .font(.system(size: 40, weight: .semibold, design: .rounded))
            }
            VStack(spacing: 0) {
                PermissionRow(symbol: "mic.fill", title: "Microphone", detail: "your voice",
                              granted: mic == .granted, denied: mic == .denied) {
                    Task { mic = await Permissions.requestMicrophone(); Chimes.shared.toggle(mic == .granted) }
                }
                Divider().opacity(0.4)
                PermissionRow(symbol: "speaker.wave.2.fill", title: "System audio", detail: "the others",
                              granted: screen == .granted, denied: false,
                              actionTitle: askedScreen ? "Restart meetre" : "Allow") {
                    if askedScreen {
                        Permissions.relaunch(resumeAt: "permissions")
                    } else {
                        Permissions.requestScreen()
                        askedScreen = true
                    }
                }
            }
            .frame(width: 440)
            .glassEffect(.regular, in: RoundedRectangle(cornerRadius: Theme.Radius.hero))

            Text(askedScreen && screen != .granted
                 ? "Turned it on in System Settings? macOS needs a restart of meetre to apply it."
                 : "macOS calls the second one “Screen Recording” — meetre only keeps the sound.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)

            InstallButton(title: mic == .granted && screen == .granted ? "Continue" : "Later", action: next)
        }
        // Settings changes happen outside the app: re-check while visible.
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                let m = Permissions.microphone, s = Permissions.screen
                if m != mic || s != screen {
                    if s == .granted && screen != .granted { Chimes.shared.toggle(true) }
                    mic = m; screen = s
                }
            }
        }
    }
}

private struct PermissionRow: View {
    let symbol: String
    let title: String
    let detail: String
    let granted: Bool
    let denied: Bool
    var actionTitle = "Allow"
    let action: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: symbol)
                .symbolRenderingMode(.hierarchical)
                .frame(width: 22)
                .foregroundStyle(granted ? Theme.live : Theme.accent)
            Text(title)
            Text(detail).foregroundStyle(.tertiary)
            Spacer()
            if granted {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(Theme.live)
                    .transition(.scale.combined(with: .opacity))
            } else if denied {
                Button("Open Settings") { Permissions.openMicrophoneSettings() }
                    .buttonStyle(.glass)
            } else {
                Button(actionTitle, action: action)
                    .buttonStyle(.glassProminent)
                    .tint(Theme.accent)
            }
        }
        .font(.system(size: 15))
        .padding(.horizontal, 18).padding(.vertical, 14)
        .animation(.bouncy(duration: 0.35), value: granted)
    }
}

// MARK: - Pieces

/// The meetre mark: a sparkle that breathes and glows.
private struct Sparkle: View {
    let size: CGFloat
    @State private var breathe = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Image(systemName: "sparkle")
            .font(.system(size: size, weight: .light))
            .foregroundStyle(LinearGradient(colors: [.white, Color(nsColor: NSColor(hex: 0xB9B9FF))],
                                            startPoint: .top, endPoint: .bottom))
            .shadow(color: Color(nsColor: NSColor(hex: 0x8B8BF5)).opacity(0.9), radius: breathe ? 34 : 18)
            .scaleEffect(breathe ? 1.04 : 0.97)
            .rotationEffect(.degrees(breathe ? 8 : -4))
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(.easeInOut(duration: 2.6).repeatForever(autoreverses: true)) { breathe = true }
            }
    }
}

private struct ProgressRing: View {
    let progress: Double?
    @State private var spin = false

    var body: some View {
        ZStack {
            Circle().stroke(.white.opacity(0.12), lineWidth: 10)
            Circle()
                .trim(from: 0, to: max(0.02, progress ?? 0.25))
                .stroke(AngularGradient(colors: [Color(nsColor: NSColor(hex: 0x8B8BF5)), Color(nsColor: NSColor(hex: 0x3DD68C)),
                                                 Color(nsColor: NSColor(hex: 0x8B8BF5))], center: .center),
                        style: StrokeStyle(lineWidth: 10, lineCap: .round))
                .rotationEffect(.degrees(progress == nil ? (spin ? 270 : -90) : -90))
                .animation(.smooth(duration: 0.6), value: progress)
            Image(systemName: "sparkle")
                .font(.system(size: 46, weight: .light))
                .rotationEffect(.degrees(spin ? 360 : 0))
            if let progress {
                Text("\(Int(progress * 100))%")
                    .font(.system(size: 15, weight: .medium, design: .rounded).monospacedDigit())
                    .foregroundStyle(.secondary)
                    .contentTransition(.numericText())
                    .offset(y: 54)
            }
        }
        .frame(width: 170, height: 170)
        .onAppear { withAnimation(.linear(duration: 8).repeatForever(autoreverses: false)) { spin = true } }
    }
}

/// Mute toggle (and close, when meetre is already installed).
private struct Corner: View {
    @Binding var muted: Bool
    let showsClose: Bool
    let close: () -> Void

    var body: some View {
        VStack {
            Spacer()
            HStack(spacing: 10) {
                Spacer()
                if showsClose {
                    Button("Skip", action: close)
                        .buttonStyle(.glass)
                        .help("Skip setup")
                }
                Button { muted.toggle() } label: {
                    Image(systemName: muted ? "speaker.slash" : "speaker.wave.2")
                        .contentTransition(.symbolEffect(.replace))
                        .frame(width: 18)
                }
                .buttonStyle(.glass)
                .help(muted ? "Sound on" : "Sound off")
            }
            .padding(28)
        }
    }
}

/// Slowly drifting, translucent color field over the blurred desktop.
private struct Aurora: View {
    let animated: Bool

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30, paused: !animated)) { ctx in
            let t = animated ? ctx.date.timeIntervalSinceReferenceDate : 0
            let s = { (speed: Double, phase: Double) in Float(0.5 + 0.18 * sin(t * speed + phase)) }
            MeshGradient(
                width: 3, height: 3,
                points: [
                    [0, 0], [s(0.11, 0), 0], [1, 0],
                    [0, s(0.13, 1)], [s(0.07, 2), s(0.09, 3)], [1, s(0.1, 4)],
                    [0, 1], [s(0.12, 5), 1], [1, 1],
                ],
                colors: [
                    Color(nsColor: NSColor(hex: 0x2B2B6E)).opacity(0.55), .clear, Color(nsColor: NSColor(hex: 0x5B5BD6)).opacity(0.45),
                    .clear, Color(nsColor: NSColor(hex: 0x7A4BD6)).opacity(0.35), .clear,
                    Color(nsColor: NSColor(hex: 0x1F7A5A)).opacity(0.4), .clear, Color(nsColor: NSColor(hex: 0x2B2B6E)).opacity(0.5),
                ]
            )
            .background(Color.black.opacity(0.28))
        }
    }
}
