import AVFoundation

/// Soft, synthesized UI sounds for onboarding — no audio assets.
///
/// Each note is a small bell: a sine fundamental plus two quiet inharmonic
/// partials under an exponential decay, rendered into a buffer and played
/// through a shared `AVAudioEngine` with a touch of reverb. Notes come from a
/// major pentatonic scale, so any combination sounds friendly.
@MainActor
final class Chimes {
    static let shared = Chimes()

    var muted = false

    // The graph is rebuilt lazily and discarded whenever the audio
    // configuration changes (output device switch, a call app taking over the
    // device …). Playing into a stale graph raises an Objective-C exception
    // ("player started when in a disconnected state") that Swift can't catch,
    // so `play` only ever touches a freshly built, running engine.
    private var engine: AVAudioEngine?
    private var players: [AVAudioPlayerNode] = []
    private var nextPlayer = 0
    private var format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 2)!
    private var observer: NSObjectProtocol?

    private init() {
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.teardown() }
        }
    }

    private func teardown() {
        engine?.stop()
        engine = nil
        players = []
    }

    /// A running engine, (re)built at the output device's sample rate.
    private func runningEngine() -> AVAudioEngine? {
        if let engine, engine.isRunning { return engine }
        teardown()
        let e = AVAudioEngine()
        let hw = e.outputNode.outputFormat(forBus: 0).sampleRate
        guard let fmt = AVAudioFormat(standardFormatWithSampleRate: hw > 0 ? hw : 44_100, channels: 2) else { return nil }
        let reverb = AVAudioUnitReverb()
        reverb.loadFactoryPreset(.mediumHall)
        reverb.wetDryMix = 28
        e.attach(reverb)
        e.connect(reverb, to: e.mainMixerNode, format: fmt)
        var voices: [AVAudioPlayerNode] = []
        for _ in 0..<16 {  // voice 0 = ambient pad; the rest overlap freely
            let p = AVAudioPlayerNode()
            e.attach(p)
            e.connect(p, to: reverb, format: fmt)
            voices.append(p)
        }
        e.mainMixerNode.outputVolume = 0.55
        e.prepare()
        do { try e.start() } catch { return nil }
        engine = e
        players = voices
        format = fmt
        return e
    }

    // MARK: Cues

    /// The opening "hello": a slow, rising major-ninth arpeggio.
    func hello() { arpeggio([0, 4, 7, 11, 14], gap: 0.16, base: 60, velocity: 0.5, decay: 2.4) }
    /// Advancing to the next scene (generic).
    func step() { play(midi: 79, velocity: 0.28, decay: 0.9) }
    /// Glittering sparkle: a few random high pentatonic notes.
    func shimmer(count: Int = 7, over seconds: Double = 0.7, velocity: Float = 0.12) {
        let scale = [0, 2, 4, 7, 9]
        for i in 0..<count {
            let midi = 84 + scale.randomElement()! + 12 * Int.random(in: 0...1)
            let at = seconds * Double(i) / Double(count) + Double.random(in: 0...0.04)
            later(at) { $0.play(midi: midi, velocity: velocity * Float.random(in: 0.6...1), decay: 0.5) }
        }
    }
    /// "It listens.": a quick rising figure, like a voice level coming up.
    func listen() { arpeggio([0, 4, 7, 12], gap: 0.07, base: 69, velocity: 0.22, decay: 0.8) }
    /// "It writes it down.": a sprinkle of tiny typing ticks.
    func typing() {
        var at = 0.0
        for _ in 0..<9 {
            let midi = [88, 91, 93, 96].randomElement()!
            later(at) { $0.play(midi: midi, velocity: Float.random(in: 0.05...0.1), decay: 0.06) }
            at += Double.random(in: 0.05...0.13)
        }
        later(at + 0.1) { $0.play(midi: 81, velocity: 0.2, decay: 0.9) }  // the "done writing" note
    }
    /// "It stays on this Mac.": a warm low chord with a soft latch click.
    func lock() {
        play(midi: 100, velocity: 0.07, decay: 0.03)
        later(0.06) { $0.arpeggio([0, 7, 12], gap: 0.03, base: 50, velocity: 0.32, decay: 1.6) }
    }
    /// The setup card blooming open: a soft major-seventh chord.
    func reveal() { arpeggio([0, 4, 7, 11, 16], gap: 0.045, base: 60, velocity: 0.2, decay: 2.2) }
    /// A soft ambient pad that breathes under the intro (one long note).
    func pad(seconds: Double = 12) {
        guard !muted, let engine = runningEngine(), let player = players.first,
              let buffer = padBuffer(seconds: seconds),
              engine.isRunning, !engine.outputConnectionPoints(for: player, outputBus: 0).isEmpty else { return }
        player.scheduleBuffer(buffer, at: nil, options: [.interrupts])
        if !player.isPlaying { player.play() }
    }
    /// Hovering a big button: a tiny, high tick.
    func tick() { play(midi: 96, velocity: 0.08, decay: 0.12) }
    /// Toggling an option.
    func toggle(_ on: Bool) { play(midi: on ? 84 : 79, velocity: 0.16, decay: 0.25) }
    /// Install started.
    func begin() { arpeggio([0, 7], gap: 0.09, base: 67, velocity: 0.3, decay: 1.0) }
    /// One install step finished.
    func progress(_ index: Int) {
        let scale = [0, 2, 4, 7, 9, 12, 14, 16]
        play(midi: 72 + scale[index % scale.count], velocity: 0.18, decay: 0.6)
    }
    /// Everything done: a bright, resolving flourish.
    func success() { arpeggio([0, 4, 7, 12, 16, 19, 24], gap: 0.075, base: 64, velocity: 0.38, decay: 2.8) }
    /// Something failed: two soft descending notes.
    func failure() { arpeggio([0, -5], gap: 0.18, base: 64, velocity: 0.3, decay: 1.2) }

    // MARK: Synthesis

    private func later(_ seconds: Double, _ body: @escaping @MainActor (Chimes) -> Void) {
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { [weak self] in
            MainActor.assumeIsolated { if let self { body(self) } }
        }
    }

    /// Detuned sines (C major add9, low) with a slow swell and long release.
    private func padBuffer(seconds: Double) -> AVAudioPCMBuffer? {
        let sr = format.sampleRate
        let frames = AVAudioFrameCount(sr * seconds)
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames),
              let l = buf.floatChannelData?[0], let r = buf.floatChannelData?[1] else { return nil }
        buf.frameLength = frames
        let notes = [48, 55, 60, 62, 64].map { 440 * pow(2, Double($0 - 69) / 12) }
        let attack = 2.2, release = 3.0
        for n in 0..<Int(frames) {
            let t = Double(n) / sr
            let env = min(1, t / attack) * min(1, max(0, (seconds - t) / release))
            var left = 0.0, right = 0.0
            for (i, f) in notes.enumerated() {
                let wobble = 0.35 + 0.1 * Double(i)  // slow chorus, different per voice
                left += sin(2 * .pi * (f - wobble) * t)
                right += sin(2 * .pi * (f + wobble) * t)
            }
            let gain = 0.035 * env * (0.85 + 0.15 * sin(2 * .pi * 0.18 * t))  // gentle breathing
            l[n] = Float(left * gain)
            r[n] = Float(right * gain)
        }
        return buf
    }

    private func arpeggio(_ intervals: [Int], gap: Double, base: Int, velocity: Float, decay: Double) {
        for (i, iv) in intervals.enumerated() {
            let v = velocity * (1 - Float(i) * 0.06)
            DispatchQueue.main.asyncAfter(deadline: .now() + gap * Double(i)) { [weak self] in
                self?.play(midi: base + iv, velocity: v, decay: decay)
            }
        }
    }

    private func play(midi: Int, velocity: Float, decay: Double) {
        guard !muted, let engine = runningEngine(), !players.isEmpty,
              let buffer = bell(frequency: 440 * pow(2, Double(midi - 69) / 12),
                                velocity: velocity, decay: decay) else { return }
        // Voice 0 is reserved for the pad, so a burst of notes never cuts it.
        let player = players[1 + nextPlayer % (players.count - 1)]
        nextPlayer = (nextPlayer + 1) % (players.count - 1)
        // Never start a node that isn't wired to a running engine.
        guard engine.isRunning, player.engine === engine,
              !engine.outputConnectionPoints(for: player, outputBus: 0).isEmpty else { return }
        player.scheduleBuffer(buffer, at: nil, options: [.interrupts])
        if !player.isPlaying { player.play() }
    }

    private func bell(frequency f: Double, velocity: Float, decay: Double) -> AVAudioPCMBuffer? {
        let sr = format.sampleRate
        let frames = AVAudioFrameCount(sr * (decay + 0.05))
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames),
              let l = buf.floatChannelData?[0], let r = buf.floatChannelData?[1] else { return nil }
        buf.frameLength = frames
        let attack = 0.006 * sr
        // Partials: fundamental, a soft octave and a bell-like 2.76× overtone.
        let partials: [(ratio: Double, gain: Double, decayScale: Double)] =
            [(1, 1, 1), (2, 0.22, 0.55), (2.76, 0.09, 0.35)]
        for n in 0..<Int(frames) {
            let t = Double(n) / sr
            var s = 0.0
            for p in partials {
                s += p.gain * sin(2 * .pi * f * p.ratio * t) * exp(-t / (decay * p.decayScale / 4))
            }
            let env = min(1, Double(n) / attack)
            let v = Float(s * env) * velocity * 0.6
            l[n] = v
            r[n] = v
        }
        return buf
    }
}
