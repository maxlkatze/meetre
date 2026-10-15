// miccap — record the microphone through Apple's Voice Processing I/O.
//
// Usage:  miccap <output.wav>
//
// Voice Processing is the echo canceller FaceTime uses: it removes what the
// Mac itself plays from the speakers (e.g. the other side of a call) from the
// mic signal, plus noise suppression.
//
// OPT-IN ONLY (config `voice_processing_mic`): while it runs, macOS changes
// the mic for every app — measured: other apps hear it ~35 dB quieter, and
// the call's audio is ducked ~8 dB despite minimum ducking. Never use it
// while you're in a call others need to hear you in. Measured on a Mac
// speaker setup: the call's echo in the mic drops from +12.6 dB above the
// room to +0.1 dB. Ducking of other apps' audio is turned off so the call
// stays at full volume.
//
// Writes 16 kHz mono Float32 to <output.wav> (fixed, whatever the device
// runs at), prints "READY" once audio arrives, rebuilds itself when the audio
// configuration changes (a headset switching into call mode, a new default
// input), and finalises the file on SIGTERM / SIGINT (exit 0). Needs the
// Microphone permission of the app that launched it (Meetre.app).

import AVFoundation
import Foundation

let stderr = FileHandle.standardError
func log(_ s: String) { stderr.write((s + "\n").data(using: .utf8)!) }
func fail(_ s: String) -> Never { log("ERROR: " + s); exit(1) }

guard CommandLine.arguments.count >= 2 else { fail("usage: miccap <output.wav>") }
let outURL = URL(fileURLWithPath: CommandLine.arguments[1])
let fileFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                               channels: 1, interleaved: false)!

final class MicCapture {
    private var engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private var file: AVAudioFile?
    private var announced = false
    private let lock = NSLock()
    private var observer: NSObjectProtocol?

    func start() {
        do {
            file = try AVAudioFile(forWriting: outURL, settings: fileFormat.settings,
                                   commonFormat: .pcmFormatFloat32, interleaved: false)
        } catch {
            fail("cannot create \(outURL.path): \(error)")
        }
        if let error = run() { fail(error) }
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: nil, queue: .main
        ) { [weak self] _ in self?.restart() }
    }

    /// (Re)builds the engine on the current input device. nil = running.
    private func run() -> String? {
        let input = engine.inputNode
        do {
            try input.setVoiceProcessingEnabled(true)
        } catch {
            return "voice processing unavailable: \(error)"
        }
        if #available(macOS 14.0, *) {
            input.voiceProcessingOtherAudioDuckingConfiguration =
                AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                    enableAdvancedDucking: false, duckingLevel: .min)
        }
        // Hardware AGC off: it changes the mic level for ALL apps (the call
        // would hear you quieter). meetre levels the audio in software.
        input.isVoiceProcessingAGCEnabled = false
        let hw = input.outputFormat(forBus: 0)
        guard hw.sampleRate > 0, hw.channelCount > 0,
              let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: hw.sampleRate,
                                       channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: mono, to: fileFormat) else {
            return "no usable input format (\(hw))"
        }
        converter = conv
        input.installTap(onBus: 0, bufferSize: 1024, format: hw) { [weak self] buffer, _ in
            self?.write(buffer, mono: mono)
        }
        engine.prepare()
        do { try engine.start() } catch { return "could not start the microphone: \(error)" }
        return nil
    }

    /// The device changed under us: start over on whatever is current now.
    private func restart() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        engine = AVAudioEngine()
        for attempt in 1...5 {
            if let error = run() {
                log("restart attempt \(attempt) failed: \(error)")
                Thread.sleep(forTimeInterval: 0.5)
                engine = AVAudioEngine()
            } else {
                log("audio configuration changed — microphone restarted")
                return
            }
        }
    }

    private func write(_ buffer: AVAudioPCMBuffer, mono: AVAudioFormat) {
        guard buffer.frameLength > 0, let src = buffer.floatChannelData,
              let first = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: buffer.frameLength),
              let dst = first.floatChannelData, let converter else { return }
        // Voice processing may expose several channels; the first is the voice.
        first.frameLength = buffer.frameLength
        dst[0].update(from: src[0], count: Int(buffer.frameLength))
        let ratio = fileFormat.sampleRate / mono.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: capacity) else { return }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return first
        }
        guard error == nil, out.frameLength > 0 else { return }
        lock.lock(); defer { lock.unlock() }
        do { try file?.write(from: out) } catch { log("write error: \(error)") }
        if !announced { print("READY"); fflush(stdout); announced = true }
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        lock.lock(); file = nil; lock.unlock()   // flush + close the WAV
    }
}

let capture = MicCapture()
signal(SIGTERM, SIG_IGN)
signal(SIGINT, SIG_IGN)
var signalSources: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT] {
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler { capture.stop(); exit(0) }
    src.resume()
    signalSources.append(src)
}
capture.start()
dispatchMain()
