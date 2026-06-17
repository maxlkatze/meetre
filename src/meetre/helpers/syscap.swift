// syscap — capture macOS *system audio* natively via ScreenCaptureKit.
//
// Usage:  syscap <output.wav>
//
// Captures everything playing through the system output (i.e. the other
// meeting participants) and writes it to <output.wav> as Float32 PCM. No
// loopback driver (BlackHole) required. Runs until it receives SIGTERM /
// SIGINT, then finalises the WAV and exits 0.
//
// Requires the "Screen Recording" permission (System Settings ▸ Privacy &
// Security ▸ Screen Recording) — on macOS, system-audio capture is gated
// behind the same permission as screen capture. The first run will prompt;
// grant it, then re-run.
//
// Prints "READY" to stdout once capture has actually started so the parent
// process can begin its timer in sync.

import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

// ---------------------------------------------------------------------------

let stderr = FileHandle.standardError
func log(_ s: String) { stderr.write((s + "\n").data(using: .utf8)!) }
func fail(_ s: String) -> Never { log("ERROR: " + s); exit(1) }

guard CommandLine.arguments.count >= 2 else {
    fail("missing output path — usage: syscap <output.wav>")
}
let outURL = URL(fileURLWithPath: CommandLine.arguments[1])

// ---------------------------------------------------------------------------

final class Capturer: NSObject, SCStreamOutput, SCStreamDelegate {
    let url: URL
    var audioFile: AVAudioFile?
    var stream: SCStream?
    let queue = DispatchQueue(label: "de.agendaro.meetre.syscap")
    private var announced = false

    init(url: URL) { self.url = url }

    func start() async {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: false)
        } catch {
            fail("could not query shareable content — grant Screen Recording "
                + "permission in System Settings ▸ Privacy & Security. (\(error))")
        }
        guard let display = content.displays.first else {
            fail("no display found to attach the audio capture to")
        }

        // Audio capture still needs a content filter built around a display,
        // even though we discard the video frames.
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = 48_000
        config.channelCount = 2
        config.excludesCurrentProcessAudio = true   // don't record ourselves
        // Minimal video — we never read it, but the stream insists on a size.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 6)

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        do {
            try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
            try await stream.startCapture()
        } catch {
            fail("failed to start system-audio capture: \(error)")
        }
        self.stream = stream
    }

    func stop() {
        let sem = DispatchSemaphore(value: 0)
        Task {
            try? await self.stream?.stopCapture()
            sem.signal()
        }
        sem.wait()
        audioFile = nil   // flush + close the WAV
    }

    // MARK: SCStreamOutput

    func stream(_ stream: SCStream,
                didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid,
              CMSampleBufferGetNumSamples(sampleBuffer) > 0,
              let fmtDesc = sampleBuffer.formatDescription,
              let asbdPtr = fmtDesc.audioStreamBasicDescription.map({ $0 })
        else { return }

        var asbd = asbdPtr
        guard let format = AVAudioFormat(streamDescription: &asbd) else { return }

        do {
            try sampleBuffer.withAudioBufferList { abl, _ in
                if audioFile == nil {
                    // Create the file lazily from the real incoming format so
                    // the on-disk WAV matches what we receive exactly.
                    audioFile = try AVAudioFile(
                        forWriting: url,
                        settings: format.settings,
                        commonFormat: .pcmFormatFloat32,
                        interleaved: format.isInterleaved)
                    if !announced { print("READY"); fflush(stdout); announced = true }
                }
                guard let pcm = AVAudioPCMBuffer(
                    pcmFormat: format, bufferListNoCopy: abl.unsafePointer)
                else { return }
                try audioFile?.write(from: pcm)
            }
        } catch {
            log("write error: \(error)")
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("stream stopped with error: \(error)")
    }
}

// ---------------------------------------------------------------------------

let capturer = Capturer(url: outURL)

// Graceful shutdown on SIGTERM / SIGINT (parent sends these on stop).
// The sources must outlive this scope, so keep strong references around.
signal(SIGTERM, SIG_IGN)
signal(SIGINT, SIG_IGN)
var signalSources: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT] {
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler {
        capturer.stop()
        exit(0)
    }
    src.resume()
    signalSources.append(src)
}

Task { await capturer.start() }

// Run forever until a signal handler calls exit().
dispatchMain()
