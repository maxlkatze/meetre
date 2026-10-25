import AVFoundation
import Observation

/// Plays a meeting's audio backup for the transcript tab and publishes the
/// playhead, so the transcript can highlight the word being spoken.
@MainActor @Observable
final class TranscriptPlayer {
    private(set) var time: Double = 0
    private(set) var duration: Double = 0
    private(set) var playing = false
    private(set) var url: URL?

    @ObservationIgnored private var player: AVPlayer?
    @ObservationIgnored private var observer: Any?
    @ObservationIgnored private var endObserver: NSObjectProtocol?

    func load(_ path: String?) {
        let next = path.map { URL(fileURLWithPath: $0) }
        guard next != url else { return }
        stop()
        url = next
        guard let next, FileManager.default.fileExists(atPath: next.path) else { url = nil; return }
        let item = AVPlayerItem(url: next)
        let p = AVPlayer(playerItem: item)
        observer = p.addPeriodicTimeObserver(forInterval: CMTime(seconds: 0.05, preferredTimescale: 600),
                                             queue: .main) { [weak self] t in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.time = t.seconds.isFinite ? t.seconds : 0
                if let d = self.player?.currentItem?.duration.seconds, d.isFinite { self.duration = d }
            }
        }
        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.playing = false }
        }
        player = p
        time = 0
        Task { [weak self] in
            guard let d = try? await item.asset.load(.duration), d.seconds.isFinite else { return }
            self?.duration = d.seconds
        }
    }

    func toggle() {
        guard let player else { return }
        if playing {
            player.pause()
        } else {
            if duration > 0, time >= duration - 0.05 { seek(0) }
            player.play()
        }
        playing.toggle()
    }

    /// Jump to `seconds` and keep playing (or start).
    func play(from seconds: Double) {
        seek(seconds)
        if !playing { toggle() }
    }

    func seek(_ seconds: Double) {
        time = max(0, seconds)
        player?.seek(to: CMTime(seconds: time, preferredTimescale: 600),
                     toleranceBefore: .zero, toleranceAfter: .zero)
    }

    func stop() {
        player?.pause()
        if let observer { player?.removeTimeObserver(observer) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        observer = nil; endObserver = nil; player = nil
        playing = false; time = 0; duration = 0
    }
}
