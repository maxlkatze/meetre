// Renders the 1024 px meetre app icon: indigo (accent token) squircle with the
// ✦ sparkles symbol — same mark as the menu-bar icon (src/meetre/icon.py).
import AppKit

let size: CGFloat = 1024
let out = CommandLine.arguments[1]
let img = NSImage(size: NSSize(width: size, height: size))
img.lockFocus()
let inset: CGFloat = 100
let rect = NSRect(x: inset, y: inset, width: size - 2 * inset, height: size - 2 * inset)
let path = NSBezierPath(roundedRect: rect, xRadius: 185, yRadius: 185)
NSGradient(colors: [NSColor(srgbRed: 0.42, green: 0.42, blue: 0.93, alpha: 1),
                    NSColor(srgbRed: 0.27, green: 0.26, blue: 0.70, alpha: 1)])!
    .draw(in: path, angle: -90)
NSColor.white.withAlphaComponent(0.18).setStroke()
path.lineWidth = 4
path.stroke()
let cfg = NSImage.SymbolConfiguration(pointSize: 470, weight: .medium)
    .applying(.init(paletteColors: [.white]))
if let sym = NSImage(systemSymbolName: "sparkles", accessibilityDescription: nil)?.withSymbolConfiguration(cfg) {
    let s = sym.size
    sym.draw(in: NSRect(x: (size - s.width) / 2, y: (size - s.height) / 2, width: s.width, height: s.height))
}
img.unlockFocus()
let rep = NSBitmapImageRep(data: img.tiffRepresentation!)!
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
