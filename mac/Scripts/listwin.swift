import CoreGraphics
let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as! [[String: Any]]
for w in list where (w[kCGWindowOwnerName as String] as? String ?? "").lowercased().contains("meetre") {
    let b = w[kCGWindowBounds as String] as! [String: Any]
    print(w[kCGWindowNumber as String]!, w[kCGWindowOwnerName as String]!, w[kCGWindowName as String] ?? "", b["Width"]!, b["Height"]!, b["X"]!, b["Y"]!)
}
