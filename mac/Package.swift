// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Meetre",
    platforms: [.macOS(.v26)],
    targets: [
        .executableTarget(
            name: "Meetre",
            path: "Sources/Meetre"
        )
    ]
)
