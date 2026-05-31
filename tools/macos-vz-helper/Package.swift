// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MacOSVZHelperDaemon",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(name: "macos-vz-helper", targets: ["MacOSVZHelperDaemon"]),
    ],
    targets: [
        .executableTarget(
            name: "MacOSVZHelperDaemon",
            path: "Sources"
        ),
        .testTarget(
            name: "MacOSVZHelperDaemonTests",
            dependencies: ["MacOSVZHelperDaemon"],
            path: "Tests",
            exclude: ["TemplateFixtures"]
        ),
    ]
)
