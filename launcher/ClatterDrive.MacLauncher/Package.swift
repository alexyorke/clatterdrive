// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ClatterDriveMacLauncher",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "ClatterDriveMacCore", targets: ["ClatterDriveMacCore"]),
        .executable(name: "ClatterDriveMacLauncher", targets: ["ClatterDriveMacLauncher"]),
    ],
    targets: [
        .target(name: "ClatterDriveMacCore"),
        .executableTarget(
            name: "ClatterDriveMacLauncher",
            dependencies: ["ClatterDriveMacCore"]
        ),
        .testTarget(
            name: "ClatterDriveMacCoreTests",
            dependencies: ["ClatterDriveMacCore"]
        ),
    ]
)
