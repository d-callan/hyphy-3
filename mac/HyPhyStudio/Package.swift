// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HyPhyStudio",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "HyPhyStudio", targets: ["HyPhyStudio"]),
    ],
    targets: [
        .target(
            name: "HyPhyBridge",
            path: "Sources/HyPhyBridge",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags([
                    "-std=c++20",
                    "-O3",
                    "-I/Users/sergei/Development/hyphy-3/include",
                    "-I/Users/sergei/Development/hyphy-3/_deps/eigen3-src"
                ])
            ]
        ),
        .executableTarget(
            name: "HyPhyStudio",
            dependencies: ["HyPhyBridge"],
            path: "Sources/HyPhyStudio",
            resources: [
                .process("Resources")
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ]
        )
    ]
)
