// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DeepSeekOCR2Kit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "DeepSeekOCR2Kit", targets: ["DeepSeekOCR2Kit"]),
        .executable(name: "ocr2-cli", targets: ["ocr2-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "DeepSeekOCR2Kit",
            dependencies: [
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]),
        .executableTarget(
            name: "ocr2-cli",
            dependencies: [
                "DeepSeekOCR2Kit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]),
        .testTarget(name: "DeepSeekOCR2KitTests", dependencies: ["DeepSeekOCR2Kit"]),
    ]
)
