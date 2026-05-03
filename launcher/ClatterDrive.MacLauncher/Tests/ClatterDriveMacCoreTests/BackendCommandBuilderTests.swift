import Foundation
import XCTest
@testable import ClatterDriveMacCore

final class BackendCommandBuilderTests: XCTestCase {
    func testExplicitBackendWins() {
        let spec = BackendCommandBuilder.processSpec(
            settings: BackendSettings(),
            resourceURL: nil,
            environment: ["CLATTERDRIVE_BACKEND_EXE": "/opt/clatterdrive/backend"]
        )

        XCTAssertEqual(spec.executablePath, "/opt/clatterdrive/backend")
        XCTAssertEqual(spec.arguments.first, "serve")
    }

    func testFallbackUsesUvSourceCommand() {
        let spec = BackendCommandBuilder.processSpec(
            settings: BackendSettings(),
            resourceURL: nil,
            environment: [:]
        )

        XCTAssertEqual(spec.executablePath, "/usr/bin/env")
        XCTAssertEqual(Array(spec.arguments.prefix(4)), ["uv", "run", "python", "-m"])
        XCTAssertTrue(spec.arguments.contains("clatterdrive"))
    }
}
