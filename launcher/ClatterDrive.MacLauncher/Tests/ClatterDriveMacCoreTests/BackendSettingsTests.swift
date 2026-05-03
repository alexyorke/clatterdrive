import XCTest
@testable import ClatterDriveMacCore

final class BackendSettingsTests: XCTestCase {
    func testServeArgumentsIncludeCompatibilityFlags() {
        let settings = BackendSettings(
            host: "127.0.0.1",
            port: 9001,
            backingDirectory: "/tmp/backing folder",
            audioMode: .off,
            audioTeePath: "/tmp/audio.wav",
            eventTracePath: "/tmp/events.json",
            driveProfile: "seagate_ironwolf_pro_16tb",
            acousticProfile: "drive_on_desk",
            coldStart: false,
            asyncPowerOn: false
        )

        XCTAssertEqual(settings.validationMessage(), nil)
        let args = settings.serveArguments()
        XCTAssertTrue(args.contains("--json-status"))
        XCTAssertTrue(args.contains("--ready"))
        XCTAssertTrue(args.contains("--sync-power-on"))
        XCTAssertTrue(args.contains("/tmp/backing folder"))
        XCTAssertTrue(args.contains("/tmp/audio.wav"))
        XCTAssertEqual(settings.environment()["FAKE_HDD_AUDIO"], "off")
    }

    func testValidationRejectsBadPort() {
        let settings = BackendSettings(port: 70_000)
        XCTAssertEqual(settings.validationMessage(), "Choose a port from 1 to 65535.")
    }
}
