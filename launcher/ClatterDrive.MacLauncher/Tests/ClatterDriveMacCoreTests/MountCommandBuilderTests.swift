import XCTest
@testable import ClatterDriveMacCore

final class MountCommandBuilderTests: XCTestCase {
    func testMacMountCommandsUseBuiltInWebDavClient() {
        let settings = BackendSettings(port: 9090)

        XCTAssertEqual(MountCommandBuilder.webDavURL(settings: settings), "http://127.0.0.1:9090/")
        XCTAssertTrue(MountCommandBuilder.mountCommand(settings: settings).contains("mount_webdav"))
        XCTAssertTrue(MountCommandBuilder.unmountCommand().contains("umount"))
    }
}
