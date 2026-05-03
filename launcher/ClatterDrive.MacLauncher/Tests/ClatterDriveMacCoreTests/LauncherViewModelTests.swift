import XCTest
@testable import ClatterDriveMacCore

final class LauncherViewModelTests: XCTestCase {
    @MainActor
    func testStartUsesCurrentSettingsAndReadyCallback() async {
        let backend = FakeBackend()
        let model = LauncherViewModel(backend: backend)
        model.portText = "9099"
        model.backingDirectory = "/tmp/clatterdrive-test"

        model.start()
        backend.onReady?()
        await Task.yield()

        XCTAssertEqual(backend.startedSettings?.port, 9099)
        XCTAssertTrue(model.isRunning)
        XCTAssertEqual(model.status, "Running")
    }

    @MainActor
    func testInvalidPortDoesNotStart() {
        let backend = FakeBackend()
        let model = LauncherViewModel(backend: backend)
        model.portText = "abc"

        model.start()

        XCTAssertNil(backend.startedSettings)
        XCTAssertEqual(model.status, "Choose a port from 1 to 65535.")
    }
}

private final class FakeBackend: BackendControlling {
    var onLog: ((String) -> Void)?
    var onReady: (() -> Void)?
    var onExit: ((Int32?) -> Void)?
    var isRunning = false
    var startedSettings: BackendSettings?

    func start(settings: BackendSettings) throws {
        startedSettings = settings
        isRunning = true
    }

    func stop() {
        isRunning = false
    }

    func loadProfiles() throws -> ProfileCatalog {
        ProfileCatalog.fallback
    }
}
