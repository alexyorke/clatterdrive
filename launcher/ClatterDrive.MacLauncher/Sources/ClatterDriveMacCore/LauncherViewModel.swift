import Combine
import Foundation

@MainActor
public final class LauncherViewModel: ObservableObject {
    @Published public var backingDirectory: String
    @Published public var host: String
    @Published public var portText: String
    @Published public var driveProfile: String
    @Published public var acousticProfile: String
    @Published public var audioMode: AudioMode
    @Published public var audioDevice: String
    @Published public var status: String
    @Published public var isRunning: Bool
    @Published public var logs: [String]
    @Published public var driveProfileNames: [String]
    @Published public var acousticProfileNames: [String]

    private let backend: BackendControlling

    public init(backend: BackendControlling = BackendController()) {
        self.backend = backend
        let defaults = BackendSettings()
        backingDirectory = ProcessInfo.processInfo.environment["CLATTERDRIVE_LAUNCHER_BACKING_DIR"] ?? defaults.backingDirectory
        host = ProcessInfo.processInfo.environment["CLATTERDRIVE_LAUNCHER_HOST"] ?? defaults.host
        portText = ProcessInfo.processInfo.environment["CLATTERDRIVE_LAUNCHER_PORT"] ?? String(defaults.port)
        driveProfile = ProcessInfo.processInfo.environment["CLATTERDRIVE_LAUNCHER_DRIVE_PROFILE"] ?? defaults.driveProfile
        acousticProfile = ProcessInfo.processInfo.environment["CLATTERDRIVE_LAUNCHER_ACOUSTIC_PROFILE"] ?? defaults.acousticProfile
        audioMode = AudioMode(rawValue: ProcessInfo.processInfo.environment["CLATTERDRIVE_LAUNCHER_AUDIO"] ?? "") ?? defaults.audioMode
        audioDevice = ProcessInfo.processInfo.environment["CLATTERDRIVE_LAUNCHER_AUDIO_DEVICE"] ?? defaults.audioDevice
        status = "Stopped"
        isRunning = false
        logs = []
        driveProfileNames = ProfileCatalog.fallback.driveProfiles.map(\.name)
        acousticProfileNames = ProfileCatalog.fallback.acousticProfiles.map(\.name)

        backend.onLog = { [weak self] line in
            Task { @MainActor in
                self?.appendLog(line)
            }
        }
        backend.onReady = { [weak self] in
            Task { @MainActor in
                self?.isRunning = true
                self?.status = "Running"
            }
        }
        backend.onExit = { [weak self] _ in
            Task { @MainActor in
                self?.isRunning = false
                self?.status = "Stopped"
            }
        }
    }

    public var currentPort: Int {
        Int(portText) ?? 0
    }

    public var currentSettings: BackendSettings {
        BackendSettings(
            host: host,
            port: currentPort,
            backingDirectory: backingDirectory,
            audioMode: audioMode,
            audioDevice: audioDevice,
            driveProfile: driveProfile,
            acousticProfile: acousticProfile
        )
    }

    public var validationMessage: String {
        currentSettings.validationMessage() ?? ""
    }

    public var canStart: Bool {
        !isRunning && validationMessage.isEmpty
    }

    public var webDavURL: String {
        MountCommandBuilder.webDavURL(settings: currentSettings)
    }

    public var mountCommand: String {
        MountCommandBuilder.mountCommand(settings: currentSettings)
    }

    public var unmountCommand: String {
        MountCommandBuilder.unmountCommand()
    }

    public func refreshProfiles() {
        do {
            let catalog = try backend.loadProfiles()
            driveProfileNames = catalog.driveProfiles.map(\.name)
            acousticProfileNames = catalog.acousticProfiles.map(\.name)
            if !driveProfileNames.contains(driveProfile), let first = driveProfileNames.first {
                driveProfile = first
            }
            if !acousticProfileNames.contains(acousticProfile), let first = acousticProfileNames.first {
                acousticProfile = first
            }
        } catch {
            appendLog("Profile load failed: \(error.localizedDescription)")
        }
    }

    public func start() {
        let settings = currentSettings
        if let validation = settings.validationMessage() {
            status = validation
            return
        }
        do {
            try FileManager.default.createDirectory(
                atPath: settings.backingDirectory,
                withIntermediateDirectories: true
            )
            try backend.start(settings: settings)
            isRunning = true
            status = "Starting"
        } catch {
            status = "Start failed"
            appendLog(error.localizedDescription)
        }
    }

    public func stop() {
        backend.stop()
        isRunning = false
        status = "Stopped"
    }

    private func appendLog(_ line: String) {
        logs.append(line)
        if logs.count > 300 {
            logs.removeFirst(logs.count - 300)
        }
    }
}
