import Foundation

public enum AudioMode: String, CaseIterable, Identifiable {
    case live
    case off

    public var id: String { rawValue }
}

public struct BackendSettings: Equatable {
    public var host: String
    public var port: Int
    public var backingDirectory: String
    public var audioMode: AudioMode
    public var audioDevice: String
    public var audioTeePath: String
    public var eventTracePath: String
    public var driveProfile: String
    public var acousticProfile: String
    public var coldStart: Bool
    public var asyncPowerOn: Bool

    public init(
        host: String = "127.0.0.1",
        port: Int = 8080,
        backingDirectory: String = FileManager.default.currentDirectoryPath + "/backing_storage",
        audioMode: AudioMode = .live,
        audioDevice: String = "",
        audioTeePath: String = "",
        eventTracePath: String = "",
        driveProfile: String = "seagate_ironwolf_pro_16tb",
        acousticProfile: String = "mounted_in_case",
        coldStart: Bool = true,
        asyncPowerOn: Bool = true
    ) {
        self.host = host
        self.port = port
        self.backingDirectory = backingDirectory
        self.audioMode = audioMode
        self.audioDevice = audioDevice
        self.audioTeePath = audioTeePath
        self.eventTracePath = eventTracePath
        self.driveProfile = driveProfile
        self.acousticProfile = acousticProfile
        self.coldStart = coldStart
        self.asyncPowerOn = asyncPowerOn
    }

    public var webDavURL: String {
        "http://\(host):\(port)"
    }

    public func validationMessage() -> String? {
        if backingDirectory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Choose a backing folder."
        }
        if host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Enter a host."
        }
        if port < 1 || port > 65_535 {
            return "Choose a port from 1 to 65535."
        }
        if driveProfile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Choose a drive profile."
        }
        return nil
    }

    public func serveArguments(jsonStatus: Bool = true) -> [String] {
        var args = [
            "serve",
            "--host",
            host,
            "--port",
            String(port),
            "--backing-dir",
            backingDirectory,
            "--audio",
            audioMode.rawValue,
            "--drive-profile",
            driveProfile,
        ]
        if jsonStatus {
            args.append("--json-status")
        }
        if !acousticProfile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args.append(contentsOf: ["--acoustic-profile", acousticProfile])
        }
        if !audioDevice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args.append(contentsOf: ["--audio-device", audioDevice])
        }
        if !audioTeePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args.append(contentsOf: ["--audio-tee-path", audioTeePath])
        }
        if !eventTracePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args.append(contentsOf: ["--event-trace-path", eventTracePath])
        }
        if !coldStart {
            args.append("--ready")
        }
        if !asyncPowerOn {
            args.append("--sync-power-on")
        }
        return args
    }

    public func environment() -> [String: String] {
        var env = [
            "FAKE_HDD_HOST": host,
            "FAKE_HDD_PORT": String(port),
            "FAKE_HDD_BACKING_DIR": backingDirectory,
            "FAKE_HDD_AUDIO": audioMode.rawValue,
            "FAKE_HDD_DRIVE_PROFILE": driveProfile,
            "FAKE_HDD_COLD_START": coldStart ? "on" : "off",
            "FAKE_HDD_ASYNC_POWER_ON": asyncPowerOn ? "on" : "off",
        ]
        if !acousticProfile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            env["FAKE_HDD_ACOUSTIC_PROFILE"] = acousticProfile
        }
        if !audioDevice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            env["FAKE_HDD_AUDIO_DEVICE"] = audioDevice
        }
        if !audioTeePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            env["FAKE_HDD_AUDIO_TEE_PATH"] = audioTeePath
        }
        if !eventTracePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            env["FAKE_HDD_EVENT_TRACE_PATH"] = eventTracePath
        }
        return env
    }
}
