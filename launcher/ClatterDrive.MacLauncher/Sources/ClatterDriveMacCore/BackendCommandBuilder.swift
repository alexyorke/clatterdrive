import Foundation

public struct BackendProcessSpec: Equatable {
    public var executablePath: String
    public var arguments: [String]
    public var environment: [String: String]

    public init(executablePath: String, arguments: [String], environment: [String: String]) {
        self.executablePath = executablePath
        self.arguments = arguments
        self.environment = environment
    }
}

public enum BackendCommandBuilder {
    public static func processSpec(
        settings: BackendSettings,
        resourceURL: URL? = Bundle.main.resourceURL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> BackendProcessSpec {
        let serveArgs = settings.serveArguments()
        if let explicitBackend = environment["CLATTERDRIVE_BACKEND_EXE"],
           !explicitBackend.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return BackendProcessSpec(
                executablePath: explicitBackend,
                arguments: serveArgs,
                environment: settings.environment()
            )
        }

        if let resourceURL {
            let packagedBackend = resourceURL
                .appendingPathComponent("backend")
                .appendingPathComponent("clatterdrive-backend")
                .path
            if fileManager.isExecutableFile(atPath: packagedBackend) {
                return BackendProcessSpec(
                    executablePath: packagedBackend,
                    arguments: serveArgs,
                    environment: settings.environment()
                )
            }
        }

        return BackendProcessSpec(
            executablePath: "/usr/bin/env",
            arguments: ["uv", "run", "python", "-m", "clatterdrive"] + serveArgs,
            environment: settings.environment()
        )
    }
}
