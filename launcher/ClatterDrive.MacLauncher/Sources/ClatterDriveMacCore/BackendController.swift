import Darwin
import Foundation

public protocol BackendControlling: AnyObject {
    var onLog: ((String) -> Void)? { get set }
    var onReady: (() -> Void)? { get set }
    var onExit: ((Int32?) -> Void)? { get set }
    var isRunning: Bool { get }

    func start(settings: BackendSettings) throws
    func stop()
    func loadProfiles() throws -> ProfileCatalog
}

public final class BackendController: BackendControlling {
    public var onLog: ((String) -> Void)?
    public var onReady: (() -> Void)?
    public var onExit: ((Int32?) -> Void)?

    private var process: Process?
    private var outputPipe: Pipe?
    private var currentSettings: BackendSettings?

    public init() {}

    public var isRunning: Bool {
        process?.isRunning == true
    }

    public func start(settings: BackendSettings) throws {
        if isRunning {
            return
        }

        let spec = BackendCommandBuilder.processSpec(settings: settings)
        let launchedProcess = Process()
        launchedProcess.executableURL = URL(fileURLWithPath: spec.executablePath)
        launchedProcess.arguments = spec.arguments
        launchedProcess.environment = ProcessInfo.processInfo.environment.merging(spec.environment) { _, new in new }

        let pipe = Pipe()
        launchedProcess.standardOutput = pipe
        launchedProcess.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else {
                return
            }
            self?.handleOutput(text)
        }
        launchedProcess.terminationHandler = { [weak self] process in
            pipe.fileHandleForReading.readabilityHandler = nil
            self?.onExit?(process.terminationStatus)
        }

        try launchedProcess.run()
        process = launchedProcess
        outputPipe = pipe
        currentSettings = settings
    }

    public func stop() {
        guard let process else {
            return
        }
        if process.isRunning {
            requestBackendShutdown()
            if !waitForExit(process, timeout: 1.5) {
                process.terminate()
            }
            if !waitForExit(process, timeout: 3.0) {
                kill(process.processIdentifier, SIGKILL)
            }
        }
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        self.process = nil
        outputPipe = nil
        currentSettings = nil
    }

    public func loadProfiles() throws -> ProfileCatalog {
        let spec = BackendCommandBuilder.processSpec(settings: BackendSettings())
        let process = Process()
        process.executableURL = URL(fileURLWithPath: spec.executablePath)
        process.arguments = Array(spec.arguments.prefix { $0 != "serve" }) + ["profiles", "--json"]
        process.environment = ProcessInfo.processInfo.environment

        let output = Pipe()
        process.standardOutput = output
        process.standardError = output
        try process.run()
        process.waitUntilExit()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        if process.terminationStatus != 0 {
            let detail = String(data: data, encoding: .utf8) ?? "unknown error"
            throw NSError(
                domain: "ClatterDriveMacLauncher",
                code: Int(process.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: detail]
            )
        }
        return try JSONDecoder().decode(ProfileCatalog.self, from: data)
    }

    private func handleOutput(_ text: String) {
        for line in text.split(whereSeparator: \.isNewline).map(String.init) {
            onLog?(line)
            if let data = line.data(using: .utf8),
               let payload = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
               payload["event"] as? String == "ready" {
                onReady?()
            }
        }
    }

    private func requestBackendShutdown() {
        guard let currentSettings else {
            return
        }
        let host = currentSettings.host == "0.0.0.0" || currentSettings.host == "::" ? "127.0.0.1" : currentSettings.host
        guard let url = URL(string: "http://\(host):\(currentSettings.port)/.clatterdrive/shutdown") else {
            return
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let semaphore = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: request) { _, _, _ in
            semaphore.signal()
        }.resume()
        _ = semaphore.wait(timeout: .now() + .milliseconds(800))
    }

    private func waitForExit(_ process: Process, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        return !process.isRunning
    }
}
