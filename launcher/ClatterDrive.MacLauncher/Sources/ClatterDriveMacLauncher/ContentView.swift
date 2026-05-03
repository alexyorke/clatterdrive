import AppKit
import ClatterDriveMacCore
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = LauncherViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            settingsGrid
            commandBar
            logView
        }
        .padding(18)
        .task {
            viewModel.refreshProfiles()
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text("ClatterDrive")
                    .font(.title2.weight(.semibold))
                Text(viewModel.status)
                    .foregroundStyle(viewModel.isRunning ? .green : .secondary)
                    .accessibilityIdentifier("statusText")
            }
            Spacer()
            Button {
                viewModel.start()
            } label: {
                Label("Start", systemImage: "play.fill")
            }
            .disabled(!viewModel.canStart)
            .keyboardShortcut("r", modifiers: [.command])
            .accessibilityIdentifier("startButton")

            Button {
                viewModel.stop()
            } label: {
                Label("Stop", systemImage: "stop.fill")
            }
            .disabled(!viewModel.isRunning)
            .keyboardShortcut(".", modifiers: [.command])
            .accessibilityIdentifier("stopButton")
        }
    }

    private var settingsGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 10) {
            GridRow {
                Text("Backing Folder")
                HStack {
                    TextField("Backing folder", text: $viewModel.backingDirectory)
                        .accessibilityIdentifier("backingFolderTextField")
                    Button {
                        chooseBackingFolder()
                    } label: {
                        Image(systemName: "folder")
                    }
                    .help("Choose backing folder")
                    .accessibilityIdentifier("chooseBackingFolderButton")
                }
            }
            GridRow {
                Text("Host")
                TextField("Host", text: $viewModel.host)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("hostTextField")
            }
            GridRow {
                Text("Port")
                TextField("Port", text: $viewModel.portText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                    .accessibilityIdentifier("portTextField")
            }
            GridRow {
                Text("Drive")
                Picker("Drive", selection: $viewModel.driveProfile) {
                    ForEach(viewModel.driveProfileNames, id: \.self) { name in
                        Text(name).tag(name)
                    }
                }
                .accessibilityIdentifier("driveProfilePicker")
            }
            GridRow {
                Text("Acoustics")
                Picker("Acoustics", selection: $viewModel.acousticProfile) {
                    ForEach(viewModel.acousticProfileNames, id: \.self) { name in
                        Text(name).tag(name)
                    }
                }
                .accessibilityIdentifier("acousticProfilePicker")
            }
            GridRow {
                Text("Audio")
                HStack {
                    Picker("Audio", selection: $viewModel.audioMode) {
                        ForEach(AudioMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 160)
                    .accessibilityIdentifier("audioModePicker")
                    TextField("Device", text: $viewModel.audioDevice)
                        .textFieldStyle(.roundedBorder)
                        .accessibilityIdentifier("audioDeviceTextField")
                }
            }
            if !viewModel.validationMessage.isEmpty {
                GridRow {
                    Text("")
                    Text(viewModel.validationMessage)
                        .foregroundStyle(.red)
                        .accessibilityIdentifier("validationText")
                }
            }
        }
    }

    private var commandBar: some View {
        HStack(spacing: 8) {
            Button {
                openURL(viewModel.webDavURL)
            } label: {
                Label("Open", systemImage: "safari")
            }
            .accessibilityIdentifier("openWebDavButton")

            Button {
                copy(viewModel.webDavURL)
            } label: {
                Label("Copy URL", systemImage: "doc.on.doc")
            }
            .accessibilityIdentifier("copyWebDavUrlButton")

            Button {
                copy(viewModel.mountCommand)
            } label: {
                Label("Copy mount_webdav", systemImage: "externaldrive.connected.to.line.below")
            }
            .accessibilityIdentifier("copyMountCommandButton")

            Button {
                copy(viewModel.unmountCommand)
            } label: {
                Label("Copy unmount", systemImage: "eject")
            }
            .accessibilityIdentifier("copyUnmountCommandButton")
        }
    }

    private var logView: some View {
        TextEditor(
            text: .constant(viewModel.logs.joined(separator: "\n"))
        )
        .font(.system(.body, design: .monospaced))
        .frame(minHeight: 240)
        .accessibilityIdentifier("logsTextEditor")
    }

    private func chooseBackingFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            viewModel.backingDirectory = url.path
        }
    }

    private func copy(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
    }

    private func openURL(_ value: String) {
        guard let url = URL(string: value) else {
            return
        }
        NSWorkspace.shared.open(url)
    }
}
