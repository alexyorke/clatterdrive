import Foundation

public enum MountCommandBuilder {
    public static let defaultMountPoint = "$HOME/mnt/clatterdrive"

    public static func mountCommand(settings: BackendSettings, mountPoint: String = defaultMountPoint) -> String {
        "mkdir -p \(shellToken(mountPoint)) && mount_webdav \(shellToken(settings.webDavURL + "/")) \(shellToken(mountPoint))"
    }

    public static func unmountCommand(mountPoint: String = defaultMountPoint) -> String {
        "umount \(shellToken(mountPoint))"
    }

    public static func webDavURL(settings: BackendSettings) -> String {
        settings.webDavURL + "/"
    }

    private static func shellToken(_ value: String) -> String {
        if value.hasPrefix("$HOME/") {
            return "\"\(value)\""
        }
        return "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
