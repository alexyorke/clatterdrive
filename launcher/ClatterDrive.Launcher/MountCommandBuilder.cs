namespace ClatterDrive.Launcher;

public static class MountCommandBuilder
{
    public static string WebDavExplorerUrl(BackendSettings settings)
    {
        return settings.WebDavUrl + "/";
    }

    public static string NetUseCommand(BackendSettings settings, string driveLetter = "X:")
    {
        var normalized = driveLetter.EndsWith(":", System.StringComparison.Ordinal) ? driveLetter : driveLetter + ":";
        return $"net use {normalized} \\\\{settings.Host}@{settings.Port}\\DavWWWRoot /persistent:no";
    }

    public static string NetUseUnmountCommand(string driveLetter = "X:")
    {
        var normalized = driveLetter.EndsWith(":", System.StringComparison.Ordinal) ? driveLetter : driveLetter + ":";
        return $"net use {normalized} /delete";
    }
}
