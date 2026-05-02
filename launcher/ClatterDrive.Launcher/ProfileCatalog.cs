namespace ClatterDrive.Launcher;

public static class ProfileCatalog
{
    public static readonly string[] DriveProfiles =
    [
        "desktop_7200_internal",
        "archive_5900_internal",
        "enterprise_7200_bare",
        "wd_ultrastar_hc550",
        "seagate_ironwolf_pro_16tb",
        "external_usb_enclosure",
    ];

    public static readonly string[] AcousticProfiles =
    [
        "mounted_in_case",
        "bare_drive_lab",
        "external_enclosure",
        "drive_on_desk",
    ];

    public static readonly string[] AudioModes = ["live", "off"];
}
