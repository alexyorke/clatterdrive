using System;
using System.Collections.Generic;
using System.IO;

namespace ClatterDrive.Launcher;

public sealed class BackendSettings
{
    public string Host { get; init; } = "127.0.0.1";
    public int Port { get; init; } = 8080;
    public string BackingDirectory { get; init; } = Path.GetFullPath("backing_storage");
    public string AudioMode { get; init; } = "live";
    public string? AudioDevice { get; init; }
    public string? AudioTeePath { get; init; }
    public string? EventTracePath { get; init; }
    public string DriveProfile { get; init; } = "desktop_7200_internal";
    public string? AcousticProfile { get; init; }
    public bool ColdStart { get; init; } = true;
    public bool AsyncPowerOn { get; init; } = true;

    public string WebDavUrl => $"http://{Host}:{Port}";

    public IReadOnlyList<string> ToServeArguments(bool jsonStatus = true)
    {
        var args = new List<string>
        {
            "serve",
            "--host",
            Host,
            "--port",
            Port.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--backing-dir",
            BackingDirectory,
            "--audio",
            AudioMode,
            "--drive-profile",
            DriveProfile,
        };
        if (jsonStatus)
        {
            args.Add("--json-status");
        }
        if (!string.IsNullOrWhiteSpace(AcousticProfile))
        {
            args.Add("--acoustic-profile");
            args.Add(AcousticProfile);
        }
        if (!string.IsNullOrWhiteSpace(AudioDevice))
        {
            args.Add("--audio-device");
            args.Add(AudioDevice);
        }
        if (!string.IsNullOrWhiteSpace(AudioTeePath))
        {
            args.Add("--audio-tee-path");
            args.Add(AudioTeePath);
        }
        if (!string.IsNullOrWhiteSpace(EventTracePath))
        {
            args.Add("--event-trace-path");
            args.Add(EventTracePath);
        }
        if (!ColdStart)
        {
            args.Add("--ready");
        }
        if (!AsyncPowerOn)
        {
            args.Add("--sync-power-on");
        }
        return args;
    }

    public Dictionary<string, string> ToEnvironment()
    {
        var env = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["FAKE_HDD_HOST"] = Host,
            ["FAKE_HDD_PORT"] = Port.ToString(System.Globalization.CultureInfo.InvariantCulture),
            ["FAKE_HDD_BACKING_DIR"] = BackingDirectory,
            ["FAKE_HDD_AUDIO"] = AudioMode,
            ["FAKE_HDD_DRIVE_PROFILE"] = DriveProfile,
            ["FAKE_HDD_COLD_START"] = ColdStart ? "on" : "off",
            ["FAKE_HDD_ASYNC_POWER_ON"] = AsyncPowerOn ? "on" : "off",
        };
        if (!string.IsNullOrWhiteSpace(AcousticProfile))
        {
            env["FAKE_HDD_ACOUSTIC_PROFILE"] = AcousticProfile!;
        }
        if (!string.IsNullOrWhiteSpace(AudioDevice))
        {
            env["FAKE_HDD_AUDIO_DEVICE"] = AudioDevice!;
        }
        if (!string.IsNullOrWhiteSpace(AudioTeePath))
        {
            env["FAKE_HDD_AUDIO_TEE_PATH"] = AudioTeePath!;
        }
        if (!string.IsNullOrWhiteSpace(EventTracePath))
        {
            env["FAKE_HDD_EVENT_TRACE_PATH"] = EventTracePath!;
        }
        return env;
    }
}
