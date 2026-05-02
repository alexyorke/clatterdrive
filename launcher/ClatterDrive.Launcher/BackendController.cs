using System;
using System.Diagnostics;
using System.IO;
using System.Management;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

namespace ClatterDrive.Launcher;

public sealed class BackendController : IBackendController
{
    private Process? process;
    private BackendSettings? currentSettings;
    private string? backendExecutablePath;

    public event EventHandler<string>? LogReceived;
    public event EventHandler? Ready;
    public event EventHandler? Exited;

    public bool IsRunning => process is { HasExited: false };

    public void Start(BackendSettings settings)
    {
        if (IsRunning)
        {
            return;
        }

        var startInfo = BuildStartInfo(settings);
        backendExecutablePath = File.Exists(startInfo.FileName) ? Path.GetFullPath(startInfo.FileName) : null;
        foreach (var item in settings.ToEnvironment())
        {
            startInfo.Environment[item.Key] = item.Value;
        }
        process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        process.OutputDataReceived += (_, args) => HandleOutput(args.Data);
        process.ErrorDataReceived += (_, args) => HandleOutput(args.Data);
        process.Exited += (_, _) => Exited?.Invoke(this, EventArgs.Empty);
        if (!process.Start())
        {
            throw new InvalidOperationException("Backend process did not start.");
        }
        currentSettings = settings;
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
    }

    public void Stop()
    {
        if (process is null)
        {
            return;
        }
        try
        {
            if (!process.HasExited)
            {
                RequestBackendShutdown();
                if (!process.WaitForExit(1000))
                {
                    process.Kill(entireProcessTree: true);
                    process.WaitForExit(5000);
                }
            }
            KillOrphanedBackendProcesses();
        }
        finally
        {
            process.Dispose();
            process = null;
            currentSettings = null;
            backendExecutablePath = null;
        }
    }

    public Task StopAsync()
    {
        Stop();
        return Task.CompletedTask;
    }

    public void Dispose()
    {
        Stop();
    }

    private void RequestBackendShutdown()
    {
        if (currentSettings is null)
        {
            return;
        }
        try
        {
            var host = currentSettings.Host is "0.0.0.0" or "::" ? "127.0.0.1" : currentSettings.Host;
            using var client = new HttpClient { Timeout = TimeSpan.FromMilliseconds(500) };
            using var request = new HttpRequestMessage(HttpMethod.Post, $"http://{host}:{currentSettings.Port}/.clatterdrive/shutdown");
            client.Send(request);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or InvalidOperationException)
        {
            LogReceived?.Invoke(this, $"Graceful shutdown request failed: {ex.Message}");
        }
    }

    private void KillOrphanedBackendProcesses()
    {
        if (string.IsNullOrWhiteSpace(backendExecutablePath))
        {
            return;
        }
        var expectedPath = Path.GetFullPath(backendExecutablePath);
        if (OperatingSystem.IsWindows())
        {
            KillOrphanedBackendProcessesByWmi(expectedPath);
            return;
        }
        var processName = Path.GetFileNameWithoutExtension(expectedPath);
        foreach (var candidate in Process.GetProcessesByName(processName))
        {
            try
            {
                var candidatePath = candidate.MainModule?.FileName;
                if (candidatePath is not null && Path.GetFullPath(candidatePath).Equals(expectedPath, StringComparison.OrdinalIgnoreCase))
                {
                    candidate.Kill(entireProcessTree: true);
                    candidate.WaitForExit(5000);
                }
            }
            catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception)
            {
                LogReceived?.Invoke(this, $"Backend orphan cleanup skipped: {ex.Message}");
            }
            finally
            {
                candidate.Dispose();
            }
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private void KillOrphanedBackendProcessesByWmi(string expectedPath)
    {
        using var searcher = new ManagementObjectSearcher(
            "SELECT ProcessId, ExecutablePath, CommandLine FROM Win32_Process WHERE Name='clatterdrive-backend.exe'"
        );
        foreach (ManagementObject candidate in searcher.Get().Cast<ManagementObject>())
        {
            try
            {
                var executablePath = candidate["ExecutablePath"] as string;
                var commandLine = candidate["CommandLine"] as string;
                var matches = executablePath is not null
                    && Path.GetFullPath(executablePath).Equals(expectedPath, StringComparison.OrdinalIgnoreCase);
                matches |= commandLine?.Contains(expectedPath, StringComparison.OrdinalIgnoreCase) == true;
                if (!matches || candidate["ProcessId"] is not uint processId)
                {
                    continue;
                }
                using var processToKill = Process.GetProcessById((int)processId);
                processToKill.Kill(entireProcessTree: true);
                processToKill.WaitForExit(5000);
            }
            catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception or ManagementException)
            {
                LogReceived?.Invoke(this, $"Backend orphan cleanup skipped: {ex.Message}");
            }
            finally
            {
                candidate.Dispose();
            }
        }
    }

    internal static ProcessStartInfo BuildStartInfo(BackendSettings settings)
    {
        var explicitBackend = Environment.GetEnvironmentVariable("CLATTERDRIVE_BACKEND_EXE");
        var baseDirectory = AppContext.BaseDirectory;
        var packagedBackend = Path.Combine(baseDirectory, "backend", "clatterdrive-backend.exe");
        var siblingBackend = Path.Combine(baseDirectory, "clatterdrive-backend.exe");
        var args = settings.ToServeArguments();
        if (!string.IsNullOrWhiteSpace(explicitBackend))
        {
            return ProcessInfo(explicitBackend, args);
        }
        if (File.Exists(packagedBackend))
        {
            return ProcessInfo(packagedBackend, args);
        }
        if (File.Exists(siblingBackend))
        {
            return ProcessInfo(siblingBackend, args);
        }
        var fallback = new ProcessStartInfo("uv")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        fallback.ArgumentList.Add("run");
        fallback.ArgumentList.Add("python");
        fallback.ArgumentList.Add("-m");
        fallback.ArgumentList.Add("clatterdrive");
        foreach (var arg in args)
        {
            fallback.ArgumentList.Add(arg);
        }
        return fallback;
    }

    private static ProcessStartInfo ProcessInfo(string fileName, System.Collections.Generic.IEnumerable<string> args)
    {
        var startInfo = new ProcessStartInfo(fileName)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (var arg in args)
        {
            startInfo.ArgumentList.Add(arg);
        }
        return startInfo;
    }

    private void HandleOutput(string? line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return;
        }
        LogReceived?.Invoke(this, line);
        try
        {
            using var doc = JsonDocument.Parse(line);
            if (doc.RootElement.TryGetProperty("event", out var eventName) && eventName.GetString() == "ready")
            {
                Ready?.Invoke(this, EventArgs.Empty);
            }
        }
        catch (JsonException)
        {
        }
    }
}
