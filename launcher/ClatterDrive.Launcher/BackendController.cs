using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

namespace ClatterDrive.Launcher;

public sealed class BackendController : IBackendController
{
    private Process? process;
    private BackendSettings? currentSettings;

    public event EventHandler<string>? LogReceived;
    public event EventHandler? Ready;
    public event EventHandler? Exited;

    public bool IsRunning => process is { HasExited: false };

    public void Start(BackendSettings settings)
    {
        if (process is not null)
        {
            if (!process.HasExited)
            {
                return;
            }
            process.Dispose();
            process = null;
            currentSettings = null;
        }

        var startInfo = BuildStartInfo(settings);
        foreach (var item in settings.ToEnvironment())
        {
            startInfo.Environment[item.Key] = item.Value;
        }
        var candidate = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        candidate.OutputDataReceived += (_, args) => HandleOutput(args.Data);
        candidate.ErrorDataReceived += (_, args) => HandleOutput(args.Data);
        candidate.Exited += (_, _) => Exited?.Invoke(this, EventArgs.Empty);
        try
        {
            if (!candidate.Start())
            {
                throw new InvalidOperationException("Backend process did not start.");
            }
            process = candidate;
            currentSettings = settings;
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
        }
        catch
        {
            candidate.Dispose();
            throw;
        }
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
                    try
                    {
                        process.Kill(entireProcessTree: true);
                        process.WaitForExit(5000);
                    }
                    catch (InvalidOperationException) when (process.HasExited)
                    {
                    }
                }
            }
        }
        finally
        {
            process.Dispose();
            process = null;
            currentSettings = null;
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
            using var client = new HttpClient { Timeout = TimeSpan.FromMilliseconds(500) };
            using var request = new HttpRequestMessage(HttpMethod.Post, BuildLocalControlUrl(currentSettings));
            client.Send(request);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or InvalidOperationException)
        {
            LogReceived?.Invoke(this, $"Graceful shutdown request failed: {ex.Message}");
        }
    }

    internal static string BuildLocalControlUrl(BackendSettings settings)
    {
        var host = settings.Host is "0.0.0.0" or "::" ? "127.0.0.1" : settings.Host;
        host = BackendSettings.FormatUrlHost(host);
        return $"http://{host}:{settings.Port}/.clatterdrive/shutdown";
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
