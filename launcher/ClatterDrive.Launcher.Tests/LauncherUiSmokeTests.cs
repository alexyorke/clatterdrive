using FlaUI.Core;
using FlaUI.Core.AutomationElements;
using FlaUI.UIA3;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Net.Http;
using System.Net.Sockets;
using System.IO;
using System.Management;
using System.Threading;

namespace ClatterDrive.Launcher.Tests;

[TestClass]
public sealed class LauncherUiSmokeTests
{
    [TestMethod]
    [TestCategory("UIE2E")]
    public void MainWindowExposesAutomationIdsForFirstRunControls()
    {
        var executable = LauncherExecutable();
        if (!File.Exists(executable))
        {
            if (Environment.GetEnvironmentVariable("CLATTERDRIVE_UI_BACKEND_E2E") == "1")
            {
                Assert.Fail($"Launcher executable not built at {executable}");
            }
            Assert.Inconclusive($"Launcher executable not built at {executable}");
        }

        using var app = Application.Launch(executable);
        using var automation = new UIA3Automation();
        var window = app.GetMainWindow(automation, TimeSpan.FromSeconds(8));
        Assert.IsNotNull(window);

        AssertControl(window, "BackingDirectoryTextBox");
        AssertControl(window, "DriveProfileComboBox");
        AssertControl(window, "AcousticProfileComboBox");
        AssertControl(window, "AudioModeComboBox");
        AssertControl(window, "StartServerButton");
        AssertControl(window, "CopyMountCommandButton");
        app.Close();
    }

    [TestMethod]
    [TestCategory("UIE2E")]
    public void PackagedLauncherStartsBundledBackendAndStopsIt()
    {
        if (Environment.GetEnvironmentVariable("CLATTERDRIVE_UI_BACKEND_E2E") != "1")
        {
            Assert.Inconclusive("Set CLATTERDRIVE_UI_BACKEND_E2E=1 to run packaged launcher/backend E2E.");
        }

        var executable = LauncherExecutable();
        if (!File.Exists(executable))
        {
            if (Environment.GetEnvironmentVariable("CLATTERDRIVE_UI_BACKEND_E2E") == "1")
            {
                Assert.Fail($"Launcher executable not built at {executable}");
            }
            Assert.Inconclusive($"Launcher executable not built at {executable}");
        }

        var port = FreePort();
        var backingDir = Path.Combine(Path.GetTempPath(), $"clatterdrive-ui-e2e-{Guid.NewGuid():N}");
        Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_PORT", port.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_BACKING_DIR", backingDir);
        Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_AUDIO", "off");
        Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_READY", "1");
        Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_SYNC_POWER_ON", "1");
        using var app = Application.Launch(executable);
        var appClosed = false;
        using var automation = new UIA3Automation();
        var window = app.GetMainWindow(automation, TimeSpan.FromSeconds(8));
        Assert.IsNotNull(window);
        try
        {
            Assert.AreEqual(port, DisplayedPort(window));
            window.FindFirstDescendant(control => control.ByAutomationId("StartServerButton"))!.AsButton().Invoke();

            WaitForStatus(window, "Running", TimeSpan.FromSeconds(30));
            try
            {
                WaitForWebDav(port, TimeSpan.FromSeconds(15));
            }
            catch
            {
                DumpLauncherLogs(window);
                throw;
            }
            RoundTripWebDav(port);

            window.FindFirstDescendant(control => control.ByAutomationId("StopServerButton"))!.AsButton().Invoke();
            WaitForStatus(window, "Stopped", TimeSpan.FromSeconds(15));
            Assert.IsFalse(CanConnect(port), "Backend port was still accepting connections after launcher stop.");
            app.Close();
            appClosed = true;
            AssertNoBackendProcessForLauncher(executable);
        }
        finally
        {
            if (!appClosed)
            {
                app.Close();
            }
            if (Directory.Exists(backingDir))
            {
                Directory.Delete(backingDir, recursive: true);
            }
            Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_PORT", null);
            Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_BACKING_DIR", null);
            Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_AUDIO", null);
            Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_READY", null);
            Environment.SetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_SYNC_POWER_ON", null);
        }
    }

    private static void AssertControl(Window window, string automationId)
    {
        Assert.IsNotNull(window.FindFirstDescendant(control => control.ByAutomationId(automationId)), automationId);
    }

    private static void DumpLauncherLogs(Window window)
    {
        var logList = window.FindFirstDescendant(control => control.ByAutomationId("LogListBox"))?.AsListBox();
        if (logList is null)
        {
            return;
        }
        foreach (var item in logList.Items)
        {
            Console.WriteLine($"launcher-log: {item.Text}");
        }
    }

    private static string LauncherExecutable()
    {
        var explicitLauncher = Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_EXE");
        if (!string.IsNullOrWhiteSpace(explicitLauncher))
        {
            return explicitLauncher;
        }
        return Path.GetFullPath(
            Path.Combine(
                AppContext.BaseDirectory,
                "..",
                "..",
                "..",
                "..",
                "ClatterDrive.Launcher",
                "bin",
                "Debug",
                "net8.0-windows",
                "ClatterDrive.Launcher.exe"
            )
        );
    }

    private static void WaitForStatus(Window window, string expected, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            var status = window.FindFirstDescendant(control => control.ByAutomationId("StatusTextBlock"))?.Name ?? "";
            if (status.Contains(expected, StringComparison.OrdinalIgnoreCase))
            {
                return;
            }
            Thread.Sleep(200);
        }
        Assert.Fail($"Timed out waiting for launcher status {expected}.");
    }

    private static int FreePort()
    {
        using var listener = new TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        return ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
    }

    private static void RoundTripWebDav(int port)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
        var payload = new ByteArrayContent("hello from packaged launcher e2e"u8.ToArray());
        var put = client.PutAsync($"http://127.0.0.1:{port}/launcher-e2e.bin", payload).GetAwaiter().GetResult();
        put.EnsureSuccessStatusCode();
        var downloaded = client.GetByteArrayAsync($"http://127.0.0.1:{port}/launcher-e2e.bin").GetAwaiter().GetResult();
        CollectionAssert.AreEqual("hello from packaged launcher e2e"u8.ToArray(), downloaded);
        var delete = client.DeleteAsync($"http://127.0.0.1:{port}/launcher-e2e.bin").GetAwaiter().GetResult();
        delete.EnsureSuccessStatusCode();
    }

    private static void WaitForWebDav(int port, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
        while (DateTime.UtcNow < deadline)
        {
            try
            {
                var response = client.GetAsync($"http://127.0.0.1:{port}/").GetAwaiter().GetResult();
                if (response.IsSuccessStatusCode)
                {
                    return;
                }
            }
            catch (HttpRequestException)
            {
            }
            catch (TaskCanceledException)
            {
            }
            Thread.Sleep(200);
        }
        Assert.Fail("Timed out waiting for packaged backend WebDAV endpoint.");
    }

    private static int DisplayedPort(Window window)
    {
        var urlText = window.FindFirstDescendant(control => control.ByAutomationId("WebDavUrlTextBlock"))?.Name ?? "http://127.0.0.1:8080";
        return new Uri(urlText).Port;
    }

    private static bool CanConnect(int port)
    {
        using var client = new TcpClient();
        try
        {
            return client.ConnectAsync("127.0.0.1", port).Wait(TimeSpan.FromMilliseconds(500));
        }
        catch (SocketException)
        {
            return false;
        }
    }

    private static void AssertNoBackendProcessForLauncher(string launcherExecutable)
    {
        var backendPath = Path.Combine(Path.GetDirectoryName(launcherExecutable)!, "backend", "clatterdrive-backend.exe");
        if (!File.Exists(backendPath))
        {
            return;
        }
        var expectedPath = Path.GetFullPath(backendPath);
        using var searcher = new ManagementObjectSearcher(
            "SELECT ProcessId, ExecutablePath, CommandLine FROM Win32_Process WHERE Name='clatterdrive-backend.exe'"
        );
        foreach (ManagementObject process in searcher.Get().Cast<ManagementObject>())
        {
            try
            {
                var executablePath = process["ExecutablePath"] as string;
                var commandLine = process["CommandLine"] as string;
                var matches = executablePath is not null
                    && Path.GetFullPath(executablePath).Equals(expectedPath, StringComparison.OrdinalIgnoreCase);
                matches |= commandLine?.Contains(expectedPath, StringComparison.OrdinalIgnoreCase) == true;
                if (matches)
                {
                    Assert.Fail($"Backend process still running after launcher stop: {process["ProcessId"]}");
                }
            }
            catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception or ManagementException)
            {
            }
            finally
            {
                process.Dispose();
            }
        }
    }
}
