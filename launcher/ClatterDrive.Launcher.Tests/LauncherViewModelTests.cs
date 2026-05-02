using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;

namespace ClatterDrive.Launcher.Tests;

[TestClass]
public sealed class LauncherViewModelTests
{
    [TestMethod]
    public void StartCommandReflectsValidation()
    {
        using var backend = new FakeBackendController();
        using var viewModel = new LauncherViewModel(backend);

        Assert.IsTrue(viewModel.StartCommand.CanExecute(null));

        viewModel.Port = 0;

        Assert.IsFalse(viewModel.StartCommand.CanExecute(null));
        viewModel.Start();
        Assert.AreEqual("Choose a port from 1 to 65535.", viewModel.Status);
        Assert.AreEqual(0, backend.StartCalls);
    }

    [TestMethod]
    public void StartStopUpdatesStateAndPassesSettings()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"clatterdrive-launcher-test-{Guid.NewGuid():N}");
        try
        {
            using var backend = new FakeBackendController();
            using var viewModel = new LauncherViewModel(backend)
            {
                BackingDirectory = tempDir,
                Port = 8123,
                AudioMode = "off",
            };

            viewModel.Start();
            backend.RaiseReady();

            Assert.IsTrue(Directory.Exists(tempDir));
            Assert.AreEqual(1, backend.StartCalls);
            Assert.IsTrue(viewModel.IsRunning);
            Assert.AreEqual("Running", viewModel.Status);
            Assert.AreEqual(8123, backend.LastSettings?.Port);
            Assert.AreEqual("off", backend.LastSettings?.AudioMode);
            Assert.IsFalse(viewModel.StartCommand.CanExecute(null));
            Assert.IsTrue(viewModel.StopCommand.CanExecute(null));

            viewModel.Stop();

            Assert.AreEqual(1, backend.StopCalls);
            Assert.IsFalse(viewModel.IsRunning);
            Assert.AreEqual("Stopped", viewModel.Status);
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [TestMethod]
    public void BackendExitResetsViewModelState()
    {
        using var backend = new FakeBackendController();
        using var viewModel = new LauncherViewModel(backend);

        viewModel.Start();
        backend.RaiseExited();

        Assert.IsFalse(viewModel.IsRunning);
        Assert.AreEqual("Stopped", viewModel.Status);
    }

    private sealed class FakeBackendController : IBackendController
    {
        public event EventHandler<string>? LogReceived;
        public event EventHandler? Ready;
        public event EventHandler? Exited;

        public bool IsRunning { get; private set; }
        public int StartCalls { get; private set; }
        public int StopCalls { get; private set; }
        public BackendSettings? LastSettings { get; private set; }

        public void Start(BackendSettings settings)
        {
            IsRunning = true;
            StartCalls++;
            LastSettings = settings;
        }

        public void Stop()
        {
            IsRunning = false;
            StopCalls++;
        }

        public void RaiseReady()
        {
            Ready?.Invoke(this, EventArgs.Empty);
        }

        public void RaiseExited()
        {
            IsRunning = false;
            Exited?.Invoke(this, EventArgs.Empty);
        }

        public void Dispose()
        {
        }

        public void RaiseLog(string line)
        {
            LogReceived?.Invoke(this, line);
        }
    }
}
