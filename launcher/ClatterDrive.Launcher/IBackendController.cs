using System;

namespace ClatterDrive.Launcher;

public interface IBackendController : IDisposable
{
    event EventHandler<string>? LogReceived;
    event EventHandler? Ready;
    event EventHandler? Exited;

    bool IsRunning { get; }

    void Start(BackendSettings settings);
    void Stop();
}
