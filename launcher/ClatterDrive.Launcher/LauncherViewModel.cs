using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Input;

namespace ClatterDrive.Launcher;

public sealed class LauncherViewModel : INotifyPropertyChanged, IDisposable
{
    private readonly IBackendController backend;
    private string backingDirectory = Path.GetFullPath("backing_storage");
    private string host = "127.0.0.1";
    private int port = 8080;
    private string driveProfile = "seagate_ironwolf_pro_16tb";
    private string acousticProfile = "mounted_in_case";
    private string audioMode = "live";
    private string audioDevice = "";
    private string status = "Stopped";
    private bool isRunning;
    private bool coldStart = true;
    private bool asyncPowerOn = true;

    public LauncherViewModel()
        : this(new BackendController())
    {
    }

    public LauncherViewModel(IBackendController backend)
    {
        this.backend = backend;
        ApplyEnvironmentDefaults();
        this.backend.LogReceived += (_, line) => AddLog(line);
        this.backend.Ready += (_, _) => Status = "Running";
        this.backend.Exited += (_, _) =>
        {
            IsRunning = false;
            Status = "Stopped";
        };
        StartCommand = new RelayCommand(_ => Start(), _ => CanStart);
        StopCommand = new RelayCommand(_ => Stop(), _ => IsRunning);
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public string[] DriveProfiles => ProfileCatalog.DriveProfiles;
    public string[] AcousticProfiles => ProfileCatalog.AcousticProfiles;
    public string[] AudioModes => ProfileCatalog.AudioModes;
    public ObservableCollection<string> Logs { get; } = [];
    public ICommand StartCommand { get; }
    public ICommand StopCommand { get; }

    public string BackingDirectory
    {
        get => backingDirectory;
        set
        {
            if (SetField(ref backingDirectory, value))
            {
                RefreshValidation();
            }
        }
    }

    public string Host
    {
        get => host;
        set
        {
            if (SetField(ref host, value))
            {
                OnPropertyChanged(nameof(WebDavUrl));
                OnPropertyChanged(nameof(NetUseCommand));
                RefreshValidation();
            }
        }
    }

    public int Port
    {
        get => port;
        set
        {
            if (SetField(ref port, value))
            {
                OnPropertyChanged(nameof(WebDavUrl));
                OnPropertyChanged(nameof(NetUseCommand));
                RefreshValidation();
            }
        }
    }

    public string DriveProfile
    {
        get => driveProfile;
        set => SetField(ref driveProfile, value);
    }

    public string AcousticProfile
    {
        get => acousticProfile;
        set => SetField(ref acousticProfile, value);
    }

    public string AudioMode
    {
        get => audioMode;
        set => SetField(ref audioMode, value);
    }

    public string AudioDevice
    {
        get => audioDevice;
        set => SetField(ref audioDevice, value);
    }

    public string Status
    {
        get => status;
        private set => SetField(ref status, value);
    }

    public bool IsRunning
    {
        get => isRunning;
        private set
        {
            if (SetField(ref isRunning, value))
            {
                OnPropertyChanged(nameof(CanStart));
                ((RelayCommand)StartCommand).RaiseCanExecuteChanged();
                ((RelayCommand)StopCommand).RaiseCanExecuteChanged();
            }
        }
    }

    public bool CanStart => !IsRunning && string.IsNullOrEmpty(ValidationMessage);
    public string ValidationMessage => Validate();
    public string WebDavUrl => CurrentSettings().WebDavUrl;
    public string NetUseCommand => MountCommandBuilder.NetUseCommand(CurrentSettings());
    public string NetUseUnmountCommand => MountCommandBuilder.NetUseUnmountCommand();

    public BackendSettings CurrentSettings()
    {
        return new BackendSettings
        {
            Host = Host,
            Port = Port,
            BackingDirectory = BackingDirectory,
            AudioMode = AudioMode,
            AudioDevice = string.IsNullOrWhiteSpace(AudioDevice) ? null : AudioDevice,
            DriveProfile = DriveProfile,
            AcousticProfile = AcousticProfile,
            ColdStart = coldStart,
            AsyncPowerOn = asyncPowerOn,
        };
    }

    public void Start()
    {
        var validation = ValidationMessage;
        if (!string.IsNullOrEmpty(validation))
        {
            Status = validation;
            return;
        }
        Directory.CreateDirectory(BackingDirectory);
        Status = "Starting";
        backend.Start(CurrentSettings());
        IsRunning = true;
    }

    public void Stop()
    {
        backend.Stop();
        IsRunning = false;
        Status = "Stopped";
    }

    public void Dispose()
    {
        backend.Dispose();
    }

    private void AddLog(string line)
    {
        App.Current.Dispatcher.Invoke(() =>
        {
            Logs.Add(line);
            while (Logs.Count > 300)
            {
                Logs.RemoveAt(0);
            }
        });
    }

    private void ApplyEnvironmentDefaults()
    {
        backingDirectory = Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_BACKING_DIR") ?? backingDirectory;
        host = Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_HOST") ?? host;
        if (int.TryParse(Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_PORT"), out var envPort))
        {
            port = envPort;
        }
        audioMode = Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_AUDIO") ?? audioMode;
        audioDevice = Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_AUDIO_DEVICE") ?? audioDevice;
        driveProfile = Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_DRIVE_PROFILE") ?? driveProfile;
        acousticProfile = Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_ACOUSTIC_PROFILE") ?? acousticProfile;
        if (IsTruthy(Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_READY")))
        {
            coldStart = false;
        }
        if (IsTruthy(Environment.GetEnvironmentVariable("CLATTERDRIVE_LAUNCHER_SYNC_POWER_ON")))
        {
            asyncPowerOn = false;
        }
    }

    private static bool IsTruthy(string? value)
    {
        return value?.Trim().ToLowerInvariant() is "1" or "true" or "yes" or "on";
    }

    private string Validate()
    {
        if (string.IsNullOrWhiteSpace(BackingDirectory))
        {
            return "Choose a backing folder.";
        }
        if (string.IsNullOrWhiteSpace(Host))
        {
            return "Enter a host.";
        }
        if (Port is < 1 or > 65535)
        {
            return "Choose a port from 1 to 65535.";
        }
        return "";
    }

    private void RefreshValidation()
    {
        OnPropertyChanged(nameof(ValidationMessage));
        OnPropertyChanged(nameof(CanStart));
        ((RelayCommand)StartCommand).RaiseCanExecuteChanged();
    }

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (Equals(field, value))
        {
            return false;
        }
        field = value;
        OnPropertyChanged(propertyName);
        return true;
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
