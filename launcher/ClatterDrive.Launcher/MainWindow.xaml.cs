using System;
using System.Diagnostics;
using System.Windows;
using Microsoft.Win32;

namespace ClatterDrive.Launcher;

public partial class MainWindow : Window
{
    private LauncherViewModel ViewModel => (LauncherViewModel)DataContext;

    public MainWindow()
    {
        InitializeComponent();
        DataContext = new LauncherViewModel();
    }

    private void BrowseButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog
        {
            Title = "Choose ClatterDrive backing folder",
            Multiselect = false,
            InitialDirectory = ViewModel.BackingDirectory,
        };
        if (dialog.ShowDialog(this) == true)
        {
            ViewModel.BackingDirectory = dialog.FolderName;
        }
    }

    private void OpenBrowserButton_Click(object sender, RoutedEventArgs e)
    {
        Process.Start(new ProcessStartInfo(ViewModel.WebDavUrl) { UseShellExecute = true });
    }

    private void CopyUrlButton_Click(object sender, RoutedEventArgs e)
    {
        Clipboard.SetText(MountCommandBuilder.WebDavExplorerUrl(ViewModel.CurrentSettings()));
    }

    private void CopyMountButton_Click(object sender, RoutedEventArgs e)
    {
        Clipboard.SetText(ViewModel.NetUseCommand);
    }

    private void CopyUnmountButton_Click(object sender, RoutedEventArgs e)
    {
        Clipboard.SetText(ViewModel.NetUseUnmountCommand);
    }

    private void Window_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        ViewModel.Dispose();
    }
}
