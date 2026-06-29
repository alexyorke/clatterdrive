using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
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

    private void CopySelectedCommandButton_Click(object sender, RoutedEventArgs e)
    {
        var selected = (CopyCommandComboBox.SelectedItem as ComboBoxItem)?.Tag as string;
        var text = selected switch
        {
            "mount" => ViewModel.NetUseCommand,
            "unmount" => ViewModel.NetUseUnmountCommand,
            _ => MountCommandBuilder.WebDavExplorerUrl(ViewModel.CurrentSettings()),
        };
        Clipboard.SetText(text);
    }

    private void Window_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        ViewModel.Dispose();
    }
}
