using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace ClatterDrive.Launcher.Tests;

[TestClass]
public sealed class MountCommandBuilderTests
{
    [TestMethod]
    public void NetUseCommandUsesWebDavRedirectorShape()
    {
        var settings = new BackendSettings { Host = "127.0.0.1", Port = 8123 };

        var command = MountCommandBuilder.NetUseCommand(settings, "Y:");

        Assert.AreEqual(@"net use Y: \\127.0.0.1@8123\DavWWWRoot /persistent:no", command);
    }

    [TestMethod]
    public void BackendSettingsBuildsServeArguments()
    {
        var settings = new BackendSettings
        {
            Host = "127.0.0.1",
            Port = 8123,
            BackingDirectory = @"C:\Temp\ClatterDrive",
            AudioMode = "off",
            DriveProfile = "seagate_ironwolf_pro_16tb",
            AcousticProfile = "drive_on_desk",
        };

        var args = settings.ToServeArguments();

        CollectionAssert.Contains(args.ToList(), "serve");
        CollectionAssert.Contains(args.ToList(), "--json-status");
        CollectionAssert.Contains(args.ToList(), "seagate_ironwolf_pro_16tb");
        CollectionAssert.Contains(args.ToList(), "drive_on_desk");
    }

    [TestMethod]
    public void ViewModelExposesStableDefaultProfiles()
    {
        using var viewModel = new LauncherViewModel();

        Assert.AreEqual("seagate_ironwolf_pro_16tb", viewModel.DriveProfile);
        Assert.IsTrue(viewModel.DriveProfiles.Contains("desktop_7200_internal"));
        Assert.IsTrue(viewModel.AcousticProfiles.Contains("mounted_in_case"));
        Assert.AreEqual("http://127.0.0.1:8080", viewModel.WebDavUrl);
    }
}
