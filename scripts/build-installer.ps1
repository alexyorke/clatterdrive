param(
    [switch]$UseExistingPackage
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$IsWindowsHost = $env:OS -eq "Windows_NT"
if (-not $IsWindowsHost) {
    throw "Windows installer builds must run on Windows because the package contains Windows binaries."
}

$RuntimeDir = Join-Path $RepoRoot ".runtime"
$PackageRoot = Join-Path $RuntimeDir "package\ClatterDrive"
$InstallerDir = Join-Path $RuntimeDir "installer"
$WxsPath = Join-Path $InstallerDir "ClatterDrive.generated.wxs"
$MsiPath = Join-Path $InstallerDir "ClatterDrive-windows-x64.msi"

if (-not $UseExistingPackage -or -not (Test-Path $PackageRoot)) {
    & (Join-Path $PSScriptRoot "package-windows.ps1")
}

if (-not (Test-Path (Join-Path $PackageRoot "ClatterDrive.Launcher.exe"))) {
    throw "Packaged launcher not found under $PackageRoot."
}
if (-not (Test-Path (Join-Path $PackageRoot "backend\clatterdrive-backend.exe"))) {
    throw "Packaged backend not found under $PackageRoot."
}

New-Item -ItemType Directory -Force -Path $InstallerDir | Out-Null
dotnet tool restore

function Get-InstallerVersion {
    if ($env:CLATTERDRIVE_INSTALLER_VERSION -match '^\d+\.\d+\.\d+$') {
        return $env:CLATTERDRIVE_INSTALLER_VERSION
    }
    if ($env:GITHUB_REF_NAME -match '^v(\d+\.\d+\.\d+)') {
        return $Matches[1]
    }
    return "0.1.0"
}

function New-StableGuid([string]$Key) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $hash = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes("clatterdrive-msi:$Key"))
    $bytes = [byte[]]::new(16)
    [Array]::Copy($hash, $bytes, 16)
    $bytes[7] = ($bytes[7] -band 0x0F) -bor 0x50
    $bytes[8] = ($bytes[8] -band 0x3F) -bor 0x80
    return ([guid]::new($bytes)).ToString("B").ToUpperInvariant()
}

function New-InstallerId([string]$Prefix, [string]$Key) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $hash = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($Key))
    $hex = -join ($hash[0..9] | ForEach-Object { $_.ToString("x2") })
    return "$Prefix$hex"
}

function Get-RelativePackagePath([System.IO.FileSystemInfo]$Item) {
    $root = (Resolve-Path $PackageRoot).Path
    if (-not $root.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $root = $root + [System.IO.Path]::DirectorySeparatorChar
    }
    $rootUri = [Uri]::new($root)
    $itemUri = [Uri]::new($Item.FullName)
    return [Uri]::UnescapeDataString($rootUri.MakeRelativeUri($itemUri).ToString()).Replace("\", "/")
}

function Write-DirectoryTree($Writer, [System.IO.DirectoryInfo]$Directory, [string]$DirectoryId) {
    $subdirectories = @(Get-ChildItem -LiteralPath $Directory.FullName -Directory | Sort-Object Name)
    $files = @(Get-ChildItem -LiteralPath $Directory.FullName -File | Sort-Object Name)

    foreach ($subdirectory in $subdirectories) {
        $relative = Get-RelativePackagePath $subdirectory
        $subdirectoryId = New-InstallerId "dir_" $relative
        $Writer.WriteStartElement("Directory")
        $Writer.WriteAttributeString("Id", $subdirectoryId)
        $Writer.WriteAttributeString("Name", $subdirectory.Name)
        Write-DirectoryTree $Writer $subdirectory $subdirectoryId
        $Writer.WriteEndElement()
    }

    foreach ($file in $files) {
        $relative = Get-RelativePackagePath $file
        $componentId = New-InstallerId "cmp_" $relative
        $fileId = if ($relative -eq "ClatterDrive.Launcher.exe") { "LauncherExeFile" } else { New-InstallerId "fil_" $relative }

        $Writer.WriteStartElement("Component")
        $Writer.WriteAttributeString("Id", $componentId)
        $Writer.WriteAttributeString("Guid", (New-StableGuid "file:$relative"))
        $Writer.WriteAttributeString("Feature", "MainFeature")
        $Writer.WriteStartElement("File")
        $Writer.WriteAttributeString("Id", $fileId)
        $Writer.WriteAttributeString("Source", $file.FullName)
        $Writer.WriteAttributeString("KeyPath", "yes")
        $Writer.WriteEndElement()
        $Writer.WriteEndElement()
    }

    if ($subdirectories.Count -eq 0 -and $files.Count -eq 0 -and $Directory.FullName -ne (Resolve-Path $PackageRoot).Path) {
        $relative = Get-RelativePackagePath $Directory
        $componentId = New-InstallerId "cmp_empty_" $relative
        $Writer.WriteStartElement("Component")
        $Writer.WriteAttributeString("Id", $componentId)
        $Writer.WriteAttributeString("Guid", (New-StableGuid "empty-dir:$relative"))
        $Writer.WriteAttributeString("Feature", "MainFeature")
        $Writer.WriteStartElement("CreateFolder")
        $Writer.WriteEndElement()
        $Writer.WriteStartElement("RegistryValue")
        $Writer.WriteAttributeString("Root", "HKCU")
        $Writer.WriteAttributeString("Key", "Software\ClatterDrive\Install")
        $Writer.WriteAttributeString("Name", $componentId)
        $Writer.WriteAttributeString("Type", "integer")
        $Writer.WriteAttributeString("Value", "1")
        $Writer.WriteAttributeString("KeyPath", "yes")
        $Writer.WriteEndElement()
        $Writer.WriteEndElement()
    }
}

$settings = [System.Xml.XmlWriterSettings]::new()
$settings.Indent = $true
$settings.Encoding = [System.Text.UTF8Encoding]::new($false)
$writer = [System.Xml.XmlWriter]::Create($WxsPath, $settings)
try {
    $version = Get-InstallerVersion
    $writer.WriteStartDocument()
    $writer.WriteStartElement("Wix", "http://wixtoolset.org/schemas/v4/wxs")
    $writer.WriteStartElement("Package")
    $writer.WriteAttributeString("Id", "com.alexyorke.clatterdrive")
    $writer.WriteAttributeString("Name", "ClatterDrive")
    $writer.WriteAttributeString("Manufacturer", "ClatterDrive")
    $writer.WriteAttributeString("Version", $version)
    $writer.WriteAttributeString("Scope", "perUser")
    $writer.WriteAttributeString("UpgradeStrategy", "majorUpgrade")

    $writer.WriteStartElement("MediaTemplate")
    $writer.WriteAttributeString("EmbedCab", "yes")
    $writer.WriteEndElement()

    $writer.WriteStartElement("Feature")
    $writer.WriteAttributeString("Id", "MainFeature")
    $writer.WriteAttributeString("Title", "ClatterDrive")
    $writer.WriteAttributeString("Level", "1")
    $writer.WriteEndElement()

    $writer.WriteStartElement("StandardDirectory")
    $writer.WriteAttributeString("Id", "LocalAppDataFolder")
    $writer.WriteStartElement("Directory")
    $writer.WriteAttributeString("Id", "LocalProgramsFolder")
    $writer.WriteAttributeString("Name", "Programs")
    $writer.WriteStartElement("Directory")
    $writer.WriteAttributeString("Id", "INSTALLFOLDER")
    $writer.WriteAttributeString("Name", "ClatterDrive")
    Write-DirectoryTree $writer (Get-Item $PackageRoot) "INSTALLFOLDER"
    $writer.WriteEndElement()
    $writer.WriteEndElement()
    $writer.WriteEndElement()

    $writer.WriteStartElement("StandardDirectory")
    $writer.WriteAttributeString("Id", "ProgramMenuFolder")
    $writer.WriteStartElement("Directory")
    $writer.WriteAttributeString("Id", "ApplicationProgramsFolder")
    $writer.WriteAttributeString("Name", "ClatterDrive")
    $writer.WriteStartElement("Component")
    $writer.WriteAttributeString("Id", "ApplicationShortcutComponent")
    $writer.WriteAttributeString("Guid", (New-StableGuid "start-menu-shortcut"))
    $writer.WriteAttributeString("Feature", "MainFeature")
    $writer.WriteStartElement("Shortcut")
    $writer.WriteAttributeString("Id", "ApplicationStartMenuShortcut")
    $writer.WriteAttributeString("Name", "ClatterDrive")
    $writer.WriteAttributeString("Description", "Start ClatterDrive")
    $writer.WriteAttributeString("Target", "[#LauncherExeFile]")
    $writer.WriteAttributeString("WorkingDirectory", "INSTALLFOLDER")
    $writer.WriteEndElement()
    $writer.WriteStartElement("RemoveFolder")
    $writer.WriteAttributeString("Id", "RemoveApplicationProgramsFolder")
    $writer.WriteAttributeString("On", "uninstall")
    $writer.WriteEndElement()
    $writer.WriteStartElement("RegistryValue")
    $writer.WriteAttributeString("Root", "HKCU")
    $writer.WriteAttributeString("Key", "Software\ClatterDrive")
    $writer.WriteAttributeString("Name", "StartMenuShortcut")
    $writer.WriteAttributeString("Type", "integer")
    $writer.WriteAttributeString("Value", "1")
    $writer.WriteAttributeString("KeyPath", "yes")
    $writer.WriteEndElement()
    $writer.WriteEndElement()
    $writer.WriteEndElement()
    $writer.WriteEndElement()

    $writer.WriteEndElement()
    $writer.WriteEndElement()
    $writer.WriteEndDocument()
} finally {
    $writer.Dispose()
}

if (Test-Path $MsiPath) {
    Remove-Item -Force $MsiPath
}

dotnet tool run wix -- build $WxsPath -arch x64 -out $MsiPath
Write-Host "Installer: $MsiPath"
