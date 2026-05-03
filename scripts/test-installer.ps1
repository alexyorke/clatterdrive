param(
    [switch]$UseExistingPackage,
    [switch]$IncludeMappedDrive,
    [switch]$IncludeUiE2E
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if ($env:GITHUB_ACTIONS -ne "true" -and $env:CLATTERDRIVE_INSTALLER_E2E_VM -ne "1") {
    throw "Installer E2E mutates the host. Run it in GitHub Actions or inside a VM with CLATTERDRIVE_INSTALLER_E2E_VM=1."
}

& (Join-Path $PSScriptRoot "build-installer.ps1") -UseExistingPackage:$UseExistingPackage

$InstallerDir = Join-Path $RepoRoot ".runtime\installer"
$MsiPath = Join-Path $InstallerDir "ClatterDrive-windows-x64.msi"
$LogDir = Join-Path $InstallerDir "logs"
$InstallRoot = Join-Path $env:LOCALAPPDATA "Programs\ClatterDrive"
$LauncherExe = Join-Path $InstallRoot "ClatterDrive.Launcher.exe"
$BackendExe = Join-Path $InstallRoot "backend\clatterdrive-backend.exe"
$StartMenuDir = Join-Path ([Environment]::GetFolderPath([Environment+SpecialFolder]::Programs)) "ClatterDrive"
$StartMenuShortcut = Join-Path $StartMenuDir "ClatterDrive.lnk"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-MsiExec([string[]]$Arguments, [string]$Label, [string]$LogPath) {
    $process = Start-Process -FilePath "msiexec.exe" -ArgumentList $Arguments -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        if (Test-Path $LogPath) {
            Get-Content $LogPath -Tail 120
        }
        throw "$Label failed with msiexec exit code $($process.ExitCode)."
    }
}

function Get-FreePort {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        return ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    } finally {
        $listener.Stop()
    }
}

function Invoke-BackendJson([string[]]$Arguments, [string]$Label) {
    $output = & $BackendExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE. Output: $output"
    }
    try {
        return $output | ConvertFrom-Json
    } catch {
        throw "$Label did not return valid JSON. Output: $output"
    }
}

function Assert-InstalledShortcut {
    if (-not (Test-Path $StartMenuShortcut)) {
        throw "Start Menu shortcut missing at $StartMenuShortcut."
    }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($StartMenuShortcut)
    $targetPath = [System.IO.Path]::GetFullPath($shortcut.TargetPath)
    $expectedPath = [System.IO.Path]::GetFullPath($LauncherExe)
    if ($targetPath -ne $expectedPath) {
        throw "Start Menu shortcut points to $targetPath, expected $expectedPath."
    }
}

function Assert-NoInstalledBackendProcess {
    $expectedPath = [System.IO.Path]::GetFullPath($BackendExe)
    $processes = Get-CimInstance Win32_Process -Filter "Name='clatterdrive-backend.exe'" -ErrorAction SilentlyContinue
    foreach ($process in @($processes)) {
        $executablePath = $process.ExecutablePath
        $commandLine = $process.CommandLine
        $matches = $false
        if ($executablePath) {
            $matches = [System.IO.Path]::GetFullPath($executablePath) -eq $expectedPath
        }
        if ($commandLine -and $commandLine.IndexOf($expectedPath, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            $matches = $true
        }
        if ($matches) {
            throw "Installed backend process still running after stop/uninstall: $($process.ProcessId)."
        }
    }
}

$installLog = Join-Path $LogDir "install.log"
$repairLog = Join-Path $LogDir "repair.log"
$uninstallLog = Join-Path $LogDir "uninstall.log"
$installed = $false

try {
    Invoke-MsiExec @("/i", "`"$MsiPath`"", "/qn", "/norestart", "/L*v", "`"$installLog`"") "Installer install" $installLog
    $installed = $true

    if (-not (Test-Path $LauncherExe)) {
        throw "Installed launcher missing at $LauncherExe."
    }
    if (-not (Test-Path $BackendExe)) {
        throw "Installed backend missing at $BackendExe."
    }
    Assert-InstalledShortcut

    $profiles = Invoke-BackendJson @("profiles", "--json") "Installed backend profiles command"
    if (-not ($profiles.drive_profiles.name -contains "seagate_ironwolf_pro_16tb")) {
        throw "Installed backend profiles command did not include seagate_ironwolf_pro_16tb."
    }
    if (-not ($profiles.acoustic_profiles.name -contains "drive_on_desk")) {
        throw "Installed backend profiles command did not include drive_on_desk."
    }

    $doctorBackingDir = Join-Path $env:TEMP ("ClatterDrive Doctor Space " + [guid]::NewGuid().ToString("N"))
    $doctor = Invoke-BackendJson @(
        "doctor",
        "--json",
        "--audio",
        "off",
        "--backing-dir",
        $doctorBackingDir,
        "--port",
        (Get-FreePort).ToString([System.Globalization.CultureInfo]::InvariantCulture)
    ) "Installed backend doctor command"
    if (-not $doctor.ok) {
        throw "Installed backend doctor command reported failure."
    }

    uv run python -m tools.windows_backend_e2e --backend-exe $BackendExe
    uv run python -m tools.windows_backend_e2e --backend-exe $BackendExe --space-paths
    if ($IncludeMappedDrive) {
        uv run python -m tools.windows_backend_e2e --backend-exe $BackendExe --mapped-drive
        uv run python -m tools.windows_backend_e2e --backend-exe $BackendExe --mapped-drive --space-paths
    }

    if ($IncludeUiE2E) {
        $env:CLATTERDRIVE_LAUNCHER_EXE = $LauncherExe
        $env:CLATTERDRIVE_UI_BACKEND_E2E = "1"
        try {
            & (Join-Path $PSScriptRoot "test-ui.ps1") -IncludeUiE2E
        } finally {
            Remove-Item Env:\CLATTERDRIVE_LAUNCHER_EXE -ErrorAction SilentlyContinue
            Remove-Item Env:\CLATTERDRIVE_UI_BACKEND_E2E -ErrorAction SilentlyContinue
        }
    }

    Assert-NoInstalledBackendProcess
    Invoke-MsiExec @("/fa", "`"$MsiPath`"", "/qn", "/norestart", "/L*v", "`"$repairLog`"") "Installer repair" $repairLog
    if (-not (Test-Path $LauncherExe) -or -not (Test-Path $BackendExe)) {
        throw "Installed files missing after MSI repair."
    }
    Assert-InstalledShortcut
} finally {
    if ($installed) {
        Invoke-MsiExec @("/x", "`"$MsiPath`"", "/qn", "/norestart", "/L*v", "`"$uninstallLog`"") "Installer uninstall" $uninstallLog
    }
}

if (Test-Path $LauncherExe) {
    throw "Launcher still exists after uninstall: $LauncherExe"
}
if (Test-Path $InstallRoot) {
    throw "Install root still exists after uninstall: $InstallRoot"
}
if (Test-Path $StartMenuShortcut) {
    throw "Start Menu shortcut still exists after uninstall: $StartMenuShortcut"
}
if (Test-Path $StartMenuDir) {
    throw "Start Menu directory still exists after uninstall: $StartMenuDir"
}
Assert-NoInstalledBackendProcess

Write-Host "Installer E2E passed."
