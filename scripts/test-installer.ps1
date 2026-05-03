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

$installLog = Join-Path $LogDir "install.log"
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

    uv run python -m tools.windows_backend_e2e --backend-exe $BackendExe
    if ($IncludeMappedDrive) {
        uv run python -m tools.windows_backend_e2e --backend-exe $BackendExe --mapped-drive
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
} finally {
    if ($installed) {
        Invoke-MsiExec @("/x", "`"$MsiPath`"", "/qn", "/norestart", "/L*v", "`"$uninstallLog`"") "Installer uninstall" $uninstallLog
    }
}

if (Test-Path $LauncherExe) {
    throw "Launcher still exists after uninstall: $LauncherExe"
}

Write-Host "Installer E2E passed."
