param(
    [switch]$PythonOnly,
    [switch]$SkipE2E,
    [switch]$SkipWindowsApp,
    [switch]$IncludeUiE2E,
    [switch]$IncludeMappedDrive,
    [switch]$IncludeInstallerE2E
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$IsWindowsHost = $env:OS -eq "Windows_NT"

$runWindowsApp = $IsWindowsHost -and -not $PythonOnly -and -not $SkipWindowsApp

$bootstrapArgs = @()
if (-not $runWindowsApp) {
    $bootstrapArgs += "-PythonOnly"
}

& (Join-Path $PSScriptRoot "bootstrap.ps1") @bootstrapArgs
& (Join-Path $PSScriptRoot "lint.ps1")
& (Join-Path $PSScriptRoot "test.ps1")

if (-not $SkipE2E) {
    if ($IncludeMappedDrive) {
        & (Join-Path $PSScriptRoot "test-e2e.ps1") -IncludeMappedDrive
    } else {
        & (Join-Path $PSScriptRoot "test-e2e.ps1")
    }
}

if ($runWindowsApp) {
    if ($IncludeUiE2E) {
        & (Join-Path $PSScriptRoot "test-ui.ps1") -IncludeUiE2E
    } else {
        & (Join-Path $PSScriptRoot "test-ui.ps1")
    }
}

if ($IncludeInstallerE2E) {
    & (Join-Path $PSScriptRoot "test-installer.ps1") -IncludeUiE2E:$IncludeUiE2E -IncludeMappedDrive:$IncludeMappedDrive
}
