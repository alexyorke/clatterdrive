param(
    [switch]$IncludeMappedDrive,
    [switch]$SkipMappedDrive,
    [switch]$IncludeUiE2E
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

& (Join-Path $PSScriptRoot "package-windows.ps1")

if ($IncludeMappedDrive -or -not $SkipMappedDrive) {
    & (Join-Path $PSScriptRoot "test-e2e.ps1") -Packaged -FromZip -IncludeMappedDrive
} else {
    & (Join-Path $PSScriptRoot "test-e2e.ps1") -Packaged -FromZip
}

if ($IncludeUiE2E) {
    & (Join-Path $PSScriptRoot "test-ui.ps1") -Packaged -UseExistingPackage -IncludeUiE2E
}
