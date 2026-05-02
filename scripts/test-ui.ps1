param(
    [switch]$IncludeUiE2E,
    [switch]$Packaged,
    [switch]$UseExistingPackage
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

dotnet build launcher\ClatterDrive.Launcher\ClatterDrive.Launcher.csproj -c Debug

if ($Packaged) {
    $zipPath = Join-Path $RepoRoot ".runtime\package\ClatterDrive-windows-x64.zip"
    if (-not $UseExistingPackage -or -not (Test-Path $zipPath)) {
        & (Join-Path $PSScriptRoot "package-windows.ps1")
    }
    $extractRoot = Join-Path $RepoRoot (".runtime\ui-extracted-package-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    Expand-Archive -Path $zipPath -DestinationPath $extractRoot -Force
    $launcherExe = Join-Path $extractRoot "ClatterDrive\ClatterDrive.Launcher.exe"
    if (-not (Test-Path $launcherExe)) {
        throw "Packaged launcher not found at $launcherExe"
    }
    $env:CLATTERDRIVE_LAUNCHER_EXE = $launcherExe
    $env:CLATTERDRIVE_UI_BACKEND_E2E = "1"
}

$filter = "TestCategory!=UIE2E"
if ($IncludeUiE2E) {
    $filter = ""
}

try {
    if ($filter) {
        dotnet test launcher\ClatterDrive.Launcher.Tests\ClatterDrive.Launcher.Tests.csproj -c Debug --filter $filter
    } else {
        dotnet test launcher\ClatterDrive.Launcher.Tests\ClatterDrive.Launcher.Tests.csproj -c Debug
    }
} finally {
    if ($Packaged) {
        Remove-Item Env:\CLATTERDRIVE_LAUNCHER_EXE -ErrorAction SilentlyContinue
        Remove-Item Env:\CLATTERDRIVE_UI_BACKEND_E2E -ErrorAction SilentlyContinue
        if ($extractRoot -and (Test-Path $extractRoot)) {
            Remove-Item -Recurse -Force $extractRoot -ErrorAction SilentlyContinue
        }
    }
}
