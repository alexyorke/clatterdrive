$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$IsWindowsHost = $env:OS -eq "Windows_NT"
if (-not $IsWindowsHost) {
    throw "Windows packaging must run on Windows because WPF and PyInstaller artifacts are platform-specific."
}

uv sync --locked --group dev

$RuntimeDir = Join-Path $RepoRoot ".runtime"
$PyInstallerWork = Join-Path $RuntimeDir "build\pyinstaller"
$BackendDist = Join-Path $RuntimeDir "dist\backend"
$LauncherDist = Join-Path $RuntimeDir "dist\launcher"

New-Item -ItemType Directory -Force -Path $PyInstallerWork, $BackendDist, $LauncherDist | Out-Null

uv run pyinstaller `
    --noconfirm `
    --clean `
    --name clatterdrive-backend `
    --distpath $BackendDist `
    --workpath $PyInstallerWork `
    --collect-submodules clatterdrive `
    --collect-data wsgidav `
    --hidden-import wsgidav.error_printer `
    --hidden-import wsgidav.dir_browser._dir_browser `
    --hidden-import wsgidav.request_resolver `
    tools\clatterdrive_backend_entry.py

dotnet publish launcher\ClatterDrive.Launcher\ClatterDrive.Launcher.csproj `
    -c Release `
    -r win-x64 `
    --self-contained true `
    -p:PublishSingleFile=false `
    -o $LauncherDist

Write-Host "Backend: $BackendDist"
Write-Host "Launcher: $LauncherDist"
