param(
    [switch]$Packaged,
    [switch]$FromZip,
    [switch]$IncludeMappedDrive
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$IsWindowsHost = $env:OS -eq "Windows_NT"

$backendArgs = @()
if ($Packaged) {
    if ($FromZip) {
        $zipPath = Join-Path $RepoRoot ".runtime\package\ClatterDrive-windows-x64.zip"
        if (-not (Test-Path $zipPath)) {
            & (Join-Path $PSScriptRoot "package-windows.ps1")
        }
        $extractRoot = Join-Path $RepoRoot (".runtime\e2e-extracted-package-" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
        Expand-Archive -Path $zipPath -DestinationPath $extractRoot -Force
        $backendExe = Join-Path $extractRoot "ClatterDrive\backend\clatterdrive-backend.exe"
    } else {
        $backendExe = Join-Path $RepoRoot ".runtime\dist\backend\clatterdrive-backend\clatterdrive-backend.exe"
    }
    if (-not (Test-Path $backendExe)) {
        & (Join-Path $PSScriptRoot "build-windows.ps1")
    }
    if (-not (Test-Path $backendExe)) {
        throw "Packaged backend not found at $backendExe"
    }
    $backendArgs += @("--backend-exe", $backendExe)
}

uv run python -m tools.windows_backend_e2e @backendArgs

if ($IncludeMappedDrive) {
    if (-not $IsWindowsHost) {
        throw "Mapped-drive E2E is Windows-only."
    }
    uv run python -m tools.windows_backend_e2e @backendArgs --mapped-drive
}
