$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

& (Join-Path $PSScriptRoot "build-windows.ps1")

$PackageRoot = Join-Path $RepoRoot ".runtime\package\ClatterDrive"
$ZipPath = Join-Path $RepoRoot ".runtime\package\ClatterDrive-windows-x64.zip"
$LauncherDist = Join-Path $RepoRoot ".runtime\dist\launcher"
$BackendDist = Join-Path $RepoRoot ".runtime\dist\backend\clatterdrive-backend"

if ($env:WINDOWS_SIGNING_THUMBPRINT) {
    $signtool = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if (-not $signtool) {
        throw "WINDOWS_SIGNING_THUMBPRINT is set, but signtool.exe was not found."
    }
    $timestampUrl = if ($env:WINDOWS_SIGNING_TIMESTAMP_URL) { $env:WINDOWS_SIGNING_TIMESTAMP_URL } else { "http://timestamp.digicert.com" }
    $signTargets = @(
        (Join-Path $LauncherDist "ClatterDrive.Launcher.exe"),
        (Join-Path $BackendDist "clatterdrive-backend.exe")
    )
    foreach ($target in $signTargets) {
        if (Test-Path $target) {
            & $signtool.Source sign /sha1 $env:WINDOWS_SIGNING_THUMBPRINT /fd SHA256 /tr $timestampUrl /td SHA256 $target
        }
    }
} else {
    Write-Host "No signing certificate configured; packaging unsigned binaries."
}

if (Test-Path $PackageRoot) {
    Remove-Item -Recurse -Force $PackageRoot
}
New-Item -ItemType Directory -Force -Path $PackageRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "backend") | Out-Null

Copy-Item -Recurse -Force (Join-Path $LauncherDist "*") $PackageRoot
Copy-Item -Recurse -Force (Join-Path $BackendDist "*") (Join-Path $PackageRoot "backend")
Copy-Item -Force README.md $PackageRoot
Copy-Item -Recurse -Force docs (Join-Path $PackageRoot "docs")
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "sample-backing") | Out-Null

if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}
Compress-Archive -Path $PackageRoot -DestinationPath $ZipPath
Write-Host "Package: $ZipPath"
