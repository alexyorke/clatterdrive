param(
    [switch]$PythonOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv first, then rerun scripts\bootstrap.ps1."
}

uv sync --locked --group dev

if (-not $PythonOnly) {
    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
        throw ".NET SDK is required for the Windows launcher."
    }
    dotnet restore launcher\ClatterDrive.Launcher\ClatterDrive.Launcher.csproj
    dotnet restore launcher\ClatterDrive.Launcher.Tests\ClatterDrive.Launcher.Tests.csproj
}
