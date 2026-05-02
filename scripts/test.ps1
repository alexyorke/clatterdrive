param(
    [string[]]$PytestArgs = @("-n", "4")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

uv run python -m pytest @PytestArgs
