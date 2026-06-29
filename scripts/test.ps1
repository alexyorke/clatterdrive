param(
    [string[]]$PytestArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "Use-RepoUv.ps1")
Enable-RepoUvFallbacks

$hasWorkerCount = $false
$hasBaseTemp = $false
foreach ($arg in $PytestArgs) {
    if (
        $arg -eq "-n" -or
        ($arg.StartsWith("-n") -and $arg.Length -gt 2) -or
        $arg -eq "--numprocesses" -or
        $arg.StartsWith("--numprocesses=")
    ) {
        $hasWorkerCount = $true
    }
    if ($arg -eq "--basetemp" -or $arg.StartsWith("--basetemp=")) {
        $hasBaseTemp = $true
    }
}
if (-not $hasWorkerCount) {
    $PytestArgs = @("-n", "4") + $PytestArgs
}
if (-not $hasBaseTemp) {
    $runId = [guid]::NewGuid().ToString("N").Substring(0, 8)
    $PytestArgs = @("--basetemp=.tmp_tests/pytest-basetemp-$runId") + $PytestArgs
}

Invoke-Uv run python -m pytest @PytestArgs
