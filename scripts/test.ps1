param(
    [string[]]$PytestArgs = @("-n", "4")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "Use-RepoUv.ps1")
Enable-RepoUvFallbacks

$hasBaseTemp = $false
foreach ($arg in $PytestArgs) {
    if ($arg -eq "--basetemp" -or $arg.StartsWith("--basetemp=")) {
        $hasBaseTemp = $true
        break
    }
}
if (-not $hasBaseTemp) {
    $runId = [guid]::NewGuid().ToString("N").Substring(0, 8)
    $PytestArgs = @("--basetemp=.tmp_tests/pytest-basetemp-$runId") + $PytestArgs
}

Invoke-Uv run python -m pytest @PytestArgs
