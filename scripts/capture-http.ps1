param([ValidateRange(1, 6)][int]$Rounds = 1)

$ErrorActionPreference = 'Stop'
$traceRoot = Split-Path -Parent $PSScriptRoot
$tracePython = Join-Path $traceRoot '.venv\Scripts\python.exe'
$traceProfile = Join-Path $PSScriptRoot 'http-storage.wprp'
$traceIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$tracePrincipal = [Security.Principal.WindowsPrincipal]::new($traceIdentity)
if (-not $tracePrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in an administrator PowerShell window.'
}
if (-not (Test-Path -LiteralPath $tracePython -PathType Leaf)) {
    throw "Project Python environment is missing: $tracePython"
}
if (-not (Test-Path -LiteralPath $traceProfile -PathType Leaf)) {
    throw "Trace profile is missing: $traceProfile"
}
$traceRun = Join-Path (Join-Path $traceRoot '.tmp_tests') ('http-capture-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $traceRun | Out-Null
$traceTestExit = 0
Push-Location $traceRoot
try {
    # Start refuses an existing WPR recording; never cancel another session.
    # Only kernel file/disk timing and process identity; no bundled event
    # providers, minifilter callbacks, CPU sampling, or stack walks.
    Copy-Item -LiteralPath $traceProfile -Destination (Join-Path $traceRun 'profile.wprp')
    & wpr -profiledetails "${traceProfile}!HttpStorage" -filemode |
        Tee-Object -FilePath (Join-Path $traceRun 'profile-details.log') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'WPR rejected the custom profile.' }
    & wpr -start "${traceProfile}!HttpStorage" -filemode
    if ($LASTEXITCODE -ne 0) { throw 'WPR did not start. Existing recordings were not stopped.' }
    try {
        for ($traceRound = 1; $traceRound -le $Rounds; $traceRound++) {
            Write-Host "HTTP reproduction $traceRound/$Rounds at $((Get-Date).ToString('o'))"
            # Keep elevated runs out of pytest's shared per-user temp root.
            & $tracePython -m pytest tests/test_http_reliability.py tests/test_webdav.py -n 4 -q --tb=long --timeout=30 "--basetemp=$traceRun\pytest-$traceRound" "--junitxml=$traceRun\round-$traceRound.xml" -o junit_logging=all 2>&1 |
                Tee-Object -FilePath (Join-Path $traceRun "round-$traceRound.log")
            $traceTestExit = $LASTEXITCODE
            if ($traceTestExit -ne 0) { break }
        }
    } finally {
        # Save immediately, in the same elevated session that owns the trace.
        $traceSaveOutput = & wpr -stop (Join-Path $traceRun 'storage.etl') 'ClatterDrive HTTP reproduction' 2>&1
        $traceSaveExit = $LASTEXITCODE
        $traceSaveOutput | Tee-Object -FilePath (Join-Path $traceRun 'wpr-stop.log') | Out-Host
        if ($traceSaveExit -ne 0) {
            Write-Warning 'Trace save failed. Check wpr -status in this administrator window.'
            if ($traceTestExit -eq 0) { $traceTestExit = 2 }
        }
        if (($traceSaveOutput -join "`n") -match 'dropped\s+[1-9][0-9,]*\s+events') {
            Write-Warning 'ETW reported event loss. Do not treat this capture as complete.'
            if ($traceTestExit -eq 0) { $traceTestExit = 2 }
        }
        Write-Host "Private diagnostic output: $traceRun"
    }
} finally {
    Pop-Location
}
exit $traceTestExit
