$RepoUvCache = Join-Path $RepoRoot ".tmp_uv_cache"
$RepoUvEnvironment = Join-Path $RepoRoot ".tmp_uv_env"

function Test-ExecutablePython {
    param([string]$PythonPath)

    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $false
    }

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $PythonPath
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.Arguments = '-c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"'
    $process = $null
    try {
        $process = [System.Diagnostics.Process]::Start($startInfo)
        $process.WaitForExit()
        return $process.ExitCode -eq 0
    } catch {
        return $false
    } finally {
        if ($null -ne $process) {
            $process.Dispose()
        }
    }
}

function Test-WritableDirectory {
    param([string]$DirectoryPath)

    if (-not (Test-Path -LiteralPath $DirectoryPath -PathType Container)) {
        return $false
    }

    $probe = Join-Path $DirectoryPath (".codex-probe-" + [guid]::NewGuid().ToString("N"))
    try {
        New-Item -ItemType Directory -Path $probe -ErrorAction Stop | Out-Null
        Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
        return $true
    } catch {
        return $false
    }
}

function Enable-RepoUvFallbacks {
    $venvPython = Join-Path (Join-Path $RepoRoot ".venv") "Scripts\python.exe"
    if (-not $env:UV_PROJECT_ENVIRONMENT -and (Test-Path -LiteralPath (Join-Path $RepoRoot ".venv"))) {
        if (-not (Test-ExecutablePython $venvPython)) {
            $env:UV_PROJECT_ENVIRONMENT = $RepoUvEnvironment
        }
    }

    if ($env:UV_CACHE_DIR) {
        return
    }

    $defaultCache = Join-Path (Join-Path $env:LOCALAPPDATA "uv") "cache"
    try {
        if (Test-Path -LiteralPath $defaultCache) {
            $item = Get-Item -LiteralPath $defaultCache -ErrorAction Stop
            if ((-not $item.PSIsContainer) -or (-not (Test-WritableDirectory $defaultCache))) {
                $env:UV_CACHE_DIR = $RepoUvCache
            }
            return
        }

        $parent = Split-Path -Parent $defaultCache
        if ($parent -and -not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        New-Item -ItemType Directory -Path $defaultCache -Force -ErrorAction Stop | Out-Null
    } catch {
        $env:UV_CACHE_DIR = $RepoUvCache
    }
}

function Invoke-Uv {
    & uv @args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
