from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_powershell_test_runner_creates_fresh_temp_parent(tmp_path: Path) -> None:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell is not installed on this host")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("test.ps1", "Use-RepoUv.ps1"):
        shutil.copyfile(source / name, scripts / name)
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(tmp_path / "cache")
    env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / "venv")
    result = subprocess.run(
        [pwsh, "-NoProfile", "-Command", """
        $ErrorActionPreference = 'Stop'
        $global:uvCalls = 0
        function global:uv {
            $global:uvCalls++
            if ($args -contains 'pytest') {
                $base = @($args | Where-Object { $_ -like '--basetemp=*' })
                if ($base.Count -ne 1) { throw 'Missing unique basetemp' }
                $parent = Split-Path -Parent $base[0].Substring(11)
                if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
                    throw 'Basetemp parent missing on a fresh checkout'
                }
            }
            $global:LASTEXITCODE = 0
        }
        & ./scripts/test.ps1
        if ($global:uvCalls -ne 2) { throw 'Tests and benchmark must both run' }
        """],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
