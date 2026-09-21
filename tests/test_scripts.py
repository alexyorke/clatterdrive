from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario", ["installed", "missing", "restart", "failed", "workstation"])
def test_webdav_ci_provisioning_is_explicit_and_keeps_failures_visible(scenario: str) -> None:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell is not installed on this host")
    env = os.environ.copy()
    env["GITHUB_ACTIONS"] = "false" if scenario == "workstation" else "true"
    env.pop("CLATTERDRIVE_INSTALLER_E2E_VM", None)
    env["CLATTERDRIVE_TEST_SCENARIO"] = scenario
    env["CLATTERDRIVE_TEST_SCRIPT"] = str(Path(__file__).resolve().parents[1] / "scripts/prepare-windows-webdav.ps1")
    result = subprocess.run([pwsh, "-NoProfile", "-Command", """
        $ErrorActionPreference = 'Stop'
        $global:queried = 0
        $global:installed = $false
        $global:started = $false
        function global:Get-Service {
            param($Name, $ErrorAction)
            $global:queried++
            if ($global:queried -eq 1 -and $env:CLATTERDRIVE_TEST_SCENARIO -ne 'installed') { return $null }
            $service = [pscustomobject]@{}
            $service | Add-Member ScriptMethod WaitForStatus {
                param($Status, $Timeout)
                if (-not $global:started -or $Status -ne 'Running') { throw 'Not started' }
            }
            return $service
        }
        function global:Install-WindowsFeature {
            param($Name)
            if ($Name -ne 'WebDAV-Redirector') { throw 'Wrong feature' }
            $global:installed = $true
            return [pscustomobject]@{
                Success = ($env:CLATTERDRIVE_TEST_SCENARIO -ne 'failed')
                RestartNeeded = $(if ($env:CLATTERDRIVE_TEST_SCENARIO -eq 'restart') { 'Yes' } else { 'No' })
                ExitCode = 'MockResult'
            }
        }
        function global:Set-Service { param($Name, $StartupType) }
        function global:Start-Service { param($Name) $global:started = $true }
        & $env:CLATTERDRIVE_TEST_SCRIPT
        if (-not $global:started) { throw 'Service was not started' }
        if ($global:installed -ne ($env:CLATTERDRIVE_TEST_SCENARIO -eq 'missing')) { throw 'Wrong install path' }
        """], env=env, capture_output=True, text=True, timeout=30)
    if scenario in {"installed", "missing"}:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "mapped-drive E2E remains enabled" in result.stdout
    else:
        assert result.returncode != 0
        expected = {"restart": "requires a reboot", "failed": "installation failed", "workstation": "restricted to CI"}
        assert expected[scenario] in result.stderr


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
