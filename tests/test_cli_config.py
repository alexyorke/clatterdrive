from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from clatterdrive.__main__ import main
from clatterdrive.config import ClatterDriveConfig, config_from_env, doctor_report, profile_catalog


def test_config_from_env_preserves_existing_fake_hdd_variables(tmp_path: Path) -> None:
    env = {
        "FAKE_HDD_HOST": "127.0.0.1",
        "FAKE_HDD_PORT": "8123",
        "FAKE_HDD_BACKING_DIR": str(tmp_path),
        "FAKE_HDD_AUDIO": "off",
        "FAKE_HDD_TRACE_EVENTS": "on",
        "FAKE_HDD_COLD_START": "off",
        "FAKE_HDD_ASYNC_POWER_ON": "off",
        "FAKE_HDD_DRIVE_PROFILE": "seagate_ironwolf_pro_16tb",
        "FAKE_HDD_ACOUSTIC_PROFILE": "bare_drive_lab",
    }

    config = config_from_env(env)

    assert config.port == 8123
    assert config.audio == "off"
    assert config.trace_events is True
    assert config.cold_start is False
    assert config.async_power_on is False
    assert config.drive_profile == "seagate_ironwolf_pro_16tb"
    assert config.to_env()["FAKE_HDD_BACKING_DIR"] == str(tmp_path)


def test_profile_catalog_includes_all_user_visible_presets() -> None:
    catalog = profile_catalog()
    drive_names = {profile["name"] for profile in catalog["drive_profiles"]}
    acoustic_names = {profile["name"] for profile in catalog["acoustic_profiles"]}

    assert "desktop_7200_internal" in drive_names
    assert "seagate_ironwolf_pro_16tb" in drive_names
    assert "drive_on_desk" in acoustic_names


def test_doctor_reports_bad_profile_and_port_conflict(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = int(occupied.getsockname()[1])
        report = doctor_report(
            ClatterDriveConfig(
                port=port,
                backing_dir=str(tmp_path),
                audio="off",
                drive_profile="missing-drive",
            )
        )

    assert report["ok"] is False
    assert report["checks"]["profiles"]["ok"] is False
    assert report["checks"]["port"]["ok"] is False
    assert report["checks"]["webdav"]["ok"] is True


def test_profiles_json_cli_outputs_machine_readable_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    main(["profiles", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert "drive_profiles" in payload
    assert any(profile["name"] == "seagate_ironwolf_pro_16tb" for profile in payload["drive_profiles"])
