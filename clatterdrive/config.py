from __future__ import annotations

import json
import os
import platform
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from .profiles import ACOUSTIC_PROFILES, DRIVE_PROFILES, resolve_acoustic_profile, resolve_drive_profile


FALSE_VALUES = {"0", "false", "no", "off", "disabled", "none"}
TRUE_VALUES = {"1", "true", "yes", "on", "enabled", "live"}


@dataclass(frozen=True)
class ClatterDriveConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    backing_dir: str = "backing_storage"
    audio: str = "live"
    audio_device: str | None = None
    audio_tee_path: str | None = None
    event_trace_path: str | None = None
    trace_events: bool = False
    cold_start: bool = True
    async_power_on: bool = True
    drive_profile: str = "desktop_7200_internal"
    acoustic_profile: str | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_env(self) -> dict[str, str]:
        env = {
            "FAKE_HDD_HOST": self.host,
            "FAKE_HDD_PORT": str(self.port),
            "FAKE_HDD_BACKING_DIR": self.backing_dir,
            "FAKE_HDD_AUDIO": self.audio,
            "FAKE_HDD_TRACE_EVENTS": "on" if self.trace_events else "off",
            "FAKE_HDD_COLD_START": "on" if self.cold_start else "off",
            "FAKE_HDD_ASYNC_POWER_ON": "on" if self.async_power_on else "off",
            "FAKE_HDD_DRIVE_PROFILE": self.drive_profile,
        }
        if self.acoustic_profile is not None:
            env["FAKE_HDD_ACOUSTIC_PROFILE"] = self.acoustic_profile
        if self.audio_device:
            env["FAKE_HDD_AUDIO_DEVICE"] = self.audio_device
        if self.audio_tee_path:
            env["FAKE_HDD_AUDIO_TEE_PATH"] = self.audio_tee_path
        if self.event_trace_path:
            env["FAKE_HDD_EVENT_TRACE_PATH"] = self.event_trace_path
        return env

    def apply_to_environ(self) -> None:
        os.environ.update(self.to_env())

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "url": self.url}


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in FALSE_VALUES:
        return False
    if normalized in TRUE_VALUES:
        return True
    return default


def config_from_env(env: dict[str, str] | None = None) -> ClatterDriveConfig:
    values = os.environ if env is None else env
    return ClatterDriveConfig(
        host=values.get("FAKE_HDD_HOST", "127.0.0.1"),
        port=int(values.get("FAKE_HDD_PORT", "8080")),
        backing_dir=values.get("FAKE_HDD_BACKING_DIR", "backing_storage"),
        audio=values.get("FAKE_HDD_AUDIO", "live"),
        audio_device=values.get("FAKE_HDD_AUDIO_DEVICE") or None,
        audio_tee_path=values.get("FAKE_HDD_AUDIO_TEE_PATH") or None,
        event_trace_path=values.get("FAKE_HDD_EVENT_TRACE_PATH") or None,
        trace_events=parse_bool(values.get("FAKE_HDD_TRACE_EVENTS"), False),
        cold_start=parse_bool(values.get("FAKE_HDD_COLD_START"), True),
        async_power_on=parse_bool(values.get("FAKE_HDD_ASYNC_POWER_ON"), True),
        drive_profile=values.get("FAKE_HDD_DRIVE_PROFILE", "desktop_7200_internal"),
        acoustic_profile=values.get("FAKE_HDD_ACOUSTIC_PROFILE") or None,
    )


def profile_catalog() -> dict[str, Any]:
    return {
        "drive_profiles": [
            {
                "name": profile.name,
                "description": profile.description,
                "default_acoustic_profile": profile.default_acoustic_profile,
                "rpm": profile.rpm,
                "platters": profile.platters,
                "hardware_prior": profile.hardware_prior,
            }
            for profile in DRIVE_PROFILES.values()
        ],
        "acoustic_profiles": [
            {
                "name": profile.name,
                "description": profile.description,
            }
            for profile in ACOUSTIC_PROFILES.values()
        ],
    }


def _check_port_available(host: str, port: int) -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
    except OSError as exc:
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "message": "port is available"}


def _check_backing_dir(path: str) -> dict[str, Any]:
    target = Path(path).expanduser()
    if target.exists():
        return {
            "ok": target.is_dir() and os.access(target, os.R_OK | os.W_OK),
            "path": str(target),
            "message": "directory is writable" if target.is_dir() else "path exists but is not a directory",
        }
    parent = target.parent if str(target.parent) else Path.cwd()
    return {
        "ok": parent.exists() and os.access(parent, os.W_OK),
        "path": str(target),
        "message": "directory can be created" if parent.exists() else "parent directory does not exist",
    }


def _check_audio(config: ClatterDriveConfig) -> dict[str, Any]:
    if config.audio.strip().lower() in FALSE_VALUES:
        return {"ok": True, "mode": "off", "message": "live audio disabled"}
    if sd is None:
        return {"ok": False, "mode": config.audio, "message": "sounddevice/PortAudio is unavailable"}
    try:
        devices = sd.query_devices()
    except Exception as exc:
        return {"ok": False, "mode": config.audio, "message": f"audio device query failed: {exc}"}
    device_count = len(devices) if hasattr(devices, "__len__") else 0
    return {"ok": device_count > 0, "mode": config.audio, "devices": device_count, "message": "audio devices detected"}


def doctor_report(config: ClatterDriveConfig) -> dict[str, Any]:
    profile_ok = True
    profile_message = "profiles resolved"
    try:
        drive = resolve_drive_profile(config.drive_profile)
        acoustic = resolve_acoustic_profile(config.acoustic_profile, drive_profile=drive)
    except ValueError as exc:
        profile_ok = False
        profile_message = str(exc)
        drive = None
        acoustic = None

    host_warning = None
    if config.host in {"0.0.0.0", "::"}:
        host_warning = "external bind exposes the unauthenticated WebDAV server to the network"

    checks: dict[str, dict[str, Any]] = {
        "profiles": {
            "ok": profile_ok,
            "drive": None if drive is None else drive.name,
            "acoustic": None if acoustic is None else acoustic.name,
            "message": profile_message,
        },
        "port": _check_port_available(config.host, config.port),
        "backing_dir": _check_backing_dir(config.backing_dir),
        "audio": _check_audio(config),
        "webdav": {
            "ok": True,
            "platform": platform.system(),
            "message": (
                "On Windows, Explorer drive mapping depends on the WebClient service. "
                "If mapping fails, start WebClient or use curl/Finder/davfs."
            ),
        },
    }
    report: dict[str, Any] = {
        "ok": False,
        "config": config.as_dict(),
        "checks": checks,
        "warnings": [host_warning] if host_warning is not None else [],
    }
    report["ok"] = all(bool(check["ok"]) for check in checks.values())
    return report


def report_as_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
