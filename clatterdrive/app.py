from __future__ import annotations

import os
import signal
from typing import Any

from cheroot import wsgi
from wsgidav.wsgidav_app import WsgiDAVApp

from .audio.engine import get_runtime_engine
from .storage_events import CompositeStorageEventSink, DebugStorageEventSink, StorageEventRecorder, StorageEventSink
from .webdav.provider import HDDProvider


class NoAuthWsgiDAVApp(WsgiDAVApp):
    def __call__(self, environ: dict[str, Any], start_response: Any) -> Any:
        environ["wsgidav.auth.user_name"] = "anonymous"
        return super().__call__(environ, start_response)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized not in {"0", "false", "no", "off"}


def _raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def _install_shutdown_handlers() -> dict[int, Any]:
    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _raise_keyboard_interrupt)
        except (AttributeError, OSError, ValueError):
            continue
    return previous_handlers


def _restore_shutdown_handlers(previous_handlers: dict[int, Any]) -> None:
    for signum, handler in previous_handlers.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):
            continue


def start_server() -> None:
    audio = get_runtime_engine()
    provider: HDDProvider | None = None
    server: wsgi.Server | None = None
    event_recorder: StorageEventRecorder | None = None
    event_trace_path: str | None = None
    audio_started = False
    previous_signal_handlers = _install_shutdown_handlers()
    drive_profile = os.environ.get("FAKE_HDD_DRIVE_PROFILE")
    acoustic_profile = os.environ.get("FAKE_HDD_ACOUSTIC_PROFILE")
    try:
        try:
            audio.configure_profiles(drive_profile=drive_profile, acoustic_profile=acoustic_profile)
            audio.start()
            audio_started = True
            if audio.output_enabled:
                print("Procedural Audio Engine started.")
            elif audio.tee_path:
                print("Procedural Audio Engine recording tee output without live audio.")
            else:
                print("Procedural Audio Engine live output disabled or unavailable.")
        except Exception as exc:
            audio.stop()
            print(f"Warning: Could not start audio engine: {exc}")

        root_path = os.path.abspath(os.environ.get("FAKE_HDD_BACKING_DIR", "backing_storage"))
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        event_sinks: list[StorageEventSink] = [audio]
        if _env_flag("FAKE_HDD_TRACE_EVENTS", False):
            event_sinks.append(DebugStorageEventSink())
        event_trace_path = os.environ.get("FAKE_HDD_EVENT_TRACE_PATH")
        if event_trace_path:
            event_recorder = StorageEventRecorder()
            event_sinks.append(event_recorder)
        event_sink = CompositeStorageEventSink(event_sinks)

        provider = HDDProvider(
            root_path,
            event_sink=event_sink,
            drive_profile=drive_profile,
            acoustic_profile=acoustic_profile,
            cold_start=_env_flag("FAKE_HDD_COLD_START", True),
            async_power_on=_env_flag("FAKE_HDD_ASYNC_POWER_ON", True),
        )
        port = int(os.environ.get("FAKE_HDD_PORT", "8080"))
        host = os.environ.get("FAKE_HDD_HOST", "127.0.0.1")

        config = {
            "provider_mapping": {"/": provider},
            "http_authenticator": {
                "enabled": False,
            },
            "lock_storage": True,
            "middleware_stack": [
                "wsgidav.error_printer.ErrorPrinter",
                "wsgidav.dir_browser._dir_browser.WsgiDavDirBrowser",
                "wsgidav.request_resolver.RequestResolver",
            ],
            "port": port,
            "host": host,
            "verbose": 1,
        }

        app = NoAuthWsgiDAVApp(config)
        server = wsgi.Server((host, port), app)
        print(f"Research-Enhanced HDD WebDAV server starting on http://{host}:{port}")
        print("Simulated Stack: VFS -> Page Cache -> Block Layer (LOOK/Deadline) -> SATA/AHCI -> NCQ/RPO -> Physical HDD")
        print("Hardware Model: Async Power-On -> Host Ready Polling -> Spin-Up -> Self-Test -> Head Load -> Servo Lock -> Ready")
        print("Idle Model: Unload -> Low-RPM Idle -> Staged Spin-Down -> Standby")
        print("Acoustic Model: Spindle Hum + Windage + VCM Modal Synthesis")
        print(
            f"Profiles: drive={provider.vhdd.drive_profile.name} | acoustic={provider.vhdd.acoustic_profile.name}"
        )
        server.start()
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        print(f"Server startup failed: {exc}")
    finally:
        if server is not None:
            server.stop()
        if provider is not None:
            provider.vhdd.stop()
        if audio_started:
            audio.stop()
        if event_recorder is not None and event_trace_path:
            event_recorder.export_json(event_trace_path)
        _restore_shutdown_handlers(previous_signal_handlers)


if __name__ == "__main__":
    start_server()
