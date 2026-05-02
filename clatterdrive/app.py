from __future__ import annotations

import json
import os
import signal
import threading
from typing import Any

from cheroot import wsgi
from wsgidav.wsgidav_app import WsgiDAVApp

from .audio.engine import get_runtime_engine
from .config import ClatterDriveConfig, config_from_env
from .storage_events import CompositeStorageEventSink, DebugStorageEventSink, StorageEventRecorder, StorageEventSink
from .webdav.provider import HDDProvider


class NoAuthWsgiDAVApp(WsgiDAVApp):
    def __call__(self, environ: dict[str, Any], start_response: Any) -> Any:
        environ["wsgidav.auth.user_name"] = "anonymous"
        return super().__call__(environ, start_response)


class LocalControlApp:
    def __init__(self, inner: Any, shutdown_callback: Any) -> None:
        self.inner = inner
        self.shutdown_callback = shutdown_callback

    def __call__(self, environ: dict[str, Any], start_response: Any) -> Any:
        if environ.get("REQUEST_METHOD") == "POST" and environ.get("PATH_INFO") == "/.clatterdrive/shutdown":
            remote_addr = str(environ.get("REMOTE_ADDR", ""))
            if remote_addr not in {"127.0.0.1", "::1", "localhost"}:
                start_response("403 Forbidden", [("Content-Type", "text/plain")])
                return [b"local shutdown only"]
            start_response("204 No Content", [])
            threading.Thread(target=self.shutdown_callback, daemon=True).start()
            return [b""]
        return self.inner(environ, start_response)


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


def start_server(config: ClatterDriveConfig | None = None, *, json_status: bool = False) -> None:
    resolved_config = config_from_env() if config is None else config
    resolved_config.apply_to_environ()
    audio = get_runtime_engine()
    provider: HDDProvider | None = None
    server: wsgi.Server | None = None
    event_recorder: StorageEventRecorder | None = None
    event_trace_path: str | None = None
    audio_started = False
    previous_signal_handlers = _install_shutdown_handlers()
    drive_profile = resolved_config.drive_profile
    acoustic_profile = resolved_config.acoustic_profile
    try:
        try:
            audio.configure_profiles(drive_profile=drive_profile, acoustic_profile=acoustic_profile)
            audio.start()
            audio_started = True
            if audio.output_enabled:
                print("Procedural Audio Engine started.", flush=True)
            elif audio.tee_path:
                print("Procedural Audio Engine recording tee output without live audio.", flush=True)
            else:
                print("Procedural Audio Engine live output disabled or unavailable.", flush=True)
        except Exception as exc:
            audio.stop()
            print(f"Warning: Could not start audio engine: {exc}", flush=True)

        root_path = os.path.abspath(resolved_config.backing_dir)
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        event_sinks: list[StorageEventSink] = [audio]
        if resolved_config.trace_events:
            event_sinks.append(DebugStorageEventSink())
        event_trace_path = resolved_config.event_trace_path
        if event_trace_path:
            event_recorder = StorageEventRecorder()
            event_sinks.append(event_recorder)
        event_sink = CompositeStorageEventSink(event_sinks)

        provider = HDDProvider(
            root_path,
            event_sink=event_sink,
            drive_profile=drive_profile,
            acoustic_profile=acoustic_profile,
            cold_start=resolved_config.cold_start,
            async_power_on=resolved_config.async_power_on,
        )
        port = resolved_config.port
        host = resolved_config.host

        wsgidav_config = {
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

        def request_shutdown() -> None:
            if server is not None:
                server.stop()

        app = LocalControlApp(NoAuthWsgiDAVApp(wsgidav_config), request_shutdown)
        server = wsgi.Server((host, port), app)
        server.prepare()
        print(f"Research-Enhanced HDD WebDAV server starting on http://{host}:{port}", flush=True)
        print("Simulated Stack: VFS -> Page Cache -> Block Layer (LOOK/Deadline) -> SATA/AHCI -> NCQ/RPO -> Physical HDD", flush=True)
        print("Hardware Model: Async Power-On -> Host Ready Polling -> Spin-Up -> Self-Test -> Head Load -> Servo Lock -> Ready", flush=True)
        print("Idle Model: Unload -> Low-RPM Idle -> Staged Spin-Down -> Standby", flush=True)
        print("Acoustic Model: Spindle Hum + Windage + VCM Modal Synthesis", flush=True)
        print(
            f"Profiles: drive={provider.vhdd.drive_profile.name} | acoustic={provider.vhdd.acoustic_profile.name}",
            flush=True,
        )
        if json_status:
            print(
                json.dumps(
                    {
                        "event": "ready",
                        "url": f"http://{host}:{port}",
                        "backing_dir": root_path,
                        "drive_profile": provider.vhdd.drive_profile.name,
                        "acoustic_profile": provider.vhdd.acoustic_profile.name,
                        "audio_enabled": audio.output_enabled,
                        "tee_path": audio.tee_path,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        server.serve()
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        print(f"Server startup failed: {exc}", flush=True)
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
