from __future__ import annotations

import os
from typing import Any

from cheroot import wsgi
from wsgidav.wsgidav_app import WsgiDAVApp

from .audio.engine import engine as audio
from .webdav.provider import HDDProvider


class NoAuthWsgiDAVApp(WsgiDAVApp):
    def __call__(self, environ: dict[str, Any], start_response: Any) -> Any:
        environ["wsgidav.auth.user_name"] = "anonymous"
        return super().__call__(environ, start_response)


def start_server() -> None:
    provider: HDDProvider | None = None
    server: wsgi.Server | None = None
    audio_started = False
    drive_profile = os.environ.get("FAKE_HDD_DRIVE_PROFILE")
    acoustic_profile = os.environ.get("FAKE_HDD_ACOUSTIC_PROFILE")
    try:
        try:
            audio.configure_profiles(drive_profile=drive_profile, acoustic_profile=acoustic_profile)
            audio.start()
            if audio.output_enabled:
                audio_started = True
                print("Procedural Audio Engine started.")
            else:
                print("Procedural Audio Engine disabled by FAKE_HDD_AUDIO.")
        except Exception as exc:
            print(f"Warning: Could not start audio engine: {exc}")

        root_path = os.path.abspath(os.environ.get("FAKE_HDD_BACKING_DIR", "backing_storage"))
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        provider = HDDProvider(
            root_path,
            event_sink=audio,
            drive_profile=drive_profile,
            acoustic_profile=acoustic_profile,
        )
        port = int(os.environ.get("FAKE_HDD_PORT", "8080"))

        config = {
            "provider_mapping": {"/": provider},
            "http_authenticator": {
                "enabled": False,
            },
            "middleware_stack": [
                "wsgidav.error_printer.ErrorPrinter",
                "wsgidav.dir_browser._dir_browser.WsgiDavDirBrowser",
                "wsgidav.request_resolver.RequestResolver",
            ],
            "port": port,
            "host": "127.0.0.1",
            "verbose": 1,
        }

        app = NoAuthWsgiDAVApp(config)
        server = wsgi.Server(("127.0.0.1", port), app)
        print(f"Research-Enhanced HDD WebDAV server starting on http://127.0.0.1:{port}")
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


if __name__ == "__main__":
    start_server()
