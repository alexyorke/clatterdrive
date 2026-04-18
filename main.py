from wsgidav.wsgidav_app import WsgiDAVApp
from cheroot import wsgi
from vfs_provider import HDDProvider
import os
from audio_engine import engine as audio

class NoAuthWsgiDAVApp(WsgiDAVApp):
    def __call__(self, environ, start_response):
        environ["wsgidav.auth.user_name"] = "anonymous"
        return super().__call__(environ, start_response)

def start_server():
    provider = None
    server = None
    audio_started = False
    try:
        try:
            audio.start()
            audio_started = True
            print("Procedural Audio Engine started.")
        except Exception as exc:
            print(f"Warning: Could not start audio engine: {exc}")

        root_path = os.path.abspath("backing_storage")
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        provider = HDDProvider(root_path)
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
