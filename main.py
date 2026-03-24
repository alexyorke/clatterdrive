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
    # Start Procedural Audio Engine
    try:
        audio.start()
        print("Procedural Audio Engine started.")
    except Exception as e:
        print(f"Warning: Could not start audio engine: {e}")

    root_path = os.path.abspath("backing_storage")
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    config = {
        "provider_mapping": {"/": HDDProvider(root_path)},
        "http_authenticator": {
            "enabled": False,
        },
        "middleware_stack": [
            "wsgidav.error_printer.ErrorPrinter",
            "wsgidav.dir_browser._dir_browser.WsgiDavDirBrowser",
            "wsgidav.request_resolver.RequestResolver",
        ],
        "port": 8080,
        "host": "127.0.0.1",
        "verbose": 1,
    }
    
    app = NoAuthWsgiDAVApp(config)
    server = wsgi.Server(("127.0.0.1", 8080), app)
    print("Research-Enhanced HDD WebDAV server starting on http://127.0.0.1:8080")
    print("Simulated Stack: VFS -> Page Cache -> Block Layer (SCAN) -> SATA/AHCI -> NCQ/RPO -> Physical HDD")
    print("Acoustic Model: Spindle Hum + Windage + VCM Modal Synthesis")
    
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
        audio.stop()

if __name__ == "__main__":
    start_server()
