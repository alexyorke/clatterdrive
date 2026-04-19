from __future__ import annotations

import contextlib
import threading
import urllib.error
import urllib.request
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cheroot import wsgi
import numpy as np
from wsgidav.wsgidav_app import WsgiDAVApp

from fake_hdd_fuse.audio import HDDAudioEvent
from fake_hdd_fuse.webdav import HDDProvider


class _NoAuthWsgiDAVApp(WsgiDAVApp):
    def __call__(self, environ: dict[str, Any], start_response: Any) -> Any:
        environ["wsgidav.auth.user_name"] = "anonymous"
        return super().__call__(environ, start_response)


@contextlib.contextmanager
def _run_test_server(backing_dir: Path) -> Iterator[tuple[str, HDDProvider]]:
    provider = HDDProvider(str(backing_dir), cold_start=False, async_power_on=False)
    provider.vhdd.model.latency_scale = 0.0
    config = {
        "provider_mapping": {"/": provider},
        "http_authenticator": {"enabled": False},
        "middleware_stack": [
            "wsgidav.error_printer.ErrorPrinter",
            "wsgidav.dir_browser._dir_browser.WsgiDavDirBrowser",
            "wsgidav.request_resolver.RequestResolver",
        ],
        "verbose": 0,
    }
    app = _NoAuthWsgiDAVApp(config)
    server = wsgi.Server(("127.0.0.1", 0), app)
    server.prepare()
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.bind_addr[1]}", provider
    finally:
        server.stop()
        thread.join(timeout=2.0)
        provider.vhdd.stop()


def _request(
    base_url: str,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(f"{base_url}{path}", data=body, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)

    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _assert_provider_tree_matches_disk(provider: HDDProvider, backing_dir: Path) -> None:
    backing_root = Path(backing_dir)
    disk_dirs = {"/"}
    disk_files = {}
    for current in backing_root.rglob("*"):
        rel = current.relative_to(backing_root).as_posix()
        virtual_path = "/" + rel if rel else "/"
        if current.is_dir():
            disk_dirs.add(virtual_path)
        elif current.is_file():
            disk_files[virtual_path] = current.stat().st_size

    assert set(provider.vhdd.fs.directories) == disk_dirs
    assert set(provider.vhdd.fs.files) == set(disk_files)
    for path, size in disk_files.items():
        assert provider.vhdd.fs.files[path].size == size

    provider.vhdd.fs.assert_consistent()


def _list_disk_tree(backing_dir: Path) -> tuple[list[str], list[str]]:
    backing_root = Path(backing_dir)
    dirs = ["/"]
    files = []
    for current in backing_root.rglob("*"):
        rel = current.relative_to(backing_root).as_posix()
        virtual_path = "/" + rel if rel else "/"
        if current.is_dir():
            dirs.append(virtual_path)
        elif current.is_file():
            files.append(virtual_path)
    return sorted(dirs), sorted(files)


def _wav_metrics(path: Path) -> tuple[float, float]:
    with wave.open(str(path), "rb") as wav_file:
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16).astype(np.float64) / 32767.0
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    return rms, peak


def _audio_event(**kwargs: Any) -> HDDAudioEvent:
    return HDDAudioEvent(emitted_at=0.0, **kwargs)
