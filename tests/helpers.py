from __future__ import annotations

import contextlib
import faulthandler
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import wave
import uuid
from collections.abc import Iterator
from collections import deque
from itertools import pairwise
from pathlib import Path
from typing import Any

from cheroot import wsgi
import numpy as np
from wsgidav.wsgidav_app import WsgiDAVApp

from clatterdrive.audio import HDDAudioEvent
from clatterdrive.storage_events import StorageEventSink
from clatterdrive.webdav import HDDProvider


_http_heartbeat: deque[float] = deque(maxlen=256)


def _http_heartbeat_loop() -> None:
    while True:
        _http_heartbeat.append(time.monotonic())
        time.sleep(0.05)


threading.Thread(target=_http_heartbeat_loop, name="http-test-heartbeat", daemon=True).start()


class _NoAuthWsgiDAVApp(WsgiDAVApp):
    def __call__(self, environ: dict[str, Any], start_response: Any) -> Any:
        environ["wsgidav.auth.user_name"] = "anonymous"
        return super().__call__(environ, start_response)


@contextlib.contextmanager
def _run_test_server(
    backing_dir: Path,
    *,
    event_sink: StorageEventSink | None = None,
) -> Iterator[tuple[str, HDDProvider]]:
    provider = HDDProvider(
        str(backing_dir),
        cold_start=False,
        async_power_on=False,
        event_sink=event_sink,
    )
    provider.vhdd.model.latency_scale = 0.0
    config = {
        "provider_mapping": {"/": provider},
        "http_authenticator": {"enabled": False},
        "lock_storage": True,
        "middleware_stack": [
            "wsgidav.error_printer.ErrorPrinter",
            "wsgidav.dir_browser._dir_browser.WsgiDavDirBrowser",
            "wsgidav.request_resolver.RequestResolver",
        ],
        "verbose": 0,
        "logging": {"enable": False},
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


def _dump_http_threads(method: str, path: str) -> None:
    # Freeze frame locations before any filesystem access: the filesystem may
    # itself be stalled, and live frame references can advance during logging.
    frames = sys._current_frames()
    snapshots = {
        ident: traceback.StackSummary.extract(traceback.walk_stack(frame), lookup_lines=False)
        for ident, frame in frames.items()
    }
    names = [(thread.name, thread.ident) for thread in threading.enumerate()]
    captured_at = time.monotonic()
    ticks = tuple(_http_heartbeat)
    gaps = [later - earlier for earlier, later in pairwise(ticks)]
    trace_dir = Path(__file__).resolve().parents[1] / ".tmp_tests" / "http-timeouts"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"timeout-{uuid.uuid4().hex}.txt"
    with trace_path.open("w", encoding="utf-8") as trace_file:
        faulthandler.dump_traceback(file=trace_file, all_threads=True)
        trace_file.write(f"{method} {path}; captured={captured_at}; saved={time.monotonic()}\n")
        trace_file.write(f"Heartbeat last={ticks[-1] if ticks else None}; max_gap={max(gaps, default=0)}; last_gaps={gaps[-10:]}\n")
        for name, ident in names:
            trace_file.write(f"\nThread {name} ident={ident}\n")
            stack = snapshots.get(ident) if ident is not None else None
            if stack is not None:
                trace_file.writelines(stack.format())
    print(f"Full HTTP timeout trace: {trace_path}", file=sys.stderr)


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

    watchdog = threading.Timer(1.0, _dump_http_threads, args=(method, path))
    watchdog.daemon = True
    watchdog.start()
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except OSError:
            body = b""
        print(f"HTTP error: {method} {path}: {exc.code}; body={body[:4096]!r}", file=sys.stderr)
        return exc.code, body, dict(exc.headers)
    except ConnectionResetError:
        if method in {"LOCK", "UNLOCK"}:
            return 403, b"", {}
        raise
    except TimeoutError:
        _dump_http_threads(method, path)
        faulthandler.dump_traceback()
        raise
    finally:
        watchdog.cancel()


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
