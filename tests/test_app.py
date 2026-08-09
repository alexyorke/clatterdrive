from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _pytest.monkeypatch import MonkeyPatch

import clatterdrive.app as app_module
from clatterdrive.app import LocalControlApp
from clatterdrive.config import ClatterDriveConfig


def _start_response_recorder() -> tuple[list[str], Callable[[str, list[tuple[str, str]]], None]]:
    statuses: list[str] = []

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        del headers
        statuses.append(status)

    return statuses, start_response


def test_local_control_shutdown_allows_loopback_and_invokes_callback_once() -> None:
    called = threading.Event()
    calls = 0

    def shutdown() -> None:
        nonlocal calls
        calls += 1
        called.set()

    def inner(environ: dict[str, Any], start_response: Any) -> list[bytes]:
        del environ, start_response
        raise AssertionError("shutdown request should not reach inner app")

    app = LocalControlApp(inner, shutdown)
    statuses, start_response = _start_response_recorder()

    body = b"".join(
        app(
            {"REQUEST_METHOD": "POST", "PATH_INFO": "/.clatterdrive/shutdown", "REMOTE_ADDR": "127.0.0.1"},
            start_response,
        )
    )

    assert body == b""
    assert statuses == ["204 No Content"]
    assert called.wait(timeout=1.0)
    assert calls == 1


def test_local_control_shutdown_allows_ip_loopback_variants() -> None:
    for remote_addr in ("127.0.0.2", "::ffff:127.0.0.1", "localhost"):
        called = threading.Event()
        app = LocalControlApp(lambda _environ, _start_response: [b"inner"], called.set)
        statuses, start_response = _start_response_recorder()

        body = b"".join(
            app(
                {"REQUEST_METHOD": "POST", "PATH_INFO": "/.clatterdrive/shutdown", "REMOTE_ADDR": remote_addr},
                start_response,
            )
        )

        assert body == b""
        assert statuses == ["204 No Content"]
        assert called.wait(timeout=1.0)


def test_local_control_shutdown_rejects_non_loopback_without_callback() -> None:
    called = threading.Event()

    def inner(environ: dict[str, Any], start_response: Any) -> list[bytes]:
        del environ, start_response
        raise AssertionError("shutdown request should not reach inner app")

    app = LocalControlApp(inner, called.set)
    statuses, start_response = _start_response_recorder()

    body = b"".join(
        app(
            {"REQUEST_METHOD": "POST", "PATH_INFO": "/.clatterdrive/shutdown", "REMOTE_ADDR": "192.0.2.10"},
            start_response,
        )
    )

    assert body == b"local shutdown only"
    assert statuses == ["403 Forbidden"]
    assert not called.wait(timeout=0.05)


def test_local_control_shutdown_rejects_non_ip_hostname_without_callback() -> None:
    called = threading.Event()
    app = LocalControlApp(lambda _environ, _start_response: [b"inner"], called.set)
    statuses, start_response = _start_response_recorder()

    body = b"".join(
        app(
            {"REQUEST_METHOD": "POST", "PATH_INFO": "/.clatterdrive/shutdown", "REMOTE_ADDR": "example.com"},
            start_response,
        )
    )

    assert body == b"local shutdown only"
    assert statuses == ["403 Forbidden"]
    assert not called.wait(timeout=0.05)


def test_local_control_passes_non_control_requests_to_inner_app() -> None:
    called = False

    def inner(environ: dict[str, Any], start_response: Any) -> list[bytes]:
        nonlocal called
        called = True
        assert environ["PATH_INFO"] == "/"
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"inner"]

    app = LocalControlApp(inner, lambda: None)
    statuses, start_response = _start_response_recorder()

    body = b"".join(app({"REQUEST_METHOD": "GET", "PATH_INFO": "/", "REMOTE_ADDR": "127.0.0.1"}, start_response))

    assert called is True
    assert body == b"inner"
    assert statuses == ["200 OK"]


def test_server_bind_failure_exits_nonzero_and_cleans_up(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    class FakeAudio:
        output_enabled = False
        tee_path = None

        def configure_profiles(self, **_kwargs: Any) -> None:
            return None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def publish_event(self, _event: Any) -> None:
            return None

    class FakeVirtualHDD:
        drive_profile = SimpleNamespace(name="test-drive")
        acoustic_profile = SimpleNamespace(name="test-acoustic")

        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    fake_vhdd = FakeVirtualHDD()

    class FakeProvider:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.vhdd = fake_vhdd

    class FailingServer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.stopped = False

        def prepare(self) -> None:
            raise OSError("address already in use")

        def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr(ClatterDriveConfig, "apply_to_environ", lambda _self: None)
    monkeypatch.setattr(app_module, "get_runtime_engine", FakeAudio)
    monkeypatch.setattr(app_module, "HDDProvider", FakeProvider)
    monkeypatch.setattr(app_module, "NoAuthWsgiDAVApp", lambda _config: object())
    monkeypatch.setattr(app_module.wsgi, "Server", FailingServer)

    config = ClatterDriveConfig(backing_dir=str(tmp_path), audio="off")
    with pytest.raises(SystemExit) as exc_info:
        app_module.start_server(config)

    assert exc_info.value.code == 1
    assert fake_vhdd.stopped is True
