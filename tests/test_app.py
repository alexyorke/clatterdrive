from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from clatterdrive.app import LocalControlApp


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
