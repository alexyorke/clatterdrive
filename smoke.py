from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Final

from fake_hdd_fuse.runtime.paths import ROOT, workspace_tempdir


STARTUP_TIMEOUT_S: Final[float] = 20.0
REQUEST_TIMEOUT_S: Final[float] = 10.0


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _collect_process_output(process: subprocess.Popen[str]) -> str:
    try:
        stdout, _ = process.communicate(timeout=1.0)
        return stdout or ""
    except Exception:
        return ""


def _probe_server_ready(base_url: str) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=0.25):
            pass
    except OSError:
        return False

    try:
        status, _, _ = _request(base_url, "OPTIONS", "/")
        return 200 <= status < 500
    except urllib.error.HTTPError as exc:
        return 200 <= exc.code < 500
    except Exception:
        return False


def _wait_for_server(base_url: str, process: subprocess.Popen[str], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = _collect_process_output(process)
            raise RuntimeError(f"server exited early with code {process.returncode}\n{output}")
        if _probe_server_ready(base_url):
            return
        time.sleep(0.1)
    raise TimeoutError(f"server did not become ready within {timeout_s:.1f}s")


def _run_cli_probe(base_url: str) -> bool:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        print("CLI probe skipped: curl not available.")
        return False

    mkcol = subprocess.run(
        [curl, "-sS", "-X", "MKCOL", f"{base_url}/cli-probe"],
        capture_output=True,
        text=True,
        timeout=REQUEST_TIMEOUT_S,
        check=False,
    )
    if mkcol.returncode != 0:
        raise RuntimeError(f"curl MKCOL failed: {mkcol.stderr.strip()}")

    propfind = subprocess.run(
        [curl, "-sS", "-X", "PROPFIND", "-H", "Depth: 1", f"{base_url}/cli-probe"],
        capture_output=True,
        text=True,
        timeout=REQUEST_TIMEOUT_S,
        check=False,
    )
    if propfind.returncode != 0:
        raise RuntimeError(f"curl PROPFIND failed: {propfind.stderr.strip()}")
    if "cli-probe" not in propfind.stdout:
        raise AssertionError("curl PROPFIND response did not include the created collection")

    return True


def run_main_boot_smoke(exercise_cli: bool = True) -> None:
    port = _pick_free_port()
    with workspace_tempdir(prefix="fake-hdd-smoke-", subdir="smoke") as backing_dir:
        env = os.environ.copy()
        env["FAKE_HDD_AUDIO"] = "off"
        env["FAKE_HDD_PORT"] = str(port)
        env["FAKE_HDD_BACKING_DIR"] = str(backing_dir)

        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        failure: Exception | None = None
        try:
            _wait_for_server(base_url, process, STARTUP_TIMEOUT_S)
            print(f"Server smoke online at {base_url}")

            status, _, _ = _request(base_url, "MKCOL", "/docs")
            if status != 201:
                raise AssertionError(f"MKCOL /docs expected 201, got {status}")

            payload = b"smoke payload"
            status, _, _ = _request(base_url, "PUT", "/docs/data.txt", payload)
            if status not in {200, 201, 204}:
                raise AssertionError(f"PUT /docs/data.txt expected 200/201/204, got {status}")

            status, body, _ = _request(base_url, "GET", "/docs/data.txt")
            if status != 200 or body != payload:
                raise AssertionError("GET /docs/data.txt returned unexpected content")

            status, body, _ = _request(base_url, "PROPFIND", "/docs", headers={"Depth": "1"})
            if status != 207 or b"data.txt" not in body:
                raise AssertionError("PROPFIND /docs did not enumerate the created file")

            if exercise_cli:
                _run_cli_probe(base_url)

            print("Smoke validation passed.")
        except Exception as exc:
            failure = exc
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            output = _collect_process_output(process).strip()
        if failure is not None:
            message = str(failure)
            if output:
                message = f"{message}\n{output}"
            raise RuntimeError(message) from failure


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick local smoke validation for fake_hdd_fuse.")
    parser.add_argument(
        "--skip-cli",
        action="store_true",
        help="Skip the optional curl-based WebDAV CLI probe.",
    )
    args = parser.parse_args()
    run_main_boot_smoke(exercise_cli=not args.skip_cli)


if __name__ == "__main__":
    main()
