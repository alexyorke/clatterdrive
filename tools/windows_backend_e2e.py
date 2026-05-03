from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from contextlib import suppress
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _request(method: str, url: str, data: bytes | None = None) -> bytes:
    request = urllib.request.Request(url, data=data, method=method)
    with urllib.request.urlopen(request, timeout=8.0) as response:
        return response.read()


def _wait_ready(process: subprocess.Popen[str], timeout_s: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    lines: list[str] = []
    while time.monotonic() < deadline:
        line = process.stdout.readline() if process.stdout is not None else ""
        if not line:
            if process.poll() is not None:
                raise RuntimeError(f"backend exited early with code {process.returncode}: {' '.join(lines[-20:])}")
            time.sleep(0.05)
            continue
        lines.append(line.strip())
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "ready":
            return payload
    raise RuntimeError(f"backend did not become ready: {' '.join(lines[-20:])}")


def _launch_backend(backend_exe: Path | None, port: int, backing_dir: Path, tee_path: Path, trace_path: Path) -> subprocess.Popen[str]:
    common_args = [
        "serve",
        "--json-status",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--backing-dir",
        str(backing_dir),
        "--audio",
        "off",
        "--audio-tee-path",
        str(tee_path),
        "--event-trace-path",
        str(trace_path),
        "--ready",
        "--sync-power-on",
    ]
    if backend_exe is None:
        command = [sys.executable, "-m", "clatterdrive", *common_args]
    else:
        command = [str(backend_exe), *common_args]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def _shutdown_backend(process: subprocess.Popen[str], base_url: str | None = None) -> None:
    if base_url is not None and process.poll() is None:
        with suppress(Exception):
            _request("POST", f"{base_url}/.clatterdrive/shutdown", b"")
        try:
            process.wait(timeout=8.0)
            return
        except subprocess.TimeoutExpired:
            pass
    process.terminate()
    try:
        process.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8.0)


def _verify_wav(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"tee WAV not written: {path}")
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnframes() <= 0:
            raise AssertionError("tee WAV has no frames")


def _workspace_paths(root: Path, *, space_paths: bool) -> tuple[Path, Path, Path]:
    if space_paths:
        return root / "backing folder", root / "audio tee.wav", root / "event trace.json"
    return root / "backing", root / "audio.wav", root / "events.json"


def run_backend_e2e(backend_exe: Path | None = None, *, space_paths: bool = False) -> dict[str, object]:
    prefix = "clatterdrive e2e spaces " if space_paths else "clatterdrive-e2e-"
    with tempfile.TemporaryDirectory(prefix=prefix) as temp:
        root = Path(temp)
        backing_dir, tee_path, trace_path = _workspace_paths(root, space_paths=space_paths)
        port = _free_port()
        process = _launch_backend(backend_exe, port, backing_dir, tee_path, trace_path)
        try:
            ready = _wait_ready(process, 20.0)
            base_url = f"http://127.0.0.1:{port}"
            payload = b"hello from packaged clatterdrive e2e"
            _request("MKCOL", f"{base_url}/e2e/")
            _request("PUT", f"{base_url}/e2e/file.bin", payload)
            downloaded = _request("GET", f"{base_url}/e2e/file.bin")
            if downloaded != payload:
                raise AssertionError("downloaded bytes did not match uploaded bytes")
            _request("DELETE", f"{base_url}/e2e/file.bin")
        finally:
            _shutdown_backend(process, f"http://127.0.0.1:{port}")
        _verify_wav(tee_path)
        if not trace_path.exists() or trace_path.stat().st_size <= 0:
            raise AssertionError("event trace missing or empty")
        return {
            "ok": True,
            "port": port,
            "ready": ready,
            "space_paths": space_paths,
            "tee_bytes": tee_path.stat().st_size,
            "trace_bytes": trace_path.stat().st_size,
        }


def _candidate_drive_letters() -> list[str]:
    used = set()
    result = subprocess.run(["net.exe", "use"], capture_output=True, text=True, check=False)
    for match in re.finditer(r"(?<![A-Z])([A-Z]:)", result.stdout.upper()):
        used.add(match.group(1))
    candidates = []
    for letter in "ZYXWVUTSRQPONMLKJIHGFED":
        drive = f"{letter}:"
        if drive not in used and not Path(f"{drive}\\").exists():
            candidates.append(drive)
    return candidates


def _assert_webclient_running() -> None:
    result = subprocess.run(["sc.exe", "query", "WebClient"], capture_output=True, text=True, check=False)
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or "RUNNING" not in output:
        raise RuntimeError("Windows WebClient service must be installed and running for mapped-drive E2E.")


def _map_drive(unc: str) -> str:
    errors: list[str] = []
    for drive in _candidate_drive_letters():
        result = subprocess.run(["net.exe", "use", drive, unc, "/persistent:no"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return drive
        output = f"{result.stdout}\n{result.stderr}"
        if "System error 85" in output or "already in use" in output:
            errors.append(f"{drive}: already in use")
            continue
        raise RuntimeError(f"net use failed for {drive}: {output}")
    detail = "; ".join(errors) if errors else "no candidates"
    raise RuntimeError(f"no free drive letter found for mapped-drive E2E ({detail})")


def run_mapped_drive_e2e(backend_exe: Path | None = None, *, space_paths: bool = False) -> dict[str, object]:
    if platform.system() != "Windows":
        raise RuntimeError("mapped-drive E2E requires Windows")
    _assert_webclient_running()
    drive: str | None = None
    prefix = "clatterdrive map e2e spaces " if space_paths else "clatterdrive-map-e2e-"
    with tempfile.TemporaryDirectory(prefix=prefix) as temp:
        root = Path(temp)
        backing_dir, tee_path, trace_path = _workspace_paths(root, space_paths=space_paths)
        port = _free_port()
        process = _launch_backend(backend_exe, port, backing_dir, tee_path, trace_path)
        mounted = False
        try:
            ready = _wait_ready(process, 20.0)
            unc = f"\\\\127.0.0.1@{port}\\DavWWWRoot"
            drive = _map_drive(unc)
            mounted = True
            mapped_file = Path(f"{drive}\\mapped-e2e.bin")
            payload = b"hello from mapped clatterdrive e2e"
            mapped_file.write_bytes(payload)
            time.sleep(0.3)
            if mapped_file.read_bytes() != payload:
                raise AssertionError("mapped-drive read did not match written bytes")
            base_url = f"http://127.0.0.1:{port}"
            downloaded = _request("GET", f"{base_url}/mapped-e2e.bin")
            if downloaded != payload:
                raise AssertionError("WebDAV GET did not match mapped-drive write")
            mapped_file.unlink()
        finally:
            if mounted and drive is not None:
                subprocess.run(["net.exe", "use", drive, "/delete", "/y"], capture_output=True, text=True, check=False)
            _shutdown_backend(process, f"http://127.0.0.1:{port}")
        _verify_wav(tee_path)
        if not trace_path.exists() or trace_path.stat().st_size <= 0:
            raise AssertionError("event trace missing or empty")
        return {
            "ok": True,
            "mapped_drive": drive,
            "port": port,
            "ready": ready,
            "space_paths": space_paths,
            "tee_bytes": tee_path.stat().st_size,
            "trace_bytes": trace_path.stat().st_size,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a packaged-or-source ClatterDrive backend E2E smoke test.")
    parser.add_argument("--backend-exe", type=Path, default=None)
    parser.add_argument("--mapped-drive", action="store_true", help="Use Windows WebClient and net use for a mapped-drive E2E.")
    parser.add_argument("--space-paths", action="store_true", help="Use backing, tee, and trace paths containing spaces.")
    args = parser.parse_args()
    result = (
        run_mapped_drive_e2e(args.backend_exe, space_paths=args.space_paths)
        if args.mapped_drive
        else run_backend_e2e(args.backend_exe, space_paths=args.space_paths)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
