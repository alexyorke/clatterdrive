from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "clatterdrive-e2e"
RUNTIME_DIR = ROOT / ".runtime" / "docker-e2e"
BACKING_DIR = RUNTIME_DIR / "backing"
ARTIFACT_DIR = RUNTIME_DIR / "runtime"
WAV_PATH = ARTIFACT_DIR / "webdav.wav"
EVENTS_PATH = ARTIFACT_DIR / "events.json"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _compose_env(port: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "FAKE_HDD_PUBLISHED_PORT": str(port),
            "FAKE_HDD_AUDIO": "off",
            "FAKE_HDD_AUDIO_TEE_PATH": "/runtime/webdav.wav",
            "FAKE_HDD_EVENT_TRACE_PATH": "/runtime/events.json",
            "FAKE_HDD_COLD_START": "off",
            "FAKE_HDD_ASYNC_POWER_ON": "off",
            "FAKE_HDD_BACKING_VOLUME": "./.runtime/docker-e2e/backing",
            "FAKE_HDD_RUNTIME_VOLUME": "./.runtime/docker-e2e/runtime",
        }
    )
    return env


def _run_compose(args: list[str], env: dict[str, str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "-p", PROJECT_NAME, *args],
        cwd=ROOT,
        env=env,
        check=check,
        text=True,
    )


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
        with urllib.request.urlopen(request, timeout=8.0) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        try:
            response_body = exc.read()
        except OSError:
            response_body = b""
        return exc.code, response_body, dict(exc.headers)


def _wait_for_server(base_url: str) -> None:
    deadline = time.monotonic() + 45.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, _body, _headers = _request(base_url, "PROPFIND", "/", headers={"Depth": "0"})
            if status == 207:
                return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {base_url}") from last_error


def _exercise_webdav(base_url: str) -> None:
    status, _body, _headers = _request(base_url, "MKCOL", "/e2e")
    if status != 201:
        raise RuntimeError(f"MKCOL failed with HTTP {status}")

    payload = bytes(range(256)) * 1024
    status, _body, _headers = _request(base_url, "PUT", "/e2e/large.bin", payload)
    if status not in {200, 201, 204}:
        raise RuntimeError(f"large PUT failed with HTTP {status}")

    status, body, _headers = _request(base_url, "GET", "/e2e/large.bin")
    if status != 200 or body != payload:
        raise RuntimeError(f"GET failed with HTTP {status} and {len(body)} bytes")

    status, _body, _headers = _request(base_url, "MKCOL", "/e2e/small")
    if status != 201:
        raise RuntimeError(f"small MKCOL failed with HTTP {status}")
    for index in range(16):
        status, _body, _headers = _request(base_url, "PUT", f"/e2e/small/file-{index:02d}.bin", b"s" * 1024)
        if status not in {200, 201, 204}:
            raise RuntimeError(f"small PUT {index} failed with HTTP {status}")

    status, _body, _headers = _request(base_url, "MKCOL", "/e2e/frag")
    if status != 201:
        raise RuntimeError(f"frag MKCOL failed with HTTP {status}")
    for index in range(10):
        status, _body, _headers = _request(base_url, "PUT", f"/e2e/frag/filler-{index:02d}.bin", b"f" * 4096)
        if status not in {200, 201, 204}:
            raise RuntimeError(f"filler PUT {index} failed with HTTP {status}")
    for index in range(0, 10, 2):
        status, _body, _headers = _request(base_url, "DELETE", f"/e2e/frag/filler-{index:02d}.bin")
        if status != 204:
            raise RuntimeError(f"filler DELETE {index} failed with HTTP {status}")
    fragmented_payload = b"x" * (6 * 4096)
    status, _body, _headers = _request(base_url, "PUT", "/e2e/frag/fragmented.bin", fragmented_payload)
    if status not in {200, 201, 204}:
        raise RuntimeError(f"fragmented PUT failed with HTTP {status}")
    status, body, _headers = _request(base_url, "GET", "/e2e/frag/fragmented.bin")
    if status != 200 or body != fragmented_payload:
        raise RuntimeError(f"fragmented GET failed with HTTP {status} and {len(body)} bytes")

    status, _body, _headers = _request(base_url, "MKCOL", "/e2e/list")
    if status != 201:
        raise RuntimeError(f"list MKCOL failed with HTTP {status}")
    for index in range(96):
        status, _body, _headers = _request(base_url, "PUT", f"/e2e/list/entry-{index:03d}.txt", b"l")
        if status not in {200, 201, 204}:
            raise RuntimeError(f"list PUT {index} failed with HTTP {status}")
    status, body, _headers = _request(base_url, "PROPFIND", "/e2e/list", headers={"Depth": "1"})
    if status != 207 or b"entry-095.txt" not in body:
        raise RuntimeError(f"large directory PROPFIND failed with HTTP {status}")


def _wav_metrics(path: Path) -> tuple[int, float, float]:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)
    if not raw:
        return frames, 0.0, 0.0
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    rms = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5 / 32767.0
    peak = max(abs(sample) for sample in samples) / 32767.0
    return frames, float(rms), float(peak)


def _load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    if not isinstance(events, list):
        raise RuntimeError("Event trace does not contain an events list")
    return [event for event in events if isinstance(event, dict)]


def _validate_artifacts() -> None:
    if not EVENTS_PATH.exists():
        raise RuntimeError(f"Missing event trace: {EVENTS_PATH}")
    events = _load_events(EVENTS_PATH)
    kinds = {str(event.get("op_kind")) for event in events}
    if "data" not in kinds or not ({"metadata", "journal"} & kinds):
        raise RuntimeError(f"Event trace did not capture expected WebDAV media events: {sorted(kinds)}")
    max_block_count = max(int(event.get("block_count", 0) or 0) for event in events)
    transfer_block_count = sum(
        int(event.get("block_count", 0) or 0)
        for event in events
        if event.get("op_kind") in {"data", "writeback"}
    )
    max_transfer_ms = max(float(event.get("transfer_ms", 0.0) or 0.0) for event in events)
    max_directory_entries = max(int(event.get("directory_entry_count", 0) or 0) for event in events)
    max_fragmentation = max(int(event.get("fragmentation_score", 0) or 0) for event in events)
    metadata_events = [event for event in events if event.get("op_kind") in {"journal", "metadata"}]
    if len(metadata_events) < 16:
        raise RuntimeError(f"Expected many small-file metadata events, saw {len(metadata_events)}")
    if transfer_block_count < 64 or max_transfer_ms <= 0.0:
        raise RuntimeError(
            "Expected large transfer detail, "
            f"transfer_blocks={transfer_block_count} max_transfer_ms={max_transfer_ms}"
        )
    if max_directory_entries < 96:
        raise RuntimeError(f"Expected large directory detail, entries={max_directory_entries}")
    if max_fragmentation <= 1:
        raise RuntimeError(f"Expected fragmented workload detail, fragmentation={max_fragmentation}")

    if not WAV_PATH.exists():
        raise RuntimeError(f"Missing audio tee WAV: {WAV_PATH}")
    frames, rms, peak = _wav_metrics(WAV_PATH)
    if frames <= 0 or rms <= 0.0005 or peak <= 0.002:
        raise RuntimeError(f"Audio tee is too quiet or empty: frames={frames} rms={rms:.6f} peak={peak:.6f}")

    print(
        f"Docker WebDAV audio smoke passed: events={len(events)} "
        f"kinds={sorted(kinds)} max_blocks={max_block_count} transfer_blocks={transfer_block_count} "
        f"max_dir_entries={max_directory_entries} max_frag={max_fragmentation} "
        f"frames={frames} rms={rms:.6f} peak={peak:.6f}"
    )


def main() -> None:
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    BACKING_DIR.mkdir(parents=True)
    ARTIFACT_DIR.mkdir(parents=True)

    port = _free_port()
    env = _compose_env(port)
    stopped = False
    try:
        _run_compose(["up", "--build", "-d"], env)
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_server(base_url)
        _exercise_webdav(base_url)
        time.sleep(1.0)
        _run_compose(["stop", "--timeout", "10"], env)
        stopped = True
        _validate_artifacts()
    finally:
        if not stopped:
            _run_compose(["stop", "--timeout", "10"], env, check=False)
        _run_compose(["down", "--remove-orphans"], env, check=False)


if __name__ == "__main__":
    main()
