"""Measure render deadlines and, optionally, actual device underruns.

Run independently of the test suite or other simulator instances. --live
plays sound on the configured output device; it is never enabled by default.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from clatterdrive.audio import HDDAudioEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.chunks < 1 or args.seconds <= 0:
        parser.error("chunks and seconds must be positive")
    engine = HDDAudioEngine(seed=7)
    engine.emit_telemetry(7200, seek_trigger=True, seek_dist=700)
    times = []
    for index in range(args.chunks + 8):
        if index % 8 == 0:
            engine.emit_telemetry(7200, seek_trigger=True, seek_dist=(index * 37) % 1100)
        start = time.perf_counter()
        engine.render_chunk(engine.chunk_size)
        if index >= 8:
            times.append((time.perf_counter() - start) * 1000)
    budget = engine.chunk_size / engine.fs * 1000
    report: dict[str, object] = {
        "sample_rate": engine.fs, "chunk_frames": engine.chunk_size,
        "deadline_ms": budget, "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)), "max_ms": max(times),
        "over_deadline_chunks": sum(value > budget for value in times),
    }
    engine.stop()
    failed = float(np.percentile(times, 95)) >= budget
    if args.live:
        engine = HDDAudioEngine(seed=7)
        try:
            engine.start()
            if not engine.output_enabled or engine.stream is None:
                raise RuntimeError("Live benchmark requires a real audio device and live audio enabled")
            end = time.monotonic() + args.seconds
            while time.monotonic() < end:
                engine.emit_telemetry(7200, seek_trigger=True, seek_dist=700)
                time.sleep(0.15)
        finally:
            engine.stop()
        health = engine.playback_health()
        report["playback"] = health
        failed = failed or bool(health["buffer_underruns"] or health["device_underflows"] or health["render_error"])
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
