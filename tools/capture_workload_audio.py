from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import wave
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from clatterdrive.audio import HDDAudioEngine
from clatterdrive.audio.workload import expand_workload_event
from clatterdrive.hdd import VirtualHDD
from clatterdrive.runtime.deps import RuntimeDeps
from clatterdrive.storage_events import StorageEvent, StorageEventRecorder, StorageEventSink
from tools.generate_audio_samples import normalize_demo_audio, write_wav


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "audio" / "workloads"
DEFAULT_RUNTIME_DIR = ROOT / ".runtime" / "workload-audio"
WORKLOAD_RMS_CAP = 0.050
WORKLOAD_PEAK_CAP = 0.45


@dataclass
class ScriptClock:
    current_time: float = 0.0

    def now(self) -> float:
        return self.current_time


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    title: str
    description: str
    wav_path: Path
    events_path: Path
    metrics_path: Path
    duration_s: float
    event_count: int
    expanded_event_count: int
    metadata_event_count: int
    max_fragmentation_score: int
    max_directory_entry_count: int
    rms: float
    peak: float
    p95_lag_ms: float
    max_lag_ms: float


class TimelineStorageEventSink:
    def __init__(self, sinks: Sequence[StorageEventSink]) -> None:
        self.sinks = tuple(sinks)
        self.current_time_s = 0.0

    def publish_event(self, event: StorageEvent) -> None:
        staged = replace(event, emitted_at=self.current_time_s)
        for sink in self.sinks:
            sink.publish_event(staged)
        self.current_time_s += self._spacing_s(event)

    def _spacing_s(self, event: StorageEvent) -> float:
        transfer_s = max(0.0, event.transfer_ms / 1000.0)
        blocks = max(1, event.block_count)
        if event.op_kind in {"journal", "metadata"}:
            directory_s = min(0.014, max(0, event.directory_entry_count) * 0.000055)
            return min(0.050, 0.024 + directory_s + 0.00045 * math.log2(blocks + 1))
        if event.op_kind in {"data", "writeback"}:
            return min(0.070, max(0.010, transfer_s * 0.90))
        if event.op_kind == "flush" or event.is_flush:
            return min(0.055, max(0.018, transfer_s))
        return min(0.030, max(0.008, transfer_s))


Workload = Callable[[VirtualHDD], None]


def _large_sequential_write(vhdd: VirtualHDD) -> None:
    vhdd.access_file("/large-sequential.bin", 0, 4 * 1024 * 1024, is_write=True)
    vhdd.sync_all()


def _small_file_metadata_storm(vhdd: VirtualHDD) -> None:
    vhdd.create_directory("/small")
    for index in range(96):
        vhdd.access_file(f"/small/file-{index:03d}.txt", 0, 1024 + (index % 7) * 73, is_write=True)
    vhdd.list_directory("/small")
    vhdd.sync_all()


def _fragmented_rewrite_churn(vhdd: VirtualHDD) -> None:
    vhdd.create_directory("/frag")
    for index in range(20):
        vhdd.access_file(f"/frag/filler-{index:02d}.bin", 0, 4096 + (index % 3) * 4096, is_write=True)
    vhdd.sync_all()
    for index in range(0, 20, 2):
        vhdd.delete_path(f"/frag/filler-{index:02d}.bin")
    vhdd.access_file("/frag/fragmented.bin", 0, 128 * 1024, is_write=True)
    vhdd.access_file("/frag/fragmented.bin", 16 * 1024, 64 * 1024, is_write=True)
    vhdd.access_file("/frag/fragmented.bin", 0, 128 * 1024, is_write=False)
    vhdd.sync_all()


def _sync_heavy_flushes(vhdd: VirtualHDD) -> None:
    vhdd.create_directory("/sync")
    for index in range(24):
        vhdd.access_file(f"/sync/record-{index:02d}.bin", 0, 4096, is_write=True, sync=True)
        vhdd.touch_metadata(f"/sync/record-{index:02d}.bin", source="mtime_update")
    vhdd.list_directory("/sync")
    vhdd.sync_all()


SCENARIOS: tuple[tuple[str, str, str, Workload], ...] = (
    (
        "large-sequential-write",
        "Large Sequential Write",
        "One contiguous multi-megabyte file: a short metadata preamble followed by smoother writeback.",
        _large_sequential_write,
    ),
    (
        "small-file-metadata-storm",
        "Small File Metadata Storm",
        "Many tiny creates in one directory: dense journal, dentry, inode, and bitmap burstlets.",
        _small_file_metadata_storm,
    ),
    (
        "fragmented-rewrite-churn",
        "Fragmented Rewrite Churn",
        "Delete holes, rewrite, and read back: seeks become more irregular as extent count rises.",
        _fragmented_rewrite_churn,
    ),
    (
        "sync-heavy-flushes",
        "Sync Heavy Flushes",
        "Small forced writes and metadata updates: tighter flush hits and repeated journal commits.",
        _sync_heavy_flushes,
    ),
)


def _wav_metrics(path: Path) -> tuple[float, float]:
    with wave.open(str(path), "rb") as wav_file:
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16).astype(np.float64) / 32767.0
    if samples.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(samples**2))), float(np.max(np.abs(samples)))


def _cap_workload_loudness(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    if rms <= 1e-9 or peak <= 1e-9:
        return samples
    scale = min(1.0, WORKLOAD_RMS_CAP / rms, WORKLOAD_PEAK_CAP / peak)
    return samples * scale


def _render_audio(engine: HDDAudioEngine, duration_s: float, clock: ScriptClock) -> np.ndarray:
    total_frames = math.ceil(duration_s * engine.fs)
    rendered: list[np.ndarray] = []
    frames_left = total_frames
    frame_cursor = 0
    while frames_left > 0:
        frames = min(engine.chunk_size, frames_left)
        clock.current_time = frame_cursor / engine.fs
        rendered.append(engine.render_chunk(frames))
        frame_cursor += frames
        frames_left -= frames
    return np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.float64)


def capture_scenario(
    name: str,
    title: str,
    description: str,
    workload: Workload,
    *,
    output_dir: Path,
    runtime_dir: Path,
    seed: int = 17,
) -> ScenarioResult:
    scenario_runtime = runtime_dir / name
    if scenario_runtime.exists():
        shutil.rmtree(scenario_runtime)
    scenario_runtime.mkdir(parents=True, exist_ok=True)

    clock = ScriptClock()
    audio = HDDAudioEngine(
        seed=seed,
        max_pending_events=20000,
        deps=RuntimeDeps(clock=clock),
    )
    recorder = StorageEventRecorder(max_events=50000)
    timeline = TimelineStorageEventSink([audio, recorder])
    vhdd = VirtualHDD(str(scenario_runtime / "backing"), latency_scale=0.0, event_sink=timeline)
    try:
        workload(vhdd)
        vhdd.sync_all()
    finally:
        vhdd.stop()

    events = recorder.snapshot()
    duration_s = max(1.2, timeline.current_time_s + 1.0)
    samples = _cap_workload_loudness(normalize_demo_audio(_render_audio(audio, duration_s, clock), target_peak=0.86))

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{name}.wav"
    events_path = output_dir / f"{name}.events.json"
    metrics_path = output_dir / f"{name}.metrics.json"
    write_wav(wav_path, samples, audio.fs)
    recorder.export_json(str(events_path))
    rms, peak = _wav_metrics(wav_path)
    lag = audio.audio_lag_snapshot()
    expanded_event_count = sum(len(expand_workload_event(event, audio.fs)) for event in events)
    metadata_event_count = len([event for event in events if event.op_kind in {"journal", "metadata"}])
    max_fragmentation_score = max((event.fragmentation_score for event in events), default=0)
    max_directory_entry_count = max((event.directory_entry_count for event in events), default=0)
    metrics = {
        "name": name,
        "title": title,
        "duration_s": duration_s,
        "event_count": len(events),
        "expanded_event_count": expanded_event_count,
        "metadata_event_count": metadata_event_count,
        "max_fragmentation_score": max_fragmentation_score,
        "max_directory_entry_count": max_directory_entry_count,
        "rms": rms,
        "rms_cap": WORKLOAD_RMS_CAP,
        "peak": peak,
        "peak_cap": WORKLOAD_PEAK_CAP,
        "audio_lag": lag,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return ScenarioResult(
        name=name,
        title=title,
        description=description,
        wav_path=wav_path,
        events_path=events_path,
        metrics_path=metrics_path,
        duration_s=duration_s,
        event_count=len(events),
        expanded_event_count=expanded_event_count,
        metadata_event_count=metadata_event_count,
        max_fragmentation_score=max_fragmentation_score,
        max_directory_entry_count=max_directory_entry_count,
        rms=rms,
        peak=peak,
        p95_lag_ms=float(lag["p95_lag_ms"]),
        max_lag_ms=float(lag["max_lag_ms"]),
    )


def write_listening_page(results: Sequence[ScenarioResult], output_dir: Path) -> Path:
    rows = []
    for result in results:
        rows.append(
            f"""
      <section class="scenario">
        <h2>{html.escape(result.title)}</h2>
        <p>{html.escape(result.description)}</p>
        <audio controls preload="metadata" src="{html.escape(result.wav_path.name)}"></audio>
        <dl>
          <div><dt>Physical events</dt><dd>{result.event_count}</dd></div>
          <div><dt>Audio burstlets</dt><dd>{result.expanded_event_count}</dd></div>
          <div><dt>Metadata events</dt><dd>{result.metadata_event_count}</dd></div>
          <div><dt>Max fragmentation</dt><dd>{result.max_fragmentation_score}</dd></div>
          <div><dt>RMS</dt><dd>{result.rms:.4f}</dd></div>
          <div><dt>p95 lag</dt><dd>{result.p95_lag_ms:.1f} ms</dd></div>
        </dl>
        <p class="links"><a href="{html.escape(result.events_path.name)}">event trace</a> | <a href="{html.escape(result.metrics_path.name)}">metrics</a></p>
      </section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ClatterDrive Workload Audio Samples</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, system-ui, sans-serif; background: #f6f7f9; color: #17191c; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    .intro {{ margin: 0 0 24px; color: #4d535b; }}
    .scenario {{ border: 1px solid #d8dde4; background: #fff; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    h2 {{ font-size: 18px; margin: 0 0 6px; }}
    p {{ line-height: 1.45; }}
    audio {{ width: 100%; margin: 8px 0 12px; }}
    dl {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 0; }}
    dt {{ font-size: 12px; color: #68717c; }}
    dd {{ margin: 2px 0 0; font-weight: 650; }}
    .links {{ margin-bottom: 0; font-size: 13px; }}
  </style>
</head>
<body>
  <main>
    <h1>ClatterDrive Workload Audio Samples</h1>
    <p class="intro">Captured from simulated file workloads. Physical storage events are expanded into proportional audio burstlets at render time. The reference player keeps the original metadata storm nearby for comparison.</p>
      <section class="scenario">
        <h2>Original Metadata Storm Reference</h2>
        <p>The checked-in default storm motif that the workload captures are tuned around.</p>
        <audio controls preload="metadata" src="../metadata-storm.wav"></audio>
      </section>
{''.join(rows)}
  </main>
</body>
</html>
"""
    output_path = output_dir / "index.html"
    output_path.write_text(page, encoding="utf-8")
    return output_path


def capture_workload_audio(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
) -> list[ScenarioResult]:
    results = [
        capture_scenario(name, title, description, workload, output_dir=output_dir, runtime_dir=runtime_dir)
        for name, title, description, workload in SCENARIOS
    ]
    write_listening_page(results, output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture ClatterDrive workload audio samples and a listening page.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    args = parser.parse_args()
    results = capture_workload_audio(output_dir=args.output_dir, runtime_dir=args.runtime_dir)
    print(json.dumps([asdict(result) | {"wav_path": str(result.wav_path)} for result in results], default=str, indent=2))


if __name__ == "__main__":
    main()
