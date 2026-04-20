from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from clatterdrive.audio import HDDAudioEngine
from generate_audio_samples import (
    ROOT,
    ScenarioUpdater,
    _load_power_on_trace,
    update_idle_to_standby_wake,
    update_metadata_storm,
    update_random_flush,
    update_sequential_read,
    update_spinup_idle,
)


AUDIO_BASELINE_DIR = ROOT / ".runtime" / "audio-baseline"
FloatArray = np.ndarray
ScenarioSpec = tuple[str, float, ScenarioUpdater, int]


def _module_defs(module_path: Path) -> tuple[set[str], list[tuple[str, str]]]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    edges: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        caller = node.name
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in names:
                edges.append((caller, child.func.id))
    return names, edges


def _audio_call_graph() -> dict[str, object]:
    modules = [
        ROOT / "clatterdrive" / "audio" / "commands.py",
        ROOT / "clatterdrive" / "audio" / "core.py",
        ROOT / "clatterdrive" / "audio" / "engine.py",
    ]
    graph_modules: list[dict[str, object]] = []
    graph_edges: list[dict[str, str]] = []
    for module in modules:
        names, edges = _module_defs(module)
        graph_modules.append({"path": str(module.relative_to(ROOT)), "functions": sorted(names)})
        graph_edges.extend(
            {
                "module": str(module.relative_to(ROOT)),
                "caller": caller,
                "callee": callee,
            }
            for caller, callee in edges
        )
    return {"modules": graph_modules, "edges": graph_edges}


def _render_scenario(duration_s: float, update_func: ScenarioUpdater, seed: int) -> tuple[FloatArray, int]:
    engine = HDDAudioEngine(sample_rate=44100, seed=seed)
    frames_remaining = int(duration_s * engine.fs)
    rendered: list[FloatArray] = []
    emitted_flags: set[str] = set()
    while frames_remaining > 0:
        frames = min(engine.chunk_size, frames_remaining)
        current_time = (engine.synthesizer.state.sample_clock) / engine.fs
        update_func(engine, current_time, emitted_flags)
        rendered.append(engine.render_chunk(frames))
        frames_remaining -= frames
    return np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.float64), engine.fs


def _scenario_specs() -> tuple[ScenarioSpec, ...]:
    return (
        ("spinup-idle-park", _load_power_on_trace()[1] + 5.25, update_spinup_idle, 7),
        ("idle-standby-wake", 7.0, update_idle_to_standby_wake, 19),
        ("metadata-storm", 6.0, update_metadata_storm, 23),
        ("sequential-read-stream", 6.0, update_sequential_read, 11),
        ("random-seek-journal-flush", 6.0, update_random_flush, 13),
    )


def _scenario_metrics(samples: FloatArray, sample_rate: int) -> dict[str, float]:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
    centroid = float(np.sum(freqs * spectrum) / max(np.sum(spectrum), 1e-12))
    low_band = float(np.sum(spectrum[(freqs >= 20.0) & (freqs < 120.0)]))
    low_mid = float(np.sum(spectrum[(freqs >= 120.0) & (freqs < 500.0)]))
    return {
        "duration_s": len(samples) / sample_rate,
        "rms": float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0,
        "peak": float(np.max(np.abs(samples))) if samples.size else 0.0,
        "spectral_centroid_hz": centroid,
        "low_band_ratio": low_band / max(low_mid, 1e-12),
        "transient_density": float(np.mean(np.abs(np.diff(samples)))) if len(samples) > 1 else 0.0,
    }


def _pairwise_correlations(rendered: dict[str, FloatArray]) -> list[dict[str, float | str]]:
    names = list(rendered)
    payload: list[dict[str, float | str]] = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = rendered[left_name]
            right = rendered[right_name]
            length = min(len(left), len(right))
            correlation = float(np.corrcoef(left[:length], right[:length])[0, 1]) if length else 0.0
            payload.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "correlation": correlation,
                }
            )
    return payload


def main() -> None:
    AUDIO_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    call_graph_path = AUDIO_BASELINE_DIR / "callgraph.json"
    metrics_path = AUDIO_BASELINE_DIR / "metrics.json"

    rendered: dict[str, FloatArray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, duration_s, update_func, seed in _scenario_specs():
        samples, sample_rate = _render_scenario(duration_s, update_func, seed)
        rendered[name] = samples
        metrics[name] = _scenario_metrics(samples, sample_rate)

    call_graph_path.write_text(json.dumps(_audio_call_graph(), indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "scenarios": metrics,
                "pairwise_correlation": _pairwise_correlations(rendered),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {call_graph_path.relative_to(ROOT)}")
    print(f"wrote {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
