from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from clatterdrive.audio import HDDAudioEngine, HDDAudioEvent
from tools.generate_audio_samples import (
    ROOT,
    ScenarioUpdater,
    _load_power_on_trace,
    startup_only_duration,
    update_startup_only,
    update_idle_to_standby_wake,
    update_metadata_storm,
    update_random_flush,
    update_sequential_read,
    update_spinup_idle,
    write_wav,
)
from tools.reference_audio import (
    COMMITTED_REPORT_DIR,
    analyze_reference_bundle,
    compare_startup_features,
    compute_audio_features,
    save_reference_analysis,
    write_startup_summary_svg,
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


def _rpm_time_to_fraction(times: np.ndarray, actual_rpm: np.ndarray, target_rpm: float, fraction: float) -> float:
    threshold = target_rpm * fraction
    indices = np.flatnonzero(actual_rpm >= threshold)
    if len(indices) == 0:
        return float(times[-1]) if len(times) else 0.0
    return float(times[int(indices[0])])


def _render_startup_only_diagnostics() -> tuple[FloatArray, dict[str, float]]:
    sample_rate = 22050
    engine = HDDAudioEngine(
        seed=0,
        sample_rate=sample_rate,
        drive_profile="desktop_7200_internal",
        acoustic_profile="drive_on_desk",
    )
    total_frames = int(startup_only_duration("desktop_7200_internal") * sample_rate)
    startup_event = HDDAudioEvent(
        rpm=0.0,
        emitted_at=0.0,
        target_rpm=7200.0,
        queue_depth=1,
        op_kind="background",
        power_state="starting",
        heads_loaded=False,
        servo_mode="idle",
        transfer_activity=0.0,
        is_spinup=True,
    )
    diagnostics = engine.synthesizer.render_diagnostic_chunk(
        total_frames,
        scheduled_events=[(startup_event, int(0.85 * sample_rate))],
    )
    startup_metrics = {
        "rpm_time_to_50_s": _rpm_time_to_fraction(diagnostics.time_s, diagnostics.actual_rpm, 7200.0, 0.50),
        "rpm_time_to_90_s": _rpm_time_to_fraction(diagnostics.time_s, diagnostics.actual_rpm, 7200.0, 0.90),
        "rpm_time_to_99_s": _rpm_time_to_fraction(diagnostics.time_s, diagnostics.actual_rpm, 7200.0, 0.99),
        "steady_fundamental_hz": float(np.median(diagnostics.actual_rpm[-2048:])) / 60.0,
    }
    return diagnostics.output, startup_metrics


def _scenario_specs() -> tuple[ScenarioSpec, ...]:
    return (
        ("startup-only-desktop", startup_only_duration("desktop_7200_internal"), update_startup_only, 31),
        ("spinup-idle-park", _load_power_on_trace()[1] + 5.25, update_spinup_idle, 7),
        ("idle-standby-wake", 7.0, update_idle_to_standby_wake, 19),
        ("metadata-storm", 6.0, update_metadata_storm, 23),
        ("sequential-read-stream", 6.0, update_sequential_read, 11),
        ("random-seek-journal-flush", 6.0, update_random_flush, 13),
    )


def _scenario_metrics(samples: FloatArray, sample_rate: int) -> dict[str, float]:
    features = compute_audio_features(samples, sample_rate, "desktop_7200_internal")
    return {
        "duration_s": float(features["duration_s"]),
        "rms": float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0,
        "peak": float(np.max(np.abs(samples))) if samples.size else 0.0,
        "first_audible_s": float(features["first_audible_s"]),
        "time_to_90_s": float(features["time_to_90_s"]),
        "spectral_centroid_hz": float(features["spectral_centroid_hz"]),
        "low_band_ratio": float(features["low_band_ratio"]),
        "transient_density": float(features["transient_density"]),
        "spectral_flux": float(features["spectral_flux"]),
        "bubbly_modulation_ratio": float(features["bubbly_modulation_ratio"]),
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
    startup_summary_path = AUDIO_BASELINE_DIR / "startup_reference_summary.json"
    startup_wav_path = AUDIO_BASELINE_DIR / "startup-only-desktop.wav"
    startup_diag_path = AUDIO_BASELINE_DIR / "startup-only-desktop.diagnostics.json"

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
    analyzed_references, reference_metadata = analyze_reference_bundle()
    reference_report_path = save_reference_analysis(reference_metadata)
    startup_samples, startup_metrics = _render_startup_only_diagnostics()
    write_wav(startup_wav_path, startup_samples, 22050)
    startup_diag_path.write_text(json.dumps(startup_metrics, indent=2), encoding="utf-8")
    startup_features = compute_audio_features(startup_samples, 22050, "desktop_7200_internal")
    startup_features["time_to_50_s"] = startup_metrics["rpm_time_to_50_s"]
    startup_features["time_to_90_s"] = startup_metrics["rpm_time_to_90_s"]
    startup_features["time_to_99_s"] = startup_metrics["rpm_time_to_99_s"]
    startup_features["steady_fundamental_hz"] = startup_metrics["steady_fundamental_hz"]
    startup_summary = compare_startup_features(
        startup_features,
        [(entry, features) for entry, features, _path in analyzed_references],
    )
    startup_summary_path.write_text(json.dumps(startup_summary, indent=2), encoding="utf-8")
    COMMITTED_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    committed_json = COMMITTED_REPORT_DIR / "startup_reference_summary.json"
    committed_svg = COMMITTED_REPORT_DIR / "startup_reference_summary.svg"
    committed_json.write_text(json.dumps(startup_summary, indent=2), encoding="utf-8")
    write_startup_summary_svg(committed_svg, startup_summary)
    print(f"wrote {call_graph_path.relative_to(ROOT)}")
    print(f"wrote {metrics_path.relative_to(ROOT)}")
    print(f"wrote {reference_report_path.relative_to(ROOT)}")
    print(f"wrote {startup_wav_path.relative_to(ROOT)}")
    print(f"wrote {startup_diag_path.relative_to(ROOT)}")
    print(f"wrote {startup_summary_path.relative_to(ROOT)}")
    print(f"wrote {committed_json.relative_to(ROOT)}")
    print(f"wrote {committed_svg.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
