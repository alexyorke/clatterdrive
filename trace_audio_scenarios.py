from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from clatterdrive.audio import HDDAudioEngine
from clatterdrive.storage_events import StorageEventRecorder, storage_event_to_dict
from generate_audio_samples import (
    ROOT,
    ScenarioUpdater,
    _load_power_on_trace,
    update_copy_heavy,
    update_idle_to_standby_wake,
    update_metadata_storm,
    update_spinup_idle,
)


FloatArray = npt.NDArray[np.float64]
TRACE_DIR = ROOT / ".runtime" / "traces"
ScenarioSpec = tuple[str, float, ScenarioUpdater, int]
EventPayload = dict[str, float | int | bool | str | None]
DiagnosticPayload = dict[str, FloatArray]


def _diagnostics_payload(diagnostics: DiagnosticPayload, sample_rate: int) -> dict[str, object]:
    return {
        "sample_rate": sample_rate,
        **{key: value.tolist() for key, value in diagnostics.items()},
    }


def _concat_diagnostics(segments: list[DiagnosticPayload]) -> DiagnosticPayload:
    if not segments:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "time_s": empty,
            "target_rpm": empty,
            "actual_rpm": empty,
            "actuator_pos": empty,
            "actuator_torque": empty,
            "structure_base_velocity": empty,
            "structure_cover_velocity": empty,
            "structure_enclosure_velocity": empty,
            "structure_desk_velocity": empty,
            "output": empty,
        }
    return {
        key: np.concatenate([segment[key] for segment in segments])
        for key in segments[0]
    }


def _line_points(values: FloatArray, *, left: float, top: float, width: float, height: float) -> str:
    if len(values) == 0:
        return ""
    x_values = np.linspace(left, left + width, len(values), dtype=np.float64)
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 1e-9)
    y_values = top + height - ((values - low) / span) * height
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(x_values, y_values, strict=True))


def _event_time_s(event: EventPayload) -> float:
    return float(cast(float | int, event["time_s"]))


def _queue_depth_series(times: FloatArray, events: list[EventPayload]) -> FloatArray:
    if len(times) == 0:
        return np.zeros(0, dtype=np.float64)
    series = np.ones(len(times), dtype=np.float64)
    if not events:
        return series
    event_index = 0
    current_depth = 1.0
    sorted_events = sorted(events, key=_event_time_s)
    for index, time_s in enumerate(times):
        while event_index < len(sorted_events) and _event_time_s(sorted_events[event_index]) <= float(time_s):
            current_depth = float(cast(float | int, sorted_events[event_index]["queue_depth"]))
            event_index += 1
        series[index] = current_depth
    return series


def _power_segments(total_duration_s: float, events: list[EventPayload]) -> list[tuple[float, float, str]]:
    if total_duration_s <= 0.0:
        return []
    if not events:
        return [(0.0, total_duration_s, "active")]
    sorted_events = sorted(events, key=_event_time_s)
    segments: list[tuple[float, float, str]] = []
    start = 0.0
    current_state = str(sorted_events[0].get("power_state") or "active")
    for event in sorted_events[1:]:
        time_s = _event_time_s(event)
        if time_s > start:
            segments.append((start, time_s, current_state))
        current_state = str(event.get("power_state") or current_state)
        start = time_s
    segments.append((start, total_duration_s, current_state))
    return segments


def _render_trace_svg(
    path: Path,
    diagnostics: DiagnosticPayload,
    events: list[EventPayload],
) -> None:
    width = 1200.0
    height = 840.0
    margin_left = 72.0
    plot_width = width - margin_left - 40.0
    panel_height = 170.0
    top = 48.0

    queue_depth = _queue_depth_series(diagnostics["time_s"], events)
    panels = (
        ("RPM", diagnostics["actual_rpm"], "#1f77b4"),
        ("Queue Depth", queue_depth, "#d62728"),
        ("Head Position", diagnostics["actuator_pos"], "#2ca02c"),
        ("Output", diagnostics["output"], "#9467bd"),
    )

    power_top = top + len(panels) * (panel_height + 28.0)
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Consolas,monospace;font-size:14px} .label{font-weight:bold}</style>',
        '<rect x="0" y="0" width="100%" height="100%" fill="#fafafa"/>',
        '<text x="40" y="28" class="label">ClatterDrive trace view</text>',
    ]

    for index, (label, values, color) in enumerate(panels):
        panel_top = top + index * (panel_height + 28.0)
        svg_parts.append(f'<text x="40" y="{panel_top + 14:.1f}" class="label">{label}</text>')
        svg_parts.append(
            f'<rect x="{margin_left}" y="{panel_top}" width="{plot_width}" height="{panel_height}" fill="white" stroke="#cccccc"/>'
        )
        points = _line_points(values, left=margin_left, top=panel_top, width=plot_width, height=panel_height)
        if points:
            svg_parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{points}"/>'
            )

    svg_parts.append(f'<text x="40" y="{power_top + 14:.1f}" class="label">Power State</text>')
    svg_parts.append(
        f'<rect x="{margin_left}" y="{power_top}" width="{plot_width}" height="48" fill="white" stroke="#cccccc"/>'
    )
    power_colors = {
        "active": "#4caf50",
        "starting": "#ff9800",
        "standby": "#757575",
        "spinning_down": "#9c27b0",
        "low_rpm_idle": "#03a9f4",
        "unloaded_idle": "#8bc34a",
    }
    total_duration_s = float(diagnostics["time_s"][-1]) if len(diagnostics["time_s"]) else 0.0
    for start, end, state in _power_segments(total_duration_s, events):
        if end <= start:
            continue
        x = margin_left + (start / max(total_duration_s, 1e-9)) * plot_width
        w = ((end - start) / max(total_duration_s, 1e-9)) * plot_width
        color = power_colors.get(state, "#607d8b")
        svg_parts.append(
            f'<rect x="{x:.2f}" y="{power_top + 8:.2f}" width="{max(w, 1.0):.2f}" height="32" fill="{color}" opacity="0.75"/>'
        )
        svg_parts.append(
            f'<text x="{x + 4:.2f}" y="{power_top + 28:.2f}" fill="white">{state}</text>'
        )

    svg_parts.append("</svg>")
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def _trace_events(recorder: StorageEventRecorder) -> list[EventPayload]:
    events = recorder.snapshot()
    if not events:
        return []
    origin = min(event.emitted_at for event in events)
    payload = []
    for event in events:
        item = storage_event_to_dict(event)
        item["time_s"] = event.emitted_at - origin
        payload.append(item)
    return payload


def render_trace_scenario(
    name: str,
    duration_s: float,
    update_func: ScenarioUpdater,
    *,
    seed: int = 7,
) -> Path:
    recorder = StorageEventRecorder()
    engine = HDDAudioEngine(sample_rate=44100, seed=seed, event_trace_sink=recorder)
    total_frames = int(duration_s * engine.fs)
    remaining_frames = total_frames
    emitted_flags: set[str] = set()
    diagnostics_segments: list[DiagnosticPayload] = []

    rendered_frames = 0
    while remaining_frames > 0:
        frames = min(engine.chunk_size, remaining_frames)
        current_time = rendered_frames / engine.fs
        update_func(engine, current_time, emitted_flags)
        _chunk, diagnostics = engine.render_chunk_with_diagnostics(frames)
        diagnostics_segments.append(
            {
                "time_s": diagnostics.time_s,
                "target_rpm": diagnostics.target_rpm,
                "actual_rpm": diagnostics.actual_rpm,
                "actuator_pos": diagnostics.actuator_pos,
                "actuator_torque": diagnostics.actuator_torque,
                "structure_base_velocity": diagnostics.structure_base_velocity,
                "structure_cover_velocity": diagnostics.structure_cover_velocity,
                "structure_enclosure_velocity": diagnostics.structure_enclosure_velocity,
                "structure_desk_velocity": diagnostics.structure_desk_velocity,
                "output": diagnostics.output,
            }
        )
        rendered_frames += frames
        remaining_frames -= frames

    diagnostics_payload = _concat_diagnostics(diagnostics_segments)
    event_payload = _trace_events(recorder)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = TRACE_DIR / f"{name}.trace.json"
    svg_path = TRACE_DIR / f"{name}.trace.svg"
    payload = {
        "scenario": name,
        "events": event_payload,
        "diagnostics": _diagnostics_payload(diagnostics_payload, engine.fs),
    }
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    _render_trace_svg(svg_path, diagnostics_payload, event_payload)
    return bundle_path


def _scenario_specs() -> tuple[ScenarioSpec, ...]:
    power_on_duration = _load_power_on_trace()[1] + 5.25
    return (
        ("spinup-idle-park", power_on_duration, update_spinup_idle, 7),
        ("idle-standby-wake", 7.0, update_idle_to_standby_wake, 19),
        ("metadata-storm", 6.0, update_metadata_storm, 23),
        ("copy-heavy-writeback", 6.0, update_copy_heavy, 17),
    )


def main() -> None:
    outputs = [render_trace_scenario(name, duration_s, update, seed=seed) for name, duration_s, update, seed in _scenario_specs()]
    for output in outputs:
        print(f"generated {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
