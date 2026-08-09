from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

from clatterdrive.audio import HDDAudioEngine, HDDAudioEvent
from clatterdrive.runtime.deps import NoOpSleeper, RuntimeDeps
from tools.generate_audio_samples import (
    ScriptClock,
    normalize_demo_audio,
    render_chunk,
    startup_only_duration,
    update_metadata_storm,
)
from tools.reference_audio import compute_audio_features, evaluate_startup_acceptance, write_startup_summary_svg


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DRIVE_PROFILE = "enterprise_7200_bare"
REFERENCE_ACOUSTIC_PROFILE = "bare_drive_lab"
REFERENCE_BUCKET = "enterprise_ultrastar"
REFERENCE_SUMMARY_PATH = ROOT / "docs" / "reference-calibration" / "startup_reference_summary.json"
REFERENCE_SUMMARY_SVG_PATH = ROOT / "docs" / "reference-calibration" / "startup_reference_summary.svg"


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if values.size else 0.0


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16).astype(np.float64) / 32767.0
        return samples, wav_file.getframerate()


def _resample_series(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(length, dtype=np.float64)
    if len(values) == length:
        return values.astype(np.float64, copy=True)
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return np.interp(target, source, values).astype(np.float64)


def _resample_matrix(values: np.ndarray, frames: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0, frames), dtype=np.float64)
    return np.vstack([_resample_series(row, frames) for row in values]).astype(np.float64)


def _aligned_curve(features: dict[str, Any], key: str, frames: int) -> np.ndarray:
    values = np.asarray(features[key], dtype=np.float64)
    onset = int(features["onset_index"])
    return _resample_series(values[onset:], frames)


def _aligned_log_mel(features: dict[str, Any], frames: int) -> np.ndarray:
    values = np.asarray(features["log_mel"], dtype=np.float64)
    onset = int(features["onset_index"])
    return _resample_matrix(values[:, onset:], frames)


def render_metadata_storm() -> np.ndarray:
    clock = ScriptClock()
    engine = HDDAudioEngine(
        sample_rate=44100,
        seed=23,
        acoustic_profile="bare_drive_lab",
        deps=RuntimeDeps(clock=clock, sleeper=NoOpSleeper()),
    )
    remaining = int(6.0 * engine.fs)
    chunks: list[np.ndarray] = []
    emitted_flags: set[str] = set()
    while remaining > 0:
        frames = min(engine.chunk_size, remaining)
        current_time = (len(chunks) * engine.chunk_size) / engine.fs
        clock.current_time = current_time
        update_metadata_storm(engine, current_time, emitted_flags)
        chunks.append(render_chunk(engine, frames))
        remaining -= frames
    return normalize_demo_audio(np.concatenate(chunks), target_peak=0.92)


def metadata_storm_metrics() -> dict[str, float]:
    rendered = render_metadata_storm()
    golden, _sample_rate = _read_wav(ROOT / "samples" / "metadata-storm.wav")
    n = min(len(rendered), len(golden))
    delta = rendered[:n] - golden[:n]
    return {
        "correlation": float(np.corrcoef(rendered[:n], golden[:n])[0, 1]),
        "rms_delta": _rms(delta),
        "rendered_rms": _rms(rendered[:n]),
        "golden_rms": _rms(golden[:n]),
    }


def render_startup_only() -> np.ndarray:
    sample_rate = 22050
    engine = HDDAudioEngine(
        seed=0,
        sample_rate=sample_rate,
        drive_profile=REFERENCE_DRIVE_PROFILE,
        acoustic_profile=REFERENCE_ACOUSTIC_PROFILE,
    )
    total_frames = int(startup_only_duration(REFERENCE_DRIVE_PROFILE) * sample_rate)
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
    return diagnostics.output


def startup_reference_distances(features: dict[str, Any] | None = None) -> dict[str, float]:
    summary = json.loads(REFERENCE_SUMMARY_PATH.read_text(encoding="utf-8"))
    if features is None:
        features = compute_audio_features(render_startup_only(), 22050, REFERENCE_BUCKET)
    median = summary["median_reference_curves"]
    curve_frames = len(median["envelope"])
    mel_frames = len(median["log_mel"][0])
    return {
        "log_mel_distance": float(np.mean(np.abs(_aligned_log_mel(features, mel_frames) - np.asarray(median["log_mel"], dtype=np.float64)))),
        "envelope_error": float(np.mean(np.abs(_aligned_curve(features, "envelope_curve", curve_frames) - np.asarray(median["envelope"], dtype=np.float64)))),
        "centroid_error": float(np.mean(np.abs(_aligned_curve(features, "centroid_curve", curve_frames) - np.asarray(median["centroid_hz"], dtype=np.float64)))),
        "low_ratio_error": float(np.mean(np.abs(_aligned_curve(features, "low_ratio_curve", curve_frames) - np.asarray(median["low_ratio"], dtype=np.float64)))),
        "fundamental_error": float(np.mean(np.abs(_aligned_curve(features, "fundamental_curve", curve_frames) - np.asarray(median["fundamental_hz"], dtype=np.float64)))),
    }


def startup_reference_acceptance(features: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = json.loads(REFERENCE_SUMMARY_PATH.read_text(encoding="utf-8"))
    if features is None:
        features = compute_audio_features(render_startup_only(), 22050, REFERENCE_BUCKET)
    return evaluate_startup_acceptance(features, summary, drive_bucket=REFERENCE_BUCKET)


def refresh_startup_reference_summary(features: dict[str, Any]) -> None:
    """Refresh the generated side of the committed report without requiring local source WAVs."""
    summary = json.loads(REFERENCE_SUMMARY_PATH.read_text(encoding="utf-8"))
    median = summary["median_reference_curves"]
    curve_frames = len(median["envelope"])
    mel_frames = len(median["log_mel"][0])
    summary["generated_drive_profile"] = REFERENCE_DRIVE_PROFILE
    summary["generated_acoustic_profile"] = REFERENCE_ACOUSTIC_PROFILE
    summary["generated"] = {
        key: float(features[key])
        for key in (
            "first_audible_s",
            "time_to_90_s",
            "steady_fundamental_hz",
            "spectral_centroid_hz",
            "low_band_ratio",
            "bubbly_modulation_ratio",
            "transient_density",
        )
    }
    summary["distance"].update(startup_reference_distances(features))
    summary["generated_curves"] = {
        "envelope": _aligned_curve(features, "envelope_curve", curve_frames).tolist(),
        "centroid_hz": _aligned_curve(features, "centroid_curve", curve_frames).tolist(),
        "low_ratio": _aligned_curve(features, "low_ratio_curve", curve_frames).tolist(),
        "fundamental_hz": _aligned_curve(features, "fundamental_curve", curve_frames).tolist(),
        "log_mel": _aligned_log_mel(features, mel_frames).tolist(),
    }
    REFERENCE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_startup_summary_svg(REFERENCE_SUMMARY_SVG_PATH, summary)


def main() -> None:
    startup_features = compute_audio_features(render_startup_only(), 22050, REFERENCE_BUCKET)
    if "--refresh-summary" in sys.argv[1:]:
        refresh_startup_reference_summary(startup_features)
    acceptance = startup_reference_acceptance(startup_features)
    payload = {
        "metadata_storm": metadata_storm_metrics(),
        "startup_reference": startup_reference_distances(startup_features),
        "startup_acceptance": acceptance,
    }
    print(json.dumps(payload, indent=2))
    if not acceptance["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
