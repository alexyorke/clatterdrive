from __future__ import annotations

import argparse
import ast
import json
import math
import re
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from tools.reference_audio import compute_audio_features


ROOT = Path(__file__).resolve().parents[1]
LOCAL_REFS = ROOT / ".runtime" / "local_refs"
ACCURACY_DIR = ROOT / ".runtime" / "mh-accuracy"
HTML_PATH = ROOT / "docs" / "mh-thrash-lab.html"

SAMPLE_RATE = 24_000
TAU = math.pi * 2.0


@dataclass(frozen=True)
class ReferenceBundle:
    samples: np.ndarray
    sample_rate: int
    events: list[dict[str, float]]


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def extract_html_defaults(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const defaults = \{(?P<body>.*?)\n    \};", text, flags=re.S)
    if not match:
        raise ValueError(f"could not find defaults block in {path}")
    body = match.group("body")
    quoted = re.sub(r"(^|\s)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', body, flags=re.M)
    return {
        key: float(value)
        for key, value in ast.literal_eval("{" + quoted + "}").items()
    }


def load_reference_bundle() -> ReferenceBundle:
    samples, sample_rate = load_wav(LOCAL_REFS / "hdd-sample-best-window.wav")
    events = json.loads((LOCAL_REFS / "hdd-sample-best-window-events.json").read_text(encoding="utf-8"))
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz reference, got {sample_rate}")
    return ReferenceBundle(samples=samples, sample_rate=sample_rate, events=events)


def make_rng(seed: int = 0x4D48C0DE) -> Iterator[float]:
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 0x100000000


def percentile(values: np.ndarray, p: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, p))


def build_free_schedule(params: dict[str, float]) -> list[dict[str, Any]]:
    rand = make_rng()
    schedule: list[dict[str, Any]] = []
    t_s = 0.0
    duration = float(params["loopDurationS"])
    while t_s < duration - 0.15:
        packet_length = max(
            1,
            round(
                params["packetLengthMean"]
                + (next(rand) * 2.0 - 1.0) * params["packetLengthJitter"]
            ),
        )
        for packet_index in range(packet_length):
            if t_s >= duration - 0.15:
                break
            span_offset_ms = (next(rand) * 2.0 - 1.0) * params["seekSpanSpreadMs"]
            macro_amp = params["eventGain"] * max(
                0.10,
                1.0 + (next(rand) * 2.0 - 1.0) * params["ampJitter"],
            )
            shape = [
                {"ms": 0.0, "amp": params["launchAmp"]},
                {
                    "ms": max(0.02, params["brakeDelayMs"] + span_offset_ms),
                    "amp": params["brakeAmp"] * (0.94 + next(rand) * 0.14),
                },
            ]
            if params["settleAmp"] > 0.0:
                shape.append(
                    {
                        "ms": max(0.04, params["settleDelayMs"] + span_offset_ms),
                        "amp": params["settleAmp"] * (0.94 + next(rand) * 0.14),
                    }
                )
            schedule.append({"t": t_s, "amp": macro_amp, "shape": shape})
            if packet_index < packet_length - 1:
                t_s += params["denseGapMs"] * (0.72 + next(rand) * 0.70) / 1000.0
        if t_s >= duration - 0.15:
            break
        if next(rand) < params["resetProbability"]:
            t_s += params["resetGapMs"] * (0.70 + next(rand) * 0.85) / 1000.0
        else:
            t_s += params["midGapMs"] * (0.72 + next(rand) * 0.70) / 1000.0
    return schedule


def build_conditioned_schedule(
    params: dict[str, float],
    reference_events: list[dict[str, float]],
) -> list[dict[str, Any]]:
    amplitudes = np.asarray([event["a"] for event in reference_events], dtype=np.float64)
    amplitude_scale = max(float(np.mean(amplitudes)), 1e-9)
    schedule: list[dict[str, Any]] = []
    for event in reference_events:
        macro_amp = params["eventGain"] * (event["a"] / amplitude_scale)
        shape = [
            {"ms": 0.0, "amp": params["launchAmp"]},
            {"ms": max(0.02, params["brakeDelayMs"]), "amp": params["brakeAmp"]},
        ]
        if params["settleAmp"] > 0.0:
            shape.append({"ms": max(0.04, params["settleDelayMs"]), "amp": params["settleAmp"]})
        schedule.append({"t": float(event["t"]), "amp": macro_amp, "shape": shape})
    return schedule


def pulse_kernel(params: dict[str, float]) -> np.ndarray:
    pulse_width_samples = max(1, round((params["pulseWidthMs"] / 1000.0) * SAMPLE_RATE))
    kernel = np.asarray([params["pulseDecay"] ** index for index in range(pulse_width_samples)], dtype=np.float64)
    return kernel / max(float(np.sum(np.abs(kernel))), 1e-12)


def mode_responses(params: dict[str, float]) -> list[np.ndarray]:
    specs = [
        ("body1Hz", "body1Gain", "body1T60", params.get("body1Phase", 0.00)),
        ("body2Hz", "body2Gain", "body2T60", params.get("body2Phase", 0.08)),
        ("center1Hz", "center1Gain", "center1T60", params.get("center1Phase", 0.00)),
        ("center2Hz", "center2Gain", "center2T60", params.get("center2Phase", 0.10)),
        ("air1Hz", "air1Gain", "air1T60", params.get("air1Phase", 0.16)),
        ("air2Hz", "air2Gain", "air2T60", params.get("air2Phase", 0.22)),
    ]
    responses: list[np.ndarray] = []
    for hz_key, gain_key, t60_key, phase in specs:
        t60_s = max(params[t60_key] / 1000.0, 1e-4)
        count = max(1, int(SAMPLE_RATE * t60_s * 3.8))
        t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        env = np.exp(-6.91 * t / t60_s)
        response = params[gain_key] * env * np.sin(TAU * params[hz_key] * t + phase)
        responses.append(response)
    return responses


def render_schedule(
    params: dict[str, float],
    schedule: list[dict[str, Any]],
    duration_s: float,
) -> np.ndarray:
    frames = int(duration_s * SAMPLE_RATE)
    output = np.zeros(frames, dtype=np.float64)
    onset = int(0.08 * SAMPLE_RATE)
    kernel = pulse_kernel(params)
    responses = mode_responses(params)

    for macro_event in schedule:
        macro_index = onset + int(macro_event["t"] * SAMPLE_RATE)
        for event in macro_event["shape"]:
            index = macro_index + int((event["ms"] / 1000.0) * SAMPLE_RATE)
            if index >= frames:
                continue
            for tap, kernel_amp in enumerate(kernel):
                tap_index = index + tap
                if tap_index >= frames:
                    break
                tap_amp = macro_event["amp"] * event["amp"] * kernel_amp
                output[tap_index] += tap_amp * params["directMix"]
                for response in responses:
                    end = min(frames, tap_index + len(response))
                    output[tap_index:end] += tap_amp * response[: end - tap_index]

    peak = max(float(np.max(np.abs(output))), 1e-6)
    gain = min(0.34 / peak, 4.0)
    return np.tanh(output * gain)


def bandpass_clicks(samples: np.ndarray) -> np.ndarray:
    sos = signal.butter(4, [320.0, 5200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return signal.sosfiltfilt(sos, samples)


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / (left_norm * right_norm))


def align_by_cross_correlation(reference: np.ndarray, candidate: np.ndarray, max_lag_ms: float = 4.0) -> tuple[np.ndarray, int]:
    max_lag = int((max_lag_ms / 1000.0) * SAMPLE_RATE)
    if reference.size == 0 or candidate.size == 0:
        return candidate, 0
    reference_centered = reference - float(np.mean(reference))
    candidate_centered = candidate - float(np.mean(candidate))
    correlation = signal.correlate(candidate_centered, reference_centered, mode="full", method="fft")
    lags = signal.correlation_lags(len(candidate_centered), len(reference_centered), mode="full")
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lag = int(lags[mask][int(np.argmax(correlation[mask]))]) if np.any(mask) else 0
    if lag > 0:
        aligned = np.pad(candidate, (0, lag))[: candidate.size]
        aligned = np.pad(aligned, (lag, 0))[: candidate.size]
    elif lag < 0:
        shift = abs(lag)
        aligned = np.pad(candidate[shift:], (0, shift))
    else:
        aligned = candidate.copy()
    return aligned, lag


def envelope_curve(samples: np.ndarray) -> np.ndarray:
    absolute = np.abs(samples)
    sos = signal.butter(2, 180.0, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    return signal.sosfiltfilt(sos, absolute)


def average_event_spectrum(samples: np.ndarray, event_times: list[float], strongest: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    fft_n = 4096
    freqs = np.fft.rfftfreq(fft_n, d=1.0 / SAMPLE_RATE)
    accumulator = np.zeros(fft_n // 2 + 1, dtype=np.float64)
    count = 0
    indices = strongest if strongest is not None else np.arange(len(event_times))
    for event_index in indices:
        center = round(event_times[int(event_index)] * SAMPLE_RATE)
        lo = max(0, center - int(0.008 * SAMPLE_RATE))
        hi = min(len(samples), center + int(0.018 * SAMPLE_RATE))
        segment = samples[lo:hi]
        if len(segment) < 256:
            continue
        windowed = segment * np.hanning(len(segment))
        accumulator += np.abs(np.fft.rfft(windowed, n=fft_n))
        count += 1
    if count == 0:
        return freqs, accumulator
    accumulator /= count
    accumulator /= max(float(np.max(accumulator)), 1e-12)
    return freqs, accumulator


def band_mean(freqs: np.ndarray, spectrum: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.mean(spectrum[mask]))


def spectral_centroid(freqs: np.ndarray, spectrum: np.ndarray) -> float:
    mask = (freqs >= 320.0) & (freqs <= 5200.0)
    band_freqs = freqs[mask]
    band_spectrum = spectrum[mask]
    denominator = max(float(np.sum(band_spectrum)), 1e-12)
    return float(np.sum(band_freqs * band_spectrum) / denominator)


def strongest_event_indices(reference_events: list[dict[str, float]], count: int = 28) -> np.ndarray:
    amplitudes = np.asarray([event["a"] for event in reference_events], dtype=np.float64)
    selected = np.argpartition(amplitudes, -min(count, len(amplitudes)))[-min(count, len(amplitudes)) :]
    return selected[np.argsort(amplitudes[selected])[::-1]]


def transient_window_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
    event_times: list[float],
    strongest: np.ndarray,
) -> float:
    correlations: list[float] = []
    for event_index in strongest:
        center = round(event_times[int(event_index)] * SAMPLE_RATE)
        lo = max(0, center - int(0.004 * SAMPLE_RATE))
        hi = min(len(reference), center + int(0.020 * SAMPLE_RATE))
        ref_window = reference[lo:hi]
        candidate_window = candidate[lo:hi]
        if len(ref_window) < 64 or len(ref_window) != len(candidate_window):
            continue
        correlations.append(normalized_correlation(ref_window, candidate_window))
    return float(np.mean(correlations)) if correlations else 0.0


def compare_feature_pair(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    curve_frames = 160
    mel_frames = 120

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

    ref_onset = int(reference["onset_index"])
    cand_onset = int(candidate["onset_index"])

    ref_env = _resample_series(np.asarray(reference["envelope_curve"], dtype=np.float64)[ref_onset:], curve_frames)
    cand_env = _resample_series(np.asarray(candidate["envelope_curve"], dtype=np.float64)[cand_onset:], curve_frames)
    ref_env /= max(float(np.max(ref_env)), 1e-12)
    cand_env /= max(float(np.max(cand_env)), 1e-12)

    ref_mel = _resample_matrix(np.asarray(reference["log_mel"], dtype=np.float64)[:, ref_onset:], mel_frames)
    cand_mel = _resample_matrix(np.asarray(candidate["log_mel"], dtype=np.float64)[:, cand_onset:], mel_frames)

    return {
        "envelope_error": float(np.mean(np.abs(ref_env - cand_env))),
        "envelope_correlation": normalized_correlation(ref_env, cand_env),
        "log_mel_distance": float(np.mean(np.abs(ref_mel - cand_mel))) if ref_mel.size and cand_mel.size else 0.0,
        "centroid_error": float(abs(reference["spectral_centroid_hz"] - candidate["spectral_centroid_hz"])),
        "low_ratio_error": float(abs(reference["low_band_ratio"] - candidate["low_band_ratio"])),
    }


def scheduler_stats(schedule: list[dict[str, Any]]) -> dict[str, float]:
    if not schedule:
        return {
            "count": 0.0,
            "mean_gap_ms": 0.0,
            "p90_gap_ms": 0.0,
            "max_gap_ms": 0.0,
            "le_20ms": 0.0,
            "gt_120ms": 0.0,
            "amplitude_cv": 0.0,
        }
    times = np.asarray([event["t"] for event in schedule], dtype=np.float64)
    amps = np.asarray([event["amp"] for event in schedule], dtype=np.float64)
    gaps = np.diff(times)
    return {
        "count": float(len(schedule)),
        "mean_gap_ms": float(np.mean(gaps) * 1000.0) if gaps.size else 0.0,
        "p90_gap_ms": percentile(gaps, 90.0) * 1000.0 if gaps.size else 0.0,
        "max_gap_ms": float(np.max(gaps) * 1000.0) if gaps.size else 0.0,
        "le_20ms": float(np.sum(gaps <= 0.020)) if gaps.size else 0.0,
        "gt_120ms": float(np.sum(gaps > 0.120)) if gaps.size else 0.0,
        "amplitude_cv": float(np.std(amps) / max(float(np.mean(amps)), 1e-9)),
    }


def reference_scheduler_stats(reference_events: list[dict[str, float]]) -> dict[str, float]:
    schedule = [
        {
            "t": float(event["t"]),
            "amp": float(event["a"]),
            "shape": [],
        }
        for event in reference_events
    ]
    return scheduler_stats(schedule)


def scheduler_score(candidate: dict[str, float], target: dict[str, float]) -> float:
    error = 0.0
    error += ((candidate["count"] - target["count"]) / 16.0) ** 2
    error += ((candidate["mean_gap_ms"] - target["mean_gap_ms"]) / 8.0) ** 2
    error += ((candidate["p90_gap_ms"] - target["p90_gap_ms"]) / 16.0) ** 2
    error += ((candidate["max_gap_ms"] - target["max_gap_ms"]) / 30.0) ** 2
    error += ((candidate["le_20ms"] - target["le_20ms"]) / 18.0) ** 2
    error += ((candidate["gt_120ms"] - target["gt_120ms"]) / 4.0) ** 2
    error += ((candidate["amplitude_cv"] - target["amplitude_cv"]) / 0.20) ** 2
    return float(error)


def validate_params(params: dict[str, float], reference: ReferenceBundle) -> dict[str, Any]:
    event_times = [float(event["t"]) for event in reference.events]
    strongest = strongest_event_indices(reference.events)

    conditioned_schedule = build_conditioned_schedule(params, reference.events)
    conditioned_render = render_schedule(params, conditioned_schedule, len(reference.samples) / reference.sample_rate)

    ref_band = bandpass_clicks(reference.samples)
    conditioned_band = bandpass_clicks(conditioned_render)
    aligned_band, lag = align_by_cross_correlation(ref_band, conditioned_band)

    ref_peak = max(float(np.max(np.abs(ref_band))), 1e-9)
    aligned_peak = max(float(np.max(np.abs(aligned_band))), 1e-9)
    ref_norm = ref_band / ref_peak
    aligned_norm = aligned_band / aligned_peak

    ref_env = envelope_curve(ref_norm)
    synth_env = envelope_curve(aligned_norm)
    env_len = min(len(ref_env), len(synth_env))
    env_error = float(np.mean(np.abs(ref_env[:env_len] - synth_env[:env_len]))) if env_len else 0.0
    waveform_corr = normalized_correlation(ref_norm, aligned_norm)
    transient_corr = transient_window_correlation(ref_norm, aligned_norm, event_times, strongest)

    ref_freqs, ref_spectrum = average_event_spectrum(ref_norm, event_times, strongest)
    synth_freqs, synth_spectrum = average_event_spectrum(aligned_norm, event_times, strongest)
    body_ref = band_mean(ref_freqs, ref_spectrum, 420.0, 900.0)
    center_ref = band_mean(ref_freqs, ref_spectrum, 1000.0, 1800.0)
    air_ref = band_mean(ref_freqs, ref_spectrum, 2400.0, 4800.0)
    body_synth = band_mean(synth_freqs, synth_spectrum, 420.0, 900.0)
    center_synth = band_mean(synth_freqs, synth_spectrum, 1000.0, 1800.0)
    air_synth = band_mean(synth_freqs, synth_spectrum, 2400.0, 4800.0)

    conditioned_features = compute_audio_features(conditioned_render, reference.sample_rate, "desktop_7200_internal")
    reference_features = compute_audio_features(reference.samples, reference.sample_rate, "desktop_7200_internal")
    feature_pair = compare_feature_pair(reference_features, conditioned_features)

    free_schedule = build_free_schedule(params)
    free_render = render_schedule(params, free_schedule, len(reference.samples) / reference.sample_rate)
    free_band = bandpass_clicks(free_render)
    aligned_free, free_lag = align_by_cross_correlation(ref_band, free_band)
    free_corr = normalized_correlation(ref_norm, aligned_free / max(float(np.max(np.abs(aligned_free))), 1e-9))

    target_schedule_stats = reference_scheduler_stats(reference.events)
    candidate_schedule_stats = scheduler_stats(free_schedule)

    conditioned_score = (
        3.2 * (1.0 - waveform_corr)
        + 3.4 * (1.0 - transient_corr)
        + 1.8 * feature_pair["envelope_error"]
        + 0.10 * feature_pair["log_mel_distance"]
        + abs(body_synth - body_ref) / max(body_ref, 1e-9) * 1.6
        + abs(center_synth - center_ref) / max(center_ref, 1e-9) * 1.8
        + abs(air_synth - air_ref) / max(air_ref, 1e-9) * 1.2
        + abs(spectral_centroid(synth_freqs, synth_spectrum) - spectral_centroid(ref_freqs, ref_spectrum)) / 250.0
    )
    schedule_error = scheduler_score(candidate_schedule_stats, target_schedule_stats)
    composite = conditioned_score + 0.30 * schedule_error

    return {
        "composite_score": composite,
        "conditioned_render": conditioned_render,
        "free_render": free_render,
        "conditioned_metrics": {
            "waveform_correlation": waveform_corr,
            "transient_window_correlation": transient_corr,
            "alignment_lag_ms": lag * 1000.0 / SAMPLE_RATE,
            "envelope_error": env_error,
            "body_reference": body_ref,
            "body_synth": body_synth,
            "center_reference": center_ref,
            "center_synth": center_synth,
            "air_reference": air_ref,
            "air_synth": air_synth,
            "spectral_centroid_reference_hz": spectral_centroid(ref_freqs, ref_spectrum),
            "spectral_centroid_synth_hz": spectral_centroid(synth_freqs, synth_spectrum),
            **feature_pair,
        },
        "free_running_metrics": {
            "waveform_correlation": free_corr,
            "alignment_lag_ms": free_lag * 1000.0 / SAMPLE_RATE,
            "schedule": candidate_schedule_stats,
        },
        "reference_schedule": target_schedule_stats,
    }


def optimize_scheduler(params: dict[str, float], reference_events: list[dict[str, float]]) -> dict[str, float]:
    best = dict(params)
    target = reference_scheduler_stats(reference_events)
    best_score = scheduler_score(scheduler_stats(build_free_schedule(best)), target)
    keys = [
        "denseGapMs",
        "midGapMs",
        "resetGapMs",
        "packetLengthMean",
        "packetLengthJitter",
        "resetProbability",
        "ampJitter",
    ]
    step_sizes = {
        "denseGapMs": 3.0,
        "midGapMs": 12.0,
        "resetGapMs": 18.0,
        "packetLengthMean": 1.0,
        "packetLengthJitter": 0.6,
        "resetProbability": 0.06,
        "ampJitter": 0.06,
    }
    limits = {
        "denseGapMs": (8.0, 28.0),
        "midGapMs": (40.0, 150.0),
        "resetGapMs": (110.0, 260.0),
        "packetLengthMean": (2.0, 10.0),
        "packetLengthJitter": (0.0, 4.5),
        "resetProbability": (0.02, 0.70),
        "ampJitter": (0.0, 0.70),
    }

    for shrink in (1.0, 0.65, 0.42, 0.26, 0.16):
        improved = True
        while improved:
            improved = False
            for key in keys:
                step = step_sizes[key] * shrink
                for direction in (-1.0, 1.0):
                    candidate = dict(best)
                    lo, hi = limits[key]
                    candidate[key] = float(min(max(candidate[key] + direction * step, lo), hi))
                    score = scheduler_score(scheduler_stats(build_free_schedule(candidate)), target)
                    if score + 1e-9 < best_score:
                        best = candidate
                        best_score = score
                        improved = True
    return best


def optimize_model(params: dict[str, float], reference: ReferenceBundle) -> dict[str, float]:
    best = {
        "body1Phase": 0.00,
        "body2Phase": 0.08,
        "center1Phase": 0.00,
        "center2Phase": 0.10,
        "air1Phase": 0.16,
        "air2Phase": 0.22,
        **params,
    }
    validation = validate_params(best, reference)
    best_score = float(validation["composite_score"])
    keys = [
        "directMix",
        "pulseWidthMs",
        "pulseDecay",
        "launchAmp",
        "brakeDelayMs",
        "brakeAmp",
        "eventGain",
        "body1Hz",
        "body1Gain",
        "body1T60",
        "body2Hz",
        "body2Gain",
        "center1Hz",
        "center1Gain",
        "center2Hz",
        "center2Gain",
        "air1Hz",
        "air1Gain",
        "air2Hz",
        "air2Gain",
        "body1Phase",
        "body2Phase",
        "center1Phase",
        "center2Phase",
        "air1Phase",
        "air2Phase",
    ]
    step_sizes = {
        "directMix": 0.010,
        "pulseWidthMs": 0.025,
        "pulseDecay": 0.040,
        "launchAmp": 0.050,
        "brakeDelayMs": 0.025,
        "brakeAmp": 0.050,
        "eventGain": 0.060,
        "body1Hz": 18.0,
        "body1Gain": 0.040,
        "body1T60": 3.0,
        "body2Hz": 24.0,
        "body2Gain": 0.030,
        "center1Hz": 30.0,
        "center1Gain": 0.020,
        "center2Hz": 40.0,
        "center2Gain": 0.020,
        "air1Hz": 90.0,
        "air1Gain": 0.010,
        "air2Hz": 120.0,
        "air2Gain": 0.010,
        "body1Phase": 0.12,
        "body2Phase": 0.12,
        "center1Phase": 0.14,
        "center2Phase": 0.14,
        "air1Phase": 0.12,
        "air2Phase": 0.12,
    }
    limits = {
        "directMix": (0.0, 0.05),
        "pulseWidthMs": (0.04, 0.25),
        "pulseDecay": (0.55, 0.98),
        "launchAmp": (0.30, 1.10),
        "brakeDelayMs": (0.10, 0.50),
        "brakeAmp": (-0.80, -0.05),
        "eventGain": (0.35, 1.10),
        "body1Hz": (400.0, 620.0),
        "body1Gain": (0.05, 0.60),
        "body1T60": (12.0, 48.0),
        "body2Hz": (480.0, 920.0),
        "body2Gain": (0.0, 0.40),
        "center1Hz": (900.0, 1450.0),
        "center1Gain": (0.0, 0.20),
        "center2Hz": (1100.0, 1900.0),
        "center2Gain": (0.0, 0.20),
        "air1Hz": (2200.0, 3800.0),
        "air1Gain": (0.0, 0.08),
        "air2Hz": (3200.0, 5200.0),
        "air2Gain": (0.0, 0.08),
        "body1Phase": (-math.pi, math.pi),
        "body2Phase": (-math.pi, math.pi),
        "center1Phase": (-math.pi, math.pi),
        "center2Phase": (-math.pi, math.pi),
        "air1Phase": (-math.pi, math.pi),
        "air2Phase": (-math.pi, math.pi),
    }

    for shrink in (1.0, 0.60, 0.36, 0.22, 0.14):
        improved = True
        while improved:
            improved = False
            for key in keys:
                step = step_sizes[key] * shrink
                for direction in (-1.0, 1.0):
                    candidate = dict(best)
                    lo, hi = limits[key]
                    candidate[key] = float(min(max(candidate[key] + direction * step, lo), hi))
                    validation = validate_params(candidate, reference)
                    score = float(validation["composite_score"])
                    if score + 1e-9 < best_score:
                        best = candidate
                        best_score = score
                        improved = True
    return best


def rounded_params(params: dict[str, float]) -> dict[str, float]:
    rounded: dict[str, float] = {}
    for key, value in params.items():
        if key.endswith("Hz"):
            rounded[key] = round(float(value), 0)
        elif key.endswith("Ms") or key.endswith("S"):
            rounded[key] = round(float(value), 3)
        elif key.endswith("T60"):
            rounded[key] = round(float(value), 1)
        elif key.endswith("Phase") or key in {"pulseDecay", "resetProbability", "ampJitter", "eventGain", "directMix"}:
            rounded[key] = round(float(value), 3)
        else:
            rounded[key] = round(float(value), 3)
    return rounded


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and validate the MH thrash lab against the local HDD reference.")
    parser.add_argument("--report", type=Path, default=ACCURACY_DIR / "hdd-sample-validation.json")
    parser.add_argument("--conditioned-wav", type=Path, default=ACCURACY_DIR / "conditioned-fit.wav")
    parser.add_argument("--free-wav", type=Path, default=ACCURACY_DIR / "free-running-fit.wav")
    parser.add_argument("--recommended-json", type=Path, default=ACCURACY_DIR / "recommended-defaults.json")
    args = parser.parse_args()

    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_reference_bundle()
    starting_params = extract_html_defaults(HTML_PATH)

    baseline = validate_params(starting_params, reference)
    fitted = optimize_scheduler(starting_params, reference.events)
    fitted = optimize_model(fitted, reference)
    final_validation = validate_params(fitted, reference)

    write_wav(args.conditioned_wav, final_validation["conditioned_render"], SAMPLE_RATE)
    write_wav(args.free_wav, final_validation["free_render"], SAMPLE_RATE)
    args.recommended_json.write_text(json.dumps(rounded_params(fitted), indent=2), encoding="utf-8")

    report = {
        "reference": {
            "best_window_wav": str((LOCAL_REFS / "hdd-sample-best-window.wav").relative_to(ROOT)),
            "events_json": str((LOCAL_REFS / "hdd-sample-best-window-events.json").relative_to(ROOT)),
            "event_count": len(reference.events),
        },
        "baseline": {
            "params": rounded_params(starting_params),
            "metrics": {
                key: value
                for key, value in baseline.items()
                if key not in {"conditioned_render", "free_render"}
            },
        },
        "optimized": {
            "params": rounded_params(fitted),
            "metrics": {
                key: value
                for key, value in final_validation.items()
                if key not in {"conditioned_render", "free_render"}
            },
        },
        "improvement": {
            "composite_score_delta": float(baseline["composite_score"] - final_validation["composite_score"]),
            "conditioned_waveform_correlation_delta": float(
                final_validation["conditioned_metrics"]["waveform_correlation"]
                - baseline["conditioned_metrics"]["waveform_correlation"]
            ),
            "transient_window_correlation_delta": float(
                final_validation["conditioned_metrics"]["transient_window_correlation"]
                - baseline["conditioned_metrics"]["transient_window_correlation"]
            ),
            "log_mel_distance_delta": float(
                baseline["conditioned_metrics"]["log_mel_distance"]
                - final_validation["conditioned_metrics"]["log_mel_distance"]
            ),
        },
        "artifacts": {
            "conditioned_wav": str(args.conditioned_wav.relative_to(ROOT)),
            "free_running_wav": str(args.free_wav.relative_to(ROOT)),
            "recommended_defaults_json": str(args.recommended_json.relative_to(ROOT)),
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["improvement"], indent=2))
    print(f"wrote {args.report.relative_to(ROOT)}")
    print(f"wrote {args.conditioned_wav.relative_to(ROOT)}")
    print(f"wrote {args.free_wav.relative_to(ROOT)}")
    print(f"wrote {args.recommended_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
