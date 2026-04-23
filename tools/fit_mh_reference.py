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
class ReferenceEventFeature:
    amplitude: float
    peak_lag_ms: float
    late_ratio: float
    tail_ratio: float
    max_gap_ms: float
    air_ratio: float
    second_bump_lag_ms: float
    second_bump_prom: float
    decay_slope_db_per_ms: float


@dataclass(frozen=True)
class FamilyPrototype:
    base_family: str
    launch_mul: float = 1.0
    brake_delay_add_ms: float = 0.0
    brake_amp_mul: float = 1.0
    shell_enable: bool = False
    shell_delay_add_ms: float = 0.0
    shell_amp: float = 0.0
    settle_enable: bool = False
    settle_delay_ms: float = 0.0
    settle_amp: float = 0.0
    modal_damp_scale: float = 1.0
    air_tilt_db: float = 0.0


@dataclass(frozen=True)
class ReferenceBundle:
    samples: np.ndarray
    sample_rate: int
    events: list[dict[str, float]]
    features: list[ReferenceEventFeature]
    families: list[str]


PHASE_DEFAULTS = {
    "body1Phase": 0.00,
    "body2Phase": 0.08,
    "center1Phase": 0.00,
    "center2Phase": 0.10,
    "air1Phase": 0.16,
    "air2Phase": 0.22,
}

PROTOTYPE_PARAM_DEFAULTS = {
    "ringDelayedLaunchMul": 0.90,
    "ringDelayedBrakeDelayAddMs": 3.0,
    "ringDelayedBrakeAmpMul": 1.10,
    "ringDelayedShellDelayAddMs": 4.0,
    "ringDelayedShellAmp": 0.18,
    "ringDelayedSettleDelayMs": 22.0,
    "ringDelayedSettleAmp": 0.05,
    "ringDelayedModalDampScale": 1.00,
    "ringDelayedAirTiltDb": 1.0,
    "ringResonantLaunchMul": 0.85,
    "ringResonantBrakeDelayAddMs": 1.0,
    "ringResonantBrakeAmpMul": 0.85,
    "ringResonantShellDelayMs": 6.0,
    "ringResonantShellAmp": 0.12,
    "ringResonantSettleDelayMs": 18.0,
    "ringResonantSettleAmp": 0.16,
    "ringResonantModalDampScale": 0.82,
    "ringResonantAirTiltDb": 2.5,
}


def with_model_defaults(params: dict[str, float]) -> dict[str, float]:
    return {
        **PHASE_DEFAULTS,
        **PROTOTYPE_PARAM_DEFAULTS,
        **params,
    }


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


def family_prototypes(params: dict[str, float]) -> dict[str, FamilyPrototype]:
    params = with_model_defaults(params)
    return {
        "tight": FamilyPrototype(base_family="tight"),
        "ring_delayed": FamilyPrototype(
            base_family="ringy",
            launch_mul=params["ringDelayedLaunchMul"],
            brake_delay_add_ms=params["ringDelayedBrakeDelayAddMs"],
            brake_amp_mul=params["ringDelayedBrakeAmpMul"],
            shell_enable=True,
            shell_delay_add_ms=params["ringDelayedShellDelayAddMs"],
            shell_amp=params["ringDelayedShellAmp"],
            settle_enable=True,
            settle_delay_ms=params["ringDelayedSettleDelayMs"],
            settle_amp=params["ringDelayedSettleAmp"],
            modal_damp_scale=params["ringDelayedModalDampScale"],
            air_tilt_db=params["ringDelayedAirTiltDb"],
        ),
        "ring_resonant": FamilyPrototype(
            base_family="tight",
            launch_mul=params["ringResonantLaunchMul"],
            brake_delay_add_ms=params["ringResonantBrakeDelayAddMs"],
            brake_amp_mul=params["ringResonantBrakeAmpMul"],
            shell_enable=True,
            shell_delay_add_ms=params["ringResonantShellDelayMs"],
            shell_amp=params["ringResonantShellAmp"],
            settle_enable=True,
            settle_delay_ms=params["ringResonantSettleDelayMs"],
            settle_amp=params["ringResonantSettleAmp"],
            modal_damp_scale=params["ringResonantModalDampScale"],
            air_tilt_db=params["ringResonantAirTiltDb"],
        ),
    }


def load_reference_bundle() -> ReferenceBundle:
    samples, sample_rate = load_wav(LOCAL_REFS / "hdd-sample-best-window.wav")
    events = json.loads((LOCAL_REFS / "hdd-sample-best-window-events.json").read_text(encoding="utf-8"))
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz reference, got {sample_rate}")
    features = extract_reference_event_features(samples, events)
    return ReferenceBundle(
        samples=samples,
        sample_rate=sample_rate,
        events=events,
        features=features,
        families=classify_reference_families(features),
    )


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
    previous_gap_ms = float(params["resetGapMs"])
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
            packet_span = max(params["packetLengthMean"] + params["packetLengthJitter"], 1.0)
            packet_norm = min(max((packet_length - 1.0) / packet_span, 0.0), 1.0)
            packet_gain = 1.0 + params.get("packetAccent", 0.0) * (0.85 - 1.10 * packet_norm)
            if packet_length <= 2:
                packet_gain *= 1.0 + 0.15 * params.get("packetAccent", 0.0)
            macro_amp = params["eventGain"] * packet_gain * max(
                0.10,
                1.0 + (next(rand) * 2.0 - 1.0) * params["ampJitter"],
            )
            family = choose_free_family(packet_index, packet_length, previous_gap_ms, macro_amp, params, rand)
            shape = make_event_shape(
                params,
                family,
                span_offset_ms=span_offset_ms,
                rand=rand,
            )
            schedule.append({"t": t_s, "amp": macro_amp, "family": family, "shape": shape})
            if packet_index < packet_length - 1:
                previous_gap_ms = params["denseGapMs"] * (0.72 + next(rand) * 0.70)
                t_s += previous_gap_ms / 1000.0
        if t_s >= duration - 0.15:
            break
        if next(rand) < params["resetProbability"]:
            previous_gap_ms = params["resetGapMs"] * (0.70 + next(rand) * 0.85)
        else:
            previous_gap_ms = params["midGapMs"] * (0.72 + next(rand) * 0.70)
        t_s += previous_gap_ms / 1000.0
    return schedule


def build_conditioned_schedule(
    params: dict[str, float],
    reference_events: list[dict[str, float]],
    families: list[str],
) -> list[dict[str, Any]]:
    prototypes = family_prototypes(params)
    amplitudes = np.asarray([event["a"] for event in reference_events], dtype=np.float64)
    amplitude_scale = max(float(np.mean(amplitudes)), 1e-9)
    schedule: list[dict[str, Any]] = []
    for index, event in enumerate(reference_events):
        macro_amp = params["eventGain"] * (event["a"] / amplitude_scale)
        family = families[index] if index < len(families) else "tight"
        prototype = prototypes.get(family, prototypes["tight"])
        shape = make_event_shape(params, family, prototypes=prototypes)
        schedule.append(
            {
                "t": float(event["t"]),
                "amp": macro_amp,
                "family": family,
                "shape": shape,
                "modal_damp_scale": prototype.modal_damp_scale,
                "air_tilt_db": prototype.air_tilt_db,
            }
        )
    return schedule


def pulse_kernel(params: dict[str, float]) -> np.ndarray:
    pulse_width_samples = max(1, round((params["pulseWidthMs"] / 1000.0) * SAMPLE_RATE))
    pivot = max(1, int(round(pulse_width_samples * 0.38)))
    kernel = np.zeros(pulse_width_samples, dtype=np.float64)
    for index in range(pulse_width_samples):
        if index <= pivot:
            kernel[index] = params["pulseDecay"] ** index
        else:
            kernel[index] = -0.58 * (params["pulseDecay"] ** (index - pivot))
    return kernel / max(float(np.sum(np.abs(kernel))), 1e-12)


def make_event_shape(
    params: dict[str, float],
    family: str,
    span_offset_ms: float = 0.0,
    rand: Iterator[float] | None = None,
    prototypes: dict[str, FamilyPrototype] | None = None,
) -> list[dict[str, float | str]]:
    def jitter() -> float:
        if rand is None:
            return 1.0
        return 0.94 + next(rand) * 0.14

    if family == "ringy":
        shape: list[dict[str, float | str]] = [
            {
                "stage": "launch",
                "ms": 0.0,
                "amp": params["launchAmp"] * params["ringyLaunchScale"],
            },
            {
                "stage": "brake",
                "ms": max(0.04, params["ringyBrakeDelayMs"] + span_offset_ms * 0.45),
                "amp": params["ringyBrakeAmp"] * jitter(),
            },
            {
                "stage": "shell",
                "ms": max(1.0, params["ringyShellDelayMs"] + span_offset_ms * 2.4),
                "amp": params["ringyShellAmp"] * jitter(),
            },
        ]
        if params["settleAmp"] > 0.0:
            shape.append(
                {
                    "stage": "settle",
                    "ms": max(0.10, params["settleDelayMs"] + span_offset_ms * 0.45),
                    "amp": (params["settleAmp"] * 0.72) * jitter(),
                }
            )
        return shape

    if family in {"ring_delayed", "ring_resonant"}:
        proto = (prototypes or family_prototypes(params))[family]
        if proto.base_family == "ringy":
            shape = [
                {
                    "stage": "launch",
                    "ms": 0.0,
                    "amp": params["launchAmp"] * params["ringyLaunchScale"] * proto.launch_mul,
                },
                {
                    "stage": "brake",
                    "ms": max(0.04, params["ringyBrakeDelayMs"] + proto.brake_delay_add_ms + span_offset_ms * 0.45),
                    "amp": params["ringyBrakeAmp"] * proto.brake_amp_mul * jitter(),
                },
            ]
            if proto.shell_enable:
                shape.append(
                    {
                        "stage": "shell",
                        "ms": max(1.0, params["ringyShellDelayMs"] + proto.shell_delay_add_ms + span_offset_ms * 2.4),
                        "amp": proto.shell_amp * jitter(),
                    }
                )
            if proto.settle_enable and proto.settle_amp > 0.0:
                shape.append(
                    {
                        "stage": "settle",
                        "ms": max(0.10, proto.settle_delay_ms + span_offset_ms * 0.45),
                        "amp": proto.settle_amp * jitter(),
                    }
                )
            return shape

        shape = [
            {"stage": "launch", "ms": 0.0, "amp": params["launchAmp"] * proto.launch_mul},
            {
                "stage": "brake",
                "ms": max(0.02, params["brakeDelayMs"] + proto.brake_delay_add_ms + span_offset_ms),
                "amp": params["brakeAmp"] * proto.brake_amp_mul * jitter(),
            },
        ]
        if proto.shell_enable and proto.shell_amp > 0.0:
            shape.append(
                {
                    "stage": "shell",
                    "ms": max(1.0, proto.shell_delay_add_ms + span_offset_ms * 0.45),
                    "amp": proto.shell_amp * jitter(),
                }
            )
        if proto.settle_enable and proto.settle_amp > 0.0:
            shape.append(
                {
                    "stage": "settle",
                    "ms": max(0.04, proto.settle_delay_ms + span_offset_ms * 0.60),
                    "amp": proto.settle_amp * jitter(),
                }
            )
        return shape

    shape = [
        {"stage": "launch", "ms": 0.0, "amp": params["launchAmp"]},
        {
            "stage": "brake",
            "ms": max(0.02, params["brakeDelayMs"] + span_offset_ms),
            "amp": params["brakeAmp"] * jitter(),
        },
    ]
    if params["settleAmp"] > 0.0:
        shape.append(
            {
                "stage": "settle",
                "ms": max(0.04, params["settleDelayMs"] + span_offset_ms),
                "amp": params["settleAmp"] * jitter(),
            }
        )
    return shape


def mode_responses(
    params: dict[str, float],
    modal_damp_scale: float = 1.0,
    air_tilt_db: float = 0.0,
) -> list[dict[str, Any]]:
    specs = [
        ("body", "body1Hz", "body1Gain", "body1T60", params.get("body1Phase", 0.00)),
        ("body", "body2Hz", "body2Gain", "body2T60", params.get("body2Phase", 0.08)),
        ("center", "center1Hz", "center1Gain", "center1T60", params.get("center1Phase", 0.00)),
        ("center", "center2Hz", "center2Gain", "center2T60", params.get("center2Phase", 0.10)),
        ("air", "air1Hz", "air1Gain", "air1T60", params.get("air1Phase", 0.16)),
        ("air", "air2Hz", "air2Gain", "air2T60", params.get("air2Phase", 0.22)),
    ]
    responses: list[dict[str, Any]] = []
    for band, hz_key, gain_key, t60_key, phase in specs:
        t60_s = max((params[t60_key] / 1000.0) / max(modal_damp_scale, 1e-3), 1e-4)
        count = max(1, int(SAMPLE_RATE * t60_s * 3.8))
        t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        env = np.exp(-6.91 * t / t60_s)
        gain = params[gain_key]
        if band == "air" and abs(air_tilt_db) > 1e-9:
            gain *= 10.0 ** (air_tilt_db / 20.0)
        response = gain * env * np.sin(TAU * params[hz_key] * t + phase)
        responses.append({"band": band, "response": response})
    return responses


def choose_free_family(
    packet_index: int,
    packet_length: int,
    previous_gap_ms: float,
    macro_amp: float,
    params: dict[str, float],
    rand: Iterator[float],
) -> str:
    if packet_index == 0 and previous_gap_ms >= max(58.0, params["midGapMs"] * 0.72):
        return "ringy"
    if packet_index == 0 and packet_length <= 2:
        return "ringy"
    if packet_index == 0 and previous_gap_ms >= 36.0 and next(rand) < 0.55:
        return "ringy"
    if packet_length <= 2 and macro_amp < params["eventGain"] * 0.95 and next(rand) < 0.28:
        return "ringy"
    return "tight"


def stage_band_scale(stage: str, band: str, family: str = "tight") -> float:
    if family in {"ringy", "ring_delayed"}:
        if stage == "launch":
            return {
                "body": 0.82,
                "center": 0.66,
                "air": 0.24,
            }[band]
        if stage == "brake":
            return {
                "body": 0.28,
                "center": 1.06,
                "air": 1.22,
            }[band]
        if stage == "shell":
            return {
                "body": 1.08,
                "center": 0.74,
                "air": 0.20,
            }[band]
        return {
            "body": 0.52,
            "center": 0.38,
            "air": 0.16,
        }[band]
    if stage == "launch":
        return {
            "body": 1.00,
            "center": 0.72,
            "air": 0.32,
        }[band]
    if stage == "brake":
        return {
            "body": 0.34,
            "center": 1.00,
            "air": 1.18,
        }[band]
    return {
        "body": 0.68,
        "center": 0.42,
        "air": 0.20,
    }[band]


def stage_direct_scale(stage: str, family: str = "tight") -> float:
    if family in {"ringy", "ring_delayed"}:
        if stage == "launch":
            return 0.75
        if stage == "brake":
            return 0.42
        if stage == "shell":
            return 0.0
        return 0.12
    if stage == "launch":
        return 1.0
    if stage == "brake":
        return 0.62
    return 0.18


def render_schedule(
    params: dict[str, float],
    schedule: list[dict[str, Any]],
    duration_s: float,
) -> np.ndarray:
    frames = int(duration_s * SAMPLE_RATE)
    output = np.zeros(frames, dtype=np.float64)
    onset = int(0.08 * SAMPLE_RATE)
    kernel = pulse_kernel(params)
    response_cache: dict[tuple[float, float], list[dict[str, Any]]] = {}

    for macro_event in schedule:
        macro_index = onset + int(macro_event["t"] * SAMPLE_RATE)
        damp_scale = float(macro_event.get("modal_damp_scale", 1.0))
        air_tilt_db = float(macro_event.get("air_tilt_db", 0.0))
        cache_key = (round(damp_scale, 6), round(air_tilt_db, 6))
        responses = response_cache.setdefault(
            cache_key,
            mode_responses(params, modal_damp_scale=damp_scale, air_tilt_db=air_tilt_db),
        )
        for event in macro_event["shape"]:
            index = macro_index + int((event["ms"] / 1000.0) * SAMPLE_RATE)
            if index >= frames:
                continue
            direct_amp = (
                macro_event["amp"]
                * event["amp"]
                * params["directMix"]
                * stage_direct_scale(str(event["stage"]), str(macro_event.get("family", "tight")))
            )
            output[index] += direct_amp
            if index + 1 < frames:
                output[index + 1] -= direct_amp * 0.35
            for tap, kernel_amp in enumerate(kernel):
                tap_index = index + tap
                if tap_index >= frames:
                    break
                tap_amp = macro_event["amp"] * event["amp"] * kernel_amp
                for response_info in responses:
                    response = response_info["response"]
                    route = stage_band_scale(
                        str(event["stage"]),
                        str(response_info["band"]),
                        str(macro_event.get("family", "tight")),
                    )
                    end = min(frames, tap_index + len(response))
                    output[tap_index:end] += tap_amp * route * response[: end - tap_index]

    peak = max(float(np.max(np.abs(output))), 1e-6)
    gain = min(0.34 / peak, 4.0)
    return np.tanh(output * gain)


def bandpass_clicks(samples: np.ndarray) -> np.ndarray:
    sos = signal.butter(4, [320.0, 5200.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return signal.sosfiltfilt(sos, samples)


def extract_reference_event_features(
    samples: np.ndarray,
    reference_events: list[dict[str, float]],
) -> list[ReferenceEventFeature]:
    filtered = bandpass_clicks(samples)
    pre = int(0.004 * SAMPLE_RATE)
    early_hi = pre + int(0.0025 * SAMPLE_RATE)
    late_hi = pre + int(0.012 * SAMPLE_RATE)
    tail_hi = pre + int(0.022 * SAMPLE_RATE)
    features: list[ReferenceEventFeature] = []
    fft_n = 512
    freqs = np.fft.rfftfreq(fft_n, d=1.0 / SAMPLE_RATE)
    for index, event in enumerate(reference_events):
        center = round(float(event["t"]) * SAMPLE_RATE)
        lo = max(0, center - pre)
        hi = min(len(filtered), center + int(0.022 * SAMPLE_RATE))
        window = filtered[lo:hi]
        prev_gap_ms = 0.0 if index == 0 else (float(event["t"]) - float(reference_events[index - 1]["t"])) * 1000.0
        next_gap_ms = (
            0.0
            if index >= len(reference_events) - 1
            else (float(reference_events[index + 1]["t"]) - float(event["t"])) * 1000.0
        )
        max_gap_ms = max(prev_gap_ms, next_gap_ms)
        if len(window) < max(tail_hi, pre + 24):
            features.append(
                ReferenceEventFeature(
                    amplitude=float(event["a"]),
                    peak_lag_ms=0.0,
                    late_ratio=0.0,
                    tail_ratio=0.0,
                    max_gap_ms=max_gap_ms,
                    air_ratio=0.0,
                    second_bump_lag_ms=0.0,
                    second_bump_prom=0.0,
                    decay_slope_db_per_ms=-1.0,
                )
            )
            continue
        abs_window = np.abs(window)
        peak_idx = int(np.argmax(abs_window))
        peak_lag_ms = max(0.0, (peak_idx - pre) * 1000.0 / SAMPLE_RATE)
        early = window[pre:min(len(window), early_hi)]
        late = window[min(len(window), early_hi):min(len(window), late_hi)]
        tail = window[min(len(window), late_hi):min(len(window), tail_hi)]
        early_energy = float(np.sum(early * early))
        late_ratio = float(np.sum(late * late) / max(early_energy, 1e-9))
        tail_ratio = float(np.sum(tail * tail) / max(early_energy, 1e-9))
        smoothed_window = signal.savgol_filter(abs_window, 9, 2, mode="interp") if len(abs_window) >= 9 else abs_window
        search_lo = min(len(smoothed_window), max(peak_idx + int(0.0015 * SAMPLE_RATE), pre + int(0.006 * SAMPLE_RATE)))
        search_hi = min(len(abs_window), tail_hi)
        second_bump_lag_ms = 0.0
        second_bump_prom = 0.0
        if search_hi - search_lo >= 8:
            prominence_floor = max(float(np.max(smoothed_window)) * 0.12, 1e-6)
            peak_indices, peak_props = signal.find_peaks(
                smoothed_window[search_lo:search_hi],
                prominence=prominence_floor,
                distance=max(1, int(0.001 * SAMPLE_RATE)),
            )
            if len(peak_indices):
                prominences = peak_props.get("prominences", np.zeros(len(peak_indices), dtype=np.float64))
                best_peak = int(np.argmax(prominences))
                second_index = int(peak_indices[best_peak]) + search_lo
                second_bump_lag_ms = max(0.0, (second_index - pre) * 1000.0 / SAMPLE_RATE)
                second_bump_prom = float(prominences[best_peak] / max(float(np.max(smoothed_window)), 1e-9))

        decay_slice = abs_window[min(len(abs_window), early_hi):min(len(abs_window), tail_hi)]
        if len(decay_slice) >= 12:
            decay_db = 20.0 * np.log10(np.maximum(decay_slice, 1e-6))
            decay_time_ms = np.arange(len(decay_db), dtype=np.float64) * 1000.0 / SAMPLE_RATE
            decay_slope = float(np.polyfit(decay_time_ms, decay_db, 1)[0])
        else:
            decay_slope = -1.0

        windowed = window * np.hanning(len(window))
        spectrum = np.abs(np.fft.rfft(windowed, n=fft_n))
        body_band = band_mean(freqs, spectrum, 420.0, 900.0)
        center_band = band_mean(freqs, spectrum, 1000.0, 1800.0)
        air_band = band_mean(freqs, spectrum, 2400.0, 4800.0)
        air_ratio = float(air_band / max(body_band + center_band, 1e-9))

        features.append(
            ReferenceEventFeature(
                amplitude=float(event["a"]),
                peak_lag_ms=peak_lag_ms,
                late_ratio=late_ratio,
                tail_ratio=tail_ratio,
                max_gap_ms=max_gap_ms,
                air_ratio=air_ratio,
                second_bump_lag_ms=second_bump_lag_ms,
                second_bump_prom=second_bump_prom,
                decay_slope_db_per_ms=decay_slope,
            )
        )
    return features


def classify_reference_families(features: list[ReferenceEventFeature]) -> list[str]:
    if not features:
        return []
    amplitudes = np.asarray([feature.amplitude for feature in features], dtype=np.float64)
    decay_values = np.asarray([feature.decay_slope_db_per_ms for feature in features], dtype=np.float64)
    air_values = np.asarray([feature.air_ratio for feature in features], dtype=np.float64)
    amp_threshold = float(np.quantile(amplitudes, 0.875))
    decay_q70 = float(np.quantile(decay_values, 0.70))
    air_q70 = float(np.quantile(air_values, 0.70))
    families: list[str] = []
    for feature in features:
        amp_hint = feature.amplitude >= amp_threshold and feature.late_ratio >= 1.1
        gap_hint = feature.max_gap_ms >= 38.0 and feature.late_ratio >= 0.9
        coarse_ringy = (
            feature.peak_lag_ms >= 11.0
            or (feature.late_ratio >= 1.5 and feature.tail_ratio >= 2.1)
            or amp_hint
            or gap_hint
        )
        if not coarse_ringy:
            families.append("tight")
            continue
        delayed = (
            feature.peak_lag_ms >= 11.0
            or (feature.second_bump_lag_ms >= 7.5 and feature.second_bump_prom >= 0.18)
            or (feature.max_gap_ms >= 38.0 and feature.late_ratio >= 0.95)
        )
        resonant = (
            (
                (
                    feature.tail_ratio >= 1.9
                    and feature.late_ratio >= 1.25
                    and feature.decay_slope_db_per_ms >= decay_q70
                )
                or (feature.tail_ratio >= 2.1 and feature.air_ratio >= air_q70)
            )
        )
        if delayed and resonant:
            delay_score = (
                max((feature.peak_lag_ms - 11.0) / 4.0, 0.0)
                + 0.9 * float(feature.second_bump_lag_ms >= 7.5 and feature.second_bump_prom >= 0.18)
                + 0.4 * float(feature.max_gap_ms >= 38.0 and feature.late_ratio >= 0.95)
            )
            resonant_score = (
                max((feature.tail_ratio - 1.9) / 2.0, 0.0)
                + 0.4 * max((feature.late_ratio - 1.25) / 1.5, 0.0)
                + 0.8 * max((feature.decay_slope_db_per_ms - decay_q70) / 0.8, 0.0)
                + 0.8 * max((feature.air_ratio - air_q70) / max(air_q70, 1e-9), 0.0)
            )
            families.append("ring_delayed" if delay_score >= resonant_score else "ring_resonant")
        elif delayed:
            families.append("ring_delayed")
        elif resonant:
            families.append("ring_resonant")
        else:
            families.append("ring_resonant")
    return families


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


def _shift_window(values: np.ndarray, lag: int) -> np.ndarray:
    if lag > 0:
        return np.pad(values, (lag, 0))[: len(values)]
    if lag < 0:
        shift = abs(lag)
        return np.pad(values[shift:], (0, shift))
    return values


def _window_log_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    n_fft = 512
    ref_windowed = reference * np.hanning(len(reference))
    cand_windowed = candidate * np.hanning(len(candidate))
    ref_spec = np.log10(np.maximum(np.abs(np.fft.rfft(ref_windowed, n=n_fft)), 1e-8))
    cand_spec = np.log10(np.maximum(np.abs(np.fft.rfft(cand_windowed, n=n_fft)), 1e-8))
    return float(np.mean(np.abs(ref_spec - cand_spec)))


def event_window_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_events: list[dict[str, float]],
    selected: np.ndarray | None = None,
    max_lag_ms: float = 1.0,
) -> dict[str, float]:
    if not reference_events:
        return {
            "weighted_correlation": 0.0,
            "median_correlation": 0.0,
            "p10_correlation": 0.0,
            "spectral_distance": 0.0,
        }
    indices = selected if selected is not None else np.arange(len(reference_events))
    max_lag = int((max_lag_ms / 1000.0) * SAMPLE_RATE)
    amplitudes = np.asarray([reference_events[int(index)]["a"] for index in indices], dtype=np.float64)
    weights = amplitudes / max(float(np.sum(amplitudes)), 1e-9)
    correlations: list[float] = []
    distances: list[float] = []

    for index in indices:
        center = round(reference_events[int(index)]["t"] * SAMPLE_RATE)
        lo = max(0, center - int(0.004 * SAMPLE_RATE))
        hi = min(len(reference), center + int(0.020 * SAMPLE_RATE))
        ref_window = reference[lo:hi]
        candidate_window = candidate[lo:hi]
        if len(ref_window) < 96 or len(ref_window) != len(candidate_window):
            continue
        best_corr = -1.0
        best_window = candidate_window
        for lag in range(-max_lag, max_lag + 1):
            shifted = _shift_window(candidate_window, lag)
            corr = normalized_correlation(ref_window, shifted)
            if corr > best_corr:
                best_corr = corr
                best_window = shifted
        correlations.append(best_corr)
        distances.append(_window_log_distance(ref_window, best_window))

    if not correlations:
        return {
            "weighted_correlation": 0.0,
            "median_correlation": 0.0,
            "p10_correlation": 0.0,
            "spectral_distance": 0.0,
        }

    correlation_array = np.asarray(correlations, dtype=np.float64)
    distance_array = np.asarray(distances, dtype=np.float64)
    weight_array = weights[: len(correlation_array)]
    weight_array = weight_array / max(float(np.sum(weight_array)), 1e-9)
    return {
        "weighted_correlation": float(np.sum(correlation_array * weight_array)),
        "median_correlation": float(np.median(correlation_array)),
        "p10_correlation": float(np.percentile(correlation_array, 10.0)),
        "spectral_distance": float(np.sum(distance_array * weight_array)),
    }


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
    params = with_model_defaults(params)
    event_times = [float(event["t"]) for event in reference.events]
    strongest = strongest_event_indices(reference.events)
    ring_delayed_indices = np.asarray(
        [index for index, family in enumerate(reference.families) if family == "ring_delayed"],
        dtype=np.int64,
    )
    ring_resonant_indices = np.asarray(
        [index for index, family in enumerate(reference.families) if family == "ring_resonant"],
        dtype=np.int64,
    )
    ringy_indices = np.asarray(
        [
            index
            for index, family in enumerate(reference.families)
            if family in {"ring_delayed", "ring_resonant"}
        ],
        dtype=np.int64,
    )
    tight_indices = np.asarray([index for index, family in enumerate(reference.families) if family == "tight"], dtype=np.int64)

    conditioned_schedule = build_conditioned_schedule(params, reference.events, reference.families)
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
    all_event_metrics = event_window_metrics(ref_norm, aligned_norm, reference.events)
    strong_event_metrics = event_window_metrics(ref_norm, aligned_norm, reference.events, selected=strongest)
    tight_event_metrics = event_window_metrics(ref_norm, aligned_norm, reference.events, selected=tight_indices)
    ringy_event_metrics = event_window_metrics(ref_norm, aligned_norm, reference.events, selected=ringy_indices)
    ring_delayed_metrics = event_window_metrics(ref_norm, aligned_norm, reference.events, selected=ring_delayed_indices)
    ring_resonant_metrics = event_window_metrics(ref_norm, aligned_norm, reference.events, selected=ring_resonant_indices)

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

    event_conditioned_score = (
        2.4 * (1.0 - strong_event_metrics["weighted_correlation"])
        + 1.8 * (1.0 - strong_event_metrics["median_correlation"])
        + 1.0 * (1.0 - strong_event_metrics["p10_correlation"])
        + 0.45 * strong_event_metrics["spectral_distance"]
        + 1.8 * (1.0 - ringy_event_metrics["weighted_correlation"])
        + 1.0 * (1.0 - ringy_event_metrics["median_correlation"])
        + 0.45 * ringy_event_metrics["spectral_distance"]
        + 0.45 * (1.0 - tight_event_metrics["weighted_correlation"])
        + 1.8 * feature_pair["envelope_error"]
        + 0.10 * feature_pair["log_mel_distance"]
        + abs(body_synth - body_ref) / max(body_ref, 1e-9) * 1.6
        + abs(center_synth - center_ref) / max(center_ref, 1e-9) * 1.8
        + abs(air_synth - air_ref) / max(air_ref, 1e-9) * 1.2
        + abs(spectral_centroid(synth_freqs, synth_spectrum) - spectral_centroid(ref_freqs, ref_spectrum)) / 250.0
    )
    schedule_error_value = scheduler_score(candidate_schedule_stats, target_schedule_stats)

    return {
        "composite_score": event_conditioned_score,
        "event_conditioned_score": event_conditioned_score,
        "scheduler_score": schedule_error_value,
        "conditioned_render": conditioned_render,
        "free_render": free_render,
        "conditioned_metrics": {
            "waveform_correlation": waveform_corr,
            "transient_window_correlation": transient_corr,
            "event_window_metrics_all": all_event_metrics,
            "event_window_metrics_strong": strong_event_metrics,
            "event_window_metrics_tight": tight_event_metrics,
            "event_window_metrics_ringy": ringy_event_metrics,
            "event_window_metrics_ring_delayed": ring_delayed_metrics,
            "event_window_metrics_ring_resonant": ring_resonant_metrics,
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
        "reference_families": {
            "tight": int(np.sum(np.asarray(reference.families) == "tight")),
            "ring_delayed": int(np.sum(np.asarray(reference.families) == "ring_delayed")),
            "ring_resonant": int(np.sum(np.asarray(reference.families) == "ring_resonant")),
            "ringy": int(np.sum(np.isin(np.asarray(reference.families), ["ring_delayed", "ring_resonant"]))),
        },
    }


def feature_vector(feature: ReferenceEventFeature) -> np.ndarray:
    return np.asarray(
        [
            feature.peak_lag_ms,
            feature.late_ratio,
            feature.tail_ratio,
            feature.max_gap_ms,
            feature.air_ratio,
            feature.second_bump_lag_ms,
            feature.second_bump_prom,
            feature.decay_slope_db_per_ms,
        ],
        dtype=np.float64,
    )


def representative_family_indices(
    reference: ReferenceBundle,
    family: str,
    count: int = 10,
) -> np.ndarray:
    indices = [index for index, label in enumerate(reference.families) if label == family]
    if len(indices) <= count:
        return np.asarray(indices, dtype=np.int64)

    selected: list[int] = []

    def add_candidates(candidates: list[int]) -> None:
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= count:
                break

    family_features = [reference.features[index] for index in indices]
    amplitudes = np.asarray([feature.amplitude for feature in family_features], dtype=np.float64)
    median_index = indices[int(np.argmin(np.abs(amplitudes - float(np.median(amplitudes)))))]
    add_candidates([median_index])

    tail_ranked = sorted(indices, key=lambda idx: reference.features[idx].tail_ratio, reverse=True)
    gap_ranked = sorted(indices, key=lambda idx: reference.features[idx].max_gap_ms, reverse=True)
    add_candidates(tail_ranked[:2])
    add_candidates(gap_ranked[:2])

    if family == "ring_delayed":
        delayed_ranked = sorted(
            indices,
            key=lambda idx: (
                reference.features[idx].second_bump_prom,
                reference.features[idx].second_bump_lag_ms,
                reference.features[idx].peak_lag_ms,
            ),
            reverse=True,
        )
        add_candidates(delayed_ranked[:3])
    else:
        resonant_ranked = sorted(
            indices,
            key=lambda idx: (
                reference.features[idx].air_ratio,
                reference.features[idx].tail_ratio,
                reference.features[idx].decay_slope_db_per_ms,
            ),
            reverse=True,
        )
        add_candidates(resonant_ranked[:3])

    vectors = np.vstack([feature_vector(reference.features[index]) for index in indices]).astype(np.float64)
    means = np.mean(vectors, axis=0)
    stds = np.std(vectors, axis=0)
    stds[stds < 1e-6] = 1.0
    normalized = (vectors - means) / stds
    centroid = np.mean(normalized, axis=0)
    distances = np.linalg.norm(normalized - centroid, axis=1)
    centroid_ranked = [indices[i] for i in np.argsort(distances)]
    add_candidates(centroid_ranked)

    return np.asarray(selected[:count], dtype=np.int64)


def prototype_subset_score(
    params: dict[str, float],
    reference: ReferenceBundle,
    selected: np.ndarray,
) -> float:
    conditioned_schedule = build_conditioned_schedule(params, reference.events, reference.families)
    conditioned_render = render_schedule(params, conditioned_schedule, len(reference.samples) / reference.sample_rate)
    ref_band = bandpass_clicks(reference.samples)
    conditioned_band = bandpass_clicks(conditioned_render)
    aligned_band, _ = align_by_cross_correlation(ref_band, conditioned_band)

    ref_peak = max(float(np.max(np.abs(ref_band))), 1e-9)
    aligned_peak = max(float(np.max(np.abs(aligned_band))), 1e-9)
    ref_norm = ref_band / ref_peak
    aligned_norm = aligned_band / aligned_peak

    metrics = event_window_metrics(ref_norm, aligned_norm, reference.events, selected=selected)
    return (
        2.8 * (1.0 - metrics["weighted_correlation"])
        + 1.2 * (1.0 - metrics["median_correlation"])
        + 0.8 * (1.0 - metrics["p10_correlation"])
        + 0.55 * metrics["spectral_distance"]
    )


def optimize_family_prototypes(params: dict[str, float], reference: ReferenceBundle) -> dict[str, float]:
    best = with_model_defaults(params)
    best_full_score = float(validate_params(best, reference)["event_conditioned_score"])

    family_keys = {
        "ring_delayed": [
            "ringDelayedLaunchMul",
            "ringDelayedBrakeDelayAddMs",
            "ringDelayedBrakeAmpMul",
            "ringDelayedShellDelayAddMs",
            "ringDelayedShellAmp",
            "ringDelayedSettleDelayMs",
            "ringDelayedSettleAmp",
            "ringDelayedModalDampScale",
            "ringDelayedAirTiltDb",
        ],
        "ring_resonant": [
            "ringResonantLaunchMul",
            "ringResonantBrakeDelayAddMs",
            "ringResonantBrakeAmpMul",
            "ringResonantShellDelayMs",
            "ringResonantShellAmp",
            "ringResonantSettleDelayMs",
            "ringResonantSettleAmp",
            "ringResonantModalDampScale",
            "ringResonantAirTiltDb",
        ],
    }
    step_sizes = {
        "ringDelayedLaunchMul": 0.05,
        "ringDelayedBrakeDelayAddMs": 0.60,
        "ringDelayedBrakeAmpMul": 0.08,
        "ringDelayedShellDelayAddMs": 0.80,
        "ringDelayedShellAmp": 0.03,
        "ringDelayedSettleDelayMs": 1.8,
        "ringDelayedSettleAmp": 0.02,
        "ringDelayedModalDampScale": 0.04,
        "ringDelayedAirTiltDb": 0.35,
        "ringResonantLaunchMul": 0.05,
        "ringResonantBrakeDelayAddMs": 0.35,
        "ringResonantBrakeAmpMul": 0.06,
        "ringResonantShellDelayMs": 0.45,
        "ringResonantShellAmp": 0.02,
        "ringResonantSettleDelayMs": 1.4,
        "ringResonantSettleAmp": 0.03,
        "ringResonantModalDampScale": 0.04,
        "ringResonantAirTiltDb": 0.45,
    }
    limits = {
        "ringDelayedLaunchMul": (0.70, 1.05),
        "ringDelayedBrakeDelayAddMs": (1.0, 6.0),
        "ringDelayedBrakeAmpMul": (0.80, 1.40),
        "ringDelayedShellDelayAddMs": (2.0, 10.0),
        "ringDelayedShellAmp": (0.05, 0.45),
        "ringDelayedSettleDelayMs": (14.0, 28.0),
        "ringDelayedSettleAmp": (0.0, 0.20),
        "ringDelayedModalDampScale": (0.90, 1.10),
        "ringDelayedAirTiltDb": (0.0, 3.0),
        "ringResonantLaunchMul": (0.75, 1.0),
        "ringResonantBrakeDelayAddMs": (-1.0, 3.0),
        "ringResonantBrakeAmpMul": (0.60, 1.10),
        "ringResonantShellDelayMs": (4.5, 7.5),
        "ringResonantShellAmp": (0.0, 0.16),
        "ringResonantSettleDelayMs": (14.0, 26.0),
        "ringResonantSettleAmp": (0.08, 0.35),
        "ringResonantModalDampScale": (0.70, 0.95),
        "ringResonantAirTiltDb": (1.0, 5.0),
    }

    for family in ("ring_delayed", "ring_resonant"):
        selected = representative_family_indices(reference, family, count=10)
        if selected.size == 0:
            continue
        trial = dict(best)
        best_subset = prototype_subset_score(trial, reference, selected)
        for shrink in (1.0, 0.60, 0.36, 0.22, 0.14):
            improved = True
            while improved:
                improved = False
                for key in family_keys[family]:
                    step = step_sizes[key] * shrink
                    for direction in (-1.0, 1.0):
                        candidate = dict(trial)
                        lo, hi = limits[key]
                        candidate[key] = float(min(max(candidate[key] + direction * step, lo), hi))
                        score = prototype_subset_score(candidate, reference, selected)
                        if score + 1e-9 < best_subset:
                            trial = candidate
                            best_subset = score
                            improved = True
        trial_full_score = float(validate_params(trial, reference)["event_conditioned_score"])
        if trial_full_score + 1e-9 < best_full_score:
            best = trial
            best_full_score = trial_full_score

    return best


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
        "packetAccent",
    ]
    step_sizes = {
        "denseGapMs": 3.0,
        "midGapMs": 12.0,
        "resetGapMs": 18.0,
        "packetLengthMean": 1.0,
        "packetLengthJitter": 0.6,
        "resetProbability": 0.06,
        "ampJitter": 0.06,
        "packetAccent": 0.10,
    }
    limits = {
        "denseGapMs": (8.0, 28.0),
        "midGapMs": (40.0, 150.0),
        "resetGapMs": (110.0, 260.0),
        "packetLengthMean": (2.0, 10.0),
        "packetLengthJitter": (0.0, 4.5),
        "resetProbability": (0.02, 0.70),
        "ampJitter": (0.0, 0.70),
        "packetAccent": (0.0, 1.10),
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
    best = with_model_defaults(params)
    keys = [
        "pulseDecay",
        "launchAmp",
        "brakeAmp",
        "ringyLaunchScale",
        "ringyBrakeDelayMs",
        "body2Hz",
        "body2Gain",
        "body2T60",
        "air2Gain",
        "air2T60",
        "ringDelayedLaunchMul",
        "ringDelayedBrakeDelayAddMs",
        "ringDelayedBrakeAmpMul",
        "ringDelayedShellDelayAddMs",
        "ringDelayedShellAmp",
        "ringDelayedSettleDelayMs",
        "ringDelayedSettleAmp",
        "ringDelayedModalDampScale",
        "ringDelayedAirTiltDb",
        "ringResonantBrakeAmpMul",
        "ringResonantShellDelayMs",
        "ringResonantShellAmp",
        "ringResonantSettleAmp",
        "ringResonantModalDampScale",
        "ringResonantAirTiltDb",
    ]
    step_sizes = {
        "pulseDecay": 0.03,
        "launchAmp": 0.04,
        "brakeAmp": 0.04,
        "ringyLaunchScale": 0.05,
        "ringyBrakeDelayMs": 0.06,
        "body2Hz": 8.0,
        "body2Gain": 0.006,
        "body2T60": 1.2,
        "air2Gain": 0.008,
        "air2T60": 0.25,
        "ringDelayedLaunchMul": 0.05,
        "ringDelayedBrakeDelayAddMs": 0.45,
        "ringDelayedBrakeAmpMul": 0.06,
        "ringDelayedShellDelayAddMs": 0.50,
        "ringDelayedShellAmp": 0.02,
        "ringDelayedSettleDelayMs": 1.2,
        "ringDelayedSettleAmp": 0.02,
        "ringDelayedModalDampScale": 0.03,
        "ringDelayedAirTiltDb": 0.30,
        "ringResonantBrakeAmpMul": 0.04,
        "ringResonantShellDelayMs": 0.30,
        "ringResonantShellAmp": 0.02,
        "ringResonantSettleAmp": 0.02,
        "ringResonantModalDampScale": 0.02,
        "ringResonantAirTiltDb": 0.20,
    }
    limits = {
        "pulseDecay": (0.55, 0.98),
        "launchAmp": (0.30, 1.10),
        "brakeAmp": (-0.80, -0.05),
        "ringyLaunchScale": (0.45, 1.10),
        "ringyBrakeDelayMs": (0.16, 1.40),
        "body2Hz": (620.0, 700.0),
        "body2Gain": (0.0, 0.08),
        "body2T60": (8.0, 48.0),
        "air2Gain": (0.0, 0.08),
        "air2T60": (1.0, 10.0),
        "ringDelayedLaunchMul": (0.70, 1.05),
        "ringDelayedBrakeDelayAddMs": (1.0, 6.0),
        "ringDelayedBrakeAmpMul": (0.80, 1.40),
        "ringDelayedShellDelayAddMs": (2.0, 10.0),
        "ringDelayedShellAmp": (0.05, 0.45),
        "ringDelayedSettleDelayMs": (14.0, 28.0),
        "ringDelayedSettleAmp": (0.0, 0.20),
        "ringDelayedModalDampScale": (0.90, 1.10),
        "ringDelayedAirTiltDb": (0.0, 3.0),
        "ringResonantBrakeAmpMul": (0.60, 1.10),
        "ringResonantShellDelayMs": (4.5, 7.5),
        "ringResonantShellAmp": (0.0, 0.16),
        "ringResonantSettleAmp": (0.08, 0.35),
        "ringResonantModalDampScale": (0.70, 0.95),
        "ringResonantAirTiltDb": (1.0, 5.0),
    }

    validation = validate_params(best, reference)
    best_score = float(validation["event_conditioned_score"])

    for shrink in (1.0, 0.60, 0.36):
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
                    score = float(validation["event_conditioned_score"])
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


def export_params(params: dict[str, float]) -> dict[str, float]:
    exported: dict[str, float] = {}
    for key, value in params.items():
        exported[key] = round(float(value), 6)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and validate the MH thrash lab against the local HDD reference.")
    parser.add_argument("--report", type=Path, default=ACCURACY_DIR / "hdd-sample-validation.json")
    parser.add_argument("--conditioned-wav", type=Path, default=ACCURACY_DIR / "conditioned-fit.wav")
    parser.add_argument("--free-wav", type=Path, default=ACCURACY_DIR / "free-running-fit.wav")
    parser.add_argument("--recommended-json", type=Path, default=ACCURACY_DIR / "recommended-defaults.json")
    args = parser.parse_args()

    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_reference_bundle()
    starting_params = with_model_defaults(extract_html_defaults(HTML_PATH))

    baseline = validate_params(starting_params, reference)
    fitted = optimize_scheduler(starting_params, reference.events)
    fitted = optimize_model(fitted, reference)
    fitted = optimize_family_prototypes(fitted, reference)
    final_validation = validate_params(fitted, reference)

    exported_params = export_params(fitted)

    write_wav(args.conditioned_wav, final_validation["conditioned_render"], SAMPLE_RATE)
    write_wav(args.free_wav, final_validation["free_render"], SAMPLE_RATE)
    args.recommended_json.write_text(json.dumps(exported_params, indent=2), encoding="utf-8")

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
            "params": exported_params,
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
