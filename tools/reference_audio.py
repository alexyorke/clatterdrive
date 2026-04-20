from __future__ import annotations

import argparse
import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import numpy.typing as npt
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tools" / "reference_audio_manifest.json"
RUNTIME_DIR = ROOT / ".runtime" / "reference_audio"
RAW_DIR = RUNTIME_DIR / "raw"
WAV_DIR = RUNTIME_DIR / "wav"
REPORT_DIR = RUNTIME_DIR / "reports"
COMMITTED_REPORT_DIR = ROOT / "docs" / "reference-calibration"
TARGET_SAMPLE_RATE = 22050
STARTUP_BUCKETS = {"desktop_7200_internal", "enterprise_ultrastar"}
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ReferenceSource:
    id: str
    title: str
    url: str
    drive_bucket: str
    segment_type: str
    segment_start_s: float
    segment_end_s: float
    confidence: float
    notes: str

    @property
    def is_startup_reference(self) -> bool:
        return (
            self.segment_type in {"startup_only", "startup_plus_post_ready"}
            and self.confidence >= 0.6
            and "dual-drive" not in self.notes.lower()
            and "dual drive" not in self.notes.lower()
        )


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def _video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.rsplit("/", 1)[-1]
    if parsed.path.startswith("/shorts/"):
        return parsed.path.split("/")[-1]
    query = parse_qs(parsed.query)
    if "v" in query:
        return query["v"][0]
    return parsed.path.rsplit("/", 1)[-1]


def load_manifest(path: Path = MANIFEST_PATH) -> list[ReferenceSource]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ReferenceSource(**item) for item in payload]


def _raw_download_path(entry: ReferenceSource) -> Path | None:
    candidates = sorted(RAW_DIR.glob(f"{entry.id}.*"))
    return candidates[0] if candidates else None


def download_reference(entry: ReferenceSource) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = _raw_download_path(entry)
    if existing is not None:
        return existing
    _run(
        [
            "yt-dlp",
            "--no-playlist",
            "-o",
            str(RAW_DIR / f"{entry.id}.%(ext)s"),
            entry.url,
        ]
    )
    downloaded = _raw_download_path(entry)
    if downloaded is None:
        raise FileNotFoundError(f"download missing for {entry.id}")
    return downloaded


def extract_reference_segment(entry: ReferenceSource) -> Path:
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    output = WAV_DIR / f"{entry.id}.wav"
    if output.exists():
        return output
    raw_path = download_reference(entry)
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{entry.segment_start_s:.3f}",
            "-to",
            f"{entry.segment_end_s:.3f}",
            "-i",
            str(raw_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(output),
        ]
    )
    return output


def load_wav(path: Path) -> tuple[FloatArray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def _hz_to_mel(hz: FloatArray) -> FloatArray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: FloatArray) -> FloatArray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filter_bank(sample_rate: int, n_fft: int, n_mels: int = 48) -> FloatArray:
    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1, dtype=np.float64)
    mel_points = np.linspace(_hz_to_mel(np.array([20.0]))[0], _hz_to_mel(np.array([5000.0]))[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bank = np.zeros((n_mels, len(freqs)), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = hz_points[index : index + 3]
        left_slope = (freqs - left) / max(center - left, 1e-9)
        right_slope = (right - freqs) / max(right - center, 1e-9)
        bank[index] = np.maximum(0.0, np.minimum(left_slope, right_slope))
    row_sums = np.sum(bank, axis=1, keepdims=True)
    return bank / np.maximum(row_sums, 1e-9)


def _analysis_band(bucket: str) -> tuple[float, float]:
    if bucket == "external_enclosure":
        return 45.0, 125.0
    return 55.0, 130.0


def _resample_series(values: FloatArray, length: int) -> FloatArray:
    if len(values) == 0:
        return np.zeros(length, dtype=np.float64)
    if len(values) == length:
        return values.astype(np.float64, copy=True)
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return np.interp(target, source, values).astype(np.float64)


def _resample_matrix(values: FloatArray, frames: int) -> FloatArray:
    if values.size == 0:
        return np.zeros((0, frames), dtype=np.float64)
    resampled = np.vstack([_resample_series(row, frames) for row in values])
    return resampled.astype(np.float64)


def _first_audible_time(envelope: FloatArray, times: FloatArray) -> tuple[float, int]:
    if len(envelope) == 0:
        return 0.0, 0
    noise_window = max(4, min(len(envelope) // 8, 18))
    noise_floor = float(np.median(envelope[:noise_window])) if noise_window > 0 else 0.0
    threshold = max(noise_floor * 3.0, float(np.max(envelope)) * 0.08, 1e-4)
    indices = np.flatnonzero(envelope >= threshold)
    if len(indices) == 0:
        return 0.0, 0
    first = int(indices[0])
    return float(times[first]), first


def _harmonic_distribution(freqs: FloatArray, spectrum: FloatArray, fundamental_hz: float) -> list[float]:
    if fundamental_hz <= 1.0:
        return [0.0] * 6
    bands: list[float] = []
    for harmonic in range(1, 7):
        center = harmonic * fundamental_hz
        mask = (freqs >= center - 6.0) & (freqs <= center + 6.0)
        bands.append(float(np.sum(spectrum[mask])))
    total = sum(bands)
    if total <= 1e-12:
        return [0.0] * 6
    return [value / total for value in bands]


def compute_audio_features(samples: FloatArray, sample_rate: int, bucket: str) -> dict[str, Any]:
    if samples.size == 0:
        return {
            "sample_rate": sample_rate,
            "duration_s": 0.0,
            "first_audible_s": 0.0,
            "time_to_50_s": 0.0,
            "time_to_90_s": 0.0,
            "time_to_99_s": 0.0,
            "steady_fundamental_hz": 0.0,
            "spectral_centroid_hz": 0.0,
            "low_band_ratio": 0.0,
            "transient_density": 0.0,
            "spectral_flux": 0.0,
            "bubbly_modulation_ratio": 0.0,
            "harmonic_distribution": [0.0] * 6,
            "time_axis": np.zeros(0, dtype=np.float64),
            "envelope_curve": np.zeros(0, dtype=np.float64),
            "centroid_curve": np.zeros(0, dtype=np.float64),
            "low_ratio_curve": np.zeros(0, dtype=np.float64),
            "fundamental_curve": np.zeros(0, dtype=np.float64),
            "log_mel": np.zeros((48, 0), dtype=np.float64),
            "onset_index": 0,
            "onset_frame_time_s": 0.0,
        }

    normalized = samples.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(normalized)))
    if peak > 1e-9:
        normalized = normalized / peak

    n_fft = 1024
    hop = 256
    noverlap = n_fft - hop
    freqs, times, stft = signal.stft(
        normalized,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft)
    power = np.maximum(magnitude**2, 1e-12)
    envelope_curve = np.sqrt(np.mean(power, axis=0))
    first_audible_s, onset_index = _first_audible_time(envelope_curve, times)

    centroid_curve = np.sum(freqs[:, None] * magnitude, axis=0) / np.maximum(np.sum(magnitude, axis=0), 1e-12)
    low_band = np.sum(magnitude[(freqs >= 20.0) & (freqs < 120.0)], axis=0)
    low_mid = np.sum(magnitude[(freqs >= 120.0) & (freqs < 500.0)], axis=0)
    low_ratio_curve = low_band / np.maximum(low_mid, 1e-12)

    search_lo, search_hi = _analysis_band(bucket)
    band_mask = (freqs >= search_lo) & (freqs <= search_hi)
    if np.any(band_mask):
        band_freqs = freqs[band_mask]
        band_mag = magnitude[band_mask]
        band_energy = np.sum(band_mag, axis=0)
        activity_floor = max(float(np.max(band_energy)) * 0.02, 1e-7)
        activity_mask = (envelope_curve >= max(float(np.max(envelope_curve)) * 0.08, 1e-4)) & (band_energy >= activity_floor)
        fundamental_curve = np.where(activity_mask, band_freqs[np.argmax(band_mag, axis=0)], 0.0)
    else:
        fundamental_curve = np.zeros_like(times)

    steady_slice = slice(max(onset_index, int(len(fundamental_curve) * 0.80)), len(fundamental_curve))
    steady_segment = fundamental_curve[steady_slice]
    steady_segment = steady_segment[steady_segment > 1.0]
    steady_fundamental_hz = float(np.median(steady_segment)) if len(steady_segment) else 0.0

    def _time_to_fraction(fraction: float) -> float:
        if steady_fundamental_hz <= 1.0:
            return 0.0
        indices = np.flatnonzero(fundamental_curve[onset_index:] >= steady_fundamental_hz * fraction)
        if len(indices) == 0:
            return float(times[-1]) if len(times) else 0.0
        return float(times[onset_index + int(indices[0])])

    flux_values = np.maximum(0.0, np.diff(magnitude, axis=1))
    spectral_flux = float(np.mean(np.sum(flux_values, axis=0))) if flux_values.size else 0.0

    average_spectrum = np.mean(magnitude[:, max(onset_index, 0) :], axis=1) if magnitude.shape[1] else np.zeros_like(freqs)
    harmonic_distribution = _harmonic_distribution(freqs, average_spectrum, steady_fundamental_hz)

    modulation_start = min(len(envelope_curve), onset_index + 4)
    envelope_centered = envelope_curve[modulation_start:] - float(np.mean(envelope_curve[modulation_start:])) if modulation_start < len(envelope_curve) else np.zeros(0, dtype=np.float64)
    if len(envelope_centered) > 1:
        env_rate = sample_rate / hop
        env_shaped = signal.detrend(envelope_centered, type="linear")
        b, a = signal.butter(2, 1.5 / max(env_rate * 0.5, 1e-9), btype="highpass")
        env_shaped = signal.filtfilt(b, a, env_shaped) if len(env_shaped) > 12 else env_shaped
        env_freqs = np.fft.rfftfreq(len(env_shaped), d=hop / sample_rate)
        env_power = np.abs(np.fft.rfft(env_shaped)) ** 2
        bubbly = float(np.sum(env_power[(env_freqs >= 2.0) & (env_freqs <= 12.0)]))
        total_mod = float(np.sum(env_power[(env_freqs >= 0.25) & (env_freqs <= 20.0)]))
        bubbly_ratio = bubbly / max(total_mod, 1e-12)
    else:
        bubbly_ratio = 0.0

    mel_bank = _mel_filter_bank(sample_rate, n_fft)
    log_mel = 10.0 * np.log10(np.maximum(mel_bank @ power, 1e-10))

    return {
        "sample_rate": sample_rate,
        "duration_s": len(samples) / sample_rate,
        "first_audible_s": first_audible_s,
        "time_to_50_s": _time_to_fraction(0.50),
        "time_to_90_s": _time_to_fraction(0.90),
        "time_to_99_s": _time_to_fraction(0.99),
        "steady_fundamental_hz": steady_fundamental_hz,
        "spectral_centroid_hz": float(np.mean(centroid_curve)) if len(centroid_curve) else 0.0,
        "low_band_ratio": float(np.mean(low_ratio_curve)) if len(low_ratio_curve) else 0.0,
        "transient_density": float(np.mean(np.abs(np.diff(normalized)))) if len(normalized) > 1 else 0.0,
        "spectral_flux": spectral_flux,
        "bubbly_modulation_ratio": bubbly_ratio,
        "harmonic_distribution": harmonic_distribution,
        "time_axis": times,
        "envelope_curve": envelope_curve,
        "centroid_curve": centroid_curve,
        "low_ratio_curve": low_ratio_curve,
        "fundamental_curve": fundamental_curve,
        "log_mel": log_mel,
        "onset_index": onset_index,
        "onset_frame_time_s": float(times[onset_index]) if len(times) else 0.0,
    }


def _aligned_curve(feature: dict[str, Any], key: str, frames: int) -> FloatArray:
    values = np.asarray(feature[key], dtype=np.float64)
    onset = int(feature["onset_index"])
    return _resample_series(values[onset:], frames)


def _aligned_log_mel(feature: dict[str, Any], frames: int) -> FloatArray:
    values = np.asarray(feature["log_mel"], dtype=np.float64)
    onset = int(feature["onset_index"])
    return _resample_matrix(values[:, onset:], frames)


def compare_startup_features(
    generated: dict[str, Any],
    references: list[tuple[ReferenceSource, dict[str, Any]]],
) -> dict[str, Any]:
    startup_refs = [(entry, feature) for entry, feature in references if entry.is_startup_reference]
    if not startup_refs:
        return {"references_used": 0}

    curve_frames = 160
    mel_frames = 120
    envelopes = np.vstack([_aligned_curve(feature, "envelope_curve", curve_frames) for _, feature in startup_refs])
    centroids = np.vstack([_aligned_curve(feature, "centroid_curve", curve_frames) for _, feature in startup_refs])
    low_ratios = np.vstack([_aligned_curve(feature, "low_ratio_curve", curve_frames) for _, feature in startup_refs])
    fundamentals = np.vstack([_aligned_curve(feature, "fundamental_curve", curve_frames) for _, feature in startup_refs])
    mels = np.stack([_aligned_log_mel(feature, mel_frames) for _, feature in startup_refs], axis=0)

    median_envelope = np.median(envelopes, axis=0)
    median_centroid = np.median(centroids, axis=0)
    median_low_ratio = np.median(low_ratios, axis=0)
    median_fundamental = np.median(fundamentals, axis=0)
    median_mel = np.median(mels, axis=0)

    generated_envelope = _aligned_curve(generated, "envelope_curve", curve_frames)
    generated_centroid = _aligned_curve(generated, "centroid_curve", curve_frames)
    generated_low_ratio = _aligned_curve(generated, "low_ratio_curve", curve_frames)
    generated_fundamental = _aligned_curve(generated, "fundamental_curve", curve_frames)
    generated_mel = _aligned_log_mel(generated, mel_frames)

    envelope_error = float(np.mean(np.abs(generated_envelope - median_envelope)))
    centroid_error = float(np.mean(np.abs(generated_centroid - median_centroid)))
    low_ratio_error = float(np.mean(np.abs(generated_low_ratio - median_low_ratio)))
    fundamental_error = float(np.mean(np.abs(generated_fundamental - median_fundamental)))
    log_mel_distance = float(np.mean(np.abs(generated_mel - median_mel)))
    modulation_error = float(
        abs(float(generated["bubbly_modulation_ratio"]) - float(np.median([feature["bubbly_modulation_ratio"] for _, feature in startup_refs])))
    )

    onset_values = [float(feature["first_audible_s"]) for _, feature in startup_refs]
    ninety_values = [float(feature["time_to_90_s"]) for _, feature in startup_refs]
    centroid_values = [float(feature["spectral_centroid_hz"]) for _, feature in startup_refs]
    low_ratio_values = [float(feature["low_band_ratio"]) for _, feature in startup_refs]
    modulation_values = [float(feature["bubbly_modulation_ratio"]) for _, feature in startup_refs]

    return {
        "references_used": len(startup_refs),
        "reference_ids": [entry.id for entry, _ in startup_refs],
        "generated": {
            "first_audible_s": float(generated["first_audible_s"]),
            "time_to_90_s": float(generated["time_to_90_s"]),
            "steady_fundamental_hz": float(generated["steady_fundamental_hz"]),
            "spectral_centroid_hz": float(generated["spectral_centroid_hz"]),
            "low_band_ratio": float(generated["low_band_ratio"]),
            "bubbly_modulation_ratio": float(generated["bubbly_modulation_ratio"]),
            "transient_density": float(generated["transient_density"]),
        },
        "reference_band": {
            "first_audible_s_min": float(np.min(onset_values)) * 0.85,
            "first_audible_s_max": float(np.max(onset_values)) * 1.15,
            "time_to_90_s_min": float(np.min(ninety_values)) * 0.85,
            "time_to_90_s_max": float(np.max(ninety_values)) * 1.15,
            "spectral_centroid_hz_min": float(np.min(centroid_values)) * 0.85,
            "spectral_centroid_hz_max": float(np.max(centroid_values)) * 1.15,
            "low_band_ratio_min": float(np.min(low_ratio_values)) * 0.80,
            "low_band_ratio_max": float(np.max(low_ratio_values)) * 3.00,
            "bubbly_modulation_ratio_max": float(np.max(modulation_values)) * 1.35,
        },
        "distance": {
            "log_mel_distance": log_mel_distance,
            "envelope_error": envelope_error,
            "centroid_error": centroid_error,
            "low_ratio_error": low_ratio_error,
            "fundamental_error": fundamental_error,
            "modulation_error": modulation_error,
        },
        "median_reference_curves": {
            "envelope": median_envelope.tolist(),
            "centroid_hz": median_centroid.tolist(),
            "low_ratio": median_low_ratio.tolist(),
            "fundamental_hz": median_fundamental.tolist(),
            "log_mel": median_mel.tolist(),
        },
        "generated_curves": {
            "envelope": generated_envelope.tolist(),
            "centroid_hz": generated_centroid.tolist(),
            "low_ratio": generated_low_ratio.tolist(),
            "fundamental_hz": generated_fundamental.tolist(),
            "log_mel": generated_mel.tolist(),
        },
    }


def _palette(value: float) -> str:
    clamped = max(0.0, min(1.0, value))
    red = int(14 + 220 * clamped)
    green = int(20 + 170 * (clamped**0.7))
    blue = int(34 + 100 * (1.0 - clamped))
    return f"rgb({red},{green},{blue})"


def _heatmap_svg(matrix: FloatArray, left: float, top: float, width: float, height: float) -> list[str]:
    if matrix.size == 0:
        return []
    rows, cols = matrix.shape
    minimum = float(np.min(matrix))
    maximum = float(np.max(matrix))
    scale = maximum - minimum if maximum > minimum else 1.0
    cell_w = width / cols
    cell_h = height / rows
    elements: list[str] = []
    for row in range(rows):
        y = top + (rows - 1 - row) * cell_h
        for col in range(cols):
            value = (float(matrix[row, col]) - minimum) / scale
            x = left + col * cell_w
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.4:.2f}" height="{cell_h + 0.4:.2f}" fill="{_palette(value)}"/>'
            )
    return elements


def _line_points(values: FloatArray, left: float, top: float, width: float, height: float) -> str:
    if len(values) == 0:
        return ""
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    scale = maximum - minimum if maximum > minimum else 1.0
    points = []
    for index, value in enumerate(values):
        x = left + width * (index / max(len(values) - 1, 1))
        normalized = (float(value) - minimum) / scale
        y = top + height * (1.0 - normalized)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_startup_summary_svg(path: Path, summary: dict[str, Any]) -> None:
    median = summary["median_reference_curves"]
    generated = summary["generated_curves"]
    width = 1280.0
    height = 1120.0
    margin = 56.0
    panel_w = (width - margin * 3) / 2.0
    panel_h = 210.0
    heatmap_h = 240.0
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        "<style>",
        ".title{font:700 28px sans-serif;fill:#0f172a}",
        ".label{font:600 15px sans-serif;fill:#334155}",
        ".caption{font:13px sans-serif;fill:#475569}",
        ".panel{fill:white;stroke:#cbd5e1;stroke-width:1.4}",
        ".ref{fill:none;stroke:#2563eb;stroke-width:2.4}",
        ".gen{fill:none;stroke:#dc2626;stroke-width:2.4}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{margin:.0f}" y="40" class="title">ClatterDrive startup calibration summary</text>',
        f'<text x="{margin:.0f}" y="64" class="caption">Generated startup (red) vs median reference profile (blue)</text>',
    ]

    def add_line_panel(
        x: float,
        y: float,
        title: str,
        ref_values: list[float],
        gen_values: list[float],
    ) -> None:
        ref_arr = np.asarray(ref_values, dtype=np.float64)
        gen_arr = np.asarray(gen_values, dtype=np.float64)
        combined = np.concatenate([ref_arr, gen_arr]) if ref_arr.size and gen_arr.size else ref_arr if ref_arr.size else gen_arr
        if combined.size == 0:
            combined = np.zeros(2, dtype=np.float64)
        ref_scaled = _resample_series(ref_arr, 160)
        gen_scaled = _resample_series(gen_arr, 160)
        svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" class="panel"/>')
        svg.append(f'<text x="{x + 12:.2f}" y="{y + 22:.2f}" class="label">{title}</text>')
        svg.append(f'<polyline points="{_line_points(ref_scaled, x + 12, y + 36, panel_w - 24, panel_h - 48)}" class="ref"/>')
        svg.append(f'<polyline points="{_line_points(gen_scaled, x + 12, y + 36, panel_w - 24, panel_h - 48)}" class="gen"/>')

    add_line_panel(margin, 92.0, "Loudness envelope", median["envelope"], generated["envelope"])
    add_line_panel(margin * 2 + panel_w, 92.0, "Spectral centroid (Hz)", median["centroid_hz"], generated["centroid_hz"])
    add_line_panel(margin, 92.0 + panel_h + 26.0, "Low-band / low-mid ratio", median["low_ratio"], generated["low_ratio"])
    add_line_panel(
        margin * 2 + panel_w,
        92.0 + panel_h + 26.0,
        "Spindle fundamental track (Hz)",
        median["fundamental_hz"],
        generated["fundamental_hz"],
    )

    heatmap_top = 92.0 + panel_h * 2 + 78.0
    svg.append(f'<rect x="{margin:.2f}" y="{heatmap_top:.2f}" width="{panel_w:.2f}" height="{heatmap_h:.2f}" class="panel"/>')
    svg.append(f'<text x="{margin + 12:.2f}" y="{heatmap_top + 22:.2f}" class="label">Median reference log-mel spectrogram</text>')
    svg.extend(
        _heatmap_svg(
            np.asarray(median["log_mel"], dtype=np.float64),
            margin + 12.0,
            heatmap_top + 32.0,
            panel_w - 24.0,
            heatmap_h - 44.0,
        )
    )
    svg.append(f'<rect x="{margin * 2 + panel_w:.2f}" y="{heatmap_top:.2f}" width="{panel_w:.2f}" height="{heatmap_h:.2f}" class="panel"/>')
    svg.append(f'<text x="{margin * 2 + panel_w + 12:.2f}" y="{heatmap_top + 22:.2f}" class="label">Generated startup log-mel spectrogram</text>')
    svg.extend(
        _heatmap_svg(
            np.asarray(generated["log_mel"], dtype=np.float64),
            margin * 2 + panel_w + 12.0,
            heatmap_top + 32.0,
            panel_w - 24.0,
            heatmap_h - 44.0,
        )
    )

    distance = summary["distance"]
    ref_band = summary["reference_band"]
    metrics_top = heatmap_top + heatmap_h + 38.0
    metric_lines = [
        f"Generated first audible: {summary['generated']['first_audible_s']:.2f}s (reference band {ref_band['first_audible_s_min']:.2f}-{ref_band['first_audible_s_max']:.2f}s)",
        f"Generated time to 90% fundamental: {summary['generated']['time_to_90_s']:.2f}s (reference band {ref_band['time_to_90_s_min']:.2f}-{ref_band['time_to_90_s_max']:.2f}s)",
        f"Generated bubbly modulation ratio: {summary['generated']['bubbly_modulation_ratio']:.4f} (max target {ref_band['bubbly_modulation_ratio_max']:.4f})",
        f"Feature distances -> log-mel {distance['log_mel_distance']:.3f}, envelope {distance['envelope_error']:.3f}, centroid {distance['centroid_error']:.2f}",
        f"Startup references used: {', '.join(summary['reference_ids'])}",
    ]
    svg.append(f'<text x="{margin:.0f}" y="{metrics_top:.0f}" class="label">Summary</text>')
    for index, line in enumerate(metric_lines):
        svg.append(f'<text x="{margin:.0f}" y="{metrics_top + 24.0 + index * 20.0:.0f}" class="caption">{line}</text>')

    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")


def analyze_reference_bundle(
    manifest: list[ReferenceSource] | None = None,
) -> tuple[list[tuple[ReferenceSource, dict[str, Any], Path]], dict[str, Any]]:
    sources = manifest or load_manifest()
    analyzed: list[tuple[ReferenceSource, dict[str, Any], Path]] = []
    for entry in sources:
        wav_path = extract_reference_segment(entry)
        samples, sample_rate = load_wav(wav_path)
        analyzed.append((entry, compute_audio_features(samples, sample_rate, entry.drive_bucket), wav_path))
    metadata = {
        "sources": [
            {
                "id": entry.id,
                "title": entry.title,
                "url": entry.url,
                "drive_bucket": entry.drive_bucket,
                "segment_type": entry.segment_type,
                "segment_start_s": entry.segment_start_s,
                "segment_end_s": entry.segment_end_s,
                "confidence": entry.confidence,
                "notes": entry.notes,
                "wav_path": str(wav_path.relative_to(ROOT)),
                "metrics": {
                    "first_audible_s": float(features["first_audible_s"]),
                    "time_to_90_s": float(features["time_to_90_s"]),
                    "steady_fundamental_hz": float(features["steady_fundamental_hz"]),
                    "spectral_centroid_hz": float(features["spectral_centroid_hz"]),
                    "low_band_ratio": float(features["low_band_ratio"]),
                    "transient_density": float(features["transient_density"]),
                    "spectral_flux": float(features["spectral_flux"]),
                    "bubbly_modulation_ratio": float(features["bubbly_modulation_ratio"]),
                },
            }
            for entry, features, wav_path in analyzed
        ]
    }
    return analyzed, metadata


def save_reference_analysis(metadata: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "reference_metrics.json"
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and analyze local reference HDD audio clips.")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    if args.download_only:
        for entry in manifest:
            path = download_reference(entry)
            print(f"downloaded {entry.id} -> {path.relative_to(ROOT)}")
        return

    analyzed, metadata = analyze_reference_bundle(manifest)
    report_path = save_reference_analysis(metadata)
    print(f"wrote {report_path.relative_to(ROOT)}")
    if args.analyze_only:
        return
    print(f"analyzed {len(analyzed)} reference clips")


if __name__ == "__main__":
    main()
