from __future__ import annotations

import json
import time
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

import clatterdrive.audio.engine as audio_engine_module
from clatterdrive.audio import HDDAudioEngine, HDDAudioEvent
from clatterdrive.audio.core import AudioDiagnosticTrace, render_chunk as render_audio_chunk
from tools.generate_audio_samples import startup_only_duration
from tools.reference_audio import compute_audio_features
from tests.helpers import _audio_event, _wav_metrics


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _crest(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) / max(_rms(values), 1e-9)


def _transient_density(values: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(values))))


def _spectrum(values: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(values * np.hanning(len(values))))
    freqs = np.fft.rfftfreq(len(values), 1.0 / sample_rate)
    return freqs, spectrum


def _band_energy(values: np.ndarray, sample_rate: int, lo: float, hi: float) -> float:
    freqs, spectrum = _spectrum(values, sample_rate)
    return float(np.sum(spectrum[(freqs >= lo) & (freqs < hi)]))


def _dominant_freq(values: np.ndarray, sample_rate: int, lo: float, hi: float) -> float:
    freqs, spectrum = _spectrum(values, sample_rate)
    mask = (freqs >= lo) & (freqs <= hi)
    peak_index = np.argmax(spectrum[mask]) + np.where(mask)[0][0]
    return float(freqs[peak_index])


def _structure_metrics(profile: str) -> tuple[float, float, float, float, float]:
    engine = HDDAudioEngine(
        seed=0,
        drive_profile="desktop_7200_internal",
        acoustic_profile=profile,
    )
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=320, op_kind="data")
    _chunk, diagnostics = engine.render_chunk_with_diagnostics(4096)
    low_ratio = _band_energy(diagnostics.output, engine.fs, 20.0, 120.0) / max(
        _band_energy(diagnostics.output, engine.fs, 120.0, 500.0),
        1e-12,
    )
    freqs, spectrum = _spectrum(diagnostics.output, engine.fs)
    centroid = float(np.sum(freqs * spectrum) / max(np.sum(spectrum), 1e-12))
    return (
        _rms(diagnostics.output),
        centroid,
        low_ratio,
        _rms(diagnostics.structure_desk_velocity),
        _rms(diagnostics.structure_enclosure_velocity),
    )


def _metadata_storm_chunk() -> np.ndarray:
    engine = HDDAudioEngine(seed=0)
    return engine.synthesizer.render_chunk(
        4096,
        scheduled_events=[
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", servo_mode="seek", track_delta=0.18, motion_duration_ms=2.2, settle_duration_ms=1.3), 0),
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=2, op_kind="journal", servo_mode="seek", track_delta=-0.10, motion_duration_ms=1.8, settle_duration_ms=1.1), 600),
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="flush", servo_mode="seek", track_delta=0.22, is_flush=True, motion_duration_ms=2.6, settle_duration_ms=1.7), 1200),
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", servo_mode="calibration", motion_duration_ms=2.0, settle_duration_ms=1.2), 1900),
        ],
    )


def _render_startup_only(
    *,
    sample_rate: int = 22050,
    chunked: bool = False,
) -> tuple[np.ndarray, AudioDiagnosticTrace | None]:
    engine = HDDAudioEngine(
        seed=0,
        sample_rate=sample_rate,
        drive_profile="desktop_7200_internal",
        acoustic_profile="drive_on_desk",
    )
    total_frames = int(startup_only_duration("desktop_7200_internal") * sample_rate)
    startup_event = _audio_event(
        rpm=0.0,
        target_rpm=7200.0,
        queue_depth=1,
        op_kind="background",
        power_state="starting",
        heads_loaded=False,
        servo_mode="idle",
        transfer_activity=0.0,
        is_spinup=True,
    )
    if not chunked:
        diagnostics = engine.synthesizer.render_diagnostic_chunk(
            total_frames,
            scheduled_events=[(startup_event, int(0.85 * sample_rate))],
        )
        return diagnostics.output, diagnostics

    rebuilt: list[np.ndarray] = []
    remaining = total_frames
    first = True
    while remaining > 0:
        frames = min(1024, remaining)
        scheduled_events = [(startup_event, int(0.85 * sample_rate))] if first else []
        rebuilt.append(engine.synthesizer.render_chunk(frames, scheduled_events=scheduled_events))
        remaining -= frames
        first = False
    return np.concatenate(rebuilt), None


def _startup_time_to_fraction(diagnostics: AudioDiagnosticTrace, target_rpm: float, fraction: float) -> float:
    actual_rpm = np.asarray(diagnostics.actual_rpm, dtype=np.float64)
    time_s = np.asarray(diagnostics.time_s, dtype=np.float64)
    indices = np.flatnonzero(actual_rpm >= target_rpm * fraction)
    if len(indices) == 0:
        return float(time_s[-1]) if len(time_s) else 0.0
    return float(time_s[int(indices[0])])


def test_audio_engine_stop_is_safe_before_start() -> None:
    engine = HDDAudioEngine()
    engine.stop()


def test_audio_engine_can_disable_live_output_via_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HDD_AUDIO", "off")
    engine = HDDAudioEngine(seed=0)
    engine.start()
    try:
        assert engine.output_enabled is False
        assert engine.stream is None
        engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=240, op_kind="data")
        chunk = engine.render_chunk(1024)
        assert float(np.max(np.abs(chunk))) > 0.001
    finally:
        engine.stop()


def test_audio_engine_passes_explicit_device_to_sounddevice(monkeypatch: MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            events.append(("init", kwargs.get("device")))

        def start(self) -> None:
            events.append(("start", None))

        def stop(self) -> None:
            events.append(("stop", None))

        def close(self) -> None:
            events.append(("close", None))

    class FakeSD:
        OutputStream = FakeStream

    monkeypatch.setattr(audio_engine_module, "sd", FakeSD())
    monkeypatch.setenv("FAKE_HDD_AUDIO", "live")
    monkeypatch.setenv("FAKE_HDD_AUDIO_DEVICE", "7")
    engine = HDDAudioEngine(seed=0)
    engine.start()
    try:
        assert engine.output_enabled is True
        assert events[:2] == [("init", 7), ("start", None)]
    finally:
        engine.stop()


def test_audio_engine_can_tee_rendered_output_to_wave_file(tmp_path: Path) -> None:
    tee_path = tmp_path / "tee.wav"
    engine = HDDAudioEngine(seed=0, tee_path=str(tee_path))
    try:
        engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=900, op_kind="data")
        engine.render_chunk(4096)
        engine.render_chunk(4096)
    finally:
        engine.stop()

    assert tee_path.exists()
    with wave.open(str(tee_path), "rb") as wav_file:
        assert wav_file.getframerate() == engine.fs
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 8192
    rms, peak = _wav_metrics(tee_path)
    assert rms > 0.001
    assert peak > 0.004


def test_audio_engine_can_tee_output_from_env_without_live_device(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tee_path = tmp_path / "env-tee.wav"
    monkeypatch.setenv("FAKE_HDD_AUDIO", "off")
    monkeypatch.setenv("FAKE_HDD_AUDIO_TEE_PATH", str(tee_path))
    engine = HDDAudioEngine(seed=0)
    engine.start()
    try:
        engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=600, op_kind="flush", is_flush=True)
        engine.render_chunk(4096)
    finally:
        engine.stop()

    assert engine.output_enabled is False
    assert tee_path.exists()
    rms, peak = _wav_metrics(tee_path)
    assert rms > 0.001
    assert peak > 0.0035


def test_audio_engine_headless_tee_renders_without_manual_pull(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tee_path = tmp_path / "headless-tee.wav"
    monkeypatch.setenv("FAKE_HDD_AUDIO", "off")
    monkeypatch.setenv("FAKE_HDD_AUDIO_TEE_PATH", str(tee_path))
    engine = HDDAudioEngine(seed=0)
    engine.start()
    try:
        assert engine.output_enabled is False
        assert engine.stream is None
        engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=700, op_kind="data")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and engine.render_frame_cursor < engine.chunk_size * 4:
            time.sleep(0.02)
    finally:
        engine.stop()

    assert tee_path.exists()
    with wave.open(str(tee_path), "rb") as wav_file:
        assert wav_file.getnframes() >= engine.chunk_size * 4
    rms, peak = _wav_metrics(tee_path)
    assert rms > 0.001
    assert peak > 0.004


def test_audio_engine_events_are_buffered_until_render() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=700, op_kind="data")

    assert engine.events.pending_count() == 1
    assert engine.synthesizer.rpm == 0.0

    chunk = engine.render_chunk(4096)

    assert engine.events.pending_count() == 0
    assert engine.synthesizer.rpm == 7200.0
    assert engine.synthesizer.actual_rpm > 7000.0
    assert float(np.max(np.abs(chunk))) > 0.004


def test_audio_engine_event_bus_fifo_and_bounds_are_preserved() -> None:
    engine = HDDAudioEngine(seed=0, max_pending_events=3)
    for idx in range(5):
        engine.events.publish(_audio_event(rpm=5400.0 + idx, queue_depth=idx + 1, op_kind="metadata"))

    drained = engine.events.drain()

    assert [event.queue_depth for event in drained] == [3, 4, 5]
    assert engine.events.dropped_count() == 2


def test_audio_engine_future_events_do_not_disturb_the_current_chunk() -> None:
    baseline_engine = HDDAudioEngine(seed=0)
    baseline_engine.synthesizer.render_chunk(512)
    baseline = baseline_engine.synthesizer.render_chunk(512)

    engine = HDDAudioEngine(seed=0)
    engine.synthesizer.render_chunk(512)
    future_event = _audio_event(
        rpm=7200.0,
        target_rpm=7200.0,
        queue_depth=1,
        op_kind="data",
        servo_mode="seek",
        track_delta=0.20,
        motion_duration_ms=2.6,
        settle_duration_ms=1.6,
    )

    first = engine.synthesizer.render_chunk(512, scheduled_events=[(future_event, 700)])
    second = engine.synthesizer.render_chunk(512)

    assert float(np.sqrt(np.mean((first - baseline) ** 2))) < 2e-5
    assert float(np.max(np.abs(second))) > 0.002


def test_audio_engine_chunk_edge_commands_carry_into_the_next_chunk() -> None:
    edge_event = _audio_event(
        rpm=7200.0,
        target_rpm=7200.0,
        queue_depth=1,
        op_kind="data",
        servo_mode="seek",
        track_delta=0.25,
        motion_duration_ms=2.8,
        settle_duration_ms=1.8,
    )

    silent_engine = HDDAudioEngine(seed=0)
    silent_chunk = silent_engine.synthesizer.render_chunk(1024)

    engine = HDDAudioEngine(seed=0)
    first = engine.synthesizer.render_chunk(1024, scheduled_events=[(edge_event, 1022)])
    second = engine.synthesizer.render_chunk(1024)

    assert float(np.max(np.abs(first[:1000] - silent_chunk[:1000]))) < 1e-6
    assert float(np.max(np.abs(second - silent_chunk))) > 0.004


def test_audio_engine_active_commands_assume_a_running_drive_while_spinup_ramps() -> None:
    active_engine = HDDAudioEngine(seed=0)
    active_engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=240, op_kind="data")
    active_chunk = active_engine.render_chunk(4096)

    spinup_engine = HDDAudioEngine(seed=0)
    spinup_chunk = spinup_engine.synthesizer.render_chunk(
        4096,
        scheduled_events=[(_audio_event(rpm=1200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", is_spinup=True), 0)],
    )
    spinup_initial_rpm = spinup_engine.synthesizer.actual_rpm
    for _ in range(8):
        spinup_engine.synthesizer.render_chunk(4096)

    assert active_engine.synthesizer.actual_rpm > 7000.0
    assert 0.0 < spinup_initial_rpm < 1000.0
    assert spinup_engine.synthesizer.actual_rpm > spinup_initial_rpm * 4.0
    assert _transient_density(active_chunk) > _transient_density(spinup_chunk) * 2.0
    assert _band_energy(active_chunk, active_engine.fs, 120.0, 500.0) > _band_energy(spinup_chunk, spinup_engine.fs, 120.0, 500.0) * 4.0


def test_audio_engine_sequential_profiles_follow_the_drive_rpm() -> None:
    desktop = HDDAudioEngine(seed=0, drive_profile="desktop_7200_internal")
    desktop.emit_telemetry(7200.0, is_seq=True, queue_depth=2, op_kind="data")
    desktop_chunk = desktop.render_chunk(16384)

    archive = HDDAudioEngine(seed=0, drive_profile="archive_5900_internal")
    archive.emit_telemetry(float(archive.synthesizer.drive_profile.rpm), is_seq=True, queue_depth=2, op_kind="data")
    archive_chunk = archive.render_chunk(16384)

    desktop_peak = _dominant_freq(desktop_chunk, desktop.fs, 110.0, 130.0)
    archive_peak = _dominant_freq(archive_chunk, archive.fs, 90.0, 110.0)

    assert 118.0 <= desktop_peak <= 123.0
    assert 96.0 <= archive_peak <= 101.5


def test_audio_engine_mount_profiles_change_brightness_and_low_band_structure() -> None:
    _bare_rms, bare_centroid, bare_low_ratio, bare_desk_rms, bare_enclosure_rms = _structure_metrics("bare_drive_lab")
    case_rms, case_centroid, case_low_ratio, case_desk_rms, case_enclosure_rms = _structure_metrics("mounted_in_case")
    external_rms, external_centroid, _external_low_ratio, external_desk_rms, external_enclosure_rms = _structure_metrics("external_enclosure")
    desk_rms, _desk_centroid, desk_low_ratio, desk_desk_rms, _desk_enclosure_rms = _structure_metrics("drive_on_desk")

    assert bare_centroid > case_centroid > external_centroid
    assert desk_low_ratio > case_low_ratio > bare_low_ratio
    assert case_desk_rms > external_desk_rms > bare_desk_rms
    assert desk_desk_rms > external_desk_rms
    assert case_enclosure_rms > external_enclosure_rms > bare_enclosure_rms
    assert case_rms > desk_rms > external_rms


def test_audio_engine_metadata_storm_is_more_transient_than_sequential_stream() -> None:
    sequential = HDDAudioEngine(seed=0)
    sequential.emit_telemetry(7200.0, is_seq=True, queue_depth=2, op_kind="data")
    sequential_chunk = sequential.render_chunk(4096)
    metadata_chunk = _metadata_storm_chunk()
    sequential_mid_ratio = _band_energy(sequential_chunk, sequential.fs, 350.0, 1800.0) / max(
        _band_energy(sequential_chunk, sequential.fs, 2500.0, 7000.0),
        1e-12,
    )
    metadata_mid_ratio = _band_energy(metadata_chunk, 44100, 350.0, 1800.0) / max(
        _band_energy(metadata_chunk, 44100, 2500.0, 7000.0),
        1e-12,
    )

    assert _transient_density(metadata_chunk) > _transient_density(sequential_chunk) * 1.15
    assert _rms(metadata_chunk) > _rms(sequential_chunk) * 1.2
    assert metadata_mid_ratio > sequential_mid_ratio * 3.0


def test_audio_engine_park_contact_is_sharper_than_a_normal_seek() -> None:
    def transient_metrics(event: HDDAudioEvent) -> tuple[float, float]:
        engine = HDDAudioEngine(seed=0, acoustic_profile="drive_on_desk")
        diagnostics = engine.synthesizer.render_diagnostic_chunk(4096, scheduled_events=[(event, 0)])
        return _crest(diagnostics.output), _rms(diagnostics.structure_desk_velocity)

    seek_crest, seek_desk_rms = transient_metrics(
        _audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="data", servo_mode="seek", track_delta=0.24, motion_duration_ms=2.4, settle_duration_ms=1.6)
    )
    park_crest, park_desk_rms = transient_metrics(
        _audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", servo_mode="park")
    )

    assert park_crest > seek_crest * 1.03
    assert seek_desk_rms > park_desk_rms * 1.4


def test_audio_engine_seek_transients_are_structure_dominated_not_broadband_pops() -> None:
    engine = HDDAudioEngine(seed=0, acoustic_profile="drive_on_desk")
    diagnostics = engine.synthesizer.render_diagnostic_chunk(
        4096,
        scheduled_events=[
            (
                _audio_event(
                    rpm=7200.0,
                    target_rpm=7200.0,
                    queue_depth=1,
                    op_kind="data",
                    servo_mode="seek",
                    track_delta=0.24,
                    motion_duration_ms=2.4,
                    settle_duration_ms=1.6,
                ),
                0,
            )
        ],
    )
    mid_energy = _band_energy(diagnostics.output, engine.fs, 350.0, 1800.0)
    high_energy = _band_energy(diagnostics.output, engine.fs, 2500.0, 7000.0)

    assert mid_energy > high_energy * 2.0


def test_audio_engine_overlapping_events_render_together() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=120, op_kind="journal")
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=260, op_kind="flush", is_flush=True)
    engine.emit_telemetry(7200.0, is_cal=True, op_kind="metadata")
    engine.emit_telemetry(7200.0, is_park=True, op_kind="metadata")

    chunk = engine.render_chunk(4096)

    assert engine.events.pending_count() == 0
    assert _rms(chunk) > 0.0015


def test_audio_engine_honors_control_event_offsets_within_a_chunk() -> None:
    engine = HDDAudioEngine(seed=0)
    chunk = engine.synthesizer.render_chunk(
        4096,
        scheduled_events=[
            (_audio_event(rpm=0.0, target_rpm=0.0, queue_depth=1, op_kind="data"), 0),
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="data"), 2048),
        ],
    )

    first_half_rms = _rms(chunk[:2048])
    second_half_rms = _rms(chunk[2048:])

    assert first_half_rms < 0.0002
    assert second_half_rms > first_half_rms * 8.0


def test_audio_engine_can_export_offline_diagnostics_json(tmp_path: Path) -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=320, op_kind="data")

    output_path = tmp_path / "audio-diagnostics.json"
    diagnostics = engine.export_diagnostics_json(str(output_path), 2048, chunk_size=512)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sample_rate"] == engine.fs
    assert len(payload["time_s"]) == 2048
    assert len(payload["actual_rpm"]) == 2048
    assert len(payload["actuator_pos"]) == 2048
    assert len(payload["structure_cover_velocity"]) == 2048
    assert len(payload["structure_enclosure_velocity"]) == 2048
    assert len(payload["output"]) == 2048
    assert float(np.max(np.abs(diagnostics.output))) > 0.001


def test_audio_engine_render_is_chunk_size_invariant_for_same_event_stream() -> None:
    events = [
        (_audio_event(rpm=1200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", is_spinup=True), 0),
        (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="data", servo_mode="seek", track_delta=0.22, motion_duration_ms=2.2, settle_duration_ms=1.3), 900),
        (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="flush", servo_mode="seek", track_delta=0.10, is_flush=True, motion_duration_ms=2.0, settle_duration_ms=1.2), 2200),
        (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", servo_mode="park"), 3300),
    ]
    frames = 4096

    full_engine = HDDAudioEngine(seed=0)
    full_chunk = full_engine.synthesizer.render_chunk(frames, scheduled_events=events)

    split_engine = HDDAudioEngine(seed=0)
    rebuilt = np.concatenate(
        [
            split_engine.synthesizer.render_chunk(1024, scheduled_events=events),
            split_engine.synthesizer.render_chunk(1024),
            split_engine.synthesizer.render_chunk(1024),
            split_engine.synthesizer.render_chunk(1024),
        ]
    )

    delta = rebuilt - full_chunk
    assert _rms(delta) < 1e-9
    assert float(np.max(np.abs(delta))) < 1e-8


def test_audio_modal_decay_stays_stable_without_numeric_energy_growth() -> None:
    engine = HDDAudioEngine(seed=0)
    state = replace(engine.synthesizer.state)
    state.plant.cover_disp = np.array([0.0, 0.0, 0.0, 0.8, 0.6], dtype=np.float64)
    state.plant.cover_vel = np.zeros(5, dtype=np.float64)
    state.plant.actuator_disp = np.array([0.45, 0.35, 0.25], dtype=np.float64)
    state.plant.actuator_vel_modes = np.zeros(3, dtype=np.float64)

    next_state, chunk = render_audio_chunk(
        state,
        engine.synthesizer.mode_bank,
        engine.synthesizer.drive_profile,
        engine.synthesizer.acoustic_profile,
        8192,
        scheduled_events=[],
        bearing_noise_raw=np.zeros(8192, dtype=np.float64),
        windage_noise_raw=np.zeros(8192, dtype=np.float64),
    )

    first_quarter = _rms(chunk[:2048])
    last_quarter = _rms(chunk[-2048:])

    assert first_quarter > last_quarter * 1e6
    assert float(np.max(np.abs(chunk[-1024:]))) < float(np.max(np.abs(chunk[:1024])))
    assert float(np.max(np.abs(next_state.plant.cover_disp))) < 1e-12


def test_audio_engine_startup_only_has_real_delay_and_no_immediate_output() -> None:
    startup_chunk, diagnostics = _render_startup_only()
    assert diagnostics is not None
    features = compute_audio_features(startup_chunk, 22050, "desktop_7200_internal")
    time_to_90 = _startup_time_to_fraction(diagnostics, 7200.0, 0.90)

    assert _rms(startup_chunk[: int(0.5 * 22050)]) < 0.0002
    assert 0.75 <= float(features["first_audible_s"]) <= 1.55
    assert time_to_90 > 5.0
    assert time_to_90 < 10.5
    assert float(np.max(np.abs(diagnostics.actuator_torque))) < 0.03


def test_audio_engine_startup_only_is_less_transient_than_metadata_storm() -> None:
    startup_chunk, _diagnostics = _render_startup_only()
    metadata_chunk = _metadata_storm_chunk()
    startup_features = compute_audio_features(startup_chunk, 22050, "desktop_7200_internal")
    metadata_features = compute_audio_features(metadata_chunk, 44100, "desktop_7200_internal")

    assert _transient_density(startup_chunk) < _transient_density(metadata_chunk) * 0.80
    assert float(startup_features["spectral_centroid_hz"]) < float(metadata_features["spectral_centroid_hz"]) * 0.70


def test_audio_engine_startup_low_band_grows_during_runup() -> None:
    startup_chunk, _diagnostics = _render_startup_only()
    sample_rate = 22050
    early = _band_energy(startup_chunk[int(0.85 * sample_rate) : int(1.25 * sample_rate)], sample_rate, 20.0, 120.0)
    late = _band_energy(startup_chunk[int(8.0 * sample_rate) : int(12.0 * sample_rate)], sample_rate, 20.0, 120.0)
    assert late > early * 3.0


def test_audio_engine_startup_centroid_trajectory_is_smooth() -> None:
    startup_chunk, _diagnostics = _render_startup_only()
    features = compute_audio_features(startup_chunk, 22050, "desktop_7200_internal")
    centroid = np.asarray(features["centroid_curve"], dtype=np.float64)
    centroid_diff = np.abs(np.diff(centroid))

    assert float(np.quantile(centroid_diff, 0.95)) < 95.0


def test_audio_engine_startup_render_is_chunk_invariant() -> None:
    full_chunk, _diagnostics = _render_startup_only()
    split_chunk, _none = _render_startup_only(chunked=True)

    delta = split_chunk - full_chunk
    assert _rms(delta) < 1e-9
    assert float(np.max(np.abs(delta))) < 1e-8


def test_audio_engine_startup_only_matches_reference_summary_band() -> None:
    summary_path = Path("docs/reference-calibration/startup_reference_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    startup_chunk, diagnostics = _render_startup_only()
    assert diagnostics is not None
    features = compute_audio_features(startup_chunk, 22050, "desktop_7200_internal")
    ref_band = summary["reference_band"]
    time_to_90 = _startup_time_to_fraction(diagnostics, 7200.0, 0.90)

    assert ref_band["first_audible_s_min"] <= float(features["first_audible_s"]) <= ref_band["first_audible_s_max"]
    assert ref_band["time_to_90_s_min"] <= time_to_90 <= ref_band["time_to_90_s_max"]
    assert ref_band["spectral_centroid_hz_min"] * 0.35 <= float(features["spectral_centroid_hz"]) <= ref_band["spectral_centroid_hz_max"]
    assert float(features["low_band_ratio"]) >= ref_band["low_band_ratio_min"]
    assert float(features["bubbly_modulation_ratio"]) <= ref_band["bubbly_modulation_ratio_max"]
