from __future__ import annotations

import json
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from clatterdrive.audio import HDDAudioEngine, HDDAudioEvent
from clatterdrive.audio.core import render_chunk as render_audio_chunk
from tests.helpers import _audio_event, _wav_metrics


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


def test_audio_engine_events_are_buffered_until_render() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=700, op_kind="data")

    assert engine.events.pending_count() == 1
    assert engine.synthesizer.rpm == 0.0

    chunk = engine.render_chunk(4096)

    assert engine.events.pending_count() == 0
    assert engine.synthesizer.rpm == 7200.0
    assert engine.synthesizer.actual_rpm > 6000.0
    assert float(np.max(np.abs(chunk))) > 0.004


def test_audio_engine_overlapping_events_can_render_in_one_chunk() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=180, op_kind="journal")
    engine.emit_telemetry(7200.0, is_cal=True, op_kind="metadata")
    engine.emit_telemetry(7200.0, is_park=True, op_kind="metadata")

    chunk = engine.render_chunk(4096)

    assert engine.events.pending_count() == 0
    assert float(np.sqrt(np.mean(chunk**2))) > 0.0013


def test_audio_synth_can_schedule_future_command_across_chunks() -> None:
    engine = HDDAudioEngine(seed=0)
    idle = engine.synthesizer.render_chunk(512)
    event = _audio_event(
        rpm=7200.0,
        target_rpm=7200.0,
        queue_depth=1,
        op_kind="data",
        servo_mode="seek",
        seek_distance=240,
        track_delta=0.20,
        motion_duration_ms=2.6,
        settle_duration_ms=1.6,
    )

    first = engine.synthesizer.render_chunk(512, scheduled_events=[(event, 700)])
    second = engine.synthesizer.render_chunk(512)

    assert float(np.sqrt(np.mean((first - idle) ** 2))) < 6e-5
    assert float(np.max(np.abs(second))) > 0.002


def test_audio_engine_chunk_edge_seek_carries_into_next_chunk() -> None:
    edge_event = _audio_event(
        rpm=7200.0,
        target_rpm=7200.0,
        queue_depth=1,
        op_kind="data",
        servo_mode="seek",
        seek_distance=240,
        track_delta=0.25,
        motion_duration_ms=2.8,
        settle_duration_ms=1.8,
    )

    silent_engine = HDDAudioEngine(seed=0)
    silent_chunk = silent_engine.synthesizer.render_chunk(1024)

    engine = HDDAudioEngine(seed=0)
    first = engine.synthesizer.render_chunk(1024, scheduled_events=[(edge_event, 1022)])
    assert float(np.max(np.abs(first[:1000] - silent_chunk[:1000]))) < 1e-6
    assert float(np.max(np.abs(first[1000:] - silent_chunk[1000:]))) > 1e-4

    second = engine.synthesizer.render_chunk(1024)
    second_delta = second - silent_chunk
    assert float(np.max(np.abs(second_delta))) > 0.0045
    assert float(np.sqrt(np.mean(second_delta**2))) > 0.0015


def test_audio_engine_event_bus_drains_fifo() -> None:
    engine = HDDAudioEngine(seed=0)
    first = _audio_event(rpm=5400.0, queue_depth=1, op_kind="metadata", servo_mode="calibration")
    second = _audio_event(rpm=7200.0, queue_depth=4, op_kind="data", servo_mode="seek", seek_distance=500)

    engine.events.publish(first)
    engine.events.publish(second)
    drained = engine.events.drain()

    assert drained == [first, second]


def test_audio_engine_event_bus_is_bounded_and_keeps_recent_events() -> None:
    engine = HDDAudioEngine(seed=0, max_pending_events=3)
    for idx in range(5):
        engine.events.publish(_audio_event(rpm=5400.0 + idx, queue_depth=idx + 1, op_kind="metadata"))

    drained = engine.events.drain()

    assert [event.queue_depth for event in drained] == [3, 4, 5]
    assert engine.events.dropped_count() == 2


def test_audio_engine_spinup_event_ramps_actual_rpm_instead_of_jumping() -> None:
    engine = HDDAudioEngine(seed=0)
    spinup = _audio_event(
        rpm=1200.0,
        target_rpm=7200.0,
        queue_depth=1,
        op_kind="metadata",
        is_spinup=True,
        servo_mode="idle",
    )

    engine.synthesizer.render_chunk(4096, scheduled_events=[(spinup, 0)])

    assert 0.0 < engine.synthesizer.actual_rpm < 7200.0


def test_audio_engine_spinup_command_uses_target_rpm_not_emitted_rpm_trace() -> None:
    engine = HDDAudioEngine(seed=0)
    spinup = _audio_event(
        rpm=900.0,
        target_rpm=7200.0,
        queue_depth=1,
        op_kind="metadata",
        is_spinup=True,
        servo_mode="idle",
    )

    diagnostics = engine.synthesizer.render_diagnostic_chunk(4096, scheduled_events=[(spinup, 0)])

    assert float(diagnostics.target_rpm[0]) == 7200.0
    assert float(diagnostics.actual_rpm[-1]) > float(diagnostics.actual_rpm[0])
    assert float(np.max(diagnostics.actual_rpm)) < 7200.0


def test_audio_engine_seek_controller_leaves_seek_mode_after_settle() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=900, op_kind="data")
    engine.render_chunk(4096)
    for _ in range(6):
        engine.render_chunk(4096)

    assert engine.synthesizer.state.servo_mode in {"track", "idle"}


def test_audio_engine_seek_profile_keeps_fixed_actuator_mode_band() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=700, op_kind="data")
    chunk = engine.render_chunk(4096)

    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
    resonance_band = (freqs >= 900.0) & (freqs <= 2200.0)
    resonance_freq = float(freqs[np.argmax(spectrum[resonance_band]) + np.where(resonance_band)[0][0]])

    assert 1050.0 <= resonance_freq <= 1700.0


def test_audio_engine_sequential_profile_keeps_7200rpm_spindle_fundamental() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, is_seq=True, queue_depth=2, op_kind="data")
    chunk = engine.render_chunk(16384)

    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
    low_band = (freqs >= 110.0) & (freqs <= 130.0)
    peak_freq = float(freqs[np.argmax(spectrum[low_band]) + np.where(low_band)[0][0]])

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spectrum[mask]))

    platter_band_ratio = band(550.0, 1100.0) / max(band(0.0, 300.0), 1e-12)
    upper_mid_ratio = band(1000.0, 3000.0) / max(band(0.0, 300.0), 1e-12)

    assert 110.0 <= peak_freq <= 122.5
    assert platter_band_ratio > 0.01
    assert upper_mid_ratio > 0.02


def test_audio_engine_archive_profile_shifts_spindle_fundamental_lower() -> None:
    engine = HDDAudioEngine(seed=0, drive_profile="archive_5900_internal")
    rpm = float(engine.synthesizer.drive_profile.rpm)
    engine.emit_telemetry(rpm, is_seq=True, queue_depth=2, op_kind="data")
    chunk = engine.render_chunk(16384)

    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
    low_band = (freqs >= 90.0) & (freqs <= 110.0)
    peak_freq = float(freqs[np.argmax(spectrum[low_band]) + np.where(low_band)[0][0]])

    assert 91.0 <= peak_freq <= 101.0


def test_audio_engine_seek_resonance_stays_stable_across_seek_distance() -> None:
    def resonance_peak(seek_distance: int) -> float:
        engine = HDDAudioEngine(seed=0)
        engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=seek_distance, op_kind="data")
        chunk = engine.render_chunk(4096)
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
        resonance_band = (freqs >= 900.0) & (freqs <= 2200.0)
        return float(freqs[np.argmax(spectrum[resonance_band]) + np.where(resonance_band)[0][0]])

    short_seek_peak = resonance_peak(40)
    long_seek_peak = resonance_peak(700)

    assert abs(short_seek_peak - long_seek_peak) <= 450.0


def test_audio_engine_multi_event_render_has_higher_delta_complexity_than_single_event() -> None:
    idle_engine = HDDAudioEngine(seed=0)
    idle_chunk = idle_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="data"), 0)],
    )

    single_engine = HDDAudioEngine(seed=0)
    single_chunk = single_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[
            (
                _audio_event(
                    rpm=7200.0,
                    target_rpm=7200.0,
                    queue_depth=1,
                    op_kind="data",
                    servo_mode="seek",
                    seek_distance=220,
                    track_delta=0.22,
                ),
                0,
            )
        ],
    )
    single_delta = single_chunk - idle_chunk

    multi_engine = HDDAudioEngine(seed=0)
    multi_chunk = multi_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="data", servo_mode="seek", seek_distance=220, track_delta=0.22), 0),
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="journal", servo_mode="seek", seek_distance=80, track_delta=0.10), 50),
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", servo_mode="calibration"), 90),
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", servo_mode="park"), 140),
        ],
    )
    multi_delta = multi_chunk - idle_chunk

    single_complexity = float(np.sum(np.abs(np.diff(single_delta))))
    multi_complexity = float(np.sum(np.abs(np.diff(multi_delta))))

    assert multi_complexity > single_complexity * 0.55


def test_audio_engine_render_regression_for_startup_idle_park_and_flush_envelopes() -> None:
    idle_engine = HDDAudioEngine(seed=0)
    idle_chunk = idle_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="data"), 0)],
    )

    startup_engine = HDDAudioEngine(seed=0)
    startup_chunk = startup_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(_audio_event(rpm=1200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", is_spinup=True), 0)],
    )
    park_engine = HDDAudioEngine(seed=0)
    park_chunk = park_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="metadata", servo_mode="park"), 0)],
    )
    flush_engine = HDDAudioEngine(seed=0)
    flush_chunk = flush_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=3, op_kind="flush", servo_mode="seek", seek_distance=260, track_delta=0.26, is_flush=True), 0)],
    )

    park_delta = park_chunk - idle_chunk
    flush_delta = flush_chunk - idle_chunk

    startup_rms = float(np.sqrt(np.mean(startup_chunk**2)))
    idle_rms = float(np.sqrt(np.mean(idle_chunk**2)))
    park_delta_rms = float(np.sqrt(np.mean(park_delta**2)))
    flush_delta_rms = float(np.sqrt(np.mean(flush_delta**2)))

    assert idle_rms > startup_rms * 2.5
    assert park_delta_rms > 0.0
    assert flush_delta_rms > 0.0
    assert flush_delta_rms > park_delta_rms * 0.85


def test_audio_engine_acoustic_profiles_change_loudness_and_brightness() -> None:
    def render_metrics(acoustic_profile: str) -> tuple[float, float, float]:
        engine = HDDAudioEngine(
            seed=0,
            drive_profile="desktop_7200_internal",
            acoustic_profile=acoustic_profile,
        )
        engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=320, op_kind="data")
        chunk = engine.render_chunk(4096)
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
        centroid = float(np.sum(freqs * spectrum) / max(np.sum(spectrum), 1e-12))
        rms = float(np.sqrt(np.mean(chunk**2)))
        low_band = float(np.sum(spectrum[(freqs >= 20.0) & (freqs < 120.0)]))
        lowmid_band = float(np.sum(spectrum[(freqs >= 120.0) & (freqs < 500.0)]))
        return centroid, rms, low_band / max(lowmid_band, 1e-12)

    bare_centroid, bare_rms, bare_low_ratio = render_metrics("bare_drive_lab")
    case_centroid, case_rms, case_low_ratio = render_metrics("mounted_in_case")
    _desk_centroid, _desk_rms, desk_low_ratio = render_metrics("drive_on_desk")
    external_centroid, external_rms, _external_low_ratio = render_metrics("external_enclosure")

    assert bare_centroid > case_centroid > external_centroid
    assert bare_rms > case_rms > external_rms
    assert desk_low_ratio > case_low_ratio
    assert bare_low_ratio > 0.25


def test_audio_engine_enclosure_and_table_states_follow_mount_profile() -> None:
    def structure_metrics(acoustic_profile: str) -> tuple[float, float]:
        engine = HDDAudioEngine(
            seed=0,
            drive_profile="desktop_7200_internal",
            acoustic_profile=acoustic_profile,
        )
        engine.emit_telemetry(7200.0, is_seq=True, op_kind="data")
        engine.render_chunk(2048)
        _, diagnostics = engine.render_chunk_with_diagnostics(4096)
        enclosure_rms = float(np.sqrt(np.mean(diagnostics.structure_enclosure_velocity**2)))
        desk_rms = float(np.sqrt(np.mean(diagnostics.structure_desk_velocity**2)))
        return enclosure_rms, desk_rms

    bare_enclosure_rms, bare_desk_rms = structure_metrics("bare_drive_lab")
    case_enclosure_rms, case_desk_rms = structure_metrics("mounted_in_case")
    desk_enclosure_rms, desk_desk_rms = structure_metrics("drive_on_desk")

    assert case_enclosure_rms > bare_enclosure_rms * 2.0
    assert case_desk_rms > bare_desk_rms * 1.2
    assert desk_desk_rms > case_desk_rms * 4.0
    assert desk_enclosure_rms > bare_enclosure_rms * 4.0


def test_audio_engine_seek_ticks_prefer_structure_borne_path_on_desk_mount() -> None:
    def seek_structure_metrics(acoustic_profile: str) -> tuple[float, float]:
        engine = HDDAudioEngine(
            seed=0,
            drive_profile="desktop_7200_internal",
            acoustic_profile=acoustic_profile,
        )
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
                        seek_distance=260,
                        track_delta=0.24,
                    ),
                    0,
                )
            ],
        )
        desk_rms = float(np.sqrt(np.mean(diagnostics.structure_desk_velocity**2)))
        output_rms = float(np.sqrt(np.mean(diagnostics.output**2)))
        return desk_rms, desk_rms / max(output_rms, 1e-9)

    bare_desk_rms, bare_structure_ratio = seek_structure_metrics("bare_drive_lab")
    desk_desk_rms, desk_structure_ratio = seek_structure_metrics("drive_on_desk")

    assert desk_desk_rms > bare_desk_rms * 4.0
    assert desk_structure_ratio > bare_structure_ratio * 4.0


def test_audio_engine_park_contact_tick_is_sharper_than_normal_seek_tick() -> None:
    def transient_metrics(event: HDDAudioEvent) -> tuple[float, float]:
        engine = HDDAudioEngine(
            seed=0,
            drive_profile="desktop_7200_internal",
            acoustic_profile="drive_on_desk",
        )
        diagnostics = engine.synthesizer.render_diagnostic_chunk(
            4096,
            scheduled_events=[(event, 0)],
        )
        rms = float(np.sqrt(np.mean(diagnostics.output**2)))
        crest = float(np.max(np.abs(diagnostics.output))) / max(rms, 1e-9)
        base_peak = float(np.max(np.abs(diagnostics.structure_base_velocity)))
        return crest, base_peak

    seek_crest, seek_base_peak = transient_metrics(
        _audio_event(
            rpm=7200.0,
            target_rpm=7200.0,
            queue_depth=1,
            op_kind="data",
            servo_mode="seek",
            seek_distance=260,
            track_delta=0.24,
        )
    )
    park_crest, park_base_peak = transient_metrics(
        _audio_event(
            rpm=7200.0,
            target_rpm=7200.0,
            queue_depth=1,
            op_kind="metadata",
            servo_mode="park",
        )
    )

    assert park_crest > seek_crest * 1.07
    assert park_base_peak > seek_base_peak * 0.35


def test_audio_engine_overlapping_seek_flush_park_and_calibration_events_render_together() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=120, op_kind="journal")
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=260, op_kind="flush", is_flush=True)
    engine.emit_telemetry(7200.0, is_cal=True, op_kind="metadata")
    engine.emit_telemetry(7200.0, is_park=True, op_kind="metadata")

    chunk = engine.render_chunk(4096)

    assert engine.events.pending_count() == 0
    assert float(np.sqrt(np.mean(chunk**2))) > 0.0013


def test_audio_engine_honors_control_event_offsets_within_a_chunk() -> None:
    engine = HDDAudioEngine(seed=0)
    chunk = engine.synthesizer.render_chunk(
        4096,
        scheduled_events=[
            (_audio_event(rpm=0.0, target_rpm=0.0, queue_depth=1, op_kind="data"), 0),
            (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="data"), 2048),
        ],
    )

    first_half_rms = float(np.sqrt(np.mean(chunk[:2048] ** 2)))
    second_half_rms = float(np.sqrt(np.mean(chunk[2048:] ** 2)))

    assert first_half_rms < 0.0002
    assert second_half_rms > first_half_rms * 20.0


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
        (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="data", servo_mode="seek", seek_distance=220, track_delta=0.22), 900),
        (_audio_event(rpm=7200.0, target_rpm=7200.0, queue_depth=1, op_kind="flush", servo_mode="seek", seek_distance=120, track_delta=0.10, is_flush=True), 2200),
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
    assert float(np.sqrt(np.mean(delta**2))) < 1e-5
    assert float(np.max(np.abs(delta))) < 2e-4


def test_audio_modal_decay_stays_stable_without_numeric_energy_growth() -> None:
    engine = HDDAudioEngine(seed=0)
    state = replace(
        engine.synthesizer.state,
        cover_disp=np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.6, 0.4], dtype=np.float64),
        cover_vel=np.zeros(7, dtype=np.float64),
        actuator_disp=np.array([0.45, 0.35, 0.25, 0.15, 0.1], dtype=np.float64),
        actuator_vel_modes=np.zeros(5, dtype=np.float64),
    )

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

    first_quarter = float(np.sqrt(np.mean(chunk[:2048] ** 2)))
    last_quarter = float(np.sqrt(np.mean(chunk[-2048:] ** 2)))

    assert first_quarter > last_quarter * 4.0
    assert float(np.max(np.abs(chunk[-1024:]))) < float(np.max(np.abs(chunk[:1024])))
    assert float(np.max(np.abs(next_state.cover_disp))) < float(np.max(np.abs(state.cover_disp)))
