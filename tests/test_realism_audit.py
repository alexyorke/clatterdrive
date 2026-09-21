from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from clatterdrive.audio import HDDAudioEvent
from clatterdrive.audio.engine import HDDAudioSynthesizer
from clatterdrive.audio import physics
from clatterdrive.audio.core import build_mode_bank
from clatterdrive.audio.commands import command_from_event
from clatterdrive.hdd import HDDLatencyModel
from clatterdrive.profiles import DRIVE_PROFILES, resolve_selected_profiles
from clatterdrive.storage_events import StorageEventRecorder


@pytest.mark.parametrize("sample_rate", [22050, 44100, 48000, 96000])
def test_filter_time_constant_is_independent_of_sample_rate(sample_rate: int) -> None:
    alpha = physics.resample_alpha(0.02, sample_rate)
    assert (1 - alpha) ** (sample_rate * 0.01) == pytest.approx(0.98 ** 441, rel=1e-10)


def test_modal_force_response_is_not_proportional_to_sample_rate() -> None:
    drive, acoustic = resolve_selected_profiles(None, None)
    amplitudes = []
    for sample_rate in (22050, 44100, 48000):
        bank = build_mode_bank(drive, sample_rate, acoustic).base
        displacement = np.zeros(bank.size)
        velocity = np.zeros(bank.size)
        samples = []
        for index in range(sample_rate // 5):
            displacement, velocity, value = physics.step_modal_bank(
                bank, displacement, velocity, math.sin(2 * math.pi * 120 * index / sample_rate),
            )
            samples.append(value)
        amplitudes.append(float(np.sqrt(np.mean(np.square(samples[len(samples) // 2:])))))
    assert max(amplitudes) / min(amplitudes) < 1.02


def test_stopped_rotor_has_no_periodic_excitation() -> None:
    assert physics.spindle_rotor_excitation(
        spindle_phase=1.3, harmonics=(1, 2), weights=np.array([1.0, 0.4]),
        phase_offsets=np.array([0.1, 0.2]), rpm_norm=0.0,
        startup_active=False, platter_gain=1.0,
    ) == 0.0


def test_parking_reaches_parked_state_and_emits_stop_reaction() -> None:
    synth = HDDAudioSynthesizer(seed=0)
    synth.apply_event(HDDAudioEvent(
        rpm=7200, emitted_at=0, power_state="active", heads_loaded=True,
        servo_mode="idle",
    ))
    trace = synth.render_diagnostic_chunk(4410, scheduled_events=[(HDDAudioEvent(
        rpm=7200, emitted_at=0, power_state="unloaded_idle", heads_loaded=False,
        servo_mode="park", motion_duration_ms=24, settle_duration_ms=10,
    ), 0)])
    assert synth.state.supervisor.load_state == "parked"
    assert synth.state.supervisor.servo_mode == "idle"
    assert np.max(np.abs(trace.contact_force[1200:])) > 0.01
    assert synth.actual_rpm == pytest.approx(7200)


def test_startup_telemetry_does_not_restart_current_envelope() -> None:
    synth = HDDAudioSynthesizer(seed=0)
    event = HDDAudioEvent(rpm=0, emitted_at=0, target_rpm=7200, is_spinup=True)
    synth.apply_event(event)
    synth.render_chunk(1000)
    elapsed = synth.state.supervisor.startup_elapsed_s
    synth.apply_event(event)
    assert synth.state.supervisor.startup_elapsed_s == elapsed


def test_cache_hits_do_not_emit_physical_transfers_and_seeks_have_direction() -> None:
    recorder = StorageEventRecorder()
    model = HDDLatencyModel(
        addressable_blocks=100000, latency_scale=0, event_sink=recorder,
        enable_background_scan=False, enable_retry_recovery=False,
    )
    try:
        model.submit_physical_access(90000, 4096, False)
        assert recorder.snapshot()[-1].track_delta > 0
        recorder.clear()
        result = model.submit_physical_access(90000, 4096, False)
        assert result.cache_hit
        assert recorder.snapshot() == []
        model.submit_physical_access(0, 4096, False)
        assert recorder.snapshot()[-1].track_delta < 0
        model.reset_caches()
        model._remember_read(0, 1)
        first = model.read_ahead_window_blocks
        model._remember_read(1, 1)
        assert model.read_ahead_window_blocks > first
    finally:
        model.stop()


def test_cancelled_transition_cannot_overwrite_new_power_state() -> None:
    model = HDDLatencyModel(addressable_blocks=1000, latency_scale=0, enable_background_scan=False)
    try:
        obsolete = threading.Event()
        model.transition_cancel = threading.Event()
        model.power_state = "starting"
        model._finish_transition(obsolete, "spindown", "standby")
        assert model.power_state == "starting"
    finally:
        model.stop()


def test_physical_commands_preserve_small_seeks_and_long_transfers() -> None:
    event = HDDAudioEvent(
        rpm=7200, emitted_at=0, power_state="active", heads_loaded=True,
        servo_mode="seek", track_delta=0.001, target_track=0.101,
        motion_duration_ms=0.5, settle_duration_ms=0.3, transfer_ms=1800,
        block_count=100000, fragmentation_score=10, directory_entry_count=1000,
    )
    command = command_from_event(event)
    assert command.transfer_duration_s == 1.8
    synth = HDDAudioSynthesizer()
    synth.apply_event(event)
    assert synth.state.plant.actuator_pos == pytest.approx(0.1)
    assert synth.state.supervisor.target_track == pytest.approx(0.101)
    assert synth.state.supervisor.seek_duration_s == 0.0005
    assert synth.state.supervisor.settle_remaining_s == 0.0003
    assert synth.state.supervisor.wedge_impulse == 0
    synth.apply_event(event)
    assert synth.state.supervisor.target_track == pytest.approx(0.101)
    assert synth.state.supervisor.repetition_pressure == 0


def test_sustained_transfer_rates_use_decimal_megabytes() -> None:
    model = HDDLatencyModel(addressable_blocks=100000, latency_scale=0, enable_background_scan=False)
    try:
        expected = 4096 / (210 * 1_000_000) * 1000
        assert model._transfer_ms_for_span(0, 1) == pytest.approx(expected)
        assert model.blocks_per_track_outer == round(210 * 1_000_000 / 120 / 4096)
    finally:
        model.stop()


def test_physical_transfer_waits_for_positioning_and_stops_at_completion() -> None:
    synth = HDDAudioSynthesizer()
    synth.apply_event(HDDAudioEvent(
        rpm=7200, emitted_at=0, power_state="active", heads_loaded=True,
        servo_mode="track", target_track=0.0, transfer_delay_ms=20,
        transfer_ms=5, is_sequential=True,
    ))
    synth.render_chunk(441)  # First 10 ms is positioning, not media transfer.
    assert synth.state.supervisor.transfer_activity == 0
    assert synth.state.supervisor.transfer_remaining_s == 0.005
    synth.render_chunk(600)
    assert synth.state.supervisor.transfer_activity == 1
    synth.render_chunk(500)
    assert synth.state.supervisor.transfer_activity == 0


def test_sata_profiles_respect_five_bit_ncq_tag_space() -> None:
    assert all(1 <= profile.ncq_depth <= 32 for profile in DRIVE_PROFILES.values())
