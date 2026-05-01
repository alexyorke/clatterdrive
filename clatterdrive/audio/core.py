from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace

import numpy as np
import numpy.typing as npt

from .commands import AudioCommand, command_from_event
from . import physics
from ..profiles import AcousticProfile, DriveProfile
from ..storage_events import StorageEvent


FloatArray = npt.NDArray[np.float64]
ScheduledEvent = tuple[StorageEvent, int]
TAU = physics.TAU
EPS = physics.EPS


@dataclass(frozen=True)
class DiscreteModalBank:
    coeff_xx: FloatArray
    coeff_xv: FloatArray
    coeff_vx: FloatArray
    coeff_vv: FloatArray
    input_gain: FloatArray
    output_gain: FloatArray

    @property
    def size(self) -> int:
        return int(self.output_gain.size)


@dataclass(frozen=True)
class AudioModeBank:
    base: DiscreteModalBank
    cover: DiscreteModalBank
    actuator: DiscreteModalBank
    enclosure: DiscreteModalBank
    desk: DiscreteModalBank
    spindle_harmonics: tuple[int, ...]
    spindle_weights: FloatArray
    direct_gain: float
    platter_gain: float
    cover_gain: float
    actuator_gain: float
    structure_gain: float
    enclosure_gain: float
    desk_gain: float
    final_lowpass_alpha: float
    final_highpass_alpha: float


@dataclass
class PlantState:
    spindle_phase: float = 0.0
    spindle_omega: float = 0.0
    motor_drive: float = 0.0
    actuator_pos: float = 0.52
    actuator_vel: float = 0.0
    actuator_torque: float = 0.0
    servo_integrator: float = 0.0
    servo_wedge_timer_s: float = 0.0
    boundary_timer_s: float = 0.0
    windage_low_state: float = 0.0
    windage_high_state: float = 0.0
    bearing_state: float = 0.0
    wedge_fast_state: float = 0.0
    wedge_slow_state: float = 0.0
    contact_fast_state: float = 0.0
    contact_slow_state: float = 0.0
    base_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    base_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    cover_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    cover_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    actuator_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    actuator_vel_modes: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    enclosure_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    enclosure_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    desk_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    desk_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


@dataclass
class SupervisorState:
    target_rpm: float = 0.0
    power_state: str = "standby"
    startup_elapsed_s: float = 0.0
    servo_mode: str = "idle"
    load_state: str = "parked"
    heads_loaded: bool = False
    is_sequential: bool = False
    is_flush: bool = False
    queue_depth: int = 1
    op_kind: str = "data"
    transfer_activity: float = 0.0
    transfer_remaining_s: float = 0.0
    directory_activity: float = 0.0
    fragmentation_activity: float = 0.0
    target_track: float = 0.52
    seek_origin: float = 0.52
    seek_duration_s: float = 0.0
    seek_elapsed_s: float = 0.0
    settle_remaining_s: float = 0.0
    wedge_impulse: float = 0.0
    contact_impulse: float = 0.0
    maintenance_activity: float = 0.0
    retry_activity: float = 0.0
    last_event_signature: int = 0
    last_event_emitted_at: float = -1.0
    repetition_pressure: float = 0.0
    repetition_variant: float = 0.0


@dataclass
class AudioRenderState:
    fs: int
    sample_clock: int = 0
    plant: PlantState = field(default_factory=PlantState)
    supervisor: SupervisorState = field(default_factory=SupervisorState)
    output_lowpass_state: float = 0.0
    output_highpass_state: float = 0.0
    output_highpass_prev_input: float = 0.0

    @property
    def target_rpm(self) -> float:
        return self.supervisor.target_rpm

    @property
    def spindle_omega(self) -> float:
        return self.plant.spindle_omega

    @property
    def servo_mode(self) -> str:
        return self.supervisor.servo_mode


@dataclass
class AudioDiagnosticTrace:
    time_s: FloatArray
    target_rpm: FloatArray
    actual_rpm: FloatArray
    actuator_pos: FloatArray
    actuator_torque: FloatArray
    source_base_force: FloatArray
    source_cover_force: FloatArray
    source_actuator_force: FloatArray
    source_enclosure_force: FloatArray
    source_desk_force: FloatArray
    acoustic_airborne: FloatArray
    acoustic_structure: FloatArray
    windage: FloatArray
    bearing: FloatArray
    spindle_tone: FloatArray
    wedge_force: FloatArray
    contact_force: FloatArray
    structure_base_velocity: FloatArray
    structure_cover_velocity: FloatArray
    structure_enclosure_velocity: FloatArray
    structure_desk_velocity: FloatArray
    output: FloatArray


@dataclass
class RenderBlockResult:
    state: AudioRenderState
    samples: FloatArray
    diagnostics: AudioDiagnosticTrace


def _clamp(value: float, lo: float, hi: float) -> float:
    return physics.clamp(value, lo, hi)


def _one_pole_alpha(cutoff_hz: float, sample_rate: int) -> float:
    return physics.one_pole_alpha(cutoff_hz, sample_rate)


def _configure_modes(
    definitions: tuple[tuple[float, float, float], ...],
    *,
    sample_rate: int,
    freq_scale: float,
    gain_scale: float,
    input_scale: float,
) -> DiscreteModalBank:
    frequencies = np.asarray([definition[0] * freq_scale for definition in definitions], dtype=np.float64)
    damping = np.asarray([definition[1] for definition in definitions], dtype=np.float64)
    output_gain = np.asarray([definition[2] * gain_scale for definition in definitions], dtype=np.float64)
    wn = TAU * frequencies
    decay = np.exp(-damping * wn / sample_rate)
    wd = wn * np.sqrt(np.maximum(1.0 - damping**2, 1e-6))
    theta = wd / sample_rate
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    coeff_xx = decay * (cos_theta + (damping * wn / np.maximum(wd, EPS)) * sin_theta)
    coeff_xv = decay * (sin_theta / np.maximum(wd, EPS))
    coeff_vx = -decay * ((wn**2 / np.maximum(wd, EPS)) * sin_theta)
    coeff_vv = decay * (cos_theta - (damping * wn / np.maximum(wd, EPS)) * sin_theta)
    input_gain = 0.00135 * input_scale * np.maximum(output_gain, 0.12)
    return DiscreteModalBank(
        coeff_xx=coeff_xx,
        coeff_xv=coeff_xv,
        coeff_vx=coeff_vx,
        coeff_vv=coeff_vv,
        input_gain=input_gain,
        output_gain=output_gain,
    )


def build_mode_bank(
    drive_profile: DriveProfile,
    sample_rate: int,
    acoustic_profile: AcousticProfile,
) -> AudioModeBank:
    base_modes = (
        (72.0, 0.095, 0.92),
        (118.0, 0.082, 0.74),
        (168.0, 0.070, 0.56),
        (248.0, 0.060, 0.34),
    )
    cover_modes = (
        (212.0, 0.055, 0.44),
        (412.0, 0.048, 0.28),
        (576.0, 0.042, 0.24),
        (822.0, 0.038, 0.18),
        (1208.0, 0.034, 0.12),
    )
    actuator_modes = (
        (980.0, 0.036, 0.40),
        (1325.0, 0.032, 0.56),
        (1680.0, 0.028, 0.34),
    )
    enclosure_modes = (
        (58.0, 0.120, 0.54),
        (96.0, 0.104, 0.60),
        (146.0, 0.088, 0.42),
        (220.0, 0.074, 0.22),
    )
    desk_modes = (
        (44.0, 0.135, 0.72),
        (82.0, 0.110, 0.62),
        (142.0, 0.094, 0.36),
    )
    spindle_weights = np.asarray(drive_profile.spindle_harmonic_weights, dtype=np.float64)
    spindle_weights /= max(float(np.sum(spindle_weights)), EPS)
    base_gain_scale = acoustic_profile.structure_gain * 0.62
    cover_gain_scale = acoustic_profile.cover_gain * acoustic_profile.enclosure_coupling
    enclosure_gain_scale = acoustic_profile.structure_gain * acoustic_profile.enclosure_radiation_gain
    desk_gain_scale = acoustic_profile.structure_gain * acoustic_profile.table_radiation_gain
    actuator_gain_scale = acoustic_profile.actuator_gain * drive_profile.actuator_gain_scale
    return AudioModeBank(
        base=_configure_modes(
            base_modes,
            sample_rate=sample_rate,
            freq_scale=0.96 + 0.05 * drive_profile.cover_frequency_scale,
            gain_scale=base_gain_scale,
            input_scale=1.5,
        ),
        cover=_configure_modes(
            cover_modes,
            sample_rate=sample_rate,
            freq_scale=drive_profile.cover_frequency_scale,
            gain_scale=cover_gain_scale,
            input_scale=0.88,
        ),
        actuator=_configure_modes(
            actuator_modes,
            sample_rate=sample_rate,
            freq_scale=drive_profile.actuator_frequency_scale,
            gain_scale=actuator_gain_scale,
            input_scale=1.9,
        ),
        enclosure=_configure_modes(
            enclosure_modes,
            sample_rate=sample_rate,
            freq_scale=acoustic_profile.enclosure_resonance_scale,
            gain_scale=enclosure_gain_scale,
            input_scale=1.12,
        ),
        desk=_configure_modes(
            desk_modes,
            sample_rate=sample_rate,
            freq_scale=acoustic_profile.table_resonance_scale,
            gain_scale=desk_gain_scale,
            input_scale=1.5,
        ),
        spindle_harmonics=drive_profile.spindle_harmonics,
        spindle_weights=spindle_weights,
        direct_gain=acoustic_profile.direct_gain * acoustic_profile.output_gain,
        platter_gain=acoustic_profile.platter_gain,
        cover_gain=acoustic_profile.cover_gain,
        actuator_gain=acoustic_profile.actuator_gain,
        structure_gain=acoustic_profile.structure_gain,
        enclosure_gain=acoustic_profile.enclosure_radiation_gain,
        desk_gain=acoustic_profile.table_radiation_gain,
        final_lowpass_alpha=_one_pole_alpha(acoustic_profile.final_lowpass_hz, sample_rate),
        final_highpass_alpha=_one_pole_alpha(acoustic_profile.final_highpass_hz, sample_rate),
    )


def _zeros(size: int) -> FloatArray:
    return np.zeros(size, dtype=np.float64)


def initialize_render_state(
    sample_rate: int,
    mode_bank: AudioModeBank,
    acoustic_profile: AcousticProfile,
) -> AudioRenderState:
    del acoustic_profile
    return AudioRenderState(
        fs=sample_rate,
        plant=PlantState(
            base_disp=_zeros(mode_bank.base.size),
            base_vel=_zeros(mode_bank.base.size),
            cover_disp=_zeros(mode_bank.cover.size),
            cover_vel=_zeros(mode_bank.cover.size),
            actuator_disp=_zeros(mode_bank.actuator.size),
            actuator_vel_modes=_zeros(mode_bank.actuator.size),
            enclosure_disp=_zeros(mode_bank.enclosure.size),
            enclosure_vel=_zeros(mode_bank.enclosure.size),
            desk_disp=_zeros(mode_bank.desk.size),
            desk_vel=_zeros(mode_bank.desk.size),
        ),
    )


def reinitialize_mode_state(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    acoustic_profile: AcousticProfile,
) -> AudioRenderState:
    fresh = initialize_render_state(state.fs, mode_bank, acoustic_profile)
    fresh.sample_clock = state.sample_clock
    fresh.output_lowpass_state = state.output_lowpass_state
    fresh.output_highpass_state = state.output_highpass_state
    fresh.output_highpass_prev_input = state.output_highpass_prev_input
    fresh.plant.spindle_phase = state.plant.spindle_phase
    fresh.plant.spindle_omega = state.plant.spindle_omega
    fresh.plant.motor_drive = state.plant.motor_drive
    fresh.plant.actuator_pos = state.plant.actuator_pos
    fresh.plant.actuator_vel = state.plant.actuator_vel
    fresh.plant.actuator_torque = state.plant.actuator_torque
    fresh.plant.servo_integrator = state.plant.servo_integrator
    fresh.plant.servo_wedge_timer_s = state.plant.servo_wedge_timer_s
    fresh.plant.boundary_timer_s = state.plant.boundary_timer_s
    fresh.plant.windage_low_state = state.plant.windage_low_state
    fresh.plant.windage_high_state = state.plant.windage_high_state
    fresh.plant.bearing_state = state.plant.bearing_state
    fresh.plant.wedge_fast_state = state.plant.wedge_fast_state
    fresh.plant.wedge_slow_state = state.plant.wedge_slow_state
    fresh.plant.contact_fast_state = state.plant.contact_fast_state
    fresh.plant.contact_slow_state = state.plant.contact_slow_state
    fresh.supervisor = replace(state.supervisor)
    return fresh


def _empty_trace() -> AudioDiagnosticTrace:
    empty = np.zeros(0, dtype=np.float64)
    return AudioDiagnosticTrace(**{trace_field.name: empty for trace_field in fields(AudioDiagnosticTrace)})


def concatenate_diagnostic_traces(traces: Sequence[AudioDiagnosticTrace]) -> AudioDiagnosticTrace:
    if not traces:
        return _empty_trace()
    return AudioDiagnosticTrace(
        **{
            trace_field.name: np.concatenate([getattr(trace, trace_field.name) for trace in traces])
            for trace_field in fields(AudioDiagnosticTrace)
        }
    )


def _command_frequency(drive_profile: DriveProfile, supervisor: SupervisorState) -> float:
    if supervisor.servo_mode in {"seek", "settle"}:
        return 92.0
    if supervisor.servo_mode == "park":
        return 84.0
    if supervisor.is_sequential and supervisor.transfer_activity > 0.2:
        return 76.0
    return 44.0


def _startup_active(
    plant: PlantState,
    supervisor: SupervisorState,
    target_omega: float,
) -> bool:
    if target_omega <= EPS:
        return False
    if supervisor.power_state == "starting":
        return True
    return (
        not supervisor.heads_loaded
        and supervisor.servo_mode == "idle"
        and plant.spindle_omega < target_omega * 0.992
    )


def _apply_command(
    state: AudioRenderState,
    command: AudioCommand,
    drive_profile: DriveProfile,
) -> AudioRenderState:
    plant = state.plant
    supervisor = state.supervisor
    previous_heads_loaded = supervisor.heads_loaded
    previous_track = supervisor.target_track
    emitted_gap_s = (
        command.emitted_at - supervisor.last_event_emitted_at
        if supervisor.last_event_emitted_at >= 0.0
        else 999.0
    )
    repeated_pattern = (
        command.event_signature == supervisor.last_event_signature
        and 0.0 <= emitted_gap_s <= 0.45
    )
    if repeated_pattern:
        supervisor.repetition_pressure = min(1.0, supervisor.repetition_pressure * 0.60 + 0.34)
        variant_cycle = (-0.85, -0.20, 0.45, 0.95)
        cycle_index = int(supervisor.repetition_pressure * 4.0 + abs(previous_track) * 17.0)
        supervisor.repetition_variant = variant_cycle[cycle_index % len(variant_cycle)]
    else:
        supervisor.repetition_pressure *= 0.28
        supervisor.repetition_variant *= 0.35
    supervisor.last_event_signature = command.event_signature
    supervisor.last_event_emitted_at = command.emitted_at
    supervisor.target_rpm = max(float(command.target_rpm), 0.0)
    supervisor.power_state = command.power_state
    supervisor.queue_depth = max(1, int(command.queue_depth))
    supervisor.op_kind = command.op_kind
    supervisor.transfer_activity = float(command.transfer_activity) * (
        1.0 + 0.09 * supervisor.repetition_pressure * supervisor.repetition_variant
    )
    supervisor.transfer_remaining_s = max(supervisor.transfer_remaining_s, command.transfer_duration_s)
    supervisor.directory_activity = (
        min(1.0, math.log2(command.directory_entry_count + 1) / 11.0)
        if command.directory_entry_count > 0
        else 0.0
    )
    supervisor.fragmentation_activity = (
        min(1.0, math.log2(command.fragmentation_score + 1) / 4.0)
        if command.fragmentation_score > 1
        else 0.0
    )
    supervisor.is_sequential = bool(command.is_sequential)
    supervisor.is_flush = bool(command.is_flush)
    supervisor.maintenance_activity = 0.55 if command.maintenance else 0.0
    supervisor.retry_activity = 0.70 if command.retry else 0.0
    supervisor.heads_loaded = bool(command.heads_loaded)
    supervisor.load_state = "loaded" if supervisor.heads_loaded else "parked"
    if not previous_heads_loaded and supervisor.heads_loaded:
        supervisor.contact_impulse += physics.head_load_contact_force(previous_heads_loaded, supervisor.heads_loaded)
        supervisor.load_state = "loading"
    if command.is_spinup:
        supervisor.power_state = "starting"
        supervisor.startup_elapsed_s = 0.0
        supervisor.heads_loaded = False
        supervisor.load_state = "parked"
    elif supervisor.power_state == "active" and supervisor.target_rpm > 0.0:
        target_omega = supervisor.target_rpm * TAU / 60.0
        plant.spindle_omega = max(plant.spindle_omega, target_omega)
    if supervisor.power_state == "standby":
        supervisor.target_rpm = 0.0
        supervisor.heads_loaded = False
        supervisor.load_state = "parked"

    servo_mode = command.servo_mode or "idle"
    supervisor.servo_mode = servo_mode
    media_event_forces = physics.head_media_event_forces(
        op_kind=command.op_kind,
        directory_activity=supervisor.directory_activity,
        fragmentation_activity=supervisor.fragmentation_activity,
        repetition_pressure=supervisor.repetition_pressure,
        repetition_variant=supervisor.repetition_variant,
    )
    supervisor.wedge_impulse += media_event_forces.wedge
    supervisor.contact_impulse += media_event_forces.contact

    if servo_mode == "park":
        supervisor.seek_origin = plant.actuator_pos
        supervisor.target_track = 0.04
        supervisor.seek_duration_s = max(command.motion_duration_s, 0.024)
        supervisor.seek_elapsed_s = 0.0
        supervisor.settle_remaining_s = max(command.settle_duration_s, 0.010)
        supervisor.wedge_impulse += physics.actuator_latch_event_force(servo_mode)
        supervisor.load_state = "parking"
        supervisor.heads_loaded = False
    elif servo_mode == "calibration":
        calibration_target = 0.5 if plant.actuator_pos < 0.5 else 0.45
        supervisor.seek_origin = plant.actuator_pos
        supervisor.target_track = calibration_target
        supervisor.seek_duration_s = max(command.motion_duration_s, 0.014)
        supervisor.seek_elapsed_s = 0.0
        supervisor.settle_remaining_s = max(command.settle_duration_s, 0.006)
        supervisor.wedge_impulse += physics.actuator_latch_event_force(servo_mode)
    elif servo_mode in {"seek", "track"}:
        delta = command.track_delta
        if abs(delta) < 0.015 and servo_mode == "seek":
            delta = math.copysign(0.06, delta if delta != 0.0 else 1.0)
        if supervisor.repetition_pressure > 0.0 and command.op_kind in {"data", "writeback", "metadata"}:
            delta += 0.020 * supervisor.repetition_pressure * supervisor.repetition_variant
        target = _clamp(previous_track + delta, 0.04, 0.96)
        supervisor.seek_origin = plant.actuator_pos
        supervisor.target_track = target
        seek_duration_s = max(
            command.motion_duration_s,
            drive_profile.track_to_track_ms / 1000.0 + 0.010 * abs(delta),
        )
        seek_duration_s *= 1.0 + 0.10 * supervisor.repetition_pressure * max(0.0, supervisor.repetition_variant)
        supervisor.seek_duration_s = seek_duration_s
        supervisor.seek_elapsed_s = 0.0
        settle_duration_s = max(command.settle_duration_s, 0.004 if servo_mode == "track" else 0.009)
        settle_duration_s *= 1.0 + 0.14 * supervisor.repetition_pressure * max(0.0, -supervisor.repetition_variant)
        supervisor.settle_remaining_s = settle_duration_s
    else:
        supervisor.target_track = previous_track
        supervisor.seek_origin = plant.actuator_pos
        supervisor.seek_duration_s = 0.0
        supervisor.seek_elapsed_s = 0.0
        supervisor.settle_remaining_s = max(command.settle_duration_s, 0.0)

    plant.servo_wedge_timer_s = 0.0
    return state


def apply_event(
    state: AudioRenderState,
    event: StorageEvent,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    start_frame: int = 0,
) -> AudioRenderState:
    del mode_bank, start_frame
    command = command_from_event(event)
    return _apply_command(state, command, drive_profile)


def _sample_seek_reference(supervisor: SupervisorState) -> tuple[float, float]:
    return physics.seek_reference(
        supervisor.seek_origin,
        supervisor.target_track,
        supervisor.seek_elapsed_s,
        supervisor.seek_duration_s,
    )


def _step_modal_bank(
    bank: DiscreteModalBank,
    displacement: FloatArray,
    velocity: FloatArray,
    force: float,
) -> tuple[FloatArray, FloatArray, float]:
    return physics.step_modal_bank(bank, displacement, velocity, force)


def _step_reaction_mode(
    fast_state: float,
    slow_state: float,
    *,
    excitation: float,
    dt: float,
    fast_tau_s: float,
    slow_tau_s: float,
    slow_input_scale: float,
) -> tuple[float, float, float]:
    return physics.step_stiffness_damping_contact(
        fast_state,
        slow_state,
        excitation=excitation,
        dt=dt,
        fast_tau_s=fast_tau_s,
        slow_tau_s=slow_tau_s,
        slow_input_scale=slow_input_scale,
    )


def _render_segment_internal(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    frames: int,
    *,
    bearing_noise_raw: FloatArray,
    windage_noise_raw: FloatArray,
    with_diagnostics: bool,
) -> RenderBlockResult:
    if frames <= 0:
        return RenderBlockResult(state=state, samples=np.zeros(0, dtype=np.float64), diagnostics=_empty_trace())

    dt = 1.0 / state.fs
    samples = np.zeros(frames, dtype=np.float64)
    diagnostics = _empty_trace()
    if with_diagnostics:
        diagnostics = AudioDiagnosticTrace(
            time_s=(state.sample_clock + np.arange(frames, dtype=np.float64)) / state.fs,
            target_rpm=np.zeros(frames, dtype=np.float64),
            actual_rpm=np.zeros(frames, dtype=np.float64),
            actuator_pos=np.zeros(frames, dtype=np.float64),
            actuator_torque=np.zeros(frames, dtype=np.float64),
            source_base_force=np.zeros(frames, dtype=np.float64),
            source_cover_force=np.zeros(frames, dtype=np.float64),
            source_actuator_force=np.zeros(frames, dtype=np.float64),
            source_enclosure_force=np.zeros(frames, dtype=np.float64),
            source_desk_force=np.zeros(frames, dtype=np.float64),
            acoustic_airborne=np.zeros(frames, dtype=np.float64),
            acoustic_structure=np.zeros(frames, dtype=np.float64),
            windage=np.zeros(frames, dtype=np.float64),
            bearing=np.zeros(frames, dtype=np.float64),
            spindle_tone=np.zeros(frames, dtype=np.float64),
            wedge_force=np.zeros(frames, dtype=np.float64),
            contact_force=np.zeros(frames, dtype=np.float64),
            structure_base_velocity=np.zeros(frames, dtype=np.float64),
            structure_cover_velocity=np.zeros(frames, dtype=np.float64),
            structure_enclosure_velocity=np.zeros(frames, dtype=np.float64),
            structure_desk_velocity=np.zeros(frames, dtype=np.float64),
            output=np.zeros(frames, dtype=np.float64),
        )

    plant = state.plant
    supervisor = state.supervisor
    target_omega = supervisor.target_rpm * TAU / 60.0
    harmonic_phases = np.linspace(0.15, 1.4, len(mode_bank.spindle_harmonics), dtype=np.float64)

    for index in range(frames):
        target_omega = supervisor.target_rpm * TAU / 60.0
        spindle_step = physics.step_spindle_motor(
            plant.spindle_omega,
            plant.motor_drive,
            target_omega=target_omega,
            nominal_omega=drive_profile.rpm * TAU / 60.0,
            power_state=supervisor.power_state,
            spinup_ms=drive_profile.spinup_ms,
            spin_down_ms=drive_profile.spin_down_ms,
            dt=dt,
        )
        plant.motor_drive = spindle_step.motor_drive
        plant.spindle_omega = spindle_step.spindle_omega
        plant.spindle_phase = (plant.spindle_phase + spindle_step.phase_increment) % TAU
        rpm_norm = spindle_step.rpm_norm
        motor_reaction = spindle_step.motor_reaction
        startup_active = _startup_active(plant, supervisor, target_omega)
        if startup_active and target_omega > 0.0 and plant.spindle_omega >= target_omega * 0.992:
            supervisor.power_state = "active"
            startup_active = False
        if startup_active:
            supervisor.startup_elapsed_s += dt
        else:
            supervisor.startup_elapsed_s = 0.0

        if startup_active:
            desired_pos = plant.actuator_pos
            desired_vel = 0.0
            supervisor.seek_duration_s = 0.0
            supervisor.seek_elapsed_s = 0.0
            supervisor.settle_remaining_s = 0.0
            supervisor.servo_mode = "idle"
            plant.servo_wedge_timer_s = 0.0
            plant.boundary_timer_s = 0.0
            plant.servo_integrator *= 0.992
            plant.actuator_torque *= 0.990
        else:
            if supervisor.seek_duration_s > 0.0 and supervisor.seek_elapsed_s < supervisor.seek_duration_s:
                desired_pos, desired_vel = _sample_seek_reference(supervisor)
                supervisor.seek_elapsed_s += dt
            else:
                desired_pos = supervisor.target_track
                desired_vel = 0.0
                if supervisor.servo_mode == "seek":
                    supervisor.servo_mode = "settle"

            if supervisor.servo_mode == "settle":
                supervisor.settle_remaining_s = max(0.0, supervisor.settle_remaining_s - dt)
                if supervisor.settle_remaining_s <= 0.0:
                    if supervisor.load_state == "parking":
                        supervisor.servo_mode = "idle"
                        supervisor.load_state = "parked"
                        supervisor.contact_impulse += physics.park_stop_contact_force()
                    elif supervisor.is_sequential and supervisor.transfer_activity > 0.2:
                        supervisor.servo_mode = "track"
                    else:
                        supervisor.servo_mode = "idle"

            if supervisor.transfer_remaining_s > 0.0:
                supervisor.transfer_remaining_s = max(0.0, supervisor.transfer_remaining_s - dt)
            else:
                supervisor.transfer_activity *= 0.9991
                supervisor.directory_activity *= 0.9988
                supervisor.fragmentation_activity *= 0.9990
            if supervisor.repetition_pressure > 0.0:
                supervisor.repetition_pressure *= 0.9992
                supervisor.repetition_variant *= 0.9988

            sectors_per_rev = _command_frequency(drive_profile, supervisor)
            servo_interval = 1.0 / max((plant.spindle_omega / TAU) * sectors_per_rev, 35.0)
            plant.servo_wedge_timer_s -= dt
            if plant.servo_wedge_timer_s <= 0.0:
                plant.servo_wedge_timer_s += servo_interval
                error = desired_pos - plant.actuator_pos
                velocity_error = desired_vel - plant.actuator_vel
                servo_step = physics.voice_coil_servo_step(
                    error=error,
                    velocity_error=velocity_error,
                    integrator=plant.servo_integrator,
                    servo_interval=servo_interval,
                    servo_mode=supervisor.servo_mode,
                    queue_depth=supervisor.queue_depth,
                    retry_activity=supervisor.retry_activity,
                    fragmentation_activity=supervisor.fragmentation_activity,
                    directory_activity=supervisor.directory_activity,
                )
                plant.servo_integrator = servo_step.integrator
                torque_delta = servo_step.torque_command - plant.actuator_torque
                plant.actuator_torque += 0.62 * torque_delta
                supervisor.wedge_impulse += physics.voice_coil_force_transfer(torque_delta, supervisor.servo_mode)
            else:
                plant.actuator_torque *= 0.9992

            if supervisor.is_sequential and supervisor.transfer_activity > 0.2 and supervisor.heads_loaded:
                plant.boundary_timer_s -= dt
                if plant.boundary_timer_s <= 0.0:
                    duration_scale = 0.82 if supervisor.transfer_remaining_s > 0.0 else 1.18
                    interval = duration_scale * max(0.004, 0.010 / max(supervisor.transfer_activity, 0.25))
                    plant.boundary_timer_s += interval
                    supervisor.wedge_impulse += physics.sequential_boundary_contact_force(
                        boundary_gain=acoustic_profile.sequential_boundary_gain,
                        rpm_norm=rpm_norm,
                        fragmentation_activity=supervisor.fragmentation_activity,
                    )
            else:
                plant.boundary_timer_s = 0.0

        actuator_step = physics.step_actuator_mechanics(
            plant.actuator_pos,
            plant.actuator_vel,
            plant.actuator_torque,
            dt,
        )
        plant.actuator_vel = actuator_step.velocity
        plant.actuator_pos = actuator_step.position

        windage_step = physics.step_windage_noise(
            plant.windage_low_state,
            plant.windage_high_state,
            float(windage_noise_raw[index]),
            rpm_norm=rpm_norm,
            startup_active=startup_active,
            windage_gain=drive_profile.windage_gain,
        )
        plant.windage_low_state = windage_step.primary_state
        plant.windage_high_state = windage_step.secondary_state
        windage = windage_step.signal

        bearing_step = physics.step_bearing_noise(
            plant.bearing_state,
            float(bearing_noise_raw[index]),
            rpm_norm=rpm_norm,
            startup_active=startup_active,
            bearing_gain=drive_profile.bearing_gain,
        )
        plant.bearing_state = bearing_step.primary_state
        bearing = bearing_step.signal

        spindle_tone = physics.spindle_rotor_excitation(
            spindle_phase=plant.spindle_phase,
            harmonics=mode_bank.spindle_harmonics,
            weights=mode_bank.spindle_weights,
            phase_offsets=harmonic_phases,
            rpm_norm=rpm_norm,
            startup_active=startup_active,
            platter_gain=mode_bank.platter_gain,
        )

        startup_ramp = physics.motor_startup_current_envelope(supervisor.startup_elapsed_s, startup_active)
        startup_drive_force = physics.spindle_motor_reaction_force(plant.motor_drive, rpm_norm, startup_ramp)
        torque_structure = physics.chassis_reaction_force(
            startup_active=startup_active,
            startup_current_envelope=startup_ramp,
            motor_reaction=motor_reaction,
            motor_reaction_force=startup_drive_force,
        )
        mount_damping = max(acoustic_profile.mount_damping_scale, 0.35)
        wedge_excitation = 0.0 if startup_active else supervisor.wedge_impulse
        contact_excitation = 0.0 if startup_active else supervisor.contact_impulse
        supervisor.wedge_impulse = 0.0
        supervisor.contact_impulse = 0.0
        if startup_active:
            plant.wedge_fast_state *= 0.96
            plant.wedge_slow_state *= 0.96
            plant.contact_fast_state *= 0.95
            plant.contact_slow_state *= 0.95
            wedge_force = 0.0
            contact_force = 0.0
        else:
            plant.wedge_fast_state, plant.wedge_slow_state, wedge_force = _step_reaction_mode(
                plant.wedge_fast_state,
                plant.wedge_slow_state,
                excitation=wedge_excitation,
                dt=dt,
                fast_tau_s=0.00022 / mount_damping,
                slow_tau_s=0.00135 / mount_damping,
                slow_input_scale=0.58,
            )
            plant.contact_fast_state, plant.contact_slow_state, contact_force = _step_reaction_mode(
                plant.contact_fast_state,
                plant.contact_slow_state,
                excitation=contact_excitation,
                dt=dt,
                fast_tau_s=0.00034 / mount_damping,
                slow_tau_s=0.00260 / mount_damping,
                slow_input_scale=0.72,
            )

        source_forces = physics.source_forces(
            startup_active=startup_active,
            torque_structure=torque_structure,
            spindle_tone=spindle_tone,
            bearing=bearing,
            windage=windage,
            wedge_force=wedge_force,
            contact_force=contact_force,
            actuator_vel=plant.actuator_vel,
            transfer_activity=supervisor.transfer_activity,
            directory_activity=supervisor.directory_activity,
            fragmentation_activity=supervisor.fragmentation_activity,
            startup_ramp_value=startup_ramp,
            acoustic_profile=acoustic_profile,
        )

        plant.base_disp, plant.base_vel, base_signal = _step_modal_bank(
            mode_bank.base,
            plant.base_disp,
            plant.base_vel,
            source_forces.base,
        )
        plant.cover_disp, plant.cover_vel, cover_signal = _step_modal_bank(
            mode_bank.cover,
            plant.cover_disp,
            plant.cover_vel,
            source_forces.cover,
        )
        plant.actuator_disp, plant.actuator_vel_modes, actuator_signal = _step_modal_bank(
            mode_bank.actuator,
            plant.actuator_disp,
            plant.actuator_vel_modes,
            source_forces.actuator,
        )
        plant.enclosure_disp, plant.enclosure_vel, enclosure_signal = _step_modal_bank(
            mode_bank.enclosure,
            plant.enclosure_disp,
            plant.enclosure_vel,
            source_forces.enclosure,
        )
        plant.desk_disp, plant.desk_vel, desk_signal = _step_modal_bank(
            mode_bank.desk,
            plant.desk_disp,
            plant.desk_vel,
            source_forces.desk,
        )

        acoustic_mix = physics.mix_acoustic_output(
            startup_active=startup_active,
            rpm_norm=rpm_norm,
            mode_bank=mode_bank,
            spindle_tone=spindle_tone,
            windage=windage,
            bearing=bearing,
            base_signal=base_signal,
            cover_signal=cover_signal,
            actuator_signal=actuator_signal,
            enclosure_signal=enclosure_signal,
            desk_signal=desk_signal,
        )
        final_filter = physics.final_filter_step(
            mixed=acoustic_mix.mixed,
            highpass_prev_input=state.output_highpass_prev_input,
            highpass_state=state.output_highpass_state,
            lowpass_state=state.output_lowpass_state,
            final_highpass_alpha=mode_bank.final_highpass_alpha,
            final_lowpass_alpha=mode_bank.final_lowpass_alpha,
            output_gain=acoustic_profile.output_gain,
        )
        state.output_highpass_prev_input = final_filter.highpass_prev_input
        state.output_highpass_state = final_filter.highpass_state
        state.output_lowpass_state = final_filter.lowpass_state
        shaped = final_filter.output
        samples[index] = shaped

        if with_diagnostics:
            diagnostics.target_rpm[index] = supervisor.target_rpm
            diagnostics.actual_rpm[index] = plant.spindle_omega * 60.0 / TAU
            diagnostics.actuator_pos[index] = plant.actuator_pos
            diagnostics.actuator_torque[index] = plant.actuator_torque
            diagnostics.source_base_force[index] = source_forces.base
            diagnostics.source_cover_force[index] = source_forces.cover
            diagnostics.source_actuator_force[index] = source_forces.actuator
            diagnostics.source_enclosure_force[index] = source_forces.enclosure
            diagnostics.source_desk_force[index] = source_forces.desk
            diagnostics.acoustic_airborne[index] = acoustic_mix.airborne
            diagnostics.acoustic_structure[index] = acoustic_mix.structure
            diagnostics.windage[index] = windage
            diagnostics.bearing[index] = bearing
            diagnostics.spindle_tone[index] = spindle_tone
            diagnostics.wedge_force[index] = wedge_force
            diagnostics.contact_force[index] = contact_force
            diagnostics.structure_base_velocity[index] = base_signal
            diagnostics.structure_cover_velocity[index] = cover_signal
            diagnostics.structure_enclosure_velocity[index] = enclosure_signal
            diagnostics.structure_desk_velocity[index] = desk_signal
            diagnostics.output[index] = shaped

    state.sample_clock += frames
    return RenderBlockResult(state=state, samples=samples, diagnostics=diagnostics)


def _render_block(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    frames: int,
    *,
    scheduled_events: list[ScheduledEvent],
    bearing_noise_raw: FloatArray,
    windage_noise_raw: FloatArray,
    with_diagnostics: bool,
) -> RenderBlockResult:
    if frames <= 0:
        return RenderBlockResult(state=state, samples=np.zeros(0, dtype=np.float64), diagnostics=_empty_trace())
    if not scheduled_events:
        return _render_segment_internal(
            state,
            mode_bank,
            drive_profile,
            acoustic_profile,
            frames,
            bearing_noise_raw=bearing_noise_raw,
            windage_noise_raw=windage_noise_raw,
            with_diagnostics=with_diagnostics,
        )

    cursor = 0
    chunks: list[FloatArray] = []
    traces: list[AudioDiagnosticTrace] = []
    working_state = state
    for event, frame_offset in sorted(scheduled_events, key=lambda item: item[1]):
        offset = min(max(int(frame_offset), 0), frames)
        if offset > cursor:
            segment = _render_segment_internal(
                working_state,
                mode_bank,
                drive_profile,
                acoustic_profile,
                offset - cursor,
                bearing_noise_raw=bearing_noise_raw[cursor:offset],
                windage_noise_raw=windage_noise_raw[cursor:offset],
                with_diagnostics=with_diagnostics,
            )
            working_state = segment.state
            chunks.append(segment.samples)
            if with_diagnostics:
                traces.append(segment.diagnostics)
        working_state = _apply_command(working_state, command_from_event(event), drive_profile)
        cursor = offset

    if cursor < frames:
        segment = _render_segment_internal(
            working_state,
            mode_bank,
            drive_profile,
            acoustic_profile,
            frames - cursor,
            bearing_noise_raw=bearing_noise_raw[cursor:frames],
            windage_noise_raw=windage_noise_raw[cursor:frames],
            with_diagnostics=with_diagnostics,
        )
        working_state = segment.state
        chunks.append(segment.samples)
        if with_diagnostics:
            traces.append(segment.diagnostics)

    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
    diagnostics = concatenate_diagnostic_traces(traces) if with_diagnostics else _empty_trace()
    return RenderBlockResult(state=working_state, samples=samples, diagnostics=diagnostics)


def render_diagnostic_chunk(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    frames: int,
    *,
    scheduled_events: list[ScheduledEvent] | tuple[ScheduledEvent, ...] = (),
    bearing_noise_raw: FloatArray,
    windage_noise_raw: FloatArray,
) -> tuple[AudioRenderState, FloatArray, AudioDiagnosticTrace]:
    result = _render_block(
        state,
        mode_bank,
        drive_profile,
        acoustic_profile,
        frames,
        scheduled_events=list(scheduled_events),
        bearing_noise_raw=bearing_noise_raw,
        windage_noise_raw=windage_noise_raw,
        with_diagnostics=True,
    )
    return result.state, result.samples, result.diagnostics


def render_chunk(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    frames: int,
    *,
    scheduled_events: list[ScheduledEvent] | tuple[ScheduledEvent, ...] = (),
    bearing_noise_raw: FloatArray,
    windage_noise_raw: FloatArray,
) -> tuple[AudioRenderState, FloatArray]:
    result = _render_block(
        state,
        mode_bank,
        drive_profile,
        acoustic_profile,
        frames,
        scheduled_events=list(scheduled_events),
        bearing_noise_raw=bearing_noise_raw,
        windage_noise_raw=windage_noise_raw,
        with_diagnostics=False,
    )
    return result.state, result.samples
