from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from .commands import AudioCommand, command_from_event
from ..profiles import AcousticProfile, DriveProfile
from ..storage_events import StorageEvent


FloatArray = npt.NDArray[np.float64]
ScheduledEvent = tuple[StorageEvent, int]
TAU = 2.0 * math.pi
EPS = 1e-9


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
    target_track: float = 0.52
    seek_origin: float = 0.52
    seek_duration_s: float = 0.0
    seek_elapsed_s: float = 0.0
    settle_remaining_s: float = 0.0
    wedge_impulse: float = 0.0
    contact_impulse: float = 0.0
    maintenance_activity: float = 0.0
    retry_activity: float = 0.0


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
    return min(max(value, lo), hi)


def _one_pole_alpha(cutoff_hz: float, sample_rate: int) -> float:
    cutoff = max(float(cutoff_hz), 1.0)
    return 1.0 - math.exp(-TAU * cutoff / sample_rate)


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
    return AudioDiagnosticTrace(
        time_s=empty,
        target_rpm=empty,
        actual_rpm=empty,
        actuator_pos=empty,
        actuator_torque=empty,
        structure_base_velocity=empty,
        structure_cover_velocity=empty,
        structure_enclosure_velocity=empty,
        structure_desk_velocity=empty,
        output=empty,
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
    supervisor.target_rpm = max(float(command.target_rpm), 0.0)
    supervisor.power_state = command.power_state
    supervisor.queue_depth = max(1, int(command.queue_depth))
    supervisor.op_kind = command.op_kind
    supervisor.transfer_activity = float(command.transfer_activity)
    supervisor.is_sequential = bool(command.is_sequential)
    supervisor.is_flush = bool(command.is_flush)
    supervisor.maintenance_activity = 0.55 if command.maintenance else 0.0
    supervisor.retry_activity = 0.70 if command.retry else 0.0
    supervisor.heads_loaded = bool(command.heads_loaded)
    supervisor.load_state = "loaded" if supervisor.heads_loaded else "parked"
    if not previous_heads_loaded and supervisor.heads_loaded:
        supervisor.contact_impulse += 0.26
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

    if servo_mode == "park":
        supervisor.seek_origin = plant.actuator_pos
        supervisor.target_track = 0.04
        supervisor.seek_duration_s = max(command.motion_duration_s, 0.024)
        supervisor.seek_elapsed_s = 0.0
        supervisor.settle_remaining_s = max(command.settle_duration_s, 0.010)
        supervisor.wedge_impulse += 0.22
        supervisor.load_state = "parking"
        supervisor.heads_loaded = False
    elif servo_mode == "calibration":
        calibration_target = 0.5 if plant.actuator_pos < 0.5 else 0.45
        supervisor.seek_origin = plant.actuator_pos
        supervisor.target_track = calibration_target
        supervisor.seek_duration_s = max(command.motion_duration_s, 0.014)
        supervisor.seek_elapsed_s = 0.0
        supervisor.settle_remaining_s = max(command.settle_duration_s, 0.006)
        supervisor.wedge_impulse += 0.18
    elif servo_mode in {"seek", "track"}:
        delta = command.track_delta
        if abs(delta) < 0.015 and servo_mode == "seek":
            delta = math.copysign(0.06, delta if delta != 0.0 else 1.0)
        target = _clamp(previous_track + delta, 0.04, 0.96)
        supervisor.seek_origin = plant.actuator_pos
        supervisor.target_track = target
        supervisor.seek_duration_s = max(
            command.motion_duration_s,
            drive_profile.track_to_track_ms / 1000.0 + 0.010 * abs(delta),
        )
        supervisor.seek_elapsed_s = 0.0
        supervisor.settle_remaining_s = max(command.settle_duration_s, 0.004 if servo_mode == "track" else 0.009)
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
    if supervisor.seek_duration_s <= 0.0:
        return supervisor.target_track, 0.0
    progress = _clamp(supervisor.seek_elapsed_s / max(supervisor.seek_duration_s, EPS), 0.0, 1.0)
    eased = 0.5 - 0.5 * math.cos(math.pi * progress)
    desired_pos = supervisor.seek_origin + (supervisor.target_track - supervisor.seek_origin) * eased
    desired_vel = (
        (supervisor.target_track - supervisor.seek_origin)
        * 0.5
        * math.pi
        * math.sin(math.pi * progress)
        / max(supervisor.seek_duration_s, EPS)
    )
    return desired_pos, desired_vel


def _step_modal_bank(
    bank: DiscreteModalBank,
    displacement: FloatArray,
    velocity: FloatArray,
    force: float,
) -> tuple[FloatArray, FloatArray, float]:
    if bank.size == 0:
        return displacement, velocity, 0.0
    kicked_velocity = velocity + bank.input_gain * force
    new_displacement = bank.coeff_xx * displacement + bank.coeff_xv * kicked_velocity
    new_velocity = bank.coeff_vx * displacement + bank.coeff_vv * kicked_velocity
    signal = float(np.dot(new_velocity, bank.output_gain))
    return new_displacement, new_velocity, signal


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
    fast_state = (fast_state + excitation) * math.exp(-dt / max(fast_tau_s, 1e-5))
    slow_state = (slow_state + excitation * slow_input_scale) * math.exp(-dt / max(slow_tau_s, 1e-5))
    return fast_state, slow_state, fast_state - slow_state


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
        omega_before = plant.spindle_omega
        target_omega = supervisor.target_rpm * TAU / 60.0
        drive_target = 0.0
        if target_omega > EPS:
            drive_target = _clamp((target_omega - omega_before) / target_omega, 0.0, 1.0)
        drive_tau = 0.22 if supervisor.power_state == "starting" else 0.08
        drive_alpha = 1.0 - math.exp(-dt / drive_tau)
        plant.motor_drive += (drive_target - plant.motor_drive) * drive_alpha
        if target_omega >= omega_before:
            tau_s = max(drive_profile.spinup_ms / 1000.0, 0.35)
        else:
            tau_s = max(drive_profile.spin_down_ms / 1000.0, 0.28)
        alpha = 1.0 - math.exp(-dt / tau_s)
        plant.spindle_omega = omega_before + (target_omega - omega_before) * alpha
        phase_increment = 0.5 * (omega_before + plant.spindle_omega) * dt
        plant.spindle_phase = (plant.spindle_phase + phase_increment) % TAU
        rpm_norm = _clamp(plant.spindle_omega / max(drive_profile.rpm * TAU / 60.0, EPS), 0.0, 1.35)
        motor_reaction = (plant.spindle_omega - omega_before) / max(dt, EPS)
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
                        supervisor.contact_impulse += 0.40
                    elif supervisor.is_sequential and supervisor.transfer_activity > 0.2:
                        supervisor.servo_mode = "track"
                    else:
                        supervisor.servo_mode = "idle"

            sectors_per_rev = _command_frequency(drive_profile, supervisor)
            servo_interval = 1.0 / max((plant.spindle_omega / TAU) * sectors_per_rev, 35.0)
            plant.servo_wedge_timer_s -= dt
            if plant.servo_wedge_timer_s <= 0.0:
                plant.servo_wedge_timer_s += servo_interval
                error = desired_pos - plant.actuator_pos
                velocity_error = desired_vel - plant.actuator_vel
                mode_gain = 1.35 if supervisor.servo_mode == "seek" else 0.84 if supervisor.servo_mode == "track" else 1.08
                kp = 18.0 * mode_gain
                kd = 2.8 * mode_gain
                ki = 7.5 if supervisor.servo_mode in {"seek", "settle"} else 2.6
                plant.servo_integrator = _clamp(
                    plant.servo_integrator + error * servo_interval,
                    -0.08,
                    0.08,
                )
                torque_command = (
                    kp * error
                    + kd * velocity_error
                    + ki * plant.servo_integrator
                )
                torque_command *= 1.0 + 0.03 * max(supervisor.queue_depth - 1, 0)
                torque_command += 0.18 * supervisor.retry_activity
                torque_command = _clamp(torque_command, -8.0, 8.0)
                torque_delta = torque_command - plant.actuator_torque
                plant.actuator_torque += 0.62 * torque_delta
                impulse_scale = 0.10 if supervisor.servo_mode == "track" else 0.56 if supervisor.servo_mode == "seek" else 0.30
                supervisor.wedge_impulse += abs(torque_delta) * impulse_scale
            else:
                plant.actuator_torque *= 0.9992

            if supervisor.is_sequential and supervisor.transfer_activity > 0.2 and supervisor.heads_loaded:
                plant.boundary_timer_s -= dt
                if plant.boundary_timer_s <= 0.0:
                    interval = max(0.004, 0.010 / max(supervisor.transfer_activity, 0.25))
                    plant.boundary_timer_s += interval
                    supervisor.wedge_impulse += (
                        0.045 * acoustic_profile.sequential_boundary_gain * (0.8 + 0.2 * rpm_norm)
                    )
            else:
                plant.boundary_timer_s = 0.0

        actuator_accel = 190.0 * plant.actuator_torque - 90.0 * plant.actuator_vel
        plant.actuator_vel += actuator_accel * dt
        plant.actuator_pos = _clamp(plant.actuator_pos + plant.actuator_vel * dt, 0.0, 1.0)

        windage_low_alpha = 0.005 if startup_active else 0.020
        windage_high_alpha = 0.024 if startup_active else 0.130
        plant.windage_low_state += windage_low_alpha * (float(windage_noise_raw[index]) - plant.windage_low_state)
        plant.windage_high_state += windage_high_alpha * (plant.windage_low_state - plant.windage_high_state)
        windage_scale = (
            rpm_norm * (0.002 + 0.050 * rpm_norm**3.8)
            if startup_active
            else (0.010 * rpm_norm + 0.18 * rpm_norm * rpm_norm)
        )
        windage = (
            (plant.windage_low_state - plant.windage_high_state)
            * windage_scale
            * drive_profile.windage_gain
        )

        bearing_alpha = 0.010 if startup_active else 0.060
        plant.bearing_state += bearing_alpha * (float(bearing_noise_raw[index]) - plant.bearing_state)
        bearing_scale = (
            rpm_norm * (0.002 + 0.012 * rpm_norm**2.0)
            if startup_active
            else (0.006 * rpm_norm + 0.034 * rpm_norm**1.25)
        )
        bearing = plant.bearing_state * bearing_scale * drive_profile.bearing_gain

        harmonic = 0.0
        for harmonic_index, harmonic_weight, phase_offset in zip(
            mode_bank.spindle_harmonics,
            mode_bank.spindle_weights,
            harmonic_phases,
            strict=True,
        ):
            startup_weight = rpm_norm ** (0.55 * max(harmonic_index - 1, 0)) if startup_active else 1.0
            harmonic += harmonic_weight * startup_weight * math.sin(plant.spindle_phase * harmonic_index + float(phase_offset))
        spindle_tone = harmonic * ((0.005 + 0.018 * rpm_norm * rpm_norm) if startup_active else (0.012 + 0.040 * rpm_norm * rpm_norm)) * mode_bank.platter_gain

        startup_ramp = 1.0 - math.exp(-supervisor.startup_elapsed_s / 0.38) if startup_active else 1.0
        startup_drive_force = plant.motor_drive * (0.18 + 0.82 * rpm_norm) * startup_ramp
        if startup_active:
            torque_structure = startup_ramp * 0.00012 * motor_reaction + 0.085 * startup_drive_force
        else:
            torque_structure = 0.0008 * motor_reaction
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

        if startup_active:
            base_force = (
                1.55 * torque_structure
                + 0.14 * spindle_tone * startup_ramp
                + 0.06 * bearing
                + 0.03 * windage
            )
            cover_force = (
                0.44 * torque_structure
                + 0.08 * spindle_tone * startup_ramp
                + 0.03 * windage
                + 0.02 * bearing
            )
            actuator_force = 0.0
            enclosure_force = (
                acoustic_profile.enclosure_coupling * (0.52 * base_force + 0.22 * cover_force)
                + acoustic_profile.internal_air_coupling * (0.04 * windage + 0.04 * spindle_tone)
            )
            desk_force = acoustic_profile.desk_coupling * (0.94 * base_force + 0.20 * cover_force)
        else:
            base_force = (
                0.58 * torque_structure
                + 0.56 * wedge_force
                + 0.42 * contact_force
                + 0.16 * bearing
            )
            cover_force = (
                0.24 * torque_structure
                + 0.34 * wedge_force
                + 0.20 * contact_force
                + 0.14 * windage
            )
            actuator_force = (
                0.24 * wedge_force
                + 0.18 * abs(plant.actuator_vel)
                + 0.10 * contact_force
                + 0.06 * supervisor.transfer_activity
            )
            enclosure_force = (
                acoustic_profile.enclosure_coupling * (0.30 * base_force + 0.20 * cover_force)
                + acoustic_profile.internal_air_coupling * (0.14 * windage + 0.08 * spindle_tone)
            )
            desk_force = acoustic_profile.desk_coupling * (
                0.44 * base_force
                + 0.18 * cover_force
                + 0.58 * wedge_force
                + 0.26 * contact_force
            )

        plant.base_disp, plant.base_vel, base_signal = _step_modal_bank(
            mode_bank.base,
            plant.base_disp,
            plant.base_vel,
            base_force,
        )
        plant.cover_disp, plant.cover_vel, cover_signal = _step_modal_bank(
            mode_bank.cover,
            plant.cover_disp,
            plant.cover_vel,
            cover_force,
        )
        plant.actuator_disp, plant.actuator_vel_modes, actuator_signal = _step_modal_bank(
            mode_bank.actuator,
            plant.actuator_disp,
            plant.actuator_vel_modes,
            actuator_force,
        )
        plant.enclosure_disp, plant.enclosure_vel, enclosure_signal = _step_modal_bank(
            mode_bank.enclosure,
            plant.enclosure_disp,
            plant.enclosure_vel,
            enclosure_force,
        )
        plant.desk_disp, plant.desk_vel, desk_signal = _step_modal_bank(
            mode_bank.desk,
            plant.desk_disp,
            plant.desk_vel,
            desk_force,
        )

        if startup_active:
            startup_airborne_gate = rpm_norm**2.0
            structure = (
                mode_bank.structure_gain
                * (1.72 * base_signal + 0.78 * cover_signal + 1.02 * enclosure_signal + 1.42 * desk_signal)
            )
            airborne = (
                mode_bank.direct_gain * startup_airborne_gate * (0.16 * spindle_tone + 0.05 * windage + 0.03 * bearing)
                + 0.07 * mode_bank.cover_gain * cover_signal
            )
        else:
            structure = (
                mode_bank.structure_gain
                * (base_signal + cover_signal + enclosure_signal + desk_signal)
            )
            airborne = (
                mode_bank.direct_gain * (0.18 * spindle_tone + 0.22 * windage + 0.12 * bearing)
                + 0.12 * mode_bank.cover_gain * cover_signal
                + 0.22 * mode_bank.actuator_gain * actuator_signal
            )
        mixed = airborne + structure
        hp_output = mixed - state.output_highpass_prev_input + (1.0 - mode_bank.final_highpass_alpha) * state.output_highpass_state
        state.output_highpass_prev_input = mixed
        state.output_highpass_state = hp_output
        lp_output = state.output_lowpass_state + mode_bank.final_lowpass_alpha * (hp_output - state.output_lowpass_state)
        state.output_lowpass_state = lp_output
        shaped = math.tanh(lp_output * 2.0) * acoustic_profile.output_gain
        samples[index] = shaped

        if with_diagnostics:
            diagnostics.target_rpm[index] = supervisor.target_rpm
            diagnostics.actual_rpm[index] = plant.spindle_omega * 60.0 / TAU
            diagnostics.actuator_pos[index] = plant.actuator_pos
            diagnostics.actuator_torque[index] = plant.actuator_torque
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
    if not with_diagnostics or not traces:
        diagnostics = _empty_trace() if not with_diagnostics else AudioDiagnosticTrace(
            time_s=np.concatenate([trace.time_s for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
            target_rpm=np.concatenate([trace.target_rpm for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
            actual_rpm=np.concatenate([trace.actual_rpm for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
            actuator_pos=np.concatenate([trace.actuator_pos for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
            actuator_torque=np.concatenate([trace.actuator_torque for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
            structure_base_velocity=np.concatenate([trace.structure_base_velocity for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
            structure_cover_velocity=np.concatenate([trace.structure_cover_velocity for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
            structure_enclosure_velocity=np.concatenate([trace.structure_enclosure_velocity for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
            structure_desk_velocity=np.concatenate([trace.structure_desk_velocity for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
            output=np.concatenate([trace.output for trace in traces]) if traces else np.zeros(0, dtype=np.float64),
        )
    else:
        diagnostics = AudioDiagnosticTrace(
            time_s=np.concatenate([trace.time_s for trace in traces]),
            target_rpm=np.concatenate([trace.target_rpm for trace in traces]),
            actual_rpm=np.concatenate([trace.actual_rpm for trace in traces]),
            actuator_pos=np.concatenate([trace.actuator_pos for trace in traces]),
            actuator_torque=np.concatenate([trace.actuator_torque for trace in traces]),
            structure_base_velocity=np.concatenate([trace.structure_base_velocity for trace in traces]),
            structure_cover_velocity=np.concatenate([trace.structure_cover_velocity for trace in traces]),
            structure_enclosure_velocity=np.concatenate([trace.structure_enclosure_velocity for trace in traces]),
            structure_desk_velocity=np.concatenate([trace.structure_desk_velocity for trace in traces]),
            output=np.concatenate([trace.output for trace in traces]),
        )
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
