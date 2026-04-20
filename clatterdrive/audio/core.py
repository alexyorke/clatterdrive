from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
from scipy import signal

from .plant import SeekPlan, clamp as _clamp, plan_seek_motion, sample_seek_reference
from .voices import AudioVoice, build_voices, mix_voice_path
from ..profiles import AcousticProfile, DriveProfile
from ..storage_events import ScheduledStorageEvent, StorageEvent


FloatArray = npt.NDArray[np.float64]
ScheduledEvent = ScheduledStorageEvent


@dataclass(frozen=True)
class MechanicalMode:
    name: str
    frequency_hz: float
    damping_ratio: float
    radiation_gain: float


@dataclass(frozen=True)
class DiscreteModalBank:
    a11: FloatArray
    a12: FloatArray
    a21: FloatArray
    a22: FloatArray
    b1: FloatArray
    b2: FloatArray
    gains: FloatArray


@dataclass(frozen=True)
class CoupledStructureModel:
    ad: FloatArray
    bd: FloatArray
    velocity_readout: FloatArray


@dataclass(frozen=True)
class MechanicalPlantConfig:
    spindle_inertia: float
    spindle_drag_linear: float
    spindle_drag_quadratic: float
    spindle_torque_max: float
    spindle_torque_time_constant_s: float
    spindle_kp: float
    spindle_ki: float
    servo_sectors_per_rev: int
    actuator_inertia: float
    actuator_damping: float
    actuator_torque_max: float
    actuator_torque_time_constant_s: float
    actuator_max_velocity: float
    actuator_max_accel: float
    actuator_max_jerk: float
    track_integral_gain: float
    rro_1x: float
    rro_2x: float


@dataclass(frozen=True)
class AudioModeBank:
    platter_modes: tuple[MechanicalMode, ...]
    cover_modes: tuple[MechanicalMode, ...]
    actuator_modes: tuple[MechanicalMode, ...]
    structure_modes: tuple[MechanicalMode, ...]
    platter_step: DiscreteModalBank
    cover_step: DiscreteModalBank
    actuator_step: DiscreteModalBank
    structure_step: DiscreteModalBank
    coupled_structure: CoupledStructureModel
    plant_config: MechanicalPlantConfig
    bearing_sos: FloatArray
    windage_sos: FloatArray
    final_highpass_sos: FloatArray
    final_lowpass_sos: FloatArray


@dataclass
class AudioRenderState:
    fs: int
    sample_clock: int = 0
    target_rpm: float = 0.0
    queue_depth: int = 1
    op_kind: str = "data"
    is_flush: bool = False
    is_spinup: bool = False
    is_sequential: bool = False
    heads_loaded: bool = False
    transfer_activity: float = 0.0
    servo_mode: str = "idle"
    spindle_angle: float = 0.0
    spindle_omega: float = 0.0
    spindle_torque: float = 0.0
    spindle_integrator: float = 0.0
    servo_tick_phase: float = 0.0
    actuator_pos: float = 0.0
    actuator_vel: float = 0.0
    actuator_torque: float = 0.0
    actuator_torque_cmd: float = 0.0
    actuator_integrator: float = 0.0
    actuator_target_pos: float = 0.0
    actuator_seek_hz: float = 110.0
    actuator_track_hz: float = 54.0
    actuator_damping_ratio: float = 0.86
    actuator_direction: float = 1.0
    pending_contact_impulse: float = 0.0
    servo_tick_energy: float = 0.0
    servo_tick_cooldown_s: float = 0.0
    seek_plan: SeekPlan | None = None
    seek_plan_elapsed_s: float = 0.0
    settle_time_remaining: float = 0.0
    seek_time_remaining: float = 0.0
    output_gain: float = 0.88
    platter_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    platter_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    cover_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    cover_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    actuator_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    actuator_vel_modes: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    structure_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    structure_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    structure_state: FloatArray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    bearing_zi: FloatArray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    windage_zi: FloatArray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    final_highpass_zi: FloatArray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    final_lowpass_zi: FloatArray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))


@dataclass(frozen=True)
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


BASE_PLATTER_MODES: tuple[tuple[str, float, float, float], ...] = (
    ("platter-1", 359.0, 0.020, 0.18),
    ("platter-2", 525.0, 0.018, 0.20),
    ("platter-3", 650.0, 0.020, 0.17),
    ("platter-4", 1080.0, 0.024, 0.13),
    ("platter-5", 1838.0, 0.028, 0.09),
    ("platter-6", 2100.0, 0.030, 0.07),
)
BASE_COVER_MODES: tuple[tuple[str, float, float, float], ...] = (
    ("cover-1", 212.5, 0.050, 0.14),
    ("cover-2", 575.0, 0.042, 0.18),
    ("cover-3", 775.0, 0.040, 0.18),
    ("cover-4", 1200.0, 0.038, 0.20),
    ("cover-5", 1638.0, 0.035, 0.17),
    ("cover-6", 1850.0, 0.034, 0.15),
    ("cover-7", 2675.0, 0.030, 0.07),
)
BASE_ACTUATOR_MODES: tuple[tuple[str, float, float, float], ...] = (
    ("actuator-1", 975.0, 0.032, 0.24),
    ("actuator-2", 1325.0, 0.028, 0.33),
    ("actuator-3", 1680.0, 0.026, 0.31),
    ("actuator-4", 1980.0, 0.026, 0.26),
    ("actuator-5", 2400.0, 0.028, 0.20),
)
BASE_STRUCTURE_MODES: tuple[tuple[str, float, float, float], ...] = (
    ("structure-1", 72.0, 0.080, 0.20),
    ("structure-2", 120.0, 0.068, 0.24),
    ("structure-3", 167.0, 0.058, 0.18),
    ("structure-4", 250.0, 0.050, 0.14),
    ("structure-5", 420.0, 0.044, 0.10),
)


def _configure_mode_bank(
    base_modes: Sequence[tuple[str, float, float, float]],
    frequency_scale: float,
    gain_scale: float,
) -> tuple[MechanicalMode, ...]:
    return tuple(
        MechanicalMode(
            name=name,
            frequency_hz=freq * frequency_scale,
            damping_ratio=zeta,
            radiation_gain=gain * gain_scale,
        )
        for name, freq, zeta, gain in base_modes
    )


def _discretize_modes(modes: Sequence[MechanicalMode], sample_rate: int) -> DiscreteModalBank:
    dt = 1.0 / sample_rate
    a11: list[float] = []
    a12: list[float] = []
    a21: list[float] = []
    a22: list[float] = []
    b1: list[float] = []
    b2: list[float] = []
    gains: list[float] = []

    identity = np.eye(2, dtype=np.float64)
    zero = np.zeros((2, 1), dtype=np.float64)
    for mode in modes:
        wn = 2.0 * math.pi * mode.frequency_hz
        a_matrix = np.array(
            [[0.0, 1.0], [-(wn * wn), -(2.0 * mode.damping_ratio * wn)]],
            dtype=np.float64,
        )
        b_matrix = np.array([[0.0], [1.0]], dtype=np.float64)
        ad, bd, _, _, _ = signal.cont2discrete(
            (a_matrix, b_matrix, identity, zero),
            dt,
            method="zoh",
        )
        a11.append(float(ad[0, 0]))
        a12.append(float(ad[0, 1]))
        a21.append(float(ad[1, 0]))
        a22.append(float(ad[1, 1]))
        b1.append(float(bd[0, 0]))
        b2.append(float(bd[1, 0]))
        gains.append(mode.radiation_gain)

    return DiscreteModalBank(
        a11=np.array(a11, dtype=np.float64),
        a12=np.array(a12, dtype=np.float64),
        a21=np.array(a21, dtype=np.float64),
        a22=np.array(a22, dtype=np.float64),
        b1=np.array(b1, dtype=np.float64),
        b2=np.array(b2, dtype=np.float64),
        gains=np.array(gains, dtype=np.float64),
    )


def _build_structure_model(
    sample_rate: int,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
) -> CoupledStructureModel:
    dt = 1.0 / sample_rate
    masses = np.array(
        [
            1.0 + 0.14 * max(drive_profile.platters - 1, 0),
            0.82 + 0.06 * max(drive_profile.platters - 1, 0),
            (1.10 if drive_profile.helium else 1.45) * acoustic_profile.enclosure_mass_scale,
            2.8 * acoustic_profile.table_mass_scale,
        ],
        dtype=np.float64,
    )
    frequencies = np.array(
        [
            82.0 * drive_profile.cover_frequency_scale,
            134.0 * drive_profile.cover_frequency_scale,
            96.0 * acoustic_profile.enclosure_resonance_scale,
            58.0 * acoustic_profile.table_resonance_scale,
        ],
        dtype=np.float64,
    )
    stiffness = (2.0 * np.pi * frequencies) ** 2 * masses
    k12 = 0.72 * min(stiffness[0], stiffness[1])
    k13 = acoustic_profile.enclosure_coupling * 0.64 * min(stiffness[0], stiffness[2])
    k23 = acoustic_profile.internal_air_coupling * 0.36 * min(stiffness[1], stiffness[2])
    k34 = acoustic_profile.desk_coupling * 0.70 * min(stiffness[2], stiffness[3])
    k14 = acoustic_profile.desk_coupling * 0.18 * min(stiffness[0], stiffness[3])
    k_matrix = np.array(
        [
            [stiffness[0] + k12 + k13 + k14, -k12, -k13, -k14],
            [-k12, stiffness[1] + k12 + k23, -k23, 0.0],
            [-k13, -k23, stiffness[2] + k13 + k23 + k34, -k34],
            [-k14, 0.0, -k34, stiffness[3] + k14 + k34],
        ],
        dtype=np.float64,
    )
    alpha = 6.5 * acoustic_profile.mount_damping_scale
    beta = 5.4e-5 * acoustic_profile.mount_damping_scale
    mass_damping = np.diag(
        np.array(
            [
                0.0,
                0.0,
                5.2 * acoustic_profile.mount_damping_scale,
                3.8 * acoustic_profile.table_damping_scale,
            ],
            dtype=np.float64,
        )
    )
    c_matrix = alpha * np.diag(masses) + beta * k_matrix + mass_damping
    inv_mass = np.diag(1.0 / masses)

    g_matrix = np.array(
        [
            [1.00, 0.40, 0.05],
            [0.18, 0.95, 0.26],
            [
                0.48 * acoustic_profile.enclosure_coupling,
                0.28 * acoustic_profile.enclosure_coupling,
                0.92 * acoustic_profile.internal_air_coupling,
            ],
            [
                0.14 * acoustic_profile.desk_coupling,
                0.12 * acoustic_profile.desk_coupling,
                0.24 * acoustic_profile.desk_coupling,
            ],
        ],
        dtype=np.float64,
    )

    zeros = np.zeros((4, 4), dtype=np.float64)
    identity = np.eye(4, dtype=np.float64)
    a_matrix = np.block(
        [
            [zeros, identity],
            [-(inv_mass @ k_matrix), -(inv_mass @ c_matrix)],
        ]
    )
    b_matrix = np.vstack([np.zeros((4, 3), dtype=np.float64), inv_mass @ g_matrix])
    ad, bd, _, _, _ = signal.cont2discrete(
        (a_matrix, b_matrix, np.eye(8, dtype=np.float64), np.zeros((8, 3), dtype=np.float64)),
        dt,
        method="zoh",
    )
    velocity_readout = np.array(
        [
            0.28,
            0.18,
            acoustic_profile.enclosure_radiation_gain,
            acoustic_profile.table_radiation_gain,
        ],
        dtype=np.float64,
    )
    return CoupledStructureModel(
        ad=np.asarray(ad, dtype=np.float64),
        bd=np.asarray(bd, dtype=np.float64),
        velocity_readout=velocity_readout,
    )


def _build_plant_config(drive_profile: DriveProfile) -> MechanicalPlantConfig:
    target_omega = drive_profile.rpm * 2.0 * math.pi / 60.0
    spinup_s = max(drive_profile.spinup_ms / 1000.0, 0.25)
    spindle_inertia = 5.6e-6 * (0.86 if drive_profile.helium else 1.0) * (1.0 + 0.18 * max(drive_profile.platters - 1, 0))
    drag_linear = 1.1e-5 * drive_profile.bearing_gain * (0.9 + 0.05 * drive_profile.platters)
    drag_quadratic = 1.0e-8 * drive_profile.windage_gain * (0.72 if drive_profile.helium else 1.0) * (1.0 + 0.07 * drive_profile.platters)
    target_torque = spindle_inertia * target_omega / max(spinup_s * 0.65, 0.2)
    spindle_torque_max = target_torque + drag_linear * target_omega + drag_quadratic * target_omega * target_omega
    spindle_kp = spindle_torque_max / max(target_omega * 0.9, 1.0)
    spindle_ki = spindle_kp / max(spinup_s * 0.28, 0.04)

    actuator_inertia = 8.5e-5 / max(drive_profile.actuator_frequency_scale, 0.1)
    actuator_damping = actuator_inertia * (2.0 * math.pi * 82.0) * 0.23
    actuator_torque_max = 0.82 * (1.0 + 0.08 * max(drive_profile.platters - 1, 0))
    actuator_torque_tau = 0.00085 if drive_profile.helium else 0.00105
    servo_sectors = 120 if drive_profile.helium else 96
    actuator_velocity = 420.0 * drive_profile.actuator_frequency_scale
    actuator_accel = 2.8e5 * drive_profile.actuator_frequency_scale
    actuator_jerk = 1.8e8 * drive_profile.actuator_frequency_scale

    return MechanicalPlantConfig(
        spindle_inertia=spindle_inertia,
        spindle_drag_linear=drag_linear,
        spindle_drag_quadratic=drag_quadratic,
        spindle_torque_max=spindle_torque_max,
        spindle_torque_time_constant_s=0.018 if drive_profile.helium else 0.024,
        spindle_kp=spindle_kp,
        spindle_ki=spindle_ki,
        servo_sectors_per_rev=servo_sectors,
        actuator_inertia=actuator_inertia,
        actuator_damping=actuator_damping,
        actuator_torque_max=actuator_torque_max,
        actuator_torque_time_constant_s=actuator_torque_tau,
        actuator_max_velocity=actuator_velocity,
        actuator_max_accel=actuator_accel,
        actuator_max_jerk=actuator_jerk,
        track_integral_gain=0.32,
        rro_1x=0.00125 * (0.82 if drive_profile.helium else 1.0),
        rro_2x=0.00072 * (0.82 if drive_profile.helium else 1.0),
    )


def build_mode_bank(
    drive_profile: DriveProfile,
    sample_rate: int,
    acoustic_profile: AcousticProfile,
) -> AudioModeBank:
    platter_modes = _configure_mode_bank(
        BASE_PLATTER_MODES,
        drive_profile.platter_frequency_scale,
        drive_profile.platter_gain_scale,
    )
    cover_modes = _configure_mode_bank(
        BASE_COVER_MODES,
        drive_profile.cover_frequency_scale,
        drive_profile.cover_gain_scale,
    )
    actuator_modes = _configure_mode_bank(
        BASE_ACTUATOR_MODES,
        drive_profile.actuator_frequency_scale,
        drive_profile.actuator_gain_scale,
    )
    structure_modes = _configure_mode_bank(
        BASE_STRUCTURE_MODES,
        drive_profile.cover_frequency_scale,
        drive_profile.cover_gain_scale,
    )

    bearing_sos = signal.butter(
        2,
        [70.0, 950.0],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    windage_sos = signal.butter(
        2,
        [180.0, 2400.0],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    final_highpass_sos = signal.butter(
        2,
        acoustic_profile.final_highpass_hz,
        btype="highpass",
        fs=sample_rate,
        output="sos",
    )
    final_lowpass_sos = signal.butter(
        2,
        acoustic_profile.final_lowpass_hz,
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    )

    return AudioModeBank(
        platter_modes=platter_modes,
        cover_modes=cover_modes,
        actuator_modes=actuator_modes,
        structure_modes=structure_modes,
        platter_step=_discretize_modes(platter_modes, sample_rate),
        cover_step=_discretize_modes(cover_modes, sample_rate),
        actuator_step=_discretize_modes(actuator_modes, sample_rate),
        structure_step=_discretize_modes(structure_modes, sample_rate),
        coupled_structure=_build_structure_model(sample_rate, drive_profile, acoustic_profile),
        plant_config=_build_plant_config(drive_profile),
        bearing_sos=np.asarray(bearing_sos, dtype=np.float64),
        windage_sos=np.asarray(windage_sos, dtype=np.float64),
        final_highpass_sos=np.asarray(final_highpass_sos, dtype=np.float64),
        final_lowpass_sos=np.asarray(final_lowpass_sos, dtype=np.float64),
    )


def initialize_render_state(
    sample_rate: int,
    mode_bank: AudioModeBank,
    acoustic_profile: AcousticProfile,
) -> AudioRenderState:
    return AudioRenderState(
        fs=sample_rate,
        output_gain=acoustic_profile.output_gain,
        platter_disp=np.zeros(len(mode_bank.platter_modes), dtype=np.float64),
        platter_vel=np.zeros(len(mode_bank.platter_modes), dtype=np.float64),
        cover_disp=np.zeros(len(mode_bank.cover_modes), dtype=np.float64),
        cover_vel=np.zeros(len(mode_bank.cover_modes), dtype=np.float64),
        actuator_disp=np.zeros(len(mode_bank.actuator_modes), dtype=np.float64),
        actuator_vel_modes=np.zeros(len(mode_bank.actuator_modes), dtype=np.float64),
        structure_disp=np.zeros(len(mode_bank.structure_modes), dtype=np.float64),
        structure_vel=np.zeros(len(mode_bank.structure_modes), dtype=np.float64),
        structure_state=np.zeros(8, dtype=np.float64),
        bearing_zi=np.zeros((mode_bank.bearing_sos.shape[0], 2), dtype=np.float64),
        windage_zi=np.zeros((mode_bank.windage_sos.shape[0], 2), dtype=np.float64),
        final_highpass_zi=np.zeros((mode_bank.final_highpass_sos.shape[0], 2), dtype=np.float64),
        final_lowpass_zi=np.zeros((mode_bank.final_lowpass_sos.shape[0], 2), dtype=np.float64),
    )


def reinitialize_mode_state(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    acoustic_profile: AcousticProfile,
) -> AudioRenderState:
    fresh_state = initialize_render_state(state.fs, mode_bank, acoustic_profile)
    fresh_state.sample_clock = state.sample_clock
    return fresh_state


def _derive_transfer_activity(event: StorageEvent) -> float:
    if event.transfer_activity > 0.0:
        return event.transfer_activity
    motion_activity = 0.22 + 0.44 * _normalized_track_delta(event)
    sequential_bias = 0.20 if event.is_sequential else 0.0
    flush_bias = 0.10 if event.is_flush else 0.0
    queue_scale = 1.0 + 0.04 * max(event.queue_depth - 1, 0)
    return (motion_activity + sequential_bias + flush_bias) * queue_scale


def _normalized_track_delta(event: StorageEvent) -> float:
    if event.track_delta != 0.0:
        return abs(event.track_delta)
    if event.seek_distance <= 0.0:
        return 0.0
    return min(max(event.seek_distance / 1200.0, 0.0), 1.0)


def _derive_servo_mode(event: StorageEvent) -> str:
    if event.servo_mode:
        return event.servo_mode
    if event.track_delta > 0.0 or event.seek_distance > 0.0:
        return "seek"
    if event.is_sequential or event.transfer_activity > 0.18:
        return "track"
    return "idle"


def _command_frequency(duration_s: float, damping_ratio: float) -> float:
    if duration_s <= 0.0:
        return 70.0
    return _clamp(4.0 / (duration_s * 2.0 * math.pi * damping_ratio), 40.0, 520.0)


def _command_target_position(
    state: AudioRenderState,
    normalized_delta: float,
    command_mode: str,
) -> tuple[float, float]:
    direction = -state.actuator_direction
    if command_mode == "park":
        return -1.08, -1.0
    if command_mode == "calibration":
        return _clamp(state.actuator_target_pos + direction * 0.08, -0.92, 0.92), direction
    span = 0.07 + 0.93 * math.sqrt(max(normalized_delta, 0.0))
    return _clamp(state.actuator_target_pos + direction * span, -0.96, 0.96), direction


def _apply_motion_command(
    state: AudioRenderState,
    event: StorageEvent,
    plant_config: MechanicalPlantConfig,
    command_mode: str,
) -> AudioRenderState:
    normalized_delta = _normalized_track_delta(event)
    minimum_duration_s = max(
        max(event.motion_duration_ms, event.actuator_duration_ms) / 1000.0,
        0.0012 + 0.0045 * math.sqrt(max(normalized_delta, 0.0)),
    )
    settle_s = max(
        max(event.settle_duration_ms, event.actuator_settle_ms) / 1000.0,
        0.0010 + 0.0020 * math.sqrt(max(normalized_delta, 0.0)),
    )
    target_pos, direction = _command_target_position(state, normalized_delta, command_mode)
    seek_plan = plan_seek_motion(
        state.actuator_pos,
        target_pos,
        minimum_duration_s=minimum_duration_s,
        settle_s=settle_s,
        max_velocity=plant_config.actuator_max_velocity,
        max_accel=plant_config.actuator_max_accel,
        max_jerk=plant_config.actuator_max_jerk,
    )

    seek_hz = _command_frequency(seek_plan.duration_s, 0.82)
    track_hz = _clamp(_command_frequency(settle_s * 1.2, 0.95), 28.0, seek_hz)
    damping = 0.92 if command_mode == "park" else 0.86 if command_mode == "calibration" else 0.82
    command_energy = 0.18
    if command_mode == "seek":
        command_energy += 0.36 + 1.10 * normalized_delta
    elif command_mode == "calibration":
        command_energy += 0.42
    else:
        command_energy += 0.60

    return replace(
        state,
        heads_loaded=state.heads_loaded if command_mode == "park" else True,
        servo_mode=command_mode,
        actuator_target_pos=target_pos,
        actuator_seek_hz=seek_hz,
        actuator_track_hz=track_hz,
        actuator_damping_ratio=damping,
        actuator_direction=direction,
        seek_plan=seek_plan,
        seek_plan_elapsed_s=0.0,
        seek_time_remaining=seek_plan.duration_s,
        settle_time_remaining=settle_s,
        servo_tick_energy=max(state.servo_tick_energy * 0.5, command_energy),
        servo_tick_cooldown_s=0.0,
    )


def apply_event(
    state: AudioRenderState,
    event: StorageEvent,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    start_frame: int = 0,
) -> AudioRenderState:
    del start_frame
    target_rpm = event.target_rpm if event.target_rpm is not None else event.rpm
    command_mode = _derive_servo_mode(event)
    heads_loaded = state.heads_loaded if event.heads_loaded is None or command_mode == "park" else event.heads_loaded
    next_state = replace(
        state,
        target_rpm=target_rpm,
        queue_depth=max(1, event.queue_depth),
        op_kind=event.op_kind,
        is_flush=event.is_flush,
        is_spinup=event.is_spinup,
        is_sequential=event.is_sequential,
        heads_loaded=heads_loaded,
        transfer_activity=_derive_transfer_activity(event),
    )
    if not event.is_spinup and target_rpm > 0.0 and state.spindle_omega <= 0.0:
        target_omega = target_rpm * 2.0 * math.pi / 60.0
        config = mode_bank.plant_config
        steady_torque = (
            config.spindle_drag_linear * target_omega
            + config.spindle_drag_quadratic * target_omega * target_omega
        )
        next_state = replace(
            next_state,
            spindle_omega=target_omega,
            spindle_torque=steady_torque,
            heads_loaded=True,
        )

    if command_mode in {"seek", "park", "calibration"}:
        commanded = _apply_motion_command(next_state, event, mode_bank.plant_config, command_mode)
        unpark_impulse = 0.0
        if not state.heads_loaded and commanded.heads_loaded:
            unpark_impulse = 0.24 if command_mode == "calibration" else 0.38
        if unpark_impulse > 0.0:
            commanded = replace(commanded, pending_contact_impulse=unpark_impulse)
        return commanded
    if command_mode == "track":
        track_hz = _clamp(
            0.72 * _command_frequency(max(next_state.settle_time_remaining, drive_profile.settle_ms / 1000.0), 0.95),
            28.0,
            180.0,
        )
        return replace(
            next_state,
            heads_loaded=True,
            servo_mode="track",
            actuator_track_hz=track_hz,
            seek_plan=None,
            seek_plan_elapsed_s=0.0,
        )
    return replace(next_state, servo_mode="idle", seek_plan=None, seek_plan_elapsed_s=0.0)


def _sosfilt(
    sos: FloatArray,
    data: FloatArray,
    zi: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if len(data) == 0:
        return np.zeros(0, dtype=np.float64), zi
    filtered, next_zi = signal.sosfilt(sos, data, zi=zi)
    return np.asarray(filtered, dtype=np.float64), np.asarray(next_zi, dtype=np.float64)


def _simulate_modal_bank(
    force: FloatArray,
    bank: DiscreteModalBank,
    disp_state: FloatArray,
    vel_state: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    disp = np.array(disp_state, copy=True)
    vel = np.array(vel_state, copy=True)
    response = np.zeros(len(force), dtype=np.float64)
    for index, force_sample in enumerate(force):
        next_disp = bank.a11 * disp + bank.a12 * vel + bank.b1 * force_sample
        next_vel = bank.a21 * disp + bank.a22 * vel + bank.b2 * force_sample
        disp = next_disp
        vel = next_vel
        response[index] = float(np.dot(bank.gains, vel))
    return response, disp, vel


def _simulate_coupled_structure(
    spindle_force: FloatArray,
    actuator_force: FloatArray,
    enclosure_force: FloatArray,
    model: CoupledStructureModel,
    structure_state: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    state = np.array(structure_state, copy=True)
    response = np.zeros(len(spindle_force), dtype=np.float64)
    velocity_components = np.zeros((len(spindle_force), 4), dtype=np.float64)
    for index, (spindle_sample, actuator_sample, enclosure_sample) in enumerate(
        zip(spindle_force, actuator_force, enclosure_force, strict=True)
    ):
        inputs = np.array([spindle_sample, actuator_sample, enclosure_sample], dtype=np.float64)
        state = model.ad @ state + model.bd @ inputs
        velocity_components[index] = state[4:]
        response[index] = float(np.dot(model.velocity_readout, state[4:]))
    return response, state, velocity_components


def _render_segment_internal(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    frames: int,
    bearing_noise_raw: FloatArray,
    windage_noise_raw: FloatArray,
    *,
    capture_diagnostics: bool,
) -> tuple[AudioRenderState, FloatArray, AudioDiagnosticTrace | None]:
    if frames <= 0:
        empty = np.zeros(0, dtype=np.float64)
        diagnostics = None
        if capture_diagnostics:
            diagnostics = AudioDiagnosticTrace(
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
        return state, empty, diagnostics

    dt = 1.0 / state.fs
    config = mode_bank.plant_config
    theta = state.spindle_angle
    omega = state.spindle_omega
    spindle_torque = state.spindle_torque
    spindle_integrator = state.spindle_integrator
    servo_phase = state.servo_tick_phase
    actuator_pos = state.actuator_pos
    actuator_vel = state.actuator_vel
    actuator_torque = state.actuator_torque
    actuator_torque_cmd = state.actuator_torque_cmd
    actuator_integrator = state.actuator_integrator
    pending_contact_impulse = state.pending_contact_impulse
    servo_tick_energy = state.servo_tick_energy
    servo_tick_cooldown_s = state.servo_tick_cooldown_s
    seek_plan = state.seek_plan
    seek_plan_elapsed_s = state.seek_plan_elapsed_s
    seek_time_remaining = state.seek_time_remaining
    settle_time_remaining = state.settle_time_remaining
    servo_mode = state.servo_mode
    heads_loaded = state.heads_loaded

    omega_trace = np.zeros(frames, dtype=np.float64)
    theta_trace = np.zeros(frames, dtype=np.float64)
    spindle_reaction = np.zeros(frames, dtype=np.float64)
    actuator_reaction = np.zeros(frames, dtype=np.float64)
    actuator_jerk = np.zeros(frames, dtype=np.float64)
    track_error_trace = np.zeros(frames, dtype=np.float64)
    servo_wedge_trace = np.zeros(frames, dtype=np.float64)
    servo_wedge_structure_trace = np.zeros(frames, dtype=np.float64)
    contact_trace = np.zeros(frames, dtype=np.float64)
    actuator_pos_trace = np.zeros(frames, dtype=np.float64)
    actuator_torque_trace = np.zeros(frames, dtype=np.float64)
    target_rpm_trace = np.zeros(frames, dtype=np.float64)

    prev_actuator_torque = actuator_torque
    nominal_omega = drive_profile.rpm * 2.0 * math.pi / 60.0
    target_omega = max(state.target_rpm, 0.0) * 2.0 * math.pi / 60.0
    if pending_contact_impulse > 0.0 and frames > 0:
        contact_trace[0] += pending_contact_impulse * acoustic_profile.impulse_gain
        pending_contact_impulse = 0.0
    for index in range(frames):
        if servo_tick_cooldown_s > 0.0:
            servo_tick_cooldown_s = max(0.0, servo_tick_cooldown_s - dt)
        target_omega = max(state.target_rpm, 0.0) * 2.0 * math.pi / 60.0
        omega_error = target_omega - omega
        spindle_integrator = _clamp(spindle_integrator + omega_error * dt, -target_omega, target_omega)
        spindle_torque_cmd = _clamp(
            config.spindle_kp * omega_error + config.spindle_ki * spindle_integrator,
            -config.spindle_torque_max,
            config.spindle_torque_max,
        )
        spindle_torque += (spindle_torque_cmd - spindle_torque) * min(
            dt / config.spindle_torque_time_constant_s,
            1.0,
        )
        drag = config.spindle_drag_linear * omega + config.spindle_drag_quadratic * omega * abs(omega)
        omega = max(0.0, omega + ((spindle_torque - drag) / config.spindle_inertia) * dt)
        theta = (theta + omega * dt) % (2.0 * math.pi)

        rev_hz = omega / (2.0 * math.pi)
        servo_phase += rev_hz * config.servo_sectors_per_rev * dt
        if seek_time_remaining > 0.0:
            seek_time_remaining = max(0.0, seek_time_remaining - dt)
            seek_plan_elapsed_s += dt
        if settle_time_remaining > 0.0:
            settle_time_remaining = max(0.0, settle_time_remaining - dt)

        while heads_loaded and servo_phase >= 1.0:
            servo_phase -= 1.0
            prev_torque_cmd = actuator_torque_cmd
            disturbance = (
                config.rro_1x * math.sin(theta + 0.3)
                + config.rro_2x * math.sin(2.0 * theta + 1.1)
                + 0.00012 * state.transfer_activity * math.sin(12.0 * theta)
                + 0.00006 * state.transfer_activity * math.sin(23.0 * theta + 0.6)
            )
            measurement = actuator_pos + disturbance
            reference_pos = state.actuator_target_pos
            reference_vel = 0.0
            reference_accel = 0.0
            if servo_mode in {"seek", "calibration", "park"} and seek_plan is not None:
                reference_pos, reference_vel, reference_accel = sample_seek_reference(
                    seek_plan,
                    seek_plan_elapsed_s,
                )
            error = reference_pos - measurement
            track_error_trace[index] = error
            actuator_integrator = _clamp(actuator_integrator + error * dt, -0.35, 0.35)

            if servo_mode in {"seek", "calibration", "park"}:
                control_hz = state.actuator_seek_hz
                damping = state.actuator_damping_ratio
            else:
                control_hz = state.actuator_track_hz
                damping = 0.95
            wn = 2.0 * math.pi * control_hz
            desired_accel = reference_accel + wn * wn * error + 2.0 * damping * wn * (reference_vel - actuator_vel)
            if servo_mode == "track":
                desired_accel += config.track_integral_gain * actuator_integrator
            actuator_torque_cmd = _clamp(
                config.actuator_inertia * desired_accel,
                -config.actuator_torque_max,
                config.actuator_torque_max,
            )
            command_step = actuator_torque_cmd - prev_torque_cmd
            wedge_scale = abs(command_step) / max(config.actuator_torque_max, 1e-9)
            error_scale = min(abs(error) / 0.14, 1.8)
            transfer_scale = 0.25 * state.transfer_activity
            if servo_mode == "seek":
                mode_scale = 1.0
            elif servo_mode == "track":
                mode_scale = 0.42
            elif servo_mode == "calibration":
                mode_scale = 1.25
            else:
                mode_scale = 1.55
            # Servo wedges arrive far above the audible "tick" rate. A physically
            # plausible drive radiates the accumulated correction effort through the
            # actuator/pivot/base path, not one discrete click per wedge.
            command_reversal = 1.0 if actuator_torque_cmd * prev_torque_cmd < 0.0 else 0.0
            correction_drive = max(0.0, wedge_scale - 0.012) * 3.8
            correction_drive += max(0.0, error_scale - 0.08) * 0.30
            correction_drive += transfer_scale * (
                acoustic_profile.sequential_boundary_gain if servo_mode == "track" else 0.55
            )
            correction_drive += command_reversal * (0.34 if servo_mode == "seek" else 0.18)
            if servo_mode == "track":
                correction_drive *= 0.08 * acoustic_profile.sequential_boundary_gain
                tick_threshold = 0.62
                min_interval_s = 0.0030
            elif servo_mode == "seek":
                correction_drive *= 1.22
                tick_threshold = 0.20
                min_interval_s = 0.00085
            elif servo_mode == "calibration":
                correction_drive *= 1.30
                tick_threshold = 0.18
                min_interval_s = 0.00065
            else:
                correction_drive *= 1.55
                tick_threshold = 0.15
                min_interval_s = 0.00045

            servo_tick_energy = min(2.8, servo_tick_energy + correction_drive)
            if servo_tick_cooldown_s <= 0.0 and servo_tick_energy >= tick_threshold:
                wedge_mag = acoustic_profile.impulse_gain * mode_scale * (
                    0.045 + 0.160 * min(servo_tick_energy, 2.1)
                )
                wedge_sign_source = command_step if abs(command_step) > 1e-9 else error
                wedge_sign = math.copysign(1.0, wedge_sign_source if abs(wedge_sign_source) > 1e-9 else 1.0)
                servo_wedge_trace[index] += wedge_sign * wedge_mag
                servo_wedge_structure_trace[index] += wedge_sign * wedge_mag * (2.00 if servo_mode == "park" else 1.42)
                servo_tick_energy *= 0.16 if servo_mode == "track" else 0.22
                servo_tick_cooldown_s = min_interval_s

        actuator_torque += (actuator_torque_cmd - actuator_torque) * min(
            dt / config.actuator_torque_time_constant_s,
            1.0,
        )
        actuator_accel = (actuator_torque - config.actuator_damping * actuator_vel) / config.actuator_inertia
        actuator_vel += actuator_accel * dt
        actuator_pos += actuator_vel * dt
        actuator_pos = _clamp(actuator_pos, -1.1, 1.1)

        if (
            servo_mode in {"seek", "calibration", "park"}
            and abs(state.actuator_target_pos - actuator_pos) < 0.018
            and abs(actuator_vel) < 2.8
            and settle_time_remaining <= 0.0
        ):
            if servo_mode == "park":
                contact_trace[index] += acoustic_profile.impulse_gain * math.copysign(
                    1.45,
                    actuator_vel if abs(actuator_vel) > 1e-9 else -state.actuator_direction,
                )
            servo_mode = "idle" if state.target_rpm <= 0.0 or servo_mode == "park" else "track"
            if servo_mode == "idle":
                heads_loaded = False
            seek_plan = None
            seek_plan_elapsed_s = 0.0

        theta_trace[index] = theta
        omega_trace[index] = omega
        spindle_reaction[index] = spindle_torque
        actuator_reaction[index] = actuator_torque
        actuator_jerk[index] = (actuator_torque - prev_actuator_torque) / dt
        actuator_pos_trace[index] = actuator_pos
        actuator_torque_trace[index] = actuator_torque
        target_rpm_trace[index] = target_omega * 60.0 / (2.0 * math.pi)
        prev_actuator_torque = actuator_torque

    bearing_colored, bearing_zi = _sosfilt(mode_bank.bearing_sos, bearing_noise_raw, state.bearing_zi)
    windage_colored, windage_zi = _sosfilt(mode_bank.windage_sos, windage_noise_raw, state.windage_zi)

    omega_ratio = np.clip(omega_trace / max(nominal_omega, 1e-6), 0.0, 1.5)
    imbalance = np.zeros(frames, dtype=np.float64)
    for harmonic, amplitude in zip(
        drive_profile.spindle_harmonics,
        drive_profile.spindle_harmonic_weights,
        strict=True,
    ):
        imbalance += amplitude * np.sin(theta_trace * harmonic)
    imbalance *= 0.0105 * omega_ratio * omega_ratio

    commutation = (
        0.62 * np.sin(theta_trace * 6.0 + 0.2)
        + 0.28 * np.sin(theta_trace * 12.0 + 1.0)
        + 0.12 * np.sin(theta_trace * 18.0 + 2.1)
    )
    commutation *= (0.0018 + 0.0046 * np.clip(np.abs(spindle_reaction) / max(config.spindle_torque_max, 1e-6), 0.0, 1.5))

    bearing_force = bearing_colored * (
        (0.00025 * omega_ratio + 0.0012 * omega_ratio**1.5)
        * drive_profile.bearing_gain
    )
    windage_force = windage_colored * (
        (0.0009 * omega_ratio**1.2 + 0.0046 * omega_ratio**2.15)
        * drive_profile.windage_gain
    )
    if state.is_spinup:
        windage_force *= 1.16

    direct_force = imbalance * 0.72 + commutation * 0.58 + bearing_force * 0.34
    platter_force = imbalance * 0.30 + windage_force * 0.96 + 0.10 * commutation
    cover_force = imbalance * 0.18 + windage_force * 0.30 + 0.18 * commutation + 0.24 * servo_wedge_trace
    actuator_force = (
        0.18 * actuator_reaction
        + 0.00008 * actuator_jerk
        + 0.075 * track_error_trace
        + 1.05 * servo_wedge_trace
        + 0.62 * contact_trace
    )
    structure_force = drive_profile.boundary_excitation_gain * (
        0.18 * imbalance
        + 0.85 * spindle_reaction
        + 1.45 * servo_wedge_structure_trace
        + 1.65 * contact_trace
    )
    structure_actuator_force = drive_profile.boundary_excitation_gain * (
        1.12 * actuator_reaction
        + 0.00022 * actuator_jerk
        + 1.85 * servo_wedge_structure_trace
        + 2.40 * contact_trace
    )
    enclosure_internal_force = drive_profile.boundary_excitation_gain * (
        0.62 * windage_force
        + 0.24 * cover_force
        + 0.16 * platter_force
        + 0.08 * bearing_force
    )

    platter, platter_disp, platter_vel = _simulate_modal_bank(
        platter_force,
        mode_bank.platter_step,
        state.platter_disp,
        state.platter_vel,
    )
    cover, cover_disp, cover_vel = _simulate_modal_bank(
        cover_force,
        mode_bank.cover_step,
        state.cover_disp,
        state.cover_vel,
    )
    actuator_modes, actuator_disp, actuator_vel_modes = _simulate_modal_bank(
        actuator_force,
        mode_bank.actuator_step,
        state.actuator_disp,
        state.actuator_vel_modes,
    )
    structure_modes, structure_disp, structure_vel = _simulate_modal_bank(
        structure_force + 0.18 * structure_actuator_force,
        mode_bank.structure_step,
        state.structure_disp,
        state.structure_vel,
    )
    coupled_structure, structure_state, structure_velocity_components = _simulate_coupled_structure(
        structure_force,
        structure_actuator_force,
        enclosure_internal_force,
        mode_bank.coupled_structure,
        state.structure_state,
    )

    voices: tuple[AudioVoice, ...] = build_voices(
        direct_force=direct_force,
        platter=platter,
        cover=cover,
        actuator=actuator_modes,
        structure_modes=structure_modes,
        coupled_structure=coupled_structure,
        acoustic_profile=acoustic_profile,
    )
    airborne = mix_voice_path(voices, "airborne")
    structure = mix_voice_path(voices, "structure")
    radiated = airborne + structure * acoustic_profile.structure_gain
    highpassed, final_highpass_zi = _sosfilt(
        mode_bank.final_highpass_sos,
        radiated,
        state.final_highpass_zi,
    )
    lowpassed, final_lowpass_zi = _sosfilt(
        mode_bank.final_lowpass_sos,
        highpassed,
        state.final_lowpass_zi,
    )

    next_state = replace(
        state,
        sample_clock=state.sample_clock + frames,
        spindle_angle=theta,
        spindle_omega=omega,
        spindle_torque=spindle_torque,
        spindle_integrator=spindle_integrator,
        servo_tick_phase=servo_phase,
        actuator_pos=actuator_pos,
        actuator_vel=actuator_vel,
        actuator_torque=actuator_torque,
        actuator_torque_cmd=actuator_torque_cmd,
        actuator_integrator=actuator_integrator,
        seek_plan=seek_plan,
        seek_plan_elapsed_s=seek_plan_elapsed_s,
        seek_time_remaining=seek_time_remaining,
        settle_time_remaining=settle_time_remaining,
        servo_mode=servo_mode,
        heads_loaded=heads_loaded,
        pending_contact_impulse=pending_contact_impulse,
        servo_tick_energy=servo_tick_energy,
        servo_tick_cooldown_s=servo_tick_cooldown_s,
        platter_disp=platter_disp,
        platter_vel=platter_vel,
        cover_disp=cover_disp,
        cover_vel=cover_vel,
        actuator_disp=actuator_disp,
        actuator_vel_modes=actuator_vel_modes,
        structure_disp=structure_disp,
        structure_vel=structure_vel,
        structure_state=structure_state,
        bearing_zi=bearing_zi,
        windage_zi=windage_zi,
        final_highpass_zi=final_highpass_zi,
        final_lowpass_zi=final_lowpass_zi,
    )
    output = np.tanh(lowpassed * 0.95) * acoustic_profile.output_gain
    diagnostics = None
    if capture_diagnostics:
        time_s = (np.arange(frames, dtype=np.float64) + state.sample_clock) / state.fs
        diagnostics = AudioDiagnosticTrace(
            time_s=time_s,
            target_rpm=target_rpm_trace,
            actual_rpm=omega_trace * 60.0 / (2.0 * math.pi),
            actuator_pos=actuator_pos_trace,
            actuator_torque=actuator_torque_trace,
            structure_base_velocity=structure_velocity_components[:, 0],
            structure_cover_velocity=structure_velocity_components[:, 1],
            structure_enclosure_velocity=structure_velocity_components[:, 2],
            structure_desk_velocity=structure_velocity_components[:, 3],
            output=output,
        )
    return next_state, output, diagnostics


def _render_segment(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    frames: int,
    bearing_noise_raw: FloatArray,
    windage_noise_raw: FloatArray,
) -> tuple[AudioRenderState, FloatArray]:
    next_state, output, _ = _render_segment_internal(
        state,
        mode_bank,
        drive_profile,
        acoustic_profile,
        frames,
        bearing_noise_raw,
        windage_noise_raw,
        capture_diagnostics=False,
    )
    return next_state, output


def _concatenate_diagnostics(segments: Sequence[AudioDiagnosticTrace]) -> AudioDiagnosticTrace:
    if not segments:
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
    return AudioDiagnosticTrace(
        time_s=np.concatenate([segment.time_s for segment in segments]),
        target_rpm=np.concatenate([segment.target_rpm for segment in segments]),
        actual_rpm=np.concatenate([segment.actual_rpm for segment in segments]),
        actuator_pos=np.concatenate([segment.actuator_pos for segment in segments]),
        actuator_torque=np.concatenate([segment.actuator_torque for segment in segments]),
        structure_base_velocity=np.concatenate([segment.structure_base_velocity for segment in segments]),
        structure_cover_velocity=np.concatenate([segment.structure_cover_velocity for segment in segments]),
        structure_enclosure_velocity=np.concatenate([segment.structure_enclosure_velocity for segment in segments]),
        structure_desk_velocity=np.concatenate([segment.structure_desk_velocity for segment in segments]),
        output=np.concatenate([segment.output for segment in segments]),
    )


def render_diagnostic_chunk(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    frames: int,
    *,
    scheduled_events: Sequence[ScheduledEvent] = (),
    bearing_noise_raw: FloatArray | None = None,
    windage_noise_raw: FloatArray | None = None,
) -> tuple[AudioRenderState, FloatArray, AudioDiagnosticTrace]:
    if frames <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return (
            state,
            empty,
            AudioDiagnosticTrace(
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
            ),
        )

    bearing_noise = (
        np.zeros(frames, dtype=np.float64)
        if bearing_noise_raw is None
        else np.asarray(bearing_noise_raw, dtype=np.float64)
    )
    windage_noise = (
        np.zeros(frames, dtype=np.float64)
        if windage_noise_raw is None
        else np.asarray(windage_noise_raw, dtype=np.float64)
    )

    if not scheduled_events:
        next_state, output, diagnostics = _render_segment_internal(
            state,
            mode_bank,
            drive_profile,
            acoustic_profile,
            frames,
            bearing_noise[:frames],
            windage_noise[:frames],
            capture_diagnostics=True,
        )
        assert diagnostics is not None
        return next_state, output, diagnostics

    output = np.zeros(frames, dtype=np.float64)
    cursor = 0
    next_state = state
    diagnostic_segments: list[AudioDiagnosticTrace] = []
    sorted_events = sorted(
        ((event, max(0, min(frames, int(start_frame)))) for event, start_frame in scheduled_events),
        key=lambda item: item[1],
    )
    event_index = 0

    while event_index < len(sorted_events):
        frame_offset = sorted_events[event_index][1]
        if frame_offset > cursor:
            next_state, segment, diagnostics = _render_segment_internal(
                next_state,
                mode_bank,
                drive_profile,
                acoustic_profile,
                frame_offset - cursor,
                bearing_noise[cursor:frame_offset],
                windage_noise[cursor:frame_offset],
                capture_diagnostics=True,
            )
            output[cursor:frame_offset] = segment
            assert diagnostics is not None
            diagnostic_segments.append(diagnostics)
            cursor = frame_offset

        while event_index < len(sorted_events) and sorted_events[event_index][1] == frame_offset:
            event, _ = sorted_events[event_index]
            next_state = apply_event(next_state, event, mode_bank, drive_profile)
            event_index += 1

    if cursor < frames:
        next_state, segment, diagnostics = _render_segment_internal(
            next_state,
            mode_bank,
            drive_profile,
            acoustic_profile,
            frames - cursor,
            bearing_noise[cursor:frames],
            windage_noise[cursor:frames],
            capture_diagnostics=True,
        )
        output[cursor:] = segment
        assert diagnostics is not None
        diagnostic_segments.append(diagnostics)

    diagnostics = _concatenate_diagnostics(diagnostic_segments)
    return next_state, output, diagnostics


def render_chunk(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    frames: int,
    *,
    scheduled_events: Sequence[ScheduledEvent] = (),
    bearing_noise_raw: FloatArray | None = None,
    windage_noise_raw: FloatArray | None = None,
) -> tuple[AudioRenderState, FloatArray]:
    if frames <= 0:
        return state, np.zeros(0, dtype=np.float64)

    bearing_noise = (
        np.zeros(frames, dtype=np.float64)
        if bearing_noise_raw is None
        else np.asarray(bearing_noise_raw, dtype=np.float64)
    )
    windage_noise = (
        np.zeros(frames, dtype=np.float64)
        if windage_noise_raw is None
        else np.asarray(windage_noise_raw, dtype=np.float64)
    )

    if not scheduled_events:
        return _render_segment(
            state,
            mode_bank,
            drive_profile,
            acoustic_profile,
            frames,
            bearing_noise[:frames],
            windage_noise[:frames],
        )

    output = np.zeros(frames, dtype=np.float64)
    cursor = 0
    next_state = state
    sorted_events = sorted(
        ((event, max(0, min(frames, int(start_frame)))) for event, start_frame in scheduled_events),
        key=lambda item: item[1],
    )
    event_index = 0

    while event_index < len(sorted_events):
        frame_offset = sorted_events[event_index][1]
        if frame_offset > cursor:
            next_state, segment = _render_segment(
                next_state,
                mode_bank,
                drive_profile,
                acoustic_profile,
                frame_offset - cursor,
                bearing_noise[cursor:frame_offset],
                windage_noise[cursor:frame_offset],
            )
            output[cursor:frame_offset] = segment
            cursor = frame_offset

        while event_index < len(sorted_events) and sorted_events[event_index][1] == frame_offset:
            event, _ = sorted_events[event_index]
            next_state = apply_event(next_state, event, mode_bank, drive_profile)
            event_index += 1

    if cursor < frames:
        next_state, segment = _render_segment(
            next_state,
            mode_bank,
            drive_profile,
            acoustic_profile,
            frames - cursor,
            bearing_noise[cursor:frames],
            windage_noise[cursor:frames],
        )
        output[cursor:] = segment

    return next_state, output
