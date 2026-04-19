from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from ..profiles import AcousticProfile, DriveProfile
from ..storage_events import ScheduledStorageEvent, StorageEvent


FloatArray = npt.NDArray[np.float64]
ScheduledEvent = ScheduledStorageEvent


@dataclass(frozen=True)
class ScheduledPulse:
    start_frame: int
    width: int
    amplitude: float
    phase_offset: int = 0


@dataclass(frozen=True)
class MechanicalMode:
    """
    Reduced-order mode for a vibrating HDD substructure.

    The synth treats each subsystem as a bank of unit-mass second-order
    oscillators: x'' + 2*zeta*wn*x' + wn^2*x = F(t).
    """

    name: str
    frequency_hz: float
    damping_ratio: float
    radiation_gain: float


@dataclass(frozen=True)
class AudioModeBank:
    platter_modes: tuple[MechanicalMode, ...]
    cover_modes: tuple[MechanicalMode, ...]
    actuator_modes: tuple[MechanicalMode, ...]
    platter_wn: FloatArray
    platter_zeta: FloatArray
    platter_gain: FloatArray
    cover_wn: FloatArray
    cover_zeta: FloatArray
    cover_gain: FloatArray
    actuator_wn: FloatArray
    actuator_zeta: FloatArray
    actuator_gain: FloatArray


@dataclass
class AudioRenderState:
    fs: int
    rpm: float = 0.0
    is_sequential: bool = False
    queue_depth: int = 1
    op_kind: str = "data"
    is_flush: bool = False
    is_spinup: bool = False
    sample_clock: int = 0
    flutter_phase: float = 0.0
    windage_state: float = 0.0
    bearing_state: float = 0.0
    air_turbulence_low_state: float = 0.0
    air_turbulence_high_state: float = 0.0
    final_lowpass_state: float = 0.0
    final_highpass_state: float = 0.0
    output_gain: float = 0.88
    pending_impulses: tuple[ScheduledPulse, ...] = ()
    platter_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    platter_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    cover_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    cover_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    actuator_disp: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    actuator_vel: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


BASE_PLATTER_MODES: tuple[tuple[str, float, float, float], ...] = (
    ("platter-1", 637.5, 0.018, 0.21),
    ("platter-2", 737.5, 0.018, 0.18),
    ("platter-3", 1013.0, 0.022, 0.16),
    ("platter-4", 1838.0, 0.028, 0.09),
    ("platter-5", 2100.0, 0.030, 0.07),
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
    ("actuator-1", 87.5, 0.070, 0.05),
    ("actuator-2", 1100.0, 0.030, 0.38),
    ("actuator-3", 1450.0, 0.028, 0.44),
    ("actuator-4", 1700.0, 0.026, 0.40),
    ("actuator-5", 1850.0, 0.026, 0.34),
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


def build_mode_bank(drive_profile: DriveProfile) -> AudioModeBank:
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
    return AudioModeBank(
        platter_modes=platter_modes,
        cover_modes=cover_modes,
        actuator_modes=actuator_modes,
        platter_wn=2.0
        * np.pi
        * np.array([mode.frequency_hz for mode in platter_modes], dtype=np.float64),
        platter_zeta=np.array([mode.damping_ratio for mode in platter_modes], dtype=np.float64),
        platter_gain=np.array([mode.radiation_gain for mode in platter_modes], dtype=np.float64),
        cover_wn=2.0 * np.pi * np.array([mode.frequency_hz for mode in cover_modes], dtype=np.float64),
        cover_zeta=np.array([mode.damping_ratio for mode in cover_modes], dtype=np.float64),
        cover_gain=np.array([mode.radiation_gain for mode in cover_modes], dtype=np.float64),
        actuator_wn=2.0
        * np.pi
        * np.array([mode.frequency_hz for mode in actuator_modes], dtype=np.float64),
        actuator_zeta=np.array([mode.damping_ratio for mode in actuator_modes], dtype=np.float64),
        actuator_gain=np.array([mode.radiation_gain for mode in actuator_modes], dtype=np.float64),
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
        actuator_vel=np.zeros(len(mode_bank.actuator_modes), dtype=np.float64),
    )


def reinitialize_mode_state(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    acoustic_profile: AcousticProfile,
) -> AudioRenderState:
    return replace(
        state,
        output_gain=acoustic_profile.output_gain,
        platter_disp=np.zeros(len(mode_bank.platter_modes), dtype=np.float64),
        platter_vel=np.zeros(len(mode_bank.platter_modes), dtype=np.float64),
        cover_disp=np.zeros(len(mode_bank.cover_modes), dtype=np.float64),
        cover_vel=np.zeros(len(mode_bank.cover_modes), dtype=np.float64),
        actuator_disp=np.zeros(len(mode_bank.actuator_modes), dtype=np.float64),
        actuator_vel=np.zeros(len(mode_bank.actuator_modes), dtype=np.float64),
    )


def make_impulse_pulses(
    impulse: str,
    start_frame: int,
    seek_distance: float,
    op_kind: str,
    is_flush: bool,
    sample_rate: int,
    acoustic_profile: AcousticProfile,
) -> tuple[ScheduledPulse, ...]:
    pulses: list[ScheduledPulse] = []
    start = max(0, int(start_frame))
    if impulse == "park":
        pulses.append(ScheduledPulse(start, max(6, int(0.0012 * sample_rate)), 0.95))
        pulses.append(
            ScheduledPulse(
                start + int(0.0009 * sample_rate),
                max(5, int(0.0009 * sample_rate)),
                -0.62,
            )
        )
        return tuple(pulses)
    if impulse == "calibration":
        pulses.append(ScheduledPulse(start, max(4, int(0.00045 * sample_rate)), 0.12))
        pulses.append(
            ScheduledPulse(
                start + int(0.00045 * sample_rate),
                max(4, int(0.00035 * sample_rate)),
                -0.05,
            )
        )
        return tuple(pulses)
    if impulse != "seek":
        return ()

    kind_scale = {
        "metadata": 0.36,
        "journal": 0.48,
        "flush": 0.68,
        "data": 0.78,
    }.get(op_kind, 0.55)
    stroke = min(max(seek_distance / 1000.0, 0.0), 1.0)
    force_scale = kind_scale * (0.30 + 0.62 * np.sqrt(stroke)) * acoustic_profile.impulse_gain
    accel_width = max(4, int(sample_rate * (0.00016 + 0.00022 * stroke)))
    brake_delay = max(5, int(sample_rate * (0.00020 + 0.00018 * stroke)))
    brake_width = max(5, int(accel_width * 0.85))
    settle_width = max(4, int(accel_width * 0.55))
    pulses.append(ScheduledPulse(start, accel_width, force_scale))
    pulses.append(ScheduledPulse(start + brake_delay, brake_width, -force_scale * 0.78))
    pulses.append(
        ScheduledPulse(
            start + brake_delay + int(0.00032 * sample_rate),
            settle_width,
            force_scale * 0.24,
        )
    )
    if is_flush:
        pulses.append(
            ScheduledPulse(
                start + int(0.0007 * sample_rate),
                max(5, int(accel_width * 0.7)),
                force_scale * 0.28,
            )
        )
    return tuple(pulses)


def apply_event(
    state: AudioRenderState,
    event: StorageEvent,
    acoustic_profile: AcousticProfile,
    start_frame: int = 0,
) -> AudioRenderState:
    next_state = replace(
        state,
        rpm=event.rpm,
        is_sequential=event.is_sequential,
        queue_depth=event.queue_depth,
        op_kind=event.op_kind,
        is_flush=event.is_flush,
        is_spinup=event.is_spinup,
    )
    if not event.impulse:
        return next_state

    pulses = make_impulse_pulses(
        impulse=event.impulse,
        start_frame=max(0, int(start_frame)),
        seek_distance=event.seek_distance,
        op_kind=event.op_kind,
        is_flush=event.is_flush,
        sample_rate=state.fs,
        acoustic_profile=acoustic_profile,
    )
    return replace(next_state, pending_impulses=next_state.pending_impulses + pulses)


def _colored_noise(
    raw: FloatArray,
    level: float,
    smoothing: float,
    state: float,
) -> tuple[FloatArray, float]:
    if level <= 0.0:
        return np.zeros(len(raw), dtype=np.float64), state

    output = np.empty(len(raw), dtype=np.float64)
    next_state = state
    for index, sample in enumerate(raw):
        next_state = next_state * smoothing + (sample * level) * (1.0 - smoothing)
        output[index] = next_state
    return output, next_state


def _one_pole_lowpass(
    signal: FloatArray,
    cutoff_hz: float,
    sample_rate: int,
    state: float,
) -> tuple[FloatArray, float]:
    if cutoff_hz <= 0.0:
        return np.zeros_like(signal), state

    smoothing = np.exp(-2.0 * np.pi * cutoff_hz / sample_rate)
    output = np.empty_like(signal)
    next_state = state
    for index, sample in enumerate(signal):
        next_state = next_state * smoothing + sample * (1.0 - smoothing)
        output[index] = next_state
    return output, next_state


def _one_pole_highpass(
    signal: FloatArray,
    cutoff_hz: float,
    sample_rate: int,
    state: float,
) -> tuple[FloatArray, float]:
    if cutoff_hz <= 0.0:
        return np.array(signal, copy=True), state
    lowpassed, next_state = _one_pole_lowpass(signal, cutoff_hz, sample_rate, state)
    return signal - lowpassed, next_state


def _band_limited_noise(
    raw: FloatArray,
    level: float,
    smoothing: float,
    low_cut_hz: float,
    high_cut_hz: float,
    sample_rate: int,
    noise_state: float,
    low_state: float,
    high_state: float,
) -> tuple[FloatArray, float, float, float]:
    if level <= 0.0:
        return np.zeros(len(raw), dtype=np.float64), noise_state, low_state, high_state
    colored, next_noise_state = _colored_noise(raw, level, smoothing, noise_state)
    highpassed, next_high_state = _one_pole_lowpass(colored, high_cut_hz, sample_rate, high_state)
    lowpassed, next_low_state = _one_pole_lowpass(colored, low_cut_hz, sample_rate, low_state)
    return highpassed - lowpassed, next_noise_state, next_low_state, next_high_state


def _time_axis(sample_clock: int, frames: int, sample_rate: int) -> tuple[FloatArray, int]:
    time_axis = (np.arange(frames, dtype=np.float64) + sample_clock) / sample_rate
    return time_axis, sample_clock + frames


def _generate_spindle_forces(
    state: AudioRenderState,
    drive_profile: DriveProfile,
    t: FloatArray,
    bearing_noise_raw: FloatArray,
    windage_noise_raw: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, AudioRenderState]:
    frame_count = len(t)
    rpm = state.rpm
    if rpm <= 0.0:
        return (
            np.zeros(frame_count, dtype=np.float64),
            np.zeros(frame_count, dtype=np.float64),
            np.zeros(frame_count, dtype=np.float64),
            replace(state, flutter_phase=state.flutter_phase + frame_count),
        )

    f0 = rpm / 60.0
    flutter_t = (np.arange(frame_count, dtype=np.float64) + state.flutter_phase) / state.fs
    omega_ratio = min(rpm / max(float(drive_profile.rpm), 1.0), 1.35)
    omega_sq = omega_ratio * omega_ratio
    flutter = 1.0 + 0.02 * np.sin(2 * np.pi * 0.37 * flutter_t) + 0.008 * np.sin(2 * np.pi * 0.91 * flutter_t)
    queue_excitation = 1.0 + 0.04 * max(state.queue_depth - 1, 0)

    imbalance = np.zeros(frame_count, dtype=np.float64)
    for harmonic, amplitude in zip(
        drive_profile.spindle_harmonics,
        drive_profile.spindle_harmonic_weights,
        strict=True,
    ):
        imbalance += amplitude * np.sin(2 * np.pi * (f0 * harmonic) * t)
    imbalance *= 0.012 * omega_sq * flutter

    bearing_level = (0.00055 + 0.0012 * omega_sq) * drive_profile.bearing_gain * (0.96 + 0.04 * queue_excitation)
    bearing, next_bearing_state = _colored_noise(
        bearing_noise_raw,
        bearing_level,
        0.985,
        state.bearing_state,
    )

    windage_level = (0.0018 + 0.0048 * omega_ratio**2.2) * drive_profile.windage_gain * queue_excitation
    if state.is_spinup:
        windage_level *= 1.3
    if state.op_kind in {"journal", "flush"}:
        windage_level *= 1.08
    smoothing = 0.93 if state.is_sequential else 0.88
    windage, next_windage_state, next_low_state, next_high_state = _band_limited_noise(
        windage_noise_raw,
        windage_level,
        smoothing,
        550.0,
        3600.0,
        state.fs,
        state.windage_state,
        state.air_turbulence_low_state,
        state.air_turbulence_high_state,
    )

    direct_radiation = imbalance * 0.72 + bearing * 0.55
    platter_force = imbalance * 0.45 + bearing * 0.18 + windage * 1.1
    cover_force = imbalance * 0.30 + bearing * 0.14 + windage * 0.55
    return (
        direct_radiation,
        platter_force,
        cover_force,
        replace(
            state,
            flutter_phase=state.flutter_phase + frame_count,
            bearing_state=next_bearing_state,
            windage_state=next_windage_state,
            air_turbulence_low_state=next_low_state,
            air_turbulence_high_state=next_high_state,
        ),
    )


def _render_scheduled_pulse(
    actuator_force: FloatArray,
    pulse: ScheduledPulse,
) -> ScheduledPulse | None:
    frame_count = len(actuator_force)
    if pulse.start_frame >= frame_count:
        return replace(pulse, start_frame=pulse.start_frame - frame_count)

    remaining_width = pulse.width - pulse.phase_offset
    if remaining_width <= 0:
        return None

    render_start = max(0, pulse.start_frame)
    render_end = min(frame_count, pulse.start_frame + remaining_width)
    if render_end <= render_start:
        return None

    profile_start = pulse.phase_offset + (render_start - pulse.start_frame)
    sample_indices = np.arange(
        profile_start,
        profile_start + (render_end - render_start),
        dtype=np.float64,
    )
    window = np.sin(np.pi * sample_indices / pulse.width)
    actuator_force[render_start:render_end] += pulse.amplitude * window

    consumed = render_end - pulse.start_frame
    if pulse.start_frame + remaining_width > frame_count:
        return ScheduledPulse(
            start_frame=0,
            width=pulse.width,
            amplitude=pulse.amplitude,
            phase_offset=pulse.phase_offset + max(consumed, 0),
        )
    return None


def _generate_actuator_force(
    state: AudioRenderState,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    t: FloatArray,
) -> tuple[FloatArray, AudioRenderState]:
    frame_count = len(t)
    actuator_force = np.zeros(frame_count, dtype=np.float64)
    carry: list[ScheduledPulse] = []
    for pulse in state.pending_impulses:
        carried_pulse = _render_scheduled_pulse(actuator_force, pulse)
        if carried_pulse is not None:
            carry.append(carried_pulse)

    if state.is_sequential and state.rpm > 0.0:
        boundary_rate_hz = 24.0 + 2.5 * min(state.queue_depth, 6)
        boundary_scale = (
            0.0075
            * drive_profile.boundary_excitation_gain
            * acoustic_profile.sequential_boundary_gain
            * (1.0 + 0.05 * max(state.queue_depth - 1, 0))
        )
        actuator_force += boundary_scale * np.sin(2 * np.pi * boundary_rate_hz * t)

    if state.op_kind in {"journal", "flush"}:
        actuator_force *= 1.08 + 0.03 * max(state.queue_depth - 1, 0)
    elif state.queue_depth > 1:
        actuator_force *= 1.0 + 0.015 * min(state.queue_depth - 1, 8)

    return actuator_force, replace(state, pending_impulses=tuple(carry))


def _simulate_mode_bank(
    force: FloatArray,
    wn: FloatArray,
    zeta: FloatArray,
    gains: FloatArray,
    disp_state: FloatArray,
    vel_state: FloatArray,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    disp = np.array(disp_state, copy=True)
    vel = np.array(vel_state, copy=True)
    response = np.zeros(len(force), dtype=np.float64)
    inv_wn = 1.0 / np.maximum(wn, 1.0)
    for index, force_sample in enumerate(force):
        accel = force_sample - 2.0 * zeta * wn * vel - (wn * wn) * disp
        vel += accel * dt
        disp += vel * dt
        response[index] = float(np.dot(gains, vel * inv_wn))
    return response, disp, vel


def _radiate(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    acoustic_profile: AcousticProfile,
    direct_force: FloatArray,
    platter_force: FloatArray,
    cover_force: FloatArray,
    actuator_force: FloatArray,
) -> tuple[FloatArray, AudioRenderState]:
    dt = 1.0 / state.fs
    platter, platter_disp, platter_vel = _simulate_mode_bank(
        platter_force,
        mode_bank.platter_wn,
        mode_bank.platter_zeta,
        mode_bank.platter_gain,
        state.platter_disp,
        state.platter_vel,
        dt,
    )
    cover, cover_disp, cover_vel = _simulate_mode_bank(
        cover_force,
        mode_bank.cover_wn,
        mode_bank.cover_zeta,
        mode_bank.cover_gain,
        state.cover_disp,
        state.cover_vel,
        dt,
    )
    actuator, actuator_disp, actuator_vel = _simulate_mode_bank(
        actuator_force,
        mode_bank.actuator_wn,
        mode_bank.actuator_zeta,
        mode_bank.actuator_gain,
        state.actuator_disp,
        state.actuator_vel,
        dt,
    )
    radiated = (
        direct_force * acoustic_profile.direct_gain
        + platter * acoustic_profile.platter_gain
        + cover * acoustic_profile.cover_gain
        + actuator * acoustic_profile.actuator_gain
    )
    highpassed, next_high_state = _one_pole_highpass(
        radiated,
        acoustic_profile.final_highpass_hz,
        state.fs,
        state.final_highpass_state,
    )
    lowpassed, next_low_state = _one_pole_lowpass(
        highpassed,
        acoustic_profile.final_lowpass_hz,
        state.fs,
        state.final_lowpass_state,
    )
    next_state = replace(
        state,
        platter_disp=platter_disp,
        platter_vel=platter_vel,
        cover_disp=cover_disp,
        cover_vel=cover_vel,
        actuator_disp=actuator_disp,
        actuator_vel=actuator_vel,
        final_highpass_state=next_high_state,
        final_lowpass_state=next_low_state,
    )
    return lowpassed, next_state


def _render_segment(
    state: AudioRenderState,
    mode_bank: AudioModeBank,
    drive_profile: DriveProfile,
    acoustic_profile: AcousticProfile,
    frames: int,
    bearing_noise_raw: FloatArray,
    windage_noise_raw: FloatArray,
) -> tuple[AudioRenderState, FloatArray]:
    if frames <= 0:
        return state, np.zeros(0, dtype=np.float64)

    time_axis, next_sample_clock = _time_axis(state.sample_clock, frames, state.fs)
    timed_state = replace(state, sample_clock=next_sample_clock)
    direct_force, platter_force, cover_force, spindle_state = _generate_spindle_forces(
        timed_state,
        drive_profile,
        time_axis,
        bearing_noise_raw,
        windage_noise_raw,
    )
    actuator_force, actuator_state = _generate_actuator_force(
        spindle_state,
        drive_profile,
        acoustic_profile,
        time_axis,
    )
    radiated, radiated_state = _radiate(
        actuator_state,
        mode_bank,
        acoustic_profile,
        direct_force,
        platter_force,
        cover_force + actuator_force * 0.62,
        actuator_force,
    )
    return radiated_state, np.tanh(radiated * 1.4) * acoustic_profile.output_gain


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
    event_index = 0
    sorted_events = sorted(
        ((event, max(0, min(frames, int(start_frame)))) for event, start_frame in scheduled_events),
        key=lambda item: item[1],
    )
    next_state = state

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
            next_state = apply_event(next_state, event, acoustic_profile, start_frame=0)
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
