"""Physics-inspired HDD audio primitives with explicit honesty labels.

Model tiers used here:

- Physical state: time-integrated runtime state such as spindle phase/omega,
  actuator position/velocity, and filter/modal states. These are state variables,
  though many use normalized units instead of SI units.
- Plausible model: standard mechanical/audio shapes such as first-order motor
  lag, cosine seek profiles, PID-like servo control, damped resonators, and
  filtered windage/bearing noise.
- Artistic calibration: coupling weights, gain curves, output shaping, and
  profile scales chosen to preserve the current good sound. These are not
  measured HDD constants and should not be presented as reference physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..profiles import AcousticProfile


FloatArray = npt.NDArray[np.float64]
TAU = 2.0 * math.pi
EPS = 1e-9


@dataclass(frozen=True)
class SpindleStep:
    motor_drive: float
    spindle_omega: float
    phase_increment: float
    rpm_norm: float
    motor_reaction: float


@dataclass(frozen=True)
class ServoStep:
    integrator: float
    torque_command: float


@dataclass(frozen=True)
class ActuatorStep:
    position: float
    velocity: float


@dataclass(frozen=True)
class NoiseStep:
    primary_state: float
    secondary_state: float
    signal: float


@dataclass(frozen=True)
class SourceForces:
    base: float
    cover: float
    actuator: float
    enclosure: float
    desk: float


@dataclass(frozen=True)
class AcousticMix:
    structure: float
    airborne: float
    mixed: float


@dataclass(frozen=True)
class FinalFilterStep:
    highpass_state: float
    highpass_prev_input: float
    lowpass_state: float
    output: float


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def one_pole_alpha(cutoff_hz: float, sample_rate: int) -> float:
    cutoff = max(float(cutoff_hz), 1.0)
    return 1.0 - math.exp(-TAU * cutoff / sample_rate)


def step_spindle_motor(
    spindle_omega: float,
    motor_drive: float,
    *,
    target_omega: float,
    nominal_omega: float,
    power_state: str,
    spinup_ms: float,
    spin_down_ms: float,
    dt: float,
) -> SpindleStep:
    """Tier: physical state plus plausible model.

    Evolves spindle omega/phase and motor drive. The lag constants are calibrated
    model terms, not measured torque/inertia parameters.
    """
    omega_before = spindle_omega
    drive_target = 0.0
    if target_omega > EPS:
        drive_target = clamp((target_omega - omega_before) / target_omega, 0.0, 1.0)
    drive_tau = 0.22 if power_state == "starting" else 0.08
    drive_alpha = 1.0 - math.exp(-dt / drive_tau)
    next_motor_drive = motor_drive + (drive_target - motor_drive) * drive_alpha
    if target_omega >= omega_before:
        tau_s = max(spinup_ms / 1000.0, 0.35)
    else:
        tau_s = max(spin_down_ms / 1000.0, 0.28)
    alpha = 1.0 - math.exp(-dt / tau_s)
    next_omega = omega_before + (target_omega - omega_before) * alpha
    phase_increment = 0.5 * (omega_before + next_omega) * dt
    rpm_norm = clamp(next_omega / max(nominal_omega, EPS), 0.0, 1.35)
    motor_reaction = (next_omega - omega_before) / max(dt, EPS)
    return SpindleStep(
        motor_drive=next_motor_drive,
        spindle_omega=next_omega,
        phase_increment=phase_increment,
        rpm_norm=rpm_norm,
        motor_reaction=motor_reaction,
    )


def seek_reference(
    seek_origin: float,
    target_track: float,
    seek_elapsed_s: float,
    seek_duration_s: float,
) -> tuple[float, float]:
    """Tier: plausible model.

    Smooth normalized actuator trajectory; no attempt to model exact firmware
    seek planning or voice-coil current limits.
    """
    if seek_duration_s <= 0.0:
        return target_track, 0.0
    progress = clamp(seek_elapsed_s / max(seek_duration_s, EPS), 0.0, 1.0)
    eased = 0.5 - 0.5 * math.cos(math.pi * progress)
    desired_pos = seek_origin + (target_track - seek_origin) * eased
    desired_vel = (
        (target_track - seek_origin)
        * 0.5
        * math.pi
        * math.sin(math.pi * progress)
        / max(seek_duration_s, EPS)
    )
    return desired_pos, desired_vel


def voice_coil_servo_step(
    *,
    error: float,
    velocity_error: float,
    integrator: float,
    servo_interval: float,
    servo_mode: str,
    queue_depth: int,
    retry_activity: float,
    fragmentation_activity: float,
    directory_activity: float,
) -> ServoStep:
    """Tier: plausible model with artistic event-pressure terms.

    PID-like control is mechanical structure. Queue/retry/metadata nudges are
    sound-design calibration so busy storage activity stays audible.
    """
    mode_gain = 1.35 if servo_mode == "seek" else 0.84 if servo_mode == "track" else 1.08
    kp = 18.0 * mode_gain
    kd = 2.8 * mode_gain
    ki = 7.5 if servo_mode in {"seek", "settle"} else 2.6
    next_integrator = clamp(
        integrator + error * servo_interval,
        -0.08,
        0.08,
    )
    torque_command = (
        kp * error
        + kd * velocity_error
        + ki * next_integrator
    )
    torque_command *= 1.0 + 0.03 * max(queue_depth - 1, 0)
    torque_command += 0.18 * retry_activity
    torque_command += 0.11 * fragmentation_activity
    torque_command += 0.07 * directory_activity
    torque_command = clamp(torque_command, -8.0, 8.0)
    return ServoStep(integrator=next_integrator, torque_command=torque_command)


def step_actuator_mechanics(
    actuator_pos: float,
    actuator_vel: float,
    actuator_torque: float,
    dt: float,
) -> ActuatorStep:
    """Tier: plausible model.

    Integrates normalized actuator position/velocity with calibrated damping.
    Values are stable audio units, not SI displacement/current.
    """
    actuator_accel = 190.0 * actuator_torque - 90.0 * actuator_vel
    next_vel = actuator_vel + actuator_accel * dt
    next_pos = clamp(actuator_pos + next_vel * dt, 0.0, 1.0)
    return ActuatorStep(position=next_pos, velocity=next_vel)


def step_windage_noise(
    low_state: float,
    high_state: float,
    raw_sample: float,
    *,
    rpm_norm: float,
    startup_active: bool,
    windage_gain: float,
) -> NoiseStep:
    """Tier: plausible source, artistic level curve."""
    windage_low_alpha = 0.005 if startup_active else 0.020
    windage_high_alpha = 0.024 if startup_active else 0.130
    next_low_state = low_state + windage_low_alpha * (raw_sample - low_state)
    next_high_state = high_state + windage_high_alpha * (next_low_state - high_state)
    windage_scale = (
        rpm_norm * (0.002 + 0.050 * rpm_norm**3.8)
        if startup_active
        else (0.010 * rpm_norm + 0.18 * rpm_norm * rpm_norm)
    )
    signal = (next_low_state - next_high_state) * windage_scale * windage_gain
    return NoiseStep(primary_state=next_low_state, secondary_state=next_high_state, signal=signal)


def step_bearing_noise(
    state: float,
    raw_sample: float,
    *,
    rpm_norm: float,
    startup_active: bool,
    bearing_gain: float,
) -> NoiseStep:
    """Tier: plausible source, artistic level curve."""
    bearing_alpha = 0.010 if startup_active else 0.060
    next_state = state + bearing_alpha * (raw_sample - state)
    bearing_scale = (
        rpm_norm * (0.002 + 0.012 * rpm_norm**2.0)
        if startup_active
        else (0.006 * rpm_norm + 0.034 * rpm_norm**1.25)
    )
    signal = next_state * bearing_scale * bearing_gain
    return NoiseStep(primary_state=next_state, secondary_state=0.0, signal=signal)


def spindle_harmonic_tone(
    *,
    spindle_phase: float,
    harmonics: tuple[int, ...],
    weights: FloatArray,
    phase_offsets: FloatArray,
    rpm_norm: float,
    startup_active: bool,
    platter_gain: float,
) -> float:
    """Tier: plausible harmonic structure, artistic harmonic weighting."""
    harmonic = 0.0
    for harmonic_index, harmonic_weight, phase_offset in zip(
        harmonics,
        weights,
        phase_offsets,
        strict=True,
    ):
        startup_weight = rpm_norm ** (0.55 * max(harmonic_index - 1, 0)) if startup_active else 1.0
        harmonic += harmonic_weight * startup_weight * math.sin(spindle_phase * harmonic_index + float(phase_offset))
    return harmonic * (
        (0.005 + 0.018 * rpm_norm * rpm_norm)
        if startup_active
        else (0.012 + 0.040 * rpm_norm * rpm_norm)
    ) * platter_gain


def startup_ramp(startup_elapsed_s: float, startup_active: bool) -> float:
    """Tier: artistic calibration for audible spin-up onset."""
    return 1.0 - math.exp(-startup_elapsed_s / 0.38) if startup_active else 1.0


def startup_drive_force(motor_drive: float, rpm_norm: float, ramp: float) -> float:
    """Tier: artistic calibration for motor/body excitation."""
    return motor_drive * (0.18 + 0.82 * rpm_norm) * ramp


def structural_torque_force(
    *,
    startup_active: bool,
    startup_ramp_value: float,
    motor_reaction: float,
    startup_drive_force_value: float,
) -> float:
    """Tier: artistic calibration from motor reaction to chassis force."""
    if startup_active:
        return startup_ramp_value * 0.00012 * motor_reaction + 0.085 * startup_drive_force_value
    return 0.0008 * motor_reaction


def step_reaction_mode(
    fast_state: float,
    slow_state: float,
    *,
    excitation: float,
    dt: float,
    fast_tau_s: float,
    slow_tau_s: float,
    slow_input_scale: float,
) -> tuple[float, float, float]:
    """Tier: plausible impact/contact envelope model."""
    next_fast_state = (fast_state + excitation) * math.exp(-dt / max(fast_tau_s, 1e-5))
    next_slow_state = (slow_state + excitation * slow_input_scale) * math.exp(-dt / max(slow_tau_s, 1e-5))
    return next_fast_state, next_slow_state, next_fast_state - next_slow_state


def source_forces(
    *,
    startup_active: bool,
    torque_structure: float,
    spindle_tone: float,
    bearing: float,
    windage: float,
    wedge_force: float,
    contact_force: float,
    actuator_vel: float,
    transfer_activity: float,
    directory_activity: float,
    fragmentation_activity: float,
    startup_ramp_value: float,
    acoustic_profile: AcousticProfile,
) -> SourceForces:
    """Tier: artistic calibration.

    Routes physical-ish sources into chassis/cover/actuator/enclosure/desk
    excitations. Coefficients are timbre-preserving mix weights, not a measured
    force balance.
    """
    if startup_active:
        base_force = (
            1.55 * torque_structure
            + 0.14 * spindle_tone * startup_ramp_value
            + 0.06 * bearing
            + 0.03 * windage
        )
        cover_force = (
            0.44 * torque_structure
            + 0.08 * spindle_tone * startup_ramp_value
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
            + 0.18 * abs(actuator_vel)
            + 0.10 * contact_force
            + 0.06 * transfer_activity
            + 0.045 * directory_activity
            + 0.055 * fragmentation_activity
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
    return SourceForces(
        base=base_force,
        cover=cover_force,
        actuator=actuator_force,
        enclosure=enclosure_force,
        desk=desk_force,
    )


def step_modal_bank(
    bank: Any,
    displacement: FloatArray,
    velocity: FloatArray,
    force: float,
) -> tuple[FloatArray, FloatArray, float]:
    """Tier: plausible model.

    Discrete damped modal bank. Frequencies/damping are plausible; gains are
    calibrated for sound.
    """
    if bank.size == 0:
        return displacement, velocity, 0.0
    kicked_velocity = velocity + bank.input_gain * force
    new_displacement = bank.coeff_xx * displacement + bank.coeff_xv * kicked_velocity
    new_velocity = bank.coeff_vx * displacement + bank.coeff_vv * kicked_velocity
    signal = float(np.dot(new_velocity, bank.output_gain))
    return new_displacement, new_velocity, signal


def mix_acoustic_output(
    *,
    startup_active: bool,
    rpm_norm: float,
    mode_bank: Any,
    spindle_tone: float,
    windage: float,
    bearing: float,
    base_signal: float,
    cover_signal: float,
    actuator_signal: float,
    enclosure_signal: float,
    desk_signal: float,
) -> AcousticMix:
    """Tier: artistic calibration.

    Combines airborne and structure-borne paths with mount/profile weights.
    """
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
    return AcousticMix(structure=structure, airborne=airborne, mixed=airborne + structure)


def final_filter_step(
    *,
    mixed: float,
    highpass_prev_input: float,
    highpass_state: float,
    lowpass_state: float,
    final_highpass_alpha: float,
    final_lowpass_alpha: float,
    output_gain: float,
) -> FinalFilterStep:
    """Tier: artistic calibration for playback-safe spectral shaping."""
    hp_output = mixed - highpass_prev_input + (1.0 - final_highpass_alpha) * highpass_state
    lp_output = lowpass_state + final_lowpass_alpha * (hp_output - lowpass_state)
    shaped = math.tanh(lp_output * 2.0) * output_gain
    return FinalFilterStep(
        highpass_state=hp_output,
        highpass_prev_input=mixed,
        lowpass_state=lp_output,
        output=shaped,
    )
