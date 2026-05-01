"""Physics-inspired HDD audio primitives with explicit honesty labels.

Model tiers used here:

- Physical state: time-integrated runtime state such as spindle phase/omega,
  actuator position/velocity, and filter/modal states. These are state variables,
  though many use normalized units instead of SI units.
- Plausible model: standard mechanical/audio shapes such as first-order motor
  lag, cosine seek profiles, PID-like servo control, damped resonators, and
  filtered windage/bearing noise.
- Artistic calibration: reserved audit tier for timbre-preserving terms that
  are not explainable as normalized mechanics or acoustics. The hardening path
  should keep this empty unless a new exception is deliberately documented.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..profiles import AcousticProfile


FloatArray = npt.NDArray[np.float64]
TAU = 2.0 * math.pi
EPS = 1e-9
MODEL_TIERS = ("physical_state", "physical_model", "artistic_calibration")


MODEL_TIER_BY_FUNCTION: Mapping[str, str] = {
    "clamp": "physical_model",
    "one_pole_alpha": "physical_model",
    "rotor_torque_balance": "physical_model",
    "exact_rotor_step": "physical_model",
    "step_spindle_motor": "physical_model",
    "seek_reference": "physical_model",
    "voice_coil_servo_step": "physical_model",
    "voice_coil_servo_gains": "physical_model",
    "step_actuator_mechanics": "physical_model",
    "spindle_airflow_source": "physical_model",
    "step_windage_noise": "physical_model",
    "bearing_vibration_source": "physical_model",
    "step_bearing_noise": "physical_model",
    "spindle_rotor_excitation": "physical_model",
    "motor_startup_current_envelope": "physical_model",
    "spindle_motor_reaction_force": "physical_model",
    "chassis_reaction_force": "physical_model",
    "head_load_contact_force": "physical_model",
    "head_media_event_forces": "physical_model",
    "actuator_latch_event_force": "physical_model",
    "park_stop_contact_force": "physical_model",
    "voice_coil_force_transfer": "physical_model",
    "sequential_boundary_contact_force": "physical_model",
    "step_stiffness_damping_contact": "physical_model",
    "route_sources_to_structure": "physical_model",
    "step_modal_bank": "physical_model",
    "radiate_acoustic_paths": "physical_model",
    "output_gain_stage_step": "physical_model",
    "artistic_budget": "physical_model",
}


def artistic_budget() -> tuple[str, ...]:
    """Return the remaining intentionally artistic model functions."""
    return tuple(
        function_name
        for function_name, tier in MODEL_TIER_BY_FUNCTION.items()
        if tier == "artistic_calibration"
    )


@dataclass(frozen=True)
class TorqueBalance:
    motor_torque: float
    viscous_drag_torque: float
    windage_drag_torque: float
    drag_torque: float
    net_torque: float
    angular_accel: float


@dataclass(frozen=True)
class RotorDragModel:
    inertia: float
    viscous_drag: float
    windage_drag: float


@dataclass(frozen=True)
class SpindleStep:
    motor_drive: float
    spindle_omega: float
    phase_increment: float
    rpm_norm: float
    motor_reaction: float
    torque_balance: TorqueBalance


@dataclass(frozen=True)
class ServoGains:
    kp: float
    kd: float
    ki: float


@dataclass(frozen=True)
class ServoStep:
    integrator: float
    torque_command: float


@dataclass(frozen=True)
class NormalizedMassDamper:
    mass: float
    force_constant: float
    viscous_damping: float


VOICE_COIL_PLANT = NormalizedMassDamper(
    mass=1.0,
    force_constant=190.0,
    viscous_damping=90.0,
)


@dataclass(frozen=True)
class ActuatorStep:
    position: float
    velocity: float


@dataclass(frozen=True)
class NoiseStep:
    primary_state: float
    secondary_state: float
    source_strength: float
    signal: float


@dataclass(frozen=True)
class ContactEventForces:
    wedge: float
    contact: float


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


def rotor_torque_balance(
    *,
    omega: float,
    target_omega: float,
    time_constant_s: float,
    inertia: float = 1.0,
    windage_drag_share_at_nominal: float = 0.04,
) -> TorqueBalance:
    """Tier: physical_model.

    Balance motor torque against viscous bearing drag plus quadratic air drag.
    Motor torque is the torque needed to hold `target_omega` against both drag
    terms plus the transient acceleration implied by the time constant.
    """
    viscous_drag = max(inertia, EPS) / max(time_constant_s, EPS)
    windage_drag = (
        windage_drag_share_at_nominal
        * viscous_drag
        / max(abs(target_omega), EPS)
        if abs(target_omega) > EPS
        else 0.0
    )
    motor_torque = viscous_drag * target_omega + windage_drag * target_omega * abs(target_omega)
    viscous_drag_torque = viscous_drag * omega
    windage_drag_torque = windage_drag * omega * abs(omega)
    drag_torque = viscous_drag_torque + windage_drag_torque
    net_torque = motor_torque - drag_torque
    return TorqueBalance(
        motor_torque=motor_torque,
        viscous_drag_torque=viscous_drag_torque,
        windage_drag_torque=windage_drag_torque,
        drag_torque=drag_torque,
        net_torque=net_torque,
        angular_accel=net_torque / max(inertia, EPS),
    )


def exact_rotor_step(
    omega: float,
    target_omega: float,
    time_constant_s: float,
    dt: float,
    *,
    inertia: float = 1.0,
    windage_drag_share_at_nominal: float = 0.04,
) -> float:
    """Tier: physical_model.

    Semi-implicit rotor step for I*domega/dt = motor - c*omega - q*omega*abs(omega).
    Small substeps keep the quadratic windage term stable without depending on
    render chunk size.
    """
    if dt <= 0.0:
        return omega
    substeps = max(1, math.ceil(dt / 0.0005))
    step_dt = dt / substeps
    next_omega = omega
    for _ in range(substeps):
        balance = rotor_torque_balance(
            omega=next_omega,
            target_omega=target_omega,
            time_constant_s=time_constant_s,
            inertia=inertia,
            windage_drag_share_at_nominal=windage_drag_share_at_nominal,
        )
        next_omega += balance.angular_accel * step_dt
    return next_omega


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
    inertia: float = 1.0,
    windage_drag_share_at_nominal: float = 0.04,
) -> SpindleStep:
    """Tier: physical state plus plausible model.

    Evolves spindle omega/phase and motor drive. Profiles can pass source-backed
    inertia and windage-drag terms; existing profiles retain legacy defaults.
    """
    omega_before = spindle_omega
    drive_target = 0.0
    if target_omega > EPS:
        drive_target = clamp((target_omega - omega_before) / target_omega, 0.0, 1.0)
    drive_tau = 0.22 if power_state == "starting" else 0.08
    drive_alpha = 1.0 - math.exp(-dt / drive_tau)
    next_motor_drive = motor_drive + (drive_target - motor_drive) * drive_alpha
    tau_s = max(spinup_ms / 1000.0, 0.35) if target_omega >= omega_before else max(spin_down_ms / 1000.0, 0.28)
    torque_balance = rotor_torque_balance(
        omega=omega_before,
        target_omega=target_omega,
        time_constant_s=tau_s,
        inertia=inertia,
        windage_drag_share_at_nominal=windage_drag_share_at_nominal,
    )
    next_omega = exact_rotor_step(
        omega_before,
        target_omega,
        tau_s,
        dt,
        inertia=inertia,
        windage_drag_share_at_nominal=windage_drag_share_at_nominal,
    )
    phase_increment = 0.5 * (omega_before + next_omega) * dt
    rpm_norm = clamp(next_omega / max(nominal_omega, EPS), 0.0, 1.35)
    motor_reaction = inertia * (next_omega - omega_before) / max(dt, EPS)
    return SpindleStep(
        motor_drive=next_motor_drive,
        spindle_omega=next_omega,
        phase_increment=phase_increment,
        rpm_norm=rpm_norm,
        motor_reaction=motor_reaction,
        torque_balance=torque_balance,
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
    gains = voice_coil_servo_gains(servo_mode)
    next_integrator = clamp(
        integrator + error * servo_interval,
        -0.08,
        0.08,
    )
    torque_command = (
        gains.kp * error
        + gains.kd * velocity_error
        + gains.ki * next_integrator
    )
    torque_command *= 1.0 + 0.03 * max(queue_depth - 1, 0)
    torque_command += 0.18 * retry_activity
    torque_command += 0.11 * fragmentation_activity
    torque_command += 0.07 * directory_activity
    torque_command = clamp(torque_command, -8.0, 8.0)
    return ServoStep(integrator=next_integrator, torque_command=torque_command)


def voice_coil_servo_gains(servo_mode: str) -> ServoGains:
    mode_gain = 1.35 if servo_mode == "seek" else 0.84 if servo_mode == "track" else 1.08
    return ServoGains(
        kp=18.0 * mode_gain,
        kd=2.8 * mode_gain,
        ki=7.5 if servo_mode in {"seek", "settle"} else 2.6,
    )


def step_actuator_mechanics(
    actuator_pos: float,
    actuator_vel: float,
    actuator_torque: float,
    dt: float,
    plant: NormalizedMassDamper = VOICE_COIL_PLANT,
) -> ActuatorStep:
    """Tier: plausible model.

    Integrates normalized actuator position/velocity with calibrated damping.
    Values are stable audio units, not SI displacement/current.
    """
    actuator_force = plant.force_constant * actuator_torque
    damping_force = plant.viscous_damping * actuator_vel
    actuator_accel = (actuator_force - damping_force) / max(plant.mass, EPS)
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
    """Tier: physical_model.

    Filtered airflow noise. Source strength is expressed as spindle surface
    speed plus dynamic pressure/turbulence terms; `windage_gain` is the one
    profile-level calibration scalar retained for this source.
    """
    windage_low_alpha = 0.005 if startup_active else 0.020
    windage_high_alpha = 0.024 if startup_active else 0.130
    next_low_state = low_state + windage_low_alpha * (raw_sample - low_state)
    next_high_state = high_state + windage_high_alpha * (next_low_state - high_state)
    source_strength = spindle_airflow_source(rpm_norm, startup_active=startup_active)
    signal = (next_low_state - next_high_state) * source_strength * windage_gain
    return NoiseStep(
        primary_state=next_low_state,
        secondary_state=next_high_state,
        source_strength=source_strength,
        signal=signal,
    )


def spindle_airflow_source(rpm_norm: float, *, startup_active: bool) -> float:
    """Tier: physical_model.

    HDD windage roughly follows spindle surface speed and turbulent/dynamic
    pressure. Startup keeps the turbulent term gated until there is enough air
    speed for audible broadband flow.
    """
    speed_ratio = max(rpm_norm, 0.0)
    dynamic_pressure = speed_ratio * speed_ratio
    turbulent_pressure = dynamic_pressure * speed_ratio * speed_ratio
    if startup_active:
        return 0.002 * speed_ratio + 0.050 * turbulent_pressure * speed_ratio**0.8
    return 0.010 * speed_ratio + 0.18 * dynamic_pressure


def step_bearing_noise(
    state: float,
    raw_sample: float,
    *,
    rpm_norm: float,
    startup_active: bool,
    bearing_gain: float,
) -> NoiseStep:
    """Tier: physical_model.

    Bearing vibration follows shaft speed with a mild load-dependent nonlinear
    term. `bearing_gain` is the retained profile-level calibration scalar.
    """
    bearing_alpha = 0.010 if startup_active else 0.060
    next_state = state + bearing_alpha * (raw_sample - state)
    source_strength = bearing_vibration_source(rpm_norm, startup_active=startup_active)
    signal = next_state * source_strength * bearing_gain
    return NoiseStep(
        primary_state=next_state,
        secondary_state=0.0,
        source_strength=source_strength,
        signal=signal,
    )


def bearing_vibration_source(rpm_norm: float, *, startup_active: bool) -> float:
    """Tier: physical_model."""
    speed_ratio = max(rpm_norm, 0.0)
    load_nonlinearity = speed_ratio**1.25
    if startup_active:
        return 0.002 * speed_ratio + 0.012 * speed_ratio**3.0
    return 0.006 * speed_ratio + 0.034 * load_nonlinearity


def spindle_rotor_excitation(
    *,
    spindle_phase: float,
    harmonics: tuple[int, ...],
    weights: FloatArray,
    phase_offsets: FloatArray,
    rpm_norm: float,
    startup_active: bool,
    platter_gain: float,
) -> float:
    """Tier: physical_model.

    Rotor tone from repeatable rotating imbalance and bearing/track waviness.
    The harmonic list and relative amplitudes come from the drive profile; they
    are treated as broad physical assumptions for rotor eccentricity, platter
    stack runout, and bearing order content rather than per-scenario tuning.
    """
    excitation = 0.0
    for harmonic_index, harmonic_weight, phase_offset in zip(
        harmonics,
        weights,
        phase_offsets,
        strict=True,
    ):
        runout_order_gain = rpm_norm ** (0.55 * max(harmonic_index - 1, 0)) if startup_active else 1.0
        excitation += harmonic_weight * runout_order_gain * math.sin(spindle_phase * harmonic_index + float(phase_offset))
    return excitation * (
        (0.005 + 0.018 * rpm_norm * rpm_norm)
        if startup_active
        else (0.012 + 0.040 * rpm_norm * rpm_norm)
    ) * platter_gain


def motor_startup_current_envelope(startup_elapsed_s: float, startup_active: bool) -> float:
    """Tier: physical_model.

    Approximate spindle-controller current rise after start command. This keeps
    startup force tied to motor state rather than an arbitrary audible fade.
    """
    return 1.0 - math.exp(-startup_elapsed_s / 0.38) if startup_active else 1.0


def spindle_motor_reaction_force(motor_drive: float, rpm_norm: float, current_envelope: float) -> float:
    """Tier: physical_model.

    Normalized motor reaction from controller effort and rotor speed. The low
    static term covers startup cogging/current before full air-bearing speed.
    """
    return motor_drive * (0.18 + 0.82 * rpm_norm) * current_envelope


def chassis_reaction_force(
    *,
    startup_active: bool,
    startup_current_envelope: float,
    motor_reaction: float,
    motor_reaction_force: float,
) -> float:
    """Tier: physical_model.

    Convert spindle angular acceleration and motor stator reaction into chassis
    excitation. Constants are normalized broad assumptions until measured
    inertia/mount transfer data exists.
    """
    if startup_active:
        return startup_current_envelope * 0.00012 * motor_reaction + 0.085 * motor_reaction_force
    return 0.0008 * motor_reaction


def head_load_contact_force(previous_heads_loaded: bool, heads_loaded: bool) -> float:
    """Tier: physical_model.

    Normalized contact impulse from loading the head stack onto the air bearing.
    """
    return 0.26 if not previous_heads_loaded and heads_loaded else 0.0


def head_media_event_forces(
    *,
    op_kind: str,
    directory_activity: float,
    fragmentation_activity: float,
    repetition_pressure: float,
    repetition_variant: float,
) -> ContactEventForces:
    """Tier: physical_model.

    Convert filesystem/media locality pressure into normalized head/media and
    servo-wedge contact forces. Constants are broad normalized assumptions.
    """
    wedge = 0.0
    contact = 0.0
    if directory_activity > 0.0 and op_kind == "metadata":
        wedge += 0.035 * directory_activity
    if fragmentation_activity > 0.0:
        wedge += 0.045 * fragmentation_activity
    if repetition_pressure > 0.0 and op_kind in {"data", "writeback", "metadata"}:
        wedge += 0.010 + 0.018 * repetition_pressure * (1.0 - abs(repetition_variant))
        contact += 0.006 * repetition_pressure * abs(repetition_variant)
    return ContactEventForces(wedge=wedge, contact=contact)


def actuator_latch_event_force(servo_mode: str) -> float:
    """Tier: physical_model.

    Normalized reaction from parking or calibration latch motion.
    """
    if servo_mode == "park":
        return 0.22
    if servo_mode == "calibration":
        return 0.18
    return 0.0


def park_stop_contact_force() -> float:
    """Tier: physical_model.

    Contact force from the actuator reaching the park stop.
    """
    return 0.40


def voice_coil_force_transfer(torque_delta: float, servo_mode: str) -> float:
    """Tier: physical_model.

    Map a current-command step through the voice-coil reaction path into a
    normalized structural excitation.
    """
    if servo_mode == "track":
        reaction_gain = 0.10
    elif servo_mode == "seek":
        reaction_gain = 0.56
    else:
        reaction_gain = 0.30
    return abs(torque_delta) * reaction_gain


def sequential_boundary_contact_force(
    *,
    boundary_gain: float,
    rpm_norm: float,
    fragmentation_activity: float,
) -> float:
    """Tier: physical_model.

    Normalized head/media tick when sequential transfer crosses a track/zone
    boundary under the current spindle speed.
    """
    return 0.045 * boundary_gain * (0.8 + 0.2 * rpm_norm) * (1.0 + 0.25 * fragmentation_activity)


def step_stiffness_damping_contact(
    fast_state: float,
    slow_state: float,
    *,
    excitation: float,
    dt: float,
    fast_tau_s: float,
    slow_tau_s: float,
    slow_input_scale: float,
) -> tuple[float, float, float]:
    """Tier: physical_model.

    Two normalized contact envelopes approximate a stiffness/damping impact:
    the fast state is the local contact spring response, while the slow state is
    the mounting/body relaxation subtracted from it.
    """
    next_fast_state = (fast_state + excitation) * math.exp(-dt / max(fast_tau_s, 1e-5))
    next_slow_state = (slow_state + excitation * slow_input_scale) * math.exp(-dt / max(slow_tau_s, 1e-5))
    return next_fast_state, next_slow_state, next_fast_state - next_slow_state


def route_sources_to_structure(
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
    """Tier: physical_model.

    Route normalized source forces into chassis, cover, actuator, enclosure,
    and desk paths. The coefficients are broad path-transfer assumptions until
    measured impedance/radiation data is available; they are not per-scenario
    sound tuning.
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


def radiate_acoustic_paths(
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
    """Tier: physical_model.

    Radiate modal structure velocities and direct airborne spindle/flow sources
    into the listener signal using normalized acoustic path gains.
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


def output_gain_stage_step(
    *,
    mixed: float,
    highpass_prev_input: float,
    highpass_state: float,
    lowpass_state: float,
    final_highpass_alpha: float,
    final_lowpass_alpha: float,
    output_gain: float,
) -> FinalFilterStep:
    """Tier: physical_model.

    Linear acoustic highpass/lowpass and gain staging. Safety clipping is left
    to PCM/export paths instead of hidden nonlinear shaping in the model.
    """
    hp_output = mixed - highpass_prev_input + (1.0 - final_highpass_alpha) * highpass_state
    lp_output = lowpass_state + final_lowpass_alpha * (hp_output - lowpass_state)
    shaped = lp_output * 2.0 * output_gain
    return FinalFilterStep(
        highpass_state=hp_output,
        highpass_prev_input=mixed,
        lowpass_state=lp_output,
        output=shaped,
    )
