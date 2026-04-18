from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from profiles import AcousticProfile, DriveProfile, resolve_selected_profiles


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HDDAudioEvent:
    rpm: float
    queue_depth: int = 1
    op_kind: str = "data"
    is_sequential: bool = False
    is_flush: bool = False
    is_spinup: bool = False
    impulse: str | None = None
    seek_distance: float = 0.0
    emitted_at: float = field(default_factory=time.monotonic)


ScheduledEvent = tuple[HDDAudioEvent, int]


@dataclass
class _ScheduledPulse:
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


class HDDAudioEventBus:
    """Thread-safe control/event queue for the realtime HDD synth."""

    def __init__(self, max_pending: int = 2048) -> None:
        self._events: deque[HDDAudioEvent] = deque()
        self._lock = threading.Lock()
        self.max_pending = max(1, max_pending)
        self._dropped_events = 0

    def publish(self, event: HDDAudioEvent) -> None:
        with self._lock:
            if len(self._events) >= self.max_pending:
                self._events.popleft()
                self._dropped_events += 1
            self._events.append(event)

    def drain(self) -> list[HDDAudioEvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def pending_count(self) -> int:
        with self._lock:
            return len(self._events)

    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_events


class HDDAudioSynthesizer:
    """
    Procedural HDD synth that consumes queued control events and renders audio.

    Event production is intentionally separate from rendering so model/runtime
    code only emits mechanical events, while the synth handles overlap,
    sidechain ducking, filtering, and sample generation.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        seed: int | None = None,
        drive_profile: str | DriveProfile | None = None,
        acoustic_profile: str | AcousticProfile | None = None,
    ) -> None:
        self.fs = sample_rate
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / sample_rate

        # Continuous render state, updated by events.
        self.rpm = 0.0
        self.is_sequential = False
        self.queue_depth = 1
        self.op_kind = "data"
        self.is_flush = False
        self.is_spinup = False

        # Continuous mechanical state.
        self.sample_clock = 0
        self.flutter_phase = 0.0
        self.windage_state = 0.0
        self.bearing_state = 0.0
        self.air_turbulence_low_state = 0.0
        self.air_turbulence_high_state = 0.0
        self.final_lowpass_state = 0.0
        self.final_highpass_state = 0.0
        self.output_gain = 0.88
        self.drive_profile: DriveProfile
        self.acoustic_profile: AcousticProfile
        self.platter_modes: tuple[MechanicalMode, ...]
        self.cover_modes: tuple[MechanicalMode, ...]
        self.actuator_modes: tuple[MechanicalMode, ...]
        self.platter_disp = np.zeros(0)
        self.platter_vel = np.zeros(0)
        self.cover_disp = np.zeros(0)
        self.cover_vel = np.zeros(0)
        self.actuator_disp = np.zeros(0)
        self.actuator_vel = np.zeros(0)
        self.platter_wn = np.zeros(0)
        self.platter_zeta = np.zeros(0)
        self.platter_gain = np.zeros(0)
        self.cover_wn = np.zeros(0)
        self.cover_zeta = np.zeros(0)
        self.cover_gain = np.zeros(0)
        self.actuator_wn = np.zeros(0)
        self.actuator_zeta = np.zeros(0)
        self.actuator_gain = np.zeros(0)

        # Concurrent transient queue. Multiple pulses may overlap and spill
        # across chunk boundaries without being truncated.
        self.pending_impulses: deque[_ScheduledPulse] = deque()
        self.configure_profiles(drive_profile=drive_profile, acoustic_profile=acoustic_profile)

    def _configure_mode_bank(
        self,
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

    def _set_mode_arrays(self) -> None:
        self.platter_disp = np.zeros(len(self.platter_modes))
        self.platter_vel = np.zeros(len(self.platter_modes))
        self.cover_disp = np.zeros(len(self.cover_modes))
        self.cover_vel = np.zeros(len(self.cover_modes))
        self.actuator_disp = np.zeros(len(self.actuator_modes))
        self.actuator_vel = np.zeros(len(self.actuator_modes))
        self.platter_wn = 2.0 * np.pi * np.array([mode.frequency_hz for mode in self.platter_modes], dtype=np.float64)
        self.platter_zeta = np.array([mode.damping_ratio for mode in self.platter_modes], dtype=np.float64)
        self.platter_gain = np.array([mode.radiation_gain for mode in self.platter_modes], dtype=np.float64)
        self.cover_wn = 2.0 * np.pi * np.array([mode.frequency_hz for mode in self.cover_modes], dtype=np.float64)
        self.cover_zeta = np.array([mode.damping_ratio for mode in self.cover_modes], dtype=np.float64)
        self.cover_gain = np.array([mode.radiation_gain for mode in self.cover_modes], dtype=np.float64)
        self.actuator_wn = 2.0 * np.pi * np.array([mode.frequency_hz for mode in self.actuator_modes], dtype=np.float64)
        self.actuator_zeta = np.array([mode.damping_ratio for mode in self.actuator_modes], dtype=np.float64)
        self.actuator_gain = np.array([mode.radiation_gain for mode in self.actuator_modes], dtype=np.float64)

    def configure_profiles(
        self,
        drive_profile: str | DriveProfile | None = None,
        acoustic_profile: str | AcousticProfile | None = None,
    ) -> None:
        resolved_drive, resolved_acoustic = resolve_selected_profiles(drive_profile, acoustic_profile)
        self.drive_profile = resolved_drive
        self.acoustic_profile = resolved_acoustic
        self.output_gain = self.acoustic_profile.output_gain
        self.platter_modes = self._configure_mode_bank(
            BASE_PLATTER_MODES,
            self.drive_profile.platter_frequency_scale,
            self.drive_profile.platter_gain_scale,
        )
        self.cover_modes = self._configure_mode_bank(
            BASE_COVER_MODES,
            self.drive_profile.cover_frequency_scale,
            self.drive_profile.cover_gain_scale,
        )
        self.actuator_modes = self._configure_mode_bank(
            BASE_ACTUATOR_MODES,
            self.drive_profile.actuator_frequency_scale,
            self.drive_profile.actuator_gain_scale,
        )
        self._set_mode_arrays()

    def apply_event(self, event: HDDAudioEvent, start_frame: int = 0) -> None:
        self.rpm = event.rpm
        self.is_sequential = event.is_sequential
        self.queue_depth = event.queue_depth
        self.op_kind = event.op_kind
        self.is_flush = event.is_flush
        self.is_spinup = event.is_spinup

        if event.impulse:
            self.pending_impulses.extend(
                self._make_impulse_pulses(
                    impulse=event.impulse,
                    start_frame=max(0, int(start_frame)),
                    seek_distance=event.seek_distance,
                    op_kind=event.op_kind,
                    is_flush=event.is_flush,
                )
            )

    def _colored_noise(self, n: int, level: float, smoothing: float, state_name: str) -> FloatArray:
        if level <= 0.0:
            return np.zeros(n)

        raw = self.rng.normal(0.0, level, n)
        output = np.empty(n)
        state = getattr(self, state_name)
        for i, sample in enumerate(raw):
            state = state * smoothing + sample * (1.0 - smoothing)
            output[i] = state
        setattr(self, state_name, state)
        return output

    def _apply_impulse(
        self,
        output: FloatArray,
        amplitude: float,
        profile: Sequence[float],
        start: int = 0,
    ) -> None:
        for idx, coeff in enumerate(profile):
            position = start + idx
            if position >= len(output):
                break
            output[position] += amplitude * coeff

    def _one_pole_lowpass(self, signal: FloatArray, cutoff_hz: float, state_name: str) -> FloatArray:
        if cutoff_hz <= 0.0:
            return np.zeros_like(signal)

        smoothing = np.exp(-2.0 * np.pi * cutoff_hz / self.fs)
        output = np.empty_like(signal)
        state = getattr(self, state_name)
        for i, sample in enumerate(signal):
            state = state * smoothing + sample * (1.0 - smoothing)
            output[i] = state
        setattr(self, state_name, state)
        return output

    def _one_pole_highpass(self, signal: FloatArray, cutoff_hz: float, state_name: str) -> FloatArray:
        if cutoff_hz <= 0.0:
            return np.array(signal, copy=True)
        return signal - self._one_pole_lowpass(signal, cutoff_hz, state_name)

    def _band_limited_noise(
        self,
        n: int,
        level: float,
        smoothing: float,
        low_cut_hz: float,
        high_cut_hz: float,
        state_name: str,
    ) -> FloatArray:
        if level <= 0.0:
            return np.zeros(n)
        raw = self._colored_noise(n, level, smoothing, state_name)
        return self._one_pole_lowpass(raw, high_cut_hz, "air_turbulence_high_state") - self._one_pole_lowpass(
            raw, low_cut_hz, "air_turbulence_low_state"
        )

    def _time_axis(self, n: int) -> FloatArray:
        t = (np.arange(n) + self.sample_clock) / self.fs
        self.sample_clock += n
        return t

    def _generate_spindle_forces(self, t: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        n = len(t)
        rpm = self.rpm
        if rpm <= 0.0:
            self.flutter_phase += n
            return np.zeros(n), np.zeros(n), np.zeros(n)

        f0 = rpm / 60.0
        flutter_t = (np.arange(n) + self.flutter_phase) / self.fs
        self.flutter_phase += n

        omega_ratio = min(rpm / max(float(self.drive_profile.rpm), 1.0), 1.35)
        omega_sq = omega_ratio * omega_ratio
        flutter = 1.0 + 0.02 * np.sin(2 * np.pi * 0.37 * flutter_t) + 0.008 * np.sin(
            2 * np.pi * 0.91 * flutter_t
        )
        queue_excitation = 1.0 + 0.04 * max(self.queue_depth - 1, 0)

        # Spindle imbalance force scales with omega^2.
        imbalance = np.zeros(n)
        for harmonic, amp in zip(self.drive_profile.spindle_harmonics, self.drive_profile.spindle_harmonic_weights):
            imbalance += amp * np.sin(2 * np.pi * (f0 * harmonic) * t)
        imbalance *= 0.012 * omega_sq * flutter

        # Bearing / motor roughness adds small broadband forcing.
        bearing = self._colored_noise(
            n,
            (0.00055 + 0.0012 * omega_sq) * self.drive_profile.bearing_gain * (0.96 + 0.04 * queue_excitation),
            0.985,
            "bearing_state",
        )

        # Windage / disk flutter is the main high-speed idle source.
        windage_level = (0.0018 + 0.0048 * omega_ratio**2.2) * self.drive_profile.windage_gain * queue_excitation
        if self.is_spinup:
            windage_level *= 1.3
        if self.op_kind in {"journal", "flush"}:
            windage_level *= 1.08
        windage = self._band_limited_noise(
            n,
            windage_level,
            0.93 if self.is_sequential else 0.88,
            550.0,
            3600.0,
            "windage_state",
        )
        direct_radiation = imbalance * 0.72 + bearing * 0.55
        platter_force = imbalance * 0.45 + bearing * 0.18 + windage * 1.1
        cover_force = imbalance * 0.30 + bearing * 0.14 + windage * 0.55
        return direct_radiation, platter_force, cover_force

    def _make_impulse_pulses(
        self,
        impulse: str,
        start_frame: int,
        seek_distance: float,
        op_kind: str,
        is_flush: bool,
    ) -> list[_ScheduledPulse]:
        pulses: list[_ScheduledPulse] = []
        start = max(0, int(start_frame))
        if impulse == "park":
            pulses.append(_ScheduledPulse(start, max(6, int(0.0012 * self.fs)), 0.95))
            pulses.append(
                _ScheduledPulse(
                    start + int(0.0009 * self.fs),
                    max(5, int(0.0009 * self.fs)),
                    -0.62,
                )
            )
            return pulses
        if impulse == "calibration":
            pulses.append(_ScheduledPulse(start, max(4, int(0.00045 * self.fs)), 0.12))
            pulses.append(
                _ScheduledPulse(
                    start + int(0.00045 * self.fs),
                    max(4, int(0.00035 * self.fs)),
                    -0.05,
                )
            )
            return pulses
        if impulse != "seek":
            return pulses

        kind_scale = {
            "metadata": 0.36,
            "journal": 0.48,
            "flush": 0.68,
            "data": 0.78,
        }.get(op_kind, 0.55)
        stroke = min(max(seek_distance / 1000.0, 0.0), 1.0)
        force_scale = kind_scale * (0.30 + 0.62 * np.sqrt(stroke)) * self.acoustic_profile.impulse_gain
        accel_width = max(4, int(self.fs * (0.00016 + 0.00022 * stroke)))
        brake_delay = max(5, int(self.fs * (0.00020 + 0.00018 * stroke)))
        brake_width = max(5, int(accel_width * 0.85))
        settle_width = max(4, int(accel_width * 0.55))
        pulses.append(_ScheduledPulse(start, accel_width, force_scale))
        pulses.append(_ScheduledPulse(start + brake_delay, brake_width, -force_scale * 0.78))
        pulses.append(
            _ScheduledPulse(
                start + brake_delay + int(0.00032 * self.fs),
                settle_width,
                force_scale * 0.24,
            )
        )
        if is_flush:
            pulses.append(
                _ScheduledPulse(
                    start + int(0.0007 * self.fs),
                    max(5, int(accel_width * 0.7)),
                    force_scale * 0.28,
                )
            )
        return pulses

    def _render_scheduled_pulse(
        self,
        actuator_force: FloatArray,
        pulse: _ScheduledPulse,
    ) -> _ScheduledPulse | None:
        frame_count = len(actuator_force)
        if pulse.start_frame >= frame_count:
            pulse.start_frame -= frame_count
            return pulse

        remaining_width = pulse.width - pulse.phase_offset
        if remaining_width <= 0:
            return None

        render_start = max(0, pulse.start_frame)
        render_end = min(frame_count, pulse.start_frame + remaining_width)
        if render_end <= render_start:
            return None

        profile_start = pulse.phase_offset + (render_start - pulse.start_frame)
        sample_indices = np.arange(profile_start, profile_start + (render_end - render_start), dtype=np.float64)
        window = np.sin(np.pi * sample_indices / pulse.width)
        actuator_force[render_start:render_end] += pulse.amplitude * window

        consumed = render_end - pulse.start_frame
        if pulse.start_frame + remaining_width > frame_count:
            return _ScheduledPulse(
                start_frame=0,
                width=pulse.width,
                amplitude=pulse.amplitude,
                phase_offset=pulse.phase_offset + max(consumed, 0),
            )
        return None

    def _generate_actuator_force(self, t: FloatArray) -> FloatArray:
        n = len(t)
        actuator_force = np.zeros(n)
        carry: deque[_ScheduledPulse] = deque()
        while self.pending_impulses:
            pulse = self.pending_impulses.popleft()
            carried_pulse = self._render_scheduled_pulse(actuator_force, pulse)
            if carried_pulse is not None:
                carry.append(carried_pulse)

        self.pending_impulses = carry

        if self.is_sequential and self.rpm > 0.0:
            boundary_rate_hz = 24.0 + 2.5 * min(self.queue_depth, 6)
            boundary_scale = (
                0.0075
                * self.drive_profile.boundary_excitation_gain
                * self.acoustic_profile.sequential_boundary_gain
                * (1.0 + 0.05 * max(self.queue_depth - 1, 0))
            )
            actuator_force += boundary_scale * np.sin(2 * np.pi * boundary_rate_hz * t)

        if self.op_kind in {"journal", "flush"}:
            actuator_force *= 1.08 + 0.03 * max(self.queue_depth - 1, 0)
        elif self.queue_depth > 1:
            actuator_force *= 1.0 + 0.015 * min(self.queue_depth - 1, 8)
        return actuator_force

    def _simulate_mode_bank(
        self,
        force: FloatArray,
        wn: FloatArray,
        zeta: FloatArray,
        gains: FloatArray,
        disp_state: FloatArray,
        vel_state: FloatArray,
    ) -> FloatArray:
        response = np.zeros(len(force))
        inv_wn = 1.0 / np.maximum(wn, 1.0)
        for idx, force_sample in enumerate(force):
            accel = force_sample - 2.0 * zeta * wn * vel_state - (wn * wn) * disp_state
            vel_state += accel * self.dt
            disp_state += vel_state * self.dt
            response[idx] = np.dot(gains, vel_state * inv_wn)
        return response

    def _radiate(
        self,
        direct_force: FloatArray,
        platter_force: FloatArray,
        cover_force: FloatArray,
        actuator_force: FloatArray,
    ) -> FloatArray:
        platter = self._simulate_mode_bank(
            platter_force,
            self.platter_wn,
            self.platter_zeta,
            self.platter_gain,
            self.platter_disp,
            self.platter_vel,
        )
        cover = self._simulate_mode_bank(
            cover_force,
            self.cover_wn,
            self.cover_zeta,
            self.cover_gain,
            self.cover_disp,
            self.cover_vel,
        )
        actuator = self._simulate_mode_bank(
            actuator_force,
            self.actuator_wn,
            self.actuator_zeta,
            self.actuator_gain,
            self.actuator_disp,
            self.actuator_vel,
        )
        radiated = (
            direct_force * self.acoustic_profile.direct_gain
            + platter * self.acoustic_profile.platter_gain
            + cover * self.acoustic_profile.cover_gain
            + actuator * self.acoustic_profile.actuator_gain
        )
        radiated = self._one_pole_highpass(
            radiated,
            self.acoustic_profile.final_highpass_hz,
            "final_highpass_state",
        )
        return self._one_pole_lowpass(
            radiated,
            self.acoustic_profile.final_lowpass_hz,
            "final_lowpass_state",
        )

    def render_chunk(
        self,
        frames: int,
        scheduled_events: Sequence[ScheduledEvent] = (),
    ) -> FloatArray:
        for event, start_frame in scheduled_events:
            self.apply_event(event, start_frame=start_frame)

        t = self._time_axis(frames)
        direct_force, platter_force, cover_force = self._generate_spindle_forces(t)
        actuator_force = self._generate_actuator_force(t)
        cover_force += actuator_force * 0.62
        radiated = self._radiate(direct_force, platter_force, cover_force, actuator_force)
        return np.tanh(radiated * 1.4) * self.output_gain


class HDDAudioEngine:
    """
    Realtime HDD audio engine.

    The HDD model emits control events into an event bus. The synthesizer drains
    and renders them in realtime. This keeps event production separate from the
    actual sound synthesis/output device path.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        seed: int | None = None,
        max_pending_events: int = 2048,
        drive_profile: str | DriveProfile | None = None,
        acoustic_profile: str | AcousticProfile | None = None,
    ) -> None:
        self.fs = sample_rate
        self.chunk_size = 1024
        self.events = HDDAudioEventBus(max_pending=max_pending_events)
        self.synthesizer = HDDAudioSynthesizer(
            sample_rate=sample_rate,
            seed=seed,
            drive_profile=drive_profile,
            acoustic_profile=acoustic_profile,
        )
        self.stream = None
        self.render_lock = threading.Lock()
        self.last_render_at = None
        self.output_enabled = True

    def emit_event(self, event: HDDAudioEvent) -> None:
        self.events.publish(event)

    def emit_telemetry(
        self,
        rpm: float,
        seek_trigger: bool = False,
        seek_dist: float = 0,
        is_seq: bool = False,
        is_park: bool = False,
        is_cal: bool = False,
        queue_depth: int = 1,
        op_kind: str = "data",
        is_flush: bool = False,
        is_spinup: bool = False,
    ) -> None:
        impulse = None
        if is_park:
            impulse = "park"
        elif seek_trigger:
            impulse = "seek"
        elif is_cal:
            impulse = "calibration"

        self.emit_event(
            HDDAudioEvent(
                rpm=rpm,
                queue_depth=queue_depth,
                op_kind=op_kind,
                is_sequential=is_seq,
                is_flush=is_flush,
                is_spinup=is_spinup,
                impulse=impulse,
                seek_distance=seek_dist,
            )
        )

    # Backward compatibility for older call sites.
    def _update_telemetry(self, *args: Any, **kwargs: Any) -> None:
        self.emit_telemetry(*args, **kwargs)

    def pending_event_count(self) -> int:
        return self.events.pending_count()

    def dropped_event_count(self) -> int:
        return self.events.dropped_count()

    def configure_profiles(
        self,
        drive_profile: str | DriveProfile | None = None,
        acoustic_profile: str | AcousticProfile | None = None,
    ) -> None:
        self.synthesizer.configure_profiles(drive_profile=drive_profile, acoustic_profile=acoustic_profile)

    def _schedule_events_for_chunk(self, frames: int) -> list[ScheduledEvent]:
        events = self.events.drain()
        if not events:
            return []

        if self.stream is None:
            return [(event, 0) for event in events]

        now = time.monotonic()
        chunk_duration_s = frames / self.fs
        chunk_start = self.last_render_at
        if chunk_start is None:
            chunk_start = now - chunk_duration_s
        scheduled = []
        for event in events:
            frame_offset = int(round((event.emitted_at - chunk_start) * self.fs))
            frame_offset = max(0, min(frames - 1, frame_offset))
            scheduled.append((event, frame_offset))
        scheduled.sort(key=lambda item: item[1])
        return scheduled

    def render_chunk(self, frames: int) -> FloatArray:
        with self.render_lock:
            scheduled_events = self._schedule_events_for_chunk(frames)
            chunk = self.synthesizer.render_chunk(frames, scheduled_events=scheduled_events)
            self.last_render_at = time.monotonic()
            return chunk

    def _audio_callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        outdata[:] = self.render_chunk(frames).reshape(-1, 1)

    def start(self) -> None:
        if self.stream is not None:
            return
        self.configure_profiles(
            drive_profile=os.environ.get("FAKE_HDD_DRIVE_PROFILE", self.synthesizer.drive_profile.name),
            acoustic_profile=os.environ.get("FAKE_HDD_ACOUSTIC_PROFILE", self.synthesizer.acoustic_profile.name),
        )
        audio_setting = os.environ.get("FAKE_HDD_AUDIO", "live").strip().lower()
        if audio_setting in {"0", "off", "false", "disabled", "none"}:
            self.output_enabled = False
            self.last_render_at = None
            return
        self.output_enabled = True
        stream = sd.OutputStream(
            samplerate=self.fs,
            channels=1,
            callback=self._audio_callback,
            blocksize=self.chunk_size,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self.stream = stream
        self.last_render_at = time.monotonic()

    def stop(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
        finally:
            self.stream.close()
            self.stream = None
            self.last_render_at = None


engine = HDDAudioEngine()
