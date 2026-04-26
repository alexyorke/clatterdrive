from __future__ import annotations

import json
import threading
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from .core import (
    AudioDiagnosticTrace,
    AudioModeBank,
    AudioRenderState,
    ScheduledEvent,
    apply_event as apply_audio_event,
    build_mode_bank,
    initialize_render_state,
    reinitialize_mode_state,
    render_diagnostic_chunk as render_audio_diagnostic_chunk,
    render_chunk as render_audio_chunk,
)
from ..profiles import (
    AcousticProfile,
    DriveProfile,
    resolve_selected_profiles,
    resolve_selected_profiles_from_env,
)
from ..runtime.deps import RuntimeDeps
from ..storage_events import StorageEvent, StorageEventBus, StorageEventSink


FloatArray = npt.NDArray[np.float64]
HDDAudioEvent = StorageEvent
HDDAudioEventBus = StorageEventBus


def _parse_audio_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.isdigit():
        return int(candidate)
    return candidate


class _WaveTeeRecorder:
    """Incrementally records the rendered output stream to a mono WAV file."""

    def __init__(self, path: str, sample_rate: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("wb")
        self._writer = wave.open(self._file, "wb")  # noqa: SIM115
        self._writer.setnchannels(1)
        self._writer.setsampwidth(2)
        self._writer.setframerate(sample_rate)

    def write_chunk(self, chunk: FloatArray) -> None:
        pcm = np.clip(chunk * 32767.0, -32768.0, 32767.0).astype(np.int16)
        self._writer.writeframes(pcm.tobytes())

    def close(self) -> None:
        self._writer.close()
        self._file.close()


class HDDAudioSynthesizer:
    """
    Thin shell over the pure audio core.

    The shell owns only RNG selection and the current render state. Profile
    derivation, event application, and chunk rendering all live in `audio_core`.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        seed: int | None = None,
        drive_profile: str | DriveProfile | None = None,
        acoustic_profile: str | AcousticProfile | None = None,
        deps: RuntimeDeps | None = None,
    ) -> None:
        self.deps = deps or RuntimeDeps()
        self.fs = sample_rate
        self.seed = 0 if seed is None else int(seed) & 0xFFFFFFFFFFFFFFFF
        self.drive_profile: DriveProfile
        self.acoustic_profile: AcousticProfile
        self.mode_bank: AudioModeBank
        self.state: AudioRenderState
        self._deferred_events: list[ScheduledEvent] = []

        resolved_drive, resolved_acoustic = resolve_selected_profiles(drive_profile, acoustic_profile)
        self.drive_profile = resolved_drive
        self.acoustic_profile = resolved_acoustic
        self.mode_bank = build_mode_bank(self.drive_profile, self.fs, self.acoustic_profile)
        self.state = initialize_render_state(self.fs, self.mode_bank, self.acoustic_profile)

    @property
    def rpm(self) -> float:
        return self.state.target_rpm

    @property
    def actual_rpm(self) -> float:
        return self.state.spindle_omega * 60.0 / (2.0 * np.pi)

    @property
    def pending_impulses(self) -> tuple[Any, ...]:
        return ()

    def configure_profiles(
        self,
        drive_profile: str | DriveProfile | None = None,
        acoustic_profile: str | AcousticProfile | None = None,
    ) -> None:
        resolved_drive, resolved_acoustic = resolve_selected_profiles(drive_profile, acoustic_profile)
        self.drive_profile = resolved_drive
        self.acoustic_profile = resolved_acoustic
        self.mode_bank = build_mode_bank(self.drive_profile, self.fs, self.acoustic_profile)
        self.state = reinitialize_mode_state(self.state, self.mode_bank, self.acoustic_profile)
        self._deferred_events.clear()

    def apply_event(self, event: HDDAudioEvent, start_frame: int = 0) -> None:
        self.state = apply_audio_event(
            self.state,
            event,
            self.mode_bank,
            self.drive_profile,
            start_frame=start_frame,
        )

    def _noise_block(self, start_frame: int, frames: int, *, salt: int) -> FloatArray:
        if frames <= 0:
            return np.zeros(0, dtype=np.float64)
        mask = (1 << 64) - 1
        positions = np.arange(start_frame, start_frame + frames, dtype=np.uint64)
        seed_offset = (0x9E3779B97F4A7C15 * max(1, salt)) & mask
        state = positions + np.uint64(self.seed) + np.uint64(seed_offset)
        values = np.zeros(frames, dtype=np.float64)
        for factor in (0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0xD6E8FEB86659FD93):
            mixed = state + np.uint64(factor)
            mixed ^= mixed >> np.uint64(30)
            mixed *= np.uint64(0xBF58476D1CE4E5B9)
            mixed ^= mixed >> np.uint64(27)
            mixed *= np.uint64(0x94D049BB133111EB)
            mixed ^= mixed >> np.uint64(31)
            values += ((mixed >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))) * 2.0 - 1.0
        return values / 3.0

    def _prepare_scheduled_events(
        self,
        frames: int,
        scheduled_events: Sequence[ScheduledEvent],
    ) -> list[ScheduledEvent]:
        all_events = [*self._deferred_events, *scheduled_events]
        immediate: list[ScheduledEvent] = []
        deferred: list[ScheduledEvent] = []
        for event, start_frame in all_events:
            frame_index = int(start_frame)
            if frame_index >= frames:
                deferred.append((event, frame_index - frames))
            else:
                immediate.append((event, max(0, frame_index)))
        self._deferred_events = deferred
        immediate.sort(key=lambda item: item[1])
        return immediate

    def render_chunk(
        self,
        frames: int,
        scheduled_events: Sequence[ScheduledEvent] = (),
    ) -> FloatArray:
        if frames <= 0:
            return np.zeros(0, dtype=np.float64)

        frame_origin = self.state.sample_clock
        bearing_noise = self._noise_block(frame_origin, frames, salt=17)
        windage_noise = self._noise_block(frame_origin, frames, salt=29)
        scheduled_now = self._prepare_scheduled_events(frames, scheduled_events)
        self.state, chunk = render_audio_chunk(
            self.state,
            self.mode_bank,
            self.drive_profile,
            self.acoustic_profile,
            frames,
            scheduled_events=scheduled_now,
            bearing_noise_raw=bearing_noise,
            windage_noise_raw=windage_noise,
        )
        return chunk

    def render_diagnostic_chunk(
        self,
        frames: int,
        scheduled_events: Sequence[ScheduledEvent] = (),
    ) -> AudioDiagnosticTrace:
        if frames <= 0:
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

        frame_origin = self.state.sample_clock
        bearing_noise = self._noise_block(frame_origin, frames, salt=17)
        windage_noise = self._noise_block(frame_origin, frames, salt=29)
        scheduled_now = self._prepare_scheduled_events(frames, scheduled_events)
        self.state, _chunk, diagnostics = render_audio_diagnostic_chunk(
            self.state,
            self.mode_bank,
            self.drive_profile,
            self.acoustic_profile,
            frames,
            scheduled_events=scheduled_now,
            bearing_noise_raw=bearing_noise,
            windage_noise_raw=windage_noise,
        )
        return diagnostics


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
        tee_path: str | None = None,
        event_trace_sink: StorageEventSink | None = None,
        deps: RuntimeDeps | None = None,
    ) -> None:
        self.deps = deps or RuntimeDeps()
        self.clock = self.deps.clock
        self.env = self.deps.env
        self.fs = sample_rate
        self.chunk_size = 1024
        self.events = StorageEventBus(max_pending=max_pending_events)
        self.synthesizer = HDDAudioSynthesizer(
            sample_rate=sample_rate,
            seed=seed,
            drive_profile=drive_profile,
            acoustic_profile=acoustic_profile,
            deps=self.deps,
        )
        self.stream: Any | None = None
        self.render_lock = threading.Lock()
        self.last_render_at: float | None = None
        self.time_origin: float | None = None
        self.render_frame_cursor = 0
        self.output_enabled = True
        self.tee_path: str | None = tee_path
        self.tee_recorder: _WaveTeeRecorder | None = None
        self._headless_stop_event = threading.Event()
        self._headless_render_thread: threading.Thread | None = None
        self.event_trace_sink = event_trace_sink
        self.configure_tee(tee_path)

    def configure_tee(self, tee_path: str | None) -> None:
        normalized = None
        if tee_path is not None:
            candidate = tee_path.strip()
            if candidate:
                normalized = candidate
        if normalized == self.tee_path and ((normalized is None) == (self.tee_recorder is None)):
            return
        if self.tee_recorder is not None:
            self.tee_recorder.close()
            self.tee_recorder = None
        self.tee_path = normalized
        if normalized is not None:
            self.tee_recorder = _WaveTeeRecorder(normalized, self.fs)

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
        servo_mode = "track" if is_seq else None
        if is_park:
            servo_mode = "park"
        elif seek_trigger:
            servo_mode = "seek"
        elif is_cal:
            servo_mode = "calibration"

        self.publish_event(
            HDDAudioEvent(
                rpm=rpm,
                emitted_at=self.clock.now(),
                target_rpm=rpm,
                queue_depth=queue_depth,
                op_kind=op_kind,
                is_sequential=is_seq,
                is_flush=is_flush,
                is_spinup=is_spinup,
                power_state="active" if rpm > 0.0 else "standby",
                heads_loaded=not is_park,
                servo_mode=servo_mode,
                track_delta=min(max(seek_dist / 1200.0, 0.0), 1.0) if seek_trigger else 0.0,
                motion_duration_ms=2.4 if seek_trigger else 0.0,
                settle_duration_ms=1.5 if (seek_trigger or is_cal or is_park) else 0.0,
                transfer_activity=(
                    0.86 if is_flush else 0.70 if op_kind == "writeback" else 0.62 if op_kind == "data" else 0.44
                )
                * (1.10 if is_seq else 1.0)
                * (1.0 + 0.05 * max(queue_depth - 1, 0)),
                seek_distance=seek_dist,
            )
        )

    def publish_event(self, event: HDDAudioEvent) -> None:
        self.events.publish(event)
        if self.event_trace_sink is not None:
            self.event_trace_sink.publish_event(event)

    def _update_telemetry(self, *args: Any, **kwargs: Any) -> None:
        self.emit_telemetry(*args, **kwargs)

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

        if self.time_origin is None:
            self.time_origin = min(event.emitted_at for event in events)

        scheduled = []
        for event in events:
            absolute_frame = round((event.emitted_at - self.time_origin) * self.fs)
            frame_offset = max(0, absolute_frame - self.render_frame_cursor)
            scheduled.append((event, frame_offset))
        scheduled.sort(key=lambda item: item[1])
        return scheduled

    def render_chunk(self, frames: int) -> FloatArray:
        with self.render_lock:
            scheduled_events = self._schedule_events_for_chunk(frames)
            chunk = self.synthesizer.render_chunk(frames, scheduled_events=scheduled_events)
            if self.tee_recorder is not None:
                self.tee_recorder.write_chunk(chunk)
            self.render_frame_cursor += frames
            self.last_render_at = self.clock.now()
            return chunk

    def render_chunk_with_diagnostics(self, frames: int) -> tuple[FloatArray, AudioDiagnosticTrace]:
        with self.render_lock:
            scheduled_events = self._schedule_events_for_chunk(frames)
            diagnostics = self.synthesizer.render_diagnostic_chunk(frames, scheduled_events=scheduled_events)
            if self.tee_recorder is not None:
                self.tee_recorder.write_chunk(diagnostics.output)
            self.render_frame_cursor += frames
            self.last_render_at = self.clock.now()
            return diagnostics.output, diagnostics

    def render_diagnostics(
        self,
        total_frames: int,
        *,
        chunk_size: int | None = None,
    ) -> AudioDiagnosticTrace:
        with self.render_lock:
            frames_remaining = max(0, total_frames)
            frames_per_chunk = max(1, chunk_size or self.chunk_size)
            diagnostics: list[AudioDiagnosticTrace] = []
            while frames_remaining > 0:
                frames = min(frames_remaining, frames_per_chunk)
                scheduled_events = self._schedule_events_for_chunk(frames)
                diagnostics.append(
                    self.synthesizer.render_diagnostic_chunk(
                        frames,
                        scheduled_events=scheduled_events,
                    )
                )
                self.render_frame_cursor += frames
                frames_remaining -= frames

            if self.tee_recorder is not None and diagnostics:
                self.tee_recorder.write_chunk(np.concatenate([trace.output for trace in diagnostics]))
            self.last_render_at = self.clock.now()

        if not diagnostics:
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
            time_s=np.concatenate([trace.time_s for trace in diagnostics]),
            target_rpm=np.concatenate([trace.target_rpm for trace in diagnostics]),
            actual_rpm=np.concatenate([trace.actual_rpm for trace in diagnostics]),
            actuator_pos=np.concatenate([trace.actuator_pos for trace in diagnostics]),
            actuator_torque=np.concatenate([trace.actuator_torque for trace in diagnostics]),
            structure_base_velocity=np.concatenate([trace.structure_base_velocity for trace in diagnostics]),
            structure_cover_velocity=np.concatenate([trace.structure_cover_velocity for trace in diagnostics]),
            structure_enclosure_velocity=np.concatenate([trace.structure_enclosure_velocity for trace in diagnostics]),
            structure_desk_velocity=np.concatenate([trace.structure_desk_velocity for trace in diagnostics]),
            output=np.concatenate([trace.output for trace in diagnostics]),
        )

    def export_diagnostics_json(
        self,
        path: str,
        total_frames: int,
        *,
        chunk_size: int | None = None,
    ) -> AudioDiagnosticTrace:
        diagnostics = self.render_diagnostics(total_frames, chunk_size=chunk_size)
        payload = {
            "sample_rate": self.fs,
            "time_s": diagnostics.time_s.tolist(),
            "target_rpm": diagnostics.target_rpm.tolist(),
            "actual_rpm": diagnostics.actual_rpm.tolist(),
            "actuator_pos": diagnostics.actuator_pos.tolist(),
            "actuator_torque": diagnostics.actuator_torque.tolist(),
            "structure_base_velocity": diagnostics.structure_base_velocity.tolist(),
            "structure_cover_velocity": diagnostics.structure_cover_velocity.tolist(),
            "structure_enclosure_velocity": diagnostics.structure_enclosure_velocity.tolist(),
            "structure_desk_velocity": diagnostics.structure_desk_velocity.tolist(),
            "output": diagnostics.output.tolist(),
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return diagnostics

    def _audio_callback(self, outdata: Any, frames: int, _time_info: Any, status: Any) -> None:
        outdata[:] = self.render_chunk(frames).reshape(-1, 1)

    def _reset_render_clock(self) -> None:
        self.time_origin = self.clock.now()
        self.render_frame_cursor = 0
        self.last_render_at = self.clock.now()

    def _headless_render_loop(self) -> None:
        frame_period_s = self.chunk_size / self.fs
        while not self._headless_stop_event.is_set():
            started_at = self.clock.now()
            self.render_chunk(self.chunk_size)
            elapsed_s = max(0.0, self.clock.now() - started_at)
            self._headless_stop_event.wait(max(0.0, frame_period_s - elapsed_s))

    def _start_headless_render_loop(self) -> None:
        if self.tee_recorder is None or self._headless_render_thread is not None:
            return
        self._headless_stop_event.clear()
        self._reset_render_clock()
        self._headless_render_thread = threading.Thread(
            target=self._headless_render_loop,
            name="clatterdrive-audio-tee",
            daemon=True,
        )
        self._headless_render_thread.start()

    def _stop_headless_render_loop(self) -> None:
        if self._headless_render_thread is None:
            return
        self._headless_stop_event.set()
        self._headless_render_thread.join(timeout=2.0)
        self._headless_render_thread = None

    def start(self) -> None:
        if self.stream is not None or self._headless_render_thread is not None:
            return
        resolved_drive, resolved_acoustic = resolve_selected_profiles_from_env(
            self.synthesizer.drive_profile,
            self.synthesizer.acoustic_profile,
            env=self.env,
        )
        self.configure_profiles(
            drive_profile=resolved_drive,
            acoustic_profile=resolved_acoustic,
        )
        self.configure_tee(self.env.get("FAKE_HDD_AUDIO_TEE_PATH", self.tee_path))
        audio_setting = (self.env.get("FAKE_HDD_AUDIO", "live") or "live").strip().lower()
        if audio_setting in {"0", "off", "false", "disabled", "none"}:
            self.output_enabled = False
            self._start_headless_render_loop()
            return
        if sd is None:
            self.output_enabled = False
            self._start_headless_render_loop()
            return
        self.output_enabled = True
        stream_kwargs: dict[str, Any] = {
            "samplerate": self.fs,
            "channels": 1,
            "callback": self._audio_callback,
            "blocksize": self.chunk_size,
        }
        device = _parse_audio_device(self.env.get("FAKE_HDD_AUDIO_DEVICE"))
        if device is not None:
            stream_kwargs["device"] = device
        try:
            stream = sd.OutputStream(**stream_kwargs)
            stream.start()
        except Exception as exc:
            stream = locals().get("stream")
            if stream is not None:
                stream.close()
            pulse_server = self.env.get("PULSE_SERVER")
            detail = ""
            if device is not None:
                detail = f" (device={device!r})"
            elif pulse_server:
                detail = f" (PULSE_SERVER={pulse_server!r})"
            raise RuntimeError(
                "live audio output could not be opened"
                f"{detail}; use FAKE_HDD_AUDIO=off for headless runs or set "
                "FAKE_HDD_AUDIO_DEVICE/PULSE_SERVER for container-host audio bridging"
            ) from exc
        self.stream = stream
        self.time_origin = self.clock.now()
        self.render_frame_cursor = 0
        self.last_render_at = self.clock.now()

    def stop(self) -> None:
        self._stop_headless_render_loop()
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None
        if self.tee_recorder is not None:
            self.tee_recorder.close()
            self.tee_recorder = None
        self.last_render_at = None
        self.time_origin = None
        self.render_frame_cursor = 0

_runtime_engine: HDDAudioEngine | None = None


def get_runtime_engine() -> HDDAudioEngine:
    global _runtime_engine
    if _runtime_engine is None:
        _runtime_engine = HDDAudioEngine()
    return _runtime_engine
