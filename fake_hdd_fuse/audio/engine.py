from __future__ import annotations

import threading
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from .core import (
    AudioModeBank,
    AudioRenderState,
    ScheduledEvent,
    apply_event as apply_audio_event,
    build_mode_bank,
    initialize_render_state,
    make_impulse_pulses,
    reinitialize_mode_state,
    render_chunk as render_audio_chunk,
)
from ..profiles import (
    AcousticProfile,
    DriveProfile,
    resolve_selected_profiles,
    resolve_selected_profiles_from_env,
)
from ..runtime.deps import RuntimeDeps
from ..storage_events import StorageEvent, StorageEventBus


FloatArray = npt.NDArray[np.float64]
HDDAudioEvent = StorageEvent
HDDAudioEventBus = StorageEventBus


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
        self.rng = self.deps.rng_factory.create(seed)
        self.drive_profile: DriveProfile
        self.acoustic_profile: AcousticProfile
        self.mode_bank: AudioModeBank
        self.state: AudioRenderState

        resolved_drive, resolved_acoustic = resolve_selected_profiles(drive_profile, acoustic_profile)
        self.drive_profile = resolved_drive
        self.acoustic_profile = resolved_acoustic
        self.mode_bank = build_mode_bank(self.drive_profile)
        self.state = initialize_render_state(self.fs, self.mode_bank, self.acoustic_profile)

    @property
    def rpm(self) -> float:
        return self.state.rpm

    @property
    def pending_impulses(self) -> tuple[Any, ...]:
        return self.state.pending_impulses

    def configure_profiles(
        self,
        drive_profile: str | DriveProfile | None = None,
        acoustic_profile: str | AcousticProfile | None = None,
    ) -> None:
        resolved_drive, resolved_acoustic = resolve_selected_profiles(drive_profile, acoustic_profile)
        self.drive_profile = resolved_drive
        self.acoustic_profile = resolved_acoustic
        self.mode_bank = build_mode_bank(self.drive_profile)
        self.state = reinitialize_mode_state(self.state, self.mode_bank, self.acoustic_profile)

    def apply_event(self, event: HDDAudioEvent, start_frame: int = 0) -> None:
        self.state = apply_audio_event(
            self.state,
            event,
            self.acoustic_profile,
            start_frame=start_frame,
        )

    def _make_impulse_pulses(
        self,
        impulse: str,
        start_frame: int,
        seek_distance: float,
        op_kind: str,
        is_flush: bool,
    ) -> tuple[Any, ...]:
        return make_impulse_pulses(
            impulse,
            start_frame,
            seek_distance,
            op_kind,
            is_flush,
            self.fs,
            self.acoustic_profile,
        )

    def render_chunk(
        self,
        frames: int,
        scheduled_events: Sequence[ScheduledEvent] = (),
    ) -> FloatArray:
        if frames <= 0:
            return np.zeros(0, dtype=np.float64)

        bearing_noise = self.rng.normal(0.0, 1.0, frames).astype(np.float64, copy=False)
        windage_noise = self.rng.normal(0.0, 1.0, frames).astype(np.float64, copy=False)
        self.state, chunk = render_audio_chunk(
            self.state,
            self.mode_bank,
            self.drive_profile,
            self.acoustic_profile,
            frames,
            scheduled_events=scheduled_events,
            bearing_noise_raw=bearing_noise,
            windage_noise_raw=windage_noise,
        )
        return chunk


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
        self.output_enabled = True
        self.tee_path: str | None = tee_path
        self.tee_recorder: _WaveTeeRecorder | None = None
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
        impulse = None
        if is_park:
            impulse = "park"
        elif seek_trigger:
            impulse = "seek"
        elif is_cal:
            impulse = "calibration"

        self.publish_event(
            HDDAudioEvent(
                rpm=rpm,
                emitted_at=self.clock.now(),
                queue_depth=queue_depth,
                op_kind=op_kind,
                is_sequential=is_seq,
                is_flush=is_flush,
                is_spinup=is_spinup,
                impulse=impulse,
                seek_distance=seek_dist,
            )
        )

    def publish_event(self, event: HDDAudioEvent) -> None:
        self.events.publish(event)

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

        if self.stream is None:
            return [(event, 0) for event in events]

        now = self.clock.now()
        chunk_duration_s = frames / self.fs
        chunk_start = self.last_render_at
        if chunk_start is None:
            chunk_start = now - chunk_duration_s
        scheduled = []
        for event in events:
            frame_offset = round((event.emitted_at - chunk_start) * self.fs)
            frame_offset = max(0, min(frames - 1, frame_offset))
            scheduled.append((event, frame_offset))
        scheduled.sort(key=lambda item: item[1])
        return scheduled

    def render_chunk(self, frames: int) -> FloatArray:
        with self.render_lock:
            scheduled_events = self._schedule_events_for_chunk(frames)
            chunk = self.synthesizer.render_chunk(frames, scheduled_events=scheduled_events)
            if self.tee_recorder is not None:
                self.tee_recorder.write_chunk(chunk)
            self.last_render_at = self.clock.now()
            return chunk

    def _audio_callback(self, outdata: Any, frames: int, _time_info: Any, status: Any) -> None:
        outdata[:] = self.render_chunk(frames).reshape(-1, 1)

    def start(self) -> None:
        if self.stream is not None:
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
        self.last_render_at = self.clock.now()

    def stop(self) -> None:
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


engine = HDDAudioEngine()
