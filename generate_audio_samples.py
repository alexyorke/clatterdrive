from __future__ import annotations

import random
import wave
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from audio_engine import HDDAudioEngine
from hdd_model import HDDLatencyModel, StartupStage
from profiles import AcousticProfile, DriveProfile


ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
FloatArray = npt.NDArray[np.float64]
ScenarioUpdater = Callable[[HDDAudioEngine, float, set[str]], None]
PowerOnSequence = tuple[tuple[StartupStage, ...], float]


def _load_power_on_sequence(drive_profile: str | DriveProfile | None = None) -> PowerOnSequence:
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        start_ready=False,
        drive_profile=drive_profile,
    )
    try:
        stages = tuple(model._build_startup_sequence("power_on"))
    finally:
        model.stop()
    total_s = sum(stage.duration_ms for stage in stages) / 1000.0
    return stages, total_s


POWER_ON_STAGE_CACHE: dict[str, PowerOnSequence] = {}


def _power_on_sequence_for(engine: HDDAudioEngine) -> PowerOnSequence:
    profile_name = engine.synthesizer.drive_profile.name
    if profile_name not in POWER_ON_STAGE_CACHE:
        POWER_ON_STAGE_CACHE[profile_name] = _load_power_on_sequence(profile_name)
    return POWER_ON_STAGE_CACHE[profile_name]


def render_chunk(engine: HDDAudioEngine, frames: int) -> FloatArray:
    return engine.render_chunk(frames).astype(np.float32)


def write_wav(path: Path, samples: FloatArray, sample_rate: int) -> None:
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def render_scenario(
    name: str,
    duration_s: float,
    update_func: ScenarioUpdater,
    seed: int = 7,
    drive_profile: str | DriveProfile | None = None,
    acoustic_profile: str | AcousticProfile | None = None,
) -> Path:
    engine = HDDAudioEngine(
        sample_rate=44100,
        seed=seed,
        drive_profile=drive_profile,
        acoustic_profile=acoustic_profile,
    )
    total_frames = int(duration_s * engine.fs)
    rendered: list[FloatArray] = []
    emitted_flags: set[str] = set()

    while total_frames > 0:
        frames = min(engine.chunk_size, total_frames)
        chunk_index = len(rendered)
        current_time = (chunk_index * engine.chunk_size) / engine.fs
        update_func(engine, current_time, emitted_flags)
        rendered.append(render_chunk(engine, frames))
        total_frames -= frames

    output_path = SAMPLES_DIR / f"{name}.wav"
    samples = np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.float32)
    write_wav(output_path, samples, engine.fs)
    return output_path


def update_spinup_idle(engine: HDDAudioEngine, t: float, emitted_flags: set[str]) -> None:
    target_rpm = float(engine.synthesizer.drive_profile.rpm)
    power_on_stages, power_on_total_s = _power_on_sequence_for(engine)
    stage_start = 0.0
    for stage in power_on_stages:
        stage_duration_s = stage.duration_ms / 1000.0
        stage_end = stage_start + stage_duration_s
        if t < stage_end:
            stage_progress = 1.0 if stage_duration_s <= 0.0 else max(0.0, min((t - stage_start) / stage_duration_s, 1.0))
            rpm = stage.start_rpm + (stage.end_rpm - stage.start_rpm) * stage_progress
            calibrate = False
            if stage.calibration_pulses > 0:
                for pulse_index in range(stage.calibration_pulses):
                    pulse_t = stage_start + stage_duration_s * ((pulse_index + 1) / (stage.calibration_pulses + 1))
                    pulse_key = f"{stage.name}-cal-{pulse_index}"
                    if t >= pulse_t and pulse_key not in emitted_flags:
                        emitted_flags.add(pulse_key)
                        calibrate = True

            head_load = False
            if stage.head_load:
                head_load_key = f"{stage.name}-head-load"
                if head_load_key not in emitted_flags:
                    emitted_flags.add(head_load_key)
                    head_load = True

            engine.emit_telemetry(
                rpm,
                seek_trigger=head_load,
                seek_dist=28 if head_load else 0,
                is_cal=calibrate,
                queue_depth=1,
                op_kind="metadata" if (calibrate or head_load) else "data",
                is_spinup=stage.name in {"spinup", "rpm_recover", "spindle_unlock"},
            )
            return
        stage_start = stage_end

    idle_elapsed = t - power_on_total_s
    if idle_elapsed < 2.0:
        calibrate = False
        if idle_elapsed >= 0.9 and "idle-cal" not in emitted_flags:
            emitted_flags.add("idle-cal")
            calibrate = True
        engine.emit_telemetry(target_rpm, is_cal=calibrate, queue_depth=1, op_kind="metadata")
        return

    park = False
    if "park" not in emitted_flags:
        emitted_flags.add("park")
        park = True
    engine.emit_telemetry(target_rpm, is_park=park, queue_depth=1, op_kind="metadata")


def update_sequential_read(engine: HDDAudioEngine, t: float, emitted_flags: set[str]) -> None:
    target_rpm = float(engine.synthesizer.drive_profile.rpm)
    seek = False
    seek_dist = 0

    if "initial-seek" not in emitted_flags:
        emitted_flags.add("initial-seek")
        seek = True
        seek_dist = 220
    elif abs((t % 1.0) - 0.0) < 0.03 and f"boundary-{int(t)}" not in emitted_flags:
        emitted_flags.add(f"boundary-{int(t)}")
        seek = True
        seek_dist = 6

    engine.emit_telemetry(
        target_rpm,
        seek_trigger=seek,
        seek_dist=seek_dist,
        is_seq=True,
        queue_depth=2,
        op_kind="data",
    )


def update_random_flush(engine: HDDAudioEngine, t: float, emitted_flags: set[str]) -> None:
    target_rpm = float(engine.synthesizer.drive_profile.rpm)
    step = int(t * 12)
    should_seek = False
    seek_dist = 0
    op_kind = "data"
    is_flush = False
    queue_depth = 4 + (step % 5)

    event_key = f"event-{step}"
    if event_key not in emitted_flags:
        emitted_flags.add(event_key)
        should_seek = True

        if step % 7 == 0:
            op_kind = "flush"
            is_flush = True
            seek_dist = 260
        elif step % 3 == 0:
            op_kind = "journal"
            seek_dist = 80
        else:
            op_kind = "data"
            seek_dist = 40 + ((step * 53) % 700)

    engine.emit_telemetry(
        target_rpm,
        seek_trigger=should_seek,
        seek_dist=seek_dist,
        queue_depth=queue_depth,
        op_kind=op_kind,
        is_flush=is_flush,
    )


def main() -> None:
    random.seed(7)
    np.random.seed(7)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        render_scenario(
            "spinup-idle-park",
            _load_power_on_sequence()[1] + 3.0,
            update_spinup_idle,
            seed=7,
        ),
        render_scenario("sequential-read-stream", 6.0, update_sequential_read, seed=11),
        render_scenario("random-seek-journal-flush", 6.0, update_random_flush, seed=13),
    ]

    for output in outputs:
        print(f"generated {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
