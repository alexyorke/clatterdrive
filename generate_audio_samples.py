from __future__ import annotations

import random
import wave
from pathlib import Path

import numpy as np

from audio_engine import HDDAudioEngine
from hdd_model import HDDLatencyModel


ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"


def _load_power_on_sequence():
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0, start_ready=False)
    try:
        stages = tuple(model._build_startup_sequence("power_on"))
    finally:
        model.stop()
    total_s = sum(stage.duration_ms for stage in stages) / 1000.0
    return stages, total_s


POWER_ON_STAGES, POWER_ON_TOTAL_S = _load_power_on_sequence()


def render_chunk(engine: HDDAudioEngine, frames: int) -> np.ndarray:
    return engine.render_chunk(frames).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def render_scenario(name: str, duration_s: float, update_func, seed: int = 7) -> Path:
    engine = HDDAudioEngine(sample_rate=44100, seed=seed)
    total_frames = int(duration_s * engine.fs)
    rendered = []
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
    stage_start = 0.0
    for stage in POWER_ON_STAGES:
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

            engine._update_telemetry(
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

    idle_elapsed = t - POWER_ON_TOTAL_S
    if idle_elapsed < 2.0:
        calibrate = False
        if idle_elapsed >= 0.9 and "idle-cal" not in emitted_flags:
            emitted_flags.add("idle-cal")
            calibrate = True
        engine._update_telemetry(7200.0, is_cal=calibrate, queue_depth=1, op_kind="metadata")
        return

    park = False
    if "park" not in emitted_flags:
        emitted_flags.add("park")
        park = True
    engine._update_telemetry(7200.0, is_park=park, queue_depth=1, op_kind="metadata")


def update_sequential_read(engine: HDDAudioEngine, t: float, emitted_flags: set[str]) -> None:
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

    engine._update_telemetry(
        7200.0,
        seek_trigger=seek,
        seek_dist=seek_dist,
        is_seq=True,
        queue_depth=2,
        op_kind="data",
    )


def update_random_flush(engine: HDDAudioEngine, t: float, emitted_flags: set[str]) -> None:
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

    engine._update_telemetry(
        7200.0,
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
        render_scenario("spinup-idle-park", POWER_ON_TOTAL_S + 3.0, update_spinup_idle, seed=7),
        render_scenario("sequential-read-stream", 6.0, update_sequential_read, seed=11),
        render_scenario("random-seek-journal-flush", 6.0, update_random_flush, seed=13),
    ]

    for output in outputs:
        print(f"generated {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
