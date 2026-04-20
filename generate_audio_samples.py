from __future__ import annotations

import random
import shutil
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from clatterdrive.audio import HDDAudioEngine
from clatterdrive.hdd import HDDLatencyModel, StartupTracePoint
from clatterdrive.profiles import AcousticProfile, DriveProfile


ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
DOCS_AUDIO_DIR = ROOT / "docs" / "audio"
FloatArray = npt.NDArray[np.float64]
ScenarioUpdater = Callable[[HDDAudioEngine, float, set[str]], None]
PowerOnTrace = tuple[tuple[StartupTracePoint, ...], float]


def _load_power_on_trace(drive_profile: str | DriveProfile | None = None) -> PowerOnTrace:
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        start_ready=False,
        drive_profile=drive_profile,
    )
    try:
        stages = model._build_startup_sequence("power_on")
        trace = tuple(model._build_startup_trace_from_stages("power_on", stages))
    finally:
        model.stop()
    total_s = (trace[-1].time_ms / 1000.0) if trace else 0.0
    return trace, total_s


POWER_ON_TRACE_CACHE: dict[str, PowerOnTrace] = {}


def _power_on_trace_for(engine: HDDAudioEngine) -> PowerOnTrace:
    profile_name = engine.synthesizer.drive_profile.name
    if profile_name not in POWER_ON_TRACE_CACHE:
        POWER_ON_TRACE_CACHE[profile_name] = _load_power_on_trace(profile_name)
    return POWER_ON_TRACE_CACHE[profile_name]


def render_chunk(engine: HDDAudioEngine, frames: int) -> FloatArray:
    return engine.render_chunk(frames)


def write_wav(path: Path, samples: FloatArray, sample_rate: int) -> None:
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def normalize_demo_audio(samples: FloatArray, *, target_peak: float = 0.92) -> FloatArray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 1e-9:
        return samples
    scale = target_peak / peak
    return np.clip(samples * scale, -0.995, 0.995)


def mirror_demo_sample_to_docs(path: Path) -> None:
    DOCS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, DOCS_AUDIO_DIR / path.name)


def render_scenario(
    name: str,
    duration_s: float,
    update_func: ScenarioUpdater,
    seed: int = 7,
    drive_profile: str | DriveProfile | None = None,
    acoustic_profile: str | AcousticProfile | None = None,
    normalize_peak: float | None = None,
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
    samples = np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.float64)
    if normalize_peak is not None:
        samples = normalize_demo_audio(samples, target_peak=normalize_peak)
    write_wav(output_path, samples, engine.fs)
    return output_path


def update_spinup_idle(engine: HDDAudioEngine, t: float, emitted_flags: set[str]) -> None:
    target_rpm = float(engine.synthesizer.drive_profile.rpm)
    power_on_trace, power_on_total_s = _power_on_trace_for(engine)
    if t < power_on_total_s and power_on_trace:
        step_ms = power_on_trace[1].time_ms - power_on_trace[0].time_ms if len(power_on_trace) > 1 else 20.0
        trace_index = min(round((t * 1000.0) / max(step_ms, 1.0)), len(power_on_trace) - 1)
        point = power_on_trace[trace_index]
        if point.seek_distance > 0.0:
            engine.emit_telemetry(
                point.rpm,
                seek_trigger=True,
                seek_dist=point.seek_distance,
                queue_depth=1,
                op_kind="metadata",
                is_spinup=point.is_spinup,
            )
        if point.is_calibration:
            engine.emit_telemetry(
                point.rpm,
                is_cal=True,
                queue_depth=1,
                op_kind="metadata",
                is_spinup=point.is_spinup,
            )
        if point.seek_distance <= 0.0 and not point.is_calibration:
            engine.emit_telemetry(
                point.rpm,
                queue_depth=1,
                op_kind="data",
                is_spinup=point.is_spinup,
            )
        return

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


def update_copy_heavy(engine: HDDAudioEngine, t: float, emitted_flags: set[str]) -> None:
    target_rpm = float(engine.synthesizer.drive_profile.rpm)
    step = int(t * 10)
    event_key = f"copy-{step}"
    if event_key in emitted_flags:
        return
    emitted_flags.add(event_key)
    seek_dist = 28 + ((step * 71) % 240)
    op_kind = "writeback" if step % 2 else "data"
    queue_depth = 3 + (step % 4)
    engine.emit_telemetry(
        target_rpm,
        seek_trigger=True,
        seek_dist=seek_dist,
        queue_depth=queue_depth,
        op_kind=op_kind,
        is_seq=(step % 3 == 0),
    )


def update_idle_to_standby_wake(engine: HDDAudioEngine, t: float, emitted_flags: set[str]) -> None:
    target_rpm = float(engine.synthesizer.drive_profile.rpm)
    if t < 1.0:
        engine.emit_telemetry(target_rpm, is_seq=True, queue_depth=1, op_kind="data")
        return
    if t < 3.4:
        if "park" not in emitted_flags:
            emitted_flags.add("park")
            engine.emit_telemetry(target_rpm, is_park=True, queue_depth=1, op_kind="metadata")
        return
    if "wake" not in emitted_flags:
        emitted_flags.add("wake")
        engine.emit_telemetry(600.0, queue_depth=1, op_kind="metadata", is_spinup=True)
        return
    if t < 5.4:
        engine.emit_telemetry(target_rpm, seek_trigger=True, seek_dist=180, queue_depth=2, op_kind="data")
        return
    engine.emit_telemetry(target_rpm, is_seq=True, queue_depth=2, op_kind="data")


def update_metadata_storm(engine: HDDAudioEngine, t: float, emitted_flags: set[str]) -> None:
    target_rpm = float(engine.synthesizer.drive_profile.rpm)
    step = int(t * 22)
    event_key = f"meta-{step}"
    if event_key in emitted_flags:
        return
    emitted_flags.add(event_key)
    if step % 11 == 0:
        engine.emit_telemetry(target_rpm, is_cal=True, queue_depth=1, op_kind="metadata")
        return
    if step % 7 == 0:
        engine.emit_telemetry(
            target_rpm,
            seek_trigger=True,
            seek_dist=220 + ((step * 19) % 120),
            queue_depth=3,
            op_kind="flush",
            is_flush=True,
        )
        return
    if step % 3 == 0:
        engine.emit_telemetry(
            target_rpm,
            seek_trigger=True,
            seek_dist=80 + ((step * 17) % 120),
            queue_depth=2,
            op_kind="journal",
        )
        return
    engine.emit_telemetry(
        target_rpm,
        seek_trigger=True,
        seek_dist=28 + ((step * 41) % 220),
        queue_depth=2 + (step % 3),
        op_kind="metadata",
    )


EXTRA_SCENARIOS: tuple[tuple[str, float, ScenarioUpdater, int], ...] = (
    ("copy-heavy-writeback", 6.0, update_copy_heavy, 17),
    ("idle-standby-wake", 7.0, update_idle_to_standby_wake, 19),
    ("metadata-storm", 6.0, update_metadata_storm, 23),
)


def generate_readme_demo_samples() -> list[Path]:
    random.seed(7)
    np.random.seed(7)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        render_scenario(
            "spinup-idle-park",
            _load_power_on_trace()[1] + 3.0,
            update_spinup_idle,
            seed=7,
            acoustic_profile="drive_on_desk",
            normalize_peak=0.92,
        ),
        render_scenario(
            "idle-standby-wake",
            7.0,
            update_idle_to_standby_wake,
            seed=19,
            acoustic_profile="drive_on_desk",
            normalize_peak=0.92,
        ),
        render_scenario(
            "metadata-storm",
            6.0,
            update_metadata_storm,
            seed=23,
            acoustic_profile="bare_drive_lab",
            normalize_peak=0.92,
        ),
    ]
    for output in outputs:
        mirror_demo_sample_to_docs(output)
    return outputs


def generate_extended_scenario_samples() -> list[Path]:
    random.seed(7)
    np.random.seed(7)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    return [
        render_scenario(name, duration_s, update_func, seed=seed)
        for name, duration_s, update_func, seed in EXTRA_SCENARIOS
    ]


def main() -> None:
    outputs = generate_readme_demo_samples()

    for output in outputs:
        print(f"generated {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
