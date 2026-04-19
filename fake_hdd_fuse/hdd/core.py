from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..profiles import DriveProfile


@dataclass(frozen=True)
class Zone:
    start_cyl: int
    end_cyl: int
    start_lba: int
    end_lba: int
    blocks_per_track: int
    transfer_rate_mbps: float


@dataclass(frozen=True)
class CacheSpan:
    start_lba: int
    end_lba: int
    expires_at: float


@dataclass(frozen=True)
class StartupStage:
    name: str
    duration_ms: float
    start_rpm: float
    end_rpm: float
    calibration_pulses: int = 0
    head_load: bool = False
    head_unload: bool = False
    park: bool = False


@dataclass(frozen=True)
class StartupTracePoint:
    time_ms: float
    stage: str
    rpm: float
    heads_loaded: bool
    is_spinup: bool
    self_test_active: bool
    servo_locked: bool
    is_calibration: bool
    seek_distance: float = 0.0
    head_load_event: bool = False
    park_event: bool = False


@dataclass(frozen=True)
class CacheState:
    spans: tuple[CacheSpan, ...] = ()
    last_read_end_lba: int = -1


@dataclass(frozen=True)
class MechanicalState:
    current_rpm: float
    current_cyl: int
    current_head: int
    current_sector: int
    power_state: str
    heads_loaded: bool
    has_completed_power_on: bool
    last_access_time: float
    load_unload_count: int


@dataclass(frozen=True)
class TransitionState:
    kind: str | None = None
    origin: str | None = None
    stage: str | None = None
    total_ms: float = 0.0


@dataclass(frozen=True)
class HDDCoreState:
    mechanical: MechanicalState
    transition: TransitionState = TransitionState()
    cache: CacheState = CacheState()
    last_idle_calibration_time: float = 0.0


@dataclass(frozen=True)
class HDDCoreConfig:
    drive_profile: DriveProfile
    block_bytes: int
    addressable_blocks: int
    target_rpm: float
    num_heads: int
    ms_per_rotation: float
    avg_seek_ms: float
    track_to_track_ms: float
    settle_ms: float
    head_switch_ms: float
    read_ahead_blocks: int
    low_rpm_rpm: float
    unload_to_ready_ms: float
    low_rpm_to_ready_ms: float
    standby_to_ready_ms: float
    power_on_to_ready_ms: float
    spin_down_ms: float
    blocks_per_track_outer: int
    blocks_per_track_inner: int
    total_cylinders: int
    seek_curve_b: float
    zones: tuple[Zone, ...]
    command_overhead_ms: float
    command_overhead_by_kind: dict[str, float]
    queue_depth_penalty_ms: float
    track_skew_blocks: int
    cylinder_skew_blocks: int
    aam_seek_penalty: float


@dataclass(frozen=True)
class ReadyPollPlan:
    startup_ms: float = 0.0
    startup_origin: str | None = None
    ready_poll_ms: float = 0.0
    ready_poll_count: int = 0


@dataclass(frozen=True)
class BackgroundDecision:
    rpm: float
    park: bool = False
    calibrate: bool = False
    should_low_rpm: bool = False
    should_spindown: bool = False
    next_power_state: str | None = None
    next_heads_loaded: bool | None = None
    load_unload_delta: int = 0
    next_idle_calibration_time: float | None = None


@dataclass(frozen=True)
class OperationStats:
    total_ms: float = 0.0
    cache_hit: bool = True
    partial_hit: bool = False
    extents: int = 0
    cyl: int | str = "-"
    head: int | str = "-"
    startup_ms: float = 0.0
    startup_origin: str | None = None
    ready_poll_ms: float = 0.0
    ready_poll_count: int = 0
    op_type: str = ""
    transfer_rate_mbps: float | None = None

    def __getitem__(self, key: str) -> Any:
        if key == "type":
            return self.op_type
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default

    def with_updates(self, **changes: Any) -> OperationStats:
        if "type" in changes:
            changes["op_type"] = changes.pop("type")
        return replace(self, **changes)


def empty_operation_stats(op_type: str, total_ms: float = 0.01) -> OperationStats:
    return OperationStats(total_ms=total_ms, cache_hit=True, extents=0, op_type=op_type)


def merge_operation_stats(op_type: str, *results: OperationStats | None) -> OperationStats:
    saw_result = False
    combined = OperationStats(op_type=op_type)
    for result in results:
        if result is None:
            continue
        saw_result = True
        combined = OperationStats(
            total_ms=combined.total_ms + result.total_ms,
            cache_hit=combined.cache_hit and result.cache_hit,
            partial_hit=combined.partial_hit or result.partial_hit,
            extents=combined.extents + result.extents,
            cyl=result.cyl if result.cyl != "-" else combined.cyl,
            head=result.head if result.head != "-" else combined.head,
            startup_ms=combined.startup_ms + result.startup_ms,
            startup_origin=combined.startup_origin or result.startup_origin,
            ready_poll_ms=combined.ready_poll_ms + result.ready_poll_ms,
            ready_poll_count=combined.ready_poll_count + result.ready_poll_count,
            op_type=op_type,
            transfer_rate_mbps=result.transfer_rate_mbps
            if result.transfer_rate_mbps is not None
            else combined.transfer_rate_mbps,
        )
    if not saw_result:
        return empty_operation_stats(op_type)
    return combined


def build_zones(
    *,
    addressable_blocks: int,
    total_cylinders: int,
    num_heads: int,
    blocks_per_track_outer: int,
    blocks_per_track_inner: int,
    outer_rate: float,
    inner_rate: float,
    zone_count: int = 8,
) -> tuple[Zone, ...]:
    zones: list[Zone] = []
    cylinders_per_zone = math.ceil(total_cylinders / zone_count)
    current_lba = 0

    for idx in range(zone_count):
        if current_lba >= addressable_blocks:
            break
        start_cyl = idx * cylinders_per_zone
        if start_cyl >= total_cylinders:
            break

        end_cyl = min(total_cylinders - 1, ((idx + 1) * cylinders_per_zone) - 1)
        fraction = idx / max(zone_count - 1, 1)
        blocks_per_track = round(
            blocks_per_track_outer + (blocks_per_track_inner - blocks_per_track_outer) * fraction
        )
        transfer_rate = outer_rate + (inner_rate - outer_rate) * fraction
        zone_blocks = (end_cyl - start_cyl + 1) * num_heads * blocks_per_track
        zone_end = min(addressable_blocks - 1, current_lba + zone_blocks - 1)
        if zone_end < current_lba:
            break
        zones.append(
            Zone(
                start_cyl=start_cyl,
                end_cyl=end_cyl,
                start_lba=current_lba,
                end_lba=zone_end,
                blocks_per_track=blocks_per_track,
                transfer_rate_mbps=transfer_rate,
            )
        )
        current_lba = zone_end + 1
    return tuple(zones)


def build_resume_sequence(config: HDDCoreConfig, start_rpm: float, heads_loaded: bool) -> list[StartupStage]:
    stages: list[StartupStage] = []
    clamped_rpm = max(0.0, min(float(start_rpm), float(config.target_rpm)))

    if clamped_rpm < config.target_rpm:
        rpm_gap = max(0.0, (config.target_rpm - clamped_rpm) / config.target_rpm)
        rpm_recover_ms = max(180.0, config.low_rpm_to_ready_ms * max(rpm_gap, 0.15))
        stages.append(StartupStage("rpm_recover", rpm_recover_ms, clamped_rpm, config.target_rpm))

    if not heads_loaded:
        head_load_ms, servo_lock_ms = allocate_stage_durations(
            max(config.unload_to_ready_ms, 220.0),
            weights=[0.58, 0.42],
            minimums=[120.0, 80.0],
        )
        stages.append(
            StartupStage(
                "head_load",
                head_load_ms,
                config.target_rpm,
                config.target_rpm,
                head_load=True,
            )
        )
        stages.append(
            StartupStage(
                "servo_lock",
                servo_lock_ms,
                config.target_rpm,
                config.target_rpm,
                calibration_pulses=1,
            )
        )
    elif stages:
        stages.append(
            StartupStage(
                "servo_lock",
                max(120.0, config.settle_ms * 400.0),
                config.target_rpm,
                config.target_rpm,
                calibration_pulses=1,
            )
        )

    return stages


def resolve_startup_plan(
    config: HDDCoreConfig,
    mechanical: MechanicalState,
) -> tuple[str | None, list[StartupStage], float]:
    if not mechanical.has_completed_power_on:
        origin = "power_on"
        stages = build_startup_sequence(config, origin)
    elif mechanical.current_rpm <= 0.0:
        origin = "standby"
        stages = build_startup_sequence(config, origin)
    elif not mechanical.heads_loaded and mechanical.current_rpm >= config.target_rpm * 0.98:
        origin = "unloaded_idle"
        stages = build_startup_sequence(config, origin)
    elif not mechanical.heads_loaded and mechanical.current_rpm <= config.low_rpm_rpm * 1.05:
        origin = "low_rpm_idle"
        stages = build_startup_sequence(config, origin)
    elif mechanical.current_rpm < config.target_rpm:
        origin = "resume"
        stages = build_resume_sequence(config, mechanical.current_rpm, mechanical.heads_loaded)
    elif not mechanical.heads_loaded:
        origin = "unloaded_idle"
        stages = build_startup_sequence(config, origin)
    else:
        return None, [], 0.0

    return origin, stages, sum(stage.duration_ms for stage in stages)


def build_spindown_sequence(config: HDDCoreConfig, mechanical: MechanicalState) -> list[StartupStage]:
    if mechanical.current_rpm <= 0.0 and not mechanical.heads_loaded:
        return []

    stages = []
    current_rpm = max(0.0, mechanical.current_rpm)
    low_rpm_target = min(current_rpm, float(config.low_rpm_rpm))
    lock_entry_rpm = max(48.0, min(low_rpm_target * 0.18, current_rpm * 0.22))
    unload_end_rpm = max(
        low_rpm_target,
        current_rpm - max(current_rpm * 0.02, min(180.0, current_rpm * 0.05)),
    )

    if current_rpm > low_rpm_target + 1.0:
        durations = allocate_stage_durations(
            config.spin_down_ms,
            weights=[0.16, 0.24, 0.50, 0.10] if mechanical.heads_loaded else [0.34, 0.56, 0.10],
            minimums=[120.0, 180.0, 320.0, 90.0] if mechanical.heads_loaded else [180.0, 320.0, 90.0],
        )
        if mechanical.heads_loaded:
            unload_ms, brake_ms, coast_ms, lock_ms = durations
            stages.append(
                StartupStage(
                    "head_unload",
                    unload_ms,
                    current_rpm,
                    unload_end_rpm,
                    head_unload=True,
                    park=True,
                )
            )
        else:
            brake_ms, coast_ms, lock_ms = durations
        stages.append(
            StartupStage(
                "spindle_brake",
                brake_ms,
                unload_end_rpm if mechanical.heads_loaded else current_rpm,
                low_rpm_target,
            )
        )
        stages.append(StartupStage("coast_down", coast_ms, low_rpm_target, lock_entry_rpm))
        stages.append(StartupStage("spindle_lock", lock_ms, lock_entry_rpm, 0.0))
    else:
        durations = allocate_stage_durations(
            config.spin_down_ms,
            weights=[0.18, 0.64, 0.18] if mechanical.heads_loaded else [0.82, 0.18],
            minimums=[120.0, 240.0, 90.0] if mechanical.heads_loaded else [240.0, 90.0],
        )
        if mechanical.heads_loaded:
            unload_ms, coast_ms, lock_ms = durations
            stages.append(
                StartupStage(
                    "head_unload",
                    unload_ms,
                    current_rpm,
                    unload_end_rpm,
                    head_unload=True,
                    park=True,
                )
            )
        else:
            coast_ms, lock_ms = durations
        stages.append(
            StartupStage(
                "coast_down",
                coast_ms,
                unload_end_rpm if mechanical.heads_loaded else current_rpm,
                lock_entry_rpm,
            )
        )
        stages.append(StartupStage("spindle_lock", lock_ms, lock_entry_rpm, 0.0))

    return stages


def build_low_rpm_sequence(config: HDDCoreConfig, current_rpm: float) -> list[StartupStage]:
    clamped_rpm = max(float(current_rpm), float(config.low_rpm_rpm))
    if clamped_rpm <= config.low_rpm_rpm + 1.0:
        return []

    reduce_ms = max(
        220.0,
        config.spin_down_ms * max((clamped_rpm - config.low_rpm_rpm) / max(config.target_rpm, 1.0), 0.22),
    )
    return [StartupStage("rpm_reduce", reduce_ms, clamped_rpm, float(config.low_rpm_rpm))]


def startup_trace_step_ms(total_ms: float) -> float:
    return min(10.0, max(3.0, total_ms / 2200.0))


def rotational_drag_terms(config: HDDCoreConfig) -> tuple[float, float]:
    platter_factor = max(config.drive_profile.platters - 1, 0)
    medium_scale = 0.72 if config.drive_profile.helium else 1.0
    linear_drag = (0.10 + 0.018 * platter_factor) * medium_scale
    quadratic_drag = (0.42 + 0.055 * platter_factor) * medium_scale
    return linear_drag, quadratic_drag


def simulate_rotational_transition(
    config: HDDCoreConfig,
    start_rpm: float,
    end_rpm: float,
    duration_ms: float,
    step_ms: float,
) -> list[float]:
    total_ms = max(float(duration_ms), 0.0)
    step_ms = max(float(step_ms), 1.0)
    if total_ms <= 0.0 or math.isclose(start_rpm, end_rpm, abs_tol=1e-6):
        sample_count = max(1, math.ceil(total_ms / step_ms))
        return [float(end_rpm)] * (sample_count + 1)

    sample_count = max(1, math.ceil(total_ms / step_ms))
    duration_s = max(total_ms / 1000.0, 1e-6)
    sample_dt = max(step_ms / 1000.0, 0.001)
    integration_dt = min(0.002, max(sample_dt / 16.0, 0.0005))
    max_rpm = max(float(config.target_rpm), float(start_rpm), float(end_rpm), 1.0)
    start_norm = max(0.0, float(start_rpm) / max_rpm)
    end_norm = max(0.0, float(end_rpm) / max_rpm)
    linear_drag, quadratic_drag = rotational_drag_terms(config)
    drag_load = linear_drag * max(start_norm, end_norm) + quadratic_drag * max(start_norm, end_norm) ** 2
    mean_command = (end_norm - start_norm) / duration_s
    span = max(0.35, abs(mean_command) + drag_load + 0.35)
    lower_bound = mean_command - span
    upper_bound = mean_command + span

    def simulate(command: float) -> list[float]:
        norm = start_norm
        values = [float(start_rpm)]
        for _ in range(sample_count):
            elapsed_s = 0.0
            while elapsed_s < sample_dt - 1e-12:
                dt = min(integration_dt, sample_dt - elapsed_s)
                drag = linear_drag * norm + quadratic_drag * norm * abs(norm)
                norm = max(0.0, norm + (command - drag) * dt)
                elapsed_s += dt
            values.append(norm * max_rpm)
        return values

    low_samples = simulate(lower_bound)
    high_samples = simulate(upper_bound)
    for _ in range(12):
        low_final = low_samples[-1]
        high_final = high_samples[-1]
        if low_final <= float(end_rpm) <= high_final:
            break
        span *= 2.0
        lower_bound = mean_command - span
        upper_bound = mean_command + span
        low_samples = simulate(lower_bound)
        high_samples = simulate(upper_bound)

    best = high_samples
    for _ in range(40):
        command = (lower_bound + upper_bound) / 2.0
        candidate = simulate(command)
        if candidate[-1] < float(end_rpm):
            lower_bound = command
        else:
            upper_bound = command
            best = candidate

    best[-1] = float(end_rpm)
    return best


def simulate_spin_ramp(
    config: HDDCoreConfig,
    start_rpm: float,
    end_rpm: float,
    duration_ms: float,
    step_ms: float,
) -> list[float]:
    return simulate_rotational_transition(config, start_rpm, end_rpm, duration_ms, step_ms)


def simulate_spin_decay(
    config: HDDCoreConfig,
    start_rpm: float,
    end_rpm: float,
    duration_ms: float,
    step_ms: float,
) -> list[float]:
    return simulate_rotational_transition(config, start_rpm, end_rpm, duration_ms, step_ms)


def stage_bounds(
    stage_windows: Sequence[tuple[float, float, StartupStage]],
    *,
    name: str | None = None,
    head_load: bool = False,
) -> tuple[float | None, float | None, StartupStage | None]:
    for start_ms, end_ms, stage in stage_windows:
        if name is not None and stage.name == name:
            return start_ms, end_ms, stage
        if head_load and stage.head_load:
            return start_ms, end_ms, stage
    return None, None, None


def build_trace_times(
    total_ms: float,
    step_ms: float,
    stage_windows: Sequence[tuple[float, float, StartupStage]],
) -> list[float]:
    point_count = max(1, math.ceil(total_ms / step_ms))
    trace_times = {round(min(index * step_ms, total_ms), 6) for index in range(point_count + 1)}
    trace_times.add(round(total_ms, 6))
    for start_ms, end_ms, _stage in stage_windows:
        trace_times.add(round(start_ms, 6))
        trace_times.add(round(end_ms, 6))
    return [float(point) for point in sorted(trace_times)]


def startup_rpm_threshold(config: HDDCoreConfig, transition: str, *, start_rpm: float, end_rpm: float) -> float:
    target = float(config.target_rpm)
    if transition == "self_test":
        floor = target * 0.88
        ramp_fraction = 0.12
    elif transition == "head_load":
        floor = target * 0.972
        ramp_fraction = 0.55
    elif transition == "servo_lock":
        floor = target * 0.992
        ramp_fraction = 0.78
    else:
        return float(end_rpm)

    threshold = start_rpm + (end_rpm - start_rpm) * ramp_fraction
    return min(float(end_rpm), max(float(start_rpm), floor, threshold))


def build_startup_trace_from_stages(
    config: HDDCoreConfig,
    origin: str,
    stages: list[StartupStage],
    *,
    step_ms: float | None = None,
    initial_heads_loaded: bool = False,
) -> list[StartupTracePoint]:
    if not stages:
        return []

    total_ms = sum(stage.duration_ms for stage in stages)
    base_step_ms = startup_trace_step_ms(total_ms)
    trace_step_ms = base_step_ms if step_ms is None else min(float(step_ms), base_step_ms)
    stage_windows: list[tuple[float, float, StartupStage]] = []
    cursor_ms = 0.0
    for stage in stages:
        stage_windows.append((cursor_ms, cursor_ms + stage.duration_ms, stage))
        cursor_ms += stage.duration_ms

    rpm_windows = [
        (start_ms, end_ms, stage)
        for start_ms, end_ms, stage in stage_windows
        if stage.end_rpm > stage.start_rpm + 1e-6
    ]
    ramp_start_ms = rpm_windows[0][0] if rpm_windows else 0.0
    ramp_end_ms = rpm_windows[-1][1] if rpm_windows else 0.0
    ramp = (
        simulate_spin_ramp(
            config,
            rpm_windows[0][2].start_rpm,
            rpm_windows[-1][2].end_rpm,
            ramp_end_ms - ramp_start_ms,
            trace_step_ms,
        )
        if rpm_windows
        else []
    )

    self_test_start_ms, _self_test_end_ms, self_test_stage = stage_bounds(stage_windows, name="self_test")
    head_load_start_ms, _head_load_end_ms, head_load_stage = stage_bounds(stage_windows, head_load=True)
    servo_lock_start_ms, _servo_lock_end_ms, servo_lock_stage = stage_bounds(stage_windows, name="servo_lock")

    self_test_threshold_rpm = (
        startup_rpm_threshold(config, "self_test", start_rpm=self_test_stage.start_rpm, end_rpm=self_test_stage.end_rpm)
        if self_test_stage is not None
        else None
    )
    head_load_threshold_rpm = (
        startup_rpm_threshold(config, "head_load", start_rpm=head_load_stage.start_rpm, end_rpm=head_load_stage.end_rpm)
        if head_load_stage is not None
        else None
    )
    servo_lock_threshold_rpm = (
        startup_rpm_threshold(config, "servo_lock", start_rpm=servo_lock_stage.start_rpm, end_rpm=servo_lock_stage.end_rpm)
        if servo_lock_stage is not None
        else None
    )

    pre_ready_stage_name = {
        "power_on": "spinup",
        "standby": "spinup",
        "low_rpm_idle": "rpm_recover",
        "resume": "rpm_recover",
        "unloaded_idle": "head_load",
    }.get(origin, "spinup")

    self_test_trigger_ms: float | None = None
    head_load_trigger_ms: float | None = 0.0 if initial_heads_loaded else None
    servo_lock_trigger_ms: float | None = None
    next_seek_ms = math.inf
    next_cal_ms = math.inf
    trace_times = build_trace_times(total_ms, trace_step_ms, stage_windows)
    trace: list[StartupTracePoint] = []

    for elapsed_ms in trace_times:
        nominal_stage_name = stages[-1].name
        stage_end_rpm = stages[-1].end_rpm
        for _start_ms, end_ms, stage in stage_windows:
            if elapsed_ms <= end_ms + 1e-9:
                nominal_stage_name = stage.name
                stage_end_rpm = stage.end_rpm
                break

        if ramp and elapsed_ms < ramp_start_ms:
            rpm = float(rpm_windows[0][2].start_rpm)
        elif ramp and elapsed_ms <= ramp_end_ms:
            ramp_progress = 0.0
            if ramp_end_ms > ramp_start_ms:
                ramp_progress = max(0.0, min((elapsed_ms - ramp_start_ms) / (ramp_end_ms - ramp_start_ms), 1.0))
            ramp_index = min(round(ramp_progress * (len(ramp) - 1)), len(ramp) - 1)
            rpm = ramp[ramp_index]
        else:
            rpm = float(stage_end_rpm)

        if (
            self_test_trigger_ms is None
            and self_test_start_ms is not None
            and self_test_threshold_rpm is not None
            and elapsed_ms >= self_test_start_ms
            and rpm >= self_test_threshold_rpm
        ):
            self_test_trigger_ms = elapsed_ms
            next_cal_ms = elapsed_ms + max(35.0, trace_step_ms * 0.5)

        head_load_event = False
        if (
            head_load_trigger_ms is None
            and head_load_start_ms is not None
            and head_load_threshold_rpm is not None
            and elapsed_ms >= head_load_start_ms
            and rpm >= head_load_threshold_rpm
        ):
            head_load_trigger_ms = elapsed_ms
            head_load_event = True
            next_seek_ms = elapsed_ms + max(60.0, trace_step_ms)

        heads_loaded = initial_heads_loaded or head_load_trigger_ms is not None
        if (
            servo_lock_trigger_ms is None
            and heads_loaded
            and servo_lock_start_ms is not None
            and servo_lock_threshold_rpm is not None
            and elapsed_ms >= servo_lock_start_ms
            and rpm >= servo_lock_threshold_rpm
        ):
            servo_lock_trigger_ms = elapsed_ms

        self_test_active = self_test_trigger_ms is not None
        servo_locked = servo_lock_trigger_ms is not None

        if self_test_active:
            if head_load_trigger_ms is None:
                stage_name = "self_test"
            elif not servo_locked:
                stage_name = "head_load"
            elif elapsed_ms < total_ms:
                stage_name = "servo_lock"
            else:
                stage_name = nominal_stage_name
        elif nominal_stage_name in {"self_test", "head_load", "servo_lock"}:
            stage_name = pre_ready_stage_name
        else:
            stage_name = nominal_stage_name

        seek_distance = 0.0
        is_calibration = False
        activity_start_ms = self_test_trigger_ms if self_test_trigger_ms is not None else head_load_trigger_ms
        if activity_start_ms is not None and activity_start_ms <= elapsed_ms < total_ms:
            progress = 0.0
            if total_ms > activity_start_ms:
                progress = max(0.0, min((elapsed_ms - activity_start_ms) / (total_ms - activity_start_ms), 1.0))
            if heads_loaded and elapsed_ms + 1e-9 >= next_seek_ms:
                base_span = config.total_cylinders * (0.18 * (1.0 - progress) ** 1.55 + 0.0025)
                seek_distance = max(12.0, base_span)
                if origin in {"unloaded_idle", "resume"}:
                    seek_distance *= 0.45
                elif origin == "low_rpm_idle":
                    seek_distance *= 0.65
                if servo_locked:
                    seek_distance *= 0.45
                next_seek_ms += (120.0 + 420.0 * progress**1.2) * (1.45 if servo_locked else 1.0)
            if self_test_active and elapsed_ms + 1e-9 >= next_cal_ms:
                is_calibration = True
                next_cal_ms += (80.0 + 260.0 * progress**1.3) * (1.35 if servo_locked else 1.0)

        if head_load_event and seek_distance <= 0.0:
            seek_distance = 28.0

        trace.append(
            StartupTracePoint(
                time_ms=elapsed_ms,
                stage=stage_name,
                rpm=rpm,
                heads_loaded=heads_loaded,
                is_spinup=ramp_start_ms <= elapsed_ms <= ramp_end_ms if ramp else False,
                self_test_active=self_test_active,
                servo_locked=servo_locked,
                is_calibration=is_calibration,
                seek_distance=seek_distance,
                head_load_event=head_load_event,
            )
        )

    if trace:
        last = trace[-1]
        trace[-1] = StartupTracePoint(
            time_ms=last.time_ms,
            stage=last.stage,
            rpm=float(stages[-1].end_rpm),
            heads_loaded=last.heads_loaded or any(stage.head_load for stage in stages),
            is_spinup=False,
            self_test_active=last.self_test_active,
            servo_locked=True if servo_lock_stage is not None else last.servo_locked,
            is_calibration=False,
            seek_distance=0.0,
            head_load_event=last.head_load_event,
            park_event=False,
        )
    return trace


def build_spindown_trace_from_stages(
    config: HDDCoreConfig,
    stages: list[StartupStage],
    *,
    step_ms: float | None = None,
    initial_heads_loaded: bool = True,
) -> list[StartupTracePoint]:
    if not stages:
        return []

    total_ms = sum(stage.duration_ms for stage in stages)
    base_step_ms = startup_trace_step_ms(total_ms)
    trace_step_ms = base_step_ms if step_ms is None else min(float(step_ms), base_step_ms)
    stage_windows: list[tuple[float, float, StartupStage]] = []
    cursor_ms = 0.0
    for stage in stages:
        stage_windows.append((cursor_ms, cursor_ms + stage.duration_ms, stage))
        cursor_ms += stage.duration_ms

    stage_decays: dict[int, list[float]] = {}
    for stage_index, (_start_ms, _end_ms, stage) in enumerate(stage_windows):
        if stage.end_rpm + 1e-6 < stage.start_rpm:
            stage_decays[stage_index] = simulate_spin_decay(
                config,
                stage.start_rpm,
                stage.end_rpm,
                stage.duration_ms,
                trace_step_ms,
            )

    unload_start_ms = next((start_ms for start_ms, _, stage in stage_windows if stage.head_unload), None)
    park_emitted = False
    trace_times = build_trace_times(total_ms, trace_step_ms, stage_windows)
    trace: list[StartupTracePoint] = []

    for elapsed_ms in trace_times:
        stage_name = stages[-1].name
        stage_end_rpm = stages[-1].end_rpm
        stage_start_ms = total_ms
        stage_end_ms = total_ms
        stage_index = len(stage_windows) - 1
        for candidate_index, (_start_ms, end_ms, stage) in enumerate(stage_windows):
            if elapsed_ms <= end_ms + 1e-9:
                stage_name = stage.name
                stage_end_rpm = stage.end_rpm
                stage_start_ms = _start_ms
                stage_end_ms = end_ms
                stage_index = candidate_index
                break

        if stage_index in stage_decays:
            decay = stage_decays[stage_index]
            stage_progress = 0.0
            if stage_end_ms > stage_start_ms:
                stage_progress = max(0.0, min((elapsed_ms - stage_start_ms) / (stage_end_ms - stage_start_ms), 1.0))
            decay_index = min(round(stage_progress * (len(decay) - 1)), len(decay) - 1)
            rpm = decay[decay_index]
        else:
            rpm = float(stage_end_rpm)

        heads_loaded = initial_heads_loaded
        park_event = False
        if unload_start_ms is not None and elapsed_ms >= unload_start_ms:
            heads_loaded = False
            if not park_emitted:
                park_event = True
                park_emitted = True

        trace.append(
            StartupTracePoint(
                time_ms=elapsed_ms,
                stage=stage_name,
                rpm=rpm,
                heads_loaded=heads_loaded,
                is_spinup=False,
                self_test_active=False,
                servo_locked=False,
                is_calibration=False,
                seek_distance=0.0,
                head_load_event=False,
                park_event=park_event,
            )
        )

    if trace:
        last = trace[-1]
        trace[-1] = StartupTracePoint(
            time_ms=last.time_ms,
            stage=last.stage,
            rpm=float(stages[-1].end_rpm),
            heads_loaded=False,
            is_spinup=False,
            self_test_active=False,
            servo_locked=False,
            is_calibration=False,
            seek_distance=0.0,
            head_load_event=False,
            park_event=last.park_event,
        )
    return trace


def build_startup_sequence(config: HDDCoreConfig, origin: str) -> list[StartupStage]:
    if origin == "power_on":
        total_ms = config.power_on_to_ready_ms
        electronics_ms, spindle_unlock_ms, spinup_ms, self_test_ms, head_load_ms, servo_lock_ms = allocate_stage_durations(
            total_ms,
            weights=[0.04, 0.03, 0.68, 0.13, 0.04, 0.08],
            minimums=[120.0, 90.0, 300.0, 250.0, 100.0, 140.0],
        )
        unlock_rpm = max(config.target_rpm * 0.08, 120.0)
        spinup_end_rpm = config.target_rpm * 0.93
        self_test_end_rpm = config.target_rpm * 0.975
        head_load_end_rpm = config.target_rpm * 0.993
        return [
            StartupStage("electronics_init", electronics_ms, 0.0, 0.0, calibration_pulses=0),
            StartupStage("spindle_unlock", spindle_unlock_ms, 0.0, unlock_rpm, calibration_pulses=0),
            StartupStage("spinup", spinup_ms, unlock_rpm, spinup_end_rpm, calibration_pulses=0),
            StartupStage("self_test", self_test_ms, spinup_end_rpm, self_test_end_rpm, calibration_pulses=2),
            StartupStage(
                "head_load",
                head_load_ms,
                self_test_end_rpm,
                head_load_end_rpm,
                calibration_pulses=0,
                head_load=True,
            ),
            StartupStage(
                "servo_lock",
                servo_lock_ms,
                head_load_end_rpm,
                config.target_rpm,
                calibration_pulses=2,
            ),
        ]
    if origin == "standby":
        total_ms = config.standby_to_ready_ms
        spinup_ms, head_load_ms, servo_lock_ms = allocate_stage_durations(
            total_ms,
            weights=[0.84, 0.05, 0.11],
            minimums=[250.0, 90.0, 120.0],
        )
        spinup_end_rpm = config.target_rpm * 0.965
        head_load_end_rpm = config.target_rpm * 0.993
        return [
            StartupStage("spinup", spinup_ms, 0.0, spinup_end_rpm, calibration_pulses=0),
            StartupStage(
                "head_load",
                head_load_ms,
                spinup_end_rpm,
                head_load_end_rpm,
                calibration_pulses=0,
                head_load=True,
            ),
            StartupStage(
                "servo_lock",
                servo_lock_ms,
                head_load_end_rpm,
                config.target_rpm,
                calibration_pulses=1,
            ),
        ]
    if origin == "low_rpm_idle":
        total_ms = config.low_rpm_to_ready_ms
        rpm_recover_ms, head_load_ms, servo_lock_ms = allocate_stage_durations(
            total_ms,
            weights=[0.72, 0.12, 0.16],
            minimums=[250.0, 120.0, 120.0],
        )
        rpm_recover_end_rpm = config.target_rpm * 0.985
        head_load_end_rpm = config.target_rpm * 0.997
        return [
            StartupStage("rpm_recover", rpm_recover_ms, config.low_rpm_rpm, rpm_recover_end_rpm, calibration_pulses=0),
            StartupStage("head_load", head_load_ms, rpm_recover_end_rpm, head_load_end_rpm, head_load=True),
            StartupStage("servo_lock", servo_lock_ms, head_load_end_rpm, config.target_rpm, calibration_pulses=1),
        ]
    if origin == "unloaded_idle":
        total_ms = config.unload_to_ready_ms
        head_load_ms, servo_lock_ms = allocate_stage_durations(
            total_ms,
            weights=[0.58, 0.42],
            minimums=[120.0, 80.0],
        )
        return [
            StartupStage("head_load", head_load_ms, config.target_rpm, config.target_rpm, calibration_pulses=0, head_load=True),
            StartupStage("servo_lock", servo_lock_ms, config.target_rpm, config.target_rpm, calibration_pulses=1),
        ]
    return []


def estimated_lba(config: HDDCoreConfig, mechanical: MechanicalState) -> int:
    zone = zone_for_cyl(config, mechanical.current_cyl)
    blocks_per_cyl = config.num_heads * zone.blocks_per_track
    cyl_offset = mechanical.current_cyl - zone.start_cyl
    return zone.start_lba + cyl_offset * blocks_per_cyl + mechanical.current_head * zone.blocks_per_track + mechanical.current_sector


def zone_for_lba(config: HDDCoreConfig, lba: int) -> Zone:
    clamped_lba = min(max(lba, 0), config.addressable_blocks - 1)
    for zone in config.zones:
        if clamped_lba <= zone.end_lba:
            return zone
    return config.zones[-1]


def zone_for_cyl(config: HDDCoreConfig, cyl: int) -> Zone:
    for zone in config.zones:
        if zone.start_cyl <= cyl <= zone.end_cyl:
            return zone
    return config.zones[-1]


def lba_to_chs(config: HDDCoreConfig, lba: int) -> tuple[int, int, int, Zone]:
    zone = zone_for_lba(config, lba)
    relative_lba = max(0, min(lba, zone.end_lba) - zone.start_lba)
    blocks_per_cyl = config.num_heads * zone.blocks_per_track
    cyl = zone.start_cyl + (relative_lba // blocks_per_cyl)
    remaining = relative_lba % blocks_per_cyl
    head = remaining // zone.blocks_per_track
    sector = remaining % zone.blocks_per_track
    return cyl, head, sector, zone


def prune_cache(cache_state: CacheState, now: float) -> CacheState:
    return replace(cache_state, spans=tuple(span for span in cache_state.spans if span.expires_at >= now))


def cache_overlap_blocks(cache_state: CacheState, lba: int, blocks: int) -> int:
    requested_end = lba + blocks - 1
    overlap = 0
    for span in cache_state.spans:
        if span.start_lba <= lba <= span.end_lba:
            overlap = max(overlap, min(requested_end, span.end_lba) - lba + 1)
    return overlap


def remember_read(config: HDDCoreConfig, cache_state: CacheState, lba: int, blocks: int, now: float) -> CacheState:
    if lba <= cache_state.last_read_end_lba + 1:
        cache_blocks = max(blocks * 4, config.read_ahead_blocks)
    else:
        cache_blocks = max(blocks * 2, config.read_ahead_blocks // 2)
    span = CacheSpan(lba, lba + cache_blocks - 1, now + 2.0)
    return replace(
        prune_cache(cache_state, now),
        spans=(*prune_cache(cache_state, now).spans, span),
        last_read_end_lba=lba + blocks - 1,
    )


def remember_cached_write(config: HDDCoreConfig, cache_state: CacheState, lba: int, size_bytes: int, now: float) -> CacheState:
    blocks = max(1, math.ceil(size_bytes / config.block_bytes))
    span = CacheSpan(lba, lba + blocks - 1, now + 1.0)
    pruned = prune_cache(cache_state, now)
    return replace(pruned, spans=(*pruned.spans, span))


def calculate_position_latency(
    config: HDDCoreConfig,
    mechanical: MechanicalState,
    target_lba: int,
    block_count: int,
) -> tuple[float, int, int, int, int, Zone]:
    target_cyl, target_head, target_sector, zone = lba_to_chs(config, target_lba)
    distance = abs(target_cyl - mechanical.current_cyl)

    if distance == 0:
        seek_ms = 0.0
    else:
        seek_ms = (
            config.track_to_track_ms + config.seek_curve_b * math.sqrt(distance) + config.settle_ms
        ) * config.aam_seek_penalty

    head_switch_ms = config.head_switch_ms if (distance == 0 and target_head != mechanical.current_head) else 0.0
    skew_blocks = 0
    if distance > 0:
        skew_blocks += config.cylinder_skew_blocks
    elif target_head != mechanical.current_head:
        skew_blocks += config.track_skew_blocks

    rotational_blocks = ((seek_ms + head_switch_ms) / config.ms_per_rotation) * zone.blocks_per_track
    current_sector_after_seek = (mechanical.current_sector + rotational_blocks + skew_blocks) % zone.blocks_per_track
    sector_delta = (target_sector - current_sector_after_seek) % zone.blocks_per_track
    rotational_ms = (sector_delta / zone.blocks_per_track) * config.ms_per_rotation

    transfer_ms = transfer_ms_for_span(config, target_lba, block_count)
    return seek_ms + head_switch_ms + rotational_ms + transfer_ms, target_cyl, target_head, target_sector, distance, zone


def transfer_ms_for_span(config: HDDCoreConfig, start_lba: int, block_count: int) -> float:
    remaining_blocks = max(0, block_count)
    lba = max(0, start_lba)
    transfer_ms = 0.0
    while remaining_blocks > 0:
        zone = zone_for_lba(config, lba)
        zone_blocks = min(remaining_blocks, zone.end_lba - lba + 1)
        transfer_ms += ((zone_blocks * config.block_bytes) / (1024 * 1024)) / zone.transfer_rate_mbps * 1000.0
        remaining_blocks -= zone_blocks
        lba += zone_blocks
    return transfer_ms


def command_overhead_for(config: HDDCoreConfig, op_kind: str, queue_depth: int) -> float:
    base_ms = config.command_overhead_by_kind.get(op_kind, config.command_overhead_ms)
    queue_scale = 1.0
    if op_kind in {"journal", "flush"}:
        queue_scale = 1.25
    elif op_kind in {"metadata", "background"}:
        queue_scale = 0.85
    queue_penalty_ms = max(queue_depth - 1, 0) * config.queue_depth_penalty_ms * queue_scale
    return base_ms + queue_penalty_ms


def allocate_stage_durations(total_ms: float, weights: list[float], minimums: list[float]) -> list[float]:
    total_ms = max(float(total_ms), 0.0)
    if not weights:
        return []

    weight_sum = sum(max(weight, 0.0) for weight in weights)
    safe_weights = [max(weight, 0.0) for weight in weights]
    safe_mins = [max(minimum, 0.0) for minimum in minimums]
    minimum_sum = sum(safe_mins)

    if total_ms <= 0.0:
        return [0.0] * len(weights)

    if minimum_sum > 0.0 and total_ms <= minimum_sum:
        scale = total_ms / minimum_sum
        durations = [minimum * scale for minimum in safe_mins]
    else:
        residual = max(total_ms - minimum_sum, 0.0)
        if weight_sum <= 0.0:
            extra_share = residual / len(safe_weights)
            durations = [minimum + extra_share for minimum in safe_mins]
        else:
            durations = [
                minimum + residual * (weight / weight_sum)
                for weight, minimum in zip(safe_weights, safe_mins, strict=True)
            ]

    durations[-1] = max(0.0, total_ms - sum(durations[:-1]))
    return durations


def background_decision(
    config: HDDCoreConfig,
    mechanical: MechanicalState,
    *,
    transition_active: bool,
    now: float,
    last_idle_calibration_time: float,
    unload_after_s: float,
    low_rpm_after_s: float,
    standby_after_s: float,
) -> BackgroundDecision:
    idle_s = now - mechanical.last_access_time
    if idle_s >= unload_after_s and mechanical.power_state == "active" and not transition_active:
        return BackgroundDecision(
            rpm=mechanical.current_rpm,
            park=True,
            next_power_state="unloaded_idle",
            next_heads_loaded=False,
            load_unload_delta=1,
        )
    if idle_s >= low_rpm_after_s and mechanical.power_state == "unloaded_idle" and not transition_active:
        return BackgroundDecision(rpm=mechanical.current_rpm, should_low_rpm=True)
    if (
        idle_s >= 8.0
        and mechanical.power_state == "active"
        and not transition_active
        and now - last_idle_calibration_time >= 6.0
    ):
        return BackgroundDecision(
            rpm=mechanical.current_rpm,
            calibrate=True,
            next_idle_calibration_time=now,
        )
    if idle_s >= standby_after_s and mechanical.power_state in {"low_rpm_idle", "unloaded_idle", "active"} and not transition_active:
        return BackgroundDecision(rpm=mechanical.current_rpm, should_spindown=True)
    return BackgroundDecision(rpm=mechanical.current_rpm)
