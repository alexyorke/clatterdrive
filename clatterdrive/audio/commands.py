from __future__ import annotations

import math
from dataclasses import dataclass

from ..storage_events import StorageEvent


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


@dataclass(frozen=True)
class AudioCommand:
    emitted_at: float
    target_rpm: float
    power_state: str
    servo_mode: str
    track_delta: float
    transfer_activity: float
    motion_duration_s: float
    settle_duration_s: float
    queue_depth: int
    op_kind: str
    heads_loaded: bool
    is_spinup: bool
    is_flush: bool
    is_sequential: bool
    maintenance: bool
    retry: bool
    size_bytes: int
    block_count: int
    extent_count: int
    transfer_duration_s: float
    directory_entry_count: int
    fragmentation_score: int


ScheduledCommand = tuple[AudioCommand, int]


def _derive_power_state(event: StorageEvent) -> str:
    if event.is_spinup:
        return "starting"
    if event.power_state:
        return event.power_state
    target_rpm = event.target_rpm if event.target_rpm is not None else event.rpm
    if target_rpm <= 1.0 and event.rpm <= 1.0:
        return "standby"
    if event.rpm < target_rpm * 0.5:
        return "starting"
    return "active"


def _derive_servo_mode(event: StorageEvent) -> str:
    if event.servo_mode:
        return event.servo_mode
    if event.impulse == "park":
        return "park"
    if event.impulse == "calibration":
        return "calibration"
    if event.impulse == "seek":
        return "seek"
    if event.is_sequential:
        return "track"
    return "idle"


def _derive_track_delta(event: StorageEvent) -> float:
    if event.track_delta:
        return _clamp(float(event.track_delta), -1.0, 1.0)
    if event.seek_distance:
        return _clamp(float(event.seek_distance) / 1200.0, -1.0, 1.0)
    return 0.0


def _derive_transfer_activity(event: StorageEvent) -> float:
    base = (
        float(event.transfer_activity)
        if event.transfer_activity
        else {
            "metadata": 0.38,
            "journal": 0.58,
            "data": 0.74,
            "writeback": 0.86,
            "flush": 1.05,
            "background": 0.44,
        }.get(event.op_kind, 0.52)
    )
    block_count = max(0, int(event.block_count))
    if event.op_kind in {"data", "writeback"} and block_count > 1:
        base *= 1.0 + min(0.35, 0.045 * math_log2(block_count + 1))
    if event.op_kind == "metadata" and event.directory_entry_count > 0:
        base *= 1.0 + min(0.45, 0.040 * math_log2(event.directory_entry_count + 1))
    if event.fragmentation_score > 1:
        base *= 1.0 + min(0.30, 0.035 * (event.fragmentation_score - 1))
    if event.is_sequential:
        base *= 0.82
    if event.is_flush:
        base *= 1.10
    base *= 1.0 + 0.06 * max(event.queue_depth - 1, 0)
    return _clamp(base, 0.0, 3.0)


def math_log2(value: float) -> float:
    if value <= 0.0:
        return 0.0
    return math.log2(value)


def _derive_transfer_duration_s(event: StorageEvent) -> float:
    duration = max(0.0, float(event.transfer_ms) / 1000.0)
    if duration <= 0.0 and event.block_count > 0 and event.op_kind in {"data", "writeback", "metadata"}:
        duration = max(0.002, min(0.350, event.block_count * 0.00010))
    if event.directory_entry_count > 0:
        duration = max(duration, min(0.250, 0.003 + event.directory_entry_count * 0.00005))
    return min(duration, 0.750)


def command_from_event(event: StorageEvent) -> AudioCommand:
    target_rpm = float(event.target_rpm if event.target_rpm is not None else event.rpm)
    power_state = _derive_power_state(event)
    servo_mode = _derive_servo_mode(event)
    motion_duration_s = max(
        float(event.motion_duration_ms if event.motion_duration_ms > 0.0 else event.actuator_duration_ms) / 1000.0,
        0.0,
    )
    settle_duration_s = max(
        float(event.settle_duration_ms if event.settle_duration_ms > 0.0 else event.actuator_settle_ms) / 1000.0,
        0.0,
    )
    heads_loaded = (
        event.heads_loaded
        if event.heads_loaded is not None
        else power_state == "active" and servo_mode != "park"
    )
    op_kind = event.op_kind or "data"
    return AudioCommand(
        emitted_at=float(event.emitted_at),
        target_rpm=target_rpm,
        power_state=power_state,
        servo_mode=servo_mode,
        track_delta=_derive_track_delta(event),
        transfer_activity=_derive_transfer_activity(event),
        motion_duration_s=motion_duration_s,
        settle_duration_s=settle_duration_s,
        queue_depth=max(1, int(event.queue_depth)),
        op_kind=op_kind,
        heads_loaded=bool(heads_loaded),
        is_spinup=bool(event.is_spinup),
        is_flush=bool(event.is_flush),
        is_sequential=bool(event.is_sequential),
        maintenance=op_kind in {"background", "metadata"},
        retry=op_kind in {"retry", "recovery"},
        size_bytes=max(0, int(event.size_bytes)),
        block_count=max(0, int(event.block_count)),
        extent_count=max(0, int(event.extent_count)),
        transfer_duration_s=_derive_transfer_duration_s(event),
        directory_entry_count=max(0, int(event.directory_entry_count)),
        fragmentation_score=max(0, int(event.fragmentation_score)),
    )
