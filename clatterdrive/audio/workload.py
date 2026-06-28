from __future__ import annotations

import math
from dataclasses import replace

from ..storage_events import ScheduledStorageEvent, StorageEvent


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _log2(value: float) -> float:
    return math.log2(max(value, 1.0))


def _has_workload_shape(event: StorageEvent) -> bool:
    return any(
        (
            event.size_bytes > 0,
            event.block_count > 0,
            event.extent_count > 0,
            event.transfer_ms > 0.0,
            event.directory_entry_count > 0,
            event.fragmentation_score > 0,
        )
    )


def _variant(seed: int, index: int) -> float:
    value = (seed + index * 0x9E3779B1) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return ((value & 0xFFFF) / 32767.5) - 1.0


def _event_seed(event: StorageEvent) -> int:
    kind_code = {
        "metadata": 1,
        "journal": 2,
        "data": 3,
        "writeback": 4,
        "flush": 5,
        "background": 6,
    }.get(event.op_kind, 0)
    return (
        int(event.size_bytes // 4096)
        + int(event.block_count) * 17
        + int(event.directory_entry_count) * 31
        + int(event.fragmentation_score) * 131
        + kind_code * 1009
    ) & 0xFFFFFFFF


def _frame_delta(delta_s: float, sample_rate: int) -> int:
    return max(0, round(max(0.0, delta_s) * sample_rate))


def _base_activity(event: StorageEvent) -> float:
    if event.transfer_activity > 0.0:
        return float(event.transfer_activity)
    return {
        "metadata": 0.38,
        "journal": 0.58,
        "data": 0.74,
        "writeback": 0.86,
        "flush": 1.05,
        "background": 0.44,
    }.get(event.op_kind, 0.52)


def _workload_span_s(event: StorageEvent, count: int) -> float:
    transfer_s = max(0.0, event.transfer_ms / 1000.0)
    directory_s = 0.00006 * max(0, event.directory_entry_count)
    metadata_s = 0.0046 * max(1, count)
    return _clamp(max(transfer_s, metadata_s + directory_s), 0.010, 0.180)


def _metadata_burst_count(event: StorageEvent) -> int:
    blocks = max(1, event.block_count)
    directory_entries = max(0, event.directory_entry_count)
    fragmentation = max(0, event.fragmentation_score)
    count = 1
    count += min(3, int(_log2(blocks + 1)))
    count += min(3, directory_entries // 48)
    count += min(3, max(0, fragmentation - 1))
    if event.op_kind == "journal":
        count += 1
    if event.is_flush or event.op_kind == "flush":
        count += 1
    return min(10, max(1, count))


def _metadata_burstlets(event: StorageEvent, sample_rate: int) -> list[ScheduledStorageEvent]:
    count = _metadata_burst_count(event)
    if count <= 1:
        return [(event, 0)]

    seed = _event_seed(event)
    span_s = _workload_span_s(event, count)
    activity = _base_activity(event)
    block_count = max(1, event.block_count // count) if event.block_count else 1
    size_bytes = max(block_count * 4096, event.size_bytes // count) if event.size_bytes else 0
    result: list[ScheduledStorageEvent] = []

    for index in range(count):
        progress = index / max(count - 1, 1)
        sign = -1.0 if index % 2 else 1.0
        stroke = abs(event.track_delta) if abs(event.track_delta) > 0.0 else 0.075 + 0.017 * (index % 4)
        stroke += 0.014 * abs(_variant(seed, index))
        op_kind = event.op_kind
        if event.op_kind == "journal" and index > 0:
            op_kind = "metadata"
        burst = replace(
            event,
            emitted_at=event.emitted_at + progress * span_s,
            queue_depth=max(event.queue_depth, min(8, event.queue_depth + index // 3)),
            op_kind=op_kind,
            servo_mode="seek" if event.servo_mode in {None, "idle", "track"} else event.servo_mode,
            track_delta=_clamp(sign * stroke, -0.30, 0.30),
            transfer_activity=_clamp(activity * (0.56 + 0.035 * index), 0.0, 2.1),
            motion_duration_ms=max(event.motion_duration_ms, 2.1 + 0.24 * (index % 4)),
            settle_duration_ms=max(event.settle_duration_ms, 1.15 + 0.16 * (index % 3)),
            size_bytes=max(0, int(size_bytes)),
            block_count=max(1, int(block_count)),
        )
        result.append((burst, _frame_delta(progress * span_s, sample_rate)))

    return result


def _transfer_tick_count(event: StorageEvent) -> int:
    blocks = max(0, event.block_count)
    fragmentation = max(0, event.fragmentation_score)
    count = 0
    if blocks >= 64:
        count += min(8, max(1, blocks // 192))
    if fragmentation > 1:
        count += min(8, fragmentation - 1)
    if event.is_flush or event.op_kind == "flush":
        count += 2
    return min(14, count)


def _transfer_burstlets(event: StorageEvent, sample_rate: int) -> list[ScheduledStorageEvent]:
    count = _transfer_tick_count(event)
    if count <= 0:
        return [(event, 0)]

    seed = _event_seed(event)
    transfer_s = _workload_span_s(event, count + 1)
    activity = _base_activity(event)
    result: list[ScheduledStorageEvent] = [(event, 0)]

    for index in range(count):
        progress = (index + 1) / (count + 1)
        fragmented = event.fragmentation_score > 1 and index >= count - min(count, event.fragmentation_score - 1)
        stroke = 0.010 if event.is_sequential and not fragmented else 0.045 + 0.030 * abs(_variant(seed, index))
        tick = replace(
            event,
            emitted_at=event.emitted_at + progress * transfer_s,
            servo_mode="seek" if fragmented else "track",
            track_delta=_clamp(stroke * (1.0 if index % 2 == 0 else -1.0), -0.35, 0.35),
            transfer_activity=_clamp(activity * (0.58 if event.is_sequential and not fragmented else 0.78), 0.0, 3.0),
            motion_duration_ms=max(event.motion_duration_ms, 0.9 if event.is_sequential and not fragmented else 1.8),
            settle_duration_ms=max(event.settle_duration_ms, 0.45 if event.is_sequential and not fragmented else 1.1),
        )
        result.append((tick, _frame_delta(progress * transfer_s, sample_rate)))

    return result


def expand_workload_event(event: StorageEvent, sample_rate: int) -> list[ScheduledStorageEvent]:
    """Expand physical workload events into proportional audio control burstlets.

    Hand-authored demo events intentionally stay one-to-one so checked-in golden
    samples remain stable. Runtime storage events carry size/extent/directory
    metadata and get expanded into the small seek/journal clusters that make
    file copies feel like real disk work.
    """

    if not _has_workload_shape(event):
        return [(event, 0)]
    if event.op_kind in {"metadata", "journal", "flush"}:
        return _metadata_burstlets(event, sample_rate)
    if event.op_kind in {"data", "writeback"}:
        return _transfer_burstlets(event, sample_rate)
    return [(event, 0)]
