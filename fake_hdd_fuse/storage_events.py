from __future__ import annotations

import json
import sys
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class StorageEvent:
    rpm: float
    emitted_at: float
    target_rpm: float | None = None
    queue_depth: int = 1
    op_kind: str = "data"
    is_sequential: bool = False
    is_flush: bool = False
    is_spinup: bool = False
    power_state: str | None = None
    heads_loaded: bool | None = None
    servo_mode: str | None = None
    track_delta: float = 0.0
    transfer_activity: float = 0.0
    motion_duration_ms: float = 0.0
    settle_duration_ms: float = 0.0

    # Legacy compatibility telemetry. The current synth should prefer the
    # command-domain fields above and treat these as optional fallbacks only.
    impulse: str | None = None
    seek_distance: float = 0.0
    actuator_duration_ms: float = 0.0
    actuator_force: float = 0.0
    actuator_settle_ms: float = 0.0
    motor_drive: float = 0.0
    windage_level: float = 0.0
    structure_borne_gain: float = 0.0
    servo_activity: float = 0.0


def storage_event_to_dict(event: StorageEvent) -> dict[str, float | int | bool | str | None]:
    return asdict(event)


ScheduledStorageEvent = tuple[StorageEvent, int]


@runtime_checkable
class StorageEventSink(Protocol):
    def publish_event(self, event: StorageEvent) -> None: ...


class NullStorageEventSink:
    def publish_event(self, event: StorageEvent) -> None:
        return None


class CompositeStorageEventSink:
    def __init__(self, sinks: Sequence[StorageEventSink]) -> None:
        self._sinks = tuple(sink for sink in sinks if not isinstance(sink, NullStorageEventSink))

    def publish_event(self, event: StorageEvent) -> None:
        for sink in self._sinks:
            sink.publish_event(event)


class StorageEventRecorder:
    def __init__(self, max_events: int = 20000) -> None:
        self._events: list[StorageEvent] = []
        self._lock = threading.Lock()
        self.max_events = max(1, max_events)
        self._dropped_events = 0

    def publish_event(self, event: StorageEvent) -> None:
        with self._lock:
            if len(self._events) >= self.max_events:
                self._events.pop(0)
                self._dropped_events += 1
            self._events.append(event)

    def snapshot(self) -> list[StorageEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._dropped_events = 0

    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_events

    def export_json(self, path: str) -> Path:
        payload = {
            "dropped_events": self.dropped_count(),
            "events": [storage_event_to_dict(event) for event in self.snapshot()],
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return output_path


class DebugStorageEventSink:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stderr

    def publish_event(self, event: StorageEvent) -> None:
        flags: list[str] = []
        if event.is_spinup:
            flags.append("spinup")
        if event.is_flush:
            flags.append("flush")
        if event.is_sequential:
            flags.append("seq")
        if event.servo_mode:
            flags.append(event.servo_mode)
        track = f" track={event.track_delta:.3f}" if event.track_delta else ""
        transfer = f" transfer={event.transfer_activity:.3f}" if event.transfer_activity else ""
        target = f" target={event.target_rpm:.1f}" if event.target_rpm is not None else ""
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        print(
            (
                f"AUDIO EVENT{flag_text}: rpm={event.rpm:.1f}{target} "
                f"q={event.queue_depth} kind={event.op_kind}{track}{transfer}"
            ),
            file=self.stream,
        )


class StorageEventBus:
    """Thread-safe queue for storage telemetry events."""

    def __init__(self, max_pending: int = 2048) -> None:
        self._events: deque[StorageEvent] = deque()
        self._lock = threading.Lock()
        self.max_pending = max(1, max_pending)
        self._dropped_events = 0

    def publish(self, event: StorageEvent) -> None:
        with self._lock:
            if len(self._events) >= self.max_pending:
                self._events.popleft()
                self._dropped_events += 1
            self._events.append(event)

    def drain(self) -> list[StorageEvent]:
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
