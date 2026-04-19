from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
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


ScheduledStorageEvent = tuple[StorageEvent, int]


@runtime_checkable
class StorageEventSink(Protocol):
    def publish_event(self, event: StorageEvent) -> None: ...


class NullStorageEventSink:
    def publish_event(self, event: StorageEvent) -> None:
        return None


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
