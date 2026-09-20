from __future__ import annotations

from ..storage_events import ScheduledStorageEvent, StorageEvent


def expand_workload_event(event: StorageEvent, sample_rate: int) -> list[ScheduledStorageEvent]:
    """Keep physical telemetry one-to-one with audio commands.

    The storage scheduler already emits each physical extent access. Inventing
    extra alternating seeks from directory size or fragmentation double-counts
    that work and can leave ghost seeks after the real transfer ends.
    Sequential track-boundary timing belongs to the RPM-driven audio plant.
    The public adapter name is retained for event consumers.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return [(event, 0)]
