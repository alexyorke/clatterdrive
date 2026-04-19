from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from clatterdrive.scheduler import OSScheduler
from clatterdrive.scheduler_core import (
    build_request,
    can_submit,
    completion_ids,
    merge_request,
    outstanding_after_completion,
    outstanding_after_submit,
    pick_next_request,
)


def test_scheduler_core_merges_adjacent_requests_of_same_class() -> None:
    first, sequence = build_request(
        sequence=0,
        lba=100,
        size=4096,
        is_write=False,
        op_kind="data",
        sync=False,
        arrival_time=10.0,
        read_deadline_s=0.025,
        write_deadline_s=0.150,
    )
    second, _ = build_request(
        sequence=sequence,
        lba=101,
        size=4096,
        is_write=False,
        op_kind="data",
        sync=False,
        arrival_time=10.01,
        read_deadline_s=0.025,
        write_deadline_s=0.150,
    )

    queue, merged_into = merge_request((first,), second, block_bytes=4096)

    assert merged_into == first.id
    assert len(queue) == 1
    assert queue[0].size == 8192
    assert queue[0].deadline == first.deadline


def test_scheduler_core_picks_expired_deadline_before_look_order() -> None:
    left, sequence = build_request(
        sequence=0,
        lba=50,
        size=4096,
        is_write=False,
        op_kind="data",
        sync=False,
        arrival_time=5.0,
        read_deadline_s=0.025,
        write_deadline_s=0.150,
    )
    right, _ = build_request(
        sequence=sequence,
        lba=500,
        size=4096,
        is_write=False,
        op_kind="data",
        sync=False,
        arrival_time=5.0,
        read_deadline_s=0.025,
        write_deadline_s=0.150,
    )
    expired_right = type(right)(
        id=right.id,
        lba=right.lba,
        size=right.size,
        is_write=right.is_write,
        op_kind=right.op_kind,
        sync=right.sync,
        arrival_time=right.arrival_time,
        deadline=4.0,
    )

    queue, picked, direction = pick_next_request((left, expired_right), current_lba=0, direction=1, now=5.0)

    assert picked == expired_right
    assert direction == 1
    assert queue == (left,)


def test_scheduler_core_reverses_direction_when_no_forward_requests_exist() -> None:
    request, _ = build_request(
        sequence=0,
        lba=10,
        size=4096,
        is_write=False,
        op_kind="data",
        sync=False,
        arrival_time=1.0,
        read_deadline_s=0.025,
        write_deadline_s=0.150,
    )

    queue, picked, direction = pick_next_request((request,), current_lba=100, direction=1, now=1.0)

    assert picked == request
    assert direction == -1
    assert queue == ()


def test_scheduler_core_tracks_queue_depth_accounting_and_completion_ids() -> None:
    assert can_submit(0, 1) is True
    assert can_submit(1, 1) is False
    assert outstanding_after_submit(0) == 1
    assert outstanding_after_completion(3, 2) == 0
    assert completion_ids("req-1", ("req-2", "req-3")) == ("req-1", "req-2", "req-3")

def test_scheduler_propagates_model_failures() -> None:
    class FailingModel:
        block_bytes = 4096

        def get_estimated_lba(self) -> int:
            return 0

        def submit_physical_access(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

    scheduler = OSScheduler(FailingModel())
    try:
        request_id = scheduler.submit_bio(0, 4096, is_write=False)
        with pytest.raises(RuntimeError, match="boom"):
            scheduler.wait_for_completion(request_id)
    finally:
        scheduler.stop()

def test_scheduler_respects_queue_depth_limit() -> None:
    class SlowModel:
        block_bytes = 4096

        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def get_estimated_lba(self) -> int:
            return 0

        def submit_physical_access(self, *args: Any, **kwargs: Any) -> dict[str, float | bool | int]:
            self.started.set()
            self.release.wait(timeout=1.0)
            return {"total_ms": 0.0, "cache_hit": False, "cyl": 0, "head": 0}

    model = SlowModel()
    scheduler = OSScheduler(model, max_queue_depth=1)
    try:
        first_request = scheduler.submit_bio(0, 4096, is_write=False)
        assert model.started.wait(timeout=1.0)

        holder: dict[str, str] = {}
        finished_submit = threading.Event()

        def submit_second() -> None:
            holder["req"] = scheduler.submit_bio(1, 4096, is_write=False)
            finished_submit.set()

        submit_thread = threading.Thread(target=submit_second)
        submit_thread.start()
        time.sleep(0.05)
        assert not finished_submit.is_set()

        model.release.set()
        scheduler.wait_for_completion(first_request)
        submit_thread.join(timeout=1.0)
        assert finished_submit.is_set()
        scheduler.wait_for_completion(holder["req"])
    finally:
        scheduler.stop()
