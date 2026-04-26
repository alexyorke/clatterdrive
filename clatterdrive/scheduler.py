import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .scheduler_core import (
    SchedulerRequest,
    build_request,
    can_submit,
    completion_ids,
    merge_request,
    outstanding_after_completion,
    outstanding_after_submit,
    pick_next_request,
)

@dataclass
class _PendingDispatch:
    request: SchedulerRequest
    event: threading.Event = field(default_factory=threading.Event)
    followers: list[str] = field(default_factory=list)


class OSScheduler:
    """
    Simulates a small blk-mq + mq-deadline style scheduler.

    Features:
    - request merging for adjacent requests of the same type
    - read-preferring deadline handling
    - LOOK-like dispatch order in the absence of an expired deadline
    - finite outstanding queue depth
    """

    def __init__(
        self,
        hdd_model: Any,
        max_queue_depth: int = 32,
        read_deadline_ms: float = 25,
        write_deadline_ms: float = 150,
    ) -> None:
        self.hdd_model = hdd_model
        self.max_queue_depth = max_queue_depth
        self.read_deadline_s = read_deadline_ms / 1000.0
        self.write_deadline_s = write_deadline_ms / 1000.0

        self.staging_queue: list[_PendingDispatch] = []
        self.results: dict[str, Any | BaseException] = {}
        self.events: dict[str, threading.Event] = {}
        self.direction = 1
        self.sequence = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.running = True
        self.outstanding = 0
        self.dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatch_thread.start()

    def stop(self) -> None:
        with self.condition:
            self.running = False
            for pending in self.staging_queue:
                failure = RuntimeError("scheduler stopped before request was dispatched")
                for request_id in completion_ids(pending.request.id, tuple(pending.followers)):
                    self.results[request_id] = failure
                self.outstanding = outstanding_after_completion(self.outstanding, len(pending.followers))
                pending.event.set()
            self.staging_queue.clear()
            self.condition.notify_all()
        self.dispatch_thread.join(timeout=2.0)

    def submit_bio(
        self,
        lba: int,
        size: int,
        is_write: bool,
        op_kind: str = "data",
        sync: bool = False,
        extent_count: int = 0,
        directory_entry_count: int = 0,
        fragmentation_score: int = 0,
    ) -> str:
        now = time.monotonic()
        with self.condition:
            while self.running and not can_submit(self.outstanding, self.max_queue_depth):
                self.condition.wait(timeout=0.05)
            if not self.running:
                raise RuntimeError("scheduler is stopped")
            request, self.sequence = build_request(
                sequence=self.sequence,
                lba=lba,
                size=size,
                is_write=is_write,
                op_kind=op_kind,
                sync=sync,
                arrival_time=now,
                read_deadline_s=self.read_deadline_s,
                write_deadline_s=self.write_deadline_s,
                extent_count=extent_count,
                directory_entry_count=directory_entry_count,
                fragmentation_score=fragmentation_score,
            )
            event = threading.Event()
            self.outstanding = outstanding_after_submit(self.outstanding)
            self.events[request.id] = event

            queue, merged_into = merge_request(
                tuple(pending.request for pending in self.staging_queue),
                request,
                block_bytes=self.hdd_model.block_bytes,
            )
            if merged_into is None:
                self.staging_queue.append(_PendingDispatch(request=request, event=event))
            else:
                self._sync_queue_requests(queue)
                pending = self._find_pending(merged_into)
                if pending is None:
                    msg = f"merged request {merged_into!r} missing from scheduler queue"
                    raise RuntimeError(msg)
                pending.followers.append(request.id)
                self.events[request.id] = pending.event
            if merged_into is None:
                self._sync_queue_requests(queue)
            self.condition.notify_all()
            return request.id

    def wait_for_completion(self, req_id: str) -> Any:
        event = self.events[req_id]
        event.wait()
        with self.lock:
            self.events.pop(req_id, None)
            completion = self.results.pop(req_id)
        if isinstance(completion, BaseException):
            raise completion
        return completion

    def _find_pending(self, request_id: str) -> _PendingDispatch | None:
        for pending in self.staging_queue:
            if pending.request.id == request_id:
                return pending
        return None

    def _sync_queue_requests(self, queue: tuple[SchedulerRequest, ...]) -> None:
        followers_by_id = {pending.request.id: list(pending.followers) for pending in self.staging_queue}
        events_by_id = {pending.request.id: pending.event for pending in self.staging_queue}
        self.staging_queue = [
            _PendingDispatch(
                request=request,
                event=events_by_id.get(request.id, threading.Event()),
                followers=followers_by_id.get(request.id, []),
            )
            for request in queue
        ]

    def _dispatch_loop(self) -> None:
        while True:
            with self.condition:
                while self.running and not self.staging_queue:
                    self.condition.wait(timeout=0.05)
                if not self.running and not self.staging_queue:
                    return
                _, request, self.direction = pick_next_request(
                    tuple(pending.request for pending in self.staging_queue),
                    current_lba=self.hdd_model.get_estimated_lba(),
                    direction=self.direction,
                    now=time.monotonic(),
                )
                request_id = None if request is None else request.id
                pending = None if request_id is None else self._find_pending(request_id)
                if pending is not None:
                    self.staging_queue = [item for item in self.staging_queue if item.request.id != request_id]
                queue_depth = len(self.staging_queue) + (1 if request else 0)

            if not request or pending is None:
                continue

            try:
                result = self.hdd_model.submit_physical_access(
                    request.lba,
                    request.size,
                    request.is_write,
                    op_kind=request.op_kind,
                    force_unit_access=request.sync,
                    queue_depth=min(queue_depth, self.max_queue_depth),
                    extent_count=request.extent_count,
                    directory_entry_count=request.directory_entry_count,
                    fragmentation_score=request.fragmentation_score,
                )
            except Exception as exc:
                result = exc

            with self.lock:
                for request_id in completion_ids(request.id, tuple(pending.followers)):
                    self.results[request_id] = result
                self.outstanding = outstanding_after_completion(self.outstanding, len(pending.followers))
                self.condition.notify_all()
            pending.event.set()
