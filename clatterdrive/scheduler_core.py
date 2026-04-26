from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final


@dataclass(frozen=True)
class SchedulerRequest:
    id: str
    lba: int
    size: int
    is_write: bool
    op_kind: str
    sync: bool
    arrival_time: float
    deadline: float
    extent_count: int = 0
    directory_entry_count: int = 0
    fragmentation_score: int = 0


REQUEST_ID_PREFIX: Final[str] = "req-"


def next_request_id(sequence: int) -> tuple[str, int]:
    return f"{REQUEST_ID_PREFIX}{sequence}", sequence + 1


def build_request(
    *,
    sequence: int,
    lba: int,
    size: int,
    is_write: bool,
    op_kind: str,
    sync: bool,
    arrival_time: float,
    read_deadline_s: float,
    write_deadline_s: float,
    extent_count: int = 0,
    directory_entry_count: int = 0,
    fragmentation_score: int = 0,
) -> tuple[SchedulerRequest, int]:
    request_id, next_sequence = next_request_id(sequence)
    deadline = arrival_time + (write_deadline_s if is_write else read_deadline_s)
    return (
        SchedulerRequest(
            id=request_id,
            lba=lba,
            size=size,
            is_write=is_write,
            op_kind=op_kind,
            sync=sync,
            arrival_time=arrival_time,
            deadline=deadline,
            extent_count=max(0, int(extent_count)),
            directory_entry_count=max(0, int(directory_entry_count)),
            fragmentation_score=max(0, int(fragmentation_score)),
        ),
        next_sequence,
    )


def size_in_blocks(size_bytes: int, block_bytes: int) -> int:
    return max(1, -(-size_bytes // block_bytes))


def can_submit(outstanding: int, max_queue_depth: int) -> bool:
    return outstanding < max_queue_depth


def outstanding_after_submit(outstanding: int) -> int:
    return outstanding + 1


def outstanding_after_completion(outstanding: int, follower_count: int) -> int:
    return max(0, outstanding - 1 - follower_count)


def completion_ids(root_id: str, followers: tuple[str, ...]) -> tuple[str, ...]:
    return (root_id, *followers)


def merge_request(
    queue: tuple[SchedulerRequest, ...],
    incoming: SchedulerRequest,
    *,
    block_bytes: int,
) -> tuple[tuple[SchedulerRequest, ...], str | None]:
    updated_queue = list(queue)
    for index, request in enumerate(updated_queue):
        if (
            request.is_write == incoming.is_write
            and request.op_kind == incoming.op_kind
            and request.sync == incoming.sync
            and request.lba + size_in_blocks(request.size, block_bytes) == incoming.lba
        ):
            updated_queue[index] = replace(
                request,
                size=request.size + incoming.size,
                deadline=min(request.deadline, incoming.deadline),
                extent_count=request.extent_count + incoming.extent_count,
                directory_entry_count=max(request.directory_entry_count, incoming.directory_entry_count),
                fragmentation_score=max(request.fragmentation_score, incoming.fragmentation_score),
            )
            return tuple(updated_queue), request.id
    return (*queue, incoming), None


def pick_next_request(
    queue: tuple[SchedulerRequest, ...],
    *,
    current_lba: int,
    direction: int,
    now: float,
) -> tuple[tuple[SchedulerRequest, ...], SchedulerRequest | None, int]:
    if not queue:
        return queue, None, direction

    expired = [request for request in queue if request.deadline <= now]
    if expired:
        request = min(expired, key=lambda item: (item.deadline, item.arrival_time))
        remaining = tuple(item for item in queue if item.id != request.id)
        return remaining, request, direction

    forward = [request for request in queue if request.lba >= current_lba]
    backward = [request for request in queue if request.lba < current_lba]

    if direction >= 0 and forward:
        request = min(forward, key=lambda item: item.lba)
        remaining = tuple(item for item in queue if item.id != request.id)
        return remaining, request, direction
    if direction < 0 and backward:
        request = max(backward, key=lambda item: item.lba)
        remaining = tuple(item for item in queue if item.id != request.id)
        return remaining, request, direction

    return pick_next_request(queue, current_lba=current_lba, direction=-direction, now=now)
