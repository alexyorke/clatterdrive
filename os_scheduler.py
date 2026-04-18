import threading
import time
from dataclasses import dataclass, field


@dataclass
class IORequest:
    id: str
    lba: int
    size: int
    is_write: bool
    op_kind: str
    sync: bool
    arrival_time: float
    deadline: float
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

    def __init__(self, hdd_model, max_queue_depth=32, read_deadline_ms=25, write_deadline_ms=150):
        self.hdd_model = hdd_model
        self.max_queue_depth = max_queue_depth
        self.read_deadline_s = read_deadline_ms / 1000.0
        self.write_deadline_s = write_deadline_ms / 1000.0

        self.staging_queue = []
        self.results = {}
        self.events = {}
        self.direction = 1
        self.sequence = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.running = True
        self.outstanding = 0
        self.dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatch_thread.start()

    def stop(self):
        with self.condition:
            self.running = False
            for request in self.staging_queue:
                failure = RuntimeError("scheduler stopped before request was dispatched")
                self.results[request.id] = failure
                for follower in request.followers:
                    self.results[follower] = failure
                self.outstanding = max(0, self.outstanding - 1 - len(request.followers))
                request.event.set()
            self.staging_queue.clear()
            self.condition.notify_all()
        self.dispatch_thread.join(timeout=2.0)

    def submit_bio(self, lba, size, is_write, op_kind="data", sync=False):
        now = time.monotonic()
        with self.condition:
            while self.running and self.outstanding >= self.max_queue_depth:
                self.condition.wait(timeout=0.05)
            if not self.running:
                raise RuntimeError("scheduler is stopped")
            req_id = f"req-{self.sequence}"
            self.sequence += 1
            event = threading.Event()
            request = IORequest(
                id=req_id,
                lba=lba,
                size=size,
                is_write=is_write,
                op_kind=op_kind,
                sync=sync,
                arrival_time=now,
                deadline=now + (self.write_deadline_s if is_write else self.read_deadline_s),
                event=event,
            )
            self.outstanding += 1
            self.events[req_id] = event
            merged_into = self._merge_request(request)
            if merged_into is None:
                self.staging_queue.append(request)
            else:
                merged_into.followers.append(req_id)
                self.events[req_id] = merged_into.event
            self.condition.notify_all()
            return req_id

    def wait_for_completion(self, req_id):
        event = self.events[req_id]
        event.wait()
        with self.lock:
            self.events.pop(req_id, None)
            completion = self.results.pop(req_id)
        if isinstance(completion, BaseException):
            raise completion
        return completion

    def _merge_request(self, incoming):
        for request in self.staging_queue:
            if (
                request.is_write == incoming.is_write
                and request.op_kind == incoming.op_kind
                and request.sync == incoming.sync
                and request.lba + self._size_in_blocks(request.size) == incoming.lba
            ):
                request.size += incoming.size
                request.deadline = min(request.deadline, incoming.deadline)
                return request
        return None

    def _size_in_blocks(self, size_bytes):
        return max(1, -(-size_bytes // self.hdd_model.block_bytes))

    def _pick_next_request(self):
        if not self.staging_queue:
            return None

        now = time.monotonic()
        expired = [request for request in self.staging_queue if request.deadline <= now]
        if expired:
            request = min(expired, key=lambda item: (item.deadline, item.arrival_time))
            self.staging_queue.remove(request)
            return request

        current_lba = self.hdd_model.get_estimated_lba()
        forward = [request for request in self.staging_queue if request.lba >= current_lba]
        backward = [request for request in self.staging_queue if request.lba < current_lba]

        if self.direction >= 0 and forward:
            request = min(forward, key=lambda item: item.lba)
        elif self.direction < 0 and backward:
            request = max(backward, key=lambda item: item.lba)
        else:
            self.direction *= -1
            return self._pick_next_request()

        self.staging_queue.remove(request)
        return request

    def _dispatch_loop(self):
        while True:
            with self.condition:
                while self.running and not self.staging_queue:
                    self.condition.wait(timeout=0.05)
                if not self.running and not self.staging_queue:
                    return
                request = self._pick_next_request()
                queue_depth = len(self.staging_queue) + (1 if request else 0)

            if not request:
                continue

            try:
                result = self.hdd_model.submit_physical_access(
                    request.lba,
                    request.size,
                    request.is_write,
                    op_kind=request.op_kind,
                    force_unit_access=request.sync,
                    queue_depth=min(queue_depth, self.max_queue_depth),
                )
            except Exception as exc:
                result = exc

            with self.lock:
                self.results[request.id] = result
                for follower in request.followers:
                    self.results[follower] = result
                self.outstanding = max(0, self.outstanding - 1 - len(request.followers))
                self.condition.notify_all()
            request.event.set()
