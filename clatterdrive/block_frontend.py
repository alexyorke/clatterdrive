from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .hdd.core import OperationStats
from .hdd.latency import HDDLatencyModel
from .scheduler import OSScheduler


@dataclass(frozen=True)
class FrontendCapability:
    name: str
    status: str
    description: str


FRONTEND_CAPABILITIES = (
    FrontendCapability(
        name="webdav",
        status="production",
        description="File-oriented WebDAV frontend used by the bundled launchers.",
    ),
    FrontendCapability(
        name="block_adapter",
        status="extension_api",
        description="Raw block adapter for future WinFsp, FUSE, NBD, or driver integrations.",
    ),
)


@runtime_checkable
class BlockStore(Protocol):
    def read_blocks(self, lba: int, block_count: int, block_size: int) -> bytes: ...

    def write_blocks(self, lba: int, data: bytes, block_size: int) -> None: ...

    def flush(self) -> None: ...

    def discard_blocks(self, lba: int, block_count: int) -> None: ...


class SparseMemoryBlockStore:
    """Small sparse store intended for adapter tests and frontend prototypes."""

    def __init__(self) -> None:
        self._blocks: dict[int, bytes] = {}
        self._lock = threading.Lock()

    def read_blocks(self, lba: int, block_count: int, block_size: int) -> bytes:
        zero = bytes(block_size)
        with self._lock:
            return b"".join(self._blocks.get(block, zero) for block in range(lba, lba + block_count))

    def write_blocks(self, lba: int, data: bytes, block_size: int) -> None:
        with self._lock:
            for offset in range(0, len(data), block_size):
                self._blocks[lba + (offset // block_size)] = bytes(data[offset : offset + block_size])

    def flush(self) -> None:
        return None

    def discard_blocks(self, lba: int, block_count: int) -> None:
        with self._lock:
            for block in range(lba, lba + block_count):
                self._blocks.pop(block, None)


class LatencyBlockDevice:
    """Translate aligned raw block I/O into the shared HDD latency/NCQ model."""

    def __init__(
        self,
        model: HDDLatencyModel,
        store: BlockStore,
        scheduler: OSScheduler | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self.scheduler = scheduler or OSScheduler(model, max_queue_depth=model.ncq_depth)
        self._owns_scheduler = scheduler is None
        # Completion includes the backing store, not just simulated latency.
        # Serialize this synchronous adapter until it has dispatch callbacks
        # that can commit data in the scheduler's actual execution order.
        self._operation_lock = threading.RLock()
        self._closed = False

    @property
    def capacity_bytes(self) -> int:
        return self.model.addressable_blocks * self.model.block_bytes

    def close(self) -> None:
        with self._operation_lock:
            if self._closed:
                return
            self.store.flush()
            if self._owns_scheduler:
                self.scheduler.stop()
            self._closed = True

    def _validate_span(self, lba: int, block_count: int) -> None:
        if self._closed:
            raise RuntimeError("block device is closed")
        if lba < 0 or block_count <= 0 or lba + block_count > self.model.addressable_blocks:
            raise ValueError("block request is outside the addressable range")

    def _submit(self, lba: int, block_count: int, *, is_write: bool, op_kind: str, sync: bool) -> OperationStats:
        self._validate_span(lba, block_count)
        request_id = self.scheduler.submit_bio(
            lba,
            block_count * self.model.block_bytes,
            is_write,
            op_kind=op_kind,
            sync=sync,
            extent_count=1 if op_kind == "data" else 0,
        )
        result = self.scheduler.wait_for_completion(request_id)
        if not isinstance(result, OperationStats):
            raise TypeError("block scheduler returned an invalid result")
        return result

    def read_blocks(self, lba: int, block_count: int) -> tuple[bytes, OperationStats]:
        with self._operation_lock:
            stats = self._submit(lba, block_count, is_write=False, op_kind="data", sync=False)
            data = self.store.read_blocks(lba, block_count, self.model.block_bytes)
            if len(data) != block_count * self.model.block_bytes:
                raise OSError("backing store returned a short block read")
            return data, stats

    def write_blocks(self, lba: int, data: bytes, *, force_unit_access: bool = False) -> OperationStats:
        if not data or len(data) % self.model.block_bytes != 0:
            raise ValueError("block writes must contain a positive whole number of model blocks")
        block_count = len(data) // self.model.block_bytes
        with self._operation_lock:
            stats = self._submit(
                lba,
                block_count,
                is_write=True,
                op_kind="data",
                sync=force_unit_access,
            )
            self.store.write_blocks(lba, data, self.model.block_bytes)
            if force_unit_access:
                self.store.flush()
            return stats

    def flush(self) -> OperationStats:
        with self._operation_lock:
            lba = min(self.model.get_estimated_lba(), self.model.addressable_blocks - 1)
            stats = self._submit(lba, 1, is_write=True, op_kind="flush", sync=True)
            self.store.flush()
            return stats.with_updates(type="FLUSH")

    def discard_blocks(self, lba: int, block_count: int) -> OperationStats:
        with self._operation_lock:
            self._validate_span(lba, block_count)
            self.store.discard_blocks(lba, block_count)
            return OperationStats(total_ms=0.02, op_type="DISCARD", block_count=block_count)


__all__ = [
    "FRONTEND_CAPABILITIES",
    "BlockStore",
    "FrontendCapability",
    "LatencyBlockDevice",
    "SparseMemoryBlockStore",
]
