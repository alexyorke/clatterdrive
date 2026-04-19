from __future__ import annotations

import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from fs_simulator import FileSystemSimulator, IOOperation
from hdd_core import OperationStats, empty_operation_stats, merge_operation_stats
from hdd_latency import HDDLatencyModel
from profiles import AcousticProfile, DriveProfile, resolve_selected_profiles_from_env
from runtime_deps import RuntimeDeps
from storage_events import StorageEventSink


Stats = OperationStats


@dataclass
class DirtyWrite:
    lba: int
    size_bytes: int
    op_kind: str
    enqueued_at: float

class VirtualHDD:
    def __init__(
        self,
        backing_dir: str,
        latency_scale: float = 1.0,
        cold_start: bool = False,
        async_power_on: bool = False,
        drive_profile: str | DriveProfile | None = None,
        acoustic_profile: str | AcousticProfile | None = None,
        event_sink: StorageEventSink | None = None,
        deps: RuntimeDeps | None = None,
    ) -> None:
        self.deps = deps or RuntimeDeps()
        self.clock = self.deps.clock
        self.sleeper = self.deps.sleeper
        self.drive_profile, self.acoustic_profile = resolve_selected_profiles_from_env(
            drive_profile,
            acoustic_profile,
            env=self.deps.env,
        )
        self.fs = FileSystemSimulator()
        self.model = HDDLatencyModel(
            addressable_blocks=self.fs.total_blocks,
            block_bytes=self.fs.block_size,
            latency_scale=latency_scale,
            start_ready=not cold_start,
            drive_profile=self.drive_profile,
            event_sink=event_sink,
            deps=self.deps,
        )
        self.backing_dir = backing_dir
        self.scheduler = None
        self.lookup_cache: dict[str, float] = {}
        self.lookup_cache_ttl_s = 0.35
        self.copy_chunk_bytes = 1024 * 1024

        self.writeback_lock = threading.Lock()
        self.writeback_queue: deque[DirtyWrite] = deque()
        self.writeback_bytes = 0
        self.inflight_writebacks = 0
        self.writeback_idle_event = threading.Event()
        self.writeback_idle_event.set()
        self.running = True
        self.writeback_thread = threading.Thread(target=self._writeback_loop, daemon=True)
        self.writeback_thread.start()
        if async_power_on and cold_start:
            self.begin_async_power_on()

    def stop(self) -> None:
        try:
            self.sync_all()
        finally:
            self.running = False
            self.writeback_thread.join(timeout=2.0)
            if self.scheduler and hasattr(self.scheduler, "stop"):
                self.scheduler.stop()
            self.model.stop()

    def set_scheduler(self, scheduler: Any) -> None:
        self.scheduler = scheduler

    def begin_async_power_on(self) -> None:
        self.model.begin_async_startup()

    def _real_path(self, path: str) -> str:
        normalized = self.fs._normalize_path(path)
        return os.path.join(self.backing_dir, normalized.lstrip("/").replace("/", os.sep))

    def _invalidate_lookup_prefix(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        for cached_path in list(self.lookup_cache):
            if cached_path == normalized or cached_path.startswith(f"{normalized}/"):
                self.lookup_cache.pop(cached_path, None)

    def _invalidate_lookup(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        self._invalidate_lookup_prefix(normalized)
        self.lookup_cache.pop(self.fs._parent_dir(normalized), None)

    def _ensure_known_path(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        if normalized in self.fs.files or normalized in self.fs.directories:
            return

        real_path = self._real_path(normalized)
        if os.path.isdir(real_path):
            self.fs.materialize_existing_directory(normalized)
        elif os.path.isfile(real_path):
            self.fs.materialize_existing_file(normalized, os.path.getsize(real_path))

    def _materialize_directory_children(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        real_path = self._real_path(normalized)
        if not os.path.isdir(real_path):
            return
        self.fs.materialize_existing_directory(normalized)
        for entry in os.scandir(real_path):
            child_path = self.fs._normalize_path(f"{normalized}/{entry.name}")
            if entry.is_dir():
                self.fs.materialize_existing_directory(child_path)
            elif entry.is_file():
                self.fs.materialize_existing_file(child_path, entry.stat().st_size)

    def ensure_tree_known(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        self._ensure_known_path(normalized)
        real_path = self._real_path(normalized)
        if not os.path.isdir(real_path):
            return

        for current_root, dirnames, filenames in os.walk(real_path):
            rel_root = os.path.relpath(current_root, self.backing_dir)
            virtual_root = "/" if rel_root == "." else self.fs._normalize_path(rel_root.replace(os.sep, "/"))
            self.fs.materialize_existing_directory(virtual_root)
            for dirname in dirnames:
                self.fs.materialize_existing_directory(f"{virtual_root}/{dirname}")
            for filename in filenames:
                file_path = os.path.join(current_root, filename)
                self.fs.materialize_existing_file(f"{virtual_root}/{filename}", os.path.getsize(file_path))

    def _refresh_writeback_idle_state_locked(self) -> None:
        if self.writeback_queue or self.inflight_writebacks:
            self.writeback_idle_event.clear()
        else:
            self.writeback_idle_event.set()

    def _wait_for_writeback_idle(self, timeout_s: float = 10.0) -> None:
        deadline = self.clock.now() + timeout_s
        while True:
            with self.writeback_lock:
                if not self.writeback_queue and self.inflight_writebacks == 0:
                    self.writeback_idle_event.set()
                    return
            remaining = deadline - self.clock.now()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for writeback to go idle")
            self.writeback_idle_event.wait(timeout=min(remaining, 0.1))

    def _run_ops(
        self,
        operations: list[IOOperation],
        is_write: bool,
        force_unit_access: bool = False,
    ) -> Stats:
        total_stats = OperationStats(
            total_ms=0.0,
            extents=len([operation for operation in operations if operation.kind == "data"]),
            cyl="-",
            head="-",
            cache_hit=bool(operations),
            partial_hit=False,
            startup_ms=0.0,
            startup_origin=None,
            ready_poll_ms=0.0,
            ready_poll_count=0,
            op_type="WRITE" if is_write else "READ",
        )

        for operation in operations:
            size_bytes = operation.block_count * self.fs.block_size
            if self.scheduler:
                request_id = self.scheduler.submit_bio(
                    operation.lba,
                    size_bytes,
                    is_write,
                    op_kind=operation.kind,
                    sync=force_unit_access,
                )
                result = self.scheduler.wait_for_completion(request_id)
            else:
                result = self.model.submit_physical_access(
                    operation.lba,
                    size_bytes,
                    is_write,
                    op_kind=operation.kind,
                    force_unit_access=force_unit_access,
                )

            total_stats = OperationStats(
                total_ms=total_stats.total_ms + result.total_ms,
                extents=total_stats.extents,
                cyl=result.cyl if result.cyl != "-" else total_stats.cyl,
                head=result.head if result.head != "-" else total_stats.head,
                cache_hit=total_stats.cache_hit and result.cache_hit,
                partial_hit=total_stats.partial_hit or result.partial_hit,
                startup_ms=total_stats.startup_ms + result.startup_ms,
                startup_origin=total_stats.startup_origin or result.startup_origin,
                ready_poll_ms=total_stats.ready_poll_ms + result.ready_poll_ms,
                ready_poll_count=total_stats.ready_poll_count + result.ready_poll_count,
                op_type=total_stats.op_type,
                transfer_rate_mbps=result.transfer_rate_mbps or total_stats.transfer_rate_mbps,
            )

        return total_stats

    def _enqueue_writeback(self, operations: list[IOOperation]) -> float:
        blocked_ms = 0.0
        with self.writeback_lock:
            for operation in operations:
                size_bytes = operation.block_count * self.fs.block_size
                self.writeback_queue.append(
                    DirtyWrite(
                        lba=operation.lba,
                        size_bytes=size_bytes,
                        op_kind=operation.kind,
                        enqueued_at=self.clock.now(),
                    )
                )
                self.writeback_bytes += size_bytes
                self.model.note_cached_write(operation.lba, size_bytes)
            self._refresh_writeback_idle_state_locked()

        if self.writeback_bytes > self.model.write_cache_bytes:
            blocked_ms += self._drain_write_cache(target_bytes=self.model.write_cache_bytes // 2)
        return blocked_ms

    def _dequeue_writeback_batch(self, force: bool = False, max_items: int = 32) -> list[DirtyWrite]:
        batch: list[DirtyWrite] = []
        now = self.clock.now()
        with self.writeback_lock:
            while self.writeback_queue and len(batch) < max_items:
                head = self.writeback_queue[0]
                age = now - head.enqueued_at
                cache_pressure = self.writeback_bytes > int(self.model.write_cache_bytes * 0.75)
                if not force and not cache_pressure and age < self.model.dirty_expire_s:
                    break
                batch.append(self.writeback_queue.popleft())
                self.writeback_bytes -= head.size_bytes
            self._refresh_writeback_idle_state_locked()
        return batch

    def _drain_write_cache(self, target_bytes: int = 0) -> float:
        total_ms = 0.0
        while True:
            batch = self._dequeue_writeback_batch(force=True)
            if not batch:
                break
            operations = [
                IOOperation(
                    dirty_write.lba,
                    max(1, math.ceil(dirty_write.size_bytes / self.fs.block_size)),
                    dirty_write.op_kind,
                    "writeback",
                )
                for dirty_write in batch
            ]
            total_ms += self._run_ops(operations, is_write=True)["total_ms"]
            if self.writeback_bytes <= target_bytes:
                break
        if target_bytes <= 0:
            self._wait_for_writeback_idle()
        return total_ms

    def _writeback_loop(self) -> None:
        while self.running:
            batch = self._dequeue_writeback_batch(force=False)
            if batch:
                with self.writeback_lock:
                    self.inflight_writebacks += 1
                    self._refresh_writeback_idle_state_locked()
                operations = [
                    IOOperation(
                        dirty_write.lba,
                        max(1, math.ceil(dirty_write.size_bytes / self.fs.block_size)),
                        dirty_write.op_kind,
                        "writeback",
                    )
                    for dirty_write in batch
                ]
                try:
                    self._run_ops(operations, is_write=True)
                finally:
                    with self.writeback_lock:
                        self.inflight_writebacks -= 1
                        self._refresh_writeback_idle_state_locked()
                continue
            self.sleeper.sleep(0.05)

    def sync_all(self) -> float:
        total_ms = self._drain_write_cache(target_bytes=0)
        self._wait_for_writeback_idle()
        return total_ms

    def reset_runtime_state(self) -> None:
        self.sync_all()
        self.lookup_cache.clear()
        self.model.reset_caches()

    def _empty_stats(self, op_type: str, total_ms: float = 0.01) -> Stats:
        return empty_operation_stats(op_type, total_ms=total_ms)

    def _merge_stats(self, op_type: str, *results: Stats) -> Stats:
        return merge_operation_stats(op_type, *results)

    def _apply_buffered_write(
        self,
        operations: list[IOOperation],
        data_extent_count: int,
        sync: bool = False,
    ) -> Stats:
        if not operations:
            return self._empty_stats("WRITE")

        if sync:
            sync_total = self.sync_all()
            stats = self._run_ops(operations, is_write=True, force_unit_access=True)
            return stats.with_updates(
                total_ms=stats.total_ms + sync_total,
                type="WRITE",
                extents=data_extent_count,
            )

        journal_ops = [operation for operation in operations if operation.kind == "journal"]
        buffered_ops = [operation for operation in operations if operation.kind != "journal"]
        stats = self._run_ops(journal_ops, is_write=True) if journal_ops else OperationStats()
        blocked_ms = self._enqueue_writeback(buffered_ops)
        return stats.with_updates(
            total_ms=stats.total_ms + 0.08 + blocked_ms,
            cache_hit=stats.cache_hit and blocked_ms == 0.0,
            type="WRITE",
            extents=data_extent_count,
        )

    def lookup_path(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        now = self.clock.now()
        if self.lookup_cache.get(path, 0.0) > now:
            return self._empty_stats("LOOKUP", total_ms=0.02)
        operations = self.fs.lookup(path)
        stats = self._run_ops(operations, is_write=False)
        self.lookup_cache[path] = now + self.lookup_cache_ttl_s
        return stats.with_updates(type="LOOKUP")

    def list_directory(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        self._materialize_directory_children(path)
        operations = self.fs.list_directory(path)
        if not operations:
            return self._empty_stats("READDIR")
        stats = self._run_ops(operations, is_write=False)
        return stats.with_updates(type="READDIR", extents=0)

    def create_directory(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        operations = self.fs.create_directory(path)
        self._invalidate_lookup(path)
        if not operations:
            return self._empty_stats("MKCOL")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        return stats.with_updates(type="MKCOL", extents=0)

    def refresh_directory(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        operations = self.fs.update_directory(path)
        if not operations:
            return self._empty_stats("COPY")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        return stats.with_updates(type="COPY", extents=0)

    def create_empty_file(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        operations = self.fs.create_empty_file(path)
        self._invalidate_lookup(path)
        if not operations:
            return self._empty_stats("CREATE")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        return stats.with_updates(type="CREATE", extents=0)

    def copy_file(self, source_path: str, dest_path: str) -> Stats:
        source_path = self.fs._normalize_path(source_path)
        dest_path = self.fs._normalize_path(dest_path)
        self._ensure_known_path(source_path)
        self._ensure_known_path(self.fs._parent_dir(dest_path))

        if source_path not in self.fs.files:
            return self._empty_stats("COPY")
        if dest_path in self.fs.directories:
            raise IsADirectoryError(dest_path)
        if dest_path in self.fs.files:
            self.delete_path(dest_path)

        source_inode = self.fs.files[source_path]
        source_size = source_inode.size
        if source_size <= 0:
            return self._merge_stats("COPY", self.create_empty_file(dest_path))

        chunk_size = max(self.fs.block_size, self.copy_chunk_bytes)
        copy_stats = []
        for offset in range(0, source_size, chunk_size):
            length = min(chunk_size, source_size - offset)
            read_ops = self.fs.read(source_path, offset, length)
            read_stats = self._run_ops(read_ops, is_write=False) if read_ops else self._empty_stats("READ")

            write_ops = self.fs.write(dest_path, offset, length)
            data_extent_count = len([operation for operation in write_ops if operation.kind == "data"])
            write_stats = self._apply_buffered_write(write_ops, data_extent_count, sync=False)
            copy_stats.append(self._merge_stats("COPY", read_stats, write_stats))

        self._invalidate_lookup(dest_path)
        return self._merge_stats("COPY", *copy_stats)

    def prepare_overwrite(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        if path not in self.fs.files:
            return empty_operation_stats("TRUNCATE", total_ms=0.0)
        self._invalidate_lookup(path)
        self.sync_all()
        operations = self.fs.truncate(path, size=0)
        if not operations:
            return empty_operation_stats("TRUNCATE", total_ms=0.0)
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        return stats.with_updates(type="TRUNCATE", extents=0)

    def access_file(
        self,
        path: str,
        offset: int,
        length: int,
        is_write: bool = False,
        sync: bool = False,
    ) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        operations = self.fs.write(path, offset, length) if is_write else self.fs.read(path, offset, length)
        data_extent_count = len([operation for operation in operations if operation.kind == "data"])

        if not operations:
            return self._empty_stats("WRITE" if is_write else "READ")

        if is_write:
            self._invalidate_lookup(path)
            return self._apply_buffered_write(operations, data_extent_count, sync=sync)

        stats = self._run_ops(operations, is_write=False)
        return stats.with_updates(extents=data_extent_count)

    def rename_path(self, source_path: str, dest_path: str) -> Stats:
        source_path = self.fs._normalize_path(source_path)
        dest_path = self.fs._normalize_path(dest_path)
        self._ensure_known_path(self.fs._parent_dir(dest_path))
        operations = self.fs.rename(source_path, dest_path)
        self._invalidate_lookup(source_path)
        self._invalidate_lookup(dest_path)
        self._invalidate_lookup_prefix(source_path)
        if not operations:
            return self._empty_stats("MOVE")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        return stats.with_updates(type="MOVE", extents=0)

    def delete_path(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        self._invalidate_lookup(path)
        operations = self.fs.delete(path)
        if not operations:
            return self._empty_stats("DELETE")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        return stats.with_updates(type="DELETE", extents=0)

    def delete_directory(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        operations = self.fs.delete_directory(path)
        self._invalidate_lookup(path)
        self._invalidate_lookup_prefix(path)
        if not operations:
            return self._empty_stats("DELETE")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        return stats.with_updates(type="DELETE", extents=0)
