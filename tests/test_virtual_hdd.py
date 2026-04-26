from __future__ import annotations

import random
from pathlib import Path
from typing import Any, cast

from _pytest.monkeypatch import MonkeyPatch

from clatterdrive.hdd import VirtualHDD
from clatterdrive.profiles import resolve_acoustic_profile, resolve_drive_profile
from clatterdrive.scheduler import OSScheduler
from clatterdrive.storage_events import StorageEvent

def test_virtual_hdd_writeback_queue_flushes(isolated_backing_dir: Path) -> None:
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    try:
        stats = vhdd.access_file("/buffered.bin", 0, 32768, is_write=True)
        assert stats["type"] == "WRITE"
        assert vhdd.writeback_bytes > 0

        vhdd.sync_all()
        assert vhdd.writeback_bytes == 0
    finally:
        vhdd.stop()

def test_virtual_hdd_reset_runtime_state_flushes_dirty_writeback(isolated_backing_dir: Path) -> None:
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    try:
        vhdd.access_file("/dirty.bin", 0, 32768, is_write=True)
        assert vhdd.writeback_bytes > 0
        vhdd.reset_runtime_state()
        assert vhdd.writeback_bytes == 0
    finally:
        vhdd.stop()

def test_buffered_write_stats_do_not_report_full_cache_hit_after_journal_io(
    isolated_backing_dir: Path,
) -> None:
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    try:
        stats = vhdd.access_file("/journaled.bin", 0, 4096, is_write=True)
        assert stats["type"] == "WRITE"
        assert stats["cache_hit"] is False
    finally:
        vhdd.stop()

def test_virtual_hdd_prepare_overwrite_discards_old_tail(isolated_backing_dir: Path) -> None:
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    try:
        vhdd.access_file("/overwrite.bin", 0, 32768, is_write=True)
        vhdd.sync_all()
        assert vhdd.fs.files["/overwrite.bin"].size == 32768

        vhdd.prepare_overwrite("/overwrite.bin")
        assert vhdd.fs.files["/overwrite.bin"].size == 0
        assert vhdd.access_file("/overwrite.bin", 16384, 4096, is_write=False)["extents"] == 0
    finally:
        vhdd.stop()

def test_virtual_hdd_materializes_existing_backing_file(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()
    payload = backing / "existing.bin"
    payload.write_bytes(b"x" * 8192)

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        stats = vhdd.access_file("/existing.bin", 0, 4096, is_write=False)
        assert stats["extents"] == 1
        assert "/existing.bin" in vhdd.fs.files
    finally:
        vhdd.stop()

def test_virtual_hdd_stop_drains_background_threads(isolated_backing_dir: Path) -> None:
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    scheduler = OSScheduler(vhdd.model)
    vhdd.set_scheduler(scheduler)
    try:
        vhdd.access_file("/shutdown.bin", 0, 32768, is_write=True)
    finally:
        vhdd.stop()

    assert not vhdd.writeback_thread.is_alive()
    assert not vhdd.model.background_thread.is_alive()
    assert not scheduler.dispatch_thread.is_alive()

def test_virtual_hdd_directory_operations_update_runtime_tree(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        mkdir_stats = vhdd.create_directory("/docs")
        create_stats = vhdd.create_empty_file("/docs/empty.txt")
        move_stats = vhdd.rename_path("/docs", "/archive")
        delete_stats = vhdd.delete_directory("/archive")

        assert mkdir_stats["type"] == "MKCOL"
        assert create_stats["type"] == "CREATE"
        assert move_stats["type"] == "MOVE"
        assert delete_stats["type"] == "DELETE"
        assert "/archive" not in vhdd.fs.directories
    finally:
        vhdd.stop()

def test_virtual_hdd_loads_profiles_from_env(isolated_backing_dir: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HDD_DRIVE_PROFILE", "external_usb_enclosure")
    monkeypatch.setenv("FAKE_HDD_ACOUSTIC_PROFILE", "drive_on_desk")
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    try:
        assert vhdd.drive_profile == resolve_drive_profile("external_usb_enclosure")
        assert vhdd.acoustic_profile == resolve_acoustic_profile("drive_on_desk", drive_profile=vhdd.drive_profile)
        assert vhdd.model.target_rpm == 5400
    finally:
        vhdd.stop()

def test_virtual_hdd_copy_file_preserves_size_and_populates_destination(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        vhdd.create_directory("/docs")
        vhdd.access_file("/docs/source.bin", 0, 12288, is_write=True)
        vhdd.sync_all()

        stats = vhdd.copy_file("/docs/source.bin", "/docs/copy.bin")

        assert stats["type"] == "COPY"
        assert "/docs/copy.bin" in vhdd.fs.files
        assert vhdd.fs.files["/docs/copy.bin"].size == 12288
        assert vhdd.fs.read("/docs/copy.bin", 0, 4096)
    finally:
        vhdd.stop()

def test_virtual_hdd_copy_file_handles_multi_chunk_transfer(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        vhdd.copy_chunk_bytes = 128 * 1024
        vhdd.create_directory("/docs")
        vhdd.access_file("/docs/source.bin", 0, 640 * 1024, is_write=True)
        vhdd.sync_all()

        stats = vhdd.copy_file("/docs/source.bin", "/docs/copy.bin")

        assert stats["type"] == "COPY"
        assert stats["extents"] >= 2
        assert vhdd.fs.files["/docs/copy.bin"].size == 640 * 1024
        vhdd.fs.assert_consistent()
    finally:
        vhdd.stop()

def test_virtual_hdd_lookup_cache_invalidates_after_copy_move_delete(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        vhdd.create_directory("/docs")
        vhdd.access_file("/docs/source.bin", 0, 8192, is_write=True)
        vhdd.sync_all()
        vhdd.lookup_path("/docs")
        vhdd.lookup_path("/docs/source.bin")

        assert "/docs" in vhdd.lookup_cache
        assert "/docs/source.bin" in vhdd.lookup_cache

        vhdd.copy_file("/docs/source.bin", "/docs/copy.bin")
        assert "/docs" not in vhdd.lookup_cache
        assert "/docs/copy.bin" not in vhdd.lookup_cache

        vhdd.lookup_path("/docs")
        vhdd.lookup_path("/docs/copy.bin")
        vhdd.rename_path("/docs/copy.bin", "/docs/moved.bin")
        assert "/docs" not in vhdd.lookup_cache
        assert "/docs/copy.bin" not in vhdd.lookup_cache
        assert "/docs/moved.bin" not in vhdd.lookup_cache

        vhdd.lookup_path("/docs")
        vhdd.lookup_path("/docs/moved.bin")
        vhdd.delete_path("/docs/moved.bin")
        assert "/docs" not in vhdd.lookup_cache
        assert "/docs/moved.bin" not in vhdd.lookup_cache
    finally:
        vhdd.stop()


def test_virtual_hdd_reconciles_out_of_band_file_size_changes(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()
    payload = backing / "existing.bin"
    payload.write_bytes(b"x" * 4096)

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        vhdd.access_file("/existing.bin", 0, 4096, is_write=False)
        assert vhdd.fs.files["/existing.bin"].size == 4096

        payload.write_bytes(b"y" * 12288)
        vhdd.lookup_path("/existing.bin")

        assert vhdd.fs.files["/existing.bin"].size == 12288
    finally:
        vhdd.stop()


def test_virtual_hdd_reconciles_out_of_band_file_deletion(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()
    payload = backing / "existing.bin"
    payload.write_bytes(b"x" * 4096)

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        vhdd.access_file("/existing.bin", 0, 4096, is_write=False)
        assert "/existing.bin" in vhdd.fs.files

        payload.unlink()
        stats = vhdd.lookup_path("/existing.bin")

        assert stats["type"] == "LOOKUP"
        assert "/existing.bin" not in vhdd.fs.files
    finally:
        vhdd.stop()


def test_virtual_hdd_copy_overwrite_large_file_succeeds_while_background_writeback_is_active(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        vhdd.copy_chunk_bytes = 128 * 1024
        vhdd.create_directory("/docs")
        vhdd.access_file("/docs/source.bin", 0, 640 * 1024, is_write=True)
        vhdd.sync_all()
        vhdd.access_file("/docs/dest.bin", 0, 384 * 1024, is_write=True)
        vhdd.sync_all()

        vhdd.access_file("/docs/busy.bin", 0, 256 * 1024, is_write=True)
        assert vhdd.writeback_bytes > 0

        stats = vhdd.copy_file("/docs/source.bin", "/docs/dest.bin")

        assert stats["type"] == "COPY"
        assert vhdd.fs.files["/docs/dest.bin"].size == 640 * 1024

        vhdd.sync_all()
        vhdd.fs.assert_consistent()
    finally:
        vhdd.stop()


def test_virtual_hdd_copy_overwrite_reuses_existing_destination_extents(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        vhdd.create_directory("/docs")
        vhdd.access_file("/docs/source.bin", 0, 16384, is_write=True)
        vhdd.access_file("/docs/dest.bin", 0, 16384, is_write=True)
        vhdd.sync_all()

        original_extent = vhdd.fs.files["/docs/dest.bin"].extents[0]
        stats = vhdd.copy_file("/docs/source.bin", "/docs/dest.bin")

        assert stats["type"] == "COPY"
        assert vhdd.fs.files["/docs/dest.bin"].extents[0] == original_extent
    finally:
        vhdd.stop()


def test_virtual_hdd_partial_block_write_performs_read_modify_write(
    isolated_backing_dir: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    calls: list[tuple[bool, str, int]] = []
    original_submit = vhdd.model.submit_physical_access

    def recording_submit(lba: int, size_bytes: int, is_write: bool, **kwargs: Any) -> Any:
        calls.append((is_write, str(kwargs.get("op_kind", "")), size_bytes))
        return original_submit(lba, size_bytes, is_write, **kwargs)

    monkeypatch.setattr(vhdd.model, "submit_physical_access", cast(Any, recording_submit))
    try:
        vhdd.access_file("/rmw.bin", 0, 8192, is_write=True, sync=True)
        calls.clear()

        vhdd.access_file("/rmw.bin", 512, 1024, is_write=True, sync=True)

        assert any(not is_write and op_kind == "data" for is_write, op_kind, _ in calls)

        calls.clear()
        vhdd.access_file("/rmw.bin", 4096, 4096, is_write=True, sync=True)

        assert not any(not is_write and op_kind == "data" for is_write, op_kind, _ in calls)
    finally:
        vhdd.stop()


def test_virtual_hdd_clusters_adjacent_writeback_entries_into_fewer_media_ops(isolated_backing_dir: Path) -> None:
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    try:
        vhdd.access_file("/clustered.bin", 0, 4096, is_write=True)
        vhdd.access_file("/clustered.bin", 4096, 4096, is_write=True)
        vhdd.access_file("/clustered.bin", 8192, 4096, is_write=True)
        vhdd.access_file("/clustered.bin", 256 * 4096, 4096, is_write=True)

        batch = vhdd._dequeue_writeback_batch(force=True)
        operations = vhdd._cluster_writeback_operations(batch)
        data_operations = [operation for operation in operations if operation.kind == "writeback"]

        assert len(batch) > 4
        assert len(operations) < len(batch)
        assert len(data_operations) == 1
        assert data_operations[0].block_count >= 4
    finally:
        vhdd.stop()

def test_virtual_hdd_repeated_create_delete_recreate_cycles_keep_state_consistent(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        path = "/flappy.bin"
        for size in (0, 4096, 12288, 0, 8192):
            if size == 0:
                vhdd.create_empty_file(path)
            else:
                vhdd.access_file(path, 0, size, is_write=True)
                vhdd.sync_all()
            assert path in vhdd.fs.files
            assert vhdd.fs.files[path].size == size
            vhdd.delete_path(path)
            assert path not in vhdd.fs.files
            vhdd.fs.assert_consistent()
    finally:
        vhdd.stop()


def test_virtual_hdd_delete_flushes_buffered_writeback_before_releasing_file(isolated_backing_dir: Path) -> None:
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    try:
        vhdd.access_file("/soon-gone.bin", 0, 256 * 1024, is_write=True)
        assert vhdd.writeback_bytes > 0

        stats = vhdd.delete_path("/soon-gone.bin")

        assert stats["type"] == "DELETE"
        assert vhdd.writeback_bytes == 0
        assert "/soon-gone.bin" not in vhdd.fs.files
        vhdd.fs.assert_consistent()
    finally:
        vhdd.stop()


def test_virtual_hdd_move_directory_with_buffered_descendants_keeps_tree_consistent(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        vhdd.create_directory("/docs")
        vhdd.create_directory("/docs/sub")
        vhdd.access_file("/docs/sub/data.bin", 0, 192 * 1024, is_write=True)
        assert vhdd.writeback_bytes > 0

        stats = vhdd.rename_path("/docs", "/archive")

        assert stats["type"] == "MOVE"
        assert "/archive/sub/data.bin" in vhdd.fs.files
        assert "/docs/sub/data.bin" not in vhdd.fs.files

        vhdd.sync_all()
        vhdd.fs.assert_consistent()
    finally:
        vhdd.stop()


def test_virtual_hdd_delete_directory_flushes_buffered_descendants_before_teardown(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    vhdd = VirtualHDD(str(backing), latency_scale=0.0)
    try:
        vhdd.create_directory("/docs")
        vhdd.create_directory("/docs/sub")
        vhdd.access_file("/docs/sub/data.bin", 0, 192 * 1024, is_write=True)
        assert vhdd.writeback_bytes > 0

        stats = vhdd.delete_directory("/docs")

        assert stats["type"] == "DELETE"
        assert vhdd.writeback_bytes == 0
        assert "/docs" not in vhdd.fs.directories
        assert "/docs/sub/data.bin" not in vhdd.fs.files
        vhdd.fs.assert_consistent()
    finally:
        vhdd.stop()

def test_virtual_hdd_can_emit_to_injected_event_sink(isolated_backing_dir: Path) -> None:
    class CapturingSink:
        def __init__(self) -> None:
            self.events: list[StorageEvent] = []

        def publish_event(self, event: StorageEvent) -> None:
            self.events.append(event)

    sink = CapturingSink()
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0, event_sink=sink)
    try:
        vhdd.access_file("/captured.bin", 0, 4096, is_write=True)
        vhdd.access_file("/captured.bin", 0, 4096, is_write=False)

        assert sink.events
        assert any(event.op_kind == "data" for event in sink.events)
    finally:
        vhdd.stop()


def test_many_small_writes_emit_more_metadata_events_than_one_large_write(
    isolated_backing_dir: Path,
) -> None:
    class CapturingSink:
        def __init__(self) -> None:
            self.events: list[StorageEvent] = []

        def publish_event(self, event: StorageEvent) -> None:
            self.events.append(event)

    large_sink = CapturingSink()
    large = VirtualHDD(str(isolated_backing_dir / "large"), latency_scale=0.0, event_sink=large_sink)
    try:
        large.access_file("/large.bin", 0, 16 * 4096, is_write=True)
        large.sync_all()
    finally:
        large.stop()

    small_sink = CapturingSink()
    small = VirtualHDD(str(isolated_backing_dir / "small"), latency_scale=0.0, event_sink=small_sink)
    try:
        small.create_directory("/small")
        for index in range(16):
            small.access_file(f"/small/file-{index:02d}.bin", 0, 4096, is_write=True)
        small.sync_all()
    finally:
        small.stop()

    large_metadata_events = [event for event in large_sink.events if event.op_kind in {"journal", "metadata"}]
    small_metadata_events = [event for event in small_sink.events if event.op_kind in {"journal", "metadata"}]

    assert len(small_metadata_events) > len(large_metadata_events)
    assert max(event.block_count for event in large_sink.events) >= 16


def test_random_operation_invariants_hold_under_mixed_workload(isolated_backing_dir: Path) -> None:
    random.seed(3)
    vhdd = VirtualHDD(str(isolated_backing_dir), latency_scale=0.0)
    scheduler = OSScheduler(vhdd.model)
    vhdd.set_scheduler(scheduler)
    paths = [f"/mixed-{idx}.bin" for idx in range(6)]
    try:
        for _ in range(120):
            path = random.choice(paths)
            op = random.choice(["write", "read", "delete", "lookup", "sync"])
            if op == "write":
                vhdd.access_file(
                    path,
                    random.randint(0, 32) * 4096,
                    random.randint(1, 4) * 4096,
                    is_write=True,
                    sync=random.random() < 0.1,
                )
            elif op == "read":
                vhdd.access_file(
                    path,
                    random.randint(0, 32) * 4096,
                    random.randint(1, 4) * 4096,
                    is_write=False,
                )
            elif op == "delete":
                vhdd.delete_path(path)
            elif op == "lookup":
                vhdd.lookup_path(path)
            else:
                vhdd.sync_all()

            for inode in vhdd.fs.files.values():
                logical_end = -1
                for logical_start, physical_start, blocks in inode.extents:
                    assert logical_start > logical_end
                    assert physical_start >= vhdd.fs.data_start_block
                    assert blocks > 0
                    logical_end = logical_start + blocks - 1
            vhdd.fs.assert_consistent()
    finally:
        vhdd.stop()
