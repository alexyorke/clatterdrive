from __future__ import annotations

from dataclasses import replace
import threading
from pathlib import Path
from typing import Any

import pytest

from clatterdrive.fs import FileSystemSimulator
from clatterdrive.fs.core import IOOperation
from clatterdrive.hdd.core import OperationStats
from clatterdrive.hdd import HDDLatencyModel, VirtualHDD
from clatterdrive.scheduler import OSScheduler
from clatterdrive.scheduler_core import SchedulerRequest, merge_request
from clatterdrive.hdd.core import background_decision
from clatterdrive.audio.engine import HDDAudioEngine


@pytest.mark.parametrize("front", [False, True])
def test_partial_block_requests_do_not_lose_a_physical_block(front: bool) -> None:
    first = SchedulerRequest("first", 5, 1, False, "data", False, 0, 1)
    second = replace(first, id="second", lba=4 if front else 6)
    queue, _ = merge_request((first,), second, block_bytes=4096)
    assert sum((request.size + 4095) // 4096 for request in queue) == 2


def test_headless_stop_does_not_close_recorder_while_worker_is_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    engine = HDDAudioEngine(seed=1)
    worker = Mock()
    worker.is_alive.return_value = True
    recorder = Mock()
    monkeypatch.setattr(engine, "_headless_render_thread", worker)
    monkeypatch.setattr(engine, "tee_recorder", recorder)
    with pytest.raises(RuntimeError, match="headless"):
        engine.stop()
    recorder.close.assert_not_called()
    assert engine._headless_render_thread is worker
    worker.is_alive.return_value = False
    engine.stop()
    recorder.close.assert_called_once()


@pytest.mark.parametrize("diagnostics", [False, True])
def test_negative_audio_frames_do_not_rewind_clock(diagnostics: bool) -> None:
    engine = HDDAudioEngine(seed=1)
    render = engine.render_chunk_with_diagnostics if diagnostics else engine.render_chunk
    with pytest.raises(ValueError, match="frames"):
        render(-1)
    assert engine.render_frame_cursor == 0


def test_scheduler_survives_head_position_lookup_failure() -> None:
    class Model:
        block_bytes = 4096

        def get_estimated_lba(self) -> int:
            raise OSError("position unavailable")

        def submit_physical_access(self, *_args: Any, **_kwargs: Any) -> int:
            return 7

    scheduler = OSScheduler(Model())
    try:
        request = scheduler.submit_bio(0, 4096, False)
        assert scheduler.events[request].wait(2), "dispatch worker stranded the request"
        assert scheduler.wait_for_completion(request) == 7
    finally:
        scheduler.stop()


@pytest.mark.parametrize("depth", [0, -1])
def test_scheduler_rejects_unusable_queue_depth(depth: int) -> None:
    with pytest.raises(ValueError, match="queue depth"):
        OSScheduler(object(), max_queue_depth=depth)


def test_speculative_cache_cannot_make_out_of_range_reads_succeed() -> None:
    model = HDDLatencyModel(128, latency_scale=0, enable_background_scan=False)
    try:
        model.submit_physical_access(127, 4096, False)
        with pytest.raises(ValueError, match="range"):
            model.submit_physical_access(128, 4096, False)
        with pytest.raises(ValueError, match="range"):
            model.submit_physical_access(127, 8192, False)
    finally:
        model.stop()


@pytest.mark.parametrize("offset,length", [(-1, 1), (0, -1)])
def test_invalid_file_ranges_do_not_mutate_allocation(offset: int, length: int) -> None:
    fs = FileSystemSimulator(total_gb=0.1)
    previous = fs.state
    with pytest.raises(ValueError):
        fs.write("/file", offset, length)
    assert fs.state is previous
    with pytest.raises(ValueError):
        fs.read("/file", offset, length)


def test_empty_write_does_not_extend_file() -> None:
    fs = FileSystemSimulator(total_gb=0.1)
    fs.create_empty_file("/file")
    previous = fs.state
    assert fs.write("/file", 10000, 0) == []
    assert fs.state is previous
    assert fs.files["/file"].size == 0


def test_root_reconciliation_removes_missing_descendants(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()
    item = backing / "gone.txt"
    item.write_bytes(b"gone")
    disk = VirtualHDD(str(backing), capacity_gb=0.1, latency_scale=0)
    try:
        disk.ensure_tree_known("/")
        assert "/gone.txt" in disk.fs.files
        item.unlink()
        disk.ensure_tree_known("/")
        assert "/gone.txt" not in disk.fs.files
        disk.fs.assert_consistent()
    finally:
        disk.stop()


def test_stopped_model_rejects_readiness_wait() -> None:
    model = HDDLatencyModel(128, latency_scale=0, enable_background_scan=False)
    model.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        model._wait_for_ready_poll()


def test_startup_failure_reaches_waiting_request(monkeypatch: pytest.MonkeyPatch) -> None:
    model = HDDLatencyModel(128, latency_scale=0, start_ready=False, enable_background_scan=False)
    def fail_trace(*_args: Any, **_kwargs: Any) -> bool:
        raise OSError("startup telemetry failed")
    monkeypatch.setattr(model, "_run_startup_trace", fail_trace)
    try:
        model.begin_async_startup()
        thread = model.transition_thread
        if thread is not None:
            thread.join(2)
        with pytest.raises(RuntimeError, match="transition failed"):
            model._wait_for_ready_poll()
    finally:
        model.stop()


def test_idle_power_step_does_not_park_during_io() -> None:
    model = HDDLatencyModel(128, latency_scale=0, enable_background_scan=False)
    try:
        with model.io_lock:
            model.last_access_time = model.clock.now() - 100
            model._background_power_step()
            assert model.heads_loaded
            assert model.power_state == "active"
    finally:
        model.stop()


def test_slowdown_preserves_idle_timer_origin() -> None:
    model = HDDLatencyModel(128, latency_scale=0, enable_background_scan=False)
    try:
        with model.lock:
            last_access = model.clock.now() - 30
            model.last_access_time = last_access
            cancel = threading.Event()
            model.transition_cancel = cancel
            model._finish_transition(cancel, "slowdown", "low_rpm_idle")
            assert model.last_access_time == last_access
    finally:
        model.stop()


def test_expired_standby_timer_takes_priority_over_calibration() -> None:
    model = HDDLatencyModel(128, latency_scale=0, enable_background_scan=False)
    try:
        mechanical = replace(model._core_mechanical_state(), last_access_time=0)
        decision = background_decision(
            model.core_config, mechanical, transition_active=False, now=10,
            last_idle_calibration_time=0, unload_after_s=100,
            low_rpm_after_s=200, standby_after_s=9,
        )
        assert decision.should_spindown
        assert not decision.calibrate
    finally:
        model.stop()


def test_failed_writeback_retains_batch_and_reports_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    disk = VirtualHDD(str(tmp_path), capacity_gb=0.1, latency_scale=0)
    disk.running = False
    disk.writeback_thread.join(2)
    original = disk._run_ops
    def fail(*_args: Any, **_kwargs: Any) -> OperationStats:
        raise OSError("media failure")
    try:
        disk.model.dirty_expire_s = 0
        disk._enqueue_writeback([IOOperation(disk.fs.data_start_block, 1, "data")])
        monkeypatch.setattr(disk, "_run_ops", fail)
        with pytest.raises(OSError, match="media failure"):
            disk._process_writeback_batch(force=False)
        assert disk.writeback_bytes == 4096
        assert len(disk.writeback_queue) == 1
        assert disk.inflight_writebacks == 0
        assert not disk.writeback_idle_event.is_set()
        monkeypatch.setattr(disk, "_run_ops", original)
        with pytest.raises(OSError, match="background writeback failed"):
            disk.sync_all()
        disk.sync_all()
        assert disk.writeback_bytes == 0
        assert disk.writeback_idle_event.is_set()
    finally:
        monkeypatch.setattr(disk, "_run_ops", original)
        disk.stop()


def test_foreground_writeback_is_marked_inflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    disk = VirtualHDD(str(tmp_path), capacity_gb=0.1, latency_scale=0)
    disk.running = False
    disk.writeback_thread.join(2)
    original = disk._run_ops
    def check(*_args: Any, **_kwargs: Any) -> OperationStats:
        assert disk.inflight_writebacks == 1
        assert not disk.writeback_idle_event.is_set()
        return OperationStats(total_ms=1)
    try:
        disk._enqueue_writeback([IOOperation(disk.fs.data_start_block, 1, "data")])
        monkeypatch.setattr(disk, "_run_ops", check)
        assert disk.sync_all() == 1
        assert disk.inflight_writebacks == 0
    finally:
        monkeypatch.setattr(disk, "_run_ops", original)
        disk.stop()


def test_merged_dispatch_statistics_are_counted_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    disk = VirtualHDD(str(tmp_path), capacity_gb=0.1, latency_scale=0)
    scheduler = Mock()
    scheduler.wait_for_completion.return_value = OperationStats(total_ms=12, size_bytes=8192, block_count=2)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(disk, "scheduler", scheduler)
            result = disk._run_ops([IOOperation(300, 1, "data"), IOOperation(301, 1, "data")], False)
        assert result.total_ms == 12
        assert result.size_bytes == 8192
        assert result.block_count == 2
    finally:
        disk.stop()


@pytest.mark.parametrize("linked_tree", ["source", "destination"])
def test_recursive_copy_rejects_links_before_mutating_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked_tree: str,
) -> None:
    import os
    from wsgidav.dav_error import DAVError
    from clatterdrive.webdav.provider import _copy_directory_contents

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "new.txt").write_text("new")
    (destination / "old.txt").write_text("old")
    link = tmp_path / linked_tree / "link"
    link.mkdir()
    original = os.path.islink
    monkeypatch.setattr(os.path, "islink", lambda path: str(path) == str(link) or original(path))
    with pytest.raises(DAVError):
        _copy_directory_contents(str(source), str(destination))
    assert (destination / "old.txt").read_text() == "old"
    assert not (destination / "new.txt").exists()
