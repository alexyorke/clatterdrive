import random
import threading
import time
import wave

import numpy as np
import pytest

from audio_engine import HDDAudioEngine
from fs_simulator import FileSystemSimulator
import generate_audio_samples
from generate_audio_samples import render_scenario, update_sequential_read
from hdd_model import HDDLatencyModel, VirtualHDD
from os_scheduler import OSScheduler


def test_filesystem_write_emits_metadata_and_data():
    fs = FileSystemSimulator(total_gb=1)
    operations = fs.write("/demo.bin", 0, 8192)

    kinds = [operation.kind for operation in operations]
    assert "journal" in kinds
    assert "metadata" in kinds
    data_ops = [operation for operation in operations if operation.kind == "data"]
    assert len(data_ops) == 1
    assert data_ops[0].block_count == 2


def test_lookup_reads_directory_and_inode_metadata():
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/demo.bin", 0, 4096)

    operations = fs.lookup("/demo.bin")
    assert [operation.kind for operation in operations] == ["metadata", "metadata"]
    assert operations[0].source == "dentry_lookup"
    assert operations[1].source == "inode_lookup"


def test_path_normalization_stays_posix_like_on_windows():
    fs = FileSystemSimulator(total_gb=1)
    assert fs._normalize_path("foo/bar") == "/foo/bar"
    assert fs._normalize_path("foo\\bar") == "/foo/bar"
    assert fs._normalize_path("/foo/bar") == "/foo/bar"
    assert fs._parent_dir("/foo/bar/baz.txt") == "/foo/bar"


def test_delete_returns_metadata_operations():
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/demo.bin", 0, 16384)

    operations = fs.delete("/demo.bin")
    kinds = [operation.kind for operation in operations]
    assert kinds.count("journal") == 1
    assert kinds.count("metadata") >= 2
    assert fs.get_fragmentation_score("/demo.bin") == 0


def test_truncate_frees_tail_extents_and_clears_size():
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/demo.bin", 0, 16384)
    operations = fs.truncate("/demo.bin", size=0)

    assert fs.files["/demo.bin"].size == 0
    assert fs.read("/demo.bin", 0, 4096) == []
    assert operations[0].source == "truncate_intent"
    assert operations[1].source == "inode_truncate"


def test_sparse_write_does_not_backfill_hole_reads():
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/sparse.bin", 8192, 4096)

    assert fs.read("/sparse.bin", 0, 4096) == []
    hole_neighbor = fs.read("/sparse.bin", 8192, 4096)
    assert len(hole_neighbor) == 1
    assert hole_neighbor[0].block_count == 1


def test_reads_do_not_price_blocks_past_eof():
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/tail.bin", 0, 1)

    assert len(fs.read("/tail.bin", 0, 4096)) == 1
    assert fs.read("/tail.bin", 1, 4096) == []


def test_filesystem_small_sizes_and_disk_full_are_handled():
    with pytest.raises(ValueError):
        FileSystemSimulator(total_gb=0.0001)

    fs = FileSystemSimulator(total_gb=0.05)
    with pytest.raises(OSError):
        fs.write("/too-big.bin", 0, 200 * 1024 * 1024)


def test_partial_read_cache_is_not_reported_as_full_hit():
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0)
    model.read_cache.append((100, 101, time.monotonic() + 5.0))

    result = model.submit_physical_access(100, 3 * model.block_bytes, False, op_kind="data")

    assert result["cache_hit"] is False
    assert result["partial_hit"] is True
    model.stop()


def test_non_prefix_cache_overlap_is_not_treated_as_partial_hit():
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0)
    model.read_cache.append((101, 102, time.monotonic() + 5.0))

    result = model.submit_physical_access(100, 3 * model.block_bytes, False, op_kind="data")

    assert result["cache_hit"] is False
    assert result["partial_hit"] is False
    model.stop()


def test_zone_table_does_not_extend_past_addressable_capacity():
    model = HDDLatencyModel(addressable_blocks=2048, latency_scale=0.0)
    assert model.zones[-1].end_lba == 2047
    model.stop()


def test_power_on_sequence_reports_startup_and_finishes_ready():
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0, start_ready=False)
    try:
        result = model.submit_physical_access(0, 4096, False, op_kind="data")
        assert result["startup_origin"] == "power_on"
        assert result["startup_ms"] == pytest.approx(model.power_on_to_ready_ms)
        assert result["ready_poll_count"] >= 0
        assert result["ready_poll_ms"] >= 0.0
        assert model.power_state == "active"
        assert model.current_rpm == model.target_rpm
    finally:
        model.stop()


def test_startup_stage_partitioning_respects_configured_budget():
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        start_ready=False,
        power_on_to_ready_ms=600.0,
        standby_to_ready_ms=320.0,
        unload_to_ready_ms=150.0,
        low_rpm_to_ready_ms=260.0,
    )
    try:
        assert sum(stage.duration_ms for stage in model._build_startup_sequence("power_on")) == pytest.approx(600.0)
        assert sum(stage.duration_ms for stage in model._build_startup_sequence("standby")) == pytest.approx(320.0)
        assert sum(stage.duration_ms for stage in model._build_startup_sequence("unloaded_idle")) == pytest.approx(150.0)
        assert sum(stage.duration_ms for stage in model._build_startup_sequence("low_rpm_idle")) == pytest.approx(260.0)
    finally:
        model.stop()


def test_resume_latencies_follow_idle_state_depth():
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0)
    try:
        model.reset_caches()
        with model.lock:
            model.power_state = "unloaded_idle"
            model.current_rpm = model.target_rpm
            model.heads_loaded = False
            model.ready_event.clear()
        unloaded = model.submit_physical_access(0, 4096, False, op_kind="data")["startup_ms"]

        model.reset_caches()
        with model.lock:
            model.power_state = "low_rpm_idle"
            model.current_rpm = model.low_rpm_rpm
            model.heads_loaded = False
            model.ready_event.clear()
        low_rpm = model.submit_physical_access(1024, 4096, False, op_kind="data")["startup_ms"]

        model.reset_caches()
        with model.lock:
            model.power_state = "standby"
            model.current_rpm = 0.0
            model.heads_loaded = False
            model.ready_event.clear()
        standby = model.submit_physical_access(2048, 4096, False, op_kind="data")["startup_ms"]

        assert unloaded == pytest.approx(model.unload_to_ready_ms)
        assert low_rpm == pytest.approx(model.low_rpm_to_ready_ms)
        assert standby == pytest.approx(model.standby_to_ready_ms)
        assert unloaded < low_rpm < standby
    finally:
        model.stop()


def test_async_power_on_started_before_first_io_reaches_ready_state():
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0, start_ready=False)
    try:
        assert model.begin_async_startup() is True
        assert model.ready_event.wait(timeout=0.5)
        assert model.power_state == "active"
        assert model.current_rpm == model.target_rpm
    finally:
        model.stop()


def test_access_during_async_power_on_reports_not_ready_polling():
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=1.0,
        start_ready=False,
        power_on_to_ready_ms=70.0,
        standby_to_ready_ms=50.0,
        unload_to_ready_ms=24.0,
        low_rpm_to_ready_ms=32.0,
        spinup_ms=24.0,
        ready_poll_ms=8.0,
    )
    try:
        assert model.begin_async_startup() is True
        result = model.submit_physical_access(0, 4096, False, op_kind="data")
        assert result["startup_origin"] == "power_on"
        assert result["startup_ms"] == 0.0
        assert result["ready_poll_count"] > 0
        assert result["ready_poll_ms"] > 0.0
        assert model.power_state == "active"
    finally:
        model.stop()


def test_idle_transitions_progress_to_low_rpm_and_standby():
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        unload_after_s=0.05,
        low_rpm_after_s=0.10,
        standby_after_s=0.15,
    )
    try:
        with model.lock:
            model.last_access_time = time.monotonic() - 1.0
        time.sleep(0.4)
        with model.lock:
            assert model.power_state == "standby"
            assert model.current_rpm == 0.0
    finally:
        model.stop()


def test_staged_spindown_has_visible_intermediate_state_before_standby():
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=1.0,
        spin_down_ms=40.0,
    )
    try:
        assert model.begin_async_spindown() is True
        time.sleep(0.01)
        with model.lock:
            assert model.power_state == "spinning_down"
            assert model.current_rpm < model.target_rpm
        time.sleep(0.12)
        with model.lock:
            assert model.power_state == "standby"
            assert model.current_rpm == 0.0
            assert model.heads_loaded is False
    finally:
        model.stop()


def test_virtual_hdd_writeback_queue_flushes():
    vhdd = VirtualHDD("backing_storage", latency_scale=0.0)
    try:
        stats = vhdd.access_file("/buffered.bin", 0, 32768, is_write=True)
        assert stats["type"] == "WRITE"
        assert vhdd.writeback_bytes > 0

        vhdd.sync_all()
        assert vhdd.writeback_bytes == 0
    finally:
        vhdd.stop()


def test_virtual_hdd_reset_runtime_state_flushes_dirty_writeback():
    vhdd = VirtualHDD("backing_storage", latency_scale=0.0)
    try:
        vhdd.access_file("/dirty.bin", 0, 32768, is_write=True)
        assert vhdd.writeback_bytes > 0
        vhdd.reset_runtime_state()
        assert vhdd.writeback_bytes == 0
    finally:
        vhdd.stop()


def test_buffered_write_stats_do_not_report_full_cache_hit_after_journal_io():
    vhdd = VirtualHDD("backing_storage", latency_scale=0.0)
    try:
        stats = vhdd.access_file("/journaled.bin", 0, 4096, is_write=True)
        assert stats["type"] == "WRITE"
        assert stats["cache_hit"] is False
    finally:
        vhdd.stop()


def test_virtual_hdd_prepare_overwrite_discards_old_tail():
    vhdd = VirtualHDD("backing_storage", latency_scale=0.0)
    try:
        vhdd.access_file("/overwrite.bin", 0, 32768, is_write=True)
        vhdd.sync_all()
        assert vhdd.fs.files["/overwrite.bin"].size == 32768

        vhdd.prepare_overwrite("/overwrite.bin")
        assert vhdd.fs.files["/overwrite.bin"].size == 0
        assert vhdd.access_file("/overwrite.bin", 16384, 4096, is_write=False)["extents"] == 0
    finally:
        vhdd.stop()


def test_virtual_hdd_materializes_existing_backing_file(tmp_path):
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


def test_virtual_hdd_stop_drains_background_threads():
    vhdd = VirtualHDD("backing_storage", latency_scale=0.0)
    scheduler = OSScheduler(vhdd.model)
    vhdd.set_scheduler(scheduler)
    try:
        vhdd.access_file("/shutdown.bin", 0, 32768, is_write=True)
    finally:
        vhdd.stop()

    assert not vhdd.writeback_thread.is_alive()
    assert not vhdd.model.background_thread.is_alive()
    assert not scheduler.dispatch_thread.is_alive()


def test_audio_engine_stop_is_safe_before_start():
    engine = HDDAudioEngine()
    engine.stop()


def test_audio_engine_seek_centroid_stays_in_lower_band():
    engine = HDDAudioEngine(seed=0)
    engine._update_telemetry(7200.0, seek_trigger=True, seek_dist=700, op_kind="data")
    chunk = engine.render_chunk(4096)

    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    rms = float(np.sqrt(np.mean(chunk**2)))
    peak = float(np.max(np.abs(chunk)))

    assert centroid < 900.0
    assert 0.015 < rms < 0.09
    assert 0.05 < peak < 0.25


def test_scheduler_propagates_model_failures():
    class FailingModel:
        block_bytes = 4096

        def get_estimated_lba(self):
            return 0

        def submit_physical_access(self, *args, **kwargs):
            raise RuntimeError("boom")

    scheduler = OSScheduler(FailingModel())
    try:
        request_id = scheduler.submit_bio(0, 4096, is_write=False)
        with pytest.raises(RuntimeError, match="boom"):
            scheduler.wait_for_completion(request_id)
    finally:
        scheduler.stop()


def test_scheduler_respects_queue_depth_limit():
    class SlowModel:
        block_bytes = 4096

        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def get_estimated_lba(self):
            return 0

        def submit_physical_access(self, *args, **kwargs):
            self.started.set()
            self.release.wait(timeout=1.0)
            return {"total_ms": 0.0, "cache_hit": False, "cyl": 0, "head": 0}

    model = SlowModel()
    scheduler = OSScheduler(model, max_queue_depth=1)
    try:
        first_request = scheduler.submit_bio(0, 4096, is_write=False)
        assert model.started.wait(timeout=1.0)

        holder = {}
        finished_submit = threading.Event()

        def submit_second():
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


def test_random_operation_invariants_hold_under_mixed_workload():
    random.seed(3)
    vhdd = VirtualHDD("backing_storage", latency_scale=0.0)
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
    finally:
        vhdd.stop()


def test_render_scenario_writes_nonempty_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_audio_samples, "SAMPLES_DIR", tmp_path)
    output = render_scenario("test-sample", 0.25, update_sequential_read)
    assert output.exists()

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 44100
        assert wav_file.getnframes() > 0
