from __future__ import annotations

import contextlib
import random
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any, Iterator

from cheroot import wsgi
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from wsgidav.wsgidav_app import WsgiDAVApp

from audio_engine import HDDAudioEngine, HDDAudioEvent
from fs_simulator import FileSystemSimulator
import generate_audio_samples
from generate_audio_samples import render_scenario, update_random_flush, update_sequential_read, update_spinup_idle
from hdd_model import HDDLatencyModel, VirtualHDD
from os_scheduler import OSScheduler
import profile_core
import profile_fragmentation
from profiles import resolve_acoustic_profile, resolve_drive_profile
import smoke
from vfs_provider import HDDProvider


class _NoAuthWsgiDAVApp(WsgiDAVApp):
    def __call__(self, environ: dict[str, Any], start_response: Any) -> Any:
        environ["wsgidav.auth.user_name"] = "anonymous"
        return super().__call__(environ, start_response)


@pytest.fixture
def isolated_backing_dir(tmp_path: Path) -> Path:
    backing = tmp_path / "backing"
    backing.mkdir()
    return backing


@contextlib.contextmanager
def _run_test_server(backing_dir: Path) -> Iterator[tuple[str, HDDProvider]]:
    provider = HDDProvider(str(backing_dir))
    if not provider.vhdd.model.ready_event.wait(timeout=15.0):
        raise TimeoutError("test WebDAV server did not finish async startup")
    provider.vhdd.model.latency_scale = 0.0
    config = {
        "provider_mapping": {"/": provider},
        "http_authenticator": {"enabled": False},
        "middleware_stack": [
            "wsgidav.error_printer.ErrorPrinter",
            "wsgidav.dir_browser._dir_browser.WsgiDavDirBrowser",
            "wsgidav.request_resolver.RequestResolver",
        ],
        "verbose": 0,
    }
    app = _NoAuthWsgiDAVApp(config)
    server = wsgi.Server(("127.0.0.1", 0), app)
    server.prepare()
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.bind_addr[1]}", provider
    finally:
        server.stop()
        thread.join(timeout=2.0)
        provider.vhdd.stop()


def _request(
    base_url: str,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(f"{base_url}{path}", data=body, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)

    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _assert_provider_tree_matches_disk(provider: HDDProvider, backing_dir: Path) -> None:
    backing_root = Path(backing_dir)
    disk_dirs = {"/"}
    disk_files = {}
    for current in backing_root.rglob("*"):
        rel = current.relative_to(backing_root).as_posix()
        virtual_path = "/" + rel if rel else "/"
        if current.is_dir():
            disk_dirs.add(virtual_path)
        elif current.is_file():
            disk_files[virtual_path] = current.stat().st_size

    assert set(provider.vhdd.fs.directories) == disk_dirs
    assert set(provider.vhdd.fs.files) == set(disk_files)
    for path, size in disk_files.items():
        assert provider.vhdd.fs.files[path].size == size

    provider.vhdd.fs.assert_consistent()


def _list_disk_tree(backing_dir: Path) -> tuple[list[str], list[str]]:
    backing_root = Path(backing_dir)
    dirs = ["/"]
    files = []
    for current in backing_root.rglob("*"):
        rel = current.relative_to(backing_root).as_posix()
        virtual_path = "/" + rel if rel else "/"
        if current.is_dir():
            dirs.append(virtual_path)
        elif current.is_file():
            files.append(virtual_path)
    return sorted(dirs), sorted(files)


def _wav_metrics(path: Path) -> tuple[float, float]:
    with wave.open(str(path), "rb") as wav_file:
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16).astype(np.float64) / 32767.0
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    return rms, peak


def test_filesystem_write_emits_metadata_and_data() -> None:
    fs = FileSystemSimulator(total_gb=1)
    operations = fs.write("/demo.bin", 0, 8192)

    kinds = [operation.kind for operation in operations]
    assert "journal" in kinds
    assert "metadata" in kinds
    data_ops = [operation for operation in operations if operation.kind == "data"]
    assert len(data_ops) == 1
    assert data_ops[0].block_count == 2


def test_lookup_reads_directory_and_inode_metadata() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/demo.bin", 0, 4096)

    operations = fs.lookup("/demo.bin")
    assert [operation.kind for operation in operations] == ["metadata", "metadata"]
    assert operations[0].source == "dentry_lookup"
    assert operations[1].source == "inode_lookup"


def test_path_normalization_stays_posix_like_on_windows() -> None:
    fs = FileSystemSimulator(total_gb=1)
    assert fs._normalize_path("foo/bar") == "/foo/bar"
    assert fs._normalize_path("foo\\bar") == "/foo/bar"
    assert fs._normalize_path("/foo/bar") == "/foo/bar"
    assert fs._normalize_path("//foo//bar") == "/foo/bar"
    assert fs._parent_dir("/foo/bar/baz.txt") == "/foo/bar"


def test_delete_returns_metadata_operations() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/demo.bin", 0, 16384)

    operations = fs.delete("/demo.bin")
    kinds = [operation.kind for operation in operations]
    assert kinds.count("journal") == 1
    assert kinds.count("metadata") >= 2
    assert fs.get_fragmentation_score("/demo.bin") == 0


def test_directory_rename_updates_subtree_and_recursive_delete_cleans_it_up() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.create_directory("/docs")
    fs.create_directory("/archive")
    fs.create_empty_file("/docs/note.txt")
    fs.write("/docs/note.txt", 0, 4096)

    dir_move_ops = fs.rename("/docs", "/archive/docs")
    assert "/docs" not in fs.directories
    assert "/archive/docs" in fs.directories
    assert "/archive/docs/note.txt" in fs.files
    assert any(operation.source == "dir_parent_update" for operation in dir_move_ops)

    file_move_ops = fs.rename("/archive/docs/note.txt", "/archive/docs/final.txt")
    assert "/archive/docs/note.txt" not in fs.files
    assert "/archive/docs/final.txt" in fs.files
    assert any(operation.source == "inode_rename" for operation in file_move_ops)

    delete_ops = fs.delete_directory("/archive", recursive=True)
    assert "/archive" not in fs.directories
    assert "/archive/docs/final.txt" not in fs.files
    assert any(operation.source == "dir_teardown" for operation in delete_ops)


def test_create_empty_file_tracks_zero_length_inode_without_data_blocks() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.create_directory("/docs")

    operations = fs.create_empty_file("/docs/empty.txt")

    assert "/docs/empty.txt" in fs.files
    assert fs.files["/docs/empty.txt"].size == 0
    assert fs.read("/docs/empty.txt", 0, 4096) == []
    assert any(operation.source == "inode_create" for operation in operations)


def test_truncate_frees_tail_extents_and_clears_size() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/demo.bin", 0, 16384)
    operations = fs.truncate("/demo.bin", size=0)

    assert fs.files["/demo.bin"].size == 0
    assert fs.read("/demo.bin", 0, 4096) == []
    assert operations[0].source == "truncate_intent"
    assert operations[1].source == "inode_truncate"


def test_sparse_write_does_not_backfill_hole_reads() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/sparse.bin", 8192, 4096)

    assert fs.read("/sparse.bin", 0, 4096) == []
    hole_neighbor = fs.read("/sparse.bin", 8192, 4096)
    assert len(hole_neighbor) == 1
    assert hole_neighbor[0].block_count == 1


def test_reads_do_not_price_blocks_past_eof() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/tail.bin", 0, 1)

    assert len(fs.read("/tail.bin", 0, 4096)) == 1
    assert fs.read("/tail.bin", 1, 4096) == []


def test_filesystem_small_sizes_and_disk_full_are_handled() -> None:
    with pytest.raises(ValueError):
        FileSystemSimulator(total_gb=0.0001)

    fs = FileSystemSimulator(total_gb=0.05)
    with pytest.raises(OSError):
        fs.write("/too-big.bin", 0, 200 * 1024 * 1024)


def test_partial_read_cache_is_not_reported_as_full_hit() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0)
    model.read_cache.append((100, 101, time.monotonic() + 5.0))

    result = model.submit_physical_access(100, 3 * model.block_bytes, False, op_kind="data")

    assert result["cache_hit"] is False
    assert result["partial_hit"] is True
    model.stop()


def test_non_prefix_cache_overlap_is_not_treated_as_partial_hit() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0)
    model.read_cache.append((101, 102, time.monotonic() + 5.0))

    result = model.submit_physical_access(100, 3 * model.block_bytes, False, op_kind="data")

    assert result["cache_hit"] is False
    assert result["partial_hit"] is False
    model.stop()


def test_zone_table_does_not_extend_past_addressable_capacity() -> None:
    model = HDDLatencyModel(addressable_blocks=2048, latency_scale=0.0)
    assert model.zones[-1].end_lba == 2047
    model.stop()


def test_zone_boundary_transfer_latency_uses_per_zone_rates() -> None:
    model = HDDLatencyModel(addressable_blocks=100000, latency_scale=0.0)
    try:
        first_zone = model.zones[0]
        start_lba = first_zone.end_lba - 2
        block_count = 8
        cross_zone_ms = model._transfer_ms_for_span(start_lba, block_count)
        start_only_ms = ((block_count * model.block_bytes) / (1024 * 1024)) / first_zone.transfer_rate_mbps * 1000.0

        assert cross_zone_ms > start_only_ms
    finally:
        model.stop()


def test_drive_profile_changes_rpm_and_command_classes() -> None:
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        drive_profile="archive_5900_internal",
    )
    try:
        assert model.drive_profile.name == "archive_5900_internal"
        assert model.target_rpm == 5900
        assert model.command_overhead_by_kind["flush"] > model.command_overhead_by_kind["data"]
    finally:
        model.stop()


def test_queue_depth_materially_changes_latency() -> None:
    shallow = HDDLatencyModel(
        addressable_blocks=100000,
        latency_scale=0.0,
        drive_profile="enterprise_7200_bare",
    )
    deep = HDDLatencyModel(
        addressable_blocks=100000,
        latency_scale=0.0,
        drive_profile="enterprise_7200_bare",
    )
    try:
        shallow_result = shallow.submit_physical_access(4096, 4096, False, op_kind="data", queue_depth=1)
        deep_result = deep.submit_physical_access(4096, 4096, False, op_kind="data", queue_depth=8)
        assert deep_result["total_ms"] > shallow_result["total_ms"]
    finally:
        shallow.stop()
        deep.stop()


def test_power_on_sequence_reports_startup_and_finishes_ready() -> None:
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


def test_startup_stage_partitioning_respects_configured_budget() -> None:
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


def test_resume_latencies_follow_idle_state_depth() -> None:
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


def test_async_power_on_started_before_first_io_reaches_ready_state() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0, start_ready=False)
    try:
        assert model.begin_async_startup() is True
        assert model.ready_event.wait(timeout=0.5)
        assert model.power_state == "active"
        assert model.current_rpm == model.target_rpm
    finally:
        model.stop()


def test_access_during_async_power_on_reports_not_ready_polling() -> None:
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


def test_idle_transitions_progress_to_low_rpm_and_standby() -> None:
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


def test_staged_spindown_has_visible_intermediate_state_before_standby() -> None:
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


def test_audio_engine_stop_is_safe_before_start() -> None:
    engine = HDDAudioEngine()
    engine.stop()


def test_audio_engine_can_disable_live_output_via_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HDD_AUDIO", "off")
    engine = HDDAudioEngine(seed=0)
    engine.start()
    try:
        assert engine.output_enabled is False
        assert engine.stream is None
        engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=240, op_kind="data")
        chunk = engine.render_chunk(1024)
        assert float(np.max(np.abs(chunk))) > 0.001
    finally:
        engine.stop()


def test_audio_engine_events_are_buffered_until_render() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=700, op_kind="data")

    assert engine.pending_event_count() == 1
    assert engine.synthesizer.rpm == 0.0

    chunk = engine.render_chunk(4096)

    assert engine.pending_event_count() == 0
    assert engine.synthesizer.rpm == 7200.0
    assert float(np.max(np.abs(chunk))) > 0.005


def test_audio_engine_overlapping_events_can_render_in_one_chunk() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=180, op_kind="journal")
    engine.emit_telemetry(7200.0, is_cal=True, op_kind="metadata")
    engine.emit_telemetry(7200.0, is_park=True, op_kind="metadata")

    chunk = engine.render_chunk(4096)

    assert engine.pending_event_count() == 0
    assert len(engine.synthesizer.pending_impulses) == 0
    assert float(np.sqrt(np.mean(chunk**2))) > 0.003


def test_audio_synth_can_schedule_future_impulse_across_chunks() -> None:
    engine = HDDAudioEngine(seed=0)
    event = HDDAudioEvent(
        rpm=7200.0,
        queue_depth=1,
        op_kind="data",
        impulse="seek",
        seek_distance=240,
    )

    first = engine.synthesizer.render_chunk(512, scheduled_events=[(event, 700)])
    second = engine.synthesizer.render_chunk(512)

    assert len(engine.synthesizer.pending_impulses) == 0
    assert float(np.max(np.abs(first))) < 0.02
    assert float(np.max(np.abs(second - first))) > 0.007


def test_audio_engine_chunk_edge_seek_carries_into_next_chunk() -> None:
    edge_event = HDDAudioEvent(
        rpm=7200.0,
        queue_depth=1,
        op_kind="data",
        impulse="seek",
        seek_distance=240,
    )

    idle_engine = HDDAudioEngine(seed=0)
    idle_chunk = idle_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(HDDAudioEvent(rpm=7200.0, queue_depth=1, op_kind="data"), 0)],
    )

    engine = HDDAudioEngine(seed=0)
    first = engine.synthesizer.render_chunk(1024, scheduled_events=[(edge_event, 1022)])
    first_delta = first - idle_chunk
    assert len(engine.synthesizer.pending_impulses) > 0
    assert float(np.max(np.abs(first_delta))) < 1e-6

    second = engine.synthesizer.render_chunk(1024)
    second_delta = second - idle_chunk
    assert len(engine.synthesizer.pending_impulses) == 0
    assert float(np.max(np.abs(second_delta))) > 0.005
    assert float(np.sqrt(np.mean(second_delta**2))) > 0.002


def test_audio_engine_event_bus_drains_fifo() -> None:
    engine = HDDAudioEngine(seed=0)
    first = HDDAudioEvent(rpm=5400.0, queue_depth=1, op_kind="metadata", impulse="calibration")
    second = HDDAudioEvent(rpm=7200.0, queue_depth=4, op_kind="data", impulse="seek", seek_distance=500)

    engine.emit_event(first)
    engine.emit_event(second)
    drained = engine.events.drain()

    assert drained == [first, second]


def test_audio_engine_event_bus_is_bounded_and_keeps_recent_events() -> None:
    engine = HDDAudioEngine(seed=0, max_pending_events=3)
    for idx in range(5):
        engine.emit_event(HDDAudioEvent(rpm=5400.0 + idx, queue_depth=idx + 1, op_kind="metadata"))

    drained = engine.events.drain()

    assert [event.queue_depth for event in drained] == [3, 4, 5]
    assert engine.dropped_event_count() == 2


def test_audio_engine_flush_journal_and_generic_seek_classes_have_distinct_pulse_shapes() -> None:
    synth = HDDAudioEngine(seed=0).synthesizer

    metadata_pulses = synth._make_impulse_pulses("seek", 0, 220.0, "metadata", False)
    journal_pulses = synth._make_impulse_pulses("seek", 0, 220.0, "journal", False)
    data_pulses = synth._make_impulse_pulses("seek", 0, 220.0, "data", False)
    flush_pulses = synth._make_impulse_pulses("seek", 0, 260.0, "flush", True)

    def pulse_energy(pulses: list[Any]) -> float:
        return float(sum(abs(pulse.amplitude) for pulse in pulses))

    assert len(metadata_pulses) == 3
    assert len(journal_pulses) == 3
    assert len(data_pulses) == 3
    assert len(flush_pulses) == 4
    assert pulse_energy(flush_pulses) > pulse_energy(data_pulses) > pulse_energy(journal_pulses) > pulse_energy(metadata_pulses)


def test_audio_engine_seek_profile_keeps_fixed_actuator_mode_band() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=700, op_kind="data")
    chunk = engine.render_chunk(4096)

    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
    resonance_band = (freqs >= 900.0) & (freqs <= 2200.0)
    resonance_freq = float(freqs[np.argmax(spectrum[resonance_band]) + np.where(resonance_band)[0][0]])

    assert 900.0 <= resonance_freq <= 1250.0


def test_audio_engine_sequential_profile_keeps_7200rpm_spindle_fundamental() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, is_seq=True, queue_depth=2, op_kind="data")
    chunk = engine.render_chunk(16384)

    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
    low_band = (freqs >= 110.0) & (freqs <= 130.0)
    peak_freq = float(freqs[np.argmax(spectrum[low_band]) + np.where(low_band)[0][0]])

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spectrum[mask]))

    platter_band_ratio = band(550.0, 1100.0) / max(band(0.0, 300.0), 1e-12)
    upper_mid_ratio = band(1000.0, 3000.0) / max(band(0.0, 300.0), 1e-12)

    assert 118.0 <= peak_freq <= 122.5
    assert platter_band_ratio > 0.01
    assert upper_mid_ratio > 0.025


def test_audio_engine_archive_profile_shifts_spindle_fundamental_lower() -> None:
    engine = HDDAudioEngine(seed=0, drive_profile="archive_5900_internal")
    rpm = float(engine.synthesizer.drive_profile.rpm)
    engine.emit_telemetry(rpm, is_seq=True, queue_depth=2, op_kind="data")
    chunk = engine.render_chunk(16384)

    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
    low_band = (freqs >= 90.0) & (freqs <= 110.0)
    peak_freq = float(freqs[np.argmax(spectrum[low_band]) + np.where(low_band)[0][0]])

    assert 97.0 <= peak_freq <= 101.0


def test_audio_engine_seek_resonance_stays_stable_across_seek_distance() -> None:
    def resonance_peak(seek_distance: int) -> float:
        engine = HDDAudioEngine(seed=0)
        engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=seek_distance, op_kind="data")
        chunk = engine.render_chunk(4096)
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
        resonance_band = (freqs >= 900.0) & (freqs <= 2200.0)
        return float(freqs[np.argmax(spectrum[resonance_band]) + np.where(resonance_band)[0][0]])

    short_seek_peak = resonance_peak(40)
    long_seek_peak = resonance_peak(700)

    assert abs(short_seek_peak - long_seek_peak) <= 350.0


def test_audio_engine_multi_event_render_has_higher_delta_complexity_than_single_event() -> None:
    idle_engine = HDDAudioEngine(seed=0)
    idle_chunk = idle_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(HDDAudioEvent(rpm=7200.0, queue_depth=1, op_kind="data"), 0)],
    )

    single_engine = HDDAudioEngine(seed=0)
    single_chunk = single_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(HDDAudioEvent(rpm=7200.0, queue_depth=1, op_kind="data", impulse="seek", seek_distance=220), 0)],
    )
    single_delta = single_chunk - idle_chunk

    multi_engine = HDDAudioEngine(seed=0)
    multi_chunk = multi_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[
            (HDDAudioEvent(rpm=7200.0, queue_depth=1, op_kind="data", impulse="seek", seek_distance=220), 0),
            (HDDAudioEvent(rpm=7200.0, queue_depth=1, op_kind="journal", impulse="seek", seek_distance=80), 50),
            (HDDAudioEvent(rpm=7200.0, queue_depth=1, op_kind="metadata", impulse="calibration"), 90),
            (HDDAudioEvent(rpm=7200.0, queue_depth=1, op_kind="metadata", impulse="park"), 140),
        ],
    )
    multi_delta = multi_chunk - idle_chunk

    single_complexity = float(np.sum(np.abs(np.diff(single_delta))))
    multi_complexity = float(np.sum(np.abs(np.diff(multi_delta))))

    assert multi_complexity > single_complexity


def test_audio_engine_render_regression_for_startup_idle_park_and_flush_envelopes() -> None:
    idle_engine = HDDAudioEngine(seed=0)
    idle_chunk = idle_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(HDDAudioEvent(rpm=7200.0, queue_depth=1, op_kind="data"), 0)],
    )

    startup_engine = HDDAudioEngine(seed=0)
    startup_chunk = startup_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(HDDAudioEvent(rpm=1200.0, queue_depth=1, op_kind="metadata", is_spinup=True), 0)],
    )
    park_engine = HDDAudioEngine(seed=0)
    park_chunk = park_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(HDDAudioEvent(rpm=7200.0, queue_depth=1, op_kind="metadata", impulse="park"), 0)],
    )
    flush_engine = HDDAudioEngine(seed=0)
    flush_chunk = flush_engine.synthesizer.render_chunk(
        1024,
        scheduled_events=[(HDDAudioEvent(rpm=7200.0, queue_depth=3, op_kind="flush", impulse="seek", seek_distance=260, is_flush=True), 0)],
    )

    park_delta = park_chunk - idle_chunk
    flush_delta = flush_chunk - idle_chunk

    startup_rms = float(np.sqrt(np.mean(startup_chunk**2)))
    idle_rms = float(np.sqrt(np.mean(idle_chunk**2)))
    park_delta_rms = float(np.sqrt(np.mean(park_delta**2)))
    flush_delta_rms = float(np.sqrt(np.mean(flush_delta**2)))

    assert idle_rms > startup_rms * 40.0
    assert flush_delta_rms > park_delta_rms > 0.0


def test_audio_engine_acoustic_profiles_change_loudness_and_brightness() -> None:
    def render_metrics(acoustic_profile: str) -> tuple[float, float]:
        engine = HDDAudioEngine(
            seed=0,
            drive_profile="desktop_7200_internal",
            acoustic_profile=acoustic_profile,
        )
        engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=320, op_kind="data")
        chunk = engine.render_chunk(4096)
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / engine.fs)
        centroid = float(np.sum(freqs * spectrum) / max(np.sum(spectrum), 1e-12))
        rms = float(np.sqrt(np.mean(chunk**2)))
        return centroid, rms

    bare_centroid, bare_rms = render_metrics("bare_drive_lab")
    case_centroid, case_rms = render_metrics("mounted_in_case")
    external_centroid, external_rms = render_metrics("external_enclosure")

    assert bare_centroid > case_centroid > external_centroid
    assert bare_rms > case_rms > external_rms


def test_audio_engine_overlapping_seek_flush_park_and_calibration_events_render_together() -> None:
    engine = HDDAudioEngine(seed=0)
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=120, op_kind="journal")
    engine.emit_telemetry(7200.0, seek_trigger=True, seek_dist=260, op_kind="flush", is_flush=True)
    engine.emit_telemetry(7200.0, is_cal=True, op_kind="metadata")
    engine.emit_telemetry(7200.0, is_park=True, op_kind="metadata")

    chunk = engine.render_chunk(4096)

    assert engine.pending_event_count() == 0
    assert len(engine.synthesizer.pending_impulses) == 0
    assert float(np.sqrt(np.mean(chunk**2))) > 0.003


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


def test_filesystem_tree_invariants_hold_under_random_directory_workload() -> None:
    random.seed(7)
    fs = FileSystemSimulator(total_gb=1)
    next_dir_id = 0
    next_file_id = 0

    def fresh_dir_name() -> str:
        nonlocal next_dir_id
        next_dir_id += 1
        return f"dir-{next_dir_id}"

    def fresh_file_name() -> str:
        nonlocal next_file_id
        next_file_id += 1
        return f"file-{next_file_id}.bin"

    for _ in range(150):
        dir_paths = sorted(fs.directories)
        file_paths = sorted(fs.files)
        operation = random.choice(
            ["mkdir", "write", "rename_file", "rename_dir", "delete_file", "delete_dir", "lookup", "list"]
        )

        if operation == "mkdir":
            parent = random.choice(dir_paths)
            fs.create_directory(f"{parent}/{fresh_dir_name()}")
        elif operation == "write":
            parent = random.choice(dir_paths)
            path = f"{parent}/{fresh_file_name()}"
            fs.write(path, random.randint(0, 4) * 4096, random.randint(1, 3) * 4096)
        elif operation == "rename_file" and file_paths:
            source = random.choice(file_paths)
            dest_parent = random.choice(dir_paths)
            fs.rename(source, f"{dest_parent}/{fresh_file_name()}")
        elif operation == "rename_dir" and len(dir_paths) > 1:
            source = random.choice([path for path in dir_paths if path != "/"])
            candidate_parents = [
                path for path in dir_paths if path != source and not path.startswith(f"{source}/")
            ]
            if candidate_parents:
                dest_parent = random.choice(candidate_parents)
                fs.rename(source, f"{dest_parent}/{fresh_dir_name()}")
        elif operation == "delete_file" and file_paths:
            fs.delete(random.choice(file_paths))
        elif operation == "delete_dir" and len(dir_paths) > 1:
            target = random.choice([path for path in dir_paths if path != "/"])
            fs.delete_directory(target, recursive=True)
        elif operation == "lookup":
            target = random.choice(dir_paths + file_paths or ["/"])
            fs.lookup(target)
        else:
            fs.list_directory(random.choice(dir_paths))

        fs.assert_consistent()


def test_render_scenario_writes_nonempty_wav(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_audio_samples, "SAMPLES_DIR", tmp_path)
    output = render_scenario("test-sample", 0.25, update_sequential_read)
    assert output.exists()

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 44100
        assert wav_file.getnframes() > 0


def test_rendered_sample_scenarios_have_normalized_loudness(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_audio_samples, "SAMPLES_DIR", tmp_path)

    spinup = render_scenario("spinup", 2.0, update_spinup_idle, seed=7)
    sequential = render_scenario("sequential", 1.0, update_sequential_read, seed=11)
    random_flush = render_scenario("random", 1.0, update_random_flush, seed=13)

    spinup_rms, spinup_peak = _wav_metrics(spinup)
    sequential_rms, sequential_peak = _wav_metrics(sequential)
    random_rms, random_peak = _wav_metrics(random_flush)

    assert 1e-5 < spinup_rms < 1e-3
    assert 0.002 < sequential_rms < 0.01
    assert 0.002 < random_rms < 0.01
    assert abs(sequential_rms - random_rms) / sequential_rms < 0.1
    assert 0.0 < spinup_peak < 0.01
    assert 0.004 < sequential_peak < 0.02
    assert 0.004 < random_peak < 0.02


def test_smoke_main_boot_random_port_with_audio_disabled() -> None:
    smoke.run_main_boot_smoke(exercise_cli=False)


def test_smoke_cli_probe_works_when_curl_is_available(tmp_path: Path) -> None:
    if shutil.which("curl.exe") is None and shutil.which("curl") is None:
        pytest.skip("curl not available")

    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, _provider):
        assert smoke._run_cli_probe(base_url) is True


def test_profile_core_expectations_hold() -> None:
    metrics = profile_core.collect_core_metrics()
    profile_core.assert_core_expectations(metrics)
    assert metrics["cold_start_startup_ms"] > metrics["ready_startup_ms"]
    assert metrics["mixed_churn_total_ms"] > metrics["metadata_churn_total_ms"]


def test_profile_fragmentation_expectations_hold() -> None:
    metrics = profile_fragmentation.collect_fragmentation_metrics()
    profile_fragmentation.assert_fragmentation_expectations(metrics)
    assert metrics["fragmented_read_extents"] > metrics["contiguous_read_extents"]
    assert metrics["fragmented_read_ms"] > metrics["contiguous_read_ms"]


def test_webdav_end_to_end_directory_lifecycle(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "MKCOL", "/docs")
        assert status == 201

        status, _, _ = _request(base_url, "PUT", "/docs/empty.txt", b"")
        assert status in (200, 201, 204)

        status, _, _ = _request(base_url, "PUT", "/docs/data.txt", b"hello world")
        assert status in (200, 201, 204)

        status, body, _ = _request(base_url, "GET", "/docs/data.txt")
        assert status == 200
        assert body == b"hello world"
        assert provider.vhdd.fs.files["/docs/data.txt"].size == len(b"hello world")

        status, body, _ = _request(base_url, "PROPFIND", "/docs", headers={"Depth": "1"})
        assert status == 207
        assert b"data.txt" in body
        assert b"empty.txt" in body
        assert "/docs" in provider.vhdd.fs.directories
        assert "/docs/empty.txt" in provider.vhdd.fs.files

        status, _, _ = _request(
            base_url,
            "MOVE",
            "/docs",
            headers={"Destination": f"{base_url}/archive", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert "/archive" in provider.vhdd.fs.directories
        assert "/archive/data.txt" in provider.vhdd.fs.files

        status, body, _ = _request(base_url, "GET", "/archive/data.txt")
        assert status == 200
        assert body == b"hello world"

        status, _, _ = _request(base_url, "DELETE", "/archive")
        assert status == 204
        assert "/archive" not in provider.vhdd.fs.directories
        assert "/archive/data.txt" not in provider.vhdd.fs.files
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_copy_file_and_directory_tree(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "MKCOL", "/docs")
        assert status == 201

        status, _, _ = _request(base_url, "MKCOL", "/docs/subdir")
        assert status == 201

        payload = b"a" * (256 * 1024)
        status, _, _ = _request(base_url, "PUT", "/docs/subdir/source.bin", payload)
        assert status in (200, 201, 204)

        status, _, _ = _request(
            base_url,
            "COPY",
            "/docs/subdir/source.bin",
            headers={"Destination": f"{base_url}/docs/subdir/source-copy.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert "/docs/subdir/source-copy.bin" in provider.vhdd.fs.files
        assert provider.vhdd.fs.files["/docs/subdir/source-copy.bin"].size == len(payload)

        status, body, _ = _request(base_url, "GET", "/docs/subdir/source-copy.bin")
        assert status == 200
        assert body == payload

        status, _, _ = _request(
            base_url,
            "COPY",
            "/docs",
            headers={"Destination": f"{base_url}/docs-copy", "Depth": "infinity", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert "/docs-copy" in provider.vhdd.fs.directories
        assert "/docs-copy/subdir" in provider.vhdd.fs.directories
        assert "/docs-copy/subdir/source.bin" in provider.vhdd.fs.files
        assert "/docs-copy/subdir/source-copy.bin" in provider.vhdd.fs.files

        status, body, _ = _request(base_url, "GET", "/docs-copy/subdir/source-copy.bin")
        assert status == 200
        assert body == payload
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_overwrite_copy_updates_simulated_tree(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        _request(base_url, "MKCOL", "/src")
        _request(base_url, "MKCOL", "/dst")
        _request(base_url, "PUT", "/src/data.bin", b"s" * 8192)
        _request(base_url, "PUT", "/dst/data.bin", b"d" * 4096)

        status, _, _ = _request(
            base_url,
            "COPY",
            "/src/data.bin",
            headers={"Destination": f"{base_url}/dst/data.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)

        status, body, _ = _request(base_url, "GET", "/dst/data.bin")
        assert status == 200
        assert body == b"s" * 8192
        assert provider.vhdd.fs.files["/dst/data.bin"].size == 8192
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_move_overwrite_updates_simulated_tree(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        _request(base_url, "MKCOL", "/src")
        _request(base_url, "MKCOL", "/dst")
        _request(base_url, "PUT", "/src/data.bin", b"m" * 12288)
        _request(base_url, "PUT", "/dst/data.bin", b"z" * 2048)

        status, _, _ = _request(
            base_url,
            "MOVE",
            "/src/data.bin",
            headers={"Destination": f"{base_url}/dst/data.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)

        status, body, _ = _request(base_url, "GET", "/dst/data.bin")
        assert status == 200
        assert body == b"m" * 12288
        assert "/src/data.bin" not in provider.vhdd.fs.files
        assert provider.vhdd.fs.files["/dst/data.bin"].size == 12288
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_directory_copy_over_existing_tree_removes_stale_entries(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        _request(base_url, "MKCOL", "/src")
        _request(base_url, "MKCOL", "/src/sub")
        _request(base_url, "PUT", "/src/sub/fresh.bin", b"fresh")

        _request(base_url, "MKCOL", "/dst")
        _request(base_url, "MKCOL", "/dst/sub")
        _request(base_url, "PUT", "/dst/sub/stale.bin", b"stale")
        _request(base_url, "PUT", "/dst/old.bin", b"old")

        status, _, _ = _request(
            base_url,
            "COPY",
            "/src",
            headers={"Destination": f"{base_url}/dst", "Depth": "infinity", "Overwrite": "T"},
        )
        assert status in (201, 204)

        assert "/dst/sub/fresh.bin" in provider.vhdd.fs.files
        assert "/dst/sub/stale.bin" not in provider.vhdd.fs.files
        assert "/dst/old.bin" not in provider.vhdd.fs.files

        status, body, _ = _request(base_url, "GET", "/dst/sub/fresh.bin")
        assert status == 200
        assert body == b"fresh"
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_randomized_tree_operations_keep_simulator_in_sync(tmp_path: Path) -> None:
    random.seed(11)
    backing = tmp_path / "backing"
    backing.mkdir()

    dir_counter = 0
    file_counter = 0

    def next_dir_name() -> str:
        nonlocal dir_counter
        dir_counter += 1
        return f"dir-{dir_counter}"

    def next_file_name() -> str:
        nonlocal file_counter
        file_counter += 1
        return f"file-{file_counter}.bin"

    with _run_test_server(backing) as (base_url, provider):
        for _ in range(25):
            dirs, files = _list_disk_tree(backing)
            op = random.choice(
                ["mkdir", "put", "copy_file", "copy_dir", "move_file", "move_dir", "delete_file", "delete_dir"]
            )

            if op == "mkdir":
                parent = random.choice(dirs)
                target = (parent.rstrip("/") + "/" + next_dir_name()) if parent != "/" else "/" + next_dir_name()
                status, _, _ = _request(base_url, "MKCOL", target)
                assert status == 201
            elif op == "put":
                parent = random.choice(dirs)
                target = (parent.rstrip("/") + "/" + next_file_name()) if parent != "/" else "/" + next_file_name()
                payload = bytes([65 + (file_counter % 26)]) * random.choice([1024, 4096, 12288])
                status, _, _ = _request(base_url, "PUT", target, payload)
                assert status in (200, 201, 204)
            elif op == "copy_file" and files:
                source = random.choice(files)
                dest_parent = random.choice(dirs)
                dest = (dest_parent.rstrip("/") + "/" + next_file_name()) if dest_parent != "/" else "/" + next_file_name()
                status, _, _ = _request(
                    base_url,
                    "COPY",
                    source,
                    headers={"Destination": f"{base_url}{dest}", "Overwrite": "T"},
                )
                assert status in (201, 204)
            elif op == "copy_dir" and len(dirs) > 1:
                source = random.choice([path for path in dirs if path != "/"])
                dest_parent_choices = [
                    path for path in dirs if path != source and not path.startswith(f"{source}/")
                ]
                if not dest_parent_choices:
                    continue
                dest_parent = random.choice(dest_parent_choices)
                dest = (dest_parent.rstrip("/") + "/" + next_dir_name()) if dest_parent != "/" else "/" + next_dir_name()
                status, _, _ = _request(
                    base_url,
                    "COPY",
                    source,
                    headers={"Destination": f"{base_url}{dest}", "Depth": "infinity", "Overwrite": "T"},
                )
                assert status in (201, 204)
            elif op == "move_file" and files:
                source = random.choice(files)
                dest_parent = random.choice(dirs)
                dest = (dest_parent.rstrip("/") + "/" + next_file_name()) if dest_parent != "/" else "/" + next_file_name()
                status, _, _ = _request(
                    base_url,
                    "MOVE",
                    source,
                    headers={"Destination": f"{base_url}{dest}", "Overwrite": "T"},
                )
                assert status in (201, 204)
            elif op == "move_dir" and len(dirs) > 1:
                source = random.choice([path for path in dirs if path != "/"])
                dest_parent_choices = [
                    path for path in dirs if path != source and not path.startswith(f"{source}/")
                ]
                if not dest_parent_choices:
                    continue
                dest_parent = random.choice(dest_parent_choices)
                dest = (dest_parent.rstrip("/") + "/" + next_dir_name()) if dest_parent != "/" else "/" + next_dir_name()
                status, _, _ = _request(
                    base_url,
                    "MOVE",
                    source,
                    headers={"Destination": f"{base_url}{dest}", "Overwrite": "T"},
                )
                assert status in (201, 204)
            elif op == "delete_file" and files:
                status, _, _ = _request(base_url, "DELETE", random.choice(files))
                assert status == 204
            elif op == "delete_dir" and len(dirs) > 1:
                status, _, _ = _request(base_url, "DELETE", random.choice([path for path in dirs if path != "/"]))
                assert status == 204
            else:
                continue

            _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_lazy_restart_materializes_existing_tree_consistently(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    (backing / "docs" / "deep").mkdir(parents=True)
    (backing / "docs" / "deep" / "alpha.bin").write_bytes(b"a" * 4096)
    (backing / "docs" / "beta.bin").write_bytes(b"beta")

    with _run_test_server(backing) as (base_url, provider):
        status, body, _ = _request(base_url, "GET", "/docs/deep/alpha.bin")
        assert status == 200
        assert body == b"a" * 4096
        assert "/docs/deep/alpha.bin" in provider.vhdd.fs.files

        status, body, _ = _request(base_url, "PROPFIND", "/docs", headers={"Depth": "infinity"})
        assert status == 207
        assert b"alpha.bin" in body
        assert b"beta.bin" in body
        _assert_provider_tree_matches_disk(provider, backing)

        status, _, _ = _request(base_url, "DELETE", "/docs")
        assert status == 204
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_range_get_keeps_offsets_and_lengths_consistent(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    payload = bytes(range(256)) * 32

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "PUT", "/range.bin", payload)
        assert status in (200, 201, 204)

        status, body, headers = _request(
            base_url,
            "GET",
            "/range.bin",
            headers={"Range": "bytes=1024-2047"},
        )
        assert status == 206
        assert body == payload[1024:2048]
        assert headers.get("Content-Range", "").startswith("bytes 1024-2047/")
        assert provider.vhdd.fs.files["/range.bin"].size == len(payload)
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_handles_spaces_quotes_and_punctuation_in_names(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    directory = "/Project Files"
    filename = "/Project Files/report 'final' ! test (v1),done.txt"
    encoded_directory = urllib.parse.quote(directory)
    encoded_filename = urllib.parse.quote(filename)

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "MKCOL", encoded_directory)
        assert status == 201

        payload = b"odd names still work"
        status, _, _ = _request(base_url, "PUT", encoded_filename, payload)
        assert status in (200, 201, 204)

        status, body, _ = _request(base_url, "GET", encoded_filename)
        assert status == 200
        assert body == payload

        status, body, _ = _request(base_url, "PROPFIND", encoded_directory, headers={"Depth": "1"})
        assert status == 207
        assert b"report 'final' ! test (v1),done.txt" in body
        assert filename in provider.vhdd.fs.files
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_zero_byte_file_copy_move_delete_paths(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "MKCOL", "/docs")
        assert status == 201

        status, _, _ = _request(base_url, "PUT", "/docs/empty.bin", b"")
        assert status in (200, 201, 204)
        assert provider.vhdd.fs.files["/docs/empty.bin"].size == 0

        status, _, _ = _request(
            base_url,
            "COPY",
            "/docs/empty.bin",
            headers={"Destination": f"{base_url}/docs/empty-copy.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert provider.vhdd.fs.files["/docs/empty-copy.bin"].size == 0

        status, _, _ = _request(
            base_url,
            "MOVE",
            "/docs/empty-copy.bin",
            headers={"Destination": f"{base_url}/docs/empty-moved.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert "/docs/empty-copy.bin" not in provider.vhdd.fs.files
        assert provider.vhdd.fs.files["/docs/empty-moved.bin"].size == 0

        status, _, _ = _request(base_url, "DELETE", "/docs/empty.bin")
        assert status == 204
        status, _, _ = _request(base_url, "DELETE", "/docs/empty-moved.bin")
        assert status == 204
        assert "/docs/empty.bin" not in provider.vhdd.fs.files
        assert "/docs/empty-moved.bin" not in provider.vhdd.fs.files
        _assert_provider_tree_matches_disk(provider, backing)
