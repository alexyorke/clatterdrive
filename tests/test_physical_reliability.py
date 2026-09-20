from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from clatterdrive.audio import physics
from clatterdrive.block_frontend import LatencyBlockDevice, SparseMemoryBlockStore
from clatterdrive.fs import FileSystemSimulator
from clatterdrive.fs import persistence
from clatterdrive.hdd import HDDLatencyModel


@pytest.mark.parametrize("op_kind", ["data", "writeback", "metadata"])
def test_healthy_io_never_produces_media_contact(op_kind: str) -> None:
    forces = physics.head_media_event_forces(
        op_kind=op_kind, directory_activity=1.0, fragmentation_activity=1.0,
        repetition_pressure=1.0, repetition_variant=0.8,
    )
    assert forces.contact == 0.0
    assert forces.wedge > 0.0
    assert physics.park_stop_contact_force() > 0.0


def test_sequential_track_spacing_tracks_rotation_and_duty() -> None:
    assert physics.sequential_track_interval(7200, 1.0) == pytest.approx(1 / 120)
    assert physics.sequential_track_interval(5400, 1.0) == pytest.approx(1 / 90)
    assert physics.sequential_track_interval(7200, 0.5) == pytest.approx(1 / 60)
    assert physics.sequential_track_interval(0, 1.0) == float("inf")


def test_failed_state_flush_preserves_previous_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    fs = FileSystemSimulator(total_gb=0.1, state_path=path)
    fs.create_empty_file("/original")
    fs.persist()
    previous = path.read_bytes()
    fs.create_empty_file("/new")

    def fail_flush(_fd: int) -> None:
        raise OSError("simulated flush failure")

    with monkeypatch.context() as patch:
        patch.setattr(persistence.os, "fsync", fail_flush)
        with pytest.raises(OSError, match="flush failure"):
            fs.persist()
    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []
    fs.persist()
    restored = FileSystemSimulator(total_gb=0.1, state_path=path)
    assert "/new" in restored.files


def test_flush_waits_for_backing_write_and_close_rejects_io() -> None:
    class BlockingStore(SparseMemoryBlockStore):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()
            self.flushed = threading.Event()

        def write_blocks(self, lba: int, data: bytes, block_size: int) -> None:
            self.started.set()
            assert self.release.wait(5), "test did not release backing write"
            super().write_blocks(lba, data, block_size)

        def flush(self) -> None:
            self.flushed.set()

    model = HDDLatencyModel(addressable_blocks=128, latency_scale=0.0, enable_background_scan=False)
    store = BlockingStore()
    device = LatencyBlockDevice(model, store)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            write = pool.submit(device.write_blocks, 0, b"x" * model.block_bytes)
            try:
                assert store.started.wait(2)
                flush_started = threading.Event()

                def run_flush() -> None:
                    flush_started.set()
                    device.flush()

                flush = pool.submit(run_flush)
                assert flush_started.wait(2)
                assert not store.flushed.wait(0.1)
            finally:
                store.release.set()
            write.result(timeout=2)
            flush.result(timeout=2)
        assert device.read_blocks(0, 1)[0] == b"x" * model.block_bytes
        device.close()
        with pytest.raises(RuntimeError, match="closed"):
            device.discard_blocks(0, 1)
        with pytest.raises(RuntimeError, match="closed"):
            device.read_blocks(0, 1)
    finally:
        store.release.set()
        device.close()
        model.stop()


def test_block_device_rejects_short_backing_reads() -> None:
    class ShortStore(SparseMemoryBlockStore):
        def read_blocks(self, lba: int, block_count: int, block_size: int) -> bytes:
            return b""

    model = HDDLatencyModel(addressable_blocks=128, latency_scale=0.0, enable_background_scan=False)
    device = LatencyBlockDevice(model, ShortStore())
    try:
        with pytest.raises(OSError, match="short block read"):
            device.read_blocks(0, 1)
    finally:
        device.close()
        model.stop()
