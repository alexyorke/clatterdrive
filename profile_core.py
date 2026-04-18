from __future__ import annotations

import random
from pathlib import Path
from typing import Final

from hdd_model import VirtualHDD
from os_scheduler import OSScheduler
from runtime_paths import workspace_tempdir


SEQUENTIAL_OPS: Final[int] = 100
RANDOM_READS: Final[int] = 20
CONTENDED_READS: Final[int] = 50
METADATA_ROUNDS: Final[int] = 20
COPY_FILE_COUNT: Final[int] = 4
MIXED_ROUNDS: Final[int] = 12


def _stats_average(total_ms: float, count: int) -> float:
    return total_ms / max(count, 1)


def _profile_sequential(vhdd: VirtualHDD) -> tuple[float, float, float]:
    write_total_ms = 0.0
    for index in range(SEQUENTIAL_OPS):
        write_total_ms += vhdd.access_file(
            "/profile_seq.bin",
            index * 64 * 1024,
            64 * 1024,
            is_write=True,
        )["total_ms"]
    flush_ms = vhdd.sync_all()

    vhdd.reset_runtime_state()
    read_total_ms = 0.0
    for index in range(SEQUENTIAL_OPS):
        read_total_ms += vhdd.access_file(
            "/profile_seq.bin",
            index * 64 * 1024,
            64 * 1024,
            is_write=False,
        )["total_ms"]
    return (
        _stats_average(write_total_ms, SEQUENTIAL_OPS),
        flush_ms,
        _stats_average(read_total_ms, SEQUENTIAL_OPS),
    )


def _profile_random_reads(scheduler: OSScheduler, addressable_blocks: int) -> tuple[float, float]:
    total_ms = 0.0
    for _ in range(RANDOM_READS):
        lba = random.randint(0, addressable_blocks - 1)
        request_id = scheduler.submit_bio(lba, 4096, is_write=False, op_kind="data")
        total_ms += scheduler.wait_for_completion(request_id)["total_ms"]

    contended_total_ms = 0.0
    request_ids = [
        scheduler.submit_bio(
            random.randint(0, addressable_blocks - 1),
            4096,
            is_write=False,
            op_kind="data",
        )
        for _ in range(CONTENDED_READS)
    ]
    for request_id in request_ids:
        contended_total_ms += scheduler.wait_for_completion(request_id)["total_ms"]
    return _stats_average(total_ms, RANDOM_READS), _stats_average(contended_total_ms, CONTENDED_READS)


def _profile_metadata_churn(vhdd: VirtualHDD) -> float:
    total_ms = 0.0
    vhdd.create_directory("/meta")
    for index in range(METADATA_ROUNDS):
        path = f"/meta/file-{index}.txt"
        total_ms += vhdd.create_empty_file(path)["total_ms"]
        total_ms += vhdd.lookup_path(path)["total_ms"]
        total_ms += vhdd.list_directory("/meta")["total_ms"]
        total_ms += vhdd.delete_path(path)["total_ms"]
    return total_ms


def _profile_copy_heavy(vhdd: VirtualHDD) -> float:
    total_ms = 0.0
    vhdd.create_directory("/copy")
    for index in range(COPY_FILE_COUNT):
        source_path = f"/copy/src-{index}.bin"
        dest_path = f"/copy/dst-{index}.bin"
        vhdd.access_file(source_path, 0, 256 * 1024, is_write=True)
        vhdd.sync_all()
        total_ms += vhdd.copy_file(source_path, dest_path)["total_ms"]
    return total_ms


def _profile_mixed_churn(vhdd: VirtualHDD) -> float:
    total_ms = 0.0
    vhdd.create_directory("/mixed")
    for index in range(MIXED_ROUNDS):
        dir_path = f"/mixed/batch-{index}"
        file_path = f"{dir_path}/payload.bin"
        moved_path = f"{dir_path}/payload-moved.bin"
        total_ms += vhdd.create_directory(dir_path)["total_ms"]
        total_ms += vhdd.access_file(file_path, 0, 64 * 1024, is_write=True)["total_ms"]
        total_ms += vhdd.access_file(file_path, 0, 16 * 1024, is_write=False)["total_ms"]
        total_ms += vhdd.rename_path(file_path, moved_path)["total_ms"]
        total_ms += vhdd.lookup_path(moved_path)["total_ms"]
        total_ms += vhdd.list_directory(dir_path)["total_ms"]
        total_ms += vhdd.delete_directory(dir_path)["total_ms"]
    vhdd.sync_all()
    return total_ms


def _profile_cold_vs_ready(backing_dir: Path) -> tuple[float, float, float, float]:
    backing_file = backing_dir / "cold.bin"
    backing_file.write_bytes(b"x" * (64 * 1024))

    cold = VirtualHDD(str(backing_dir), latency_scale=0.0, cold_start=True)
    ready = VirtualHDD(str(backing_dir), latency_scale=0.0)
    try:
        cold_stats = cold.access_file("/cold.bin", 0, 64 * 1024, is_write=False)
        ready_stats = ready.access_file("/cold.bin", 0, 64 * 1024, is_write=False)
        return (
            cold_stats["total_ms"],
            cold_stats.get("startup_ms", 0.0),
            ready_stats["total_ms"],
            ready_stats.get("startup_ms", 0.0),
        )
    finally:
        cold.stop()
        ready.stop()


def collect_core_metrics() -> dict[str, float]:
    random.seed(7)
    with workspace_tempdir(prefix="fake-hdd-profile-", subdir="profiles") as backing_dir:
        vhdd = VirtualHDD(str(backing_dir), latency_scale=0.0)
        scheduler = OSScheduler(vhdd.model)
        vhdd.set_scheduler(scheduler)
        try:
            sequential_write_avg_ms, sequential_flush_ms, sequential_read_avg_ms = _profile_sequential(vhdd)
            random_read_avg_ms, contended_random_avg_ms = _profile_random_reads(
                scheduler,
                vhdd.model.addressable_blocks,
            )
            metadata_churn_total_ms = _profile_metadata_churn(vhdd)
            copy_heavy_total_ms = _profile_copy_heavy(vhdd)
            mixed_churn_total_ms = _profile_mixed_churn(vhdd)
        finally:
            vhdd.stop()

        cold_start_read_ms, cold_start_startup_ms, ready_read_ms, ready_startup_ms = _profile_cold_vs_ready(backing_dir)

    return {
        "sequential_write_64k_avg_ms": sequential_write_avg_ms,
        "sequential_flush_total_ms": sequential_flush_ms,
        "sequential_read_64k_avg_ms": sequential_read_avg_ms,
        "random_read_4k_avg_ms": random_read_avg_ms,
        "contended_random_4k_avg_ms": contended_random_avg_ms,
        "metadata_churn_total_ms": metadata_churn_total_ms,
        "copy_heavy_total_ms": copy_heavy_total_ms,
        "mixed_churn_total_ms": mixed_churn_total_ms,
        "cold_start_read_ms": cold_start_read_ms,
        "cold_start_startup_ms": cold_start_startup_ms,
        "ready_read_ms": ready_read_ms,
        "ready_startup_ms": ready_startup_ms,
    }


def assert_core_expectations(metrics: dict[str, float]) -> None:
    assert metrics["sequential_write_64k_avg_ms"] < metrics["random_read_4k_avg_ms"]
    assert metrics["sequential_read_64k_avg_ms"] < metrics["random_read_4k_avg_ms"]
    assert metrics["contended_random_4k_avg_ms"] > metrics["sequential_read_64k_avg_ms"]
    assert metrics["metadata_churn_total_ms"] > 0.0
    assert metrics["copy_heavy_total_ms"] > 0.0
    assert metrics["mixed_churn_total_ms"] > max(metrics["metadata_churn_total_ms"], metrics["copy_heavy_total_ms"])
    assert metrics["cold_start_startup_ms"] > metrics["ready_startup_ms"]


def profile_core() -> dict[str, float]:
    metrics = collect_core_metrics()
    print("=== HIGH-FIDELITY CORE PROFILING (Simulated Milliseconds) ===")
    for key, value in metrics.items():
        print(f"{key:32} {value:10.2f}")
    assert_core_expectations(metrics)
    print("Core profiling expectations passed.")
    return metrics


if __name__ == "__main__":
    profile_core()
