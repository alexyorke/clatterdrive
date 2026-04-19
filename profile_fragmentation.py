from __future__ import annotations

from hdd_model import VirtualHDD
from runtime_paths import workspace_tempdir


def collect_fragmentation_metrics() -> dict[str, float]:
    with workspace_tempdir(prefix="fake-hdd-frag-", subdir="profiles") as backing_dir:
        vhdd = VirtualHDD(str(backing_dir), latency_scale=0.0)
        try:
            contiguous_write = vhdd.access_file("/contiguous.bin", 0, 10 * 1024 * 1024, is_write=True)["total_ms"]
            contiguous_write += vhdd.sync_all()
            vhdd.reset_runtime_state()
            contiguous_read_stats = vhdd.access_file("/contiguous.bin", 0, 10 * 1024 * 1024, is_write=False)

            for index in range(500):
                vhdd.access_file(f"/noise_{index}.bin", 0, 16384, is_write=True)
            vhdd.sync_all()
            for index in range(0, 500, 2):
                vhdd.delete_path(f"/noise_{index}.bin")

            fragmented_write = vhdd.access_file("/fragmented.bin", 0, 10 * 1024 * 1024, is_write=True)["total_ms"]
            fragmented_write += vhdd.sync_all()
            vhdd.reset_runtime_state()
            fragmented_read_stats = vhdd.access_file("/fragmented.bin", 0, 10 * 1024 * 1024, is_write=False)
            contiguous_fragmentation = float(vhdd.fs.get_fragmentation_score("/contiguous.bin"))
            fragmented_fragmentation = float(vhdd.fs.get_fragmentation_score("/fragmented.bin"))
            vhdd.fs.assert_consistent()
        finally:
            vhdd.stop()

    return {
        "contiguous_write_ms": contiguous_write,
        "contiguous_read_ms": contiguous_read_stats["total_ms"],
        "contiguous_read_extents": float(contiguous_read_stats["extents"]),
        "contiguous_fragmentation_score": contiguous_fragmentation,
        "fragmented_write_ms": fragmented_write,
        "fragmented_read_ms": fragmented_read_stats["total_ms"],
        "fragmented_read_extents": float(fragmented_read_stats["extents"]),
        "fragmented_fragmentation_score": fragmented_fragmentation,
    }


def assert_fragmentation_expectations(metrics: dict[str, float]) -> None:
    assert metrics["contiguous_read_extents"] <= 8
    assert metrics["contiguous_fragmentation_score"] <= 8
    assert metrics["fragmented_read_extents"] > metrics["contiguous_read_extents"]
    assert metrics["fragmented_fragmentation_score"] > metrics["contiguous_fragmentation_score"]
    assert metrics["fragmented_read_ms"] > metrics["contiguous_read_ms"]
    assert metrics["fragmented_write_ms"] >= metrics["contiguous_write_ms"]


def profile_fragmentation() -> dict[str, float]:
    metrics = collect_fragmentation_metrics()
    print("=== HIGH-FIDELITY FRAGMENTATION PROFILING (Simulated Milliseconds) ===")
    for key, value in metrics.items():
        print(f"{key:28} {value:10.2f}")
    assert_fragmentation_expectations(metrics)
    print("Fragmentation profiling expectations passed.")
    return metrics


if __name__ == "__main__":
    profile_fragmentation()
