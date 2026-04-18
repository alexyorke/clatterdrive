import time
import tempfile
from pathlib import Path
from hdd_model import VirtualHDD

def profile_fragmentation():
    with tempfile.TemporaryDirectory(prefix="fake-hdd-frag-") as backing_dir:
        vhdd = VirtualHDD(str(Path(backing_dir)))
        try:
            print("=== HIGH-FIDELITY FRAGMENTATION PROFILING (No Network) ===")

            # 1. Sequential Case (Clean disk)
            print("\n[Case 1] Writing 10MB sequentially to clean disk...")
            start = time.perf_counter()
            vhdd.access_file("contiguous.bin", 0, 10 * 1024 * 1024, is_write=True)
            vhdd.sync_all()
            write_dur = (time.perf_counter() - start) * 1000
            print(f"Contiguous Write: {write_dur:.2f}ms")

            vhdd.reset_runtime_state()
            start = time.perf_counter()
            stats = vhdd.access_file("contiguous.bin", 0, 10 * 1024 * 1024, is_write=False)
            read_dur = (time.perf_counter() - start) * 1000
            print(f"Contiguous Read: {read_dur:.2f}ms | Extents: {stats['extents']}")

            # 2. Fragmented Case
            print("\n[Case 2] Creating fragmentation (filling/deleting)...")
            for i in range(500):
                vhdd.access_file(f"noise_{i}.bin", 0, 16384, is_write=True)
            vhdd.sync_all()
            for i in range(0, 500, 2):
                vhdd.delete_path(f"noise_{i}.bin")

            print("Writing 10MB to fragmented disk...")
            start = time.perf_counter()
            vhdd.access_file("fragmented.bin", 0, 10 * 1024 * 1024, is_write=True)
            vhdd.sync_all()
            write_dur = (time.perf_counter() - start) * 1000
            print(f"Fragmented Write: {write_dur:.2f}ms")

            print("Reading 10MB fragmented file...")
            vhdd.reset_runtime_state()
            start = time.perf_counter()
            stats = vhdd.access_file("fragmented.bin", 0, 10 * 1024 * 1024, is_write=False)
            read_dur = (time.perf_counter() - start) * 1000
            print(f"Fragmented Read: {read_dur:.2f}ms | Extents: {stats['extents']}")

            if stats['extents'] > 1:
                print(f"\nSUCCESS: Fragmentation confirmed. File split into {stats['extents']} pieces.")
                print("Performance penalty is mechanically simulated.")
        finally:
            vhdd.stop()

if __name__ == "__main__":
    profile_fragmentation()
