import time
import random
import concurrent.futures
import tempfile
from pathlib import Path
from hdd_model import VirtualHDD
from os_scheduler import OSScheduler

def profile_core():
    with tempfile.TemporaryDirectory(prefix="fake-hdd-profile-") as backing_dir:
        vhdd = VirtualHDD(str(Path(backing_dir)))
        scheduler = OSScheduler(vhdd.model)
        vhdd.set_scheduler(scheduler)
        try:
            print("=== HIGH-FIDELITY CORE PROFILING (No Network Overhead) ===")

            # Test 1: Sequential Write (Write-Back Cache)
            print("\nTesting 100 Sequential 64KB Writes (Write-Back)...")
            start = time.perf_counter()
            for i in range(100):
                vhdd.access_file("profile_seq.bin", i * 64 * 1024, 64 * 1024, is_write=True)
            dur = (time.perf_counter() - start) * 1000
            print(f"Result: {dur:.2f}ms total ({dur/100:.2f}ms avg per write)")
            print("Interpretation: Write-Back cache absorbs mechanical latency.")
            flush_dur = vhdd.sync_all()
            print(f"Background Flush Drain: {flush_dur:.2f}ms")

            # Test 2: Sequential Read (Read-Ahead)
            print("\nTesting 100 Sequential 64KB Reads (Read-Ahead)...")
            vhdd.reset_runtime_state()
            start = time.perf_counter()
            for i in range(100):
                vhdd.access_file("profile_seq.bin", i * 64 * 1024, 64 * 1024, is_write=False)
            dur = (time.perf_counter() - start) * 1000
            print(f"Result: {dur:.2f}ms total ({dur/100:.2f}ms avg per read)")
            print("Interpretation: Early misses should establish a sequential stream and later reads should smooth out.")

            # Test 3: Random Reads (Mechanical Seek/Rotation)
            print("\nTesting 20 Random 4KB Reads (NCQ/RPO Optimization)...")
            vhdd.reset_runtime_state()
            start = time.perf_counter()
            for _ in range(20):
                lba = random.randint(0, vhdd.model.addressable_blocks - 1)
                req_id = scheduler.submit_bio(lba, 4 * 1024, is_write=False, op_kind="data")
                scheduler.wait_for_completion(req_id)
            dur = (time.perf_counter() - start) * 1000
            print(f"Result: {dur:.2f}ms total ({dur/20:.2f}ms avg per read)")
            print("Interpretation: Full mechanical latency (Seek + Rotation) per request.")

            # Test 4: Multi-threaded Contention (Elevator/SCAN Scheduler)
            print("\nTesting 50 Concurrent Random Reads (10 threads)...")
            vhdd.reset_runtime_state()
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for _ in range(50):
                    lba = random.randint(0, vhdd.model.addressable_blocks - 1)
                    futures.append(
                        executor.submit(
                            lambda value: scheduler.wait_for_completion(
                                scheduler.submit_bio(value, 4096, False, op_kind="data")
                            ),
                            lba,
                        )
                    )
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            dur = (time.perf_counter() - start) * 1000
            print(f"Result: {dur:.2f}ms total for 50 ops")
            print("Interpretation: Scheduler and cache pressure should shape tail latency under contention.")
        finally:
            vhdd.stop()

if __name__ == "__main__":
    profile_core()
