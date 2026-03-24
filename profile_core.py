import time
import os
import random
import concurrent.futures
from hdd_model import VirtualHDD
from os_scheduler import OSScheduler

def profile_core():
    vhdd = VirtualHDD("backing_storage")
    scheduler = OSScheduler(vhdd.model)
    
    print("=== HIGH-FIDELITY CORE PROFILING (No Network Overhead) ===")
    
    # Test 1: Sequential Write (Write-Back Cache)
    print("\nTesting 100 Sequential 64KB Writes (Write-Back)...")
    start = time.time()
    for i in range(100):
        req_id = scheduler.submit_bio(i * 128, 64 * 1024, is_write=True)
        scheduler.wait_for_completion(req_id)
    dur = (time.time() - start) * 1000
    print(f"Result: {dur:.2f}ms total ({dur/100:.2f}ms avg per write)")
    print("Interpretation: Write-Back cache absorbs mechanical latency.")

    # Test 2: Sequential Read (Read-Ahead)
    print("\nTesting 100 Sequential 64KB Reads (Read-Ahead)...")
    # First read triggers mechanical
    req_id = scheduler.submit_bio(0, 64 * 1024, is_write=False)
    scheduler.wait_for_completion(req_id)
    
    start = time.time()
    for i in range(1, 100):
        req_id = scheduler.submit_bio(i * 128, 64 * 1024, is_write=False)
        scheduler.wait_for_completion(req_id)
    dur = (time.time() - start) * 1000
    print(f"Result: {dur:.2f}ms total ({dur/99:.2f}ms avg per read)")
    print("Interpretation: Read-Ahead buffer provides DRAM-speed sequential access.")

    # Test 3: Random Reads (Mechanical Seek/Rotation)
    print("\nTesting 20 Random 4KB Reads (NCQ/RPO Optimization)...")
    start = time.time()
    for i in range(20):
        lba = random.randint(0, 1000000)
        req_id = scheduler.submit_bio(lba, 4 * 1024, is_write=False)
        scheduler.wait_for_completion(req_id)
    dur = (time.time() - start) * 1000
    print(f"Result: {dur:.2f}ms total ({dur/20:.2f}ms avg per read)")
    print("Interpretation: Full mechanical latency (Seek + Rotation) per request.")

    # Test 4: Multi-threaded Contention (Elevator/SCAN Scheduler)
    print("\nTesting 50 Concurrent Random Reads (10 threads)...")
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _ in range(50):
            lba = random.randint(0, 1000000)
            futures.append(executor.submit(lambda l: scheduler.wait_for_completion(scheduler.submit_bio(l, 4096, False)), lba))
        concurrent.futures.wait(futures)
    dur = (time.time() - start) * 1000
    print(f"Result: {dur:.2f}ms total for 50 ops")
    print("Interpretation: OS Elevator (SCAN) sorts requests to optimize seek path.")

if __name__ == "__main__":
    profile_core()
