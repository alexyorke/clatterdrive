import time
import requests
import random
import os
import concurrent.futures

BASE_URL = "http://localhost:8080"
TEST_FILE = "stress_test_file.bin"
NUM_TESTS = 100


def _checked(response):
    response.raise_for_status()
    return response

def log_test(name, duration, size_mb=0):
    throughput = (size_mb / (duration / 1000.0)) if duration > 0 and size_mb > 0 else 0
    print(f"[TEST] {name:30} | Duration: {duration:7.2f}ms | Throughput: {throughput:7.2f} MB/s")

def test_sequential_write(size_kb=1024):
    data = os.urandom(size_kb * 1024)
    start = time.time()
    _checked(requests.put(f"{BASE_URL}/seq_write.bin", data=data))
    duration = (time.time() - start) * 1000
    return duration, size_kb / 1024.0

def test_sequential_read(size_kb=1024):
    start = time.time()
    _checked(requests.get(f"{BASE_URL}/seq_write.bin"))
    duration = (time.time() - start) * 1000
    return duration, size_kb / 1024.0

def test_random_read(file_path, size_kb=4):
    # Use Range header to simulate random seek
    offset = random.randint(0, 1024 * 1024 - size_kb * 1024)
    headers = {"Range": f"bytes={offset}-{offset + size_kb * 1024 - 1}"}
    start = time.time()
    _checked(requests.get(f"{BASE_URL}/{file_path}", headers=headers))
    duration = (time.time() - start) * 1000
    return duration

def run_suite():
    print(f"Starting Stress Test Suite ({NUM_TESTS} operations)...")
    
    # 1. Warm up: Sequential Write (10MB)
    dur, size = test_sequential_write(10240)
    log_test("Initial 10MB Seq Write", dur, size)

    # 2. Sequential Read (10MB) - Should hit Read-Ahead
    dur, size = test_sequential_read(10240)
    log_test("Initial 10MB Seq Read", dur, size)

    # 3. 20x Random Reads (4KB) - Mechanical Seek
    rand_read_times = []
    for i in range(20):
        dur = test_random_read("seq_write.bin", 4)
        rand_read_times.append(dur)
    log_test("20x 4KB Random Reads (Avg)", sum(rand_read_times)/20, (20*4)/1024.0)

    # 4. Multi-threaded Contention (10 threads)
    print("\nStarting Thread Contention Test (10 threads)...")
    start_contention = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(test_random_read, "seq_write.bin", 4) for _ in range(50)]
        concurrent.futures.wait(futures)
    dur_contention = (time.time() - start_contention) * 1000
    log_test("50x Contended Random Reads", dur_contention, (50*4)/1024.0)

    # 5. Directory Operations (Create 20 small files)
    print("\nCreating 20 small files...")
    start_dir = time.time()
    for i in range(20):
        _checked(requests.put(f"{BASE_URL}/file_{i}.txt", data=b"small data"))
    dur_dir = (time.time() - start_dir) * 1000
    log_test("20x File Creations", dur_dir)

    print("\nStress Test Complete.")

if __name__ == "__main__":
    try:
        run_suite()
    except Exception as e:
        print(f"Error during stress test: {e}")
