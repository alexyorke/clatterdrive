import time
import os
import requests

BASE_URL = "http://localhost:8080"


def _checked(response):
    response.raise_for_status()
    return response

def test_fragmentation():
    print("=== REAL FRAGMENTATION TEST ===")
    
    # 1. Fill disk with many small files to create 'blocks'
    print("Step 1: Creating noise (100 small files)...")
    for i in range(100):
        _checked(requests.put(f"{BASE_URL}/noise_{i}.bin", data=os.urandom(16384)))
        
    # 2. Delete every second file to create 'holes'
    print("Step 2: Creating holes (deleting 50 files)...")
    for i in range(0, 100, 2):
        _checked(requests.delete(f"{BASE_URL}/noise_{i}.bin"))
        
    # 3. Write a large file that must fit into those holes
    print("Step 3: Writing fragmented file (5MB)...")
    start_write = time.time()
    _checked(requests.put(f"{BASE_URL}/fragmented.bin", data=os.urandom(5 * 1024 * 1024)))
    write_dur = (time.time() - start_write) * 1000
    print(f"Fragmented Write Duration: {write_dur:.2f}ms")
    
    # 4. Read the fragmented file and measure mechanical seeks
    print("Step 4: Reading fragmented file...")
    start_read = time.time()
    # Force bypass of simple sequential cache by reading in chunks or random order
    # Actually, the FS simulator will return multiple extents anyway
    _checked(requests.get(f"{BASE_URL}/fragmented.bin"))
    read_dur = (time.time() - start_read) * 1000
    print(f"Fragmented Read Duration: {read_dur:.2f}ms")
    print("Check server console for 'extents > 1' and multiple 'Seek' logs.")

if __name__ == "__main__":
    try:
        test_fragmentation()
    except Exception as e:
        print(f"Error: {e}")
