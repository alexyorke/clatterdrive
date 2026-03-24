import time
import os
import random
from hdd_model import VirtualHDD
from os_scheduler import OSScheduler

def profile_fragmentation():
    vhdd = VirtualHDD("backing_storage")
    scheduler = OSScheduler(vhdd.model)
    
    print("=== HIGH-FIDELITY FRAGMENTATION PROFILING (No Network) ===")
    
    # 1. Sequential Case (Clean disk)
    print("\n[Case 1] Writing 10MB sequentially to clean disk...")
    start = time.time()
    for i in range(160): # ~10MB
        req_id = scheduler.submit_bio(i * 16, 64 * 1024, is_write=True) # Fake offset for bio
        # Actually we need to call vhdd.access_file to trigger FS allocator
    
    # Let's use vhdd.access_file directly for better FS simulation tracing
    start = time.time()
    vhdd.access_file("contiguous.bin", 0, 10 * 1024 * 1024, is_write=True)
    write_dur = (time.time() - start) * 1000
    print(f"Contiguous Write: {write_dur:.2f}ms")
    
    start = time.time()
    stats = vhdd.access_file("contiguous.bin", 0, 10 * 1024 * 1024, is_write=False)
    read_dur = (time.time() - start) * 1000
    print(f"Contiguous Read: {read_dur:.2f}ms | Extents: {stats['extents']}")

    # 2. Fragmented Case
    print("\n[Case 2] Creating fragmentation (filling/deleting)...")
    for i in range(500):
        vhdd.access_file(f"noise_{i}.bin", 0, 16384, is_write=True)
    for i in range(0, 500, 2):
        vhdd.fs.delete(f"noise_{i}.bin")
        
    print("Writing 10MB to fragmented disk...")
    start = time.time()
    vhdd.access_file("fragmented.bin", 0, 10 * 1024 * 1024, is_write=True)
    write_dur = (time.time() - start) * 1000
    print(f"Fragmented Write: {write_dur:.2f}ms")
    
    print("Reading 10MB fragmented file...")
    # Disable read-ahead for this test to see pure mechanical impact
    vhdd.model.read_ahead_lba = -1 
    
    start = time.time()
    stats = vhdd.access_file("fragmented.bin", 0, 10 * 1024 * 1024, is_write=False)
    read_dur = (time.time() - start) * 1000
    print(f"Fragmented Read: {read_dur:.2f}ms | Extents: {stats['extents']}")
    
    if stats['extents'] > 1:
        print(f"\nSUCCESS: Fragmentation confirmed. File split into {stats['extents']} pieces.")
        print("Performance penalty is mechanically simulated.")

if __name__ == "__main__":
    profile_fragmentation()
