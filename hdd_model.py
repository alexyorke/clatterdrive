import time
import math
import random

class HDDLatenyModel:
    """
    Advanced HDD Latency Model based on Electro-Mechanical and Computational Architecture research.
    Simulates: 
    - ZBR (Zone Bit Recording): Faster outer tracks (low LBA), slower inner tracks (high LBA).
    - VCM (Voice-Coil Motor) Seek: T = a + b * sqrt(d) + settle_time.
    - Head Switching: Switching between heads on the same cylinder.
    - Advanced Format: 4KB physical sectors.
    - Spindle Dynamics: 7200 RPM constant velocity.
    - SMR (Shingled Magnetic Recording): Optional write penalty.
    """
    
    # Shared physical state (Head Stack Assembly)
    _current_cyl = 0
    _current_head = 0
    _current_sector = 0
    _last_access_time = time.time()

    def __init__(self, 
                 rpm=7200, 
                 platters=4,
                 cylinders_per_surface=200000,
                 avg_seek_ms=8.5,
                 track_to_track_ms=0.5,
                 settle_ms=0.2, # Time for head to stabilize over track
                 head_switch_ms=1.5, # Time to switch electrical signals between heads
                 transfer_rate_outer_mbps=250, # ZBR: Outer tracks are faster
                 transfer_rate_inner_mbps=120, # ZBR: Inner tracks are slower
                 is_smr=False):
        
        self.rpm = rpm
        self.num_heads = platters * 2
        self.total_cylinders = cylinders_per_surface
        self.settle_ms = settle_ms
        self.head_switch_ms = head_switch_ms
        self.is_smr = is_smr
        
        # Spindle timing
        self.ms_per_rotation = 60000.0 / rpm
        
        # ZBR Profile: Outer tracks (Cyl 0) to Inner tracks (Cyl max)
        self.rate_outer = transfer_rate_outer_mbps
        self.rate_inner = transfer_rate_inner_mbps
        
        # Seek model: T = a + b*sqrt(d)
        # track_to_track_ms is 'a' (min seek)
        # b calculated from avg_seek (distance = total_cyl / 3)
        self.a = track_to_track_ms
        self.b = (avg_seek_ms - self.a - self.settle_ms) / math.sqrt(self.total_cylinders / 3.0)

        # Advanced Format: 4KB sectors (8x 512B legacy sectors)
        self.physical_sector_size = 4096
        # Assume ~1000 to ~2000 sectors per track depending on zone
        self.max_sectors_per_track = 2000 
        self.min_sectors_per_track = 1000

    def _lba_to_chs(self, lba):
        """Maps LBA to Physical Geometry (Cylinder, Head, Sector)."""
        # Simplified linear mapping for simulation
        sectors_per_cyl = self.num_heads * self.max_sectors_per_track
        cyl = (lba // sectors_per_cyl) % self.total_cylinders
        rem = lba % sectors_per_cyl
        head = rem // self.max_sectors_per_track
        sector = rem % self.max_sectors_per_track
        return cyl, head, sector

    def _get_seek_time(self, target_cyl):
        distance = abs(target_cyl - self._current_cyl)
        if distance == 0:
            return 0
        # VCM Lorentz force acceleration/deceleration profile
        return self.a + self.b * math.sqrt(distance) + self.settle_ms

    def _get_rotational_latency(self, target_sector, cyl):
        # Calculate sectors on this track (ZBR)
        # Closer to 0 (outer) -> more sectors
        zone_factor = 1.0 - (cyl / self.total_cylinders)
        sectors_this_track = self.min_sectors_per_track + (self.max_sectors_per_track - self.min_sectors_per_track) * zone_factor
        
        # Platter movement since last access
        elapsed_ms = (time.time() - self._last_access_time) * 1000
        rotations = elapsed_ms / self.ms_per_rotation
        current_sector_now = (self._current_sector + rotations * sectors_this_track) % sectors_this_track
        
        sector_diff = (target_sector - current_sector_now) % sectors_this_track
        return (sector_diff / sectors_this_track) * self.ms_per_rotation

    def _get_transfer_rate(self, cyl):
        """ZBR: Outer tracks have higher linear velocity and density."""
        zone_factor = 1.0 - (cyl / self.total_cylinders)
        return self.rate_inner + (self.rate_outer - self.rate_inner) * zone_factor

    def simulate_access(self, lba, size_bytes, is_write=False):
        target_cyl, target_head, target_sector = self._lba_to_chs(lba)
        
        # 1. Seek Latency (VCM movement)
        seek_time = self._get_seek_time(target_cyl)
        
        # 2. Head Switch Latency
        # If we are on the same cylinder but different head, incur a switch delay
        head_switch_time = 0
        if seek_time == 0 and target_head != self._current_head:
            head_switch_time = self.head_switch_ms
            
        # 3. Rotational Latency (Waiting for sector)
        rot_lat = self._get_rotational_latency(target_sector, target_cyl)
        
        # 4. Transfer Time (ZBR limited)
        rate = self._get_transfer_rate(target_cyl)
        transfer_time = (size_bytes / (1024 * 1024)) / rate * 1000
        
        # 5. SMR Write Penalty
        # Shingled Magnetic Recording requires a zone rewrite for random writes
        smr_penalty = 0
        if is_write and self.is_smr:
            # Random write in SMR is extremely slow (simulating read-modify-write of 256MB zone)
            # Only apply if it's not a sequential stream (simplified check)
            smr_penalty = random.uniform(10, 50) 
        
        total_latency_ms = seek_time + head_switch_time + rot_lat + transfer_time + smr_penalty
        
        # Add Aerodynamic/Thermal Jitter (nanoscale fly height modulation)
        total_latency_ms += random.uniform(-0.02, 0.02)
        
        if total_latency_ms > 0:
            time.sleep(total_latency_ms / 1000.0)
            
        # Update shared state
        sectors_passed = math.ceil(size_bytes / 512.0)
        HDDLatenyModel._current_cyl = target_cyl
        HDDLatenyModel._current_head = target_head
        HDDLatenyModel._current_sector = (target_sector + sectors_passed) % 1000 # simplified
        HDDLatenyModel._last_access_time = time.time()
        
        return {
            "total_ms": total_latency_ms,
            "seek_ms": seek_time,
            "rot_ms": rot_lat,
            "transfer_ms": transfer_time,
            "head_switch_ms": head_switch_time,
            "rate_mbps": rate,
            "cyl": target_cyl,
            "head": target_head
        }

class VirtualHDD:
    def __init__(self, backing_dir):
        self.model = HDDLatenyModel()
        self.backing_dir = backing_dir
        self.file_lba_start = {} 

    def get_file_lba(self, path):
        if path not in self.file_lba_start:
            # Map file to a random starting LBA
            self.file_lba_start[path] = random.randint(0, 100000000)
        return self.file_lba_start[path]

    def access_file(self, path, offset, length, is_write=False):
        start_lba = self.get_file_lba(path)
        # Offset to LBA (using 512B logic for mapping)
        target_lba = start_lba + (offset // 512)
        return self.model.simulate_access(target_lba, length, is_write)
