import time
import math
import random
import threading
from collections import deque
from audio_engine import engine as audio

class HDDLatenyModel:
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
                 settle_ms=0.2,
                 head_switch_ms=1.5,
                 transfer_rate_outer_mbps=250,
                 transfer_rate_inner_mbps=120,
                 ncq_depth=32,
                 write_cache_enabled=True):
        
        self.target_rpm = rpm
        self.current_rpm = 0.0
        self.num_heads = platters * 2
        self.total_cylinders = cylinders_per_surface
        self.ms_per_rotation = 60000.0 / rpm
        self.ncq_depth = ncq_depth
        self.write_cache_enabled = write_cache_enabled
        
        self.rate_outer = transfer_rate_outer_mbps
        self.rate_inner = transfer_rate_inner_mbps
        self.a = track_to_track_ms
        self.b = (avg_seek_ms - self.a - settle_ms) / math.sqrt(self.total_cylinders / 3.0)
        self.settle_ms = settle_ms
        self.head_switch_ms = head_switch_ms
        
        self.lock = threading.Lock()
        self.read_ahead_lba = -1
        self.read_ahead_size = 0
        
        self.running = True
        threading.Thread(target=self._spin_up_loop, daemon=True).start()
        # Background calibration thread
        threading.Thread(target=self._background_tasks_loop, daemon=True).start()

    def _spin_up_loop(self):
        while self.current_rpm < self.target_rpm:
            self.current_rpm += 150 
            audio._update_telemetry(self.current_rpm)
            time.sleep(0.1)
        self.current_rpm = self.target_rpm

    def _background_tasks_loop(self):
        """Simulates autonomous background firmware tasks."""
        while self.running:
            time.sleep(random.uniform(5, 15)) # Wait some time
            # 1. Thermal Calibration Tick
            if time.time() - self._last_access_time > 5:
                audio._update_telemetry(self.current_rpm, is_cal=True)
            
            # 2. Idle Ramp Parking (if idle long enough)
            if time.time() - self._last_access_time > 30:
                audio._update_telemetry(self.current_rpm, is_park=True)
                time.sleep(2) # Stay parked

    def _lba_to_chs(self, lba):
        sectors_per_cyl = self.num_heads * 2000 
        cyl = (lba // sectors_per_cyl) % self.total_cylinders
        rem = lba % sectors_per_cyl
        head = rem // 2000
        sector = rem % 2000
        return cyl, head, sector

    def _calculate_total_latency(self, start_cyl, start_head, start_sector, target_lba):
        t_cyl, t_head, t_sector = self._lba_to_chs(target_lba)
        dist = abs(t_cyl - start_cyl)
        seek_ms = 0 if dist == 0 else self.a + self.b * math.sqrt(dist) + self.settle_ms
        head_switch_ms = self.head_switch_ms if (dist == 0 and t_head != start_head) else 0
        
        zone_factor = 1.0 - (t_cyl / self.total_cylinders)
        sectors_this_track = 1000 + 1000 * zone_factor
        seek_rotations = (seek_ms + head_switch_ms) / self.ms_per_rotation
        current_sector_after_seek = (start_sector + seek_rotations * sectors_this_track) % sectors_this_track
        
        sector_diff = (t_sector - current_sector_after_seek) % sectors_this_track
        rot_ms = (sector_diff / sectors_this_track) * self.ms_per_rotation
        
        return seek_ms + head_switch_ms + rot_ms, t_cyl, t_head, t_sector, dist

    def _get_transfer_time(self, lba, size_bytes):
        cyl, _, _ = self._lba_to_chs(lba)
        zone_factor = 1.0 - (cyl / self.total_cylinders)
        rate = self.rate_inner + (self.rate_outer - self.rate_inner) * zone_factor
        return (size_bytes / (1024 * 1024)) / rate * 1000, rate

    def submit_request(self, lba, size_bytes, is_write=False):
        with self.lock:
            if not is_write and lba >= self.read_ahead_lba and lba < (self.read_ahead_lba + self.read_ahead_size // 512):
                audio._update_telemetry(self.current_rpm, is_seq=True)
                return {"total_ms": 0.01, "cache_hit": True, "type": "READ"}

            if is_write and self.write_cache_enabled:
                return {"total_ms": 0.05, "cache_hit": True, "type": "WRITE"}

        with self.lock:
            total_lat, t_cyl, t_head, t_sector, dist = self._calculate_total_latency(
                self._current_cyl, self._current_head, self._current_sector, lba
            )
            
            audio._update_telemetry(self.current_rpm, seek_trigger=True, seek_dist=dist)
            
            xfer_ms, rate = self._get_transfer_time(lba, size_bytes)
            total_lat += xfer_ms
            time.sleep(total_lat / 1000.0)
            
            sectors_passed = math.ceil(size_bytes / 512.0)
            HDDLatenyModel._current_cyl = t_cyl
            HDDLatenyModel._current_head = t_head
            HDDLatenyModel._current_sector = (t_sector + sectors_passed) % 2000
            HDDLatenyModel._last_access_time = time.time()
            
            if not is_write:
                self.read_ahead_lba = lba + sectors_passed
                self.read_ahead_size = 128 * 1024
                
        return {
            "total_ms": total_lat,
            "seek_ms": total_lat - xfer_ms,
            "transfer_ms": xfer_ms,
            "rate_mbps": rate,
            "cyl": t_cyl,
            "head": t_head,
            "type": "WRITE" if is_write else "READ",
            "cache_hit": False
        }

class VirtualHDD:
    def __init__(self, backing_dir):
        self.model = HDDLatenyModel()
        self.backing_dir = backing_dir
        self.file_lba_start = {}

    def get_file_lba(self, path):
        if path not in self.file_lba_start:
            self.file_lba_start[path] = random.randint(0, 50000000)
        return self.file_lba_start[path]

    def access_file(self, path, offset, length, is_write=False):
        start_lba = self.get_file_lba(path)
        target_lba = start_lba + (offset // 512)
        return self.model.submit_request(target_lba, length, is_write)
