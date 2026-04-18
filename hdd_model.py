from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from audio_engine import engine as audio
from fs_simulator import FileSystemSimulator, IOOperation
from profiles import AcousticProfile, DriveProfile, resolve_drive_profile, resolve_selected_profiles


Stats = dict[str, Any]


@dataclass(frozen=True)
class Zone:
    start_cyl: int
    end_cyl: int
    start_lba: int
    end_lba: int
    blocks_per_track: int
    transfer_rate_mbps: float


@dataclass
class DirtyWrite:
    lba: int
    size_bytes: int
    op_kind: str
    enqueued_at: float


@dataclass(frozen=True)
class StartupStage:
    name: str
    duration_ms: float
    start_rpm: float
    end_rpm: float
    calibration_pulses: int = 0
    head_load: bool = False
    head_unload: bool = False
    park: bool = False


def _prefer_profile_value(value: Any, legacy_default: Any, profile_value: Any) -> Any:
    return profile_value if value == legacy_default else value


class HDDLatencyModel:
    def __init__(
        self,
        addressable_blocks: int,
        block_bytes: int = 4096,
        rpm: int = 7200,
        platters: int = 4,
        avg_seek_ms: float = 8.4,
        track_to_track_ms: float = 0.25,
        settle_ms: float = 0.35,
        head_switch_ms: float = 0.35,
        transfer_rate_outer_mbps: float = 210,
        transfer_rate_inner_mbps: float = 120,
        ncq_depth: int = 32,
        read_ahead_kb: int = 512,
        write_cache_mb: int = 32,
        dirty_expire_ms: float = 350,
        standby_after_s: float = 60,
        unload_after_s: float = 12,
        low_rpm_after_s: float = 30,
        spinup_ms: float = 3200,
        latency_scale: float = 1.0,
        start_ready: bool = True,
        standby_to_ready_ms: float | None = None,
        power_on_to_ready_ms: float | None = None,
        unload_to_ready_ms: float | None = None,
        low_rpm_to_ready_ms: float | None = None,
        low_rpm_rpm: int | None = None,
        spin_down_ms: float | None = None,
        ready_poll_ms: float = 24.0,
        identify_poll_ms: float = 0.35,
        test_unit_ready_ms: float = 0.18,
        drive_profile: str | DriveProfile | None = None,
    ) -> None:
        resolved_drive = resolve_drive_profile(drive_profile)
        rpm = int(_prefer_profile_value(rpm, 7200, resolved_drive.rpm))
        platters = int(_prefer_profile_value(platters, 4, resolved_drive.platters))
        avg_seek_ms = float(_prefer_profile_value(avg_seek_ms, 8.4, resolved_drive.avg_seek_ms))
        track_to_track_ms = float(_prefer_profile_value(track_to_track_ms, 0.25, resolved_drive.track_to_track_ms))
        settle_ms = float(_prefer_profile_value(settle_ms, 0.35, resolved_drive.settle_ms))
        head_switch_ms = float(_prefer_profile_value(head_switch_ms, 0.35, resolved_drive.head_switch_ms))
        transfer_rate_outer_mbps = float(
            _prefer_profile_value(transfer_rate_outer_mbps, 210.0, resolved_drive.transfer_rate_outer_mbps)
        )
        transfer_rate_inner_mbps = float(
            _prefer_profile_value(transfer_rate_inner_mbps, 120.0, resolved_drive.transfer_rate_inner_mbps)
        )
        ncq_depth = int(_prefer_profile_value(ncq_depth, 32, resolved_drive.ncq_depth))
        read_ahead_kb = int(_prefer_profile_value(read_ahead_kb, 512, resolved_drive.read_ahead_kb))
        write_cache_mb = int(_prefer_profile_value(write_cache_mb, 32, resolved_drive.write_cache_mb))
        dirty_expire_ms = float(_prefer_profile_value(dirty_expire_ms, 350.0, resolved_drive.dirty_expire_ms))
        standby_after_s = float(_prefer_profile_value(standby_after_s, 60.0, resolved_drive.standby_after_s))
        unload_after_s = float(_prefer_profile_value(unload_after_s, 12.0, resolved_drive.unload_after_s))
        low_rpm_after_s = float(_prefer_profile_value(low_rpm_after_s, 30.0, resolved_drive.low_rpm_after_s))
        spinup_ms = float(_prefer_profile_value(spinup_ms, 3200.0, resolved_drive.spinup_ms))
        ready_poll_ms = float(_prefer_profile_value(ready_poll_ms, 24.0, resolved_drive.ready_poll_ms))
        identify_poll_ms = float(_prefer_profile_value(identify_poll_ms, 0.35, resolved_drive.identify_poll_ms))
        test_unit_ready_ms = float(
            _prefer_profile_value(test_unit_ready_ms, 0.18, resolved_drive.test_unit_ready_ms)
        )
        if standby_to_ready_ms is None:
            standby_to_ready_ms = resolved_drive.standby_to_ready_ms
        if power_on_to_ready_ms is None:
            power_on_to_ready_ms = resolved_drive.power_on_to_ready_ms
        if unload_to_ready_ms is None:
            unload_to_ready_ms = resolved_drive.unload_to_ready_ms
        if low_rpm_to_ready_ms is None:
            low_rpm_to_ready_ms = resolved_drive.low_rpm_to_ready_ms
        if low_rpm_rpm is None:
            low_rpm_rpm = resolved_drive.low_rpm_rpm
        if spin_down_ms is None:
            spin_down_ms = resolved_drive.spin_down_ms

        self.drive_profile = resolved_drive
        self.block_bytes = block_bytes
        self.addressable_blocks = max(1, addressable_blocks)
        self.target_rpm = rpm
        self.current_rpm = rpm
        self.num_heads = platters * 2
        self.ms_per_rotation = 60000.0 / rpm
        self.ncq_depth = ncq_depth
        self.latency_scale = latency_scale

        self.avg_seek_ms = avg_seek_ms
        self.track_to_track_ms = track_to_track_ms
        self.settle_ms = settle_ms
        self.head_switch_ms = head_switch_ms
        self.spinup_ms = spinup_ms
        self.command_overhead_ms = resolved_drive.command_overhead_ms
        self.command_overhead_by_kind = dict(resolved_drive.command_overheads_by_kind)
        self.queue_depth_penalty_ms = resolved_drive.queue_depth_penalty_ms
        self.flush_penalty_ms = 4.0
        self.read_ahead_blocks = max(8, (read_ahead_kb * 1024) // block_bytes)
        self.write_cache_bytes = write_cache_mb * 1024 * 1024
        self.dirty_expire_s = dirty_expire_ms / 1000.0
        self.standby_after_s = standby_after_s
        self.unload_after_s = unload_after_s
        self.low_rpm_after_s = low_rpm_after_s
        self.track_skew_blocks = 12
        self.cylinder_skew_blocks = 24
        self.aam_seek_penalty = 1.0
        self.low_rpm_rpm = low_rpm_rpm or max(int(self.target_rpm * 0.875), self.target_rpm - 900)
        self.unload_to_ready_ms = unload_to_ready_ms or (700.0 if platters <= 2 else 1000.0)
        self.low_rpm_to_ready_ms = low_rpm_to_ready_ms or max(4000.0, self.spinup_ms + 800.0)
        self.standby_to_ready_ms = standby_to_ready_ms or max(self.spinup_ms * 2.5, 6000.0 + platters * 1000.0)
        self.power_on_to_ready_ms = power_on_to_ready_ms or (self.standby_to_ready_ms + 1500.0 + platters * 250.0)
        self.spin_down_ms = spin_down_ms or max(2200.0, self.spinup_ms * 0.65)
        self.ready_poll_ms = ready_poll_ms
        self.identify_poll_ms = identify_poll_ms
        self.test_unit_ready_ms = test_unit_ready_ms

        outer_blocks = max(
            32,
            round((transfer_rate_outer_mbps * 1024 * 1024 * (self.ms_per_rotation / 1000.0)) / block_bytes),
        )
        inner_blocks = max(
            16,
            round((transfer_rate_inner_mbps * 1024 * 1024 * (self.ms_per_rotation / 1000.0)) / block_bytes),
        )
        self.blocks_per_track_outer = max(outer_blocks, inner_blocks)
        self.blocks_per_track_inner = min(outer_blocks, inner_blocks)
        self.total_cylinders = max(
            128,
            math.ceil(self.addressable_blocks / (self.num_heads * self.blocks_per_track_inner)),
        )
        self.seek_curve_b = (
            (self.avg_seek_ms - self.track_to_track_ms - self.settle_ms)
            / math.sqrt(max(self.total_cylinders / 3.0, 1.0))
        )
        self.zones = self._build_zones(transfer_rate_outer_mbps, transfer_rate_inner_mbps)

        self.current_cyl = 0
        self.current_head = 0
        self.current_sector = 0
        self.last_access_time = time.monotonic()
        self.power_state = "active" if start_ready else "power_on"
        self.load_unload_count = 0
        self.heads_loaded = start_ready
        self.has_completed_power_on = start_ready
        if not start_ready:
            self.current_rpm = 0.0

        self.read_cache = deque(maxlen=16)
        self.last_read_end_lba = -1
        self.lock = threading.RLock()
        self.io_lock = threading.Lock()
        self.ready_event = threading.Event()
        if start_ready:
            self.ready_event.set()
        self.transition_thread = None
        self.transition_cancel = None
        self.transition_kind = None
        self.transition_origin = None
        self.transition_stage = None
        self.transition_total_ms = 0.0
        self.last_startup_origin = None
        self.last_startup_total_ms = 0.0
        self.running = True
        self.background_thread = threading.Thread(target=self._background_tasks_loop, daemon=True)
        self.background_thread.start()

    def _build_zones(self, outer_rate: float, inner_rate: float, zone_count: int = 8) -> list[Zone]:
        zones: list[Zone] = []
        cylinders_per_zone = math.ceil(self.total_cylinders / zone_count)
        current_lba = 0

        for idx in range(zone_count):
            if current_lba >= self.addressable_blocks:
                break
            start_cyl = idx * cylinders_per_zone
            if start_cyl >= self.total_cylinders:
                break

            end_cyl = min(self.total_cylinders - 1, ((idx + 1) * cylinders_per_zone) - 1)
            fraction = idx / max(zone_count - 1, 1)
            blocks_per_track = round(
                self.blocks_per_track_outer
                + (self.blocks_per_track_inner - self.blocks_per_track_outer) * fraction
            )
            transfer_rate = outer_rate + (inner_rate - outer_rate) * fraction
            zone_blocks = (end_cyl - start_cyl + 1) * self.num_heads * blocks_per_track
            zone_end = min(self.addressable_blocks - 1, current_lba + zone_blocks - 1)
            if zone_end < current_lba:
                break
            zones.append(
                Zone(
                    start_cyl=start_cyl,
                    end_cyl=end_cyl,
                    start_lba=current_lba,
                    end_lba=zone_end,
                    blocks_per_track=blocks_per_track,
                    transfer_rate_mbps=transfer_rate,
                )
            )
            current_lba = zone_end + 1

        return zones

    def stop(self) -> None:
        self.running = False
        transition_thread = None
        with self.lock:
            if self.transition_cancel is not None:
                self.transition_cancel.set()
                transition_thread = self.transition_thread
        if transition_thread is not None and transition_thread is not threading.current_thread():
            transition_thread.join(timeout=2.0)
        self.background_thread.join(timeout=2.0)

    def reset_caches(self) -> None:
        with self.lock:
            self.read_cache.clear()
            self.last_read_end_lba = -1

    def power_on(self) -> None:
        with self.lock:
            if self.transition_cancel is not None:
                self.transition_cancel.set()
            self.power_state = "power_on"
            self.current_rpm = 0.0
            self.heads_loaded = False
            self.has_completed_power_on = False
            self.last_access_time = time.monotonic()
            self.ready_event.clear()
            self._clear_read_cache_locked()
            self.transition_kind = None
            self.transition_origin = None
            self.transition_stage = None
            self.transition_total_ms = 0.0

    def _clear_read_cache_locked(self) -> None:
        self.read_cache.clear()
        self.last_read_end_lba = -1

    def _build_resume_sequence(self, start_rpm: float, heads_loaded: bool) -> list[StartupStage]:
        stages: list[StartupStage] = []
        clamped_rpm = max(0.0, min(float(start_rpm), float(self.target_rpm)))

        if clamped_rpm < self.target_rpm:
            rpm_gap = max(0.0, (self.target_rpm - clamped_rpm) / self.target_rpm)
            rpm_recover_ms = max(180.0, self.low_rpm_to_ready_ms * max(rpm_gap, 0.15))
            stages.append(
                StartupStage("rpm_recover", rpm_recover_ms, clamped_rpm, self.target_rpm)
            )

        if not heads_loaded:
            head_load_ms, servo_lock_ms = self._allocate_stage_durations(
                max(self.unload_to_ready_ms, 220.0),
                weights=[0.58, 0.42],
                minimums=[120.0, 80.0],
            )
            stages.append(
                StartupStage(
                    "head_load",
                    head_load_ms,
                    self.target_rpm,
                    self.target_rpm,
                    head_load=True,
                )
            )
            stages.append(
                StartupStage(
                    "servo_lock",
                    servo_lock_ms,
                    self.target_rpm,
                    self.target_rpm,
                    calibration_pulses=1,
                )
            )
        elif stages:
            stages.append(
                StartupStage(
                    "servo_lock",
                    max(120.0, self.settle_ms * 400.0),
                    self.target_rpm,
                    self.target_rpm,
                    calibration_pulses=1,
                )
            )

        return stages

    def _resolve_startup_plan_locked(self) -> tuple[str | None, list[StartupStage], float]:
        if not self.has_completed_power_on:
            origin = "power_on"
            stages = self._build_startup_sequence(origin)
        elif self.current_rpm <= 0.0:
            origin = "standby"
            stages = self._build_startup_sequence(origin)
        elif not self.heads_loaded and self.current_rpm >= self.target_rpm * 0.98:
            origin = "unloaded_idle"
            stages = self._build_startup_sequence(origin)
        elif not self.heads_loaded and self.current_rpm <= self.low_rpm_rpm * 1.05:
            origin = "low_rpm_idle"
            stages = self._build_startup_sequence(origin)
        elif self.current_rpm < self.target_rpm:
            origin = "resume"
            stages = self._build_resume_sequence(self.current_rpm, self.heads_loaded)
        elif not self.heads_loaded:
            origin = "unloaded_idle"
            stages = self._build_startup_sequence(origin)
        else:
            return None, [], 0.0

        return origin, stages, sum(stage.duration_ms for stage in stages)

    def _build_spindown_sequence_locked(self) -> list[StartupStage]:
        if self.current_rpm <= 0.0 and not self.heads_loaded:
            return []

        stages = []
        current_rpm = max(0.0, self.current_rpm)
        low_rpm_target = min(current_rpm, float(self.low_rpm_rpm))

        if current_rpm > low_rpm_target + 1.0:
            durations = self._allocate_stage_durations(
                self.spin_down_ms,
                weights=[0.16, 0.24, 0.50, 0.10] if self.heads_loaded else [0.34, 0.56, 0.10],
                minimums=[120.0, 180.0, 320.0, 90.0] if self.heads_loaded else [180.0, 320.0, 90.0],
            )
            if self.heads_loaded:
                unload_ms, brake_ms, coast_ms, lock_ms = durations
                stages.append(
                    StartupStage(
                        "head_unload",
                        unload_ms,
                        current_rpm,
                        current_rpm,
                        head_unload=True,
                        park=True,
                    )
                )
            else:
                brake_ms, coast_ms, lock_ms = durations
            stages.append(
                StartupStage("spindle_brake", brake_ms, current_rpm, low_rpm_target)
            )
            stages.append(
                StartupStage("coast_down", coast_ms, low_rpm_target, 0.0)
            )
            stages.append(StartupStage("spindle_lock", lock_ms, 0.0, 0.0))
        else:
            durations = self._allocate_stage_durations(
                self.spin_down_ms,
                weights=[0.18, 0.64, 0.18] if self.heads_loaded else [0.82, 0.18],
                minimums=[120.0, 240.0, 90.0] if self.heads_loaded else [240.0, 90.0],
            )
            if self.heads_loaded:
                unload_ms, coast_ms, lock_ms = durations
                stages.append(
                    StartupStage(
                        "head_unload",
                        unload_ms,
                        current_rpm,
                        current_rpm,
                        head_unload=True,
                        park=True,
                    )
                )
            else:
                coast_ms, lock_ms = durations
            stages.append(StartupStage("coast_down", coast_ms, current_rpm, 0.0))
            stages.append(StartupStage("spindle_lock", lock_ms, 0.0, 0.0))

        return stages

    def _finish_transition(self, cancel_event: threading.Event, kind: str, origin: str) -> None:
        with self.lock:
            if self.transition_cancel is cancel_event:
                self.transition_thread = None
                self.transition_cancel = None
                self.transition_kind = None
                self.transition_origin = None
                self.transition_stage = None
                self.transition_total_ms = 0.0
            if kind == "startup":
                self.power_state = "active"
                self.current_rpm = self.target_rpm
                self.heads_loaded = True
                self.has_completed_power_on = True
                self.last_access_time = time.monotonic()
                self.ready_event.set()
            else:
                self.power_state = "standby"
                self.current_rpm = 0.0
                self.heads_loaded = False
                self.last_access_time = time.monotonic()
                self.ready_event.clear()
                self._clear_read_cache_locked()

    def _run_transition_sequence(
        self,
        kind: str,
        origin: str,
        stages: list[StartupStage],
        cancel_event: threading.Event,
    ) -> None:
        try:
            for stage in stages:
                slices = max(1, min(8, math.ceil(stage.duration_ms / 500.0)))
                slice_ms = stage.duration_ms / slices
                for index in range(slices):
                    if cancel_event.is_set() or not self.running:
                        return
                    progress = (index + 1) / slices
                    rpm = stage.start_rpm + (stage.end_rpm - stage.start_rpm) * progress
                    is_cal = stage.calibration_pulses > 0 and index < stage.calibration_pulses
                    seek_trigger = stage.head_load and index == 0
                    park = stage.park and index == 0
                    with self.lock:
                        if self.transition_cancel is not cancel_event:
                            return
                        self.transition_stage = stage.name
                        self.current_rpm = rpm
                        if kind == "startup":
                            self.power_state = "starting"
                        else:
                            self.power_state = "spinning_down"
                        if stage.head_unload and index == 0 and self.heads_loaded:
                            self.heads_loaded = False
                            self.load_unload_count += 1
                        if stage.head_load and index == 0:
                            self.heads_loaded = True
                    audio.emit_telemetry(
                        rpm,
                        seek_trigger=seek_trigger,
                        seek_dist=28 if seek_trigger else 0,
                        is_park=park,
                        queue_depth=1,
                        op_kind="metadata" if (is_cal or seek_trigger or park or stage.head_unload) else "data",
                        is_cal=is_cal,
                        is_spinup=kind == "startup" and stage.name in {"spinup", "rpm_recover", "spindle_unlock"},
                    )
                    self._sleep_ms(slice_ms)
            self._finish_transition(cancel_event, kind, origin)
        finally:
            with self.lock:
                if self.transition_cancel is cancel_event:
                    self.transition_thread = None
                    self.transition_cancel = None
                    self.transition_kind = None
                    self.transition_origin = None
                    self.transition_stage = None
                    self.transition_total_ms = 0.0

    def begin_async_startup(self) -> bool:
        thread_to_join = None
        with self.lock:
            if self.transition_kind == "startup" and self.transition_thread is not None:
                return False
            if self.ready_event.is_set() and self.power_state == "active":
                return False
            if self.transition_kind == "spindown" and self.transition_cancel is not None:
                self.transition_cancel.set()
                thread_to_join = self.transition_thread
        if thread_to_join is not None and thread_to_join is not threading.current_thread():
            thread_to_join.join(timeout=2.0)

        with self.lock:
            origin, stages, total_ms = self._resolve_startup_plan_locked()
            if not stages:
                self.power_state = "active"
                self.current_rpm = self.target_rpm
                self.heads_loaded = True
                self.ready_event.set()
                return False
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=self._run_transition_sequence,
                args=("startup", origin, stages, cancel_event),
                daemon=True,
            )
            self.transition_thread = thread
            self.transition_cancel = cancel_event
            self.transition_kind = "startup"
            self.transition_origin = origin
            self.transition_stage = stages[0].name
            self.transition_total_ms = total_ms
            self.last_startup_origin = origin
            self.last_startup_total_ms = total_ms
            self.power_state = "starting"
            self.ready_event.clear()
        thread.start()
        return True

    def begin_async_spindown(self) -> bool:
        with self.lock:
            if self.transition_kind is not None or self.power_state == "standby":
                return False
            stages = self._build_spindown_sequence_locked()
            if not stages:
                return False
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=self._run_transition_sequence,
                args=("spindown", "standby", stages, cancel_event),
                daemon=True,
            )
            self.transition_thread = thread
            self.transition_cancel = cancel_event
            self.transition_kind = "spindown"
            self.transition_origin = "standby"
            self.transition_stage = stages[0].name
            self.transition_total_ms = sum(stage.duration_ms for stage in stages)
            self.power_state = "spinning_down"
            self.ready_event.clear()
        thread.start()
        return True

    def _wait_for_ready_poll(self) -> Stats:
        startup_ms = 0.0
        startup_origin = None
        startup_started_here = self.begin_async_startup()
        with self.lock:
            if self.transition_kind == "startup":
                startup_origin = self.transition_origin
                if startup_started_here:
                    startup_ms = self.transition_total_ms
            elif startup_started_here:
                startup_origin = self.last_startup_origin
                startup_ms = self.last_startup_total_ms

        poll_ms = 0.0
        poll_count = 0
        while not self.ready_event.is_set():
            probe_ms = self.identify_poll_ms if poll_count == 0 else self.test_unit_ready_ms
            cycle_ms = probe_ms + self.ready_poll_ms
            audio.emit_telemetry(self.current_rpm, queue_depth=1, op_kind="metadata")
            self._sleep_ms(cycle_ms)
            poll_ms += cycle_ms
            poll_count += 1
            if self.latency_scale <= 0.0:
                self.ready_event.wait(timeout=0.001)

        return {
            "startup_ms": startup_ms,
            "startup_origin": startup_origin,
            "ready_poll_ms": poll_ms,
            "ready_poll_count": poll_count,
        }

    def _sleep_ms(self, latency_ms: float) -> None:
        if self.latency_scale <= 0.0:
            return
        time.sleep(max(latency_ms, 0.0) * self.latency_scale / 1000.0)

    def get_estimated_lba(self) -> int:
        zone = self._zone_for_cyl(self.current_cyl)
        blocks_per_cyl = self.num_heads * zone.blocks_per_track
        cyl_offset = self.current_cyl - zone.start_cyl
        return zone.start_lba + cyl_offset * blocks_per_cyl + self.current_head * zone.blocks_per_track + self.current_sector

    def _zone_for_lba(self, lba: int) -> Zone:
        clamped_lba = min(max(lba, 0), self.addressable_blocks - 1)
        for zone in self.zones:
            if clamped_lba <= zone.end_lba:
                return zone
        return self.zones[-1]

    def _zone_for_cyl(self, cyl: int) -> Zone:
        for zone in self.zones:
            if zone.start_cyl <= cyl <= zone.end_cyl:
                return zone
        return self.zones[-1]

    def _lba_to_chs(self, lba: int) -> tuple[int, int, int, Zone]:
        zone = self._zone_for_lba(lba)
        relative_lba = max(0, min(lba, zone.end_lba) - zone.start_lba)
        blocks_per_cyl = self.num_heads * zone.blocks_per_track
        cyl = zone.start_cyl + (relative_lba // blocks_per_cyl)
        remaining = relative_lba % blocks_per_cyl
        head = remaining // zone.blocks_per_track
        sector = remaining % zone.blocks_per_track
        return cyl, head, sector, zone

    def _cache_overlap_blocks(self, lba: int, blocks: int, now: float) -> int:
        requested_end = lba + blocks - 1
        overlap = 0
        while self.read_cache and self.read_cache[0][2] < now:
            self.read_cache.popleft()
        for start, end, _ in self.read_cache:
            if start <= lba <= end:
                overlap = max(overlap, min(requested_end, end) - lba + 1)
        return overlap

    def _remember_read(self, lba: int, blocks: int) -> None:
        if lba <= self.last_read_end_lba + 1:
            cache_blocks = max(blocks * 4, self.read_ahead_blocks)
        else:
            cache_blocks = max(blocks * 2, self.read_ahead_blocks // 2)
        self.last_read_end_lba = lba + blocks - 1
        self.read_cache.append((lba, lba + cache_blocks - 1, time.monotonic() + 2.0))

    def note_cached_write(self, lba: int, size_bytes: int) -> None:
        blocks = max(1, math.ceil(size_bytes / self.block_bytes))
        self.read_cache.append((lba, lba + blocks - 1, time.monotonic() + 1.0))

    def _calculate_position_latency(
        self,
        target_lba: int,
        block_count: int,
    ) -> tuple[float, int, int, int, int, Zone]:
        target_cyl, target_head, target_sector, zone = self._lba_to_chs(target_lba)
        distance = abs(target_cyl - self.current_cyl)

        if distance == 0:
            seek_ms = 0.0
        else:
            seek_ms = (
                self.track_to_track_ms
                + self.seek_curve_b * math.sqrt(distance)
                + self.settle_ms
            ) * self.aam_seek_penalty

        head_switch_ms = self.head_switch_ms if (distance == 0 and target_head != self.current_head) else 0.0
        skew_blocks = 0
        if distance > 0:
            skew_blocks += self.cylinder_skew_blocks
        elif target_head != self.current_head:
            skew_blocks += self.track_skew_blocks

        rotational_blocks = ((seek_ms + head_switch_ms) / self.ms_per_rotation) * zone.blocks_per_track
        current_sector_after_seek = (self.current_sector + rotational_blocks + skew_blocks) % zone.blocks_per_track
        sector_delta = (target_sector - current_sector_after_seek) % zone.blocks_per_track
        rotational_ms = (sector_delta / zone.blocks_per_track) * self.ms_per_rotation

        transfer_ms = self._transfer_ms_for_span(target_lba, block_count)
        return seek_ms + head_switch_ms + rotational_ms + transfer_ms, target_cyl, target_head, target_sector, distance, zone

    def _transfer_ms_for_span(self, start_lba: int, block_count: int) -> float:
        remaining_blocks = max(0, block_count)
        lba = max(0, start_lba)
        transfer_ms = 0.0
        while remaining_blocks > 0:
            zone = self._zone_for_lba(lba)
            zone_blocks = min(remaining_blocks, zone.end_lba - lba + 1)
            transfer_ms += ((zone_blocks * self.block_bytes) / (1024 * 1024)) / zone.transfer_rate_mbps * 1000.0
            remaining_blocks -= zone_blocks
            lba += zone_blocks
        return transfer_ms

    def _command_overhead_for(self, op_kind: str, queue_depth: int) -> float:
        base_ms = self.command_overhead_by_kind.get(op_kind, self.command_overhead_ms)
        queue_scale = 1.0
        if op_kind in {"journal", "flush"}:
            queue_scale = 1.25
        elif op_kind in {"metadata", "background"}:
            queue_scale = 0.85
        queue_penalty_ms = max(queue_depth - 1, 0) * self.queue_depth_penalty_ms * queue_scale
        return base_ms + queue_penalty_ms

    def _allocate_stage_durations(
        self,
        total_ms: float,
        weights: list[float],
        minimums: list[float],
    ) -> list[float]:
        total_ms = max(float(total_ms), 0.0)
        if not weights:
            return []

        weight_sum = sum(max(weight, 0.0) for weight in weights)
        safe_weights = [max(weight, 0.0) for weight in weights]
        safe_mins = [max(minimum, 0.0) for minimum in minimums]
        minimum_sum = sum(safe_mins)

        if total_ms <= 0.0:
            return [0.0] * len(weights)

        if minimum_sum > 0.0 and total_ms <= minimum_sum:
            scale = total_ms / minimum_sum
            durations = [minimum * scale for minimum in safe_mins]
        else:
            residual = max(total_ms - minimum_sum, 0.0)
            if weight_sum <= 0.0:
                extra_share = residual / len(safe_weights)
                durations = [minimum + extra_share for minimum in safe_mins]
            else:
                durations = [
                    minimum + residual * (weight / weight_sum)
                    for weight, minimum in zip(safe_weights, safe_mins)
                ]

        durations[-1] = max(0.0, total_ms - sum(durations[:-1]))
        return durations

    def _build_startup_sequence(self, origin: str) -> list[StartupStage]:
        if origin == "power_on":
            total_ms = self.power_on_to_ready_ms
            (
                electronics_ms,
                spindle_unlock_ms,
                spinup_ms,
                self_test_ms,
                head_load_ms,
                servo_lock_ms,
            ) = self._allocate_stage_durations(
                total_ms,
                weights=[0.04, 0.03, 0.68, 0.13, 0.04, 0.08],
                minimums=[120.0, 90.0, 300.0, 250.0, 100.0, 140.0],
            )
            return [
                StartupStage("electronics_init", electronics_ms, 0.0, 0.0, calibration_pulses=0),
                StartupStage("spindle_unlock", spindle_unlock_ms, 0.0, max(self.target_rpm * 0.08, 120.0), calibration_pulses=0),
                StartupStage("spinup", spinup_ms, max(self.target_rpm * 0.08, 120.0), self.target_rpm, calibration_pulses=0),
                StartupStage("self_test", self_test_ms, self.target_rpm, self.target_rpm, calibration_pulses=2),
                StartupStage("head_load", head_load_ms, self.target_rpm, self.target_rpm, calibration_pulses=0, head_load=True),
                StartupStage("servo_lock", servo_lock_ms, self.target_rpm, self.target_rpm, calibration_pulses=2),
            ]
        if origin == "standby":
            total_ms = self.standby_to_ready_ms
            spinup_ms, head_load_ms, servo_lock_ms = self._allocate_stage_durations(
                total_ms,
                weights=[0.84, 0.05, 0.11],
                minimums=[250.0, 90.0, 120.0],
            )
            return [
                StartupStage("spinup", spinup_ms, 0.0, self.target_rpm, calibration_pulses=0),
                StartupStage("head_load", head_load_ms, self.target_rpm, self.target_rpm, calibration_pulses=0, head_load=True),
                StartupStage("servo_lock", servo_lock_ms, self.target_rpm, self.target_rpm, calibration_pulses=1),
            ]
        if origin == "low_rpm_idle":
            total_ms = self.low_rpm_to_ready_ms
            rpm_recover_ms, head_load_ms, servo_lock_ms = self._allocate_stage_durations(
                total_ms,
                weights=[0.72, 0.12, 0.16],
                minimums=[250.0, 120.0, 120.0],
            )
            return [
                StartupStage("rpm_recover", rpm_recover_ms, self.low_rpm_rpm, self.target_rpm, calibration_pulses=0),
                StartupStage("head_load", head_load_ms, self.target_rpm, self.target_rpm, head_load=True),
                StartupStage("servo_lock", servo_lock_ms, self.target_rpm, self.target_rpm, calibration_pulses=1),
            ]
        if origin == "unloaded_idle":
            total_ms = self.unload_to_ready_ms
            head_load_ms, servo_lock_ms = self._allocate_stage_durations(
                total_ms,
                weights=[0.58, 0.42],
                minimums=[120.0, 80.0],
            )
            return [
                StartupStage("head_load", head_load_ms, self.target_rpm, self.target_rpm, calibration_pulses=0, head_load=True),
                StartupStage("servo_lock", servo_lock_ms, self.target_rpm, self.target_rpm, calibration_pulses=1),
            ]
        return []

    def _run_startup_sequence(self, origin: str) -> float:
        stages = self._build_startup_sequence(origin)
        cancel_event = threading.Event()
        self._run_transition_sequence("startup", origin, stages, cancel_event)
        return sum(stage.duration_ms for stage in stages)

    def _transition_to_active_if_needed(self) -> tuple[float, str | None]:
        ready_info = self._wait_for_ready_poll()
        return ready_info["startup_ms"], ready_info["startup_origin"]

    def submit_physical_access(
        self,
        lba: int,
        size_bytes: int,
        is_write: bool,
        op_kind: str = "data",
        force_unit_access: bool = False,
        queue_depth: int = 1,
    ) -> Stats:
        block_count = max(1, math.ceil(size_bytes / self.block_bytes))
        ready_info = self._wait_for_ready_poll()

        with self.io_lock:
            with self.lock:
                now = time.monotonic()
                cache_latency_ms = 0.0
                cache_hit = False
                partial_hit = False

                if not is_write and op_kind == "data":
                    overlap = self._cache_overlap_blocks(lba, block_count, now)
                    if overlap >= block_count:
                        audio.emit_telemetry(
                            self.current_rpm,
                            is_seq=True,
                            queue_depth=queue_depth,
                            op_kind=op_kind,
                        )
                        self.last_access_time = now
                        return {
                            "total_ms": 0.03 + ready_info["ready_poll_ms"],
                            "cache_hit": True,
                            "partial_hit": False,
                            "cyl": self.current_cyl,
                            "head": self.current_head,
                            "startup_ms": ready_info["startup_ms"],
                            "startup_origin": ready_info["startup_origin"],
                            "ready_poll_ms": ready_info["ready_poll_ms"],
                            "ready_poll_count": ready_info["ready_poll_count"],
                        }
                    if overlap > 0:
                        partial_hit = True
                        cache_latency_ms = 0.05
                        lba += overlap
                        block_count -= overlap
                        size_bytes = block_count * self.block_bytes

                total_latency_ms, target_cyl, target_head, target_sector, distance, zone = self._calculate_position_latency(
                    lba,
                    block_count,
                )
                command_overhead_ms = self._command_overhead_for(op_kind, queue_depth)
                flush_ms = self.flush_penalty_ms if (force_unit_access or op_kind == "flush") else 0.0
                total_latency_ms += ready_info["ready_poll_ms"] + flush_ms + command_overhead_ms + cache_latency_ms

                audio.emit_telemetry(
                    self.current_rpm,
                    seek_trigger=(distance > 0 or target_head != self.current_head),
                    seek_dist=distance,
                    is_seq=(distance == 0 and op_kind == "data" and not is_write),
                    queue_depth=queue_depth,
                    op_kind=op_kind,
                    is_flush=(force_unit_access or op_kind == "flush"),
                    is_spinup=ready_info["ready_poll_ms"] > 0.0,
                )

            self._sleep_ms(total_latency_ms - ready_info["ready_poll_ms"])

            with self.lock:
                final_lba = min(self.addressable_blocks - 1, lba + max(block_count - 1, 0))
                final_cyl, final_head, final_sector, final_zone = self._lba_to_chs(final_lba)
                self.current_cyl = final_cyl
                self.current_head = final_head
                self.current_sector = (final_sector + 1) % final_zone.blocks_per_track
                self.last_access_time = time.monotonic()
                self.power_state = "active"

                if not is_write and op_kind == "data":
                    self._remember_read(lba, block_count)
                else:
                    self.last_read_end_lba = -1

                return {
                    "total_ms": total_latency_ms,
                    "cyl": target_cyl,
                    "head": target_head,
                    "cache_hit": cache_hit,
                    "partial_hit": partial_hit,
                    "transfer_rate_mbps": zone.transfer_rate_mbps,
                    "startup_ms": ready_info["startup_ms"],
                    "startup_origin": ready_info["startup_origin"],
                    "ready_poll_ms": ready_info["ready_poll_ms"],
                    "ready_poll_count": ready_info["ready_poll_count"],
                }

    def _background_tasks_loop(self) -> None:
        while self.running:
            time.sleep(0.05)
            park = False
            calibrate = False
            should_spindown = False
            rpm = 0.0
            with self.lock:
                idle_s = time.monotonic() - self.last_access_time
                rpm = self.current_rpm
                transition_active = self.transition_kind is not None

                if idle_s >= self.unload_after_s and self.power_state == "active" and not transition_active:
                    self.power_state = "unloaded_idle"
                    self.heads_loaded = False
                    self.load_unload_count += 1
                    park = True
                elif idle_s >= self.low_rpm_after_s and self.power_state == "unloaded_idle" and not transition_active:
                    self.power_state = "low_rpm_idle"
                    self.current_rpm = self.low_rpm_rpm
                    rpm = self.current_rpm
                elif idle_s >= 8.0 and self.power_state == "active" and not transition_active:
                    calibrate = True

                if (
                    idle_s >= self.standby_after_s
                    and self.power_state in {"low_rpm_idle", "unloaded_idle", "active"}
                    and not transition_active
                ):
                    should_spindown = True

            if park:
                audio.emit_telemetry(rpm, is_park=True)
            elif calibrate:
                audio.emit_telemetry(rpm, is_cal=True)
            if should_spindown:
                self.begin_async_spindown()


class VirtualHDD:
    def __init__(
        self,
        backing_dir: str,
        latency_scale: float = 1.0,
        cold_start: bool = False,
        async_power_on: bool = False,
        drive_profile: str | DriveProfile | None = None,
        acoustic_profile: str | AcousticProfile | None = None,
    ) -> None:
        self.drive_profile, self.acoustic_profile = resolve_selected_profiles(drive_profile, acoustic_profile)
        audio.configure_profiles(self.drive_profile, self.acoustic_profile)
        self.fs = FileSystemSimulator()
        self.model = HDDLatencyModel(
            addressable_blocks=self.fs.total_blocks,
            block_bytes=self.fs.block_size,
            latency_scale=latency_scale,
            start_ready=not cold_start,
            drive_profile=self.drive_profile,
        )
        self.backing_dir = backing_dir
        self.scheduler = None
        self.lookup_cache = {}
        self.lookup_cache_ttl_s = 0.35
        self.copy_chunk_bytes = 1024 * 1024

        self.writeback_lock = threading.Lock()
        self.writeback_queue = deque()
        self.writeback_bytes = 0
        self.inflight_writebacks = 0
        self.writeback_idle_event = threading.Event()
        self.writeback_idle_event.set()
        self.running = True
        self.writeback_thread = threading.Thread(target=self._writeback_loop, daemon=True)
        self.writeback_thread.start()
        if async_power_on and cold_start:
            self.begin_async_power_on()

    def stop(self) -> None:
        try:
            self.sync_all()
        finally:
            self.running = False
            self.writeback_thread.join(timeout=2.0)
            if self.scheduler and hasattr(self.scheduler, "stop"):
                self.scheduler.stop()
            self.model.stop()

    def set_scheduler(self, scheduler: Any) -> None:
        self.scheduler = scheduler

    def begin_async_power_on(self) -> None:
        self.model.begin_async_startup()

    def _real_path(self, path: str) -> str:
        normalized = self.fs._normalize_path(path)
        return os.path.join(self.backing_dir, normalized.lstrip("/").replace("/", os.sep))

    def _invalidate_lookup_prefix(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        for cached_path in list(self.lookup_cache):
            if cached_path == normalized or cached_path.startswith(f"{normalized}/"):
                self.lookup_cache.pop(cached_path, None)

    def _invalidate_lookup(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        self._invalidate_lookup_prefix(normalized)
        self.lookup_cache.pop(self.fs._parent_dir(normalized), None)

    def _ensure_known_path(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        if normalized in self.fs.files or normalized in self.fs.directories:
            return

        real_path = self._real_path(normalized)
        if os.path.isdir(real_path):
            self.fs.materialize_existing_directory(normalized)
        elif os.path.isfile(real_path):
            self.fs.materialize_existing_file(normalized, os.path.getsize(real_path))

    def _materialize_directory_children(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        real_path = self._real_path(normalized)
        if not os.path.isdir(real_path):
            return
        self.fs.materialize_existing_directory(normalized)
        for entry in os.scandir(real_path):
            child_path = self.fs._normalize_path(f"{normalized}/{entry.name}")
            if entry.is_dir():
                self.fs.materialize_existing_directory(child_path)
            elif entry.is_file():
                self.fs.materialize_existing_file(child_path, entry.stat().st_size)

    def ensure_tree_known(self, path: str) -> None:
        normalized = self.fs._normalize_path(path)
        self._ensure_known_path(normalized)
        real_path = self._real_path(normalized)
        if not os.path.isdir(real_path):
            return

        for current_root, dirnames, filenames in os.walk(real_path):
            rel_root = os.path.relpath(current_root, self.backing_dir)
            virtual_root = "/" if rel_root == "." else self.fs._normalize_path(rel_root.replace(os.sep, "/"))
            self.fs.materialize_existing_directory(virtual_root)
            for dirname in dirnames:
                self.fs.materialize_existing_directory(f"{virtual_root}/{dirname}")
            for filename in filenames:
                file_path = os.path.join(current_root, filename)
                self.fs.materialize_existing_file(f"{virtual_root}/{filename}", os.path.getsize(file_path))

    def _refresh_writeback_idle_state_locked(self) -> None:
        if self.writeback_queue or self.inflight_writebacks:
            self.writeback_idle_event.clear()
        else:
            self.writeback_idle_event.set()

    def _wait_for_writeback_idle(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            with self.writeback_lock:
                if not self.writeback_queue and self.inflight_writebacks == 0:
                    self.writeback_idle_event.set()
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for writeback to go idle")
            self.writeback_idle_event.wait(timeout=min(remaining, 0.1))

    def _run_ops(
        self,
        operations: list[IOOperation],
        is_write: bool,
        force_unit_access: bool = False,
    ) -> Stats:
        total_stats = {
            "total_ms": 0.0,
            "extents": len([operation for operation in operations if operation.kind == "data"]),
            "cyl": "-",
            "head": "-",
            "cache_hit": True if operations else False,
            "partial_hit": False,
            "startup_ms": 0.0,
            "startup_origin": None,
            "ready_poll_ms": 0.0,
            "ready_poll_count": 0,
        }

        for operation in operations:
            size_bytes = operation.block_count * self.fs.block_size
            if self.scheduler:
                request_id = self.scheduler.submit_bio(
                    operation.lba,
                    size_bytes,
                    is_write,
                    op_kind=operation.kind,
                    sync=force_unit_access,
                )
                result = self.scheduler.wait_for_completion(request_id)
            else:
                result = self.model.submit_physical_access(
                    operation.lba,
                    size_bytes,
                    is_write,
                    op_kind=operation.kind,
                    force_unit_access=force_unit_access,
                )

            total_stats["total_ms"] += result["total_ms"]
            if "cyl" in result:
                total_stats["cyl"] = result["cyl"]
            if "head" in result:
                total_stats["head"] = result["head"]
            if not result.get("cache_hit"):
                total_stats["cache_hit"] = False
            if result.get("partial_hit"):
                total_stats["partial_hit"] = True
            total_stats["startup_ms"] += result.get("startup_ms", 0.0)
            if total_stats["startup_origin"] is None and result.get("startup_origin") is not None:
                total_stats["startup_origin"] = result["startup_origin"]
            total_stats["ready_poll_ms"] += result.get("ready_poll_ms", 0.0)
            total_stats["ready_poll_count"] += result.get("ready_poll_count", 0)

        total_stats["type"] = "WRITE" if is_write else "READ"
        return total_stats

    def _enqueue_writeback(self, operations: list[IOOperation]) -> float:
        blocked_ms = 0.0
        with self.writeback_lock:
            for operation in operations:
                size_bytes = operation.block_count * self.fs.block_size
                self.writeback_queue.append(
                    DirtyWrite(
                        lba=operation.lba,
                        size_bytes=size_bytes,
                        op_kind=operation.kind,
                        enqueued_at=time.monotonic(),
                    )
                )
                self.writeback_bytes += size_bytes
                self.model.note_cached_write(operation.lba, size_bytes)
            self._refresh_writeback_idle_state_locked()

        if self.writeback_bytes > self.model.write_cache_bytes:
            blocked_ms += self._drain_write_cache(target_bytes=self.model.write_cache_bytes // 2)
        return blocked_ms

    def _dequeue_writeback_batch(self, force: bool = False, max_items: int = 32) -> list[DirtyWrite]:
        batch: list[DirtyWrite] = []
        now = time.monotonic()
        with self.writeback_lock:
            while self.writeback_queue and len(batch) < max_items:
                head = self.writeback_queue[0]
                age = now - head.enqueued_at
                cache_pressure = self.writeback_bytes > int(self.model.write_cache_bytes * 0.75)
                if not force and not cache_pressure and age < self.model.dirty_expire_s:
                    break
                batch.append(self.writeback_queue.popleft())
                self.writeback_bytes -= head.size_bytes
            self._refresh_writeback_idle_state_locked()
        return batch

    def _drain_write_cache(self, target_bytes: int = 0) -> float:
        total_ms = 0.0
        while True:
            batch = self._dequeue_writeback_batch(force=True)
            if not batch:
                break
            operations = [
                IOOperation(
                    dirty_write.lba,
                    max(1, math.ceil(dirty_write.size_bytes / self.fs.block_size)),
                    dirty_write.op_kind,
                    "writeback",
                )
                for dirty_write in batch
            ]
            total_ms += self._run_ops(operations, is_write=True)["total_ms"]
            if self.writeback_bytes <= target_bytes:
                break
        if target_bytes <= 0:
            self._wait_for_writeback_idle()
        return total_ms

    def _writeback_loop(self) -> None:
        while self.running:
            batch = self._dequeue_writeback_batch(force=False)
            if batch:
                with self.writeback_lock:
                    self.inflight_writebacks += 1
                    self._refresh_writeback_idle_state_locked()
                operations = [
                    IOOperation(
                        dirty_write.lba,
                        max(1, math.ceil(dirty_write.size_bytes / self.fs.block_size)),
                        dirty_write.op_kind,
                        "writeback",
                    )
                    for dirty_write in batch
                ]
                try:
                    self._run_ops(operations, is_write=True)
                finally:
                    with self.writeback_lock:
                        self.inflight_writebacks -= 1
                        self._refresh_writeback_idle_state_locked()
                continue
            time.sleep(0.05)

    def sync_all(self) -> float:
        total_ms = self._drain_write_cache(target_bytes=0)
        self._wait_for_writeback_idle()
        return total_ms

    def reset_runtime_state(self) -> None:
        self.sync_all()
        self.lookup_cache.clear()
        self.model.reset_caches()

    def _empty_stats(self, op_type: str, total_ms: float = 0.01) -> Stats:
        return {"total_ms": total_ms, "cache_hit": True, "extents": 0, "type": op_type}

    def _merge_stats(self, op_type: str, *results: Stats) -> Stats:
        combined = {
            "total_ms": 0.0,
            "cache_hit": True,
            "partial_hit": False,
            "extents": 0,
            "cyl": "-",
            "head": "-",
            "startup_ms": 0.0,
            "startup_origin": None,
            "ready_poll_ms": 0.0,
            "ready_poll_count": 0,
            "type": op_type,
        }
        saw_result = False
        for result in results:
            if not result:
                continue
            saw_result = True
            combined["total_ms"] += result.get("total_ms", 0.0)
            combined["extents"] += result.get("extents", 0)
            if "cyl" in result:
                combined["cyl"] = result["cyl"]
            if "head" in result:
                combined["head"] = result["head"]
            if not result.get("cache_hit", False):
                combined["cache_hit"] = False
            if result.get("partial_hit"):
                combined["partial_hit"] = True
            combined["startup_ms"] += result.get("startup_ms", 0.0)
            if combined["startup_origin"] is None and result.get("startup_origin") is not None:
                combined["startup_origin"] = result["startup_origin"]
            combined["ready_poll_ms"] += result.get("ready_poll_ms", 0.0)
            combined["ready_poll_count"] += result.get("ready_poll_count", 0)
        if not saw_result:
            return self._empty_stats(op_type)
        return combined

    def _apply_buffered_write(
        self,
        operations: list[IOOperation],
        data_extent_count: int,
        sync: bool = False,
    ) -> Stats:
        if not operations:
            return self._empty_stats("WRITE")

        if sync:
            sync_total = self.sync_all()
            stats = self._run_ops(operations, is_write=True, force_unit_access=True)
            stats["total_ms"] += sync_total
            stats["type"] = "WRITE"
            stats["extents"] = data_extent_count
            return stats

        journal_ops = [operation for operation in operations if operation.kind == "journal"]
        buffered_ops = [operation for operation in operations if operation.kind != "journal"]
        stats = self._run_ops(journal_ops, is_write=True) if journal_ops else {
            "total_ms": 0.0,
            "cache_hit": True,
            "cyl": "-",
            "head": "-",
            "partial_hit": False,
            "startup_ms": 0.0,
            "startup_origin": None,
            "ready_poll_ms": 0.0,
            "ready_poll_count": 0,
        }
        blocked_ms = self._enqueue_writeback(buffered_ops)
        stats["total_ms"] += 0.08 + blocked_ms
        stats["cache_hit"] = stats.get("cache_hit", True) and blocked_ms == 0.0
        stats["type"] = "WRITE"
        stats["extents"] = data_extent_count
        return stats

    def lookup_path(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        now = time.monotonic()
        if self.lookup_cache.get(path, 0.0) > now:
            return self._empty_stats("LOOKUP", total_ms=0.02)
        operations = self.fs.lookup(path)
        stats = self._run_ops(operations, is_write=False)
        self.lookup_cache[path] = now + self.lookup_cache_ttl_s
        stats["type"] = "LOOKUP"
        return stats

    def list_directory(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        self._materialize_directory_children(path)
        operations = self.fs.list_directory(path)
        if not operations:
            return self._empty_stats("READDIR")
        stats = self._run_ops(operations, is_write=False)
        stats["type"] = "READDIR"
        stats["extents"] = 0
        return stats

    def create_directory(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        operations = self.fs.create_directory(path)
        self._invalidate_lookup(path)
        if not operations:
            return self._empty_stats("MKCOL")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        stats["type"] = "MKCOL"
        stats["extents"] = 0
        return stats

    def refresh_directory(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        operations = self.fs.update_directory(path)
        if not operations:
            return self._empty_stats("COPY")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        stats["type"] = "COPY"
        stats["extents"] = 0
        return stats

    def create_empty_file(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        operations = self.fs.create_empty_file(path)
        self._invalidate_lookup(path)
        if not operations:
            return self._empty_stats("CREATE")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        stats["type"] = "CREATE"
        stats["extents"] = 0
        return stats

    def copy_file(self, source_path: str, dest_path: str) -> Stats:
        source_path = self.fs._normalize_path(source_path)
        dest_path = self.fs._normalize_path(dest_path)
        self._ensure_known_path(source_path)
        self._ensure_known_path(self.fs._parent_dir(dest_path))

        if source_path not in self.fs.files:
            return self._empty_stats("COPY")
        if dest_path in self.fs.directories:
            raise IsADirectoryError(dest_path)
        if dest_path in self.fs.files:
            self.delete_path(dest_path)

        source_inode = self.fs.files[source_path]
        source_size = source_inode.size
        if source_size <= 0:
            return self._merge_stats("COPY", self.create_empty_file(dest_path))

        chunk_size = max(self.fs.block_size, self.copy_chunk_bytes)
        copy_stats = []
        for offset in range(0, source_size, chunk_size):
            length = min(chunk_size, source_size - offset)
            read_ops = self.fs.read(source_path, offset, length)
            read_stats = self._run_ops(read_ops, is_write=False) if read_ops else self._empty_stats("READ")

            write_ops = self.fs.write(dest_path, offset, length)
            data_extent_count = len([operation for operation in write_ops if operation.kind == "data"])
            write_stats = self._apply_buffered_write(write_ops, data_extent_count, sync=False)
            copy_stats.append(self._merge_stats("COPY", read_stats, write_stats))

        self._invalidate_lookup(dest_path)
        return self._merge_stats("COPY", *copy_stats)

    def prepare_overwrite(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        if path not in self.fs.files:
            return {"total_ms": 0.0, "cache_hit": True, "extents": 0, "type": "TRUNCATE"}
        self._invalidate_lookup(path)
        self.sync_all()
        operations = self.fs.truncate(path, size=0)
        if not operations:
            return {"total_ms": 0.0, "cache_hit": True, "extents": 0, "type": "TRUNCATE"}
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        stats["type"] = "TRUNCATE"
        stats["extents"] = 0
        return stats

    def access_file(
        self,
        path: str,
        offset: int,
        length: int,
        is_write: bool = False,
        sync: bool = False,
    ) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        operations = self.fs.write(path, offset, length) if is_write else self.fs.read(path, offset, length)
        data_extent_count = len([operation for operation in operations if operation.kind == "data"])

        if not operations:
            return self._empty_stats("WRITE" if is_write else "READ")

        if is_write:
            self._invalidate_lookup(path)
            return self._apply_buffered_write(operations, data_extent_count, sync=sync)

        stats = self._run_ops(operations, is_write=False)
        stats["extents"] = data_extent_count
        return stats

    def rename_path(self, source_path: str, dest_path: str) -> Stats:
        source_path = self.fs._normalize_path(source_path)
        dest_path = self.fs._normalize_path(dest_path)
        self._ensure_known_path(self.fs._parent_dir(dest_path))
        operations = self.fs.rename(source_path, dest_path)
        self._invalidate_lookup(source_path)
        self._invalidate_lookup(dest_path)
        self._invalidate_lookup_prefix(source_path)
        if not operations:
            return self._empty_stats("MOVE")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        stats["type"] = "MOVE"
        stats["extents"] = 0
        return stats

    def delete_path(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        self._ensure_known_path(path)
        self._invalidate_lookup(path)
        operations = self.fs.delete(path)
        if not operations:
            return self._empty_stats("DELETE")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        stats["type"] = "DELETE"
        stats["extents"] = 0
        return stats

    def delete_directory(self, path: str) -> Stats:
        path = self.fs._normalize_path(path)
        operations = self.fs.delete_directory(path)
        self._invalidate_lookup(path)
        self._invalidate_lookup_prefix(path)
        if not operations:
            return self._empty_stats("DELETE")
        stats = self._run_ops(operations, is_write=True, force_unit_access=True)
        stats["type"] = "DELETE"
        stats["extents"] = 0
        return stats


# Backward compatibility for the old typo'd class name.
HDDLatenyModel = HDDLatencyModel
