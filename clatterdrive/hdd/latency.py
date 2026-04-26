from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any

from .core import (
    BackgroundDecision,
    CacheSpan,
    CacheState,
    HDDCoreConfig,
    MechanicalState,
    OperationStats,
    ReadyPollPlan,
    StartupStage,
    StartupTracePoint,
    TransitionState,
    Zone,
    allocate_stage_durations,
    background_decision,
    build_low_rpm_sequence,
    build_resume_sequence,
    build_spindown_sequence,
    build_spindown_trace_from_stages,
    build_startup_sequence,
    build_startup_trace_from_stages,
    build_zones,
    cache_overlap_blocks,
    calculate_position_latency,
    command_overhead_for,
    estimated_lba,
    remember_cached_write,
    remember_read,
    resolve_startup_plan,
    simulate_rotational_transition,
    startup_trace_step_ms,
    transfer_ms_for_span,
    zone_for_cyl,
    zone_for_lba,
)
from ..profiles import DriveProfile, resolve_drive_profile_from_env
from ..runtime.deps import RuntimeDeps
from ..storage_events import NullStorageEventSink, StorageEvent, StorageEventSink


Stats = OperationStats

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
        event_sink: StorageEventSink | None = None,
        deps: RuntimeDeps | None = None,
        enable_retry_recovery: bool = True,
        enable_background_scan: bool = True,
    ) -> None:
        self.deps = deps or RuntimeDeps()
        self.clock = self.deps.clock
        self.sleeper = self.deps.sleeper
        resolved_drive = resolve_drive_profile_from_env(drive_profile, env=self.deps.env)
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
        self.target_rpm: float = float(rpm)
        self.current_rpm: float = float(rpm)
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
        self.low_rpm_rpm: float = float(
            low_rpm_rpm or max(int(self.target_rpm * 0.875), int(self.target_rpm - 900))
        )
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
        self.core_config = HDDCoreConfig(
            drive_profile=self.drive_profile,
            block_bytes=self.block_bytes,
            addressable_blocks=self.addressable_blocks,
            target_rpm=self.target_rpm,
            num_heads=self.num_heads,
            ms_per_rotation=self.ms_per_rotation,
            avg_seek_ms=self.avg_seek_ms,
            track_to_track_ms=self.track_to_track_ms,
            settle_ms=self.settle_ms,
            head_switch_ms=self.head_switch_ms,
            read_ahead_blocks=self.read_ahead_blocks,
            low_rpm_rpm=self.low_rpm_rpm,
            unload_to_ready_ms=self.unload_to_ready_ms,
            low_rpm_to_ready_ms=self.low_rpm_to_ready_ms,
            standby_to_ready_ms=self.standby_to_ready_ms,
            power_on_to_ready_ms=self.power_on_to_ready_ms,
            spin_down_ms=self.spin_down_ms,
            blocks_per_track_outer=self.blocks_per_track_outer,
            blocks_per_track_inner=self.blocks_per_track_inner,
            total_cylinders=self.total_cylinders,
            seek_curve_b=self.seek_curve_b,
            zones=tuple(self.zones),
            command_overhead_ms=self.command_overhead_ms,
            command_overhead_by_kind=self.command_overhead_by_kind,
            queue_depth_penalty_ms=self.queue_depth_penalty_ms,
            track_skew_blocks=self.track_skew_blocks,
            cylinder_skew_blocks=self.cylinder_skew_blocks,
            aam_seek_penalty=self.aam_seek_penalty,
        )

        self.current_cyl = 0
        self.current_head = 0
        self.current_sector = 0
        self.last_access_time = self.clock.now()
        self.power_state = "active" if start_ready else "power_on"
        self.load_unload_count = 0
        self.heads_loaded = start_ready
        self.has_completed_power_on = start_ready
        if not start_ready:
            self.current_rpm = 0.0

        self.read_cache: deque[CacheSpan] = deque(maxlen=16)
        self.last_read_end_lba = -1
        self.lock = threading.RLock()
        self.io_lock = threading.Lock()
        self.ready_event = threading.Event()
        if start_ready:
            self.ready_event.set()
        self.transition_thread: threading.Thread | None = None
        self.transition_cancel: threading.Event | None = None
        self.transition_kind: str | None = None
        self.transition_origin: str | None = None
        self.transition_stage: str | None = None
        self.transition_total_ms: float = 0.0
        self.last_startup_origin: str | None = None
        self.last_startup_total_ms: float = 0.0
        self.last_idle_calibration_time = 0.0
        self.event_sink: StorageEventSink = event_sink or NullStorageEventSink()
        self.enable_retry_recovery = enable_retry_recovery
        self.enable_background_scan = enable_background_scan
        self.background_busy_until = 0.0
        self.background_scan_lba = 0
        self.last_background_scan_time = 0.0
        self.background_scan_interval_s = min(max(self.unload_after_s * 0.12, 0.15), 0.45)
        self.background_scan_activation_s = min(
            max(self.unload_after_s * 0.55, 3.0),
            max(self.unload_after_s - 0.5, 3.0),
        )
        self.background_scan_blocks = max(16, min(self.read_ahead_blocks, 256))
        self.running = True
        self.background_thread = threading.Thread(target=self._background_tasks_loop, daemon=True)
        self.background_thread.start()

    def _build_zones(self, outer_rate: float, inner_rate: float, zone_count: int = 8) -> list[Zone]:
        return list(
            build_zones(
                addressable_blocks=self.addressable_blocks,
                total_cylinders=self.total_cylinders,
                num_heads=self.num_heads,
                blocks_per_track_outer=self.blocks_per_track_outer,
                blocks_per_track_inner=self.blocks_per_track_inner,
                outer_rate=outer_rate,
                inner_rate=inner_rate,
                zone_count=zone_count,
            )
        )

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
            self.last_access_time = self.clock.now()
            self.ready_event.clear()
            self._clear_read_cache_locked()
            self.transition_kind = None
            self.transition_origin = None
            self.transition_stage = None
            self.transition_total_ms = 0.0

    def _clear_read_cache_locked(self) -> None:
        self.read_cache.clear()
        self.last_read_end_lba = -1

    def _core_mechanical_state(self) -> MechanicalState:
        return MechanicalState(
            current_rpm=self.current_rpm,
            current_cyl=self.current_cyl,
            current_head=self.current_head,
            current_sector=self.current_sector,
            power_state=self.power_state,
            heads_loaded=self.heads_loaded,
            has_completed_power_on=self.has_completed_power_on,
            last_access_time=self.last_access_time,
            load_unload_count=self.load_unload_count,
        )

    def _core_transition_state(self) -> TransitionState:
        return TransitionState(
            kind=self.transition_kind,
            origin=self.transition_origin,
            stage=self.transition_stage,
            total_ms=self.transition_total_ms,
        )

    def _core_cache_state(self) -> CacheState:
        return CacheState(spans=tuple(self.read_cache), last_read_end_lba=self.last_read_end_lba)

    def _apply_cache_state(self, cache_state: CacheState) -> None:
        self.read_cache = deque(cache_state.spans, maxlen=16)
        self.last_read_end_lba = cache_state.last_read_end_lba

    def _build_resume_sequence(self, start_rpm: float, heads_loaded: bool) -> list[StartupStage]:
        return build_resume_sequence(self.core_config, start_rpm, heads_loaded)

    def _resolve_startup_plan_locked(self) -> tuple[str | None, list[StartupStage], float]:
        return resolve_startup_plan(self.core_config, self._core_mechanical_state())

    def _build_spindown_sequence_locked(self) -> list[StartupStage]:
        return build_spindown_sequence(self.core_config, self._core_mechanical_state())

    def _build_low_rpm_sequence_locked(self) -> list[StartupStage]:
        return build_low_rpm_sequence(self.core_config, self.current_rpm)

    def _startup_trace_step_ms(self, total_ms: float) -> float:
        return startup_trace_step_ms(total_ms)

    def _rotational_drag_terms(self) -> tuple[float, float]:
        from .core import rotational_drag_terms

        return rotational_drag_terms(self.core_config)

    def _simulate_rotational_transition(
        self,
        start_rpm: float,
        end_rpm: float,
        duration_ms: float,
        step_ms: float,
    ) -> list[float]:
        return simulate_rotational_transition(self.core_config, start_rpm, end_rpm, duration_ms, step_ms)

    def _simulate_spin_ramp(
        self,
        start_rpm: float,
        end_rpm: float,
        duration_ms: float,
        step_ms: float,
    ) -> list[float]:
        return simulate_rotational_transition(self.core_config, start_rpm, end_rpm, duration_ms, step_ms)

    def _simulate_spin_decay(
        self,
        start_rpm: float,
        end_rpm: float,
        duration_ms: float,
        step_ms: float,
    ) -> list[float]:
        return simulate_rotational_transition(self.core_config, start_rpm, end_rpm, duration_ms, step_ms)

    def _build_startup_trace_from_stages(
        self,
        origin: str,
        stages: list[StartupStage],
        *,
        step_ms: float | None = None,
        initial_heads_loaded: bool = False,
    ) -> list[StartupTracePoint]:
        return build_startup_trace_from_stages(
            self.core_config,
            origin,
            stages,
            step_ms=step_ms,
            initial_heads_loaded=initial_heads_loaded,
        )

    def _build_spindown_trace_from_stages(
        self,
        stages: list[StartupStage],
        *,
        step_ms: float | None = None,
        initial_heads_loaded: bool = True,
    ) -> list[StartupTracePoint]:
        return build_spindown_trace_from_stages(
            self.core_config,
            stages,
            step_ms=step_ms,
            initial_heads_loaded=initial_heads_loaded,
        )

    def _run_startup_trace(
        self,
        origin: str,
        stages: list[StartupStage],
        cancel_event: threading.Event,
    ) -> bool:
        with self.lock:
            initial_heads_loaded = self.heads_loaded
        trace = self._build_startup_trace_from_stages(
            origin,
            stages,
            initial_heads_loaded=initial_heads_loaded,
        )
        previous_time_ms = 0.0
        for point in trace:
            if cancel_event.is_set() or not self.running:
                return False
            with self.lock:
                if self.transition_cancel is not cancel_event:
                    return False
                self.transition_stage = point.stage
                self.current_rpm = point.rpm
                self.power_state = "starting"
                if point.head_load_event and not self.heads_loaded:
                    self.heads_loaded = True
                elif not point.heads_loaded:
                    self.heads_loaded = False

            motion_duration_ms, settle_duration_ms = (
                self._actuator_profile(
                    seek_distance=point.seek_distance,
                    queue_depth=1,
                    op_kind="metadata",
                    is_flush=False,
                    is_cal=point.is_calibration,
                    is_park=False,
                    is_sequential=False,
                )
            )
            if point.seek_distance > 0.0:
                self._publish_event(
                    point.rpm,
                    target_rpm=self.target_rpm,
                    seek_trigger=True,
                    seek_dist=point.seek_distance,
                    queue_depth=1,
                    op_kind="metadata",
                    is_spinup=point.is_spinup,
                    motion_duration_ms=motion_duration_ms,
                    settle_duration_ms=settle_duration_ms,
                )
            if point.is_calibration:
                self._publish_event(
                    point.rpm,
                    target_rpm=self.target_rpm,
                    queue_depth=1,
                    op_kind="metadata",
                    is_cal=True,
                    is_spinup=point.is_spinup,
                    motion_duration_ms=motion_duration_ms,
                    settle_duration_ms=settle_duration_ms,
                )
            if point.seek_distance <= 0.0 and not point.is_calibration:
                self._publish_event(
                    point.rpm,
                    target_rpm=self.target_rpm,
                    queue_depth=1,
                    op_kind="data",
                    is_spinup=point.is_spinup,
                )

            self._sleep_ms(max(point.time_ms - previous_time_ms, 0.0))
            previous_time_ms = point.time_ms
        return True

    def _run_spindown_trace(
        self,
        kind: str,
        stages: list[StartupStage],
        cancel_event: threading.Event,
    ) -> bool:
        with self.lock:
            initial_heads_loaded = self.heads_loaded
        trace = self._build_spindown_trace_from_stages(
            stages,
            initial_heads_loaded=initial_heads_loaded,
        )
        previous_time_ms = 0.0
        for point in trace:
            if cancel_event.is_set() or not self.running:
                return False
            with self.lock:
                if self.transition_cancel is not cancel_event:
                    return False
                self.transition_stage = point.stage
                self.current_rpm = point.rpm
                self.power_state = "spinning_down" if kind == "spindown" else "slowing_to_low_rpm"
                if self.heads_loaded and not point.heads_loaded:
                    self.heads_loaded = False
                    self.load_unload_count += 1

            self._publish_event(
                point.rpm,
                target_rpm=0.0 if kind == "spindown" else self.low_rpm_rpm,
                is_park=point.park_event,
                queue_depth=1,
                op_kind="metadata" if point.park_event or point.stage == "spindle_lock" else "data",
                motion_duration_ms=2.4 if point.park_event else 0.0,
                settle_duration_ms=1.8 if point.park_event else 0.0,
            )
            self._sleep_ms(max(point.time_ms - previous_time_ms, 0.0))
            previous_time_ms = point.time_ms
        return True

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
                self.last_access_time = self.clock.now()
                self.ready_event.set()
            elif kind == "slowdown":
                self.power_state = "low_rpm_idle"
                self.current_rpm = self.low_rpm_rpm
                self.heads_loaded = False
                self.last_access_time = self.clock.now()
            else:
                self.power_state = "standby"
                self.current_rpm = 0.0
                self.heads_loaded = False
                self.last_access_time = self.clock.now()
                self.ready_event.clear()
                self._clear_read_cache_locked()

    def _run_transition_sequence(
        self,
        kind: str,
        origin: str,
        stages: list[StartupStage],
        cancel_event: threading.Event,
    ) -> None:
        if kind in {"startup", "spindown", "slowdown"}:
            try:
                if kind == "startup":
                    completed = self._run_startup_trace(origin, stages, cancel_event)
                else:
                    completed = self._run_spindown_trace(kind, stages, cancel_event)
                if completed:
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
            return
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
                    self._publish_event(
                        rpm,
                        target_rpm=self.target_rpm if kind == "startup" else (0.0 if kind == "spindown" else self.low_rpm_rpm),
                        seek_trigger=seek_trigger,
                        seek_dist=28 if seek_trigger else 0,
                        is_park=park,
                        queue_depth=1,
                        op_kind="metadata" if (is_cal or seek_trigger or park or stage.head_unload) else "data",
                        is_cal=is_cal,
                        is_spinup=kind == "startup" and stage.name in {"spinup", "rpm_recover", "spindle_unlock"},
                        motion_duration_ms=1.6 if (seek_trigger or is_cal or park) else 0.0,
                        settle_duration_ms=1.1 if (seek_trigger or is_cal or park) else 0.0,
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
            if self.transition_kind is not None and self.transition_kind != "startup" and self.transition_cancel is not None:
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

    def begin_async_low_rpm(self) -> bool:
        with self.lock:
            if self.transition_kind is not None or self.power_state != "unloaded_idle":
                return False
            stages = self._build_low_rpm_sequence_locked()
            if not stages:
                self.power_state = "low_rpm_idle"
                self.current_rpm = self.low_rpm_rpm
                return False
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=self._run_transition_sequence,
                args=("slowdown", "low_rpm_idle", stages, cancel_event),
                daemon=True,
            )
            self.transition_thread = thread
            self.transition_cancel = cancel_event
            self.transition_kind = "slowdown"
            self.transition_origin = "low_rpm_idle"
            self.transition_stage = stages[0].name
            self.transition_total_ms = sum(stage.duration_ms for stage in stages)
            self.power_state = "slowing_to_low_rpm"
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

    def _wait_for_ready_poll(self) -> ReadyPollPlan:
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
            self._publish_event(
                self.current_rpm,
                target_rpm=self.target_rpm,
                queue_depth=1,
                op_kind="metadata",
                is_spinup=self.transition_kind == "startup",
            )
            self._sleep_ms(cycle_ms)
            poll_ms += cycle_ms
            poll_count += 1
            if self.latency_scale <= 0.0:
                self.ready_event.wait(timeout=0.001)

        return ReadyPollPlan(
            startup_ms=startup_ms,
            startup_origin=startup_origin,
            ready_poll_ms=poll_ms,
            ready_poll_count=poll_count,
        )

    def _sleep_ms(self, latency_ms: float) -> None:
        if self.latency_scale <= 0.0:
            return
        self.sleeper.sleep(max(latency_ms, 0.0) * self.latency_scale / 1000.0)

    def get_estimated_lba(self) -> int:
        return estimated_lba(self.core_config, self._core_mechanical_state())

    def _zone_for_lba(self, lba: int) -> Zone:
        return zone_for_lba(self.core_config, lba)

    def _zone_for_cyl(self, cyl: int) -> Zone:
        return zone_for_cyl(self.core_config, cyl)

    def _lba_to_chs(self, lba: int) -> tuple[int, int, int, Zone]:
        from .core import lba_to_chs

        return lba_to_chs(self.core_config, lba)

    def _cache_overlap_blocks(self, lba: int, blocks: int, now: float) -> int:
        from .core import prune_cache

        pruned = prune_cache(self._core_cache_state(), now)
        self._apply_cache_state(pruned)
        return cache_overlap_blocks(pruned, lba, blocks)

    def _remember_read(self, lba: int, blocks: int) -> None:
        cache_state = remember_read(self.core_config, self._core_cache_state(), lba, blocks, self.clock.now())
        self._apply_cache_state(cache_state)

    def note_cached_write(self, lba: int, size_bytes: int) -> None:
        cache_state = remember_cached_write(
            self.core_config,
            self._core_cache_state(),
            lba,
            size_bytes,
            self.clock.now(),
        )
        self._apply_cache_state(cache_state)

    def _calculate_position_latency(
        self,
        target_lba: int,
        block_count: int,
    ) -> tuple[float, int, int, int, int, Zone]:
        return calculate_position_latency(self.core_config, self._core_mechanical_state(), target_lba, block_count)

    def _transfer_ms_for_span(self, start_lba: int, block_count: int) -> float:
        return transfer_ms_for_span(self.core_config, start_lba, block_count)

    def _command_overhead_for(self, op_kind: str, queue_depth: int) -> float:
        return command_overhead_for(self.core_config, op_kind, queue_depth)

    def _allocate_stage_durations(
        self,
        total_ms: float,
        weights: list[float],
        minimums: list[float],
    ) -> list[float]:
        return allocate_stage_durations(total_ms, weights, minimums)

    def _actuator_profile(
        self,
        *,
        seek_distance: float,
        queue_depth: int,
        op_kind: str,
        is_flush: bool,
        is_cal: bool = False,
        is_park: bool = False,
        head_switch: bool = False,
        is_sequential: bool = False,
    ) -> tuple[float, float]:
        stroke = min(max(seek_distance / max(self.total_cylinders, 1), 0.0), 1.0)
        queue_scale = 1.0 + 0.05 * max(queue_depth - 1, 0)
        if is_park:
            return 2.6, 2.0
        if is_cal:
            return 1.4, 1.2
        if seek_distance <= 0.0 and not head_switch:
            return 0.0, 0.0

        move_ms = self.track_to_track_ms + self.seek_curve_b * math.sqrt(max(seek_distance, 0.0)) + self.settle_ms
        if head_switch and seek_distance <= 0.0:
            move_ms = max(move_ms, self.head_switch_ms + self.settle_ms)
        duration_ms = max(1.2, 0.55 * move_ms + 0.45 * self.settle_ms) * queue_scale
        settle_ms = max(0.9, self.settle_ms * (3.4 if is_flush else 2.8))
        if is_flush:
            settle_ms *= 1.2
        if is_sequential:
            duration_ms *= 0.85
        settle_ms += 0.6 * math.sqrt(max(stroke, 0.0))
        return duration_ms, settle_ms

    def _publish_event(
        self,
        rpm: float,
        *,
        target_rpm: float | None = None,
        seek_trigger: bool = False,
        seek_dist: float = 0.0,
        is_seq: bool = False,
        is_park: bool = False,
        is_cal: bool = False,
        queue_depth: int = 1,
        op_kind: str = "data",
        is_flush: bool = False,
        is_spinup: bool = False,
        servo_mode: str | None = None,
        motion_duration_ms: float = 0.0,
        settle_duration_ms: float = 0.0,
        size_bytes: int = 0,
        block_count: int = 0,
        extent_count: int = 0,
        transfer_ms: float = 0.0,
        directory_entry_count: int = 0,
        fragmentation_score: int = 0,
    ) -> None:
        if servo_mode is None:
            servo_mode = "track" if is_seq else None
        if is_park:
            servo_mode = "park"
        elif seek_trigger:
            servo_mode = "seek"
        elif is_cal:
            servo_mode = "calibration"

        transfer_activity = {
            "metadata": 0.24,
            "journal": 0.42,
            "data": 0.60,
            "writeback": 0.66,
            "flush": 0.82,
            "background": 0.28,
        }.get(op_kind, 0.40)
        if is_seq:
            transfer_activity *= 1.10
        if is_flush:
            transfer_activity *= 1.14
        transfer_activity *= 1.0 + 0.05 * max(queue_depth - 1, 0)

        self.event_sink.publish_event(
            StorageEvent(
                rpm=rpm,
                emitted_at=self.clock.now(),
                target_rpm=rpm if target_rpm is None else target_rpm,
                queue_depth=queue_depth,
                op_kind=op_kind,
                is_sequential=is_seq,
                is_flush=is_flush,
                is_spinup=is_spinup,
                power_state=self.power_state,
                heads_loaded=self.heads_loaded,
                servo_mode=servo_mode,
                track_delta=min(max(seek_dist / max(self.total_cylinders, 1), 0.0), 1.0),
                transfer_activity=transfer_activity,
                motion_duration_ms=motion_duration_ms,
                settle_duration_ms=settle_duration_ms,
                size_bytes=max(0, int(size_bytes)),
                block_count=max(0, int(block_count)),
                extent_count=max(0, int(extent_count)),
                transfer_ms=max(0.0, float(transfer_ms)),
                directory_entry_count=max(0, int(directory_entry_count)),
                fragmentation_score=max(0, int(fragmentation_score)),
                seek_distance=seek_dist,
            )
        )

    def _read_recovery_tail(
        self,
        lba: int,
        block_count: int,
        *,
        is_write: bool,
        op_kind: str,
        queue_depth: int,
    ) -> tuple[float, int]:
        if not self.enable_retry_recovery or is_write or op_kind in {"journal", "flush", "background", "writeback"}:
            return 0.0, 0
        if op_kind not in {"data", "metadata"}:
            return 0.0, 0

        signature = (
            (lba * 1103515245)
            + (block_count * 12345)
            + (queue_depth * 97)
            + int(self.target_rpm)
        ) & 0x1FFF
        if signature == 0:
            retry_count = 3
        elif signature in {1, 2}:
            retry_count = 2
        elif signature in {3, 4, 5, 6}:
            retry_count = 1
        else:
            return 0.0, 0

        zone = self._zone_for_lba(lba)
        zone_scale = self.zones[0].transfer_rate_mbps / max(zone.transfer_rate_mbps, 1e-6)
        recovery_ms = retry_count * self.ms_per_rotation * (0.55 + 0.08 * zone_scale)
        recovery_ms += 0.18 * retry_count + 0.05 * math.log2(block_count + 1)
        return recovery_ms, retry_count

    def _background_scan_step(self, now: float) -> None:
        if not self.enable_background_scan:
            return
        if not self.io_lock.acquire(blocking=False):
            return
        try:
            with self.lock:
                if self.transition_kind is not None or self.power_state != "active":
                    return
                idle_s = now - self.last_access_time
                if idle_s < self.background_scan_activation_s or idle_s >= self.unload_after_s:
                    return
                if now - self.last_background_scan_time < self.background_scan_interval_s:
                    return

                scan_lba = min(self.background_scan_lba, self.addressable_blocks - 1)
                block_count = min(self.background_scan_blocks, self.addressable_blocks - scan_lba)
                if block_count <= 0:
                    self.background_scan_lba = 0
                    return

                total_latency_ms, _target_cyl, target_head, _target_sector, distance, _zone = self._calculate_position_latency(
                    scan_lba,
                    block_count,
                )
                total_latency_ms += self._command_overhead_for("background", 1)
                seek_trigger = distance > 0 or target_head != self.current_head
                motion_duration_ms, settle_duration_ms = self._actuator_profile(
                    seek_distance=distance,
                    queue_depth=1,
                    op_kind="background",
                    is_flush=False,
                    head_switch=(target_head != self.current_head),
                    is_sequential=(distance == 0),
                )

                final_lba = min(self.addressable_blocks - 1, scan_lba + block_count - 1)
                final_cyl, final_head, final_sector, final_zone = self._lba_to_chs(final_lba)
                self.current_cyl = final_cyl
                self.current_head = final_head
                self.current_sector = (final_sector + 1) % final_zone.blocks_per_track
                self.background_scan_lba = 0 if final_lba >= self.addressable_blocks - 1 else final_lba + 1
                self.background_busy_until = max(self.background_busy_until, now + (total_latency_ms / 1000.0))
                self.last_background_scan_time = now
                current_rpm = self.current_rpm

            self._publish_event(
                current_rpm,
                target_rpm=self.target_rpm,
                seek_trigger=seek_trigger,
                seek_dist=distance,
                is_seq=distance == 0,
                queue_depth=1,
                op_kind="background",
                motion_duration_ms=motion_duration_ms,
                settle_duration_ms=settle_duration_ms,
            )
        finally:
            self.io_lock.release()

    def _build_startup_sequence(self, origin: str) -> list[StartupStage]:
        return build_startup_sequence(self.core_config, origin)

    def submit_physical_access(
        self,
        lba: int,
        size_bytes: int,
        is_write: bool,
        op_kind: str = "data",
        force_unit_access: bool = False,
        queue_depth: int = 1,
        extent_count: int = 0,
        directory_entry_count: int = 0,
        fragmentation_score: int = 0,
    ) -> Stats:
        block_count = max(1, math.ceil(size_bytes / self.block_bytes))
        requested_size_bytes = max(0, int(size_bytes))
        reported_extent_count = (
            max(1, int(extent_count))
            if op_kind in {"data", "writeback"}
            else max(0, int(extent_count))
        )
        ready_info = self._wait_for_ready_poll()

        with self.io_lock:
            with self.lock:
                now = self.clock.now()
                cache_latency_ms = 0.0
                cache_hit = False
                partial_hit = False
                maintenance_wait_ms = 0.0
                recovery_ms = 0.0
                retry_count = 0

                if self.background_busy_until > now:
                    maintenance_wait_ms = (self.background_busy_until - now) * 1000.0
                    self.background_busy_until = now

                if not is_write and op_kind == "data":
                    overlap = self._cache_overlap_blocks(lba, block_count, now)
                    if overlap >= block_count:
                        self._publish_event(
                            self.current_rpm,
                            is_seq=True,
                            queue_depth=queue_depth,
                            op_kind=op_kind,
                            size_bytes=requested_size_bytes,
                            block_count=block_count,
                            extent_count=reported_extent_count,
                            directory_entry_count=directory_entry_count,
                            fragmentation_score=fragmentation_score,
                        )
                        self.last_access_time = now
                        return OperationStats(
                            total_ms=0.03 + ready_info.ready_poll_ms,
                            cache_hit=True,
                            partial_hit=False,
                            cyl=self.current_cyl,
                            head=self.current_head,
                            startup_ms=ready_info.startup_ms,
                            startup_origin=ready_info.startup_origin,
                            ready_poll_ms=ready_info.ready_poll_ms,
                            ready_poll_count=ready_info.ready_poll_count,
                            maintenance_wait_ms=maintenance_wait_ms,
                            size_bytes=requested_size_bytes,
                            block_count=block_count,
                            extent_count=reported_extent_count,
                            directory_entry_count=directory_entry_count,
                            fragmentation_score=fragmentation_score,
                        )
                    if overlap > 0:
                        partial_hit = True
                        cache_latency_ms = 0.05
                        lba += overlap
                        block_count -= overlap
                        size_bytes = block_count * self.block_bytes

                total_latency_ms, target_cyl, target_head, _target_sector, distance, zone = self._calculate_position_latency(
                    lba,
                    block_count,
                )
                command_overhead_ms = self._command_overhead_for(op_kind, queue_depth)
                flush_ms = self.flush_penalty_ms if (force_unit_access or op_kind == "flush") else 0.0
                recovery_ms, retry_count = self._read_recovery_tail(
                    lba,
                    block_count,
                    is_write=is_write,
                    op_kind=op_kind,
                    queue_depth=queue_depth,
                )
                total_latency_ms += (
                    ready_info.ready_poll_ms
                    + flush_ms
                    + command_overhead_ms
                    + cache_latency_ms
                    + maintenance_wait_ms
                    + recovery_ms
                )
                seek_trigger = distance > 0 or target_head != self.current_head
                transfer_ms = self._transfer_ms_for_span(lba, block_count)
                is_sequential_access = op_kind in {"data", "writeback"} and (
                    (op_kind == "data" and distance == 0 and not is_write)
                    or (block_count >= 16 and reported_extent_count <= 1)
                )
                motion_duration_ms, settle_duration_ms = self._actuator_profile(
                    seek_distance=distance,
                    queue_depth=queue_depth,
                    op_kind=op_kind,
                    is_flush=(force_unit_access or op_kind == "flush"),
                    head_switch=(target_head != self.current_head),
                    is_sequential=is_sequential_access,
                )

                self._publish_event(
                    self.current_rpm,
                    seek_trigger=seek_trigger,
                    seek_dist=distance,
                    is_seq=is_sequential_access,
                    queue_depth=queue_depth,
                    op_kind=op_kind,
                    is_flush=(force_unit_access or op_kind == "flush"),
                    is_spinup=ready_info.ready_poll_ms > 0.0,
                    motion_duration_ms=motion_duration_ms,
                    settle_duration_ms=settle_duration_ms,
                    target_rpm=self.target_rpm,
                    size_bytes=size_bytes,
                    block_count=block_count,
                    extent_count=reported_extent_count,
                    transfer_ms=transfer_ms,
                    directory_entry_count=directory_entry_count,
                    fragmentation_score=fragmentation_score,
                )

            self._sleep_ms(total_latency_ms - ready_info.ready_poll_ms)

            with self.lock:
                final_lba = min(self.addressable_blocks - 1, lba + max(block_count - 1, 0))
                final_cyl, final_head, final_sector, final_zone = self._lba_to_chs(final_lba)
                self.current_cyl = final_cyl
                self.current_head = final_head
                self.current_sector = (final_sector + 1) % final_zone.blocks_per_track
                self.last_access_time = self.clock.now()
                self.power_state = "active"

                if not is_write and op_kind == "data":
                    self._remember_read(lba, block_count)
                else:
                    self.last_read_end_lba = -1

                return OperationStats(
                    total_ms=total_latency_ms,
                    cyl=target_cyl,
                    head=target_head,
                    cache_hit=cache_hit,
                    partial_hit=partial_hit,
                    transfer_rate_mbps=zone.transfer_rate_mbps,
                    startup_ms=ready_info.startup_ms,
                    startup_origin=ready_info.startup_origin,
                    ready_poll_ms=ready_info.ready_poll_ms,
                    ready_poll_count=ready_info.ready_poll_count,
                    retry_count=retry_count,
                    recovery_ms=recovery_ms,
                    maintenance_wait_ms=maintenance_wait_ms,
                    size_bytes=size_bytes,
                    block_count=block_count,
                    extent_count=reported_extent_count,
                    transfer_ms=transfer_ms,
                    directory_entry_count=directory_entry_count,
                    fragmentation_score=fragmentation_score,
                )

    def _background_tasks_loop(self) -> None:
        while self.running:
            self.sleeper.sleep(0.05)
            decision = BackgroundDecision(rpm=0.0)
            with self.lock:
                decision = background_decision(
                    self.core_config,
                    self._core_mechanical_state(),
                    transition_active=self.transition_kind is not None,
                    now=self.clock.now(),
                    last_idle_calibration_time=self.last_idle_calibration_time,
                    unload_after_s=self.unload_after_s,
                    low_rpm_after_s=self.low_rpm_after_s,
                    standby_after_s=self.standby_after_s,
                )
                if decision.next_power_state is not None:
                    self.power_state = decision.next_power_state
                if decision.next_heads_loaded is not None:
                    self.heads_loaded = decision.next_heads_loaded
                if decision.load_unload_delta:
                    self.load_unload_count += decision.load_unload_delta
                if decision.next_idle_calibration_time is not None:
                    self.last_idle_calibration_time = decision.next_idle_calibration_time

            if decision.park:
                self._publish_event(
                    decision.rpm,
                    target_rpm=0.0,
                    is_park=True,
                    motion_duration_ms=2.4,
                    settle_duration_ms=1.8,
                )
            elif decision.calibrate:
                self._publish_event(
                    decision.rpm,
                    is_cal=True,
                    motion_duration_ms=1.2,
                    settle_duration_ms=1.0,
                )
            self._background_scan_step(self.clock.now())
            if decision.should_low_rpm:
                self.begin_async_low_rpm()
            if decision.should_spindown:
                self.begin_async_spindown()
