from __future__ import annotations

import time

import numpy as np
import pytest

from fake_hdd_fuse.hdd import HDDLatencyModel
from fake_hdd_fuse.hdd.core import CacheSpan
from fake_hdd_fuse.storage_events import StorageEvent

def test_partial_read_cache_is_not_reported_as_full_hit() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0)
    model.read_cache.append(CacheSpan(start_lba=100, end_lba=101, expires_at=time.monotonic() + 5.0))

    result = model.submit_physical_access(100, 3 * model.block_bytes, False, op_kind="data")

    assert result["cache_hit"] is False
    assert result["partial_hit"] is True
    model.stop()

def test_non_prefix_cache_overlap_is_not_treated_as_partial_hit() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0)
    model.read_cache.append(CacheSpan(start_lba=101, end_lba=102, expires_at=time.monotonic() + 5.0))

    result = model.submit_physical_access(100, 3 * model.block_bytes, False, op_kind="data")

    assert result["cache_hit"] is False
    assert result["partial_hit"] is False
    model.stop()

def test_zone_table_does_not_extend_past_addressable_capacity() -> None:
    model = HDDLatencyModel(addressable_blocks=2048, latency_scale=0.0)
    assert model.zones[-1].end_lba == 2047
    model.stop()

def test_zone_boundary_transfer_latency_uses_per_zone_rates() -> None:
    model = HDDLatencyModel(addressable_blocks=100000, latency_scale=0.0)
    try:
        first_zone = model.zones[0]
        start_lba = first_zone.end_lba - 2
        block_count = 8
        cross_zone_ms = model._transfer_ms_for_span(start_lba, block_count)
        start_only_ms = ((block_count * model.block_bytes) / (1024 * 1024)) / first_zone.transfer_rate_mbps * 1000.0

        assert cross_zone_ms > start_only_ms
    finally:
        model.stop()

def test_drive_profile_changes_rpm_and_command_classes() -> None:
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        drive_profile="archive_5900_internal",
    )
    try:
        assert model.drive_profile.name == "archive_5900_internal"
        assert model.target_rpm == 5900
        assert model.command_overhead_by_kind["flush"] > model.command_overhead_by_kind["data"]
    finally:
        model.stop()

def test_queue_depth_materially_changes_latency() -> None:
    shallow = HDDLatencyModel(
        addressable_blocks=100000,
        latency_scale=0.0,
        drive_profile="enterprise_7200_bare",
    )
    deep = HDDLatencyModel(
        addressable_blocks=100000,
        latency_scale=0.0,
        drive_profile="enterprise_7200_bare",
    )
    try:
        shallow_result = shallow.submit_physical_access(4096, 4096, False, op_kind="data", queue_depth=1)
        deep_result = deep.submit_physical_access(4096, 4096, False, op_kind="data", queue_depth=8)
        assert deep_result["total_ms"] > shallow_result["total_ms"]
    finally:
        shallow.stop()
        deep.stop()

def test_power_on_sequence_reports_startup_and_finishes_ready() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0, start_ready=False)
    try:
        result = model.submit_physical_access(0, 4096, False, op_kind="data")
        assert result["startup_origin"] == "power_on"
        assert result["startup_ms"] == pytest.approx(model.power_on_to_ready_ms)
        assert result["ready_poll_count"] >= 0
        assert result["ready_poll_ms"] >= 0.0
        assert model.power_state == "active"
        assert model.current_rpm == model.target_rpm
    finally:
        model.stop()

def test_startup_stage_partitioning_respects_configured_budget() -> None:
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        start_ready=False,
        power_on_to_ready_ms=600.0,
        standby_to_ready_ms=320.0,
        unload_to_ready_ms=150.0,
        low_rpm_to_ready_ms=260.0,
    )
    try:
        assert sum(stage.duration_ms for stage in model._build_startup_sequence("power_on")) == pytest.approx(600.0)
        assert sum(stage.duration_ms for stage in model._build_startup_sequence("standby")) == pytest.approx(320.0)
        assert sum(stage.duration_ms for stage in model._build_startup_sequence("unloaded_idle")) == pytest.approx(150.0)
        assert sum(stage.duration_ms for stage in model._build_startup_sequence("low_rpm_idle")) == pytest.approx(260.0)
    finally:
        model.stop()

def test_ultrastar_startup_trace_has_smooth_runup_and_late_activity_burst() -> None:
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        start_ready=False,
        drive_profile="wd_ultrastar_hc550",
    )
    try:
        stages = model._build_startup_sequence("power_on")
        trace = model._build_startup_trace_from_stages("power_on", stages, step_ms=20.0)
        spinup_points = [point for point in trace if point.is_spinup and point.rpm > 0.0]
        rpm_values = np.array([point.rpm for point in spinup_points], dtype=np.float64)
        rpm_deltas = np.diff(rpm_values)
        self_test_index = next(index for index, point in enumerate(trace) if point.self_test_active)
        head_load_index = next(index for index, point in enumerate(trace) if point.head_load_event)
        servo_lock_index = next(index for index, point in enumerate(trace) if point.servo_locked)

        assert len(spinup_points) > 300
        assert np.all(rpm_deltas >= -1e-6)
        assert float(np.max(rpm_deltas)) < model.target_rpm * 0.02
        assert self_test_index < head_load_index < servo_lock_index
        assert trace[self_test_index].rpm >= model.target_rpm * 0.88
        assert trace[head_load_index].rpm >= model.target_rpm * 0.94
        assert trace[head_load_index].rpm >= model.target_rpm * 0.972
        assert trace[servo_lock_index].rpm >= model.target_rpm * 0.992
        assert any(point.is_spinup for point in trace[self_test_index:servo_lock_index + 1])
        assert any(
            point.seek_distance > 0.0 or point.is_calibration
            for point in trace[self_test_index: min(head_load_index + 120, len(trace))]
        )
    finally:
        model.stop()

def test_resume_latencies_follow_idle_state_depth() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0)
    try:
        model.reset_caches()
        with model.lock:
            model.power_state = "unloaded_idle"
            model.current_rpm = model.target_rpm
            model.heads_loaded = False
            model.ready_event.clear()
        unloaded = model.submit_physical_access(0, 4096, False, op_kind="data")["startup_ms"]

        model.reset_caches()
        with model.lock:
            model.power_state = "low_rpm_idle"
            model.current_rpm = model.low_rpm_rpm
            model.heads_loaded = False
            model.ready_event.clear()
        low_rpm = model.submit_physical_access(1024, 4096, False, op_kind="data")["startup_ms"]

        model.reset_caches()
        with model.lock:
            model.power_state = "standby"
            model.current_rpm = 0.0
            model.heads_loaded = False
            model.ready_event.clear()
        standby = model.submit_physical_access(2048, 4096, False, op_kind="data")["startup_ms"]

        assert unloaded == pytest.approx(model.unload_to_ready_ms)
        assert low_rpm == pytest.approx(model.low_rpm_to_ready_ms)
        assert standby == pytest.approx(model.standby_to_ready_ms)
        assert unloaded < low_rpm < standby
    finally:
        model.stop()

def test_async_power_on_started_before_first_io_reaches_ready_state() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0, start_ready=False)
    try:
        assert model.begin_async_startup() is True
        deadline = time.monotonic() + 1.5
        while True:
            if model.ready_event.wait(timeout=0.05):
                with model.lock:
                    if model.power_state == "active" and model.current_rpm == model.target_rpm:
                        break
            if time.monotonic() >= deadline:
                pytest.fail("timed out waiting for async power-on to reach ready state")
    finally:
        model.stop()

def test_access_during_async_power_on_reports_not_ready_polling() -> None:
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=1.0,
        start_ready=False,
        power_on_to_ready_ms=70.0,
        standby_to_ready_ms=50.0,
        unload_to_ready_ms=24.0,
        low_rpm_to_ready_ms=32.0,
        spinup_ms=24.0,
        ready_poll_ms=8.0,
    )
    try:
        assert model.begin_async_startup() is True
        result = model.submit_physical_access(0, 4096, False, op_kind="data")
        assert result["startup_origin"] == "power_on"
        assert result["startup_ms"] == 0.0
        assert result["ready_poll_count"] > 0
        assert result["ready_poll_ms"] > 0.0
        assert model.power_state == "active"
    finally:
        model.stop()

def test_idle_transitions_progress_to_low_rpm_and_standby() -> None:
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        unload_after_s=0.05,
        low_rpm_after_s=0.10,
        standby_after_s=0.15,
    )
    try:
        with model.lock:
            model.last_access_time = time.monotonic() - 1.0
        deadline = time.monotonic() + 1.5
        while True:
            with model.lock:
                if model.power_state == "standby" and model.current_rpm == 0.0:
                    break
            if time.monotonic() >= deadline:
                pytest.fail("timed out waiting for idle transitions to reach standby")
            time.sleep(0.02)
    finally:
        model.stop()

def test_staged_spindown_has_visible_intermediate_state_before_standby() -> None:
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=1.0,
        spin_down_ms=120.0,
    )
    try:
        assert model.begin_async_spindown() is True
        time.sleep(0.03)
        with model.lock:
            assert model.power_state == "spinning_down"
            assert model.current_rpm < model.target_rpm
        deadline = time.monotonic() + 0.5
        while True:
            with model.lock:
                if model.power_state == "standby":
                    assert model.current_rpm == 0.0
                    assert model.heads_loaded is False
                    break
            if time.monotonic() >= deadline:
                pytest.fail("timed out waiting for staged spindown to reach standby")
            time.sleep(0.01)
    finally:
        model.stop()

def test_read_recovery_retry_tail_is_optional_and_adds_latency() -> None:
    model = HDDLatencyModel(addressable_blocks=200000, latency_scale=0.0, enable_retry_recovery=True)
    quiet_model = HDDLatencyModel(addressable_blocks=200000, latency_scale=0.0, enable_retry_recovery=False)
    try:
        retry_lba = next(
            lba
            for lba in range(1, 50000)
            if model._read_recovery_tail(lba, 1, is_write=False, op_kind="data", queue_depth=1)[1] > 0
        )

        recovered = model.submit_physical_access(retry_lba, model.block_bytes, False, op_kind="data")
        clean = quiet_model.submit_physical_access(retry_lba, quiet_model.block_bytes, False, op_kind="data")
        write_result = model.submit_physical_access(retry_lba, model.block_bytes, True, op_kind="data")

        assert recovered.retry_count > 0
        assert recovered.recovery_ms > 0.0
        assert recovered.total_ms > clean.total_ms
        assert write_result.retry_count == 0
        assert write_result.recovery_ms == 0.0
    finally:
        model.stop()
        quiet_model.stop()


def test_background_scan_activity_can_delay_foreground_access() -> None:
    class CapturingSink:
        def __init__(self) -> None:
            self.events: list[StorageEvent] = []

        def publish_event(self, event: StorageEvent) -> None:
            self.events.append(event)

    sink = CapturingSink()
    model = HDDLatencyModel(
        addressable_blocks=100000,
        latency_scale=0.0,
        unload_after_s=10.0,
        low_rpm_after_s=30.0,
        standby_after_s=120.0,
        event_sink=sink,
        enable_background_scan=True,
    )
    try:
        with model.lock:
            model.last_access_time = time.monotonic() - 7.0
        deadline = time.monotonic() + 1.5
        while True:
            with model.lock:
                if model.background_busy_until > time.monotonic():
                    break
            if time.monotonic() >= deadline:
                pytest.fail("timed out waiting for background scan activity")
            time.sleep(0.02)

        result = model.submit_physical_access(0, model.block_bytes, False, op_kind="data")

        assert any(event.op_kind == "background" for event in sink.events)
        assert result.maintenance_wait_ms > 0.0
    finally:
        model.stop()

def test_spindown_trace_is_monotone_and_continuous() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=0.0, spin_down_ms=320.0)
    try:
        with model.lock:
            stages = model._build_spindown_sequence_locked()
        trace = model._build_spindown_trace_from_stages(stages, step_ms=10.0, initial_heads_loaded=True)
        rpm_values = np.array([point.rpm for point in trace], dtype=np.float64)
        rpm_deltas = np.diff(rpm_values)
        park_index = next(index for index, point in enumerate(trace) if point.park_event)

        assert len(trace) > 20
        assert np.all(rpm_deltas <= 1e-6)
        assert float(np.max(np.abs(rpm_deltas))) < model.target_rpm * 0.03
        assert trace[park_index].heads_loaded is False
        assert trace[-1].rpm == 0.0
    finally:
        model.stop()

def test_low_rpm_entry_is_continuous_not_instant() -> None:
    model = HDDLatencyModel(addressable_blocks=4096, latency_scale=1.0, spin_down_ms=120.0)
    try:
        with model.lock:
            model.power_state = "unloaded_idle"
            model.current_rpm = model.target_rpm
            model.heads_loaded = False
        assert model.begin_async_low_rpm() is True

        time.sleep(0.02)
        with model.lock:
            assert model.power_state == "slowing_to_low_rpm"
            assert model.low_rpm_rpm < model.current_rpm < model.target_rpm

        time.sleep(0.35)
        with model.lock:
            assert model.power_state == "low_rpm_idle"
            assert model.current_rpm == pytest.approx(model.low_rpm_rpm)
    finally:
        model.stop()

def test_idle_calibration_is_not_spammed_every_background_tick() -> None:
    class CapturingSink:
        def __init__(self) -> None:
            self.events: list[StorageEvent] = []

        def publish_event(self, event: StorageEvent) -> None:
            self.events.append(event)

    sink = CapturingSink()
    model = HDDLatencyModel(
        addressable_blocks=4096,
        latency_scale=0.0,
        unload_after_s=100.0,
        low_rpm_after_s=100.0,
        standby_after_s=1000.0,
        event_sink=sink,
    )
    try:
        with model.lock:
            model.last_access_time = time.monotonic() - 9.0
        time.sleep(0.22)
        calibration_events = [event for event in sink.events if event.servo_mode == "calibration"]
        assert len(calibration_events) == 1
    finally:
        model.stop()
