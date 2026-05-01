from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from tools import generate_audio_samples
from tools import audio_physics_benchmark
from tools import profile_core
from tools import profile_fragmentation
import smoke
from tools import trace_audio_scenarios
from tools.generate_audio_samples import render_scenario, update_random_flush, update_sequential_read, update_spinup_idle
from tests.helpers import _run_test_server, _wav_metrics

def test_render_scenario_writes_nonempty_wav(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_audio_samples, "SAMPLES_DIR", tmp_path)
    output = render_scenario("test-sample", 0.25, update_sequential_read)
    assert output.exists()

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 44100
        assert wav_file.getnframes() > 0

def test_rendered_sample_scenarios_have_normalized_loudness(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_audio_samples, "SAMPLES_DIR", tmp_path)

    spinup = render_scenario("spinup", 2.0, update_spinup_idle, seed=7)
    sequential = render_scenario("sequential", 1.0, update_sequential_read, seed=11)
    random_flush = render_scenario("random", 1.0, update_random_flush, seed=13)

    spinup_rms, spinup_peak = _wav_metrics(spinup)
    sequential_rms, sequential_peak = _wav_metrics(sequential)
    random_rms, random_peak = _wav_metrics(random_flush)

    assert 1e-5 < spinup_rms < 2.0e-3
    assert 0.01 < sequential_rms < 0.03
    assert 0.02 < random_rms < 0.05
    assert random_rms > sequential_rms * 1.7
    assert 0.0 < spinup_peak < 0.01
    assert 0.05 < sequential_peak < 0.2
    assert 0.08 < random_peak < 0.25
    assert random_peak > sequential_peak * 1.1


def test_generate_extended_scenario_samples_writes_outputs(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(generate_audio_samples, "SAMPLES_DIR", tmp_path)
    outputs = generate_audio_samples.generate_extended_scenario_samples()
    assert len(outputs) == 3
    assert all(output.exists() for output in outputs)


def test_audio_physics_benchmark_reports_required_metrics() -> None:
    metadata = audio_physics_benchmark.metadata_storm_metrics()
    assert metadata["correlation"] >= 0.999
    assert metadata["rms_delta"] <= 0.010


def test_trace_audio_scenario_writes_json_and_svg(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(trace_audio_scenarios, "TRACE_DIR", tmp_path)
    output = trace_audio_scenarios.render_trace_scenario(
        "trace-test",
        0.5,
        update_sequential_read,
        seed=5,
    )

    svg_path = tmp_path / "trace-test.trace.svg"
    payload = output.read_text(encoding="utf-8")
    assert output.exists()
    assert svg_path.exists()
    assert '"events"' in payload
    assert '"diagnostics"' in payload

def test_smoke_main_boot_random_port_with_audio_disabled() -> None:
    smoke.run_main_boot_smoke(exercise_cli=False)

def test_smoke_cli_probe_works_when_curl_is_available(tmp_path: Path) -> None:
    if shutil.which("curl.exe") is None and shutil.which("curl") is None:
        pytest.skip("curl not available")

    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, _provider):
        assert smoke._run_cli_probe(base_url) is True

def test_profile_core_expectations_hold() -> None:
    metrics = profile_core.collect_core_metrics()
    profile_core.assert_core_expectations(metrics)
    assert metrics["cold_start_startup_ms"] > metrics["ready_startup_ms"]
    assert metrics["mixed_churn_total_ms"] > metrics["metadata_churn_total_ms"]

def test_profile_fragmentation_expectations_hold() -> None:
    metrics = profile_fragmentation.collect_fragmentation_metrics()
    profile_fragmentation.assert_fragmentation_expectations(metrics)
    assert metrics["fragmented_read_extents"] > metrics["contiguous_read_extents"]
    assert metrics["fragmented_read_ms"] > metrics["contiguous_read_ms"]
