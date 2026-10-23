from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
import pytest

from tools.backend_e2e import _wait_ready
from tools.fit_mh_reference import align_by_cross_correlation, event_window_metrics
from tools.reference_audio import compute_audio_features


def test_backend_readiness_timeout_does_not_block_on_silent_output() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="did not become ready"):
            _wait_ready(process, 0.1)
        assert time.monotonic() - started < 3.0
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_backend_output_is_drained_after_ready() -> None:
    process = subprocess.Popen(
        [sys.executable, "-u", "-c",
         'print("[]"); print(\'{"event":"ready"}\'); '
         'print(("x" * 1024 + "\\n") * 1024)'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        assert _wait_ready(process, 5) == {"event": "ready"}
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("delay", [-24, 24])
def test_cross_correlation_corrects_delay_direction(delay: int) -> None:
    reference = np.zeros(1024)
    reference[300:500] = np.random.default_rng(7).normal(size=200)
    candidate = np.roll(reference, delay)
    aligned, correction = align_by_cross_correlation(reference, candidate)
    assert correction == -delay
    np.testing.assert_allclose(aligned, reference)


@pytest.mark.parametrize("frames", [1, 256, 769, 1023])
def test_reference_features_accept_short_audio(frames: int) -> None:
    features = compute_audio_features(np.ones(frames), 22050, "enterprise_ultrastar")
    assert features["duration_s"] == pytest.approx(frames / 22050)
    assert np.isfinite(features["log_mel"]).all()


def test_event_weights_follow_valid_windows() -> None:
    reference = np.random.default_rng(9).normal(size=2400)
    candidate = reference.copy()
    candidate[1400:2200] *= -1
    events = [{"t": 5.0, "a": 100.0}, {"t": 0.02, "a": 1.0}, {"t": 0.065, "a": 9.0}]
    with_invalid = event_window_metrics(reference, candidate, events, max_lag_ms=0)
    without_invalid = event_window_metrics(reference, candidate, events[1:], max_lag_ms=0)
    assert with_invalid == without_invalid
