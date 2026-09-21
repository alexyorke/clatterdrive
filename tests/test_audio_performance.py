from __future__ import annotations

import numpy as np
import pytest

from clatterdrive.audio import HDDAudioEngine, HDDAudioEvent
from clatterdrive.audio.engine import HDDAudioSynthesizer


def test_batched_noise_preserves_mid_block_startup_transition() -> None:
    fast = HDDAudioSynthesizer(seed=11)
    reference = HDDAudioSynthesizer(seed=11)
    event = HDDAudioEvent(rpm=0, target_rpm=7200, emitted_at=0, is_spinup=True)
    for synth in (fast, reference):
        synth.apply_event(event)
        synth.state.plant.spindle_omega = 7200 * 2 * np.pi / 60 * 0.9919999
    actual = fast.render_chunk(2048)
    expected = reference.render_diagnostic_chunk(2048).output
    assert fast.state.supervisor.power_state == "active"
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("sample_rate", [22050, 44100, 48000])
@pytest.mark.parametrize("profile", ["desktop_7200_internal", "wd_ultrastar_hc550"])
def test_compiled_modal_path_matches_scalar_reference(sample_rate: int, profile: str) -> None:
    fast = HDDAudioSynthesizer(sample_rate=sample_rate, seed=37, drive_profile=profile)
    reference = HDDAudioSynthesizer(sample_rate=sample_rate, seed=37, drive_profile=profile)
    for event in (
        HDDAudioEvent(rpm=0, target_rpm=7200, emitted_at=0, is_spinup=True),
        HDDAudioEvent(rpm=7200, emitted_at=0, power_state="active", heads_loaded=True,
                      servo_mode="seek", target_track=0.8, track_delta=0.6,
                      motion_duration_ms=8, settle_duration_ms=3, transfer_ms=30),
        HDDAudioEvent(rpm=7200, emitted_at=0, servo_mode="park", heads_loaded=False),
    ):
        fast.apply_event(event)
        reference.apply_event(event)
        # Irregular chunks verify the conversion of persisted resonator state.
        for frames in (1, 17, 1024, 3000):
            output = fast.render_chunk(frames)
            expected = reference.render_diagnostic_chunk(frames).output
            np.testing.assert_allclose(output, expected, atol=1e-10, rtol=1e-8)
            np.testing.assert_allclose(fast.state.plant.base_vel, reference.state.plant.base_vel, atol=1e-10)


def test_device_callback_only_copies_pcm_and_handles_variable_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = HDDAudioEngine()

    def forbidden_render(frames: int) -> np.ndarray:
        raise AssertionError("Device callback must not synthesize or write WAV files")

    monkeypatch.setattr(engine, "render_chunk", forbidden_render)
    engine._live_chunks.extend([np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5])])
    first = np.empty((2, 1))
    engine._audio_callback(first, 2, None, None)
    np.testing.assert_array_equal(first[:, 0], [0.1, 0.2])
    second = np.empty((5, 1))
    engine._audio_callback(second, 5, None, None)
    np.testing.assert_array_equal(second[:, 0], [0.3, 0.4, 0.5, 0.0, 0.0])
    assert engine.playback_health()["buffer_underruns"] == 1


def test_live_worker_has_bounded_lookahead_and_stops() -> None:
    engine = HDDAudioEngine()
    engine._start_live_render_loop()
    thread = engine._live_render_thread
    try:
        assert len(engine._live_chunks) == 4
    finally:
        engine.stop()
    assert thread is not None and not thread.is_alive()
    assert engine._live_render_thread is None


def test_device_stop_failure_still_stops_renderer() -> None:
    class BrokenStream:
        def stop(self) -> None:
            raise RuntimeError("device disconnected")

        def close(self) -> None:
            pass

    engine = HDDAudioEngine()
    engine._start_live_render_loop()
    thread = engine._live_render_thread
    engine.stream = BrokenStream()
    with pytest.raises(RuntimeError, match="device disconnected"):
        engine.stop()
    assert thread is not None and not thread.is_alive()
    assert engine.stream is None
