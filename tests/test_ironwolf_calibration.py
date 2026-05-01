from __future__ import annotations

import pytest

from clatterdrive.hardware_priors import (
    bel_to_sound_power_w,
    fit_ironwolf_physics,
    ironwolf_calibration_report,
    ironwolf_constraint_violations,
    ironwolf_pro_16tb_prior,
)
from clatterdrive.profiles import DRIVE_PROFILES


def test_ironwolf_prior_matches_seagate_source_table() -> None:
    prior = ironwolf_pro_16tb_prior()

    assert prior.model_ids == ("ST16000NT001", "ST16000NTZ01")
    assert prior.rpm == 7200
    assert prior.capacity_tb == 16
    assert (prior.heads_min, prior.heads_max) == (14, 15)
    assert (prior.disks_min, prior.disks_max) == (7, 8)
    assert prior.weight_g == 670.0
    assert prior.cache_mb == 512
    assert (prior.power_on_ready_typ_s, prior.power_on_ready_max_s) == (25.0, 30.0)
    assert prior.ready_to_spindle_stop_max_s == 20.0
    assert prior.max_startup_electrical_power_w == pytest.approx(28.82)
    assert prior.idle_power_w == 6.19
    assert prior.random_read_power_w == 8.98
    assert prior.random_write_power_w == 8.75
    assert prior.sequential_read_power_w == 8.30
    assert prior.sequential_write_power_w == 8.58
    assert (prior.acoustic_idle_typ_bels, prior.acoustic_idle_max_bels) == (2.8, 3.0)
    assert (prior.acoustic_seek_typ_bels, prior.acoustic_seek_max_bels) == (3.2, 3.4)
    assert bel_to_sound_power_w(2.8) == pytest.approx(6.30957344480193e-10)


def test_ironwolf_fit_is_deterministic_and_physical() -> None:
    first = fit_ironwolf_physics(seed=13, samples=24, coordinate_passes=2)
    second = fit_ironwolf_physics(seed=13, samples=24, coordinate_passes=2)

    assert first.candidate == second.candidate
    assert first.violations == ()
    assert first.derived.estimated_ready_s <= first.prior.power_on_ready_max_s
    assert abs(first.derived.estimated_ready_s - first.prior.power_on_ready_typ_s) < 1.0
    assert first.derived.viscous_drag_coefficient_n_m_s >= 0.0
    assert first.derived.windage_drag_coefficient_n_m_s2 >= 0.0
    assert first.derived.idle_sound_power_w <= bel_to_sound_power_w(first.prior.acoustic_idle_max_bels)
    assert first.derived.seek_sound_power_w <= bel_to_sound_power_w(first.prior.acoustic_seek_max_bels)
    assert ironwolf_constraint_violations(first.prior, first.candidate, first.derived) == ()


def test_ironwolf_profile_references_prior_and_derived_spindle_terms() -> None:
    profile = DRIVE_PROFILES["seagate_ironwolf_pro_16tb"]

    assert profile.hardware_prior == "seagate_ironwolf_pro_16tb"
    assert profile.rpm == 7200
    assert profile.platters == 8
    assert profile.write_cache_mb == 512
    assert profile.spinup_ms == 25000.0
    assert profile.power_on_to_ready_ms == 25000.0
    assert profile.spindle_inertia_scale > 1.0
    assert 0.12 <= profile.windage_drag_share_at_nominal <= 0.58


def test_ironwolf_calibration_report_marks_fitted_and_source_values() -> None:
    result = fit_ironwolf_physics(seed=7, samples=12, coordinate_passes=1)
    report = ironwolf_calibration_report(result)

    assert report["prior"]["source_url"] == result.prior.source_url
    assert report["prior"]["model_ids"] == ["ST16000NT001", "ST16000NTZ01"]
    assert report["fit"]["violations"] == []
    assert report["fit"]["parameters"]["windage_drag_share"]["kind"] == "fitted_latent_physics"
    assert report["derived"]["estimated_ready_s"] <= result.prior.power_on_ready_max_s
