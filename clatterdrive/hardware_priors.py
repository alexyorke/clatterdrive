from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


TAU = 2.0 * math.pi
BEL_REFERENCE_POWER_W = 1e-12
IRONWOLF_SOURCE_URL = (
    "https://www.seagate.com/content/dam/seagate/assets/products/nas-drives/"
    "ironwolf-pro-hard-drive/files/Seagate_IronWolf_Pro_SATA_Product_Manual_24-20-16-12TB_206815300B.pdf"
)


@dataclass(frozen=True)
class HardwarePrior:
    name: str
    model_ids: tuple[str, ...]
    source_url: str
    source_note: str
    capacity_tb: int
    rpm: int
    heads_min: int
    heads_max: int
    disks_min: int
    disks_max: int
    weight_g: float
    cache_mb: int
    power_on_ready_typ_s: float
    power_on_ready_max_s: float
    standby_ready_typ_s: float
    standby_ready_max_s: float
    ready_to_spindle_stop_max_s: float
    startup_current_12v_peak_typ_a: float
    max_start_current_5v_ac_peak_a: float
    max_start_current_12v_ac_peak_a: float
    idle_power_w: float
    random_read_power_w: float
    random_write_power_w: float
    sequential_read_power_w: float
    sequential_write_power_w: float
    acoustic_idle_typ_bels: float
    acoustic_idle_max_bels: float
    acoustic_seek_typ_bels: float
    acoustic_seek_max_bels: float
    acoustic_standard: str

    @property
    def nominal_omega_rad_s(self) -> float:
        return self.rpm * TAU / 60.0

    @property
    def nominal_disk_count(self) -> int:
        return self.disks_max

    @property
    def max_startup_electrical_power_w(self) -> float:
        return 5.0 * self.max_start_current_5v_ac_peak_a + 12.0 * self.max_start_current_12v_ac_peak_a


@dataclass(frozen=True)
class ParameterBound:
    low: float
    high: float
    unit: str
    source: str
    reason: str

    @property
    def center(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def span(self) -> float:
        return max(self.high - self.low, 1e-12)


@dataclass(frozen=True)
class IronWolfPhysicsCandidate:
    platter_radius_m: float
    platter_mass_kg: float
    motor_efficiency: float
    startup_current_utilization: float
    ready_overhead_s: float
    idle_mechanical_power_fraction: float
    windage_drag_share: float
    seek_power_to_mechanics_fraction: float
    contact_duration_ms: float
    installation_loss_db: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "platter_radius_m": self.platter_radius_m,
            "platter_mass_kg": self.platter_mass_kg,
            "motor_efficiency": self.motor_efficiency,
            "startup_current_utilization": self.startup_current_utilization,
            "ready_overhead_s": self.ready_overhead_s,
            "idle_mechanical_power_fraction": self.idle_mechanical_power_fraction,
            "windage_drag_share": self.windage_drag_share,
            "seek_power_to_mechanics_fraction": self.seek_power_to_mechanics_fraction,
            "contact_duration_ms": self.contact_duration_ms,
            "installation_loss_db": self.installation_loss_db,
        }


@dataclass(frozen=True)
class DerivedIronWolfPhysics:
    nominal_omega_rad_s: float
    rotor_inertia_kg_m2: float
    spinup_energy_j: float
    startup_available_mechanical_power_w: float
    estimated_ready_s: float
    idle_mechanical_power_w: float
    nominal_drag_torque_n_m: float
    viscous_drag_coefficient_n_m_s: float
    windage_drag_coefficient_n_m_s2: float
    seek_delta_power_w: float
    seek_mechanical_power_w: float
    idle_sound_power_w: float
    seek_sound_power_w: float
    contact_stiffness_proxy: float
    normalized_spindle_inertia_scale: float


@dataclass(frozen=True)
class IronWolfFitResult:
    prior: HardwarePrior
    candidate: IronWolfPhysicsCandidate
    derived: DerivedIronWolfPhysics
    score: float
    evaluations: int
    violations: tuple[str, ...]


def bel_to_sound_power_w(bels: float) -> float:
    return BEL_REFERENCE_POWER_W * (10.0 ** bels)


def ironwolf_pro_16tb_prior() -> HardwarePrior:
    return HardwarePrior(
        name="Seagate IronWolf Pro 16TB",
        model_ids=("ST16000NT001", "ST16000NTZ01"),
        source_url=IRONWOLF_SOURCE_URL,
        source_note="Seagate IronWolf Pro SATA Product Manual, Rev B, tables 2, 5, 6, and 18.",
        capacity_tb=16,
        rpm=7200,
        heads_min=14,
        heads_max=15,
        disks_min=7,
        disks_max=8,
        weight_g=670.0,
        cache_mb=512,
        power_on_ready_typ_s=25.0,
        power_on_ready_max_s=30.0,
        standby_ready_typ_s=25.0,
        standby_ready_max_s=30.0,
        ready_to_spindle_stop_max_s=20.0,
        startup_current_12v_peak_typ_a=2.0,
        max_start_current_5v_ac_peak_a=1.06,
        max_start_current_12v_ac_peak_a=1.96,
        idle_power_w=6.19,
        random_read_power_w=8.98,
        random_write_power_w=8.75,
        sequential_read_power_w=8.30,
        sequential_write_power_w=8.58,
        acoustic_idle_typ_bels=2.8,
        acoustic_idle_max_bels=3.0,
        acoustic_seek_typ_bels=3.2,
        acoustic_seek_max_bels=3.4,
        acoustic_standard="A-weighted sound power, ISO 7779-style free-field over reflecting plane.",
    )


def ironwolf_parameter_bounds(_prior: HardwarePrior | None = None) -> dict[str, ParameterBound]:
    return {
        "platter_radius_m": ParameterBound(
            0.046,
            0.048,
            "m",
            "3.5-inch HDD platter geometry",
            "Outer platter radius after hub and enclosure clearances.",
        ),
        "platter_mass_kg": ParameterBound(
            0.025,
            0.055,
            "kg",
            "broad glass/aluminum platter mass assumption",
            "Latent mass is not published; range is constrained by drive mass and disk count.",
        ),
        "motor_efficiency": ParameterBound(
            0.45,
            0.80,
            "ratio",
            "brushless motor efficiency assumption",
            "Efficiency converts electrical startup power into mechanical rotor work.",
        ),
        "startup_current_utilization": ParameterBound(
            0.08,
            0.40,
            "ratio",
            "Seagate startup current envelope",
            "Average startup draw should be below peak current for a 25s ready time.",
        ),
        "ready_overhead_s": ParameterBound(
            2.0,
            10.0,
            "s",
            "Seagate ready-time definition",
            "Power-on ready includes non-spindle electronics, head load, and servo lock.",
        ),
        "idle_mechanical_power_fraction": ParameterBound(
            0.18,
            0.46,
            "ratio",
            "Seagate idle power",
            "Only part of idle electrical power is spindle drag after electronics losses.",
        ),
        "windage_drag_share": ParameterBound(
            0.12,
            0.58,
            "ratio",
            "viscous-plus-quadratic drag model",
            "Windage is the quadratic drag share at nominal RPM; bearing drag takes the rest.",
        ),
        "seek_power_to_mechanics_fraction": ParameterBound(
            0.10,
            0.48,
            "ratio",
            "Seagate random-operation power delta",
            "Fraction of random-operation power delta converted to actuator/contact mechanics.",
        ),
        "contact_duration_ms": ParameterBound(
            0.20,
            2.60,
            "ms",
            "head/media and latch contact event width",
            "Bounds the normalized stiffness/damping contact event duration.",
        ),
        "installation_loss_db": ParameterBound(
            0.0,
            18.0,
            "dB",
            "listener/install transfer assumption",
            "Only remaining global transfer between ISO sound-power spec and playback level.",
        ),
    }


def _candidate_from_values(values: Mapping[str, float]) -> IronWolfPhysicsCandidate:
    return IronWolfPhysicsCandidate(
        platter_radius_m=float(values["platter_radius_m"]),
        platter_mass_kg=float(values["platter_mass_kg"]),
        motor_efficiency=float(values["motor_efficiency"]),
        startup_current_utilization=float(values["startup_current_utilization"]),
        ready_overhead_s=float(values["ready_overhead_s"]),
        idle_mechanical_power_fraction=float(values["idle_mechanical_power_fraction"]),
        windage_drag_share=float(values["windage_drag_share"]),
        seek_power_to_mechanics_fraction=float(values["seek_power_to_mechanics_fraction"]),
        contact_duration_ms=float(values["contact_duration_ms"]),
        installation_loss_db=float(values["installation_loss_db"]),
    )


def candidate_from_unit_values(
    unit_values: Mapping[str, float],
    bounds: Mapping[str, ParameterBound] | None = None,
) -> IronWolfPhysicsCandidate:
    resolved_bounds = ironwolf_parameter_bounds() if bounds is None else bounds
    return _candidate_from_values(
        {
            name: bound.low + max(0.0, min(1.0, float(unit_values[name]))) * bound.span
            for name, bound in resolved_bounds.items()
        }
    )


def center_candidate(bounds: Mapping[str, ParameterBound] | None = None) -> IronWolfPhysicsCandidate:
    resolved_bounds = ironwolf_parameter_bounds() if bounds is None else bounds
    return _candidate_from_values({name: bound.center for name, bound in resolved_bounds.items()})


def latin_hypercube_candidates(
    bounds: Mapping[str, ParameterBound],
    *,
    samples: int,
    seed: int,
) -> tuple[IronWolfPhysicsCandidate, ...]:
    sample_count = max(int(samples), 1)
    rng = random.Random(seed)
    unit_columns: dict[str, list[float]] = {}
    for name in bounds:
        values = [(index + rng.random()) / sample_count for index in range(sample_count)]
        rng.shuffle(values)
        unit_columns[name] = values
    return tuple(
        candidate_from_unit_values(
            {name: unit_columns[name][index] for name in bounds},
            bounds,
        )
        for index in range(sample_count)
    )


def derive_ironwolf_physics(
    prior: HardwarePrior,
    candidate: IronWolfPhysicsCandidate,
) -> DerivedIronWolfPhysics:
    omega = prior.nominal_omega_rad_s
    disk_count = prior.nominal_disk_count
    inertia = 0.5 * disk_count * candidate.platter_mass_kg * candidate.platter_radius_m**2
    spinup_energy = 0.5 * inertia * omega**2
    startup_available_power = (
        prior.max_startup_electrical_power_w
        * candidate.motor_efficiency
        * candidate.startup_current_utilization
    )
    estimated_ready_s = spinup_energy / max(startup_available_power, 1e-12) + candidate.ready_overhead_s
    idle_mechanical_power = prior.idle_power_w * candidate.idle_mechanical_power_fraction
    nominal_drag_torque = idle_mechanical_power / max(omega, 1e-12)
    windage_drag_torque = nominal_drag_torque * candidate.windage_drag_share
    viscous_drag_torque = nominal_drag_torque - windage_drag_torque
    seek_delta_power = max(prior.random_read_power_w, prior.random_write_power_w) - prior.idle_power_w
    seek_mechanical_power = seek_delta_power * candidate.seek_power_to_mechanics_fraction
    contact_duration_s = candidate.contact_duration_ms / 1000.0
    contact_stiffness_proxy = seek_mechanical_power / max(contact_duration_s**2, 1e-12)
    baseline_inertia = 0.5 * 8.0 * 0.040 * 0.047**2
    return DerivedIronWolfPhysics(
        nominal_omega_rad_s=omega,
        rotor_inertia_kg_m2=inertia,
        spinup_energy_j=spinup_energy,
        startup_available_mechanical_power_w=startup_available_power,
        estimated_ready_s=estimated_ready_s,
        idle_mechanical_power_w=idle_mechanical_power,
        nominal_drag_torque_n_m=nominal_drag_torque,
        viscous_drag_coefficient_n_m_s=viscous_drag_torque / max(omega, 1e-12),
        windage_drag_coefficient_n_m_s2=windage_drag_torque / max(omega * abs(omega), 1e-12),
        seek_delta_power_w=seek_delta_power,
        seek_mechanical_power_w=seek_mechanical_power,
        idle_sound_power_w=bel_to_sound_power_w(prior.acoustic_idle_typ_bels),
        seek_sound_power_w=bel_to_sound_power_w(prior.acoustic_seek_typ_bels),
        contact_stiffness_proxy=contact_stiffness_proxy,
        normalized_spindle_inertia_scale=inertia / max(baseline_inertia, 1e-12),
    )


def ironwolf_constraint_violations(
    prior: HardwarePrior,
    candidate: IronWolfPhysicsCandidate,
    derived: DerivedIronWolfPhysics,
) -> tuple[str, ...]:
    violations: list[str] = []
    if derived.estimated_ready_s > prior.power_on_ready_max_s:
        violations.append("estimated_ready_exceeds_seagate_max")
    if derived.estimated_ready_s <= 0.0:
        violations.append("nonpositive_ready_time")
    if derived.viscous_drag_coefficient_n_m_s < 0.0:
        violations.append("negative_viscous_drag")
    if derived.windage_drag_coefficient_n_m_s2 < 0.0:
        violations.append("negative_windage_drag")
    if derived.startup_available_mechanical_power_w > prior.max_startup_electrical_power_w:
        violations.append("startup_mechanical_power_exceeds_electrical_limit")
    if derived.idle_sound_power_w > bel_to_sound_power_w(prior.acoustic_idle_max_bels):
        violations.append("idle_sound_power_exceeds_max")
    if derived.seek_sound_power_w > bel_to_sound_power_w(prior.acoustic_seek_max_bels):
        violations.append("seek_sound_power_exceeds_max")
    if not (0.0 <= candidate.windage_drag_share <= 1.0):
        violations.append("windage_share_outside_unit_interval")
    return tuple(violations)


def score_ironwolf_candidate(
    prior: HardwarePrior,
    candidate: IronWolfPhysicsCandidate,
    *,
    benchmarks: Mapping[str, float] | None = None,
    bounds: Mapping[str, ParameterBound] | None = None,
) -> tuple[float, DerivedIronWolfPhysics, tuple[str, ...]]:
    resolved_bounds = ironwolf_parameter_bounds(prior) if bounds is None else bounds
    derived = derive_ironwolf_physics(prior, candidate)
    violations = ironwolf_constraint_violations(prior, candidate, derived)
    if violations:
        return float("inf"), derived, violations

    score = ((derived.estimated_ready_s - prior.power_on_ready_typ_s) / 2.5) ** 2
    target_idle_mechanical_w = 0.32 * prior.idle_power_w
    score += 0.35 * ((derived.idle_mechanical_power_w - target_idle_mechanical_w) / 0.9) ** 2
    score += 0.20 * ((candidate.windage_drag_share - 0.34) / 0.18) ** 2
    score += 0.08 * ((candidate.installation_loss_db - 9.0) / 5.0) ** 2

    for name, value in candidate.as_mapping().items():
        bound = resolved_bounds[name]
        score += 0.012 * ((value - bound.center) / bound.span) ** 2

    if benchmarks is not None:
        score += 1.0e5 * max(0.0, 0.999 - float(benchmarks.get("metadata_correlation", 1.0))) ** 2
        score += 1.0e5 * max(0.0, float(benchmarks.get("metadata_rms_delta", 0.0)) - 0.010) ** 2
        score += 0.0005 * max(0.0, float(benchmarks.get("startup_log_mel_distance", 0.0)) - 20.0) ** 2

    return float(score), derived, ()


def _clamp_to_bound(value: float, bound: ParameterBound) -> float:
    return min(max(value, bound.low), bound.high)


def fit_ironwolf_physics(
    *,
    prior: HardwarePrior | None = None,
    seed: int = 23,
    samples: int = 96,
    coordinate_passes: int = 5,
    benchmarks: Mapping[str, float] | None = None,
) -> IronWolfFitResult:
    resolved_prior = ironwolf_pro_16tb_prior() if prior is None else prior
    bounds = ironwolf_parameter_bounds(resolved_prior)
    candidates = (center_candidate(bounds), *latin_hypercube_candidates(bounds, samples=samples, seed=seed))

    best_candidate = candidates[0]
    best_score, best_derived, best_violations = score_ironwolf_candidate(
        resolved_prior,
        best_candidate,
        benchmarks=benchmarks,
        bounds=bounds,
    )
    evaluations = 1
    for candidate in candidates[1:]:
        score, derived, violations = score_ironwolf_candidate(
            resolved_prior,
            candidate,
            benchmarks=benchmarks,
            bounds=bounds,
        )
        evaluations += 1
        if score < best_score:
            best_candidate = candidate
            best_score = score
            best_derived = derived
            best_violations = violations

    for shrink in (1.0, 0.52, 0.27, 0.14, 0.07)[: max(coordinate_passes, 0)]:
        improved = True
        while improved:
            improved = False
            values = best_candidate.as_mapping()
            for name, bound in bounds.items():
                step = bound.span * 0.12 * shrink
                for direction in (-1.0, 1.0):
                    trial_values = dict(values)
                    trial_values[name] = _clamp_to_bound(trial_values[name] + direction * step, bound)
                    trial_candidate = _candidate_from_values(trial_values)
                    score, derived, violations = score_ironwolf_candidate(
                        resolved_prior,
                        trial_candidate,
                        benchmarks=benchmarks,
                        bounds=bounds,
                    )
                    evaluations += 1
                    if score + 1e-12 < best_score:
                        best_candidate = trial_candidate
                        best_score = score
                        best_derived = derived
                        best_violations = violations
                        improved = True
                        values = best_candidate.as_mapping()

    return IronWolfFitResult(
        prior=resolved_prior,
        candidate=best_candidate,
        derived=best_derived,
        score=float(best_score),
        evaluations=evaluations,
        violations=best_violations,
    )


def ironwolf_calibration_report(
    result: IronWolfFitResult,
    *,
    benchmarks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bounds = ironwolf_parameter_bounds(result.prior)
    return {
        "prior": {
            "name": result.prior.name,
            "model_ids": list(result.prior.model_ids),
            "source_url": result.prior.source_url,
            "source_note": result.prior.source_note,
            "capacity_tb": result.prior.capacity_tb,
            "rpm": result.prior.rpm,
            "heads": [result.prior.heads_min, result.prior.heads_max],
            "disks": [result.prior.disks_min, result.prior.disks_max],
            "weight_g": result.prior.weight_g,
            "cache_mb": result.prior.cache_mb,
            "ready_s_typ_max": [result.prior.power_on_ready_typ_s, result.prior.power_on_ready_max_s],
            "startup_current_peak_a": {
                "typ_12v": result.prior.startup_current_12v_peak_typ_a,
                "max_5v": result.prior.max_start_current_5v_ac_peak_a,
                "max_12v": result.prior.max_start_current_12v_ac_peak_a,
            },
            "power_w": {
                "idle": result.prior.idle_power_w,
                "random_read": result.prior.random_read_power_w,
                "random_write": result.prior.random_write_power_w,
                "sequential_read": result.prior.sequential_read_power_w,
                "sequential_write": result.prior.sequential_write_power_w,
            },
            "acoustics_bels": {
                "idle_typ": result.prior.acoustic_idle_typ_bels,
                "idle_max": result.prior.acoustic_idle_max_bels,
                "seek_typ": result.prior.acoustic_seek_typ_bels,
                "seek_max": result.prior.acoustic_seek_max_bels,
            },
        },
        "fit": {
            "score": result.score,
            "evaluations": result.evaluations,
            "violations": list(result.violations),
            "parameters": {
                name: {
                    "value": value,
                    "bounds": [bounds[name].low, bounds[name].high],
                    "unit": bounds[name].unit,
                    "source": bounds[name].source,
                    "kind": "fitted_latent_physics",
                }
                for name, value in result.candidate.as_mapping().items()
            },
        },
        "derived": {
            "nominal_omega_rad_s": result.derived.nominal_omega_rad_s,
            "rotor_inertia_kg_m2": result.derived.rotor_inertia_kg_m2,
            "spinup_energy_j": result.derived.spinup_energy_j,
            "startup_available_mechanical_power_w": result.derived.startup_available_mechanical_power_w,
            "estimated_ready_s": result.derived.estimated_ready_s,
            "idle_mechanical_power_w": result.derived.idle_mechanical_power_w,
            "nominal_drag_torque_n_m": result.derived.nominal_drag_torque_n_m,
            "viscous_drag_coefficient_n_m_s": result.derived.viscous_drag_coefficient_n_m_s,
            "windage_drag_coefficient_n_m_s2": result.derived.windage_drag_coefficient_n_m_s2,
            "seek_delta_power_w": result.derived.seek_delta_power_w,
            "seek_mechanical_power_w": result.derived.seek_mechanical_power_w,
            "idle_sound_power_w": result.derived.idle_sound_power_w,
            "seek_sound_power_w": result.derived.seek_sound_power_w,
            "contact_stiffness_proxy": result.derived.contact_stiffness_proxy,
            "normalized_spindle_inertia_scale": result.derived.normalized_spindle_inertia_scale,
        },
        "benchmarks": {} if benchmarks is None else dict(benchmarks),
    }
