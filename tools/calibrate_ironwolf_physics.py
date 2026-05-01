from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clatterdrive.hardware_priors import (
    fit_ironwolf_physics,
    ironwolf_calibration_report,
    ironwolf_pro_16tb_prior,
)
from tools import audio_physics_benchmark


ROOT = Path(__file__).resolve().parents[1]


def audio_benchmarks() -> dict[str, Any]:
    metadata = audio_physics_benchmark.metadata_storm_metrics()
    startup = audio_physics_benchmark.startup_reference_distances()
    return {
        "metadata_storm": metadata,
        "startup_reference": startup,
    }


def flattened_benchmark_penalties(benchmarks: dict[str, Any]) -> dict[str, float]:
    metadata = benchmarks.get("metadata_storm", {})
    startup = benchmarks.get("startup_reference", {})
    return {
        "metadata_correlation": float(metadata.get("correlation", 1.0)),
        "metadata_rms_delta": float(metadata.get("rms_delta", 0.0)),
        "startup_log_mel_distance": float(startup.get("log_mel_distance", 0.0)),
    }


def build_report(
    *,
    seed: int = 23,
    samples: int = 96,
    coordinate_passes: int = 5,
    include_benchmark: bool = True,
) -> dict[str, Any]:
    benchmarks = audio_benchmarks() if include_benchmark else {}
    penalties = flattened_benchmark_penalties(benchmarks)
    result = fit_ironwolf_physics(
        prior=ironwolf_pro_16tb_prior(),
        seed=seed,
        samples=samples,
        coordinate_passes=coordinate_passes,
        benchmarks=penalties,
    )
    return ironwolf_calibration_report(result, benchmarks=benchmarks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Seagate IronWolf Pro 16TB physics priors against guarded audio benchmarks.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--coordinate-passes", type=int, default=5)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / ".runtime" / "ironwolf-physics-calibration.json")
    args = parser.parse_args()

    report = build_report(
        seed=args.seed,
        samples=args.samples,
        coordinate_passes=args.coordinate_passes,
        include_benchmark=not args.no_benchmark,
    )
    payload = json.dumps(report, indent=2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
