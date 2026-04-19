from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SeekPlan:
    start_pos: float
    target_pos: float
    direction: float
    duration_s: float
    settle_s: float
    max_velocity: float
    max_accel: float
    max_jerk: float


_QUINTIC_VELOCITY_COEFF = 1.875
_QUINTIC_ACCEL_COEFF = 5.773502691896258
_QUINTIC_JERK_COEFF = 60.0


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def plan_seek_motion(
    start_pos: float,
    target_pos: float,
    *,
    minimum_duration_s: float,
    settle_s: float,
    max_velocity: float,
    max_accel: float,
    max_jerk: float,
) -> SeekPlan:
    distance = abs(target_pos - start_pos)
    if distance <= 1e-9:
        return SeekPlan(
            start_pos=start_pos,
            target_pos=target_pos,
            direction=1.0,
            duration_s=max(minimum_duration_s, 0.0),
            settle_s=max(settle_s, 0.0),
            max_velocity=max_velocity,
            max_accel=max_accel,
            max_jerk=max_jerk,
        )

    velocity_duration = (_QUINTIC_VELOCITY_COEFF * distance) / max(max_velocity, 1e-6)
    accel_duration = math.sqrt((_QUINTIC_ACCEL_COEFF * distance) / max(max_accel, 1e-6))
    jerk_duration = ((_QUINTIC_JERK_COEFF * distance) / max(max_jerk, 1e-6)) ** (1.0 / 3.0)
    duration_s = max(minimum_duration_s, velocity_duration, accel_duration, jerk_duration)

    return SeekPlan(
        start_pos=start_pos,
        target_pos=target_pos,
        direction=1.0 if target_pos >= start_pos else -1.0,
        duration_s=duration_s,
        settle_s=max(settle_s, 0.0),
        max_velocity=max_velocity,
        max_accel=max_accel,
        max_jerk=max_jerk,
    )


def sample_seek_reference(plan: SeekPlan | None, elapsed_s: float) -> tuple[float, float, float]:
    if plan is None:
        return 0.0, 0.0, 0.0
    if plan.duration_s <= 0.0:
        return plan.target_pos, 0.0, 0.0

    s = clamp(elapsed_s / plan.duration_s, 0.0, 1.0)
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s

    blend = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
    velocity_blend = 30.0 * s2 - 60.0 * s3 + 30.0 * s4
    accel_blend = 60.0 * s - 180.0 * s2 + 120.0 * s3

    distance = plan.target_pos - plan.start_pos
    position = plan.start_pos + distance * blend
    velocity = distance * velocity_blend / max(plan.duration_s, 1e-9)
    acceleration = distance * accel_blend / max(plan.duration_s * plan.duration_s, 1e-9)
    return position, velocity, acceleration
