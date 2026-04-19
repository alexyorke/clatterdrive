from __future__ import annotations

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
)
from .latency import HDDLatencyModel
from .virtual import VirtualHDD


HDDLatenyModel = HDDLatencyModel

__all__ = [
    "BackgroundDecision",
    "CacheSpan",
    "CacheState",
    "HDDCoreConfig",
    "HDDLatencyModel",
    "HDDLatenyModel",
    "MechanicalState",
    "OperationStats",
    "ReadyPollPlan",
    "StartupStage",
    "StartupTracePoint",
    "TransitionState",
    "VirtualHDD",
    "Zone",
]
