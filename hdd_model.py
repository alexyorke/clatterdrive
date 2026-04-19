from __future__ import annotations

from hdd_core import (
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
from hdd_latency import HDDLatencyModel
from virtual_hdd import VirtualHDD


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
