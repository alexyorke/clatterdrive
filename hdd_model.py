from __future__ import annotations

from clatterdrive.hdd import (
    BackgroundDecision,
    CacheSpan,
    CacheState,
    HDDCoreConfig,
    HDDLatencyModel,
    MechanicalState,
    OperationStats,
    ReadyPollPlan,
    StartupStage,
    StartupTracePoint,
    TransitionState,
    VirtualHDD,
    Zone,
)


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
