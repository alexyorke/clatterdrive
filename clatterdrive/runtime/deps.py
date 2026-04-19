from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...


@runtime_checkable
class Sleeper(Protocol):
    def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class EnvReader(Protocol):
    def get(self, key: str, default: str | None = None) -> str | None: ...


@runtime_checkable
class RNGFactory(Protocol):
    def create(self, seed: int | None = None) -> np.random.Generator: ...


@dataclass(frozen=True)
class SystemClock:
    def now(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class RealSleeper:
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class NoOpSleeper:
    def sleep(self, seconds: float) -> None:
        return None


@dataclass(frozen=True)
class OSEnvReader:
    def get(self, key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)


@dataclass(frozen=True)
class NumpyRNGFactory:
    def create(self, seed: int | None = None) -> np.random.Generator:
        return np.random.default_rng(seed)


@dataclass(frozen=True)
class RuntimeDeps:
    clock: Clock = SystemClock()
    sleeper: Sleeper = RealSleeper()
    env: EnvReader = OSEnvReader()
    rng_factory: RNGFactory = NumpyRNGFactory()

