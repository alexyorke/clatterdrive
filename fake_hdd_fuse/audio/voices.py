from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
VoicePath = Literal["airborne", "structure"]


@dataclass(frozen=True)
class AudioVoice:
    name: str
    path: VoicePath
    signal: FloatArray


def build_voices(
    *,
    direct_force: FloatArray,
    platter: FloatArray,
    cover: FloatArray,
    actuator: FloatArray,
    structure_modes: FloatArray,
    coupled_structure: FloatArray,
) -> tuple[AudioVoice, ...]:
    return (
        AudioVoice("direct", "airborne", direct_force),
        AudioVoice("platter", "airborne", platter),
        AudioVoice("cover", "airborne", cover),
        AudioVoice("actuator", "airborne", actuator),
        AudioVoice("structure_modes", "structure", structure_modes),
        AudioVoice("structure_coupled", "structure", coupled_structure),
    )


def mix_voice_path(voices: Sequence[AudioVoice], path: VoicePath) -> FloatArray:
    matching = [voice.signal for voice in voices if voice.path == path]
    if not matching:
        return np.zeros(0, dtype=np.float64)
    return np.sum(np.stack(matching, axis=0), axis=0)
