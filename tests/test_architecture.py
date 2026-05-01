from __future__ import annotations

import ast
from pathlib import Path

from clatterdrive.audio import physics


PURE_MODULES = (
    "clatterdrive/audio/commands.py",
    "clatterdrive/audio/core.py",
    "clatterdrive/audio/physics.py",
    "clatterdrive/fs/core.py",
    "clatterdrive/hdd/core.py",
    "clatterdrive/scheduler_core.py",
)
FORBIDDEN_IMPORTS = {
    "os",
    "socket",
    "sounddevice",
    "subprocess",
    "threading",
    "time",
    "urllib",
}
FORBIDDEN_CALLS = {
    "open",
    "os.environ.get",
    "pathlib.Path.open",
    "time.monotonic",
    "time.sleep",
}


def _attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _module_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / name


def test_pure_modules_do_not_import_forbidden_side_effect_modules() -> None:
    for module_name in PURE_MODULES:
        tree = ast.parse(_module_path(module_name).read_text(encoding="utf-8"), filename=module_name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in FORBIDDEN_IMPORTS, f"{module_name} imports forbidden module {alias.name!r}"
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                root = node.module.split(".")[0]
                assert root not in FORBIDDEN_IMPORTS, f"{module_name} imports forbidden module {node.module!r}"


def test_pure_modules_do_not_call_forbidden_side_effect_apis() -> None:
    for module_name in PURE_MODULES:
        tree = ast.parse(_module_path(module_name).read_text(encoding="utf-8"), filename=module_name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            path = _attribute_path(node.func)
            if path is None:
                continue
            assert path not in FORBIDDEN_CALLS, f"{module_name} calls forbidden side-effect API {path!r}"


def test_audio_physics_honesty_tiers_are_documented() -> None:
    physics_text = _module_path("clatterdrive/audio/physics.py").read_text(encoding="utf-8")
    profiles_text = _module_path("clatterdrive/profiles.py").read_text(encoding="utf-8")
    normalized_profiles_text = " ".join(profiles_text.split())
    readme_text = _module_path("README.md").read_text(encoding="utf-8")

    for label in ("Physical state", "Plausible model", "Artistic calibration"):
        assert label in physics_text
        assert label.lower().replace(" ", "-") in readme_text

    assert "not measured hardware constants" in normalized_profiles_text
    assert "not a measured CAD/acoustics transfer function" in normalized_profiles_text


def test_audio_physics_artistic_budget_is_explicit() -> None:
    tree = ast.parse(_module_path("clatterdrive/audio/physics.py").read_text(encoding="utf-8"))
    public_functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }

    assert set(physics.MODEL_TIER_BY_FUNCTION) == public_functions
    assert set(physics.MODEL_TIER_BY_FUNCTION.values()) <= set(physics.MODEL_TIERS)
    assert physics.artistic_budget() == tuple(
        name
        for name, tier in physics.MODEL_TIER_BY_FUNCTION.items()
        if tier == "artistic_calibration"
    )
    assert "step_windage_noise" not in physics.artistic_budget()
    assert "step_bearing_noise" not in physics.artistic_budget()
    assert "spindle_rotor_excitation" not in physics.artistic_budget()
    assert "motor_startup_current_envelope" not in physics.artistic_budget()
    assert "spindle_motor_reaction_force" not in physics.artistic_budget()
    assert "chassis_reaction_force" not in physics.artistic_budget()
    assert "head_media_event_forces" not in physics.artistic_budget()
    assert "voice_coil_force_transfer" not in physics.artistic_budget()
    assert "step_stiffness_damping_contact" not in physics.artistic_budget()
    assert "route_sources_to_structure" not in physics.artistic_budget()
    assert "radiate_acoustic_paths" not in physics.artistic_budget()
