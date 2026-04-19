from __future__ import annotations

import ast
from pathlib import Path


PURE_MODULES = (
    "fake_hdd_fuse/audio/core.py",
    "fake_hdd_fuse/audio/plant.py",
    "fake_hdd_fuse/audio/voices.py",
    "fake_hdd_fuse/fs/core.py",
    "fake_hdd_fuse/hdd/core.py",
    "fake_hdd_fuse/scheduler_core.py",
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
