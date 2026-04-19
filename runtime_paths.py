from __future__ import annotations

import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = ROOT / ".runtime"


def ensure_runtime_subdir(name: str) -> Path:
    path = RUNTIME_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def workspace_tempdir(prefix: str, subdir: str) -> Generator[Path, None, None]:
    parent = ensure_runtime_subdir(subdir)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as temp_dir:
        yield Path(temp_dir)
