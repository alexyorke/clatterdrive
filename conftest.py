import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Generator

import pytest
from _pytest.fixtures import FixtureRequest


_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_tests"


def _worker_id() -> str:
    return os.environ.get("PYTEST_XDIST_WORKER", "gw0")


@pytest.fixture
def tmp_path(request: FixtureRequest) -> Generator[Path, None, None]:
    """
    Repo-local temp directory fixture.

    The default pytest tmp root is blocked in this environment, so keep temp
    directories under the workspace while preserving one-directory-per-test
    isolation for future xdist workers.
    """

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.node.name).strip("-") or "test"
    path = _TMP_ROOT / _worker_id() / f"{slug}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
