from pathlib import Path

import pytest
from tests.helpers import _request, _run_test_server

from tests.test_webdav import test_webdav_parent_reference_directory_destinations_stay_in_backing_root as directory_workflow


@pytest.mark.parametrize("iteration", range(24))
def test_repeated_directory_copy_move(tmp_path: Path, iteration: int) -> None:
    del iteration
    directory_workflow(tmp_path)


@pytest.mark.parametrize("method", ["COPY", "MOVE"])
def test_normalized_descendant_destination_is_rejected(tmp_path: Path, method: str) -> None:
    backing = tmp_path / "backing"
    (backing / "docs").mkdir(parents=True)
    (backing / "docs" / "file.bin").write_bytes(b"preserved")
    with _run_test_server(backing) as (base_url, _provider):
        status, _, _ = _request(base_url, method, "/docs", headers={
            "Destination": f"{base_url}/../docs/child", "Depth": "infinity",
        })
        assert status == 403
        assert (backing / "docs" / "file.bin").read_bytes() == b"preserved"
        assert not (backing / "docs" / "child").exists()
