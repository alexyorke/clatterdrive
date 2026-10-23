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


def test_long_lived_server_copy_move_during_writeback(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()
    with _run_test_server(backing) as (base_url, provider):
        provider.vhdd.model.dirty_expire_s = 0.01
        assert _request(base_url, "MKCOL", "/docs")[0] == 201
        assert _request(base_url, "PUT", "/docs/file.bin", b"payload" * 1024)[0] == 201
        for _ in range(100):
            assert _request(base_url, "COPY", "/docs", headers={
                "Destination": f"{base_url}/../copy", "Depth": "infinity",
            })[0] in (201, 204)
            assert _request(base_url, "MOVE", "/copy", headers={
                "Destination": f"{base_url}/../moved",
            })[0] in (201, 204)
            assert _request(base_url, "GET", "/moved/file.bin")[1] == b"payload" * 1024
            assert _request(base_url, "DELETE", "/moved")[0] == 204
