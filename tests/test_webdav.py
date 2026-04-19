from __future__ import annotations

import os
import random
import urllib.parse
from pathlib import Path

from tests.helpers import _assert_provider_tree_matches_disk, _list_disk_tree, _request, _run_test_server

def test_webdav_end_to_end_directory_lifecycle(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "MKCOL", "/docs")
        assert status == 201

        status, _, _ = _request(base_url, "PUT", "/docs/empty.txt", b"")
        assert status in (200, 201, 204)

        status, _, _ = _request(base_url, "PUT", "/docs/data.txt", b"hello world")
        assert status in (200, 201, 204)

        status, body, _ = _request(base_url, "GET", "/docs/data.txt")
        assert status == 200
        assert body == b"hello world"
        assert provider.vhdd.fs.files["/docs/data.txt"].size == len(b"hello world")

        status, body, _ = _request(base_url, "PROPFIND", "/docs", headers={"Depth": "1"})
        assert status == 207
        assert b"data.txt" in body
        assert b"empty.txt" in body
        assert "/docs" in provider.vhdd.fs.directories
        assert "/docs/empty.txt" in provider.vhdd.fs.files

        status, _, _ = _request(
            base_url,
            "MOVE",
            "/docs",
            headers={"Destination": f"{base_url}/archive", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert "/archive" in provider.vhdd.fs.directories
        assert "/archive/data.txt" in provider.vhdd.fs.files

        status, body, _ = _request(base_url, "GET", "/archive/data.txt")
        assert status == 200
        assert body == b"hello world"

        status, _, _ = _request(base_url, "DELETE", "/archive")
        assert status == 204
        assert "/archive" not in provider.vhdd.fs.directories
        assert "/archive/data.txt" not in provider.vhdd.fs.files
        _assert_provider_tree_matches_disk(provider, backing)

def test_webdav_copy_file_and_directory_tree(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "MKCOL", "/docs")
        assert status == 201

        status, _, _ = _request(base_url, "MKCOL", "/docs/subdir")
        assert status == 201

        payload = b"a" * (256 * 1024)
        status, _, _ = _request(base_url, "PUT", "/docs/subdir/source.bin", payload)
        assert status in (200, 201, 204)

        status, _, _ = _request(
            base_url,
            "COPY",
            "/docs/subdir/source.bin",
            headers={"Destination": f"{base_url}/docs/subdir/source-copy.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert "/docs/subdir/source-copy.bin" in provider.vhdd.fs.files
        assert provider.vhdd.fs.files["/docs/subdir/source-copy.bin"].size == len(payload)

        status, body, _ = _request(base_url, "GET", "/docs/subdir/source-copy.bin")
        assert status == 200
        assert body == payload

        status, _, _ = _request(
            base_url,
            "COPY",
            "/docs",
            headers={"Destination": f"{base_url}/docs-copy", "Depth": "infinity", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert "/docs-copy" in provider.vhdd.fs.directories
        assert "/docs-copy/subdir" in provider.vhdd.fs.directories
        assert "/docs-copy/subdir/source.bin" in provider.vhdd.fs.files
        assert "/docs-copy/subdir/source-copy.bin" in provider.vhdd.fs.files

        status, body, _ = _request(base_url, "GET", "/docs-copy/subdir/source-copy.bin")
        assert status == 200
        assert body == payload
        _assert_provider_tree_matches_disk(provider, backing)

def test_webdav_overwrite_copy_updates_simulated_tree(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        _request(base_url, "MKCOL", "/src")
        _request(base_url, "MKCOL", "/dst")
        _request(base_url, "PUT", "/src/data.bin", b"s" * 8192)
        _request(base_url, "PUT", "/dst/data.bin", b"d" * 4096)

        status, _, _ = _request(
            base_url,
            "COPY",
            "/src/data.bin",
            headers={"Destination": f"{base_url}/dst/data.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)

        status, body, _ = _request(base_url, "GET", "/dst/data.bin")
        assert status == 200
        assert body == b"s" * 8192
        assert provider.vhdd.fs.files["/dst/data.bin"].size == 8192
        _assert_provider_tree_matches_disk(provider, backing)

def test_webdav_move_overwrite_updates_simulated_tree(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        _request(base_url, "MKCOL", "/src")
        _request(base_url, "MKCOL", "/dst")
        _request(base_url, "PUT", "/src/data.bin", b"m" * 12288)
        _request(base_url, "PUT", "/dst/data.bin", b"z" * 2048)

        status, _, _ = _request(
            base_url,
            "MOVE",
            "/src/data.bin",
            headers={"Destination": f"{base_url}/dst/data.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)

        status, body, _ = _request(base_url, "GET", "/dst/data.bin")
        assert status == 200
        assert body == b"m" * 12288
        assert "/src/data.bin" not in provider.vhdd.fs.files
        assert provider.vhdd.fs.files["/dst/data.bin"].size == 12288
        _assert_provider_tree_matches_disk(provider, backing)

def test_webdav_directory_copy_over_existing_tree_removes_stale_entries(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        _request(base_url, "MKCOL", "/src")
        _request(base_url, "MKCOL", "/src/sub")
        _request(base_url, "PUT", "/src/sub/fresh.bin", b"fresh")

        _request(base_url, "MKCOL", "/dst")
        _request(base_url, "MKCOL", "/dst/sub")
        _request(base_url, "PUT", "/dst/sub/stale.bin", b"stale")
        _request(base_url, "PUT", "/dst/old.bin", b"old")

        status, _, _ = _request(
            base_url,
            "COPY",
            "/src",
            headers={"Destination": f"{base_url}/dst", "Depth": "infinity", "Overwrite": "T"},
        )
        assert status in (201, 204)

        assert "/dst/sub/fresh.bin" in provider.vhdd.fs.files
        assert "/dst/sub/stale.bin" not in provider.vhdd.fs.files
        assert "/dst/old.bin" not in provider.vhdd.fs.files

        status, body, _ = _request(base_url, "GET", "/dst/sub/fresh.bin")
        assert status == 200
        assert body == b"fresh"
        _assert_provider_tree_matches_disk(provider, backing)

def test_webdav_randomized_tree_operations_keep_simulator_in_sync(tmp_path: Path) -> None:
    random.seed(11)
    backing = tmp_path / "backing"
    backing.mkdir()

    dir_counter = 0
    file_counter = 0

    def next_dir_name() -> str:
        nonlocal dir_counter
        dir_counter += 1
        return f"dir-{dir_counter}"

    def next_file_name() -> str:
        nonlocal file_counter
        file_counter += 1
        return f"file-{file_counter}.bin"

    with _run_test_server(backing) as (base_url, provider):
        for _ in range(25):
            dirs, files = _list_disk_tree(backing)
            op = random.choice(
                ["mkdir", "put", "copy_file", "copy_dir", "move_file", "move_dir", "delete_file", "delete_dir"]
            )

            if op == "mkdir":
                parent = random.choice(dirs)
                target = (parent.rstrip("/") + "/" + next_dir_name()) if parent != "/" else "/" + next_dir_name()
                status, _, _ = _request(base_url, "MKCOL", target)
                assert status == 201
            elif op == "put":
                parent = random.choice(dirs)
                target = (parent.rstrip("/") + "/" + next_file_name()) if parent != "/" else "/" + next_file_name()
                payload = bytes([65 + (file_counter % 26)]) * random.choice([1024, 4096, 12288])
                status, _, _ = _request(base_url, "PUT", target, payload)
                assert status in (200, 201, 204)
            elif op == "copy_file" and files:
                source = random.choice(files)
                dest_parent = random.choice(dirs)
                dest = (dest_parent.rstrip("/") + "/" + next_file_name()) if dest_parent != "/" else "/" + next_file_name()
                status, _, _ = _request(
                    base_url,
                    "COPY",
                    source,
                    headers={"Destination": f"{base_url}{dest}", "Overwrite": "T"},
                )
                assert status in (201, 204)
            elif op == "copy_dir" and len(dirs) > 1:
                source = random.choice([path for path in dirs if path != "/"])
                dest_parent_choices = [
                    path for path in dirs if path != source and not path.startswith(f"{source}/")
                ]
                if not dest_parent_choices:
                    continue
                dest_parent = random.choice(dest_parent_choices)
                dest = (dest_parent.rstrip("/") + "/" + next_dir_name()) if dest_parent != "/" else "/" + next_dir_name()
                status, _, _ = _request(
                    base_url,
                    "COPY",
                    source,
                    headers={"Destination": f"{base_url}{dest}", "Depth": "infinity", "Overwrite": "T"},
                )
                assert status in (201, 204)
            elif op == "move_file" and files:
                source = random.choice(files)
                dest_parent = random.choice(dirs)
                dest = (dest_parent.rstrip("/") + "/" + next_file_name()) if dest_parent != "/" else "/" + next_file_name()
                status, _, _ = _request(
                    base_url,
                    "MOVE",
                    source,
                    headers={"Destination": f"{base_url}{dest}", "Overwrite": "T"},
                )
                assert status in (201, 204)
            elif op == "move_dir" and len(dirs) > 1:
                source = random.choice([path for path in dirs if path != "/"])
                dest_parent_choices = [
                    path for path in dirs if path != source and not path.startswith(f"{source}/")
                ]
                if not dest_parent_choices:
                    continue
                dest_parent = random.choice(dest_parent_choices)
                dest = (dest_parent.rstrip("/") + "/" + next_dir_name()) if dest_parent != "/" else "/" + next_dir_name()
                status, _, _ = _request(
                    base_url,
                    "MOVE",
                    source,
                    headers={"Destination": f"{base_url}{dest}", "Overwrite": "T"},
                )
                assert status in (201, 204)
            elif op == "delete_file" and files:
                status, _, _ = _request(base_url, "DELETE", random.choice(files))
                assert status == 204
            elif op == "delete_dir" and len(dirs) > 1:
                status, _, _ = _request(base_url, "DELETE", random.choice([path for path in dirs if path != "/"]))
                assert status == 204
            else:
                continue

            _assert_provider_tree_matches_disk(provider, backing)

def test_webdav_lazy_restart_materializes_existing_tree_consistently(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    (backing / "docs" / "deep").mkdir(parents=True)
    (backing / "docs" / "deep" / "alpha.bin").write_bytes(b"a" * 4096)
    (backing / "docs" / "beta.bin").write_bytes(b"beta")

    with _run_test_server(backing) as (base_url, provider):
        status, body, _ = _request(base_url, "GET", "/docs/deep/alpha.bin")
        assert status == 200
        assert body == b"a" * 4096
        assert "/docs/deep/alpha.bin" in provider.vhdd.fs.files

        status, body, _ = _request(base_url, "PROPFIND", "/docs", headers={"Depth": "infinity"})
        assert status == 207
        assert b"alpha.bin" in body
        assert b"beta.bin" in body
        _assert_provider_tree_matches_disk(provider, backing)

        status, _, _ = _request(base_url, "DELETE", "/docs")
        assert status == 204
        _assert_provider_tree_matches_disk(provider, backing)

def test_webdav_range_get_keeps_offsets_and_lengths_consistent(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    payload = bytes(range(256)) * 32

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "PUT", "/range.bin", payload)
        assert status in (200, 201, 204)

        status, body, headers = _request(
            base_url,
            "GET",
            "/range.bin",
            headers={"Range": "bytes=1024-2047"},
        )
        assert status == 206
        assert body == payload[1024:2048]
        assert headers.get("Content-Range", "").startswith("bytes 1024-2047/")
        assert provider.vhdd.fs.files["/range.bin"].size == len(payload)
        _assert_provider_tree_matches_disk(provider, backing)

def test_webdav_handles_spaces_quotes_and_punctuation_in_names(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    directory = "/Project Files"
    filename = "/Project Files/report 'final' ! test (v1),done.txt"
    encoded_directory = urllib.parse.quote(directory)
    encoded_filename = urllib.parse.quote(filename)

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "MKCOL", encoded_directory)
        assert status == 201

        payload = b"odd names still work"
        status, _, _ = _request(base_url, "PUT", encoded_filename, payload)
        assert status in (200, 201, 204)

        status, body, _ = _request(base_url, "GET", encoded_filename)
        assert status == 200
        assert body == payload

        status, body, _ = _request(base_url, "PROPFIND", encoded_directory, headers={"Depth": "1"})
        assert status == 207
        assert b"report 'final' ! test (v1),done.txt" in body
        assert filename in provider.vhdd.fs.files
        _assert_provider_tree_matches_disk(provider, backing)

def test_webdav_zero_byte_file_copy_move_delete_paths(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "MKCOL", "/docs")
        assert status == 201

        status, _, _ = _request(base_url, "PUT", "/docs/empty.bin", b"")
        assert status in (200, 201, 204)
        assert provider.vhdd.fs.files["/docs/empty.bin"].size == 0

        status, _, _ = _request(
            base_url,
            "COPY",
            "/docs/empty.bin",
            headers={"Destination": f"{base_url}/docs/empty-copy.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert provider.vhdd.fs.files["/docs/empty-copy.bin"].size == 0

        status, _, _ = _request(
            base_url,
            "MOVE",
            "/docs/empty-copy.bin",
            headers={"Destination": f"{base_url}/docs/empty-moved.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)
        assert "/docs/empty-copy.bin" not in provider.vhdd.fs.files
        assert provider.vhdd.fs.files["/docs/empty-moved.bin"].size == 0

        status, _, _ = _request(base_url, "DELETE", "/docs/empty.bin")
        assert status == 204
        status, _, _ = _request(base_url, "DELETE", "/docs/empty-moved.bin")
        assert status == 204
        assert "/docs/empty.bin" not in provider.vhdd.fs.files
        assert "/docs/empty-moved.bin" not in provider.vhdd.fs.files
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_large_directory_listing_handles_many_entries(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        status, _, _ = _request(base_url, "MKCOL", "/bulk")
        assert status == 201

        for index in range(96):
            payload = f"item-{index:03d}".encode("ascii")
            status, _, _ = _request(base_url, "PUT", f"/bulk/file-{index:03d}.txt", payload)
            assert status in (200, 201, 204)

        status, body, _ = _request(base_url, "PROPFIND", "/bulk", headers={"Depth": "1"})
        assert status == 207
        assert b"file-000.txt" in body
        assert b"file-095.txt" in body
        assert len(provider.vhdd.fs.dir_children["/bulk"]) == 96
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_repeated_propfind_requests_keep_provider_state_consistent(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        _request(base_url, "MKCOL", "/docs")
        _request(base_url, "MKCOL", "/docs/deep")
        _request(base_url, "PUT", "/docs/deep/data.txt", b"propfind traffic")

        for depth in ("0", "1", "infinity") * 6:
            status, body, _ = _request(base_url, "PROPFIND", "/docs", headers={"Depth": depth})
            assert status == 207
            assert b"docs" in body

        assert "/docs/deep/data.txt" in provider.vhdd.fs.files
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_deep_nested_rename_followed_by_propfind_and_get(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        for path in ("/alpha", "/alpha/bravo", "/alpha/bravo/charlie", "/alpha/bravo/charlie/delta"):
            status, _, _ = _request(base_url, "MKCOL", path)
            assert status == 201

        payload = b"deep payload"
        status, _, _ = _request(base_url, "PUT", "/alpha/bravo/charlie/delta/file.bin", payload)
        assert status in (200, 201, 204)

        status, _, _ = _request(
            base_url,
            "MOVE",
            "/alpha",
            headers={"Destination": f"{base_url}/renamed", "Overwrite": "T"},
        )
        assert status in (201, 204)

        status, body, _ = _request(base_url, "PROPFIND", "/renamed", headers={"Depth": "infinity"})
        assert status == 207
        assert b"file.bin" in body

        status, body, _ = _request(base_url, "GET", "/renamed/bravo/charlie/delta/file.bin")
        assert status == 200
        assert body == payload
        _assert_provider_tree_matches_disk(provider, backing)


def test_webdav_case_only_copy_and_move_do_not_duplicate_windows_paths(tmp_path: Path) -> None:
    if os.name != "nt":
        return

    backing = tmp_path / "backing"
    backing.mkdir()

    with _run_test_server(backing) as (base_url, provider):
        _request(base_url, "MKCOL", "/docs")
        payload = b"case sensitive enough"
        status, _, _ = _request(base_url, "PUT", "/docs/source.bin", payload)
        assert status in (200, 201, 204)

        status, _, _ = _request(
            base_url,
            "MOVE",
            "/docs/source.bin",
            headers={"Destination": f"{base_url}/docs/Source.bin", "Overwrite": "T"},
        )
        assert status in (201, 204)

        status, body, _ = _request(base_url, "GET", "/docs/Source.bin")
        assert status == 200
        assert body == payload

        status, _, _ = _request(
            base_url,
            "COPY",
            "/docs/Source.bin",
            headers={"Destination": f"{base_url}/docs/SOURCE.bin", "Overwrite": "T"},
        )
        assert status < 500

        casefold_matches = [
            path for path in provider.vhdd.fs.files if path.casefold() == "/docs/source.bin".casefold()
        ]
        assert len(casefold_matches) == 1
        _assert_provider_tree_matches_disk(provider, backing)
