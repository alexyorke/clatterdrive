from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from clatterdrive.fs import FileSystemSimulator
from clatterdrive.fs.core import create_filesystem_state, journal_op

def test_filesystem_write_emits_metadata_and_data() -> None:
    fs = FileSystemSimulator(total_gb=1)
    operations = fs.write("/demo.bin", 0, 8192)

    kinds = [operation.kind for operation in operations]
    assert "journal" in kinds
    assert "metadata" in kinds
    data_ops = [operation for operation in operations if operation.kind == "data"]
    assert len(data_ops) == 1
    assert data_ops[0].block_count == 2

def test_lookup_reads_directory_and_inode_metadata() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/demo.bin", 0, 4096)

    operations = fs.lookup("/demo.bin")
    assert [operation.kind for operation in operations] == ["metadata", "metadata"]
    assert operations[0].source == "dentry_lookup"
    assert operations[1].source == "inode_lookup"


def test_directory_listing_cost_scales_with_child_count() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.create_directory("/small")
    fs.create_directory("/large")

    for index in range(10):
        fs.create_empty_file(f"/small/file-{index:04d}.txt")
    for index in range(1000):
        fs.create_empty_file(f"/large/file-{index:04d}.txt")

    small_readdir = fs.list_directory("/small")[0]
    large_readdir = fs.list_directory("/large")[0]

    assert small_readdir.source == "readdir"
    assert large_readdir.source == "readdir"
    assert small_readdir.directory_entry_count == 10
    assert large_readdir.directory_entry_count == 1000
    assert large_readdir.block_count > small_readdir.block_count


def test_fragmented_read_produces_more_data_operations_than_contiguous_read() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.create_directory("/frag")
    fs.write("/contiguous.bin", 0, 6 * 4096)

    for index in range(10):
        fs.write(f"/frag/filler-{index}.bin", 0, 4096)
    for index in range(0, 10, 2):
        fs.delete(f"/frag/filler-{index}.bin")
    fs.write("/frag/fragmented.bin", 0, 6 * 4096)

    contiguous_ops = fs.read("/contiguous.bin", 0, 6 * 4096)
    fragmented_ops = fs.read("/frag/fragmented.bin", 0, 6 * 4096)

    assert len(fragmented_ops) > len(contiguous_ops)
    assert max(operation.fragmentation_score for operation in fragmented_ops) > 1


def test_path_normalization_stays_posix_like_on_windows() -> None:
    fs = FileSystemSimulator(total_gb=1)
    assert fs._normalize_path("foo/bar") == "/foo/bar"
    assert fs._normalize_path("foo\\bar") == "/foo/bar"
    assert fs._normalize_path("/foo/bar") == "/foo/bar"
    assert fs._normalize_path("//foo//bar") == "/foo/bar"
    assert fs._normalize_path("../escape") == "/escape"
    assert fs._normalize_path("../../escape") == "/escape"
    assert fs._normalize_path("/../../escape") == "/escape"
    assert fs._parent_dir("/foo/bar/baz.txt") == "/foo/bar"

def test_delete_returns_metadata_operations() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/demo.bin", 0, 16384)

    operations = fs.delete("/demo.bin")
    kinds = [operation.kind for operation in operations]
    assert kinds.count("journal") == 1
    assert kinds.count("metadata") >= 2
    assert fs.get_fragmentation_score("/demo.bin") == 0

def test_directory_rename_updates_subtree_and_recursive_delete_cleans_it_up() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.create_directory("/docs")
    fs.create_directory("/archive")
    fs.create_empty_file("/docs/note.txt")
    fs.write("/docs/note.txt", 0, 4096)

    dir_move_ops = fs.rename("/docs", "/archive/docs")
    assert "/docs" not in fs.directories
    assert "/archive/docs" in fs.directories
    assert "/archive/docs/note.txt" in fs.files
    assert any(operation.source == "dir_parent_update" for operation in dir_move_ops)

    file_move_ops = fs.rename("/archive/docs/note.txt", "/archive/docs/final.txt")
    assert "/archive/docs/note.txt" not in fs.files
    assert "/archive/docs/final.txt" in fs.files
    assert any(operation.source == "inode_rename" for operation in file_move_ops)

    delete_ops = fs.delete_directory("/archive", recursive=True)
    assert "/archive" not in fs.directories
    assert "/archive/docs/final.txt" not in fs.files
    assert any(operation.source == "dir_teardown" for operation in delete_ops)

def test_create_empty_file_tracks_zero_length_inode_without_data_blocks() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.create_directory("/docs")

    operations = fs.create_empty_file("/docs/empty.txt")

    assert "/docs/empty.txt" in fs.files
    assert fs.files["/docs/empty.txt"].size == 0
    assert fs.read("/docs/empty.txt", 0, 4096) == []
    assert any(operation.source == "inode_create" for operation in operations)

def test_truncate_frees_tail_extents_and_clears_size() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/demo.bin", 0, 16384)
    operations = fs.truncate("/demo.bin", size=0)

    assert fs.files["/demo.bin"].size == 0
    assert fs.read("/demo.bin", 0, 4096) == []
    assert operations[0].source == "truncate_intent"
    assert operations[1].source == "inode_truncate"


def test_filesystem_mutation_helpers_keep_state_consistent() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.create_directory("/work")
    fs.write("/work/data.bin", 0, 32 * 4096)
    fs.write("/work/data.bin", 64 * 4096, 8 * 4096)
    fs.assert_consistent()

    fs.truncate("/work/data.bin", size=16 * 4096)
    assert fs.files["/work/data.bin"].size == 16 * 4096
    fs.assert_consistent()

    fs.rename("/work/data.bin", "/work/renamed.bin")
    assert "/work/data.bin" not in fs.files
    assert "/work/renamed.bin" in fs.files
    fs.assert_consistent()

    fs.delete("/work/renamed.bin")
    assert "/work/renamed.bin" not in fs.files
    fs.assert_consistent()


def test_sparse_write_does_not_backfill_hole_reads() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/sparse.bin", 8192, 4096)

    assert fs.read("/sparse.bin", 0, 4096) == []
    hole_neighbor = fs.read("/sparse.bin", 8192, 4096)
    assert len(hole_neighbor) == 1
    assert hole_neighbor[0].block_count == 1

def test_reads_do_not_price_blocks_past_eof() -> None:
    fs = FileSystemSimulator(total_gb=1)
    fs.write("/tail.bin", 0, 1)

    assert len(fs.read("/tail.bin", 0, 4096)) == 1
    assert fs.read("/tail.bin", 1, 4096) == []

def test_filesystem_small_sizes_and_disk_full_are_handled() -> None:
    with pytest.raises(ValueError):
        FileSystemSimulator(total_gb=0.0001)

    fs = FileSystemSimulator(total_gb=0.05)
    with pytest.raises(OSError):
        fs.write("/too-big.bin", 0, 200 * 1024 * 1024)

def test_filesystem_tree_invariants_hold_under_random_directory_workload() -> None:
    random.seed(7)
    fs = FileSystemSimulator(total_gb=1)
    next_dir_id = 0
    next_file_id = 0

    def fresh_dir_name() -> str:
        nonlocal next_dir_id
        next_dir_id += 1
        return f"dir-{next_dir_id}"

    def fresh_file_name() -> str:
        nonlocal next_file_id
        next_file_id += 1
        return f"file-{next_file_id}.bin"

    for _ in range(150):
        dir_paths = sorted(fs.directories)
        file_paths = sorted(fs.files)
        operation = random.choice(
            ["mkdir", "write", "rename_file", "rename_dir", "delete_file", "delete_dir", "lookup", "list"]
        )

        if operation == "mkdir":
            parent = random.choice(dir_paths)
            fs.create_directory(f"{parent}/{fresh_dir_name()}")
        elif operation == "write":
            parent = random.choice(dir_paths)
            path = f"{parent}/{fresh_file_name()}"
            fs.write(path, random.randint(0, 4) * 4096, random.randint(1, 3) * 4096)
        elif operation == "rename_file" and file_paths:
            source = random.choice(file_paths)
            dest_parent = random.choice(dir_paths)
            fs.rename(source, f"{dest_parent}/{fresh_file_name()}")
        elif operation == "rename_dir" and len(dir_paths) > 1:
            source = random.choice([path for path in dir_paths if path != "/"])
            candidate_parents = [
                path for path in dir_paths if path != source and not path.startswith(f"{source}/")
            ]
            if candidate_parents:
                dest_parent = random.choice(candidate_parents)
                fs.rename(source, f"{dest_parent}/{fresh_dir_name()}")
        elif operation == "delete_file" and file_paths:
            fs.delete(random.choice(file_paths))
        elif operation == "delete_dir" and len(dir_paths) > 1:
            target = random.choice([path for path in dir_paths if path != "/"])
            fs.delete_directory(target, recursive=True)
        elif operation == "lookup":
            target = random.choice(dir_paths + file_paths or ["/"])
            fs.lookup(target)
        else:
            fs.list_directory(random.choice(dir_paths))

        fs.assert_consistent()


def test_deleted_inode_and_directory_metadata_blocks_are_reused() -> None:
    fs = FileSystemSimulator(total_gb=1)

    fs.create_empty_file("/old-file")
    old_file_inode = fs.files["/old-file"].inode_block
    fs.delete("/old-file")
    fs.create_empty_file("/new-file")
    assert fs.files["/new-file"].inode_block == old_file_inode

    fs.create_directory("/old-dir")
    old_dir_inode = fs.directories["/old-dir"].inode_block
    old_dir_block = fs.directories["/old-dir"].dir_block
    fs.delete_directory("/old-dir")
    fs.create_directory("/new-dir")
    assert fs.directories["/new-dir"].inode_block == old_dir_inode
    assert fs.directories["/new-dir"].dir_block == old_dir_block
    fs.assert_consistent()


def test_journal_operations_split_at_the_reserved_region_boundary() -> None:
    state = create_filesystem_state(total_gb=1, journal_blocks=4)
    state = replace(state, journal_cursor=3)

    next_state, operations = journal_op(state, 3, "wrap_test")

    assert [(operation.lba, operation.block_count) for operation in operations] == [
        (state.journal_start + 3, 1),
        (state.journal_start, 2),
    ]
    assert next_state.journal_cursor == 2
    assert all(operation.lba + operation.block_count <= state.inode_table_start for operation in operations)


def test_filesystem_profiles_change_directory_metadata_cost() -> None:
    ntfs = FileSystemSimulator(total_gb=1, filesystem_profile="ntfs_like")
    ext4 = FileSystemSimulator(total_gb=1, filesystem_profile="ext4_like")
    for index in range(80):
        ntfs.create_empty_file(f"/file-{index}")
        ext4.create_empty_file(f"/file-{index}")

    ntfs_scan = ntfs.list_directory("/")[0]
    ext4_scan = ext4.list_directory("/")[0]

    assert ntfs.profile.name == "ntfs_like"
    assert ext4.profile.name == "ext4_like"
    assert ntfs_scan.block_count > ext4_scan.block_count


def test_filesystem_state_persists_allocation_and_tree_metadata(tmp_path: Path) -> None:
    state_path = tmp_path / "volume-state.json"
    first = FileSystemSimulator(
        total_gb=0.1,
        filesystem_profile="generic_journaled",
        state_path=state_path,
    )
    first.create_directory("/docs")
    first.write("/docs/sparse.bin", 8192, 12288)
    first.create_empty_file("/deleted.bin")
    deleted_inode = first.files["/deleted.bin"].inode_block
    first.delete("/deleted.bin")
    expected_extents = list(first.files["/docs/sparse.bin"].extents)
    expected_cursor = first.journal_cursor
    first.persist()

    restored = FileSystemSimulator(
        total_gb=0.1,
        filesystem_profile="generic_journaled",
        state_path=state_path,
    )

    assert restored.loaded_from_state is True
    assert restored.files["/docs/sparse.bin"].extents == expected_extents
    assert restored.files["/docs/sparse.bin"].size == 20480
    assert restored.journal_cursor == expected_cursor
    assert deleted_inode in restored.state.free_inode_blocks
    restored.assert_consistent()


def test_persisted_state_rejects_changed_volume_geometry(tmp_path: Path) -> None:
    state_path = tmp_path / "volume-state.json"
    fs = FileSystemSimulator(total_gb=0.1, state_path=state_path)
    fs.create_empty_file("/file.bin")
    fs.persist()

    with pytest.raises(ValueError, match="geometry mismatch"):
        FileSystemSimulator(total_gb=0.2, state_path=state_path)
    with pytest.raises(ValueError, match="profile mismatch"):
        FileSystemSimulator(total_gb=0.1, filesystem_profile="ntfs_like", state_path=state_path)


def test_persisted_state_rejects_corrupt_tree_metadata(tmp_path: Path) -> None:
    state_path = tmp_path / "volume-state.json"
    fs = FileSystemSimulator(total_gb=0.1, state_path=state_path)
    fs.create_empty_file("/orphan.txt")
    fs.persist()

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["files"][0]["parent_dir"] = "/missing"
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing parent directory"):
        FileSystemSimulator(total_gb=0.1, state_path=state_path)
