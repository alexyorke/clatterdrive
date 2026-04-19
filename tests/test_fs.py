from __future__ import annotations

import random

import pytest

from fake_hdd_fuse.fs import FileSystemSimulator

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

def test_path_normalization_stays_posix_like_on_windows() -> None:
    fs = FileSystemSimulator(total_gb=1)
    assert fs._normalize_path("foo/bar") == "/foo/bar"
    assert fs._normalize_path("foo\\bar") == "/foo/bar"
    assert fs._normalize_path("/foo/bar") == "/foo/bar"
    assert fs._normalize_path("//foo//bar") == "/foo/bar"
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
