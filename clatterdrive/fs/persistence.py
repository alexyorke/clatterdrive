from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .core import DirectoryInode, FileInode, FileSystemState, assert_consistent, basename


STATE_FORMAT_VERSION = 1
_GEOMETRY_FIELDS = (
    "block_size",
    "total_blocks",
    "superblock_blocks",
    "journal_blocks",
    "inode_table_blocks",
    "directory_region_blocks",
    "bitmap_blocks",
    "journal_start",
    "inode_table_start",
    "directory_start",
    "bitmap_start",
    "data_start_block",
    "directory_entry_bytes",
)


def state_payload(state: FileSystemState, filesystem_profile: str) -> dict[str, Any]:
    geometry = {field: getattr(state, field) for field in _GEOMETRY_FIELDS}
    return {
        "format_version": STATE_FORMAT_VERSION,
        "filesystem_profile": filesystem_profile,
        "geometry": geometry,
        "next_inode_block": state.next_inode_block,
        "next_directory_block": state.next_directory_block,
        "free_inode_blocks": sorted(state.free_inode_blocks),
        "free_directory_blocks": sorted(state.free_directory_blocks),
        "journal_cursor": state.journal_cursor,
        "files": [
            {
                "path": inode.path,
                "inode_block": inode.inode_block,
                "parent_dir": inode.parent_dir,
                "extents": [list(extent) for extent in inode.extents],
                "size": inode.size,
            }
            for inode in sorted(state.files.values(), key=lambda item: item.path)
        ],
        "directories": [
            {
                "path": directory.path,
                "inode_block": directory.inode_block,
                "parent_dir": directory.parent_dir,
                "dir_block": directory.dir_block,
            }
            for directory in sorted(state.directories.values(), key=lambda item: item.path)
        ],
    }


def save_filesystem_state(path: str | Path, state: FileSystemState, filesystem_profile: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    encoded = json.dumps(state_payload(state, filesystem_profile), sort_keys=True, separators=(",", ":"))
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, destination)
    return destination


def load_filesystem_state(
    path: str | Path,
    expected: FileSystemState,
    filesystem_profile: str,
) -> FileSystemState:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format_version") != STATE_FORMAT_VERSION:
        raise ValueError(f"unsupported filesystem state format in {source}")
    if payload.get("filesystem_profile") != filesystem_profile:
        raise ValueError(
            f"filesystem profile mismatch for {source}: state uses {payload.get('filesystem_profile')!r}, "
            f"configuration requested {filesystem_profile!r}"
        )

    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"missing filesystem geometry in {source}")
    for field in _GEOMETRY_FIELDS:
        if geometry.get(field) != getattr(expected, field):
            raise ValueError(f"filesystem geometry mismatch for {field!r} in {source}")

    files: dict[str, FileInode] = {}
    for item in payload.get("files", []):
        inode = FileInode(
            path=str(item["path"]),
            inode_block=int(item["inode_block"]),
            parent_dir=str(item["parent_dir"]),
            extents=[
                (int(extent[0]), int(extent[1]), int(extent[2]))
                for extent in item.get("extents", [])
            ],
            size=int(item.get("size", 0)),
        )
        if inode.path in files:
            raise ValueError(f"duplicate file path in {source}: {inode.path}")
        files[inode.path] = inode

    directories: dict[str, DirectoryInode] = {}
    for item in payload.get("directories", []):
        directory = DirectoryInode(
            path=str(item["path"]),
            inode_block=int(item["inode_block"]),
            parent_dir=str(item["parent_dir"]),
            dir_block=int(item["dir_block"]),
        )
        if directory.path in directories:
            raise ValueError(f"duplicate directory path in {source}: {directory.path}")
        directories[directory.path] = directory

    bitmap = bytearray(expected.bitmap)
    for inode in files.values():
        for _logical_start, physical_start, length in inode.extents:
            if physical_start < expected.data_start_block or physical_start + length > expected.total_blocks:
                raise ValueError(f"file extent outside data region in {source}: {inode.path}")
            for block in range(physical_start, physical_start + length):
                if bitmap[block]:
                    raise ValueError(f"overlapping file extent at block {block} in {source}")
                bitmap[block] = 1

    dir_children: dict[str, set[str]] = {directory_path: set() for directory_path in directories}
    for directory in directories.values():
        if directory.path != "/":
            if directory.parent_dir not in dir_children:
                raise ValueError(f"missing parent directory for {directory.path} in {source}")
            dir_children[directory.parent_dir].add(basename(directory.path))
    for inode in files.values():
        if inode.parent_dir not in dir_children:
            raise ValueError(f"missing parent directory for {inode.path} in {source}")
        dir_children[inode.parent_dir].add(basename(inode.path))

    state = FileSystemState(
        **{field: getattr(expected, field) for field in _GEOMETRY_FIELDS},
        bitmap=bitmap,
        files=files,
        directories=directories,
        directory_blocks={path: directory.dir_block for path, directory in directories.items()},
        dir_children=dir_children,
        next_inode_block=int(payload["next_inode_block"]),
        next_directory_block=int(payload["next_directory_block"]),
        free_inode_blocks=[int(block) for block in payload.get("free_inode_blocks", [])],
        free_directory_blocks=[int(block) for block in payload.get("free_directory_blocks", [])],
        journal_cursor=int(payload.get("journal_cursor", 0)),
    )
    try:
        assert_consistent(state)
    except AssertionError as exc:
        raise ValueError(f"inconsistent filesystem state in {source}") from exc
    return state


__all__ = ["load_filesystem_state", "save_filesystem_state", "state_payload"]
