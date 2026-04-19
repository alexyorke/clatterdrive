from __future__ import annotations

import math
import posixpath
from dataclasses import dataclass, field, replace


Extent = tuple[int, int, int]
BlockRun = tuple[int, int]


@dataclass(frozen=True)
class IOOperation:
    lba: int
    block_count: int
    kind: str
    source: str = ""


@dataclass
class FileInode:
    path: str
    inode_block: int
    parent_dir: str
    extents: list[Extent] = field(default_factory=list)
    size: int = 0


@dataclass
class DirectoryInode:
    path: str
    inode_block: int
    parent_dir: str
    dir_block: int


@dataclass
class FileSystemState:
    block_size: int
    total_blocks: int
    superblock_blocks: int
    journal_blocks: int
    inode_table_blocks: int
    directory_region_blocks: int
    bitmap_blocks: int
    journal_start: int
    inode_table_start: int
    directory_start: int
    bitmap_start: int
    data_start_block: int
    bitmap: bytearray
    files: dict[str, FileInode]
    directories: dict[str, DirectoryInode]
    directory_blocks: dict[str, int]
    dir_children: dict[str, set[str]]
    next_inode_block: int
    next_directory_block: int
    journal_cursor: int = 0


def clone_state(state: FileSystemState) -> FileSystemState:
    return FileSystemState(
        block_size=state.block_size,
        total_blocks=state.total_blocks,
        superblock_blocks=state.superblock_blocks,
        journal_blocks=state.journal_blocks,
        inode_table_blocks=state.inode_table_blocks,
        directory_region_blocks=state.directory_region_blocks,
        bitmap_blocks=state.bitmap_blocks,
        journal_start=state.journal_start,
        inode_table_start=state.inode_table_start,
        directory_start=state.directory_start,
        bitmap_start=state.bitmap_start,
        data_start_block=state.data_start_block,
        bitmap=bytearray(state.bitmap),
        files={
            path: FileInode(
                path=inode.path,
                inode_block=inode.inode_block,
                parent_dir=inode.parent_dir,
                extents=list(inode.extents),
                size=inode.size,
            )
            for path, inode in state.files.items()
        },
        directories={
            path: DirectoryInode(
                path=directory.path,
                inode_block=directory.inode_block,
                parent_dir=directory.parent_dir,
                dir_block=directory.dir_block,
            )
            for path, directory in state.directories.items()
        },
        directory_blocks=dict(state.directory_blocks),
        dir_children={path: set(children) for path, children in state.dir_children.items()},
        next_inode_block=state.next_inode_block,
        next_directory_block=state.next_directory_block,
        journal_cursor=state.journal_cursor,
    )


def normalize_path(path: str) -> str:
    normalized = posixpath.normpath((path or "").replace("\\", "/"))
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    if normalized in ("", "."):
        return "/"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized


def basename(path: str) -> str:
    normalized = normalize_path(path)
    if normalized == "/":
        return ""
    return posixpath.basename(normalized.rstrip("/"))


def parent_dir(path: str) -> str:
    normalized = normalize_path(path)
    parent = posixpath.dirname(normalized.rstrip("/")) or "/"
    return parent if parent.startswith("/") else f"/{parent}"


def create_filesystem_state(
    total_gb: float = 10,
    block_size: int = 4096,
    *,
    superblock_blocks: int = 1024,
    journal_blocks: int = 4096,
    inode_table_blocks: int = 4096,
    directory_blocks: int = 2048,
    bitmap_blocks: int = 256,
) -> FileSystemState:
    total_blocks = int((total_gb * 1024 * 1024 * 1024) // block_size)
    journal_start = superblock_blocks
    inode_table_start = journal_start + journal_blocks
    directory_start = inode_table_start + inode_table_blocks
    bitmap_start = directory_start + directory_blocks
    data_start_block = bitmap_start + bitmap_blocks
    if total_blocks <= data_start_block:
        raise ValueError("filesystem too small for reserved metadata regions")

    bitmap = bytearray(total_blocks)
    for block in range(data_start_block):
        bitmap[block] = 1

    root_inode_block = inode_table_start
    root_dir = DirectoryInode("/", root_inode_block, "/", directory_start)
    return FileSystemState(
        block_size=block_size,
        total_blocks=total_blocks,
        superblock_blocks=superblock_blocks,
        journal_blocks=journal_blocks,
        inode_table_blocks=inode_table_blocks,
        directory_region_blocks=directory_blocks,
        bitmap_blocks=bitmap_blocks,
        journal_start=journal_start,
        inode_table_start=inode_table_start,
        directory_start=directory_start,
        bitmap_start=bitmap_start,
        data_start_block=data_start_block,
        bitmap=bitmap,
        files={},
        directories={"/": root_dir},
        directory_blocks={"/": root_dir.dir_block},
        dir_children={"/": set()},
        next_inode_block=root_inode_block + 1,
        next_directory_block=directory_start + 1,
    )


def allocate_inode_block(state: FileSystemState) -> tuple[FileSystemState, int]:
    if state.next_inode_block >= state.inode_table_start + state.inode_table_blocks:
        raise RuntimeError("inode table exhausted")
    block = state.next_inode_block
    return replace(state, next_inode_block=state.next_inode_block + 1), block


def allocate_directory_block(state: FileSystemState) -> tuple[FileSystemState, int]:
    if state.next_directory_block >= state.directory_start + state.directory_region_blocks:
        raise RuntimeError("directory metadata area exhausted")
    block = state.next_directory_block
    return replace(state, next_directory_block=state.next_directory_block + 1), block


def journal_op(state: FileSystemState, block_count: int, source: str) -> tuple[FileSystemState, IOOperation]:
    bounded_count = max(1, int(block_count))
    start = state.journal_start + state.journal_cursor
    next_cursor = (state.journal_cursor + bounded_count) % state.journal_blocks
    next_state = replace(state, journal_cursor=next_cursor)
    return next_state, IOOperation(start, bounded_count, "journal", source)


def bitmap_ops_for_extents(state: FileSystemState, extents: list[Extent]) -> list[IOOperation]:
    if not extents:
        return []

    bitmap_blocks = set()
    bits_per_bitmap_block = state.block_size * 8
    for _, start, length in extents:
        first = max(0, start - state.data_start_block)
        last = max(0, start + length - 1 - state.data_start_block)
        first_bitmap = first // bits_per_bitmap_block
        last_bitmap = last // bits_per_bitmap_block
        for idx in range(first_bitmap, last_bitmap + 1):
            bitmap_blocks.add(state.bitmap_start + idx)

    return [IOOperation(block, 1, "metadata", "block_bitmap") for block in sorted(bitmap_blocks)]


def allocate_blocks(state: FileSystemState, count: int) -> tuple[FileSystemState, list[BlockRun]]:
    if count <= 0:
        return state, []

    next_state = clone_state(state)
    allocated: list[BlockRun] = []
    found_count = 0
    current_extent_start = -1
    current_extent_len = 0

    for block in range(next_state.data_start_block, next_state.total_blocks):
        if next_state.bitmap[block] == 0:
            if current_extent_start == -1:
                current_extent_start = block
            current_extent_len += 1
            next_state.bitmap[block] = 1
            found_count += 1
            if found_count == count:
                allocated.append((current_extent_start, current_extent_len))
                return next_state, allocated
        elif current_extent_start != -1:
            allocated.append((current_extent_start, current_extent_len))
            current_extent_start = -1
            current_extent_len = 0

    if current_extent_start != -1:
        allocated.append((current_extent_start, current_extent_len))

    raise OSError("disk full")


def free_extents(state: FileSystemState, extents: list[Extent]) -> FileSystemState:
    next_state = clone_state(state)
    for _, start, length in extents:
        for block in range(start, start + length):
            next_state.bitmap[block] = 0
    return next_state


def coalesce_extents(inode: FileInode) -> FileInode:
    next_inode = FileInode(
        path=inode.path,
        inode_block=inode.inode_block,
        parent_dir=inode.parent_dir,
        extents=list(inode.extents),
        size=inode.size,
    )
    next_inode.extents.sort(key=lambda item: item[0])
    merged: list[Extent] = []
    for logical_start, physical_start, length in next_inode.extents:
        if not merged:
            merged.append((logical_start, physical_start, length))
            continue

        prev_logical, prev_physical, prev_length = merged[-1]
        if prev_logical + prev_length == logical_start and prev_physical + prev_length == physical_start:
            merged[-1] = (prev_logical, prev_physical, prev_length + length)
        else:
            merged.append((logical_start, physical_start, length))
    next_inode.extents = merged
    return next_inode


def range_to_ops(state: FileSystemState, inode: FileInode, offset: int, length: int) -> list[IOOperation]:
    if length <= 0:
        return []

    start_block = offset // state.block_size
    end_block = (offset + length - 1) // state.block_size

    operations = []
    for logical_start, physical_start, extent_len in inode.extents:
        extent_end_block = logical_start + extent_len - 1
        overlap_start = max(start_block, logical_start)
        overlap_end = min(end_block, extent_end_block)

        if overlap_start <= overlap_end:
            lba_offset = overlap_start - logical_start
            block_count = overlap_end - overlap_start + 1
            operations.append(IOOperation(physical_start + lba_offset, block_count, "data", "file_data"))

    return operations


def missing_logical_ranges(inode: FileInode, start_block: int, end_block: int) -> list[tuple[int, int]]:
    if start_block > end_block:
        return []

    missing = []
    cursor = start_block
    for logical_start, _, extent_len in sorted(inode.extents, key=lambda item: item[0]):
        logical_end = logical_start + extent_len - 1
        if logical_end < cursor:
            continue
        if logical_start > end_block:
            break
        if logical_start > cursor:
            missing.append((cursor, logical_start - 1))
        cursor = max(cursor, logical_end + 1)
        if cursor > end_block:
            break

    if cursor <= end_block:
        missing.append((cursor, end_block))
    return missing


def allocate_missing_ranges(
    state: FileSystemState,
    inode: FileInode,
    start_block: int,
    end_block: int,
) -> tuple[FileSystemState, FileInode, list[Extent]]:
    next_state = clone_state(state)
    next_inode = FileInode(
        path=inode.path,
        inode_block=inode.inode_block,
        parent_dir=inode.parent_dir,
        extents=list(inode.extents),
        size=inode.size,
    )
    new_extents: list[Extent] = []
    for missing_start, missing_end in missing_logical_ranges(next_inode, start_block, end_block):
        blocks_needed = missing_end - missing_start + 1
        next_state, runs = allocate_blocks(next_state, blocks_needed)
        for physical_start, length in runs:
            new_extents.append((missing_start, physical_start, length))
            missing_start += length
    next_inode.extents.extend(new_extents)
    next_inode = coalesce_extents(next_inode)
    return next_state, next_inode, new_extents


def ensure_directory_entry(state: FileSystemState, path: str) -> tuple[FileSystemState, DirectoryInode]:
    normalized = normalize_path(path)
    if normalized in state.directories:
        return state, state.directories[normalized]
    if normalized in state.files:
        raise NotADirectoryError(normalized)
    if normalized == "/":
        return state, state.directories["/"]

    next_state = clone_state(state)
    parent_path = parent_dir(normalized)
    next_state, parent_entry = ensure_directory_entry(next_state, parent_path)
    next_state, inode_block = allocate_inode_block(next_state)
    next_state, dir_block = allocate_directory_block(next_state)
    entry = DirectoryInode(normalized, inode_block, parent_entry.path, dir_block)
    next_state.directories[normalized] = entry
    next_state.directory_blocks[normalized] = entry.dir_block
    next_state.dir_children[normalized] = set()
    next_state.dir_children[parent_entry.path].add(basename(normalized))
    return next_state, entry


def rename_metadata_ops(
    state: FileSystemState,
    old_parent: str,
    new_parent: str,
    inode_block: int,
    source: str,
) -> tuple[FileSystemState, list[IOOperation]]:
    next_state, journal = journal_op(state, 2, source)
    operations = [journal]
    if old_parent == new_parent:
        operations.append(IOOperation(next_state.directories[old_parent].dir_block, 1, "metadata", "dir_rename"))
    else:
        operations.extend(
            [
                IOOperation(next_state.directories[old_parent].dir_block, 1, "metadata", "dir_remove"),
                IOOperation(next_state.directories[new_parent].dir_block, 1, "metadata", "dir_insert"),
            ]
        )
    operations.append(IOOperation(inode_block, 1, "metadata", "inode_rename"))
    return next_state, operations


def lookup(state: FileSystemState, path: str) -> list[IOOperation]:
    normalized = normalize_path(path)
    if normalized == "/":
        root = state.directories["/"]
        return [
            IOOperation(root.dir_block, 1, "metadata", "dentry_lookup"),
            IOOperation(root.inode_block, 1, "metadata", "inode_lookup"),
        ]

    parent = state.directories.get(parent_dir(normalized))
    if not parent:
        return []

    operations = [IOOperation(parent.dir_block, 1, "metadata", "dentry_lookup")]
    if normalized in state.files:
        operations.append(IOOperation(state.files[normalized].inode_block, 1, "metadata", "inode_lookup"))
    elif normalized in state.directories:
        operations.append(IOOperation(state.directories[normalized].inode_block, 1, "metadata", "inode_lookup"))
    return operations


def list_directory(state: FileSystemState, path: str) -> list[IOOperation]:
    normalized = normalize_path(path)
    directory = state.directories.get(normalized)
    if not directory:
        return []
    return [IOOperation(directory.dir_block, 1, "metadata", "readdir")]


def materialize_existing_directory(state: FileSystemState, path: str) -> tuple[FileSystemState, DirectoryInode]:
    return ensure_directory_entry(state, path)


def materialize_existing_file(state: FileSystemState, path: str, size: int) -> tuple[FileSystemState, FileInode]:
    next_state = clone_state(state)
    normalized = normalize_path(path)
    if normalized in next_state.directories:
        raise IsADirectoryError(normalized)
    if normalized in next_state.files:
        return next_state, next_state.files[normalized]

    next_state, parent_entry = ensure_directory_entry(next_state, parent_dir(normalized))
    next_state, inode_block = allocate_inode_block(next_state)
    inode = FileInode(normalized, inode_block, parent_entry.path, size=max(0, int(size)))
    next_state.files[normalized] = inode
    next_state.dir_children[parent_entry.path].add(basename(normalized))

    if inode.size > 0:
        total_blocks = math.ceil(inode.size / next_state.block_size)
        next_state, next_inode, missing = allocate_missing_ranges(next_state, inode, 0, total_blocks - 1)
        if not missing and total_blocks > 0:
            raise OSError("disk full")
        next_state.files[normalized] = next_inode
        return next_state, next_inode
    return next_state, inode


def create_directory(state: FileSystemState, path: str) -> tuple[FileSystemState, list[IOOperation]]:
    normalized = normalize_path(path)
    if normalized == "/":
        return state, []
    if normalized in state.files or normalized in state.directories:
        raise FileExistsError(normalized)

    parent_path = parent_dir(normalized)
    if parent_path not in state.directories:
        raise FileNotFoundError(parent_path)

    next_state = clone_state(state)
    parent_entry = next_state.directories[parent_path]
    next_state, inode_block = allocate_inode_block(next_state)
    next_state, dir_block = allocate_directory_block(next_state)
    directory = DirectoryInode(normalized, inode_block, parent_entry.path, dir_block)
    next_state.directories[normalized] = directory
    next_state.directory_blocks[normalized] = directory.dir_block
    next_state.dir_children[normalized] = set()
    next_state.dir_children[parent_entry.path].add(basename(normalized))

    next_state, journal = journal_op(next_state, 2, "mkdir_intent")
    return next_state, [
        journal,
        IOOperation(parent_entry.dir_block, 1, "metadata", "dir_insert"),
        IOOperation(directory.inode_block, 1, "metadata", "inode_create"),
        IOOperation(directory.dir_block, 1, "metadata", "dir_init"),
    ]


def update_directory(
    state: FileSystemState,
    path: str,
    source: str = "dir_attr_update",
) -> tuple[FileSystemState, list[IOOperation]]:
    normalized = normalize_path(path)
    if normalized not in state.directories:
        return state, []
    next_state = clone_state(state)
    directory = next_state.directories[normalized]
    next_state, journal = journal_op(next_state, 1, source)
    return next_state, [
        journal,
        IOOperation(directory.inode_block, 1, "metadata", "inode_update"),
        IOOperation(directory.dir_block, 1, "metadata", "dir_metadata"),
    ]


def update_file_metadata(
    state: FileSystemState,
    path: str,
    source: str = "file_attr_update",
) -> tuple[FileSystemState, list[IOOperation]]:
    normalized = normalize_path(path)
    if normalized not in state.files:
        return state, []
    next_state = clone_state(state)
    inode = next_state.files[normalized]
    next_state, journal = journal_op(next_state, 1, source)
    return next_state, [
        journal,
        IOOperation(inode.inode_block, 1, "metadata", "inode_metadata"),
    ]


def create_empty_file(state: FileSystemState, path: str) -> tuple[FileSystemState, list[IOOperation]]:
    normalized = normalize_path(path)
    if normalized in state.directories:
        raise IsADirectoryError(normalized)
    if normalized in state.files:
        return state, []

    parent_path = parent_dir(normalized)
    if parent_path not in state.directories:
        raise FileNotFoundError(parent_path)

    next_state = clone_state(state)
    parent_entry = next_state.directories[parent_path]
    next_state, inode_block = allocate_inode_block(next_state)
    inode = FileInode(normalized, inode_block, parent_entry.path)
    next_state.files[normalized] = inode
    next_state.dir_children[parent_entry.path].add(basename(normalized))
    next_state, journal = journal_op(next_state, 2, "create_intent")
    return next_state, [
        journal,
        IOOperation(parent_entry.dir_block, 1, "metadata", "dir_insert"),
        IOOperation(inode.inode_block, 1, "metadata", "inode_create"),
    ]


def write(state: FileSystemState, path: str, offset: int, length: int) -> tuple[FileSystemState, list[IOOperation]]:
    normalized = normalize_path(path)
    if normalized in state.directories:
        raise IsADirectoryError(normalized)

    parent_path = parent_dir(normalized)
    if parent_path not in state.directories:
        raise FileNotFoundError(parent_path)

    next_state = clone_state(state)
    parent_entry = next_state.directories[parent_path]
    created = False
    if normalized not in next_state.files:
        next_state, inode_block = allocate_inode_block(next_state)
        next_state.files[normalized] = FileInode(normalized, inode_block, parent_entry.path)
        next_state.dir_children[parent_entry.path].add(basename(normalized))
        created = True

    inode = next_state.files[normalized]
    start_block = offset // next_state.block_size
    end_block = (offset + length - 1) // next_state.block_size if length > 0 else start_block - 1
    next_state, next_inode, new_extents = allocate_missing_ranges(next_state, inode, start_block, end_block)
    next_inode.size = max(next_inode.size, offset + length)
    next_state.files[normalized] = next_inode

    next_state, journal = journal_op(next_state, 2 if (created or new_extents) else 1, "write_intent")
    metadata_ops = [
        journal,
        IOOperation(
            next_inode.inode_block,
            1,
            "metadata",
            "inode_create" if created else "inode_update",
        ),
    ]
    if created:
        metadata_ops.append(IOOperation(parent_entry.dir_block, 1, "metadata", "dir_insert"))
    if new_extents:
        metadata_ops.extend(bitmap_ops_for_extents(next_state, new_extents))

    data_ops = range_to_ops(next_state, next_inode, offset, length)
    return next_state, metadata_ops + data_ops


def read(state: FileSystemState, path: str, offset: int, length: int) -> list[IOOperation]:
    normalized = normalize_path(path)
    if normalized in state.directories:
        raise IsADirectoryError(normalized)
    if normalized not in state.files:
        return []
    inode = state.files[normalized]
    if length <= 0 or offset >= inode.size:
        return []
    clamped_length = min(length, inode.size - offset)
    return range_to_ops(state, inode, offset, clamped_length)


def delete(state: FileSystemState, path: str) -> tuple[FileSystemState, list[IOOperation]]:
    normalized = normalize_path(path)
    if normalized in state.directories:
        raise IsADirectoryError(normalized)
    if normalized not in state.files:
        return state, []

    next_state = clone_state(state)
    inode = next_state.files[normalized]
    parent_entry = next_state.directories[inode.parent_dir]
    next_state, journal = journal_op(next_state, 2, "delete_intent")
    operations = [
        journal,
        IOOperation(parent_entry.dir_block, 1, "metadata", "dir_remove"),
        IOOperation(inode.inode_block, 1, "metadata", "inode_delete"),
    ]
    operations.extend(bitmap_ops_for_extents(next_state, inode.extents))

    next_state = free_extents(next_state, inode.extents)
    next_state.dir_children[parent_entry.path].discard(basename(normalized))
    del next_state.files[normalized]
    return next_state, operations


def delete_directory(
    state: FileSystemState,
    path: str,
    recursive: bool = True,
) -> tuple[FileSystemState, list[IOOperation]]:
    normalized = normalize_path(path)
    if normalized == "/":
        raise PermissionError("cannot delete root directory")
    if normalized not in state.directories:
        return state, []
    if not recursive and state.dir_children.get(normalized):
        raise OSError("directory not empty")

    file_paths = sorted(
        [name for name in state.files if name.startswith(f"{normalized}/")],
        key=lambda item: item.count("/"),
        reverse=True,
    )
    child_dirs = sorted(
        [name for name in state.directories if name != normalized and name.startswith(f"{normalized}/")],
        key=lambda item: item.count("/"),
        reverse=True,
    )
    if not recursive and (file_paths or child_dirs):
        raise OSError("directory not empty")

    next_state = clone_state(state)
    next_state, journal = journal_op(next_state, max(2, 1 + len(file_paths) + len(child_dirs)), "rmdir_intent")
    operations = [journal]

    for file_path in file_paths:
        inode = next_state.files[file_path]
        parent_entry = next_state.directories[inode.parent_dir]
        operations.extend(
            [
                IOOperation(parent_entry.dir_block, 1, "metadata", "dir_remove"),
                IOOperation(inode.inode_block, 1, "metadata", "inode_delete"),
            ]
        )
        operations.extend(bitmap_ops_for_extents(next_state, inode.extents))

    for dir_path in [*child_dirs, normalized]:
        directory = next_state.directories[dir_path]
        parent_entry = next_state.directories[directory.parent_dir]
        operations.extend(
            [
                IOOperation(parent_entry.dir_block, 1, "metadata", "dir_remove"),
                IOOperation(directory.dir_block, 1, "metadata", "dir_teardown"),
                IOOperation(directory.inode_block, 1, "metadata", "inode_delete"),
            ]
        )

    for file_path in file_paths:
        inode = next_state.files.pop(file_path)
        next_state = free_extents(next_state, inode.extents)
        next_state.dir_children[inode.parent_dir].discard(basename(file_path))

    for dir_path in [*child_dirs, normalized]:
        directory = next_state.directories.pop(dir_path)
        next_state.directory_blocks.pop(dir_path, None)
        next_state.dir_children.pop(dir_path, None)
        next_state.dir_children[directory.parent_dir].discard(basename(dir_path))

    return next_state, operations


def rename(state: FileSystemState, source_path: str, dest_path: str) -> tuple[FileSystemState, list[IOOperation]]:
    source = normalize_path(source_path)
    dest = normalize_path(dest_path)
    if source == dest:
        return state, []
    if dest in state.files or dest in state.directories:
        raise FileExistsError(dest)

    dest_parent_path = parent_dir(dest)
    if dest_parent_path not in state.directories:
        raise FileNotFoundError(dest_parent_path)

    next_state = clone_state(state)
    if source in next_state.files:
        inode = next_state.files.pop(source)
        old_parent = inode.parent_dir
        next_state.dir_children[old_parent].discard(basename(source))
        next_state.dir_children[dest_parent_path].add(basename(dest))
        inode.path = dest
        inode.parent_dir = dest_parent_path
        next_state.files[dest] = inode
        return rename_metadata_ops(next_state, old_parent, dest_parent_path, inode.inode_block, "rename_intent")

    if source not in next_state.directories:
        return state, []
    if source == "/" or dest.startswith(f"{source}/"):
        raise ValueError("cannot move a directory into itself")

    root_dir = next_state.directories[source]
    old_parent = root_dir.parent_dir
    next_state.dir_children[old_parent].discard(basename(source))
    next_state.dir_children[dest_parent_path].add(basename(dest))

    dir_paths = sorted(
        [path for path in next_state.directories if path == source or path.startswith(f"{source}/")],
        key=len,
    )
    file_paths = sorted([path for path in next_state.files if path.startswith(f"{source}/")], key=len)

    path_map = {old_path: dest + old_path[len(source):] for old_path in dir_paths + file_paths}

    remapped_children = {}
    for old_path in dir_paths:
        remapped_children[path_map[old_path]] = next_state.dir_children.pop(old_path)

    remapped_dirs = {}
    for old_path in dir_paths:
        directory = next_state.directories.pop(old_path)
        directory.path = path_map[old_path]
        directory.parent_dir = dest_parent_path if old_path == source else path_map[directory.parent_dir]
        remapped_dirs[directory.path] = directory

    remapped_files = {}
    for old_path in file_paths:
        inode = next_state.files.pop(old_path)
        inode.path = path_map[old_path]
        inode.parent_dir = path_map[inode.parent_dir]
        remapped_files[inode.path] = inode

    next_state.dir_children.update(remapped_children)
    next_state.directories.update(remapped_dirs)
    next_state.files.update(remapped_files)
    next_state.directory_blocks = {
        directory.path: directory.dir_block for directory in next_state.directories.values()
    }

    next_state, operations = rename_metadata_ops(
        next_state,
        old_parent,
        dest_parent_path,
        root_dir.inode_block,
        "rename_intent",
    )
    if old_parent != dest_parent_path:
        operations.append(IOOperation(root_dir.dir_block, 1, "metadata", "dir_parent_update"))
    return next_state, operations


def truncate(state: FileSystemState, path: str, size: int = 0) -> tuple[FileSystemState, list[IOOperation]]:
    normalized = normalize_path(path)
    if normalized in state.directories:
        raise IsADirectoryError(normalized)
    if normalized not in state.files:
        return state, []
    if size < 0:
        raise ValueError("size must be non-negative")

    next_state = clone_state(state)
    inode = next_state.files[normalized]
    if size == inode.size:
        return state, []

    freed_extents = []
    if size <= 0:
        freed_extents = list(inode.extents)
        inode.extents = []
        inode.size = 0
    elif size < inode.size:
        last_block = (size - 1) // next_state.block_size
        retained = []
        for logical_start, physical_start, block_count in inode.extents:
            logical_end = logical_start + block_count - 1
            if logical_start > last_block:
                freed_extents.append((logical_start, physical_start, block_count))
                continue
            if logical_end <= last_block:
                retained.append((logical_start, physical_start, block_count))
                continue

            keep_blocks = last_block - logical_start + 1
            retained.append((logical_start, physical_start, keep_blocks))
            freed_extents.append(
                (logical_start + keep_blocks, physical_start + keep_blocks, block_count - keep_blocks)
            )
        inode.extents = retained
        inode.size = size
    else:
        inode.size = size

    if freed_extents:
        next_state = free_extents(next_state, freed_extents)

    next_state, journal = journal_op(next_state, 2 if freed_extents else 1, "truncate_intent")
    operations = [
        journal,
        IOOperation(inode.inode_block, 1, "metadata", "inode_truncate"),
    ]
    if freed_extents:
        operations.extend(bitmap_ops_for_extents(next_state, freed_extents))
    return next_state, operations


def get_fragmentation_score(state: FileSystemState, path: str) -> int:
    normalized = normalize_path(path)
    if normalized in state.files:
        return len(state.files[normalized].extents)
    return 0


def assert_consistent(state: FileSystemState) -> None:
    assert "/" in state.directories
    assert set(state.directory_blocks) == set(state.directories)
    assert set(state.dir_children) == set(state.directories)

    expected_children: dict[str, set[str]] = {path: set() for path in state.directories}
    seen_inode_blocks = set()
    seen_dir_blocks = set()
    seen_data_blocks = set()

    for path, directory in state.directories.items():
        assert path == directory.path
        assert state.directory_blocks[path] == directory.dir_block
        assert directory.inode_block not in seen_inode_blocks
        assert directory.dir_block not in seen_dir_blocks
        seen_inode_blocks.add(directory.inode_block)
        seen_dir_blocks.add(directory.dir_block)
        if path == "/":
            assert directory.parent_dir == "/"
            continue
        assert directory.parent_dir in state.directories
        expected_children[directory.parent_dir].add(basename(path))

    for path, inode in state.files.items():
        assert path == inode.path
        assert path not in state.directories
        assert inode.parent_dir in state.directories
        assert inode.inode_block not in seen_inode_blocks
        seen_inode_blocks.add(inode.inode_block)
        expected_children[inode.parent_dir].add(basename(path))

        logical_end = -1
        for logical_start, physical_start, block_count in inode.extents:
            assert block_count > 0
            assert logical_start > logical_end
            assert physical_start >= state.data_start_block
            assert physical_start + block_count <= state.total_blocks
            logical_end = logical_start + block_count - 1
            for block in range(physical_start, physical_start + block_count):
                assert state.bitmap[block] == 1
                assert block not in seen_data_blocks
                seen_data_blocks.add(block)

    actual_children = {path: set(children) for path, children in state.dir_children.items()}
    assert actual_children == expected_children
