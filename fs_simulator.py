from __future__ import annotations

import math
import posixpath
import threading
from dataclasses import dataclass

Extent = tuple[int, int, int]
BlockRun = tuple[int, int]


@dataclass(frozen=True)
class IOOperation:
    lba: int
    block_count: int
    kind: str
    source: str = ""


class FileInode:
    def __init__(self, path: str, inode_block: int, parent_dir: str) -> None:
        self.path = path
        self.inode_block = inode_block
        self.parent_dir = parent_dir
        # (logical_file_block, physical_lba, num_blocks)
        self.extents: list[Extent] = []
        self.size = 0


class DirectoryInode:
    def __init__(self, path: str, inode_block: int, parent_dir: str, dir_block: int) -> None:
        self.path = path
        self.inode_block = inode_block
        self.parent_dir = parent_dir
        self.dir_block = dir_block


class FileSystemSimulator:
    """
    Simulates a block-based filesystem with lightweight metadata traffic.

    The model is intentionally compact rather than fully faithful, but it now
    tracks an explicit directory tree so lookups, rename/move, and recursive
    deletion behave like filesystem operations instead of loose path rewrites.
    """

    SUPERBLOCK_BLOCKS = 1024
    JOURNAL_BLOCKS = 4096
    INODE_TABLE_BLOCKS = 4096
    DIRECTORY_BLOCKS = 2048
    BITMAP_BLOCKS = 256

    def __init__(self, total_gb: float = 10, block_size: int = 4096) -> None:
        self.block_size = block_size
        self.total_blocks = int((total_gb * 1024 * 1024 * 1024) // block_size)

        self.journal_start = self.SUPERBLOCK_BLOCKS
        self.inode_table_start = self.journal_start + self.JOURNAL_BLOCKS
        self.directory_start = self.inode_table_start + self.INODE_TABLE_BLOCKS
        self.bitmap_start = self.directory_start + self.DIRECTORY_BLOCKS
        self.data_start_block = self.bitmap_start + self.BITMAP_BLOCKS
        if self.total_blocks <= self.data_start_block:
            raise ValueError("filesystem too small for reserved metadata regions")

        # 0 = free, 1 = used
        self.bitmap = bytearray(self.total_blocks)
        for block in range(self.data_start_block):
            self.bitmap[block] = 1

        self.files: dict[str, FileInode] = {}  # path -> FileInode
        self.directories: dict[str, DirectoryInode] = {}  # path -> DirectoryInode
        self.directory_blocks: dict[str, int] = {}
        self.dir_children: dict[str, set[str]] = {}  # path -> set(child names)
        self.next_inode_block = self.inode_table_start
        self.next_directory_block = self.directory_start + 1
        self.journal_cursor = 0
        self.lock = threading.Lock()

        root_inode = self._allocate_inode_block()
        root_dir = DirectoryInode("/", root_inode, "/", self.directory_start)
        self.directories["/"] = root_dir
        self.directory_blocks["/"] = root_dir.dir_block
        self.dir_children["/"] = set()

    def _normalize_path(self, path: str) -> str:
        normalized = posixpath.normpath((path or "").replace("\\", "/"))
        if normalized.startswith("//"):
            normalized = "/" + normalized.lstrip("/")
        if normalized in ("", "."):
            return "/"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized

    def _basename(self, path: str) -> str:
        normalized = self._normalize_path(path)
        if normalized == "/":
            return ""
        return posixpath.basename(normalized.rstrip("/"))

    def _parent_dir(self, path: str) -> str:
        normalized = self._normalize_path(path)
        parent = posixpath.dirname(normalized.rstrip("/")) or "/"
        return parent if parent.startswith("/") else f"/{parent}"

    def _allocate_inode_block(self) -> int:
        if self.next_inode_block >= self.inode_table_start + self.INODE_TABLE_BLOCKS:
            raise RuntimeError("inode table exhausted")
        block = self.next_inode_block
        self.next_inode_block += 1
        return block

    def _allocate_directory_block(self) -> int:
        if self.next_directory_block >= self.directory_start + self.DIRECTORY_BLOCKS:
            raise RuntimeError("directory metadata area exhausted")
        block = self.next_directory_block
        self.next_directory_block += 1
        return block

    def _journal_op(self, block_count: int, source: str) -> IOOperation:
        block_count = max(1, int(block_count))
        start = self.journal_start + self.journal_cursor
        self.journal_cursor = (self.journal_cursor + block_count) % self.JOURNAL_BLOCKS
        return IOOperation(start, block_count, "journal", source)

    def _bitmap_ops_for_extents(self, extents: list[Extent]) -> list[IOOperation]:
        if not extents:
            return []

        bitmap_blocks = set()
        bits_per_bitmap_block = self.block_size * 8
        for _, start, length in extents:
            first = max(0, start - self.data_start_block)
            last = max(0, start + length - 1 - self.data_start_block)
            first_bitmap = first // bits_per_bitmap_block
            last_bitmap = last // bits_per_bitmap_block
            for idx in range(first_bitmap, last_bitmap + 1):
                bitmap_blocks.add(self.bitmap_start + idx)

        return [
            IOOperation(block, 1, "metadata", "block_bitmap")
            for block in sorted(bitmap_blocks)
        ]

    def _allocate_blocks(self, count: int) -> list[BlockRun]:
        if count <= 0:
            return []

        allocated = []
        found_count = 0
        current_extent_start = -1
        current_extent_len = 0

        for block in range(self.data_start_block, self.total_blocks):
            if self.bitmap[block] == 0:
                if current_extent_start == -1:
                    current_extent_start = block
                current_extent_len += 1
                self.bitmap[block] = 1
                found_count += 1
                if found_count == count:
                    allocated.append((current_extent_start, current_extent_len))
                    return allocated
            elif current_extent_start != -1:
                allocated.append((current_extent_start, current_extent_len))
                current_extent_start = -1
                current_extent_len = 0

        if current_extent_start != -1:
            allocated.append((current_extent_start, current_extent_len))

        for start, length in allocated:
            for block in range(start, start + length):
                self.bitmap[block] = 0
        raise OSError("disk full")

    def _free_extents(self, extents: list[Extent]) -> None:
        for _, start, length in extents:
            for block in range(start, start + length):
                self.bitmap[block] = 0

    def _coalesce_extents(self, inode: FileInode) -> None:
        inode.extents.sort(key=lambda item: item[0])
        merged: list[Extent] = []
        for logical_start, physical_start, length in inode.extents:
            if not merged:
                merged.append((logical_start, physical_start, length))
                continue

            prev_logical, prev_physical, prev_length = merged[-1]
            if (
                prev_logical + prev_length == logical_start
                and prev_physical + prev_length == physical_start
            ):
                merged[-1] = (prev_logical, prev_physical, prev_length + length)
            else:
                merged.append((logical_start, physical_start, length))
        inode.extents = merged

    def _range_to_ops(self, inode: FileInode, offset: int, length: int) -> list[IOOperation]:
        if length <= 0:
            return []

        start_block = offset // self.block_size
        end_block = (offset + length - 1) // self.block_size

        operations = []
        for logical_start, physical_start, extent_len in inode.extents:
            extent_end_block = logical_start + extent_len - 1
            overlap_start = max(start_block, logical_start)
            overlap_end = min(end_block, extent_end_block)

            if overlap_start <= overlap_end:
                lba_offset = overlap_start - logical_start
                block_count = overlap_end - overlap_start + 1
                operations.append(
                    IOOperation(physical_start + lba_offset, block_count, "data", "file_data")
                )

        return operations

    def _missing_logical_ranges(
        self,
        inode: FileInode,
        start_block: int,
        end_block: int,
    ) -> list[tuple[int, int]]:
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

    def _allocate_missing_ranges(self, inode: FileInode, start_block: int, end_block: int) -> list[Extent]:
        new_extents: list[Extent] = []
        for missing_start, missing_end in self._missing_logical_ranges(inode, start_block, end_block):
            blocks_needed = missing_end - missing_start + 1
            for physical_start, length in self._allocate_blocks(blocks_needed):
                new_extents.append((missing_start, physical_start, length))
                missing_start += length
        inode.extents.extend(new_extents)
        self._coalesce_extents(inode)
        return new_extents

    def _ensure_directory_entry(self, path: str) -> DirectoryInode:
        normalized = self._normalize_path(path)
        if normalized in self.directories:
            return self.directories[normalized]
        if normalized in self.files:
            raise NotADirectoryError(normalized)
        if normalized == "/":
            return self.directories["/"]

        parent_path = self._parent_dir(normalized)
        parent_dir = self._ensure_directory_entry(parent_path)
        entry = DirectoryInode(
            normalized,
            self._allocate_inode_block(),
            parent_dir.path,
            self._allocate_directory_block(),
        )
        self.directories[normalized] = entry
        self.directory_blocks[normalized] = entry.dir_block
        self.dir_children[normalized] = set()
        self.dir_children[parent_dir.path].add(self._basename(normalized))
        return entry

    def _rename_metadata_ops(
        self,
        old_parent: str,
        new_parent: str,
        inode_block: int,
        source: str,
    ) -> list[IOOperation]:
        operations = [self._journal_op(2, source)]
        if old_parent == new_parent:
            operations.append(
                IOOperation(self.directories[old_parent].dir_block, 1, "metadata", "dir_rename")
            )
        else:
            operations.extend(
                [
                    IOOperation(self.directories[old_parent].dir_block, 1, "metadata", "dir_remove"),
                    IOOperation(self.directories[new_parent].dir_block, 1, "metadata", "dir_insert"),
                ]
            )
        operations.append(IOOperation(inode_block, 1, "metadata", "inode_rename"))
        return operations

    def lookup(self, path: str) -> list[IOOperation]:
        path = self._normalize_path(path)
        if path == "/":
            root = self.directories["/"]
            return [
                IOOperation(root.dir_block, 1, "metadata", "dentry_lookup"),
                IOOperation(root.inode_block, 1, "metadata", "inode_lookup"),
            ]

        parent_dir = self.directories.get(self._parent_dir(path))
        if not parent_dir:
            return []

        operations = [IOOperation(parent_dir.dir_block, 1, "metadata", "dentry_lookup")]
        if path in self.files:
            operations.append(
                IOOperation(self.files[path].inode_block, 1, "metadata", "inode_lookup")
            )
        elif path in self.directories:
            operations.append(
                IOOperation(self.directories[path].inode_block, 1, "metadata", "inode_lookup")
            )
        return operations

    def list_directory(self, path: str) -> list[IOOperation]:
        path = self._normalize_path(path)
        directory = self.directories.get(path)
        if not directory:
            return []
        return [IOOperation(directory.dir_block, 1, "metadata", "readdir")]

    def materialize_existing_directory(self, path: str) -> DirectoryInode:
        with self.lock:
            return self._ensure_directory_entry(path)

    def materialize_existing_file(self, path: str, size: int) -> FileInode:
        with self.lock:
            path = self._normalize_path(path)
            if path in self.directories:
                raise IsADirectoryError(path)
            if path in self.files:
                return self.files[path]

            parent_dir = self._ensure_directory_entry(self._parent_dir(path))
            inode = FileInode(path, self._allocate_inode_block(), parent_dir.path)
            inode.size = max(0, int(size))
            self.files[path] = inode
            self.dir_children[parent_dir.path].add(self._basename(path))

            if inode.size > 0:
                total_blocks = math.ceil(inode.size / self.block_size)
                missing = self._allocate_missing_ranges(inode, 0, total_blocks - 1)
                if not missing and total_blocks > 0:
                    raise OSError("disk full")
            return inode

    def create_directory(self, path: str) -> list[IOOperation]:
        with self.lock:
            path = self._normalize_path(path)
            if path == "/":
                return []
            if path in self.files or path in self.directories:
                raise FileExistsError(path)

            parent_path = self._parent_dir(path)
            if parent_path not in self.directories:
                raise FileNotFoundError(parent_path)
            parent_dir = self.directories[parent_path]

            directory = DirectoryInode(
                path,
                self._allocate_inode_block(),
                parent_dir.path,
                self._allocate_directory_block(),
            )
            self.directories[path] = directory
            self.directory_blocks[path] = directory.dir_block
            self.dir_children[path] = set()
            self.dir_children[parent_dir.path].add(self._basename(path))

            return [
                self._journal_op(2, "mkdir_intent"),
                IOOperation(parent_dir.dir_block, 1, "metadata", "dir_insert"),
                IOOperation(directory.inode_block, 1, "metadata", "inode_create"),
                IOOperation(directory.dir_block, 1, "metadata", "dir_init"),
            ]

    def update_directory(self, path: str, source: str = "dir_attr_update") -> list[IOOperation]:
        with self.lock:
            path = self._normalize_path(path)
            if path not in self.directories:
                return []
            directory = self.directories[path]
            return [
                self._journal_op(1, source),
                IOOperation(directory.inode_block, 1, "metadata", "inode_update"),
                IOOperation(directory.dir_block, 1, "metadata", "dir_metadata"),
            ]

    def create_empty_file(self, path: str) -> list[IOOperation]:
        with self.lock:
            path = self._normalize_path(path)
            if path in self.directories:
                raise IsADirectoryError(path)
            if path in self.files:
                return []

            parent_path = self._parent_dir(path)
            if parent_path not in self.directories:
                raise FileNotFoundError(parent_path)
            parent_dir = self.directories[parent_path]

            inode = FileInode(path, self._allocate_inode_block(), parent_dir.path)
            self.files[path] = inode
            self.dir_children[parent_dir.path].add(self._basename(path))

            return [
                self._journal_op(2, "create_intent"),
                IOOperation(parent_dir.dir_block, 1, "metadata", "dir_insert"),
                IOOperation(inode.inode_block, 1, "metadata", "inode_create"),
            ]

    def write(self, path: str, offset: int, length: int) -> list[IOOperation]:
        """
        Simulates a buffered write and returns metadata + data blocks touched.
        """
        with self.lock:
            path = self._normalize_path(path)
            if path in self.directories:
                raise IsADirectoryError(path)

            parent_path = self._parent_dir(path)
            if parent_path not in self.directories:
                raise FileNotFoundError(parent_path)
            parent_dir = self.directories[parent_path]
            created = False

            if path not in self.files:
                self.files[path] = FileInode(path, self._allocate_inode_block(), parent_dir.path)
                self.dir_children[parent_dir.path].add(self._basename(path))
                created = True

            inode = self.files[path]
            start_block = offset // self.block_size
            end_block = (offset + length - 1) // self.block_size if length > 0 else start_block - 1
            new_extents = self._allocate_missing_ranges(inode, start_block, end_block)
            inode.size = max(inode.size, offset + length)

            metadata_ops = [
                self._journal_op(2 if (created or new_extents) else 1, "write_intent"),
                IOOperation(
                    inode.inode_block,
                    1,
                    "metadata",
                    "inode_create" if created else "inode_update",
                ),
            ]
            if created:
                metadata_ops.append(IOOperation(parent_dir.dir_block, 1, "metadata", "dir_insert"))
            if new_extents:
                metadata_ops.extend(self._bitmap_ops_for_extents(new_extents))

            data_ops = self._range_to_ops(inode, offset, length)
            return metadata_ops + data_ops

    def read(self, path: str, offset: int, length: int) -> list[IOOperation]:
        with self.lock:
            path = self._normalize_path(path)
            if path in self.directories:
                raise IsADirectoryError(path)
            if path not in self.files:
                return []
            inode = self.files[path]
            if length <= 0 or offset >= inode.size:
                return []
            clamped_length = min(length, inode.size - offset)
            return self._range_to_ops(inode, offset, clamped_length)

    def delete(self, path: str) -> list[IOOperation]:
        with self.lock:
            path = self._normalize_path(path)
            if path in self.directories:
                raise IsADirectoryError(path)
            if path not in self.files:
                return []

            inode = self.files[path]
            parent_dir = self.directories[inode.parent_dir]
            operations = [
                self._journal_op(2, "delete_intent"),
                IOOperation(parent_dir.dir_block, 1, "metadata", "dir_remove"),
                IOOperation(inode.inode_block, 1, "metadata", "inode_delete"),
            ]
            operations.extend(self._bitmap_ops_for_extents(inode.extents))

            self._free_extents(inode.extents)
            self.dir_children[parent_dir.path].discard(self._basename(path))
            del self.files[path]
            return operations

    def delete_directory(self, path: str, recursive: bool = True) -> list[IOOperation]:
        with self.lock:
            path = self._normalize_path(path)
            if path == "/":
                raise PermissionError("cannot delete root directory")
            if path not in self.directories:
                return []
            if not recursive and self.dir_children.get(path):
                raise OSError("directory not empty")

            file_paths = sorted(
                [name for name in self.files if name.startswith(f"{path}/")],
                key=lambda item: item.count("/"),
                reverse=True,
            )
            child_dirs = sorted(
                [name for name in self.directories if name != path and name.startswith(f"{path}/")],
                key=lambda item: item.count("/"),
                reverse=True,
            )
            if not recursive and (file_paths or child_dirs):
                raise OSError("directory not empty")

            operations = [self._journal_op(max(2, 1 + len(file_paths) + len(child_dirs)), "rmdir_intent")]

            for file_path in file_paths:
                inode = self.files[file_path]
                parent_dir = self.directories[inode.parent_dir]
                operations.extend(
                    [
                        IOOperation(parent_dir.dir_block, 1, "metadata", "dir_remove"),
                        IOOperation(inode.inode_block, 1, "metadata", "inode_delete"),
                    ]
                )
                operations.extend(self._bitmap_ops_for_extents(inode.extents))

            for dir_path in [*child_dirs, path]:
                directory = self.directories[dir_path]
                parent_dir = self.directories[directory.parent_dir]
                operations.extend(
                    [
                        IOOperation(parent_dir.dir_block, 1, "metadata", "dir_remove"),
                        IOOperation(directory.dir_block, 1, "metadata", "dir_teardown"),
                        IOOperation(directory.inode_block, 1, "metadata", "inode_delete"),
                    ]
                )

            for file_path in file_paths:
                inode = self.files.pop(file_path)
                self._free_extents(inode.extents)
                self.dir_children[inode.parent_dir].discard(self._basename(file_path))

            for dir_path in [*child_dirs, path]:
                directory = self.directories.pop(dir_path)
                self.directory_blocks.pop(dir_path, None)
                self.dir_children.pop(dir_path, None)
                self.dir_children[directory.parent_dir].discard(self._basename(dir_path))

            return operations

    def rename(self, source_path: str, dest_path: str) -> list[IOOperation]:
        with self.lock:
            source_path = self._normalize_path(source_path)
            dest_path = self._normalize_path(dest_path)
            if source_path == dest_path:
                return []
            if dest_path in self.files or dest_path in self.directories:
                raise FileExistsError(dest_path)

            dest_parent_path = self._parent_dir(dest_path)
            if dest_parent_path not in self.directories:
                raise FileNotFoundError(dest_parent_path)

            if source_path in self.files:
                inode = self.files.pop(source_path)
                old_parent = inode.parent_dir
                self.dir_children[old_parent].discard(self._basename(source_path))
                self.dir_children[dest_parent_path].add(self._basename(dest_path))
                inode.path = dest_path
                inode.parent_dir = dest_parent_path
                self.files[dest_path] = inode
                return self._rename_metadata_ops(old_parent, dest_parent_path, inode.inode_block, "rename_intent")

            if source_path not in self.directories:
                return []
            if source_path == "/" or dest_path.startswith(f"{source_path}/"):
                raise ValueError("cannot move a directory into itself")

            root_dir = self.directories[source_path]
            old_parent = root_dir.parent_dir
            self.dir_children[old_parent].discard(self._basename(source_path))
            self.dir_children[dest_parent_path].add(self._basename(dest_path))

            dir_paths = sorted(
                [path for path in self.directories if path == source_path or path.startswith(f"{source_path}/")],
                key=len,
            )
            file_paths = sorted(
                [path for path in self.files if path.startswith(f"{source_path}/")],
                key=len,
            )

            path_map = {
                old_path: dest_path + old_path[len(source_path):]
                for old_path in dir_paths + file_paths
            }

            remapped_children = {}
            for old_path in dir_paths:
                remapped_children[path_map[old_path]] = self.dir_children.pop(old_path)

            remapped_dirs = {}
            for old_path in dir_paths:
                directory = self.directories.pop(old_path)
                directory.path = path_map[old_path]
                directory.parent_dir = (
                    dest_parent_path if old_path == source_path else path_map[directory.parent_dir]
                )
                remapped_dirs[directory.path] = directory

            remapped_files = {}
            for old_path in file_paths:
                inode = self.files.pop(old_path)
                inode.path = path_map[old_path]
                inode.parent_dir = path_map[inode.parent_dir]
                remapped_files[inode.path] = inode

            self.dir_children.update(remapped_children)
            self.directories.update(remapped_dirs)
            self.files.update(remapped_files)
            self.directory_blocks = {
                directory.path: directory.dir_block for directory in self.directories.values()
            }

            operations = self._rename_metadata_ops(
                old_parent,
                dest_parent_path,
                root_dir.inode_block,
                "rename_intent",
            )
            if old_parent != dest_parent_path:
                operations.append(IOOperation(root_dir.dir_block, 1, "metadata", "dir_parent_update"))
            return operations

    def truncate(self, path: str, size: int = 0) -> list[IOOperation]:
        with self.lock:
            path = self._normalize_path(path)
            if path in self.directories:
                raise IsADirectoryError(path)
            if path not in self.files:
                return []
            if size < 0:
                raise ValueError("size must be non-negative")

            inode = self.files[path]
            if size == inode.size:
                return []

            freed_extents = []
            if size <= 0:
                freed_extents = list(inode.extents)
                inode.extents = []
                inode.size = 0
            elif size < inode.size:
                last_block = (size - 1) // self.block_size
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
                        (
                            logical_start + keep_blocks,
                            physical_start + keep_blocks,
                            block_count - keep_blocks,
                        )
                    )
                inode.extents = retained
                inode.size = size
            else:
                inode.size = size

            if freed_extents:
                self._free_extents(freed_extents)

            operations = [
                self._journal_op(2 if freed_extents else 1, "truncate_intent"),
                IOOperation(inode.inode_block, 1, "metadata", "inode_truncate"),
            ]
            if freed_extents:
                operations.extend(self._bitmap_ops_for_extents(freed_extents))
            return operations

    def get_fragmentation_score(self, path: str) -> int:
        with self.lock:
            path = self._normalize_path(path)
            if path in self.files:
                return len(self.files[path].extents)
            return 0

    def assert_consistent(self) -> None:
        with self.lock:
            assert "/" in self.directories
            assert set(self.directory_blocks) == set(self.directories)
            assert set(self.dir_children) == set(self.directories)

            expected_children: dict[str, set[str]] = {path: set() for path in self.directories}
            seen_inode_blocks = set()
            seen_dir_blocks = set()
            seen_data_blocks = set()

            for path, directory in self.directories.items():
                assert path == directory.path
                assert self.directory_blocks[path] == directory.dir_block
                assert directory.inode_block not in seen_inode_blocks
                assert directory.dir_block not in seen_dir_blocks
                seen_inode_blocks.add(directory.inode_block)
                seen_dir_blocks.add(directory.dir_block)
                if path == "/":
                    assert directory.parent_dir == "/"
                    continue
                assert directory.parent_dir in self.directories
                expected_children[directory.parent_dir].add(self._basename(path))

            for path, inode in self.files.items():
                assert path == inode.path
                assert path not in self.directories
                assert inode.parent_dir in self.directories
                assert inode.inode_block not in seen_inode_blocks
                seen_inode_blocks.add(inode.inode_block)
                expected_children[inode.parent_dir].add(self._basename(path))

                logical_end = -1
                for logical_start, physical_start, block_count in inode.extents:
                    assert block_count > 0
                    assert logical_start > logical_end
                    assert physical_start >= self.data_start_block
                    assert physical_start + block_count <= self.total_blocks
                    logical_end = logical_start + block_count - 1
                    for block in range(physical_start, physical_start + block_count):
                        assert self.bitmap[block] == 1
                        assert block not in seen_data_blocks
                        seen_data_blocks.add(block)

            actual_children = {
                path: set(children)
                for path, children in self.dir_children.items()
            }
            assert actual_children == expected_children
