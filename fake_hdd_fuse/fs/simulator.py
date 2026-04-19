from __future__ import annotations

import threading

from .core import (
    DirectoryInode,
    FileInode,
    FileSystemState,
    IOOperation,
    assert_consistent,
    basename,
    create_directory,
    create_empty_file,
    create_filesystem_state,
    delete,
    delete_directory,
    get_fragmentation_score,
    list_directory,
    lookup,
    materialize_existing_directory,
    materialize_existing_file,
    normalize_path,
    parent_dir,
    read,
    rename,
    truncate,
    update_directory,
    write,
)


class FileSystemSimulator:
    """
    Locked shell around the pure filesystem core.

    The public API remains intentionally stable for the rest of the repo, but
    tree mutation, extent allocation, and metadata planning now happen in
    `fs_core.py` against an explicit `FileSystemState`.
    """

    SUPERBLOCK_BLOCKS = 1024
    JOURNAL_BLOCKS = 4096
    INODE_TABLE_BLOCKS = 4096
    DIRECTORY_BLOCKS = 2048
    BITMAP_BLOCKS = 256

    def __init__(self, total_gb: float = 10, block_size: int = 4096) -> None:
        self.lock = threading.Lock()
        self._state = create_filesystem_state(
            total_gb=total_gb,
            block_size=block_size,
            superblock_blocks=self.SUPERBLOCK_BLOCKS,
            journal_blocks=self.JOURNAL_BLOCKS,
            inode_table_blocks=self.INODE_TABLE_BLOCKS,
            directory_blocks=self.DIRECTORY_BLOCKS,
            bitmap_blocks=self.BITMAP_BLOCKS,
        )

    @property
    def state(self) -> FileSystemState:
        return self._state

    @property
    def block_size(self) -> int:
        return self._state.block_size

    @property
    def total_blocks(self) -> int:
        return self._state.total_blocks

    @property
    def journal_start(self) -> int:
        return self._state.journal_start

    @property
    def inode_table_start(self) -> int:
        return self._state.inode_table_start

    @property
    def directory_start(self) -> int:
        return self._state.directory_start

    @property
    def bitmap_start(self) -> int:
        return self._state.bitmap_start

    @property
    def data_start_block(self) -> int:
        return self._state.data_start_block

    @property
    def bitmap(self) -> bytearray:
        return self._state.bitmap

    @property
    def files(self) -> dict[str, FileInode]:
        return self._state.files

    @property
    def directories(self) -> dict[str, DirectoryInode]:
        return self._state.directories

    @property
    def directory_blocks(self) -> dict[str, int]:
        return self._state.directory_blocks

    @property
    def dir_children(self) -> dict[str, set[str]]:
        return self._state.dir_children

    @property
    def next_inode_block(self) -> int:
        return self._state.next_inode_block

    @property
    def next_directory_block(self) -> int:
        return self._state.next_directory_block

    @property
    def journal_cursor(self) -> int:
        return self._state.journal_cursor

    def _normalize_path(self, path: str) -> str:
        return normalize_path(path)

    def _basename(self, path: str) -> str:
        return basename(path)

    def _parent_dir(self, path: str) -> str:
        return parent_dir(path)

    def lookup(self, path: str) -> list[IOOperation]:
        with self.lock:
            return lookup(self._state, path)

    def list_directory(self, path: str) -> list[IOOperation]:
        with self.lock:
            return list_directory(self._state, path)

    def materialize_existing_directory(self, path: str) -> DirectoryInode:
        with self.lock:
            self._state, directory = materialize_existing_directory(self._state, path)
            return directory

    def materialize_existing_file(self, path: str, size: int) -> FileInode:
        with self.lock:
            self._state, inode = materialize_existing_file(self._state, path, size)
            return inode

    def reconcile_existing_file(self, path: str, size: int) -> FileInode:
        normalized = normalize_path(path)
        with self.lock:
            if normalized not in self._state.files:
                self._state, inode = materialize_existing_file(self._state, normalized, size)
                return inode

            current = self._state.files[normalized]
            if size < current.size:
                self._state, _ = truncate(self._state, normalized, size=size)
            elif size > current.size:
                self._state, _ = write(self._state, normalized, current.size, size - current.size)
            return self._state.files[normalized]

    def reconcile_missing_path(self, path: str) -> None:
        normalized = normalize_path(path)
        with self.lock:
            if normalized in self._state.files:
                self._state, _ = delete(self._state, normalized)
            elif normalized in self._state.directories and normalized != "/":
                self._state, _ = delete_directory(self._state, normalized, recursive=True)

    def create_directory(self, path: str) -> list[IOOperation]:
        with self.lock:
            self._state, operations = create_directory(self._state, path)
            return operations

    def update_directory(self, path: str, source: str = "dir_attr_update") -> list[IOOperation]:
        with self.lock:
            self._state, operations = update_directory(self._state, path, source=source)
            return operations

    def create_empty_file(self, path: str) -> list[IOOperation]:
        with self.lock:
            self._state, operations = create_empty_file(self._state, path)
            return operations

    def write(self, path: str, offset: int, length: int) -> list[IOOperation]:
        with self.lock:
            self._state, operations = write(self._state, path, offset, length)
            return operations

    def read(self, path: str, offset: int, length: int) -> list[IOOperation]:
        with self.lock:
            return read(self._state, path, offset, length)

    def delete(self, path: str) -> list[IOOperation]:
        with self.lock:
            self._state, operations = delete(self._state, path)
            return operations

    def delete_directory(self, path: str, recursive: bool = True) -> list[IOOperation]:
        with self.lock:
            self._state, operations = delete_directory(self._state, path, recursive=recursive)
            return operations

    def rename(self, source_path: str, dest_path: str) -> list[IOOperation]:
        with self.lock:
            self._state, operations = rename(self._state, source_path, dest_path)
            return operations

    def truncate(self, path: str, size: int = 0) -> list[IOOperation]:
        with self.lock:
            self._state, operations = truncate(self._state, path, size=size)
            return operations

    def get_fragmentation_score(self, path: str) -> int:
        with self.lock:
            return get_fragmentation_score(self._state, path)

    def assert_consistent(self) -> None:
        with self.lock:
            assert_consistent(self._state)


__all__ = [
    "DirectoryInode",
    "FileInode",
    "FileSystemSimulator",
    "IOOperation",
]
