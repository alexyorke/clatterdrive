import math
import posixpath
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class IOOperation:
    lba: int
    block_count: int
    kind: str
    source: str = ""


class FileInode:
    def __init__(self, path, inode_block, parent_dir):
        self.path = path
        self.inode_block = inode_block
        self.parent_dir = parent_dir
        # (logical_file_block, physical_lba, num_blocks)
        self.extents = []
        self.size = 0


class FileSystemSimulator:
    """
    Simulates a block-based filesystem with lightweight metadata traffic.

    The model is intentionally compact rather than fully faithful, but it now
    preserves logical file offsets so sparse files and restart materialization
    do not behave like fully allocated contiguous files.
    """

    SUPERBLOCK_BLOCKS = 1024
    JOURNAL_BLOCKS = 4096
    INODE_TABLE_BLOCKS = 4096
    DIRECTORY_BLOCKS = 2048
    BITMAP_BLOCKS = 256

    def __init__(self, total_gb=10, block_size=4096):
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

        self.files = {}  # path -> FileInode
        self.directory_blocks = {"/": self.directory_start}
        self.next_inode_block = self.inode_table_start
        self.next_directory_block = self.directory_start + 1
        self.journal_cursor = 0
        self.lock = threading.Lock()

    def _normalize_path(self, path):
        normalized = posixpath.normpath((path or "").replace("\\", "/"))
        if normalized in ("", "."):
            return "/"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized

    def _parent_dir(self, path):
        normalized = self._normalize_path(path)
        parent = posixpath.dirname(normalized.rstrip("/")) or "/"
        return parent if parent.startswith("/") else f"/{parent}"

    def _allocate_inode_block(self):
        if self.next_inode_block >= self.inode_table_start + self.INODE_TABLE_BLOCKS:
            raise RuntimeError("inode table exhausted")
        block = self.next_inode_block
        self.next_inode_block += 1
        return block

    def _ensure_directory_block(self, path):
        normalized = self._normalize_path(path)
        if normalized not in self.directory_blocks:
            if self.next_directory_block >= self.directory_start + self.DIRECTORY_BLOCKS:
                raise RuntimeError("directory metadata area exhausted")
            self.directory_blocks[normalized] = self.next_directory_block
            self.next_directory_block += 1
        return self.directory_blocks[normalized]

    def _journal_op(self, block_count, source):
        start = self.journal_start + self.journal_cursor
        self.journal_cursor = (self.journal_cursor + block_count) % self.JOURNAL_BLOCKS
        return IOOperation(start, block_count, "journal", source)

    def _bitmap_ops_for_extents(self, extents):
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

    def _allocate_blocks(self, count):
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

    def _free_extents(self, extents):
        for _, start, length in extents:
            for block in range(start, start + length):
                self.bitmap[block] = 0

    def _coalesce_extents(self, inode):
        inode.extents.sort(key=lambda item: item[0])
        merged = []
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

    def _range_to_ops(self, inode, offset, length):
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

    def _missing_logical_ranges(self, inode, start_block, end_block):
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

    def _allocate_missing_ranges(self, inode, start_block, end_block):
        new_extents = []
        for missing_start, missing_end in self._missing_logical_ranges(inode, start_block, end_block):
            blocks_needed = missing_end - missing_start + 1
            for physical_start, length in self._allocate_blocks(blocks_needed):
                new_extents.append((missing_start, physical_start, length))
                missing_start += length
        inode.extents.extend(new_extents)
        self._coalesce_extents(inode)
        return new_extents

    def lookup(self, path):
        path = self._normalize_path(path)
        parent_dir = self._parent_dir(path)
        operations = [
            IOOperation(self._ensure_directory_block(parent_dir), 1, "metadata", "dentry_lookup")
        ]
        if path in self.files:
            operations.append(
                IOOperation(self.files[path].inode_block, 1, "metadata", "inode_lookup")
            )
        return operations

    def list_directory(self, path):
        path = self._normalize_path(path)
        return [IOOperation(self._ensure_directory_block(path), 1, "metadata", "readdir")]

    def materialize_existing_file(self, path, size):
        with self.lock:
            path = self._normalize_path(path)
            if path in self.files:
                return self.files[path]

            parent_dir = self._parent_dir(path)
            self._ensure_directory_block(parent_dir)
            inode = FileInode(path, self._allocate_inode_block(), parent_dir)
            inode.size = max(0, int(size))
            self.files[path] = inode

            if inode.size > 0:
                total_blocks = math.ceil(inode.size / self.block_size)
                missing = self._allocate_missing_ranges(inode, 0, total_blocks - 1)
                if not missing and total_blocks > 0:
                    raise OSError("disk full")
            return inode

    def write(self, path, offset, length):
        """
        Simulates a buffered write and returns metadata + data blocks touched.
        """
        with self.lock:
            path = self._normalize_path(path)
            parent_dir = self._parent_dir(path)
            parent_dir_block = self._ensure_directory_block(parent_dir)
            created = False

            if path not in self.files:
                self.files[path] = FileInode(path, self._allocate_inode_block(), parent_dir)
                created = True

            inode = self.files[path]
            start_block = offset // self.block_size
            end_block = (offset + length - 1) // self.block_size if length > 0 else start_block - 1
            new_extents = self._allocate_missing_ranges(inode, start_block, end_block)
            inode.size = max(inode.size, offset + length)

            metadata_ops = [
                self._journal_op(2 if (created or new_extents) else 1, "write_intent"),
                IOOperation(inode.inode_block, 1, "metadata", "inode_update"),
            ]
            if created:
                metadata_ops.append(IOOperation(parent_dir_block, 1, "metadata", "dir_insert"))
            if new_extents:
                metadata_ops.extend(self._bitmap_ops_for_extents(new_extents))

            data_ops = self._range_to_ops(inode, offset, length)
            return metadata_ops + data_ops

    def read(self, path, offset, length):
        with self.lock:
            path = self._normalize_path(path)
            if path not in self.files:
                return []
            inode = self.files[path]
            if length <= 0 or offset >= inode.size:
                return []
            clamped_length = min(length, inode.size - offset)
            return self._range_to_ops(inode, offset, clamped_length)

    def delete(self, path):
        with self.lock:
            path = self._normalize_path(path)
            if path not in self.files:
                return []

            inode = self.files[path]
            parent_dir_block = self._ensure_directory_block(inode.parent_dir)
            operations = [
                self._journal_op(2, "delete_intent"),
                IOOperation(parent_dir_block, 1, "metadata", "dir_remove"),
                IOOperation(inode.inode_block, 1, "metadata", "inode_delete"),
            ]
            operations.extend(self._bitmap_ops_for_extents(inode.extents))

            self._free_extents(inode.extents)
            del self.files[path]
            return operations

    def truncate(self, path, size=0):
        with self.lock:
            path = self._normalize_path(path)
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

    def get_fragmentation_score(self, path):
        with self.lock:
            path = self._normalize_path(path)
            if path in self.files:
                return len(self.files[path].extents)
            return 0
