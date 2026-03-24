import math
import random
import threading

class FileInode:
    def __init__(self, path):
        self.path = path
        self.extents = [] # List of (start_lba, num_blocks)
        self.size = 0

class FileSystemSimulator:
    """
    Simulates a real block-based filesystem.
    - Block Size: 4096 bytes (Advanced Format)
    - Allocation: Bitmap-based, non-contiguous (Extent-based)
    - Real Fragmentation: Occurs when files are deleted and new ones are written into the gaps.
    """
    def __init__(self, total_gb=10, block_size=4096):
        self.block_size = block_size
        self.total_blocks = (total_gb * 1024 * 1024 * 1024) // block_size
        
        # 0 = Free, 1 = Used
        self.bitmap = bytearray(self.total_blocks)
        self.files = {} # path -> FileInode
        self.lock = threading.Lock()
        
        # Metadata overhead: Reserve some blocks for 'System Area'
        for i in range(1024):
            self.bitmap[i] = 1

    def _allocate_blocks(self, count):
        """Finds free blocks and returns a list of extents."""
        allocated = []
        found_count = 0
        
        # Simple linear scan (First-fit) to encourage realistic fragmentation over time
        current_extent_start = -1
        current_extent_len = 0
        
        for i in range(self.total_blocks):
            if self.bitmap[i] == 0:
                if current_extent_start == -1:
                    current_extent_start = i
                current_extent_len += 1
                self.bitmap[i] = 1
                found_count += 1
                
                if found_count == count:
                    allocated.append((current_extent_start, current_extent_len))
                    return allocated
            else:
                if current_extent_start != -1:
                    allocated.append((current_extent_start, current_extent_len))
                    current_extent_start = -1
                    current_extent_len = 0
        
        return allocated # Partial allocation if disk full

    def _free_extents(self, extents):
        for start, length in extents:
            for i in range(start, start + length):
                self.bitmap[i] = 0

    def write(self, path, offset, length):
        """
        Simulates writing to a file. 
        Returns a list of LBAs that were actually accessed (to trigger HDD latency).
        """
        with self.lock:
            if path not in self.files:
                self.files[path] = FileInode(path)
            
            inode = self.files[path]
            blocks_needed = math.ceil((offset + length) / self.block_size)
            current_blocks = sum(ext[1] for ext in inode.extents)
            
            if blocks_needed > current_blocks:
                new_extents = self._allocate_blocks(blocks_needed - current_blocks)
                inode.extents.extend(new_extents)
            
            inode.size = max(inode.size, offset + length)
            
            # Return LBAs touched for this specific write range
            return self._get_lbas_for_range(inode, offset, length)

    def _get_lbas_for_range(self, inode, offset, length):
        """Maps a file byte-range to physical LBAs across multiple extents."""
        start_block = offset // self.block_size
        end_block = (offset + length - 1) // self.block_size
        
        target_lbas = []
        current_file_block = 0
        
        for start_lba, ext_len in inode.extents:
            ext_end_block = current_file_block + ext_len - 1
            
            # Check if this extent overlaps with our requested range
            overlap_start = max(start_block, current_file_block)
            overlap_end = min(end_block, ext_end_block)
            
            if overlap_start <= overlap_end:
                # Calculate physical LBA for this overlap
                lba_offset = overlap_start - current_file_block
                physical_start = start_lba + lba_offset
                lba_len = overlap_end - overlap_start + 1
                target_lbas.append((physical_start, lba_len))
            
            current_file_block += ext_len
            if current_file_block > end_block:
                break
                
        return target_lbas

    def read(self, path, offset, length):
        with self.lock:
            if path not in self.files:
                return []
            return self._get_lbas_for_range(self.files[path], offset, length)

    def delete(self, path):
        with self.lock:
            if path in self.files:
                self._free_extents(self.files[path].extents)
                del self.files[path]

    def get_fragmentation_score(self, path):
        """Returns the number of non-contiguous extents for a file."""
        with self.lock:
            if path in self.files:
                return len(self.files[path].extents)
            return 0
