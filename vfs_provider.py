import os
import sys
import random
from wsgidav.fs_dav_provider import FilesystemProvider, FileResource
from hdd_model import VirtualHDD
from os_scheduler import OSScheduler

class LatencyFileResource(FileResource):
    def __init__(self, path, environ, file_path, vhdd, scheduler):
        super().__init__(path, environ, file_path)
        self.vhdd = vhdd
        self.scheduler = scheduler

    def get_content(self):
        original_reader = super().get_content()
        if hasattr(original_reader, "read"):
            return LatencyReader(original_reader, self.path, self.vhdd, self.scheduler)
        return original_reader

    def begin_write(self, content_type=None):
        original_writer = super().begin_write(content_type)
        return LatencyWriter(original_writer, self.path, self.vhdd, self.scheduler)

class LatencyReader:
    def __init__(self, reader, path, vhdd, scheduler):
        self.reader = reader
        self.path = path
        self.vhdd = vhdd
        self.scheduler = scheduler
        self.offset = 0

    def read(self, size=-1):
        if size == -1: size = 4096 
        
        # 1. Translate path/offset to LBA
        start_lba = self.vhdd.get_file_lba(self.path)
        target_lba = start_lba + (self.offset // 512)
        
        # 2. Submit to OS Scheduler (VFS -> Block Layer)
        req_id = self.scheduler.submit_bio(target_lba, size, is_write=False)
        stats = self.scheduler.wait_for_completion(req_id)
        
        # Log the detailed stats including Cache Hit (Read-Ahead)
        hit_str = "[CACHE HIT]" if stats.get("cache_hit") else ""
        print(f"READ: {self.path} {hit_str} | Cyl: {stats.get('cyl','-')} Head: {stats.get('head','-')} | "
              f"Total Latency: {stats['total_ms']:.2f}ms", file=sys.stderr)
        
        data = self.reader.read(size)
        self.offset += len(data)
        return data

    def seek(self, offset, whence=0):
        if whence == 0: new_offset = offset
        elif whence == 1: new_offset = self.offset + offset
        else:
            self.reader.seek(offset, whence)
            new_offset = self.reader.tell()
        
        # No simulation for the seek itself, as the next READ will trigger the mechanical delay
        if whence != 2: self.reader.seek(offset, whence)
        self.offset = new_offset

    def tell(self):
        return self.offset

    def close(self):
        self.reader.close()

class LatencyWriter:
    def __init__(self, writer, path, vhdd, scheduler):
        self.writer = writer
        self.path = path
        self.vhdd = vhdd
        self.scheduler = scheduler
        self.offset = 0

    def write(self, data):
        size = len(data)
        
        start_lba = self.vhdd.get_file_lba(self.path)
        target_lba = start_lba + (self.offset // 512)
        
        # Submit to OS Scheduler
        req_id = self.scheduler.submit_bio(target_lba, size, is_write=True)
        stats = self.scheduler.wait_for_completion(req_id)
        
        hit_str = "[WRITE-BACK]" if stats.get("cache_hit") else ""
        print(f"WRITE: {self.path} {hit_str} | Cyl: {stats.get('cyl','-')} Head: {stats.get('head','-')} | "
              f"Total Latency: {stats['total_ms']:.2f}ms", file=sys.stderr)
        
        self.writer.write(data)
        self.offset += size

    def close(self):
        self.writer.close()

class HDDProvider(FilesystemProvider):
    def __init__(self, root_folder_path):
        super().__init__(root_folder_path)
        self.vhdd = VirtualHDD(root_folder_path)
        # Initialize the OS Scheduler stack
        self.scheduler = OSScheduler(self.vhdd.model)

    def get_resource_inst(self, path, environ):
        res = super().get_resource_inst(path, environ)
        if res and isinstance(res, FileResource):
            fp = getattr(res, "file_path", getattr(res, "_file_path", None))
            if not fp:
                fp = os.path.join(self.root_folder_path, path.lstrip("/"))
            return LatencyFileResource(res.path, environ, fp, self.vhdd, self.scheduler)
        return res
