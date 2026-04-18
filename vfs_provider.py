import os
import sys
from wsgidav.fs_dav_provider import FilesystemProvider, FileResource
from hdd_model import VirtualHDD
from os_scheduler import OSScheduler

class LatencyFileResource(FileResource):
    def __init__(self, path, environ, file_path, vhdd):
        super().__init__(path, environ, file_path)
        self.vhdd = vhdd

    def get_content(self):
        original_reader = super().get_content()
        if hasattr(original_reader, "read"):
            return LatencyReader(original_reader, self.path, self.vhdd)
        return original_reader

    def begin_write(self, *, content_type=None):
        original_writer = super().begin_write(content_type=content_type)
        return LatencyWriter(original_writer, self.path, self.vhdd)

class LatencyReader:
    def __init__(self, reader, path, vhdd):
        self.reader = reader
        self.path = path
        self.vhdd = vhdd
        try:
            self.offset = int(reader.tell())
        except Exception:
            self.offset = 0

    def read(self, size=-1):
        if size == 0:
            return b""

        data = self.reader.read(size)
        actual_size = len(data)
        if actual_size == 0:
            return data

        # Price the actual bytes returned, not the requested size.
        stats = self.vhdd.access_file(self.path, self.offset, actual_size, is_write=False)

        if stats.get("cache_hit"):
            hit_str = "[CACHE HIT]"
        elif stats.get("partial_hit"):
            hit_str = "[PARTIAL CACHE]"
        else:
            hit_str = ""
        if stats.get("ready_poll_count"):
            hit_str = f"{hit_str} [NOT READY x{stats['ready_poll_count']}]".strip()
        ext_str = f"| Extents: {stats['extents']}" if stats['extents'] > 1 else ""
        print(f"READ: {self.path} {hit_str} | Cyl: {stats.get('cyl','-')} Head: {stats.get('head','-')} {ext_str} | "
              f"Total Latency: {stats['total_ms']:.2f}ms", file=sys.stderr)

        self.offset += actual_size
        return data

    def seek(self, offset, whence=0):
        if whence == 0:
            new_offset = offset
            self.reader.seek(offset, whence)
        elif whence == 1:
            new_offset = self.offset + offset
            self.reader.seek(offset, whence)
        else:
            self.reader.seek(offset, whence)
            new_offset = self.reader.tell()
        self.offset = new_offset
        return self.offset

    def tell(self):
        return self.offset

    def close(self):
        if hasattr(self.reader, "close"):
            self.reader.close()

    def __getattr__(self, name):
        return getattr(self.reader, name)

class LatencyWriter:
    def __init__(self, writer, path, vhdd):
        self.writer = writer
        self.path = path
        self.vhdd = vhdd
        try:
            self.offset = int(writer.tell())
        except Exception:
            self.offset = 0
        self.vhdd.prepare_overwrite(self.path)

    def write(self, data):
        size = len(data)
        if size == 0:
            return 0

        written = self.writer.write(data)
        actual_size = size if written is None else int(written)
        if actual_size <= 0:
            return written

        stats = self.vhdd.access_file(self.path, self.offset, actual_size, is_write=True)
        hit_str = "[WRITE-BACK]" if stats.get("cache_hit") else ""
        if stats.get("ready_poll_count"):
            hit_str = f"{hit_str} [NOT READY x{stats['ready_poll_count']}]".strip()
        print(f"WRITE: {self.path} {hit_str} | Cyl: {stats.get('cyl','-')} Head: {stats.get('head','-')} | "
              f"Total Latency: {stats['total_ms']:.2f}ms", file=sys.stderr)
        self.offset += actual_size
        return written

    def close(self):
        if hasattr(self.writer, "close"):
            self.writer.close()

    def seek(self, offset, whence=0):
        if whence == 0:
            new_offset = offset
            self.writer.seek(offset, whence)
        elif whence == 1:
            new_offset = self.offset + offset
            self.writer.seek(offset, whence)
        else:
            self.writer.seek(offset, whence)
            new_offset = self.writer.tell()
        self.offset = new_offset
        return self.offset

    def tell(self):
        return self.offset

    def __getattr__(self, name):
        return getattr(self.writer, name)

class HDDProvider(FilesystemProvider):
    def __init__(self, root_folder_path):
        super().__init__(root_folder_path)
        self.vhdd = VirtualHDD(root_folder_path, cold_start=True, async_power_on=True)
        self.scheduler = OSScheduler(self.vhdd.model)
        # Link scheduler to VirtualHDD
        self.vhdd.set_scheduler(self.scheduler)

    def get_resource_inst(self, path, environ):
        if path not in ("", "/"):
            self.vhdd.lookup_path(path)
        res = super().get_resource_inst(path, environ)
        if res and isinstance(res, FileResource):
            fp = getattr(res, "file_path", getattr(res, "_file_path", None))
            if not fp:
                fp = os.path.join(self.root_folder_path, path.lstrip("/"))
            return LatencyFileResource(res.path, environ, fp, self.vhdd)
        return res

    def delete_resource(self, path, environ):
        self.vhdd.delete_path(path)
        return super().delete_resource(path, environ)
