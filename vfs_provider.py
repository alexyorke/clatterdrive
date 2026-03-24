import os
import sys
from wsgidav.fs_dav_provider import FilesystemProvider, FileResource
from hdd_model import VirtualHDD

class LatencyFileResource(FileResource):
    def __init__(self, path, environ, file_path, vhdd):
        super().__init__(path, environ, file_path)
        self.vhdd = vhdd

    def get_content(self):
        # We don't log here to avoid noise
        original_reader = super().get_content()
        if hasattr(original_reader, "read"):
            return LatencyReader(original_reader, self.path, self.vhdd)
        return original_reader

    def begin_write(self, content_type=None):
        original_writer = super().begin_write(content_type)
        return LatencyWriter(original_writer, self.path, self.vhdd)

class LatencyReader:
    def __init__(self, reader, path, vhdd):
        self.reader = reader
        self.path = path
        self.vhdd = vhdd
        self.offset = 0

    def read(self, size=-1):
        if size == -1: size = 4096 
        stats = self.vhdd.access_file(self.path, self.offset, size, is_write=False)
        print(f"READ: {self.path} | Cyl: {stats['cyl']} Head: {stats['head']} | "
              f"Seek: {stats['seek_ms']:.2f}ms Rot: {stats['rot_ms']:.2f}ms "
              f"HeadSw: {stats['head_switch_ms']:.2f}ms Xfer: {stats['transfer_ms']:.2f}ms | "
              f"Total: {stats['total_ms']:.2f}ms", file=sys.stderr)
        
        data = self.reader.read(size)
        self.offset += len(data)
        return data

    def seek(self, offset, whence=0):
        if whence == 0: new_offset = offset
        elif whence == 1: new_offset = self.offset + offset
        else:
            self.reader.seek(offset, whence)
            new_offset = self.reader.tell()
        
        # Seek/Rotation simulation for the seek itself
        self.vhdd.access_file(self.path, new_offset, 0, is_write=False)
        
        if whence != 2: self.reader.seek(offset, whence)
        self.offset = new_offset

    def tell(self):
        return self.offset

    def close(self):
        self.reader.close()

class LatencyWriter:
    def __init__(self, writer, path, vhdd):
        self.writer = writer
        self.path = path
        self.vhdd = vhdd
        self.offset = 0

    def write(self, data):
        size = len(data)
        stats = self.vhdd.access_file(self.path, self.offset, size, is_write=True)
        print(f"WRITE: {self.path} | Cyl: {stats['cyl']} Head: {stats['head']} | "
              f"Seek: {stats['seek_ms']:.2f}ms Rot: {stats['rot_ms']:.2f}ms "
              f"Xfer: {stats['transfer_ms']:.2f}ms | Total: {stats['total_ms']:.2f}ms", file=sys.stderr)
        
        self.writer.write(data)
        self.offset += size

    def close(self):
        self.writer.close()

class HDDProvider(FilesystemProvider):
    def __init__(self, root_folder_path):
        super().__init__(root_folder_path)
        self.vhdd = VirtualHDD(root_folder_path)

    def get_resource_inst(self, path, environ):
        res = super().get_resource_inst(path, environ)
        if res and isinstance(res, FileResource):
            fp = getattr(res, "file_path", getattr(res, "_file_path", None))
            if not fp:
                fp = os.path.join(self.root_folder_path, path.lstrip("/"))
            return LatencyFileResource(res.path, environ, fp, self.vhdd)
        return res
