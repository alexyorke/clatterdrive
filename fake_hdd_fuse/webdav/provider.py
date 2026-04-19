import os
import sys
from collections.abc import Iterable
from typing import Any

from wsgidav import util
from wsgidav.fs_dav_provider import FileResource, FilesystemProvider, FolderResource

from ..hdd import VirtualHDD
from ..hdd.core import OperationStats
from ..scheduler import OSScheduler
from ..storage_events import StorageEventSink


def _is_case_only_alias(path: str, dest_path: str) -> bool:
    return os.name == "nt" and path != dest_path and path.casefold() == dest_path.casefold()


def _resource_file_path(resource: Any) -> str:
    if hasattr(resource, "file_path"):
        return resource.file_path
    return resource._file_path


def _stats_flags(stats: OperationStats, writeback: bool = False) -> str:
    flags = []
    if writeback and stats.cache_hit:
        flags.append("WRITE-BACK")
    elif stats.cache_hit:
        flags.append("CACHE HIT")
    elif stats.partial_hit:
        flags.append("PARTIAL CACHE")
    if stats.ready_poll_count:
        flags.append(f"NOT READY x{stats.ready_poll_count}")
    if stats.retry_count:
        flags.append(f"RETRY x{stats.retry_count}")
    if stats.maintenance_wait_ms > 0.0:
        flags.append(f"BG WAIT {stats.maintenance_wait_ms:.1f}ms")
    return "".join(f"[{flag}] " for flag in flags).strip()


def _log_op(label: str, path: str, stats: OperationStats, writeback: bool = False) -> None:
    hit_str = _stats_flags(stats, writeback=writeback)
    ext_str = f" | Extents: {stats.extents}" if stats.extents > 1 else ""
    prefix = f"{label}: {path}"
    if hit_str:
        prefix = f"{prefix} {hit_str}"
    print(
        f"{prefix} | Cyl: {stats.cyl} Head: {stats.head}{ext_str} | "
        f"Total Latency: {stats.total_ms:.2f}ms",
        file=sys.stderr,
    )


class LatencyFileResource(FileResource):
    def __init__(self, path: str, environ: dict[str, Any], file_path: str, vhdd: VirtualHDD) -> None:
        super().__init__(path, environ, file_path)
        self.vhdd = vhdd

    def get_content(self) -> Any:
        original_reader = super().get_content()
        if hasattr(original_reader, "read"):
            return LatencyReader(original_reader, self.path, self.vhdd)
        return original_reader

    def begin_write(self, *, content_type: str | None = None) -> Any:
        original_writer = super().begin_write(content_type=content_type)
        return LatencyWriter(original_writer, self.path, self.vhdd)

    def delete(self) -> None:
        self.vhdd.ensure_tree_known(self.path)
        super().delete()
        _log_op("DELETE", self.path, self.vhdd.delete_path(self.path))

    def handle_move(self, dest_path: str) -> bool:
        if not _is_case_only_alias(self.path, dest_path):
            return False
        self.vhdd.ensure_tree_known(self.path)
        dest_file_path = os.path.join(self.provider.root_folder_path, dest_path.lstrip("/"))
        os.rename(_resource_file_path(self), dest_file_path)
        _log_op("MOVE", f"{self.path} -> {dest_path}", self.vhdd.rename_path(self.path, dest_path))
        return True

    def handle_copy(self, dest_path: str, *, depth_infinity: bool) -> bool:
        if not _is_case_only_alias(self.path, dest_path):
            return False
        _log_op("COPY", f"{self.path} -> {dest_path}", self.vhdd.copy_file(self.path, dest_path))
        return True

    def move_recursive(self, dest_path: str) -> None:
        self.vhdd.ensure_tree_known(self.path)
        if _is_case_only_alias(self.path, dest_path):
            dest_file_path = os.path.join(self.provider.root_folder_path, dest_path.lstrip("/"))
            os.rename(_resource_file_path(self), dest_file_path)
            _log_op("MOVE", f"{self.path} -> {dest_path}", self.vhdd.rename_path(self.path, dest_path))
            return
        super().move_recursive(dest_path)
        _log_op("MOVE", f"{self.path} -> {dest_path}", self.vhdd.rename_path(self.path, dest_path))

    def copy_move_single(self, dest_path: str, *, is_move: bool) -> Any:
        if is_move:
            return super().copy_move_single(dest_path, is_move=is_move)
        if _is_case_only_alias(self.path, dest_path):
            _log_op("COPY", f"{self.path} -> {dest_path}", self.vhdd.copy_file(self.path, dest_path))
            return True
        self.vhdd.ensure_tree_known(self.path)
        super().copy_move_single(dest_path, is_move=is_move)
        _log_op("COPY", f"{self.path} -> {dest_path}", self.vhdd.copy_file(self.path, dest_path))
        return True


class LatencyFolderResource(FolderResource):
    def __init__(self, path: str, environ: dict[str, Any], file_path: str, vhdd: VirtualHDD) -> None:
        super().__init__(path, environ, file_path)
        self.vhdd = vhdd

    def get_member_names(self) -> list[str]:
        stats = self.vhdd.list_directory(self.path)
        _log_op("READDIR", self.path, stats)
        return super().get_member_names()

    def get_member(self, name: str) -> Any:
        child_path = util.join_uri(self.path, name)
        return self.provider.get_resource_inst(child_path, self.environ)

    def create_collection(self, name: str) -> None:
        super().create_collection(name)
        child_path = util.join_uri(self.path, name)
        _log_op("MKCOL", child_path, self.vhdd.create_directory(child_path))

    def create_empty_resource(self, name: str) -> Any:
        resource = super().create_empty_resource(name)
        child_path = util.join_uri(self.path, name)
        _log_op("CREATE", child_path, self.vhdd.create_empty_file(child_path))
        return resource

    def delete(self) -> None:
        self.vhdd.ensure_tree_known(self.path)
        super().delete()
        _log_op("DELETE", self.path, self.vhdd.delete_directory(self.path))

    def handle_move(self, dest_path: str) -> bool:
        if not _is_case_only_alias(self.path, dest_path):
            return False
        self.vhdd.ensure_tree_known(self.path)
        dest_file_path = os.path.join(self.provider.root_folder_path, dest_path.lstrip("/"))
        os.rename(_resource_file_path(self), dest_file_path)
        _log_op("MOVE", f"{self.path} -> {dest_path}", self.vhdd.rename_path(self.path, dest_path))
        return True

    def handle_copy(self, dest_path: str, *, depth_infinity: bool) -> bool:
        if not _is_case_only_alias(self.path, dest_path):
            return False
        _log_op("COPY", f"{self.path} -> {dest_path}", OperationStats(op_type="COPY"))
        return True

    def move_recursive(self, dest_path: str) -> None:
        self.vhdd.ensure_tree_known(self.path)
        if _is_case_only_alias(self.path, dest_path):
            dest_file_path = os.path.join(self.provider.root_folder_path, dest_path.lstrip("/"))
            os.rename(_resource_file_path(self), dest_file_path)
            _log_op("MOVE", f"{self.path} -> {dest_path}", self.vhdd.rename_path(self.path, dest_path))
            return
        super().move_recursive(dest_path)
        _log_op("MOVE", f"{self.path} -> {dest_path}", self.vhdd.rename_path(self.path, dest_path))

    def copy_move_single(self, dest_path: str, *, is_move: bool) -> Any:
        if is_move:
            return super().copy_move_single(dest_path, is_move=is_move)
        if _is_case_only_alias(self.path, dest_path):
            _log_op("COPY", f"{self.path} -> {dest_path}", OperationStats(op_type="COPY"))
            return True
        self.vhdd.ensure_tree_known(self.path)
        super().copy_move_single(dest_path, is_move=is_move)
        if self.vhdd.fs._normalize_path(dest_path) in self.vhdd.fs.directories:
            stats = self.vhdd.refresh_directory(dest_path)
        else:
            stats = self.vhdd.create_directory(dest_path)
        _log_op("COPY", f"{self.path} -> {dest_path}", stats)
        return True


class LatencyReader:
    def __init__(self, reader: Any, path: str, vhdd: VirtualHDD) -> None:
        self.reader = reader
        self.path = path
        self.vhdd = vhdd
        try:
            self.offset = int(reader.tell())
        except Exception:
            self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""

        data = self.reader.read(size)
        actual_size = len(data)
        if actual_size == 0:
            return data

        stats = self.vhdd.access_file(self.path, self.offset, actual_size, is_write=False)
        _log_op("READ", self.path, stats)

        self.offset += actual_size
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
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

    def tell(self) -> int:
        return self.offset

    def close(self) -> None:
        if hasattr(self.reader, "close"):
            self.reader.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)


class LatencyWriter:
    def __init__(self, writer: Any, path: str, vhdd: VirtualHDD) -> None:
        self.writer = writer
        self.path = path
        self.vhdd = vhdd
        try:
            self.offset = int(writer.tell())
        except Exception:
            self.offset = 0
        self.vhdd.prepare_overwrite(self.path)

    def write(self, data: bytes) -> Any:
        size = len(data)
        if size == 0:
            return 0

        written = self.writer.write(data)
        actual_size = size if written is None else int(written)
        if actual_size <= 0:
            return written

        stats = self.vhdd.access_file(self.path, self.offset, actual_size, is_write=True)
        _log_op("WRITE", self.path, stats, writeback=True)
        self.offset += actual_size
        return written

    def writelines(self, lines: Iterable[bytes]) -> None:
        for chunk in lines:
            self.write(chunk)

    def close(self) -> None:
        if hasattr(self.writer, "close"):
            self.writer.close()

    def seek(self, offset: int, whence: int = 0) -> int:
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

    def tell(self) -> int:
        return self.offset

    def __getattr__(self, name: str) -> Any:
        return getattr(self.writer, name)


class HDDProvider(FilesystemProvider):
    def __init__(
        self,
        root_folder_path: str,
        *,
        event_sink: StorageEventSink | None = None,
        drive_profile: str | None = None,
        acoustic_profile: str | None = None,
        cold_start: bool = True,
        async_power_on: bool = True,
    ) -> None:
        super().__init__(root_folder_path)
        self.vhdd = VirtualHDD(
            root_folder_path,
            cold_start=cold_start,
            async_power_on=async_power_on,
            drive_profile=drive_profile,
            acoustic_profile=acoustic_profile,
            event_sink=event_sink,
        )
        self.scheduler = OSScheduler(self.vhdd.model)
        self.vhdd.set_scheduler(self.scheduler)

    def get_resource_inst(self, path: str, environ: dict[str, Any]) -> Any:
        self.vhdd.lookup_path(path)
        resource = super().get_resource_inst(path, environ)
        if resource is None:
            return None

        file_path = getattr(resource, "file_path", getattr(resource, "_file_path", None))
        if not file_path:
            file_path = os.path.join(self.root_folder_path, path.lstrip("/"))

        if isinstance(resource, FolderResource):
            return LatencyFolderResource(resource.path, environ, file_path, self.vhdd)
        if isinstance(resource, FileResource):
            return LatencyFileResource(resource.path, environ, file_path, self.vhdd)
        return resource
