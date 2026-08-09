from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileSystemProfile:
    name: str
    description: str
    superblock_blocks: int
    journal_blocks: int
    inode_table_blocks: int
    directory_blocks: int
    bitmap_blocks: int
    directory_entry_bytes: int


FILESYSTEM_PROFILES: dict[str, FileSystemProfile] = {
    "generic_journaled": FileSystemProfile(
        name="generic_journaled",
        description="Neutral journaled filesystem with balanced metadata costs.",
        superblock_blocks=1024,
        journal_blocks=4096,
        inode_table_blocks=4096,
        directory_blocks=2048,
        bitmap_blocks=256,
        directory_entry_bytes=192,
    ),
    "ntfs_like": FileSystemProfile(
        name="ntfs_like",
        description="NTFS-inspired metadata density and larger directory records.",
        superblock_blocks=1024,
        journal_blocks=6144,
        inode_table_blocks=6144,
        directory_blocks=3072,
        bitmap_blocks=256,
        directory_entry_bytes=256,
    ),
    "ext4_like": FileSystemProfile(
        name="ext4_like",
        description="ext4-inspired larger journal with compact directory entries.",
        superblock_blocks=512,
        journal_blocks=8192,
        inode_table_blocks=8192,
        directory_blocks=2048,
        bitmap_blocks=512,
        directory_entry_bytes=160,
    ),
    "apfs_like": FileSystemProfile(
        name="apfs_like",
        description="APFS-inspired metadata-heavy copy-on-write approximation.",
        superblock_blocks=2048,
        journal_blocks=4096,
        inode_table_blocks=8192,
        directory_blocks=4096,
        bitmap_blocks=512,
        directory_entry_bytes=320,
    ),
}


def resolve_filesystem_profile(profile: str | FileSystemProfile | None) -> FileSystemProfile:
    if isinstance(profile, FileSystemProfile):
        return profile
    name = profile or "generic_journaled"
    try:
        return FILESYSTEM_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(FILESYSTEM_PROFILES))
        raise ValueError(f"unknown filesystem profile {name!r}; choose one of: {choices}") from exc


__all__ = ["FILESYSTEM_PROFILES", "FileSystemProfile", "resolve_filesystem_profile"]
