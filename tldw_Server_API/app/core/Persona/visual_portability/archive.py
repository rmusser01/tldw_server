"""Safe ZIP member validation for persona visual pack archives."""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

from .constants import (
    ALLOWED_TOP_LEVEL_DIRS,
    ALLOWED_TOP_LEVEL_FILES,
    REQUIRED_MEMBERS,
)


DEFAULT_MAX_MEMBER_SIZE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_SIZE_BYTES = 500 * 1024 * 1024


def validate_archive_members(
    archive_path: Path,
    *,
    max_member_size_bytes: int = DEFAULT_MAX_MEMBER_SIZE_BYTES,
    max_archive_size_bytes: int = DEFAULT_MAX_ARCHIVE_SIZE_BYTES,
) -> list[str]:
    """Validate archive member names and size limits without extracting files."""
    normalized_members: list[str] = []
    seen_members: set[str] = set()
    total_uncompressed_size = 0

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in archive.infolist():
                raw_member_name = getattr(info, "orig_filename", info.filename)
                normalized_name = normalize_member_name(raw_member_name)
                if normalized_name in seen_members:
                    raise ValueError(f"duplicate_archive_member: {normalized_name}")
                if _is_zip_symlink(info):
                    raise ValueError(f"unsafe_archive_member: symlink:{normalized_name}")
                if info.flag_bits & 0x1:
                    raise ValueError(
                        f"unsupported_archive_member: encrypted:{normalized_name}"
                    )

                _validate_top_level_member(normalized_name)

                if info.file_size > max_member_size_bytes:
                    raise ValueError(f"archive_member_too_large: {normalized_name}")

                total_uncompressed_size += info.file_size
                if total_uncompressed_size > max_archive_size_bytes:
                    raise ValueError("archive_too_large")

                seen_members.add(normalized_name)
                normalized_members.append(normalized_name)
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid_archive: bad_zip_file") from exc

    missing_members = REQUIRED_MEMBERS - seen_members
    if missing_members:
        raise ValueError(
            f"missing_required_archive_member: {sorted(missing_members)[0]}"
        )
    return normalized_members


def normalize_member_name(member_name: str) -> str:
    """Return a normalized POSIX archive member name or raise ValueError."""
    if not member_name:
        raise ValueError("unsafe_archive_member: empty_name")
    if "\x00" in member_name:
        raise ValueError("unsafe_archive_member: null_byte")
    if "\\" in member_name:
        raise ValueError("unsafe_archive_member: backslash")
    if member_name.startswith("/"):
        raise ValueError("unsafe_archive_member: absolute_path")

    raw_parts = member_name.split("/")
    if any(part == "" for part in raw_parts):
        raise ValueError("unsafe_archive_member: empty_part")
    if any(_has_windows_drive_letter(part) for part in raw_parts):
        raise ValueError("unsafe_archive_member: drive_letter")

    normalized_path = PurePosixPath(member_name)
    if normalized_path.is_absolute():
        raise ValueError("unsafe_archive_member: absolute_path")

    normalized_parts = normalized_path.parts
    if not normalized_parts:
        raise ValueError("unsafe_archive_member: empty_name")
    if any(part == ".." for part in normalized_parts):
        raise ValueError("unsafe_archive_member: parent_reference")
    if any(_has_windows_drive_letter(part) for part in normalized_parts):
        raise ValueError("unsafe_archive_member: drive_letter")

    normalized_name = normalized_path.as_posix()
    if normalized_name in {"", "."}:
        raise ValueError("unsafe_archive_member: empty_name")
    return normalized_name


def _validate_top_level_member(normalized_name: str) -> None:
    parts = PurePosixPath(normalized_name).parts
    if len(parts) == 1:
        if parts[0] not in ALLOWED_TOP_LEVEL_FILES:
            raise ValueError(f"unexpected_archive_member: {normalized_name}")
        return

    if parts[0] not in ALLOWED_TOP_LEVEL_DIRS:
        raise ValueError(f"unexpected_archive_member: {normalized_name}")


def _has_windows_drive_letter(part: str) -> bool:
    return len(part) >= 2 and part[0].isalpha() and part[1] == ":"


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_file_type = (info.external_attr >> 16) & 0o170000
    return unix_file_type == 0o120000
