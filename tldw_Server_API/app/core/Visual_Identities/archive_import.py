"""Secure ZIP import parsing for visual identity expression pack drafts."""

from __future__ import annotations

import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.Visual_Identities.constraints import (
    MAX_EXPRESSION_ASSET_BYTES,
)
from tldw_Server_API.app.core.Visual_Identities.expression_slots import (
    display_label_for_expression_key,
    normalize_expression_filename,
)
from tldw_Server_API.app.core.Visual_Identities.storage import (
    validate_and_store_visual_identity_asset,
)

MAX_EXPRESSION_ZIP_BYTES = 100 * 1024 * 1024
MAX_EXPRESSION_ZIP_ENTRIES = 128
MAX_EXPRESSION_ZIP_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_EXPRESSION_ZIP_DECOMPRESSION_RATIO = 100

_SUPPORTED_ARCHIVE_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"}
)
_NESTED_ARCHIVE_EXTENSIONS = frozenset(
    {".zip", ".zipx", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}
)
_ZIP_CHUNK_SIZE = 1024 * 1024
_FATAL_ARCHIVE_ERROR_CODES = frozenset(
    {
        "archive_size_exceeded",
        "decompression_ratio_exceeded",
        "duplicate_archive_path",
        "encrypted_entry",
        "entry_compressed_size_exceeded",
        "entry_count_exceeded",
        "entry_uncompressed_size_exceeded",
        "nested_archive",
        "symlink_entry",
        "total_uncompressed_size_exceeded",
        "unsafe_archive_path",
        "zip_archive_not_found",
    }
)


@dataclass(frozen=True)
class _ImportCandidate:
    info: zipfile.ZipInfo
    normalized_path: str
    expression_key: str


def import_visual_identity_expression_zip(
    repo: VisualIdentityRepository,
    *,
    owner_user_id: int,
    draft_id: int,
    archive_path: str | Path,
    storage_root: str | Path | None = None,
) -> dict[str, Any]:
    """Import validated image entries from a ZIP archive into a draft."""
    draft = repo.get_draft(draft_id, owner_user_id=owner_user_id)
    if draft is None:
        raise ValueError("visual_identity_draft_not_found")

    source = Path(archive_path)
    summary = _empty_summary(source_filename=str(source.name))
    candidates: list[_ImportCandidate] = []
    selected: list[_ImportCandidate] = []

    try:
        repo.mark_draft_assets_deleted(draft_id=draft_id, owner_user_id=owner_user_id)
        _validate_archive_size(source, summary)
        with zipfile.ZipFile(source) as archive:
            candidates = _collect_candidates(archive, summary)
            selected = _select_first_expression_candidates(candidates, summary)
            slot_map = _store_candidates(
                archive,
                selected,
                repo=repo,
                owner_user_id=owner_user_id,
                draft_id=draft_id,
                storage_root=storage_root,
                summary=summary,
            )
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        if not summary["errors"]:
            _record_error(summary, code=str(exc) or "invalid_zip_archive", source_filename=str(source))
        return _finish_import(
            repo,
            owner_user_id=owner_user_id,
            draft_id=draft_id,
            slot_map={},
            summary=summary,
            status="failed",
        )

    status = "ready_for_review" if selected and summary["accepted"] else "failed"
    if status == "failed" and not summary["errors"]:
        _record_error(summary, code="no_valid_expression_assets")
    return _finish_import(
        repo,
        owner_user_id=owner_user_id,
        draft_id=draft_id,
        slot_map=slot_map,
        summary=summary,
        status=status,
    )


def _validate_archive_size(source: Path, summary: dict[str, Any]) -> None:
    if not source.is_file():
        _record_error(summary, code="zip_archive_not_found", source_filename=str(source))
        raise ValueError("zip_archive_not_found")
    archive_size = source.stat().st_size
    summary["archive_bytes"] = archive_size
    if archive_size > MAX_EXPRESSION_ZIP_BYTES:
        _record_error(summary, code="archive_size_exceeded", source_filename=source.name)
        raise ValueError("archive_size_exceeded")


def _collect_candidates(
    archive: zipfile.ZipFile,
    summary: dict[str, Any],
) -> list[_ImportCandidate]:
    infos = archive.infolist()
    summary["entry_count"] = len(infos)
    if len(infos) > MAX_EXPRESSION_ZIP_ENTRIES:
        _record_error(summary, code="entry_count_exceeded")

    candidates: list[_ImportCandidate] = []
    seen_paths: set[str] = set()
    total_uncompressed = 0

    for info in infos:
        normalized_path = _normalized_archive_path(info.filename)
        if normalized_path is None:
            _record_error(summary, code="unsafe_archive_path", source_filename=info.filename)
            continue
        duplicate_key = normalized_path.lower()
        if duplicate_key in seen_paths:
            _record_error(summary, code="duplicate_archive_path", source_filename=info.filename)
            continue
        seen_paths.add(duplicate_key)

        _validate_entry_metadata(info, normalized_path, summary)
        if info.is_dir():
            if not _entry_has_errors(summary, info.filename):
                summary["directories"].append(normalized_path)
            continue

        total_uncompressed += int(info.file_size)
        expression_key = normalize_expression_filename(PurePosixPath(normalized_path).name)
        if expression_key is None:
            _record_error(summary, code="empty_expression_key", source_filename=info.filename)
            continue
        if _entry_has_errors(summary, info.filename):
            continue
        candidates.append(
            _ImportCandidate(
                info=info,
                normalized_path=normalized_path,
                expression_key=expression_key,
            )
        )

    summary["total_uncompressed_bytes"] = total_uncompressed
    if total_uncompressed > MAX_EXPRESSION_ZIP_TOTAL_UNCOMPRESSED_BYTES:
        _record_error(summary, code="total_uncompressed_size_exceeded")
    if _has_fatal_errors(summary):
        return []
    return sorted(candidates, key=lambda candidate: candidate.normalized_path.lower())


def _validate_entry_metadata(
    info: zipfile.ZipInfo,
    normalized_path: str,
    summary: dict[str, Any],
) -> None:
    source_filename = info.filename
    extension = PurePosixPath(normalized_path).suffix.lower()
    if info.flag_bits & 0x1:
        _record_error(summary, code="encrypted_entry", source_filename=source_filename)
    if _is_symlink(info):
        _record_error(summary, code="symlink_entry", source_filename=source_filename)
    if info.is_dir():
        return
    if extension in _NESTED_ARCHIVE_EXTENSIONS:
        _record_error(summary, code="nested_archive", source_filename=source_filename)
    elif extension not in _SUPPORTED_ARCHIVE_IMAGE_EXTENSIONS:
        _record_error(summary, code="unsupported_archive_entry", source_filename=source_filename)
    if info.file_size > MAX_EXPRESSION_ASSET_BYTES:
        _record_error(summary, code="entry_uncompressed_size_exceeded", source_filename=source_filename)
    if info.compress_size > MAX_EXPRESSION_ASSET_BYTES:
        _record_error(summary, code="entry_compressed_size_exceeded", source_filename=source_filename)
    if _decompression_ratio(info) > MAX_EXPRESSION_ZIP_DECOMPRESSION_RATIO:
        _record_error(summary, code="decompression_ratio_exceeded", source_filename=source_filename)


def _select_first_expression_candidates(
    candidates: list[_ImportCandidate],
    summary: dict[str, Any],
) -> list[_ImportCandidate]:
    selected_by_key: dict[str, _ImportCandidate] = {}
    for candidate in candidates:
        existing = selected_by_key.get(candidate.expression_key)
        if existing is None:
            selected_by_key[candidate.expression_key] = candidate
        else:
            summary["duplicates"].append(
                {
                    "expression_key": candidate.expression_key,
                    "source_filename": candidate.normalized_path,
                    "selected_source_filename": existing.normalized_path,
                }
            )
    return list(selected_by_key.values())


def _store_candidates(
    archive: zipfile.ZipFile,
    candidates: list[_ImportCandidate],
    *,
    repo: VisualIdentityRepository,
    owner_user_id: int,
    draft_id: int,
    storage_root: str | Path | None,
    summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    slot_map: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="visual_identity_import_") as tmp_dir:
        temp_root = Path(tmp_dir)
        for candidate in candidates:
            temp_path = temp_root / PurePosixPath(candidate.normalized_path).name
            try:
                _copy_zip_entry_to_temp_file(archive, candidate.info, temp_path)
                stored = validate_and_store_visual_identity_asset(
                    source_path=temp_path,
                    owner_user_id=owner_user_id,
                    expression_key=candidate.expression_key,
                    storage_root=storage_root,
                    pack_id=f"draft-{draft_id}",
                )
                asset = repo.create_asset(
                    owner_user_id=owner_user_id,
                    draft_id=draft_id,
                    expression_key=candidate.expression_key,
                    original_expression_key=PurePosixPath(candidate.normalized_path).stem,
                    display_label=display_label_for_expression_key(candidate.expression_key),
                    source_filename=candidate.normalized_path,
                    storage_relpath=stored.relpath,
                    content_type=stored.content_type,
                    bytes=stored.bytes,
                    sha256=stored.sha256,
                    width=stored.width,
                    height=stored.height,
                    is_animated=stored.is_animated,
                    frame_count=stored.frame_count,
                    duration_ms=stored.duration_ms,
                    preview_relpath=stored.preview_relpath,
                )
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                _record_error(
                    summary,
                    code=str(exc) or "asset_validation_failed",
                    source_filename=candidate.normalized_path,
                )
                continue

            slot_map[candidate.expression_key] = {
                "asset_id": asset["id"],
                "source_filename": candidate.normalized_path,
                "display_label": asset["display_label"],
                "content_type": asset["content_type"],
                "bytes": asset["bytes"],
                "width": asset["width"],
                "height": asset["height"],
            }
            summary["accepted"].append(
                {
                    "asset_id": asset["id"],
                    "expression_key": candidate.expression_key,
                    "source_filename": candidate.normalized_path,
                }
            )
    return slot_map


def _copy_zip_entry_to_temp_file(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    temp_path: Path,
) -> None:
    written = 0
    with archive.open(info, "r") as source, temp_path.open("wb") as target:
        while True:
            chunk = source.read(_ZIP_CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_EXPRESSION_ASSET_BYTES:
                raise ValueError("entry_uncompressed_size_exceeded")
            target.write(chunk)
    if written != info.file_size:
        raise ValueError("zip_entry_size_mismatch")


def _finish_import(
    repo: VisualIdentityRepository,
    *,
    owner_user_id: int,
    draft_id: int,
    slot_map: dict[str, Any],
    summary: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    repo.update_draft_slot_map(
        draft_id=draft_id,
        owner_user_id=owner_user_id,
        slot_map=slot_map,
    )
    repo.update_draft_validation_summary(
        draft_id=draft_id,
        owner_user_id=owner_user_id,
        validation_summary=summary,
    )
    error = {"errors": summary["errors"]} if status == "failed" else None
    return repo.set_draft_status(
        draft_id=draft_id,
        owner_user_id=owner_user_id,
        status=status,
        error=error,
    )


def _normalized_archive_path(raw_name: str) -> str | None:
    if not raw_name or "\x00" in raw_name:
        return None
    if "\\" in raw_name:
        return None
    member_name = raw_name[:-1] if raw_name.endswith("/") else raw_name
    if not member_name:
        return None
    raw_parts = member_name.split("/")
    if any(part == "" for part in raw_parts):
        return None
    if any(_has_windows_drive_letter(part) for part in raw_parts):
        return None
    path = PurePosixPath(member_name)
    if path.is_absolute():
        return None
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    if any(_has_windows_drive_letter(part) for part in parts):
        return None
    normalized_path = PurePosixPath(*parts).as_posix()
    if not PurePosixPath(normalized_path).name:
        return None
    return normalized_path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_IFMT(mode) == stat.S_IFLNK


def _has_windows_drive_letter(part: str) -> bool:
    return len(part) >= 2 and part[0].isalpha() and part[1] == ":"


def _decompression_ratio(info: zipfile.ZipInfo) -> float:
    if info.file_size <= 0:
        return 0.0
    if info.compress_size <= 0:
        return float("inf")
    return info.file_size / info.compress_size


def _entry_has_errors(summary: dict[str, Any], source_filename: str) -> bool:
    skippable_entry_codes = _FATAL_ARCHIVE_ERROR_CODES | {
        "empty_expression_key",
        "unsupported_archive_entry",
    }
    return any(
        error.get("source_filename") == source_filename
        and error.get("code") in skippable_entry_codes
        for error in summary["errors"]
    )


def _has_fatal_errors(summary: dict[str, Any]) -> bool:
    return any(error.get("code") in _FATAL_ARCHIVE_ERROR_CODES for error in summary["errors"])


def _record_error(
    summary: dict[str, Any],
    *,
    code: str,
    source_filename: str | None = None,
) -> None:
    error = {"code": code}
    if source_filename is not None:
        error["source_filename"] = source_filename
    summary["errors"].append(error)


def _empty_summary(*, source_filename: str) -> dict[str, Any]:
    return {
        "accepted": [],
        "archive_bytes": 0,
        "directories": [],
        "duplicates": [],
        "entry_count": 0,
        "errors": [],
        "source_filename": source_filename,
        "total_uncompressed_bytes": 0,
    }


__all__ = [
    "MAX_EXPRESSION_ZIP_BYTES",
    "MAX_EXPRESSION_ZIP_DECOMPRESSION_RATIO",
    "MAX_EXPRESSION_ZIP_ENTRIES",
    "MAX_EXPRESSION_ZIP_TOTAL_UNCOMPRESSED_BYTES",
    "import_visual_identity_expression_zip",
]
