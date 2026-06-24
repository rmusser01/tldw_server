from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    normalize_output_storage_filename,
)
from tldw_Server_API.app.core.Ingestion_Sources.diffing import normalize_archive_members
from tldw_Server_API.app.core.Ingestion_Sources.diffing import normalized_archive_member_paths
from tldw_Server_API.app.core.Ingestion_Sources.local_directory import (
    MEDIA_SUPPORTED_SUFFIXES,
    NOTES_SUPPORTED_SUFFIXES,
)
from tldw_Server_API.app.core.Ingestion_Sources.service import (
    create_source_artifact,
    delete_source_artifact,
    delete_source_snapshot,
    list_source_artifacts,
    list_source_snapshots,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files import (
    convert_document_to_text,
)

SUPPORTED_ARCHIVE_SUFFIXES: frozenset[str] = frozenset(
    NOTES_SUPPORTED_SUFFIXES | MEDIA_SUPPORTED_SUFFIXES
)
_ZIP_ARCHIVE_SUFFIXES: tuple[str, ...] = (".zip",)
_TAR_ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
)
ARCHIVE_UPLOAD_SUFFIXES: tuple[str, ...] = _ZIP_ARCHIVE_SUFFIXES + _TAR_ARCHIVE_SUFFIXES
_ARCHIVE_MAX_MEMBERS_ENV = "INGESTION_SOURCES_ARCHIVE_MAX_MEMBERS"
_ARCHIVE_MEMBER_MAX_BYTES_ENV = "INGESTION_SOURCES_ARCHIVE_MEMBER_MAX_BYTES"
_ARCHIVE_TOTAL_MAX_BYTES_ENV = "INGESTION_SOURCES_ARCHIVE_TOTAL_MAX_BYTES"
_DEFAULT_ARCHIVE_MAX_MEMBERS = 10_000
_DEFAULT_ARCHIVE_MEMBER_MAX_BYTES = 50 * 1024 * 1024
_DEFAULT_ARCHIVE_TOTAL_MAX_BYTES = 250 * 1024 * 1024
_ARCHIVE_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class _ArchiveMember:
    filename: str


class _ArchiveFormatError(ValueError):
    """Raised when archive bytes do not match the requested container format."""


def _archive_limit_int(env_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{env_name} must be a positive integer")
    return value


def _archive_limits() -> tuple[int, int, int]:
    return (
        _archive_limit_int(_ARCHIVE_MAX_MEMBERS_ENV, _DEFAULT_ARCHIVE_MAX_MEMBERS),
        _archive_limit_int(_ARCHIVE_MEMBER_MAX_BYTES_ENV, _DEFAULT_ARCHIVE_MEMBER_MAX_BYTES),
        _archive_limit_int(_ARCHIVE_TOTAL_MAX_BYTES_ENV, _DEFAULT_ARCHIVE_TOTAL_MAX_BYTES),
    )


def _read_archive_member_stream_limited(handle: Any, *, member_name: str, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total_read = 0
    while True:
        chunk = handle.read(_ARCHIVE_READ_CHUNK_BYTES)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > max_bytes:
            raise ValueError(
                f"Archive member '{member_name}' exceeds archive member size limit of {max_bytes} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def process_pdf(file_input, *, filename: str, **kwargs):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import (
        process_pdf as _process_pdf,
    )

    return _process_pdf(file_input, filename=filename, **kwargs)


def process_epub(file_path, **kwargs):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib import (
        process_epub as _process_epub,
    )

    return _process_epub(file_path, **kwargs)


def _is_safe_archive_member_name(member_name: str) -> bool:
    if not member_name:
        return False
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    pure_path = PurePosixPath(normalized)
    if ".." in pure_path.parts:
        return False
    return True


def _archive_kind_from_filename(filename: str) -> str | None:
    normalized = str(filename or "").strip().lower()
    if normalized.endswith(_ZIP_ARCHIVE_SUFFIXES):
        return "zip"
    if normalized.endswith(_TAR_ARCHIVE_SUFFIXES):
        return "tar"
    return None


def validate_archive_upload_filename(filename: str) -> str:
    normalized = str(filename or "").strip()
    if not normalized:
        raise ValueError("Archive upload filename is required.")
    if _archive_kind_from_filename(normalized) is None:
        supported = ", ".join(ARCHIVE_UPLOAD_SUFFIXES)
        raise ValueError(f"Unsupported archive type. Supported uploads: {supported}")
    return normalized


async def stage_archive_candidate(
    *,
    source_id: int,
    filename: str,
    current_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": f"candidate-{uuid4()}",
        "source_id": int(source_id),
        "filename": filename,
        "status": "candidate",
        "previous_snapshot_id": None if current_snapshot is None else current_snapshot.get("id"),
    }


async def mark_snapshot_failed(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate["status"] = "failed"
    return candidate


def validate_archive_members(
    archive_bytes: bytes,
    *,
    filename: str,
) -> list[tuple[_ArchiveMember, bytes]]:
    archive_kind = _archive_kind_from_filename(filename)
    if archive_kind == "zip":
        return _validate_zip_archive_members(archive_bytes, filename=filename)
    if archive_kind == "tar":
        return _validate_tar_archive_members(archive_bytes, filename=filename)

    try:
        return _validate_zip_archive_members(archive_bytes, filename=filename)
    except _ArchiveFormatError:
        return _validate_tar_archive_members(archive_bytes, filename=filename)


def _validate_zip_archive_members(
    archive_bytes: bytes,
    *,
    filename: str,
) -> list[tuple[_ArchiveMember, bytes]]:
    max_members, member_max_bytes, total_max_bytes = _archive_limits()
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            members: list[tuple[_ArchiveMember, bytes]] = []
            member_count = 0
            total_uncompressed_bytes = 0
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if not _is_safe_archive_member_name(member.filename):
                    raise ValueError(f"Archive contains unsafe path: {member.filename}")
                if getattr(member, "flag_bits", 0) & 0x1:
                    raise ValueError("Encrypted ZIP archives are not supported.")
                external_type = (member.external_attr >> 16) & 0xFFFF
                if external_type and stat.S_ISLNK(external_type):
                    raise ValueError(f"Archive contains symbolic link: {member.filename}")
                member_count += 1
                if member_count > max_members:
                    raise ValueError(
                        f"Archive exceeds archive member count limit of {max_members} files"
                    )
                if int(member.file_size) > member_max_bytes:
                    raise ValueError(
                        f"Archive member '{member.filename}' exceeds archive member size limit "
                        f"of {member_max_bytes} bytes"
                    )
                if total_uncompressed_bytes + int(member.file_size) > total_max_bytes:
                    raise ValueError(
                        f"Archive exceeds archive total uncompressed size limit of {total_max_bytes} bytes"
                    )
                with archive.open(member, "r") as handle:
                    content = _read_archive_member_stream_limited(
                        handle,
                        member_name=member.filename,
                        max_bytes=member_max_bytes,
                    )
                total_uncompressed_bytes += len(content)
                if total_uncompressed_bytes > total_max_bytes:
                    raise ValueError(
                        f"Archive exceeds archive total uncompressed size limit of {total_max_bytes} bytes"
                    )
                members.append((_ArchiveMember(member.filename), content))
            return members
    except zipfile.BadZipFile as exc:
        raise _ArchiveFormatError(f"Invalid ZIP archive: {filename}") from exc


def _validate_tar_archive_members(
    archive_bytes: bytes,
    *,
    filename: str,
) -> list[tuple[_ArchiveMember, bytes]]:
    max_members, member_max_bytes, total_max_bytes = _archive_limits()
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as archive:
            members: list[tuple[_ArchiveMember, bytes]] = []
            member_count = 0
            total_uncompressed_bytes = 0
            for member in archive:
                if member.isdir():
                    continue
                if not _is_safe_archive_member_name(member.name):
                    raise ValueError(f"Archive contains unsafe path: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive contains symbolic link: {member.name}")
                if not member.isfile():
                    raise ValueError(f"Archive contains unsupported member type: {member.name}")
                member_count += 1
                if member_count > max_members:
                    raise ValueError(
                        f"Archive exceeds archive member count limit of {max_members} files"
                    )
                member_size = int(member.size)
                if member_size > member_max_bytes:
                    raise ValueError(
                        f"Archive member '{member.name}' exceeds archive member size limit "
                        f"of {member_max_bytes} bytes"
                    )
                if total_uncompressed_bytes + member_size > total_max_bytes:
                    raise ValueError(
                        f"Archive exceeds archive total uncompressed size limit of {total_max_bytes} bytes"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Archive member could not be read: {member.name}")
                with extracted:
                    content = _read_archive_member_stream_limited(
                        extracted,
                        member_name=member.name,
                        max_bytes=member_max_bytes,
                    )
                total_uncompressed_bytes += len(content)
                if total_uncompressed_bytes > total_max_bytes:
                    raise ValueError(
                        f"Archive exceeds archive total uncompressed size limit of {total_max_bytes} bytes"
                    )
                members.append((_ArchiveMember(member.name), content))
            return members
    except (tarfile.CompressionError, tarfile.ReadError, EOFError) as exc:
        raise _ArchiveFormatError(f"Invalid TAR archive: {filename}") from exc


def build_archive_snapshot(
    members: list[tuple[_ArchiveMember, bytes]],
) -> dict[str, dict[str, Any]]:
    items, failures = build_archive_snapshot_with_failures(members, sink_type="notes")
    if failures:
        first_failure = next(iter(failures.values()))
        raise ValueError(str(first_failure.get("error") or "Archive member ingestion failed."))
    return items


def _supported_archive_suffixes_for_sink(sink_type: str) -> frozenset[str]:
    if str(sink_type or "").strip().lower() == "media":
        return MEDIA_SUPPORTED_SUFFIXES
    return NOTES_SUPPORTED_SUFFIXES


def _raw_metadata_from_processing_result(
    result: dict[str, Any],
    *,
    source_format: str,
) -> dict[str, Any]:
    raw_metadata: dict[str, Any] = {"source_format": source_format}
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        raw_metadata.update(metadata)
        title = metadata.get("title")
        author = metadata.get("author")
        if title and "extracted_title" not in raw_metadata:
            raw_metadata["extracted_title"] = title
        if author and "extracted_author" not in raw_metadata:
            raw_metadata["extracted_author"] = author
    parser_used = result.get("parser_used")
    if parser_used:
        raw_metadata["parser_used"] = parser_used
    return raw_metadata


def _media_member_content_to_text(
    *,
    member_name: str,
    content: bytes,
) -> tuple[str, str, dict[str, Any]]:
    suffix = PurePosixPath(member_name).suffix.lower()
    filename = PurePosixPath(member_name).name or member_name
    if suffix == ".pdf":
        result = process_pdf(
            content,
            filename=filename,
            perform_chunking=False,
            perform_analysis=False,
        )
        if not isinstance(result, dict) or str(result.get("status") or "").strip().lower() == "error":
            raise ValueError(str((result or {}).get("error") or f"Failed to process PDF '{filename}'"))
        return (
            str(result.get("content") or ""),
            "pdf",
            _raw_metadata_from_processing_result(result, source_format="pdf"),
        )

    if suffix == ".epub":
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(content)
                temp_path = Path(handle.name)
            result = process_epub(
                str(temp_path),
                perform_chunking=False,
                perform_analysis=False,
            )
        finally:
            if temp_path is not None:
                with contextlib.suppress(FileNotFoundError, OSError):
                    temp_path.unlink()
        if not isinstance(result, dict) or str(result.get("status") or "").strip().lower() == "error":
            raise ValueError(str((result or {}).get("error") or f"Failed to process EPUB '{filename}'"))
        return (
            str(result.get("content") or ""),
            "epub",
            _raw_metadata_from_processing_result(result, source_format="epub"),
        )

    return _member_content_to_text(member_name=member_name, content=content, sink_type="notes")


def build_archive_snapshot_with_failures(
    members: list[tuple[_ArchiveMember, bytes]],
    *,
    sink_type: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    supported_suffixes = _supported_archive_suffixes_for_sink(sink_type)
    hashes: dict[str, str] = {}
    member_names: list[str] = []
    raw_contents: dict[str, bytes] = {}
    for member, content in members:
        suffix = PurePosixPath(member.filename).suffix.lower()
        if suffix not in supported_suffixes:
            continue
        member_names.append(member.filename)
        raw_contents[member.filename] = content
        hashes[member.filename] = hashlib.sha256(content).hexdigest()

    member_relative_paths = normalized_archive_member_paths(member_names)
    items = normalize_archive_members(member_names, hashes)
    failed_items: dict[str, dict[str, Any]] = {}
    for member_name in member_names:
        relative_path = member_relative_paths.get(member_name)
        if not relative_path:
            continue
        try:
            if str(sink_type or "").strip().lower() == "media" and PurePosixPath(member_name).suffix.lower() in {".epub", ".pdf"}:
                text_content, source_format, raw_metadata = _media_member_content_to_text(
                    member_name=member_name,
                    content=raw_contents[member_name],
                )
            else:
                text_content, source_format, raw_metadata = _member_content_to_text(
                    member_name=member_name,
                    content=raw_contents[member_name],
                    sink_type=sink_type,
                )
        except (AttributeError, LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            items.pop(relative_path, None)
            failed_items[relative_path] = {
                "relative_path": relative_path,
                "source_format": PurePosixPath(member_name).suffix.lower().lstrip(".") or "unknown",
                "size": len(raw_contents[member_name]),
                "error": str(exc),
            }
            continue
        items[relative_path].update(
            {
                "text": text_content,
                "source_format": source_format,
                "raw_metadata": raw_metadata,
                "size": len(raw_contents[member_name]),
                "content_hash": hashlib.sha256(text_content.encode("utf-8")).hexdigest(),
            }
        )
    return items, failed_items


def _member_content_to_text(
    *,
    member_name: str,
    content: bytes,
    sink_type: str,
) -> tuple[str, str, dict[str, Any]]:
    suffix = PurePosixPath(member_name).suffix.lower()
    if suffix == ".markdown":
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content.decode("latin-1")
        return text_content, "markdown", {}

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        if str(sink_type or "").strip().lower() == "media" and suffix in {".epub", ".pdf"}:
            return _media_member_content_to_text(member_name=member_name, content=content)
        return convert_document_to_text(temp_path)
    finally:
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError, OSError):
                temp_path.unlink()


def _archive_artifact_dir(*, user_id: int, source_id: int) -> Path:
    base_dir = DatabasePaths.get_user_base_directory(user_id)
    artifact_dir = base_dir / "ingestion_sources" / str(int(source_id)) / "archives"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _archive_artifact_path(*, user_id: int, source_id: int, filename: str) -> Path:
    safe_name = normalize_output_storage_filename(
        filename or "archive.zip",
        allow_absolute=False,
        reject_relative_with_separators=True,
        expand_user=False,
    )
    return _archive_artifact_dir(user_id=user_id, source_id=source_id) / f"{uuid4().hex}-{safe_name}"


async def persist_archive_artifact(
    db,
    *,
    user_id: int,
    source_id: int,
    snapshot_id: int,
    filename: str,
    archive_bytes: bytes,
) -> dict[str, Any]:
    storage_path = _archive_artifact_path(
        user_id=user_id,
        source_id=source_id,
        filename=filename,
    )
    temp_path = storage_path.with_suffix(f"{storage_path.suffix}.tmp")
    temp_path.write_bytes(archive_bytes)
    temp_path.replace(storage_path)
    try:
        artifact = await create_source_artifact(
            db,
            source_id=source_id,
            snapshot_id=snapshot_id,
            artifact_kind="archive_upload",
            status="staged",
            storage_path=str(storage_path),
            metadata={
                "filename": filename or "archive.zip",
                "byte_size": len(archive_bytes),
                "checksum": hashlib.sha256(archive_bytes).hexdigest(),
            },
        )
    except Exception:
        with contextlib.suppress(FileNotFoundError, OSError):
            storage_path.unlink()
        raise
    return artifact


async def persist_archive_artifact_from_file(
    db,
    *,
    user_id: int,
    source_id: int,
    snapshot_id: int,
    filename: str,
    staged_file_path: str | Path,
    byte_size: int,
    checksum: str,
) -> dict[str, Any]:
    storage_path = _archive_artifact_path(
        user_id=user_id,
        source_id=source_id,
        filename=filename,
    )
    temp_storage_path = storage_path.with_suffix(f"{storage_path.suffix}.tmp")
    source_path = Path(staged_file_path)
    try:
        with source_path.open("rb") as source_handle, temp_storage_path.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        temp_storage_path.replace(storage_path)
        artifact = await create_source_artifact(
            db,
            source_id=source_id,
            snapshot_id=snapshot_id,
            artifact_kind="archive_upload",
            status="staged",
            storage_path=str(storage_path),
            metadata={
                "filename": filename or "archive.zip",
                "byte_size": int(byte_size),
                "checksum": checksum,
            },
        )
    except Exception:
        with contextlib.suppress(FileNotFoundError, OSError):
            temp_storage_path.unlink()
        with contextlib.suppress(FileNotFoundError, OSError):
            storage_path.unlink()
        raise
    finally:
        with contextlib.suppress(FileNotFoundError, OSError):
            source_path.unlink()
    return artifact


def load_archive_artifact_bytes(artifact: dict[str, Any]) -> bytes:
    storage_path = str(artifact.get("storage_path") or "").strip()
    if not storage_path:
        raise ValueError("Archive artifact is missing a storage_path.")
    path = Path(storage_path)
    if not path.exists():
        raise ValueError(f"Archive artifact is missing from storage: {path}")
    return path.read_bytes()


def build_archive_snapshot_from_bytes(
    *,
    archive_bytes: bytes,
    filename: str,
) -> dict[str, dict[str, Any]]:
    members = validate_archive_members(archive_bytes, filename=filename)
    return build_archive_snapshot(members)


def build_archive_snapshot_from_bytes_with_failures(
    *,
    archive_bytes: bytes,
    filename: str,
    sink_type: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    members = validate_archive_members(archive_bytes, filename=filename)
    return build_archive_snapshot_with_failures(members, sink_type=sink_type)


def inspect_archive_candidate_file(
    *,
    archive_path: str | Path,
    filename: str,
    sink_type: str,
) -> dict[str, Any]:
    normalized_filename = validate_archive_upload_filename(filename)
    archive_kind = _archive_kind_from_filename(normalized_filename)
    if archive_kind == "zip":
        member_names = _inspect_zip_archive_file(archive_path, filename=normalized_filename)
    elif archive_kind == "tar":
        member_names = _inspect_tar_archive_file(archive_path, filename=normalized_filename)
    else:
        raise ValueError(f"Unsupported archive type. Supported uploads: {', '.join(ARCHIVE_UPLOAD_SUFFIXES)}")

    supported_suffixes = _supported_archive_suffixes_for_sink(sink_type)
    supported_names = [
        member_name
        for member_name in member_names
        if PurePosixPath(member_name).suffix.lower() in supported_suffixes
    ]
    normalized_items = normalize_archive_members(
        supported_names,
        {member_name: "" for member_name in supported_names},
    )
    return {
        "filename": normalized_filename,
        "item_count": len(normalized_items),
    }


def _inspect_zip_archive_file(archive_path: str | Path, *, filename: str) -> list[str]:
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            member_names: list[str] = []
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if not _is_safe_archive_member_name(member.filename):
                    raise ValueError(f"Archive contains unsafe path: {member.filename}")
                if getattr(member, "flag_bits", 0) & 0x1:
                    raise ValueError("Encrypted ZIP archives are not supported.")
                external_type = (member.external_attr >> 16) & 0xFFFF
                if external_type and stat.S_ISLNK(external_type):
                    raise ValueError(f"Archive contains symbolic link: {member.filename}")
                member_names.append(member.filename)
            return member_names
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {filename}") from exc


def _inspect_tar_archive_file(archive_path: str | Path, *, filename: str) -> list[str]:
    try:
        with tarfile.open(name=str(archive_path), mode="r:*") as archive:
            member_names: list[str] = []
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not _is_safe_archive_member_name(member.name):
                    raise ValueError(f"Archive contains unsafe path: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive contains symbolic link: {member.name}")
                if not member.isfile():
                    raise ValueError(f"Archive contains unsupported member type: {member.name}")
                member_names.append(member.name)
            return member_names
    except (tarfile.CompressionError, tarfile.ReadError, EOFError) as exc:
        raise ValueError(f"Invalid TAR archive: {filename}") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_created_at(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return _utc_now()
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return _utc_now()


def _retention_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _remove_storage_path(storage_path: str | None) -> bool:
    if not storage_path:
        return False
    path = Path(str(storage_path))
    removed = False
    with contextlib.suppress(FileNotFoundError, OSError):
        path.unlink()
        removed = True
    current = path.parent
    for _ in range(3):
        with contextlib.suppress(OSError):
            current.rmdir()
        current = current.parent
    return removed


async def prune_archive_source_retention(
    db,
    *,
    source_id: int,
    successful_snapshots_to_keep: int | None = None,
    failed_snapshot_max_age_seconds: int | None = None,
    staged_snapshot_max_age_seconds: int | None = None,
) -> dict[str, int]:
    success_keep_count = max(
        1,
        successful_snapshots_to_keep
        if successful_snapshots_to_keep is not None
        else _retention_int("INGESTION_SOURCES_SUCCESSFUL_SNAPSHOT_RETENTION_COUNT", 3),
    )
    failed_max_age = (
        failed_snapshot_max_age_seconds
        if failed_snapshot_max_age_seconds is not None
        else _retention_int("INGESTION_SOURCES_FAILED_SNAPSHOT_RETENTION_SECONDS", 86400)
    )
    staged_max_age = (
        staged_snapshot_max_age_seconds
        if staged_snapshot_max_age_seconds is not None
        else _retention_int("INGESTION_SOURCES_STAGED_SNAPSHOT_RETENTION_SECONDS", 3600)
    )

    snapshots = await list_source_snapshots(db, source_id=source_id)
    artifacts = await list_source_artifacts(db, source_id=source_id, artifact_kind="archive_upload")
    artifacts_by_snapshot: dict[int, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        snapshot_id = artifact.get("snapshot_id")
        if snapshot_id is None:
            continue
        artifacts_by_snapshot.setdefault(int(snapshot_id), []).append(artifact)

    now = _utc_now()
    success_snapshot_ids = [
        int(snapshot["id"])
        for snapshot in snapshots
        if str(snapshot.get("status") or "").strip().lower() == "success"
    ]
    kept_success_ids = set(success_snapshot_ids[:success_keep_count])

    deleted_snapshots = 0
    deleted_artifacts = 0
    deleted_files = 0
    for snapshot in snapshots:
        snapshot_id = int(snapshot["id"])
        status = str(snapshot.get("status") or "").strip().lower()
        created_at = _parse_created_at(snapshot.get("created_at"))
        age_seconds = max(0, int((now - created_at).total_seconds()))
        should_delete = False
        if status == "success":
            should_delete = snapshot_id not in kept_success_ids
        elif status == "failed":
            should_delete = age_seconds >= max(0, failed_max_age)
        elif status == "staged":
            should_delete = age_seconds >= max(0, staged_max_age)

        if not should_delete:
            continue

        for artifact in artifacts_by_snapshot.get(snapshot_id, []):
            if _remove_storage_path(artifact.get("storage_path")):
                deleted_files += 1
            await delete_source_artifact(db, artifact_id=int(artifact["id"]))
            deleted_artifacts += 1
        await delete_source_snapshot(db, snapshot_id=snapshot_id)
        deleted_snapshots += 1

    return {
        "deleted_snapshots": deleted_snapshots,
        "deleted_artifacts": deleted_artifacts,
        "deleted_files": deleted_files,
    }


async def apply_archive_candidate(
    *,
    source_id: int,
    archive_bytes: bytes,
    filename: str,
    current_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate = await stage_archive_candidate(
        source_id=source_id,
        filename=filename,
        current_snapshot=current_snapshot,
    )
    try:
        members = validate_archive_members(archive_bytes, filename=filename)
        items = build_archive_snapshot(members)
    except Exception:
        await mark_snapshot_failed(candidate)
        raise

    candidate["status"] = "staged"
    return {
        "candidate_snapshot": candidate,
        "items": items,
    }
