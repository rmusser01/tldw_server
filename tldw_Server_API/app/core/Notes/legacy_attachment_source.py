"""Bounded, read-only access to legacy Notes attachment files."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import LegacyAttachmentSourceError
from tldw_Server_API.app.core.Utils.Utils import sanitize_filename

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


LEGACY_ATTACHMENTS_DIRNAME = "notes_attachments"
LEGACY_ATTACHMENT_META_SUFFIX = ".meta.json"
LEGACY_ATTACHMENT_NOTE_PAGE_LIMIT = 200
LEGACY_ATTACHMENT_CANDIDATE_LIMIT = 1_000
LEGACY_ATTACHMENT_SIDECAR_LIMIT_BYTES = 64 * 1024
LEGACY_ATTACHMENT_CURSOR_LIMIT_BYTES = 4_096
_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class LegacyAttachmentCandidate:
    """Immutable snapshot of one confined legacy attachment source."""

    note_id: str
    file_name: str
    source_key: str
    relative_path: str
    size_bytes: int
    modified_ns: int
    sha256: str
    metadata: dict[str, Any]


def safe_legacy_note_attachment_dirname(note_id: str) -> str:
    """Derive the existing legacy directory name from an authoritative note ID."""

    text = str(note_id or "").strip()
    if not text:
        return "note"
    safe = sanitize_filename(text, max_total_length=96).replace(" ", "_").strip("._")
    if safe and safe not in {".", ".."}:
        return safe
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"note_{digest}"


def legacy_attachment_base_directory(
    user_id: int | str,
    *,
    user_root: Path | None = None,
) -> Path:
    """Return the legacy attachment root without creating it."""

    root = (
        Path(user_root)
        if user_root is not None
        else DatabasePaths.resolve_user_base_directory(user_id)
    ).absolute()
    return root / LEGACY_ATTACHMENTS_DIRNAME


def legacy_attachment_note_directory(
    user_id: int | str,
    note_id: str,
    *,
    user_root: Path | None = None,
) -> Path:
    """Return the lexically confined legacy directory for an owned note ID."""

    base = legacy_attachment_base_directory(user_id, user_root=user_root)
    note_dir = base / safe_legacy_note_attachment_dirname(note_id)
    if not note_dir.absolute().is_relative_to(base.absolute()):
        raise LegacyAttachmentSourceError("notes_attachment_source_unsafe")
    return note_dir


def legacy_attachment_metadata_path(file_path: Path) -> Path:
    """Return the legacy sidecar path for an attachment path."""

    return file_path.with_name(f"{file_path.name}{LEGACY_ATTACHMENT_META_SUFFIX}")


class LegacyAttachmentSource:
    """Enumerate and hash owned legacy attachment files without mutating them."""

    def __init__(
        self,
        note_db: CharactersRAGDB,
        *,
        owner_user_id: str,
        user_root: Path | None = None,
    ) -> None:
        if str(note_db.client_id) != str(owner_user_id):
            raise LegacyAttachmentSourceError("notes_attachment_source_owner_mismatch")
        self._notes = note_db.note_store
        self._owner_user_id = str(owner_user_id)
        self._user_root = (
            Path(user_root)
            if user_root is not None
            else DatabasePaths.resolve_user_base_directory(owner_user_id)
        ).absolute()
        self._base_dir = legacy_attachment_base_directory(
            owner_user_id,
            user_root=self._user_root,
        )

    def note_directory(self, note_id: str) -> Path:
        """Return the confined legacy directory derived from the owned note ID."""

        return legacy_attachment_note_directory(
            self._owner_user_id,
            note_id,
            user_root=self._user_root,
        )

    def list_note_ids(
        self,
        *,
        after_note_id: str | None = None,
        limit: int = LEGACY_ATTACHMENT_NOTE_PAGE_LIMIT,
    ) -> tuple[str, ...]:
        """List owned live and soft-deleted note IDs in bounded keyset order."""

        if not 1 <= limit <= LEGACY_ATTACHMENT_NOTE_PAGE_LIMIT:
            raise ValueError("note page limit must be 1..200")
        return self._notes.list_note_ids_page(
            after_note_id=after_note_id,
            limit=limit,
        )

    def list_candidates(
        self,
        note_id: str,
        *,
        after_source_key: str | None = None,
        limit: int = LEGACY_ATTACHMENT_CANDIDATE_LIMIT,
    ) -> tuple[LegacyAttachmentCandidate, ...]:
        """Return a bounded page of sorted, immutable source snapshots."""

        if not 1 <= limit <= LEGACY_ATTACHMENT_CANDIDATE_LIMIT:
            raise ValueError("candidate limit must be 1..1000")
        self._validate_cursor(note_id, after_source_key)
        if not self._notes.owns_note_id(note_id):
            raise LegacyAttachmentSourceError(
                "notes_attachment_source_note_not_found"
            )
        note_dir = self.note_directory(note_id)
        if not note_dir.exists():
            return ()
        base_fd = self._open_directory(self._base_dir)
        try:
            note_fd = self._open_directory(
                safe_legacy_note_attachment_dirname(note_id),
                dir_fd=base_fd,
            )
            try:
                names = heapq.nsmallest(
                    limit,
                    self._candidate_names(note_fd, note_id, after_source_key),
                    key=lambda item: item[0],
                )
                return tuple(
                    self._snapshot_candidate(note_fd, note_id, name, source_key)
                    for source_key, name in names
                )
            finally:
                os.close(note_fd)
        finally:
            os.close(base_fd)

    def iter_candidate_chunks(
        self,
        candidate: LegacyAttachmentCandidate,
        *,
        chunk_size: int,
    ) -> Iterator[bytes]:
        """Stream one unchanged candidate through confined no-follow descriptors."""

        if chunk_size < 1 or chunk_size > 64 * 1024:
            raise ValueError("candidate chunk size must be 1..65536")
        if not self._notes.owns_note_id(candidate.note_id):
            raise LegacyAttachmentSourceError(
                "notes_attachment_source_note_not_found"
            )
        expected_key = self._source_key(candidate.note_id, candidate.file_name)
        if candidate.source_key != expected_key or candidate.relative_path != expected_key:
            raise LegacyAttachmentSourceError("notes_attachment_source_unsafe")
        base_fd = self._open_directory(self._base_dir)
        try:
            note_fd = self._open_directory(
                safe_legacy_note_attachment_dirname(candidate.note_id),
                dir_fd=base_fd,
            )
            try:
                file_fd = self._open_regular(candidate.file_name, dir_fd=note_fd)
                try:
                    before = os.fstat(file_fd)
                    if (
                        int(before.st_size) != candidate.size_bytes
                        or int(before.st_mtime_ns) != candidate.modified_ns
                    ):
                        raise LegacyAttachmentSourceError(
                            "notes_attachment_source_changed"
                        )
                    digest = hashlib.sha256()
                    while chunk := os.read(file_fd, chunk_size):
                        digest.update(chunk)
                        yield chunk
                    after = os.fstat(file_fd)
                    if (
                        self._stat_identity(before) != self._stat_identity(after)
                        or f"sha256:{digest.hexdigest()}" != candidate.sha256
                    ):
                        raise LegacyAttachmentSourceError(
                            "notes_attachment_source_changed"
                        )
                finally:
                    os.close(file_fd)
            finally:
                os.close(note_fd)
        finally:
            os.close(base_fd)

    def verify_candidate(self, candidate: LegacyAttachmentCandidate) -> bool:
        """Re-read and verify one immutable candidate snapshot."""

        for _chunk in self.iter_candidate_chunks(
            candidate,
            chunk_size=_READ_CHUNK_BYTES,
        ):
            pass
        return True

    def _validate_cursor(self, note_id: str, cursor: str | None) -> None:
        if cursor is None:
            return
        if not isinstance(cursor, str):
            raise LegacyAttachmentSourceError("notes_attachment_source_cursor_invalid")
        if len(cursor.encode("utf-8")) > LEGACY_ATTACHMENT_CURSOR_LIMIT_BYTES:
            raise LegacyAttachmentSourceError(
                "notes_attachment_source_cursor_too_large"
            )
        prefix = self._source_prefix(note_id)
        if not cursor.startswith(prefix) or not cursor.removeprefix(prefix):
            raise LegacyAttachmentSourceError(
                "notes_attachment_source_cursor_invalid"
            )

    def _candidate_names(
        self,
        note_fd: int,
        note_id: str,
        after_source_key: str | None,
    ):
        with os.scandir(note_fd) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise LegacyAttachmentSourceError(
                        "notes_attachment_source_unsafe"
                    )
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.name.endswith(LEGACY_ATTACHMENT_META_SUFFIX):
                    continue
                source_key = self._source_key(note_id, entry.name)
                if after_source_key is None or source_key > after_source_key:
                    yield source_key, entry.name

    def _source_key(self, note_id: str, file_name: str) -> str:
        relative_path = f"{self._source_prefix(note_id)}{file_name}"
        if len(relative_path.encode("utf-8")) > LEGACY_ATTACHMENT_CURSOR_LIMIT_BYTES:
            raise LegacyAttachmentSourceError("notes_attachment_source_key_too_large")
        return relative_path

    @staticmethod
    def _source_prefix(note_id: str) -> str:
        return (
            f"{LEGACY_ATTACHMENTS_DIRNAME}/"
            f"{safe_legacy_note_attachment_dirname(note_id)}/"
        )

    def _snapshot_candidate(
        self,
        note_fd: int,
        note_id: str,
        file_name: str,
        source_key: str,
    ) -> LegacyAttachmentCandidate:
        file_fd = self._open_regular(file_name, dir_fd=note_fd)
        try:
            before = os.fstat(file_fd)
            digest = hashlib.sha256()
            while chunk := os.read(file_fd, _READ_CHUNK_BYTES):
                digest.update(chunk)
            after = os.fstat(file_fd)
            if self._stat_identity(before) != self._stat_identity(after):
                raise LegacyAttachmentSourceError("notes_attachment_source_changed")
        finally:
            os.close(file_fd)
        metadata = self._read_sidecar(note_fd, file_name)
        return LegacyAttachmentCandidate(
            note_id=note_id,
            file_name=file_name,
            source_key=source_key,
            relative_path=source_key,
            size_bytes=int(after.st_size),
            modified_ns=int(after.st_mtime_ns),
            sha256=f"sha256:{digest.hexdigest()}",
            metadata=metadata,
        )

    def _read_sidecar(self, note_fd: int, file_name: str) -> dict[str, Any]:
        sidecar_name = f"{file_name}{LEGACY_ATTACHMENT_META_SUFFIX}"
        try:
            sidecar_fd = self._open_regular(sidecar_name, dir_fd=note_fd)
        except FileNotFoundError:
            return {}
        try:
            sidecar_stat = os.fstat(sidecar_fd)
            if sidecar_stat.st_size > LEGACY_ATTACHMENT_SIDECAR_LIMIT_BYTES:
                raise LegacyAttachmentSourceError(
                    "notes_attachment_sidecar_too_large"
                )
            payload = os.read(
                sidecar_fd,
                LEGACY_ATTACHMENT_SIDECAR_LIMIT_BYTES + 1,
            )
            if len(payload) > LEGACY_ATTACHMENT_SIDECAR_LIMIT_BYTES:
                raise LegacyAttachmentSourceError(
                    "notes_attachment_sidecar_too_large"
                )
        finally:
            os.close(sidecar_fd)
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LegacyAttachmentSourceError(
                "notes_attachment_sidecar_invalid"
            ) from exc
        if not isinstance(decoded, dict):
            raise LegacyAttachmentSourceError("notes_attachment_sidecar_invalid")
        return decoded

    @staticmethod
    def _secure_open_flags(*, directory: bool) -> int:
        required = ("O_NOFOLLOW", "O_NONBLOCK")
        if directory:
            required += ("O_DIRECTORY",)
        if any(not hasattr(os, name) for name in required):
            raise LegacyAttachmentSourceError(
                "notes_attachment_source_platform_unsupported"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        return flags

    @classmethod
    def _open_directory(cls, path: str | Path, *, dir_fd: int | None = None) -> int:
        try:
            return os.open(path, cls._secure_open_flags(directory=True), dir_fd=dir_fd)
        except (NotADirectoryError, OSError) as exc:
            raise LegacyAttachmentSourceError("notes_attachment_source_unsafe") from exc

    @classmethod
    def _open_regular(cls, name: str, *, dir_fd: int) -> int:
        file_fd = os.open(name, cls._secure_open_flags(directory=False), dir_fd=dir_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            os.close(file_fd)
            raise LegacyAttachmentSourceError("notes_attachment_source_unsafe")
        return file_fd

    @staticmethod
    def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
