"""Durable managed reference image resolution for image generation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from PIL import Image

from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import MediaDbLike
from tldw_Server_API.app.core.Image_Generation.capabilities import ResolvedReferenceImage
from tldw_Server_API.app.core.Image_Generation.config import get_image_generation_config
from tldw_Server_API.app.core.Image_Generation.request_validation import effective_inline_max_bytes
from tldw_Server_API.app.core.Storage import get_storage_backend
from tldw_Server_API.app.core.Storage.storage_interface import StorageBackend, StorageError
from tldw_Server_API.app.core.exceptions import FileArtifactsValidationError

_REFERENCE_IMAGE_PAGE_SIZE = 50


@dataclass(frozen=True)
class ReferenceImageCandidate:
    """Picker-safe metadata for a durable managed reference image."""

    file_id: int
    title: str
    mime_type: str
    width: int | None
    height: int | None
    created_at: str


@dataclass(frozen=True)
class ManagedReferenceImageRow:
    """Durable source row resolved from MediaFiles + Media."""

    file_id: int
    media_id: int
    title: str
    storage_path: str
    filename: str | None
    mime_type: str
    file_size: int | None
    created_at: str


class ManagedReferenceImageRepository:
    """Repository seam for durable reference image lookup."""

    def __init__(self, media_db: MediaDbLike, *, user_id: str | int) -> None:
        self._media_db = media_db
        self._user_id = str(user_id)

    def get_candidate_row(self, file_id: int) -> ManagedReferenceImageRow:
        rows = self._query(file_id=file_id, limit=1, offset=0)
        if not rows:
            raise FileArtifactsValidationError("reference_image_not_found")
        return rows[0]

    def list_candidate_rows(self, *, limit: int = 100, offset: int = 0) -> list[ManagedReferenceImageRow]:
        return self._query(file_id=None, limit=limit, offset=offset)

    def _query(self, *, file_id: int | None, limit: int, offset: int) -> list[ManagedReferenceImageRow]:
        query = """
            SELECT
                mf.id AS file_id,
                mf.media_id AS media_id,
                COALESCE(NULLIF(TRIM(m.title), ''), NULLIF(TRIM(mf.original_filename), ''), 'Untitled image') AS title,
                mf.storage_path AS storage_path,
                mf.original_filename AS filename,
                mf.mime_type AS mime_type,
                mf.file_size AS file_size,
                mf.created_at AS created_at
            FROM MediaFiles mf
            INNER JOIN Media m ON m.id = mf.media_id
            WHERE mf.deleted = 0
              AND m.deleted = 0
              AND m.is_trash = 0
              AND lower(m.type) = 'image'
              AND lower(mf.file_type) = 'original'
              AND mf.storage_path IS NOT NULL
              AND TRIM(mf.storage_path) != ''
              AND mf.mime_type IS NOT NULL
              AND lower(mf.mime_type) LIKE 'image/%'
        """
        params: list[Any] = []
        if file_id is not None:
            query += " AND mf.id = ?"
            params.append(file_id)
        query += " ORDER BY mf.created_at DESC, mf.id DESC LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)

        cursor = self._media_db.execute_query(query, tuple(params))
        rows = cursor.fetchall()
        return [
            ManagedReferenceImageRow(
                file_id=int(row["file_id"]),
                media_id=int(row["media_id"]),
                title=str(row["title"]),
                storage_path=str(row["storage_path"]),
                filename=row["filename"],
                mime_type=str(row["mime_type"]),
                file_size=int(row["file_size"]) if row["file_size"] is not None else None,
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]


class ReferenceImageOperationalError(RuntimeError):
    """Operational failure while accessing durable reference-image storage."""


def _probe_dimensions(image_bytes: bytes) -> tuple[int | None, int | None]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            image.verify()
            return int(width), int(height)
    except Exception as exc:
        raise FileArtifactsValidationError("reference_image_invalid") from exc


def _reference_image_max_bytes() -> int:
    return effective_inline_max_bytes(get_image_generation_config())


def _is_known_oversized(row: ManagedReferenceImageRow, max_bytes: int) -> bool:
    return row.file_size is not None and row.file_size > max_bytes


def _sanitize_path_component(component: str) -> str:
    sanitized = component.replace("/", "_").replace("\\", "_")
    sanitized = sanitized.replace("..", "_")
    sanitized = sanitized.strip().strip(".")
    if not sanitized:
        sanitized = "unnamed"
    return sanitized


def _validate_owned_media_storage_path(*, user_id: str | int, media_id: int, storage_path: str) -> None:
    normalized = storage_path.replace("\\", "/").strip("/")
    parts = normalized.split("/")
    if len(parts) < 4:
        raise FileArtifactsValidationError("reference_image_invalid")
    if any(part in {"", ".", ".."} for part in parts):
        raise FileArtifactsValidationError("reference_image_invalid")
    expected_user = _sanitize_path_component(str(user_id))
    expected_prefix = (expected_user, "media", str(media_id))
    if tuple(parts[:3]) != expected_prefix:
        raise FileArtifactsValidationError("reference_image_invalid")
    if len(parts) <= 3:
        raise FileArtifactsValidationError("reference_image_invalid")


async def _load_durable_bytes(
    storage: StorageBackend,
    *,
    user_id: str | int,
    media_id: int,
    storage_path: str,
    max_bytes: int | None = None,
) -> bytes:
    _validate_owned_media_storage_path(user_id=user_id, media_id=media_id, storage_path=storage_path)
    try:
        if not await storage.exists(storage_path):
            raise FileArtifactsValidationError("reference_image_invalid")
        handle = await storage.retrieve(storage_path)
    except FileArtifactsValidationError:
        raise
    except FileNotFoundError as exc:
        raise FileArtifactsValidationError("reference_image_invalid") from exc
    except StorageError as exc:
        raise ReferenceImageOperationalError("reference_image_storage_unavailable") from exc

    try:
        content = handle.read()
    except (OSError, StorageError) as exc:
        raise ReferenceImageOperationalError("reference_image_storage_unavailable") from exc
    except Exception as exc:
        raise FileArtifactsValidationError("reference_image_invalid") from exc
    if not isinstance(content, bytes) or not content:
        raise FileArtifactsValidationError("reference_image_invalid")
    if max_bytes is not None and len(content) > max_bytes:
        raise FileArtifactsValidationError("reference_image_invalid")
    return content


async def resolve_reference_image(
    media_db: MediaDbLike,
    *,
    user_id: str | int,
    file_id: int,
    storage: StorageBackend | None = None,
) -> ResolvedReferenceImage:
    """Resolve a durable managed reference image into the shared contract."""

    repo = ManagedReferenceImageRepository(media_db, user_id=user_id)
    row = repo.get_candidate_row(file_id)
    max_bytes = _reference_image_max_bytes()
    if _is_known_oversized(row, max_bytes):
        raise FileArtifactsValidationError("reference_image_invalid")
    storage_backend = storage or get_storage_backend()
    content = await _load_durable_bytes(
        storage_backend,
        user_id=user_id,
        media_id=row.media_id,
        storage_path=row.storage_path,
        max_bytes=max_bytes,
    )
    width, height = _probe_dimensions(content)
    return ResolvedReferenceImage(
        file_id=row.file_id,
        filename=row.filename,
        mime_type=row.mime_type,
        width=width,
        height=height,
        bytes_len=len(content),
        content=content,
        temp_path=None,
    )


async def list_reference_image_candidates(
    media_db: MediaDbLike,
    *,
    user_id: str | int,
    limit: int = 100,
    storage: StorageBackend | None = None,
) -> list[ReferenceImageCandidate]:
    """List picker-safe durable managed image candidates."""

    repo = ManagedReferenceImageRepository(media_db, user_id=user_id)
    storage_backend = storage or get_storage_backend()
    max_bytes = _reference_image_max_bytes()
    items: list[ReferenceImageCandidate] = []
    offset = 0
    while len(items) < limit:
        batch_size = min(_REFERENCE_IMAGE_PAGE_SIZE, max(limit - len(items), 1))
        rows = repo.list_candidate_rows(limit=batch_size, offset=offset)
        if not rows:
            break
        for row in rows:
            if _is_known_oversized(row, max_bytes):
                continue
            try:
                content = await _load_durable_bytes(
                    storage_backend,
                    user_id=user_id,
                    media_id=row.media_id,
                    storage_path=row.storage_path,
                    max_bytes=max_bytes,
                )
                width, height = _probe_dimensions(content)
            except FileArtifactsValidationError:
                continue
            items.append(
                ReferenceImageCandidate(
                    file_id=row.file_id,
                    title=row.title,
                    mime_type=row.mime_type,
                    width=width,
                    height=height,
                    created_at=row.created_at,
                )
            )
            if len(items) >= limit:
                break
        offset += len(rows)
    return items
