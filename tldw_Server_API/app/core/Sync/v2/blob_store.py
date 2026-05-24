"""Local filesystem blob storage for Sync v2 M2."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

STREAM_CHUNK_SIZE = 64 * 1024


class SyncBlobStoreError(ValueError):
    """Raised when blob storage input or integrity checks fail."""


class LocalSyncBlobStore:
    """Path-contained local blob store rooted under a user's sync blob directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_upload_chunk(
        self,
        *,
        upload_id: str,
        chunk_index: int,
        payload: bytes,
        expected_hash: str,
    ) -> str:
        """Write one verified upload chunk and return its relative storage key."""

        actual_hash = _sha256(payload)
        if actual_hash != expected_hash:
            raise SyncBlobStoreError("Sync blob chunk hash does not match payload")
        upload_segment = _safe_segment(upload_id, field_name="upload_id")
        if chunk_index < 0:
            raise SyncBlobStoreError("chunk_index must be non-negative")
        storage_key = f"_uploads/{upload_segment}/{chunk_index}.part"
        target = self.resolve_storage_key(storage_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_suffix(target.suffix + ".tmp")
        temp_path.write_bytes(payload)
        temp_path.replace(target)
        return storage_key

    def commit_upload(
        self,
        *,
        upload_id: str,
        payload_hash: str,
        chunk_indexes: list[int],
    ) -> str:
        """Assemble verified chunks into a committed blob path atomically."""

        digest = _hash_digest(payload_hash)
        upload_segment = _safe_segment(upload_id, field_name="upload_id")
        final_key = f"blobs/sha256/{digest[:2]}/{digest}.blob"
        final_path = self.resolve_storage_key(final_key)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        hasher = hashlib.sha256()
        try:
            with temp_path.open("wb") as output:
                for chunk_index in chunk_indexes:
                    chunk_path = self.resolve_storage_key(
                        f"_uploads/{upload_segment}/{chunk_index}.part"
                    )
                    if not chunk_path.exists():
                        raise SyncBlobStoreError(f"Missing upload chunk: {chunk_index}")
                    with chunk_path.open("rb") as chunk_file:
                        while chunk := chunk_file.read(STREAM_CHUNK_SIZE):
                            hasher.update(chunk)
                            output.write(chunk)
            actual_hash = "sha256:" + hasher.hexdigest()
            if actual_hash != payload_hash:
                raise SyncBlobStoreError("Sync blob payload hash does not match chunks")
        except SyncBlobStoreError:
            _discard_temp_path(temp_path)
            raise
        except OSError as exc:
            _discard_temp_path(temp_path)
            raise SyncBlobStoreError("Sync blob upload commit failed") from exc
        try:
            temp_path.replace(final_path)
        except OSError as exc:
            _discard_temp_path(temp_path)
            raise SyncBlobStoreError("Sync blob upload commit failed") from exc
        try:
            shutil.rmtree(self.resolve_storage_key(f"_uploads/{upload_segment}"))
        except OSError as exc:
            logger.warning(
                "Sync blob upload committed but cleanup failed for upload_id={}: {}",
                upload_id,
                exc,
            )
        return final_key

    def read_blob(self, storage_key: str) -> bytes:
        """Read a committed blob by storage key after path-containment checks."""

        return b"".join(self.iter_blob(storage_key))

    def iter_blob(
        self,
        storage_key: str,
        *,
        offset: int = 0,
        size: int | None = None,
        chunk_size: int = STREAM_CHUNK_SIZE,
    ) -> Iterator[bytes]:
        """Yield a committed blob or byte range in bounded chunks."""

        if offset < 0:
            raise SyncBlobStoreError("offset must be non-negative")
        if size is not None and size < 0:
            raise SyncBlobStoreError("size must be non-negative")
        if chunk_size < 1:
            raise SyncBlobStoreError("chunk_size must be positive")

        remaining = size
        with self.resolve_storage_key(storage_key).open("rb") as blob_file:
            if offset:
                blob_file.seek(offset)
            while remaining is None or remaining > 0:
                read_size = chunk_size if remaining is None else min(chunk_size, remaining)
                chunk = blob_file.read(read_size)
                if not chunk:
                    break
                yield chunk
                if remaining is not None:
                    remaining -= len(chunk)

    def blob_size(self, storage_key: str) -> int:
        """Return the size of a committed blob after path-containment checks."""

        return self.resolve_storage_key(storage_key).stat().st_size

    def discard_upload(self, upload_id: str) -> None:
        """Remove staged chunks for an upload if present."""

        upload_segment = _safe_segment(upload_id, field_name="upload_id")
        shutil.rmtree(
            self.resolve_storage_key(f"_uploads/{upload_segment}"),
            ignore_errors=True,
        )

    def resolve_storage_key(self, storage_key: str) -> Path:
        """Return a contained filesystem path for a relative storage key."""

        if not storage_key or Path(storage_key).is_absolute():
            raise SyncBlobStoreError("storage_key must be relative")
        target = (self.root / storage_key).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise SyncBlobStoreError("storage_key escapes blob store root") from exc
        return target


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _hash_digest(value: str) -> str:
    prefix = "sha256:"
    if not value.startswith(prefix) or len(value) != len(prefix) + 64:
        raise SyncBlobStoreError("payload_hash must be sha256:<64 hex chars>")
    digest = value[len(prefix) :]
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise SyncBlobStoreError("payload_hash digest must be hex") from exc
    return digest


def _safe_segment(value: str, *, field_name: str) -> str:
    if not value or any(char in value for char in {"/", "\\", "\x00"}):
        raise SyncBlobStoreError(f"{field_name} contains unsafe path characters")
    if value in {".", ".."}:
        raise SyncBlobStoreError(f"{field_name} contains unsafe path characters")
    return value


def _discard_temp_path(temp_path: Path) -> None:
    try:
        temp_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to remove temporary Sync blob path {}: {}", temp_path, exc)


__all__ = ["LocalSyncBlobStore", "STREAM_CHUNK_SIZE", "SyncBlobStoreError"]
