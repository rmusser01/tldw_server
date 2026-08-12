"""Local filesystem blob storage for Sync v2 M2."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from loguru import logger

from tldw_Server_API.app.core.Infrastructure.distributed_lock import (
    LockAcquisitionError,
    _acquire_platform_file_lock,
    _release_platform_file_lock,
)
from tldw_Server_API.app.core.Utils.path_utils import safe_join

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
        temp_path = _chunk_temp_path(target, chunk_index=chunk_index)
        try:
            temp_path.write_bytes(payload)
            os.link(temp_path, target)
        except FileExistsError:
            if _sha256_file(target) != expected_hash:
                raise SyncBlobStoreError(
                    "Sync blob chunk storage key already contains different content"
                ) from None
        except OSError as exc:
            raise SyncBlobStoreError("Sync blob chunk write failed") from exc
        finally:
            _discard_temp_path(temp_path)
        return storage_key

    def commit_upload(
        self,
        *,
        upload_id: str,
        payload_hash: str,
        chunk_indexes: list[int],
        storage_namespace_id: str | None = None,
    ) -> str:
        """Assemble verified chunks into a committed blob path atomically."""

        digest = _hash_digest(payload_hash)
        upload_segment = _safe_segment(upload_id, field_name="upload_id")
        final_key = (
            self.legacy_storage_key(payload_hash)
            if storage_namespace_id is None
            else self.namespace_storage_key(storage_namespace_id, payload_hash)
        )
        final_path = self.resolve_storage_key(final_key)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = _commit_temp_path(final_path, digest=digest, upload_segment=upload_segment)
        hasher = hashlib.sha256()
        try:
            with temp_path.open("wb") as output:
                for chunk_index in chunk_indexes:
                    chunk_path = self.resolve_storage_key(
                        f"_uploads/{upload_segment}/{chunk_index}.part"
                    )
                    # lgtm[py/path-injection]: chunk_path is constrained by resolve_storage_key.
                    if not chunk_path.exists():
                        raise SyncBlobStoreError(f"Missing upload chunk: {chunk_index}")
                    # lgtm[py/path-injection]: chunk_path is constrained by resolve_storage_key.
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
        return final_key

    @staticmethod
    def legacy_storage_key(payload_hash: str) -> str:
        """Return the exact historical global content-addressed storage key."""

        digest = _hash_digest(payload_hash)
        return f"blobs/sha256/{digest[:2]}/{digest}.blob"

    @staticmethod
    def namespace_storage_key(
        storage_namespace_id: str,
        payload_hash: str,
    ) -> str:
        """Return a v2 path composed only from an opaque namespace and digest."""

        namespace = _storage_namespace_id(storage_namespace_id)
        digest = _hash_digest(payload_hash)
        return f"blobs/v2/{namespace}/{digest}.blob"

    def verify_blob(
        self,
        storage_key: str,
        *,
        payload_hash: str,
        expected_size: int,
    ) -> None:
        """Verify one contained blob's exact logical size and lowercase digest."""

        if isinstance(expected_size, bool) or expected_size < 1:
            raise SyncBlobStoreError("expected_size must be positive")
        _hash_digest(payload_hash)
        try:
            _require_secure_relocation_capabilities()
            with _open_contained_regular_file(
                self.root,
                storage_key,
                error_message="Sync blob verification failed",
                recheck_name=True,
            ) as handle:
                _verify_open_blob(
                    handle,
                    payload_hash=payload_hash,
                    expected_size=expected_size,
                )
        except SyncBlobStoreError:
            raise
        except OSError as exc:
            raise SyncBlobStoreError("Sync blob verification failed") from exc

    def relocate_legacy_blob(
        self,
        *,
        legacy_storage_key: str,
        storage_namespace_id: str,
        payload_hash: str,
        expected_size: int,
    ) -> str:
        """Verify/copy/reverify a global key without ever unlinking the source."""

        canonical_legacy_key = self.legacy_storage_key(payload_hash)
        if legacy_storage_key != canonical_legacy_key:
            raise SyncBlobStoreError("Sync legacy blob storage key is not canonical")
        target_key = self.namespace_storage_key(storage_namespace_id, payload_hash)
        digest = _hash_digest(payload_hash)
        namespace = _storage_namespace_id(storage_namespace_id)
        try:
            _require_secure_relocation_capabilities()
            with _open_directory_chain(
                self.root,
                ("_locks", "legacy-relocation"),
                create=True,
            ) as lock_directory:
                with _lock_file_at(
                    lock_directory,
                    f"{namespace}.{digest}.lock",
                    timeout=10,
                ):
                    try:
                        with _open_contained_regular_file(
                            self.root,
                            canonical_legacy_key,
                            error_message=(
                                "Sync legacy blob relocation source failed verification"
                            ),
                        ) as source_handle:
                            try:
                                _verify_open_blob(
                                    source_handle,
                                    payload_hash=payload_hash,
                                    expected_size=expected_size,
                                )
                            except SyncBlobStoreError as exc:
                                raise SyncBlobStoreError(
                                    "Sync legacy blob relocation source failed verification"
                                ) from exc
                            source_handle.seek(0)
                            with _open_directory_chain(
                                self.root,
                                ("blobs", "v2", namespace),
                                create=True,
                            ) as target_directory:
                                _relocate_open_blob(
                                    source_handle,
                                    target_directory=target_directory,
                                    target_name=f"{digest}.blob",
                                    digest=digest,
                                    namespace=namespace,
                                    payload_hash=payload_hash,
                                    expected_size=expected_size,
                                )
                    except SyncBlobStoreError:
                        raise
                    except OSError as exc:
                        raise SyncBlobStoreError(
                            "Sync legacy blob relocation failed"
                        ) from exc
                    return target_key
        except NotImplementedError as exc:
            raise SyncBlobStoreError(
                "Sync legacy blob relocation has an unsupported platform"
            ) from exc
        except LockAcquisitionError as exc:
            raise SyncBlobStoreError("Sync legacy blob relocation lock is unavailable") from exc

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
        # lgtm[py/path-injection]: upload_segment is sanitized and resolve_storage_key enforces root containment.
        shutil.rmtree(
            self.resolve_storage_key(f"_uploads/{upload_segment}"),
            ignore_errors=True,
        )

    def resolve_storage_key(self, storage_key: str) -> Path:
        """Return a contained filesystem path for a relative storage key."""

        if not storage_key or Path(storage_key).is_absolute():
            raise SyncBlobStoreError("storage_key must be relative")
        target = safe_join(
            str(self.root),
            storage_key,
            error_factory=lambda _exc: SyncBlobStoreError("storage_key escapes blob store root"),
        )
        return Path(target)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file without loading it all into memory."""

    hasher = hashlib.sha256()
    with _open_regular_file_no_follow(
        path,
        error_message="Sync blob path could not be opened safely",
    ) as handle:
        while chunk := handle.read(STREAM_CHUNK_SIZE):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _open_regular_file_no_follow(path: Path, *, error_message: str) -> BinaryIO:
    """Open a regular file without following a last-component symlink race."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SyncBlobStoreError(error_message) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SyncBlobStoreError(error_message)
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _require_secure_relocation_capabilities() -> None:
    """Reject platforms lacking the descriptor-relative primitives we rely on."""

    supported_dir_fd = {
        getattr(function, "__name__", "")
        for function in getattr(os, "supports_dir_fd", set())
    }
    supported_follow = {
        getattr(function, "__name__", "")
        for function in getattr(os, "supports_follow_symlinks", set())
    }
    if (
        not getattr(os, "O_DIRECTORY", 0)
        or not getattr(os, "O_NOFOLLOW", 0)
        or not {"open", "stat", "mkdir", "unlink", "link"}.issubset(
            supported_dir_fd
        )
        or "link" not in supported_follow
    ):
        raise SyncBlobStoreError(
            "Sync legacy blob relocation has an unsupported platform"
        )


@contextmanager
def _open_directory_chain(
    root: Path,
    segments: tuple[str, ...],
    *,
    create: bool,
) -> Iterator[int]:
    """Open a directory chain through anchored, no-follow descriptors."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        current = os.open(root, flags)
    except OSError as exc:
        raise SyncBlobStoreError("Sync blob root could not be opened safely") from exc
    try:
        for raw_segment in segments:
            segment = _safe_segment(raw_segment, field_name="storage path segment")
            try:
                expected = os.stat(segment, dir_fd=current, follow_symlinks=False)
                child = os.open(segment, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise SyncBlobStoreError(
                        "Sync blob directory path does not exist"
                    ) from None
                try:
                    os.mkdir(segment, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise SyncBlobStoreError(
                        "Sync blob directory could not be created safely"
                    ) from exc
                try:
                    expected = os.stat(
                        segment,
                        dir_fd=current,
                        follow_symlinks=False,
                    )
                    child = os.open(segment, flags, dir_fd=current)
                except OSError as exc:
                    raise SyncBlobStoreError(
                        "Sync blob directory could not be opened safely"
                    ) from exc
            except OSError as exc:
                raise SyncBlobStoreError(
                    "Sync blob directory could not be opened safely"
                ) from exc
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(expected.st_mode)
                or opened.st_dev != expected.st_dev
                or opened.st_ino != expected.st_ino
            ):
                os.close(child)
                raise SyncBlobStoreError("Sync blob directory identity changed")
            os.close(current)
            current = child
        yield current
    finally:
        os.close(current)


def _open_lock_file_at(directory: int, name: str) -> int:
    """Open a stable regular lock file relative to an anchored directory."""

    lock_name = _safe_segment(name, field_name="lock file name")
    flags = (
        os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _attempt in range(16):
        try:
            expected = os.stat(lock_name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            try:
                descriptor = os.open(
                    lock_name,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise SyncBlobStoreError(
                    "Sync legacy blob relocation lock file could not be created safely"
                ) from exc
            try:
                expected = os.fstat(descriptor)
            except OSError as exc:
                os.close(descriptor)
                raise SyncBlobStoreError(
                    "Sync legacy blob relocation lock file could not be inspected safely"
                ) from exc
        except OSError as exc:
            raise SyncBlobStoreError(
                "Sync legacy blob relocation lock file could not be inspected safely"
            ) from exc
        else:
            try:
                descriptor = os.open(lock_name, flags, dir_fd=directory)
            except OSError as exc:
                raise SyncBlobStoreError(
                    "Sync legacy blob relocation lock file could not be opened safely"
                ) from exc

        try:
            opened = os.fstat(descriptor)
            named = os.stat(lock_name, dir_fd=directory, follow_symlinks=False)
            if (
                not stat.S_ISREG(expected.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(named.st_mode)
                or opened.st_dev != expected.st_dev
                or opened.st_ino != expected.st_ino
                or named.st_dev != opened.st_dev
                or named.st_ino != opened.st_ino
            ):
                raise SyncBlobStoreError(
                    "Sync legacy blob relocation lock file identity changed"
                )
            return descriptor
        except Exception:
            os.close(descriptor)
            raise
    raise SyncBlobStoreError(
        "Sync legacy blob relocation lock file could not be created safely"
    )


@contextmanager
def _lock_file_at(directory: int, name: str, *, timeout: float) -> Iterator[None]:
    """Lock a no-follow file descriptor without reopening its pathname."""

    descriptor = _open_lock_file_at(directory, name)
    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while True:
            try:
                _acquire_platform_file_lock(descriptor)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LockAcquisitionError(
                        f"Failed to acquire sync relocation lock within {timeout}s"
                    ) from None
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        yield
    finally:
        if acquired:
            try:
                _release_platform_file_lock(descriptor)
            except OSError:
                pass
        os.close(descriptor)


@contextmanager
def _open_contained_regular_file(
    root: Path,
    storage_key: str,
    *,
    error_message: str,
    recheck_name: bool = False,
) -> Iterator[BinaryIO]:
    """Open one contained regular file without following any path symlink."""

    path = Path(storage_key)
    if not storage_key or path.is_absolute() or len(path.parts) < 2:
        raise SyncBlobStoreError(error_message)
    segments = tuple(
        _safe_segment(part, field_name="storage path segment") for part in path.parts
    )
    with _open_directory_chain(root, segments[:-1], create=False) as directory:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(segments[-1], flags, dir_fd=directory)
        except OSError as exc:
            raise SyncBlobStoreError(error_message) from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise SyncBlobStoreError(error_message)
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                yield handle
                if recheck_name:
                    try:
                        named = os.stat(
                            segments[-1],
                            dir_fd=directory,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise SyncBlobStoreError(
                            "Sync blob path identity changed"
                        ) from exc
                    if not stat.S_ISREG(named.st_mode) or not _same_inode(
                        opened, named
                    ):
                        raise SyncBlobStoreError("Sync blob path identity changed")
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _verify_open_blob(
    handle: BinaryIO,
    *,
    payload_hash: str,
    expected_size: int,
) -> None:
    """Verify size and digest through an already anchored regular-file handle."""

    if isinstance(expected_size, bool) or expected_size < 1:
        raise SyncBlobStoreError("expected_size must be positive")
    _hash_digest(payload_hash)
    try:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
            raise SyncBlobStoreError("Sync blob size does not match expected bytes")
        handle.seek(0)
        hasher = hashlib.sha256()
        while chunk := handle.read(STREAM_CHUNK_SIZE):
            hasher.update(chunk)
        if "sha256:" + hasher.hexdigest() != payload_hash:
            raise SyncBlobStoreError("Sync blob digest does not match expected bytes")
        handle.seek(0)
    except SyncBlobStoreError:
        raise
    except (OSError, ValueError) as exc:
        raise SyncBlobStoreError("Sync blob verification failed") from exc


def _open_target_at(directory: int, name: str) -> BinaryIO:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SyncBlobStoreError("Sync relocated blob target is not a regular file")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _verify_named_open_blob(
    handle: BinaryIO,
    *,
    directory: int,
    name: str,
    payload_hash: str,
    expected_size: int,
    expected_inode: os.stat_result | None = None,
) -> os.stat_result:
    """Verify bytes and then prove the directory name still names this inode."""

    opened = os.fstat(handle.fileno())
    if expected_inode is not None and not _same_inode(opened, expected_inode):
        raise SyncBlobStoreError("Sync relocated blob target inode changed")
    _verify_open_blob(
        handle,
        payload_hash=payload_hash,
        expected_size=expected_size,
    )
    try:
        named = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except OSError as exc:
        raise SyncBlobStoreError("Sync relocated blob target identity changed") from exc
    if not stat.S_ISREG(named.st_mode) or not _same_inode(opened, named):
        raise SyncBlobStoreError("Sync relocated blob target identity changed")
    return opened


def _relocate_open_blob(
    source: BinaryIO,
    *,
    target_directory: int,
    target_name: str,
    digest: str,
    namespace: str,
    payload_hash: str,
    expected_size: int,
) -> None:
    """Publish one verified blob through directory-relative, no-follow handles."""

    expected_digest = _hash_digest(payload_hash)
    if target_name != f"{expected_digest}.blob" or digest != expected_digest:
        raise SyncBlobStoreError("Sync relocated blob target identity is invalid")
    _storage_namespace_id(namespace)
    try:
        existing = _open_target_at(target_directory, target_name)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise SyncBlobStoreError("Sync relocated blob target could not be opened safely") from exc
    if existing is not None:
        with existing:
            _verify_named_open_blob(
                existing,
                directory=target_directory,
                name=target_name,
                payload_hash=payload_hash,
                expected_size=expected_size,
            )
        return

    temp_name = f".{digest}.{namespace}.{secrets.token_hex(8)}.relocating"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=target_directory)
    except OSError as exc:
        raise SyncBlobStoreError("Sync relocation temporary file could not be created") from exc

    expected = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "r+b", closefd=False) as temp:
            source.seek(0)
            while chunk := source.read(STREAM_CHUNK_SIZE):
                temp.write(chunk)
            temp.flush()
            os.fsync(descriptor)
            _verify_open_blob(
                temp,
                payload_hash=payload_hash,
                expected_size=expected_size,
            )

        named = os.stat(temp_name, dir_fd=target_directory, follow_symlinks=False)
        if (
            named.st_dev != expected.st_dev
            or named.st_ino != expected.st_ino
            or not stat.S_ISREG(named.st_mode)
        ):
            raise SyncBlobStoreError("Sync relocation temporary inode changed")
        try:
            os.link(
                temp_name,
                target_name,
                src_dir_fd=target_directory,
                dst_dir_fd=target_directory,
                follow_symlinks=False,
            )
        except FileExistsError:
            with _open_target_at(target_directory, target_name) as existing:
                _verify_named_open_blob(
                    existing,
                    directory=target_directory,
                    name=target_name,
                    payload_hash=payload_hash,
                    expected_size=expected_size,
                )
            return
        with _open_target_at(target_directory, target_name) as target:
            _verify_named_open_blob(
                target,
                directory=target_directory,
                name=target_name,
                payload_hash=payload_hash,
                expected_size=expected_size,
                expected_inode=expected,
            )
    except SyncBlobStoreError:
        raise
    except OSError as exc:
        raise SyncBlobStoreError("Sync legacy blob relocation failed") from exc
    finally:
        os.close(descriptor)


def _hash_digest(value: str) -> str:
    prefix = "sha256:"
    if not value.startswith(prefix) or len(value) != len(prefix) + 64:
        raise SyncBlobStoreError("payload_hash must be sha256:<64 hex chars>")
    digest = value[len(prefix) :]
    if any(character not in "0123456789abcdef" for character in digest):
        raise SyncBlobStoreError("payload_hash digest must be lowercase hex")
    return digest


def _storage_namespace_id(value: str) -> str:
    if len(value) != 32 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SyncBlobStoreError(
            "storage_namespace_id must be 32 lowercase hexadecimal characters"
        )
    return value


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


def _commit_temp_path(final_path: Path, *, digest: str, upload_segment: str) -> Path:
    if len(digest) != 64:
        raise SyncBlobStoreError("digest must be 64 hex chars")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise SyncBlobStoreError("digest must be hex") from exc
    safe_upload_segment = _safe_segment(upload_segment, field_name="upload_segment")
    # lgtm[py/path-injection]: final_path is constrained by resolve_storage_key.
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=final_path.parent,
        # lgtm[py/path-injection]: digest and upload_segment are validated before temp path creation.
        prefix=f"{digest}.{safe_upload_segment}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    return temp_path


def _chunk_temp_path(target: Path, *, chunk_index: int) -> Path:
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=target.parent,
        prefix=f"{chunk_index}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    return temp_path


__all__ = ["LocalSyncBlobStore", "STREAM_CHUNK_SIZE", "SyncBlobStoreError"]
