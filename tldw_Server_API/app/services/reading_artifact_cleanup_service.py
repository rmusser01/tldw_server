"""Storage identity and exclusion prerequisites for durable Reading cleanup.

These primitives are not wired to archive creation, unlink, or capability
readiness yet. Callers must supply an explicitly provisioned per-user output
root, take the storage lock BEFORE any database transaction, and keep it through
file I/O and the final adoption/cleanup transaction. No lease-only fallback is
safe. Shared/network volumes still require deployment-level lock verification.
"""

from __future__ import annotations

import os
import stat
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import normalize_output_storage_filename
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError

try:
    import fcntl
except ImportError:  # Unsupported platforms must remain unavailable.
    fcntl = None

_MARKER_NAME = ".reading-storage-namespace"
_LOCK_NAME = ".reading-storage.lock"


class ReadingStorageUnavailable(RuntimeError):
    """The expected output volume or its exclusion mechanism is unavailable."""


class ReadingStorageBusy(RuntimeError):
    """Another process holds the storage lock; retry without changing files."""


def _open_private_file(directory: int, name: str, flags: int) -> int:
    """Open a single-link regular file without following a symlink or blocking."""
    fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ReadingStorageUnavailable("reading_storage_unavailable")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_namespace(directory: int, *, durable: bool = False) -> str:
    fd = _open_private_file(directory, _MARKER_NAME, os.O_RDWR if durable else os.O_RDONLY)
    try:
        raw = os.read(fd, 34)
        # Bounded exact representation; malformed/partial provisioning fails closed.
        if len(raw) != 33 or raw[-1:] != b"\n" or any(c not in b"0123456789abcdef" for c in raw[:32]):
            raise ReadingStorageUnavailable("reading_storage_unavailable")
        if durable:
            # A complete marker can survive a failed sync in a prior attempt.
            os.fsync(fd)
            os.fsync(directory)
        return raw[:32].decode("ascii")
    finally:
        os.close(fd)


@contextmanager
def _locked_directory(output_root: Path, *, provision: bool = False) -> Iterator[tuple[Path, int]]:
    """Acquire nonblocking OS exclusion and verify the named directory/lock inode."""
    with ExitStack() as resources:
        try:
            if fcntl is None or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
                raise ReadingStorageUnavailable("reading_storage_unavailable")
            root = Path(output_root).absolute()
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            resources.callback(os.close, directory)
            flags = os.O_RDWR
            if provision:
                try:
                    os.stat(_MARKER_NAME, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    # Only explicit first-time provisioning may create the stable lock.
                    flags |= os.O_CREAT
            lock = _open_private_file(directory, _LOCK_NAME, flags)
            resources.callback(os.close, lock)  # close releases flock even after exceptions.
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ReadingStorageBusy("reading_storage_busy") from None
            for actual, expected in (
                (os.stat(root, follow_symlinks=False), os.fstat(directory)),
                (os.stat(_LOCK_NAME, dir_fd=directory, follow_symlinks=False), os.fstat(lock)),
            ):
                if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                    raise ReadingStorageUnavailable("reading_storage_unavailable")
        except (OSError, TypeError, ValueError):
            raise ReadingStorageUnavailable("reading_storage_unavailable") from None
        # Do not translate exceptions from the caller's DB/file operation.
        yield root, directory


def provision_reading_storage_namespace(output_root: Path) -> str:
    """Explicitly initialize an existing output root; never rotate its identity.

    This is an operator/upgrade primitive, not automatic runtime recovery.
    Missing roots are not created. A marker with a missing lock is rejected.
    Interrupted marker creation stays invalid for explicit operator repair.
    """
    try:
        with _locked_directory(output_root, provision=True) as (_, directory):
            try:
                return _read_namespace(directory, durable=True)
            except FileNotFoundError:
                namespace = uuid.uuid4().hex
                fd = _open_private_file(directory, _MARKER_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                try:
                    data = (namespace + "\n").encode("ascii")
                    if os.write(fd, data) != len(data):
                        raise ReadingStorageUnavailable("reading_storage_unavailable")
                    os.fsync(fd)
                    os.fsync(directory)
                finally:
                    os.close(fd)
                return namespace
    except (OSError, TypeError, ValueError):
        raise ReadingStorageUnavailable("reading_storage_unavailable") from None


@contextmanager
def _validated_storage_directory(output_root: Path, *, storage_namespace_id: str) -> Iterator[tuple[Path, int]]:
    """Hold the persistent per-user lock only on the expected mounted namespace.

    Busy is retryable. Missing/invalid/replaced state is unavailable, never proof
    that an archive is absent. Runtime access creates or repairs no files.
    """
    with _locked_directory(output_root) as (root, directory):
        try:
            if _read_namespace(directory) != storage_namespace_id:
                raise ReadingStorageUnavailable("reading_storage_unavailable")
        except (OSError, TypeError, ValueError):
            raise ReadingStorageUnavailable("reading_storage_unavailable") from None
        yield root, directory


@contextmanager
def reading_storage_lock(output_root: Path, *, storage_namespace_id: str) -> Iterator[Path]:
    """Hold verified storage exclusion; internal file I/O must use the held directory FD."""
    with _validated_storage_directory(output_root, storage_namespace_id=storage_namespace_id) as (root, _):
        yield root


def _artifact_filename(name: str) -> str:
    filename = normalize_output_storage_filename(
        name,
        allow_absolute=False,
        reject_relative_with_separators=True,
    )
    if filename.lower() in {_MARKER_NAME, _LOCK_NAME}:
        raise InvalidStoragePathError("invalid_path")
    return filename


def _sync_directory(directory: int) -> None:
    os.fsync(directory)


def write_staged_reading_artifact(
    db: CollectionsDatabase, token: str, *, output_root: Path, storage_namespace_id: str, body: str
) -> None:
    """Write a reserved archive; adoption remains a separate revision-guarded step.

    The trusted caller supplies already-bounded rendered content. This helper is
    not yet wired into production capture paths. Expired/retired tokens abort
    before opening a file; any partially written file retains its durable intent.
    """
    with _validated_storage_directory(output_root, storage_namespace_id=storage_namespace_id) as (_, directory):
        _write_staged_artifact(db, token, storage_namespace_id, directory, body)


def _write_staged_artifact(db: CollectionsDatabase, token: str, namespace: str, directory: int, body: str) -> None:
    """Write only under the caller's existing verified storage lock."""
    row = db.validate_reading_artifact_for_write(token, namespace, now=int(time.time()))
    try:
        filename = _artifact_filename(row["storage_path"])
        fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(directory)
    except (OSError, InvalidStoragePathError) as exc:
        category = (
            "invalid_path"
            if isinstance(exc, InvalidStoragePathError)
            else ("path_collision" if isinstance(exc, FileExistsError) else "io")
        )
        db.record_reading_artifact_error(token, namespace, category=category, now=int(time.time()))
        raise


def write_and_adopt_reading_artifact(
    db: CollectionsDatabase,
    token: str,
    *,
    output_root: Path,
    storage_namespace_id: str,
    body: str,
    title: str,
    retention_until: str | None = None,
) -> CollectionsDatabase.OutputArtifactRow:
    """Keep storage exclusion through exclusive write, sync and guarded adoption.

    Failed adoption leaves a pending file rather than recreating the parent.
    If scheduling itself fails, the durable staged lease still permits recovery.
    This trusted service is not yet wired into production capture endpoints.
    """
    with _validated_storage_directory(output_root, storage_namespace_id=storage_namespace_id) as (_, directory):
        try:
            _write_staged_artifact(db, token, storage_namespace_id, directory, body)
            return db.adopt_reading_artifact(
                token,
                storage_namespace_id,
                title=title,
                now=int(time.time()),
                retention_until=retention_until,
            )
        except Exception:
            db.cancel_reading_artifact(token, storage_namespace_id)
            raise


def drain_reading_artifact_cleanup(
    db: CollectionsDatabase, *, output_root: Path, storage_namespace_id: str, limit: int = 100
) -> int:
    """Drain a bounded unadopted-artifact batch on a verified volume outside DB locks."""
    completed = 0
    with _validated_storage_directory(output_root, storage_namespace_id=storage_namespace_id) as (_, directory):
        rows = db.prepare_reading_artifact_cleanup(storage_namespace_id, now=int(time.time()), limit=limit)
        for row in rows:
            try:
                filename = _artifact_filename(row["storage_path"])
                try:
                    info = os.stat(filename, dir_fd=directory, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise InvalidStoragePathError("invalid_path")
                    os.unlink(filename, dir_fd=directory)
                except FileNotFoundError:
                    pass
                _sync_directory(directory)
            except (OSError, InvalidStoragePathError) as exc:
                category = (
                    "invalid_path"
                    if isinstance(exc, InvalidStoragePathError)
                    else ("permission" if isinstance(exc, PermissionError) else "io")
                )
                db.record_reading_artifact_error(
                    row["token"], storage_namespace_id, category=category, now=int(time.time())
                )
                continue
            completed += db.finish_reading_artifact_cleanup(row["token"], storage_namespace_id)
    return completed
