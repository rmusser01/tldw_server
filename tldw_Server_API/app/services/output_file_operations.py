"""Reserved, bounded output mutations and recovery; runtime wiring pending.

Every file operation uses the verified directory descriptor. Failed or ambiguous
work retains journal authority until identity-verified cleanup is durable.
Async callers wait for each offloaded lock interval to close its writable FD.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import stat
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import anyio
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.services.reading_artifact_cleanup_service import (
    ReadingStorageBusy,
    ReadingStorageUnavailable,
    _validated_storage_directory,
)

MAX_CHUNK_BYTES = 1024 * 1024


def _identity(info: os.stat_result, *, source: bool = False) -> dict[str, int]:
    identity = {"dev": info.st_dev, "ino": info.st_ino, "mode": stat.S_IFMT(info.st_mode), "nlink": info.st_nlink}
    if source:
        identity.update(size=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)
    return identity


@contextmanager
def _open_regular(directory: int, name: str, flags: int):
    fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("output_source_unavailable")
        yield fd
    finally:
        os.close(fd)


def _source_identity(directory: int, name: str) -> dict[str, int]:
    try:
        with _open_regular(directory, name, os.O_RDONLY) as fd:
            return _identity(os.fstat(fd), source=True)
    except OSError:
        raise RuntimeError("output_source_unavailable") from None


def _check_space(directory: int, required: int, margin: int) -> None:
    info = os.fstatvfs(directory)
    if required > info.f_bavail * info.f_frsize - margin:
        raise RuntimeError("output_storage_capacity")


class _UnprovedOutputIdentity(RuntimeError):
    """Files must be preserved for explicit operator verification."""


def _stat_optional(directory: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _require_identity(info: os.stat_result, identity: dict | None, *, size: int) -> None:
    if identity is None or _identity(info) != identity or info.st_size != size:
        raise _UnprovedOutputIdentity("output_identity_unconfirmed")


async def _wait_worker(function):
    """Drain the worker even after repeated direct asyncio cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function))
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise


class OutputFileOperations:
    """Internal file mutations on one bound volume, never an activation API."""

    def __init__(self, db: CollectionsDatabase, *, output_root: Path, storage_namespace_id: str) -> None:
        self.db = db
        self.output_root = output_root
        self.namespace = storage_namespace_id

    def _abort(self, token: str) -> None:
        with _validated_storage_directory(self.output_root, storage_namespace_id=self.namespace):
            self.db.abort_output_file_operation(token, self.namespace)

    async def _run_interval(self, function, *, token: str | None = None):
        # Shield AnyIO level cancellation too; direct Task.cancel is drained below.
        result = None
        try:
            with anyio.CancelScope(shield=True):
                result, cancelled = await _wait_worker(function)
            if cancelled:
                raise asyncio.CancelledError
            await anyio.lowlevel.checkpoint_if_cancelled()
            return result
        except asyncio.CancelledError:
            try:
                if token is not None or result is not None:
                    with anyio.CancelScope(shield=True):
                        await _wait_worker(partial(self._abort, token or result["token"]))
            finally:
                raise asyncio.CancelledError from None
        except OSError as exc:
            code = "output_storage_capacity" if exc.errno == errno.ENOSPC else "output_storage_unavailable"
            raise RuntimeError(code) from None

    async def prepare(
        self,
        *,
        kind: str,
        max_output_bytes: int,
        output_id: int | None = None,
        destination_path: str | None = None,
        intended: dict | None = None,
        lease_seconds: int = 60,
    ) -> dict:
        """Reserve budget and identity before exclusive creation of an empty stage."""
        return await self._run_interval(
            partial(
                self._prepare,
                kind=kind,
                max_output_bytes=max_output_bytes,
                output_id=output_id,
                destination_path=destination_path,
                intended=intended,
                lease_seconds=lease_seconds,
            )
        )

    def _prepare(self, *, kind, max_output_bytes, output_id, destination_path, intended, lease_seconds):
        if type(max_output_bytes) is not int or not 0 <= max_output_bytes <= 2**63 - 1:
            raise ValueError("output_operation_invalid")
        if kind not in {"create", "replace", "remove"} or (kind == "remove" and max_output_bytes):
            raise ValueError("output_operation_invalid")
        with _validated_storage_directory(self.output_root, storage_namespace_id=self.namespace) as (_, directory):
            policy = self.db.get_output_storage_policy(self.namespace)
            source, source_name = None, None
            if kind != "create":
                output = self.db.get_output_artifact(output_id, include_deleted=kind == "remove")
                source_name = self.db._output_operation_filename(output.storage_path)
                source = _source_identity(directory, source_name)
            if kind != "remove":
                destination_path = self.db._output_operation_filename(destination_path)
                try:
                    os.stat(destination_path, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise RuntimeError("output_path_conflict")
            _check_space(directory, max_output_bytes, policy["free_space_margin_bytes"])
            row = self.db.prepare_output_file_operation(
                self.namespace,
                kind=kind,
                output_id=output_id,
                destination_path=destination_path,
                intended=intended,
                reserved_bytes=max_output_bytes + (source["size"] if source else 0),
                lease_seconds=lease_seconds,
            )
            token = row["token"]
            try:
                if row["source_path"] != source_name:
                    raise RuntimeError("output_operation_conflict")
                self.db.record_output_file_progress(
                    token,
                    self.namespace,
                    source_identity=source,
                    expected_offset=0,
                    written_bytes=0,
                    lease_seconds=lease_seconds,
                )
                if row["stage_path"] is not None:
                    with _open_regular(directory, row["stage_path"], os.O_WRONLY | os.O_CREAT | os.O_EXCL) as fd:
                        stage = _identity(os.fstat(fd))
                        os.fsync(fd)
                    os.fsync(directory)
                    row = self.db.record_output_file_progress(
                        token,
                        self.namespace,
                        stage_identity=stage,
                        expected_offset=0,
                        written_bytes=0,
                        lease_seconds=lease_seconds,
                    )
                else:
                    row = self.db.get_output_file_operation(token, self.namespace)
                return row
            except BaseException as exc:
                # No file-first rollback: even a pre-identity file keeps its claim.
                self.db.abort_output_file_operation(token, self.namespace)
                if isinstance(exc, OSError):
                    code = "output_storage_capacity" if exc.errno == errno.ENOSPC else "output_storage_unavailable"
                    raise RuntimeError(code) from None
                raise

    async def write_chunk(self, token: str, data: bytes, *, expected_offset: int) -> int:
        """Write/sync at most 1 MiB and close before releasing storage exclusion.

        Byte production happens before this call, outside the lock. Cancellation
        waits for writable descriptors to close, then conditionally aborts. If
        storage/DB are unavailable, the durable lease remains for recovery.
        """
        if type(data) is not bytes or len(data) > MAX_CHUNK_BYTES:
            raise ValueError("output_size_limit")
        if type(expected_offset) is not int or expected_offset < 0:
            raise ValueError("output_operation_invalid")
        return await self._run_interval(partial(self._write_chunk, token, data, expected_offset), token=token)

    def _write_chunk(self, token: str, data: bytes, expected_offset: int) -> int:
        with _validated_storage_directory(self.output_root, storage_namespace_id=self.namespace) as (_, directory):
            row = self.db.validate_output_file_operation(token, self.namespace)
            if row["written_bytes"] != expected_offset or row["stage_identity_json"] is None:
                raise RuntimeError("output_operation_conflict")
            source = json.loads(row["source_identity_json"]) if row["source_identity_json"] else None
            if expected_offset + len(data) > row["reserved_bytes"] - (source["size"] if source else 0):
                raise RuntimeError("output_size_limit")
            policy = self.db.get_output_storage_policy(self.namespace)
            _check_space(directory, len(data), policy["free_space_margin_bytes"])
            try:
                if source is not None and _source_identity(directory, row["source_path"]) != source:
                    raise RuntimeError("output_source_unavailable")
                with _open_regular(directory, row["stage_path"], os.O_WRONLY) as fd:
                    info = os.fstat(fd)
                    if _identity(info) != json.loads(row["stage_identity_json"]) or info.st_size != expected_offset:
                        raise RuntimeError("output_operation_conflict")
                    os.lseek(fd, expected_offset, os.SEEK_SET)
                    view = memoryview(data)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise RuntimeError("output_storage_unavailable")
                        view = view[written:]
                    os.fsync(fd)
                if source is not None and _source_identity(directory, row["source_path"]) != source:
                    raise RuntimeError("output_source_unavailable")
                self.db.record_output_file_progress(
                    token, self.namespace, expected_offset=expected_offset, written_bytes=expected_offset + len(data)
                )
            except BaseException as exc:
                self.db.abort_output_file_operation(token, self.namespace)
                if isinstance(exc, OSError):
                    code = "output_storage_capacity" if exc.errno == errno.ENOSPC else "output_storage_unavailable"
                    raise RuntimeError(code) from None
                raise
            return expected_offset + len(data)

    async def copy_source(self, token: str, *, expected_offset: int = 0) -> int:
        """Copy the recorded source in 1 MiB intervals without publishing it.

        Resume only from the caller's acknowledged offset. Both the read and
        write intervals revalidate authority, and neither holds a descriptor
        across an await. No source or ambiguous stage is truncated.
        """
        if type(expected_offset) is not int or expected_offset < 0:
            raise ValueError("output_operation_invalid")
        offset = expected_offset
        while True:
            data = await self._run_interval(partial(self._read_source_chunk, token, offset), token=token)
            if not data:
                return offset
            offset = await self.write_chunk(token, data, expected_offset=offset)

    def _read_source_chunk(self, token: str, offset: int) -> bytes:
        with _validated_storage_directory(self.output_root, storage_namespace_id=self.namespace) as (_, directory):
            row = self.db.validate_output_file_operation(token, self.namespace)
            if row["kind"] != "replace" or row["written_bytes"] != offset or not row["source_identity_json"]:
                raise RuntimeError("output_operation_conflict")
            source = json.loads(row["source_identity_json"])
            if source["size"] > row["reserved_bytes"] - source["size"]:
                raise RuntimeError("output_size_limit")
            try:
                if not row["stage_identity_json"]:
                    raise RuntimeError("output_operation_conflict")
                with _open_regular(directory, row["stage_path"], os.O_RDONLY) as fd:
                    info = os.fstat(fd)
                    if _identity(info) != json.loads(row["stage_identity_json"]) or info.st_size != offset:
                        raise RuntimeError("output_operation_conflict")
                with _open_regular(directory, row["source_path"], os.O_RDONLY) as fd:
                    if _identity(os.fstat(fd), source=True) != source or offset > source["size"]:
                        raise RuntimeError("output_source_unavailable")
                    os.lseek(fd, offset, os.SEEK_SET)
                    data = os.read(fd, min(MAX_CHUNK_BYTES, source["size"] - offset))
                    if _identity(os.fstat(fd), source=True) != source:
                        raise RuntimeError("output_source_unavailable")
                    if not data and offset != source["size"]:
                        raise RuntimeError("output_source_unavailable")
                if _source_identity(directory, row["source_path"]) != source:
                    raise RuntimeError("output_source_unavailable")
                return data
            except BaseException:
                self.db.abort_output_file_operation(token, self.namespace)
                raise

    async def publish_and_commit(self, token: str) -> CollectionsDatabase.OutputArtifactRow | None:
        """Publish without replacement, then atomically apply the recorded mutation.

        Attempt cleanup under the same exclusion after confirming the commit.
        A successful return denotes logical commit; failed cleanup remains due
        for recovery and must not turn a committed mutation into a failed one.
        """
        return await self._run_interval(partial(self._publish_and_commit, token), token=token)

    def _publish_and_commit(self, token: str) -> CollectionsDatabase.OutputArtifactRow | None:
        with _validated_storage_directory(self.output_root, storage_namespace_id=self.namespace) as (_, directory):
            row = self.db.validate_output_file_operation(token, self.namespace)
            publication = None
            try:
                source = json.loads(row["source_identity_json"]) if row["source_identity_json"] else None
                if row["kind"] != "create" and (
                    source is None or _source_identity(directory, row["source_path"]) != source
                ):
                    raise RuntimeError("output_source_unavailable")
                if row["kind"] != "remove":
                    if not row["stage_identity_json"]:
                        raise RuntimeError("output_operation_conflict")
                    stage = json.loads(row["stage_identity_json"])
                    with _open_regular(directory, row["stage_path"], os.O_RDONLY) as fd:
                        info = os.fstat(fd)
                        if _identity(info) != stage or info.st_size != row["written_bytes"]:
                            raise RuntimeError("output_operation_conflict")
                        os.fsync(fd)
                    try:
                        os.link(
                            row["stage_path"],
                            row["destination_path"],
                            src_dir_fd=directory,
                            dst_dir_fd=directory,
                            follow_symlinks=False,
                        )
                    except FileExistsError:
                        raise RuntimeError("output_path_conflict") from None
                    publication = {**stage, "nlink": 2}
                    for name in (row["stage_path"], row["destination_path"]):
                        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                        if _identity(info) != publication or info.st_size != row["written_bytes"]:
                            raise RuntimeError("output_operation_conflict")
                    os.fsync(directory)
                if source is not None and _source_identity(directory, row["source_path"]) != source:
                    raise RuntimeError("output_source_unavailable")
            except BaseException:
                # No DB commit was attempted. Keep witness/source for recovery.
                self.db.abort_output_file_operation(token, self.namespace)
                raise
            try:
                output = self.db.apply_output_file_operation(token, self.namespace, publication_identity=publication)
            except Exception:  # noqa: BLE001 - any commit failure may hide durable success
                try:
                    outcome, output = self.db.read_output_file_operation_outcome(token, self.namespace)
                except Exception:  # noqa: BLE001 - unknown outcome must preserve all evidence
                    # Never abort or discard evidence when the outcome is unknown.
                    raise RuntimeError("output_update_unconfirmed") from None
                if outcome["phase"] == "committed":
                    return self._complete_committed(directory, token, output)
                try:
                    aborted = self.db.abort_output_file_operation(token, self.namespace)
                    if not aborted:
                        outcome, output = self.db.read_output_file_operation_outcome(token, self.namespace)
                        if outcome["phase"] == "committed":
                            return self._complete_committed(directory, token, output)
                        if outcome["phase"] != "aborting":
                            raise RuntimeError("output_update_unconfirmed")
                except Exception:  # noqa: BLE001 - abort acknowledgement can also be lost
                    raise RuntimeError("output_update_unconfirmed") from None
                raise RuntimeError("output_operation_conflict") from None
            return self._complete_committed(directory, token, output)

    def _complete_committed(
        self, directory: int, token: str, output: CollectionsDatabase.OutputArtifactRow | None
    ) -> CollectionsDatabase.OutputArtifactRow | None:
        """Best-effort disposal cannot revoke a known logical commit."""
        try:
            self._recover_locked(directory, token)
            return output
        except _UnprovedOutputIdentity:
            category = "output_identity_unconfirmed"
        except (ReadingStorageUnavailable, OSError):
            category = "output_storage_unavailable"
        except Exception:  # noqa: BLE001 - preserve confirmed success and durable recovery authority
            logger.warning("Output post-commit cleanup deferred: output_update_unconfirmed")
            return output
        try:
            self.db.record_output_file_recovery_failure(token, self.namespace, category)
        except Exception:  # noqa: BLE001 - failure reporting cannot revoke the confirmed commit either
            logger.warning("Output post-commit cleanup status unconfirmed: {}", category)
        else:
            logger.warning("Output post-commit cleanup deferred: {}", category)
        return output

    async def recover_due(self, *, limit: int = 20) -> dict[str, int]:
        """Recover a bounded batch, yielding storage exclusion between operations.

        Cancellation drains the current interval without aborting unrelated live
        producers. History-only rows are excluded before any filesystem access.
        """
        try:
            tokens = await asyncio.to_thread(self.db.list_due_output_file_operations, self.namespace, limit=limit)
        except (DatabaseError, OSError):
            raise RuntimeError("output_update_unconfirmed") from None
        counts = {"finished": 0, "blocked": 0, "retry": 0, "skipped": 0}
        for token in tokens:
            with anyio.CancelScope(shield=True):
                status, cancelled = await _wait_worker(partial(self._recover_one, token))
            if cancelled:
                raise asyncio.CancelledError
            await anyio.lowlevel.checkpoint_if_cancelled()
            counts[status] += 1
        return counts

    def _recover_one(self, token: str) -> str:
        try:
            with _validated_storage_directory(self.output_root, storage_namespace_id=self.namespace) as (_, directory):
                return self._recover_locked(directory, token)
        except DatabaseError:
            raise RuntimeError("output_update_unconfirmed") from None
        except _UnprovedOutputIdentity:
            category = "output_identity_unconfirmed"
        except ReadingStorageBusy:
            category = "output_storage_busy"
        except (ReadingStorageUnavailable, OSError):
            category = "output_storage_unavailable"
        try:
            self.db.record_output_file_recovery_failure(token, self.namespace, category)
        except (DatabaseError, OSError):
            raise RuntimeError("output_update_unconfirmed") from None
        return "blocked" if category == "output_identity_unconfirmed" else "retry"

    def _recover_locked(self, directory: int, token: str) -> str:
        row = self.db.begin_output_file_recovery(token, self.namespace)
        if row is None:
            return "skipped"
        if row["phase"] == "aborting":
            self._clean_aborted(directory, row)
        else:
            self._clean_committed(directory, row)
        # Also sync absent paths left by a crash before an earlier fsync.
        os.fsync(directory)
        self.db.finish_output_file_operation(token, self.namespace)
        return "finished"

    def _clean_aborted(self, directory: int, row: dict) -> None:
        if row["stage_path"] is None:  # Removal has no private/public file to undo.
            return
        stage = _stat_optional(directory, row["stage_path"])
        destination = _stat_optional(directory, row["destination_path"])
        identity = json.loads(row["stage_identity_json"]) if row["stage_identity_json"] else None
        if destination is not None:
            if stage is None or identity is None:
                raise _UnprovedOutputIdentity("output_identity_unconfirmed")
            linked = {**identity, "nlink": 2}
            _require_identity(stage, linked, size=row["written_bytes"])
            _require_identity(destination, linked, size=row["written_bytes"])
            os.unlink(row["destination_path"], dir_fd=directory)
            os.fsync(directory)  # Witness must survive any failure of this sync.
        if stage is not None:
            _require_identity(
                os.stat(row["stage_path"], dir_fd=directory, follow_symlinks=False), identity, size=row["written_bytes"]
            )
            # A prior attempt may have unlinked destination but failed its sync.
            if destination is None:
                os.fsync(directory)
            os.unlink(row["stage_path"], dir_fd=directory)
            os.fsync(directory)

    def _clean_committed(self, directory: int, row: dict) -> None:
        if row["kind"] != "remove":
            stage = _stat_optional(directory, row["stage_path"])
            destination = _stat_optional(directory, row["destination_path"])
            identity = json.loads(row["publication_identity_json"]) if row["publication_identity_json"] else None
            if destination is None or identity is None:
                raise _UnprovedOutputIdentity("output_identity_unconfirmed")
            _require_identity(destination, {**identity, "nlink": 2 if stage else 1}, size=row["written_bytes"])
            if stage is not None:
                _require_identity(stage, identity, size=row["written_bytes"])
                os.unlink(row["stage_path"], dir_fd=directory)
                os.fsync(directory)
        if row["source_path"] is not None and not row["source_referenced"]:
            source = _stat_optional(directory, row["source_path"])
            identity = json.loads(row["source_identity_json"]) if row["source_identity_json"] else None
            if identity is None:
                raise _UnprovedOutputIdentity("output_identity_unconfirmed")
            if source is not None:
                if _identity(source, source=True) != identity:
                    raise _UnprovedOutputIdentity("output_identity_unconfirmed")
                os.unlink(row["source_path"], dir_fd=directory)
                os.fsync(directory)
