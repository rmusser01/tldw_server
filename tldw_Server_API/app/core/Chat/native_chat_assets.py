"""Private native candidate byte fencing; quota authority and MIME admission are separate."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.native_chat_asset_store import NativeQuotaReleasePendingError
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Sync.v2.blob_store import (
    LocalSyncBlobStore,
    SyncBlobStoreError,
    _lock_file_at,
    _MissingBlobDirectoryError,
    _open_directory_chain,
    _open_target_at,
    _require_native_storage_capabilities,
    _same_inode,
    _verify_named_open_blob,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import (
        NativeQuotaOwner,
        NativeQuotaReservation,
    )
    from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService


class NativeChatAssetIO:
    """Use only server-derived user storage and original durable candidate descriptors.

    This private foundation reconciles quota reservations but does not mark MIME-ready
    candidates. Public native capabilities stay disabled until those callers qualify.
    """

    def __init__(self, db: CharactersRAGDB, *, user_id: int) -> None:
        if type(user_id) is not int or user_id < 1:
            raise ValueError("native_asset_user_required")
        if db.owner_user_id != str(user_id) or db.client_id != db.owner_user_id:
            raise ValueError("native_asset_owner_mismatch")
        self._db = db
        self._owner_user_id = db.owner_user_id
        self._user_root = DatabasePaths.resolve_user_base_directory(user_id)

    async def reserve_quota(
        self, owner: AuthorizedNativeOwner, kind: str, operation_id: str, request_digest: str,
        generation: int, candidate_id: str, *, quota_owner: NativeQuotaOwner, quota_service: StorageQuotaService,
    ) -> NativeQuotaReservation:
        """Reconcile original authority outside chat transactions, then confirm the live writer."""
        self._require_outermost(owner)
        self._db.native_assets._quota_identity(owner, quota_owner)
        reservation = await quota_service.resolve_native_reservation(quota_owner, candidate_id)
        self._require_outermost(owner)
        with self._db.transaction() as conn:
            size = self._db.native_assets.record_quota_reservation_intent(
                owner, kind, operation_id, request_digest, candidate_id, generation, quota_owner, conn=conn,
            )
        if reservation is None:
            reservation = await quota_service.reserve_native_bytes(quota_owner, candidate_id, size)
        self._require_outermost(owner)
        with self._db.transaction() as conn:
            self._db.native_assets.confirm_quota_reservation(
                owner, kind, operation_id, request_digest, candidate_id, generation, quota_owner, reservation, conn=conn,
            )
        return reservation

    async def reclaim_quota(
        self, owner: AuthorizedNativeOwner, kind: str, operation_id: str, request_digest: str,
        candidate_id: str, *, quota_service: StorageQuotaService,
    ) -> bool:
        """Drain/remove bytes first; settle only the persisted original release intent."""
        self._require_outermost(owner)
        try:
            return await asyncio.to_thread(self.reclaim, owner, kind, operation_id, request_digest, candidate_id)
        except NativeQuotaReleasePendingError:
            pass
        self._require_outermost(owner)
        with self._db.transaction() as conn:
            quota_owner, _size, _generation = self._db.native_assets.read_quota_release_intent(
                owner, kind, operation_id, request_digest, candidate_id, conn=conn,
            )
        released = await quota_service.release_native_bytes(quota_owner, candidate_id)
        self._require_outermost(owner)
        with self._db.transaction() as conn:
            generation = self._db.native_assets.confirm_quota_release(
                owner, kind, operation_id, request_digest, candidate_id, released, conn=conn,
            )
            return self._db.native_assets.finish_reclamation(
                owner, kind, operation_id, request_digest, candidate_id, generation, conn=conn,
            )

    def _require_outermost(self, owner: AuthorizedNativeOwner) -> None:
        if (
            owner.client_id != self._owner_user_id
            or self._db.client_id != self._owner_user_id
            or self._db.owner_user_id != self._owner_user_id
        ):
            raise ValueError("operation_owner_mismatch")
        state = self._db._connection_state()
        connection = getattr(state, "conn", None)
        backend = self._db._get_pinned_backend() or self._db._backend
        if backend.backend_type == BackendType.SQLITE:
            nested = bool(getattr(connection, "in_transaction", False))
        else:
            nested = bool(getattr(state, "tx_depth", 0))
            if connection is not None:
                status = getattr(getattr(connection, "info", None), "transaction_status", None)
                nested = bool(nested or backend._tx_depth(connection) or getattr(status, "name", None) != "IDLE")
        if nested:
            raise ValueError("native_assets_outermost_required")

    @contextmanager
    def _guard(self, owner: AuthorizedNativeOwner, candidate_id: str) -> Iterator[tuple[int, Callable[[], None]]]:
        self._require_outermost(owner)
        if not isinstance(candidate_id, str) or not 1 <= len(candidate_id) <= 256:
            raise ValueError("invalid_candidate_identity")
        _require_native_storage_capabilities()
        name = (
            hashlib.sha256(json.dumps([owner.client_id, candidate_id], separators=(",", ":")).encode()).hexdigest()
            + ".lock"
        )
        with _open_directory_chain(self._user_root.parent, (), create=False) as parent:
            with _open_directory_chain(parent, (self._user_root.name,), create=False) as user:
                with _open_directory_chain(user, ("chat_assets",), create=True) as root:
                    with _open_directory_chain(root, ("_locks", "native-candidates"), create=True) as locks:
                        with _lock_file_at(locks, name, timeout=10) as locked:

                            def check_identity() -> None:
                                self._require_outermost(owner)
                                named_user = os.stat(self._user_root.name, dir_fd=parent, follow_symlinks=False)
                                if not stat.S_ISDIR(named_user.st_mode) or not _same_inode(named_user, os.fstat(user)):
                                    raise SyncBlobStoreError("Native user directory identity changed")
                                named_root = os.stat("chat_assets", dir_fd=user, follow_symlinks=False)
                                if not stat.S_ISDIR(named_root.st_mode) or not _same_inode(named_root, os.fstat(root)):
                                    raise SyncBlobStoreError("Native storage root identity changed")
                                with _open_directory_chain(
                                    root, ("_locks", "native-candidates"), create=False
                                ) as current:
                                    named_lock = os.stat(name, dir_fd=current, follow_symlinks=False)
                                    if not stat.S_ISREG(named_lock.st_mode) or not _same_inode(
                                        named_lock, os.fstat(locked)
                                    ):
                                        raise SyncBlobStoreError("Native candidate guard identity changed")

                            check_identity()
                            yield root, check_identity
                            check_identity()

    def _descriptor(
        self,
        owner: AuthorizedNativeOwner,
        kind: str,
        operation_id: str,
        digest: str,
        generation: int,
        candidate_id: str,
    ) -> dict[str, Any]:
        with self._db.transaction() as conn:
            row = self._db.native_assets.read_candidate_for_write(
                owner,
                kind,
                operation_id,
                digest,
                generation,
                candidate_id,
                now=datetime.now(timezone.utc),
                conn=conn,
            )
        self._canonical_key(
            row["storage_namespace_id"],
            row["upload_id"],
            row["expected_hash"],
            row["expected_size_bytes"],
            row["storage_key"],
        )
        if row["expected_size_bytes"] == 0:
            raise ValueError("unsupported_zero_size_native_asset")
        return row

    @staticmethod
    def _canonical_key(namespace: str, upload: str, digest: str, size: int, stored: str | None) -> str:
        if not isinstance(upload, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,255}", upload) is None:
            raise ValueError("invalid_native_upload_identity")
        if type(size) is not int or size < 0:
            raise ValueError("invalid_native_asset_size")
        key = LocalSyncBlobStore.namespace_storage_key(namespace, "sha256:" + digest)
        if stored is not None and stored != key:
            raise ValueError("candidate_storage_identity_conflict")
        return key

    def write_chunk(
        self,
        owner: AuthorizedNativeOwner,
        kind: str,
        operation_id: str,
        request_digest: str,
        generation: int,
        candidate_id: str,
        *,
        chunk_index: int,
        payload: bytes,
        expected_hash: str,
    ) -> str:
        """Admit under the stable guard, then close DB work before bounded byte IO."""
        with self._guard(owner, candidate_id) as (root, check):
            row = self._descriptor(owner, kind, operation_id, request_digest, generation, candidate_id)
            check()
            key = LocalSyncBlobStore._write_native_chunk(
                root_directory=root,
                upload_id=row["upload_id"],
                chunk_index=chunk_index,
                payload=payload,
                expected_hash=expected_hash,
                expected_total_size=row["expected_size_bytes"],
            )
            check()
            return key

    def publish(
        self,
        owner: AuthorizedNativeOwner,
        kind: str,
        operation_id: str,
        request_digest: str,
        generation: int,
        candidate_id: str,
        *,
        chunk_indexes: list[int],
    ) -> str:
        """Publish verified bytes; separate actual MIME admission owns prepared state."""
        with self._guard(owner, candidate_id) as (root, check):
            row = self._descriptor(owner, kind, operation_id, request_digest, generation, candidate_id)
            check()
            key = LocalSyncBlobStore._commit_native_upload(
                root_directory=root,
                upload_id=row["upload_id"],
                storage_namespace_id=row["storage_namespace_id"],
                payload_hash="sha256:" + row["expected_hash"],
                expected_size=row["expected_size_bytes"],
                chunk_indexes=chunk_indexes,
            )
            check()
            return key

    @staticmethod
    def _remove_directory(root: int, parents: tuple[str, ...], name: str) -> None:
        try:
            with _open_directory_chain(root, parents, create=False) as directory:
                try:
                    shutil.rmtree(name, dir_fd=directory)
                except FileNotFoundError:
                    pass
                try:
                    os.stat(name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    os.fsync(directory)
                    return
                raise SyncBlobStoreError("Native candidate removal is unconfirmed")
        except _MissingBlobDirectoryError:
            return

    def reclaim(
        self, owner: AuthorizedNativeOwner, kind: str, operation_id: str, request_digest: str, candidate_id: str
    ) -> bool:
        """Commit the fence, drain writers, remove exact bytes, and retain uncertain quota."""
        self._require_outermost(owner)
        with self._db.transaction() as conn:
            candidate = self._db.native_assets.claim_reclamation(
                owner,
                kind,
                operation_id,
                request_digest,
                candidate_id,
                now=datetime.now(timezone.utc),
                conn=conn,
            )
            if candidate is not None:
                self._db.native_assets._require_exclusive_paths(
                    owner.client_id,
                    candidate_id,
                    candidate.namespace_id,
                    candidate.upload_id,
                    conn=conn,
                )
        if candidate is None:
            return False
        key = self._canonical_key(
            candidate.namespace_id,
            candidate.upload_id,
            candidate.expected_hash,
            candidate.size_bytes,
            candidate.storage_key,
        )
        with self._guard(owner, candidate_id) as (root, check):
            with self._db.transaction() as conn:
                _, current = self._db.native_assets._locked_candidate_for_operation(
                    owner, kind, operation_id, request_digest, candidate_id, conn=conn
                )
                expected = {
                    "state": "reclaiming",
                    "attempt_generation": candidate.generation,
                    "storage_namespace_id": candidate.namespace_id,
                    "upload_id": candidate.upload_id,
                    "expected_hash": candidate.expected_hash,
                    "expected_size_bytes": candidate.size_bytes,
                    "storage_key": candidate.storage_key,
                }
                if any(current[field] != value for field, value in expected.items()):
                    raise ValueError("stale_reclamation")
                self._db.native_assets._require_unclaimed(owner, candidate_id, conn=conn)
                self._db.native_assets._require_exclusive_paths(
                    owner.client_id, candidate_id, candidate.namespace_id, candidate.upload_id, conn=conn
                )
            try:
                with _open_directory_chain(root, ("blobs", "v2", candidate.namespace_id), create=False) as directory:
                    try:
                        handle = _open_target_at(directory, key.rsplit("/", 1)[1])
                    except FileNotFoundError:
                        handle = None
                    if handle is not None:
                        with handle:
                            _verify_named_open_blob(
                                handle,
                                directory=directory,
                                name=key.rsplit("/", 1)[1],
                                payload_hash="sha256:" + candidate.expected_hash,
                                expected_size=candidate.size_bytes,
                            )
            except _MissingBlobDirectoryError:
                pass
            check()
            self._remove_directory(root, ("blobs", "v2"), candidate.namespace_id)
            self._remove_directory(root, ("_uploads",), candidate.upload_id)
            check()
            with self._db.transaction() as conn:
                self._db.native_assets.record_release_after_removal(
                    owner, kind, operation_id, request_digest, candidate_id, candidate.generation, conn=conn
                )
            with self._db.transaction() as conn:
                return self._db.native_assets.finish_reclamation(
                    owner, kind, operation_id, request_digest, candidate_id, candidate.generation, conn=conn
                )
