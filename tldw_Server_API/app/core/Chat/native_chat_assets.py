"""Private native preparation, guarded exact reads and conservative byte reclamation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.native_chat_asset_store import NativeQuotaReleasePendingError
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Sync.v2.blob_store import (
    STREAM_CHUNK_SIZE,
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
    from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeAssetRetentionRequestV1
    from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
    from tldw_Server_API.app.core.DB_Management.chacha.native_chat_asset_store import NativeCandidate
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import (
        NativeQuotaOwner,
        NativeQuotaReservation,
    )
    from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService


class NativeChatAssetIO:
    """Use only server-derived user storage and original durable candidate descriptors.

    Preparation verifies uploaded bytes before prepared CAS; it does not adopt claims.
    Public native capabilities stay disabled until their callers qualify.
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
        self,
        owner: AuthorizedNativeOwner,
        kind: str,
        operation_id: str,
        request_digest: str,
        generation: int,
        candidate_id: str,
        *,
        quota_owner: NativeQuotaOwner,
        quota_service: StorageQuotaService,
    ) -> NativeQuotaReservation:
        """Reconcile original authority outside chat transactions, then confirm the live writer."""
        self._require_outermost(owner)
        self._db.native_assets._quota_identity(owner, quota_owner)
        reservation = await quota_service.resolve_native_reservation(quota_owner, candidate_id)
        self._require_outermost(owner)
        with self._db.transaction() as conn:
            refresh = False
            if reservation is None:
                self._db.native_assets._require_live_attempt(
                    owner,
                    kind,
                    operation_id,
                    request_digest,
                    generation,
                    datetime.now(timezone.utc),
                    conn=conn,
                )
                _, candidate = self._db.native_assets._locked_candidate_for_operation(
                    owner,
                    kind,
                    operation_id,
                    request_digest,
                    candidate_id,
                    conn=conn,
                )
                refresh = candidate["state"] == "prepared"
            if not refresh:
                if self._db.native_assets.reconcile_prepared_quota(
                    owner,
                    kind,
                    operation_id,
                    request_digest,
                    candidate_id,
                    generation,
                    quota_owner,
                    reservation,
                    conn=conn,
                ):
                    return reservation
                size = self._db.native_assets.record_quota_reservation_intent(
                    owner,
                    kind,
                    operation_id,
                    request_digest,
                    candidate_id,
                    generation,
                    quota_owner,
                    conn=conn,
                )
        if refresh:
            reservation = await quota_service.resolve_native_reservation(quota_owner, candidate_id)
            self._require_outermost(owner)
            with self._db.transaction() as conn:
                if not self._db.native_assets.reconcile_prepared_quota(
                    owner,
                    kind,
                    operation_id,
                    request_digest,
                    candidate_id,
                    generation,
                    quota_owner,
                    reservation,
                    conn=conn,
                ):
                    raise ValueError("stale_candidate")
            return reservation
        if reservation is None:
            reservation = await quota_service.reserve_native_bytes(quota_owner, candidate_id, size)
        self._require_outermost(owner)
        with self._db.transaction() as conn:
            if not self._db.native_assets.reconcile_prepared_quota(
                owner,
                kind,
                operation_id,
                request_digest,
                candidate_id,
                generation,
                quota_owner,
                reservation,
                conn=conn,
            ):
                self._db.native_assets.confirm_quota_reservation(
                    owner,
                    kind,
                    operation_id,
                    request_digest,
                    candidate_id,
                    generation,
                    quota_owner,
                    reservation,
                    conn=conn,
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

    def _upload_policy(self, row: dict[str, Any], filename: str, validator: Any) -> tuple[str, int]:
        """Reuse destination policy without treating a filename as byte evidence."""
        from tldw_Server_API.app.core.config import get_config_value, settings
        from tldw_Server_API.app.core.Ingestion_Media_Processing.document_upload_preflight import (
            DEFAULT_MAX_CHAT_UPLOAD_BYTES,
            _media_type_for,
        )
        from tldw_Server_API.app.core.Utils.image_validation import get_allowed_image_mime_types

        representation = row["representation"]
        if representation in {"embedded_image_v1", "generated_image_revision_v1"}:
            if row["mime_type"] not in get_allowed_image_mime_types():
                raise ValueError("native_asset_mime_unsupported")
            media_type = "image"
            limit = int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
        elif representation in {"document_original_v1", "document_text_v1"}:
            media_type = _media_type_for(filename)
            if media_type == "unsupported":
                raise ValueError("native_asset_representation_unsupported")
            if representation == "document_text_v1" and row["mime_type"] not in {"text/plain", "text/markdown"}:
                raise ValueError("native_asset_mime_unsupported")
            key = {"pdf": "max_pdf_file_size_mb", "ebook": "max_epub_file_size_mb", "document": "max_document_file_size_mb"}[media_type]
            default = "100" if media_type == "ebook" else "50"
            current_bytes = int(float(get_config_value("Media-Processing", key, default=default)) * 1024 * 1024)
            limit = min(DEFAULT_MAX_CHAT_UPLOAD_BYTES, current_bytes)
        else:
            raise ValueError("native_asset_representation_unsupported")
        config = validator.get_media_config(media_type)
        if not config or row["mime_type"] not in config.get("allowed_mimetypes", set()):
            raise ValueError("native_asset_mime_unsupported")
        configured_bytes = int(config["max_size_mb"] * 1024 * 1024)
        limit = min(limit, configured_bytes)
        if limit < 1 or row["expected_size_bytes"] > limit:
            raise ValueError("native_asset_size_limit")
        return media_type, limit

    async def prepare_retention_upload(
        self,
        owner: AuthorizedNativeOwner,
        request: NativeAssetRetentionRequestV1,
        generation: int,
        *,
        reference_id: str,
        candidate_id: str,
        namespace_id: str,
        upload_id: str,
        size_bytes: int,
        mime_type: str,
        upload: Any,
        quota_owner: NativeQuotaOwner,
        quota_service: StorageQuotaService,
    ) -> NativeCandidate:
        """Prepare one reviewed candidate without publishing claims or readiness."""
        from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeAssetRetentionRequestV1

        self._require_outermost(owner)
        self._db.native_assets._quota_identity(owner, quota_owner)
        if not isinstance(request, NativeAssetRetentionRequestV1):
            raise ValueError("invalid_retention_request")
        request = NativeAssetRetentionRequestV1.model_validate(request.model_dump(mode="json"))

        def admit() -> NativeCandidate:
            self._require_outermost(owner)
            with self._db.transaction() as conn:
                return self._db.native_assets.reserve_retention_candidate(
                    owner,
                    request,
                    generation,
                    reference_id=reference_id,
                    candidate_id=candidate_id,
                    namespace_id=namespace_id,
                    upload_id=upload_id,
                    size_bytes=size_bytes,
                    mime_type=mime_type,
                    conn=conn,
                )

        admit()
        await self.reserve_quota(
            owner,
            request.operation_kind,
            request.operation_id,
            request.request_digest,
            generation,
            candidate_id,
            quota_owner=quota_owner,
            quota_service=quota_service,
        )
        candidate = admit()
        if candidate.state == "prepared":
            return candidate
        await self.prepare_upload(
            owner,
            request.operation_kind,
            request.operation_id,
            request.request_digest,
            generation,
            candidate_id,
            upload=upload,
        )
        return admit()

    async def prepare_upload(
        self, owner: AuthorizedNativeOwner, kind: str, operation_id: str, request_digest: str,
        generation: int, candidate_id: str, *, upload: Any,
    ) -> NativeCandidate:
        """Validate through the house sink, then prepare the original reserved candidate."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
            TempDirManager,
            save_uploaded_files,
        )
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import get_default_file_validator

        self._require_outermost(owner)
        row = self._descriptor(owner, kind, operation_id, request_digest, generation, candidate_id)
        filename = getattr(upload, "filename", None)
        if not isinstance(filename, str) or not filename:
            raise ValueError("native_upload_filename_required")
        validator = get_default_file_validator()
        media_type, limit = self._upload_policy(row, filename, validator)
        with TempDirManager(prefix="native_chat_upload_") as directory:
            files, errors = await save_uploaded_files(
                [upload], directory, validator, expected_media_type_key=media_type,
                max_upload_bytes=min(limit, row["expected_size_bytes"]),
            )
            self._require_outermost(owner)
            if errors or len(files) != 1:
                raise ValueError("native_upload_validation_failed")
            cancelled = threading.Event()
            work = asyncio.create_task(asyncio.to_thread(
                self._prepare_upload, owner, kind, operation_id, request_digest,
                generation, candidate_id, Path(files[0]["path"]), filename, validator, cancelled,
            ))
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                cancelled.set()
                # Keep the private staging directory until even repeated cancellation drains the worker.
                while not work.done():
                    try:
                        await asyncio.wait({work})
                    except asyncio.CancelledError:
                        continue
                if not work.cancelled() and work.exception() is not None:
                    from loguru import logger
                    logger.debug("Cancelled native upload worker settled (error_type={})", type(work.exception()).__name__)
                raise

    def _prepare_upload(
        self, owner: AuthorizedNativeOwner, kind: str, operation_id: str, digest: str,
        generation: int, candidate_id: str, path: Path, filename: str, validator: Any, cancelled: threading.Event,
    ) -> NativeCandidate:
        from tldw_Server_API.app.core.Utils.image_validation import validate_uploaded_image_bytes

        def require_active() -> None:
            if cancelled.is_set():
                raise ValueError("native_upload_cancelled")

        require_active()
        with self._guard(owner, candidate_id) as (root, check):
            row = self._descriptor(owner, kind, operation_id, digest, generation, candidate_id)
            media_type, limit = self._upload_policy(row, filename, validator)
            with path.open("rb") as handle:
                payload = handle.read(row["expected_size_bytes"] + 1)
            if len(payload) != row["expected_size_bytes"] or hashlib.sha256(payload).hexdigest() != row["expected_hash"]:
                raise ValueError("native_asset_content_mismatch")
            if media_type == "image":
                if not validate_uploaded_image_bytes(payload, row["mime_type"], max_bytes=limit)[0]:
                    raise ValueError("native_asset_image_invalid")
            elif row["mime_type"] in {"text/plain", "text/markdown"}:
                try:
                    payload.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise ValueError("native_asset_text_invalid") from exc
            else:
                config = validator.get_media_config(media_type)
                result = validator.validate_file(
                    path, original_filename=filename, media_type_key=media_type,
                    allowed_extensions_override=config["allowed_extensions"],
                    allowed_mimetypes_override={row["mime_type"]}, max_size_mb_override=limit / (1024 * 1024),
                )
                if result.mime_detection_source not in {"magic", "python-magic"}:
                    raise ValueError("native_asset_mime_unverified")
                if not result or result.detected_mime_type != row["mime_type"]:
                    raise ValueError("native_asset_mime_mismatch")
            check()
            indexes = []
            for offset in range(0, len(payload), STREAM_CHUNK_SIZE):
                require_active()
                self._descriptor(owner, kind, operation_id, digest, generation, candidate_id)
                check()
                require_active()
                chunk = payload[offset:offset + STREAM_CHUNK_SIZE]
                index = offset // STREAM_CHUNK_SIZE
                LocalSyncBlobStore._write_native_chunk(
                    root_directory=root, upload_id=row["upload_id"], chunk_index=index, payload=chunk,
                    expected_hash="sha256:" + hashlib.sha256(chunk).hexdigest(), expected_total_size=len(payload),
                )
                indexes.append(index)
            require_active()
            self._descriptor(owner, kind, operation_id, digest, generation, candidate_id)
            check()
            require_active()
            key = LocalSyncBlobStore._commit_native_upload(
                root_directory=root, upload_id=row["upload_id"], storage_namespace_id=row["storage_namespace_id"],
                payload_hash="sha256:" + row["expected_hash"], expected_size=len(payload), chunk_indexes=indexes,
            )
            check()
            require_active()
            with self._db.transaction() as conn:
                current = self._db.native_assets.read_candidate_for_write(
                    owner, kind, operation_id, digest, generation, candidate_id,
                    now=datetime.now(timezone.utc), conn=conn,
                )
                self._upload_policy(current, filename, validator)
                require_active()
                prepared = self._db.native_assets.mark_candidate_prepared(
                    owner, kind, operation_id, digest, generation, candidate_id,
                    row["expected_hash"], len(payload), row["mime_type"], key,
                    now=datetime.now(timezone.utc), conn=conn,
                )
                require_active()
                return prepared

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

    @contextmanager
    def read_owned_asset(
        self, owner: AuthorizedNativeOwner, conversation_id: str, asset_id: str, revision: str,
    ) -> Iterator[tuple[dict[str, Any], BinaryIO]]:
        """Yield exact committed fork bytes under the guard, without source fallback.

        The trusted caller owns bounded consumption inside this context. Public
        streaming, retention claims and canonical bundle admission remain unqualified.
        """
        self._require_outermost(owner)

        def descriptor() -> dict[str, Any]:
            self._require_outermost(owner)
            with self._db.transaction() as conn:
                return self._db.native_assets.read_committed_asset(owner, conversation_id, asset_id, revision, conn=conn)

        admitted = descriptor()
        key = self._canonical_key(
            admitted["storage_namespace_id"], admitted["upload_id"], admitted["content_hash"],
            admitted["size_bytes"], admitted["storage_key"],
        )
        with self._guard(owner, admitted["candidate_id"]) as (root, check):
            if descriptor() != admitted:
                raise ValueError("native_asset_claim_conflict")
            with _open_directory_chain(root, ("blobs", "v2", admitted["storage_namespace_id"]), create=False) as directory:
                name = key.rsplit("/", 1)[1]
                with _open_target_at(directory, name) as handle:
                    check()
                    inode = _verify_named_open_blob(
                        handle, directory=directory, name=name,
                        payload_hash="sha256:" + admitted["content_hash"], expected_size=admitted["size_bytes"],
                    )
                    check()
                    if descriptor() != admitted:
                        raise ValueError("native_asset_claim_conflict")
                    check()
                    yield dict(admitted), handle
                    check()
                    _verify_named_open_blob(
                        handle, directory=directory, name=name, expected_inode=inode,
                        payload_hash="sha256:" + admitted["content_hash"], expected_size=admitted["size_bytes"],
                    )
                    if descriptor() != admitted:
                        raise ValueError("native_asset_claim_conflict")

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
