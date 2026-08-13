"""Resumable source-verified bootstrap for legacy Notes attachments."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import (
    LegacyAttachmentSourceError,
    NoteAttachmentPolicyError,
    NotesAttachmentBootstrapInterrupted,
)
from tldw_Server_API.app.core.Notes.attachment_policy import (
    canonicalize_note_attachment_file_name,
    validate_note_attachment_content_type,
    validate_note_attachment_original_file_name,
)
from tldw_Server_API.app.core.Notes.legacy_attachment_source import (
    LEGACY_ATTACHMENT_NOTE_PAGE_LIMIT,
    LegacyAttachmentCandidate,
    LegacyAttachmentSource,
)

from .attachment_refs_v2 import parse_attachment_ref_v2_payload
from .errors import SyncStoreError
from .models import SyncDataset, SyncEnvelope
from .server_origin import SERVER_ORIGIN_DEVICE_ID
from .server_origin_batch import (
    ServerOriginMutationStep,
    SyncServerOriginBatchMaterializationError,
    capture_server_origin_mutation_batch,
)
from .service import SyncV2Service

_SAFE_SOURCE_ERROR = "notes_attachment_source_invalid"
_SAFE_SOURCE_CHANGED = "notes_attachment_source_changed"
_SAFE_SOURCE_TOO_LARGE = "notes_attachment_source_too_large"
_SAFE_CAPTURE_ERROR = "notes_attachment_capture_failed"
_CANDIDATE_PAGE_SIZE = 1_000
_STREAM_CHUNK_SIZE = 64 * 1024
_EMPTY_SOURCE_HASH = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True, slots=True)
class _BootstrapCursor:
    phase: str
    note_id: str | None
    source_key_hash: str | None
    rolling_hash: str
    processed_count: int


@dataclass(frozen=True, slots=True)
class _SourcePage:
    candidates: tuple[LegacyAttachmentCandidate, ...]
    cursor_note_id: str | None
    cursor_source_key: str | None
    exhausted: bool


class NotesAttachmentBootstrapper:
    """Import legacy files through verified blobs and trusted v2 projection."""

    def __init__(
        self,
        note_db: CharactersRAGDB,
        *,
        user_root: Path | None = None,
        max_candidates_per_run: int = 1_000,
        after_upload: Callable[[LegacyAttachmentCandidate], None] | None = None,
        after_capture: Callable[[LegacyAttachmentCandidate], None] | None = None,
    ) -> None:
        if not 1 <= max_candidates_per_run <= 1_000:
            raise ValueError("Notes attachment bootstrap limit must be 1..1000")
        self._note_db = note_db
        self._user_root = user_root
        self._max_candidates_per_run = max_candidates_per_run
        self._after_upload = after_upload
        self._after_capture = after_capture

    def dry_run(
        self,
        *,
        service: SyncV2Service,
        user_id: str,
    ) -> dict[str, object]:
        """Return one bounded source count without mutating Sync or legacy state."""

        source = LegacyAttachmentSource(
            self._note_db,
            owner_user_id=user_id,
            user_root=self._user_root,
        )
        count = 0
        lower_bound = False
        try:
            note_ids = source.list_note_ids(
                after_note_id=None,
                limit=LEGACY_ATTACHMENT_NOTE_PAGE_LIMIT,
            )
            for note_id in note_ids:
                remaining = self._max_candidates_per_run - count
                if remaining == 0:
                    lower_bound = True
                    break
                candidates = source.list_candidates(
                    note_id,
                    after_source_key=None,
                    limit=remaining,
                )
                for candidate in candidates:
                    if candidate.size_bytes > min(
                        service.settings.max_attachment_bytes,
                        service.settings.max_blob_bytes,
                    ):
                        raise LegacyAttachmentSourceError(_SAFE_SOURCE_TOO_LARGE)
                    validate_note_attachment_original_file_name(candidate.file_name)
                    canonicalize_note_attachment_file_name(candidate.file_name)
                    _content_type(candidate)
                    source.verify_candidate(candidate)
                count += len(candidates)
                if len(candidates) == remaining:
                    lower_bound = True
                    break
            if len(note_ids) == LEGACY_ATTACHMENT_NOTE_PAGE_LIMIT:
                lower_bound = True
            return {
                "candidate_count": count,
                "candidate_count_is_lower_bound": lower_bound,
                "error_code": None,
            }
        except (LegacyAttachmentSourceError, NoteAttachmentPolicyError, ValueError) as exc:
            error_code = (
                _safe_source_error(exc.error_code)
                if isinstance(exc, LegacyAttachmentSourceError)
                else _SAFE_SOURCE_ERROR
            )
            return {
                "candidate_count": count,
                "candidate_count_is_lower_bound": lower_bound,
                "error_code": error_code,
            }

    def bootstrap(
        self,
        *,
        service: SyncV2Service,
        user_id: str,
        dataset: SyncDataset,
    ) -> SyncDataset:
        """Resume one bounded source page or persist a sanitized failed state."""

        if dataset.owner_user_id != user_id:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        metadata = dataset.metadata.get("notes_attachment_v2")
        if not isinstance(metadata, Mapping):
            raise SyncStoreError("notes_attachment_sync_not_ready")
        if metadata.get("state") == "ready":
            return dataset
        bootstrap_id = metadata.get("bootstrap_id")
        if (
            metadata.get("state") != "initializing"
            or not isinstance(bootstrap_id, str)
            or not bootstrap_id
        ):
            raise SyncStoreError("notes_attachment_sync_not_ready")

        captured_count = _non_negative_int(metadata.get("captured_count"))
        expected_count = _non_negative_int(metadata.get("expected_count"))
        established_hash = metadata.get("source_hash")
        source = LegacyAttachmentSource(
            self._note_db,
            owner_user_id=user_id,
            user_root=self._user_root,
        )
        try:
            cursor = _decode_cursor(
                metadata.get("source_cursor"),
                captured_count=captured_count,
                source_hash=established_hash,
            )
            remaining = self._max_candidates_per_run
            if cursor.phase == "capture":
                page = _source_page(
                    source,
                    service=service,
                    dataset=dataset,
                    user_id=user_id,
                    bootstrap_id=bootstrap_id,
                    cursor=cursor,
                    limit=remaining,
                )
                for candidate in page.candidates:
                    self._capture_candidate(
                        source=source,
                        candidate=candidate,
                        service=service,
                        user_id=user_id,
                        dataset=dataset,
                        bootstrap_id=bootstrap_id,
                    )
                    captured_count += 1
                    expected_count = captured_count
                    remaining -= 1
                    cursor = _advanced_cursor(cursor, candidate)
                    dataset = self._persist_progress(
                        service,
                        dataset,
                        user_id=user_id,
                        bootstrap_id=bootstrap_id,
                        captured_count=captured_count,
                        expected_count=expected_count,
                        source_hash=None,
                        cursor=cursor,
                    )
                if not page.exhausted:
                    if page.cursor_source_key is None and page.cursor_note_id is not None:
                        cursor = _cursor_after_empty_notes(cursor, page.cursor_note_id)
                        dataset = self._persist_progress(
                            service,
                            dataset,
                            user_id=user_id,
                            bootstrap_id=bootstrap_id,
                            captured_count=captured_count,
                            expected_count=expected_count,
                            source_hash=None,
                            cursor=cursor,
                        )
                    return dataset
                established_hash = cursor.rolling_hash
                cursor = _initial_cursor("verify")
                dataset = self._persist_progress(
                    service,
                    dataset,
                    user_id=user_id,
                    bootstrap_id=bootstrap_id,
                    captured_count=captured_count,
                    expected_count=expected_count,
                    source_hash=established_hash,
                    cursor=cursor,
                )

            if remaining == 0:
                return dataset
            page = _source_page(
                source,
                service=service,
                dataset=dataset,
                user_id=user_id,
                bootstrap_id=bootstrap_id,
                cursor=cursor,
                limit=remaining,
            )
            for candidate in page.candidates:
                self._verify_candidate(
                    source=source,
                    candidate=candidate,
                    service=service,
                    user_id=user_id,
                    dataset=dataset,
                    bootstrap_id=bootstrap_id,
                )
                cursor = _advanced_cursor(cursor, candidate)
                if cursor.processed_count > expected_count:
                    raise LegacyAttachmentSourceError(_SAFE_SOURCE_CHANGED)
                dataset = self._persist_progress(
                    service,
                    dataset,
                    user_id=user_id,
                    bootstrap_id=bootstrap_id,
                    captured_count=captured_count,
                    expected_count=expected_count,
                    source_hash=(
                        established_hash if isinstance(established_hash, str) else None
                    ),
                    cursor=cursor,
                )
            if not page.exhausted:
                if page.cursor_source_key is None and page.cursor_note_id is not None:
                    cursor = _cursor_after_empty_notes(cursor, page.cursor_note_id)
                    dataset = self._persist_progress(
                        service,
                        dataset,
                        user_id=user_id,
                        bootstrap_id=bootstrap_id,
                        captured_count=captured_count,
                        expected_count=expected_count,
                        source_hash=(
                            established_hash
                            if isinstance(established_hash, str)
                            else None
                        ),
                        cursor=cursor,
                    )
                return dataset
            if (
                cursor.processed_count != expected_count
                or cursor.rolling_hash != established_hash
            ):
                raise LegacyAttachmentSourceError(_SAFE_SOURCE_CHANGED)
            return service.store.transition_notes_attachment_bootstrap(
                dataset.dataset_id,
                owner_user_id=user_id,
                bootstrap_id=bootstrap_id,
                expected_state="initializing",
                state="ready",
                captured_count=expected_count,
                expected_count=expected_count,
                source_hash=established_hash,
                source_cursor=None,
                ready_verifier=lambda: True,
            )
        except LegacyAttachmentSourceError as exc:
            return self._fail(
                service,
                dataset,
                user_id=user_id,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                source_hash=(established_hash if isinstance(established_hash, str) else None),
                error_code=_safe_source_error(exc.error_code),
            )
        except (NoteAttachmentPolicyError, ValueError):
            return self._fail(
                service,
                dataset,
                user_id=user_id,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                source_hash=(established_hash if isinstance(established_hash, str) else None),
                error_code=_SAFE_SOURCE_ERROR,
            )
        except SyncServerOriginBatchMaterializationError as exc:
            if exc.retryable:
                return dataset
            return self._fail(
                service,
                dataset,
                user_id=user_id,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                source_hash=(established_hash if isinstance(established_hash, str) else None),
                error_code=_SAFE_CAPTURE_ERROR,
            )
        except NotesAttachmentBootstrapInterrupted:
            return service.store.get_dataset(
                dataset.dataset_id,
                owner_user_id=user_id,
            ) or dataset
        except Exception:  # noqa: BLE001 - durable state exposes only safe codes.
            return self._fail(
                service,
                dataset,
                user_id=user_id,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                source_hash=(established_hash if isinstance(established_hash, str) else None),
                error_code=_SAFE_CAPTURE_ERROR,
            )

    @staticmethod
    def _persist_progress(
        service: SyncV2Service,
        dataset: SyncDataset,
        *,
        user_id: str,
        bootstrap_id: str,
        captured_count: int,
        expected_count: int,
        source_hash: str | None,
        cursor: _BootstrapCursor,
    ) -> SyncDataset:
        return service.store.transition_notes_attachment_bootstrap(
            dataset.dataset_id,
            owner_user_id=user_id,
            bootstrap_id=bootstrap_id,
            expected_state="initializing",
            state="initializing",
            captured_count=captured_count,
            expected_count=expected_count,
            source_hash=source_hash,
            source_cursor=_encode_cursor(cursor),
        )

    def _capture_candidate(
        self,
        *,
        source: LegacyAttachmentSource,
        candidate: LegacyAttachmentCandidate,
        service: SyncV2Service,
        user_id: str,
        dataset: SyncDataset,
        bootstrap_id: str,
    ) -> None:
        if candidate.size_bytes > min(
            service.settings.max_attachment_bytes,
            service.settings.max_blob_bytes,
        ):
            raise LegacyAttachmentSourceError(_SAFE_SOURCE_TOO_LARGE)
        mapping = service.store.resolve_notes_attachment_source_map(
            dataset.dataset_id,
            owner_user_id=user_id,
            bootstrap_id=bootstrap_id,
            source_key=candidate.source_key,
            note_id=candidate.note_id,
        )
        file_name = _available_file_name(
            self._note_db,
            dataset.dataset_id,
            candidate,
            attachment_id=mapping.attachment_id,
        )
        original_file_name = validate_note_attachment_original_file_name(
            candidate.file_name
        )
        content_type = _content_type(candidate)
        timestamp = datetime.fromtimestamp(candidate.modified_ns / 1_000_000_000, UTC).isoformat()
        upload = service.create_blob_upload_session(
            user_id=user_id,
            dataset_id=dataset.dataset_id,
            device_id=None,
            domain="attachment.ref",
            entity_id=mapping.attachment_id,
            attachment_id=mapping.attachment_id,
            content_type=content_type,
            size_bytes=candidate.size_bytes,
            payload_hash=candidate.sha256,
            chunk_size=min(service.settings.max_chunk_bytes, _STREAM_CHUNK_SIZE),
            chunk_count=(
                candidate.size_bytes
                + min(service.settings.max_chunk_bytes, _STREAM_CHUNK_SIZE)
                - 1
            )
            // min(service.settings.max_chunk_bytes, _STREAM_CHUNK_SIZE),
            idempotency_key=f"{bootstrap_id}:{mapping.source_key_hash}",
            metadata={
                "notes_attachment_intent": {
                    "intent": "create",
                    "note_id": candidate.note_id,
                    "attachment_id": mapping.attachment_id,
                    "file_name": file_name,
                }
            },
            trusted_notes_attachment_bootstrap_id=bootstrap_id,
        )
        if upload.status != "complete":
            offset = 0
            for chunk_index, chunk in enumerate(
                source.iter_candidate_chunks(
                    candidate,
                    chunk_size=upload.chunk_size,
                )
            ):
                service.upload_blob_chunk(
                    user_id=user_id,
                    dataset_id=dataset.dataset_id,
                    upload_id=upload.upload_id,
                    chunk_index=chunk_index,
                    offset_bytes=offset,
                    chunk_payload=chunk,
                    chunk_hash="sha256:" + hashlib.sha256(chunk).hexdigest(),
                )
                offset += len(chunk)
        blob = service.complete_blob_upload(
            user_id=user_id,
            dataset_id=dataset.dataset_id,
            upload_id=upload.upload_id,
        )
        if self._after_upload is not None:
            self._after_upload(candidate)
        source.verify_candidate(candidate)

        payload = {
            "attachment_id": mapping.attachment_id,
            "parent_domain": "notes.note",
            "parent_object_id": candidate.note_id,
            "file_name": file_name,
            "original_file_name": original_file_name,
            "content_type": content_type,
            "size_bytes": candidate.size_bytes,
            "blob_hash": candidate.sha256,
            "created_at": timestamp,
            "last_modified": timestamp,
            "created_by": SERVER_ORIGIN_DEVICE_ID,
        }
        capture_server_origin_mutation_batch(
            service=service,
            user_id=user_id,
            steps=(
                ServerOriginMutationStep(
                    domain="attachment.ref",
                    operation="upsert",
                    object_id=mapping.attachment_id,
                    parent_id=candidate.note_id,
                    payload=payload,
                    routing_metadata={
                        "bootstrap_capture": True,
                        "bootstrap_id": bootstrap_id,
                    },
                    stable_key=f"attachment.ref:{mapping.attachment_id}",
                    created_at_client=timestamp,
                    schema_version=2,
                    adapter_version=2,
                ),
            ),
            source="notes-attachment-bootstrap",
            idempotency_key=f"{bootstrap_id}:{mapping.source_key_hash}",
            trusted_notes_attachment_bootstrap_id=bootstrap_id,
            bootstrap_step_verifier=lambda envelope: self._step_matches_source(
                envelope,
                source=source,
                candidate=candidate,
                payload=payload,
                service=service,
                blob_storage_key=blob.storage_key,
            ),
        )
        if self._after_capture is not None:
            self._after_capture(candidate)
        service.store.record_notes_attachment_cleanup_candidate(
            dataset.dataset_id,
            owner_user_id=user_id,
            bootstrap_id=bootstrap_id,
            source_key=candidate.source_key,
            source_relative_path=candidate.relative_path,
            source_blob_hash=candidate.sha256,
            source_size_bytes=candidate.size_bytes,
            source_modified_ns=candidate.modified_ns,
        )

    def _step_matches_source(
        self,
        envelope: SyncEnvelope,
        *,
        source: LegacyAttachmentSource,
        candidate: LegacyAttachmentCandidate,
        payload: Mapping[str, object],
        service: SyncV2Service,
        blob_storage_key: str,
    ) -> bool:
        try:
            source.verify_candidate(candidate)
            if service.blob_store is None:
                return False
            service.blob_store.verify_blob(
                blob_storage_key,
                payload_hash=candidate.sha256,
                expected_size=candidate.size_bytes,
            )
            parsed = parse_attachment_ref_v2_payload(envelope.operation, envelope.payload)
            blob_store = envelope.payload.get("blob_hash")
            service_blob = parsed.blob_hash
            if blob_store != candidate.sha256 or service_blob != candidate.sha256:
                return False
            # The canonical blob path is verified independently of source bytes.
            return (
                envelope.adapter_version == 2
                and envelope.schema_version == 2
                and envelope.object_id == payload["attachment_id"]
                and dict(envelope.payload) == dict(payload)
                and bool(blob_storage_key)
            )
        except (LegacyAttachmentSourceError, ValueError):
            return False

    def _verify_candidate(
        self,
        *,
        source: LegacyAttachmentSource,
        candidate: LegacyAttachmentCandidate,
        service: SyncV2Service,
        dataset: SyncDataset,
        user_id: str,
        bootstrap_id: str,
    ) -> None:
        source.verify_candidate(candidate)
        persisted = service.store.get_notes_attachment_bootstrap_source_by_hash(
            dataset.dataset_id,
            owner_user_id=user_id,
            bootstrap_id=bootstrap_id,
            source_key_hash=_source_key_hash(candidate.source_key),
        )
        if persisted is None:
            raise LegacyAttachmentSourceError(_SAFE_SOURCE_CHANGED)
        mapping, cleanup = persisted
        head = service.store.get_current_head(
            dataset.dataset_id,
            "attachment.ref",
            mapping.attachment_id,
        )
        projected = self._note_db.note_attachment_store.get(
            dataset.dataset_id,
            mapping.attachment_id,
        )
        if (
            mapping.note_id != candidate.note_id
            or cleanup.source_relative_path != candidate.relative_path
            or cleanup.source_blob_hash != candidate.sha256
            or cleanup.source_size_bytes != candidate.size_bytes
            or cleanup.source_modified_ns != candidate.modified_ns
            or head is None
            or head.adapter_version != 2
            or head.schema_version != 2
            or head.operation != "upsert"
            or projected is None
            or projected.source_kind != "legacy_bootstrap"
            or projected.note_id != candidate.note_id
            or projected.blob_hash != candidate.sha256
            or projected.size_bytes != candidate.size_bytes
        ):
            raise LegacyAttachmentSourceError(_SAFE_SOURCE_CHANGED)
        revision = head.object_revision
        if revision is None:
            raise LegacyAttachmentSourceError(_SAFE_SOURCE_CHANGED)
        binding = service.store.get_attachment_revision_binding(
            dataset.dataset_id,
            mapping.attachment_id,
            revision,
            owner_user_id=user_id,
        )
        if binding is None or binding.resolved_blob_id is None:
            raise LegacyAttachmentSourceError(_SAFE_SOURCE_CHANGED)
        blob = service.store.get_blob_object(
            dataset.dataset_id,
            blob_id=binding.resolved_blob_id,
            owner_user_id=user_id,
        )
        if (
            blob is None
            or blob.payload_hash != candidate.sha256
            or blob.size_bytes != candidate.size_bytes
            or service.blob_store is None
        ):
            raise LegacyAttachmentSourceError(_SAFE_SOURCE_CHANGED)
        service.blob_store.verify_blob(
            blob.storage_key,
            payload_hash=candidate.sha256,
            expected_size=candidate.size_bytes,
        )

    @staticmethod
    def _fail(
        service: SyncV2Service,
        dataset: SyncDataset,
        *,
        user_id: str,
        bootstrap_id: str,
        captured_count: int,
        expected_count: int,
        source_hash: str | None,
        error_code: str,
    ) -> SyncDataset:
        current = service.store.get_dataset(
            dataset.dataset_id,
            owner_user_id=user_id,
        ) or dataset
        metadata = current.metadata.get("notes_attachment_v2")
        current_hash = metadata.get("source_hash") if isinstance(metadata, Mapping) else None
        return service.store.transition_notes_attachment_bootstrap(
            dataset.dataset_id,
            owner_user_id=user_id,
            bootstrap_id=bootstrap_id,
            expected_state="initializing",
            state="failed",
            captured_count=min(captured_count, expected_count),
            expected_count=expected_count,
            source_hash=(current_hash if isinstance(current_hash, str) else source_hash),
            source_cursor=None,
            error_code=error_code,
        )


def _source_page(
    source: LegacyAttachmentSource,
    *,
    service: SyncV2Service,
    dataset: SyncDataset,
    user_id: str,
    bootstrap_id: str,
    cursor: _BootstrapCursor,
    limit: int,
) -> _SourcePage:
    """Read at most one note page and ``limit`` immutable candidates."""

    if not 1 <= limit <= _CANDIDATE_PAGE_SIZE:
        raise ValueError("Notes attachment source page limit must be 1..1000")
    candidates: list[LegacyAttachmentCandidate] = []
    last_note_id = cursor.note_id
    last_source_key: str | None = None
    note_budget = LEGACY_ATTACHMENT_NOTE_PAGE_LIMIT

    if cursor.note_id is not None and cursor.source_key_hash is not None:
        persisted = service.store.get_notes_attachment_bootstrap_source_by_hash(
            dataset.dataset_id,
            owner_user_id=user_id,
            bootstrap_id=bootstrap_id,
            source_key_hash=cursor.source_key_hash,
        )
        if persisted is None or persisted[0].note_id != cursor.note_id:
            raise LegacyAttachmentSourceError("notes_attachment_source_cursor_invalid")
        after_source_key = persisted[1].source_relative_path
        current_page = source.list_candidates(
            cursor.note_id,
            after_source_key=after_source_key,
            limit=limit,
        )
        candidates.extend(current_page)
        note_budget -= 1
        if current_page:
            last_source_key = current_page[-1].source_key
        if len(candidates) == limit:
            return _SourcePage(
                tuple(candidates),
                cursor.note_id,
                last_source_key,
                False,
            )
        last_source_key = None

    note_ids = source.list_note_ids(
        after_note_id=cursor.note_id,
        limit=max(1, note_budget),
    )
    for note_id in note_ids:
        last_note_id = note_id
        remaining = limit - len(candidates)
        current_page = source.list_candidates(
            note_id,
            after_source_key=None,
            limit=remaining,
        )
        candidates.extend(current_page)
        if current_page:
            last_source_key = current_page[-1].source_key
        else:
            last_source_key = None
        if len(candidates) == limit:
            return _SourcePage(
                tuple(candidates),
                last_note_id,
                last_source_key,
                False,
            )
    exhausted = len(note_ids) < max(1, note_budget)
    return _SourcePage(
        tuple(candidates),
        last_note_id,
        last_source_key,
        exhausted,
    )


def _initial_cursor(phase: str) -> _BootstrapCursor:
    """Create an empty bootstrap cursor for the requested processing phase."""

    return _BootstrapCursor(
        phase=phase,
        note_id=None,
        source_key_hash=None,
        rolling_hash=_EMPTY_SOURCE_HASH,
        processed_count=0,
    )


def _decode_cursor(
    value: object,
    *,
    captured_count: int,
    source_hash: object,
) -> _BootstrapCursor:
    if value is None:
        if captured_count != 0:
            raise LegacyAttachmentSourceError("notes_attachment_source_cursor_invalid")
        return _initial_cursor("verify" if isinstance(source_hash, str) else "capture")
    if not isinstance(value, str):
        raise LegacyAttachmentSourceError("notes_attachment_source_cursor_invalid")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LegacyAttachmentSourceError(
            "notes_attachment_source_cursor_invalid"
        ) from exc
    if not isinstance(decoded, dict) or set(decoded) != {
        "note_id",
        "phase",
        "processed_count",
        "rolling_hash",
        "source_key_hash",
    }:
        raise LegacyAttachmentSourceError("notes_attachment_source_cursor_invalid")
    cursor = _BootstrapCursor(
        phase=decoded["phase"],
        note_id=decoded["note_id"],
        source_key_hash=decoded["source_key_hash"],
        rolling_hash=decoded["rolling_hash"],
        processed_count=decoded["processed_count"],
    )
    if (
        cursor.phase not in {"capture", "verify"}
        or (cursor.note_id is not None and not isinstance(cursor.note_id, str))
        or (
            cursor.source_key_hash is not None
            and (
                not isinstance(cursor.source_key_hash, str)
                or not cursor.source_key_hash.startswith("sha256:")
                or len(cursor.source_key_hash) != 71
            )
        )
        or not isinstance(cursor.rolling_hash, str)
        or len(cursor.rolling_hash) != 64
        or any(character not in "0123456789abcdef" for character in cursor.rolling_hash)
        or isinstance(cursor.processed_count, bool)
        or not isinstance(cursor.processed_count, int)
        or cursor.processed_count < 0
        or (cursor.phase == "capture" and cursor.processed_count != captured_count)
        or (cursor.phase == "capture" and source_hash is not None)
        or (cursor.phase == "verify" and not isinstance(source_hash, str))
    ):
        raise LegacyAttachmentSourceError("notes_attachment_source_cursor_invalid")
    return cursor


def _encode_cursor(cursor: _BootstrapCursor) -> str:
    return json.dumps(
        {
            "note_id": cursor.note_id,
            "phase": cursor.phase,
            "processed_count": cursor.processed_count,
            "rolling_hash": cursor.rolling_hash,
            "source_key_hash": cursor.source_key_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _advanced_cursor(
    cursor: _BootstrapCursor,
    candidate: LegacyAttachmentCandidate,
) -> _BootstrapCursor:
    record = json.dumps(
        {
            "file_name": candidate.file_name,
            "metadata": candidate.metadata,
            "modified_ns": candidate.modified_ns,
            "note_id": candidate.note_id,
            "sha256": candidate.sha256,
            "size_bytes": candidate.size_bytes,
            "source_key": candidate.source_key,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    rolling = hashlib.sha256(
        bytes.fromhex(cursor.rolling_hash)
        + len(record).to_bytes(8, "big")
        + record
    ).hexdigest()
    return _BootstrapCursor(
        phase=cursor.phase,
        note_id=candidate.note_id,
        source_key_hash=_source_key_hash(candidate.source_key),
        rolling_hash=rolling,
        processed_count=cursor.processed_count + 1,
    )


def _cursor_after_empty_notes(
    cursor: _BootstrapCursor,
    note_id: str,
) -> _BootstrapCursor:
    return _BootstrapCursor(
        phase=cursor.phase,
        note_id=note_id,
        source_key_hash=None,
        rolling_hash=cursor.rolling_hash,
        processed_count=cursor.processed_count,
    )


def _source_key_hash(source_key: str) -> str:
    return "sha256:" + hashlib.sha256(source_key.encode("utf-8")).hexdigest()


def _content_type(candidate: LegacyAttachmentCandidate) -> str:
    supplied = candidate.metadata.get("content_type")
    guessed = mimetypes.guess_type(candidate.file_name, strict=False)[0]
    value = supplied if supplied is not None else guessed or "application/octet-stream"
    return validate_note_attachment_content_type(value)


def _available_file_name(
    note_db: CharactersRAGDB,
    dataset_id: str,
    candidate: LegacyAttachmentCandidate,
    *,
    attachment_id: str,
) -> str:
    existing = note_db.note_attachment_store.get(dataset_id, attachment_id)
    if existing is not None:
        if (
            existing.note_id != candidate.note_id
            or existing.original_file_name != candidate.file_name
            or existing.blob_hash != candidate.sha256
        ):
            raise LegacyAttachmentSourceError("notes_attachment_source_ambiguous")
        return existing.file_name
    base_name, base_key = canonicalize_note_attachment_file_name(candidate.file_name)
    occupied = _occupied_name_keys(note_db, dataset_id, candidate.note_id)
    if base_key not in occupied:
        return base_name
    suffixes = Path(base_name).suffixes
    extension = "".join(suffixes) if suffixes else ""
    stem = base_name[: -len(extension)] if extension else base_name
    for index in range(1, 1_000):
        suffix = f"-{index}"
        trimmed = stem[: max(1, 180 - len(extension) - len(suffix))]
        display, key = canonicalize_note_attachment_file_name(
            f"{trimmed}{suffix}{extension}"
        )
        if key not in occupied:
            return display
    raise LegacyAttachmentSourceError("notes_attachment_source_ambiguous")


def _occupied_name_keys(
    note_db: CharactersRAGDB,
    dataset_id: str,
    note_id: str,
) -> set[str]:
    occupied: set[str] = set()
    after_attachment_id: str | None = None
    while True:
        page = note_db.note_attachment_store.list_page(
            dataset_id,
            note_id,
            after_attachment_id=after_attachment_id,
            limit=200,
            state="live",
            include_deleted_note=True,
        )
        if not page:
            return occupied
        occupied.update(item.normalized_file_name for item in page)
        after_attachment_id = page[-1].attachment_id
        if len(page) < 200:
            return occupied


def _non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _safe_source_error(error_code: str) -> str:
    if error_code in {
        "notes_attachment_source_changed",
        "notes_attachment_source_too_large",
        "notes_attachment_source_ambiguous",
    }:
        return error_code
    if "too_large" in error_code:
        return _SAFE_SOURCE_TOO_LARGE
    return _SAFE_SOURCE_ERROR


__all__ = ["NotesAttachmentBootstrapInterrupted", "NotesAttachmentBootstrapper"]
