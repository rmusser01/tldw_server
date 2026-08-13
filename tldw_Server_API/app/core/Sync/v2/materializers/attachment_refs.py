from __future__ import annotations

"""Materialize Sync attachment references into canonical Notes metadata."""

from dataclasses import dataclass

from loguru import logger

from tldw_Server_API.app.core.DB_Management.chacha.note_attachment_store import (
    NoteAttachment,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)

from ..adapters import (
    AttachmentRefValidationError,
    extract_attachment_ref_metadata,
)
from ..attachment_refs_v2 import (
    AttachmentRefV2Payload,
    AttachmentRefV2TombstonePayload,
    AttachmentRefV2ValidationError,
    parse_attachment_ref_v2_payload,
)
from ..models import SyncEnvelope, SyncObjectState
from ..store import SyncV2Store
from .base import MaterializationResult


@dataclass(slots=True)
class AttachmentRefMaterializer:
    """Project v2 refs into Notes while preserving legacy v1 object state."""

    note_db: CharactersRAGDB | None = None
    domain: str = "attachment.ref"

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Project one accepted attachment-ref envelope into object state."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code="attachment_ref_projection_failed",
                message="Stored Sync envelope is missing a server cursor",
            )

        if envelope.adapter_version == 2:
            return self._apply_v2(envelope, store=store)
        return self._apply_legacy(envelope, store=store)

    def _apply_v2(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        if self.note_db is None:
            return _mark_failed(
                envelope,
                store,
                "attachment_ref_projection_failed",
                "Notes attachment projection is unavailable",
            )
        try:
            if envelope.schema_version != 2:
                raise AttachmentRefV2ValidationError(
                    "attachment.ref v2 schema version must be 2"
                )
            payload = parse_attachment_ref_v2_payload(
                envelope.operation,
                envelope.payload or envelope.payload_clear,
            )
            revision = _required_revision(envelope)
        except AttachmentRefV2ValidationError:
            return _mark_failed(
                envelope,
                store,
                "attachment_ref_v2_payload_invalid",
                "attachment.ref v2 payload validation failed",
            )

        current_state = store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        if _already_materialized(envelope, current_state):
            store.mark_envelope_apply_status(
                envelope.server_cursor or 0,
                apply_status="applied",
            )
            return MaterializationResult(status="applied")
        conflict_reason = _v2_state_conflict(envelope, current_state)
        if conflict_reason is not None:
            return _mark_v2_conflict(envelope, store, conflict_reason)

        projection = self.note_db.note_attachment_store
        existing = projection.get(envelope.dataset_id, envelope.object_id)
        desired_deleted = envelope.operation == "tombstone"
        if _is_exact_product_postcondition(
            existing,
            envelope=envelope,
            payload=payload,
            revision=revision,
            deleted=desired_deleted,
        ):
            _record_attachment_ref_state(
                store=store,
                envelope=envelope,
                object_revision=revision,
                object_hash=envelope.payload_hash or "",
                deleted=desired_deleted,
            )
            return MaterializationResult(status="applied")

        try:
            _validate_product_base(existing, envelope=envelope, payload=payload)
            if existing is None:
                projection.create(
                    dataset_id=envelope.dataset_id,
                    attachment_id=envelope.object_id,
                    note_id=str(payload.parent_object_id),
                    file_name=payload.file_name,
                    original_file_name=payload.original_file_name,
                    content_type=payload.content_type,
                    size_bytes=payload.size_bytes,
                    blob_hash=payload.blob_hash,
                    object_hash=envelope.payload_hash or "",
                    created_at=payload.created_at,
                    last_modified=payload.last_modified,
                    created_by=payload.created_by,
                    source_kind=(
                        "legacy_bootstrap"
                        if envelope.routing_metadata.get("bootstrap_capture") is True
                        else "sync"
                    ),
                )
            elif envelope.operation == "tombstone":
                if not isinstance(payload, AttachmentRefV2TombstonePayload):
                    raise AttachmentRefV2ValidationError(
                        "attachment.ref tombstone payload is invalid"
                    )
                projection.tombstone(
                    dataset_id=envelope.dataset_id,
                    attachment_id=envelope.object_id,
                    expected_version=envelope.base_object_revision or 0,
                    expected_object_hash=envelope.base_object_hash or "",
                    object_hash=envelope.payload_hash or "",
                    last_modified=payload.last_modified,
                    deleted_at=payload.deleted_at,
                    delete_reason=payload.reason,
                )
            elif envelope.routing_metadata.get("restore_intent") is True:
                projection.restore(
                    dataset_id=envelope.dataset_id,
                    attachment_id=envelope.object_id,
                    expected_version=envelope.base_object_revision or 0,
                    expected_object_hash=envelope.base_object_hash or "",
                    object_hash=envelope.payload_hash or "",
                    last_modified=payload.last_modified,
                )
            else:
                projection.compare_and_set(
                    dataset_id=envelope.dataset_id,
                    attachment_id=envelope.object_id,
                    expected_version=envelope.base_object_revision or 0,
                    expected_object_hash=envelope.base_object_hash or "",
                    file_name=payload.file_name,
                    content_type=payload.content_type,
                    size_bytes=payload.size_bytes,
                    blob_hash=payload.blob_hash,
                    object_hash=envelope.payload_hash or "",
                    last_modified=payload.last_modified,
                )
        except ConflictError:
            return _mark_v2_conflict(envelope, store, "product_state_conflict")
        except Exception as exc:  # noqa: BLE001 - failures must remain replayable.
            logger.warning(
                "Failed attachment.ref v2 projection for cursor {}: {}",
                envelope.server_cursor,
                type(exc).__name__,
            )
            return _mark_failed(
                envelope,
                store,
                "attachment_ref_projection_failed",
                "Notes attachment projection failed",
            )

        projected = projection.get(envelope.dataset_id, envelope.object_id)
        if not _is_exact_product_postcondition(
            projected,
            envelope=envelope,
            payload=payload,
            revision=revision,
            deleted=desired_deleted,
        ):
            return _mark_failed(
                envelope,
                store,
                "attachment_ref_projection_failed",
                "Notes attachment projection did not reach the requested state",
            )
        _record_attachment_ref_state(
            store=store,
            envelope=envelope,
            object_revision=revision,
            object_hash=envelope.payload_hash or "",
            deleted=desired_deleted,
        )
        return MaterializationResult(status="applied")

    def _apply_legacy(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Keep adapter-v1 replay compatibility out of the product registry."""

        try:
            metadata = extract_attachment_ref_metadata(envelope)
        except AttachmentRefValidationError as exc:
            logger.warning(
                "Invalid attachment.ref envelope {} for object {}: {}",
                envelope.client_envelope_id,
                envelope.object_id,
                exc.error_code,
            )
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code=exc.error_code,
                apply_error_message=str(exc),
            )
            return MaterializationResult(
                status="failed",
                error_code=exc.error_code,
                message=str(exc),
            )

        current_state = store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        if envelope.operation == "upsert":
            if current_state is not None:
                if current_state.deleted:
                    store.mark_envelope_apply_status(
                        envelope.server_cursor,
                        apply_status="conflict",
                        apply_error_code="attachment_ref_tombstoned",
                        apply_error_message=(
                            "attachment.ref upsert cannot resurrect a tombstoned reference"
                        ),
                    )
                    return _tombstone_conflict_result(envelope, current_state=current_state)
                if current_state.object_hash == metadata.payload_hash:
                    store.mark_envelope_apply_status(
                        envelope.server_cursor,
                        apply_status="applied",
                    )
                    return MaterializationResult(status="applied")
                store.mark_envelope_apply_status(
                    envelope.server_cursor,
                    apply_status="conflict",
                    apply_error_code="attachment_ref_hash_mismatch",
                    apply_error_message=(
                        "attachment.ref stable attachment ID was reused with a " "different payload hash"
                    ),
                )
                return _conflict_result(envelope, current_state=current_state)

            _record_attachment_ref_state(
                store=store,
                envelope=envelope,
                object_revision=_next_object_revision(envelope, current_state),
                object_hash=metadata.payload_hash,
                deleted=False,
            )
            return MaterializationResult(status="applied")

        if envelope.operation == "tombstone":
            object_hash = current_state.object_hash if current_state is not None else metadata.payload_hash
            _record_attachment_ref_state(
                store=store,
                envelope=envelope,
                object_revision=_next_object_revision(envelope, current_state),
                object_hash=object_hash,
                deleted=True,
            )
            return MaterializationResult(status="applied")

        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="failed",
            apply_error_code="attachment_ref_projection_failed",
            apply_error_message=f"Unsupported attachment.ref operation: {envelope.operation}",
        )
        return MaterializationResult(
            status="failed",
            error_code="attachment_ref_projection_failed",
            message=f"Unsupported attachment.ref operation: {envelope.operation}",
        )


def _required_revision(envelope: SyncEnvelope) -> int:
    revision = envelope.object_revision
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise AttachmentRefV2ValidationError(
            "attachment.ref v2 object revision must be positive"
        )
    return revision


def _already_materialized(
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
) -> bool:
    return bool(
        current_state is not None
        and current_state.latest_server_cursor == envelope.server_cursor
        and current_state.object_revision == envelope.object_revision
        and current_state.object_hash == (envelope.payload_hash or "")
        and current_state.deleted == (envelope.operation == "tombstone")
    )


def _v2_state_conflict(
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
) -> str | None:
    has_base = any(
        value is not None
        for value in (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
        )
    )
    restore = envelope.routing_metadata.get("restore_intent") is True
    if current_state is None:
        if (
            has_base
            or restore
            or envelope.operation != "upsert"
            or envelope.object_revision != 1
        ):
            return "missing_server_object"
        return None
    if (
        envelope.base_server_cursor != current_state.latest_server_cursor
        or envelope.base_object_revision != current_state.object_revision
        or envelope.base_object_hash != current_state.object_hash
        or envelope.object_revision != current_state.object_revision + 1
    ):
        return "stale_base_state"
    if current_state.deleted and not restore:
        return "restore_intent_required"
    if restore and not current_state.deleted:
        return "restore_target_not_deleted"
    return None


def _validate_product_base(
    existing: NoteAttachment | None,
    *,
    envelope: SyncEnvelope,
    payload: AttachmentRefV2Payload,
) -> None:
    if existing is None:
        if envelope.object_revision != 1:
            raise ConflictError("Attachment product base is missing")
        return
    if (
        existing.version != envelope.base_object_revision
        or existing.object_hash != envelope.base_object_hash
        or existing.note_id != str(payload.parent_object_id)
        or existing.original_file_name != payload.original_file_name
        or existing.created_at != payload.created_at
        or existing.created_by != payload.created_by
    ):
        raise ConflictError("Attachment product base conflicts")
    restore = envelope.routing_metadata.get("restore_intent") is True
    if restore and (
        existing.file_name != payload.file_name
        or existing.content_type != payload.content_type
        or existing.size_bytes != payload.size_bytes
        or existing.blob_hash != payload.blob_hash
    ):
        raise ConflictError("Attachment restore metadata conflicts")


def _is_exact_product_postcondition(
    attachment: NoteAttachment | None,
    *,
    envelope: SyncEnvelope,
    payload: AttachmentRefV2Payload,
    revision: int,
    deleted: bool,
) -> bool:
    if attachment is None:
        return False
    return bool(
        attachment.attachment_id == envelope.object_id
        and attachment.note_id == str(payload.parent_object_id)
        and attachment.file_name == payload.file_name
        and attachment.original_file_name == payload.original_file_name
        and attachment.content_type == payload.content_type
        and attachment.size_bytes == payload.size_bytes
        and attachment.blob_hash == payload.blob_hash
        and attachment.object_hash == (envelope.payload_hash or "")
        and attachment.version == revision
        and attachment.deleted == deleted
        and attachment.created_at == payload.created_at
        and attachment.last_modified == payload.last_modified
        and attachment.created_by == payload.created_by
        and (
            not isinstance(payload, AttachmentRefV2TombstonePayload)
            or (
                attachment.deleted_at == payload.deleted_at
                and attachment.delete_reason == payload.reason
            )
        )
    )


def _mark_v2_conflict(
    envelope: SyncEnvelope,
    store: SyncV2Store,
    reason: str,
) -> MaterializationResult:
    message = "Notes attachment base state does not match the current projection"
    store.mark_envelope_apply_status(
        envelope.server_cursor or 0,
        apply_status="conflict",
        apply_error_code="attachment_ref_product_conflict",
        apply_error_message=message,
    )
    return MaterializationResult(
        status="conflict",
        conflict_type="attachment_ref_product_conflict",
        message=message,
        metadata={"attachment_id": envelope.object_id, "reason": reason},
    )


def _mark_failed(
    envelope: SyncEnvelope,
    store: SyncV2Store,
    error_code: str,
    message: str,
) -> MaterializationResult:
    store.mark_envelope_apply_status(
        envelope.server_cursor or 0,
        apply_status="failed",
        apply_error_code=error_code,
        apply_error_message=message,
    )
    return MaterializationResult(
        status="failed",
        error_code=error_code,
        message=message,
    )


def _record_attachment_ref_state(
    *,
    store: SyncV2Store,
    envelope: SyncEnvelope,
    object_revision: int,
    object_hash: str,
    deleted: bool,
) -> None:
    store.upsert_object_state(
        SyncObjectState(
            dataset_id=envelope.dataset_id,
            domain=envelope.domain,
            object_id=envelope.object_id,
            object_revision=object_revision,
            object_hash=object_hash,
            latest_server_cursor=envelope.server_cursor or 0,
            deleted=deleted,
        )
    )
    store.mark_envelope_apply_status(envelope.server_cursor or 0, apply_status="applied")


def _next_object_revision(
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
) -> int:
    if envelope.object_revision is not None:
        return envelope.object_revision
    if current_state is None:
        return 1
    return current_state.object_revision + 1


def _conflict_result(
    envelope: SyncEnvelope,
    *,
    current_state: SyncObjectState,
) -> MaterializationResult:
    return MaterializationResult(
        status="conflict",
        conflict_type="attachment_ref_hash_mismatch",
        message=("attachment.ref stable attachment ID was reused with a different " "payload hash"),
        metadata={
            "attachment_id": envelope.object_id,
            "incoming_payload_hash": envelope.payload_hash or "",
            "server_object_hash": current_state.object_hash,
            "server_object_revision": current_state.object_revision,
            "server_cursor": current_state.latest_server_cursor,
            "server_deleted": current_state.deleted,
        },
    )


def _tombstone_conflict_result(
    envelope: SyncEnvelope,
    *,
    current_state: SyncObjectState,
) -> MaterializationResult:
    return MaterializationResult(
        status="conflict",
        conflict_type="attachment_ref_tombstoned",
        message="attachment.ref upsert cannot resurrect a tombstoned reference",
        metadata={
            "attachment_id": envelope.object_id,
            "incoming_payload_hash": envelope.payload_hash or "",
            "server_object_hash": current_state.object_hash,
            "server_object_revision": current_state.object_revision,
            "server_cursor": current_state.latest_server_cursor,
            "server_deleted": current_state.deleted,
        },
    )


__all__ = ["AttachmentRefMaterializer"]
