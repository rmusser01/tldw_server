from __future__ import annotations

"""Materialize Sync v2 notes envelopes into ChaChaNotes."""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

from ..models import SyncEnvelope, SyncObjectState, validate_notes_note_upsert_payload
from ..store import SyncV2Store
from .base import MaterializationResult

_INGESTION_EXPECTED_VERSION_KEY = "notes_ingestion_expected_product_version"
_SERVER_ORIGIN_DEVICE_ID = "server-origin"


def _trusted_ingestion_expected_version(
    envelope: SyncEnvelope,
    note_db: CharactersRAGDB,
) -> int | None:
    """Return an owner-bound local ingestion projection precondition."""

    routing = envelope.routing_metadata
    if (
        envelope.domain != "notes.note"
        or envelope.operation != "upsert"
        or envelope.device_id != _SERVER_ORIGIN_DEVICE_ID
        or routing.get("source") != "notes-ingestion"
        or routing.get("origin") != "server"
        or routing.get("server_device_id") != _SERVER_ORIGIN_DEVICE_ID
        or routing.get("server_owner_user_id") != str(note_db.client_id)
    ):
        return None
    expected_version = routing.get(_INGESTION_EXPECTED_VERSION_KEY)
    if (
        isinstance(expected_version, bool)
        or not isinstance(expected_version, int)
        or expected_version < 0
    ):
        return None
    return expected_version


@dataclass(slots=True)
class NotesMaterializer:
    """Apply `notes.note` upserts and tombstones to the ChaChaNotes note store."""

    note_db: CharactersRAGDB
    domain: str = "notes.note"

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Project one accepted note envelope and record apply status."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code="notes_projection_failed",
                message="Stored Sync envelope is missing a server cursor",
            )

        current_state = store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        if self._is_already_materialized(envelope, current_state):
            try:
                store.mark_envelope_apply_status(
                    envelope.server_cursor,
                    apply_status="applied",
                )
            except Exception as exc:  # noqa: BLE001 - projection state must record every backend failure
                logger.warning(
                    "Failed to complete notes.note apply status for envelope {} "
                    "on object {}: {}",
                    envelope.client_envelope_id,
                    envelope.object_id,
                    type(exc).__name__,
                )
                store.mark_envelope_apply_status(
                    envelope.server_cursor,
                    apply_status="failed",
                    apply_error_code="notes_projection_failed",
                    apply_error_message=_safe_error_message(exc),
                )
                return MaterializationResult(
                    status="failed",
                    error_code="notes_projection_failed",
                    message=_safe_error_message(exc),
                )
            return MaterializationResult(status="applied")

        conflict = self._detect_conflict(envelope, current_state)
        if conflict is not None:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="conflict",
                apply_error_code="whole_object_conflict",
                apply_error_message=conflict.message,
            )
            return conflict

        object_revision = self._next_object_revision(envelope, current_state)
        object_hash = envelope.payload_hash or ""
        try:
            if envelope.operation == "upsert":
                payload = _note_payload(envelope.payload)
                expected_product_version = _trusted_ingestion_expected_version(
                    envelope,
                    self.note_db,
                )
                projection_timestamp = None
                if expected_product_version is not None:
                    if object_revision != expected_product_version + 1:
                        raise ValueError(
                            "Trusted ingestion product revision is inconsistent"
                        )
                    projection_timestamp = (
                        envelope.server_timestamp or envelope.received_at_server
                    )
                    if not projection_timestamp:
                        raise ValueError(
                            "Trusted ingestion envelope is missing a server timestamp"
                        )
                self.note_db.upsert_note_from_sync(
                    note_id=envelope.object_id,
                    title=payload["title"],
                    content=payload["content"],
                    conversation_id=payload.get("conversation_id"),
                    message_id=payload.get("message_id"),
                    sync_client_id=_projection_client_id(self.note_db),
                    object_revision=object_revision,
                    object_hash=object_hash,
                    expected_product_version=expected_product_version,
                    projection_timestamp=projection_timestamp,
                    semantic_dataset_id=envelope.dataset_id,
                )
                deleted = False
            elif envelope.operation == "tombstone":
                self.note_db.tombstone_note_from_sync(
                    note_id=envelope.object_id,
                    sync_client_id=_projection_client_id(self.note_db),
                    object_revision=object_revision,
                    object_hash=object_hash,
                    semantic_dataset_id=envelope.dataset_id,
                )
                deleted = True
            else:
                raise ValueError(f"Unsupported notes.note operation: {envelope.operation}")

            store.upsert_object_state(
                SyncObjectState(
                    dataset_id=envelope.dataset_id,
                    domain=envelope.domain,
                    object_id=envelope.object_id,
                    object_revision=object_revision,
                    object_hash=object_hash,
                    latest_server_cursor=envelope.server_cursor,
                    deleted=deleted,
                )
            )
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="applied",
            )
        except Exception as exc:  # noqa: BLE001 - projection failures are persisted for replay
            logger.warning(
                "Failed to materialize notes.note envelope {} for object {}: {}",
                envelope.client_envelope_id,
                envelope.object_id,
                type(exc).__name__,
            )
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="notes_projection_failed",
                apply_error_message=_safe_error_message(exc),
            )
            return MaterializationResult(
                status="failed",
                error_code="notes_projection_failed",
                message=_safe_error_message(exc),
            )

        return MaterializationResult(status="applied")

    def _detect_conflict(
        self,
        envelope: SyncEnvelope,
        current_state: SyncObjectState | None,
    ) -> MaterializationResult | None:
        base_values = (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
        )
        has_base = all(value is not None for value in base_values)
        restore_requested = (
            envelope.operation == "upsert"
            and envelope.routing_metadata.get("restore_intent") is True
        )
        if current_state is None:
            if restore_requested or any(value is not None for value in base_values):
                return _conflict_result(
                    reason="missing_server_object",
                    envelope=envelope,
                    current_state=None,
                )
            return None

        if not has_base:
            return _conflict_result(
                reason="missing_base_state",
                envelope=envelope,
                current_state=current_state,
            )
        base_matches = (
            envelope.base_server_cursor == current_state.latest_server_cursor
            and envelope.base_object_revision == current_state.object_revision
            and envelope.base_object_hash == current_state.object_hash
        )
        if restore_requested and not current_state.deleted:
            return _conflict_result(
                reason="restore_target_not_deleted",
                envelope=envelope,
                current_state=current_state,
            )
        if restore_requested and current_state.deleted and base_matches:
            return None
        if envelope.operation == "upsert" and current_state.deleted:
            return _conflict_result(
                reason="server_object_deleted",
                envelope=envelope,
                current_state=current_state,
            )
        if not base_matches:
            return _conflict_result(
                reason="stale_base_state",
                envelope=envelope,
                current_state=current_state,
            )
        return None

    @staticmethod
    def _next_object_revision(
        envelope: SyncEnvelope,
        current_state: SyncObjectState | None,
    ) -> int:
        if envelope.object_revision is not None:
            return envelope.object_revision
        if current_state is None:
            return 1
        return current_state.object_revision + 1

    @staticmethod
    def _is_already_materialized(
        envelope: SyncEnvelope,
        current_state: SyncObjectState | None,
    ) -> bool:
        if envelope.server_cursor is None or current_state is None:
            return False
        if current_state.latest_server_cursor != envelope.server_cursor:
            return False
        if (
            envelope.object_revision is not None
            and current_state.object_revision != envelope.object_revision
        ):
            return False
        if current_state.object_hash != (envelope.payload_hash or ""):
            return False
        return current_state.deleted == (envelope.operation == "tombstone")


def _note_payload(payload: dict[str, Any]) -> dict[str, str | None]:
    return validate_notes_note_upsert_payload(payload)


def _projection_client_id(note_db: CharactersRAGDB) -> str:
    return str(note_db.client_id)


def _conflict_result(
    *,
    reason: str,
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
) -> MaterializationResult:
    metadata: dict[str, object] = {
        "reason": reason,
        "client_base_object_revision": envelope.base_object_revision,
        "client_base_object_hash": envelope.base_object_hash,
        "client_base_server_cursor": envelope.base_server_cursor,
    }
    if current_state is not None:
        metadata.update(
            {
                "server_object_revision": current_state.object_revision,
                "server_object_hash": current_state.object_hash,
                "server_cursor": current_state.latest_server_cursor,
                "server_deleted": current_state.deleted,
            }
        )
    return MaterializationResult(
        status="conflict",
        conflict_type="whole_object_conflict",
        message="notes.note base state does not match the current server projection",
        metadata=metadata,
    )


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message[:200] if message else type(exc).__name__


__all__ = ["NotesMaterializer"]
