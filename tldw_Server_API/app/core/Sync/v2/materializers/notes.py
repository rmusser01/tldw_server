from __future__ import annotations

"""Materialize Sync v2 notes envelopes into ChaChaNotes."""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

from ..models import SyncEnvelope, SyncObjectState
from ..store import SyncV2Store
from .base import MaterializationResult


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
            except Exception as exc:
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
                self.note_db.upsert_note_from_sync(
                    note_id=envelope.object_id,
                    title=payload["title"],
                    content=payload["content"],
                    conversation_id=payload.get("conversation_id"),
                    message_id=payload.get("message_id"),
                    sync_client_id=envelope.device_id or "sync-v2",
                    object_revision=object_revision,
                    object_hash=object_hash,
                )
                deleted = False
            elif envelope.operation == "tombstone":
                self.note_db.tombstone_note_from_sync(
                    note_id=envelope.object_id,
                    sync_client_id=envelope.device_id or "sync-v2",
                    object_revision=object_revision,
                    object_hash=object_hash,
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
        except Exception as exc:
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
        if current_state is None:
            if any(value is not None for value in base_values):
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
        if envelope.operation == "upsert" and current_state.deleted:
            return _conflict_result(
                reason="server_object_deleted",
                envelope=envelope,
                current_state=current_state,
            )
        if (
            envelope.base_server_cursor != current_state.latest_server_cursor
            or envelope.base_object_revision != current_state.object_revision
            or envelope.base_object_hash != current_state.object_hash
        ):
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
    title = payload.get("title")
    content = payload.get("content", payload.get("body", ""))
    if not isinstance(title, str) or not title.strip():
        raise ValueError("notes.note payload requires a non-empty title")
    if not isinstance(content, str):
        raise ValueError("notes.note payload content must be a string")
    conversation_id = payload.get("conversation_id")
    message_id = payload.get("message_id")
    return {
        "title": title,
        "content": content,
        "conversation_id": str(conversation_id) if conversation_id is not None else None,
        "message_id": str(message_id) if message_id is not None else None,
    }


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
