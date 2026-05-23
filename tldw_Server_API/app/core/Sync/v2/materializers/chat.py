from __future__ import annotations

"""Materialize Sync v2 chat envelopes into ChaChaNotes conversations and messages."""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

from ..models import SyncEnvelope, SyncObjectState
from ..store import SyncV2Store
from .base import MaterializationResult


@dataclass(slots=True)
class ChatConversationMaterializer:
    """Apply `chat.conversation` upserts and tombstones to ChaChaNotes."""

    note_db: CharactersRAGDB
    domain: str = "chat.conversation"

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Project one accepted conversation envelope and record apply status."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code="chat_projection_failed",
                message="Stored Sync envelope is missing a server cursor",
            )

        current_state = store.get_object_state(envelope.dataset_id, envelope.domain, envelope.object_id)
        if _is_already_materialized(envelope, current_state):
            return _complete_already_materialized(
                envelope,
                store=store,
                error_code="chat_projection_failed",
            )

        conflict = _detect_whole_object_conflict(
            envelope,
            current_state,
            conflict_message="chat.conversation base state does not match the current server projection",
        )
        if conflict is not None:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="conflict",
                apply_error_code="whole_object_conflict",
                apply_error_message=conflict.message,
            )
            return conflict

        object_revision = _next_object_revision(envelope, current_state)
        object_hash = envelope.payload_hash or ""
        try:
            if envelope.operation == "upsert":
                payload = _conversation_payload(envelope.payload)
                self.note_db.upsert_conversation_from_sync(
                    conversation_id=envelope.object_id,
                    title=payload.get("title"),
                    sync_client_id=_projection_client_id(envelope, payload),
                    object_revision=object_revision,
                    object_hash=object_hash,
                    root_id=payload.get("root_id"),
                    assistant_kind=payload.get("assistant_kind"),
                    assistant_id=payload.get("assistant_id"),
                    character_id=payload.get("character_id"),
                    persona_memory_mode=payload.get("persona_memory_mode"),
                    state=payload.get("state"),
                    topic_label=payload.get("topic_label"),
                    cluster_id=payload.get("cluster_id"),
                    source=payload.get("source"),
                    external_ref=payload.get("external_ref"),
                    rating=payload.get("rating"),
                    scope_type=payload.get("scope_type"),
                    workspace_id=payload.get("workspace_id"),
                )
                deleted = False
            elif envelope.operation == "tombstone":
                self.note_db.tombstone_conversation_from_sync(
                    conversation_id=envelope.object_id,
                    sync_client_id=_projection_client_id(envelope, envelope.payload),
                    object_revision=object_revision,
                    object_hash=object_hash,
                )
                deleted = True
            else:
                raise ValueError(f"Unsupported chat.conversation operation: {envelope.operation}")

            _record_applied_state(
                store=store,
                envelope=envelope,
                object_revision=object_revision,
                object_hash=object_hash,
                deleted=deleted,
            )
        except Exception as exc:
            logger.warning(
                "Failed to materialize chat.conversation envelope {} for object {}: {}",
                envelope.client_envelope_id,
                envelope.object_id,
                type(exc).__name__,
            )
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="chat_projection_failed",
                apply_error_message=_safe_error_message(exc),
            )
            return MaterializationResult(
                status="failed",
                error_code="chat_projection_failed",
                message=_safe_error_message(exc),
            )

        return MaterializationResult(status="applied")


@dataclass(slots=True)
class ChatMessageMaterializer:
    """Apply `chat.message` appends and tombstones to ChaChaNotes."""

    note_db: CharactersRAGDB
    domain: str = "chat.message"

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Project one accepted message envelope and record apply status."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code="chat_projection_failed",
                message="Stored Sync envelope is missing a server cursor",
            )

        current_state = store.get_object_state(envelope.dataset_id, envelope.domain, envelope.object_id)
        if envelope.apply_status == "conflict":
            return _message_conflict_result(envelope, current_state=current_state)
        if _is_already_materialized(envelope, current_state):
            return _complete_already_materialized(
                envelope,
                store=store,
                error_code="chat_projection_failed",
            )

        object_revision = _next_object_revision(envelope, current_state)
        object_hash = envelope.payload_hash or ""
        try:
            if envelope.operation == "append":
                if current_state is not None and current_state.deleted:
                    result = _message_conflict_result(
                        envelope,
                        current_state=current_state,
                        conflict_type="message_deleted_conflict",
                        message="chat.message append cannot resurrect a tombstoned message",
                    )
                    store.mark_envelope_apply_status(
                        envelope.server_cursor,
                        apply_status="conflict",
                        apply_error_code="message_deleted_conflict",
                        apply_error_message=result.message,
                    )
                    return result
                payload = _message_payload(envelope.payload)
                divergent_stable_id = current_state is not None and current_state.object_hash != object_hash
                append_result = self.note_db.append_message_from_sync(
                    stable_message_id=envelope.object_id,
                    conversation_id=payload["conversation_id"],
                    sender=payload["sender"],
                    content=payload.get("content"),
                    timestamp=payload.get("timestamp"),
                    sync_client_id=_projection_client_id(envelope, payload),
                    object_revision=object_revision,
                    payload_hash=object_hash,
                    parent_message_id=payload.get("parent_message_id"),
                    ranking=payload.get("ranking"),
                    projection_message_id=_conflict_projection_id(envelope) if divergent_stable_id else None,
                )
                if divergent_stable_id or append_result["conflict"]:
                    result = _message_conflict_result(
                        envelope,
                        current_state=current_state,
                        projection_message_id=str(append_result["message_id"]),
                    )
                    store.mark_envelope_apply_status(
                        envelope.server_cursor,
                        apply_status="conflict",
                        apply_error_code="message_stable_id_conflict",
                        apply_error_message=result.message,
                    )
                    return result
                deleted = False
            elif envelope.operation == "tombstone":
                conflict = _detect_message_tombstone_conflict(envelope, current_state)
                if conflict is not None:
                    store.mark_envelope_apply_status(
                        envelope.server_cursor,
                        apply_status="conflict",
                        apply_error_code=conflict.conflict_type or "message_base_conflict",
                        apply_error_message=conflict.message,
                    )
                    return conflict
                self.note_db.tombstone_message_from_sync(
                    stable_message_id=envelope.object_id,
                    sync_client_id=_projection_client_id(envelope, envelope.payload),
                    object_revision=object_revision,
                    object_hash=current_state.object_hash if current_state is not None else object_hash,
                )
                deleted = True
            else:
                raise ValueError(f"Unsupported chat.message operation: {envelope.operation}")

            _record_applied_state(
                store=store,
                envelope=envelope,
                object_revision=object_revision,
                object_hash=object_hash,
                deleted=deleted,
            )
        except Exception as exc:
            logger.warning(
                "Failed to materialize chat.message envelope {} for object {}: {}",
                envelope.client_envelope_id,
                envelope.object_id,
                type(exc).__name__,
            )
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="chat_projection_failed",
                apply_error_message=_safe_error_message(exc),
            )
            return MaterializationResult(
                status="failed",
                error_code="chat_projection_failed",
                message=_safe_error_message(exc),
            )

        return MaterializationResult(status="applied")


def _record_applied_state(
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


def _complete_already_materialized(
    envelope: SyncEnvelope,
    *,
    store: SyncV2Store,
    error_code: str,
) -> MaterializationResult:
    try:
        store.mark_envelope_apply_status(envelope.server_cursor or 0, apply_status="applied")
    except Exception as exc:
        logger.warning(
            "Failed to complete chat apply status for envelope {} on object {}: {}",
            envelope.client_envelope_id,
            envelope.object_id,
            type(exc).__name__,
        )
        store.mark_envelope_apply_status(
            envelope.server_cursor or 0,
            apply_status="failed",
            apply_error_code=error_code,
            apply_error_message=_safe_error_message(exc),
        )
        return MaterializationResult(
            status="failed",
            error_code=error_code,
            message=_safe_error_message(exc),
        )
    return MaterializationResult(status="applied")


def _detect_whole_object_conflict(
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
    *,
    conflict_message: str,
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
                conflict_type="whole_object_conflict",
                message=conflict_message,
            )
        return None
    if not has_base:
        return _conflict_result(
            reason="missing_base_state",
            envelope=envelope,
            current_state=current_state,
            conflict_type="whole_object_conflict",
            message=conflict_message,
        )
    if envelope.operation == "upsert" and current_state.deleted:
        return _conflict_result(
            reason="server_object_deleted",
            envelope=envelope,
            current_state=current_state,
            conflict_type="whole_object_conflict",
            message=conflict_message,
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
            conflict_type="whole_object_conflict",
            message=conflict_message,
        )
    return None


def _detect_message_tombstone_conflict(
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
        return _conflict_result(
            reason="missing_server_message",
            envelope=envelope,
            current_state=None,
            conflict_type="message_base_conflict",
            message="chat.message tombstone requires an existing server message base state",
        )
    if not has_base:
        return _conflict_result(
            reason="missing_base_state",
            envelope=envelope,
            current_state=current_state,
            conflict_type="message_base_conflict",
            message="chat.message tombstone requires base server cursor, revision, and hash",
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
            conflict_type="message_base_conflict",
            message="chat.message tombstone base state does not match the current server projection",
        )
    return None


def _message_conflict_result(
    envelope: SyncEnvelope,
    *,
    current_state: SyncObjectState | None,
    projection_message_id: str | None = None,
    conflict_type: str = "message_stable_id_conflict",
    message: str = "chat.message stable message ID was reused with a different payload hash",
) -> MaterializationResult:
    metadata: dict[str, object] = {
        "stable_message_id": envelope.object_id,
        "incoming_payload_hash": envelope.payload_hash or "",
    }
    if projection_message_id is not None:
        metadata["projection_message_id"] = projection_message_id
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
        conflict_type=conflict_type,
        message=message,
        metadata=metadata,
    )


def _conflict_result(
    *,
    reason: str,
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
    conflict_type: str,
    message: str,
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
        conflict_type=conflict_type,
        message=message,
        metadata=metadata,
    )


def _conversation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    assistant_kind = payload.get("assistant_kind")
    assistant_id = payload.get("assistant_id")
    character_id = payload.get("character_id")
    if assistant_kind is None and assistant_id is None and character_id is None:
        assistant_kind = "persona"
        assistant_id = "sync-v2"
    return {**payload, "assistant_kind": assistant_kind, "assistant_id": assistant_id}


def _message_payload(payload: dict[str, Any]) -> dict[str, Any]:
    conversation_id = payload.get("conversation_id")
    sender = payload.get("sender")
    content = payload.get("content", payload.get("body", ""))
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("chat.message payload requires a non-empty conversation_id")
    if not isinstance(sender, str) or not sender.strip():
        raise ValueError("chat.message payload requires a non-empty sender")
    if not isinstance(content, str):
        raise ValueError("chat.message payload content must be a string")
    return {
        **payload,
        "conversation_id": conversation_id,
        "sender": sender,
        "content": content,
    }


def _projection_client_id(envelope: SyncEnvelope, payload: dict[str, Any]) -> str:
    if envelope.routing_metadata.get("origin") == "server":
        client_id = payload.get("client_id") or payload.get("owner_user_id")
        if isinstance(client_id, str) and client_id.strip():
            return client_id.strip()
    return envelope.device_id or "sync-v2"


def _next_object_revision(envelope: SyncEnvelope, current_state: SyncObjectState | None) -> int:
    if envelope.object_revision is not None:
        return envelope.object_revision
    if current_state is None:
        return 1
    return current_state.object_revision + 1


def _is_already_materialized(
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
) -> bool:
    if envelope.server_cursor is None or current_state is None:
        return False
    if current_state.latest_server_cursor != envelope.server_cursor:
        return False
    if envelope.object_revision is not None and current_state.object_revision != envelope.object_revision:
        return False
    if current_state.object_hash != (envelope.payload_hash or ""):
        return False
    return current_state.deleted == (envelope.operation == "tombstone")


def _conflict_projection_id(envelope: SyncEnvelope) -> str:
    return f"{envelope.object_id}__sync_conflict__{envelope.server_cursor}"


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message[:200] if message else type(exc).__name__


__all__ = ["ChatConversationMaterializer", "ChatMessageMaterializer"]
