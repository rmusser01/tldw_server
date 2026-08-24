"""Materialize immutable ``notes.task_activity`` envelopes into task events."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.exceptions import NotesTaskContractError

from ..models import SyncEnvelope, SyncObjectState
from ..notes_task_contract import (
    NotesTaskActivityTombstoneV1,
    NotesTaskActivityV1,
    notes_task_activity_object_hash,
    parse_notes_task_activity_tombstone_v1,
    parse_notes_task_activity_v1,
)
from ..store import SyncV2Store
from .base import MaterializationResult

_ERROR_CODE = "notes_task_activity_projection_failed"
_CONFLICT_TYPE = "notes_task_activity_product_conflict"


@dataclass(frozen=True, slots=True)
class _ParsedActivityEnvelope:
    payload: NotesTaskActivityV1 | NotesTaskActivityTombstoneV1
    original: NotesTaskActivityV1


@dataclass(slots=True)
class NotesTaskActivityMaterializer:
    """Apply accepted dormant activity envelopes with immutable replay checks."""

    note_db: CharactersRAGDB
    domain: str = "notes.task_activity"

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Commit one product event, then repair Sync projection bookkeeping."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code=_ERROR_CODE,
                message="Stored notes.task_activity envelope is missing a server cursor",
            )
        try:
            parsed = _parse_envelope(
                envelope,
                store=store,
                owner_user_id=str(self.note_db.client_id),
            )
        except NotesTaskContractError:
            return _mark_failed(
                envelope,
                store,
                "notes.task_activity envelope validation failed",
            )

        task_store = self.note_db.task_store
        if task_store.verify_sync_task_activity_postcondition(
            owner_user_id=str(self.note_db.client_id),
            dataset_id=envelope.dataset_id,
            payload=parsed.payload,
            original_payload=parsed.original,
            sync_revision=int(envelope.object_revision or 0),
            sync_object_hash=envelope.payload_hash or "",
            sync_server_cursor=envelope.server_cursor,
        ):
            return _record_applied(envelope, store)

        current_state = store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        conflict = _state_conflict(envelope, current_state)
        if conflict is not None:
            return _mark_conflict(envelope, store, conflict)

        try:
            with self.note_db.transaction() as conn:
                common = {
                    "owner_user_id": str(self.note_db.client_id),
                    "dataset_id": envelope.dataset_id,
                    "sync_object_hash": envelope.payload_hash or "",
                    "sync_server_cursor": envelope.server_cursor,
                    "conn": conn,
                }
                if isinstance(parsed.payload, NotesTaskActivityV1):
                    task_store.create_sync_task_activity(
                        payload=parsed.payload,
                        **common,
                    )
                else:
                    task_store.tombstone_sync_task_activity(
                        activity_id=envelope.object_id,
                        payload=parsed.payload,
                        original_payload=parsed.original,
                        base_server_cursor=envelope.base_server_cursor or 0,
                        base_hash=envelope.base_object_hash or "",
                        **common,
                    )
            return _record_applied(envelope, store)
        except ConflictError:
            return _mark_conflict(
                envelope,
                store,
                _conflict_result("product_state_conflict"),
            )
        except Exception as exc:  # noqa: BLE001 - product commit must remain replayable.
            logger.warning(
                "Failed to materialize notes.task_activity envelope {} for object {}: {}",
                envelope.client_envelope_id,
                envelope.object_id,
                type(exc).__name__,
            )
            return _mark_failed(envelope, store, _safe_error_message(exc))


def _parse_create(envelope: SyncEnvelope, owner_user_id: str) -> NotesTaskActivityV1:
    """Parse and verify one immutable activity create envelope."""

    source_kind = envelope.payload.get("source_kind")
    trusted = source_kind != "client"
    payload = parse_notes_task_activity_v1(
        envelope.payload,
        owner_user_id=owner_user_id,
        bound_actor_type=str(envelope.payload.get("actor_type")),
        bound_actor_id=envelope.payload.get("actor_id"),
        authenticated_device_id=None if trusted else envelope.device_id,
        trusted_server_origin=trusted,
    )
    if (
        envelope.operation != "upsert"
        or envelope.object_revision != 1
        or envelope.entity_version != 1
        or envelope.object_id != payload.activity_id
        or envelope.parent_id != payload.note_id
        or envelope.payload_hash
        != notes_task_activity_object_hash(payload, revision=1, deleted=False)
    ):
        raise NotesTaskContractError("notes.task_activity create lineage is invalid")
    return payload


def _parse_envelope(
    envelope: SyncEnvelope,
    *,
    store: SyncV2Store,
    owner_user_id: str,
) -> _ParsedActivityEnvelope:
    """Parse and verify one canonical activity envelope and its original create."""

    routing = envelope.routing_metadata
    trusted_bootstrap_routing = bool(
        set(routing)
        == {
            "bootstrap_capture",
            "bootstrap_id",
            "source",
            "origin",
            "server_device_id",
            "server_owner_user_id",
        }
        and routing.get("bootstrap_capture") is True
        and isinstance(routing.get("bootstrap_id"), str)
        and routing.get("bootstrap_id")
        and routing.get("source") == "notes-task-activity-bootstrap"
        and routing.get("origin") == "server"
        and envelope.payload.get("source_kind") == "trusted_bootstrap_v1"
    )
    trusted_server_routing = _valid_server_routing(envelope)
    if (
        envelope.adapter_version != 1
        or envelope.schema_version != 1
        or envelope.operation not in {"upsert", "tombstone"}
        or (routing and not trusted_bootstrap_routing and not trusted_server_routing)
    ):
        raise NotesTaskContractError("notes.task_activity envelope lineage is invalid")
    if envelope.operation == "upsert":
        payload = _parse_create(envelope, owner_user_id)
        return _ParsedActivityEnvelope(payload=payload, original=payload)
    if (
        envelope.object_revision != 2
        or envelope.entity_version != 2
        or envelope.base_server_cursor is None
        or envelope.base_object_revision != 1
        or envelope.base_object_hash is None
    ):
        raise NotesTaskContractError("notes.task_activity tombstone lineage is invalid")
    original_envelope = store.get_envelope_by_server_cursor(envelope.base_server_cursor)
    if (
        original_envelope is None
        or original_envelope.dataset_id != envelope.dataset_id
        or original_envelope.domain != envelope.domain
        or original_envelope.object_id != envelope.object_id
        or original_envelope.payload_hash != envelope.base_object_hash
    ):
        raise NotesTaskContractError("notes.task_activity original create is unavailable")
    original = _parse_create(original_envelope, owner_user_id)
    payload = parse_notes_task_activity_tombstone_v1(
        envelope.payload,
        envelope_created_at_client=envelope.created_at_client or "",
        original_activity=original,
    )
    expected_hash = notes_task_activity_object_hash(
        payload,
        revision=2,
        deleted=True,
        activity_id=envelope.object_id,
        original_create_hash=envelope.base_object_hash,
    )
    if (
        envelope.object_id != original.activity_id
        or envelope.parent_id != payload.note_id
        or envelope.payload_hash != expected_hash
    ):
        raise NotesTaskContractError("notes.task_activity tombstone identity is invalid")
    return _ParsedActivityEnvelope(payload=payload, original=original)


def _valid_server_routing(envelope: SyncEnvelope) -> bool:
    """Accept only coordinator-produced server provenance and projection metadata."""

    routing = envelope.routing_metadata
    base_fields = {
        "source",
        "origin",
        "server_device_id",
        "server_owner_user_id",
    }
    allowed_fields = base_fields | {"task_projection"}
    if not (
        set(routing).issubset(allowed_fields)
        and base_fields.issubset(routing)
        and routing.get("origin") == "server"
        and routing.get("server_device_id") == "server-origin"
        and envelope.device_id == "server-origin"
        and isinstance(routing.get("server_owner_user_id"), str)
        and routing.get("server_owner_user_id")
        and isinstance(routing.get("source"), str)
        and 1 <= len(str(routing["source"])) <= 128
        and envelope.payload.get("source_kind") != "client"
    ):
        return False
    projection = routing.get("task_projection")
    if projection is None:
        return True
    if not isinstance(projection, Mapping):
        return False
    try:
        from ..notes_task_coordinator import (  # Local import avoids materializer cycles.
            _validate_task_projection_group_metadata,
        )

        anchor = _validate_task_projection_group_metadata(projection)
    except (ImportError, ValueError):
        return False
    return (
        envelope.payload.get("task_id") == anchor.task_id
        and envelope.payload.get("note_id") == envelope.parent_id
    )


def _state_conflict(
    envelope: SyncEnvelope,
    current: SyncObjectState | None,
) -> MaterializationResult | None:
    """Return a deterministic conflict unless the immutable transition is current."""

    if current is None:
        if envelope.operation == "upsert" and all(
            value is None
            for value in (
                envelope.base_server_cursor,
                envelope.base_object_revision,
                envelope.base_object_hash,
            )
        ):
            return None
        return _conflict_result("missing_server_object")
    if (
        current.latest_server_cursor == envelope.server_cursor
        and current.object_revision == envelope.object_revision
        and current.object_hash == envelope.payload_hash
        and current.deleted == (envelope.operation == "tombstone")
    ):
        return _conflict_result("divergent_product_replay")
    if envelope.operation != "tombstone" or current.deleted:
        return _conflict_result("immutable_activity")
    if (
        envelope.base_server_cursor != current.latest_server_cursor
        or envelope.base_object_revision != current.object_revision
        or envelope.base_object_hash != current.object_hash
    ):
        return _conflict_result("stale_base_state")
    return None


def _record_applied(envelope: SyncEnvelope, store: SyncV2Store) -> MaterializationResult:
    """Record the immutable activity head and applied envelope status."""

    try:
        store.upsert_object_state(
            SyncObjectState(
                dataset_id=envelope.dataset_id,
                domain=envelope.domain,
                object_id=envelope.object_id,
                object_revision=int(envelope.object_revision or 0),
                object_hash=envelope.payload_hash or "",
                latest_server_cursor=envelope.server_cursor,
                deleted=envelope.operation == "tombstone",
            )
        )
        store.mark_envelope_apply_status(envelope.server_cursor, apply_status="applied")
        return MaterializationResult(status="applied")
    except Exception as exc:  # noqa: BLE001 - split commits must remain replayable.
        return _mark_failed(envelope, store, _safe_error_message(exc))


def _mark_conflict(
    envelope: SyncEnvelope,
    store: SyncV2Store,
    result: MaterializationResult,
) -> MaterializationResult:
    """Persist one deterministic immutable-product conflict."""

    try:
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="conflict",
            apply_error_code=_CONFLICT_TYPE,
            apply_error_message=result.message,
        )
    except Exception as exc:  # noqa: BLE001 - preserve the deterministic conflict.
        logger.warning(
            "Could not persist notes.task_activity conflict for cursor {}: {}",
            envelope.server_cursor,
            type(exc).__name__,
        )
    return result


def _mark_failed(
    envelope: SyncEnvelope,
    store: SyncV2Store,
    message: str,
) -> MaterializationResult:
    """Persist and return a bounded replayable activity projection failure."""

    try:
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="failed",
            apply_error_code=_ERROR_CODE,
            apply_error_message=message,
        )
    except Exception as exc:  # noqa: BLE001 - preserve a replayable failed result.
        logger.warning(
            "Could not persist failed notes.task_activity status for cursor {}: {}",
            envelope.server_cursor,
            type(exc).__name__,
        )
    return MaterializationResult(status="failed", error_code=_ERROR_CODE, message=message)


def _conflict_result(reason: str) -> MaterializationResult:
    """Build the stable immutable product conflict result."""

    return MaterializationResult(
        status="conflict",
        conflict_type=_CONFLICT_TYPE,
        message="notes.task_activity product state does not match its immutable lineage",
        metadata={"reason": reason},
    )


def _safe_error_message(exc: Exception) -> str:
    """Map internal activity projection failures to bounded public messages."""

    if isinstance(exc, NotesTaskContractError):
        return "notes.task_activity envelope validation failed"
    if isinstance(exc, InputError):
        return "notes.task_activity dependency validation failed"
    if isinstance(exc, (CharactersRAGDBError, BackendDatabaseError, sqlite3.DatabaseError)):
        return "notes.task_activity product database operation failed"
    return "notes.task_activity projection failed"


__all__ = ["NotesTaskActivityMaterializer"]
