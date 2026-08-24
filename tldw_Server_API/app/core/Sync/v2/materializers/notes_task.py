"""Materialize canonical ``notes.task`` envelopes into the product task store."""

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
    NotesTaskV1Payload,
    notes_task_object_hash,
    parse_notes_task_tombstone_v1,
    parse_notes_task_v1,
)
from ..store import SyncV2Store
from .base import MaterializationResult

_ERROR_CODE = "notes_task_projection_failed"
_CONFLICT_TYPE = "notes_task_product_conflict"


@dataclass(slots=True)
class NotesTaskMaterializer:
    """Apply accepted dormant task envelopes without creating activity."""

    note_db: CharactersRAGDB
    domain: str = "notes.task"

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Commit product state, then repair Sync projection bookkeeping."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code=_ERROR_CODE,
                message="Stored notes.task envelope is missing a server cursor",
            )

        try:
            payload = _parse_envelope(envelope, owner_user_id=str(self.note_db.client_id))
        except NotesTaskContractError:
            return _mark_failed(
                envelope,
                store,
                "notes.task envelope validation failed",
            )

        expected_projection = _expected_projection_status(envelope)
        task_store = self.note_db.task_store
        if task_store.verify_sync_task_postcondition(
            owner_user_id=str(self.note_db.client_id),
            dataset_id=envelope.dataset_id,
            payload=payload,
            canonical_revision=int(envelope.object_revision or 0),
            canonical_hash=envelope.payload_hash or "",
            deleted=envelope.operation == "tombstone",
            expected_projection_status=expected_projection,
        ):
            return _record_applied(envelope, store)

        current_state = store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        state_conflict = _state_conflict(envelope, current_state)
        if state_conflict is not None:
            return _mark_conflict(envelope, store, state_conflict)

        try:
            with self.note_db.transaction() as conn:
                common = {
                    "owner_user_id": str(self.note_db.client_id),
                    "dataset_id": envelope.dataset_id,
                    "payload": payload,
                    "canonical_revision": int(envelope.object_revision or 0),
                    "canonical_hash": envelope.payload_hash or "",
                    "conn": conn,
                }
                if current_state is None:
                    task_store.apply_sync_task_create(**common)
                else:
                    transition = {
                        **common,
                        "base_revision": current_state.object_revision,
                        "base_hash": current_state.object_hash,
                    }
                    if envelope.operation == "tombstone":
                        task_store.apply_sync_task_tombstone(**transition)
                    elif envelope.routing_metadata.get("restore_intent") is True:
                        task_store.apply_sync_task_restore(**transition)
                    else:
                        task_store.apply_sync_task_upsert(**transition)
                projection = envelope.routing_metadata.get("task_projection")
                if (
                    envelope.operation == "upsert"
                    and isinstance(projection, Mapping)
                    and type(projection.get("linked")) is bool
                ):
                    task_store.apply_sync_task_projection_status(
                        owner_user_id=str(self.note_db.client_id),
                        dataset_id=envelope.dataset_id,
                        task_id=envelope.object_id,
                        note_id=str(envelope.parent_id),
                        projection_status=(
                            "live" if projection["linked"] else "unlinked"
                        ),
                        conn=conn,
                    )
            return _record_applied(envelope, store)
        except ConflictError:
            return _mark_conflict(
                envelope,
                store,
                _conflict_result("product_state_conflict"),
            )
        except Exception as exc:  # noqa: BLE001 - product commit must remain replayable.
            message = _safe_error_message(exc)
            logger.warning(
                "Failed to materialize notes.task envelope {} for object {}: {}",
                envelope.client_envelope_id,
                envelope.object_id,
                type(exc).__name__,
            )
            return _mark_failed(envelope, store, message)


def _parse_envelope(
    envelope: SyncEnvelope,
    *,
    owner_user_id: str,
) -> NotesTaskV1Payload:
    """Parse and verify one canonical task envelope against its lineage hash."""

    if (
        envelope.adapter_version != 1
        or envelope.schema_version != 1
        or envelope.operation not in {"upsert", "tombstone"}
        or envelope.object_revision is None
        or envelope.entity_version != envelope.object_revision
    ):
        raise NotesTaskContractError("notes.task envelope lineage is invalid")
    parser = (
        parse_notes_task_tombstone_v1
        if envelope.operation == "tombstone"
        else parse_notes_task_v1
    )
    payload = parser(envelope.payload, owner_user_id=owner_user_id)
    if envelope.object_id != payload.task_id or envelope.parent_id != payload.note_id:
        raise NotesTaskContractError("notes.task envelope identity is invalid")
    expected_hash = notes_task_object_hash(
        payload,
        revision=envelope.object_revision,
        deleted=envelope.operation == "tombstone",
    )
    if envelope.payload_hash != expected_hash:
        raise NotesTaskContractError("notes.task envelope hash is invalid")
    return payload


def _expected_projection_status(envelope: SyncEnvelope) -> str | None:
    """Return the projection status required after the envelope is applied."""

    if envelope.operation == "tombstone":
        return "deleted"
    projection = envelope.routing_metadata.get("task_projection")
    if isinstance(projection, Mapping) and type(projection.get("linked")) is bool:
        return "live" if projection["linked"] else "unlinked"
    if envelope.object_revision == 1 or envelope.routing_metadata.get("restore_intent") is True:
        return "unlinked"
    return None


def _state_conflict(
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
) -> MaterializationResult | None:
    """Return a deterministic conflict when the Sync base is not current."""

    has_base = all(
        value is not None
        for value in (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
        )
    )
    if current_state is None:
        if has_base or envelope.routing_metadata.get("restore_intent") is True:
            return _conflict_result("missing_server_object")
        return None
    if (
        envelope.base_server_cursor != current_state.latest_server_cursor
        or envelope.base_object_revision != current_state.object_revision
        or envelope.base_object_hash != current_state.object_hash
    ):
        return _conflict_result("stale_base_state")
    if current_state.deleted:
        if envelope.routing_metadata.get("restore_intent") is not True:
            return _conflict_result("restore_intent_required")
    elif envelope.routing_metadata.get("restore_intent") is True:
        return _conflict_result("restore_target_not_deleted")
    return None


def _record_applied(
    envelope: SyncEnvelope,
    store: SyncV2Store,
) -> MaterializationResult:
    """Record the applied object state and envelope status."""

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
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="applied",
        )
        return MaterializationResult(status="applied")
    except Exception as exc:  # noqa: BLE001 - split commits must remain replayable.
        return _mark_failed(envelope, store, _safe_error_message(exc))


def _mark_conflict(
    envelope: SyncEnvelope,
    store: SyncV2Store,
    result: MaterializationResult,
) -> MaterializationResult:
    """Persist a deterministic conflict status without replacing its result."""

    try:
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="conflict",
            apply_error_code=_CONFLICT_TYPE,
            apply_error_message=result.message,
        )
    except Exception as exc:  # noqa: BLE001 - preserve the deterministic conflict.
        logger.warning(
            "Could not persist notes.task conflict status for cursor {}: {}",
            envelope.server_cursor,
            type(exc).__name__,
        )
    return result


def _mark_failed(
    envelope: SyncEnvelope,
    store: SyncV2Store,
    message: str,
) -> MaterializationResult:
    """Persist and return a bounded replayable projection failure."""

    try:
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="failed",
            apply_error_code=_ERROR_CODE,
            apply_error_message=message,
        )
    except Exception as exc:  # noqa: BLE001 - preserve a replayable failed result.
        logger.warning(
            "Could not persist failed notes.task status for cursor {}: {}",
            envelope.server_cursor,
            type(exc).__name__,
        )
    return MaterializationResult(
        status="failed",
        error_code=_ERROR_CODE,
        message=message,
    )


def _conflict_result(reason: str) -> MaterializationResult:
    """Build the stable product-state conflict result."""

    return MaterializationResult(
        status="conflict",
        conflict_type=_CONFLICT_TYPE,
        message="notes.task product state does not match the current canonical base",
        metadata={"reason": reason},
    )


def _safe_error_message(exc: Exception) -> str:
    """Map internal projection failures to bounded public messages."""

    if isinstance(exc, NotesTaskContractError):
        return "notes.task envelope validation failed"
    if isinstance(exc, InputError):
        return "notes.task dependency validation failed"
    if isinstance(
        exc,
        (CharactersRAGDBError, BackendDatabaseError, sqlite3.DatabaseError),
    ):
        return "notes.task product database operation failed"
    return "notes.task projection failed"


__all__ = ["NotesTaskMaterializer"]
