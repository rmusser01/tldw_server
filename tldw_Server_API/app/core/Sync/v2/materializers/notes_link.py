"""Materialize canonical ``notes.link`` envelopes into one owner's Notes DB."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLinkStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

from ..errors import SyncMaterializationContractError
from ..models import SyncEnvelope, SyncObjectState
from ..notes_link import (
    NotesLinkValidationError,
    parse_notes_link_payload,
    validate_notes_link_object_id,
)
from ..store import SyncV2Store
from .base import MaterializationResult
from .guarded_product_mutation import GuardedProductMutation

_ERROR_CODE = "notes_link_projection_failed"
_CONFLICT_TYPE = "notes_link_product_conflict"


@dataclass(slots=True)
class NotesLinkMaterializer:
    """Apply accepted explicit-link envelopes to the user-bound product store."""

    note_db: CharactersRAGDB
    domain: str = "notes.link"

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
        guarded_mutation: GuardedProductMutation | None = None,
    ) -> MaterializationResult:
        """Project one accepted notes.link envelope and persist its apply state."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code=_ERROR_CODE,
                message="Stored notes.link envelope is missing a server cursor",
            )
        if guarded_mutation is not None:
            guarded_mutation.require_identity(envelope.domain, envelope.object_id)
            if (
                envelope.operation != "upsert"
                or envelope.routing_metadata.get("restore_intent") is True
            ):
                raise SyncMaterializationContractError()

        current_state = store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        already_materialized = _already_materialized(envelope, current_state)
        if guarded_mutation is None and already_materialized:
            return _mark_applied(envelope, store)
        state_conflict = (
            None if already_materialized else _state_conflict(envelope, current_state)
        )
        if state_conflict is not None:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="conflict",
                apply_error_code=_CONFLICT_TYPE,
                apply_error_message=state_conflict.message,
            )
            return state_conflict

        try:
            if envelope.schema_version != 1:
                raise NotesLinkValidationError("notes.link schema version must be 1")
            payload = parse_notes_link_payload(envelope.operation, envelope.payload)
            validate_notes_link_object_id(envelope.object_id)
            expected_version = _expected_product_version(envelope)
            projection = NotesLinkStore(self.note_db)
            if envelope.operation == "tombstone":
                if expected_version is None:
                    raise ConflictError("notes.link tombstone requires a base version")
                projection.tombstone(
                    edge_id=envelope.object_id,
                    payload=payload,
                    expected_version=expected_version,
                )
                deleted = True
            elif envelope.routing_metadata.get("restore_intent") is True:
                if expected_version is None:
                    raise ConflictError("notes.link restore requires a base version")
                projection.restore(
                    edge_id=envelope.object_id,
                    payload=payload,
                    expected_version=expected_version,
                    allow_deleted_endpoints=True,
                )
                deleted = False
            else:
                projection.upsert(
                    edge_id=envelope.object_id,
                    payload=payload,
                    expected_version=expected_version,
                    allow_deleted_endpoints=True,
                    before=(
                        guarded_mutation.before
                        if guarded_mutation is not None
                        else None
                    ),
                    after=(
                        guarded_mutation.after
                        if guarded_mutation is not None
                        else None
                    ),
                )
                deleted = False

            object_revision = (
                envelope.object_revision
                if envelope.object_revision is not None
                else 1
                if current_state is None
                else current_state.object_revision + 1
            )
            store.upsert_object_state(
                SyncObjectState(
                    dataset_id=envelope.dataset_id,
                    domain=envelope.domain,
                    object_id=envelope.object_id,
                    object_revision=object_revision,
                    object_hash=envelope.payload_hash or "",
                    latest_server_cursor=envelope.server_cursor,
                    deleted=deleted,
                )
            )
            store.mark_envelope_apply_status(envelope.server_cursor, apply_status="applied")
            return MaterializationResult(status="applied")
        except ConflictError:
            message = "notes.link projection conflicts with existing product state"
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="conflict",
                apply_error_code=_CONFLICT_TYPE,
                apply_error_message=message,
            )
            return MaterializationResult(
                status="conflict",
                conflict_type=_CONFLICT_TYPE,
                message=message,
                metadata={"reason": "product_state_conflict"},
            )
        except Exception as exc:  # noqa: BLE001 - failed projection must remain replayable.
            message = _safe_error_message(exc)
            logger.warning(
                "Failed to materialize notes.link envelope {} for object {}: {}",
                envelope.client_envelope_id,
                envelope.object_id,
                type(exc).__name__,
            )
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code=_ERROR_CODE,
                apply_error_message=message,
            )
            return MaterializationResult(
                status="failed",
                error_code=_ERROR_CODE,
                message=message,
            )


def _expected_product_version(envelope: SyncEnvelope) -> int | None:
    value = envelope.base_version
    if value is None:
        return None
    if isinstance(value, bool):
        raise NotesLinkValidationError("notes.link base version must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise NotesLinkValidationError("notes.link base version must be an integer") from exc
    if parsed < 1 or str(parsed) != str(value):
        raise NotesLinkValidationError("notes.link base version must be a positive integer")
    return parsed


def _already_materialized(
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
) -> bool:
    return bool(
        current_state is not None
        and current_state.latest_server_cursor == envelope.server_cursor
        and (envelope.object_revision is None or current_state.object_revision == envelope.object_revision)
        and current_state.object_hash == (envelope.payload_hash or "")
        and current_state.deleted == (envelope.operation == "tombstone")
    )


def _state_conflict(
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
) -> MaterializationResult | None:
    has_any_base = any(
        value is not None
        for value in (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
        )
    )
    if current_state is None:
        if has_any_base or envelope.routing_metadata.get("restore_intent") is True:
            return _conflict_result("missing_server_object")
        return None
    base_matches = (
        envelope.base_server_cursor == current_state.latest_server_cursor
        and envelope.base_object_revision == current_state.object_revision
        and envelope.base_object_hash == current_state.object_hash
    )
    if not base_matches:
        return _conflict_result("stale_base_state")
    if envelope.operation == "upsert" and current_state.deleted:
        if envelope.routing_metadata.get("restore_intent") is not True:
            return _conflict_result("restore_intent_required")
    if envelope.routing_metadata.get("restore_intent") is True and not current_state.deleted:
        return _conflict_result("restore_target_not_deleted")
    return None


def _conflict_result(reason: str) -> MaterializationResult:
    return MaterializationResult(
        status="conflict",
        conflict_type=_CONFLICT_TYPE,
        message="notes.link base state does not match the current server projection",
        metadata={"reason": reason},
    )


def _mark_applied(envelope: SyncEnvelope, store: SyncV2Store) -> MaterializationResult:
    try:
        store.mark_envelope_apply_status(envelope.server_cursor, apply_status="applied")
    except Exception as exc:  # noqa: BLE001 - state failure must remain replayable.
        message = _safe_error_message(exc)
        try:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code=_ERROR_CODE,
                apply_error_message=message,
            )
        except Exception as status_exc:  # noqa: BLE001 - preserve the failed result.
            logger.warning(
                "Could not persist failed notes.link apply status for cursor {}: {}",
                envelope.server_cursor,
                type(status_exc).__name__,
            )
        return MaterializationResult(
            status="failed",
            error_code=_ERROR_CODE,
            message=message,
        )
    return MaterializationResult(status="applied")


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, NotesLinkValidationError):
        return "notes.link envelope validation failed"
    if isinstance(exc, InputError):
        return "notes.link dependency validation failed"
    if isinstance(
        exc,
        (CharactersRAGDBError, BackendDatabaseError, sqlite3.DatabaseError),
    ):
        return "notes.link product database operation failed"
    return "notes.link projection failed"


__all__ = ["NotesLinkMaterializer"]
