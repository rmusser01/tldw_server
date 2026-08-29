from __future__ import annotations

"""Materialize Notes organization Sync envelopes into one user's ChaChaNotes DB."""

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    GuardedKeywordIdentityCollision,
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

from ..models import (
    NOTES_ORGANIZATION_DOMAINS,
    SyncDomain,
    SyncEnvelope,
    SyncObjectState,
)
from ..notes_organization import (
    NotesOrganizationValidationError,
    parse_notes_organization_payload,
    validate_organization_object_id,
)
from ..store import SyncV2Store
from .base import MaterializationResult
from .guarded_product_mutation import GuardedProductMutation

_RESOURCE_DOMAINS = frozenset(
    {"notes.keyword", "notes.keyword_collection", "notes.folder"}
)
_ERROR_CODE = "notes_organization_projection_failed"
_FOLDER_PROVENANCE_KEY = "notes_folder_origin_provenance"
_KEYWORD_MERGE_PRECONDITION_KEY = "notes_keyword_merge_precondition"
_SERVER_ORIGIN_DEVICE_ID = "server-origin"


def _trusted_folder_origin_provenance(
    envelope: SyncEnvelope,
    note_db: CharactersRAGDB,
) -> dict[str, object] | None:
    """Return safe owner-bound local provenance; ignore every other origin."""

    routing = envelope.routing_metadata
    if (
        envelope.domain != "notes.folder_link"
        or envelope.device_id != _SERVER_ORIGIN_DEVICE_ID
        or routing.get("origin") != "server"
        or routing.get("server_device_id") != _SERVER_ORIGIN_DEVICE_ID
        or routing.get("server_owner_user_id") != str(note_db.client_id)
    ):
        return None
    raw = routing.get(_FOLDER_PROVENANCE_KEY)
    if not isinstance(raw, Mapping) or set(raw) not in (
        {"operation", "source_id"},
        {"operation", "source_id", "read_set_hash"},
        {"operation", "source_id", "pre_state_hash", "post_state_hash"},
    ):
        return None
    operation = raw.get("operation")
    source_id = raw.get("source_id")
    if operation not in {"source_upsert", "source_delete"}:
        return None
    if isinstance(source_id, bool) or not isinstance(source_id, int) or source_id <= 0:
        return None
    provenance: dict[str, object] = {"operation": operation, "source_id": source_id}
    for key in ("read_set_hash", "pre_state_hash", "post_state_hash"):
        state_hash = raw.get(key)
        if state_hash is None:
            continue
        if (
            not isinstance(state_hash, str)
            or len(state_hash) != 64
            or any(character not in "0123456789abcdef" for character in state_hash)
        ):
            return None
        provenance[key] = state_hash
    return provenance


def _trusted_keyword_merge_precondition(
    envelope: SyncEnvelope,
    note_db: CharactersRAGDB,
) -> str | None:
    """Return an owner-bound local merge-final relationship token."""

    routing = envelope.routing_metadata
    if (
        envelope.domain != "notes.keyword"
        or envelope.operation != "tombstone"
        or envelope.device_id != _SERVER_ORIGIN_DEVICE_ID
        or routing.get("origin") != "server"
        or routing.get("server_device_id") != _SERVER_ORIGIN_DEVICE_ID
        or routing.get("server_owner_user_id") != str(note_db.client_id)
    ):
        return None
    raw = routing.get(_KEYWORD_MERGE_PRECONDITION_KEY)
    if not isinstance(raw, Mapping) or set(raw) != {"relationship_set_hash"}:
        return None
    relationship_set_hash = raw.get("relationship_set_hash")
    if (
        not isinstance(relationship_set_hash, str)
        or len(relationship_set_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in relationship_set_hash
        )
    ):
        return None
    return relationship_set_hash


@dataclass(slots=True)
class NotesOrganizationMaterializer:
    """Apply one Notes organization domain into a user-bound product database."""

    note_db: CharactersRAGDB
    domain: SyncDomain

    def __post_init__(self) -> None:
        if self.domain not in NOTES_ORGANIZATION_DOMAINS:
            raise ValueError(f"Unsupported Notes organization materializer domain: {self.domain}")

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
        guarded_mutation: GuardedProductMutation | None = None,
    ) -> MaterializationResult:
        """Project one accepted organization envelope and record apply state."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code=_ERROR_CODE,
                message="Stored Sync envelope is missing a server cursor",
            )
        if guarded_mutation is not None:
            guarded_mutation.require_identity(envelope.domain, envelope.object_id)
            if envelope.operation != "upsert":
                raise ValueError("Guarded Notes organization mutation must be an upsert")

        current_state = store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        already_materialized = self._is_already_materialized(envelope, current_state)
        if guarded_mutation is None and already_materialized:
            return self._mark_applied(envelope, store=store)

        conflict = None if already_materialized else self._detect_conflict(envelope, current_state)
        if conflict is not None:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="conflict",
                apply_error_code="whole_object_conflict",
                apply_error_message=conflict.message,
            )
            return conflict

        try:
            if envelope.schema_version != 1:
                raise ValueError("Notes organization schema version must be 1")
            payload = parse_notes_organization_payload(
                envelope.domain,
                envelope.operation,
                envelope.payload,
            )
            validate_organization_object_id(envelope.domain, envelope.object_id, payload)

            projection = NotesOrganizationSyncStore(self.note_db)
            if envelope.domain in _RESOURCE_DOMAINS:
                projection.apply_resource(
                    domain=envelope.domain,
                    object_id=envelope.object_id,
                    operation=envelope.operation,
                    payload=payload,
                    merge_relationship_set_hash=(
                        _trusted_keyword_merge_precondition(envelope, self.note_db)
                    ),
                    before=(
                        guarded_mutation.before
                        if guarded_mutation is not None
                        else None
                    ),
                    after=(
                        guarded_mutation.after
                        if guarded_mutation is not None
                        and envelope.domain != "notes.keyword"
                        else None
                    ),
                )
            else:
                projection.apply_relationship(
                    domain=envelope.domain,
                    object_id=envelope.object_id,
                    operation=envelope.operation,
                    payload=payload,
                    routing_metadata=envelope.routing_metadata,
                    origin_provenance=_trusted_folder_origin_provenance(
                        envelope, self.note_db
                    ),
                    source_transition_identity=envelope.mutation_group_id,
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

            object_revision = self._next_object_revision(envelope, current_state)
            store.upsert_object_state(
                SyncObjectState(
                    dataset_id=envelope.dataset_id,
                    domain=envelope.domain,
                    object_id=envelope.object_id,
                    object_revision=object_revision,
                    object_hash=envelope.payload_hash or "",
                    latest_server_cursor=envelope.server_cursor,
                    deleted=envelope.operation == "tombstone",
                )
            )
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="applied",
            )
        except GuardedKeywordIdentityCollision:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="superseded",
            )
            return MaterializationResult(status="applied")
        except Exception as exc:  # noqa: BLE001 - every projection failure is replayable state.
            message = _safe_error_message(exc)
            logger.warning(
                "Failed to materialize {} envelope {} for object {}: {}",
                envelope.domain,
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
        return MaterializationResult(status="applied")

    def _mark_applied(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        try:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="applied",
            )
        except Exception as exc:  # noqa: BLE001 - apply state failures must remain replayable.
            message = _safe_error_message(exc)
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
        return MaterializationResult(status="applied")

    @staticmethod
    def _next_object_revision(
        envelope: SyncEnvelope,
        current_state: SyncObjectState | None,
    ) -> int:
        if envelope.object_revision is not None:
            return envelope.object_revision
        return 1 if current_state is None else current_state.object_revision + 1

    @staticmethod
    def _is_already_materialized(
        envelope: SyncEnvelope,
        current_state: SyncObjectState | None,
    ) -> bool:
        if current_state is None or current_state.latest_server_cursor != envelope.server_cursor:
            return False
        if (
            envelope.object_revision is not None
            and current_state.object_revision != envelope.object_revision
        ):
            return False
        return (
            current_state.object_hash == (envelope.payload_hash or "")
            and current_state.deleted == (envelope.operation == "tombstone")
        )

    @staticmethod
    def _detect_conflict(
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
                return _conflict_result("missing_server_object", envelope, None)
            return None
        if not has_base:
            return _conflict_result("missing_base_state", envelope, current_state)
        base_matches = (
            envelope.base_server_cursor == current_state.latest_server_cursor
            and envelope.base_object_revision == current_state.object_revision
            and envelope.base_object_hash == current_state.object_hash
        )
        if restore_requested and not current_state.deleted:
            return _conflict_result("restore_target_not_deleted", envelope, current_state)
        if restore_requested and current_state.deleted and base_matches:
            return None
        if envelope.operation == "upsert" and current_state.deleted:
            return _conflict_result("server_object_deleted", envelope, current_state)
        if not base_matches:
            return _conflict_result("stale_base_state", envelope, current_state)
        return None


def _conflict_result(
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
        message=f"{envelope.domain} base state does not match the current server projection",
        metadata=metadata,
    )


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, NotesOrganizationValidationError):
        return "Notes organization envelope validation failed"
    if isinstance(exc, InputError):
        return "Notes organization dependency or hierarchy validation failed"
    if isinstance(exc, ConflictError):
        return "Notes organization projection conflicts with existing state"
    if isinstance(
        exc,
        (CharactersRAGDBError, BackendDatabaseError, sqlite3.DatabaseError),
    ):
        return "Notes organization product database operation failed"
    return "Notes organization projection failed"


__all__ = ["NotesOrganizationMaterializer"]
