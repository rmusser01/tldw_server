"""Safe HTTP mapping for server-origin Notes Sync failures."""

from __future__ import annotations

from fastapi import HTTPException

from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
    NotesKeywordMergeUnsynchronizedDependencyError,
    NotesOrganizationDomainsIncompleteError,
    NotesOrganizationNotReadyError,
    NotesOrganizationPreflightError,
    NotesOrganizationResourceNotFoundError,
    NotesOrganizationVersionConflictError,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    SyncServerOriginIdempotencyConflictError,
    SyncServerOriginMaterializationError,
    SyncServerOriginMutationNotSupportedError,
    SyncServerOriginRestoreConflictError,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    SyncServerOriginBatchAppendError,
    SyncServerOriginBatchIdempotencyConflictError,
    SyncServerOriginBatchMaterializationError,
)

NOTES_SYNC_EXCEPTIONS = (
    SyncStoreError,
    NotesKeywordMergeUnsynchronizedDependencyError,
    NotesOrganizationResourceNotFoundError,
    NotesOrganizationVersionConflictError,
)


def notes_sync_http_error(exc: Exception) -> HTTPException:
    """Map Sync failures without exposing storage or projection internals."""

    if isinstance(exc, NotesKeywordMergeUnsynchronizedDependencyError):
        return _error(409, exc.error_code, "The keyword has a dependency that is not synchronized.")
    if isinstance(exc, NotesOrganizationResourceNotFoundError):
        return _error(404, exc.error_code, "The Notes organization resource was not found.")
    if isinstance(exc, NotesOrganizationVersionConflictError):
        return _error(409, exc.error_code, "The Notes organization resource has changed; refresh and retry.")
    if isinstance(exc, NotesOrganizationDomainsIncompleteError):
        error = _error(
            409,
            exc.error_code,
            "The active Sync dataset lacks the complete Notes organization domain group.",
        )
        error.detail["missing_domains"] = list(exc.missing_domains)
        return error
    if isinstance(exc, NotesOrganizationNotReadyError):
        error = _error(409, exc.error_code, "Notes organization Sync is not ready for writes.")
        error.detail["state"] = exc.state
        if exc.repair_error_code:
            error.detail["repair_error_code"] = exc.repair_error_code
        return error
    if isinstance(exc, NotesOrganizationPreflightError):
        return _error(409, exc.error_code, "The Notes organization change conflicts with canonical state.")
    if isinstance(exc, SyncServerOriginBatchIdempotencyConflictError):
        error = _error(
            409,
            exc.error_code,
            "The idempotency key was already used for a different Notes organization change.",
        )
        error.detail["mutation_group_id"] = exc.mutation_group_id
        return error
    if isinstance(exc, SyncServerOriginBatchMaterializationError):
        error = _error(
            503,
            exc.error_code,
            "The canonical Notes organization change is durable but its projection is incomplete.",
        )
        error.detail.update(
            mutation_group_id=(exc.result.envelopes[0].mutation_group_id if exc.result.envelopes else None),
            retryable=exc.retryable,
        )
        return error
    if isinstance(exc, SyncServerOriginBatchAppendError):
        error = _error(
            503,
            exc.error_code,
            "Sync could not durably append the complete Notes organization change.",
        )
        error.detail["mutation_group_id"] = exc.mutation_group_id
        return error
    if isinstance(exc, SyncServerOriginRestoreConflictError):
        error = _error(409, exc.error_code, "Note restore requires the current deleted note version.")
        error.detail["object_id"] = exc.object_id
        return error
    if isinstance(exc, SyncServerOriginIdempotencyConflictError):
        return _error(
            409,
            "sync_server_origin_idempotency_conflict",
            "The idempotency key was already used for a different note change.",
        )
    if isinstance(exc, SyncServerOriginMaterializationError):
        error = _error(
            503,
            "sync_server_origin_materialization_failed",
            "Sync accepted the server-origin note change but projection apply failed.",
        )
        error.detail.update(
            server_cursor=exc.envelope.server_cursor,
            apply_status=exc.envelope.apply_status,
            apply_error_code=exc.envelope.apply_error_code,
            apply_error_message=exc.envelope.apply_error_message,
        )
        return error
    if isinstance(exc, SyncServerOriginMutationNotSupportedError):
        error = _error(409, exc.error_code, str(exc))
        error.detail.update(dataset_id=exc.dataset.dataset_id, domain=exc.domain)
        return error
    if isinstance(exc, SyncStoreError):
        return _error(503, "sync_server_origin_append_failed", "Sync could not record the server-origin note change.")
    return _error(500, "sync_server_origin_failed", "Sync failed while recording the server-origin note change.")


def _error(status_code: int, error_code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error_code": error_code, "message": message},
    )
