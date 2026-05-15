# Server_API/app/api/v1/endpoints/sync-endpoint.py
# Description: This code provides a FastAPI endpoint for all Sync operations.
#
# Imports
import asyncio
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

#
# 3rd-party imports
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user

#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.sync_server_models import (
    ALLOWED_SYNC_OPERATIONS,
    ALLOWED_SYNC_SEND_ENTITIES,
    ClientChangesPayload,
    ServerChangesResponse,
    SyncLogEntry,
)
from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
    ConflictStatus,
    SyncCapabilitiesResponse,
    SyncConflictRecord,
    SyncConflictResolveRequest,
    SyncDatasetEnrollRequest,
    SyncDatasetEnrollResponse,
    SyncDeviceRegisterRequest,
    SyncDeviceRegisterResponse,
    SyncDomain,
    SyncKeyRecoveryBundleListResponse,
    SyncKeyRecoveryBundleRequest,
    SyncKeyRecoveryBundleRecord,
    SyncPullResponse,
    SyncPushAcceptedEnvelope,
    SyncPushConflictEnvelope,
    SyncPushRejectedEnvelope,
    SyncPushRequest,
    SyncPushResponse,
    SyncRestoreManifestResponse,
    SyncV2Envelope,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User

#
# DB Mgmt
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.Sync.Sync_Client import SYNC_BATCH_SIZE
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.factory import (
    default_sync_v2_registry,
    sync_v2_service_for_user,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

#
#
#######################################################################################################################
#
# Functions:

# All functions below are endpoints callable via HTTP requests and the corresponding code executed as a result of it.
#
# The router is a FastAPI object that allows us to define multiple endpoints under a single prefix.
# Create a new router instance
router = APIRouter()

# Explanation and Key Server-Side Aspects:
#     FastAPI Structure: Uses APIRouter, Pydantic models (SyncLogEntry, ClientChangesPayload, ServerChangesResponse), and dependency injection (Depends) for authentication and database access.
#     User-Scoped DB: The get_media_db_for_user dependency is critical. It ensures that all operations within an endpoint call correctly target the database belonging to the authenticated user.
#
#     /sync/send Endpoint:
#         Receives changes via ClientChangesPayload.
#         Authenticates the user and gets their Database instance.
#         Uses ServerSyncProcessor to handle the batch application.
#         The processor uses a transaction to apply all changes atomically.
#         Authoritative Timestamp: It captures datetime.now(timezone.utc) once at the start of processing the batch (server_authoritative_timestamp) and uses this timestamp when calling _execute_server_change_sql.
#         Conflict Resolution (LWW): The _resolve_server_conflict method compares the server_authoritative_timestamp with the last_modified timestamp already present in the user's DB record. If the server's current time is newer or equal, the incoming change wins and is forcefully applied; otherwise, the existing server state wins, and the incoming change is skipped.
#         Originating Client ID: _execute_server_change_sql correctly uses the originating_client_id from the change record when updating the client_id column in the user's DB.
#         Returns success (200) or error (500/409).
#
#     /sync/get Endpoint:
#         Authenticates the user and gets their Database instance.
#         Queries the user's sync_log for changes after the since_change_id provided by the client.
#         Filters out echo: The SQL query includes AND client_id != ? to avoid sending changes back to the client that originally made them.
#         Includes the latest_change_id from the user's log in the response so the client knows the server's current state for that user.
#         Returns the list of changes and the latest ID.
#     ServerSyncProcessor: Encapsulates the server-side application logic, mirroring the structure of the ClientSyncEngine but adapted for the server's role (using authoritative timestamps, handling requests for a specific user).
#     _execute_server_change_sql: This is intentionally almost identical to the client's version, leveraging the shared library and schema. The key difference is the source of the timestamp and potentially client_id parameters passed into it.
#     Async Considerations: While FastAPI is async, the underlying sqlite_db.py library uses synchronous operations with check_same_thread=False. For low-to-moderate load, FastAPI handles this by running sync code in a thread pool. For very high concurrency, switching the DB library and these endpoints to use asyncio and an async DB driver (aiosqlite) would be necessary for optimal performance, but significantly increases complexity. The current implementation will work but might block the event loop under heavy load.

####################################################################################################
#
# --- FastAPI Endpoint Definitions ---


def _is_client_validation_error(errors: list[str]) -> bool:
    if not errors:
        return False

    validation_markers = (
        "unsupported sync entity",
        "unsupported sync operation",
        "only valid for mediakeywords",
        "only supports 'link'/'unlink'",
        "missing payload",
        "failed to decode payload",
        "payload decode error",
        "rejecting sync change",
    )
    batch_wrapper_markers = (
        "failed processing change id",
        "rolling back batch",
    )
    saw_validation_error = False

    for err in errors:
        lowered = err.lower()
        if any(marker in lowered for marker in validation_markers):
            saw_validation_error = True
            continue

        # Batch-level wrappers can accompany client validation failures.
        if all(marker in lowered for marker in batch_wrapper_markers):
            continue

        if "failed to apply change" in lowered and "rolling back batch" in lowered:
            continue

        if "transaction rolled back" in lowered:
            continue

        if "unexpected server error" in lowered:
            return False

        # Unknown/non-validation errors are treated as server faults.
        return False

    return saw_validation_error


def _sync_user_id(user: User) -> str:
    user_id = getattr(user, "id_str", "") or str(getattr(user, "id", "") or "")
    return user_id or user.username


_default_sync_v2_registry = default_sync_v2_registry


def get_sync_v2_service(
    user: User = Depends(get_request_user),
) -> SyncV2Service:
    """Build the per-user Sync v2 service dependency."""

    return sync_v2_service_for_user(_sync_user_id(user))


def _safe_sync_v2_http_error(exc: Exception, **context: object) -> HTTPException:
    safe_context = {
        key: value
        for key, value in context.items()
        if value not in (None, "")
    }
    logger.bind(error_type=type(exc).__name__, **safe_context).warning(
        "Sync v2 request failed"
    )
    if isinstance(exc, SyncIdempotencyConflictError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "sync_idempotency_conflict",
                "message": "Sync idempotency key was reused with different content.",
            },
        )
    if isinstance(exc, SyncInvalidDomainError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "sync_invalid_domain",
                "message": "Sync domain is not valid for the requested dataset.",
            },
        )
    if isinstance(exc, SyncStoreError):
        lowered = str(exc).lower()
        if (
            "invalid sync cursor" in lowered
            or "page_size" in lowered
            or "resolution envelope" in lowered
            or "payload exceeds" in lowered
        ):
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "sync_validation_failed",
                    "message": "Sync request parameters are invalid.",
                },
            )
        if "already belongs" in lowered:
            return HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "sync_resource_conflict",
                    "message": "Sync resource conflicts with an existing registration.",
                },
            )
        not_found_markers = (
            "not found or is not accessible",
            "not found:",
            "was not found",
            "not accessible",
        )
        if any(marker in lowered for marker in not_found_markers):
            return HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "sync_resource_not_found",
                    "message": "Requested sync resource was not found or is not accessible.",
                },
            )
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "sync_store_error",
                "message": "Internal sync storage error while processing request.",
            },
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error_code": "sync_internal_error",
            "message": "Internal server error while processing sync request.",
        },
    )


def _core_envelope_from_api(envelope: SyncV2Envelope) -> SyncEnvelopeCreate:
    payload = model_dump_compat(
        envelope,
        exclude={"server_sequence", "server_timestamp", "encryption_policy"},
    )
    return SyncEnvelopeCreate(**payload)


def _api_envelope_from_core(
    envelope: Any,
    *,
    encryption_policy: str = "client_private_v1",
) -> SyncV2Envelope:
    payload = asdict(envelope)
    payload["encryption_policy"] = encryption_policy
    return SyncV2Envelope(**payload)


def _api_conflict_from_core(conflict: Any) -> SyncConflictRecord:
    return SyncConflictRecord(**asdict(conflict))


def _api_capabilities_from_core(capabilities: Any) -> SyncCapabilitiesResponse:
    return SyncCapabilitiesResponse(**asdict(capabilities))


def _api_key_record_metadata(record: Any) -> dict[str, Any]:
    return {
        "key_record_id": record.key_record_id,
        "dataset_id": record.dataset_id,
        "device_id": record.device_id,
        "key_purpose": record.key_purpose,
        "recovery_hint": record.recovery_hint,
        "rotation_of_key_record_id": record.rotation_of_key_record_id,
        "created_at": record.created_at,
    }


def _api_key_record_export(record: Any) -> SyncKeyRecoveryBundleRecord:
    return SyncKeyRecoveryBundleRecord(
        key_record_id=record.key_record_id,
        dataset_id=record.dataset_id,
        device_id=record.device_id,
        key_purpose=record.key_purpose,
        wrapped_key_blob=record.wrapped_key_blob,
        kdf_metadata=record.kdf_metadata,
        recovery_hint=record.recovery_hint,
        rotation_of_key_record_id=record.rotation_of_key_record_id,
        created_at=record.created_at,
        revoked_at=record.revoked_at,
    )


@router.get(
    "/capabilities",
    response_model=SyncCapabilitiesResponse,
    summary="Return Sync v2 protocol capabilities",
)
def get_sync_v2_capabilities(
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    return _api_capabilities_from_core(service.capabilities())


@router.post(
    "/devices/register",
    response_model=SyncDeviceRegisterResponse,
    summary="Register or refresh a Sync v2 device",
)
def register_sync_v2_device(
    request: SyncDeviceRegisterRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        registration = service.register_device(
            user_id=_sync_user_id(user),
            device_id=request.device_id,
            display_name=request.display_name,
            client_type=request.client_type,
            client_version=request.client_version,
            capabilities=request.capabilities,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            device_id=request.device_id,
        ) from exc
    return SyncDeviceRegisterResponse(
        device_id=registration.device.device_id,
        server_capabilities=_api_capabilities_from_core(registration.server_capabilities),
        required_actions=registration.required_actions,
        registered_at=registration.device.registered_at,
        last_seen_at=registration.device.last_seen_at,
    )


@router.post(
    "/datasets/enroll",
    response_model=SyncDatasetEnrollResponse,
    summary="Create or join a Sync v2 dataset",
)
def enroll_sync_v2_dataset(
    request: SyncDatasetEnrollRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        enrollment = service.enroll_dataset(
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            scope_type=request.scope_type,
            domains=request.domains,
            encryption_policy=request.encryption_policy,
            workspace_id=request.workspace_id,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            workspace_id=request.workspace_id,
        ) from exc
    dataset = enrollment.dataset
    return SyncDatasetEnrollResponse(
        dataset_id=dataset.dataset_id,
        scope_type=dataset.scope_type,
        encryption_policy=dataset.encryption_policy,
        domains=dataset.domains,
        workspace_id=dataset.workspace_id,
        cursors=enrollment.cursors,
        key_setup_required=enrollment.key_setup_required,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        metadata=dataset.metadata,
    )


@router.get(
    "/restore-manifest",
    response_model=SyncRestoreManifestResponse,
    summary="Return Sync v2 restore inventory metadata",
)
def get_sync_v2_restore_manifest(
    dataset_ids: list[str] | None = Query(None, alias="dataset_id"),
    domains: list[SyncDomain] | None = Query(None, alias="domain"),
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        manifest = service.restore_manifest(
            user_id=_sync_user_id(user),
            dataset_ids=dataset_ids,
            domains=domains,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_ids=dataset_ids,
            domains=domains,
        ) from exc
    return SyncRestoreManifestResponse(
        datasets=[asdict(dataset) for dataset in manifest.datasets],
        devices=[asdict(device) for device in manifest.devices],
        generated_at=manifest.generated_at,
        filters_applied=manifest.filters_applied,
    )


@router.post(
    "/push",
    response_model=SyncPushResponse,
    summary="Push Sync v2 envelopes",
)
def push_sync_v2_envelopes(
    request: SyncPushRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        result = service.push(
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            device_id=request.device_id,
            envelopes=[_core_envelope_from_api(envelope) for envelope in request.envelopes],
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            device_id=request.device_id,
            envelope_count=len(request.envelopes),
        ) from exc
    return SyncPushResponse(
        dataset_id=result.dataset_id,
        accepted=[SyncPushAcceptedEnvelope(**asdict(item)) for item in result.accepted],
        rejected=[SyncPushRejectedEnvelope(**asdict(item)) for item in result.rejected],
        conflicts=[SyncPushConflictEnvelope(**asdict(item)) for item in result.conflicts],
        next_cursor=result.next_cursor,
    )


@router.get(
    "/pull",
    response_model=SyncPullResponse,
    summary="Pull Sync v2 envelopes",
)
def pull_sync_v2_envelopes(
    dataset_id: str,
    device_id: str,
    cursor: str | None = None,
    domains: list[SyncDomain] | None = Query(None, alias="domain"),
    page_size: int | None = Query(None, ge=1),
    include_own_changes: bool = False,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        result = service.pull(
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            device_id=device_id,
            cursor=cursor,
            domains=domains,
            page_size=page_size,
            include_own_changes=include_own_changes,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            device_id=device_id,
            cursor=cursor,
        ) from exc
    return SyncPullResponse(
        dataset_id=result.dataset_id,
        envelopes=[
            _api_envelope_from_core(
                envelope,
                encryption_policy=result.encryption_policy,
            )
            for envelope in result.envelopes
        ],
        next_cursor=result.next_cursor,
        has_more=result.has_more,
    )


@router.get(
    "/conflicts",
    response_model=list[SyncConflictRecord],
    summary="List Sync v2 conflicts",
)
def list_sync_v2_conflicts(
    dataset_id: str,
    conflict_status: ConflictStatus | None = Query(None, alias="status"),
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        conflicts = service.list_conflicts(
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            status=conflict_status,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            conflict_status=conflict_status,
        ) from exc
    return [_api_conflict_from_core(conflict) for conflict in conflicts]


@router.post(
    "/conflicts/{conflict_id}/resolve",
    response_model=SyncConflictRecord,
    summary="Resolve a Sync v2 conflict",
)
def resolve_sync_v2_conflict(
    conflict_id: str,
    request: SyncConflictResolveRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        conflict = service.resolve_conflict(
            user_id=_sync_user_id(user),
            conflict_id=conflict_id,
            action=request.action,
            resolution_envelope=(
                _core_envelope_from_api(request.resolution_envelope)
                if request.resolution_envelope is not None
                else None
            ),
            resolved_by_envelope_id=(
                request.resolution_envelope.client_envelope_id
                if request.resolution_envelope is not None
                else None
            ),
            resolved_by_device_id=request.resolved_by_device_id,
            notes=request.notes,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            conflict_id=conflict_id,
            resolved_by_device_id=request.resolved_by_device_id,
        ) from exc
    return _api_conflict_from_core(conflict)


@router.get(
    "/keys/recovery-bundle",
    response_model=SyncKeyRecoveryBundleListResponse,
    summary="List Sync v2 encrypted key recovery material",
)
def list_sync_v2_key_recovery_bundles(
    dataset_id: str = Query(...),
    device_id: str | None = Query(None),
    key_purpose: str | None = Query("dataset_recovery"),
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        records = service.list_key_recovery_bundles(
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            device_id=device_id,
            key_purpose=key_purpose,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            device_id=device_id,
            key_purpose=key_purpose,
        ) from exc
    return SyncKeyRecoveryBundleListResponse(
        dataset_id=dataset_id,
        key_records=[_api_key_record_export(record) for record in records],
    )


@router.post(
    "/attachments",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Feature-detect Sync v2 attachment upload support",
)
def upload_sync_v2_attachment(
    _request: Request,
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    supports_attachments = service.capabilities().supports_attachments
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "error_code": "sync_attachments_not_enabled",
            "message": "Sync v2 attachment persistence is not enabled on this server.",
            "supports_attachments": supports_attachments,
        },
    )


@router.post(
    "/keys/recovery-bundle",
    summary="Store Sync v2 encrypted key recovery material",
)
def store_sync_v2_key_recovery_bundle(
    request: SyncKeyRecoveryBundleRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        record = service.store_key_recovery_bundle(
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            device_id=request.device_id,
            key_purpose=request.key_purpose,
            wrapped_key_blob=request.wrapped_key_blob,
            kdf_metadata=request.kdf_metadata,
            recovery_hint=request.recovery_hint,
            rotation_of_key_record_id=request.rotation_of_key_record_id,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            device_id=request.device_id,
            key_purpose=request.key_purpose,
        ) from exc
    return _api_key_record_metadata(record)


@router.post("/send",
             status_code=status.HTTP_200_OK,
             summary="Receive changes from a client")
async def receive_changes_from_client(
    payload: ClientChangesPayload,
    user_id: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user)
):
    """
    Receives a batch of sync log entries from a client, applies them
    to the user's database on the server synchronously in a thread pool.
    """
    requesting_client_id = payload.client_id
    if not payload.changes:
        logger.info(f"[{user_id.username}] Received empty change batch from client {requesting_client_id}.")
        return {"status": "success", "message": "No changes received."}

    logger.info(f"[{user_id.username}] Received {len(payload.changes)} changes from client {requesting_client_id}. Applying to DB: {db.db_path_str}")

    processor = ServerSyncProcessor(db=db, user_id=user_id.username, requesting_client_id=requesting_client_id)
    changes_as_dicts = [model_dump_compat(change) for change in payload.changes]

    try:
        # --- Run the SYNCHRONOUS method in a thread pool ---
        success, errors = await asyncio.to_thread(
            processor.apply_client_changes_batch,
            changes_as_dicts
        )
        # --- End Thread Pool Execution ---

        if not success:
            logger.error(f"[{user_id.username}] Failed processing batch from {requesting_client_id}: {errors}")
            # Use 400 Bad Request if errors suggest bad client data, 500 otherwise.
            is_client_error = _is_client_validation_error(errors)
            error_code = (
                status.HTTP_400_BAD_REQUEST
                if is_client_error
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            safe_errors = errors if is_client_error else ["Internal sync processing failed."]
            detail = {
                "message": "Failed to apply changes atomically.",
                "errors": safe_errors,
            }
            raise HTTPException(status_code=error_code, detail=detail)

        logger.info(f"[{user_id.username}] Successfully processed batch from client {requesting_client_id}.")
        return {"status": "success"}

    except HTTPException:
        raise
    except Exception as e:
        # Catch unexpected errors during thread execution or processing
        logger.exception(f"[{user_id.username}] Unexpected error in /send endpoint from client {requesting_client_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing sync changes.",
        ) from e


@router.get("/get",
            response_model=ServerChangesResponse,
            summary="Send changes back to a client")
async def send_changes_to_client(
    client_id: str,
    since_change_id: int = 0,
    user_id: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user)
):
    """
    Sends sync log entries from the user's server-side database back to
    the requesting client, starting after the `since_change_id`.
    Filters out changes that originated from the requesting client.
    Uses asyncio.to_thread for DB access.
    """
    logger.info(f"[{user_id.username}] Client '{client_id}' requesting changes since server log ID {since_change_id} from DB: {db.db_path_str}.")

    def _get_changes_sync():
        """Synchronous helper to fetch data"""
        changes_raw_list = []
        latest_id = 0
        try:
            # Note: Using the db object directly, assuming its methods are thread-safe
            # or that run_in_executor handles the context correctly.
            query = """
                SELECT change_id, entity, entity_uuid, operation, timestamp, client_id, version, payload
                FROM sync_log
                WHERE change_id > ? AND client_id != ?
                ORDER BY change_id ASC
                LIMIT ?
            """
            params = (since_change_id, client_id, SYNC_BATCH_SIZE)
            # Assuming db.execute_query is synchronous and thread-safe
            cursor_changes = db.execute_query(query, params)
            changes_raw_list = cursor_changes.fetchall() # This fetch happens synchronously

            cursor_latest = db.execute_query("SELECT MAX(change_id) FROM sync_log")
            latest_row = cursor_latest.fetchone()
            latest_id = latest_row[0] if latest_row and latest_row[0] is not None else 0
            return changes_raw_list, latest_id
        except (DatabaseError, sqlite3.Error) as db_err:
            logger.bind(error_type=type(db_err).__name__).error(
                "Sync DB error in _get_changes_sync"
            )
            # Re-raise to be caught by the main handler
            raise

    try:
        # --- Run the SYNCHRONOUS DB fetching in a thread pool ---
        changes_raw, latest_server_id = await asyncio.to_thread(_get_changes_sync)
        # --- End Thread Pool Execution ---

        changes_models: list[SyncLogEntry] = []
        invalid_rows: list[int] = []
        for row_dict in changes_raw: # Assuming fetchall returns list of dicts/Rows
             try:
                  # Convert row (which should be dict-like if row_factory is set) to SyncLogEntry model
                  row_data = dict(row_dict)
                  entry = SyncLogEntry(**row_data)
                  changes_models.append(entry)
             except Exception as pydantic_err:
                  row_change_id = "N/A"
                  try:
                      row_change_id = dict(row_dict).get("change_id", "N/A")
                  except Exception as change_id_error:
                      logger.debug("Failed to extract change_id from sync log row during validation error path", exc_info=change_id_error)
                  logger.bind(
                      error_type=type(pydantic_err).__name__,
                      change_id=row_change_id,
                  ).error(
                      "Invalid sync log entry encountered during /sync/get validation"
                  )
                  if isinstance(row_change_id, int):
                      invalid_rows.append(row_change_id)
                  continue

        if invalid_rows:
            logger.warning(
                f"[{user_id.username}] Invalid sync log rows encountered; returning partial /sync/get response. "
                f"invalid_change_ids={invalid_rows} valid_rows={len(changes_models)} latest_change_id={latest_server_id}"
            )

        logger.info(f"[{user_id.username}] Sending {len(changes_models)} changes to client '{client_id}'. Server latest ID: {latest_server_id}.")

        return ServerChangesResponse(
            changes=changes_models,
            latest_change_id=latest_server_id
        )

    except DatabaseError as e: # Catch mapped DB-layer errors raised from sync helper
        logger.bind(error_type=type(e).__name__).error(
            "Database error getting changes from sync store"
        )
        raise map_db_error_to_http(
            e,
            default_detail="Failed to retrieve changes from database.",
        ) from e
    except sqlite3.Error as e: # Catch raw sqlite errors raised from sync helper
        logger.bind(error_type=type(e).__name__).error(
            "Database error getting changes from sync store"
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve changes from database.") from e
    except HTTPException:
        raise
    except Exception as e: # Catch unexpected errors
        logger.bind(error_type=type(e).__name__).error(
            "Unexpected server error getting changes from sync store"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving sync changes.",
        ) from e

#
# End of API endpoints
#####################################################################################


# --- Server-Side Processing Logic ---
# Encapsulate server-side logic similar to ClientSyncEngine
class ServerSyncProcessor:
    """Handles applying changes received from a client to the server's user DB."""

    def __init__(self, db: Any, user_id: str, requesting_client_id: str):
        self.db = db
        self.user_id = user_id
        self.requesting_client_id = requesting_client_id # Client making the current API call
        logger.info(f"ServerSyncProcessor initialized for user '{self.user_id}', request from '{self.requesting_client_id}'.")

    @staticmethod
    def _validate_entity_operation(entity: str, operation: str) -> Optional[str]:
        if entity not in ALLOWED_SYNC_SEND_ENTITIES:
            return f"Unsupported sync entity '{entity}'"
        if operation not in ALLOWED_SYNC_OPERATIONS:
            return f"Unsupported sync operation '{operation}'"
        if entity == "MediaKeywords" and operation not in {"link", "unlink"}:
            return "MediaKeywords only supports 'link'/'unlink' operations"
        if entity != "MediaKeywords" and operation in {"link", "unlink"}:
            return f"Operation '{operation}' is only valid for MediaKeywords"
        return None

    @staticmethod
    def _parse_iso8601_timestamp_utc(timestamp_str: str, *, label: str) -> datetime:
        """
        Parse an ISO-8601 timestamp and normalize it to UTC.

        Accepted formats:
        - YYYY-MM-DDTHH:MM:SS.sssZ
        - YYYY-MM-DDTHH:MM:SSZ
        - YYYY-MM-DDTHH:MM:SS.sss+HH:MM
        - YYYY-MM-DDTHH:MM:SS+HH:MM
        """
        if not isinstance(timestamp_str, str) or not timestamp_str.strip():
            raise ValueError(f"{label} timestamp is empty or invalid")

        ts_value = timestamp_str.strip()
        parse_formats = (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
        )

        parse_error: ValueError | None = None
        for parse_format in parse_formats:
            try:
                parsed_dt = datetime.strptime(ts_value, parse_format)
                if parse_format.endswith("Z") or parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                return parsed_dt.astimezone(timezone.utc)
            except ValueError as exc:
                parse_error = exc

        raise ValueError(
            f"Could not parse {label} timestamp '{timestamp_str}' as ISO-8601 (Z or offset)."
        ) from parse_error

    # --- SYNCHRONOUS BATCH APPLICATION ---
    def apply_client_changes_batch(self, changes: list[dict]) -> tuple[bool, list[str]]:
        """
        Applies a batch of ordered changes received from a client within a single transaction.
        Returns (success_status, list_of_error_messages).
        THIS RUNS SYNCHRONOUSLY (intended to be called via asyncio.to_thread).
        """
        all_applied_or_skipped = True
        errors = []
        changes.sort(key=lambda x: x['change_id'])
        server_authoritative_timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        try:
            with self.db.transaction() as conn: # Use SYNCHRONOUS context manager
                cursor = conn.cursor()
                for change in changes:
                    try:
                        success, error_msg = self._apply_single_client_change_sync( # Call SYNC helper
                            cursor,
                            change,
                            server_authoritative_timestamp
                        )
                        if not success:
                            full_error = f"Failed Change ID {change.get('change_id', 'N/A')}: {error_msg}"
                            logger.error(f"[{self.user_id}] {full_error}")
                            errors.append(full_error)
                            all_applied_or_skipped = False
                            raise DatabaseError(f"Failed to apply change {change.get('change_id', 'N/A')}, rolling back batch.")

                    # Catch specific errors from _apply_single to ensure rollback
                    except (DatabaseError, sqlite3.Error, ConflictError, json.JSONDecodeError, KeyError, InputError, ValueError) as item_error:
                        error_msg = f"Failed Processing Change ID {change.get('change_id','?')}: {item_error}"
                        logger.error(f"[{self.user_id}] {error_msg}", exc_info=True)
                        if error_msg not in errors:
                            errors.append(error_msg) # Avoid duplicate if already logged
                        all_applied_or_skipped = False
                        raise # Re-raise to trigger transaction rollback

            # Transaction commits here if no exceptions raised

        except (DatabaseError, sqlite3.Error, ConflictError) as e:
            logger.error(f"Transaction rolled back applying changes for user {self.user_id} from client {self.requesting_client_id}: {e}")
            if not errors:
                errors.append(f"Transaction rolled back: {e}")
            all_applied_or_skipped = False
        except Exception as e:
            logger.error(f"Unexpected error applying client changes batch for user {self.user_id}: {e}", exc_info=True)
            errors.append(f"Unexpected server error: {e}")
            all_applied_or_skipped = False

        return all_applied_or_skipped, errors

    # --- SYNCHRONOUS SINGLE CHANGE APPLICATION ---
    def _apply_single_client_change_sync(self, cursor: sqlite3.Cursor, change: dict, current_server_time_str: str) -> tuple[bool, Optional[str]]:
        """
        Applies a single change from a client synchronously. Handles conflict detection/resolution.
        Returns (success, error_message). Runs within the transaction from apply_client_changes_batch.
        """
        entity = change['entity']
        entity_uuid = change['entity_uuid']
        operation = change['operation']
        client_version = change['version']
        originating_client_id = change['client_id']
        payload_str = change.get('payload')
        payload = {}
        error_msg = None

        validation_error = self._validate_entity_operation(str(entity), str(operation))
        if validation_error:
            logger.warning(f"[{self.user_id}] Rejecting sync change: {validation_error}")
            return False, validation_error

        # Payload validation
        if not payload_str and not (entity == "MediaKeywords" and operation in ['link', 'unlink']):
            error_msg = f"Missing payload for incoming change: {change.get('change_id', 'N/A')}"
            logger.error(f"[{self.user_id}] {error_msg}")
            return False, error_msg
        try:
            if payload_str:
                payload = json.loads(payload_str)
        except json.JSONDecodeError as e:
             error_msg = f"Failed to decode payload for change ID {change.get('change_id', 'N/A')}: {e}"
             logger.error(f"[{self.user_id}] {error_msg}", exc_info=True)
             return False, error_msg

        # MediaKeywords is a non-versioned junction table. Apply idempotently and
        # skip generic uuid/version lookups used by versioned entities.
        if entity == "MediaKeywords":
            try:
                self._execute_server_change_sql_sync(
                    cursor,
                    entity,
                    operation,
                    payload,
                    entity_uuid,
                    client_version,
                    originating_client_id,
                    current_server_time_str,
                    force_apply=False,
                )
                return True, None
            except (DatabaseError, sqlite3.Error, KeyError, ValueError) as e:
                error_msg = f"Error applying MediaKeywords change for {entity_uuid}: {e}"
                logger.error(f"[{self.user_id}] {error_msg}", exc_info=True)
                return False, error_msg

        try:
            # --- Fetch current server state ---
            cursor.execute(f"SELECT version, client_id, last_modified FROM `{entity}` WHERE uuid = ?", (entity_uuid,))  # nosec B608
            server_record_info = cursor.fetchone() # Sync fetchone
            server_version = server_record_info[0] if server_record_info else 0
            server_client_id = server_record_info[1] if server_record_info else None

            logger.debug(f"[{self.user_id}] Applying change: Op={operation}, Entity={entity}, UUID={entity_uuid}, ClientVer={client_version}, ServerVer={server_version}, OrigClient={originating_client_id}, ServerClient={server_client_id}")

            # --- Idempotency/Conflict Check ---
            if client_version <= server_version:
                if client_version < server_version or (client_version == server_version and originating_client_id == server_client_id):
                    logger.debug(f"[{self.user_id}] Server skipping old/duplicate change for {entity} UUID {entity_uuid}...")
                    return True, None # Skip
                elif client_version == server_version and originating_client_id != server_client_id:
                    logger.warning(f"[{self.user_id}] Server detected conflict for {entity} UUID {entity_uuid}. ClientVer==ServerVer, different clients.")
                    # Resolve conflict using sync helper
                    resolved, error_msg = self._resolve_server_conflict_sync(cursor, change, server_record_info, current_server_time_str)
                    return resolved, error_msg
                elif operation == 'create' and server_version > 0:
                    logger.warning(f"[{self.user_id}] Server skipping client 'create' for existing {entity} UUID {entity_uuid}...")
                    return True, None # Skip
                else:
                    logger.error(f"[{self.user_id}] Unhandled state in server idempotency check...")
                    return True, None # Skip defensively

            # --- Apply the change (if client_version > server_version) ---
            logger.debug(f"[{self.user_id}] Applying change with ClientVer {client_version} > ServerVer {server_version}")
            self._execute_server_change_sql_sync( # Call sync version
                cursor, entity, operation, payload, entity_uuid,
                client_version, originating_client_id, current_server_time_str,
                force_apply=False
            )
            return True, None # Success

        except ConflictError:
            # Optimistic lock failed during direct apply
            logger.warning(f"[{self.user_id}] Optimistic lock failed applying change for {entity} {entity_uuid}. Attempting resolution.")
            # Fetch current state again for resolution
            cursor.execute(f"SELECT version, client_id, last_modified FROM `{entity}` WHERE uuid = ?", (entity_uuid,))  # nosec B608
            current_server_state = cursor.fetchone() # Sync fetchone
            resolved, error_msg = self._resolve_server_conflict_sync( # Call sync version
                 cursor, change, current_server_state, current_server_time_str
            )
            return resolved, error_msg

        # Keep specific exception handling
        except (DatabaseError, sqlite3.Error, json.JSONDecodeError, KeyError, InputError, ValueError) as e:
            error_msg = f"Error applying single change for {entity} {entity_uuid}: {e}"
            logger.error(f"[{self.user_id}] {error_msg}", exc_info=True)
            return False, error_msg # Return failure status and message
        except Exception as e:
            error_msg = f"Unexpected error in _apply_single_client_change: {e}"
            logger.critical(f"[{self.user_id}] {error_msg}", exc_info=True)
            return False, error_msg # Return failure status and message

    # --- SYNCHRONOUS CONFLICT RESOLUTION ---
    def _resolve_server_conflict_sync(self, cursor: sqlite3.Cursor, client_change: dict, server_record_info: Optional[tuple], current_server_time_str: str) -> tuple[bool, Optional[str]]:
        """
        Resolves conflict synchronously using LWW based on parsed timestamps.
        Returns (resolved_successfully, error_message).
        """
        entity = client_change['entity']
        entity_uuid = client_change['entity_uuid']
        operation = client_change['operation']
        client_version = client_change['version']
        originating_client_id = client_change['client_id']
        payload_str = client_change.get('payload')
        payload = json.loads(payload_str) if payload_str else {}
        error_msg = None # Initialize

        if server_record_info is None:
            logger.error(f"[{self.user_id}] Cannot resolve conflict for {entity} {entity_uuid}: server record info is missing.")
            return False, "Cannot resolve conflict without server record state."

        server_record_info[0]
        server_last_modified_str = server_record_info[2] if len(server_record_info) > 2 and isinstance(
            server_record_info[2], str) else '1970-01-01T00:00:00Z'

        # --- LWW Resolution (using datetime objects) ---
        try:
            server_dt = self._parse_iso8601_timestamp_utc(
                server_last_modified_str,
                label="server",
            )
            authoritative_dt = self._parse_iso8601_timestamp_utc(
                current_server_time_str,
                label="authoritative",
            )

        except ValueError as parse_err:
            error_msg = f"Timestamp parsing error during conflict resolution: {parse_err}"
            logger.error(
                f"[{self.user_id}] {error_msg} (Server='{server_last_modified_str}', Authoritative='{current_server_time_str}')",
                exc_info=True)
            return False, error_msg  # Return failure

            # --- Compare datetime objects ---
        logger.debug(
            f"[{self.user_id}] Comparing Timestamps: ServerDT={server_dt}, AuthoritativeDT={authoritative_dt}")  # Add debug log
        if server_dt >= authoritative_dt:
            logger.warning(
                f"[{self.user_id}] Resolving Conflict (LWW): Server wins (ServerDT {server_dt} >= AuthoritativeDT {authoritative_dt}). Skipping client change.")
            return True, None  # Resolved by skipping client change
        else:
            # Client wins - proceed with force apply
            logger.warning(
                f"[{self.user_id}] Resolving Conflict (LWW): Client change wins (ServerDT {server_dt} < AuthoritativeDT {authoritative_dt}). Forcing apply.")
            try:
                # Ensure correct arguments are passed, especially the timestamp string
                self._execute_server_change_sql_sync(
                    cursor, entity, operation, payload, entity_uuid,
                    client_version, originating_client_id, current_server_time_str,
                    # Pass the authoritative string timestamp
                    force_apply=True
                )
                return True, None
            # ... (Keep existing exception handling for force apply errors) ...
            except (DatabaseError, sqlite3.Error, ValueError) as e:
                error_msg = f"Failed to force apply winning change: {e}"
                logger.error(
                    f"[{self.user_id}] Error forcing update during LWW conflict resolution for {entity} {entity_uuid}: {e}",
                    exc_info=True)
                return False, error_msg
            except Exception as e:
                error_msg = f"Unexpected server error during resolution force apply: {e}"
                logger.critical(f"[{self.user_id}] Unexpected error during conflict resolution force apply: {e}",
                                exc_info=True)
                return False, error_msg

    # --- SYNCHRONOUS SQL EXECUTION ---
    def _execute_server_change_sql_sync(self, cursor: sqlite3.Cursor, entity: str, operation: str, payload: dict, uuid: str,
                                     client_version: int, originating_client_id: str, server_timestamp: str,
                                     force_apply: bool = False):
        """Executes SQL synchronously on server DB."""
        logger.debug(f"[{self.user_id}] Executing Server SQL (Sync): Op='{operation}', Entity='{entity}', UUID='{uuid}', ClientVer='{client_version}', Force='{force_apply}'")

        # --- Special handling for MediaKeywords ---
        if entity == "MediaKeywords":
            return self._execute_server_media_keyword_sql_sync(cursor, operation, payload)

        # --- Standard handling ---
        table_columns = self._get_table_columns_sync(cursor, entity)
        if not table_columns:
            raise DatabaseError(f"Cannot proceed: Failed to get columns for entity '{entity}'.")

        # --- Version Calculation (Server Side - Sync) ---
        current_server_version = 0
        cursor.execute(f"SELECT version FROM `{entity}` WHERE uuid = ?", (uuid,))  # nosec B608
        current_rec = cursor.fetchone()
        if current_rec:
            current_server_version = current_rec[0]

        target_sql_version = client_version
        if force_apply:
            target_sql_version = current_server_version + 1
            logger.debug(f"[{self.user_id}] Forcing apply (Sync): Setting target SQL version to {target_sql_version}")

        # --- Optimistic Lock (Sync - same logic) ---
        optimistic_lock_sql = ""
        optimistic_lock_param = []
        expected_base_version = client_version - 1
        if not force_apply and operation in ['update', 'delete']:
             if expected_base_version > 0:
                 optimistic_lock_sql = " AND version = ?"
                 optimistic_lock_param = [expected_base_version]
             else:
                 logger.debug(f"[{self.user_id}] Applying {operation} (Sync) based on client version {client_version}...")

        sql = ""
        params_tuple = ()

        # --- SQL Generation (Sync) ---
        if operation == 'create':
            target_sql_version = 1
            cols_sql, placeholders_sql, params_list = [], [], []
            core_sync_meta = {'uuid': uuid, 'last_modified': server_timestamp, 'version': target_sql_version, 'client_id': originating_client_id}
            all_data = {**payload, **core_sync_meta, 'deleted': payload.get('deleted', 0)}
            for col in table_columns:
                if col == 'id':
                    continue
                if col in all_data:
                    value = all_data[col]
                    if isinstance(value, bool):
                        value = 1 if value else 0
                    cols_sql.append(f"`{col}`")
                    placeholders_sql.append("?")
                    params_list.append(value)
            if not cols_sql:
                raise ValueError(f"Cannot build INSERT for {entity} {uuid}: No columns")
            sql = f"INSERT OR IGNORE INTO `{entity}` ({', '.join(cols_sql)}) VALUES ({', '.join(placeholders_sql)})"
            params_tuple = tuple(params_list)

        elif operation == 'update':
            set_clauses, params_list = [], []
            for col in table_columns:
                 if col in payload and col not in ['id', 'uuid']:
                     value = payload[col]
                     if isinstance(value, bool):
                         value = 1 if value else 0
                     set_clauses.append(f"`{col}` = ?")
                     params_list.append(value)

            set_clauses.extend(["`last_modified` = ?", "`version` = ?", "`client_id` = ?", "`deleted` = ?"])
            params_list.extend([server_timestamp, target_sql_version, originating_client_id, 1 if payload.get('deleted', False) else 0])

            if not set_clauses:
                 logger.warning(
                     "[{}] Update operation for {} {} resulted in no SET clauses.",
                     self.user_id,
                     entity,
                     uuid,
                 )
                 return # Skip

            where_clause = " WHERE uuid = ?"
            where_params = [uuid]
            if not force_apply:
                 where_clause += optimistic_lock_sql
                 where_params.extend(optimistic_lock_param)

            set_clause_sql = ", ".join(set_clauses)
            update_sql_template = "UPDATE `{entity}` SET {set_clause_sql}{where_clause}"
            sql = update_sql_template.format_map(locals())  # nosec B608
            params_tuple = tuple(params_list + where_params)

        elif operation == 'delete':
            where_clause = " WHERE uuid = ?"
            where_params = [uuid]
            if not force_apply:
                 where_clause += optimistic_lock_sql
                 where_params.extend(optimistic_lock_param)
            delete_sql_template = (
                "UPDATE `{entity}` SET deleted = 1, last_modified = ?, version = ?, client_id = ?{where_clause}"
            )
            sql = delete_sql_template.format_map(locals())  # nosec B608
            params_tuple = tuple([server_timestamp, target_sql_version, originating_client_id] + where_params)

        else:
            raise ValueError(f"Unsupported operation '{operation}' from client for entity '{entity}'")

        # --- Execute SQL (Sync) ---
        try:
            logger.debug(f"[{self.user_id}] Executing Server SQL (Sync): {sql} | Params: {params_tuple}")
            cursor.execute(sql, params_tuple)
            # Optimistic Lock Check
            if not force_apply and operation in ['update', 'delete'] and optimistic_lock_sql:
                if cursor.rowcount == 0:
                    logger.warning(f"[{self.user_id}] Optimistic lock failed applying client change (Sync)...")
                    raise ConflictError("Optimistic lock failed applying client change.", entity=entity, identifier=uuid)
                else:
                    logger.debug(f"[{self.user_id}] Optimistic lock successful applying client change (Sync). Rowcount {cursor.rowcount}.")

            # Keep FTS index synchronized for sync-path Media/Keywords writes.
            conn = getattr(cursor, "connection", None)
            if conn is None:
                raise DatabaseError("Sync write completed but cursor has no connection for FTS refresh.")
            self.db.sync_refresh_fts_for_entity(
                conn,
                entity=entity,
                entity_uuid=uuid,
                operation=operation,
                payload=payload,
            )

        except sqlite3.IntegrityError as ie:
            logger.error(f"[{self.user_id}] Integrity error applying client change (Sync) for {entity} {uuid}: {ie}", exc_info=True)
            raise DatabaseError(f"Integrity error applying client change: {ie}") from ie
        except sqlite3.Error as e:
            logger.error(f"[{self.user_id}] SQLite error applying client change (Sync) for {entity} {uuid}: {e}", exc_info=True)
            raise

    # --- Define SYNCHRONOUS versions of helpers ---
    _server_column_cache = {}
    def _get_table_columns_sync(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[list[str]]:
         """Synchronous version for getting table columns."""
         if table_name in self._server_column_cache:
             return self._server_column_cache[table_name]
         try:
              if not table_name.replace('_','').isalnum():
                  raise ValueError(f"Invalid table name: {table_name}")
              cursor.execute(f"PRAGMA table_info(`{table_name}`)")
              columns_info = cursor.fetchall() # Sync fetchall
              columns = [row[1] for row in columns_info]
              if columns:
                   self._server_column_cache[table_name] = columns
                   logger.debug(f"[{self.user_id}] Cached columns for server table '{table_name}': {columns}")
                   return columns
              else:
                  logger.error(f"[{self.user_id}] Could not retrieve columns for server table: {table_name}")
                  return None
         except (sqlite3.Error, ValueError) as e:
             logger.error(f"[{self.user_id}] Error getting columns for server table {table_name}: {e}")
             return None

    def _execute_server_media_keyword_sql_sync(self, cursor: sqlite3.Cursor, operation: str, payload: dict):
        """Synchronous version for MediaKeywords SQL execution."""
        media_uuid = payload.get('media_uuid')
        keyword_uuid = payload.get('keyword_uuid')
        if not media_uuid or not keyword_uuid:
            raise ValueError(f"Missing UUIDs for MediaKeywords {operation}")
        cursor.execute("SELECT id FROM Media WHERE uuid = ?", (media_uuid,))
        media_rec = cursor.fetchone() # Sync
        cursor.execute("SELECT id FROM Keywords WHERE uuid = ?", (keyword_uuid,))
        kw_rec = cursor.fetchone() # Sync
        if not media_rec:
            logger.warning(f"[{self.user_id}] Skipping Server MediaKeywords {operation}: Media UUID {media_uuid} not found.")
            return
        if not kw_rec:
            logger.warning(f"[{self.user_id}] Skipping Server MediaKeywords {operation}: Keyword UUID {keyword_uuid} not found.")
            return
        media_id_local, keyword_id_local = media_rec[0], kw_rec[0]
        if operation == 'link':
             sql = "INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)"
             params_tuple = (media_id_local, keyword_id_local)
             logger.debug(f"[{self.user_id}] Executing Server SQL (Sync): {sql} | Params: {params_tuple}")
             cursor.execute(sql, params_tuple) # Sync
        elif operation == 'unlink':
             sql = "DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?"
             params_tuple = (media_id_local, keyword_id_local)
             logger.debug(f"[{self.user_id}] Executing Server SQL (Sync): {sql} | Params: {params_tuple}")
             cursor.execute(sql, params_tuple) # Sync
        else:
            raise ValueError(f"Unsupported server operation '{operation}' for MediaKeywords entity")


    def _update_fts_manually_sync(self, cursor: sqlite3.Cursor, entity: str, payload: dict, uuid: str):
         """Updates the corresponding FTS table manually on the server (sync version)."""
         if entity == 'Media':
             if 'title' in payload or 'content' in payload:
                 cursor.execute("SELECT id, title, content FROM Media WHERE uuid = ?", (uuid,))
                 media_row = cursor.fetchone() # Sync
                 if media_row:
                     media_id, current_title, current_content = media_row
                     new_title = payload.get('title', current_title)
                     new_content = payload.get('content', current_content)
                     try:
                          logger.debug(f"[{self.user_id}] Manually updating media_fts (Sync) for Media UUID {uuid} (ID: {media_id})")
                          cursor.execute("UPDATE media_fts SET title = ?, content = ? WHERE rowid = ?", (new_title, new_content, media_id)) # Sync
                          if cursor.rowcount == 0:
                              logger.warning(f"[{self.user_id}] FTS row not found for update (Media ID: {media_id}), attempting insert.")
                              cursor.execute("INSERT INTO media_fts (rowid, title, content) VALUES (?, ?, ?)", (media_id, new_title, new_content)) # Sync
                     except sqlite3.Error as fts_err:
                         logger.error(f"[{self.user_id}] Failed to manually update media_fts (Sync) for Media ID {media_id}: {fts_err}", exc_info=True)
                 else:
                     logger.warning(f"[{self.user_id}] Cannot update media_fts (Sync): Media record not found for UUID {uuid}")
         elif entity == 'Keywords':
             if 'keyword' in payload:
                 cursor.execute("SELECT id FROM Keywords WHERE uuid = ?", (uuid,))
                 kw_row = cursor.fetchone() # Sync
                 if kw_row:
                     kw_id = kw_row[0]
                     new_keyword = payload['keyword']
                     try:
                          logger.debug(f"[{self.user_id}] Manually updating keyword_fts (Sync) for Keyword UUID {uuid} (ID: {kw_id})")
                          cursor.execute("UPDATE keyword_fts SET keyword = ? WHERE rowid = ?", (new_keyword, kw_id)) # Sync
                          if cursor.rowcount == 0:
                              logger.warning(f"[{self.user_id}] FTS row not found for update (Keyword ID: {kw_id}), attempting insert.")
                              cursor.execute("INSERT INTO keyword_fts (rowid, keyword) VALUES (?, ?)", (kw_id, new_keyword)) # Sync
                     except sqlite3.Error as fts_err:
                         logger.error(f"[{self.user_id}] Failed to manually update keyword_fts (Sync) for Keyword ID {kw_id}: {fts_err}", exc_info=True)
                 else:
                     logger.warning(f"[{self.user_id}] Cannot update keyword_fts (Sync): Keyword record not found for UUID {uuid}")

#
# End of sync-endpoint.py
#######################################################################################################################
