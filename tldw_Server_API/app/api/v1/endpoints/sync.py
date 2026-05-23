# Server_API/app/api/v1/endpoints/sync-endpoint.py
# Description: This code provides a FastAPI endpoint for all Sync operations.
#
# Imports
import asyncio
import json
from dataclasses import asdict
from typing import Any

#
# 3rd-party imports
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user

#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
    ConflictStatus,
    SyncAttachmentUploadRequest,
    SyncAttachmentUploadResponse,
    SyncBlobChunkUploadResponse,
    SyncBlobDownloadManifestResponse,
    SyncBlobUploadCompleteResponse,
    SyncBlobUploadCreateRequest,
    SyncBlobUploadSessionResponse,
    SyncCapabilitiesResponse,
    SyncConflictRecord,
    SyncConflictResolveRejectedItem,
    SyncConflictResolveRequest,
    SyncConflictResolveResolvedItem,
    SyncConflictResolveResponse,
    SyncDatasetEnrollRequest,
    SyncDatasetEnrollResponse,
    SyncDeviceRegisterRequest,
    SyncDeviceRegisterResponse,
    SyncDomain,
    SyncKeyRecoveryBundleListResponse,
    SyncKeyRecoveryBundleRecord,
    SyncKeyRecoveryBundleRequest,
    SyncProfileBootstrapRequest,
    SyncProfileBootstrapResponse,
    SyncProfileResponse,
    SyncPullResponse,
    SyncPushAcceptedEnvelope,
    SyncPushConflictEnvelope,
    SyncPushRejectedEnvelope,
    SyncPushRequest,
    SyncPushResponse,
    SyncRepairRequest,
    SyncRepairResponse,
    SyncRestoreManifestResponse,
    SyncRestorePreviewRequest,
    SyncRestorePreviewResponse,
    SyncV2Envelope,
)
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.factory import (
    default_sync_v2_registry,
    sync_v2_service_for_user,
    sync_v2_storage_exists_for_user,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_env,
)
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


def _sync_user_id(user: User) -> str:
    user_id = getattr(user, "id_str", "") or str(getattr(user, "id", "") or "")
    return user_id or user.username


_default_sync_v2_registry = default_sync_v2_registry


def get_sync_v2_service(
    user: User = Depends(get_request_user),
) -> SyncV2Service:
    """Build the per-user Sync v2 service dependency."""

    return sync_v2_service_for_user(_sync_user_id(user))


def get_sync_v2_profile_service(
    user: User = Depends(get_request_user),
) -> SyncV2Service | None:
    """Build a profile service only when Sync v2 storage already exists."""

    user_id = _sync_user_id(user)
    if not sync_v2_storage_exists_for_user(user_id):
        return None
    return sync_v2_service_for_user(user_id)


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
        if "sync_blob_transfer_not_supported" in lowered:
            return HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail={
                    "error_code": "sync_blob_transfer_not_supported",
                    "message": "Sync v2 M1 does not support binary blob transfer.",
                },
            )
        if "sync_encryption_attestation_required" in lowered:
            return HTTPException(
                status_code=status.HTTP_412_PRECONDITION_FAILED,
                detail={
                    "error_code": "sync_encryption_attestation_required",
                    "message": (
                        "Sync v2 M1 requires deployment-level at-rest encryption "
                        "coverage before profile bootstrap or dataset enrollment."
                    ),
                },
            )
        if "attachment payload exceeds" in lowered:
            return HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error_code": "sync_attachment_too_large",
                    "message": "Sync attachment exceeds the server size limit.",
                },
            )
        if "quota" in lowered:
            return HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error_code": "sync_blob_quota_exceeded",
                    "message": "Sync blob quota would be exceeded.",
                },
            )
        if (
            "invalid sync cursor" in lowered
            or "page_size" in lowered
            or "resolution envelope" in lowered
            or "payload exceeds" in lowered
            or "chunk" in lowered
            or "hash" in lowered
            or "bootstrap mode" in lowered
            or "requested unsupported domains" in lowered
            or "client_family" in lowered
            or "client_profile_id" in lowered
            or "key recovery bundle" in lowered
            or "wrapping metadata" in lowered
            or "key purpose" in lowered
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
        exclude={
            "envelope_id",
            "server_cursor",
            "server_sequence",
            "object_revision",
            "received_at_server",
            "server_timestamp",
            "status",
            "apply_status",
            "encryption_policy",
        },
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


def _api_profile_from_core(profile: Any) -> SyncProfileResponse:
    return SyncProfileResponse(**asdict(profile))


def _api_bootstrap_profile_from_core(profile: Any) -> SyncProfileBootstrapResponse:
    return SyncProfileBootstrapResponse(**asdict(profile))


def _api_blob_session_from_core(session: Any) -> SyncBlobUploadSessionResponse:
    return SyncBlobUploadSessionResponse(**asdict(session))


def _api_empty_profile(user_id: str, device_id: str | None) -> SyncProfileResponse:
    encryption_status = server_trusted_encryption_status_from_env()
    device = None
    if device_id is not None:
        device = {
            "device_id": device_id,
            "registered": False,
        }
    return SyncProfileResponse(
        protocol_version="sync-v2-m1",
        min_supported_protocol_version="sync-v2-m1",
        profile_bootstrapped=False,
        user_id=user_id,
        active_dataset_id=None,
        device=device,
        dataset=None,
        server_cursor=0,
        capabilities=SyncCapabilitiesResponse(
            encryption=encryption_status.encryption,
            warnings=encryption_status.warnings,
        ),
        domain_status=[],
        warnings=list(encryption_status.warnings),
    )


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


@router.get(
    "/profile",
    response_model=SyncProfileResponse,
    summary="Return Sync v2 M1 profile state",
)
def get_sync_v2_profile(
    device_id: str | None = Query(None),
    user: User = Depends(get_request_user),
    service: SyncV2Service | None = Depends(get_sync_v2_profile_service),
):
    user_id = _sync_user_id(user)
    if service is None:
        return _api_empty_profile(user_id, device_id)
    try:
        profile = service.profile(
            user_id=user_id,
            device_id=device_id,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=user_id,
            device_id=device_id,
        ) from exc
    return _api_profile_from_core(profile)


@router.post(
    "/profile/bootstrap",
    response_model=SyncProfileBootstrapResponse,
    summary="Bootstrap a Sync v2 M1 Chatbook profile",
)
def bootstrap_sync_v2_profile(
    request: SyncProfileBootstrapRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        profile = service.bootstrap_profile(
            user_id=_sync_user_id(user),
            mode=request.mode,
            device_id=request.device_id,
            device_name=request.device_name,
            client_profile_id=request.client_profile_id,
            client_family=request.client_family,
            client_instance=request.client_instance,
            requested_domains=request.requested_domains,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            device_id=request.device_id,
            mode=request.mode,
        ) from exc
    return _api_bootstrap_profile_from_core(profile)


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
    "/restore/preview",
    response_model=SyncRestorePreviewResponse,
    summary="Preview a Sync v2 M1 restore plan",
)
def preview_sync_v2_restore(
    request: SyncRestorePreviewRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        preview = service.restore_preview(
            user_id=_sync_user_id(user),
            dataset_ids=request.dataset_ids,
            domains=request.domains,
            selected_object_ids=request.selected_object_ids,
            selected_attachment_ids=request.selected_attachment_ids,
            metadata_only=request.metadata_only,
            local_inventory=[model_dump_compat(item) for item in request.local_inventory],
            attachment_availability=request.attachment_availability,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_ids=request.dataset_ids,
            domains=request.domains,
        ) from exc
    return SyncRestorePreviewResponse(**asdict(preview))


@router.post(
    "/repair",
    response_model=SyncRepairResponse,
    summary="Replay accepted Sync v2 envelopes into server projections",
)
def repair_sync_v2_projections(
    request: SyncRepairRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        result = service.repair(
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            domains=request.domains,
            since_cursor=request.since_cursor,
            failed_only=request.failed_only,
            limit=request.limit,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            domains=request.domains,
        ) from exc
    return SyncRepairResponse(**asdict(result))


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
    "/conflicts/resolve",
    response_model=SyncConflictResolveResponse,
    summary="Resolve Sync v2 conflicts",
)
def resolve_sync_v2_conflicts(
    request: SyncConflictResolveRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    user_id = _sync_user_id(user)
    resolved: list[SyncConflictResolveResolvedItem] = []
    rejected: list[SyncConflictResolveRejectedItem] = []
    server_cursors: list[int] = []
    for resolution in request.resolutions:
        resolution_envelope = (
            _core_envelope_from_api(resolution.resolution_envelope)
            if resolution.resolution_envelope is not None
            else None
        )
        try:
            conflict = service.resolve_conflict(
                user_id=user_id,
                dataset_id=request.dataset_id,
                conflict_id=resolution.conflict_id,
                action=resolution.action,
                resolution_envelope=resolution_envelope,
                resolved_by_device_id=request.device_id,
                notes=None,
            )
        except Exception as exc:
            logger.bind(
                error_type=type(exc).__name__,
                user_id=user_id,
                dataset_id=request.dataset_id,
                device_id=request.device_id,
                conflict_id=resolution.conflict_id,
            ).warning("Sync v2 conflict resolution item failed")
            rejected.append(
                SyncConflictResolveRejectedItem(
                    conflict_id=resolution.conflict_id,
                    action=resolution.action,
                    error_code="sync_conflict_resolution_failed",
                    message="Conflict resolution could not be applied.",
                    retryable=False,
                )
            )
            continue
        if conflict.server_cursor is not None:
            server_cursors.append(conflict.server_cursor)
        resolved.append(
            SyncConflictResolveResolvedItem(
                conflict_id=conflict.conflict_id,
                action=resolution.action,
                status=conflict.status,
                envelope_id=conflict.resolved_by_envelope_id,
                server_cursor=conflict.server_cursor,
            )
        )
    return SyncConflictResolveResponse(
        dataset_id=request.dataset_id,
        server_cursor=max(server_cursors, default=None),
        resolved=resolved,
        rejected=rejected,
    )


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
    "/blob-uploads",
    response_model=SyncBlobUploadSessionResponse,
    summary="Create or resume a Sync v2 M2 blob upload session",
)
def create_sync_v2_blob_upload(
    request: SyncBlobUploadCreateRequest,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        session = service.create_blob_upload_session(
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            device_id=request.device_id,
            domain=request.domain,
            entity_id=request.entity_id,
            attachment_id=request.attachment_id,
            content_type=request.content_type,
            size_bytes=request.size_bytes,
            payload_hash=request.payload_hash,
            chunk_size=request.chunk_size,
            chunk_count=request.chunk_count,
            idempotency_key=request.idempotency_key,
            encryption_policy=request.encryption_policy,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            domain=request.domain,
            attachment_id=request.attachment_id,
        ) from exc
    return _api_blob_session_from_core(session)


@router.get(
    "/blob-uploads/{upload_id}",
    response_model=SyncBlobUploadSessionResponse,
    summary="Return Sync v2 M2 blob upload session status",
)
def get_sync_v2_blob_upload(
    upload_id: str,
    dataset_id: str = Query(...),
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        session = service.get_blob_upload_session(
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            upload_id=upload_id,
        ) from exc
    return _api_blob_session_from_core(session)


@router.put(
    "/blob-uploads/{upload_id}/chunks/{chunk_index}",
    response_model=SyncBlobChunkUploadResponse,
    summary="Upload one raw Sync v2 M2 blob chunk",
)
async def upload_sync_v2_blob_chunk(
    raw_request: Request,
    upload_id: str,
    chunk_index: int,
    dataset_id: str = Query(...),
    offset_bytes: int = Query(..., ge=0),
    chunk_hash: str = Query(...),
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    payload = await raw_request.body()
    try:
        chunk = await asyncio.to_thread(
            service.upload_blob_chunk,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            upload_id=upload_id,
            chunk_index=chunk_index,
            offset_bytes=offset_bytes,
            chunk_payload=payload,
            chunk_hash=chunk_hash,
        )
        session = await asyncio.to_thread(
            service.get_blob_upload_session,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            upload_id=upload_id,
            chunk_index=chunk_index,
        ) from exc
    return SyncBlobChunkUploadResponse(
        upload_id=chunk.upload_id,
        chunk_index=chunk.chunk_index,
        accepted=True,
        size_bytes=chunk.size_bytes,
        chunk_hash=chunk.chunk_hash,
        missing_chunks=session.missing_chunks,
    )


@router.post(
    "/blob-uploads/{upload_id}/complete",
    response_model=SyncBlobUploadCompleteResponse,
    summary="Complete and verify a Sync v2 M2 blob upload",
)
def complete_sync_v2_blob_upload(
    upload_id: str,
    dataset_id: str = Query(...),
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    user_id = _sync_user_id(user)
    try:
        blob = service.complete_blob_upload(
            user_id=user_id,
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
        quota = service.store.summarize_blob_quota(user_id, dataset_id=dataset_id)
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=user_id,
            dataset_id=dataset_id,
            upload_id=upload_id,
        ) from exc
    return SyncBlobUploadCompleteResponse(
        upload_id=upload_id,
        dataset_id=blob.dataset_id,
        attachment_id=blob.attachment_id,
        blob_id=blob.blob_id,
        status=blob.status,
        stored=blob.status == "available",
        size_bytes=blob.size_bytes,
        payload_hash=blob.payload_hash,
        download_url=f"/api/v1/sync/attachments/{blob.attachment_id}?dataset_id={blob.dataset_id}",
        quota=asdict(quota),
    )


@router.delete(
    "/blob-uploads/{upload_id}",
    response_model=SyncBlobUploadSessionResponse,
    summary="Cancel a Sync v2 M2 blob upload session",
)
def cancel_sync_v2_blob_upload(
    upload_id: str,
    dataset_id: str = Query(...),
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        session = service.cancel_blob_upload(
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            upload_id=upload_id,
        ) from exc
    return _api_blob_session_from_core(session)


@router.post(
    "/attachments",
    response_model=SyncAttachmentUploadResponse,
    summary="Upload a small encrypted Sync v2 attachment",
)
async def upload_sync_v2_attachment(
    raw_request: Request,
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    """Validate and persist a small encrypted Sync v2 attachment upload."""

    if not service.settings.supports_attachments:
        raise _safe_sync_v2_http_error(
            SyncStoreError(
                "sync_blob_transfer_not_supported: Sync v2 M1 does not support binary blob transfer"
            ),
            user_id=_sync_user_id(user),
        )

    try:
        request = SyncAttachmentUploadRequest.model_validate(await raw_request.json())
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.bind(error_type=type(exc).__name__).warning(
            "Sync v2 attachment request validation failed"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "sync_validation_failed",
                "message": "Sync attachment request is invalid.",
            },
        ) from exc
    try:
        attachment = await asyncio.to_thread(
            service.store_attachment,
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            domain=request.domain,
            entity_id=request.entity_id,
            attachment_id=request.attachment_id,
            content_type=request.content_type,
            size_bytes=request.size_bytes,
            payload_ciphertext=request.payload_ciphertext,
            payload_hash=request.payload_hash,
            encryption_policy=request.encryption_policy,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=request.dataset_id,
            domain=request.domain,
            attachment_id=request.attachment_id,
        ) from exc
    return SyncAttachmentUploadResponse(
        attachment_id=attachment.attachment_id,
        dataset_id=attachment.dataset_id,
        stored=attachment.stored,
        size_bytes=attachment.size_bytes,
        payload_hash=attachment.payload_hash,
    )


@router.get(
    "/attachments/{attachment_id}/manifest",
    response_model=SyncBlobDownloadManifestResponse,
    summary="Return a Sync v2 M2 attachment blob download manifest",
)
def get_sync_v2_attachment_download_manifest(
    attachment_id: str,
    dataset_id: str = Query(...),
    chunk_size: int | None = Query(None, ge=1),
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        manifest = service.blob_download_manifest(
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            attachment_id=attachment_id,
            chunk_size=chunk_size,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            attachment_id=attachment_id,
        ) from exc
    return SyncBlobDownloadManifestResponse.model_validate(asdict(manifest))


@router.get(
    "/attachments/{attachment_id}",
    summary="Download a Sync v2 attachment blob",
)
def download_sync_v2_attachment(
    attachment_id: str,
    dataset_id: str = Query(...),
    offset: int = Query(0, ge=0),
    size: int | None = Query(None, ge=1),
    user: User = Depends(get_request_user),
    service: SyncV2Service = Depends(get_sync_v2_service),
):
    try:
        manifest = service.blob_download_manifest(
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            attachment_id=attachment_id,
            chunk_size=size,
        )
        payload = service.read_blob_bytes(
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            attachment_id=attachment_id,
            offset=offset,
            size=size,
        )
    except Exception as exc:
        raise _safe_sync_v2_http_error(
            exc,
            user_id=_sync_user_id(user),
            dataset_id=dataset_id,
            attachment_id=attachment_id,
        ) from exc
    return Response(
        content=payload,
        media_type=manifest.content_type,
        headers={
            "Accept-Ranges": "bytes",
            "X-Sync-Payload-Hash": manifest.payload_hash,
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


def _legacy_sync_replaced_error(replacement: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_410_GONE,
        detail={
            "error_code": "sync_legacy_endpoint_replaced",
            "message": (
                "The legacy sync endpoint has been replaced by the Sync v2 M1 "
                "envelope API."
            ),
            "replacement": replacement,
        },
    )


@router.post("/send",
             status_code=status.HTTP_410_GONE,
             summary="Legacy sync send endpoint replaced by Sync v2 push")
async def receive_changes_from_client(
    request: Request,
    user_id: User = Depends(get_request_user),
):
    """Return a stable replacement response for the removed legacy sync send API."""

    del request, user_id
    raise _legacy_sync_replaced_error("/api/v1/sync/push")


@router.get("/get",
            status_code=status.HTTP_410_GONE,
            summary="Legacy sync get endpoint replaced by Sync v2 pull")
async def send_changes_to_client(
    request: Request,
    user_id: User = Depends(get_request_user),
):
    """Return a stable replacement response for the removed legacy sync get API."""

    del request, user_id
    raise _legacy_sync_replaced_error("/api/v1/sync/pull")

#
# End of sync-endpoint.py
#######################################################################################################################
