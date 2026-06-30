# admin_bundle_ops.py
# Description: Admin REST endpoints for backup bundle operations.
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import (
    build_offset_pagination_meta,
)
from tldw_Server_API.app.api.v1.schemas.bundle_schemas import (
    BundleCreateRequest,
    BundleCreateResponse,
    BundleDeleteResponse,
    BundleImportResponse,
    BundleImportValidation,
    BundleItem,
    BundleListResponse,
    BundleMetadataResponse,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.exceptions import (
    BundleConcurrencyError,
    BundleDiskSpaceError,
    BundleError,
    BundleExportError,
    BundleImportError,
    BundleNotFoundError,
    BundleRateLimitError,
    BundleSchemaIncompatibleError,
)
from tldw_Server_API.app.services import admin_bundle_service as svc

router = APIRouter()

_BUNDLE_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    HTTPException,
)


async def _emit_admin_audit_event(
    request: Request,
    principal: AuthPrincipal,
    *,
    event_type: str,
    category: str,
    resource_type: str,
    resource_id: str | None,
    action: str,
    metadata: dict[str, Any],
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    await admin_mod._emit_admin_audit_event(
        request,
        principal,
        event_type=event_type,
        category=category,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        metadata=metadata,
    )


async def _enforce_admin_user_scope(
    principal: AuthPrincipal,
    target_user_id: int,
    *,
    require_hierarchy: bool,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    await admin_mod._enforce_admin_user_scope(
        principal,
        target_user_id,
        require_hierarchy=require_hierarchy,
    )


def _metadata_to_item(m: svc.BundleMetadata) -> BundleItem:
    return BundleItem(
        bundle_id=m.bundle_id,
        user_id=m.user_id,
        created_at=m.created_at,
        size_bytes=m.size_bytes,
        datasets=m.datasets,
        schema_versions=m.schema_versions,
        app_version=m.app_version,
        manifest_version=m.manifest_version,
        notes=m.notes,
    )


def _handle_bundle_error(exc: BundleError) -> HTTPException:
    """Map domain exceptions to HTTP responses using structured error_code."""
    code = getattr(exc, "error_code", "bundle_error")

    # --- Export errors ---
    if isinstance(exc, BundleExportError):
        if code == "vector_store_export_not_supported":
            return HTTPException(status_code=422, detail=code)
        if code == "user_id_required":
            return HTTPException(status_code=400, detail=code)
        return HTTPException(status_code=400, detail=code)

    # --- Specific typed exceptions with fixed status codes ---
    if isinstance(exc, BundleNotFoundError):
        return HTTPException(status_code=404, detail=code)
    if isinstance(exc, BundleSchemaIncompatibleError):
        return HTTPException(status_code=409, detail=code)
    if isinstance(exc, BundleConcurrencyError):
        return HTTPException(status_code=409, detail=code)
    if isinstance(exc, BundleRateLimitError):
        retry_after = str(getattr(exc, "retry_after", 3600))
        return HTTPException(
            status_code=429,
            detail=code,
            headers={"Retry-After": retry_after},
        )
    if isinstance(exc, BundleDiskSpaceError):
        return HTTPException(status_code=507, detail=code)

    # --- Import errors ---
    if isinstance(exc, BundleImportError):
        if code == "restore_failed":
            detail: dict[str, Any] = {
                "error_code": code,
                "message": str(exc),
            }
            rollback_failures = getattr(exc, "rollback_failures", None)
            if rollback_failures:
                detail["rollback_failures"] = list(rollback_failures)
            return HTTPException(status_code=400, detail=detail)
        if code in (
            "checksum_verification_failed",
            "unsupported_manifest_version",
            "user_id_required",
        ):
            return HTTPException(status_code=400, detail=code)
        return HTTPException(status_code=400, detail=code)

    return HTTPException(status_code=500, detail="bundle_operation_failed")


# ---------------------------------------------------------------------------
# Route 1: POST /backups/bundles
# ---------------------------------------------------------------------------
@router.post("/backups/bundles", response_model=BundleCreateResponse)
async def create_bundle(
    payload: BundleCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> BundleCreateResponse:
    try:
        # Validate datasets
        if payload.datasets:
            all_known = {"media", "chacha", "prompts", "evaluations", "audit", "authnz"}
            for ds in payload.datasets:
                if ds.strip().lower() not in all_known:
                    raise HTTPException(status_code=400, detail="unknown_dataset")

        # Scope enforcement for per-user datasets
        if payload.user_id is not None:
            await _enforce_admin_user_scope(principal, payload.user_id, require_hierarchy=False)

        admin_uid = getattr(principal, "user_id", 0) or 0
        try:
            admin_uid = int(admin_uid)
        except (TypeError, ValueError):
            admin_uid = 0

        result = await svc.create_bundle_async(
            datasets=[ds.strip().lower() for ds in payload.datasets] if payload.datasets else None,
            user_id=payload.user_id,
            include_vector_store=payload.include_vector_store,
            notes=payload.notes,
            max_backups=payload.max_backups,
            retention_hours=payload.retention_hours,
            admin_user_id=admin_uid,
        )

        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="system",
            resource_type="bundle",
            resource_id=result.bundle_id,
            action="backup.bundle.create",
            metadata={
                "datasets": result.datasets,
                "user_id": result.user_id,
                "size_bytes": result.size_bytes,
            },
        )

        return BundleCreateResponse(
            item=_metadata_to_item(result),
            status="created",
            message="Bundle created successfully",
        )
    except BundleError as exc:
        raise _handle_bundle_error(exc) from exc
    except HTTPException:
        raise
    except _BUNDLE_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to create bundle")
        raise HTTPException(status_code=500, detail="Failed to create bundle") from exc


# ---------------------------------------------------------------------------
# Route 2: GET /backups/bundles
# ---------------------------------------------------------------------------
@router.get("/backups/bundles", response_model=BundleListResponse)
async def list_bundles(
    user_id: int | None = Query(None, description="Filter by user_id"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> BundleListResponse:
    try:
        if user_id is not None:
            await _enforce_admin_user_scope(principal, user_id, require_hierarchy=False)

        items, total = await asyncio.to_thread(svc.list_bundles, user_id=user_id, limit=limit, offset=offset)
        return BundleListResponse(
            items=[_metadata_to_item(m) for m in items],
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(items),
            ),
        )
    except BundleError as exc:
        raise _handle_bundle_error(exc) from exc
    except HTTPException:
        raise
    except _BUNDLE_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list bundles")
        raise HTTPException(status_code=500, detail="Failed to list bundles") from exc


# ---------------------------------------------------------------------------
# Route 3: POST /backups/bundles/import  (MUST be before {bundle_id})
# ---------------------------------------------------------------------------
@router.post("/backups/bundles/import", response_model=BundleImportResponse)
async def import_bundle(
    request: Request,
    file: UploadFile = File(...),
    user_id: int | None = Query(None),
    dry_run: bool = Query(False),
    allow_downgrade: bool = Query(False),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> BundleImportResponse:
    tmp_path = None
    try:
        if user_id is not None:
            await _enforce_admin_user_scope(principal, user_id, require_hierarchy=False)

        # Save upload to temp file using chunked streaming
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip", prefix="tldw_bundle_import_") as tmp:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = tmp.name

        admin_uid = getattr(principal, "user_id", 0) or 0
        try:
            admin_uid = int(admin_uid)
        except (TypeError, ValueError):
            admin_uid = 0

        result = await svc.import_bundle_async(
            file_path=tmp_path,
            user_id=user_id,
            dry_run=dry_run,
            allow_downgrade=allow_downgrade,
            admin_user_id=admin_uid,
        )

        if not dry_run and result.get("status") == "imported":
            await _emit_admin_audit_event(
                request,
                principal,
                event_type="data.write",
                category="system",
                resource_type="bundle",
                resource_id=None,
                action="backup.bundle.import",
                metadata={
                    "datasets_restored": result.get("datasets_restored", []),
                    "user_id": user_id,
                    "dry_run": dry_run,
                },
            )

        return BundleImportResponse(
            status=result["status"],
            datasets_restored=result.get("datasets_restored", []),
            warnings=result.get("warnings", []),
            safety_snapshots=result.get("safety_snapshots", {}),
            rollback_failures=result.get("rollback_failures", []),
            validations=[BundleImportValidation(**v) for v in result.get("validations", [])],
        )
    except BundleError as exc:
        raise _handle_bundle_error(exc) from exc
    except HTTPException:
        raise
    except _BUNDLE_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to import bundle")
        raise HTTPException(status_code=500, detail="Failed to import bundle") from exc
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Route 4: GET /backups/bundles/{bundle_id}
# ---------------------------------------------------------------------------
@router.get("/backups/bundles/{bundle_id}", response_model=BundleMetadataResponse)
async def get_bundle_metadata(
    bundle_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> BundleMetadataResponse:
    try:
        meta = await asyncio.to_thread(svc.get_bundle_metadata, bundle_id)
        return BundleMetadataResponse(item=_metadata_to_item(meta))
    except BundleError as exc:
        raise _handle_bundle_error(exc) from exc
    except HTTPException:
        raise
    except _BUNDLE_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to get bundle metadata")
        raise HTTPException(status_code=500, detail="Failed to get bundle metadata") from exc


# ---------------------------------------------------------------------------
# Route 5: GET /backups/bundles/{bundle_id}/download
# ---------------------------------------------------------------------------
@router.get(
    "/backups/bundles/{bundle_id}/download",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Backup bundle zip file",
            "content": {"application/zip": {}},
        },
    },
)
async def download_bundle(
    bundle_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> FileResponse:
    try:
        path = await asyncio.to_thread(svc.get_bundle_path, bundle_id)

        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.read",
            category="system",
            resource_type="bundle",
            resource_id=bundle_id,
            action="backup.bundle.download",
            metadata={"bundle_id": bundle_id},
        )

        return FileResponse(
            path=path,
            media_type="application/zip",
            filename=bundle_id,
        )
    except BundleError as exc:
        raise _handle_bundle_error(exc) from exc
    except HTTPException:
        raise
    except _BUNDLE_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to download bundle")
        raise HTTPException(status_code=500, detail="Failed to download bundle") from exc


# ---------------------------------------------------------------------------
# Route 6: DELETE /backups/bundles/{bundle_id}
# ---------------------------------------------------------------------------
@router.delete("/backups/bundles/{bundle_id}", response_model=BundleDeleteResponse)
async def delete_bundle_endpoint(
    bundle_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> BundleDeleteResponse:
    try:
        await asyncio.to_thread(svc.delete_bundle, bundle_id)

        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="system",
            resource_type="bundle",
            resource_id=bundle_id,
            action="backup.bundle.delete",
            metadata={"bundle_id": bundle_id},
        )

        return BundleDeleteResponse(status="deleted", bundle_id=bundle_id)
    except BundleError as exc:
        raise _handle_bundle_error(exc) from exc
    except HTTPException:
        raise
    except _BUNDLE_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to delete bundle")
        raise HTTPException(status_code=500, detail="Failed to delete bundle") from exc
