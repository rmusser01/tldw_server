"""
Datasets endpoints extracted from evaluations_unified.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import (
    create_error_response,
    get_evaluation_identity,
    get_eval_request_user,
    require_eval_permissions,
    sanitize_error_message,
    verify_api_key,
)
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    CreateDatasetRequest,
    DatasetListResponse,
    DatasetResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.permissions import EVALS_MANAGE, EVALS_READ
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS,
    get_unified_evaluation_service_for_user,
)
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

datasets_router = APIRouter()


def _normalize_dataset_payload(dataset: dict[str, Any]) -> dict[str, Any]:
    """Ensure required dataset response fields are populated."""
    # Work on a shallow copy to avoid mutating upstream caches
    normalized = dict(dataset)
    normalized.setdefault("object", "dataset")

    created = normalized.get("created")
    created_at = normalized.get("created_at")

    timestamp: Optional[int] = None
    if isinstance(created, (int, float)):
        timestamp = int(created)
    elif isinstance(created_at, (int, float)):
        timestamp = int(created_at)
    elif isinstance(created_at, str):
        try:
            # Support both ISO-8601 and SQLite timestamp formats
            ts = created_at.replace("Z", "+00:00")
            timestamp = int(datetime.fromisoformat(ts).timestamp())
        except (ValueError, TypeError, OverflowError):
            timestamp = None

    if timestamp is None:
        timestamp = int(datetime.now(timezone.utc).timestamp())

    normalized["created"] = timestamp
    normalized["created_at"] = timestamp
    return normalized


@datasets_router.post(
    "/datasets",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_eval_permissions(EVALS_MANAGE))],
)
async def create_dataset(
    dataset_request: CreateDatasetRequest,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    response: Response = ...,
):
    try:
        identity = get_evaluation_identity(current_user)
        svc = get_unified_evaluation_service_for_user(identity.user_scope)
        if idempotency_key:
            try:
                existing_id = svc.db.lookup_idempotency("dataset", idempotency_key, identity.created_by)
                if existing_id:
                    existing = await svc.get_dataset(existing_id, created_by=identity.created_by)
                    if existing:
                        try:
                            if response is not None:
                                response.headers["X-Idempotent-Replay"] = "true"
                                response.headers["Idempotency-Key"] = idempotency_key
                        except (AttributeError, TypeError, ValueError) as e:
                            logger.debug(f"Failed to set idempotency headers: {e}")
                        return DatasetResponse(**_normalize_dataset_payload(existing))
            except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Idempotency lookup failed, proceeding with creation: {e}")
        dataset_id = await svc.create_dataset(
            name=dataset_request.name,
            samples=[model_dump_compat(s) for s in dataset_request.samples],
            description=dataset_request.description or "",
            metadata=model_dump_compat(dataset_request.metadata) if dataset_request.metadata else None,
            created_by=identity.created_by,
        )
        row = await svc.get_dataset(dataset_id, created_by=identity.created_by)
        normalized = _normalize_dataset_payload(row)
        try:
            if idempotency_key:
                svc.db.record_idempotency("dataset", idempotency_key, dataset_id, identity.created_by)
        except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Failed to record idempotency key for dataset {dataset_id}: {e}")
        return DatasetResponse(**normalized)
    except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
        logger.exception(f"Failed to create dataset: {e}")
        raise create_error_response(
            message=f"Failed to create dataset: {sanitize_error_message(e, 'creating dataset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@datasets_router.get(
    "/datasets",
    response_model=DatasetListResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def list_datasets(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity = get_evaluation_identity(current_user)
        svc = get_unified_evaluation_service_for_user(identity.user_scope)
        items, has_more = svc.db.list_datasets(limit=limit, offset=offset, created_by=identity.created_by)
        resp = [DatasetResponse(**_normalize_dataset_payload(r)) for r in items]
        first_id = resp[0].id if resp else None
        last_id = resp[-1].id if resp else None
        return DatasetListResponse(data=resp, has_more=has_more, first_id=first_id, last_id=last_id)
    except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
        logger.exception(f"Failed to list datasets: {e}")
        raise create_error_response(
            message=f"Failed to list datasets: {sanitize_error_message(e, 'listing datasets')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@datasets_router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def get_dataset(
    dataset_id: str,
    include_samples: bool = Query(True),
    limit: int | None = Query(None, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity = get_evaluation_identity(current_user)
        svc = get_unified_evaluation_service_for_user(identity.user_scope)
        row = svc.db.get_dataset(
            dataset_id,
            created_by=identity.created_by,
            include_samples=include_samples,
            sample_limit=limit,
            sample_offset=offset,
        )
        if not row:
            raise create_error_response(
                message="Dataset not found",
                error_type="not_found_error",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return DatasetResponse(**_normalize_dataset_payload(row))
    except HTTPException:
        raise
    except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to get dataset: {e}")
        raise create_error_response(
            message=f"Failed to get dataset: {sanitize_error_message(e, 'retrieving dataset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@datasets_router.delete(
    "/datasets/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[Depends(require_eval_permissions(EVALS_MANAGE))],
)
async def delete_dataset(
    dataset_id: str,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
) -> Response:
    try:
        identity = get_evaluation_identity(current_user)
        svc = get_unified_evaluation_service_for_user(identity.user_scope)
        ok = await svc.delete_dataset(
            dataset_id,
            deleted_by=identity.created_by,
            created_by=identity.created_by,
        )
        if not ok:
            raise create_error_response(
                message="Dataset not found",
                error_type="not_found_error",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except _UNIFIED_EVAL_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to delete dataset: {e}")
        raise create_error_response(
            message=f"Failed to delete dataset: {sanitize_error_message(e, 'deleting dataset')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e
