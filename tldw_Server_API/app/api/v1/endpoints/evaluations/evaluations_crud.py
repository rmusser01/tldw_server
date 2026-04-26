"""
Evaluations CRUD and Runs endpoints extracted from evaluations_unified.
"""

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    rbac_rate_limit,
    require_token_scope,
)
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import (
    check_evaluation_rate_limit,
    create_error_response,
    get_evaluation_identity,
    get_eval_request_user,
    require_eval_permissions,
    sanitize_error_message,
    verify_api_key,
)
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    CreateEvaluationRequest,
    DatasetOverride,
    EvaluationListResponse,
    EvaluationResponse,
    RunListResponse,
    RunResponse,
    UpdateEvaluationRequest,
)
from tldw_Server_API.app.core.AuthNZ.permissions import EVALS_MANAGE, EVALS_READ
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Evaluations.audit_adapter import MandatoryAuditWriteError
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    get_unified_evaluation_service_for_user,
)
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

_EVALS_CRUD_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


class CreateRunSimpleRequest(BaseModel):
    """Create run request for CRUD endpoint.

    Relaxed variant that allows free-form config while forbidding extra top-level keys.
    """
    model_config = ConfigDict(extra='forbid')
    target_model: Optional[str] = Field(default=None, description="Model to evaluate")
    dataset_override: Optional[DatasetOverride] = Field(default=None, description="Override dataset for this run")
    config: dict[str, Any] = Field(default_factory=dict, description="Run configuration (free-form)")
    webhook_url: Optional[str] = Field(default=None, description="Optional webhook URL for run events")


crud_router = APIRouter()
# Define once so dependency overrides work reliably in tests and downstream apps
RBAC_EVALS_CREATE = rbac_rate_limit("evals.create")


def _get_crud_identity_and_service(current_user: User):
    identity = get_evaluation_identity(current_user)
    return identity, get_unified_evaluation_service_for_user(identity.user_scope)


@crud_router.post(
    "/",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[
        Depends(RBAC_EVALS_CREATE),
        Depends(require_eval_permissions(EVALS_MANAGE)),
    ],
)
async def create_evaluation(
    eval_request: CreateEvaluationRequest,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    response: Response = None,
):
    try:
        identity, svc = _get_crud_identity_and_service(current_user)
        if idempotency_key:
            try:
                existing_id = svc.db.lookup_idempotency("evaluation", idempotency_key, identity.created_by)
                if existing_id:
                    existing = await svc.get_evaluation(existing_id, created_by=identity.created_by)
                    if existing:
                        try:
                            if response is not None:
                                response.headers["X-Idempotent-Replay"] = "true"
                                response.headers["Idempotency-Key"] = idempotency_key
                        except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"evaluations_crud: failed to set idempotency headers for {existing_id}: {e}")
                        return EvaluationResponse(**existing)
            except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"evaluations_crud: idempotency lookup failed for key {idempotency_key}: {e}")
        evaluation = await svc.create_evaluation(
            name=eval_request.name,
            description=eval_request.description,
            eval_type=eval_request.eval_type,
            eval_spec=model_dump_compat(eval_request.eval_spec),
            dataset_id=eval_request.dataset_id,
            dataset=[model_dump_compat(s) for s in eval_request.dataset] if eval_request.dataset else None,
            metadata=model_dump_compat(eval_request.metadata) if eval_request.metadata else None,
            created_by=identity.created_by,
        )
        try:
            if idempotency_key and evaluation.get("id"):
                svc.db.record_idempotency("evaluation", idempotency_key, evaluation["id"], identity.created_by)
        except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(
                f"evaluations_crud: failed to record idempotency for evaluation {evaluation.get('id')}: {e}"
            )
        return EvaluationResponse(**evaluation)
    except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to create evaluation: {e}")
        raise create_error_response(
            message=f"Failed to create evaluation: {sanitize_error_message(e, 'evaluation creation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@crud_router.get(
    "/",
    response_model=EvaluationListResponse,
    dependencies=[
        Depends(require_eval_permissions(EVALS_READ)),
    ],
)
async def list_evaluations(
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
    eval_type: Optional[str] = Query(None),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity, svc = _get_crud_identity_and_service(current_user)
        evaluations, has_more = await svc.list_evaluations(
            limit=limit,
            after=after,
            eval_type=eval_type,
            created_by=identity.created_by,
        )
        first_id = evaluations[0]["id"] if evaluations else None
        last_id = evaluations[-1]["id"] if evaluations else None
        return EvaluationListResponse(
            object="list",
            data=[EvaluationResponse(**eval) for eval in evaluations],
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )
    except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to list evaluations: {e}")
        raise create_error_response(
            message=f"Failed to list evaluations: {sanitize_error_message(e, 'listing evaluations')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@crud_router.get(
    "/{eval_id}",
    response_model=EvaluationResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def get_evaluation(
    eval_id: str,
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity, svc = _get_crud_identity_and_service(current_user)
        evaluation = await svc.get_evaluation(eval_id, created_by=identity.created_by)
        if not evaluation:
            raise create_error_response(
                message="Evaluation not found",
                error_type="not_found_error",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return EvaluationResponse(**evaluation)
    except HTTPException:
        raise
    except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to get evaluation: {e}")
        raise create_error_response(
            message=f"Failed to get evaluation: {sanitize_error_message(e, 'retrieving evaluation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@crud_router.patch(
    "/{eval_id}",
    response_model=EvaluationResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_MANAGE))],
)
async def update_evaluation(
    eval_id: str,
    update_request: UpdateEvaluationRequest,
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity, svc = _get_crud_identity_and_service(current_user)
        # Only include explicitly provided fields; avoid overwriting with None
        updates = model_dump_compat(update_request, exclude_none=True, exclude_unset=True)
        evaluation = await svc.update_evaluation(
            eval_id,
            updates,
            updated_by=identity.created_by,
            created_by=identity.created_by,
        )
        if not evaluation:
            raise create_error_response(
                message="Evaluation not found",
                error_type="not_found_error",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return EvaluationResponse(**evaluation)
    except HTTPException:
        raise
    except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to update evaluation: {e}")
        raise create_error_response(
            message=f"Failed to update evaluation: {sanitize_error_message(e, 'updating evaluation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@crud_router.delete(
    "/{eval_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[Depends(require_eval_permissions(EVALS_MANAGE))],
)
async def delete_evaluation(
    eval_id: str,
    current_user: User = Depends(get_eval_request_user),
) -> Response:
    try:
        identity, svc = _get_crud_identity_and_service(current_user)
        success = await svc.delete_evaluation(
            eval_id,
            deleted_by=identity.created_by,
            created_by=identity.created_by,
        )
        if not success:
            raise create_error_response(
                message="Evaluation not found",
                error_type="not_found_error",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to delete evaluation: {e}")
        raise create_error_response(
            message=f"Failed to delete evaluation: {sanitize_error_message(e, 'deleting evaluation')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@crud_router.post(
    "/{eval_id}/runs",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[
        Depends(require_eval_permissions(EVALS_MANAGE)),
        Depends(check_evaluation_rate_limit),
        Depends(require_token_scope(
            "workflows",
            require_if_present=True,
            require_schedule_match=False,
            allow_admin_bypass=True,
            endpoint_id="evals.create_run",
            count_as="run",
        )),
    ],
)
async def create_run(
    eval_id: str,
    request: CreateRunSimpleRequest,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    response: Response = None,
):
    try:
        identity, svc = _get_crud_identity_and_service(current_user)
        if idempotency_key:
            try:
                existing_id = svc.db.lookup_idempotency("run", idempotency_key, identity.created_by)
                if existing_id:
                    existing = await svc.get_run(existing_id, created_by=identity.created_by)
                    if existing:
                        try:
                            if response is not None:
                                response.headers["X-Idempotent-Replay"] = "true"
                                response.headers["Idempotency-Key"] = idempotency_key
                        except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(
                                f"evaluations_crud: failed to set idempotency headers for {existing_id}: {e}"
                            )
                        return RunResponse(**existing)
            except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"evaluations_crud: idempotency lookup failed for key {idempotency_key}: {e}")
        target_model = request.target_model
        # Allow free-form config; convert Pydantic models if provided in future
        config = model_dump_compat(request.config) if hasattr(request.config, 'model_dump') else (request.config or {})
        dataset_override = model_dump_compat(request.dataset_override) if request.dataset_override else None
        webhook_url = request.webhook_url
        run = await svc.create_run(
            eval_id=eval_id,
            target_model=target_model,
            config=config,
            dataset_override=dataset_override,
            webhook_url=webhook_url,
            created_by=identity.created_by,
            webhook_user_id=identity.webhook_user_id,
        )
        try:
            if idempotency_key and run.get("id"):
                svc.db.record_idempotency("run", idempotency_key, run["id"], identity.created_by)
        except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"evaluations_crud: failed to record idempotency for run {run.get('id')}: {e}")
        return RunResponse(**run)
    except HTTPException:
        raise
    except MandatoryAuditWriteError as e:
        raise create_error_response(
            message="Mandatory audit persistence unavailable",
            error_type="service_unavailable",
            code="audit_persistence_failure",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        ) from e
    except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to create run: {e}")
        raise create_error_response(
            message=f"Failed to create run: {sanitize_error_message(e, 'creating run')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@crud_router.get(
    "/{eval_id}/runs",
    response_model=RunListResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def list_runs(
    eval_id: str,
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity, svc = _get_crud_identity_and_service(current_user)
        runs, has_more = await svc.list_runs(
            eval_id=eval_id,
            status=status,
            limit=limit,
            after=after,
            created_by=identity.created_by,
        )
        first_id = runs[0]["id"] if runs else None
        last_id = runs[-1]["id"] if runs else None
        return RunListResponse(object="list", data=[RunResponse(**run) for run in runs], has_more=has_more, first_id=first_id, last_id=last_id)
    except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to list runs: {e}")
        raise create_error_response(
            message=f"Failed to list runs: {sanitize_error_message(e, 'listing runs')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@crud_router.get(
    "/runs/{run_id}",
    response_model=RunResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def get_run(
    run_id: str,
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity, svc = _get_crud_identity_and_service(current_user)
        run = await svc.get_run(run_id, created_by=identity.created_by)
        if not run:
            raise create_error_response(
                message="Run not found",
                error_type="not_found_error",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return RunResponse(**run)
    except HTTPException:
        raise
    except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to get run: {e}")
        raise create_error_response(
            message=f"Failed to get run: {sanitize_error_message(e, 'retrieving run')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@crud_router.post(
    "/runs/{run_id}/cancel",
    dependencies=[Depends(require_eval_permissions(EVALS_MANAGE))],
)
async def cancel_run(
    run_id: str,
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity, svc = _get_crud_identity_and_service(current_user)
        ok = await svc.cancel_run(
            run_id,
            cancelled_by=identity.created_by,
            created_by=identity.created_by,
        )
        if not ok:
            raise create_error_response(
                message="Run not found",
                error_type="not_found_error",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return {"status": "cancelled", "run_id": run_id}
    except HTTPException:
        raise
    except _EVALS_CRUD_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to cancel run: {e}")
        raise create_error_response(
            message=f"Failed to cancel run: {sanitize_error_message(e, 'cancelling run')}",
            error_type="server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e
