"""
Prompt Studio Evaluations API

Runs and manages evaluations of prompts against selected test cases.
Exposes synchronous and asynchronous evaluation creation, and listing
of previous evaluations with pagination.

Key responsibilities
- Create evaluations (sync or background job)
- List evaluations filtered by project/prompt
- Persist metrics for later analysis and comparison

Security
- Project-scoped access controls
- Background execution via FastAPI BackgroundTasks or job queue
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status

from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    get_prompt_studio_db,
    get_prompt_studio_user,
    require_project_write_access,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.endpoints.prompt_studio.resource_binding import (
    authoritative_prompt_project,
    require_test_cases_in_project,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_schemas import (
    EvaluationCreate,
    EvaluationList,
    EvaluationResponse,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    derive_trusted_credential_scope,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    record_byok_missing_credentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    capture_provider_override_call_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
    ProviderCredentialRuntime,
    mark_provider_credential_used,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker
from tldw_Server_API.app.core.Chat.streaming_utils import sanitized_provider_stream_exception
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError, PromptStudioDatabase
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import provider_auth_is_resolved
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.testing import is_test_mode

router = APIRouter(prefix="/api/v1/prompt-studio", tags=["prompt-studio"])

from tldw_Server_API.app.core.Logging.log_context import (
    ensure_request_id,
    ensure_traceparent,
    get_ps_logger,
    log_context,
)

_PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    DatabaseError,
    FileNotFoundError,
    ImportError,
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
    json.JSONDecodeError,
)


def _is_prompt_studio_test_mode() -> bool:
    return is_test_mode() or os.getenv("PYTEST_CURRENT_TEST") is not None


def _prompt_studio_credential_http_exception(exc: ByokResolutionError) -> HTTPException:
    """Map credential policy/storage failures to bounded Prompt Studio responses."""
    code = getattr(exc, "policy_code", exc.code)
    if code in {"provider_disabled", "model_not_allowed", "credential_scope_revoked"}:
        message = (
            "The active credential scope is no longer available."
            if code == "credential_scope_revoked"
            else "The selected provider or model is disabled by administrator policy."
        )
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": code,
                "message": message,
            },
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error_code": code,
            "message": "Provider credentials are temporarily unavailable.",
        },
    )


def _prompt_studio_provider_auth_is_resolved(
    provider: str,
    credentials: ProviderCallCredentials,
) -> bool:
    """Honor provider-specific auth contracts such as the AWS default chain."""

    return provider_auth_is_resolved(
        provider,
        api_key=credentials.api_key,
        app_config=credentials.app_config,
        credentials_resolved=credentials.credentials_resolved,
    )


@router.post(
    "/evaluations",
    response_model=EvaluationResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Create an evaluation",
                            "value": {
                                "project_id": 1,
                                "prompt_id": 12,
                                "name": "Baseline Eval",
                                "test_case_ids": [1, 2, 3],
                                "model_configs": [
                                    {"model_name": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 256}
                                ]
                            }
                        },
                        "with_program_evaluator": {
                            "summary": "Evaluation with Program Evaluator (feature flag)",
                            "description": "Enable PROMPT_STUDIO_ENABLE_CODE_EVAL=true and use test cases with runner=\"python\". See Prompt Studio README for safety and runner spec.",
                            "value": {
                                "project_id": 1,
                                "prompt_id": 12,
                                "name": "Code Eval",
                                "test_case_ids": [101, 102],
                                "model_configs": [
                                    {"model_name": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 256}
                                ]
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Evaluation created",
                "content": {
                    "application/json": {
                        "examples": {
                            "created": {
                                "summary": "Created evaluation",
                                "value": {
                                    "id": 501,
                                    "uuid": "e1b2...",
                                    "project_id": 1,
                                    "prompt_id": 12,
                                    "status": "running",
                                    "created_at": "2024-09-21T10:00:00"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def create_evaluation(
    evaluation: EvaluationCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user),
) -> EvaluationResponse:
    """
    Create a new evaluation for a prompt.

    Args:
        evaluation: Evaluation configuration
        background_tasks: FastAPI background tasks
        db: Database instance
        user_context: Current user context

    Returns:
        Created evaluation response
    """
    credential_runtime: ProviderCredentialRuntime | None = None
    credential_runtime_transferred = False
    try:
        _, project_id = authoritative_prompt_project(
            db,
            evaluation.prompt_id,
            compatibility_project_id=evaluation.project_id,
        )
        await require_project_write_access(
            project_id,
            user_context=user_context,
            db=db,
        )
        test_case_ids = require_test_cases_in_project(
            db,
            evaluation.test_case_ids or [],
            project_id,
        )

        # Normalize incoming model configuration to a list of dicts for storage.
        # Support both legacy shape (model_configs: List[dict]) and new shape (config: dict).
        incoming_configs = None
        try:
            incoming_configs = evaluation.model_configs
        except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
            incoming_configs = None

        if incoming_configs and isinstance(incoming_configs, list):
            configs_list: list[dict[str, Any]] = [
                item.model_dump(exclude_none=True)
                if hasattr(item, "model_dump")
                else dict(item)
                for item in incoming_configs
                if hasattr(item, "model_dump") or isinstance(item, dict)
            ]
        else:
            single_cfg = getattr(evaluation, "config", None)
            if single_cfg is not None:
                try:
                    # Support pydantic model or plain dict
                    if hasattr(single_cfg, "model_dump"):
                        cfg_dict = single_cfg.model_dump(exclude_none=True)
                    elif isinstance(single_cfg, dict):
                        cfg_dict = single_cfg
                    else:
                        cfg_dict = {}
                except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
                    cfg_dict = {}
                configs_list = [cfg_dict] if cfg_dict else []
            else:
                configs_list = []

        # Determine effective config to run with (first item if provided)
        first_cfg = configs_list[0] if configs_list else {}
        provider_name = canonical_provider_name(
            (first_cfg.get("provider") or first_cfg.get("api_name") or "openai").strip() or "openai"
        )
        configured_model = first_cfg.get("model_name") or first_cfg.get("model")
        if configured_model:
            model_name = configured_model
        elif provider_name == "openai":
            model_name = "gpt-3.5-turbo"
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Model is required for an explicit non-OpenAI provider.",
            )
        temperature = first_cfg.get("temperature", 0.7)
        max_tokens = first_cfg.get("max_tokens", 1000)
        timeout_seconds = first_cfg.get("timeout_seconds")
        provider_key = provider_name

        # Metrics-only evaluations do not dispatch an LLM and need no credentials.
        if evaluation.metrics is not None:
            return EvaluationResponse(
                id=0,
                uuid=str(uuid.uuid4()),
                project_id=project_id,
                prompt_id=evaluation.prompt_id,
                name=evaluation.name or "Evaluation",
                description=evaluation.description or "",
                status="completed",
                created_at=datetime.now().isoformat(),
                metrics=evaluation.metrics.model_dump() if hasattr(evaluation.metrics, "model_dump") else dict(evaluation.metrics),
                config=evaluation.config.model_dump() if hasattr(evaluation.config, "model_dump") and evaluation.config else {},
            )

        user_id_int = None
        try:
            user_id_int = int(user_context.get("user_id"))
        except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
            user_id_int = None

        runtime_user_id, team_ids, org_ids, trusted_base_url_override = (
            derive_trusted_credential_scope(request, None)
        )
        if runtime_user_id is None:
            runtime_user_id = user_id_int
        credential_runtime = ProviderCredentialRuntime(
            user_id=runtime_user_id,
            team_ids=team_ids,
            org_ids=org_ids,
            trusted_base_url_override=trusted_base_url_override,
            override_snapshot_resolver=capture_provider_override_call_snapshot,
        )
        provider_credentials = await credential_runtime.resolve(
            provider_key,
            model=model_name,
        )
        provider_api_key = provider_credentials.api_key
        app_config_override = provider_credentials.app_config

        async def _mark_provider_success() -> None:
            await mark_provider_credential_used(
                credential_runtime,
                provider_credentials,
            )

        if (
            provider_requires_api_key(provider_key)
            and not _prompt_studio_provider_auth_is_resolved(
                provider_key,
                provider_credentials,
            )
            and not _is_prompt_studio_test_mode()
        ):
            record_byok_missing_credentials(provider_key, operation="prompt_studio")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "missing_provider_credentials",
                    "message": f"Provider '{provider_name}' requires an API key.",
                },
            )

        # Use EvaluationManager for sync path; for async we create a record and update later
        eval_manager = EvaluationManager(db)

        if getattr(evaluation, 'run_async', False):
            # Create a pending evaluation record tied to this request
            created_record = db.create_evaluation(
                prompt_id=evaluation.prompt_id,
                project_id=project_id,
                model_configs=configs_list,
                status="running",
                test_case_ids=test_case_ids,
                client_id=user_context.get("client_id", "api"),
                name=evaluation.name or "Evaluation",
                description=evaluation.description or "",
            )
            eval_id = created_record["id"]
            eval_uuid = created_record["uuid"]

            # In test environments, run inline to ensure timely completion for polling tests
            import os as _os
            if _os.getenv("PYTEST_CURRENT_TEST") or is_test_mode():
                req_id = ensure_request_id(request) if request is not None else None
                tp = ensure_traceparent(request) if request is not None else ""
                credential_runtime_transferred = True
                await run_evaluation_async(
                    eval_id,
                    db,
                    user_id=user_id_int,
                    provider=provider_name,
                    model=model_name,
                    timeout_seconds=timeout_seconds,
                    credential_runtime=credential_runtime,
                    provider_credentials=provider_credentials,
                    request_id=req_id,
                    traceparent=tp,
                )
                refreshed = db.get_evaluation(eval_id) or {}
                return EvaluationResponse(
                    id=eval_id,
                    uuid=eval_uuid,
                    project_id=project_id,
                    prompt_id=evaluation.prompt_id,
                    name=evaluation.name or "Evaluation",
                    description=evaluation.description or "",
                    status=refreshed.get("status", "completed"),
                    created_at=datetime.now().isoformat(),
                    metrics=refreshed.get("aggregate_metrics"),
                )
            else:
                # Schedule via FastAPI BackgroundTasks for normal operation.
                # Propagate request_id/traceparent for log correlation.
                req_id = ensure_request_id(request) if request is not None else None
                tp = ensure_traceparent(request) if request is not None else ""
                background_tasks.add_task(
                    run_evaluation_async,
                    eval_id,
                    db,
                    user_id=user_id_int,
                    provider=provider_name,
                    model=model_name,
                    timeout_seconds=timeout_seconds,
                    credential_runtime=credential_runtime,
                    provider_credentials=provider_credentials,
                    request_id=req_id,
                    traceparent=tp,
                )
                credential_runtime_transferred = True
                return EvaluationResponse(
                    id=eval_id,
                    uuid=eval_uuid,
                    project_id=project_id,
                    prompt_id=evaluation.prompt_id,
                    name=evaluation.name or "Evaluation",
                    description=evaluation.description or "",
                    status="running",
                    created_at=datetime.now().isoformat(),
                )
        else:
            # Run synchronously and return results
            result = await eval_manager.run_evaluation_async(
                prompt_id=evaluation.prompt_id,
                test_case_ids=test_case_ids,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=provider_name,
                api_key=provider_api_key,
                app_config=app_config_override,
                credentials_resolved=True,
                provider_credentials=provider_credentials,
                timeout_seconds=timeout_seconds,
                on_provider_success=_mark_provider_success,
            )

            return EvaluationResponse(
                id=result["id"],
                uuid=result["uuid"],
                project_id=project_id,
                prompt_id=result["prompt_id"],
                name=evaluation.name or "Evaluation",
                description=evaluation.description or "",
                status=result["status"],
                created_at=datetime.now().isoformat(),
                metrics=result.get("metrics"),
            )

    except HTTPException:
        raise
    except ByokResolutionError as e:
        raise_detached_error(_prompt_studio_credential_http_exception(e))
    except DatabaseError as e:
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        get_ps_logger(
            request_id=rid,
            ps_component="endpoint",
            ps_job_kind="evaluations",
            traceparent=tp,
        ).error("Failed to create evaluation")
        raise_detached_error(
            map_db_error_to_http(
                e,
                default_detail="Failed to create evaluation",
                log_error=False,
            )
        )
    except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        get_ps_logger(
            request_id=rid,
            ps_component="endpoint",
            ps_job_kind="evaluations",
            traceparent=tp,
        ).error("Failed to create evaluation")
        raise_detached_error(
            HTTPException(status_code=500, detail="Failed to create evaluation")
        )
    finally:
        if credential_runtime is not None and not credential_runtime_transferred:
            await await_owned_worker(credential_runtime.close())

@router.get("/evaluations", response_model=EvaluationList, openapi_extra={
    "responses": {"200": {"description": "Evaluations", "content": {"application/json": {"examples": {"list": {"summary": "Eval list", "value": [{"id": 501, "project_id": 1, "prompt_id": 12, "status": "running"}]}}}}}}
})
async def list_evaluations(
    request: Request,
    project_id: int = Query(..., description="Project ID"),
    prompt_id: Optional[int] = Query(None, description="Filter by prompt ID"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user),
) -> EvaluationList:
    """
    List evaluations for a project.

    Args:
        project_id: Project ID
        prompt_id: Optional prompt ID filter
        limit: Maximum results
        offset: Pagination offset
        db: Database instance
        user_context: Current user context

    Returns:
        List of evaluations
    """
    try:
        # Use EvaluationManager to list evaluations
        eval_manager = EvaluationManager(db)

        # Convert the public offset contract onto the manager's page API without
        # returning rows before the requested offset.
        page = (offset // limit) + 1 if limit > 0 else 1
        start_in_page = offset % limit if limit > 0 else 0

        result = eval_manager.list_evaluations(
            project_id=project_id,
            prompt_id=prompt_id,
            page=page,
            per_page=limit
        )
        evaluation_rows = list(result.get("evaluations", []))
        if start_in_page and len(evaluation_rows) < start_in_page + limit:
            next_result = eval_manager.list_evaluations(
                project_id=project_id,
                prompt_id=prompt_id,
                page=page + 1,
                per_page=limit,
            )
            evaluation_rows.extend(next_result.get("evaluations", []))
        evaluation_rows = evaluation_rows[start_in_page:start_in_page + limit]

        # Convert to response format
        evaluations = []
        for eval_data in evaluation_rows:
            evaluations.append(EvaluationResponse(
                id=eval_data["id"],
                uuid=eval_data.get("uuid", ""),
                project_id=project_id,  # Add project_id since it might not be in the result
                prompt_id=eval_data["prompt_id"],
                name=eval_data.get("prompt_name", "Evaluation"),
                description="",  # Not returned by manager
                status=eval_data.get("status", "pending"),
                created_at=eval_data.get("created_at", ""),
                completed_at=eval_data.get("completed_at"),
                metrics=eval_data.get("aggregate_metrics") or {},
            ))

        pagination_data = result.get("pagination") if isinstance(result.get("pagination"), dict) else {}
        total = int(pagination_data.get("total", result.get("total", len(evaluations))))
        return {
            "evaluations": evaluations,
            "total": total,
            "limit": limit,
            "offset": offset,
            "pagination": build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(evaluations),
            ),
        }

    except DatabaseError as e:
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        get_ps_logger(
            request_id=rid,
            ps_component="endpoint",
            ps_job_kind="evaluations",
            traceparent=tp,
        ).error("Failed to list evaluations")
        raise_detached_error(
            map_db_error_to_http(
                e,
                default_detail="Failed to list evaluations",
                log_error=False,
            )
        )
    except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        get_ps_logger(
            request_id=rid,
            ps_component="endpoint",
            ps_job_kind="evaluations",
            traceparent=tp,
        ).error("Failed to list evaluations")
        raise_detached_error(
            HTTPException(status_code=500, detail="Failed to list evaluations")
        )

@router.get(
    "/evaluations/{evaluation_id}",
    response_model=EvaluationResponse,
    openapi_extra={
        "responses": {
            "200": {
                "description": "Evaluation",
                "content": {
                    "application/json": {
                        "examples": {
                            "get": {
                                "summary": "Evaluation details",
                                "value": {
                                    "id": 501,
                                    "project_id": 1,
                                    "prompt_id": 12,
                                    "status": "completed"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def get_evaluation(
    evaluation_id: int,
    request: Request,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user),
) -> EvaluationResponse:
    """
    Get a specific evaluation.

    Args:
        evaluation_id: Evaluation ID
        db: Database instance
        user_context: Current user context

    Returns:
        Evaluation details
    """
    try:
        eval_dict = db.get_evaluation(evaluation_id)
        if not eval_dict:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        # Normalize metrics
        agg = eval_dict.get("aggregate_metrics")
        if isinstance(agg, str) and agg:
            try:
                metrics_obj = json.loads(agg)
            except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
                metrics_obj = {}
        elif isinstance(agg, dict):
            metrics_obj = agg
        else:
            metrics_obj = {}

        # Normalize timestamps to strings
        def _ts(val):
            try:
                import datetime as _dt
                if isinstance(val, (_dt.datetime, _dt.date)):
                    return val.isoformat()
            except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
                pass
            return val

        # Derive status fallback: if pending but started_at set, treat as running
        status_val = eval_dict.get("status", "pending")
        if status_val == "pending" and eval_dict.get("started_at"):
            status_val = "running"

        return {
            "id": int(eval_dict["id"]),
            "uuid": str(eval_dict.get("uuid", "")),
            "project_id": int(eval_dict.get("project_id", 0)),
            "prompt_id": int(eval_dict.get("prompt_id", 0)) if eval_dict.get("prompt_id") is not None else 0,
            "name": eval_dict.get("name", ""),
            "description": eval_dict.get("description", ""),
            "status": status_val,
            "created_at": _ts(eval_dict.get("created_at", "")),
            "completed_at": _ts(eval_dict.get("completed_at")),
            "metrics": metrics_obj,
            "config": {},
            "tags": []
        }

    except HTTPException:
        raise
    except DatabaseError as e:
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        get_ps_logger(
            request_id=rid,
            ps_component="endpoint",
            ps_job_kind="evaluations",
            traceparent=tp,
        ).error("Failed to get evaluation")
        raise_detached_error(
            map_db_error_to_http(
                e,
                default_detail="Failed to get evaluation",
                log_error=False,
            )
        )
    except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        get_ps_logger(
            request_id=rid,
            ps_component="endpoint",
            ps_job_kind="evaluations",
            traceparent=tp,
        ).error("Failed to get evaluation")
        raise_detached_error(
            HTTPException(status_code=500, detail="Failed to get evaluation")
        )

@router.delete("/evaluations/{evaluation_id}", openapi_extra={
    "responses": {"200": {"description": "Deleted", "content": {"application/json": {"examples": {"deleted": {"value": {"message": "Evaluation 123 deleted successfully"}}}}}}}
})
async def delete_evaluation(
    evaluation_id: int,
    request: Request,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user),
) -> dict[str, str]:
    """
    Delete an evaluation (soft delete).

    Args:
        evaluation_id: Evaluation ID
        db: Database instance
        user_context: Current user context

    Returns:
        Success message
    """
    try:
        if not db.delete_evaluation(evaluation_id):
            raise HTTPException(status_code=404, detail="Evaluation not found")

        return {"message": f"Evaluation {evaluation_id} deleted successfully"}

    except HTTPException:
        raise
    except DatabaseError as e:
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        get_ps_logger(
            request_id=rid,
            ps_component="endpoint",
            ps_job_kind="evaluations",
            traceparent=tp,
        ).error("Failed to delete evaluation")
        raise_detached_error(
            map_db_error_to_http(
                e,
                default_detail="Failed to delete evaluation",
                log_error=False,
            )
        )
    except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        get_ps_logger(
            request_id=rid,
            ps_component="endpoint",
            ps_job_kind="evaluations",
            traceparent=tp,
        ).error("Failed to delete evaluation")
        raise_detached_error(
            HTTPException(status_code=500, detail="Failed to delete evaluation")
        )

########################################################################################################################
# Background Task Health

# Minimal in-memory ping registry for background tasks health checks
_BG_PINGS: dict[str, dict[str, Any]] = {}


async def _complete_ping(ping_id: str):
    try:
        # Yield to event loop briefly to simulate background work
        await asyncio.sleep(0.01)
        _BG_PINGS[ping_id]["status"] = "completed"
        _BG_PINGS[ping_id]["completed_at"] = datetime.now().isoformat()
    except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
        _BG_PINGS[ping_id]["status"] = "failed"


@router.post("/background/ping", openapi_extra={
    "responses": {"200": {"description": "Ping scheduled", "content": {"application/json": {"examples": {"scheduled": {"value": {"id": "abc123", "status": "processing", "created_at": "2024-09-21T12:00:00"}}}}}}}
})
async def background_ping(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Schedule a trivial background task to verify background execution works."""
    pid = str(uuid.uuid4())
    _BG_PINGS[pid] = {"id": pid, "status": "processing", "created_at": datetime.now().isoformat()}
    background_tasks.add_task(_complete_ping, pid)
    return _BG_PINGS[pid]


@router.get("/background/pings/{ping_id}", openapi_extra={
    "responses": {"200": {"description": "Ping status", "content": {"application/json": {"examples": {"done": {"value": {"id": "abc123", "status": "completed", "completed_at": "2024-09-21T12:00:01"}}}}}}, "404": {"description": "Not found"}}
})
async def get_ping_status(ping_id: str) -> dict[str, Any]:
    if ping_id not in _BG_PINGS:
        raise HTTPException(status_code=404, detail="Ping not found")
    return _BG_PINGS[ping_id]

async def run_evaluation_async(
    evaluation_id: int,
    db: PromptStudioDatabase,
    *,
    user_id: Optional[int] = None,
    provider: str = "openai",
    model: str | None = None,
    timeout_seconds: float | None = None,
    credential_runtime: ProviderCredentialRuntime | None = None,
    provider_credentials: ProviderCallCredentials | None = None,
    request_id: str | None = None,
    traceparent: str = "",
) -> None:
    """
    Execute an evaluation and update the existing record.

    Best-effort: computes simple metrics; tolerates missing LLM credentials by
    marking failures while still completing the record.
    """
    import json as _json

    record_loaded = False
    try:
        with log_context(
            request_id=request_id,
            traceparent=traceparent,
            ps_component="evaluation_bg",
        ) as _log:
            _log.info(
                "PS evaluation.async.start evaluation_id={}",
                evaluation_id,
            )

            record = db.get_evaluation(evaluation_id)
            if not record:
                raise RuntimeError("Evaluation not found")
            record_loaded = True
            db.update_evaluation(evaluation_id, {"status": "running", "started_at": datetime.utcnow()})

            prompt_id = record.get("prompt_id")
            # The repository decodes JSON columns; tolerate a raw string from older rows.
            test_case_ids = record.get("test_case_ids") or []
            cfg_raw = record.get("model_configs") or {}
            for name, value in (("test_case_ids", test_case_ids), ("model_configs", cfg_raw)):
                if isinstance(value, str):
                    try:
                        decoded = _json.loads(value)
                    except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
                        decoded = [] if name == "test_case_ids" else {}
                    if name == "test_case_ids":
                        test_case_ids = decoded
                    else:
                        cfg_raw = decoded

            if isinstance(cfg_raw, list) and cfg_raw:
                cfg = cfg_raw[0]
            elif isinstance(cfg_raw, dict):
                cfg = cfg_raw
            else:
                cfg = {}

            try:
                temperature = float(cfg.get("temperature", 0.7))
            except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
                temperature = 0.7
            try:
                max_tokens = int(cfg.get("max_tokens", 1000))
            except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
                max_tokens = 1000
            provider_name = (
                provider
                if provider_credentials is not None
                else (cfg.get("provider") or cfg.get("api_name") or provider or "openai")
            )
            provider_name = canonical_provider_name(str(provider_name).strip() or "openai")
            configured_model = model or cfg.get("model_name") or cfg.get("model")
            if configured_model:
                model_name = configured_model
            elif provider_name == "openai":
                model_name = "gpt-3.5-turbo"
            else:
                raise RuntimeError("Model is required for the selected provider.")
            effective_timeout_seconds = timeout_seconds
            if effective_timeout_seconds is None and provider_credentials is None:
                effective_timeout_seconds = cfg.get("timeout_seconds")
            provider_key = provider_name

            if credential_runtime is None:
                credential_runtime = ProviderCredentialRuntime(
                    user_id=user_id,
                    team_ids=None,
                    org_ids=None,
                    trusted_base_url_override=False,
                    override_snapshot_resolver=capture_provider_override_call_snapshot,
                )
            if provider_credentials is None:
                provider_credentials = await credential_runtime.resolve(
                    provider_key,
                    model=model_name,
                )
            provider_api_key = provider_credentials.api_key
            app_config_override = provider_credentials.app_config

            async def _mark_provider_success() -> None:
                await mark_provider_credential_used(
                    credential_runtime,
                    provider_credentials,
                )

            if (
                provider_requires_api_key(provider_key)
                and not _prompt_studio_provider_auth_is_resolved(
                    provider_key,
                    provider_credentials,
                )
                and not _is_prompt_studio_test_mode()
            ):
                raise RuntimeError(f"Provider '{provider_name}' requires an API key.")

            eval_manager = EvaluationManager(db)
            result = await eval_manager.run_evaluation_with_existing_record(
                evaluation_id=evaluation_id,
                prompt_id=int(prompt_id),
                test_case_ids=[int(t) for t in (test_case_ids or [])],
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=provider_name,
                api_key=provider_api_key,
                app_config=app_config_override,
                credentials_resolved=provider_credentials.credentials_resolved,
                provider_credentials=provider_credentials,
                timeout_seconds=effective_timeout_seconds,
                on_provider_success=_mark_provider_success,
            )

            _log.info(
                "PS evaluation.async.done evaluation_id={} total_tests={} pass_rate={}",
                evaluation_id,
                (result.get("metrics") or {}).get("total_tests", 0),
                round(float((result.get("metrics") or {}).get("pass_rate", 0.0)), 3),
            )

    except asyncio.CancelledError:
        cancellation_log = get_ps_logger(
            request_id=request_id,
            ps_component="evaluation_bg",
            ps_job_kind="evaluations",
            traceparent=traceparent,
        )
        cancellation_log.info(
            "PS evaluation.async.cancelled evaluation_id={}",
            evaluation_id,
        )
        if record_loaded:
            try:
                db.cancel_evaluation_if_active(evaluation_id, "Evaluation cancelled")
            except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
                cancellation_log.warning(
                    "PS evaluation.async.cancel_persist_failed evaluation_id={}",
                    evaluation_id,
                )
        raise
    except (ByokResolutionError, *_PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS) as e:
        safe_error = sanitized_provider_stream_exception(e)
        get_ps_logger(
            request_id=request_id,
            ps_component="evaluation_bg",
            ps_job_kind="evaluations",
            traceparent=traceparent,
        ).error("Failed to run async evaluation error_code={}", safe_error.code)
        if record_loaded:
            try:
                db.update_evaluation(evaluation_id, {"status": "failed", "error_message": str(safe_error)})
            except _PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS:
                pass
    finally:
        if credential_runtime is not None:
            await await_owned_worker(credential_runtime.close())
