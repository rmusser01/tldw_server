# evaluations_unified.py - Unified evaluation API endpoints
"""
Unified evaluation API combining OpenAI-compatible and tldw-specific endpoints.

This module provides a single, cohesive API for all evaluation functionality.
"""

import asyncio
import json
import os
import time
from typing import Annotated, Any, Optional, Union

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.routing import APIRoute
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import rbac_rate_limit, require_roles

# Import unified schemas
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationHistoryRequest,
    EvaluationHistoryResponse,
    EvaluationMetric,
    # tldw-specific schemas
    GEvalRequest,
    GEvalResponse,
    # Common schemas
    HealthCheckResponse,
    PropositionEvaluationRequest,
    PropositionEvaluationResponse,
    RAGEvaluationRequest,
    RAGEvaluationResponse,
    RateLimitStatusResponse,
    ResponseQualityRequest,
    ResponseQualityResponse,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ResolvedByokCredentials,
    record_byok_missing_credentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.permissions import EVALS_READ
from tldw_Server_API.app.core.Chat.chat_service import resolve_provider_api_key
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations.audit_adapter import (
    log_evaluation_deleted,
    log_evaluation_exported,
)

# Import unified service
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    get_unified_evaluation_service_for_user,
)

# Import additional services
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import get_user_rate_limiter_for_user
from tldw_Server_API.app.core.Evaluations.webhook_identity import webhook_user_id_from_user
from tldw_Server_API.app.core.Evaluations.webhook_manager import WebhookEvent, WebhookManager
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.testing import (
    is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
    is_test_mode as _is_test_mode,
)

# Non-critical exceptions for defensive guards and best-effort flows
_EVALS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
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

# Create router
router = APIRouter(prefix="/evaluations", tags=["evaluations"])

_webhook_managers: dict = {}
_wm_lock = None

import contextlib

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

from .evaluations_auth import (
    _apply_rate_limit_headers,
    check_evaluation_rate_limit,
    enforce_heavy_evaluations_admin,
    get_evaluation_identity,
    get_eval_request_user,
    require_eval_permissions,
    sanitize_error_message,
    verify_api_key,
)
from .evaluations_recipes import recipes_router
from .evaluations_synthetic import synthetic_router


def _get_webhook_manager_for_user(user_id: int) -> WebhookManager:
    global _wm_lock
    if _wm_lock is None:
        import threading as _threading
        _wm_lock = _threading.Lock()
    with _wm_lock:
        mgr = _webhook_managers.get(user_id)
        import os as _os
        _override_db = _os.getenv("EVALUATIONS_TEST_DB_PATH")
        if mgr is not None and _override_db:
            try:
                cfg = getattr(getattr(mgr, "db_adapter", None), "config", None)
                current_conn = getattr(cfg, "connection_string", None)
                if current_conn and current_conn != _override_db:
                    mgr = WebhookManager(db_path=_override_db)
                    _webhook_managers[user_id] = mgr
            except _EVALS_NONCRITICAL_EXCEPTIONS:
                pass
        if mgr is None:
            db_path = _override_db or str(DatabasePaths.get_evaluations_db_path(user_id))
            mgr = WebhookManager(db_path=db_path)
            _webhook_managers[user_id] = mgr
    return mgr

def get_db_for_user(user_id: int):
    svc = get_unified_evaluation_service_for_user(user_id)
    return getattr(svc, 'db', None)


def _is_eval_test_mode() -> bool:
    return bool(_is_test_mode() or _is_explicit_pytest_runtime())


def _normalize_eval_user_id(current_user: User) -> Optional[int]:
    user_id = getattr(current_user, "id", None)
    if user_id is None:
        return None
    try:
        return int(user_id)
    except (TypeError, ValueError):
        return None


async def _get_admin_principal_if_needed(
    request: Request,
) -> Optional[AuthPrincipal]:
    """Resolve AuthPrincipal only when heavy-eval admin gating is enabled."""
    if os.getenv("EVALS_HEAVY_ADMIN_ONLY", "true").lower() in {"true", "1", "yes", "on"}:
        dep = get_auth_principal
        try:
            overrides = getattr(request.app, "dependency_overrides", {}) or {}
            dep = overrides.get(get_auth_principal, get_auth_principal)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            dep = get_auth_principal
        result = dep(request)
        try:
            import inspect as _inspect

            if _inspect.isawaitable(result):
                result = await result
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            result = await get_auth_principal(request)
        return result
    return None


async def _resolve_eval_credentials(
    provider: str,
    *,
    current_user: User,
    request: Optional[Request],
) -> ResolvedByokCredentials:
    def _fallback_resolver(name: str) -> Optional[str]:
        key_val, _ = resolve_provider_api_key(
            name,
            prefer_module_keys_in_tests=True,
        )
        return key_val

    return await resolve_byok_credentials(
        provider,
        user_id=_normalize_eval_user_id(current_user),
        request=request,
        fallback_resolver=_fallback_resolver,
    )


async def _validate_provider_credentials(
    eval_type: str,
    provider_key: str,
    provider_name: str,
    provider_api_key: Optional[str],
) -> None:
    """Validate required provider credentials for evaluation endpoints."""
    if (
        eval_type in {"geval", "rag", "response_quality"}
        and provider_requires_api_key(provider_key)
        and not provider_api_key
        and not _is_eval_test_mode()
    ):
        record_byok_missing_credentials(provider_key, operation="evaluations")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "missing_provider_credentials",
                "message": f"Provider '{provider_name}' requires an API key.",
            },
        )


async def _resolve_and_validate_eval_provider(
    request: Union[GEvalRequest, RAGEvaluationRequest, ResponseQualityRequest],
    eval_type: str,
    *,
    current_user: User,
    http_request: Request,
) -> tuple[str, Optional[str], Optional[str], Optional[ResolvedByokCredentials]]:
    """Resolve provider credentials and validate BYOK requirements for evaluation requests."""
    provider_name = (request.api_name or "openai").strip() or "openai"
    provider_key = provider_name.lower()
    raw_api_key = getattr(request, "api_key", None)
    if raw_api_key:
        logger.debug("Ignoring per-request api_key override for provider={}", provider_name)
    explicit_key = None
    provider_api_key = None
    byok_resolution: Optional[ResolvedByokCredentials] = None

    if not provider_api_key:
        byok_resolution = await _resolve_eval_credentials(
            provider_key,
            current_user=current_user,
            request=http_request,
        )
        provider_api_key = byok_resolution.api_key

    await _validate_provider_credentials(
        eval_type,
        provider_key,
        provider_name,
        provider_api_key,
    )

    return provider_name, provider_api_key, explicit_key, byok_resolution


# verify_api_key et al. imported from evaluations_auth


@router.post(
    "/admin/idempotency/cleanup",
)
async def admin_cleanup_idempotency(
    principal: Annotated[Optional[AuthPrincipal], Depends(_get_admin_principal_if_needed)],
    _current_user: Annotated[User, Depends(get_eval_request_user)],  # dependency for side effects
    ttl_hours: int = Query(72, ge=1, le=720, description="Delete idempotency keys older than this TTL (hours)"),
    target_user_id: Optional[int] = Query(None, description="If provided, only clean this user's evaluations DB"),
    _user_ctx: str = Depends(verify_api_key),  # dependency for side effects
):
    """Admin-only: purge stale idempotency keys in Evaluations DBs on-demand.

    Returns a summary of deleted rows per user and total.
    """
    # Admin gate (claim-first + legacy shim)
    enforce_heavy_evaluations_admin(principal)
    try:
        from pathlib import Path as _Path

        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DP
        from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase as _EDB

        deleted_total = 0
        details = []

        # Build candidate user ids
        candidate_ids = set()
        if target_user_id is not None:
            candidate_ids.add(int(target_user_id))
        else:
            with contextlib.suppress(_EVALS_NONCRITICAL_EXCEPTIONS):
                candidate_ids.add(int(_DP.get_single_user_id()))
            try:
                base = _Path(_DP.get_user_base_directory(_DP.get_single_user_id())).parent
                if base.exists():
                    for entry in base.iterdir():
                        if entry.is_dir():
                            try:
                                candidate_ids.add(int(entry.name))
                            except _EVALS_NONCRITICAL_EXCEPTIONS:
                                continue
            except _EVALS_NONCRITICAL_EXCEPTIONS:
                pass

        for uid in sorted(candidate_ids):
            try:
                db_path = _DP.get_evaluations_db_path(uid)
                if not db_path.exists():
                    continue
                db = _EDB(str(db_path))
                deleted = db.cleanup_idempotency_keys(ttl_hours=int(ttl_hours))
                deleted_total += int(deleted)
                details.append({"user_id": uid, "deleted": int(deleted)})
            except _EVALS_NONCRITICAL_EXCEPTIONS:
                continue

        return {"deleted_total": deleted_total, "details": details}
    except HTTPException:
        raise
    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Admin idempotency cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Idempotency cleanup failed") from e


def _estimate_tokens_from_texts(*texts: Optional[str], provider: Optional[str] = None, model: Optional[str] = None) -> int:
    """Estimate tokens with provider/model hints when available.

    Preference order when tiktoken is available:
    1) encoding_for_model(model) if model provided and recognized
    2) Heuristic by model name -> o200k_base for 4o/4.1/o1 families
    3) cl100k_base as a general default

    Falls back to chars/4 if tiktoken or encoding lookup is unavailable.
    """
    try:
        import tiktoken  # type: ignore
        enc = None

        # Try exact model mapping first (OpenAI models)
        if isinstance(model, str) and model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except _EVALS_NONCRITICAL_EXCEPTIONS:
                enc = None

        # Heuristic for modern OpenAI models when model hint exists
        if enc is None and isinstance(model, str):
            m = model.lower()
            try:
                if ("gpt-4o" in m) or ("gpt-4.1" in m) or m.startswith("o1"):
                    enc = tiktoken.get_encoding("o200k_base")
            except _EVALS_NONCRITICAL_EXCEPTIONS:
                enc = None

        # Provider fallback (OpenAI -> cl100k_base)
        if enc is None and isinstance(provider, str) and provider.lower() == "openai":
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except _EVALS_NONCRITICAL_EXCEPTIONS:
                enc = None

        # Final default
        if enc is None:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except _EVALS_NONCRITICAL_EXCEPTIONS:
                enc = None

        if enc is not None:
            total = 0
            for t in texts:
                if isinstance(t, str) and t:
                    total += len(enc.encode(t))
            return total
    except _EVALS_NONCRITICAL_EXCEPTIONS:
        pass

    # Fallback: character-based approximation
    total_chars = 0
    for t in texts:
        if isinstance(t, str):
            total_chars += len(t)
    return max(0, total_chars // 4)


# ============= Rate Limiting =============

# check_evaluation_rate_limit imported


# ============= Error Handling =============

"""Auth and error helpers imported from evaluations_auth."""


# create_error_response imported


"""Admin checker imported"""


from .evaluations_embeddings_abtest import abtest_router

router.include_router(abtest_router)

from tldw_Server_API.app.core.Evaluations.run_state import (
    TERMINAL_RUN_STATUSES,
    normalize_run_status,
)


@router.get("/embeddings/abtest/{test_id}/events")
async def stream_embeddings_abtest_events(
    test_id: str,
    user_ctx: str = Depends(verify_api_key),
    _: None = Depends(check_evaluation_rate_limit),
    current_user: User = Depends(get_eval_request_user),
):
    """SSE stream of progress and updates for an A/B test, using SSEStream for heartbeats and metrics."""
    import asyncio as _aio
    import json as _json

    from fastapi.responses import StreamingResponse

    from tldw_Server_API.app.core.Streaming.streams import SSEStream

    identity = get_evaluation_identity(current_user)
    svc = get_unified_evaluation_service_for_user(identity.user_scope)

    stream = SSEStream(
        # Use env-driven heartbeat defaults; standard labels for dashboards
        heartbeat_interval_s=None,
        heartbeat_mode=None,
        labels={"component": "evaluations", "endpoint": "embeddings_abtest_events"},
    )

    async def _produce() -> None:
        last_payload = None
        while True:
            row = svc.db.get_abtest(test_id, created_by=identity.created_by)
            if not row:
                await stream.error("not_found", "A/B test not found")
                return
            status = normalize_run_status(row.get("status"))
            stats = row.get("stats_json")
            payload = {"type": "status", "status": status}
            try:
                payload["stats"] = _json.loads(stats) if stats else {}
            except _EVALS_NONCRITICAL_EXCEPTIONS:
                payload["stats"] = {}

            if payload != last_payload:
                await stream.send_json(payload)
                last_payload = payload

            if status in TERMINAL_RUN_STATUSES:
                await stream.done()
                return
            await _aio.sleep(1.0)

    async def _gen():
        producer = _aio.create_task(_produce())
        try:
            async for line in stream.iter_sse():
                yield line
        except _aio.CancelledError:
            # On client cancellation, stop the producer promptly
            if not producer.done():
                with contextlib.suppress(_EVALS_NONCRITICAL_EXCEPTIONS):
                    producer.cancel()
                with contextlib.suppress(_EVALS_NONCRITICAL_EXCEPTIONS):
                    await _aio.gather(producer, return_exceptions=True)
            raise
        else:
            # Normal shutdown: ensure producer completes without forced cancel
            if not producer.done():
                with contextlib.suppress(_EVALS_NONCRITICAL_EXCEPTIONS):
                    await _aio.gather(producer, return_exceptions=True)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_gen(), media_type="text/event-stream", headers=headers)


@router.delete("/embeddings/abtest/{test_id}")
async def delete_embeddings_abtest(
    test_id: str,
    user_ctx: str = Depends(verify_api_key),
    _: None = Depends(check_evaluation_rate_limit),
    current_user: User = Depends(get_eval_request_user),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """Cancel/cleanup an embeddings A/B test."""
    identity = get_evaluation_identity(current_user)
    svc = get_unified_evaluation_service_for_user(identity.user_scope)
    if not svc.db.get_abtest(test_id, created_by=identity.created_by):
        raise HTTPException(status_code=404, detail="A/B test not found")
    # Idempotency: if prior mapping exists, return canonical response without side effects
    try:
        if idempotency_key:
            prior = svc.db.lookup_idempotency("emb_abtest_delete", idempotency_key, identity.created_by)
            if prior:
                logger.info(f"A/B test delete idempotent hit: {test_id}")
                return Response(content=json.dumps({"status": "deleted", "test_id": test_id}), media_type='application/json', headers={"X-Idempotent-Replay": "true", "Idempotency-Key": idempotency_key})
    except _EVALS_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        from tldw_Server_API.app.core.Evaluations.embeddings_abtest_service import cleanup_abtest_resources
        cleanup_abtest_resources(
            svc.db,
            identity.user_scope,
            test_id,
            delete_db=True,
            delete_idempotency=True,
            created_by=identity.created_by,
        )
        logger.info(f"A/B test deleted: {test_id} by {identity.created_by}")
    except _EVALS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"A/B test cleanup failed for {test_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to delete A/B test") from exc
    log_evaluation_deleted(user_id=identity.user_scope, eval_id=test_id)
    try:
        if idempotency_key:
            svc.db.record_idempotency("emb_abtest_delete", idempotency_key, test_id, identity.created_by)
    except _EVALS_NONCRITICAL_EXCEPTIONS:
        pass
    return {"status": "deleted", "test_id": test_id}


@router.get(
    "/embeddings/abtest/{test_id}/export",
    dependencies=[
        Depends(require_roles("admin")),
        Depends(rbac_rate_limit("evals.abtest.export")),
    ],
)
async def export_embeddings_abtest(
    test_id: str,
    principal: Annotated[AuthPrincipal, Depends(get_auth_principal)],
    current_user: Annotated[User, Depends(get_eval_request_user)],
    format: str = Query("json", pattern="^(json|csv)$"),
    user_ctx: str = Depends(verify_api_key),
    _: None = Depends(check_evaluation_rate_limit),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """Export AB test results (JSON or CSV). Admin-only."""
    enforce_heavy_evaluations_admin(principal)
    identity = get_evaluation_identity(current_user)
    svc = get_unified_evaluation_service_for_user(identity.user_scope)
    if not svc.db.get_abtest(test_id, created_by=identity.created_by):
        raise HTTPException(status_code=404, detail="A/B test not found")
    rows, total = svc.db.list_abtest_results(test_id, limit=100000, offset=0, created_by=identity.created_by)
    def _parse_json(value, default):
        if value is None:
            return default
        if isinstance(value, (list, dict)):
            return value
        try:
            parsed = json.loads(value)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            return default
        return parsed if isinstance(parsed, (list, dict)) else default

    def _parse_float_list(value):
        parsed = _parse_json(value, None)
        if not isinstance(parsed, list):
            return None
        out = []
        for item in parsed:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out

    if format == 'json':
        normalized_rows = []
        for r in rows:
            normalized_rows.append(
                {
                    **r,
                    "ranked_ids": _parse_json(r.get("ranked_ids"), []),
                    "scores": _parse_json(r.get("scores"), None),
                    "metrics_json": _parse_json(r.get("metrics_json"), {}),
                    "ranked_distances": _parse_float_list(r.get("ranked_distances")),
                    "ranked_metadatas": _parse_json(r.get("ranked_metadatas"), None),
                    "ranked_documents": _parse_json(r.get("ranked_documents"), None),
                    "rerank_scores": _parse_float_list(r.get("rerank_scores")),
                }
            )
        # Idempotency: record export mapping (best-effort)
        try:
            if idempotency_key:
                svc.db.record_idempotency("emb_abtest_export_json", idempotency_key, f"{test_id}:json", identity.created_by)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        log_evaluation_exported(
            user_id=identity.user_scope,
            eval_id=test_id,
            eval_type="embeddings_abtest",
            export_format="json",
            total=total,
        )
        headers = {}
        if idempotency_key:
            headers = {"Idempotency-Key": idempotency_key}
        return Response(content=json.dumps({"test_id": test_id, "total": total, "results": normalized_rows}), media_type='application/json', headers=headers)
    # CSV
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["result_id", "arm_id", "query_id", "ranked_ids", "latency_ms", "metrics_json"])
    for r in rows:
        writer.writerow([r.get('result_id'), r.get('arm_id'), r.get('query_id'), r.get('ranked_ids'), r.get('latency_ms'), r.get('metrics_json')])
    try:
        if idempotency_key:
            svc.db.record_idempotency("emb_abtest_export_csv", idempotency_key, f"{test_id}:csv", identity.created_by)
    except _EVALS_NONCRITICAL_EXCEPTIONS:
        pass
    log_evaluation_exported(
        user_id=identity.user_scope,
        eval_id=test_id,
        eval_type="embeddings_abtest",
        export_format="csv",
        total=total,
    )
    headers = {"Content-Disposition": f"attachment; filename=abtest_{test_id}.csv"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return Response(content=output.getvalue(), media_type='text/csv', headers=headers)


# ============= OpenAI-Compatible Evaluation Endpoints =============

# Moved: CRUD create/list to evaluations_crud.py


# ============= Rate Limit Management =============

@router.get("/rate-limits", response_model=RateLimitStatusResponse)
async def get_rate_limit_status(
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    """Get current rate limit status for the authenticated user"""
    try:
        limiter = get_user_rate_limiter_for_user(current_user.id)
        summary = await limiter.get_usage_summary(user_id)

        # Convert the nested structure to flat structure expected by RateLimitStatusResponse
        from datetime import datetime, timedelta, timezone
        return RateLimitStatusResponse(
            tier=summary.get("tier", "free"),
            limits={
                "evaluations_per_minute": summary.get("limits", {}).get("per_minute", {}).get("evaluations", 0),
                "evaluations_per_day": summary.get("limits", {}).get("daily", {}).get("evaluations", 0),
                "tokens_per_day": summary.get("limits", {}).get("daily", {}).get("tokens", 0),
                "cost_per_day": int(summary.get("limits", {}).get("daily", {}).get("cost", 0)),
                "cost_per_month": int(summary.get("limits", {}).get("monthly", {}).get("cost", 0))
            },
            usage={
                "evaluations_today": summary.get("usage", {}).get("today", {}).get("evaluations", 0),
                "tokens_today": summary.get("usage", {}).get("today", {}).get("tokens", 0),
                "cost_today": int(summary.get("usage", {}).get("today", {}).get("cost", 0)),
                "cost_month": int(summary.get("usage", {}).get("month", {}).get("cost", 0))
            },
            remaining={
                "daily_evaluations": summary.get("remaining", {}).get("daily_evaluations", 0),
                "daily_tokens": summary.get("remaining", {}).get("daily_tokens", 0),
                "daily_cost": int(summary.get("remaining", {}).get("daily_cost", 0)),
                "monthly_cost": int(summary.get("remaining", {}).get("monthly_cost", 0))
            },
            reset_at=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        )

    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get rate limit status: {sanitize_error_message(e, 'rate limit check')}"
        ) from e


from .evaluations_crud import crud_router
from .evaluations_benchmarks import benchmarks_router
from .evaluations_datasets import datasets_router
from .evaluations_rag_pipeline import pipeline_router
from .evaluations_webhooks import webhooks_router

router.include_router(benchmarks_router)
router.include_router(pipeline_router)
router.include_router(datasets_router)
router.include_router(webhooks_router)
router.include_router(synthetic_router)
router.include_router(recipes_router)
router.include_router(crud_router)
# ============= Health & Metrics Endpoints =============

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check evaluation service health"""
    logger.warning("Evaluations health endpoint invoked")
    try:
        # Default to single-user instance for health when no auth context
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DP
        uid = _DP.get_single_user_id()
        svc = get_unified_evaluation_service_for_user(uid)
        health = await svc.health_check()
        return HealthCheckResponse(**health)
    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version="1.0.0",
            uptime=0,
            database="disconnected"
        )


@router.get("/metrics", dependencies=[Depends(require_eval_permissions(EVALS_READ))])
async def get_metrics(
    request: Request,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    """Get Prometheus metrics"""
    try:
        _ = user_id
        svc = get_unified_evaluation_service_for_user(current_user.id)
        metrics_summary = await svc.get_metrics_summary()

        # Handle failure from service so error message is never exposed
        if "error" in metrics_summary:
            logger.error(f"Metrics endpoint service error: {metrics_summary['error']}")
            # Return a generic error message in both text/plain and JSON responses
            if "text/plain" in request.headers.get("accept", ""):
                # Prometheus format error response
                output = "# HELP evaluation_metrics_failed Metric collection failure\n"
                output += "# TYPE evaluation_metrics_failed counter\n"
                output += "evaluation_metrics_failed{} 1\n"
                return Response(
                    content=output,
                    media_type="text/plain; version=0.0.4; charset=utf-8"
                )
            # Return JSON error response
            return {"error": "Metrics are currently unavailable"}

        # Format as Prometheus text format if requested
        if "text/plain" in request.headers.get("accept", ""):
            # Convert to Prometheus format (simplified)
            output = "# HELP evaluation_requests_total Total evaluation requests\n"
            output += "# TYPE evaluation_requests_total counter\n"
            output += f"evaluation_requests_total {{}} {metrics_summary.get('total_requests', 0)}\n"

            return Response(
                content=output,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )

        # Return JSON format
        return metrics_summary

    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to get metrics: {e}")
        # Do not expose internal error, return generic error
        return {"error": "Metrics are currently unavailable"}


"""Webhook endpoints moved to evaluations_webhooks module."""
# ============= tldw-Specific Evaluation Endpoints =============

@router.post("/geval", response_model=GEvalResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_geval(
    request: GEvalRequest,
    http_request: Request,
    response: Response,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    """
    Evaluate a summary using G-Eval metrics.

    G-Eval evaluates summaries on fluency, consistency, relevance, and coherence.
    """
    try:
        # Per-user usage limits
        limiter = get_user_rate_limiter_for_user(current_user.id)
        tokens_est = _estimate_tokens_from_texts(
            request.source_text,
            request.summary,
            provider=getattr(request, "api_name", None),
            model=None,
        )
        allowed, meta = await limiter.check_rate_limit(user_id, endpoint="evals:geval", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
        if not allowed:
            retry_after = meta.get("retry_after", 60)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=meta.get("error", "Rate limit exceeded"),
                headers={"Retry-After": str(retry_after)}
            )
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        webhook_user_id = webhook_user_id_from_user(current_user)

        provider_name, provider_api_key, explicit_key, byok_resolution = await _resolve_and_validate_eval_provider(
            request,
            "geval",
            current_user=current_user,
            http_request=http_request,
        )

        # Send webhook: evaluation started (await in TEST_MODE)
        import asyncio as _asyncio
        import os as _os
        import time as _time
        wm = _get_webhook_manager_for_user(current_user.id)
        start_event_id = f"geval_{int(_time.time())}_{webhook_user_id[:8]}"
        if _is_eval_test_mode():
            await wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_STARTED,
                evaluation_id=start_event_id,
                data={
                    "evaluation_type": "geval",
                    "api_name": request.api_name
                }
            )
        else:
            _asyncio.create_task(wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_STARTED,
                evaluation_id=start_event_id,
                data={
                    "evaluation_type": "geval",
                    "api_name": request.api_name
                }
            ))
        svc = get_unified_evaluation_service_for_user(current_user.id)
        result = await svc.evaluate_geval(
            source_text=request.source_text,
            summary=request.summary,
            metrics=request.metrics,
            api_name=provider_name,
            api_key=provider_api_key,
            user_id=stable_user_id,
            webhook_user_id=webhook_user_id,
        )
        if byok_resolution and byok_resolution.uses_byok and not explicit_key:
            await byok_resolution.touch_last_used()
        # If provider returned actual usage, record it
        try:
            usage = result.get("usage") if isinstance(result, dict) else None
            if usage and isinstance(usage, dict):
                await limiter.record_actual_usage(user_id, "evals:geval", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0))
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass

        # Format response - accept either numeric or structured metric dicts
        raw_metrics = result["results"].get("metrics", {})
        formatted_metrics = {}
        explanations_fallback = result["results"].get("explanations", {})

        def _normalize_geval_metric(metric_name: str, score_val: Optional[float], raw_score_val: Optional[float] = None) -> tuple[float, Optional[float]]:
            raw_candidate = raw_score_val if raw_score_val is not None else score_val
            try:
                raw_value = float(raw_candidate) if raw_candidate is not None else 0.0
            except (TypeError, ValueError):
                raw_value = 0.0
            if metric_name == "fluency":
                # Some providers report fluency on a 1-3 scale, others on 1-5.
                max_score = 3.0 if raw_value <= 3.0 else 5.0
            else:
                max_score = 5.0
            normalized = raw_value / max_score if raw_value >= 1.0 else raw_value
            if normalized > 1.0:
                normalized = 1.0
            return normalized, raw_value

        for metric_name, metric_value in raw_metrics.items():
            if isinstance(metric_value, dict):
                # Structured metric already provided
                name = metric_value.get("name", metric_name)
                raw_score_val = metric_value.get("raw_score", None)
                score_val = metric_value.get("score", raw_score_val)
                score_float, raw_score_float = _normalize_geval_metric(metric_name, score_val, raw_score_val)
                explanation = metric_value.get("explanation")
                metadata = metric_value.get("metadata", {})
                formatted_metrics[metric_name] = EvaluationMetric(
                    name=name,
                    score=score_float,
                    raw_score=raw_score_float,
                    explanation=explanation,
                    metadata=metadata,
                )
            else:
                # Backward-compat: simple numeric score; normalize if on 1-5 scale
                try:
                    score_num = float(metric_value)
                except (TypeError, ValueError):
                    score_num = 0.0
                normalized, raw_score = _normalize_geval_metric(metric_name, score_num)
                formatted_metrics[metric_name] = EvaluationMetric(
                    name=metric_name,
                    score=normalized,
                    raw_score=raw_score,
                    explanation=explanations_fallback.get(metric_name, ""),
                )

        # Send webhook: evaluation completed (await in TEST_MODE)
        if _is_eval_test_mode():
            await wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_COMPLETED,
                evaluation_id=result["evaluation_id"],
                data={
                    "evaluation_type": "geval",
                    "average_score": result["results"].get("average_score", 0.0),
                    "processing_time": result.get("evaluation_time", 0.0)
                }
            )
        else:
            _asyncio.create_task(wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_COMPLETED,
                evaluation_id=result["evaluation_id"],
                data={
                    "evaluation_type": "geval",
                    "average_score": result["results"].get("average_score", 0.0),
                    "processing_time": result.get("evaluation_time", 0.0)
                }
            ))

        # Normalize average score to 0-1 if provided on 1-5 scale
        _avg_raw = result["results"].get("average_score", 0.0)
        try:
            _avg_val = float(_avg_raw)
        except (TypeError, ValueError):
            _avg_val = 0.0
        _avg_norm = (_avg_val / 5.0) if _avg_val >= 1.0 else _avg_val

        resp_payload = GEvalResponse(
            metrics=formatted_metrics,
            average_score=_avg_norm,
            summary_assessment=result["results"].get("assessment", "Evaluation complete"),
            evaluation_time=result["evaluation_time"],
            metadata={"evaluation_id": result["evaluation_id"]}
        )
        try:
            if response is not None:
                await _apply_rate_limit_headers(limiter, user_id, response, meta)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        return resp_payload

    except HTTPException:
        raise
    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        # Log with stack trace for diagnostics
        logger.exception(f"G-Eval evaluation failed: {e}")
        # In TEST_MODE, surface a slightly more verbose message to aid debugging
        _detail = f"Evaluation failed: {sanitize_error_message(e, 'G-Eval evaluation')}"
        try:
            import os as _os
            if _is_eval_test_mode():
                _detail = _detail + f" (debug: {str(e)})"
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_detail
        ) from e


@router.post("/rag", response_model=RAGEvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_rag(
    request: RAGEvaluationRequest,
    http_request: Request,
    response: Response,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    """
    Evaluate RAG system performance.

    Evaluates relevance, faithfulness, answer similarity, and context precision.
    """
    try:
        # Per-user usage limits
        limiter = get_user_rate_limiter_for_user(current_user.id)
        tokens_est = _estimate_tokens_from_texts(
            request.query,
            "\n".join(request.retrieved_contexts or []),
            request.generated_response,
            request.ground_truth,
            provider=getattr(request, "api_name", None),
            model=None,
        )
        allowed, meta = await limiter.check_rate_limit(user_id, endpoint="evals:rag", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
        if not allowed:
            retry_after = meta.get("retry_after", 60)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=meta.get("error", "Rate limit exceeded"),
                headers={"Retry-After": str(retry_after)}
            )
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        webhook_user_id = webhook_user_id_from_user(current_user)

        provider_name, provider_api_key, explicit_key, byok_resolution = await _resolve_and_validate_eval_provider(
            request,
            "rag",
            current_user=current_user,
            http_request=http_request,
        )

        # Send webhook: evaluation started (await in TEST_MODE)
        import asyncio as _asyncio
        import os as _os
        wm = _get_webhook_manager_for_user(current_user.id)
        import time as _time
        start_event_id = f"rag_{int(_time.time())}_{webhook_user_id[:8]}"
        if _is_eval_test_mode():
            await wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_STARTED,
                evaluation_id=start_event_id,
                data={
                    "evaluation_type": "rag",
                    "api_name": request.api_name
                }
            )
        else:
            _asyncio.create_task(wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_STARTED,
                evaluation_id=start_event_id,
                data={
                    "evaluation_type": "rag",
                    "api_name": request.api_name
                }
            ))
        svc = get_unified_evaluation_service_for_user(current_user.id)
        result = await svc.evaluate_rag(
            query=request.query,
            contexts=request.retrieved_contexts,
            response=request.generated_response,
            ground_truth=request.ground_truth,
            metrics=request.metrics,
            api_name=provider_name,
            api_key=provider_api_key,
            user_id=stable_user_id,
            webhook_user_id=webhook_user_id,
        )
        if byok_resolution and byok_resolution.uses_byok and not explicit_key:
            await byok_resolution.touch_last_used()
        try:
            usage = result.get("usage") if isinstance(result, dict) else None
            if usage and isinstance(usage, dict):
                await limiter.record_actual_usage(user_id, "evals:rag", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0))
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass

        # Extract and format metrics from results
        raw_metrics = result["results"].get("metrics", {})
        formatted_metrics = {}
        for metric_name, metric_value in raw_metrics.items():
            if isinstance(metric_value, dict):
                score_val = metric_value.get("score", metric_value.get("raw_score", 0.0))
                try:
                    score_float = float(score_val) if score_val is not None else 0.0
                except (TypeError, ValueError):
                    score_float = 0.0
                raw_score_val = metric_value.get("raw_score")
                formatted_metrics[metric_name] = EvaluationMetric(
                    name=metric_value.get("name", metric_name),
                    score=score_float,
                    raw_score=raw_score_val,
                    explanation=metric_value.get("explanation"),
                    metadata=metric_value.get("metadata", {})
                )
            else:
                try:
                    score_float = float(metric_value) if metric_value is not None else 0.0
                except (TypeError, ValueError):
                    score_float = 0.0
                formatted_metrics[metric_name] = EvaluationMetric(
                    name=metric_name,
                    score=score_float,
                    raw_score=score_float,
                    explanation=""
                )

        # Send webhook: evaluation completed (await in TEST_MODE)
        if _is_eval_test_mode():
            await wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_COMPLETED,
                evaluation_id=result["evaluation_id"],
                data={
                    "evaluation_type": "rag",
                    "overall_score": result["results"].get("overall_score", 0.0),
                    "processing_time": result.get("evaluation_time", 0.0)
                }
            )
        else:
            _asyncio.create_task(wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_COMPLETED,
                evaluation_id=result["evaluation_id"],
                data={
                    "evaluation_type": "rag",
                    "overall_score": result["results"].get("overall_score", 0.0),
                    "processing_time": result.get("evaluation_time", 0.0)
                }
            ))

        resp_payload = RAGEvaluationResponse(
            metrics=formatted_metrics,
            overall_score=result["results"].get("overall_score", 0.0),
            retrieval_quality=result["results"].get("retrieval_quality", 0.0),
            generation_quality=result["results"].get("generation_quality", 0.0),
            suggestions=result["results"].get("suggestions", []),
            metadata={"evaluation_id": result["evaluation_id"]}
        )
        try:
            if response is not None:
                await _apply_rate_limit_headers(limiter, user_id, response, meta)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        return resp_payload

    except HTTPException:
        raise
    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"RAG evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG evaluation failed: {sanitize_error_message(e, 'RAG evaluation')}"
        ) from e


@router.post("/response-quality", response_model=ResponseQualityResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_response_quality(
    request: ResponseQualityRequest,
    http_request: Request,
    response: Response,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    """
    Evaluate the quality of a generated response.

    Checks relevance, completeness, accuracy, and format compliance.
    """
    try:
        # Per-user usage limits
        limiter = get_user_rate_limiter_for_user(current_user.id)
        tokens_est = _estimate_tokens_from_texts(
            request.prompt,
            request.response,
            request.expected_format,
            provider=getattr(request, "api_name", None),
            model=None,
        )
        allowed, meta = await limiter.check_rate_limit(user_id, endpoint="evals:response_quality", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
        if not allowed:
            retry_after = meta.get("retry_after", 60)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=meta.get("error", "Rate limit exceeded"),
                headers={"Retry-After": str(retry_after)}
            )
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        webhook_user_id = webhook_user_id_from_user(current_user)

        provider_name, provider_api_key, explicit_key, byok_resolution = await _resolve_and_validate_eval_provider(
            request,
            "response_quality",
            current_user=current_user,
            http_request=http_request,
        )

        # Send webhook: evaluation started (await in TEST_MODE)
        import asyncio as _asyncio
        import os as _os
        wm = _get_webhook_manager_for_user(current_user.id)
        import time as _time
        start_event_id = f"response_quality_{int(_time.time())}_{webhook_user_id[:8]}"
        if _is_eval_test_mode():
            await wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_STARTED,
                evaluation_id=start_event_id,
                data={
                    "evaluation_type": "response_quality",
                    "api_name": request.api_name
                }
            )
        else:
            _asyncio.create_task(wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_STARTED,
                evaluation_id=start_event_id,
                data={
                    "evaluation_type": "response_quality",
                    "api_name": request.api_name
                }
            ))

        svc = get_unified_evaluation_service_for_user(current_user.id)
        result = await svc.evaluate_response_quality(
            prompt=request.prompt,
            response=request.response,
            expected_format=request.expected_format,
            custom_criteria=request.evaluation_criteria,
            api_name=provider_name,
            api_key=provider_api_key,
            user_id=stable_user_id,
            webhook_user_id=webhook_user_id,
        )
        if byok_resolution and byok_resolution.uses_byok and not explicit_key:
            await byok_resolution.touch_last_used()
        try:
            usage = result.get("usage") if isinstance(result, dict) else None
            if usage and isinstance(usage, dict):
                await limiter.record_actual_usage(user_id, "evals:response_quality", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0))
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass

        # Convert metrics to proper EvaluationMetric structure
        metrics = {}
        for metric_name, metric_data in result["results"].get("metrics", {}).items():
            if isinstance(metric_data, dict):
                metrics[metric_name] = EvaluationMetric(
                    name=metric_data.get("name", metric_name),
                    score=metric_data.get("score", 0.0),
                    raw_score=metric_data.get("raw_score"),
                    explanation=metric_data.get("explanation"),
                    metadata=metric_data.get("metadata", {})
                )
            else:
                # Handle flat metric values (for backward compatibility)
                metrics[metric_name] = EvaluationMetric(
                    name=metric_name,
                    score=float(metric_data) if isinstance(metric_data, (int, float)) else 0.0,
                    explanation=f"{metric_name} score"
                )

        # Convert format_compliance to proper structure
        format_compliance = None
        if "format_compliance" in result["results"]:
            fc_value = result["results"]["format_compliance"]
            if isinstance(fc_value, bool):
                format_compliance = {"compliant": fc_value}
            elif isinstance(fc_value, dict):
                format_compliance = fc_value
            else:
                format_compliance = None

        # Send webhook: evaluation completed (await in TEST_MODE)
        if _is_eval_test_mode():
            await wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_COMPLETED,
                evaluation_id=result["evaluation_id"],
                data={
                    "evaluation_type": "response_quality",
                    "overall_quality": result["results"].get("overall_quality", 0.0),
                    "processing_time": result.get("evaluation_time", 0.0)
                }
            )
        else:
            _asyncio.create_task(wm.send_webhook(
                user_id=webhook_user_id,
                event=WebhookEvent.EVALUATION_COMPLETED,
                evaluation_id=result["evaluation_id"],
                data={
                    "evaluation_type": "response_quality",
                    "overall_quality": result["results"].get("overall_quality", 0.0),
                    "processing_time": result.get("evaluation_time", 0.0)
                }
            ))

        resp_payload = ResponseQualityResponse(
            metrics=metrics,
            overall_quality=result["results"].get("overall_quality", 0.0),
            format_compliance=format_compliance,
            issues=result["results"].get("issues", []),
            improvements=result["results"].get("improvements", [])
        )
        try:
            if response is not None:
                await _apply_rate_limit_headers(limiter, user_id, response, meta)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        return resp_payload

    except HTTPException:
        raise
    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Response quality evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality evaluation failed: {sanitize_error_message(e, 'quality evaluation')}"
        ) from e


@router.post("/propositions", response_model=PropositionEvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_propositions_endpoint(
    request: PropositionEvaluationRequest,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
    response: Response = None,
):
    """
    Evaluate proposition extraction quality.
    Computes precision/recall/F1 with semantic or Jaccard matching and density metrics.
    """
    try:
        # Per-user usage limits (approximate with total text tokens)
        limiter = get_user_rate_limiter_for_user(current_user.id)
        tokens_est = _estimate_tokens_from_texts(
            "\n".join(request.extracted or []),
            "\n".join(request.reference or []),
        )
        allowed, meta = await limiter.check_rate_limit(
            user_id,
            endpoint="evals:propositions",
            is_batch=False,
            tokens_requested=tokens_est,
            estimated_cost=0.0,
        )
        if not allowed:
            retry_after = meta.get("retry_after", 60)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=meta.get("error", "Rate limit exceeded"),
                headers={"Retry-After": str(retry_after)},
            )

        svc = get_unified_evaluation_service_for_user(current_user.id)
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        result = await svc.evaluate_propositions(
            extracted=request.extracted,
            reference=request.reference,
            method=request.method or 'semantic',
            threshold=request.threshold or 0.7,
            user_id=stable_user_id
        )

        metrics = result["results"].get("metrics", {})
        counts = result["results"].get("counts", {})

        resp_payload = PropositionEvaluationResponse(
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1=metrics.get("f1", 0.0),
            matched=counts.get("matched", 0),
            total_extracted=counts.get("total_extracted", 0),
            total_reference=counts.get("total_reference", 0),
            claim_density_per_100_tokens=metrics.get("claim_density_per_100_tokens", 0.0),
            avg_prop_len_tokens=metrics.get("avg_prop_len_tokens", 0.0),
            dedup_rate=metrics.get("dedup_rate", 0.0),
            details=result["results"].get("details", {}),
            metadata={"evaluation_id": result["evaluation_id"], "evaluation_time": result.get("evaluation_time")}
        )
        try:
            if response is not None:
                await _apply_rate_limit_headers(limiter, user_id, response, meta)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        return resp_payload

    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Proposition evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Proposition evaluation failed: {sanitize_error_message(e, 'proposition evaluation')}"
        ) from e









# ============= Batch Evaluation Endpoint =============

@router.post("/batch", response_model=BatchEvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def batch_evaluate(
    request: BatchEvaluationRequest,
    http_request: Request,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
    response: Response = None,
):
    """
    Run multiple evaluations in batch.

    Supports running multiple evaluation types with configurable parallelism.
    """
    try:
        start_time = time.time()
        service = get_unified_evaluation_service_for_user(current_user.id)
        byok_cache: dict[str, ResolvedByokCredentials] = {}

        async def _resolve_byok(provider_name: str) -> ResolvedByokCredentials:
            key = (provider_name or "openai").strip().lower()
            cached = byok_cache.get(key)
            if cached:
                return cached
            resolved = await _resolve_eval_credentials(
                key,
                current_user=current_user,
                request=http_request,
            )
            byok_cache[key] = resolved
            return resolved

        async def _extract_provider_and_key(
            eval_request: Any,
            eval_type: str,
        ) -> tuple[str, Optional[str], Optional[str], Optional[ResolvedByokCredentials]]:
            """Extract provider name and API key for a batch evaluation item."""
            provider_name = "openai"
            explicit_key: Optional[str] = None
            if isinstance(eval_request, dict):
                provider_name = (eval_request.get("api_name") or "openai").strip() or "openai"
                raw_api_key = eval_request.get("api_key")
                if raw_api_key:
                    logger.debug("Ignoring per-request api_key override for provider={}", provider_name)
            provider_key = provider_name.lower()
            provider_api_key = None
            byok_resolution = None

            if eval_type in {"geval", "rag", "response_quality"} and not provider_api_key:
                if _is_eval_test_mode():
                    provider_api_key = "test_api_key"
                else:
                    byok_resolution = await _resolve_byok(provider_key)
                    provider_api_key = byok_resolution.api_key

            await _validate_provider_credentials(
                eval_type,
                provider_key,
                provider_name,
                provider_api_key,
            )

            return provider_name, provider_api_key, explicit_key, byok_resolution

        # Per-user usage limits (aggregate all items)
        limiter = get_user_rate_limiter_for_user(current_user.id)
        tokens_total = 0
        etype = (request.evaluation_type or "").lower()
        for item in request.items or []:
            provider_hint = item.get("api_name") if isinstance(item, dict) else None
            model_hint = None
            # Commonly used keys for model hints across clients
            if isinstance(item, dict):
                model_hint = item.get("model") or item.get("target_model")
            if etype == "geval":
                tokens_total += _estimate_tokens_from_texts(
                    (item.get("source_text") if isinstance(item, dict) else None),
                    (item.get("summary") if isinstance(item, dict) else None),
                    provider=provider_hint,
                    model=model_hint,
                )
            elif etype == "rag":
                ctx = []
                if isinstance(item, dict):
                    rc = item.get("retrieved_contexts")
                    if isinstance(rc, list):
                        ctx = rc
                tokens_total += _estimate_tokens_from_texts(
                    (item.get("query") if isinstance(item, dict) else None),
                    "\n".join(ctx),
                    (item.get("generated_response") if isinstance(item, dict) else None),
                    (item.get("ground_truth") if isinstance(item, dict) else None),
                    provider=provider_hint,
                    model=model_hint,
                )
            elif etype == "response_quality":
                tokens_total += _estimate_tokens_from_texts(
                    (item.get("prompt") if isinstance(item, dict) else None),
                    (item.get("response") if isinstance(item, dict) else None),
                    (item.get("expected_format") if isinstance(item, dict) else None),
                    provider=provider_hint,
                    model=model_hint,
                )
            elif etype == "propositions":
                extracted = []
                reference = []
                if isinstance(item, dict):
                    if isinstance(item.get("extracted"), list):
                        extracted = item.get("extracted")
                    if isinstance(item.get("reference"), list):
                        reference = item.get("reference")
                tokens_total += _estimate_tokens_from_texts(
                    "\n".join(extracted),
                    "\n".join(reference),
                )

        allowed, meta = await limiter.check_rate_limit(
            user_id,
            endpoint=f"evals:batch:{etype}",
            is_batch=True,
            tokens_requested=tokens_total,
            estimated_cost=0.0,
        )
        if not allowed:
            retry_after = meta.get("retry_after", 60)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=meta.get("error", "Rate limit exceeded"),
                headers={"Retry-After": str(retry_after)},
            )

        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        batch_webhook_user_id = webhook_user_id_from_user(current_user)

        results = []
        failed_count = 0
        eval_type = request.evaluation_type

        # Process evaluations based on parallel setting (use parallel_workers > 1 as indicator)
        if request.parallel_workers > 1:
            # Run evaluations in parallel
            tasks_with_meta: list[
                tuple[int, dict[str, Any], str, Optional[str], Optional[ResolvedByokCredentials], Optional[str]]
            ] = []
            results_by_index: dict[int, dict[str, Any]] = {}

            def _build_parallel_task(
                eval_request: dict[str, Any],
                provider_name: str,
                provider_api_key: Optional[str],
            ):
                if eval_type == "geval":
                    return service.evaluate_geval(
                        source_text=eval_request.get("source_text", ""),
                        summary=eval_request.get("summary", ""),
                        metrics=eval_request.get("metrics", ["coherence"]),
                        api_name=provider_name,
                        api_key=provider_api_key,
                        user_id=stable_user_id,
                        webhook_user_id=batch_webhook_user_id,
                    )
                if eval_type == "rag":
                    return service.evaluate_rag(
                        query=eval_request.get("query", ""),
                        contexts=eval_request.get("retrieved_contexts", []),
                        response=eval_request.get("generated_response", ""),
                        ground_truth=eval_request.get("ground_truth"),
                        metrics=eval_request.get("metrics", ["relevance", "faithfulness"]),
                        api_name=provider_name,
                        api_key=provider_api_key,
                        user_id=stable_user_id,
                        webhook_user_id=batch_webhook_user_id,
                    )
                if eval_type == "response_quality":
                    return service.evaluate_response_quality(
                        prompt=eval_request.get("prompt", ""),
                        response=eval_request.get("response", ""),
                        expected_format=eval_request.get("expected_format"),
                        custom_criteria=eval_request.get("evaluation_criteria"),
                        api_name=provider_name,
                        api_key=provider_api_key,
                        user_id=stable_user_id,
                        webhook_user_id=batch_webhook_user_id,
                    )
                if eval_type == "ocr":
                    return service.evaluate_ocr(
                        items=eval_request.get("items", []),
                        metrics=eval_request.get("metrics"),
                        ocr_options=eval_request.get("ocr_options"),
                        thresholds=eval_request.get("thresholds"),
                        user_id=stable_user_id,
                    )
                if eval_type == "propositions":
                    return service.evaluate_propositions(
                        extracted=eval_request.get("extracted", []),
                        reference=eval_request.get("reference", []),
                        method=eval_request.get("method", "semantic"),
                        threshold=eval_request.get("threshold", 0.7),
                        user_id=stable_user_id,
                    )
                return None

            for item_index, eval_request in enumerate(request.items):
                provider_name, provider_api_key, explicit_key, byok_resolution = await _extract_provider_and_key(
                    eval_request,
                    eval_type,
                )

                if eval_type not in {"geval", "rag", "response_quality", "ocr", "propositions"}:
                    # Unknown type, create failed result
                    results_by_index[item_index] = {
                        "evaluation_id": None,
                        "status": "failed",
                        "error": f"Unknown evaluation type: {eval_type}"
                    }
                    failed_count += 1
                    if not request.continue_on_error:
                        break
                    continue

                tasks_with_meta.append(
                    (
                        item_index,
                        eval_request,
                        provider_name,
                        provider_api_key,
                        byok_resolution,
                        explicit_key,
                    )
                )

            # Wait for all tasks
            if tasks_with_meta:
                in_flight: dict[asyncio.Task[Any], tuple[int, Optional[ResolvedByokCredentials], Optional[str]]] = {}
                next_task_offset = 0
                fail_fast_cancelled = False
                cancel_reason = "Cancelled due to strict fail-fast after a prior item failure."

                while len(in_flight) < request.parallel_workers and next_task_offset < len(tasks_with_meta):
                    (
                        item_index,
                        task_request,
                        task_provider_name,
                        task_provider_api_key,
                        byok_resolution,
                        explicit_key,
                    ) = tasks_with_meta[next_task_offset]
                    next_task_offset += 1
                    task_coro = _build_parallel_task(task_request, task_provider_name, task_provider_api_key)
                    if task_coro is None:
                        results_by_index[item_index] = {
                            "evaluation_id": None,
                            "status": "failed",
                            "error": f"Unknown evaluation type: {eval_type}",
                        }
                        failed_count += 1
                        if not request.continue_on_error:
                            fail_fast_cancelled = True
                            break
                        continue
                    running_task = asyncio.create_task(task_coro)
                    in_flight[running_task] = (item_index, byok_resolution, explicit_key)

                while in_flight:
                    done, _ = await asyncio.wait(set(in_flight.keys()), return_when=asyncio.FIRST_COMPLETED)

                    for done_task in done:
                        item_index, byok_resolution, explicit_key = in_flight.pop(done_task)
                        try:
                            result = done_task.result()
                        except asyncio.CancelledError:
                            if item_index not in results_by_index:
                                results_by_index[item_index] = {
                                    "evaluation_id": None,
                                    "status": "failed",
                                    "error": cancel_reason,
                                }
                                failed_count += 1
                        except _EVALS_NONCRITICAL_EXCEPTIONS as e:
                            results_by_index[item_index] = {
                                "evaluation_id": None,
                                "status": "failed",
                                "error": str(e),
                            }
                            failed_count += 1
                            if not request.continue_on_error:
                                fail_fast_cancelled = True
                        except Exception as e:
                            results_by_index[item_index] = {
                                "evaluation_id": None,
                                "status": "failed",
                                "error": str(e),
                            }
                            failed_count += 1
                            if not request.continue_on_error:
                                fail_fast_cancelled = True
                        else:
                            if byok_resolution and byok_resolution.uses_byok and not explicit_key:
                                await byok_resolution.touch_last_used()
                            results_by_index[item_index] = {
                                "evaluation_id": result.get("evaluation_id"),
                                "status": "completed",
                                "results": result.get("results", {})
                            }

                    if fail_fast_cancelled:
                        remaining_tasks = list(in_flight.keys())
                        for remaining_task in remaining_tasks:
                            remaining_task.cancel()
                        if remaining_tasks:
                            await asyncio.gather(*remaining_tasks, return_exceptions=True)
                        for remaining_task in remaining_tasks:
                            item_index, _, _ = in_flight.pop(remaining_task)
                            if item_index not in results_by_index:
                                results_by_index[item_index] = {
                                    "evaluation_id": None,
                                    "status": "failed",
                                    "error": cancel_reason,
                                }
                                failed_count += 1

                        while next_task_offset < len(tasks_with_meta):
                            item_index, _, _, _, _, _ = tasks_with_meta[next_task_offset]
                            next_task_offset += 1
                            if item_index not in results_by_index:
                                results_by_index[item_index] = {
                                    "evaluation_id": None,
                                    "status": "failed",
                                    "error": cancel_reason,
                                }
                                failed_count += 1
                        break

                    while len(in_flight) < request.parallel_workers and next_task_offset < len(tasks_with_meta):
                        (
                            item_index,
                            task_request,
                            task_provider_name,
                            task_provider_api_key,
                            byok_resolution,
                            explicit_key,
                        ) = tasks_with_meta[next_task_offset]
                        next_task_offset += 1
                        task_coro = _build_parallel_task(task_request, task_provider_name, task_provider_api_key)
                        if task_coro is None:
                            results_by_index[item_index] = {
                                "evaluation_id": None,
                                "status": "failed",
                                "error": f"Unknown evaluation type: {eval_type}",
                            }
                            failed_count += 1
                            if not request.continue_on_error:
                                fail_fast_cancelled = True
                                break
                            continue
                        running_task = asyncio.create_task(task_coro)
                        in_flight[running_task] = (item_index, byok_resolution, explicit_key)

                for item_index in range(len(request.items)):
                    if item_index in results_by_index:
                        results.append(results_by_index[item_index])
        else:
            # Run evaluations sequentially
            for eval_request in request.items:
                try:
                    provider_name, provider_api_key, explicit_key, byok_resolution = await _extract_provider_and_key(
                        eval_request,
                        eval_type,
                    )

                    if eval_type == "geval":
                        result = await service.evaluate_geval(
                            source_text=eval_request.get("source_text", ""),
                            summary=eval_request.get("summary", ""),
                            metrics=eval_request.get("metrics", ["coherence"]),
                            api_name=provider_name,
                            api_key=provider_api_key,
                            user_id=stable_user_id,
                            webhook_user_id=batch_webhook_user_id,
                        )
                    elif eval_type == "rag":
                        result = await service.evaluate_rag(
                            query=eval_request.get("query", ""),
                            contexts=eval_request.get("retrieved_contexts", []),
                            response=eval_request.get("generated_response", ""),
                            ground_truth=eval_request.get("ground_truth"),
                            metrics=eval_request.get("metrics", ["relevance", "faithfulness"]),
                            api_name=provider_name,
                            api_key=provider_api_key,
                            user_id=stable_user_id,
                            webhook_user_id=batch_webhook_user_id,
                        )
                    elif eval_type == "response_quality":
                        result = await service.evaluate_response_quality(
                            prompt=eval_request.get("prompt", ""),
                            response=eval_request.get("response", ""),
                            expected_format=eval_request.get("expected_format"),
                            custom_criteria=eval_request.get("evaluation_criteria"),
                            api_name=provider_name,
                            api_key=provider_api_key,
                            user_id=stable_user_id,
                            webhook_user_id=batch_webhook_user_id,
                        )
                    elif eval_type == "ocr":
                        result = await service.evaluate_ocr(
                            items=eval_request.get("items", []),
                            metrics=eval_request.get("metrics"),
                            ocr_options=eval_request.get("ocr_options"),
                            thresholds=eval_request.get("thresholds"),
                            user_id=stable_user_id,
                        )
                    elif eval_type == "propositions":
                        result = await service.evaluate_propositions(
                            extracted=eval_request.get("extracted", []),
                            reference=eval_request.get("reference", []),
                            method=eval_request.get("method", "semantic"),
                            threshold=eval_request.get("threshold", 0.7),
                            user_id=stable_user_id,
                        )
                    else:
                        results.append({
                            "evaluation_id": None,
                            "status": "failed",
                            "error": f"Unknown evaluation type: {eval_type}"
                        })
                        failed_count += 1
                        continue

                    if byok_resolution and byok_resolution.uses_byok and not explicit_key:
                        await byok_resolution.touch_last_used()
                    results.append({
                        "evaluation_id": result.get("evaluation_id"),
                        "status": "completed",
                        "results": result.get("results", {})
                    })

                except _EVALS_NONCRITICAL_EXCEPTIONS as e:
                    results.append({
                        "evaluation_id": None,
                        "status": "failed",
                        "error": str(e)
                    })
                    failed_count += 1

                    # Check continue_on_error setting (inverse logic)
                    if not request.continue_on_error:
                        break

        processing_time = time.time() - start_time

        resp_payload = BatchEvaluationResponse(
            total_items=len(request.items),
            successful=len(results) - failed_count,
            failed=failed_count,
            results=results,
            aggregate_metrics={},  # TODO: Calculate aggregate metrics
            processing_time=processing_time
        )
        try:
            if response is not None:
                await _apply_rate_limit_headers(limiter, user_id, response, meta)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        return resp_payload

    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Batch evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch evaluation failed: {sanitize_error_message(e, 'batch evaluation')}"
        ) from e


# ============= OCR Evaluation Endpoint =============

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    OCREvaluationRequest,
    OCREvaluationResponse,
)


@router.post("/ocr", response_model=OCREvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_ocr_endpoint(
    request: OCREvaluationRequest,
    response: Response,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    """Evaluate OCR effectiveness on provided items (text-to-text comparison).

    Note: This endpoint currently expects pre-extracted text per item.
    PDF-based OCR execution can be added in future if needed.
    """
    try:
        # Per-user usage limits (approximate by total text length)
        limiter = get_user_rate_limiter_for_user(current_user.id)
        texts = []
        try:
            for it in request.items or []:
                if getattr(it, "extracted_text", None):
                    texts.append(it.extracted_text)
                if getattr(it, "ground_truth_text", None):
                    texts.append(it.ground_truth_text)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        tokens_est = _estimate_tokens_from_texts("\n".join(texts))
        allowed, meta = await limiter.check_rate_limit(user_id, endpoint="evals:ocr", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
        if not allowed:
            retry_after = meta.get("retry_after", 60)
            raise HTTPException(status_code=429, detail=meta.get("error", "Rate limit exceeded"), headers={"Retry-After": str(retry_after)})

        service = get_unified_evaluation_service_for_user(current_user.id)
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        result = await service.evaluate_ocr(
            items=[i.model_dump() for i in request.items],
            metrics=request.metrics,
            ocr_options=request.ocr_options,
            thresholds=request.thresholds,
            user_id=stable_user_id,
        )
        try:
            usage = result.get("usage") if isinstance(result, dict) else None
            if usage and isinstance(usage, dict):
                await limiter.record_actual_usage(user_id, "evals:ocr", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0))
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        # Apply headers
        try:
            if response is not None:
                await _apply_rate_limit_headers(limiter, user_id, response, meta)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        return OCREvaluationResponse(**result)
    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"OCR evaluation endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR evaluation failed: {sanitize_error_message(e, 'ocr evaluation')}"
        ) from e


@router.post("/ocr-pdf", response_model=OCREvaluationResponse, dependencies=[Depends(check_evaluation_rate_limit)])
async def evaluate_ocr_pdf_endpoint(
    response: Response,
    files: list[UploadFile] = File(..., description="PDF files to OCR and evaluate"),
    ground_truths: Optional[list[str]] = Form(None, description="Ground-truth text per file (order aligned)"),
    ground_truths_json: Optional[str] = Form(None, description="JSON array of ground-truth texts aligned to files"),
    metrics: Optional[list[str]] = Form(None, description="Metrics to compute (cer, wer, coverage, page_coverage)"),
    ground_truths_pages_json: Optional[str] = Form(None, description="JSON array of per-file page arrays of ground truth text (e.g., [[...],[...]])"),
    thresholds_json: Optional[str] = Form(None, description="JSON dict of thresholds (max_cer, max_wer, min_coverage, min_page_coverage)"),
    enable_ocr: bool = Form(True, description="Enable OCR"),
    ocr_backend: Optional[str] = Form(None, description="OCR backend (e.g., 'tesseract' or 'auto')"),
    ocr_lang: str = Form("eng", description="OCR language"),
    ocr_dpi: int = Form(300, description="Render DPI (72-600)"),
    ocr_mode: str = Form("fallback", description="OCR mode: 'always' or 'fallback'"),
    ocr_min_page_text_chars: int = Form(40, description="Threshold for per-page OCR fallback"),
    ocr_output_format: Optional[str] = Form(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Form(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    """Evaluate OCR by running OCR on uploaded PDFs and comparing to provided ground-truths."""
    try:
        # Per-user usage limits (approximate by file sizes if available)
        limiter = get_user_rate_limiter_for_user(current_user.id)
        size_est = 0
        try:
            for f in files:
                s = getattr(f, "size", None)
                if isinstance(s, int):
                    size_est += s
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        tokens_est = max(0, size_est // 4)
        allowed, meta = await limiter.check_rate_limit(user_id, endpoint="evals:ocr_pdf", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
        if not allowed:
            retry_after = meta.get("retry_after", 60)
            raise HTTPException(status_code=429, detail=meta.get("error", "Rate limit exceeded"), headers={"Retry-After": str(retry_after)})
        if ground_truths is not None and len(ground_truths) not in (0, len(files)):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ground_truths count must match files length or be omitted")

        items: list[dict[str, Any]] = []
        gt_list = ground_truths or []
        if ground_truths_json:
            try:
                parsed = json.loads(ground_truths_json)
            except (json.JSONDecodeError, TypeError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid ground_truths_json",
                ) from exc
            if isinstance(parsed, list):
                gt_list = parsed

        gt_pages_list = None
        if ground_truths_pages_json:
            try:
                parsed_pages = json.loads(ground_truths_pages_json)
            except (json.JSONDecodeError, TypeError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid ground_truths_pages_json",
                ) from exc
            if isinstance(parsed_pages, list):
                gt_pages_list = parsed_pages

        for idx, f in enumerate(files):
            content = await f.read()
            gt = gt_list[idx] if idx < len(gt_list) else None
            item = {
                "id": f.filename or f"file_{idx}",
                "pdf_bytes": content,
                "ground_truth_text": gt,
            }
            if gt_pages_list and idx < len(gt_pages_list) and isinstance(gt_pages_list[idx], list):
                item["ground_truth_pages"] = gt_pages_list[idx]
            items.append(item)

        ocr_options = {
            "enable_ocr": enable_ocr,
            "ocr_backend": ocr_backend,
            "ocr_lang": ocr_lang,
            "ocr_dpi": int(ocr_dpi),
            "ocr_mode": ocr_mode,
            "ocr_min_page_text_chars": int(ocr_min_page_text_chars),
            "ocr_output_format": ocr_output_format,
            "ocr_prompt_preset": ocr_prompt_preset,
        }

        service = get_unified_evaluation_service_for_user(current_user.id)
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)
        thresholds = None
        if thresholds_json:
            try:
                thresholds = json.loads(thresholds_json)
            except (json.JSONDecodeError, TypeError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid thresholds_json",
                ) from exc
        result = await service.evaluate_ocr(
            items=items,
            metrics=metrics,
            ocr_options=ocr_options,
            thresholds=thresholds,
            user_id=stable_user_id,
        )
        try:
            usage = result.get("usage") if isinstance(result, dict) else None
            if usage and isinstance(usage, dict):
                await limiter.record_actual_usage(user_id, "evals:ocr_pdf", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0))
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if response is not None:
                await _apply_rate_limit_headers(limiter, user_id, response, meta)
        except _EVALS_NONCRITICAL_EXCEPTIONS:
            pass
        return OCREvaluationResponse(**result)
    except HTTPException:
        raise
    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"OCR PDF evaluation endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR PDF evaluation failed: {sanitize_error_message(e, 'ocr pdf evaluation')}"
        ) from e


# ============= Evaluation History Endpoint =============

@router.post("/history", response_model=EvaluationHistoryResponse)
async def get_evaluation_history(
    request: EvaluationHistoryRequest,
    principal: Annotated[AuthPrincipal, Depends(get_auth_principal)],
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    """
    Retrieve evaluation history for a user.

    Supports filtering by date range, evaluation type, and pagination.
    """
    try:
        service = get_unified_evaluation_service_for_user(current_user.id)
        stable_user_id = getattr(current_user, "id_str", None) or str(current_user.id)

        def _user_id_variants(raw: Optional[str]) -> set[str]:
            if not raw:
                return set()
            val = str(raw).strip()
            if not val:
                return set()
            variants = {val}
            if val.startswith("user_"):
                core = val[5:]
                if core:
                    variants.add(core)
            elif val.isdigit():
                variants.add(f"user_{val}")
            return variants

        requested_user_id = request.user_id
        target_user_id = requested_user_id or stable_user_id
        if requested_user_id:
            allowed = _user_id_variants(stable_user_id)
            if requested_user_id not in allowed:
                roles = {
                    str(role).strip().lower()
                    for role in ((principal.roles if principal else []) or [])
                    if str(role).strip()
                }
                permissions = {
                    str(permission).strip().lower()
                    for permission in ((principal.permissions if principal else []) or [])
                    if str(permission).strip()
                }
                is_admin = bool("admin" in roles or ("*" in permissions) or ("system.configure" in permissions))
                if not is_admin:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin privileges required to request other users' evaluation history",
                    )

        # Get evaluations from database
        evaluations = await service.get_evaluation_history(
            user_id=target_user_id,
            evaluation_type=request.evaluation_type,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.limit or 100,
            offset=request.offset or 0
        )

        # Get total count for pagination
        total_count = await service.count_evaluations(
            user_id=target_user_id,
            evaluation_type=request.evaluation_type,
            start_date=request.start_date,
            end_date=request.end_date
        )

        return EvaluationHistoryResponse(
            items=evaluations,
            total_count=total_count,
            aggregations={
                "limit": request.limit or 100,
                "offset": request.offset or 0,
                "filtered_by": {
                    "user_id": target_user_id,
                    "evaluation_type": request.evaluation_type,
                    "date_range": {
                        "start": request.start_date.isoformat() if request.start_date else None,
                        "end": request.end_date.isoformat() if request.end_date else None
                    }
                }
            }
        )

    except _EVALS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to retrieve evaluation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {sanitize_error_message(e, 'retrieving history')}"
        ) from e


def _promote_static_routes(_router: APIRouter) -> None:
    """Ensure static paths are registered before catch-all routes."""
    try:
        prioritized_suffixes = ("/health", "/metrics")
        # Move each target to the front, preserving relative order
        for suffix in reversed(prioritized_suffixes):
            prefixed = f"{_router.prefix}{suffix}" if _router.prefix else suffix
            for idx, route in enumerate(list(_router.routes)):
                path = getattr(route, "path", "")
                if path in {suffix, prefixed} and isinstance(route, APIRoute):
                    _router.routes.insert(0, _router.routes.pop(idx))
                    break
    except _EVALS_NONCRITICAL_EXCEPTIONS as exc:  # Safety: never fail import due to ordering tweak
        with contextlib.suppress(_EVALS_NONCRITICAL_EXCEPTIONS):
            logger.debug(f"Failed to promote evaluation routes: {exc}")


_promote_static_routes(router)
