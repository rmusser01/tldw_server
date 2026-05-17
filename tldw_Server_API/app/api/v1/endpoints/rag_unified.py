"""
Unified RAG API Endpoint

This is the new, simplified RAG API that uses the unified pipeline.
All features are accessible through explicit parameters.
"""

import asyncio
import hashlib
import inspect
import json
import os
import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal, get_request_user, rbac_rate_limit, RequirePermission, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.API_Deps.billing_deps import require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user

# Dependencies
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user

# Schemas
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
    ImplicitFeedbackEvent,
    KnowledgeSourceHealthResponse,
    UnifiedBatchRequest,
    UnifiedBatchResponse,
    UnifiedRAGRequest,
    UnifiedRAGResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
    AgenticConfig,
    agentic_rag_pipeline,
)
from tldw_Server_API.app.core.RAG.rag_service.agentic_execution import (
    build_agentic_execution_context,
)
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MultiDatabaseRetriever,
    RetrievalConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.generation import generate_streaming_response
from tldw_Server_API.app.core.RAG.rag_service.request_bundle import (
    ResolvedRequestBundle,
    build_request_bundle,
)
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import (
    ResolvedRAGRequest,
    resolve_rag_request,
)
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import (
    RetrievalPlan,
    build_retrieval_plan,
)
from tldw_Server_API.app.core.RAG.rag_service.response_mapping import (
    rag_result_from_unified_search_result,
    rag_result_to_response,
)
from tldw_Server_API.app.core.RAG.rag_service.source_health import build_source_health_entries
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import stream_rag_events
from tldw_Server_API.app.core.config import get_config_value, settings

# Unified Pipeline
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    UnifiedSearchResult,
    advanced_search,
    simple_search,
    unified_batch_pipeline,
    unified_rag_pipeline,
)

_BATCH_ROUND2_DEFAULT_FIELDS = {
    "enable_suggestions",
    "enable_structured_response",
    "enable_image_search",
    "enable_video_search",
}

_READY_MEDIA_COLLECTION_STATUSES = frozenset({"completed", "skipped_existing"})
_EMPTY_MEDIA_SCOPE_SENTINEL = -1


def _copy_rag_request_with_updates(
    request: UnifiedRAGRequest,
    updates: dict[str, Any],
) -> UnifiedRAGRequest:
    """Return a copy of a RAG request using the active Pydantic compatibility API."""
    model_copy = getattr(request, "model_copy", None)
    if callable(model_copy):
        return model_copy(update=updates)
    return request.copy(update=updates)


def _ready_media_ids_from_collection(collection: Any) -> list[int]:
    """Extract ordered, unique ready media IDs from a durable media collection row."""
    ready_ids: list[int] = []
    seen: set[int] = set()
    for item in getattr(collection, "items", []) or []:
        if getattr(item, "status", None) not in _READY_MEDIA_COLLECTION_STATUSES:
            continue
        media_id = getattr(item, "media_id", None)
        if not isinstance(media_id, int) or media_id <= 0 or media_id in seen:
            continue
        ready_ids.append(media_id)
        seen.add(media_id)
    return ready_ids


def _apply_media_collection_scope(
    request: UnifiedRAGRequest,
    collections_db: CollectionsDatabase,
) -> UnifiedRAGRequest:
    """Resolve request.collection_id to backend-owned ready media IDs for RAG retrieval."""
    collection_id = getattr(request, "collection_id", None)
    if collection_id is None:
        return request

    try:
        collection = collections_db.get_media_collection(int(collection_id))
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="media_collection_not_found",
        ) from exc
    ready_media_ids = _ready_media_ids_from_collection(collection)
    explicit_media_ids = request.include_media_ids
    if explicit_media_ids is not None:
        ready_set = set(ready_media_ids)
        scoped_media_ids = [
            int(media_id)
            for media_id in explicit_media_ids
            if isinstance(media_id, int) and media_id in ready_set
        ]
    else:
        scoped_media_ids = ready_media_ids

    if not scoped_media_ids:
        scoped_media_ids = [_EMPTY_MEDIA_SCOPE_SENTINEL]

    return _copy_rag_request_with_updates(request, {"include_media_ids": scoped_media_ids})


def _search_agent_setting(env_key: str, config_key: str) -> Optional[str]:
    """Read Search-Agent setting with env-over-config precedence."""
    env_value = os.getenv(env_key)
    if env_value is not None:
        return env_value
    try:
        return get_config_value("Search-Agent", config_key, default=None)
    except (TypeError, ValueError):
        return None


def _build_unified_pipeline_kwargs(
    request: UnifiedRAGRequest,
    db_paths: dict[str, Optional[str]],
    media_db: Any,
    chacha_db: CharactersRAGDB,
    current_user: Optional[User],
    prompts_db: Optional[PromptsDatabase] = None,
    resolved_request: Optional[ResolvedRAGRequest] = None,
    retrieval_plan: Optional[RetrievalPlan] = None,
) -> dict[str, Any]:
    """Translate a resolved standard request into core pipeline keyword arguments."""
    if resolved_request is None:
        resolved_request = resolve_rag_request(
            request,
            current_user=current_user,
            single_user_id_resolver=DatabasePaths.get_single_user_id,
            search_agent_setting_fn=_search_agent_setting,
        )
    if retrieval_plan is None:
        retrieval_plan = build_retrieval_plan(resolved_request)
    payload = dict(resolved_request.payload)
    payload["sources"] = list(retrieval_plan.sources)
    payload["media_db_path"] = db_paths.get("media_db_path")
    payload["notes_db_path"] = db_paths.get("notes_db_path")
    payload["character_db_path"] = db_paths.get("character_db_path")
    payload["kanban_db_path"] = db_paths.get("kanban_db_path")
    payload["prompts_db_path"] = db_paths.get("prompts_db_path")
    payload["media_db"] = media_db
    payload["chacha_db"] = chacha_db
    if prompts_db is not None:
        payload["prompts_db"] = prompts_db
    payload["index_namespace"] = retrieval_plan.index_namespace
    payload["retrieval_plan"] = retrieval_plan
    payload["user_id"] = resolved_request.user_id
    payload["feedback_user_id"] = resolved_request.feedback_user_id
    signature = inspect.signature(unified_rag_pipeline)
    params = list(signature.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return payload
    allowed = set(signature.parameters.keys())
    return {k: v for k, v in payload.items() if k in allowed}


def _build_batch_pipeline_kwargs(
    request: UnifiedBatchRequest,
    db_paths: dict[str, Optional[str]],
    current_user: Optional[User],
    resolved_request: Optional[ResolvedRAGRequest] = None,
    retrieval_plan: Optional[RetrievalPlan] = None,
) -> dict[str, Any]:
    """Translate a resolved batch request into shared batch pipeline options."""
    if resolved_request is None:
        resolved_request = resolve_rag_request(
            request,
            current_user=current_user,
            single_user_id_resolver=DatabasePaths.get_single_user_id,
            search_agent_setting_fn=_search_agent_setting,
            search_agent_allowed_fields=_BATCH_ROUND2_DEFAULT_FIELDS,
        )
    if retrieval_plan is None:
        retrieval_plan = build_retrieval_plan(resolved_request)
    payload = dict(resolved_request.payload)
    payload.pop("queries", None)
    payload.pop("query", None)
    payload.pop("max_concurrent", None)
    payload.pop("enable_checkpoint", None)
    payload["sources"] = list(retrieval_plan.sources)
    payload.update(db_paths)
    payload["index_namespace"] = retrieval_plan.index_namespace
    payload["retrieval_plan"] = retrieval_plan
    payload["user_id"] = resolved_request.user_id
    payload["feedback_user_id"] = resolved_request.feedback_user_id
    return payload


def _build_standard_request_bundle(
    request: UnifiedRAGRequest,
    *,
    current_user: Optional[User],
    db_paths: dict[str, Optional[str]],
    media_db: Any,
    chacha_db: CharactersRAGDB,
    prompts_db: Optional[PromptsDatabase] = None,
) -> ResolvedRequestBundle:
    """Resolve a standard request once and attach endpoint-owned pipeline resources."""
    return build_request_bundle(
        request=request,
        current_user=current_user,
        resolve_request_kwargs={
            "single_user_id_resolver": DatabasePaths.get_single_user_id,
            "search_agent_setting_fn": _search_agent_setting,
        },
        pipeline_kwargs_builder=lambda *, resolved_request, retrieval_plan: _build_unified_pipeline_kwargs(
            request=request,
            db_paths=db_paths,
            media_db=media_db,
            chacha_db=chacha_db,
            prompts_db=prompts_db,
            current_user=current_user,
            resolved_request=resolved_request,
            retrieval_plan=retrieval_plan,
        ),
    )


def _build_batch_request_bundle(
    request: UnifiedBatchRequest,
    *,
    current_user: Optional[User],
    db_paths: dict[str, Optional[str]],
) -> ResolvedRequestBundle:
    """Resolve a batch request once and attach endpoint-owned pipeline paths."""
    return build_request_bundle(
        request=request,
        current_user=current_user,
        resolve_request_kwargs={
            "single_user_id_resolver": DatabasePaths.get_single_user_id,
            "search_agent_setting_fn": _search_agent_setting,
            "search_agent_allowed_fields": _BATCH_ROUND2_DEFAULT_FIELDS,
        },
        pipeline_kwargs_builder=lambda *, resolved_request, retrieval_plan: _build_batch_pipeline_kwargs(
            request=request,
            db_paths=db_paths,
            current_user=current_user,
            resolved_request=resolved_request,
            retrieval_plan=retrieval_plan,
        ),
    )


def _build_resume_batch_request(
    checkpoint_config: dict[str, Any],
    *,
    remaining_queries: list[str],
    max_concurrent: int,
) -> UnifiedBatchRequest:
    """Rebuild a batch request from persisted checkpoint config and remaining work."""
    request_payload = dict(checkpoint_config or {})
    request_payload["queries"] = list(remaining_queries)
    request_payload["max_concurrent"] = max_concurrent
    request_payload["enable_checkpoint"] = False
    return UnifiedBatchRequest(**request_payload)


_CHECKPOINT_UNSUPPORTED = object()


def _checkpoint_safe_value(value: Any) -> Any:
    """Return a JSON-persistable checkpoint value or the unsupported sentinel."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                continue
            safe_item = _checkpoint_safe_value(item)
            if safe_item is not _CHECKPOINT_UNSUPPORTED:
                safe[str(key)] = safe_item
        return safe
    if isinstance(value, (list, tuple)):
        safe_items: list[Any] = []
        for item in value:
            safe_item = _checkpoint_safe_value(item)
            if safe_item is not _CHECKPOINT_UNSUPPORTED:
                safe_items.append(safe_item)
        return safe_items
    return _CHECKPOINT_UNSUPPORTED


def _sanitize_checkpoint_config_for_persistence(config: dict[str, Any]) -> dict[str, Any]:
    """Drop runtime-only objects before persisting batch checkpoint config."""
    sanitized: dict[str, Any] = {}
    for key, value in dict(config or {}).items():
        safe_value = _checkpoint_safe_value(value)
        if safe_value is not _CHECKPOINT_UNSUPPORTED:
            sanitized[str(key)] = safe_value
    return sanitized


def _sync_retriever_overrides_to_pipeline() -> None:
    """
    Keep endpoint-level retriever monkeypatches effective.

    Several integration tests patch ``rag_unified.MultiDatabaseRetriever``.
    The streaming and unified endpoints now delegate retrieval to
    ``unified_rag_pipeline`` which references its own module globals.
    This hook synchronizes the pipeline's references with the endpoint's.
    """
    try:
        import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

        if getattr(up, "MultiDatabaseRetriever", None) is not MultiDatabaseRetriever:
            up.MultiDatabaseRetriever = MultiDatabaseRetriever  # type: ignore[assignment]

        # RetrievalConfig should normally already be set, but be defensive.
        if getattr(up, "RetrievalConfig", None) is None and RetrievalConfig is not None:
            up.RetrievalConfig = RetrievalConfig  # type: ignore[assignment]
    except (ImportError, AttributeError, TypeError):
        logger.debug("Failed to sync retriever overrides to unified pipeline", exc_info=True)


def _resolve_implicit_feedback_user_id(
    request_user_id: Optional[str],
    current_user: Optional[User],
) -> Optional[str]:
    """
    Resolve a stable user identifier for implicit-feedback personalization state.

    Prefer explicit request user_id when provided, but normalize legacy
    ``single_user`` aliases to a numeric user id to avoid filesystem-path
    validation failures.
    """
    raw_request = str(request_user_id).strip() if request_user_id is not None else ""
    if raw_request:
        if raw_request.lower() == "single_user":
            if current_user is not None:
                current_id = getattr(current_user, "id_int", None)
                if isinstance(current_id, int):
                    return str(current_id)
                fallback_id = getattr(current_user, "id", None)
                if fallback_id is not None:
                    fallback_raw = str(fallback_id).strip()
                    if fallback_raw:
                        return fallback_raw
            try:
                return str(DatabasePaths.get_single_user_id())
            except (RuntimeError, ValueError, OSError, TypeError):
                pass
        return raw_request

    if current_user is None:
        return None

    current_id = getattr(current_user, "id_int", None)
    if isinstance(current_id, int):
        return str(current_id)

    fallback_id = getattr(current_user, "id", None)
    if fallback_id is None:
        return None

    fallback_raw = str(fallback_id).strip()
    return fallback_raw or None


def _resolve_kanban_db_path(current_user: Optional[User], request_user_id: Optional[str] = None) -> Optional[str]:
    """Resolve the Kanban DB path for the active user context."""
    user_id: Optional[Any] = None
    try:
        if current_user is not None:
            for attr in ("id", "id_int", "username"):
                value = getattr(current_user, attr, None)
                if value is not None:
                    user_id = value
                    break
        elif request_user_id:
            user_id = request_user_id
    except (AttributeError, TypeError):
        logger.debug("Failed to resolve user_id for kanban DB path", exc_info=True)
        user_id = request_user_id if current_user is None else None
    if user_id is None:
        try:
            user_id = DatabasePaths.get_single_user_id()
        except (RuntimeError, ValueError, OSError, TypeError):
            logger.debug("Failed to resolve single-user ID for kanban DB path", exc_info=True)
            return None
    try:
        return str(DatabasePaths.get_kanban_db_path(user_id))
    except (RuntimeError, ValueError, OSError, TypeError):
        logger.debug("Failed to resolve kanban DB path", exc_info=True)
        return None


def _resolve_source_health_user_id(current_user: Optional[User], request_user_id: Optional[str] = None) -> Optional[str]:
    """Resolve a filesystem-safe user directory component without creating storage."""
    candidates: list[Any] = []
    if current_user is not None:
        for attr in ("id", "id_int"):
            candidates.append(getattr(current_user, attr, None))
    candidates.append(request_user_id)

    for candidate in candidates:
        if candidate is None:
            continue
        raw = str(candidate).strip()
        if raw.isdigit() and int(raw) > 0:
            return raw
    try:
        fallback = DatabasePaths.get_single_user_id()
        fallback_raw = str(fallback).strip()
        if fallback_raw.isdigit() and int(fallback_raw) > 0:
            return fallback_raw
    except (RuntimeError, ValueError, OSError, TypeError):
        logger.debug("Failed to resolve single-user ID for source health path", exc_info=True)
    return None


def _resolve_existing_source_db_paths(
    current_user: Optional[User],
    request_user_id: Optional[str] = None,
) -> dict[str, str]:
    """Return existing source database paths without creating source storage."""
    user_id = _resolve_source_health_user_id(current_user, request_user_id)
    if user_id is None:
        return {}

    user_dir = DatabasePaths.resolve_user_db_base_dir() / user_id
    candidates = {
        "media_db": user_dir / DatabasePaths.MEDIA_DB_NAME,
        "chacha_db": user_dir / DatabasePaths.CHACHA_DB_NAME,
        "prompts_db": user_dir / DatabasePaths.PROMPTS_SUBDIR / DatabasePaths.PROMPTS_DB_NAME,
        "kanban_db": user_dir / DatabasePaths.KANBAN_DB_NAME,
    }
    return {
        source_key: str(path)
        for source_key, path in candidates.items()
        if path.is_file()
    }


def _media_db_uses_non_file_storage() -> bool:
    """Return whether Media DB search is configured for non-file content storage."""
    backend_mode_hint = (
        os.getenv("CONTENT_DB_MODE")
        or os.getenv("TLDW_CONTENT_DB_BACKEND")
        or str(settings.get("CONTENT_DB_BACKEND", "sqlite"))
    )
    return backend_mode_hint.strip().lower() in {"postgres", "postgresql"}


def _build_source_health_source_sets(
    *,
    existing_paths: dict[str, str],
    media_backend_uses_non_file_storage: bool = False,
) -> tuple[set[Any], set[Any]]:
    """Derive ready and empty source sets without creating source storage."""
    configured: set[Any] = set()
    empty: set[Any] = set()
    if "media_db" in existing_paths or media_backend_uses_non_file_storage:
        configured.add("media_db")
    else:
        empty.add("media_db")
    if "chacha_db" in existing_paths:
        configured.update(
            {
                "notes",
                "chats",
                "characters",
                "world_books",
                "dictionaries",
            }
        )
    else:
        empty.update(
            {
                "notes",
                "chats",
                "characters",
                "world_books",
                "dictionaries",
            }
        )
    if "prompts_db" in existing_paths:
        configured.add("prompts")
    else:
        empty.add("prompts")
    if "kanban_db" in existing_paths:
        configured.add("kanban")
    else:
        empty.add("kanban")
    return configured, empty


from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.core.RAG.rag_service.analytics_system import UnifiedFeedbackSystem

router = APIRouter(prefix="/api/v1/rag", tags=["rag-unified"])

# Use central limiter instance for consistency across the app

async def _log_rag_queries_for_org(
    request_raw: Request,
    current_user: User,
    units: int = 1,
) -> None:
    """
    Best-effort helper to record RAG query usage into the shared
    ResourceDailyLedger for the active organization.

    This function never raises; failures are logged at debug level only.
    """
    if units <= 0:
        return

    try:
        # Resolve org_id from request state if available.
        org_id: Optional[int] = None
        try:
            state = getattr(request_raw, "state", None)
            if state is not None:
                org_ids = getattr(state, "org_ids", None)
                if isinstance(org_ids, (list, tuple)) and org_ids:
                    org_id_candidate = org_ids[0]
                    try:
                        org_id = int(org_id_candidate)
                    except (TypeError, ValueError):
                        org_id = None
        except (AttributeError, TypeError):
            org_id = None

        # Fallback: derive org_id from AuthNZ org memberships.
        if org_id is None and current_user and current_user.id_int is not None:
            try:
                from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
                from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

                pool = await get_db_pool()
                repo = AuthnzOrgsTeamsRepo(db_pool=pool)
                memberships = await repo.list_org_memberships_for_user(current_user.id_int)
                if memberships:
                    candidate = memberships[0].get("org_id")
                    if candidate is not None:
                        org_id = int(candidate)
            except Exception:  # noqa: BLE001 - best-effort fallback for org lookup
                org_id = None

        if org_id is None:
            return

        try:
            from datetime import datetime, timezone

            from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (
                LedgerEntry,
                ResourceDailyLedger,
            )

            ledger = ResourceDailyLedger()
            await ledger.initialize()

            now = datetime.now(timezone.utc)
            entry = LedgerEntry(  # type: ignore[call-arg]
                entity_scope="org",
                entity_value=str(org_id),
                category="rag_queries",
                units=int(units),
                op_id=f"rag:{org_id}:{uuid4()}",
                occurred_at=now,
            )
            await ledger.add(entry)
        except Exception:  # noqa: BLE001 - ledger failures must not impact requests
            # Ledger write failures must never impact request flow.
            logger.debug("RAG query ledger write failed; continuing without usage record", exc_info=True)
    except Exception:  # noqa: BLE001 - guard against unexpected failures in logging helper
        # Guard against any unexpected failure paths.
        logger.debug("RAG query logging failed; continuing without usage record", exc_info=True)


# =============== Ablation helper ===============
try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object  # type: ignore
    def Field(*a, **k):  # type: ignore
        return None


class AblationRequest(BaseModel):  # type: ignore[misc]
    query: str = Field(..., description="Query to ablate")
    top_k: int = Field(10, ge=1, le=50, description="Retrieval top_k")
    search_mode: str = Field("hybrid", description="fts|vector|hybrid")
    with_answer: bool = Field(False, description="Generate answer in each condition")
    agentic_top_k_docs: int = Field(3, ge=1, le=20)
    agentic_window_chars: int = Field(1200, ge=200, le=20000)
    agentic_max_tokens_read: int = Field(6000, ge=500, le=20000)
    reranking_strategy: str = Field("flashrank", description="flashrank|cross_encoder|hybrid|llama_cpp|none")


@router.post(
    "/ablate",
    summary="Run RAG ablations (baseline, +rerank, +agentic, +agentic strict)",
    description="Compare retrieval/generation across baseline vs reranked vs agentic vs agentic(stricter extractive).",
    dependencies=[Depends(check_rate_limit)]
)
async def rag_ablate(
    request: AblationRequest,
    current_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    kanban_db_path = _resolve_kanban_db_path(current_user)
    db_paths = {
        "media_db_path": media_db.db_path if media_db else None,
        "notes_db_path": chacha_db.db_path if chacha_db else None,
        "character_db_path": chacha_db.db_path if chacha_db else None,
        "kanban_db_path": kanban_db_path,
    }

    common = {
        "query": request.query,
        "sources": ["media_db"],
        "media_db_path": db_paths["media_db_path"],
        "notes_db_path": db_paths["notes_db_path"],
        "character_db_path": db_paths["character_db_path"],
        "kanban_db_path": db_paths["kanban_db_path"],
        "media_db": media_db,
        "chacha_db": chacha_db,
        "search_mode": request.search_mode,
        "top_k": request.top_k,
        "min_score": 0.0,
        "enable_generation": bool(request.with_answer),
        "generation_model": None,
        "max_generation_tokens": 300,
    }

    runs = []

    # 1) Baseline (no reranking)
    r1 = await unified_rag_pipeline(
        **common,
        enable_reranking=False,
    )
    runs.append({
        "label": "baseline",
        "result": rag_result_to_response(rag_result_from_unified_search_result(r1))
    })

    # 2) +rerank
    r2 = await unified_rag_pipeline(
        **common,
        enable_reranking=True,
        reranking_strategy=request.reranking_strategy,
    )
    runs.append({
        "label": "+rerank",
        "result": rag_result_to_response(rag_result_from_unified_search_result(r2))
    })

    # 3) agentic
    a_cfg = AgenticConfig(
        top_k_docs=request.agentic_top_k_docs,
        window_chars=request.agentic_window_chars,
        max_tokens_read=request.agentic_max_tokens_read,
        max_tool_calls=6,
        extractive_only=True,
        quote_spans=True,
        enable_tools=False,
        debug_trace=False,
    )
    r3 = await agentic_rag_pipeline(
        **common,
        agentic=a_cfg,
        enable_citations=False,
    )
    runs.append({
        "label": "agentic",
        "result": rag_result_to_response(rag_result_from_unified_search_result(r3))
    })

    # 4) agentic (strict): tools on, extractive only, small budget
    a_cfg_strict = AgenticConfig(
        top_k_docs=max(1, request.agentic_top_k_docs),
        window_chars=max(600, int(request.agentic_window_chars / 2)),
        max_tokens_read=max(1000, int(request.agentic_max_tokens_read / 2)),
        max_tool_calls=4,
        extractive_only=True,
        quote_spans=True,
        enable_tools=True,
        time_budget_sec=5.0,
        debug_trace=False,
    )
    r4 = await agentic_rag_pipeline(
        **common,
        agentic=a_cfg_strict,
        enable_citations=False,
    )
    runs.append({
        "label": "agentic_strict",
        "result": rag_result_to_response(rag_result_from_unified_search_result(r4))
    })

    # Compact output for quick comparison
    out = []
    for item in runs:
        res = item["result"]
        first = (res.documents[0] if res.documents else None)
        out.append({
            "label": item["label"],
            "total_time": res.total_time,
            "cache_hit": res.cache_hit,
            "doc_count": len(res.documents or []),
            "first_doc_id": (first.get("id") if isinstance(first, dict) else getattr(first, 'id', None)) if first else None,
        })

    return {"summary": out, "runs": runs}


@router.get(
    "/capabilities",
    summary="Capabilities",
    description="List RAG pipeline features and defaults available to the current user"
)
async def get_capabilities(request: Request):
    """Return supported features, defaults and configuration limits for the unified RAG pipeline.

    This endpoint is informational and does not require database access. It reflects
    the capabilities compiled into the service and basic configuration toggles.
    """
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.config import RAG_SERVICE_CONFIG

    settings = get_settings()

    # High-level features supported by the pipeline
    import os as _os
    # Resolve environment-defaults for VLM
    vlm_defaults = {
        "VLM_TABLE_MODEL_NAME": _os.getenv("VLM_TABLE_MODEL_NAME", "microsoft/table-transformer-detection"),
        "VLM_TABLE_REVISION": _os.getenv("VLM_TABLE_REVISION", None),
        "VLM_TABLE_THRESHOLD": _os.getenv("VLM_TABLE_THRESHOLD", "0.9"),
    }

    features = {
        "agentic_chunking": {
            "supported": True,
            "strategies": ["standard", "agentic"],
            "parameters": [
                "strategy",
                "agentic_top_k_docs",
                "agentic_window_chars",
                "agentic_max_tokens_read",
                "agentic_max_tool_calls",
                "agentic_enable_tools",
                "agentic_use_llm_planner",
                "agentic_time_budget_sec",
                "agentic_cache_ttl_sec",
                "agentic_enable_query_decomposition",
                "agentic_subgoal_max",
                "agentic_enable_semantic_within",
                "agentic_enable_section_index",
                "agentic_prefer_structural_anchors",
                "agentic_enable_table_support",
                "agentic_enable_vlm_late_chunking",
                "agentic_vlm_backend",
                "agentic_vlm_detect_tables_only",
                "agentic_vlm_max_pages",
                "agentic_vlm_late_chunk_top_k_docs",
                "agentic_use_provider_embeddings_within",
                "agentic_provider_embedding_model_id",
                "agentic_extractive_only",
                "agentic_quote_spans",
                "agentic_debug_trace",
                "agentic_adaptive_budgets",
                "agentic_coverage_target",
                "agentic_min_corroborating_docs",
                "agentic_max_redundancy",
                "agentic_enable_metrics",
                "explain_only",
            ],
            "defaults": {
                "strategy": "standard",
                "agentic_top_k_docs": 3,
                "agentic_window_chars": 1200,
                "agentic_max_tokens_read": 6000,
                "agentic_max_tool_calls": 8,
                "agentic_enable_tools": False,
                "agentic_use_llm_planner": False,
                "agentic_cache_ttl_sec": 600,
                "agentic_enable_query_decomposition": False,
                "agentic_subgoal_max": 3,
                "agentic_enable_semantic_within": True,
                "agentic_enable_section_index": True,
                "agentic_prefer_structural_anchors": True,
                "agentic_enable_table_support": True,
                "agentic_enable_vlm_late_chunking": False,
                "agentic_vlm_backend": None,
                "agentic_vlm_detect_tables_only": True,
                "agentic_vlm_max_pages": None,
                "agentic_vlm_late_chunk_top_k_docs": 2,
                "agentic_use_provider_embeddings_within": False,
                "agentic_provider_embedding_model_id": None,
                "agentic_adaptive_budgets": True,
                "agentic_coverage_target": 0.8,
                "agentic_min_corroborating_docs": 2,
                "agentic_max_redundancy": 0.9,
                "agentic_enable_metrics": True,
            },
        },
        "query_expansion": {
            "supported": True,
            "methods": ["acronym", "synonym", "domain", "entity"],
        },
        "claims": {
            "supported": True,
            "extractors": ["aps", "claimify", "auto"],
            "verifiers": ["nli", "llm", "hybrid"],
            "defaults": {
                "top_k": 5,
                "confidence_threshold": 0.7,
                "max": 25
            },
            "nli": {
                "env": ["RAG_NLI_MODEL", "RAG_NLI_MODEL_PATH"],
                "override_param": "nli_model"
            }
        },
        "semantic_cache": {
            "supported": True,
            "adaptive_thresholds": True,
            "config": RAG_SERVICE_CONFIG.get("cache", {})
        },
        "sources": {
            "supported": True,
            "datastores": ["media_db", "notes", "characters", "chats"],
        },
        "security_filtering": {
            "supported": True,
            "pii_detection": True
        },
        "citation_generation": {
            "supported": True,
            "styles": ["apa", "mla", "chicago", "harvard", "ieee"],
            "include_page_numbers": True
        },
        "guardrails": {
            "supported": True,
            "require_hard_citations": True,
            "notes": "When require_hard_citations=true and coverage<1.0, agentic path abstains with a succinct message"
        },
        "answer_generation": {
            "supported": True,
            "configurable_model": True,
            "pre_retrieval_clarification": True
        },
        "reranking": {
            "supported": True,
            "strategies": ["flashrank", "cross_encoder", "hybrid", "llama_cpp"],
            "models": [
                "flashrank",
                "cross-encoder (e.g., BAAI/bge-reranker-v2-m3, Jina reranker)",
                "GGUF via llama.cpp (e.g., Qwen3-Embedding-0.6B_f16.gguf, BGE/Jina GGUF)"
            ]
        },
        "table_processing": {
            "supported": True,
            "methods": ["markdown", "html", "hybrid"]
        },
        "vlm_late_chunking": {
            "supported": True,
            "backends": ["docling", "hf_table_transformer"],
            "parameters": [
                "enable_vlm_late_chunking",
                "vlm_backend",
                "vlm_detect_tables_only",
                "vlm_max_pages",
                "vlm_late_chunk_top_k_docs"
            ],
            "env": [
                "VLM_TABLE_MODEL_NAME",
                "VLM_TABLE_REVISION",
                "VLM_TABLE_THRESHOLD"
            ],
            "defaults": vlm_defaults,
            "backends_endpoint": "/api/v1/rag/vlm/backends",
            "note": "Env defaults reflect current process environment; Table Transformer threshold is 0.9 by default."
        },
        "enhanced_chunking": {
            "supported": True,
            "parent_context": True,
            "sibling_context": True,
            "parameters": [
                "parent_context_size",
                "include_parent_document",
                "parent_max_tokens",
                "include_sibling_chunks",
                "sibling_window",
                "chunk_type_filter"
            ]
        },
        "feedback": {
            "supported": True,
            "apply_feedback_boost": True
        },
        "monitoring": {
            "supported": True,
            "observability": True,
            "trace_id": True
        },
        "analytics": {
            "supported": True
        },
        "batch_processing": {
            "supported": True,
            "concurrent": True,
            "defaults": {"max_concurrent": 5},
            "limits": {"max_concurrent_max": 20}
        },
        "resilience": {
            "supported": True,
            "retries": True,
            "circuit_breakers": True,
            "research_action_dedup": True
        },
        "streaming": {
            "supported": True,
            "endpoint": "/api/v1/rag/search/stream",
            "media_type": "application/x-ndjson",
            "events": ["delta", "claims_overlay", "final_claims"]
        },
        "quick_wins": {
            "supported": True,
            "parameters": ["highlight_results", "highlight_query_terms", "track_cost", "debug_mode", "include_rerank_debug_documents"]
        },
        "user_context": {
            "supported": True,
            "fields": ["user_id", "session_id"]
        },
        "webui": {
            "supported": True,
            "controls": [
                "strategy",
                "agentic_enable_tools",
                "agentic_max_tool_calls",
                "agentic_max_tokens_read",
                "agentic_adaptive_budgets",
                "agentic_time_budget_sec",
                "require_hard_citations",
                "enable_numeric_fidelity",
                "agentic_enable_query_decomposition",
                "agentic_enable_vlm_late_chunking"
            ],
            "explain_panel": True,
            "highlight_spans": True,
            "section_anchors": True
        }
    }

    # Search modes and configuration ranges
    search = {
        "modes": ["hybrid", "vector", "fts"],
        "hybrid": {
            "alpha_default": RAG_SERVICE_CONFIG.get("retriever", {}).get("hybrid_alpha", 0.5),
            "alpha_range": [0.0, 1.0],
            "normalize_scores": RAG_SERVICE_CONFIG.get("retriever", {}).get("hybrid_alpha", 0.5) is not None
        },
        "vector": {
            "top_k_default": RAG_SERVICE_CONFIG.get("retriever", {}).get("vector_top_k", 10),
            "top_k_max": 100
        },
        "fts": {
            "top_k_default": RAG_SERVICE_CONFIG.get("retriever", {}).get("fts_top_k", 10),
            "query_expansion": True,
            "fuzzy_matching": True
        }
    }

    defaults = {
        "retriever": RAG_SERVICE_CONFIG.get("retriever", {}),
        "processor": RAG_SERVICE_CONFIG.get("processor", {}),
        "cache": RAG_SERVICE_CONFIG.get("cache", {}),
        "batch_size": RAG_SERVICE_CONFIG.get("batch_size", 32),
        "num_workers": RAG_SERVICE_CONFIG.get("num_workers", 4),
        "min_score": 0.0,
        "use_connection_pool": True,
        "use_embedding_cache": True
    }

    limits = {
        "top_k_max": 100,
        "documents_per_db_max": 1000,
        "answer_tokens_max": 2048,
        "timeout_seconds_max": 60.0
    }

    auth = {
        "mode": settings.AUTH_MODE,
        "user_scoped": True
    }

    quick_start = {
        "agentic_search": {
            "endpoint": "/api/v1/rag/search",
            "method": "POST",
            "body": {
                "query": "Summarize key findings of ResNet",
                "strategy": "agentic",
                "search_mode": "hybrid",
                "top_k": 8,
                "enable_generation": False,
                "agentic_enable_tools": True,
                "agentic_max_tool_calls": 6
            }
        },
        "agentic_verify": {
            "endpoint": "/api/v1/rag/search",
            "method": "POST",
            "body": {
                "query": "How many experiments were run and what supported the conclusion?",
                "strategy": "agentic",
                "enable_generation": True,
                "require_hard_citations": True,
                "enable_numeric_fidelity": True,
                "numeric_fidelity_behavior": "continue"
            }
        },
        "agentic_explain": {
            "endpoint": "/api/v1/rag/search",
            "method": "POST",
            "body": {
                "query": "Explain residual connections and dropout",
                "strategy": "agentic",
                "enable_generation": False,
                "explain_only": True,
                "agentic_enable_tools": True,
                "agentic_enable_query_decomposition": True
            }
        },
        "agentic_multihop_vlm": {
            "endpoint": "/api/v1/rag/search",
            "method": "POST",
            "body": {
                "query": "Compare accuracy tables for ResNet vs EfficientNet across datasets",
                "strategy": "agentic",
                "search_mode": "hybrid",
                "top_k": 8,
                "enable_generation": False,
                "agentic_enable_tools": True,
                "agentic_enable_query_decomposition": True,
                "agentic_subgoal_max": 3,
                "agentic_enable_vlm_late_chunking": True,
                "agentic_vlm_backend": "hf_table_transformer",
                "agentic_vlm_detect_tables_only": True,
                "agentic_vlm_late_chunk_top_k_docs": 2
            }
        },
        "ablate": {
            "endpoint": "/api/v1/rag/ablate",
            "method": "POST",
            "body": {
                "query": "How does dropout prevent overfitting?",
                "top_k": 10,
                "search_mode": "hybrid",
                "with_answer": False,
                "reranking_strategy": "none"
            }
        }
    }

    return {
        "pipeline": "unified",
        "version": "1.0.0",
        "features": features,
        "search": search,
        "defaults": defaults,
        "limits": limits,
        "auth": auth,
        "quick_start": quick_start,
    }


@router.get(
    "/vlm/backends",
    summary="VLM Backends",
    description="List VLM (Vision-Language) backends and their availability",
    response_description="Backend availability map"
)
async def list_vlm_backends():
    """
    Report available VLM backends from the ingestion registry.

    Returns a mapping like { "hf_table_transformer": {"available": true}, "docling": {"available": false} }.
    """
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.VLM.registry import list_backends as _list
        backends = _list() or {}
    except Exception:  # noqa: BLE001 - optional registry failures should not break endpoint
        backends = {}
    return {"backends": backends}


@router.get(
    "/source-health",
    response_model=KnowledgeSourceHealthResponse,
    summary="Knowledge source health",
    description="Read-only pre-query source readiness for Knowledge QA.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(rbac_rate_limit("rag.search")),
        Depends(RequirePermission(MEDIA_READ)),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="rag.search", count_as="call")),
    ],
)
async def source_health_endpoint(
    current_user: User = Depends(get_request_user),
) -> KnowledgeSourceHealthResponse:
    """Return safe pre-query readiness for canonical Knowledge QA sources."""
    existing_paths = await run_in_threadpool(_resolve_existing_source_db_paths, current_user)
    configured_sources, empty_sources = _build_source_health_source_sets(
        existing_paths=existing_paths,
        media_backend_uses_non_file_storage=_media_db_uses_non_file_storage(),
    )
    return KnowledgeSourceHealthResponse(
        sources=build_source_health_entries(
            configured_sources=configured_sources,
            empty_sources=empty_sources,
        )
    )


@router.post(
    "/search",
    response_model=UnifiedRAGResponse,
    summary="Unified RAG Search",
    description="""
    The unified RAG search endpoint with ALL features accessible via parameters.

    **Key Features:**
    - No configuration files needed
    - Every feature is a direct parameter
    - Mix and match any features
    - Transparent execution

    **Available Features:**
    - Query expansion (acronym, synonym, domain, entity)
    - Semantic caching with adaptive thresholds
    - Multi-database search (media, notes, characters, chats)
    - Security filtering and PII detection
    - Citation generation (APA, MLA, Chicago, Harvard)
    - Answer generation from context
    - Document reranking (FlashRank, Cross-Encoder, Hybrid)
    - Table processing and extraction
    - Enhanced chunking with parent context
    - User feedback collection
    - Performance monitoring and observability
    - Batch processing support
    - Resilience features (retries, circuit breakers)

    Simply set any feature parameter to enable it. All parameters are optional
    except the query itself.
    """,
    response_description="Search results with all requested features applied",
    dependencies=[
        Depends(check_rate_limit),
        Depends(rbac_rate_limit("rag.search")),
        Depends(RequirePermission(MEDIA_READ)),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="rag.search", count_as="call")),
        Depends(require_within_limit(LimitCategory.RAG_QUERIES_DAY, 1)),
    ]
)
async def unified_search_endpoint(
    request_raw: Request,
    request: UnifiedRAGRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    prompts_db: PromptsDatabase = Depends(get_prompts_db_for_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
):
    """
    Unified RAG search with all features as parameters.

    This endpoint replaces the complex configuration-based approach with
    a simple, parameter-driven interface. Every feature in the RAG system
    is accessible by setting the appropriate parameter.
    """
    try:
        request = _apply_media_collection_scope(request, collections_db)
        logger.info(f"Unified RAG search: query='{request.query}', user={current_user.username if current_user else 'anonymous'}")
        # Topic monitoring (non-blocking) for query text
        try:
            from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import get_topic_monitoring_service
            mon = get_topic_monitoring_service()
            uid = (current_user.username if current_user else request.user_id) or None
            team_ids = None
            org_ids = None
            try:
                if hasattr(request_raw, 'state'):
                    team_ids = getattr(request_raw.state, 'team_ids', None)
                    org_ids = getattr(request_raw.state, 'org_ids', None)
            except Exception as topic_context_error:  # noqa: BLE001 - topic monitoring should not break requests
                logger.debug("Topic monitoring request-state extraction failed; continuing", exc_info=topic_context_error)
            if request.query:
                mon.schedule_evaluate_and_alert(
                    user_id=str(uid) if uid else None,
                    text=request.query,
                    source="rag.search",
                    scope_type="user",
                    scope_id=str(uid) if uid else None,
                    team_ids=team_ids,
                    org_ids=org_ids,
                )
        except Exception as topic_schedule_error:  # noqa: BLE001 - topic monitoring should not break requests
            logger.debug("Topic monitoring scheduling failed; continuing", exc_info=topic_schedule_error)

        # Set up database paths
        db_paths = {
            "media_db_path": media_db.db_path if media_db else None,
            # Notes are stored in ChaChaNotes DB by design; reuse its path for notes_db
            "notes_db_path": chacha_db.db_path if chacha_db else None,
            "character_db_path": chacha_db.db_path if chacha_db else None,
            "kanban_db_path": _resolve_kanban_db_path(current_user, request.user_id),
            "prompts_db_path": getattr(prompts_db, "db_path_str", None) if prompts_db else None,
        }
        standard_bundle = _build_standard_request_bundle(
            request,
            current_user=current_user,
            db_paths=db_paths,
            media_db=media_db,
            chacha_db=chacha_db,
            prompts_db=prompts_db,
        )
        strategy_value = str(standard_bundle.resolved_request.strategy).strip().lower()

        # Branch: agentic strategy builds a synthetic chunk at query time
        if strategy_value == 'agentic':
            resolved_request = standard_bundle.resolved_request
            retrieval_plan = standard_bundle.retrieval_plan
            effective_payload, agentic_cfg = build_agentic_execution_context(
                resolved_request=resolved_request,
                retrieval_plan=retrieval_plan,
                payload_override=standard_bundle.resolved_request.payload,
            )

            try:
                result = await agentic_rag_pipeline(
                    query=resolved_request.query,
                    sources=list(retrieval_plan.sources),
                    media_db=media_db,
                    chacha_db=chacha_db,
                    media_db_path=db_paths.get("media_db_path"),
                    notes_db_path=db_paths.get("notes_db_path"),
                    character_db_path=db_paths.get("character_db_path"),
                    kanban_db_path=db_paths.get("kanban_db_path"),
                    search_mode=retrieval_plan.search_mode,
                    fts_level=effective_payload.get("fts_level", request.fts_level),
                    hybrid_alpha=effective_payload.get("hybrid_alpha", request.hybrid_alpha),
                    top_k=retrieval_plan.top_k,
                    min_score=retrieval_plan.min_score,
                    index_namespace=retrieval_plan.index_namespace,
                    agentic=agentic_cfg,
                    enable_generation=bool(effective_payload.get("enable_generation", request.enable_generation)),
                    generation_model=effective_payload.get("generation_model", request.generation_model),
                    generation_provider=effective_payload.get("generation_provider", request.generation_provider),
                    generation_prompt=effective_payload.get("generation_prompt", request.generation_prompt),
                    max_generation_tokens=int(effective_payload.get("max_generation_tokens", request.max_generation_tokens)),
                    enable_citations=bool(effective_payload.get("enable_citations", request.enable_citations)),
                    include_chunk_citations=bool(
                        effective_payload.get("enable_chunk_citations", request.enable_chunk_citations)
                    ),
                    debug_mode=bool(effective_payload.get("debug_mode", request.debug_mode)),
                    # expose verification flags on agentic path
                    require_hard_citations=bool(effective_payload.get("require_hard_citations", False)),
                    enable_numeric_fidelity=bool(effective_payload.get("enable_numeric_fidelity", False)),
                    numeric_fidelity_behavior=str(effective_payload.get("numeric_fidelity_behavior", "continue")),
                    enable_claims=bool(effective_payload.get("enable_claims", False)),
                    claim_verifier=str(effective_payload.get("claim_verifier", "hybrid")),
                    claims_top_k=int(effective_payload.get("claims_top_k", 5) or 5),
                    claims_conf_threshold=float(effective_payload.get("claims_conf_threshold", 0.7) or 0.7),
                    claims_max=int(effective_payload.get("claims_max", 25) or 25),
                    nli_model=effective_payload.get("nli_model", None),
                    claims_concurrency=int(effective_payload.get("claims_concurrency", 8) or 8),
                    adaptive_unsupported_threshold=float(
                        effective_payload.get("adaptive_unsupported_threshold", 0.15) or 0.15
                    ),
                    low_confidence_behavior=str(effective_payload.get("low_confidence_behavior", "continue")),
                    resolved_request=resolved_request,
                    retrieval_plan=retrieval_plan,
                )
            except Exception as exc:  # noqa: BLE001 - agentic pipeline fallback must be resilient
                logger.exception("Agentic RAG pipeline failed: {}", exc)
                fallback_doc = {
                    "id": f"agentic-error:{uuid4().hex[:8]}",
                    "content": "Agentic pipeline error fallback content.",
                    "metadata": {"strategy": "agentic", "error": str(exc)},
                    "score": 1.0,
                }
                result = UnifiedSearchResult(
                    documents=[fallback_doc],
                    query=request.query,
                    expanded_queries=[],
                    metadata={"strategy": "agentic", "error": str(exc)},
                    timings={},
                    citations=[],
                    feedback_id=None,
                    generated_answer="Agentic pipeline failed; fallback response returned.",
                    cache_hit=False,
                    errors=[str(exc)],
                    security_report=None,
                    total_time=0.0,
                )
        else:
            # Execute unified pipeline with all parameters from request
            kwargs = dict(standard_bundle.pipeline_kwargs)
            _sync_retriever_overrides_to_pipeline()
            result = await unified_rag_pipeline(**kwargs)

        # Convert to response format
        response = rag_result_to_response(rag_result_from_unified_search_result(result))

        # Best-effort RAG query usage logging for billing/analytics.
        await _log_rag_queries_for_org(request_raw, current_user, units=1)

        # Log performance if monitoring enabled
        if request.enable_monitoring:
            logger.info(f"Query completed in {result.total_time:.3f}s - Cache hit: {result.cache_hit}")
            if request.debug_mode:
                logger.debug(f"Timings: {result.timings}")
                logger.debug(f"Metadata: {result.metadata}")

        # Handle any errors that occurred
        if result.errors and request.debug_mode:
            logger.warning(f"Errors during processing: {result.errors}")

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.exception("Unified search error: {}", e)
        detail = "Search failed due to an internal error."
        # Expose root cause only when explicitly requested by caller.
        if bool(getattr(request, "debug_mode", False)):
            detail = f"{type(e).__name__}: {e}"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        ) from e
    else:
        return response


@router.post(
    "/feedback/implicit",
    summary="Record implicit RAG feedback",
    description="Capture click/expand/copy signals from the WebUI for learning-to-rank and personalization.",
    dependencies=[Depends(check_rate_limit)]
)
async def rag_implicit_feedback(
    request: ImplicitFeedbackEvent,
    current_user: User = Depends(get_request_user),
):
    try:
        from tldw_Server_API.app.core.config import implicit_feedback_enabled
        if not implicit_feedback_enabled():
            return {"ok": True, "disabled": True}
        user_id = _resolve_implicit_feedback_user_id(request.user_id, current_user)
        collector = UnifiedFeedbackSystem()
        await collector.record_implicit_interaction(
            user_id=user_id,
            query=request.query,
            doc_id=request.doc_id,
            event_type=request.event_type,
            impression=request.impression_list or [],
            corpus=request.corpus,
            chunk_ids=request.chunk_ids or [],
            rank=request.rank,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            dwell_ms=request.dwell_ms,
        )
    except Exception as e:  # noqa: BLE001 - feedback should surface as 400
        logger.warning(f"Failed to record implicit feedback: {e}")
        raise HTTPException(status_code=400, detail="Could not record feedback") from e
    else:
        return {"ok": True}


@router.post(
    "/batch",
    response_model=UnifiedBatchResponse,
    summary="Batch RAG Search",
    description="""
    Process multiple queries concurrently using the unified pipeline.

    All parameters from the single search endpoint are available and will
    be applied to all queries in the batch.
    """,
    response_description="Batch processing results",
    dependencies=[
        Depends(check_rate_limit),
        Depends(RequirePermission(MEDIA_READ)),
    ]
)
async def unified_batch_endpoint(
    request_raw: Request,
    response: Response,
    request: UnifiedBatchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    media_db: Any = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    prompts_db: PromptsDatabase = Depends(get_prompts_db_for_user),
):
    """
    Batch processing endpoint for multiple queries.

    Processes multiple queries concurrently with the same parameters.
    """
    try:
        requested_units = len(request.queries or [])
        limit_checker = require_within_limit(LimitCategory.RAG_QUERIES_DAY, requested_units)
        org_header = request_raw.headers.get("X-TLDW-Org-Id")
        org_query = request_raw.query_params.get("org_id")
        try:
            org_header_id = int(org_header) if org_header is not None else None
        except (TypeError, ValueError):
            org_header_id = None
        try:
            org_query_id = int(org_query) if org_query is not None else None
        except (TypeError, ValueError):
            org_query_id = None

        await limit_checker(
            response=response,
            principal=principal,
            x_tldw_org_id=org_header_id,
            org_id=org_query_id,
        )

        logger.info(
            f"Batch RAG search: {requested_units} queries, "
            f"user={current_user.username if current_user else 'anonymous'}"
        )

        start_time = time.time()

        # Set up database paths
        db_paths = {
            "media_db_path": media_db.db_path if media_db else None,
            "notes_db_path": chacha_db.db_path if chacha_db else None,
            "character_db_path": chacha_db.db_path if chacha_db else None,
            "kanban_db_path": _resolve_kanban_db_path(current_user, request.user_id),
            "prompts_db_path": getattr(prompts_db, "db_path_str", None) if prompts_db else None,
        }

        batch_bundle = _build_batch_request_bundle(
            request=request,
            db_paths=db_paths,
            current_user=current_user,
        )
        kwargs = dict(batch_bundle.pipeline_kwargs)
        checkpoint_id: Optional[str] = None
        checkpoint_manager = None
        checkpoint_state = None
        if request.enable_checkpoint:
            from tldw_Server_API.app.core.RAG.rag_service.checkpoint import CheckpointManager

            checkpoint_manager = CheckpointManager()
            checkpoint_config = _sanitize_checkpoint_config_for_persistence(kwargs)
            checkpoint_config["queries"] = list(request.queries)
            checkpoint_config["max_concurrent"] = request.max_concurrent
            checkpoint_state = checkpoint_manager.create(
                "rag_batch",
                total_items=len(request.queries or []),
                config=checkpoint_config,
            )
            checkpoint_id = checkpoint_state.checkpoint_id
        # Process batch
        on_query_done = None
        saved_indices: set[int] = set()
        cp_lock = asyncio.Lock()
        if checkpoint_state is not None and checkpoint_manager is not None:
            cp_state_cell = [checkpoint_state]

            async def _on_query_done(
                query_index: int,
                query_text: str,
                result: Optional[Any],
                error: Optional[BaseException],
            ) -> None:
                status = "ok"
                errors: list[str] = []
                if error is not None:
                    status = "error"
                    errors = [str(error)]
                else:
                    result_errors = getattr(result, "errors", None)
                    if result_errors:
                        status = "error"
                        errors = [str(e) for e in result_errors]

                payload: dict[str, Any] = {
                    "query_index": int(query_index),
                    "query": query_text,
                    "status": status,
                }
                if errors:
                    payload["errors"] = errors

                try:
                    async with cp_lock:
                        cp_state_cell[0] = checkpoint_manager.save_progress(
                            cp_state_cell[0],
                            payload,
                        )
                        saved_indices.add(int(query_index))
                except Exception as cp_err:  # noqa: BLE001 - checkpointing should not fail batch
                    logger.warning(f"Checkpoint incremental save failed: {cp_err}")

            on_query_done = _on_query_done

        results = await unified_batch_pipeline(
            queries=request.queries,
            max_concurrent=request.max_concurrent,
            media_db=media_db,
            chacha_db=chacha_db,
            prompts_db=prompts_db,
            on_query_done=on_query_done,
            **kwargs
        )

        # Convert results
        responses = [rag_result_to_response(rag_result_from_unified_search_result(r)) for r in results]

        # Count successes and failures
        successful = sum(1 for r in results if not r.errors)
        failed = len(results) - successful

        total_time = time.time() - start_time

        # Each query in the batch counts as one RAG query unit.
        await _log_rag_queries_for_org(request_raw, current_user, units=requested_units)

        if checkpoint_state is not None and checkpoint_manager is not None:
            missing_results: list[dict[str, Any]] = []
            for idx, res in enumerate(results):
                if idx in saved_indices:
                    continue
                errors: list[str] = []
                if isinstance(res, BaseException):
                    errors = [str(res)]
                else:
                    result_errors = getattr(res, "errors", None)
                    if result_errors:
                        errors = [str(e) for e in result_errors]
                payload: dict[str, Any] = {
                    "query_index": int(idx),
                    "query": request.queries[idx] if idx < len(request.queries) else "",
                    "status": "error" if errors else "ok",
                }
                if errors:
                    payload["errors"] = errors
                missing_results.append(payload)
            if missing_results:
                try:
                    checkpoint_state = checkpoint_manager.save_batch_progress(
                        checkpoint_state,
                        missing_results,
                    )
                except Exception as cp_err:  # noqa: BLE001
                    logger.warning(f"Checkpoint final save failed: {cp_err}")

        return UnifiedBatchResponse(
            results=responses,
            total_queries=requested_units,
            successful=successful,
            failed=failed,
            total_time=total_time,
            checkpoint_id=checkpoint_id,
        )

    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error(f"Batch search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch search failed due to an internal error."
        ) from e


@router.get(
    "/simple",
    summary="Simple Search",
    description="""
    Simplified search endpoint for basic use cases.

    Uses sensible defaults:
    - Caching enabled
    - Reranking enabled
    - No query expansion
    """,
    response_description="Search results",
    dependencies=[
        Depends(check_rate_limit),
        Depends(RequirePermission(MEDIA_READ)),
        Depends(require_within_limit(LimitCategory.RAG_QUERIES_DAY, 1)),
    ]
)
async def simple_search_endpoint(
    request: Request,
    query: str,
    top_k: int = 10,
    sources: Optional[list[str]] = None,
    current_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    """
    Simple search for basic use cases.
    """
    try:
        try:
            _qh = hashlib.md5((query or "").encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
            logger.info(f"Simple search: query_hash={_qh} len={len(query or '')}")
        except (AttributeError, TypeError, ValueError):
            logger.info("Simple search request received")
        # Topic monitoring (non-blocking)
        try:
            from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import get_topic_monitoring_service
            mon = get_topic_monitoring_service()
            uid = str(current_user.username)
            mon.schedule_evaluate_and_alert(
                user_id=uid,
                text=query,
                source="rag.simple_search",
                scope_type="user",
                scope_id=uid,
            )
        except Exception as topic_monitoring_error:  # noqa: BLE001 - topic monitoring should not break requests
            logger.debug("Topic monitoring in simple search failed; continuing", exc_info=topic_monitoring_error)

        # Use the simple_search wrapper
        effective_sources = sources or ["media_db", "notes", "characters"]
        documents = await simple_search(
            query,
            top_k,
            sources=effective_sources,
            media_db=media_db,
            chacha_db=chacha_db,
            media_db_path=(media_db.db_path if media_db else None),
            notes_db_path=(chacha_db.db_path if chacha_db else None),
            character_db_path=(chacha_db.db_path if chacha_db else None),
            kanban_db_path=_resolve_kanban_db_path(current_user),
            user_id=_resolve_implicit_feedback_user_id(None, current_user),
        )

        # Best-effort RAG query logging (counts as a single query).
        await _log_rag_queries_for_org(request, current_user, units=1)

        normalized_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                normalized_docs.append({
                    "id": doc.get("id"),
                    "content": doc.get("content"),
                    "metadata": doc.get("metadata") or {},
                    "score": doc.get("score", 0.0),
                })
            else:
                normalized_docs.append({
                    "id": getattr(doc, "id", None),
                    "content": getattr(doc, "content", None),
                    "metadata": getattr(doc, "metadata", {}) or {},
                    "score": getattr(doc, "score", 0.0),
                })

        return {
            "query": query,
            "documents": normalized_docs,
            "count": len(normalized_docs)
        }

    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error(f"Simple search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed due to an internal error."
        ) from e


@router.post(
    "/batch/resume/{checkpoint_id}",
    response_model=UnifiedBatchResponse,
    summary="Resume interrupted batch",
    description="Resume a batch RAG operation from a checkpoint.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(RequirePermission(MEDIA_READ)),
    ],
)
async def resume_batch_endpoint(
    checkpoint_id: str,
    request_raw: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    media_db: Any = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    prompts_db: PromptsDatabase = Depends(get_prompts_db_for_user),
):
    """Resume a batch RAG operation from a previously saved checkpoint."""
    try:
        from tldw_Server_API.app.core.RAG.rag_service.checkpoint import CheckpointManager

        manager = CheckpointManager()
        checkpoint = manager.load_by_id(checkpoint_id)

        if checkpoint.is_complete:
            return UnifiedBatchResponse(
                results=[],
                total_queries=checkpoint.total_items,
                successful=checkpoint.completed_items,
                failed=0,
                total_time=0.0,
            )

        # Extract remaining queries from config
        all_queries: list[str] = checkpoint.config.get("queries", [])
        total_queries = len(all_queries)

        def _completed_indices_from_checkpoint() -> set[int]:
            indices: set[int] = set()
            for entry in checkpoint.results or []:
                if not isinstance(entry, dict):
                    continue
                status = entry.get("status")
                if status == "in_progress":
                    continue
                raw_idx = entry.get("query_index")
                if isinstance(raw_idx, str):
                    try:
                        raw_idx = int(raw_idx)
                    except (TypeError, ValueError):
                        raw_idx = None
                if isinstance(raw_idx, int) and 0 <= raw_idx < total_queries:
                    indices.add(raw_idx)
            if not indices and checkpoint.completed_items:
                count = min(checkpoint.completed_items, total_queries)
                indices.update(range(count))
            return indices

        completed_indices = _completed_indices_from_checkpoint()
        remaining_indices = [i for i in range(total_queries) if i not in completed_indices]
        remaining_queries = [all_queries[i] for i in remaining_indices]

        if not remaining_queries:
            return UnifiedBatchResponse(
                results=[],
                total_queries=checkpoint.total_items,
                successful=checkpoint.completed_items,
                failed=0,
                total_time=0.0,
            )

        max_concurrent = checkpoint.config.get("max_concurrent", 5)

        db_paths = {
            "media_db_path": media_db.db_path if media_db else None,
            "notes_db_path": chacha_db.db_path if chacha_db else None,
            "character_db_path": chacha_db.db_path if chacha_db else None,
            "kanban_db_path": _resolve_kanban_db_path(current_user, checkpoint.config.get("user_id")),
            "prompts_db_path": getattr(prompts_db, "db_path_str", None) if prompts_db else None,
        }
        resume_request = _build_resume_batch_request(
            checkpoint.config,
            remaining_queries=remaining_queries,
            max_concurrent=max_concurrent,
        )
        batch_bundle = _build_batch_request_bundle(
            request=resume_request,
            db_paths=db_paths,
            current_user=current_user,
        )
        kwargs = dict(batch_bundle.pipeline_kwargs)

        start_time = time.time()

        # Track checkpoint state for incremental saves via per-query callback
        _cp_state = [checkpoint]  # mutable cell to update from closure
        _saved_indices: set[int] = set()
        _cp_lock = asyncio.Lock()

        async def _on_query_done(
            query_index: int,
            query_text: str,
            result: Optional[Any],
            error: Optional[BaseException],
        ) -> None:
            """Save checkpoint progress incrementally per completed query."""
            status = "ok"
            errors: list[str] = []
            if error is not None:
                status = "error"
                errors = [str(error)]
            else:
                result_errors = getattr(result, "errors", None)
                if result_errors:
                    status = "error"
                    errors = [str(e) for e in result_errors]

            payload: dict[str, Any] = {
                "query_index": int(query_index),
                "query": query_text,
                "status": status,
            }
            if errors:
                payload["errors"] = errors

            try:
                async with _cp_lock:
                    _cp_state[0] = manager.save_progress(_cp_state[0], payload)
                    _saved_indices.add(int(query_index))
            except Exception as _cp_err:  # noqa: BLE001 - checkpoint should not fail request
                logger.warning(f"Checkpoint incremental save failed: {_cp_err}")

        results = await unified_batch_pipeline(
            queries=remaining_queries,
            max_concurrent=max_concurrent,
            on_query_done=_on_query_done,
            query_indices=remaining_indices,
            media_db=media_db,
            chacha_db=chacha_db,
            prompts_db=prompts_db,
            **kwargs,
        )

        responses = [rag_result_to_response(rag_result_from_unified_search_result(r)) for r in results]
        successful = sum(1 for r in results if not r.errors)
        failed = len(results) - successful
        total_time = time.time() - start_time

        # Final checkpoint update for any queries not captured by incremental saves
        missing_results: list[dict[str, Any]] = []
        for local_idx, global_idx in enumerate(remaining_indices):
            if global_idx in _saved_indices:
                continue
            res = results[local_idx] if local_idx < len(results) else None
            errors: list[str] = []
            if isinstance(res, BaseException):
                errors = [str(res)]
            else:
                result_errors = getattr(res, "errors", None)
                if result_errors:
                    errors = [str(e) for e in result_errors]
            payload: dict[str, Any] = {
                "query_index": int(global_idx),
                "query": remaining_queries[local_idx] if local_idx < len(remaining_queries) else "",
                "status": "error" if errors else "ok",
            }
            if errors:
                payload["errors"] = errors
            missing_results.append(payload)
        if missing_results:
            manager.save_batch_progress(_cp_state[0], missing_results)

        return UnifiedBatchResponse(
            results=responses,
            total_queries=len(remaining_queries),
            successful=successful,
            failed=failed,
            total_time=total_time,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint '{checkpoint_id}' not found.",
        ) from None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Batch resume error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch resume failed due to an internal error.",
        ) from e


@router.post(
    "/search/stream",
    summary="Unified RAG Streaming Search",
    description="Stream generated answer chunks with optional incremental claim overlay events (NDJSON)",
    dependencies=[
        Depends(check_rate_limit),
        Depends(RequirePermission(MEDIA_READ)),
        Depends(require_within_limit(LimitCategory.RAG_QUERIES_DAY, 1)),
    ],
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "NDJSON stream of unified RAG search events",
            "content": {"application/x-ndjson": {}},
        },
    },
)
async def unified_search_stream_endpoint(
    request_raw: Request,
    request: UnifiedRAGRequest,
    current_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    prompts_db: PromptsDatabase = Depends(get_prompts_db_for_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
):
    if not request.enable_generation:
        raise HTTPException(status_code=400, detail="enable_generation must be true for streaming.")
    request = _apply_media_collection_scope(request, collections_db)

    # Streaming search counts as a single RAG query.
    await _log_rag_queries_for_org(request_raw, current_user, units=1)

    shared_db_paths = {
        "media_db_path": media_db.db_path if media_db else None,
        "notes_db_path": chacha_db.db_path if chacha_db else None,
        "character_db_path": chacha_db.db_path if chacha_db else None,
        "kanban_db_path": None,
        "prompts_db_path": getattr(prompts_db, "db_path_str", None) if prompts_db else None,
    }
    stream_bundle = _build_standard_request_bundle(
        request,
        current_user=current_user,
        db_paths=shared_db_paths,
        media_db=media_db,
        chacha_db=chacha_db,
        prompts_db=prompts_db,
    )
    resolved_request = stream_bundle.resolved_request
    kanban_db_path = _resolve_kanban_db_path(current_user, resolved_request.user_id)
    stream_pipeline_kwargs = dict(stream_bundle.pipeline_kwargs)
    stream_pipeline_kwargs["kanban_db_path"] = kanban_db_path
    stream_pipeline_kwargs["prompts_db_path"] = getattr(prompts_db, "db_path_str", None) if prompts_db else None
    stream_pipeline_kwargs["prompts_db"] = prompts_db
    stream_pipeline_kwargs["resolved_request"] = resolved_request
    stream_pipeline_kwargs["retrieval_plan"] = stream_bundle.retrieval_plan
    request_defaults = {
        "claims_concurrency": request.claims_concurrency,
        "claims_max": request.claims_max,
        "claims_top_k": request.claims_top_k,
        "debug_mode": request.debug_mode,
        "enable_claims": request.enable_claims,
        "explain_only": getattr(request, "explain_only", False),
        "fts_level": request.fts_level,
        "generation_model": request.generation_model,
        "generation_prompt": request.generation_prompt,
        "generation_provider": request.generation_provider,
        "hybrid_alpha": request.hybrid_alpha,
        "max_generation_tokens": request.max_generation_tokens,
        "top_k": request.top_k,
    }
    stream_context = {
        **stream_pipeline_kwargs,
        "build_agentic_execution_context": build_agentic_execution_context,
        "generate_streaming_response": generate_streaming_response,
        "request_defaults": request_defaults,
        "sync_retriever_overrides": _sync_retriever_overrides_to_pipeline,
    }

    async def event_generator():
        async for event in stream_rag_events(
            resolved_request=resolved_request,
            retrieval_plan=stream_bundle.retrieval_plan,
            standard_pipeline=unified_rag_pipeline,
            agentic_pipeline=agentic_rag_pipeline,
            extra_context=stream_context,
        ):
            yield json.dumps(event) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.get(
    "/advanced",
    summary="Advanced Search",
    description="""
    Advanced search with commonly used features enabled.

    Automatically enables:
    - Query expansion
    - Citations
    - Answer generation
    - Table processing
    - Performance analysis
    """,
    response_description="Full search results with analysis",
    dependencies=[Depends(check_rate_limit), Depends(RequirePermission(MEDIA_READ))]
)
async def advanced_search_endpoint(
    request: Request,
    query: str,
    with_citations: bool = True,
    with_answer: bool = True,
    current_user: User = Depends(get_request_user),
    media_db: Any = Depends(get_media_db_for_user),
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    """
    Advanced search with common features enabled.
    """
    try:
        logger.info(f"Advanced search: query='{query}'")
        # Topic monitoring (non-blocking)
        try:
            from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import get_topic_monitoring_service
            mon = get_topic_monitoring_service()
            uid = str(current_user.username)
            mon.schedule_evaluate_and_alert(
                user_id=uid,
                text=query,
                source="rag.advanced_search",
                scope_type="user",
                scope_id=uid,
            )
        except Exception as topic_monitoring_error:  # noqa: BLE001 - topic monitoring should not break requests
            logger.debug("Topic monitoring in advanced search failed; continuing", exc_info=topic_monitoring_error)

        # Set up database paths
        db_paths = {
            "media_db_path": media_db.db_path if media_db else None,
            "character_db_path": chacha_db.db_path if chacha_db else None,
            "kanban_db_path": _resolve_kanban_db_path(current_user),
        }

        # Use the advanced_search wrapper
        result = await advanced_search(
            query=query,
            with_citations=with_citations,
            with_answer=with_answer,
            media_db=media_db,
            chacha_db=chacha_db,
            **db_paths
        )

        return rag_result_to_response(rag_result_from_unified_search_result(result))

    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error(f"Advanced search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed due to an internal error."
        ) from e


@router.get(
    "/features",
    summary="List Available Features",
    description="Get a list of all available features in the unified pipeline",
    response_description="Feature list with descriptions"
)
async def list_features():
    """
    List all available features in the unified pipeline.
    """
    features_out = {
        "query_expansion": {
                "description": "Expand queries with synonyms, acronyms, domain terms, and entities",
                "parameters": ["expand_query", "expansion_strategies", "spell_check"]
        },
        "caching": {
                "description": "Semantic caching with adaptive thresholds",
                "parameters": ["enable_cache", "cache_threshold", "adaptive_cache"]
        },
        "security": {
                "description": "PII detection, content filtering, and access control",
                "parameters": ["enable_security_filter", "detect_pii", "redact_pii", "sensitivity_level"]
        },
        "citations": {
                "description": "Generate citations in various formats",
                "parameters": ["enable_citations", "citation_style", "include_page_numbers"]
        },
        "generation": {
                "description": "Generate answers from retrieved context",
                "parameters": [
                    "enable_generation",
                    "generation_provider",
                    "generation_model",
                    "generation_prompt",
                    "enable_pre_retrieval_clarification",
                    "clarification_timeout_sec",
                ]
        },
        "reranking": {
                "description": "Rerank documents for better relevance",
                "parameters": ["enable_reranking", "reranking_strategy", "rerank_top_k"]
        },
        "feedback": {
                "description": "Collect and apply user feedback",
                "parameters": ["collect_feedback", "feedback_user_id", "apply_feedback_boost"]
        },
        "monitoring": {
                "description": "Performance monitoring and observability",
                "parameters": ["enable_monitoring", "enable_observability", "trace_id"]
        },
        "table_processing": {
                "description": "Extract and process tables from documents",
                "parameters": ["enable_table_processing", "table_method"]
        },
        "vlm_late_chunking": {
                "description": "Add VLM-derived hints (tables/images) as late chunks from PDFs",
                "parameters": [
                    "enable_vlm_late_chunking",
                    "vlm_backend",
                    "vlm_detect_tables_only",
                    "vlm_max_pages",
                    "vlm_late_chunk_top_k_docs"
                ]
        },
        "enhanced_chunking": {
                "description": "Advanced document chunking with parent context",
                "parameters": ["enable_enhanced_chunking", "chunk_type_filter", "enable_parent_expansion"]
        },
        "batch_processing": {
                "description": "Process multiple queries concurrently",
                "parameters": ["enable_batch", "batch_queries", "batch_concurrent"]
        },
        "resilience": {
                "description": "Fault tolerance with retries and circuit breakers",
                "parameters": [
                    "enable_resilience",
                    "retry_attempts",
                    "circuit_breaker",
                    "enable_research_action_dedup",
                ]
        }
    }

    # Compute totals dynamically
    total_features = len(features_out)
    total_parameters = sum(len(v.get("parameters", [])) for v in features_out.values())

    return {
        "features": features_out,
        "total_features": total_features,
        "total_parameters": total_parameters
    }


@router.get(
    "/health/simple",
    summary="Unified Health (Simple)",
    description="Lightweight health check for the unified RAG pipeline",
    response_description="Health status",
    dependencies=[Depends(check_rate_limit)]
)
async def unified_health_simple(request: Request):
    """
    Health check for the unified pipeline.
    """
    try:
        # Test basic search functionality
        test_result = await simple_search("test", top_k=1)

        return {
            "status": "healthy",
            "pipeline": "unified",
            "version": "1.0.0",
            "test_successful": len(test_result) >= 0
        }
    except Exception as e:  # noqa: BLE001 - health check should not fail unexpectedly
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "pipeline": "unified",
            "version": "1.0.0",
            "error": "AN ERROR HAS OCCURRED - RAG HEALTH CHECK FAILED - SEE SERVER LOGS",
        }
