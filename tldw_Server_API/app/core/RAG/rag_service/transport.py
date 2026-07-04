"""Transport-neutral RAG helper seams shared by HTTP and MCP callers."""

from __future__ import annotations

import inspect
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
    KnowledgeSourceHealthResponse,
    UnifiedBatchRequest,
    UnifiedRAGRequest,
)
from tldw_Server_API.app.core.config import RAG_SERVICE_CONFIG, get_config_value, settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
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
from tldw_Server_API.app.core.RAG.rag_service.source_health import build_source_health_entries
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    unified_batch_pipeline,
    unified_rag_pipeline,
)

SearchAgentSettingFn = Callable[[str, str], Optional[str]]
ExistingSourceDbPathsFn = Callable[..., dict[str, str]]
MediaBackendStorageFn = Callable[[], bool]

_BATCH_ROUND2_DEFAULT_FIELDS = {
    "enable_suggestions",
    "enable_structured_response",
    "enable_image_search",
    "enable_video_search",
}


def search_agent_setting(env_key: str, config_key: str) -> Optional[str]:
    """Read Search-Agent setting with env-over-config precedence."""
    env_value = os.getenv(env_key)
    if env_value is not None:
        return env_value
    try:
        return get_config_value("Search-Agent", config_key, default=None)
    except (TypeError, ValueError):
        return None


def build_unified_pipeline_kwargs(
    request: UnifiedRAGRequest,
    db_paths: dict[str, Optional[str]],
    media_db: Any,
    chacha_db: Any,
    current_user: Optional[Any],
    prompts_db: Optional[Any] = None,
    resolved_request: Optional[ResolvedRAGRequest] = None,
    retrieval_plan: Optional[RetrievalPlan] = None,
    *,
    search_agent_setting_fn: SearchAgentSettingFn = search_agent_setting,
    single_user_id_resolver: Callable[[], Any] = DatabasePaths.get_single_user_id,
) -> dict[str, Any]:
    """Translate a resolved standard request into core pipeline keyword arguments."""
    if resolved_request is None:
        resolved_request = resolve_rag_request(
            request,
            current_user=current_user,
            single_user_id_resolver=single_user_id_resolver,
            search_agent_setting_fn=search_agent_setting_fn,
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
    return {key: value for key, value in payload.items() if key in allowed}


def build_batch_pipeline_kwargs(
    request: UnifiedBatchRequest,
    db_paths: dict[str, Optional[str]],
    current_user: Optional[Any],
    resolved_request: Optional[ResolvedRAGRequest] = None,
    retrieval_plan: Optional[RetrievalPlan] = None,
    *,
    search_agent_setting_fn: SearchAgentSettingFn = search_agent_setting,
    single_user_id_resolver: Callable[[], Any] = DatabasePaths.get_single_user_id,
) -> dict[str, Any]:
    """Translate a resolved batch request into shared batch pipeline options."""
    if resolved_request is None:
        resolved_request = resolve_rag_request(
            request,
            current_user=current_user,
            single_user_id_resolver=single_user_id_resolver,
            search_agent_setting_fn=search_agent_setting_fn,
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
    signature = inspect.signature(unified_batch_pipeline)
    params = list(signature.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return payload
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in payload.items() if key in allowed}


def build_standard_request_bundle(
    request: UnifiedRAGRequest,
    *,
    current_user: Optional[Any],
    db_paths: dict[str, Optional[str]],
    media_db: Any,
    chacha_db: Any,
    prompts_db: Optional[Any] = None,
    search_agent_setting_fn: SearchAgentSettingFn = search_agent_setting,
    single_user_id_resolver: Callable[[], Any] = DatabasePaths.get_single_user_id,
) -> ResolvedRequestBundle:
    """Resolve a standard request once and attach transport-owned pipeline resources."""
    return build_request_bundle(
        request=request,
        current_user=current_user,
        resolve_request_kwargs={
            "single_user_id_resolver": single_user_id_resolver,
            "search_agent_setting_fn": search_agent_setting_fn,
        },
        pipeline_kwargs_builder=lambda *, resolved_request, retrieval_plan: build_unified_pipeline_kwargs(
            request=request,
            db_paths=db_paths,
            media_db=media_db,
            chacha_db=chacha_db,
            prompts_db=prompts_db,
            current_user=current_user,
            resolved_request=resolved_request,
            retrieval_plan=retrieval_plan,
            search_agent_setting_fn=search_agent_setting_fn,
            single_user_id_resolver=single_user_id_resolver,
        ),
    )


def resolve_source_health_user_id(current_user: Optional[Any], request_user_id: Optional[str] = None) -> Optional[str]:
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


def resolve_existing_source_db_paths(
    current_user: Optional[Any],
    request_user_id: Optional[str] = None,
) -> dict[str, str]:
    """Return existing source database paths without creating source storage."""
    user_id = resolve_source_health_user_id(current_user, request_user_id)
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


def media_db_uses_non_file_storage() -> bool:
    """Return whether Media DB search is configured for non-file content storage."""
    backend_mode_hint = (
        os.getenv("CONTENT_DB_MODE")
        or os.getenv("TLDW_CONTENT_DB_BACKEND")
        or str(settings.get("CONTENT_DB_BACKEND", "sqlite"))
    )
    return backend_mode_hint.strip().lower() in {"postgres", "postgresql"}


def build_source_health_source_sets(
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
        configured.update({"notes", "chats", "characters", "world_books", "dictionaries"})
    else:
        empty.update({"notes", "chats", "characters", "world_books", "dictionaries"})
    if "prompts_db" in existing_paths:
        configured.add("prompts")
    else:
        empty.add("prompts")
    if "kanban_db" in existing_paths:
        configured.add("kanban")
    else:
        empty.add("kanban")
    return configured, empty


def build_source_health_payload(
    *,
    current_user: Optional[Any],
    request_user_id: Optional[str] = None,
    existing_source_db_paths_fn: ExistingSourceDbPathsFn = resolve_existing_source_db_paths,
    media_db_uses_non_file_storage_fn: MediaBackendStorageFn = media_db_uses_non_file_storage,
) -> KnowledgeSourceHealthResponse:
    """Build a safe source-health response without opening retrievers or source databases."""
    existing_paths = existing_source_db_paths_fn(current_user, request_user_id)
    configured_sources, empty_sources = build_source_health_source_sets(
        existing_paths=existing_paths,
        media_backend_uses_non_file_storage=media_db_uses_non_file_storage_fn(),
    )
    return KnowledgeSourceHealthResponse(
        sources=build_source_health_entries(
            configured_sources=configured_sources,
            empty_sources=empty_sources,
        )
    )


def _first_positive_int(*values: Any) -> int | None:
    """Return the first positive integer value from a sequence of loose inputs."""
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (list, tuple)) and value:
            nested = _first_positive_int(*value)
            if nested is not None:
                return nested
            continue
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            continue
        if candidate > 0:
            return candidate
    return None


def _allow_orgless_rag_billing_access() -> bool:
    """Return whether RAG billing checks may pass without org context."""
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime, is_test_mode

        auth_mode = str(getattr(get_settings(), "AUTH_MODE", "") or "").strip().lower()
        if auth_mode == "single_user":
            return True
        return bool(is_test_mode() or is_explicit_pytest_runtime())
    except Exception:  # noqa: BLE001 - fail closed if the auth mode cannot be resolved.
        return False


def _membership_is_active(membership: dict[str, Any]) -> bool:
    """Return whether an org membership row grants active access."""
    return str(membership.get("status") or "").strip().lower() == "active"


async def _verified_org_ids_for_user(current_user: Optional[Any]) -> list[int]:
    """Return active organization ids for a user, or an empty list if unavailable."""
    if not current_user or getattr(current_user, "id_int", None) is None:
        return []
    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

        pool = await get_db_pool()
        repo = AuthnzOrgsTeamsRepo(db_pool=pool)
        memberships = await repo.list_org_memberships_for_user(current_user.id_int)
    except Exception:  # noqa: BLE001 - callers decide whether missing org context is fatal.
        return []

    org_ids: list[int] = []
    for membership in memberships:
        if not _membership_is_active(membership):
            continue
        org_id = _first_positive_int(membership.get("org_id"))
        if org_id is not None and org_id not in org_ids:
            org_ids.append(org_id)
    return org_ids


async def resolve_org_id_for_rag_context(
    *,
    request_like: Any = None,
    current_user: Optional[Any] = None,
) -> int | None:
    """Resolve a billing organization from trusted state or verified user memberships."""
    state = getattr(request_like, "state", None)
    if state is not None:
        org_id = _first_positive_int(
            getattr(state, "org_id", None),
            getattr(state, "org_ids", None),
        )
        if org_id is not None:
            return org_id

    metadata = getattr(request_like, "metadata", None)
    hinted_org_id = None
    if isinstance(metadata, dict):
        hinted_org_id = _first_positive_int(metadata.get("org_id"), metadata.get("org_ids"))

    hinted_org_id = hinted_org_id or _first_positive_int(getattr(request_like, "org_id", None))
    verified_org_ids = await _verified_org_ids_for_user(current_user)
    if not verified_org_ids:
        return None

    if hinted_org_id in verified_org_ids:
        return hinted_org_id

    return verified_org_ids[0]


async def enforce_rag_query_limit_for_org_context(
    *,
    request_like: Any = None,
    current_user: Optional[Any] = None,
    units: int = 1,
) -> None:
    """Enforce the shared RAG daily-query billing limit outside FastAPI DI."""
    if units <= 0:
        return

    from tldw_Server_API.app.core.Billing.enforcement import (
        LimitCategory,
        enforcement_enabled,
        get_billing_enforcer,
    )

    if not enforcement_enabled():
        return

    org_id = await resolve_org_id_for_rag_context(request_like=request_like, current_user=current_user)
    if org_id is None:
        if _allow_orgless_rag_billing_access():
            return
        raise PermissionError("An active organization context is required for billing enforcement")

    result = await get_billing_enforcer().check_limit(
        org_id,
        LimitCategory.RAG_QUERIES_DAY,
        requested_units=units,
    )
    if result.should_block:
        raise PermissionError(result.message or f"Limit exceeded for {LimitCategory.RAG_QUERIES_DAY.value}")


async def log_rag_queries_for_org_context(
    *,
    request_like: Any = None,
    current_user: Optional[Any] = None,
    units: int = 1,
) -> None:
    """Best-effort RAG query usage logger for an HTTP/MCP request context."""
    if units <= 0:
        return
    try:
        org_id = await resolve_org_id_for_rag_context(request_like=request_like, current_user=current_user)
        if org_id is None:
            return

        from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (
            LedgerEntry,
            ResourceDailyLedger,
        )

        ledger = ResourceDailyLedger()
        await ledger.initialize()
        await ledger.add(
            LedgerEntry(  # type: ignore[call-arg]
                entity_scope="org",
                entity_value=str(org_id),
                category="rag_queries",
                units=int(units),
                op_id=f"rag:{org_id}:{uuid4()}",
                occurred_at=datetime.now(timezone.utc),
            )
        )
    except Exception:  # noqa: BLE001 - ledger failures must not impact callers.
        logger.debug("RAG query logging failed; continuing without usage record", exc_info=True)


def build_rag_capabilities_payload() -> dict[str, Any]:
    """Return the curated RAG capabilities summary shared by HTTP and MCP."""
    return {
        "features": {
            "agentic_chunking": {
                "supported": True,
                "strategies": ["standard", "agentic"],
            },
            "sources": {
                "supported": True,
                "datastores": [
                    "media_db",
                    "notes",
                    "chats",
                    "characters",
                    "kanban",
                    "prompts",
                    "world_books",
                    "dictionaries",
                ],
            },
            "citation_generation": {
                "supported": True,
                "styles": ["apa", "mla", "chicago", "harvard", "ieee"],
            },
            "answer_generation": {
                "supported": True,
                "configurable_model": True,
            },
            "reranking": {
                "supported": True,
                "strategies": ["flashrank", "cross_encoder", "hybrid", "llama_cpp"],
            },
        },
        "defaults": {
            "sources": ["media_db"],
            "search_mode": "hybrid",
            "top_k": 10,
            "rag_service": RAG_SERVICE_CONFIG,
        },
        "limits": {
            "query_max_length": 20000,
            "top_k_max": 100,
        },
    }
