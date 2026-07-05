"""MCP ``rag.*`` module backed by the unified RAG pipeline."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
    UnifiedRAGRequest,
    UnifiedRAGResponse,
)
from tldw_Server_API.app.core.RAG.rag_service import transport as rag_transport
from tldw_Server_API.app.core.RAG.rag_service.response_mapping import (
    rag_result_from_unified_search_result,
    rag_result_to_response,
)
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.core.Text2SQL.source_registry import normalize_sources_public

from ..base import BaseModule, create_tool_definition

_TOOL_CAPABILITIES = "rag.capabilities"
_TOOL_SOURCE_HEALTH = "rag.source_health"
_TOOL_SEARCH = "rag.search"
_TOOL_ANSWER = "rag.answer"

_CANONICAL_PUBLIC_SOURCES = (
    "media_db",
    "notes",
    "chats",
    "characters",
    "kanban",
    "prompts",
    "world_books",
    "dictionaries",
)
_DEFERRED_STAGE_ONE_SOURCES = frozenset({"sql"})
_SEARCH_MODES = ("hybrid", "vector", "fts")
_PROFILES = ("fast", "balanced", "accuracy")
_MCP_TOP_K_MAX = 50
_MAX_CITATIONS = 20

_EXTERNAL_RESEARCH_DISABLED_OVERRIDES: dict[str, Any] = {
    "search_depth_mode": None,
    "enable_query_classification": False,
    "enable_research_loop": False,
    "search_url_scraping": False,
    "enable_image_search": False,
    "enable_video_search": False,
    "enable_discussion_search": False,
    "enable_web_fallback": False,
    "web_fallback_enabled": False,
}

_UNSUPPORTED_SOURCE_SCOPES: dict[str, tuple[str, ...]] = {
    "conversation_id": ("chats",),
    "conversation_ids": ("chats",),
    "character_id": ("characters", "chats"),
    "character_ids": ("characters", "chats"),
    "prompt_id": ("prompts",),
    "prompt_ids": ("prompts",),
}
_SOURCE_PERMISSION_TOOLS: dict[str, str] = {
    "media_db": "media.search",
    "notes": "notes.search",
    "chats": "chats.search",
    "characters": "characters.search",
    "kanban": "kanban.cards.search",
    "prompts": "prompts.search",
    "world_books": "characters.search",
    "dictionaries": "characters.search",
}
_SAFE_DOCUMENT_KEYS = frozenset(
    {
        "id",
        "source",
        "source_type",
        "source_id",
        "title",
        "uri",
        "url",
        "score",
        "score_type",
        "rank",
        "chunk_id",
        "document_id",
        "media_id",
        "note_id",
        "created_at",
        "updated_at",
    }
)
_SAFE_DOCUMENT_METADATA_KEYS = frozenset(
    {
        "source",
        "source_type",
        "source_id",
        "title",
        "uri",
        "url",
        "score_type",
        "chunk_id",
        "document_id",
        "media_id",
        "note_id",
        "page",
        "page_number",
        "start_time",
        "end_time",
        "timestamp",
        "created_at",
        "updated_at",
    }
)
_SAFE_CITATION_KEYS = frozenset(
    {
        "id",
        "source",
        "source_type",
        "source_id",
        "document_id",
        "chunk_id",
        "media_id",
        "note_id",
        "title",
        "uri",
        "url",
        "text",
        "quote",
        "snippet",
        "page",
        "page_number",
        "start_time",
        "end_time",
        "timestamp",
        "score",
    }
)


@dataclass(frozen=True, slots=True)
class _SourceAuthorization:
    """Result of per-source MCP authorization for a RAG request."""

    sources: list[str]
    sources_unavailable: list[str]
    warnings: list[str]


class _McpRagControls:
    """Injectable MCP control hooks used by the RAG module.

    The protocol layer performs the primary module/tool execution checks before
    calling modules. These hooks reuse that server-owned protocol state for
    direct module invocation and per-source RAG authorization.
    """

    async def enforce_protocol_rate(self, context: Any | None, tool_name: str, category: str) -> None:
        """Apply MCP protocol rate limits for a RAG tool call."""
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("rg_ingress_enforced"):
            return
        limiter = getattr(self._protocol(), "rate_limiter", None)
        if limiter is None:
            from ...auth.rate_limiter import get_rate_limiter

            limiter = get_rate_limiter()
        key = self._rate_limit_key(context, tool_name)
        if key is None:
            return
        try:
            await limiter.check_rate_limit(key, category=category)
        except Exception as exc:  # noqa: BLE001 - normalizes host limiter exceptions for MCP payloads.
            from ...auth.rate_limiter import RateLimitExceeded

            if isinstance(exc, RateLimitExceeded):
                raise PermissionError(f"MCP rate limit exceeded; retry after {exc.retry_after}s") from exc
            raise

    async def enforce_rag_rbac_rate_limit(self, context: Any | None, resource: str) -> None:
        """Require read access to the RAG protocol resource."""
        await self._ensure_tool_allowed(context, resource)

    async def require_mcp_rag_read_scope(self, context: Any | None, tool_name: str) -> None:
        """Require the caller's MCP scopes to allow the requested RAG tool."""
        await self._ensure_tool_allowed(context, tool_name)

    async def authorize_sources(
        self,
        context: Any | None,
        sources: list[str],
        *,
        sources_explicit: bool,
        allow_partial: bool,
    ) -> _SourceAuthorization:
        """Filter requested RAG sources by installed source modules and permissions."""
        del sources_explicit, allow_partial
        allowed_sources: list[str] = []
        unavailable_sources: list[str] = []
        warnings: list[str] = []
        for source in sources:
            permission_tool = _SOURCE_PERMISSION_TOOLS.get(source)
            if permission_tool is None:
                unavailable_sources.append(source)
                warnings.append(f"{source} unavailable: no MCP source permission tool is registered")
                continue
            if not await self._source_tool_registered(permission_tool):
                unavailable_sources.append(source)
                warnings.append(f"{source} unavailable: MCP source module for {permission_tool} is not active")
                continue
            if await self._tool_allowed(context, permission_tool):
                allowed_sources.append(source)
                continue
            unavailable_sources.append(source)
            warnings.append(f"{source} unavailable: MCP tool permission denied for {permission_tool}")
        return _SourceAuthorization(
            sources=allowed_sources,
            sources_unavailable=unavailable_sources,
            warnings=warnings,
        )

    async def check_rag_query_quota(self, context: Any | None, units: int = 1) -> None:
        """Enforce MCP and billing quotas before running a RAG query."""
        if units <= 0:
            return
        await self.enforce_protocol_rate(context, "rag.query", "rag_generation")
        await rag_transport.enforce_rag_query_limit_for_org_context(
            request_like=context,
            current_user=_current_user_from_context(context),
            units=units,
        )

    async def log_rag_query_usage(self, context: Any | None, units: int = 1) -> None:
        """Record successful RAG query usage for the caller's billing context."""
        await rag_transport.log_rag_queries_for_org_context(
            request_like=context,
            current_user=_current_user_from_context(context),
            units=units,
        )

    def _protocol(self) -> Any:
        from ...server import get_mcp_server

        return get_mcp_server().protocol

    @staticmethod
    def _rate_limit_key(context: Any | None, tool_name: str) -> str | None:
        user_id = getattr(context, "user_id", None)
        if user_id:
            return f"user:{user_id}:tool:{tool_name}"
        client_id = getattr(context, "client_id", None)
        if client_id:
            return f"client:{client_id}:tool:{tool_name}"
        return None

    async def _tool_allowed(self, context: Any | None, tool_name: str) -> bool:
        """Return whether protocol RBAC grants the given read-only tool."""
        if context is None or not getattr(context, "user_id", None):
            return False
        has_tool_permission = getattr(self._protocol(), "_has_tool_permission", None)
        if not callable(has_tool_permission):
            return False
        return bool(await has_tool_permission(context, tool_name, is_write=False))

    async def _source_tool_registered(self, tool_name: str) -> bool:
        """Return whether the source-owning MCP module exposes a backing tool."""
        module_registry = getattr(self._protocol(), "module_registry", None)
        find_module_for_tool = getattr(module_registry, "find_module_for_tool", None)
        if not callable(find_module_for_tool):
            return False
        result = find_module_for_tool(tool_name)
        if inspect.isawaitable(result):
            result = await result
        return result is not None

    async def _ensure_tool_allowed(self, context: Any | None, tool_name: str) -> None:
        """Raise when the caller lacks permission for a read-only MCP tool."""
        if not await self._tool_allowed(context, tool_name):
            raise PermissionError(f"MCP tool permission denied: {tool_name}")

_SEARCH_PROPERTIES: dict[str, Any] = {
    "query": {"type": "string", "minLength": 1, "maxLength": 20000},
    "sources": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Canonical or accepted alias source ids.",
    },
    "search_mode": {"type": "string", "enum": list(_SEARCH_MODES), "default": "hybrid"},
    "top_k": {"type": "integer", "minimum": 1, "maximum": _MCP_TOP_K_MAX, "default": 10},
    "min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
    "rag_profile": {"type": "string", "enum": list(_PROFILES)},
    "include_documents": {"type": "boolean", "default": True},
    "include_chunk_citations": {"type": "boolean", "default": True},
    "allow_partial": {"type": "boolean", "default": False},
    "max_documents": {"type": "integer", "minimum": 0, "maximum": 20, "default": 6},
    "max_content_chars": {"type": "integer", "minimum": 0, "maximum": 8000, "default": 2000},
}


def _sources_were_explicit(arguments: dict[str, Any]) -> bool:
    """Return whether the caller supplied the sources key before model defaults."""
    return "sources" in arguments and arguments.get("sources") is not None


def _required_string(arguments: dict[str, Any], key: str, *, max_length: int) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    value = value.strip()
    if len(value) > max_length:
        raise ValueError(f"{key} exceeds maximum length")
    return value


def _bounded_int(value: Any, *, minimum: int, maximum: int, default: int | None = None) -> int:
    """Parse an integer argument using the schema default for explicit null."""
    if value is None:
        return minimum if default is None else default
    if isinstance(value, bool):
        raise ValueError("integer value must not be boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("integer value required") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"integer value must be between {minimum} and {maximum}")
    return result


def _bounded_float(value: Any, *, minimum: float, maximum: float, default: float | None = None) -> float:
    """Parse a float argument using the schema default for explicit null."""
    if value is None:
        return minimum if default is None else default
    if isinstance(value, bool):
        raise ValueError("float value must not be boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("float value required") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"float value must be between {minimum} and {maximum}")
    return result


def _bounded_bool(value: Any, *, default: bool) -> bool:
    """Parse a boolean argument using the schema default for explicit null."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("boolean value required")
    return value


def _enum(value: Any, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise ValueError("enum value must be a string")
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ValueError(f"unsupported value: {value}")
    return normalized


def _optional_enum(value: Any, allowed: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    return _enum(value, allowed)


def _normalize_mcp_sources(value: Any) -> list[str]:
    if value is None:
        return normalize_sources_public(None)
    if not isinstance(value, list):
        raise ValueError("sources must be a list of strings")
    if any(not isinstance(source, str) for source in value):
        raise ValueError("sources entries must be strings")
    sources = normalize_sources_public(value)
    deferred = sorted(set(sources) & _DEFERRED_STAGE_ONE_SOURCES)
    if deferred:
        raise ValueError(f"unsupported Stage 1 source: {deferred[0]}")
    disallowed = sorted(set(sources) - set(_CANONICAL_PUBLIC_SOURCES))
    if disallowed:
        raise ValueError(f"unsupported Stage 1 source: {disallowed[0]}")
    return sources


def _build_mcp_rag_request(
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[UnifiedRAGRequest, dict[str, Any]]:
    """Map strict MCP arguments into a unified RAG request plus MCP metadata."""
    if tool_name not in {_TOOL_SEARCH, _TOOL_ANSWER}:
        raise ValueError(f"unsupported rag request tool: {tool_name}")
    if "advanced" in arguments:
        raise ValueError("advanced is not supported by rag.* first slice")

    sources_explicit = _sources_were_explicit(arguments)
    sources = _normalize_mcp_sources(arguments.get("sources"))
    max_documents = _bounded_int(arguments.get("max_documents", 6), minimum=0, maximum=20, default=6)
    max_content_chars = _bounded_int(
        arguments.get("max_content_chars", 2000),
        minimum=0,
        maximum=8000,
        default=2000,
    )
    include_documents = _bounded_bool(arguments.get("include_documents", True), default=True)

    payload: dict[str, Any] = {
        "query": _required_string(arguments, "query", max_length=20000),
        "sources": sources,
        "search_mode": _enum(arguments.get("search_mode", "hybrid"), _SEARCH_MODES),
        "top_k": _bounded_int(arguments.get("top_k", 10), minimum=1, maximum=_MCP_TOP_K_MAX, default=10),
        "min_score": _bounded_float(arguments.get("min_score", 0.0), minimum=0.0, maximum=1.0, default=0.0),
        "rag_profile": _optional_enum(arguments.get("rag_profile"), _PROFILES),
        "enable_generation": tool_name == _TOOL_ANSWER,
        "enable_citations": True,
        "enable_chunk_citations": _bounded_bool(arguments.get("include_chunk_citations", True), default=True),
        "include_metadata": True,
    }
    if tool_name == _TOOL_SEARCH:
        payload["enable_generation"] = False

    request = UnifiedRAGRequest(**payload)
    return request, {
        "sources_explicit": sources_explicit,
        "sources_requested": list(sources),
        "allow_partial": _bounded_bool(arguments.get("allow_partial", False), default=False),
        "include_documents": include_documents,
        "max_documents": max_documents,
        "max_content_chars": max_content_chars,
    }


def _source_from_document(document: dict[str, Any]) -> str | None:
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        raw_source = metadata.get("source") or metadata.get("source_type")
        if isinstance(raw_source, str) and raw_source.strip():
            return raw_source.strip()
    raw_source = document.get("source") or document.get("source_type")
    if isinstance(raw_source, str) and raw_source.strip():
        return raw_source.strip()
    return None


def _safe_json_value(value: Any) -> Any | None:
    """Return bounded scalar/list metadata values suitable for MCP responses."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, (list, tuple)):
        safe_items = [_safe_json_value(item) for item in value[:20]]
        return [item for item in safe_items if item is not None]
    return None


def _safe_metadata(metadata: Any, allowed_keys: frozenset[str]) -> dict[str, Any]:
    """Allowlist metadata keys and scalar values before returning to MCP clients."""
    if not isinstance(metadata, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in sorted(allowed_keys):
        if key not in metadata:
            continue
        value = _safe_json_value(metadata.get(key))
        if value is not None:
            safe[key] = value
    return safe


def _compact_citations(citations: Any) -> tuple[list[dict[str, Any]], bool]:
    """Return bounded citation dictionaries without backend/debug payloads."""
    if not isinstance(citations, list):
        return [], False
    selected = citations[:_MAX_CITATIONS]
    compacted = [
        _safe_metadata(citation, _SAFE_CITATION_KEYS)
        for citation in selected
        if isinstance(citation, dict)
    ]
    return compacted, len(citations) > len(selected)


def _compact_errors(errors: Any) -> tuple[list[dict[str, Any]], int]:
    """Return a stable client-safe error summary without backend exception text."""
    if not isinstance(errors, list) or not errors:
        return [], 0
    return [
        {
            "reason_code": "rag_pipeline_error",
            "message": "RAG pipeline returned one or more errors.",
            "count": len(errors),
        }
    ], len(errors)


def _model_to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert Pydantic v1/v2 models to plain JSON-safe dictionaries."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return dict_method(exclude_none=True)
    return dict(value) if isinstance(value, dict) else {}


def _compact_document(document: dict[str, Any], *, max_content_chars: int) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key in sorted(_SAFE_DOCUMENT_KEYS):
        if key not in document:
            continue
        value = _safe_json_value(document.get(key))
        if value is not None:
            compacted[key] = value
    safe_metadata = _safe_metadata(document.get("metadata"), _SAFE_DOCUMENT_METADATA_KEYS)
    if safe_metadata:
        compacted["metadata"] = safe_metadata
    content = document.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    truncated = max_content_chars >= 0 and len(content) > max_content_chars
    compacted["content"] = content[:max_content_chars] if truncated else content
    compacted["content_truncated"] = truncated
    return compacted


def _hard_citation_coverage(metadata: dict[str, Any]) -> float | None:
    hard_citations = metadata.get("hard_citations")
    if not isinstance(hard_citations, dict):
        return None
    coverage = hard_citations.get("coverage")
    if isinstance(coverage, (int, float)):
        return float(coverage)
    return None


def _compact_rag_response(
    response: UnifiedRAGResponse,
    *,
    mode: str,
    request_metadata: dict[str, Any],
    max_documents: int,
    max_content_chars: int,
) -> dict[str, Any]:
    """Return a bounded JSON-serializable MCP payload from a RAG response."""
    include_documents = bool(request_metadata.get("include_documents", True))
    raw_documents = list(response.documents or [])
    documents = raw_documents if include_documents else []
    selected_documents = documents[:max_documents]
    compacted_documents = [
        _compact_document(document, max_content_chars=max_content_chars)
        for document in selected_documents
        if isinstance(document, dict)
    ]
    sources_used = []
    for document in compacted_documents:
        source = _source_from_document(document)
        if source and source not in sources_used:
            sources_used.append(source)

    response_metadata = response.metadata or {}
    citations, citations_truncated = _compact_citations(response.citations or [])
    chunk_citations, chunk_citations_truncated = _compact_citations(response.chunk_citations or [])
    errors, error_count = _compact_errors(response.errors or [])
    trust = response_metadata.get("knowledge_trust")
    trust_state = str(trust.get("state", "")).strip().lower() if isinstance(trust, dict) else None
    metadata = {
        "sources_requested": list(request_metadata.get("sources_requested") or []),
        "sources_used": sources_used,
        "sources_unavailable": list(request_metadata.get("sources_unavailable") or []),
        "warnings": list(request_metadata.get("warnings") or []),
        "documents_truncated": include_documents and len(raw_documents) > len(selected_documents),
        "citations_truncated": citations_truncated,
        "chunk_citations_truncated": chunk_citations_truncated,
        "error_count": error_count,
        "max_documents": max_documents,
        "max_content_chars": max_content_chars,
        "hard_citation_coverage": _hard_citation_coverage(response_metadata),
    }
    if trust_state:
        metadata["knowledge_trust_state"] = trust_state

    payload: dict[str, Any] = {
        "ok": error_count == 0,
        "mode": mode,
        "query": response.query,
        "documents": compacted_documents,
        "citations": citations,
        "chunk_citations": chunk_citations,
        "metadata": metadata,
        "timings": dict(response.timings or {}),
        "errors": errors,
    }
    if mode == "answer":
        payload["answer"] = {
            "text": _answer_text(response.generated_answer),
            "status": _answer_status(response),
        }
    return payload


def _answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text") or value.get("answer") or value.get("content")
        if text is not None:
            return str(text)
    return "" if value is None else str(value)


def _answer_status(response: UnifiedRAGResponse) -> str:
    """Classify generated answer grounding for MCP clients."""
    answer = _answer_text(response.generated_answer).strip()
    if not answer:
        return "abstained"
    citations = list(response.citations or []) + list(response.chunk_citations or [])
    coverage = _hard_citation_coverage(response.metadata or {})
    trust = (response.metadata or {}).get("knowledge_trust")
    trust_state = str(trust.get("state", "")).strip().lower() if isinstance(trust, dict) else ""
    if citations and (coverage is None or coverage >= 0.99) and trust_state not in {"weak", "uncited", "unsupported"}:
        return "answered"
    return "partial"


def _mcp_safe_search_agent_overrides() -> dict[str, Any]:
    """Force Stage 1 MCP RAG away from external research/web provider paths."""
    return dict(_EXTERNAL_RESEARCH_DISABLED_OVERRIDES)


def _copy_rag_request_with_updates(request: UnifiedRAGRequest, updates: dict[str, Any]) -> UnifiedRAGRequest:
    model_copy = getattr(request, "model_copy", None)
    if callable(model_copy):
        return model_copy(update=updates)
    return request.copy(update=updates)


def _list_from_scalar_or_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _metadata_values(metadata: dict[str, Any], *keys: str) -> list[Any]:
    values: list[Any] = []
    for key in keys:
        values.extend(_list_from_scalar_or_list(metadata.get(key)))
    return values


def _normalize_media_ids(values: list[Any]) -> list[int]:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("media_id scope must be an integer")
        try:
            media_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("media_id scope must be an integer") from exc
        if media_id <= 0:
            raise ValueError("media_id scope must be positive")
        if media_id not in normalized:
            normalized.append(media_id)
    return normalized


def _normalize_note_ids(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        raw = str(value).strip()
        if not raw:
            raise ValueError("note_id scope must not be empty")
        if raw not in normalized:
            normalized.append(raw)
    return normalized


def _intersect_or_apply(existing: list[Any] | None, scoped: list[Any]) -> list[Any]:
    if existing is None:
        return list(scoped)
    scoped_set = {str(item) for item in scoped}
    return [item for item in existing if str(item) in scoped_set]


def _apply_supported_source_scopes(
    request: UnifiedRAGRequest,
    context: Any | None,
) -> UnifiedRAGRequest:
    """Apply enforceable item/workspace scopes before retrieval."""
    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    updates: dict[str, Any] = {}

    media_values = _metadata_values(metadata, "media_id", "media_ids")
    if media_values:
        scoped_media_ids = _normalize_media_ids(media_values)
        updates["include_media_ids"] = _intersect_or_apply(request.include_media_ids, scoped_media_ids)

    note_values = _metadata_values(metadata, "note_id", "note_ids")
    if note_values:
        scoped_note_ids = _normalize_note_ids(note_values)
        updates["include_note_ids"] = _intersect_or_apply(request.include_note_ids, scoped_note_ids)

    workspace_id = metadata.get("workspace_id")
    if isinstance(workspace_id, str) and workspace_id.strip():
        updates["workspace_id"] = workspace_id.strip()

    if not updates:
        return request
    return _copy_rag_request_with_updates(request, updates)


def _unsupported_scope_warnings_or_error(
    request: UnifiedRAGRequest,
    request_metadata: dict[str, Any],
    context: Any | None,
) -> tuple[UnifiedRAGRequest | None, dict[str, Any] | None, list[str]]:
    """Fail or filter sources whose scoped retrieval is not enforceable in Stage 1."""
    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        return request, None, []
    active_scopes = [
        key
        for key in _UNSUPPORTED_SOURCE_SCOPES
        if _list_from_scalar_or_list(metadata.get(key))
    ]
    if not active_scopes:
        return request, None, []

    requested_sources = set(request.sources or [])
    affected_sources: set[str] = set()
    for key in active_scopes:
        affected_sources.update(_UNSUPPORTED_SOURCE_SCOPES[key])
    affected_requested = sorted(requested_sources & affected_sources)
    if not affected_requested:
        return request, None, []

    if request_metadata.get("sources_explicit", False):
        return None, {
            "ok": False,
            "reason_code": "unsupported_scope",
            "message": "Requested scope cannot be enforced for one or more Stage 1 RAG sources.",
            "scopes": active_scopes,
            "sources": affected_requested,
        }, []

    filtered_sources = [source for source in (request.sources or []) if source not in affected_sources]
    warnings = [
        f"filtered unsupported scoped source {source}"
        for source in affected_requested
    ]
    return _copy_rag_request_with_updates(request, {"sources": filtered_sources}), None, warnings


def _db_paths_from_context(context: Any | None) -> dict[str, str | None]:
    raw_paths = getattr(context, "db_paths", None)
    paths = raw_paths if isinstance(raw_paths, dict) else {}
    chacha_path = paths.get("chacha") or paths.get("chacha_db") or paths.get("notes") or paths.get("notes_db_path")
    return {
        "media_db_path": paths.get("media_db_path") or paths.get("media") or paths.get("media_db"),
        "notes_db_path": chacha_path,
        "character_db_path": paths.get("character_db_path") or paths.get("characters") or chacha_path,
        "kanban_db_path": paths.get("kanban_db_path") or paths.get("kanban"),
        "prompts_db_path": paths.get("prompts_db_path") or paths.get("prompts"),
    }


def _source_health_existing_paths_from_context(context: Any | None) -> dict[str, str]:
    """Map MCP context DB paths to source-health storage keys."""
    db_paths = _db_paths_from_context(context)
    source_paths = {
        "media_db": db_paths.get("media_db_path"),
        "chacha_db": db_paths.get("notes_db_path"),
        "prompts_db": db_paths.get("prompts_db_path"),
        "kanban_db": db_paths.get("kanban_db_path"),
    }
    return {
        source_key: str(path)
        for source_key, path in source_paths.items()
        if path
    }


def _current_user_from_context(context: Any | None) -> Any | None:
    user_id = getattr(context, "user_id", None)
    if user_id is None:
        return None
    id_int = None
    try:
        id_int = int(user_id)
    except (TypeError, ValueError):
        id_int = None
    return SimpleNamespace(id=user_id, id_int=id_int, username=str(user_id))


class RagModule(BaseModule):
    """Read-only MCP tools for curated RAG search and grounded answers."""

    def __init__(
        self,
        config: Any,
        *,
        controls: _McpRagControls | None = None,
    ) -> None:
        super().__init__(config)
        self._controls = controls or _McpRagControls()

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        tools = [
            create_tool_definition(
                name=_TOOL_CAPABILITIES,
                description="Describe the curated rag.* MCP tool surface and supported sources.",
                parameters={"properties": {}, "required": []},
                metadata={"category": "utility", "readOnlyHint": True},
            ),
            create_tool_definition(
                name=_TOOL_SOURCE_HEALTH,
                description="Report safe readiness details for RAG sources available to this context.",
                parameters={
                    "properties": {
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "allow_partial": {"type": "boolean", "default": False},
                    },
                    "required": [],
                },
                metadata={"category": "search", "readOnlyHint": True},
            ),
            create_tool_definition(
                name=_TOOL_SEARCH,
                description="Run retrieval over configured local knowledge sources without answer generation.",
                parameters={"properties": dict(_SEARCH_PROPERTIES), "required": ["query"]},
                metadata={"category": "search", "readOnlyHint": True},
            ),
            create_tool_definition(
                name=_TOOL_ANSWER,
                description="Run retrieval and return a grounded answer with citation-aware status metadata.",
                parameters={"properties": dict(_SEARCH_PROPERTIES), "required": ["query"]},
                metadata={"category": "rag_generation", "readOnlyHint": True},
            ),
        ]
        for tool in tools:
            tool["inputSchema"]["additionalProperties"] = False
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        args = arguments or {}
        if tool_name == _TOOL_CAPABILITIES:
            await self._controls.enforce_protocol_rate(context, tool_name, "utility")
            payload = rag_transport.build_rag_capabilities_payload()
            payload["limits"] = {**payload.get("limits", {}), "top_k_max": _MCP_TOP_K_MAX}
            return {"ok": True, **payload}
        if tool_name == _TOOL_SOURCE_HEALTH:
            await self._controls.enforce_protocol_rate(context, tool_name, "search")
            await self._controls.enforce_rag_rbac_rate_limit(context, "rag.search")
            await self._controls.require_mcp_rag_read_scope(context, tool_name)
            sources_explicit = _sources_were_explicit(args)
            allow_partial = _bounded_bool(args.get("allow_partial", False), default=False)
            requested_sources = (
                _normalize_mcp_sources(args.get("sources"))
                if args.get("sources") is not None
                else list(_CANONICAL_PUBLIC_SOURCES)
            )
            authorization = await self._controls.authorize_sources(
                context,
                requested_sources,
                sources_explicit=sources_explicit,
                allow_partial=allow_partial,
            )
            if authorization.sources_unavailable and sources_explicit and not allow_partial:
                return {
                    "ok": False,
                    "reason_code": "source_unavailable",
                    "sources_requested": requested_sources,
                    "sources_unavailable": authorization.sources_unavailable,
                    "warnings": authorization.warnings,
                }
            current_user = _current_user_from_context(context)
            existing_source_paths = _source_health_existing_paths_from_context(context)
            if existing_source_paths:
                def _context_existing_paths(
                    _current_user: Any | None = None,
                    _request_user_id: str | None = None,
                ) -> dict[str, str]:
                    return dict(existing_source_paths)

                payload = rag_transport.build_source_health_payload(
                    current_user=current_user,
                    existing_source_db_paths_fn=_context_existing_paths,
                )
            else:
                payload = rag_transport.build_source_health_payload(current_user=current_user)
            visible_sources = [
                entry
                for entry in payload.sources
                if entry.source_id in set(authorization.sources)
            ]
            return {
                "ok": True,
                "sources": [_model_to_plain_dict(entry) for entry in visible_sources],
                "sources_requested": requested_sources,
                "sources_unavailable": list(authorization.sources_unavailable),
                "warnings": list(authorization.warnings),
                "metadata": {
                    "sources_explicit": sources_explicit,
                    "allow_partial": allow_partial,
                },
            }
        if tool_name in {_TOOL_SEARCH, _TOOL_ANSWER}:
            return await self._execute_rag_query(tool_name, args, context)
        return {"ok": False, "reason_code": "unknown_tool", "message": f"Unknown tool: {tool_name}"}

    async def _execute_rag_query(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None,
    ) -> dict[str, Any]:
        category = "rag_generation" if tool_name == _TOOL_ANSWER else "search"
        await self._controls.enforce_protocol_rate(context, tool_name, category)
        try:
            request, request_metadata = _build_mcp_rag_request(tool_name, arguments)
            request = _apply_supported_source_scopes(request, context)
            request, scope_error, scope_warnings = _unsupported_scope_warnings_or_error(
                request,
                request_metadata,
                context,
            )
            if scope_error is not None:
                scope_error["mode"] = "answer" if tool_name == _TOOL_ANSWER else "search"
                return scope_error
            if scope_warnings:
                request_metadata["warnings"] = list(scope_warnings)
        except ValueError as exc:
            return {"ok": False, "reason_code": "invalid_arguments", "message": str(exc)}

        mode = "answer" if tool_name == _TOOL_ANSWER else "search"
        try:
            await self._controls.enforce_rag_rbac_rate_limit(context, "rag.search")
            await self._controls.require_mcp_rag_read_scope(context, tool_name)
            authorization = await self._controls.authorize_sources(
                context,
                list(request.sources or []),
                sources_explicit=bool(request_metadata.get("sources_explicit", False)),
                allow_partial=bool(request_metadata.get("allow_partial", False)),
            )
            if authorization.sources_unavailable and not request_metadata.get("allow_partial", False):
                return {
                    "ok": False,
                    "mode": mode,
                    "reason_code": "source_unavailable",
                    "sources_unavailable": authorization.sources_unavailable,
                    "warnings": authorization.warnings,
                }
            request_metadata["sources_unavailable"] = list(authorization.sources_unavailable)
            request_metadata["warnings"] = list(request_metadata.get("warnings") or []) + list(authorization.warnings)
            if not authorization.sources:
                return {
                    "ok": False,
                    "mode": mode,
                    "reason_code": "source_unavailable",
                    "message": "No authorized RAG sources are available for this request.",
                    "sources_requested": list(request_metadata.get("sources_requested") or []),
                    "sources_unavailable": list(authorization.sources_unavailable),
                    "warnings": list(request_metadata.get("warnings") or []),
                }
            request = _copy_rag_request_with_updates(request, {"sources": list(authorization.sources)})
            await self._controls.check_rag_query_quota(context, units=1)
            bundle = rag_transport.build_standard_request_bundle(
                request=request,
                current_user=_current_user_from_context(context),
                db_paths=_db_paths_from_context(context),
                media_db=None,
                chacha_db=None,
                prompts_db=None,
            )
            pipeline_kwargs = dict(bundle.pipeline_kwargs)
            pipeline_kwargs.update(_mcp_safe_search_agent_overrides())
            pipeline_kwargs["enable_generation"] = tool_name == _TOOL_ANSWER
            pipeline_kwargs["include_sources"] = bool(request_metadata.get("include_documents", True))
            result = await unified_rag_pipeline(**pipeline_kwargs)
            response = rag_result_to_response(rag_result_from_unified_search_result(result))
            payload = _compact_rag_response(
                response,
                mode=mode,
                request_metadata=request_metadata,
                max_documents=int(request_metadata["max_documents"]),
                max_content_chars=int(request_metadata["max_content_chars"]),
            )
            if payload.get("ok") is True:
                await self._controls.log_rag_query_usage(context, units=1)
            return payload
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001 - RAG domain errors become structured MCP payloads.
            logger.exception(
                "RAG MCP pipeline failed for tool={tool_name} mode={mode}: {error_type}",
                tool_name=tool_name,
                mode=mode,
                error_type=exc.__class__.__name__,
            )
            return {
                "ok": False,
                "mode": mode,
                "reason_code": "rag_pipeline_error",
                "message": "RAG pipeline failed.",
                "error_type": exc.__class__.__name__,
            }
