"""MCP ``rag.*`` module backed by the unified RAG pipeline."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
    UnifiedRAGRequest,
    UnifiedRAGResponse,
)
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

_SEARCH_PROPERTIES: dict[str, Any] = {
    "query": {"type": "string", "minLength": 1, "maxLength": 20000},
    "sources": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Canonical or accepted alias source ids.",
    },
    "search_mode": {"type": "string", "enum": list(_SEARCH_MODES), "default": "hybrid"},
    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
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


def _bounded_int(value: Any, *, minimum: int, maximum: int) -> int:
    if value is None:
        return minimum
    if isinstance(value, bool):
        raise ValueError("integer value must not be boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("integer value required") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"integer value must be between {minimum} and {maximum}")
    return result


def _bounded_float(value: Any, *, minimum: float, maximum: float) -> float:
    if value is None:
        return minimum
    if isinstance(value, bool):
        raise ValueError("float value must not be boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("float value required") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"float value must be between {minimum} and {maximum}")
    return result


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
    max_documents = _bounded_int(arguments.get("max_documents", 6), minimum=0, maximum=20)
    max_content_chars = _bounded_int(arguments.get("max_content_chars", 2000), minimum=0, maximum=8000)

    payload: dict[str, Any] = {
        "query": _required_string(arguments, "query", max_length=20000),
        "sources": sources,
        "search_mode": _enum(arguments.get("search_mode", "hybrid"), _SEARCH_MODES),
        "top_k": _bounded_int(arguments.get("top_k", 10), minimum=1, maximum=50),
        "min_score": _bounded_float(arguments.get("min_score", 0.0), minimum=0.0, maximum=1.0),
        "rag_profile": _optional_enum(arguments.get("rag_profile"), _PROFILES),
        "enable_generation": tool_name == _TOOL_ANSWER,
        "enable_citations": True,
        "enable_chunk_citations": bool(arguments.get("include_chunk_citations", True)),
        "include_metadata": True,
        "include_sources": bool(arguments.get("include_documents", True)),
    }
    if tool_name == _TOOL_SEARCH:
        payload["enable_generation"] = False

    request = UnifiedRAGRequest(**payload)
    return request, {
        "sources_explicit": sources_explicit,
        "sources_requested": list(sources),
        "allow_partial": bool(arguments.get("allow_partial", False)),
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


def _compact_document(document: dict[str, Any], *, max_content_chars: int) -> dict[str, Any]:
    compacted = dict(document)
    content = compacted.get("content")
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
    documents = list(response.documents or [])
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
    if not sources_used:
        sources_used = list(request_metadata.get("sources_requested") or [])

    metadata = dict(response.metadata or {})
    metadata.update(
        {
            "sources_requested": list(request_metadata.get("sources_requested") or []),
            "sources_used": sources_used,
            "sources_unavailable": list(request_metadata.get("sources_unavailable") or []),
            "documents_truncated": len(documents) > len(selected_documents),
            "max_documents": max_documents,
            "max_content_chars": max_content_chars,
            "hard_citation_coverage": _hard_citation_coverage(response.metadata or {}),
        }
    )

    payload: dict[str, Any] = {
        "ok": not bool(response.errors),
        "mode": mode,
        "query": response.query,
        "documents": compacted_documents,
        "citations": list(response.citations or []),
        "chunk_citations": list(response.chunk_citations or []),
        "metadata": metadata,
        "timings": dict(response.timings or {}),
        "errors": list(response.errors or []),
    }
    return payload


class RagModule(BaseModule):
    """Read-only MCP tools for curated RAG search and grounded answers."""

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
        del context
        args = arguments or {}
        if tool_name == _TOOL_CAPABILITIES:
            return {"ok": True, "tools": [_TOOL_CAPABILITIES, _TOOL_SOURCE_HEALTH, _TOOL_SEARCH, _TOOL_ANSWER]}
        if tool_name == _TOOL_SOURCE_HEALTH:
            sources = _normalize_mcp_sources(args.get("sources"))
            return {"ok": True, "sources": [{"source_id": source, "status": "unknown"} for source in sources]}
        if tool_name in {_TOOL_SEARCH, _TOOL_ANSWER}:
            request, metadata = _build_mcp_rag_request(tool_name, args)
            return {
                "ok": False,
                "reason_code": "not_implemented",
                "query": request.query,
                "metadata": metadata,
            }
        return {"ok": False, "reason_code": "unknown_tool", "message": f"Unknown tool: {tool_name}"}
