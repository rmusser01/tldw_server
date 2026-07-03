from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.rag_module import (
    RagModule,
    _build_mcp_rag_request,
    _compact_rag_response,
)


def test_mcp_sources_accept_aliases_but_return_canonical_ids() -> None:
    request, metadata = _build_mcp_rag_request(
        "rag.search",
        {"query": "q", "sources": ["media", "notes"]},
    )

    assert request.sources == ["media_db", "notes"]  # nosec B101
    assert metadata["sources_explicit"] is True  # nosec B101
    assert metadata["sources_requested"] == ["media_db", "notes"]  # nosec B101


def test_mcp_sources_omitted_tracks_implicit_default() -> None:
    request, metadata = _build_mcp_rag_request("rag.search", {"query": "q"})

    assert request.sources == ["media_db"]  # nosec B101
    assert metadata["sources_explicit"] is False  # nosec B101


def test_advanced_is_rejected() -> None:
    with pytest.raises(ValueError, match="advanced"):
        _build_mcp_rag_request("rag.search", {"query": "q", "advanced": {"debug_mode": True}})


def test_sql_source_is_rejected_in_stage_one() -> None:
    with pytest.raises(ValueError, match="sql"):
        _build_mcp_rag_request("rag.search", {"query": "q", "sources": ["sql"]})


def test_unknown_and_internal_sources_are_rejected() -> None:
    for source in ("unknown", "claims"):
        with pytest.raises(ValueError, match=source):
            _build_mcp_rag_request("rag.search", {"query": "q", "sources": [source]})


def test_compact_response_truncates_documents_and_preserves_citations() -> None:
    response = UnifiedRAGResponse(
        query="q",
        documents=[{"id": "d1", "content": "abcdef", "metadata": {"source": "media_db"}, "score": 0.9}],
        citations=[{"id": "c1"}],
        chunk_citations=[{"id": "chunk-1"}],
        metadata={
            "hard_citations": {"coverage": 0.5},
            "knowledge_trust": {"state": "grounded"},
        },
    )

    payload = _compact_rag_response(
        response,
        mode="search",
        request_metadata={"sources_requested": ["media_db"], "sources_explicit": True},
        max_documents=1,
        max_content_chars=3,
    )

    assert payload["documents"][0]["content"] == "abc"  # nosec B101
    assert payload["documents"][0]["content_truncated"] is True  # nosec B101
    assert payload["chunk_citations"] == [{"id": "chunk-1"}]  # nosec B101
    assert payload["metadata"]["hard_citation_coverage"] == 0.5  # nosec B101
    assert payload["metadata"]["sources_used"] == ["media_db"]  # nosec B101
    assert payload["metadata"]["sources_unavailable"] == []  # nosec B101
    assert payload["metadata"]["documents_truncated"] is False  # nosec B101
    assert payload["metadata"]["max_documents"] == 1  # nosec B101
    assert payload["metadata"]["max_content_chars"] == 3  # nosec B101


@pytest.mark.asyncio
async def test_rag_module_exposes_four_strict_tools() -> None:
    module = RagModule(ModuleConfig(name="rag"))

    tools = {tool["name"]: tool for tool in await module.get_tools()}

    assert set(tools) == {"rag.capabilities", "rag.source_health", "rag.search", "rag.answer"}  # nosec B101
    for tool_name in ("rag.capabilities", "rag.source_health", "rag.search", "rag.answer"):
        assert tools[tool_name]["inputSchema"]["additionalProperties"] is False  # nosec B101
    assert tools["rag.answer"]["metadata"]["category"] == "rag_generation"  # nosec B101
