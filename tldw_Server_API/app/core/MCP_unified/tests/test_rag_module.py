from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import rag_module as rag_module_impl
from tldw_Server_API.app.core.MCP_unified.modules.implementations.rag_module import (
    RagModule,
    _build_mcp_rag_request,
    _compact_rag_response,
)
from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext


class _RecordingControls:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def enforce_protocol_rate(self, context, tool_name: str, category: str):  # noqa: ANN001
        self.calls.append(("protocol_rate", tool_name, category))

    async def enforce_rag_rbac_rate_limit(self, context, resource: str):  # noqa: ANN001
        self.calls.append(("rbac_rate", resource))

    async def require_mcp_rag_read_scope(self, context, tool_name: str):  # noqa: ANN001
        self.calls.append(("read_scope", tool_name))

    async def authorize_sources(
        self,
        context,  # noqa: ANN001
        sources: list[str],
        *,
        sources_explicit: bool,
        allow_partial: bool,
    ):
        self.calls.append(("authorize_sources", tuple(sources), sources_explicit, allow_partial))
        return rag_module_impl._SourceAuthorization(sources=sources, sources_unavailable=[], warnings=[])

    async def check_rag_query_quota(self, context, units: int = 1):  # noqa: ANN001
        self.calls.append(("quota", units))

    async def log_rag_query_usage(self, context, units: int = 1):  # noqa: ANN001
        self.calls.append(("usage", units))


class _UnavailableNotesControls(_RecordingControls):
    async def authorize_sources(
        self,
        context,  # noqa: ANN001
        sources: list[str],
        *,
        sources_explicit: bool,
        allow_partial: bool,
    ):
        self.calls.append(("authorize_sources", tuple(sources), sources_explicit, allow_partial))
        allowed = [source for source in sources if source != "notes"]
        unavailable = ["notes"] if "notes" in sources else []
        return rag_module_impl._SourceAuthorization(
            sources=allowed,
            sources_unavailable=unavailable,
            warnings=["notes unavailable"] if unavailable else [],
        )


class _AllowAllRBAC:
    async def check_permission(self, *args, **kwargs):  # noqa: ANN001
        del args, kwargs
        return True


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


@pytest.mark.asyncio
async def test_rag_search_executes_shared_pipeline_without_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"))
    calls: dict[str, object] = {}

    async def fake_pipeline(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            documents=[{"id": "d1", "content": "Evidence", "metadata": {"source": "media_db"}, "score": 0.9}],
            query="q",
            metadata={"hard_citations": {"coverage": 1.0}},
            timings={"retrieval": 1.2},
            citations=[],
            chunk_citations=[],
            generated_answer=None,
            errors=[],
            total_time=1.2,
            cache_hit=False,
            expanded_queries=[],
        )

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)
    ctx = RequestContext(
        request_id="r1",
        user_id="1",
        db_paths={"media": "media.db", "chacha": "notes.db"},
    )

    out = await module.execute_tool("rag.search", {"query": "q", "sources": ["media"]}, context=ctx)

    assert out["ok"] is True  # nosec B101
    assert out["mode"] == "search"  # nosec B101
    assert "answer" not in out  # nosec B101
    assert calls["enable_generation"] is False  # nosec B101
    assert calls["sources"] == ["media_db"]  # nosec B101
    assert calls["media_db_path"] == "media.db"  # nosec B101
    assert calls["notes_db_path"] == "notes.db"  # nosec B101


@pytest.mark.asyncio
async def test_rag_answer_marks_grounded_cited_output_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"))

    async def fake_pipeline(**kwargs):  # noqa: ARG001
        return SimpleNamespace(
            documents=[{"id": "d1", "content": "Evidence", "metadata": {"source": "media_db"}, "score": 0.9}],
            query="q",
            metadata={"hard_citations": {"coverage": 1.0}, "knowledge_trust": {"state": "grounded"}},
            timings={},
            citations=[{"id": "c1"}],
            chunk_citations=[],
            generated_answer="Grounded answer.",
            errors=[],
            total_time=0.1,
            cache_hit=False,
            expanded_queries=[],
        )

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)

    out = await module.execute_tool("rag.answer", {"query": "q"}, context=RequestContext(request_id="r2", user_id="1"))

    assert out["ok"] is True  # nosec B101
    assert out["answer"]["text"] == "Grounded answer."  # nosec B101
    assert out["answer"]["status"] == "answered"  # nosec B101


@pytest.mark.asyncio
async def test_rag_answer_never_marks_uncited_output_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"))

    async def fake_pipeline(**kwargs):  # noqa: ARG001
        return SimpleNamespace(
            documents=[{"id": "d1", "content": "Weak evidence", "metadata": {"source": "media_db"}, "score": 0.4}],
            query="q",
            metadata={"hard_citations": {"coverage": 0.0}, "knowledge_trust": {"state": "weak"}},
            timings={},
            citations=[],
            chunk_citations=[],
            generated_answer="Unsupported answer.",
            errors=[],
            total_time=0.1,
            cache_hit=False,
            expanded_queries=[],
        )

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)

    out = await module.execute_tool("rag.answer", {"query": "q"}, context=RequestContext(request_id="r3", user_id="1"))

    assert out["ok"] is True  # nosec B101
    assert out["answer"]["status"] in {"partial", "abstained"}  # nosec B101
    assert out["answer"]["status"] != "answered"  # nosec B101


@pytest.mark.asyncio
async def test_rag_search_applies_supported_scopes_and_disables_external_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = RagModule(ModuleConfig(name="rag"))
    calls: dict[str, object] = {}

    async def fake_pipeline(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            documents=[],
            query="q",
            metadata={},
            timings={},
            citations=[],
            chunk_citations=[],
            generated_answer=None,
            errors=[],
            total_time=0.1,
            cache_hit=False,
            expanded_queries=[],
        )

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)
    ctx = RequestContext(
        request_id="r4",
        user_id="1",
        session_id="session-1",
        metadata={"media_ids": [1, "2"], "note_id": "note-7", "workspace_id": "ws-1"},
    )

    await module.execute_tool("rag.search", {"query": "q", "sources": ["media", "notes"]}, context=ctx)

    assert calls["include_media_ids"] == [1, 2]  # nosec B101
    assert calls["include_note_ids"] == ["note-7"]  # nosec B101
    assert calls["workspace_id"] == "ws-1"  # nosec B101
    assert calls["enable_research_loop"] is False  # nosec B101
    assert calls["search_url_scraping"] is False  # nosec B101
    assert calls["enable_image_search"] is False  # nosec B101
    assert calls["enable_video_search"] is False  # nosec B101


@pytest.mark.asyncio
async def test_rag_capabilities_uses_only_protocol_rate_control() -> None:
    controls = _RecordingControls()
    module = RagModule(ModuleConfig(name="rag"), controls=controls)

    out = await module.execute_tool("rag.capabilities", {}, context=RequestContext(request_id="r5", user_id="1"))

    assert out["ok"] is True  # nosec B101
    assert controls.calls == [("protocol_rate", "rag.capabilities", "utility")]  # nosec B101


@pytest.mark.asyncio
async def test_rag_source_health_uses_read_controls_without_query_quota() -> None:
    controls = _RecordingControls()
    module = RagModule(ModuleConfig(name="rag"), controls=controls)

    out = await module.execute_tool("rag.source_health", {}, context=RequestContext(request_id="r6", user_id="1"))

    assert hasattr(out, "sources")  # nosec B101
    assert ("protocol_rate", "rag.source_health", "search") in controls.calls  # nosec B101
    assert ("rbac_rate", "rag.search") in controls.calls  # nosec B101
    assert ("read_scope", "rag.source_health") in controls.calls  # nosec B101
    assert not any(call[0] == "quota" for call in controls.calls)  # nosec B101
    assert not any(call[0] == "usage" for call in controls.calls)  # nosec B101


@pytest.mark.asyncio
async def test_rag_search_uses_quota_and_usage_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    controls = _RecordingControls()
    module = RagModule(ModuleConfig(name="rag"), controls=controls)

    async def fake_pipeline(**kwargs):  # noqa: ARG001
        return SimpleNamespace(
            documents=[],
            query="q",
            metadata={},
            timings={},
            citations=[],
            chunk_citations=[],
            generated_answer=None,
            errors=[],
            total_time=0.1,
            cache_hit=False,
            expanded_queries=[],
        )

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)

    out = await module.execute_tool("rag.search", {"query": "q"}, context=RequestContext(request_id="r7", user_id="1"))

    assert out["ok"] is True  # nosec B101
    assert ("protocol_rate", "rag.search", "search") in controls.calls  # nosec B101
    assert ("rbac_rate", "rag.search") in controls.calls  # nosec B101
    assert ("read_scope", "rag.search") in controls.calls  # nosec B101
    assert ("quota", 1) in controls.calls  # nosec B101
    assert ("usage", 1) in controls.calls  # nosec B101


@pytest.mark.asyncio
async def test_rag_answer_uses_generation_category_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    controls = _RecordingControls()
    module = RagModule(ModuleConfig(name="rag"), controls=controls)

    async def fake_pipeline(**kwargs):  # noqa: ARG001
        return SimpleNamespace(
            documents=[],
            query="q",
            metadata={},
            timings={},
            citations=[],
            chunk_citations=[],
            generated_answer=None,
            errors=[],
            total_time=0.1,
            cache_hit=False,
            expanded_queries=[],
        )

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)

    await module.execute_tool("rag.answer", {"query": "q"}, context=RequestContext(request_id="r8", user_id="1"))

    assert ("protocol_rate", "rag.answer", "rag_generation") in controls.calls  # nosec B101
    assert ("quota", 1) in controls.calls  # nosec B101
    assert ("usage", 1) in controls.calls  # nosec B101


@pytest.mark.asyncio
async def test_explicit_unavailable_source_fails_closed() -> None:
    module = RagModule(ModuleConfig(name="rag"), controls=_UnavailableNotesControls())

    out = await module.execute_tool(
        "rag.search",
        {"query": "q", "sources": ["notes"]},
        context=RequestContext(request_id="r9", user_id="1"),
    )

    assert out["ok"] is False  # nosec B101
    assert out["reason_code"] == "source_unavailable"  # nosec B101
    assert out["sources_unavailable"] == ["notes"]  # nosec B101


@pytest.mark.asyncio
async def test_allow_partial_filters_unavailable_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"), controls=_UnavailableNotesControls())
    calls: dict[str, object] = {}

    async def fake_pipeline(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            documents=[],
            query="q",
            metadata={},
            timings={},
            citations=[],
            chunk_citations=[],
            generated_answer=None,
            errors=[],
            total_time=0.1,
            cache_hit=False,
            expanded_queries=[],
        )

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)

    out = await module.execute_tool(
        "rag.search",
        {"query": "q", "sources": ["media", "notes"], "allow_partial": True},
        context=RequestContext(request_id="r10", user_id="1"),
    )

    assert out["ok"] is True  # nosec B101
    assert calls["sources"] == ["media_db"]  # nosec B101
    assert out["metadata"]["sources_unavailable"] == ["notes"]  # nosec B101


@pytest.mark.asyncio
async def test_explicit_unsupported_conversation_scope_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"))
    called = False

    async def fake_pipeline(**kwargs):  # noqa: ARG001
        nonlocal called
        called = True
        return SimpleNamespace(documents=[], query="q", metadata={}, timings={}, citations=[], chunk_citations=[])

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)

    out = await module.execute_tool(
        "rag.search",
        {"query": "q", "sources": ["chats"]},
        context=RequestContext(request_id="r11", user_id="1", metadata={"conversation_id": "c1"}),
    )

    assert out["ok"] is False  # nosec B101
    assert out["reason_code"] == "unsupported_scope"  # nosec B101
    assert called is False  # nosec B101


@pytest.mark.asyncio
async def test_rag_module_jsonrpc_tools_call_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_pipeline(**kwargs):  # noqa: ANN001
        generated_answer = "Grounded answer." if kwargs.get("enable_generation") else None
        return SimpleNamespace(
            documents=[{"id": "d1", "content": "Evidence", "metadata": {"source": "media_db"}, "score": 0.9}],
            query=kwargs["query"],
            metadata={"hard_citations": {"coverage": 1.0}, "knowledge_trust": {"state": "grounded"}},
            timings={},
            citations=[{"id": "c1"}] if generated_answer else [],
            chunk_citations=[],
            generated_answer=generated_answer,
            errors=[],
            total_time=0.1,
            cache_hit=False,
            expanded_queries=[],
        )

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)

    registry = ModuleRegistry()
    await registry.register_module("rag", RagModule, ModuleConfig(name="rag"))
    protocol = MCPProtocol()
    protocol.module_registry = registry
    protocol.rbac_policy = _AllowAllRBAC()
    context = RequestContext(request_id="rag-jsonrpc-smoke", user_id="1", client_id="unit")

    search = await protocol.process_request(
        MCPRequest(
            method="tools/call",
            params={"name": "rag.search", "arguments": {"query": "q"}},
            id="smoke-rag-search",
        ),
        context,
    )
    answer = await protocol.process_request(
        MCPRequest(
            method="tools/call",
            params={"name": "rag.answer", "arguments": {"query": "q"}},
            id="smoke-rag-answer",
        ),
        context,
    )

    assert search.error is None  # nosec B101
    assert answer.error is None  # nosec B101
    search_payload = search.result["content"][0]["json"]
    answer_payload = answer.result["content"][0]["json"]
    assert search.result["content"][0]["type"] == "json"  # nosec B101
    assert answer.result["content"][0]["type"] == "json"  # nosec B101
    assert search_payload["ok"] is True  # nosec B101
    assert "answer" not in search_payload  # nosec B101
    assert answer_payload["answer"]["status"] in {"answered", "partial", "abstained"}  # nosec B101
