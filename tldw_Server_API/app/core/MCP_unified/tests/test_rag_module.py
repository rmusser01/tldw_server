from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
    KnowledgeSourceHealthResponse,
    UnifiedRAGResponse,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import rag_module as rag_module_impl
from tldw_Server_API.app.core.MCP_unified.modules.implementations.rag_module import (
    RagModule,
    _build_mcp_rag_request,
    _compact_rag_response,
)
from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext
from tldw_Server_API.app.core.MCP_unified import server as mcp_server_module


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


class _DenyNotesSearchRBAC:
    async def check_permission(self, _user_id, resource, _action, resource_id=None):  # noqa: ANN001
        if getattr(resource, "value", resource) == "tool" and resource_id == "notes.search":
            return False
        return True


class _NoopRateLimiter:
    async def check_rate_limit(self, *args, **kwargs):  # noqa: ANN001
        del args, kwargs


def _install_source_tool_registry(protocol: MCPProtocol, *tool_names: str) -> None:
    available = set(tool_names)
    original_find_module_for_tool = protocol.module_registry.find_module_for_tool

    async def find_module_for_tool(tool_name: str):  # noqa: ANN001
        if tool_name in available:
            return SimpleNamespace(name="source-module")
        return await original_find_module_for_tool(tool_name)

    protocol.module_registry.find_module_for_tool = find_module_for_tool  # type: ignore[method-assign]


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


def test_compact_response_redacts_raw_document_and_response_metadata() -> None:
    response = UnifiedRAGResponse(
        query="q",
        documents=[
            {
                "id": "d1",
                "content": "Evidence",
                "score": 0.9,
                "source": "media_db",
                "metadata": {
                    "source": "media_db",
                    "source_id": "42",
                    "title": "Allowed title",
                    "path": "SECRET_PATH",
                    "db_path": "SECRET_DB",
                    "provider": "SECRET_PROVIDER",
                    "prompt": "SECRET_PROMPT",
                    "debug": {"raw": "SECRET_DEBUG"},
                },
                "provider_payload": {"raw_value": "SECRET_VALUE"},
            }
        ],
        citations=[],
        chunk_citations=[],
        metadata={
            "hard_citations": {"coverage": 0.5},
            "knowledge_trust": {"state": "grounded"},
            "provider_debug": "SECRET_PROVIDER",
            "prompt": "SECRET_PROMPT",
            "db_path": "SECRET_DB",
        },
    )

    payload = _compact_rag_response(
        response,
        mode="search",
        request_metadata={"sources_requested": ["media_db"], "sources_explicit": True},
        max_documents=1,
        max_content_chars=2000,
    )

    document = payload["documents"][0]
    assert document["metadata"] == {  # nosec B101
        "source": "media_db",
        "source_id": "42",
        "title": "Allowed title",
    }
    assert "provider_payload" not in document  # nosec B101
    rendered_payload = repr(payload)
    assert "SECRET_" not in rendered_payload  # nosec B101
    assert "knowledge_trust" not in payload["metadata"]  # nosec B101
    assert payload["metadata"]["knowledge_trust_state"] == "grounded"  # nosec B101


def test_compact_response_bounds_and_redacts_citations() -> None:
    response = UnifiedRAGResponse(
        query="q",
        documents=[],
        citations=[
            {
                "id": f"c-{index}",
                "text": "x" * 1200,
                "title": "Allowed",
                "metadata": {"path": "SECRET_PATH"},
                "provider_payload": "SECRET_PROVIDER",
            }
            for index in range(25)
        ],
        chunk_citations=[
            {
                "id": f"chunk-{index}",
                "snippet": "y" * 1200,
                "debug": "SECRET_DEBUG",
            }
            for index in range(25)
        ],
        metadata={},
    )

    payload = _compact_rag_response(
        response,
        mode="search",
        request_metadata={"sources_requested": ["media_db"], "sources_explicit": True},
        max_documents=1,
        max_content_chars=2000,
    )

    assert len(payload["citations"]) == 20  # nosec B101
    assert len(payload["chunk_citations"]) == 20  # nosec B101
    assert len(payload["citations"][0]["text"]) == 1000  # nosec B101
    assert len(payload["chunk_citations"][0]["snippet"]) == 1000  # nosec B101
    assert "SECRET_" not in repr(payload["citations"])  # nosec B101
    assert "SECRET_" not in repr(payload["chunk_citations"])  # nosec B101
    assert payload["metadata"]["citations_truncated"] is True  # nosec B101
    assert payload["metadata"]["chunk_citations_truncated"] is True  # nosec B101


def test_compact_response_does_not_report_requested_sources_as_used_when_empty() -> None:
    response = UnifiedRAGResponse(
        query="q",
        documents=[],
        citations=[],
        chunk_citations=[],
        metadata={},
    )

    payload = _compact_rag_response(
        response,
        mode="search",
        request_metadata={"sources_requested": ["media_db"], "sources_explicit": True},
        max_documents=1,
        max_content_chars=2000,
    )

    assert payload["metadata"]["sources_used"] == []  # nosec B101


@pytest.mark.asyncio
async def test_rag_module_exposes_four_strict_tools() -> None:
    module = RagModule(ModuleConfig(name="rag"))

    tools = {tool["name"]: tool for tool in await module.get_tools()}

    assert set(tools) == {"rag.capabilities", "rag.source_health", "rag.search", "rag.answer"}  # nosec B101
    for tool_name in ("rag.capabilities", "rag.source_health", "rag.search", "rag.answer"):
        assert tools[tool_name]["inputSchema"]["additionalProperties"] is False  # nosec B101
    assert tools["rag.answer"]["metadata"]["category"] == "rag_generation"  # nosec B101


@pytest.mark.asyncio
async def test_rag_capabilities_reports_mcp_top_k_limit() -> None:
    controls = _RecordingControls()
    module = RagModule(ModuleConfig(name="rag"), controls=controls)

    out = await module.execute_tool("rag.capabilities", {}, context=RequestContext(request_id="limits", user_id="1"))

    assert out["limits"]["top_k_max"] == 50  # nosec B101


@pytest.mark.asyncio
async def test_rag_search_executes_shared_pipeline_without_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"), controls=_RecordingControls())
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
    module = RagModule(ModuleConfig(name="rag"), controls=_RecordingControls())

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
    module = RagModule(ModuleConfig(name="rag"), controls=_RecordingControls())

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
    module = RagModule(ModuleConfig(name="rag"), controls=_RecordingControls())
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
    assert calls["enable_web_fallback"] is False  # nosec B101
    assert calls["enable_query_classification"] is False  # nosec B101
    assert calls["search_depth_mode"] is None  # nosec B101


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

    assert out["ok"] is True  # nosec B101
    assert isinstance(out["sources"], list)  # nosec B101
    assert ("protocol_rate", "rag.source_health", "search") in controls.calls  # nosec B101
    assert ("rbac_rate", "rag.search") in controls.calls  # nosec B101
    assert ("read_scope", "rag.source_health") in controls.calls  # nosec B101
    assert not any(call[0] == "quota" for call in controls.calls)  # nosec B101
    assert not any(call[0] == "usage" for call in controls.calls)  # nosec B101


@pytest.mark.asyncio
async def test_rag_source_health_reports_unavailable_sources_with_ok_contract() -> None:
    module = RagModule(ModuleConfig(name="rag"), controls=_UnavailableNotesControls())

    out = await module.execute_tool(
        "rag.source_health",
        {"sources": ["media", "notes"], "allow_partial": True},
        context=RequestContext(request_id="source-health-partial", user_id="1"),
    )

    assert out["ok"] is True  # nosec B101
    assert out["sources_unavailable"] == ["notes"]  # nosec B101
    assert out["warnings"] == ["notes unavailable"]  # nosec B101
    assert all(source["source_id"] != "notes" for source in out["sources"])  # nosec B101


@pytest.mark.asyncio
async def test_rag_source_health_fails_closed_when_explicit_source_is_unavailable() -> None:
    module = RagModule(ModuleConfig(name="rag"), controls=_UnavailableNotesControls())

    out = await module.execute_tool(
        "rag.source_health",
        {"sources": ["notes"]},
        context=RequestContext(request_id="source-health-denied", user_id="1"),
    )

    assert out["ok"] is False  # nosec B101
    assert out["reason_code"] == "source_unavailable"  # nosec B101
    assert out["sources_unavailable"] == ["notes"]  # nosec B101
    assert out["warnings"] == ["notes unavailable"]  # nosec B101


@pytest.mark.asyncio
async def test_rag_source_health_uses_context_db_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    controls = _RecordingControls()
    module = RagModule(ModuleConfig(name="rag"), controls=controls)
    captured_paths: dict[str, str] = {}

    def fake_health_payload(**kwargs):  # noqa: ANN003
        resolver = kwargs["existing_source_db_paths_fn"]
        captured_paths.update(resolver(None, None))
        return KnowledgeSourceHealthResponse(sources=[])

    monkeypatch.setattr(rag_module_impl.rag_transport, "build_source_health_payload", fake_health_payload)

    out = await module.execute_tool(
        "rag.source_health",
        {},
        context=RequestContext(
            request_id="source-health-db-paths",
            user_id="1",
            db_paths={
                "media": "media.db",
                "chacha": "notes.db",
                "prompts": "prompts.db",
                "kanban": "kanban.db",
            },
        ),
    )

    assert out["ok"] is True  # nosec B101
    assert captured_paths == {  # nosec B101
        "media_db": "media.db",
        "chacha_db": "notes.db",
        "prompts_db": "prompts.db",
        "kanban_db": "kanban.db",
    }


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
async def test_default_controls_enforce_source_tool_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"))
    protocol = MCPProtocol()
    protocol.rbac_policy = _DenyNotesSearchRBAC()
    protocol.rate_limiter = _NoopRateLimiter()
    monkeypatch.setattr(mcp_server_module, "_server", SimpleNamespace(protocol=protocol))

    async def fail_if_pipeline_runs(**kwargs):  # noqa: ANN001
        del kwargs
        raise AssertionError("pipeline must not run when source authorization fails")

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fail_if_pipeline_runs)

    out = await module.execute_tool(
        "rag.search",
        {"query": "q", "sources": ["notes"]},
        context=RequestContext(request_id="source-auth", user_id="1", client_id="unit"),
    )

    assert out["ok"] is False  # nosec B101
    assert out["reason_code"] == "source_unavailable"  # nosec B101
    assert out["sources_unavailable"] == ["notes"]  # nosec B101


@pytest.mark.asyncio
async def test_default_controls_require_source_tool_module_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"))
    protocol = MCPProtocol()
    protocol.rbac_policy = _AllowAllRBAC()
    protocol.rate_limiter = _NoopRateLimiter()
    monkeypatch.setattr(mcp_server_module, "_server", SimpleNamespace(protocol=protocol))

    async def fail_if_pipeline_runs(**kwargs):  # noqa: ANN001
        del kwargs
        raise AssertionError("pipeline must not run when source module is unavailable")

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fail_if_pipeline_runs)

    out = await module.execute_tool(
        "rag.search",
        {"query": "q", "sources": ["media"]},
        context=RequestContext(request_id="source-module-required", user_id="1", client_id="unit"),
    )

    assert out["ok"] is False  # nosec B101
    assert out["reason_code"] == "source_unavailable"  # nosec B101
    assert out["sources_unavailable"] == ["media_db"]  # nosec B101


@pytest.mark.asyncio
async def test_default_controls_authorize_character_backed_rag_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"))
    protocol = MCPProtocol()
    protocol.rbac_policy = _AllowAllRBAC()
    protocol.rate_limiter = _NoopRateLimiter()
    _install_source_tool_registry(protocol, "characters.search")
    monkeypatch.setattr(mcp_server_module, "_server", SimpleNamespace(protocol=protocol))
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
        {"query": "q", "sources": ["world_books", "dictionaries"]},
        context=RequestContext(request_id="character-backed-sources", user_id="1", client_id="unit"),
    )

    assert out["ok"] is True  # nosec B101
    assert calls["sources"] == ["world_books", "dictionaries"]  # nosec B101


@pytest.mark.asyncio
async def test_allow_partial_does_not_run_pipeline_when_all_sources_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"), controls=_UnavailableNotesControls())

    async def fail_if_pipeline_runs(**kwargs):  # noqa: ANN001
        del kwargs
        raise AssertionError("pipeline must not run with no authorized sources")

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fail_if_pipeline_runs)

    out = await module.execute_tool(
        "rag.search",
        {"query": "q", "sources": ["notes"], "allow_partial": True},
        context=RequestContext(request_id="all-filtered", user_id="1"),
    )

    assert out["ok"] is False  # nosec B101
    assert out["reason_code"] == "source_unavailable"  # nosec B101
    assert out["sources_unavailable"] == ["notes"]  # nosec B101


@pytest.mark.asyncio
async def test_default_controls_enforce_rag_query_billing_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    module = RagModule(ModuleConfig(name="rag"))
    protocol = MCPProtocol()
    protocol.rbac_policy = _AllowAllRBAC()
    protocol.rate_limiter = _NoopRateLimiter()
    _install_source_tool_registry(protocol, "media.search")
    monkeypatch.setattr(mcp_server_module, "_server", SimpleNamespace(protocol=protocol))

    async def deny_limit(*, request_like, current_user=None, units: int = 1):  # noqa: ANN001
        del request_like, current_user, units
        raise PermissionError("RAG query daily limit exceeded")

    async def fail_if_pipeline_runs(**kwargs):  # noqa: ANN001
        del kwargs
        raise AssertionError("pipeline must not run when RAG query billing limit is exceeded")

    monkeypatch.setattr(
        rag_module_impl.rag_transport,
        "enforce_rag_query_limit_for_org_context",
        deny_limit,
        raising=False,
    )
    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fail_if_pipeline_runs)

    with pytest.raises(PermissionError, match="daily limit"):
        await module.execute_tool(
            "rag.search",
            {"query": "q", "sources": ["media"]},
            context=RequestContext(request_id="rag-quota", user_id="1", client_id="unit"),
        )


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
    protocol.rate_limiter = _NoopRateLimiter()
    _install_source_tool_registry(protocol, "media.search")
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
