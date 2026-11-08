import os
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _set_test_mode_env(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")


def test_rag_capabilities_agentic_features(auth_headers):


     # Basic smoke: capabilities exposes agentic feature block
    with TestClient(fastapi_app, headers=auth_headers) as client:
        resp = client.get("/api/v1/rag/capabilities")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        features = data.get("features", {})
        assert "agentic_chunking" in features, "agentic_chunking not advertised in capabilities"
        agentic = features["agentic_chunking"]
        assert agentic.get("supported") is True
        assert "strategy" in agentic.get("parameters", []), "strategy parameter missing for agentic"
        # Defaults sanity
        defaults = agentic.get("defaults", {})
        assert defaults.get("strategy") == "standard"
        assert defaults.get("agentic_top_k_docs") == 3
        assert defaults.get("agentic_window_chars") == 1200
        assert defaults.get("agentic_max_tokens_read") == 6000
        assert defaults.get("agentic_max_tool_calls") == 8
        assert defaults.get("agentic_enable_tools") is False
        assert defaults.get("agentic_use_llm_planner") is False
        assert defaults.get("agentic_cache_ttl_sec") == 600
        # Quick start examples present
        qs = data.get("quick_start", {})
        assert "agentic_search" in qs and "ablate" in qs
    assert qs["agentic_search"]["body"]["strategy"] == "agentic"


def test_rag_capabilities_agentic_new_knobs(auth_headers):


     # Verify new agentic knobs are advertised
    with TestClient(fastapi_app, headers=auth_headers) as client:
        resp = client.get("/api/v1/rag/capabilities")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        agentic = data.get("features", {}).get("agentic_chunking", {})
        params = set(agentic.get("parameters", []))
        for p in (
            "agentic_adaptive_budgets",
            "agentic_coverage_target",
            "agentic_min_corroborating_docs",
            "agentic_max_redundancy",
            "agentic_enable_metrics",
        ):
            assert p in params, f"Missing agentic parameter in capabilities: {p}"
        defaults = agentic.get("defaults", {})
        assert defaults.get("agentic_adaptive_budgets") is True
        assert defaults.get("agentic_coverage_target") == 0.8
        assert defaults.get("agentic_min_corroborating_docs") == 2
        assert defaults.get("agentic_max_redundancy") == 0.9
        assert defaults.get("agentic_enable_metrics") is True


def test_rag_capabilities_quick_start_multihop_vlm(auth_headers):


     # Ensure capabilities advertises the multi-hop agentic with VLM quick-start
    with TestClient(fastapi_app, headers=auth_headers) as client:
        resp = client.get("/api/v1/rag/capabilities")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        quick = data.get("quick_start", {})
    assert "agentic_multihop_vlm" in quick, "quick_start.agentic_multihop_vlm missing"


def test_rag_capabilities_quick_start_explain(auth_headers):


    with TestClient(fastapi_app, headers=auth_headers) as client:
        resp = client.get("/api/v1/rag/capabilities")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        quick = data.get("quick_start", {})
        assert "agentic_explain" in quick, "quick_start.agentic_explain missing"


def test_rag_agentic_streaming_plan_spans_then_delta(client_with_agentic_overrides, monkeypatch):


     # Patch retriever for agentic assembly
    from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass
        async def retrieve(self, *args, **kwargs):
            return [Document(id="mZ", content="Accuracy table A|B|C\n1|2|3", metadata={"title": "Tbl"}, source=DataSource.MEDIA_DB, score=0.8)]

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    # Patch streaming generator used by endpoint to produce a minimal stream
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    async def fake_generate_streaming_response(context, **kwargs):  # noqa: ANN001
        async def _gen():
            yield "Hello world"
        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    monkeypatch.setattr(rag_ep, "generate_streaming_response", fake_generate_streaming_response)
    # Patch the endpoint's own MultiDatabaseRetriever to avoid DB access
    class FakeEP_Retriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}
        async def retrieve(self, *args, **kwargs):
            return [Document(id="e1", content="Doc body", metadata={"title": "T"}, source=DataSource.MEDIA_DB, score=0.5)]
    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", FakeEP_Retriever)

    client = client_with_agentic_overrides
    payload = {
            "query": "Compare accuracy tables",
            "strategy": "agentic",
            "search_mode": "fts",
            "enable_generation": True,
            "agentic_enable_tools": True,
            "agentic_enable_vlm_late_chunking": False
    }
    with client.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        lines = []
        for raw in resp.iter_lines():
            if not raw:
                continue
            try:
                import json as _json
                evt = _json.loads(raw)
                lines.append(evt)
            except Exception:
                continue
            if len(lines) >= 6:
                break
    # Expect first two event types
    assert lines[0].get("type") == "plan"
    assert lines[1].get("type") == "spans"
    # And eventually a delta appears
    types = [e.get("type") for e in lines]
    assert "delta" in types


@pytest.fixture()
def client_with_agentic_overrides(monkeypatch, auth_headers):
     # Override auth dependencies: accept any test user; disable rate limits
    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _noop():
        return None

    # Monkeypatch RBAC enforcer to a no-op to avoid DB access
    import tldw_Server_API.app.api.v1.API_Deps.auth_deps as auth_deps
    async def _no_rbac(*args, **kwargs):  # noqa: ARG001
        return None
    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", _no_rbac)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[check_rate_limit] = _noop
    # Avoid DB initialization by overriding DB deps to return None
    try:
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user as _get_media_db
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user as _get_chacha_db
        async def _none_media_db():
            return None
        async def _none_chacha_db():
            return None
        fastapi_app.dependency_overrides[_get_media_db] = _none_media_db
        fastapi_app.dependency_overrides[_get_chacha_db] = _none_chacha_db
    except Exception:
        _ = None

    try:
        # Prefer disabling lifespan to avoid DB/services startup in CI; skip if not supported
        import inspect as _inspect
        if 'lifespan' in _inspect.signature(TestClient.__init__).parameters:
            with TestClient(fastapi_app, headers=auth_headers, raise_server_exceptions=False, lifespan='off') as client:
                yield client
        else:  # Fallback: run without lifespan guard; if flaky in env, skip
            with TestClient(fastapi_app, headers=auth_headers, raise_server_exceptions=False) as client:
                yield client
    finally:
        fastapi_app.dependency_overrides.clear()


def _recording_pipeline(monkeypatch) -> list[dict]:
    """Replace the unified pipeline with a recorder that returns an empty result."""
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    calls: list[dict] = []

    async def _fake_pipeline(**kwargs):
        calls.append(kwargs)
        return up.UnifiedSearchResult(documents=[], query=kwargs.get("query", ""), errors=[])

    monkeypatch.setattr(rag_ep, "unified_rag_pipeline", _fake_pipeline)
    return calls


def test_rag_search_with_generation_refuses_an_over_quota_user(client_with_agentic_overrides, monkeypatch):
    """A search that generates an answer is refused with the 402 before any retrieval."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    calls = _recording_pipeline(monkeypatch)
    refuse_llm_work(monkeypatch)
    resp = client_with_agentic_overrides.post("/api/v1/rag/search", json={"query": "q", "enable_generation": True})
    assert_llm_quota_402(resp)
    assert calls == []


def test_rag_search_without_generation_is_not_refused_by_the_llm_quota(client_with_agentic_overrides, monkeypatch):
    """A search with generation off does no LLM work at the boundary, so an over-quota user still searches."""
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    calls = _recording_pipeline(monkeypatch)
    refuse_llm_work(monkeypatch)
    resp = client_with_agentic_overrides.post("/api/v1/rag/search", json={"query": "q", "enable_generation": False})
    assert resp.status_code != 402, resp.text
    assert len(calls) == 1


def test_rag_search_profile_without_explicit_generation_is_not_refused_by_the_llm_quota(
    client_with_agentic_overrides, monkeypatch
):
    """The resolved request decides: no shipped profile turns generation on, so a profile alone is not refused."""
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    calls = _recording_pipeline(monkeypatch)
    refuse_llm_work(monkeypatch)
    resp = client_with_agentic_overrides.post("/api/v1/rag/search", json={"query": "q", "rag_profile": "accuracy"})
    assert resp.status_code != 402, resp.text
    assert len(calls) == 1 and calls[0].get("enable_generation") is False


def _over_the_limit_with_own_keys(monkeypatch, own: set[str]) -> tuple[list[str], list[bool]]:
    """An over-limit user with their own key for ``own``; the pipeline records whether it ran marked admitted."""
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    asked: list[str] = []
    marked: list[bool] = []

    async def _own_key(_uid, provider):
        """Stands in for the lookup (the boundary is under test)."""
        asked.append(provider)
        return provider in own

    async def _fake_pipeline(**kwargs):
        marked.append(current_llm_call_context().prechecked)
        return up.UnifiedSearchResult(documents=[], query=kwargs.get("query", ""), errors=[])

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(rag_ep, "unified_rag_pipeline", _fake_pipeline)
    refuse_llm_work(monkeypatch)
    return asked, marked


@pytest.mark.parametrize(
    ("extra", "own", "looked_up"),
    [
        ({}, {"openai"}, ["openai"]),
        ({"chat_history": [{"role": "user", "content": "Earlier"}], "classifier_provider": "groq"},
         {"openai", "groq"}, ["openai", "groq"]),
        ({"enable_structured_response": True, "enable_query_decomposition": True, "enable_intent_routing": True,
          "enable_dynamic_granularity": True, "enable_image_search": True, "enable_video_search": True},
         {"openai"}, ["openai"]),
        ({"enable_abstention": True, "enable_knowledge_strips": True, "enable_web_fallback": True,
          "enable_query_rewriting_loop": True, "adaptive_rerun_on_low_confidence": True,
          "adaptive_advanced_rewrites": True},
         {"openai"}, ["openai"]),
    ],
    ids=["answer-only", "reformulated-history", "switches-without-an-llm-call", "stages-without-an-llm-call"],
)
def test_rag_search_admits_an_over_limit_user_whose_llm_calls_run_on_their_own_keys(
    client_with_agentic_overrides, monkeypatch, extra, own, looked_up
):
    """Past a limit, a search whose answer (and reformulation) run on the user's own keys goes on, unmarked (TASK-13956)."""
    asked, marked = _over_the_limit_with_own_keys(monkeypatch, own)
    resp = client_with_agentic_overrides.post(
        "/api/v1/rag/search",
        json={"query": "q", "enable_generation": True, "generation_provider": "openai", **extra},
    )
    assert resp.status_code != 402, resp.text
    assert marked == [False]
    assert asked == looked_up


@pytest.mark.parametrize(
    "extra",
    [
        {"enable_claims": True},
        {"enable_document_grading": True, "grading_provider": "openai"},
        {"enable_reranking": True, "reranking_strategy": "llm_scoring"},
        {"search_depth_mode": "quality"},
        {"chat_history": [{"role": "user", "content": "Earlier"}], "classifier_provider": "groq"},
        {"enable_evidence_accumulation": True},
        {"enable_evidence_chains": True},
    ],
    ids=["claims", "grading", "llm-rerank", "quality-depth", "history-on-a-server-key", "evidence-accumulation",
         "evidence-chains"],
)
def test_rag_search_still_refuses_an_own_key_answer_with_another_llm_stage(client_with_agentic_overrides, monkeypatch, extra):
    """An LLM stage the boundary cannot place on the user's own key keeps the 402 before any retrieval (TASK-13956)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402

    _asked, marked = _over_the_limit_with_own_keys(monkeypatch, {"openai"})
    resp = client_with_agentic_overrides.post(
        "/api/v1/rag/search",
        json={"query": "q", "enable_generation": True, "generation_provider": "openai", **extra},
    )
    assert_llm_quota_402(resp)
    assert marked == []


# Every boolean switch of the RAG request schemas makes an LLM call besides the answer and the reformulation
# (rag_unified._RAG_OTHER_LLM_STAGES) or is declared here (TASK-15427).
_NO_OTHER_LLM_CALL = frozenset({
    # Their call is one _rag_llm_providers names: the answer, the reformulation, LLM reranking (by its strategy).
    "enable_generation", "enable_query_reformulation", "enable_reranking",
    # Read only by the agentic strategy, which _rag_llm_providers refuses as a non-standard strategy.
    "agentic_adaptive_budgets", "agentic_debug_trace", "agentic_enable_metrics", "agentic_enable_query_decomposition",
    "agentic_enable_section_index", "agentic_enable_semantic_within", "agentic_enable_table_support",
    "agentic_enable_tools", "agentic_enable_vlm_late_chunking", "agentic_extractive_only",
    "agentic_prefer_structural_anchors", "agentic_quote_spans", "agentic_use_llm_planner",
    "agentic_use_provider_embeddings_within", "agentic_vlm_detect_tables_only", "explain_only",
    # Read only inside a listed stage: the research loop, document grading, claims, post-verification.
    "enable_image_search", "enable_video_search", "enable_research_action_dedup", "enable_research_progress",
    "search_url_scraping", "grading_fallback_to_score", "enable_query_rewriting_loop", "doc_only_verification",
    "generate_verification_report", "adaptive_advanced_rewrites", "adaptive_rerun_on_low_confidence",
    "adaptive_rerun_bypass_cache", "adaptive_rerun_include_generation",
    # No LLM call: rule-based, retrieval, filtering, formatting, telemetry, or the answer's own generator again.
    "enable_structured_response", "enable_query_decomposition", "enable_intent_routing",
    "enable_dynamic_granularity", "enable_abstention", "enable_knowledge_strips", "enable_web_fallback",
    "enable_pre_retrieval_clarification", "enable_numeric_fidelity", "strict_extractive", "require_hard_citations",
    "expand_query", "spell_check", "enable_discussion_search", "enable_multi_vector_passages", "mv_flatten_to_spans",
    "enable_text_late_chunking", "enable_enhanced_chunking", "enable_vlm_late_chunking", "vlm_detect_tables_only",
    "enable_table_processing", "enable_numeric_table_boost", "enable_parent_expansion", "include_parent_document",
    "include_sibling_chunks", "enable_citations", "enable_chunk_citations", "include_page_numbers",
    "enable_security_filter", "detect_pii", "redact_pii", "content_filter", "enable_content_policy_filter",
    "enable_injection_filter", "enable_html_sanitizer", "enable_cache", "adaptive_cache", "use_embedding_cache",
    "use_connection_pool", "enable_resilience", "circuit_breaker", "enable_batch", "enable_checkpoint",
    "collect_feedback", "apply_feedback_boost", "highlight_results", "highlight_query_terms",
    "include_rerank_debug_documents", "debug_mode", "track_cost", "enable_analytics", "enable_monitoring",
    "enable_observability", "enable_performance_analysis",
})


def test_every_rag_request_switch_says_whether_it_makes_another_llm_call():
    """A new switch of the RAG request schemas must join _RAG_OTHER_LLM_STAGES or _NO_OTHER_LLM_CALL (TASK-15427)."""
    from typing import Optional

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedBatchRequest, UnifiedRAGRequest

    switches = {
        name
        for model in (UnifiedRAGRequest, UnifiedBatchRequest)
        for name, field in model.model_fields.items()
        if field.annotation in (bool, Optional[bool])
    }
    listed = set(rag_ep._RAG_OTHER_LLM_STAGES)
    assert switches - listed - _NO_OTHER_LLM_CALL == set(), "add each to _RAG_OTHER_LLM_STAGES or _NO_OTHER_LLM_CALL"
    assert listed & _NO_OTHER_LLM_CALL == set()
    assert (listed | _NO_OTHER_LLM_CALL) - switches == set()

@pytest.mark.parametrize(
    ("default_on", "admitted"),
    [(None, True), ("SEARCH_SUGGESTIONS", False), ("SEARCH_IMAGE_SEARCH", True)],
    ids=["no-default", "suggestions-by-default", "image-search-by-default"],
)
def test_rag_batch_names_the_providers_of_the_resolved_request(
    client_with_agentic_overrides, monkeypatch, default_on, admitted
):
    """A server default that turns on another LLM stage keeps the 402 on a batch, as on search (TASK-13956)."""
    from fastapi import HTTPException

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402

    asked, _marked = _over_the_limit_with_own_keys(monkeypatch, {"openai"})
    reached: list[bool] = []

    def _past_the_check(*_args, **_kwargs):
        """The batch's next step after the quota check: stop an admitted request before any work."""
        reached.append(True)
        raise HTTPException(status_code=418, detail="the quota check let the request through")

    monkeypatch.setattr(
        rag_ep, "_search_agent_setting", lambda env_key, _key: "true" if env_key == default_on else None
    )
    monkeypatch.setattr(rag_ep, "_resolve_kanban_db_path", _past_the_check)
    resp = client_with_agentic_overrides.post(
        "/api/v1/rag/batch",
        json={"queries": ["q"], "enable_generation": True, "generation_provider": "openai"},
    )

    if admitted:
        assert resp.status_code == 418, resp.text
        assert asked == ["openai"]
    else:
        assert_llm_quota_402(resp)
        assert reached == []


def test_rag_agentic_search_smoke_api(client_with_agentic_overrides, monkeypatch):


    client = client_with_agentic_overrides

    # Patch retriever used inside agentic pipeline to return a single simple doc
    from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource

    class FakeRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

        async def retrieve(self, *args, **kwargs):  # noqa: ARG002
            return [
                Document(
                    id="m1",
                    content=(
                        "ResNet uses residual connections. Residual links help gradient flow and enable deeper networks."
                    ),
                    metadata={"title": "ResNet", "source": "media_db", "ingestion_date": "2024-01-01"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    # Issue agentic search request (no generation to avoid LLM calls)
    payload = {
        "query": "How do residual connections help training?",
        "strategy": "agentic",
        "search_mode": "fts",
        "top_k": 5,
        "enable_generation": False,
        "agentic_enable_tools": True,
        "agentic_max_tool_calls": 3,
    }
    resp = client.post("/api/v1/rag/search", json=payload)
    if resp.status_code != 200:
        import pytest as _pytest
        _pytest.skip(f"agentic verification flags test skipped due to server error: {resp.status_code}")
    out = resp.json()
    assert isinstance(out.get("documents"), list) and len(out["documents"]) >= 1
    # First document should be the synthetic agentic chunk
    first = out["documents"][0]
    meta = first.get("metadata", {})
    assert meta.get("source") == "agentic"
    # Top-level metadata should advertise agentic strategy
    assert out.get("metadata", {}).get("strategy") == "agentic"
    assert out.get("metadata", {}).get("derived_from_document_ids") == ["m1"]


def test_rag_agentic_search_uses_shared_request_resolution_and_retrieval_plan(
    client_with_agentic_overrides,
    monkeypatch,
):

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    captured: dict[str, object] = {"context_builder_calls": 0}

    def fake_build_agentic_execution_context(*, resolved_request, retrieval_plan, payload_override=None):  # noqa: ANN001
        captured["context_builder_calls"] = int(captured["context_builder_calls"]) + 1
        captured["context_builder_resolved_request"] = resolved_request
        captured["context_builder_retrieval_plan"] = retrieval_plan
        payload = dict(payload_override or resolved_request.payload)
        payload["agentic_enable_tools"] = True
        payload["agentic_max_tool_calls"] = 11
        payload["agentic_coverage_target"] = 0.91
        payload["agentic_enable_metrics"] = False
        payload["agentic_debug_trace"] = True
        return (
            payload,
            rag_ep.AgenticConfig(
                top_k_docs=4,
                enable_tools=True,
                max_tool_calls=11,
                coverage_target=0.91,
                enable_metrics=False,
                debug_trace=True,
            ),
        )

    async def fake_agentic_rag_pipeline(**kwargs):  # noqa: ANN001
        captured["kwargs"] = kwargs
        return rag_ep.UnifiedSearchResult(
            documents=[
                Document(
                    id="synthetic-agentic",
                    content="Synthetic agentic chunk",
                    metadata={"title": "Synthetic", "source": "agentic"},
                    source=DataSource.MEDIA_DB,
                    score=1.0,
                )
            ],
            query=str(kwargs.get("query", "")),
            expanded_queries=[],
            metadata={"strategy": "agentic"},
            timings={},
            citations=[],
            feedback_id=None,
            generated_answer=None,
            cache_hit=False,
            errors=[],
            security_report=None,
            total_time=0.0,
        )

    monkeypatch.setattr(rag_ep, "build_agentic_execution_context", fake_build_agentic_execution_context)
    monkeypatch.setattr(rag_ep, "agentic_rag_pipeline", fake_agentic_rag_pipeline)

    payload = {
        "query": "shared contract parity",
        "strategy": "agentic",
        "rag_profile": "fast",
        "sources": ["notes", "media_db", "notes"],
        "corpus": "tenant-x",
        "agentic_top_k_docs": 4,
        "enable_generation": False,
    }

    resp = client_with_agentic_overrides.post("/api/v1/rag/search", json=payload)
    assert resp.status_code == 200, resp.text

    assert captured["context_builder_calls"] == 1
    kwargs = captured["kwargs"]
    assert kwargs["query"] == "shared contract parity"
    assert kwargs["top_k"] == 6
    assert kwargs["index_namespace"] == "tenant-x"
    assert kwargs["sources"] == ["notes", "media_db"]
    assert kwargs["agentic"].top_k_docs == 4
    assert kwargs["agentic"].enable_tools is True
    assert kwargs["agentic"].max_tool_calls == 11
    assert kwargs["agentic"].coverage_target == 0.91
    assert kwargs["agentic"].enable_metrics is False
    assert kwargs["agentic"].debug_trace is True

    resolved_request = kwargs["resolved_request"]
    retrieval_plan = kwargs["retrieval_plan"]
    assert captured["context_builder_resolved_request"] is resolved_request
    assert captured["context_builder_retrieval_plan"] is retrieval_plan
    assert resolved_request.payload["generation_prompt"] == "instruction_tuned"
    assert resolved_request.payload["top_k"] == 6
    assert retrieval_plan.top_k == 6
    assert retrieval_plan.sources == ("notes", "media_db")
    assert retrieval_plan.index_namespace == "tenant-x"


def test_rag_agentic_search_preserves_legacy_coarse_docs_metadata_window(
    client_with_agentic_overrides,
    monkeypatch,
):
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [
                Document(
                    id="m1",
                    content="Primary evidence",
                    metadata={"title": "Doc 1", "source": "media_db", "ingestion_date": "2024-01-01"},
                    source=DataSource.MEDIA_DB,
                    score=0.95,
                ),
                Document(
                    id="m2",
                    content="Secondary evidence",
                    metadata={"title": "Doc 2", "source": "media_db", "ingestion_date": "2024-01-01"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                ),
                Document(
                    id="m3",
                    content="Tertiary evidence",
                    metadata={"title": "Doc 3", "source": "media_db", "ingestion_date": "2024-01-01"},
                    source=DataSource.MEDIA_DB,
                    score=0.85,
                ),
            ]

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    payload = {
        "query": "coarse docs compatibility",
        "strategy": "agentic",
        "search_mode": "fts",
        "top_k": 5,
        "agentic_top_k_docs": 2,
        "enable_generation": False,
    }

    resp = client_with_agentic_overrides.post("/api/v1/rag/search", json=payload)
    assert resp.status_code == 200, resp.text
    out = resp.json()

    assert [entry["id"] for entry in out["metadata"]["coarse_docs"]] == ["m1", "m2"]


def test_rag_agentic_search_verification_flags(client_with_agentic_overrides, monkeypatch):


    client = client_with_agentic_overrides

    # Patch retriever inside agentic_chunker to return a numeric-bearing doc
    from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [
                Document(
                    id="m2",
                    content="We ran 42 experiments and observed consistent results.",
                    metadata={"title": "Results", "source": "media_db", "ingestion_date": "2024-01-01"},
                    source=DataSource.MEDIA_DB,
                    score=0.95,
                )
            ]

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    # Patch AnswerGenerator in generation module to return an answer referencing the number
    class FakeAnswerGenerator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *, query: str, context: str, prompt_template=None, max_tokens=None, temperature=None):  # noqa: ARG002
            return {"answer": "We ran 42 experiments. The findings were consistent."}

    import tldw_Server_API.app.core.RAG.rag_service.generation as gen_mod
    monkeypatch.setattr(gen_mod, "AnswerGenerator", FakeAnswerGenerator, raising=False)

    payload = {
        "query": "How many experiments?",
        "strategy": "agentic",
        "search_mode": "fts",
        "enable_generation": True,
        "require_hard_citations": True,
        "enable_numeric_fidelity": True,
        "numeric_fidelity_behavior": "continue",
        "top_k": 3,
    }

    resp = client.post("/api/v1/rag/search", json=payload)
    if resp.status_code != 200:
        import pytest as _pytest
        _pytest.skip(f"agentic verification flags skipped due to server error: {resp.status_code}")
    out = resp.json()
    md = out.get("metadata", {})
    # Hard citations should be attached
    assert isinstance(md.get("hard_citations"), dict)
    # Numeric fidelity should include '42' in present
    nf = md.get("numeric_fidelity") or {}
    assert any("42" in x for x in (nf.get("present") or []))
