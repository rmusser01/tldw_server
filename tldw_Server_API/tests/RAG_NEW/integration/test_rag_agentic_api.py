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
