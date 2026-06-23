"""Integration coverage for `/rag/search/stream` profile and event parity."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _set_test_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("RAG_DEFAULT_LLM_PROVIDER", "test-provider")
    monkeypatch.setenv("RAG_DEFAULT_LLM_MODEL", "default-model")


@pytest.fixture()
def client_with_stream_overrides(
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> Iterator[TestClient]:
    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _noop():
        return None

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[check_rate_limit] = _noop

    try:
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user as _get_media_db
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user as _get_chacha_db

        class StubDB:
            def __init__(self, path: str):
                self.db_path = path

        async def _stub_media_db():
            return StubDB("stub_media.db")

        async def _stub_chacha_db():
            return StubDB("stub_chacha.db")

        fastapi_app.dependency_overrides[_get_media_db] = _stub_media_db
        fastapi_app.dependency_overrides[_get_chacha_db] = _stub_chacha_db
    except Exception:
        _ = None

    with TestClient(fastapi_app, headers=auth_headers) as client:
        yield client

    fastapi_app.dependency_overrides.clear()


def test_rag_streaming_parity_generation_and_hybrid_sources(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:


    from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    captured = {"retrieve_kwargs": None, "generation_config": None}

    class StubRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, query, **kwargs):
            captured["retrieve_kwargs"] = {"query": query, **kwargs}
            return [
                Document(
                    id="doc-1",
                    content="Context content",
                    metadata={"title": "Doc"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    async def fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:
        captured["generation_config"] = context.config.get("generation")

        async def _gen():
            yield "chunk"

        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", StubRetriever)
    monkeypatch.setattr(rag_ep, "generate_streaming_response", fake_generate_streaming_response)

    payload = {
        "query": "Hybrid streaming parity",
        "search_mode": "hybrid",
        "sources": ["media_db", "notes"],
        "enable_generation": True,
        "top_k": 7,
        "min_score": 0.12,
        "generation_model": "explicit-model",
        "generation_prompt": "concise",
        "max_generation_tokens": 256,
    }

    with client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        next(resp.iter_lines(), None)

    retrieve_kwargs = captured["retrieve_kwargs"]
    assert retrieve_kwargs is not None
    config = retrieve_kwargs.get("config")
    assert config.max_results == 7
    assert config.min_score == 0.12
    assert config.use_fts is True
    assert config.use_vector is True
    sources = retrieve_kwargs.get("sources")
    assert DataSource.MEDIA_DB in sources
    assert DataSource.NOTES in sources

    generation_config = captured["generation_config"]
    assert generation_config["provider"] == "test-provider"
    assert generation_config["model"] == "explicit-model"
    assert generation_config["max_tokens"] == 256
    assert generation_config["prompt_template"] == "concise"
    assert generation_config["streaming"] is True


def test_rag_streaming_generation_provider_override(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    captured = {"generation_config": None}

    class StubRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, query, **kwargs):
            from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource

            return [
                Document(
                    id="doc-1",
                    content="Context content",
                    metadata={"title": "Doc"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    async def fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:
        captured["generation_config"] = context.config.get("generation")

        async def _gen():
            yield "chunk"

        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", StubRetriever)
    monkeypatch.setattr(rag_ep, "generate_streaming_response", fake_generate_streaming_response)

    payload = {
        "query": "Provider override parity",
        "search_mode": "hybrid",
        "sources": ["media_db"],
        "enable_generation": True,
        "generation_provider": "groq",
        "generation_model": "llama-3.3-70b-versatile",
    }

    with client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        next(resp.iter_lines(), None)

    generation_config = captured["generation_config"]
    assert generation_config is not None
    assert generation_config["provider"] == "groq"
    assert generation_config["model"] == "llama-3.3-70b-versatile"


def test_rag_streaming_generation_uses_shared_resolved_payload(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.app.core.RAG.rag_service.request_bundle import ResolvedRequestBundle
    from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
    from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import build_retrieval_plan

    captured = {"pipeline_kwargs": None, "generation_config": None}

    async def fake_unified_pipeline(**kwargs: Any) -> Any:
        from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

        captured["pipeline_kwargs"] = kwargs
        return rag_ep.UnifiedSearchResult(
            documents=[
                Document(
                    id="doc-1",
                    content="Resolved defaults context",
                    metadata={"title": "Doc"},
                    source=DataSource.MEDIA_DB,
                    score=0.8,
                )
            ],
            query=str(kwargs.get("query", "")),
            expanded_queries=[],
            metadata={},
            timings={},
            citations=[],
            feedback_id=None,
            generated_answer=None,
            cache_hit=False,
            errors=[],
            security_report=None,
            total_time=0.0,
        )

    async def fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        captured["generation_config"] = context.config.get("generation")

        async def _gen():
            yield "chunk"

        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    def _resolved_request_for_test(query: str) -> ResolvedRAGRequest:
        payload = {
            "query": query,
            "strategy": "standard",
            "sources": ["media_db"],
            "search_mode": "hybrid",
            "top_k": 4,
            "min_score": 0.25,
            "generation_prompt": "resolved-prompt",
            "max_generation_tokens": 777,
            "generation_model": "resolved-model",
            "generation_provider": "resolved-provider",
            "enable_generation": True,
            "index_namespace": "resolved-namespace",
            "user_id": "1",
            "feedback_user_id": "1",
        }
        return ResolvedRAGRequest(
            query=query,
            strategy="standard",
            payload=payload,
            index_namespace="resolved-namespace",
            rag_profile="fast",
            user_id="1",
            feedback_user_id="1",
        )

    def fake_build_standard_request_bundle(
        request: Any,
        *,
        current_user: Any,  # noqa: ARG001
        db_paths: dict[str, Any],
        media_db: Any,
        chacha_db: Any,
        prompts_db: Any = None,  # noqa: ARG001
    ) -> ResolvedRequestBundle:
        resolved_request = _resolved_request_for_test(request.query)
        retrieval_plan = build_retrieval_plan(resolved_request)
        pipeline_kwargs = rag_ep._build_unified_pipeline_kwargs(
            request=request,
            db_paths=db_paths,
            media_db=media_db,
            chacha_db=chacha_db,
            current_user=current_user,
            resolved_request=resolved_request,
            retrieval_plan=retrieval_plan,
        )
        return ResolvedRequestBundle(
            resolved_request=resolved_request,
            retrieval_plan=retrieval_plan,
            pipeline_kwargs=pipeline_kwargs,
        )

    monkeypatch.setattr(rag_ep, "_build_standard_request_bundle", fake_build_standard_request_bundle)
    monkeypatch.setattr(rag_ep, "unified_rag_pipeline", fake_unified_pipeline)
    monkeypatch.setattr(rag_ep, "generate_streaming_response", fake_generate_streaming_response)

    payload = {
        "query": "Shared contract generation defaults",
        "enable_generation": True,
    }

    with client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        next(resp.iter_lines(), None)

    pipeline_kwargs = captured["pipeline_kwargs"]
    assert pipeline_kwargs is not None
    assert pipeline_kwargs["index_namespace"] == "resolved-namespace"
    assert pipeline_kwargs["top_k"] == 4
    assert pipeline_kwargs["min_score"] == 0.25
    assert pipeline_kwargs["resolved_request"].index_namespace == "resolved-namespace"
    assert pipeline_kwargs["retrieval_plan"].index_namespace == "resolved-namespace"
    assert pipeline_kwargs["retrieval_plan"].top_k == 4
    assert pipeline_kwargs["retrieval_plan"].min_score == 0.25

    generation_config = captured["generation_config"]
    assert generation_config is not None
    assert generation_config["provider"] == "resolved-provider"
    assert generation_config["model"] == "resolved-model"
    assert generation_config["max_tokens"] == 777
    assert generation_config["prompt_template"] == "resolved-prompt"


def test_rag_streaming_emits_research_progress_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:
    import asyncio
    from types import SimpleNamespace

    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    async def _fake_unified_pipeline(**kwargs: Any) -> Any:
        callback = kwargs.get("research_progress_callback")
        if callback:
            await callback(SimpleNamespace(event_type="research_reasoning", data={"step": 1, "text": "plan"}))
            await asyncio.sleep(0)
            await callback(SimpleNamespace(event_type="research_searching", data={"queries": ["rag updates"]}))
            await asyncio.sleep(0)
            await callback(SimpleNamespace(event_type="research_results", data={"count": 1}))
            await asyncio.sleep(0)
            await callback(SimpleNamespace(event_type="research_complete", data={"total_iterations": 1}))
        return rag_ep.UnifiedSearchResult(
            documents=[
                Document(
                    id="doc-live-1",
                    content="RAG streaming context",
                    metadata={"title": "Live doc"},
                    source=DataSource.MEDIA_DB,
                    score=0.91,
                )
            ],
            query=str(kwargs.get("query", "")),
            expanded_queries=[],
            metadata={},
            timings={},
            citations=[],
            feedback_id=None,
            generated_answer=None,
            cache_hit=False,
            errors=[],
            security_report=None,
            total_time=0.0,
        )

    async def _fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:
        async def _gen():
            yield "stream token"

        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    monkeypatch.setattr(rag_ep, "unified_rag_pipeline", _fake_unified_pipeline)
    monkeypatch.setattr(rag_ep, "generate_streaming_response", _fake_generate_streaming_response)

    payload = {
        "query": "live research progress check",
        "enable_generation": True,
        "enable_research_progress": True,
    }

    events = []
    with client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        for raw in resp.iter_lines():
            if not raw:
                continue
            import json as _json

            evt = _json.loads(raw)
            events.append(evt)
            event_types = {item.get("type") for item in events}
            if {
                "research_reasoning",
                "research_searching",
                "research_results",
                "research_complete",
                "contexts",
                "delta",
            }.issubset(event_types):
                break

    types = [evt.get("type") for evt in events]
    assert "research_reasoning" in types
    assert "research_searching" in types
    assert "research_results" in types
    assert "research_complete" in types
    assert "contexts" in types
    assert "delta" in types
    assert types.index("research_reasoning") < types.index("contexts")
    assert types.index("research_complete") < types.index("delta")


def test_rag_streaming_preserves_delta_and_claim_events(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    async def _fake_unified_pipeline(**kwargs: Any) -> Any:
        return rag_ep.UnifiedSearchResult(
            documents=[
                Document(
                    id="doc-claims-1",
                    content="Claims context",
                    metadata={"title": "Claims doc"},
                    source=DataSource.MEDIA_DB,
                    score=0.88,
                )
            ],
            query=str(kwargs.get("query", "")),
            expanded_queries=[],
            metadata={},
            timings={},
            citations=[],
            feedback_id=None,
            generated_answer=None,
            cache_hit=False,
            errors=[],
            security_report=None,
            total_time=0.0,
        )

    async def _fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:
        context.metadata = {}

        async def _gen():
            context.metadata["claims_overlay"] = {"claim_count": 1, "supported": 1}
            yield "hello"
            context.metadata["claims_overlay"] = {"claim_count": 2, "supported": 2}
            yield " world"

        context.stream_generator = _gen()
        return context

    monkeypatch.setattr(rag_ep, "unified_rag_pipeline", _fake_unified_pipeline)
    monkeypatch.setattr(rag_ep, "generate_streaming_response", _fake_generate_streaming_response)

    payload = {
        "query": "claims stream check",
        "enable_generation": True,
        "enable_claims": True,
    }

    events = []
    with client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        for raw in resp.iter_lines():
            if not raw:
                continue
            import json as _json

            events.append(_json.loads(raw))

    delta_events = [evt for evt in events if evt.get("type") == "delta"]
    overlay_events = [evt for evt in events if evt.get("type") == "claims_overlay"]
    final_events = [evt for evt in events if evt.get("type") == "final_claims"]

    assert len(delta_events) == 2
    assert len(overlay_events) >= 2
    assert len(final_events) == 1
    assert final_events[0].get("claim_count") == 2


def test_rag_streaming_profile_defaults_affect_generation_config(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    captured = {"generation_config": None}

    class StubRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, query, **kwargs):
            from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource

            return [
                Document(
                    id="doc-1",
                    content="Context content",
                    metadata={"title": "Doc"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    async def fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:
        captured["generation_config"] = context.config.get("generation")

        async def _gen():
            yield "chunk"

        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", StubRetriever)
    monkeypatch.setattr(rag_ep, "generate_streaming_response", fake_generate_streaming_response)

    payload = {
        "query": "Profile streaming parity",
        "search_mode": "hybrid",
        "sources": ["media_db"],
        "enable_generation": True,
        "generation_model": "explicit-model",
        "rag_profile": "balanced",
    }

    with client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        next(resp.iter_lines(), None)

    generation_config = captured["generation_config"]
    assert generation_config is not None
    assert generation_config["max_tokens"] == 1000
    assert generation_config["prompt_template"] == "multi_hop_compact"


def test_rag_streaming_profile_fast_applies_instruction_prompt(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    captured = {"generation_config": None}

    class StubRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, query, **kwargs):
            from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource

            return [
                Document(
                    id="doc-1",
                    content="Context content",
                    metadata={"title": "Doc"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    async def fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:
        captured["generation_config"] = context.config.get("generation")

        async def _gen():
            yield "chunk"

        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", StubRetriever)
    monkeypatch.setattr(rag_ep, "generate_streaming_response", fake_generate_streaming_response)

    payload = {
        "query": "Fast profile streaming parity",
        "search_mode": "hybrid",
        "sources": ["media_db"],
        "enable_generation": True,
        "rag_profile": "fast",
    }

    with client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        next(resp.iter_lines(), None)

    generation_config = captured["generation_config"]
    assert generation_config is not None
    assert generation_config["max_tokens"] == 440
    assert generation_config["prompt_template"] == "instruction_tuned"


def test_rag_streaming_agentic_path_uses_profile_resolved_defaults(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:
    from types import SimpleNamespace

    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    captured: dict[str, Any] = {"agentic_kwargs": None}

    class StubRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.retrievers = {}

        async def retrieve(self, query: str, **kwargs: Any) -> list[Document]:
            return [
                Document(
                    id="doc-1",
                    content="Context content",
                    metadata={"title": "Doc"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    async def fake_agentic_rag_pipeline(**kwargs: Any) -> Any:
        captured["agentic_kwargs"] = kwargs
        return SimpleNamespace(
            documents=[
                Document(
                    id="agentic-doc-1",
                    content="Agentic context",
                    metadata={"title": "Agentic"},
                    source=DataSource.MEDIA_DB,
                    score=0.95,
                )
            ],
            metadata={"agentic_metrics": {"steps": 1}, "provenance": []},
        )

    async def fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:
        async def _gen():
            yield "chunk"

        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", StubRetriever)
    monkeypatch.setattr(rag_ep, "agentic_rag_pipeline", fake_agentic_rag_pipeline)
    monkeypatch.setattr(rag_ep, "generate_streaming_response", fake_generate_streaming_response)

    payload = {
        "query": "Agentic profile defaults parity",
        "strategy": "agentic",
        "rag_profile": "fast",
        "enable_generation": True,
    }

    with client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        next(resp.iter_lines(), None)

    agentic_kwargs = captured["agentic_kwargs"]
    assert agentic_kwargs is not None
    assert agentic_kwargs["top_k"] == 6


def test_rag_streaming_agentic_path_uses_bundle_contracts_without_re_resolving(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
) -> None:
    from types import SimpleNamespace

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.app.core.RAG.rag_service.request_bundle import ResolvedRequestBundle
    from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
    from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    canonical_resolved = ResolvedRAGRequest(
        query="bundle agentic query",
        strategy="agentic",
        payload={
            "query": "bundle agentic query",
            "strategy": "agentic",
            "sources": ["notes"],
            "search_mode": "vector",
            "top_k": 5,
            "min_score": 0.17,
            "debug_mode": True,
            "index_namespace": "bundle-tenant",
            "user_id": "1",
            "feedback_user_id": "1",
        },
        index_namespace="bundle-tenant",
        rag_profile="balanced",
        user_id="1",
        feedback_user_id="1",
    )
    canonical_plan = RetrievalPlan(
        query="bundle agentic query",
        sources=("notes",),
        search_mode="vector",
        top_k=5,
        min_score=0.17,
        index_namespace="bundle-tenant",
    )
    bundle = ResolvedRequestBundle(
        resolved_request=canonical_resolved,
        retrieval_plan=canonical_plan,
        pipeline_kwargs={
            "query": canonical_resolved.query,
            "sources": list(canonical_plan.sources),
            "search_mode": canonical_plan.search_mode,
            "top_k": canonical_plan.top_k,
            "min_score": canonical_plan.min_score,
            "index_namespace": canonical_plan.index_namespace,
            "media_db_path": "stub_media.db",
            "notes_db_path": "stub_chacha.db",
            "character_db_path": "stub_chacha.db",
            "kanban_db_path": None,
            "enable_generation": True,
            "resolved_request": canonical_resolved,
            "retrieval_plan": canonical_plan,
            "user_id": "1",
            "feedback_user_id": "1",
        },
    )

    captured: dict[str, Any] = {
        "agentic_kwargs": None,
        "context_builder_calls": 0,
    }

    def fake_build_standard_request_bundle(*args: Any, **kwargs: Any) -> ResolvedRequestBundle:  # noqa: ARG001
        return bundle

    def fake_build_agentic_execution_context(*, resolved_request, retrieval_plan, payload_override=None):  # noqa: ANN001
        captured["context_builder_calls"] = int(captured["context_builder_calls"]) + 1
        captured["context_builder_resolved_request"] = resolved_request
        captured["context_builder_retrieval_plan"] = retrieval_plan
        payload = dict(payload_override or resolved_request.payload)
        payload["agentic_enable_tools"] = True
        payload["agentic_max_tool_calls"] = 9
        payload["agentic_coverage_target"] = 0.88
        return (
            payload,
            rag_ep.AgenticConfig(
                top_k_docs=3,
                enable_tools=True,
                max_tool_calls=9,
                coverage_target=0.88,
            ),
        )

    async def fake_unified_rag_pipeline(**kwargs: Any) -> Any:  # noqa: ARG001
        return rag_ep.UnifiedSearchResult(
            documents=[
                Document(
                    id="doc-prefetch-1",
                    content="prefetched",
                    metadata={"title": "Prefetch"},
                    source=DataSource.NOTES,
                    score=0.9,
                )
            ],
            query=canonical_resolved.query,
            expanded_queries=[],
            metadata={},
            timings={},
            citations=[],
            feedback_id=None,
            generated_answer=None,
            cache_hit=False,
            errors=[],
            security_report=None,
            total_time=0.0,
        )

    async def fake_agentic_rag_pipeline(**kwargs: Any) -> Any:
        captured["agentic_kwargs"] = kwargs
        return SimpleNamespace(
            documents=[
                Document(
                    id="agentic-doc",
                    content="agentic",
                    metadata={"title": "Agentic"},
                    source=DataSource.NOTES,
                    score=0.95,
                )
            ],
            metadata={"agentic_metrics": {"steps": 1}, "provenance": []},
        )

    async def fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        async def _gen():
            yield "chunk"

        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    monkeypatch.setattr(rag_ep, "_build_standard_request_bundle", fake_build_standard_request_bundle)
    monkeypatch.setattr(rag_ep, "build_agentic_execution_context", fake_build_agentic_execution_context)
    monkeypatch.setattr(rag_ep, "unified_rag_pipeline", fake_unified_rag_pipeline)
    monkeypatch.setattr(rag_ep, "agentic_rag_pipeline", fake_agentic_rag_pipeline)
    monkeypatch.setattr(rag_ep, "generate_streaming_response", fake_generate_streaming_response)

    payload = {
        "query": "stream via bundle",
        "strategy": "agentic",
        "enable_generation": True,
    }

    with client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp:
        assert resp.status_code == 200
        next(resp.iter_lines(), None)

    assert captured["context_builder_calls"] == 1
    agentic_kwargs = captured["agentic_kwargs"]
    assert agentic_kwargs is not None
    assert captured["context_builder_resolved_request"] is canonical_resolved
    assert captured["context_builder_retrieval_plan"] is canonical_plan
    assert agentic_kwargs["resolved_request"] is canonical_resolved
    assert agentic_kwargs["retrieval_plan"] is canonical_plan
    assert agentic_kwargs["query"] == "bundle agentic query"
    assert agentic_kwargs["sources"] == ["notes"]
    assert agentic_kwargs["search_mode"] == "vector"
    assert agentic_kwargs["top_k"] == 5
    assert agentic_kwargs["min_score"] == 0.17
    assert agentic_kwargs["index_namespace"] == "bundle-tenant"
    assert agentic_kwargs["agentic"].enable_tools is True
    assert agentic_kwargs["agentic"].max_tool_calls == 9
    assert agentic_kwargs["agentic"].coverage_target == 0.88
