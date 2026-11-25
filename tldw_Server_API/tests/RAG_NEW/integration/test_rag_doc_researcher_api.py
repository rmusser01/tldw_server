import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.core.DB_Management.media_db.media_database import (
    MediaDatabase,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification
from tldw_Server_API.app.core.RAG.rag_service.research_agent import ResearchOutput


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")


@pytest.fixture()
def client_with_overrides(monkeypatch, auth_headers):
    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _noop():
        return None

    # Disable RBAC enforcement to avoid DB access
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
        else:
            with TestClient(fastapi_app, headers=auth_headers, raise_server_exceptions=False) as client:
                yield client
    finally:
        fastapi_app.dependency_overrides.clear()


def test_rag_search_doc_researcher_flags(client_with_overrides, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult

    captured: dict[str, object] = {}

    async def _fake_unified_pipeline(rcfg):  # noqa: ARG001
        kwargs = rcfg.as_kwargs()
        captured.update(kwargs)
        return UnifiedSearchResult(
            documents=[],
            query=str(kwargs.get("query", "")),
            expanded_queries=[],
            metadata={
                "granularity_routing": {"enabled": True},
                "evidence_accumulation": {"enabled": True},
                "evidence_chains": {"total_chains": 1},
            },
            timings={},
            citations=[],
            feedback_id=None,
            generated_answer=None,
            cache_hit=False,
            errors=[],
            security_report=None,
            total_time=0.0,
        )

    monkeypatch.setattr("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.run_rag_pipeline", _fake_unified_pipeline)

    payload = {
        "query": "alpha beta gamma",
        "sources": ["media_db", "notes"],
        "search_mode": "fts",
        "enable_dynamic_granularity": True,
        "enable_evidence_accumulation": True,
        "accumulation_max_rounds": 2,
        "accumulation_time_budget_sec": 1.0,
        "enable_evidence_chains": True,
        "enable_cache": False,
        "enable_generation": False,
        "enable_reranking": False,
    }

    resp = client_with_overrides.post("/api/v1/rag/search", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    metadata = data.get("metadata", {})

    assert captured.get("enable_dynamic_granularity") is True
    assert captured.get("enable_evidence_accumulation") is True
    assert captured.get("accumulation_max_rounds") == 2
    assert captured.get("enable_evidence_chains") is True
    assert "granularity_routing" in metadata
    assert "evidence_accumulation" in metadata
    assert "evidence_chains" in metadata


def test_rag_search_natural_language_media_fallback(
    client_with_overrides,
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user as _get_media_db

    db_path = tmp_path / "media.db"
    media_db = MediaDatabase(db_path=str(db_path), client_id="route-fallback")
    media_db.initialize_db()

    try:
        media_id, _, _ = media_db.add_media_with_keywords(
            title="Transcript of the duel",
            media_type="transcript",
            content=(
                "This transcript states the exact fallback phrase weakness frieza saiyans "
                "without adding any chunk rows."
            ),
            keywords=["frieza", "saiyans", "weakness"],
        )

        async def _override_media_db():
            return media_db

        monkeypatch.setitem(fastapi_app.dependency_overrides, _get_media_db, _override_media_db)

        resp = client_with_overrides.post(
            "/api/v1/rag/search",
            json={
                "query": "What weakness does Frieza mention about the Saiyans during the fight?",
                "sources": ["media_db"],
                "search_mode": "fts",
                "fts_level": "chunk",
                "enable_query_classification": False,
                "enable_research_loop": False,
                "enable_generation": False,
                "enable_cache": False,
                "enable_reranking": False,
            },
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        docs = data.get("documents", [])

        assert docs, data
        assert any(doc.get("metadata", {}).get("media_id") == str(media_id) for doc in docs)
        assert any(doc.get("metadata", {}).get("retrieval_mode") == "late_chunk" for doc in docs)
        assert any("weakness frieza saiyans" in (doc.get("content") or "").lower() for doc in docs)
        assert all(doc.get("metadata", {}).get("source") == "media_db" for doc in docs)
    finally:
        media_db.close_connection()


def test_rag_search_endpoint_skip_search_bypasses_retrieval(client_with_overrides, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    classification = QueryClassification(
        skip_search=True,
        search_local_db=False,
        search_web=False,
        search_academic=False,
        search_discussions=False,
        standalone_query="hello",
        detected_intent="conversational",
        confidence=0.99,
        reasoning="Greeting should not trigger retrieval",
    )

    async def _fake_classifier(**kwargs):  # noqa: ARG001
        return classification

    class GuardRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise AssertionError("Retriever should not be constructed for skip_search=true")

    monkeypatch.setattr(up, "classify_and_reformulate", _fake_classifier)
    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", GuardRetriever)
    monkeypatch.setattr(up, "MultiDatabaseRetriever", GuardRetriever)

    payload = {
        "query": "hello",
        "sources": ["media_db"],
        "search_mode": "fts",
        "enable_query_classification": True,
        "enable_generation": False,
        "enable_cache": True,
        "enable_reranking": False,
    }

    resp = client_with_overrides.post("/api/v1/rag/search", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    metadata = data.get("metadata", {})
    assert metadata.get("classification_skip_search") is True
    assert metadata.get("retrieval_bypassed", {}).get("reason") == "classification_skip_search"
    assert data.get("documents") == []


def test_rag_search_endpoint_research_loop_preserves_docs(client_with_overrides, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    classification = QueryClassification(
        skip_search=False,
        search_local_db=True,
        search_web=True,
        search_academic=False,
        search_discussions=False,
        standalone_query="latest python release notes",
        detected_intent="factual",
        confidence=0.85,
        reasoning="Needs iterative research",
    )
    research_output = ResearchOutput(
        query="latest python release notes",
        standalone_query="latest python release notes",
        all_results=[
            {
                "id": "research-doc-ep-1",
                "title": "Python Release Notes",
                "url": "https://example.com/python-release-notes",
                "content": "Python 3.x release notes and highlights.",
                "source": "web",
                "score": 0.91,
            }
        ],
        total_iterations=1,
        total_results=1,
        total_duration_sec=0.05,
        final_reasoning="Enough evidence collected",
        completed=True,
    )

    async def _fake_classifier(**kwargs):  # noqa: ARG001
        return classification

    async def _fake_research_loop(**kwargs):  # noqa: ARG001
        return research_output

    class GuardRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise AssertionError("Retriever should not be constructed after successful research loop")

    monkeypatch.setattr(up, "classify_and_reformulate", _fake_classifier)
    monkeypatch.setattr(up, "research_loop", _fake_research_loop)
    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", GuardRetriever)
    monkeypatch.setattr(up, "MultiDatabaseRetriever", GuardRetriever)

    payload = {
        "query": "latest python release notes",
        "sources": ["media_db"],
        "search_mode": "hybrid",
        "enable_query_classification": True,
        "enable_research_loop": True,
        "search_depth_mode": "balanced",
        "enable_generation": False,
        "enable_cache": False,
        "enable_reranking": False,
    }

    resp = client_with_overrides.post("/api/v1/rag/search", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    metadata = data.get("metadata", {})
    docs = data.get("documents", [])

    assert metadata.get("retrieval_bypassed", {}).get("reason") == "research_loop"
    assert metadata.get("research", {}).get("total_results") == 1
    assert data.get("research_summary", {}).get("total_results") == 1
    assert isinstance(docs, list) and len(docs) == 1
    assert docs[0].get("id") == "research-doc-ep-1"


def test_rag_search_endpoint_research_loop_controls_forwarded(client_with_overrides, monkeypatch):
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    classification = QueryClassification(
        skip_search=False,
        search_local_db=True,
        search_web=True,
        search_academic=False,
        search_discussions=True,
        standalone_query="community sentiment on retrieval",
        detected_intent="exploratory",
        confidence=0.8,
        reasoning="Needs mixed search",
    )
    research_output = ResearchOutput(
        query="community sentiment on retrieval",
        standalone_query="community sentiment on retrieval",
        all_results=[],
        total_iterations=2,
        total_results=0,
        total_duration_sec=0.02,
        final_reasoning="Done",
        completed=True,
    )

    captured: dict[str, object] = {}

    async def _fake_classifier(**kwargs):  # noqa: ARG001
        return classification

    async def _fake_research_loop(**kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return research_output

    monkeypatch.setattr(up, "classify_and_reformulate", _fake_classifier)
    monkeypatch.setattr(up, "research_loop", _fake_research_loop)

    payload = {
        "query": "community sentiment on retrieval",
        "sources": ["media_db"],
        "search_mode": "hybrid",
        "enable_query_classification": True,
        "enable_research_loop": True,
        "search_depth_mode": "balanced",
        "research_max_iterations": 4,
        "discussion_platforms": ["reddit"],
        "search_url_scraping": False,
        "enable_generation": False,
        "enable_cache": False,
        "enable_reranking": False,
    }

    resp = client_with_overrides.post("/api/v1/rag/search", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    metadata = data.get("metadata", {})

    assert captured.get("max_iterations") == 4
    assert captured.get("discussion_platforms") == ["reddit"]
    assert captured.get("enable_url_scraping") is False
    registry = captured.get("registry")
    assert registry is not None
    assert registry.get("scrape_url") is None
    assert metadata.get("research", {}).get("max_iterations_requested") == 4


def test_rag_search_endpoint_applies_search_agent_config_defaults(client_with_overrides, monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult

    captured: dict[str, object] = {}

    async def _fake_unified_pipeline(rcfg):  # noqa: ARG001
        kwargs = rcfg.as_kwargs()
        captured.update(kwargs)
        return UnifiedSearchResult(
            documents=[],
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

    def _fake_get_config_value(section: str, key: str, default=None, reload: bool = False):  # noqa: ANN001, ARG001
        if section != "Search-Agent":
            return default
        values = {
            "search_query_classification": "true",
            "search_default_mode": "quality",
            "search_query_reformulation": "false",
            "search_research_loop": "true",
            "search_discussions_enabled": "true",
            "search_discussion_platforms": "reddit,stackoverflow",
            "search_progress_streaming": "true",
            "search_url_scraping": "false",
            "search_classifier_provider": "openai",
            "search_classifier_model": "gpt-4o-mini",
            "search_max_iterations_speed": "4",
            "search_max_iterations_balanced": "8",
            "search_max_iterations_quality": "16",
        }
        return values.get(key, default)

    monkeypatch.setattr("tldw_Server_API.app.core.config.get_config_value", _fake_get_config_value)
    monkeypatch.setattr("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.run_rag_pipeline", _fake_unified_pipeline)

    for env_key in (
        "SEARCH_QUERY_CLASSIFICATION",
        "SEARCH_DEFAULT_MODE",
        "SEARCH_QUERY_REFORMULATION",
        "SEARCH_RESEARCH_LOOP",
        "SEARCH_DISCUSSIONS_ENABLED",
        "SEARCH_DISCUSSION_PLATFORMS",
        "SEARCH_PROGRESS_STREAMING",
        "SEARCH_URL_SCRAPING",
        "SEARCH_CLASSIFIER_PROVIDER",
        "SEARCH_CLASSIFIER_MODEL",
        "SEARCH_MAX_ITERATIONS_SPEED",
        "SEARCH_MAX_ITERATIONS_BALANCED",
        "SEARCH_MAX_ITERATIONS_QUALITY",
    ):
        monkeypatch.delenv(env_key, raising=False)

    resp = client_with_overrides.post(
        "/api/v1/rag/search",
        json={
            "query": "config default routing",
            "enable_generation": False,
            "enable_cache": False,
            "enable_reranking": False,
        },
    )
    assert resp.status_code == 200, resp.text

    assert captured.get("enable_query_classification") is True
    assert captured.get("search_depth_mode") == "quality"
    assert captured.get("enable_query_reformulation") is False
    assert captured.get("enable_research_loop") is True
    assert captured.get("enable_discussion_search") is True
    assert captured.get("discussion_platforms") == ["reddit", "stackoverflow"]
    assert captured.get("enable_research_progress") is True
    assert captured.get("search_url_scraping") is False
    assert captured.get("classifier_provider") == "openai"
    assert captured.get("classifier_model") == "gpt-4o-mini"
    assert captured.get("research_max_iterations_speed") == 4
    assert captured.get("research_max_iterations_balanced") == 8
    assert captured.get("research_max_iterations_quality") == 16


def test_rag_search_endpoint_rejects_research_loop_without_classification(client_with_overrides):
    resp = client_with_overrides.post(
        "/api/v1/rag/search",
        json={
            "query": "invalid contract",
            "enable_research_loop": True,
            "enable_generation": False,
        },
    )
    assert resp.status_code == 422
