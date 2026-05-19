"""
Integration tests for /api/v1/research/websearch with specific engines.
Monkeypatch provider layer to avoid network.
"""
import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

# Disable rate limiting and heavy imports
os.environ["TEST_MODE"] = "true"


def _mini_app_with_user():
    from tldw_Server_API.app.api.v1.endpoints import research as research_module
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

    app = FastAPI()
    app.include_router(research_module.router, prefix="/api/v1/research")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    return app


def test_websearch_searx_engine(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "searx"
        return {
            "results": [
                {
                    "title": "Searx Result",
                    "url": "https://searx.example/result",
                    "content": "An example.",
                    "metadata": {"date_published": None},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.01,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={"query": "q", "engine": "searx", "result_count": 3, "aggregate": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "web_search_results_dict" in data
        assert data["web_search_results_dict"]["results"]


def test_websearch_searxng_engine_alias(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "searx"
        return {
            "results": [
                {
                    "title": "Searx Alias Result",
                    "url": "https://searx.example/alias",
                    "content": "Alias handling.",
                    "metadata": {"date_published": None},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.01,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={"query": "q", "engine": "searxng", "result_count": 3, "aggregate": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "web_search_results_dict" in data
        assert data["web_search_results_dict"]["results"]


def test_websearch_tavily_engine(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "tavily"
        return {
            "results": [
                {
                    "title": "Tavily Result",
                    "url": "https://tavily.example/article",
                    "content": "Content.",
                    "metadata": {"date_published": "2024-01-01"},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.02,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={"query": "q", "engine": "tavily", "result_count": 3, "aggregate": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "web_search_results_dict" in data
        assert data["web_search_results_dict"]["results"]


def test_websearch_kagi_engine(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "kagi"
        return {
            "results": [
                {
                    "title": "Kagi Result",
                    "url": "https://kagi.example/article",
                    "content": "Kagi content.",
                    "metadata": {"date_published": None},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.01,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={"query": "q", "engine": "kagi", "result_count": 2, "aggregate": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "web_search_results_dict" in data
        assert data["web_search_results_dict"]["results"]


def test_websearch_exa_engine(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "exa"
        return {
            "results": [
                {
                    "title": "Exa Result",
                    "url": "https://exa.example/article",
                    "content": "Exa content.",
                    "metadata": {"date_published": "2024-01-01"},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.01,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={"query": "q", "engine": "exa", "result_count": 2, "aggregate": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "web_search_results_dict" in data
        assert data["web_search_results_dict"]["results"]


def test_websearch_firecrawl_engine(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "firecrawl"
        return {
            "results": [
                {
                    "title": "Firecrawl Result",
                    "url": "https://firecrawl.example/article",
                    "content": "Firecrawl content.",
                    "metadata": {"date_published": "2024-02-02"},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.02,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={"query": "q", "engine": "firecrawl", "result_count": 2, "aggregate": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "web_search_results_dict" in data
        assert data["web_search_results_dict"]["results"]


def test_websearch_serper_engine(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "serper"
        return {
            "results": [
                {
                    "title": "Serper Result",
                    "url": "https://serper.example/article",
                    "content": "Serper content.",
                    "metadata": {"date_published": "2024-03-03"},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.02,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={"query": "q", "engine": "serper", "result_count": 2, "aggregate": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "web_search_results_dict" in data
        assert data["web_search_results_dict"]["results"]


def test_websearch_4chan_engine(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    captured: dict[str, object] = {}

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "4chan"
        captured["search_params"] = kwargs.get("search_params")
        return {
            "results": [
                {
                    "title": "/g/ Thread 123",
                    "url": "https://boards.4chan.org/g/thread/123",
                    "content": "Thread snippet.",
                    "metadata": {"date_published": "2026-02-08T00:00:00Z"},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.02,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={
                "query": "rust memory safety",
                "engine": "4chan",
                "result_count": 2,
                "boards": ["g", "tv"],
                "max_threads_per_board": 120,
                "max_archived_threads_per_board": 40,
                "include_archived": True,
                "aggregate": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "web_search_results_dict" in data
        assert data["web_search_results_dict"]["results"]

    search_params = captured.get("search_params")
    assert isinstance(search_params, dict)
    assert search_params.get("boards") == ["g", "tv"]
    assert search_params.get("max_threads_per_board") == 120
    assert search_params.get("max_archived_threads_per_board") == 40
    assert search_params.get("include_archived") is True


def test_websearch_4chan_engine_omitted_max_archived_threads_is_backward_compatible(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    captured: dict[str, object] = {}

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "4chan"
        captured["search_params"] = kwargs.get("search_params")
        return {
            "results": [
                {
                    "title": "/g/ Thread 999",
                    "url": "https://boards.4chan.org/g/thread/999",
                    "content": "Thread snippet.",
                    "metadata": {"date_published": "2026-02-08T00:00:00Z"},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.02,
            "processing_error": None,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={
                "query": "rust memory safety",
                "engine": "4chan",
                "result_count": 2,
                "boards": ["g"],
                "max_threads_per_board": 120,
                "include_archived": True,
                "aggregate": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "web_search_results_dict" in data
        assert data["web_search_results_dict"]["results"]

    search_params = captured.get("search_params")
    assert isinstance(search_params, dict)
    assert "max_archived_threads_per_board" in search_params
    assert search_params.get("max_archived_threads_per_board") is None


def test_websearch_4chan_engine_surfaces_all_board_failure_diagnostics(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "4chan"
        return {
            "results": [],
            "total_results_found": 0,
            "search_time": 0.02,
            "warnings": [
                {"board": "g", "phase": "catalog", "message": "catalog timeout"},
                {"board": "tv", "phase": "catalog", "message": "catalog timeout"},
            ],
            "error": "4chan search failed for all requested boards.",
            "processing_error": None,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={
                "query": "rust memory safety",
                "engine": "4chan",
                "result_count": 5,
                "boards": ["g", "tv"],
                "include_archived": False,
                "aggregate": False,
            },
        )
        assert resp.status_code == 502
        assert "4chan search failed for all requested boards." in resp.json()["detail"]


def test_websearch_4chan_engine_surfaces_partial_failure_warnings(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

    def fake_perform_websearch(search_engine, search_query, *args, **kwargs):
        assert search_engine == "4chan"
        return {
            "results": [
                {
                    "title": "Rust thread",
                    "url": "https://boards.4chan.org/tv/thread/90210",
                    "content": "Thread from tv board",
                    "metadata": {"date_published": "2026-02-08T00:00:00Z"},
                }
            ],
            "total_results_found": 1,
            "search_time": 0.02,
            "warnings": [
                {"board": "g", "phase": "catalog", "message": "catalog timeout"},
            ],
            "processing_error": None,
        }

    monkeypatch.setattr(ws, "perform_websearch", fake_perform_websearch)

    app = _mini_app_with_user()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/research/websearch",
            json={
                "query": "rust memory safety",
                "engine": "4chan",
                "result_count": 5,
                "boards": ["g", "tv"],
                "include_archived": False,
                "aggregate": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        payload = data["web_search_results_dict"]
        assert payload["results"]
        assert payload.get("error") is None
        assert isinstance(payload.get("warnings"), list)
        assert any(
            warning.get("board") == "g" and warning.get("phase") == "catalog"
            for warning in payload["warnings"]
            if isinstance(warning, dict)
        )
