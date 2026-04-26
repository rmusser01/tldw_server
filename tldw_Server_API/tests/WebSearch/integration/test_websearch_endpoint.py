"""
Integration test for /api/v1/research/websearch endpoint.
Allows 200 or 500 due to environment/network variability.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Disable rate limiting before importing app (keeps tests permissive)
os.environ["TEST_MODE"] = "true"

from fastapi import FastAPI
from tldw_Server_API.app.api.v1.endpoints.research import router as research_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

# Minimal app to avoid importing full server (audio limiter issue)
mini_app = FastAPI()
mini_app.include_router(research_router, prefix="/api/v1/research")


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user():
    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)
    mini_app.dependency_overrides[get_request_user] = override_user
    with TestClient(mini_app) as client:
        yield client
    mini_app.dependency_overrides.clear()


def test_websearch_minimal(client_with_user: TestClient):
    client = client_with_user
    payload = {
        "query": "what is the capital of france",
        "engine": "duckduckgo",
        "result_count": 5,
        "aggregate": False,
    }
    resp = client.post("/api/v1/research/websearch", json=payload)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "web_search_results_dict" in data
        assert "sub_query_dict" in data


def test_websearch_aggregate_path(client_with_user: TestClient, monkeypatch: pytest.MonkeyPatch):
    from tldw_Server_API.app.api.v1.endpoints import research as research_module

    def fake_generate_and_search(question, params):
        return {
            "web_search_results_dict": {"results": [], "total_results_found": 0},
            "sub_query_dict": {"main_goal": question, "sub_questions": []},
        }

    async def fake_analyze_and_aggregate(web_results_dict, sub_query_dict, search_params, **_):
        assert search_params.get("relevance_analysis_llm") == "openai"
        return {
            "final_answer": {
                "text": "Paris is the capital of France.",
                "evidence": [
                    {"content": "Paris is the capital of France", "reasoning": "Geography"}
                ],
                "confidence": 0.85,
                "chunks": [],
            },
            "relevant_results": {"0": {"content": "Paris", "reasoning": "Geography"}},
            "web_search_results_dict": web_results_dict,
        }

    monkeypatch.setattr(research_module, "generate_and_search", fake_generate_and_search)
    monkeypatch.setattr(research_module, "analyze_and_aggregate", fake_analyze_and_aggregate)

    client = client_with_user
    payload = {
        "query": "what is the capital of france",
        "engine": "google",
        "result_count": 3,
        "aggregate": True,
        "final_answer_llm": "openai",
    }

    resp = client.post("/api/v1/research/websearch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_answer"]["text"].startswith("Paris")
    assert isinstance(data["final_answer"]["confidence"], float)
    assert 0.0 <= data["final_answer"]["confidence"] <= 1.0
    assert data["final_answer"]["evidence"]
    assert "chunks" in data["final_answer"]


def test_websearch_aggregate_requires_llm_config(client_with_user: TestClient):
    client = client_with_user
    payload = {
        "query": "what is the capital of france",
        "engine": "google",
        "result_count": 3,
        "aggregate": True,
    }

    resp = client.post("/api/v1/research/websearch", json=payload)
    assert resp.status_code == 422
    detail = resp.json().get("detail", "")
    assert "aggregate=true requires" in detail


def test_websearch_sanitizes_generic_failure(
    client_with_user: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import research as research_module

    def fake_generate_and_search(question, params):
        raise RuntimeError("search backend exploded")

    monkeypatch.setattr(research_module, "generate_and_search", fake_generate_and_search)

    resp = client_with_user.post(
        "/api/v1/research/websearch",
        json={
            "query": "what is the capital of france",
            "engine": "google",
            "result_count": 3,
            "aggregate": False,
        },
    )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Websearch failed"


def test_websearch_fatal_phase1_error_returns_502(
    client_with_user: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import research as research_module

    def fake_generate_and_search(question, params):
        return {
            "web_search_results_dict": {
                "results": [],
                "total_results_found": 0,
                "error": "provider failed",
            },
            "sub_query_dict": {"main_goal": question, "sub_questions": []},
        }

    monkeypatch.setattr(research_module, "generate_and_search", fake_generate_and_search)

    resp = client_with_user.post(
        "/api/v1/research/websearch",
        json={
            "query": "what is the capital of france",
            "engine": "google",
            "result_count": 3,
            "aggregate": False,
        },
    )

    assert resp.status_code == 502
    assert resp.json()["detail"] == "Websearch failed"
    assert "provider failed" not in resp.text


def test_websearch_aggregate_fatal_phase1_error_returns_502(
    client_with_user: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import research as research_module

    def fake_generate_and_search(question, params):
        return {
            "web_search_results_dict": {
                "results": [],
                "total_results_found": 0,
                "error": "provider failed",
            },
            "sub_query_dict": {"main_goal": question, "sub_questions": []},
        }

    async def fake_analyze_and_aggregate(*args, **kwargs):
        raise AssertionError("aggregate path should short-circuit on fatal phase1 error")

    monkeypatch.setattr(research_module, "generate_and_search", fake_generate_and_search)
    monkeypatch.setattr(research_module, "analyze_and_aggregate", fake_analyze_and_aggregate)

    resp = client_with_user.post(
        "/api/v1/research/websearch",
        json={
            "query": "what is the capital of france",
            "engine": "google",
            "result_count": 3,
            "aggregate": True,
            "final_answer_llm": "openai",
        },
    )

    assert resp.status_code == 502
    assert resp.json()["detail"] == "Websearch failed"
    assert "provider failed" not in resp.text


def test_websearch_partial_provider_warning_with_results_returns_200(
    client_with_user: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import research as research_module

    def fake_generate_and_search(question, params):
        return {
            "web_search_results_dict": {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com/result",
                        "content": "A result survived",
                    }
                ],
                "total_results_found": 1,
                "warnings": [
                    {"phase": "provider", "query": question, "message": "secondary provider failed"}
                ],
                "error": None,
            },
            "sub_query_dict": {"main_goal": question, "sub_questions": []},
        }

    monkeypatch.setattr(research_module, "generate_and_search", fake_generate_and_search)

    resp = client_with_user.post(
        "/api/v1/research/websearch",
        json={
            "query": "what is the capital of france",
            "engine": "google",
            "result_count": 3,
            "aggregate": False,
        },
    )

    assert resp.status_code == 200
    payload = resp.json()["web_search_results_dict"]
    assert len(payload["results"]) == 1
    assert payload["warnings"][0]["message"] == "secondary provider failed"
