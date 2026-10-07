from __future__ import annotations

import os

import pytest

from tldw_Server_API.app.services import enhanced_web_scraping_service as enhanced_ws_service


pytestmark = pytest.mark.unit


class _ClientWithToken:
    def __init__(self, client):
        self._client = client

    def post(self, *args, **kwargs):
        default_headers = dict(getattr(self._client, "headers", {}) or {})
        request_headers = kwargs.pop("headers", {}) or {}
        default_headers.update(request_headers)
        default_headers.setdefault("token", "test-token")
        default_headers.setdefault(
            "X-API-KEY",
            os.getenv("SINGLE_USER_API_KEY", "test-api-key-12345"),
        )
        return self._client.post(*args, headers=default_headers, **kwargs)


def test_web_chunking_forms_preserve_request_llm_selection():
    enhanced_form = enhanced_ws_service._web_chunking_form(
        perform_chunking=True,
        chunking_mode="auto",
        auto_chunking_goal="balanced",
        auto_chunking_use_llm=True,
        api_name="openai/gpt-4o",
    )

    assert enhanced_form.api_name == "openai/gpt-4o"


def test_process_web_scraping_endpoint_forwards_auto_chunking_fields(
    client_user_only,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints.media import (
        process_web_scraping as endpoint_mod,
    )

    captured: dict[str, object] = {}

    async def _fake_task(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "results": []}

    monkeypatch.setattr(endpoint_mod, "_resolve_process_web_scraping_task", lambda: _fake_task)

    response = client_user_only.post(
        "/api/v1/media/process-web-scraping",
        json={
            "scrape_method": "Individual URLs",
            "url_input": "https://example.com",
            "mode": "ephemeral",
            "perform_chunking": True,
            "chunking_mode": "auto",
            "auto_chunking_goal": "qa_search",
            "auto_chunking_use_llm": True,
        },
    )

    assert response.status_code == 200, response.text
    assert captured["perform_chunking"] is True
    assert captured["chunking_mode"] == "auto"
    assert captured["auto_chunking_goal"] == "qa_search"
    assert captured["auto_chunking_use_llm"] is True


def test_ingest_web_content_auto_chunking_adds_plan_metadata(
    client_user_only,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints.media import (
        ingest_web_content as endpoint_mod,
    )

    async def _fake_orchestrate(**_kwargs):
        return [
            {
                "url": "https://example.com/article",
                "title": "Article",
                "content": "# Intro\n\nArticle body.",
                "metadata": {"source": "test"},
            }
        ]

    monkeypatch.setattr(endpoint_mod, "ingest_web_content_orchestrate", _fake_orchestrate)

    response = _ClientWithToken(client_user_only).post(
        "/api/v1/media/ingest-web-content",
        json={
            "urls": ["https://example.com/article"],
            "scrape_method": "individual",
            "perform_chunking": True,
            "chunking_mode": "auto",
            "auto_chunking_goal": "balanced",
            "auto_chunking_use_llm": True,
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    plan = result["metadata"]["chunking_plan"]
    assert plan["mode"] == "auto"
    assert plan["method"] == "structure_aware"
    assert "ai_assist_unavailable" in plan["fallback_reason"]
