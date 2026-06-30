from __future__ import annotations

import json
import os
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.services import web_scraping_service as ws_service
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


class _FakeDB:
    def __init__(self):
        self.calls: list[dict] = []
        self.closed = False

    def add_media_with_keywords(self, **kwargs):
        self.calls.append(kwargs)
        return len(self.calls), f"uuid-{len(self.calls)}", "ok"

    def close_connection(self):
        self.closed = True


def _force_fallback(monkeypatch):
    def _raise():
        raise RuntimeError("enhanced service unavailable in test")

    monkeypatch.setattr(ws_service, "get_web_scraping_service", _raise, raising=True)


def test_web_chunking_forms_preserve_request_llm_selection():
    legacy_form = ws_service._web_chunking_form(
        perform_chunking=True,
        chunking_mode="auto",
        auto_chunking_goal="balanced",
        auto_chunking_use_llm=True,
        api_name="openai/gpt-4o",
    )
    enhanced_form = enhanced_ws_service._web_chunking_form(
        perform_chunking=True,
        chunking_mode="auto",
        auto_chunking_goal="balanced",
        auto_chunking_use_llm=True,
        api_name="openai/gpt-4o",
    )

    assert legacy_form.api_name == "openai/gpt-4o"
    assert enhanced_form.api_name == "openai/gpt-4o"


@pytest.fixture(autouse=True)
def _enable_legacy_web_scraping_fallback(monkeypatch):
    monkeypatch.setenv("TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK", "1")


@pytest.mark.asyncio
async def test_web_scraping_fallback_persists_auto_chunking_plan(monkeypatch):
    _force_fallback(monkeypatch)
    fake_db = _FakeDB()
    chunk_calls: list[dict] = []

    async def _fake_scrape_and_summarize_multiple(**_kwargs):
        return [
            {
                "url": "https://example.com/article",
                "title": "Article",
                "author": "Author",
                "content": "# Intro\n\n- one\n- two",
                "summary": "summary",
                "extraction_successful": True,
            }
        ]

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    class _Chunker:
        def chunk_text_hierarchical_flat(self, text, **kwargs):
            chunk_calls.append({"text": text, **kwargs})
            return [
                {
                    "text": text,
                    "metadata": {"paragraph_kind": "paragraph"},
                }
            ]

    monkeypatch.setattr(
        ws_service,
        "scrape_and_summarize_multiple",
        _fake_scrape_and_summarize_multiple,
        raising=True,
    )
    monkeypatch.setattr(ws_service, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(ws_service, "get_user_media_db_path", lambda _user_id: "/tmp/media.db")  # nosec B108
    monkeypatch.setattr(ws_service, "get_media_repository", lambda db: db)
    monkeypatch.setattr(ws_service, "Chunker", _Chunker)

    result = await ws_service.process_web_scraping_task(
        scrape_method="Individual URLs",
        url_input="https://example.com/article",
        url_level=None,
        max_pages=1,
        max_depth=1,
        summarize_checkbox=False,
        custom_prompt=None,
        api_name=None,
        api_key=None,
        keywords="",
        custom_titles=None,
        system_prompt=None,
        temperature=0.7,
        custom_cookies=None,
        mode="persist",
        user_id=1,
        user_agent=None,
        custom_headers=None,
        crawl_strategy=None,
        include_external=None,
        score_threshold=None,
        perform_chunking=True,
        chunking_mode="auto",
        auto_chunking_goal="navigation_summary",
        auto_chunking_use_llm=True,
    )

    assert result["status"] == "persist-ok"
    assert fake_db.closed is True
    assert chunk_calls[0]["method"] == "structure_aware"
    safe_metadata = json.loads(fake_db.calls[0]["safe_metadata"])
    plan = safe_metadata["chunking_plan"]
    assert plan["mode"] == "auto"
    assert plan["goal"] == "navigation_summary"
    assert plan["method"] == "structure_aware"
    assert "ai_assist_unavailable" in plan["fallback_reason"]


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
