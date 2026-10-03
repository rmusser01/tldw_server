from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.media_request_models import IngestWebContentRequest
from tldw_Server_API.app.services import web_scraping_service as ws_service


def _base_kwargs(**overrides):
    payload = {
        "scrape_method": "Recursive Scraping",
        "url_input": "https://example.com",
        "url_level": None,
        "max_pages": 5,
        "max_depth": 2,
        "summarize_checkbox": False,
        "custom_prompt": None,
        "api_name": None,
        "api_key": None,
        "keywords": "",
        "custom_titles": None,
        "system_prompt": None,
        "temperature": 0.7,
        "custom_cookies": None,
        "mode": "ephemeral",
        "user_id": 1,
        "user_agent": None,
        "custom_headers": None,
        "crawl_strategy": None,
        "include_external": None,
        "score_threshold": None,
    }
    payload.update(overrides)
    return payload


def _force_enhanced_failure(monkeypatch):
    def _raise():
        raise RuntimeError("enhanced service unavailable in test")

    monkeypatch.setattr(ws_service, "get_web_scraping_service", _raise, raising=True)


class _UsageLog:
    def log_event(self, *args, **kwargs):
        pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enhanced_service_failure_propagates_without_fallback(monkeypatch):
    """The legacy fallback was removed; enhanced service failures must propagate."""
    _force_enhanced_failure(monkeypatch)

    with pytest.raises(RuntimeError, match="enhanced service unavailable in test"):
        await ws_service.process_web_scraping_task(**_base_kwargs())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validation_errors_raise_before_enhanced_dispatch(monkeypatch):
    """Pre-dispatch validation is unchanged by the fallback removal."""
    enhanced_calls = []

    class _EnhancedService:
        async def process_web_scraping_task(self, **kwargs):
            enhanced_calls.append(kwargs)
            return {"status": "ok"}

    monkeypatch.setattr(
        ws_service,
        "get_web_scraping_service",
        lambda: _EnhancedService(),
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await ws_service.process_web_scraping_task(
            **_base_kwargs(crawl_strategy="not-a-strategy")
        )

    assert exc_info.value.status_code == 400
    assert enhanced_calls == []


@pytest.mark.unit
def test_legacy_fallback_env_flag_is_no_longer_referenced():
    """TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK must not resurrect the fallback."""
    source = Path(ws_service.__file__).read_text(encoding="utf-8")
    assert "TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK" not in source  # nosec B101
    assert "legacy_fallback" not in source  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_web_content_skips_analysis_without_provider(monkeypatch):
    captured = {}

    async def fake_scrape_article(url, custom_cookies=None, *, allow_llm_extraction=None):
        captured["allow_llm_extraction"] = allow_llm_extraction
        return {
            "url": url,
            "title": "Example",
            "content": "article body",
            "extraction_successful": True,
        }

    def fail_analyze(**kwargs):
        raise AssertionError("analysis should not run without a provider")

    monkeypatch.setattr(ws_service, "scrape_article", fake_scrape_article, raising=True)
    monkeypatch.setattr(ws_service, "analyze", fail_analyze, raising=True)

    result = await ws_service.ingest_web_content_orchestrate(
        request=IngestWebContentRequest(
            urls=["https://example.com/a"],
            perform_analysis=True,
            api_name=None,
        ),
        db=object(),
        usage_log=_UsageLog(),
    )

    assert result is not None
    assert result[0]["analysis"] is None
    assert captured["allow_llm_extraction"] is True
    assert result[0]["analysis_status"] == "skipped"
    assert result[0]["analysis_error"] == "Choose an analysis provider before running ingest analysis."


@pytest.mark.unit
def test_process_web_scraping_endpoint_returns_500_when_enhanced_service_fails(
    client_user_only, monkeypatch
):
    _force_enhanced_failure(monkeypatch)

    payload = {
        "scrape_method": "URL Level",
        "url_input": "https://example.com",
        "url_level": 2,
        "max_pages": 1,
        "mode": "ephemeral",
    }
    response = client_user_only.post("/api/v1/media/process-web-scraping", json=payload)
    assert response.status_code == 500
    assert response.json().get("detail") == "Web scraping failed due to an internal error."
