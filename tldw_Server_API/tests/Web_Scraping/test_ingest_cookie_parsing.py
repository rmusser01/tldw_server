from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import tldw_Server_API.app.services.web_scraping_service as web_scraping_service
from tldw_Server_API.app.api.v1.schemas.media_request_models import ScrapeMethod


class _DummyUsageLog:
    def log_event(self, *args, **kwargs):
        return None


def _request(**overrides):
    payload = {
        "scrape_method": ScrapeMethod.INDIVIDUAL,
        "urls": ["https://example.com/article"],
        "titles": [],
        "authors": [],
        "keywords": [],
        "use_cookies": True,
        "cookies": None,
        "perform_analysis": False,
        "perform_rolling_summarization": False,
        "perform_confabulation_check_of_analysis": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_orchestrate_accepts_cookie_dict(monkeypatch):
    captured = {}

    async def fake_scrape_article(url, custom_cookies=None, *, allow_llm_extraction=None):
        captured["cookies"] = custom_cookies
        captured["allow_llm_extraction"] = allow_llm_extraction
        return {
            "url": url,
            "content": "content",
            "extraction_successful": True,
        }

    monkeypatch.setattr(
        web_scraping_service,
        "scrape_article",
        fake_scrape_article,
        raising=True,
    )

    result = await web_scraping_service.ingest_web_content_orchestrate(
        _request(cookies={"name": "session", "value": "abc"}),
        db=SimpleNamespace(client_id="1"),
        usage_log=_DummyUsageLog(),
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert captured.get("cookies") == [{"name": "session", "value": "abc"}]
    assert captured["allow_llm_extraction"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_orchestrate_accepts_cookie_list(monkeypatch):
    captured = {}

    async def fake_scrape_article(url, custom_cookies=None, *, allow_llm_extraction=None):
        captured["cookies"] = custom_cookies
        captured["allow_llm_extraction"] = allow_llm_extraction
        return {
            "url": url,
            "content": "content",
            "extraction_successful": True,
        }

    monkeypatch.setattr(
        web_scraping_service,
        "scrape_article",
        fake_scrape_article,
        raising=True,
    )

    cookie_payload = [
        {"name": "session", "value": "abc"},
        {"name": "token", "value": "def"},
    ]

    result = await web_scraping_service.ingest_web_content_orchestrate(
        _request(cookies=cookie_payload),
        db=SimpleNamespace(client_id="1"),
        usage_log=_DummyUsageLog(),
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert captured.get("cookies") == cookie_payload
    assert captured["allow_llm_extraction"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_orchestrate_invalid_cookie_object_type_returns_400():
    with pytest.raises(HTTPException) as excinfo:
        await web_scraping_service.ingest_web_content_orchestrate(
            _request(cookies=123),
            db=SimpleNamespace(client_id="1"),
            usage_log=_DummyUsageLog(),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "Invalid cookies format"
