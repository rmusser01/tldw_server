from __future__ import annotations

import pytest
from fastapi import HTTPException

import tldw_Server_API.app.services.enhanced_web_scraping_service as enhanced_svc_mod
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService


def _cfg(
    *,
    max_pages: int = 42,
    strategy: str = "default",
    include_external: bool = True,
    score_threshold: float = 0.33,
) -> dict[str, object]:
    return {
        "web_scraper": {
            "web_crawl_max_pages": max_pages,
            "web_crawl_strategy": strategy,
            "web_crawl_include_external": include_external,
            "web_crawl_score_threshold": score_threshold,
        }
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enhanced_service_uses_config_defaults_when_request_values_omitted(monkeypatch):
    svc = WebScrapingService()
    svc._initialized = True
    captured: dict[str, object] = {}

    async def fake_scrape_recursive(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"method": "Recursive Scraping", "articles": []}

    async def fake_store_ephemeral(result, task_id, user_id):
        return result

    monkeypatch.setattr(enhanced_svc_mod, "load_and_log_configs", lambda: _cfg())
    monkeypatch.setattr(svc, "_scrape_recursive", fake_scrape_recursive)
    monkeypatch.setattr(svc, "_store_ephemeral", fake_store_ephemeral)

    result = await svc.process_web_scraping_task(
        scrape_method="Recursive Scraping",
        url_input="https://example.com",
        max_pages=None,
        max_depth=2,
        mode="ephemeral",
        crawl_strategy=None,
        include_external=None,
        score_threshold=None,
    )

    args = captured.get("args")
    kwargs = captured.get("kwargs")
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[1] == 42
    assert kwargs.get("crawl_strategy") == "default"
    assert kwargs.get("include_external") is True
    assert kwargs.get("score_threshold") == pytest.approx(0.33, abs=1e-9)

    crawl_cfg = result.get("crawl_config", {})
    assert crawl_cfg.get("max_pages_source") == "config_default"
    assert crawl_cfg.get("requested_max_pages") is None
    assert crawl_cfg.get("effective_max_pages") == 42
    assert crawl_cfg.get("strategy_source") == "config_default"
    assert crawl_cfg.get("include_external_source") == "config_default"
    assert crawl_cfg.get("score_threshold_source") == "config_default"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enhanced_service_request_values_override_config(monkeypatch):
    svc = WebScrapingService()
    svc._initialized = True
    captured: dict[str, object] = {}

    async def fake_scrape_recursive(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"method": "Recursive Scraping", "articles": []}

    async def fake_store_ephemeral(result, task_id, user_id):
        return result

    monkeypatch.setattr(enhanced_svc_mod, "load_and_log_configs", lambda: _cfg(max_pages=99, strategy="default"))
    monkeypatch.setattr(svc, "_scrape_recursive", fake_scrape_recursive)
    monkeypatch.setattr(svc, "_store_ephemeral", fake_store_ephemeral)

    result = await svc.process_web_scraping_task(
        scrape_method="Recursive Scraping",
        url_input="https://example.com",
        max_pages=7,
        max_depth=2,
        mode="ephemeral",
        crawl_strategy="best_first",
        include_external=False,
        score_threshold=0.15,
    )

    args = captured.get("args")
    kwargs = captured.get("kwargs")
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    assert args[1] == 7
    assert kwargs.get("crawl_strategy") == "best_first"
    assert kwargs.get("include_external") is False
    assert kwargs.get("score_threshold") == pytest.approx(0.15, abs=1e-9)

    crawl_cfg = result.get("crawl_config", {})
    assert crawl_cfg.get("max_pages_source") == "request"
    assert crawl_cfg.get("requested_max_pages") == 7
    assert crawl_cfg.get("effective_max_pages") == 7
    assert crawl_cfg.get("strategy_source") == "request"
    assert crawl_cfg.get("include_external_source") == "request"
    assert crawl_cfg.get("score_threshold_source") == "request"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enhanced_service_sanitizes_unexpected_runtime_error(monkeypatch):
    svc = WebScrapingService()
    svc._initialized = True

    async def raise_scraper_error(*args, **kwargs):
        raise RuntimeError("enhanced scraper leaked /private/web/cache/token")

    monkeypatch.setattr(enhanced_svc_mod, "load_and_log_configs", lambda: _cfg())
    monkeypatch.setattr(svc, "_scrape_by_url_level", raise_scraper_error)

    with pytest.raises(HTTPException) as exc_info:
        await svc.process_web_scraping_task(
            scrape_method="URL Level",
            url_input="https://example.com",
            url_level=2,
            mode="ephemeral",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Enhanced web scraping task failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_max_pages_100_is_treated_as_request_override(monkeypatch):
    svc = WebScrapingService()
    svc._initialized = True
    captured: dict[str, object] = {}

    async def fake_scrape_recursive(*args, **kwargs):
        captured["args"] = args
        return {"method": "Recursive Scraping", "articles": []}

    async def fake_store_ephemeral(result, task_id, user_id):
        return result

    monkeypatch.setattr(enhanced_svc_mod, "load_and_log_configs", lambda: _cfg(max_pages=12))
    monkeypatch.setattr(svc, "_scrape_recursive", fake_scrape_recursive)
    monkeypatch.setattr(svc, "_store_ephemeral", fake_store_ephemeral)

    result = await svc.process_web_scraping_task(
        scrape_method="Recursive Scraping",
        url_input="https://example.com",
        max_pages=100,
        max_depth=2,
        mode="ephemeral",
    )

    args = captured.get("args")
    assert isinstance(args, tuple)
    assert args[1] == 100
    crawl_cfg = result.get("crawl_config", {})
    assert crawl_cfg.get("max_pages_source") == "request"
    assert crawl_cfg.get("effective_max_pages") == 100


@pytest.mark.unit
def test_process_web_scraping_endpoint_omitted_max_pages_forwards_none(
    client_user_only, monkeypatch
):
    import tldw_Server_API.app.api.v1.endpoints.media as media_mod

    captured = {}

    async def fake_process_web_scraping_task(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "results": []}

    monkeypatch.setattr(
        media_mod,
        "process_web_scraping_task",
        fake_process_web_scraping_task,
        raising=True,
    )

    payload = {
        "scrape_method": "Recursive Scraping",
        "url_input": "https://example.com",
        "max_depth": 2,
        "mode": "ephemeral",
    }
    response = client_user_only.post("/api/v1/media/process-web-scraping", json=payload)
    assert response.status_code == 200
    assert captured.get("max_pages") is None
