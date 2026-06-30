from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.services import web_scraping_service as ws_service
from tldw_Server_API.app.services.ephemeral_store import EphemeralStorage


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


def _force_fallback(monkeypatch):
    def _raise():
        raise RuntimeError("enhanced service unavailable in test")

    monkeypatch.setattr(ws_service, "get_web_scraping_service", _raise, raising=True)


@pytest.fixture(autouse=True)
def _enable_legacy_web_scraping_fallback(monkeypatch):
    monkeypatch.setenv("TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK", "1")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_recursive_rejects_unsupported_controls(monkeypatch):
    _force_fallback(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await ws_service.process_web_scraping_task(
            **_base_kwargs(
                scrape_method="Recursive Scraping",
                crawl_strategy="best_first",
                include_external=True,
            )
        )

    assert exc_info.value.status_code == 400
    detail = str(exc_info.value.detail)
    assert "legacy fallback for 'Recursive Scraping'" in detail
    assert "crawl_strategy" in detail
    assert "include_external" in detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_url_level_rejects_threshold_control(monkeypatch):
    _force_fallback(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await ws_service.process_web_scraping_task(
            **_base_kwargs(
                scrape_method="URL Level",
                url_level=2,
                score_threshold=0.25,
            )
        )

    assert exc_info.value.status_code == 400
    detail = str(exc_info.value.detail)
    assert "legacy fallback for 'URL Level'" in detail
    assert "score_threshold" in detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_url_level_ephemeral_smoke_applies_max_pages_cap(monkeypatch):
    _force_fallback(monkeypatch)

    async def fake_scrape_by_url_level(base_url, level):
        return [
            {"url": "https://example.com/1", "title": "A", "content": "c1", "extraction_successful": True},
            {"url": "https://example.com/2", "title": "B", "content": "c2", "extraction_successful": True},
            {"url": "https://example.com/3", "title": "C", "content": "c3", "extraction_successful": True},
        ]

    async def fake_to_thread(func, *args, **kwargs):
        return await fake_scrape_by_url_level(*args, **kwargs)

    monkeypatch.setattr(ws_service, "scrape_by_url_level", lambda *_a, **_k: [], raising=True)
    monkeypatch.setattr(ws_service.asyncio, "to_thread", fake_to_thread, raising=True)
    monkeypatch.setattr(
        ws_service.ephemeral_storage,
        "store_data",
        lambda data: "ephemeral-fallback-id",
        raising=True,
    )

    result = await ws_service.process_web_scraping_task(
        **_base_kwargs(
            scrape_method="URL Level",
            url_level=2,
            max_pages=2,
            mode="ephemeral",
        )
    )

    assert result["status"] == "ephemeral-ok"
    assert result["engine"] == "legacy_fallback"
    assert result["total_articles"] == 2
    assert len(result["results"]) == 2
    fallback_context = result["fallback_context"]
    assert fallback_context["enabled"] is True
    assert fallback_context["trigger_error_type"] == "RuntimeError"
    assert "max_pages" in fallback_context["degraded_controls_applied"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_url_level_sanitizes_unexpected_runtime_error(monkeypatch):
    _force_fallback(monkeypatch)

    async def raise_legacy_scraper_error(*args, **kwargs):
        raise RuntimeError("legacy scraper leaked /private/web/cache/token")

    monkeypatch.setattr(ws_service.asyncio, "to_thread", raise_legacy_scraper_error, raising=True)

    with pytest.raises(HTTPException) as exc_info:
        await ws_service.process_web_scraping_task(
            **_base_kwargs(
                scrape_method="URL Level",
                url_level=2,
                mode="ephemeral",
            )
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Legacy web scraping fallback failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_ephemeral_result_retrievable_within_ttl(monkeypatch):
    _force_fallback(monkeypatch)

    async def fake_scrape_by_url_level(base_url, level):
        return [
            {"url": "https://example.com/a", "title": "A", "content": "c1", "extraction_successful": True},
            {"url": "https://example.com/b", "title": "B", "content": "c2", "extraction_successful": True},
        ]

    async def fake_to_thread(func, *args, **kwargs):
        return await fake_scrape_by_url_level(*args, **kwargs)

    state = {"now": 100.0}

    def fake_clock() -> float:
        return float(state["now"])

    local_store = EphemeralStorage(default_ttl_seconds=120, max_entries=16, clock=fake_clock)

    monkeypatch.setattr(ws_service, "scrape_by_url_level", lambda *_a, **_k: [], raising=True)
    monkeypatch.setattr(ws_service.asyncio, "to_thread", fake_to_thread, raising=True)
    monkeypatch.setattr(ws_service, "ephemeral_storage", local_store, raising=True)

    result = await ws_service.process_web_scraping_task(
        **_base_kwargs(
            scrape_method="URL Level",
            url_level=2,
            max_pages=2,
            mode="ephemeral",
        )
    )

    assert result["status"] == "ephemeral-ok"
    ephemeral_id = result["media_id"]
    stored = local_store.get_data(ephemeral_id)
    assert isinstance(stored, dict)
    assert len(stored.get("articles", [])) == 2

    state["now"] = 300.0
    assert local_store.get_data(ephemeral_id) is None


class _FakeDB:
    def __init__(self):
        self.calls = []
        self.closed = False

    def add_media_with_keywords(self, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls)
        return idx, f"uuid-{idx}", "ok"

    def close_connection(self):
        self.closed = True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_persist_smoke_includes_rollout_metadata(monkeypatch, tmp_path):
    _force_fallback(monkeypatch)
    fake_db = _FakeDB()
    fake_repo = _FakeDB()

    @contextmanager
    def _fake_managed_media_database(*args, **kwargs):  # noqa: ARG001
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    async def fake_scrape_and_summarize_multiple(**kwargs):
        return [
            {
                "url": "https://example.com/a",
                "title": "A",
                "author": "Author",
                "content": "content",
                "summary": "summary",
                "extraction_successful": True,
            }
        ]

    monkeypatch.setattr(
        ws_service,
        "scrape_and_summarize_multiple",
        fake_scrape_and_summarize_multiple,
        raising=True,
    )
    monkeypatch.setattr(
        ws_service,
        "managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )
    monkeypatch.setattr(
        ws_service,
        "create_media_database",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy raw factory should not be used")),
        raising=False,
    )
    monkeypatch.setattr(
        ws_service,
        "get_user_media_db_path",
        lambda _user_id: str(tmp_path / "fallback-media.db"),
        raising=True,
    )
    monkeypatch.setattr(
        ws_service,
        "get_media_repository",
        lambda db: fake_repo,
        raising=False,
    )

    result = await ws_service.process_web_scraping_task(
        **_base_kwargs(
            scrape_method="Individual URLs",
            url_input="https://example.com/a",
            mode="persist",
            user_id=1,
        )
    )

    assert result["status"] == "persist-ok"
    assert result["engine"] == "legacy_fallback"
    assert result["total_articles"] == 1
    assert result["media_ids"] == [1]
    assert fake_db.closed is True
    assert len(fake_repo.calls) == 1
    fallback_context = result["fallback_context"]
    assert fallback_context["enabled"] is True
    assert fallback_context["trigger_error_type"] == "RuntimeError"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_persist_uses_media_repository_api(monkeypatch, tmp_path):
    _force_fallback(monkeypatch)

    class _FakeDbNoLegacyInsert:
        def __init__(self):
            self.closed = False

        def close_connection(self):
            self.closed = True

    class _FakeRepo:
        def __init__(self):
            self.calls = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 71, "repo-71", "stored"

    fake_db = _FakeDbNoLegacyInsert()
    fake_repo = _FakeRepo()

    @contextmanager
    def _fake_managed_media_database(*args, **kwargs):  # noqa: ARG001
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    async def fake_scrape_and_summarize_multiple(**kwargs):
        return [
            {
                "url": "https://example.com/repo",
                "title": "Repo Title",
                "author": "Author",
                "content": "content",
                "summary": "summary",
                "extraction_successful": True,
            }
        ]

    monkeypatch.setattr(
        ws_service,
        "scrape_and_summarize_multiple",
        fake_scrape_and_summarize_multiple,
        raising=True,
    )
    monkeypatch.setattr(
        ws_service,
        "managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )
    monkeypatch.setattr(
        ws_service,
        "create_media_database",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy raw factory should not be used")),
        raising=False,
    )
    monkeypatch.setattr(
        ws_service,
        "get_user_media_db_path",
        lambda _user_id: str(tmp_path / "fallback-media.db"),
        raising=True,
    )
    monkeypatch.setattr(
        ws_service,
        "get_media_repository",
        lambda db: fake_repo,
        raising=False,
    )

    result = await ws_service.process_web_scraping_task(
        **_base_kwargs(
            scrape_method="Individual URLs",
            url_input="https://example.com/repo",
            mode="persist",
            user_id=1,
        )
    )

    assert result["status"] == "persist-ok"
    assert result["media_ids"] == [71]
    assert fake_db.closed is True
    assert len(fake_repo.calls) == 1
    assert fake_repo.calls[0]["url"] == "https://example.com/repo"
    assert fake_repo.calls[0]["title"] == "Repo Title"
    assert fake_repo.calls[0]["media_type"] == "web_document"
    assert fake_repo.calls[0]["content"] == "content"
    assert fake_repo.calls[0]["analysis_content"] == "summary"
    assert fake_repo.calls[0]["transcription_model"] == "web-scraping-import"


@pytest.mark.unit
def test_process_web_scraping_endpoint_fallback_contract_error_for_url_level_controls(
    client_user_only, monkeypatch
):
    _force_fallback(monkeypatch)

    payload = {
        "scrape_method": "URL Level",
        "url_input": "https://example.com",
        "url_level": 2,
        "max_pages": 5,
        "mode": "ephemeral",
        "score_threshold": 0.3,
    }
    response = client_user_only.post("/api/v1/media/process-web-scraping", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert "legacy fallback for 'URL Level'" in body.get("detail", "")


@pytest.mark.unit
def test_process_web_scraping_endpoint_rejects_when_legacy_fallback_disabled(
    client_user_only, monkeypatch
):
    _force_fallback(monkeypatch)
    monkeypatch.setenv("TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK", "0")

    async def _fake_to_thread(func, *args, **kwargs):
        _ = (func, args, kwargs)
        return [
            {
                "url": "https://example.com",
                "title": "Example",
                "content": "content",
                "extraction_successful": True,
            }
        ]

    monkeypatch.setattr(ws_service, "scrape_by_url_level", lambda *_a, **_k: [], raising=True)
    monkeypatch.setattr(ws_service.asyncio, "to_thread", _fake_to_thread, raising=True)
    monkeypatch.setattr(
        ws_service.ephemeral_storage,
        "store_data",
        lambda data: "ephemeral-fallback-id",
        raising=True,
    )

    payload = {
        "scrape_method": "URL Level",
        "url_input": "https://example.com",
        "url_level": 2,
        "max_pages": 1,
        "mode": "ephemeral",
    }
    response = client_user_only.post("/api/v1/media/process-web-scraping", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert "deprecated" in str(body.get("detail", "")).lower()
