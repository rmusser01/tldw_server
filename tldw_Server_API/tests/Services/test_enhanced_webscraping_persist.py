import asyncio
from pathlib import Path

import pytest

from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_with_mocked_db_path(tmp_path: Path, monkeypatch):
    # Arrange: route user DB path to a temp file
    db_file = tmp_path / "media_test.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Patch the function used by the service module to resolve DB path
    import tldw_Server_API.app.services.enhanced_web_scraping_service as svc_mod

    def _fake_get_user_media_db_path(user_id: int):
        return str(db_file)

    monkeypatch.setattr(svc_mod, "get_user_media_db_path", _fake_get_user_media_db_path)

    # Prepare a small batch of articles (simulate successful scraping)
    result = {
        "method": "Individual URLs",
        "articles": [
            {
                "url": "https://example.com/a",
                "title": "Article A",
                "author": "Alice",
                "date": "2024-10-01",
                "content": "Hello world from article A.",
                "extraction_successful": True,
                "summary": "Summary A",
                "method": "enhanced",
            },
            {
                "url": "https://example.com/b",
                "title": "Article B",
                "author": "Bob",
                "date": "2024-10-02",
                "content": "Hello world from article B.",
                "extraction_successful": True,
                "summary": "Summary B",
                "method": "enhanced",
            },
        ],
    }

    svc = WebScrapingService()

    # Act: directly exercise the persistence path
    res = await svc._store_persistent(
        result=result,
        keywords="foo,bar",
        user_id=7,
        perform_chunking=False,
        chunking_mode=None,
        auto_chunking_goal="balanced",
        auto_chunking_use_llm=False,
    )

    # Assert: verify status and number of stored articles
    assert res["status"] == "persist-ok"
    assert res["total_articles"] == 2
    assert isinstance(res.get("media_ids"), list)
    assert len(res["media_ids"]) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_sanitizes_storage_failure(
    tmp_path: Path,
    monkeypatch,
):
    import tldw_Server_API.app.services.enhanced_web_scraping_service as svc_mod

    class _FailingDB:
        def add_media_with_keywords(self, **_kwargs):
            raise RuntimeError("storage backend exploded at /private/db/media.sqlite")

    class _FailingDBContext:
        def __enter__(self):
            return _FailingDB()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        svc_mod,
        "get_user_media_db_path",
        lambda _user_id: str(tmp_path / "media_test.db"),
    )
    monkeypatch.setattr(
        svc_mod,
        "managed_media_database",
        lambda **_kwargs: _FailingDBContext(),
    )

    result = {
        "method": "Individual URLs",
        "articles": [
            {
                "url": "https://example.com/a",
                "title": "Article A",
                "author": "Alice",
                "date": "2024-10-01",
                "content": "Hello world from article A.",
                "extraction_successful": True,
                "summary": "Summary A",
                "method": "enhanced",
            },
        ],
    }

    res = await WebScrapingService()._store_persistent(
        result=result,
        keywords="foo,bar",
        user_id=7,
        perform_chunking=False,
        chunking_mode=None,
        auto_chunking_goal="balanced",
        auto_chunking_use_llm=False,
    )

    assert res["status"] == "persist-ok"
    assert res["media_ids"] == []
    assert res["errors"] == ["Storage failed for article"]
    assert "storage backend exploded" not in str(res)
    assert "/private/db/media.sqlite" not in str(res)
