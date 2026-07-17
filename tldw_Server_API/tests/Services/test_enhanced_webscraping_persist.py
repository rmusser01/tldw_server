from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

import tldw_Server_API.app.services.enhanced_web_scraping_service as svc_mod
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService


def _article(
    url: str = "https://example.com/a",
    title: str = "Article A",
    **overrides: Any,
) -> dict[str, Any]:
    article = {
        "url": url,
        "title": title,
        "author": "Alice",
        "date": "2024-10-01",
        "content": "Extracted article content.",
        "extraction_successful": True,
        "summary": "Article summary",
        "method": "enhanced",
    }
    article.update(overrides)
    return article


async def _persist(
    articles: list[dict[str, Any]],
    *,
    service: WebScrapingService | None = None,
    keywords: str = "",
) -> dict[str, Any]:
    return await (service or WebScrapingService())._store_persistent(
        result={"method": "Individual URLs", "articles": articles},
        keywords=keywords,
        user_id=7,
        perform_chunking=False,
        chunking_mode=None,
        auto_chunking_goal="balanced",
        auto_chunking_use_llm=False,
    )


def _patch_db(monkeypatch: pytest.MonkeyPatch, *responses: Any) -> None:
    pending = list(responses)

    class _StaticDB:
        def add_media_with_keywords(self, **_kwargs: Any) -> tuple[Any, Any, Any]:
            response = pending.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response

    class _StaticDBContext:
        def __enter__(self) -> _StaticDB:
            return _StaticDB()

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    monkeypatch.setattr(svc_mod, "get_user_media_db_path", lambda _user_id: "unused.db")
    monkeypatch.setattr(svc_mod, "managed_media_database", lambda **_kwargs: _StaticDBContext())


def _patch_real_db_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        svc_mod,
        "get_user_media_db_path",
        lambda _user_id: str(tmp_path / "media_test.db"),
    )


@contextmanager
def _captured_logs() -> Iterator[list[str]]:
    records: list[str] = []
    sink_id = logger.add(records.append, level="DEBUG", format="{message}")
    try:
        yield records
    finally:
        logger.remove(sink_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_stores_successful_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_real_db_path(monkeypatch, tmp_path)

    response = await _persist(
        [
            _article(),
            _article(
                "https://example.com/b",
                "Article B",
                author="Bob",
                content="Second extracted article.",
            ),
        ],
        keywords="foo,bar",
    )

    assert response["status"] == "persist-ok"
    assert response["total_articles"] == 2
    assert len(response["media_ids"]) == 2
    assert response["stored_articles"] == 2
    assert response["errors"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_repeated_real_repository_write_is_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_real_db_path(monkeypatch, tmp_path)
    article = _article(
        "https://example.com/repeated",
        "Repeated Article",
        content="The same persisted article content.",
    )
    service = WebScrapingService()

    first = await _persist([article], service=service, keywords="foo,bar")
    second = await _persist([article], service=service, keywords="foo,bar")

    assert first["stored_articles"] == 1
    assert second["status"] == "duplicate"
    assert second["media_ids"] == []
    assert second["stored_articles"] == 0
    assert second["skipped_articles"] == 1
    assert second["duplicate_articles"] == 1
    assert second["errors"] is None


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "duplicate_fields",
    [{"is_duplicate": True}, {"error_code": "duplicate_content"}],
    ids=["duplicate-flag", "duplicate-error-code"],
)
async def test_enhanced_webscraping_persist_uses_structured_extraction_duplicate_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duplicate_fields: dict[str, Any],
):
    _patch_real_db_path(monkeypatch, tmp_path)

    response = await _persist(
        [
            _article(
                "https://example.com/a?repeat=1",
                extraction_successful=False,
                content="",
                error="Extraction skipped",
                **duplicate_fields,
            )
        ]
    )

    assert response["status"] == "duplicate"
    assert response["media_ids"] == []
    assert response["stored_articles"] == 0
    assert response["skipped_articles"] == 1
    assert response["duplicate_articles"] == 1
    assert response["errors"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_does_not_infer_duplicate_from_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_real_db_path(monkeypatch, tmp_path)

    response = await _persist(
        [
            _article(
                "https://example.com/unavailable",
                extraction_successful=False,
                content="",
                error="deduplicate worker unavailable",
            )
        ]
    )

    assert response["status"] == "persist-ok"
    assert response["duplicate_articles"] == 0
    assert response["errors"] == ["Failed to extract: https://example.com/unavailable"]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Media 'Article A' already exists. Overwrite not enabled.",
        "Media 'Article A' already exists (concurrent insert). Overwrite not enabled.",
    ],
)
async def test_enhanced_webscraping_persist_recognizes_exact_repository_duplicate_messages(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
):
    _patch_db(monkeypatch, (42, "existing-uuid", message))

    response = await _persist([_article(content="Existing content")])

    assert response["status"] == "duplicate"
    assert response["media_ids"] == []
    assert response["stored_articles"] == 0
    assert response["skipped_articles"] == 1
    assert response["duplicate_articles"] == 1
    assert response["errors"] is None


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Media 'Article A' URL canonicalized.",
        "Media 'Article A' already exists (handled concurrent insert).",
        "Media 'Other Article' already exists. Overwrite not enabled.",
    ],
)
async def test_enhanced_webscraping_persist_does_not_expand_repository_duplicate_messages(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
):
    _patch_db(monkeypatch, (42, "stored-uuid", message))

    response = await _persist([_article(content="Stored content")])

    assert response["status"] == "persist-ok"
    assert response["media_ids"] == [42]
    assert response["stored_articles"] == 1
    assert response["skipped_articles"] == 0
    assert response["duplicate_articles"] == 0
    assert response["errors"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_mixed_duplicate_and_failure_retains_error(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_db(
        monkeypatch,
        (42, "existing-uuid", "Media 'Article A' already exists. Overwrite not enabled."),
    )

    response = await _persist(
        [
            _article(content="Existing content"),
            _article(
                "https://example.com/b",
                "Article B",
                extraction_successful=False,
                content="",
                error="extractor unavailable",
            ),
        ]
    )

    assert response["status"] == "persist-ok"
    assert response["media_ids"] == []
    assert response["stored_articles"] == 0
    assert response["skipped_articles"] == 1
    assert response["duplicate_articles"] == 1
    assert response["errors"] == ["Failed to extract: https://example.com/b"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_null_media_id_is_storage_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_db(monkeypatch, (None, None, "No media stored"))

    response = await _persist([_article()])

    assert response["media_ids"] == []
    assert response["stored_articles"] == 0
    assert response["errors"] == ["Storage failed for article"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_mixed_stored_and_null_id_accounts_for_both(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_db(
        monkeypatch,
        (101, "stored-uuid", "Media stored successfully"),
        (None, None, "No media stored"),
    )

    response = await _persist([_article(), _article("https://example.com/b", "Article B")])

    assert response["status"] == "persist-ok"
    assert response["total_articles"] == 2
    assert response["media_ids"] == [101]
    assert response["stored_articles"] == 1
    assert response["errors"] == ["Storage failed for article"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_redacts_article_urls_from_logs(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_db(monkeypatch, (None, None, "No media stored"))
    secret = "secret-value"
    url = f"https://example.com/private?token={secret}"

    with _captured_logs() as records:
        await _persist(
            [
                _article(url, extraction_successful=False, content="", is_duplicate=True),
                _article(
                    url,
                    "Failed Article",
                    extraction_successful=False,
                    content="",
                    error="extractor unavailable",
                ),
                _article(url, "Unstored Article"),
            ]
        )

    captured = "\n".join(records)
    assert "https://example.com/private" in captured
    assert secret not in captured


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_webscraping_persist_redacts_storage_exception_from_logs(
    monkeypatch: pytest.MonkeyPatch,
):
    secret = "secret-value"
    url = f"https://example.com/private?token={secret}"
    _patch_db(monkeypatch, RuntimeError(f"storage failed for {url}"))

    with _captured_logs() as records:
        response = await _persist([_article(url)])

    assert response["errors"] == ["Storage failed for article"]
    assert secret not in "\n".join(records)
