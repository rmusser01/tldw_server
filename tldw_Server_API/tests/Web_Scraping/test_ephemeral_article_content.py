"""Check Quick Ingest's opt-in response through the authenticated web scrape endpoint."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.services import web_scraping_service
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage


@pytest.mark.parametrize("include", [False, True])
def test_ephemeral_article_content_is_opt_in(
    client_user_only: TestClient, monkeypatch: pytest.MonkeyPatch, include: bool
) -> None:
    """The same ephemeral job can return exact article text without exposing scraper internals."""
    service = WebScrapingService()
    service._initialized = True
    article = {"url": "https://example.org/report", "title": "Field report", "content": "  First line\nSecond line\n"}
    monkeypatch.setattr(
        service,
        "_scrape_individual_urls",
        AsyncMock(return_value={"total_articles": 1, "articles": [{**article, "internal_debug": "omitted"}]}),
    )
    monkeypatch.setattr(web_scraping_service, "get_web_scraping_service", lambda: service)
    body = {"scrape_method": "Individual URLs", "url_input": article["url"], "mode": "ephemeral"}
    if include:
        body["include_extracted_content"] = True
    response = client_user_only.post("/api/v1/media/process-web-scraping", json=body)
    assert response.status_code == 200, response.text
    receipt = response.json()
    try:
        assert receipt["status"] == "ephemeral-ok"
        if include:
            assert receipt["articles"] == [article]
        else:
            assert "articles" not in receipt
    finally:
        ephemeral_storage.remove_data(receipt["ephemeral_id"])


@pytest.mark.parametrize("partial", [False, True])
def test_ephemeral_article_errors_keep_their_source_outcomes(
    client_user_only: TestClient, monkeypatch: pytest.MonkeyPatch, partial: bool
) -> None:
    """A rejected page cannot hide another page's text or become an empty success."""
    service = WebScrapingService()
    service._initialized = True
    rejected = {
        "url": "https://example.org/rejected?secret=hidden",
        "error": "Blocked by outbound policy",
        "extraction_successful": False,
    }
    good = {"url": "https://example.org/notes", "content": "Notes", "extraction_successful": True}
    articles = [rejected, good] if partial else [rejected]
    monkeypatch.setattr(service, "_scrape_individual_urls", AsyncMock(return_value={"articles": articles}))
    monkeypatch.setattr(web_scraping_service, "get_web_scraping_service", lambda: service)
    response = client_user_only.post(
        "/api/v1/media/process-web-scraping",
        json={
            "scrape_method": "Individual URLs",
            "url_input": rejected["url"],
            "mode": "ephemeral",
            "include_extracted_content": True,
        },
    )
    assert response.status_code == 200
    receipt = response.json()
    try:
        assert receipt["extraction_failures"][0]["code"] == "source_access_denied"
        assert "hidden" not in str(receipt)
        if partial:
            assert receipt["articles"] == [{"url": good["url"], "content": "Notes"}]
            assert receipt["warnings"] and not receipt.get("errors")
        else:
            assert receipt["status"] == "error" and receipt["errors"]
    finally:
        ephemeral_storage.remove_data(receipt["ephemeral_id"])
