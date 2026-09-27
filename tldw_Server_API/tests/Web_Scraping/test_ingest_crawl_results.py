"""Public ingestion compatibility and crawl-storage lifecycle boundaries."""

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import media
from tldw_Server_API.app.api.v1.schemas.media_request_models import IngestWebContentRequest
from tldw_Server_API.app.core.exceptions import ResourceNotFoundError
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
from tldw_Server_API.app.services.web_scraping_service import ingest_web_content_orchestrate

pytestmark = pytest.mark.unit
IngestResult = Callable[[Any], Awaitable[list[dict[str, Any]] | None]]


@pytest.fixture(params=["url_level", "recursive_scraping"])
def ingest_result(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> IngestResult:
    """Replace only the crawl producer; keep public orchestration and storage real."""

    async def ingest(result: Any) -> list[dict[str, Any]] | None:
        """Feed a crawl response through either public ingestion crawl mode."""

        async def scrape_task(**kwargs: Any) -> Any:
            """Stand in for the remote crawl's completed result."""
            return result

        monkeypatch.setattr(media, "process_web_scraping_task", scrape_task)
        return await ingest_web_content_orchestrate(
            IngestWebContentRequest(
                urls=["https://example.com"],
                scrape_method=request.param,
                perform_analysis=False,
                perform_chunking=False,
            ),
            db=SimpleNamespace(),
            usage_log=SimpleNamespace(),
        )

    return ingest


@pytest.mark.parametrize("wrapper", [None, "articles", "results"])
async def test_inline_results_keep_existing_analysis(ingest_result: IngestResult, wrapper: str | None) -> None:
    """Legacy list/envelope results retain analysis rather than overwriting it."""
    articles = [{"summary": "summary", "analysis": "existing"}, {"summary": "new"}]
    result = {wrapper: articles} if wrapper else articles
    assert await ingest_result(result) == [
        {"summary": "summary", "analysis": "existing"},
        {"summary": "new", "analysis": "new"},
    ]


async def test_expired_crawl_results_do_not_look_like_empty_scraping_success(ingest_result: IngestResult) -> None:
    """An expired real store entry must surface retrieval failure, not no articles."""
    result_id = ephemeral_storage.store_data({"result": {"articles": []}}, ttl_seconds=0)
    with pytest.raises(ResourceNotFoundError) as error:
        await ingest_result({"ephemeral_id": result_id})
    assert error.value.identifier == result_id
    assert result_id in str(error.value)


@pytest.mark.parametrize("articles", [[], [{"summary": "Crawl summary"}]])
async def test_ingestion_consumes_stored_crawl_result(
    ingest_result: IngestResult, articles: list[dict[str, str]]
) -> None:
    """Successful crawls return their articles and release storage, including empty crawls."""
    result_id = ephemeral_storage.store_data({"result": {"articles": articles}})
    try:
        result = await ingest_result({"ephemeral_id": result_id})
        assert result == ([{"summary": "Crawl summary", "analysis": "Crawl summary"}] if articles else [])
        assert ephemeral_storage.get_data(result_id) is None
    finally:
        ephemeral_storage.remove_data(result_id)


async def test_malformed_stored_result_is_removed_even_when_ingestion_fails(ingest_result: IngestResult) -> None:
    """Invalid stored envelopes must not leak their payload on the failure path."""
    result_id = ephemeral_storage.store_data({"unexpected": "payload"})
    try:
        with pytest.raises(KeyError):
            await ingest_result({"ephemeral_id": result_id})
        assert ephemeral_storage.get_data(result_id) is None
    finally:
        ephemeral_storage.remove_data(result_id)


async def test_legacy_inline_result_releases_its_unused_storage(ingest_result: IngestResult) -> None:
    """Legacy crawls return inline articles and an unused temporary media ID."""
    result_id = ephemeral_storage.store_data({"articles": [{"summary": "Legacy summary"}]})
    try:
        result = await ingest_result(
            {
                "status": "ephemeral-ok",
                "media_id": result_id,
                "results": [{"summary": "Legacy summary"}],
            }
        )
        assert result == [{"summary": "Legacy summary", "analysis": "Legacy summary"}]
        assert ephemeral_storage.get_data(result_id) is None
    finally:
        ephemeral_storage.remove_data(result_id)
