"""Compatibility and failure boundaries for crawl-result retrieval."""

import pytest

from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
from tldw_Server_API.app.services.web_scraping_service import _ingest_crawl_articles

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("wrapper", [None, "articles", "results"])
def test_inline_results_keep_existing_analysis(wrapper: str | None) -> None:
    """Legacy list/envelope results retain analysis rather than overwriting it."""
    articles = [{"summary": "summary", "analysis": "existing"}, {"summary": "new"}]
    result = {wrapper: articles} if wrapper else articles
    assert _ingest_crawl_articles(result) == [
        {"summary": "summary", "analysis": "existing"},
        {"summary": "new", "analysis": "new"},
    ]


def test_expired_crawl_results_do_not_look_like_empty_scraping_success() -> None:
    """An expired real store entry must surface retrieval failure, not no articles."""
    result_id = ephemeral_storage.store_data({"result": {"articles": []}}, ttl_seconds=0)
    with pytest.raises(RuntimeError, match="expired before ingestion"):
        _ingest_crawl_articles({"ephemeral_id": result_id})


def test_empty_crawl_is_a_valid_empty_result() -> None:
    """Successful extraction of zero pages remains distinct from missing storage."""
    result_id = ephemeral_storage.store_data({"result": {"articles": []}})
    try:
        assert _ingest_crawl_articles({"ephemeral_id": result_id}) == []
    finally:
        ephemeral_storage.remove_data(result_id)
