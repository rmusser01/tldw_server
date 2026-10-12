"""Two thread pools created at their call site carry the LLM call state (LLM usage accounting spec §1)."""

import json

import pytest

from tldw_Server_API.app.core.Claims_Extraction import ingestion_claims
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id, llm_call_context
from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as extractor_lib

pytestmark = pytest.mark.unit


def test_the_claims_extraction_pool_sees_the_submitters_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider call runs on a one-thread pool; it sees the user set where the pool was fed."""
    seen: list[int | None] = []

    def _probe(**kwargs: object) -> str:
        """Record the user and return one claim."""
        seen.append(current_llm_user_id())
        return json.dumps({"claims": [{"text": "Claim A."}]})

    monkeypatch.setattr(ingestion_claims, "chat_api_call", _probe, raising=False)
    chunks = [{"text": "irrelevant", "metadata": {"chunk_index": 0}}]
    with llm_call_context(user_id=7):
        claims = ingestion_claims.extract_claims_for_chunks(chunks, extractor_mode="openai", max_per_chunk=5)
    assert [c.get("claim_text") for c in claims] == ["Claim A."]
    assert seen == [7]


def test_the_article_extractor_pool_sees_the_submitters_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scrape runs on its own loop in a pool thread; the task on that loop sees the user."""
    seen: list[int | None] = []

    async def _probe(*_args: object, **_kwargs: object) -> list:
        """Record the user and return no pages."""
        seen.append(current_llm_user_id())
        return []

    monkeypatch.setattr(extractor_lib, "recursive_scrape", _probe)
    with llm_call_context(user_id=7):
        assert extractor_lib.sync_recursive_scrape("https://example.com", 1, 1) == []
    assert seen == [7]
