"""Comprehensive tests for RAG adapters.

This module tests all RAG and search-related workflow adapters:
- run_rag_search_adapter: Execute RAG search via unified pipeline
- run_web_search_adapter: Web search via various engines
- run_rss_fetch_adapter: Fetch RSS/Atom feeds
- run_atom_fetch_adapter: Atom feed fetch (alias)
- run_query_rewrite_adapter: Rewrite search queries
- run_query_expand_adapter: Expand queries with synonyms
- run_hyde_generate_adapter: HyDE (hypothetical document) generation
- run_semantic_cache_check_adapter: Check semantic cache
- run_search_aggregate_adapter: Aggregate search results
"""

import pytest
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockRAGDocument:
    """Mock document returned by RAG pipeline."""

    id: str
    content: str
    metadata: Dict[str, Any]
    score: float = 0.8


@dataclass
class MockRAGResult:
    """Mock result from unified_rag_pipeline."""

    documents: List[MockRAGDocument]
    metadata: Dict[str, Any]
    timings: Dict[str, float]
    citations: Optional[List[str]] = None
    generated_answer: Optional[str] = None


@dataclass
class MockExpandedQuery:
    """Mock result from query expansion."""

    variations: List[str]
    synonyms: Dict[str, List[str]]
    keywords: List[str]
    entities: List[str]


@pytest.fixture
def mock_rag_result():
    """Create a mock RAG result for testing."""
    return MockRAGResult(
        documents=[
            MockRAGDocument(
                id="doc1",
                content="Test document content about AI",
                metadata={"source": "test.pdf", "page": 1},
                score=0.95,
            ),
            MockRAGDocument(
                id="doc2",
                content="Another document about machine learning",
                metadata={"source": "test2.pdf", "page": 2},
                score=0.85,
            ),
        ],
        metadata={"search_mode": "hybrid", "sources": ["media_db"]},
        timings={"search": 0.1, "total": 0.15},
        citations=["[1] test.pdf, p.1", "[2] test2.pdf, p.2"],
        generated_answer="AI is a field of computer science.",
    )


@pytest.fixture
def mock_web_search_result():
    """Create mock web search results."""
    return {
        "results": [
            {"title": "Result 1", "url": "https://example.com/1", "content": "First result"},
            {"title": "Result 2", "url": "https://example.com/2", "content": "Second result"},
        ],
        "total_results_found": 100,
    }


@pytest.fixture
def base_context():
    """Base context dict for adapter calls."""
    return {"user_id": "test_user", "tenant_id": "default"}


# ---------------------------------------------------------------------------
# RAG Search Adapter Tests
# ---------------------------------------------------------------------------


class TestRAGSearchAdapter:
    """Tests for run_rag_search_adapter."""

    @pytest.mark.asyncio
    async def test_rag_search_adapter_valid(self, monkeypatch, mock_rag_result, base_context):
        """Test RAG search adapter with valid config."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rag_search_adapter

        # Mock the unified_rag_pipeline
        mock_pipeline = AsyncMock(return_value=mock_rag_result)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.search.unified_rag_pipeline",
            mock_pipeline,
        )

        # Mock DatabasePaths
        mock_db_paths = MagicMock()
        mock_db_paths.get_single_user_id.return_value = "test_user"
        mock_db_paths.get_media_db_path.return_value = "/path/to/media.db"
        from tldw_Server_API.app.core.DB_Management import db_path_utils
        monkeypatch.setattr(db_path_utils, "DatabasePaths", mock_db_paths)

        config = {
            "query": "What is artificial intelligence?",
            "top_k": 5,
            "search_mode": "hybrid",
            "hybrid_alpha": 0.7,
        }

        result = await run_rag_search_adapter(config, base_context)

        assert "documents" in result
        assert "metadata" in result
        assert "timings" in result
        assert len(result["documents"]) == 2
        assert result["documents"][0]["id"] == "doc1"
        assert result["citations"] is not None
        assert result["generated_answer"] is not None

    @pytest.mark.asyncio
    async def test_rag_search_adapter_missing_query(self, monkeypatch, base_context):
        """Test RAG search adapter returns error for missing query."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rag_search_adapter

        # Mock DatabasePaths to avoid import errors
        mock_db_paths = MagicMock()
        mock_db_paths.get_single_user_id.return_value = "test_user"
        mock_db_paths.get_media_db_path.return_value = "/path/to/media.db"

        # Mock the import within the function
        with patch.dict("sys.modules", {
            "tldw_Server_API.app.core.DB_Management.db_path_utils": MagicMock(DatabasePaths=mock_db_paths)
        }):
            # Mock the pipeline to raise error on empty query
            mock_pipeline = AsyncMock(side_effect=ValueError("Empty query"))
            monkeypatch.setattr(
                "tldw_Server_API.app.core.Workflows.adapters.rag.search.unified_rag_pipeline",
                mock_pipeline,
            )

            config = {"query": "", "top_k": 5}

            # With empty query, the adapter should still call pipeline but may return error
            # The actual behavior depends on how the pipeline handles empty queries
            with pytest.raises(Exception):
                await run_rag_search_adapter(config, base_context)

    @pytest.mark.asyncio
    async def test_rag_search_adapter_with_template(self, monkeypatch, mock_rag_result, base_context):
        """Test RAG search adapter with templated query."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rag_search_adapter

        mock_pipeline = AsyncMock(return_value=mock_rag_result)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.search.unified_rag_pipeline",
            mock_pipeline,
        )

        mock_db_paths = MagicMock()
        mock_db_paths.get_single_user_id.return_value = "test_user"
        mock_db_paths.get_media_db_path.return_value = "/path/to/media.db"
        from tldw_Server_API.app.core.DB_Management import db_path_utils
        monkeypatch.setattr(db_path_utils, "DatabasePaths", mock_db_paths)

        # Context with variables for template
        context = {**base_context, "topic": "machine learning"}
        config = {"query": "Tell me about {{topic}}", "top_k": 10}

        result = await run_rag_search_adapter(config, context)

        assert "documents" in result
        # Check that template was applied (pipeline was called)
        mock_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_search_adapter_cancelled(self, base_context):
        """Test RAG search adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rag_search_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"query": "test query"}

        result = await run_rag_search_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_rag_search_adapter_sanitizes_media_db_path_logs(self, monkeypatch, base_context):
        """Test Media DB path resolution logs hide backend path details."""
        from tldw_Server_API.app.core.DB_Management import db_path_utils
        from tldw_Server_API.app.core.Workflows.adapters.rag import search as search_module
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rag_search_adapter

        class BrokenDatabasePaths:
            @staticmethod
            def get_single_user_id():
                return "test_user"

            @staticmethod
            def get_media_db_path(_user_id):
                raise RuntimeError("media db secret at /private/rag-media.db")

        monkeypatch.setattr(db_path_utils, "DatabasePaths", BrokenDatabasePaths)
        messages: list[str] = []
        sink_id = search_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
        try:
            with pytest.raises(RuntimeError) as exc_info:
                await run_rag_search_adapter({"query": "test"}, base_context)
        finally:
            search_module.logger.remove(sink_id)

        assert str(exc_info.value) == "Failed to resolve Media DB path for workflow search"
        joined = "\n".join(messages)
        assert "Failed to resolve Media DB path for workflow search" in joined
        assert "rag-media.db" not in joined

    @pytest.mark.asyncio
    async def test_rag_search_adapter_passthrough_options(self, monkeypatch, mock_rag_result, base_context):
        """Test RAG search adapter passes through advanced options."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rag_search_adapter

        mock_pipeline = AsyncMock(return_value=mock_rag_result)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.search.unified_rag_pipeline",
            mock_pipeline,
        )

        mock_db_paths = MagicMock()
        mock_db_paths.get_single_user_id.return_value = "test_user"
        mock_db_paths.get_media_db_path.return_value = "/path/to/media.db"
        from tldw_Server_API.app.core.DB_Management import db_path_utils
        monkeypatch.setattr(db_path_utils, "DatabasePaths", mock_db_paths)

        config = {
            "query": "test",
            "enable_reranking": True,
            "enable_citations": True,
            "enable_generation": True,
            "min_score": 0.5,
        }

        await run_rag_search_adapter(config, base_context)

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs.get("enable_reranking") is True
        assert call_kwargs.get("enable_citations") is True
        assert call_kwargs.get("enable_generation") is True
        assert call_kwargs.get("min_score") == 0.5


# ---------------------------------------------------------------------------
# Web Search Adapter Tests
# ---------------------------------------------------------------------------


class TestWebSearchAdapter:
    """Tests for run_web_search_adapter."""

    @pytest.mark.asyncio
    async def test_web_search_adapter_test_mode(self, monkeypatch, base_context):
        """Test web search adapter in test mode returns simulated results."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        config = {"query": "artificial intelligence", "engine": "google", "num_results": 5}

        result = await run_web_search_adapter(config, base_context)

        assert "results" in result
        assert result.get("simulated") is True
        assert result["query"] == "artificial intelligence"
        assert result["engine"] == "google"
        assert len(result["results"]) == 2  # Test mode returns 2 results

    @pytest.mark.asyncio
    async def test_web_search_adapter_test_mode_y(self, monkeypatch, base_context):
        """Test web search adapter treats TEST_MODE=y as test mode."""
        monkeypatch.setenv("TEST_MODE", "y")
        monkeypatch.setenv("TLDW_TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        config = {"query": "artificial intelligence", "engine": "google", "num_results": 5}

        result = await run_web_search_adapter(config, base_context)

        assert result.get("simulated") is True
        assert result["query"] == "artificial intelligence"

    @pytest.mark.asyncio
    async def test_web_search_adapter_missing_query(self, monkeypatch, base_context):
        """Test web search adapter returns error for missing query."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        config = {"query": "", "engine": "google"}

        result = await run_web_search_adapter(config, base_context)

        assert "error" in result
        assert result["error"] == "missing_query"

    @pytest.mark.asyncio
    async def test_web_search_adapter_invalid_engine(self, monkeypatch, base_context):
        """Test web search adapter returns error for invalid engine."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        config = {"query": "test", "engine": "invalid_engine"}

        result = await run_web_search_adapter(config, base_context)

        assert "error" in result
        assert "invalid_engine" in result["error"]
        assert "valid_engines" in result

    @pytest.mark.asyncio
    async def test_web_search_adapter_valid_with_mock(self, monkeypatch, mock_web_search_result, base_context):
        """Test web search adapter with mocked perform_websearch."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        mock_websearch = MagicMock(return_value=mock_web_search_result)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            mock_websearch,
        )

        config = {"query": "test query", "engine": "google", "num_results": 10}

        result = await run_web_search_adapter(config, base_context)

        assert "results" in result
        assert result["count"] == 2
        assert result["query"] == "test query"
        assert "text" in result
        assert "First result" in result["text"]

    @pytest.mark.asyncio
    async def test_web_search_adapter_sanitizes_backend_errors(self, monkeypatch, base_context):
        """Test web search adapter hides backend exception details in fallback errors."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        def explode(**_kwargs):
            raise RuntimeError("web search backend exploded at /private/rag-web-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            explode,
        )

        result = await run_web_search_adapter(
            {"query": "private topic", "engine": "google"},
            base_context,
        )

        assert result == {"error": "web_search_error"}

    @pytest.mark.asyncio
    async def test_web_search_adapter_cancelled(self, base_context):
        """Test web search adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"query": "test"}

        result = await run_web_search_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_web_search_adapter_with_summarize(self, monkeypatch, mock_web_search_result, base_context):
        """Test web search adapter with summarization enabled."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        mock_websearch = MagicMock(return_value=mock_web_search_result)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            mock_websearch,
        )

        mock_summarize = MagicMock(return_value="Summarized content about the query.")
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.summarize",
            mock_summarize,
        )

        config = {"query": "test query", "engine": "google", "summarize": True}

        result = await run_web_search_adapter(config, base_context)

        assert "summary" in result
        assert result["summary"] == "Summarized content about the query."

    @pytest.mark.asyncio
    async def test_web_search_adapter_uses_provider_alias(self, monkeypatch, base_context):
        """Test provider alias maps to engine when engine is not provided."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        captured = {}

        def fake_perform_websearch(**kwargs):
            captured.update(kwargs)
            return {"results": [], "total_results_found": 0}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            fake_perform_websearch,
        )

        result = await run_web_search_adapter({"query": "q", "provider": "duckduckgo"}, base_context)

        assert "error" not in result
        assert captured.get("search_engine") == "duckduckgo"

    @pytest.mark.asyncio
    async def test_web_search_adapter_maps_searxng_and_forwards_overrides(self, monkeypatch, base_context):
        """Test searxng alias normalization and Searx override forwarding."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        captured = {}

        def fake_perform_websearch(**kwargs):
            captured.update(kwargs)
            return {"results": [], "total_results_found": 0}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            fake_perform_websearch,
        )

        result = await run_web_search_adapter(
            {
                "query": "q",
                "engine": "searxng",
                "searx_url": "https://searx.example",
                "searx_json_mode": True,
            },
            base_context,
        )

        assert "error" not in result
        assert captured.get("search_engine") == "searx"
        assert captured.get("search_params", {}).get("searx_url") == "https://searx.example"
        assert captured.get("search_params", {}).get("searx_json_mode") is True

    @pytest.mark.asyncio
    async def test_web_search_adapter_searx_json_mode_accepts_y(self, monkeypatch, base_context):
        """Test searx_json_mode='y' is treated as enabled."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        captured = {}

        def fake_perform_websearch(**kwargs):
            captured.update(kwargs)
            return {"results": [], "total_results_found": 0}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            fake_perform_websearch,
        )

        result = await run_web_search_adapter(
            {
                "query": "q",
                "engine": "searx",
                "searx_json_mode": "y",
            },
            base_context,
        )

        assert "error" not in result
        assert captured.get("search_params", {}).get("searx_json_mode") is True

    @pytest.mark.asyncio
    async def test_web_search_adapter_forwards_domain_and_term_aliases(self, monkeypatch, base_context):
        """Test include/exclude-domain and term aliases map to perform_websearch args."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        captured = {}

        def fake_perform_websearch(**kwargs):
            captured.update(kwargs)
            return {"results": [], "total_results_found": 0}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            fake_perform_websearch,
        )

        result = await run_web_search_adapter(
            {
                "query": "q",
                "search_engine": "tavily",
                "include_domains": ["allowed.example"],
                "exclude_domains": ["blocked.example"],
                "exact_terms": "must include",
                "exclude_terms": "must exclude",
                "safeSearch": False,
                "timeRange": "week",
                "searxUrl": "https://searx.alias.example",
                "searxJsonMode": "true",
            },
            base_context,
        )

        assert "error" not in result
        assert captured.get("search_engine") == "tavily"
        assert captured.get("site_whitelist") == ["allowed.example"]
        assert captured.get("site_blacklist") == ["blocked.example"]
        assert captured.get("exactTerms") == "must include"
        assert captured.get("excludeTerms") == "must exclude"
        assert captured.get("safesearch") == "off"
        assert captured.get("date_range") == "week"
        assert captured.get("search_params", {}).get("searx_url") == "https://searx.alias.example"
        assert captured.get("search_params", {}).get("searx_json_mode") is True

    @pytest.mark.asyncio
    async def test_web_search_adapter_fetch_content_enriches_results(self, monkeypatch, base_context):
        """Test optional page-content enrichment with token truncation hook."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        mock_web_results = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "snippet one"},
                {"title": "Result 2", "url": "https://example.com/2", "content": "snippet two"},
            ],
            "total_results_found": 2,
        }
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            lambda **_: mock_web_results,
        )

        async def fake_scrape_article(url: str, custom_cookies=None):
            return {"url": url, "content": f"full content for {url}", "extraction_successful": True}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.scrape_article",
            fake_scrape_article,
        )

        trunc_calls = {}

        def fake_truncate(content: str, max_tokens: int, model=None) -> str:
            trunc_calls["content"] = content
            trunc_calls["max_tokens"] = max_tokens
            trunc_calls["model"] = model
            return "TRUNCATED_CONTENT"

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.search._truncate_content_by_tokens",
            fake_truncate,
        )

        result = await run_web_search_adapter(
            {
                "query": "test query",
                "engine": "google",
                "fetch_content": True,
                "fetch_limit": 1,
                "max_content_tokens": 123,
                "tokenizer_model": "gpt-4o-mini",
            },
            base_context,
        )

        assert "error" not in result
        assert result["results"][0]["content"] == "TRUNCATED_CONTENT"
        assert "content" not in result["results"][1]
        assert "TRUNCATED_CONTENT" in result["text"]
        assert result.get("fetch_content", {}).get("attempted") == 1
        assert result.get("fetch_content", {}).get("fetched") == 1
        assert result.get("fetch_content", {}).get("max_content_tokens") == 123
        assert trunc_calls.get("max_tokens") == 123
        assert trunc_calls.get("model") == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_web_search_adapter_fetch_content_failure_keeps_snippet(self, monkeypatch, base_context):
        """Test failed fetches are dropped from enriched output by default."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        mock_web_results = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "snippet one"},
            ],
            "total_results_found": 1,
        }
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            lambda **_: mock_web_results,
        )

        async def fake_scrape_article(url: str, custom_cookies=None):
            return {"url": url, "content": "", "extraction_successful": False, "error": "blocked"}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.scrape_article",
            fake_scrape_article,
        )

        result = await run_web_search_adapter(
            {
                "query": "test query",
                "engine": "google",
                "fetch_content": True,
                "fetch_limit": 1,
            },
            base_context,
        )

        assert "error" not in result
        assert result["count"] == 0
        assert result.get("fetch_content", {}).get("attempted") == 1
        assert result.get("fetch_content", {}).get("fetched") == 0
        assert result.get("fetch_content", {}).get("dropped_failed") == 1
        assert result.get("fetch_content", {}).get("errors")

    @pytest.mark.asyncio
    async def test_web_search_adapter_fetch_content_can_keep_failed_results(self, monkeypatch, base_context):
        """Test optional disablement of failed-fetch filtering keeps snippet results."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        mock_web_results = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "snippet one"},
            ],
            "total_results_found": 1,
        }
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            lambda **_: mock_web_results,
        )

        async def fake_scrape_article(url: str, custom_cookies=None):
            return {"url": url, "content": "", "extraction_successful": False, "error": "blocked"}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.scrape_article",
            fake_scrape_article,
        )

        result = await run_web_search_adapter(
            {
                "query": "test query",
                "engine": "google",
                "fetch_content": True,
                "fetch_limit": 1,
                "filter_failed_fetches": False,
            },
            base_context,
        )

        assert "error" not in result
        assert result["count"] == 1
        assert "snippet one" in result["text"]
        assert result.get("fetch_content", {}).get("dropped_failed") == 0

    @pytest.mark.asyncio
    async def test_web_search_adapter_auto_query_rewrite(self, monkeypatch, base_context):
        """Test optional query rewrite updates the search query before provider call."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        captured = {}

        def fake_perform_websearch(**kwargs):
            captured.update(kwargs)
            return {"results": [], "total_results_found": 0}

        async def fake_run_query_rewrite_adapter(config, context):
            assert config.get("query") == "original query"
            return {
                "original_query": "original query",
                "rewritten_queries": ["rewritten compact query"],
                "strategy": "simplify",
            }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            fake_perform_websearch,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query.run_query_rewrite_adapter",
            fake_run_query_rewrite_adapter,
        )

        result = await run_web_search_adapter(
            {
                "query": "original query",
                "engine": "google",
                "auto_query_rewrite": True,
                "query_rewrite_strategy": "simplify",
                "query_rewrite_max_rewrites": 1,
            },
            base_context,
        )

        assert "error" not in result
        assert captured.get("search_query") == "rewritten compact query"
        assert result.get("query") == "rewritten compact query"
        assert result.get("original_query") == "original query"
        assert result.get("query_rewrite", {}).get("rewritten") is True
        assert result.get("query_rewrite", {}).get("query_used") == "rewritten compact query"

    @pytest.mark.asyncio
    async def test_web_search_adapter_sanitizes_query_rewrite_errors(self, monkeypatch, base_context):
        """Test query rewrite failure metadata hides backend details."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        def fake_perform_websearch(**kwargs):
            return {"results": [], "total_results_found": 0}

        async def fake_run_query_rewrite_adapter(config, context):
            raise RuntimeError("rewrite token at /private/rag-rewrite-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            fake_perform_websearch,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query.run_query_rewrite_adapter",
            fake_run_query_rewrite_adapter,
        )

        result = await run_web_search_adapter(
            {"query": "original query", "engine": "google", "auto_query_rewrite": True},
            base_context,
        )

        assert result.get("error") is None
        assert result.get("query_rewrite", {}).get("error") == "query_rewrite_error"
        assert "rag-rewrite-cache" not in str(result)

    @pytest.mark.asyncio
    async def test_web_search_adapter_sanitizes_fetch_and_summary_errors(self, monkeypatch, base_context):
        """Test fetch and summarization failures hide backend details."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import search as search_module
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            lambda **_: {
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1", "content": "snippet one"},
                ],
                "total_results_found": 1,
            },
        )

        async def fake_scrape_article(url: str, custom_cookies=None):
            raise RuntimeError("scrape token at /private/rag-scrape-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib.scrape_article",
            fake_scrape_article,
        )

        def fake_summarize(**kwargs):
            raise RuntimeError("summary token at /private/rag-summary-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.summarize",
            fake_summarize,
        )

        messages: list[str] = []
        sink_id = search_module.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
        try:
            result = await run_web_search_adapter(
                {
                    "query": "test query",
                    "engine": "google",
                    "fetch_content": True,
                    "fetch_limit": 1,
                    "filter_failed_fetches": False,
                    "summarize": True,
                },
                base_context,
            )
        finally:
            search_module.logger.remove(sink_id)

        assert result.get("fetch_content", {}).get("errors") == [
            {"index": 0, "error": "fetch_failed"}
        ]
        assert result.get("summary_error") == "summary_failed"
        assert "rag-scrape-cache" not in str(result)
        assert "rag-summary-cache" not in str(result)
        assert "rag-summary-cache" not in "\n".join(messages)

    @pytest.mark.asyncio
    async def test_web_search_adapter_sanitizes_provider_errors(self, monkeypatch, base_context):
        """Test provider failure payloads and logs hide backend details."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import search as search_module
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        def fake_perform_websearch(**kwargs):
            raise RuntimeError("provider token at /private/rag-web-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            fake_perform_websearch,
        )

        messages: list[str] = []
        sink_id = search_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
        try:
            result = await run_web_search_adapter({"query": "test query", "engine": "google"}, base_context)
        finally:
            search_module.logger.remove(sink_id)

        assert result == {"error": "web_search_error"}
        assert "rag-web-cache" not in "\n".join(messages)

    @pytest.mark.asyncio
    async def test_web_search_adapter_sanitizes_processing_errors(self, monkeypatch, base_context):
        """Test provider processing-error payloads hide backend details."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_web_search_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch",
            lambda **_: {"processing_error": "provider token at /private/rag-processing-cache"},
        )

        result = await run_web_search_adapter({"query": "test query", "engine": "google"}, base_context)

        assert result == {"error": "search_error", "query": "test query"}
        assert "rag-processing-cache" not in str(result)

    def test_truncate_content_by_tokens_respects_budget(self, monkeypatch):
        """Test token truncation helper enforces the requested token budget."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import search as search_module

        class _FakeEncoding:
            def encode(self, value: str):
                return [0] * len(value)

        class _FakeTiktoken:
            @staticmethod
            def get_encoding(_name: str):
                return _FakeEncoding()

            @staticmethod
            def encoding_for_model(_model: str):
                return _FakeEncoding()

        monkeypatch.setitem(sys.modules, "tiktoken", _FakeTiktoken())

        truncated = search_module._truncate_content_by_tokens("abcdefghij", max_tokens=5, model="fake-model")
        assert truncated.startswith("abcde")
        assert truncated.endswith("\n...[truncated]")


# ---------------------------------------------------------------------------
# RSS Fetch Adapter Tests
# ---------------------------------------------------------------------------


class TestRSSFetchAdapter:
    """Tests for run_rss_fetch_adapter."""

    @pytest.mark.asyncio
    async def test_rss_fetch_adapter_test_mode(self, monkeypatch, base_context):
        """Test RSS fetch adapter in test mode returns simulated results."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rss_fetch_adapter

        config = {"urls": ["https://example.com/feed.xml"], "limit": 5}

        result = await run_rss_fetch_adapter(config, base_context)

        assert "results" in result
        assert "count" in result
        assert result["results"][0]["title"] == "Test Item"

    @pytest.mark.asyncio
    async def test_rss_fetch_adapter_test_mode_y(self, monkeypatch, base_context):
        """Test RSS fetch adapter treats TEST_MODE=y as test mode."""
        monkeypatch.setenv("TEST_MODE", "y")
        monkeypatch.setenv("TLDW_TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rss_fetch_adapter

        config = {"urls": ["https://example.com/feed.xml"], "limit": 5}

        result = await run_rss_fetch_adapter(config, base_context)

        assert "results" in result
        assert result["results"][0]["title"] == "Test Item"

    @pytest.mark.asyncio
    async def test_rss_fetch_adapter_empty_urls(self, monkeypatch, base_context):
        """Test RSS fetch adapter with empty URLs returns empty results."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rss_fetch_adapter

        config = {"urls": [], "limit": 10}

        result = await run_rss_fetch_adapter(config, base_context)

        assert result["results"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_rss_fetch_adapter_sanitizes_backend_errors(self, monkeypatch, base_context):
        """Test RSS adapter hides backend exception details in fallback errors."""
        import builtins

        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rss_fetch_adapter

        real_import = builtins.__import__

        def fail_urllib_parse_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "urllib.parse":
                raise RuntimeError("rss backend exploded at /private/rss-cache")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fail_urllib_parse_import)

        result = await run_rss_fetch_adapter(
            {"urls": ["https://example.com/feed.xml"]},
            base_context,
        )

        assert result == {"error": "rss_error"}

    @pytest.mark.asyncio
    async def test_rss_fetch_adapter_string_urls(self, monkeypatch, base_context):
        """Test RSS fetch adapter with comma-separated URL string."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rss_fetch_adapter

        config = {"urls": "https://example.com/feed1.xml, https://example.com/feed2.xml", "limit": 5}

        result = await run_rss_fetch_adapter(config, base_context)

        assert "results" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_rss_fetch_adapter_limit(self, monkeypatch, base_context):
        """Test RSS fetch adapter respects limit parameter."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rss_fetch_adapter

        config = {"urls": ["https://example.com/feed.xml"], "limit": 1}

        result = await run_rss_fetch_adapter(config, base_context)

        assert result["count"] <= 1

    @pytest.mark.asyncio
    async def test_rss_fetch_adapter_include_content(self, monkeypatch, base_context):
        """Test RSS fetch adapter include_content option."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rss_fetch_adapter

        config = {"urls": ["https://example.com/feed.xml"], "include_content": True}

        result = await run_rss_fetch_adapter(config, base_context)

        assert "text" in result

    @pytest.mark.asyncio
    async def test_rss_fetch_adapter_sanitizes_outer_errors(self, monkeypatch, base_context):
        """Test RSS outer fallback hides import/backend details."""
        monkeypatch.setenv("TEST_MODE", "0")

        import builtins

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_rss_fetch_adapter

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "urllib.parse" and fromlist == ("urlparse",):
                raise ImportError("rss import token at /private/rag-rss-cache")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        result = await run_rss_fetch_adapter({"urls": ["https://example.com/feed.xml"]}, base_context)

        assert result == {"error": "rss_error"}
        assert "rag-rss-cache" not in str(result)


# ---------------------------------------------------------------------------
# Atom Fetch Adapter Tests
# ---------------------------------------------------------------------------


class TestAtomFetchAdapter:
    """Tests for run_atom_fetch_adapter (alias for rss_fetch)."""

    @pytest.mark.asyncio
    async def test_atom_fetch_adapter_test_mode(self, monkeypatch, base_context):
        """Test Atom fetch adapter in test mode returns simulated results."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_atom_fetch_adapter

        config = {"urls": ["https://example.com/atom.xml"], "limit": 5}

        result = await run_atom_fetch_adapter(config, base_context)

        assert "results" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_atom_fetch_adapter_same_as_rss(self, monkeypatch, base_context):
        """Test Atom fetch adapter returns same results as RSS fetch."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import (
            run_atom_fetch_adapter,
            run_rss_fetch_adapter,
        )

        config = {"urls": ["https://example.com/feed.xml"], "limit": 5}

        rss_result = await run_rss_fetch_adapter(config, base_context)
        atom_result = await run_atom_fetch_adapter(config, base_context)

        assert rss_result == atom_result


# ---------------------------------------------------------------------------
# Query Rewrite Adapter Tests
# ---------------------------------------------------------------------------


class TestQueryRewriteAdapter:
    """Tests for run_query_rewrite_adapter."""

    @pytest.mark.asyncio
    async def test_query_rewrite_adapter_valid(self, monkeypatch, base_context):
        """Test query rewrite adapter with valid config."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_rewrite_adapter

        # Mock perform_chat_api_call_async
        mock_chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "rewritten query 1\nrewritten query 2\nrewritten query 3"}}]
        })
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat,
        )

        config = {"query": "What is AI?", "strategy": "expand", "max_rewrites": 3}

        result = await run_query_rewrite_adapter(config, base_context)

        assert "original_query" in result
        assert "rewritten_queries" in result
        assert result["original_query"] == "What is AI?"
        assert result["strategy"] == "expand"

    @pytest.mark.asyncio
    async def test_query_rewrite_adapter_missing_query(self, base_context):
        """Test query rewrite adapter returns error for missing query."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_rewrite_adapter

        config = {"query": "", "strategy": "expand"}

        result = await run_query_rewrite_adapter(config, base_context)

        assert "error" in result
        assert result["error"] == "missing_query"

    @pytest.mark.asyncio
    async def test_query_rewrite_adapter_cancelled(self, base_context):
        """Test query rewrite adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_rewrite_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"query": "test query"}

        result = await run_query_rewrite_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_query_rewrite_adapter_sanitizes_backend_errors(self, monkeypatch, base_context):
        """Test query rewrite failures hide raw backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_rewrite_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            AsyncMock(side_effect=RuntimeError("rewrite backend exploded at /private/rag-cache")),
        )

        result = await run_query_rewrite_adapter({"query": "What is AI?"}, base_context)

        assert result == {
            "error": "query_rewrite_error",
            "original_query": "What is AI?",
            "rewritten_queries": [],
        }

    @pytest.mark.asyncio
    async def test_query_rewrite_adapter_different_strategies(self, monkeypatch, base_context):
        """Test query rewrite adapter with different strategies."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_rewrite_adapter

        mock_chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "simplified query"}}]
        })
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat,
        )

        strategies = ["expand", "clarify", "simplify", "all"]

        for strategy in strategies:
            config = {"query": "complex query about AI", "strategy": strategy, "max_rewrites": 1}
            result = await run_query_rewrite_adapter(config, base_context)
            assert result["strategy"] == strategy


# ---------------------------------------------------------------------------
# Query Expand Adapter Tests
# ---------------------------------------------------------------------------


class TestQueryExpandAdapter:
    """Tests for run_query_expand_adapter."""

    @pytest.mark.asyncio
    async def test_query_expand_adapter_test_mode(self, monkeypatch, base_context):
        """Test query expand adapter in test mode returns simulated results."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_expand_adapter

        config = {"query": "machine learning", "strategies": ["synonym"], "max_expansions": 5}

        result = await run_query_expand_adapter(config, base_context)

        assert "original" in result
        assert "variations" in result
        assert "combined" in result
        assert result.get("simulated") is True
        assert result["original"] == "machine learning"

    @pytest.mark.asyncio
    async def test_query_expand_adapter_missing_query(self, monkeypatch, base_context):
        """Test query expand adapter returns error for missing query."""
        monkeypatch.setenv("TEST_MODE", "0")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_expand_adapter

        config = {"query": "", "strategies": ["synonym"]}

        result = await run_query_expand_adapter(config, base_context)

        assert "error" in result
        assert result["error"] == "missing_query"

    @pytest.mark.asyncio
    async def test_query_expand_adapter_cancelled(self, base_context):
        """Test query expand adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_expand_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"query": "test query"}

        result = await run_query_expand_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_query_expand_adapter_sanitizes_backend_errors(self, monkeypatch, base_context):
        """Test query expansion failures hide raw backend exception details."""
        monkeypatch.delenv("TEST_MODE", raising=False)

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_expand_adapter

        class BrokenHybridExpansion:
            async def expand(self, _query):
                raise RuntimeError("query expansion exploded at /private/rag-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.RAG.rag_service.query_expansion.HybridQueryExpansion",
            BrokenHybridExpansion,
        )

        result = await run_query_expand_adapter(
            {"query": "machine learning", "strategies": ["hybrid"]},
            base_context,
        )

        assert result == {"error": "query_expand_error"}

    @pytest.mark.asyncio
    async def test_query_expand_adapter_sanitizes_strategy_failure_logs(self, monkeypatch, base_context):
        """Test per-strategy query expansion logs hide raw backend exception details."""
        monkeypatch.delenv("TEST_MODE", raising=False)

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_expand_adapter
        from tldw_Server_API.app.core.Workflows.adapters.rag import query as query_module

        class BrokenSynonymExpansion:
            async def expand(self, _query):
                raise RuntimeError("strategy backend exploded at /private/rag-strategy.db")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.RAG.rag_service.query_expansion.SynonymExpansion",
            BrokenSynonymExpansion,
        )

        messages: list[str] = []
        sink_id = query_module.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
        try:
            result = await run_query_expand_adapter(
                {"query": "machine learning", "strategies": ["synonym"]},
                base_context,
            )
        finally:
            query_module.logger.remove(sink_id)

        assert result["original"] == "machine learning"
        assert result["variations"] == []
        joined = "\n".join(messages)
        assert "Query expansion strategy failed" in joined
        assert "synonym" not in joined
        assert "private/rag-strategy.db" not in joined

    @pytest.mark.asyncio
    async def test_query_expand_adapter_multiple_strategies(self, monkeypatch, base_context):
        """Test query expand adapter with multiple strategies."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_expand_adapter

        config = {
            "query": "deep learning neural networks",
            "strategies": ["synonym", "acronym", "domain"],
            "max_expansions": 10,
        }

        result = await run_query_expand_adapter(config, base_context)

        assert "variations" in result
        assert "strategies_used" in result

    @pytest.mark.asyncio
    async def test_query_expand_adapter_from_context(self, monkeypatch, base_context):
        """Test query expand adapter gets query from context if not in config."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import run_query_expand_adapter

        context = {**base_context, "prev": {"query": "context query"}}
        config = {"strategies": ["synonym"]}

        result = await run_query_expand_adapter(config, context)

        assert result["original"] == "context query"


# ---------------------------------------------------------------------------
# HyDE Generate Adapter Tests
# ---------------------------------------------------------------------------


class TestHyDEGenerateAdapter:
    """Tests for run_hyde_generate_adapter."""

    @pytest.mark.asyncio
    async def test_hyde_generate_adapter_valid(self, monkeypatch, base_context):
        """Test HyDE generate adapter with valid config."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_hyde_generate_adapter

        mock_chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "This is a hypothetical document about AI."}}]
        })
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat,
        )

        config = {
            "query": "What is artificial intelligence?",
            "num_hypothetical": 1,
            "document_type": "passage",
        }

        result = await run_hyde_generate_adapter(config, base_context)

        assert "query" in result
        assert "hypothetical_documents" in result
        assert result["query"] == "What is artificial intelligence?"
        assert len(result["hypothetical_documents"]) >= 1

    @pytest.mark.asyncio
    async def test_hyde_generate_adapter_missing_query(self, base_context):
        """Test HyDE generate adapter returns error for missing query."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_hyde_generate_adapter

        config = {"query": "", "document_type": "passage"}

        result = await run_hyde_generate_adapter(config, base_context)

        assert "error" in result
        assert result["error"] == "missing_query"

    @pytest.mark.asyncio
    async def test_hyde_generate_adapter_cancelled(self, base_context):
        """Test HyDE generate adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_hyde_generate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"query": "test query"}

        result = await run_hyde_generate_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_hyde_generate_adapter_sanitizes_backend_errors(self, monkeypatch, base_context):
        """Test HyDE generation failures hide raw backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_hyde_generate_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            AsyncMock(side_effect=RuntimeError("hyde backend exploded at /private/rag-cache")),
        )

        result = await run_hyde_generate_adapter({"query": "What is AI?"}, base_context)

        assert result == {
            "error": "hyde_error",
            "query": "What is AI?",
            "hypothetical_documents": [],
        }

    @pytest.mark.asyncio
    async def test_hyde_generate_adapter_multiple_docs(self, monkeypatch, base_context):
        """Test HyDE generate adapter with multiple hypothetical documents."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_hyde_generate_adapter

        mock_chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "Doc 1 about AI\n---\nDoc 2 about AI\n---\nDoc 3 about AI"}}]
        })
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat,
        )

        config = {"query": "What is AI?", "num_hypothetical": 3, "document_type": "answer"}

        result = await run_hyde_generate_adapter(config, base_context)

        assert len(result["hypothetical_documents"]) == 3
        assert result["document_type"] == "answer"

    @pytest.mark.asyncio
    async def test_hyde_generate_adapter_different_doc_types(self, monkeypatch, base_context):
        """Test HyDE generate adapter with different document types."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_hyde_generate_adapter

        mock_chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "Hypothetical content"}}]
        })
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            mock_chat,
        )

        doc_types = ["answer", "passage", "article"]

        for doc_type in doc_types:
            config = {"query": "test query", "document_type": doc_type}
            result = await run_hyde_generate_adapter(config, base_context)
            assert result["document_type"] == doc_type


# ---------------------------------------------------------------------------
# Semantic Cache Check Adapter Tests
# ---------------------------------------------------------------------------


class TestSemanticCacheCheckAdapter:
    """Tests for run_semantic_cache_check_adapter."""

    @pytest.mark.asyncio
    async def test_semantic_cache_check_adapter_cache_hit(self, monkeypatch, base_context):
        """Test semantic cache check adapter with cache hit."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_semantic_cache_check_adapter
        import time

        # Mock semantic-cache collection path
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "distances": [[0.05]],  # Low distance = high similarity
            "metadatas": [[{"cached_at": time.time(), "result": '{"data": "cached"}'}]],
            "documents": [["similar cached query"]],
        }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query._get_semantic_cache_collection",
            lambda _name: (MagicMock(), mock_collection),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query._build_semantic_cache_query_embedding",
            lambda _query, _manager: [0.1, 0.2, 0.3],
        )

        config = {"query": "test query", "similarity_threshold": 0.9}

        result = await run_semantic_cache_check_adapter(config, base_context)

        assert result["cache_hit"] is True
        assert result["query"] == "test query"
        assert "cached_result" in result

    @pytest.mark.asyncio
    async def test_semantic_cache_check_adapter_cache_miss(self, monkeypatch, base_context):
        """Test semantic cache check adapter with cache miss."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_semantic_cache_check_adapter

        # Mock semantic-cache collection with no matches
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "distances": [[0.8]],  # High distance = low similarity
            "metadatas": [[]],
            "documents": [[]],
        }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query._get_semantic_cache_collection",
            lambda _name: (MagicMock(), mock_collection),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query._build_semantic_cache_query_embedding",
            lambda _query, _manager: [0.1, 0.2, 0.3],
        )

        config = {"query": "unique query", "similarity_threshold": 0.9}

        result = await run_semantic_cache_check_adapter(config, base_context)

        assert result["cache_hit"] is False
        assert result["query"] == "unique query"

    @pytest.mark.asyncio
    async def test_semantic_cache_check_adapter_missing_query(self, base_context):
        """Test semantic cache check adapter returns error for missing query."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_semantic_cache_check_adapter

        config = {"query": ""}

        result = await run_semantic_cache_check_adapter(config, base_context)

        assert result["cache_hit"] is False
        assert "error" in result
        assert result["error"] == "missing_query"

    @pytest.mark.asyncio
    async def test_semantic_cache_check_adapter_cancelled(self, base_context):
        """Test semantic cache check adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_semantic_cache_check_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"query": "test query"}

        result = await run_semantic_cache_check_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_semantic_cache_check_adapter_sanitizes_backend_errors(
        self, monkeypatch, base_context
    ):
        """Test semantic cache failures hide raw backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_semantic_cache_check_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query._get_semantic_cache_collection",
            lambda _name: (_ for _ in ()).throw(
                RuntimeError("semantic cache exploded at /private/rag-cache")
            ),
        )

        result = await run_semantic_cache_check_adapter({"query": "test query"}, base_context)

        assert result == {
            "cache_hit": False,
            "query": "test query",
            "error": "semantic_cache_error",
        }

    @pytest.mark.asyncio
    async def test_semantic_cache_check_adapter_chroma_unavailable(self, monkeypatch, base_context):
        """Test semantic cache check adapter handles unavailable ChromaDB."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_semantic_cache_check_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query._get_semantic_cache_collection",
            lambda _name: (None, None),
        )

        config = {"query": "test query"}

        result = await run_semantic_cache_check_adapter(config, base_context)

        assert result["cache_hit"] is False
        assert "error" in result
        assert result["error"] == "chroma_unavailable"

    @pytest.mark.asyncio
    async def test_semantic_cache_check_adapter_expired_cache(self, monkeypatch, base_context):
        """Test semantic cache check adapter rejects expired cache entries."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_semantic_cache_check_adapter
        import time

        # Cache entry from 2 hours ago (expired with 1 hour TTL)
        cached_time = time.time() - 7200

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "distances": [[0.05]],
            "metadatas": [[{"cached_at": cached_time, "result": '{"data": "old"}'}]],
            "documents": [["old query"]],
        }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query._get_semantic_cache_collection",
            lambda _name: (MagicMock(), mock_collection),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.rag.query._build_semantic_cache_query_embedding",
            lambda _query, _manager: [0.1, 0.2, 0.3],
        )

        config = {"query": "test query", "max_age_seconds": 3600}  # 1 hour TTL

        result = await run_semantic_cache_check_adapter(config, base_context)

        assert result["cache_hit"] is False


# ---------------------------------------------------------------------------
# Search Aggregate Adapter Tests
# ---------------------------------------------------------------------------


class TestSearchAggregateAdapter:
    """Tests for run_search_aggregate_adapter."""

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_basic(self, base_context):
        """Test search aggregate adapter with basic input."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        config = {
            "results": [
                {
                    "documents": [
                        {"id": "1", "content": "Doc 1", "score": 0.9},
                        {"id": "2", "content": "Doc 2", "score": 0.8},
                    ]
                },
                {
                    "documents": [
                        {"id": "3", "content": "Doc 3", "score": 0.85},
                        {"id": "1", "content": "Doc 1 dup", "score": 0.95},  # Duplicate
                    ]
                },
            ],
            "dedup_field": "id",
            "sort_by": "score",
            "sort_order": "desc",
            "limit": 10,
        }

        result = await run_search_aggregate_adapter(config, base_context)

        assert "documents" in result
        assert result["total_before_dedup"] == 4
        assert result["total_after_dedup"] == 3  # One duplicate removed

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_cancelled(self, base_context):
        """Test search aggregate adapter respects cancellation."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        context = {**base_context, "is_cancelled": lambda: True}
        config = {"results": []}

        result = await run_search_aggregate_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_from_context(self, base_context):
        """Test search aggregate adapter gets results from context."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        context = {
            **base_context,
            "prev": {
                "documents": [
                    {"id": "1", "content": "From context", "score": 0.9},
                ]
            },
        }
        config = {}  # No results in config

        result = await run_search_aggregate_adapter(config, context)

        assert "documents" in result
        assert result["total_before_dedup"] == 1

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_merge_scores_sum(self, base_context):
        """Test search aggregate adapter with sum merge strategy."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        config = {
            "results": [
                {"documents": [{"id": "1", "content": "Doc 1", "score": 0.5}]},
                {"documents": [{"id": "1", "content": "Doc 1 dup", "score": 0.5}]},
            ],
            "dedup_field": "id",
            "merge_scores": "sum",
        }

        result = await run_search_aggregate_adapter(config, base_context)

        # Score should be summed: 0.5 + 0.5 = 1.0
        assert result["documents"][0]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_merge_scores_max(self, base_context):
        """Test search aggregate adapter with max merge strategy."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        config = {
            "results": [
                {"documents": [{"id": "1", "content": "Doc 1", "score": 0.3}]},
                {"documents": [{"id": "1", "content": "Doc 1 dup", "score": 0.9}]},
            ],
            "dedup_field": "id",
            "merge_scores": "max",
        }

        result = await run_search_aggregate_adapter(config, base_context)

        # Score should be max: max(0.3, 0.9) = 0.9
        assert result["documents"][0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_merge_scores_avg(self, base_context):
        """Test search aggregate adapter with average merge strategy."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        config = {
            "results": [
                {"documents": [{"id": "1", "content": "Doc 1", "score": 0.4}]},
                {"documents": [{"id": "1", "content": "Doc 1 dup", "score": 0.6}]},
            ],
            "dedup_field": "id",
            "merge_scores": "avg",
        }

        result = await run_search_aggregate_adapter(config, base_context)

        # Score should be average: (0.4 + 0.6) / 2 = 0.5
        assert result["documents"][0]["score"] == 0.5

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_sort_ascending(self, base_context):
        """Test search aggregate adapter with ascending sort order."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        config = {
            "results": [
                {
                    "documents": [
                        {"id": "1", "content": "High score", "score": 0.9},
                        {"id": "2", "content": "Low score", "score": 0.1},
                    ]
                }
            ],
            "sort_by": "score",
            "sort_order": "asc",
        }

        result = await run_search_aggregate_adapter(config, base_context)

        # Lower score should come first
        assert result["documents"][0]["score"] == 0.1
        assert result["documents"][1]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_limit(self, base_context):
        """Test search aggregate adapter respects limit."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        config = {
            "results": [
                {
                    "documents": [
                        {"id": str(i), "content": f"Doc {i}", "score": 0.9 - i * 0.1}
                        for i in range(10)
                    ]
                }
            ],
            "limit": 3,
        }

        result = await run_search_aggregate_adapter(config, base_context)

        assert len(result["documents"]) == 3

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_sources_tracking(self, base_context):
        """Test search aggregate adapter tracks sources."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        config = {
            "results": [
                {"documents": [{"id": "1", "score": 0.9}], "source": "web_search"},
                {"documents": [{"id": "2", "score": 0.8}], "source": "rag_search"},
            ],
        }

        result = await run_search_aggregate_adapter(config, base_context)

        assert "sources" in result
        assert "web_search" in result["sources"]
        assert "rag_search" in result["sources"]

    @pytest.mark.asyncio
    async def test_search_aggregate_adapter_empty_results(self, base_context):
        """Test search aggregate adapter with empty results."""
        from tldw_Server_API.app.core.Workflows.adapters.rag import run_search_aggregate_adapter

        config = {"results": []}

        result = await run_search_aggregate_adapter(config, base_context)

        assert result["documents"] == []
        assert result["total_before_dedup"] == 0
        assert result["total_after_dedup"] == 0


# ---------------------------------------------------------------------------
# Integration-style Tests
# ---------------------------------------------------------------------------


class TestRAGAdaptersIntegration:
    """Integration-style tests for RAG adapters working together."""

    @pytest.mark.asyncio
    async def test_query_expand_then_aggregate(self, monkeypatch, base_context):
        """Test query expansion followed by aggregation."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import (
            run_query_expand_adapter,
            run_search_aggregate_adapter,
        )

        # First expand the query
        expand_config = {"query": "machine learning", "strategies": ["synonym"]}
        expand_result = await run_query_expand_adapter(expand_config, base_context)

        assert "variations" in expand_result

        # Then aggregate results (simulated)
        agg_config = {
            "results": [
                {"documents": [{"id": "1", "content": "ML doc", "score": 0.9}]},
                {"documents": [{"id": "2", "content": "AI doc", "score": 0.85}]},
            ],
        }
        agg_result = await run_search_aggregate_adapter(agg_config, base_context)

        assert "documents" in agg_result
        assert len(agg_result["documents"]) == 2

    @pytest.mark.asyncio
    async def test_web_and_rss_search_aggregate(self, monkeypatch, base_context):
        """Test aggregating web search and RSS feed results."""
        monkeypatch.setenv("TEST_MODE", "1")

        from tldw_Server_API.app.core.Workflows.adapters.rag import (
            run_web_search_adapter,
            run_rss_fetch_adapter,
            run_search_aggregate_adapter,
        )

        # Run web search
        web_config = {"query": "AI news", "engine": "google"}
        web_result = await run_web_search_adapter(web_config, base_context)

        # Run RSS fetch
        rss_config = {"urls": ["https://example.com/feed.xml"]}
        rss_result = await run_rss_fetch_adapter(rss_config, base_context)

        # Aggregate both
        agg_config = {
            "results": [
                {"documents": web_result.get("results", []), "source": "web"},
                {"documents": rss_result.get("results", []), "source": "rss"},
            ],
        }
        agg_result = await run_search_aggregate_adapter(agg_config, base_context)

        assert "documents" in agg_result
        assert "sources" in agg_result
