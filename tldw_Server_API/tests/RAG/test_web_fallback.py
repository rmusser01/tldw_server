"""
Tests for Web Search Fallback (Self-Correcting RAG Stage 3)

These tests cover:
- WebFallbackConfig defaults and customization
- Web search fallback function
- Document conversion from web results
- Merge strategies
- Convenience function for pipeline integration
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from tldw_Server_API.app.core.RAG.rag_service import web_fallback as web_fallback_module
from tldw_Server_API.app.core.RAG.rag_service.web_fallback import (
    WebFallbackConfig,
    WebFallbackResult,
    web_search_fallback,
    merge_web_results,
    fallback_to_web_search,
    _convert_web_results_to_documents,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource


@dataclass
class MockDocument:
    """Mock document for testing."""
    id: str
    content: str
    score: float = 0.5
    source: DataSource = DataSource.MEDIA_DB
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestWebFallbackConfig:
    """Tests for WebFallbackConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = WebFallbackConfig()

        assert config.engine == "duckduckgo"
        assert config.result_count == 5
        assert config.content_country == "US"
        assert config.search_lang == "en"
        assert config.output_lang == "en"
        assert config.max_content_chars == 2000
        assert config.max_content_tokens == 500
        assert config.tokenizer_model is None
        assert config.subquery_generation is False
        assert config.safesearch == "active"
        assert config.timeout_seconds == 30.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WebFallbackConfig(
            engine="google",
            result_count=10,
            content_country="GB",
            search_lang="de",
            max_content_chars=5000,
            max_content_tokens=900,
            tokenizer_model="gpt-4o-mini",
            timeout_seconds=60.0,
        )

        assert config.engine == "google"
        assert config.result_count == 10
        assert config.content_country == "GB"
        assert config.search_lang == "de"
        assert config.max_content_chars == 5000
        assert config.max_content_tokens == 900
        assert config.tokenizer_model == "gpt-4o-mini"
        assert config.timeout_seconds == 60.0


class TestConvertWebResultsToDocuments:
    """Tests for converting web search results to Documents."""

    def test_convert_basic_result(self):
        """Test converting a basic web search result."""
        raw_results = [
            {
                "title": "Test Page",
                "snippet": "This is a test snippet.",
                "url": "https://example.com/test",
            }
        ]

        docs = _convert_web_results_to_documents(raw_results, "test query", 2000)

        assert len(docs) == 1
        assert "Test Page" in docs[0].content
        assert "test snippet" in docs[0].content
        assert docs[0].source == DataSource.WEB_CONTENT
        assert docs[0].metadata["url"] == "https://example.com/test"
        assert docs[0].metadata["title"] == "Test Page"

    def test_convert_multiple_results(self):
        """Test converting multiple results with decreasing scores."""
        raw_results = [
            {"title": "First", "snippet": "First result", "url": "https://example.com/1"},
            {"title": "Second", "snippet": "Second result", "url": "https://example.com/2"},
            {"title": "Third", "snippet": "Third result", "url": "https://example.com/3"},
        ]

        docs = _convert_web_results_to_documents(raw_results, "test", 2000)

        assert len(docs) == 3
        # Scores should decrease by position
        assert docs[0].score > docs[1].score
        assert docs[1].score > docs[2].score

    def test_convert_with_body_content(self):
        """Test converting results with body content."""
        raw_results = [
            {
                "title": "Full Article",
                "snippet": "Short snippet",
                "url": "https://example.com/article",
                "body": "This is the full body content of the article.",
            }
        ]

        docs = _convert_web_results_to_documents(raw_results, "test", 2000)

        assert len(docs) == 1
        assert "full body content" in docs[0].content

    def test_convert_truncates_long_content(self):
        """Test that long content is truncated."""
        long_body = "A" * 5000
        raw_results = [
            {
                "title": "Long Article",
                "snippet": "Snippet",
                "url": "https://example.com",
                "body": long_body,
            }
        ]

        docs = _convert_web_results_to_documents(raw_results, "test", 1000)

        assert len(docs) == 1
        # Body should be truncated
        assert len(docs[0].content) < 5000
        assert "..." in docs[0].content

    def test_convert_respects_token_budget_override(self):
        """Test explicit max_content_tokens overrides char-based fallback."""
        long_body = "B" * 5000
        raw_results = [
            {
                "title": "Token Limited Article",
                "snippet": "Snippet",
                "url": "https://example.com",
                "body": long_body,
            }
        ]

        docs = _convert_web_results_to_documents(
            raw_results,
            "test",
            max_content_chars=5000,
            max_content_tokens=10,
        )

        assert len(docs) == 1
        assert "Content:" in docs[0].content
        assert len(docs[0].content) < 400
        assert "...[truncated]" in docs[0].content

    def test_convert_empty_results(self):
        """Test converting empty results."""
        docs = _convert_web_results_to_documents([], "test", 2000)
        assert docs == []

    def test_convert_skips_empty_content(self):
        """Test that results without content are skipped."""
        raw_results = [
            {"title": "", "snippet": "", "url": "https://empty.com"},
            {"title": "Has Content", "snippet": "Valid content", "url": "https://valid.com"},
        ]

        docs = _convert_web_results_to_documents(raw_results, "test", 2000)

        assert len(docs) == 1
        assert "Valid content" in docs[0].content

    def test_convert_skips_failed_result_and_sanitizes_conversion_log(self):
        """Test failed result conversion skips item without logging sensitive exception details."""
        secret = "sk-web-fallback-secret-123"
        private_path = "/tmp/private/user_databases/42/Media_DB_v2.db"

        class SecretBearingResult(dict):
            def get(self, key, default=None):
                raise RuntimeError(f"failed opening {private_path} with api_key={secret}")

        raw_results = [
            {"title": "First", "snippet": "First result", "url": "https://example.com/1"},
            SecretBearingResult(),
            {"title": "Second", "snippet": "Second result", "url": "https://example.com/2"},
        ]
        messages = []
        sink_id = web_fallback_module.logger.add(
            lambda message: messages.append(message.record["message"]),
            level="WARNING",
        )

        try:
            docs = _convert_web_results_to_documents(raw_results, "test", 2000)
        finally:
            web_fallback_module.logger.remove(sink_id)

        assert len(docs) == 2
        assert "First result" in docs[0].content
        assert "Second result" in docs[1].content

        log_text = "\n".join(messages)
        assert "Failed to convert web result 1:" in log_text
        assert "failed opening" not in log_text
        assert private_path not in log_text
        assert secret not in log_text


class TestMergeWebResults:
    """Tests for merging local and web documents."""

    @pytest.fixture
    def local_docs(self) -> List[Document]:
        """Create local documents."""
        return [
            Document(id="local1", content="Local content 1", source=DataSource.MEDIA_DB, metadata={}),
            Document(id="local2", content="Local content 2", source=DataSource.MEDIA_DB, metadata={}),
        ]

    @pytest.fixture
    def web_docs(self) -> List[Document]:
        """Create web documents."""
        return [
            Document(id="web1", content="Web content 1", source=DataSource.WEB_CONTENT, metadata={}),
            Document(id="web2", content="Web content 2", source=DataSource.WEB_CONTENT, metadata={}),
        ]

    def test_merge_prepend(self, local_docs, web_docs):
        """Test prepend merge strategy."""
        merged = merge_web_results(local_docs, web_docs, strategy="prepend")

        assert len(merged) == 4
        assert merged[0].id == "web1"
        assert merged[1].id == "web2"
        assert merged[2].id == "local1"
        assert merged[3].id == "local2"

    def test_merge_append(self, local_docs, web_docs):
        """Test append merge strategy."""
        merged = merge_web_results(local_docs, web_docs, strategy="append")

        assert len(merged) == 4
        assert merged[0].id == "local1"
        assert merged[1].id == "local2"
        assert merged[2].id == "web1"
        assert merged[3].id == "web2"

    def test_merge_interleave(self, local_docs, web_docs):
        """Test interleave merge strategy."""
        merged = merge_web_results(local_docs, web_docs, strategy="interleave")

        assert len(merged) == 4
        assert merged[0].id == "web1"
        assert merged[1].id == "local1"
        assert merged[2].id == "web2"
        assert merged[3].id == "local2"

    def test_merge_with_max_total(self, local_docs, web_docs):
        """Test merge with max_total limit."""
        merged = merge_web_results(local_docs, web_docs, strategy="prepend", max_total=2)

        assert len(merged) == 2
        assert merged[0].id == "web1"
        assert merged[1].id == "web2"

    def test_merge_empty_local(self, web_docs):
        """Test merge when local docs are empty."""
        merged = merge_web_results([], web_docs, strategy="prepend")

        assert len(merged) == 2
        assert all(d.source == DataSource.WEB_CONTENT for d in merged)

    def test_merge_empty_web(self, local_docs):
        """Test merge when web docs are empty."""
        merged = merge_web_results(local_docs, [], strategy="prepend")

        assert len(merged) == 2
        assert all(d.source == DataSource.MEDIA_DB for d in merged)


class TestWebSearchFallback:
    """Tests for the web_search_fallback function."""

    @pytest.mark.asyncio
    async def test_fallback_success(self):
        """Test successful web search fallback."""
        mock_search_result = {
            "web_search_results_dict": {
                "results": [
                    {"title": "Result 1", "snippet": "Snippet 1", "url": "https://example.com/1"},
                    {"title": "Result 2", "snippet": "Snippet 2", "url": "https://example.com/2"},
                ],
                "total_results_found": 100,
                "search_time": 0.5,
            },
            "sub_query_dict": {"sub_questions": []},
        }

        with patch(
            "tldw_Server_API.app.core.RAG.rag_service.web_fallback.asyncio.to_thread",
            return_value=mock_search_result,
        ):
            config = WebFallbackConfig(result_count=2)
            result = await web_search_fallback("test query", config)

            assert isinstance(result, WebFallbackResult)
            assert len(result.documents) == 2
            assert result.result_count == 2
            assert result.engine_used == "duckduckgo"
            assert result.query_used == "test query"

    @pytest.mark.asyncio
    async def test_fallback_timeout(self):
        """Test web search fallback timeout handling."""
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(10)
            return {}

        with patch(
            "tldw_Server_API.app.core.RAG.rag_service.web_fallback.asyncio.to_thread",
            side_effect=asyncio.TimeoutError,
        ):
            config = WebFallbackConfig(timeout_seconds=0.1)
            result = await web_search_fallback("test query", config)

            assert result.documents == []
            assert result.metadata.get("error") == "timeout"

    @pytest.mark.asyncio
    async def test_fallback_import_error(self):
        """Test web search fallback when module is unavailable."""
        with patch(
            "tldw_Server_API.app.core.RAG.rag_service.web_fallback.asyncio.to_thread",
            side_effect=ImportError("Module not found"),
        ):
            result = await web_search_fallback("test query")

            # Should handle the error gracefully
            assert result.documents == []

    @pytest.mark.asyncio
    async def test_fallback_failure_log_sanitizes_exception_details(self):
        """Test fallback failure log does not expose raw exception details."""
        secret = "sk-web-fallback-secret-456"
        private_path = "/tmp/private/user_databases/42/Media_DB_v2.db"
        sensitive_query = "summarize private token"
        raw_error = RuntimeError(f"failed opening {private_path} with api_key={secret} query={sensitive_query}")
        messages = []
        sink_id = web_fallback_module.logger.add(
            lambda message: messages.append(message.record["message"]),
            level="WARNING",
        )

        try:
            with patch(
                "tldw_Server_API.app.core.RAG.rag_service.web_fallback.asyncio.to_thread",
                side_effect=raw_error,
            ):
                result = await web_search_fallback(sensitive_query)
        finally:
            web_fallback_module.logger.remove(sink_id)

        assert result.documents == []
        assert result.metadata == {"error": "search_failed"}

        log_text = "\n".join(messages)
        assert "Web search fallback failed" in log_text
        assert "failed opening" not in log_text
        assert private_path not in log_text
        assert secret not in log_text
        assert sensitive_query not in log_text


class TestFallbackToWebSearch:
    """Tests for the convenience fallback_to_web_search function."""

    @pytest.fixture
    def local_docs(self) -> List[Document]:
        """Create local documents."""
        return [
            Document(id="doc1", content="Content 1", score=0.3, source=DataSource.MEDIA_DB, metadata={}),
            Document(id="doc2", content="Content 2", score=0.2, source=DataSource.MEDIA_DB, metadata={}),
        ]

    @pytest.mark.asyncio
    async def test_no_fallback_above_threshold(self, local_docs):
        """Test that fallback is not triggered when relevance is above threshold."""
        docs, metadata = await fallback_to_web_search(
            query="test query",
            local_docs=local_docs,
            relevance_signal=0.6,
            threshold=0.25,
        )

        assert docs == local_docs
        assert metadata["triggered"] is False
        assert metadata["relevance_signal"] == 0.6

    @pytest.mark.asyncio
    async def test_fallback_below_threshold(self, local_docs):
        """Test that fallback is triggered when relevance is below threshold."""
        mock_search_result = {
            "web_search_results_dict": {
                "results": [
                    {"title": "Web Result", "snippet": "Web content", "url": "https://example.com"},
                ],
                "total_results_found": 10,
                "search_time": 0.3,
            },
            "sub_query_dict": {"sub_questions": []},
        }

        with patch(
            "tldw_Server_API.app.core.RAG.rag_service.web_fallback.asyncio.to_thread",
            return_value=mock_search_result,
        ):
            docs, metadata = await fallback_to_web_search(
                query="test query",
                local_docs=local_docs,
                relevance_signal=0.1,
                threshold=0.25,
            )

            assert metadata["triggered"] is True
            # Should have merged docs (web + local)
            assert len(docs) > len(local_docs)

    @pytest.mark.asyncio
    async def test_fallback_merge_strategy_append(self, local_docs):
        """Test fallback with append merge strategy."""
        mock_search_result = {
            "web_search_results_dict": {
                "results": [
                    {"title": "Web Result", "snippet": "Web content", "url": "https://example.com"},
                ],
                "total_results_found": 10,
                "search_time": 0.3,
            },
            "sub_query_dict": {"sub_questions": []},
        }

        with patch(
            "tldw_Server_API.app.core.RAG.rag_service.web_fallback.asyncio.to_thread",
            return_value=mock_search_result,
        ):
            docs, metadata = await fallback_to_web_search(
                query="test query",
                local_docs=local_docs,
                relevance_signal=0.1,
                threshold=0.25,
                merge_strategy="append",
            )

            assert metadata["merge_strategy"] == "append"
            # First docs should be local
            assert docs[0].id == "doc1"


class TestWebFallbackResultDataclass:
    """Tests for WebFallbackResult dataclass."""

    def test_create_result(self):
        """Test creating a WebFallbackResult."""
        docs = [Document(id="test", content="content", source=DataSource.WEB_CONTENT, metadata={})]
        result = WebFallbackResult(
            documents=docs,
            search_time_ms=500,
            result_count=1,
            engine_used="duckduckgo",
            query_used="test query",
        )

        assert result.documents == docs
        assert result.search_time_ms == 500
        assert result.result_count == 1
        assert result.engine_used == "duckduckgo"
        assert result.query_used == "test query"
        assert result.metadata == {}

    def test_create_result_with_metadata(self):
        """Test creating WebFallbackResult with metadata."""
        result = WebFallbackResult(
            documents=[],
            search_time_ms=0,
            result_count=0,
            engine_used="google",
            query_used="query",
            metadata={"error": "timeout"},
        )

        assert result.metadata == {"error": "timeout"}
