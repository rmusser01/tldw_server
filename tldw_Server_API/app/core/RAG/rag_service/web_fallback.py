"""
Web Search Fallback for Self-Correcting RAG

This module provides web search fallback when local retrieval fails to find
relevant documents. Uses the existing WebSearch_APIs infrastructure.

Part of the Self-Correcting RAG feature set (Stage 3).
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from loguru import logger

from .types import DataSource, Document


@dataclass
class WebFallbackConfig:
    """Configuration for web search fallback."""

    engine: str = "duckduckgo"
    result_count: int = 5
    content_country: str = "US"
    search_lang: str = "en"
    output_lang: str = "en"
    max_content_chars: int = 2000  # deprecated fallback knob (chars)
    max_content_tokens: int = 500
    tokenizer_model: str | None = None
    subquery_generation: bool = False
    safesearch: str = "active"
    timeout_seconds: float = 30.0


@dataclass
class WebFallbackResult:
    """Result of web search fallback."""

    documents: list[Document]
    search_time_ms: int
    result_count: int
    engine_used: str
    query_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _truncate_content_by_tokens(
    content: str,
    max_tokens: int,
    model: str | None = None,
) -> str:
    """Truncate text by token budget using binary search with fallback."""
    text = str(content or "")
    if not text:
        return ""

    token_budget = max(1, int(max_tokens))

    try:
        import tiktoken  # type: ignore

        if model:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        if len(encoding.encode(text)) <= token_budget:
            return text

        left, right = 0, len(text)
        while left < right:
            mid = (left + right + 1) // 2
            if len(encoding.encode(text[:mid])) <= token_budget:
                left = mid
            else:
                right = mid - 1

        truncated = text[:left].rstrip()
        return f"{truncated}\n...[truncated]" if left < len(text) else truncated

    except Exception:
        char_budget = max(1, token_budget) * 4
        if len(text) <= char_budget:
            return text
        return f"{text[:char_budget].rstrip()}\n...[truncated]"


async def web_search_fallback(
    query: str,
    config: Optional[WebFallbackConfig] = None,
) -> WebFallbackResult:
    """
    Perform web search fallback when local retrieval fails.

    This function calls the existing generate_and_search function and converts
    the results to Document objects with source=DataSource.WEB_CONTENT.

    Args:
        query: The search query
        config: Optional configuration for web search

    Returns:
        WebFallbackResult with converted documents
    """
    if config is None:
        config = WebFallbackConfig()

    start_time = time.time()

    try:
        # Import the web search function
        from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import generate_and_search

        # Build search parameters
        search_params = {
            "engine": config.engine,
            "content_country": config.content_country,
            "search_lang": config.search_lang,
            "output_lang": config.output_lang,
            "result_count": config.result_count,
            "subquery_generation": config.subquery_generation,
            "safesearch": config.safesearch,
        }

        # Run the synchronous search in a thread pool
        search_result = await asyncio.wait_for(
            asyncio.to_thread(generate_and_search, query, search_params),
            timeout=config.timeout_seconds,
        )

        search_time_ms = int((time.time() - start_time) * 1000)

        # Extract web search results
        web_results = search_result.get("web_search_results_dict", {})
        raw_results = web_results.get("results", [])

        # Convert to Document objects
        documents = _convert_web_results_to_documents(
            raw_results,
            query,
            config.max_content_chars,
            max_content_tokens=config.max_content_tokens,
            tokenizer_model=config.tokenizer_model,
        )

        return WebFallbackResult(
            documents=documents,
            search_time_ms=search_time_ms,
            result_count=len(documents),
            engine_used=config.engine,
            query_used=query,
            metadata={
                "total_raw_results": web_results.get("total_results_found", 0),
                "search_time_seconds": web_results.get("search_time", 0.0),
                "sub_queries": search_result.get("sub_query_dict", {}).get("sub_questions", []),
            },
        )

    except asyncio.TimeoutError:
        logger.warning(f"Web search fallback timed out after {config.timeout_seconds}s")
        return WebFallbackResult(
            documents=[],
            search_time_ms=int((time.time() - start_time) * 1000),
            result_count=0,
            engine_used=config.engine,
            query_used=query,
            metadata={"error": "timeout"},
        )

    except ImportError:
        logger.warning("WebSearch_APIs module not available for web fallback")
        return WebFallbackResult(
            documents=[],
            search_time_ms=int((time.time() - start_time) * 1000),
            result_count=0,
            engine_used=config.engine,
            query_used=query,
            metadata={"error": "module_not_available"},
        )

    except Exception as e:  # noqa: BLE001 - fallback should be resilient to unexpected failures
        logger.warning("Web search fallback failed: search execution error")
        return WebFallbackResult(
            documents=[],
            search_time_ms=int((time.time() - start_time) * 1000),
            result_count=0,
            engine_used=config.engine,
            query_used=query,
            metadata={"error": "search_failed"},
        )


def _convert_web_results_to_documents(
    raw_results: list[dict[str, Any]],
    query: str,
    max_content_chars: int,
    max_content_tokens: int | None = None,
    tokenizer_model: str | None = None,
) -> list[Document]:
    """
    Convert raw web search results to Document objects.

    Args:
        raw_results: Raw results from web search
        query: Original query
        max_content_chars: Deprecated char limit fallback for content
        max_content_tokens: Maximum tokens to include in content
        tokenizer_model: Optional tokenizer model for token counting

    Returns:
        List of Document objects
    """
    documents = []

    for idx, result in enumerate(raw_results):
        try:
            # Extract content from result
            title = result.get("title", "")
            snippet = result.get("snippet", "") or result.get("description", "")
            url = result.get("url", "") or result.get("link", "")
            body = result.get("body", "") or result.get("content", "")

            # Build content from available fields
            content_parts = []
            if title:
                content_parts.append(f"Title: {title}")
            if snippet:
                content_parts.append(f"Summary: {snippet}")
            if body:
                # Prefer token-aware truncation with max_content_chars fallback.
                token_budget = max_content_tokens
                if token_budget is None:
                    token_budget = max(32, int(max(1, max_content_chars) / 4))
                truncated_body = _truncate_content_by_tokens(str(body), token_budget, model=tokenizer_model)
                content_parts.append(f"Content: {truncated_body}")

            content = "\n\n".join(content_parts)

            if not content.strip():
                continue

            # Create Document
            doc_id = f"web_{uuid.uuid4().hex[:8]}_{idx}"
            doc = Document(
                id=doc_id,
                content=content,
                source=DataSource.WEB_CONTENT,
                score=max(0.0, 1.0 - (idx * 0.05)),  # Decrease score by position
                metadata={
                    "title": title,
                    "url": url,
                    "snippet": snippet[:500] if snippet else "",
                    "source": "web_search",
                    "query": query,
                    "position": idx + 1,
                },
            )
            documents.append(doc)

        except Exception:  # noqa: BLE001 - best-effort conversion of results
            logger.warning("Failed to convert web result {}: conversion error", idx)
            continue

    return documents


def merge_web_results(
    local_docs: list[Document],
    web_docs: list[Document],
    strategy: Literal["prepend", "append", "interleave"] = "prepend",
    max_total: Optional[int] = None,
) -> list[Document]:
    """
    Merge local and web documents according to a strategy.

    Args:
        local_docs: Documents from local retrieval
        web_docs: Documents from web search
        strategy: How to merge ("prepend", "append", "interleave")
        max_total: Maximum total documents to return

    Returns:
        Merged list of documents
    """
    if strategy == "prepend":
        # Web results first, then local
        merged = web_docs + local_docs
    elif strategy == "append":
        # Local results first, then web
        merged = local_docs + web_docs
    elif strategy == "interleave":
        # Alternate between web and local
        merged = []
        max_len = max(len(local_docs), len(web_docs))
        for i in range(max_len):
            if i < len(web_docs):
                merged.append(web_docs[i])
            if i < len(local_docs):
                merged.append(local_docs[i])
    else:
        # Default to prepend
        merged = web_docs + local_docs

    if max_total is not None:
        merged = merged[:max_total]

    return merged


# Convenience function for pipeline integration
async def fallback_to_web_search(
    query: str,
    local_docs: list[Document],
    relevance_signal: float,
    threshold: float = 0.25,
    engine: str = "duckduckgo",
    result_count: int = 5,
    merge_strategy: Literal["prepend", "append", "interleave"] = "prepend",
    max_total: Optional[int] = None,
) -> tuple[list[Document], dict[str, Any]]:
    """
    Convenience function to conditionally fall back to web search.

    Args:
        query: The search query
        local_docs: Documents from local retrieval
        relevance_signal: Current relevance score (e.g., from reranker calibration)
        threshold: Threshold below which to trigger web fallback
        engine: Web search engine to use
        result_count: Number of web results to fetch
        merge_strategy: How to merge local and web results
        max_total: Maximum total documents to return

    Returns:
        Tuple of (merged_documents, metadata)
    """
    metadata: dict[str, Any] = {
        "web_fallback_enabled": True,
        "relevance_signal": relevance_signal,
        "threshold": threshold,
        "triggered": False,
    }

    # Check if we should trigger web fallback
    if relevance_signal >= threshold:
        metadata["triggered"] = False
        return local_docs, metadata

    # Trigger web fallback
    metadata["triggered"] = True

    config = WebFallbackConfig(
        engine=engine,
        result_count=result_count,
    )

    result = await web_search_fallback(query, config)

    if not result.documents:
        metadata["web_search_failed"] = True
        metadata["error"] = result.metadata.get("error")
        return local_docs, metadata

    # Merge results
    merged = merge_web_results(
        local_docs=local_docs,
        web_docs=result.documents,
        strategy=merge_strategy,
        max_total=max_total,
    )

    metadata["web_results_count"] = result.result_count
    metadata["search_time_ms"] = result.search_time_ms
    metadata["engine_used"] = result.engine_used
    metadata["merged_count"] = len(merged)
    metadata["merge_strategy"] = merge_strategy

    return merged, metadata
