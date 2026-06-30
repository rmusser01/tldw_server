"""RAG and web search adapters.

This module includes adapters for search operations:
- rag_search: Execute RAG search using unified pipeline
- web_search: Web search via various engines
- rss_fetch: Fetch RSS/Atom feeds
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from defusedxml import ElementTree as DET
from defusedxml.common import DefusedXmlException
from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.http_client import create_client as _wf_create_client
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.core.Security.egress import is_url_allowed, is_url_allowed_for_tenant
from tldw_Server_API.app.core.testing import is_test_mode, is_truthy
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.rag._config import (
    RAGSearchConfig,
    RSSFetchConfig,
    WebSearchConfig,
)

_RAG_ADAPTER_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeError,
    ValueError,
    json.JSONDecodeError,
)


@registry.register(
    "rag_search",
    category="rag",
    description="Execute RAG search",
    parallelizable=True,
    tags=["rag", "search"],
    config_model=RAGSearchConfig,
)
async def run_rag_search_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Execute a RAG search via the unified pipeline.

    Config:
      - query: str (templated) - Search query
      - sources: list[str] = ["media_db"] - Data sources
      - search_mode: Literal["fts", "vector", "hybrid"] = "hybrid"
      - top_k: int = 10 - Number of results
      - hybrid_alpha: float = 0.7 - Weight for hybrid search
      - min_score: float - Minimum relevance score
      - expand_query: bool - Enable query expansion
      - enable_cache: bool - Enable caching
      - enable_reranking: bool - Enable reranking
      - enable_citations: bool - Enable citation generation
      - enable_generation: bool - Enable answer generation
    Output:
      - {"documents": [...], "metadata": {...}}
    """
    # Cooperative cancel (no-op if cancelled)
    try:
        if callable(context.get("is_cancelled")) and context["is_cancelled"]():
            return {"__status__": "cancelled"}
    except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
        pass

    template_query = config.get("query") or ""
    rendered_query = apply_template_to_string(template_query, context) or template_query

    sources = config.get("sources") or ["media_db"]
    search_mode = config.get("search_mode") or "hybrid"
    top_k = int(config.get("top_k", 10))
    hybrid_alpha = float(config.get("hybrid_alpha", 0.7))

    # Default DB path for media; prefer per-user default
    try:
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
        media_db_path = str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
    except Exception as exc:
        logger.error("Failed to resolve Media DB path for workflow search")
        raise RuntimeError("Failed to resolve Media DB path for workflow search") from exc

    # Map supported options directly to pipeline
    passthrough_keys = {
        # retrieval/search
        "min_score", "expand_query", "expansion_strategies", "spell_check",
        # caching
        "enable_cache", "cache_threshold", "adaptive_cache", "cache_ttl",
        # table processing
        "enable_table_processing", "table_method",
        # context enhancements
        "include_sibling_chunks", "sibling_window",
        "enable_parent_expansion", "include_parent_document", "parent_max_tokens",
        # reranking
        "enable_reranking", "reranking_strategy", "rerank_top_k",
        # citations
        "enable_citations", "citation_style", "include_page_numbers", "enable_chunk_citations",
        # generation
        "enable_generation", "generation_model", "generation_prompt", "max_generation_tokens",
        # security
        "enable_security_filter", "detect_pii", "redact_pii", "sensitivity_level", "content_filter",
        # performance
        "timeout_seconds",
        # quick wins
        "highlight_results", "highlight_query_terms", "track_cost",
    }
    kwargs: dict[str, Any] = {k: v for k, v in (config or {}).items() if k in passthrough_keys}

    result = await unified_rag_pipeline(
        query=rendered_query,
        sources=sources,
        search_mode=search_mode,
        top_k=top_k,
        hybrid_alpha=hybrid_alpha,
        media_db_path=media_db_path,
        **kwargs,
    )

    docs = []
    for d in result.documents:
        try:
            docs.append({
                "id": d.id,
                "content": d.content,
                "metadata": d.metadata,
                "score": float(getattr(d, "score", 0.0) or 0.0),
            })
        except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
            # Be robust to different shapes
            try:
                doc_dict = d if isinstance(d, dict) else json.loads(json.dumps(d, default=str))
            except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
                doc_dict = {"id": "unknown", "content": str(d)}
            docs.append(doc_dict)

    out: dict[str, Any] = {
        "documents": docs,
        "metadata": result.metadata,
        "timings": result.timings,
    }
    if getattr(result, "citations", None):
        out["citations"] = result.citations
    if getattr(result, "generated_answer", None) is not None:
        out["generated_answer"] = result.generated_answer
    return out


def _truncate_content_by_tokens(
    content: str,
    max_tokens: int,
    model: str | None = None,
) -> str:
    """Truncate content by token budget using binary search.

    Uses tiktoken when available; falls back to a conservative char estimate.
    """
    text = str(content or "")
    if not text:
        return ""

    token_budget = max(1, int(max_tokens))

    try:
        import tiktoken  # type: ignore

        if model:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
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

    except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
        # Approximate 4 chars/token if tokenizer is unavailable.
        char_budget = max(1, token_budget) * 4
        if len(text) <= char_budget:
            return text
        return f"{text[:char_budget].rstrip()}\n...[truncated]"


@registry.register(
    "web_search",
    category="rag",
    description="Web search",
    parallelizable=True,
    tags=["rag", "search", "web"],
    config_model=WebSearchConfig,
)
async def run_web_search_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Perform web search using various search engines.

    Config:
      - query: str (templated) - Search query
      - engine: str (or legacy aliases: provider/search_engine)
      - auto_query_rewrite: bool = False - Rewrite query before search using query_rewrite adapter
      - query_rewrite_strategy: str = "simplify"
      - query_rewrite_max_rewrites: int = 1
      - num_results: int = 10 - Number of results
      - content_country: str = "US"
      - search_lang: str = "en"
      - output_lang: str = "en"
      - safesearch: str = "active"
      - date_range: Optional[str]
      - include_domains/exclude_domains (aliases for site_whitelist/site_blacklist)
      - searx_url/searx_json_mode for custom Searx endpoint behavior
      - fetch_content: bool = False - Fetch page content for top search results
      - fetch_limit: int = 3 - Number of results to enrich with fetched content
      - filter_failed_fetches: bool = True - Drop failed fetches from enriched subset
      - max_content_tokens: int = 1200 - Token budget per fetched page
      - summarize: bool = False - Summarize results with LLM
      - api_name: Optional[str] - LLM provider for summarization
    Output:
      - {"results": [{title, link, snippet}], "count": int, "text": str}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    def _cfg(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in config and config.get(key) is not None:
                return config.get(key)
        return default

    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return is_truthy(value)
        return bool(value)

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    query = str(_render(_cfg("query", default="")) or "").strip()
    if not query:
        return {"error": "missing_query"}
    original_query = query

    # Prefer "engine"; keep legacy "provider" for backwards compatibility.
    engine = str(_cfg("engine", "provider", "search_engine", default="google")).strip().lower()
    if engine == "searxng":
        engine = "searx"
    auto_query_rewrite = _as_bool(_cfg("auto_query_rewrite", "rewrite_query", "enable_query_rewrite", default=False), default=False)
    query_rewrite_strategy = str(_cfg("query_rewrite_strategy", "rewrite_strategy", default="simplify")).strip().lower() or "simplify"
    query_rewrite_max_rewrites = max(1, int(_cfg("query_rewrite_max_rewrites", "rewrite_max_rewrites", default=1)))
    query_rewrite_provider = _cfg("query_rewrite_provider", "rewrite_provider")
    query_rewrite_model = _cfg("query_rewrite_model", "rewrite_model")
    num_results = int(_cfg("num_results", "result_count", "max_results", default=10))
    content_country = str(_cfg("content_country", "country", default="US"))
    search_lang = str(_cfg("search_lang", "search_language", default="en"))
    output_lang = str(_cfg("output_lang", "output_language", "ui_lang", default="en"))
    raw_safesearch = _cfg("safesearch", "safe_search", "safeSearch", default="active")
    safesearch = ("active" if raw_safesearch else "off") if isinstance(raw_safesearch, bool) else str(raw_safesearch or "active")
    date_range = _cfg("date_range", "time_range", "timeRange")
    summarize = _as_bool(_cfg("summarize", default=False), default=False)
    fetch_content = _as_bool(_cfg("fetch_content", "fetchContent", default=False), default=False)
    fetch_limit = max(0, int(_cfg("fetch_limit", "fetch_max_results", "max_fetch_results", default=min(num_results, 3))))
    filter_failed_fetches = _as_bool(_cfg("filter_failed_fetches", "drop_failed_fetches", default=True), default=True)
    fetch_timeout_seconds = float(_cfg("fetch_timeout_seconds", "fetch_timeout_s", default=12.0))
    max_content_tokens = max(32, int(_cfg("max_content_tokens", "content_max_tokens", default=1200)))
    tokenizer_model = _cfg("tokenizer_model", "content_tokenizer_model")
    site_whitelist = _cfg("site_whitelist", "include_domains", "includeDomains")
    site_blacklist = _cfg("site_blacklist", "exclude_domains", "excludeDomains")
    exact_terms = _cfg("exactTerms", "exact_terms")
    exclude_terms = _cfg("excludeTerms", "exclude_terms")
    searx_url = _cfg("searx_url", "searxUrl")
    searx_json_mode = _as_bool(_cfg("searx_json_mode", "searxJsonMode", default=False), default=False)

    valid_engines = {
        "google", "duckduckgo", "brave", "kagi", "tavily", "searx", "exa", "firecrawl",
        "baidu", "bing", "yandex", "sogou", "startpage", "stract", "serper",
    }
    if engine not in valid_engines:
        return {"error": f"invalid_engine:{engine}", "valid_engines": list(valid_engines)}

    query_rewrite_meta: dict[str, Any] | None = None
    if auto_query_rewrite:
        try:
            from tldw_Server_API.app.core.Workflows.adapters.rag.query import run_query_rewrite_adapter

            rewrite_result = await run_query_rewrite_adapter(
                {
                    "query": query,
                    "strategy": query_rewrite_strategy,
                    "max_rewrites": query_rewrite_max_rewrites,
                    "provider": query_rewrite_provider,
                    "model": query_rewrite_model,
                },
                context,
            )
            rewrites = rewrite_result.get("rewritten_queries") if isinstance(rewrite_result, dict) else None
            rewritten_query = None
            if isinstance(rewrites, list):
                rewritten_query = next((str(item).strip() for item in rewrites if str(item).strip()), None)
            if rewritten_query:
                query = rewritten_query
            query_rewrite_meta = {
                "enabled": True,
                "strategy": query_rewrite_strategy,
                "original_query": original_query,
                "query_used": query,
                "rewritten": bool(rewritten_query),
            }
            if isinstance(rewrite_result, dict) and rewrite_result.get("error"):
                query_rewrite_meta["error"] = rewrite_result.get("error")
        except Exception:
            query_rewrite_meta = {
                "enabled": True,
                "strategy": query_rewrite_strategy,
                "original_query": original_query,
                "query_used": query,
                "rewritten": False,
                "error": "query_rewrite_error",
            }

    # Test mode simulation
    if is_test_mode():
        mock_results = [
            {"title": f"Result 1 for {query}", "link": "https://example.com/1", "snippet": f"Snippet about {query}"},
            {"title": f"Result 2 for {query}", "link": "https://example.com/2", "snippet": f"More info about {query}"},
        ]
        return {
            "results": mock_results,
            "count": len(mock_results),
            "query": query,
            "engine": engine,
            "text": "\n".join([f"- {r['title']}: {r['snippet']}" for r in mock_results]),
            "simulated": True,
        }

    try:
        from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

        raw_results = ws.perform_websearch(
            search_engine=engine,
            search_query=query,
            content_country=content_country,
            search_lang=search_lang,
            output_lang=output_lang,
            result_count=num_results,
            date_range=date_range,
            safesearch=safesearch,
            site_blacklist=site_blacklist,
            site_whitelist=site_whitelist,
            exactTerms=exact_terms,
            excludeTerms=exclude_terms,
            search_params={
                "searx_url": searx_url,
                "searx_json_mode": searx_json_mode,
            },
        )

        if not isinstance(raw_results, dict):
            return {"error": "search_failed", "query": query}

        if raw_results.get("processing_error"):
            return {"error": "search_error", "query": query}

        results = raw_results.get("results") or []
        formatted_results = []
        for r in results:
            formatted_results.append({
                "title": r.get("title") or "",
                "link": r.get("link") or r.get("url") or "",
                "snippet": r.get("snippet") or r.get("description") or r.get("content") or "",
            })

        fetch_summary: dict[str, Any] | None = None
        if fetch_content and formatted_results:
            from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article

            async def _fetch_one(idx: int, item: dict[str, Any]) -> tuple[int, str | None, str | None]:
                link = str(item.get("link") or "").strip()
                if not link:
                    return idx, None, "missing_link"
                try:
                    scraped = await asyncio.wait_for(scrape_article(link), timeout=fetch_timeout_seconds)
                except asyncio.TimeoutError:
                    return idx, None, "timeout"
                except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
                    return idx, None, "fetch_failed"
                if not isinstance(scraped, dict):
                    return idx, None, "invalid_scrape_payload"
                scraped_content = str(scraped.get("content") or "").strip()
                if not scraped_content:
                    return idx, None, "empty_content"
                return idx, _truncate_content_by_tokens(scraped_content, max_content_tokens, model=tokenizer_model), None

            targets = list(enumerate(formatted_results[:fetch_limit]))
            fetched = await asyncio.gather(*[_fetch_one(idx, item) for idx, item in targets])
            fetch_errors: list[dict[str, Any]] = []
            fetched_count = 0
            success_indices: list[int] = []
            for idx, content, err in fetched:
                if content:
                    formatted_results[idx]["content"] = content
                    fetched_count += 1
                    success_indices.append(idx)
                    if not formatted_results[idx].get("snippet"):
                        formatted_results[idx]["snippet"] = content[:300]
                elif err:
                    fetch_errors.append({"index": idx, "error": err})

            dropped_count = 0
            if filter_failed_fetches and targets:
                attempted_set = {idx for idx, _ in targets}
                success_set = set(success_indices)
                dropped_count = len(attempted_set - success_set)
                formatted_results = [
                    item
                    for idx, item in enumerate(formatted_results)
                    if idx not in attempted_set or idx in success_set
                ]

            fetch_summary = {
                "enabled": True,
                "attempted": len(targets),
                "fetched": fetched_count,
                "dropped_failed": dropped_count,
                "filter_failed_fetches": filter_failed_fetches,
                "max_content_tokens": max_content_tokens,
            }
            if fetch_errors:
                fetch_summary["errors"] = fetch_errors[:10]

        # Combine snippets into text for downstream steps
        text = "\n".join([
            f"- {r['title']}: {r.get('content') or r.get('snippet') or ''}"
            for r in formatted_results
            if r.get("title")
        ])

        out: dict[str, Any] = {
            "results": formatted_results,
            "count": len(formatted_results),
            "query": query,
            "engine": engine,
            "text": text,
            "total_found": raw_results.get("total_results_found", len(formatted_results)),
        }
        if query != original_query:
            out["original_query"] = original_query
        if query_rewrite_meta is not None:
            out["query_rewrite"] = query_rewrite_meta
        if fetch_summary is not None:
            out["fetch_content"] = fetch_summary

        # Optional summarization
        if summarize and text:
            try:
                api_name = _render(_cfg("api_name", "api_provider", "apiProvider", default="openai"))
                summary = ws.summarize(
                    input_data=text,
                    custom_prompt_arg=f"Summarize the following search results for the query '{query}':",
                    api_name=api_name,
                )
                out["summary"] = summary
                out["text"] = summary  # Replace text with summary for downstream
            except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
                logger.debug("Web search summarization failed")
                out["summary_error"] = "summary_failed"

        return out

    except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
        logger.error("Web search adapter error")
        return {"error": "web_search_error"}


@registry.register(
    "rss_fetch",
    category="rag",
    description="Fetch RSS/Atom feeds",
    parallelizable=True,
    tags=["rag", "feed"],
    config_model=RSSFetchConfig,
)
async def run_rss_fetch_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Fetch RSS/Atom feeds and return items.

    Config:
      - urls: list[str] | str - Feed URLs (newline/comma separated)
      - limit: int = 10 - Maximum items to return
      - include_content: bool = True - Include summary/content in results
    Output:
      - {"results": [{title, link, summary, published}], "count": int, "text": str}
    """
    urls_cfg = config.get("urls")
    if isinstance(urls_cfg, list):
        urls = [str(u).strip() for u in urls_cfg if str(u).strip()]
    else:
        raw = str(urls_cfg or "").strip()
        if raw:
            # split by newline or comma
            parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
            urls = parts
        else:
            urls = []
    limit = int(config.get("limit", 10))
    include_content = bool(config.get("include_content", True))

    # Test-friendly behavior without network
    if is_test_mode():
        fake = [{"title": "Test Item", "link": "https://example.com/x", "summary": "Test", "published": None}]
        return {"results": fake[:limit], "count": min(limit, len(fake)), "text": fake[0]["summary"]}

    results: list[dict] = []
    if not urls:
        return {"results": [], "count": 0}
    try:
        from urllib.parse import urlparse
        for u in urls:
            try:
                if not (u.startswith("http://") or u.startswith("https://")):
                    continue
                tenant_id = str(context.get("tenant_id") or "default") if isinstance(context, dict) else "default"
                allowed = False
                try:
                    allowed = is_url_allowed_for_tenant(u, tenant_id)
                except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
                    allowed = is_url_allowed(u)
                if not allowed:
                    continue
                urlparse(u).hostname or ""
                timeout = float(os.getenv("WORKFLOWS_RSS_TIMEOUT", "8"))
                with _wf_create_client(timeout=timeout) as client:
                    resp = client.get(u)
                    if resp.status_code // 100 != 2:
                        continue
                    text = resp.text
                # Parse as XML (RSS or Atom)
                try:
                    root = DET.fromstring(text)
                except (DefusedXmlException, TypeError, ValueError):
                    continue
                # Heuristic: RSS <item> or Atom <entry>
                items = root.findall('.//item')
                if not items:
                    items = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                for it in items:
                    title = None
                    link = None
                    summary = None
                    published = None
                    guid = None
                    # Namespaces
                    def _find_text(node, names):
                        for n in names:
                            x = node.find(n)
                            if x is not None and (x.text or "").strip():
                                return x.text.strip()
                        return None
                    title = _find_text(it, ["title", "{http://www.w3.org/2005/Atom}title"]) or ""
                    # Atom links are in attributes
                    lnode = it.find("link")
                    if lnode is not None and (lnk := lnode.get("href")):
                        link = lnk
                    else:
                        link = _find_text(it, ["link", "{http://www.w3.org/2005/Atom}link"]) or ""
                    summary = _find_text(it, ["description", "{http://www.w3.org/2005/Atom}summary", "{http://www.w3.org/2005/Atom}content"]) or ""
                    published = _find_text(it, ["pubDate", "{http://www.w3.org/2005/Atom}updated", "{http://www.w3.org/2005/Atom}published"]) or None
                    guid = _find_text(it, ["guid", "{http://www.w3.org/2005/Atom}id"]) or None
                    rec = {"title": title, "link": link}
                    if include_content:
                        rec["summary"] = summary
                    if published:
                        rec["published"] = published
                    if guid:
                        rec["guid"] = guid
                    results.append(rec)
            except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
                continue
        results = results[:limit]
        text_concat = "\n\n".join([r.get("summary") or r.get("title") or "" for r in results if (r.get("summary") or r.get("title"))])
        return {"results": results, "count": len(results), "text": text_concat}
    except _RAG_ADAPTER_NONCRITICAL_EXCEPTIONS:
        return {"error": "rss_error"}


@registry.register(
    "atom_fetch",
    category="rag",
    description="Fetch Atom feeds (alias)",
    parallelizable=True,
    tags=["rag", "feed"],
    config_model=RSSFetchConfig,
)
async def run_atom_fetch_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Fetch Atom feeds (alias for rss_fetch).

    Config:
      - urls: list[str] | str - Feed URLs
      - limit: int = 10
      - include_content: bool = True
    Output:
      - {"results": [...], "count": int, "text": str}
    """
    return await run_rss_fetch_adapter(config, context)
