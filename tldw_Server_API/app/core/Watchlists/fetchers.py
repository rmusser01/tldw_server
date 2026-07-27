"""
Watchlists fetch helpers for RSS/Atom feeds and HTML scraping.

Highlights:
- RSS fetch: reuses workflows adapter logic for URL policy checks and XML parsing.
  TEST_MODE returns static items so offline unit tests stay deterministic.
- Site fetch: relies on the blocking article extractor plus optional rule-based
  list scraping informed by FreshRSS-style XPath/CSS selectors.

Returned item structure (normalized):
- RSS items: { 'title': str, 'url': str, 'summary': Optional[str], 'published': Optional[str] }
- Site articles: { 'title': str, 'url': str, 'content': str, 'author': Optional[str] }
- Scraped list items: { 'title': str, 'url': str, 'summary': Optional[str], 'content': Optional[str], ... }
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urljoin

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
from loguru import logger
from lxml import html
from lxml.html import HtmlElement

from tldw_Server_API.app.core.Security.egress import is_url_allowed, is_url_allowed_for_tenant
from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode
from tldw_Server_API.app.core.Web_Scraping.selectors import (
    clear_selector_caches,
    extract_schema_fields,
    get_selector_cache_stats,
    validate_selector_rules,
)
from tldw_Server_API.app.core.Web_Scraping.selectors.engine import (
    _coerce_value,
    _ensure_sequence,
    _extract_value,
    _select_nodes,
    reload_selector_guardrails_from_env,
)
from tldw_Server_API.app.core.Web_Scraping.selectors.schema import _normalize_datetime

# Keep intentional compatibility re-exports visible to static analysis.
_SELECTOR_COMPATIBILITY_EXPORTS = (
    clear_selector_caches,
    extract_schema_fields,
    get_selector_cache_stats,
    reload_selector_guardrails_from_env,
    validate_selector_rules,
)

_WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS = (
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    ET.ParseError,
    DefusedXmlException,
)

def _in_test_mode() -> bool:
    return _is_test_mode()


async def _close_response(resp: Any) -> None:
    if resp is None:
        return
    close = getattr(resp, "aclose", None)
    if callable(close):
        await close()
        return
    close = getattr(resp, "close", None)
    if callable(close):
        close()

def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        return default


def parse_scraped_items(html_text: str, base_url: str, rules: dict[str, Any]) -> dict[str, Any]:
    """Parse HTML into structured items using XPath/CSS rules.

    Returns dict: { "items": [...], "next_pages": [...] } to support pagination-aware callers.
    """
    result: dict[str, Any] = {"items": [], "next_pages": []}
    if not html_text:
        return result
    try:
        document = html.fromstring(html_text)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"parse_scraped_items HTML parse failed: {exc}")
        return result

    with contextlib.suppress(_WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS):
        document.make_links_absolute(base_url)

    def _gather_items(rule_set: dict[str, Any], *, seen: set[str], items: list[dict[str, Any]], limit: int | None) -> None:
        entry_selectors = (
            rule_set.get("entry_xpath")
            or rule_set.get("item_xpath")
            or rule_set.get("items_xpath")
            or rule_set.get("entry_selector")
            or rule_set.get("item_selector")
            or rules.get("entry_xpath")
            or rules.get("item_xpath")
            or rules.get("items_xpath")
            or rules.get("entry_selector")
            or rules.get("item_selector")
        )
        nodes: list[HtmlElement] = []
        for selector in _ensure_sequence(entry_selectors) or ["//article", "//item"]:
            nodes.extend([n for n in _select_nodes(document, selector) if isinstance(n, HtmlElement)])
        if not nodes:
            nodes = [document]

        summary_join = str(rule_set.get("summary_join_with") or rules.get("summary_join_with") or " ")
        content_join = str(rule_set.get("content_join_with") or rules.get("content_join_with") or "\n")

        for node in nodes:
            link = _extract_value(
                node,
                rule_set.get("link_xpath")
                or rule_set.get("url_xpath")
                or rules.get("link_xpath")
                or rules.get("url_xpath"),
                join=False,
            )
            if not link:
                continue
            link = link.strip()
            if not link or link in seen:
                continue
            seen.add(link)

            item: dict[str, Any] = {"url": link}
            title = _extract_value(
                node,
                rule_set.get("title_xpath") or rule_set.get("title_selector") or rules.get("title_xpath") or rules.get("title_selector"),
                join=False,
            )
            if title:
                item["title"] = title
            summary = _extract_value(
                node,
                rule_set.get("summary_xpath")
                or rule_set.get("description_xpath")
                or rule_set.get("summary_selector")
                or rules.get("summary_xpath")
                or rules.get("description_xpath")
                or rules.get("summary_selector"),
                join=True,
                join_with=summary_join,
            )
            if summary:
                item["summary"] = summary
            content = _extract_value(
                node,
                rule_set.get("content_xpath")
                or rule_set.get("content_selector")
                or rules.get("content_xpath")
                or rules.get("content_selector"),
                join=True,
                join_with=content_join,
            )
            if content:
                item["content"] = content
            author = _extract_value(
                node,
                rule_set.get("author_xpath") or rule_set.get("author_selector") or rules.get("author_xpath") or rules.get("author_selector"),
                join=False,
            )
            if author:
                item["author"] = author
            guid = _extract_value(
                node,
                rule_set.get("guid_xpath") or rule_set.get("id_xpath") or rules.get("guid_xpath") or rules.get("id_xpath"),
                join=False,
            )
            if guid:
                item["guid"] = guid
            published_raw = _extract_value(
                node,
                rule_set.get("published_xpath")
                or rule_set.get("date_xpath")
                or rule_set.get("date_selector")
                or rules.get("published_xpath")
                or rules.get("date_xpath")
                or rules.get("date_selector"),
                join=False,
            )
            if published_raw:
                item["published_raw"] = published_raw
                fmt = rule_set.get("published_format") or rules.get("published_format") or rule_set.get("date_format") or rules.get("date_format")
                parsed = _normalize_datetime(published_raw, fmt if isinstance(fmt, str) else None)
                if parsed:
                    item["published"] = parsed

            items.append(item)
            if limit is not None and len(items) >= limit:
                break
        return

    limit = _coerce_int(rules.get("limit") or rules.get("max_items"))
    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    _gather_items(rules, seen=seen_urls, items=items, limit=limit)

    alternates = rules.get("alternates")
    if isinstance(alternates, list):
        for alt in alternates:
            if not isinstance(alt, dict):
                continue
            if limit is not None and len(items) >= limit:
                break
            merged = {**rules, **alt}
            _gather_items(merged, seen=seen_urls, items=items, limit=limit)
            if limit is not None and len(items) >= limit:
                break

    pagination_cfg = rules.get("pagination") if isinstance(rules.get("pagination"), dict) else None
    next_pages: list[str] = []
    if pagination_cfg:
        candidate_selectors = _ensure_sequence(
            pagination_cfg.get("next_xpath")
            or pagination_cfg.get("next_selector")
            or pagination_cfg.get("next_link_xpath")
            or pagination_cfg.get("next_link_selector")
        )
        attr = pagination_cfg.get("next_attribute") or "href"
        for selector in candidate_selectors:
            matches = _select_nodes(document, selector)
            for match in matches:
                url = None
                if isinstance(match, HtmlElement):
                    url = match.get(attr)
                    if not url:
                        url = _coerce_value(match)
                else:
                    url = _coerce_value(match)
                if not url:
                    continue
                absolute = urljoin(base_url, url)
                if absolute not in next_pages:
                    next_pages.append(absolute)
    result["items"] = items
    result["next_pages"] = next_pages
    return result


async def fetch_rss_items(urls: list[str], *, limit: int = 10, tenant_id: str = "default") -> list[dict[str, Any]]:
    """Fetch RSS/Atom feed items for the given URLs.

    Uses the workflows adapter implementation for consistency with URL
    allowlisting and parsing heuristics. In TEST_MODE, returns a single fake
    item to keep tests offline.
    """
    urls = [u for u in (urls or []) if isinstance(u, str) and u.strip()]
    if not urls:
        return []

    # Offline mode for unit tests
    if _in_test_mode():
        return [{"title": "Test Item", "url": "https://example.com/x", "summary": "Test", "published": None}][:limit]

    try:
        from tldw_Server_API.app.core.Workflows.adapters import run_rss_fetch_adapter  # reuse existing parser
        cfg = {"urls": urls, "limit": limit, "include_content": True}
        ctx = {"tenant_id": tenant_id}
        res = await run_rss_fetch_adapter(cfg, ctx)
        items = []
        for r in (res.get("results") or []):
            items.append({
                "title": r.get("title") or "",
                "url": r.get("link") or r.get("url") or "",
                "summary": r.get("summary"),
                "published": r.get("published"),
                "guid": r.get("guid"),
            })
        return items[:limit]
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"fetch_rss_items failed: {e}")
        return []


def fetch_site_article(url: str) -> dict[str, Any] | None:
    """Fetch and extract a single site article.

    Uses the blocking path from Article_Extractor, which works without a
    Playwright runtime. Returns None on failure.
    """
    try:
        from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
            ContentMetadataHandler,  # type: ignore
            scrape_article_blocking,
        )
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Article extractor import failed: {e}")
        return None

    try:
        data = scrape_article_blocking(url)
        if not data:
            return None
        # Normalize
        title = data.get("title") or "Untitled"
        author = data.get("author") or None
        content = data.get("content") or ""
        try:
            content = ContentMetadataHandler.strip_metadata(content)  # type: ignore[attr-defined]
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            pass
        return {"title": title, "url": url, "content": content, "author": author}
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"fetch_site_article failed for {url}: {e}")
        return None


async def fetch_site_article_async(url: str) -> dict[str, Any] | None:
    """Async wrapper for blocking article extraction."""
    try:
        return await asyncio.to_thread(fetch_site_article, url)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"fetch_site_article_async failed for {url}: {exc}")
        return None


async def fetch_site_top_links(base_url: str, *, top_n: int = 10, method: str = "frontpage") -> list[str]:
    """Discover top-N content links from a site.

    - method="frontpage": fetch homepage and extract likely-article links
    - method="sitemap": try EnhancedWebScraper.scrape_sitemap to pull URLs only

    Returns a list of URLs (same-origin) deduplicated, up to top_n.
    TEST_MODE: returns [base_url] repeated to satisfy callers without network.
    """
    if top_n <= 0:
        return []

    if _in_test_mode():
        # Provide stable deterministic links
        return [base_url] * min(top_n, 3)

    # Try using EnhancedWebScraper when available
    try:
        from urllib.parse import urljoin, urlparse

        from bs4 import BeautifulSoup

        from tldw_Server_API.app.core.http_client import afetch
        from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import is_content_page
        from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper

        parsed = urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        headers = {
            "User-Agent": "tldw-watchlist/0.1 (+https://github.com/your-org/tldw_server2)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        async def _fetch_text(url: str, timeout: float) -> tuple[int, str]:
            resp = None
            try:
                resp = await afetch(method="GET", url=url, headers=headers, timeout=timeout)
                return int(resp.status_code), resp.text or ""
            finally:
                await _close_response(resp)

        # Auto-detect sitemap via robots.txt or common path when method='auto'
        async def _detect_sitemap(u: str) -> str | None:
            try:
                robots_url = urljoin(origin, "/robots.txt")
                status, txt = await _fetch_text(robots_url, timeout=6)
                if status // 100 == 2:
                    # Look for Sitemap lines
                    for line in txt.splitlines():
                        if line.lower().startswith("sitemap:"):
                            sitemap_url = line.split(":", 1)[1].strip()
                            return sitemap_url
                # Try common location
                common = urljoin(origin, "/sitemap.xml")
                status, _ = await _fetch_text(common, timeout=6)
                if status // 100 == 2:
                    return common
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                return None
            return None

        effective_method = method or "auto"
        sitemap_url_for_auto: str | None = None
        if effective_method == "auto":
            sitemap_url_for_auto = await _detect_sitemap(base_url)
            effective_method = "sitemap" if sitemap_url_for_auto else "frontpage"

        # Sitemap method
        if effective_method == "sitemap" or base_url.endswith(".xml") or "sitemap" in base_url:
            scraper = EnhancedWebScraper()
            # We only need URLs; call scrape_sitemap then pluck the url fields
            sitemap_to_use = sitemap_url_for_auto or base_url
            results = await scraper.scrape_sitemap(sitemap_to_use, filter_func=is_content_page, max_urls=top_n)
            urls = []
            for r in results:
                u = r.get("url") or r.get("source_url")
                if not u:
                    continue
                if urlparse(u).netloc == parsed.netloc:
                    urls.append(u)
            # Dedup while preserving order
            seen = set()
            uniq = []
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    uniq.append(u)
            return uniq[:top_n]

        # Frontpage method: fetch HTML and pull article-like links
        status, html = await _fetch_text(base_url, timeout=15)
        if status // 100 != 2:
            return [base_url]
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for a in soup.find_all("a"):
            href = a.get("href")
            if not href:
                continue
            href = urljoin(origin, href)
            # Same origin and looks like content
            try:
                if urlparse(href).netloc != parsed.netloc:
                    continue
                if not is_content_page(href):
                    continue
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                continue
            links.append(href)
        # Dedup preserve order
        seen = set()
        uniq = []
        for u in links:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq[:top_n] if uniq else [base_url]
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"fetch_site_top_links fallback: {e}")
        return [base_url]


async def fetch_site_items_with_rules(
    base_url: str,
    rules: dict[str, Any],
    *,
    tenant_id: str = "default",
    timeout: float = 10.0,
    fetch_diagnostics: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Fetch a list page and extract items using scrape rules."""
    list_url = str(rules.get("list_url") or base_url or "").strip()
    if not list_url:
        return []

    limit = _coerce_int(rules.get("limit") or rules.get("max_items"), default=10)
    if limit is not None and limit < 0:
        limit = 0
    if limit == 0:
        return []

    pagination_cfg = rules.get("pagination") if isinstance(rules.get("pagination"), dict) else {}
    max_pages = _coerce_int(pagination_cfg.get("max_pages"), default=1)
    if max_pages is None or max_pages < 1:
        max_pages = 1

    if _in_test_mode():
        max_items = limit if limit is not None else 3
        max_items = min(max_items, 5)
        samples: list[dict[str, Any]] = []
        for idx in range(max_items):
            url = f"{list_url.rstrip('/')}/test-scrape-{idx + 1}"
            samples.append(
                {
                    "title": f"Test scraped item {idx + 1}",
                    "url": url,
                    "summary": "Test summary from scrape rules.",
                    "content": "Test content from scrape rules.",
                }
            )
        return samples

    headers = {
        "User-Agent": "tldw-watchlist/0.1 (+https://github.com/your-org/tldw_server2)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
    }

    queue: list[str] = [list_url]
    visited: set[str] = set()
    seen_items: set[str] = set()
    collected: list[dict[str, Any]] = []

    def emit_fetch_diagnostic(event: dict[str, Any]) -> None:
        if fetch_diagnostics is None:
            return
        with contextlib.suppress(*_WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS):
            fetch_diagnostics(event)

    try:
        from tldw_Server_API.app.core.http_client import afetch
        while queue and len(visited) < max_pages:
            page_url = queue.pop(0)
            if page_url in visited:
                continue
            visited.add(page_url)

            allowed = False
            try:
                allowed = is_url_allowed_for_tenant(page_url, tenant_id)
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                allowed = is_url_allowed(page_url)
            if not allowed:
                logger.debug(f"Scrape rules blocked by URL policy: {page_url}")
                emit_fetch_diagnostic({"url": page_url, "error": "blocked_by_url_policy"})
                continue

            resp = None
            try:
                resp = await afetch(method="GET", url=page_url, headers=headers, timeout=timeout)
                status_code = int(resp.status_code)
                diagnostic_event: dict[str, Any] = {"url": page_url, "status": status_code}
                if status_code // 100 != 2 and status_code != 304:
                    diagnostic_event["error"] = f"HTTP {status_code}"
                emit_fetch_diagnostic(diagnostic_event)
                if status_code // 100 != 2:
                    logger.debug(f"fetch_site_items_with_rules HTTP {status_code} for {page_url}")
                    continue
                parsed = parse_scraped_items(resp.text or "", page_url, rules)
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"fetch_site_items_with_rules request failed ({page_url}): {exc}")
                emit_fetch_diagnostic({"url": page_url, "error": str(exc) or exc.__class__.__name__})
                continue
            finally:
                await _close_response(resp)

            page_items = parsed.get("items") or []
            for item in page_items:
                url = item.get("url")
                if not url or url in seen_items:
                    continue
                seen_items.add(url)
                collected.append(item)
                if limit is not None and len(collected) >= limit:
                    break
            if limit is not None and len(collected) >= limit:
                break

            next_pages = parsed.get("next_pages") or []
            for nxt in next_pages:
                if not nxt or nxt in visited or nxt in queue:
                    continue
                if len(visited) + len(queue) >= max_pages:
                    break
                queue.append(nxt)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"fetch_site_items_with_rules pagination failed: {exc}")

    if limit is not None:
        return collected[:limit]
    return collected


async def fetch_rss_feed(
    url: str,
    *,
    etag: str | None = None,
    last_modified: str | None = None,
    timeout: float = 8.0,
    tenant_id: str = "default",
) -> dict[str, Any]:
    """Fetch a single RSS/Atom feed with conditional headers.

    Returns dict:
      - status: int HTTP code (200/304/429/other)
      - items: list[dict] when 200
      - etag: str|None (from response headers)
      - last_modified: str|None (from response headers)
      - retry_after: int seconds (only when 429 and header present)
    """
    try:
        if not (url.startswith("http://") or url.startswith("https://")):
            return {"status": 400, "items": []}
        allowed = False
        try:
            allowed = is_url_allowed_for_tenant(url, tenant_id)
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            allowed = is_url_allowed(url)
        if not allowed:
            return {"status": 403, "items": []}

        headers = {
            "Accept": "application/atom+xml, application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": "tldw-watchlist/0.1 (+https://github.com/your-org/tldw_server2)"
        }
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        from tldw_Server_API.app.core.http_client import afetch
        resp = None
        status = 0
        resp_headers: dict[str, Any] = {}
        text = ""
        try:
            resp = await afetch(method="GET", url=url, headers=headers, timeout=timeout)
            if resp is not None:
                status = int(resp.status_code)
                resp_headers = dict(getattr(resp, "headers", {}) or {})
                text = resp.text or ""
        finally:
            await _close_response(resp)

        if resp is None:
            return {"status": 500, "items": []}
        if status == 0:
            return {"status": 500, "items": []}
        # Retry-After handling
        if status == 429:
            ra = resp_headers.get("Retry-After")
            retry_after_secs = None
            if ra:
                ra = ra.strip()
                # Seconds or HTTP-date
                try:
                    retry_after_secs = int(ra)
                except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                    from email.utils import parsedate_to_datetime
                    try:
                        dt = parsedate_to_datetime(ra)
                        import datetime as _dt
                        retry_after_secs = max(0, int((dt - _dt.datetime.utcnow().replace(tzinfo=dt.tzinfo)).total_seconds()))
                    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                        retry_after_secs = None
            return {"status": 429, "items": [], "retry_after": retry_after_secs}

        if status == 304:
            return {
                "status": 304,
                "items": [],
                "etag": resp_headers.get("ETag"),
                "last_modified": resp_headers.get("Last-Modified"),
            }

        if status // 100 != 2:
            return {"status": status, "items": []}

        try:
            root = ET.fromstring(text)
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            return {
                "status": status,
                "items": [],
                "etag": resp_headers.get("ETag"),
                "last_modified": resp_headers.get("Last-Modified"),
            }

        def _find_text(node, names):
            for n in names:
                x = node.find(n)
                if x is not None and (x.text or "").strip():
                    return x.text.strip()
            return None

        items_nodes = root.findall('.//item')
        if not items_nodes:
            items_nodes = root.findall('.//{http://www.w3.org/2005/Atom}entry')
        items: list[dict[str, Any]] = []
        atom_link_tag = "{http://www.w3.org/2005/Atom}link"
        atom_title_tag = "{http://www.w3.org/2005/Atom}title"
        atom_summary_tag = "{http://www.w3.org/2005/Atom}summary"
        atom_content_tag = "{http://www.w3.org/2005/Atom}content"
        atom_updated_tag = "{http://www.w3.org/2005/Atom}updated"
        atom_published_tag = "{http://www.w3.org/2005/Atom}published"
        atom_id_tag = "{http://www.w3.org/2005/Atom}id"
        for it in items_nodes:
            title = _find_text(it, ["title", atom_title_tag]) or ""

            link = ""
            link_nodes = list(it.findall("link")) + list(it.findall(atom_link_tag))
            preferred_link = ""
            fallback_link = ""
            for node in link_nodes:
                candidate = (node.get("href") or (node.text or "")).strip()
                if not candidate:
                    continue
                rel = (node.get("rel") or "").lower()
                if rel == "alternate" and not preferred_link:
                    preferred_link = candidate
                elif rel not in {"self"} and not fallback_link:
                    fallback_link = candidate
            link = preferred_link or fallback_link or _find_text(it, ["link", atom_link_tag]) or ""

            summary = _find_text(it, ["description", atom_summary_tag, atom_content_tag]) or ""
            published = _find_text(it, ["pubDate", atom_updated_tag, atom_published_tag]) or None
            guid = _find_text(it, ["guid", atom_id_tag]) or None
            rec = {"title": title, "url": link or "", "summary": summary, "published": published}
            if guid:
                rec["guid"] = guid
            items.append(rec)
        # Atom RFC5005: collect top-level archive paging links when present
        atom_links: list[dict[str, str]] = []
        try:
            # Look for link rel="prev-archive"/"next-archive" on the root <feed>
            for ln in list(root.findall(atom_link_tag)) + list(root.findall("link")):
                href = (ln.get("href") or (ln.text or "")).strip()
                if not href:
                    continue
                rel = (ln.get("rel") or "").strip().lower()
                # Resolve relative IRIs against the feed URL to produce an absolute URL
                try:
                    resolved = urljoin(url, href)
                except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                    resolved = href  # fall back to original if resolution fails
                if rel in {"prev-archive", "next-archive", "current", "self"}:
                    atom_links.append({"rel": rel, "href": resolved})
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            atom_links = []

        return {
            "status": 200,
            "items": items,
            "etag": resp_headers.get("ETag"),
            "last_modified": resp_headers.get("Last-Modified"),
            "atom_links": atom_links,
        }
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"fetch_rss_feed error: {e}")
        return {"status": 500, "items": []}


def discover_hub_url(
    xml_text: str,
    response_headers: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Return (hub_url, self_url) from feed XML and/or HTTP Link headers.

    Parses both:
    - HTTP ``Link`` header: ``<https://hub.example.com>; rel="hub"``
    - Atom/RSS XML: ``<link rel="hub" href="..." />`` and ``<link rel="self" href="..." />``
    """
    hub_url: str | None = None
    self_url: str | None = None

    # Phase 1: Check HTTP Link headers
    if response_headers:
        link_header = response_headers.get("Link") or response_headers.get("link") or ""
        for part in link_header.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                url_part, *attrs = part.split(";")
                url_val = url_part.strip().strip("<>").strip()
                rel_val = ""
                for attr in attrs:
                    attr = attr.strip()
                    if attr.lower().startswith("rel="):
                        rel_val = attr.split("=", 1)[1].strip().strip('"').strip("'").lower()
                if rel_val == "hub" and not hub_url:
                    hub_url = url_val
                elif rel_val == "self" and not self_url:
                    self_url = url_val
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                continue

    # Phase 2: Parse XML for <link rel="hub"> and <link rel="self">
    try:
        root = ET.fromstring(xml_text)
        atom_link_tag = "{http://www.w3.org/2005/Atom}link"
        # Search both direct children and descendants (RSS puts atom:link inside <channel>)
        for ln in list(root.findall(atom_link_tag)) + list(root.findall(".//" + atom_link_tag)) + list(root.findall("link")) + list(root.findall(".//link")):
            href = (ln.get("href") or (ln.text or "")).strip()
            if not href:
                continue
            rel = (ln.get("rel") or "").strip().lower()
            if rel == "hub" and not hub_url:
                hub_url = href
            elif rel == "self" and not self_url:
                self_url = href
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        pass

    return hub_url, self_url


async def fetch_rss_feed_history(
    url: str,
    *,
    etag: str | None = None,
    last_modified: str | None = None,
    timeout: float = 8.0,
    tenant_id: str = "default",
    strategy: str = "auto",
    max_pages: int = 1,
    per_page_limit: int | None = None,
    on_304: bool = False,
    stop_on_seen: bool = False,
    seen_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fetch a feed and optionally traverse history pages.

    - strategy: "auto" | "atom" | "wordpress" | "none"
    - max_pages: total pages to fetch including the first page
    - per_page_limit: trim items per page (None = keep all)
    - on_304: when True, still attempt history traversal when the first page returns 304
    """
    try:
        max_pages = int(max_pages)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        max_pages = 1
    if max_pages < 1:
        max_pages = 1

    # First page with conditional headers
    first = await fetch_rss_feed(
        url,
        etag=etag,
        last_modified=last_modified,
        timeout=timeout,
        tenant_id=tenant_id,
    )
    status = int(first.get("status", 0) or 0)
    agg_items: list[dict[str, Any]] = []
    etag_out = first.get("etag")
    last_mod_out = first.get("last_modified")
    pages_fetched = 0

    def _trim(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if per_page_limit is None or per_page_limit <= 0:
            return items
        return items[: per_page_limit]

    if status == 304 and not on_304:
        return {"status": 304, "items": [], "etag": etag_out, "last_modified": last_mod_out, "pages_fetched": 0, "strategy_used": strategy}

    if status == 429:
        # Pass through rate-limit signal
        out = {k: first.get(k) for k in ("status", "retry_after")}
        out.update({"items": [], "pages_fetched": 0})
        return out

    if status // 100 != 2:
        return {"status": status, "items": [], "pages_fetched": 0, "strategy_used": strategy}

    base_items = list(first.get("items") or [])
    agg_items.extend(_trim(base_items))
    pages_fetched += 1
    stop_triggered = False

    # Early exit
    if max_pages == 1:
        return {
            "status": 200,
            "items": agg_items,
            "etag": etag_out,
            "last_modified": last_mod_out,
            "pages_fetched": pages_fetched,
            "strategy_used": (strategy or "auto").lower(),
            "stop_on_seen_triggered": False,
        }

    # Helper: follow Atom RFC5005 prev-archive links
    async def _follow_atom_prev(href: str, remaining: int) -> int:
        nonlocal agg_items, pages_fetched
        current_url = href
        # Track aggregate keys and DB-seen keys separately
        agg_seen: set[str] = set()
        for it in agg_items:
            key = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
            if key:
                agg_seen.add(key)
        db_seen: set[str] = {k.strip() for k in (seen_keys or []) if isinstance(k, str)}
        fetched_here = 0
        while remaining > 0 and current_url:
            try:
                res = await fetch_rss_feed(current_url, timeout=timeout, tenant_id=tenant_id)
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                break
            if int(res.get("status", 0) or 0) // 100 != 2:
                break
            items = list(res.get("items") or [])
            # Dedup across pages and check DB-seen condition
            new: list[dict[str, Any]] = []
            for it in items:
                key = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
                if not key or key in agg_seen:
                    continue
                agg_seen.add(key)
                new.append(it)
            if not new:
                break
            if stop_on_seen:
                # If none of the items on this page are new relative to DB-seen keys, stop
                page_new_vs_db = [it for it in new if ((it.get("guid") or it.get("url") or it.get("link") or "").strip()) not in db_seen]
                if not page_new_vs_db:
                    nonlocal stop_triggered
                    stop_triggered = True
                    break
                # Update db_seen with truly-new keys so that further pages respect boundary condition
                for it in page_new_vs_db:
                    k = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
                    if k:
                        db_seen.add(k)
            agg_items.extend(_trim(new))
            pages_fetched += 1
            fetched_here += 1
            remaining -= 1
            # Look for next prev-archive
            next_links = [ln for ln in (res.get("atom_links") or []) if ln.get("rel") == "prev-archive" and ln.get("href")]
            current_url = next_links[0]["href"] if next_links else None
        return fetched_here

    # Helper: try common WordPress paged feed patterns
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    def _wp_paged_urls(base: str, pages: int) -> list[str]:
        out: list[str] = []
        try:
            parsed = urlparse(base)
            qs = parse_qs(parsed.query)
            # Case 1: ?feed=rss2 or ?feed=atom → add paged param
            if "feed" in qs:
                for p in range(2, pages + 1):
                    new_qs = qs.copy()
                    new_qs["paged"] = [str(p)]
                    new_url = urlunparse(parsed._replace(query=urlencode(new_qs, doseq=True)))
                    out.append(new_url)
            # Case 2: path endswith /feed or /feed/ → append ?paged=N
            path = parsed.path or ""
            if path.rstrip("/").endswith("/feed"):
                for p in range(2, pages + 1):
                    q = urlencode({"paged": p})
                    base2 = urlunparse(parsed._replace(query=q))
                    out.append(base2)
            # Fallback: append ?paged=N generally
            if not out:
                for p in range(2, pages + 1):
                    q = urlencode({"paged": p})
                    out.append(urlunparse(parsed._replace(query=q)))
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            pass
        # Dedup preserve order
        dedup: list[str] = []
        seen: set[str] = set()
        for u in out:
            if u not in seen:
                seen.add(u)
                dedup.append(u)
        return dedup

    pages_left = max(0, max_pages - 1)
    used_strategy = (strategy or "auto").lower()
    strategy_used = used_strategy

    # Try Atom prev-archive first when auto/atom
    if used_strategy in {"auto", "atom"} and pages_left > 0:
        prev_links = [ln for ln in (first.get("atom_links") or []) if ln.get("rel") == "prev-archive" and ln.get("href")]
        if prev_links:
            consumed = await _follow_atom_prev(prev_links[0]["href"], pages_left)
            pages_left -= consumed
            strategy_used = "atom"

    # Then try WordPress style when auto/wordpress and still have budget
    if used_strategy in {"auto", "wordpress"} and pages_left > 0 and not stop_triggered:
        wp_urls = _wp_paged_urls(url, pages_left + 1)
        # Track DB-seen and aggregate keys
        prior_keys = {(it.get("guid") or it.get("url") or it.get("link") or "").strip() for it in agg_items}
        db_seen_wp: set[str] = {k.strip() for k in (seen_keys or []) if isinstance(k, str)}
        for u in wp_urls:
            if pages_left <= 0:
                break
            try:
                res = await fetch_rss_feed(u, timeout=timeout, tenant_id=tenant_id)
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                continue
            if int(res.get("status", 0) or 0) // 100 != 2:
                continue
            items = list(res.get("items") or [])
            if not items:
                continue
            # Dedup vs aggregate and check DB-seen boundary condition
            new: list[dict[str, Any]] = []
            for it in items:
                key = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
                if not key or key in prior_keys:
                    continue
                prior_keys.add(key)
                new.append(it)
            if not new:
                continue
            if stop_on_seen:
                page_new_vs_db = [it for it in new if ((it.get("guid") or it.get("url") or it.get("link") or "").strip()) not in db_seen_wp]
                if not page_new_vs_db:
                    stop_triggered = True
                    break
                for it in page_new_vs_db:
                    k = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
                    if k:
                        db_seen_wp.add(k)
            agg_items.extend(_trim(new))
            pages_fetched += 1
            pages_left -= 1
        if pages_fetched > 1:
            strategy_used = "wordpress" if strategy_used == "auto" else strategy_used

    return {
        "status": 200,
        "items": agg_items,
        "etag": etag_out,
        "last_modified": last_mod_out,
        "pages_fetched": pages_fetched,
        "strategy_used": strategy_used,
        "stop_on_seen_triggered": stop_triggered,
    }
