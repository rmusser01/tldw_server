# Article_Extractor_Lib.py
#########################################
# Article Extraction Library
# This library is used to handle scraping and extraction of articles from web pages.
#
####################
# Function List
#
# 1. get_page_title(url)
# 2. get_article_text(url)
# 3. get_article_title(article_url_arg)
#
####################
#
# Import necessary libraries
#
# 3rd-Party Imports
import asyncio
import builtins
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Callable, Optional, Union
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import trafilatura

# External Libraries
from bs4 import BeautifulSoup
from defusedxml import ElementTree as xET
from defusedxml import minidom
from defusedxml.common import DefusedXmlException
from loguru import logger
from playwright.async_api import TimeoutError, async_playwright
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.DB_Management.DB_Manager import ingest_article_to_db
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.http_client import afetch
from tldw_Server_API.app.core.http_client import fetch as http_fetch

#
# Import Local
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.testing import is_truthy as _is_truthy
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.content import (
    ContentMetadataHandler,
    convert_html_to_markdown,
)
from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import RateLimiter
from tldw_Server_API.app.core.Web_Scraping.extraction import (
    clear_extraction_caches,
    extract_jsonld_entities,
    extract_regex_entities,
)
from tldw_Server_API.app.core.Web_Scraping.extraction import (
    get_extraction_cache_stats as _get_extraction_cache_stats,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.caches import (
    _cluster_cache_get as _canonical_cluster_cache_get,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.caches import (
    _cluster_cache_put as _canonical_cluster_cache_put,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.caches import (
    _schema_cache_get as _canonical_schema_cache_get,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.caches import (
    _schema_cache_put as _canonical_schema_cache_put,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.throttles import (
    apply_llm_delay as _canonical_apply_llm_delay,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.throttles import (
    get_llm_semaphore as _canonical_get_llm_semaphore,
)
from tldw_Server_API.app.core.Web_Scraping.extraction.throttles import (
    get_strategy_semaphore as _canonical_get_strategy_semaphore,
)
from tldw_Server_API.app.core.Web_Scraping.filters import (
    ContentTypeFilter,
    FilterChain,
    URLPatternFilter,
)
from tldw_Server_API.app.core.Web_Scraping.handlers import resolve_handler
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import decide_web_outbound_policy_sync
from tldw_Server_API.app.core.Web_Scraping.policy import DefaultWebOutboundPolicyChecker
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    DefaultFetchClient,
    FetchClient,
    FetchRequest,
    FetchResponse,
    OutboundPolicyChecker,
    PolicyDecision,
    RuntimeRequestContext,
)
from tldw_Server_API.app.core.Web_Scraping.safe_regex import (
    SafeRegexLimits,
    search_untrusted,
)
from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScraperRouter
from tldw_Server_API.app.core.Web_Scraping.selectors import (
    clear_selector_caches as _clear_selector_caches,
)
from tldw_Server_API.app.core.Web_Scraping.selectors import (
    extract_schema_fields,
    validate_selector_rules,
)
from tldw_Server_API.app.core.Web_Scraping.selectors import (
    get_selector_cache_stats as _get_selector_cache_stats,
)
from tldw_Server_API.app.core.Web_Scraping.ua_profiles import (
    build_browser_headers,
    pick_ua_profile,
)

_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
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
    builtins.TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    DefusedXmlException,
)

get_extraction_cache_stats = _get_extraction_cache_stats
clear_selector_caches = _clear_selector_caches
get_selector_cache_stats = _get_selector_cache_stats

#
#######################################################################################################################
# Function Definitions
#

# FIXME - Add a config file option/check for the user agent
web_scraping_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


def _default_rules_path() -> str:
    # Resolve project root: .../tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py -> project root
    here = Path(__file__).resolve()
    project_root = here.parents[4]  # tldw_Server_API/ at index 3; repo root at 4
    return str(project_root / "tldw_Server_API" / "Config_Files" / "custom_scrapers.yaml")


def _merge_cookie_list_to_map(custom_cookies: Optional[list[dict[str, Any]]]) -> dict[str, str]:
    cookies: dict[str, str] = {}
    if not custom_cookies:
        return cookies
    for c in custom_cookies:
        if isinstance(c, dict) and "name" in c and "value" in c:
            cookies[str(c["name"])] = str(c["value"])
    return cookies


def _robots_url_for(target_url: str) -> str:
    p = urlparse(target_url)
    return f"{p.scheme}://{p.netloc}/robots.txt"


_JS_REQUIRED_DOMAINS = {
    "medium.com",
    "substack.com",
    "notion.site",
    "notion.so",
    "webflow.io",
    "squarespace.com",
    "wixsite.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "instagram.com",
    "facebook.com",
    "linkedin.com",
}


def _js_required(html: str, headers: dict[str, Any], url: Optional[str] = None) -> bool:
    """Heuristics to detect pages that require JS rendering.

    Signals fallback to Playwright when:
    - Contains noscript prompts to enable JavaScript
    - Shows common interstitials (Cloudflare/CAPTCHA)
    - HTML very small with many scripts / SPA shells
    - Meta refresh redirect without content
    - Domain-specific hints when content is thin
    """
    try:
        domain = ""
        if url:
            try:
                domain = urlparse(url).netloc.lower()
            except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
                domain = ""
        text = html.lower()
        if not text.strip():
            return True
        # Common phrases
        js_phrases = (
            "enable javascript",
            "please enable javascript",
            "requires javascript",
            "enable your javascript",
            "javascript is disabled",
            "please turn on javascript",
            "please turn on js",
        )
        if any(p in text for p in js_phrases):
            return True
        if "<noscript" in text and any(
            p in text for p in ("enable javascript", "javascript is disabled", "requires javascript")
        ):
            return True
        # Cloudflare / anti-bot hints
        bot_phrases = (
            "cf-browser-verification",
            "cf-chl-bypass",
            "cloudflare ray id",
            "attention required",
            "checking your browser",
            "verify you are human",
            "hcaptcha",
            "recaptcha",
            "turnstile",
            "just a moment",
        )
        if any(p in text for p in bot_phrases):
            return True
        # Meta refresh
        if "http-equiv=\"refresh\"" in text or "http-equiv='refresh'" in text:
            # if no body text present
            if len(text) < 1500:
                return True
        soup = BeautifulSoup(html, "html.parser")
        visible_text = soup.get_text(" ", strip=True)
        visible_len = len(visible_text)
        script_count = len(soup.find_all("script"))
        if script_count >= 25 and visible_len < 800:
            return True
        if script_count >= 10 and visible_len < 400 and (
            "__next" in text or "__nuxt" in text or "data-reactroot" in text
        ):
            return True
        app_shell_ids = ("__next", "__nuxt", "root", "app", "app-root")
        if script_count >= 1 and visible_len < 600:
            for shell_id in app_shell_ids:
                if f'id="{shell_id}"' in text or f"id='{shell_id}'" in text:
                    return True
        if ("data-reactroot" in text or "data-reactid" in text) and visible_len < 600:
            return True
        # Domain hints for JS-heavy sites when content is minimal
        if domain and any(domain == d or domain.endswith("." + d) for d in _JS_REQUIRED_DOMAINS):
            if visible_len < 1200:
                return True
            if script_count >= 15 and visible_len < 2500:
                return True
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        return False
    return False


def _resp_get(resp: Any, key: str, default: Any = None) -> Any:
    """Best-effort fetch of a key from a response-like object.

    Supports mapping-like objects, dotted attributes, and objects exposing a
    'data' dict. Falls back to default if missing.
    """
    try:
        if isinstance(resp, dict):
            return resp.get(key, default)
        # Mapping-like via __getitem__
        try:
            return resp[key]  # type: ignore[index]
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
            pass
        # Direct attribute
        v = getattr(resp, key, None)
        if v is not None:
            return v
        # Nested 'data' mapping commonly used in tests/doubles
        data = getattr(resp, "data", None)
        if isinstance(data, dict):
            return data.get(key, default)
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        return default
    return default


_ARTICLE_POLICY_CHECKER: OutboundPolicyChecker = DefaultWebOutboundPolicyChecker()
_ARTICLE_FETCH_CLIENT: FetchClient = DefaultFetchClient()


def is_allowed_by_robots(url: str, user_agent: str, *, timeout: float = 5.0) -> bool:
    """Check robots.txt for allow/deny. Fails open (allow) if robots not reachable.

    Enforces egress policy via http_client.fetch().
    """
    try:
        robots_url = _robots_url_for(url)
        resp = http_fetch(method="GET", url=robots_url, timeout=timeout, allow_redirects=True)
        # Use robust getter to support dicts, objects, and test doubles
        status = _resp_get(resp, "status")
        if status is None:
            status = _resp_get(resp, "status_code")
        text = _resp_get(resp, "text", "")
        if (int(status or 0) >= 400) or (not text):
            return True  # treat missing/unreadable robots as allow
        rp = RobotFileParser()
        rp.parse(str(text).splitlines())
        return bool(rp.can_fetch(user_agent, url))
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        # On any error, allow by default to avoid false negatives
        return True


async def is_allowed_by_robots_async(url: str, user_agent: str, *, timeout: float = 5.0) -> bool:
    """Async robots.txt check using asyncio.to_thread for network fetch."""
    try:
        robots_url = _robots_url_for(url)
        # Use keyword args expected by http_fetch
        resp = await asyncio.to_thread(
            http_fetch,
            method="GET",
            url=robots_url,
            timeout=timeout,
            allow_redirects=True,
        )
        status = _resp_get(resp, "status")
        if status is None:
            status = _resp_get(resp, "status_code")
        text = _resp_get(resp, "text", "")
        if (int(status or 0) >= 400) or (not text):
            return True
        rp = RobotFileParser()
        rp.parse(str(text).splitlines())
        return bool(rp.can_fetch(user_agent, url))
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        return True


DEFAULT_BOILERPLATE_PATTERNS = [
    r"\bsubscribe\s+now\b",
    r"\bsubscribe\s+today\b",
    r"\bsign\s+up\b",
    r"\bshare\s+this\b",
    r"\bshare\s+on\s+(facebook|twitter|linkedin|reddit)\b",
    r"\bfollow\s+us\b",
    r"\bnewsletter\b",
    r"\bread\s+more\b",
    r"\bthanks\s+for\s+reading\b",
]

_BOILERPLATE_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in DEFAULT_BOILERPLATE_PATTERNS]


def _strip_boilerplate_sections(text: str) -> str:
    """Remove common boilerplate phrases from extracted article text."""
    if not text:
        return text

    lines = text.splitlines()

    def _is_boilerplate(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return any(regex.search(stripped) for regex in _BOILERPLATE_REGEXES)

    filtered_lines: list[str] = [line for line in lines if not _is_boilerplate(line)]

    # Collapse consecutive blank lines introduced by removals.
    collapsed: list[str] = []
    previous_blank = False
    for line in filtered_lines:
        if line.strip():
            collapsed.append(line)
            previous_blank = False
        else:
            if not previous_blank:
                collapsed.append(line)
            previous_blank = True

    return "\n".join(collapsed)

#################################################################
#
# Scraping-related functions:

def get_page_title(url: str) -> str:
    try:
        resp = http_fetch(method="GET", url=url, timeout=10)
        if resp.get("status", 0) == 200:
            soup = BeautifulSoup(resp.get("text", ""), 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.string.strip() if title_tag and title_tag.string else "Untitled"
            log_counter("page_title_extracted", labels={"success": "true"})
            return title
        else: #debug code for problem in suceeded request but non 200 code
            logging.error(f"Failed to fetch {url}, status code: {resp.get('status')}")
            return "Untitled"
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error fetching page title: {e}")
        log_counter("page_title_extracted", labels={"success": "false"})
        return "Untitled"


DEFAULT_EXTRACTION_STRATEGY_ORDER = [
    "jsonld",
    "schema",
    "regex",
    "llm",
    "cluster",
    "trafilatura",
]
_STRATEGY_ALIASES = {
    "json-ld": "jsonld",
    "json_ld": "jsonld",
    "microdata": "jsonld",
    "schema_css": "schema",
    "schema_xpath": "schema",
    "clustering": "cluster",
}
_KNOWN_STRATEGIES = set(DEFAULT_EXTRACTION_STRATEGY_ORDER)
_CLUSTER_EMBED_DIM = 128
_CLUSTER_PREFILTER_THRESHOLD = 0.2
_CLUSTER_SIM_THRESHOLD = 0.4
_CLUSTER_MIN_BLOCK_CHARS = 40
_CLUSTER_MIN_WORDS = 8
_CLUSTER_MAX_BLOCKS = 60
_CLUSTER_LINKAGE = "average"
_CLUSTER_TAG_TOP_K = 3
_DEFAULT_CLUSTER_TAG_KEYWORDS: dict[str, list[str]] = {
    "marketing": ["subscribe", "newsletter", "promotion", "marketing"],
    "commerce": ["price", "pricing", "cost", "$"],
    "product": ["feature", "release", "roadmap", "product"],
    "research": ["study", "research", "paper", "dataset"],
    "security": ["security", "encrypt", "token", "oauth"],
}
def _schema_cache_key(html_text: str, url: str, schema_rules: dict[str, Any]) -> str:
    html_hash = hashlib.sha1(
        html_text.encode("utf-8", errors="ignore"),
        usedforsecurity=False
    ).hexdigest()
    try:
        rules_repr = json.dumps(schema_rules, sort_keys=True, ensure_ascii=True)
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        rules_repr = str(schema_rules)
    raw = f"{url}|{rules_repr}|{html_hash}"
    return hashlib.sha1(
        raw.encode("utf-8", errors="ignore"),
        usedforsecurity=False
    ).hexdigest()


def _schema_cache_get(key: str) -> Optional[dict[str, Any]]:
    return _canonical_schema_cache_get(key)


def _schema_cache_put(key: str, value: dict[str, Any]) -> None:
    _canonical_schema_cache_put(key, value)


def _cluster_cache_get(key: str) -> Optional[list[float]]:
    def _record_cache_metric(*args: Any, **kwargs: Any) -> None:
        with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
            increment_counter(*args, **kwargs)

    return _canonical_cluster_cache_get(key, increment_counter=_record_cache_metric)


def _cluster_cache_put(key: str, value: list[float]) -> None:
    _canonical_cluster_cache_put(key, value)


def _normalize_strategy_order(
    strategy_order: Optional[list[str]],
) -> tuple[list[str], list[str]]:
    if strategy_order:
        raw = strategy_order
    else:
        return list(DEFAULT_EXTRACTION_STRATEGY_ORDER), []
    normalized: list[str] = []
    unknown: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if not key:
            continue
        key = _STRATEGY_ALIASES.get(key, key)
        if key in _KNOWN_STRATEGIES:
            if key not in normalized:
                normalized.append(key)
        else:
            unknown.append(key)
    if not normalized:
        normalized = list(DEFAULT_EXTRACTION_STRATEGY_ORDER)
    return normalized, unknown


def _trace_entry(strategy: str, status: str, reason: str, detail: Optional[str] = None) -> dict[str, Any]:
    with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
        log_counter("extraction_strategy_total", labels={"strategy": strategy, "status": status})
    entry = {"strategy": strategy, "status": status, "reason": reason}
    if detail:
        entry["detail"] = detail
    return entry


def _record_strategy_metrics(
    strategy: str,
    status: str,
    duration_s: float,
    result: Optional[dict[str, Any]] = None,
) -> None:
    with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
        observe_histogram(
            "extraction_strategy_duration_seconds",
            duration_s,
            labels={"strategy": strategy, "status": status},
        )
    if status != "success" or not result:
        return
    content = result.get("content")
    if isinstance(content, str) and content:
        with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
            observe_histogram(
                "extraction_content_length_bytes",
                len(content.encode("utf-8", errors="ignore")),
                labels={"strategy": strategy},
            )


def _attach_trace(
    result: dict[str, Any],
    trace: list[dict[str, Any]],
    strategy: Optional[str],
    strategy_order: list[str],
) -> dict[str, Any]:
    result["extraction_trace"] = trace
    result["extraction_strategy"] = strategy
    result["extraction_strategy_order"] = strategy_order
    return result


def _extract_with_trafilatura(html: str, url: str) -> dict[str, Any]:
    """Extract article metadata and body from raw HTML."""
    logging.info(f"Extracting article data from HTML for {url}")
    downloaded = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        include_images=False,
    )
    downloaded = _strip_boilerplate_sections(downloaded)
    metadata = trafilatura.extract_metadata(html)

    result: dict[str, Any] = {
        "title": "N/A",
        "author": "N/A",
        "content": "",
        "date": "N/A",
        "url": url,
        "extraction_successful": False,
    }

    if downloaded:
        logging.info(f"Content extracted successfully from {url}")
        log_counter("article_extracted", labels={"success": "true", "url": url})
        result["content"] = ContentMetadataHandler.format_content_with_metadata(
            url=url,
            content=downloaded,
            pipeline="Trafilatura",
            additional_metadata={
                "extracted_date": metadata.date if metadata and metadata.date else "N/A",
                "author": metadata.author if metadata and metadata.author else "N/A",
            },
        )
        result["extraction_successful"] = True
    else:
        log_counter("article_extracted", labels={"success": "false", "url": url})
        logging.warning("Content extraction failed.")

    if metadata:
        result.update(
            {
                "title": metadata.title if metadata.title else "N/A",
                "author": metadata.author if metadata.author else "N/A",
                "date": metadata.date if metadata.date else "N/A",
            }
        )
    else:
        logging.warning("Metadata extraction failed.")

    return result


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _regex_mask_override(settings: Optional[dict[str, Any]]) -> Optional[bool]:
    if not isinstance(settings, dict):
        return None
    for key in ("mask_pii", "pii_mask"):
        if key in settings:
            return bool(settings.get(key))
    return None


def _extractor_max_workers() -> Optional[int]:
    value = _env_int("EXTRACTOR_MAX_WORKERS")
    if value is None or value <= 0:
        return None
    return value


def _should_clear_caches(stage: str) -> bool:
    raw = os.getenv("EXTRACTOR_CLEAR_CACHES", "")
    if not raw:
        return False
    value = raw.strip().lower()
    if _is_truthy(value):
        return stage == "end"
    if value in {"both", "all"}:
        return True
    if value in {"start", "before", "entry"}:
        return stage == "start"
    if value in {"end", "after", "exit"}:
        return stage == "end"
    return False


def _get_strategy_semaphore(strategy: str) -> Optional[BoundedSemaphore]:
    return _canonical_get_strategy_semaphore(strategy, _extractor_max_workers())


@contextmanager
def _strategy_throttle(strategy: str) -> Any:
    sem = _get_strategy_semaphore(strategy)
    if sem is None:
        yield
        return
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def _tokenize_cluster_text(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower())


def _cluster_word_count(text: str) -> int:
    return len(_tokenize_cluster_text(text))


def _normalize_vector(vec: list[float]) -> list[float]:
    if not vec:
        return vec
    norm = math.sqrt(sum(val * val for val in vec))
    if norm <= 0.0:
        return vec
    return [val / norm for val in vec]


def _hash_embedding(text: str, dims: int) -> list[float]:
    tokens = _tokenize_cluster_text(text)
    if not tokens:
        return [0.0] * dims
    vec = [0.0] * dims
    for token in tokens:
        token_hash = hashlib.md5(
            token.encode("utf-8", errors="ignore"),
            usedforsecurity=False
        ).hexdigest()
        idx = int(token_hash, 16) % dims
        vec[idx] += 1.0
    return _normalize_vector(vec)


def _cluster_embedding(text: str, dims: int) -> list[float]:
    key = hashlib.sha1(
        text.encode("utf-8", errors="ignore"),
        usedforsecurity=False
    ).hexdigest()
    cached = _cluster_cache_get(key)
    if cached is not None:
        return cached
    vec = _hash_embedding(text, dims)
    _cluster_cache_put(key, vec)
    return vec


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return float(dot)


def _extract_cluster_blocks(
    html_text: str,
    *,
    min_block_chars: int,
    min_word_count: int,
    max_blocks: int,
) -> list[str]:
    if not html_text:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    blocks = [tag.get_text(" ", strip=True) for tag in soup.find_all(["p", "li"])]
    if not blocks:
        raw_text = soup.get_text("\n", strip=True)
        blocks = [line.strip() for line in raw_text.splitlines() if line.strip()]
    filtered = [
        block
        for block in blocks
        if len(block) >= min_block_chars and _cluster_word_count(block) >= min_word_count
    ]
    if not filtered and blocks:
        filtered = [max(blocks, key=len)]
    if len(filtered) > max_blocks:
        indexed = list(enumerate(filtered))
        top = sorted(indexed, key=lambda item: len(item[1]), reverse=True)[:max_blocks]
        keep_indexes = {idx for idx, _value in top}
        filtered = [block for idx, block in indexed if idx in keep_indexes]
    return filtered


def _extract_cluster_title(html_text: str) -> Optional[str]:
    if not html_text:
        return None
    soup = BeautifulSoup(html_text, "html.parser")
    title_tag = soup.find("title")
    if not title_tag:
        return None
    title = title_tag.get_text(strip=True)
    return title or None


def _cluster_assignments_hierarchical(
    vectors: list[list[float]],
    *,
    similarity_threshold: float,
    linkage: str,
) -> Optional[list[int]]:
    if not vectors:
        return None
    if len(vectors) == 1:
        return [0]
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        return None
    distance_threshold = max(0.0, 1.0 - similarity_threshold)
    size = len(vectors)
    distances = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            sim = _cosine_similarity(vectors[i], vectors[j])
            dist = max(0.0, 1.0 - sim)
            distances[i][j] = dist
            distances[j][i] = dist
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=linkage,
            distance_threshold=distance_threshold,
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage=linkage,
            distance_threshold=distance_threshold,
        )
    labels = clusterer.fit_predict(distances)
    return [int(label) for label in labels]


def _build_clusters_from_assignments(
    assignments: list[int],
    items: list[tuple[int, str, list[float], float]],
) -> list[dict[str, Any]]:
    clusters: dict[int, dict[str, Any]] = {}
    for label, item in zip(assignments, items):
        idx, block, vec, sim_to_doc = item
        cluster = clusters.get(label)
        if cluster is None:
            cluster = {
                "members": [],
                "sum_vec": [0.0 for _ in vec],
                "centroid": [0.0 for _ in vec],
                "total_chars": 0,
            }
            clusters[label] = cluster
        cluster["members"].append((idx, block, sim_to_doc))
        cluster["sum_vec"] = [a + b for a, b in zip(cluster["sum_vec"], vec)]
        cluster["total_chars"] += len(block)
    for cluster in clusters.values():
        cluster["centroid"] = _normalize_vector(cluster["sum_vec"])
    return list(clusters.values())


def _cluster_blocks_greedy(
    items: list[tuple[int, str, list[float], float]],
    *,
    cluster_threshold: float,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for idx, block, vec, sim_to_doc in items:
        best_idx = None
        best_sim = -1.0
        for c_idx, cluster in enumerate(clusters):
            sim = _cosine_similarity(vec, cluster["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_idx = c_idx
        if best_idx is None or best_sim < cluster_threshold:
            clusters.append(
                {
                    "members": [(idx, block, sim_to_doc)],
                    "sum_vec": list(vec),
                    "centroid": list(vec),
                    "total_chars": len(block),
                }
            )
            continue
        cluster = clusters[best_idx]
        cluster["members"].append((idx, block, sim_to_doc))
        cluster["sum_vec"] = [a + b for a, b in zip(cluster["sum_vec"], vec)]
        cluster["centroid"] = _normalize_vector(cluster["sum_vec"])
        cluster["total_chars"] += len(block)
    return clusters


def _tag_cluster_text(
    text: str,
    *,
    tag_keywords: dict[str, list[str]],
    top_k: int,
) -> tuple[list[str], dict[str, int]]:
    if top_k <= 0 or not text:
        return [], {}
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for tag, keywords in tag_keywords.items():
        if not keywords:
            continue
        score = 0
        for keyword in keywords:
            if not keyword:
                continue
            score += text_lower.count(str(keyword).lower())
        if score > 0:
            scores[tag] = score
    if not scores:
        return [], {}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    tags = [tag for tag, _score in ranked[:top_k]]
    return tags, scores


def extract_cluster_entities(
    html_text: str,
    url: str,
    *,
    cluster_settings: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "content": "",
        "date": "N/A",
        "extraction_successful": False,
        "cluster_blocks": [],
        "cluster_block_count": 0,
    }
    if not html_text:
        result["cluster_error"] = "cluster_empty_html"
        return result

    settings = dict(cluster_settings or {})
    env_similarity = _env_float("SIM_THRESHOLD")
    env_min_words = _env_int("WORD_COUNT_THRESHOLD")
    env_linkage = os.getenv("CLUSTER_LINKAGE", "").strip().lower()
    min_block_chars = int(settings.get("min_block_chars", _CLUSTER_MIN_BLOCK_CHARS))
    min_word_count = int(
        settings.get("min_word_count")
        or settings.get("min_words")
        or settings.get("word_count_threshold")
        or (env_min_words if env_min_words is not None else _CLUSTER_MIN_WORDS)
    )
    max_blocks = int(settings.get("max_blocks", _CLUSTER_MAX_BLOCKS))
    prefilter_threshold = float(settings.get("prefilter_threshold", _CLUSTER_PREFILTER_THRESHOLD))
    cluster_threshold = float(
        settings.get("cluster_threshold")
        or settings.get("similarity_threshold")
        or (env_similarity if env_similarity is not None else _CLUSTER_SIM_THRESHOLD)
    )
    embed_dims = int(settings.get("embed_dims", _CLUSTER_EMBED_DIM))
    method = str(settings.get("method") or settings.get("cluster_method") or "greedy").strip().lower()
    linkage = str(
        settings.get("linkage")
        or settings.get("cluster_linkage")
        or (env_linkage if env_linkage else _CLUSTER_LINKAGE)
    ).strip().lower()
    tag_top_k = int(settings.get("tag_top_k", _CLUSTER_TAG_TOP_K))
    tag_keywords = settings.get("tag_keywords") or _DEFAULT_CLUSTER_TAG_KEYWORDS
    if not isinstance(tag_keywords, dict):
        tag_keywords = _DEFAULT_CLUSTER_TAG_KEYWORDS

    with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
        increment_counter("extraction_cluster_total", labels={"status": "started"})

    blocks = _extract_cluster_blocks(
        html_text,
        min_block_chars=min_block_chars,
        min_word_count=min_word_count,
        max_blocks=max_blocks,
    )
    if not blocks:
        result["cluster_error"] = "cluster_no_blocks"
        with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
            increment_counter("extraction_cluster_total", labels={"status": "no_blocks"})
        return result

    doc_vec = _cluster_embedding(" ".join(blocks), embed_dims)
    scored_blocks: list[tuple[int, str, list[float], float]] = []
    for idx, block in enumerate(blocks):
        vec = _cluster_embedding(block, embed_dims)
        sim = _cosine_similarity(vec, doc_vec)
        scored_blocks.append((idx, block, vec, sim))

    kept = [item for item in scored_blocks if item[3] >= prefilter_threshold]
    if not kept:
        kept = sorted(scored_blocks, key=lambda item: item[3], reverse=True)[: min(2, len(scored_blocks))]

    clusters: list[dict[str, Any]] = []
    cluster_method = method
    if method == "hierarchical":
        assignments = _cluster_assignments_hierarchical(
            [item[2] for item in kept],
            similarity_threshold=cluster_threshold,
            linkage=linkage,
        )
        if assignments and len(assignments) == len(kept):
            clusters = _build_clusters_from_assignments(assignments, kept)
        else:
            cluster_method = "greedy_fallback"
            clusters = _cluster_blocks_greedy(kept, cluster_threshold=cluster_threshold)
    else:
        cluster_method = "greedy"
        clusters = _cluster_blocks_greedy(kept, cluster_threshold=cluster_threshold)

    if not clusters:
        result["cluster_error"] = "cluster_no_clusters"
        with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
            increment_counter("extraction_cluster_total", labels={"status": "no_clusters"})
        return result

    def _cluster_score(cluster: dict[str, Any]) -> tuple[int, int]:
        return (int(cluster.get("total_chars", 0)), len(cluster.get("members", [])))

    best_cluster = max(clusters, key=_cluster_score)
    ordered_members = sorted(best_cluster["members"], key=lambda item: item[0])
    content_blocks = [block for _idx, block, _sim in ordered_members if block]
    content = "\n\n".join(content_blocks).strip()

    if not content:
        result["cluster_error"] = "cluster_empty_content"
        with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
            increment_counter("extraction_cluster_total", labels={"status": "empty"})
        return result

    title = _extract_cluster_title(html_text)
    if title:
        result["title"] = title
    result["content"] = content
    result["cluster_blocks"] = content_blocks
    result["cluster_block_count"] = len(content_blocks)
    result["cluster_prefiltered_count"] = len(kept)
    result["cluster_total_blocks"] = len(blocks)
    result["cluster_cluster_count"] = len(clusters)
    result["cluster_method"] = cluster_method
    if method == "hierarchical":
        result["cluster_linkage"] = linkage
    result["cluster_similarity_threshold"] = cluster_threshold
    result["cluster_word_threshold"] = min_word_count
    tags, tag_scores = _tag_cluster_text(
        content,
        tag_keywords=tag_keywords,
        top_k=tag_top_k,
    )
    if tags:
        result["cluster_tags"] = tags
        result["cluster_tag_scores"] = tag_scores
    result["extraction_successful"] = True
    with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
        increment_counter("extraction_cluster_total", labels={"status": "success"})
    return result


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        return None
    return parsed if parsed > 0 else None


def _coerce_non_negative_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        return default
    return parsed if parsed >= 0.0 else default


def _resolve_llm_throttle_settings(settings: dict[str, Any]) -> tuple[Optional[int], float, float]:
    max_concurrency = _coerce_positive_int(
        settings.get("max_concurrency") if "max_concurrency" in settings else os.getenv("LLM_MAX_CONCURRENCY")
    )
    delay_ms = _coerce_non_negative_float(
        settings.get("delay_ms") if "delay_ms" in settings else os.getenv("LLM_DELAY_MS"),
        default=0.0,
    )
    jitter_ms = _coerce_non_negative_float(
        settings.get("delay_jitter_ms")
        if "delay_jitter_ms" in settings
        else settings.get("delay_jitter")
        if "delay_jitter" in settings
        else os.getenv("LLM_DELAY_JITTER_MS"),
        default=0.0,
    )
    return max_concurrency, delay_ms, jitter_ms


def _extractor_retry_settings() -> tuple[int, float, float]:
    max_retries = _coerce_positive_int(os.getenv("EXTRACTOR_MAX_RETRIES")) or 0
    base_delay_ms = _coerce_non_negative_float(os.getenv("EXTRACTOR_RETRY_BASE_MS"), default=0.0)
    jitter_ms = _coerce_non_negative_float(os.getenv("EXTRACTOR_RETRY_JITTER_MS"), default=0.0)
    return max_retries, base_delay_ms, jitter_ms


def _run_with_retries(
    func: Callable[[], dict[str, Any]],
    *,
    strategy: str,
) -> tuple[Optional[dict[str, Any]], Optional[Exception], int]:
    max_retries, base_delay_ms, jitter_ms = _extractor_retry_settings()
    attempts = 0
    while True:
        try:
            return func(), None, attempts
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
            if attempts >= max_retries:
                return None, exc, attempts
            delay_s = (base_delay_ms / 1000.0) * (2 ** attempts)
            if jitter_ms:
                delay_s += random.uniform(0.0, jitter_ms / 1000.0)  # nosec B311
            attempts += 1
            with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
                increment_counter(
                    "extraction_retry_total",
                    labels={"strategy": strategy, "attempt": str(attempts)},
                )
            if delay_s > 0.0:
                time.sleep(delay_s)


def _get_llm_semaphore(provider: str, max_concurrency: int) -> Any:
    return _canonical_get_llm_semaphore(provider, max_concurrency)


def _apply_llm_delay(provider: str, delay_ms: float, jitter_ms: float) -> None:
    _canonical_apply_llm_delay(
        provider,
        delay_ms,
        jitter_ms,
        wall_time=time.time,
        sleep=time.sleep,
    )


@contextmanager
def _llm_throttle(provider: str, settings: dict[str, Any]):
    max_concurrency, delay_ms, jitter_ms = _resolve_llm_throttle_settings(settings)
    semaphore = _get_llm_semaphore(provider, max_concurrency) if max_concurrency else None
    if semaphore is not None:
        semaphore.acquire()
    try:
        _apply_llm_delay(provider, delay_ms, jitter_ms)
        yield
    finally:
        if semaphore is not None:
            semaphore.release()


def _extract_text_for_llm(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def _split_llm_chunks(
    text: str,
    *,
    chunk_token_threshold: int,
    overlap_rate: float,
    word_token_rate: float,
) -> list[str]:
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    rate = max(0.1, float(word_token_rate))
    token_est = len(words) * rate
    if token_est <= max(1, int(chunk_token_threshold)):
        return [" ".join(words)]
    chunk_words = max(50, int(chunk_token_threshold / rate))
    overlap = max(0, min(int(chunk_words * max(0.0, min(overlap_rate, 0.9))), chunk_words - 1))
    step = max(1, chunk_words - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def _extract_llm_response_text(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(choice.get("text"), str):
                    return choice["text"]
        if isinstance(resp.get("content"), str):
            return resp["content"]
    return ""


def _extract_usage_from_response(resp: Any) -> dict[str, int]:
    if not isinstance(resp, dict):
        return {}
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = usage.get(key)
        if isinstance(val, (int, float)):
            out[key] = int(val)
    return out


def _record_llm_usage_metrics(usage: dict[str, int], *, provider: str, model: str) -> None:
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if pt:
        increment_counter(
            "llm_tokens_used_total",
            float(pt),
            labels={"provider": provider, "model": model, "type": "prompt"},
        )
        increment_counter(
            "llm_tokens_used_total_by_operation",
            float(pt),
            labels={"provider": provider, "model": model, "type": "prompt", "operation": "extraction"},
        )
    if ct:
        increment_counter(
            "llm_tokens_used_total",
            float(ct),
            labels={"provider": provider, "model": model, "type": "completion"},
        )
        increment_counter(
            "llm_tokens_used_total_by_operation",
            float(ct),
            labels={"provider": provider, "model": model, "type": "completion", "operation": "extraction"},
        )


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*", "", stripped).strip()
        if stripped.endswith("```"):
            stripped = stripped[: -3].strip()
    return stripped


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    if not text:
        return candidates
    for match in re.finditer(r"```(?:json)?\\s*(.*?)```", text, re.IGNORECASE | re.DOTALL):
        payload = match.group(1).strip()
        if payload:
            candidates.append(payload)
    for match in re.finditer(r"<json>(.*?)</json>", text, re.IGNORECASE | re.DOTALL):
        payload = match.group(1).strip()
        if payload:
            candidates.append(payload)
    candidates.append(_strip_code_fences(text))
    return candidates


def _decode_all_json(payload: str) -> list[Any]:
    decoder = json.JSONDecoder()
    idx = 0
    length = len(payload)
    objects: list[Any] = []
    while idx < length:
        brace = payload.find("{", idx)
        bracket = payload.find("[", idx)
        if brace == -1 and bracket == -1:
            break
        start = bracket if brace == -1 or bracket != -1 and bracket < brace else brace
        try:
            obj, end = decoder.raw_decode(payload, start)
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
            idx = start + 1
            continue
        objects.append(obj)
        idx = end
    return objects


def _parse_llm_json(text: str, *, strict: bool) -> tuple[Optional[Any], dict[str, Any]]:
    meta: dict[str, Any] = {"objects": []}
    if not text:
        return None, meta
    payload = text.strip()
    if strict:
        try:
            obj = json.loads(payload)
            meta["objects"] = [obj]
            return obj, meta
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
            meta["error"] = f"strict_json_failed: {exc}"
            return None, meta
    candidates = _extract_json_candidates(payload)
    for candidate in candidates:
        objs = _decode_all_json(candidate)
        if not objs:
            try:
                obj = json.loads(candidate)
            except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
                continue
            objs = [obj]
        if objs:
            meta["objects"].extend(objs)
            return objs[0], meta
    return None, meta


def _resolve_llm_provider(settings: dict[str, Any]) -> tuple[str, Optional[dict[str, Any]]]:
    provider = str(settings.get("provider") or "").strip().lower()
    app_config = None
    if not provider:
        try:
            from tldw_Server_API.app.core.LLM_Calls.adapter_utils import ensure_app_config

            app_config = ensure_app_config(None)
            provider = str(app_config.get("RAG_DEFAULT_LLM_PROVIDER") or "").strip().lower()
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
            provider = ""
    if app_config is None:
        try:
            from tldw_Server_API.app.core.LLM_Calls.adapter_utils import ensure_app_config

            app_config = ensure_app_config(None)
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
            app_config = None
    return provider, app_config


def _parse_regex_flags(flags_spec: Any) -> int:
    if isinstance(flags_spec, int):
        return flags_spec
    flags = 0
    if isinstance(flags_spec, list):
        flags_spec = "".join(str(item) for item in flags_spec)
    if isinstance(flags_spec, str):
        mapping = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}
        for char in flags_spec:
            flag = mapping.get(char.lower())
            if flag:
                flags |= flag
    return flags


def _llm_prompt_for_schema_generation(
    html_text: str,
    *,
    url: str,
    query: Optional[str],
    example_json: Optional[str],
) -> str:
    snippet = html_text.strip()
    if len(snippet) > 8000:
        snippet = f"{snippet[:8000]}\n...[truncated]"
    parts = [
        "Generate a schema DSL for extracting structured data from this HTML.",
        "Return JSON with key `schema` containing fields: name, baseSelector, baseFields, fields.",
        "Selectors should use XPath or prefix CSS with `css:`. Use `type` and `selector` per field.",
        f"URL: {url}",
    ]
    if query:
        parts.append(f"User query: {query}")
    if example_json:
        parts.append(f"Example JSON output: {example_json}")
    parts.append(f"HTML:\n{snippet}")
    return "\n".join(parts)


def _llm_prompt_for_regex_generation(
    html_text: str,
    *,
    url: str,
    label: Optional[str],
    query: Optional[str],
    examples: Optional[list[str]],
) -> str:
    snippet = html_text.strip()
    if len(snippet) > 8000:
        snippet = f"{snippet[:8000]}\n...[truncated]"
    parts = [
        "Generate a regex pattern to extract the requested value from this HTML/text.",
        "Return JSON with keys: pattern (no delimiters), flags (e.g. 'i'), group (optional).",
        f"URL: {url}",
    ]
    if label:
        parts.append(f"Label: {label}")
    if query:
        parts.append(f"Query: {query}")
    if examples:
        parts.append(f"Examples: {examples}")
    parts.append(f"HTML:\n{snippet}")
    return "\n".join(parts)


def generate_schema_rules_from_llm(
    html_text: str,
    url: str,
    *,
    llm_settings: Optional[dict[str, Any]] = None,
    query: Optional[str] = None,
    example_json: Optional[str] = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"success": False}
    if not html_text:
        result["error"] = "schema_llm_empty_html"
        return result

    settings = dict(llm_settings or {})
    provider, app_config = _resolve_llm_provider(settings)
    if not provider:
        result["error"] = "schema_llm_provider_missing"
        return result

    system_message = settings.get("system_message")
    model = settings.get("model")
    api_key = settings.get("api_key")
    temperature = settings.get("temperature")
    max_tokens = settings.get("max_tokens")
    response_format = settings.get("response_format")
    strict_json = bool(settings.get("strict_json") or False)
    if strict_json and response_format is None:
        response_format = {"type": "json_object"}

    prompt = _llm_prompt_for_schema_generation(
        html_text,
        url=url,
        query=query,
        example_json=example_json,
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call

        with _llm_throttle(provider, settings):
            resp = perform_chat_api_call(
                api_provider=provider,
                messages=messages,
                system_message=system_message,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                app_config=app_config,
            )
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
        result["error"] = f"schema_llm_call_failed: {exc}"
        return result

    usage = _extract_usage_from_response(resp)
    model_name = str(resp.get("model") if isinstance(resp, dict) else model or "unknown")
    _record_llm_usage_metrics(usage, provider=provider, model=model_name)
    result["llm_usage"] = usage
    result["llm_provider"] = provider
    result["llm_model"] = model_name

    raw_text = _extract_llm_response_text(resp)
    obj, meta = _parse_llm_json(raw_text, strict=strict_json)
    if obj is None:
        result["error"] = meta.get("error") or "schema_llm_parse_failed"
        return result

    schema_obj: Optional[dict[str, Any]] = None
    if isinstance(obj, dict):
        if isinstance(obj.get("schema"), dict):
            schema_obj = obj.get("schema")
        elif "fields" in obj or "baseFields" in obj:
            schema_obj = obj
    if not isinstance(schema_obj, dict):
        result["error"] = "schema_llm_no_schema"
        return result

    try:
        validation = validate_selector_rules(schema_obj, html_text=html_text)
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
        validation = {"errors": [{"key": "validation", "error": str(exc)}], "warnings": []}

    result["schema_rules"] = schema_obj
    result["schema_validation"] = validation
    result["success"] = not bool(validation.get("errors"))
    return result


def generate_regex_pattern_from_llm(
    html_text: str,
    url: str,
    *,
    llm_settings: Optional[dict[str, Any]] = None,
    label: Optional[str] = None,
    query: Optional[str] = None,
    examples: Optional[list[str]] = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"success": False}
    if not html_text:
        result["error"] = "regex_llm_empty_html"
        return result

    settings = dict(llm_settings or {})
    provider, app_config = _resolve_llm_provider(settings)
    if not provider:
        result["error"] = "regex_llm_provider_missing"
        return result

    system_message = settings.get("system_message")
    model = settings.get("model")
    api_key = settings.get("api_key")
    temperature = settings.get("temperature")
    max_tokens = settings.get("max_tokens")
    response_format = settings.get("response_format")
    strict_json = bool(settings.get("strict_json") or False)
    if strict_json and response_format is None:
        response_format = {"type": "json_object"}

    prompt = _llm_prompt_for_regex_generation(
        html_text,
        url=url,
        label=label,
        query=query,
        examples=examples,
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call

        with _llm_throttle(provider, settings):
            resp = perform_chat_api_call(
                api_provider=provider,
                messages=messages,
                system_message=system_message,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                app_config=app_config,
            )
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
        result["error"] = f"regex_llm_call_failed: {exc}"
        return result

    usage = _extract_usage_from_response(resp)
    model_name = str(resp.get("model") if isinstance(resp, dict) else model or "unknown")
    _record_llm_usage_metrics(usage, provider=provider, model=model_name)
    result["llm_usage"] = usage
    result["llm_provider"] = provider
    result["llm_model"] = model_name

    raw_text = _extract_llm_response_text(resp)
    obj, meta = _parse_llm_json(raw_text, strict=strict_json)
    if obj is None or not isinstance(obj, dict):
        result["error"] = meta.get("error") or "regex_llm_parse_failed"
        return result

    pattern = obj.get("pattern") or obj.get("regex")
    if not isinstance(pattern, str) or not pattern.strip():
        result["error"] = "regex_llm_no_pattern"
        return result
    pattern = pattern.strip()
    flags = _parse_regex_flags(obj.get("flags"))
    if obj.get("ignore_case") is True:
        flags |= re.IGNORECASE
    group = obj.get("group")
    group_idx = int(group) if isinstance(group, int) else None

    sample_too_large = len(html_text) > 1_000_000
    safe_result = search_untrusted(
        pattern,
        "" if sample_too_large else html_text,
        flags=flags,
        limits=SafeRegexLimits(
            max_pattern_chars=4_096,
            max_input_chars=1_000_000,
            timeout_s=0.100,
        ),
    )
    if safe_result.code is not None:
        result["error"] = safe_result.code
        return result

    if sample_too_large:
        result["sample_status"] = "skipped_input_too_large"
    else:
        match = safe_result.match
        if match:
            try:
                matched_value = match.group(group_idx) if group_idx is not None else match.group(0)
            except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
                matched_value = match.group(0)
            result["sample_match"] = matched_value
            result["sample_span"] = [match.start(), match.end()]

    result["pattern"] = pattern
    if obj.get("flags") is not None:
        result["flags"] = obj.get("flags")
    if group_idx is not None:
        result["group"] = group_idx
    result["success"] = True
    return result


def _schema_rules_to_field_specs(schema_rules: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(schema_rules, dict):
        return []
    fields: list[dict[str, Any]] = []
    if isinstance(schema_rules.get("fields"), list) or isinstance(schema_rules.get("baseFields"), (list, dict)):
        def _normalize_field_definitions(raw: Any) -> list[dict[str, Any]]:
            if isinstance(raw, list):
                return [f for f in raw if isinstance(f, dict)]
            if isinstance(raw, dict):
                normalized: list[dict[str, Any]] = []
                for name, spec in raw.items():
                    entry = dict(spec) if isinstance(spec, dict) else {"selector": spec}
                    entry.setdefault("name", str(name))
                    normalized.append(entry)
                return normalized
            return []

        for group in ("baseFields", "fields"):
            for field in _normalize_field_definitions(schema_rules.get(group) or []):
                name = field.get("name")
                if isinstance(name, str) and name.strip():
                    fields.append(
                        {
                            "name": name.strip(),
                            "type": str(field.get("type") or "text").strip().lower(),
                        }
                    )
        return fields
    selector_fields = {
        "title": ("title_xpath", "title_selector"),
        "summary": ("summary_xpath", "summary_selector", "description_xpath"),
        "content": ("content_xpath", "content_selector"),
        "author": ("author_xpath", "author_selector"),
        "published": ("published_xpath", "date_xpath", "date_selector"),
    }
    for name, keys in selector_fields.items():
        if any(schema_rules.get(key) for key in keys):
            fields.append({"name": name, "type": "text"})
    return fields


def _schema_rule_keys(schema_rules: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(schema_rules, dict):
        return []
    keys: list[str] = []
    if any(schema_rules.get(key) for key in ("baseSelector", "base_selector", "baseXpath", "base_xpath")):
        keys.append("baseSelector")
    fields = _schema_rules_to_field_specs(schema_rules)
    if fields:
        keys.extend([f["name"] for f in fields if isinstance(f.get("name"), str)])
    else:
        selector_keys = (
            "title_xpath",
            "title_selector",
            "summary_xpath",
            "summary_selector",
            "description_xpath",
            "content_xpath",
            "content_selector",
            "author_xpath",
            "author_selector",
            "published_xpath",
            "date_xpath",
            "date_selector",
        )
        for key in selector_keys:
            if schema_rules.get(key):
                keys.append(key)
    unique = sorted({key for key in keys if key})
    return unique


def _llm_prompt_for_mode(
    *,
    mode: str,
    chunk: str,
    url: str,
    fields: list[dict[str, Any]],
    chunk_index: int,
    chunk_count: int,
    extra_instructions: Optional[str],
) -> str:
    header = (
        "Extract structured information from the following webpage text."
        " Return only JSON with nulls for unknown fields."
    )
    chunk_note = f"Chunk {chunk_index + 1} of {chunk_count}."
    field_spec = json.dumps(fields, ensure_ascii=True)
    if mode == "schema":
        prompt = (
            f"{header}\nURL: {url}\n{chunk_note}\n"
            f"Schema fields (name/type): {field_spec}\n"
            "Return a JSON object with those fields at the top level."
        )
    elif mode == "infer_schema":
        prompt = (
            f"{header}\nURL: {url}\n{chunk_note}\n"
            "Infer a compact schema for the content and return:\n"
            "{\"schema\": {\"fields\": [...]}, \"data\": {...}}"
        )
    else:
        prompt = (
            f"{header}\nURL: {url}\n{chunk_note}\n"
            "Return a JSON object with keys: title, author, date, content, blocks.\n"
            "Blocks should be a list of {type, text}."
        )
    if extra_instructions:
        prompt = f"{prompt}\nAdditional instructions: {extra_instructions}"
    return f"{prompt}\n\nContent:\n{chunk}"


def _merge_llm_data(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in incoming.items():
        if key not in base or base[key] in (None, "", [], {}):
            base[key] = value
            continue
        if isinstance(base[key], list) and isinstance(value, list):
            base[key].extend(value)
    return base


def _merge_llm_results(objs: list[dict[str, Any]], mode: str) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    merged: dict[str, Any] = {}
    schema: Optional[dict[str, Any]] = None
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        if mode == "infer_schema":
            if isinstance(obj.get("schema"), dict):
                schema = obj.get("schema")
            data = obj.get("data") if isinstance(obj.get("data"), dict) else obj
            _merge_llm_data(merged, data)
        else:
            _merge_llm_data(merged, obj)
    return merged, schema


def _llm_has_content(data: dict[str, Any]) -> bool:
    for _key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and value:
            return True
        if isinstance(value, dict) and value:
            return True
    return False


def extract_llm_entities(
    html_text: str,
    url: str,
    *,
    llm_settings: Optional[dict[str, Any]] = None,
    schema_rules: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "content": "",
        "date": "N/A",
        "extraction_successful": False,
        "llm_mode": None,
    }
    if not html_text:
        return result

    settings = dict(llm_settings or {})
    provider = str(settings.get("provider") or "").strip().lower()
    app_config = None
    if not provider:
        try:
            from tldw_Server_API.app.core.LLM_Calls.adapter_utils import ensure_app_config

            app_config = ensure_app_config(None)
            provider = str(app_config.get("RAG_DEFAULT_LLM_PROVIDER") or "").strip().lower()
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
            provider = ""
    if not provider:
        result["llm_error"] = "llm_provider_missing"
        return result

    if app_config is None:
        try:
            from tldw_Server_API.app.core.LLM_Calls.adapter_utils import ensure_app_config

            app_config = ensure_app_config(None)
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
            app_config = None

    text = _extract_text_for_llm(html_text)
    if not text:
        result["llm_error"] = "llm_empty_text"
        return result

    mode = str(settings.get("mode") or "").strip().lower()
    if not mode:
        mode = "schema" if schema_rules else "blocks"
    if mode not in {"blocks", "schema", "infer_schema"}:
        mode = "blocks"

    chunk_token_threshold = int(settings.get("chunk_token_threshold") or 1200)
    overlap_rate = float(settings.get("overlap_rate") or 0.1)
    word_token_rate = float(settings.get("word_token_rate") or 1.3)
    strict_json = bool(settings.get("strict_json") or False)
    chunks = _split_llm_chunks(
        text,
        chunk_token_threshold=chunk_token_threshold,
        overlap_rate=overlap_rate,
        word_token_rate=word_token_rate,
    )
    if not chunks:
        result["llm_error"] = "llm_no_chunks"
        return result

    fields = _schema_rules_to_field_specs(schema_rules)
    extra_prompt = settings.get("prompt")
    system_message = settings.get("system_message")
    model = settings.get("model")
    api_key = settings.get("api_key")
    temperature = settings.get("temperature")
    max_tokens = settings.get("max_tokens")
    response_format = settings.get("response_format")
    if strict_json and response_format is None:
        response_format = {"type": "json_object"}

    parsed_objects: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    llm_errors: list[str] = []
    for idx, chunk in enumerate(chunks):
        prompt = _llm_prompt_for_mode(
            mode=mode,
            chunk=chunk,
            url=url,
            fields=fields,
            chunk_index=idx,
            chunk_count=len(chunks),
            extra_instructions=str(extra_prompt) if extra_prompt else None,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call

            with _llm_throttle(provider, settings):
                resp = perform_chat_api_call(
                    api_provider=provider,
                    messages=messages,
                    system_message=system_message,
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    app_config=app_config,
                )
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
            llm_errors.append(f"llm_call_failed: {exc}")
            continue

        usage = _extract_usage_from_response(resp)
        model_name = str(resp.get("model") if isinstance(resp, dict) else model or "unknown")
        _record_llm_usage_metrics(usage, provider=provider, model=model_name)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_total[key] = usage_total.get(key, 0) + int(usage.get(key, 0))

        raw_text = _extract_llm_response_text(resp)
        obj, meta = _parse_llm_json(raw_text, strict=strict_json)
        if obj is None:
            detail = meta.get("error") or "no_json"
            llm_errors.append(f"llm_parse_failed: {detail}")
            continue
        if isinstance(obj, dict):
            parsed_objects.append(obj)

    if not parsed_objects:
        result["llm_error"] = "; ".join(llm_errors) if llm_errors else "llm_no_parseable_output"
        result["llm_mode"] = mode
        return result

    merged, inferred_schema = _merge_llm_results(parsed_objects, mode)
    result["llm_extraction"] = merged
    result["llm_schema"] = inferred_schema
    result["llm_provider"] = provider
    result["llm_mode"] = mode
    result["llm_usage"] = usage_total

    for key in ("title", "author", "date", "summary", "content"):
        if key in merged and merged[key] is not None:
            result[key] = merged[key]
    if not result.get("content") and isinstance(merged.get("blocks"), list):
        blocks = merged.get("blocks") or []
        parts = []
        for block in blocks:
            if isinstance(block, dict):
                text_val = block.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())
        if parts:
            result["content"] = "\n\n".join(parts)

    result["extraction_successful"] = _llm_has_content(merged)
    return result


def extract_article_with_pipeline(
    html: str,
    url: str,
    *,
    strategy_order: Optional[list[str]] = None,
    handler: Optional[Callable[[str, str], dict[str, Any]]] = None,
    fallback_extractor: Optional[Callable[[str, str], dict[str, Any]]] = None,
    schema_rules: Optional[dict[str, Any]] = None,
    llm_settings: Optional[dict[str, Any]] = None,
    regex_settings: Optional[dict[str, Any]] = None,
    cluster_settings: Optional[dict[str, Any]] = None,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    jsonld_summary: Optional[str] = None
    order, unknown = _normalize_strategy_order(strategy_order)
    if not allow_llm_extraction:
        order = [strategy for strategy in order if strategy != "llm"]
    if _should_clear_caches("start"):
        clear_extraction_caches()

    def _finalize_result(
        result: dict[str, Any],
        *,
        strategy: Optional[str],
    ) -> dict[str, Any]:
        summary = result.get("summary")
        if (
            result.get("extraction_successful")
            and jsonld_summary
            and (not isinstance(summary, str) or not summary.strip())
        ):
            result["summary"] = jsonld_summary
        final = _attach_trace(result, trace, strategy, order)
        if _should_clear_caches("end"):
            clear_extraction_caches()
        return final

    for strategy in unknown:
        trace.append(_trace_entry(strategy, "skipped", "unknown_strategy"))

    last_result: Optional[dict[str, Any]] = None
    for strategy in order:
        start = time.perf_counter()
        if strategy == "jsonld":
            with _strategy_throttle(strategy):
                result = extract_jsonld_entities(html, url)
            summary = result.get("summary")
            if isinstance(summary, str) and summary.strip():
                jsonld_summary = summary
            last_result = result
            if result.get("extraction_successful"):
                trace.append(_trace_entry(strategy, "success", "jsonld_extracted"))
                _record_strategy_metrics(strategy, "success", time.perf_counter() - start, result)
                return _finalize_result(result, strategy=strategy)
            detail = result.get("jsonld_error")
            trace.append(_trace_entry(strategy, "failed", "jsonld_no_content", str(detail) if detail else None))
            _record_strategy_metrics(strategy, "failed", time.perf_counter() - start, result)
            continue
        if strategy == "schema":
            if isinstance(schema_rules, dict) and schema_rules:
                cache_key = _schema_cache_key(html, url, schema_rules)
                cached = _schema_cache_get(cache_key)
                if cached and cached.get("extraction_successful"):
                    cached["schema_cache_hit"] = True
                    trace.append(_trace_entry(strategy, "success", "schema_cached"))
                    _record_strategy_metrics(strategy, "success", time.perf_counter() - start, cached)
                    return _finalize_result(cached, strategy=strategy)
                try:
                    validation = validate_selector_rules(
                        schema_rules,
                        html_text=html,
                        include_counts=True,
                    )
                except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as validation_exc:
                    reason = "schema_import_error" if isinstance(validation_exc, ImportError) else "schema_error"
                    trace.append(_trace_entry(strategy, "failed", reason))
                    _record_strategy_metrics(strategy, "failed", time.perf_counter() - start)
                    continue
                if validation is None:
                    trace.append(_trace_entry(strategy, "failed", "schema_error"))
                    _record_strategy_metrics(strategy, "failed", time.perf_counter() - start)
                    continue
                errors = validation.get("errors") if isinstance(validation, dict) else None
                warnings = validation.get("warnings") if isinstance(validation, dict) else None
                selector_counts = validation.get("selector_counts") if isinstance(validation, dict) else None
                warning_detail = None
                if isinstance(warnings, list) and warnings:
                    warning_detail = f"{len(warnings)} selector warning(s)"
                if errors:
                    trace.append(
                        _trace_entry(
                            strategy,
                            "failed",
                            "schema_invalid_selectors",
                            f"{len(errors)} invalid selector(s)",
                        )
                    )
                    _record_strategy_metrics(strategy, "failed", time.perf_counter() - start)
                    continue
                with _strategy_throttle(strategy):
                    result, exc, _attempts = _run_with_retries(
                        lambda: extract_schema_fields(html, url, schema_rules),
                        strategy=strategy,
                    )
                if exc or result is None:
                    reason = "schema_import_error" if isinstance(exc, ImportError) else "schema_error"
                    trace.append(_trace_entry(strategy, "failed", reason))
                    _record_strategy_metrics(strategy, "failed", time.perf_counter() - start)
                    continue
                if warning_detail:
                    result["schema_selector_warnings"] = warnings
                if isinstance(selector_counts, dict):
                    normalized_counts: dict[str, int] = {}
                    for key, count in selector_counts.items():
                        if not isinstance(key, str):
                            continue
                        if key.startswith("fields.") or key.startswith("baseFields."):
                            norm_key = key.split(".", 1)[1]
                        else:
                            norm_key = key
                        normalized_counts[norm_key] = int(count)
                    result["schema_selector_counts"] = normalized_counts
                result["schema_rule_keys"] = _schema_rule_keys(schema_rules)
                if result.get("extraction_successful"):
                    _schema_cache_put(cache_key, result)
                    trace.append(_trace_entry(strategy, "success", "schema_extracted", warning_detail))
                    _record_strategy_metrics(strategy, "success", time.perf_counter() - start, result)
                    return _finalize_result(result, strategy=strategy)
                trace.append(_trace_entry(strategy, "failed", "schema_no_content", warning_detail))
                _record_strategy_metrics(strategy, "failed", time.perf_counter() - start, result)
                last_result = result
                continue
            if handler is None:
                trace.append(_trace_entry(strategy, "skipped", "no_schema_rules_or_handler"))
                _record_strategy_metrics(strategy, "skipped", time.perf_counter() - start)
                continue
            with _strategy_throttle(strategy):
                result, exc, _attempts = _run_with_retries(lambda: handler(html, url), strategy=strategy)
            if exc or result is None:
                trace.append(_trace_entry(strategy, "failed", "handler_error"))
                _record_strategy_metrics(strategy, "failed", time.perf_counter() - start)
                continue
            if "extraction_trace" in result:
                result["handler_trace"] = result.pop("extraction_trace")
            if result.get("extraction_successful"):
                trace.append(_trace_entry(strategy, "success", "handler_extracted"))
                _record_strategy_metrics(strategy, "success", time.perf_counter() - start, result)
                return _finalize_result(result, strategy=strategy)
            trace.append(_trace_entry(strategy, "failed", "handler_no_content"))
            _record_strategy_metrics(strategy, "failed", time.perf_counter() - start, result)
            last_result = result
            continue
        if strategy == "regex":
            mask_override = _regex_mask_override(regex_settings)
            with _strategy_throttle(strategy):
                result = extract_regex_entities(html, url, mask_pii=mask_override)
            last_result = result
            if result.get("extraction_successful"):
                trace.append(_trace_entry(strategy, "success", "regex_extracted"))
                _record_strategy_metrics(strategy, "success", time.perf_counter() - start, result)
                return _finalize_result(result, strategy=strategy)
            trace.append(_trace_entry(strategy, "failed", "regex_no_matches"))
            _record_strategy_metrics(strategy, "failed", time.perf_counter() - start, result)
            continue
        if strategy == "llm":
            with _strategy_throttle(strategy):
                result = extract_llm_entities(html, url, llm_settings=llm_settings, schema_rules=schema_rules)
            last_result = result
            if result.get("extraction_successful"):
                trace.append(_trace_entry(strategy, "success", "llm_extracted"))
                _record_strategy_metrics(strategy, "success", time.perf_counter() - start, result)
                return _finalize_result(result, strategy=strategy)
            detail = result.get("llm_error")
            trace.append(_trace_entry(strategy, "failed", "llm_no_content", str(detail) if detail else None))
            _record_strategy_metrics(strategy, "failed", time.perf_counter() - start, result)
            continue
        if strategy == "cluster":
            with _strategy_throttle(strategy):
                result = extract_cluster_entities(html, url, cluster_settings=cluster_settings)
            last_result = result
            if result.get("extraction_successful"):
                detail = f"cluster_blocks={result.get('cluster_block_count')}"
                trace.append(_trace_entry(strategy, "success", "cluster_extracted", detail))
                _record_strategy_metrics(strategy, "success", time.perf_counter() - start, result)
                return _finalize_result(result, strategy=strategy)
            detail = result.get("cluster_error")
            trace.append(_trace_entry(strategy, "failed", "cluster_no_content", str(detail) if detail else None))
            _record_strategy_metrics(strategy, "failed", time.perf_counter() - start, result)
            continue
        if strategy == "trafilatura":
            extractor = fallback_extractor or _extract_with_trafilatura
            with _strategy_throttle(strategy):
                result, exc, _attempts = _run_with_retries(
                    lambda _extractor=extractor: _extractor(html, url),
                    strategy=strategy,
                )
            if exc or result is None:
                trace.append(_trace_entry(strategy, "failed", "extractor_error"))
                _record_strategy_metrics(strategy, "failed", time.perf_counter() - start)
                continue
            last_result = result
            if result.get("extraction_successful"):
                trace.append(_trace_entry(strategy, "success", "extracted"))
                _record_strategy_metrics(strategy, "success", time.perf_counter() - start, result)
                return _finalize_result(result, strategy=strategy)
            trace.append(_trace_entry(strategy, "failed", "no_content"))
            _record_strategy_metrics(strategy, "failed", time.perf_counter() - start, result)

    if last_result is None:
        last_result = {
            "title": "N/A",
            "author": "N/A",
            "content": "",
            "date": "N/A",
            "url": url,
            "extraction_successful": False,
        }
    return _finalize_result(last_result, strategy=None)


def extract_article_data_from_html(
    html: str,
    url: str,
    strategy_order: Optional[list[str]] = None,
    handler: Optional[Callable[[str, str], dict[str, Any]]] = None,
    schema_rules: Optional[dict[str, Any]] = None,
    llm_settings: Optional[dict[str, Any]] = None,
    regex_settings: Optional[dict[str, Any]] = None,
    cluster_settings: Optional[dict[str, Any]] = None,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    """Extract article metadata and body from raw HTML."""
    return extract_article_with_pipeline(
        html,
        url,
        strategy_order=strategy_order,
        handler=handler,
        schema_rules=schema_rules,
        llm_settings=llm_settings,
        regex_settings=regex_settings,
        cluster_settings=cluster_settings,
        allow_llm_extraction=allow_llm_extraction,
    )


def _record_robot_policy_block(url: str, reason: str) -> None:
    if not reason.startswith("robots_"):
        return
    try:
        parsed = urlparse(url)
        increment_counter("scrape_blocked_by_robots_total", labels={"domain": parsed.netloc})
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        increment_counter("scrape_blocked_by_robots_total", labels={})


def _blocked_article_result(
    url: str,
    decision: PolicyDecision,
) -> dict[str, Any]:
    _record_robot_policy_block(url, decision.reason)
    if decision.reason.startswith("robots_"):
        error = "Blocked by outbound policy"
    else:
        error = f"Egress denied: {decision.reason}"
    return {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": error,
        "policy_reason": decision.reason,
        "policy_mode": decision.mode,
        "policy_stage": decision.stage,
        "policy_source": decision.source,
    }


def _fetch_article_lightweight(
    url: str,
    *,
    backend_choice: str,
    headers: dict[str, str],
    cookies: dict[str, str] | None,
    timeout: float,
    impersonate: str | None,
    proxies: dict[str, str] | None,
    fetch_client: FetchClient | None = None,
) -> tuple[FetchResponse, str]:
    client = fetch_client or _ARTICLE_FETCH_CLIENT

    def _fetch_with_backend(backend: str) -> FetchResponse:
        return client.fetch(
            FetchRequest(
                url=url,
                method="GET",
                headers=headers,
                cookies=cookies or {},
                timeout=timeout,
                backend=backend,
                allow_redirects=True,
                impersonate=impersonate,
                proxies=proxies,
                context=RuntimeRequestContext(source="article_extract", stage="fetch"),
            )
        )

    if backend_choice == "curl":
        try:
            response = _fetch_with_backend("curl")
            return response, response.backend or "curl"
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("curl backend failed; falling back to httpx: {}", exc)

    response = _fetch_with_backend("httpx")
    return response, response.backend or "httpx"


async def scrape_article(
    url: str,
    custom_cookies: Optional[list[dict[str, Any]]] = None,
    *,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    logging.info("Scraping article request.")
    # Resolve scraper plan via router (configurable via YAML)
    ws_cfg: dict[str, Any] = {}

    def _resolve_scrape_plan() -> Any:
        nonlocal ws_cfg
        cfg = load_and_log_configs() or {}
        ws_cfg = cfg.get("web_scraper", {}) or {}
        rules_path = ws_cfg.get("custom_scrapers_yaml_path", _default_rules_path())
        rules = ScraperRouter.load_rules_from_yaml(rules_path)
        ua_mode = str(ws_cfg.get("web_scraper_ua_mode", "fixed") or "fixed")
        respect_robots_default = ws_cfg.get("web_scraper_respect_robots", True)
        if isinstance(respect_robots_default, str):
            respect_robots_default = _is_truthy(respect_robots_default.strip())
        router = ScraperRouter(
            rules,
            ua_mode=ua_mode,
            default_respect_robots=bool(respect_robots_default),
        )
        return router.resolve(url)

    try:
        plan = await asyncio.to_thread(_resolve_scrape_plan)
    except asyncio.CancelledError:
        raise
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        # Safe default plan
        class _P:  # minimal stand-in
            backend = "auto"
            handler = "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html"
            ua_profile = pick_ua_profile("fixed")
            impersonate = None
            extra_headers = {}
            cookies = {}
            respect_robots = True

        plan = _P()  # type: ignore

    handler_path = str(getattr(plan, "handler", "") or "")
    handler_func = resolve_handler(handler_path) if handler_path else None
    use_handler = bool(handler_path)
    strategy_order = getattr(plan, "strategy_order", None)
    schema_rules = getattr(plan, "schema_rules", None)
    llm_settings = getattr(plan, "llm_settings", None)
    regex_settings = getattr(plan, "regex_settings", None)
    cluster_settings = getattr(plan, "cluster_settings", None)

    # Build effective headers from UA profile + extras
    ua_headers = build_browser_headers(plan.ua_profile, accept_lang="en-US,en;q=0.9")
    if isinstance(plan.extra_headers, dict) and plan.extra_headers:
        ua_headers.update({str(k): str(v) for k, v in plan.extra_headers.items()})

    backend_choice = str(getattr(plan, "backend", "auto") or "auto").lower().strip()
    if backend_choice not in {"auto", "curl", "httpx", "playwright"}:
        backend_choice = "auto"
    if backend_choice == "auto":
        default_backend = ws_cfg.get("web_scraper_default_backend")
        if isinstance(default_backend, str):
            backend_choice = default_backend.lower().strip() or "auto"
            if backend_choice not in {"auto", "curl", "httpx", "playwright"}:
                backend_choice = "auto"

    preflight_payload = None

    def _attach_preflight(result: dict[str, Any]) -> dict[str, Any]:
        if preflight_payload and isinstance(result, dict):
            result.setdefault("preflight_analysis", preflight_payload)
        return result

    effective_ua = ua_headers.get("User-Agent", web_scraping_user_agent)
    try:
        target = await preflight_facade.evaluate_target(
            url,
            respect_robots=bool(getattr(plan, "respect_robots", True)),
            user_agent=effective_ua,
            request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
            config={"web_scraper": ws_cfg},
            policy_checker=_ARTICLE_POLICY_CHECKER,
        )
    except asyncio.CancelledError:
        raise
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
        logging.error(
            "Outbound policy evaluation failed. "
            "source=article_extract stage=pre_fetch "
            f"exception_type={type(exc).__name__[:80]}"
        )
        return _attach_preflight(
            {
                "url": url,
                "title": "N/A",
                "author": "N/A",
                "date": "N/A",
                "content": "",
                "extraction_successful": False,
                "error": "Outbound policy evaluation failed. Please contact system administrator.",
            }
        )

    if not target.decision.allowed:
        return _attach_preflight(_blocked_article_result(url, target.decision))

    options = preflight_facade.PreflightOptions.from_mapping(ws_cfg)
    preflight_result = None
    try:
        if options.enabled:
            context = preflight_facade.build_execution_context(
                target,
                options,
                policy_checker=_ARTICLE_POLICY_CHECKER,
            )
            preflight_result = await preflight_facade.run_preflight(target, options, context)
    except asyncio.CancelledError:
        raise
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        logging.debug("Preflight analysis failed.")

    backend_choice, preflight_method, preflight_result = preflight_facade.apply_preflight_advice(
        preflight_result,
        backend=backend_choice,
        method="auto",
        backend_setting=str(getattr(plan, "backend", "auto") or "auto").lower().strip(),
    )
    preflight_payload = preflight_facade.public_preflight_payload(
        preflight_result,
        options.include_results,
    )

    if backend_choice != "playwright" and preflight_method != "playwright":
        # First try lightweight HTTP path (curl/httpx) before Playwright
        try:
            cookies_map = _merge_cookie_list_to_map(custom_cookies)
            # Combine with plan cookies (plan cookies win)
            if getattr(plan, "cookies", {}):
                cookies_map.update({str(k): str(v) for k, v in plan.cookies.items()})

            t0 = time.time()
            resp, backend_used = await asyncio.to_thread(
                _fetch_article_lightweight,
                url,
                backend_choice=backend_choice,
                headers=ua_headers,
                cookies=cookies_map or None,
                timeout=15.0,
                impersonate=getattr(plan, "impersonate", None),
                proxies=getattr(plan, "proxies", None) or None,
            )

            elapsed = max(0.0, time.time() - t0)
            observe_histogram("scrape_fetch_latency_seconds", elapsed, labels={"backend": backend_used})

            status = _resp_get(resp, "status")
            if status is None:
                status = _resp_get(resp, "status_code")
            text = _resp_get(resp, "text", "")
            if int(status or 0) < 400 and text:
                # JS-required detection heuristic: decide earlier fallback
                if _js_required(text, _resp_get(resp, "headers", {}), url=url):
                    increment_counter("scrape_playwright_fallback_total", labels={"reason": "js_required"})
                    raise RuntimeError("js_required_detected")
                article_data = extract_article_with_pipeline(
                    text,
                    url,
                    strategy_order=strategy_order,
                    handler=handler_func if use_handler else None,
                    schema_rules=schema_rules,
                    llm_settings=llm_settings,
                    regex_settings=regex_settings,
                    cluster_settings=cluster_settings,
                    allow_llm_extraction=allow_llm_extraction,
                )
                if article_data.get("extraction_successful"):
                    if not use_handler and article_data.get("content"):
                        article_data["content"] = convert_html_to_markdown(article_data["content"])
                    content = article_data.get("content", "") or ""
                    logging.info(f"Article content length: {len(content)}")
                    observe_histogram(
                        "scrape_content_length_bytes",
                        len(content.encode("utf-8", errors="ignore")),
                        labels={"backend": backend_used},
                    )
                    increment_counter("scrape_fetch_total", labels={"backend": backend_used, "outcome": "success"})
                    return _attach_preflight(article_data)
            # No extractable content
            increment_counter("scrape_fetch_total", labels={"backend": backend_used, "outcome": "no_extract"})
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as _e:
            logging.debug(f"Lightweight fetch path failed or yielded no extractable content: {_e}")
            increment_counter("scrape_fetch_total", labels={"backend": backend_choice, "outcome": "error"})
            increment_counter("scrape_playwright_fallback_total", labels={"reason": "error"})
        else:
            # Falling back due to no extractable content
            increment_counter("scrape_playwright_fallback_total", labels={"reason": "no_extract"})

    async def fetch_html(url: str) -> str:
        # Load and log the configuration
        loaded_config = load_and_log_configs()

        # load retry count from config
        scrape_retry_count = loaded_config['web_scraper'].get('web_scraper_retry_count', 3)
        retries = int(scrape_retry_count) if isinstance(scrape_retry_count, str) else scrape_retry_count
        # Load retry timeout value from config
        web_scraper_retry_timeout = loaded_config['web_scraper'].get('web_scraper_retry_timeout', 60)
        # Interpret config as seconds; Playwright expects milliseconds
        timeout_sec = int(web_scraper_retry_timeout) if isinstance(web_scraper_retry_timeout, str) else web_scraper_retry_timeout
        timeout_ms = max(0, int(timeout_sec) * 1000)

        # Whether stealth mode is enabled
        stealth_raw = loaded_config['web_scraper'].get('web_scraper_stealth_playwright', False)
        if isinstance(stealth_raw, str):
            stealth_enabled = _is_truthy(stealth_raw.strip())
        else:
            stealth_enabled = bool(stealth_raw)

        for attempt in range(retries):  # Introduced a retry loop to attempt fetching HTML multiple times
            browser = None
            try:
                logging.info(f"Fetching HTML from {url} (Attempt {attempt + 1}/{retries})")
                t0 = time.time()
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context(
                        user_agent=effective_ua,
                        # Simulating a normal browser window size for better compatibility
                        viewport={"width": 1280, "height": 720},
                    )
                    if custom_cookies:
                        # Apply cookies if provided
                        await context.add_cookies(custom_cookies)

                    page = await context.new_page()

                    # Check if stealth mode is enabled in the config
                    if stealth_enabled:
                        try:
                            from playwright_stealth import stealth_async
                            await stealth_async(page)
                        except ImportError:
                            # Fallback if stealth_async is not available
                            logging.debug("playwright_stealth not properly installed, skipping stealth mode")

                    # Navigate to the URL
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

                    # If stealth is enabled, give the page extra time to finish loading/spawning content
                    if stealth_enabled:
                        # Try to get from config, fallback to hardcoded default
                        try:
                            from tldw_Server_API.app.core.config import config
                            stealth_wait_ms = config.get("STEALTH_WAIT_MS", 5000)
                        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"Falling back to default STEALTH_WAIT_MS; error={e}")
                            stealth_wait_ms = 5000
                        await page.wait_for_timeout(stealth_wait_ms)  # configurable delay
                    else:
                        # Alternatively, wait for network to be idle
                        await page.wait_for_load_state("networkidle", timeout=timeout_ms)

                    # Capture final HTML
                    content = await page.content()

                    # Metrics for Playwright path
                    elapsed = max(0.0, time.time() - t0)
                    observe_histogram("scrape_fetch_latency_seconds", elapsed, labels={"backend": "playwright"})
                    increment_counter("scrape_fetch_total", labels={"backend": "playwright", "outcome": "success"})
                    logging.info(f"HTML fetched successfully from {url}")
                    log_counter("html_fetched", labels={"url": url})

                # Return the scraped HTML
                return content

            except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
                logging.error(f"Error fetching HTML from {url} on attempt {attempt + 1}: {e}")
                increment_counter("scrape_fetch_total", labels={"backend": "playwright", "outcome": "error"})

                if attempt < retries - 1:
                    logging.info("Retrying...")
                    await asyncio.sleep(2)
                else:
                    logging.error("Max retries reached, giving up on this URL.")
                    log_counter("html_fetch_error", labels={"url": url, "error": str(e)})
                    return ""  # Return empty string on final failure

            finally:
                # Ensure the browser is closed before returning
                if browser is not None:
                    await browser.close()

        # If for some reason you exit the loop without returning (unlikely), return empty string
        return ""

    html = await fetch_html(url)
    article_data = extract_article_with_pipeline(
        html,
        url,
        strategy_order=strategy_order,
        handler=handler_func if use_handler else None,
        schema_rules=schema_rules,
        llm_settings=llm_settings,
        regex_settings=regex_settings,
        cluster_settings=cluster_settings,
        allow_llm_extraction=allow_llm_extraction,
    )
    if article_data.get("extraction_successful") and not use_handler and article_data.get("content"):
        article_data["content"] = convert_html_to_markdown(article_data["content"])
    if article_data.get("extraction_successful"):
        content = article_data.get("content", "") or ""
        logging.info(f"Article content length: {len(content)}")
        observe_histogram(
            "scrape_content_length_bytes",
            len(content.encode("utf-8", errors="ignore")),
            labels={"backend": "playwright"},
        )
    return _attach_preflight(article_data)


def scrape_article_blocking(
    url: str,
    custom_cookies: Optional[list[dict[str, Any]]] = None,
    *,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    """Blocking scraper for synchronous code paths.

    Fetches HTML with http_client using a desktop-like user agent and optional cookies,
    then extracts article content via trafilatura and converts to display text.
    """
    try:
        try:
            decision = decide_web_outbound_policy_sync(
                url,
                respect_robots=False,
                source="article_extract_blocking",
                stage="pre_fetch",
            )
            if not decision.allowed:
                return _blocked_article_result(url, decision)
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
            return {
                "url": url,
                "title": "N/A",
                "author": "N/A",
                "date": "N/A",
                "content": "",
                "extraction_successful": False,
                "error": "Outbound policy evaluation failed",
            }

        headers = {"User-Agent": web_scraping_user_agent}
        # If cookies are provided in Playwright-style dicts, reduce to name->value and set Cookie header
        if custom_cookies:
            cookie_map = _merge_cookie_list_to_map(custom_cookies)
            if cookie_map:
                cookie_hdr = "; ".join([f"{k}={v}" for k, v in cookie_map.items()])
                headers["Cookie"] = cookie_hdr
        resp = http_fetch(method="GET", url=url, timeout=30, headers=headers)
        try:
            status = _resp_get(resp, "status")
            if status is None:
                status = _resp_get(resp, "status_code", 0)
            text = _resp_get(resp, "text", "")
            if not text:
                content = _resp_get(resp, "content", b"")
                try:
                    text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else str(content)
                except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
                    text = ""
        finally:
            close = getattr(resp, "close", None)
            if callable(close):
                close()

        if int(status or 0) != 200:
            logging.error(f"Failed to fetch {url}, status: {status}")
            return {"url": url, "title": "N/A", "author": "N/A", "date": "N/A", "content": "", "extraction_successful": False}

        article_data = extract_article_data_from_html(
            text,
            url,
            allow_llm_extraction=allow_llm_extraction,
        )
        if article_data.get("extraction_successful"):
            article_data["content"] = convert_html_to_markdown(article_data["content"])
        return article_data
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Blocking scrape failed for {url}: {e}")
        return {"url": url, "title": "N/A", "author": "N/A", "date": "N/A", "content": "", "extraction_successful": False}


# FIXME - Add keyword integration/tagging
async def scrape_and_summarize_multiple(
    urls: str,
    custom_prompt_arg: Optional[str],
    api_name: str,
    api_key: Optional[str],
    keywords: str,
    custom_article_titles: Optional[str],
    system_message: Optional[str] = None,
    summarize_checkbox: bool = False,
    custom_cookies: Optional[list[dict[str, Any]]] = None,
    temperature: float = 0.7,
    allow_llm_extraction: bool = True,
) -> list[dict[str, Any]]:
    urls_list = [url.strip() for url in urls.split('\n') if url.strip()]
    custom_titles = custom_article_titles.split('\n') if custom_article_titles else []

    results = []
    errors = []

    # Apply polite scraping rate limits (and optional Resource Governor backoff)
    # for each outbound fetch. This is intentionally best-effort and must never
    # block scraping when the limiter cannot be constructed.
    try:
        rate_limiter = RateLimiter()
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        rate_limiter = None

    # Create a tqdm progress bar
    progress_bar = tqdm(total=len(urls_list), desc="Scraping and Summarizing")

    # Loop over each URL to scrape and optionally summarize
    for i, url in enumerate(urls_list):
        custom_title = custom_titles[i] if i < len(custom_titles) else None
        try:
            if rate_limiter is not None:
                with suppress(_ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS):
                    await rate_limiter.acquire()
            # Scrape the article
            article = await scrape_article(
                url,
                custom_cookies=custom_cookies,
                allow_llm_extraction=allow_llm_extraction,
            )
            if article and article['extraction_successful']:
                log_counter("article_scraped", labels={"success": "true", "url": url})
                if custom_title:
                    article['title'] = custom_title

                # If summarization is requested
                if summarize_checkbox:
                    content = article.get('content', '')
                    if content:
                        # Prepare prompts
                        system_message_final = system_message or \
                                               "Act as a professional summarizer and summarize this article."
                        article_custom_prompt = custom_prompt_arg or \
                                                "Act as a professional summarizer and summarize this article."

                        # Summarize the content using the summarize function
                        summary = analyze(
                            input_data=content,
                            custom_prompt_arg=article_custom_prompt,
                            api_name=api_name,
                            api_key=api_key,
                            temp=temperature,
                            system_message=system_message_final
                        )
                        article['summary'] = summary
                        log_counter("article_summarized", labels={"success": "true", "url": url})
                        logging.info(f"Summary generated for URL {url}")
                    else:
                        article['summary'] = "No content available to summarize."
                        logging.warning(f"No content to summarize for URL {url}")
                else:
                    article.setdefault('summary', None)

                results.append(article)
            else:
                error_message = f"Extraction unsuccessful for URL {url}"
                errors.append(error_message)
                logging.error(error_message)
                log_counter("article_scraped", labels={"success": "false", "url": url})
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
            log_counter("article_processing_error", labels={"url": url})
            error_message = f"Error processing URL {i + 1} ({url}): {str(e)}"
            errors.append(error_message)
            logging.error(error_message, exc_info=True)
        finally:
            # Update the progress bar
            progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    if errors:
        logging.error("\n".join(errors))

    if not results:
        logging.error("No articles were successfully scraped and summarized/analyzed.")
        return []

    log_histogram("articles_processed", len(results))
    return results


async def async_scrape_and_no_summarize_then_ingest(url, keywords, custom_article_title):
    try:
        # Step 1: Scrape the article
        article_data = await scrape_article(url)
        if not article_data:
            log_counter("article_scrape_failed", labels={"url": url})
            return "Failed to scrape the article."

        # Use the custom title if provided, otherwise use the scraped title
        title = custom_article_title.strip() if custom_article_title else article_data.get('title', 'Untitled')
        author = article_data.get('author', 'Unknown')
        content = article_data.get('content', '')
        ingestion_date = datetime.now().strftime('%Y-%m-%d')

        # Step 2: Ingest the article into the database
        with managed_media_database(
            client_id="article_extractor",
            initialize=False,
        ) as db_instance:
            # Ensure keywords list
            kw_list = [kw.strip() for kw in str(keywords).split(',')] if isinstance(keywords, str) else (keywords or [])
            ingestion_result = ingest_article_to_db(
                db_instance=db_instance,
                url=url,
                title=title,
                author=author,
                content=content,
                keywords=kw_list,
                ingestion_date=ingestion_date,
                custom_prompt=None,
                summary=None,
            )
        log_counter("article_ingested", labels={"success": str(ingestion_result).lower(), "url": url})

        # When displaying content, we might want to strip metadata
        display_content = ContentMetadataHandler.strip_metadata(content)
        return f"Title: {title}\nAuthor: {author}\nIngestion Result: {ingestion_result}\n\nArticle Contents: {display_content}"
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
        log_counter("article_processing_error", labels={"url": url})
        logging.error(f"Error processing URL {url}: {str(e)}")
        return f"Failed to process URL {url}: {str(e)}"

def scrape_and_no_summarize_then_ingest(url, keywords, custom_article_title):
    """Synchronous wrapper for CLI usage.

    In async contexts, prefer calling async_scrape_and_no_summarize_then_ingest.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_scrape_and_no_summarize_then_ingest(url, keywords, custom_article_title))
    raise RuntimeError("Call async_scrape_and_no_summarize_then_ingest() within async contexts")


def scrape_from_filtered_sitemap(sitemap_file: str, filter_function) -> list:
    """
    Scrape articles from a sitemap file, applying an additional filter function.

    :param sitemap_file: Path to the sitemap file
    :param filter_function: A function that takes a URL and returns True if it should be scraped
    :return: List of scraped articles
    """
    try:
        tree = xET.parse(sitemap_file)
        root = tree.getroot()

        articles = []
        for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
            if filter_function(url.text):
                article_data = scrape_article_blocking(url.text)
                if article_data:
                    articles.append(article_data)

        return articles
    except (xET.ParseError, DefusedXmlException) as e:
        logging.error(f"Error parsing sitemap: {e}")
        return []


def is_content_page(url: str) -> bool:
    """
    Determine if a URL is likely to be a content page.
    This is a basic implementation and may need to be adjusted based on the specific website structure.

    :param url: The URL to check
    :return: True if the URL is likely a content page, False otherwise
    """
    # Exclude common non-content pages
    exclude_patterns = [
        '/tag/', '/category/', '/author/', '/search/', '/page/',
        'wp-content', 'wp-includes', 'wp-json', 'wp-admin',
        'login', 'register', 'cart', 'checkout', 'account',
        '.jpg', '.png', '.gif', '.pdf', '.zip'
    ]
    chain = FilterChain([
        ContentTypeFilter(),
        URLPatternFilter(include_patterns=None, exclude_patterns=exclude_patterns)
    ])
    return chain.apply(url)

def scrape_and_convert_with_filter(source: str, output_file: str, filter_function=is_content_page, level: int = None):
    """
    Scrape articles from a sitemap or by URL level, apply filtering, and convert to a single markdown file.

    :param source: URL of the sitemap, base URL for level-based scraping, or path to a local sitemap file
    :param output_file: Path to save the output markdown file
    :param filter_function: Function to filter URLs (default is is_content_page)
    :param level: URL level for scraping (None if using sitemap)
    """
    if level is not None:
        # Scraping by URL level
        articles = scrape_by_url_level(source, level)
        articles = [article for article in articles if filter_function(article['url'])]
    elif source.startswith('http'):
        # Scraping from online sitemap
        articles = scrape_from_sitemap(source)
        articles = [article for article in articles if filter_function(article['url'])]
    else:
        # Scraping from local sitemap file
        articles = scrape_from_filtered_sitemap(source, filter_function)

    articles = [article for article in articles if filter_function(article['url'])]
    markdown_content = convert_to_markdown(articles)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    logging.info(f"Scraped and filtered content saved to {output_file}")


async def scrape_entire_site(base_url: str) -> list[dict]:
    """
    Scrape the entire site by generating a temporary sitemap and extracting content from each page.

    :param base_url: The base URL of the site to scrape
    :return: A list of dictionaries containing scraped article data
    """
    # Step 1: Collect internal links from the site (async, with rate limiting)
    try:
        rate = RateLimiter(max_requests_per_second=1.5, max_requests_per_minute=60, max_requests_per_hour=1000)
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
        rate = None
    links = await async_collect_internal_links(base_url, rate_limiter=rate)
    log_histogram("internal_links_collected", len(links), labels={"base_url": base_url})
    logging.info(f"Collected {len(links)} internal links.")

    # Step 2: Generate the temporary sitemap
    temp_sitemap_path = generate_temp_sitemap_from_links(links)

    # Step 3: Scrape each URL in the sitemap
    scraped_articles = []
    try:
        async def scrape_and_log(link):
            logging.info(f"Scraping {link} ...")
            article_data = await scrape_article(link)

            if article_data:
                logging.info(f"Title: {article_data['title']}")
                logging.info(f"Author: {article_data['author']}")
                logging.info(f"Date: {article_data['date']}")
                logging.info(f"Content: {article_data['content'][:500]}...")

                return article_data
            return None

        # Use asyncio.gather to scrape multiple articles concurrently
        scraped_articles = await asyncio.gather(*[scrape_and_log(link) for link in links])
        # Remove any None values (failed scrapes)
        scraped_articles = [article for article in scraped_articles if article is not None]
        log_histogram("articles_scraped", len(scraped_articles), labels={"base_url": base_url})

    finally:
        # Clean up the temporary sitemap file
        os.unlink(temp_sitemap_path)
        logging.info("Temporary sitemap file deleted")

    return scraped_articles


def scrape_by_url_level(
    base_url: str,
    level: int,
    *,
    allow_llm_extraction: bool = True,
) -> list:
    """Scrape articles from URLs up to a certain level under the base URL."""

    def get_url_level(url: str) -> int:
        return len(urlparse(url).path.strip('/').split('/'))

    links = collect_internal_links(base_url)
    filtered_links = [link for link in links if get_url_level(link) <= level]

    results = []
    for link in filtered_links:
        article = scrape_article_blocking(
            link,
            allow_llm_extraction=allow_llm_extraction,
        )
        if article:
            results.append(article)
    return results


def scrape_from_sitemap(
    sitemap_url: str,
    *,
    allow_llm_extraction: bool = True,
) -> list:
    """Scrape articles from a sitemap URL."""
    try:
        try:
            decision = decide_web_outbound_policy_sync(
                sitemap_url,
                respect_robots=False,
                source="sitemap_scrape",
                stage="pre_fetch",
            )
            if not decision.allowed:
                logging.error(f"Sitemap blocked by outbound policy: {decision.reason}")
                return []
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
            logging.error(f"Outbound policy evaluation failed: {exc}")
            return []

        try:
            resp = http_fetch(method="GET", url=sitemap_url, timeout=10)
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as fetch_err:
            logging.error(f"Sitemap fetch failed via http_fetch: {fetch_err}")
            return []
        try:
            status = _resp_get(resp, "status")
            if status is None:
                status = _resp_get(resp, "status_code", 0)
            text = _resp_get(resp, "text", "")
            if not text:
                # Fallback for response objects that expose `content` only
                content = _resp_get(resp, "content", b"")
                try:
                    text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else str(content)
                except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
                    text = ""
        finally:
            close = getattr(resp, "close", None)
            if callable(close):
                close()

        if int(status or 0) >= 400 or not text:
            return []
        try:
            root = xET.fromstring(text)
        except (xET.ParseError, DefusedXmlException) as parse_err:
            logging.error(f"Failed to parse sitemap XML from {sitemap_url}: {parse_err}")
            return []

        results = []
        for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
            article = scrape_article_blocking(
                url.text,
                allow_llm_extraction=allow_llm_extraction,
            )
            if article:
                results.append(article)
        return results
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error fetching sitemap: {e}")
        return []

#
# End of Scraping Functions
#######################################################
#
# Sitemap/Crawling-related Functions


def collect_internal_links(base_url: str) -> set:
    visited = set()
    to_visit = {base_url}

    try:
        decision = decide_web_outbound_policy_sync(
            base_url,
            respect_robots=False,
            source="collect_internal_links",
            stage="pre_fetch",
        )
        if not decision.allowed:
            logging.error(f"Base URL blocked by outbound policy: {decision.reason}")
            return visited
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
        logging.error(f"Outbound policy evaluation failed: {exc}")
        return visited

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        try:
            resp = http_fetch(method="GET", url=current_url, timeout=10)
            if resp.get("status", 0) >= 400:
                continue
            soup = BeautifulSoup(resp.get("text", ""), 'html.parser')

            # Collect internal links
            for link in soup.find_all('a', href=True):
                full_url = urljoin(base_url, link['href'])
                # Only process links within the same domain
                if urlparse(full_url).netloc == urlparse(base_url).netloc and full_url not in visited:
                    to_visit.add(full_url)

            visited.add(current_url)
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Error visiting {current_url}: {e}")
            continue

    return visited


async def async_collect_internal_links(base_url: str,
                                       max_pages: int = 500,
                                       rate_limiter: Optional[RateLimiter] = None,
                                       request_timeout: int = 20) -> set:
    """Async internal link collector using http_client and optional rate limiter."""
    visited: set = set()
    to_visit: set = {base_url}

    headers = {"User-Agent": web_scraping_user_agent}
    timeout = float(request_timeout)

    async def _close_resp(resp: Any) -> None:
        close = getattr(resp, "aclose", None)
        if callable(close):
            await close()
            return
        close = getattr(resp, "close", None)
        if callable(close):
            close()

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop()
        if current_url in visited:
            continue
        try:
            if rate_limiter:
                await rate_limiter.acquire()
            resp = await afetch(
                method="GET",
                url=current_url,
                headers=headers,
                timeout=timeout,
            )
            try:
                status = getattr(resp, "status_code", None)
                if status is None:
                    status = getattr(resp, "status", None)
                if status is not None and int(status) != 200:
                    continue
                text = resp.text
            finally:
                await _close_resp(resp)
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
            continue

        visited.add(current_url)
        try:
            soup = BeautifulSoup(text, 'html.parser')
            for link in soup.find_all('a', href=True):
                full_url = urljoin(base_url, link['href'])
                if urlparse(full_url).netloc == urlparse(base_url).netloc and full_url not in visited:
                    to_visit.add(full_url)
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
            continue

    return visited

def generate_temp_sitemap_from_links(links: set) -> str:
    """
    Generate a temporary sitemap file from collected links and return its path.

    :param links: A set of URLs to include in the sitemap
    :return: Path to the temporary sitemap file
    """
    # Create the root element
    urlset = xET.Element("urlset")
    urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    # Add each link to the sitemap
    for link in links:
        url = xET.SubElement(urlset, "url")
        loc = xET.SubElement(url, "loc")
        loc.text = link
        lastmod = xET.SubElement(url, "lastmod")
        lastmod.text = datetime.now().strftime("%Y-%m-%d")
        changefreq = xET.SubElement(url, "changefreq")
        changefreq.text = "daily"
        priority = xET.SubElement(url, "priority")
        priority.text = "0.5"

    # Create the tree and get it as a string
    xml_string = xET.tostring(urlset, 'utf-8')

    # Pretty print the XML
    pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as temp_file:
        temp_file.write(pretty_xml)
        temp_file_path = temp_file.name

    logging.info(f"Temporary sitemap created at: {temp_file_path}")
    return temp_file_path


def generate_sitemap_for_url(url: str) -> list[dict[str, str]]:
    """
    Generate a sitemap for the given URL using the create_filtered_sitemap function.

    Args:
        url (str): The base URL to generate the sitemap for

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'url' and 'title' keys
    """
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", delete=False) as temp_file:
        create_filtered_sitemap(url, temp_file.name, is_content_page)
        temp_file.seek(0)
        tree = xET.parse(temp_file.name)
        root = tree.getroot()

        sitemap = []
        for url_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
            sitemap.append({"url": loc, "title": loc.split("/")[-1] or url})  # Use the last part of the URL as a title

    return sitemap

def create_filtered_sitemap(base_url: str, output_file: str, filter_function):
    """
    Create a sitemap from internal links and filter them based on a custom function.

    :param base_url: The base URL of the website
    :param output_file: The file to save the sitemap to
    :param filter_function: A function that takes a URL and returns True if it should be included
    """
    links = collect_internal_links(base_url)
    filtered_links = set(filter(filter_function, links))

    root = xET.Element("urlset")
    root.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    for link in filtered_links:
        url = xET.SubElement(root, "url")
        loc = xET.SubElement(url, "loc")
        loc.text = link

    tree = xET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Filtered sitemap saved to {output_file}")


#
# End of Crawling Functions
#################################################################
#
# Utility Functions

def convert_to_markdown(articles: list) -> str:
    """Convert a list of article data into a single markdown document."""
    markdown = ""
    for article in articles:
        markdown += f"# {article['title']}\n\n"
        markdown += f"Author: {article['author']}\n"
        markdown += f"Date: {article['date']}\n\n"
        markdown += f"{article['content']}\n\n"
        markdown += "---\n\n"  # Separator between articles
    return markdown

def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def load_hashes(filename: str) -> dict[str, str]:
    if os.path.exists(filename):
        with open(filename) as f:
            return json.load(f)
    else:
        return {}

def save_hashes(hashes: dict[str, str], filename: str):
    with open(filename, 'w') as f:
        json.dump(hashes, f)

def has_page_changed(url: str, new_hash: str, stored_hashes: dict[str, str]) -> bool:
    old_hash = stored_hashes.get(url)
    return old_hash != new_hash


#
#
###################################################
#
# Bookmark Parsing Functions

def parse_chromium_bookmarks(json_data: dict) -> dict[str, Union[str, list[str]]]:
    """
    Parse Chromium-based browser bookmarks from JSON data.

    :param json_data: The JSON data from the bookmarks file
    :return: A dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    """
    bookmarks = {}

    def recurse_bookmarks(nodes):
        for node in nodes:
            if node.get('type') == 'url':
                name = node.get('name')
                url = node.get('url')
                if name and url:
                    if name in bookmarks:
                        if isinstance(bookmarks[name], list):
                            bookmarks[name].append(url)
                        else:
                            bookmarks[name] = [bookmarks[name], url]
                    else:
                        bookmarks[name] = url
            elif node.get('type') == 'folder' and 'children' in node:
                recurse_bookmarks(node['children'])

    # Chromium bookmarks have a 'roots' key
    if 'roots' in json_data:
        for root in json_data['roots'].values():
            if 'children' in root:
                recurse_bookmarks(root['children'])
    else:
        recurse_bookmarks(json_data.get('children', []))

    return bookmarks


def parse_firefox_bookmarks(html_content: str) -> dict[str, Union[str, list[str]]]:
    """
    Parse Firefox bookmarks from HTML content.

    :param html_content: The HTML content from the bookmarks file
    :return: A dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    """
    bookmarks = {}
    soup = BeautifulSoup(html_content, 'html.parser')

    # Firefox stores bookmarks within <a> tags inside <dt>
    for a in soup.find_all('a'):
        name = a.get_text()
        url = a.get('href')
        if name and url:
            if name in bookmarks:
                if isinstance(bookmarks[name], list):
                    bookmarks[name].append(url)
                else:
                    bookmarks[name] = [bookmarks[name], url]
            else:
                bookmarks[name] = url

    return bookmarks


def load_bookmarks(file_path: str) -> dict[str, Union[str, list[str]]]:
    """
    Load bookmarks from a file (JSON for Chrome/Edge or HTML for Firefox).

    :param file_path: Path to the bookmarks file
    :return: A dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    :raises ValueError: If the file format is unsupported or parsing fails
    """
    if not os.path.isfile(file_path):
        logging.error(f"File '{file_path}' does not exist.")
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.json' or ext == '':
        # Attempt to parse as JSON (Chrome/Edge)
        try:
            with open(file_path, encoding='utf-8') as f:
                json_data = json.load(f)
            return parse_chromium_bookmarks(json_data)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON. Ensure the file is a valid Chromium bookmarks JSON file.")
            raise ValueError("Invalid JSON format for Chromium bookmarks.") from None
    elif ext in ['.html', '.htm']:
        # Parse as HTML (Firefox)
        try:
            with open(file_path, encoding='utf-8') as f:
                html_content = f.read()
            return parse_firefox_bookmarks(html_content)
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Failed to parse HTML bookmarks: {e}")
            raise ValueError(f"Failed to parse HTML bookmarks: {e}") from e
    else:
        logging.error("Unsupported file format. Please provide a JSON (Chrome/Edge) or HTML (Firefox) bookmarks file.")
        raise ValueError("Unsupported file format for bookmarks.")


def collect_bookmarks(file_path: str) -> dict[str, Union[str, list[str]]]:
    """
    Collect bookmarks from the provided bookmarks file and return a dictionary.

    :param file_path: Path to the bookmarks file
    :return: Dictionary with bookmark names as keys and URLs as values or lists of URLs if duplicates exist
    """
    try:
        bookmarks = load_bookmarks(file_path)
        logging.info(f"Successfully loaded {len(bookmarks)} bookmarks from '{file_path}'.")
        return bookmarks
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading bookmarks: {e}")
        return {}


def parse_csv_urls(file_path: str) -> dict[str, Union[str, list[str]]]:
    """
    Parse URLs from a CSV file. The CSV should have at minimum a 'url' column,
    and optionally a 'title' or 'name' column.

    :param file_path: Path to the CSV file
    :return: Dictionary with titles/names as keys and URLs as values
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Check if required columns exist
        if 'url' not in df.columns:
            raise ValueError("CSV must contain a 'url' column")

        # Initialize result dictionary
        urls_dict = {}

        # Determine which column to use as key
        key_column = next((col for col in ['title', 'name'] if col in df.columns), None)

        for idx in range(len(df)):
            url = df.iloc[idx]['url'].strip()

            # Use title/name if available, otherwise use URL as key
            key = df.iloc[idx][key_column].strip() if key_column else f"Article {idx + 1}"

            # Handle duplicate keys
            if key in urls_dict:
                if isinstance(urls_dict[key], list):
                    urls_dict[key].append(url)
                else:
                    urls_dict[key] = [urls_dict[key], url]
            else:
                urls_dict[key] = url

        return urls_dict

    except pd.errors.EmptyDataError:
        logging.error("The CSV file is empty")
        return {}
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error parsing CSV file: {str(e)}")
        return {}


def collect_urls_from_file(file_path: str) -> dict[str, Union[str, list[str]]]:
    """
    Unified function to collect URLs from either bookmarks or CSV files.

    :param file_path: Path to the file (bookmarks or CSV)
    :return: Dictionary with names as keys and URLs as values
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.csv':
        return parse_csv_urls(file_path)
    else:
        return collect_bookmarks(file_path)

# Usage:
# from Article_Extractor_Lib import collect_bookmarks
#
# # Path to your bookmarks file
# # For Chrome or Edge (JSON format)
# chromium_bookmarks_path = "/path/to/Bookmarks"
#
# # For Firefox (HTML format)
# firefox_bookmarks_path = "/path/to/bookmarks.html"
#
# # Collect bookmarks from Chromium-based browser
# chromium_bookmarks = collect_bookmarks(chromium_bookmarks_path)
# print("Chromium Bookmarks:")
# for name, url in chromium_bookmarks.items():
#     print(f"{name}: {url}")
#
# # Collect bookmarks from Firefox
# firefox_bookmarks = collect_bookmarks(firefox_bookmarks_path)
# print("\nFirefox Bookmarks:")
# for name, url in firefox_bookmarks.items():
#     print(f"{name}: {url}")

#
# End of Bookmarking Parsing Functions
#####################################################################


#####################################################################
#
# Article Scraping Metadata Functions

##############################################################
#
# Scraping Functions

def get_url_depth(url: str) -> int:
    return len(urlparse(url).path.strip('/').split('/'))

def sync_recursive_scrape(
    url_input,
    max_pages,
    max_depth,
    delay=1.0,
    custom_cookies=None,
    allow_llm_extraction: bool = True,
):
    def run_async_scrape():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            recursive_scrape(
                url_input,
                max_pages,
                max_depth,
                delay=delay,
                custom_cookies=custom_cookies,
                allow_llm_extraction=allow_llm_extraction,
            )
        )

    with ThreadPoolExecutor(max_workers=_extractor_max_workers()) as executor:
        future = executor.submit(run_async_scrape)
        return future.result()

async def recursive_scrape(
        base_url: str,
        max_pages: int,
        max_depth: int,
        delay: float = 1.0,
        resume_file: str = 'scrape_progress.json',
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        custom_cookies: Optional[list[dict[str, Any]]] = None,
        progress_callback: Optional[callable] = None,
        allow_llm_extraction: bool = True,
) -> list[dict]:
    async def save_progress():
        temp_file = resume_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump({
                'visited': list(visited),
                'to_visit': to_visit,
                'scraped_articles': scraped_articles,
                'pages_scraped': pages_scraped
            }, f)
        os.replace(temp_file, resume_file)  # Atomic replace

    def is_valid_url(url: str) -> bool:
        return url.startswith("http") and len(url) > 0

    # Load progress if resume file exists
    if os.path.exists(resume_file):
        with open(resume_file) as f:
            progress_data = json.load(f)
            visited = set(progress_data['visited'])
            to_visit = progress_data['to_visit']
            scraped_articles = progress_data['scraped_articles']
            pages_scraped = progress_data['pages_scraped']
    else:
        visited = set()
        to_visit = [(base_url, 0)]  # (url, depth)
        scraped_articles = []
        pages_scraped = 0

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=user_agent)

            # Set custom cookies if provided
            if custom_cookies:
                await context.add_cookies(custom_cookies)

            try:
                while to_visit and pages_scraped < max_pages:
                    current_url, current_depth = to_visit.pop(0)

                    if current_url in visited or current_depth > max_depth:
                        continue

                    visited.add(current_url)

                    # Update progress if callback provided
                    if progress_callback:
                        progress_callback(f"Scraping page {pages_scraped + 1}/{max_pages}: {current_url}")

                    try:
                        await asyncio.sleep(random.uniform(delay * 0.8, delay * 1.2))  # nosec B311

                        article_data = await scrape_article_async(
                            context,
                            current_url,
                            allow_llm_extraction=allow_llm_extraction,
                        )

                        if article_data and article_data['extraction_successful']:
                            scraped_articles.append(article_data)
                            pages_scraped += 1

                        # If we haven't reached max depth, add child links to to_visit
                        if current_depth < max_depth:
                            page = await context.new_page()
                            await page.goto(current_url)
                            await page.wait_for_load_state("networkidle")

                            links = await page.eval_on_selector_all('a[href]',
                                                                    "(elements) => elements.map(el => el.href)")
                            for link in links:
                                child_url = urljoin(base_url, link)
                                if is_valid_url(child_url) and child_url.startswith(
                                        base_url) and child_url not in visited and should_scrape_url(child_url):
                                    to_visit.append((child_url, current_depth + 1))

                            await page.close()

                    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
                        logging.error(f"Error scraping {current_url}: {str(e)}")

                    # Save progress periodically (e.g., every 10 pages)
                    if pages_scraped % 10 == 0:
                        await save_progress()

            finally:
                await browser.close()

    finally:
        # These statements are now guaranteed to be reached after the scraping is done
        await save_progress()

        # Remove the progress file when scraping is completed successfully
        if os.path.exists(resume_file):
            os.remove(resume_file)

        # Final progress update
        if progress_callback:
            progress_callback(f"Scraping completed. Total pages scraped: {pages_scraped}")

    return scraped_articles

async def scrape_article_async(
    context,
    url: str,
    *,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]:
    page = await context.new_page()
    try:
        await page.goto(url)
        await page.wait_for_load_state("networkidle")

        title = await page.title()
        content = await page.content()

        article_data = extract_article_with_pipeline(
            content,
            url,
            allow_llm_extraction=allow_llm_extraction,
        )
        if article_data.get("extraction_successful"):
            if article_data.get("title") in {None, "", "N/A"}:
                article_data["title"] = title
            if article_data.get("content"):
                article_data["content"] = convert_html_to_markdown(article_data["content"])
        return article_data
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error scraping article {url}: {str(e)}")
        return {
            'url': url,
            'extraction_successful': False,
            'error': str(e)
        }
    finally:
        await page.close()

def scrape_article_sync(url: str) -> dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url)
            page.wait_for_load_state("networkidle")

            title = page.title()
            content = page.content()

            return {
                'url': url,
                'title': title,
                'content': content,
                'extraction_successful': True
            }
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Error scraping article {url}: {str(e)}")
            return {
                'url': url,
                'extraction_successful': False,
                'error': str(e)
            }
        finally:
            browser.close()

def should_scrape_url(url: str) -> bool:
    """Deprecated: use FilterChain externally where possible.

    Kept for backward compatibility and implemented via FilterChain
    using include/exclude substring patterns and content type check.
    """
    exclude_patterns = [
        '/tag/', '/category/', '/author/', '/search/', '/page/',
        'wp-content', 'wp-includes', 'wp-json', 'wp-admin',
        'login', 'register', 'cart', 'checkout', 'account',
        '.jpg', '.png', '.gif', '.pdf', '.zip'
    ]
    include_patterns = ['/article/', '/post/', '/blog/']
    chain = FilterChain([
        ContentTypeFilter(),
        URLPatternFilter(include_patterns=include_patterns, exclude_patterns=exclude_patterns)
    ])
    return chain.apply(url)

async def scrape_with_retry(url: str, max_retries: int = 3, retry_delay: float = 5.0):
    for attempt in range(max_retries):
        try:
            return await scrape_article(url)
        except TimeoutError:
            if attempt < max_retries - 1:
                logging.warning(f"Timeout error scraping {url}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error(f"Failed to scrape {url} after {max_retries} attempts.")
                return None
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return None

def convert_json_to_markdown(json_str: str) -> str:
    """
    Converts the JSON output from the scraping process into a markdown format.

    Args:
        json_str (str): JSON-formatted string containing the website collection data

    Returns:
        str: Markdown-formatted string of the website collection data
    """
    try:
        # Parse the JSON string
        data = json.loads(json_str)

        # Check if there's an error in the JSON
        if "error" in data:
            return f"# Error\n\n{data['error']}"

        # Start building the markdown string
        markdown = f"# Website Collection: {data['base_url']}\n\n"

        # Add metadata
        markdown += "## Metadata\n\n"
        markdown += f"- **Scrape Method:** {data['scrape_method']}\n"
        markdown += f"- **API Used:** {data['api_used']}\n"
        markdown += f"- **Keywords:** {data['keywords']}\n"
        if data.get('url_level') is not None:
            markdown += f"- **URL Level:** {data['url_level']}\n"
        if data.get('max_pages') is not None:
            markdown += f"- **Maximum Pages:** {data['max_pages']}\n"
        if data.get('max_depth') is not None:
            markdown += f"- **Maximum Depth:** {data['max_depth']}\n"
        markdown += f"- **Total Articles Scraped:** {data['total_articles_scraped']}\n\n"

        # Add URLs Scraped
        markdown += "## URLs Scraped\n\n"
        for url in data['urls_scraped']:
            markdown += f"- {url}\n"
        markdown += "\n"

        # Add the content
        markdown += "## Content\n\n"
        markdown += data['content']

        return markdown

    except json.JSONDecodeError:
        return "# Error\n\nInvalid JSON string provided."
    except KeyError as e:
        return f"# Error\n\nMissing key in JSON data: {str(e)}"
    except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as e:
        return f"# Error\n\nAn unexpected error occurred: {str(e)}"

#
# End of Scraping functions
##################################################################

#
# End of Article_Extractor_Lib.py
#######################################################################################################################
