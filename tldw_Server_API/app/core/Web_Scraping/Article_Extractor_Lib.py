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
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd

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
from tldw_Server_API.app.core.Web_Scraping.extraction import (  # noqa: F401
    DEFAULT_EXTRACTION_STRATEGY_ORDER,
    clear_extraction_caches,
    extract_article_data_from_html,
    extract_article_with_pipeline,
    extract_cluster_entities,
    extract_jsonld_entities,
    extract_llm_entities,
    extract_regex_entities,
    generate_regex_pattern_from_llm,  # noqa: F401
    generate_schema_rules_from_llm,  # noqa: F401
)
from tldw_Server_API.app.core.Web_Scraping.extraction import (
    get_extraction_cache_stats as _get_extraction_cache_stats,
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
from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScraperRouter
from tldw_Server_API.app.core.Web_Scraping.selectors import (
    clear_selector_caches as _clear_selector_caches,
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


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _extractor_max_workers() -> Optional[int]:
    value = _env_int("EXTRACTOR_MAX_WORKERS")
    if value is None or value <= 0:
        return None
    return value


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
