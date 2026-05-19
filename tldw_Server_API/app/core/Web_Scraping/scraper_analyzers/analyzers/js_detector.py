from __future__ import annotations

import random
from typing import Any

from bs4 import BeautifulSoup

try:
    from curl_cffi.requests import Session as CurlCffiSession
except ImportError:  # pragma: no cover - optional dependency guard
    CurlCffiSession = None

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional dependency guard
    PlaywrightTimeoutError = TimeoutError  # type: ignore[misc,assignment]
    sync_playwright = None

from ..utils.browser_identities import MODERN_BROWSER_IDENTITIES
from ..utils.impersonate_target import get_impersonate_target


def _extract_visible_text(html_content: str) -> str:
    """Parse HTML and extract the visible text."""
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
    return "\n".join(chunk for chunk in chunks if chunk)


def analyze_js_rendering(url: str) -> dict[str, Any]:
    """
    Determine if JavaScript is required to render the main content for the URL.
    """
    if sync_playwright is None:
        return {
            "status": "error",
            "message": "playwright is not installed; install the 'scrape-analyzers[browser]' extra.",
            "error_code": "missing_dependency",
        }

    if CurlCffiSession is None:
        return {
            "status": "error",
            "message": "curl-cffi is not installed; install the 'scrape-analyzers[browser]' extra.",
            "error_code": "missing_dependency",
        }

    browser_identity = random.choice(MODERN_BROWSER_IDENTITIES)
    user_agent = browser_identity.get("User-Agent", "")
    impersonate_target = get_impersonate_target(user_agent)

    try:
        with CurlCffiSession(impersonate=impersonate_target) as session:
            no_js_response = session.get(url, headers=browser_identity, timeout=30)
            no_js_response.raise_for_status()
            no_js_text = _extract_visible_text(no_js_response.text)
            len_no_js = len(no_js_text)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(extra_http_headers=browser_identity)
                try:
                    page.goto(url, wait_until="load", timeout=30_000)
                    page.wait_for_load_state("networkidle", timeout=5_000)
                except Exception as navigation_error:
                    _ = navigation_error  # continue JS detector with current page state
                page.wait_for_timeout(2_000)
                js_html = page.content()
            finally:
                browser.close()

        js_text = _extract_visible_text(js_html)
        len_js = len(js_text)

        if len_js == 0:
            return {
                "status": "error",
                "message": "Could not extract content from the page with JS enabled.",
            }

        difference_percentage = (1 - (len_no_js / len_js)) * 100 if len_js else 0
        difference_percentage = max(0.0, round(difference_percentage, 2))

        js_required = difference_percentage > 25
        is_single_page_app = difference_percentage > 75

        return {
            "status": "success",
            "js_required": js_required,
            "is_spa": is_single_page_app,
            "content_difference_%": difference_percentage,
        }
    except PlaywrightTimeoutError:
        return {"status": "error", "message": "Page load timed out.", "error_code": "timeout"}
    except Exception as exc:  # pragma: no cover - defensive catch
        return {"status": "error", "message": "JavaScript rendering analysis failed."}
