from __future__ import annotations

import math
import random
from typing import Any, Optional

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional dependency guard
    PlaywrightTimeoutError = TimeoutError  # type: ignore[misc,assignment]
    sync_playwright = None

from ..utils.browser_identities import MODERN_BROWSER_IDENTITIES

HONEYPOT_THRESHOLD = 3

ScanDepth = Optional[str]


def _choose_identity() -> dict[str, str]:
    return random.choice(MODERN_BROWSER_IDENTITIES)


def detect_honeypots(url: str, scan_depth: ScanDepth = "default") -> dict[str, Any]:
    """
    Launch a headless browser and count invisible links that may act as honeypots.

    Returns a status payload describing total links, invisible links, and whether
    the invisible count breached ``HONEYPOT_THRESHOLD``.
    """
    if sync_playwright is None:
        return {
            "status": "error",
            "message": "playwright is not installed; install the 'scrape-analyzers[browser]' extra.",
            "error_code": "missing_dependency",
        }

    browser_identity = _choose_identity()
    scan_depth = scan_depth or "default"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(extra_http_headers=browser_identity)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                links_locator = page.locator("a")
                total_links = links_locator.count()

                if total_links == 0:
                    return {
                        "status": "success",
                        "total_links": 0,
                        "invisible_links": 0,
                        "honeypot_detected": False,
                        "links_checked": 0,
                    }

                if scan_depth == "thorough":
                    links_to_check = math.ceil(total_links * 0.66)
                elif scan_depth == "deep":
                    links_to_check = total_links
                else:
                    links_to_check = min(math.ceil(total_links * 0.33), 250)

                links_to_check = min(links_to_check, total_links)

                invisible_links_count = 0
                for index in range(links_to_check):
                    link = links_locator.nth(index)
                    if not link.is_visible():
                        invisible_links_count += 1

                honeypot_detected = invisible_links_count > HONEYPOT_THRESHOLD
                return {
                    "status": "success",
                    "total_links": total_links,
                    "invisible_links": invisible_links_count,
                    "honeypot_detected": honeypot_detected,
                    "links_checked": links_to_check,
                }
            finally:
                browser.close()
    except PlaywrightTimeoutError:
        return {"status": "error", "message": "Page load timed out.", "error_code": "timeout"}
    except Exception as exc:  # pragma: no cover - defensive catch
        return {"status": "error", "message": "Honeypot detection failed."}
