"""Governed honeypot analyzer and historical public wrapper."""

from __future__ import annotations

import math
from typing import Any, Optional

from ..compatibility import _run_sync_compat
from ..context import PreflightExecutionContext
from ..facade import run_legacy_analyzer
from ..probes import BrowserProbeOptions
from ._shared import _safe_analyzer_call

HONEYPOT_THRESHOLD = 3
ScanDepth = Optional[str]

_TIMEOUT = {"status": "error", "message": "Page load timed out.", "error_code": "timeout"}
_ANALYZER_ERROR = {"status": "error", "message": "Honeypot detection failed."}


async def _detect_honeypots_impl(
    url: str,
    context: PreflightExecutionContext,
    scan_depth: ScanDepth,
) -> dict[str, Any]:
    identity = context.browser_identity()
    normalized_depth = scan_depth or "default"

    async with context.browser.open_page(BrowserProbeOptions(extra_headers=identity)) as page:
        await page.goto(url, wait_until="domcontentloaded", timeout_ms=30_000)
        total_links = await page.link_count()

        if total_links == 0:
            return {
                "status": "success",
                "total_links": 0,
                "invisible_links": 0,
                "honeypot_detected": False,
                "links_checked": 0,
            }

        if normalized_depth == "thorough":
            links_to_check = math.ceil(total_links * 0.66)
        elif normalized_depth == "deep":
            links_to_check = total_links
        else:
            links_to_check = min(math.ceil(total_links * 0.33), 250)
        links_to_check = min(links_to_check, total_links)

        invisible_links = 0
        for index in range(links_to_check):
            if not await page.link_is_visible(index):
                invisible_links += 1

    return {
        "status": "success",
        "total_links": total_links,
        "invisible_links": invisible_links,
        "honeypot_detected": invisible_links > HONEYPOT_THRESHOLD,
        "links_checked": links_to_check,
    }


async def _detect_honeypots(
    url: str,
    context: PreflightExecutionContext,
    scan_depth: ScanDepth = "default",
) -> dict[str, Any]:
    result = await _safe_analyzer_call(_detect_honeypots_impl(url, context, scan_depth))
    if result.get("error_code") == "timeout":
        return dict(_TIMEOUT)
    if result.get("error_code") == "analyzer_error":
        return dict(_ANALYZER_ERROR)
    return result


def detect_honeypots(
    url: str,
    scan_depth: ScanDepth = "default",
) -> dict[str, Any]:
    """Run honeypot detection through the governed legacy boundary."""
    return _run_sync_compat(run_legacy_analyzer(url, _detect_honeypots, scan_depth=scan_depth))


__all__ = ["HONEYPOT_THRESHOLD", "ScanDepth", "detect_honeypots"]
