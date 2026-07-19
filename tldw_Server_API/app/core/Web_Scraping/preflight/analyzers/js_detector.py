"""Governed JavaScript-rendering analyzer and historical public wrapper."""

from __future__ import annotations

import asyncio
from typing import Any

from bs4 import BeautifulSoup

from ..compatibility import _run_sync_compat
from ..context import PreflightDeadlineExceeded, PreflightExecutionContext
from ..facade import run_legacy_analyzer
from ..probes import (
    BrowserProbeOptions,
    ProbeHttpRequest,
    ProbeUnavailable,
)
from ..utils.impersonate_target import get_impersonate_target
from ._shared import _safe_analyzer_call

_TIMEOUT = {"status": "error", "message": "Page load timed out.", "error_code": "timeout"}
_ANALYZER_ERROR = {
    "status": "error",
    "message": "JavaScript rendering analysis failed.",
}


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


async def _navigate_best_effort(page: Any, url: str) -> None:
    try:
        await page.goto(url, wait_until="load", timeout_ms=30_000)
        await page.wait_for_load_state("networkidle", timeout_ms=5_000)
    except asyncio.CancelledError:
        raise
    except (PreflightDeadlineExceeded, ProbeUnavailable):
        raise
    except Exception:  # noqa: BLE001 - preserve partial-page legacy heuristic
        return


async def _analyze_js_rendering_impl(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    identity = context.browser_identity()
    response = await context.http.get(
        ProbeHttpRequest(
            url=url,
            headers=identity,
            timeout_s=30,
            impersonate=get_impersonate_target(identity.get("User-Agent", "")),
            allow_redirects=True,
        )
    )
    if response.status >= 400:
        raise RuntimeError("HTTP probe returned an unsuccessful status")
    len_no_js = len(_extract_visible_text(response.text))

    async with context.browser.open_page(BrowserProbeOptions(extra_headers=identity)) as page:
        await _navigate_best_effort(page, url)
        await page.wait_for_timeout(2_000)
        js_html = await page.content()

    len_js = len(_extract_visible_text(js_html))
    if len_js == 0:
        return {
            "status": "error",
            "message": "Could not extract content from the page with JS enabled.",
        }

    difference_percentage = max(0.0, round((1 - (len_no_js / len_js)) * 100, 2))
    return {
        "status": "success",
        "js_required": difference_percentage > 25,
        "is_spa": difference_percentage > 75,
        "content_difference_%": difference_percentage,
    }


async def _analyze_js_rendering(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    result = await _safe_analyzer_call(_analyze_js_rendering_impl(url, context))
    if result.get("error_code") == "timeout":
        return dict(_TIMEOUT)
    if result.get("error_code") == "analyzer_error":
        return dict(_ANALYZER_ERROR)
    return result


def analyze_js_rendering(url: str) -> dict[str, Any]:
    """Run JavaScript rendering analysis through the governed legacy boundary."""
    return _run_sync_compat(run_legacy_analyzer(url, _analyze_js_rendering))


__all__ = ["analyze_js_rendering"]
