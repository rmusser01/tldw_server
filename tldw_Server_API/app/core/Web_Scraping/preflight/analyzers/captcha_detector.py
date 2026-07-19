"""Governed CAPTCHA analyzer and historical public wrapper."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from ..compatibility import _run_sync_compat
from ..context import PreflightDeadlineExceeded, PreflightExecutionContext
from ..facade import run_legacy_analyzer
from ..probes import BrowserProbeOptions, ProbeUnavailable
from ._shared import _safe_analyzer_call

CAPTCHA_FINGERPRINTS: dict[str, list[str]] = {
    "reCAPTCHA": ["google.com/recaptcha", "recaptcha/api.js", "g-recaptcha"],
    "hCaptcha": ["hcaptcha.com", "hcaptcha-box", "h-captcha"],
    "Cloudflare Turnstile": [
        "challenges.cloudflare.com/turnstile",
        "cf-turnstile",
    ],
}

_TIMEOUT = {"status": "error", "message": "Page load timed out.", "error_code": "timeout"}
_ANALYZER_ERROR = {"status": "error", "message": "Captcha detection failed."}


def _scan_for_captcha_fingerprints(
    html_content: str,
    network_requests: Sequence[str],
) -> str | None:
    """Return the first known CAPTCHA provider found in page evidence."""
    evidence = tuple(str(item).lower() for item in network_requests) + (str(html_content).lower(),)
    for provider, patterns in CAPTCHA_FINGERPRINTS.items():
        if any(pattern in item for pattern in patterns for item in evidence):
            return provider
    return None


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


async def _detect_captcha_impl(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    async with context.browser.open_page(BrowserProbeOptions(capture_requests=True)) as page:
        await _navigate_best_effort(page, url)
        await page.wait_for_timeout(2_000)

        provider = _scan_for_captcha_fingerprints(
            await page.content(),
            page.captured_request_urls(),
        )
        if provider:
            return {
                "status": "success",
                "captcha_detected": True,
                "captcha_type": provider,
                "trigger_condition": "on page load",
            }

        page.clear_captured_request_urls()
        for _ in range(10):
            await page.reload(wait_until="domcontentloaded", timeout_ms=30_000)

        provider = _scan_for_captcha_fingerprints(
            await page.content(),
            page.captured_request_urls(),
        )
        if provider:
            return {
                "status": "success",
                "captcha_detected": True,
                "captcha_type": provider,
                "trigger_condition": "after burst of requests",
            }

    return {"status": "success", "captcha_detected": False}


async def _detect_captcha(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    result = await _safe_analyzer_call(_detect_captcha_impl(url, context))
    if result.get("error_code") == "timeout":
        return dict(_TIMEOUT)
    if result.get("error_code") == "analyzer_error":
        return dict(_ANALYZER_ERROR)
    return result


def detect_captcha(url: str) -> dict[str, Any]:
    """Run CAPTCHA detection through the governed legacy boundary."""
    return _run_sync_compat(run_legacy_analyzer(url, _detect_captcha))


__all__ = ["CAPTCHA_FINGERPRINTS", "detect_captcha"]
