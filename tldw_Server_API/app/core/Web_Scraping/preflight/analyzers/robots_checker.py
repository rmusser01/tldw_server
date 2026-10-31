"""Governed robots.txt analyzer and historical public wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..compatibility import _run_sync_compat
from ..context import PreflightExecutionContext
from ..facade import run_legacy_analyzer
from ..probes import ProbeHttpRequest
from ._shared import _safe_analyzer_call

_ROBOTS_USER_AGENT = "Mozilla/5.0 (compatible; caniscrape-bot/1.0)"
_ANALYZER_ERROR = {
    "status": "error",
    "message": "Robots.txt check failed.",
    "error_code": "analyzer_error",
}


def _header_value(headers: Mapping[str, str], name: str) -> str:
    normalized_name = name.lower()
    for header_name, value in headers.items():
        if header_name.lower() == normalized_name:
            return value
    return ""


async def _check_robots_txt_impl(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    parsed = urlsplit(url)
    robots_url = urlunsplit((parsed.scheme, parsed.netloc, "/robots.txt", "", ""))
    response = await context.http.get(
        ProbeHttpRequest(
            url=robots_url,
            headers={"User-Agent": _ROBOTS_USER_AGENT},
            timeout_s=10,
            allow_redirects=True,
        )
    )

    if response.status == 200:
        if "text/html" in _header_value(response.headers, "content-type").lower():
            return {"status": "not_found"}

        crawl_delay: float | None = None
        scraping_disallowed = False
        is_generic_agent_block = False
        for raw_line in response.text.splitlines():
            line = raw_line.strip().lower()
            if not line or line.startswith("#"):
                continue
            if line.startswith("user-agent:"):
                is_generic_agent_block = line.split(":", 1)[1].strip() == "*"
                continue
            if not is_generic_agent_block:
                continue
            if line.startswith("disallow:"):
                if line.split(":", 1)[1].strip() == "/":
                    scraping_disallowed = True
            elif line.startswith("crawl-delay:"):
                try:
                    crawl_delay = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    continue

        return {
            "status": "success",
            "crawl_delay": crawl_delay,
            "scraping_disallowed": scraping_disallowed,
        }

    if 400 <= response.status < 500:
        return {"status": "not_found"}
    return {"status": "error", "message": str(response.status)}


async def _check_robots_txt(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    result = await _safe_analyzer_call(_check_robots_txt_impl(url, context))
    return dict(_ANALYZER_ERROR) if result.get("error_code") == "analyzer_error" else result


def check_robots_txt(url: str) -> dict[str, Any]:
    """Run the robots.txt analyzer through the governed legacy boundary."""
    return _run_sync_compat(run_legacy_analyzer(url, _check_robots_txt))


__all__ = ["check_robots_txt"]
