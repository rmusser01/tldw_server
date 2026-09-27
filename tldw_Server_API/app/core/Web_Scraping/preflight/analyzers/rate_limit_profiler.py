"""Governed rate-limit profiler and historical public wrapper."""

from __future__ import annotations

import asyncio
from typing import Any

from ..context import PreflightExecutionContext
from ..facade import run_legacy_analyzer
from ..probes import ProbeHttpRequest, ProbeUnavailable
from ..utils.impersonate_target import get_impersonate_target
from ._shared import _safe_analyzer_call

GENTLE_PROBE_COUNT = 4
BURST_COUNT = 8
DEFAULT_DELAY = 3.0
BLOCKING_STATUS_CODES = {429, 403, 503, 401}

_MISSING_DEPENDENCY = {
    "status": "error",
    "message": "curl-cffi is not installed; install the 'scrape-analyzers[browser]' extra.",
    "error_code": "missing_dependency",
}
_ANALYZER_ERROR = {
    "status": "error",
    "message": "Rate limit profiling failed.",
    "error_code": "analyzer_error",
}


async def _request_status(
    url: str,
    context: PreflightExecutionContext,
    *,
    identity: dict[str, str],
    impersonate_target: str | None,
) -> int:
    response = await context.http.get(
        ProbeHttpRequest(
            url=url,
            headers=identity,
            timeout_s=15,
            impersonate=impersonate_target,
            allow_redirects=True,
        )
    )
    return response.status


async def _profile_rate_limits_impl(
    url: str,
    context: PreflightExecutionContext,
    crawl_delay: float | None,
    impersonate: bool,
) -> dict[str, Any]:
    delay = crawl_delay if crawl_delay is not None else DEFAULT_DELAY
    identity = context.browser_identity()
    impersonate_target = get_impersonate_target(identity.get("User-Agent", "")) if impersonate else None
    results: dict[str, Any] = {
        "requests_sent": 0,
        "blocking_code": None,
        "details": "",
    }

    try:
        for index in range(GENTLE_PROBE_COUNT):
            status = await _request_status(
                url,
                context,
                identity=identity,
                impersonate_target=impersonate_target,
            )
            results["requests_sent"] += 1
            if status in BLOCKING_STATUS_CODES:
                results["blocking_code"] = status
                results["details"] = f'Blocked after {results["requests_sent"]} requests with a {delay:.1f}s delay.'
                return {"status": "success", "results": results}
            if index < GENTLE_PROBE_COUNT - 1:
                await context.controls.sleep(delay)

        burst_tasks = [
            asyncio.create_task(
                _request_status(
                    url,
                    context,
                    identity=identity,
                    impersonate_target=impersonate_target,
                )
            )
            for _ in range(BURST_COUNT)
        ]
        try:
            burst_statuses = await asyncio.gather(*burst_tasks)
        except BaseException:
            for task in burst_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*burst_tasks, return_exceptions=True)
            raise
    except ProbeUnavailable as exc:
        if impersonate and exc.error_code == "missing_dependency":
            return dict(_MISSING_DEPENDENCY)
        raise

    results["requests_sent"] += len(burst_statuses)
    for status in burst_statuses:
        if status in BLOCKING_STATUS_CODES:
            results["blocking_code"] = status
            results["details"] = f"Blocked during a concurrent burst of {BURST_COUNT} requests."
            return {"status": "success", "results": results}

    results["details"] = f'No blocking detected after {results["requests_sent"]} requests.'
    return {"status": "success", "results": results}


async def _profile_rate_limits(
    url: str,
    context: PreflightExecutionContext,
    crawl_delay: float | None,
    impersonate: bool = False,
) -> dict[str, Any]:
    result = await _safe_analyzer_call(_profile_rate_limits_impl(url, context, crawl_delay, impersonate))
    return dict(_ANALYZER_ERROR) if result.get("error_code") == "analyzer_error" else result


async def profile_rate_limits(
    url: str,
    crawl_delay: float | None,
    impersonate: bool = False,
) -> dict[str, Any]:
    """Run the rate-limit profiler through the governed legacy boundary."""
    return await run_legacy_analyzer(
        url,
        _profile_rate_limits,
        crawl_delay,
        impersonate=impersonate,
    )


__all__ = [
    "GENTLE_PROBE_COUNT",
    "BURST_COUNT",
    "DEFAULT_DELAY",
    "BLOCKING_STATUS_CODES",
    "profile_rate_limits",
]
