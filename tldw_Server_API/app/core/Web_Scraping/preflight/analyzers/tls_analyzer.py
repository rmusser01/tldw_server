"""Governed TLS fingerprint analyzer and historical public wrapper."""

from __future__ import annotations

from typing import Any

from ..context import PreflightExecutionContext
from ..facade import run_legacy_analyzer
from ..probes import ProbeHttpRequest, ProbeUnavailable
from ..utils.impersonate_target import get_impersonate_target
from ._shared import _safe_analyzer_call

_MISSING_DEPENDENCY = {
    "status": "error",
    "message": "curl-cffi is required for TLS impersonation.",
    "error_code": "missing_dependency",
}


async def _analyze_tls_fingerprint_impl(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    identity = context.browser_identity()
    standard_response = await context.http.get(
        ProbeHttpRequest(
            url=url,
            headers=identity,
            timeout_s=20,
            allow_redirects=True,
        )
    )

    impersonate_target = get_impersonate_target(identity.get("User-Agent", ""))
    try:
        browser_response = await context.http.get(
            ProbeHttpRequest(
                url=url,
                headers=identity,
                timeout_s=20,
                impersonate=impersonate_target,
                allow_redirects=True,
            )
        )
    except ProbeUnavailable as exc:
        if exc.error_code == "missing_dependency":
            return dict(_MISSING_DEPENDENCY)
        raise

    python_blocked = standard_response.status >= 400
    browser_blocked = browser_response.status >= 400
    if python_blocked and not browser_blocked:
        return {
            "status": "active",
            "details": "Site blocks standard Python clients but allows browser-like clients.",
        }
    if not python_blocked and not browser_blocked:
        return {
            "status": "inactive",
            "details": "Site does not appear to block based on TLS fingerprint.",
        }
    return {
        "status": "inconclusive",
        "details": "Could not determine fingerprinting status; site may be blocking all requests.",
    }


async def _analyze_tls_fingerprint(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    return await _safe_analyzer_call(_analyze_tls_fingerprint_impl(url, context))


async def analyze_tls_fingerprint(url: str) -> dict[str, Any]:
    """Run the TLS analyzer through the governed legacy boundary."""
    return await run_legacy_analyzer(url, _analyze_tls_fingerprint)


__all__ = ["analyze_tls_fingerprint"]
