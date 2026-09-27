"""Governed WAF analyzer and historical public wrapper."""

from __future__ import annotations

from typing import Any

from ..compatibility import _run_sync_compat
from ..context import PreflightExecutionContext
from ..facade import run_legacy_analyzer
from ..probes import ProbeTimeout, ProbeUnavailable
from ..utils.waf_result_parser import parse_wafw00f_output
from ._shared import _safe_analyzer_call

_MISSING_DEPENDENCY = {
    "status": "error",
    "message": "wafw00f missing",
    "error_code": "missing_dependency",
}
_TIMEOUT = {"status": "error", "message": "timeout", "error_code": "timeout"}
_ANALYZER_ERROR = {
    "status": "error",
    "message": "Analyzer failed.",
    "error_code": "analyzer_error",
}


async def _detect_waf_impl(
    url: str,
    context: PreflightExecutionContext,
    find_all: bool,
    external_tools_enabled: bool | None,
) -> dict[str, Any]:
    try:
        result = await context.external_tools.run_waf(
            url,
            find_all=find_all,
            enabled=external_tools_enabled,
        )
    except ProbeUnavailable as exc:
        if exc.error_code == "missing_dependency":
            return dict(_MISSING_DEPENDENCY)
        raise
    except ProbeTimeout:
        return dict(_TIMEOUT)

    wafs_found = parse_wafw00f_output(result.stdout, result.stderr)
    if wafs_found:
        return {"status": "success", "wafs": wafs_found}
    if result.returncode != 0:
        return dict(_ANALYZER_ERROR)
    return {"status": "success", "wafs": []}


async def _detect_waf(
    url: str,
    context: PreflightExecutionContext,
    find_all: bool = False,
    external_tools_enabled: bool | None = None,
) -> dict[str, Any]:
    return await _safe_analyzer_call(_detect_waf_impl(url, context, find_all, external_tools_enabled))


def detect_waf(url: str, find_all: bool = False) -> dict[str, Any]:
    """Run WAF detection through the governed legacy boundary."""
    return _run_sync_compat(run_legacy_analyzer(url, _detect_waf, find_all=find_all))


__all__ = ["detect_waf"]
