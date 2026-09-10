"""Governed browser function-integrity analyzer and historical public wrapper."""

from __future__ import annotations

from typing import Any

from ..compatibility import _run_sync_compat
from ..context import PreflightExecutionContext
from ..facade import run_legacy_analyzer
from ..probes import BrowserProbeOptions, BrowserProbePage
from ._shared import _safe_analyzer_call

FUNCTIONS_TO_CHECK: list[str] = [
    "HTMLCanvasElement.prototype.toDataURL",
    "HTMLCanvasElement.prototype.getImageData",
    "HTMLCanvasElement.prototype.getContext",
    "navigator.plugins.length",
    "navigator.mimeTypes.length",
    "navigator.webdriver",
    "window.fetch",
    "XMLHttpRequest.prototype.open",
    "Date.now",
    "performance.now",
    "console.log",
]

FUNCTION_SUSPICION_MAP: dict[str, str] = {
    "HTMLCanvasElement.prototype.toDataURL": "Strong indicator of Canvas fingerprinting.",
    "HTMLCanvasElement.prototype.getImageData": "Strong indicator of Canvas fingerprinting.",
    "HTMLCanvasElement.prototype.getContext": "Strong indicator of Canvas fingerprinting.",
    "navigator.plugins.length": "Indicator of headless browser evasion (plugin spoofing).",
    "navigator.mimeTypes.length": "Indicator of headless browser evasion (mime type spoofing).",
    "navigator.webdriver": "Indicator of headless browser evasion.",
    "window.fetch": "Indicator of network traffic monitoring.",
    "XMLHttpRequest.prototype.open": "Indicator of network traffic monitoring.",
    "Date.now": "Indicator of timing/behavioral analysis.",
    "performance.now": "Indicator of timing/behavioral analysis.",
    "console.log": "Indicator of anti-debugging techniques.",
}

_FUNCTION_SIGNATURE_SCRIPT = """
(func_paths) => {
    const signatures = {};
    for (const path of func_paths) {
        try {
            let obj = window;
            const parts = path.split('.');
            for (let i = 0; i < parts.length; i++) {
                if (obj === undefined || obj === null) {
                    break;
                }
                obj = obj[parts[i]];
            }
            signatures[path] = String(obj);
        } catch (err) {
            signatures[path] = 'Error: ' + err.message;
        }
    }
    return signatures;
}
"""

_TIMEOUT = {
    "status": "error",
    "message": "Page load timed out.",
    "modified_functions": {},
}
_ANALYZER_ERROR = {
    "status": "error",
    "message": "Function integrity analysis failed.",
    "error_code": "analyzer_error",
    "modified_functions": {},
}


async def _get_function_signatures(
    page: BrowserProbePage,
    functions: list[str],
) -> dict[str, str]:
    """Get browser function string representations from one governed page."""
    return await page.evaluate(_FUNCTION_SIGNATURE_SCRIPT, functions)


async def _analyze_function_integrity_impl(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    options = BrowserProbeOptions(block_resource_types=("image", "font", "media"))

    async with context.browser.open_page(options) as clean_page:
        await clean_page.goto(
            "about:blank",
            wait_until="load",
            timeout_ms=30_000,
        )
        clean_signatures = await _get_function_signatures(
            clean_page,
            FUNCTIONS_TO_CHECK,
        )

    async with context.browser.open_page(options) as target_page:
        await target_page.goto(url, wait_until="load", timeout_ms=30_000)
        target_signatures = await _get_function_signatures(
            target_page,
            FUNCTIONS_TO_CHECK,
        )

    modified: dict[str, str] = {}
    for function_path, clean_signature in clean_signatures.items():
        if clean_signature != target_signatures.get(function_path):
            modified[function_path] = FUNCTION_SUSPICION_MAP.get(
                function_path,
                "Unknown modification.",
            )

    return {
        "status": "success",
        "message": "Analysis complete.",
        "modified_functions": modified,
    }


async def _analyze_function_integrity(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    result = await _safe_analyzer_call(_analyze_function_integrity_impl(url, context))
    if result.get("error_code") == "timeout":
        return dict(_TIMEOUT)
    if result.get("error_code") == "analyzer_error":
        return dict(_ANALYZER_ERROR)
    return result


def analyze_function_integrity(url: str) -> dict[str, Any]:
    """Run function integrity analysis through the governed legacy boundary."""
    return _run_sync_compat(run_legacy_analyzer(url, _analyze_function_integrity))


__all__ = [
    "FUNCTIONS_TO_CHECK",
    "FUNCTION_SUSPICION_MAP",
    "analyze_function_integrity",
]
