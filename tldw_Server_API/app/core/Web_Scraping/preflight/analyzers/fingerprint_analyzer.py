"""Governed browser-fingerprinting analyzer and historical public wrapper."""

from __future__ import annotations

import json
from typing import Any

from ..compatibility import _run_sync_compat
from ..context import PreflightExecutionContext
from ..facade import run_legacy_analyzer
from ..probes import BrowserProbeOptions
from ._shared import _safe_analyzer_call

KNOWN_BOT_DETECTION_SCRIPTS: dict[str, list[str]] = {
    "PerimeterX (HUMAN)": [
        "client.perimeterx.net",
        "px-cdn.net",
        "collector-px.perimeterx.net",
    ],
    "DataDome": [
        "datadome.co/js",
        "api.datadome.co/js",
        "js.datadome.co",
    ],
    "Akamai Bot Manager": [
        "akam-bm.net",
        "ak-bm.net",
        "ds-aksb-a.akamaihd.net",
    ],
    "Cloudflare Bot Management": [
        "/cf-challenge/",
        "cdn-cgi/challenge-platform",
        "cf_bm",
    ],
    "Imperva (Incapsula)": [
        "incapsula.com",
        "/_Incapsula_Resource",
    ],
    "Kasada": [
        "api.kasada.io",
        "/kasada-api/",
    ],
    "Shape Security (F5)": [
        "shapeshifter.io",
        "shape-only.com",
        "/F5-shape-security-js",
    ],
    "CHEQ": [
        "cheqzone.com",
        "api.cheq.ai",
    ],
    "Radware Bot Manager": [
        "radwarebotmanager.com",
        "/rbm/rbm.js",
    ],
}

KNOWN_BOT_GLOBAL_OBJECTS: dict[str, list[str]] = {
    "PerimeterX (HUMAN)": ["_px", "PX", "px"],
    "DataDome": ["ddjskey", "datadome"],
    "Akamai Bot Manager": ["bmak"],
    "Imperva (Incapsula)": ["Reese84"],
    "Kasada": ["kasada"],
    "Shape Security (F5)": ["_sd"],
}

JS_PROBE_SCRIPT = """
() => {
    window.__caniscrape_listeners_log = [];
    const log = window.__caniscrape_listeners_log;

    const originalAddEventListener = EventTarget.prototype.addEventListener;
    const suspiciousEvents = ['mousemove', 'mousedown', 'mouseup', 'keydown', 'keyup', 'scroll', 'touchstart', 'touchend'];

    EventTarget.prototype.addEventListener = function(type, listener, options) {
        if (suspiciousEvents.includes(type)) {
            log.push(type);
        }
        return originalAddEventListener.call(this, type, listener, options);
    };
};
"""

_STATIC_PROBE_SCRIPT = f"""
() => {{
    const results = {{
        canvas_patched: HTMLCanvasElement.prototype.toDataURL.toString().indexOf('native code') === -1,
        found_globals: []
    }};

    const global_objects = {json.dumps(KNOWN_BOT_GLOBAL_OBJECTS)};

    for (const [service, objects] of Object.entries(global_objects)) {{
        for (const obj_name of objects) {{
            if (window[obj_name]) {{
                results.found_globals.push(service);
                break;
            }}
        }}
    }}
    return results;
}}
"""

_TIMEOUT = {
    "status": "error",
    "message": "Page load timed out.",
    "detected_services": [],
    "canvas_fingerprinting_signal": False,
    "behavioral_listeners_detected": [],
}
_ANALYZER_ERROR = {
    "status": "error",
    "message": "Fingerprint analysis failed.",
    "error_code": "analyzer_error",
    "detected_services": [],
    "canvas_fingerprinting_signal": False,
    "behavioral_listeners_detected": [],
}


async def _analyze_fingerprinting_impl(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    identity = context.browser_identity()
    options = BrowserProbeOptions(
        extra_headers=identity,
        block_resource_types=("image", "font", "media"),
        init_scripts=(JS_PROBE_SCRIPT,),
        capture_requests=True,
    )

    async with context.browser.open_page(options) as page:
        await page.goto(url, wait_until="load", timeout_ms=30_000)
        await page.wait_for_timeout(3_000)
        static_probes = await page.evaluate(_STATIC_PROBE_SCRIPT)
        listener_log = await page.evaluate("() => window.__caniscrape_listeners_log")
        captured_script_urls = set(page.captured_request_urls())

    detected_services: list[str] = []
    for service, patterns in KNOWN_BOT_DETECTION_SCRIPTS.items():
        if any(pattern in captured_url for pattern in patterns for captured_url in captured_script_urls):
            detected_services.append(service)

    for service in static_probes.get("found_globals", []) or []:
        if service not in detected_services:
            detected_services.append(service)

    return {
        "status": "success",
        "message": "Analysis complete.",
        "detected_services": detected_services,
        "canvas_fingerprinting_signal": bool(static_probes.get("canvas_patched")),
        "behavioral_listeners_detected": (sorted(set(listener_log)) if listener_log else []),
    }


async def _analyze_fingerprinting(
    url: str,
    context: PreflightExecutionContext,
) -> dict[str, Any]:
    result = await _safe_analyzer_call(_analyze_fingerprinting_impl(url, context))
    if result.get("error_code") == "timeout":
        return dict(_TIMEOUT)
    if result.get("error_code") == "analyzer_error":
        return dict(_ANALYZER_ERROR)
    return result


def analyze_fingerprinting(url: str) -> dict[str, Any]:
    """Run fingerprint analysis through the governed legacy boundary."""
    return _run_sync_compat(run_legacy_analyzer(url, _analyze_fingerprinting))


__all__ = [
    "JS_PROBE_SCRIPT",
    "KNOWN_BOT_DETECTION_SCRIPTS",
    "KNOWN_BOT_GLOBAL_OBJECTS",
    "analyze_fingerprinting",
]
