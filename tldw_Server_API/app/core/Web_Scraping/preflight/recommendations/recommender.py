from __future__ import annotations

from typing import Any


def _extract_waf_name(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("name", "")).strip()
    if isinstance(entry, (list, tuple)) and entry:
        return str(entry[0]).strip()
    if isinstance(entry, str):
        return entry.strip()
    return ""


def generate_recommendations(results: dict[str, Any]) -> dict[str, list[str]]:
    """
    Generate a list of recommended tools and strategy tips based on analyzer results.
    """
    tools: set[str] = set()
    strategy: set[str] = set()

    if results.get("js", {}).get("js_required"):
        tools.add("A headless browser such as Playwright or Selenium for JavaScript rendering.")
        strategy.add("Wait for dynamic content to load before extracting data.")

    tls_status = results.get("tls", {}).get("status")
    if tls_status == "active":
        tools.add("A library with browser impersonation (e.g. curl_cffi) or a full headless browser.")
        strategy.add("Standard Python HTTP clients are blocked; impersonate a real browser.")

    captcha = results.get("captcha", {})
    if captcha.get("captcha_detected"):
        tools.add("A CAPTCHA solving service (e.g. 2Captcha, Anti-Captcha).")
        strategy.add("Integrate the CAPTCHA solver when challenges appear.")

    behavioral = results.get("behavioral", {})
    if behavioral.get("honeypot_detected"):
        strategy.add("Avoid interacting with invisible elements; drive the page like a human.")

    rate_limit_results = results.get("rate_limit", {}).get("results", {})
    if rate_limit_results.get("blocking_code"):
        tools.add("A pool of high-quality rotating proxies (residential or mobile).")
        strategy.add("Add delays between requests (3-5 seconds) and rotate request headers.")

    wafs_list = results.get("waf", {}).get("wafs", [])
    for waf_name in map(_extract_waf_name, wafs_list):
        waf_name_lower = waf_name.lower()
        if any(keyword in waf_name_lower for keyword in ("cloudflare", "datadome", "perimeterx")):
            tools.add("A pool of high-quality rotating proxies (residential or mobile).")
            strategy.add("Use a modern, non-generic User-Agent and align headers with real browsers.")
            break

    fingerprint = results.get("fingerprint", {})
    if fingerprint.get("status") == "success":
        detected_services = fingerprint.get("detected_services", []) or []
        behavioral_listeners = fingerprint.get("behavioral_listeners_detected", []) or []
        canvas_signal = fingerprint.get("canvas_fingerprinting_signal", False)

        if detected_services:
            services_str = ", ".join(detected_services)
            strategy.add(
                f"Site uses advanced bot detection ({services_str}). Use playwright-stealth or undetected-chromedriver."
            )
            tools.add(
                "An anti-detection browser automation library (e.g. playwright-stealth, undetected-chromedriver)."
            )

        if behavioral_listeners:
            strategy.add("Site monitors user behavior (mouse, keyboard, scroll). Simulate realistic interaction.")
            strategy.add("Add random delays and jitter between actions to appear more human.")

        if canvas_signal:
            strategy.add("Canvas fingerprinting detected. Use automation with built-in evasion (not basic requests).")

    integrity = results.get("integrity", {})
    if integrity.get("status") == "success":
        modified_functions = integrity.get("modified_functions", {}) or {}
        if modified_functions:
            has_canvas_mods = any("Canvas" in func for func in modified_functions)
            has_timing_mods = any(("Date.now" in func or "performance.now" in func) for func in modified_functions)

            if has_canvas_mods:
                strategy.add("Site modifies canvas functions (strong fingerprinting). Avoid basic automation.")
            if has_timing_mods:
                strategy.add("Site monitors timing patterns. Vary your request timing to look less robotic.")
            if not (has_canvas_mods or has_timing_mods):
                strategy.add("Site modifies browser functions. Use advanced evasion techniques and test thoroughly.")

    if not tools:
        tools.add("Standard HTTP clients (requests, aiohttp) should be sufficient.")

    if not strategy:
        strategy.add("A simple, direct scraping approach is likely to work.")

    return {"tools": sorted(tools), "strategy": sorted(strategy)}
