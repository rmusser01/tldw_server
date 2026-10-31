from __future__ import annotations

from typing import Any


def _normalise_waf(entry: Any) -> tuple[str, str | None]:
    if isinstance(entry, dict):
        return str(entry.get("name", "")).strip(), str(entry.get("manufacturer")) if entry.get("manufacturer") else None
    if isinstance(entry, (list, tuple)):
        name = str(entry[0]) if entry else ""
        manufacturer = str(entry[1]) if len(entry) > 1 and entry[1] else None
        return name.strip(), manufacturer
    if isinstance(entry, str):
        return entry.strip(), None
    return "", None


def calculate_difficulty_score(results: dict[str, Any]) -> dict[str, Any]:
    """
    Calculate a difficulty score from 0-10 based on the collected analysis results.
    """
    score = 0

    captcha = results.get("captcha", {})
    if captcha.get("captcha_detected"):
        trigger = captcha.get("trigger_condition", "")
        score += 5 if trigger == "on page load" else 4

    wafs_found = results.get("waf", {}).get("wafs", [])
    waf_scores = []
    for entry in wafs_found:
        waf_name, _ = _normalise_waf(entry)
        if not waf_name:
            continue
        lower = waf_name.lower()
        if "datadome" in lower or "perimeterx" in lower:
            waf_scores.append(4)
        elif "akamai" in lower or "imperva" in lower:
            waf_scores.append(3)
        elif "cloudflare" in lower or "cloudfront" in lower:
            waf_scores.append(2)
    if waf_scores:
        score += max(waf_scores)

    rate_limit_results = results.get("rate_limit", {}).get("results", {})
    requests_sent = rate_limit_results.get("requests_sent", 0)
    if rate_limit_results.get("blocking_code") and 1 < requests_sent < 5:
        score += 3

    if results.get("behavioral", {}).get("honeypot_detected"):
        score += 2

    if results.get("tls", {}).get("status") == "active":
        score += 1

    fingerprint = results.get("fingerprint", {})
    if fingerprint.get("status") == "success":
        if fingerprint.get("detected_services"):
            score += 2
        if fingerprint.get("canvas_fingerprinting_signal"):
            score += 1
        if fingerprint.get("behavioral_listeners_detected"):
            score += 1

    integrity = results.get("integrity", {})
    if integrity.get("status") == "success":
        modified = integrity.get("modified_functions") or {}
        if modified:
            has_canvas = any("Canvas" in func for func in modified)
            has_timing = any("Date.now" in func or "performance.now" in func for func in modified)
            if has_canvas or has_timing:
                score += 2
            else:
                score += 1

    final_score = min(score, 10)

    if final_score >= 8:
        difficulty_label = "Very Hard"
    elif final_score >= 5:
        difficulty_label = "Hard"
    elif final_score >= 3:
        difficulty_label = "Medium"
    else:
        difficulty_label = "Easy"

    return {"score": final_score, "label": difficulty_label}
