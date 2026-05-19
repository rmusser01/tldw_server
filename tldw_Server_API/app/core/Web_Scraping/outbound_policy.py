"""Shared outbound-policy decisions for scrape and websearch callers.

This module centralizes the ordering of egress checks and optional robots.txt
enforcement so that all data-plane callers share the same compat/strict
semantics. Compat mode preserves the historical fail-open robots behavior,
while strict mode fails closed when robots.txt cannot be retrieved or parsed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from loguru import logger

from tldw_Server_API.app.core.Metrics import increment_counter
from tldw_Server_API.app.core.Security import egress as egress_policy
from tldw_Server_API.app.core.Web_Scraping.filters import RobotsFilter
from tldw_Server_API.app.core.config import web_outbound_policy_mode


_Mode = Literal["compat", "strict"]


@dataclass(slots=True, frozen=True)
class WebOutboundPolicyDecision:
    """Normalized outbound-policy decision consumed by scrape/search call sites."""

    allowed: bool
    mode: _Mode
    reason: str
    stage: str
    source: str
    details: dict[str, Any] | None = None


def evaluate_url_policy(url: str) -> egress_policy.URLPolicyResult:
    """Proxy the centralized egress-policy evaluator for local monkeypatching."""
    return egress_policy.evaluate_url_policy(url)


def get_web_outbound_policy_mode(config: dict[str, Any] | None = None) -> _Mode:
    """Resolve the rollout mode from env/config with compat as the safe default."""
    raw_mode = os.getenv("WEB_OUTBOUND_POLICY_MODE")
    if raw_mode is None:
        if config is not None:
            raw_mode = str(
                ((config.get("web_scraper", {}) or {}).get("web_outbound_policy_mode", "compat"))
            )
        else:
            raw_mode = web_outbound_policy_mode()

    mode = str(raw_mode or "compat").strip().lower()
    return "strict" if mode == "strict" else "compat"


def _metric_reason_label(reason: str) -> str:
    """Map raw policy reasons to a bounded label set for metrics."""
    normalized = str(reason or "").strip()
    exact_mappings = {
        "allowed": "allowed",
        "robots_skipped": "robots_skipped",
        "robots_disallowed": "robots_disallowed",
        "robots_unreachable": "robots_unreachable",
        "robots_unreachable_allowed": "robots_unreachable_allowed",
        "robots_check_error_internal": "robots_check_error_internal",
        "robots_sync_unsupported": "robots_sync_unsupported",
        "Host in denylist": "host_in_denylist",
        "Host not in allowlist": "host_not_in_allowlist",
        "No allowlist configured (strict)": "strict_allowlist_missing",
        "URL resolves to a private or reserved address": "resolves_private_address",
        "Host could not be resolved": "host_unresolved",
        "Invalid URL": "invalid_url",
        "Unsupported URL scheme": "unsupported_url_scheme",
        "URL must include a hostname": "missing_hostname",
        "Invalid URL port": "invalid_url_port",
        "egress_denied": "egress_denied",
    }
    if normalized in exact_mappings:
        return exact_mappings[normalized]
    if normalized.startswith("Port not allowed:"):
        return "port_not_allowed"
    return "other"


def _record_decision_metric(decision: WebOutboundPolicyDecision) -> None:
    """Emit a bounded-cardinality metric for the shared outbound-policy helper."""
    increment_counter(
        "web_outbound_policy_decisions_total",
        labels={
            "mode": decision.mode,
            "source": decision.source,
            "stage": decision.stage,
            "outcome": "allowed" if decision.allowed else "blocked",
            "reason": _metric_reason_label(decision.reason),
        },
    )


def _decision(
    allowed: bool,
    *,
    mode: _Mode,
    reason: str,
    stage: str,
    source: str,
    details: dict[str, Any] | None = None,
) -> WebOutboundPolicyDecision:
    """Create a decision object and record the matching metric event."""
    decision = WebOutboundPolicyDecision(
        allowed=allowed,
        mode=mode,
        reason=reason,
        stage=stage,
        source=source,
        details=details,
    )
    _record_decision_metric(decision)
    return decision


def _egress_denial_reason(raw: egress_policy.URLPolicyResult) -> str:
    """Extract a stable human-readable reason from an egress policy result."""
    return str(getattr(raw, "reason", "egress_denied") or "egress_denied")


def _sanitize_url_for_logs(url: str) -> str:
    """Strip query fragments from a URL before attaching it to logs."""
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    return f"{host}{path}"


def decide_web_outbound_policy_sync(
    url: str,
    *,
    respect_robots: bool,
    source: str,
    stage: str,
    config: dict[str, Any] | None = None,
) -> WebOutboundPolicyDecision:
    """Evaluate outbound policy for synchronous callers without robots fetching."""
    mode = get_web_outbound_policy_mode(config)
    raw = evaluate_url_policy(url)
    if not getattr(raw, "allowed", False):
        return _decision(
            False,
            mode=mode,
            reason=_egress_denial_reason(raw),
            stage=stage,
            source=source,
        )

    if respect_robots:
        reason = "robots_sync_unsupported" if mode == "strict" else "robots_skipped"
        return _decision(mode != "strict", mode=mode, reason=reason, stage=stage, source=source)

    return _decision(True, mode=mode, reason="allowed", stage=stage, source=source)


async def decide_web_outbound_policy(
    url: str,
    *,
    respect_robots: bool,
    user_agent: str | None,
    source: str,
    stage: str,
    config: dict[str, Any] | None = None,
    robots_filter: RobotsFilter | None = None,
) -> WebOutboundPolicyDecision:
    """Evaluate egress plus optional robots policy for async scrape callers."""
    mode = get_web_outbound_policy_mode(config)
    raw = evaluate_url_policy(url)
    if not getattr(raw, "allowed", False):
        return _decision(
            False,
            mode=mode,
            reason=_egress_denial_reason(raw),
            stage=stage,
            source=source,
        )

    if not respect_robots:
        return _decision(True, mode=mode, reason="robots_skipped", stage=stage, source=source)

    robots_filter = robots_filter or RobotsFilter(user_agent=user_agent or "*")
    try:
        robots_result = await robots_filter.check(
            url,
            skip_egress_check=True,
            fail_open=(mode != "strict"),
        )
    except (ConnectionError, OSError, TimeoutError) as exc:  # pragma: no cover - defensive guard
        logger.bind(
            source=source,
            stage=stage,
            mode=mode,
            sanitized_url=_sanitize_url_for_logs(url),
            error_type=exc.__class__.__name__,
        ).debug("Web outbound policy robots check failed")
        if mode == "strict":
            return _decision(False, mode=mode, reason="robots_unreachable", stage=stage, source=source)
        return _decision(
            True,
            mode=mode,
            reason="robots_unreachable_allowed",
            stage=stage,
            source=source,
        )
    except Exception as exc:  # pragma: no cover - explicit behavior covered by monkeypatched tests
        logger.bind(
            source=source,
            stage=stage,
            mode=mode,
            sanitized_url=_sanitize_url_for_logs(url),
            error_type=exc.__class__.__name__,
        ).debug("Web outbound policy robots check failed")
        return _decision(
            False,
            mode=mode,
            reason="robots_check_error_internal",
            stage=stage,
            source=source,
        )

    if robots_result.status == "unreachable":
        reason = "robots_unreachable" if mode == "strict" else "robots_unreachable_allowed"
        return _decision(robots_result.allowed, mode=mode, reason=reason, stage=stage, source=source)

    if not robots_result.allowed:
        return _decision(False, mode=mode, reason="robots_disallowed", stage=stage, source=source)

    return _decision(True, mode=mode, reason="allowed", stage=stage, source=source)
