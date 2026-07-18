"""Concrete outbound policy adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from tldw_Server_API.app.core.Security import egress as egress_policy
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import decide_web_outbound_policy
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    PolicyDecision,
    ProbeEgressDecision,
    RuntimeRequestContext,
)

_ALLOWED_REASON_CODES = frozenset(
    {
        "address_forbidden",
        "dns_changed",
        "dns_unresolved",
        "host_denied",
        "invalid_url",
        "origin_mismatch",
        "port_not_allowed",
        "unsupported_scheme",
        "userinfo_not_allowed",
    }
)
_LEGACY_REASON_MAP = MappingProxyType(
    {
        "DNS resolution changed since policy check": "dns_changed",
        "Host could not be resolved": "dns_unresolved",
        "Host in denylist": "host_denied",
        "Host not in allowlist": "host_denied",
        "Invalid URL": "invalid_url",
        "Invalid URL port": "invalid_url",
        "No allowlist configured (strict)": "host_denied",
        "Unsupported URL scheme": "unsupported_scheme",
        "URL must include a hostname": "invalid_url",
        "URL origin does not match configured endpoint": "origin_mismatch",
        "URL resolves to a forbidden address": "address_forbidden",
        "URL resolves to a private or reserved address": "address_forbidden",
        "URL userinfo is not allowed": "userinfo_not_allowed",
    }
)


def _bounded_reason(raw: Any, *, allowed: bool) -> str:
    if allowed:
        return "allowed"
    reason_code = getattr(raw, "reason_code", None)
    if reason_code is not None:
        if isinstance(reason_code, str) and reason_code in _ALLOWED_REASON_CODES:
            return reason_code
        return "other"
    reason = getattr(raw, "reason", None)
    if not isinstance(reason, str):
        return "other"
    return _LEGACY_REASON_MAP.get(reason, "other")


def _sanitized_context_label(value: str, *, fallback: str) -> str:
    raw = str(value or fallback)
    if any(marker in raw for marker in ("://", "/", "\\", "?", "#", "@")):
        return fallback
    label = "".join(
        character for character in raw if character.isascii() and (character.isalnum() or character in "_.-")
    )[:64]
    return label or fallback


def _sanitized_host_label(url: str) -> str:
    try:
        host = urlsplit(url).hostname or ""
        host = host.encode("idna").decode("ascii").lower().rstrip(".")
    except (AttributeError, TypeError, UnicodeError, ValueError):
        return "unknown"
    label = "".join(
        character for character in host if character.isascii() and (character.isalnum() or character in ".:-")
    )[:253]
    return label or "unknown"


def _log_probe_policy_event(
    message: str,
    *,
    url: str,
    context: RuntimeRequestContext,
) -> None:
    logger.bind(
        source=_sanitized_context_label(context.source, fallback="web_scraping"),
        stage=_sanitized_context_label(context.stage, fallback="runtime"),
        host=_sanitized_host_label(url),
    ).warning(message)


class DefaultProbeEgressGuard:
    """Delegate each probe dispatch decision to the central egress policy."""

    async def decide(
        self,
        url: str,
        *,
        context: RuntimeRequestContext,
    ) -> ProbeEgressDecision:
        try:
            raw = await asyncio.to_thread(egress_policy.evaluate_url_policy, url)
            allowed = bool(raw.allowed)
            decision = ProbeEgressDecision(
                allowed=allowed,
                reason=_bounded_reason(raw, allowed=allowed),
                resolved_ips=tuple(raw.resolved_ips or ()),
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _log_probe_policy_event(
                "Probe egress policy evaluation failed",
                url=url,
                context=context,
            )
            return ProbeEgressDecision(allowed=False, reason="policy_error")
        if not decision.allowed:
            _log_probe_policy_event(
                "Probe egress policy denied target",
                url=url,
                context=context,
            )
        return decision


class DefaultWebOutboundPolicyChecker:
    async def decide(
        self,
        url: str,
        *,
        respect_robots: bool,
        user_agent: str | None,
        context: RuntimeRequestContext,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        raw = await decide_web_outbound_policy(
            url,
            respect_robots=respect_robots,
            user_agent=user_agent,
            source=context.source,
            stage=context.stage,
            config=dict(config or {}),
        )
        return PolicyDecision(
            allowed=bool(raw.allowed),
            mode=str(raw.mode),
            reason=str(raw.reason),
            stage=str(raw.stage),
            source=str(raw.source),
            details=getattr(raw, "details", None),
        )
