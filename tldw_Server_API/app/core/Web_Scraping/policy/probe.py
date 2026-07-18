"""Narrow concrete egress policy adapter for fresh probe dispatches."""

from __future__ import annotations

import asyncio
import ipaddress
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from tldw_Server_API.app.core.Security import egress as egress_policy
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressDecision
from tldw_Server_API.app.core.Web_Scraping.runtime.requests import RuntimeRequestContext

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
_SOURCE_LABEL_MAP = MappingProxyType(
    {
        label: label
        for label in (
            "article_extract",
            "characterization",
            "enhanced_scrape",
            "preflight",
            "test",
            "web_scraping",
        )
    }
)
_STAGE_LABEL_MAP = MappingProxyType(
    {
        label: label
        for label in (
            "fetch",
            "pre_fetch",
            "preflight",
            "preflight_subrequest",
            "runtime",
        )
    }
)
_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _bounded_reason(raw: Any, *, allowed: bool) -> str:
    if allowed:
        return "allowed"
    reason_code = getattr(raw, "reason_code", None)
    if reason_code is not None:
        if type(reason_code) is str and reason_code in _ALLOWED_REASON_CODES:
            return reason_code
        return "other"
    reason = getattr(raw, "reason", None)
    if type(reason) is not str:
        return "other"
    return _LEGACY_REASON_MAP.get(reason, "other")


def _bounded_context_label(
    value: str,
    *,
    labels: MappingProxyType[str, str],
    fallback: str,
) -> str:
    if type(value) is not str:
        return fallback
    return labels.get(value, fallback)


def _is_canonical_idna_label(label: str) -> bool:
    if not label.startswith("xn--"):
        return True
    try:
        decoded = label.encode("ascii").decode("idna")
        canonical = decoded.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    return canonical == label


def _is_valid_dns_name(host: str) -> bool:
    if len(host) > 253 or host.replace(".", "").isdigit():
        return False
    labels = host.split(".")
    return all(
        1 <= len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isascii() and (character.isalnum() or character == "-") for character in label)
        and _is_canonical_idna_label(label)
        for label in labels
    )


def _sanitized_host_label(url: str) -> str:
    if type(url) is not str or "\\" in url or any(ord(character) < 32 or ord(character) == 127 for character in url):
        return "unknown"
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
            return "unknown"
        if any(ord(character) <= 32 or ord(character) == 127 for character in parsed.netloc):
            return "unknown"
        _port = parsed.port
        host = parsed.hostname
    except (AttributeError, TypeError, UnicodeError, ValueError):
        return "unknown"
    if not host or "%" in host:
        return "unknown"
    if host.endswith("."):
        host = host[:-1]
    if not host or host.endswith("."):
        return "unknown"
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        if ":" in host:
            return "unknown"
    try:
        canonical_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return "unknown"
    if not _is_valid_dns_name(canonical_host):
        return "unknown"
    return canonical_host


def _log_probe_policy_event(
    message: str,
    *,
    url: str,
    context: RuntimeRequestContext,
) -> None:
    logger.bind(
        source=_bounded_context_label(
            context.source,
            labels=_SOURCE_LABEL_MAP,
            fallback="web_scraping",
        ),
        stage=_bounded_context_label(
            context.stage,
            labels=_STAGE_LABEL_MAP,
            fallback="runtime",
        ),
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
