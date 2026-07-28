"""
Scraper Router

Maps URL -> ScrapePlan using per-domain rules with precedence:
 1) Exact domain match
 2) Wildcard domain (e.g., *.example.com)
 3) Regex url_patterns within matched domain rule

Security:
- Handler strings are validated against an allowlist of module prefixes to
  avoid arbitrary imports/code execution via YAML config.
- Supports a 'respect_robots' flag carried on the plan for fetchers to
  enforce using a robots.txt check at fetch time (not performed here to
  keep routing offline and testable without network).
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import yaml

from .safe_regex import SafeRegexLimits, search_untrusted
from .ua_profiles import pick_ua_profile, profile_to_impersonate

DEFAULT_HANDLER = "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html"

DEFAULT_HANDLER_ALLOWLIST = ("tldw_Server_API.app.core.Web_Scraping.handlers:",)

_ROUTER_REGEX_LIMITS = SafeRegexLimits()
# Configured patterns are user-controlled, so cap aggregate work per domain rule.
_MAX_URL_PATTERNS = 32
_URL_PATTERN_TOTAL_BUDGET_S = 0.100
_BACKEND_LOOKUP = {
    "auto": "auto",
    "curl": "curl",
    "httpx": "httpx",
    "playwright": "playwright",
}
_INVALID_VALUE = object()


def _monotonic() -> float:
    return time.monotonic()


def _router_regex_limits(timeout_s: float) -> SafeRegexLimits:
    return SafeRegexLimits(
        max_pattern_chars=_ROUTER_REGEX_LIMITS.max_pattern_chars,
        max_input_chars=_ROUTER_REGEX_LIMITS.max_input_chars,
        timeout_s=min(timeout_s, _ROUTER_REGEX_LIMITS.timeout_s),
    )


def _snapshot_mapping(value: Any) -> dict[Any, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return dict(value)
    except Exception:
        return None


def _snapshot_config_mapping(value: Any) -> dict[str, Any] | None:
    snapshot = _snapshot_mapping(value)
    if snapshot is None:
        return None
    return {key: item for key, item in snapshot.items() if type(key) is str}


def _normalize_backend(value: Any) -> str:
    if type(value) is not str:
        return "auto"
    normalized = value.strip().lower()
    return _BACKEND_LOOKUP.get(normalized, "auto")


def _normalize_scalar_string(
    value: Any,
    *,
    allow_none: bool = False,
) -> str | None | object:
    if value is None:
        return None if allow_none else _INVALID_VALUE
    if type(value) is str:
        return value
    if type(value) in {bool, int, float}:
        return str(value)
    return _INVALID_VALUE


def _normalize_bool(value: Any) -> bool | object:
    if type(value) in {bool, int, float, str}:
        return bool(value)
    return _INVALID_VALUE


def _stringify_safe_scalar(value: Any) -> str | object:
    if type(value) is str:
        return value
    if type(value) in {bool, int, float}:
        return str(value)
    return _INVALID_VALUE


def _normalize_string_mapping(value: Any) -> dict[str, str]:
    snapshot = _snapshot_mapping(value)
    if snapshot is None:
        return {}

    normalized: dict[str, str] = {}
    for key, item in snapshot.items():
        normalized_key = _stringify_safe_scalar(key)
        normalized_item = _stringify_safe_scalar(item)
        if normalized_key is not _INVALID_VALUE and normalized_item is not _INVALID_VALUE:
            normalized[normalized_key] = normalized_item
    return normalized


def _normalize_string_list(value: Any) -> list[str] | object:
    if type(value) is not list:
        return _INVALID_VALUE
    return [item for item in value if type(item) is str]


def _normalize_object_mapping(value: Any) -> dict[str, Any] | object:
    snapshot = _snapshot_config_mapping(value)
    if snapshot is None:
        return _INVALID_VALUE
    return snapshot


def _mapping_alias(
    rule: dict[str, Any],
    primary: str,
    alias: str,
) -> dict[str, Any] | None:
    for key in (primary, alias):
        normalized = _normalize_object_mapping(rule.get(key, _INVALID_VALUE))
        if normalized is not _INVALID_VALUE:
            return normalized
    return None


def _normalize_handler_allowlist(value: Any) -> tuple[str, ...]:
    if type(value) not in {list, tuple, set, frozenset}:
        return DEFAULT_HANDLER_ALLOWLIST

    prefixes = [item for item in value if type(item) is str and len(item) > 0]
    if type(value) in {set, frozenset}:
        prefixes.sort()
    if len(prefixes) == 0:
        return DEFAULT_HANDLER_ALLOWLIST
    return tuple(prefixes)


@dataclass
class ScrapePlan:
    url: str
    domain: str
    backend: str = "auto"  # auto|curl|httpx|playwright
    handler: str = DEFAULT_HANDLER
    ua_profile: str = "chrome_120_win"
    impersonate: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)
    respect_robots: bool = True
    proxies: dict[str, str] = field(default_factory=dict)  # e.g., {"http": ...}
    strategy_order: list[str] | None = None
    schema_rules: dict[str, Any] | None = None
    llm_settings: dict[str, Any] | None = None
    regex_settings: dict[str, Any] | None = None
    cluster_settings: dict[str, Any] | None = None


def _validate_handler(handler: Any, allowlist: Any) -> str:
    if type(handler) is not str or len(handler) == 0:
        return DEFAULT_HANDLER
    for prefix in _normalize_handler_allowlist(allowlist):
        if handler.startswith(prefix):
            return handler
    # Fallback to safe default
    return DEFAULT_HANDLER


def _parse_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _match_domain_rule(
    domain: str,
    rules: Any,
) -> tuple[str, dict[str, Any]] | None:
    # 1) Exact
    if type(domain) is not str:
        return None
    rules_snapshot = _snapshot_config_mapping(rules)
    if rules_snapshot is None:
        return None
    dom_rules = _snapshot_config_mapping(rules_snapshot.get("domains"))
    if dom_rules is None:
        return None
    for key, raw_rule in dom_rules.items():
        if type(key) is not str:
            continue
        if key == domain:
            rule = _snapshot_config_mapping(raw_rule)
            return (domain, rule) if rule is not None else None

    # 2) Wildcard (*.example.com)
    best_match: tuple[str, dict[str, Any]] | None = None
    best_suffix_len = -1
    for key, raw_rule in dom_rules.items():
        if type(key) is not str or not key.startswith("*."):
            continue
        suffix = key[1:]  # remove leading '*'
        if not domain.endswith(suffix) or len(suffix) <= best_suffix_len:
            continue
        rule = _snapshot_config_mapping(raw_rule)
        if rule is not None:
            best_match = (key, rule)
            best_suffix_len = len(suffix)

    if best_match:
        return best_match

    # 3) No domain-level match
    return None


class ScraperRouter:
    def __init__(
        self,
        rules: Any = None,
        *,
        handler_allowlist: Any = None,
        ua_mode: Any = "fixed",
        default_respect_robots: bool = True,
    ) -> None:
        rules_snapshot = _snapshot_config_mapping(rules)
        self.rules = rules_snapshot if rules_snapshot is not None else {}
        self.allowlist = _normalize_handler_allowlist(handler_allowlist)
        self.ua_mode = ua_mode if type(ua_mode) is str else "fixed"
        normalized_robots = _normalize_bool(default_respect_robots)
        self.default_respect_robots = normalized_robots if normalized_robots is not _INVALID_VALUE else True

    @staticmethod
    def load_rules_from_yaml(path: str) -> dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Basic normalization
        if not isinstance(data, dict):
            return {}
        data.setdefault("domains", {})
        return ScraperRouter.validate_rules(data)

    @staticmethod
    def validate_rules(data: Any) -> dict[str, Any]:
        """Validate and normalize rules loaded from YAML.

        - Ensure top-level 'domains' mapping
        - Keep only known keys per domain rule
        - Validate backend and url_patterns
        - Normalize headers/cookies to string maps
        """
        out: dict[str, Any] = {"domains": {}}
        data_snapshot = _snapshot_config_mapping(data)
        if data_snapshot is None:
            return out
        domains = _snapshot_config_mapping(data_snapshot.get("domains"))
        if domains is None:
            return out

        allowed_keys = {
            "backend",
            "handler",
            "ua_profile",
            "impersonate",
            "extra_headers",
            "url_patterns",
            "cookies",
            "respect_robots",
            "proxies",
            "strategy_order",
            "schema_rules",
            "schema",
            "llm_settings",
            "llm",
            "regex_settings",
            "regex",
            "cluster_settings",
            "cluster",
        }
        for dom, raw_rule in domains.items():
            if type(dom) is not str:
                continue
            # minimal domain/wildcard sanity: must contain a dot or start with '*.'
            if not (dom.startswith("*.") or "." in dom):
                continue
            rule = _snapshot_config_mapping(raw_rule)
            if rule is None:
                continue

            cleaned: dict[str, Any] = {}
            discard_rule = False
            for k, v in rule.items():
                if type(k) is not str or k not in allowed_keys:
                    continue
                if k == "backend":
                    cleaned[k] = _normalize_backend(v)
                elif k == "handler":
                    cleaned[k] = v if type(v) is str and len(v) > 0 else DEFAULT_HANDLER
                elif k == "ua_profile":
                    normalized = _normalize_scalar_string(v)
                    if normalized is not _INVALID_VALUE:
                        cleaned[k] = normalized
                elif k == "impersonate":
                    normalized = _normalize_scalar_string(v, allow_none=True)
                    if normalized is not _INVALID_VALUE:
                        cleaned[k] = normalized
                elif k == "url_patterns":
                    if type(v) is not list:
                        discard_rule = True
                        break
                    pattern_count = len(v)
                    if pattern_count == 0:
                        cleaned[k] = []
                        continue

                    pats: list[str] = []
                    deadline = _monotonic() + _URL_PATTERN_TOTAL_BUDGET_S
                    for p in v[: min(pattern_count, _MAX_URL_PATTERNS)]:
                        if type(p) is not str:
                            continue
                        remaining_s = deadline - _monotonic()
                        if remaining_s <= 0:
                            break
                        validation = search_untrusted(
                            p,
                            "",
                            limits=_router_regex_limits(remaining_s),
                        )
                        if validation.code is None:
                            pats.append(p)
                    if not pats:
                        discard_rule = True
                        break
                    cleaned[k] = pats
                elif k in ("extra_headers", "cookies", "proxies"):
                    cleaned[k] = _normalize_string_mapping(v)
                elif k == "respect_robots":
                    normalized = _normalize_bool(v)
                    if normalized is not _INVALID_VALUE:
                        cleaned[k] = normalized
                elif k == "strategy_order":
                    normalized = _normalize_string_list(v)
                    if normalized is not _INVALID_VALUE:
                        cleaned[k] = normalized
                elif (
                    k in {"schema_rules", "schema"}
                    or k in {"llm_settings", "llm"}
                    or k in {"regex_settings", "regex"}
                    or k in {"cluster_settings", "cluster"}
                ):
                    normalized = _normalize_object_mapping(v)
                    if normalized is not _INVALID_VALUE:
                        cleaned[k] = normalized
                else:
                    cleaned[k] = v

            if not discard_rule:
                out["domains"][dom] = cleaned
        return out

    def resolve(self, url: str) -> ScrapePlan:
        domain = _parse_domain(url)
        match = _match_domain_rule(domain, self.rules)

        # Pick UA profile (fixed or rotate)
        ua_profile = pick_ua_profile(self.ua_mode, domain=domain)
        impersonate = profile_to_impersonate(ua_profile)

        plan = ScrapePlan(
            url=url,
            domain=domain,
            ua_profile=ua_profile,
            impersonate=impersonate,
            respect_robots=self.default_respect_robots,
        )

        if not match:
            return plan

        _key, rule = match
        # Build from rule
        backend = _normalize_backend(rule.get("backend", plan.backend))
        handler_raw = rule.get("handler", plan.handler)
        handler = _validate_handler(handler_raw, self.allowlist)

        # If url_patterns present, apply only if any matches
        if "url_patterns" in rule:
            raw_patterns = rule.get("url_patterns")
            if type(raw_patterns) is not list:
                return plan

            pattern_count = len(raw_patterns)
            if pattern_count > 0:
                matched_pattern = False
                deadline = _monotonic() + _URL_PATTERN_TOTAL_BUDGET_S
                for pattern in raw_patterns[: min(pattern_count, _MAX_URL_PATTERNS)]:
                    if type(pattern) is not str:
                        continue
                    remaining_s = deadline - _monotonic()
                    if remaining_s <= 0:
                        break
                    result = search_untrusted(
                        pattern,
                        url,
                        limits=_router_regex_limits(remaining_s),
                    )
                    if result.matched:
                        matched_pattern = True
                        break

                if not matched_pattern:
                    # If rule has patterns and none matched, do not apply; fall back
                    return plan

        plan.backend = backend
        plan.handler = handler
        normalized_ua_profile = _normalize_scalar_string(rule.get("ua_profile", _INVALID_VALUE))
        if normalized_ua_profile is not _INVALID_VALUE:
            plan.ua_profile = normalized_ua_profile

        normalized_impersonate = _normalize_scalar_string(
            rule.get("impersonate", _INVALID_VALUE),
            allow_none=True,
        )
        if normalized_impersonate is _INVALID_VALUE:
            plan.impersonate = profile_to_impersonate(plan.ua_profile)
        else:
            plan.impersonate = normalized_impersonate

        plan.extra_headers = _normalize_string_mapping(rule.get("extra_headers"))
        # Cookies can be provided as simple name->value map
        plan.cookies = _normalize_string_mapping(rule.get("cookies"))
        # Per-domain proxies
        plan.proxies = _normalize_string_mapping(rule.get("proxies"))
        # Per-rule robots override
        if "respect_robots" in rule:
            normalized_robots = _normalize_bool(rule.get("respect_robots"))
            if normalized_robots is not _INVALID_VALUE:
                plan.respect_robots = normalized_robots
        strategy_order = _normalize_string_list(rule.get("strategy_order", _INVALID_VALUE))
        if strategy_order is not _INVALID_VALUE:
            plan.strategy_order = strategy_order
        plan.schema_rules = _mapping_alias(rule, "schema_rules", "schema")
        plan.llm_settings = _mapping_alias(rule, "llm_settings", "llm")
        plan.regex_settings = _mapping_alias(rule, "regex_settings", "regex")
        plan.cluster_settings = _mapping_alias(rule, "cluster_settings", "cluster")
        return plan


__all__ = [
    "ScraperRouter",
    "ScrapePlan",
]
