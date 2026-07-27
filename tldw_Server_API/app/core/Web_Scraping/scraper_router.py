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
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import yaml

from .safe_regex import SafeRegexLimits, search_untrusted
from .ua_profiles import pick_ua_profile, profile_to_impersonate

DEFAULT_HANDLER = (
    "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html"
)

DEFAULT_HANDLER_ALLOWLIST = [
    "tldw_Server_API.app.core.Web_Scraping.handlers:",
]

_ROUTER_REGEX_LIMITS = SafeRegexLimits()
# Configured patterns are user-controlled, so cap aggregate work per domain rule.
_MAX_URL_PATTERNS = 32
_URL_PATTERN_TOTAL_BUDGET_S = 0.100


def _monotonic() -> float:
    return time.monotonic()


def _router_regex_limits(timeout_s: float) -> SafeRegexLimits:
    return SafeRegexLimits(
        max_pattern_chars=_ROUTER_REGEX_LIMITS.max_pattern_chars,
        max_input_chars=_ROUTER_REGEX_LIMITS.max_input_chars,
        timeout_s=min(timeout_s, _ROUTER_REGEX_LIMITS.timeout_s),
    )


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
    proxies: dict[str, str] = field(default_factory=dict)  # e.g., {"http": "http://host:port", "https": "http://host:port"}
    strategy_order: list[str] | None = None
    schema_rules: dict[str, Any] | None = None
    llm_settings: dict[str, Any] | None = None
    regex_settings: dict[str, Any] | None = None
    cluster_settings: dict[str, Any] | None = None


def _validate_handler(handler: str, allowlist: list[str]) -> str:
    if any(handler.startswith(prefix) for prefix in allowlist):
        return handler
    # Fallback to safe default
    return DEFAULT_HANDLER


def _parse_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _match_domain_rule(domain: str, rules: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    # 1) Exact
    dom_rules = rules.get("domains", {})
    if domain in dom_rules:
        return domain, dom_rules[domain]

    # 2) Wildcard (*.example.com)
    best_match: tuple[str, dict[str, Any]] | None = None
    best_suffix_len = -1
    for key, rule in dom_rules.items():
        if key.startswith("*."):
            suffix = key[1:]  # remove leading '*'
            if domain.endswith(suffix):
                # Pick the longest suffix for specificity
                if len(suffix) > best_suffix_len:
                    best_match = (key, rule)
                    best_suffix_len = len(suffix)

    if best_match:
        return best_match

    # 3) No domain-level match
    return None


class ScraperRouter:
    def __init__(
        self,
        rules: dict[str, Any] | None = None,
        *,
        handler_allowlist: list[str] | None = None,
        ua_mode: str = "fixed",
        default_respect_robots: bool = True,
    ) -> None:
        self.rules = rules or {}
        self.allowlist = handler_allowlist or DEFAULT_HANDLER_ALLOWLIST
        self.ua_mode = ua_mode
        self.default_respect_robots = bool(default_respect_robots)

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
    def validate_rules(data: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize rules loaded from YAML.

        - Ensure top-level 'domains' mapping
        - Keep only known keys per domain rule
        - Validate backend and url_patterns
        - Normalize headers/cookies to string maps
        """
        out: dict[str, Any] = {"domains": {}}
        if not isinstance(data, dict):
            return out
        domains = data.get("domains", {}) or {}
        if not isinstance(domains, dict):
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
        allowed_backends = {"auto", "curl", "httpx", "playwright"}

        for dom, rule in domains.items():
            if not isinstance(dom, str) or not isinstance(rule, dict):
                continue
            # minimal domain/wildcard sanity: must contain a dot or start with '*.'
            if not (dom.startswith("*.") or "." in dom):
                continue

            cleaned: dict[str, Any] = {}
            discard_rule = False
            for k, v in rule.items():
                if k not in allowed_keys:
                    continue
                if k == "backend":
                    val = str(v).lower().strip()
                    cleaned[k] = val if val in allowed_backends else "auto"
                elif k == "url_patterns":
                    if not isinstance(v, list):
                        discard_rule = True
                        break
                    if not v:
                        cleaned[k] = []
                        continue

                    pats: list[str] = []
                    deadline = _monotonic() + _URL_PATTERN_TOTAL_BUDGET_S
                    for p in v[:_MAX_URL_PATTERNS]:
                        if not isinstance(p, str):
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
                elif k in ("extra_headers", "cookies"):
                    m = {str(kk): str(vv) for kk, vv in v.items()} if isinstance(v, dict) else {}
                    cleaned[k] = m
                elif k == "respect_robots":
                    cleaned[k] = bool(v)
                elif k == "strategy_order":
                    order: list[str] = []
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                order.append(item)
                    cleaned[k] = order
                elif k in {"schema_rules", "schema"} or k in {"llm_settings", "llm"} or k in {"regex_settings", "regex"} or k in {"cluster_settings", "cluster"}:
                    cleaned[k] = v if isinstance(v, dict) else {}
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
        backend = rule.get("backend") or plan.backend
        handler_raw = rule.get("handler") or plan.handler
        handler = _validate_handler(handler_raw, self.allowlist)

        # If url_patterns present, apply only if any matches
        raw_patterns = rule.get("url_patterns", [])
        if raw_patterns:
            if not isinstance(raw_patterns, list):
                return plan

            matched_pattern = False
            deadline = _monotonic() + _URL_PATTERN_TOTAL_BUDGET_S
            for pattern in raw_patterns[:_MAX_URL_PATTERNS]:
                if not isinstance(pattern, str):
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

        plan.backend = str(backend)
        plan.handler = handler
        plan.ua_profile = str(rule.get("ua_profile", plan.ua_profile))
        plan.impersonate = rule.get("impersonate", profile_to_impersonate(plan.ua_profile))
        plan.extra_headers = dict(rule.get("extra_headers", {}))
        # Cookies can be provided as simple name->value map
        plan.cookies = dict(rule.get("cookies", {}))
        # Per-domain proxies
        plan.proxies = dict(rule.get("proxies", {}))
        # Per-rule robots override
        if "respect_robots" in rule:
            plan.respect_robots = bool(rule.get("respect_robots"))
        strategy_order = rule.get("strategy_order")
        if isinstance(strategy_order, list):
            plan.strategy_order = [str(item) for item in strategy_order if isinstance(item, str)]
        schema_rules = rule.get("schema_rules")
        if isinstance(schema_rules, dict):
            plan.schema_rules = schema_rules
        else:
            schema_alt = rule.get("schema")
            if isinstance(schema_alt, dict):
                plan.schema_rules = schema_alt
        llm_settings = rule.get("llm_settings")
        if isinstance(llm_settings, dict):
            plan.llm_settings = llm_settings
        else:
            llm_alt = rule.get("llm")
            if isinstance(llm_alt, dict):
                plan.llm_settings = llm_alt
        regex_settings = rule.get("regex_settings")
        if isinstance(regex_settings, dict):
            plan.regex_settings = regex_settings
        else:
            regex_alt = rule.get("regex")
            if isinstance(regex_alt, dict):
                plan.regex_settings = regex_alt
        cluster_settings = rule.get("cluster_settings")
        if isinstance(cluster_settings, dict):
            plan.cluster_settings = cluster_settings
        else:
            cluster_alt = rule.get("cluster")
            if isinstance(cluster_alt, dict):
                plan.cluster_settings = cluster_alt
        return plan


__all__ = [
    "ScraperRouter",
    "ScrapePlan",
]
