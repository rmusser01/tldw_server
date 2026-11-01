"""Resolve characterized profile options without importing legacy adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts.requests import CrawlRequest
from .models import CrawlLimits, CrawlPlan

_CONFIG_OPTIONS = {
    "include_external": ("web_crawl_include_external", False),
    "score_threshold": ("web_crawl_score_threshold", 0.0),
    "crawl_strategy": ("web_crawl_strategy", "best_first"),
    "allowed_domains": ("web_crawl_allowed_domains", ""),
    "blocked_domains": ("web_crawl_blocked_domains", ""),
    "respect_robots": ("web_scraper_respect_robots", True),
    "enable_keyword_scorer": ("web_crawl_enable_keyword_scorer", False),
    "keywords": ("web_crawl_keywords", ""),
    "enable_domain_map": ("web_crawl_enable_domain_map", False),
    "domain_map": ("web_crawl_domain_map", ""),
}


def resolve_crawl_plan(
    request: CrawlRequest,
    *,
    profile: str,
    options: Mapping[str, Any],
    supplied: frozenset[str],
    config: Mapping[str, Any],
    limits: CrawlLimits,
) -> CrawlPlan:
    """Keep presence separate from constructor defaults and execution credentials.

    Canonical option names are used internally; existing override names are accepted
    at this boundary. Instance config remains authoritative, as characterized.
    Explicit None for enhanced overrides falls back to config; false/zero do not.
    """
    enhanced = profile.startswith("enhanced_")
    recursive = profile.endswith("recursive") or profile == "enhanced_url_level"
    values: dict[str, Any] = {"allow_llm_extraction": True}
    if recursive or profile == "legacy_url_level":
        values.update(max_pages=100 if enhanced else None, max_depth=3 if enhanced else None, include_external=False)
    if profile == "legacy_recursive":
        values.update(delay=1.0, resume_file="scrape_progress.json")
    if profile == "enhanced_sitemap":
        values["max_urls"] = None
    if profile == "enhanced_batch":
        values.update(method="auto", priority="NORMAL", summarize=False)
    sources = dict.fromkeys(values, "default")
    for name in ("max_pages", "max_depth"):
        value = getattr(request, name)
        if name in values and value is not None:
            values[name], sources[name] = value, "request"
    if enhanced and recursive:
        for name, (key, default) in _CONFIG_OPTIONS.items():
            values[name] = config.get(key, default)
            sources[name] = "config" if key in config else "default"
        values["include_external"] = bool(values["include_external"])
        try:
            values["score_threshold"] = float(values["score_threshold"])
        except (ValueError, TypeError, OverflowError):
            values["score_threshold"] = 0.0
    aliases = {"include_external_override": "include_external", "score_threshold_override": "score_threshold"}
    for supplied_name in supplied:
        name = aliases.get(supplied_name, supplied_name)
        if supplied_name in aliases and name in supplied:
            raise ValueError("duplicate crawl option presence")
        if supplied_name in options:
            value = options[supplied_name]
        elif name in {"max_pages", "max_depth", "include_external"}:
            value = getattr(request, name)
        else:
            continue
        if name not in values and name not in {"level"}:
            continue
        if enhanced and recursive and name in {"include_external", "score_threshold", "crawl_strategy"}:
            if value is None or (name == "crawl_strategy" and not value):
                sources[name] = "supplied:" + sources[name]
                continue
            if name == "include_external":
                value = bool(value)
            elif name == "score_threshold":
                try:
                    value = float(value)
                except (ValueError, TypeError, OverflowError):
                    sources[name] = "supplied:" + sources[name]
                    continue
        values[name], sources[name] = value, "supplied"
    if enhanced and recursive:
        values["crawl_strategy"] = str(values["crawl_strategy"]).strip().lower()
    return CrawlPlan(request, profile, values, sources, limits)
