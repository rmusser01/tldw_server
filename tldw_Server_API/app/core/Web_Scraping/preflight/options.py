"""Configuration normalization for governed preflight analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from loguru import logger

ScanDepth = Literal["default", "thorough", "deep"]

_ABSENT = object()
_TRUE_STRINGS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "n", "off"})
_SCAN_DEPTHS = frozenset({"default", "thorough", "deep"})
_EXTERNAL_TOOL_WARNING = "Invalid preflight external-tool setting; external tools disabled."


def _legacy_bool(value: Any, default: bool) -> bool:
    """Preserve the boolean coercion used by the two legacy consumers."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return default


def _positive_timeout(value: Any) -> float | None:
    """Return a positive finite timeout or the legacy unbounded default."""
    try:
        normalized = float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(normalized) or normalized <= 0:
        return None
    return normalized


def _scan_depth(value: Any) -> ScanDepth:
    """Normalize scan depth to the approved vocabulary."""
    normalized = str(value or "").strip().lower()
    if normalized not in _SCAN_DEPTHS:
        return "default"
    return cast(ScanDepth, normalized)


def _explicit_external_bool(value: Any) -> bool:
    """Normalize the opt-in external-tool switch, failing closed when malformed."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    logger.warning(_EXTERNAL_TOOL_WARNING)
    return False


@dataclass(frozen=True, slots=True)
class PreflightOptions:
    """Normalized preflight settings shared by both scrape consumers."""

    enabled: bool = False
    timeout_s: float | None = None
    scan_depth: ScanDepth = "default"
    find_all_waf: bool = False
    impersonate: bool = False
    include_results: bool = False
    external_tools_enabled: bool | None = None
    playwright_no_sandbox: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> PreflightOptions:
        """Build options from the existing web-scraper configuration mapping."""
        values = dict(config or {})
        external = values.get("web_scraper_preflight_enable_external_tools", _ABSENT)
        return cls(
            enabled=_legacy_bool(values.get("web_scraper_preflight_analyzers"), False),
            timeout_s=_positive_timeout(values.get("web_scraper_preflight_timeout_s")),
            scan_depth=_scan_depth(values.get("web_scraper_preflight_scan_depth")),
            find_all_waf=_legacy_bool(values.get("web_scraper_preflight_find_all_waf"), False),
            impersonate=_legacy_bool(values.get("web_scraper_preflight_impersonate"), False),
            include_results=_legacy_bool(values.get("web_scraper_preflight_include_results"), False),
            external_tools_enabled=(None if external is _ABSENT else _explicit_external_bool(external)),
            playwright_no_sandbox=_legacy_bool(values.get("web_scraper_playwright_no_sandbox"), False),
        )
