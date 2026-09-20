"""Non-terminal extraction result enrichment helpers."""

from copy import deepcopy
from typing import Any, Optional


def regex_mask_override(settings: Optional[dict[str, Any]]) -> Optional[bool]:
    """Return an explicit regex PII-mask override when configured."""
    if not isinstance(settings, dict):
        return None
    for key in ("mask_pii", "pii_mask"):
        if key in settings:
            return bool(settings.get(key))
    return None


def enrich_with_regex_matches(
    result: dict[str, Any],
    regex_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Copy regex matches into a later strategy result without replacing its body."""
    enriched = deepcopy(result)
    if isinstance(regex_result, dict) and "regex_matches" in regex_result:
        enriched["regex_matches"] = deepcopy(regex_result["regex_matches"])
    return enriched
