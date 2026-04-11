"""Field-type-aware config merging utility.

Extracted from mcp_hub_policy_resolver._merge_policy_documents for reuse
by both policy merging and template inheritance.

Merge rules:
- Scalar fields (str, int, bool): overlay value wins if non-None
- Dict fields: recursive merge (overlay keys override base keys)
- List fields in _UNION_LIST_KEYS: append with deduplication
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


# Keys whose values are lists that should be merged via union (append + dedup)
_UNION_LIST_KEYS = frozenset({
    "allowed_tools",
    "denied_tools",
    "tool_names",
    "tool_patterns",
    "capabilities",
    "tool_modules",
    "module_ids",
    "allowed_models",
    "denied_models",
})


def _unique(items: list) -> list:
    """Deduplicate while preserving order."""
    seen: set = set()
    result: list = []
    for item in items:
        key = str(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _as_str_list(val: Any) -> list[str]:
    """Normalize a scalar or iterable value into a list of non-empty strings."""
    if isinstance(val, str):
        cleaned = val.strip()
        return [cleaned] if cleaned else []
    if not isinstance(val, (list, tuple, set)):
        return []
    out: list[str] = []
    for entry in val:
        cleaned = str(entry or "").strip()
        if cleaned:
            out.append(cleaned)
    return out


def _as_dict(val: Any) -> dict[str, Any]:
    """Return a dict for mapping values, otherwise an empty dict."""
    return dict(val) if isinstance(val, dict) else {}


def merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge two config dicts with field-type-aware rules.

    - Known list keys (allowed_tools, denied_tools, etc.): union with dedup
    - Dict values: recursive merge
    - ``None`` overlay values: skipped (base preserved)
    - Everything else: overlay wins
    """
    merged = deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            continue
        if key in _UNION_LIST_KEYS:
            merged[key] = _unique(_as_str_list(merged.get(key)) + _as_str_list(value))
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_config(_as_dict(merged.get(key)), value)
            continue
        merged[key] = deepcopy(value)
    return merged
