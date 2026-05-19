"""Field-type-aware config merging utilities.

Shared logic extracted from ``mcp_hub_policy_resolver._merge_policy_documents``
so that both the policy resolver and the template inheritance system can reuse
the same semantics:

* **Scalars** in *overlay* override values in *base*.
* **Dicts** are merged recursively.
* **Union-list keys** (e.g. ``allowed_tools``, ``capabilities``) are appended
  with deduplication.
* ``None`` values in *overlay* are silently skipped (the base value is kept).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

_MISSING = object()

# Keys whose values are treated as lists and merged via append+dedup.
UNION_LIST_KEYS: frozenset[str] = frozenset({
    "allowed_tools",
    "denied_tools",
    "tool_names",
    "tool_patterns",
    "capabilities",
    "tool_modules",
    "module_ids",
})


def _as_str_list(value: Any) -> list[str]:
    """Normalize a scalar or iterable value into a list of non-empty strings."""
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, (list, tuple, set)):
        return []
    out: list[str] = []
    for entry in value:
        cleaned = str(entry or "").strip()
        if cleaned:
            out.append(cleaned)
    return out


def _unique(items: list[str]) -> list[str]:
    """Preserve order while removing duplicate strings."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a shallow dict copy for mapping values, otherwise an empty dict."""
    return dict(value) if isinstance(value, dict) else {}


def merge_config(
    base: dict[str, Any],
    overlay: dict[str, Any],
    *,
    skip_none: bool = True,
) -> dict[str, Any]:
    """Merge *overlay* on top of *base* with field-type-aware semantics.

    Rules:
    1. ``None`` values in *overlay* are skipped when ``skip_none`` is true.
    2. Keys in :data:`UNION_LIST_KEYS` are appended and de-duplicated.
    3. If both sides are dicts the merge recurses.
    4. Everything else: overlay value replaces base value (deep-copied).

    Neither *base* nor *overlay* is mutated.
    """
    merged = dict(base)
    for key, value in overlay.items():
        base_value = base.get(key, _MISSING)
        if value is None and skip_none:
            if base_value is not _MISSING:
                merged[key] = deepcopy(base_value)
            continue
        if key in UNION_LIST_KEYS:
            merged[key] = _unique(_as_str_list(base_value) + _as_str_list(value))
            continue
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = merge_config(base_value, value, skip_none=skip_none)
            continue
        merged[key] = deepcopy(value)
    for key, value in base.items():
        if key not in overlay:
            merged[key] = deepcopy(value)
    return merged
