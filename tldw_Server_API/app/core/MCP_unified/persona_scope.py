"""Helpers for persona-scoped MCP retrieval filtering."""

from __future__ import annotations

from typing import Any

_SCOPE_RULE_TYPES = {
    "conversation_id",
    "character_id",
    "media_id",
    "note_id",
    "prompt_id",
}


def _coerce_values_to_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        items = raw
    else:
        items = [raw]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def normalize_persona_scope_payload(payload: Any) -> dict[str, Any] | None:
    """
    Normalize a persona scope payload into a canonical shape.

    Accepted inputs:
    - Metadata payload (`{"explicit_ids": ...}`).
    - Stored scope snapshot (`{"materialized_scope": {"explicit_ids": ...}, ...}`).
    """
    if not isinstance(payload, dict):
        return None

    raw_scope_snapshot_id = str(payload.get("scope_snapshot_id") or "").strip()
    if not raw_scope_snapshot_id:
        audit = payload.get("audit")
        if isinstance(audit, dict):
            raw_scope_snapshot_id = str(audit.get("scope_snapshot_id") or "").strip()

    explicit_ids_raw = payload.get("explicit_ids")
    if not isinstance(explicit_ids_raw, dict):
        materialized_scope = payload.get("materialized_scope")
        if isinstance(materialized_scope, dict):
            explicit_ids_raw = materialized_scope.get("explicit_ids")
    if not isinstance(explicit_ids_raw, dict):
        scope_snapshot = payload.get("scope_snapshot")
        if isinstance(scope_snapshot, dict):
            materialized_scope = scope_snapshot.get("materialized_scope")
            if isinstance(materialized_scope, dict):
                explicit_ids_raw = materialized_scope.get("explicit_ids")
    if not isinstance(explicit_ids_raw, dict):
        explicit_ids_raw = {}

    normalized_ids: dict[str, list[str]] = {}
    for rule_type in _SCOPE_RULE_TYPES:
        if rule_type not in explicit_ids_raw:
            continue
        normalized_ids[rule_type] = _coerce_values_to_list(explicit_ids_raw.get(rule_type))

    return {
        "scope_snapshot_id": raw_scope_snapshot_id or None,
        "explicit_ids": normalized_ids,
    }


def get_persona_scope_payload(context: Any | None) -> dict[str, Any] | None:
    """Return normalized persona scope metadata from a request context."""
    if context is None:
        return None
    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    raw_scope = metadata.get("persona_scope")
    return normalize_persona_scope_payload(raw_scope)


def get_explicit_scope_ids(context: Any | None, rule_type: str) -> set[str] | None:
    """
    Return explicit scoped IDs for a rule type.

    Returns:
    - `None` when scope metadata is absent or this rule type is not constrained.
    - `set()` when this rule type is constrained to an empty set (deny-all).
    - `set({"..."})` when constrained to specific IDs.
    """
    normalized_rule_type = str(rule_type or "").strip().lower()
    if normalized_rule_type not in _SCOPE_RULE_TYPES:
        return None

    payload = get_persona_scope_payload(context)
    if not payload:
        return None

    explicit_ids = payload.get("explicit_ids")
    if not isinstance(explicit_ids, dict):
        return None
    if normalized_rule_type not in explicit_ids:
        return None

    return set(_coerce_values_to_list(explicit_ids.get(normalized_rule_type)))


def assert_identifier_in_scope(
    context: Any | None,
    rule_type: str,
    identifier: Any,
    *,
    label: str,
) -> None:
    scoped_ids = get_explicit_scope_ids(context, rule_type)
    if scoped_ids is None:
        return
    if str(identifier or "") not in scoped_ids:
        raise PermissionError(f"{label} access denied by persona scope")


def merge_requested_ids_with_scope(
    requested_ids: Any,
    *,
    scoped_ids: set[str] | None,
) -> set[str] | None:
    """
    Merge user/requested IDs with persona scope constraints.

    - No request + no scope -> None (unrestricted).
    - No request + scope -> scoped IDs.
    - Request + no scope -> requested IDs.
    - Request + scope -> intersection.
    """
    if requested_ids is None:
        return set(scoped_ids) if scoped_ids is not None else None

    requested_set = set(_coerce_values_to_list(requested_ids))
    if scoped_ids is None:
        return requested_set
    return requested_set & set(scoped_ids)
