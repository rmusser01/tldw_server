"""Precedence for UserProfiles ``limits.*`` keys (spec 2 §2).

The user's own value wins; otherwise the most generous value among the user's
teams that set the key; otherwise the most generous among their orgs. 0 is the
least generous value. Joining a group never lowers an allowance.
"""

from __future__ import annotations

from typing import Any

LIMITS_PREFIX = "limits."


def _numeric(value: Any) -> int | float | None:
    """The value when it is a real number (bools are not), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def most_generous_values(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    """For each ``limits.*`` key, the largest numeric value among the rows that set it."""
    best: dict[str, int | float] = {}
    for row in rows:
        key = str(row.get("key") or "")
        value = _numeric(row.get("value"))
        if not key.startswith(LIMITS_PREFIX) or value is None:
            continue
        if key not in best or value > best[key]:
            best[key] = value
    return best


def effective_limits(
    user_rows: list[dict[str, Any]],
    team_rows: list[dict[str, Any]],
    org_rows: list[dict[str, Any]],
) -> dict[str, int | float]:
    """The user's effective ``limits.*`` values: user over team over org."""
    effective = most_generous_values(org_rows)
    effective.update(most_generous_values(team_rows))
    for row in user_rows:
        key = str(row.get("key") or "")
        value = _numeric(row.get("value"))
        if key.startswith(LIMITS_PREFIX) and value is not None:
            effective[key] = value
    return effective
