"""Validation for optional pack-level Persona companion behavior metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

ALLOWED_TRIGGERS = frozenset({"ambient", "click", "drag"})
ALLOWED_CATEGORIES = frozenset({"idle_variant", "move", "reaction"})
MAX_BEHAVIOR_ENTRIES = 128
MAX_BEHAVIOR_STATE_ID_LENGTH = 128
MAX_BEHAVIOR_WEIGHT = 10_000.0
MAX_BEHAVIOR_COOLDOWN_MS = 86_400_000

_ENTRY_KEYS = frozenset(
    {
        "state",
        "trigger",
        "category",
        "suggested_weight",
        "suggested_cooldown_ms",
        "movement",
    }
)
_MOVEMENT_KEYS = frozenset(
    {"direction", "motion_start_ratio", "motion_end_ratio"}
)


class CompanionBehaviorValidationError(ValueError):
    """Raised when pack-level companion behavior is invalid."""


def normalize_companion_behavior(
    value: Mapping[str, Any] | None,
    *,
    resolvable_state_ids: set[str],
) -> dict[str, Any] | None:
    """Validate and return the canonical companion behavior document."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "entries"}:
        raise CompanionBehaviorValidationError("companion behavior must be a strict object")
    schema_version = value.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise CompanionBehaviorValidationError(
            "unsupported companion behavior schema_version"
        )
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) > MAX_BEHAVIOR_ENTRIES:
        raise CompanionBehaviorValidationError("companion behavior entries are invalid")
    return {
        "schema_version": 1,
        "entries": _normalize_entries(entries, resolvable_state_ids),
    }


def _normalize_entries(
    entries: list[Any],
    resolvable_state_ids: set[str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not set(entry) <= _ENTRY_KEYS:
            raise CompanionBehaviorValidationError(
                f"companion behavior entry {index} is invalid"
            )
        state = entry.get("state")
        if (
            not isinstance(state, str)
            or not state
            or len(state) > MAX_BEHAVIOR_STATE_ID_LENGTH
            or state not in resolvable_state_ids
        ):
            raise CompanionBehaviorValidationError(
                f"companion behavior entry {index} state is invalid"
            )
        trigger = entry.get("trigger")
        if trigger not in ALLOWED_TRIGGERS:
            raise CompanionBehaviorValidationError(
                f"companion behavior entry {index} trigger is invalid"
            )
        category = entry.get("category")
        if category not in ALLOWED_CATEGORIES:
            raise CompanionBehaviorValidationError(
                f"companion behavior entry {index} category is invalid"
            )
        identity = (str(trigger), state)
        if identity in seen:
            raise CompanionBehaviorValidationError(
                "companion behavior trigger/state pairs must be unique"
            )
        seen.add(identity)

        item: dict[str, Any] = {
            "state": state,
            "trigger": trigger,
            "category": category,
        }
        if "suggested_weight" in entry:
            item["suggested_weight"] = _bounded_number(
                entry["suggested_weight"],
                field_name=f"entries[{index}].suggested_weight",
                minimum=0,
                maximum=MAX_BEHAVIOR_WEIGHT,
            )
        if "suggested_cooldown_ms" in entry:
            cooldown = entry["suggested_cooldown_ms"]
            if (
                isinstance(cooldown, bool)
                or not isinstance(cooldown, int)
                or not 0 <= cooldown <= MAX_BEHAVIOR_COOLDOWN_MS
            ):
                raise CompanionBehaviorValidationError(
                    f"entries[{index}].suggested_cooldown_ms is invalid"
                )
            item["suggested_cooldown_ms"] = cooldown

        movement = entry.get("movement")
        if movement is not None:
            if category != "move":
                raise CompanionBehaviorValidationError(
                    f"entries[{index}].movement requires category move"
                )
            item["movement"] = _normalize_movement(movement, index=index)
        elif category == "move":
            raise CompanionBehaviorValidationError(
                f"entries[{index}].movement is required for category move"
            )
        normalized.append(item)
    return normalized


def _normalize_movement(value: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MOVEMENT_KEYS:
        raise CompanionBehaviorValidationError(
            f"entries[{index}].movement is invalid"
        )
    if value.get("direction") != "horizontal":
        raise CompanionBehaviorValidationError(
            f"entries[{index}].movement.direction must be horizontal"
        )
    start = _bounded_number(
        value.get("motion_start_ratio"),
        field_name=f"entries[{index}].movement.motion_start_ratio",
        minimum=0,
        maximum=1,
    )
    end = _bounded_number(
        value.get("motion_end_ratio"),
        field_name=f"entries[{index}].movement.motion_end_ratio",
        minimum=0,
        maximum=1,
    )
    if start > end:
        raise CompanionBehaviorValidationError(
            f"entries[{index}].movement start must not exceed end"
        )
    return {
        "direction": "horizontal",
        "motion_start_ratio": start,
        "motion_end_ratio": end,
    }


def _bounded_number(
    value: Any,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompanionBehaviorValidationError(f"{field_name} is invalid")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise CompanionBehaviorValidationError(f"{field_name} is invalid")
    return number


__all__ = [
    "ALLOWED_CATEGORIES",
    "ALLOWED_TRIGGERS",
    "MAX_BEHAVIOR_ENTRIES",
    "CompanionBehaviorValidationError",
    "normalize_companion_behavior",
]
