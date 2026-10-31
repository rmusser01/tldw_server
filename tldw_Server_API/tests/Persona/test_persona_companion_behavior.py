from __future__ import annotations

import math

import pytest

from tldw_Server_API.app.core.Persona.companion_behavior import (
    CompanionBehaviorValidationError,
    normalize_companion_behavior,
)

pytestmark = pytest.mark.unit


def _entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "state": "ambient.look",
        "trigger": "ambient",
        "category": "idle_variant",
    }
    entry.update(overrides)
    return entry


def test_normalize_behavior_preserves_relative_weights() -> None:
    normalized = normalize_companion_behavior(
        {
            "schema_version": 1,
            "entries": [
                _entry(
                    suggested_weight=3,
                    suggested_cooldown_ms=45_000,
                )
            ],
        },
        resolvable_state_ids={"idle", "ambient.look"},
    )

    assert normalized is not None
    assert normalized["entries"][0]["suggested_weight"] == 3.0


def test_behavior_none_remains_absent_without_inference() -> None:
    assert normalize_companion_behavior(None, resolvable_state_ids={"idle"}) is None


def test_behavior_rejects_float_schema_version() -> None:
    with pytest.raises(CompanionBehaviorValidationError):
        normalize_companion_behavior(
            {"schema_version": 1.0, "entries": []},
            resolvable_state_ids={"idle"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trigger", []),
        ("trigger", {}),
        ("category", []),
        ("category", {}),
    ],
)
def test_behavior_rejects_non_string_enums_with_contract_error(
    field: str,
    value: object,
) -> None:
    with pytest.raises(CompanionBehaviorValidationError):
        normalize_companion_behavior(
            {"schema_version": 1, "entries": [_entry(**{field: value})]},
            resolvable_state_ids={"ambient.look"},
        )


@pytest.mark.parametrize(
    "weight",
    [1 << 100_000, -(1 << 100_000)],
    ids=["huge-positive-json-integer", "huge-negative-json-integer"],
)
def test_behavior_rejects_huge_json_integers_with_contract_error(weight: int) -> None:
    with pytest.raises(CompanionBehaviorValidationError):
        normalize_companion_behavior(
            {"schema_version": 1, "entries": [_entry(suggested_weight=weight)]},
            resolvable_state_ids={"ambient.look"},
        )


@pytest.mark.parametrize("weight", [-1, 10_001, float("inf"), float("nan"), True])
def test_behavior_rejects_invalid_weights(weight: object) -> None:
    with pytest.raises(CompanionBehaviorValidationError):
        normalize_companion_behavior(
            {"schema_version": 1, "entries": [_entry(suggested_weight=weight)]},
            resolvable_state_ids={"ambient.look"},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "entries": []},
        {"schema_version": 1, "entries": "invalid"},
        {"schema_version": 1, "entries": [_entry(state="x" * 129)]},
        {"schema_version": 1, "entries": [_entry(trigger="hover")]},
        {"schema_version": 1, "entries": [_entry(category="dance")]},
        {"schema_version": 1, "entries": [_entry(state="missing")]},
        {"schema_version": 1, "entries": [_entry(suggested_cooldown_ms=-1)]},
        {"schema_version": 1, "entries": [_entry(suggested_cooldown_ms=86_400_001)]},
        {"schema_version": 1, "entries": [_entry(), _entry()]},
        {"schema_version": 1, "entries": [_entry(extra="unknown")]},
    ],
)
def test_behavior_rejects_invalid_contracts(payload: dict[str, object]) -> None:
    with pytest.raises(CompanionBehaviorValidationError):
        normalize_companion_behavior(payload, resolvable_state_ids={"ambient.look"})


def test_behavior_accepts_limits_and_canonical_movement() -> None:
    entries = [
        _entry(
            state=f"ambient.{index}",
            suggested_weight=0,
            suggested_cooldown_ms=86_400_000,
        )
        for index in range(127)
    ]
    entries.append(
        _entry(
            state="m" * 128,
            category="move",
            movement={
                "direction": "horizontal",
                "motion_start_ratio": 0,
                "motion_end_ratio": 1,
            },
        )
    )
    states = {str(entry["state"]) for entry in entries}

    normalized = normalize_companion_behavior(
        {"schema_version": 1, "entries": entries},
        resolvable_state_ids=states,
    )

    assert normalized is not None
    assert len(normalized["entries"]) == 128
    assert normalized["entries"][-1]["movement"] == {
        "direction": "horizontal",
        "motion_start_ratio": 0.0,
        "motion_end_ratio": 1.0,
    }


@pytest.mark.parametrize("category", ["idle_variant", "move"])
def test_behavior_rejects_explicit_null_movement(category: str) -> None:
    """Movement is presence-sensitive: explicit null is never an omitted field."""
    with pytest.raises(CompanionBehaviorValidationError):
        normalize_companion_behavior(
            {
                "schema_version": 1,
                "entries": [_entry(category=category, movement=None)],
            },
            resolvable_state_ids={"ambient.look"},
        )


def test_behavior_allows_absent_movement_only_for_non_move_entries() -> None:
    normalized = normalize_companion_behavior(
        {"schema_version": 1, "entries": [_entry()]},
        resolvable_state_ids={"ambient.look"},
    )

    assert normalized is not None
    assert "movement" not in normalized["entries"][0]


@pytest.mark.parametrize(
    "movement",
    [
        {"direction": "vertical", "motion_start_ratio": 0, "motion_end_ratio": 1},
        {"direction": "horizontal", "motion_start_ratio": -0.1, "motion_end_ratio": 1},
        {"direction": "horizontal", "motion_start_ratio": 0, "motion_end_ratio": 1.1},
        {"direction": "horizontal", "motion_start_ratio": 0.8, "motion_end_ratio": 0.2},
        {"direction": "horizontal", "motion_start_ratio": math.inf, "motion_end_ratio": 1},
    ],
)
def test_behavior_rejects_invalid_movement_without_clamping(
    movement: dict[str, object],
) -> None:
    with pytest.raises(CompanionBehaviorValidationError):
        normalize_companion_behavior(
            {
                "schema_version": 1,
                "entries": [_entry(category="move", movement=movement)],
            },
            resolvable_state_ids={"ambient.look"},
        )


def test_behavior_rejects_more_than_128_entries() -> None:
    entries = [_entry(state=f"ambient.{index}") for index in range(129)]
    with pytest.raises(CompanionBehaviorValidationError):
        normalize_companion_behavior(
            {"schema_version": 1, "entries": entries},
            resolvable_state_ids={str(entry["state"]) for entry in entries},
        )
