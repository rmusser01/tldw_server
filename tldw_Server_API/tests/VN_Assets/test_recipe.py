"""Stored recipe validation preserves valid snapshots and legacy absence."""

import json

import pytest

from tldw_Server_API.app.core.VN_Assets.recipe import load_execution_recipe, load_recipe

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("variant_count", [0, 1, 2])
def test_load_recipe_preserves_valid_slots_including_lazy_depth(variant_count: int) -> None:
    """Keep accepted slot values and prompt metadata unchanged on replay."""
    snapshot = {
        "version": 1,
        "pack_id": 1,
        "owner_user_id": 1,
        "primary_character_id": 1,
        "slots": [{
            "slot_id": 1,
            "slot_key": "background.depth",
            "asset_type": "depth_companion",
            "labels": {},
            "variant_count": variant_count,
            "prompt_snapshot": {"prompt": "Recorded prompt", "negative_prompt": None, "warnings": []},
            "requested_backend": None,
            "requested_model": None,
            "width": None,
            "height": None,
            "format": "png",
            "extra_params": {},
            "seeds": [None] * variant_count,
        }],
    }

    assert load_recipe(json.dumps(snapshot), pack_id=1, owner_user_id=1) == snapshot


def test_load_execution_recipe_preserves_optional_and_unknown_metadata() -> None:
    """Validation must not discard recorded backend metadata."""
    snapshot = {
        "version": 1,
        "slots": [{"slot_id": 1, "backend": "test", "model": None, "recorded_metadata": {"key": "value"}}],
    }

    assert load_execution_recipe(json.dumps(snapshot)) == snapshot


def test_load_execution_recipe_reports_absent_snapshot_as_unavailable() -> None:
    """Distinguish a missing execution choice from a malformed one."""
    with pytest.raises(ValueError, match="^vn_asset_execution_recipe_unavailable$"):
        load_execution_recipe(None)


def test_load_recipe_reports_legacy_absence_as_unavailable() -> None:
    """Retain the existing legacy-batch recovery error."""
    with pytest.raises(ValueError, match="^vn_asset_recipe_unavailable$"):
        load_recipe(None, pack_id=1, owner_user_id=1)


@pytest.mark.parametrize("version", [True, 1.0], ids=["boolean", "float"])
@pytest.mark.parametrize("execution", [False, True], ids=["authored", "execution"])
def test_recipe_loaders_reject_non_integer_versions(version: bool | float, execution: bool) -> None:
    """Reject version values that compare equal to one without being integers."""
    snapshot = {"version": version, "pack_id": 1, "owner_user_id": 1, "primary_character_id": 1, "slots": []}
    code = "vn_asset_execution_recipe_invalid" if execution else "vn_asset_recipe_invalid"

    with pytest.raises(ValueError, match=f"^{code}$"):
        if execution:
            load_execution_recipe(json.dumps(snapshot))
        else:
            load_recipe(json.dumps(snapshot), pack_id=1, owner_user_id=1)
