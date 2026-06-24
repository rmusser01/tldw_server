from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.UserProfiles.contracts import (
    EffectDescriptor,
    EffectPolicy,
    EffectTiming,
    ProfileContractMode,
    ProfileUpdateCommand,
    PlannedUpdateResult,
    UpdateMutation,
    UpdatePlan,
)


def test_update_plan_separates_pre_commit_and_post_commit_effects() -> None:
    command = ProfileUpdateCommand(
        actor_user_id=5,
        target_user_id=7,
        updates=(("preferences.ui.theme", "paper"),),
        roles=frozenset({"user"}),
        dry_run=False,
        expected_profile_version=datetime(2026, 1, 1, tzinfo=timezone.utc),
        contract_mode=ProfileContractMode.LEGACY_V1,
    )

    plan = UpdatePlan(
        command=command,
        mutations=(
            UpdateMutation(
                key="preferences.ui.theme",
                operation="upsert_override",
                payload={"value": "paper"},
            ),
        ),
        effects=(
            EffectDescriptor(
                name="audit_preflight",
                timing=EffectTiming.PRE_COMMIT,
                policy=EffectPolicy.REQUIRED,
            ),
            EffectDescriptor(
                name="refresh_cache",
                timing=EffectTiming.POST_COMMIT,
                policy=EffectPolicy.BEST_EFFORT,
            ),
            EffectDescriptor(
                name="validate_version",
                timing=EffectTiming.PRE_COMMIT,
                policy=EffectPolicy.REQUIRED,
            ),
            EffectDescriptor(
                name="emit_profile_changed",
                timing=EffectTiming.POST_COMMIT,
                policy=EffectPolicy.BEST_EFFORT,
            ),
        ),
    )

    assert plan.command.target_user_id == 7
    assert plan.mutations[0].operation == "upsert_override"
    assert tuple(effect.name for effect in plan.pre_commit_effects) == (
        "audit_preflight",
        "validate_version",
    )
    assert tuple(effect.name for effect in plan.post_commit_effects) == (
        "refresh_cache",
        "emit_profile_changed",
    )


def test_contract_payloads_are_immutable_copies() -> None:
    mutation_payload = {
        "value": {
            "theme": "paper",
            "tags": ["reader"],
        },
    }
    effect_payload = {
        "event": {
            "kind": "profile.updated",
            "ids": [7],
        },
    }
    rejected = ({"field": "display_name", "reason": "blank"},)

    mutation = UpdateMutation(
        key="preferences.ui.theme",
        operation="upsert_override",
        payload=mutation_payload,
    )
    effect = EffectDescriptor(
        name="emit_profile_changed",
        timing=EffectTiming.POST_COMMIT,
        policy=EffectPolicy.BEST_EFFORT,
        payload=effect_payload,
    )
    result = PlannedUpdateResult(
        profile_version=datetime(2026, 1, 1, tzinfo=timezone.utc),
        rejected=rejected,
    )

    mutation_payload["value"]["theme"] = "dark"
    mutation_payload["value"]["tags"].append("mutated")
    effect_payload["event"]["kind"] = "profile.deleted"
    effect_payload["event"]["ids"].append(9)
    rejected[0]["reason"] = "changed"

    assert mutation.payload["value"]["theme"] == "paper"
    assert mutation.payload["value"]["tags"] == ("reader",)
    assert effect.payload["event"]["kind"] == "profile.updated"
    assert effect.payload["event"]["ids"] == (7,)
    assert result.rejected[0]["reason"] == "blank"

    with pytest.raises(TypeError):
        mutation.payload["value"]["theme"] = "dark"
    with pytest.raises(TypeError):
        effect.payload["event"]["ids"] = (9,)
    with pytest.raises(TypeError):
        result.rejected[0]["reason"] = "changed"

    default_mutation = UpdateMutation(key="profile", operation="noop")
    with pytest.raises(TypeError):
        default_mutation.payload["unexpected"] = True


def test_effect_policy_values_are_stable() -> None:
    assert ProfileContractMode.LEGACY_V1 == "legacy_v1"
    assert ProfileContractMode.CLEAN_V2 == "clean_v2"
    assert EffectTiming.PRE_COMMIT.value == "pre_commit"
    assert EffectTiming.PRE_COMMIT == "pre_commit"
    assert EffectTiming.POST_COMMIT.value == "post_commit"
    assert EffectTiming.POST_COMMIT == "post_commit"
    assert EffectPolicy.REQUIRED.value == "required"
    assert EffectPolicy.REQUIRED == "required"
    assert EffectPolicy.BEST_EFFORT.value == "best_effort"
    assert EffectPolicy.BEST_EFFORT == "best_effort"
