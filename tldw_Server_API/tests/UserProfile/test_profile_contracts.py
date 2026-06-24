from __future__ import annotations

from datetime import datetime, timezone

from tldw_Server_API.app.core.UserProfiles.contracts import (
    EffectPolicy,
    EffectTiming,
    ProfileContractMode,
    ProfileUpdateCommand,
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
        effects=(),
    )

    assert plan.command.target_user_id == 7
    assert plan.mutations[0].operation == "upsert_override"
    assert plan.pre_commit_effects == ()
    assert plan.post_commit_effects == ()


def test_effect_policy_values_are_stable() -> None:
    assert EffectTiming.PRE_COMMIT.value == "pre_commit"
    assert EffectTiming.POST_COMMIT.value == "post_commit"
    assert EffectPolicy.REQUIRED.value == "required"
    assert EffectPolicy.BEST_EFFORT.value == "best_effort"
