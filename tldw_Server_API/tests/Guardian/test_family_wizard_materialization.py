from __future__ import annotations

import pytest

import tldw_Server_API.app.core.Moderation.family_wizard_materializer as materializer_module
from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
from tldw_Server_API.app.core.Moderation.family_wizard_materializer import (
    materialize_pending_plans_for_relationship,
)


@pytest.fixture
def db(tmp_path) -> GuardianDB:
    return GuardianDB(str(tmp_path / "family_wizard_materialization.db"))


def _create_pending_plan_for_relationship(db: GuardianDB) -> dict[str, str]:
    draft_id = db.create_household_draft(owner_user_id="guardian-1", mode="family", name="Home")
    guardian_member_id = db.add_household_member_draft(
        household_draft_id=draft_id,
        role="guardian",
        display_name="Parent",
        user_id="guardian-1",
    )
    dependent_member_id = db.add_household_member_draft(
        household_draft_id=draft_id,
        role="dependent",
        display_name="Child",
        user_id="child-1",
    )

    relationship = db.create_relationship(
        guardian_user_id="guardian-1",
        dependent_user_id="child-1",
        relationship_type="parent",
    )
    relationship_draft_id = db.create_relationship_draft(
        household_draft_id=draft_id,
        guardian_member_draft_id=guardian_member_id,
        dependent_member_draft_id=dependent_member_id,
        relationship_type="parent",
    )
    db.link_relationship_draft(
        relationship_draft_id=relationship_draft_id,
        relationship_id=relationship.id,
        status=relationship.status,
    )

    plan_id = db.create_guardrail_plan_draft(
        household_draft_id=draft_id,
        dependent_user_id="child-1",
        relationship_draft_id=relationship_draft_id,
        template_id="default-child-safe",
        overrides={"category": "explicit_content", "action": "block"},
    )
    assert db.accept_relationship(relationship.id) is True

    return {
        "draft_id": draft_id,
        "relationship_id": relationship.id,
        "plan_id": plan_id,
    }


def test_acceptance_materializes_queued_plans(db: GuardianDB) -> None:
    draft_id = db.create_household_draft(owner_user_id="guardian-1", mode="family", name="Home")
    guardian_member_id = db.add_household_member_draft(
        household_draft_id=draft_id,
        role="guardian",
        display_name="Parent",
        user_id="guardian-1",
    )
    dependent_member_id = db.add_household_member_draft(
        household_draft_id=draft_id,
        role="dependent",
        display_name="Child",
        user_id="child-1",
    )

    relationship = db.create_relationship(
        guardian_user_id="guardian-1",
        dependent_user_id="child-1",
        relationship_type="parent",
    )
    relationship_draft_id = db.create_relationship_draft(
        household_draft_id=draft_id,
        guardian_member_draft_id=guardian_member_id,
        dependent_member_draft_id=dependent_member_id,
        relationship_type="parent",
    )
    db.link_relationship_draft(
        relationship_draft_id=relationship_draft_id,
        relationship_id=relationship.id,
        status=relationship.status,
    )

    plan_id = db.create_guardrail_plan_draft(
        household_draft_id=draft_id,
        dependent_user_id="child-1",
        relationship_draft_id=relationship_draft_id,
        template_id="default-child-safe",
        overrides={"category": "explicit_content", "action": "block"},
    )

    assert db.accept_relationship(relationship.id) is True
    result = materialize_pending_plans_for_relationship(
        db=db,
        relationship_id=relationship.id,
        actor_user_id="child-1",
    )

    policies = db.list_policies_for_relationship(relationship.id)
    runs = db.list_activation_runs(draft_id)
    updated_plan = db.get_guardrail_plan_draft(plan_id)

    assert result["materialized_count"] == 1
    assert len(policies) == 1
    assert updated_plan is not None
    assert updated_plan["status"] == "active"
    assert any(run["plan_draft_id"] == plan_id and run["status"] == "active" for run in runs)


def test_materializer_noops_when_no_queued_plans(db: GuardianDB) -> None:
    relationship = db.create_relationship(
        guardian_user_id="guardian-1",
        dependent_user_id="child-1",
        relationship_type="parent",
    )
    db.accept_relationship(relationship.id)

    result = materialize_pending_plans_for_relationship(
        db=db,
        relationship_id=relationship.id,
        actor_user_id="child-1",
    )

    assert result["materialized_count"] == 0
    assert result["failed_count"] == 0


def test_materializer_failure_sanitizes_persisted_and_logged_details(
    monkeypatch: pytest.MonkeyPatch,
    db: GuardianDB,
) -> None:
    ids = _create_pending_plan_for_relationship(db)

    def fail_create_policy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("materialization failed at /private/family-wizard.db")

    monkeypatch.setattr(db, "create_policy", fail_create_policy)

    messages: list[str] = []
    sink_id = materializer_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = materialize_pending_plans_for_relationship(
            db=db,
            relationship_id=ids["relationship_id"],
            actor_user_id="child-1",
        )
    finally:
        materializer_module.logger.remove(sink_id)

    updated_plan = db.get_guardrail_plan_draft(ids["plan_id"])
    runs = db.list_activation_runs(ids["draft_id"])
    failed_run = next(run for run in runs if run["plan_draft_id"] == ids["plan_id"])
    joined_logs = "\n".join(messages)

    assert result["materialized_count"] == 0
    assert result["failed_count"] == 1
    assert updated_plan is not None
    assert updated_plan["status"] == "failed"
    assert updated_plan["failure_reason"] == "Wizard plan materialization failed"
    assert failed_run["status"] == "failed"
    assert failed_run["detail"] == "Wizard plan materialization failed"
    assert "family-wizard.db" not in joined_logs
    assert "materialization failed at" not in joined_logs
