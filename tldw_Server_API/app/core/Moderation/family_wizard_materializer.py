"""
Materialization helpers for Family Guardrails Wizard queued plans.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB


MATERIALIZATION_FAILURE_DETAIL = "Wizard plan materialization failed"


def materialize_pending_plans_for_relationship(
    db: GuardianDB,
    relationship_id: str,
    actor_user_id: str,
) -> dict[str, Any]:
    """Materialize queued wizard plans into supervised policies for one relationship."""
    queued_plans = db.list_pending_plans_for_relationship(relationship_id)
    if not queued_plans:
        return {"materialized_count": 0, "failed_count": 0, "policy_ids": []}

    materialized_count = 0
    failed_count = 0
    policy_ids: list[str] = []

    for plan in queued_plans:
        overrides = plan.get("overrides") or {}
        try:
            policy = db.create_policy(
                relationship_id=relationship_id,
                policy_type=overrides.get("policy_type", "block"),
                category=overrides.get("category", ""),
                pattern=overrides.get("pattern", ""),
                pattern_type=overrides.get("pattern_type", "literal"),
                action=overrides.get("action", "block"),
                phase=overrides.get("phase", "both"),
                severity=overrides.get("severity", "warning"),
                notify_guardian=overrides.get("notify_guardian", True),
                notify_context=overrides.get("notify_context", "topic_only"),
                message_to_dependent=overrides.get("message_to_dependent"),
                enabled=overrides.get("enabled", True),
                metadata={"source": "family_wizard", "template_id": plan.get("template_id")},
            )
            db.update_guardrail_plan_draft_status(
                plan_draft_id=plan["id"],
                status="active",
                materialized_policy_id=policy.id,
            )
            db.record_activation_run(
                household_draft_id=plan["household_draft_id"],
                relationship_id=relationship_id,
                dependent_user_id=plan["dependent_user_id"],
                plan_draft_id=plan["id"],
                status="active",
                detail="Queued wizard plan materialized after acceptance",
                metadata={"actor_user_id": actor_user_id, "policy_id": policy.id},
            )
            db.log_action(
                relationship_id=relationship_id,
                actor_user_id=actor_user_id,
                action="wizard_plan_materialized",
                target_user_id=plan["dependent_user_id"],
                policy_id=policy.id,
                detail=f"plan_draft_id={plan['id']}",
            )
            materialized_count += 1
            policy_ids.append(policy.id)
        except Exception:
            failed_count += 1
            error_detail = MATERIALIZATION_FAILURE_DETAIL
            logger.error(
                "Failed to materialize wizard plan {} for relationship {}",
                plan["id"],
                relationship_id,
            )
            db.update_guardrail_plan_draft_status(
                plan_draft_id=plan["id"],
                status="failed",
                failure_reason=error_detail,
            )
            db.record_activation_run(
                household_draft_id=plan["household_draft_id"],
                relationship_id=relationship_id,
                dependent_user_id=plan["dependent_user_id"],
                plan_draft_id=plan["id"],
                status="failed",
                detail=error_detail,
                metadata={"actor_user_id": actor_user_id},
            )

    return {
        "materialized_count": materialized_count,
        "failed_count": failed_count,
        "policy_ids": policy_ids,
    }
