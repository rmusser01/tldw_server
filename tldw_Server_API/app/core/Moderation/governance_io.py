"""
governance_io.py

JSON import/export for governance rules (self-monitoring rules,
supervised policies, governance policies).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB

_IO_FORMAT_VERSION = "1.0"


@dataclass
class GovernanceExportBundle:
    """Container for exported governance data."""
    format_version: str = _IO_FORMAT_VERSION
    exported_at: str = ""
    governance_policies: list[dict[str, Any]] = field(default_factory=list)
    self_monitoring_rules: list[dict[str, Any]] = field(default_factory=list)
    supervised_policies: list[dict[str, Any]] = field(default_factory=list)


def export_governance_rules(
    db: GuardianDB,
    user_id: str,
    include_governance_policies: bool = True,
    include_self_monitoring: bool = True,
    include_supervised: bool = False,
) -> GovernanceExportBundle:
    """Export governance rules for a user as a JSON-serializable bundle.

    Args:
        db: GuardianDB instance
        user_id: User whose rules to export
        include_governance_policies: Include governance policy groups
        include_self_monitoring: Include self-monitoring rules
        include_supervised: Include supervised policies (guardian use)

    Returns:
        GovernanceExportBundle with the exported data
    """
    bundle = GovernanceExportBundle(
        exported_at=datetime.now(timezone.utc).isoformat(),
    )

    if include_governance_policies:
        policies = db.list_governance_policies(str(user_id))
        for gp in policies:
            bundle.governance_policies.append(asdict(gp))

    if include_self_monitoring:
        rules = db.list_self_monitoring_rules(str(user_id))
        for rule in rules:
            bundle.self_monitoring_rules.append(asdict(rule))

    if include_supervised:
        # Export supervised policies across all relationships where user is guardian
        rels = db.get_relationships_for_guardian(str(user_id))
        for rel in rels:
            policies = db.list_policies_for_relationship(rel.id)
            for pol in policies:
                d = asdict(pol)
                d["_relationship_id"] = rel.id
                d["_dependent_user_id"] = rel.dependent_user_id
                bundle.supervised_policies.append(d)

    return bundle


def export_to_json(bundle: GovernanceExportBundle) -> str:
    """Serialize a GovernanceExportBundle to JSON string."""
    return json.dumps(asdict(bundle), indent=2, ensure_ascii=False)


def import_governance_rules(
    db: GuardianDB,
    user_id: str,
    bundle_data: dict[str, Any],
    merge_mode: str = "add",
) -> dict[str, int]:
    """Import governance rules from a bundle dict.

    Args:
        db: GuardianDB instance
        user_id: User to import rules for
        bundle_data: Parsed JSON dict (from GovernanceExportBundle)
        merge_mode: "add" (append) or "replace" (clear existing first)

    Returns:
        Dict with counts of imported items by type
    """
    counts: dict[str, int] = {
        "governance_policies": 0,
        "self_monitoring_rules": 0,
    }
    uid = str(user_id)

    if merge_mode == "replace":
        # Delete existing rules before importing
        existing_rules = db.list_self_monitoring_rules(uid)
        for rule in existing_rules:
            db.delete_self_monitoring_rule(rule.id)
        existing_policies = db.list_governance_policies(uid)
        for gp in existing_policies:
            db.delete_governance_policy(gp.id)

    # Import governance policies first (rules may reference them)
    id_remap: dict[str, str] = {}  # old_id -> new_id
    for gp_data in bundle_data.get("governance_policies", []):
        try:
            old_id = gp_data.get("id", "")
            new_gp = db.create_governance_policy(
                owner_user_id=uid,
                name=gp_data.get("name", "Imported Policy"),
                description=gp_data.get("description", ""),
                policy_mode=gp_data.get("policy_mode", "self"),
                scope_chat_types=gp_data.get("scope_chat_types", "all"),
                enabled=gp_data.get("enabled", True),
                schedule_start=gp_data.get("schedule_start"),
                schedule_end=gp_data.get("schedule_end"),
                schedule_days=gp_data.get("schedule_days"),
                schedule_timezone=gp_data.get("schedule_timezone", "UTC"),
                transparent=gp_data.get("transparent", False),
            )
            id_remap[old_id] = new_gp.id
            counts["governance_policies"] += 1
        except (ValueError, TypeError, KeyError):
            logger.warning("Failed to import governance policy")
            continue

    # Import self-monitoring rules
    for rule_data in bundle_data.get("self_monitoring_rules", []):
        try:
            # Remap governance_policy_id if present
            gov_id = rule_data.get("governance_policy_id")
            if gov_id and gov_id in id_remap:
                gov_id = id_remap[gov_id]
            elif gov_id:
                gov_id = None  # Original governance policy not in export

            db.create_self_monitoring_rule(
                user_id=uid,
                name=rule_data.get("name", "Imported Rule"),
                category=rule_data.get("category", ""),
                patterns=rule_data.get("patterns", []),
                pattern_type=rule_data.get("pattern_type", "literal"),
                except_patterns=rule_data.get("except_patterns", []),
                rule_type=rule_data.get("rule_type", "notify"),
                action=rule_data.get("action", "notify"),
                phase=rule_data.get("phase", "both"),
                severity=rule_data.get("severity", "info"),
                display_mode=rule_data.get("display_mode", "inline_banner"),
                block_message=rule_data.get("block_message"),
                context_note=rule_data.get("context_note"),
                notification_frequency=rule_data.get("notification_frequency", "once_per_conversation"),
                notification_channels=rule_data.get("notification_channels", ["in_app"]),
                webhook_url=rule_data.get("webhook_url"),
                trusted_contact_email=rule_data.get("trusted_contact_email"),
                crisis_resources_enabled=rule_data.get("crisis_resources_enabled", False),
                cooldown_minutes=rule_data.get("cooldown_minutes", 0),
                bypass_protection=rule_data.get("bypass_protection", "none"),
                bypass_partner_user_id=rule_data.get("bypass_partner_user_id"),
                escalation_session_threshold=rule_data.get("escalation_session_threshold", 0),
                escalation_session_action=rule_data.get("escalation_session_action"),
                escalation_window_days=rule_data.get("escalation_window_days", 0),
                escalation_window_threshold=rule_data.get("escalation_window_threshold", 0),
                escalation_window_action=rule_data.get("escalation_window_action"),
                min_context_length=rule_data.get("min_context_length", 0),
                governance_policy_id=gov_id,
                enabled=rule_data.get("enabled", True),
            )
            counts["self_monitoring_rules"] += 1
        except (ValueError, TypeError, KeyError):
            logger.warning("Failed to import self-monitoring rule")
            continue

    return counts
