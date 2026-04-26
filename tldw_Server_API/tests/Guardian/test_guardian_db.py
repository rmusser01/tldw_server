"""Comprehensive tests for GuardianDB CRUD operations.

Pure SQLite -- no network, no mocking required.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone, timedelta

import pytest

from tldw_Server_API.app.core.DB_Management.Guardian_DB import (
    GuardianDB,
    GuardianRelationship,
    SupervisedPolicy,
    SupervisionAuditEntry,
    GovernancePolicy,
    SelfMonitoringRule,
    SelfMonitoringAlert,
    EscalationState,
)


@pytest.fixture
def db(tmp_path):
    return GuardianDB(str(tmp_path / "test_guardian.db"))


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    def test_db_initializes_without_error(self, tmp_path):
        db = GuardianDB(str(tmp_path / "fresh.db"))
        assert db.db_path.endswith("fresh.db")

    def test_db_preserves_sqlite_in_memory_sentinel(self):
        db = GuardianDB(":memory:")
        conn = db._connect()
        try:
            assert db.db_path == ":memory:"
            assert conn.execute("SELECT 1").fetchone()[0] == 1
        finally:
            conn.close()

    def test_double_init_is_idempotent(self, tmp_path):
        path = str(tmp_path / "dup.db")
        GuardianDB(path)
        GuardianDB(path)

    def test_legacy_schema_without_governance_policy_id_migrates_cleanly(self, tmp_path):
        path = tmp_path / "legacy_guardian.db"
        conn = sqlite3.connect(path)
        try:
            # Legacy shape intentionally omits supervised_policies.governance_policy_id.
            conn.executescript(
                """
                CREATE TABLE guardian_relationships (
                    id TEXT PRIMARY KEY,
                    guardian_user_id TEXT NOT NULL,
                    dependent_user_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL DEFAULT 'parent',
                    status TEXT NOT NULL DEFAULT 'pending',
                    consent_given_by_dependent INTEGER NOT NULL DEFAULT 0,
                    consent_given_at TEXT,
                    dependent_visible INTEGER NOT NULL DEFAULT 1,
                    dissolution_reason TEXT,
                    dissolved_at TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(guardian_user_id, dependent_user_id)
                );

                CREATE TABLE supervised_policies (
                    id TEXT PRIMARY KEY,
                    relationship_id TEXT NOT NULL,
                    policy_type TEXT NOT NULL DEFAULT 'block',
                    category TEXT NOT NULL DEFAULT '',
                    pattern TEXT NOT NULL DEFAULT '',
                    pattern_type TEXT NOT NULL DEFAULT 'literal',
                    action TEXT NOT NULL DEFAULT 'block',
                    phase TEXT NOT NULL DEFAULT 'both',
                    severity TEXT NOT NULL DEFAULT 'warning',
                    notify_guardian INTEGER NOT NULL DEFAULT 1,
                    notify_context TEXT NOT NULL DEFAULT 'topic_only',
                    message_to_dependent TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (relationship_id) REFERENCES guardian_relationships(id)
                );
                """
            )
        finally:
            conn.close()

        # Should not raise OperationalError during schema bootstrap.
        GuardianDB(str(path))

        conn = sqlite3.connect(path)
        try:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(supervised_policies)").fetchall()
            }
            assert "governance_policy_id" in columns
        finally:
            conn.close()

    def test_legacy_family_wizard_tables_migrate_before_optional_indexes(self, tmp_path):
        path = tmp_path / "legacy_family_wizard_guardian.db"
        conn = sqlite3.connect(path)
        try:
            conn.executescript(
                """
                CREATE TABLE guardian_household_drafts (
                    id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'family',
                    status TEXT NOT NULL DEFAULT 'draft',
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE guardian_household_member_drafts (
                    id TEXT PRIMARY KEY,
                    household_draft_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    user_id TEXT,
                    email TEXT,
                    invite_required INTEGER NOT NULL DEFAULT 1,
                    invite_status TEXT NOT NULL DEFAULT 'pending',
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (household_draft_id) REFERENCES guardian_household_drafts(id) ON DELETE CASCADE
                );

                CREATE TABLE guardian_relationship_drafts (
                    id TEXT PRIMARY KEY,
                    household_draft_id TEXT NOT NULL,
                    guardian_member_draft_id TEXT NOT NULL,
                    dependent_member_draft_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL DEFAULT 'parent',
                    dependent_visible INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'pending',
                    relationship_id TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (household_draft_id) REFERENCES guardian_household_drafts(id) ON DELETE CASCADE,
                    FOREIGN KEY (guardian_member_draft_id) REFERENCES guardian_household_member_drafts(id) ON DELETE CASCADE,
                    FOREIGN KEY (dependent_member_draft_id) REFERENCES guardian_household_member_drafts(id) ON DELETE CASCADE,
                    UNIQUE(household_draft_id, guardian_member_draft_id, dependent_member_draft_id)
                );

                CREATE TABLE guardian_guardrail_plan_drafts (
                    id TEXT PRIMARY KEY,
                    household_draft_id TEXT NOT NULL,
                    dependent_user_id TEXT NOT NULL,
                    relationship_draft_id TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    overrides TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'queued',
                    materialized_policy_id TEXT,
                    failure_reason TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (household_draft_id) REFERENCES guardian_household_drafts(id) ON DELETE CASCADE,
                    FOREIGN KEY (relationship_draft_id) REFERENCES guardian_relationship_drafts(id) ON DELETE CASCADE
                );
                """
            )
        finally:
            conn.close()

        GuardianDB(str(path))

        conn = sqlite3.connect(path)
        try:
            member_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(guardian_household_member_drafts)").fetchall()
            }
            assert "account_mode" in member_columns
            assert "provisioning_status" in member_columns

            plan_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(guardian_guardrail_plan_drafts)").fetchall()
            }
            assert "dependent_member_draft_id" in plan_columns

            member_indexes = {
                row[1] for row in conn.execute("PRAGMA index_list(guardian_household_member_drafts)").fetchall()
            }
            assert "idx_ghmd_account_mode" in member_indexes

            plan_indexes = {
                row[1] for row in conn.execute("PRAGMA index_list(guardian_guardrail_plan_drafts)").fetchall()
            }
            assert "idx_ggpd_member" in plan_indexes
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Guardian Relationships
# ---------------------------------------------------------------------------


class TestRelationshipCRUD:
    def test_create_relationship(self, db):
        rel = db.create_relationship("guardian1", "dependent1")
        assert isinstance(rel, GuardianRelationship)
        assert rel.guardian_user_id == "guardian1"
        assert rel.dependent_user_id == "dependent1"
        assert rel.status == "pending"
        assert rel.relationship_type == "parent"
        assert rel.dependent_visible is True
        assert rel.consent_given_by_dependent is False

    def test_create_relationship_with_metadata(self, db):
        meta = {"note": "test relationship"}
        rel = db.create_relationship(
            "g1", "d1",
            relationship_type="legal_guardian",
            dependent_visible=False,
            metadata=meta,
        )
        assert rel.relationship_type == "legal_guardian"
        assert rel.dependent_visible is False
        assert rel.metadata == meta

    def test_get_relationship(self, db):
        rel = db.create_relationship("g1", "d1")
        fetched = db.get_relationship(rel.id)
        assert fetched is not None
        assert fetched.id == rel.id
        assert fetched.guardian_user_id == "g1"

    def test_get_relationship_nonexistent(self, db):
        assert db.get_relationship("nonexistent") is None

    def test_get_relationships_for_guardian(self, db):
        db.create_relationship("g1", "d1")
        db.create_relationship("g1", "d2")
        db.create_relationship("g2", "d3")
        results = db.get_relationships_for_guardian("g1")
        assert len(results) == 2

    def test_get_relationships_for_guardian_with_status_filter(self, db):
        rel = db.create_relationship("g1", "d1")
        db.accept_relationship(rel.id)
        db.create_relationship("g1", "d2")
        active = db.get_relationships_for_guardian("g1", status="active")
        assert len(active) == 1
        assert active[0].id == rel.id

    def test_get_relationships_for_dependent(self, db):
        db.create_relationship("g1", "d1")
        db.create_relationship("g2", "d1")
        results = db.get_relationships_for_dependent("d1")
        assert len(results) == 2

    def test_get_relationships_for_dependent_with_status_filter(self, db):
        rel = db.create_relationship("g1", "d1")
        db.create_relationship("g2", "d1")
        db.accept_relationship(rel.id)
        active = db.get_relationships_for_dependent("d1", status="active")
        assert len(active) == 1

    def test_accept_relationship(self, db):
        rel = db.create_relationship("g1", "d1")
        assert db.accept_relationship(rel.id) is True
        fetched = db.get_relationship(rel.id)
        assert fetched.status == "active"
        assert fetched.consent_given_by_dependent is True
        assert fetched.consent_given_at is not None

    def test_accept_already_active_returns_false(self, db):
        rel = db.create_relationship("g1", "d1")
        db.accept_relationship(rel.id)
        assert db.accept_relationship(rel.id) is False

    def test_suspend_relationship(self, db):
        rel = db.create_relationship("g1", "d1")
        db.accept_relationship(rel.id)
        assert db.suspend_relationship(rel.id) is True
        fetched = db.get_relationship(rel.id)
        assert fetched.status == "suspended"

    def test_suspend_non_active_returns_false(self, db):
        rel = db.create_relationship("g1", "d1")
        assert db.suspend_relationship(rel.id) is False

    def test_reactivate_relationship(self, db):
        rel = db.create_relationship("g1", "d1")
        db.accept_relationship(rel.id)
        db.suspend_relationship(rel.id)
        assert db.reactivate_relationship(rel.id) is True
        fetched = db.get_relationship(rel.id)
        assert fetched.status == "active"

    def test_reactivate_non_suspended_returns_false(self, db):
        rel = db.create_relationship("g1", "d1")
        db.accept_relationship(rel.id)
        assert db.reactivate_relationship(rel.id) is False

    def test_dissolve_relationship(self, db):
        rel = db.create_relationship("g1", "d1")
        db.accept_relationship(rel.id)
        assert db.dissolve_relationship(rel.id, reason="test") is True
        fetched = db.get_relationship(rel.id)
        assert fetched.status == "dissolved"
        assert fetched.dissolution_reason == "test"
        assert fetched.dissolved_at is not None

    def test_dissolve_pending(self, db):
        rel = db.create_relationship("g1", "d1")
        assert db.dissolve_relationship(rel.id) is True
        assert db.get_relationship(rel.id).status == "dissolved"

    def test_dissolve_already_dissolved_returns_false(self, db):
        rel = db.create_relationship("g1", "d1")
        db.dissolve_relationship(rel.id)
        assert db.dissolve_relationship(rel.id) is False

    def test_is_guardian_of_active(self, db):
        rel = db.create_relationship("g1", "d1")
        assert db.is_guardian_of("g1", "d1") is False
        db.accept_relationship(rel.id)
        assert db.is_guardian_of("g1", "d1") is True

    def test_is_guardian_of_suspended(self, db):
        rel = db.create_relationship("g1", "d1")
        db.accept_relationship(rel.id)
        db.suspend_relationship(rel.id)
        assert db.is_guardian_of("g1", "d1") is False

    def test_full_lifecycle(self, db):
        rel = db.create_relationship("g1", "d1")
        assert rel.status == "pending"
        db.accept_relationship(rel.id)
        assert db.get_relationship(rel.id).status == "active"
        db.suspend_relationship(rel.id)
        assert db.get_relationship(rel.id).status == "suspended"
        db.reactivate_relationship(rel.id)
        assert db.get_relationship(rel.id).status == "active"
        db.dissolve_relationship(rel.id, reason="done")
        assert db.get_relationship(rel.id).status == "dissolved"


class TestRelationshipValidation:
    def test_self_relationship_rejected(self, db):
        with pytest.raises(ValueError, match="same user"):
            db.create_relationship("user1", "user1")

    def test_duplicate_relationship_rejected(self, db):
        db.create_relationship("g1", "d1")
        with pytest.raises(ValueError, match="already exists"):
            db.create_relationship("g1", "d1")


# ---------------------------------------------------------------------------
# Supervised Policies
# ---------------------------------------------------------------------------


class TestPolicyCRUD:
    def test_create_policy(self, db):
        rel = db.create_relationship("g1", "d1")
        pol = db.create_policy(
            relationship_id=rel.id,
            category="explicit_content",
            pattern="bad_word",
        )
        assert isinstance(pol, SupervisedPolicy)
        assert pol.relationship_id == rel.id
        assert pol.action == "block"
        assert pol.phase == "both"
        assert pol.severity == "warning"
        assert pol.enabled is True

    def test_get_policy(self, db):
        rel = db.create_relationship("g1", "d1")
        pol = db.create_policy(rel.id, category="test")
        fetched = db.get_policy(pol.id)
        assert fetched is not None
        assert fetched.category == "test"

    def test_get_policy_nonexistent(self, db):
        assert db.get_policy("no_such_id") is None

    def test_list_policies_for_relationship(self, db):
        rel = db.create_relationship("g1", "d1")
        db.create_policy(rel.id, category="a")
        db.create_policy(rel.id, category="b", enabled=False)
        all_policies = db.list_policies_for_relationship(rel.id)
        assert len(all_policies) == 2
        enabled_only = db.list_policies_for_relationship(rel.id, enabled_only=True)
        assert len(enabled_only) == 1
        assert enabled_only[0].category == "a"

    def test_update_policy(self, db):
        rel = db.create_relationship("g1", "d1")
        pol = db.create_policy(rel.id, category="old")
        assert db.update_policy(pol.id, category="new", action="notify") is True
        fetched = db.get_policy(pol.id)
        assert fetched.category == "new"
        assert fetched.action == "notify"

    def test_update_policy_nonexistent(self, db):
        assert db.update_policy("nope", category="x") is False

    def test_update_policy_no_valid_fields(self, db):
        rel = db.create_relationship("g1", "d1")
        pol = db.create_policy(rel.id)
        assert db.update_policy(pol.id, unknown_field="val") is False

    def test_delete_policy(self, db):
        rel = db.create_relationship("g1", "d1")
        pol = db.create_policy(rel.id)
        assert db.delete_policy(pol.id) is True
        assert db.get_policy(pol.id) is None

    def test_delete_policy_nonexistent(self, db):
        assert db.delete_policy("no_such") is False

    def test_list_active_policies_for_dependent(self, db):
        rel_active = db.create_relationship("g1", "d1")
        db.accept_relationship(rel_active.id)
        db.create_policy(rel_active.id, category="active_policy")
        db.create_policy(rel_active.id, category="disabled", enabled=False)

        rel_pending = db.create_relationship("g2", "d1")
        db.create_policy(rel_pending.id, category="pending_policy")

        active = db.list_active_policies_for_dependent("d1")
        assert len(active) == 1
        assert active[0].category == "active_policy"

    def test_create_policy_with_metadata(self, db):
        rel = db.create_relationship("g1", "d1")
        pol = db.create_policy(
            rel.id,
            metadata={"reason": "safety"},
            message_to_dependent="This content is restricted.",
        )
        fetched = db.get_policy(pol.id)
        assert fetched.metadata == {"reason": "safety"}
        assert fetched.message_to_dependent == "This content is restricted."


class TestPolicyValidation:
    def test_invalid_action(self, db):
        rel = db.create_relationship("g1", "d1")
        with pytest.raises(ValueError, match="Invalid action"):
            db.create_policy(rel.id, action="invalid")

    def test_invalid_phase(self, db):
        rel = db.create_relationship("g1", "d1")
        with pytest.raises(ValueError, match="Invalid phase"):
            db.create_policy(rel.id, phase="invalid")

    def test_invalid_severity(self, db):
        rel = db.create_relationship("g1", "d1")
        with pytest.raises(ValueError, match="Invalid severity"):
            db.create_policy(rel.id, severity="invalid")

    def test_invalid_notify_context(self, db):
        rel = db.create_relationship("g1", "d1")
        with pytest.raises(ValueError, match="Invalid notify_context"):
            db.create_policy(rel.id, notify_context="invalid")

    def test_update_invalid_action(self, db):
        rel = db.create_relationship("g1", "d1")
        pol = db.create_policy(rel.id)
        with pytest.raises(ValueError, match="Invalid action"):
            db.update_policy(pol.id, action="bad")

    def test_update_invalid_phase(self, db):
        rel = db.create_relationship("g1", "d1")
        pol = db.create_policy(rel.id)
        with pytest.raises(ValueError, match="Invalid phase"):
            db.update_policy(pol.id, phase="bad")

    def test_update_invalid_severity(self, db):
        rel = db.create_relationship("g1", "d1")
        pol = db.create_policy(rel.id)
        with pytest.raises(ValueError, match="Invalid severity"):
            db.update_policy(pol.id, severity="bad")


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_log_action(self, db):
        rel = db.create_relationship("g1", "d1")
        entry_id = db.log_action(
            relationship_id=rel.id,
            actor_user_id="g1",
            action="policy_created",
            detail="created test policy",
        )
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    def test_get_audit_log(self, db):
        rel = db.create_relationship("g1", "d1")
        db.log_action(rel.id, "g1", "a1", detail="first")
        db.log_action(rel.id, "g1", "a2", detail="second")
        entries = db.get_audit_log(rel.id)
        assert len(entries) == 2
        assert all(isinstance(e, SupervisionAuditEntry) for e in entries)

    def test_get_audit_log_empty(self, db):
        rel = db.create_relationship("g1", "d1")
        assert db.get_audit_log(rel.id) == []

    def test_count_audit_entries(self, db):
        rel = db.create_relationship("g1", "d1")
        assert db.count_audit_entries(rel.id) == 0
        db.log_action(rel.id, "g1", "test")
        db.log_action(rel.id, "g1", "test2")
        assert db.count_audit_entries(rel.id) == 2

    def test_audit_log_pagination(self, db):
        rel = db.create_relationship("g1", "d1")
        for i in range(5):
            db.log_action(rel.id, "g1", f"action_{i}")
        page1 = db.get_audit_log(rel.id, limit=2, offset=0)
        assert len(page1) == 2
        page2 = db.get_audit_log(rel.id, limit=2, offset=2)
        assert len(page2) == 2
        page3 = db.get_audit_log(rel.id, limit=2, offset=4)
        assert len(page3) == 1
        all_ids = {e.id for e in page1 + page2 + page3}
        assert len(all_ids) == 5

    def test_audit_log_with_metadata(self, db):
        rel = db.create_relationship("g1", "d1")
        db.log_action(
            rel.id, "g1", "test",
            target_user_id="d1",
            policy_id="pol_123",
            metadata={"key": "value"},
        )
        entries = db.get_audit_log(rel.id)
        assert len(entries) == 1
        assert entries[0].target_user_id == "d1"
        assert entries[0].policy_id == "pol_123"
        assert entries[0].metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# Governance Policies
# ---------------------------------------------------------------------------


class TestGovernancePolicyCRUD:
    def test_create_governance_policy(self, db):
        gp = db.create_governance_policy("user1", "Safety Rules")
        assert isinstance(gp, GovernancePolicy)
        assert gp.owner_user_id == "user1"
        assert gp.name == "Safety Rules"
        assert gp.policy_mode == "guardian"
        assert gp.enabled is True

    def test_create_governance_policy_self_mode(self, db):
        gp = db.create_governance_policy("user1", "Self Rules", policy_mode="self")
        assert gp.policy_mode == "self"

    def test_create_governance_policy_all_fields(self, db):
        gp = db.create_governance_policy(
            "user1", "Full",
            description="desc",
            policy_mode="self",
            scope_chat_types="regular,character",
            enabled=False,
            schedule_start="09:00",
            schedule_end="17:00",
            schedule_days="mon,tue,wed",
            schedule_timezone="US/Eastern",
            transparent=True,
            metadata={"version": 2},
        )
        fetched = db.get_governance_policy(gp.id)
        assert fetched.description == "desc"
        assert fetched.scope_chat_types == "regular,character"
        assert fetched.enabled is False
        assert fetched.schedule_start == "09:00"
        assert fetched.schedule_end == "17:00"
        assert fetched.schedule_days == "mon,tue,wed"
        assert fetched.schedule_timezone == "US/Eastern"
        assert fetched.transparent is True
        assert fetched.metadata == {"version": 2}

    def test_get_governance_policy(self, db):
        gp = db.create_governance_policy("user1", "Test")
        fetched = db.get_governance_policy(gp.id)
        assert fetched is not None
        assert fetched.name == "Test"

    def test_get_governance_policy_nonexistent(self, db):
        assert db.get_governance_policy("nope") is None

    def test_list_governance_policies(self, db):
        db.create_governance_policy("user1", "A")
        db.create_governance_policy("user1", "B", policy_mode="self")
        db.create_governance_policy("user2", "C")
        all_user1 = db.list_governance_policies("user1")
        assert len(all_user1) == 2
        self_only = db.list_governance_policies("user1", policy_mode="self")
        assert len(self_only) == 1
        assert self_only[0].name == "B"

    def test_delete_governance_policy(self, db):
        gp = db.create_governance_policy("user1", "ToDelete")
        assert db.delete_governance_policy(gp.id) is True
        assert db.get_governance_policy(gp.id) is None

    def test_delete_governance_policy_nonexistent(self, db):
        assert db.delete_governance_policy("nope") is False


class TestGovernancePolicyValidation:
    def test_invalid_policy_mode(self, db):
        with pytest.raises(ValueError, match="Invalid policy_mode"):
            db.create_governance_policy("user1", "Bad", policy_mode="invalid")


# ---------------------------------------------------------------------------
# Self-Monitoring Rules
# ---------------------------------------------------------------------------


class TestSelfMonitoringRuleCRUD:
    def test_create_rule(self, db):
        rule = db.create_self_monitoring_rule("user1", "No gambling")
        assert isinstance(rule, SelfMonitoringRule)
        assert rule.user_id == "user1"
        assert rule.name == "No gambling"
        assert rule.action == "notify"
        assert rule.enabled is True

    def test_get_rule(self, db):
        rule = db.create_self_monitoring_rule("user1", "Test")
        fetched = db.get_self_monitoring_rule(rule.id)
        assert fetched is not None
        assert fetched.name == "Test"

    def test_get_rule_nonexistent(self, db):
        assert db.get_self_monitoring_rule("nope") is None

    def test_list_rules(self, db):
        db.create_self_monitoring_rule("user1", "A", category="health")
        db.create_self_monitoring_rule("user1", "B", category="finance", enabled=False)
        db.create_self_monitoring_rule("user2", "C")
        all_user1 = db.list_self_monitoring_rules("user1")
        assert len(all_user1) == 2
        enabled = db.list_self_monitoring_rules("user1", enabled_only=True)
        assert len(enabled) == 1
        by_cat = db.list_self_monitoring_rules("user1", category="finance")
        assert len(by_cat) == 1
        assert by_cat[0].name == "B"

    def test_update_rule(self, db):
        rule = db.create_self_monitoring_rule("user1", "Old Name")
        assert db.update_self_monitoring_rule(rule.id, name="New Name", action="block") is True
        fetched = db.get_self_monitoring_rule(rule.id)
        assert fetched.name == "New Name"
        assert fetched.action == "block"

    def test_update_rule_nonexistent(self, db):
        assert db.update_self_monitoring_rule("nope", name="x") is False

    def test_update_rule_no_valid_fields(self, db):
        rule = db.create_self_monitoring_rule("user1", "X")
        assert db.update_self_monitoring_rule(rule.id, garbage="y") is False

    def test_delete_rule(self, db):
        rule = db.create_self_monitoring_rule("user1", "ToDelete")
        assert db.delete_self_monitoring_rule(rule.id) is True
        assert db.get_self_monitoring_rule(rule.id) is None

    def test_delete_rule_nonexistent(self, db):
        assert db.delete_self_monitoring_rule("nope") is False


class TestSelfMonitoringRuleJsonFields:
    def test_patterns_roundtrip(self, db):
        rule = db.create_self_monitoring_rule(
            "user1", "Pat",
            patterns=["gambling", "casino", "slots"],
            except_patterns=["board game"],
            notification_channels=["in_app", "email"],
        )
        fetched = db.get_self_monitoring_rule(rule.id)
        assert fetched.patterns == ["gambling", "casino", "slots"]
        assert fetched.except_patterns == ["board game"]
        assert fetched.notification_channels == ["in_app", "email"]

    def test_empty_patterns_default(self, db):
        rule = db.create_self_monitoring_rule("user1", "Empty")
        fetched = db.get_self_monitoring_rule(rule.id)
        assert fetched.patterns == []
        assert fetched.except_patterns == []
        assert fetched.notification_channels == ["in_app"]

    def test_update_json_list_fields(self, db):
        rule = db.create_self_monitoring_rule("user1", "Upd")
        db.update_self_monitoring_rule(
            rule.id,
            patterns=["new_pat"],
            except_patterns=["exception"],
            notification_channels=["webhook"],
        )
        fetched = db.get_self_monitoring_rule(rule.id)
        assert fetched.patterns == ["new_pat"]
        assert fetched.except_patterns == ["exception"]
        assert fetched.notification_channels == ["webhook"]


class TestSelfMonitoringRuleValidation:
    def test_invalid_action(self, db):
        with pytest.raises(ValueError, match="Invalid action"):
            db.create_self_monitoring_rule("user1", "Bad", action="invalid")

    def test_invalid_phase(self, db):
        with pytest.raises(ValueError, match="Invalid phase"):
            db.create_self_monitoring_rule("user1", "Bad", phase="invalid")

    def test_invalid_severity(self, db):
        with pytest.raises(ValueError, match="Invalid severity"):
            db.create_self_monitoring_rule("user1", "Bad", severity="invalid")

    def test_update_invalid_action(self, db):
        rule = db.create_self_monitoring_rule("user1", "OK")
        with pytest.raises(ValueError, match="Invalid action"):
            db.update_self_monitoring_rule(rule.id, action="invalid")

    def test_update_invalid_phase(self, db):
        rule = db.create_self_monitoring_rule("user1", "OK")
        with pytest.raises(ValueError, match="Invalid phase"):
            db.update_self_monitoring_rule(rule.id, phase="invalid")

    def test_update_invalid_severity(self, db):
        rule = db.create_self_monitoring_rule("user1", "OK")
        with pytest.raises(ValueError, match="Invalid severity"):
            db.update_self_monitoring_rule(rule.id, severity="invalid")


class TestSelfMonitoringRuleDeletionCascade:
    def test_deleting_rule_removes_alerts_and_escalation(self, db):
        rule = db.create_self_monitoring_rule("user1", "Cascade")
        db.create_self_monitoring_alert("user1", rule.id, rule_name="Cascade")
        db.create_self_monitoring_alert("user1", rule.id, rule_name="Cascade")
        db.upsert_escalation_state(rule.id, "user1", session_trigger_count=3)

        assert len(db.list_self_monitoring_alerts("user1", rule_id=rule.id)) == 2
        assert db.get_escalation_state(rule.id, "user1") is not None

        db.delete_self_monitoring_rule(rule.id)

        assert db.get_self_monitoring_rule(rule.id) is None
        assert len(db.list_self_monitoring_alerts("user1", rule_id=rule.id)) == 0
        assert db.get_escalation_state(rule.id, "user1") is None


class TestSelfMonitoringRuleGovernanceLink:
    def test_rule_with_governance_policy_id(self, db):
        gp = db.create_governance_policy("user1", "Parent Policy", policy_mode="self")
        rule = db.create_self_monitoring_rule(
            "user1", "Linked",
            governance_policy_id=gp.id,
        )
        fetched = db.get_self_monitoring_rule(rule.id)
        assert fetched.governance_policy_id == gp.id


# ---------------------------------------------------------------------------
# Self-Monitoring Alerts
# ---------------------------------------------------------------------------


class TestSelfMonitoringAlertCRUD:
    def test_create_alert(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        alert = db.create_self_monitoring_alert(
            user_id="user1",
            rule_id=rule.id,
            rule_name="R1",
            category="health",
            severity="warning",
            matched_pattern="gambling",
            context_snippet="I want to gamble",
            conversation_id="conv_1",
            phase="input",
            action_taken="blocked",
        )
        assert isinstance(alert, SelfMonitoringAlert)
        assert alert.user_id == "user1"
        assert alert.is_read is False

    def test_list_alerts(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        db.create_self_monitoring_alert("user1", rule.id)
        db.create_self_monitoring_alert("user1", rule.id)
        db.create_self_monitoring_alert("user2", rule.id)
        alerts = db.list_self_monitoring_alerts("user1")
        assert len(alerts) == 2

    def test_list_alerts_by_rule(self, db):
        r1 = db.create_self_monitoring_rule("user1", "R1")
        r2 = db.create_self_monitoring_rule("user1", "R2")
        db.create_self_monitoring_alert("user1", r1.id)
        db.create_self_monitoring_alert("user1", r2.id)
        alerts = db.list_self_monitoring_alerts("user1", rule_id=r1.id)
        assert len(alerts) == 1

    def test_list_alerts_pagination(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        for _ in range(5):
            db.create_self_monitoring_alert("user1", rule.id)
        page1 = db.list_self_monitoring_alerts("user1", limit=2, offset=0)
        page2 = db.list_self_monitoring_alerts("user1", limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    def test_list_alerts_unread_only(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        a1 = db.create_self_monitoring_alert("user1", rule.id)
        db.create_self_monitoring_alert("user1", rule.id)
        db.mark_alerts_read("user1", [a1.id])
        unread = db.list_self_monitoring_alerts("user1", unread_only=True)
        assert len(unread) == 1


class TestAlertReadManagement:
    def test_mark_alerts_read(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        a1 = db.create_self_monitoring_alert("user1", rule.id)
        a2 = db.create_self_monitoring_alert("user1", rule.id)
        count = db.mark_alerts_read("user1", [a1.id, a2.id])
        assert count == 2

    def test_mark_alerts_read_empty_list(self, db):
        assert db.mark_alerts_read("user1", []) == 0

    def test_mark_alerts_read_wrong_user(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        a = db.create_self_monitoring_alert("user1", rule.id)
        count = db.mark_alerts_read("user2", [a.id])
        assert count == 0

    def test_count_unread_alerts(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        assert db.count_unread_alerts("user1") == 0
        a1 = db.create_self_monitoring_alert("user1", rule.id)
        db.create_self_monitoring_alert("user1", rule.id)
        assert db.count_unread_alerts("user1") == 2
        db.mark_alerts_read("user1", [a1.id])
        assert db.count_unread_alerts("user1") == 1


class TestAlertDedup:
    def test_has_recent_alert_basic(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        assert db.has_recent_alert("user1", rule.id) is False
        db.create_self_monitoring_alert("user1", rule.id)
        assert db.has_recent_alert("user1", rule.id) is True

    def test_has_recent_alert_with_conversation_id(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        db.create_self_monitoring_alert("user1", rule.id, conversation_id="conv_a")
        assert db.has_recent_alert("user1", rule.id, conversation_id="conv_a") is True
        assert db.has_recent_alert("user1", rule.id, conversation_id="conv_b") is False

    def test_has_recent_alert_with_since_iso(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        db.create_self_monitoring_alert("user1", rule.id)
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        assert db.has_recent_alert("user1", rule.id, since_iso=future) is False
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        assert db.has_recent_alert("user1", rule.id, since_iso=past) is True


# ---------------------------------------------------------------------------
# Escalation State
# ---------------------------------------------------------------------------


class TestEscalationState:
    def test_upsert_and_get(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        db.upsert_escalation_state(
            rule.id, "user1",
            session_id="sess_1",
            session_trigger_count=3,
            window_trigger_count=10,
            current_escalated_action="block",
            escalated_at="2025-01-01T00:00:00+00:00",
        )
        state = db.get_escalation_state(rule.id, "user1")
        assert isinstance(state, EscalationState)
        assert state.session_trigger_count == 3
        assert state.window_trigger_count == 10
        assert state.current_escalated_action == "block"
        assert state.session_id == "sess_1"

    def test_get_nonexistent(self, db):
        assert db.get_escalation_state("no_rule", "no_user") is None

    def test_upsert_updates_existing(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        db.upsert_escalation_state(rule.id, "user1", session_trigger_count=1)
        db.upsert_escalation_state(rule.id, "user1", session_trigger_count=5)
        state = db.get_escalation_state(rule.id, "user1")
        assert state.session_trigger_count == 5

    def test_reset_escalation_state(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        db.upsert_escalation_state(rule.id, "user1", session_trigger_count=3)
        assert db.reset_escalation_state(rule.id, "user1") is True
        assert db.get_escalation_state(rule.id, "user1") is None

    def test_reset_nonexistent(self, db):
        assert db.reset_escalation_state("nope", "nope") is False


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestDeleteAllForUser:
    def test_delete_all_as_guardian(self, db):
        rel = db.create_relationship("user1", "d1")
        db.accept_relationship(rel.id)
        db.create_policy(rel.id, category="test")
        db.log_action(rel.id, "user1", "test_action")
        db.create_self_monitoring_rule("user1", "Rule1")
        db.create_governance_policy("user1", "GP1")

        counts = db.delete_all_for_user("user1")
        assert counts["relationships"] == 1
        assert counts["policies"] == 1
        assert counts["audit_entries"] == 1
        assert counts["self_monitoring_rules"] == 1
        assert counts["governance_policies"] == 1

    def test_delete_all_as_dependent(self, db):
        rel = db.create_relationship("g1", "user1")
        db.create_policy(rel.id, category="dep_policy")
        db.log_action(rel.id, "g1", "action_for_dep")

        counts = db.delete_all_for_user("user1")
        assert counts["relationships"] == 1
        assert counts["policies"] == 1
        assert counts["audit_entries"] == 1

    def test_delete_all_includes_alerts_and_escalation(self, db):
        rule = db.create_self_monitoring_rule("user1", "R1")
        db.create_self_monitoring_alert("user1", rule.id)
        db.create_self_monitoring_alert("user1", rule.id)
        db.upsert_escalation_state(rule.id, "user1", session_trigger_count=2)

        counts = db.delete_all_for_user("user1")
        assert counts["self_monitoring_rules"] == 1
        assert counts["self_monitoring_alerts"] == 2
        assert db.get_escalation_state(rule.id, "user1") is None

    def test_delete_all_empty_user(self, db):
        counts = db.delete_all_for_user("nobody")
        assert all(v == 0 for v in counts.values())
