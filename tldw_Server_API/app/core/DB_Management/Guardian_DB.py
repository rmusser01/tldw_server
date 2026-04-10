"""
Guardian_DB

SQLite wrapper for guardian-child account relationships, supervised policies,
and self-monitoring rules.

Tables:
- guardian_relationships: links between guardian and dependent accounts
- supervised_policies: per-dependent block/notify rules configured by guardians
- supervision_audit_log: audit trail for guardian actions
- governance_policies: named policy groups that bundle multiple rules
- self_monitoring_rules: user-defined awareness/self-block rules
- self_monitoring_alerts: alert log for self-monitoring triggers
- escalation_state: per-rule escalation counters (session + rolling window)

Thread-safe with RLock + WAL mode. Foreign keys enabled.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from loguru import logger
from tldw_Server_API.app.core.DB_Management.db_path_utils import resolve_trusted_database_path
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    begin_immediate_if_needed,
    configure_sqlite_connection,
)

def _validate_regex_pattern(pattern: str) -> None:
    """Validate a regex pattern for ReDoS safety at creation time.

    Raises ValueError if the pattern is unsafe.
    """
    try:
        from tldw_Server_API.app.core.Character_Chat.regex_safety import validate_regex_safety
        is_safe, reason = validate_regex_safety(pattern)
        if not is_safe:
            raise ValueError(f"Unsafe regex pattern: {reason}")
    except ImportError:
        # Fallback: at minimum verify it compiles
        import re as _re
        _re.compile(pattern)


_GUARDIAN_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    json.JSONDecodeError,
    sqlite3.Error,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid4().hex[:16]


def _safe_get(row: Any, key: str, default: Any = None) -> Any:
    """Safely read a column from a sqlite3.Row, returning default if missing."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


# ── Data classes ─────────────────────────────────────────────

@dataclass
class GuardianRelationship:
    id: str
    guardian_user_id: str
    dependent_user_id: str
    relationship_type: str = "parent"  # parent | legal_guardian | institutional
    status: str = "pending"  # pending | active | suspended | dissolved
    consent_given_by_dependent: bool = False
    consent_given_at: str | None = None
    dependent_visible: bool = True  # whether dependent can see monitoring is active
    dissolution_reason: str | None = None
    dissolved_at: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SupervisedPolicy:
    id: str
    relationship_id: str
    governance_policy_id: str | None = None  # optional parent governance policy group
    policy_type: str = "block"  # block | notify
    category: str = ""  # e.g. "explicit_content", "self_harm", "bullying"
    pattern: str = ""  # regex or literal pattern
    pattern_type: str = "literal"  # literal | regex
    action: str = "block"  # block | redact | warn | notify
    phase: str = "both"  # input | output | both
    severity: str = "warning"  # info | warning | critical
    notify_guardian: bool = True
    notify_context: str = "topic_only"  # topic_only | snippet | full_message
    message_to_dependent: str | None = None  # what the child sees on block
    enabled: bool = True
    metadata: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SupervisionAuditEntry:
    id: str
    relationship_id: str
    actor_user_id: str  # who performed the action
    action: str  # e.g. "policy_created", "content_blocked", "alert_sent"
    target_user_id: str | None = None
    policy_id: str | None = None
    detail: str = ""
    metadata: dict[str, Any] | None = None
    created_at: str = ""


@dataclass
class GovernancePolicy:
    """Named policy group that bundles multiple rules under one umbrella."""
    id: str
    owner_user_id: str
    name: str
    description: str = ""
    policy_mode: str = "guardian"  # guardian | self
    scope_chat_types: str = "all"  # all | regular | character | rag | comma-separated
    enabled: bool = True
    schedule_start: str | None = None  # HH:MM (24h) or None
    schedule_end: str | None = None
    schedule_days: str | None = None  # comma-separated: mon,tue,wed,...
    schedule_timezone: str = "UTC"
    transparent: bool = False  # if True, managed user can see rule names
    metadata: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SelfMonitoringRule:
    """User-defined self-awareness or self-block rule."""
    id: str
    user_id: str
    governance_policy_id: str | None = None  # optional parent policy group
    name: str = ""
    category: str = ""
    patterns: list[str] = field(default_factory=list)
    pattern_type: str = "literal"  # literal | regex
    except_patterns: list[str] = field(default_factory=list)  # false-positive exclusions
    rule_type: str = "notify"  # block | notify
    action: str = "notify"  # block | redact | notify
    phase: str = "both"  # input | output | both
    severity: str = "info"  # info | warning | critical
    display_mode: str = "inline_banner"  # inline_banner | sidebar_note | post_session_summary | silent_log
    block_message: str | None = None  # custom message for self-block
    context_note: str | None = None  # user's personal reminder
    notification_frequency: str = "once_per_conversation"
    notification_channels: list[str] = field(default_factory=lambda: ["in_app"])
    webhook_url: str | None = None
    trusted_contact_email: str | None = None
    crisis_resources_enabled: bool = False
    cooldown_minutes: int = 0  # anti-impulsive-disable window
    bypass_protection: str = "none"  # none | cooldown | confirmation | partner_approval
    bypass_partner_user_id: str | None = None
    # Escalation config
    escalation_session_threshold: int = 0  # 0 = disabled; after N triggers in session
    escalation_session_action: str | None = None  # e.g. "block"
    escalation_window_days: int = 0  # rolling window for cross-session escalation
    escalation_window_threshold: int = 0
    escalation_window_action: str | None = None
    min_context_length: int = 0  # minimum chars for a match to fire
    enabled: bool = True
    pending_deactivation_at: str | None = None  # cooldown deactivation timestamp
    deactivation_confirmation_token: str | None = None  # token for confirmation/partner bypass
    deactivation_requested_at: str | None = None  # ISO timestamp of deactivation request
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SelfMonitoringAlert:
    """Alert record for a self-monitoring rule trigger."""
    id: str
    user_id: str
    rule_id: str
    rule_name: str = ""
    category: str = ""
    severity: str = "info"
    matched_pattern: str = ""
    context_snippet: str | None = None
    snippet_mode: str = "full_snippet"
    conversation_id: str | None = None
    session_id: str | None = None
    chat_type: str | None = None
    phase: str = "input"
    action_taken: str = "notified"  # blocked | redacted | notified
    notification_sent: bool = False
    notification_channels_used: list[str] = field(default_factory=list)
    crisis_resources_shown: bool = False
    display_mode: str = "inline_banner"
    escalation_info: dict[str, Any] | None = None
    is_read: bool = False
    metadata: dict[str, Any] | None = None
    created_at: str = ""


@dataclass
class EscalationState:
    """Tracks per-rule escalation counters."""
    rule_id: str
    user_id: str
    session_id: str | None = None
    session_trigger_count: int = 0
    window_trigger_count: int = 0
    current_escalated_action: str | None = None
    escalated_at: str | None = None
    cooldown_until: str | None = None
    updated_at: str = ""


@dataclass
class HouseholdDraft:
    id: str
    owner_user_id: str
    name: str
    mode: str = "family"  # family | institutional
    status: str = "draft"  # draft | invites_pending | partially_active | active | needs_attention
    metadata: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class HouseholdMemberDraft:
    id: str
    household_draft_id: str
    role: str  # guardian | dependent | caregiver
    display_name: str
    user_id: str | None = None
    email: str | None = None
    invite_required: bool = True
    invite_status: str = "pending"
    account_mode: str = "existing_account"
    provisioning_status: str = "not_started"
    metadata: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class RelationshipDraft:
    id: str
    household_draft_id: str
    guardian_member_draft_id: str
    dependent_member_draft_id: str
    relationship_type: str = "parent"
    dependent_visible: bool = True
    status: str = "pending"
    relationship_id: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class GuardrailPlanDraft:
    id: str
    household_draft_id: str
    dependent_member_draft_id: str
    relationship_draft_id: str
    template_id: str
    dependent_user_id: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued | active | failed
    materialized_policy_id: str | None = None
    failure_reason: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ActivationRun:
    id: str
    household_draft_id: str
    relationship_id: str | None = None
    dependent_user_id: str | None = None
    plan_draft_id: str | None = None
    status: str = "queued"  # queued | active | failed
    detail: str = ""
    metadata: dict[str, Any] | None = None
    created_at: str = ""


# ── Database class ───────────────────────────────────────────

class GuardianDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(resolve_trusted_database_path(db_path, label="guardian database"))
        self._lock = threading.RLock()
        try:
            self._ensure_schema()
            self._migrate_schema()
        except sqlite3.OperationalError as exc:
            if "unable to open database file" in str(exc).lower():
                raise ValueError("GuardianDB parent directory must already exist") from exc
            raise

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            configure_sqlite_connection(conn)
        except _GUARDIAN_NONCRITICAL_EXCEPTIONS:
            pass
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS guardian_relationships (
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

                    CREATE INDEX IF NOT EXISTS idx_gr_guardian
                        ON guardian_relationships(guardian_user_id);
                    CREATE INDEX IF NOT EXISTS idx_gr_dependent
                        ON guardian_relationships(dependent_user_id);
                    CREATE INDEX IF NOT EXISTS idx_gr_status
                        ON guardian_relationships(status);

                    CREATE TABLE IF NOT EXISTS supervised_policies (
                        id TEXT PRIMARY KEY,
                        relationship_id TEXT NOT NULL,
                        governance_policy_id TEXT,
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
                        FOREIGN KEY (relationship_id) REFERENCES guardian_relationships(id),
                        FOREIGN KEY (governance_policy_id) REFERENCES governance_policies(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_sp_relationship
                        ON supervised_policies(relationship_id);
                    CREATE INDEX IF NOT EXISTS idx_sp_category
                        ON supervised_policies(category);
                    CREATE INDEX IF NOT EXISTS idx_sp_enabled
                        ON supervised_policies(enabled);

                    CREATE TABLE IF NOT EXISTS supervision_audit_log (
                        id TEXT PRIMARY KEY,
                        relationship_id TEXT NOT NULL,
                        actor_user_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        target_user_id TEXT,
                        policy_id TEXT,
                        detail TEXT NOT NULL DEFAULT '',
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (relationship_id) REFERENCES guardian_relationships(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_sal_relationship
                        ON supervision_audit_log(relationship_id);
                    CREATE INDEX IF NOT EXISTS idx_sal_created
                        ON supervision_audit_log(created_at);

                    CREATE TABLE IF NOT EXISTS governance_policies (
                        id TEXT PRIMARY KEY,
                        owner_user_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        policy_mode TEXT NOT NULL DEFAULT 'guardian',
                        scope_chat_types TEXT NOT NULL DEFAULT 'all',
                        enabled INTEGER NOT NULL DEFAULT 1,
                        schedule_start TEXT,
                        schedule_end TEXT,
                        schedule_days TEXT,
                        schedule_timezone TEXT NOT NULL DEFAULT 'UTC',
                        transparent INTEGER NOT NULL DEFAULT 0,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_gp_owner
                        ON governance_policies(owner_user_id);
                    CREATE INDEX IF NOT EXISTS idx_gp_mode
                        ON governance_policies(policy_mode);

                    CREATE TABLE IF NOT EXISTS self_monitoring_rules (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        governance_policy_id TEXT,
                        name TEXT NOT NULL DEFAULT '',
                        category TEXT NOT NULL DEFAULT '',
                        patterns TEXT NOT NULL DEFAULT '[]',
                        pattern_type TEXT NOT NULL DEFAULT 'literal',
                        except_patterns TEXT NOT NULL DEFAULT '[]',
                        rule_type TEXT NOT NULL DEFAULT 'notify',
                        action TEXT NOT NULL DEFAULT 'notify',
                        phase TEXT NOT NULL DEFAULT 'both',
                        severity TEXT NOT NULL DEFAULT 'info',
                        display_mode TEXT NOT NULL DEFAULT 'inline_banner',
                        block_message TEXT,
                        context_note TEXT,
                        notification_frequency TEXT NOT NULL DEFAULT 'once_per_conversation',
                        notification_channels TEXT NOT NULL DEFAULT '["in_app"]',
                        webhook_url TEXT,
                        trusted_contact_email TEXT,
                        crisis_resources_enabled INTEGER NOT NULL DEFAULT 0,
                        cooldown_minutes INTEGER NOT NULL DEFAULT 0,
                        bypass_protection TEXT NOT NULL DEFAULT 'none',
                        bypass_partner_user_id TEXT,
                        escalation_session_threshold INTEGER NOT NULL DEFAULT 0,
                        escalation_session_action TEXT,
                        escalation_window_days INTEGER NOT NULL DEFAULT 0,
                        escalation_window_threshold INTEGER NOT NULL DEFAULT 0,
                        escalation_window_action TEXT,
                        min_context_length INTEGER NOT NULL DEFAULT 0,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        pending_deactivation_at TEXT,
                        deactivation_confirmation_token TEXT,
                        deactivation_requested_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (governance_policy_id) REFERENCES governance_policies(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_smr_user
                        ON self_monitoring_rules(user_id);
                    CREATE INDEX IF NOT EXISTS idx_smr_enabled
                        ON self_monitoring_rules(enabled);

                    CREATE TABLE IF NOT EXISTS self_monitoring_alerts (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        rule_id TEXT NOT NULL,
                        rule_name TEXT NOT NULL DEFAULT '',
                        category TEXT NOT NULL DEFAULT '',
                        severity TEXT NOT NULL DEFAULT 'info',
                        matched_pattern TEXT NOT NULL DEFAULT '',
                        context_snippet TEXT,
                        snippet_mode TEXT NOT NULL DEFAULT 'full_snippet',
                        conversation_id TEXT,
                        session_id TEXT,
                        chat_type TEXT,
                        phase TEXT NOT NULL DEFAULT 'input',
                        action_taken TEXT NOT NULL DEFAULT 'notified',
                        notification_sent INTEGER NOT NULL DEFAULT 0,
                        notification_channels_used TEXT NOT NULL DEFAULT '[]',
                        crisis_resources_shown INTEGER NOT NULL DEFAULT 0,
                        display_mode TEXT NOT NULL DEFAULT 'inline_banner',
                        escalation_info TEXT,
                        is_read INTEGER NOT NULL DEFAULT 0,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_sma_user
                        ON self_monitoring_alerts(user_id);
                    CREATE INDEX IF NOT EXISTS idx_sma_rule
                        ON self_monitoring_alerts(rule_id);
                    CREATE INDEX IF NOT EXISTS idx_sma_created
                        ON self_monitoring_alerts(created_at);
                    CREATE INDEX IF NOT EXISTS idx_sma_read
                        ON self_monitoring_alerts(is_read);

                    CREATE TABLE IF NOT EXISTS escalation_state (
                        rule_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        session_trigger_count INTEGER NOT NULL DEFAULT 0,
                        window_trigger_count INTEGER NOT NULL DEFAULT 0,
                        current_escalated_action TEXT,
                        escalated_at TEXT,
                        cooldown_until TEXT,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (rule_id, user_id)
                    );

                    CREATE TABLE IF NOT EXISTS guardian_household_drafts (
                        id TEXT PRIMARY KEY,
                        owner_user_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        mode TEXT NOT NULL DEFAULT 'family',
                        status TEXT NOT NULL DEFAULT 'draft',
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_ghd_owner
                        ON guardian_household_drafts(owner_user_id);
                    CREATE INDEX IF NOT EXISTS idx_ghd_status
                        ON guardian_household_drafts(status);

                    CREATE TABLE IF NOT EXISTS guardian_household_member_drafts (
                        id TEXT PRIMARY KEY,
                        household_draft_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        user_id TEXT,
                        email TEXT,
                        invite_required INTEGER NOT NULL DEFAULT 1,
                        invite_status TEXT NOT NULL DEFAULT 'pending',
                        account_mode TEXT NOT NULL DEFAULT 'existing_account',
                        provisioning_status TEXT NOT NULL DEFAULT 'not_started',
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (household_draft_id) REFERENCES guardian_household_drafts(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_ghmd_household
                        ON guardian_household_member_drafts(household_draft_id);
                    CREATE INDEX IF NOT EXISTS idx_ghmd_role
                        ON guardian_household_member_drafts(role);
                    CREATE INDEX IF NOT EXISTS idx_ghmd_user
                        ON guardian_household_member_drafts(user_id);
                    CREATE TABLE IF NOT EXISTS guardian_household_member_invites (
                        id TEXT PRIMARY KEY,
                        household_draft_id TEXT NOT NULL,
                        member_draft_id TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'ready',
                        delivery_channel TEXT NOT NULL,
                        delivery_target TEXT,
                        invite_token TEXT NOT NULL,
                        resend_count INTEGER NOT NULL DEFAULT 0,
                        last_sent_at TEXT,
                        accepted_at TEXT,
                        expires_at TEXT,
                        revoked_at TEXT,
                        failure_reason TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (household_draft_id) REFERENCES guardian_household_drafts(id) ON DELETE CASCADE,
                        FOREIGN KEY (member_draft_id) REFERENCES guardian_household_member_drafts(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_ghmi_household
                        ON guardian_household_member_invites(household_draft_id);
                    CREATE INDEX IF NOT EXISTS idx_ghmi_member
                        ON guardian_household_member_invites(member_draft_id);
                    CREATE INDEX IF NOT EXISTS idx_ghmi_status
                        ON guardian_household_member_invites(status);

                    CREATE TABLE IF NOT EXISTS guardian_relationship_drafts (
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

                    CREATE INDEX IF NOT EXISTS idx_grd_household
                        ON guardian_relationship_drafts(household_draft_id);
                    CREATE INDEX IF NOT EXISTS idx_grd_status
                        ON guardian_relationship_drafts(status);

                    CREATE TABLE IF NOT EXISTS guardian_guardrail_plan_drafts (
                        id TEXT PRIMARY KEY,
                        household_draft_id TEXT NOT NULL,
                        dependent_member_draft_id TEXT NOT NULL,
                        dependent_user_id TEXT,
                        relationship_draft_id TEXT NOT NULL,
                        template_id TEXT NOT NULL,
                        overrides TEXT NOT NULL DEFAULT '{}',
                        status TEXT NOT NULL DEFAULT 'queued',
                        materialized_policy_id TEXT,
                        failure_reason TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (dependent_member_draft_id) REFERENCES guardian_household_member_drafts(id) ON DELETE CASCADE,
                        FOREIGN KEY (household_draft_id) REFERENCES guardian_household_drafts(id) ON DELETE CASCADE,
                        FOREIGN KEY (relationship_draft_id) REFERENCES guardian_relationship_drafts(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_ggpd_household
                        ON guardian_guardrail_plan_drafts(household_draft_id);
                    CREATE INDEX IF NOT EXISTS idx_ggpd_status
                        ON guardian_guardrail_plan_drafts(status);
                    CREATE INDEX IF NOT EXISTS idx_ggpd_relationship
                        ON guardian_guardrail_plan_drafts(relationship_draft_id);

                    CREATE TABLE IF NOT EXISTS guardian_activation_runs (
                        id TEXT PRIMARY KEY,
                        household_draft_id TEXT NOT NULL,
                        relationship_id TEXT,
                        dependent_user_id TEXT,
                        plan_draft_id TEXT,
                        status TEXT NOT NULL,
                        detail TEXT NOT NULL DEFAULT '',
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (household_draft_id) REFERENCES guardian_household_drafts(id) ON DELETE CASCADE,
                        FOREIGN KEY (relationship_id) REFERENCES guardian_relationships(id),
                        FOREIGN KEY (plan_draft_id) REFERENCES guardian_guardrail_plan_drafts(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_gar_household
                        ON guardian_activation_runs(household_draft_id);
                    CREATE INDEX IF NOT EXISTS idx_gar_relationship
                        ON guardian_activation_runs(relationship_id);
                    CREATE INDEX IF NOT EXISTS idx_gar_plan
                        ON guardian_activation_runs(plan_draft_id);
                """)
            finally:
                conn.close()

    def _migrate_schema(self) -> None:
        """Add columns introduced after the initial schema, if missing."""
        migrations = [
            ("supervised_policies", "governance_policy_id", "TEXT"),
            ("self_monitoring_rules", "governance_policy_id", "TEXT"),
            ("self_monitoring_rules", "deactivation_confirmation_token", "TEXT"),
            ("self_monitoring_rules", "deactivation_requested_at", "TEXT"),
            (
                "guardian_household_member_drafts",
                "account_mode",
                "TEXT NOT NULL DEFAULT 'existing_account'",
            ),
            (
                "guardian_household_member_drafts",
                "provisioning_status",
                "TEXT NOT NULL DEFAULT 'not_started'",
            ),
        ]
        with self._lock:
            conn = self._connect()
            try:
                for table, column, col_type in migrations:
                    try:
                        existing = {
                            row["name"]
                            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                        }
                        if column not in existing:
                            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                            logger.debug(f"Migrated: added {table}.{column} ({col_type})")
                    except _GUARDIAN_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(
                            f"Guardian DB migration error for {table}.{column} (non-critical): {e}"
                        )
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS guardian_household_member_invites (
                        id TEXT PRIMARY KEY,
                        household_draft_id TEXT NOT NULL,
                        member_draft_id TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'ready',
                        delivery_channel TEXT NOT NULL,
                        delivery_target TEXT,
                        invite_token TEXT NOT NULL,
                        resend_count INTEGER NOT NULL DEFAULT 0,
                        last_sent_at TEXT,
                        accepted_at TEXT,
                        expires_at TEXT,
                        revoked_at TEXT,
                        failure_reason TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (household_draft_id) REFERENCES guardian_household_drafts(id) ON DELETE CASCADE,
                        FOREIGN KEY (member_draft_id) REFERENCES guardian_household_member_drafts(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_ghmi_household
                        ON guardian_household_member_invites(household_draft_id);
                    CREATE INDEX IF NOT EXISTS idx_ghmi_member
                        ON guardian_household_member_invites(member_draft_id);
                    CREATE INDEX IF NOT EXISTS idx_ghmi_status
                        ON guardian_household_member_invites(status);
                """)
                self._migrate_guardrail_plan_drafts_table(conn)
                self._ensure_optional_indexes(conn)
            finally:
                conn.close()

    @staticmethod
    def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(_safe_get(row, "name") == column for row in rows)

    def _ensure_optional_indexes(self, conn: sqlite3.Connection) -> None:
        optional_indexes = [
            ("idx_sp_governance", "supervised_policies", "governance_policy_id"),
            ("idx_smr_policy", "self_monitoring_rules", "governance_policy_id"),
            ("idx_ghmd_account_mode", "guardian_household_member_drafts", "account_mode"),
            ("idx_ggpd_member", "guardian_guardrail_plan_drafts", "dependent_member_draft_id"),
        ]
        for idx_name, table, column in optional_indexes:
            if self._table_has_column(conn, table, column):
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})"
                )

    def _migrate_guardrail_plan_drafts_table(self, conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]: row
            for row in conn.execute(
                "PRAGMA table_info(guardian_guardrail_plan_drafts)"
            ).fetchall()
        }
        if not columns:
            return

        dependent_user_column = columns.get("dependent_user_id")
        dependent_user_not_null = bool(_safe_get(dependent_user_column, "notnull", 0))
        needs_rebuild = (
            "dependent_member_draft_id" not in columns
            or dependent_user_not_null
        )
        if not needs_rebuild:
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_ggpd_member
                   ON guardian_guardrail_plan_drafts(dependent_member_draft_id)"""
            )
            return

        conn.execute(
            "ALTER TABLE guardian_guardrail_plan_drafts RENAME TO guardian_guardrail_plan_drafts_legacy"
        )
        conn.executescript("""
            CREATE TABLE guardian_guardrail_plan_drafts (
                id TEXT PRIMARY KEY,
                household_draft_id TEXT NOT NULL,
                dependent_member_draft_id TEXT NOT NULL,
                dependent_user_id TEXT,
                relationship_draft_id TEXT NOT NULL,
                template_id TEXT NOT NULL,
                overrides TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'queued',
                materialized_policy_id TEXT,
                failure_reason TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (dependent_member_draft_id) REFERENCES guardian_household_member_drafts(id) ON DELETE CASCADE,
                FOREIGN KEY (household_draft_id) REFERENCES guardian_household_drafts(id) ON DELETE CASCADE,
                FOREIGN KEY (relationship_draft_id) REFERENCES guardian_relationship_drafts(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_ggpd_household
                ON guardian_guardrail_plan_drafts(household_draft_id);
            CREATE INDEX IF NOT EXISTS idx_ggpd_member
                ON guardian_guardrail_plan_drafts(dependent_member_draft_id);
            CREATE INDEX IF NOT EXISTS idx_ggpd_status
                ON guardian_guardrail_plan_drafts(status);
            CREATE INDEX IF NOT EXISTS idx_ggpd_relationship
                ON guardian_guardrail_plan_drafts(relationship_draft_id);
        """)

        legacy_has_member_column = "dependent_member_draft_id" in columns
        if legacy_has_member_column:
            conn.execute(
                """INSERT INTO guardian_guardrail_plan_drafts
                (id, household_draft_id, dependent_member_draft_id, dependent_user_id,
                 relationship_draft_id, template_id, overrides, status, materialized_policy_id,
                 failure_reason, metadata, created_at, updated_at)
                SELECT p.id,
                       p.household_draft_id,
                       COALESCE(p.dependent_member_draft_id, r.dependent_member_draft_id),
                       p.dependent_user_id,
                       p.relationship_draft_id,
                       p.template_id,
                       p.overrides,
                       p.status,
                       p.materialized_policy_id,
                       p.failure_reason,
                       p.metadata,
                       p.created_at,
                       p.updated_at
                  FROM guardian_guardrail_plan_drafts_legacy p
                  LEFT JOIN guardian_relationship_drafts r ON p.relationship_draft_id = r.id"""
            )
        else:
            conn.execute(
                """INSERT INTO guardian_guardrail_plan_drafts
                (id, household_draft_id, dependent_member_draft_id, dependent_user_id,
                 relationship_draft_id, template_id, overrides, status, materialized_policy_id,
                 failure_reason, metadata, created_at, updated_at)
                SELECT p.id,
                       p.household_draft_id,
                       r.dependent_member_draft_id,
                       p.dependent_user_id,
                       p.relationship_draft_id,
                       p.template_id,
                       p.overrides,
                       p.status,
                       p.materialized_policy_id,
                       p.failure_reason,
                       p.metadata,
                       p.created_at,
                       p.updated_at
                  FROM guardian_guardrail_plan_drafts_legacy p
                  LEFT JOIN guardian_relationship_drafts r ON p.relationship_draft_id = r.id"""
            )

        conn.execute("DROP TABLE guardian_guardrail_plan_drafts_legacy")

    # ── Guardian Relationships ─────────────────────────────────

    def create_relationship(
        self,
        guardian_user_id: str,
        dependent_user_id: str,
        relationship_type: str = "parent",
        dependent_visible: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> GuardianRelationship:
        if guardian_user_id == dependent_user_id:
            raise ValueError("Guardian and dependent cannot be the same user")
        now = _utcnow_iso()
        rel = GuardianRelationship(
            id=_new_id(),
            guardian_user_id=str(guardian_user_id),
            dependent_user_id=str(dependent_user_id),
            relationship_type=relationship_type,
            status="pending",
            dependent_visible=dependent_visible,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO guardian_relationships
                    (id, guardian_user_id, dependent_user_id, relationship_type,
                     status, consent_given_by_dependent, consent_given_at,
                     dependent_visible, dissolution_reason, dissolved_at,
                     metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        rel.id, rel.guardian_user_id, rel.dependent_user_id,
                        rel.relationship_type, rel.status,
                        int(rel.consent_given_by_dependent), rel.consent_given_at,
                        int(rel.dependent_visible), rel.dissolution_reason,
                        rel.dissolved_at,
                        json.dumps(rel.metadata) if rel.metadata else None,
                        rel.created_at, rel.updated_at,
                    ),
                )
                return rel
            except sqlite3.IntegrityError as e:
                raise ValueError(
                    f"Relationship already exists between {guardian_user_id} and {dependent_user_id}"
                ) from e
            finally:
                conn.close()

    def get_relationship(self, relationship_id: str) -> GuardianRelationship | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM guardian_relationships WHERE id = ?",
                    (relationship_id,),
                ).fetchone()
                if not row:
                    return None
                return self._row_to_relationship(row)
            finally:
                conn.close()

    def get_relationships_for_guardian(
        self,
        guardian_user_id: str,
        status: str | None = None,
    ) -> list[GuardianRelationship]:
        with self._lock:
            conn = self._connect()
            try:
                if status:
                    rows = conn.execute(
                        "SELECT * FROM guardian_relationships WHERE guardian_user_id = ? AND status = ? ORDER BY created_at DESC",
                        (str(guardian_user_id), status),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM guardian_relationships WHERE guardian_user_id = ? ORDER BY created_at DESC",
                        (str(guardian_user_id),),
                    ).fetchall()
                return [self._row_to_relationship(r) for r in rows]
            finally:
                conn.close()

    def get_relationships_for_dependent(
        self,
        dependent_user_id: str,
        status: str | None = None,
    ) -> list[GuardianRelationship]:
        with self._lock:
            conn = self._connect()
            try:
                if status:
                    rows = conn.execute(
                        "SELECT * FROM guardian_relationships WHERE dependent_user_id = ? AND status = ? ORDER BY created_at DESC",
                        (str(dependent_user_id), status),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM guardian_relationships WHERE dependent_user_id = ? ORDER BY created_at DESC",
                        (str(dependent_user_id),),
                    ).fetchall()
                return [self._row_to_relationship(r) for r in rows]
            finally:
                conn.close()

    def accept_relationship(self, relationship_id: str) -> bool:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    """UPDATE guardian_relationships
                       SET status = 'active',
                           consent_given_by_dependent = 1,
                           consent_given_at = ?,
                           updated_at = ?
                       WHERE id = ? AND status = 'pending'""",
                    (now, now, relationship_id),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def dissolve_relationship(
        self,
        relationship_id: str,
        reason: str = "manual",
    ) -> bool:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    """UPDATE guardian_relationships
                       SET status = 'dissolved',
                           dissolution_reason = ?,
                           dissolved_at = ?,
                           updated_at = ?
                       WHERE id = ? AND status IN ('pending', 'active', 'suspended')""",
                    (reason, now, now, relationship_id),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def suspend_relationship(self, relationship_id: str) -> bool:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    """UPDATE guardian_relationships
                       SET status = 'suspended', updated_at = ?
                       WHERE id = ? AND status = 'active'""",
                    (now, relationship_id),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def reactivate_relationship(self, relationship_id: str) -> bool:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    """UPDATE guardian_relationships
                       SET status = 'active', updated_at = ?
                       WHERE id = ? AND status = 'suspended'""",
                    (now, relationship_id),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def is_guardian_of(self, guardian_user_id: str, dependent_user_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT 1 FROM guardian_relationships
                       WHERE guardian_user_id = ? AND dependent_user_id = ?
                       AND status = 'active'""",
                    (str(guardian_user_id), str(dependent_user_id)),
                ).fetchone()
                return row is not None
            finally:
                conn.close()

    @staticmethod
    def _row_to_relationship(row: sqlite3.Row) -> GuardianRelationship:
        meta = None
        raw_meta = row["metadata"]
        if raw_meta:
            try:
                meta = json.loads(raw_meta)
            except _GUARDIAN_NONCRITICAL_EXCEPTIONS:
                meta = None
        return GuardianRelationship(
            id=row["id"],
            guardian_user_id=row["guardian_user_id"],
            dependent_user_id=row["dependent_user_id"],
            relationship_type=row["relationship_type"],
            status=row["status"],
            consent_given_by_dependent=bool(row["consent_given_by_dependent"]),
            consent_given_at=row["consent_given_at"],
            dependent_visible=bool(row["dependent_visible"]),
            dissolution_reason=row["dissolution_reason"],
            dissolved_at=row["dissolved_at"],
            metadata=meta,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── Supervised Policies ────────────────────────────────────

    def create_policy(
        self,
        relationship_id: str,
        policy_type: str = "block",
        category: str = "",
        pattern: str = "",
        pattern_type: str = "literal",
        action: str = "block",
        phase: str = "both",
        severity: str = "warning",
        notify_guardian: bool = True,
        notify_context: str = "topic_only",
        message_to_dependent: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        governance_policy_id: str | None = None,
    ) -> SupervisedPolicy:
        if action not in ("block", "redact", "warn", "notify"):
            raise ValueError(f"Invalid action: {action}")
        if phase not in ("input", "output", "both"):
            raise ValueError(f"Invalid phase: {phase}")
        if severity not in ("info", "warning", "critical"):
            raise ValueError(f"Invalid severity: {severity}")
        if notify_context not in ("topic_only", "snippet", "full_message"):
            raise ValueError(f"Invalid notify_context: {notify_context}")
        if pattern and pattern_type == "regex":
            _validate_regex_pattern(pattern)
        now = _utcnow_iso()
        pol = SupervisedPolicy(
            id=_new_id(),
            relationship_id=relationship_id,
            governance_policy_id=governance_policy_id,
            policy_type=policy_type,
            category=category,
            pattern=pattern,
            pattern_type=pattern_type,
            action=action,
            phase=phase,
            severity=severity,
            notify_guardian=notify_guardian,
            notify_context=notify_context,
            message_to_dependent=message_to_dependent,
            enabled=enabled,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO supervised_policies
                    (id, relationship_id, governance_policy_id, policy_type,
                     category, pattern, pattern_type, action, phase, severity,
                     notify_guardian, notify_context, message_to_dependent,
                     enabled, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pol.id, pol.relationship_id, pol.governance_policy_id,
                        pol.policy_type, pol.category, pol.pattern,
                        pol.pattern_type, pol.action, pol.phase, pol.severity,
                        int(pol.notify_guardian), pol.notify_context,
                        pol.message_to_dependent, int(pol.enabled),
                        json.dumps(pol.metadata) if pol.metadata else None,
                        pol.created_at, pol.updated_at,
                    ),
                )
                return pol
            finally:
                conn.close()

    def get_policy(self, policy_id: str) -> SupervisedPolicy | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM supervised_policies WHERE id = ?",
                    (policy_id,),
                ).fetchone()
                if not row:
                    return None
                return self._row_to_policy(row)
            finally:
                conn.close()

    def list_policies_for_relationship(
        self,
        relationship_id: str,
        enabled_only: bool = False,
    ) -> list[SupervisedPolicy]:
        with self._lock:
            conn = self._connect()
            try:
                if enabled_only:
                    rows = conn.execute(
                        """SELECT * FROM supervised_policies
                           WHERE relationship_id = ? AND enabled = 1
                           ORDER BY created_at""",
                        (relationship_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM supervised_policies
                           WHERE relationship_id = ?
                           ORDER BY created_at""",
                        (relationship_id,),
                    ).fetchall()
                return [self._row_to_policy(r) for r in rows]
            finally:
                conn.close()

    def list_active_policies_for_dependent(
        self,
        dependent_user_id: str,
    ) -> list[SupervisedPolicy]:
        """Get all enabled policies from active guardian relationships for a dependent."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT sp.* FROM supervised_policies sp
                       JOIN guardian_relationships gr ON sp.relationship_id = gr.id
                       WHERE gr.dependent_user_id = ?
                       AND gr.status = 'active'
                       AND sp.enabled = 1
                       ORDER BY sp.created_at""",
                    (str(dependent_user_id),),
                ).fetchall()
                return [self._row_to_policy(r) for r in rows]
            finally:
                conn.close()

    def update_policy(
        self,
        policy_id: str,
        **kwargs: Any,
    ) -> bool:
        allowed_fields = {
            "policy_type", "category", "pattern", "pattern_type",
            "action", "phase", "severity", "notify_guardian",
            "notify_context", "message_to_dependent", "enabled", "metadata",
            "governance_policy_id",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return False
        # Validate constraints
        if "action" in updates and updates["action"] not in ("block", "redact", "warn", "notify"):
            raise ValueError(f"Invalid action: {updates['action']}")
        if "phase" in updates and updates["phase"] not in ("input", "output", "both"):
            raise ValueError(f"Invalid phase: {updates['phase']}")
        if "severity" in updates and updates["severity"] not in ("info", "warning", "critical"):
            raise ValueError(f"Invalid severity: {updates['severity']}")
        now = _utcnow_iso()
        set_clauses = []
        params: list[Any] = []
        for k, v in updates.items():
            if k == "metadata":
                set_clauses.append(f"{k} = ?")
                params.append(json.dumps(v) if v else None)
            elif k in ("notify_guardian", "enabled"):
                set_clauses.append(f"{k} = ?")
                params.append(int(bool(v)))
            else:
                set_clauses.append(f"{k} = ?")
                params.append(v)
        set_clauses.append("updated_at = ?")
        params.append(now)
        params.append(policy_id)
        set_clause_sql = ", ".join(set_clauses)
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    f"UPDATE supervised_policies SET {set_clause_sql} WHERE id = ?",  # nosec B608
                    params,
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def delete_policy(self, policy_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    "DELETE FROM supervised_policies WHERE id = ?",
                    (policy_id,),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    @staticmethod
    def _row_to_policy(row: sqlite3.Row) -> SupervisedPolicy:
        meta = None
        raw_meta = row["metadata"]
        if raw_meta:
            try:
                meta = json.loads(raw_meta)
            except _GUARDIAN_NONCRITICAL_EXCEPTIONS:
                meta = None
        try:
            governance_policy_id = row["governance_policy_id"]
        except (IndexError, KeyError):
            governance_policy_id = None
        return SupervisedPolicy(
            id=row["id"],
            relationship_id=row["relationship_id"],
            governance_policy_id=governance_policy_id,
            policy_type=row["policy_type"],
            category=row["category"],
            pattern=row["pattern"],
            pattern_type=row["pattern_type"],
            action=row["action"],
            phase=row["phase"],
            severity=row["severity"],
            notify_guardian=bool(row["notify_guardian"]),
            notify_context=row["notify_context"],
            message_to_dependent=row["message_to_dependent"],
            enabled=bool(row["enabled"]),
            metadata=meta,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── Audit Log ──────────────────────────────────────────────

    def log_action(
        self,
        relationship_id: str,
        actor_user_id: str,
        action: str,
        target_user_id: str | None = None,
        policy_id: str | None = None,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        entry_id = _new_id()
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO supervision_audit_log
                    (id, relationship_id, actor_user_id, action,
                     target_user_id, policy_id, detail, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry_id, relationship_id, str(actor_user_id),
                        action, target_user_id, policy_id, detail,
                        json.dumps(metadata) if metadata else None,
                        now,
                    ),
                )
                return entry_id
            finally:
                conn.close()

    def get_audit_log(
        self,
        relationship_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SupervisionAuditEntry]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM supervision_audit_log
                       WHERE relationship_id = ?
                       ORDER BY created_at DESC
                       LIMIT ? OFFSET ?""",
                    (relationship_id, limit, offset),
                ).fetchall()
                return [self._row_to_audit_entry(r) for r in rows]
            finally:
                conn.close()

    def count_audit_entries(self, relationship_id: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM supervision_audit_log WHERE relationship_id = ?",
                    (relationship_id,),
                ).fetchone()
                return row["cnt"] if row else 0
            finally:
                conn.close()

    @staticmethod
    def _row_to_audit_entry(row: sqlite3.Row) -> SupervisionAuditEntry:
        meta = None
        raw_meta = row["metadata"]
        if raw_meta:
            try:
                meta = json.loads(raw_meta)
            except _GUARDIAN_NONCRITICAL_EXCEPTIONS:
                meta = None
        return SupervisionAuditEntry(
            id=row["id"],
            relationship_id=row["relationship_id"],
            actor_user_id=row["actor_user_id"],
            action=row["action"],
            target_user_id=row["target_user_id"],
            policy_id=row["policy_id"],
            detail=row["detail"],
            metadata=meta,
            created_at=row["created_at"],
        )

    # ── Governance Policies ───────────────────────────────────

    def create_governance_policy(
        self,
        owner_user_id: str,
        name: str,
        description: str = "",
        policy_mode: str = "guardian",
        scope_chat_types: str = "all",
        enabled: bool = True,
        schedule_start: str | None = None,
        schedule_end: str | None = None,
        schedule_days: str | None = None,
        schedule_timezone: str = "UTC",
        transparent: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> GovernancePolicy:
        if policy_mode not in ("guardian", "self"):
            raise ValueError(f"Invalid policy_mode: {policy_mode}")
        now = _utcnow_iso()
        gp = GovernancePolicy(
            id=_new_id(),
            owner_user_id=str(owner_user_id),
            name=name,
            description=description,
            policy_mode=policy_mode,
            scope_chat_types=scope_chat_types,
            enabled=enabled,
            schedule_start=schedule_start,
            schedule_end=schedule_end,
            schedule_days=schedule_days,
            schedule_timezone=schedule_timezone,
            transparent=transparent,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO governance_policies
                    (id, owner_user_id, name, description, policy_mode,
                     scope_chat_types, enabled, schedule_start, schedule_end,
                     schedule_days, schedule_timezone, transparent,
                     metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        gp.id, gp.owner_user_id, gp.name, gp.description,
                        gp.policy_mode, gp.scope_chat_types, int(gp.enabled),
                        gp.schedule_start, gp.schedule_end, gp.schedule_days,
                        gp.schedule_timezone, int(gp.transparent),
                        json.dumps(gp.metadata) if gp.metadata else None,
                        gp.created_at, gp.updated_at,
                    ),
                )
                return gp
            finally:
                conn.close()

    def get_governance_policy(self, policy_id: str) -> GovernancePolicy | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM governance_policies WHERE id = ?",
                    (policy_id,),
                ).fetchone()
                if not row:
                    return None
                return self._row_to_governance_policy(row)
            finally:
                conn.close()

    def list_governance_policies(
        self,
        owner_user_id: str,
        policy_mode: str | None = None,
    ) -> list[GovernancePolicy]:
        with self._lock:
            conn = self._connect()
            try:
                if policy_mode:
                    rows = conn.execute(
                        """SELECT * FROM governance_policies
                           WHERE owner_user_id = ? AND policy_mode = ?
                           ORDER BY created_at DESC""",
                        (str(owner_user_id), policy_mode),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM governance_policies
                           WHERE owner_user_id = ?
                           ORDER BY created_at DESC""",
                        (str(owner_user_id),),
                    ).fetchall()
                return [self._row_to_governance_policy(r) for r in rows]
            finally:
                conn.close()

    def delete_governance_policy(self, policy_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    "DELETE FROM governance_policies WHERE id = ?",
                    (policy_id,),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    @staticmethod
    def _row_to_governance_policy(row: sqlite3.Row) -> GovernancePolicy:
        meta = None
        raw_meta = row["metadata"]
        if raw_meta:
            try:
                meta = json.loads(raw_meta)
            except _GUARDIAN_NONCRITICAL_EXCEPTIONS:
                meta = None
        return GovernancePolicy(
            id=row["id"],
            owner_user_id=row["owner_user_id"],
            name=row["name"],
            description=row["description"],
            policy_mode=row["policy_mode"],
            scope_chat_types=row["scope_chat_types"],
            enabled=bool(row["enabled"]),
            schedule_start=row["schedule_start"],
            schedule_end=row["schedule_end"],
            schedule_days=row["schedule_days"],
            schedule_timezone=row["schedule_timezone"],
            transparent=bool(row["transparent"]),
            metadata=meta,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── Self-Monitoring Rules ──────────────────────────────────

    def create_self_monitoring_rule(
        self,
        user_id: str,
        name: str,
        category: str = "",
        patterns: list[str] | None = None,
        pattern_type: str = "literal",
        except_patterns: list[str] | None = None,
        rule_type: str = "notify",
        action: str = "notify",
        phase: str = "both",
        severity: str = "info",
        display_mode: str = "inline_banner",
        block_message: str | None = None,
        context_note: str | None = None,
        notification_frequency: str = "once_per_conversation",
        notification_channels: list[str] | None = None,
        webhook_url: str | None = None,
        trusted_contact_email: str | None = None,
        crisis_resources_enabled: bool = False,
        cooldown_minutes: int = 0,
        bypass_protection: str = "none",
        bypass_partner_user_id: str | None = None,
        escalation_session_threshold: int = 0,
        escalation_session_action: str | None = None,
        escalation_window_days: int = 0,
        escalation_window_threshold: int = 0,
        escalation_window_action: str | None = None,
        min_context_length: int = 0,
        governance_policy_id: str | None = None,
        enabled: bool = True,
    ) -> SelfMonitoringRule:
        if action not in ("block", "redact", "notify"):
            raise ValueError(f"Invalid action: {action}")
        if phase not in ("input", "output", "both"):
            raise ValueError(f"Invalid phase: {phase}")
        if severity not in ("info", "warning", "critical"):
            raise ValueError(f"Invalid severity: {severity}")
        if pattern_type == "regex":
            for pat_str in (patterns or []):
                _validate_regex_pattern(pat_str)
            for exc_str in (except_patterns or []):
                _validate_regex_pattern(exc_str)
        now = _utcnow_iso()
        rule = SelfMonitoringRule(
            id=_new_id(),
            user_id=str(user_id),
            governance_policy_id=governance_policy_id,
            name=name,
            category=category,
            patterns=patterns or [],
            pattern_type=pattern_type,
            except_patterns=except_patterns or [],
            rule_type=rule_type,
            action=action,
            phase=phase,
            severity=severity,
            display_mode=display_mode,
            block_message=block_message,
            context_note=context_note,
            notification_frequency=notification_frequency,
            notification_channels=notification_channels or ["in_app"],
            webhook_url=webhook_url,
            trusted_contact_email=trusted_contact_email,
            crisis_resources_enabled=crisis_resources_enabled,
            cooldown_minutes=cooldown_minutes,
            bypass_protection=bypass_protection,
            bypass_partner_user_id=bypass_partner_user_id,
            escalation_session_threshold=escalation_session_threshold,
            escalation_session_action=escalation_session_action,
            escalation_window_days=escalation_window_days,
            escalation_window_threshold=escalation_window_threshold,
            escalation_window_action=escalation_window_action,
            min_context_length=min_context_length,
            enabled=enabled,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO self_monitoring_rules
                    (id, user_id, governance_policy_id, name, category,
                     patterns, pattern_type, except_patterns, rule_type,
                     action, phase, severity, display_mode, block_message,
                     context_note, notification_frequency,
                     notification_channels, webhook_url,
                     trusted_contact_email, crisis_resources_enabled,
                     cooldown_minutes, bypass_protection,
                     bypass_partner_user_id,
                     escalation_session_threshold, escalation_session_action,
                     escalation_window_days, escalation_window_threshold,
                     escalation_window_action, min_context_length,
                     enabled, pending_deactivation_at,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        rule.id, rule.user_id, rule.governance_policy_id,
                        rule.name, rule.category,
                        json.dumps(rule.patterns), rule.pattern_type,
                        json.dumps(rule.except_patterns), rule.rule_type,
                        rule.action, rule.phase, rule.severity,
                        rule.display_mode, rule.block_message,
                        rule.context_note, rule.notification_frequency,
                        json.dumps(rule.notification_channels),
                        rule.webhook_url, rule.trusted_contact_email,
                        int(rule.crisis_resources_enabled),
                        rule.cooldown_minutes, rule.bypass_protection,
                        rule.bypass_partner_user_id,
                        rule.escalation_session_threshold,
                        rule.escalation_session_action,
                        rule.escalation_window_days,
                        rule.escalation_window_threshold,
                        rule.escalation_window_action,
                        rule.min_context_length,
                        int(rule.enabled), rule.pending_deactivation_at,
                        rule.created_at, rule.updated_at,
                    ),
                )
                return rule
            finally:
                conn.close()

    def get_self_monitoring_rule(self, rule_id: str) -> SelfMonitoringRule | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM self_monitoring_rules WHERE id = ?",
                    (rule_id,),
                ).fetchone()
                if not row:
                    return None
                return self._row_to_self_monitoring_rule(row)
            finally:
                conn.close()

    def list_self_monitoring_rules(
        self,
        user_id: str,
        enabled_only: bool = False,
        category: str | None = None,
    ) -> list[SelfMonitoringRule]:
        with self._lock:
            conn = self._connect()
            try:
                query = "SELECT * FROM self_monitoring_rules WHERE user_id = ?"
                params: list[Any] = [str(user_id)]
                if enabled_only:
                    query += " AND enabled = 1"
                if category:
                    query += " AND category = ?"
                    params.append(category)
                query += " ORDER BY created_at DESC"
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_self_monitoring_rule(r) for r in rows]
            finally:
                conn.close()

    def update_self_monitoring_rule(
        self,
        rule_id: str,
        **kwargs: Any,
    ) -> bool:
        allowed_fields = {
            "name", "category", "patterns", "pattern_type", "except_patterns",
            "rule_type", "action", "phase", "severity", "display_mode",
            "block_message", "context_note", "notification_frequency",
            "notification_channels", "webhook_url", "trusted_contact_email",
            "crisis_resources_enabled", "cooldown_minutes", "bypass_protection",
            "bypass_partner_user_id", "escalation_session_threshold",
            "escalation_session_action", "escalation_window_days",
            "escalation_window_threshold", "escalation_window_action",
            "min_context_length", "enabled", "pending_deactivation_at",
            "governance_policy_id", "deactivation_confirmation_token",
            "deactivation_requested_at",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return False
        if "action" in updates and updates["action"] not in ("block", "redact", "notify"):
            raise ValueError(f"Invalid action: {updates['action']}")
        if "phase" in updates and updates["phase"] not in ("input", "output", "both"):
            raise ValueError(f"Invalid phase: {updates['phase']}")
        if "severity" in updates and updates["severity"] not in ("info", "warning", "critical"):
            raise ValueError(f"Invalid severity: {updates['severity']}")
        now = _utcnow_iso()
        set_clauses = []
        params: list[Any] = []
        for k, v in updates.items():
            if k in ("patterns", "except_patterns", "notification_channels"):
                set_clauses.append(f"{k} = ?")
                params.append(json.dumps(v) if v is not None else "[]")
            elif k in ("crisis_resources_enabled", "enabled"):
                set_clauses.append(f"{k} = ?")
                params.append(int(bool(v)))
            else:
                set_clauses.append(f"{k} = ?")
                params.append(v)
        set_clauses.append("updated_at = ?")
        params.append(now)
        params.append(rule_id)
        set_clause_sql = ", ".join(set_clauses)
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    f"UPDATE self_monitoring_rules SET {set_clause_sql} WHERE id = ?",  # nosec B608
                    params,
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def delete_self_monitoring_rule(self, rule_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                # Also clean up alerts and escalation state
                conn.execute("DELETE FROM self_monitoring_alerts WHERE rule_id = ?", (rule_id,))
                conn.execute("DELETE FROM escalation_state WHERE rule_id = ?", (rule_id,))
                result = conn.execute(
                    "DELETE FROM self_monitoring_rules WHERE id = ?",
                    (rule_id,),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    @staticmethod
    def _row_to_self_monitoring_rule(row: sqlite3.Row) -> SelfMonitoringRule:
        def _parse_json_list(raw: str | None) -> list[str]:
            if not raw:
                return []
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, list) else []
            except _GUARDIAN_NONCRITICAL_EXCEPTIONS:
                return []

        return SelfMonitoringRule(
            id=row["id"],
            user_id=row["user_id"],
            governance_policy_id=row["governance_policy_id"],
            name=row["name"],
            category=row["category"],
            patterns=_parse_json_list(row["patterns"]),
            pattern_type=row["pattern_type"],
            except_patterns=_parse_json_list(row["except_patterns"]),
            rule_type=row["rule_type"],
            action=row["action"],
            phase=row["phase"],
            severity=row["severity"],
            display_mode=row["display_mode"],
            block_message=row["block_message"],
            context_note=row["context_note"],
            notification_frequency=row["notification_frequency"],
            notification_channels=_parse_json_list(row["notification_channels"]),
            webhook_url=row["webhook_url"],
            trusted_contact_email=row["trusted_contact_email"],
            crisis_resources_enabled=bool(row["crisis_resources_enabled"]),
            cooldown_minutes=row["cooldown_minutes"],
            bypass_protection=row["bypass_protection"],
            bypass_partner_user_id=row["bypass_partner_user_id"],
            escalation_session_threshold=row["escalation_session_threshold"],
            escalation_session_action=row["escalation_session_action"],
            escalation_window_days=row["escalation_window_days"],
            escalation_window_threshold=row["escalation_window_threshold"],
            escalation_window_action=row["escalation_window_action"],
            min_context_length=row["min_context_length"],
            enabled=bool(row["enabled"]),
            pending_deactivation_at=row["pending_deactivation_at"],
            deactivation_confirmation_token=_safe_get(row, "deactivation_confirmation_token"),
            deactivation_requested_at=_safe_get(row, "deactivation_requested_at"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── Self-Monitoring Alerts ─────────────────────────────────

    def create_self_monitoring_alert(
        self,
        user_id: str,
        rule_id: str,
        rule_name: str = "",
        category: str = "",
        severity: str = "info",
        matched_pattern: str = "",
        context_snippet: str | None = None,
        snippet_mode: str = "full_snippet",
        conversation_id: str | None = None,
        session_id: str | None = None,
        chat_type: str | None = None,
        phase: str = "input",
        action_taken: str = "notified",
        notification_sent: bool = False,
        notification_channels_used: list[str] | None = None,
        crisis_resources_shown: bool = False,
        display_mode: str = "inline_banner",
        escalation_info: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SelfMonitoringAlert:
        now = _utcnow_iso()
        alert = SelfMonitoringAlert(
            id=_new_id(),
            user_id=str(user_id),
            rule_id=rule_id,
            rule_name=rule_name,
            category=category,
            severity=severity,
            matched_pattern=matched_pattern,
            context_snippet=context_snippet,
            snippet_mode=snippet_mode,
            conversation_id=conversation_id,
            session_id=session_id,
            chat_type=chat_type,
            phase=phase,
            action_taken=action_taken,
            notification_sent=notification_sent,
            notification_channels_used=notification_channels_used or [],
            crisis_resources_shown=crisis_resources_shown,
            display_mode=display_mode,
            escalation_info=escalation_info,
            metadata=metadata,
            created_at=now,
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO self_monitoring_alerts
                    (id, user_id, rule_id, rule_name, category, severity,
                     matched_pattern, context_snippet, snippet_mode,
                     conversation_id, session_id, chat_type, phase,
                     action_taken, notification_sent,
                     notification_channels_used, crisis_resources_shown,
                     display_mode, escalation_info, is_read,
                     metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        alert.id, alert.user_id, alert.rule_id,
                        alert.rule_name, alert.category, alert.severity,
                        alert.matched_pattern, alert.context_snippet,
                        alert.snippet_mode, alert.conversation_id,
                        alert.session_id, alert.chat_type, alert.phase,
                        alert.action_taken, int(alert.notification_sent),
                        json.dumps(alert.notification_channels_used),
                        int(alert.crisis_resources_shown),
                        alert.display_mode,
                        json.dumps(alert.escalation_info) if alert.escalation_info else None,
                        int(alert.is_read),
                        json.dumps(alert.metadata) if alert.metadata else None,
                        alert.created_at,
                    ),
                )
                return alert
            finally:
                conn.close()

    def list_self_monitoring_alerts(
        self,
        user_id: str,
        rule_id: str | None = None,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SelfMonitoringAlert]:
        with self._lock:
            conn = self._connect()
            try:
                query = "SELECT * FROM self_monitoring_alerts WHERE user_id = ?"
                params: list[Any] = [str(user_id)]
                if rule_id:
                    query += " AND rule_id = ?"
                    params.append(rule_id)
                if unread_only:
                    query += " AND is_read = 0"
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_self_monitoring_alert(r) for r in rows]
            finally:
                conn.close()

    def mark_alerts_read(self, user_id: str, alert_ids: list[str]) -> int:
        if not alert_ids:
            return 0
        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(alert_ids))
                alert_ids_clause = f"({placeholders})"
                result = conn.execute(
                    f"UPDATE self_monitoring_alerts SET is_read = 1 WHERE user_id = ? AND id IN {alert_ids_clause}",  # nosec B608
                    [str(user_id), *alert_ids],
                )
                return result.rowcount or 0
            finally:
                conn.close()

    def count_unread_alerts(self, user_id: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM self_monitoring_alerts WHERE user_id = ? AND is_read = 0",
                    (str(user_id),),
                ).fetchone()
                return row["cnt"] if row else 0
            finally:
                conn.close()

    @staticmethod
    def _row_to_self_monitoring_alert(row: sqlite3.Row) -> SelfMonitoringAlert:
        def _parse_json(raw: str | None, default: Any = None) -> Any:
            if not raw:
                return default
            try:
                return json.loads(raw)
            except _GUARDIAN_NONCRITICAL_EXCEPTIONS:
                return default

        return SelfMonitoringAlert(
            id=row["id"],
            user_id=row["user_id"],
            rule_id=row["rule_id"],
            rule_name=row["rule_name"],
            category=row["category"],
            severity=row["severity"],
            matched_pattern=row["matched_pattern"],
            context_snippet=row["context_snippet"],
            snippet_mode=row["snippet_mode"],
            conversation_id=row["conversation_id"],
            session_id=row["session_id"],
            chat_type=row["chat_type"],
            phase=row["phase"],
            action_taken=row["action_taken"],
            notification_sent=bool(row["notification_sent"]),
            notification_channels_used=_parse_json(row["notification_channels_used"], []),
            crisis_resources_shown=bool(row["crisis_resources_shown"]),
            display_mode=row["display_mode"],
            escalation_info=_parse_json(row["escalation_info"]),
            is_read=bool(row["is_read"]),
            metadata=_parse_json(row["metadata"]),
            created_at=row["created_at"],
        )

    # ── Escalation State ───────────────────────────────────────

    def get_escalation_state(self, rule_id: str, user_id: str) -> EscalationState | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM escalation_state WHERE rule_id = ? AND user_id = ?",
                    (rule_id, str(user_id)),
                ).fetchone()
                if not row:
                    return None
                return EscalationState(
                    rule_id=row["rule_id"],
                    user_id=row["user_id"],
                    session_id=row["session_id"],
                    session_trigger_count=row["session_trigger_count"],
                    window_trigger_count=row["window_trigger_count"],
                    current_escalated_action=row["current_escalated_action"],
                    escalated_at=row["escalated_at"],
                    cooldown_until=row["cooldown_until"],
                    updated_at=row["updated_at"],
                )
            finally:
                conn.close()

    def upsert_escalation_state(
        self,
        rule_id: str,
        user_id: str,
        session_id: str | None = None,
        session_trigger_count: int = 0,
        window_trigger_count: int = 0,
        current_escalated_action: str | None = None,
        escalated_at: str | None = None,
        cooldown_until: str | None = None,
    ) -> None:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO escalation_state
                    (rule_id, user_id, session_id, session_trigger_count,
                     window_trigger_count, current_escalated_action,
                     escalated_at, cooldown_until, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(rule_id, user_id) DO UPDATE SET
                        session_id = excluded.session_id,
                        session_trigger_count = excluded.session_trigger_count,
                        window_trigger_count = excluded.window_trigger_count,
                        current_escalated_action = excluded.current_escalated_action,
                        escalated_at = excluded.escalated_at,
                        cooldown_until = excluded.cooldown_until,
                        updated_at = excluded.updated_at""",
                    (
                        rule_id, str(user_id), session_id,
                        session_trigger_count, window_trigger_count,
                        current_escalated_action, escalated_at,
                        cooldown_until, now,
                    ),
                )
            finally:
                conn.close()

    def reset_escalation_state(self, rule_id: str, user_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    "DELETE FROM escalation_state WHERE rule_id = ? AND user_id = ?",
                    (rule_id, str(user_id)),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    # ── Dedup helpers ──────────────────────────────────────────

    def has_recent_alert(
        self,
        user_id: str,
        rule_id: str,
        conversation_id: str | None = None,
        since_iso: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Check if an alert already exists for dedup purposes."""
        with self._lock:
            conn = self._connect()
            try:
                query = "SELECT 1 FROM self_monitoring_alerts WHERE user_id = ? AND rule_id = ?"
                params: list[Any] = [str(user_id), rule_id]
                if conversation_id:
                    query += " AND conversation_id = ?"
                    params.append(conversation_id)
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                if since_iso:
                    query += " AND created_at >= ?"
                    params.append(since_iso)
                query += " LIMIT 1"
                row = conn.execute(query, params).fetchone()
                return row is not None
            finally:
                conn.close()

    def count_alerts_in_window(
        self,
        user_id: str,
        rule_id: str,
        window_days: int,
    ) -> int:
        """Count alerts for a rule within the last *window_days* days.

        Used by escalation logic to compute a rolling-window trigger count
        instead of a monotonically increasing counter.
        """
        if window_days <= 0:
            return 0
        since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM self_monitoring_alerts "
                    "WHERE user_id = ? AND rule_id = ? AND created_at >= ?",
                    (str(user_id), rule_id, since),
                ).fetchone()
                return row["cnt"] if row else 0
            finally:
                conn.close()

    # ── Analytics / Aggregation ────────────────────────────────

    def aggregate_alerts_by_category(
        self, user_id: str, since_iso: str | None = None,
    ) -> list[dict[str, Any]]:
        """Group alerts by category, return [{category, count}]."""
        with self._lock:
            conn = self._connect()
            try:
                query = (
                    "SELECT category, COUNT(*) AS cnt "
                    "FROM self_monitoring_alerts WHERE user_id = ?"
                )
                params: list[Any] = [str(user_id)]
                if since_iso:
                    query += " AND created_at >= ?"
                    params.append(since_iso)
                query += " GROUP BY category ORDER BY cnt DESC"
                rows = conn.execute(query, params).fetchall()
                return [{"category": r["category"], "count": r["cnt"]} for r in rows]
            finally:
                conn.close()

    def aggregate_alerts_by_severity(
        self, user_id: str, since_iso: str | None = None,
    ) -> list[dict[str, Any]]:
        """Group alerts by severity, return [{severity, count}]."""
        with self._lock:
            conn = self._connect()
            try:
                query = (
                    "SELECT severity, COUNT(*) AS cnt "
                    "FROM self_monitoring_alerts WHERE user_id = ?"
                )
                params: list[Any] = [str(user_id)]
                if since_iso:
                    query += " AND created_at >= ?"
                    params.append(since_iso)
                query += " GROUP BY severity ORDER BY cnt DESC"
                rows = conn.execute(query, params).fetchall()
                return [{"severity": r["severity"], "count": r["cnt"]} for r in rows]
            finally:
                conn.close()

    def aggregate_alerts_by_time(
        self, user_id: str, bucket: str = "day", since_iso: str | None = None,
    ) -> list[dict[str, Any]]:
        """Group alerts by time bucket (day|week|month), return [{bucket, count}]."""
        fmt_map = {"day": "%Y-%m-%d", "week": "%Y-W%W", "month": "%Y-%m"}
        fmt = fmt_map.get(bucket, "%Y-%m-%d")
        with self._lock:
            conn = self._connect()
            try:
                query = (
                    f"SELECT strftime('{fmt}', created_at) AS bucket, COUNT(*) AS cnt "  # nosec B608
                    f"FROM self_monitoring_alerts WHERE user_id = ?"
                )
                params: list[Any] = [str(user_id)]
                if since_iso:
                    query += " AND created_at >= ?"
                    params.append(since_iso)
                query += " GROUP BY bucket ORDER BY bucket"
                rows = conn.execute(query, params).fetchall()
                return [{"bucket": r["bucket"], "count": r["cnt"]} for r in rows]
            finally:
                conn.close()

    def get_top_matched_patterns(
        self, user_id: str, limit: int = 10, since_iso: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return most-frequent matched_pattern values, [{pattern, count}]."""
        with self._lock:
            conn = self._connect()
            try:
                query = (
                    "SELECT matched_pattern, COUNT(*) AS cnt "
                    "FROM self_monitoring_alerts WHERE user_id = ?"
                )
                params: list[Any] = [str(user_id)]
                if since_iso:
                    query += " AND created_at >= ?"
                    params.append(since_iso)
                query += " GROUP BY matched_pattern ORDER BY cnt DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(query, params).fetchall()
                return [{"pattern": r["matched_pattern"], "count": r["cnt"]} for r in rows]
            finally:
                conn.close()

    def get_escalation_summary(self, user_id: str) -> list[dict[str, Any]]:
        """Return current escalation states for a user."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM escalation_state WHERE user_id = ?",
                    (str(user_id),),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ── Family Wizard Drafts ─────────────────────────────────

    @staticmethod
    def _loads_json_or_default(raw_value: str | None, default: Any) -> Any:
        if not raw_value:
            return default
        try:
            return json.loads(raw_value)
        except _GUARDIAN_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Guardian DB JSON parse failed; using default fallback. value_preview={} error={}",
                str(raw_value)[:120],
                exc,
            )
            return default

    def create_household_draft(
        self,
        owner_user_id: str,
        mode: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if mode not in ("family", "institutional"):
            raise ValueError(f"Invalid household mode: {mode}")
        now = _utcnow_iso()
        draft_id = _new_id()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO guardian_household_drafts
                    (id, owner_user_id, name, mode, status, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'draft', ?, ?, ?)""",
                    (
                        draft_id,
                        str(owner_user_id),
                        name,
                        mode,
                        json.dumps(metadata) if metadata else None,
                        now,
                        now,
                    ),
                )
                return draft_id
            finally:
                conn.close()

    def get_household_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM guardian_household_drafts WHERE id = ?",
                    (draft_id,),
                ).fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "owner_user_id": row["owner_user_id"],
                    "name": row["name"],
                    "mode": row["mode"],
                    "status": row["status"],
                    "metadata": self._loads_json_or_default(row["metadata"], None),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            finally:
                conn.close()

    def get_latest_household_draft(self, owner_user_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT * FROM guardian_household_drafts
                       WHERE owner_user_id = ?
                       ORDER BY updated_at DESC, created_at DESC
                       LIMIT 1""",
                    (str(owner_user_id),),
                ).fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "owner_user_id": row["owner_user_id"],
                    "name": row["name"],
                    "mode": row["mode"],
                    "status": row["status"],
                    "metadata": self._loads_json_or_default(row["metadata"], None),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            finally:
                conn.close()

    def list_household_drafts(self, owner_user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM guardian_household_drafts
                       WHERE owner_user_id = ?
                       ORDER BY updated_at DESC, created_at DESC""",
                    (str(owner_user_id),),
                ).fetchall()
                return [
                    {
                        "id": row["id"],
                        "owner_user_id": row["owner_user_id"],
                        "name": row["name"],
                        "mode": row["mode"],
                        "status": row["status"],
                        "metadata": self._loads_json_or_default(row["metadata"], None),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def update_household_draft(self, draft_id: str, **fields: Any) -> bool:
        allowed_fields = {"name", "mode", "status", "metadata"}
        updates: list[str] = []
        params: list[Any] = []
        for key, value in fields.items():
            if key not in allowed_fields:
                continue
            if key == "mode" and value not in ("family", "institutional"):
                raise ValueError(f"Invalid household mode: {value}")
            if key == "metadata":
                value = json.dumps(value) if value is not None else None
            updates.append(f"{key} = ?")
            params.append(value)
        if not updates:
            return False
        updates.append("updated_at = ?")
        params.append(_utcnow_iso())
        params.append(draft_id)
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    f"UPDATE guardian_household_drafts SET {', '.join(updates)} WHERE id = ?",  # nosec B608
                    params,
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def add_household_member_draft(
        self,
        household_draft_id: str,
        role: str,
        display_name: str,
        user_id: str | None = None,
        email: str | None = None,
        invite_required: bool = True,
        invite_status: str = "pending",
        account_mode: str = "existing_account",
        provisioning_status: str = "not_started",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if role not in ("guardian", "dependent", "caregiver"):
            raise ValueError(f"Invalid member role: {role}")
        if account_mode not in ("existing_account", "invite_new"):
            raise ValueError(f"Invalid account_mode: {account_mode}")
        if provisioning_status not in (
            "not_started",
            "invite_ready",
            "sent",
            "accepted",
            "expired",
            "failed",
        ):
            raise ValueError(f"Invalid provisioning_status: {provisioning_status}")
        now = _utcnow_iso()
        member_id = _new_id()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO guardian_household_member_drafts
                    (id, household_draft_id, role, display_name, user_id, email,
                     invite_required, invite_status, account_mode, provisioning_status,
                     metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        member_id,
                        household_draft_id,
                        role,
                        display_name,
                        str(user_id) if user_id else None,
                        email,
                        int(invite_required),
                        invite_status,
                        account_mode,
                        provisioning_status,
                        json.dumps(metadata) if metadata else None,
                        now,
                        now,
                    ),
                )
                return member_id
            finally:
                conn.close()

    def remove_household_member_draft(self, member_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    "DELETE FROM guardian_household_member_drafts WHERE id = ?",
                    (member_id,),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def list_household_member_drafts(self, household_draft_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM guardian_household_member_drafts
                       WHERE household_draft_id = ?
                       ORDER BY created_at ASC""",
                    (household_draft_id,),
                ).fetchall()
                return [
                    {
                        "id": row["id"],
                        "household_draft_id": row["household_draft_id"],
                        "role": row["role"],
                        "display_name": row["display_name"],
                        "user_id": row["user_id"],
                        "email": row["email"],
                        "invite_required": bool(row["invite_required"]),
                        "invite_status": row["invite_status"],
                        "account_mode": _safe_get(row, "account_mode", "existing_account"),
                        "provisioning_status": _safe_get(
                            row,
                            "provisioning_status",
                            "not_started",
                        ),
                        "metadata": self._loads_json_or_default(row["metadata"], None),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def get_household_member_draft(self, member_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM guardian_household_member_drafts WHERE id = ?",
                    (member_id,),
                ).fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "household_draft_id": row["household_draft_id"],
                    "role": row["role"],
                    "display_name": row["display_name"],
                    "user_id": row["user_id"],
                    "email": row["email"],
                    "invite_required": bool(row["invite_required"]),
                    "invite_status": row["invite_status"],
                    "account_mode": _safe_get(row, "account_mode", "existing_account"),
                    "provisioning_status": _safe_get(
                        row,
                        "provisioning_status",
                        "not_started",
                    ),
                    "metadata": self._loads_json_or_default(row["metadata"], None),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            finally:
                conn.close()

    def resolve_household_member_draft_user(self, member_id: str, user_id: str) -> bool:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    """UPDATE guardian_household_member_drafts
                       SET user_id = ?,
                           invite_status = 'accepted',
                           provisioning_status = 'accepted',
                           updated_at = ?
                       WHERE id = ?""",
                    (str(user_id), now, member_id),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def _household_member_invite_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "household_draft_id": row["household_draft_id"],
            "member_draft_id": row["member_draft_id"],
            "status": row["status"],
            "delivery_channel": row["delivery_channel"],
            "delivery_target": row["delivery_target"],
            "invite_token": row["invite_token"],
            "resend_count": row["resend_count"],
            "last_sent_at": row["last_sent_at"],
            "accepted_at": row["accepted_at"],
            "expires_at": row["expires_at"],
            "revoked_at": row["revoked_at"],
            "failure_reason": row["failure_reason"],
            "metadata": self._loads_json_or_default(row["metadata"], None),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_household_member_invite(
        self,
        household_draft_id: str,
        member_draft_id: str,
        delivery_channel: str,
        delivery_target: str | None = None,
        status: str = "ready",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = _utcnow_iso()
        invite_id = _new_id()
        with self._lock:
            conn = self._connect()
            try:
                household_row = conn.execute(
                    """SELECT owner_user_id
                       FROM guardian_household_drafts
                       WHERE id = ?""",
                    (household_draft_id,),
                ).fetchone()
                if not household_row:
                    raise ValueError("Household draft not found")
                invite_token = f"{household_row['owner_user_id']}.{invite_id}.{_new_id()}"
                conn.execute(
                    """INSERT INTO guardian_household_member_invites
                    (id, household_draft_id, member_draft_id, status, delivery_channel,
                     delivery_target, invite_token, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        invite_id,
                        household_draft_id,
                        member_draft_id,
                        status,
                        delivery_channel,
                        delivery_target,
                        invite_token,
                        json.dumps(metadata) if metadata else None,
                        now,
                        now,
                    ),
                )
                return invite_id
            finally:
                conn.close()

    def get_household_member_invite(self, invite_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM guardian_household_member_invites WHERE id = ?",
                    (invite_id,),
                ).fetchone()
                if not row:
                    return None
                return self._household_member_invite_from_row(row)
            finally:
                conn.close()

    def get_household_member_invite_by_token(self, invite_token: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT * FROM guardian_household_member_invites
                       WHERE invite_token = ?""",
                    (invite_token,),
                ).fetchone()
                if not row:
                    return None
                return self._household_member_invite_from_row(row)
            finally:
                conn.close()

    def list_household_member_invites(
        self,
        household_draft_id: str,
        member_draft_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if member_draft_id:
                    rows = conn.execute(
                        """SELECT * FROM guardian_household_member_invites
                           WHERE household_draft_id = ? AND member_draft_id = ?
                           ORDER BY created_at DESC""",
                        (household_draft_id, member_draft_id),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM guardian_household_member_invites
                           WHERE household_draft_id = ?
                           ORDER BY created_at DESC""",
                        (household_draft_id,),
                    ).fetchall()
                return [self._household_member_invite_from_row(row) for row in rows]
            finally:
                conn.close()

    def get_latest_household_member_invite_for_member(
        self,
        member_draft_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT * FROM guardian_household_member_invites
                       WHERE member_draft_id = ?
                       ORDER BY created_at DESC
                       LIMIT 1""",
                    (member_draft_id,),
                ).fetchone()
                if not row:
                    return None
                return self._household_member_invite_from_row(row)
            finally:
                conn.close()

    def update_household_member_invite_status(
        self,
        invite_id: str,
        *,
        status: str,
        last_sent_at: str | None = None,
        accepted_at: str | None = None,
        expires_at: str | None = None,
        revoked_at: str | None = None,
        failure_reason: str | None = None,
        increment_resend_count: bool = False,
    ) -> bool:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                current = conn.execute(
                    """SELECT resend_count
                       FROM guardian_household_member_invites
                       WHERE id = ?""",
                    (invite_id,),
                ).fetchone()
                if not current:
                    return False
                resend_count = int(current["resend_count"] or 0)
                resolved_last_sent_at = last_sent_at
                if increment_resend_count:
                    resend_count += 1
                    if resolved_last_sent_at is None:
                        resolved_last_sent_at = now
                result = conn.execute(
                    """UPDATE guardian_household_member_invites
                       SET status = ?,
                           resend_count = ?,
                           last_sent_at = COALESCE(?, last_sent_at),
                           accepted_at = COALESCE(?, accepted_at),
                           expires_at = COALESCE(?, expires_at),
                           revoked_at = COALESCE(?, revoked_at),
                           failure_reason = ?,
                           updated_at = ?
                       WHERE id = ?""",
                    (
                        status,
                        resend_count,
                        resolved_last_sent_at,
                        accepted_at,
                        expires_at,
                        revoked_at,
                        failure_reason,
                        now,
                        invite_id,
                    ),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def reissue_household_member_invite(self, invite_id: str) -> str:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT * FROM guardian_household_member_invites
                       WHERE id = ?""",
                    (invite_id,),
                ).fetchone()
                if not row:
                    raise ValueError("Invite not found")
                conn.execute(
                    """UPDATE guardian_household_member_invites
                       SET status = 'revoked',
                           revoked_at = ?,
                           updated_at = ?
                       WHERE id = ?""",
                    (now, now, invite_id),
                )
                replacement_id = _new_id()
                household_row = conn.execute(
                    """SELECT owner_user_id
                       FROM guardian_household_drafts
                       WHERE id = ?""",
                    (row["household_draft_id"],),
                ).fetchone()
                if not household_row:
                    raise ValueError("Household draft not found")
                conn.execute(
                    """INSERT INTO guardian_household_member_invites
                    (id, household_draft_id, member_draft_id, status, delivery_channel,
                     delivery_target, invite_token, resend_count, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, 'ready', ?, ?, ?, 0, ?, ?, ?)""",
                    (
                        replacement_id,
                        row["household_draft_id"],
                        row["member_draft_id"],
                        row["delivery_channel"],
                        row["delivery_target"],
                        f"{household_row['owner_user_id']}.{replacement_id}.{_new_id()}",
                        row["metadata"],
                        now,
                        now,
                    ),
                )
                return replacement_id
            finally:
                conn.close()

    def mark_household_member_invite_resent(self, member_id: str) -> bool:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT metadata FROM guardian_household_member_drafts WHERE id = ?",
                    (member_id,),
                ).fetchone()
                if not row:
                    return False

                metadata = self._loads_json_or_default(row["metadata"], {})
                if not isinstance(metadata, dict):
                    metadata = {}
                prior_count = metadata.get("invite_resend_count", 0)
                try:
                    resend_count = int(prior_count)
                except (TypeError, ValueError):
                    resend_count = 0
                metadata["invite_resend_count"] = max(0, resend_count) + 1
                metadata["invite_last_resent_at"] = now

                result = conn.execute(
                    """UPDATE guardian_household_member_drafts
                       SET invite_status = 'pending',
                           metadata = ?,
                           updated_at = ?
                       WHERE id = ?""",
                    (json.dumps(metadata), now, member_id),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def create_relationship_draft(
        self,
        household_draft_id: str,
        guardian_member_draft_id: str,
        dependent_member_draft_id: str,
        relationship_type: str = "parent",
        dependent_visible: bool = True,
        status: str = "pending",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if relationship_type not in ("parent", "legal_guardian", "institutional"):
            raise ValueError(f"Invalid relationship_type: {relationship_type}")
        if status not in ("pending", "pending_provisioning", "active", "declined", "revoked"):
            raise ValueError(f"Invalid relationship draft status: {status}")
        now = _utcnow_iso()
        relationship_draft_id = _new_id()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO guardian_relationship_drafts
                    (id, household_draft_id, guardian_member_draft_id, dependent_member_draft_id,
                     relationship_type, dependent_visible, status, relationship_id,
                     metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)""",
                    (
                        relationship_draft_id,
                        household_draft_id,
                        guardian_member_draft_id,
                        dependent_member_draft_id,
                        relationship_type,
                        int(dependent_visible),
                        status,
                        json.dumps(metadata) if metadata else None,
                        now,
                        now,
                    ),
                )
                return relationship_draft_id
            except sqlite3.IntegrityError as e:
                raise ValueError(
                    "Relationship draft already exists for this guardian/dependent pair"
                ) from e
            finally:
                conn.close()

    def list_relationship_drafts(self, household_draft_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM guardian_relationship_drafts
                       WHERE household_draft_id = ?
                       ORDER BY created_at ASC""",
                    (household_draft_id,),
                ).fetchall()
                return [
                    {
                        "id": row["id"],
                        "household_draft_id": row["household_draft_id"],
                        "guardian_member_draft_id": row["guardian_member_draft_id"],
                        "dependent_member_draft_id": row["dependent_member_draft_id"],
                        "relationship_type": row["relationship_type"],
                        "dependent_visible": bool(row["dependent_visible"]),
                        "status": row["status"],
                        "relationship_id": row["relationship_id"],
                        "metadata": self._loads_json_or_default(row["metadata"], None),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def get_relationship_draft(self, relationship_draft_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM guardian_relationship_drafts WHERE id = ?",
                    (relationship_draft_id,),
                ).fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "household_draft_id": row["household_draft_id"],
                    "guardian_member_draft_id": row["guardian_member_draft_id"],
                    "dependent_member_draft_id": row["dependent_member_draft_id"],
                    "relationship_type": row["relationship_type"],
                    "dependent_visible": bool(row["dependent_visible"]),
                    "status": row["status"],
                    "relationship_id": row["relationship_id"],
                    "metadata": self._loads_json_or_default(row["metadata"], None),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            finally:
                conn.close()

    def link_relationship_draft(
        self,
        relationship_draft_id: str,
        relationship_id: str,
        status: str = "pending",
    ) -> bool:
        if status not in ("pending", "pending_provisioning", "active", "declined", "revoked"):
            raise ValueError(f"Invalid relationship draft status: {status}")
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    """UPDATE guardian_relationship_drafts
                       SET relationship_id = ?, status = ?, updated_at = ?
                       WHERE id = ?""",
                    (relationship_id, status, _utcnow_iso(), relationship_draft_id),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def create_guardrail_plan_draft(
        self,
        household_draft_id: str,
        relationship_draft_id: str,
        template_id: str,
        dependent_member_draft_id: str | None = None,
        dependent_user_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not template_id:
            raise ValueError("template_id is required")
        now = _utcnow_iso()
        plan_id = _new_id()
        with self._lock:
            conn = self._connect()
            try:
                resolved_member_draft_id = dependent_member_draft_id
                if not resolved_member_draft_id:
                    relationship_row = conn.execute(
                        """SELECT dependent_member_draft_id
                           FROM guardian_relationship_drafts
                           WHERE id = ?""",
                        (relationship_draft_id,),
                    ).fetchone()
                    if not relationship_row:
                        raise ValueError("relationship_draft_id is invalid")
                    resolved_member_draft_id = relationship_row["dependent_member_draft_id"]
                conn.execute(
                    """INSERT INTO guardian_guardrail_plan_drafts
                    (id, household_draft_id, dependent_member_draft_id, dependent_user_id,
                     relationship_draft_id, template_id, overrides, status,
                     materialized_policy_id, failure_reason, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', NULL, NULL, ?, ?, ?)""",
                    (
                        plan_id,
                        household_draft_id,
                        resolved_member_draft_id,
                        str(dependent_user_id) if dependent_user_id else None,
                        relationship_draft_id,
                        template_id,
                        json.dumps(overrides or {}),
                        json.dumps(metadata) if metadata else None,
                        now,
                        now,
                    ),
                )
                return plan_id
            finally:
                conn.close()

    def list_guardrail_plan_drafts(
        self,
        household_draft_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if status:
                    rows = conn.execute(
                        """SELECT * FROM guardian_guardrail_plan_drafts
                           WHERE household_draft_id = ? AND status = ?
                           ORDER BY created_at ASC""",
                        (household_draft_id, status),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM guardian_guardrail_plan_drafts
                           WHERE household_draft_id = ?
                           ORDER BY created_at ASC""",
                        (household_draft_id,),
                    ).fetchall()
                return [
                    {
                        "id": row["id"],
                        "household_draft_id": row["household_draft_id"],
                        "dependent_member_draft_id": row["dependent_member_draft_id"],
                        "dependent_user_id": row["dependent_user_id"],
                        "relationship_draft_id": row["relationship_draft_id"],
                        "template_id": row["template_id"],
                        "overrides": self._loads_json_or_default(row["overrides"], {}),
                        "status": row["status"],
                        "materialized_policy_id": row["materialized_policy_id"],
                        "failure_reason": row["failure_reason"],
                        "metadata": self._loads_json_or_default(row["metadata"], None),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def get_guardrail_plan_draft(self, plan_draft_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM guardian_guardrail_plan_drafts WHERE id = ?",
                    (plan_draft_id,),
                ).fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "household_draft_id": row["household_draft_id"],
                    "dependent_member_draft_id": row["dependent_member_draft_id"],
                    "dependent_user_id": row["dependent_user_id"],
                    "relationship_draft_id": row["relationship_draft_id"],
                    "template_id": row["template_id"],
                    "overrides": self._loads_json_or_default(row["overrides"], {}),
                    "status": row["status"],
                    "materialized_policy_id": row["materialized_policy_id"],
                    "failure_reason": row["failure_reason"],
                    "metadata": self._loads_json_or_default(row["metadata"], None),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            finally:
                conn.close()

    def resolve_guardrail_plan_drafts_for_member(
        self,
        member_draft_id: str,
        dependent_user_id: str,
    ) -> int:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    """UPDATE guardian_guardrail_plan_drafts
                       SET dependent_user_id = ?, updated_at = ?
                       WHERE dependent_member_draft_id = ?""",
                    (str(dependent_user_id), now, member_draft_id),
                )
                return result.rowcount or 0
            finally:
                conn.close()

    def list_pending_plans_for_relationship_draft(
        self,
        relationship_draft_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM guardian_guardrail_plan_drafts
                       WHERE relationship_draft_id = ? AND status = 'queued'
                       ORDER BY created_at ASC""",
                    (relationship_draft_id,),
                ).fetchall()
                return [
                    {
                        "id": row["id"],
                        "household_draft_id": row["household_draft_id"],
                        "dependent_member_draft_id": row["dependent_member_draft_id"],
                        "dependent_user_id": row["dependent_user_id"],
                        "relationship_draft_id": row["relationship_draft_id"],
                        "template_id": row["template_id"],
                        "overrides": self._loads_json_or_default(row["overrides"], {}),
                        "status": row["status"],
                        "materialized_policy_id": row["materialized_policy_id"],
                        "failure_reason": row["failure_reason"],
                        "metadata": self._loads_json_or_default(row["metadata"], None),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def list_pending_plans_for_relationship(
        self,
        relationship_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT p.*
                       FROM guardian_guardrail_plan_drafts p
                       JOIN guardian_relationship_drafts r ON p.relationship_draft_id = r.id
                       WHERE r.relationship_id = ? AND p.status = 'queued'
                       ORDER BY p.created_at ASC""",
                    (relationship_id,),
                ).fetchall()
                return [
                    {
                        "id": row["id"],
                        "household_draft_id": row["household_draft_id"],
                        "dependent_member_draft_id": row["dependent_member_draft_id"],
                        "dependent_user_id": row["dependent_user_id"],
                        "relationship_draft_id": row["relationship_draft_id"],
                        "template_id": row["template_id"],
                        "overrides": self._loads_json_or_default(row["overrides"], {}),
                        "status": row["status"],
                        "materialized_policy_id": row["materialized_policy_id"],
                        "failure_reason": row["failure_reason"],
                        "metadata": self._loads_json_or_default(row["metadata"], None),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def update_guardrail_plan_draft_status(
        self,
        plan_draft_id: str,
        status: str,
        materialized_policy_id: str | None = None,
        failure_reason: str | None = None,
    ) -> bool:
        if status not in ("queued", "active", "failed"):
            raise ValueError(f"Invalid plan status: {status}")
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    """UPDATE guardian_guardrail_plan_drafts
                       SET status = ?,
                           materialized_policy_id = ?,
                           failure_reason = ?,
                           updated_at = ?
                       WHERE id = ?""",
                    (
                        status,
                        materialized_policy_id,
                        failure_reason,
                        _utcnow_iso(),
                        plan_draft_id,
                    ),
                )
                return (result.rowcount or 0) > 0
            finally:
                conn.close()

    def record_activation_run(
        self,
        household_draft_id: str,
        relationship_id: str | None,
        dependent_user_id: str | None,
        plan_draft_id: str | None,
        status: str,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if status not in ("queued", "active", "failed"):
            raise ValueError(f"Invalid activation status: {status}")
        run_id = _new_id()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO guardian_activation_runs
                    (id, household_draft_id, relationship_id, dependent_user_id, plan_draft_id,
                     status, detail, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        household_draft_id,
                        relationship_id,
                        str(dependent_user_id) if dependent_user_id else None,
                        plan_draft_id,
                        status,
                        detail,
                        json.dumps(metadata) if metadata else None,
                        _utcnow_iso(),
                    ),
                )
                return run_id
            finally:
                conn.close()

    def list_activation_runs(
        self,
        household_draft_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM guardian_activation_runs
                       WHERE household_draft_id = ?
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (household_draft_id, int(limit)),
                ).fetchall()
                return [
                    {
                        "id": row["id"],
                        "household_draft_id": row["household_draft_id"],
                        "relationship_id": row["relationship_id"],
                        "dependent_user_id": row["dependent_user_id"],
                        "plan_draft_id": row["plan_draft_id"],
                        "status": row["status"],
                        "detail": row["detail"],
                        "metadata": self._loads_json_or_default(row["metadata"], None),
                        "created_at": row["created_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    # ── Cleanup ────────────────────────────────────────────────

    def delete_all_for_user(self, user_id: str) -> dict[str, int]:
        """Delete all guardian + self-monitoring data for a user. Returns counts."""
        counts: dict[str, int] = {
            "relationships": 0, "policies": 0, "audit_entries": 0,
            "self_monitoring_rules": 0, "self_monitoring_alerts": 0,
            "governance_policies": 0,
        }
        uid = str(user_id)
        with self._lock:
            conn = self._connect()
            try:
                begin_immediate_if_needed(conn)
                # Guardian relationships and linked data
                rows = conn.execute(
                    """SELECT id FROM guardian_relationships
                       WHERE guardian_user_id = ? OR dependent_user_id = ?""",
                    (uid, uid),
                ).fetchall()
                rel_ids = [r["id"] for r in rows]
                if rel_ids:
                    placeholders = ",".join("?" * len(rel_ids))
                    rel_ids_clause = f"({placeholders})"
                    result = conn.execute(
                        f"DELETE FROM supervision_audit_log WHERE relationship_id IN {rel_ids_clause}",  # nosec B608
                        rel_ids,
                    )
                    counts["audit_entries"] = result.rowcount or 0
                    result = conn.execute(
                        f"DELETE FROM supervised_policies WHERE relationship_id IN {rel_ids_clause}",  # nosec B608
                        rel_ids,
                    )
                    counts["policies"] = result.rowcount or 0
                    result = conn.execute(
                        f"DELETE FROM guardian_relationships WHERE id IN {rel_ids_clause}",  # nosec B608
                        rel_ids,
                    )
                    counts["relationships"] = result.rowcount or 0

                # Self-monitoring data
                result = conn.execute(
                    "DELETE FROM escalation_state WHERE user_id = ?", (uid,),
                )
                result = conn.execute(
                    "DELETE FROM self_monitoring_alerts WHERE user_id = ?", (uid,),
                )
                counts["self_monitoring_alerts"] = result.rowcount or 0
                result = conn.execute(
                    "DELETE FROM self_monitoring_rules WHERE user_id = ?", (uid,),
                )
                counts["self_monitoring_rules"] = result.rowcount or 0
                result = conn.execute(
                    "DELETE FROM governance_policies WHERE owner_user_id = ?", (uid,),
                )
                counts["governance_policies"] = result.rowcount or 0

                conn.execute("COMMIT")
                return counts
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
