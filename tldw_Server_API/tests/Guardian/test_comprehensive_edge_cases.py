"""
Stage 7: Comprehensive edge-case, error-path, and integration tests.

Covers gaps across all implemented features:
  1. Escalation edge cases (window boundary, simultaneous, session change)
  2. Notification delivery (severity filtering, generic payloads, settings)
  3. Import/export (malformed JSON, partial import, conflicting IDs)
  4. Analytics (single entry, boundary buckets, tied patterns, multi-user)
  5. Category taxonomy (compiled patterns, case sensitivity)
  6. Semantic matcher (multiple refs, custom prompt)
  7. Guardian DB edge cases (all fields, filters)
  8. Integration (full pipeline, redact, block, schedule, multi-message)
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
from tldw_Server_API.app.core.Moderation.category_taxonomy import (
    CATEGORY_TAXONOMY,
    get_category_patterns,
)
from tldw_Server_API.app.core.Moderation.governance_io import (
    export_governance_rules,
    export_to_json,
    import_governance_rules,
)
from tldw_Server_API.app.core.Moderation.semantic_matcher import SemanticMatcher
from tldw_Server_API.app.core.Moderation.supervised_policy import (
    SupervisedPolicyEngine,
)
from tldw_Server_API.app.core.Monitoring.self_monitoring_service import (
    SelfMonitoringService,
)


class _StubAuditService:
    """No-op audit service so mandatory moderation-audit writes succeed in tests."""

    async def log_event(self, **kwargs):
        return None

    async def flush(self, *, raise_on_failure=False):
        return None


# ── Shared fixtures ─────────────────────────────────────────


@pytest.fixture()
def db(tmp_path):
    return GuardianDB(str(tmp_path / "test_comprehensive.db"))


@pytest.fixture()
def svc(db):
    return SelfMonitoringService(db)


@pytest.fixture()
def engine(db):
    return SupervisedPolicyEngine(db)


def _create_rule(db, **kwargs):
    defaults = {
        "user_id": "user1",
        "name": "Test Rule",
        "patterns": ["test_word"],
        "notification_frequency": "every_message",
    }
    defaults.update(kwargs)
    return db.create_self_monitoring_rule(**defaults)


def _setup_active_relationship(db, guardian="guardian1", dependent="child1"):
    rel = db.create_relationship(guardian, dependent)
    db.accept_relationship(rel.id)
    return rel


# ═══════════════════════════════════════════════════════════════
# 1. ESCALATION EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestEscalationEdgeCases:
    def test_session_change_resets_session_counter(self, db, svc):
        """Changing session_id should reset the session trigger counter."""
        _create_rule(
            db, patterns=["trigger"],
            action="notify",
            escalation_session_threshold=3,
            escalation_session_action="block",
        )
        # Trigger twice in session s1 (not at threshold yet)
        svc.check_text("trigger", "user1", session_id="s1")
        r2 = svc.check_text("trigger", "user1", session_id="s1")
        assert r2.action == "notify"

        # Switch to new session — counter resets
        r3 = svc.check_text("trigger", "user1", session_id="s2")
        assert r3.action == "notify"  # Counter = 1, not 3

    def test_both_session_and_window_escalation_independently(self, db, svc):
        """Session and window escalation should work independently."""
        _create_rule(
            db, patterns=["trigger"],
            action="notify",
            escalation_session_threshold=5,
            escalation_session_action="redact",
            escalation_window_days=7,
            escalation_window_threshold=3,
            escalation_window_action="block",
        )
        # Trigger 3 times across different sessions (hits window threshold)
        for i in range(3):
            r = svc.check_text("trigger", "user1", session_id=f"s{i}")
        # Window threshold reached, should escalate to block
        assert r.action == "block"

    def test_escalation_with_zero_session_threshold_uses_window(self, db, svc):
        """When session threshold is 0, only window escalation applies."""
        _create_rule(
            db, patterns=["trigger"],
            action="notify",
            escalation_session_threshold=0,
            escalation_window_days=7,
            escalation_window_threshold=2,
            escalation_window_action="block",
        )
        svc.check_text("trigger", "user1", session_id="s1")
        r = svc.check_text("trigger", "user1", session_id="s2")
        assert r.action == "block"

    def test_no_escalation_with_both_disabled(self, db, svc):
        """With both session and window escalation disabled, action stays at base."""
        _create_rule(
            db, patterns=["trigger"],
            action="notify",
            escalation_session_threshold=0,
            escalation_window_threshold=0,
        )
        for _ in range(20):
            r = svc.check_text("trigger", "user1", session_id="s1")
        assert r.action == "notify"
        assert r.escalation_triggered is False

    def test_session_escalation_exact_threshold(self, db, svc):
        """Escalation should trigger exactly at the threshold count."""
        _create_rule(
            db, patterns=["trigger"],
            action="notify",
            escalation_session_threshold=3,
            escalation_session_action="block",
        )
        r1 = svc.check_text("trigger", "user1", session_id="s1")
        r2 = svc.check_text("trigger", "user1", session_id="s1")
        r3 = svc.check_text("trigger", "user1", session_id="s1")
        assert r1.action == "notify"
        assert r2.action == "notify"
        assert r3.action == "block"
        assert r3.escalation_triggered is True


# ═══════════════════════════════════════════════════════════════
# 2. NOTIFICATION DELIVERY EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestNotificationDeliveryEdgeCases:
    def _make_service(self, **overrides):
        from tldw_Server_API.app.core.Monitoring.notification_service import NotificationService
        env = {
            "MONITORING_NOTIFY_ENABLED": "true",
            "MONITORING_NOTIFY_MIN_SEVERITY": overrides.pop("min_severity", "info"),
            "MONITORING_NOTIFY_DIGEST_MODE": "immediate",
            "MONITORING_NOTIFY_WEBHOOK_URL": "",
            "MONITORING_NOTIFY_EMAIL_TO": "",
            "MONITORING_NOTIFY_SMTP_HOST": "",
        }
        with patch.dict("os.environ", env, clear=False):
            svc = NotificationService()
        svc.file_path = "/dev/null"
        return svc

    def test_severity_filtering_critical_only(self):
        """With min_severity=critical, info and warning should be skipped."""
        svc = self._make_service(min_severity="critical")
        r_info = svc.notify_generic({"type": "t", "severity": "info", "user_id": "u1"})
        assert r_info == "skipped"
        r_warn = svc.notify_generic({"type": "t", "severity": "warning", "user_id": "u1"})
        assert r_warn == "skipped"
        r_crit = svc.notify_generic({"type": "t", "severity": "critical", "user_id": "u1"})
        assert r_crit in ("logged", "failed")

    def test_severity_filtering_warning_threshold(self):
        """With min_severity=warning, info should be skipped."""
        svc = self._make_service(min_severity="warning")
        r_info = svc.notify_generic({"type": "t", "severity": "info", "user_id": "u1"})
        assert r_info == "skipped"
        r_warn = svc.notify_generic({"type": "t", "severity": "warning", "user_id": "u1"})
        assert r_warn in ("logged", "failed")

    def test_disabled_service_skips_everything(self):
        """When MONITORING_NOTIFY_ENABLED=false, all notifications should be skipped."""
        from tldw_Server_API.app.core.Monitoring.notification_service import NotificationService
        env = {
            "MONITORING_NOTIFY_ENABLED": "false",
            "MONITORING_NOTIFY_MIN_SEVERITY": "info",
            "MONITORING_NOTIFY_DIGEST_MODE": "immediate",
            "MONITORING_NOTIFY_WEBHOOK_URL": "",
            "MONITORING_NOTIFY_EMAIL_TO": "",
            "MONITORING_NOTIFY_SMTP_HOST": "",
        }
        with patch.dict("os.environ", env, clear=False):
            svc = NotificationService()
        svc.file_path = "/dev/null"

        r = svc.notify_generic({"type": "t", "severity": "critical", "user_id": "u1"})
        assert r == "skipped"

    def test_update_settings_changes_runtime_config(self):
        """update_settings should change runtime config without restart."""
        svc = self._make_service()
        assert svc.enabled is True

        result = svc.update_settings(enabled=False)
        assert result["enabled"] is False
        assert svc.enabled is False

        svc.update_settings(min_severity="critical")
        assert svc.min_severity == "critical"

    def test_get_settings_returns_all_fields(self):
        """get_settings should include all expected keys."""
        svc = self._make_service()
        settings = svc.get_settings()
        expected_keys = {
            "enabled", "min_severity", "file", "webhook_url", "email_to",
            "smtp_host", "smtp_port", "smtp_starttls", "smtp_user", "email_from",
        }
        assert expected_keys.issubset(settings.keys())

    def test_generic_notify_does_not_mutate_caller_payload(self):
        """notify_generic should keep caller-owned payloads unchanged."""
        svc = self._make_service()
        payload = {"type": "test", "severity": "info", "user_id": "u1"}
        svc.notify_generic(payload)
        assert "ts" not in payload


# ═══════════════════════════════════════════════════════════════
# 3. IMPORT/EXPORT EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestImportExportEdgeCases:
    def test_import_malformed_rule_skipped(self, db):
        """Malformed self_monitoring_rules entries should be skipped, not crash."""
        bundle_data = {
            "self_monitoring_rules": [
                {"name": "Valid Rule", "patterns": ["pat1"]},
                {"name": "Rule With String Pattern", "patterns": "bad_string"},
            ],
        }
        counts = import_governance_rules(db, "user1", bundle_data)
        assert counts["self_monitoring_rules"] >= 1

    def test_import_empty_self_monitoring_rules(self, db):
        """Empty self_monitoring_rules list should import zero."""
        bundle_data = {"self_monitoring_rules": []}
        counts = import_governance_rules(db, "user1", bundle_data)
        assert counts["self_monitoring_rules"] == 0

    def test_import_with_unknown_keys_ignored(self, db):
        """Extra keys in bundle should be ignored."""
        bundle_data = {
            "unknown_section": [{"foo": "bar"}],
            "self_monitoring_rules": [
                {"name": "Rule 1", "patterns": ["p1"]},
            ],
        }
        counts = import_governance_rules(db, "user1", bundle_data)
        assert counts["self_monitoring_rules"] == 1

    def test_export_bundle_has_format_version(self, db):
        """Export bundle should include format_version."""
        bundle = export_governance_rules(db, "user1")
        assert bundle.format_version == "1.0"

    def test_import_replaces_then_imports_fresh(self, db):
        """Replace mode should clear then import, resulting in exact count."""
        _create_rule(db, name="Existing 1", patterns=["p1"])
        _create_rule(db, name="Existing 2", patterns=["p2"])
        assert len(db.list_self_monitoring_rules("user1")) == 2

        bundle_data = {
            "self_monitoring_rules": [
                {"name": "New Rule", "patterns": ["new"]},
            ],
        }
        counts = import_governance_rules(db, "user1", bundle_data, merge_mode="replace")
        assert counts["self_monitoring_rules"] == 1
        rules = db.list_self_monitoring_rules("user1")
        assert len(rules) == 1
        assert rules[0].name == "New Rule"

    def test_export_preserves_patterns(self, db):
        """Exported rule should have patterns preserved."""
        _create_rule(db, name="Pat Rule", patterns=["alpha", "beta", "gamma"])
        bundle = export_governance_rules(db, "user1")
        rule_data = bundle.self_monitoring_rules[0]
        assert rule_data["patterns"] == ["alpha", "beta", "gamma"]

    def test_roundtrip_preserves_category(self, db, tmp_path):
        """Export -> import should preserve category field."""
        _create_rule(db, name="Cat Rule", patterns=["p"], category="violence")
        bundle = export_governance_rules(db, "user1")
        json_str = export_to_json(bundle)

        db2 = GuardianDB(str(tmp_path / "roundtrip.db"))
        parsed = json.loads(json_str)
        import_governance_rules(db2, "user2", parsed)
        rules = db2.list_self_monitoring_rules("user2")
        assert len(rules) == 1
        assert rules[0].category == "violence"

    def test_import_governance_policy_with_all_fields(self, db):
        """Import governance policy with all optional fields."""
        bundle_data = {
            "governance_policies": [
                {
                    "id": "old-id-1",
                    "name": "Full Policy",
                    "description": "A detailed description",
                    "policy_mode": "self",
                    "scope_chat_types": "character,regular",
                    "enabled": True,
                    "schedule_start": "09:00",
                    "schedule_end": "17:00",
                    "schedule_days": "mon,tue,wed",
                    "schedule_timezone": "US/Pacific",
                    "transparent": True,
                },
            ],
        }
        counts = import_governance_rules(db, "user1", bundle_data)
        assert counts["governance_policies"] == 1
        policies = db.list_governance_policies("user1")
        assert len(policies) == 1
        assert policies[0].description == "A detailed description"
        assert policies[0].transparent is True

    def test_export_empty_user(self, db):
        """Export for user with no data should return empty bundle."""
        bundle = export_governance_rules(db, "nobody")
        assert bundle.self_monitoring_rules == []
        assert bundle.governance_policies == []


# ═══════════════════════════════════════════════════════════════
# 4. ANALYTICS EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestAnalyticsEdgeCases:
    def _create_rule_and_alerts(self, db, user_id="user1", count=1, **kwargs):
        defaults = {
            "user_id": user_id,
            "name": f"Rule {kwargs.get('category', 'test')}",
            "patterns": ["trigger"],
            "notification_frequency": "every_message",
        }
        defaults.update({k: v for k, v in kwargs.items() if k in ("category",)})
        rule = db.create_self_monitoring_rule(**defaults)
        for _ in range(count):
            db.create_self_monitoring_alert(
                user_id=user_id,
                rule_id=rule.id,
                rule_name=rule.name,
                category=kwargs.get("category", "test"),
                severity=kwargs.get("severity", "info"),
                matched_pattern="trigger",
            )
        return rule

    def test_aggregate_single_category(self, db):
        """Single category with one alert."""
        self._create_rule_and_alerts(db, category="single", count=1)
        result = db.aggregate_alerts_by_category("user1")
        assert len(result) == 1
        assert result[0]["count"] == 1

    def test_aggregate_multiple_users_isolated(self, db):
        """Aggregation should only count alerts for the specified user."""
        self._create_rule_and_alerts(db, user_id="user1", category="cat1", count=3)
        self._create_rule_and_alerts(db, user_id="user2", category="cat1", count=5)

        result_u1 = db.aggregate_alerts_by_category("user1")
        result_u2 = db.aggregate_alerts_by_category("user2")

        u1_total = sum(r["count"] for r in result_u1)
        u2_total = sum(r["count"] for r in result_u2)
        assert u1_total == 3
        assert u2_total == 5

    def test_aggregate_by_severity_mixed(self, db):
        """Mixed severities should each appear with correct counts."""
        self._create_rule_and_alerts(db, category="c1", severity="info", count=2)
        self._create_rule_and_alerts(db, category="c2", severity="critical", count=1)
        result = db.aggregate_alerts_by_severity("user1")
        sevs = {r["severity"]: r["count"] for r in result}
        assert sevs.get("info", 0) == 2
        assert sevs.get("critical", 0) == 1

    def test_top_patterns_with_tied_counts(self, db):
        """Patterns with equal frequency should all appear."""
        rule = db.create_self_monitoring_rule(
            user_id="user1", name="TiedRule", patterns=["a", "b"],
        )
        for _ in range(3):
            db.create_self_monitoring_alert(
                user_id="user1", rule_id=rule.id, matched_pattern="pattern_a",
            )
        for _ in range(3):
            db.create_self_monitoring_alert(
                user_id="user1", rule_id=rule.id, matched_pattern="pattern_b",
            )
        result = db.get_top_matched_patterns("user1")
        counts = [r["count"] for r in result]
        assert counts[0] == 3
        assert counts[1] == 3

    def test_escalation_summary_multiple_rules(self, db):
        """Escalation summary with multiple rule states."""
        db.upsert_escalation_state(
            rule_id="r1", user_id="user1",
            session_trigger_count=2, window_trigger_count=5,
            current_escalated_action="block",
        )
        db.upsert_escalation_state(
            rule_id="r2", user_id="user1",
            session_trigger_count=1, window_trigger_count=2,
            current_escalated_action=None,
        )
        result = db.get_escalation_summary("user1")
        assert len(result) == 2
        rule_ids = {r["rule_id"] for r in result}
        assert "r1" in rule_ids
        assert "r2" in rule_ids

    def test_time_aggregation_daily(self, db):
        """Time aggregation with daily bucket should group all today's alerts."""
        self._create_rule_and_alerts(db, count=4)
        result = db.aggregate_alerts_by_time("user1", bucket="day")
        total = sum(r["count"] for r in result)
        assert total == 4

    def test_time_aggregation_monthly(self, db):
        """Time aggregation with monthly bucket should group alerts."""
        self._create_rule_and_alerts(db, count=3)
        result = db.aggregate_alerts_by_time("user1", bucket="month")
        assert len(result) >= 1
        total = sum(r["count"] for r in result)
        assert total == 3


# ═══════════════════════════════════════════════════════════════
# 5. CATEGORY TAXONOMY EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestCategoryTaxonomyEdgeCases:
    def test_all_categories_patterns_compile(self):
        """All regex patterns in all categories should compile without error."""
        for name in CATEGORY_TAXONOMY:
            patterns = get_category_patterns(name)
            for pat in patterns:
                assert isinstance(pat, re.Pattern), f"{name}: pattern not compiled"

    def test_category_patterns_case_insensitive(self):
        """Category patterns should be case-insensitive."""
        patterns = get_category_patterns("violence")
        text_upper = "KNIFE"
        text_lower = "knife"
        upper_matches = [p for p in patterns if p.search(text_upper)]
        lower_matches = [p for p in patterns if p.search(text_lower)]
        assert len(upper_matches) == len(lower_matches)

    def test_profanity_patterns_exist(self):
        """Profanity category should have patterns."""
        patterns = get_category_patterns("profanity")
        assert len(patterns) > 0

    def test_unknown_category_safe(self):
        """Unknown category should not raise, just return empty."""
        patterns = get_category_patterns("totally_fake_category_xyz")
        assert patterns == []


# ═══════════════════════════════════════════════════════════════
# 6. SEMANTIC MATCHER EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestSemanticMatcherEdgeCases:
    def test_multiple_references_best_match_selected(self):
        """check_similarity should select the best matching reference."""
        matcher = SemanticMatcher()
        with patch.object(matcher, "_embed_text") as mock_embed:
            def embed_fn(text):
                if text == "input":
                    return [0.9, 0.1]
                elif text == "close_ref":
                    return [0.95, 0.05]
                elif text == "far_ref":
                    return [0.0, 1.0]
                return [0.5, 0.5]

            mock_embed.side_effect = embed_fn
            matched, score, best_ref = matcher.check_similarity(
                "input", ["close_ref", "far_ref"], threshold=0.8
            )
            assert matched is True
            assert best_ref == "close_ref"

    def test_classify_with_custom_prompt_template(self):
        """classify_with_llm should accept custom prompt template."""
        matcher = SemanticMatcher()
        with patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
            return_value="drugs",
        ):
            matched, category, conf = matcher.classify_with_llm(
                "I want some drugs",
                ["violence", "drugs"],
                prompt_template="Custom: classify '{text}' into {categories}",
            )
            assert matched is True
            assert category == "drugs"

    def test_clear_cache_resets(self):
        """clear_cache should reset the reference cache."""
        matcher = SemanticMatcher()
        matcher._ref_cache[("key",)] = [[1.0]]
        matcher.clear_cache()
        assert len(matcher._ref_cache) == 0


# ═══════════════════════════════════════════════════════════════
# 7. GUARDIAN DB EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestGuardianDBEdgeCases:
    def test_create_alert_with_all_optional_fields(self, db):
        """Alert with all optional fields should store correctly."""
        rule = db.create_self_monitoring_rule("user1", "R1", patterns=["p"])
        alert = db.create_self_monitoring_alert(
            user_id="user1",
            rule_id=rule.id,
            rule_name="R1",
            category="test_cat",
            severity="critical",
            matched_pattern="pattern123",
            context_snippet="some context here",
            conversation_id="conv-abc",
            session_id="sess-xyz",
            phase="output",
            action_taken="blocked",
            display_mode="sidebar_note",
            notification_channels_used=["in_app", "webhook"],
            escalation_info={"escalated": True, "action": "block"},
        )
        assert alert.category == "test_cat"
        assert alert.severity == "critical"
        assert alert.phase == "output"
        assert alert.action_taken == "blocked"
        assert alert.display_mode == "sidebar_note"
        assert alert.notification_channels_used == ["in_app", "webhook"]
        assert alert.escalation_info["escalated"] is True

    def test_list_alerts_with_rule_filter(self, db):
        """list_self_monitoring_alerts with rule_id filter should work."""
        r1 = db.create_self_monitoring_rule("user1", "R1")
        r2 = db.create_self_monitoring_rule("user1", "R2")
        db.create_self_monitoring_alert("user1", r1.id, severity="info")
        db.create_self_monitoring_alert("user1", r2.id, severity="critical")

        by_rule = db.list_self_monitoring_alerts("user1", rule_id=r1.id)
        assert len(by_rule) == 1

    def test_has_recent_alert_with_session_id(self, db):
        """has_recent_alert should filter by session_id."""
        rule = db.create_self_monitoring_rule("user1", "R1")
        db.create_self_monitoring_alert(
            "user1", rule.id, session_id="sess_A",
        )
        assert db.has_recent_alert("user1", rule.id, session_id="sess_A") is True
        assert db.has_recent_alert("user1", rule.id, session_id="sess_B") is False

    def test_governance_policy_all_fields_roundtrip(self, db):
        """Governance policy with all fields should roundtrip correctly."""
        gp = db.create_governance_policy(
            "user1", "Full GP",
            description="desc",
            policy_mode="self",
            scope_chat_types="regular,character",
            enabled=False,
            schedule_start="09:00",
            schedule_end="17:00",
            schedule_days="mon,tue",
            schedule_timezone="US/Pacific",
            transparent=True,
        )
        fetched = db.get_governance_policy(gp.id)
        assert fetched.description == "desc"
        assert fetched.scope_chat_types == "regular,character"
        assert fetched.enabled is False
        assert fetched.transparent is True
        assert fetched.schedule_timezone == "US/Pacific"

    def test_rule_with_governance_policy_link(self, db):
        """Self-monitoring rule with governance_policy_id should store it."""
        gp = db.create_governance_policy("user1", "Policy", policy_mode="self")
        rule = _create_rule(db, governance_policy_id=gp.id)
        fetched = db.get_self_monitoring_rule(rule.id)
        assert fetched.governance_policy_id == gp.id

    def test_delete_all_for_empty_user(self, db):
        """delete_all_for_user on nonexistent user should return zero counts."""
        counts = db.delete_all_for_user("nobody")
        assert all(v == 0 for v in counts.values())


# ═══════════════════════════════════════════════════════════════
# 8. SELF-MONITORING SERVICE ADDITIONAL TESTS
# ═══════════════════════════════════════════════════════════════


class TestSelfMonitoringAdditional:
    def test_dedup_once_per_session(self, db, svc):
        """once_per_session dedup should suppress within same session."""
        _create_rule(
            db, patterns=["word"],
            notification_frequency="once_per_session",
        )
        r1 = svc.check_text("word", "user1", session_id="s1")
        assert r1.triggered is True
        r2 = svc.check_text("word", "user1", session_id="s1")
        assert r2.triggered is False
        # New session should fire again
        r3 = svc.check_text("word", "user1", session_id="s2")
        assert r3.triggered is True

    def test_multiple_rules_different_actions(self, db, svc):
        """Multiple matching rules with different actions — highest wins."""
        _create_rule(db, name="R1", patterns=["test_word"], action="notify")
        _create_rule(db, name="R2", patterns=["test_word"], action="block")
        result = svc.check_text("test_word here", "user1")
        assert result.action == "block"

    def test_escalation_info_in_alert_record(self, db, svc):
        """Escalated alerts should have escalation_info set."""
        _create_rule(
            db, patterns=["trigger"],
            escalation_session_threshold=2,
            escalation_session_action="block",
        )
        svc.check_text("trigger", "user1", session_id="s1")
        svc.check_text("trigger", "user1", session_id="s1")
        alerts = db.list_self_monitoring_alerts("user1")
        escalated_alerts = [a for a in alerts if a.escalation_info]
        assert len(escalated_alerts) >= 1

    def test_request_deactivation_immediate(self, db, svc):
        """Deactivation with no cooldown should be immediate."""
        rule = _create_rule(db, patterns=["word"], cooldown_minutes=0)
        result = svc.request_deactivation(rule.id, "user1")
        assert result["ok"] is True
        updated = db.get_self_monitoring_rule(rule.id)
        assert updated.enabled is False

    def test_request_deactivation_wrong_user(self, db, svc):
        """Deactivation request by wrong user should fail."""
        rule = _create_rule(db, patterns=["word"])
        result = svc.request_deactivation(rule.id, "wrong_user")
        assert result["ok"] is False

    def test_request_deactivation_nonexistent(self, svc):
        """Deactivation request for nonexistent rule should fail."""
        result = svc.request_deactivation("nonexistent", "user1")
        assert result["ok"] is False


# ═══════════════════════════════════════════════════════════════
# 9. INTEGRATION: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════


class _FakeModerationService:
    """Minimal moderation service for integration tests."""
    from tldw_Server_API.app.core.Moderation.moderation_service import (
        ModerationPolicy,
        PatternRule,
    )

    def __init__(self, base_policy):
        self._policy = base_policy

    def get_effective_policy(self, user_id=None):
        return self._policy

    def evaluate_action_with_match(self, text, policy, phase):
        if not policy.enabled or not text or not policy.block_patterns:
            return "pass", None, None, None, None
        for rule in policy.block_patterns:
            if not isinstance(rule, self.PatternRule):
                continue
            m = rule.regex.search(text)
            if m:
                action = rule.action or (
                    policy.input_action if phase == "input" else policy.output_action
                )
                redacted = None
                if action == "redact":
                    repl = rule.replacement or policy.redact_replacement or "[REDACTED]"
                    redacted = rule.regex.sub(repl, text)
                cats = rule.categories or set()
                cat = next(iter(cats), None)
                return action, redacted, rule.regex.pattern, cat, (m.start(), m.end())
        return "pass", None, None, None, None

    def check_text(self, text, policy, phase=None):
        if not policy.enabled or not text or not policy.block_patterns:
            return False, None
        for rule in policy.block_patterns:
            if not isinstance(rule, self.PatternRule):
                continue
            if rule.regex.search(text):
                return True, text[:50]
        return False, None

    def redact_text(self, text, policy):
        if not text or not policy.block_patterns:
            return text
        result = text
        for rule in policy.block_patterns:
            if not isinstance(rule, self.PatternRule):
                continue
            repl = rule.replacement or policy.redact_replacement or "[REDACTED]"
            result = rule.regex.sub(repl, result)
        return result

    def build_sanitized_snippet(self, text, policy, match_span, pattern=None):
        return text[:50] if text else None


class TestIntegrationPipeline:
    """Integration tests combining guardian + self-monitoring + moderation."""

    @pytest.mark.asyncio
    async def test_self_monitoring_redact_pipeline(self, db):
        """Self-monitoring redact should modify text in pipeline."""
        from tldw_Server_API.app.core.Chat.chat_service import moderate_input_messages
        from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy

        svc = SelfMonitoringService(db)
        db.create_self_monitoring_rule(
            user_id="user1",
            name="Redact Secret",
            patterns=["secret"],
            action="redact",
        )

        moderation_svc = _FakeModerationService(ModerationPolicy(enabled=False))
        msg = SimpleNamespace(role="user", content="the secret code is 42", type="text")
        request_data = SimpleNamespace(messages=[msg], conversation_id=None)
        request = SimpleNamespace(
            state=SimpleNamespace(user_id="user1", team_ids=None, org_ids=None)
        )

        await moderate_input_messages(
            request_data=request_data,
            request=request,
            moderation_service=moderation_svc,
            topic_monitoring_service=None,
            metrics=MagicMock(),
            audit_service=_StubAuditService(),
            audit_context=SimpleNamespace(),
            client_id="user1",
            self_monitoring_service=svc,
        )
        assert "secret" not in request_data.messages[0].content
        assert "[SELF-REDACTED]" in request_data.messages[0].content

    @pytest.mark.asyncio
    async def test_guardian_block_raises_400(self, db, engine):
        """Guardian block policy should raise HTTPException(400)."""
        from tldw_Server_API.app.core.Chat.chat_service import moderate_input_messages
        from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy

        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="forbidden",
            action="block",
        )

        moderation_svc = _FakeModerationService(ModerationPolicy(enabled=False))
        msg = SimpleNamespace(role="user", content="this is forbidden content", type="text")
        request_data = SimpleNamespace(messages=[msg], conversation_id=None)
        request = SimpleNamespace(
            state=SimpleNamespace(user_id="child1", team_ids=None, org_ids=None)
        )

        with pytest.raises(HTTPException) as exc_info:
            await moderate_input_messages(
                request_data=request_data,
                request=request,
                moderation_service=moderation_svc,
                topic_monitoring_service=None,
                metrics=MagicMock(),
                audit_service=_StubAuditService(),
                audit_context=SimpleNamespace(),
                client_id="child1",
                supervised_policy_engine=engine,
                dependent_user_id="child1",
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_self_monitoring_escalation_block(self, db):
        """Self-monitoring escalation from notify to block."""
        svc = SelfMonitoringService(db)
        db.create_self_monitoring_rule(
            user_id="user1",
            name="Escalation Rule",
            patterns=["danger"],
            action="notify",
            escalation_session_threshold=2,
            escalation_session_action="block",
            block_message="Escalated: too many triggers.",
        )

        r1 = svc.check_text("danger zone", "user1", session_id="sess1")
        assert r1.triggered is True
        assert r1.action == "notify"

        r2 = svc.check_text("danger again", "user1", session_id="sess1")
        assert r2.triggered is True
        assert r2.action == "block"
        assert r2.escalation_triggered is True

    @pytest.mark.asyncio
    async def test_multiple_user_messages_all_checked(self, db, engine):
        """All user messages in the request should be moderation-checked."""
        from tldw_Server_API.app.core.Chat.chat_service import moderate_input_messages
        from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy

        rel = _setup_active_relationship(db)
        db.create_policy(
            relationship_id=rel.id,
            pattern="secret",
            action="redact",
        )

        moderation_svc = _FakeModerationService(ModerationPolicy(enabled=False))
        msg1 = SimpleNamespace(role="user", content="the secret is out", type="text")
        msg2 = SimpleNamespace(
            role="assistant", content="secret should not be touched", type="text"
        )
        msg3 = SimpleNamespace(role="user", content="another secret here", type="text")
        request_data = SimpleNamespace(messages=[msg1, msg2, msg3], conversation_id=None)
        request = SimpleNamespace(
            state=SimpleNamespace(user_id="child1", team_ids=None, org_ids=None)
        )

        await moderate_input_messages(
            request_data=request_data,
            request=request,
            moderation_service=moderation_svc,
            topic_monitoring_service=None,
            metrics=MagicMock(),
            audit_service=_StubAuditService(),
            audit_context=SimpleNamespace(),
            client_id="child1",
            supervised_policy_engine=engine,
            dependent_user_id="child1",
        )

        # User messages should be redacted
        assert "secret" not in request_data.messages[0].content
        # Assistant messages are not checked on input phase
        assert "secret" in request_data.messages[1].content
        # Second user message should also be redacted
        assert "secret" not in request_data.messages[2].content

    @pytest.mark.asyncio
    async def test_no_services_passes_through(self):
        """When all optional services are None, text passes through."""
        from tldw_Server_API.app.core.Chat.chat_service import moderate_input_messages
        from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy

        moderation_svc = _FakeModerationService(ModerationPolicy(enabled=False))
        original = "normal text"
        msg = SimpleNamespace(role="user", content=original, type="text")
        request_data = SimpleNamespace(messages=[msg], conversation_id=None)
        request = SimpleNamespace(
            state=SimpleNamespace(user_id="user1", team_ids=None, org_ids=None)
        )

        await moderate_input_messages(
            request_data=request_data,
            request=request,
            moderation_service=moderation_svc,
            topic_monitoring_service=None,
            metrics=MagicMock(),
            audit_service=_StubAuditService(),
            audit_context=SimpleNamespace(),
            client_id="client1",
            supervised_policy_engine=None,
            self_monitoring_service=None,
            dependent_user_id=None,
        )

        assert request_data.messages[0].content == original
