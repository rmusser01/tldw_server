"""
Tests for governance rule import/export functionality.
"""
from __future__ import annotations

import json

import pytest

import tldw_Server_API.app.core.Moderation.governance_io as governance_io_module
from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
from tldw_Server_API.app.core.Moderation.governance_io import (
    GovernanceExportBundle,
    export_governance_rules,
    export_to_json,
    import_governance_rules,
)


@pytest.fixture()
def db(tmp_path):
    return GuardianDB(str(tmp_path / "test_io.db"))


def _create_rule(db, **kwargs):
    defaults = {
        "user_id": "user1",
        "name": "Test Rule",
        "patterns": ["test_word"],
        "notification_frequency": "every_message",
    }
    defaults.update(kwargs)
    return db.create_self_monitoring_rule(**defaults)


class TestExport:
    def test_export_produces_valid_json(self, db):
        """Export should produce valid JSON with correct structure."""
        _create_rule(db, name="Rule 1", patterns=["pattern1"])
        _create_rule(db, name="Rule 2", patterns=["pattern2"])
        db.create_governance_policy(owner_user_id="user1", name="Policy 1")

        bundle = export_governance_rules(db, "user1")
        json_str = export_to_json(bundle)
        parsed = json.loads(json_str)

        assert parsed["format_version"] == "1.0"
        assert "exported_at" in parsed
        assert len(parsed["self_monitoring_rules"]) == 2
        assert len(parsed["governance_policies"]) == 1

    def test_export_empty_returns_empty_lists(self, db):
        """Export with no data should return empty lists."""
        bundle = export_governance_rules(db, "user1")
        assert bundle.self_monitoring_rules == []
        assert bundle.governance_policies == []

    def test_export_only_self_monitoring(self, db):
        """Can export only self-monitoring rules."""
        _create_rule(db, name="Rule 1")
        db.create_governance_policy(owner_user_id="user1", name="Policy 1")

        bundle = export_governance_rules(
            db, "user1",
            include_governance_policies=False,
            include_self_monitoring=True,
        )
        assert len(bundle.self_monitoring_rules) == 1
        assert len(bundle.governance_policies) == 0


class TestImport:
    def test_import_creates_rules(self, db):
        """Import should create rules from bundle data."""
        bundle_data = {
            "format_version": "1.0",
            "governance_policies": [
                {"id": "old-gp-1", "name": "Imported Policy"},
            ],
            "self_monitoring_rules": [
                {
                    "name": "Imported Rule",
                    "patterns": ["imported_pattern"],
                    "category": "test",
                    "action": "notify",
                },
            ],
        }
        counts = import_governance_rules(db, "user1", bundle_data)
        assert counts["governance_policies"] == 1
        assert counts["self_monitoring_rules"] == 1

        rules = db.list_self_monitoring_rules("user1")
        assert len(rules) == 1
        assert rules[0].name == "Imported Rule"

    def test_import_replace_mode_clears_existing(self, db):
        """Replace mode should clear existing rules before importing."""
        _create_rule(db, name="Existing Rule")
        assert len(db.list_self_monitoring_rules("user1")) == 1

        bundle_data = {
            "self_monitoring_rules": [
                {
                    "name": "New Rule",
                    "patterns": ["new_pattern"],
                },
            ],
        }
        counts = import_governance_rules(db, "user1", bundle_data, merge_mode="replace")
        assert counts["self_monitoring_rules"] == 1

        rules = db.list_self_monitoring_rules("user1")
        assert len(rules) == 1
        assert rules[0].name == "New Rule"

    def test_import_add_mode_appends(self, db):
        """Add mode should append to existing rules."""
        _create_rule(db, name="Existing Rule")

        bundle_data = {
            "self_monitoring_rules": [
                {"name": "New Rule", "patterns": ["new"]},
            ],
        }
        counts = import_governance_rules(db, "user1", bundle_data, merge_mode="add")
        assert counts["self_monitoring_rules"] == 1

        rules = db.list_self_monitoring_rules("user1")
        assert len(rules) == 2

    def test_roundtrip_export_import_preserves_data(self, db, tmp_path):
        """Export then import should preserve rule data."""
        db.create_governance_policy(owner_user_id="user1", name="GP1")
        _create_rule(db, name="Rule A", patterns=["aaa"], category="cat1")
        _create_rule(db, name="Rule B", patterns=["bbb"], category="cat2")

        bundle = export_governance_rules(db, "user1")
        json_str = export_to_json(bundle)

        # Import into a fresh DB
        db2 = GuardianDB(str(tmp_path / "test_import.db"))
        parsed = json.loads(json_str)
        counts = import_governance_rules(db2, "user2", parsed)

        assert counts["governance_policies"] == 1
        assert counts["self_monitoring_rules"] == 2

        rules = db2.list_self_monitoring_rules("user2")
        names = {r.name for r in rules}
        assert "Rule A" in names
        assert "Rule B" in names

    def test_import_remaps_governance_policy_ids(self, db, tmp_path):
        """Import should remap governance_policy_id to newly created IDs."""
        gp = db.create_governance_policy(owner_user_id="user1", name="GP1")
        _create_rule(db, name="Rule With GP", governance_policy_id=gp.id)

        bundle = export_governance_rules(db, "user1")
        json_str = export_to_json(bundle)

        db2 = GuardianDB(str(tmp_path / "test_remap.db"))
        parsed = json.loads(json_str)
        counts = import_governance_rules(db2, "user2", parsed)

        rules = db2.list_self_monitoring_rules("user2")
        assert len(rules) == 1
        # governance_policy_id should be remapped to new ID (not the old one)
        if rules[0].governance_policy_id:
            assert rules[0].governance_policy_id != gp.id

    def test_import_empty_bundle(self, db):
        """Importing empty bundle should succeed with zero counts."""
        counts = import_governance_rules(db, "user1", {})
        assert counts["governance_policies"] == 0
        assert counts["self_monitoring_rules"] == 0

    def test_import_governance_policy_warning_sanitizes_backend_details(self, monkeypatch, db):
        def fail_create_policy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise ValueError("policy import failed at /private/governance-import.json")

        monkeypatch.setattr(db, "create_governance_policy", fail_create_policy)
        bundle_data = {
            "governance_policies": [
                {"id": "old-gp-1", "name": "Imported Policy"},
            ],
        }

        messages: list[str] = []
        sink_id = governance_io_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            counts = import_governance_rules(db, "user1", bundle_data)
        finally:
            governance_io_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert counts["governance_policies"] == 0
        assert "Failed to import governance policy" in joined
        assert "governance-import.json" not in joined

    def test_import_self_monitoring_warning_sanitizes_backend_details(self, monkeypatch, db):
        def fail_create_rule(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise ValueError("rule import failed at /private/selfmon-import.json")

        monkeypatch.setattr(db, "create_self_monitoring_rule", fail_create_rule)
        bundle_data = {
            "self_monitoring_rules": [
                {"name": "Imported Rule", "patterns": ["imported_pattern"]},
            ],
        }

        messages: list[str] = []
        sink_id = governance_io_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            counts = import_governance_rules(db, "user1", bundle_data)
        finally:
            governance_io_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert counts["self_monitoring_rules"] == 0
        assert "Failed to import self-monitoring rule" in joined
        assert "selfmon-import.json" not in joined
