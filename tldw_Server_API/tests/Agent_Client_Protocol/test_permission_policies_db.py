"""Tests for permission policy DB persistence.

Verifies that policies stored in the ACP Sessions DB survive restarts,
and that the resolve_permission_tier function uses fnmatch-based matching.
"""
import json
import os
import tempfile

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "acp_sessions.db")
        instance = ACPSessionsDB(db_path=path)
        yield instance
        instance.close()


class TestPermissionPolicyPersistence:
    """Policies survive DB re-creation (simulating server restart)."""

    def test_permission_policy_persists_across_restart(self, tmp_path):
        """Policies stored in DB survive when a new DB instance is created."""
        db_path = str(tmp_path / "test_acp.db")

        # Create first DB instance and add a policy
        db1 = ACPSessionsDB(db_path=db_path)
        policy_id = db1.create_permission_policy(
            name="bash-individual",
            rules_json=json.dumps([{"tool_pattern": "Bash(*)", "tier": "individual"}]),
            priority=10,
            description="Bash tools need individual approval",
        )
        assert policy_id > 0

        # Verify it exists
        policy = db1.get_permission_policy(policy_id)
        assert policy is not None
        assert policy["name"] == "bash-individual"
        db1.close()

        # Create a NEW DB instance pointing to the same file
        db2 = ACPSessionsDB(db_path=db_path)
        policy = db2.get_permission_policy(policy_id)
        assert policy is not None, "Policy should survive across DB instances"
        assert policy["name"] == "bash-individual"
        assert policy["priority"] == 10
        rules = json.loads(policy["rules_json"])
        assert rules == [{"tool_pattern": "Bash(*)", "tier": "individual"}]
        db2.close()


class TestResolvePermissionTierFromDB:
    """resolve_permission_tier queries DB-backed policies."""

    def test_resolve_basic_match(self, db):
        db.create_permission_policy(
            name="bash-individual",
            rules_json=json.dumps([{"tool_pattern": "Bash(*)", "tier": "individual"}]),
            priority=10,
        )
        result = db.resolve_permission_tier("Bash(git:status)")
        assert result == "individual"

    def test_resolve_no_match(self, db):
        db.create_permission_policy(
            name="bash-individual",
            rules_json=json.dumps([{"tool_pattern": "Bash(*)", "tier": "individual"}]),
            priority=10,
        )
        result = db.resolve_permission_tier("Read(file.txt)")
        assert result is None

    def test_resolve_case_insensitive(self, db):
        db.create_permission_policy(
            name="bash-auto",
            rules_json=json.dumps([{"tool_pattern": "bash(*)", "tier": "auto"}]),
            priority=5,
        )
        result = db.resolve_permission_tier("Bash(ls)")
        assert result == "auto"

    def test_resolve_priority_ordering(self, db):
        """Higher-priority policy wins when multiple rules match."""
        db.create_permission_policy(
            name="general-auto",
            rules_json=json.dumps([{"tool_pattern": "*", "tier": "auto"}]),
            priority=1,
        )
        db.create_permission_policy(
            name="bash-individual",
            rules_json=json.dumps([{"tool_pattern": "Bash(*)", "tier": "individual"}]),
            priority=10,
        )
        # Bash matches both, but priority=10 wins
        assert db.resolve_permission_tier("Bash(rm)") == "individual"
        # Read matches only the general policy
        assert db.resolve_permission_tier("Read(file.txt)") == "auto"

    def test_resolve_wildcard_patterns(self, db):
        db.create_permission_policy(
            name="read-auto",
            rules_json=json.dumps([
                {"tool_pattern": "Read(*)", "tier": "auto"},
                {"tool_pattern": "Write(*)", "tier": "batch"},
            ]),
            priority=5,
        )
        assert db.resolve_permission_tier("Read(/etc/passwd)") == "auto"
        assert db.resolve_permission_tier("Write(/tmp/out)") == "batch"


class TestCRUDOperations:
    """Create, list, update, delete permission policies."""

    def test_create_returns_id(self, db):
        policy_id = db.create_permission_policy(
            name="test-policy",
            rules_json=json.dumps([{"tool_pattern": "*", "tier": "auto"}]),
        )
        assert isinstance(policy_id, int)
        assert policy_id > 0

    def test_create_with_all_fields(self, db):
        policy_id = db.create_permission_policy(
            name="full-policy",
            rules_json=json.dumps([{"tool_pattern": "X(*)", "tier": "batch"}]),
            priority=42,
            description="A full policy",
            org_id="org-123",
            team_id="team-456",
        )
        policy = db.get_permission_policy(policy_id)
        assert policy["name"] == "full-policy"
        assert policy["description"] == "A full policy"
        assert policy["priority"] == 42
        assert policy["org_id"] == "org-123"
        assert policy["team_id"] == "team-456"
        assert policy["created_at"] is not None
        assert policy["updated_at"] is not None

    def test_list_permission_policies(self, db):
        db.create_permission_policy(
            name="policy-a",
            rules_json=json.dumps([{"tool_pattern": "*", "tier": "auto"}]),
            priority=1,
        )
        db.create_permission_policy(
            name="policy-b",
            rules_json=json.dumps([{"tool_pattern": "Bash(*)", "tier": "individual"}]),
            priority=10,
        )
        policies = db.list_permission_policies()
        assert len(policies) == 2
        # Should be ordered by priority desc, then name
        assert policies[0]["name"] == "policy-b"
        assert policies[1]["name"] == "policy-a"

    def test_update_permission_policy(self, db):
        policy_id = db.create_permission_policy(
            name="old-name",
            rules_json=json.dumps([{"tool_pattern": "*", "tier": "auto"}]),
            priority=1,
        )
        updated = db.update_permission_policy(
            policy_id,
            name="new-name",
            priority=99,
            description="Updated",
        )
        assert updated is True
        policy = db.get_permission_policy(policy_id)
        assert policy["name"] == "new-name"
        assert policy["priority"] == 99
        assert policy["description"] == "Updated"

    def test_update_rules_json(self, db):
        policy_id = db.create_permission_policy(
            name="p1",
            rules_json=json.dumps([{"tool_pattern": "*", "tier": "auto"}]),
        )
        new_rules = json.dumps([{"tool_pattern": "Bash(*)", "tier": "individual"}])
        db.update_permission_policy(policy_id, rules_json=new_rules)
        policy = db.get_permission_policy(policy_id)
        assert json.loads(policy["rules_json"]) == [{"tool_pattern": "Bash(*)", "tier": "individual"}]

    def test_update_nonexistent_returns_false(self, db):
        assert db.update_permission_policy(9999, name="nope") is False

    def test_delete_permission_policy(self, db):
        policy_id = db.create_permission_policy(
            name="doomed",
            rules_json=json.dumps([]),
        )
        assert db.delete_permission_policy(policy_id) is True
        assert db.get_permission_policy(policy_id) is None

    def test_delete_nonexistent_returns_false(self, db):
        assert db.delete_permission_policy(9999) is False

    def test_get_nonexistent_returns_none(self, db):
        assert db.get_permission_policy(9999) is None

    def test_auto_increment_ids(self, db):
        id1 = db.create_permission_policy(name="a", rules_json="[]")
        id2 = db.create_permission_policy(name="b", rules_json="[]")
        assert id2 > id1
