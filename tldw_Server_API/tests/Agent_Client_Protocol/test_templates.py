"""Tests for the inheritable ACP config template system.

Covers: system template resolution, persona overrides system, session overrides
persona, inheritance chain, merge rules, seed idempotent, circular inheritance
prevented, and empty fallback.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.templates import (
    ACPConfigTemplate,
    _resolve_inheritance,
    _row_to_template,
    resolve_for_session,
    resolve_template_chain,
    seed_system_templates,
)
from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Any) -> ACPSessionsDB:
    """Create a fresh in-memory-style DB in a temp dir."""
    db_path = os.path.join(str(tmp_path), "test_acp.db")
    return ACPSessionsDB(db_path=db_path)


# ---------------------------------------------------------------------------
# 1. System template resolution
# ---------------------------------------------------------------------------


class TestSystemTemplateResolution:
    def test_system_template_found(self, db: ACPSessionsDB) -> None:
        db.create_config_template(
            name="developer",
            scope="system",
            config_json=json.dumps({"tool_tier_overrides": {"Bash(git:*)": "auto"}}),
        )
        result = resolve_for_session(db, session_id=None, persona_id=None, template_name="developer")
        assert result is not None
        assert result["tool_tier_overrides"]["Bash(git:*)"] == "auto"

    def test_system_template_not_found_returns_none(self, db: ACPSessionsDB) -> None:
        result = resolve_for_session(db, session_id=None, persona_id=None, template_name="nonexistent")
        assert result is None

    def test_no_template_name_returns_none(self, db: ACPSessionsDB) -> None:
        db.create_config_template(
            name="developer",
            scope="system",
            config_json=json.dumps({"x": 1}),
        )
        result = resolve_for_session(db, session_id=None, persona_id=None, template_name=None)
        assert result is None


# ---------------------------------------------------------------------------
# 2. Persona overrides system
# ---------------------------------------------------------------------------


class TestPersonaOverridesSystem:
    def test_persona_overrides_system_values(self, db: ACPSessionsDB) -> None:
        db.create_config_template(
            name="developer",
            scope="system",
            config_json=json.dumps({
                "tool_tier_overrides": {"Read(*)": "auto", "Write(*)": "batch"},
                "approval_mode": "require",
            }),
        )
        db.create_config_template(
            name="developer",
            scope="persona",
            scope_id="persona-abc",
            config_json=json.dumps({
                "tool_tier_overrides": {"Write(*)": "individual"},
            }),
        )
        result = resolve_for_session(
            db,
            session_id=None,
            persona_id="persona-abc",
            template_name="developer",
        )
        assert result is not None
        assert result["tool_tier_overrides"]["Read(*)"] == "auto"
        assert result["tool_tier_overrides"]["Write(*)"] == "individual"
        assert result["approval_mode"] == "require"


# ---------------------------------------------------------------------------
# 3. Session overrides persona
# ---------------------------------------------------------------------------


class TestSessionOverridesPersona:
    def test_session_overrides_persona(self, db: ACPSessionsDB) -> None:
        db.create_config_template(
            name="dev",
            scope="system",
            config_json=json.dumps({"mode": "system", "tool_tier_overrides": {"*": "auto"}}),
        )
        db.create_config_template(
            name="dev",
            scope="persona",
            scope_id="p1",
            config_json=json.dumps({"mode": "persona"}),
        )
        db.create_config_template(
            name="dev",
            scope="session",
            scope_id="s1",
            config_json=json.dumps({"mode": "session", "extra": True}),
        )
        result = resolve_for_session(db, session_id="s1", persona_id="p1", template_name="dev")
        assert result is not None
        assert result["mode"] == "session"
        assert result["extra"] is True
        assert result["tool_tier_overrides"]["*"] == "auto"


# ---------------------------------------------------------------------------
# 4. Inheritance chain (base_template_id)
# ---------------------------------------------------------------------------


class TestInheritanceChain:
    def test_single_parent_inheritance(self, db: ACPSessionsDB) -> None:
        parent_id = db.create_config_template(
            name="base-strict",
            scope="system",
            config_json=json.dumps({"approval_mode": "require", "max_retries": 3}),
        )
        db.create_config_template(
            name="child",
            scope="system",
            base_template_id=parent_id,
            config_json=json.dumps({"max_retries": 5}),
        )
        result = resolve_for_session(db, session_id=None, persona_id=None, template_name="child")
        assert result is not None
        assert result["approval_mode"] == "require"
        assert result["max_retries"] == 5

    def test_multi_level_inheritance(self, db: ACPSessionsDB) -> None:
        grandparent_id = db.create_config_template(
            name="gp",
            scope="system",
            config_json=json.dumps({"a": 1, "b": 2}),
        )
        parent_id = db.create_config_template(
            name="parent",
            scope="system",
            base_template_id=grandparent_id,
            config_json=json.dumps({"b": 20, "c": 3}),
        )
        db.create_config_template(
            name="child",
            scope="system",
            base_template_id=parent_id,
            config_json=json.dumps({"c": 30}),
        )
        result = resolve_for_session(db, session_id=None, persona_id=None, template_name="child")
        assert result is not None
        assert result["a"] == 1
        assert result["b"] == 20
        assert result["c"] == 30

    def test_missing_parent_gracefully_skipped(self, db: ACPSessionsDB) -> None:
        db.create_config_template(
            name="orphan",
            scope="system",
            base_template_id=99999,
            config_json=json.dumps({"x": 1}),
        )
        result = resolve_for_session(db, session_id=None, persona_id=None, template_name="orphan")
        assert result is not None
        assert result["x"] == 1


# ---------------------------------------------------------------------------
# 5. Merge rules (via merge_config)
# ---------------------------------------------------------------------------


class TestMergeRules:
    def test_union_list_append_dedup(self, db: ACPSessionsDB) -> None:
        db.create_config_template(
            name="t",
            scope="system",
            config_json=json.dumps({"allowed_tools": ["a", "b"]}),
        )
        db.create_config_template(
            name="t",
            scope="persona",
            scope_id="p1",
            config_json=json.dumps({"allowed_tools": ["b", "c"]}),
        )
        result = resolve_for_session(db, session_id=None, persona_id="p1", template_name="t")
        assert result is not None
        assert result["allowed_tools"] == ["a", "b", "c"]

    def test_nested_dict_merge(self, db: ACPSessionsDB) -> None:
        db.create_config_template(
            name="t",
            scope="system",
            config_json=json.dumps({"tool_tier_overrides": {"Read(*)": "auto", "*": "individual"}}),
        )
        db.create_config_template(
            name="t",
            scope="persona",
            scope_id="p1",
            config_json=json.dumps({"tool_tier_overrides": {"Write(*)": "batch"}}),
        )
        result = resolve_for_session(db, session_id=None, persona_id="p1", template_name="t")
        assert result is not None
        tto = result["tool_tier_overrides"]
        assert tto["Read(*)"] == "auto"
        assert tto["*"] == "individual"
        assert tto["Write(*)"] == "batch"


# ---------------------------------------------------------------------------
# 6. Seed system templates (idempotent)
# ---------------------------------------------------------------------------


class TestSeedSystemTemplates:
    def test_seed_inserts_all_templates(self, db: ACPSessionsDB) -> None:
        from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
            PERMISSION_POLICY_TEMPLATES,
        )
        count = seed_system_templates(db)
        assert count == len(PERMISSION_POLICY_TEMPLATES)
        for name in PERMISSION_POLICY_TEMPLATES:
            rows = db.list_config_templates(scope="system", name=name)
            assert len(rows) == 1
            assert rows[0]["scope"] == "system"

    def test_seed_is_idempotent(self, db: ACPSessionsDB) -> None:
        count1 = seed_system_templates(db)
        count2 = seed_system_templates(db)
        assert count1 > 0
        assert count2 == 0

    def test_seeded_templates_contain_correct_config(self, db: ACPSessionsDB) -> None:
        from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
            PERMISSION_POLICY_TEMPLATES,
        )
        seed_system_templates(db)
        rows = db.list_config_templates(scope="system", name="lockdown")
        assert len(rows) == 1
        config = json.loads(rows[0]["config_json"])
        assert config == PERMISSION_POLICY_TEMPLATES["lockdown"]


# ---------------------------------------------------------------------------
# 7. Circular inheritance prevented
# ---------------------------------------------------------------------------


class TestCircularInheritancePrevented:
    def test_direct_self_reference(self, db: ACPSessionsDB) -> None:
        tid = db.create_config_template(
            name="self-ref",
            scope="system",
            config_json=json.dumps({"x": 1}),
        )
        db.update_config_template(tid, base_template_id=tid)
        row = db.get_config_template(tid)
        tpl = _row_to_template(row)
        with pytest.raises(ValueError, match="Circular"):
            _resolve_inheritance(db, tpl)

    def test_indirect_cycle(self, db: ACPSessionsDB) -> None:
        id_a = db.create_config_template(
            name="a", scope="system", config_json="{}",
        )
        id_b = db.create_config_template(
            name="b", scope="system", base_template_id=id_a, config_json="{}",
        )
        # Create cycle: a -> b -> a
        db.update_config_template(id_a, base_template_id=id_b)
        row = db.get_config_template(id_b)
        tpl = _row_to_template(row)
        with pytest.raises(ValueError, match="Circular"):
            _resolve_inheritance(db, tpl)


# ---------------------------------------------------------------------------
# 8. Empty fallback
# ---------------------------------------------------------------------------


class TestEmptyFallback:
    def test_empty_db_returns_none(self, db: ACPSessionsDB) -> None:
        result = resolve_for_session(db, session_id="s1", persona_id="p1", template_name="whatever")
        assert result is None

    def test_wrong_scope_returns_none(self, db: ACPSessionsDB) -> None:
        """A persona template with no matching system template still resolves."""
        db.create_config_template(
            name="t",
            scope="persona",
            scope_id="p1",
            config_json=json.dumps({"x": 1}),
        )
        result = resolve_for_session(db, session_id=None, persona_id="p1", template_name="t")
        assert result is not None
        assert result["x"] == 1


# ---------------------------------------------------------------------------
# 9. resolve_template_chain standalone
# ---------------------------------------------------------------------------


class TestResolveTemplateChain:
    def test_empty_list(self) -> None:
        assert resolve_template_chain([]) == {}

    def test_single_template(self) -> None:
        tpl = ACPConfigTemplate(config={"a": 1})
        assert resolve_template_chain([tpl]) == {"a": 1}

    def test_ordered_merge(self) -> None:
        t1 = ACPConfigTemplate(config={"a": 1, "b": 2})
        t2 = ACPConfigTemplate(config={"b": 20, "c": 3})
        result = resolve_template_chain([t1, t2])
        assert result == {"a": 1, "b": 20, "c": 3}


# ---------------------------------------------------------------------------
# 10. _row_to_template
# ---------------------------------------------------------------------------


class TestRowToTemplate:
    def test_valid_row(self) -> None:
        row = {
            "id": 42,
            "name": "test",
            "description": "desc",
            "scope": "persona",
            "scope_id": "p1",
            "base_template_id": 10,
            "schema_version": "2",
            "config_json": json.dumps({"x": 1}),
            "created_at": "2024-01-01",
            "updated_at": "2024-01-02",
        }
        tpl = _row_to_template(row)
        assert tpl.id == 42
        assert tpl.name == "test"
        assert tpl.scope == "persona"
        assert tpl.scope_id == "p1"
        assert tpl.base_template_id == 10
        assert tpl.config == {"x": 1}

    def test_invalid_json_defaults_to_empty(self) -> None:
        row = {"config_json": "not-json{{{"}
        tpl = _row_to_template(row)
        assert tpl.config == {}

    def test_dict_config_json(self) -> None:
        row = {"config_json": {"already": "parsed"}}
        tpl = _row_to_template(row)
        assert tpl.config == {"already": "parsed"}


# ---------------------------------------------------------------------------
# 11. DB CRUD operations
# ---------------------------------------------------------------------------


class TestDBCrud:
    def test_create_and_get(self, db: ACPSessionsDB) -> None:
        tid = db.create_config_template(
            name="test",
            description="A test template",
            scope="session",
            scope_id="s1",
            config_json=json.dumps({"mode": "test"}),
        )
        row = db.get_config_template(tid)
        assert row is not None
        assert row["name"] == "test"
        assert row["scope"] == "session"
        assert row["scope_id"] == "s1"

    def test_list_with_filters(self, db: ACPSessionsDB) -> None:
        db.create_config_template(name="a", scope="system", config_json="{}")
        db.create_config_template(name="b", scope="system", config_json="{}")
        db.create_config_template(name="a", scope="persona", scope_id="p1", config_json="{}")

        all_rows = db.list_config_templates()
        assert len(all_rows) == 3

        system_rows = db.list_config_templates(scope="system")
        assert len(system_rows) == 2

        named_rows = db.list_config_templates(name="a")
        assert len(named_rows) == 2

        specific = db.list_config_templates(scope="persona", scope_id="p1", name="a")
        assert len(specific) == 1

    def test_update(self, db: ACPSessionsDB) -> None:
        tid = db.create_config_template(
            name="original", scope="system", config_json="{}",
        )
        updated = db.update_config_template(tid, name="renamed", config_json='{"x":1}')
        assert updated is True
        row = db.get_config_template(tid)
        assert row["name"] == "renamed"
        assert json.loads(row["config_json"]) == {"x": 1}

    def test_delete(self, db: ACPSessionsDB) -> None:
        tid = db.create_config_template(name="ephemeral", scope="system", config_json="{}")
        assert db.delete_config_template(tid) is True
        assert db.get_config_template(tid) is None
        assert db.delete_config_template(tid) is False

    def test_get_nonexistent_returns_none(self, db: ACPSessionsDB) -> None:
        assert db.get_config_template(99999) is None

    def test_duplicate_name_within_same_scope_is_rejected(self, db: ACPSessionsDB) -> None:
        db.create_config_template(name="dupe", scope="system", config_json="{}")

        with pytest.raises(ValueError, match="already exists"):
            db.create_config_template(name="dupe", scope="system", config_json="{}")


# ---------------------------------------------------------------------------
# 12. build_snapshot uses DB templates with fallback
# ---------------------------------------------------------------------------


class _StubPolicyResolver:
    """Minimal stub that returns a canned resolved policy."""

    def __init__(self, policy_document: dict[str, Any] | None = None) -> None:
        self._doc = policy_document or {}

    async def resolve_for_context(
        self,
        *,
        user_id: int | str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "policy_document": dict(self._doc),
            "sources": [],
            "provenance": [],
        }


@pytest.mark.asyncio
async def test_build_snapshot_falls_back_to_flat_templates() -> None:
    """When no DB templates exist, build_snapshot uses flat PERMISSION_POLICY_TEMPLATES."""
    from tldw_Server_API.app.services.admin_acp_sessions_service import (
        SessionRecord,
        SessionTokenUsage,
    )
    from tldw_Server_API.app.services.acp_runtime_policy_service import (
        ACPRuntimePolicyService,
    )

    resolver = _StubPolicyResolver(policy_document={})
    service = ACPRuntimePolicyService(policy_resolver=resolver)
    session = SessionRecord(
        session_id="test-session",
        user_id=1,
        usage=SessionTokenUsage(),
    )

    snapshot = await service.build_snapshot(
        session_record=session,
        user_id=1,
        template_name="lockdown",
    )
    merged = snapshot.resolved_policy_document.get("tool_tier_overrides", {})
    assert merged == {"*": "individual"}
