"""Tests for the inheritable config template system (templates.py).

Covers:
- Template chain resolution (system -> persona -> session)
- Inheritance via base_template_id
- Field-type-aware merge rules (scalars override, dicts merge, lists append)
- seed_system_templates from PERMISSION_POLICY_TEMPLATES
- Circular inheritance prevention
- resolve_for_session fallback when no templates exist
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.core.Agent_Client_Protocol.templates import (
    ACPConfigTemplate,
    resolve_for_session,
    resolve_template_chain,
    seed_system_templates,
    _resolve_inheritance,
    _row_to_template,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    """Return an ACPSessionsDB backed by a temporary file."""
    db_path = str(tmp_path / "test_templates.db")
    _db = ACPSessionsDB(db_path=db_path)
    # Force schema initialization
    _db._get_conn()
    yield _db
    _db.close()


# ---------------------------------------------------------------------------
# 1. System template loads correctly
# ---------------------------------------------------------------------------


class TestResolveSystemTemplate:
    def test_resolve_system_template(self, db: ACPSessionsDB) -> None:
        config = {"tool_tier_overrides": {"*": "auto"}, "approval_mode": "relaxed"}
        db.create_config_template(
            name="my-system",
            scope="system",
            config_json=json.dumps(config),
        )

        result = resolve_for_session(db, template_name="my-system")

        assert result["tool_tier_overrides"] == {"*": "auto"}
        assert result["approval_mode"] == "relaxed"


# ---------------------------------------------------------------------------
# 2. Persona template overrides system fields
# ---------------------------------------------------------------------------


class TestPersonaOverridesSystem:
    def test_persona_overrides_system(self, db: ACPSessionsDB) -> None:
        # System template
        system_config = {
            "tool_tier_overrides": {"Read(*)": "auto", "Write(*)": "batch"},
            "approval_mode": "strict",
        }
        db.create_config_template(
            name="base-system",
            scope="system",
            config_json=json.dumps(system_config),
        )

        # Persona template overrides approval_mode and one tool tier
        persona_config = {
            "tool_tier_overrides": {"Write(*)": "auto"},
            "approval_mode": "relaxed",
        }
        db.create_config_template(
            name="persona-override",
            scope="persona",
            scope_id="persona-123",
            config_json=json.dumps(persona_config),
        )

        result = resolve_for_session(
            db,
            template_name="base-system",
            persona_id="persona-123",
        )

        # Persona overrides scalar
        assert result["approval_mode"] == "relaxed"
        # Dict merge: Write(*) overridden, Read(*) preserved
        assert result["tool_tier_overrides"]["Write(*)"] == "auto"
        assert result["tool_tier_overrides"]["Read(*)"] == "auto"


# ---------------------------------------------------------------------------
# 3. Session template has highest precedence
# ---------------------------------------------------------------------------


class TestSessionOverridesPersona:
    def test_session_overrides_persona(self, db: ACPSessionsDB) -> None:
        system_config = {"tool_tier_overrides": {"*": "individual"}}
        db.create_config_template(
            name="sys",
            scope="system",
            config_json=json.dumps(system_config),
        )

        persona_config = {"tool_tier_overrides": {"*": "batch"}}
        db.create_config_template(
            name="persona-cfg",
            scope="persona",
            scope_id="p1",
            config_json=json.dumps(persona_config),
        )

        session_config = {"tool_tier_overrides": {"*": "auto"}}
        db.create_config_template(
            name="session-cfg",
            scope="session",
            scope_id="s1",
            config_json=json.dumps(session_config),
        )

        result = resolve_for_session(
            db,
            template_name="sys",
            persona_id="p1",
            session_id="s1",
        )

        # Session wins
        assert result["tool_tier_overrides"]["*"] == "auto"


# ---------------------------------------------------------------------------
# 4. Inheritance chain merges bottom-up
# ---------------------------------------------------------------------------


class TestTemplateInheritanceChain:
    def test_template_inheritance_chain(self, db: ACPSessionsDB) -> None:
        # Create a base template
        base_id = db.create_config_template(
            name="grandparent",
            scope="system",
            config_json=json.dumps({
                "tool_tier_overrides": {"Read(*)": "auto", "Write(*)": "individual"},
                "approval_mode": "strict",
            }),
        )

        # Create a child that inherits from base
        child_id = db.create_config_template(
            name="parent",
            scope="system",
            base_template_id=base_id,
            config_json=json.dumps({
                "tool_tier_overrides": {"Write(*)": "batch"},
            }),
        )

        # Create a grandchild that inherits from child -- this is the one we query
        db.create_config_template(
            name="child-template",
            scope="system",
            base_template_id=child_id,
            config_json=json.dumps({
                "approval_mode": "relaxed",
            }),
        )

        result = resolve_for_session(db, template_name="child-template")

        # grandparent Read(*) preserved through chain
        assert result["tool_tier_overrides"]["Read(*)"] == "auto"
        # parent overrode Write(*)
        assert result["tool_tier_overrides"]["Write(*)"] == "batch"
        # grandchild overrode approval_mode
        assert result["approval_mode"] == "relaxed"


# ---------------------------------------------------------------------------
# 5. Merge uses field-type rules
# ---------------------------------------------------------------------------


class TestMergeUsesFieldTypeRules:
    def test_merge_uses_field_type_rules(self) -> None:
        """Scalars override, dicts merge, known list keys append with dedup."""
        base = ACPConfigTemplate(
            id="base",
            name="base",
            config={
                "approval_mode": "strict",
                "tool_tier_overrides": {"Read(*)": "auto"},
                "denied_tools": ["rm", "dd"],
            },
        )
        overlay = ACPConfigTemplate(
            id="overlay",
            name="overlay",
            config={
                "approval_mode": "relaxed",
                "tool_tier_overrides": {"Write(*)": "batch"},
                "denied_tools": ["dd", "mkfs"],
            },
        )

        result = resolve_template_chain([base, overlay])

        # Scalar: overlay wins
        assert result["approval_mode"] == "relaxed"
        # Dict: merged
        assert result["tool_tier_overrides"] == {"Read(*)": "auto", "Write(*)": "batch"}
        # List (denied_tools is a union-list key): appended + deduped
        assert result["denied_tools"] == ["rm", "dd", "mkfs"]


# ---------------------------------------------------------------------------
# 6. Seed system templates from PERMISSION_POLICY_TEMPLATES
# ---------------------------------------------------------------------------


class TestSeedSystemTemplates:
    def test_seed_system_templates(self, db: ACPSessionsDB) -> None:
        from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
            PERMISSION_POLICY_TEMPLATES,
        )

        count = seed_system_templates(db)

        assert count == len(PERMISSION_POLICY_TEMPLATES)
        # Verify each template was created
        for name in PERMISSION_POLICY_TEMPLATES:
            templates = db.list_config_templates(scope="system", name=name)
            assert len(templates) == 1
            tpl = templates[0]
            assert tpl["scope"] == "system"
            parsed = json.loads(tpl["config_json"])
            assert "tool_tier_overrides" in parsed


# ---------------------------------------------------------------------------
# 7. Seed is idempotent
# ---------------------------------------------------------------------------


class TestSeedIdempotent:
    def test_seed_idempotent(self, db: ACPSessionsDB) -> None:
        count1 = seed_system_templates(db)
        count2 = seed_system_templates(db)

        assert count1 > 0
        assert count2 == 0  # No new templates created on second run

        # Total count should still equal initial seed
        from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
            PERMISSION_POLICY_TEMPLATES,
        )
        all_templates = db.list_config_templates(scope="system")
        assert len(all_templates) == len(PERMISSION_POLICY_TEMPLATES)


# ---------------------------------------------------------------------------
# 8. resolve_for_session returns empty dict when no templates exist
# ---------------------------------------------------------------------------


class TestResolveForSessionFallback:
    def test_resolve_for_session_fallback(self, db: ACPSessionsDB) -> None:
        result = resolve_for_session(
            db,
            session_id="nonexistent-session",
            persona_id="nonexistent-persona",
            template_name="nonexistent-template",
        )
        assert result == {}


# ---------------------------------------------------------------------------
# 9. Circular inheritance is prevented
# ---------------------------------------------------------------------------


class TestCircularInheritancePrevented:
    def test_circular_inheritance_prevented(self, db: ACPSessionsDB) -> None:
        # Create two templates that point to each other
        id_a = db.create_config_template(
            name="template-a",
            scope="system",
            config_json=json.dumps({"approval_mode": "a"}),
            template_id="tpl-a",
        )
        id_b = db.create_config_template(
            name="template-b",
            scope="system",
            base_template_id=id_a,
            config_json=json.dumps({"approval_mode": "b"}),
            template_id="tpl-b",
        )

        # Now update A to point to B (creating a cycle)
        db.update_config_template(id_a, base_template_id=id_b)

        # Resolution should not infinite loop -- it should terminate
        row_b = db.get_config_template(id_b)
        assert row_b is not None
        tpl_b = _row_to_template(row_b)
        chain = _resolve_inheritance(db, tpl_b)

        # The chain should contain each template at most once
        ids_in_chain = [t.id for t in chain]
        assert len(ids_in_chain) == len(set(ids_in_chain))
        # Both templates should be present (the cycle was broken)
        assert id_a in ids_in_chain
        assert id_b in ids_in_chain


# ---------------------------------------------------------------------------
# 10. DB CRUD basics
# ---------------------------------------------------------------------------


class TestConfigTemplateCRUD:
    def test_create_and_get(self, db: ACPSessionsDB) -> None:
        tid = db.create_config_template(
            name="test-tpl",
            scope="persona",
            scope_id="p-42",
            config_json='{"foo": "bar"}',
            description="A test template",
        )
        row = db.get_config_template(tid)
        assert row is not None
        assert row["name"] == "test-tpl"
        assert row["scope"] == "persona"
        assert row["scope_id"] == "p-42"
        assert json.loads(row["config_json"]) == {"foo": "bar"}

    def test_list_with_filters(self, db: ACPSessionsDB) -> None:
        db.create_config_template(name="a", scope="system")
        db.create_config_template(name="b", scope="persona", scope_id="p1")
        db.create_config_template(name="c", scope="persona", scope_id="p2")

        system = db.list_config_templates(scope="system")
        assert len(system) == 1
        assert system[0]["name"] == "a"

        persona_p1 = db.list_config_templates(scope="persona", scope_id="p1")
        assert len(persona_p1) == 1
        assert persona_p1[0]["name"] == "b"

        by_name = db.list_config_templates(name="c")
        assert len(by_name) == 1

    def test_update(self, db: ACPSessionsDB) -> None:
        tid = db.create_config_template(name="old-name", scope="system")
        updated = db.update_config_template(tid, name="new-name", config_json='{"x": 1}')
        assert updated is True
        row = db.get_config_template(tid)
        assert row is not None
        assert row["name"] == "new-name"
        assert json.loads(row["config_json"]) == {"x": 1}

    def test_delete(self, db: ACPSessionsDB) -> None:
        tid = db.create_config_template(name="to-delete", scope="system")
        assert db.delete_config_template(tid) is True
        assert db.get_config_template(tid) is None
        # Deleting again returns False
        assert db.delete_config_template(tid) is False

    def test_update_nonexistent_returns_false(self, db: ACPSessionsDB) -> None:
        assert db.update_config_template("no-such-id", name="x") is False

    def test_get_nonexistent_returns_none(self, db: ACPSessionsDB) -> None:
        assert db.get_config_template("no-such-id") is None
