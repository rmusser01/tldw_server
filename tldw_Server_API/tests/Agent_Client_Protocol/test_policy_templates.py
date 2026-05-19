"""Tests for permission policy templates (read-only, developer, admin, lockdown).

Validates that the 4 preset policy configurations:
- Contain the expected structure and keys.
- Map tools to the correct tiers.
- Act as base layers that can be overridden by user customisations.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
    PERMISSION_POLICY_TEMPLATES,
)
from tldw_Server_API.app.services.admin_acp_sessions_service import (
    SessionRecord,
    SessionTokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
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


def _make_session(**kwargs: Any) -> SessionRecord:
    defaults: dict[str, Any] = {
        "session_id": "test-session",
        "user_id": 1,
        "usage": SessionTokenUsage(),
    }
    defaults.update(kwargs)
    return SessionRecord(**defaults)


# ---------------------------------------------------------------------------
# 1. All four templates exist with expected keys
# ---------------------------------------------------------------------------


class TestTemplatesHaveExpectedKeys:
    EXPECTED_NAMES = ("read-only", "developer", "admin", "lockdown")

    def test_all_templates_present(self) -> None:
        for name in self.EXPECTED_NAMES:
            assert name in PERMISSION_POLICY_TEMPLATES, f"Missing template: {name}"

    def test_each_template_has_tool_tier_overrides(self) -> None:
        for name in self.EXPECTED_NAMES:
            tpl = PERMISSION_POLICY_TEMPLATES[name]
            assert "tool_tier_overrides" in tpl, (
                f"Template '{name}' missing tool_tier_overrides"
            )
            assert isinstance(tpl["tool_tier_overrides"], dict)

    def test_each_template_has_description(self) -> None:
        for name in self.EXPECTED_NAMES:
            tpl = PERMISSION_POLICY_TEMPLATES[name]
            assert "description" in tpl
            assert isinstance(tpl["description"], str)
            assert len(tpl["description"]) > 0


# ---------------------------------------------------------------------------
# 2. read-only template blocks writes
# ---------------------------------------------------------------------------


class TestReadOnlyTemplateBlocksWrites:
    def test_write_tool_is_individual(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["read-only"]["tool_tier_overrides"]
        # The wildcard catch-all should be "individual", which covers Write(*)
        assert overrides.get("*") == "individual"

    def test_read_tools_are_auto(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["read-only"]["tool_tier_overrides"]
        for tool in ("Read(*)", "Glob(*)", "Grep(*)"):
            assert overrides.get(tool) == "auto", f"{tool} should be auto"

    def test_safe_git_commands_are_auto(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["read-only"]["tool_tier_overrides"]
        for tool in ("Bash(git:log*)", "Bash(git:status*)", "Bash(git:diff*)"):
            assert overrides.get(tool) == "auto", f"{tool} should be auto"


# ---------------------------------------------------------------------------
# 3. developer template allows git
# ---------------------------------------------------------------------------


class TestDeveloperTemplateAllowsGit:
    def test_git_is_auto(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["developer"]["tool_tier_overrides"]
        assert overrides.get("Bash(git:*)") == "auto"

    def test_npm_is_auto(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["developer"]["tool_tier_overrides"]
        assert overrides.get("Bash(npm:*)") == "auto"

    def test_file_writes_are_batch(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["developer"]["tool_tier_overrides"]
        assert overrides.get("Write(*)") == "batch"
        assert overrides.get("Edit(*)") == "batch"

    def test_shell_exec_is_individual(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["developer"]["tool_tier_overrides"]
        assert overrides.get("Bash(*)") == "individual"


# ---------------------------------------------------------------------------
# 4. admin template blocks destructive ops
# ---------------------------------------------------------------------------


class TestAdminTemplateBlocksDestructive:
    def test_rm_rf_is_individual(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["admin"]["tool_tier_overrides"]
        assert overrides.get("Bash(rm:-rf*)") == "individual"

    def test_force_push_is_individual(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["admin"]["tool_tier_overrides"]
        assert overrides.get("Bash(git:push*--force*)") == "individual"

    def test_hard_reset_is_individual(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["admin"]["tool_tier_overrides"]
        assert overrides.get("Bash(git:reset*--hard*)") == "individual"

    def test_everything_else_is_auto(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["admin"]["tool_tier_overrides"]
        assert overrides.get("*") == "auto"


# ---------------------------------------------------------------------------
# 5. lockdown template — everything individual
# ---------------------------------------------------------------------------


class TestLockdownTemplateAllIndividual:
    def test_wildcard_is_individual(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["lockdown"]["tool_tier_overrides"]
        assert overrides.get("*") == "individual"

    def test_only_wildcard_key(self) -> None:
        overrides = PERMISSION_POLICY_TEMPLATES["lockdown"]["tool_tier_overrides"]
        assert list(overrides.keys()) == ["*"]


# ---------------------------------------------------------------------------
# 6. User overrides take precedence over template
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_overrides_take_precedence() -> None:
    from tldw_Server_API.app.services.acp_runtime_policy_service import (
        ACPRuntimePolicyService,
    )

    # The resolver returns a policy document that already has a user override
    # for "Read(*)" set to "individual" (overriding the read-only template's
    # "auto" default).
    user_override = {"Read(*)": "individual"}
    resolver = _StubPolicyResolver(
        policy_document={"tool_tier_overrides": user_override},
    )
    service = ACPRuntimePolicyService(policy_resolver=resolver)
    session = _make_session()

    snapshot = await service.build_snapshot(
        session_record=session,
        user_id=1,
        template_name="read-only",
    )

    merged = snapshot.resolved_policy_document.get("tool_tier_overrides", {})

    # The user override must win over the template default.
    assert merged["Read(*)"] == "individual"

    # Template defaults that were NOT overridden should still be present.
    assert merged.get("Glob(*)") == "auto"
    assert merged.get("*") == "individual"


@pytest.mark.asyncio
async def test_template_applied_when_no_existing_overrides() -> None:
    from tldw_Server_API.app.services.acp_runtime_policy_service import (
        ACPRuntimePolicyService,
    )

    resolver = _StubPolicyResolver(policy_document={})
    service = ACPRuntimePolicyService(policy_resolver=resolver)
    session = _make_session()

    snapshot = await service.build_snapshot(
        session_record=session,
        user_id=1,
        template_name="lockdown",
    )

    merged = snapshot.resolved_policy_document.get("tool_tier_overrides", {})
    assert merged == {"*": "individual"}


@pytest.mark.asyncio
async def test_no_template_leaves_policy_unchanged() -> None:
    from tldw_Server_API.app.services.acp_runtime_policy_service import (
        ACPRuntimePolicyService,
    )

    original_doc = {"allowed_tools": ["web.search"]}
    resolver = _StubPolicyResolver(policy_document=original_doc)
    service = ACPRuntimePolicyService(policy_resolver=resolver)
    session = _make_session()

    snapshot = await service.build_snapshot(
        session_record=session,
        user_id=1,
        # No template_name provided
    )

    # Should NOT have tool_tier_overrides injected
    assert "tool_tier_overrides" not in snapshot.resolved_policy_document


@pytest.mark.asyncio
async def test_unknown_template_name_is_ignored() -> None:
    from tldw_Server_API.app.services.acp_runtime_policy_service import (
        ACPRuntimePolicyService,
    )

    resolver = _StubPolicyResolver(policy_document={})
    service = ACPRuntimePolicyService(policy_resolver=resolver)
    session = _make_session()

    snapshot = await service.build_snapshot(
        session_record=session,
        user_id=1,
        template_name="nonexistent-template",
    )

    # No template matched, so no overrides should be injected
    assert "tool_tier_overrides" not in snapshot.resolved_policy_document
