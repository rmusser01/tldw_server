"""Tests for structured MCP profile resolution primitives."""

from __future__ import annotations

import mcp_unified.profiles.resolution as profile_resolution
import pytest
from mcp_unified.profiles.models import MCPProfile, ProfilePolicy
from mcp_unified.profiles.resolver import StoreBackedProfileResolver
from mcp_unified.profiles.store import InMemoryProfileStore, ProfileStoreUnavailableError


@pytest.mark.asyncio
async def test_profile_result_reports_required_when_no_explicit_or_default_profile() -> None:
    resolver = StoreBackedProfileResolver(InMemoryProfileStore())

    result = await resolver.resolve_profile_result(None)

    assert result.status == "profile_required"
    assert result.reason_code == "profile_required"
    assert result.profile is None


@pytest.mark.asyncio
async def test_profile_result_reports_disabled_profile_with_provenance() -> None:
    store = InMemoryProfileStore(
        [
            MCPProfile(id="disabled", name="Disabled", enabled=False),
        ]
    )
    resolver = StoreBackedProfileResolver(store)

    result = await resolver.resolve_profile_result("disabled")

    assert result.status == "profile_disabled"
    assert result.reason_code == "profile_disabled"
    assert result.profile is None
    assert result.provenance["profile_id"] == "disabled"


@pytest.mark.asyncio
async def test_profile_result_reports_missing_profile_with_default_provenance() -> None:
    store = InMemoryProfileStore(
        [
            MCPProfile(id="default", name="Default"),
        ]
    )
    resolver = StoreBackedProfileResolver(store, default_profile_id="default")

    missing = await resolver.resolve_profile_result("missing")
    resolved_default = await resolver.resolve_profile_result(None)

    assert missing.status == "profile_not_found"
    assert missing.reason_code == "profile_not_found"
    assert missing.provenance["requested_profile_id"] == "missing"
    assert missing.provenance["resolved_profile_id"] == "missing"
    assert missing.provenance["used_default_profile"] is False
    assert resolved_default.status == "resolved"
    assert resolved_default.profile is not None
    assert resolved_default.profile.id == "default"
    assert resolved_default.provenance["requested_profile_id"] is None
    assert resolved_default.provenance["resolved_profile_id"] == "default"
    assert resolved_default.provenance["used_default_profile"] is True


@pytest.mark.asyncio
async def test_profile_result_reports_store_unavailable_with_reason_code() -> None:
    class UnavailableStore:
        async def get_profile(self, profile_id: str) -> MCPProfile | None:
            raise ProfileStoreUnavailableError(
                f"profile store unavailable: {profile_id}"
            )

        async def list_profiles(self) -> list[MCPProfile]:
            raise ProfileStoreUnavailableError("profile store unavailable")

        async def upsert_profile(self, profile: MCPProfile) -> MCPProfile:
            raise ProfileStoreUnavailableError("profile store unavailable")

        async def delete_profile(self, profile_id: str) -> bool:
            raise ProfileStoreUnavailableError(
                f"profile store unavailable: {profile_id}"
            )

    resolver = StoreBackedProfileResolver(UnavailableStore(), default_profile_id="default")

    result = await resolver.resolve_profile_result(None)

    assert result.status == "store_unavailable"
    assert result.reason_code == "store_unavailable"
    assert result.profile is None
    assert result.provenance["profile_id"] == "default"
    assert result.provenance["used_default_profile"] is True


@pytest.mark.asyncio
async def test_resolve_profile_keeps_legacy_none_wrapper_behavior() -> None:
    resolver = StoreBackedProfileResolver(InMemoryProfileStore())

    assert await resolver.resolve_profile(None) is None


def test_effective_policy_requires_workspace_scope_for_write_capable_profiles() -> None:
    profile = MCPProfile(
        id="backend-engineer",
        name="Backend Engineer",
        policy_document=ProfilePolicy(
            capabilities=["source.write_scoped"],
            risk_classes=["mutating"],
        ),
    )

    result = profile_resolution.build_effective_policy_result(profile)

    assert result.status == "denied"
    assert result.reason_code == "workspace_scope_required"
    assert result.policy is None
    assert result.provenance["profile_id"] == "backend-engineer"


def test_effective_policy_allows_read_only_profile_without_workspace_binding() -> None:
    profile = MCPProfile(
        id="code-reviewer",
        name="Code Reviewer",
        policy_document=ProfilePolicy(
            capabilities=["code_search", "filesystem.read"],
        ),
    )

    result = profile_resolution.build_effective_policy_result(profile)

    assert result.status == "resolved"
    assert result.reason_code == "resolved"
    assert result.policy is not None
    assert result.policy.capabilities == ["code_search", "filesystem.read"]


def test_effective_policy_allows_write_profile_with_assignment_binding() -> None:
    profile = MCPProfile(
        id="backend-engineer",
        name="Backend Engineer",
        policy_document=ProfilePolicy(
            capabilities=["source.write_scoped"],
            risk_classes=["mutating"],
        ),
    )

    result = profile_resolution.build_effective_policy_result(
        profile,
        assignment_binding={"workspace_id": "workspace-1"},
    )

    assert result.status == "resolved"
    assert result.reason_code == "resolved"
    assert result.policy is not None
    assert result.policy.profile_id == "backend-engineer"


def test_effective_policy_denied_tool_overrides_allowed_tool() -> None:
    profile = MCPProfile(
        id="strict-reviewer",
        name="Strict Reviewer",
        policy_document=ProfilePolicy(
            allowed_tools=["filesystem.read"],
            denied_tools=["filesystem.read"],
        ),
    )

    result = profile_resolution.build_effective_policy_result(
        profile,
        tool_name="filesystem.read",
    )

    assert result.status == "denied"
    assert result.reason_code == "tool_denied"
    assert result.policy is None


def test_effective_policy_requires_tool_or_capability_allow_for_tool_execution() -> None:
    profile = MCPProfile(
        id="capability-only",
        name="Capability Only",
        policy_document=ProfilePolicy(
            capabilities=["filesystem.read"],
        ),
    )

    result = profile_resolution.build_effective_policy_result(
        profile,
        tool_name="filesystem.read",
    )

    assert result.status == "denied"
    assert result.reason_code == "tool_not_allowed"
    assert result.policy is None


def test_effective_policy_allows_tool_execution_with_allowed_capability_mapping() -> None:
    profile = MCPProfile(
        id="capability-mapped",
        name="Capability Mapped",
        policy_document=ProfilePolicy(
            capabilities=["filesystem.read"],
        ),
    )

    result = profile_resolution.build_effective_policy_result(
        profile,
        tool_name="filesystem.read",
        capability="filesystem.read",
    )

    assert result.status == "resolved"
    assert result.reason_code == "resolved"
    assert result.policy is not None


def test_effective_policy_defaults_to_deny_for_unlisted_tool_execution() -> None:
    profile = MCPProfile(
        id="empty",
        name="Empty",
    )

    result = profile_resolution.build_effective_policy_result(
        profile,
        tool_name="filesystem.read",
    )

    assert result.status == "denied"
    assert result.reason_code == "tool_not_allowed"
    assert result.policy is None
