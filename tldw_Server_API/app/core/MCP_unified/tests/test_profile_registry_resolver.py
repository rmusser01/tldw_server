"""Tests for package-local MCP profile store and resolver primitives."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import mcp_unified.profiles as profiles_pkg
import pytest
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.resolver import (
    AssignmentBackedProfileResolver,
    StoreBackedProfileResolver,
)
from mcp_unified.profiles.store import (
    InMemoryProfileAssignmentStore,
    InMemoryProfileStore,
    ProfileAssignmentStoreUnavailableError,
    ProfileStoreUnavailableError,
)
from mcp_unified.storage.models import ProfileAssignment


def _tldw_imports_for(path: Path) -> list[str]:
    """Return imports from a Python file that cross into the host package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API" or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tldw_Server_API" or node.module.startswith("tldw_Server_API."):
                imports.append(node.module)
    return imports


def test_profile_store_and_resolver_modules_have_no_tldw_server_imports() -> None:
    package_root = Path(profiles_pkg.__file__).resolve().parent

    offenders: dict[str, list[str]] = {}
    for path in package_root.rglob("*.py"):
        imports = _tldw_imports_for(path)
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


@pytest.mark.asyncio
async def test_in_memory_profile_store_returns_copy_isolated_profiles() -> None:
    store = InMemoryProfileStore()
    profile = MCPProfile(id="architect-workspace", name="Architect Workspace")

    stored = await store.upsert_profile(profile)
    stored.name = "Mutated Stored"

    first = await store.get_profile("architect-workspace")
    assert first is not None
    assert first.name == "Architect Workspace"
    first.name = "Mutated First"

    second = await store.get_profile("architect-workspace")
    assert second is not None
    assert second.name == "Architect Workspace"


@pytest.mark.asyncio
async def test_in_memory_profile_store_lists_and_deletes_profiles() -> None:
    store = InMemoryProfileStore(
        [
            MCPProfile(id="code-reviewer", name="Code Reviewer"),
            {
                "id": "architect",
                "name": "Architect",
                "metadata": {"agent_metadata": {"ui_label": "Architect"}},
            },
        ]
    )

    listed = await store.list_profiles()
    assert [profile.id for profile in listed] == ["architect", "code-reviewer"]

    listed[0].name = "Mutated"
    architect = await store.get_profile("architect")
    assert architect is not None
    assert architect.name == "Architect"

    assert await store.delete_profile("architect") is True
    assert await store.get_profile("architect") is None
    assert await store.delete_profile("architect") is False


@pytest.mark.asyncio
async def test_store_backed_resolver_resolves_explicit_enabled_profile() -> None:
    store = InMemoryProfileStore()
    await store.upsert_profile(MCPProfile(id="code-reviewer", name="Code Reviewer"))
    resolver = StoreBackedProfileResolver(store)

    profile = await resolver.resolve_profile("code-reviewer")

    assert profile is not None
    assert profile.id == "code-reviewer"
    profile.name = "Mutated"

    original = await store.get_profile("code-reviewer")
    assert original is not None
    assert original.name == "Code Reviewer"


@pytest.mark.asyncio
async def test_store_backed_resolver_uses_default_only_without_explicit_id() -> None:
    store = InMemoryProfileStore()
    await store.upsert_profile(MCPProfile(id="default", name="Default"))
    resolver = StoreBackedProfileResolver(store, default_profile_id="default")

    default_profile = await resolver.resolve_profile(None)
    assert default_profile is not None
    assert default_profile.id == "default"
    assert await resolver.resolve_profile("missing") is None


@pytest.mark.asyncio
async def test_store_backed_resolver_returns_none_for_disabled_profiles() -> None:
    store = InMemoryProfileStore()
    await store.upsert_profile(MCPProfile(id="disabled", name="Disabled", enabled=False))
    resolver = StoreBackedProfileResolver(store, default_profile_id="disabled")

    assert await resolver.resolve_profile("disabled") is None
    assert await resolver.resolve_profile(None) is None


@pytest.mark.asyncio
async def test_store_backed_resolver_fails_closed_when_store_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class UnavailableStore:
        async def get_profile(self, profile_id: str) -> MCPProfile | None:
            raise ProfileStoreUnavailableError(f"profile store unavailable: {profile_id}")

        async def list_profiles(self) -> list[MCPProfile]:
            raise ProfileStoreUnavailableError("profile store unavailable")

        async def upsert_profile(
            self,
            profile: MCPProfile,
        ) -> MCPProfile:
            raise ProfileStoreUnavailableError("profile store unavailable")

        async def delete_profile(self, profile_id: str) -> bool:
            raise ProfileStoreUnavailableError(f"profile store unavailable: {profile_id}")

    resolver = StoreBackedProfileResolver(UnavailableStore(), default_profile_id="default")
    caplog.set_level(logging.WARNING, logger="mcp_unified.profiles.resolver")

    assert await resolver.resolve_profile("explicit") is None
    assert await resolver.resolve_profile(None) is None
    assert "MCP profile 'explicit'" in caplog.text
    assert "MCP profile 'default'" in caplog.text


@pytest.mark.asyncio
async def test_assignment_backed_resolver_uses_stored_default_assignment() -> None:
    profile_store = InMemoryProfileStore(
        [
            MCPProfile(id="default", name="Default"),
            MCPProfile(id="explicit", name="Explicit"),
        ]
    )
    assignment_store = InMemoryProfileAssignmentStore(
        [
            ProfileAssignment(
                id="gateway-default",
                profile_id="default",
                is_default=True,
            ),
        ]
    )
    resolver = AssignmentBackedProfileResolver(
        profile_store,
        assignment_store=assignment_store,
        fallback_default_profile_id=None,
    )

    result = await resolver.resolve_profile_result(None)

    assert result.status == "resolved"
    assert result.profile is not None
    assert result.profile.id == "default"
    assert result.provenance["used_default_assignment"] is True
    assert result.provenance["default_assignment_id"] == "gateway-default"


@pytest.mark.asyncio
async def test_assignment_backed_resolver_explicit_profile_bypasses_default() -> None:
    class RaisingAssignmentStore(InMemoryProfileAssignmentStore):
        async def list_assignments(
            self,
            *,
            profile_id: str | None = None,
            principal_id: str | None = None,
            workspace_id: str | None = None,
        ) -> list[ProfileAssignment]:
            raise AssertionError("explicit profile should bypass default assignment")

    profile_store = InMemoryProfileStore(
        [
            MCPProfile(id="default", name="Default"),
            MCPProfile(id="explicit", name="Explicit"),
        ]
    )
    resolver = AssignmentBackedProfileResolver(
        profile_store,
        assignment_store=RaisingAssignmentStore(
            [
                ProfileAssignment(
                    id="gateway-default",
                    profile_id="default",
                    is_default=True,
                ),
            ]
        ),
        fallback_default_profile_id=None,
    )

    result = await resolver.resolve_profile_result("explicit")

    assert result.status == "resolved"
    assert result.profile is not None
    assert result.profile.id == "explicit"
    assert result.provenance["used_default_assignment"] is False
    assert result.provenance["default_assignment_id"] is None


@pytest.mark.asyncio
async def test_assignment_backed_resolver_falls_back_when_no_assignment_exists() -> None:
    profile_store = InMemoryProfileStore([MCPProfile(id="fallback", name="Fallback")])
    resolver = AssignmentBackedProfileResolver(
        profile_store,
        assignment_store=InMemoryProfileAssignmentStore(),
        fallback_default_profile_id="fallback",
    )

    result = await resolver.resolve_profile_result(None)

    assert result.status == "resolved"
    assert result.profile is not None
    assert result.profile.id == "fallback"
    assert result.provenance["used_default_assignment"] is False
    assert result.provenance["used_default_profile"] is True


@pytest.mark.asyncio
async def test_assignment_backed_resolver_fails_closed_when_assignment_store_unavailable() -> None:
    class UnavailableAssignmentStore:
        async def get_assignment(
            self,
            assignment_id: str,
        ) -> ProfileAssignment | None:
            raise ProfileAssignmentStoreUnavailableError(assignment_id)

        async def list_assignments(
            self,
            *,
            profile_id: str | None = None,
            principal_id: str | None = None,
            workspace_id: str | None = None,
        ) -> list[ProfileAssignment]:
            raise ProfileAssignmentStoreUnavailableError("assignment store unavailable")

        async def upsert_assignment(
            self,
            assignment: ProfileAssignment,
        ) -> ProfileAssignment:
            raise ProfileAssignmentStoreUnavailableError(assignment.id)

        async def delete_assignment(self, assignment_id: str) -> bool:
            raise ProfileAssignmentStoreUnavailableError(assignment_id)

    resolver = AssignmentBackedProfileResolver(
        InMemoryProfileStore([MCPProfile(id="fallback", name="Fallback")]),
        assignment_store=UnavailableAssignmentStore(),
        fallback_default_profile_id="fallback",
    )

    result = await resolver.resolve_profile_result(None)

    assert result.status == "store_unavailable"
    assert result.reason_code == "assignment_store_unavailable"
    assert result.provenance["used_default_assignment"] is False
    assert result.provenance["resolved_profile_id"] is None
