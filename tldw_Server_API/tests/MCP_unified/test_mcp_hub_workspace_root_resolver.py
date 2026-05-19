from __future__ import annotations

from pathlib import Path

import pytest


class _FakeSandboxService:
    def __init__(
        self,
        *,
        session_roots: dict[str, str] | None = None,
        session_owners: dict[str, str] | None = None,
        workspace_paths: dict[tuple[str, str], list[str]] | None = None,
    ) -> None:
        self.session_roots = dict(session_roots or {})
        self.session_owners = {str(key): str(value) for key, value in dict(session_owners or {}).items()}
        self.workspace_paths = {
            (str(user_id), str(workspace_id)): list(paths)
            for (user_id, workspace_id), paths in dict(workspace_paths or {}).items()
        }

    def get_session_workspace_path(self, session_id: str) -> str | None:
        return self.session_roots.get(session_id)

    def get_session_workspace_path_for_user(self, session_id: str, user_id: str) -> str | None:
        if self.session_owners.get(str(session_id)) != str(user_id):
            return None
        return self.get_session_workspace_path(session_id)

    def list_workspace_paths_for_user_workspace(self, *, user_id: str, workspace_id: str) -> list[str]:
        return list(self.workspace_paths.get((str(user_id), str(workspace_id)), []))


class _FakeRepo:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = list(rows or [])

    async def list_shared_workspace_entries(self, **kwargs) -> list[dict]:
        scope_type = kwargs.get("owner_scope_type")
        scope_id = kwargs.get("owner_scope_id")
        workspace_id = kwargs.get("workspace_id")
        rows = list(self.rows)
        if scope_type is not None:
            rows = [row for row in rows if row.get("owner_scope_type") == scope_type]
        if scope_id is not None or scope_type == "global":
            rows = [row for row in rows if row.get("owner_scope_id") == scope_id]
        if workspace_id is not None:
            rows = [row for row in rows if row.get("workspace_id") == workspace_id]
        return rows


@pytest.mark.asyncio
async def test_workspace_root_resolver_prefers_session_root() -> None:
    from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
        McpHubWorkspaceRootResolver,
    )

    resolver = McpHubWorkspaceRootResolver(
        sandbox_service=_FakeSandboxService(
            session_roots={"sess-1": "/tmp/mcp-hub-workspace/session-root"},  # nosec B108
            session_owners={"sess-1": "7"},
            workspace_paths={("7", "workspace-direct"): ["/tmp/mcp-hub-workspace/direct-root"]},  # nosec B108
        )
    )

    result = await resolver.resolve_for_context(
        session_id="sess-1",
        user_id="7",
        workspace_id="workspace-direct",
    )

    assert result["workspace_root"] == str(Path("/tmp/mcp-hub-workspace/session-root").resolve())  # nosec
    assert result["source"] == "sandbox_session"  # nosec B101
    assert result["reason"] is None  # nosec B101


@pytest.mark.asyncio
async def test_workspace_root_resolver_fails_closed_without_user_context() -> None:
    from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
        McpHubWorkspaceRootResolver,
    )

    resolver = McpHubWorkspaceRootResolver(
        sandbox_service=_FakeSandboxService(
            session_roots={"sess-1": "/tmp/mcp-hub-workspace/session-root"},  # nosec B108
            session_owners={"sess-1": "7"},
        )
    )

    result = await resolver.resolve_for_context(
        session_id="sess-1",
        user_id=None,
        workspace_id=None,
    )

    assert result["workspace_root"] is None  # nosec B101
    assert result["source"] is None  # nosec B101
    assert result["reason"] == "workspace_root_unavailable"  # nosec B101


@pytest.mark.asyncio
async def test_workspace_root_resolver_ignores_session_root_owned_by_other_user() -> None:
    from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
        McpHubWorkspaceRootResolver,
    )

    resolver = McpHubWorkspaceRootResolver(
        sandbox_service=_FakeSandboxService(
            session_roots={"sess-1": "/tmp/mcp-hub-workspace/session-root"},  # nosec B108
            session_owners={"sess-1": "99"},
            workspace_paths={("7", "workspace-direct"): ["/tmp/mcp-hub-workspace/direct-root"]},  # nosec B108
        )
    )

    result = await resolver.resolve_for_context(
        session_id="sess-1",
        user_id="7",
        workspace_id="workspace-direct",
    )

    assert result["workspace_root"] == str(Path("/tmp/mcp-hub-workspace/direct-root").resolve())  # nosec
    assert result["source"] == "sandbox_workspace_lookup"  # nosec B101
    assert result["reason"] is None  # nosec B101


@pytest.mark.asyncio
async def test_workspace_root_resolver_resolves_direct_workspace_id_for_user() -> None:
    from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
        McpHubWorkspaceRootResolver,
    )

    resolver = McpHubWorkspaceRootResolver(
        sandbox_service=_FakeSandboxService(
            workspace_paths={("7", "workspace-direct"): ["/tmp/mcp-hub-workspace/direct-root"]}  # nosec B108
        )
    )

    result = await resolver.resolve_for_context(
        session_id=None,
        user_id="7",
        workspace_id="workspace-direct",
    )

    assert result["workspace_root"] == str(Path("/tmp/mcp-hub-workspace/direct-root").resolve())  # nosec
    assert result["source"] == "sandbox_workspace_lookup"  # nosec B101
    assert result["reason"] is None  # nosec B101


@pytest.mark.asyncio
async def test_workspace_root_resolver_fails_closed_for_ambiguous_workspace_id() -> None:
    from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
        McpHubWorkspaceRootResolver,
    )

    resolver = McpHubWorkspaceRootResolver(
        sandbox_service=_FakeSandboxService(
            workspace_paths={
                ("7", "workspace-direct"): [
                    "/tmp/mcp-hub-workspace/direct-root-a",  # nosec B108
                    "/tmp/mcp-hub-workspace/direct-root-b",  # nosec B108
                ]
            }
        )
    )

    result = await resolver.resolve_for_context(
        session_id=None,
        user_id="7",
        workspace_id="workspace-direct",
    )

    assert result["workspace_root"] is None  # nosec B101
    assert result["reason"] == "workspace_root_ambiguous"  # nosec B101


@pytest.mark.asyncio
async def test_workspace_root_resolver_uses_shared_registry_same_scope_first() -> None:
    from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
        McpHubWorkspaceRootResolver,
    )

    resolver = McpHubWorkspaceRootResolver(
        sandbox_service=_FakeSandboxService(),
        repo=_FakeRepo(
            [
                {
                    "workspace_id": "shared-docs",
                    "absolute_root": "/srv/shared/docs-team",
                    "owner_scope_type": "team",
                    "owner_scope_id": 21,
                    "is_active": True,
                },
                {
                    "workspace_id": "shared-docs",
                    "absolute_root": "/srv/shared/docs-global",
                    "owner_scope_type": "global",
                    "owner_scope_id": None,
                    "is_active": True,
                },
            ]
        ),
    )

    result = await resolver.resolve_for_context(
        session_id=None,
        user_id="7",
        workspace_id="shared-docs",
        workspace_trust_source="shared_registry",
        owner_scope_type="team",
        owner_scope_id=21,
    )

    assert result["workspace_root"] == str(Path("/srv/shared/docs-team").resolve())  # nosec B101
    assert result["source"] == "shared_registry"  # nosec B101
    assert result["reason"] is None  # nosec B101


@pytest.mark.asyncio
async def test_workspace_root_resolver_falls_back_to_global_shared_registry_entry() -> None:
    from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
        McpHubWorkspaceRootResolver,
    )

    resolver = McpHubWorkspaceRootResolver(
        sandbox_service=_FakeSandboxService(),
        repo=_FakeRepo(
            [
                {
                    "workspace_id": "shared-docs",
                    "absolute_root": "/srv/shared/docs-global",
                    "owner_scope_type": "global",
                    "owner_scope_id": None,
                    "is_active": True,
                },
            ]
        ),
    )

    result = await resolver.resolve_for_context(
        session_id=None,
        user_id="7",
        workspace_id="shared-docs",
        workspace_trust_source="shared_registry",
        owner_scope_type="team",
        owner_scope_id=21,
    )

    assert result["workspace_root"] == str(Path("/srv/shared/docs-global").resolve())  # nosec B101
    assert result["source"] == "shared_registry"  # nosec B101
    assert result["reason"] is None  # nosec B101
