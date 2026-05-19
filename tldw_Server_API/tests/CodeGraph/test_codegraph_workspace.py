from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings
from tldw_Server_API.app.core.CodeGraph.workspace import CodeGraphWorkspaceResolver
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


class FakeRootResolver:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict[str, Any]] = []

    async def resolve_for_context(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"workspace_root": str(self.root), "workspace_id": "ws-1", "source": "test"}


@pytest.mark.asyncio
async def test_workspace_resolver_rejects_session_only_without_user(tmp_path: Path) -> None:
    resolver = CodeGraphWorkspaceResolver(FakeRootResolver(tmp_path), CodeGraphSettings.from_mapping({}))
    context = RequestContext(request_id="req", session_id="sess-1", user_id=None, metadata={})

    with pytest.raises(PermissionError, match="workspace_root_unavailable"):
        await resolver.resolve(context)


@pytest.mark.asyncio
async def test_workspace_key_is_stable_and_index_path_is_not_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    index_base = tmp_path / "indexes"
    resolver = CodeGraphWorkspaceResolver(
        FakeRootResolver(workspace),
        CodeGraphSettings.from_mapping({"index_base_dir": str(index_base)}),
    )
    context = RequestContext(request_id="req", session_id="sess-1", user_id="7", metadata={"workspace_id": "ws-1"})

    first = await resolver.resolve(context)
    second = await resolver.resolve(context)

    assert first.workspace_key == second.workspace_key
    assert first.workspace_key.startswith("ws_")
    assert first.workspace_root == workspace.resolve()
    assert first.index_db_path == index_base / first.workspace_key / "codegraph.db"
    assert workspace.resolve() not in first.index_db_path.resolve(strict=False).parents
    assert resolver._workspace_root_resolver.calls[0]["workspace_id"] == "ws-1"


@pytest.mark.asyncio
async def test_workspace_resolver_uses_selected_scope_id_when_primary_scope_id_is_none(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root_resolver = FakeRootResolver(workspace)
    resolver = CodeGraphWorkspaceResolver(root_resolver, CodeGraphSettings.from_mapping({}))
    context = RequestContext(
        request_id="req",
        session_id="sess-1",
        user_id="7",
        metadata={
            "workspace_id": "ws-1",
            "owner_scope_id": None,
            "selected_workspace_scope_id": "shared-scope-7",
        },
    )

    await resolver.resolve(context)

    assert root_resolver.calls[0]["owner_scope_id"] == "shared-scope-7"
