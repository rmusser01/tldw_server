import pytest

from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
    McpHubWorkspaceRootResolver,
)


class _SandboxStub:
    def get_session_workspace_path_for_user(self, session_id, user_id):
        return None

    def get_session_workspace_path(self, session_id):
        return "/tmp/unsafe-session-only-root"  # nosec B108

    def list_workspace_paths_for_user_workspace(self, user_id, workspace_id):
        return []


@pytest.mark.asyncio
async def test_resolver_does_not_use_session_only_workspace_path_without_user_binding():
    resolver = McpHubWorkspaceRootResolver(sandbox_service=_SandboxStub())

    result = await resolver.resolve_for_context(
        session_id="sess-1",
        user_id=None,
        workspace_id="ws-1",
    )

    assert result["workspace_root"] is None  # nosec B101
    assert result["reason"] == "workspace_root_unavailable"  # nosec B101
    assert result["source"] is None  # nosec B101
