import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import mcp_discovery_module
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


_SENSITIVE_MEMBERSHIP_FAILURE = "membership backend exploded"


class _EmptyPool:
    async def fetchall(self, *_args, **_kwargs):
        return []


class _ExplodingPool:
    async def fetchall(self, *_args, **_kwargs):
        raise RuntimeError(_SENSITIVE_MEMBERSHIP_FAILURE)


def _capture_discovery_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = mcp_discovery_module.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level=level,
    )
    return messages, sink_id


def _module() -> mcp_discovery_module.MCPDiscoveryModule:
    return mcp_discovery_module.MCPDiscoveryModule(
        ModuleConfig(name="mcp_discovery", description="Discovery")
    )


@pytest.mark.asyncio
async def test_org_membership_lookup_failure_log_is_sanitized(monkeypatch):
    async def fail_membership_lookup(_user_id):
        raise RuntimeError(_SENSITIVE_MEMBERSHIP_FAILURE)

    async def get_empty_pool():
        return _EmptyPool()

    monkeypatch.setattr(
        mcp_discovery_module,
        "list_org_memberships_for_user",
        fail_membership_lookup,
    )
    monkeypatch.setattr(mcp_discovery_module, "get_db_pool", get_empty_pool)
    messages, sink_id = _capture_discovery_logs()

    try:
        org_ids, team_ids = await _module()._resolve_memberships(
            RequestContext(request_id="req-org", user_id="42", client_id="test"),
            admin_all=False,
        )
    finally:
        mcp_discovery_module.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert org_ids == set()
    assert team_ids == set()
    assert "MCP discovery: org membership lookup failed" in rendered_logs
    assert _SENSITIVE_MEMBERSHIP_FAILURE not in rendered_logs


@pytest.mark.asyncio
async def test_team_membership_lookup_failure_log_is_sanitized(monkeypatch):
    async def list_no_org_memberships(_user_id):
        return []

    async def get_exploding_pool():
        return _ExplodingPool()

    monkeypatch.setattr(
        mcp_discovery_module,
        "list_org_memberships_for_user",
        list_no_org_memberships,
    )
    monkeypatch.setattr(mcp_discovery_module, "get_db_pool", get_exploding_pool)
    messages, sink_id = _capture_discovery_logs()

    try:
        org_ids, team_ids = await _module()._resolve_memberships(
            RequestContext(request_id="req-team", user_id="42", client_id="test"),
            admin_all=False,
        )
    finally:
        mcp_discovery_module.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert org_ids == set()
    assert team_ids == set()
    assert "MCP discovery: team membership lookup failed" in rendered_logs
    assert _SENSITIVE_MEMBERSHIP_FAILURE not in rendered_logs
