import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig, create_tool_definition
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
from tldw_Server_API.app.core.MCP_unified.tests.support import (
    build_mcp_admin_auth_override,
    build_mcp_test_client,
    reset_mcp_test_state,
)


def _setup_env(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    monkeypatch.setenv("MCP_JWT_SECRET", "x" * 64)
    monkeypatch.setenv("MCP_API_KEY_SALT", "s" * 64)
    monkeypatch.setenv("MCP_TRUST_X_FORWARDED", "1")
    monkeypatch.setenv("MCP_ALLOWED_IPS", "")
    monkeypatch.setenv("MCP_CLIENT_CERT_REQUIRED", "false")
    reset_mcp_test_state()


class IdempWriteModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict:
        return {"ready": True}

    async def get_tools(self) -> list[dict]:
        # Mark as write via metadata.category
        return [
            create_tool_definition(
                name="idemp_write",
                description="Write with idempotency",
                parameters={
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
                metadata={"category": "ingestion"},
            )
        ]

    def validate_tool_arguments(self, tool_name: str, arguments: dict):
        if tool_name == "idemp_write":
            if not isinstance(arguments, dict) or "x" not in arguments:
                raise ValueError("'x' is required")

    async def execute_tool(self, tool_name: str, arguments: dict, context=None):
        return f"ok:{arguments.get('x')}"


@pytest.fixture
def client(monkeypatch):
    _setup_env(monkeypatch)
    with build_mcp_test_client(
        auth_principal_override=build_mcp_admin_auth_override(),
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_prometheus_exports_idempotency_counters(client):
    # Relax IP guard for tests
    from tldw_Server_API.app.core.MCP_unified.security.ip_filter import get_ip_access_controller
    ctrl = get_ip_access_controller()
    ctrl.trust_x_forwarded_for = True
    ctrl.allowed_networks = []
    ctrl.blocked_networks = []

    # Bypass RBAC in protocol
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server
    class _AllowAll:
        async def check_permission(self, *args, **kwargs):
            return True
    get_mcp_server().protocol.rbac_policy = _AllowAll()

    # Register module
    reg = get_module_registry()
    await reg.register_module("idemp_mod", IdempWriteModule, ModuleConfig(name="idemp_mod"))

    token = get_jwt_manager().create_access_token(subject="1", roles=["admin"])  # auth for /request

    # First call (miss)
    req1 = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "idemp_write",
            "arguments": {"x": "A"},
            "idempotencyKey": "k-abc",
        },
        "id": "i1",
    }
    r1 = client.post(
        "/api/v1/mcp/request",
        json=req1,
        headers={"Authorization": f"Bearer {token}", "X-Forwarded-For": "127.0.0.1"},
    )
    assert r1.status_code == 200  # nosec B101

    # Second call (hit)
    req2 = req1 | {"id": "i2"}
    r2 = client.post(
        "/api/v1/mcp/request",
        json=req2,
        headers={"Authorization": f"Bearer {token}", "X-Forwarded-For": "127.0.0.1"},
    )
    assert r2.status_code == 200  # nosec B101

    # Scrape Prometheus metrics with auth
    r3 = client.get(
        "/api/v1/mcp/metrics/prometheus",
        headers={
            "X-Forwarded-For": "127.0.0.1",
            "Authorization": f"Bearer {token}",
        },
    )
    assert r3.status_code == 200  # nosec B101
    text = r3.text
    # Expect idempotency miss + hit counters labeled for our tool
    assert "mcp_idempotency_misses_total" in text  # nosec B101
    assert 'tool="idemp_write"' in text  # nosec B101
    assert "mcp_idempotency_hits_total" in text  # nosec B101
    assert 'tool="idemp_write"' in text  # nosec B101
