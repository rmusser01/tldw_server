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


class StubWriteModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict:
        return {"ready": True}

    async def get_tools(self) -> list[dict]:
        # A write-capable tool with required 'id' (schema) and custom validator
        return [
            create_tool_definition(
                name="delete_item",
                description="Delete an item",
                parameters={
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
                metadata={"category": "ingestion"},
            )
        ]

    def validate_tool_arguments(self, tool_name: str, arguments: dict):
        if tool_name == "delete_item":
            if not isinstance(arguments, dict) or not arguments.get("id"):
                raise ValueError("'id' is required")

    async def execute_tool(self, tool_name: str, arguments: dict, context=None):
        return "ok"


@pytest.fixture
def client(monkeypatch):
    _setup_env(monkeypatch)
    with build_mcp_test_client(
        auth_principal_override=build_mcp_admin_auth_override(),
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_prometheus_exports_validation_counters(client):
    # Relax IP guard dynamically for testclient
    from tldw_Server_API.app.core.MCP_unified.security.ip_filter import get_ip_access_controller
    try:
        ctrl = get_ip_access_controller()
        ctrl.trust_x_forwarded_for = True
        ctrl.allowed_networks = []
        ctrl.blocked_networks = []
    except Exception:
        _ = None
    # Bypass RBAC in protocol for this test
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server
    class _AllowAll:
        async def check_permission(self, *args, **kwargs):
            return True
    get_mcp_server().protocol.rbac_policy = _AllowAll()

    # Register stub module
    reg = get_module_registry()
    await reg.register_module("stub_write", StubWriteModule, ModuleConfig(name="stub_write"))

    # Auth token
    token = get_jwt_manager().create_access_token(subject="1", roles=["admin"])  # ensure allowed

    # Trigger invalid params (missing id)
    bad = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "delete_item", "arguments": {}},
        "id": "m1",
    }
    r1 = client.post(
        "/api/v1/mcp/request",
        json=bad,
        headers={"Authorization": f"Bearer {token}", "X-Forwarded-For": "127.0.0.1"},
    )
    assert r1.status_code == 200  # nosec B101
    body1 = r1.json()
    assert isinstance(body1, dict)  # nosec B101
    assert body1.get("error") is not None  # nosec B101

    # Scrape Prometheus metrics with auth
    r2 = client.get(
        "/api/v1/mcp/metrics/prometheus",
        headers={
            "X-Forwarded-For": "127.0.0.1",
            "Authorization": f"Bearer {token}",
        },
    )
    assert r2.status_code == 200  # nosec B101
    text = r2.text
    # Expect either schema or validator invalid counter present for our tool
    assert "mcp_tool_invalid_params_total" in text  # nosec B101
    assert "tool=\"delete_item\"" in text  # nosec B101


class StubWriteNoValidator(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict:
        return {"ready": True}

    async def get_tools(self) -> list[dict]:
        # Marked as write-capable but without validate_tool_arguments override
        return [
            create_tool_definition(
                name="delete_item2",
                description="Delete an item (no validator)",
                parameters={
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
                metadata={"category": "ingestion"},
            )
        ]

    async def execute_tool(self, tool_name: str, arguments: dict, context=None):
        return "ok"


@pytest.mark.asyncio
async def test_prometheus_validator_missing_counter(client):
    # Relax IP guard dynamically for testclient
    from tldw_Server_API.app.core.MCP_unified.security.ip_filter import get_ip_access_controller
    try:
        ctrl = get_ip_access_controller()
        ctrl.trust_x_forwarded_for = True
        ctrl.allowed_networks = []
        ctrl.blocked_networks = []
    except Exception:
        _ = None
    # Bypass RBAC in protocol for this test
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server
    class _AllowAll:
        async def check_permission(self, *args, **kwargs):
            return True
    get_mcp_server().protocol.rbac_policy = _AllowAll()

    # Register module without validator override
    reg = get_module_registry()
    await reg.register_module("stub_write2", StubWriteNoValidator, ModuleConfig(name="stub_write2"))

    token = get_jwt_manager().create_access_token(subject="1", roles=["admin"])  # ensure allowed

    # Even with valid args, protocol should enforce validator presence and count it
    good = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "delete_item2", "arguments": {"id": "abc"}},
        "id": "m2",
    }
    r1 = client.post(
        "/api/v1/mcp/request",
        json=good,
        headers={"Authorization": f"Bearer {token}", "X-Forwarded-For": "127.0.0.1"},
    )
    assert r1.status_code == 200  # nosec B101
    body1 = r1.json()
    assert isinstance(body1, dict)  # nosec B101
    assert body1.get("error") is not None  # nosec B101

    # Scrape metrics and assert validator-missing counter appears
    r2 = client.get(
        "/api/v1/mcp/metrics/prometheus",
        headers={
            "X-Forwarded-For": "127.0.0.1",
            "Authorization": f"Bearer {token}",
        },
    )
    assert r2.status_code == 200  # nosec B101
    text = r2.text
    assert "mcp_tool_validator_missing_total" in text  # nosec B101
    assert "tool=\"delete_item2\"" in text  # nosec B101
