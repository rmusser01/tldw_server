"""
Basic functionality tests for unified MCP module

Run with: python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py -v
"""

import os
import asyncio
import pytest
from typing import Dict, Any
from datetime import datetime

# Set test environment variables before importing modules
os.environ["MCP_JWT_SECRET"] = "test_secret_key_for_testing_only_32_chars_minimum"
os.environ["MCP_API_KEY_SALT"] = "test_salt_key_for_testing_only_32_chars_minimum"
os.environ["MCP_LOG_LEVEL"] = "DEBUG"
os.environ["MCP_RATE_LIMIT_ENABLED"] = "false"  # Disable rate limiting for tests

from tldw_Server_API.app.core.MCP_unified import (
    get_config,
    MCPServer,
    get_mcp_server,
    MCPRequest,
    MCPResponse,
    BaseModule,
    ModuleConfig,
    ModuleRegistry,
    get_module_registry,
    JWTManager,
    get_jwt_manager,
    RBACPolicy,
    get_rbac_policy,
    UserRole
)
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


# Test Module Implementation
class TestModule(BaseModule):
    """Simple test module for testing"""

    async def on_initialize(self) -> None:
        """Initialize test module"""
        self.initialized = True

    async def on_shutdown(self) -> None:
        """Shutdown test module"""
        self.initialized = False

    async def check_health(self) -> Dict[str, bool]:
        """Health check"""
        return {
            "test_check": True,
            "initialization": self.initialized
        }

    async def get_tools(self) -> list[Dict[str, Any]]:
        """Get test tools"""
        return [
            {
                "name": "echo",
                "description": "Echo back the input",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "add",
                "description": "Add two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            }
        ]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], context: Any | None = None) -> Any:
        """Execute test tool"""
        if tool_name == "echo":
            return arguments.get("message", "")
        elif tool_name == "add":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            return a + b
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


class TestConfiguration:
    """Test configuration loading"""

    def test_config_loads(self):
        """Test that configuration loads without errors"""
        config = get_config()
        assert config is not None
        assert config.server_name == "tldw-mcp-unified"
        assert config.jwt_secret_key is not None
        assert config.api_key_salt is not None

    def test_config_validates_secrets(self):
        """Test that configuration validates secrets properly"""
        config = get_config()
        # Should not be the default hardcoded value
        assert config.jwt_secret_key.get_secret_value() != "your-secret-key-change-this-in-production"
        assert len(config.jwt_secret_key.get_secret_value()) >= 32


class TestJWTManager:
    """Test JWT authentication manager"""

    def test_jwt_manager_initialization(self):
        """Test JWT manager initializes properly"""
        manager = get_jwt_manager()
        assert manager is not None

    def test_create_and_verify_token(self):
        """Test token creation and verification"""
        manager = get_jwt_manager()

        # Create token
        token = manager.create_access_token(
            subject="test_user",
            username="testuser",
            roles=["user"],
            permissions=["read", "write"]
        )

        assert token is not None
        assert isinstance(token, str)

        # Verify token
        token_data = manager.verify_token(token)
        assert token_data.sub == "test_user"
        assert token_data.username == "testuser"
        assert "user" in token_data.roles
        assert "read" in token_data.permissions

    def test_password_hashing(self):
        """Test password hashing and verification"""
        manager = get_jwt_manager()

        password = "test_password_123"
        hashed = manager.hash_password(password)

        assert hashed != password
        assert manager.verify_password(password, hashed)
        assert not manager.verify_password("wrong_password", hashed)


class TestRBACPolicy:
    """Test role-based access control"""

    def test_rbac_initialization(self):
        """Test RBAC policy initializes with default roles"""
        policy = get_rbac_policy()
        assert policy is not None

        # Check default roles exist
        assert UserRole.ADMIN.value in policy.roles
        assert UserRole.USER.value in policy.roles
        assert UserRole.GUEST.value in policy.roles

    def test_permission_checking(self):
        """Test permission checking"""
        from tldw_Server_API.app.core.MCP_unified.auth.rbac import Resource, Action

        policy = get_rbac_policy()

        # Assign admin role to test user
        policy.assign_role("test_admin", UserRole.ADMIN.value)

        # Admin should have all permissions
        assert policy.check_permission(
            "test_admin",
            Resource.TOOL,
            Action.EXECUTE,
            "any_tool"
        )

        # Assign user role to another test user
        policy.assign_role("test_user", UserRole.USER.value)

        # User should have limited permissions
        assert policy.check_permission(
            "test_user",
            Resource.TOOL,
            Action.EXECUTE
        )
        assert policy.check_permission(
            "test_user",
            Resource.MEDIA,
            Action.READ
        )


class TestBaseModule:
    """Test shared module base behavior."""

    def test_module_config_preserves_positional_settings_argument(self) -> None:
        """Test the new factory field does not shift legacy positional settings."""
        config = ModuleConfig(
            "positional_module",
            "1.0.0",
            "",
            "general",
            True,
            30,
            3,
            5,
            60,
            20,
            2.0,
            300,
            {"mode": "legacy"},
        )

        assert config.settings == {"mode": "legacy"}
        assert config.circuit_breaker_factory is None

    def test_module_config_accepts_circuit_breaker_factory(self) -> None:
        """Test module construction can inject circuit breaker creation."""
        fake_breaker = object()
        calls: list[tuple[str, Any]] = []

        def _fake_factory(*, name: str, config: Any) -> object:
            calls.append((name, config))
            return fake_breaker

        module = TestModule(
            ModuleConfig(
                name="custom_breaker_module",
                circuit_breaker_factory=_fake_factory,
            )
        )

        assert module._circuit_breaker is fake_breaker  # noqa: SLF001
        assert calls
        breaker_name, breaker_config = calls[0]
        assert breaker_name == "mcp_custom_breaker_module"
        assert breaker_config.failure_threshold == 5
        assert breaker_config.category == "mcp"
        assert breaker_config.service == "custom_breaker_module"


@pytest.mark.asyncio
class TestModuleRegistry:
    """Test module registry"""

    async def test_module_registration(self):
        """Test registering a module"""
        registry = ModuleRegistry()  # Create new instance for test

        config = ModuleConfig(
            name="test_module",
            version="1.0.0",
            description="Test module",
            department="test"
        )

        await registry.register_module("test", TestModule, config)

        # Check module is registered
        module = await registry.get_module("test")
        assert module is not None
        assert isinstance(module, TestModule)

    async def test_module_health_check(self):
        """Test module health checking"""
        registry = ModuleRegistry()

        config = ModuleConfig(
            name="test_module",
            version="1.0.0"
        )

        await registry.register_module("test", TestModule, config)

        # Check health
        health_results = await registry.check_all_health()
        assert "test" in health_results
        assert health_results["test"].is_healthy

    async def test_find_module_for_tool(self):
        """Test finding module that provides a tool"""
        registry = ModuleRegistry()

        config = ModuleConfig(name="test_module")
        await registry.register_module("test", TestModule, config)

        # Find module for "echo" tool
        module = await registry.find_module_for_tool("echo")
        assert module is not None
        assert await module.has_tool("echo")

        # Non-existent tool
        module = await registry.find_module_for_tool("non_existent")
        assert module is None


@pytest.mark.asyncio
class TestMCPProtocol:
    """Test MCP protocol handler"""

    async def test_initialize_request(self):
        """Test initialize request"""
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

        protocol = MCPProtocol()

        request = MCPRequest(
            method="initialize",
            params={"clientInfo": {"name": "Test Client"}},
            id=1
        )

        context = RequestContext(
            request_id="test_1",
            client_id="test_client"
        )

        response = await protocol.process_request(request, context)

        assert response.error is None
        assert response.result is not None
        assert response.result["protocolVersion"] == "2024-11-05"
        assert "capabilities" in response.result

    async def test_ping_request(self):
        """Test ping request"""
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

        protocol = MCPProtocol()

        request = MCPRequest(method="ping", id=2)
        context = RequestContext(request_id="test_2")

        response = await protocol.process_request(request, context)

        assert response.error is None
        assert response.result is not None
        assert response.result["pong"] is True

    async def test_invalid_method(self):
        """Test invalid method returns error"""
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

        protocol = MCPProtocol()

        request = MCPRequest(method="invalid_method", id=3)
        context = RequestContext(request_id="test_3")

        response = await protocol.process_request(request, context)

        assert response.error is not None
        assert response.error.code == -32601  # Method not found
        assert response.result is None


@pytest.mark.asyncio
class TestMCPServer:
    """Test MCP server"""

    async def test_server_initialization(self):
        """Test server initializes properly"""
        server = MCPServer()  # Create new instance for test

        assert not server.initialized

        await server.initialize()
        assert server.initialized

        await server.shutdown()
        assert not server.initialized

    async def test_server_status(self):
        """Test getting server status"""
        server = MCPServer()
        await server.initialize()

        status = await server.get_status()

        assert status["status"] == "healthy"
        assert status["version"] == "3.0.0"
        assert "uptime_seconds" in status
        assert status["uptime_seconds"] >= 0

        await server.shutdown()

    async def test_server_metrics(self):
        """Test getting server metrics"""
        server = MCPServer()
        await server.initialize()

        metrics = await server.get_metrics()

        assert "connections" in metrics
        assert "modules" in metrics

        await server.shutdown()


@pytest.mark.asyncio
async def test_server_initialize_single_flight(monkeypatch):
    server = MCPServer()
    server.config.metrics_enabled = False

    calls = {"health": 0, "register": 0, "seed": 0}

    async def _start_health():
        calls["health"] += 1
        await asyncio.sleep(0.02)

    async def _register_modules():
        calls["register"] += 1
        await asyncio.sleep(0.02)

    async def _seed_perms():
        calls["seed"] += 1
        await asyncio.sleep(0.02)

    monkeypatch.setattr(server.module_registry, "start_health_monitoring", _start_health)
    monkeypatch.setattr(server, "_register_default_modules", _register_modules)
    monkeypatch.setattr(server, "_ensure_default_tool_permissions", _seed_perms)
    monkeypatch.setattr("tldw_Server_API.app.core.MCP_unified.server.validate_config", lambda: True)

    try:
        await asyncio.gather(*(server.initialize() for _ in range(5)))

        assert server.initialized is True
        assert calls == {"health": 1, "register": 1, "seed": 1}
        # With metrics disabled, only connection/session cleanup loops should run.
        assert len(server.background_tasks) <= 2
    finally:
        await server.shutdown()


@pytest.mark.asyncio
class TestEndToEnd:
    """End-to-end integration tests"""

    async def test_tool_execution_flow(self):
        """Test complete tool execution flow"""
        # Create and initialize server
        server = MCPServer()
        await server.initialize()
        # Provide user context and permissive RBAC for this unit test
        class _AllowAll:
            async def check_permission(self, *args, **kwargs):
                return True
        server.protocol.rbac_policy = _AllowAll()

        # Register test module
        registry = server.module_registry
        config = ModuleConfig(name="test_module")
        await registry.register_module("test", TestModule, config)

        # Create request to execute tool
        request = MCPRequest(
            method="tools/call",
            params={
                "name": "echo",
                "arguments": {"message": "Hello, MCP!"}
            },
            id="test_tool_exec"
        )

        # Process request
        response = await server.handle_http_request(
            request,
            client_id="test_client",
            user_id="test_user"
        )

        assert response.error is None
        assert response.result is not None
        assert response.result["content"][0]["text"] == "Hello, MCP!"

        # Cleanup
        await server.shutdown()

    async def test_math_tool_execution(self):
        """Test math tool execution"""
        server = MCPServer()
        await server.initialize()
        # Provide user context and permissive RBAC for this unit test
        class _AllowAll:
            async def check_permission(self, *args, **kwargs):
                return True
        server.protocol.rbac_policy = _AllowAll()

        # Register test module
        registry = server.module_registry
        config = ModuleConfig(name="test_module")
        await registry.register_module("test", TestModule, config)

        # Execute add tool
        request = MCPRequest(
            method="tools/call",
            params={
                "name": "add",
                "arguments": {"a": 5, "b": 3}
            },
            id="test_add"
        )

        response = await server.handle_http_request(request, user_id="test_user")

        assert response.error is None
        assert response.result["content"][0]["text"] == "8"

        await server.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
