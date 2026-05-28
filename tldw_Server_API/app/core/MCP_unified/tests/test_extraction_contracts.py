from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INTERFACE_FILES = {"runtime.py", "policy.py", "storage.py"}


def _fake_runtime_dependencies() -> SimpleNamespace:
    lifecycle_guard = _RecordingLifecycleGuard()
    return SimpleNamespace(
        module_registry=object(),
        rbac_policy=object(),
        rate_limiter=object(),
        metrics_collector=object(),
        telemetry_provider=object(),
        database_path_resolver=object(),
        api_key_scope_normalizer=object(),
        effective_policy_resolver=object(),
        approval_evaluator=object(),
        path_scope_enforcer=object(),
        external_access_evaluator=object(),
        redis_client_factory=object(),
        circuit_breaker_factory=object(),
        auth_provider=_FakeAuthProvider(),
        lifecycle_guard=lifecycle_guard,
        permission_seeder=object(),
        module_config_provider=object(),
        policy_context_provider=object(),
    )


class _RecordingLifecycleGuard:
    """Lifecycle guard double that records shutdown-family registrations."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []

    def register_shutdown_transport_family(self, family: str, **kwargs: Any) -> None:
        self.registered.append({"family": family, **kwargs})

    def assert_may_start_work(self, app: Any, family: str) -> None:
        del app, family


class _RecordingModuleRegistry:
    """Small module registry double for MCPServer dependency-injection tests."""

    def __init__(self) -> None:
        self.started = 0
        self.registrations: list[dict[str, Any]] = []

    async def start_health_monitoring(self) -> None:
        self.started += 1

    async def register_module(self, module_id: str, module_type: type[Any], config: Any) -> None:
        self.registrations.append(
            {"module_id": module_id, "module_type": module_type, "config": config}
        )

    async def shutdown_all(self) -> None:
        return None


class _NoopMetrics:
    """Metrics double with async lifecycle methods used by server tests."""

    async def start_collection(self) -> None:
        return None

    async def stop_collection(self) -> None:
        return None

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


class _RecordingPermissionSeeder:
    """Permission seeder double that records whether initialize invoked it."""

    def __init__(self) -> None:
        self.calls = 0

    async def seed_default_tool_permissions(self) -> None:
        self.calls += 1


class _FakeModuleConfigProvider:
    """Module config provider double with deterministic default paths."""

    def __init__(self) -> None:
        self.media_db_path = "/tmp/mcp-stage3-media.db"

    def default_media_db_path(self) -> str:
        return self.media_db_path


class _FakePolicyContextProvider:
    """Policy-context provider double with a caller-controlled flag."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def is_policy_context_enabled(self) -> bool:
        return self.enabled


class _FakeAuthProvider:
    """Auth provider double for API-key permission normalization."""

    def get_mcp_jwt_manager(self) -> object:
        return object()

    def is_authnz_access_token(self, _token: str) -> bool:
        return False

    async def authenticate_authnz_websocket_token(
        self,
        _token: str,
        *,
        websocket: Any,
    ) -> None:
        del websocket
        return None

    async def validate_api_key(
        self,
        _api_key: str,
        *,
        ip_address: str | None = None,
    ) -> None:
        del ip_address
        return None

    def normalize_api_key_permissions(self, info: dict[str, Any] | None) -> list[str]:
        return ["fake.scope"] if info else []


def _server_runtime_dependencies() -> SimpleNamespace:
    deps = _fake_runtime_dependencies()
    deps.module_registry = _RecordingModuleRegistry()
    deps.metrics_collector = _NoopMetrics()
    deps.permission_seeder = _RecordingPermissionSeeder()
    deps.module_config_provider = _FakeModuleConfigProvider()
    deps.policy_context_provider = _FakePolicyContextProvider(enabled=False)
    deps.auth_provider = _FakeAuthProvider()
    return deps


def _interface_boundary_violations_for(path: Path, interface_dir: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_parent = path.parent.relative_to(interface_dir)
    relative_depth = 0 if relative_parent == Path(".") else len(relative_parent.parts)
    max_interface_relative_level = relative_depth + 1
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API"
                or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level > max_interface_relative_level:
                violations.append(
                    f"relative import escapes interfaces package: level={node.level}"
                )
            elif (
                node.module
                and node.level == 0
                and (
                    node.module == "tldw_Server_API"
                    or node.module.startswith("tldw_Server_API.")
                )
            ):
                violations.append(node.module)
    return violations


def test_new_interface_modules_do_not_import_tldw_server_api() -> None:
    interface_dir = MCP_ROOT / "interfaces"
    assert interface_dir.exists()
    interface_files = {path.name for path in interface_dir.rglob("*.py")}
    assert EXPECTED_INTERFACE_FILES.issubset(interface_files)
    offenders: dict[str, list[str]] = {}
    for path in interface_dir.rglob("*.py"):
        violations = _interface_boundary_violations_for(path, interface_dir)
        if violations:
            offenders[str(path)] = violations
    assert offenders == {}


def test_default_runtime_dependency_builder_exposes_core_dependencies() -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
        build_default_runtime_dependencies,
    )

    deps = build_default_runtime_dependencies()
    assert hasattr(deps, "module_registry")
    assert hasattr(deps, "rbac_policy")


@pytest.mark.asyncio
async def test_tldw_auth_provider_fails_closed_for_unencodable_ws_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime

    verify_calls = 0

    async def _unexpected_verify(*_args: Any, **_kwargs: Any) -> None:
        nonlocal verify_calls
        verify_calls += 1
        raise AssertionError("verify_jwt_and_fetch_user should not be called")

    monkeypatch.setattr(tldw_runtime, "get_jwt_manager", lambda: object())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.verify_jwt_and_fetch_user",
        _unexpected_verify,
    )

    provider = tldw_runtime.TldwServerAuthProvider()
    websocket = SimpleNamespace(headers={"x-bad": "snowman \u2603"}, client=None)

    assert (
        await provider.authenticate_authnz_websocket_token(
            "token",
            websocket=websocket,
        )
        is None
    )
    assert verify_calls == 0


def test_stage3_runtime_contracts_are_exported_by_interface_packages() -> None:
    import mcp_unified.interfaces as standalone_interfaces

    import tldw_Server_API.app.core.MCP_unified.interfaces as compat_interfaces

    names = [
        "AuthenticatedIdentity",
        "LifecycleGuard",
        "ModuleConfigProvider",
        "PermissionSeeder",
        "PolicyContextProvider",
        "ServerAuthProvider",
    ]

    for name in names:
        assert getattr(standalone_interfaces, name) is getattr(compat_interfaces, name)


def test_mcp_protocol_accepts_runtime_dependencies() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    deps = _fake_runtime_dependencies()
    protocol = MCPProtocol(dependencies=deps)
    assert protocol.module_registry is deps.module_registry
    assert protocol.rbac_policy is deps.rbac_policy


def test_mcp_server_accepts_runtime_dependencies() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _fake_runtime_dependencies()
    server = MCPServer(dependencies=deps)
    assert server.protocol.module_registry is deps.module_registry
    assert server.protocol.rbac_policy is deps.rbac_policy
    assert server.module_registry is deps.module_registry
    assert server.rbac_policy is deps.rbac_policy
    assert server.metrics_collector is deps.metrics_collector
    assert server.auth_provider is deps.auth_provider
    assert server.lifecycle_guard is deps.lifecycle_guard
    assert server.permission_seeder is deps.permission_seeder
    assert server.module_config_provider is deps.module_config_provider
    assert server.policy_context_provider is deps.policy_context_provider


def test_mcp_server_registers_shutdown_family_through_injected_lifecycle_guard() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _fake_runtime_dependencies()
    MCPServer(dependencies=deps)

    assert deps.lifecycle_guard.registered
    registration = deps.lifecycle_guard.registered[-1]
    assert registration["family"] == "mcp.websocket"
    assert callable(registration["active_count"])
    assert callable(registration["drain"])


@pytest.mark.asyncio
async def test_mcp_server_initialize_uses_injected_permission_seeder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_MODULES_CONFIG", "/tmp/mcp-stage3-no-modules.yaml")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.server.validate_config",
        lambda: True,
    )
    deps = _server_runtime_dependencies()
    server = MCPServer(dependencies=deps)
    server.config.metrics_enabled = False

    await server.initialize()

    assert deps.permission_seeder.calls == 1


@pytest.mark.asyncio
async def test_mcp_server_default_media_module_path_uses_injected_module_config_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "1")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_MODULES_CONFIG", "/tmp/mcp-stage3-no-modules.yaml")
    deps = _server_runtime_dependencies()
    server = MCPServer(dependencies=deps)

    await server._register_default_modules()  # noqa: SLF001

    media_registration = next(
        row for row in deps.module_registry.registrations if row["module_id"] == "media"
    )
    assert media_registration["config"].settings["db_path"] == (
        deps.module_config_provider.media_db_path
    )


def test_mcp_server_uses_injected_auth_and_policy_context_helpers() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _server_runtime_dependencies()
    server = MCPServer(dependencies=deps)

    assert server._extract_api_key_permissions({"scopes": "ignored"}) == ["fake.scope"]  # noqa: SLF001
    assert server._policy_context_enabled() is False  # noqa: SLF001


def test_default_server_protocol_uses_current_telemetry_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import protocol as protocol_mod
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    first_manager = object()
    second_manager = object()
    deps = _fake_runtime_dependencies()
    deps.telemetry_provider = first_manager
    current = {"manager": first_manager}

    monkeypatch.setattr(protocol_mod, "build_default_runtime_dependencies", lambda: deps)
    monkeypatch.setattr(protocol_mod, "get_telemetry_manager", lambda: current["manager"])

    server = MCPServer()

    assert server.dependencies is deps
    assert server.protocol.telemetry is first_manager
    current["manager"] = second_manager
    assert server.protocol.telemetry is second_manager


def test_protocol_instances_do_not_share_prepared_call_secrets() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    first = MCPProtocol()
    second = MCPProtocol()
    assert first is not second
    assert first._prepared_call_secret != second._prepared_call_secret  # noqa: SLF001
