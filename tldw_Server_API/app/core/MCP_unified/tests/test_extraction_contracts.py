from __future__ import annotations

import ast
import contextlib
import inspect
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
STANDALONE_MCP_ROOT = REPO_ROOT / "mcp_unified"
MCP_PACKAGE = "tldw_Server_API.app.core.MCP_unified"
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
        tool_catalog_provider=object(),
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
        environment_flags_provider=_FakeEnvironmentFlagsProvider(),
        websocket_stream_factory=_RecordingWebSocketStreamFactory(),
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


class _FakeEnvironmentFlagsProvider:
    """Environment flag provider double with caller-controlled results."""

    def __init__(
        self,
        *,
        flags: dict[str, bool] | None = None,
        test_mode: bool = False,
        explicit_pytest_runtime: bool = False,
        truthy_values: dict[Any, bool] | None = None,
    ) -> None:
        self.flags = flags or {}
        self.test_mode = test_mode
        self.explicit_pytest_runtime = explicit_pytest_runtime
        self.truthy_values = truthy_values or {}
        self.flag_calls: list[str] = []
        self.truthy_calls: list[Any] = []

    def env_flag_enabled(self, name: str) -> bool:
        self.flag_calls.append(name)
        return self.flags.get(name, False)

    def is_test_mode(self) -> bool:
        return self.test_mode

    def is_explicit_pytest_runtime(self) -> bool:
        return self.explicit_pytest_runtime

    def is_truthy(self, value: Any) -> bool:
        self.truthy_calls.append(value)
        return self.truthy_values.get(value, False)


class _FakeWebSocketStream:
    """Sentinel stream object returned by the injected WebSocket stream factory."""

    def __init__(self) -> None:
        self.ws = SimpleNamespace(close=lambda *args, **kwargs: None)
        self.sent: list[dict[str, Any]] = []
        self.started = False
        self.stopped = False
        self.activity_marks = 0

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def mark_activity(self) -> None:
        self.activity_marks += 1

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class _RecordingWebSocketStreamFactory:
    """WebSocket stream factory double that records construction calls."""

    def __init__(self) -> None:
        self.stream = _FakeWebSocketStream()
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        websocket: Any,
        *,
        heartbeat_interval_s: float | None,
        idle_timeout_s: float | None,
        close_on_done: bool,
        labels: dict[str, str],
    ) -> _FakeWebSocketStream:
        self.calls.append(
            {
                "websocket": websocket,
                "heartbeat_interval_s": heartbeat_interval_s,
                "idle_timeout_s": idle_timeout_s,
                "close_on_done": close_on_done,
                "labels": labels,
            }
        )
        return self.stream


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


class _TelemetryManagerDouble:
    """Telemetry manager double that exposes trace_context calls."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.trace_calls: list[tuple[str, dict[str, Any] | None]] = []

    def trace_context(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        self.trace_calls.append((operation_name, attributes))
        return contextlib.nullcontext(self.label)


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


def _resolve_import_from_source(package: str, module: str | None, level: int) -> str:
    """Return an absolute module path for an ``ast.ImportFrom`` source."""
    if level == 0:
        return module or ""

    package_parts = package.split(".")
    base_parts = package_parts[: len(package_parts) - level + 1]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def _resolved_import_sources_for(
    path: Path,
    package: str,
    *,
    top_level_only: bool = False,
) -> list[str]:
    """Return import module sources from Python AST, resolving relative imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = tree.body if top_level_only else ast.walk(tree)
    imports: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(_resolve_import_from_source(package, node.module, node.level))
    return imports


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


def test_protocol_uses_runtime_dependencies_for_stage3b_host_services() -> None:
    forbidden_imports = {
        "tldw_Server_API.app.core.Infrastructure.redis_factory",
        "tldw_Server_API.app.core.Metrics.telemetry",
        "tldw_Server_API.app.core.testing",
    }

    imports = _resolved_import_sources_for(MCP_ROOT / "protocol.py", MCP_PACKAGE)
    offenders = sorted(
        source
        for source in imports
        if source in forbidden_imports
        or any(source.startswith(f"{forbidden}.") for forbidden in forbidden_imports)
    )

    assert offenders == []


def test_protocol_catalog_lookup_uses_runtime_dependencies() -> None:
    forbidden_import = "tldw_Server_API.app.core.AuthNZ.database"

    imports = _resolved_import_sources_for(MCP_ROOT / "protocol.py", MCP_PACKAGE)
    offenders = sorted(
        source
        for source in imports
        if source == forbidden_import or source.startswith(f"{forbidden_import}.")
    )

    assert offenders == []


def test_tldw_runtime_catalog_provider_does_not_inline_tool_catalog_sql() -> None:
    adapter = MCP_ROOT / "adapters" / "tldw_runtime.py"
    tree = ast.parse(adapter.read_text(encoding="utf-8"), filename=str(adapter))
    table_sql_constants = sorted(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (
            "tool_catalogs" in node.value
            or "tool_catalog_entries" in node.value
        )
    )

    assert table_sql_constants == []


def test_server_uses_runtime_dependencies_for_stage3c_host_services() -> None:
    forbidden_imports = {
        "tldw_Server_API.app.core.AuthNZ.exceptions",
        "tldw_Server_API.app.core.Streaming.streams",
        "tldw_Server_API.app.core.testing",
    }

    imports = _resolved_import_sources_for(MCP_ROOT / "server.py", MCP_PACKAGE)
    offenders = sorted(
        source
        for source in imports
        if source in forbidden_imports
        or any(source.startswith(f"{forbidden}.") for forbidden in forbidden_imports)
    )

    assert offenders == []


def test_security_and_config_use_package_local_environment_helpers() -> None:
    forbidden_import = "tldw_Server_API.app.core.testing"
    targets = {
        MCP_ROOT / "config.py": MCP_PACKAGE,
        MCP_ROOT / "security" / "ip_filter.py": f"{MCP_PACKAGE}.security",
        MCP_ROOT / "security" / "request_guards.py": f"{MCP_PACKAGE}.security",
    }

    offenders: dict[str, list[str]] = {}
    for path, package in targets.items():
        imports = _resolved_import_sources_for(path, package)
        blocked = sorted(
            source
            for source in imports
            if source == forbidden_import or source.startswith(f"{forbidden_import}.")
        )
        if blocked:
            offenders[str(path.relative_to(MCP_ROOT))] = blocked

    assert offenders == {}


def test_mcp_discovery_module_uses_package_local_environment_helpers() -> None:
    forbidden_import = "tldw_Server_API.app.core.testing"
    discovery_module = MCP_ROOT / "modules" / "implementations" / "mcp_discovery_module.py"

    imports = _resolved_import_sources_for(
        discovery_module,
        f"{MCP_PACKAGE}.modules.implementations",
    )
    offenders = sorted(
        source
        for source in imports
        if source == forbidden_import or source.startswith(f"{forbidden_import}.")
    )

    assert offenders == []


def test_base_module_does_not_import_host_circuit_breaker() -> None:
    forbidden_import = "tldw_Server_API.app.core.Infrastructure.circuit_breaker"
    imports = _resolved_import_sources_for(
        MCP_ROOT / "modules" / "base.py",
        f"{MCP_PACKAGE}.modules",
    )
    offenders = sorted(
        source
        for source in imports
        if source == forbidden_import or source.startswith(f"{forbidden_import}.")
    )

    assert offenders == []


def test_catalog_loader_uses_standalone_catalog_schema() -> None:
    """Catalog loading must depend on standalone MCP package schemas."""
    forbidden_import = "tldw_Server_API.app.api.v1.schemas.archetype_schemas"
    imports = _resolved_import_sources_for(MCP_ROOT / "catalog_loader.py", MCP_PACKAGE)
    offenders = sorted(
        source
        for source in imports
        if source == forbidden_import or source.startswith(f"{forbidden_import}.")
    )

    assert offenders == []


def test_catalog_loader_delegates_to_standalone_package_loader() -> None:
    """Host catalog loader should be a thin wrapper around package loader code."""
    imports = _resolved_import_sources_for(
        MCP_ROOT / "catalog_loader.py",
        MCP_PACKAGE,
        top_level_only=True,
    )
    forbidden_sources = {"yaml", "loguru", "pydantic"}
    offenders = sorted(
        source
        for source in imports
        if source in forbidden_sources
        or any(source.startswith(f"{forbidden}.") for forbidden in forbidden_sources)
    )

    assert "mcp_unified.federation.catalog_loader" in imports
    assert offenders == []


def test_external_server_config_schema_delegates_to_standalone_package_schema() -> None:
    """Host external registry config schema should be a package-wrapper seam."""
    imports = _resolved_import_sources_for(
        MCP_ROOT / "external_servers" / "config_schema.py",
        f"{MCP_PACKAGE}.external_servers",
        top_level_only=True,
    )
    forbidden_sources = {"fnmatch", "json", "loguru", "pydantic", "yaml"}
    offenders = sorted(
        source
        for source in imports
        if source in forbidden_sources
        or any(source.startswith(f"{forbidden}.") for forbidden in forbidden_sources)
    )

    assert "mcp_unified.federation.config_schema" in imports
    assert offenders == []


def test_standalone_catalog_loader_uses_loguru_not_stdlib_logging() -> None:
    """Package catalog loader should use the project-standard Loguru logger."""
    imports = _resolved_import_sources_for(
        STANDALONE_MCP_ROOT / "federation" / "catalog_loader.py",
        "mcp_unified.federation",
        top_level_only=True,
    )

    assert "loguru" in imports
    assert "logging" not in imports


def test_protocol_boundary_scan_resolves_relative_imports(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from ..testing import is_truthy\n"
        "from .adapters.tldw_runtime import build_default_runtime_dependencies\n",
        encoding="utf-8",
    )

    imports = _resolved_import_sources_for(sample, MCP_PACKAGE)

    assert imports == [
        "tldw_Server_API.app.core.testing",
        "tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime",
    ]


def test_protocol_import_time_boundary_does_not_load_tldw_runtime_adapter() -> None:
    imports = _resolved_import_sources_for(
        MCP_ROOT / "protocol.py",
        MCP_PACKAGE,
        top_level_only=True,
    )

    assert "tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime" not in imports


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


@pytest.mark.asyncio
async def test_tldw_auth_provider_fails_closed_when_authnz_ws_verify_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime

    async def _reject_token(*_args: Any, **_kwargs: Any) -> None:
        raise InvalidTokenError("bad token")

    monkeypatch.setattr(tldw_runtime, "get_jwt_manager", lambda: object())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.verify_jwt_and_fetch_user",
        _reject_token,
    )

    provider = tldw_runtime.TldwServerAuthProvider()
    websocket = SimpleNamespace(headers={}, client=None)

    assert (
        await provider.authenticate_authnz_websocket_token(
            "token",
            websocket=websocket,
        )
        is None
    )


def test_stage3_runtime_contracts_are_exported_by_interface_packages() -> None:
    import mcp_unified.interfaces as standalone_interfaces

    import tldw_Server_API.app.core.MCP_unified.interfaces as compat_interfaces

    names = [
        "AuthenticatedIdentity",
        "LifecycleGuard",
        "NoopToolUseRecorder",
        "ModuleConfigProvider",
        "PermissionSeeder",
        "PolicyContextProvider",
        "ServerAuthProvider",
        "ToolCatalogProvider",
        "ToolUseRecorder",
        "EnvironmentFlagsProvider",
        "WebSocketStream",
        "WebSocketStreamFactory",
    ]

    for name in names:
        assert getattr(standalone_interfaces, name) is getattr(compat_interfaces, name)


def test_mcp_protocol_accepts_runtime_dependencies() -> None:
    from mcp_unified.tool_use_reporting.recorder import NoopToolUseRecorder

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    deps = _fake_runtime_dependencies()
    protocol = MCPProtocol(dependencies=deps)
    assert protocol.module_registry is deps.module_registry
    assert protocol.rbac_policy is deps.rbac_policy
    assert isinstance(protocol._tool_use_recorder, NoopToolUseRecorder)


def test_mcp_protocol_treats_none_tool_use_recorder_as_unconfigured() -> None:
    from mcp_unified.tool_use_reporting.recorder import NoopToolUseRecorder

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    deps = _fake_runtime_dependencies()
    deps.tool_use_recorder = None

    protocol = MCPProtocol(dependencies=deps)

    assert isinstance(protocol._tool_use_recorder, NoopToolUseRecorder)


def test_runtime_dependencies_default_to_noop_tool_use_recorder() -> None:
    from mcp_unified.interfaces.runtime import MCPRuntimeDependencies
    from mcp_unified.tool_use_reporting.recorder import NoopToolUseRecorder

    deps = _fake_runtime_dependencies()
    dependency_kwargs = vars(deps).copy()
    runtime_dependencies = MCPRuntimeDependencies(**dependency_kwargs)

    assert isinstance(runtime_dependencies.tool_use_recorder, NoopToolUseRecorder)


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
    assert server.environment_flags_provider is deps.environment_flags_provider
    assert server.websocket_stream_factory is deps.websocket_stream_factory


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
    deps.environment_flags_provider = _FakeEnvironmentFlagsProvider(
        flags={"MCP_ENABLE_MEDIA_MODULE": True}
    )
    server = MCPServer(dependencies=deps)

    await server._register_default_modules()  # noqa: SLF001

    media_registration = next(
        row for row in deps.module_registry.registrations if row["module_id"] == "media"
    )
    assert media_registration["config"].settings["db_path"] == (
        deps.module_config_provider.media_db_path
    )


@pytest.mark.asyncio
async def test_mcp_server_default_module_configs_use_injected_circuit_breaker_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "1")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_MODULES_CONFIG", "/tmp/mcp-stage3-no-modules.yaml")
    deps = _server_runtime_dependencies()
    deps.environment_flags_provider = _FakeEnvironmentFlagsProvider(
        flags={"MCP_ENABLE_MEDIA_MODULE": True}
    )
    server = MCPServer(dependencies=deps)

    await server._register_default_modules()  # noqa: SLF001

    assert deps.module_registry.registrations
    for registration in deps.module_registry.registrations:
        assert registration["config"].circuit_breaker_factory is deps.circuit_breaker_factory


def test_mcp_server_uses_injected_auth_and_policy_context_helpers() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _server_runtime_dependencies()
    server = MCPServer(dependencies=deps)

    assert server._extract_api_key_permissions({"scopes": "ignored"}) == ["fake.scope"]  # noqa: SLF001
    assert server._policy_context_enabled() is False  # noqa: SLF001


def test_websocket_stream_factory_declares_returned_stream_contract() -> None:
    from mcp_unified.interfaces import runtime as standalone_runtime

    hints = get_type_hints(standalone_runtime.WebSocketStreamFactory.__call__)

    assert hints["return"] is standalone_runtime.WebSocketStream
    assert hasattr(standalone_runtime.WebSocketStream, "start")
    assert hasattr(standalone_runtime.WebSocketStream, "stop")
    assert hasattr(standalone_runtime.WebSocketStream, "mark_activity")
    assert hasattr(standalone_runtime.WebSocketStream, "send_json")


def test_mcp_server_websocket_stream_annotations_use_runtime_contract() -> None:
    from tldw_Server_API.app.core.MCP_unified.interfaces import WebSocketStream
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    create_hints = get_type_hints(MCPServer._create_websocket_stream)
    handle_hints = get_type_hints(MCPServer._handle_websocket_messages)

    assert create_hints["return"] is WebSocketStream
    assert handle_hints["stream"] is WebSocketStream
    assert handle_hints["return"] is type(None)


def test_mcp_server_uses_injected_environment_flags_provider() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _server_runtime_dependencies()
    deps.environment_flags_provider = _FakeEnvironmentFlagsProvider(
        flags={"MCP_ENABLE_DEMO_AUTH": True},
        test_mode=True,
        explicit_pytest_runtime=True,
        truthy_values={"yes": True},
    )
    server = MCPServer(dependencies=deps)

    assert server._env_flag_enabled("MCP_ENABLE_DEMO_AUTH") is True  # noqa: SLF001
    assert server._is_test_mode() is True  # noqa: SLF001
    assert server._is_explicit_pytest_runtime() is True  # noqa: SLF001
    assert server._is_truthy("yes") is True  # noqa: SLF001


def test_mcp_server_uses_injected_websocket_stream_factory() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _server_runtime_dependencies()
    factory = _RecordingWebSocketStreamFactory()
    deps.websocket_stream_factory = factory
    server = MCPServer(dependencies=deps)
    server.config.ws_ping_interval = 7
    server.config.ws_idle_timeout_seconds = 11
    websocket = object()

    stream = server._create_websocket_stream(websocket)  # noqa: SLF001

    assert stream is factory.stream
    assert factory.calls == [
        {
            "websocket": websocket,
            "heartbeat_interval_s": 7.0,
            "idle_timeout_s": 11.0,
            "close_on_done": True,
            "labels": {"component": "mcp", "endpoint": "mcp_ws"},
        }
    ]


def test_mcp_server_logs_host_adapter_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import server as server_mod
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    class _FailingAuthProvider(_FakeAuthProvider):
        def normalize_api_key_permissions(self, info: dict[str, Any] | None) -> list[str]:
            del info
            raise RuntimeError("scope failure")

    class _FailingPolicyContextProvider:
        def is_policy_context_enabled(self) -> bool:
            raise RuntimeError("policy failure")

    class _FailingModuleConfigProvider:
        def default_media_db_path(self) -> str:
            raise RuntimeError("path failure")

    warnings: list[str] = []

    def _record_warning(message: str, *args: Any, **_kwargs: Any) -> None:
        warnings.append(message.format(*args))

    monkeypatch.setattr(server_mod.logger, "warning", _record_warning)
    deps = _server_runtime_dependencies()
    deps.auth_provider = _FailingAuthProvider()
    deps.policy_context_provider = _FailingPolicyContextProvider()
    deps.module_config_provider = _FailingModuleConfigProvider()
    server = MCPServer(dependencies=deps)

    assert server._extract_api_key_permissions({"scopes": "ignored"}) == []  # noqa: SLF001
    assert server._policy_context_enabled() is False  # noqa: SLF001
    assert server._default_media_db_path() == ""  # noqa: SLF001
    assert warnings == [
        "MCP API-key permission normalization failed: scope failure",
        "MCP policy-context flag resolution failed: policy failure",
        "MCP default media DB path resolution failed: path failure",
    ]


@pytest.mark.asyncio
async def test_tldw_permission_seeder_uses_acquired_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime

    class _AcquireContext:
        def __init__(self, conn: object) -> None:
            self.conn = conn

        async def __aenter__(self) -> object:
            return self.conn

        async def __aexit__(self, *_exc_info: Any) -> None:
            return None

    class _Pool:
        def __init__(self) -> None:
            self.conn = object()

        def acquire(self) -> _AcquireContext:
            return _AcquireContext(self.conn)

    pool = _Pool()
    ensure_calls: list[tuple[object, str, str, str]] = []

    async def _get_db_pool() -> _Pool:
        return pool

    async def _ensure_permission(
        db: object,
        name: str,
        description: str,
        *,
        category: str,
    ) -> dict[str, Any]:
        ensure_calls.append((db, name, description, category))
        return {"id": 1, "name": name, "description": description, "category": category}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _get_db_pool,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_roles_permissions_service.ensure_permission",
        _ensure_permission,
    )

    await tldw_runtime.TldwPermissionSeeder().seed_default_tool_permissions()

    assert ensure_calls == [
        (pool.conn, "tools.execute:*", "Wildcard tool execution", "tools")
    ]


def test_authnz_access_token_helper_documents_boolean_semantics() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import _is_authnz_access_token

    doc = inspect.getdoc(_is_authnz_access_token)

    assert doc is not None
    assert "Return True" in doc
    assert "MCP AuthNZ access token" in doc


def test_default_server_protocol_uses_current_telemetry_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    first_manager = _TelemetryManagerDouble("first")
    second_manager = _TelemetryManagerDouble("second")
    current = {"manager": first_manager}

    monkeypatch.setattr(tldw_runtime, "get_telemetry_manager", lambda: current["manager"])

    server = MCPServer()

    with server.protocol.telemetry.trace_context("first-op", {"generation": 1}) as span:
        assert span == "first"
    current["manager"] = second_manager
    with server.protocol.telemetry.trace_context("second-op", {"generation": 2}) as span:
        assert span == "second"
    assert first_manager.trace_calls == [("first-op", {"generation": 1})]
    assert second_manager.trace_calls == [("second-op", {"generation": 2})]


def test_protocol_instances_do_not_share_prepared_call_secrets() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    first = MCPProtocol()
    second = MCPProtocol()
    assert first is not second
    assert first._prepared_call_secret != second._prepared_call_secret  # noqa: SLF001
