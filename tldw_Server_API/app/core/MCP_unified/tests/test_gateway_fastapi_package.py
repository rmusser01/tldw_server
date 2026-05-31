from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mcp_unified.gateway.fastapi as gateway_fastapi
import mcp_unified.gateway.jsonrpc as gateway_jsonrpc
import pytest
from fastapi.testclient import TestClient
from mcp_unified.gateway import create_gateway_app

REPO_ROOT = Path(__file__).resolve().parents[5]
GATEWAY_ROOT = REPO_ROOT / "mcp_unified" / "gateway"


def _import_sources(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


class _FakeGatewayRuntime:
    name = "unit-gateway"
    version = "0.0-test"

    def __init__(self) -> None:
        self.list_contexts: list[Any] = []
        self.call_requests: list[tuple[str, dict[str, Any], Any]] = []
        self.resource_list_contexts: list[Any] = []
        self.resource_read_requests: list[tuple[str, Any]] = []
        self.prompt_list_contexts: list[Any] = []
        self.prompt_get_requests: list[tuple[str, dict[str, Any], Any]] = []
        self.module_list_contexts: list[Any] = []
        self.module_health_contexts: list[Any] = []

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        self.list_contexts.append(context)
        return [
            {
                "name": "echo.search",
                "description": "Echo a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "metadata": {"category": "test"},
            }
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        self.call_requests.append((name, arguments, context))
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"{name}:{arguments['query']}",
                }
            ]
        }

    async def list_resources(self, context: Any) -> list[dict[str, Any]]:
        self.resource_list_contexts.append(context)
        return [
            {
                "uri": "resource://unit/doc",
                "name": "Unit Doc",
                "mimeType": "text/plain",
            }
        ]

    async def read_resource(self, uri: str, context: Any) -> dict[str, Any]:
        self.resource_read_requests.append((uri, context))
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": "hello resource",
                }
            ]
        }

    async def list_prompts(self, context: Any) -> list[dict[str, Any]]:
        self.prompt_list_contexts.append(context)
        return [
            {
                "name": "review.prompt",
                "description": "Review a focused topic.",
            }
        ]

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        self.prompt_get_requests.append((name, arguments, context))
        topic = arguments.get("topic", "")
        return {
            "description": "Review a focused topic.",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"{name}:{topic}",
                    },
                }
            ],
        }

    async def list_modules(self, context: Any) -> list[dict[str, Any]]:
        self.module_list_contexts.append(context)
        return [{"module_id": "unit", "name": "Unit Module"}]

    async def get_modules_health(self, context: Any) -> dict[str, Any]:
        self.module_health_contexts.append(context)
        return {
            "unit": {
                "status": "healthy",
                "message": "ok",
                "checks": {},
                "last_check": None,
            }
        }


class _CustomExplodingGatewayRuntime(_FakeGatewayRuntime):
    class RuntimeBackendError(Exception):
        pass

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        raise self.RuntimeBackendError("backend unavailable")


class _MultiToolGatewayRuntime(_FakeGatewayRuntime):
    """Fake runtime that advertises multiple tools for profile filtering tests."""

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        """Return tools with distinct capabilities so profiles can filter them."""

        self.list_contexts.append(context)
        return [
            {
                "name": "echo.search",
                "description": "Echo a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "metadata": {"category": "test", "capability": "code_search"},
            },
            {
                "name": "admin.delete",
                "description": "Delete an admin resource.",
                "inputSchema": {"type": "object", "properties": {}},
                "metadata": {"category": "test", "capability": "admin.delete"},
            },
        ]


class _CustomToolListGatewayRuntime(_FakeGatewayRuntime):
    """Fake runtime that returns caller-supplied tool discovery payloads."""

    def __init__(self, tools: Any) -> None:
        super().__init__()
        self._tools = tools

    async def list_tools(self, context: Any) -> Any:
        """Return the configured discovery payload without normalizing it."""

        self.list_contexts.append(context)
        return self._tools


class _FakeLogger:
    def __init__(self) -> None:
        self.opt_calls: list[dict[str, Any]] = []
        self.error_calls: list[tuple[str, tuple[Any, ...]]] = []

    def opt(self, **kwargs: Any) -> _FakeLogger:
        self.opt_calls.append(kwargs)
        return self

    def error(self, message: str, *args: Any) -> None:
        self.error_calls.append((message, args))


def _assert_jsonrpc_error(
    body: dict[str, Any],
    *,
    code: int,
    request_id: Any,
) -> None:
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == request_id
    assert body["error"]["code"] == code
    assert "message" in body["error"]


def _profile_with_allowed_tools(profile_id: str, allowed_tools: list[str]) -> Any:
    """Build a profile that allows only the supplied explicit tool names."""

    from mcp_unified.profiles.models import MCPProfile, ProfilePolicy

    return MCPProfile(
        id=profile_id,
        name=f"Profile {profile_id}",
        policy_document=ProfilePolicy(allowed_tools=allowed_tools),
    )


def _profile_with_capabilities(profile_id: str, capabilities: list[str]) -> Any:
    """Build a profile that allows tools by advertised capability metadata."""

    from mcp_unified.profiles.models import MCPProfile, ProfilePolicy

    return MCPProfile(
        id=profile_id,
        name=f"Profile {profile_id}",
        policy_document=ProfilePolicy(capabilities=capabilities),
    )


def test_gateway_package_does_not_import_tldw_server_api() -> None:
    assert GATEWAY_ROOT.exists()
    offenders: dict[str, list[str]] = {}
    for path in GATEWAY_ROOT.rglob("*.py"):
        blocked = sorted(
            source
            for source in _import_sources(path)
            if source == "tldw_Server_API" or source.startswith("tldw_Server_API.")
        )
        if blocked:
            offenders[str(path.relative_to(REPO_ROOT))] = blocked

    assert offenders == {}


def test_gateway_stdio_submodule_import_does_not_eagerly_import_fastapi_transport() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mcp_unified.gateway.stdio; print('mcp_unified.gateway.fastapi' in sys.modules)",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_gateway_profile_runtime_requires_profile_for_tool_execution() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _MultiToolGatewayRuntime()
    runtime = ProfileAwareGatewayRuntime(backend, profile_store=InMemoryProfileStore())
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-profile"},
        )
        called = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "echo.search", "arguments": {"query": "hello"}},
                "id": "call-profile",
            },
        )

    assert listed.status_code == 200
    assert listed.json()["result"]["tools"] == []
    assert called.status_code == 200
    body = called.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="call-profile")
    assert body["error"]["data"]["reason_code"] == "profile_required"
    assert backend.call_requests == []


def test_gateway_profile_runtime_filters_and_allows_default_profile_tools() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _MultiToolGatewayRuntime()
    profile = _profile_with_allowed_tools("reviewer", ["echo.search"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-default"},
        )
        allowed = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "echo.search", "arguments": {"query": "hello"}},
                "id": "call-allowed",
            },
        )
        denied = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "admin.delete", "arguments": {}},
                "id": "call-denied",
            },
        )

    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["echo.search"]
    assert allowed.json()["result"]["content"][0]["text"] == "echo.search:hello"
    assert backend.call_requests[-1][0] == "echo.search"
    denied_body = denied.json()
    _assert_jsonrpc_error(denied_body, code=-32001, request_id="call-denied")
    assert denied_body["error"]["data"]["reason_code"] == "tool_not_allowed"
    assert all(call[0] != "admin.delete" for call in backend.call_requests)


def test_gateway_profile_runtime_denies_explicit_tool_before_backend_discovery() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomExplodingGatewayRuntime()
    profile = _profile_with_allowed_tools("reviewer", ["echo.search"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "admin.delete", "arguments": {}},
                "id": "call-denied-without-discovery",
            },
        )

    denied_body = denied.json()
    _assert_jsonrpc_error(
        denied_body,
        code=-32001,
        request_id="call-denied-without-discovery",
    )
    assert denied_body["error"]["data"]["reason_code"] == "tool_not_allowed"
    assert backend.call_requests == []


def test_gateway_profile_runtime_fails_closed_when_capability_discovery_fails() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomExplodingGatewayRuntime()
    profile = _profile_with_capabilities("researcher", ["code_search"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="researcher",
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "echo.search", "arguments": {"query": "hello"}},
                "id": "capability-discovery-failed",
            },
        )

    denied_body = denied.json()
    _assert_jsonrpc_error(
        denied_body,
        code=-32001,
        request_id="capability-discovery-failed",
    )
    assert denied_body["error"]["data"]["reason_code"] == "tool_metadata_unavailable"
    assert backend.call_requests == []


def test_gateway_profile_runtime_ignores_invalid_backend_tool_descriptors() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            None,
            "not-a-tool",
            {"name": "  "},
            {"name": "echo.search", "metadata": {"capability": "code_search"}},
        ]
    )
    profile = _profile_with_capabilities("researcher", ["code_search"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="researcher",
    )

    tools = asyncio.run(runtime.list_tools(GatewayRequestContext(request_id="invalid-tools")))

    assert [tool["name"] for tool in tools] == ["echo.search"]


def test_gateway_profile_runtime_treats_non_list_backend_tools_as_empty() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(None)
    profile = _profile_with_allowed_tools("reviewer", ["echo.search"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
    )

    assert asyncio.run(runtime.list_tools(GatewayRequestContext(request_id="non-list-tools"))) == []


def test_gateway_profile_bootstrap_seeds_default_builtin_preset_profile() -> None:
    from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway

    backend = _MultiToolGatewayRuntime()
    bootstrap = asyncio.run(
        bootstrap_profile_gateway(
            backend,
            default_preset_id="project-researcher",
        )
    )
    app = create_gateway_app(bootstrap.runtime, prefix="/mcp")

    with TestClient(app) as client:
        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-bootstrap"},
        )
        allowed = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "echo.search", "arguments": {"query": "bootstrap"}},
                "id": "call-bootstrap",
            },
        )

    assert bootstrap.default_profile_id == "project-researcher"
    assert bootstrap.seeded_profile_ids == ("project-researcher",)
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["echo.search"]
    assert allowed.json()["result"]["content"][0]["text"] == "echo.search:bootstrap"


def test_gateway_profile_bootstrap_uses_caller_profiles_as_default() -> None:
    from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway

    backend = _MultiToolGatewayRuntime()
    profile = _profile_with_allowed_tools("reviewer", ["echo.search"])

    bootstrap = asyncio.run(
        bootstrap_profile_gateway(
            backend,
            profiles=[profile],
            default_profile_id="reviewer",
        )
    )
    stored_profiles = asyncio.run(bootstrap.profile_store.list_profiles())
    app = create_gateway_app(bootstrap.runtime, prefix="/mcp")

    with TestClient(app) as client:
        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-caller-default"},
        )

    assert bootstrap.default_profile_id == "reviewer"
    assert bootstrap.seeded_profile_ids == ()
    assert [profile.id for profile in stored_profiles] == ["reviewer"]
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["echo.search"]


def test_gateway_profile_bootstrap_keeps_explicit_default_when_seeding_preset() -> None:
    from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway

    backend = _MultiToolGatewayRuntime()
    profile = _profile_with_allowed_tools("reviewer", ["admin.delete"])

    bootstrap = asyncio.run(
        bootstrap_profile_gateway(
            backend,
            profiles=[profile],
            default_profile_id="reviewer",
            default_preset_id="project-researcher",
        )
    )
    stored_profiles = asyncio.run(bootstrap.profile_store.list_profiles())
    app = create_gateway_app(bootstrap.runtime, prefix="/mcp")

    with TestClient(app) as client:
        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-caller-plus-preset"},
        )

    assert bootstrap.default_profile_id == "reviewer"
    assert bootstrap.seeded_profile_ids == ("project-researcher",)
    assert {profile.id for profile in stored_profiles} == {"reviewer", "project-researcher"}
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["admin.delete"]


def test_gateway_profile_bootstrap_rejects_existing_preset_profile_collision() -> None:
    from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway

    profile = _profile_with_allowed_tools("project-researcher", ["admin.delete"])

    with pytest.raises(ValueError, match="profile id 'project-researcher' already exists"):
        asyncio.run(
            bootstrap_profile_gateway(
                _MultiToolGatewayRuntime(),
                profiles=[profile],
                default_preset_id="project-researcher",
            )
        )


def test_gateway_profile_bootstrap_rejects_unknown_default_preset() -> None:
    from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway

    with pytest.raises(ValueError, match="Unknown MCP profile preset"):
        asyncio.run(
            bootstrap_profile_gateway(
                _MultiToolGatewayRuntime(),
                default_preset_id="not-a-preset",
            )
        )


def test_gateway_config_bootstrap_uses_memory_store_default_preset() -> None:
    """Build a profile runtime from default memory-store config and preset."""

    from mcp_unified.gateway.config import (
        GatewayProfileBootstrapConfig,
        bootstrap_profile_gateway_from_config,
    )

    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(default_preset_id="project-researcher"),
        )
    )
    app = create_gateway_app(bootstrap.runtime, prefix="/mcp")

    with TestClient(app) as client:
        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-config-memory"},
        )

    assert bootstrap.default_profile_id == "project-researcher"
    assert bootstrap.seeded_profile_ids == ("project-researcher",)
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["echo.search"]


def test_gateway_config_bootstrap_uses_sqlite_profile_store(tmp_path: Path) -> None:
    """Build a profile runtime backed by the configured SQLite profile store."""

    from mcp_unified.gateway.config import (
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.storage.sqlite import SQLiteMCPStore

    sqlite_path = tmp_path / "gateway.db"

    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                default_preset_id="project-researcher",
            ),
        )
    )
    stored_profiles = asyncio.run(bootstrap.profile_store.list_profiles())

    assert isinstance(bootstrap.profile_store, SQLiteMCPStore)
    assert sqlite_path.exists()
    assert [profile.id for profile in stored_profiles] == ["project-researcher"]
    asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_preserves_injected_profile_store(tmp_path: Path) -> None:
    """Prefer caller-injected stores over config-selected store creation."""

    from mcp_unified.gateway.config import (
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.profiles.store import InMemoryProfileStore

    sqlite_path = tmp_path / "unused.db"
    profile = _profile_with_allowed_tools("reviewer", ["admin.delete"])
    injected_store = InMemoryProfileStore([profile])

    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                default_profile_id="reviewer",
            ),
            profile_store=injected_store,
        )
    )
    app = create_gateway_app(bootstrap.runtime, prefix="/mcp")

    with TestClient(app) as client:
        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-config-injected"},
        )

    assert bootstrap.profile_store is injected_store
    assert not sqlite_path.exists()
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["admin.delete"]


def test_gateway_config_rejects_invalid_store_kind() -> None:
    """Reject unsupported profile-store kinds during config construction."""

    from mcp_unified.gateway.config import GatewayProfileStoreConfig

    with pytest.raises(ValueError, match="Unsupported gateway profile store kind"):
        GatewayProfileStoreConfig(kind="postgres")


def test_gateway_config_rejects_sqlite_store_without_path() -> None:
    """Reject SQLite profile-store config when no database path is supplied."""

    from mcp_unified.gateway.config import GatewayProfileStoreConfig

    with pytest.raises(ValueError, match="sqlite_path is required"):
        GatewayProfileStoreConfig(kind="sqlite")


@pytest.mark.parametrize("sqlite_path", ["", "   "])
def test_gateway_config_rejects_blank_sqlite_store_path(sqlite_path: str) -> None:
    """Reject blank SQLite profile-store paths before store construction."""

    from mcp_unified.gateway.config import GatewayProfileStoreConfig

    with pytest.raises(ValueError, match="sqlite_path cannot be empty"):
        GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path)


def test_gateway_config_bootstrap_copies_profile_mapping_inputs() -> None:
    """Copy profile mapping inputs so later caller mutation cannot affect policy."""

    from mcp_unified.gateway.config import (
        GatewayProfileBootstrapConfig,
        bootstrap_profile_gateway_from_config,
    )

    profile_payload = _profile_with_allowed_tools("reviewer", ["echo.search"]).model_dump(
        mode="json"
    )
    config = GatewayProfileBootstrapConfig(
        profiles=[profile_payload],
        default_profile_id="reviewer",
    )
    profile_payload["policy_document"]["allowed_tools"] = ["admin.delete"]

    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(_MultiToolGatewayRuntime(), config)
    )
    app = create_gateway_app(bootstrap.runtime, prefix="/mcp")

    with TestClient(app) as client:
        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-config-copy"},
        )

    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["echo.search"]


def test_gateway_config_loader_reads_json_and_bootstraps_default_preset(tmp_path: Path) -> None:
    """Load JSON gateway config and feed it into the config bootstrap helper."""

    from mcp_unified.gateway import load_gateway_profile_bootstrap_config
    from mcp_unified.gateway.config import bootstrap_profile_gateway_from_config

    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "store": {"kind": "memory"},
                "default_preset_id": "project-researcher",
            }
        ),
        encoding="utf-8",
    )

    config = load_gateway_profile_bootstrap_config(config_path)
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(_MultiToolGatewayRuntime(), config)
    )
    app = create_gateway_app(bootstrap.runtime, prefix="/mcp")

    with TestClient(app) as client:
        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-config-json"},
        )

    assert bootstrap.default_profile_id == "project-researcher"
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["echo.search"]


def test_gateway_config_loader_reads_toml_store_config(tmp_path: Path) -> None:
    """Load TOML gateway config into validated profile bootstrap config."""

    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        load_gateway_profile_bootstrap_config,
    )

    sqlite_path = tmp_path / "gateway.db"
    config_path = tmp_path / "gateway.toml"
    config_path.write_text(
        "\n".join(
            [
                'default_profile_id = "reviewer"',
                "",
                "[store]",
                'kind = "sqlite"',
                f'sqlite_path = "{sqlite_path}"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_gateway_profile_bootstrap_config(config_path)

    assert isinstance(config.store, GatewayProfileStoreConfig)
    assert config.store.kind == "sqlite"
    assert config.store.sqlite_path == str(sqlite_path)
    assert config.default_profile_id == "reviewer"


@pytest.mark.parametrize("suffix", [".yaml", ".txt"])
def test_gateway_config_loader_rejects_unsupported_suffix(tmp_path: Path, suffix: str) -> None:
    """Reject config files whose format cannot be inferred safely."""

    from mcp_unified.gateway.config import load_gateway_profile_bootstrap_config

    config_path = tmp_path / f"gateway{suffix}"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported gateway config format"):
        load_gateway_profile_bootstrap_config(config_path)


@pytest.mark.parametrize(
    ("suffix", "content", "message"),
    [
        (".json", "{", "Invalid gateway config JSON"),
        (".toml", "[store", "Invalid gateway config TOML"),
    ],
)
def test_gateway_config_loader_rejects_malformed_files(
    tmp_path: Path,
    suffix: str,
    content: str,
    message: str,
) -> None:
    """Reject malformed config files with parser-specific error context."""

    from mcp_unified.gateway.config import load_gateway_profile_bootstrap_config

    config_path = tmp_path / f"gateway{suffix}"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_gateway_profile_bootstrap_config(config_path)


def test_gateway_config_loader_json_parse_error_reports_location(tmp_path: Path) -> None:
    """Include JSON parser location details in malformed config errors."""

    from mcp_unified.gateway.config import load_gateway_profile_bootstrap_config

    config_path = tmp_path / "gateway.json"
    config_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_gateway_profile_bootstrap_config(config_path)

    assert "Invalid gateway config JSON" in str(exc_info.value)
    assert "line 1 column" in str(exc_info.value)


def test_gateway_config_loader_rejects_non_object_top_level_payload(tmp_path: Path) -> None:
    """Reject JSON config payloads that are not top-level objects."""

    from mcp_unified.gateway.config import load_gateway_profile_bootstrap_config

    config_path = tmp_path / "gateway.json"
    config_path.write_text(json.dumps([{"store": {"kind": "memory"}}]), encoding="utf-8")

    with pytest.raises(ValueError, match="Gateway config file must contain an object"):
        load_gateway_profile_bootstrap_config(config_path)


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": True},
        {"store": []},
    ],
)
def test_gateway_config_loader_wraps_schema_type_errors(tmp_path: Path, payload: dict[str, Any]) -> None:
    """Report config schema/type failures as deterministic ValueErrors."""

    from mcp_unified.gateway.config import load_gateway_profile_bootstrap_config

    config_path = tmp_path / "gateway.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid gateway config schema or types"):
        load_gateway_profile_bootstrap_config(config_path)


def test_gateway_transport_profile_selector_handles_missing_request_attributes() -> None:
    class _BareRequest:
        """Request double without FastAPI transport attributes."""

    assert gateway_fastapi._profile_id_from_transport(_BareRequest()) is None


def test_gateway_profile_runtime_selects_profile_from_http_header() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _MultiToolGatewayRuntime()
    profile = _profile_with_allowed_tools("reviewer", ["echo.search"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            headers={"X-MCP-Profile": "reviewer"},
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "echo.search", "arguments": {"query": "header"}},
                "id": "call-header-profile",
            },
        )

    assert response.status_code == 200
    assert response.json()["result"]["content"][0]["text"] == "echo.search:header"
    assert backend.call_requests[-1][2].metadata["profile_id"] == "reviewer"


def test_gateway_profile_runtime_selects_profile_from_websocket_query() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _MultiToolGatewayRuntime()
    profile = _profile_with_allowed_tools("reviewer", ["echo.search"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws?profile_id=reviewer") as websocket:
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "echo.search", "arguments": {"query": "ws"}},
                    "id": "call-ws-profile",
                }
            )
            body = websocket.receive_json()

    assert body["result"]["content"][0]["text"] == "echo.search:ws"
    assert backend.call_requests[-1][2].metadata["profile_id"] == "reviewer"


def test_gateway_fastapi_app_handles_basic_jsonrpc_flow() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        status = client.get("/mcp/status")
        assert status.status_code == 200
        assert status.json() == {
            "status": "ok",
            "name": "unit-gateway",
            "version": "0.0-test",
        }

        initialized = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"clientInfo": {"name": "pytest"}},
                "id": "init-1",
            },
        )
        assert initialized.status_code == 200
        assert initialized.json()["result"]["serverInfo"] == {
            "name": "unit-gateway",
            "version": "0.0-test",
        }
        capabilities = initialized.json()["result"]["capabilities"]
        assert capabilities["resources"]["available"] is True
        assert capabilities["prompts"]["available"] is True

        listed = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-1"},
        )
        assert listed.status_code == 200
        listed_body = listed.json()
        assert listed_body["id"] == "tools-1"
        assert listed_body["result"]["tools"][0]["name"] == "echo.search"
        assert runtime.list_contexts[-1].request_id == "tools-1"

        called = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "echo.search",
                    "arguments": {"query": "hello"},
                },
                "id": "call-1",
            },
        )
        assert called.status_code == 200
        called_body = called.json()
        assert called_body["id"] == "call-1"
        assert called_body["result"]["content"][0]["text"] == "echo.search:hello"
        assert runtime.call_requests[-1][0] == "echo.search"
        assert runtime.call_requests[-1][1] == {"query": "hello"}
        assert runtime.call_requests[-1][2].request_id == "call-1"


def test_gateway_request_rejects_malformed_json_with_jsonrpc_parse_error() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            content=b"{not valid json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32700, request_id=None)


def test_gateway_fastapi_app_handles_resource_prompt_and_module_methods() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        resources = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "resources/list", "params": {}, "id": "resources-1"},
        )
        assert resources.status_code == 200
        resources_body = resources.json()
        assert resources_body["id"] == "resources-1"
        assert resources_body["result"]["resources"][0]["uri"] == "resource://unit/doc"
        assert runtime.resource_list_contexts[-1].request_id == "resources-1"

        resource = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "resources/read",
                "params": {"uri": "resource://unit/doc"},
                "id": "read-1",
            },
        )
        assert resource.status_code == 200
        resource_body = resource.json()
        assert resource_body["id"] == "read-1"
        assert resource_body["result"]["contents"][0]["text"] == "hello resource"
        assert runtime.resource_read_requests[-1][0] == "resource://unit/doc"
        assert runtime.resource_read_requests[-1][1].request_id == "read-1"

        prompts = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "prompts/list", "params": {}, "id": "prompts-1"},
        )
        assert prompts.status_code == 200
        prompts_body = prompts.json()
        assert prompts_body["id"] == "prompts-1"
        assert prompts_body["result"]["prompts"][0]["name"] == "review.prompt"
        assert runtime.prompt_list_contexts[-1].request_id == "prompts-1"

        prompt = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {"name": "review.prompt", "arguments": {"topic": "gateway"}},
                "id": "prompt-1",
            },
        )
        assert prompt.status_code == 200
        prompt_body = prompt.json()
        assert prompt_body["id"] == "prompt-1"
        assert prompt_body["result"]["messages"][0]["content"]["text"] == "review.prompt:gateway"
        assert runtime.prompt_get_requests[-1][0] == "review.prompt"
        assert runtime.prompt_get_requests[-1][1] == {"topic": "gateway"}
        assert runtime.prompt_get_requests[-1][2].request_id == "prompt-1"

        modules = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "modules/list", "params": {}, "id": "modules-1"},
        )
        assert modules.status_code == 200
        modules_body = modules.json()
        assert modules_body["id"] == "modules-1"
        assert modules_body["result"]["modules"][0]["module_id"] == "unit"
        assert runtime.module_list_contexts[-1].request_id == "modules-1"

        health = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "modules/health", "params": {}, "id": "health-1"},
        )
        assert health.status_code == 200
        health_body = health.json()
        assert health_body["id"] == "health-1"
        assert health_body["result"]["health"]["unit"]["status"] == "healthy"
        assert runtime.module_health_contexts[-1].request_id == "health-1"


def test_gateway_websocket_handles_basic_jsonrpc_flow() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws") as websocket:
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {"clientInfo": {"name": "pytest-ws"}},
                    "id": "ws-init",
                }
            )
            initialized = websocket.receive_json()
            assert initialized["jsonrpc"] == "2.0"
            assert initialized["id"] == "ws-init"
            assert initialized["result"]["protocolVersion"] == "2024-11-05"

            websocket.send_json({"jsonrpc": "2.0", "method": "ping", "id": "ws-ping"})
            ping = websocket.receive_json()
            assert ping == {"jsonrpc": "2.0", "result": {"pong": True}, "id": "ws-ping"}

            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "method": "resources/list",
                    "params": {},
                    "id": "ws-resources",
                }
            )
            resources = websocket.receive_json()
            assert resources["jsonrpc"] == "2.0"
            assert resources["id"] == "ws-resources"
            assert resources["result"]["resources"][0]["uri"] == "resource://unit/doc"
            assert runtime.resource_list_contexts[-1].request_id == "ws-resources"
            assert runtime.resource_list_contexts[-1].metadata["path"] == "/mcp/ws"


def test_gateway_websocket_maps_invalid_json_to_parse_error() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws") as websocket:
            websocket.send_text("not-json")
            body = websocket.receive_json()

    assert body["jsonrpc"] == "2.0"
    assert body["id"] is None
    assert body["error"]["code"] == -32700
    assert "Parse error" in body["error"]["message"]


def test_gateway_websocket_accepts_binary_json_frames() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws") as websocket:
            websocket.send_bytes(b'{"jsonrpc":"2.0","method":"ping","id":"binary-ping"}')
            body = websocket.receive_json()

    assert body == {"jsonrpc": "2.0", "result": {"pong": True}, "id": "binary-ping"}


def test_gateway_websocket_suppresses_notification_response() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws") as websocket:
            websocket.send_json({"jsonrpc": "2.0", "method": "ping"})
            websocket.send_json({"jsonrpc": "2.0", "method": "ping", "id": "after-notification"})
            body = websocket.receive_json()

    assert body == {"jsonrpc": "2.0", "result": {"pong": True}, "id": "after-notification"}


def test_gateway_websocket_batch_omits_notification_responses() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        with client.websocket_connect("/mcp/ws") as websocket:
            websocket.send_json(
                [
                    {"jsonrpc": "2.0", "method": "ping"},
                    {"jsonrpc": "2.0", "method": "ping", "id": "batch-ping"},
                ]
            )
            body = websocket.receive_json()

    assert body == [{"jsonrpc": "2.0", "result": {"pong": True}, "id": "batch-ping"}]


def test_gateway_stdio_handles_initialize_and_request_context() -> None:
    from mcp_unified.gateway.stdio import handle_stdio_line

    runtime = _FakeGatewayRuntime()

    initialized_line = asyncio.run(
        handle_stdio_line(
            runtime,
            '{"jsonrpc":"2.0","method":"initialize","params":{"clientInfo":{"name":"pytest-stdio"}},"id":"stdio-init"}\n',
        )
    )
    assert initialized_line is not None
    assert initialized_line.endswith("\n")
    assert not initialized_line.endswith("\n\n")
    initialized = json.loads(initialized_line)
    assert initialized["jsonrpc"] == "2.0"
    assert initialized["id"] == "stdio-init"
    assert initialized["result"]["serverInfo"] == {
        "name": "unit-gateway",
        "version": "0.0-test",
    }

    resources_line = asyncio.run(
        handle_stdio_line(
            runtime,
            '{"jsonrpc":"2.0","method":"resources/list","params":{},"id":"stdio-resources"}\n',
        )
    )
    assert resources_line is not None
    resources = json.loads(resources_line)
    assert resources["id"] == "stdio-resources"
    assert resources["result"]["resources"][0]["uri"] == "resource://unit/doc"
    assert runtime.resource_list_contexts[-1].request_id == "stdio-resources"
    assert runtime.resource_list_contexts[-1].metadata["path"] == "stdio://stdin"
    assert runtime.resource_list_contexts[-1].metadata["transport"] == "stdio"


def test_gateway_stdio_suppresses_notification_response() -> None:
    from mcp_unified.gateway.stdio import handle_stdio_line

    runtime = _FakeGatewayRuntime()

    response_line = asyncio.run(handle_stdio_line(runtime, '{"jsonrpc":"2.0","method":"ping"}\n'))

    assert response_line is None


def test_gateway_stdio_ignores_blank_lines() -> None:
    from mcp_unified.gateway.stdio import handle_stdio_line

    runtime = _FakeGatewayRuntime()

    assert asyncio.run(handle_stdio_line(runtime, "\n")) is None
    assert asyncio.run(handle_stdio_line(runtime, "  \t\r\n")) is None
    assert asyncio.run(handle_stdio_line(runtime, b"\r\n")) is None


def test_gateway_stdio_metadata_keeps_reserved_transport_values() -> None:
    from mcp_unified.gateway.stdio import handle_stdio_line

    runtime = _FakeGatewayRuntime()

    response_line = asyncio.run(
        handle_stdio_line(
            runtime,
            '{"jsonrpc":"2.0","method":"resources/list","params":{},"id":"stdio-metadata"}\n',
            metadata={
                "transport": "user-override",
                "path": "user-path",
                "method": "user-method",
                "client_host": "user-client",
                "extra": "kept",
            },
        )
    )

    assert response_line is not None
    context_metadata = runtime.resource_list_contexts[-1].metadata
    assert context_metadata["transport"] == "stdio"
    assert context_metadata["path"] == "stdio://stdin"
    assert context_metadata["method"] == "resources/list"
    assert "client_host" not in context_metadata
    assert context_metadata["extra"] == "kept"


def test_gateway_stdio_batch_omits_notification_responses() -> None:
    from mcp_unified.gateway.stdio import handle_stdio_line

    runtime = _FakeGatewayRuntime()

    response_line = asyncio.run(
        handle_stdio_line(
            runtime,
            json.dumps(
                [
                    {"jsonrpc": "2.0", "method": "ping"},
                    {"jsonrpc": "2.0", "method": "ping", "id": "stdio-batch"},
                ]
            )
            + "\n",
        )
    )

    assert response_line is not None
    assert response_line.endswith("\n")
    assert json.loads(response_line) == [{"jsonrpc": "2.0", "result": {"pong": True}, "id": "stdio-batch"}]


def test_gateway_stdio_maps_invalid_json_to_parse_error() -> None:
    from mcp_unified.gateway.stdio import handle_stdio_line

    runtime = _FakeGatewayRuntime()

    response_line = asyncio.run(handle_stdio_line(runtime, "not-json\n"))

    assert response_line is not None
    body = json.loads(response_line)
    assert body["jsonrpc"] == "2.0"
    assert body["id"] is None
    assert body["error"]["code"] == -32700
    assert "Parse error" in body["error"]["message"]


def test_gateway_response_to_json_fallback_is_json_serializable() -> None:
    class _PydanticV1LikeResponse:
        def dict(self) -> dict[str, Any]:
            return {"created_at": datetime(2026, 5, 30, 16, 0, tzinfo=timezone.utc)}

    body = gateway_fastapi._response_to_json(_PydanticV1LikeResponse())  # noqa: SLF001

    assert body == {"created_at": "2026-05-30T16:00:00+00:00"}


def test_gateway_request_rejects_missing_jsonrpc_member() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"method": "ping", "id": "missing-version"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32600, request_id="missing-version")


def test_gateway_request_rejects_invalid_jsonrpc_id_type() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "ping", "id": {"bad": 1}},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32600, request_id=None)


def test_gateway_request_rejects_non_object_params_without_coercion() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": [], "id": "bad-params"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-params")
    assert runtime.list_contexts == []


def test_gateway_request_rejects_non_object_tool_arguments_without_coercion() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "echo.search",
                    "arguments": [],
                },
                "id": "bad-args",
            },
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-args")
    assert runtime.call_requests == []


def test_gateway_request_rejects_missing_resource_uri() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "resources/read", "params": {}, "id": "bad-resource"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-resource")
    assert runtime.resource_read_requests == []


def test_gateway_request_rejects_missing_prompt_name() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "prompts/get", "params": {}, "id": "bad-prompt"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-prompt")
    assert runtime.prompt_get_requests == []


def test_gateway_prompt_get_accepts_missing_arguments() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {"name": "review.prompt"},
                "id": "prompt-no-args",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "prompt-no-args"
    assert body["result"]["messages"][0]["content"]["text"] == "review.prompt:"
    assert runtime.prompt_get_requests[-1][1] == {}


def test_gateway_request_rejects_non_object_prompt_arguments_without_coercion() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "prompts/get",
                "params": {
                    "name": "review.prompt",
                    "arguments": [],
                },
                "id": "bad-prompt-args",
            },
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32602, request_id="bad-prompt-args")
    assert runtime.prompt_get_requests == []


def test_gateway_request_maps_custom_runtime_exceptions_to_jsonrpc_internal_error() -> None:
    runtime = _CustomExplodingGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "explode"},
        )

    assert response.status_code == 200
    _assert_jsonrpc_error(response.json(), code=-32603, request_id="explode")


def test_gateway_request_logs_custom_runtime_exceptions(monkeypatch: Any) -> None:
    runtime = _CustomExplodingGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")
    fake_logger = _FakeLogger()
    monkeypatch.setattr(gateway_jsonrpc, "logger", fake_logger, raising=False)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "explode"},
        )

    assert response.status_code == 200
    assert fake_logger.opt_calls == [{"exception": True}]
    assert fake_logger.error_calls == [
        ("Gateway runtime error while handling method={!r} request_id={!r}", ("tools/list", "explode"))
    ]


def test_gateway_notification_runtime_errors_do_not_return_jsonrpc_response() -> None:
    runtime = _CustomExplodingGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}},
        )

    assert response.status_code == 204
    assert response.content == b""


def test_gateway_batch_omits_notification_runtime_errors() -> None:
    runtime = _CustomExplodingGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/mcp/request",
            json=[
                {"jsonrpc": "2.0", "method": "tools/list", "params": {}},
                {"jsonrpc": "2.0", "method": "ping", "id": "ok"},
            ],
        )

    assert response.status_code == 200
    assert response.json() == [{"jsonrpc": "2.0", "result": {"pong": True}, "id": "ok"}]
