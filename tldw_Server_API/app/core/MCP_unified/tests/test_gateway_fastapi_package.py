from __future__ import annotations

import ast
import asyncio
import json
import os
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
from mcp_unified.gateway.admin_auth import GatewayAdminAuthConfig
from mcp_unified.gateway.credential_grants import (
    GatewayCredentialGrantManagementError,
)
from mcp_unified.gateway.external_runtime import GatewayExternalRuntimeError
from mcp_unified.storage.models import ExternalServerDefinition

REPO_ROOT = Path(__file__).resolve().parents[5]
GATEWAY_PACKAGE_ROOT = REPO_ROOT / "apps" / "mcp-unified" / "src"
GATEWAY_ROOT = GATEWAY_PACKAGE_ROOT / "mcp_unified" / "gateway"
PROFILE_DISCOVERY_READ_TOOL_NAMES = {
    "tool_categories.list",
    "profile.tools.list",
    "tool_search",
    "tool_describe",
}
PROFILE_DISCOVERY_CALL_TOOL_NAME = "tool_call"


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
                "metadata": {"category": "code", "capability": "code_search"},
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


class _SkillsRenderGatewayRuntime(_CustomToolListGatewayRuntime):
    """Fake backend that accepts the real skills.render argument shape."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        self.call_requests.append((name, arguments, context))
        return {"content": [{"type": "text", "text": f"{name}:{arguments['skill_name']}"}]}


class _FakeLogger:
    def __init__(self) -> None:
        self.opt_calls: list[dict[str, Any]] = []
        self.error_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.warning_calls: list[tuple[str, tuple[Any, ...]]] = []

    def opt(self, **kwargs: Any) -> _FakeLogger:
        self.opt_calls.append(kwargs)
        return self

    def error(self, message: str, *args: Any) -> None:
        self.error_calls.append((message, args))

    def warning(self, message: str, *args: Any) -> None:
        self.warning_calls.append((message, args))


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


def _listed_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    """Return tool names from JSON-RPC tool descriptors."""

    return [tool["name"] for tool in tools]


def _assert_profile_runtime_tool_names(
    tools: list[dict[str, Any]],
    *,
    backend_tools: list[str],
    includes_tool_call: bool = False,
) -> None:
    """Assert ordinary profile tools plus synthetic discovery helpers are exposed."""

    expected_names = {
        *backend_tools,
        *PROFILE_DISCOVERY_READ_TOOL_NAMES,
    }
    if includes_tool_call:
        expected_names.add(PROFILE_DISCOVERY_CALL_TOOL_NAME)
    else:
        expected_names.discard(PROFILE_DISCOVERY_CALL_TOOL_NAME)
    listed_names = _listed_tool_names(tools)
    assert len(listed_names) == len(set(listed_names))
    assert set(listed_names) == expected_names


def _profile_with_allowed_tools(profile_id: str, allowed_tools: list[str]) -> Any:
    """Build a profile that allows only the supplied explicit tool names."""

    from mcp_unified.profiles.models import MCPProfile, ProfilePolicy

    return MCPProfile(
        id=profile_id,
        name=f"Profile {profile_id}",
        policy_document=ProfilePolicy(allowed_tools=allowed_tools),
    )


def _profile_with_allowed_tools_and_permission_rules(
    profile_id: str,
    *,
    allowed_tools: list[str],
    permission_rules: list[Any],
) -> Any:
    """Build a profile with explicit tools plus Claude-style permission rules."""

    from mcp_unified.profiles.models import MCPProfile, ProfilePolicy

    return MCPProfile(
        id=profile_id,
        name=f"Profile {profile_id}",
        policy_document=ProfilePolicy(
            allowed_tools=allowed_tools,
            permission_rules=permission_rules,
        ),
    )


def _profile_with_capabilities(profile_id: str, capabilities: list[str]) -> Any:
    """Build a profile that allows tools by advertised capability metadata."""

    from mcp_unified.profiles.models import MCPProfile, ProfilePolicy

    return MCPProfile(
        id=profile_id,
        name=f"Profile {profile_id}",
        policy_document=ProfilePolicy(capabilities=capabilities),
    )


def _profile_with_tooling_metadata(
    profile_id: str,
    *,
    capabilities: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    recommended_tools: list[dict[str, Any]] | None = None,
    direct_categories: list[str] | None = None,
    deferred_categories: list[str] | None = None,
) -> Any:
    """Build a profile with default-profile tooling metadata."""

    from mcp_unified.profiles.models import MCPProfile, ProfilePolicy

    return MCPProfile(
        id=profile_id,
        name=f"Profile {profile_id}",
        policy_document=ProfilePolicy(
            allowed_tools=allowed_tools or [],
            denied_tools=denied_tools or [],
            capabilities=capabilities or [],
        ),
        metadata={
            "tooling": {
                "recommended_tools": recommended_tools or [],
                "progressive_disclosure": {
                    "direct_categories": direct_categories or [],
                    "deferred_categories": deferred_categories or [],
                    "max_direct_tools": 24,
                },
            }
        },
    )


class _ProfileManagementManagerDouble:
    """Small manager double that returns deterministic profile-management payloads."""

    def __init__(self, marker: str = "manager") -> None:
        self.marker = marker
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def list_profiles(self) -> dict[str, Any]:
        self.calls.append(("list_profiles", (), {}))
        return {
            "ok": True,
            "profiles": [{"id": self.marker, "name": f"Profile {self.marker}"}],
            "store": {"kind": "memory", "persistent": False},
        }

    async def show_profile(self, profile_id: str) -> dict[str, Any]:
        self.calls.append(("show_profile", (profile_id,), {}))
        return {
            "ok": True,
            "profile": {"id": profile_id, "name": f"Profile {profile_id}"},
            "store": {"kind": "memory", "persistent": False},
        }

    async def duplicate_preset(
        self,
        preset_id: str,
        *,
        profile_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "duplicate_preset",
                (preset_id,),
                {"profile_id": profile_id, "name": name},
            )
        )
        resolved_profile_id = profile_id or preset_id
        return {
            "ok": True,
            "profile": {
                "id": resolved_profile_id,
                "name": name or f"Profile {resolved_profile_id}",
                "preset_id": preset_id,
                "preset_version": "2026.05.27",
            },
            "store": {"kind": "memory", "persistent": False},
        }

    async def get_default_profile(self) -> dict[str, Any]:
        self.calls.append(("get_default_profile", (), {}))
        return {
            "ok": True,
            "profile": {"id": self.marker, "name": f"Profile {self.marker}"},
            "assignment": None,
            "default": {
                "source": "fallback_default_profile_id",
                "profile_id": self.marker,
                "assignment_id": None,
            },
            "store": {"kind": "memory", "persistent": False},
        }

    async def set_default_profile(self, profile_id: str) -> dict[str, Any]:
        self.calls.append(("set_default_profile", (profile_id,), {}))
        return {
            "ok": True,
            "profile": {"id": profile_id, "name": f"Profile {profile_id}"},
            "assignment": {
                "id": "gateway-default",
                "profile_id": profile_id,
                "is_default": True,
            },
            "default": {
                "source": "assignment",
                "profile_id": profile_id,
                "assignment_id": "gateway-default",
            },
            "store": {"kind": "memory", "persistent": False},
        }

    async def create_profile(self, profile_payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create_profile", (profile_payload,), {}))
        return {
            "ok": True,
            "profile": profile_payload,
            "store": {"kind": "memory", "persistent": False},
        }

    async def patch_profile(
        self,
        profile_id: str,
        patch_payload: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("patch_profile", (profile_id, patch_payload), {}))
        return {
            "ok": True,
            "profile": {"id": profile_id, "name": f"Profile {profile_id}", **patch_payload},
            "store": {"kind": "memory", "persistent": False},
        }

    async def delete_profile(self, profile_id: str) -> dict[str, Any]:
        self.calls.append(("delete_profile", (profile_id,), {}))
        return {
            "ok": True,
            "profile_id": profile_id,
            "store": {"kind": "memory", "persistent": False},
        }


class _ProfileManagementBootstrapDouble:
    def __init__(self, manager: _ProfileManagementManagerDouble) -> None:
        self.profile_manager = manager


class _ProfileManagementErrorManagerDouble(_ProfileManagementManagerDouble):
    def __init__(self, method: str, reason_code: str) -> None:
        super().__init__()
        self.method = method
        self.reason_code = reason_code

    async def _raise_if_targeted(self, method: str) -> None:
        if method == self.method:
            from mcp_unified.gateway.profiles import GatewayProfileManagementError

            raise GatewayProfileManagementError(
                f"domain failure: {self.reason_code}",
                reason_code=self.reason_code,
                profile_id="missing-profile" if "profile" in self.reason_code else None,
                preset_id="missing-preset" if "preset" in self.reason_code else None,
            )

    async def list_profiles(self) -> dict[str, Any]:
        await self._raise_if_targeted("list_profiles")
        return await super().list_profiles()

    async def show_profile(self, profile_id: str) -> dict[str, Any]:
        await self._raise_if_targeted("show_profile")
        return await super().show_profile(profile_id)

    async def duplicate_preset(
        self,
        preset_id: str,
        *,
        profile_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        await self._raise_if_targeted("duplicate_preset")
        return await super().duplicate_preset(
            preset_id,
            profile_id=profile_id,
            name=name,
        )

    async def get_default_profile(self) -> dict[str, Any]:
        await self._raise_if_targeted("get_default_profile")
        return await super().get_default_profile()

    async def set_default_profile(self, profile_id: str) -> dict[str, Any]:
        await self._raise_if_targeted("set_default_profile")
        return await super().set_default_profile(profile_id)

    async def create_profile(self, profile_payload: dict[str, Any]) -> dict[str, Any]:
        await self._raise_if_targeted("create_profile")
        return await super().create_profile(profile_payload)

    async def patch_profile(
        self,
        profile_id: str,
        patch_payload: dict[str, Any],
    ) -> dict[str, Any]:
        await self._raise_if_targeted("patch_profile")
        return await super().patch_profile(profile_id, patch_payload)

    async def delete_profile(self, profile_id: str) -> dict[str, Any]:
        await self._raise_if_targeted("delete_profile")
        return await super().delete_profile(profile_id)


class _ProfileManagementRuntimeErrorManagerDouble(_ProfileManagementManagerDouble):
    async def get_default_profile(self) -> dict[str, Any]:
        raise RuntimeError("profile backend leaked detail")


def _external_server_response_payload(
    server_id: str,
    server_name: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Return a complete external server response payload for schema validation."""

    timestamp = datetime(2026, 5, 31, tzinfo=timezone.utc)
    payload: dict[str, Any] = {
        "id": server_id,
        "name": server_name,
        "transport": "websocket",
        "url": "wss://example.test/mcp",
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    payload.update(overrides)
    return ExternalServerDefinition(**payload).model_dump(mode="json")


class _ExternalRegistryManagerDouble:
    """Small manager double that returns deterministic external registry payloads."""

    def __init__(self, marker: str = "manager") -> None:
        self.marker = marker
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def list_servers(self, enabled: bool | None = None) -> dict[str, Any]:
        self.calls.append(("list_servers", (), {"enabled": enabled}))
        return {
            "ok": True,
            "servers": [
                _external_server_response_payload(
                    self.marker,
                    f"External {self.marker}",
                )
            ],
            "store": {"kind": "memory", "persistent": False},
        }

    async def show_server(self, server_id: str) -> dict[str, Any]:
        self.calls.append(("show_server", (server_id,), {}))
        return {
            "ok": True,
            "server": _external_server_response_payload(
                server_id,
                f"External {server_id}",
            ),
            "store": {"kind": "memory", "persistent": False},
        }

    async def create_server(self, server_payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create_server", (server_payload,), {}))
        return {
            "ok": True,
            "server": _external_server_response_payload(
                str(server_payload["id"]),
                str(server_payload["name"]),
                **{
                    key: value
                    for key, value in server_payload.items()
                    if key not in {"id", "name"}
                },
            ),
            "store": {"kind": "memory", "persistent": False},
        }

    async def patch_server(
        self,
        server_id: str,
        patch_payload: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("patch_server", (server_id, patch_payload), {}))
        return {
            "ok": True,
            "server": _external_server_response_payload(
                server_id,
                f"External {server_id}",
                **patch_payload,
            ),
            "store": {"kind": "memory", "persistent": False},
        }

    async def delete_server(self, server_id: str) -> dict[str, Any]:
        self.calls.append(("delete_server", (server_id,), {}))
        return {
            "ok": True,
            "server_id": server_id,
            "store": {"kind": "memory", "persistent": False},
        }


class _ExternalRegistryBootstrapDouble:
    def __init__(self, manager: _ExternalRegistryManagerDouble) -> None:
        self.external_registry_manager = manager


class _ExternalRegistryRuntimeErrorManagerDouble(_ExternalRegistryManagerDouble):
    async def list_servers(self, enabled: bool | None = None) -> dict[str, Any]:
        raise RuntimeError("external registry leaked detail")


class _ExternalRegistryErrorManagerDouble(_ExternalRegistryManagerDouble):
    def __init__(self, method: str, reason_code: str) -> None:
        super().__init__()
        self.method = method
        self.reason_code = reason_code

    async def _raise_if_targeted(self, method: str) -> None:
        if method == self.method:
            from mcp_unified.gateway.external_registry import (
                GatewayExternalRegistryManagementError,
            )

            raise GatewayExternalRegistryManagementError(
                f"domain failure: {self.reason_code}",
                reason_code=self.reason_code,
                server_id="external-search",
            )

    async def list_servers(self, enabled: bool | None = None) -> dict[str, Any]:
        await self._raise_if_targeted("list_servers")
        return await super().list_servers(enabled=enabled)

    async def show_server(self, server_id: str) -> dict[str, Any]:
        await self._raise_if_targeted("show_server")
        return await super().show_server(server_id)

    async def create_server(self, server_payload: dict[str, Any]) -> dict[str, Any]:
        await self._raise_if_targeted("create_server")
        return await super().create_server(server_payload)

    async def patch_server(
        self,
        server_id: str,
        patch_payload: dict[str, Any],
    ) -> dict[str, Any]:
        await self._raise_if_targeted("patch_server")
        return await super().patch_server(server_id, patch_payload)

    async def delete_server(self, server_id: str) -> dict[str, Any]:
        await self._raise_if_targeted("delete_server")
        return await super().delete_server(server_id)


class _CredentialGrantManagerDouble:
    """Small manager double for credential-grant management route tests."""

    def __init__(self, marker: str = "grant-one") -> None:
        self.marker = marker
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "list_grants",
                (),
                {
                    "profile_id": profile_id,
                    "external_server_id": external_server_id,
                },
            )
        )
        return {
            "ok": True,
            "grants": [
                {
                    "id": self.marker,
                    "profile_id": "reviewer",
                    "broker_id": "env-broker",
                    "credential_slot": "github_token",
                    "external_server_id": "github-mcp",
                    "scopes": ["repo:read"],
                    "metadata": {"label": "GitHub read token"},
                    "provenance": {"source": "test"},
                    "enabled": True,
                }
            ],
            "store": {"kind": "memory", "persistent": False},
        }

    async def show_grant(self, grant_id: str) -> dict[str, Any]:
        self.calls.append(("show_grant", (grant_id,), {}))
        return {
            "ok": True,
            "grant": {
                "id": grant_id,
                "profile_id": "reviewer",
                "broker_id": "env-broker",
                "credential_slot": "github_token",
                "external_server_id": "github-mcp",
                "scopes": ["repo:read"],
                "metadata": {"label": "GitHub read token"},
                "provenance": {"source": "test"},
                "enabled": True,
            },
            "store": {"kind": "memory", "persistent": False},
        }

    async def create_grant(self, grant_payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("create_grant", (grant_payload,), {}))
        return {
            "ok": True,
            "grant": grant_payload,
            "store": {"kind": "memory", "persistent": False},
        }

    async def patch_grant(
        self,
        grant_id: str,
        patch_payload: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("patch_grant", (grant_id, patch_payload), {}))
        return {
            "ok": True,
            "grant": {
                "id": grant_id,
                "profile_id": "reviewer",
                "broker_id": "env-broker",
                "credential_slot": "github_token",
                "external_server_id": "github-mcp",
                "scopes": ["repo:read"],
                "metadata": patch_payload.get("metadata", {}),
                "provenance": {"source": "test"},
                "enabled": patch_payload.get("enabled", True),
            },
            "store": {"kind": "memory", "persistent": False},
        }

    async def delete_grant(self, grant_id: str) -> dict[str, Any]:
        self.calls.append(("delete_grant", (grant_id,), {}))
        return {
            "ok": True,
            "grant_id": grant_id,
            "store": {"kind": "memory", "persistent": False},
        }


class _CredentialGrantErrorManagerDouble(_CredentialGrantManagerDouble):
    def __init__(self, method: str, reason_code: str) -> None:
        super().__init__()
        self.method = method
        self.reason_code = reason_code

    async def _raise_if_targeted(self, method: str) -> None:
        if method == self.method:
            raise GatewayCredentialGrantManagementError(
                f"domain failure: {self.reason_code}",
                reason_code=self.reason_code,
                grant_id="grant-one",
            )

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> dict[str, Any]:
        await self._raise_if_targeted("list_grants")
        return await super().list_grants(
            profile_id=profile_id,
            external_server_id=external_server_id,
        )

    async def create_grant(self, grant_payload: dict[str, Any]) -> dict[str, Any]:
        await self._raise_if_targeted("create_grant")
        return await super().create_grant(grant_payload)


class _ExternalRuntimeManagerDouble:
    """Small manager double for external runtime route tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def list_runtime_servers(self) -> dict[str, Any]:
        self.calls.append(("list_runtime_servers", (), {}))
        return {
            "ok": True,
            "servers": [
                {
                    "id": "research",
                    "status": "healthy",
                    "installer": {
                        "available": False,
                        "reason_code": "external_server_installer_not_configured",
                        "server_id": "research",
                    },
                }
            ],
        }

    async def start_server(self, server_id: str) -> dict[str, Any]:
        self.calls.append(("start_server", (server_id,), {}))
        return {
            "ok": True,
            "reason_code": "external_server_started",
            "server_id": server_id,
        }

    async def stop_server(self, server_id: str) -> dict[str, Any]:
        self.calls.append(("stop_server", (server_id,), {}))
        return {
            "ok": True,
            "reason_code": "external_server_stopped",
            "server_id": server_id,
        }

    async def restart_server(self, server_id: str) -> dict[str, Any]:
        self.calls.append(("restart_server", (server_id,), {}))
        return {
            "ok": True,
            "reason_code": "external_server_restarted",
            "server_id": server_id,
        }

    async def refresh_server(self, server_id: str | None = None) -> dict[str, Any]:
        self.calls.append(("refresh_server", (server_id,), {}))
        return {
            "ok": True,
            "reason_code": "external_server_refreshed",
            "server_id": server_id,
        }

    async def reconcile(self, server_id: str | None = None) -> dict[str, Any]:
        self.calls.append(("reconcile", (server_id,), {}))
        return {
            "ok": True,
            "reason_code": "external_server_reconciled",
            "server_id": server_id,
        }

    async def stop_all(self) -> dict[str, Any]:
        self.calls.append(("stop_all", (), {}))
        return {
            "ok": True,
            "reason_code": "external_runtime_stopped",
            "stopped_servers": 1,
            "total_servers": 1,
            "errors": {},
        }

    async def install_server(
        self,
        server_id: str,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(("install_server", (server_id,), {"context": context}))
        return {
            "ok": False,
            "reason_code": "external_server_install_not_configured",
            "server_id": server_id,
        }

    async def update_server(
        self,
        server_id: str,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(("update_server", (server_id,), {"context": context}))
        return {
            "ok": False,
            "reason_code": "external_server_update_not_configured",
            "server_id": server_id,
        }


class _ExternalRuntimeBootstrapDouble:
    def __init__(
        self,
        manager: _ExternalRuntimeManagerDouble,
        *,
        lifecycle: Any = None,
    ) -> None:
        self.external_runtime_manager = manager
        self.external_runtime_lifecycle = lifecycle


class _ExternalRuntimeErrorManagerDouble(_ExternalRuntimeManagerDouble):
    def __init__(self, method: str, reason_code: str) -> None:
        super().__init__()
        self.method = method
        self.reason_code = reason_code

    async def _raise_if_targeted(self, method: str) -> None:
        if method == self.method:
            raise GatewayExternalRuntimeError(
                f"runtime failure: {self.reason_code}",
                reason_code=self.reason_code,
                server_id="research",
            )

    async def start_server(self, server_id: str) -> dict[str, Any]:
        await self._raise_if_targeted("start_server")
        return await super().start_server(server_id)

    async def refresh_server(self, server_id: str | None = None) -> dict[str, Any]:
        await self._raise_if_targeted("refresh_server")
        return await super().refresh_server(server_id)


class _ExternalRuntimeReconcileFailureManagerDouble(_ExternalRuntimeManagerDouble):
    async def reconcile(self, server_id: str | None = None) -> dict[str, Any]:
        self.calls.append(("reconcile", (server_id,), {}))
        return {
            "ok": False,
            "reason_code": "external_server_reconciled",
            "server_id": server_id,
            "started_servers": 0,
            "stopped_servers": 0,
            "restarted_servers": 0,
            "refreshed_servers": 0,
            "total_servers": 1,
            "errors": {"research": "external_server_start_failed"},
        }


class _ExternalRuntimeLifecycleExceptionManagerDouble(_ExternalRuntimeManagerDouble):
    async def reconcile(self, server_id: str | None = None) -> dict[str, Any]:
        self.calls.append(("reconcile", (server_id,), {}))
        raise RuntimeError("startup exploded")


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
    env = {
        **os.environ,
        "PYTHONPATH": (
            f"{GATEWAY_PACKAGE_ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
            if os.environ.get("PYTHONPATH")
            else str(GATEWAY_PACKAGE_ROOT)
        ),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mcp_unified.gateway.stdio; print('mcp_unified.gateway.fastapi' in sys.modules)",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_gateway_profile_management_routes_are_not_mounted_by_default() -> None:
    app = create_gateway_app(_FakeGatewayRuntime(), prefix="/mcp")

    with TestClient(app) as client:
        response = client.get("/mcp/profiles")

    assert response.status_code == 404


def test_gateway_profile_management_routes_mount_with_manager() -> None:
    manager = _ProfileManagementManagerDouble("direct")
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_manager=manager,
    )

    with TestClient(app) as client:
        response = client.get("/mcp/profiles")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "profiles": [{"id": "direct", "name": "Profile direct"}],
        "store": {"kind": "memory", "persistent": False},
    }


def test_gateway_profile_management_routes_mount_with_bootstrap() -> None:
    manager = _ProfileManagementManagerDouble("bootstrap")
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_bootstrap=_ProfileManagementBootstrapDouble(manager),
    )

    with TestClient(app) as client:
        response = client.get("/mcp/profiles")

    assert response.status_code == 200
    assert response.json()["profiles"] == [
        {"id": "bootstrap", "name": "Profile bootstrap"}
    ]


def test_gateway_profile_management_routes_mount_when_enabled_with_manager() -> None:
    manager = _ProfileManagementManagerDouble("enabled")
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(
        gateway_fastapi.create_gateway_router(
            _FakeGatewayRuntime(),
            profile_manager=manager,
            enable_profile_management=True,
        ),
        prefix="/mcp",
    )

    with TestClient(app) as client:
        response = client.get("/mcp/profiles")

    assert response.status_code == 200
    assert response.json()["profiles"] == [{"id": "enabled", "name": "Profile enabled"}]


def test_gateway_profile_management_enabled_without_manager_raises() -> None:
    with pytest.raises(ValueError, match="profile management requires"):
        gateway_fastapi.create_gateway_router(
            _FakeGatewayRuntime(),
            enable_profile_management=True,
        )

    with pytest.raises(ValueError, match="profile management requires"):
        create_gateway_app(
            _FakeGatewayRuntime(),
            prefix="/mcp",
            enable_profile_management=True,
        )


def test_gateway_profile_management_explicit_manager_precedes_bootstrap_manager() -> None:
    direct = _ProfileManagementManagerDouble("direct")
    bootstrap = _ProfileManagementBootstrapDouble(
        _ProfileManagementManagerDouble("bootstrap")
    )
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_manager=direct,
        profile_bootstrap=bootstrap,
    )

    with TestClient(app) as client:
        response = client.get("/mcp/profiles")

    assert response.status_code == 200
    assert response.json()["profiles"] == [{"id": "direct", "name": "Profile direct"}]
    assert direct.calls == [("list_profiles", (), {})]
    assert bootstrap.profile_manager.calls == []


def test_gateway_profile_management_success_envelopes() -> None:
    manager = _ProfileManagementManagerDouble("default")
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_manager=manager,
    )

    with TestClient(app) as client:
        listed = client.get("/mcp/profiles")
        shown = client.get("/mcp/profiles/reviewer")
        duplicated_default = client.post(
            "/mcp/profiles/from-preset",
            json={"preset_id": "project-researcher"},
        )
        duplicated_custom = client.post(
            "/mcp/profiles/from-preset",
            json={
                "preset_id": "project-researcher",
                "profile_id": "custom-researcher",
                "name": "Custom Researcher",
            },
        )
        default_before = client.get("/mcp/profiles/default")
        default_after = client.put(
            "/mcp/profiles/default",
            json={"profile_id": "architect"},
        )
        created = client.post(
            "/mcp/profiles",
            json={
                "id": "custom-reviewer",
                "name": "Custom Reviewer",
                "policy_document": {"allowed_tools": ["echo.search"]},
            },
        )
        patched = client.patch(
            "/mcp/profiles/custom-reviewer",
            json={
                "name": "Custom Reviewer v2",
                "description": "updated",
            },
        )
        deleted = client.delete("/mcp/profiles/custom-reviewer")

    assert listed.json() == {
        "ok": True,
        "profiles": [{"id": "default", "name": "Profile default"}],
        "store": {"kind": "memory", "persistent": False},
    }
    assert shown.json() == {
        "ok": True,
        "profile": {"id": "reviewer", "name": "Profile reviewer"},
        "store": {"kind": "memory", "persistent": False},
    }
    assert duplicated_default.json() == {
        "ok": True,
        "profile": {
            "id": "project-researcher",
            "name": "Profile project-researcher",
            "preset_id": "project-researcher",
            "preset_version": "2026.05.27",
        },
        "preset_id": "project-researcher",
        "preset_version": "2026.05.27",
        "store": {"kind": "memory", "persistent": False},
    }
    assert duplicated_custom.json() == {
        "ok": True,
        "profile": {
            "id": "custom-researcher",
            "name": "Custom Researcher",
            "preset_id": "project-researcher",
            "preset_version": "2026.05.27",
        },
        "preset_id": "project-researcher",
        "preset_version": "2026.05.27",
        "store": {"kind": "memory", "persistent": False},
    }
    assert default_before.json() == {
        "ok": True,
        "profile": {"id": "default", "name": "Profile default"},
        "assignment": None,
        "default": {
            "source": "fallback_default_profile_id",
            "profile_id": "default",
            "assignment_id": None,
        },
        "store": {"kind": "memory", "persistent": False},
    }
    assert default_after.json() == {
        "ok": True,
        "profile": {"id": "architect", "name": "Profile architect"},
        "assignment": {
            "id": "gateway-default",
            "profile_id": "architect",
            "is_default": True,
        },
        "default": {
            "source": "assignment",
            "profile_id": "architect",
            "assignment_id": "gateway-default",
        },
        "store": {"kind": "memory", "persistent": False},
    }
    assert created.json() == {
        "ok": True,
        "profile": {
            "id": "custom-reviewer",
            "name": "Custom Reviewer",
            "policy_document": {"allowed_tools": ["echo.search"]},
        },
        "store": {"kind": "memory", "persistent": False},
    }
    assert patched.json() == {
        "ok": True,
        "profile": {
            "id": "custom-reviewer",
            "name": "Custom Reviewer v2",
            "description": "updated",
        },
        "store": {"kind": "memory", "persistent": False},
    }
    assert deleted.json() == {
        "ok": True,
        "profile_id": "custom-reviewer",
        "store": {"kind": "memory", "persistent": False},
    }
    assert manager.calls == [
        ("list_profiles", (), {}),
        ("show_profile", ("reviewer",), {}),
        (
            "duplicate_preset",
            ("project-researcher",),
            {"profile_id": None, "name": None},
        ),
        (
            "duplicate_preset",
            ("project-researcher",),
            {"profile_id": "custom-researcher", "name": "Custom Researcher"},
        ),
        ("get_default_profile", (), {}),
        ("set_default_profile", ("architect",), {}),
        (
            "create_profile",
            (
                {
                    "id": "custom-reviewer",
                    "name": "Custom Reviewer",
                    "policy_document": {"allowed_tools": ["echo.search"]},
                },
            ),
            {},
        ),
        (
            "patch_profile",
            (
                "custom-reviewer",
                {"name": "Custom Reviewer v2", "description": "updated"},
            ),
            {},
        ),
        ("delete_profile", ("custom-reviewer",), {}),
    ]


def test_gateway_profile_management_routes_have_pydantic_response_models() -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_manager=_ProfileManagementManagerDouble(),
    )

    paths = app.openapi()["paths"]
    expected_refs = {
        ("/mcp/profiles", "get"): "#/components/schemas/ProfileListResponse",
        ("/mcp/profiles", "post"): "#/components/schemas/ProfileResponse",
        ("/mcp/profiles/{profile_id}", "get"): "#/components/schemas/ProfileResponse",
        ("/mcp/profiles/{profile_id}", "patch"): "#/components/schemas/ProfileResponse",
        ("/mcp/profiles/{profile_id}", "delete"): "#/components/schemas/DeleteProfileResponse",
        ("/mcp/profiles/from-preset", "post"): "#/components/schemas/DuplicatePresetResponse",
        ("/mcp/profiles/default", "get"): "#/components/schemas/DefaultProfileResponse",
        ("/mcp/profiles/default", "put"): "#/components/schemas/DefaultProfileResponse",
    }

    for (path, method), expected_ref in expected_refs.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": expected_ref}

@pytest.mark.parametrize(
    ("method", "path", "json_body", "manager_method", "reason_code", "status_code"),
    [
        ("GET", "/mcp/profiles/missing", None, "show_profile", "profile_not_found", 404),
        (
            "POST",
            "/mcp/profiles/from-preset",
            {"preset_id": "missing-preset"},
            "duplicate_preset",
            "preset_not_found",
            404,
        ),
        (
            "GET",
            "/mcp/profiles/default",
            None,
            "get_default_profile",
            "default_profile_not_configured",
            404,
        ),
        (
            "PUT",
            "/mcp/profiles/default",
            {"profile_id": "disabled"},
            "set_default_profile",
            "profile_disabled",
            409,
        ),
        (
            "POST",
            "/mcp/profiles/from-preset",
            {"preset_id": "project-researcher"},
            "duplicate_preset",
            "profile_already_exists",
            409,
        ),
        (
            "PATCH",
            "/mcp/profiles/reviewer",
            {"name": "Reviewer"},
            "patch_profile",
            "invalid_profile_patch",
            422,
        ),
        (
            "PATCH",
            "/mcp/profiles/reviewer",
            {"name": "Reviewer"},
            "patch_profile",
            "permission_change_denied",
            403,
        ),
        (
            "PATCH",
            "/mcp/profiles/reviewer",
            {"name": "Reviewer"},
            "patch_profile",
            "permission_change_requires_approval",
            409,
        ),
        (
            "DELETE",
            "/mcp/profiles/reviewer",
            None,
            "delete_profile",
            "profile_is_default",
            409,
        ),
        (
            "DELETE",
            "/mcp/profiles/reviewer",
            None,
            "delete_profile",
            "profile_has_assignments",
            409,
        ),
        ("GET", "/mcp/profiles", None, "list_profiles", "profile_store_unavailable", 503),
        (
            "PUT",
            "/mcp/profiles/default",
            {"profile_id": "reviewer"},
            "set_default_profile",
            "assignment_store_unavailable",
            503,
        ),
    ],
)
def test_gateway_profile_management_error_status_mapping(
    method: str,
    path: str,
    json_body: dict[str, Any] | None,
    manager_method: str,
    reason_code: str,
    status_code: int,
) -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_manager=_ProfileManagementErrorManagerDouble(
            manager_method,
            reason_code,
        ),
    )

    with TestClient(app) as client:
        response = client.request(method, path, json=json_body)

    assert response.status_code == status_code
    body = response.json()
    assert body["ok"] is False
    assert body["reason_code"] == reason_code
    assert isinstance(body["error"], str)
    assert body["error"]
    assert "domain failure" not in body["error"]


def test_gateway_profile_management_malformed_or_missing_bodies_return_422() -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_manager=_ProfileManagementManagerDouble(),
    )

    with TestClient(app) as client:
        missing_duplicate = client.post("/mcp/profiles/from-preset", json={})
        malformed_duplicate = client.post(
            "/mcp/profiles/from-preset",
            content="{",
            headers={"content-type": "application/json"},
        )
        missing_default = client.put("/mcp/profiles/default", json={})
        malformed_default = client.put(
            "/mcp/profiles/default",
            content="{",
            headers={"content-type": "application/json"},
        )
        missing_create = client.post("/mcp/profiles", json={})
        malformed_create = client.post(
            "/mcp/profiles",
            content="{",
            headers={"content-type": "application/json"},
        )

    assert missing_duplicate.status_code == 422
    assert malformed_duplicate.status_code == 422
    assert missing_default.status_code == 422
    assert malformed_default.status_code == 422
    assert missing_create.status_code == 422
    assert malformed_create.status_code == 422


def test_gateway_external_registry_management_routes_are_not_mounted_by_default() -> None:
    app = create_gateway_app(_FakeGatewayRuntime(), prefix="/mcp")

    with TestClient(app) as client:
        response = client.get("/mcp/external-servers")

    assert response.status_code == 404


def test_gateway_external_registry_management_routes_mount_with_manager() -> None:
    manager = _ExternalRegistryManagerDouble("direct")
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_registry_manager=manager,
    )

    with TestClient(app) as client:
        response = client.get("/mcp/external-servers")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "servers": [
            _external_server_response_payload("direct", "External direct")
        ],
        "store": {"kind": "memory", "persistent": False},
    }


def test_gateway_external_registry_management_routes_mount_when_enabled_with_manager() -> None:
    manager = _ExternalRegistryManagerDouble("enabled")
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(
        gateway_fastapi.create_gateway_router(
            _FakeGatewayRuntime(),
            external_registry_manager=manager,
            enable_external_registry_management=True,
        ),
        prefix="/mcp",
    )

    with TestClient(app) as client:
        response = client.get("/mcp/external-servers")

    assert response.status_code == 200
    assert response.json()["servers"] == [
        _external_server_response_payload("enabled", "External enabled")
    ]


def test_gateway_external_registry_management_enabled_without_manager_raises() -> None:
    with pytest.raises(ValueError, match="external registry management requires"):
        gateway_fastapi.create_gateway_router(
            _FakeGatewayRuntime(),
            enable_external_registry_management=True,
        )

    with pytest.raises(ValueError, match="external registry management requires"):
        create_gateway_app(
            _FakeGatewayRuntime(),
            prefix="/mcp",
            enable_external_registry_management=True,
        )


def test_gateway_external_registry_management_explicit_manager_precedes_bootstrap_manager() -> None:
    direct = _ExternalRegistryManagerDouble("direct")
    bootstrap = _ExternalRegistryBootstrapDouble(
        _ExternalRegistryManagerDouble("bootstrap")
    )
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_registry_manager=direct,
        profile_bootstrap=bootstrap,
    )

    with TestClient(app) as client:
        response = client.get("/mcp/external-servers")

    assert response.status_code == 200
    assert response.json()["servers"] == [
        _external_server_response_payload("direct", "External direct")
    ]
    assert direct.calls == [("list_servers", (), {"enabled": None})]
    assert bootstrap.external_registry_manager.calls == []


def test_gateway_external_registry_management_success_envelopes() -> None:
    manager = _ExternalRegistryManagerDouble("default")
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_registry_manager=manager,
    )

    with TestClient(app) as client:
        listed = client.get("/mcp/external-servers")
        listed_enabled = client.get("/mcp/external-servers?enabled=true")
        shown = client.get("/mcp/external-servers/search")
        created = client.post(
            "/mcp/external-servers",
            json={
                "id": "search",
                "name": "Search",
                "transport": "websocket",
                "url": "wss://example.test/mcp",
                "metadata": {"tier": "test"},
            },
        )
        patched = client.patch(
            "/mcp/external-servers/search",
            json={
                "name": "Search v2",
                "enabled": False,
            },
        )
        deleted = client.delete("/mcp/external-servers/search")

    assert listed.json() == {
        "ok": True,
        "servers": [
            _external_server_response_payload("default", "External default"),
        ],
        "store": {"kind": "memory", "persistent": False},
    }
    assert listed_enabled.json()["servers"] == [
        _external_server_response_payload("default", "External default"),
    ]
    assert shown.json() == {
        "ok": True,
        "server": _external_server_response_payload("search", "External search"),
        "store": {"kind": "memory", "persistent": False},
    }
    assert created.json() == {
        "ok": True,
        "server": _external_server_response_payload(
            "search",
            "Search",
            metadata={"tier": "test"},
        ),
        "store": {"kind": "memory", "persistent": False},
    }
    assert patched.json() == {
        "ok": True,
        "server": _external_server_response_payload(
            "search",
            "Search v2",
            enabled=False,
        ),
        "store": {"kind": "memory", "persistent": False},
    }
    assert deleted.json() == {
        "ok": True,
        "server_id": "search",
        "store": {"kind": "memory", "persistent": False},
    }
    assert manager.calls == [
        ("list_servers", (), {"enabled": None}),
        ("list_servers", (), {"enabled": True}),
        ("show_server", ("search",), {}),
        (
            "create_server",
            (
                {
                    "id": "search",
                    "name": "Search",
                    "transport": "websocket",
                    "command": [],
                    "url": "wss://example.test/mcp",
                    "metadata": {"tier": "test"},
                },
            ),
            {},
        ),
        (
            "patch_server",
            ("search", {"name": "Search v2", "enabled": False}),
            {},
        ),
        ("delete_server", ("search",), {}),
    ]


def test_gateway_external_registry_management_routes_have_pydantic_response_models() -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_registry_manager=_ExternalRegistryManagerDouble(),
    )

    paths = app.openapi()["paths"]
    expected_refs = {
        ("/mcp/external-servers", "get"): "#/components/schemas/ExternalServerListResponse",
        ("/mcp/external-servers", "post"): "#/components/schemas/ExternalServerResponse",
        (
            "/mcp/external-servers/{server_id}",
            "get",
        ): "#/components/schemas/ExternalServerResponse",
        (
            "/mcp/external-servers/{server_id}",
            "patch",
        ): "#/components/schemas/ExternalServerResponse",
        (
            "/mcp/external-servers/{server_id}",
            "delete",
        ): "#/components/schemas/DeleteExternalServerResponse",
    }

    for (path, method), expected_ref in expected_refs.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": expected_ref}

    schemas = app.openapi()["components"]["schemas"]
    assert schemas["ExternalServerResponse"]["properties"]["server"] == {
        "$ref": "#/components/schemas/ExternalServerDefinition"
    }
    assert schemas["ExternalServerListResponse"]["properties"]["servers"]["items"] == {
        "$ref": "#/components/schemas/ExternalServerDefinition"
    }


def test_gateway_credential_grant_management_routes_are_not_mounted_by_default() -> None:
    app = create_gateway_app(_FakeGatewayRuntime(), prefix="/mcp")

    with TestClient(app) as client:
        response = client.get("/mcp/credential-grants")

    assert response.status_code == 404


def test_gateway_credential_grant_management_enabled_without_manager_raises() -> None:
    with pytest.raises(ValueError, match="credential grant management requires"):
        gateway_fastapi.create_gateway_router(
            _FakeGatewayRuntime(),
            enable_credential_grant_management=True,
        )

    with pytest.raises(ValueError, match="credential grant management requires"):
        create_gateway_app(
            _FakeGatewayRuntime(),
            prefix="/mcp",
            enable_credential_grant_management=True,
        )


def test_gateway_credential_grant_management_success_envelopes() -> None:
    manager = _CredentialGrantManagerDouble("grant-one")
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        credential_grant_manager=manager,
    )

    with TestClient(app) as client:
        listed = client.get(
            "/mcp/credential-grants",
            params={
                "profile_id": "reviewer",
                "external_server_id": "github-mcp",
            },
        )
        shown = client.get("/mcp/credential-grants/grant-one")
        created = client.post(
            "/mcp/credential-grants",
            json={
                "id": "grant-one",
                "profile_id": "reviewer",
                "broker_id": "env-broker",
                "credential_slot": "github_token",
                "external_server_id": "github-mcp",
                "scopes": ["repo:read"],
                "metadata": {"label": "GitHub read token"},
                "provenance": {"source": "test"},
            },
        )
        patched = client.patch(
            "/mcp/credential-grants/grant-one",
            json={"metadata": {"label": "Updated"}, "enabled": False},
        )
        deleted = client.delete("/mcp/credential-grants/grant-one")

    assert listed.status_code == 200
    assert listed.json()["grants"][0]["id"] == "grant-one"
    assert shown.status_code == 200
    assert shown.json()["grant"]["credential_slot"] == "github_token"
    assert created.status_code == 200
    expected_created_grant = {
        "id": "grant-one",
        "profile_id": "reviewer",
        "broker_id": "env-broker",
        "credential_slot": "github_token",
        "external_server_id": "github-mcp",
        "scopes": ["repo:read"],
        "metadata": {"label": "GitHub read token"},
        "provenance": {"source": "test"},
    }
    created_grant = created.json()["grant"]
    assert {
        key: created_grant[key]
        for key in expected_created_grant
    } == expected_created_grant
    assert created_grant["enabled"] is True
    assert isinstance(created_grant["created_at"], str)
    assert isinstance(created_grant["updated_at"], str)
    assert patched.json()["grant"]["metadata"] == {"label": "Updated"}
    assert patched.json()["grant"]["enabled"] is False
    assert deleted.json() == {
        "ok": True,
        "grant_id": "grant-one",
        "store": {"kind": "memory", "persistent": False},
    }
    assert manager.calls[:2] == [
        (
            "list_grants",
            (),
            {"profile_id": "reviewer", "external_server_id": "github-mcp"},
        ),
        ("show_grant", ("grant-one",), {}),
    ]
    created_payload = manager.calls[2][1][0]
    assert manager.calls[2][0] == "create_grant"
    assert {
        key: created_payload[key]
        for key in expected_created_grant
    } == expected_created_grant
    assert "enabled" not in created_payload
    assert "created_at" not in created_payload
    assert "updated_at" not in created_payload
    assert manager.calls[3:] == [
        (
            "patch_grant",
            (
                "grant-one",
                {"metadata": {"label": "Updated"}, "enabled": False},
            ),
            {},
        ),
        ("delete_grant", ("grant-one",), {}),
    ]


def test_gateway_credential_grant_management_routes_have_pydantic_response_models() -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        credential_grant_manager=_CredentialGrantManagerDouble(),
    )

    paths = app.openapi()["paths"]
    expected_refs = {
        ("/mcp/credential-grants", "get"): "#/components/schemas/CredentialGrantListResponse",
        ("/mcp/credential-grants", "post"): "#/components/schemas/CredentialGrantResponse",
        (
            "/mcp/credential-grants/{grant_id}",
            "get",
        ): "#/components/schemas/CredentialGrantResponse",
        (
            "/mcp/credential-grants/{grant_id}",
            "patch",
        ): "#/components/schemas/CredentialGrantResponse",
        (
            "/mcp/credential-grants/{grant_id}",
            "delete",
        ): "#/components/schemas/DeleteCredentialGrantResponse",
    }

    for (path, method), expected_ref in expected_refs.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": expected_ref}

    schemas = app.openapi()["components"]["schemas"]
    assert schemas["CredentialGrantResponse"]["properties"]["grant"] == {
        "$ref": "#/components/schemas/CredentialGrant"
    }
    assert schemas["CredentialGrantListResponse"]["properties"]["grants"]["items"] == {
        "$ref": "#/components/schemas/CredentialGrant"
    }


@pytest.mark.parametrize(
    ("method", "path", "json_body", "manager_method", "reason_code", "status_code"),
    [
        (
            "POST",
            "/mcp/credential-grants",
            {
                "id": "grant-one",
                "profile_id": "reviewer",
                "broker_id": "env-broker",
                "credential_slot": "github_token",
            },
            "create_grant",
            "credential_grant_already_exists",
            409,
        ),
        (
            "GET",
            "/mcp/credential-grants",
            None,
            "list_grants",
            "credential_grant_store_unavailable",
            503,
        ),
    ],
)
def test_gateway_credential_grant_management_domain_errors(
    method: str,
    path: str,
    json_body: dict[str, Any] | None,
    manager_method: str,
    reason_code: str,
    status_code: int,
) -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        credential_grant_manager=_CredentialGrantErrorManagerDouble(
            manager_method,
            reason_code,
        ),
    )

    with TestClient(app) as client:
        response = client.request(method, path, json=json_body)

    assert response.status_code == status_code
    assert response.json()["reason_code"] == reason_code
    assert response.json()["grant_id"] == "grant-one"
    assert "domain failure" not in response.json()["error"]


def test_gateway_credential_grant_management_routes_use_admin_auth() -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        credential_grant_manager=_CredentialGrantManagerDouble(),
        admin_auth=GatewayAdminAuthConfig(
            enabled=True,
            header_name="X-Test-Gateway-Admin",
            api_key="test-admin-key",
        ),
    )

    with TestClient(app) as client:
        missing = client.get("/mcp/credential-grants")
        allowed = client.get(
            "/mcp/credential-grants",
            headers={"X-Test-Gateway-Admin": "test-admin-key"},
        )

    assert missing.status_code == 401
    assert missing.json()["reason_code"] == "admin_auth_required"
    assert allowed.status_code == 200
    assert allowed.json()["grants"][0]["id"] == "grant-one"


@pytest.mark.parametrize("server_id", ["runtime", "refresh", "reconcile"])
def test_gateway_external_registry_management_rejects_reserved_server_ids(
    server_id: str,
) -> None:
    manager = _ExternalRegistryManagerDouble()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_registry_manager=manager,
    )

    with TestClient(app) as client:
        created = client.post(
            "/mcp/external-servers",
            json={
                "id": server_id,
                "name": "Reserved",
                "transport": "websocket",
                "url": "wss://example.test/mcp",
            },
        )
        shown = client.get(f"/mcp/external-servers/{server_id}")
        patched = client.patch(
            f"/mcp/external-servers/{server_id}",
            json={"name": "Reserved v2"},
        )
        deleted = client.delete(f"/mcp/external-servers/{server_id}")

    for response in (created, shown, patched, deleted):
        assert response.status_code == 422
        assert response.json()["reason_code"] == "invalid_external_server_request"
        assert response.json()["server_id"] == server_id
    assert manager.calls == []


@pytest.mark.parametrize(
    ("method", "path", "json_body", "manager_method", "reason_code", "status_code"),
    [
        (
            "GET",
            "/mcp/external-servers/missing",
            None,
            "show_server",
            "external_server_not_found",
            404,
        ),
        (
            "POST",
            "/mcp/external-servers",
            {
                "id": "search",
                "name": "Search",
                "transport": "websocket",
                "url": "wss://example.test/mcp",
            },
            "create_server",
            "external_server_already_exists",
            409,
        ),
        (
            "GET",
            "/mcp/external-servers",
            None,
            "list_servers",
            "external_registry_store_unavailable",
            503,
        ),
        (
            "GET",
            "/mcp/external-servers",
            None,
            "list_servers",
            "credential_grant_store_unavailable",
            503,
        ),
        (
            "DELETE",
            "/mcp/external-servers/search",
            None,
            "delete_server",
            "external_server_has_credential_grants",
            409,
        ),
        (
            "PATCH",
            "/mcp/external-servers/search",
            {"credential_slots": ["api_key"]},
            "patch_server",
            "credential_slot_change_requires_disabled_server",
            409,
        ),
        (
            "POST",
            "/mcp/external-servers",
            {
                "id": "search",
                "name": "Search",
                "transport": "websocket",
                "url": "wss://example.test/mcp",
            },
            "create_server",
            "invalid_external_server_request",
            422,
        ),
        (
            "PATCH",
            "/mcp/external-servers/search",
            {"name": "Search"},
            "patch_server",
            "invalid_external_server_patch",
            422,
        ),
        (
            "DELETE",
            "/mcp/external-servers/search",
            None,
            "delete_server",
            "unexpected_external_server_delete_result",
            500,
        ),
        (
            "DELETE",
            "/mcp/external-servers/search",
            None,
            "delete_server",
            "unknown_external_registry_reason",
            500,
        ),
    ],
)
def test_gateway_external_registry_management_error_status_mapping(
    method: str,
    path: str,
    json_body: dict[str, Any] | None,
    manager_method: str,
    reason_code: str,
    status_code: int,
) -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_registry_manager=_ExternalRegistryErrorManagerDouble(
            manager_method,
            reason_code,
        ),
    )

    with TestClient(app) as client:
        response = client.request(method, path, json=json_body)

    assert response.status_code == status_code
    body = response.json()
    assert body["ok"] is False
    assert body["reason_code"] == reason_code
    assert isinstance(body["error"], str)
    assert body["error"]
    assert "domain failure" not in body["error"]
    assert body["server_id"] == "external-search"


def test_gateway_external_registry_management_raw_failures_return_structured_503() -> None:
    class _FailingExternalRegistryManager:
        async def list_servers(self, enabled: bool | None = None) -> dict[str, Any]:
            raise RuntimeError("raw store failure")

    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_registry_manager=_FailingExternalRegistryManager(),
    )

    with TestClient(app) as client:
        response = client.get("/mcp/external-servers")

    assert response.status_code == 503
    assert response.json() == {
        "error": "External registry store unavailable",
        "ok": False,
        "reason_code": "external_registry_store_unavailable",
    }


def test_gateway_external_registry_management_malformed_or_missing_bodies_return_422() -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_registry_manager=_ExternalRegistryManagerDouble(),
    )

    with TestClient(app) as client:
        missing_create = client.post("/mcp/external-servers")
        malformed_create = client.post(
            "/mcp/external-servers",
            content="{",
            headers={"content-type": "application/json"},
        )
        missing_patch = client.patch("/mcp/external-servers/search")
        malformed_patch = client.patch(
            "/mcp/external-servers/search",
            content="{",
            headers={"content-type": "application/json"},
        )
        extra_create = client.post(
            "/mcp/external-servers",
            json={
                "id": "search",
                "name": "Search",
                "transport": "websocket",
                "url": "wss://example.test/mcp",
                "unsupported": True,
            },
        )
        extra_patch = client.patch(
            "/mcp/external-servers/search",
            json={"name": "Search", "unsupported": True},
        )

    assert missing_create.status_code == 422
    assert malformed_create.status_code == 422
    assert missing_patch.status_code == 422
    assert malformed_patch.status_code == 422
    assert extra_create.status_code == 422
    assert extra_patch.status_code == 422


def test_gateway_external_runtime_routes_are_not_mounted_by_default() -> None:
    app = create_gateway_app(_FakeGatewayRuntime(), prefix="/mcp")

    with TestClient(app) as client:
        response = client.get("/mcp/external-servers/runtime")

    assert response.status_code == 404


def test_gateway_external_runtime_management_routes_mount_with_manager() -> None:
    manager = _ExternalRuntimeManagerDouble()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_runtime_manager=manager,
    )

    with TestClient(app) as client:
        listed = client.get("/mcp/external-servers/runtime")
        started = client.post("/mcp/external-servers/research/start")
        stopped = client.post("/mcp/external-servers/research/stop")
        refreshed_all = client.post("/mcp/external-servers/refresh")
        refreshed_one = client.post("/mcp/external-servers/research/refresh")
        reconciled_all = client.post("/mcp/external-servers/reconcile")
        reconciled_one = client.post("/mcp/external-servers/research/reconcile")
        installed = client.post("/mcp/external-servers/research/install")
        updated = client.post("/mcp/external-servers/research/update")

    assert listed.status_code == 200
    assert listed.json()["servers"] == [
        {
            "id": "research",
            "status": "healthy",
            "installer": {
                "available": False,
                "reason_code": "external_server_installer_not_configured",
                "server_id": "research",
            },
        }
    ]
    assert started.json()["reason_code"] == "external_server_started"
    assert stopped.json()["reason_code"] == "external_server_stopped"
    assert refreshed_all.json()["server_id"] is None
    assert refreshed_one.json()["server_id"] == "research"
    assert reconciled_all.json()["server_id"] is None
    assert reconciled_one.json()["server_id"] == "research"
    assert installed.json()["reason_code"] == "external_server_install_not_configured"
    assert updated.json()["reason_code"] == "external_server_update_not_configured"
    assert manager.calls == [
        ("list_runtime_servers", (), {}),
        ("start_server", ("research",), {}),
        ("stop_server", ("research",), {}),
        ("refresh_server", (None,), {}),
        ("refresh_server", ("research",), {}),
        ("reconcile", (None,), {}),
        ("reconcile", ("research",), {}),
        ("install_server", ("research",), {"context": None}),
        ("update_server", ("research",), {"context": None}),
    ]


def test_gateway_external_runtime_management_routes_mount_with_bootstrap() -> None:
    manager = _ExternalRuntimeManagerDouble()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_bootstrap=_ExternalRuntimeBootstrapDouble(manager),
    )

    with TestClient(app) as client:
        response = client.get("/mcp/external-servers/runtime")

    assert response.status_code == 200
    assert manager.calls == [("list_runtime_servers", (), {})]


def test_gateway_external_runtime_routes_have_pydantic_response_models() -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_runtime_manager=_ExternalRuntimeManagerDouble(),
    )

    paths = app.openapi()["paths"]
    expected_refs = {
        (
            "/mcp/external-servers/runtime",
            "get",
        ): "#/components/schemas/ExternalRuntimeServerListResponse",
        (
            "/mcp/external-servers/{server_id}/start",
            "post",
        ): "#/components/schemas/ExternalRuntimeOperationResponse",
        (
            "/mcp/external-servers/{server_id}/stop",
            "post",
        ): "#/components/schemas/ExternalRuntimeOperationResponse",
        (
            "/mcp/external-servers/{server_id}/restart",
            "post",
        ): "#/components/schemas/ExternalRuntimeOperationResponse",
        (
            "/mcp/external-servers/refresh",
            "post",
        ): "#/components/schemas/ExternalRuntimeOperationResponse",
        (
            "/mcp/external-servers/{server_id}/refresh",
            "post",
        ): "#/components/schemas/ExternalRuntimeOperationResponse",
        (
            "/mcp/external-servers/reconcile",
            "post",
        ): "#/components/schemas/ExternalRuntimeOperationResponse",
        (
            "/mcp/external-servers/{server_id}/reconcile",
            "post",
        ): "#/components/schemas/ExternalRuntimeOperationResponse",
        (
            "/mcp/external-servers/{server_id}/install",
            "post",
        ): "#/components/schemas/ExternalRuntimeOperationResponse",
        (
            "/mcp/external-servers/{server_id}/update",
            "post",
        ): "#/components/schemas/ExternalRuntimeOperationResponse",
    }

    for (path, method), expected_ref in expected_refs.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": expected_ref}


def test_gateway_external_runtime_response_models_document_installer_fields() -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_runtime_manager=_ExternalRuntimeManagerDouble(),
    )

    schemas = app.openapi()["components"]["schemas"]

    status_props = schemas["ExternalRuntimeServerStatusResponse"]["properties"]
    operation_props = schemas["ExternalRuntimeOperationResponse"]["properties"]
    assert "installer" in status_props
    assert "available" in operation_props


def test_gateway_external_runtime_management_enabled_without_manager_raises() -> None:
    with pytest.raises(ValueError, match="external runtime management requires"):
        gateway_fastapi.create_gateway_router(
            _FakeGatewayRuntime(),
            enable_external_runtime_management=True,
        )

    with pytest.raises(ValueError, match="external runtime management requires"):
        create_gateway_app(
            _FakeGatewayRuntime(),
            prefix="/mcp",
            enable_external_runtime_management=True,
        )


def test_gateway_external_runtime_lifecycle_is_disabled_by_default() -> None:
    manager = _ExternalRuntimeManagerDouble()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_runtime_manager=manager,
    )

    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    assert manager.calls == []
    assert not hasattr(app.state, "external_runtime_startup")
    assert not hasattr(app.state, "external_runtime_shutdown")


def test_gateway_external_runtime_lifecycle_reconciles_on_startup() -> None:
    from mcp_unified.gateway.lifecycle import GatewayExternalRuntimeLifecycleConfig

    manager = _ExternalRuntimeManagerDouble()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_runtime_manager=manager,
        external_runtime_lifecycle=GatewayExternalRuntimeLifecycleConfig(
            reconcile_on_startup=True,
        ),
    )

    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    assert manager.calls == [("reconcile", (None,), {})]
    assert app.state.external_runtime_startup == {
        "ok": True,
        "reason_code": "external_server_reconciled",
        "server_id": None,
    }
    assert not hasattr(app.state, "external_runtime_shutdown")


def test_gateway_external_runtime_lifecycle_uses_bootstrap_config_on_startup() -> None:
    from mcp_unified.gateway.lifecycle import GatewayExternalRuntimeLifecycleConfig

    manager = _ExternalRuntimeManagerDouble()
    bootstrap = _ExternalRuntimeBootstrapDouble(
        manager,
        lifecycle=GatewayExternalRuntimeLifecycleConfig(reconcile_on_startup=True),
    )
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_bootstrap=bootstrap,
    )

    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    assert manager.calls == [("reconcile", (None,), {})]
    assert app.state.external_runtime_startup["reason_code"] == "external_server_reconciled"


def test_gateway_external_runtime_lifecycle_startup_failure_payload_does_not_block_status() -> None:
    from mcp_unified.gateway.lifecycle import GatewayExternalRuntimeLifecycleConfig

    manager = _ExternalRuntimeReconcileFailureManagerDouble()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_runtime_manager=manager,
        external_runtime_lifecycle=GatewayExternalRuntimeLifecycleConfig(
            reconcile_on_startup=True,
        ),
    )

    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    assert manager.calls == [("reconcile", (None,), {})]
    assert app.state.external_runtime_startup["ok"] is False
    assert app.state.external_runtime_startup["errors"] == {
        "research": "external_server_start_failed",
    }


def test_gateway_external_runtime_lifecycle_logs_unexpected_startup_exception(
    monkeypatch: Any,
) -> None:
    from mcp_unified.gateway.lifecycle import GatewayExternalRuntimeLifecycleConfig

    manager = _ExternalRuntimeLifecycleExceptionManagerDouble()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_runtime_manager=manager,
        external_runtime_lifecycle=GatewayExternalRuntimeLifecycleConfig(
            reconcile_on_startup=True,
        ),
    )
    fake_logger = _FakeLogger()
    monkeypatch.setattr(gateway_fastapi, "logger", fake_logger, raising=False)

    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    assert app.state.external_runtime_startup == {
        "ok": False,
        "reason_code": "external_runtime_startup_failed",
        "error_type": "RuntimeError",
        "error": "External runtime lifecycle operation failed",
    }
    assert fake_logger.opt_calls == [{"exception": True}]
    assert fake_logger.error_calls == [
        (
            "External runtime lifecycle operation failed reason_code={!r} error_type={!r}",
            ("external_runtime_startup_failed", "RuntimeError"),
        )
    ]


def test_gateway_status_includes_package_boundary_metadata() -> None:
    app = create_gateway_app(_FakeGatewayRuntime())
    status_route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == "/mcp/status"
    )
    assert getattr(status_route, "response_model", None) is gateway_fastapi.GatewayReadinessStatusResponse

    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["name"] == "unit-gateway"
    assert payload["version"] == "0.0-test"
    assert payload["package"]["package_status"] == "public-alpha"
    assert payload["package"]["publishing_status"] == "published"
    assert payload["package"]["source_distribution"] == "tldw-server"
    assert payload["transport"]["mount_path"] == "/mcp"
    assert "next_actions" in payload
    assert not any(
        warning["reason_code"] == "package_not_published"
        for warning in payload["warnings"]
    )


def test_gateway_status_generic_readiness_warnings_do_not_leak_exception_types(
    monkeypatch,
) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr(gateway_fastapi, "logger", fake_logger)
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        profile_manager=_ProfileManagementRuntimeErrorManagerDouble("runtime-error"),
        enable_profile_management=True,
        external_registry_manager=_ExternalRegistryRuntimeErrorManagerDouble("runtime-error"),
        enable_external_registry_management=True,
    )

    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    payload = response.json()
    warning_messages = {
        warning["reason_code"]: warning["message"]
        for warning in payload["warnings"]
    }
    assert warning_messages["default_profile_status_unavailable"] == (
        "Default profile readiness check failed."
    )
    assert warning_messages["external_registry_status_unavailable"] == (
        "External registry readiness check failed."
    )
    serialized_payload = json.dumps(payload)
    assert "RuntimeError" not in serialized_payload
    assert "profile backend leaked detail" not in serialized_payload
    assert "external registry leaked detail" not in serialized_payload
    assert fake_logger.opt_calls == [{"exception": True}, {"exception": True}]
    assert fake_logger.warning_calls == [
        ("Gateway default profile readiness check failed", ()),
        ("Gateway external registry readiness check failed", ()),
    ]


def test_gateway_status_route_has_pydantic_response_model() -> None:
    app = create_gateway_app(_FakeGatewayRuntime())

    schema = app.openapi()["paths"]["/mcp/status"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]

    assert schema == {"$ref": "#/components/schemas/GatewayReadinessStatusResponse"}


def test_gateway_status_tolerates_missing_package_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        gateway_fastapi,
        "package_metadata_summary",
        lambda: {"package_name": "mcp-unified"},
    )
    app = create_gateway_app(_FakeGatewayRuntime())

    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["package"]["package_name"] == "mcp-unified"
    assert payload["package"]["publishing_status"] is None
    assert not any(warning["reason_code"] == "package_not_published" for warning in payload["warnings"])


@pytest.mark.asyncio
async def test_gateway_readiness_status_handles_missing_admin_auth() -> None:
    payload = await gateway_fastapi._gateway_readiness_status(
        _FakeGatewayRuntime(),
        profile_manager=None,
        external_registry_manager=None,
        admin_auth=None,
        status_path="/mcp/status",
    )

    assert payload["admin_auth"] == {
        "enabled": False,
        "configured": False,
        "header_name": None,
    }


def test_gateway_status_reports_profile_store_admin_auth_and_default_profile() -> None:
    manager = _ProfileManagementManagerDouble("default")
    manager.store_metadata = type(
        "_StoreMetadata",
        (),
        {"to_payload": lambda self: {"kind": "memory", "persistent": False}},
    )()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        profile_manager=manager,
        enable_profile_management=True,
        admin_auth=GatewayAdminAuthConfig(enabled=True, api_key="unit-test-key"),
    )
    with TestClient(app) as client:
        response = client.get("/mcp/status")

    payload = response.json()
    assert payload["profile_store"]["kind"] in {"memory", "sqlite"}
    assert payload["default_profile"]["configured"] is True
    assert payload["admin_auth"]["enabled"] is True
    assert payload["admin_auth"]["configured"] is True
    assert "unit-test-key" not in json.dumps(payload)


def test_gateway_status_uses_generic_profile_warning_for_unexpected_failures() -> None:
    class _UnexpectedProfileFailureManager(_ProfileManagementManagerDouble):
        async def get_default_profile(self) -> dict[str, Any]:
            raise RuntimeError("profile store exploded")

    app = create_gateway_app(
        _FakeGatewayRuntime(),
        profile_manager=_UnexpectedProfileFailureManager("default"),
        enable_profile_management=True,
    )
    with TestClient(app) as client:
        payload = client.get("/mcp/status").json()

    warning = next(
        item
        for item in payload["warnings"]
        if item["reason_code"] == "default_profile_status_unavailable"
    )
    assert warning["message"] == "Default profile readiness check failed."
    assert "RuntimeError" not in json.dumps(payload)


def test_gateway_status_counts_external_servers_best_effort() -> None:
    manager = _ExternalRegistryManagerDouble("enabled")
    manager.store_metadata = type(
        "_StoreMetadata",
        (),
        {"to_payload": lambda self: {"kind": "memory", "persistent": False}},
    )()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        external_registry_manager=manager,
        enable_external_registry_management=True,
    )
    with TestClient(app) as client:
        payload = client.get("/mcp/status").json()

    assert payload["external_servers"]["total"] >= 1
    assert payload["external_servers"]["enabled"] >= 1
    assert payload["external_registry_store"]["kind"] in {"memory", "sqlite"}
    assert manager.calls == [("list_servers", (), {"enabled": None})]


def test_gateway_status_uses_generic_external_registry_warning_for_unexpected_failures() -> None:
    class _UnexpectedExternalRegistryFailureManager(_ExternalRegistryManagerDouble):
        async def list_servers(self, enabled: bool | None = None) -> dict[str, Any]:
            raise RuntimeError("external registry exploded")

    app = create_gateway_app(
        _FakeGatewayRuntime(),
        external_registry_manager=_UnexpectedExternalRegistryFailureManager("enabled"),
        enable_external_registry_management=True,
    )
    with TestClient(app) as client:
        payload = client.get("/mcp/status").json()

    warning = next(
        item
        for item in payload["warnings"]
        if item["reason_code"] == "external_registry_status_unavailable"
    )
    assert warning["message"] == "External registry readiness check failed."
    assert "RuntimeError" not in json.dumps(payload)


def test_gateway_external_runtime_lifecycle_stops_on_shutdown() -> None:
    from mcp_unified.gateway.lifecycle import GatewayExternalRuntimeLifecycleConfig

    manager = _ExternalRuntimeManagerDouble()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_runtime_manager=manager,
        external_runtime_lifecycle=GatewayExternalRuntimeLifecycleConfig(
            stop_on_shutdown=True,
        ),
    )

    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    assert manager.calls == [("stop_all", (), {})]
    assert app.state.external_runtime_shutdown == {
        "ok": True,
        "reason_code": "external_runtime_stopped",
        "stopped_servers": 1,
        "total_servers": 1,
        "errors": {},
    }


@pytest.mark.parametrize(
    ("reason_code", "status_code"),
    [
        ("external_server_not_found", 404),
        ("external_server_disabled", 409),
        ("credential_broker_unavailable", 503),
        ("external_server_transport_unavailable", 503),
        ("external_virtual_tool_not_found", 404),
        ("external_tool_call_failed", 503),
        ("invalid_external_runtime_request", 422),
        ("unexpected_external_runtime_reason", 500),
    ],
)
def test_gateway_external_runtime_management_error_status_mapping(
    reason_code: str,
    status_code: int,
) -> None:
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        external_runtime_manager=_ExternalRuntimeErrorManagerDouble(
            "start_server",
            reason_code,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/mcp/external-servers/research/start")

    assert response.status_code == status_code
    body = response.json()
    assert body["ok"] is False
    assert body["reason_code"] == reason_code
    assert body["server_id"] == "research"
    assert isinstance(body["error"], str)
    assert body["error"]
    assert "runtime failure" not in body["error"]


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

    _assert_profile_runtime_tool_names(
        listed.json()["result"]["tools"],
        backend_tools=["echo.search"],
    )
    assert allowed.json()["result"]["content"][0]["text"] == "echo.search:hello"
    assert backend.call_requests[-1][0] == "echo.search"
    denied_body = denied.json()
    _assert_jsonrpc_error(denied_body, code=-32001, request_id="call-denied")
    assert denied_body["error"]["data"]["reason_code"] == "tool_not_allowed"
    assert all(call[0] != "admin.delete" for call in backend.call_requests)


def test_gateway_profile_runtime_blocks_matching_path_permission_rule() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            {
                "name": "fs.read_text",
                "description": "Read a text file.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                "metadata": {"category": "filesystem", "capability": "filesystem.read"},
            }
        ]
    )
    profile = _profile_with_allowed_tools_and_permission_rules(
        "reviewer",
        allowed_tools=["fs.read_text"],
        permission_rules=[
            {
                "pattern": "Read(docs/private/**)",
                "outcome": "deny",
                "reason_code": "private_docs_denied",
            }
        ],
    )
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
                "params": {
                    "name": "fs.read_text",
                    "arguments": {"path": "docs/private/secret.txt", "query": "secret"},
                },
                "id": "path-rule-denied",
            },
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="path-rule-denied")
    error_data = body["error"]["data"]
    assert error_data["reason_code"] == "private_docs_denied"
    assert error_data["status"] == "denied"
    assert error_data["provenance"]["tool_name"] == "fs.read_text"
    assert error_data["provenance"]["subject_type"] == "path"
    assert error_data["provenance"]["matched_rules"][0]["pattern"] == "docs/private/**"
    assert "docs/private/secret.txt" not in json.dumps(error_data)
    assert backend.call_requests == []


def test_gateway_profile_runtime_blocks_matching_ask_permission_rule() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            {
                "name": "web.fetch",
                "description": "Fetch a URL.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
                "metadata": {"category": "web", "capability": "network.fetch"},
            }
        ]
    )
    profile = _profile_with_allowed_tools_and_permission_rules(
        "researcher",
        allowed_tools=["web.fetch"],
        permission_rules=[{"pattern": "WebFetch(example.com)", "outcome": "ask"}],
    )
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
                "params": {
                    "name": "web.fetch",
                    "arguments": {"url": "https://example.com/private", "query": "private"},
                },
                "id": "domain-rule-ask",
            },
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="domain-rule-ask")
    error_data = body["error"]["data"]
    assert error_data["reason_code"] == "approval_required"
    assert error_data["status"] == "approval_required"
    assert error_data["provenance"]["subject_type"] == "domain"
    assert "example.com/private" not in json.dumps(error_data)
    assert backend.call_requests == []


def test_gateway_profile_runtime_blocks_matching_mcp_permission_rule() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            {
                "name": "mcp__github__delete_repo",
                "description": "Delete a GitHub repository.",
                "inputSchema": {"type": "object", "properties": {}},
                "metadata": {"category": "external", "capability": "repo.delete"},
            }
        ]
    )
    profile = _profile_with_allowed_tools_and_permission_rules(
        "devops",
        allowed_tools=["mcp__github__delete_repo"],
        permission_rules=[
            {
                "pattern": "MCP__GitHub__delete_*",
                "outcome": "deny",
                "reason_code": "mcp_delete_denied",
            }
        ],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="devops",
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = client.post(
            "/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "mcp__github__delete_repo", "arguments": {"query": "repo"}},
                "id": "mcp-rule-denied",
            },
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="mcp-rule-denied")
    error_data = body["error"]["data"]
    assert error_data["reason_code"] == "mcp_delete_denied"
    assert error_data["status"] == "denied"
    assert error_data["provenance"]["subject_type"] == "mcp"
    assert backend.call_requests == []


def _read_text_tool_descriptor() -> dict[str, Any]:
    return {
        "name": "fs.read_text",
        "description": "Read a text file.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "metadata": {"category": "filesystem", "capability": "filesystem.read"},
    }


def _post_tool_call(client: Any, name: str, arguments: dict[str, Any], request_id: str) -> Any:
    return client.post(
        "/mcp/request",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
            "id": request_id,
        },
    )


def test_gateway_profile_runtime_caches_compiled_permission_rules(monkeypatch) -> None:
    from mcp_unified.gateway import profile_runtime as profile_runtime_module
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_read_text_tool_descriptor()])
    profile = _profile_with_allowed_tools_and_permission_rules(
        "reviewer",
        allowed_tools=["fs.read_text"],
        permission_rules=[{"pattern": "Read(docs/private/**)", "outcome": "deny"}],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
    )
    compile_calls: list[Any] = []
    real_compile = profile_runtime_module.compile_permission_rules

    def _counting_compile(*args: Any, **kwargs: Any) -> Any:
        compile_calls.append(args)
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(profile_runtime_module, "compile_permission_rules", _counting_compile)
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        for request_id in ("cache-first", "cache-second"):
            allowed = _post_tool_call(
                client,
                "fs.read_text",
                {"path": "docs/public/notes.txt", "query": "notes"},
                request_id,
            )
            assert allowed.json().get("error") is None

    assert len(backend.call_requests) == 2
    assert len(compile_calls) == 1


def test_gateway_profile_runtime_recompiles_permission_rules_when_profile_updated() -> None:
    import asyncio
    from datetime import datetime, timezone

    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.models import MCPProfile, ProfilePolicy
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_read_text_tool_descriptor()])
    first_version = MCPProfile(
        id="reviewer",
        name="Profile reviewer",
        policy_document=ProfilePolicy(
            allowed_tools=["fs.read_text"],
            permission_rules=[{"pattern": "Read(docs/private/**)", "outcome": "deny"}],
        ),
        updated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    store = InMemoryProfileStore([first_version])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=store,
        default_profile_id="reviewer",
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        allowed = _post_tool_call(
            client,
            "fs.read_text",
            {"path": "docs/public/notes.txt", "query": "notes"},
            "before-update",
        )
        assert allowed.json().get("error") is None

        second_version = first_version.model_copy(
            update={
                "policy_document": ProfilePolicy(
                    allowed_tools=["fs.read_text"],
                    permission_rules=[
                        {
                            "pattern": "Read(docs/public/**)",
                            "outcome": "deny",
                            "reason_code": "public_docs_denied",
                        }
                    ],
                ),
                "updated_at": datetime(2026, 6, 2, tzinfo=timezone.utc),
            }
        )
        asyncio.run(store.upsert_profile(second_version))

        denied = _post_tool_call(
            client,
            "fs.read_text",
            {"path": "docs/public/notes.txt", "query": "notes"},
            "after-update",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="after-update")
    assert body["error"]["data"]["reason_code"] == "public_docs_denied"
    assert len(backend.call_requests) == 1


def test_gateway_profile_runtime_permission_rule_cache_stays_bounded() -> None:
    from mcp_unified.gateway.profile_runtime import (
        _PERMISSION_RULE_CACHE_MAX_ENTRIES,
        ProfileAwareGatewayRuntime,
    )
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_read_text_tool_descriptor()])
    placeholder = _profile_with_allowed_tools("reviewer", ["fs.read_text"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([placeholder]),
        default_profile_id="reviewer",
    )

    for index in range(_PERMISSION_RULE_CACHE_MAX_ENTRIES + 16):
        profile = _profile_with_allowed_tools_and_permission_rules(
            f"profile-{index}",
            allowed_tools=["fs.read_text"],
            permission_rules=[{"pattern": "Read(docs/private/**)", "outcome": "deny"}],
        )
        assert runtime._compiled_permission_rules(profile)

    assert len(runtime._permission_rule_cache) == _PERMISSION_RULE_CACHE_MAX_ENTRIES


def test_gateway_profile_runtime_skips_permission_rules_for_missing_policy_document() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.models import MCPProfile
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_read_text_tool_descriptor()])
    placeholder = _profile_with_allowed_tools("reviewer", ["fs.read_text"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([placeholder]),
        default_profile_id="reviewer",
    )
    unvalidated = MCPProfile.model_construct(
        id="reviewer",
        name="Profile reviewer",
        policy_document=None,
    )

    assert runtime._compiled_permission_rules(unvalidated) == ()
    assert runtime._compiled_permission_rules(None) == ()


def test_gateway_profile_runtime_denies_oversized_permission_subject_count() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_read_text_tool_descriptor()])
    profile = _profile_with_allowed_tools_and_permission_rules(
        "reviewer",
        allowed_tools=["fs.read_text"],
        permission_rules=[{"pattern": "Read(docs/private/**)", "outcome": "deny"}],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = _post_tool_call(
            client,
            "fs.read_text",
            {"paths": [f"docs/public/file-{index}.txt" for index in range(200)], "query": "files"},
            "subject-count-limit",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="subject-count-limit")
    error_data = body["error"]["data"]
    assert error_data["reason_code"] == "permission_subject_limits_exceeded"
    assert error_data["status"] == "denied"
    assert error_data["provenance"]["tool_name"] == "fs.read_text"
    assert backend.call_requests == []


def test_gateway_profile_runtime_denies_oversized_permission_subject_value() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_read_text_tool_descriptor()])
    profile = _profile_with_allowed_tools_and_permission_rules(
        "reviewer",
        allowed_tools=["fs.read_text"],
        permission_rules=[{"pattern": "Read(docs/private/**)", "outcome": "deny"}],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
    )
    app = create_gateway_app(runtime, prefix="/mcp")
    oversized_path = "docs/public/" + ("a" * 5000)

    with TestClient(app) as client:
        denied = _post_tool_call(
            client,
            "fs.read_text",
            {"path": oversized_path, "query": "oversized"},
            "subject-value-limit",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="subject-value-limit")
    error_data = body["error"]["data"]
    assert error_data["reason_code"] == "permission_subject_limits_exceeded"
    assert error_data["status"] == "denied"
    assert oversized_path not in json.dumps(error_data)
    assert backend.call_requests == []


def test_gateway_profile_runtime_denies_oversized_permission_argv() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            {
                "name": "shell.run",
                "description": "Run a governed shell command.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"argv": {"type": "array"}},
                },
                "metadata": {"category": "shell", "capability": "shell.execute"},
            }
        ]
    )
    profile = _profile_with_allowed_tools_and_permission_rules(
        "operator",
        allowed_tools=["shell.run"],
        permission_rules=[{"pattern": "Bash(rm *)", "outcome": "deny"}],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="operator",
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = _post_tool_call(
            client,
            "shell.run",
            {"argv": ["echo"] * 300, "query": "argv"},
            "argv-token-limit",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="argv-token-limit")
    error_data = body["error"]["data"]
    assert error_data["reason_code"] == "permission_subject_limits_exceeded"
    assert error_data["status"] == "denied"
    assert backend.call_requests == []


def test_gateway_profile_runtime_allows_oversized_arguments_without_permission_rules() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_read_text_tool_descriptor()])
    profile = _profile_with_allowed_tools("reviewer", ["fs.read_text"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        allowed = _post_tool_call(
            client,
            "fs.read_text",
            {"paths": [f"docs/public/file-{index}.txt" for index in range(200)], "query": "files"},
            "rule-free-oversized",
        )

    assert allowed.json().get("error") is None
    assert len(backend.call_requests) == 1


def _web_fetch_tool_descriptor() -> dict[str, Any]:
    return {
        "name": "web.fetch",
        "description": "Fetch a URL.",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        "metadata": {"category": "web", "capability": "network.fetch"},
    }


def _skills_render_tool_descriptor() -> dict[str, Any]:
    return {
        "name": "skills.render",
        "description": "Render a Skill without executing it.",
        "inputSchema": {
            "type": "object",
            "properties": {"skill_name": {"type": "string"}},
            "required": ["skill_name"],
        },
        "metadata": {"category": "knowledge", "capability": "skills.render"},
    }


def _skill_rule_runtime_with_grant_store(permission_rules: list[Any]) -> tuple[Any, Any, Any]:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _SkillsRenderGatewayRuntime([_skills_render_tool_descriptor()])
    profile = _profile_with_allowed_tools_and_permission_rules(
        "reviewer",
        allowed_tools=["skills.render"],
        permission_rules=permission_rules,
    )
    grant_store = InMemoryPolicyGrantStore()
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
        policy_grant_store=grant_store,
    )
    return runtime, backend, grant_store


def _ask_rule_runtime_with_grant_store() -> tuple[Any, Any, Any]:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_web_fetch_tool_descriptor()])
    profile = _profile_with_allowed_tools_and_permission_rules(
        "researcher",
        allowed_tools=["web.fetch"],
        permission_rules=[{"pattern": "WebFetch(example.com)", "outcome": "ask"}],
    )
    grant_store = InMemoryPolicyGrantStore()
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="researcher",
        policy_grant_store=grant_store,
    )
    return runtime, backend, grant_store


def test_gateway_profile_runtime_blocks_skill_permission_denial() -> None:
    runtime, backend, _grant_store = _skill_rule_runtime_with_grant_store(
        [{"pattern": "Skill(secret-*)", "outcome": "deny"}]
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = _post_tool_call(
            client,
            "skills.render",
            {"skill_name": "secret-plan"},
            "skill-denied",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="skill-denied")
    assert body["error"]["data"]["status"] == "denied"
    assert body["error"]["data"]["provenance"]["subject_type"] == "skill"
    assert backend.call_requests == []


def test_gateway_profile_runtime_skill_permission_ask_requires_approval() -> None:
    runtime, backend, _grant_store = _skill_rule_runtime_with_grant_store(
        [{"pattern": "Skill(review-*)", "outcome": "ask"}]
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = _post_tool_call(
            client,
            "skills.render",
            {"skill_name": "Review-Paper"},
            "skill-ask",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="skill-ask")
    assert body["error"]["data"]["status"] == "approval_required"
    assert body["error"]["data"]["provenance"]["approval"]["subject_type"] == "skill"
    assert backend.call_requests == []


def test_gateway_profile_runtime_skill_approval_lease_delegates_with_redacted_marker() -> None:
    runtime, backend, grant_store = _skill_rule_runtime_with_grant_store(
        [{"pattern": "Skill(review-*)", "outcome": "ask"}]
    )
    grant = grant_store.create_grant(
        profile_id="reviewer",
        grant_type="approval",
        subject_type="skill",
        value="REVIEW-PAPER",
        ttl_seconds=900,
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        allowed = _post_tool_call(
            client,
            "skills.render",
            {"skill_name": "Review-Paper"},
            "skill-lease",
        )

    assert allowed.json().get("error") is None
    assert len(backend.call_requests) == 1
    markers = backend.call_requests[-1][2].metadata.get("mcp_policy_approval_grants")
    assert markers
    assert grant.grant_id not in json.dumps(markers)


def test_gateway_profile_runtime_ask_denial_reports_approval_availability() -> None:
    runtime, backend, _grant_store = _ask_rule_runtime_with_grant_store()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = _post_tool_call(
            client,
            "web.fetch",
            {"url": "https://example.com/private", "query": "private"},
            "ask-no-lease",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="ask-no-lease")
    error_data = body["error"]["data"]
    assert error_data["status"] == "approval_required"
    approval = error_data["provenance"]["approval"]
    assert approval["available"] is True
    assert approval["subject_type"] == "domain"
    assert "example.com/private" not in json.dumps(error_data)
    assert backend.call_requests == []


def test_gateway_profile_runtime_active_approval_lease_allows_ask_decision() -> None:
    runtime, backend, grant_store = _ask_rule_runtime_with_grant_store()
    grant = grant_store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=900,
        granted_by="operator",
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        allowed = _post_tool_call(
            client,
            "web.fetch",
            {"url": "https://example.com/private", "query": "private"},
            "ask-with-lease",
        )

    assert allowed.json().get("error") is None
    assert len(backend.call_requests) == 1
    delegated_context = backend.call_requests[-1][2]
    markers = delegated_context.metadata.get("mcp_policy_approval_grants")
    assert markers
    assert grant.grant_id not in json.dumps(markers)


def test_gateway_profile_runtime_expired_approval_lease_blocks_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp_unified.policy_grants.memory as memory_grants

    runtime, backend, grant_store = _ask_rule_runtime_with_grant_store()
    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_000.0)
    grant_store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=10,
    )
    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_011.0)
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = _post_tool_call(
            client,
            "web.fetch",
            {"url": "https://example.com/private", "query": "private"},
            "ask-expired-lease",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="ask-expired-lease")
    assert body["error"]["data"]["status"] == "approval_required"
    assert backend.call_requests == []


def test_gateway_profile_runtime_approval_lease_never_overrides_deny_rule() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_read_text_tool_descriptor()])
    profile = _profile_with_allowed_tools_and_permission_rules(
        "reviewer",
        allowed_tools=["fs.read_text"],
        permission_rules=[
            {
                "pattern": "Read(docs/private/**)",
                "outcome": "deny",
                "reason_code": "private_docs_denied",
            }
        ],
    )
    grant_store = InMemoryPolicyGrantStore()
    grant_store.create_grant(
        profile_id="reviewer",
        grant_type="approval",
        subject_type="path",
        value="docs/private/secret.txt",
        ttl_seconds=900,
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
        policy_grant_store=grant_store,
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = _post_tool_call(
            client,
            "fs.read_text",
            {"path": "docs/private/secret.txt", "query": "secret"},
            "deny-with-lease",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="deny-with-lease")
    assert body["error"]["data"]["reason_code"] == "private_docs_denied"
    assert body["error"]["data"]["status"] == "denied"
    assert backend.call_requests == []


def test_gateway_profile_runtime_session_scoped_lease_only_matches_its_session() -> None:
    import asyncio

    from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext

    runtime, backend, grant_store = _ask_rule_runtime_with_grant_store()
    grant_store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=900,
        session_id="session-1",
    )
    arguments = {"url": "https://example.com/private", "query": "private"}

    matching_context = GatewayRequestContext(
        request_id="session-match",
        metadata={"session_id": "session-1"},
    )
    result = asyncio.run(runtime.call_tool("web.fetch", arguments, matching_context))
    assert result is not None
    assert len(backend.call_requests) == 1

    mismatched_context = GatewayRequestContext(
        request_id="session-mismatch",
        metadata={"session_id": "session-2"},
    )
    with pytest.raises(GatewayPolicyDenied):
        asyncio.run(runtime.call_tool("web.fetch", arguments, mismatched_context))
    assert len(backend.call_requests) == 1


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

    _assert_profile_runtime_tool_names(tools, backend_tools=["echo.search"])


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

    tools = asyncio.run(runtime.list_tools(GatewayRequestContext(request_id="non-list-tools")))

    _assert_profile_runtime_tool_names(tools, backend_tools=[])


def test_profile_runtime_exposes_discovery_bridge_tools_for_deferred_categories() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            {
                "name": "browser.snapshot",
                "description": "Inspect browser DOM.",
                "metadata": {
                    "capability": "browser.inspect",
                    "category": "browser",
                },
            }
        ]
    )
    profile = _profile_with_tooling_metadata(
        "frontend",
        capabilities=["browser.inspect"],
        deferred_categories=["browser"],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="frontend",
    )

    tools = asyncio.run(runtime.list_tools(GatewayRequestContext(request_id="bridge-list")))
    descriptors = {tool["name"]: tool for tool in tools}

    _assert_profile_runtime_tool_names(
        tools,
        backend_tools=[],
        includes_tool_call=True,
    )
    for name in (
        *PROFILE_DISCOVERY_READ_TOOL_NAMES,
        PROFILE_DISCOVERY_CALL_TOOL_NAME,
    ):
        descriptor = descriptors[name]
        assert descriptor["metadata"]["category"] == "tool_discovery"
        assert "tool_discovery.read" in descriptor["metadata"]["capabilities"]
        assert descriptor["inputSchema"]["type"] == "object"
    assert descriptors["tool_call"]["inputSchema"]["required"] == [
        "tool_id",
        "arguments",
    ]
    assert descriptors["tool_call"]["inputSchema"]["additionalProperties"] is False


def test_profile_runtime_omits_tool_call_without_deferred_categories() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _MultiToolGatewayRuntime()
    profile = _profile_with_tooling_metadata(
        "researcher",
        capabilities=["code_search"],
        direct_categories=["code"],
        deferred_categories=[],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="researcher",
    )

    tools = asyncio.run(runtime.list_tools(GatewayRequestContext(request_id="bridge-read-only")))

    _assert_profile_runtime_tool_names(tools, backend_tools=["echo.search"])


def test_profile_runtime_omits_tool_call_for_recommendations_without_deferred_categories() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([])
    profile = _profile_with_tooling_metadata(
        "frontend",
        allowed_tools=["browser.trace"],
        recommended_tools=[
            {
                "id": "browser.trace",
                "category": "browser",
                "description": "Browser trace capture.",
                "activation": "requires_browser_runtime",
            }
        ],
        direct_categories=["code"],
        deferred_categories=[],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="frontend",
    )
    context = GatewayRequestContext(request_id="recommendation-only-tools")

    tools = asyncio.run(runtime.list_tools(context))
    catalog = asyncio.run(runtime.call_tool("profile.tools.list", {}, context))

    _assert_profile_runtime_tool_names(tools, backend_tools=[])
    assert [tool["tool_id"] for tool in catalog["tools"]] == ["browser.trace"]
    assert catalog["tools"][0]["installation_status"] == "recommended_unavailable"
    with pytest.raises(GatewayPolicyDenied) as exc_info:
        asyncio.run(
            runtime.call_tool(
                "tool_call",
                {"tool_id": "browser.trace", "arguments": {}},
                context,
            )
        )
    assert exc_info.value.reason_code == "tool_not_allowed"
    assert backend.call_requests == []


def test_profile_runtime_tool_search_bridge_returns_profile_scoped_results() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            {
                "name": "browser.snapshot",
                "description": "Browser DOM snapshot.",
                "metadata": {
                    "capability": "browser.inspect",
                    "category": "browser",
                },
            },
            {
                "name": "shell.run",
                "description": "Run shell commands.",
                "metadata": {
                    "capability": "process.execute",
                    "category": "shell",
                },
            },
        ]
    )
    profile = _profile_with_tooling_metadata(
        "frontend",
        capabilities=["browser.inspect"],
        recommended_tools=[
            {
                "id": "browser.trace",
                "category": "browser",
                "description": "Browser trace capture.",
                "capability": "browser.inspect",
                "activation": "requires_browser_runtime",
            }
        ],
        deferred_categories=["browser"],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="frontend",
    )
    context = GatewayRequestContext(request_id="bridge-search")

    categories = asyncio.run(runtime.call_tool("tool_categories.list", {}, context))
    catalog = asyncio.run(runtime.call_tool("profile.tools.list", {}, context))
    results = asyncio.run(
        runtime.call_tool(
            "tool_search",
            {"query": "browser", "category": "browser", "limit": 10},
            context,
        )
    )

    assert [category["category"] for category in categories["categories"]] == ["browser"]
    assert [tool["tool_id"] for tool in catalog["tools"]] == [
        "browser.snapshot",
        "browser.trace",
    ]
    assert [tool["tool_id"] for tool in results["tools"]] == [
        "browser.snapshot",
        "browser.trace",
    ]
    assert results["tools"][0]["installation_status"] == "installed"
    assert results["tools"][1]["installation_status"] == "recommended_unavailable"
    assert backend.call_requests == []


def test_profile_runtime_hides_deferred_installed_tools_from_initial_list() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            {
                "name": "code.search",
                "description": "Search code.",
                "metadata": {
                    "capability": "code_search",
                    "category": "code",
                },
            },
            {
                "name": "browser.snapshot",
                "description": "Browser DOM snapshot.",
                "metadata": {
                    "capability": "browser.inspect",
                    "category": "browser",
                },
            },
        ]
    )
    profile = _profile_with_tooling_metadata(
        "frontend",
        capabilities=["code_search", "browser.inspect"],
        direct_categories=["code"],
        deferred_categories=[],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="frontend",
    )

    tools = asyncio.run(runtime.list_tools(GatewayRequestContext(request_id="direct-only")))

    _assert_profile_runtime_tool_names(
        tools,
        backend_tools=["code.search"],
        includes_tool_call=True,
    )
    assert "browser.snapshot" not in _listed_tool_names(tools)


def test_profile_runtime_search_and_calls_deferred_installed_tools() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            {
                "name": "code.search",
                "description": "Search code.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "metadata": {
                    "capability": "code_search",
                    "category": "code",
                },
            },
            {
                "name": "browser.snapshot",
                "description": "Browser DOM snapshot.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "metadata": {
                    "capability": "browser.inspect",
                    "category": "browser",
                },
            },
        ]
    )
    profile = _profile_with_tooling_metadata(
        "frontend",
        capabilities=["code_search", "browser.inspect"],
        direct_categories=["code"],
        deferred_categories=["browser"],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="frontend",
    )
    context = GatewayRequestContext(request_id="deferred-search-call")

    tools = asyncio.run(runtime.list_tools(context))
    catalog = asyncio.run(runtime.call_tool("profile.tools.list", {}, context))
    results = asyncio.run(
        runtime.call_tool(
            "tool_search",
            {"query": "browser", "category": "browser", "limit": 10},
            context,
        )
    )
    payload = asyncio.run(
        runtime.call_tool(
            "tool_call",
            {"tool_id": "browser.snapshot", "arguments": {"query": "dom"}},
            context,
        )
    )

    catalog_entries = {tool["tool_id"]: tool for tool in catalog["tools"]}
    _assert_profile_runtime_tool_names(
        tools,
        backend_tools=["code.search"],
        includes_tool_call=True,
    )
    assert catalog_entries["code.search"]["exposure"] == "direct"
    assert catalog_entries["browser.snapshot"]["exposure"] == "deferred"
    assert [tool["tool_id"] for tool in results["tools"]] == ["browser.snapshot"]
    assert results["tools"][0]["installation_status"] == "installed"
    assert results["tools"][0]["exposure"] == "deferred"
    assert payload["content"][0]["text"] == "browser.snapshot:dom"
    assert backend.call_requests[-1][0] == "browser.snapshot"


def test_profile_runtime_tool_describe_bridge_hides_denied_tools() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _MultiToolGatewayRuntime()
    profile = _profile_with_capabilities("researcher", ["code_search"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="researcher",
    )

    payload = asyncio.run(
        runtime.call_tool(
            "tool_describe",
            {"tool_id": "admin.delete"},
            GatewayRequestContext(request_id="bridge-describe-denied"),
        )
    )

    assert payload["error"]["reason_code"] == "tool_not_found"
    assert payload["tool_id"] == "admin.delete"
    assert backend.call_requests == []


def test_profile_runtime_tool_call_rejects_recommended_unavailable_tool() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([])
    profile = _profile_with_tooling_metadata(
        "frontend",
        allowed_tools=["browser.trace"],
        recommended_tools=[
            {
                "id": "browser.trace",
                "category": "browser",
                "description": "Browser trace capture.",
                "activation": "requires_browser_runtime",
            }
        ],
        deferred_categories=["browser"],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="frontend",
    )

    payload = asyncio.run(
        runtime.call_tool(
            "tool_call",
            {"tool_id": "browser.trace", "arguments": {}},
            GatewayRequestContext(request_id="bridge-call-unavailable"),
        )
    )

    assert payload["error"]["reason_code"] == "tool_not_enabled"
    assert payload["tool_id"] == "browser.trace"
    assert backend.call_requests == []


def test_profile_runtime_tool_call_delegates_installed_tool_through_policy() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _MultiToolGatewayRuntime()
    profile = _profile_with_tooling_metadata(
        "researcher",
        capabilities=["code_search"],
        deferred_categories=["test"],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="researcher",
    )

    payload = asyncio.run(
        runtime.call_tool(
            "tool_call",
            {"tool_id": "echo.search", "arguments": {"query": "bridge"}},
            GatewayRequestContext(request_id="bridge-call-installed"),
        )
    )

    assert payload["content"][0]["text"] == "echo.search:bridge"
    assert backend.call_requests[-1][0] == "echo.search"
    assert backend.call_requests[-1][1] == {"query": "bridge"}


def test_profile_runtime_backend_tool_with_bridge_name_wins_collision() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime(
        [
            {
                "name": "tool_search",
                "description": "Backend search tool.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "metadata": {"category": "test"},
            },
            {
                "name": "admin.delete",
                "description": "Denied admin tool.",
                "metadata": {"category": "admin"},
            },
        ]
    )
    profile = _profile_with_allowed_tools("collision", ["tool_search"])
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="collision",
    )
    context = GatewayRequestContext(request_id="bridge-name-collision")

    tools = asyncio.run(runtime.list_tools(context))
    descriptors = {tool["name"]: tool for tool in tools}
    payload = asyncio.run(
        runtime.call_tool(
            "tool_search",
            {"query": "backend"},
            context,
        )
    )

    _assert_profile_runtime_tool_names(tools, backend_tools=["tool_search"])
    assert descriptors["tool_search"]["description"] == "Backend search tool."
    assert descriptors["tool_search"]["metadata"]["category"] == "test"
    assert payload["content"][0]["text"] == "tool_search:backend"
    assert backend.call_requests[-1][0] == "tool_search"
    assert all(tool["name"] != "admin.delete" for tool in tools)


@pytest.mark.parametrize(
    "arguments",
    [
        {"arguments": {}},
        {"tool_id": "echo.search"},
        {"tool_id": 123, "arguments": {}},
        {"tool_id": "echo.search", "arguments": []},
        {"tool_id": "echo.search", "arguments": {}, "extra": True},
        {"tool_id": "echo.search", "arguments": {}, "extra": True, 1: True},
    ],
)
def test_profile_runtime_tool_call_rejects_invalid_arguments(
    arguments: Any,
) -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _MultiToolGatewayRuntime()
    profile = _profile_with_tooling_metadata(
        "researcher",
        capabilities=["code_search"],
        deferred_categories=["test"],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="researcher",
    )

    with pytest.raises(GatewayPolicyDenied) as exc_info:
        asyncio.run(
            runtime.call_tool(
                "tool_call",
                arguments,
                GatewayRequestContext(request_id="bridge-call-invalid"),
            )
        )

    assert exc_info.value.reason_code == "invalid_tool_call_arguments"
    assert backend.call_requests == []


def test_profile_runtime_tool_call_validates_invalid_arguments_before_backend_failure() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomExplodingGatewayRuntime()
    profile = _profile_with_tooling_metadata(
        "researcher",
        capabilities=["code_search"],
        deferred_categories=["test"],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="researcher",
    )

    with pytest.raises(GatewayPolicyDenied) as exc_info:
        asyncio.run(
            runtime.call_tool(
                "tool_call",
                {"extra": True, 1: True},
                GatewayRequestContext(request_id="bridge-call-invalid-before-backend"),
            )
        )

    assert exc_info.value.reason_code == "invalid_tool_call_arguments"
    assert backend.call_requests == []


def test_gateway_profile_bootstrap_seeds_default_builtin_preset_profile() -> None:
    from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway
    from mcp_unified.gateway.profiles import GatewayProfileManager
    from mcp_unified.profiles.store import InMemoryProfileAssignmentStore

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
    assert isinstance(bootstrap.assignment_store, InMemoryProfileAssignmentStore)
    assert bootstrap.audit_store is None
    assert isinstance(bootstrap.profile_manager, GatewayProfileManager)
    assert bootstrap.profile_manager.assignment_store is bootstrap.assignment_store
    assert bootstrap.store_metadata.to_payload() == {
        "kind": "memory",
        "persistent": False,
    }
    _assert_profile_runtime_tool_names(
        listed.json()["result"]["tools"],
        backend_tools=["echo.search"],
        includes_tool_call=True,
    )
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
    _assert_profile_runtime_tool_names(
        listed.json()["result"]["tools"],
        backend_tools=["echo.search"],
    )


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
    _assert_profile_runtime_tool_names(
        listed.json()["result"]["tools"],
        backend_tools=["admin.delete"],
    )


def test_gateway_profile_bootstrap_manager_default_changes_runtime_without_restart() -> None:
    """Share default assignment state between bootstrap manager and runtime."""

    from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway

    backend = _MultiToolGatewayRuntime()
    reviewer = _profile_with_allowed_tools("reviewer", ["echo.search"])
    architect = _profile_with_allowed_tools("architect", ["admin.delete"])
    bootstrap = asyncio.run(
        bootstrap_profile_gateway(
            backend,
            profiles=[reviewer, architect],
        )
    )
    app = create_gateway_app(bootstrap.runtime, prefix="/mcp")

    with TestClient(app) as client:
        asyncio.run(bootstrap.profile_manager.set_default_profile("reviewer"))
        first = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-default-1"},
        )

        asyncio.run(bootstrap.profile_manager.set_default_profile("architect"))
        second = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-default-2"},
        )

        explicit = client.post(
            "/mcp/request",
            headers={"X-MCP-Profile": "reviewer"},
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-explicit"},
        )

    _assert_profile_runtime_tool_names(
        first.json()["result"]["tools"],
        backend_tools=["echo.search"],
    )
    _assert_profile_runtime_tool_names(
        second.json()["result"]["tools"],
        backend_tools=["admin.delete"],
    )
    _assert_profile_runtime_tool_names(
        explicit.json()["result"]["tools"],
        backend_tools=["echo.search"],
    )


def test_gateway_profile_management_http_default_changes_runtime_without_restart() -> None:
    """Change the runtime default through HTTP management routes."""

    from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway

    backend = _MultiToolGatewayRuntime()
    reviewer = _profile_with_allowed_tools("reviewer", ["echo.search"])
    architect = _profile_with_allowed_tools("architect", ["admin.delete"])
    bootstrap = asyncio.run(
        bootstrap_profile_gateway(
            backend,
            profiles=[reviewer, architect],
        )
    )
    app = create_gateway_app(
        bootstrap.runtime,
        prefix="/mcp",
        profile_bootstrap=bootstrap,
    )

    with TestClient(app) as client:
        first_default = client.put(
            "/mcp/profiles/default",
            json={"profile_id": "reviewer"},
        )
        first_tools = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-http-default-1"},
        )
        second_default = client.put(
            "/mcp/profiles/default",
            json={"profile_id": "architect"},
        )
        second_tools = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "tools-http-default-2"},
        )

    assert first_default.status_code == 200
    assert first_default.json()["default"]["profile_id"] == "reviewer"
    _assert_profile_runtime_tool_names(
        first_tools.json()["result"]["tools"],
        backend_tools=["echo.search"],
    )
    assert second_default.status_code == 200
    assert second_default.json()["default"]["profile_id"] == "architect"
    _assert_profile_runtime_tool_names(
        second_tools.json()["result"]["tools"],
        backend_tools=["admin.delete"],
    )


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
    _assert_profile_runtime_tool_names(
        listed.json()["result"]["tools"],
        backend_tools=["echo.search"],
        includes_tool_call=True,
    )


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
    assert bootstrap.assignment_store is bootstrap.profile_store
    assert bootstrap.audit_store is bootstrap.profile_store
    assert bootstrap.profile_manager.profile_store is bootstrap.profile_store
    assert bootstrap.profile_manager.assignment_store is bootstrap.assignment_store
    assert bootstrap.profile_manager.audit_store is bootstrap.audit_store
    assert bootstrap.store_metadata.to_payload() == {
        "kind": "sqlite",
        "persistent": True,
    }
    assert sqlite_path.exists()
    assert [profile.id for profile in stored_profiles] == ["project-researcher"]
    asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_sqlite_bootstrap_exposes_external_registry_manager(
    tmp_path: Path,
) -> None:
    """Mount external registry routes from a real SQLite profile bootstrap."""

    from mcp_unified.gateway.config import (
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )

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
    app = create_gateway_app(
        bootstrap.runtime,
        prefix="/mcp",
        profile_bootstrap=bootstrap,
        enable_external_registry_management=True,
    )

    with TestClient(app) as client:
        response = client.get("/mcp/external-servers")

    assert bootstrap.external_registry_manager is not None
    assert bootstrap.external_runtime_manager is None
    assert bootstrap.external_registry_manager.external_registry_store is bootstrap.profile_store
    assert bootstrap.external_registry_manager.credential_grant_store is bootstrap.profile_store
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "servers": [],
        "store": {"kind": "sqlite", "persistent": True},
    }
    asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_builds_stdio_external_runtime_manager(
    tmp_path: Path,
) -> None:
    """Opt-in runtime config should wire the package stdio transport factory."""

    from mcp_unified.federation import create_external_transport
    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.gateway.external_runtime import GatewayExternalRuntimeManager

    sqlite_path = tmp_path / "gateway.db"
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                external_runtime=GatewayExternalRuntimeBootstrapConfig(enabled=True),
            ),
        )
    )

    try:
        manager = bootstrap.external_runtime_manager

        assert isinstance(manager, GatewayExternalRuntimeManager)
        assert manager._transport_factory is create_external_transport
        assert bootstrap.external_registry_manager is not None
        assert bootstrap.external_registry_manager.external_registry_store is bootstrap.profile_store
    finally:
        asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_default_factory_rejects_unsupported_transport(
    tmp_path: Path,
) -> None:
    """Unsupported server transports should fail at runtime start, not bootstrap."""

    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.storage.models import ExternalServerDefinition

    sqlite_path = tmp_path / "gateway.db"
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                external_runtime=GatewayExternalRuntimeBootstrapConfig(enabled=True),
            ),
        )
    )

    try:
        manager = bootstrap.external_runtime_manager
        assert manager is not None
        asyncio.run(
            bootstrap.profile_store.create_server(
                ExternalServerDefinition(
                    id="remote-docs",
                    name="Remote Docs",
                    transport="websocket",
                    url="wss://example.invalid/mcp",
                )
            )
        )

        with pytest.raises(GatewayExternalRuntimeError) as exc_info:
            asyncio.run(manager.start_server("remote-docs"))

        assert exc_info.value.reason_code == "external_server_start_failed"
    finally:
        asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_external_runtime_uses_injected_transport_factory(
    tmp_path: Path,
) -> None:
    """Caller-injected transport factories should override the package default."""

    from mcp_unified.federation.models import ExternalToolDefinition
    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.storage.models import ExternalServerDefinition

    class RecordingTransport:
        transport_name = "recording"

        def __init__(self, server_id: str) -> None:
            self.server_id = server_id
            self.connected = False
            self.close_count = 0

        async def connect(self) -> None:
            self.connected = True

        async def close(self) -> None:
            self.close_count += 1
            self.connected = False

        async def health_check(self) -> dict[str, bool]:
            return {
                "configured": True,
                "connected": self.connected,
                "initialized": self.connected,
            }

        async def list_tools(self) -> list[ExternalToolDefinition]:
            return [ExternalToolDefinition(name="search", description="Search")]

        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            context: Any = None,
            runtime_auth: Any = None,
        ) -> dict[str, Any]:
            del tool_name, arguments, context, runtime_auth
            return {"content": []}

    transports: list[RecordingTransport] = []

    def factory(server: ExternalServerDefinition) -> RecordingTransport:
        transport = RecordingTransport(server.id)
        transports.append(transport)
        return transport

    sqlite_path = tmp_path / "gateway.db"
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                external_runtime=GatewayExternalRuntimeBootstrapConfig(enabled=True),
            ),
            external_transport_factory=factory,
        )
    )

    try:
        manager = bootstrap.external_runtime_manager
        assert manager is not None
        asyncio.run(
            bootstrap.profile_store.create_server(
                ExternalServerDefinition(
                    id="research",
                    name="Research",
                    transport="stdio",
                    command=["fake-mcp-server"],
                )
            )
        )

        payload = asyncio.run(manager.start_server("research"))
        stop_payload = asyncio.run(manager.stop_server("research"))

        assert payload["ok"] is True
        assert payload["tool_count"] == 1
        assert stop_payload["reason_code"] == "external_server_stopped"
        assert [transport.server_id for transport in transports] == ["research"]
        assert transports[0].close_count == 1
    finally:
        asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_exposes_external_resources_through_runtime(
    tmp_path: Path,
) -> None:
    """Configured external runtime should be visible through the bootstrapped runtime."""

    from mcp_unified.federation.models import ExternalToolDefinition
    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.gateway.runtime import GatewayRequestContext
    from mcp_unified.storage.models import ExternalServerDefinition

    upstream_uri = "secret://docs/source?token=do-not-leak"

    class ResourceTransport:
        transport_name = "resource"

        def __init__(self, server_id: str) -> None:
            self.server_id = server_id
            self.connected = False

        async def connect(self) -> None:
            self.connected = True

        async def close(self) -> None:
            self.connected = False

        async def health_check(self) -> dict[str, bool]:
            return {
                "configured": True,
                "connected": self.connected,
                "initialized": self.connected,
            }

        async def list_tools(self) -> list[ExternalToolDefinition]:
            return [ExternalToolDefinition(name="search", description="Search")]

        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            context: Any = None,
            runtime_auth: Any = None,
        ) -> dict[str, Any]:
            del tool_name, arguments, context, runtime_auth
            return {"content": []}

        async def list_resources(self) -> list[dict[str, Any]]:
            return [{"uri": upstream_uri, "name": "Research Source"}]

        async def read_resource(
            self,
            uri: str,
            *,
            context: Any = None,
        ) -> dict[str, Any]:
            del context
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": f"read {uri}",
                    }
                ]
            }

    def factory(server: ExternalServerDefinition) -> ResourceTransport:
        return ResourceTransport(server.id)

    sqlite_path = tmp_path / "gateway.db"
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                profiles=[
                    {
                        "id": "researcher",
                        "name": "Researcher",
                        "external_server_grants": [{"server_id": "research"}],
                    }
                ],
                default_profile_id="researcher",
                external_runtime=GatewayExternalRuntimeBootstrapConfig(enabled=True),
            ),
            external_transport_factory=factory,
        )
    )

    try:
        manager = bootstrap.external_runtime_manager
        assert manager is not None
        asyncio.run(
            bootstrap.profile_store.create_server(
                ExternalServerDefinition(
                    id="research",
                    name="Research",
                    transport="stdio",
                    command=["fake-mcp-server"],
                )
            )
        )
        asyncio.run(manager.start_server("research"))

        context = GatewayRequestContext(request_id="config-resource", user_id="user-1")
        resources = asyncio.run(bootstrap.runtime.list_resources(context))
        external_uri = next(
            resource["uri"]
            for resource in resources
            if resource.get("metadata", {}).get("external_server_id") == "research"
        )
        result = asyncio.run(bootstrap.runtime.read_resource(external_uri, context))

        payload = json.dumps(result, sort_keys=True)
        assert external_uri.startswith("external://research/")
        assert "do-not-leak" not in payload
        assert result["contents"][0]["uri"] == external_uri
    finally:
        asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_accepts_process_policy_mapping() -> None:
    """External runtime config validates and stores stdio process policy mappings."""

    from mcp_unified.gateway.config import GatewayExternalRuntimeBootstrapConfig

    config = GatewayExternalRuntimeBootstrapConfig(
        enabled=True,
        process_policy={"allow_path_lookup": False, "allowed_env_names": ["PATH"]},
    )

    assert config.process_policy is not None
    assert config.process_policy.allow_path_lookup is False
    assert config.process_policy.allowed_env_names == ("PATH",)
    assert config.process_policy_configured is True


def test_gateway_config_bootstrap_rejects_invalid_process_policy_mapping() -> None:
    """External runtime config rejects invalid process-policy mappings early."""

    from mcp_unified.gateway.config import GatewayExternalRuntimeBootstrapConfig

    with pytest.raises(ValueError, match="allowed_executables"):
        GatewayExternalRuntimeBootstrapConfig(
            enabled=True,
            process_policy={"allowed_executables": ["python", ""]},
        )


def test_gateway_config_bootstrap_wraps_default_factory_when_policy_configured(
    tmp_path: Path,
) -> None:
    """Configured process policy should wrap only the package stdio factory."""

    from mcp_unified.federation import create_external_transport
    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )

    sqlite_path = tmp_path / "gateway.db"
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                external_runtime=GatewayExternalRuntimeBootstrapConfig(
                    enabled=True,
                    process_policy={"allow_path_lookup": False},
                ),
            ),
        )
    )

    try:
        manager = bootstrap.external_runtime_manager
        assert manager is not None
        assert manager._transport_factory is not create_external_transport
    finally:
        asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_wraps_explicit_package_factory_when_policy_configured(
    tmp_path: Path,
) -> None:
    """Explicit package stdio factories receive configured process policy too."""

    from mcp_unified.federation import create_external_transport
    from mcp_unified.federation.stdio_transport import StdioExternalTransportError
    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.storage.models import ExternalServerDefinition

    sqlite_path = tmp_path / "gateway.db"
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                external_runtime=GatewayExternalRuntimeBootstrapConfig(
                    enabled=True,
                    process_policy={"allow_path_lookup": False},
                ),
            ),
            external_transport_factory=create_external_transport,
        )
    )

    try:
        manager = bootstrap.external_runtime_manager
        assert manager is not None
        assert manager._transport_factory is not create_external_transport
        with pytest.raises(StdioExternalTransportError) as exc_info:
            manager._transport_factory(
                ExternalServerDefinition(
                    id="research",
                    name="Research",
                    transport="stdio",
                    command=["python"],
                    env_allowlist=["PATH"],
                )
            )

        assert exc_info.value.reason_code == "process_policy_path_lookup_denied"
    finally:
        asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_custom_factory_ignores_config_process_policy(
    tmp_path: Path,
) -> None:
    """Caller-injected factories own their own process-policy boundary."""

    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )

    def factory(server: ExternalServerDefinition) -> Any:
        return server.id

    sqlite_path = tmp_path / "gateway.db"
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                external_runtime=GatewayExternalRuntimeBootstrapConfig(
                    enabled=True,
                    process_policy={"allow_path_lookup": False},
                ),
            ),
            external_transport_factory=factory,
        )
    )

    try:
        manager = bootstrap.external_runtime_manager
        assert manager is not None
        assert manager._transport_factory is factory
    finally:
        asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_external_runtime_process_policy_start_failure_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime status should expose policy-denied starts without command/env secrets."""

    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.gateway.external_runtime import GatewayExternalRuntimeError
    from mcp_unified.storage.models import ExternalServerDefinition

    secret_arg = "do-not-leak-command-argument"
    secret_env_value = "do-not-leak-env-value"
    monkeypatch.setenv("MCP_POLICY_SECRET", secret_env_value)
    sqlite_path = tmp_path / "gateway.db"
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                external_runtime=GatewayExternalRuntimeBootstrapConfig(
                    enabled=True,
                    process_policy={"allow_path_lookup": False},
                ),
            ),
        )
    )

    try:
        manager = bootstrap.external_runtime_manager
        assert manager is not None
        asyncio.run(
            bootstrap.profile_store.create_server(
                ExternalServerDefinition(
                    id="research",
                    name="Research",
                    transport="stdio",
                    command=["python", secret_arg],
                    env_allowlist=["PATH", "MCP_POLICY_SECRET"],
                )
            )
        )

        with pytest.raises(GatewayExternalRuntimeError) as exc_info:
            asyncio.run(manager.start_server("research"))

        status = asyncio.run(manager.list_runtime_servers())
        status_json = json.dumps(status, sort_keys=True)
        assert exc_info.value.reason_code == "external_server_start_failed"
        assert "process_policy_path_lookup_denied" in status_json
        assert secret_arg not in str(exc_info.value)
        assert secret_arg not in status_json
        assert secret_env_value not in str(exc_info.value)
        assert secret_env_value not in status_json
    finally:
        asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_rejects_unsupported_external_runtime_factory() -> None:
    """Reject unsupported transport factory selectors instead of silently degrading."""

    from mcp_unified.gateway.config import GatewayExternalRuntimeBootstrapConfig

    with pytest.raises(ValueError, match="Unsupported gateway external runtime factory"):
        GatewayExternalRuntimeBootstrapConfig(
            enabled=True,
            transport_factory="websocket",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("lifecycle_flag", ["reconcile_on_startup", "stop_on_shutdown"])
def test_gateway_config_bootstrap_rejects_lifecycle_flags_when_external_runtime_disabled(
    lifecycle_flag: str,
) -> None:
    """Lifecycle hooks require the external runtime manager to be enabled."""

    from mcp_unified.gateway.config import GatewayExternalRuntimeBootstrapConfig

    with pytest.raises(ValueError, match="external_runtime.enabled"):
        GatewayExternalRuntimeBootstrapConfig(**{lifecycle_flag: True})


def test_gateway_config_bootstrap_rejects_external_runtime_without_registry_store() -> None:
    """Runtime management requires an external registry-capable store."""

    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        bootstrap_profile_gateway_from_config,
    )

    with pytest.raises(ValueError, match="external runtime management requires"):
        asyncio.run(
            bootstrap_profile_gateway_from_config(
                _MultiToolGatewayRuntime(),
                GatewayProfileBootstrapConfig(
                    external_runtime=GatewayExternalRuntimeBootstrapConfig(enabled=True),
                ),
            )
        )


def test_gateway_config_bootstrap_carries_external_runtime_lifecycle(
    tmp_path: Path,
) -> None:
    """Config bootstrap should carry lifecycle preferences into app creation."""

    from mcp_unified.gateway.config import (
        GatewayExternalRuntimeBootstrapConfig,
        GatewayProfileBootstrapConfig,
        GatewayProfileStoreConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.gateway.lifecycle import GatewayExternalRuntimeLifecycleConfig

    sqlite_path = tmp_path / "gateway.db"
    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
                store=GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
                external_runtime=GatewayExternalRuntimeBootstrapConfig(
                    enabled=True,
                    reconcile_on_startup=True,
                    stop_on_shutdown=True,
                ),
            ),
        )
    )

    try:
        assert bootstrap.external_runtime_lifecycle == GatewayExternalRuntimeLifecycleConfig(
            reconcile_on_startup=True,
            stop_on_shutdown=True,
        )
    finally:
        asyncio.run(bootstrap.profile_store.aclose())


def test_gateway_config_bootstrap_preserves_injected_profile_store(tmp_path: Path) -> None:
    """Prefer caller-injected stores over config-selected store creation."""

    from mcp_unified.gateway.config import (
        GatewayProfileBootstrapConfig,
        bootstrap_profile_gateway_from_config,
    )
    from mcp_unified.profiles.store import InMemoryProfileStore

    profile = _profile_with_allowed_tools("reviewer", ["admin.delete"])
    injected_store = InMemoryProfileStore([profile])

    bootstrap = asyncio.run(
        bootstrap_profile_gateway_from_config(
            _MultiToolGatewayRuntime(),
            GatewayProfileBootstrapConfig(
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
    _assert_profile_runtime_tool_names(
        listed.json()["result"]["tools"],
        backend_tools=["admin.delete"],
    )


def test_gateway_config_storage_reuses_injected_sqlite_store_capabilities(tmp_path: Path) -> None:
    """Reuse assignment and audit capabilities from injected persistent stores."""

    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        build_gateway_profile_storage,
    )
    from mcp_unified.storage.sqlite import SQLiteMCPStore

    sqlite_path = tmp_path / "gateway.db"
    store = SQLiteMCPStore(sqlite_path)

    try:
        bundle = build_gateway_profile_storage(
            GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
            profile_store=store,
        )

        assert bundle.profile_store is store
        assert bundle.assignment_store is store
        assert bundle.audit_store is store
        assert bundle.metadata.to_payload() == {
            "kind": "sqlite",
            "persistent": True,
        }
    finally:
        asyncio.run(store.aclose())


def test_gateway_config_builds_sqlite_external_registry_storage(tmp_path: Path) -> None:
    """Build external registry storage from the configured SQLite store."""

    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        build_gateway_external_registry_storage,
    )
    from mcp_unified.storage.sqlite import SQLiteMCPStore

    bundle = build_gateway_external_registry_storage(
        GatewayProfileStoreConfig(kind="sqlite", sqlite_path=tmp_path / "gateway.db"),
    )

    try:
        assert isinstance(bundle.external_registry_store, SQLiteMCPStore)
        assert bundle.credential_grant_store is bundle.external_registry_store
        assert bundle.audit_store is bundle.external_registry_store
        assert bundle.metadata.to_payload() == {
            "kind": "sqlite",
            "persistent": True,
        }
    finally:
        asyncio.run(bundle.external_registry_store.aclose())


def test_gateway_config_external_registry_storage_reuses_injected_sqlite_store_capabilities(
    tmp_path: Path,
) -> None:
    """Reuse external registry, grant, and audit capabilities from SQLite stores."""

    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        build_gateway_external_registry_storage,
    )
    from mcp_unified.storage.sqlite import SQLiteMCPStore

    sqlite_path = tmp_path / "gateway.db"
    store = SQLiteMCPStore(sqlite_path)

    try:
        bundle = build_gateway_external_registry_storage(
            GatewayProfileStoreConfig(kind="sqlite", sqlite_path=sqlite_path),
            external_registry_store=store,
        )

        assert bundle.external_registry_store is store
        assert bundle.credential_grant_store is store
        assert bundle.audit_store is store
        assert bundle.metadata.to_payload() == {
            "kind": "sqlite",
            "persistent": True,
        }
    finally:
        asyncio.run(store.aclose())


def test_gateway_config_external_registry_storage_manager_from_storage_uses_bundle(
    tmp_path: Path,
) -> None:
    """Build an external registry manager from a resolved storage bundle."""

    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        build_gateway_external_registry_storage,
        external_registry_manager_from_storage,
    )

    bundle = build_gateway_external_registry_storage(
        GatewayProfileStoreConfig(kind="sqlite", sqlite_path=tmp_path / "gateway.db"),
    )

    try:
        manager = external_registry_manager_from_storage(bundle)

        assert manager.external_registry_store is bundle.external_registry_store
        assert manager.credential_grant_store is bundle.credential_grant_store
        assert manager.audit_store is bundle.audit_store
        assert manager.store_metadata is bundle.metadata
    finally:
        asyncio.run(bundle.external_registry_store.aclose())


def test_gateway_config_external_registry_memory_requires_injected_store() -> None:
    """Reject production memory config without an injected external registry."""

    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        build_gateway_external_registry_storage,
    )

    with pytest.raises(ValueError, match="sqlite.*injected equivalent"):
        build_gateway_external_registry_storage(
            GatewayProfileStoreConfig(kind="memory"),
        )


def test_gateway_config_external_registry_injected_store_can_omit_grant_store() -> None:
    """Allow injected external registry stores without credential grant storage."""

    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        build_gateway_external_registry_storage,
        external_registry_manager_from_storage,
    )
    from mcp_unified.gateway.external_registry import GatewayExternalRegistryManagementError
    from mcp_unified.storage.models import ExternalServerDefinition

    class ExternalRegistryOnlyStore:
        def __init__(self, server: ExternalServerDefinition) -> None:
            self.server = server

        async def get_server(self, server_id: str) -> ExternalServerDefinition | None:
            if server_id == self.server.id:
                return self.server
            return None

        async def list_servers(self) -> list[ExternalServerDefinition]:
            return [self.server]

        async def list_server_definitions(
            self,
            *,
            enabled: bool | None = None,
        ) -> list[ExternalServerDefinition]:
            if enabled is None or enabled is self.server.enabled:
                return [self.server]
            return []

        async def create_server(
            self,
            server: ExternalServerDefinition,
        ) -> ExternalServerDefinition:
            self.server = server
            return server

        async def upsert_server(
            self,
            server: ExternalServerDefinition,
        ) -> ExternalServerDefinition:
            self.server = server
            return server

        async def delete_server(self, server_id: str) -> bool:
            return server_id == self.server.id

    store = ExternalRegistryOnlyStore(
        ExternalServerDefinition(
            id="local-research",
            name="Local Research",
            transport="stdio",
            command=["mcp-local-research"],
        )
    )
    bundle = build_gateway_external_registry_storage(
        GatewayProfileStoreConfig(kind="memory"),
        external_registry_store=store,
    )
    manager = external_registry_manager_from_storage(bundle)

    assert bundle.external_registry_store is store
    assert bundle.credential_grant_store is None
    assert manager.credential_grant_store is None
    with pytest.raises(GatewayExternalRegistryManagementError) as exc_info:
        asyncio.run(manager.delete_server("local-research"))
    assert exc_info.value.reason_code == "credential_grant_store_unavailable"


def test_gateway_config_rejects_sqlite_injected_store_without_assignment_store(
    tmp_path: Path,
) -> None:
    """Reject divergent injected SQLite profile stores without assignment support."""

    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        build_gateway_profile_storage,
    )
    from mcp_unified.profiles.store import InMemoryProfileStore

    with pytest.raises(ValueError, match="assignment_store"):
        build_gateway_profile_storage(
            GatewayProfileStoreConfig(
                kind="sqlite",
                sqlite_path=tmp_path / "gateway.db",
            ),
            profile_store=InMemoryProfileStore(),
        )


def test_gateway_config_rejects_sqlite_injected_store_without_audit_store(
    tmp_path: Path,
) -> None:
    """Reject divergent injected SQLite profile stores without audit support."""

    from mcp_unified.gateway.config import (
        GatewayProfileStoreConfig,
        build_gateway_profile_storage,
    )
    from mcp_unified.profiles.store import InMemoryProfileStore

    class AssignmentOnlyProfileStore(InMemoryProfileStore):
        async def get_assignment(self, *_: object, **__: object) -> None:
            return None

        async def list_assignments(self, *_: object, **__: object) -> list[object]:
            return []

        async def upsert_assignment(self, assignment: object) -> object:
            return assignment

        async def delete_assignment(self, *_: object, **__: object) -> bool:
            return False

    with pytest.raises(ValueError, match="audit_store"):
        build_gateway_profile_storage(
            GatewayProfileStoreConfig(
                kind="sqlite",
                sqlite_path=tmp_path / "gateway.db",
            ),
            profile_store=AssignmentOnlyProfileStore(),
        )


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

    _assert_profile_runtime_tool_names(
        listed.json()["result"]["tools"],
        backend_tools=["echo.search"],
    )


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
    _assert_profile_runtime_tool_names(
        listed.json()["result"]["tools"],
        backend_tools=["echo.search"],
        includes_tool_call=True,
    )


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
        status_payload = status.json()
        assert status_payload["status"] == "ok"
        assert status_payload["name"] == "unit-gateway"
        assert status_payload["version"] == "0.0-test"
        assert status_payload["package"]["package_status"] == "public-alpha"

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


def test_gateway_jsonrpc_notification_and_explicit_null_id_are_distinct() -> None:
    response = asyncio.run(
        gateway_jsonrpc.handle_jsonrpc(
            _FakeGatewayRuntime(),
            {"jsonrpc": "2.0", "method": "ping"},
            path="/mcp/request",
        )
    )
    assert isinstance(response, gateway_jsonrpc.GatewayNoResponse)

    explicit_null_response = asyncio.run(
        gateway_jsonrpc.handle_jsonrpc(
            _FakeGatewayRuntime(),
            {"jsonrpc": "2.0", "method": "ping", "id": None},
            path="/mcp/request",
        )
    )
    assert isinstance(explicit_null_response, gateway_jsonrpc.GatewayJSONRPCSuccessResponse)
    assert explicit_null_response.id is None
    assert explicit_null_response.result == {"pong": True}


def test_gateway_explicit_null_id_runtime_context_is_not_notification() -> None:
    runtime = _FakeGatewayRuntime()

    response = asyncio.run(
        gateway_jsonrpc.handle_jsonrpc(
            runtime,
            {"jsonrpc": "2.0", "method": "tools/list", "id": None},
            path="/mcp/request",
        )
    )

    assert isinstance(response, gateway_jsonrpc.GatewayJSONRPCSuccessResponse)
    assert response.id is None
    assert runtime.list_contexts[-1].request_id == "null"


def test_gateway_response_to_json_omits_invalid_optional_null_fields() -> None:
    success = gateway_jsonrpc.response_to_json(
        gateway_jsonrpc.GatewayJSONRPCSuccessResponse(result={"ok": True}, id=None)
    )
    assert success == {"jsonrpc": "2.0", "result": {"ok": True}, "id": None}
    assert "error" not in success

    error = gateway_jsonrpc.response_to_json(
        gateway_jsonrpc.jsonrpc_error(-32600, "Invalid request", None)
    )
    assert error == {
        "jsonrpc": "2.0",
        "error": {"code": -32600, "message": "Invalid request"},
        "id": None,
    }
    assert "result" not in error
    assert "data" not in error["error"]


def test_gateway_request_parse_error_omits_null_error_data() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        response = client.post(
            "/mcp/request",
            content=b"{not-json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] is None
    assert body["error"]["code"] == -32700
    assert "data" not in body["error"]


def test_gateway_request_preserves_http_notification_and_explicit_null_id_semantics() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        notification = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "ping"},
        )
        explicit_null = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "ping", "id": None},
        )

    assert notification.status_code == 204
    assert notification.content == b""
    assert explicit_null.status_code == 200
    assert explicit_null.json() == {"jsonrpc": "2.0", "result": {"pong": True}, "id": None}


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


def test_gateway_request_rejects_invalid_ids_with_null_error_id() -> None:
    runtime = _FakeGatewayRuntime()
    app = create_gateway_app(runtime, prefix="/mcp")

    invalid_ids: list[Any] = [True, 1.25]

    with TestClient(app) as client:
        responses = [
            client.post(
                "/mcp/request",
                json={"jsonrpc": "2.0", "method": "ping", "id": invalid_id},
            )
            for invalid_id in invalid_ids
        ]

    for response in responses:
        assert response.status_code == 200
        body = response.json()
        assert body["id"] is None
        assert body["error"]["code"] == -32600


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


def test_gateway_stdio_returns_response_for_explicit_null_id_request() -> None:
    from mcp_unified.gateway.stdio import handle_stdio_line

    runtime = _FakeGatewayRuntime()

    response_line = asyncio.run(handle_stdio_line(runtime, '{"jsonrpc":"2.0","method":"ping","id":null}\n'))

    assert response_line is not None
    assert json.loads(response_line) == {"jsonrpc": "2.0", "result": {"pong": True}, "id": None}


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


def test_gateway_profile_runtime_failing_grant_store_fails_closed_to_denial() -> None:
    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.profiles.store import InMemoryProfileStore

    class _FailingGrantStore:
        def find_active_grant(self, **kwargs: Any) -> Any:
            raise RuntimeError("grant store unavailable")

        def list_active_grants(self, **kwargs: Any) -> Any:
            raise RuntimeError("grant store unavailable")

    backend = _CustomToolListGatewayRuntime([_web_fetch_tool_descriptor()])
    profile = _profile_with_allowed_tools_and_permission_rules(
        "researcher",
        allowed_tools=["web.fetch"],
        permission_rules=[{"pattern": "WebFetch(example.com)", "outcome": "ask"}],
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="researcher",
        policy_grant_store=_FailingGrantStore(),
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        denied = _post_tool_call(
            client,
            "web.fetch",
            {"url": "https://example.com/private", "query": "private"},
            "ask-store-failure",
        )

    body = denied.json()
    _assert_jsonrpc_error(body, code=-32001, request_id="ask-store-failure")
    assert body["error"]["data"]["status"] == "approval_required"
    assert backend.call_requests == []


def _path_grant_runtime(profile_path_scopes: list[dict[str, Any]] | None = None) -> tuple[Any, Any, Any]:
    """Build a grant-store-backed runtime around one fs.read_text reviewer profile."""

    from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore
    from mcp_unified.profiles.models import MCPProfile, ProfilePolicy
    from mcp_unified.profiles.store import InMemoryProfileStore

    backend = _CustomToolListGatewayRuntime([_read_text_tool_descriptor()])
    profile = MCPProfile(
        id="reviewer",
        name="Profile reviewer",
        policy_document=ProfilePolicy(allowed_tools=["fs.read_text"]),
        path_scopes=profile_path_scopes or [],
    )
    grant_store = InMemoryPolicyGrantStore()
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="reviewer",
        policy_grant_store=grant_store,
    )
    return runtime, backend, grant_store


def _delegated_path_scopes(backend: Any) -> list[dict[str, Any]]:
    """Return the path scopes from the last delegated call's effective policy."""

    from mcp_unified.gateway.profile_runtime import EFFECTIVE_POLICY_METADATA_KEY

    delegated_context = backend.call_requests[-1][2]
    effective_policy = delegated_context.metadata[EFFECTIVE_POLICY_METADATA_KEY]
    return effective_policy["path_scopes"]


def test_gateway_profile_runtime_merges_ttl_path_grants_into_effective_policy() -> None:
    runtime, backend, grant_store = _path_grant_runtime(
        profile_path_scopes=[{"prefix": "docs/manuals", "actions": ["read"], "effect": "allow"}]
    )
    grant = grant_store.create_grant(
        profile_id="reviewer",
        grant_type="path",
        subject_type="path",
        value="docs/scratch",
        actions=("read", "write"),
        ttl_seconds=900,
    )
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        allowed = _post_tool_call(
            client,
            "fs.read_text",
            {"path": "docs/scratch/notes.txt", "query": "notes"},
            "ttl-path-grant-merged",
        )

    assert allowed.json().get("error") is None
    path_scopes = _delegated_path_scopes(backend)
    assert {"prefix": "docs/manuals", "actions": ["read"], "effect": "allow"} in path_scopes
    merged = [scope for scope in path_scopes if scope.get("source") == "ttl_grant"]
    assert len(merged) == 1
    assert merged[0]["prefix"] == "docs/scratch"
    assert merged[0]["actions"] == ["read", "write"]
    assert merged[0]["effect"] == "allow"
    assert merged[0]["grant_id"] == grant.grant_id
    assert merged[0]["expires_at"] == grant.expires_at_iso()


def test_gateway_profile_runtime_skips_inapplicable_ttl_path_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp_unified.policy_grants.memory as memory_grants

    runtime, backend, grant_store = _path_grant_runtime()
    grant_store.create_grant(
        profile_id="reviewer",
        grant_type="path",
        subject_type="path",
        value="docs/other-session",
        actions=("read",),
        ttl_seconds=900,
        session_id="session-1",
    )
    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_000.0)
    grant_store.create_grant(
        profile_id="reviewer",
        grant_type="path",
        subject_type="path",
        value="docs/expired",
        actions=("read",),
        ttl_seconds=10,
    )
    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_011.0)
    app = create_gateway_app(runtime, prefix="/mcp")

    with TestClient(app) as client:
        allowed = _post_tool_call(
            client,
            "fs.read_text",
            {"path": "docs/anything.txt", "query": "notes"},
            "ttl-path-grant-skipped",
        )

    assert allowed.json().get("error") is None
    path_scopes = _delegated_path_scopes(backend)
    assert [scope for scope in path_scopes if scope.get("source") == "ttl_grant"] == []
