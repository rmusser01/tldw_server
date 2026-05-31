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
    assert body["error"] == f"domain failure: {reason_code}"


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

    assert [tool["name"] for tool in first.json()["result"]["tools"]] == ["echo.search"]
    assert [tool["name"] for tool in second.json()["result"]["tools"]] == ["admin.delete"]
    assert [tool["name"] for tool in explicit.json()["result"]["tools"]] == ["echo.search"]


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
    assert [tool["name"] for tool in first_tools.json()["result"]["tools"]] == ["echo.search"]
    assert second_default.status_code == 200
    assert second_default.json()["default"]["profile_id"] == "architect"
    assert [tool["name"] for tool in second_tools.json()["result"]["tools"]] == ["admin.delete"]


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
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["admin.delete"]


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

    with pytest.raises(ValueError, match="external registry.*sqlite"):
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
