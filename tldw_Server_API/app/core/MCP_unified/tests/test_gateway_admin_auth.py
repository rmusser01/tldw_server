from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mcp_unified.gateway.admin_auth import GatewayAdminAuthConfig
from mcp_unified.gateway.config import load_gateway_profile_bootstrap_config
from mcp_unified.gateway.fastapi import create_gateway_app


class _FakeGatewayRuntime:
    """Small gateway runtime for admin auth route tests."""

    name = "admin-auth-test"
    version = "0.0-test"

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        """Return one deterministic tool."""

        del context
        return [
            {
                "name": "echo.search",
                "description": "Echo a query.",
                "inputSchema": {"type": "object", "properties": {}},
                "metadata": {"category": "test"},
            }
        ]


class _ProfileManagerDouble:
    """Small profile manager double for protected management routes."""

    async def list_profiles(self) -> dict[str, Any]:
        """Return a deterministic profile-management payload."""

        return {
            "ok": True,
            "profiles": [{"id": "reviewer", "name": "Reviewer"}],
            "store": {"kind": "memory", "persistent": False},
        }

    async def show_profile(self, profile_id: str) -> dict[str, Any]:
        """Return a deterministic profile payload."""

        return {
            "ok": True,
            "profile": {"id": profile_id, "name": f"Profile {profile_id}"},
            "store": {"kind": "memory", "persistent": False},
        }


def test_gateway_admin_auth_protects_management_routes() -> None:
    """Admin-authenticated gateway apps protect management routes only."""

    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_manager=_ProfileManagerDouble(),
        admin_auth=GatewayAdminAuthConfig(
            enabled=True,
            header_name="X-Test-Gateway-Admin",
            api_key="test-admin-key",
        ),
    )

    with TestClient(app) as client:
        missing = client.get("/mcp/profiles")
        invalid = client.get(
            "/mcp/profiles",
            headers={"X-Test-Gateway-Admin": "wrong-key"},
        )
        allowed = client.get(
            "/mcp/profiles",
            headers={"X-Test-Gateway-Admin": "test-admin-key"},
        )

    assert missing.status_code == 401
    assert missing.json() == {
        "ok": False,
        "error": "Gateway admin authentication required",
        "reason_code": "admin_auth_required",
    }
    assert invalid.status_code == 403
    assert invalid.json() == {
        "ok": False,
        "error": "Gateway admin authentication failed",
        "reason_code": "admin_auth_invalid",
    }
    assert allowed.status_code == 200
    assert allowed.json()["profiles"] == [{"id": "reviewer", "name": "Reviewer"}]


def test_gateway_admin_auth_does_not_gate_status_or_jsonrpc() -> None:
    """Admin auth should not accidentally protect status or JSON-RPC calls."""

    app = create_gateway_app(
        _FakeGatewayRuntime(),
        prefix="/mcp",
        profile_manager=_ProfileManagerDouble(),
        admin_auth=GatewayAdminAuthConfig(enabled=True, api_key="test-admin-key"),
    )

    with TestClient(app) as client:
        status = client.get("/mcp/status")
        tools = client.post(
            "/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": "tools-1"},
        )

    assert status.status_code == 200
    assert status.json() == {
        "status": "ok",
        "name": "admin-auth-test",
        "version": "0.0-test",
    }
    assert tools.status_code == 200
    assert tools.json()["result"]["tools"][0]["name"] == "echo.search"


@pytest.mark.parametrize("suffix", [".json", ".toml"])
def test_gateway_config_loads_admin_auth_without_plaintext_key(
    tmp_path: Path,
    suffix: str,
) -> None:
    """Standalone config may enable admin auth through an env-var reference."""

    config_path = tmp_path / f"gateway{suffix}"
    if suffix == ".json":
        config_path.write_text(
            json.dumps(
                {
                    "store": {"kind": "memory"},
                    "admin_auth": {
                        "enabled": True,
                        "header_name": "X-Test-Gateway-Admin",
                        "api_key_env_var": "TEST_GATEWAY_ADMIN_KEY",
                    },
                }
            ),
            encoding="utf-8",
        )
    else:
        config_path.write_text(
            "\n".join(
                [
                    "[store]",
                    'kind = "memory"',
                    "",
                    "[admin_auth]",
                    "enabled = true",
                    'header_name = "X-Test-Gateway-Admin"',
                    'api_key_env_var = "TEST_GATEWAY_ADMIN_KEY"',
                ]
            ),
            encoding="utf-8",
        )

    config = load_gateway_profile_bootstrap_config(config_path)

    assert config.admin_auth.enabled is True
    assert config.admin_auth.header_name == "X-Test-Gateway-Admin"
    assert config.admin_auth.api_key_env_var == "TEST_GATEWAY_ADMIN_KEY"


def test_gateway_config_rejects_plaintext_admin_auth_key(tmp_path: Path) -> None:
    """File-backed config must not accept a persisted admin key."""

    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "store": {"kind": "memory"},
                "admin_auth": {
                    "enabled": True,
                    "api_key": "plaintext-secret",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="admin_auth.api_key"):
        load_gateway_profile_bootstrap_config(config_path)


def test_gateway_config_rejects_blank_admin_auth_header(tmp_path: Path) -> None:
    """Configured admin auth header names must be non-blank."""

    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "store": {"kind": "memory"},
                "admin_auth": {
                    "enabled": True,
                    "header_name": " ",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="admin_auth.header_name"):
        load_gateway_profile_bootstrap_config(config_path)
