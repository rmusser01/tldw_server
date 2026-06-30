from __future__ import annotations

import pytest

from tldw_Server_API.app.core.MCP_unified.external_servers import config_schema as config_schema_mod
from tldw_Server_API.app.core.MCP_unified.external_servers.config_schema import (
    ExternalAuthMode,
    ExternalServerRegistryConfig,
    parse_external_server_registry,
)


def test_parse_registry_accepts_websocket_and_stdio() -> None:
    cfg = parse_external_server_registry(
        {
            "servers": [
                {
                    "id": "docs",
                    "name": "Docs",
                    "transport": "websocket",
                    "websocket": {"url": "wss://example.test/ws"},
                    "auth": {"mode": "bearer_env", "token_env": "DOCS_TOKEN"},
                },
                {
                    "id": "local_ci",
                    "name": "Local CI",
                    "transport": "stdio",
                    "stdio": {"command": "node", "args": ["ci.js"]},
                },
            ]
        }
    )

    assert len(cfg.servers) == 2
    assert cfg.servers[0].id == "docs"
    assert cfg.servers[0].auth.mode == ExternalAuthMode.BEARER_ENV
    assert cfg.servers[1].id == "local_ci"
    assert cfg.servers[1].stdio is not None


def test_parse_registry_requires_transport_specific_config() -> None:
    with pytest.raises(ValueError, match="requires websocket config"):
        parse_external_server_registry(
            {
                "servers": [
                    {
                        "id": "docs",
                        "name": "Docs",
                        "transport": "websocket",
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="requires stdio config"):
        parse_external_server_registry(
            {
                "servers": [
                    {
                        "id": "local",
                        "name": "Local",
                        "transport": "stdio",
                    }
                ]
            }
        )


def test_parse_registry_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate external server id"):
        parse_external_server_registry(
            {
                "servers": [
                    {
                        "id": "dup",
                        "name": "First",
                        "transport": "websocket",
                        "websocket": {"url": "wss://a.example/ws"},
                    },
                    {
                        "id": "dup",
                        "name": "Second",
                        "transport": "stdio",
                        "stdio": {"command": "echo"},
                    },
                ]
            }
        )


def test_policy_allow_and_deny_patterns() -> None:
    cfg = parse_external_server_registry(
        {
            "servers": [
                {
                    "id": "policy",
                    "name": "Policy",
                    "transport": "websocket",
                    "websocket": {"url": "wss://policy.example/ws"},
                    "policy": {
                        "allow_tool_patterns": ["docs.*"],
                        "deny_tool_patterns": ["docs.delete"],
                    },
                }
            ]
        }
    )

    policy = cfg.servers[0].policy
    assert policy.allows_tool("docs.search") is True
    assert policy.allows_tool("docs.delete") is False
    assert policy.allows_tool("ci.run") is False


def test_standalone_package_exports_external_server_config_schema() -> None:
    from mcp_unified import federation
    from mcp_unified.federation import config_schema as package_schema

    assert federation.ExternalAuthMode is package_schema.ExternalAuthMode
    assert federation.ExternalMCPServerConfig is package_schema.ExternalMCPServerConfig
    assert federation.parse_external_server_registry is package_schema.parse_external_server_registry
    assert config_schema_mod.ExternalAuthMode is package_schema.ExternalAuthMode
    assert config_schema_mod.ExternalServerRegistryConfig is package_schema.ExternalServerRegistryConfig

    cfg = package_schema.parse_external_server_registry(
        {
            "servers": [
                {
                    "id": "docs",
                    "name": "Docs",
                    "transport": "websocket",
                    "websocket": {"url": "wss://example.test/ws"},
                }
            ]
        }
    )

    assert cfg.servers[0].id == "docs"


def test_host_loader_preserves_legacy_default_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | None, str | None]] = []

    def fake_package_loader(
        config_path: str | None = None,
        *,
        default_config_path: str | None = None,
    ) -> ExternalServerRegistryConfig:
        calls.append((config_path, default_config_path))
        return ExternalServerRegistryConfig()

    monkeypatch.setattr(config_schema_mod, "_load_external_server_registry", fake_package_loader)

    assert config_schema_mod.load_external_server_registry() == ExternalServerRegistryConfig()
    assert calls == [
        (None, "tldw_Server_API/Config_Files/mcp_external_servers.yaml"),
    ]


def test_load_registry_directory_path_returns_empty_config(tmp_path) -> None:
    from mcp_unified.federation.config_schema import load_external_server_registry

    assert load_external_server_registry(str(tmp_path)) == ExternalServerRegistryConfig()


def test_load_registry_rejects_yaml_non_mapping_root(tmp_path) -> None:
    from mcp_unified.federation.config_schema import load_external_server_registry

    config_path = tmp_path / "external_servers.yaml"
    config_path.write_text("- docs\n- ci\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_external_server_registry(str(config_path))
