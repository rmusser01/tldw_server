from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core import config


@pytest.fixture(autouse=True)
def _reset_config_state(monkeypatch):
    config.clear_config_cache()
    original_data = object.__getattribute__(config.loaded_config_data, "_data")
    monkeypatch.setattr(config.loaded_config_data, "_data", None, raising=False)
    yield
    monkeypatch.setattr(config.loaded_config_data, "_data", original_data, raising=False)
    config.clear_config_cache()


def _set_config_data(monkeypatch, data: dict) -> None:
    monkeypatch.setattr(config.loaded_config_data, "_data", data, raising=False)


def test_validate_config_walks_nested_values_and_redacts_invalid_url_values(monkeypatch):
    _set_config_data(
        monkeypatch,
        {
            "embedding_config": {
                "embedding_api_url": "ftp://embeddings.invalid/v1",
            },
            "providers": {
                "fallback": {
                    "api_key": "TODO",
                }
            },
        },
    )

    overrides = {
        ("Database", "pg_connection_string"): "mysql://user:super-secret@db.invalid/app",
        ("Image-Generation", "swarmui_base_url"): "ftp://swarm.invalid/api",
    }

    def _fake_get_config_value(section: str, key: str, default=None, *, reload: bool = False):  # noqa: ARG001
        return overrides.get((section, key), default)

    monkeypatch.setattr(config, "get_config_value", _fake_get_config_value, raising=True)

    warnings = config.validate_config()

    assert any("embedding_config.embedding_api_url" in warning for warning in warnings)
    assert any("providers.fallback.api_key" in warning and "placeholder value" in warning for warning in warnings)
    assert any("Database.pg_connection_string" in warning for warning in warnings)
    assert any("Image-Generation.swarmui_base_url" in warning for warning in warnings)
    assert all("super-secret" not in warning for warning in warnings)
    assert all("mysql://user" not in warning for warning in warnings)


def test_validate_config_accepts_legacy_postgres_scheme(monkeypatch):
    _set_config_data(
        monkeypatch,
        {
            "embedding_config": {
                "embedding_api_url": "https://embeddings.invalid/v1",
            },
        },
    )

    overrides = {
        ("Database", "pg_connection_string"): "postgres://db.invalid/app",
        ("Image-Generation", "swarmui_base_url"): "https://swarm.invalid/api",
    }

    def _fake_get_config_value(section: str, key: str, default=None, *, reload: bool = False):  # noqa: ARG001
        return overrides.get((section, key), default)

    monkeypatch.setattr(config, "get_config_value", _fake_get_config_value, raising=True)

    warnings = config.validate_config()

    assert not any("Database.pg_connection_string" in warning for warning in warnings)


def test_get_ingestion_source_allowed_roots_resolves_relative_paths_from_project_root(monkeypatch):
    monkeypatch.delenv("INGESTION_SOURCE_ALLOWED_ROOTS", raising=False)
    monkeypatch.setenv("TLDW_INGESTION_SOURCE_ALLOWED_ROOTS", "fixtures/input")

    roots = config.get_ingestion_source_allowed_roots()

    assert roots == (config.ACTUAL_PROJECT_ROOT / "fixtures" / "input",)
    assert all(isinstance(root, Path) and root.is_absolute() for root in roots)


def test_get_workspace_project_root_allowed_roots_prefers_workspace_specific_config(monkeypatch):
    workspace_root = config.ACTUAL_PROJECT_ROOT / "workspace-projects"
    acp_root = config.ACTUAL_PROJECT_ROOT / "legacy-acp"

    def _fake_get_config_value(section: str, key: str, default=None, *, reload: bool = False):
        values = {
            ("WORKSPACES", "project_root_allowed_base_paths"): "workspace-projects",
            ("ACP-WORKSPACE", "allowed_base_paths"): "legacy-acp",
        }
        return values.get((section, key), default)

    monkeypatch.setattr(config, "get_config_value", _fake_get_config_value, raising=True)
    monkeypatch.delenv("WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)
    monkeypatch.delenv("TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)
    monkeypatch.delenv("ACP_WORKSPACE_ALLOWED_BASE_PATHS", raising=False)

    assert config.get_workspace_project_root_allowed_roots() == (workspace_root,)
    assert acp_root not in config.get_workspace_project_root_allowed_roots()


def test_get_workspace_project_root_allowed_roots_uses_acp_fallback_only_when_workspace_empty(monkeypatch):
    def _fake_get_config_value(section: str, key: str, default=None, *, reload: bool = False):
        values = {
            ("WORKSPACES", "project_root_allowed_base_paths"): "",
            ("ACP-WORKSPACE", "allowed_base_paths"): "legacy-acp",
        }
        return values.get((section, key), default)

    monkeypatch.setattr(config, "get_config_value", _fake_get_config_value, raising=True)
    monkeypatch.delenv("WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)
    monkeypatch.delenv("TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)
    monkeypatch.delenv("ACP_WORKSPACE_ALLOWED_BASE_PATHS", raising=False)

    assert config.get_workspace_project_root_allowed_roots() == (
        config.ACTUAL_PROJECT_ROOT / "legacy-acp",
    )


def test_get_workspace_project_root_allowed_roots_dedupes_config_and_env(monkeypatch):
    def _fake_get_config_value(section: str, key: str, default=None, *, reload: bool = False):
        if (section, key) == ("WORKSPACES", "project_root_allowed_base_paths"):
            return "projects,projects"
        return default

    monkeypatch.setattr(config, "get_config_value", _fake_get_config_value, raising=True)
    monkeypatch.setenv("WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", "projects")
    monkeypatch.delenv("TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)

    assert config.get_workspace_project_root_allowed_roots() == (
        config.ACTUAL_PROJECT_ROOT / "projects",
    )
