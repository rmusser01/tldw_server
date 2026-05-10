"""Tests for workspace API helper functions and MCP resolver fallback."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.DB_Management.Orchestration_DB import OrchestrationDB
from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
    McpHubWorkspaceRootResolver,
)


# ---------------------------------------------------------------------------
# _validate_workspace_root tests
# ---------------------------------------------------------------------------


class TestValidateWorkspaceRoot:
    def test_valid_absolute_path(self):
        def mock_config(section, key, default=""):
            if section == "ACP-WORKSPACE" and key == "allowed_base_paths":
                return "/tmp"
            return default

        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _validate_workspace_root

        with patch(
            "tldw_Server_API.app.core.config.get_config_value",
            side_effect=mock_config,
        ):
            result = _validate_workspace_root("/tmp")
        assert result == "/private/tmp" or result == "/tmp"  # macOS resolves /tmp

    def test_expands_user_home(self):
        def mock_config(section, key, default=""):
            if section == "ACP-WORKSPACE" and key == "allowed_base_paths":
                return str(Path.home())
            return default

        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _validate_workspace_root

        with patch(
            "tldw_Server_API.app.core.config.get_config_value",
            side_effect=mock_config,
        ):
            result = _validate_workspace_root("~/")
        assert not result.startswith("~")
        assert result.startswith("/")

    def test_resolves_relative_components(self):
        def mock_config(section, key, default=""):
            if section == "ACP-WORKSPACE" and key == "allowed_base_paths":
                return "/tmp"
            return default

        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _validate_workspace_root

        with patch(
            "tldw_Server_API.app.core.config.get_config_value",
            side_effect=mock_config,
        ):
            result = _validate_workspace_root("/tmp/../tmp")
        assert ".." not in result

    def test_rejects_relative_root_path(self):
        from fastapi import HTTPException
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _validate_workspace_root

        with pytest.raises(HTTPException) as exc_info:
            _validate_workspace_root("relative/path")

        assert exc_info.value.status_code == 400

    def test_allowed_base_paths_enforcement(self):
        from fastapi import HTTPException

        def mock_config(section, key, default=""):
            if section == "ACP-WORKSPACE" and key == "allowed_base_paths":
                return "/allowed/path,/another/allowed"
            return default

        with patch(
            "tldw_Server_API.app.core.config.get_config_value",
            side_effect=mock_config,
        ):
            from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _validate_workspace_root
            with pytest.raises(HTTPException) as exc_info:
                _validate_workspace_root("/not/allowed/path")
            assert exc_info.value.status_code == 403
            assert exc_info.value.detail["code"] == "workspace_root_not_allowed"
            assert "allowed_base_paths" not in exc_info.value.detail
            assert "root_path" not in exc_info.value.detail
            assert "ACP-WORKSPACE.allowed_base_paths" in exc_info.value.detail["message"]

    def test_empty_allowed_base_paths_rejects_workspace_creation(self):
        from fastapi import HTTPException

        def mock_config(section, key, default=""):
            if section == "ACP-WORKSPACE" and key == "allowed_base_paths":
                return ""
            return default

        with patch(
            "tldw_Server_API.app.core.config.get_config_value",
            side_effect=mock_config,
        ):
            from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _validate_workspace_root
            with pytest.raises(HTTPException) as exc_info:
                _validate_workspace_root("/tmp")
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["code"] == "workspace_roots_not_configured"
            assert "ACP_WORKSPACE_ALLOWED_BASE_PATHS" in exc_info.value.detail["configure"]


class TestResolveDispatchCwd:
    def test_relative_cwd_resolves_inside_workspace_root(self, tmp_path):
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _resolve_dispatch_cwd

        workspace_root = tmp_path / "workspace"
        nested = workspace_root / "nested"
        nested.mkdir(parents=True)

        def mock_config(section, key, default=""):
            if section == "ACP-WORKSPACE" and key == "allowed_base_paths":
                return str(tmp_path)
            return default

        with patch(
            "tldw_Server_API.app.core.config.get_config_value",
            side_effect=mock_config,
        ):
            result = _resolve_dispatch_cwd("nested", workspace_root=str(workspace_root))

        assert result == str(nested.resolve())

    def test_relative_cwd_cannot_escape_workspace_root(self, tmp_path):
        from fastapi import HTTPException
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _resolve_dispatch_cwd

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()

        def mock_config(section, key, default=""):
            if section == "ACP-WORKSPACE" and key == "allowed_base_paths":
                return str(tmp_path)
            return default

        with patch(
            "tldw_Server_API.app.core.config.get_config_value",
            side_effect=mock_config,
        ):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_dispatch_cwd("../outside", workspace_root=str(workspace_root))

        assert exc_info.value.status_code == 403

    def test_absolute_cwd_override_is_rejected_when_workspace_root_present(self, tmp_path):
        from fastapi import HTTPException
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _resolve_dispatch_cwd

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        nested = workspace_root / "nested"
        nested.mkdir()

        def mock_config(section, key, default=""):
            if section == "ACP-WORKSPACE" and key == "allowed_base_paths":
                return str(tmp_path)
            return default

        with patch(
            "tldw_Server_API.app.core.config.get_config_value",
            side_effect=mock_config,
        ):
            with pytest.raises(HTTPException) as exc_info:
                _resolve_dispatch_cwd(str(nested.resolve()), workspace_root=str(workspace_root))

        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# _user_id_int fix tests
# ---------------------------------------------------------------------------


class TestUserIdInt:
    def test_numeric_id_int_attr(self):
        """User with id_int attribute should return it directly."""
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _user_id_int

        class MockUser:
            id_int = 42
            id = "42"

        assert _user_id_int(MockUser()) == 42

    def test_string_numeric_id(self):
        """User with string numeric ID but no id_int should convert."""
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _user_id_int

        class MockUser:
            id_int = None
            id = "123"

        assert _user_id_int(MockUser()) == 123

    def test_non_numeric_id_raises_400(self):
        """User with non-numeric ID should raise HTTPException 400."""
        from fastapi import HTTPException
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _user_id_int

        class MockUser:
            id_int = None
            id = "not-a-number"

        with pytest.raises(HTTPException) as exc_info:
            _user_id_int(MockUser())
        assert exc_info.value.status_code == 400

    def test_no_infinite_recursion(self):
        """Verify the _user_id_int fix doesn't recurse infinitely."""
        from fastapi import HTTPException
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import _user_id_int

        class MockUser:
            id_int = None
            id = None  # Will fail int() conversion

        # Should raise HTTPException, not RecursionError
        with pytest.raises(HTTPException):
            _user_id_int(MockUser())


# ---------------------------------------------------------------------------
# MCP resolver acp_workspace fallback
# ---------------------------------------------------------------------------


class TestMcpResolverAcpFallback:
    def test_resolve_acp_workspace_valid(self):
        """_resolve_acp_workspace should return root_path for valid workspace."""
        with tempfile.TemporaryDirectory() as tmp:
            db = OrchestrationDB(user_id=1, db_dir=tmp)
            ws = db.create_workspace(name="WS1", root_path="/test/path")

            # Patch get_orchestration_db where it's imported inside _resolve_acp_workspace
            with patch(
                "tldw_Server_API.app.core.Agent_Orchestration.orchestration_service.get_orchestration_db",
                return_value=db,
            ):
                result = McpHubWorkspaceRootResolver._resolve_acp_workspace("1", str(ws.id))
            assert result == "/test/path"
            db.close()

    def test_resolve_acp_workspace_not_found(self):
        """_resolve_acp_workspace should return None for nonexistent workspace."""
        with tempfile.TemporaryDirectory() as tmp:
            db = OrchestrationDB(user_id=1, db_dir=tmp)
            db._ensure_schema()
            with patch(
                "tldw_Server_API.app.core.Agent_Orchestration.orchestration_service.get_orchestration_db",
                return_value=db,
            ):
                result = McpHubWorkspaceRootResolver._resolve_acp_workspace("1", "99999")
            assert result is None
            db.close()

    def test_resolve_acp_workspace_non_numeric(self):
        """Non-numeric user/workspace IDs should return None."""
        result = McpHubWorkspaceRootResolver._resolve_acp_workspace("abc", "def")
        assert result is None

    def test_resolve_acp_workspace_empty(self):
        """Empty IDs should return None."""
        result = McpHubWorkspaceRootResolver._resolve_acp_workspace("", "")
        assert result is None
