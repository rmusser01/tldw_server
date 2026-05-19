"""Tests for ACP runner_cwd resolution relative to config file directory."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
    _resolve_cwd,
    _get_config_file_dir,
    load_acp_runner_config,
)


class TestResolveCwd:
    """Unit tests for _resolve_cwd helper."""

    def test_relative_path_resolved_against_config_dir(self):
        """A relative runner_cwd should be resolved against the config dir."""
        fake_config_dir = "/srv/tldw/Config_Files"
        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config._get_config_file_dir",
            return_value=fake_config_dir,
        ):
            result = _resolve_cwd("../tldw-agent")
        expected = os.path.normpath(os.path.join(fake_config_dir, "../tldw-agent"))
        assert result == expected

    def test_absolute_path_unchanged(self):
        """An absolute runner_cwd should be returned as-is."""
        abs_path = "/opt/agents/workspace"
        result = _resolve_cwd(abs_path)
        assert result == abs_path

    def test_none_returns_none(self):
        """None cwd should return None."""
        assert _resolve_cwd(None) is None

    def test_empty_string_returns_none(self):
        """Empty string cwd should return None."""
        assert _resolve_cwd("") is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only cwd should return None."""
        assert _resolve_cwd("   ") is None

    def test_relative_dot_path(self):
        """A dot-prefixed relative path should resolve correctly."""
        fake_config_dir = "/srv/tldw/Config_Files"
        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config._get_config_file_dir",
            return_value=fake_config_dir,
        ):
            result = _resolve_cwd("./agents")
        expected = os.path.normpath(os.path.join(fake_config_dir, "./agents"))
        assert result == expected


class TestLoadAcpRunnerConfigCwd:
    """Integration-level tests verifying cwd resolution in load_acp_runner_config."""

    def test_relative_cwd_resolved_in_loaded_config(self):
        """load_acp_runner_config should resolve a relative runner_cwd."""
        fake_config_dir = "/srv/tldw/Config_Files"
        fake_section = {"runner_command": "node", "runner_cwd": "../agent-workspace"}

        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config.get_config_section",
            return_value=fake_section,
        ), patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config._get_config_file_dir",
            return_value=fake_config_dir,
        ), patch.dict(os.environ, {}, clear=False):
            # Remove any env overrides that could interfere
            for key in ("ACP_RUNNER_CWD", "ACP_RUNNER_COMMAND", "ACP_RUNNER_ARGS",
                        "ACP_RUNNER_ENV", "ACP_RUNNER_BINARY_PATH", "ACP_RUNNER_STARTUP_TIMEOUT_MS"):
                os.environ.pop(key, None)

            cfg = load_acp_runner_config()

        expected = os.path.normpath(os.path.join(fake_config_dir, "../agent-workspace"))
        assert cfg.cwd == expected

    def test_absolute_cwd_unchanged_in_loaded_config(self):
        """load_acp_runner_config should leave an absolute runner_cwd unchanged."""
        fake_section = {"runner_command": "node", "runner_cwd": "/opt/workspace"}

        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config.get_config_section",
            return_value=fake_section,
        ), patch.dict(os.environ, {}, clear=False):
            for key in ("ACP_RUNNER_CWD", "ACP_RUNNER_COMMAND", "ACP_RUNNER_ARGS",
                        "ACP_RUNNER_ENV", "ACP_RUNNER_BINARY_PATH", "ACP_RUNNER_STARTUP_TIMEOUT_MS"):
                os.environ.pop(key, None)

            cfg = load_acp_runner_config()

        assert cfg.cwd == "/opt/workspace"

    def test_no_cwd_returns_none(self):
        """load_acp_runner_config should return None for cwd when not set."""
        fake_section = {"runner_command": "node"}

        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config.get_config_section",
            return_value=fake_section,
        ), patch.dict(os.environ, {}, clear=False):
            for key in ("ACP_RUNNER_CWD", "ACP_RUNNER_COMMAND", "ACP_RUNNER_ARGS",
                        "ACP_RUNNER_ENV", "ACP_RUNNER_BINARY_PATH", "ACP_RUNNER_STARTUP_TIMEOUT_MS"):
                os.environ.pop(key, None)

            cfg = load_acp_runner_config()

        assert cfg.cwd is None


class TestLoadAcpRunnerConfigEnv:
    """Integration-level tests verifying runner_env path handling."""

    def test_relative_home_in_runner_env_resolved_against_config_dir(self):
        """A relative HOME in runner_env should be resolved against the config dir."""
        fake_config_dir = "/srv/tldw/Config_Files"
        fake_section = {
            "runner_command": "node",
            "runner_env": "HOME=./acp_runner_home,PYTHONUNBUFFERED=1",
        }

        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config.get_config_section",
            return_value=fake_section,
        ), patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config._get_config_file_dir",
            return_value=fake_config_dir,
        ), patch.dict(os.environ, {}, clear=False):
            for key in (
                "ACP_RUNNER_CWD",
                "ACP_RUNNER_COMMAND",
                "ACP_RUNNER_ARGS",
                "ACP_RUNNER_ENV",
                "ACP_RUNNER_BINARY_PATH",
                "ACP_RUNNER_STARTUP_TIMEOUT_MS",
            ):
                os.environ.pop(key, None)

            cfg = load_acp_runner_config()

        expected_home = os.path.normpath(os.path.join(fake_config_dir, "./acp_runner_home"))
        assert cfg.env["HOME"] == expected_home
        assert cfg.env["PYTHONUNBUFFERED"] == "1"

    def test_absolute_home_in_runner_env_is_unchanged(self):
        """An absolute HOME in runner_env should be preserved as-is."""
        fake_section = {
            "runner_command": "node",
            "runner_env": "HOME=/opt/acp_runner_home,PYTHONUNBUFFERED=1",
        }

        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config.get_config_section",
            return_value=fake_section,
        ), patch.dict(os.environ, {}, clear=False):
            for key in (
                "ACP_RUNNER_CWD",
                "ACP_RUNNER_COMMAND",
                "ACP_RUNNER_ARGS",
                "ACP_RUNNER_ENV",
                "ACP_RUNNER_BINARY_PATH",
                "ACP_RUNNER_STARTUP_TIMEOUT_MS",
            ):
                os.environ.pop(key, None)

            cfg = load_acp_runner_config()

        assert cfg.env["HOME"] == "/opt/acp_runner_home"

    def test_relative_home_in_env_override_is_preserved(self):
        """An explicit ACP_RUNNER_ENV override should keep its original HOME value."""
        fake_config_dir = "/srv/tldw/Config_Files"
        fake_section = {
            "runner_command": "node",
            "runner_env": "HOME=./acp_runner_home,PYTHONUNBUFFERED=1",
        }

        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config.get_config_section",
            return_value=fake_section,
        ), patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config._get_config_file_dir",
            return_value=fake_config_dir,
        ), patch.dict(
            os.environ,
            {"ACP_RUNNER_ENV": "HOME=./override_home,PYTHONUNBUFFERED=1"},
            clear=False,
        ):
            for key in (
                "ACP_RUNNER_CWD",
                "ACP_RUNNER_COMMAND",
                "ACP_RUNNER_ARGS",
                "ACP_RUNNER_BINARY_PATH",
                "ACP_RUNNER_STARTUP_TIMEOUT_MS",
            ):
                os.environ.pop(key, None)

            cfg = load_acp_runner_config()

        assert cfg.env["HOME"] == "./override_home"
        assert cfg.env["PYTHONUNBUFFERED"] == "1"

    def test_empty_env_override_clears_config_runner_env(self):
        """An explicit empty ACP_RUNNER_ENV should override the config runner_env."""
        fake_section = {
            "runner_command": "node",
            "runner_env": "HOME=./acp_runner_home,PYTHONUNBUFFERED=1",
        }

        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config.get_config_section",
            return_value=fake_section,
        ), patch.dict(os.environ, {"ACP_RUNNER_ENV": ""}, clear=False):
            for key in (
                "ACP_RUNNER_CWD",
                "ACP_RUNNER_COMMAND",
                "ACP_RUNNER_ARGS",
                "ACP_RUNNER_BINARY_PATH",
                "ACP_RUNNER_STARTUP_TIMEOUT_MS",
            ):
                os.environ.pop(key, None)

            cfg = load_acp_runner_config()

        assert cfg.env == {}

    def test_empty_cwd_override_clears_config_runner_cwd(self):
        """An explicit empty ACP_RUNNER_CWD should override the config runner_cwd."""
        fake_section = {
            "runner_command": "node",
            "runner_cwd": "../agent-workspace",
        }

        with patch(
            "tldw_Server_API.app.core.Agent_Client_Protocol.config.get_config_section",
            return_value=fake_section,
        ), patch.dict(os.environ, {"ACP_RUNNER_CWD": ""}, clear=False):
            for key in (
                "ACP_RUNNER_COMMAND",
                "ACP_RUNNER_ARGS",
                "ACP_RUNNER_ENV",
                "ACP_RUNNER_BINARY_PATH",
                "ACP_RUNNER_STARTUP_TIMEOUT_MS",
            ):
                os.environ.pop(key, None)

            cfg = load_acp_runner_config()

        assert cfg.cwd is None
