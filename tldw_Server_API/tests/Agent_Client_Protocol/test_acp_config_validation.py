"""Tests for ACP config validation (validate_acp_config)."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
    ACPRunnerConfig,
    validate_acp_config,
)


class TestValidateAcpConfig:
    """Unit tests for validate_acp_config."""

    def test_no_warnings_when_valid(self):
        """A valid config with an existing command should produce no warnings."""
        # Use 'python' which is always on PATH during tests
        import sys

        cfg = ACPRunnerConfig(command=sys.executable, cwd=None)
        warnings = validate_acp_config(cfg)
        assert warnings == []

    def test_empty_command_warns(self):
        """An empty command should produce a warning about ACP sessions."""
        cfg = ACPRunnerConfig(command="", cwd=None)
        warnings = validate_acp_config(cfg)
        assert len(warnings) == 1
        assert "runner_command is empty" in warnings[0]
        assert "config.txt" in warnings[0]

    def test_missing_command_binary_warns(self):
        """A command not found on PATH or as a file should produce a warning."""
        cfg = ACPRunnerConfig(
            command="nonexistent-binary-that-does-not-exist-xyz123",
            cwd=None,
        )
        warnings = validate_acp_config(cfg)
        assert len(warnings) == 1
        assert "not found on PATH" in warnings[0]

    def test_nonexistent_cwd_warns(self):
        """A cwd pointing to a non-existent directory should produce a warning."""
        import sys

        cfg = ACPRunnerConfig(
            command=sys.executable,
            cwd="/nonexistent/directory/that/does/not/exist",
        )
        warnings = validate_acp_config(cfg)
        assert len(warnings) == 1
        assert "does not exist" in warnings[0]
        assert "/nonexistent/directory" in warnings[0]

    def test_valid_cwd_no_warning(self):
        """An existing cwd directory should not produce a warning."""
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ACPRunnerConfig(command=sys.executable, cwd=tmpdir)
            warnings = validate_acp_config(cfg)
        assert warnings == []

    def test_empty_command_and_bad_cwd_produces_two_warnings(self):
        """Multiple issues should produce multiple warnings."""
        cfg = ACPRunnerConfig(
            command="",
            cwd="/nonexistent/path/xyz",
        )
        warnings = validate_acp_config(cfg)
        assert len(warnings) == 2
        assert any("runner_command is empty" in w for w in warnings)
        assert any("does not exist" in w for w in warnings)

    def test_none_cwd_no_cwd_warning(self):
        """None cwd should not produce a cwd-related warning."""
        cfg = ACPRunnerConfig(
            command="nonexistent-binary-xyz",
            cwd=None,
        )
        warnings = validate_acp_config(cfg)
        # Should only have the command warning, not a cwd warning
        assert len(warnings) == 1
        assert "not found on PATH" in warnings[0]
