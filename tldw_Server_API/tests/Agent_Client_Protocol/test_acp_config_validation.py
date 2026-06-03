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
from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
    ACPAgentEntrypointStatus,
    ACPAgentRegisterRequest,
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


def test_agent_entrypoint_status_accepts_external_adapter() -> None:
    status = ACPAgentEntrypointStatus(
        profile_key="codex",
        entrypoint_strategy="external_acp_adapter",
        probe_state="blocked",
    )
    assert status.entrypoint_strategy == "external_acp_adapter"


def test_agent_entrypoint_status_imports_legacy_adapter_acp_alias() -> None:
    status = ACPAgentEntrypointStatus(
        profile_key="codex",
        entrypoint_strategy="adapter_acp",
        probe_state="blocked",
    )
    assert status.entrypoint_strategy == "external_acp_adapter"


def test_register_request_imports_legacy_adapter_acp_alias() -> None:
    request = ACPAgentRegisterRequest(
        agent_type="legacy_codex",
        name="Legacy Codex",
        entrypoint_strategy="adapter_acp",
    )
    assert request.entrypoint_strategy == "external_acp_adapter"


def test_static_codex_fallback_uses_external_adapter_and_delegated_credentials(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import _get_static_agents

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agents, _default_agent = _get_static_agents()
    codex = next(agent for agent in agents if agent.type == "codex")

    assert codex.requires_api_key is None
    assert codex.entrypoint.entrypoint_strategy == "external_acp_adapter"
    assert codex.entrypoint.acp_command == "codex-acp"
    assert codex.entrypoint.credential_state == "delegated"
    assert codex.entrypoint.primary_blocker in {
        "adapter_missing",
        "agent_binary_missing",
        "live_certification_required",
    }
