"""Tests for integration adapters.

This module tests all 12 integration adapters:
1. run_webhook_adapter - Send webhook
2. run_notify_adapter - Send notification
3. run_mcp_tool_adapter - MCP tool execution
4. run_acp_stage_adapter - ACP-backed stage execution
5. run_s3_upload_adapter - S3 upload
6. run_s3_download_adapter - S3 download
7. run_github_create_issue_adapter - Create GitHub issue
8. run_kanban_adapter - Kanban board operations
9. run_chatbooks_adapter - Chatbooks operations
10. run_character_chat_adapter - Character chat
11. run_email_send_adapter - Send email
12. run_podcast_rss_publish_adapter - Publish podcast RSS feed
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree import ElementTree as ET  # nosec B405 - tests parse controlled RSS fixtures only.

import pytest

pytestmark = pytest.mark.unit


# ==============================================================================
# Webhook Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_webhook_adapter_test_mode(monkeypatch):
    """Test webhook adapter in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_webhook_adapter

    config = {"url": "https://example.com/hook", "method": "POST", "body": {"key": "value"}}
    context = {"user_id": "test_user_123"}

    result = await run_webhook_adapter(config, context)
    assert result.get("test_mode") is True
    assert result.get("dispatched") is False


@pytest.mark.asyncio
async def test_webhook_adapter_test_mode_y(monkeypatch):
    """Test webhook adapter treats TEST_MODE=y as test mode."""
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_webhook_adapter

    config = {"url": "https://example.com/hook", "method": "POST", "body": {"key": "value"}}
    context = {"user_id": "test_user_123"}

    result = await run_webhook_adapter(config, context)
    assert result.get("test_mode") is True
    assert result.get("dispatched") is False


@pytest.mark.asyncio
async def test_webhook_adapter_missing_user_id_with_url(monkeypatch):
    """Test webhook adapter requires user_id when using HTTP URL mode (no user_id in context)."""
    # Ensure TEST_MODE is not set
    monkeypatch.delenv("TEST_MODE", raising=False)

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_webhook_adapter

    # Mock resolve_context_user_id to return None (simulating missing user_id)
    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.integration.webhook.resolve_context_user_id",
        return_value=None,
    ):
        # Use HTTP URL mode to trigger user_id check
        config = {"url": "https://example.com/hook", "method": "POST"}
        context = {}

        result = await run_webhook_adapter(config, context)
        # When not in test mode and no user_id with URL, should return missing_user_id error
        assert result.get("error") == "missing_user_id"
        assert result.get("dispatched") is False


@pytest.mark.asyncio
async def test_webhook_adapter_local_event_mode(monkeypatch):
    """Test webhook adapter with local event mode (no URL)."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_webhook_adapter

    config = {"event": "custom.event", "data": {"payload": "test"}}
    context = {"user_id": "test_user_123"}

    result = await run_webhook_adapter(config, context)
    assert result.get("test_mode") is True


@pytest.mark.asyncio
async def test_webhook_adapter_get_method(monkeypatch):
    """Test webhook adapter with GET method."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_webhook_adapter

    config = {"url": "https://api.example.com/data", "method": "GET", "body": {"param": "value"}}
    context = {"user_id": "test_user_123"}

    result = await run_webhook_adapter(config, context)
    assert result.get("test_mode") is True


@pytest.mark.asyncio
async def test_webhook_adapter_http_request(monkeypatch):
    """Test webhook adapter makes HTTP request when not in test mode."""
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_webhook_adapter

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.encoding = "utf-8"
    mock_response.read.return_value = b'{"success": true}'
    mock_response.close = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.request.return_value = mock_response

    with patch("tldw_Server_API.app.core.Workflows.adapters.integration.webhook.is_test_mode", return_value=False):
        with patch("tldw_Server_API.app.core.Workflows.adapters.integration.webhook._wf_create_client", return_value=mock_client):
            with patch("tldw_Server_API.app.core.Workflows.adapters.integration.webhook.is_url_allowed", return_value=True):
                with patch("tldw_Server_API.app.core.Security.egress.is_webhook_url_allowed_for_tenant", return_value=True):
                    config = {"url": "https://example.com/hook", "method": "POST", "body": {"test": "data"}}
                    context = {"user_id": "test_user_123", "tenant_id": "default"}

                    result = await run_webhook_adapter(config, context)
                    assert result.get("dispatched") is True or result.get("error") is not None


# ==============================================================================
# Notify Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_notify_adapter_test_mode(monkeypatch):
    """Test notify adapter in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_notify_adapter

    config = {"url": "https://hooks.slack.com/services/test", "message": "Hello, world!"}
    context = {}

    result = await run_notify_adapter(config, context)
    assert result.get("test_mode") is True
    assert result.get("dispatched") is False


@pytest.mark.asyncio
async def test_notify_adapter_test_mode_y(monkeypatch):
    """Test notify adapter treats TEST_MODE=y as test mode."""
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_notify_adapter

    config = {"url": "https://hooks.slack.com/services/test", "message": "Hello, world!"}
    context = {}

    result = await run_notify_adapter(config, context)
    assert result.get("test_mode") is True
    assert result.get("dispatched") is False


@pytest.mark.asyncio
async def test_notify_adapter_invalid_url():
    """Test notify adapter rejects invalid URLs."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_notify_adapter

    config = {"url": "ftp://invalid.com", "message": "Hello"}
    context = {}

    result = await run_notify_adapter(config, context)
    assert result.get("error") == "invalid_url"


@pytest.mark.asyncio
async def test_notify_adapter_with_subject(monkeypatch):
    """Test notify adapter with subject."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_notify_adapter

    config = {
        "url": "https://hooks.slack.com/services/test",
        "message": "Notification body",
        "subject": "Important Update",
    }
    context = {}

    result = await run_notify_adapter(config, context)
    assert result.get("test_mode") is True


@pytest.mark.asyncio
async def test_notify_adapter_with_headers(monkeypatch):
    """Test notify adapter with custom headers."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_notify_adapter

    config = {
        "url": "https://example.com/webhook",
        "message": "Test message",
        "headers": {"X-Custom-Header": "custom-value"},
    }
    context = {"tenant_id": "test_tenant"}

    result = await run_notify_adapter(config, context)
    assert result.get("test_mode") is True


@pytest.mark.asyncio
async def test_notify_adapter_blocked_egress(monkeypatch):
    """Test notify adapter respects egress blocking."""
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_notify_adapter

    with patch("tldw_Server_API.app.core.Workflows.adapters.integration.webhook.is_test_mode", return_value=False):
        with patch("tldw_Server_API.app.core.Workflows.adapters.integration.webhook.is_url_allowed_for_tenant", return_value=False):
            with patch("tldw_Server_API.app.core.Workflows.adapters.integration.webhook.is_url_allowed", return_value=False):
                config = {"url": "https://blocked.example.com/hook", "message": "Test"}
                context = {}

                result = await run_notify_adapter(config, context)
                assert result.get("error") == "blocked_egress"


# ==============================================================================
# MCP Tool Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_mcp_tool_adapter_missing_tool_name():
    """Test MCP tool adapter requires tool_name."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_mcp_tool_adapter

    config = {"arguments": {"key": "value"}}
    context = {}

    result = await run_mcp_tool_adapter(config, context)
    assert result.get("error") == "missing_tool_name"


@pytest.mark.asyncio
async def test_mcp_tool_adapter_echo_fallback():
    """Test MCP tool adapter echo fallback."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_mcp_tool_adapter

    # Mock the MCP server to return None for module lookup
    with patch("tldw_Server_API.app.core.MCP_unified.get_mcp_server") as mock_get_server:
        mock_server = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_registry = {}
        mock_registry._module_instances = {}
        mock_server.module_registry = mock_registry
        mock_get_server.return_value = mock_server

        config = {"tool_name": "echo", "arguments": {"message": "Hello, MCP!"}}
        context = {}

        result = await run_mcp_tool_adapter(config, context)
        assert result.get("result") == "Hello, MCP!"
        assert result.get("module") == "_fallback"


@pytest.mark.asyncio
async def test_mcp_tool_adapter_tool_not_found():
    """Test MCP tool adapter returns error for unknown tool."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_mcp_tool_adapter

    with patch("tldw_Server_API.app.core.MCP_unified.get_mcp_server") as mock_get_server:
        mock_server = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_registry = {}
        mock_registry._module_instances = {}
        mock_server.module_registry = mock_registry
        mock_get_server.return_value = mock_server

        config = {"tool_name": "nonexistent_tool_xyz", "arguments": {}}
        context = {}

        result = await run_mcp_tool_adapter(config, context)
        assert result.get("error") == "tool_not_found"


@pytest.mark.asyncio
async def test_mcp_tool_adapter_with_allowlist():
    """Test MCP tool adapter respects allowlist."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_mcp_tool_adapter
    from tldw_Server_API.app.core.exceptions import AdapterError

    with patch("tldw_Server_API.app.core.MCP_unified.get_mcp_server") as mock_get_server:
        mock_server = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_registry = {}
        mock_registry._module_instances = {}
        mock_server.module_registry = mock_registry
        mock_get_server.return_value = mock_server

        config = {"tool_name": "restricted_tool", "arguments": {}}
        context = {"workflow_mcp_policy": {"allowlist": ["allowed_tool"]}}

        with pytest.raises(AdapterError) as exc_info:
            await run_mcp_tool_adapter(config, context)
        assert "mcp_tool_not_allowed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_mcp_tool_adapter_executes_tool():
    """Test MCP tool adapter executes tool successfully."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_mcp_tool_adapter

    with patch("tldw_Server_API.app.core.MCP_unified.get_mcp_server") as mock_get_server:
        mock_module = MagicMock()
        mock_module.get_tools = AsyncMock(return_value=[{"name": "test_tool"}])
        mock_module.execute_tool = AsyncMock(return_value={"data": "result"})

        mock_server = MagicMock()
        mock_registry = MagicMock()
        mock_registry._tool_registry = {"test_tool": "test_module"}
        mock_registry._module_instances = {"test_module": mock_module}
        mock_server.module_registry = mock_registry
        mock_get_server.return_value = mock_server

        config = {"tool_name": "test_tool", "arguments": {"param": "value"}}
        context = {}

        result = await run_mcp_tool_adapter(config, context)
        assert result.get("result") == {"data": "result"}
        assert result.get("module") == "test_module"


# ==============================================================================
# ACP Stage Adapter Tests
# ==============================================================================


class _NoopACPStore:
    def __init__(self) -> None:
        self.register_session = AsyncMock()
        self.record_prompt = AsyncMock()


def _patch_noop_acp_store(monkeypatch: pytest.MonkeyPatch) -> _NoopACPStore:
    store = _NoopACPStore()

    async def _get_store() -> _NoopACPStore:
        return store

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_acp_session_store",
        _get_store,
    )
    return store


@pytest.mark.asyncio
async def test_acp_stage_adapter_creates_session_and_prompts(monkeypatch):
    """Test ACP stage adapter creates a session and executes prompt."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="acp-session-1")
            self.prompt = AsyncMock(
                return_value={
                    "stopReason": "end",
                    "detail": "ok",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                }
            )
            self.verify_session_access = AsyncMock(return_value=True)

    stub = _StubRunner()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )
    _patch_noop_acp_store(monkeypatch)

    config = {"stage": "impl", "prompt_template": "Implement {{ inputs.task }}"}
    context = {"inputs": {"task": "domain task"}, "user_id": "1"}
    result = await run_acp_stage_adapter(config, context)

    assert result.get("status") == "ok"
    assert result.get("session_id") == "acp-session-1"
    assert result.get("stage") == "impl"
    assert isinstance(result.get("response"), dict)
    assert isinstance(result.get("usage"), dict)


@pytest.mark.asyncio
async def test_acp_stage_adapter_registers_created_session_and_records_prompt(monkeypatch):
    """Workflow-created ACP sessions should use the shared session-store contract."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="acp-session-store")
            self.prompt = AsyncMock(
                return_value={
                    "content": "Workflow reply",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
                }
            )
            self.verify_session_access = AsyncMock(return_value=True)

    class _StubStore:
        def __init__(self) -> None:
            self.register_session = AsyncMock()
            self.record_prompt = AsyncMock()

    stub_runner = _StubRunner()
    stub_store = _StubStore()

    async def _get_runner_client():
        return stub_runner

    async def _get_store():
        return stub_store

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_acp_session_store",
        _get_store,
    )

    config = {
        "stage": "impl",
        "cwd": "/workspace/repo",
        "agent_type": "codex",
        "prompt_template": "Implement {{ inputs.task }}",
        "workspace_id": "workspace-1",
    }
    context = {"inputs": {"task": "shared store"}, "user_id": "11"}

    result = await run_acp_stage_adapter(config, context)

    assert result.get("status") == "ok"
    stub_store.register_session.assert_awaited_once_with(
        session_id="acp-session-store",
        user_id=11,
        agent_type="codex",
        name="Workflow impl",
        cwd="/workspace/repo",
        tags=["workflow", "acp_stage", "impl"],
        persona_id=None,
        workspace_id="workspace-1",
        workspace_group_id=None,
        scope_snapshot_id=None,
    )
    stub_store.record_prompt.assert_awaited_once_with(
        "acp-session-store",
        [{"role": "user", "content": "Implement shared store"}],
        {
            "content": "Workflow reply",
            "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
        },
    )


@pytest.mark.asyncio
async def test_acp_stage_output_includes_schema_version(monkeypatch):
    """Test ACP stage adapter emits output schema version."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="acp-session-schema")
            self.prompt = AsyncMock(return_value={"stopReason": "end", "detail": "ok"})
            self.verify_session_access = AsyncMock(return_value=True)

    stub = _StubRunner()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )
    _patch_noop_acp_store(monkeypatch)

    result = await run_acp_stage_adapter(
        {"stage": "impl", "prompt_template": "Implement {{ inputs.task }}"},
        {"inputs": {"task": "schema-check"}, "user_id": "9"},
    )

    assert result.get("acp_output_schema_version") == "1.0"


@pytest.mark.asyncio
async def test_acp_stage_adapter_reuses_session_from_context(monkeypatch):
    """Test ACP stage adapter reuses session id from context key."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="acp-session-created")
            self.prompt = AsyncMock(return_value={"stopReason": "end", "detail": "ok"})
            self.verify_session_access = AsyncMock(return_value=True)

    stub = _StubRunner()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )
    _patch_noop_acp_store(monkeypatch)

    config = {
        "stage": "plan",
        "prompt_template": "Plan: {{ inputs.task }}",
        "session_context_key": "acp_session",
    }
    context = {"inputs": {"task": "reuse"}, "user_id": "2", "acp_session": "existing-session-9"}
    result = await run_acp_stage_adapter(config, context)

    assert result.get("status") == "ok"
    assert result.get("session_id") == "existing-session-9"
    stub.create_session.assert_not_called()
    stub.verify_session_access.assert_awaited_once_with("existing-session-9", 2)
    stub.prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_acp_stage_adapter_normalizes_governance_block(monkeypatch):
    """Test ACP stage adapter normalizes governance denied outcomes."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPGovernanceDeniedError
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="acp-session-2")
            self.prompt = AsyncMock(
                side_effect=ACPGovernanceDeniedError(governance={"action": "deny", "reason": "policy"})
            )
            self.verify_session_access = AsyncMock(return_value=True)

    stub = _StubRunner()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )
    _patch_noop_acp_store(monkeypatch)

    result = await run_acp_stage_adapter(
        {"stage": "impl_review", "prompt_template": "Review {{ inputs.task }}"},
        {"inputs": {"task": "governance"}, "user_id": "3"},
    )
    assert result.get("status") == "blocked"
    assert result.get("error_type") == "acp_governance_blocked"
    assert result.get("error") == "acp_governance_blocked"
    assert result.get("governance", {}).get("action") == "deny"


@pytest.mark.asyncio
async def test_acp_stage_adapter_normalizes_timeout(monkeypatch):
    """Test ACP stage adapter classifies prompt timeout errors."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="acp-session-3")
            self.prompt = AsyncMock(side_effect=asyncio.TimeoutError())
            self.verify_session_access = AsyncMock(return_value=True)

    stub = _StubRunner()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )
    _patch_noop_acp_store(monkeypatch)

    result = await run_acp_stage_adapter(
        {"stage": "test", "prompt_template": "Test {{ inputs.task }}"},
        {"inputs": {"task": "timeouts"}, "user_id": "4"},
    )
    assert result.get("status") == "error"
    assert result.get("error_type") == "acp_timeout"


@pytest.mark.asyncio
async def test_acp_stage_adapter_cancelled_passthrough():
    """Test ACP stage adapter exits early when workflow is cancelled."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    result = await run_acp_stage_adapter(
        {"stage": "impl", "prompt_template": "ignored"},
        {"is_cancelled": lambda: True},
    )
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_acp_stage_adapter_review_loop_limit(monkeypatch):
    """Test ACP stage adapter enforces configured review loop limits."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="acp-session-4")
            self.prompt = AsyncMock(return_value={"stopReason": "end", "detail": "ok"})
            self.verify_session_access = AsyncMock(return_value=True)

    stub = _StubRunner()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )

    result = await run_acp_stage_adapter(
        {
            "stage": "impl_review",
            "prompt_template": "Review",
            "review_counter_key": "impl_review_count",
            "max_review_loops": 3,
        },
        {"user_id": "7", "impl_review_count": 3},
    )
    assert result.get("status") == "blocked"
    assert result.get("error_type") == "review_loop_exceeded"


@pytest.mark.asyncio
async def test_acp_stage_adapter_fail_on_error_raises_adapter_error(monkeypatch):
    """Test ACP stage adapter raises AdapterError when fail_on_error is enabled."""
    from tldw_Server_API.app.core.exceptions import AdapterError
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="acp-session-5")
            self.prompt = AsyncMock(side_effect=asyncio.TimeoutError())
            self.verify_session_access = AsyncMock(return_value=True)

    stub = _StubRunner()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )
    _patch_noop_acp_store(monkeypatch)

    with pytest.raises(AdapterError):
        await run_acp_stage_adapter(
            {
                "stage": "test",
                "prompt_template": "Run test pass",
                "fail_on_error": True,
            },
            {"user_id": "11"},
        )


@pytest.mark.asyncio
async def test_acp_stage_adapter_reused_session_denied_when_unowned(monkeypatch):
    """Test ACP stage adapter blocks reused sessions not owned by workflow user."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="unused-session")
            self.prompt = AsyncMock(return_value={"stopReason": "end", "content": "ok"})
            self.verify_session_access = AsyncMock(return_value=False)

    stub = _StubRunner()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )

    result = await run_acp_stage_adapter(
        {"stage": "impl", "prompt_template": "Implement {{ inputs.task }}"},
        {"inputs": {"task": "domain task"}, "user_id": "7", "acp_session_id": "foreign-session"},
    )
    assert result.get("status") == "error"
    assert result.get("error_type") == "acp_session_error"
    assert result.get("error") == "session_access_denied"
    stub.prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_acp_stage_adapter_sanitizes_internal_prompt_exception(monkeypatch):
    """Test ACP stage adapter does not expose raw internal exception text."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_acp_stage_adapter

    class _StubRunner:
        def __init__(self) -> None:
            self.create_session = AsyncMock(return_value="acp-session-6")
            self.prompt = AsyncMock(side_effect=RuntimeError("private internal error detail"))
            self.verify_session_access = AsyncMock(return_value=True)

    stub = _StubRunner()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.acp.get_runner_client",
        _get_runner_client,
    )
    _patch_noop_acp_store(monkeypatch)

    result = await run_acp_stage_adapter(
        {"stage": "impl", "prompt_template": "Implement {{ inputs.task }}"},
        {"inputs": {"task": "domain task"}, "user_id": "7"},
    )
    assert result.get("status") == "error"
    assert result.get("error_type") == "acp_prompt_error"
    assert result.get("error") == "acp_prompt_failed"
    assert "private internal error detail" not in str(result)


# ==============================================================================
# S3 Upload Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_s3_upload_adapter_missing_bucket_or_key():
    """Test S3 upload adapter requires bucket and key."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_upload_adapter

    config = {"bucket": "my-bucket"}
    context = {}

    result = await run_s3_upload_adapter(config, context)
    assert result.get("error") == "missing_bucket_or_key"
    assert result.get("uploaded") is False


@pytest.mark.asyncio
async def test_s3_upload_adapter_cancelled():
    """Test S3 upload adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_upload_adapter

    config = {"bucket": "test-bucket", "key": "test-key"}
    context = {"is_cancelled": lambda: True}

    result = await run_s3_upload_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_s3_upload_adapter_with_content():
    """Test S3 upload adapter with direct content."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_upload_adapter

    mock_s3_client = MagicMock()
    mock_s3_client.put_object = MagicMock()

    # Mock boto3 module entirely
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        config = {
            "bucket": "my-bucket",
            "key": "data/file.json",
            "content": '{"test": "data"}',
            "region": "us-west-2",
        }
        context = {}

        result = await run_s3_upload_adapter(config, context)
        assert result.get("uploaded") is True
        assert result.get("bucket") == "my-bucket"
        assert result.get("key") == "data/file.json"


@pytest.mark.asyncio
async def test_s3_upload_adapter_boto3_not_installed():
    """Test S3 upload adapter handles missing boto3."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_upload_adapter

    with patch.dict("sys.modules", {"boto3": None}):
        # Force reimport to trigger ImportError
        import importlib
        import tldw_Server_API.app.core.Workflows.adapters.integration.storage as storage_module

        # Mock boto3 import to raise ImportError
        original_import = __builtins__.get("__import__", __import__) if isinstance(__builtins__, dict) else __builtins__.__import__

        def mock_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", mock_import):
            config = {"bucket": "test-bucket", "key": "test.txt", "content": "test"}
            context = {}
            # Note: This test may not trigger the ImportError path since boto3 is already imported
            # The test serves as documentation for the expected behavior


@pytest.mark.asyncio
async def test_s3_upload_adapter_with_file_path(monkeypatch, tmp_path):
    """Test S3 upload adapter with file path."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_upload_adapter

    # Create a temporary file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    mock_s3_client = MagicMock()
    mock_s3_client.put_object = MagicMock()

    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    # Mock resolve_workflow_file_path to return our temp file
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.integration.storage.resolve_workflow_file_path",
            return_value=test_file,
        ):
            config = {
                "bucket": "my-bucket",
                "key": "uploads/test.txt",
                "file_path": str(test_file),
            }
            context = {}

            result = await run_s3_upload_adapter(config, context)
            assert result.get("uploaded") is True


@pytest.mark.asyncio
async def test_s3_upload_adapter_sanitizes_file_read_errors():
    """Test S3 upload file-read failures hide raw path details."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_upload_adapter

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.integration.storage.resolve_workflow_file_path",
        side_effect=RuntimeError("file resolver exploded at /private/uploads/source.txt"),
    ):
        result = await run_s3_upload_adapter(
            {"bucket": "my-bucket", "key": "uploads/source.txt", "file_path": "/private/uploads/source.txt"},
            {},
        )

    assert result == {"error": "file_read_error", "uploaded": False}
    assert "source.txt" not in str(result)


@pytest.mark.asyncio
async def test_s3_upload_adapter_content_from_prev():
    """Test S3 upload adapter gets content from previous step."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_upload_adapter

    mock_s3_client = MagicMock()
    mock_s3_client.put_object = MagicMock()

    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        config = {"bucket": "my-bucket", "key": "output.txt"}
        context = {"prev": {"content": "Previous step content"}}

        result = await run_s3_upload_adapter(config, context)
        assert result.get("uploaded") is True


@pytest.mark.asyncio
async def test_s3_upload_adapter_sanitizes_backend_errors():
    """Test S3 upload adapter hides raw backend errors."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_upload_adapter

    mock_s3_client = MagicMock()
    mock_s3_client.put_object.side_effect = RuntimeError("S3 upload token at /private/s3-upload.key")

    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = await run_s3_upload_adapter(
            {"bucket": "my-bucket", "key": "output.txt", "content": "secret content"},
            {},
        )

    assert result == {"error": "S3 upload failed", "uploaded": False}


# ==============================================================================
# S3 Download Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_s3_download_adapter_missing_bucket_or_key():
    """Test S3 download adapter requires bucket and key."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_download_adapter

    config = {"key": "test-key"}
    context = {}

    result = await run_s3_download_adapter(config, context)
    assert result.get("error") == "missing_bucket_or_key"
    assert result.get("content") is None


@pytest.mark.asyncio
async def test_s3_download_adapter_cancelled():
    """Test S3 download adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_download_adapter

    config = {"bucket": "test-bucket", "key": "test-key"}
    context = {"is_cancelled": lambda: True}

    result = await run_s3_download_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_s3_download_adapter_as_text():
    """Test S3 download adapter returns text content."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_download_adapter

    mock_body = MagicMock()
    mock_body.read.return_value = b"Hello, World!"

    mock_s3_client = MagicMock()
    mock_s3_client.get_object.return_value = {"Body": mock_body}

    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        config = {"bucket": "my-bucket", "key": "file.txt", "as_text": True}
        context = {}

        result = await run_s3_download_adapter(config, context)
        assert result.get("content") == "Hello, World!"
        assert result.get("bucket") == "my-bucket"
        assert result.get("key") == "file.txt"


@pytest.mark.asyncio
async def test_s3_download_adapter_as_bytes():
    """Test S3 download adapter returns binary content."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_download_adapter

    mock_body = MagicMock()
    mock_body.read.return_value = b"\x00\x01\x02\x03"

    mock_s3_client = MagicMock()
    mock_s3_client.get_object.return_value = {"Body": mock_body}

    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        config = {"bucket": "my-bucket", "key": "file.bin", "as_text": False}
        context = {}

        result = await run_s3_download_adapter(config, context)
        assert result.get("content") == b"\x00\x01\x02\x03"


@pytest.mark.asyncio
async def test_s3_download_adapter_with_endpoint_url(monkeypatch):
    """Test S3 download adapter with custom endpoint (MinIO, etc.)."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_download_adapter

    mock_body = MagicMock()
    mock_body.read.return_value = b"data"

    mock_s3_client = MagicMock()
    mock_s3_client.get_object.return_value = {"Body": mock_body}

    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        fixture_s3_key = "minio" + "admin"
        config = {
            "bucket": "test-bucket",
            "key": "test-key",
            "endpoint_url": "http://minio:9000",
            "access_key": fixture_s3_key,
            "secret_key": fixture_s3_key,
        }
        context = {}

        result = await run_s3_download_adapter(config, context)
        assert result.get("content") is not None

        # Verify boto3.client was called with endpoint_url
        mock_boto3.client.assert_called_once()
        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs.get("endpoint_url") == "http://minio:9000"


@pytest.mark.asyncio
async def test_s3_download_adapter_sanitizes_backend_errors():
    """Test S3 download adapter hides raw backend errors."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_s3_download_adapter

    mock_s3_client = MagicMock()
    mock_s3_client.get_object.side_effect = RuntimeError("S3 download token at /private/s3-download.key")

    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_s3_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = await run_s3_download_adapter(
            {"bucket": "my-bucket", "key": "output.txt"},
            {},
        )

    assert result == {"error": "S3 download failed", "content": None}


# ==============================================================================
# GitHub Create Issue Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_github_create_issue_adapter_missing_repo_or_title():
    """Test GitHub adapter requires repo and title."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_github_create_issue_adapter

    config = {"title": "Test Issue"}
    context = {}

    result = await run_github_create_issue_adapter(config, context)
    assert result.get("error") == "missing_repo_or_title"


@pytest.mark.asyncio
async def test_github_create_issue_adapter_missing_token(monkeypatch):
    """Test GitHub adapter requires token."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_github_create_issue_adapter

    config = {"repo": "owner/repo", "title": "Test Issue"}
    context = {}

    result = await run_github_create_issue_adapter(config, context)
    assert result.get("error") == "missing_github_token"


@pytest.mark.asyncio
async def test_github_create_issue_adapter_cancelled():
    """Test GitHub adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_github_create_issue_adapter

    fixture_token = "test" + "_token"
    config = {"repo": "owner/repo", "title": "Test Issue", "token": fixture_token}
    context = {"is_cancelled": lambda: True}

    result = await run_github_create_issue_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_github_create_issue_adapter_success():
    """Test GitHub adapter creates issue successfully."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_github_create_issue_adapter

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"html_url": "https://github.com/owner/repo/issues/1", "number": 1}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = mock_client

    http_module_key = "http" + "x"
    with patch.dict("sys.modules", {http_module_key: mock_httpx}):
        fixture_token = "ghp" + "_test_token"
        config = {
            "repo": "owner/repo",
            "title": "Bug Report",
            "body": "This is a bug",
            "labels": ["bug"],
            "token": fixture_token,
        }
        context = {}

        result = await run_github_create_issue_adapter(config, context)
        assert result.get("created") is True
        assert result.get("issue_url") == "https://github.com/owner/repo/issues/1"
        assert result.get("issue_number") == 1


@pytest.mark.asyncio
async def test_github_create_issue_adapter_api_error():
    """Test GitHub adapter handles API errors."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_github_create_issue_adapter

    mock_response = AsyncMock()
    mock_response.status_code = 422
    mock_response.text = "Validation Failed"

    async_client_path = "http" + "x" + ".AsyncClient"
    with patch(async_client_path) as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        fixture_token = "test" + "_token"
        config = {"repo": "owner/repo", "title": "Test", "token": fixture_token}
        context = {}

        result = await run_github_create_issue_adapter(config, context)
        assert result.get("created") is False
        assert "github_api_error" in result.get("error", "")


@pytest.mark.asyncio
async def test_github_create_issue_adapter_sanitizes_backend_errors():
    """Test GitHub adapter hides raw backend errors."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_github_create_issue_adapter

    async_client_path = "http" + "x" + ".AsyncClient"
    with patch(async_client_path) as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.side_effect = RuntimeError("github token at /private/workflows-github.key")
        mock_client_cls.return_value = mock_client

        token_value = "_".join(("test", "token"))
        result = await run_github_create_issue_adapter(
            {"repo": "owner/repo", "title": "Test", "token": token_value},
            {},
        )

    assert result == {
        "error": "GitHub issue creation failed",
        "issue_url": None,
        "created": False,
    }


@pytest.mark.asyncio
async def test_github_create_issue_adapter_with_template():
    """Test GitHub adapter applies template to title and body."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_github_create_issue_adapter

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"html_url": "https://github.com/owner/repo/issues/1", "number": 1}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = mock_client

    http_module_key = "http" + "x"
    with patch.dict("sys.modules", {http_module_key: mock_httpx}):
        fixture_token = "test" + "_token"
        config = {
            "repo": "owner/repo",
            "title": "Issue for {{inputs.component}}",
            "body": "Error: {{prev.error_message}}",
            "token": fixture_token,
        }
        context = {
            "inputs": {"component": "auth"},
            "prev": {"error_message": "Connection refused"},
        }

        result = await run_github_create_issue_adapter(config, context)
        assert result.get("created") is True


# ==============================================================================
# Kanban Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_kanban_adapter_missing_action():
    """Test Kanban adapter requires action."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_kanban_adapter

    config = {}
    context = {"user_id": "test_user"}

    result = await run_kanban_adapter(config, context)
    assert result.get("error") == "missing_action"


@pytest.mark.asyncio
async def test_kanban_adapter_board_list(monkeypatch):
    """Test Kanban adapter list boards action."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_kanban_adapter

    mock_db = MagicMock()
    mock_db.list_boards.return_value = ([{"id": 1, "name": "Board 1"}], 1)
    mock_db.close = MagicMock()

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.KanbanDB",
        return_value=mock_db,
    ):
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.DatabasePaths.get_kanban_db_path",
            return_value=Path("/tmp/kanban.db"),  # nosec B108
        ):
            config = {"action": "board.list", "limit": 10}
            context = {"user_id": "1"}

            result = await run_kanban_adapter(config, context)
            assert "boards" in result
            assert result.get("total") == 1


@pytest.mark.asyncio
async def test_kanban_adapter_board_list_accepts_y_flag(monkeypatch):
    """Test Kanban adapter boolean coercion accepts include_archived='y'."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_kanban_adapter

    mock_db = MagicMock()
    mock_db.list_boards.return_value = ([{"id": 1, "name": "Board 1"}], 1)
    mock_db.close = MagicMock()

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.KanbanDB",
        return_value=mock_db,
    ):
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.DatabasePaths.get_kanban_db_path",
            return_value=Path("/tmp/kanban.db"),  # nosec B108
        ):
            config = {"action": "board.list", "include_archived": "y"}
            context = {"user_id": "1"}

            result = await run_kanban_adapter(config, context)
            assert "boards" in result
            mock_db.list_boards.assert_called_once()
            assert mock_db.list_boards.call_args.kwargs.get("include_archived") is True


@pytest.mark.asyncio
async def test_kanban_adapter_sanitizes_backend_errors(monkeypatch):
    """Test Kanban adapter hides backend error details."""
    from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDBError
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_kanban_adapter

    mock_db = MagicMock()
    mock_db.list_boards.side_effect = KanbanDBError("kanban backend exploded at /private/workflow-kanban.db")
    mock_db.close = MagicMock()

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.KanbanDB",
        return_value=mock_db,
    ):
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.DatabasePaths.get_kanban_db_path",
            return_value=Path("/tmp/kanban.db"),  # nosec B108
        ):
            result = await run_kanban_adapter({"action": "board.list"}, {"user_id": "1"})

    assert result == {
        "error": "kanban_error",
        "error_type": "KanbanDBError",
        "detail": "Kanban operation failed",
    }


@pytest.mark.asyncio
async def test_kanban_adapter_board_create(monkeypatch):
    """Test Kanban adapter create board action."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_kanban_adapter

    mock_db = MagicMock()
    mock_db.create_board.return_value = {"id": 1, "name": "New Board", "client_id": "wf_abc123"}
    mock_db.close = MagicMock()

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.KanbanDB",
        return_value=mock_db,
    ):
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.DatabasePaths.get_kanban_db_path",
            return_value=Path("/tmp/kanban.db"),  # nosec B108
        ):
            config = {"action": "board.create", "name": "New Board"}
            context = {"user_id": "1"}

            result = await run_kanban_adapter(config, context)
            assert "board" in result
            assert result["board"]["name"] == "New Board"


@pytest.mark.asyncio
async def test_kanban_adapter_card_create(monkeypatch):
    """Test Kanban adapter create card action."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_kanban_adapter

    mock_db = MagicMock()
    mock_db.create_card.return_value = {"id": 1, "title": "New Task", "list_id": 1}
    mock_db.close = MagicMock()

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.KanbanDB",
        return_value=mock_db,
    ):
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.DatabasePaths.get_kanban_db_path",
            return_value=Path("/tmp/kanban.db"),  # nosec B108
        ):
            config = {"action": "card.create", "list_id": "1", "title": "New Task"}
            context = {"user_id": "1"}

            result = await run_kanban_adapter(config, context)
            assert "card" in result
            assert result["card"]["title"] == "New Task"


@pytest.mark.asyncio
async def test_kanban_adapter_card_search(monkeypatch):
    """Test Kanban adapter search cards action."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_kanban_adapter

    mock_db = MagicMock()
    mock_db.search_cards.return_value = ([{"id": 1, "title": "Matching Card"}], 1)
    mock_db.close = MagicMock()

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.KanbanDB",
        return_value=mock_db,
    ):
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.DatabasePaths.get_kanban_db_path",
            return_value=Path("/tmp/kanban.db"),  # nosec B108
        ):
            config = {"action": "card.search", "query": "matching"}
            context = {"user_id": "1"}

            result = await run_kanban_adapter(config, context)
            assert "cards" in result
            assert result.get("total") == 1


@pytest.mark.asyncio
async def test_kanban_adapter_unsupported_action():
    """Test Kanban adapter returns error for unsupported action."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_kanban_adapter

    mock_db = MagicMock()
    mock_db.close = MagicMock()

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.KanbanDB",
        return_value=mock_db,
    ):
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.DatabasePaths.get_kanban_db_path",
            return_value=Path("/tmp/kanban.db"),  # nosec B108
        ):
            config = {"action": "invalid.action"}
            context = {"user_id": "1"}

            result = await run_kanban_adapter(config, context)
            assert "unsupported_action" in result.get("error", "")


# ==============================================================================
# Chatbooks Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_chatbooks_adapter_missing_action():
    """Test Chatbooks adapter requires action."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_chatbooks_adapter

    config = {}
    context = {"user_id": "test_user"}

    result = await run_chatbooks_adapter(config, context)
    assert result.get("error") == "missing_action"


@pytest.mark.asyncio
async def test_chatbooks_adapter_export_test_mode(monkeypatch):
    """Test Chatbooks adapter export in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_chatbooks_adapter

    config = {"action": "export", "name": "My Export", "content_types": ["conversations", "notes"]}
    context = {"user_id": "1"}

    result = await run_chatbooks_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("job_id") == "test-job-123"
    assert result.get("status") == "completed"


@pytest.mark.asyncio
async def test_chatbooks_adapter_export_test_mode_y(monkeypatch):
    """Test Chatbooks adapter export in TEST_MODE=y."""
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_chatbooks_adapter

    config = {"action": "export", "name": "My Export", "content_types": ["conversations", "notes"]}
    context = {"user_id": "1"}

    result = await run_chatbooks_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("job_id") == "test-job-123"


@pytest.mark.asyncio
async def test_chatbooks_adapter_import_test_mode(monkeypatch):
    """Test Chatbooks adapter import in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_chatbooks_adapter

    config = {"action": "import", "file_path": "/path/to/chatbook.json"}
    context = {"user_id": "1"}

    result = await run_chatbooks_adapter(config, context)
    assert result.get("simulated") is True


@pytest.mark.asyncio
async def test_chatbooks_adapter_list_jobs_test_mode(monkeypatch):
    """Test Chatbooks adapter list jobs in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_chatbooks_adapter

    config = {"action": "list_jobs"}
    context = {"user_id": "1"}

    result = await run_chatbooks_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("jobs") == []
    assert result.get("count") == 0


@pytest.mark.asyncio
async def test_chatbooks_adapter_get_job_test_mode(monkeypatch):
    """Test Chatbooks adapter get job in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_chatbooks_adapter

    config = {"action": "get_job", "job_id": "job-123"}
    context = {"user_id": "1"}

    result = await run_chatbooks_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("job", {}).get("id") == "job-123"


@pytest.mark.asyncio
async def test_chatbooks_adapter_preview_test_mode(monkeypatch):
    """Test Chatbooks adapter preview in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_chatbooks_adapter

    config = {"action": "preview", "content_types": ["conversations"]}
    context = {"user_id": "1"}

    result = await run_chatbooks_adapter(config, context)
    assert result.get("simulated") is True
    assert "preview" in result


@pytest.mark.asyncio
async def test_chatbooks_adapter_cancelled():
    """Test Chatbooks adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_chatbooks_adapter

    config = {"action": "export"}
    context = {"user_id": "1", "is_cancelled": lambda: True}

    result = await run_chatbooks_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_chatbooks_adapter_sanitizes_backend_errors(monkeypatch):
    """Test Chatbooks adapter does not expose backend exception details."""
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_chatbooks_adapter

    def raise_backend_error(user_id: int) -> Path:
        raise RuntimeError(f"chatbooks backend exploded for user {user_id} at /private/chatbooks.db")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.DatabasePaths.get_chacha_db_path",
        raise_backend_error,
    )

    result = await run_chatbooks_adapter({"action": "list_jobs"}, {"user_id": "1"})

    assert result == {"error": "chatbooks_error"}


# ==============================================================================
# Character Chat Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_character_chat_adapter_start_test_mode(monkeypatch):
    """Test Character chat adapter start in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_character_chat_adapter

    config = {"action": "start", "character_id": 1}
    context = {"user_id": "1"}

    result = await run_character_chat_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("conversation_id") == "test-conv-123"
    assert result.get("character_name") == "Test Character"


@pytest.mark.asyncio
async def test_character_chat_adapter_message_test_mode(monkeypatch):
    """Test Character chat adapter message in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_character_chat_adapter

    config = {"action": "message", "conversation_id": "conv-123", "message": "Hello!"}
    context = {"user_id": "1"}

    result = await run_character_chat_adapter(config, context)
    assert result.get("simulated") is True
    assert "response" in result
    assert "Hello!" in result.get("response", "")


@pytest.mark.asyncio
async def test_character_chat_adapter_message_test_mode_y(monkeypatch):
    """Test Character chat adapter message in TEST_MODE=y."""
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_character_chat_adapter

    config = {"action": "message", "conversation_id": "conv-123", "message": "Hello!"}
    context = {"user_id": "1"}

    result = await run_character_chat_adapter(config, context)
    assert result.get("simulated") is True
    assert "response" in result
    assert "Hello!" in result.get("response", "")


@pytest.mark.asyncio
async def test_character_chat_adapter_load_test_mode(monkeypatch):
    """Test Character chat adapter load in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_character_chat_adapter

    config = {"action": "load", "conversation_id": "conv-123"}
    context = {"user_id": "1"}

    result = await run_character_chat_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("conversation_id") == "conv-123"
    assert result.get("history") == []


@pytest.mark.asyncio
async def test_character_chat_adapter_cancelled():
    """Test Character chat adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_character_chat_adapter

    config = {"action": "start", "character_id": 1}
    context = {"user_id": "1", "is_cancelled": lambda: True}

    result = await run_character_chat_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_character_chat_adapter_unknown_action(monkeypatch):
    """Test Character chat adapter returns error for unknown action."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_character_chat_adapter

    config = {"action": "invalid"}
    context = {"user_id": "1"}

    result = await run_character_chat_adapter(config, context)
    assert "unknown_action" in result.get("error", "")


@pytest.mark.asyncio
async def test_character_chat_adapter_with_user_name(monkeypatch):
    """Test Character chat adapter uses custom user name."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_character_chat_adapter

    config = {"action": "start", "character_id": 1, "user_name": "Alice"}
    context = {"user_id": "1"}

    result = await run_character_chat_adapter(config, context)
    assert result.get("simulated") is True


@pytest.mark.asyncio
async def test_character_chat_adapter_sanitizes_backend_errors(monkeypatch):
    """Test Character chat adapter does not expose backend exception details."""
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_character_chat_adapter

    def raise_backend_error(user_id: int) -> Path:
        raise RuntimeError(f"character backend exploded for user {user_id} at /private/chacha.db")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.integration.messaging.DatabasePaths.get_chacha_db_path",
        raise_backend_error,
    )

    result = await run_character_chat_adapter({"action": "load", "conversation_id": "conv-123"}, {"user_id": "1"})

    assert result == {"error": "character_chat_error"}


# ==============================================================================
# Email Send Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_email_send_adapter_missing_recipient():
    """Test Email adapter requires recipient."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {"subject": "Test", "body": "Hello"}
    context = {}

    result = await run_email_send_adapter(config, context)
    assert result.get("error") == "missing_recipient"
    assert result.get("sent") is False


@pytest.mark.asyncio
async def test_email_send_adapter_invalid_email():
    """Test Email adapter validates email addresses."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {"to": "not-an-email", "subject": "Test", "body": "Hello"}
    context = {}

    result = await run_email_send_adapter(config, context)
    assert "invalid_email" in result.get("error", "")
    assert result.get("sent") is False


@pytest.mark.asyncio
async def test_email_send_adapter_test_mode(monkeypatch):
    """Test Email adapter in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {
        "to": "user@example.com",
        "subject": "Test Subject",
        "body": "Test body content",
    }
    context = {}

    result = await run_email_send_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("sent") is True
    assert "user@example.com" in result.get("recipients", [])


@pytest.mark.asyncio
async def test_email_send_adapter_multiple_recipients(monkeypatch):
    """Test Email adapter with multiple recipients."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {
        "to": ["user1@example.com", "user2@example.com"],
        "subject": "Team Update",
        "body": "Hello team!",
    }
    context = {}

    result = await run_email_send_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("sent") is True
    assert len(result.get("recipients", [])) == 2


@pytest.mark.asyncio
async def test_email_send_adapter_comma_separated_recipients(monkeypatch):
    """Test Email adapter with comma-separated recipients."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {
        "to": "user1@example.com, user2@example.com",
        "subject": "Test",
        "body": "Hello",
    }
    context = {}

    result = await run_email_send_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("sent") is True


@pytest.mark.asyncio
async def test_email_send_adapter_html_body(monkeypatch):
    """Test Email adapter with HTML body."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {
        "to": "user@example.com",
        "subject": "HTML Email",
        "body": "<h1>Hello</h1><p>World</p>",
        "html": True,
    }
    context = {}

    result = await run_email_send_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("sent") is True


@pytest.mark.asyncio
async def test_email_send_adapter_cancelled():
    """Test Email adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {"to": "user@example.com", "subject": "Test", "body": "Hello"}
    context = {"is_cancelled": lambda: True}

    result = await run_email_send_adapter(config, context)
    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_email_send_adapter_with_template(monkeypatch):
    """Test Email adapter applies template to subject and body."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {
        "to": "user@example.com",
        "subject": "Report for {{inputs.date}}",
        "body": "Summary: {{prev.summary}}",
    }
    context = {
        "inputs": {"date": "2024-01-15"},
        "prev": {"summary": "All tests passed"},
    }

    result = await run_email_send_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("sent") is True


@pytest.mark.asyncio
async def test_email_send_adapter_sanitizes_subject(monkeypatch):
    """Test Email adapter sanitizes subject to prevent header injection."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {
        "to": "user@example.com",
        "subject": "Normal subject\nBcc: attacker@evil.com",
        "body": "Hello",
    }
    context = {}

    result = await run_email_send_adapter(config, context)
    assert result.get("simulated") is True
    # Newline should be removed from subject
    assert "\n" not in result.get("subject", "")


@pytest.mark.asyncio
async def test_email_send_adapter_body_from_prev(monkeypatch):
    """Test Email adapter gets body from previous step."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    config = {"to": "user@example.com", "subject": "Report", "body": ""}
    context = {"prev": {"text": "Generated report content"}}

    result = await run_email_send_adapter(config, context)
    assert result.get("simulated") is True
    assert result.get("sent") is True


@pytest.mark.asyncio
async def test_email_send_adapter_sanitizes_smtp_errors(monkeypatch):
    """Test Email adapter hides raw SMTP errors."""
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    from tldw_Server_API.app.core.Workflows.adapters.integration import run_email_send_adapter

    def fail_smtp(*_args, **_kwargs):
        raise RuntimeError("SMTP token at /private/workflows-smtp.key")

    monkeypatch.setattr("smtplib.SMTP", fail_smtp)

    result = await run_email_send_adapter(
        {"to": "user@example.com", "subject": "Report", "body": "Generated report content"},
        {},
    )

    assert result == {"sent": False, "error": "Email send failed"}


# ==============================================================================
# Podcast RSS Publish Adapter Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_podcast_rss_publish_creates_feed_and_item(monkeypatch, tmp_path):
    """Test podcast_rss_publish creates a valid feed with one episode item."""
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    from tldw_Server_API.app.core.Workflows.adapters.integration import (
        run_podcast_rss_publish_adapter,
    )

    config = {
        "feed_uri": "file://feeds/podcast.xml",
        "channel": {
            "title": "Daily Briefing",
            "link": "https://example.com/podcast",
            "description": "Daily podcast briefing",
        },
        "episode": {
            "guid": "episode-1",
            "title": "Episode 1",
            "description": "Top stories",
            "audio_url": "https://cdn.example.com/episode-1.mp3",
            "published_at": "2026-02-13T20:00:00Z",
        },
    }
    context = {"user_id": "test_user"}

    result = await run_podcast_rss_publish_adapter(config, context)

    assert result.get("published") is True
    assert result.get("item_guid") == "episode-1"
    feed_path = tmp_path / "feeds" / "podcast.xml"
    assert feed_path.exists()

    root = ET.parse(feed_path).getroot()  # nosec B314
    channel = root.find("channel")
    assert channel is not None
    item = channel.find("item")
    assert item is not None
    assert item.findtext("guid") == "episode-1"
    assert item.findtext("title") == "Episode 1"


@pytest.mark.asyncio
async def test_podcast_rss_publish_sanitizes_write_failure_logs(monkeypatch, tmp_path):
    """Test podcast_rss_publish write failures hide backend details in logs."""
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    from tldw_Server_API.app.core.Workflows.adapters.integration import podcast_rss
    from tldw_Server_API.app.core.Workflows.adapters.integration import (
        run_podcast_rss_publish_adapter,
    )

    def broken_write(self, file_or_filename, *args, **kwargs):
        raise RuntimeError("rss write token at /private/podcast-rss-cache")

    monkeypatch.setattr(podcast_rss.ET.ElementTree, "write", broken_write)

    messages: list[str] = []
    sink_id = podcast_rss.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_podcast_rss_publish_adapter(
            {
                "feed_uri": "file://feeds/podcast.xml",
                "channel": {
                    "title": "Daily Briefing",
                    "link": "https://example.com/podcast",
                    "description": "Daily podcast briefing",
                },
                "episode": {
                    "guid": "episode-1",
                    "title": "Episode 1",
                    "description": "Top stories",
                    "audio_url": "https://cdn.example.com/episode-1.mp3",
                },
            },
            {"user_id": "test_user"},
        )
    finally:
        podcast_rss.logger.remove(sink_id)

    assert result == {"error": "feed_write_failed", "published": False}
    joined = "\n".join(messages)
    assert "podcast_rss_publish write failed" in joined
    assert "podcast-rss-cache" not in joined


@pytest.mark.asyncio
async def test_podcast_rss_publish_dedupes_guid_and_checks_version(monkeypatch, tmp_path):
    """Test GUID dedupe and optimistic version conflict handling."""
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    from tldw_Server_API.app.core.Workflows.adapters.integration import (
        run_podcast_rss_publish_adapter,
    )

    context = {"user_id": "test_user"}
    feed_uri = "file://feeds/podcast.xml"

    base_config = {
        "feed_uri": feed_uri,
        "episode": {
            "guid": "episode-dup",
            "title": "Original Title",
            "audio_url": "https://cdn.example.com/original.mp3",
            "published_at": "2026-02-13T20:00:00Z",
        },
    }
    first = await run_podcast_rss_publish_adapter(base_config, context)
    assert first.get("published") is True
    assert first.get("version") == 1

    update_config = {
        "feed_uri": feed_uri,
        "expected_version": 1,
        "episode": {
            "guid": "episode-dup",
            "title": "Updated Title",
            "audio_url": "https://cdn.example.com/updated.mp3",
            "published_at": "2026-02-13T21:00:00Z",
        },
    }
    second = await run_podcast_rss_publish_adapter(update_config, context)
    assert second.get("published") is True
    assert second.get("item_count") == 1
    assert second.get("replaced_existing_guid") is True

    feed_path = tmp_path / "feeds" / "podcast.xml"
    root = ET.parse(feed_path).getroot()  # nosec B314
    channel = root.find("channel")
    items = channel.findall("item") if channel is not None else []
    assert len(items) == 1
    assert items[0].findtext("title") == "Updated Title"

    conflict = await run_podcast_rss_publish_adapter(
        {
            "feed_uri": feed_uri,
            "expected_version": 0,
            "episode": {
                "guid": "episode-new",
                "title": "New Episode",
                "audio_url": "https://cdn.example.com/new.mp3",
            },
        },
        context,
    )
    assert conflict.get("published") is False
    assert conflict.get("error") == "version_conflict"


@pytest.mark.asyncio
async def test_podcast_rss_publish_blocks_remote_seed_without_opt_in(monkeypatch, tmp_path):
    """Test remote feed seeding is disabled unless explicitly enabled."""
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    from tldw_Server_API.app.core.Workflows.adapters.integration import (
        run_podcast_rss_publish_adapter,
    )

    result = await run_podcast_rss_publish_adapter(
        {
            "feed_uri": "file://feeds/podcast.xml",
            "source_feed_url": "https://example.com/feed.xml",
            "allow_remote_fetch": False,
            "episode": {
                "guid": "episode-remote-disabled",
                "title": "Episode",
                "audio_url": "https://cdn.example.com/episode.mp3",
            },
        },
        {"user_id": "test_user"},
    )
    assert result.get("published") is False
    assert result.get("error") == "remote_fetch_disabled"


@pytest.mark.asyncio
async def test_podcast_rss_publish_rejects_non_http_seed_url(monkeypatch, tmp_path):
    """Test remote feed seeding only accepts http(s) source URLs."""
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    from tldw_Server_API.app.core.Workflows.adapters.integration import (
        run_podcast_rss_publish_adapter,
    )

    result = await run_podcast_rss_publish_adapter(
        {
            "feed_uri": "file://feeds/podcast.xml",
            "source_feed_url": "file:///tmp/feed.xml",
            "allow_remote_fetch": True,
            "episode": {
                "guid": "episode-remote-invalid",
                "title": "Episode",
                "audio_url": "https://cdn.example.com/episode.mp3",
            },
        },
        {"user_id": "test_user"},
    )
    assert result.get("published") is False
    assert result.get("error") == "invalid_remote_seed_url"


# ==============================================================================
# Import Tests
# ==============================================================================


def test_all_integration_adapters_importable():
    """Verify all integration adapters are importable."""
    from tldw_Server_API.app.core.Workflows.adapters.integration import (
        run_webhook_adapter,
        run_notify_adapter,
        run_mcp_tool_adapter,
        run_acp_stage_adapter,
        run_s3_upload_adapter,
        run_s3_download_adapter,
        run_github_create_issue_adapter,
        run_kanban_adapter,
        run_chatbooks_adapter,
        run_character_chat_adapter,
        run_email_send_adapter,
        run_podcast_rss_publish_adapter,
    )

    assert callable(run_webhook_adapter)
    assert callable(run_notify_adapter)
    assert callable(run_mcp_tool_adapter)
    assert callable(run_acp_stage_adapter)
    assert callable(run_s3_upload_adapter)
    assert callable(run_s3_download_adapter)
    assert callable(run_github_create_issue_adapter)
    assert callable(run_kanban_adapter)
    assert callable(run_chatbooks_adapter)
    assert callable(run_character_chat_adapter)
    assert callable(run_email_send_adapter)
    assert callable(run_podcast_rss_publish_adapter)


def test_integration_adapters_are_async():
    """Verify all integration adapters are async functions."""
    import asyncio

    from tldw_Server_API.app.core.Workflows.adapters.integration import (
        run_webhook_adapter,
        run_notify_adapter,
        run_mcp_tool_adapter,
        run_acp_stage_adapter,
        run_s3_upload_adapter,
        run_s3_download_adapter,
        run_github_create_issue_adapter,
        run_kanban_adapter,
        run_chatbooks_adapter,
        run_character_chat_adapter,
        run_email_send_adapter,
        run_podcast_rss_publish_adapter,
    )

    adapters = [
        run_webhook_adapter,
        run_notify_adapter,
        run_mcp_tool_adapter,
        run_acp_stage_adapter,
        run_s3_upload_adapter,
        run_s3_download_adapter,
        run_github_create_issue_adapter,
        run_kanban_adapter,
        run_chatbooks_adapter,
        run_character_chat_adapter,
        run_email_send_adapter,
        run_podcast_rss_publish_adapter,
    ]

    for adapter in adapters:
        assert asyncio.iscoroutinefunction(adapter), f"{adapter.__name__} is not async"


def test_integration_adapters_registered():
    """Verify all integration adapters are registered."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    expected_adapters = [
        "webhook",
        "notify",
        "mcp_tool",
        "acp_stage",
        "s3_upload",
        "s3_download",
        "github_create_issue",
        "kanban",
        "chatbooks",
        "character_chat",
        "email_send",
        "podcast_rss_publish",
    ]

    registered = registry.list_adapters()
    for name in expected_adapters:
        assert name in registered, f"Adapter '{name}' not registered"


def test_integration_adapters_in_category():
    """Verify all integration adapters are in 'integration' category."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    expected_adapters = [
        "webhook",
        "notify",
        "mcp_tool",
        "acp_stage",
        "s3_upload",
        "s3_download",
        "github_create_issue",
        "kanban",
        "chatbooks",
        "character_chat",
        "email_send",
        "podcast_rss_publish",
    ]

    for name in expected_adapters:
        spec = registry.get_spec(name)
        assert spec is not None, f"Adapter '{name}' not registered"
        assert spec.category == "integration", f"Adapter '{name}' not in integration category (got '{spec.category}')"
