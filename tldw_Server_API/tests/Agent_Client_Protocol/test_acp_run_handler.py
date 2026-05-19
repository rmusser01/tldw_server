"""Tests for the acp_run scheduler handler.

Verifies that importing the handler module registers the task,
and that the handler creates a session, sends a prompt, closes the session,
and returns the expected result structure.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Patch target: the *source* module where get_runner_client is defined,
# because the handler imports it lazily inside the function body.
_RUNNER_CLIENT_PATCH = (
    "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client"
)


@pytest.mark.asyncio
async def test_acp_run_handler_creates_session_sends_prompt_closes():
    """acp_run creates a session, sends prompt, closes, returns result."""
    mock_runner = AsyncMock()
    mock_runner.create_session = AsyncMock(return_value="sess-abc-123")
    mock_runner.prompt = AsyncMock(return_value={
        "content": "Hello world",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    })
    mock_runner.close_session = AsyncMock()

    with patch(_RUNNER_CLIENT_PATCH, new_callable=AsyncMock, return_value=mock_runner):
        from tldw_Server_API.app.core.Scheduler.handlers.acp import acp_run

        result = await acp_run({
            "user_id": 42,
            "prompt": "Summarize this document",
            "cwd": "/workspace",
            "agent_type": "research",
        })

    assert result["session_id"] == "sess-abc-123"
    assert result["error"] is None
    assert result["duration_ms"] >= 0
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}
    assert result["result"]["content"] == "Hello world"

    mock_runner.create_session.assert_awaited_once()
    call_kwargs = mock_runner.create_session.call_args
    assert call_kwargs.kwargs.get("user_id") == 42
    mock_runner.prompt.assert_awaited_once_with(
        "sess-abc-123",
        [{"role": "user", "content": "Summarize this document"}],
    )
    # Session is closed in the finally block
    mock_runner.close_session.assert_awaited_once_with("sess-abc-123")


@pytest.mark.asyncio
async def test_acp_run_handler_accepts_list_prompt():
    """acp_run passes through a list prompt without re-wrapping."""
    mock_runner = AsyncMock()
    mock_runner.create_session = AsyncMock(return_value="sess-list")
    mock_runner.prompt = AsyncMock(return_value={"content": "ok"})
    mock_runner.close_session = AsyncMock()

    prompt_list = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    with patch(_RUNNER_CLIENT_PATCH, new_callable=AsyncMock, return_value=mock_runner):
        from tldw_Server_API.app.core.Scheduler.handlers.acp import acp_run

        result = await acp_run({
            "user_id": 1,
            "prompt": prompt_list,
        })

    assert result["error"] is None
    mock_runner.prompt.assert_awaited_once_with("sess-list", prompt_list)


@pytest.mark.asyncio
async def test_acp_run_handler_returns_error_on_missing_user_id():
    """acp_run returns error dict when user_id is missing."""
    from tldw_Server_API.app.core.Scheduler.handlers.acp import acp_run

    result = await acp_run({"prompt": "hello"})

    assert result["error"] == "Missing user_id"
    assert result["session_id"] is None
    assert result["result"] is None


@pytest.mark.asyncio
async def test_acp_run_handler_returns_error_on_failure():
    """acp_run returns error dict on exception."""
    mock_runner = AsyncMock()
    mock_runner.create_session = AsyncMock(side_effect=RuntimeError("connection refused"))
    mock_runner.close_session = AsyncMock()

    with patch(_RUNNER_CLIENT_PATCH, new_callable=AsyncMock, return_value=mock_runner):
        from tldw_Server_API.app.core.Scheduler.handlers.acp import acp_run

        result = await acp_run({
            "user_id": 1,
            "prompt": "test",
        })

    assert result["error"] is not None
    assert "connection refused" in result["error"]
    assert result["session_id"] is None
    assert result["result"] is None
    assert result["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_acp_run_handler_closes_session_on_prompt_failure():
    """acp_run still attempts to close session when prompt fails."""
    mock_runner = AsyncMock()
    mock_runner.create_session = AsyncMock(return_value="sess-fail")
    mock_runner.prompt = AsyncMock(side_effect=ValueError("bad prompt"))
    mock_runner.close_session = AsyncMock()

    with patch(_RUNNER_CLIENT_PATCH, new_callable=AsyncMock, return_value=mock_runner):
        from tldw_Server_API.app.core.Scheduler.handlers.acp import acp_run

        result = await acp_run({
            "user_id": 1,
            "prompt": "test",
        })

    assert result["error"] is not None
    assert result["session_id"] == "sess-fail"
    # Session should have been closed in finally block
    mock_runner.close_session.assert_awaited_once_with("sess-fail")


def test_acp_run_is_registered():
    """Importing the module registers the task in the global registry."""
    from tldw_Server_API.app.core.Scheduler.handlers import acp  # noqa: F401
    from tldw_Server_API.app.core.Scheduler.base.registry import _global_registry

    assert "acp_run" in _global_registry._handlers
    metadata = _global_registry._metadata["acp_run"]
    assert metadata["queue"] == "acp"
    assert metadata["timeout"] == 7200
    assert metadata["max_retries"] == 1
