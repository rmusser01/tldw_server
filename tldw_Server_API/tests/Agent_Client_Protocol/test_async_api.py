"""Tests for the ACP async fire-and-forget API.

Verifies:
- POST /acp/sessions/prompt-async returns task_id and poll_url
- GET /acp/tasks/{task_id} returns status and result
- Missing prompt returns 422
- Unknown task_id returns 404

Note: Tests that require the full FastAPI test client (``client_user_only``)
are skipped when the import chain fails on Python < 3.10 environments.
The async prompt endpoint now uses the global Scheduler for task persistence.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# Stub heavyweight deps that the app import chain may pull in.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *a: Any, **kw: Any) -> _StubProcessor:
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *a: Any, **kw: Any) -> _StubModel:
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


# ---------------------------------------------------------------------------
# Lazy import helper -- the endpoint module may fail to import on Py 3.9
# because of ``str | Path`` annotations in an upstream module.
# ---------------------------------------------------------------------------

_acp_ep = None


def _import_acp_ep():
    """Import the ACP endpoints module, returning None on failure."""
    global _acp_ep
    if _acp_ep is not None:
        return _acp_ep
    try:
        _acp_ep = importlib.import_module(
            "tldw_Server_API.app.api.v1.endpoints.agent_client_protocol"
        )
        return _acp_ep
    except (ImportError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_acp_run_result(
    session_id: str = "sess-async-1",
    result: dict | None = None,
    usage: dict | None = None,
    error: str | None = None,
    duration_ms: int = 42,
) -> dict:
    return {
        "session_id": session_id,
        "result": result or {"content": "done"},
        "usage": usage or {"prompt_tokens": 5, "completion_tokens": 10},
        "duration_ms": duration_ms,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Mark for skipping when the full app can't import (Py 3.9 compat issue)
# ---------------------------------------------------------------------------

_skip_if_no_app = pytest.mark.skipif(
    _import_acp_ep() is None,
    reason="Full app import chain fails on this Python version",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _patch_runner_and_store(monkeypatch, tmp_path):
    """Patch runner client and session store for client-based tests."""
    acp_ep = _import_acp_ep()
    if acp_ep is None:
        pytest.skip("Cannot import ACP endpoints module")

    stub_runner = AsyncMock()
    stub_runner.create_session = AsyncMock(return_value="sess-async-1")
    stub_runner.prompt = AsyncMock(
        return_value={
            "content": "done",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
    )
    stub_runner.close_session = AsyncMock()

    async def _get_runner():
        return stub_runner

    monkeypatch.setattr(acp_ep, "get_runner_client", _get_runner)

    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
    from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore

    _test_db = ACPSessionsDB(db_path=str(tmp_path / "test_acp_async.db"))
    _test_store = ACPSessionStore(db=_test_db)

    async def _get_store():
        return _test_store

    monkeypatch.setattr(acp_ep, "get_acp_session_store", _get_store)
    return stub_runner


@pytest.fixture()
def _patch_scheduler(monkeypatch):
    """Patch the global scheduler for async prompt tests."""
    from tldw_Server_API.app.core.Scheduler.base.task import Task, TaskStatus

    mock_scheduler = AsyncMock()
    mock_scheduler.submit = AsyncMock(return_value="test-task-001")
    mock_scheduler.get_task = AsyncMock(return_value=None)

    async def _get_scheduler(*args, **kwargs):
        return mock_scheduler

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Scheduler.get_global_scheduler",
        _get_scheduler,
    )
    return mock_scheduler


# ===========================================================================
# Client-based endpoint tests (require full app import)
# ===========================================================================


@_skip_if_no_app
def test_prompt_async_returns_task_id(client_user_only, _patch_runner_and_store, _patch_scheduler):
    """POST /acp/sessions/prompt-async returns task_id and poll_url."""
    resp = client_user_only.post(
        "/api/v1/acp/sessions/prompt-async",
        json={"prompt": "Hello async world"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "task_id" in data
    assert data["task_id"]  # non-empty
    assert data["poll_url"] == f"/api/v1/acp/tasks/{data['task_id']}"
    assert data["status"] == "queued"


@_skip_if_no_app
def test_task_status_not_found(client_user_only, _patch_runner_and_store, _patch_scheduler):
    """GET /acp/tasks/{nonexistent} returns 404."""
    resp = client_user_only.get("/api/v1/acp/tasks/nonexistent-task-id")
    assert resp.status_code == 404


@_skip_if_no_app
def test_prompt_async_validates_payload(client_user_only, _patch_runner_and_store, _patch_scheduler):
    """Missing prompt returns 422."""
    resp = client_user_only.post(
        "/api/v1/acp/sessions/prompt-async",
        json={},
    )
    assert resp.status_code == 422


@_skip_if_no_app
def test_prompt_async_accepts_message_list(client_user_only, _patch_runner_and_store, _patch_scheduler):
    """POST /acp/sessions/prompt-async accepts a list[dict] prompt."""
    resp = client_user_only.post(
        "/api/v1/acp/sessions/prompt-async",
        json={
            "prompt": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["task_id"]
    assert data["status"] == "queued"


@_skip_if_no_app
def test_task_status_returns_result(client_user_only, _patch_runner_and_store, _patch_scheduler):
    """GET /acp/tasks/{task_id} returns status and result after completion."""
    from tldw_Server_API.app.core.Scheduler.base.task import Task, TaskStatus

    completed_task = Task(
        id="test-task-001",
        handler="acp_run",
        status=TaskStatus.COMPLETED,
        result={"result": {"content": "Hello"}, "usage": {"prompt_tokens": 5, "completion_tokens": 10}, "duration_ms": 100},
        metadata={"user_id": "1"},
    )
    _patch_scheduler.get_task = AsyncMock(return_value=completed_task)

    resp = client_user_only.get("/api/v1/acp/tasks/test-task-001")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["task_id"] == "test-task-001"
    assert data["status"] == "completed"
    assert data["result"] == {"content": "Hello"}
    assert data["usage"]["prompt_tokens"] == 5
    assert data["error"] is None
    assert data["duration_ms"] == 100


@_skip_if_no_app
def test_task_status_failed_task(client_user_only, _patch_runner_and_store, _patch_scheduler):
    """GET /acp/tasks/{task_id} returns error for failed tasks."""
    from tldw_Server_API.app.core.Scheduler.base.task import Task, TaskStatus

    failed_task = Task(
        id="test-fail-001",
        handler="acp_run",
        status=TaskStatus.FAILED,
        error="connection refused",
        result={"duration_ms": 5},
        metadata={"user_id": "1"},
    )
    _patch_scheduler.get_task = AsyncMock(return_value=failed_task)

    resp = client_user_only.get("/api/v1/acp/tasks/test-fail-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert data["error"] == "connection refused"
    assert data["result"] is None


@_skip_if_no_app
def test_task_status_running_task(client_user_only, _patch_runner_and_store, _patch_scheduler):
    """GET /acp/tasks/{task_id} returns running status for in-progress tasks."""
    from tldw_Server_API.app.core.Scheduler.base.task import Task, TaskStatus

    running_task = Task(
        id="test-running-001",
        handler="acp_run",
        status=TaskStatus.RUNNING,
        metadata={"user_id": "1"},
    )
    _patch_scheduler.get_task = AsyncMock(return_value=running_task)

    resp = client_user_only.get("/api/v1/acp/tasks/test-running-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["result"] is None
    assert data["error"] is None


# ===========================================================================
# Schema validation tests (lightweight, no app import needed)
# ===========================================================================


def test_async_prompt_request_requires_prompt():
    """ACPAsyncPromptRequest requires prompt field."""
    from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
        ACPAsyncPromptRequest,
    )

    with pytest.raises(Exception):
        ACPAsyncPromptRequest()

    # Valid with just prompt
    req = ACPAsyncPromptRequest(prompt="Hello")
    assert req.prompt == "Hello"
    assert req.cwd == "."
    assert req.agent_type is None


def test_async_prompt_request_accepts_list():
    """ACPAsyncPromptRequest accepts list prompt."""
    from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
        ACPAsyncPromptRequest,
    )

    messages = [{"role": "user", "content": "hi"}]
    req = ACPAsyncPromptRequest(prompt=messages)
    assert req.prompt == messages


def test_async_prompt_response_schema():
    """ACPAsyncPromptResponse has expected fields."""
    from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
        ACPAsyncPromptResponse,
    )

    resp = ACPAsyncPromptResponse(
        task_id="abc-123",
        poll_url="/api/v1/acp/tasks/abc-123",
    )
    assert resp.task_id == "abc-123"
    assert resp.status == "queued"


def test_task_status_response_schema():
    """ACPTaskStatusResponse has expected fields and defaults."""
    from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
        ACPTaskStatusResponse,
    )

    resp = ACPTaskStatusResponse(task_id="t-1", status="completed")
    assert resp.result is None
    assert resp.usage == {}
    assert resp.error is None
    assert resp.duration_ms is None

    resp_full = ACPTaskStatusResponse(
        task_id="t-2",
        status="completed",
        result={"content": "hi"},
        usage={"prompt_tokens": 5},
        duration_ms=100,
    )
    assert resp_full.result == {"content": "hi"}
    assert resp_full.duration_ms == 100
