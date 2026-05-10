"""Tests for Agent Orchestration API endpoints (Phase 4.2)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Agent_Orchestration.orchestration_service import (
    OrchestrationService,
)
from tldw_Server_API.app.core.Agent_Orchestration.models import TaskStatus
from tldw_Server_API.app.core.DB_Management.Orchestration_DB import OrchestrationDB

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
def svc():
    return OrchestrationService()


# ---- Project API scenarios ----


async def test_create_and_list_projects(svc):
    """Create multiple projects and verify listing."""
    await svc.create_project(name="Alpha", user_id=1)
    await svc.create_project(name="Beta", user_id=1)
    await svc.create_project(name="Gamma", user_id=2)

    user1_projects = await svc.list_projects(user_id=1)
    assert len(user1_projects) == 2
    names = {p.name for p in user1_projects}
    assert names == {"Alpha", "Beta"}


async def test_get_project_not_found(svc):
    """Getting a nonexistent project returns None."""
    assert await svc.get_project(999) is None


async def test_delete_nonexistent_project(svc):
    """Deleting a nonexistent project returns False."""
    assert await svc.delete_project(999) is False


# ---- Task API scenarios ----


async def test_create_task_with_all_fields(svc):
    """Create a task with all optional fields."""
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(
        project.id,
        title="Full Task",
        description="A comprehensive task",
        agent_type="codex",
        max_review_attempts=5,
        user_id=1,
    )
    assert task.description == "A comprehensive task"
    assert task.agent_type == "codex"
    assert task.max_review_attempts == 5


async def test_list_tasks_all_statuses(svc):
    """List tasks returns all statuses when no filter."""
    project = await svc.create_project(name="P1", user_id=1)
    await svc.create_task(project.id, title="T1", user_id=1)
    t2 = await svc.create_task(project.id, title="T2", user_id=1)
    await svc.transition_task(t2.id, TaskStatus.IN_PROGRESS)
    t3 = await svc.create_task(project.id, title="T3", user_id=1)
    await svc.transition_task(t3.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(t3.id, TaskStatus.REVIEW)

    all_tasks = await svc.list_tasks(project.id)
    assert len(all_tasks) == 3


# ---- Run dispatch scenarios ----


async def test_run_inherits_agent_type(svc):
    """Run should inherit agent_type from its parent task."""
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(
        project.id, title="T1", agent_type="codex", user_id=1
    )
    run = await svc.create_run(task.id, session_id="sess-1")
    assert run.agent_type == "codex"


async def test_multiple_runs_per_task(svc):
    """Multiple runs can be created for the same task."""
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", user_id=1)

    r1 = await svc.create_run(task.id, session_id="s1")
    r2 = await svc.create_run(task.id, session_id="s2")
    r3 = await svc.create_run(task.id, session_id="s3")

    runs = await svc.list_runs(task.id)
    assert len(runs) == 3
    assert {r.session_id for r in runs} == {"s1", "s2", "s3"}


async def test_get_task_run_history_includes_acp_session_drillthrough(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as acp_mod
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    project = db.create_project(name="P1")
    task = db.create_task(project.id, title="T1", description="Dispatch me", agent_type="codex")
    run = db.create_run(task.id, agent_type="codex", session_id="session-success")
    db.complete_run(run.id, result_summary="Implemented dispatch work", token_usage={"input_tokens": 10})

    _clear_acp_audit_events()
    acp_mod._acp_record_audit_event(
        action="orchestration_task_completed",
        user_id=1,
        session_id="session-success",
        metadata={"task_id": task.id},
    )
    session = SimpleNamespace(
        session_id="session-success",
        user_id=1,
        agent_type="codex",
        name="orchestration-task-1",
        status="closed",
        created_at="2026-05-10T01:00:00+00:00",
        last_activity_at="2026-05-10T01:01:00+00:00",
        message_count=2,
        usage=SimpleNamespace(to_dict=lambda: {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        messages=[
            {
                "role": "user",
                "content": "Task prompt text",
                "timestamp": "2026-05-10T01:00:00+00:00",
                "raw_prompt": {"role": "user", "content": "Task prompt text"},
            },
            {
                "role": "assistant",
                "content": "Done",
                "timestamp": "2026-05-10T01:01:00+00:00",
                "raw_result": {
                    "content": "Done",
                    "stopReason": "end",
                    "tool_calls": [{"name": "write_file"}],
                    "artifacts": [{"id": "artifact-1", "type": "summary"}],
                },
            },
        ],
    )

    async def fake_store():
        return _RunHistorySessionStore({"session-success": session})

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )

    try:
        detail = await orch_mod.get_task(task.id, user=_TestUser())
        enriched_run = detail.runs[0]

        assert enriched_run["session"]["available"] is True
        assert enriched_run["session"]["links"]["detail"] == "/api/v1/acp/sessions/session-success/detail"
        assert enriched_run["session"]["links"]["events"] == "/api/v1/acp/sessions/session-success/events"
        assert enriched_run["session"]["links"]["artifacts"] == "/api/v1/acp/sessions/session-success/artifacts"
        assert enriched_run["session"]["links"]["diagnostics"] == "/api/v1/acp/sessions/session-success/diagnostics"
        assert enriched_run["history"]["event_count"] == 2
        assert enriched_run["history"]["audit_event_count"] == 1
        assert enriched_run["history"]["artifact_count"] == 1
        assert enriched_run["history"]["tool_call_count"] == 1
        assert enriched_run["history"]["stop_reason"] == "end"
        assert enriched_run["history"]["prompt"]["preview"] == "Task prompt text"
        assert enriched_run["history"]["result"]["preview"] == "Done"
        assert enriched_run["history"]["artifacts"][0]["id"] == "artifact-1"
        assert enriched_run["failure_context"] is None
    finally:
        db.close()


async def test_get_task_run_history_includes_failed_session_diagnostics(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    project = db.create_project(name="P1")
    task = db.create_task(project.id, title="T1", description="Dispatch me", agent_type="codex")
    run = db.create_run(task.id, agent_type="codex", session_id="session-failed")
    db.fail_run(run.id, error="ACP prompt failed")

    session = SimpleNamespace(
        session_id="session-failed",
        user_id=1,
        agent_type="codex",
        name="orchestration-task-1",
        status="error",
        created_at="2026-05-10T01:00:00+00:00",
        last_activity_at="2026-05-10T01:01:00+00:00",
        message_count=2,
        usage=SimpleNamespace(to_dict=lambda: {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4}),
        messages=[
            {
                "role": "user",
                "content": "Task prompt text",
                "timestamp": "2026-05-10T01:00:00+00:00",
                "raw_prompt": {"role": "user", "content": "Task prompt text"},
            },
            {
                "role": "assistant",
                "content": {
                    "status": "timeout",
                    "error": "Execution timed out",
                    "diagnostic_uri": "file:///tmp/acp-timeout.json",
                },
                "timestamp": "2026-05-10T01:01:00+00:00",
                "raw_result": {
                    "status": "timeout",
                    "error": "Execution timed out",
                    "diagnostic_uri": "file:///tmp/acp-timeout.json",
                },
            },
        ],
    )

    async def fake_store():
        return _RunHistorySessionStore({"session-failed": session})

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )

    try:
        detail = await orch_mod.get_task(task.id, user=_TestUser())
        enriched_run = detail.runs[0]

        assert enriched_run["session"]["available"] is True
        assert enriched_run["history"]["diagnostic_count"] == 1
        assert enriched_run["history"]["diagnostics"][0]["reason_code"] == "timed_out"
        assert enriched_run["failure_context"] == {
            "reason_code": "timed_out",
            "message": "Execution timed out",
            "diagnostic_uri": "file:///tmp/acp-timeout.json",
            "source": "session_diagnostic",
        }
        assert enriched_run["session"]["links"]["audit"] == "/api/v1/acp/sessions/session-failed/audit"
    finally:
        db.close()


class _TestUser:
    id = 1
    id_int = 1


class _NoopSessionStore:
    async def check_session_quota(self, _user_id):
        return None

    async def register_session(self, **_kwargs):
        return None


class _RunHistorySessionStore(_NoopSessionStore):
    def __init__(self, sessions):
        self.sessions = sessions

    async def get_session(self, session_id):
        return self.sessions.get(session_id)


class _RegisterFailingSessionStore(_NoopSessionStore):
    async def register_session(self, **_kwargs):
        raise RuntimeError("acp register backend leaked /private/acp-store.db")


class _CreateSessionFailingClient:
    async def create_session(self, *_args, **_kwargs):
        raise RuntimeError("acp create backend exploded")


class _SuccessfulClient:
    async def create_session(self, *_args, **_kwargs):
        return "session-private-1"

    async def prompt(self, *_args, **_kwargs):
        return {
            "stopReason": "end",
            "content": (
                "Done.\n"
                '<acp-task-completion>{"status":"completed",'
                '"summary":"Implemented dispatch work"}</acp-task-completion>'
            ),
            "usage": {},
        }


class _MissingCompletionClient:
    async def create_session(self, *_args, **_kwargs):
        return "session-1"

    async def prompt(self, *_args, **_kwargs):
        return {"stopReason": "end", "content": "Done without a structured marker", "usage": {}}


class _MalformedCompletionClient:
    async def create_session(self, *_args, **_kwargs):
        return "session-1"

    async def prompt(self, *_args, **_kwargs):
        return {
            "stopReason": "end",
            "content": '<acp-task-completion>{"status":</acp-task-completion>',
            "usage": {},
        }


class _RejectedCompletionClient:
    async def create_session(self, *_args, **_kwargs):
        return "session-1"

    async def prompt(self, *_args, **_kwargs):
        return {
            "stopReason": "end",
            "taskCompletion": {
                "status": "rejected",
                "summary": "Success criteria were not satisfied",
            },
            "usage": {},
        }


class _PromptFailingClient:
    async def create_session(self, *_args, **_kwargs):
        return "session-1"

    async def prompt(self, *_args, **_kwargs):
        raise RuntimeError("acp prompt backend exploded")


class _ReviewerDecisionClient:
    def __init__(self, review_payload):
        self.review_payload = review_payload
        self.create_session_calls = []
        self.prompt_calls = []

    async def create_session(self, *_args, **kwargs):
        self.create_session_calls.append(kwargs)
        return f"session-{len(self.create_session_calls)}"

    async def prompt(self, *_args, **_kwargs):
        self.prompt_calls.append(_args)
        if len(self.prompt_calls) == 1:
            return {
                "stopReason": "end",
                "taskCompletion": {
                    "status": "completed",
                    "summary": "Implementation ready for review",
                },
                "usage": {"input_tokens": 10},
            }
        return {
            "stopReason": "end",
            "reviewDecision": self.review_payload,
            "usage": {"input_tokens": 3},
        }


class _WorkspaceSessionCaptureClient:
    def __init__(self):
        self.create_session_calls = []

    async def create_session(self, *_args, **kwargs):
        self.create_session_calls.append(kwargs)
        return "session-workspace-env"

    async def prompt(self, *_args, **_kwargs):
        return {
            "stopReason": "end",
            "taskCompletion": {
                "status": "completed",
                "summary": "Workspace env and MCP server were injected",
            },
            "usage": {},
        }


def _clear_acp_audit_events() -> None:
    from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as acp_mod

    with acp_mod._ACP_AUDIT_LOCK:
        acp_mod._ACP_AUDIT_EVENTS.clear()


def _acp_audit_events_for_task(task_id: int) -> list[dict]:
    from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as acp_mod

    with acp_mod._ACP_AUDIT_LOCK:
        return [
            dict(event)
            for event in acp_mod._ACP_AUDIT_EVENTS
            if event.get("metadata", {}).get("task_id") == task_id
        ]


async def _build_dispatch_task(tmp_path):
    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    project = db.create_project(name="P1")
    task = db.create_task(project.id, title="T1", description="Dispatch me", agent_type="codex")
    return db, task


async def test_dispatch_run_sanitizes_create_session_failure(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db, task = await _build_dispatch_task(tmp_path)
    fake_logger = MagicMock()

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return _CreateSessionFailingClient()

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "logger", fake_logger)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            await orch_mod.dispatch_run(
                task.id,
                orch_mod.RunDispatchRequest(),
                user=_TestUser(),
            )

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail == "Failed to create ACP session"
        fake_logger.error.assert_called_once_with("Failed to create ACP session")
    finally:
        db.close()


async def test_dispatch_run_sanitizes_register_session_warning(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    project = db.create_project(name="P1")
    task = db.create_task(
        project.id,
        title="T1",
        description="Dispatch me",
        agent_type="codex",
    )
    fake_logger = MagicMock()

    async def fake_store():
        return _RegisterFailingSessionStore()

    async def fake_client():
        return _SuccessfulClient()

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "logger", fake_logger)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        result = await orch_mod.dispatch_run(
            task.id,
            orch_mod.RunDispatchRequest(),
            user=_TestUser(),
        )

        assert result["task_id"] == task.id
        assert result["status"] == TaskStatus.COMPLETE
        runs = db.list_runs(task.id)
        assert runs[0].status.value == "completed"
        assert runs[0].result_summary == "Implemented dispatch work"
        fake_logger.warning.assert_called_once_with("Failed to register orchestration ACP session")
    finally:
        db.close()


async def test_dispatch_run_reviewer_agent_approval_completes_and_records_review(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    project = db.create_project(name="P1")
    task = db.create_task(
        project.id,
        title="T1",
        description="Dispatch me",
        agent_type="codex",
        reviewer_agent_type="reviewer",
    )
    client = _ReviewerDecisionClient(
        {"approved": True, "feedback": "Meets the success criteria"}
    )
    _clear_acp_audit_events()

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return client

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        result = await orch_mod.dispatch_run(
            task.id,
            orch_mod.RunDispatchRequest(),
            user=_TestUser(),
        )

        assert result["status"] == TaskStatus.COMPLETE
        assert [call["agent_type"] for call in client.create_session_calls] == [
            "codex",
            "reviewer",
        ]
        reviews = db.list_reviews(task.id)
        assert len(reviews) == 1
        assert reviews[0]["approved"] is True
        assert reviews[0]["feedback"] == "Meets the success criteria"
        assert reviews[0]["reviewer"] == "reviewer"
        detail = await orch_mod.get_task(task.id, user=_TestUser())
        assert detail.reviews == reviews
        runs = db.list_runs(task.id)
        assert [run.agent_type for run in runs] == ["reviewer", "codex"]
        assert all(run.status.value == "completed" for run in runs)
        audit_actions = [event["action"] for event in _acp_audit_events_for_task(task.id)]
        assert audit_actions == [
            "orchestration_dispatch_started",
            "orchestration_task_completed",
            "orchestration_review_started",
            "orchestration_review_decision",
            "orchestration_task_finalized",
        ]
        serialized = json.dumps(_acp_audit_events_for_task(task.id))
        assert "Dispatch me" not in serialized
        assert "Meets the success criteria" not in serialized
    finally:
        db.close()


async def test_dispatch_run_reviewer_agent_rejection_retries_with_history(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    project = db.create_project(name="P1")
    task = db.create_task(
        project.id,
        title="T1",
        description="Dispatch me",
        agent_type="codex",
        reviewer_agent_type="reviewer",
        max_review_attempts=2,
    )
    client = _ReviewerDecisionClient(
        {"approved": False, "feedback": "Missing required tests"}
    )
    _clear_acp_audit_events()

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return client

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        result = await orch_mod.dispatch_run(
            task.id,
            orch_mod.RunDispatchRequest(),
            user=_TestUser(),
        )

        assert result["status"] == TaskStatus.IN_PROGRESS
        updated_task = db.get_task(task.id)
        assert updated_task.review_count == 1
        reviews = db.list_reviews(task.id)
        assert reviews[0]["approved"] is False
        assert reviews[0]["feedback"] == "Missing required tests"
        detail = await orch_mod.get_task(task.id, user=_TestUser())
        assert detail.reviews == reviews
        audit_actions = [event["action"] for event in _acp_audit_events_for_task(task.id)]
        assert "orchestration_task_requeued" in audit_actions
        serialized = json.dumps(_acp_audit_events_for_task(task.id))
        assert "Missing required tests" not in serialized
    finally:
        db.close()


async def test_dispatch_run_reviewer_agent_rejection_max_attempts_triages(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    project = db.create_project(name="P1")
    task = db.create_task(
        project.id,
        title="T1",
        description="Dispatch me",
        agent_type="codex",
        reviewer_agent_type="reviewer",
        max_review_attempts=1,
    )
    client = _ReviewerDecisionClient(
        {"approved": False, "feedback": "Still fails review"}
    )
    _clear_acp_audit_events()

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return client

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        result = await orch_mod.dispatch_run(
            task.id,
            orch_mod.RunDispatchRequest(),
            user=_TestUser(),
        )

        assert result["status"] == TaskStatus.TRIAGE
        detail = await orch_mod.get_task(task.id, user=_TestUser())
        assert detail.review_count == 1
        assert detail.reviews[0]["feedback"] == "Still fails review"
        assert detail.runs[0]["agent_type"] == "reviewer"
        assert detail.runs[0]["result_summary"] == "Still fails review"
        audit_actions = [event["action"] for event in _acp_audit_events_for_task(task.id)]
        assert "orchestration_task_triaged" in audit_actions
        serialized = json.dumps(_acp_audit_events_for_task(task.id))
        assert "Still fails review" not in serialized
    finally:
        db.close()


async def test_dispatch_run_valid_completion_without_reviewer_completes_task(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db, task = await _build_dispatch_task(tmp_path)

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return _SuccessfulClient()

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        result = await orch_mod.dispatch_run(
            task.id,
            orch_mod.RunDispatchRequest(),
            user=_TestUser(),
        )

        assert result["status"] == TaskStatus.COMPLETE
        updated_task = db.get_task(task.id)
        assert updated_task.status == TaskStatus.COMPLETE
        runs = db.list_runs(task.id)
        assert runs[0].status.value == "completed"
        assert runs[0].result_summary == "Implemented dispatch work"
    finally:
        db.close()


async def test_dispatch_run_injects_workspace_env_and_mcp_servers(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = db.create_workspace(
        name="Workspace",
        root_path=str(workspace_root),
        env_vars={"WORKSPACE_TOKEN": "abc"},
    )
    db.create_workspace_mcp_server(
        workspace.id,
        server_name="workspace-files",
        server_type="stdio",
        command="workspace-mcp",
        args=["--root", "."],
        env={"MCP_TOKEN": "xyz"},
    )
    project = db.create_project(name="P1", workspace_id=workspace.id)
    task = db.create_task(project.id, title="T1", description="Dispatch me", agent_type="codex")
    client = _WorkspaceSessionCaptureClient()

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return client

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )
    monkeypatch.setattr(
        orch_mod,
        "_resolve_dispatch_cwd",
        lambda raw_cwd, *, workspace_root=None: workspace_root or raw_cwd,
    )

    try:
        result = await orch_mod.dispatch_run(
            task.id,
            orch_mod.RunDispatchRequest(),
            user=_TestUser(),
        )

        assert result["status"] == TaskStatus.COMPLETE
        assert client.create_session_calls[0]["session_env"] == {"WORKSPACE_TOKEN": "abc"}
        assert client.create_session_calls[0]["mcp_servers"] == [
            {
                "name": "workspace-files",
                "type": "stdio",
                "command": "workspace-mcp",
                "args": ["--root", "."],
                "env": {"MCP_TOKEN": "xyz"},
            }
        ]
    finally:
        db.close()


async def test_dispatch_run_requires_completion_signal(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db, task = await _build_dispatch_task(tmp_path)

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return _MissingCompletionClient()

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            await orch_mod.dispatch_run(
                task.id,
                orch_mod.RunDispatchRequest(),
                user=_TestUser(),
            )

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail == "ACP completion signal invalid"
        updated_task = db.get_task(task.id)
        assert updated_task.status == TaskStatus.TRIAGE
        runs = db.list_runs(task.id)
        assert runs[0].status.value == "failed"
        assert "missing" in (runs[0].error or "")
    finally:
        db.close()


async def test_dispatch_run_rejects_malformed_completion_signal(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db, task = await _build_dispatch_task(tmp_path)

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return _MalformedCompletionClient()

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            await orch_mod.dispatch_run(
                task.id,
                orch_mod.RunDispatchRequest(),
                user=_TestUser(),
            )

        assert exc_info.value.status_code == 502
        updated_task = db.get_task(task.id)
        assert updated_task.status == TaskStatus.TRIAGE
        runs = db.list_runs(task.id)
        assert runs[0].status.value == "failed"
        assert "malformed" in (runs[0].error or "")
    finally:
        db.close()


async def test_dispatch_run_rejects_rejected_completion_signal(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db, task = await _build_dispatch_task(tmp_path)

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return _RejectedCompletionClient()

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            await orch_mod.dispatch_run(
                task.id,
                orch_mod.RunDispatchRequest(),
                user=_TestUser(),
            )

        assert exc_info.value.status_code == 502
        updated_task = db.get_task(task.id)
        assert updated_task.status == TaskStatus.TRIAGE
        runs = db.list_runs(task.id)
        assert runs[0].status.value == "failed"
        assert "rejected" in (runs[0].error or "")
    finally:
        db.close()


async def test_dispatch_run_sanitizes_prompt_failure(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db, task = await _build_dispatch_task(tmp_path)
    fake_logger = MagicMock()

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return _PromptFailingClient()

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "logger", fake_logger)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )

    try:
        with pytest.raises(HTTPException) as exc_info:
            await orch_mod.dispatch_run(
                task.id,
                orch_mod.RunDispatchRequest(),
                user=_TestUser(),
            )

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail == "ACP prompt failed"
        fake_logger.error.assert_called_once_with("ACP prompt failed")
    finally:
        db.close()


# ---- Review gate edge cases ----


async def test_review_approval_after_rejection(svc):
    """Approve should work after a previous rejection."""
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(
        project.id, title="T1", max_review_attempts=5, user_id=1
    )

    # First cycle: reject
    await svc.transition_task(task.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(task.id, TaskStatus.REVIEW)
    await svc.submit_review(task.id, approved=False)
    assert (await svc.get_task(task.id)).status == TaskStatus.IN_PROGRESS

    # Second cycle: approve
    await svc.transition_task(task.id, TaskStatus.REVIEW)
    result = await svc.submit_review(task.id, approved=True)
    assert result.status == TaskStatus.COMPLETE
    assert result.review_count == 2


# ---- Dependency chain ----


async def test_three_level_dependency_chain(svc):
    """Three-level dependency chain: T3 → T2 → T1."""
    project = await svc.create_project(name="P1", user_id=1)
    t1 = await svc.create_task(project.id, title="T1", user_id=1)
    t2 = await svc.create_task(
        project.id, title="T2", dependency_id=t1.id, user_id=1
    )
    t3 = await svc.create_task(
        project.id, title="T3", dependency_id=t2.id, user_id=1
    )

    # T3 not ready because T2 not complete
    assert await svc.check_dependency_ready(t3.id) is False

    # Complete T1
    await svc.transition_task(t1.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(t1.id, TaskStatus.REVIEW)
    await svc.transition_task(t1.id, TaskStatus.COMPLETE)

    # T2 is ready, T3 still not (T2 not complete)
    assert await svc.check_dependency_ready(t2.id) is True
    assert await svc.check_dependency_ready(t3.id) is False

    # Complete T2
    await svc.transition_task(t2.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(t2.id, TaskStatus.REVIEW)
    await svc.transition_task(t2.id, TaskStatus.COMPLETE)

    # Now T3 is ready
    assert await svc.check_dependency_ready(t3.id) is True


# ---- Summary ----


async def test_project_summary_includes_all_statuses(svc):
    """Project summary should include all task status categories."""
    project = await svc.create_project(name="P1", user_id=1)
    t1 = await svc.create_task(project.id, title="T1", user_id=1)
    t2 = await svc.create_task(project.id, title="T2", user_id=1)
    t3 = await svc.create_task(project.id, title="T3", user_id=1)

    await svc.transition_task(t1.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(t2.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(t2.id, TaskStatus.REVIEW)
    await svc.transition_task(t3.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(t3.id, TaskStatus.TRIAGE)

    summary = await svc.get_project_summary(project.id)
    assert summary["total_tasks"] == 3
    counts = summary["status_counts"]
    assert counts.get("inprogress", 0) == 1
    assert counts.get("review", 0) == 1
    assert counts.get("triage", 0) == 1


async def test_get_task_not_found(svc):
    """Getting a nonexistent task returns None."""
    assert await svc.get_task(999) is None


async def test_transition_nonexistent_task_raises(svc):
    """Transitioning a nonexistent task raises ValueError."""
    with pytest.raises(ValueError, match="Task 999 not found"):
        await svc.transition_task(999, TaskStatus.IN_PROGRESS)
