"""Tests for Agent Orchestration API endpoints (Phase 4.2)."""
from __future__ import annotations

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


class _TestUser:
    id = 1
    id_int = 1


class _NoopSessionStore:
    async def check_session_quota(self, _user_id):
        return None

    async def register_session(self, **_kwargs):
        return None


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
        return {"stopReason": "complete", "usage": {}}


class _PromptFailingClient:
    async def create_session(self, *_args, **_kwargs):
        return "session-1"

    async def prompt(self, *_args, **_kwargs):
        raise RuntimeError("acp prompt backend exploded")


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
        reviewer_agent_type="reviewer",
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
        assert result["status"] == TaskStatus.REVIEW
        fake_logger.warning.assert_called_once_with("Failed to register orchestration ACP session")
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
