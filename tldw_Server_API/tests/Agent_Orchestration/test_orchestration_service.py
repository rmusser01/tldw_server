"""Tests for Agent Orchestration service (Phase 4)."""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Agent_Orchestration import db_factory
from tldw_Server_API.app.core.Agent_Orchestration.models import (
    TaskStatus,
    RunStatus,
    is_valid_transition,
)
from tldw_Server_API.app.core.Agent_Orchestration.orchestration_service import (
    CycleDependencyError,
    OrchestrationService,
    get_orchestration_db,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def svc():
    return OrchestrationService()


# ---- State machine tests ----

def test_valid_transitions():
    assert is_valid_transition(TaskStatus.TODO, TaskStatus.IN_PROGRESS)
    assert is_valid_transition(TaskStatus.IN_PROGRESS, TaskStatus.REVIEW)
    assert is_valid_transition(TaskStatus.IN_PROGRESS, TaskStatus.TRIAGE)
    assert is_valid_transition(TaskStatus.REVIEW, TaskStatus.COMPLETE)
    assert is_valid_transition(TaskStatus.REVIEW, TaskStatus.TRIAGE)
    assert is_valid_transition(TaskStatus.REVIEW, TaskStatus.IN_PROGRESS)
    assert is_valid_transition(TaskStatus.TRIAGE, TaskStatus.TODO)
    assert is_valid_transition(TaskStatus.TRIAGE, TaskStatus.IN_PROGRESS)


def test_invalid_transitions():
    assert not is_valid_transition(TaskStatus.TODO, TaskStatus.COMPLETE)
    assert not is_valid_transition(TaskStatus.TODO, TaskStatus.REVIEW)
    assert not is_valid_transition(TaskStatus.COMPLETE, TaskStatus.TODO)
    assert not is_valid_transition(TaskStatus.COMPLETE, TaskStatus.IN_PROGRESS)
    assert not is_valid_transition(TaskStatus.IN_PROGRESS, TaskStatus.TODO)


# ---- Project CRUD tests ----

@pytest.mark.asyncio
async def test_create_project(svc):
    project = await svc.create_project(name="Test Project", user_id=1)
    assert project.id == 1
    assert project.name == "Test Project"
    assert project.user_id == 1


@pytest.mark.asyncio
async def test_list_projects_filters_by_user(svc):
    await svc.create_project(name="P1", user_id=1)
    await svc.create_project(name="P2", user_id=2)

    projects = await svc.list_projects(user_id=1)
    assert len(projects) == 1
    assert projects[0].name == "P1"


@pytest.mark.asyncio
async def test_delete_project_cascades(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", user_id=1)
    run = await svc.create_run(task.id)

    assert await svc.delete_project(project.id) is True
    assert await svc.get_project(project.id) is None
    assert await svc.get_task(task.id) is None


# ---- Task CRUD tests ----

@pytest.mark.asyncio
async def test_create_task(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(
        project.id, title="Task 1",
        description="Do something", agent_type="claude_code",
        user_id=1,
    )
    assert task.id == 1
    assert task.title == "Task 1"
    assert task.status == TaskStatus.TODO
    assert task.agent_type == "claude_code"


@pytest.mark.asyncio
async def test_create_task_nonexistent_project(svc):
    with pytest.raises(ValueError, match="Project 999 not found"):
        await svc.create_task(999, title="T1", user_id=1)


@pytest.mark.asyncio
async def test_list_tasks_with_status_filter(svc):
    project = await svc.create_project(name="P1", user_id=1)
    await svc.create_task(project.id, title="T1", user_id=1)
    t2 = await svc.create_task(project.id, title="T2", user_id=1)
    await svc.transition_task(t2.id, TaskStatus.IN_PROGRESS)

    todo_tasks = await svc.list_tasks(project.id, status=TaskStatus.TODO)
    assert len(todo_tasks) == 1

    ip_tasks = await svc.list_tasks(project.id, status=TaskStatus.IN_PROGRESS)
    assert len(ip_tasks) == 1


# ---- Dependency tests ----

@pytest.mark.asyncio
async def test_dependency_gating(svc):
    project = await svc.create_project(name="P1", user_id=1)
    t1 = await svc.create_task(project.id, title="T1", user_id=1)
    t2 = await svc.create_task(project.id, title="T2", dependency_id=t1.id, user_id=1)

    # T2 depends on T1, which is TODO → not ready
    assert await svc.check_dependency_ready(t2.id) is False

    # Complete T1
    await svc.transition_task(t1.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(t1.id, TaskStatus.REVIEW)
    await svc.transition_task(t1.id, TaskStatus.COMPLETE)

    # Now T2 should be ready
    assert await svc.check_dependency_ready(t2.id) is True


@pytest.mark.asyncio
async def test_cycle_detection(svc):
    project = await svc.create_project(name="P1", user_id=1)
    t1 = await svc.create_task(project.id, title="T1", user_id=1)
    t2 = await svc.create_task(project.id, title="T2", dependency_id=t1.id, user_id=1)

    # T3 depending on T2 depending on T1 → no cycle
    t3 = await svc.create_task(project.id, title="T3", dependency_id=t2.id, user_id=1)
    assert t3.dependency_id == t2.id


@pytest.mark.asyncio
async def test_no_dependency_always_ready(svc):
    project = await svc.create_project(name="P1", user_id=1)
    t1 = await svc.create_task(project.id, title="T1", user_id=1)

    assert await svc.check_dependency_ready(t1.id) is True


# ---- State transition tests ----

@pytest.mark.asyncio
async def test_transition_todo_to_inprogress(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", user_id=1)

    updated = await svc.transition_task(task.id, TaskStatus.IN_PROGRESS)
    assert updated.status == TaskStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_invalid_transition_raises(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", user_id=1)

    with pytest.raises(ValueError, match="Invalid transition"):
        await svc.transition_task(task.id, TaskStatus.COMPLETE)


# ---- Run tests ----

@pytest.mark.asyncio
async def test_create_and_complete_run(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", agent_type="claude_code", user_id=1)

    run = await svc.create_run(task.id, session_id="acp-session-1")
    assert run.status == RunStatus.RUNNING
    assert run.session_id == "acp-session-1"
    assert run.agent_type == "claude_code"

    completed = await svc.complete_run(run.id, result_summary="Done")
    assert completed.status == RunStatus.COMPLETED
    assert completed.result_summary == "Done"


@pytest.mark.asyncio
async def test_fail_run(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", user_id=1)

    run = await svc.create_run(task.id)
    failed = await svc.fail_run(run.id, error="Timeout")
    assert failed.status == RunStatus.FAILED
    assert failed.error == "Timeout"


def test_legacy_sqlite_factory_reexports_dedicated_factory():
    assert get_orchestration_db is db_factory.get_orchestration_db


def test_sqlite_factory_is_available_from_dedicated_module(monkeypatch, tmp_path):
    sentinel = object()

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    db_factory.get_orchestration_db.cache_clear()
    monkeypatch.setattr(
        db_factory.OrchestrationDB,
        "for_user",
        classmethod(lambda cls, user_id: sentinel),
        raising=False,
    )

    try:
        assert db_factory.get_orchestration_db(7) is sentinel
    finally:
        db_factory.get_orchestration_db.cache_clear()


@pytest.mark.asyncio
async def test_list_runs(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", user_id=1)

    await svc.create_run(task.id, session_id="s1")
    await svc.create_run(task.id, session_id="s2")

    runs = await svc.list_runs(task.id)
    assert len(runs) == 2


# ---- Reviewer gate tests ----

@pytest.mark.asyncio
async def test_review_approved(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", user_id=1)
    await svc.transition_task(task.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(task.id, TaskStatus.REVIEW)

    result = await svc.submit_review(task.id, approved=True)
    assert result.status == TaskStatus.COMPLETE
    assert result.review_count == 1


@pytest.mark.asyncio
async def test_review_rejected_returns_to_inprogress(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", max_review_attempts=3, user_id=1)
    await svc.transition_task(task.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(task.id, TaskStatus.REVIEW)

    result = await svc.submit_review(task.id, approved=False)
    assert result.status == TaskStatus.IN_PROGRESS
    assert result.review_count == 1


@pytest.mark.asyncio
async def test_review_max_attempts_triggers_triage(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", max_review_attempts=2, user_id=1)

    # First review cycle
    await svc.transition_task(task.id, TaskStatus.IN_PROGRESS)
    await svc.transition_task(task.id, TaskStatus.REVIEW)
    await svc.submit_review(task.id, approved=False)  # review_count=1, back to inprogress

    # Second review cycle
    await svc.transition_task(task.id, TaskStatus.REVIEW)
    result = await svc.submit_review(task.id, approved=False)  # review_count=2 >= max(2) → triage
    assert result.status == TaskStatus.TRIAGE
    assert result.review_count == 2


@pytest.mark.asyncio
async def test_review_on_non_review_task_raises(svc):
    project = await svc.create_project(name="P1", user_id=1)
    task = await svc.create_task(project.id, title="T1", user_id=1)

    with pytest.raises(ValueError, match="not in review status"):
        await svc.submit_review(task.id, approved=True)


# ---- Summary test ----

@pytest.mark.asyncio
async def test_project_summary(svc):
    project = await svc.create_project(name="P1", user_id=1)
    await svc.create_task(project.id, title="T1", user_id=1)
    t2 = await svc.create_task(project.id, title="T2", user_id=1)
    await svc.transition_task(t2.id, TaskStatus.IN_PROGRESS)

    summary = await svc.get_project_summary(project.id)
    assert summary["total_tasks"] == 2
    assert summary["status_counts"]["todo"] == 1
    assert summary["status_counts"]["inprogress"] == 1
