"""Tests for Orchestration SQLite persistence."""

import sqlite3
import tempfile

import pytest

from tldw_Server_API.app.core.Agent_Orchestration.models import RunStatus, TaskStatus
from tldw_Server_API.app.core.DB_Management.Orchestration_DB import (
    InvalidTransitionError,
    OrchestrationDB,
    OrchestrationNotFoundError,
)


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        instance = OrchestrationDB(user_id=1, db_dir=tmp)
        yield instance
        instance.close()


class TestProjectCRUD:
    def test_create_and_get_project(self, db):
        project = db.create_project(name="Test Project", description="A test")
        assert project.id > 0
        assert project.name == "Test Project"
        fetched = db.get_project(project.id)
        assert fetched is not None
        assert fetched.name == "Test Project"

    def test_list_projects(self, db):
        db.create_project(name="P1")
        db.create_project(name="P2")
        projects = db.list_projects()
        assert len(projects) == 2

    def test_delete_project_cascades(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        db.create_run(t.id)
        assert db.delete_project(p.id) is True
        assert db.get_project(p.id) is None
        assert db.get_task(t.id) is None

    def test_delete_nonexistent(self, db):
        assert db.delete_project(999) is False


class TestTaskCRUD:
    def test_create_task_with_dependency(self, db):
        p = db.create_project(name="P1")
        t1 = db.create_task(p.id, title="T1")
        t2 = db.create_task(p.id, title="T2", dependency_id=t1.id)
        assert t2.dependency_id == t1.id

    def test_create_task_invalid_project(self, db):
        with pytest.raises(ValueError, match="not found"):
            db.create_task(999, title="T1")

    def test_create_task_invalid_dependency(self, db):
        p = db.create_project(name="P1")
        with pytest.raises(ValueError, match="not found"):
            db.create_task(p.id, title="T1", dependency_id=999)

    def test_transition_task(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        updated = db.transition_task(t.id, TaskStatus.IN_PROGRESS)
        assert updated.status == TaskStatus.IN_PROGRESS

    def test_invalid_transition_raises(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        with pytest.raises(ValueError, match="Invalid transition"):
            db.transition_task(t.id, TaskStatus.COMPLETE)

    def test_check_dependency_ready(self, db):
        p = db.create_project(name="P1")
        t1 = db.create_task(p.id, title="T1")
        t2 = db.create_task(p.id, title="T2", dependency_id=t1.id)
        assert db.check_dependency_ready(t2.id) is False
        db.transition_task(t1.id, TaskStatus.IN_PROGRESS)
        db.transition_task(t1.id, TaskStatus.REVIEW)
        db.transition_task(t1.id, TaskStatus.COMPLETE)
        assert db.check_dependency_ready(t2.id) is True

    def test_check_dependency_ready_no_dep(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        assert db.check_dependency_ready(t.id) is True

    def test_list_tasks_filter_status(self, db):
        p = db.create_project(name="P1")
        db.create_task(p.id, title="T1")
        t2 = db.create_task(p.id, title="T2")
        db.transition_task(t2.id, TaskStatus.IN_PROGRESS)
        tasks = db.list_tasks(p.id, status=TaskStatus.IN_PROGRESS)
        assert len(tasks) == 1
        assert tasks[0].title == "T2"

    def test_cycle_detection_simple(self, db):
        """A->B->C chain is valid (no cycle)."""
        p = db.create_project(name="P1")
        t1 = db.create_task(p.id, title="A")
        t2 = db.create_task(p.id, title="B", dependency_id=t1.id)
        t3 = db.create_task(p.id, title="C", dependency_id=t2.id)
        assert t3.dependency_id == t2.id

    def test_create_task_with_metadata(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1", metadata={"key": "value"})
        assert t.metadata == {"key": "value"}
        fetched = db.get_task(t.id)
        assert fetched.metadata == {"key": "value"}

    def test_transition_nonexistent_task_raises(self, db):
        with pytest.raises(ValueError, match="not found"):
            db.transition_task(999, TaskStatus.IN_PROGRESS)


class TestRunCRUD:
    def test_create_and_complete_run(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        run = db.create_run(t.id, agent_type="claude_code", session_id="sess-1")
        assert run.status == RunStatus.RUNNING
        completed = db.complete_run(run.id, result_summary="done", token_usage={"input_tokens": 100})
        assert completed.status == RunStatus.COMPLETED
        assert completed.result_summary == "done"
        assert completed.token_usage == {"input_tokens": 100}

    def test_fail_run(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        run = db.create_run(t.id)
        failed = db.fail_run(run.id, error="something broke")
        assert failed.status == RunStatus.FAILED
        assert failed.error == "something broke"

    def test_list_runs(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        first = db.create_run(t.id)
        db.complete_run(first.id)
        db.create_run(t.id)
        runs = db.list_runs(t.id)
        assert len(runs) == 2

    def test_create_run_rejects_duplicate_running_run(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        db.create_run(t.id)

        with pytest.raises(InvalidTransitionError, match="running run"):
            db.create_run(t.id)

    def test_running_run_uniqueness_is_enforced_by_database(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        db.create_run(t.id, agent_type="codex", session_id="session-1")

        with pytest.raises(sqlite3.IntegrityError):
            db._get_conn().execute(
                "INSERT INTO runs (task_id, session_id, status, agent_type, started_at) " "VALUES (?, ?, ?, ?, ?)",
                (
                    t.id,
                    "session-2",
                    RunStatus.RUNNING.value,
                    "codex",
                    "2026-06-24T00:00:00+00:00",
                ),
            )

    def test_has_running_run_tracks_active_status(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")

        assert db.has_running_run(t.id) is False
        run = db.create_run(t.id)
        assert db.has_running_run(t.id) is True
        db.complete_run(run.id)
        assert db.has_running_run(t.id) is False

    def test_terminal_run_cannot_be_rewritten(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        run = db.create_run(t.id)
        db.complete_run(run.id)

        with pytest.raises(InvalidTransitionError, match="completed to failed"):
            db.fail_run(run.id, error="late failure")

    def test_missing_run_update_raises_not_found(self, db):
        with pytest.raises(OrchestrationNotFoundError, match="Run 999 not found"):
            db.complete_run(999)

    def test_missing_run_session_update_raises_not_found(self, db):
        with pytest.raises(OrchestrationNotFoundError, match="Run 999 not found"):
            db.update_run_session_id(999, "session-999")


class TestReviewerGate:
    def test_submit_review_approved(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1", reviewer_agent_type="reviewer")
        db.transition_task(t.id, TaskStatus.IN_PROGRESS)
        db.transition_task(t.id, TaskStatus.REVIEW)
        result = db.submit_review(t.id, approved=True, feedback="LGTM")
        assert result.status == TaskStatus.COMPLETE
        reviews = db.list_reviews(t.id)
        assert len(reviews) == 1
        assert reviews[0]["feedback"] == "LGTM"

    def test_submit_review_rejected_back_to_inprogress(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1", max_review_attempts=3)
        db.transition_task(t.id, TaskStatus.IN_PROGRESS)
        db.transition_task(t.id, TaskStatus.REVIEW)
        result = db.submit_review(t.id, approved=False, feedback="Needs work")
        assert result.status == TaskStatus.IN_PROGRESS
        assert result.review_count == 1

    def test_submit_review_rejected_triage(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1", max_review_attempts=1)
        db.transition_task(t.id, TaskStatus.IN_PROGRESS)
        db.transition_task(t.id, TaskStatus.REVIEW)
        result = db.submit_review(t.id, approved=False, feedback="Still bad")
        assert result.status == TaskStatus.TRIAGE

    def test_submit_review_not_in_review_raises(self, db):
        p = db.create_project(name="P1")
        t = db.create_task(p.id, title="T1")
        with pytest.raises(ValueError, match="not in review"):
            db.submit_review(t.id, approved=True)


class TestProjectSummary:
    def test_get_project_summary(self, db):
        p = db.create_project(name="P1")
        db.create_task(p.id, title="T1")
        t2 = db.create_task(p.id, title="T2")
        db.transition_task(t2.id, TaskStatus.IN_PROGRESS)
        summary = db.get_project_summary(p.id)
        assert summary["total_tasks"] == 2
        assert summary["status_counts"]["todo"] == 1
        assert summary["status_counts"]["inprogress"] == 1


class TestUserScoping:
    def test_shared_db_project_task_run_and_review_methods_are_user_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            db1 = OrchestrationDB(user_id=1, db_dir=tmp)
            db2 = OrchestrationDB(user_id=2, db_dir=tmp)
            try:
                project = db1.create_project(name="P1")
                task = db1.create_task(project.id, title="T1")
                db1.transition_task(task.id, TaskStatus.IN_PROGRESS)
                run = db1.create_run(task.id)

                assert db2.get_project(project.id) is None
                assert db2.get_task(task.id) is None
                assert db2.list_tasks(project.id) == []
                assert db2.list_runs(task.id) == []
                assert db2.list_reviews(task.id) == []
                assert db2.get_project_summary(project.id) == {
                    "project_id": project.id,
                    "total_tasks": 0,
                    "status_counts": {},
                }

                with pytest.raises(OrchestrationNotFoundError, match="Project"):
                    db2.create_task(project.id, title="T2")
                with pytest.raises(OrchestrationNotFoundError, match="Task"):
                    db2.transition_task(task.id, TaskStatus.REVIEW)
                with pytest.raises(OrchestrationNotFoundError, match="Task"):
                    db2.create_run(task.id)
                with pytest.raises(OrchestrationNotFoundError, match="Run"):
                    db2.complete_run(run.id)
                with pytest.raises(OrchestrationNotFoundError, match="Run"):
                    db2.fail_run(run.id)
                with pytest.raises(OrchestrationNotFoundError, match="Task"):
                    db2.submit_review(task.id, approved=True)
            finally:
                db1.close()
                db2.close()
