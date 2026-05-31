"""Tests for Agent Orchestration API endpoints (Phase 4.2)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Agent_Orchestration.models import TaskStatus
from tldw_Server_API.app.core.Agent_Orchestration.orchestration_service import (
    OrchestrationService,
)
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

    await svc.create_run(task.id, session_id="s1")
    await svc.create_run(task.id, session_id="s2")
    await svc.create_run(task.id, session_id="s3")

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

    def fake_audit_events(*, session_id: str):
        with acp_mod._ACP_AUDIT_LOCK:
            return [
                dict(event)
                for event in acp_mod._ACP_AUDIT_EVENTS
                if event.get("session_id") == session_id
            ]

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(acp_mod, "_acp_list_audit_events", fake_audit_events)
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
        assert enriched_run["history"]["artifacts"] == [
            {"artifact_count": 1, "session_id": "session-success"}
        ]
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


class _CanonicalWorkspaceDB:
    def __init__(self, workspaces: dict[str, dict[str, Any]] | None = None) -> None:
        self.workspaces = dict(workspaces or {})

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        return self.workspaces.get(workspace_id)


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


class _ArtifactPromotionClient:
    async def create_session(self, *_args, **_kwargs):
        return "session-promotion-1"

    async def prompt(self, *_args, **_kwargs):
        return {
            "stopReason": "end",
            "taskCompletion": {
                "status": "completed",
                "summary": "Brief ready",
                "artifacts": [
                    {
                        "id": "brief-1",
                        "artifact_type": "workspace_brief",
                        "title": "ACP Research Brief",
                        "content": "# Brief\nGrounded answer.",
                        "summary": "Grounded answer.",
                        "source_lineage": {
                            "sources": [
                                {
                                    "source_id": "src-1",
                                    "source_type": "media",
                                    "label": "Transcript",
                                }
                            ]
                        },
                    }
                ],
            },
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
    def __init__(self) -> None:
        self.create_session_calls = []

    async def create_session(self, *_args: Any, **kwargs: Any) -> str:
        self.create_session_calls.append({"args": _args, "kwargs": kwargs})
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


async def test_canonical_workspace_bridge_creates_linked_execution_workspace(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    canonical_db = _CanonicalWorkspaceDB(
        {"workspace-alpha": {"id": "workspace-alpha", "name": "Alpha Workspace"}}
    )

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "_allowed_workspace_roots", lambda: (tmp_path.resolve(),))

    try:
        response = await orch_mod.ensure_canonical_workspace_bridge(
            orch_mod.CanonicalWorkspaceBridgeRequest(
                canonical_workspace_id="workspace-alpha",
                root_path=str(workspace_root),
                metadata={"existing": "kept"},
            ),
            user=_TestUser(),
            canonical_db=canonical_db,
        )

        assert response.id > 0
        assert response.root_path == str(workspace_root)
        assert response.metadata["existing"] == "kept"
        assert response.metadata["canonical_workspace_id"] == "workspace-alpha"
        assert response.metadata["canonical_workspace_source"] == "research_workspace"
        assert response.metadata["link_status"] == "linked"
        assert response.canonical_workspace.canonical_workspace_id == "workspace-alpha"
        assert response.canonical_workspace.acp_workspace_id == response.id
    finally:
        db.close()


async def test_canonical_workspace_bridge_reuses_existing_link(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    db.create_workspace(
        name="Linked",
        root_path=str(workspace_root),
        metadata={
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "research_workspace",
            "link_status": "linked",
        },
    )
    canonical_db = _CanonicalWorkspaceDB(
        {"workspace-alpha": {"id": "workspace-alpha", "name": "Alpha Workspace"}}
    )

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "_allowed_workspace_roots", lambda: (tmp_path.resolve(),))

    try:
        response = await orch_mod.ensure_canonical_workspace_bridge(
            orch_mod.CanonicalWorkspaceBridgeRequest(
                canonical_workspace_id="workspace-alpha",
                root_path=str(workspace_root),
            ),
            user=_TestUser(),
            canonical_db=canonical_db,
        )

        assert response.name == "Linked"
        assert len(db.list_workspaces()) == 1
    finally:
        db.close()


async def test_canonical_workspace_bridge_links_existing_unlinked_root(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    existing = db.create_workspace(
        name="Existing Root",
        root_path=str(workspace_root),
        metadata={"owner": "research"},
    )
    canonical_db = _CanonicalWorkspaceDB(
        {"workspace-alpha": {"id": "workspace-alpha", "name": "Alpha Workspace"}}
    )

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "_allowed_workspace_roots", lambda: (tmp_path.resolve(),))

    try:
        response = await orch_mod.ensure_canonical_workspace_bridge(
            orch_mod.CanonicalWorkspaceBridgeRequest(
                canonical_workspace_id="workspace-alpha",
                root_path=str(workspace_root),
            ),
            user=_TestUser(),
            canonical_db=canonical_db,
        )

        assert response.id == existing.id
        assert response.metadata["owner"] == "research"
        assert response.metadata["canonical_workspace_id"] == "workspace-alpha"
        assert len(db.list_workspaces()) == 1
    finally:
        db.close()


async def test_canonical_workspace_bridge_rejects_missing_canonical_workspace(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "_allowed_workspace_roots", lambda: (tmp_path.resolve(),))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await orch_mod.ensure_canonical_workspace_bridge(
                orch_mod.CanonicalWorkspaceBridgeRequest(
                    canonical_workspace_id="workspace-missing",
                    root_path=str(workspace_root),
                ),
                user=_TestUser(),
                canonical_db=_CanonicalWorkspaceDB(),
            )

        assert exc_info.value.status_code == 404
        assert db.list_workspaces() == []
    finally:
        db.close()


async def test_canonical_workspace_bridge_requires_allowed_root(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    canonical_db = _CanonicalWorkspaceDB(
        {"workspace-alpha": {"id": "workspace-alpha", "name": "Alpha Workspace"}}
    )

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "_allowed_workspace_roots", lambda: ())

    try:
        with pytest.raises(HTTPException) as exc_info:
            await orch_mod.ensure_canonical_workspace_bridge(
                orch_mod.CanonicalWorkspaceBridgeRequest(
                    canonical_workspace_id="workspace-alpha",
                    root_path=str(workspace_root),
                ),
                user=_TestUser(),
                canonical_db=canonical_db,
            )

        assert exc_info.value.status_code == 503
        assert db.list_workspaces() == []
    finally:
        db.close()


async def test_canonical_workspace_bridge_rejects_root_linked_to_other_workspace(
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    db.create_workspace(
        name="Linked Elsewhere",
        root_path=str(workspace_root),
        metadata={
            "canonical_workspace_id": "workspace-other",
            "canonical_workspace_source": "research_workspace",
            "link_status": "linked",
        },
    )
    canonical_db = _CanonicalWorkspaceDB(
        {"workspace-alpha": {"id": "workspace-alpha", "name": "Alpha Workspace"}}
    )

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "_allowed_workspace_roots", lambda: (tmp_path.resolve(),))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await orch_mod.ensure_canonical_workspace_bridge(
                orch_mod.CanonicalWorkspaceBridgeRequest(
                    canonical_workspace_id="workspace-alpha",
                    root_path=str(workspace_root),
                ),
                user=_TestUser(),
                canonical_db=canonical_db,
            )

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "canonical_workspace_bridge_conflict"
    finally:
        db.close()


async def test_project_and_task_detail_include_canonical_workspace(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace = db.create_workspace(
        name="Linked",
        root_path=str(tmp_path / "workspace"),
        metadata={
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "research_workspace",
            "link_status": "linked",
        },
    )
    project = db.create_project(name="P1", workspace_id=workspace.id)
    task = db.create_task(project.id, title="T1", description="Dispatch me", agent_type="codex")

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)

    try:
        project_detail = await orch_mod.get_project(project.id, user=_TestUser())
        task_detail = await orch_mod.get_task(task.id, user=_TestUser())

        assert project_detail.workspace.canonical_workspace.canonical_workspace_id == "workspace-alpha"
        assert project_detail.canonical_workspace.canonical_workspace_id == "workspace-alpha"
        assert project_detail.canonical_workspace.acp_workspace_id == workspace.id
        assert task_detail.canonical_workspace.canonical_workspace_id == "workspace-alpha"
        assert task_detail.canonical_workspace.acp_workspace_id == workspace.id
    finally:
        db.close()


async def test_list_projects_filters_by_canonical_workspace(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    alpha_workspace = db.create_workspace(
        name="Alpha execution",
        root_path=str(tmp_path / "alpha"),
        metadata={
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "research_workspace",
            "link_status": "linked",
        },
    )
    beta_workspace = db.create_workspace(
        name="Beta execution",
        root_path=str(tmp_path / "beta"),
        metadata={
            "canonical_workspace_id": "workspace-beta",
            "canonical_workspace_source": "research_workspace",
            "link_status": "linked",
        },
    )
    alpha_project = db.create_project(
        name="Alpha agent work",
        workspace_id=alpha_workspace.id,
    )
    db.create_project(name="Beta agent work", workspace_id=beta_workspace.id)
    metadata_project = db.create_project(
        name="Alpha legacy work",
        metadata={
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "research_workspace",
        },
    )
    db.create_project(name="Unrelated unbound work")

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)

    try:
        results = await orch_mod.list_projects(
            workspace_id=None,
            unbound=False,
            canonical_workspace_id="workspace-alpha",
            canonical_workspace_source="research_workspace",
            user=_TestUser(),
        )

        assert [project.id for project in results] == [
            metadata_project.id,
            alpha_project.id,
        ]
        assert {project.name for project in results} == {
            "Alpha agent work",
            "Alpha legacy work",
        }
        assert all(
            project.canonical_workspace is None
            or project.canonical_workspace.canonical_workspace_id == "workspace-alpha"
            for project in results
        )
    finally:
        db.close()


async def test_list_projects_canonical_filter_normalizes_legacy_source(
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace = db.create_workspace(
        name="Legacy source link",
        root_path=str(tmp_path / "legacy"),
        metadata={
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "workspace_playground",
            "link_status": "linked",
        },
    )
    project = db.create_project(name="Legacy source work", workspace_id=workspace.id)
    db.create_project(
        name="Wrong source work",
        metadata={
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "other",
        },
    )

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)

    try:
        results = await orch_mod.list_projects(
            workspace_id=None,
            unbound=False,
            canonical_workspace_id="workspace-alpha",
            canonical_workspace_source="research_workspace",
            user=_TestUser(),
        )

        assert [item.id for item in results] == [project.id]
        assert (
            results[0].canonical_workspace.canonical_workspace_source
            == "research_workspace"
        )
    finally:
        db.close()


async def test_dispatch_run_inherits_trusted_root_from_canonical_bridge(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    workspace_root = tmp_path / "workspace"
    (workspace_root / "src").mkdir(parents=True)
    workspace = db.create_workspace(
        name="Linked",
        root_path=str(workspace_root),
        metadata={
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "research_workspace",
            "link_status": "linked",
        },
    )
    project = db.create_project(name="P1", workspace_id=workspace.id)
    task = db.create_task(project.id, title="T1", description="Dispatch me", agent_type="codex")
    client = _WorkspaceSessionCaptureClient()

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return client

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(orch_mod, "_allowed_workspace_roots", lambda: (tmp_path.resolve(),))
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
            orch_mod.RunDispatchRequest(cwd="src"),
            user=_TestUser(),
        )

        assert result["status"] == TaskStatus.COMPLETE
        assert client.create_session_calls[0]["args"][0] == str(workspace_root / "src")

        escape_task = db.create_task(
            project.id,
            title="T2",
            description="Reject escaped cwd",
            agent_type="codex",
        )
        with pytest.raises(HTTPException) as exc_info:
            await orch_mod.dispatch_run(
                escape_task.id,
                orch_mod.RunDispatchRequest(cwd="../outside"),
                user=_TestUser(),
            )
        assert exc_info.value.status_code == 403
    finally:
        db.close()


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


async def test_task_completion_signal_rejects_multiple_markers():
    """Completion validation should require exactly one structured marker."""
    from tldw_Server_API.app.core.Agent_Orchestration.completion_signals import (
        CompletionSignalValidationError,
        validate_task_completion_signal,
    )

    with pytest.raises(CompletionSignalValidationError) as exc_info:
        validate_task_completion_signal(
            {
                "content": (
                    '<acp-task-completion>{"status":"completed","summary":"one"}</acp-task-completion>'
                    '<acp-task-completion>{"status":"rejected","summary":"two"}</acp-task-completion>'
                )
            }
        )

    assert exc_info.value.reason == "multiple"


async def test_review_decision_signal_rejects_multiple_markers():
    """Reviewer validation should require exactly one structured marker."""
    from tldw_Server_API.app.core.Agent_Orchestration.completion_signals import (
        ReviewDecisionValidationError,
        validate_review_decision_signal,
    )

    with pytest.raises(ReviewDecisionValidationError) as exc_info:
        validate_review_decision_signal(
            {
                "content": (
                    '<acp-review-decision>{"approved":true,"feedback":"pass"}</acp-review-decision>'
                    '<acp-review-decision>{"approved":false,"feedback":"fail"}</acp-review-decision>'
                )
            }
        )

    assert exc_info.value.reason == "multiple"


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
        fake_logger.exception.assert_called_once_with("Failed to create ACP session")
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


async def test_dispatch_run_promotes_completion_artifact_to_canonical_workspace(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    note_db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    note_db.upsert_workspace("workspace-alpha", "Alpha Workspace")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = db.create_workspace(
        name="Linked",
        root_path=str(workspace_root),
        metadata={
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "research_workspace",
            "link_status": "linked",
        },
    )
    project = db.create_project(name="P1", workspace_id=workspace.id)
    task = db.create_task(project.id, title="T1", description="Create a brief", agent_type="codex")
    client = _ArtifactPromotionClient()

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
    monkeypatch.setattr(orch_mod, "_allowed_workspace_roots", lambda: (tmp_path.resolve(),))

    try:
        result = await orch_mod.dispatch_run(
            task.id,
            orch_mod.RunDispatchRequest(),
            user=_TestUser(),
            canonical_db=note_db,
        )

        assert result["status"] == TaskStatus.COMPLETE
        assert result["artifact_promotion"] == {
            "created_artifact_ids": ["brief-1"],
            "updated_artifact_ids": [],
            "skipped": [],
            "errors": [],
        }
        [artifact] = note_db.list_workspace_artifacts("workspace-alpha")
        assert artifact["id"] == "brief-1"
        assert artifact["review_state"] == "accepted"
        assert artifact["producer_metadata"]["producer_type"] == "acp"
        assert artifact["producer_metadata"]["run_id"] == str(result["run_id"])
        assert artifact["producer_metadata"]["session_id"] == "session-promotion-1"
        assert artifact["source_lineage"]["sources"][0]["source_id"] == "src-1"
    finally:
        note_db.close_all_connections()
        db.close()


async def test_dispatch_run_reports_artifact_promotion_failure_without_rolling_back_task(
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration as orch_mod
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    note_db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    note_db.upsert_workspace("workspace-alpha", "Alpha Workspace")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = db.create_workspace(
        name="Linked",
        root_path=str(workspace_root),
        metadata={
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "research_workspace",
            "link_status": "linked",
        },
    )
    project = db.create_project(name="P1", workspace_id=workspace.id)
    task = db.create_task(project.id, title="T1", description="Create a brief", agent_type="codex")
    client = _ArtifactPromotionClient()

    async def fake_store():
        return _NoopSessionStore()

    async def fake_client():
        return client

    def failing_promotion(*_args, **_kwargs):
        raise RuntimeError("promotion backend unavailable")

    monkeypatch.setattr(orch_mod, "get_orchestration_db", lambda _user_id: db)
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
        fake_store,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.runner_client.get_runner_client",
        fake_client,
    )
    monkeypatch.setattr(orch_mod, "_allowed_workspace_roots", lambda: (tmp_path.resolve(),))
    monkeypatch.setattr(orch_mod, "promote_acp_completion_artifacts", failing_promotion)

    try:
        result = await orch_mod.dispatch_run(
            task.id,
            orch_mod.RunDispatchRequest(),
            user=_TestUser(),
            canonical_db=note_db,
        )

        assert result["status"] == TaskStatus.COMPLETE
        assert result["artifact_promotion"] == {
            "created_artifact_ids": [],
            "updated_artifact_ids": [],
            "skipped": [],
            "errors": [{"artifact_id": "all", "reason": "promotion_failed"}],
        }
        updated_task = db.get_task(task.id)
        assert updated_task.status == TaskStatus.COMPLETE
        assert note_db.list_workspace_artifacts("workspace-alpha") == []
    finally:
        note_db.close_all_connections()
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
        assert client.create_session_calls[0]["kwargs"]["session_env"] == {"WORKSPACE_TOKEN": "abc"}
        assert client.create_session_calls[0]["kwargs"]["mcp_servers"] == [
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
        assert runs[0].error == "completion_signal_invalid"
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
        assert runs[0].error == "completion_signal_invalid"
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
        assert runs[0].error == "completion_signal_invalid"
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
        fake_logger.exception.assert_called_once_with("ACP prompt failed")
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
