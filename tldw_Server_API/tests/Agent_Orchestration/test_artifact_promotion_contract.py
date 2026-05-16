"""Release contract tests for ACP completion artifact promotion."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from tldw_Server_API.app.core.Agent_Orchestration.artifact_promotion import (
    promote_acp_completion_artifacts,
)
from tldw_Server_API.app.core.Agent_Orchestration.completion_signals import (
    TaskCompletionSignal,
    TaskReviewDecision,
)
from tldw_Server_API.app.core.Agent_Orchestration.models import (
    ACPWorkspace,
    AgentProject,
    AgentRun,
    AgentTask,
    TaskStatus,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Orchestration_DB import (
    CANONICAL_WORKSPACE_ID_METADATA_KEY,
)
from tldw_Server_API.app.core.Workspaces.workspace_artifact_exports import (
    export_workspace_artifact_version,
)
from tldw_Server_API.app.core.exceptions import WorkspaceArtifactExportStateError

pytestmark = pytest.mark.unit


@pytest.fixture
def note_db(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    db.upsert_workspace("workspace-alpha", "Alpha Workspace")
    try:
        yield db
    finally:
        db.close_all_connections()


@pytest.fixture
def promotion_context():
    workspace = ACPWorkspace(
        id=9,
        name="Linked ACP Workspace",
        root_path="workspace-root",
        metadata={
            CANONICAL_WORKSPACE_ID_METADATA_KEY: "workspace-alpha",
            "canonical_workspace_source": "workspace_playground",
            "link_status": "linked",
        },
    )
    project = AgentProject(
        id=3,
        name="Release Signoff",
        workspace_id=workspace.id,
        user_id=1,
    )
    task = AgentTask(
        id=42,
        project_id=project.id,
        title="Create traceable ACP brief",
        status=TaskStatus.REVIEW,
        agent_type="codex",
        reviewer_agent_type="reviewer",
        review_count=1,
        max_review_attempts=3,
        user_id=1,
    )
    run = AgentRun(
        id=7,
        task_id=task.id,
        session_id="session-abc",
        agent_type="codex",
    )
    review_run = AgentRun(
        id=8,
        task_id=task.id,
        session_id="session-review",
        agent_type="reviewer",
    )
    review_decision = TaskReviewDecision(
        approved=True,
        feedback="Reviewer accepted the source-grounded brief.",
        raw_payload={"approved": True, "status": "accepted"},
    )
    return {
        "workspace": workspace,
        "project": project,
        "task": task,
        "run": run,
        "review_run": review_run,
        "review_decision": review_decision,
    }


def _brief_artifact(**overrides: Any) -> dict[str, Any]:
    artifact = {
        "id": "brief-1",
        "artifact_type": "workspace_brief",
        "title": "ACP Research Brief",
        "status": "completed",
        "content": "# Brief\nGrounded answer with citations.",
        "content_type": "text/markdown",
        "preview_text": "Grounded answer with citations.",
        "summary": "Grounded answer.",
        "source_lineage": {
            "sources": [
                {
                    "source_id": "src-1",
                    "source_type": "media",
                    "label": "Transcript",
                    "citation_spans": [{"start": 12, "end": 48}],
                }
            ]
        },
        "review_metadata": {"rubric": "release-signoff"},
        "version_metadata": {"revision_reason": "initial accepted version"},
        "export_refs": [{"format": "legacy", "artifact_version_id": "brief-1:v1"}],
        "redaction": {
            "support_safe": True,
            "redacted": False,
            "retention_class": "standard",
        },
        "schema_version": 1,
    }
    artifact.update(overrides)
    return artifact


def _completion_signal(artifacts: list[Any], *, summary: str = "Brief ready") -> TaskCompletionSignal:
    return TaskCompletionSignal(
        status="completed",
        summary=summary,
        artifacts=artifacts,
        raw_payload={"status": "completed", "summary": summary, "artifacts": artifacts},
    )


def _promote(
    note_db: CharactersRAGDB,
    promotion_context: dict[str, Any],
    artifacts: list[Any],
    *,
    final_status: TaskStatus = TaskStatus.COMPLETE,
    review_decision: TaskReviewDecision | None = None,
    review_run: AgentRun | None = None,
    summary: str = "Brief ready",
):
    return promote_acp_completion_artifacts(
        note_db,
        task=promotion_context["task"],
        project=promotion_context["project"],
        workspace=promotion_context["workspace"],
        run=promotion_context["run"],
        completion_signal=_completion_signal(artifacts, summary=summary),
        final_status=final_status,
        review_decision=(
            promotion_context["review_decision"]
            if review_decision is None
            else review_decision
        ),
        review_run=promotion_context["review_run"] if review_run is None else review_run,
    )


def test_accepted_acp_completion_promotes_exportable_traceable_artifact(
    note_db,
    promotion_context,
):
    result = _promote(note_db, promotion_context, [_brief_artifact()])

    assert result.created_artifact_ids == ["brief-1"]
    assert result.updated_artifact_ids == []
    assert result.skipped == []
    assert result.errors == []

    artifact = note_db.get_workspace_artifact("workspace-alpha", "brief-1")
    assert artifact is not None
    assert artifact["review_state"] == "accepted"
    assert artifact["owner_scope"] == "workspace"
    assert artifact["owner_id"] == "workspace-alpha"
    assert artifact["root_artifact_id"] == "brief-1"
    assert artifact["artifact_version_id"] == "brief-1:v1"
    assert artifact["previous_version_id"] is None
    assert artifact["producer_metadata"] == {
        "producer_type": "acp",
        "producer_id": "42",
        "task_id": "42",
        "task_ref": "acp-task:42",
        "project_id": "3",
        "acp_workspace_id": "9",
        "canonical_workspace_id": "workspace-alpha",
        "run_id": "7",
        "session_id": "session-abc",
        "agent_type": "codex",
        "completion_status": "completed",
        "review_run_id": "8",
        "review_session_id": "session-review",
        "reviewer_agent_type": "reviewer",
    }
    assert artifact["source_lineage"]["sources"][0]["source_id"] == "src-1"
    assert artifact["review_metadata"]["decision"] == "accepted"
    assert artifact["review_metadata"]["feedback_present"] is True
    assert artifact["version_metadata"]["completion_summary"] == "Brief ready"
    assert artifact["export_refs"][0] == {
        "format": "legacy",
        "artifact_version_id": "brief-1:v1",
    }
    assert artifact["redaction"] == {
        "support_safe": True,
        "redacted": False,
        "retention_class": "standard",
    }
    assert artifact["schema_version"] == 1

    exported = export_workspace_artifact_version(
        artifact,
        export_format="json",
        generated_at="2026-05-15T12:00:00+00:00",
    )
    exported_payload = json.loads(exported["content"])
    assert exported["artifact_version_id"] == "brief-1:v1"
    assert exported["export_ref"]["artifact_version_id"] == "brief-1:v1"
    assert exported_payload["metadata"]["producer_metadata"]["session_id"] == "session-abc"
    assert exported_payload["metadata"]["source_lineage"]["sources"][0]["source_id"] == "src-1"
    assert exported_payload["metadata"]["redaction"]["support_safe"] is True


def test_accepted_acp_completion_updates_existing_artifact_with_version_lineage(
    note_db,
    promotion_context,
):
    first = _promote(note_db, promotion_context, [_brief_artifact()])
    assert first.created_artifact_ids == ["brief-1"]

    second_run = replace(
        promotion_context["run"],
        id=11,
        session_id="session-updated",
    )
    promotion_context = {**promotion_context, "run": second_run}
    result = _promote(
        note_db,
        promotion_context,
        [
            _brief_artifact(
                content="# Brief\nUpdated, still grounded answer.",
                preview_text="Updated, still grounded answer.",
                summary="Updated grounded answer.",
                version_metadata={"revision_reason": "review follow-up incorporated"},
                export_refs=[],
                schema_version=2,
            )
        ],
        summary="Updated brief ready",
    )

    assert result.created_artifact_ids == []
    assert result.updated_artifact_ids == ["brief-1"]
    assert result.errors == []

    artifact = note_db.get_workspace_artifact("workspace-alpha", "brief-1")
    assert artifact is not None
    assert artifact["version"] == 2
    assert artifact["artifact_version_id"] == "brief-1:v2"
    assert artifact["previous_version_id"] == "brief-1:v1"
    assert artifact["producer_metadata"]["run_id"] == "11"
    assert artifact["producer_metadata"]["session_id"] == "session-updated"
    assert artifact["version_metadata"]["completion_summary"] == "Updated brief ready"
    assert artifact["version_metadata"]["revision_reason"] == "review follow-up incorporated"
    assert artifact["schema_version"] == 2

    versions = note_db.list_workspace_artifact_versions("workspace-alpha", "brief-1")
    assert [version["artifact_version_id"] for version in versions] == [
        "brief-1:v1",
        "brief-1:v2",
    ]
    assert versions[0]["previous_version_id"] is None
    assert versions[1]["previous_version_id"] == "brief-1:v1"
    assert versions[1]["source_lineage"]["sources"][0]["source_id"] == "src-1"


@pytest.mark.parametrize(
    ("final_status", "review_decision", "expected_reason"),
    [
        (
            TaskStatus.TRIAGE,
            TaskReviewDecision(approved=False, feedback="Rejected", raw_payload={}),
            "rejected",
        ),
        (
            TaskStatus.IN_PROGRESS,
            TaskReviewDecision(approved=False, feedback="Needs work", raw_payload={}),
            "needs_revision",
        ),
    ],
)
def test_non_accepted_acp_completion_does_not_promote_artifacts(
    note_db,
    promotion_context,
    final_status,
    review_decision,
    expected_reason,
):
    result = _promote(
        note_db,
        promotion_context,
        [_brief_artifact()],
        final_status=final_status,
        review_decision=review_decision,
    )

    assert result.created_artifact_ids == []
    assert result.updated_artifact_ids == []
    assert result.errors == []
    assert result.skipped == [{"artifact_id": "brief-1", "reason": expected_reason}]
    assert note_db.list_workspace_artifacts("workspace-alpha") == []


def test_malformed_and_non_promotable_acp_artifacts_are_reported_without_partial_create(
    note_db,
    promotion_context,
):
    result = _promote(
        note_db,
        promotion_context,
        [
            _brief_artifact(id="bad-redaction", redaction={"support_safe": "yes"}),
            _brief_artifact(id="missing-lineage", source_lineage={}),
            {
                "id": "session-log-1",
                "artifact_type": "session_log",
                "title": "Raw execution log",
                "content": "Tool trace",
                "source_lineage": {"sources": [{"source_id": "src-1"}]},
            },
        ],
    )

    assert result.created_artifact_ids == []
    assert result.updated_artifact_ids == []
    assert result.errors == [
        {"artifact_id": "bad-redaction", "reason": "invalid_redaction"},
        {"artifact_id": "missing-lineage", "reason": "missing_source_lineage"},
    ]
    assert result.skipped == [{"artifact_id": "session-log-1", "reason": "not_promotable"}]
    assert note_db.list_workspace_artifacts("workspace-alpha") == []


def test_unaccepted_traceable_artifact_snapshot_remains_non_exportable(note_db):
    artifact = note_db.add_workspace_artifact(
        "workspace-alpha",
        {
            "id": "brief-draft",
            "artifact_type": "workspace_brief",
            "title": "Draft Brief",
            "content": "Draft",
            "review_state": "needs_revision",
            "producer_metadata": {"producer_type": "acp", "session_id": "session-draft"},
            "source_lineage": {"sources": [{"source_id": "src-1"}]},
            "redaction": {"support_safe": True, "redacted": False},
        },
    )

    with pytest.raises(WorkspaceArtifactExportStateError, match="workspace_artifact_not_accepted"):
        export_workspace_artifact_version(artifact, export_format="md")
