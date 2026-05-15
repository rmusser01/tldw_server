"""Tests for promoting accepted ACP deliverables into workspace artifacts."""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Agent_Orchestration.artifact_promotion import (
    promote_acp_completion_artifacts,
)
from tldw_Server_API.app.core.Agent_Orchestration.completion_signals import (
    TaskCompletionSignal,
    TaskReviewDecision,
)
from tldw_Server_API.app.core.Agent_Orchestration.models import TaskStatus
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Orchestration_DB import (
    CANONICAL_WORKSPACE_ID_METADATA_KEY,
    CANONICAL_WORKSPACE_LINK_STATUS_METADATA_KEY,
    CANONICAL_WORKSPACE_LINKED_STATUS,
    CANONICAL_WORKSPACE_SOURCE_METADATA_KEY,
    OrchestrationDB,
)

pytestmark = pytest.mark.unit


def _build_context(tmp_path):
    orch_db = OrchestrationDB(user_id=1, db_dir=tmp_path)
    note_db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    note_db.upsert_workspace("workspace-alpha", "Alpha Workspace")
    workspace = orch_db.create_workspace(
        name="ACP Linked Workspace",
        root_path=str(tmp_path / "workspace"),
        metadata={
            CANONICAL_WORKSPACE_ID_METADATA_KEY: "workspace-alpha",
            CANONICAL_WORKSPACE_SOURCE_METADATA_KEY: "workspace_playground",
            CANONICAL_WORKSPACE_LINK_STATUS_METADATA_KEY: CANONICAL_WORKSPACE_LINKED_STATUS,
        },
    )
    project = orch_db.create_project(name="ACP Project", workspace_id=workspace.id)
    task = orch_db.create_task(
        project.id,
        title="Create a grounded brief",
        description="Summarize the selected sources.",
        agent_type="codex",
        reviewer_agent_type="reviewer",
    )
    run = orch_db.create_run(task.id, agent_type="codex", session_id="session-run-1")
    review_run = orch_db.create_run(task.id, agent_type="reviewer", session_id="session-review-1")
    return orch_db, note_db, project, workspace, task, run, review_run


def _brief_payload(**overrides):
    payload = {
        "id": "brief-1",
        "artifact_type": "workspace_brief",
        "title": "ACP Research Brief",
        "content": "# Brief\nGrounded finding.",
        "summary": "Grounded finding.",
        "source_lineage": {
            "sources": [
                {"source_id": "src-1", "source_type": "media", "label": "Transcript"}
            ]
        },
    }
    payload.update(overrides)
    return payload


def test_promotes_accepted_acp_brief_with_traceable_metadata(tmp_path):
    orch_db, note_db, project, workspace, task, run, review_run = _build_context(tmp_path)
    signal = TaskCompletionSignal(
        status="completed",
        summary="Brief ready",
        artifacts=[_brief_payload()],
        raw_payload={},
    )
    review = TaskReviewDecision(approved=True, feedback="Meets criteria")

    try:
        result = promote_acp_completion_artifacts(
            note_db,
            task=task,
            project=project,
            workspace=workspace,
            run=run,
            completion_signal=signal,
            final_status=TaskStatus.COMPLETE,
            review_decision=review,
            review_run=review_run,
        )

        assert result.created_artifact_ids == ["brief-1"]
        assert result.updated_artifact_ids == []
        assert result.skipped == []
        assert result.errors == []

        [artifact] = note_db.list_workspace_artifacts("workspace-alpha")
        assert artifact["id"] == "brief-1"
        assert artifact["artifact_type"] == "workspace_brief"
        assert artifact["status"] == "completed"
        assert artifact["review_state"] == "accepted"
        assert artifact["owner_scope"] == "workspace"
        assert artifact["owner_id"] == "workspace-alpha"
        assert artifact["project_id"] == str(project.id)
        assert artifact["task_id"] == str(task.id)
        assert artifact["producer_metadata"]["producer_type"] == "acp"
        assert artifact["producer_metadata"]["producer_id"] == str(task.id)
        assert artifact["producer_metadata"]["run_id"] == str(run.id)
        assert artifact["producer_metadata"]["session_id"] == "session-run-1"
        assert artifact["producer_metadata"]["review_run_id"] == str(review_run.id)
        assert artifact["producer_metadata"]["review_session_id"] == "session-review-1"
        assert artifact["source_lineage"]["sources"][0]["source_id"] == "src-1"
        assert artifact["review_metadata"]["decision"] == "accepted"
        assert artifact["review_metadata"]["reviewer"] == "reviewer"
        assert artifact["version_metadata"]["source"] == "acp_completion_signal"
    finally:
        note_db.close_all_connections()
        orch_db.close()


def test_updates_existing_promoted_artifact_version(tmp_path):
    orch_db, note_db, project, workspace, task, run, review_run = _build_context(tmp_path)
    first_signal = TaskCompletionSignal(
        status="completed",
        summary="Initial brief",
        artifacts=[_brief_payload(content="# Brief\nInitial.")],
        raw_payload={},
    )
    second_signal = TaskCompletionSignal(
        status="completed",
        summary="Updated brief",
        artifacts=[_brief_payload(content="# Brief\nUpdated.", summary="Updated finding.")],
        raw_payload={},
    )
    review = TaskReviewDecision(approved=True, feedback="Accepted")

    try:
        promote_acp_completion_artifacts(
            note_db,
            task=task,
            project=project,
            workspace=workspace,
            run=run,
            completion_signal=first_signal,
            final_status=TaskStatus.COMPLETE,
            review_decision=review,
            review_run=review_run,
        )

        result = promote_acp_completion_artifacts(
            note_db,
            task=task,
            project=project,
            workspace=workspace,
            run=run,
            completion_signal=second_signal,
            final_status=TaskStatus.COMPLETE,
            review_decision=review,
            review_run=review_run,
        )

        assert result.created_artifact_ids == []
        assert result.updated_artifact_ids == ["brief-1"]
        [artifact] = note_db.list_workspace_artifacts("workspace-alpha")
        assert artifact["version"] == 2
        assert artifact["content"] == "# Brief\nUpdated."
        versions = note_db.list_workspace_artifact_versions("workspace-alpha", "brief-1")
        assert [version["artifact_version_id"] for version in versions] == [
            "brief-1:v1",
            "brief-1:v2",
        ]
        assert versions[1]["previous_version_id"] == "brief-1:v1"
    finally:
        note_db.close_all_connections()
        orch_db.close()


@pytest.mark.parametrize(
    ("final_status", "expected_reason"),
    [
        (TaskStatus.IN_PROGRESS, "needs_revision"),
        (TaskStatus.TRIAGE, "rejected"),
    ],
)
def test_rejected_or_retry_outputs_do_not_create_accepted_artifacts(
    tmp_path,
    final_status,
    expected_reason,
):
    orch_db, note_db, project, workspace, task, run, review_run = _build_context(tmp_path)
    signal = TaskCompletionSignal(
        status="completed",
        summary="Brief ready",
        artifacts=[_brief_payload()],
        raw_payload={},
    )
    review = TaskReviewDecision(approved=False, feedback="Missing required evidence")

    try:
        result = promote_acp_completion_artifacts(
            note_db,
            task=task,
            project=project,
            workspace=workspace,
            run=run,
            completion_signal=signal,
            final_status=final_status,
            review_decision=review,
            review_run=review_run,
        )

        assert result.created_artifact_ids == []
        assert result.updated_artifact_ids == []
        assert result.errors == []
        assert result.skipped == [{"artifact_id": "brief-1", "reason": expected_reason}]
        assert note_db.list_workspace_artifacts("workspace-alpha") == []
    finally:
        note_db.close_all_connections()
        orch_db.close()


def test_preserves_redaction_contract_for_promoted_artifact(tmp_path):
    orch_db, note_db, project, workspace, task, run, review_run = _build_context(tmp_path)
    signal = TaskCompletionSignal(
        status="completed",
        summary="Redacted brief ready",
        artifacts=[
            _brief_payload(
                redaction={
                    "support_safe": False,
                    "redacted": True,
                    "retention_class": "restricted",
                    "redacted_fields": ["content"],
                },
            )
        ],
        raw_payload={},
    )

    try:
        promote_acp_completion_artifacts(
            note_db,
            task=task,
            project=project,
            workspace=workspace,
            run=run,
            completion_signal=signal,
            final_status=TaskStatus.COMPLETE,
            review_decision=TaskReviewDecision(approved=True, feedback="Accepted"),
            review_run=review_run,
        )

        [artifact] = note_db.list_workspace_artifacts("workspace-alpha")
        assert artifact["redaction"] == {
            "support_safe": False,
            "redacted": True,
            "retention_class": "restricted",
            "redacted_fields": ["content"],
        }
    finally:
        note_db.close_all_connections()
        orch_db.close()


def test_malformed_promotion_payload_is_not_promoted(tmp_path):
    orch_db, note_db, project, workspace, task, run, review_run = _build_context(tmp_path)
    signal = TaskCompletionSignal(
        status="completed",
        summary="Malformed artifact",
        artifacts=[_brief_payload(source_lineage={})],
        raw_payload={},
    )

    try:
        result = promote_acp_completion_artifacts(
            note_db,
            task=task,
            project=project,
            workspace=workspace,
            run=run,
            completion_signal=signal,
            final_status=TaskStatus.COMPLETE,
            review_decision=TaskReviewDecision(approved=True, feedback="Accepted"),
            review_run=review_run,
        )

        assert result.created_artifact_ids == []
        assert result.updated_artifact_ids == []
        assert result.skipped == []
        assert result.errors == [{"artifact_id": "brief-1", "reason": "missing_source_lineage"}]
        assert note_db.list_workspace_artifacts("workspace-alpha") == []
    finally:
        note_db.close_all_connections()
        orch_db.close()
