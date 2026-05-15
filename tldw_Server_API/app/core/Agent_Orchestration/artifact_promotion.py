"""Promotion helpers for ACP deliverables that become workspace artifacts."""
from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

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

_PROMOTABLE_ARTIFACT_TYPES = frozenset(
    {
        "workspace_brief",
        "source_grounded_workspace_brief",
        "source_grounded_brief",
        "workspace_report",
        "workspace_spec",
        "workspace_action_plan",
        "workspace_table",
    }
)
_DEFAULT_REDACTION = {"support_safe": True, "redacted": False}
_PREVIEW_TEXT_MAX_CHARS = 500
_OPTIONAL_MAPPING_FIELDS = {
    "producer_metadata": "invalid_producer_metadata",
    "review_metadata": "invalid_review_metadata",
    "version_metadata": "invalid_version_metadata",
}


@dataclass(frozen=True)
class ACPArtifactPromotionResult:
    """Outcome summary for a single ACP completion promotion pass."""

    created_artifact_ids: list[str] = field(default_factory=list)
    updated_artifact_ids: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for API responses and audit metadata."""
        return asdict(self)


def promote_acp_completion_artifacts(
    note_db: CharactersRAGDB,
    *,
    task: AgentTask,
    project: AgentProject | None,
    workspace: ACPWorkspace | None,
    run: AgentRun,
    completion_signal: TaskCompletionSignal,
    final_status: TaskStatus | str,
    review_decision: TaskReviewDecision | None = None,
    review_run: AgentRun | None = None,
) -> ACPArtifactPromotionResult:
    """Create or update traceable workspace artifacts from accepted ACP output.

    Raw session artifacts remain ACP execution evidence. Only structured
    work-product candidates with source lineage are promoted.
    """
    result = ACPArtifactPromotionResult()
    if not completion_signal.artifacts:
        return result

    decision = _promotion_decision(final_status, review_decision)
    target_workspace_id = _target_workspace_id(workspace)
    existing_by_id: dict[str, dict[str, Any]] = {}
    if target_workspace_id and note_db.get_workspace(target_workspace_id):
        existing_by_id = {
            str(item["id"]): item
            for item in note_db.list_workspace_artifacts(target_workspace_id)
            if item.get("id")
        }

    for index, raw_artifact in enumerate(completion_signal.artifacts):
        artifact_id = _artifact_identifier(raw_artifact, task=task, run=run, index=index)
        if decision != "accepted":
            result.skipped.append({"artifact_id": artifact_id, "reason": decision})
            continue

        if not isinstance(raw_artifact, Mapping):
            _record_error(result, artifact_id, "malformed_artifact")
            continue
        artifact = dict(raw_artifact)

        artifact_type = _artifact_type(artifact)
        if not artifact_type:
            _record_error(result, artifact_id, "missing_artifact_type")
            continue

        if not _is_promotable_artifact(artifact):
            result.skipped.append({"artifact_id": artifact_id, "reason": "not_promotable"})
            continue

        error_reason = _validate_artifact_payload(artifact, target_workspace_id, note_db)
        if error_reason:
            _record_error(result, artifact_id, error_reason)
            continue

        try:
            payload = _workspace_artifact_payload(
                artifact,
                artifact_id=artifact_id,
                workspace_id=str(target_workspace_id),
                task=task,
                project=project,
                workspace=workspace,
                run=run,
                completion_signal=completion_signal,
                review_decision=review_decision,
                review_run=review_run,
            )

            existing = existing_by_id.get(artifact_id)
            if existing:
                updated = note_db.update_workspace_artifact(
                    str(target_workspace_id),
                    artifact_id,
                    _update_payload(payload),
                    expected_version=int(existing.get("version") or 1),
                )
                existing_by_id[artifact_id] = updated
                result.updated_artifact_ids.append(artifact_id)
            else:
                created = note_db.add_workspace_artifact(str(target_workspace_id), payload)
                existing_by_id[artifact_id] = created
                result.created_artifact_ids.append(artifact_id)
        except (RuntimeError, TypeError, ValueError, sqlite3.Error):
            logger.exception("ACP artifact promotion failed for artifact {}", artifact_id)
            _record_error(result, artifact_id, "promotion_failed")

    return result


def _record_error(result: ACPArtifactPromotionResult, artifact_id: str, reason: str) -> None:
    logger.warning(
        "ACP artifact promotion skipped malformed artifact {}: {}",
        artifact_id,
        reason,
    )
    result.errors.append({"artifact_id": artifact_id, "reason": reason})


def _status_value(value: TaskStatus | str) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _promotion_decision(
    final_status: TaskStatus | str,
    review_decision: TaskReviewDecision | None,
) -> str:
    status_value = _status_value(final_status)
    if review_decision is not None and review_decision.approved is False:
        return "needs_revision" if status_value == TaskStatus.IN_PROGRESS.value else "rejected"
    if status_value == TaskStatus.COMPLETE.value:
        return "accepted"
    if status_value == TaskStatus.IN_PROGRESS.value:
        return "needs_revision"
    if status_value == TaskStatus.TRIAGE.value:
        return "rejected"
    return "not_accepted"


def _target_workspace_id(workspace: ACPWorkspace | None) -> str | None:
    if workspace is None or not isinstance(workspace.metadata, dict):
        return None
    canonical_id = str(workspace.metadata.get(CANONICAL_WORKSPACE_ID_METADATA_KEY) or "").strip()
    return canonical_id or None


def _artifact_identifier(raw_artifact: Any, *, task: AgentTask, run: AgentRun, index: int) -> str:
    if isinstance(raw_artifact, Mapping):
        artifact_id = (
            raw_artifact.get("id")
            or raw_artifact.get("artifact_id")
            or raw_artifact.get("root_artifact_id")
        )
        if artifact_id:
            return str(artifact_id)
    return f"acp-task-{task.id}-run-{run.id}-{index + 1}"


def _artifact_type(artifact: Mapping[str, Any]) -> str:
    return str(
        artifact.get("artifact_type")
        or artifact.get("type")
        or artifact.get("kind")
        or artifact.get("promote_as")
        or ""
    ).strip().lower()


def _is_promotable_artifact(artifact: Mapping[str, Any]) -> bool:
    return _artifact_type(artifact) in _PROMOTABLE_ARTIFACT_TYPES


def _validate_artifact_payload(
    artifact: Mapping[str, Any],
    target_workspace_id: str | None,
    note_db: CharactersRAGDB,
) -> str | None:
    if not target_workspace_id:
        return "missing_workspace"
    if note_db.get_workspace(target_workspace_id) is None:
        return "workspace_not_found"
    if not str(artifact.get("title") or "").strip():
        return "missing_title"
    if not str(artifact.get("content") or "").strip():
        return "missing_content"
    source_lineage = artifact.get("source_lineage")
    if not isinstance(source_lineage, Mapping) or not source_lineage:
        return "missing_source_lineage"
    for field_name, reason in _OPTIONAL_MAPPING_FIELDS.items():
        if field_name in artifact and artifact[field_name] is not None:
            if not isinstance(artifact[field_name], Mapping):
                return reason
    export_refs = artifact.get("export_refs")
    if export_refs is not None and not isinstance(export_refs, (list, tuple)):
        return "invalid_export_refs"
    if "schema_version" in artifact:
        try:
            _schema_version(artifact.get("schema_version"))
        except (TypeError, ValueError):
            return "invalid_schema_version"
    return _validate_redaction(artifact.get("redaction"))


def _validate_redaction(redaction: Any) -> str | None:
    if redaction is None:
        return None
    if not isinstance(redaction, Mapping):
        return "invalid_redaction"
    for field_name in ("support_safe", "redacted"):
        if field_name in redaction and not isinstance(redaction[field_name], bool):
            return "invalid_redaction"
    return None


def _preview_text(artifact: Mapping[str, Any]) -> str | None:
    preview = str(artifact.get("preview_text") or "").strip()
    if preview:
        return preview
    summary = str(artifact.get("summary") or "").strip()
    if summary:
        return summary[:_PREVIEW_TEXT_MAX_CHARS]
    content = str(artifact.get("content") or "").strip()
    return content[:_PREVIEW_TEXT_MAX_CHARS] if content else None


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _schema_version(value: Any) -> int:
    if value is None or value == "":
        return 1
    if isinstance(value, bool):
        raise ValueError("schema_version must be an integer")
    return int(value)


def _workspace_artifact_payload(
    artifact: Mapping[str, Any],
    *,
    artifact_id: str,
    workspace_id: str,
    task: AgentTask,
    project: AgentProject | None,
    workspace: ACPWorkspace | None,
    run: AgentRun,
    completion_signal: TaskCompletionSignal,
    review_decision: TaskReviewDecision | None,
    review_run: AgentRun | None,
) -> dict[str, Any]:
    producer_metadata = _coerce_mapping(artifact.get("producer_metadata"))
    producer_metadata.update(
        {
            "producer_type": "acp",
            "producer_id": str(task.id),
            "task_id": str(task.id),
            "task_ref": f"acp-task:{task.id}",
            "project_id": str(project.id) if project else None,
            "acp_workspace_id": str(workspace.id) if workspace else None,
            "canonical_workspace_id": workspace_id,
            "run_id": str(run.id),
            "session_id": run.session_id,
            "agent_type": run.agent_type or task.agent_type,
            "completion_status": completion_signal.status,
        }
    )
    if review_run is not None:
        producer_metadata["review_run_id"] = str(review_run.id)
        producer_metadata["review_session_id"] = review_run.session_id
        producer_metadata["reviewer_agent_type"] = review_run.agent_type or task.reviewer_agent_type

    review_metadata = _coerce_mapping(artifact.get("review_metadata"))
    review_metadata.update(
        {
            "decision": "accepted",
            "reviewer": task.reviewer_agent_type,
            "feedback_present": bool(review_decision and review_decision.feedback),
            "review_count": int(task.review_count or 0),
            "max_review_attempts": int(task.max_review_attempts or 0),
        }
    )
    if review_run is not None:
        review_metadata["review_run_id"] = str(review_run.id)
        review_metadata["review_session_id"] = review_run.session_id

    version_metadata = _coerce_mapping(artifact.get("version_metadata"))
    version_metadata.update(
        {
            "source": "acp_completion_signal",
            "completion_summary": completion_signal.summary,
            "completion_status": completion_signal.status,
            "created_from_run_id": str(run.id),
        }
    )

    return {
        "id": artifact_id,
        "artifact_type": _artifact_type(artifact),
        "title": str(artifact.get("title") or "").strip(),
        "status": str(artifact.get("status") or "completed"),
        "content": artifact.get("content"),
        "content_type": str(artifact.get("content_type") or "text/markdown"),
        "preview_text": _preview_text(artifact),
        "summary": artifact.get("summary") or completion_signal.summary,
        "review_state": "accepted",
        "owner_scope": "workspace",
        "owner_id": workspace_id,
        "project_id": str(project.id) if project else None,
        "task_id": str(task.id),
        "source_collection_id": artifact.get("source_collection_id"),
        "producer_metadata": producer_metadata,
        "source_lineage": _coerce_mapping(artifact.get("source_lineage")),
        "review_metadata": review_metadata,
        "version_metadata": version_metadata,
        "export_refs": _coerce_list(artifact.get("export_refs")),
        "redaction": _coerce_mapping(artifact.get("redaction")) or dict(_DEFAULT_REDACTION),
        "schema_version": _schema_version(artifact.get("schema_version")),
    }


def _update_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    update = dict(payload)
    update.pop("id", None)
    return update
