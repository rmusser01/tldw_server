from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows.failures import build_failure_envelope

_SCHEMA_VERSION = 1
_MAX_EVENTS = 1000


def build_run_investigation(
    db: WorkflowsDatabase,
    *,
    run_id: str,
    include_operator_detail: bool = False,
) -> dict[str, Any] | None:
    run = db.get_run(run_id)
    if run is None:
        return None

    step_runs = db.list_step_runs(run_id=run_id)
    attempts = db.list_step_attempts(run_id=run_id)
    events = db.get_events(run_id, limit=_MAX_EVENTS)
    artifacts = db.list_artifacts_for_run(run_id)

    failed_step = _select_failed_step(
        step_runs,
        allow_latest_fallback=str(getattr(run, "status", "") or "") == "failed",
    )
    failed_attempts = (
        _attempts_for_step(
            attempts,
            step_id=failed_step.get("step_id") if failed_step else None,
            step_run_id=failed_step.get("step_run_id") if failed_step else None,
        )
        if failed_step is not None
        else []
    )
    primary_failure = _build_primary_failure(
        run=run,
        failed_step=failed_step,
        failed_attempts=failed_attempts,
        events=events,
        artifacts=artifacts,
        include_operator_detail=include_operator_detail,
    )

    return {
        "run_id": run_id,
        "status": run.status,
        "schema_version": _SCHEMA_VERSION,
        "derived_from_event_seq": max((int(event.get("event_seq") or 0) for event in events), default=0),
        "failed_step": _serialize_step(
            failed_step,
            attempts=failed_attempts,
            include_operator_detail=include_operator_detail,
        ) if failed_step else None,
        "primary_failure": primary_failure,
        "attempts": [
            _serialize_attempt(attempt, include_operator_detail=include_operator_detail)
            for attempt in failed_attempts
        ],
        "evidence": {
            "event_count": len(events),
            "artifact_count": len(artifacts),
            "webhook_delivery_count": sum(1 for event in events if event.get("event_type") == "webhook_delivery"),
            "artifact_types": [artifact.get("type") for artifact in artifacts],
        },
        "recommended_actions": _recommended_actions(primary_failure, artifacts),
    }


def list_run_steps(
    db: WorkflowsDatabase,
    *,
    run_id: str,
    include_operator_detail: bool = False,
) -> dict[str, Any] | None:
    run = db.get_run(run_id)
    if run is None:
        return None
    attempts = db.list_step_attempts(run_id=run_id)
    attempts_by_step_run: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        attempts_by_step_run.setdefault(str(attempt.get("step_run_id") or ""), []).append(attempt)

    steps = []
    for step_run in db.list_step_runs(run_id=run_id):
        step_attempts = attempts_by_step_run.get(str(step_run.get("step_run_id") or ""), [])
        steps.append(
            _serialize_step(
                step_run,
                attempts=step_attempts,
                include_operator_detail=include_operator_detail,
            )
        )
    return {"run_id": run_id, "steps": steps}


def list_step_attempts(
    db: WorkflowsDatabase,
    *,
    run_id: str,
    step_id: str,
    include_operator_detail: bool = False,
) -> dict[str, Any] | None:
    run = db.get_run(run_id)
    if run is None:
        return None
    attempts = db.list_step_attempts(run_id=run_id, step_id=step_id)
    return {
        "run_id": run_id,
        "step_id": step_id,
        "attempts": [
            _serialize_attempt(attempt, include_operator_detail=include_operator_detail)
            for attempt in attempts
        ],
    }


def _select_failed_step(
    step_runs: list[dict[str, Any]],
    *,
    allow_latest_fallback: bool = False,
) -> dict[str, Any] | None:
    for step_run in reversed(step_runs):
        if str(step_run.get("status") or "") == "failed":
            return step_run
    if allow_latest_fallback and step_runs:
        return step_runs[-1]
    return None


def _attempts_for_step(
    attempts: list[dict[str, Any]],
    *,
    step_id: str | None,
    step_run_id: str | None,
) -> list[dict[str, Any]]:
    filtered = attempts
    if step_id:
        filtered = [attempt for attempt in filtered if str(attempt.get("step_id") or "") == str(step_id)]
    if step_run_id:
        scoped = [attempt for attempt in filtered if str(attempt.get("step_run_id") or "") == str(step_run_id)]
        if scoped:
            filtered = scoped
    return filtered


def _build_primary_failure(
    *,
    run: Any,
    failed_step: dict[str, Any] | None,
    failed_attempts: list[dict[str, Any]],
    events: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    include_operator_detail: bool,
) -> dict[str, Any] | None:
    latest_attempt = failed_attempts[-1] if failed_attempts else None
    step_type = str((failed_step or {}).get("type") or "unknown")
    if latest_attempt and latest_attempt.get("reason_code_core"):
        metadata = latest_attempt.get("metadata_json") or {}
        failure = {
            "reason_code_core": latest_attempt.get("reason_code_core"),
            "reason_code_detail": latest_attempt.get("reason_code_detail"),
            "category": metadata.get("category"),
            "blame_scope": metadata.get("blame_scope"),
            "retryable": bool(latest_attempt.get("retryable")) if latest_attempt.get("retryable") is not None else None,
            "retry_recommendation": metadata.get("retry_recommendation"),
            "error_summary": latest_attempt.get("error_summary"),
            "internal_detail": None,
        }
    else:
        status = str(getattr(run, "status", "") or "").strip().lower()
        if status != "failed":
            return None
        envelope = build_failure_envelope(run.error or run.status_reason, step_type=step_type)
        failure = {
            "reason_code_core": envelope.reason_code_core,
            "reason_code_detail": envelope.reason_code_detail,
            "category": envelope.category,
            "blame_scope": envelope.blame_scope,
            "retryable": envelope.retryable,
            "retry_recommendation": envelope.retry_recommendation,
            "error_summary": envelope.error_summary,
            "internal_detail": None,
        }
    if include_operator_detail:
        failure["internal_detail"] = {
            "run_error": getattr(run, "error", None),
            "event_count": len(events),
            "artifact_count": len(artifacts),
            "latest_event_types": [event.get("event_type") for event in events[-5:]],
        }
    return failure


def _serialize_step(
    step_run: dict[str, Any],
    *,
    attempts: list[dict[str, Any]],
    include_operator_detail: bool,
) -> dict[str, Any]:
    try:
        legacy_attempt_count = int(step_run.get("attempt") or 0)
    except (TypeError, ValueError):
        legacy_attempt_count = 0
    highest_attempt_number = 0
    for attempt in attempts:
        try:
            highest_attempt_number = max(
                highest_attempt_number,
                int(attempt.get("attempt_number") or 0),
            )
        except (TypeError, ValueError):
            continue
    attempt_count = max(legacy_attempt_count, len(attempts), highest_attempt_number)
    latest_failed_attempt = None
    for attempt in reversed(attempts):
        if str(attempt.get("status") or "") == "failed":
            latest_failed_attempt = attempt
            break
    latest_failure = None
    if latest_failed_attempt is not None:
        latest_failure = _serialize_failure(
            latest_failed_attempt,
            include_operator_detail=include_operator_detail,
        )
    return {
        "step_run_id": step_run.get("step_run_id"),
        "step_id": step_run.get("step_id"),
        "name": step_run.get("name"),
        "type": step_run.get("type"),
        "status": step_run.get("status"),
        "attempt_count": attempt_count,
        "started_at": step_run.get("started_at"),
        "ended_at": step_run.get("ended_at"),
        "error": step_run.get("error"),
        "latest_failure": latest_failure,
    }


def _serialize_failure(
    attempt: dict[str, Any],
    *,
    include_operator_detail: bool,
) -> dict[str, Any]:
    metadata = attempt.get("metadata_json") or {}
    return {
        "reason_code_core": attempt.get("reason_code_core"),
        "reason_code_detail": attempt.get("reason_code_detail"),
        "category": metadata.get("category"),
        "blame_scope": metadata.get("blame_scope"),
        "retryable": bool(attempt.get("retryable")) if attempt.get("retryable") is not None else None,
        "retry_recommendation": metadata.get("retry_recommendation"),
        "error_summary": attempt.get("error_summary"),
        "internal_detail": (
            {"failure_envelope": metadata.get("failure_envelope")}
            if include_operator_detail and metadata.get("failure_envelope") is not None
            else None
        ),
    }


def _serialize_attempt(
    attempt: dict[str, Any],
    *,
    include_operator_detail: bool,
) -> dict[str, Any]:
    metadata = dict(attempt.get("metadata_json") or {})
    if not include_operator_detail:
        metadata.pop("failure_envelope", None)
    return {
        "attempt_id": attempt.get("attempt_id"),
        "step_run_id": attempt.get("step_run_id"),
        "step_id": attempt.get("step_id"),
        "attempt_number": int(attempt.get("attempt_number") or 0),
        "status": attempt.get("status"),
        "started_at": attempt.get("started_at"),
        "ended_at": attempt.get("ended_at"),
        "duration_ms": attempt.get("duration_ms"),
        "reason_code_core": attempt.get("reason_code_core"),
        "reason_code_detail": attempt.get("reason_code_detail"),
        "retryable": bool(attempt.get("retryable")) if attempt.get("retryable") is not None else None,
        "error_summary": attempt.get("error_summary"),
        "metadata": metadata,
    }


def _recommended_actions(primary_failure: dict[str, Any] | None, artifacts: list[dict[str, Any]]) -> list[str]:
    if not primary_failure:
        return []
    recommendation = str(primary_failure.get("retry_recommendation") or "").strip().lower()
    if recommendation == "safe":
        actions = ["Retry the failed step with the same inputs."]
    elif recommendation == "conditional":
        actions = [
            "Review side effects before retrying the failed step.",
            "Inspect external dependency and delivery evidence before rerun.",
        ]
    else:
        actions = ["Inspect workflow definition, inputs, or policy state before retrying."]
    if artifacts:
        actions.append("Review captured artifacts for the failed run.")
    return actions
