"""Real backend timestamps keep the workflow API string contract and redaction."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.workflows import (
    EventResponse,
    WorkflowRunInvestigationResponse,
    WorkflowRunListItem,
    WorkflowRunStepResponse,
    WorkflowRunStepsResponse,
    WorkflowStepAttemptResponse,
    WorkflowStepAttemptsResponse,
)
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows.investigation import (
    build_run_investigation,
    list_run_steps,
    list_step_attempts,
)

pytestmark = pytest.mark.integration


@pytest.fixture(params=("sqlite", "postgres"))
def workflow_db(request, tmp_path):
    backend = request.getfixturevalue("pg_restricted_backend") if request.param == "postgres" else None
    db = WorkflowsDatabase(str(tmp_path / "workflows.db"), backend=backend)
    try:
        yield db
    finally:
        db.close()


def seed(db, status):
    run_id = uuid4().hex
    step_run_id = run_id + ":first:1"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        definition_version=1,
        definition_snapshot={
            "name": "timestamp contract",
            "version": 1,
            "steps": [{"id": "first", "type": "template_render", "config": {"template": "ok"}}],
        },
    )
    db.create_step_run(
        step_run_id=step_run_id,
        tenant_id="default",
        run_id=run_id,
        step_id="first",
        name="First",
        step_type="template_render",
        status="running",
        inputs={},
    )
    attempt_id = db.create_step_attempt(
        tenant_id="default",
        run_id=run_id,
        step_run_id=step_run_id,
        step_id="first",
        attempt_number=1,
        metadata={"failure_envelope": {"private_marker": "operator-only-detail"}},
    )
    if status != "running":
        db.complete_step_run(step_run_id=step_run_id, status=status, outputs={"text": "ok"})
        db.complete_step_attempt(
            attempt_id=attempt_id,
            status=status,
            reason_code_core="transient_network_error" if status == "failed" else None,
            retryable=status == "failed",
            error_summary="request failed" if status == "failed" else None,
        )
    db.update_run_status(
        run_id,
        status=status,
        started_at="2026-10-21T10:00:00+00:00",
        ended_at=None if status == "running" else "2026-10-21T10:00:01+00:00",
    )
    db.append_event("default", run_id, "step_started", {"step_id": "first"}, step_run_id=step_run_id)
    return run_id


@pytest.mark.parametrize(
    "resource,status",
    (
        ("steps", "running"),
        ("steps", "succeeded"),
        ("attempts", "running"),
        ("attempts", "succeeded"),
        ("investigation", "failed"),
        ("runs", "running"),
        ("runs", "succeeded"),
        ("events", "running"),
    ),
)
def test_real_backend_workflow_response_timestamps(workflow_db, resource, status):
    db = workflow_db
    run_id = seed(db, status)
    if resource == "steps":
        response = WorkflowRunStepsResponse.model_validate(list_run_steps(db, run_id=run_id))
        projected = response.steps[0]
        raw = db.list_step_runs(run_id=run_id)[0]
        fields = ("started_at", "ended_at")
    elif resource == "attempts":
        response = WorkflowStepAttemptsResponse.model_validate(list_step_attempts(db, run_id=run_id, step_id="first"))
        projected = response.attempts[0]
        raw = db.list_step_attempts(run_id=run_id)[0]
        fields = ("started_at", "ended_at")
    elif resource == "investigation":
        response = WorkflowRunInvestigationResponse.model_validate(build_run_investigation(db, run_id=run_id))
        projected = response.failed_step
        raw = db.list_step_runs(run_id=run_id)[0]
        fields = ("started_at", "ended_at")
    elif resource == "runs":
        record = db.get_run(run_id)
        raw = {key: getattr(record, key) for key in ("created_at", "started_at", "ended_at")}
        response = WorkflowRunListItem(run_id=run_id, status=status, **raw)
        projected = response
        fields = tuple(raw)
    else:
        raw = db.get_events(run_id)[0]
        response = EventResponse(
            event_seq=raw["event_seq"],
            event_type=raw["event_type"],
            payload=raw.get("payload_json") or {},
            created_at=raw["created_at"],
        )
        projected = response
        fields = ("created_at",)
    for field in fields:
        value, stored = getattr(projected, field), raw[field]
        if isinstance(stored, datetime):
            assert datetime.fromisoformat(value) == stored  # nosec B101 -- exact stored instant survives public string projection.
        else:
            assert value == stored  # nosec B101 -- existing SQLite strings and unfinished nulls remain unchanged.
    raw_attempt = db.list_step_attempts(run_id=run_id)[0]
    assert raw_attempt["metadata_json"]["failure_envelope"]["private_marker"] == "operator-only-detail"  # nosec B101 -- prove the redaction sentinel exists before checking the public projection.
    assert "operator-only-detail" not in response.model_dump_json()  # nosec B101 -- timestamp conversion must retain private failure-detail redaction.


@pytest.mark.parametrize("value", (42, True, {}, []))
def test_workflow_timestamp_rejects_unrelated_types(value):
    with pytest.raises(ValidationError):
        EventResponse(event_seq=1, event_type="step_started", created_at=value)


@pytest.mark.parametrize(
    "model,fields",
    (
        (EventResponse, {"created_at": False}),
        (WorkflowRunListItem, {"created_at": True, "started_at": True, "ended_at": True}),
        (WorkflowRunStepResponse, {"started_at": True, "ended_at": True}),
        (WorkflowStepAttemptResponse, {"started_at": False, "ended_at": True}),
    ),
)
def test_workflow_timestamp_wire_schema_stays_string(model, fields):
    properties = model.model_json_schema()["properties"]
    for name, nullable in fields.items():
        field = properties[name]
        types = {branch.get("type") for branch in field.get("anyOf", [field])}
        assert types == ({"string", "null"} if nullable else {"string"})  # nosec B101 -- preserve all eight public string/null field contracts.


def test_workflow_timestamp_preserves_offset_and_precision():
    value = datetime(2026, 10, 21, 12, 3, 4, 5, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    event = EventResponse(event_seq=1, event_type="step_started", created_at=value)
    assert event.created_at == "2026-10-21T12:03:04.000005+05:30"  # nosec B101 -- retain the exact offset and microseconds at the API boundary.
