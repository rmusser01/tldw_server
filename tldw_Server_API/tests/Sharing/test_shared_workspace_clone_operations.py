"""Contracts for recipient-facing shared Workspace clone operations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.shared_workspace_recipient_schemas import (
    SharedWorkspaceCloneOperationResponse,
    SharedWorkspaceCloneResult,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_clone_operations import (
    CLONE_COMMAND,
    CLONE_DOMAIN,
    CLONE_JOB_TYPE,
    CLONE_QUEUE,
    CloneOperationNotFound,
    CloneOperationUnavailable,
    build_clone_admission_command,
    normalize_clone_name,
    project_clone_operation,
    target_workspace_id,
    validate_idempotency_key,
)

pytestmark = pytest.mark.unit


def _job(**overrides):
    values = {
        "id": 17,
        "uuid": "de305d54-75b4-431b-adb2-eb6b9e546014",
        "domain": CLONE_DOMAIN,
        "queue": CLONE_QUEUE,
        "job_type": CLONE_JOB_TYPE,
        "owner_user_id": "9",
        "batch_group": "share:42",
        "status": "queued",
        "payload": {
            "schema_version": 1,
            "share_id": 42,
            "recipient_user_id": 9,
            "requested_name": "Evidence Copy",
            "request_fingerprint": "b" * 64,
        },
        "result": None,
        "created_at": "2026-08-25T10:00:00+00:00",
        "updated_at": "2026-08-25T10:00:01+00:00",
        "progress_percent": None,
        "progress_message": None,
        "error_code": None,
        "error_message": None,
        "last_error": None,
        "archived": False,
    }
    values.update(overrides)
    return values


def _successful_result(*, publication_confirmed: bool = True):
    return {
        "schema_version": 1,
        "outcome": "partial",
        "workspace_id": target_workspace_id(
            "de305d54-75b4-431b-adb2-eb6b9e546014"
        ),
        "name": "Evidence Copy",
        "publication_confirmed": publication_confirmed,
        "counts": {
            "sources_attempted": 3,
            "sources_copied": 2,
            "sources_failed": 1,
            "notes_attempted": 1,
            "notes_copied": 1,
            "notes_failed": 0,
            "artifacts_attempted": 0,
            "artifacts_copied": 0,
            "artifacts_failed": 0,
            "media_attempted": 2,
            "media_copied": 2,
            "media_failed": 0,
            "operation_owned_media_count": 2,
        },
        "readiness": {
            "text_search": "ready",
            "citations": "ready",
            "vector_search": "needs_indexing",
        },
        "warnings": [{"code": "vector_index_not_generated", "count": 2}],
    }


@pytest.mark.parametrize(
    "value",
    (
        "a" * 16,
        "A0._~-recipient-key",
        "z" * 200,
    ),
)
def test_idempotency_key_accepts_only_exact_bounded_wire_values(value: str) -> None:
    assert validate_idempotency_key(value) == value


@pytest.mark.parametrize(
    "value",
    (
        "a" * 15,
        "a" * 201,
        " leading-key-value",
        "trailing-key-value ",
        "contains/slash-key",
        "contains:colon-key",
        "contains-unicode-N{SNOWMAN}",
    ),
)
def test_idempotency_key_rejects_invalid_or_normalizable_values(value: str) -> None:
    with pytest.raises(ValueError, match="Idempotency-Key"):
        validate_idempotency_key(value)


def test_clone_name_is_optional_and_normalized_once() -> None:
    assert normalize_clone_name(None) is None
    assert normalize_clone_name("  Evidence\n   Copy  ") == "Evidence Copy"
    with pytest.raises(ValueError, match="name"):
        normalize_clone_name(" \t ")
    with pytest.raises(ValueError, match="name"):
        normalize_clone_name("x" * 256)


def test_admission_command_is_bounded_and_never_persists_raw_key() -> None:
    now = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)

    first = build_clone_admission_command(
        share_id=42,
        recipient_user_id=9,
        requested_name="  Evidence Copy ",
        idempotency_key="recipient-key-0001",
        now=now,
    )
    replay = build_clone_admission_command(
        share_id=42,
        recipient_user_id=9,
        requested_name="Evidence Copy",
        idempotency_key="recipient-key-0001",
        now=now,
    )

    assert first == replay
    assert first.job.domain == CLONE_DOMAIN
    assert first.job.queue == CLONE_QUEUE
    assert first.job.job_type == CLONE_JOB_TYPE
    assert first.job.owner_user_id == "9"
    assert first.job.batch_group == first.operation_scope == "share:42"
    assert first.job.max_retries == 0
    assert first.job.priority == 10
    assert first.job.idempotency_key is None
    assert first.job.payload == {
        "schema_version": 1,
        "share_id": 42,
        "recipient_user_id": 9,
        "requested_name": "Evidence Copy",
        "request_fingerprint": first.request_fingerprint,
    }
    assert len(first.key_digest) == len(first.request_fingerprint) == 64
    assert "recipient-key-0001" not in repr(first)
    assert (first.receipt_expires_at - now).days == 31


def test_admission_fingerprint_changes_with_semantic_request_identity() -> None:
    now = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)

    baseline = build_clone_admission_command(
        share_id=42,
        recipient_user_id=9,
        requested_name=None,
        idempotency_key="recipient-key-0001",
        now=now,
    )
    renamed = build_clone_admission_command(
        share_id=42,
        recipient_user_id=9,
        requested_name="Copy",
        idempotency_key="recipient-key-0001",
        now=now,
    )

    assert baseline.request_fingerprint != renamed.request_fingerprint
    assert baseline.key_digest == renamed.key_digest


def test_target_workspace_identity_is_stable_uuid_and_operation_specific() -> None:
    operation_id = "de305d54-75b4-431b-adb2-eb6b9e546014"

    target = target_workspace_id(operation_id)

    assert target == target_workspace_id(operation_id)
    assert UUID(target).version == 5
    assert target != target_workspace_id("de305d54-75b4-431b-adb2-eb6b9e546015")


def test_queued_projection_contains_only_canonical_workspace_fields() -> None:
    job = _job(
        diagnostics={"raw_path": "/owner/private.db"},
    )

    response = project_clone_operation(job, share_id=42, recipient_user_id=9)

    assert response == SharedWorkspaceCloneOperationResponse(
        schema_version=1,
        operation_id=job["uuid"],
        workspace_id=target_workspace_id(job["uuid"]),
        command=CLONE_COMMAND,
        status="queued",
        started_at=job["created_at"],
        updated_at=job["updated_at"],
        retryable=False,
        diagnostics={},
        poll_href=f"/api/v1/sharing/shared-with-me/42/clone/{job['uuid']}",
        share_id=42,
        progress={"phase": "queued", "percent": 0, "message_code": "clone_queued"},
        result=None,
        error=None,
    )
    assert "raw_path" not in response.model_dump_json()


def test_unknown_job_payload_fields_fail_closed() -> None:
    with pytest.raises(CloneOperationUnavailable):
        project_clone_operation(
            _job(payload={**_job()["payload"], "source_title": "must not escape"}),
            share_id=42,
            recipient_user_id=9,
        )


def test_processing_projection_uses_only_typed_progress() -> None:
    response = project_clone_operation(
        _job(
            status="processing",
            progress_percent=44.6,
            progress_message="sources",
        ),
        share_id=42,
        recipient_user_id=9,
    )

    assert response.status == "running"
    assert response.progress is not None
    assert response.progress.model_dump() == {
        "phase": "sources",
        "percent": 45,
        "message_code": "clone_sources",
    }


def test_completed_projection_requires_valid_confirmed_result() -> None:
    response = project_clone_operation(
        _job(status="completed", result=_successful_result(), archived=True),
        share_id=42,
        recipient_user_id=9,
    )

    assert response.status == "succeeded"
    assert response.progress is None
    assert response.result == SharedWorkspaceCloneResult.model_validate(
        _successful_result()
    )
    assert response.error is None
    assert response.diagnostics == {}


@pytest.mark.parametrize(
    "result",
    (
        None,
        {},
        _successful_result(publication_confirmed=False),
        {**_successful_result(), "private_path": "/owner/private.db"},
        {**_successful_result(), "workspace_id": "wrong-target"},
    ),
)
def test_completed_projection_fails_closed_on_malformed_terminal_result(result) -> None:
    with pytest.raises(CloneOperationUnavailable):
        project_clone_operation(
            _job(status="completed", result=result),
            share_id=42,
            recipient_user_id=9,
        )


def test_failed_projection_uses_stable_error_and_never_echoes_backend_text() -> None:
    response = project_clone_operation(
        _job(
            status="failed",
            error_code="source_snapshot_unavailable",
            error_message="/owner/private.db: password=hunter2",
            last_error="traceback with source title",
            result={"schema_version": 1, "cleanup_state": "complete"},
        ),
        share_id=42,
        recipient_user_id=9,
    )

    assert response.status == "failed"
    assert response.retryable is True
    assert response.result is None
    assert response.error is not None
    assert response.error.model_dump() == {
        "code": "source_snapshot_unavailable",
        "message_key": "sharing.clone.errors.source_snapshot_unavailable",
        "message": "The source snapshot was unavailable. Try the copy again.",
        "cleanup_state": "complete",
    }
    serialized = response.model_dump_json()
    assert "private.db" not in serialized
    assert "hunter2" not in serialized
    assert "source title" not in serialized


@pytest.mark.parametrize("status", ("cancelled", "quarantined"))
def test_cancelled_and_quarantined_jobs_map_to_failed(status: str) -> None:
    response = project_clone_operation(
        _job(status=status, error_code="clone_interrupted"),
        share_id=42,
        recipient_user_id=9,
    )

    assert response.status == "failed"
    assert response.error is not None
    assert response.error.cleanup_state == "unknown"


@pytest.mark.parametrize(
    "overrides",
    (
        {"owner_user_id": "10"},
        {"domain": "research"},
        {"queue": "default"},
        {"job_type": "other"},
        {"batch_group": "share:41"},
        {"payload": {**_job()["payload"], "share_id": 41}},
        {"payload": {**_job()["payload"], "recipient_user_id": 10}},
    ),
)
def test_wrong_owner_or_operation_scope_is_neutral_not_found(overrides) -> None:
    with pytest.raises(CloneOperationNotFound):
        project_clone_operation(
            _job(**overrides),
            share_id=42,
            recipient_user_id=9,
        )


@pytest.mark.parametrize("status", ("paused", "unknown", ""))
def test_unknown_job_status_fails_closed(status: str) -> None:
    with pytest.raises(CloneOperationUnavailable):
        project_clone_operation(
            _job(status=status),
            share_id=42,
            recipient_user_id=9,
        )


def test_clone_response_forbids_extras_and_enforces_state_shape() -> None:
    valid = project_clone_operation(_job(), share_id=42, recipient_user_id=9)
    payload = valid.model_dump()

    with pytest.raises(ValidationError, match="extra_forbidden"):
        SharedWorkspaceCloneOperationResponse.model_validate(
            {**payload, "job_id": 17}
        )
    with pytest.raises(ValidationError, match="queued"):
        SharedWorkspaceCloneOperationResponse.model_validate(
            {**payload, "result": _successful_result()}
        )
