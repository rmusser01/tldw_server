from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import ScheduledTasksDatabase


def _repo(tmp_path, monkeypatch, *, user_id: int = 101) -> ScheduledTasksDatabase:
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    repo = ScheduledTasksDatabase.for_user(user_id=user_id)
    repo.ensure_schema()
    return repo


def _create_preview(
    repo: ScheduledTasksDatabase,
    *,
    owner_id: int = 101,
    family: str = "recurring_question",
    mode: str = "create",
    definition_id: str | None = None,
    definition_version: int | None = None,
    normalized_config: dict[str, Any] | None = None,
):
    return repo.create_preview(
        owner_id=owner_id,
        mode=mode,
        family=family,
        definition_id=definition_id,
        definition_version=definition_version,
        status="valid",
        payload_hash=f"hash-{owner_id}-{family}-{mode}",
        normalized_config=normalized_config
        or {
            "input": {"question": "What changed?"},
            "name": "Question",
            "schedule": {"cron": "0 9 * * *"},
        },
        validation_errors=[],
        warnings=[],
        risk_class=None,
        visibility_policy="findings_only",
        schedule_preview={"summary": "daily"},
        redaction_policy={"mode": "none"},
        expires_at="2026-06-10T00:00:00+00:00",
        created_by=str(owner_id),
    )


def _create_definition(
    repo: ScheduledTasksDatabase,
    *,
    owner_id: int = 101,
    family: str = "recurring_question",
    name: str = "Daily question",
    lifecycle: str = "configured",
    health: str = "execution_unavailable",
    description: str | None = "Ask the question every day",
    disabled_lock_kind: str = "none",
    disabled_reason: str | None = None,
):
    preview = _create_preview(repo, owner_id=owner_id, family=family)
    return repo.create_definition(
        owner_id=owner_id,
        family=family,
        name=name,
        description=description,
        lifecycle=lifecycle,
        health=health,
        disabled_lock_kind=disabled_lock_kind,
        disabled_reason=disabled_reason,
        schedule={"timezone": "UTC", "cron": "0 9 * * *"},
        input={"question": "What changed?"},
        visibility_policy="findings_only",
        notification_policy={"channels": []},
        approval_policy={"required": False},
        preview_id=preview.id,
        created_by=str(owner_id),
        updated_by=str(owner_id),
    )


def test_scheduled_tasks_repository_isolates_users(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    repo_a = ScheduledTasksDatabase.for_user(user_id=101)
    repo_b = ScheduledTasksDatabase.for_user(user_id=202)
    repo_a.ensure_schema()
    repo_b.ensure_schema()

    preview = repo_a.create_preview(
        owner_id=101,
        mode="create",
        family="recurring_question",
        definition_id=None,
        definition_version=None,
        status="valid",
        payload_hash="hash-a",
        normalized_config={"name": "Question", "input": {"question": "What changed?"}},
        validation_errors=[],
        warnings=[],
        risk_class=None,
        visibility_policy="findings_only",
        schedule_preview={"summary": "daily"},
        redaction_policy={"mode": "none"},
        expires_at="2026-06-10T00:00:00+00:00",
        created_by="101",
    )

    assert repo_a.get_preview(owner_id=101, preview_id=preview.id).id == preview.id  # nosec B101
    assert repo_b.get_preview(owner_id=202, preview_id=preview.id) is None  # nosec B101


def test_create_definition_and_audit_roundtrip(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    preview = _create_preview(
        repo,
        normalized_config={"z": "last", "a": {"nested": True}},
    )

    definition = repo.create_definition(
        owner_id=101,
        family="recurring_question",
        name="Daily research check",
        description="Ask about new research",
        lifecycle="configured",
        health="execution_unavailable",
        schedule={"b": 2, "a": 1},
        input={"question": "What changed?"},
        visibility_policy="findings_only",
        notification_policy={"channels": ["in_app"]},
        approval_policy={"required": False},
        preview_id=preview.id,
        created_by="101",
        updated_by="101",
    )
    audit = repo.create_audit_event(
        owner_id=101,
        definition_id=definition.id,
        event_type="definition.created",
        actor="101",
        summary="Created definition",
        before=None,
        after={"name": definition.name, "version": definition.version},
        request_id="req-1",
        idempotency_key="idem-1",
    )

    loaded = repo.get_definition(owner_id=101, definition_id=definition.id)
    events, total = repo.list_audit_events(
        owner_id=101,
        definition_id=definition.id,
        limit=10,
        offset=0,
    )
    db_bytes = repo.db_path.read_bytes()

    assert loaded == definition  # nosec B101
    assert audit.id == events[0].id  # nosec B101
    assert total == 1  # nosec B101
    assert events[0].after == {"name": definition.name, "version": 1}  # nosec B101
    assert b'{"a":1,"b":2}' in db_bytes  # nosec B101
    assert b'{"b":2,"a":1}' not in db_bytes  # nosec B101


def test_update_preview_consumption_sets_consumed_at(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    preview = _create_preview(repo)

    consumed = repo.mark_preview_consumed(
        owner_id=101,
        preview_id=preview.id,
        created_definition_id="definition-1",
    )

    assert consumed.id == preview.id  # nosec B101
    assert consumed.status == "consumed"  # nosec B101
    assert consumed.consumed_at is not None  # nosec B101
    assert consumed.created_definition_id == "definition-1"  # nosec B101


def test_idempotency_records_are_owner_and_route_scoped(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    first = repo.create_idempotency_record(
        owner_id=101,
        route="/api/v1/scheduled-tasks/previews",
        key="same-key",
        payload_hash="hash-1",
        response_ref={"preview_id": "preview-1"},
        expires_at="2026-06-10T00:00:00+00:00",
    )
    by_route = repo.create_idempotency_record(
        owner_id=101,
        route="/api/v1/scheduled-tasks/definitions",
        key="same-key",
        payload_hash="hash-2",
        response_ref={"definition_id": "definition-1"},
        expires_at="2026-06-10T00:00:00+00:00",
    )
    by_owner = repo.create_idempotency_record(
        owner_id=202,
        route="/api/v1/scheduled-tasks/previews",
        key="same-key",
        payload_hash="hash-3",
        response_ref={"preview_id": "preview-2"},
        expires_at="2026-06-10T00:00:00+00:00",
    )

    assert (
        repo.get_idempotency_record(  # nosec B101
            owner_id=101,
            route="/api/v1/scheduled-tasks/previews",
            key="same-key",
        )
        == first
    )
    assert (
        repo.get_idempotency_record(  # nosec B101
            owner_id=101,
            route="/api/v1/scheduled-tasks/definitions",
            key="same-key",
        )
        == by_route
    )
    assert (
        repo.get_idempotency_record(  # nosec B101
            owner_id=202,
            route="/api/v1/scheduled-tasks/previews",
            key="same-key",
        )
        == by_owner
    )

    with pytest.raises(sqlite3.IntegrityError):
        repo.create_idempotency_record(
            owner_id=101,
            route="/api/v1/scheduled-tasks/previews",
            key="same-key",
            payload_hash="hash-conflict",
            response_ref={"preview_id": "preview-conflict"},
            expires_at="2026-06-10T00:00:00+00:00",
        )


def test_list_definitions_filters_by_family_lifecycle_health_and_query(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    _create_definition(
        repo,
        family="recurring_question",
        name="Daily climate question",
        lifecycle="configured",
        health="execution_unavailable",
    )
    _create_definition(
        repo,
        family="agent_task",
        name="Weekly agent triage",
        lifecycle="paused",
        health="needs_attention",
    )
    _create_definition(
        repo,
        family="recurring_question",
        name="Weekly finance question",
        lifecycle="paused",
        health="ready",
    )
    _create_definition(
        repo,
        family="recurring_question",
        name="Archived question",
        lifecycle="archived",
        health="ready",
    )

    family_rows, family_total = repo.list_definitions(owner_id=101, family="recurring_question", limit=10, offset=0)
    lifecycle_rows, lifecycle_total = repo.list_definitions(owner_id=101, lifecycle="paused", limit=10, offset=0)
    health_rows, health_total = repo.list_definitions(owner_id=101, health="ready", limit=10, offset=0)
    query_rows, query_total = repo.list_definitions(owner_id=101, query="climate", limit=10, offset=0)
    page_rows, page_total = repo.list_definitions(owner_id=101, limit=2, offset=1)

    assert family_total == 3  # nosec B101
    assert {row.family for row in family_rows} == {"recurring_question"}  # nosec B101
    assert lifecycle_total == 2  # nosec B101
    assert {row.lifecycle for row in lifecycle_rows} == {"paused"}  # nosec B101
    assert health_total == 2  # nosec B101
    assert {row.health for row in health_rows} == {"ready"}  # nosec B101
    assert query_total == 1  # nosec B101
    assert query_rows[0].name == "Daily climate question"  # nosec B101
    assert page_total == 4  # nosec B101
    assert len(page_rows) == 2  # nosec B101


def test_definition_persists_disabled_lock_kind_and_reason(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)

    definition = _create_definition(
        repo,
        lifecycle="disabled",
        health="capability_unavailable",
        disabled_lock_kind="security",
        disabled_reason="Agent capability was disabled by policy",
    )

    loaded = repo.get_definition(owner_id=101, definition_id=definition.id)
    assert loaded.disabled_lock_kind == "security"  # nosec B101
    assert loaded.disabled_reason == "Agent capability was disabled by policy"  # nosec B101


def test_agent_task_definition_and_audit_storage_do_not_contain_raw_message_secret(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    sentinel = "RAW_AGENT_TASK_SENTINEL_SHOULD_NOT_PERSIST"
    raw_requested_payload = {"input": {"message": sentinel}}
    redacted_input = {"message_ref": "msg_123", "message_preview": "[redacted]"}
    preview = _create_preview(
        repo,
        family="agent_task",
        normalized_config={
            "family": "agent_task",
            "input": redacted_input,
            "redaction_policy": {"fields": ["input.message"]},
        },
    )
    definition = repo.create_definition(
        owner_id=101,
        family="agent_task",
        name="Agent follow-up",
        description=None,
        lifecycle="paused",
        health="execution_unavailable",
        schedule={"cron": "0 12 * * *", "timezone": "UTC"},
        input=redacted_input,
        visibility_policy="findings_only",
        notification_policy={"channels": []},
        approval_policy={"required": True},
        preview_id=preview.id,
        created_by="101",
        updated_by="101",
    )
    repo.create_audit_event(
        owner_id=101,
        definition_id=definition.id,
        event_type="definition.created",
        actor="101",
        summary="Created redacted Agent Task definition",
        before=None,
        after={"input": redacted_input, "source_payload_keys": list(raw_requested_payload.keys())},
        request_id=None,
        idempotency_key=None,
    )

    preview_payload = asdict(repo.get_preview(owner_id=101, preview_id=preview.id))
    definition_payload = asdict(repo.get_definition(owner_id=101, definition_id=definition.id))
    audit_payloads = [
        asdict(row) for row in repo.list_audit_events(owner_id=101, definition_id=definition.id, limit=10, offset=0)[0]
    ]
    persisted_text = json.dumps([preview_payload, definition_payload, audit_payloads], sort_keys=True)

    assert sentinel not in persisted_text  # nosec B101
    assert sentinel.encode("utf-8") not in repo.db_path.read_bytes()  # nosec B101
