from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import ScheduledTasksDatabase


def _repo(tmp_path, monkeypatch, *, user_id: int = 101) -> ScheduledTasksDatabase:
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    repo = ScheduledTasksDatabase.for_user(user_id=user_id)
    repo.ensure_schema()
    return repo


def _database_bytes(repo: ScheduledTasksDatabase) -> bytes:
    payload = repo.db_path.read_bytes()
    wal_path = Path(f"{repo.db_path}-wal")
    if wal_path.exists():
        payload += wal_path.read_bytes()
    return payload


def _future_expires_at() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()


def _create_preview(
    repo: ScheduledTasksDatabase,
    *,
    owner_id: int = 101,
    family: str = "recurring_question",
    mode: str = "create",
    definition_id: str | None = None,
    definition_version: int | None = None,
    status: str = "valid",
    expires_at: str | None = None,
    normalized_config: dict[str, Any] | None = None,
):
    return repo.create_preview(
        owner_id=owner_id,
        mode=mode,
        family=family,
        definition_id=definition_id,
        definition_version=definition_version,
        status=status,
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
        expires_at=expires_at or _future_expires_at(),
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


def test_connect_context_manager_closes_connection(tmp_path):
    repo = ScheduledTasksDatabase(tmp_path / "scheduled_tasks.db")

    with repo._connect() as conn:
        conn.execute("SELECT 1")

    try:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
    finally:
        conn.close()


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
        expires_at=_future_expires_at(),
        created_by="101",
    )

    assert repo_a.get_preview(owner_id=101, preview_id=preview.id).id == preview.id  # nosec B101
    assert repo_b.get_preview(owner_id=202, preview_id=preview.id) is None  # nosec B101


def test_bulk_preview_lookup_returns_owner_scoped_preview_map(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    preview_a = _create_preview(repo, owner_id=101, normalized_config={"config": {"rank": 1}})
    preview_b = _create_preview(repo, owner_id=101, normalized_config={"config": {"rank": 2}})
    other_owner_preview = _create_preview(repo, owner_id=202, normalized_config={"config": {"rank": 3}})

    previews = repo.get_previews_by_ids(
        owner_id=101,
        preview_ids=[preview_a.id, preview_b.id, preview_a.id, other_owner_preview.id, "missing"],
    )

    assert set(previews) == {preview_a.id, preview_b.id}  # nosec B101
    assert previews[preview_a.id].normalized_config["config"] == {"rank": 1}  # nosec B101
    assert previews[preview_b.id].normalized_config["config"] == {"rank": 2}  # nosec B101


def test_time_expired_valid_previews_read_and_filter_as_expired(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    expired = _create_preview(
        repo,
        expires_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    )
    active = _create_preview(repo)

    detail = repo.get_preview(owner_id=101, preview_id=expired.id)
    expired_rows, expired_total = repo.list_previews(owner_id=101, status="expired")
    valid_rows, valid_total = repo.list_previews(owner_id=101, status="valid")

    assert detail is not None  # nosec B101
    assert detail.status == "expired"  # nosec B101
    assert expired_total == 1  # nosec B101
    assert [row.id for row in expired_rows] == [expired.id]  # nosec B101
    assert valid_total == 1  # nosec B101
    assert [row.id for row in valid_rows] == [active.id]  # nosec B101


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
    db_bytes = _database_bytes(repo)

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


def test_mark_preview_consumed_rejects_second_consume_and_preserves_definition_id(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    preview = _create_preview(repo)
    first = repo.mark_preview_consumed(
        owner_id=101,
        preview_id=preview.id,
        created_definition_id="definition-1",
    )

    with pytest.raises(ValueError, match="preview already consumed"):
        repo.mark_preview_consumed(
            owner_id=101,
            preview_id=preview.id,
            created_definition_id="definition-2",
        )

    loaded = repo.get_preview(owner_id=101, preview_id=preview.id)
    assert loaded.consumed_at == first.consumed_at  # nosec B101
    assert loaded.created_definition_id == "definition-1"  # nosec B101


def test_idempotency_records_are_owner_and_route_scoped(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    first = repo.create_idempotency_record(
        owner_id=101,
        route="/api/v1/scheduled-tasks/previews",
        key="same-key",
        payload_hash="hash-1",
        response_ref={"preview_id": "preview-1"},
        expires_at="2099-01-01T00:00:00+00:00",
    )
    by_route = repo.create_idempotency_record(
        owner_id=101,
        route="/api/v1/scheduled-tasks/definitions",
        key="same-key",
        payload_hash="hash-2",
        response_ref={"definition_id": "definition-1"},
        expires_at="2099-01-01T00:00:00+00:00",
    )
    by_owner = repo.create_idempotency_record(
        owner_id=202,
        route="/api/v1/scheduled-tasks/previews",
        key="same-key",
        payload_hash="hash-3",
        response_ref={"preview_id": "preview-2"},
        expires_at="2099-01-01T00:00:00+00:00",
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
            expires_at="2099-01-01T00:00:00+00:00",
        )


def test_expired_idempotency_record_lookup_returns_none_and_key_can_be_reused(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    repo.create_idempotency_record(
        owner_id=101,
        route="/api/v1/scheduled-tasks/previews",
        key="reusable-key",
        payload_hash="old-hash",
        response_ref={"preview_id": "old-preview"},
        expires_at="2020-01-01T00:00:00+00:00",
    )

    assert (  # nosec B101
        repo.get_idempotency_record(
            owner_id=101,
            route="/api/v1/scheduled-tasks/previews",
            key="reusable-key",
        )
        is None
    )
    replacement = repo.create_idempotency_record(
        owner_id=101,
        route="/api/v1/scheduled-tasks/previews",
        key="reusable-key",
        payload_hash="new-hash",
        response_ref={"preview_id": "new-preview"},
        expires_at="2099-01-01T00:00:00+00:00",
    )

    assert replacement.payload_hash == "new-hash"  # nosec B101
    assert replacement.response_ref == {"preview_id": "new-preview"}  # nosec B101


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


def test_update_definition_with_expected_version_updates_and_conflicts_on_wrong_version(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    definition = _create_definition(repo, name="Original")

    updated = repo.update_definition(
        owner_id=101,
        definition_id=definition.id,
        patch={"name": "Updated", "updated_by": "101"},
        expected_version=1,
    )

    assert updated.version == 2  # nosec B101
    assert updated.name == "Updated"  # nosec B101
    with pytest.raises(ValueError, match="definition version conflict"):
        repo.update_definition(
            owner_id=101,
            definition_id=definition.id,
            patch={"name": "Stale update", "updated_by": "101"},
            expected_version=1,
        )


def test_update_definition_expected_version_conflicts_if_row_changes_between_read_and_write(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    definition = _create_definition(repo, name="Original")
    concurrent_repo = ScheduledTasksDatabase(repo.db_path)
    original_get_definition = repo.get_definition
    raced = False

    def _get_definition_after_concurrent_update(owner_id: int, definition_id: str):
        nonlocal raced
        row = original_get_definition(owner_id=owner_id, definition_id=definition_id)
        if row is not None and not raced:
            raced = True
            concurrent_repo.update_definition(
                owner_id=owner_id,
                definition_id=definition_id,
                patch={
                    "name": "Concurrent update",
                    "updated_by": "202",
                },
                expected_version=None,
            )
        return row

    monkeypatch.setattr(repo, "get_definition", _get_definition_after_concurrent_update)

    with pytest.raises(ValueError, match="definition version conflict"):
        repo.update_definition(
            owner_id=101,
            definition_id=definition.id,
            patch={"name": "Stale update", "updated_by": "101"},
            expected_version=1,
        )

    loaded = original_get_definition(owner_id=101, definition_id=definition.id)
    assert loaded.version == 2  # nosec B101
    assert loaded.name == "Concurrent update"  # nosec B101


def test_create_definition_rejects_missing_preview(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)

    with pytest.raises(KeyError, match="preview not found"):
        repo.create_definition(
            owner_id=101,
            family="recurring_question",
            name="Missing preview",
            description=None,
            lifecycle="configured",
            health="execution_unavailable",
            schedule={"cron": "0 9 * * *"},
            input={"question": "What changed?"},
            visibility_policy="findings_only",
            notification_policy={"channels": []},
            approval_policy={"required": False},
            preview_id="missing-preview",
            created_by="101",
            updated_by="101",
        )


def test_create_definition_rejects_cross_owner_preview(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    owner_202_preview = _create_preview(repo, owner_id=202)

    with pytest.raises(KeyError, match="preview not found"):
        repo.create_definition(
            owner_id=101,
            family="recurring_question",
            name="Cross-owner preview",
            description=None,
            lifecycle="configured",
            health="execution_unavailable",
            schedule={"cron": "0 9 * * *"},
            input={"question": "What changed?"},
            visibility_policy="findings_only",
            notification_policy={"channels": []},
            approval_policy={"required": False},
            preview_id=owner_202_preview.id,
            created_by="101",
            updated_by="101",
        )


def test_create_definition_consumes_preview_and_rejects_second_definition(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    preview = _create_preview(repo)
    first_definition = repo.create_definition(
        owner_id=101,
        family="recurring_question",
        name="First definition",
        description=None,
        lifecycle="configured",
        health="execution_unavailable",
        schedule={"cron": "0 9 * * *"},
        input={"question": "What changed?"},
        visibility_policy="findings_only",
        notification_policy={"channels": []},
        approval_policy={"required": False},
        preview_id=preview.id,
        created_by="101",
        updated_by="101",
    )

    with pytest.raises(ValueError, match="preview already consumed"):
        repo.create_definition(
            owner_id=101,
            family="recurring_question",
            name="Second definition",
            description=None,
            lifecycle="configured",
            health="execution_unavailable",
            schedule={"cron": "0 9 * * *"},
            input={"question": "What changed?"},
            visibility_policy="findings_only",
            notification_policy={"channels": []},
            approval_policy={"required": False},
            preview_id=preview.id,
            created_by="101",
            updated_by="101",
        )

    loaded_preview = repo.get_preview(owner_id=101, preview_id=preview.id)
    assert loaded_preview.status == "consumed"  # nosec B101
    assert loaded_preview.created_definition_id == first_definition.id  # nosec B101
    definitions, total = repo.list_definitions(owner_id=101, limit=10, offset=0)
    assert total == 1  # nosec B101
    assert definitions[0].id == first_definition.id  # nosec B101


def test_update_definition_rejects_missing_preview_patch(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    definition = _create_definition(repo)

    with pytest.raises(KeyError, match="preview not found"):
        repo.update_definition(
            owner_id=101,
            definition_id=definition.id,
            patch={
                "preview_id": "missing-preview",
                "updated_by": "101",
            },
            expected_version=1,
        )


def test_update_definition_rejects_cross_owner_preview_patch(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    definition = _create_definition(repo)
    owner_202_preview = _create_preview(
        repo,
        owner_id=202,
        mode="update",
        definition_id=definition.id,
        definition_version=definition.version,
    )

    with pytest.raises(KeyError, match="preview not found"):
        repo.update_definition(
            owner_id=101,
            definition_id=definition.id,
            patch={
                "preview_id": owner_202_preview.id,
                "updated_by": "101",
            },
            expected_version=1,
        )


def test_create_audit_event_rejects_missing_definition(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)

    with pytest.raises(KeyError, match="definition not found"):
        repo.create_audit_event(
            owner_id=101,
            definition_id="missing-definition",
            event_type="definition.created",
            actor="101",
            summary="Should fail",
            before=None,
            after={"name": "Missing"},
        )


def test_create_audit_event_rejects_cross_owner_definition(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    owner_202_definition = _create_definition(
        repo,
        owner_id=202,
        name="Other owner definition",
    )

    with pytest.raises(KeyError, match="definition not found"):
        repo.create_audit_event(
            owner_id=101,
            definition_id=owner_202_definition.id,
            event_type="definition.updated",
            actor="101",
            summary="Should fail",
            before=None,
            after={"name": "Other owner definition"},
        )


def test_json_persistence_rejects_nan_values(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        _create_preview(
            repo,
            normalized_config={"threshold": math.nan},
        )


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
    assert sentinel.encode("utf-8") not in _database_bytes(repo)  # nosec B101
