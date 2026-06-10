from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_automation_schemas import (
    ScheduledTaskDefinitionCreateRequest,
    ScheduledTaskDefinitionUpdateRequest,
    ScheduledTaskDuplicateRequest,
    ScheduledTaskPreviewCreateRequest,
)
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import ScheduledTasksDatabase
from tldw_Server_API.app.services.scheduled_task_automation_service import (
    ScheduledTaskAutomationError,
    ScheduledTaskAutomationService,
)

OWNER_ID = 4101
OTHER_OWNER_ID = 4102
ACTOR = "task-service-test"
RAW_SENTINEL = "RAW_AGENT_SECRET_DO_NOT_LEAK_4B"


def _service(tmp_path: Path) -> tuple[ScheduledTaskAutomationService, ScheduledTasksDatabase]:
    repo = ScheduledTasksDatabase(tmp_path / "scheduled_tasks_service.db")
    repo.ensure_schema()
    return ScheduledTaskAutomationService(repository=repo), repo


def _payload(
    *,
    family: str = "recurring_question",
    mode: str = "create",
    definition_id: str | None = None,
    definition_version: int | None = None,
    name: str = "Daily research check",
    config_payload: dict[str, Any] | None = None,
    input_payload: dict[str, Any] | None = None,
    schedule: dict[str, Any] | None = None,
) -> ScheduledTaskPreviewCreateRequest:
    if input_payload is None:
        input_payload = (
            {"question": "What changed in the selected sources?"}
            if family == "recurring_question"
            else {"agent_ref": "agent:triage", "message": "Summarize high priority changes."}
        )
    return ScheduledTaskPreviewCreateRequest(
        mode=mode,
        family=family,
        definition_id=definition_id,
        definition_version=definition_version,
        name=name,
        description="Service lifecycle test",
        config=config_payload or {},
        input=input_payload,
        schedule=schedule or {"kind": "daily", "time": "09:00", "timezone": "UTC"},
        visibility_policy={"mode": "findings_only"},
        notification_policy={"channels": ["in_app"]},
        approval_policy={"required": False},
    )


def _create_definition(
    service: ScheduledTaskAutomationService,
    *,
    owner_id: int = OWNER_ID,
    name: str = "Daily research check",
    initial_lifecycle: str = "configured",
):
    preview = service.create_preview(
        owner_id=owner_id,
        actor=ACTOR,
        payload=_payload(name=name),
    )
    return service.create_definition(
        owner_id=owner_id,
        actor=ACTOR,
        payload=ScheduledTaskDefinitionCreateRequest(
            preview_id=preview.id,
            initial_lifecycle=initial_lifecycle,
        ),
    )


def _as_json_text(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True)


def _database_bytes(repo: ScheduledTasksDatabase) -> bytes:
    payload = repo.db_path.read_bytes()
    wal_path = Path(f"{repo.db_path}-wal")
    if wal_path.exists():
        payload += wal_path.read_bytes()
    return payload


def _set_preview_expires_at(repo: ScheduledTasksDatabase, *, preview_id: str, expires_at: str) -> None:
    with sqlite3.connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE scheduled_task_previews SET expires_at = ? WHERE id = ?",
            [expires_at, preview_id],
        )


def _audit_count(repo: ScheduledTasksDatabase, definition_id: str) -> int:
    return repo.list_audit_events(
        owner_id=OWNER_ID,
        definition_id=definition_id,
        limit=100,
        offset=0,
    )[1]


def _install_idempotency_miss_barrier(
    monkeypatch: pytest.MonkeyPatch,
    repo: ScheduledTasksDatabase,
    *,
    parties: int = 2,
) -> None:
    original_get_idempotency_record = repo.get_idempotency_record
    barrier = threading.Barrier(parties)
    lock = threading.Lock()
    waits_remaining = parties

    def _raced_get_idempotency_record(owner_id: int, route: str, key: str):
        nonlocal waits_remaining
        row = original_get_idempotency_record(owner_id, route, key)
        should_wait = False
        if row is None:
            with lock:
                if waits_remaining > 0:
                    waits_remaining -= 1
                    should_wait = True
        if should_wait:
            barrier.wait(timeout=5)
        return row

    monkeypatch.setattr(repo, "get_idempotency_record", _raced_get_idempotency_record)


def test_invalid_semantic_preview_is_persisted_with_errors_and_id(tmp_path):
    service, repo = _service(tmp_path)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(input_payload={"question": "   "}, schedule={"kind": "not-a-schedule"}),
    )

    stored = repo.get_preview(owner_id=OWNER_ID, preview_id=preview.id)
    assert preview.id  # nosec B101
    assert preview.status == "invalid"  # nosec B101
    assert preview.validation_errors  # nosec B101
    assert stored is not None  # nosec B101
    assert stored.status == "invalid"  # nosec B101
    assert stored.validation_errors == preview.validation_errors  # nosec B101


def test_non_object_schedule_preview_is_persisted_as_invalid(tmp_path):
    service, repo = _service(tmp_path)
    payload = _payload()
    raw_payload = payload.model_dump(mode="json")
    raw_payload["schedule"] = ["not", "an", "object"]
    constructed_payload = ScheduledTaskPreviewCreateRequest.model_construct(**raw_payload)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=constructed_payload,
    )

    stored = repo.get_preview(owner_id=OWNER_ID, preview_id=preview.id)
    assert preview.status == "invalid"  # nosec B101
    assert preview.validation_errors == [
        {"field": "schedule", "code": "invalid_type", "message": "Schedule must be an object."}
    ]  # nosec B101
    assert stored is not None  # nosec B101
    assert stored.validation_errors == preview.validation_errors  # nosec B101


def test_preview_mode_linkage_invariants_are_persisted_as_invalid(tmp_path):
    service, _repo = _service(tmp_path)

    update_without_definition = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(mode="update", definition_id=None, definition_version=None),
    )
    create_with_definition_linkage = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(mode="create", definition_id="definition_1", definition_version=1),
    )

    assert update_without_definition.status == "invalid"  # nosec B101
    assert {
        (error["field"], error["code"]) for error in update_without_definition.validation_errors
    } >= {
        ("definition_id", "required_for_update"),
        ("definition_version", "required_for_update"),
    }  # nosec B101
    assert create_with_definition_linkage.status == "invalid"  # nosec B101
    assert {
        (error["field"], error["code"]) for error in create_with_definition_linkage.validation_errors
    } >= {
        ("definition_id", "not_allowed_for_create"),
        ("definition_version", "not_allowed_for_create"),
    }  # nosec B101


def test_agent_task_preview_redacts_raw_message_in_responses_and_storage(tmp_path):
    service, repo = _service(tmp_path)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            family="agent_task",
            input_payload={"agent_ref": "agent:incident-triage", "message": RAW_SENTINEL},
        ),
    )

    stored = repo.get_preview(owner_id=OWNER_ID, preview_id=preview.id)
    assert preview.status == "valid"  # nosec B101
    assert preview.normalized_config["input"]["message_redacted"] is True  # nosec B101
    assert "message_length" not in preview.normalized_config["input"]  # nosec B101
    assert not preview.normalized_config["input"]["message_ref"].startswith("sha256:")  # nosec B101
    assert RAW_SENTINEL not in _as_json_text(preview)  # nosec B101
    assert stored is not None  # nosec B101
    assert RAW_SENTINEL not in json.dumps(stored.normalized_config, sort_keys=True)  # nosec B101
    assert RAW_SENTINEL.encode("utf-8") not in _database_bytes(repo)  # nosec B101


def test_create_consumes_valid_preview_and_rejects_consumed_reuse_without_idempotency_key(tmp_path):
    service, repo = _service(tmp_path)
    preview = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=_payload())

    definition = service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=ScheduledTaskDefinitionCreateRequest(preview_id=preview.id),
    )

    consumed = repo.get_preview(owner_id=OWNER_ID, preview_id=preview.id)
    assert definition.preview_id == preview.id  # nosec B101
    assert definition.health == "execution_unavailable"  # nosec B101
    assert consumed.status == "consumed"  # nosec B101
    with pytest.raises(ScheduledTaskAutomationError, match="preview_consumed"):
        service.create_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=ScheduledTaskDefinitionCreateRequest(preview_id=preview.id),
        )


def test_preview_validation_rejects_expired_stale_consumed_and_cross_user_previews(tmp_path):
    service, repo = _service(tmp_path)
    expired = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=_payload())
    _set_preview_expires_at(repo, preview_id=expired.id, expires_at="2020-01-01T00:00:00+00:00")

    with pytest.raises(ScheduledTaskAutomationError, match="preview_expired"):
        service.create_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=ScheduledTaskDefinitionCreateRequest(preview_id=expired.id),
        )

    definition = _create_definition(service)
    stale_preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            mode="update",
            definition_id=definition.id,
            definition_version=definition.version,
            name="Updated name",
        ),
    )
    service.pause_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id)

    with pytest.raises(ScheduledTaskAutomationError, match="definition_version_mismatch"):
        service.update_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=definition.id,
            payload=ScheduledTaskDefinitionUpdateRequest(preview_id=stale_preview.id),
        )

    consumed_preview = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=_payload(name="Consumed"))
    service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=ScheduledTaskDefinitionCreateRequest(preview_id=consumed_preview.id),
    )
    with pytest.raises(ScheduledTaskAutomationError, match="preview_consumed"):
        service.create_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=ScheduledTaskDefinitionCreateRequest(preview_id=consumed_preview.id),
        )

    other_preview = service.create_preview(owner_id=OTHER_OWNER_ID, actor=ACTOR, payload=_payload())
    with pytest.raises(ScheduledTaskAutomationError, match="preview_not_found"):
        service.create_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=ScheduledTaskDefinitionCreateRequest(preview_id=other_preview.id),
        )


def test_update_requires_preview_version_match_and_consumes_preview(tmp_path):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)
    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            mode="update",
            definition_id=definition.id,
            definition_version=definition.version,
            name="Renamed question",
        ),
    )

    updated = service.update_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=ScheduledTaskDefinitionUpdateRequest(preview_id=preview.id),
    )

    consumed = repo.get_preview(owner_id=OWNER_ID, preview_id=preview.id)
    assert updated.version == definition.version + 1  # nosec B101
    assert updated.name == "Renamed question"  # nosec B101
    assert consumed.status == "consumed"  # nosec B101


def test_duplicate_creates_paused_copy_and_two_audit_events(tmp_path):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)

    copy = service.duplicate_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=ScheduledTaskDuplicateRequest(name="Paused copy"),
    )

    original_events, original_total = repo.list_audit_events(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        limit=10,
        offset=0,
    )
    copy_events, copy_total = repo.list_audit_events(
        owner_id=OWNER_ID,
        definition_id=copy.id,
        limit=10,
        offset=0,
    )
    assert copy.id != definition.id  # nosec B101
    assert copy.lifecycle == "paused"  # nosec B101
    assert copy.name == "Paused copy"  # nosec B101
    assert original_total == 2  # nosec B101
    assert copy_total == 1  # nosec B101
    assert {event.event_type for event in original_events + copy_events} >= {  # nosec B101
        "definition.created",
        "definition_duplicated",
        "definition_duplicate_created",
    }


@pytest.mark.parametrize("lock_kind", ["none", "system"])
def test_duplicate_disabled_definition_succeeds_for_non_admin_non_security_locks(tmp_path, lock_kind):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)
    repo.update_definition(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        patch={
            "lifecycle": "disabled",
            "disabled_lock_kind": lock_kind,
            "disabled_reason": "Temporarily unavailable",
            "updated_by": ACTOR,
        },
        expected_version=definition.version,
    )

    copy = service.duplicate_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=ScheduledTaskDuplicateRequest(name=f"Copy {lock_kind}"),
    )

    assert copy.lifecycle == "paused"  # nosec B101
    assert copy.disabled_lock_kind == "none"  # nosec B101


@pytest.mark.parametrize("lock_kind", ["admin", "security"])
def test_duplicate_disabled_definition_with_admin_or_security_lock_fails_without_copy(tmp_path, lock_kind):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)
    repo.update_definition(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        patch={
            "lifecycle": "disabled",
            "disabled_lock_kind": lock_kind,
            "disabled_reason": "Locked by policy",
            "updated_by": ACTOR,
        },
        expected_version=definition.version,
    )

    with pytest.raises(ScheduledTaskAutomationError, match="definition_disabled_locked"):
        service.duplicate_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=definition.id,
            payload=ScheduledTaskDuplicateRequest(name="Should not exist"),
        )

    definitions, total = repo.list_definitions(owner_id=OWNER_ID, limit=10, offset=0)
    assert total == 1  # nosec B101
    assert definitions[0].id == definition.id  # nosec B101


def test_pause_resume_archive_transition_matrix(tmp_path):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)

    paused = service.pause_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id)
    paused_again = service.pause_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id)
    resumed = service.resume_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id)
    resumed_again = service.resume_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id)
    archived = service.archive_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id)
    archived_again = service.archive_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id)

    assert paused.lifecycle == "paused"  # nosec B101
    assert paused_again.lifecycle == "paused"  # nosec B101
    assert resumed.lifecycle == "configured"  # nosec B101
    assert resumed_again.lifecycle == "configured"  # nosec B101
    assert archived.lifecycle == "archived"  # nosec B101
    assert archived_again.lifecycle == "archived"  # nosec B101
    assert _audit_count(repo, definition.id) == 4  # created, paused, resumed, archived  # nosec B101

    for operation in (
        lambda: service.pause_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id),
        lambda: service.resume_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id),
        lambda: service.update_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=definition.id,
            payload=ScheduledTaskDefinitionUpdateRequest(preview_id="unused"),
        ),
        lambda: service.duplicate_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=definition.id,
            payload=ScheduledTaskDuplicateRequest(name="Archived copy"),
        ),
    ):
        with pytest.raises(ScheduledTaskAutomationError, match="definition_archived"):
            operation()


def test_preview_idempotency_replay_and_same_key_different_payload_conflict(tmp_path):
    service, repo = _service(tmp_path)
    payload = _payload(name="Idempotent preview")

    first = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=payload, idempotency_key="preview-key")
    replay = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=payload, idempotency_key="preview-key")

    with pytest.raises(ScheduledTaskAutomationError, match="scheduled_task_idempotency_conflict"):
        service.create_preview(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=_payload(name="Different preview"),
            idempotency_key="preview-key",
        )

    previews, total = repo.list_previews(owner_id=OWNER_ID, limit=10, offset=0)
    assert replay.id == first.id  # nosec B101
    assert total == 1  # nosec B101
    assert previews[0].id == first.id  # nosec B101


def test_list_definitions_uses_bulk_preview_lookup_for_config(tmp_path, monkeypatch):
    service, repo = _service(tmp_path)
    for index in range(3):
        preview = service.create_preview(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=_payload(
                name=f"Bulk config {index}",
                config_payload={"rank": index},
            ),
        )
        service.create_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=ScheduledTaskDefinitionCreateRequest(preview_id=preview.id),
        )

    def _unexpected_get_preview(*_args, **_kwargs):
        raise AssertionError("list_definitions should use bulk preview lookup")

    monkeypatch.setattr(repo, "get_preview", _unexpected_get_preview)

    definitions = service.list_definitions(owner_id=OWNER_ID, limit=10, offset=0)

    assert definitions.total == 3  # nosec B101
    assert {item.config["rank"] for item in definitions.items} == {0, 1, 2}  # nosec B101


def test_preview_idempotency_race_executes_side_effects_once(tmp_path, monkeypatch):
    service, repo = _service(tmp_path)
    _install_idempotency_miss_barrier(monkeypatch, repo)
    payload = _payload(name="Raced preview")

    def _create_preview():
        return service.create_preview(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=payload,
            idempotency_key="preview-race-key",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_create_preview) for _ in range(2)]
        responses = [future.result(timeout=10) for future in futures]

    previews, total = repo.list_previews(owner_id=OWNER_ID, limit=10, offset=0)
    assert responses[0].id == responses[1].id  # nosec B101
    assert total == 1  # nosec B101
    assert previews[0].id == responses[0].id  # nosec B101


def test_create_idempotency_replay_before_preview_consumption(tmp_path):
    service, repo = _service(tmp_path)
    preview = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=_payload())
    payload = ScheduledTaskDefinitionCreateRequest(preview_id=preview.id)

    first = service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=payload,
        idempotency_key="create-key",
    )
    replay = service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=payload,
        idempotency_key="create-key",
    )

    assert replay.id == first.id  # nosec B101
    assert repo.list_definitions(owner_id=OWNER_ID, limit=10, offset=0)[1] == 1  # nosec B101
    assert _audit_count(repo, first.id) == 1  # nosec B101


def test_duplicate_idempotency_race_does_not_create_extra_definitions_or_audits(tmp_path, monkeypatch):
    service, repo = _service(tmp_path)
    source = _create_definition(service, name="Duplicate race source")
    _install_idempotency_miss_barrier(monkeypatch, repo)
    payload = ScheduledTaskDuplicateRequest(name="Raced duplicate")

    def _duplicate_definition():
        return service.duplicate_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=source.id,
            payload=payload,
            idempotency_key="duplicate-race-key",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_duplicate_definition) for _ in range(2)]
        responses = [future.result(timeout=10) for future in futures]

    definitions, total = repo.list_definitions(owner_id=OWNER_ID, limit=10, offset=0)
    copy_ids = {definition.id for definition in definitions if definition.id != source.id}
    assert responses[0].id == responses[1].id  # nosec B101
    assert total == 2  # nosec B101
    assert copy_ids == {responses[0].id}  # nosec B101
    assert _audit_count(repo, source.id) == 2  # created + duplicated  # nosec B101
    assert _audit_count(repo, responses[0].id) == 1  # duplicate created  # nosec B101


def test_update_idempotency_replay_after_preview_consumption(tmp_path):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)
    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            mode="update",
            definition_id=definition.id,
            definition_version=definition.version,
            name="Idempotently updated",
        ),
    )
    payload = ScheduledTaskDefinitionUpdateRequest(preview_id=preview.id)

    first = service.update_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=payload,
        idempotency_key="update-key",
    )
    replay = service.update_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=payload,
        idempotency_key="update-key",
    )

    assert replay.id == first.id  # nosec B101
    assert replay.version == first.version  # nosec B101
    assert repo.get_definition(owner_id=OWNER_ID, definition_id=definition.id).version == first.version  # nosec B101
    assert _audit_count(repo, definition.id) == 2  # created + updated  # nosec B101


def test_duplicate_idempotency_replay_does_not_create_second_copy_or_audit_pair(tmp_path):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)
    payload = ScheduledTaskDuplicateRequest(name="Idempotent duplicate")

    first = service.duplicate_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=payload,
        idempotency_key="duplicate-key",
    )
    replay = service.duplicate_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=payload,
        idempotency_key="duplicate-key",
    )

    assert replay.id == first.id  # nosec B101
    assert repo.list_definitions(owner_id=OWNER_ID, limit=10, offset=0)[1] == 2  # nosec B101
    assert _audit_count(repo, definition.id) == 2  # created + duplicated  # nosec B101
    assert _audit_count(repo, first.id) == 1  # duplicate created only  # nosec B101


def test_pause_resume_archive_idempotency_replay_without_extra_audit_events(tmp_path):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)

    paused = service.pause_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        idempotency_key="pause-key",
    )
    pause_replay = service.pause_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        idempotency_key="pause-key",
    )
    resumed = service.resume_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        idempotency_key="resume-key",
    )
    resume_replay = service.resume_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        idempotency_key="resume-key",
    )
    archived = service.archive_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        idempotency_key="archive-key",
    )
    archive_replay = service.archive_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        idempotency_key="archive-key",
    )

    assert pause_replay.version == paused.version  # nosec B101
    assert resume_replay.version == resumed.version  # nosec B101
    assert archive_replay.version == archived.version  # nosec B101
    assert _audit_count(repo, definition.id) == 4  # created, paused, resumed, archived  # nosec B101


def test_update_rolls_back_definition_and_preview_consumption_when_audit_fails(tmp_path, monkeypatch):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)
    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            mode="update",
            definition_id=definition.id,
            definition_version=definition.version,
            name="Should roll back",
        ),
    )

    def _fail_audit(**_kwargs):
        raise RuntimeError("injected_audit_failure")

    monkeypatch.setattr(service, "_create_audit", _fail_audit)

    with pytest.raises(RuntimeError, match="injected_audit_failure"):
        service.update_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=definition.id,
            payload=ScheduledTaskDefinitionUpdateRequest(preview_id=preview.id),
        )

    loaded_definition = repo.get_definition(owner_id=OWNER_ID, definition_id=definition.id)
    loaded_preview = repo.get_preview(owner_id=OWNER_ID, preview_id=preview.id)
    assert loaded_definition.version == definition.version  # nosec B101
    assert loaded_definition.name == definition.name  # nosec B101
    assert loaded_preview.status == "valid"  # nosec B101
    assert loaded_preview.consumed_at is None  # nosec B101
    assert _audit_count(repo, definition.id) == 1  # created only  # nosec B101


def test_lifecycle_rolls_back_mutation_and_audit_when_idempotency_snapshot_fails(tmp_path, monkeypatch):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)

    def _fail_response_ref(_response):
        raise RuntimeError("injected_snapshot_failure")

    monkeypatch.setattr(service, "_response_ref", _fail_response_ref)

    with pytest.raises(RuntimeError, match="injected_snapshot_failure"):
        service.pause_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=definition.id,
            idempotency_key="snapshot-failure-key",
        )

    loaded_definition = repo.get_definition(owner_id=OWNER_ID, definition_id=definition.id)
    assert loaded_definition.lifecycle == "configured"  # nosec B101
    assert loaded_definition.version == definition.version  # nosec B101
    assert _audit_count(repo, definition.id) == 1  # created only  # nosec B101
    assert (  # nosec B101
        repo.get_idempotency_record(
            owner_id=OWNER_ID,
            route="scheduled_task_automation.definition.pause",
            key="snapshot-failure-key",
        )
        is None
    )


def test_lifecycle_idempotency_replay_returns_original_snapshot_after_later_mutation(tmp_path):
    service, repo = _service(tmp_path)
    definition = _create_definition(service)

    paused_snapshot = service.pause_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        idempotency_key="snapshot-pause-key",
    )
    service.resume_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id)
    replay = service.pause_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        idempotency_key="snapshot-pause-key",
    )
    current = repo.get_definition(owner_id=OWNER_ID, definition_id=definition.id)

    assert paused_snapshot.lifecycle == "paused"  # nosec B101
    assert paused_snapshot.version == 2  # nosec B101
    assert current.lifecycle == "configured"  # nosec B101
    assert current.version == 3  # nosec B101
    assert replay.model_dump(mode="json") == paused_snapshot.model_dump(mode="json")  # nosec B101


def test_create_idempotency_replay_returns_original_snapshot_after_later_mutation(tmp_path):
    service, _repo = _service(tmp_path)
    preview = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=_payload())
    payload = ScheduledTaskDefinitionCreateRequest(preview_id=preview.id)

    created_snapshot = service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=payload,
        idempotency_key="snapshot-create-key",
    )
    service.pause_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=created_snapshot.id)
    replay = service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=payload,
        idempotency_key="snapshot-create-key",
    )

    assert created_snapshot.lifecycle == "configured"  # nosec B101
    assert created_snapshot.version == 1  # nosec B101
    assert replay.model_dump(mode="json") == created_snapshot.model_dump(mode="json")  # nosec B101


def test_update_idempotency_replay_returns_original_snapshot_after_later_update(tmp_path):
    service, _repo = _service(tmp_path)
    definition = _create_definition(service)
    first_preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            mode="update",
            definition_id=definition.id,
            definition_version=definition.version,
            name="First keyed update",
        ),
    )
    first_payload = ScheduledTaskDefinitionUpdateRequest(preview_id=first_preview.id)
    first_snapshot = service.update_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=first_payload,
        idempotency_key="snapshot-update-key",
    )
    second_preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            mode="update",
            definition_id=definition.id,
            definition_version=first_snapshot.version,
            name="Second unkeyed update",
        ),
    )
    service.update_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=ScheduledTaskDefinitionUpdateRequest(preview_id=second_preview.id),
    )

    replay = service.update_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=first_payload,
        idempotency_key="snapshot-update-key",
    )

    assert first_snapshot.name == "First keyed update"  # nosec B101
    assert first_snapshot.version == 2  # nosec B101
    assert replay.model_dump(mode="json") == first_snapshot.model_dump(mode="json")  # nosec B101


def test_duplicate_idempotency_replay_returns_original_snapshot_after_copy_mutation(tmp_path):
    service, _repo = _service(tmp_path)
    definition = _create_definition(service)
    payload = ScheduledTaskDuplicateRequest(name="Snapshot duplicate")

    duplicate_snapshot = service.duplicate_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=payload,
        idempotency_key="snapshot-duplicate-key",
    )
    service.resume_definition(owner_id=OWNER_ID, actor=ACTOR, definition_id=duplicate_snapshot.id)
    replay = service.duplicate_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=payload,
        idempotency_key="snapshot-duplicate-key",
    )

    assert duplicate_snapshot.lifecycle == "paused"  # nosec B101
    assert duplicate_snapshot.version == 1  # nosec B101
    assert replay.model_dump(mode="json") == duplicate_snapshot.model_dump(mode="json")  # nosec B101


def test_same_key_different_payload_conflict_for_create_update_duplicate_and_lifecycle_routes(tmp_path):
    service, _repo = _service(tmp_path)
    preview = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=_payload(name="Create one"))
    service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=ScheduledTaskDefinitionCreateRequest(preview_id=preview.id),
        idempotency_key="create-conflict",
    )
    other_preview = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=_payload(name="Create two"))
    with pytest.raises(ScheduledTaskAutomationError, match="scheduled_task_idempotency_conflict"):
        service.create_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=ScheduledTaskDefinitionCreateRequest(preview_id=other_preview.id),
            idempotency_key="create-conflict",
        )

    definition = _create_definition(service, name="Update conflict")
    update_preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            mode="update",
            definition_id=definition.id,
            definition_version=definition.version,
            name="Update one",
        ),
    )
    service.update_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=definition.id,
        payload=ScheduledTaskDefinitionUpdateRequest(preview_id=update_preview.id),
        idempotency_key="update-conflict",
    )
    next_update_preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            mode="update",
            definition_id=definition.id,
            definition_version=definition.version + 1,
            name="Update two",
        ),
    )
    with pytest.raises(ScheduledTaskAutomationError, match="scheduled_task_idempotency_conflict"):
        service.update_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=definition.id,
            payload=ScheduledTaskDefinitionUpdateRequest(preview_id=next_update_preview.id),
            idempotency_key="update-conflict",
        )

    duplicate_source = _create_definition(service, name="Duplicate conflict")
    service.duplicate_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=duplicate_source.id,
        payload=ScheduledTaskDuplicateRequest(name="Duplicate one"),
        idempotency_key="duplicate-conflict",
    )
    with pytest.raises(ScheduledTaskAutomationError, match="scheduled_task_idempotency_conflict"):
        service.duplicate_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=duplicate_source.id,
            payload=ScheduledTaskDuplicateRequest(name="Duplicate two"),
            idempotency_key="duplicate-conflict",
        )

    lifecycle_definition = _create_definition(service, name="Lifecycle conflict one")
    other_lifecycle_definition = _create_definition(service, name="Lifecycle conflict two")
    service.pause_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        definition_id=lifecycle_definition.id,
        idempotency_key="lifecycle-conflict",
    )
    with pytest.raises(ScheduledTaskAutomationError, match="scheduled_task_idempotency_conflict"):
        service.pause_definition(
            owner_id=OWNER_ID,
            actor=ACTOR,
            definition_id=other_lifecycle_definition.id,
            idempotency_key="lifecycle-conflict",
        )


def test_agent_task_raw_sentinel_absent_from_all_response_and_repository_surfaces(tmp_path):
    service, repo = _service(tmp_path)
    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_payload(
            family="agent_task",
            input_payload={"agent_ref": "agent:security", "message": RAW_SENTINEL},
            name="Agent redaction",
        ),
    )
    definition = service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=ScheduledTaskDefinitionCreateRequest(preview_id=preview.id, initial_lifecycle="paused"),
    )
    detail = service.get_definition(owner_id=OWNER_ID, definition_id=definition.id)
    definitions = service.list_definitions(owner_id=OWNER_ID, limit=10, offset=0)
    audits = service.list_audit_events(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        limit=10,
        offset=0,
    )
    stored_preview = repo.get_preview(owner_id=OWNER_ID, preview_id=preview.id)
    stored_definition = repo.get_definition(owner_id=OWNER_ID, definition_id=definition.id)
    stored_audits = repo.list_audit_events(owner_id=OWNER_ID, definition_id=definition.id, limit=10, offset=0)[0]

    all_serialized = "\n".join(
        [
            _as_json_text(preview),
            _as_json_text(definition),
            _as_json_text(detail),
            _as_json_text(definitions),
            _as_json_text(audits),
            json.dumps(stored_preview.normalized_config, sort_keys=True),
            json.dumps(stored_definition.input, sort_keys=True),
            json.dumps([event.after for event in stored_audits], sort_keys=True),
        ]
    )
    assert RAW_SENTINEL not in all_serialized  # nosec B101
    assert RAW_SENTINEL.encode("utf-8") not in _database_bytes(repo)  # nosec B101
