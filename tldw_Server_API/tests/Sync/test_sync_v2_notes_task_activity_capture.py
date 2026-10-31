"""Dormant capture contracts for canonical Notes task activity."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import (
    build_task_activity_capture,
    build_task_capture_mutation,
)
from tldw_Server_API.app.core.Sync.v2.models import SYNC_V2_SUPPORTED_DOMAINS
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    notes_task_object_hash,
    parse_notes_task_v1,
)

pytestmark = pytest.mark.unit

OWNER_ID = "activity-capture-owner"
DATASET_ID = "local-unbound"
NOTE_ID = "11111111-1111-4111-8111-111111111111"
TASK_ID = "22222222-2222-4222-8222-222222222222"
BEFORE_TIME = "2026-08-23T10:00:00+00:00"
AFTER_TIME = "2026-08-23T10:01:00+00:00"


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "activity-capture.db", client_id=OWNER_ID)
    try:
        yield database
    finally:
        database.close_connection()


def _metadata(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "description": None,
        "priority": None,
        "due_date": None,
        "estimate": None,
        "recurrence": None,
        "assignee_id": None,
        "tags": [],
        "custom": {},
    }
    value.update(changes)
    return value


def _task_row(
    *,
    title: str = "Prepare launch",
    status: str = "open",
    completed_at: str | None = None,
    metadata: dict[str, object] | None = None,
    projection_status: str = "live",
    deleted: bool = False,
    version: int = 1,
    canonical_revision: int = 1,
    updated_at: str = BEFORE_TIME,
) -> dict[str, Any]:
    canonical_metadata = deepcopy(metadata or _metadata())
    payload = parse_notes_task_v1(
        {
            "task_id": TASK_ID,
            "note_id": NOTE_ID,
            "title": title,
            "status": status,
            "completed_at": completed_at,
            **canonical_metadata,
        },
        owner_user_id=OWNER_ID,
    )
    return {
        "owner_user_id": OWNER_ID,
        "dataset_id": DATASET_ID,
        "id": TASK_ID,
        "note_id": NOTE_ID,
        "text": title,
        "status": status,
        "completed_at": completed_at,
        "metadata_json": canonical_metadata,
        "projection_status": projection_status,
        "deleted": deleted,
        "version": version,
        "canonical_revision": canonical_revision,
        "canonical_hash": notes_task_object_hash(
            payload,
            revision=canonical_revision,
            deleted=deleted,
        ),
        "updated_at": updated_at,
        "source_diagnostic_code": None,
    }


def _actor() -> TaskActor:
    return TaskActor(
        actor_type="user",
        actor_id=OWNER_ID,
        idempotency_key="activity-request-1",
    )


def _capture(
    db: CharactersRAGDB,
    before: dict[str, Any] | None,
    after: dict[str, Any],
):
    return build_task_activity_capture(
        db=db,
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        actor=_actor(),
        before=before,
        after=after,
        source_kind="rest",
    )


def test_create_capture_uses_exact_snapshot_and_stable_step(db: CharactersRAGDB) -> None:
    after = _task_row(metadata=_metadata(description="Ship safely", priority="high"))

    first = _capture(db, None, after)
    replay = _capture(db, None, after)

    assert first == replay
    assert UUID(first.payload.activity_id).version == 4
    assert first.payload.event_type == "created"
    assert first.payload.old_value is None
    assert first.payload.new_value == {
        "title": "Prepare launch",
        "status": "open",
        "completed_at": None,
        "metadata": _metadata(description="Ship safely", priority="high"),
    }
    assert first.payload.client_occurred_at == BEFORE_TIME
    assert first.payload.source_kind == "rest"
    assert first.payload.source_device_id is None
    assert first.step.domain == "notes.task_activity"
    assert first.step.object_id == first.payload.activity_id
    assert first.step.parent_id == NOTE_ID
    assert first.step.object_revision == 1
    assert first.step.payload == first.payload.model_dump(mode="json")
    assert first.step.created_at_client == BEFORE_TIME
    assert first.step.client_envelope_id == replay.step.client_envelope_id


@pytest.mark.parametrize(
    ("before", "after", "event_type", "old_value", "new_value"),
    [
        (
            _task_row(),
            _task_row(
                title="Prepare launch notes",
                metadata=_metadata(description="Detailed"),
                version=2,
                canonical_revision=2,
                updated_at=AFTER_TIME,
            ),
            "updated",
            {"title": "Prepare launch", "metadata": _metadata()},
            {
                "title": "Prepare launch notes",
                "metadata": _metadata(description="Detailed"),
            },
        ),
        (
            _task_row(),
            _task_row(
                metadata=_metadata(
                    recurrence={
                        "frequency": "weekly",
                        "interval": 1,
                        "by_weekday": ["mo"],
                        "until": None,
                        "state": "active",
                        "occurrence_index": 0,
                    }
                ),
                version=2,
                canonical_revision=2,
                updated_at=AFTER_TIME,
            ),
            "updated",
            {"metadata": _metadata()},
            {
                "metadata": _metadata(
                    recurrence={
                        "frequency": "weekly",
                        "interval": 1,
                        "by_weekday": ["mo"],
                        "until": None,
                        "state": "active",
                        "occurrence_index": 0,
                    }
                )
            },
        ),
        (
            _task_row(),
            _task_row(
                status="done",
                completed_at=AFTER_TIME,
                version=2,
                canonical_revision=2,
                updated_at=AFTER_TIME,
            ),
            "completed",
            {"status": "open"},
            {"status": "done"},
        ),
        (
            _task_row(status="done", completed_at=BEFORE_TIME),
            _task_row(
                status="open",
                completed_at=None,
                version=2,
                canonical_revision=2,
                updated_at=AFTER_TIME,
            ),
            "reopened",
            {"status": "done"},
            {"status": "open"},
        ),
        (
            _task_row(),
            _task_row(
                projection_status="unlinked",
                version=2,
                canonical_revision=1,
                updated_at=AFTER_TIME,
            ),
            "projection_unlinked",
            {"projection_status": "live"},
            {"projection_status": "unlinked"},
        ),
        (
            _task_row(projection_status="unlinked"),
            _task_row(
                projection_status="deleted",
                deleted=True,
                version=2,
                canonical_revision=2,
                updated_at=AFTER_TIME,
            ),
            "deleted",
            {"deleted": False, "projection_status": "unlinked"},
            {"deleted": True, "projection_status": "deleted"},
        ),
        (
            _task_row(projection_status="deleted", deleted=True),
            _task_row(
                projection_status="unlinked",
                deleted=False,
                version=2,
                canonical_revision=2,
                updated_at=AFTER_TIME,
            ),
            "restored",
            {"deleted": True, "projection_status": "deleted"},
            {"deleted": False, "projection_status": "unlinked"},
        ),
    ],
)
def test_capture_derives_each_portable_transition_shape(
    db: CharactersRAGDB,
    before: dict[str, Any],
    after: dict[str, Any],
    event_type: str,
    old_value: dict[str, object],
    new_value: dict[str, object],
) -> None:
    capture = _capture(db, before, after)

    assert capture.payload.event_type == event_type
    assert capture.payload.old_value == old_value
    assert capture.payload.new_value == new_value
    assert capture.payload.note_id == NOTE_ID
    assert capture.payload.task_id == TASK_ID


def test_task_capture_returns_stable_ordered_task_and_activity_plan(
    db: CharactersRAGDB,
) -> None:
    before = _task_row()
    after = _task_row(
        status="done",
        completed_at=AFTER_TIME,
        version=2,
        canonical_revision=2,
        updated_at=AFTER_TIME,
    )

    first = build_task_capture_mutation(
        db=db,
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        actor=_actor(),
        before=before,
        after=after,
    )
    repair = build_task_capture_mutation(
        db=db,
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        actor=_actor(),
        before=before,
        after=after,
    )

    assert first == repair
    assert first.steps == (first.step, first.activity.step)
    assert [step.domain for step in first.steps] == [
        "notes.task",
        "notes.task_activity",
    ]
    assert first.activity.payload.event_type == "completed"
    assert first.activity.payload.activity_id == repair.activity.payload.activity_id
    assert first.activity.step.client_envelope_id == repair.activity.step.client_envelope_id
    assert len({step.object_id for step in repair.steps}) == 2


def test_capture_does_not_activate_public_task_domains() -> None:
    dormant = {"notes.task", "notes.task_activity"}

    assert dormant.isdisjoint(SYNC_V2_SUPPORTED_DOMAINS)
