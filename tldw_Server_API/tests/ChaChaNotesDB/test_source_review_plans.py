import json
from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackSourceSelection
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.Flashcards.source_review import (
    build_source_review_launch_metadata,
    compute_source_review_due_at,
    compute_source_review_schedule,
    normalize_source_review_bundle,
)


def test_source_review_day_offset_uses_local_midnight():
    due_at = compute_source_review_due_at(
        starts_on=date(2026, 7, 9),
        timezone_name="America/Los_Angeles",
        offset_value=1,
        offset_unit="day",
    )

    assert due_at == datetime(2026, 7, 10, 7, 0, tzinfo=timezone.utc)  # nosec B101


def test_source_review_month_offset_clamps_to_month_end():
    due_at = compute_source_review_due_at(
        starts_on=date(2026, 1, 31),
        timezone_name="America/Los_Angeles",
        offset_value=1,
        offset_unit="month",
    )

    assert due_at == datetime(2026, 2, 28, 8, 0, tzinfo=timezone.utc)  # nosec B101


@pytest.mark.parametrize(
    ("starts_on", "offset_value", "offset_unit", "expected_due_at"),
    [
        (
            date(2026, 1, 1),
            3650,
            "day",
            datetime(2035, 12, 30, tzinfo=timezone.utc),
        ),
        (
            date(2026, 1, 31),
            120,
            "month",
            datetime(2036, 1, 31, tzinfo=timezone.utc),
        ),
    ],
)
def test_source_review_offset_accepts_exact_caps(
    starts_on,
    offset_value,
    offset_unit,
    expected_due_at,
):
    due_at = compute_source_review_due_at(
        starts_on=starts_on,
        timezone_name="UTC",
        offset_value=offset_value,
        offset_unit=offset_unit,
    )

    assert due_at == expected_due_at  # nosec B101


@pytest.mark.parametrize(
    ("offset_value", "offset_unit"),
    [
        (0, "day"),
        (-1, "month"),
        (1.5, "day"),
        (True, "day"),
        (3651, "day"),
        (121, "month"),
    ],
)
def test_source_review_offset_rejects_non_positive_and_over_cap_values(
    offset_value,
    offset_unit,
):
    with pytest.raises(ValueError, match="offset_value"):
        compute_source_review_due_at(
            starts_on=date(2026, 1, 1),
            timezone_name="UTC",
            offset_value=offset_value,
            offset_unit=offset_unit,
        )


def test_source_review_due_at_rejects_unsupported_unit():
    with pytest.raises(ValueError, match="offset_unit"):
        compute_source_review_due_at(
            starts_on=date(2026, 1, 1),
            timezone_name="UTC",
            offset_value=1,
            offset_unit="week",
        )


def test_source_review_due_at_rejects_invalid_timezone():
    with pytest.raises(ValueError, match="timezone"):
        compute_source_review_due_at(
            starts_on=date(2026, 1, 1),
            timezone_name="Not/A_Timezone",
            offset_value=1,
            offset_unit="day",
        )


def test_source_review_day_offset_rejects_date_overflow_consistently():
    with pytest.raises(ValueError, match="supported date range"):
        compute_source_review_due_at(
            starts_on=date.max,
            timezone_name="UTC",
            offset_value=1,
            offset_unit="day",
        )


def test_source_review_month_offset_rejects_date_overflow_consistently():
    with pytest.raises(ValueError, match="supported date range"):
        compute_source_review_due_at(
            starts_on=date.max,
            timezone_name="UTC",
            offset_value=1,
            offset_unit="month",
        )


def test_source_review_schedule_rejects_duplicate_computed_due_activity():
    with pytest.raises(ValueError, match="Duplicate"):
        compute_source_review_schedule(
            starts_on=date(2026, 1, 31),
            timezone_name="UTC",
            schedule=[
                {
                    "offset_value": 28,
                    "offset_unit": "day",
                    "activity_type": "quiz",
                },
                {
                    "offset_value": 1,
                    "offset_unit": "month",
                    "activity_type": "quiz",
                },
            ],
        )


def test_source_review_schedule_allows_different_activities_at_same_due_at():
    rows = compute_source_review_schedule(
        starts_on=date(2026, 1, 31),
        timezone_name="UTC",
        schedule=[
            {
                "offset_value": 28,
                "offset_unit": "day",
                "activity_type": "quiz",
            },
            {
                "offset_value": 1,
                "offset_unit": "month",
                "activity_type": "flashcards",
            },
        ],
    )

    assert rows[0]["due_at"] == rows[1]["due_at"]  # nosec B101


@pytest.mark.parametrize(
    "schedule_row",
    [
        {"offset_value": 1, "offset_unit": "day"},
        {"offset_value": 1, "offset_unit": "day", "activity_type": 7},
        {
            "offset_value": 1,
            "offset_unit": "day",
            "activity_type": "matching",
        },
    ],
)
def test_source_review_schedule_rejects_invalid_activity_values(schedule_row):
    with pytest.raises(ValueError, match="activity_type"):
        compute_source_review_schedule(
            starts_on=date(2026, 1, 1),
            timezone_name="UTC",
            schedule=[schedule_row],
        )


@pytest.mark.parametrize(
    ("activity_type", "expected_launch_fields"),
    [
        (
            "reread",
            (
                "/flashcards",
                "source_review_due_panel",
                "show_reread_snapshot",
                "source_bundle",
            ),
        ),
        (
            "quiz",
            (
                "/quiz",
                "quiz_generation",
                "prefill_generation_sources",
                "source_items",
            ),
        ),
        (
            "flashcards",
            (
                "/flashcards",
                "flashcard_generation",
                "prefill_generation_sources",
                "source_items",
            ),
        ),
        (
            "cloze",
            (
                "/flashcards",
                "cloze_flashcard_generation",
                "prefill_generation_sources",
                "source_items",
            ),
        ),
    ],
)
def test_source_review_launch_metadata_maps_activity_to_existing_surface(
    activity_type,
    expected_launch_fields,
):
    metadata = build_source_review_launch_metadata(
        activity_type=activity_type,
        plan_id=7,
        occurrence_id=11,
        created_at="2026-07-09T12:00:00Z",
    )

    assert (
        metadata["target_route"],
        metadata["target_surface"],
        metadata["action"],
        metadata["source_payload_field"],
    ) == expected_launch_fields  # nosec B101
    assert metadata["activity_type"] == activity_type  # nosec B101
    assert metadata["plan_id"] == 7  # nosec B101
    assert metadata["occurrence_id"] == 11  # nosec B101
    assert metadata["completion_required"] is True  # nosec B101
    assert metadata["created_at"] == "2026-07-09T12:00:00Z"  # nosec B101


def test_source_review_launch_metadata_is_thin_and_under_size_cap():
    metadata = build_source_review_launch_metadata(
        activity_type="quiz",
        plan_id=7,
        occurrence_id=11,
        created_at="2026-07-09T12:00:00Z",
    )

    assert set(metadata) == {  # nosec B101
        "activity_type",
        "plan_id",
        "occurrence_id",
        "target_route",
        "target_surface",
        "action",
        "source_payload_field",
        "completion_required",
        "created_at",
    }
    assert len(json.dumps(metadata).encode("utf-8")) <= 16 * 1024  # nosec B101


def test_source_review_launch_metadata_accepts_exact_default_json_size_cap():
    baseline = build_source_review_launch_metadata(
        activity_type="quiz",
        plan_id=7,
        occurrence_id=11,
        created_at="",
    )
    created_at = "x" * (16 * 1024 - len(json.dumps(baseline).encode("utf-8")))

    metadata = build_source_review_launch_metadata(
        activity_type="quiz",
        plan_id=7,
        occurrence_id=11,
        created_at=created_at,
    )

    assert len(json.dumps(metadata).encode("utf-8")) == 16 * 1024  # nosec B101


def test_source_review_launch_metadata_rejects_unsupported_activity():
    with pytest.raises(ValueError, match="activity_type"):
        build_source_review_launch_metadata(
            activity_type="matching",
            plan_id=7,
            occurrence_id=11,
            created_at="2026-07-09T12:00:00Z",
        )


def test_source_review_launch_metadata_rejects_payload_over_size_cap():
    with pytest.raises(ValueError, match="16 KiB"):
        build_source_review_launch_metadata(
            activity_type="reread",
            plan_id=7,
            occurrence_id=11,
            created_at="x" * (16 * 1024),
        )


def test_source_review_launch_metadata_rejects_default_json_over_size_cap():
    baseline = build_source_review_launch_metadata(
        activity_type="quiz",
        plan_id=7,
        occurrence_id=11,
        created_at="",
    )
    created_at = "x" * (16 * 1024 - len(json.dumps(baseline).encode("utf-8")) + 1)
    oversized_metadata = {**baseline, "created_at": created_at}

    assert len(json.dumps(oversized_metadata).encode("utf-8")) == 16 * 1024 + 1  # nosec B101
    with pytest.raises(ValueError, match="16 KiB"):
        build_source_review_launch_metadata(
            activity_type="quiz",
            plan_id=7,
            occurrence_id=11,
            created_at=created_at,
        )


def test_source_review_bundle_normalizes_models_and_source_title_alias():
    bundle = normalize_source_review_bundle(
        [
            {
                "source_type": "media",
                "source_id": " 42 ",
                "source_title": " Lecture 42 ",
                "excerpt_text": " Additive increase. ",
                "locator": {"page": 12, "empty": None},
            },
            StudyPackSourceSelection(
                source_type="note",
                source_id="note-1",
                label="Notes",
            ),
        ]
    )

    assert bundle == {  # nosec B101
        "items": [
            {
                "source_type": "media",
                "source_id": "42",
                "label": "Lecture 42",
                "excerpt_text": "Additive increase.",
                "locator": {"page": 12},
            },
            {
                "source_type": "note",
                "source_id": "note-1",
                "label": "Notes",
                "locator": {},
            },
        ]
    }
    assert "source_title" not in bundle["items"][0]  # nosec B101


@pytest.fixture
def source_review_db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    db = CharactersRAGDB(tmp_path / "source-review.db", "source-review-test-client")
    try:
        yield db
    finally:
        db.close_connection()


def _source_bundle() -> dict[str, object]:
    return normalize_source_review_bundle(
        [
            {
                "source_type": "note",
                "source_id": "note-42",
                "source_title": "Congestion control",
                "excerpt_text": "Additive increase and multiplicative decrease.",
                "locator": {"line": 12},
            }
        ]
    )


def _schedule_row(
    due_at: str,
    *,
    activity_type: str = "quiz",
    offset_value: int = 1,
    offset_unit: str = "day",
) -> dict[str, object]:
    return {
        "offset_value": offset_value,
        "offset_unit": offset_unit,
        "activity_type": activity_type,
        "due_at": due_at,
    }


def _create_source_review_plan(
    db: CharactersRAGDB,
    *,
    title: str = "TCP review",
    schedule: list[dict[str, object]] | None = None,
) -> int:
    return db.create_source_review_plan(
        title=title,
        starts_on="2026-07-09",
        timezone_name="America/Los_Angeles",
        source_bundle_json=_source_bundle(),
        schedule=schedule or [_schedule_row("2026-07-10T07:00:00Z")],
    )


def _occurrence_rows(db: CharactersRAGDB, plan_id: int) -> list[dict[str, object]]:
    rows = db.execute_query(
        "SELECT * FROM source_review_occurrences WHERE plan_id = ? ORDER BY id",
        (plan_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def test_create_source_review_plan_persists_bundle_and_occurrences_atomically(
    source_review_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = "2026-07-09T12:00:00.000Z"
    monkeypatch.setattr(
        source_review_db,
        "_get_current_utc_timestamp_iso",
        lambda: created_at,
    )
    schedule = compute_source_review_schedule(
        starts_on=date(2026, 7, 9),
        timezone_name="America/Los_Angeles",
        schedule=[
            {"offset_value": 1, "offset_unit": "day", "activity_type": "reread"},
            {"offset_value": 3, "offset_unit": "day", "activity_type": "quiz"},
        ],
    )

    plan_id = _create_source_review_plan(source_review_db, schedule=schedule)

    plans, total = source_review_db.list_source_review_plans()
    occurrences = _occurrence_rows(source_review_db, plan_id)
    assert total == 1  # nosec B101
    assert plans[0]["source_bundle_json"] == _source_bundle()  # nosec B101
    assert plans[0]["version"] == 1  # nosec B101
    assert plans[0]["client_id"] == "source-review-test-client"  # nosec B101
    assert plans[0]["created_at"] == created_at  # nosec B101
    assert plans[0]["last_modified"] == created_at  # nosec B101
    assert [row["activity_type"] for row in occurrences] == ["reread", "quiz"]  # nosec B101
    assert [row["status"] for row in occurrences] == ["pending", "pending"]  # nosec B101
    assert all(row["version"] == 1 for row in occurrences)  # nosec B101
    assert all(row["client_id"] == "source-review-test-client" for row in occurrences)  # nosec B101
    assert all(row["created_at"] == created_at for row in occurrences)  # nosec B101
    assert all(row["last_modified"] == created_at for row in occurrences)  # nosec B101


def test_create_source_review_plan_rolls_back_after_occurrence_insert_failure(
    source_review_db: CharactersRAGDB,
) -> None:
    schedule = [
        _schedule_row("2026-07-10T07:00:00Z", activity_type="reread"),
        _schedule_row(
            "2026-07-12T07:00:00Z",
            activity_type="unsupported",
            offset_value=3,
        ),
    ]

    with pytest.raises(CharactersRAGDBError, match="create source review plan"):
        _create_source_review_plan(source_review_db, schedule=schedule)

    plan_count = source_review_db.execute_query("SELECT COUNT(*) AS total FROM source_review_plans").fetchone()["total"]
    occurrence_count = source_review_db.execute_query(
        "SELECT COUNT(*) AS total FROM source_review_occurrences"
    ).fetchone()["total"]
    sync_count = source_review_db.execute_query(
        """
        SELECT COUNT(*) AS total
          FROM sync_log
         WHERE entity IN ('source_review_plans', 'source_review_occurrences')
        """
    ).fetchone()["total"]
    assert plan_count == 0  # nosec B101
    assert occurrence_count == 0  # nosec B101
    assert sync_count == 0  # nosec B101


def test_source_review_plan_and_due_lists_have_stable_pagination_order(
    source_review_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        source_review_db,
        "_get_current_utc_timestamp_iso",
        lambda: "2026-07-09T12:00:00.000Z",
    )
    plan_ids = [
        _create_source_review_plan(
            source_review_db,
            title=title,
            schedule=[_schedule_row("2026-07-08T12:00:00Z", activity_type=activity)],
        )
        for title, activity in (("First", "reread"), ("Second", "quiz"), ("Third", "cloze"))
    ]

    plans, plan_total = source_review_db.list_source_review_plans(limit=2, offset=1)
    due_rows, due_total = source_review_db.list_due_source_review_occurrences(
        now_utc="2026-07-09T12:00:00Z",
        limit=2,
        offset=1,
    )

    occurrence_ids = [row["id"] for plan_id in plan_ids for row in _occurrence_rows(source_review_db, plan_id)]
    assert plan_total == 3  # nosec B101
    assert [row["id"] for row in plans] == list(reversed(plan_ids))[1:3]  # nosec B101
    assert due_total == 3  # nosec B101
    assert [row["id"] for row in due_rows] == occurrence_ids[1:3]  # nosec B101


def test_source_review_due_query_uses_stable_order_index_without_temp_sort(
    source_review_db: CharactersRAGDB,
) -> None:
    _create_source_review_plan(
        source_review_db,
        schedule=[
            _schedule_row("2026-07-08T12:00:00Z", activity_type="reread"),
            _schedule_row("2026-07-08T12:00:00Z", activity_type="quiz", offset_value=2),
        ],
    )

    query_plan = source_review_db.execute_query(
        """
        EXPLAIN QUERY PLAN
        SELECT o.id, o.plan_id, o.due_at, o.status, p.title
          FROM source_review_occurrences o
          JOIN source_review_plans p ON p.id = o.plan_id
         WHERE o.deleted = ? AND p.deleted = ?
           AND o.status IN ('pending', 'in_progress')
           AND o.due_at <= ?
         ORDER BY o.due_at ASC, o.id ASC
         LIMIT ? OFFSET ?
        """,
        (False, False, "2026-07-09T12:00:00Z", 50, 0),
    ).fetchall()
    details = [str(row["detail"]) for row in query_plan]

    assert any("idx_source_review_occurrences_due_list" in detail for detail in details)  # nosec B101
    assert not any("USE TEMP B-TREE FOR ORDER BY" in detail for detail in details)  # nosec B101


def test_source_review_schema_ensure_replaces_legacy_due_order_index(
    source_review_db: CharactersRAGDB,
) -> None:
    source_review_db.execute_query(
        """
        DROP INDEX IF EXISTS idx_source_review_occurrences_due_list;
        CREATE INDEX idx_source_review_occurrences_due_list
          ON source_review_occurrences(deleted, status, due_at, id);
        """,
        script=True,
    )

    with source_review_db.transaction() as conn:
        source_review_db._ensure_source_review_schema_sqlite(conn)

    index_sql = source_review_db.execute_query(
        """
        SELECT sql
          FROM sqlite_master
         WHERE type = 'index'
           AND name = 'idx_source_review_occurrences_due_list'
        """
    ).fetchone()["sql"]
    assert "(deleted,due_at,id,status)" in "".join(index_sql.split()).lower()  # nosec B101


def test_due_source_review_occurrences_filter_status_time_and_deleted_rows(
    source_review_db: CharactersRAGDB,
) -> None:
    due_at = "2026-07-08T12:00:00Z"
    plan_id = _create_source_review_plan(
        source_review_db,
        schedule=[
            _schedule_row(due_at, activity_type="reread", offset_value=1),
            _schedule_row(due_at, activity_type="quiz", offset_value=2),
            _schedule_row(due_at, activity_type="flashcards", offset_value=3),
            _schedule_row(due_at, activity_type="cloze", offset_value=4),
            _schedule_row("2026-07-11T12:00:00Z", activity_type="reread", offset_value=5),
            _schedule_row(due_at, activity_type="quiz", offset_value=6),
        ],
    )
    occurrence_ids = [int(row["id"]) for row in _occurrence_rows(source_review_db, plan_id)]
    source_review_db.start_source_review_occurrence(occurrence_ids[1])
    source_review_db.start_source_review_occurrence(occurrence_ids[2])
    source_review_db.complete_source_review_occurrence(occurrence_ids[2])
    source_review_db.skip_source_review_occurrence(occurrence_ids[3])
    source_review_db.execute_query(
        "UPDATE source_review_occurrences SET deleted = 1 WHERE id = ?",
        (occurrence_ids[5],),
        commit=True,
    )
    hidden_plan_id = _create_source_review_plan(
        source_review_db,
        title="Deleted plan",
        schedule=[_schedule_row(due_at, activity_type="reread")],
    )
    source_review_db.execute_query(
        "UPDATE source_review_plans SET deleted = 1 WHERE id = ?",
        (hidden_plan_id,),
        commit=True,
    )

    rows, total = source_review_db.list_due_source_review_occurrences(now_utc="2026-07-09T12:00:00Z")

    assert total == 2  # nosec B101
    assert [row["id"] for row in rows] == occurrence_ids[:2]  # nosec B101
    assert [row["status"] for row in rows] == ["pending", "in_progress"]  # nosec B101
    assert all(row["plan_title"] == "TCP review" for row in rows)  # nosec B101
    assert all(row["source_bundle_json"] == _source_bundle() for row in rows)  # nosec B101


def test_start_source_review_occurrence_is_idempotent_and_stores_thin_metadata(
    source_review_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = "2026-07-09T12:00:00.000Z"
    started_at = "2026-07-09T13:00:00.000Z"
    timestamps = iter((created_at, started_at))
    monkeypatch.setattr(
        source_review_db,
        "_get_current_utc_timestamp_iso",
        lambda: next(timestamps),
    )
    plan_id = _create_source_review_plan(source_review_db)
    occurrence_id = int(_occurrence_rows(source_review_db, plan_id)[0]["id"])
    source_review_db.client_id = "source-review-action-client"

    started = source_review_db.start_source_review_occurrence(occurrence_id)
    resumed = source_review_db.start_source_review_occurrence(occurrence_id)
    stored = _occurrence_rows(source_review_db, plan_id)[0]

    expected_keys = {
        "activity_type",
        "plan_id",
        "occurrence_id",
        "target_route",
        "target_surface",
        "action",
        "source_payload_field",
        "completion_required",
        "created_at",
    }
    assert started["status"] == "in_progress"  # nosec B101
    assert resumed["launch_state_json"] == started["launch_state_json"]  # nosec B101
    assert json.loads(stored["launch_state_json"]) == started["launch_state_json"]  # nosec B101
    assert set(started["launch_state_json"]) == expected_keys  # nosec B101
    assert "source_bundle" not in started["launch_state_json"]  # nosec B101
    assert "generated_content" not in started["launch_state_json"]  # nosec B101
    assert stored["created_at"] == created_at  # nosec B101
    assert stored["started_at"] == started_at  # nosec B101
    assert stored["last_modified"] == started_at  # nosec B101
    assert stored["client_id"] == "source-review-action-client"  # nosec B101
    assert stored["version"] == 2  # nosec B101


def test_start_source_review_occurrence_rolls_back_status_and_launch_on_update_failure(
    source_review_db: CharactersRAGDB,
) -> None:
    plan_id = _create_source_review_plan(source_review_db)
    occurrence_id = int(_occurrence_rows(source_review_db, plan_id)[0]["id"])
    source_review_db.execute_query(
        """
        CREATE TRIGGER reject_source_review_start
        BEFORE UPDATE OF status, launch_state_json ON source_review_occurrences
        WHEN OLD.status = 'pending' AND NEW.status = 'in_progress'
        BEGIN
          SELECT RAISE(ABORT, 'reject source review start');
        END;
        """,
        script=True,
    )

    with pytest.raises(CharactersRAGDBError, match="start source review occurrence"):
        source_review_db.start_source_review_occurrence(occurrence_id)

    stored = _occurrence_rows(source_review_db, plan_id)[0]
    update_sync_count = source_review_db.execute_query(
        """
        SELECT COUNT(*) AS total
          FROM sync_log
         WHERE entity = 'source_review_occurrences'
           AND entity_id = ?
           AND operation = 'update'
        """,
        (str(occurrence_id),),
    ).fetchone()["total"]
    assert stored["status"] == "pending"  # nosec B101
    assert stored["launch_state_json"] is None  # nosec B101
    assert stored["started_at"] is None  # nosec B101
    assert stored["version"] == 1  # nosec B101
    assert update_sync_count == 0  # nosec B101


def test_complete_source_review_occurrence_updates_metadata_and_completion_source(
    source_review_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = "2026-07-09T12:00:00.000Z"
    started_at = "2026-07-09T13:00:00.000Z"
    completed_at = "2026-07-09T14:00:00.000Z"
    timestamps = iter((created_at, started_at, completed_at))
    monkeypatch.setattr(
        source_review_db,
        "_get_current_utc_timestamp_iso",
        lambda: next(timestamps),
    )
    plan_id = _create_source_review_plan(source_review_db)
    occurrence_id = int(_occurrence_rows(source_review_db, plan_id)[0]["id"])
    source_review_db.start_source_review_occurrence(occurrence_id)
    source_review_db.client_id = "source-review-completion-client"

    completed = source_review_db.complete_source_review_occurrence(
        occurrence_id,
        completion_source="quiz_attempt",
    )

    assert completed["status"] == "completed"  # nosec B101
    assert completed["created_at"] == created_at  # nosec B101
    assert completed["started_at"] == started_at  # nosec B101
    assert completed["completed_at"] == completed_at  # nosec B101
    assert completed["last_modified"] == completed_at  # nosec B101
    assert completed["completion_source"] == "quiz_attempt"  # nosec B101
    assert completed["client_id"] == "source-review-completion-client"  # nosec B101
    assert completed["version"] == 3  # nosec B101


def test_skip_source_review_occurrence_updates_timestamp_client_and_version(
    source_review_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = "2026-07-09T12:00:00.000Z"
    skipped_at = "2026-07-09T13:00:00.000Z"
    timestamps = iter((created_at, skipped_at))
    monkeypatch.setattr(
        source_review_db,
        "_get_current_utc_timestamp_iso",
        lambda: next(timestamps),
    )
    plan_id = _create_source_review_plan(source_review_db)
    occurrence_id = int(_occurrence_rows(source_review_db, plan_id)[0]["id"])
    source_review_db.client_id = "source-review-skip-client"

    skipped = source_review_db.skip_source_review_occurrence(occurrence_id)

    assert skipped["status"] == "skipped"  # nosec B101
    assert skipped["created_at"] == created_at  # nosec B101
    assert skipped["last_modified"] == skipped_at  # nosec B101
    assert skipped["client_id"] == "source-review-skip-client"  # nosec B101
    assert skipped["version"] == 2  # nosec B101


def _occurrence_in_status(db: CharactersRAGDB, status: str) -> tuple[int, int]:
    plan_id = _create_source_review_plan(db)
    occurrence_id = int(_occurrence_rows(db, plan_id)[0]["id"])
    if status in {"in_progress", "completed"}:
        db.start_source_review_occurrence(occurrence_id)
    if status == "completed":
        db.complete_source_review_occurrence(occurrence_id)
    elif status == "skipped":
        db.skip_source_review_occurrence(occurrence_id)
    return plan_id, occurrence_id


@pytest.mark.parametrize(
    ("current_status", "action", "expected_status"),
    [
        ("pending", "start", "in_progress"),
        ("pending", "complete", None),
        ("pending", "skip", "skipped"),
        ("in_progress", "start", "in_progress"),
        ("in_progress", "complete", "completed"),
        ("in_progress", "skip", "skipped"),
        ("completed", "start", "completed"),
        ("completed", "complete", "completed"),
        ("completed", "skip", None),
        ("skipped", "start", None),
        ("skipped", "complete", None),
        ("skipped", "skip", "skipped"),
    ],
)
def test_source_review_occurrence_transition_table(
    source_review_db: CharactersRAGDB,
    current_status: str,
    action: str,
    expected_status: str | None,
) -> None:
    plan_id, occurrence_id = _occurrence_in_status(source_review_db, current_status)
    before = _occurrence_rows(source_review_db, plan_id)[0]
    method = getattr(source_review_db, f"{action}_source_review_occurrence")

    if expected_status is None:
        with pytest.raises(ConflictError):
            method(occurrence_id)
        result = None
    else:
        result = method(occurrence_id)

    after = _occurrence_rows(source_review_db, plan_id)[0]
    if result is not None:
        assert result["status"] == expected_status  # nosec B101
    assert after["status"] == (expected_status or current_status)  # nosec B101
    idempotent_transition = (current_status, action) in {
        ("in_progress", "start"),
        ("completed", "start"),
        ("completed", "complete"),
        ("skipped", "skip"),
    }
    if expected_status is None or idempotent_transition:
        assert after["version"] == before["version"]  # nosec B101
    else:
        assert after["version"] == before["version"] + 1  # nosec B101


@pytest.mark.parametrize("action", ["start", "complete", "skip"])
def test_deleted_source_review_occurrence_actions_act_not_found(
    source_review_db: CharactersRAGDB,
    action: str,
) -> None:
    plan_id = _create_source_review_plan(source_review_db)
    occurrence_id = int(_occurrence_rows(source_review_db, plan_id)[0]["id"])
    assert source_review_db.soft_delete_source_review_plan(plan_id) is True  # nosec B101

    with pytest.raises(ConflictError, match="not found"):
        getattr(source_review_db, f"{action}_source_review_occurrence")(occurrence_id)


def test_source_review_delete_rolls_back_plan_when_occurrence_delete_fails(
    source_review_db: CharactersRAGDB,
) -> None:
    plan_id = _create_source_review_plan(source_review_db)
    source_review_db.execute_query(
        """
        CREATE TRIGGER reject_source_review_occurrence_delete
        BEFORE UPDATE OF deleted ON source_review_occurrences
        WHEN OLD.deleted = 0 AND NEW.deleted = 1
        BEGIN
          SELECT RAISE(ABORT, 'reject occurrence delete');
        END;
        """,
        script=True,
    )

    with pytest.raises(CharactersRAGDBError, match="delete source review plan"):
        source_review_db.soft_delete_source_review_plan(plan_id)

    plan = source_review_db.execute_query(
        "SELECT deleted, version FROM source_review_plans WHERE id = ?",
        (plan_id,),
    ).fetchone()
    occurrence = _occurrence_rows(source_review_db, plan_id)[0]
    assert not plan["deleted"]  # nosec B101
    assert plan["version"] == 1  # nosec B101
    assert not occurrence["deleted"]  # nosec B101
    assert occurrence["version"] == 1  # nosec B101


def test_source_review_delete_is_idempotent_and_syncs_each_row_once(
    source_review_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = "2026-07-09T12:00:00.000Z"
    deleted_at = "2026-07-09T13:00:00.000Z"
    timestamps = iter((created_at, deleted_at))
    monkeypatch.setattr(
        source_review_db,
        "_get_current_utc_timestamp_iso",
        lambda: next(timestamps),
    )
    plan_id = _create_source_review_plan(
        source_review_db,
        schedule=[
            _schedule_row("2026-07-10T07:00:00Z", activity_type="reread"),
            _schedule_row("2026-07-12T07:00:00Z", activity_type="quiz", offset_value=3),
        ],
    )
    source_review_db.client_id = "source-review-delete-client"

    assert source_review_db.soft_delete_source_review_plan(plan_id) is True  # nosec B101
    versions_after_first_delete = [
        dict(
            source_review_db.execute_query(
                "SELECT deleted, created_at, last_modified, client_id, version "
                "FROM source_review_plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
        ),
        *[
            {
                "deleted": row["deleted"],
                "created_at": row["created_at"],
                "last_modified": row["last_modified"],
                "client_id": row["client_id"],
                "version": row["version"],
            }
            for row in _occurrence_rows(source_review_db, plan_id)
        ],
    ]
    delete_sync_rows_after_first = source_review_db.execute_query(
        """
        SELECT entity, entity_id, operation
          FROM sync_log
         WHERE entity IN ('source_review_plans', 'source_review_occurrences')
           AND operation = 'delete'
         ORDER BY change_id
        """
    ).fetchall()

    assert source_review_db.soft_delete_source_review_plan(plan_id) is False  # nosec B101
    versions_after_second_delete = [
        dict(
            source_review_db.execute_query(
                "SELECT deleted, created_at, last_modified, client_id, version "
                "FROM source_review_plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
        ),
        *[
            {
                "deleted": row["deleted"],
                "created_at": row["created_at"],
                "last_modified": row["last_modified"],
                "client_id": row["client_id"],
                "version": row["version"],
            }
            for row in _occurrence_rows(source_review_db, plan_id)
        ],
    ]
    delete_sync_rows_after_second = source_review_db.execute_query(
        """
        SELECT entity, entity_id, operation
          FROM sync_log
         WHERE entity IN ('source_review_plans', 'source_review_occurrences')
           AND operation = 'delete'
         ORDER BY change_id
        """
    ).fetchall()
    plans, total = source_review_db.list_source_review_plans()
    due, due_total = source_review_db.list_due_source_review_occurrences(now_utc="2030-01-01T00:00:00Z")

    assert versions_after_first_delete == [  # nosec B101
        {
            "deleted": 1,
            "created_at": created_at,
            "last_modified": deleted_at,
            "client_id": "source-review-delete-client",
            "version": 2,
        },
        {
            "deleted": 1,
            "created_at": created_at,
            "last_modified": deleted_at,
            "client_id": "source-review-delete-client",
            "version": 2,
        },
        {
            "deleted": 1,
            "created_at": created_at,
            "last_modified": deleted_at,
            "client_id": "source-review-delete-client",
            "version": 2,
        },
    ]
    assert versions_after_second_delete == versions_after_first_delete  # nosec B101
    assert len(delete_sync_rows_after_first) == 3  # nosec B101
    assert [dict(row) for row in delete_sync_rows_after_second] == [  # nosec B101
        dict(row) for row in delete_sync_rows_after_first
    ]
    assert plans == [] and total == 0  # nosec B101
    assert due == [] and due_total == 0  # nosec B101


def test_source_review_delete_missing_plan_uses_not_found_convention(
    source_review_db: CharactersRAGDB,
) -> None:
    with pytest.raises(ConflictError, match="not found"):
        source_review_db.soft_delete_source_review_plan(999_999)
