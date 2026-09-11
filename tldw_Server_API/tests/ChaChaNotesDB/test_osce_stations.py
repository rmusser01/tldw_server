"""OSCE station CRUD and atomic persistence contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.osce import OsceStationStoredContent
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.services.osce_practice import materialize_station_content


def stored_content(title: str = "Warfarin counselling") -> OsceStationStoredContent:
    return materialize_station_content(
        {
            "title": title,
            "candidate_instructions": "You are speaking with a simulated patient.",
            "candidate_task": "Explain safe medicine use.",
            "patient_context": {"text": "A fictional adult has started warfarin.", "citations": []},
            "recommended_duration_seconds": 480,
            "checklist_items": [
                {
                    "label": "Explains monitoring",
                    "rationale": "Monitoring supports safe treatment.",
                    "citations": [],
                }
            ],
            "rubric_domains": [
                {
                    "label": "Communication",
                    "levels": [
                        {"label": "Developing", "description": "The explanation is incomplete."},
                        {"label": "Effective", "description": "The explanation is clear."},
                    ],
                }
            ],
            "expected_key_points": [
                {"text": "Discusses monitoring and warning signs.", "citations": []}
            ],
        }
    )


@pytest.fixture
def db(quiz_db: CharactersRAGDB) -> CharactersRAGDB:
    return quiz_db


@pytest.fixture
def osce_quiz(db: CharactersRAGDB) -> dict[str, Any]:
    quiz_id = db.create_quiz(name="OSCE", activity_type="osce")
    quiz = db.get_quiz(quiz_id)
    assert quiz is not None
    return quiz


def test_create_station_updates_total_stations_in_same_transaction(
    osce_quiz: dict[str, Any],
    db: CharactersRAGDB,
) -> None:
    station = db.create_osce_station(
        osce_quiz["id"],
        stored_content(),
        origin="manual",
    )

    assert station["version"] == 1
    assert station["verification_state"] == "manually_authored"
    assert db.get_quiz(osce_quiz["id"])["total_stations"] == 1


def test_list_stations_is_ordered_and_paginated(
    osce_quiz: dict[str, Any],
    db: CharactersRAGDB,
) -> None:
    created = [
        db.create_osce_station(
            osce_quiz["id"],
            stored_content(f"Station {order}"),
            order_index=order,
            origin="manual",
        )
        for order in (2, 0, 1)
    ]

    page = db.list_osce_stations(osce_quiz["id"], limit=2, offset=1)

    assert page["count"] == 3
    assert [item["id"] for item in page["items"]] == [created[2]["id"], created[0]["id"]]


def test_station_paths_reject_question_quiz(db: CharactersRAGDB) -> None:
    quiz_id = db.create_quiz(name="Questions")

    with pytest.raises(ConflictError, match="Quiz not found"):
        db.create_osce_station(quiz_id, stored_content(), origin="manual")
    with pytest.raises(ConflictError, match="Quiz not found"):
        db.list_osce_stations(quiz_id)


def test_cross_quiz_station_lookup_is_not_found(
    osce_quiz: dict[str, Any],
    db: CharactersRAGDB,
) -> None:
    station = db.create_osce_station(osce_quiz["id"], stored_content(), origin="manual")
    other_quiz_id = db.create_quiz(name="Other OSCE", activity_type="osce")

    assert db.get_osce_station(other_quiz_id, station["id"]) is None
    assert (
        db.update_osce_station(
            other_quiz_id,
            station["id"],
            stored_content("Changed"),
            expected_version=1,
        )
        is None
    )
    assert db.delete_osce_station(other_quiz_id, station["id"]) is False


def test_update_station_uses_optimistic_version(
    osce_quiz: dict[str, Any],
    db: CharactersRAGDB,
) -> None:
    station = db.create_osce_station(osce_quiz["id"], stored_content(), origin="manual")

    updated = db.update_osce_station(
        osce_quiz["id"],
        station["id"],
        stored_content("Updated"),
        expected_version=station["version"],
        order_index=3,
    )

    assert updated is not None
    assert updated["content"]["title"] == "Updated"
    assert updated["order_index"] == 3
    assert updated["version"] == 2
    with pytest.raises(ConflictError, match="Version mismatch"):
        db.update_osce_station(
            osce_quiz["id"],
            station["id"],
            stored_content("Stale"),
            expected_version=station["version"],
        )


def test_soft_delete_recounts_and_is_idempotent(
    osce_quiz: dict[str, Any],
    db: CharactersRAGDB,
) -> None:
    station = db.create_osce_station(osce_quiz["id"], stored_content(), origin="manual")

    assert db.delete_osce_station(
        osce_quiz["id"], station["id"], expected_version=station["version"]
    ) is True
    assert db.get_osce_station(osce_quiz["id"], station["id"]) is None
    deleted = db.get_osce_station(osce_quiz["id"], station["id"], include_deleted=True)
    assert deleted is not None
    assert deleted["deleted"] is True
    assert deleted["version"] == 2
    assert db.get_quiz(osce_quiz["id"])["total_stations"] == 0
    assert db.delete_osce_station(osce_quiz["id"], station["id"]) is True


def test_create_station_rolls_back_when_recount_fails(
    osce_quiz: dict[str, Any],
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_recount(_conn: Any, _quiz_id: int) -> int:
        raise RuntimeError("recount failed")

    monkeypatch.setattr(db, "_recount_quiz_stations", fail_recount)

    with pytest.raises(RuntimeError, match="recount failed"):
        db.create_osce_station(osce_quiz["id"], stored_content(), origin="manual")

    count = db.execute_query(
        "SELECT COUNT(*) AS count FROM osce_stations WHERE quiz_id = ?",
        (osce_quiz["id"],),
    ).fetchone()["count"]
    assert count == 0
    assert db.get_quiz(osce_quiz["id"])["total_stations"] == 0


def test_create_quiz_with_stations_is_atomic(db: CharactersRAGDB) -> None:
    quiz, stations = db.create_quiz_with_osce_stations_atomic(
        {
            "name": "Generated OSCE",
            "description": "Two generated stations",
            "activity_type": "osce",
            "generation_profile": "osce_scenario",
        },
        [
            {
                "content": stored_content("Generated 1"),
                "order_index": 0,
                "origin": "generated",
                "provenance": {"provider": "test-provider"},
                "source_bundle": [{"source_id": "source-1"}],
                "verification_state": "source_verified",
                "verification_timestamp": "2026-09-10T12:00:00.000Z",
                "verification_summary": "Verified against the source bundle.",
            },
            {
                "content": stored_content("Generated 2"),
                "order_index": 1,
                "origin": "generated",
                "verification_state": "source_verified",
            },
        ],
    )

    assert quiz["activity_type"] == "osce"
    assert quiz["total_stations"] == 2
    assert [station["content"]["title"] for station in stations] == [
        "Generated 1",
        "Generated 2",
    ]
    assert stations[0]["provenance"] == {"provider": "test-provider"}


def test_atomic_quiz_creation_rolls_back_all_rows_on_second_station_failure(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_insert = db._insert_osce_station_row
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second station failed")
        return real_insert(*args, **kwargs)

    monkeypatch.setattr(db, "_insert_osce_station_row", fail_second)

    with pytest.raises(RuntimeError, match="second station failed"):
        db.create_quiz_with_osce_stations_atomic(
            {"name": "Rolled back", "activity_type": "osce"},
            [
                {"content": stored_content("One"), "origin": "generated"},
                {"content": stored_content("Two"), "origin": "generated"},
            ],
        )

    assert db.list_quizzes(include_workspace_items=True)["count"] == 0
    assert db.execute_query("SELECT COUNT(*) AS count FROM osce_stations").fetchone()["count"] == 0


def test_hard_quiz_delete_cascades_stations_and_attempts(
    osce_quiz: dict[str, Any],
    db: CharactersRAGDB,
) -> None:
    station = db.create_osce_station(osce_quiz["id"], stored_content(), origin="manual")
    now = db._get_current_utc_timestamp_iso()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO osce_practice_attempts("
            "station_id, quiz_id, client_attempt_id, station_snapshot_json, state, "
            "started_at, last_modified_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (station["id"], osce_quiz["id"], "attempt-1", "{}", "in_progress", now, now),
        )

    assert db.delete_quiz(osce_quiz["id"], hard_delete=True) is True
    assert db.execute_query("SELECT COUNT(*) AS count FROM osce_stations").fetchone()["count"] == 0
    assert (
        db.execute_query("SELECT COUNT(*) AS count FROM osce_practice_attempts").fetchone()["count"]
        == 0
    )


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_postgres_station_crud_and_atomic_bundle(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(Path(":memory:"), client_id="osce-postgres", backend=backend)
    try:
        quiz_id = db.create_quiz(
            name="Postgres OSCE",
            activity_type="osce",
            client_id=db.client_id,
        )
        station = db.create_osce_station(quiz_id, stored_content(), origin="manual")
        assert db.get_osce_station(quiz_id, station["id"])["content"]["title"] == "Warfarin counselling"
        updated = db.update_osce_station(
            quiz_id,
            station["id"],
            stored_content("Updated PostgreSQL station"),
            expected_version=station["version"],
        )
        assert updated is not None
        assert updated["content"]["title"] == "Updated PostgreSQL station"
        assert db.delete_osce_station(
            quiz_id,
            station["id"],
            expected_version=updated["version"],
        )
        quiz_count = db.execute_query(
            "SELECT total_stations FROM quizzes WHERE id = ?",
            (quiz_id,),
        ).fetchone()
        assert quiz_count["total_stations"] == 0

        bundle_quiz, bundle_stations = db.create_quiz_with_osce_stations_atomic(
            {"name": "Postgres bundle", "activity_type": "osce"},
            [
                {"content": stored_content("One"), "origin": "generated"},
                {"content": stored_content("Two"), "origin": "generated"},
            ],
        )
        assert bundle_quiz["total_stations"] == 2
        assert len(bundle_stations) == 2

        invalid_content = stored_content("Invalid second station").model_dump(mode="json")
        invalid_content["schema_version"] = "unsupported"
        with pytest.raises(InputError, match="schema_version"):
            db.create_quiz_with_osce_stations_atomic(
                {"name": "Postgres rolled back", "activity_type": "osce"},
                [
                    {"content": stored_content("First insert"), "origin": "generated"},
                    {"content": invalid_content, "origin": "generated"},
                ],
            )
        rolled_back = db.execute_query(
            "SELECT COUNT(*) AS count FROM quizzes WHERE name = ?",
            ("Postgres rolled back",),
        ).fetchone()
        assert rolled_back["count"] == 0
    finally:
        db.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_postgres_station_operations_are_scoped_to_the_parent_quiz_owner(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    attacker_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner = CharactersRAGDB(Path(":memory:"), client_id="station-owner", backend=owner_backend)
    attacker = CharactersRAGDB(
        Path(":memory:"), client_id="station-other-user", backend=attacker_backend
    )
    try:
        quiz_id = owner.create_quiz(
            name="Owner OSCE",
            activity_type="osce",
            client_id=owner.client_id,
        )
        readable = owner.create_osce_station(
            quiz_id, stored_content("Readable"), origin="manual"
        )
        updateable = owner.create_osce_station(
            quiz_id, stored_content("Updateable"), origin="manual"
        )
        deletable = owner.create_osce_station(
            quiz_id, stored_content("Deletable"), origin="manual"
        )

        operations = (
            lambda: attacker.create_osce_station(
                quiz_id, stored_content("Injected"), origin="manual"
            ),
            lambda: attacker.list_osce_stations(quiz_id),
            lambda: attacker.get_osce_station(quiz_id, readable["id"]),
            lambda: attacker.update_osce_station(
                quiz_id,
                updateable["id"],
                stored_content("Captured"),
                expected_version=updateable["version"],
            ),
            lambda: attacker.delete_osce_station(
                quiz_id,
                deletable["id"],
                expected_version=deletable["version"],
            ),
        )
        outcomes: list[ConflictError | None] = []
        for operation in operations:
            try:
                operation()
            except ConflictError as exc:
                outcomes.append(exc)
            else:
                outcomes.append(None)

        assert attacker.get_quiz(quiz_id) is None
        assert all(isinstance(outcome, ConflictError) for outcome in outcomes)
        assert all("Quiz not found" in str(outcome) for outcome in outcomes)
        assert owner.list_osce_stations(quiz_id)["count"] == 3
        assert owner.get_osce_station(quiz_id, updateable["id"])["content"]["title"] == (
            "Updateable"
        )
        assert owner.get_osce_station(quiz_id, deletable["id"]) is not None
    finally:
        attacker.close_all_connections()
        owner.close_all_connections()
