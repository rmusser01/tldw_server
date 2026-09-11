"""Retry-safe OSCE attempt persistence and lifecycle contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any
from uuid import UUID, uuid4

import pytest

pytestmark = pytest.mark.integration

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
            "patient_context": {
                "text": "A fictional adult has started warfarin.",
                "citations": [
                    {
                        "source_type": "document",
                        "source_id": "doc-1",
                        "label": "Anticoagulation guide",
                        "quote": "Use regular monitoring.",
                        "chunk_id": "chunk-1",
                    }
                ],
            },
            "recommended_duration_seconds": 480,
            "checklist_items": [
                {
                    "label": "Explains monitoring",
                    "rationale": "Monitoring supports safe treatment.",
                    "citations": [],
                },
                {
                    "label": "Explains warning signs",
                    "rationale": "Warning signs require prompt review.",
                    "citations": [],
                },
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
def station(db: CharactersRAGDB) -> dict[str, Any]:
    quiz_id = db.create_quiz(name="OSCE", activity_type="osce")
    return db.create_osce_station(
        quiz_id,
        stored_content(),
        origin="generated",
        provenance={"provider": "test-provider", "private": "snapshot provenance"},
        source_bundle=[{"source_id": "source-1", "quote": "snapshot source"}],
        verification_state="source_verified",
    )


def complete_selections(station_row: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    content = OsceStationStoredContent.model_validate(station_row["content"])
    checklist = {str(item.id): "met" for item in content.checklist_items}
    rubric = {
        str(domain.id): str(domain.levels[-1].id) for domain in content.rubric_domains
    }
    return checklist, rubric


def test_start_attempt_is_retry_safe_and_snapshots_station_metadata(
    db: CharactersRAGDB,
    station: dict[str, Any],
) -> None:
    client_attempt_id = uuid4()

    first = db.start_osce_attempt(station["id"], client_attempt_id)
    repeated = db.start_osce_attempt(station["id"], client_attempt_id)

    assert first is not None
    assert repeated is not None
    assert repeated["id"] == first["id"]
    assert repeated["version"] == 1
    assert first["station_snapshot"]["title"] == "Warfarin counselling"
    assert first["station_snapshot"]["provenance"]["provider"] == "test-provider"
    assert first["station_snapshot"]["source_bundle"][0]["source_id"] == "source-1"
    count = db.execute_query(
        "SELECT COUNT(*) AS count FROM osce_practice_attempts WHERE station_id = ?",
        (station["id"],),
    ).fetchone()["count"]
    assert count == 1


def test_attempt_snapshot_survives_station_edit_and_delete(
    db: CharactersRAGDB,
    station: dict[str, Any],
) -> None:
    existing_key = uuid4()
    attempt = db.start_osce_attempt(station["id"], existing_key)
    assert attempt is not None

    updated = db.update_osce_station(
        station["quiz_id"],
        station["id"],
        stored_content("Changed live station"),
        expected_version=station["version"],
    )
    assert updated is not None
    assert db.delete_osce_station(
        station["quiz_id"], station["id"], expected_version=updated["version"]
    )

    loaded = db.get_osce_attempt(attempt["id"])
    retried = db.start_osce_attempt(station["id"], existing_key)
    assert loaded is not None
    assert retried is not None
    assert loaded["station_snapshot"]["content"]["title"] == "Warfarin counselling"
    assert retried["id"] == attempt["id"]
    assert db.start_osce_attempt(station["id"], uuid4()) is None


def test_candidate_phase_accepts_notes_only(
    db: CharactersRAGDB,
    station: dict[str, Any],
) -> None:
    attempt = db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None
    checklist, rubric = complete_selections(station)

    updated = db.patch_osce_attempt(
        attempt["id"], expected_version=1, notes="Private draft"
    )

    assert updated is not None
    assert updated["candidate_notes"] == "Private draft"
    assert updated["version"] == 2
    with pytest.raises(InputError, match="self-assessment"):
        db.patch_osce_attempt(
            attempt["id"],
            expected_version=2,
            checklist_selections=checklist,
            rubric_selections=rubric,
        )


def test_revealed_phase_accepts_notes_and_domain_valid_assessment_selections(
    db: CharactersRAGDB,
    station: dict[str, Any],
) -> None:
    attempt = db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None
    revealed = db.transition_osce_attempt(
        attempt["id"], "self_assessment", expected_version=1
    )
    checklist, rubric = complete_selections(station)

    updated = db.patch_osce_attempt(
        attempt["id"],
        expected_version=revealed["version"],
        notes="Revised notes",
        checklist_selections=checklist,
        rubric_selections=rubric,
    )

    assert updated is not None
    assert updated["candidate_notes"] == "Revised notes"
    assert updated["checklist_selections"] == checklist
    assert updated["rubric_selections"] == rubric
    assert updated["version"] == 3


def test_patch_rejects_stale_version_and_completed_attempt_is_immutable(
    db: CharactersRAGDB,
    station: dict[str, Any],
) -> None:
    attempt = db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None
    updated = db.patch_osce_attempt(attempt["id"], expected_version=1, notes="First")
    assert updated is not None
    with pytest.raises(ConflictError, match="Version mismatch"):
        db.patch_osce_attempt(attempt["id"], expected_version=1, notes="Stale")

    revealed = db.transition_osce_attempt(
        attempt["id"], "self_assessment", expected_version=updated["version"]
    )
    checklist, rubric = complete_selections(station)
    assessed = db.patch_osce_attempt(
        attempt["id"],
        expected_version=revealed["version"],
        checklist_selections=checklist,
        rubric_selections=rubric,
    )
    assert assessed is not None
    completed = db.transition_osce_attempt(
        attempt["id"], "completed", expected_version=assessed["version"]
    )

    with pytest.raises(ConflictError, match="immutable"):
        db.patch_osce_attempt(
            attempt["id"], expected_version=completed["version"], notes="Too late"
        )


def test_reveal_freezes_server_elapsed_time_once_and_retries_are_idempotent(
    db: CharactersRAGDB,
    station: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None
    db.execute_query(
        "UPDATE osce_practice_attempts SET started_at = ? WHERE id = ?",
        ("2026-09-10T12:00:00.000Z", attempt["id"]),
    )
    monkeypatch.setattr(
        db,
        "_get_current_utc_timestamp_iso",
        lambda: "2026-09-10T12:05:07.900Z",
    )

    first = db.transition_osce_attempt(
        attempt["id"], "self_assessment", expected_version=1
    )
    repeated = db.transition_osce_attempt(
        attempt["id"], "self_assessment", expected_version=1
    )

    assert first["elapsed_seconds"] == 307
    assert repeated["id"] == first["id"]
    assert repeated["state"] == "self_assessment"
    assert repeated["version"] == first["version"] == 2
    assert repeated["self_assessment_started_at"] == first["self_assessment_started_at"]


def test_transition_requires_ordered_complete_lifecycle(
    db: CharactersRAGDB,
    station: dict[str, Any],
) -> None:
    attempt = db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None
    with pytest.raises(ConflictError, match="Invalid OSCE attempt transition"):
        db.transition_osce_attempt(attempt["id"], "completed", expected_version=1)

    revealed = db.transition_osce_attempt(
        attempt["id"], "self_assessment", expected_version=1
    )
    with pytest.raises(InputError, match="incomplete"):
        db.transition_osce_attempt(
            attempt["id"], "completed", expected_version=revealed["version"]
        )

    checklist, rubric = complete_selections(station)
    assessed = db.patch_osce_attempt(
        attempt["id"],
        expected_version=revealed["version"],
        checklist_selections=checklist,
        rubric_selections=rubric,
    )
    assert assessed is not None
    completed = db.transition_osce_attempt(
        attempt["id"], "completed", expected_version=assessed["version"]
    )
    repeated_complete = db.transition_osce_attempt(
        attempt["id"], "completed", expected_version=assessed["version"]
    )
    repeated_reveal = db.transition_osce_attempt(
        attempt["id"], "self_assessment", expected_version=1
    )

    assert completed["state"] == "completed"
    assert repeated_complete["version"] == completed["version"]
    assert repeated_reveal["state"] == "completed"
    assert repeated_reveal["version"] == completed["version"]


def test_list_attempts_filters_sorts_and_uses_snapshot_summaries(
    db: CharactersRAGDB,
    station: dict[str, Any],
) -> None:
    first = db.start_osce_attempt(
        station["id"], UUID("11111111-1111-4111-8111-111111111111")
    )
    second = db.start_osce_attempt(
        station["id"], UUID("22222222-2222-4222-8222-222222222222")
    )
    assert first is not None and second is not None
    second = db.patch_osce_attempt(second["id"], expected_version=1, notes="Never summarize")
    assert second is not None
    db.update_osce_station(
        station["quiz_id"],
        station["id"],
        stored_content("Edited live title"),
        expected_version=station["version"],
    )

    page = db.list_osce_attempts(
        quiz_id=station["quiz_id"],
        station_id=station["id"],
        states=["in_progress"],
        limit=10,
        offset=0,
    )

    assert page["count"] == 2
    assert [item["id"] for item in page["items"]] == [second["id"], first["id"]]
    assert all(item["station_title"] == "Warfarin counselling" for item in page["items"])
    assert all("notes" not in item and "candidate_notes" not in item for item in page["items"])


def test_two_sqlite_handles_create_one_retry_row(
    db: CharactersRAGDB,
    station: dict[str, Any],
) -> None:
    other = CharactersRAGDB(db.db_path, client_id="osce-second-handle")
    barrier = Barrier(2)
    client_attempt_id = uuid4()

    def start(handle: CharactersRAGDB) -> dict[str, Any] | None:
        barrier.wait()
        return handle.start_osce_attempt(station["id"], client_attempt_id)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(pool.map(start, (db, other)))
        assert all(attempt is not None for attempt in attempts)
        assert len({attempt["id"] for attempt in attempts if attempt is not None}) == 1
        count = db.execute_query(
            "SELECT COUNT(*) AS count FROM osce_practice_attempts WHERE station_id = ?",
            (station["id"],),
        ).fetchone()["count"]
        assert count == 1
    finally:
        other.close_all_connections()


def test_two_sqlite_handles_allow_only_one_expected_version_update(
    db: CharactersRAGDB,
    station: dict[str, Any],
) -> None:
    attempt = db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None
    other = CharactersRAGDB(db.db_path, client_id="osce-second-handle")
    barrier = Barrier(2)

    def patch(handle_and_note: tuple[CharactersRAGDB, str]) -> str:
        handle, note = handle_and_note
        barrier.wait()
        try:
            handle.patch_osce_attempt(attempt["id"], expected_version=1, notes=note)
        except ConflictError:
            return "conflict"
        return "updated"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(patch, ((db, "One"), (other, "Two"))))
        assert sorted(outcomes) == ["conflict", "updated"]
        assert db.get_osce_attempt(attempt["id"])["version"] == 2
    finally:
        other.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_postgres_retry_and_optimistic_attempt_writes(
    pg_database_config: DatabaseConfig,
) -> None:
    first_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    second_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    first_db = CharactersRAGDB(Path(":memory:"), client_id="osce-pg-1", backend=first_backend)
    second_db = CharactersRAGDB(Path(":memory:"), client_id="osce-pg-1", backend=second_backend)
    try:
        quiz_id = first_db.create_quiz(
            name="Postgres OSCE",
            activity_type="osce",
            client_id=first_db.client_id,
        )
        station = first_db.create_osce_station(
            quiz_id, stored_content(), origin="manual"
        )
        client_attempt_id = uuid4()
        barrier = Barrier(2)

        def start(handle: CharactersRAGDB) -> dict[str, Any] | None:
            barrier.wait()
            return handle.start_osce_attempt(station["id"], client_attempt_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(pool.map(start, (first_db, second_db)))
        assert all(attempt is not None for attempt in attempts)
        assert len({attempt["id"] for attempt in attempts if attempt is not None}) == 1

        attempt = attempts[0]
        assert attempt is not None
        first_db.patch_osce_attempt(attempt["id"], expected_version=1, notes="winner")
        with pytest.raises(ConflictError, match="Version mismatch"):
            second_db.patch_osce_attempt(attempt["id"], expected_version=1, notes="stale")
    finally:
        second_db.close_all_connections()
        first_db.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_postgres_attempts_are_scoped_to_the_parent_quiz_owner(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    attacker_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner = CharactersRAGDB(Path(":memory:"), client_id="osce-owner", backend=owner_backend)
    attacker = CharactersRAGDB(
        Path(":memory:"), client_id="osce-other-user", backend=attacker_backend
    )
    try:
        quiz_id = owner.create_quiz(
            name="Owner OSCE",
            activity_type="osce",
            client_id=owner.client_id,
        )
        station = owner.create_osce_station(
            quiz_id, stored_content(), origin="manual"
        )
        retry_key = uuid4()
        readable = owner.start_osce_attempt(station["id"], retry_key)
        patchable = owner.start_osce_attempt(station["id"], uuid4())
        transitionable = owner.start_osce_attempt(station["id"], uuid4())
        assert readable is not None and patchable is not None and transitionable is not None

        unfiltered = attacker.list_osce_attempts()
        filtered = attacker.list_osce_attempts(
            quiz_id=quiz_id,
            station_id=station["id"],
            states=["in_progress"],
        )
        loaded = attacker.get_osce_attempt(readable["id"])
        patched = attacker.patch_osce_attempt(
            patchable["id"], expected_version=1, notes="captured"
        )
        transitioned = attacker.transition_osce_attempt(
            transitionable["id"], "self_assessment", expected_version=1
        )
        captured_retry = attacker.start_osce_attempt(station["id"], retry_key)
        unauthorized_start = attacker.start_osce_attempt(station["id"], uuid4())

        assert unfiltered == {"items": [], "count": 0}
        assert filtered == {"items": [], "count": 0}
        assert loaded is None
        assert patched is None
        assert transitioned is None
        assert captured_retry is None
        assert unauthorized_start is None
        assert owner.get_osce_attempt(patchable["id"])["candidate_notes"] == ""
        assert owner.get_osce_attempt(transitionable["id"])["state"] == "in_progress"
        assert owner.list_osce_attempts()["count"] == 3
    finally:
        attacker.close_all_connections()
        owner.close_all_connections()
