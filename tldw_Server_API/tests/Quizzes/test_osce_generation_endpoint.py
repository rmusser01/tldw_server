from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import quizzes as quizzes_endpoint
from tldw_Server_API.app.api.v1.schemas.quizzes import QuizGenerateRequest, QuizGenerateResponse
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.services import quiz_generator
from tldw_Server_API.app.services.osce_generator import (
    OsceCitationError,
    OsceGenerationError,
    OsceProviderError,
    OsceUnsupportedContractError,
    OsceVerificationError,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def quizzes_db(tmp_path) -> CharactersRAGDB:
    db = CharactersRAGDB(
        str(tmp_path / "osce-generation.db"),
        client_id=f"authenticated-{uuid4().hex[:8]}",
    )
    yield db
    db.close_connection()


@pytest.fixture
def media_db(tmp_path) -> MediaDatabase:
    db = MediaDatabase(
        str(tmp_path / "osce-generation-media.db"),
        client_id=f"test-{uuid4().hex[:8]}",
    )
    yield db
    db.close_connection()


@pytest.fixture
def enabled_osce_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        quiz_generator._PROFILE_BY_ID["osce_scenario"],
        "status",
        "available",
    )


def _request(note_id: str, *, num_stations: int = 1) -> QuizGenerateRequest:
    return QuizGenerateRequest.model_validate(
        {
            "sources": [{"source_type": "note", "source_id": note_id}],
            "generation_profile": "osce_scenario",
            "num_stations": num_stations,
        }
    )


def _row_counts(db: CharactersRAGDB) -> tuple[int, int]:
    quiz_count = int(db.execute_query("SELECT COUNT(*) AS count FROM quizzes").fetchone()["count"])
    station_count = int(
        db.execute_query("SELECT COUNT(*) AS count FROM osce_stations").fetchone()["count"]
    )
    return quiz_count, station_count


@pytest.mark.asyncio
async def test_planned_osce_profile_is_unavailable_without_persistence(
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
) -> None:
    note_id = quizzes_db.add_note(title="Guide", content="Warfarin requires INR monitoring.")

    with pytest.raises(HTTPException) as exc_info:
        await quizzes_endpoint.generate_quiz(
            request=_request(note_id),
            db=quizzes_db,
            media_db=media_db,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {"code": "generation_profile_unavailable"}
    assert _row_counts(quizzes_db) == (0, 0)


@pytest.mark.asyncio
async def test_enabled_osce_generation_persists_exact_count_with_active_db_identity(
    monkeypatch: pytest.MonkeyPatch,
    enabled_osce_profile: None,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    note_id = quizzes_db.add_note(
        title="Guide",
        content="Warfarin requires INR monitoring and review of bleeding warning signs.",
    )

    result = await quizzes_endpoint.generate_quiz(
        request=_request(note_id, num_stations=2),
        db=quizzes_db,
        media_db=media_db,
    )
    response = QuizGenerateResponse.model_validate(result)

    assert response.output_kind == "osce_stations"
    assert result["questions"] == []
    assert len(result["osce_stations"]) == 2
    assert result["quiz"]["total_questions"] == 0
    assert result["quiz"]["total_stations"] == 2
    assert result["quiz"]["client_id"] == quizzes_db.client_id
    assert result["quiz"]["client_id"] != "unknown"
    assert all(station["verification_state"] == "source_verified" for station in result["osce_stations"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_stage", "error_type"),
    [
        ("normalize", OsceUnsupportedContractError),
        ("citation", OsceCitationError),
        ("verify", OsceVerificationError),
        ("provider", OsceProviderError),
    ],
)
async def test_pre_persistence_osce_failure_leaves_no_partial_rows(
    monkeypatch: pytest.MonkeyPatch,
    enabled_osce_profile: None,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
    failure_stage: str,
    error_type: type[Exception],
) -> None:
    note_id = quizzes_db.add_note(title="Guide", content="Warfarin requires INR monitoring.")

    async def fail_generation(**_: Any) -> Any:
        raise error_type(
            {
                "normalize": "osce_unsupported_contract",
                "citation": "osce_citation_failure",
                "verify": "osce_verification_failure",
                "provider": "osce_provider_failure",
            }[failure_stage]
        )

    monkeypatch.setattr(quiz_generator, "generate_osce_stations_from_sources", fail_generation)

    with pytest.raises(error_type):
        await quiz_generator.generate_quiz_from_sources(
            db=quizzes_db,
            media_db=media_db,
            sources=[{"source_type": "note", "source_id": note_id}],
            generation_profile="osce_scenario",
            num_stations=2,
        )

    assert _row_counts(quizzes_db) == (0, 0)


@pytest.mark.asyncio
async def test_second_station_insert_failure_rolls_back_quiz_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    enabled_osce_profile: None,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    note_id = quizzes_db.add_note(title="Guide", content="Warfarin requires INR monitoring.")
    original_insert = quizzes_db._insert_osce_station_row
    insert_count = 0

    def fail_second_insert(*args: Any, **kwargs: Any) -> int:
        nonlocal insert_count
        insert_count += 1
        if insert_count == 2:
            raise RuntimeError("injected second station failure")
        return original_insert(*args, **kwargs)

    monkeypatch.setattr(quizzes_db, "_insert_osce_station_row", fail_second_insert)

    with pytest.raises(OsceVerificationError, match="^osce_verification_failure$"):
        await quiz_generator.generate_quiz_from_sources(
            db=quizzes_db,
            media_db=media_db,
            sources=[{"source_type": "note", "source_id": note_id}],
            generation_profile="osce_scenario",
            num_stations=2,
        )

    assert _row_counts(quizzes_db) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (OsceCitationError("private-source-body"), 422),
        (OsceProviderError("private-provider-body"), 502),
    ],
)
async def test_endpoint_maps_osce_failures_to_bounded_public_codes(
    monkeypatch: pytest.MonkeyPatch,
    enabled_osce_profile: None,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
    error: OsceGenerationError,
    expected_status: int,
) -> None:
    note_id = quizzes_db.add_note(title="Guide", content="Warfarin requires INR monitoring.")

    async def fail_generation(**_: Any) -> Any:
        raise error

    monkeypatch.setattr(quizzes_endpoint, "generate_quiz_from_sources", fail_generation)

    with pytest.raises(HTTPException) as exc_info:
        await quizzes_endpoint.generate_quiz(
            request=_request(note_id),
            db=quizzes_db,
            media_db=media_db,
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == {"code": error.code}
    assert "private" not in repr(exc_info.value.detail)


def test_osce_request_ignores_unknown_candidate_notes_without_forwarding() -> None:
    request = QuizGenerateRequest.model_validate(
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "generation_profile": "osce_scenario",
            "candidate_notes": "private notes must not cross the boundary",
        }
    )

    assert "candidate_notes" not in request.model_dump()
