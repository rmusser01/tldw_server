from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import quizzes as quizzes_endpoint
from tldw_Server_API.app.api.v1.schemas.quizzes import QuizGenerateRequest, QuizGenerateResponse
from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
    ArtifactUnitResult,
    ArtifactVerificationResult,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.exceptions import (
    OsceCitationError,
    OsceGenerationError,
    OsceProviderError,
    OsceUnsupportedContractError,
    OsceVerificationError,
)
from tldw_Server_API.app.services import osce_generator, quiz_generator

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


def _create_shared_source(
    db: CharactersRAGDB,
    source_type: str,
) -> tuple[dict[str, str], str]:
    if source_type == "flashcard_deck":
        deck_id = db.add_deck(name="Anticoagulation")
        card_uuid = db.add_flashcard(
            {
                "deck_id": deck_id,
                "front": "Warfarin monitoring",
                "back": "Regular INR monitoring is required.",
            }
        )
        return {"source_type": source_type, "source_id": str(deck_id)}, card_uuid
    if source_type == "flashcard_card":
        card_uuid = db.add_flashcard(
            {
                "front": "Warfarin monitoring",
                "back": "Regular INR monitoring is required.",
            }
        )
        return {"source_type": source_type, "source_id": card_uuid}, card_uuid

    quiz_id = db.create_quiz(name="Anticoagulation review")
    question_id = db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which test monitors warfarin treatment?",
        correct_answer=0,
        options=["INR", "HbA1c"],
        explanation="Regular INR monitoring is required.",
    )
    attempt = db.start_attempt(quiz_id)
    db.submit_attempt(
        attempt["id"],
        [{"question_id": question_id, "user_answer": 1}],
    )
    attempt_id = int(attempt["id"])
    if source_type == "quiz_attempt":
        return {"source_type": source_type, "source_id": str(attempt_id)}, (
            f"{attempt_id}:{question_id}"
        )
    source_id = f"{attempt_id}:{question_id}"
    return {"source_type": source_type, "source_id": source_id}, source_id


@pytest.mark.asyncio
async def test_recovery_planned_osce_profile_is_unavailable_without_persistence(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
) -> None:
    monkeypatch.setitem(
        quiz_generator._PROFILE_BY_ID["osce_scenario"],
        "status",
        "planned",
    )
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
    "source_type",
    [
        "flashcard_deck",
        "flashcard_card",
        "quiz_attempt",
        "quiz_attempt_question",
    ],
)
async def test_osce_generation_accepts_every_shared_quiz_source_with_chunk_locator(
    source_type: str,
    monkeypatch: pytest.MonkeyPatch,
    enabled_osce_profile: None,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    source, expected_chunk_id = _create_shared_source(quizzes_db, source_type)

    result = await quiz_generator.generate_quiz_from_sources(
        db=quizzes_db,
        media_db=media_db,
        sources=[source],
        generation_profile="osce_scenario",
        num_stations=1,
    )

    citation = result["osce_stations"][0]["content"]["patient_context"]["citations"][0]
    assert citation["source_type"] == source_type
    assert citation["source_id"] == source["source_id"]
    assert citation["chunk_id"] == expected_chunk_id


@pytest.mark.asyncio
async def test_claim_supported_only_by_uncited_source_fails_without_persistence(
    monkeypatch: pytest.MonkeyPatch,
    enabled_osce_profile: None,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
) -> None:
    source_a = quizzes_db.add_note(
        title="Warfarin guide",
        content="Warfarin requires regular INR monitoring.",
    )
    source_b = quizzes_db.add_note(
        title="Aspirin guide",
        content="Aspirin requires review of gastrointestinal bleeding risk.",
    )
    citation_b = {
        "source_type": "note",
        "source_id": source_b,
        "chunk_id": source_b,
        "quote": "Aspirin requires review of gastrointestinal bleeding risk.",
    }
    generated_station = {
        "schema_version": "osce.station.v1",
        "title": "Anticoagulation review",
        "candidate_instructions": "Speak with a simulated patient.",
        "candidate_task": "Explain appropriate monitoring.",
        "patient_context": {
            "text": "Warfarin requires regular INR monitoring.",
            "citations": [citation_b],
        },
        "recommended_duration_seconds": 480,
        "checklist_items": [
            {
                "label": "Explains monitoring",
                "rationale": "Warfarin requires regular INR monitoring.",
                "citations": [citation_b],
            }
        ],
        "rubric_domains": [
            {
                "label": "Communication",
                "levels": [
                    {"label": "Needs development", "description": "Incomplete."},
                    {"label": "Effective", "description": "Clear and complete."},
                ],
            }
        ],
        "expected_key_points": [
            {
                "text": "Warfarin requires regular INR monitoring.",
                "citations": [citation_b],
            }
        ],
    }

    async def provider(**_: object) -> object:
        return {"stations": [generated_station]}

    async def verifier(**kwargs: object) -> ArtifactVerificationResult:
        documents = kwargs["source_documents"]
        source_text = "\n".join(document.content for document in documents)  # type: ignore[union-attr]
        unit_results = []
        for unit in kwargs["units"]:  # type: ignore[union-attr]
            grounded = all(claim in source_text for claim in unit.claims or [])
            unit_results.append(
                ArtifactUnitResult(
                    unit_id=unit.unit_id,
                    verdict="grounded" if grounded else "needs_revision",
                    statuses=["verified" if grounded else "unverified"],
                )
            )
        verdict = (
            "grounded"
            if all(result.verdict == "grounded" for result in unit_results)
            else "needs_revision"
        )
        return ArtifactVerificationResult(
            verdict=verdict,
            report={},
            unit_results=unit_results,
            metadata={},
        )

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", provider)
    monkeypatch.setattr(osce_generator, "verify_generated_artifact_against_sources", verifier)

    with pytest.raises(OsceVerificationError, match="^osce_verification_failure$"):
        await quiz_generator.generate_quiz_from_sources(
            db=quizzes_db,
            media_db=media_db,
            sources=[
                {"source_type": "note", "source_id": source_a},
                {"source_type": "note", "source_id": source_b},
            ],
            generation_profile="osce_scenario",
            num_stations=1,
        )

    assert _row_counts(quizzes_db) == (0, 0)


@pytest.mark.asyncio
async def test_global_verification_unit_cap_cannot_be_bypassed_by_source_groups(
    monkeypatch: pytest.MonkeyPatch,
    enabled_osce_profile: None,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
) -> None:
    source_a = quizzes_db.add_note(title="Source A", content="Canonical statement A.")
    source_b = quizzes_db.add_note(title="Source B", content="Canonical statement B.")
    citations = [
        {
            "source_type": "note",
            "source_id": source_id,
            "chunk_id": source_id,
            "quote": quote,
        }
        for source_id, quote in (
            (source_a, "Canonical statement A."),
            (source_b, "Canonical statement B."),
        )
    ]
    stations: list[dict[str, object]] = []
    unit_index = 0
    for station_index in range(9):
        patient_citation = citations[unit_index % 2]
        unit_index += 1
        checklist_items = []
        for checklist_index in range(7):
            citation = citations[unit_index % 2]
            unit_index += 1
            checklist_items.append(
                {
                    "label": f"Checklist item {checklist_index}",
                    "rationale": citation["quote"],
                    "citations": [citation],
                }
            )
        key_point_citation = citations[unit_index % 2]
        unit_index += 1
        stations.append(
            {
                "schema_version": "osce.station.v1",
                "title": f"Station {station_index}",
                "candidate_instructions": "Speak with a simulated patient.",
                "candidate_task": "Explain the source-backed point.",
                "patient_context": {
                    "text": patient_citation["quote"],
                    "citations": [patient_citation],
                },
                "recommended_duration_seconds": 480,
                "checklist_items": checklist_items,
                "rubric_domains": [
                    {
                        "label": "Communication",
                        "levels": [
                            {"label": "Needs development", "description": "Incomplete."},
                            {"label": "Effective", "description": "Complete."},
                        ],
                    }
                ],
                "expected_key_points": [
                    {
                        "text": key_point_citation["quote"],
                        "citations": [key_point_citation],
                    }
                ],
            }
        )
    assert unit_index == 81
    verifier_calls = 0

    async def provider(**_: object) -> object:
        return {"stations": stations}

    async def verifier(**kwargs: object) -> ArtifactVerificationResult:
        nonlocal verifier_calls
        verifier_calls += 1
        unit_results = [
            ArtifactUnitResult(
                unit_id=unit.unit_id,
                verdict="grounded",
                claim_ids=[f"{unit.unit_id}:c1"],
                statuses=["verified"],
            )
            for unit in kwargs["units"]
        ]
        return ArtifactVerificationResult(
            verdict="grounded",
            report={},
            unit_results=unit_results,
            metadata={},
        )

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", provider)
    monkeypatch.setattr(osce_generator, "verify_generated_artifact_against_sources", verifier)

    with pytest.raises(OsceVerificationError, match="^osce_verification_failure$"):
        await quiz_generator.generate_quiz_from_sources(
            db=quizzes_db,
            media_db=media_db,
            sources=[
                {"source_type": "note", "source_id": source_a},
                {"source_type": "note", "source_id": source_b},
            ],
            generation_profile="osce_scenario",
            num_stations=9,
        )

    assert verifier_calls == 0
    assert _row_counts(quizzes_db) == (0, 0)


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
