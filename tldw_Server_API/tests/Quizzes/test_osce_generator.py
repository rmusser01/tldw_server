from __future__ import annotations

from copy import deepcopy
from uuid import UUID

import pytest
from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
    ArtifactVerificationResult,
)
from tldw_Server_API.app.services import osce_generator, quiz_generator
from tldw_Server_API.app.services.osce_generator import (
    OsceCitationError,
    OsceMalformedOutputError,
    OsceProviderError,
    OsceVerificationError,
    build_osce_generation_prompt,
    build_osce_verification_units,
    generate_osce_stations_from_sources,
    normalize_generated_station,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def resolved_sources() -> list[dict[str, object]]:
    return [
        {
            "source_type": "note",
            "source_id": "note-1",
            "chunk_id": "chunk-1",
            "label": "Anticoagulation guide",
            "text": (
                "Warfarin requires regular INR monitoring. Patients should seek help for "
                "clinically important bleeding warning signs."
            ),
        }
    ]


@pytest.fixture
def normalized_sources() -> list[dict[str, str]]:
    return [{"source_type": "note", "source_id": "note-1"}]


@pytest.fixture
def valid_generated_station() -> dict[str, object]:
    citation = {
        "source_type": "note",
        "source_id": "note-1",
        "chunk_id": "chunk-1",
        "quote": "Warfarin requires regular INR monitoring.",
    }
    return {
        "id": "provider-station-id",
        "schema_version": "osce.station.v1",
        "title": "Counsel a patient starting warfarin",
        "candidate_instructions": "You are speaking with a simulated patient.",
        "candidate_task": "Explain monitoring and important warning signs.",
        "patient_context": {
            "id": "provider-patient-context-id",
            "text": "A fictional adult has recently started warfarin.",
            "citations": [{**deepcopy(citation), "id": "provider-citation-id"}],
        },
        "recommended_duration_seconds": 480,
        "checklist_items": [
            {
                "id": "provider-checklist-id",
                "label": "Explains INR monitoring",
                "rationale": "Warfarin requires regular INR monitoring.",
                "citations": [deepcopy(citation)],
            }
        ],
        "rubric_domains": [
            {
                "id": "provider-domain-id",
                "label": "Communication",
                "levels": [
                    {
                        "id": "provider-level-one",
                        "label": "Needs development",
                        "description": "The explanation is incomplete.",
                    },
                    {
                        "id": "provider-level-two",
                        "label": "Effective",
                        "description": "The explanation is clear and checks understanding.",
                    },
                ],
            }
        ],
        "expected_key_points": [
            {
                "id": "provider-key-point-id",
                "text": "Warfarin requires regular INR monitoring.",
                "citations": [deepcopy(citation)],
            }
        ],
    }


@pytest.mark.parametrize("evidence_unit", ["patient_context", "checklist", "key_point"])
def test_generated_station_requires_citation_for_each_evidence_unit(
    evidence_unit: str,
    valid_generated_station: dict[str, object],
    resolved_sources: list[dict[str, object]],
) -> None:
    station = deepcopy(valid_generated_station)
    if evidence_unit == "patient_context":
        station["patient_context"]["citations"] = []  # type: ignore[index]
    elif evidence_unit == "checklist":
        station["checklist_items"][0]["citations"] = []  # type: ignore[index]
    else:
        station["expected_key_points"][0]["citations"] = []  # type: ignore[index]

    with pytest.raises(OsceCitationError, match=evidence_unit.replace("_", " ")):
        normalize_generated_station(station, resolved_sources)


def test_normalization_strips_all_provider_nested_ids(
    valid_generated_station: dict[str, object],
    resolved_sources: list[dict[str, object]],
) -> None:
    station = normalize_generated_station(valid_generated_station, resolved_sources)

    nested_ids = [item.id for item in station.checklist_items]
    for domain in station.rubric_domains:
        nested_ids.append(domain.id)
        nested_ids.extend(level.id for level in domain.levels)
    nested_ids.extend(point.id for point in station.expected_key_points)

    assert all(isinstance(value, UUID) for value in nested_ids)
    assert not {str(value) for value in nested_ids} & {
        "provider-checklist-id",
        "provider-domain-id",
        "provider-level-one",
        "provider-level-two",
        "provider-key-point-id",
    }


def test_prompt_requires_fictional_or_deidentified_patient_data(
    resolved_sources: list[dict[str, object]],
    normalized_sources: list[dict[str, str]],
) -> None:
    prompt = build_osce_generation_prompt(
        evidence=resolved_sources,
        normalized_sources=normalized_sources,
        num_stations=1,
        difficulty="medium",
        focus_topics=[],
    )

    assert "fictional or deidentified" in prompt.lower()
    assert "candidate notes" not in prompt.lower()


@pytest.mark.parametrize(
    ("citation_update", "match"),
    [
        ({"source_id": "missing-note"}, "inaccessible"),
        ({"chunk_id": "missing-chunk"}, "inaccessible"),
        ({"quote": "Unsupported provider quotation"}, "source-inconsistent"),
    ],
)
def test_normalization_rejects_inaccessible_or_inconsistent_citations(
    citation_update: dict[str, str],
    match: str,
    valid_generated_station: dict[str, object],
    resolved_sources: list[dict[str, object]],
) -> None:
    station = deepcopy(valid_generated_station)
    station["patient_context"]["citations"][0].update(citation_update)  # type: ignore[index,union-attr]

    with pytest.raises(OsceCitationError, match=match):
        normalize_generated_station(station, resolved_sources)


def test_verification_units_exclude_candidate_facing_text_and_notes(
    valid_generated_station: dict[str, object],
    resolved_sources: list[dict[str, object]],
) -> None:
    station_payload = deepcopy(valid_generated_station)
    station_payload["candidate_instructions"] = "CANDIDATE-NOTES-SENTINEL"
    station = normalize_generated_station(station_payload, resolved_sources)

    units = build_osce_verification_units([station])
    serialized = repr(units)

    assert len(units) == 3
    assert "CANDIDATE-NOTES-SENTINEL" not in serialized
    assert "candidate_notes" not in serialized


@pytest.mark.asyncio
async def test_test_mode_generation_returns_exact_deterministic_station_count(
    monkeypatch: pytest.MonkeyPatch,
    resolved_sources: list[dict[str, object]],
    normalized_sources: list[dict[str, str]],
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    bundle = await generate_osce_stations_from_sources(
        evidence=resolved_sources,
        normalized_sources=normalized_sources,
        num_stations=2,
        difficulty="medium",
        focus_topics=[],
        model=None,
        api_provider=None,
        verification_provider=None,
        verification_model=None,
    )

    assert len(bundle.stations) == 2
    assert bundle.verification_result.verdict == "grounded"
    assert bundle.provenance["origin"] == "generated"


@pytest.mark.asyncio
async def test_provider_failure_is_bounded_and_does_not_expose_payload(
    monkeypatch: pytest.MonkeyPatch,
    resolved_sources: list[dict[str, object]],
    normalized_sources: list[dict[str, str]],
) -> None:
    secret_body = "raw-provider-body-source-and-note-content"
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")

    async def fail_provider(**_: object) -> object:
        raise RuntimeError(secret_body)

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fail_provider)

    try:
        with pytest.raises(OsceProviderError) as exc_info:
            await generate_osce_stations_from_sources(
                evidence=resolved_sources,
                normalized_sources=normalized_sources,
                num_stations=1,
                difficulty="medium",
                focus_topics=[],
                model="model-id",
                api_provider="provider-id",
                verification_provider=None,
                verification_model=None,
            )
    finally:
        logger.remove(sink_id)

    assert str(exc_info.value) == "osce_provider_failure"
    assert secret_body not in str(exc_info.value)
    assert secret_body not in "\n".join(messages)


@pytest.mark.asyncio
async def test_malformed_provider_output_has_stable_error(
    monkeypatch: pytest.MonkeyPatch,
    resolved_sources: list[dict[str, object]],
    normalized_sources: list[dict[str, str]],
) -> None:
    async def malformed_provider(**_: object) -> object:
        return "not-json"

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", malformed_provider)

    with pytest.raises(OsceMalformedOutputError, match="^osce_malformed_output$"):
        await generate_osce_stations_from_sources(
            evidence=resolved_sources,
            normalized_sources=normalized_sources,
            num_stations=1,
            difficulty="medium",
            focus_topics=[],
            model=None,
            api_provider=None,
            verification_provider=None,
            verification_model=None,
        )


@pytest.mark.asyncio
async def test_provider_station_count_must_match_exact_request(
    monkeypatch: pytest.MonkeyPatch,
    valid_generated_station: dict[str, object],
    resolved_sources: list[dict[str, object]],
    normalized_sources: list[dict[str, str]],
) -> None:
    async def short_provider(**_: object) -> object:
        return {"stations": [valid_generated_station]}

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", short_provider)

    with pytest.raises(OsceMalformedOutputError, match="^osce_malformed_output$"):
        await generate_osce_stations_from_sources(
            evidence=resolved_sources,
            normalized_sources=normalized_sources,
            num_stations=2,
            difficulty="medium",
            focus_topics=[],
            model=None,
            api_provider=None,
            verification_provider=None,
            verification_model=None,
        )


@pytest.mark.asyncio
async def test_non_grounded_verification_has_stable_error_and_receives_only_evidence_units(
    monkeypatch: pytest.MonkeyPatch,
    valid_generated_station: dict[str, object],
    resolved_sources: list[dict[str, object]],
    normalized_sources: list[dict[str, str]],
) -> None:
    captured: dict[str, object] = {}

    async def provider(**_: object) -> object:
        return {"stations": [valid_generated_station]}

    async def verifier(**kwargs: object) -> ArtifactVerificationResult:
        captured.update(kwargs)
        return ArtifactVerificationResult(
            verdict="needs_revision",
            report={"summary": "bounded"},
            unit_results=[],
            metadata={},
        )

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", provider)
    monkeypatch.setattr(osce_generator, "verify_generated_artifact_against_sources", verifier)

    with pytest.raises(OsceVerificationError, match="^osce_verification_failure$"):
        await generate_osce_stations_from_sources(
            evidence=resolved_sources,
            normalized_sources=normalized_sources,
            num_stations=1,
            difficulty="medium",
            focus_topics=[],
            model=None,
            api_provider=None,
            verification_provider=None,
            verification_model=None,
        )

    units = captured["units"]
    assert len(units) == 3  # type: ignore[arg-type]
    assert "candidate_instructions" not in repr(units)
    assert "candidate_notes" not in repr(captured)


@pytest.mark.asyncio
async def test_generation_rejects_resolved_evidence_outside_canonical_source_bundle(
    monkeypatch: pytest.MonkeyPatch,
    valid_generated_station: dict[str, object],
    resolved_sources: list[dict[str, object]],
    normalized_sources: list[dict[str, str]],
) -> None:
    unselected_station = deepcopy(valid_generated_station)
    for citation in [
        *unselected_station["patient_context"]["citations"],  # type: ignore[index]
        *unselected_station["checklist_items"][0]["citations"],  # type: ignore[index]
        *unselected_station["expected_key_points"][0]["citations"],  # type: ignore[index]
    ]:
        citation["source_id"] = "note-2"
    contaminated_evidence = [
        *resolved_sources,
        {
            **resolved_sources[0],
            "source_id": "note-2",
        },
    ]

    async def provider(**_: object) -> object:
        return {"stations": [unselected_station]}

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", provider)

    with pytest.raises(OsceCitationError, match="inaccessible"):
        await generate_osce_stations_from_sources(
            evidence=contaminated_evidence,
            normalized_sources=normalized_sources,
            num_stations=1,
            difficulty="medium",
            focus_topics=[],
            model=None,
            api_provider=None,
            verification_provider=None,
            verification_model=None,
        )
