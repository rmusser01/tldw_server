"""Source-backed generation and verification for OSCE practice stations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.api.v1.schemas.osce import (
    OsceCitation,
    OsceCitationSourceType,
    OsceStationCreateContent,
    OsceStationStoredContent,
)
from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
    ArtifactUnitResult,
    ArtifactVerificationResult,
    ArtifactVerificationUnit,
    verify_generated_artifact_against_sources,
)
from tldw_Server_API.app.core.exceptions import (
    OsceCitationError,
    OsceGenerationError,
    OsceMalformedOutputError,
    OsceProviderError,
    OsceUnsupportedContractError,
    OsceVerificationError,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.services.osce_practice import materialize_station_content

MAX_OSCE_STATIONS = 10
# Match the shared verifier's quiz artifact budget before splitting units by citation set.
MAX_OSCE_VERIFICATION_UNITS = 80
# Bound sequential verifier fan-out to one fifth of the full artifact unit budget.
MAX_OSCE_VERIFICATION_GROUPS = 16
_SUPPORTED_SOURCE_TYPES = {member.value for member in OsceCitationSourceType}


@dataclass(frozen=True)
class GeneratedOsceBundle:
    stations: tuple[OsceStationStoredContent, ...]
    verification_result: ArtifactVerificationResult
    provenance: dict[str, Any]


def _bounded_identifier(value: object | None, *, default: str = "default") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return text[:128]


def _log_failure(code: str, exc: Exception, *, provider: str | None = None) -> None:
    logger.warning(
        "OSCE generation boundary failed code={} provider={} exception={}",
        code,
        _bounded_identifier(provider),
        type(exc).__name__,
    )


def _strip_provider_ids(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_provider_ids(item)
            for key, item in value.items()
            if str(key) != "id"
        }
    if isinstance(value, list):
        return [_strip_provider_ids(item) for item in value]
    return value


def _evidence_candidates(
    evidence: Sequence[dict[str, Any]],
    *,
    source_type: str,
    source_id: str,
) -> list[dict[str, Any]]:
    return [
        item
        for item in evidence
        if str(item.get("source_type") or "").strip() == source_type
        and str(item.get("source_id") or "").strip() == source_id
        and str(item.get("text") or "").strip()
    ]


def _canonicalize_citation(
    value: Any,
    evidence: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OsceCitationError()
    try:
        citation = OsceCitation.model_validate(dict(value))
    except ValidationError as exc:
        raise OsceCitationError() from exc

    source_type = citation.source_type.value
    source_id = citation.source_id
    candidates = _evidence_candidates(
        evidence,
        source_type=source_type,
        source_id=source_id,
    )
    if not candidates:
        raise OsceCitationError("inaccessible citation")

    if citation.chunk_id is not None:
        candidates = [
            item
            for item in candidates
            if str(item.get("chunk_id") or "").strip() == citation.chunk_id
        ]
        if not candidates:
            raise OsceCitationError("inaccessible citation chunk")

    quote = str(citation.quote or "").strip()
    if quote:
        candidates = [
            item for item in candidates if quote in str(item.get("text") or "")
        ]
        if not candidates:
            raise OsceCitationError("source-inconsistent citation quote")
    if len(candidates) != 1:
        raise OsceCitationError("ambiguous citation evidence")

    if citation.source_type is OsceCitationSourceType.MEDIA and citation.media_id is not None:
        if str(citation.media_id) != source_id:
            raise OsceCitationError("source-inconsistent media citation")

    candidate = candidates[0]

    canonical: dict[str, Any] = {
        "source_type": source_type,
        "source_id": source_id,
    }
    label = str(candidate.get("label") or "").strip()
    if label:
        canonical["label"] = label[:200]
    if quote:
        canonical["quote"] = quote

    chunk_id = str(candidate.get("chunk_id") or "").strip()
    if chunk_id and citation.source_type in {
        OsceCitationSourceType.MEDIA,
        OsceCitationSourceType.DOCUMENT,
        OsceCitationSourceType.NOTE,
        OsceCitationSourceType.FLASHCARD_DECK,
        OsceCitationSourceType.FLASHCARD_CARD,
        OsceCitationSourceType.QUIZ_ATTEMPT,
        OsceCitationSourceType.QUIZ_ATTEMPT_QUESTION,
    }:
        canonical["chunk_id"] = chunk_id[:512]

    if citation.source_type is OsceCitationSourceType.MEDIA:
        try:
            canonical["media_id"] = int(source_id)
        except ValueError as exc:
            raise OsceCitationError("source-inconsistent media citation") from exc
        if citation.timestamp_seconds is not None:
            if not _media_timestamp_is_supported(candidate, citation.timestamp_seconds):
                raise OsceCitationError("source-inconsistent media timestamp")
            canonical["timestamp_seconds"] = citation.timestamp_seconds
    elif citation.source_type is OsceCitationSourceType.DOCUMENT:
        candidate_page = candidate.get("page_number")
        if citation.page_number is not None:
            if not _document_page_matches(candidate_page, citation.page_number):
                raise OsceCitationError("source-inconsistent document citation")
            canonical["page_number"] = citation.page_number
    elif citation.source_type is OsceCitationSourceType.URL:
        candidate_url = str(candidate.get("source_url") or "").strip()
        if not candidate_url or candidate_url != citation.source_url:
            raise OsceCitationError("inaccessible URL citation")
        canonical["source_url"] = candidate_url

    try:
        return OsceCitation.model_validate(canonical).model_dump(mode="json", exclude_none=True)
    except ValidationError as exc:
        raise OsceCitationError() from exc


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _media_timestamp_is_supported(candidate: Mapping[str, Any], timestamp: float) -> bool:
    exact = _finite_float(candidate.get("timestamp_seconds"))
    if exact is not None and math.isclose(
        exact,
        timestamp,
        rel_tol=0.0,
        abs_tol=0.001,
    ):
        return True

    for start_key, end_key in (
        ("start_seconds", "end_seconds"),
        ("start_time", "end_time"),
    ):
        start = _finite_float(candidate.get(start_key))
        end = _finite_float(candidate.get(end_key))
        if start is not None and end is not None and start <= timestamp <= end:
            return True
    return False


def _document_page_matches(candidate_page: Any, citation_page: int) -> bool:
    if candidate_page is None or isinstance(candidate_page, bool):
        return False
    try:
        return int(candidate_page) == citation_page and float(candidate_page) == citation_page
    except (TypeError, ValueError):
        return False


def _canonicalize_citations(
    values: Any,
    evidence: Sequence[dict[str, Any]],
    *,
    unit_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise OsceCitationError(f"{unit_name} citation missing")
    return [_canonicalize_citation(value, evidence) for value in values]


def normalize_generated_station(
    raw_station: Any,
    evidence: Sequence[dict[str, Any]],
) -> OsceStationStoredContent:
    """Strip model IDs, resolve citations, and materialize server-owned IDs."""

    if not isinstance(raw_station, Mapping):
        raise OsceUnsupportedContractError()
    payload = _strip_provider_ids(deepcopy(dict(raw_station)))
    try:
        patient_context = payload["patient_context"]
        patient_context["citations"] = _canonicalize_citations(
            patient_context.get("citations"),
            evidence,
            unit_name="patient context",
        )
        checklist_items = payload["checklist_items"]
        for item in checklist_items:
            if not str(item.get("rationale") or "").strip():
                raise OsceCitationError("checklist rationale missing")
            item["citations"] = _canonicalize_citations(
                item.get("citations"),
                evidence,
                unit_name="checklist",
            )
        expected_key_points = payload["expected_key_points"]
        for point in expected_key_points:
            point["citations"] = _canonicalize_citations(
                point.get("citations"),
                evidence,
                unit_name="key point",
            )
    except OsceGenerationError:
        raise
    except (AttributeError, KeyError, TypeError) as exc:
        raise OsceUnsupportedContractError() from exc

    try:
        create_content = OsceStationCreateContent.model_validate(payload)
        return materialize_station_content(create_content)
    except ValidationError as exc:
        raise OsceUnsupportedContractError() from exc


def build_osce_verification_units(
    stations: Sequence[OsceStationStoredContent],
) -> list[ArtifactVerificationUnit]:
    """Build only the evidence-bearing claims from generated stations."""

    units: list[ArtifactVerificationUnit] = []
    for station_index, station in enumerate(stations, start=1):
        units.append(
            ArtifactVerificationUnit(
                unit_id=f"osce-station:{station_index}:patient-context",
                text=station.patient_context.text,
                claims=[station.patient_context.text],
                metadata={
                    "station_index": station_index,
                    "evidence_kind": "patient_context",
                    "citations": [
                        citation.model_dump(mode="json", exclude_none=True)
                        for citation in station.patient_context.citations
                    ],
                },
            )
        )
        for item_index, item in enumerate(station.checklist_items, start=1):
            rationale = str(item.rationale or "").strip()
            units.append(
                ArtifactVerificationUnit(
                    unit_id=f"osce-station:{station_index}:checklist:{item_index}",
                    text=rationale,
                    claims=[rationale],
                    metadata={
                        "station_index": station_index,
                        "evidence_kind": "checklist_rationale",
                        "citations": [
                            citation.model_dump(mode="json", exclude_none=True)
                            for citation in item.citations
                        ],
                    },
                )
            )
        for point_index, point in enumerate(station.expected_key_points, start=1):
            units.append(
                ArtifactVerificationUnit(
                    unit_id=f"osce-station:{station_index}:key-point:{point_index}",
                    text=point.text,
                    claims=[point.text],
                    metadata={
                        "station_index": station_index,
                        "evidence_kind": "expected_key_point",
                        "citations": [
                            citation.model_dump(mode="json", exclude_none=True)
                            for citation in point.citations
                        ],
                    },
                )
            )
    return units


def build_osce_generation_prompt(
    *,
    evidence: Sequence[dict[str, Any]],
    normalized_sources: Sequence[dict[str, str]],
    num_stations: int,
    difficulty: str,
    focus_topics: Sequence[str],
) -> str:
    """Build the provider prompt from canonical resolved source evidence."""

    from tldw_Server_API.app.services import quiz_generator

    content = quiz_generator._build_content_from_evidence(evidence)
    source_contract = quiz_generator._build_source_contract(normalized_sources)
    focus = ", ".join(str(topic).strip() for topic in focus_topics if str(topic).strip())
    focus_instruction = f"- Focus topics: {focus}\n" if focus else ""
    return f"""Generate exactly {num_stations} source-backed OSCE practice stations.

Source evidence:
{content}

Requirements:
- Difficulty: {difficulty}
{focus_instruction}- Use only fictional or deidentified patient data.
- Do not reproduce identifying patient details from the source evidence.
- {source_contract.removeprefix('- ')}
- Every patient_context assertion, checklist rationale, and expected key point must have at least one citation.
- Citation quotes must be exact excerpts from the cited source evidence.
- Checklist rationales are required.
- Return no IDs; all IDs are assigned by the server.
- Do not include scores, pass thresholds, grading, generated feedback, or hidden reasoning.

Return only JSON with one top-level key named "stations". Each station must match:
{{
  "schema_version": "osce.station.v1",
  "title": "...",
  "candidate_instructions": "...",
  "candidate_task": "...",
  "patient_context": {{"text": "...", "citations": [{{"source_type": "note", "source_id": "...", "chunk_id": "...", "quote": "..."}}]}},
  "recommended_duration_seconds": 480,
  "checklist_items": [{{"label": "...", "rationale": "...", "citations": [{{"source_type": "note", "source_id": "...", "chunk_id": "...", "quote": "..."}}]}}],
  "rubric_domains": [{{"label": "Communication", "levels": [{{"label": "Needs development", "description": "..."}}, {{"label": "Effective", "description": "..."}}]}}],
  "expected_key_points": [{{"text": "...", "citations": [{{"source_type": "note", "source_id": "...", "chunk_id": "...", "quote": "..."}}]}}]
}}
"""


def _citation_for_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    source_type = str(item.get("source_type") or "").strip()
    source_id = str(item.get("source_id") or "").strip()
    if source_type not in _SUPPORTED_SOURCE_TYPES or not source_id:
        raise OsceUnsupportedContractError()
    source_text = str(item.get("text") or "").strip()
    if not source_text:
        raise OsceUnsupportedContractError()
    citation: dict[str, Any] = {
        "source_type": source_type,
        "source_id": source_id,
        "quote": source_text[:1000],
    }
    chunk_id = str(item.get("chunk_id") or "").strip()
    if chunk_id:
        citation["chunk_id"] = chunk_id
    if source_type == "media":
        try:
            citation["media_id"] = int(source_id)
        except ValueError as exc:
            raise OsceUnsupportedContractError() from exc
    if source_type == "url":
        source_url = str(item.get("source_url") or "").strip()
        if not source_url:
            raise OsceUnsupportedContractError()
        citation["source_url"] = source_url
    return citation


def _build_test_mode_station(
    *,
    evidence: Sequence[dict[str, Any]],
    station_index: int,
) -> dict[str, Any]:
    supported = [
        item
        for item in evidence
        if str(item.get("source_type") or "").strip() in _SUPPORTED_SOURCE_TYPES
        and str(item.get("text") or "").strip()
    ]
    if not supported:
        raise OsceUnsupportedContractError()
    item = supported[(station_index - 1) % len(supported)]
    citation = _citation_for_evidence(item)
    evidence_text = " ".join(str(item.get("text") or "").split()).strip()[:2000]
    return {
        "schema_version": "osce.station.v1",
        "title": f"Source-backed OSCE station {station_index}",
        "candidate_instructions": "You are speaking with a simulated patient.",
        "candidate_task": "Explain the source-backed safety point and check understanding.",
        "patient_context": {
            "text": "A fictional adult is seeking guidance related to the selected source.",
            "citations": [deepcopy(citation)],
        },
        "recommended_duration_seconds": 480,
        "checklist_items": [
            {
                "label": "Explains the source-backed safety point",
                "rationale": evidence_text,
                "citations": [deepcopy(citation)],
            }
        ],
        "rubric_domains": [
            {
                "label": "Communication",
                "levels": [
                    {
                        "label": "Needs development",
                        "description": "The explanation is incomplete or unclear.",
                    },
                    {
                        "label": "Effective",
                        "description": "The explanation is clear and checks understanding.",
                    },
                ],
            }
        ],
        "expected_key_points": [
            {
                "text": evidence_text,
                "citations": [deepcopy(citation)],
            }
        ],
    }


_ORIGINAL_VERIFY_GENERATED_ARTIFACT = verify_generated_artifact_against_sources


def _documents_for_unit(
    unit: ArtifactVerificationUnit,
    source_documents: Sequence[Document],
) -> tuple[Document, ...]:
    citations = unit.metadata.get("citations")
    if not isinstance(citations, list) or not citations:
        raise OsceVerificationError()

    selected: dict[str, Document] = {}
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise OsceVerificationError()
        source_type = str(citation.get("source_type") or "").strip()
        source_id = str(citation.get("source_id") or "").strip()
        chunk_id = str(citation.get("chunk_id") or "").strip()
        matches = [
            document
            for document in source_documents
            if str(document.metadata.get("source_type") or "").strip() == source_type
            and str(document.metadata.get("source_id") or "").strip() == source_id
            and (
                not chunk_id
                or str(document.metadata.get("chunk_id") or "").strip() == chunk_id
            )
        ]
        if len(matches) != 1:
            raise OsceVerificationError()
        selected[matches[0].id] = matches[0]
    if not selected:
        raise OsceVerificationError()
    return tuple(selected[document_id] for document_id in sorted(selected))


def _group_units_by_cited_documents(
    units: Sequence[ArtifactVerificationUnit],
    source_documents: Sequence[Document],
) -> list[tuple[list[ArtifactVerificationUnit], tuple[Document, ...]]]:
    groups: dict[tuple[str, ...], tuple[list[ArtifactVerificationUnit], tuple[Document, ...]]] = {}
    for unit in units:
        documents = _documents_for_unit(unit, source_documents)
        document_ids = tuple(document.id for document in documents)
        if document_ids not in groups:
            groups[document_ids] = ([], documents)
        groups[document_ids][0].append(unit)
    return list(groups.values())


def _test_mode_verification_result(
    units: Sequence[ArtifactVerificationUnit],
) -> ArtifactVerificationResult:
    unit_results = [
        ArtifactUnitResult(
            unit_id=unit.unit_id,
            verdict="grounded",
            claim_ids=[f"{unit.unit_id}:c1"],
            statuses=["verified"],
            metadata=dict(unit.metadata),
        )
        for unit in units
    ]
    return ArtifactVerificationResult(
        verdict="grounded",
        report={"verified_units": len(unit_results), "test_mode": True},
        unit_results=unit_results,
        metadata={"artifact_type": "quiz", "test_mode": True},
    )


def _require_complete_grounded_result(
    result: ArtifactVerificationResult,
    expected_units: Sequence[ArtifactVerificationUnit],
) -> None:
    expected_ids = [unit.unit_id for unit in expected_units]
    actual_ids = [unit.unit_id for unit in result.unit_results]
    if len(expected_ids) != len(set(expected_ids)):
        raise OsceVerificationError()
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise OsceVerificationError()
    if result.metadata.get("cap_hit"):
        raise OsceVerificationError()
    if result.verdict != "grounded":
        raise OsceVerificationError()
    for unit_result in result.unit_results:
        if unit_result.verdict != "grounded":
            raise OsceVerificationError()
        if unit_result.metadata.get("text_truncated") or unit_result.metadata.get(
            "claims_truncated"
        ):
            raise OsceVerificationError()
        if not unit_result.claim_ids or len(unit_result.claim_ids) != len(unit_result.statuses):
            raise OsceVerificationError()
        if any(status != "verified" for status in unit_result.statuses):
            raise OsceVerificationError()


async def _verify_stations(
    *,
    stations: Sequence[OsceStationStoredContent],
    evidence: Sequence[dict[str, Any]],
    generation_provider: str | None,
    generation_model: str | None,
    verification_provider: str | None,
    verification_model: str | None,
) -> ArtifactVerificationResult:
    units = build_osce_verification_units(stations)
    if len(units) > MAX_OSCE_VERIFICATION_UNITS:
        raise OsceVerificationError()

    from tldw_Server_API.app.services import quiz_generator

    source_documents = quiz_generator._build_quiz_source_documents(evidence)
    groups = _group_units_by_cited_documents(units, source_documents)
    if len(groups) > MAX_OSCE_VERIFICATION_GROUPS:
        raise OsceVerificationError()

    group_results: list[ArtifactVerificationResult] = []
    group_reports: list[dict[str, Any]] = []
    unit_results_by_id: dict[str, ArtifactUnitResult] = {}
    for grouped_units, cited_documents in groups:
        if (
            is_test_mode()
            and verify_generated_artifact_against_sources
            is _ORIGINAL_VERIFY_GENERATED_ARTIFACT
        ):
            result = _test_mode_verification_result(grouped_units)
        else:
            result = await verify_generated_artifact_against_sources(
                artifact_type="quiz",
                units=grouped_units,
                source_documents=list(cited_documents),
                generation_provider=generation_provider,
                generation_model=generation_model,
                verification_provider=verification_provider,
                verification_model=verification_model,
                generation_context={"query": "generated OSCE evidence claims"},
            )
        _require_complete_grounded_result(result, grouped_units)
        group_results.append(result)
        group_reports.append(
            {
                "cited_document_ids": [document.id for document in cited_documents],
                "verdict": result.verdict,
                "report": result.report,
            }
        )
        unit_results_by_id.update(
            {unit_result.unit_id: unit_result for unit_result in result.unit_results}
        )

    if set(unit_results_by_id) != {unit.unit_id for unit in units}:
        raise OsceVerificationError()
    ordered_results = [unit_results_by_id[unit.unit_id] for unit in units]
    return ArtifactVerificationResult(
        verdict="grounded",
        report={
            "total_units": len(units),
            "verified_units": len(ordered_results),
            "groups": group_reports,
        },
        unit_results=ordered_results,
        metadata={
            "artifact_type": "quiz",
            "verification_group_count": len(group_results),
            "group_metadata": [result.metadata for result in group_results],
        },
    )


async def generate_osce_stations_from_sources(
    *,
    evidence: Sequence[dict[str, Any]],
    normalized_sources: Sequence[dict[str, str]],
    num_stations: int,
    difficulty: str,
    focus_topics: Sequence[str],
    model: str | None,
    api_provider: str | None,
    verification_provider: str | None,
    verification_model: str | None,
) -> GeneratedOsceBundle:
    """Generate, normalize, cite, and verify all stations before persistence."""

    if not 1 <= num_stations <= MAX_OSCE_STATIONS:
        raise OsceUnsupportedContractError()
    if not evidence or not normalized_sources:
        raise OsceCitationError()

    allowed_sources = {
        (
            str(source.get("source_type") or "").strip(),
            str(source.get("source_id") or "").strip(),
        )
        for source in normalized_sources
    }
    if any(source_type not in _SUPPORTED_SOURCE_TYPES for source_type, _ in allowed_sources):
        raise OsceUnsupportedContractError()
    canonical_evidence = [
        item
        for item in evidence
        if (
            str(item.get("source_type") or "").strip(),
            str(item.get("source_id") or "").strip(),
        )
        in allowed_sources
    ]
    accessible_sources = {
        (
            str(item.get("source_type") or "").strip(),
            str(item.get("source_id") or "").strip(),
        )
        for item in canonical_evidence
        if str(item.get("text") or "").strip()
    }
    if accessible_sources != allowed_sources:
        raise OsceCitationError("inaccessible canonical source")

    from tldw_Server_API.app.services import quiz_generator

    if quiz_generator._should_use_deterministic_test_mode():
        raw_stations = [
            _build_test_mode_station(evidence=canonical_evidence, station_index=index)
            for index in range(1, num_stations + 1)
        ]
    else:
        prompt = build_osce_generation_prompt(
            evidence=canonical_evidence,
            normalized_sources=normalized_sources,
            num_stations=num_stations,
            difficulty=difficulty,
            focus_topics=focus_topics,
        )
        try:
            raw_response = await quiz_generator._call_quiz_generation_llm(
                prompt=prompt,
                model=model,
                api_provider=api_provider,
                max_tokens=min(12000, max(3000, num_stations * 1200)),
            )
        except Exception as exc:
            _log_failure(OsceProviderError.code, exc, provider=api_provider)
            raise OsceProviderError() from exc
        try:
            content = quiz_generator.extract_response_content(raw_response)
            payload = quiz_generator._extract_json_payload(
                content if content is not None else raw_response
            )
        except Exception as exc:
            _log_failure(OsceMalformedOutputError.code, exc, provider=api_provider)
            raise OsceMalformedOutputError() from exc
        raw_stations = payload.get("stations") if isinstance(payload, Mapping) else None
        if not isinstance(raw_stations, list) or len(raw_stations) != num_stations:
            raise OsceMalformedOutputError()

    if len(raw_stations) != num_stations:
        raise OsceMalformedOutputError()
    stations = tuple(
        normalize_generated_station(station, canonical_evidence)
        for station in raw_stations
    )
    if len(stations) != num_stations:
        raise OsceMalformedOutputError()

    generation_provider = api_provider or DEFAULT_LLM_PROVIDER
    try:
        verification_result = await _verify_stations(
            stations=stations,
            evidence=canonical_evidence,
            generation_provider=generation_provider,
            generation_model=model,
            verification_provider=verification_provider,
            verification_model=verification_model,
        )
    except OsceGenerationError:
        raise
    except Exception as exc:
        _log_failure(OsceVerificationError.code, exc, provider=verification_provider)
        raise OsceVerificationError() from exc
    if verification_result.verdict != "grounded":
        raise OsceVerificationError()

    provenance = {
        "origin": "generated",
        "generation_provider": _bounded_identifier(generation_provider),
        "generation_model": _bounded_identifier(model, default="configured-default"),
        "verification_provider": _bounded_identifier(
            verification_provider or generation_provider
        ),
        "verification_model": _bounded_identifier(
            verification_model or model,
            default="configured-default",
        ),
        "verification_verdict": "grounded",
        "verification_unit_count": len(build_osce_verification_units(stations)),
    }
    return GeneratedOsceBundle(
        stations=stations,
        verification_result=verification_result,
        provenance=provenance,
    )


__all__ = [
    "GeneratedOsceBundle",
    "build_osce_generation_prompt",
    "build_osce_verification_units",
    "generate_osce_stations_from_sources",
    "normalize_generated_station",
]
