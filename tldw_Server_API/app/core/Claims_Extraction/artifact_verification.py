"""Internal ClaimsEngine gate for generated Research Workspace artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tldw_Server_API.app.core.Claims_Extraction.analyze_types import ClaimsAnalyzeCallable
from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
    Claim,
    ClaimsEngine,
    classify_claim_type,
)
from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_verification_llm_config,
)
from tldw_Server_API.app.core.Claims_Extraction.verification_report import generate_verification_report
from tldw_Server_API.app.core.RAG.rag_service.types import Document, VerificationStatus


ArtifactVerificationVerdict = Literal["grounded", "needs_revision", "failed"]

_FAILED_STATUSES = {
    VerificationStatus.REFUTED,
    VerificationStatus.HALLUCINATION,
    VerificationStatus.NUMERICAL_ERROR,
    VerificationStatus.MISQUOTED,
    VerificationStatus.CITATION_NOT_FOUND,
}
_NEEDS_REVISION_STATUSES = {
    VerificationStatus.UNVERIFIED,
    VerificationStatus.CONTESTED,
    VerificationStatus.MISLEADING,
}
_MAX_UNITS_BY_ARTIFACT = {
    "quiz": 80,
    "flashcards": 120,
    "data_table": 120,
    "slides": 120,
    "audio_summary": 60,
    "audio_overview": 60,
    "mindmap": 120,
}
_MAX_CLAIMS_PER_UNIT_BY_ARTIFACT = {
    "quiz": 8,
    "flashcards": 6,
    "data_table": 6,
    "slides": 8,
    "audio_summary": 10,
    "audio_overview": 10,
    "mindmap": 6,
}
_MAX_TEXT_CHARS_BY_ARTIFACT = {
    "quiz": 4_000,
    "flashcards": 2_000,
    "data_table": 2_000,
    "slides": 5_000,
    "audio_summary": 6_000,
    "audio_overview": 6_000,
    "mindmap": 3_000,
}


@dataclass
class ArtifactVerificationUnit:
    """Smallest artifact fragment that can be mapped back to a user-visible item."""

    unit_id: str
    text: str
    claims: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactUnitResult:
    """Verification result for one artifact unit."""

    unit_id: str
    verdict: ArtifactVerificationVerdict
    claim_ids: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "verdict": self.verdict,
            "claim_ids": self.claim_ids,
            "statuses": self.statuses,
            "metadata": self.metadata,
        }


@dataclass
class ArtifactVerificationResult:
    """Artifact-level verification report returned to generation services."""

    verdict: ArtifactVerificationVerdict
    report: dict[str, Any]
    unit_results: list[ArtifactUnitResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "report": self.report,
            "unitResults": [unit.to_dict() for unit in self.unit_results],
            "metadata": self.metadata,
        }


def _default_analyze_fn(
    api_endpoint: str | None,
    input_data: Any,
    prompt: str | None,
    api_key: str | None,
    system_message: str | None,
    temp: float | None = None,
    streaming: bool = False,
    recursive_summarization: bool = False,
    chunked_summarization: bool = False,
    chunk_options: Any = None,
    model_override: str | None = None,
    response_format: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """ClaimsEngine-compatible internal LLM call; this does not call Claims HTTP endpoints."""
    from tldw_Server_API.app.core.Claims_Extraction.claims_service import (
        _fva_claims_analyze_call,
    )

    return _fva_claims_analyze_call(
        api_endpoint,
        input_data,
        prompt,
        api_key,
        system_message,
        temp,
        streaming,
        recursive_summarization,
        chunked_summarization,
        chunk_options,
        model_override=model_override,
        response_format=response_format,
        **kwargs,
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bound_verification_analyze_fn(
    analyze_fn: ClaimsAnalyzeCallable,
    *,
    verification_provider: str | None,
    verification_model: str | None,
) -> ClaimsAnalyzeCallable:
    def _call(
        api_endpoint: str | None,
        input_data: Any,
        prompt: str | None,
        api_key: str | None,
        system_message: str | None,
        temp: float | None = None,
        streaming: bool = False,
        recursive_summarization: bool = False,
        chunked_summarization: bool = False,
        chunk_options: Any = None,
        model_override: str | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return analyze_fn(
            verification_provider or api_endpoint,
            input_data,
            prompt,
            api_key,
            system_message,
            temp=temp,
            streaming=streaming,
            recursive_summarization=recursive_summarization,
            chunked_summarization=chunked_summarization,
            chunk_options=chunk_options,
            model_override=verification_model or model_override,
            response_format=response_format,
            **kwargs,
        )

    return _call


def _status_verdict(statuses: list[VerificationStatus]) -> ArtifactVerificationVerdict:
    if any(status in _FAILED_STATUSES for status in statuses):
        return "failed"
    if not statuses or any(status in _NEEDS_REVISION_STATUSES for status in statuses):
        return "needs_revision"
    if all(status == VerificationStatus.VERIFIED for status in statuses):
        return "grounded"
    return "needs_revision"


def _coerce_status(status: Any) -> VerificationStatus:
    if isinstance(status, VerificationStatus):
        return status
    try:
        return VerificationStatus(str(status))
    except ValueError:
        return VerificationStatus.UNVERIFIED


def _empty_report(*, metadata: dict[str, Any], reason: str) -> dict[str, Any]:
    report = generate_verification_report([], metadata={**metadata, "reason": reason}).to_dict()
    report["metadata"] = {**report.get("metadata", {}), "reason": reason}
    return report


def _add_cap_hit(metadata: dict[str, Any], cap_name: str) -> None:
    existing = metadata.get("cap_hit")
    if isinstance(existing, list):
        if cap_name not in existing:
            existing.append(cap_name)
        return
    if isinstance(existing, str):
        metadata["cap_hit"] = sorted({existing, cap_name})
        return
    metadata["cap_hit"] = [cap_name]


def _explicit_claims_for_unit(unit: ArtifactVerificationUnit, *, claims_max: int) -> list[Claim]:
    raw_claims = unit.claims or []
    claims: list[Claim] = []
    for text in raw_claims:
        cleaned = str(text or "").strip()
        if not cleaned:
            continue
        if len(claims) >= claims_max:
            break
        claim_index = len(claims) + 1
        claim_type, extracted_values = classify_claim_type(cleaned)
        claims.append(
            Claim(
                id=f"{unit.unit_id}:c{claim_index}",
                text=cleaned,
                span=None,
                claim_type=claim_type,
                extracted_values=extracted_values,
            )
        )
    return claims


async def _claims_for_unit(
    *,
    unit: ArtifactVerificationUnit,
    engine: ClaimsEngine,
    claims_max: int = 8,
) -> list[Claim]:
    if unit.claims is not None:
        return _explicit_claims_for_unit(unit, claims_max=claims_max)

    extraction = await engine.extract_claims_only(
        unit.text,
        claim_extractor="auto",
        claims_max=claims_max,
    )
    claims: list[Claim] = []
    for claim_index, extracted in enumerate(extraction.claims, start=1):
        cleaned = str(getattr(extracted, "text", "") or "").strip()
        if not cleaned:
            continue
        claim_type = getattr(extracted, "claim_type", None)
        extracted_values = getattr(extracted, "extracted_values", None)
        if claim_type is None or extracted_values is None:
            claim_type, extracted_values = classify_claim_type(cleaned)
        source_id = str(getattr(extracted, "id", "") or f"c{claim_index}").strip() or f"c{claim_index}"
        claims.append(
            Claim(
                id=f"{unit.unit_id}:{source_id}",
                text=cleaned,
                span=getattr(extracted, "span", None),
                claim_type=claim_type,
                extracted_values=extracted_values,
            )
        )
    return claims


async def verify_generated_artifact_against_sources(
    *,
    artifact_type: str,
    units: list[ArtifactVerificationUnit],
    source_documents: list[Document],
    generation_provider: str | None,
    generation_model: str | None,
    verification_provider: str | None = None,
    verification_model: str | None = None,
    generation_context: dict[str, Any] | None = None,
    analyze_fn: ClaimsAnalyzeCallable | None = None,
) -> ArtifactVerificationResult:
    """Verify generated artifact units against explicit source documents."""
    generation_provider = _clean(generation_provider)
    generation_model = _clean(generation_model)
    request_verification_provider = _clean(verification_provider)
    request_verification_model = _clean(verification_model)
    config_verification_provider, config_verification_model = resolve_claims_verification_llm_config()
    configured_verification_provider = request_verification_provider or config_verification_provider
    configured_verification_model = request_verification_model or config_verification_model
    effective_verification_provider = configured_verification_provider or generation_provider
    effective_verification_model = configured_verification_model or generation_model
    verification_source = (
        "request"
        if request_verification_provider or request_verification_model
        else "config"
        if config_verification_provider or config_verification_model
        else "generation"
    )
    differs_from_generation = (
        effective_verification_provider != generation_provider
        or effective_verification_model != generation_model
    )
    metadata: dict[str, Any] = {
        "artifact_type": artifact_type,
        "generation_provider": generation_provider,
        "generation_model": generation_model,
        "verification_provider": effective_verification_provider,
        "verification_model": effective_verification_model,
        "verification_provider_configured": configured_verification_provider is not None,
        "verification_model_configured": configured_verification_model is not None,
        "verification_llm_source": verification_source,
        "verification_llm_is_default": not differs_from_generation,
        "verification_llm_differs_from_generation": differs_from_generation,
    }

    if not source_documents:
        metadata["reason"] = "no_source_documents"
        return ArtifactVerificationResult(
            verdict="failed",
            report=_empty_report(metadata=metadata, reason="no_source_documents"),
            unit_results=[],
            metadata=metadata,
        )

    normalized_artifact_type = (artifact_type or "").strip().lower()
    max_units = _MAX_UNITS_BY_ARTIFACT.get(normalized_artifact_type, 80)
    claims_max = _MAX_CLAIMS_PER_UNIT_BY_ARTIFACT.get(normalized_artifact_type, 8)
    max_text_chars = _MAX_TEXT_CHARS_BY_ARTIFACT.get(normalized_artifact_type, 4_000)
    metadata["claims_max_per_unit"] = claims_max
    metadata["max_text_chars_per_unit"] = max_text_chars
    clean_units = [unit for unit in units if str(unit.unit_id or "").strip() and str(unit.text or "").strip()]
    capped = len(clean_units) > max_units
    if capped:
        clean_units = clean_units[:max_units]
        _add_cap_hit(metadata, "units")

    prepared_units: list[ArtifactVerificationUnit] = []
    for unit in clean_units:
        unit_text = str(unit.text or "").strip()
        unit_metadata = dict(unit.metadata)
        if len(unit_text) > max_text_chars:
            unit_metadata.update(
                {
                    "text_truncated": True,
                    "original_text_length": len(unit_text),
                    "max_text_chars": max_text_chars,
                }
            )
            unit_text = unit_text[:max_text_chars].rstrip()
            capped = True
            _add_cap_hit(metadata, "text")

        if unit.claims is not None:
            claim_count = len([claim for claim in unit.claims if str(claim or "").strip()])
            if claim_count > claims_max:
                unit_metadata.update(
                    {
                        "claims_truncated": True,
                        "original_claim_count": claim_count,
                        "claims_max": claims_max,
                    }
                )
                capped = True
                _add_cap_hit(metadata, "claims")

        prepared_units.append(
            ArtifactVerificationUnit(
                unit_id=unit.unit_id,
                text=unit_text,
                claims=unit.claims,
                metadata=unit_metadata,
            )
        )
    clean_units = prepared_units

    bound_analyze = _bound_verification_analyze_fn(
        analyze_fn or _default_analyze_fn,
        verification_provider=effective_verification_provider,
        verification_model=effective_verification_model,
    )
    engine = ClaimsEngine(bound_analyze)

    claims: list[Claim] = []
    claim_to_unit: dict[str, str] = {}
    for unit in clean_units:
        for claim in await _claims_for_unit(unit=unit, engine=engine, claims_max=claims_max):
            claims.append(claim)
            claim_to_unit[claim.id] = unit.unit_id

    if not claims:
        metadata["reason"] = "no_claims"
        return ArtifactVerificationResult(
            verdict="needs_revision",
            report=_empty_report(metadata=metadata, reason="no_claims"),
            unit_results=[
                ArtifactUnitResult(
                    unit_id=unit.unit_id,
                    verdict="needs_revision",
                    metadata={**unit.metadata, "reason": "no_claims"},
                )
                for unit in clean_units
            ],
            metadata=metadata,
        )

    context = generation_context or {}
    query = str(context.get("query") or artifact_type or "generated artifact")
    answer_text = "\n".join(unit.text for unit in clean_units)
    verification = await engine.verify_claims_only(
        claims=claims,
        query=query,
        documents=source_documents,
        retrieve_fn=None,
        doc_only_mode=True,
    )

    verifications = list(verification.verifications)
    statuses = [_coerce_status(getattr(item, "status", VerificationStatus.UNVERIFIED)) for item in verifications]
    verdict = _status_verdict(statuses)
    if capped and verdict == "grounded":
        verdict = "needs_revision"

    unit_statuses: dict[str, list[VerificationStatus]] = {unit.unit_id: [] for unit in clean_units}
    unit_claim_ids: dict[str, list[str]] = {unit.unit_id: [] for unit in clean_units}
    for item in verifications:
        claim = getattr(item, "claim", None)
        claim_id = str(getattr(claim, "id", "") or "")
        unit_id = claim_to_unit.get(claim_id)
        if unit_id is None:
            continue
        unit_claim_ids[unit_id].append(claim_id)
        unit_statuses[unit_id].append(_coerce_status(getattr(item, "status", VerificationStatus.UNVERIFIED)))

    unit_results: list[ArtifactUnitResult] = []
    for unit in clean_units:
        unit_verdict = _status_verdict(unit_statuses.get(unit.unit_id, []))
        if unit_verdict == "grounded" and (
            unit.metadata.get("text_truncated") or unit.metadata.get("claims_truncated")
        ):
            unit_verdict = "needs_revision"
        unit_results.append(
            ArtifactUnitResult(
                unit_id=unit.unit_id,
                verdict=unit_verdict,
                claim_ids=unit_claim_ids.get(unit.unit_id, []),
                statuses=[status.value for status in unit_statuses.get(unit.unit_id, [])],
                metadata=unit.metadata,
            )
        )
    report = generate_verification_report(
        verifications,
        query=query,
        answer_text=answer_text,
        metadata=metadata,
    ).to_dict()
    return ArtifactVerificationResult(
        verdict=verdict,
        report=report,
        unit_results=unit_results,
        metadata=metadata,
    )


__all__ = [
    "ArtifactUnitResult",
    "ArtifactVerificationResult",
    "ArtifactVerificationUnit",
    "verify_generated_artifact_against_sources",
]
