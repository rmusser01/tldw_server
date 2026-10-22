"""
adjudicator.py - Evidence weighing and final verdict determination for FVA.

This module implements the adjudicator component of the FVA pipeline. It weighs
supporting vs contradicting evidence to determine a final verification status,
including the new CONTESTED status for claims with conflicting evidence.

Inspired by FVA-RAG paper (arXiv:2512.07015).
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger
from tldw_Server_API.app.core.Claims_Extraction.analyze_types import (
    AdjudicatorAnalyzeCallable,
)
from tldw_Server_API.app.core.Claims_Extraction.claims_engine import _run_bounded_claims_analyze
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.RAG.rag_service.types import (
    Document,
    VerificationStatus,
)
from tldw_Server_API.app.core.Claims_Extraction.output_parser import (
    ClaimsOutputParseError,
    coerce_llm_response_text,
    parse_claims_llm_output,
    resolve_claims_response_format,
)
from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_json_parse_mode,
    resolve_claims_llm_config,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
        Claim,
        ClaimVerification,
    )


_ADJUDICATOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stance": {
            "type": "string",
            "enum": ["SUPPORTS", "CONTRADICTS", "NEUTRAL"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["stance", "confidence"],
    "additionalProperties": True,
}


class EvidenceStance(str, Enum):
    """Stance of evidence toward a claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


@dataclass
class EvidenceAssessment:
    """Assessment of a single document's stance toward a claim."""

    document: Document
    stance: EvidenceStance
    confidence: float
    rationale: str | None = None  # Use 'rationale' to match existing ClaimVerification


@dataclass
class AdjudicationResult:
    """Result of adjudicating between supporting and contradicting evidence."""

    final_status: VerificationStatus
    support_score: float  # 0-1
    contradict_score: float  # 0-1
    supporting_evidence: list[EvidenceAssessment]
    contradicting_evidence: list[EvidenceAssessment]
    adjudication_rationale: str  # Use 'rationale' to match existing pattern
    contestation_score: float = 0.0  # 0 = one-sided, 1 = perfectly balanced

    def __post_init__(self):
        """Calculate contestation score after initialization."""
        self._calculate_contestation_score()

    def _calculate_contestation_score(self) -> None:
        """Calculate how contested the evidence is (0 = one-sided, 1 = balanced)."""
        total = self.support_score + self.contradict_score
        if total > 0:
            min_score = min(self.support_score, self.contradict_score)
            max_score = max(self.support_score, self.contradict_score)
            if max_score > 0:
                self.contestation_score = min_score / max_score
            else:
                self.contestation_score = 0.0
        else:
            self.contestation_score = 0.0


class ClaimAdjudicator:
    """
    Weighs supporting vs contradicting evidence to reach a final verdict.

    This component implements the adjudication logic from FVA-RAG, determining
    whether a claim should be VERIFIED, REFUTED, or CONTESTED based on the
    balance of evidence.
    """

    def __init__(
        self,
        nli_pipeline: Any | None = None,
        llm_analyze_fn: AdjudicatorAnalyzeCallable | None = None,
        contested_threshold: float = 0.4,
        min_contradict_score: float = 0.1,
        strong_contradict_threshold: float = 0.7,
        weak_support_threshold: float = 0.4,
    ):
        """
        Initialize the adjudicator.

        Args:
            nli_pipeline: Transformers NLI pipeline (from claims_engine)
            llm_analyze_fn: LLM analyze function for fallback
            contested_threshold: Ratio threshold for CONTESTED status (0.4 means 40-60% split)
            min_contradict_score: Minimum contradict score to consider evidence
            strong_contradict_threshold: Threshold for strong contradiction
            weak_support_threshold: Threshold for weak support
        """
        self.nli_pipeline = nli_pipeline
        self.llm_analyze_fn = llm_analyze_fn
        self.contested_threshold = contested_threshold
        self.min_contradict_score = min_contradict_score
        self.strong_contradict_threshold = strong_contradict_threshold
        self.weak_support_threshold = weak_support_threshold

    async def adjudicate(
        self,
        claim: Claim,
        supporting_docs: list[Document],
        contradicting_docs: list[Document],
        original_verification: ClaimVerification,
    ) -> AdjudicationResult:
        """
        Given both supporting and potentially contradicting documents,
        determine final verification status.

        Args:
            claim: The claim being adjudicated
            supporting_docs: Documents from original retrieval
            contradicting_docs: Documents from anti-context retrieval
            original_verification: Initial verification result

        Returns:
            AdjudicationResult with final status and evidence assessments
        """
        # Assess stance of each document
        support_assessments = await self._assess_documents(
            claim, supporting_docs, expected_stance=EvidenceStance.SUPPORTS
        )
        contradict_assessments = await self._assess_documents(
            claim, contradicting_docs, expected_stance=EvidenceStance.CONTRADICTS
        )

        # Calculate aggregate scores
        support_score = self._aggregate_score(
            [a for a in support_assessments if a.stance == EvidenceStance.SUPPORTS]
        )
        contradict_score = self._aggregate_score(
            [a for a in contradict_assessments if a.stance == EvidenceStance.CONTRADICTS]
        )

        # Determine final status
        final_status, rationale = self._determine_status(
            support_score,
            contradict_score,
            original_verification,
        )

        return AdjudicationResult(
            final_status=final_status,
            support_score=support_score,
            contradict_score=contradict_score,
            supporting_evidence=[
                a for a in support_assessments if a.stance == EvidenceStance.SUPPORTS
            ],
            contradicting_evidence=[
                a for a in contradict_assessments if a.stance == EvidenceStance.CONTRADICTS
            ],
            adjudication_rationale=rationale,
        )

    async def _assess_documents(
        self,
        claim: Claim,
        documents: list[Document],
        expected_stance: EvidenceStance,
    ) -> list[EvidenceAssessment]:
        """Assess each document's actual stance toward the claim."""
        assessments: list[EvidenceAssessment] = []

        for doc in documents:
            try:
                # Use NLI pipeline if available (matches claims_engine pattern)
                if self.nli_pipeline:
                    stance, confidence = await self._nli_assess(claim.text, doc.content)
                elif self.llm_analyze_fn:
                    stance, confidence = await self._llm_assess(claim.text, doc.content)
                else:
                    # No model available - assume expected stance with low confidence
                    stance = expected_stance
                    confidence = 0.5

                assessments.append(
                    EvidenceAssessment(
                        document=doc,
                        stance=stance,
                        confidence=confidence,
                    )
                )
            except LLMQuotaExceededError:
                raise  # the user's limit: not a document that could not be assessed
            except Exception as e:
                logger.warning(f"Failed to assess document {doc.id}: {e}")
                continue

        return assessments

    async def _nli_assess(
        self, claim: str, evidence: str
    ) -> tuple[EvidenceStance, float]:
        """
        Use NLI pipeline to assess evidence stance.

        Matches the transformers pipeline interface used in claims_engine.py.
        """
        # Format for transformers NLI: "premise </s></s> hypothesis"
        # Truncate to avoid token limits
        evidence_truncated = evidence[:1000] if len(evidence) > 1000 else evidence
        input_text = f"{evidence_truncated} </s></s> {claim}"

        # Run in executor since transformers is sync
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: self.nli_pipeline(input_text)
            )
        except Exception as e:
            logger.warning(f"NLI assessment failed: {e}")
            return EvidenceStance.NEUTRAL, 0.5

        # results is [[{label, score}, ...]] or [{label, score}, ...]
        if not results:
            return EvidenceStance.NEUTRAL, 0.5

        # Handle nested list structure
        result_list = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results

        if not result_list:
            return EvidenceStance.NEUTRAL, 0.5

        scores = {r["label"].lower(): r["score"] for r in result_list}

        entailment = scores.get("entailment", 0)
        contradiction = scores.get("contradiction", 0)
        neutral = scores.get("neutral", 0)

        if entailment > contradiction and entailment > neutral:
            return EvidenceStance.SUPPORTS, entailment
        elif contradiction > entailment and contradiction > neutral:
            return EvidenceStance.CONTRADICTS, contradiction
        return EvidenceStance.NEUTRAL, neutral

    async def _invoke_llm_assess_call(
        self,
        *,
        prompt: str,
        provider: str,
        model_override: str | None,
        temperature: float,
        response_format: dict[str, Any] | None,
    ) -> str:
        fn = self.llm_analyze_fn
        if fn is None:
            return ""

        system = (
            "You classify whether evidence supports, contradicts, or is neutral toward a claim. "
            "Return strict JSON only."
        )
        extended_error: Exception | None = None

        try:
            # Off the loop: a sync LLM call on the loop's thread blocks it and cannot be quota-checked.
            response = await _run_bounded_claims_analyze(
                fn,
                provider,
                prompt,
                prompt,
                None,
                system,
                temperature,
                streaming=False,
                recursive_summarization=False,
                chunked_summarization=False,
                chunk_options=None,
                model_override=model_override,
                response_format=response_format,
                max_tokens=300,
            )
            if inspect.isawaitable(response):
                response = await response
            text = coerce_llm_response_text(response)
            if text.strip():
                return text
        except TypeError as exc:
            # Most likely callback expects the simple prompt-only signature.
            extended_error = exc

        try:
            fallback = await _run_bounded_claims_analyze(fn, prompt)
            if inspect.isawaitable(fallback):
                fallback = await fallback
            return coerce_llm_response_text(fallback)
        except Exception as fallback_exc:
            if extended_error is not None:
                raise fallback_exc from extended_error
            raise

    async def _llm_assess(
        self, claim: str, evidence: str
    ) -> tuple[EvidenceStance, float]:
        """Fallback: Use LLM to assess evidence stance."""
        prompt = f"""Assess whether the following evidence SUPPORTS, CONTRADICTS, or is NEUTRAL toward the claim.

Claim: {claim}

Evidence: {evidence[:1500]}

Respond with a JSON object:
{{"stance": "SUPPORTS" | "CONTRADICTS" | "NEUTRAL", "confidence": 0.0-1.0}}"""

        try:
            provider, model_override, temperature = resolve_claims_llm_config(default_temperature=0.1)
            parse_mode = resolve_claims_json_parse_mode(default_mode="lenient")
            response_format = resolve_claims_response_format(
                provider,
                schema_name="claims_adjudication",
                json_schema=_ADJUDICATOR_RESPONSE_SCHEMA,
            )
            response_text = await self._invoke_llm_assess_call(
                prompt=prompt,
                provider=provider,
                model_override=model_override,
                temperature=temperature,
                response_format=response_format,
            )
            parsed = parse_claims_llm_output(
                response_text,
                parse_mode=parse_mode,
                strip_think_tags=True,
            )
            if not isinstance(parsed, dict):
                raise ClaimsOutputParseError(
                    f"Expected object payload from adjudicator LLM output, got {type(parsed).__name__}."
                )

            stance_str = str(parsed.get("stance", "NEUTRAL")).strip().upper()
            confidence_raw = parsed.get("confidence", 0.5)
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            stance_map = {
                "SUPPORTS": EvidenceStance.SUPPORTS,
                "CONTRADICTS": EvidenceStance.CONTRADICTS,
                "NEUTRAL": EvidenceStance.NEUTRAL,
            }
            return stance_map.get(stance_str, EvidenceStance.NEUTRAL), confidence
        except LLMQuotaExceededError:
            raise  # the user's limit: not a neutral stance
        except Exception as e:
            logger.warning(f"LLM assessment failed: {e}")
            return EvidenceStance.NEUTRAL, 0.5

    def _aggregate_score(self, assessments: list[EvidenceAssessment]) -> float:
        """
        Aggregate multiple evidence assessments into a single score.

        Uses confidence-weighted averaging with optional authority boost from
        document metadata.
        """
        if not assessments:
            return 0.0

        # Weight by confidence and source authority
        total_weight = 0.0
        weighted_sum = 0.0

        for assessment in assessments:
            # Get authority from document metadata if available
            authority = assessment.document.metadata.get("authority_score", 1.0)
            if not isinstance(authority, (int, float)):
                authority = 1.0

            weight = assessment.confidence * float(authority)
            weighted_sum += weight
            total_weight += float(authority)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_status(
        self,
        support_score: float,
        contradict_score: float,
        original: ClaimVerification,
    ) -> tuple[VerificationStatus, str]:
        """
        Determine final verification status based on evidence balance.

        Returns:
            Tuple of (final_status, rationale)
        """
        # No contradicting evidence found - keep original
        if contradict_score < self.min_contradict_score:
            return original.status, "No significant contradicting evidence found."

        # Strong contradiction, weak support -> REFUTED
        if (
            contradict_score > self.strong_contradict_threshold
            and support_score < self.weak_support_threshold
        ):
            return VerificationStatus.REFUTED, (
                f"Strong contradicting evidence (score={contradict_score:.2f}) "
                f"outweighs support (score={support_score:.2f})."
            )

        # Both have significant evidence -> check if CONTESTED
        total = support_score + contradict_score
        ratio = support_score / total if total > 0 else 0.5

        if self.contested_threshold < ratio < (1 - self.contested_threshold):
            return VerificationStatus.CONTESTED, (
                f"Evidence is contested: support={support_score:.2f}, "
                f"contradict={contradict_score:.2f}, ratio={ratio:.2f}."
            )

        # Support dominates -> keep VERIFIED (if original was verified)
        if ratio >= (1 - self.contested_threshold):
            if original.status == VerificationStatus.VERIFIED:
                return VerificationStatus.VERIFIED, (
                    f"Supporting evidence ({support_score:.2f}) dominates "
                    f"despite some contradiction ({contradict_score:.2f})."
                )
            # If original wasn't verified, keep original status
            return original.status, (
                f"Support dominates but original status was not VERIFIED. "
                f"Support={support_score:.2f}, contradict={contradict_score:.2f}."
            )

        # Contradiction dominates
        if ratio <= self.contested_threshold:
            return VerificationStatus.REFUTED, (
                f"Contradicting evidence ({contradict_score:.2f}) dominates "
                f"over support ({support_score:.2f})."
            )

        # Default: keep original with note
        return original.status, (
            f"Adjudication inconclusive. Support={support_score:.2f}, "
            f"Contradict={contradict_score:.2f}. Keeping original status."
        )


def create_adjudicator(
    nli_pipeline: Any | None = None,
    llm_analyze_fn: AdjudicatorAnalyzeCallable | None = None,
    contested_threshold: float = 0.4,
) -> ClaimAdjudicator:
    """
    Factory function to create a ClaimAdjudicator instance.

    Args:
        nli_pipeline: Optional NLI pipeline for stance assessment
        llm_analyze_fn: Optional LLM function for fallback assessment
        contested_threshold: Threshold for CONTESTED status

    Returns:
        Configured ClaimAdjudicator instance
    """
    return ClaimAdjudicator(
        nli_pipeline=nli_pipeline,
        llm_analyze_fn=llm_analyze_fn,
        contested_threshold=contested_threshold,
    )
