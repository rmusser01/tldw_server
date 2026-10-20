"""
Tests for the adjudicator module.

These tests verify the evidence weighing and status determination logic
for the FVA pipeline's adjudicator component.
"""

import asyncio
import threading

import pytest
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import tldw_Server_API.app.core.Claims_Extraction.adjudicator as adjudicator_module
from tldw_Server_API.app.core.Claims_Extraction.adjudicator import (
    ClaimAdjudicator,
    EvidenceStance,
    EvidenceAssessment,
    AdjudicationResult,
    create_adjudicator,
)
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal

# Import types
try:
    from tldw_Server_API.app.core.RAG.rag_service.types import (
        VerificationStatus,
        Document,
        MatchLevel,
        SourceAuthority,
    )
except ImportError:
    from enum import Enum

    class VerificationStatus(Enum):
        VERIFIED = "verified"
        REFUTED = "refuted"
        CONTESTED = "contested"
        UNVERIFIED = "unverified"

    class MatchLevel(Enum):
        EXACT = "exact"

    class SourceAuthority(Enum):
        SECONDARY = 1

    @dataclass
    class Document:
        id: str
        content: str
        metadata: dict[str, Any] = field(default_factory=dict)
        score: float = 0.0


@dataclass
class MockClaim:
    """Mock claim for testing."""
    id: str
    text: str
    claim_type: Any = None
    extracted_values: dict[str, Any] = field(default_factory=dict)
    span: tuple[int, int] | None = None


@dataclass
class MockClaimVerification:
    """Mock claim verification for testing."""
    claim: MockClaim
    status: VerificationStatus = VerificationStatus.VERIFIED
    confidence: float = 0.8
    evidence: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    rationale: str | None = None
    match_level: MatchLevel = MatchLevel.EXACT
    source_authority: SourceAuthority = SourceAuthority.SECONDARY
    requires_external_knowledge: bool = False


class TestEvidenceStance:
    """Tests for EvidenceStance enum."""

    @pytest.mark.unit
    def test_stance_values(self):
        """EvidenceStance should have expected values."""
        assert EvidenceStance.SUPPORTS.value == "supports"
        assert EvidenceStance.CONTRADICTS.value == "contradicts"
        assert EvidenceStance.NEUTRAL.value == "neutral"

    @pytest.mark.unit
    def test_stance_is_string_enum(self):
        """EvidenceStance should be a string enum."""
        assert isinstance(EvidenceStance.SUPPORTS, str)
        assert EvidenceStance.SUPPORTS == "supports"


class TestEvidenceAssessment:
    """Tests for EvidenceAssessment dataclass."""

    @pytest.mark.unit
    def test_assessment_fields(self):
        """EvidenceAssessment should have expected fields."""
        doc = Document(id="1", content="Test", metadata={}, score=0.8)
        assessment = EvidenceAssessment(
            document=doc,
            stance=EvidenceStance.SUPPORTS,
            confidence=0.9,
            rationale="Strong support",
        )

        assert assessment.document == doc
        assert assessment.stance == EvidenceStance.SUPPORTS
        assert assessment.confidence == 0.9
        assert assessment.rationale == "Strong support"

    @pytest.mark.unit
    def test_assessment_optional_rationale(self):
        """Rationale should be optional."""
        doc = Document(id="1", content="Test", metadata={}, score=0.8)
        assessment = EvidenceAssessment(
            document=doc,
            stance=EvidenceStance.NEUTRAL,
            confidence=0.5,
        )

        assert assessment.rationale is None


class TestAdjudicationResult:
    """Tests for AdjudicationResult dataclass."""

    @pytest.mark.unit
    def test_result_fields(self):
        """AdjudicationResult should have expected fields."""
        result = AdjudicationResult(
            final_status=VerificationStatus.CONTESTED,
            support_score=0.6,
            contradict_score=0.5,
            supporting_evidence=[],
            contradicting_evidence=[],
            adjudication_rationale="Evidence is balanced",
        )

        assert result.final_status == VerificationStatus.CONTESTED
        assert result.support_score == 0.6
        assert result.contradict_score == 0.5
        assert result.adjudication_rationale == "Evidence is balanced"

    @pytest.mark.unit
    def test_contestation_score_balanced(self):
        """Contestation score should be ~1 for balanced evidence."""
        result = AdjudicationResult(
            final_status=VerificationStatus.CONTESTED,
            support_score=0.5,
            contradict_score=0.5,
            supporting_evidence=[],
            contradicting_evidence=[],
            adjudication_rationale="Balanced",
        )

        assert result.contestation_score == 1.0

    @pytest.mark.unit
    def test_contestation_score_one_sided(self):
        """Contestation score should be ~0 for one-sided evidence."""
        result = AdjudicationResult(
            final_status=VerificationStatus.VERIFIED,
            support_score=0.9,
            contradict_score=0.1,
            supporting_evidence=[],
            contradicting_evidence=[],
            adjudication_rationale="Support dominates",
        )

        # 0.1 / 0.9 ≈ 0.111
        assert result.contestation_score < 0.2

    @pytest.mark.unit
    def test_contestation_score_zero_evidence(self):
        """Contestation score should be 0 when no evidence."""
        result = AdjudicationResult(
            final_status=VerificationStatus.UNVERIFIED,
            support_score=0.0,
            contradict_score=0.0,
            supporting_evidence=[],
            contradicting_evidence=[],
            adjudication_rationale="No evidence",
        )

        assert result.contestation_score == 0.0


class TestClaimAdjudicatorStatusDetermination:
    """Tests for status determination logic."""

    @pytest.mark.unit
    def test_no_contradiction_keeps_original(self):
        """Should keep original status when no contradicting evidence."""
        adjudicator = ClaimAdjudicator()
        claim = MockClaim(id="1", text="Test claim")
        original = MockClaimVerification(claim=claim, status=VerificationStatus.VERIFIED)

        status, rationale = adjudicator._determine_status(
            support_score=0.8,
            contradict_score=0.05,  # Below min_contradict_score (0.1)
            original=original,
        )

        assert status == VerificationStatus.VERIFIED
        assert "No significant contradicting evidence" in rationale

    @pytest.mark.unit
    def test_strong_contradiction_refutes(self):
        """Should refute when strong contradiction with weak support."""
        adjudicator = ClaimAdjudicator()
        claim = MockClaim(id="1", text="Test claim")
        original = MockClaimVerification(claim=claim, status=VerificationStatus.VERIFIED)

        status, rationale = adjudicator._determine_status(
            support_score=0.3,  # Below weak_support_threshold (0.4)
            contradict_score=0.8,  # Above strong_contradict_threshold (0.7)
            original=original,
        )

        assert status == VerificationStatus.REFUTED
        assert "Strong contradicting evidence" in rationale

    @pytest.mark.unit
    def test_balanced_evidence_is_contested(self):
        """Should return CONTESTED when evidence is balanced."""
        adjudicator = ClaimAdjudicator(contested_threshold=0.4)
        claim = MockClaim(id="1", text="Test claim")
        original = MockClaimVerification(claim=claim, status=VerificationStatus.VERIFIED)

        status, rationale = adjudicator._determine_status(
            support_score=0.5,
            contradict_score=0.5,
            original=original,
        )

        assert status == VerificationStatus.CONTESTED
        assert "contested" in rationale.lower()

    @pytest.mark.unit
    def test_support_dominates_keeps_verified(self):
        """Should keep VERIFIED when support clearly dominates."""
        adjudicator = ClaimAdjudicator(contested_threshold=0.4)
        claim = MockClaim(id="1", text="Test claim")
        original = MockClaimVerification(claim=claim, status=VerificationStatus.VERIFIED)

        status, rationale = adjudicator._determine_status(
            support_score=0.8,
            contradict_score=0.2,  # ratio = 0.8 / 1.0 = 0.8 > 0.6
            original=original,
        )

        assert status == VerificationStatus.VERIFIED
        assert "dominates" in rationale.lower()

    @pytest.mark.unit
    def test_contradiction_dominates_refutes(self):
        """Should REFUTE when contradiction dominates."""
        adjudicator = ClaimAdjudicator(contested_threshold=0.4)
        claim = MockClaim(id="1", text="Test claim")
        original = MockClaimVerification(claim=claim, status=VerificationStatus.VERIFIED)

        status, rationale = adjudicator._determine_status(
            support_score=0.2,
            contradict_score=0.8,  # ratio = 0.2 / 1.0 = 0.2 < 0.4
            original=original,
        )

        assert status == VerificationStatus.REFUTED
        assert "contradicting evidence" in rationale.lower()


class TestClaimAdjudicatorScoreAggregation:
    """Tests for score aggregation logic."""

    @pytest.mark.unit
    def test_empty_assessments_returns_zero(self):
        """Should return 0 for empty assessment list."""
        adjudicator = ClaimAdjudicator()

        score = adjudicator._aggregate_score([])

        assert score == 0.0

    @pytest.mark.unit
    def test_single_assessment(self):
        """Should return confidence for single assessment."""
        adjudicator = ClaimAdjudicator()
        doc = Document(id="1", content="Test", metadata={}, score=0.8)
        assessment = EvidenceAssessment(
            document=doc,
            stance=EvidenceStance.SUPPORTS,
            confidence=0.9,
        )

        score = adjudicator._aggregate_score([assessment])

        assert score == 0.9

    @pytest.mark.unit
    def test_multiple_assessments_weighted(self):
        """Should weight by confidence for multiple assessments."""
        adjudicator = ClaimAdjudicator()
        doc1 = Document(id="1", content="Test 1", metadata={}, score=0.8)
        doc2 = Document(id="2", content="Test 2", metadata={}, score=0.7)
        assessments = [
            EvidenceAssessment(document=doc1, stance=EvidenceStance.SUPPORTS, confidence=0.9),
            EvidenceAssessment(document=doc2, stance=EvidenceStance.SUPPORTS, confidence=0.5),
        ]

        score = adjudicator._aggregate_score(assessments)

        # (0.9 * 1.0 + 0.5 * 1.0) / (1.0 + 1.0) = 1.4 / 2 = 0.7
        assert abs(score - 0.7) < 0.01

    @pytest.mark.unit
    def test_authority_weighting(self):
        """Should weight by authority_score from metadata."""
        adjudicator = ClaimAdjudicator()
        doc1 = Document(id="1", content="Test 1", metadata={"authority_score": 2.0}, score=0.8)
        doc2 = Document(id="2", content="Test 2", metadata={"authority_score": 1.0}, score=0.7)
        assessments = [
            EvidenceAssessment(document=doc1, stance=EvidenceStance.SUPPORTS, confidence=0.9),
            EvidenceAssessment(document=doc2, stance=EvidenceStance.SUPPORTS, confidence=0.6),
        ]

        score = adjudicator._aggregate_score(assessments)

        # (0.9 * 2.0 + 0.6 * 1.0) / (2.0 + 1.0) = 2.4 / 3 = 0.8
        assert abs(score - 0.8) < 0.01


class TestClaimAdjudicatorAdjudicate:
    """Tests for the main adjudicate method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_adjudicate_no_models(self):
        """Should work without NLI or LLM models (fallback to expected stance)."""
        adjudicator = ClaimAdjudicator()
        claim = MockClaim(id="1", text="Test claim")
        original = MockClaimVerification(claim=claim, status=VerificationStatus.VERIFIED)

        supporting_docs = [
            Document(id="1", content="Support", metadata={}, score=0.8),
        ]
        contradicting_docs = [
            Document(id="2", content="Contradict", metadata={}, score=0.7),
        ]

        result = await adjudicator.adjudicate(
            claim=claim,
            supporting_docs=supporting_docs,
            contradicting_docs=contradicting_docs,
            original_verification=original,
        )

        assert isinstance(result, AdjudicationResult)
        assert result.support_score > 0
        assert result.contradict_score > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_adjudicate_with_nli_pipeline(self):
        """Should use NLI pipeline when available."""
        # Create mock NLI pipeline
        mock_nli = MagicMock()
        mock_nli.return_value = [
            {"label": "entailment", "score": 0.8},
            {"label": "contradiction", "score": 0.1},
            {"label": "neutral", "score": 0.1},
        ]

        adjudicator = ClaimAdjudicator(nli_pipeline=mock_nli)
        claim = MockClaim(id="1", text="Test claim")
        original = MockClaimVerification(claim=claim, status=VerificationStatus.VERIFIED)

        supporting_docs = [
            Document(id="1", content="Support", metadata={}, score=0.8),
        ]

        result = await adjudicator.adjudicate(
            claim=claim,
            supporting_docs=supporting_docs,
            contradicting_docs=[],
            original_verification=original,
        )

        assert mock_nli.called
        assert result.support_score > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_adjudicate_with_llm_fallback(self):
        """Should use LLM fallback when no NLI pipeline."""
        mock_llm = AsyncMock(return_value='{"stance": "SUPPORTS", "confidence": 0.85}')

        adjudicator = ClaimAdjudicator(llm_analyze_fn=mock_llm)
        claim = MockClaim(id="1", text="Test claim")
        original = MockClaimVerification(claim=claim, status=VerificationStatus.VERIFIED)

        supporting_docs = [
            Document(id="1", content="Support", metadata={}, score=0.8),
        ]

        result = await adjudicator.adjudicate(
            claim=claim,
            supporting_docs=supporting_docs,
            contradicting_docs=[],
            original_verification=original,
        )

        assert mock_llm.called
        assert result.support_score > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_adjudicate_handles_assessment_failure(self):
        """Should handle individual document assessment failures gracefully."""
        # Create failing NLI pipeline
        mock_nli = MagicMock(side_effect=Exception("NLI error"))

        adjudicator = ClaimAdjudicator(nli_pipeline=mock_nli)
        claim = MockClaim(id="1", text="Test claim")
        original = MockClaimVerification(claim=claim, status=VerificationStatus.VERIFIED)

        supporting_docs = [
            Document(id="1", content="Support", metadata={}, score=0.8),
        ]

        # Should not raise, should return result with no evidence
        result = await adjudicator.adjudicate(
            claim=claim,
            supporting_docs=supporting_docs,
            contradicting_docs=[],
            original_verification=original,
        )

        assert isinstance(result, AdjudicationResult)


class TestClaimAdjudicatorNLIAssess:
    """Tests for NLI-based stance assessment."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nli_supports(self):
        """Should detect SUPPORTS stance from NLI entailment."""
        mock_nli = MagicMock()
        mock_nli.return_value = [
            {"label": "entailment", "score": 0.9},
            {"label": "contradiction", "score": 0.05},
            {"label": "neutral", "score": 0.05},
        ]

        adjudicator = ClaimAdjudicator(nli_pipeline=mock_nli)

        stance, confidence = await adjudicator._nli_assess("claim", "evidence")

        assert stance == EvidenceStance.SUPPORTS
        assert confidence == 0.9

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nli_contradicts(self):
        """Should detect CONTRADICTS stance from NLI contradiction."""
        mock_nli = MagicMock()
        mock_nli.return_value = [
            {"label": "entailment", "score": 0.05},
            {"label": "contradiction", "score": 0.9},
            {"label": "neutral", "score": 0.05},
        ]

        adjudicator = ClaimAdjudicator(nli_pipeline=mock_nli)

        stance, confidence = await adjudicator._nli_assess("claim", "evidence")

        assert stance == EvidenceStance.CONTRADICTS
        assert confidence == 0.9

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nli_neutral(self):
        """Should detect NEUTRAL stance from NLI neutral."""
        mock_nli = MagicMock()
        mock_nli.return_value = [
            {"label": "entailment", "score": 0.1},
            {"label": "contradiction", "score": 0.1},
            {"label": "neutral", "score": 0.8},
        ]

        adjudicator = ClaimAdjudicator(nli_pipeline=mock_nli)

        stance, confidence = await adjudicator._nli_assess("claim", "evidence")

        assert stance == EvidenceStance.NEUTRAL
        assert confidence == 0.8

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_nli_handles_nested_results(self):
        """Should handle nested list results from some NLI pipelines."""
        mock_nli = MagicMock()
        mock_nli.return_value = [[
            {"label": "entailment", "score": 0.85},
            {"label": "contradiction", "score": 0.1},
            {"label": "neutral", "score": 0.05},
        ]]

        adjudicator = ClaimAdjudicator(nli_pipeline=mock_nli)

        stance, confidence = await adjudicator._nli_assess("claim", "evidence")

        assert stance == EvidenceStance.SUPPORTS
        assert confidence == 0.85


class TestClaimAdjudicatorLLMAssess:
    """Tests for LLM-based stance assessment."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_parses_json(self):
        """Should parse JSON response from LLM."""
        mock_llm = AsyncMock(return_value='{"stance": "CONTRADICTS", "confidence": 0.75}')

        adjudicator = ClaimAdjudicator(llm_analyze_fn=mock_llm)

        stance, confidence = await adjudicator._llm_assess("claim", "evidence")

        assert stance == EvidenceStance.CONTRADICTS
        assert confidence == 0.75

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_handles_markdown_wrapped_json(self):
        """Should handle JSON wrapped in markdown code blocks."""
        mock_llm = AsyncMock(return_value='```json\n{"stance": "SUPPORTS", "confidence": 0.9}\n```')

        adjudicator = ClaimAdjudicator(llm_analyze_fn=mock_llm)

        stance, confidence = await adjudicator._llm_assess("claim", "evidence")

        assert stance == EvidenceStance.SUPPORTS
        assert confidence == 0.9

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_handles_think_tag_wrapped_json(self):
        """Should handle <think> blocks before fenced JSON."""
        mock_llm = AsyncMock(
            return_value=(
                "<think>intermediate reasoning</think>\n"
                "```json\n{\"stance\": \"CONTRADICTS\", \"confidence\": 0.88}\n```"
            )
        )

        adjudicator = ClaimAdjudicator(llm_analyze_fn=mock_llm)

        stance, confidence = await adjudicator._llm_assess("claim", "evidence")

        assert stance == EvidenceStance.CONTRADICTS
        assert confidence == 0.88

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_clamps_confidence_bounds(self):
        """Should clamp confidence to [0, 1] range."""
        mock_llm = AsyncMock(return_value='{"stance": "SUPPORTS", "confidence": 1.7}')

        adjudicator = ClaimAdjudicator(llm_analyze_fn=mock_llm)

        stance, confidence = await adjudicator._llm_assess("claim", "evidence")

        assert stance == EvidenceStance.SUPPORTS
        assert confidence == 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_handles_failure(self):
        """Should return NEUTRAL on LLM failure."""
        mock_llm = AsyncMock(side_effect=Exception("API error"))

        adjudicator = ClaimAdjudicator(llm_analyze_fn=mock_llm)

        stance, confidence = await adjudicator._llm_assess("claim", "evidence")

        assert stance == EvidenceStance.NEUTRAL
        assert confidence == 0.5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_assess_supports_claims_engine_signature(self, monkeypatch):
        """Should support sync analyze callbacks with ClaimsEngine signature."""
        captured: dict[str, Any] = {}

        def mock_claims_engine_analyze(
            api_endpoint: str | None,
            input_data: Any,
            prompt: str | None,
            api_key: str | None,
            system_message: str | None,
            temp: float | None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            captured["api_endpoint"] = api_endpoint
            captured["input_data"] = input_data
            captured["prompt"] = prompt
            captured["system_message"] = system_message
            captured["temp"] = temp
            captured["response_format"] = kwargs.get("response_format")
            return {"choices": [{"message": {"content": '{"stance":"SUPPORTS","confidence":0.91}'}}]}

        monkeypatch.setattr(
            adjudicator_module,
            "resolve_claims_response_format",
            lambda *args, **kwargs: {"type": "json_object"},
        )

        adjudicator = ClaimAdjudicator(llm_analyze_fn=mock_claims_engine_analyze)

        stance, confidence = await adjudicator._llm_assess("claim", "evidence")

        assert stance == EvidenceStance.SUPPORTS
        assert confidence == 0.91
        assert captured["api_endpoint"]
        assert isinstance(captured["input_data"], str)
        assert isinstance(captured["prompt"], str)
        assert captured["response_format"] == {"type": "json_object"}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_assess_falls_back_to_prompt_only_signature(self):
        """Should gracefully fallback when callback only accepts a prompt string."""
        calls = {"count": 0}

        def prompt_only_llm(prompt: str) -> str:
            calls["count"] += 1
            return '{"stance": "NEUTRAL", "confidence": 0.66}'

        adjudicator = ClaimAdjudicator(llm_analyze_fn=prompt_only_llm)

        stance, confidence = await adjudicator._llm_assess("claim", "evidence")

        assert calls["count"] == 1
        assert stance == EvidenceStance.NEUTRAL
        assert confidence == 0.66


class TestCreateAdjudicator:
    """Tests for the factory function."""

    @pytest.mark.unit
    def test_creates_with_defaults(self):
        """Should create adjudicator with default settings."""
        adjudicator = create_adjudicator()

        assert isinstance(adjudicator, ClaimAdjudicator)
        assert adjudicator.nli_pipeline is None
        assert adjudicator.llm_analyze_fn is None
        assert adjudicator.contested_threshold == 0.4

    @pytest.mark.unit
    def test_creates_with_custom_threshold(self):
        """Should respect custom contested threshold."""
        adjudicator = create_adjudicator(contested_threshold=0.3)

        assert adjudicator.contested_threshold == 0.3

    @pytest.mark.unit
    def test_creates_with_nli_pipeline(self):
        """Should accept NLI pipeline."""
        mock_nli = MagicMock()
        adjudicator = create_adjudicator(nli_pipeline=mock_nli)

        assert adjudicator.nli_pipeline == mock_nli

    @pytest.mark.unit
    def test_creates_with_llm_fn(self):
        """Should accept LLM analyze function."""
        mock_llm = AsyncMock()
        adjudicator = create_adjudicator(llm_analyze_fn=mock_llm)

        assert adjudicator.llm_analyze_fn == mock_llm


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


class TestClaimAdjudicatorLLMQuota:
    """The adjudicator's LLM call is quota-checked, and a refusal is not an assessment (TASK-13862)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_call_runs_off_the_loop_where_the_quota_check_runs(self, monkeypatch):
        """On the loop's own thread the check at the call is skipped (LLM usage accounting design section 5)."""
        decided_for: list[int] = []

        async def _admit(user_id, requested_tokens=1):
            decided_for.append(user_id)
            return None

        monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
        quota_resolver.invalidate_all()
        monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _admit)
        monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
        loop_thread = threading.get_ident()
        seen: list[tuple[bool, bool]] = []

        def _analyze(*_args, **_kwargs):
            # (off the loop thread, checked) for the call the provider layer would make here
            seen.append((threading.get_ident() != loop_thread, llm_quota_gate.check_llm_call_sync()))
            return '{"stance": "CONTRADICTS", "confidence": 0.8}'

        adjudicator = ClaimAdjudicator(llm_analyze_fn=_analyze)
        with llm_call_context(user_id=7):
            stance, confidence = await adjudicator._llm_assess("claim", "evidence")

        assert (stance, confidence) == (EvidenceStance.CONTRADICTS, 0.8)
        assert seen == [(True, True)]
        assert decided_for == [7]  # the request's user crossed into the worker thread

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_quota_refusal_is_not_a_neutral_assessment(self):
        """On dev the refusal became NEUTRAL 0.5, and the claim kept its verdict."""
        adjudicator = ClaimAdjudicator(llm_analyze_fn=AsyncMock(side_effect=_refusal()))
        claim = MockClaim(id="1", text="The sky is green.")

        with pytest.raises(LLMQuotaExceededError) as refused:
            await adjudicator.adjudicate(
                claim=claim,
                supporting_docs=[Document(id="1", content="The sky is blue.", metadata={}, score=0.8)],
                contradicting_docs=[],
                original_verification=MockClaimVerification(claim=claim),
            )
        assert refused.value.detail["category"] == "llm_tokens_month"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_refusal_in_the_worker_thread_does_not_reach_the_prompt_only_fallback(self, monkeypatch):
        """The app's analyze shape: sync, full signature only, refused by the check in the worker thread.

        Sent on to the prompt-only fallback, the refusal would be replaced by that call's TypeError: NEUTRAL 0.5.
        """
        decided_for: list[int] = []

        async def _refuse(user_id, requested_tokens=1):
            decided_for.append(user_id)
            return LLMQuotaRefusal(
                category="llm_tokens_month", current=1, limit=1, message="Monthly LLM token limit reached", retry_after=60
            )

        monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
        quota_resolver.invalidate_all()
        monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _refuse)
        monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
        loop_thread = threading.get_ident()
        off_loop: list[bool] = []

        def _analyze(api_endpoint, input_data, prompt, api_key, system_message, temp=None, **_kwargs):
            # The shape of claims_service._fva_claims_analyze_call; perform_chat_api_call runs this check first.
            off_loop.append(threading.get_ident() != loop_thread)
            llm_quota_gate.check_llm_call_sync()
            return '{"stance": "SUPPORTS", "confidence": 0.9}'

        adjudicator = ClaimAdjudicator(llm_analyze_fn=_analyze)
        with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError) as refused:
            await adjudicator._llm_assess("claim", "evidence")

        assert refused.value.detail["category"] == "llm_tokens_month"
        assert off_loop == [True]
        assert decided_for == [7]
