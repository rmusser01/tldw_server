"""
Tests for the FVA (Falsification-Verification Alignment) pipeline.

These tests verify the complete FVA pipeline integration including
falsification triggering, anti-context retrieval, and adjudication.
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from tldw_Server_API.app.core.Claims_Extraction.fva_pipeline import (
    FVAPipeline,
    FVAConfig,
    FVAResult,
    FVABatchResult,
    create_fva_pipeline,
)
from tldw_Server_API.app.core.Claims_Extraction.budget_guard import (
    ClaimsJobBudget,
    ClaimsJobContext,
)

# Import types
try:
    from tldw_Server_API.app.core.RAG.rag_service.types import (
        VerificationStatus,
        Document,
        MatchLevel,
        SourceAuthority,
        ClaimType,
    )
except ImportError:
    from enum import Enum

    class VerificationStatus(Enum):
        VERIFIED = "verified"
        REFUTED = "refuted"
        CONTESTED = "contested"
        UNVERIFIED = "unverified"

    class ClaimType(Enum):
        STATISTIC = "statistic"
        GENERAL = "general"

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
class MockEvidence:
    """Mock evidence for testing."""
    doc_id: str
    snippet: str
    score: float = 0.8
    authority: SourceAuthority = SourceAuthority.SECONDARY


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


class MockVerifier:
    """Mock verifier for testing."""

    def __init__(self, verification: Optional[MockClaimVerification] = None):
        self.verification = verification
        self.verify_calls = []

    async def verify(
        self,
        claim,
        query: str,
        base_documents: list,
        retrieve_fn=None,
        top_k: int = 5,
        conf_threshold: float = 0.7,
        mode: str = "hybrid",
        budget=None,
        job_context=None,
    ):
        self.verify_calls.append({
            "claim": claim,
            "query": query,
            "documents": base_documents,
        })
        if self.verification:
            return self.verification
        return MockClaimVerification(claim=claim)


class MockClaimsEngine:
    """Mock claims engine for testing."""

    def __init__(self, verifier: Optional[MockVerifier] = None):
        self.verifier = verifier or MockVerifier()
        self._analyze = AsyncMock(return_value='{"stance": "SUPPORTS", "confidence": 0.8}')
        self._nli = None


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, documents: Optional[list[Document]] = None):
        self.documents = documents or []
        self.retrieve_calls = []

    async def retrieve(
        self,
        query: str,
        *,
        sources=None,
        config=None,
        **kwargs
    ) -> list[Document]:
        self.retrieve_calls.append({"query": query, "sources": sources})
        return self.documents


class TestFVAConfig:
    """Tests for FVAConfig dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """Default config should have sensible values."""
        config = FVAConfig()

        assert config.enabled is True
        assert config.max_concurrent_falsifications == 5
        assert config.falsification_timeout_seconds == 30.0
        assert config.min_confidence_for_skip == 0.9
        assert config.max_budget_ratio_for_fva == 0.3
        assert config.confidence_threshold == 0.7
        assert config.contested_threshold == 0.4

    @pytest.mark.unit
    def test_custom_values(self):
        """Custom config values should be respected."""
        config = FVAConfig(
            enabled=False,
            max_concurrent_falsifications=3,
            falsification_timeout_seconds=15.0,
            confidence_threshold=0.8,
            force_falsification_claim_types=["statistic", "causal"],
        )

        assert config.enabled is False
        assert config.max_concurrent_falsifications == 3
        assert config.falsification_timeout_seconds == 15.0
        assert config.confidence_threshold == 0.8
        assert "statistic" in config.force_falsification_claim_types


class TestFVAPipelineInit:
    """Tests for FVAPipeline initialization wiring."""

    @pytest.mark.unit
    def test_prefers_verifier_nli_pipeline(self):
        """Pipeline should wire adjudicator NLI from claims_engine.verifier._nli."""
        nli_obj = object()
        verifier = MockVerifier()
        verifier._nli = nli_obj  # type: ignore[attr-defined]

        class _ClaimsEngineNoLegacyNli:
            def __init__(self):
                self.verifier = verifier
                self._analyze = AsyncMock(return_value='{"stance":"SUPPORTS","confidence":0.8}')

        pipeline = FVAPipeline(_ClaimsEngineNoLegacyNli(), MockRetriever())
        assert pipeline.adjudicator.nli_pipeline is nli_obj


class TestFVAPipelineProcessClaim:
    """Tests for FVAPipeline.process_claim()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_claim_basic(self):
        """Should process a claim through the pipeline."""
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever)

        claim = MockClaim(id="1", text="Test claim")
        documents = [Document(id="1", content="Test content", metadata={}, score=0.8)]

        result = await pipeline.process_claim(
            claim=claim,
            query="test query",
            documents=documents,
        )

        assert isinstance(result, FVAResult)
        assert result.original_verification is not None
        assert result.final_verification is not None
        assert result.processing_time_ms > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_claim_disabled(self):
        """Should skip falsification when disabled."""
        config = FVAConfig(enabled=False)
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever, config)

        claim = MockClaim(id="1", text="Test claim")
        documents = [Document(id="1", content="Test", metadata={}, score=0.8)]

        result = await pipeline.process_claim(
            claim=claim,
            query="test",
            documents=documents,
        )

        assert result.falsification_triggered is False
        assert result.falsification_decision is None
        assert result.adjudication is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_claim_triggers_falsification_low_confidence(self):
        """Should trigger falsification for low confidence claims."""
        # Create verification with low confidence
        low_conf_verification = MockClaimVerification(
            claim=MockClaim(id="1", text="Test"),
            status=VerificationStatus.VERIFIED,
            confidence=0.5,  # Below threshold
            evidence=[MockEvidence(doc_id="1", snippet="Evidence")],
        )
        verifier = MockVerifier(verification=low_conf_verification)
        claims_engine = MockClaimsEngine(verifier=verifier)

        # Return anti-context documents
        anti_docs = [
            Document(id="anti_1", content="Contradicting evidence", metadata={}, score=0.7),
        ]
        retriever = MockRetriever(documents=anti_docs)

        pipeline = FVAPipeline(claims_engine, retriever)

        claim = MockClaim(id="1", text="Test claim")
        documents = [Document(id="1", content="Support", metadata={}, score=0.8)]

        result = await pipeline.process_claim(
            claim=claim,
            query="test",
            documents=documents,
        )

        assert result.falsification_triggered is True
        assert result.falsification_decision is not None
        assert result.falsification_decision.should_falsify is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_claim_records_adjudication_scores_when_anti_context_found(self):
        """Should record adjudication scores when anti-context produces adjudication."""
        low_conf_verification = MockClaimVerification(
            claim=MockClaim(id="1", text="Test"),
            status=VerificationStatus.VERIFIED,
            confidence=0.5,
            evidence=[MockEvidence(doc_id="1", snippet="Evidence")],
        )
        verifier = MockVerifier(verification=low_conf_verification)
        claims_engine = MockClaimsEngine(verifier=verifier)
        anti_docs = [
            Document(id="anti_1", content="Contradicting evidence", metadata={}, score=0.7),
        ]
        retriever = MockRetriever(documents=anti_docs)
        pipeline = FVAPipeline(claims_engine, retriever)

        with patch(
            "tldw_Server_API.app.core.Claims_Extraction.fva_pipeline.observe_histogram"
        ) as observe_histogram:
            await pipeline.process_claim(
                claim=MockClaim(id="1", text="Test claim"),
                query="test",
                documents=[Document(id="1", content="Support", metadata={}, score=0.8)],
            )

        score_types = {
            call.kwargs["labels"]["score_type"]
            for call in observe_histogram.call_args_list
            if call.args and call.args[0] == "fva_adjudication_scores"
        }
        assert score_types == {"support", "contradict", "contestation"}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_claim_no_falsification_high_confidence(self):
        """Should not trigger falsification for high confidence claims."""
        # Create verification with high confidence
        high_conf_verification = MockClaimVerification(
            claim=MockClaim(id="1", text="Test"),
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[
                MockEvidence(doc_id="1", snippet="Strong evidence 1"),
                MockEvidence(doc_id="2", snippet="Strong evidence 2"),
                MockEvidence(doc_id="3", snippet="Strong evidence 3"),
            ],
        )
        verifier = MockVerifier(verification=high_conf_verification)
        claims_engine = MockClaimsEngine(verifier=verifier)
        retriever = MockRetriever()

        pipeline = FVAPipeline(claims_engine, retriever)

        claim = MockClaim(id="1", text="Well-supported claim")
        documents = [Document(id="1", content="Support", metadata={}, score=0.9)]

        result = await pipeline.process_claim(
            claim=claim,
            query="test",
            documents=documents,
        )

        assert result.falsification_triggered is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_claim_force_falsification_by_type(self):
        """Should force falsification for configured claim types."""
        config = FVAConfig(
            force_falsification_claim_types=["statistic"],
        )
        # High confidence but should still trigger due to type
        high_conf_verification = MockClaimVerification(
            claim=MockClaim(id="1", text="Test", claim_type=ClaimType.STATISTIC),
            status=VerificationStatus.VERIFIED,
            confidence=0.95,
            evidence=[MockEvidence(doc_id="1", snippet="Evidence")],
        )
        verifier = MockVerifier(verification=high_conf_verification)
        claims_engine = MockClaimsEngine(verifier=verifier)
        retriever = MockRetriever()

        pipeline = FVAPipeline(claims_engine, retriever, config)

        claim = MockClaim(id="1", text="Revenue grew 15%", claim_type=ClaimType.STATISTIC)
        documents = [Document(id="1", content="Support", metadata={}, score=0.9)]

        result = await pipeline.process_claim(
            claim=claim,
            query="test",
            documents=documents,
        )

        assert result.falsification_triggered is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_claim_handles_timeout(self):
        """Should handle timeout gracefully."""
        claims_engine = MockClaimsEngine()

        # Create a retriever that takes too long
        slow_retriever = MockRetriever()

        async def slow_retrieve(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return []

        slow_retriever.retrieve = slow_retrieve

        config = FVAConfig(falsification_timeout_seconds=0.1)
        pipeline = FVAPipeline(claims_engine, slow_retriever, config)

        # Force low confidence to trigger falsification
        claims_engine.verifier.verification = MockClaimVerification(
            claim=MockClaim(id="1", text="Test"),
            confidence=0.3,
            evidence=[],
        )

        claim = MockClaim(id="1", text="Test claim")
        documents = [Document(id="1", content="Test", metadata={}, score=0.8)]

        result = await pipeline.process_claim(
            claim=claim,
            query="test",
            documents=documents,
        )

        # Should complete without error, keeping original verification
        assert result.final_verification.status == result.original_verification.status


class TestFVAPipelineProcessBatch:
    """Tests for FVAPipeline.process_batch()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_batch_empty(self):
        """Should handle empty claims list."""
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever)

        result = await pipeline.process_batch(
            claims=[],
            query="test",
            documents=[],
        )

        assert isinstance(result, FVABatchResult)
        assert result.total_claims == 0
        assert result.results == []
        assert result.total_time_ms == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_batch_multiple_claims(self):
        """Should process multiple claims."""
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever)

        claims = [
            MockClaim(id="1", text="Claim 1"),
            MockClaim(id="2", text="Claim 2"),
            MockClaim(id="3", text="Claim 3"),
        ]
        documents = [Document(id="1", content="Test", metadata={}, score=0.8)]

        result = await pipeline.process_batch(
            claims=claims,
            query="test",
            documents=documents,
        )

        assert result.total_claims == 3
        assert len(result.results) == 3
        assert result.total_time_ms > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_batch_respects_concurrency(self):
        """Should respect concurrency limits."""
        config = FVAConfig(max_concurrent_falsifications=2)
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever, config)

        claims = [MockClaim(id=str(i), text=f"Claim {i}") for i in range(5)]
        documents = [Document(id="1", content="Test", metadata={}, score=0.8)]

        result = await pipeline.process_batch(
            claims=claims,
            query="test",
            documents=documents,
        )

        # All claims should be processed despite concurrency limit
        assert len(result.results) == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_batch_tracks_status_changes(self):
        """Should track status changes in batch results."""
        claims_engine = MockClaimsEngine()

        # Return anti-context that will cause status change
        anti_docs = [
            Document(id="anti", content="Strong contradiction", metadata={}, score=0.9),
        ]
        retriever = MockRetriever(documents=anti_docs)

        # Low confidence to trigger falsification
        claims_engine.verifier.verification = MockClaimVerification(
            claim=MockClaim(id="1", text="Test"),
            status=VerificationStatus.VERIFIED,
            confidence=0.4,
            evidence=[],
        )

        pipeline = FVAPipeline(claims_engine, retriever)

        claims = [MockClaim(id="1", text="Claim 1")]
        documents = [Document(id="1", content="Support", metadata={}, score=0.8)]

        result = await pipeline.process_batch(
            claims=claims,
            query="test",
            documents=documents,
        )

        # Status changes dict should be populated if any changes occurred
        assert isinstance(result.status_changes, dict)


class TestFVAPipelineBudget:
    """Tests for budget integration."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_respects_budget_exhausted(self):
        """Should skip falsification when budget exhausted."""
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever)

        budget = ClaimsJobBudget(max_cost_usd=0.01)
        budget.used_cost_usd = 0.01  # Exhaust budget
        budget.exhausted = True

        claim = MockClaim(id="1", text="Test claim")
        documents = [Document(id="1", content="Test", metadata={}, score=0.8)]

        result = await pipeline.process_claim(
            claim=claim,
            query="test",
            documents=documents,
            budget=budget,
        )

        # Falsification should be skipped due to budget
        assert result.falsification_triggered is False

    @pytest.mark.unit
    def test_can_afford_falsification_no_limit(self):
        """Should allow falsification when no budget limit."""
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever)

        budget = ClaimsJobBudget()  # No limits

        assert pipeline._can_afford_falsification(budget) is True

    @pytest.mark.unit
    def test_can_afford_falsification_within_budget(self):
        """Should allow falsification when within budget."""
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever)

        budget = ClaimsJobBudget(max_cost_usd=1.0)
        budget.used_cost_usd = 0.1  # Well within budget

        assert pipeline._can_afford_falsification(budget) is True


class TestFVAPipelineStats:
    """Tests for pipeline statistics."""

    @pytest.mark.unit
    def test_get_stats(self):
        """Should return pipeline statistics."""
        config = FVAConfig(
            max_concurrent_falsifications=3,
            falsification_timeout_seconds=20.0,
        )
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever, config)

        stats = pipeline.get_stats()

        assert stats["enabled"] is True
        assert stats["max_concurrent"] == 3
        assert stats["timeout_seconds"] == 20.0
        assert "anti_context_cache_size" in stats

    @pytest.mark.unit
    def test_clear_caches(self):
        """Should clear caches."""
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()
        pipeline = FVAPipeline(claims_engine, retriever)

        # Add something to cache
        pipeline.anti_retriever._query_cache["test"] = []

        assert pipeline.anti_retriever.get_cache_size() > 0

        pipeline.clear_caches()

        assert pipeline.anti_retriever.get_cache_size() == 0


class TestCreateFVAPipeline:
    """Tests for the factory function."""

    @pytest.mark.unit
    def test_creates_with_defaults(self):
        """Should create pipeline with default settings."""
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()

        pipeline = create_fva_pipeline(claims_engine, retriever)

        assert isinstance(pipeline, FVAPipeline)
        assert pipeline.config.enabled is True
        assert pipeline.config.max_concurrent_falsifications == 5

    @pytest.mark.unit
    def test_creates_with_custom_settings(self):
        """Should create pipeline with custom settings."""
        claims_engine = MockClaimsEngine()
        retriever = MockRetriever()

        pipeline = create_fva_pipeline(
            claims_engine,
            retriever,
            enabled=False,
            max_concurrent=10,
            timeout_seconds=60.0,
            confidence_threshold=0.8,
            contested_threshold=0.3,
            force_claim_types=["statistic"],
        )

        assert pipeline.config.enabled is False
        assert pipeline.config.max_concurrent_falsifications == 10
        assert pipeline.config.falsification_timeout_seconds == 60.0
        assert pipeline.config.confidence_threshold == 0.8
        assert pipeline.config.contested_threshold == 0.3
        assert "statistic" in pipeline.config.force_falsification_claim_types


class TestFVAResult:
    """Tests for FVAResult dataclass."""

    @pytest.mark.unit
    def test_result_fields(self):
        """FVAResult should have expected fields."""
        claim = MockClaim(id="1", text="Test")
        original = MockClaimVerification(claim=claim)
        final = MockClaimVerification(claim=claim)

        result = FVAResult(
            original_verification=original,
            falsification_triggered=True,
            falsification_decision=None,
            anti_context_found=3,
            adjudication=None,
            final_verification=final,
            processing_time_ms=150.5,
        )

        assert result.falsification_triggered is True
        assert result.anti_context_found == 3
        assert result.processing_time_ms == 150.5


class TestFVABatchResult:
    """Tests for FVABatchResult dataclass."""

    @pytest.mark.unit
    def test_batch_result_fields(self):
        """FVABatchResult should have expected fields."""
        result = FVABatchResult(
            results=[],
            total_claims=5,
            falsification_triggered_count=2,
            status_changes={"verified->contested": 1},
            total_time_ms=500.0,
            budget_exhausted=False,
        )

        assert result.total_claims == 5
        assert result.falsification_triggered_count == 2
        assert result.status_changes["verified->contested"] == 1
        assert result.total_time_ms == 500.0
        assert result.budget_exhausted is False
