"""
fva_pipeline.py - Falsification-Verification Alignment Pipeline.

This module implements the complete FVA pipeline that extends standard
claim verification with active counter-evidence retrieval. It integrates
the falsification trigger, anti-context retrieval, and adjudicator
components into a unified workflow.

Inspired by FVA-RAG paper (arXiv:2512.07015).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
        MultiDatabaseRetriever,
    )

from tldw_Server_API.app.core.Claims_Extraction.adjudicator import (
    AdjudicationResult,
    ClaimAdjudicator,
)
from tldw_Server_API.app.core.Claims_Extraction.anti_context_retriever import (
    AntiContextConfig,
    AntiContextRetriever,
)
from tldw_Server_API.app.core.Claims_Extraction.budget_guard import (
    ClaimsJobBudget,
    ClaimsJobContext,
)
from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
    Claim,
    ClaimsEngine,
    ClaimVerification,
)
from tldw_Server_API.app.core.RAG.rag_service.types import (
    Document,
    VerificationStatus,
)
from tldw_Server_API.app.core.Claims_Extraction.falsification import (
    FalsificationDecision,
    should_trigger_falsification,
)


# Metrics integration - graceful fallback if not available
try:
    from tldw_Server_API.app.core.Metrics.metrics_manager import (
        increment_counter,
        observe_histogram,
    )
except ImportError:

    def increment_counter(*args, **kwargs) -> None:
        pass

    def observe_histogram(*args, **kwargs) -> None:
        pass


def _load_fva_config_from_settings() -> dict[str, Any]:
    """Load FVA configuration from application settings."""
    try:
        from tldw_Server_API.app.core.config import settings
    except ImportError:
        return {}

    return {
        "enabled": settings.get("FVA_ENABLED", True),
        "confidence_threshold": float(settings.get("FVA_CONFIDENCE_THRESHOLD", 0.7)),
        "contested_threshold": float(settings.get("FVA_CONTESTED_THRESHOLD", 0.4)),
        "max_concurrent_falsifications": int(settings.get("FVA_MAX_CONCURRENT", 5)),
        "falsification_timeout_seconds": float(settings.get("FVA_TIMEOUT_SECONDS", 30.0)),
        "max_budget_ratio_for_fva": float(settings.get("FVA_MAX_BUDGET_RATIO", 0.3)),
        "min_confidence_for_skip": float(settings.get("FVA_MIN_CONFIDENCE_FOR_SKIP", 0.9)),
        "force_falsification_claim_types": [
            t.strip() for t in str(settings.get("FVA_FORCE_CLAIM_TYPES", "")).split(",")
            if t.strip()
        ],
    }


@dataclass
class FVAConfig:
    """Configuration for Falsification-Verification Alignment pipeline."""

    enabled: bool = True
    max_concurrent_falsifications: int = 5
    falsification_timeout_seconds: float = 30.0
    min_confidence_for_skip: float = 0.9  # Skip falsification if very confident
    force_falsification_claim_types: list[str] = field(default_factory=list)
    anti_context_config: AntiContextConfig | None = None
    # Budget integration
    max_budget_ratio_for_fva: float = 0.3  # Max 30% of budget for FVA
    # Thresholds
    confidence_threshold: float = 0.7
    contested_threshold: float = 0.4


@dataclass
class FVAResult:
    """Result of FVA pipeline processing for a single claim."""

    original_verification: ClaimVerification
    falsification_triggered: bool
    falsification_decision: FalsificationDecision | None
    anti_context_found: int
    adjudication: AdjudicationResult | None
    final_verification: ClaimVerification
    processing_time_ms: float


@dataclass
class FVABatchResult:
    """Batch result for multiple claims."""

    results: list[FVAResult]
    total_claims: int
    falsification_triggered_count: int
    status_changes: dict[str, int]  # e.g., {"verified->contested": 2}
    total_time_ms: float
    budget_exhausted: bool = False


class FVAPipeline:
    """
    Falsification-Verification Alignment Pipeline.

    Extends standard verification with active counter-evidence retrieval
    for uncertain or high-risk claims.
    """

    def __init__(
        self,
        claims_engine: ClaimsEngine,
        retriever: MultiDatabaseRetriever,
        config: FVAConfig | None = None,
    ):
        """
        Initialize the FVA pipeline.

        Args:
            claims_engine: Existing ClaimsEngine instance
            retriever: MultiDatabaseRetriever for anti-context retrieval
            config: FVA configuration options
        """
        self.claims_engine = claims_engine
        self.retriever = retriever
        self.config = config or FVAConfig()

        self.anti_retriever = AntiContextRetriever(
            retriever,
            self.config.anti_context_config,
        )

        # Prefer verifier-managed NLI pipeline (where ClaimsEngine stores it), then legacy fallback.
        nli_pipeline = getattr(getattr(claims_engine, "verifier", None), "_nli", None)
        if nli_pipeline is None:
            nli_pipeline = getattr(claims_engine, "_nli", None)

        self.adjudicator = ClaimAdjudicator(
            nli_pipeline=nli_pipeline,
            llm_analyze_fn=getattr(claims_engine, "_analyze", None),
            contested_threshold=self.config.contested_threshold,
        )

    async def process_claim(
        self,
        claim: Claim,
        query: str,
        documents: list[Document],
        user_id: str | None = None,
        budget: ClaimsJobBudget | None = None,
        job_context: ClaimsJobContext | None = None,
    ) -> FVAResult:
        """
        Process a single claim through the FVA pipeline.

        1. Standard verification
        2. Decide if falsification needed
        3. If yes, retrieve anti-context and adjudicate

        Args:
            claim: The claim to process
            query: Original user query (required for verification)
            documents: Documents from original retrieval
            user_id: User ID for scoped retrieval
            budget: Budget constraints for cost tracking
            job_context: Job context for logging

        Returns:
            FVAResult with verification outcome
        """
        start_time = datetime.now(timezone.utc)

        # Step 1: Standard verification using existing verifier
        original_verification = await self.claims_engine.verifier.verify(
            claim=claim,
            query=query,
            base_documents=documents,
            budget=budget,
            job_context=job_context,
        )

        # Step 2: Falsification trigger decision
        falsification_decision: FalsificationDecision | None = None
        anti_context_count = 0
        adjudication: AdjudicationResult | None = None
        final_verification = original_verification

        if self.config.enabled:
            # Check budget before proceeding
            if budget and not self._can_afford_falsification(budget):
                logger.debug("Skipping falsification due to budget constraints")
            else:
                # Check if claim type should force falsification
                force = (
                    claim.claim_type is not None
                    and claim.claim_type.value in self.config.force_falsification_claim_types
                )

                falsification_decision = should_trigger_falsification(
                    claim=claim,
                    verification_confidence=original_verification.confidence,
                    evidence_count=len(original_verification.evidence),
                    force_falsification=force,
                    confidence_threshold=self.config.confidence_threshold,
                )

                # Step 3: If triggered, retrieve anti-context and adjudicate
                if falsification_decision.should_falsify:
                    increment_counter(
                        "fva_falsification_triggered_total",
                        labels={
                            "reason": (
                                falsification_decision.reason.value
                                if falsification_decision.reason
                                else "unknown"
                            )
                        },
                    )

                    try:
                        anti_results = await asyncio.wait_for(
                            self.anti_retriever.retrieve_anti_context(
                                claim=claim,
                                original_doc_ids={d.id for d in documents},
                                user_id=user_id,
                            ),
                            timeout=self.config.falsification_timeout_seconds,
                        )

                        anti_docs: list[Document] = []
                        for result in anti_results:
                            anti_docs.extend(result.documents)
                        anti_context_count = len(anti_docs)

                        observe_histogram(
                            "fva_anti_context_docs",
                            anti_context_count,
                        )

                        if anti_docs:
                            adjudication = await self.adjudicator.adjudicate(
                                claim=claim,
                                supporting_docs=documents,
                                contradicting_docs=anti_docs,
                                original_verification=original_verification,
                            )

                            # Update final verification based on adjudication
                            final_verification = ClaimVerification(
                                claim=claim,
                                status=adjudication.final_status,
                                confidence=max(
                                    adjudication.support_score,
                                    adjudication.contradict_score,
                                ),
                                evidence=original_verification.evidence,
                                rationale=adjudication.adjudication_rationale,
                                match_level=original_verification.match_level,
                                source_authority=original_verification.source_authority,
                            )

                            # Track status changes
                            if original_verification.status != final_verification.status:
                                increment_counter(
                                    "fva_status_changes_total",
                                    labels={
                                        "from_status": original_verification.status.value,
                                        "to_status": final_verification.status.value,
                                    },
                                )

                            # Record adjudication scores
                            observe_histogram(
                                "fva_adjudication_scores",
                                adjudication.support_score,
                                labels={"score_type": "support"},
                            )
                            observe_histogram(
                                "fva_adjudication_scores",
                                adjudication.contradict_score,
                                labels={"score_type": "contradict"},
                            )
                            observe_histogram(
                                "fva_adjudication_scores",
                                adjudication.contestation_score,
                                labels={"score_type": "contestation"},
                            )
                        else:
                            increment_counter("fva_wasted_falsification_total")

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Falsification timeout for claim: {claim.text[:50]}..."
                        )
                        increment_counter("fva_timeout_total")
                        # Keep original verification on timeout
                        final_verification = original_verification

                    except Exception as e:
                        logger.error(f"Falsification error for claim {claim.id}: {e}")
                        # Keep original verification on error
                        final_verification = original_verification

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        observe_histogram(
            "fva_processing_duration_seconds",
            elapsed_ms / 1000,
            labels={"phase": "total"},
        )

        # Record claim processed with final status
        increment_counter(
            "fva_claims_processed_total",
            labels={"final_status": final_verification.status.value},
        )

        return FVAResult(
            original_verification=original_verification,
            falsification_triggered=(
                falsification_decision.should_falsify if falsification_decision else False
            ),
            falsification_decision=falsification_decision,
            anti_context_found=anti_context_count,
            adjudication=adjudication,
            final_verification=final_verification,
            processing_time_ms=elapsed_ms,
        )

    async def process_batch(
        self,
        claims: list[Claim],
        query: str,
        documents: list[Document],
        user_id: str | None = None,
        budget: ClaimsJobBudget | None = None,
        job_context: ClaimsJobContext | None = None,
    ) -> FVABatchResult:
        """
        Process multiple claims with concurrency control.

        Args:
            claims: List of claims to process
            query: Original user query
            documents: Documents from original retrieval
            user_id: User ID for scoped retrieval
            budget: Budget constraints
            job_context: Job context for logging

        Returns:
            FVABatchResult with all results and summary
        """
        start_time = datetime.now(timezone.utc)
        budget_exhausted = False

        if not claims:
            return FVABatchResult(
                results=[],
                total_claims=0,
                falsification_triggered_count=0,
                status_changes={},
                total_time_ms=0.0,
                budget_exhausted=False,
            )

        semaphore = asyncio.Semaphore(self.config.max_concurrent_falsifications)

        async def process_with_semaphore(claim: Claim) -> FVAResult:
            async with semaphore:
                return await self.process_claim(
                    claim=claim,
                    query=query,
                    documents=documents,
                    user_id=user_id,
                    budget=budget,
                    job_context=job_context,
                )

        results = await asyncio.gather(
            *[process_with_semaphore(c) for c in claims],
            return_exceptions=True,
        )

        # Filter out exceptions
        valid_results: list[FVAResult] = [
            r for r in results if isinstance(r, FVAResult)
        ]

        # Log any exceptions
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"FVA batch processing error for claim {i}: {r}")

        # Check for budget exhaustion
        if budget and budget.exhausted:
            budget_exhausted = True

        # Calculate status changes
        status_changes: dict[str, int] = {}
        for r in valid_results:
            if r.original_verification.status != r.final_verification.status:
                key = f"{r.original_verification.status.value}->{r.final_verification.status.value}"
                status_changes[key] = status_changes.get(key, 0) + 1

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return FVABatchResult(
            results=valid_results,
            total_claims=len(claims),
            falsification_triggered_count=sum(
                1 for r in valid_results if r.falsification_triggered
            ),
            status_changes=status_changes,
            total_time_ms=elapsed_ms,
            budget_exhausted=budget_exhausted,
        )

    def _can_afford_falsification(self, budget: ClaimsJobBudget) -> bool:
        """Check if budget allows for falsification overhead."""
        if budget.max_cost_usd is None:
            return True

        # Estimate FVA cost (rough: 3 queries + 1 adjudication)
        estimated_fva_cost = 0.01  # $0.01 per claim for FVA

        remaining = budget.remaining_cost_usd()
        if remaining is None:
            return True

        allowed_for_fva = budget.max_cost_usd * self.config.max_budget_ratio_for_fva

        return remaining >= estimated_fva_cost and budget.used_cost_usd < allowed_for_fva

    def get_stats(self) -> dict[str, Any]:
        """Get current pipeline statistics."""
        return {
            "enabled": self.config.enabled,
            "max_concurrent": self.config.max_concurrent_falsifications,
            "timeout_seconds": self.config.falsification_timeout_seconds,
            "confidence_threshold": self.config.confidence_threshold,
            "contested_threshold": self.config.contested_threshold,
            "anti_context_cache_size": self.anti_retriever.get_cache_size(),
        }

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self.anti_retriever.clear_cache()


def create_fva_pipeline(
    claims_engine: ClaimsEngine,
    retriever: MultiDatabaseRetriever,
    enabled: bool = True,
    max_concurrent: int = 5,
    timeout_seconds: float = 30.0,
    confidence_threshold: float = 0.7,
    contested_threshold: float = 0.4,
    force_claim_types: list[str] | None = None,
) -> FVAPipeline:
    """
    Factory function to create an FVA pipeline.

    Args:
        claims_engine: ClaimsEngine instance
        retriever: MultiDatabaseRetriever for searching
        enabled: Whether FVA is enabled
        max_concurrent: Maximum concurrent falsifications
        timeout_seconds: Timeout for falsification operations
        confidence_threshold: Confidence threshold for triggering
        contested_threshold: Threshold for CONTESTED status
        force_claim_types: Claim types that always trigger falsification

    Returns:
        Configured FVAPipeline instance
    """
    config = FVAConfig(
        enabled=enabled,
        max_concurrent_falsifications=max_concurrent,
        falsification_timeout_seconds=timeout_seconds,
        confidence_threshold=confidence_threshold,
        contested_threshold=contested_threshold,
        force_falsification_claim_types=force_claim_types or [],
    )

    return FVAPipeline(
        claims_engine=claims_engine,
        retriever=retriever,
        config=config,
    )


def create_fva_pipeline_from_settings(
    claims_engine: ClaimsEngine,
    retriever: MultiDatabaseRetriever,
    config_overrides: dict[str, Any] | None = None,
) -> FVAPipeline:
    """
    Factory function to create an FVA pipeline from application settings.

    Loads configuration from config.txt [Claims] section FVA_* settings,
    with optional overrides for specific parameters.

    Args:
        claims_engine: ClaimsEngine instance
        retriever: MultiDatabaseRetriever for searching
        config_overrides: Optional dict of config values to override settings

    Returns:
        Configured FVAPipeline instance using settings
    """
    # Load from settings
    settings_config = _load_fva_config_from_settings()

    # Apply overrides if provided
    if config_overrides:
        settings_config.update(config_overrides)

    config = FVAConfig(
        enabled=settings_config.get("enabled", True),
        max_concurrent_falsifications=settings_config.get("max_concurrent_falsifications", 5),
        falsification_timeout_seconds=settings_config.get("falsification_timeout_seconds", 30.0),
        confidence_threshold=settings_config.get("confidence_threshold", 0.7),
        contested_threshold=settings_config.get("contested_threshold", 0.4),
        force_falsification_claim_types=settings_config.get("force_falsification_claim_types", []),
        max_budget_ratio_for_fva=settings_config.get("max_budget_ratio_for_fva", 0.3),
        min_confidence_for_skip=settings_config.get("min_confidence_for_skip", 0.9),
    )

    logger.debug(
        f"Created FVA pipeline from settings: enabled={config.enabled}, "
        f"confidence_threshold={config.confidence_threshold}, "
        f"contested_threshold={config.contested_threshold}"
    )

    return FVAPipeline(
        claims_engine=claims_engine,
        retriever=retriever,
        config=config,
    )


def get_fva_config_from_settings() -> FVAConfig:
    """
    Get FVA configuration from application settings.

    Returns:
        FVAConfig instance with values from settings
    """
    settings_config = _load_fva_config_from_settings()

    return FVAConfig(
        enabled=settings_config.get("enabled", True),
        max_concurrent_falsifications=settings_config.get("max_concurrent_falsifications", 5),
        falsification_timeout_seconds=settings_config.get("falsification_timeout_seconds", 30.0),
        confidence_threshold=settings_config.get("confidence_threshold", 0.7),
        contested_threshold=settings_config.get("contested_threshold", 0.4),
        force_falsification_claim_types=settings_config.get("force_falsification_claim_types", []),
        max_budget_ratio_for_fva=settings_config.get("max_budget_ratio_for_fva", 0.3),
        min_confidence_for_skip=settings_config.get("min_confidence_for_skip", 0.9),
    )
