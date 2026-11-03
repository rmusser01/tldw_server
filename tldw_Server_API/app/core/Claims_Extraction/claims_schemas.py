"""Core-owned claims verification models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FVAEvidenceItem(BaseModel):
    """Evidence item for FVA verification results."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="Document ID.")
    snippet: str = Field(..., description="Evidence snippet text.")
    score: float = Field(..., description="Evidence relevance score.")
    stance: str | None = Field(default=None, description="Evidence stance: supports, contradicts, neutral.")
    confidence: float | None = Field(default=None, description="Stance assessment confidence.")


class FVAAdjudicationResult(BaseModel):
    """Adjudication details from FVA pipeline."""

    model_config = ConfigDict(extra="forbid")

    support_score: float = Field(..., description="Aggregate supporting evidence score.")
    contradict_score: float = Field(..., description="Aggregate contradicting evidence score.")
    contestation_score: float = Field(..., description="Evidence balance (0=one-sided, 1=balanced).")
    rationale: str = Field(..., description="Adjudication rationale explaining the verdict.")


class FVAClaimResult(BaseModel):
    """Verification result for a single claim from FVA pipeline."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(..., description="Original claim text.")
    claim_type: str | None = Field(default=None, description="Claim type classification.")
    original_status: str = Field(..., description="Status from initial verification.")
    final_status: str = Field(..., description="Status after FVA adjudication.")
    confidence: float = Field(..., description="Final verification confidence.")
    falsification_triggered: bool = Field(..., description="Whether falsification was triggered.")
    anti_context_found: int = Field(..., description="Number of anti-context documents found.")
    supporting_evidence: list[FVAEvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[FVAEvidenceItem] = Field(default_factory=list)
    adjudication: FVAAdjudicationResult | None = Field(default=None)
    rationale: str | None = Field(default=None, description="Verification rationale.")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds.")
