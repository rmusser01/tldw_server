"""Bounded public contracts for strict, citation-bound source support checks."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkspaceCitationLocator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=64, pattern=r"^(?:E\d+(?:\.\d+)+|P\d+|\d+)$")
    source_id: str = Field(min_length=1, max_length=256)
    quote: str = Field(min_length=1, max_length=100_000)
    start_char: int | None = Field(None, ge=0, strict=True)
    end_char: int | None = Field(None, ge=1, strict=True)
    navigation_version: str | None = Field(None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "WorkspaceCitationLocator":
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("Both source offsets are required together")
        if self.start_char is not None and self.end_char <= self.start_char:
            raise ValueError("Source end must follow source start")
        return self


class ResearchWorkspaceCitationVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: str = Field(min_length=1, max_length=256)
    text: str = Field(max_length=100_000)
    api_provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    evidence: list[WorkspaceCitationLocator] = Field(default_factory=list, max_length=256)

    @model_validator(mode="after")
    def validate_evidence(self) -> "ResearchWorkspaceCitationVerifyRequest":
        ids = [item.id for item in self.evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("Citation IDs must be unique")
        if sum(len(item.quote) for item in self.evidence) > 100_000:
            raise ValueError("Total citation locator text exceeds 100000 characters")
        return self


class CitationSupportFinding(BaseModel):
    id: str
    text: str
    start: int
    end: int
    evidence_ids: list[str]
    status: Literal["supported", "unsupported", "contradicted", "uncited", "not_checked", "error"]
    reason: str


class CitationSupportCoverage(BaseModel):
    total: int
    evaluated: int
    supported: int
    uncovered: int
    complete: bool


class CitationSupportReport(BaseModel):
    version: Literal[1]
    text_sha256: str
    verdict: Literal["supported", "needs_revision", "unverified"]
    provider: str
    model: str
    coverage: CitationSupportCoverage
    findings: list[CitationSupportFinding]
    evidence: dict[str, dict[str, Any]]
    limits: dict[str, int | float]
    context: dict[str, Any]
