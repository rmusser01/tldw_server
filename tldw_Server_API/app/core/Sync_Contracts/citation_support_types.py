"""Portable citation-check receipts: validate structure, never confer authority."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_EVALUATED = {"supported", "unsupported", "contradicted"}
MAX_CLAIM_VERIFICATION_BYTES = 2_000_000


class CitationSupportFinding(BaseModel):
    """One exact, ordered slice of historical generated text and its judgment."""

    model_config = ConfigDict(strict=True, extra="allow")
    id: str = Field(min_length=1)
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    evidence_ids: list[str]
    status: Literal["supported", "unsupported", "contradicted", "uncited", "not_checked", "error"]
    reason: str

    @model_validator(mode="after")
    def validate_binding(self) -> CitationSupportFinding:
        if self.end - self.start != len(self.text):
            raise ValueError("Finding coordinates must match its text length")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Finding citation IDs must be unique")
        if self.status in _EVALUATED and not self.evidence_ids:
            raise ValueError("Evaluated findings require cited evidence")
        if self.status == "uncited" and self.evidence_ids:
            raise ValueError("Uncited findings cannot contain citation IDs")
        return self


class CitationSupportCoverage(BaseModel):
    """Mechanical counters, recomputed from findings when validating a report."""

    model_config = ConfigDict(strict=True, extra="allow")
    total: int = Field(ge=0)
    evaluated: int = Field(ge=0)
    supported: int = Field(ge=0)
    uncovered: int = Field(ge=0)
    complete: bool


class CitationSupportReport(BaseModel):
    """Persistable review data, independently bound to a historical text hash."""

    model_config = ConfigDict(strict=True, extra="allow")
    version: Literal[1]
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verdict: Literal["supported", "needs_revision", "unverified"]
    provider: str | None = Field(min_length=1)
    model: str | None = Field(min_length=1)
    coverage: CitationSupportCoverage
    findings: list[CitationSupportFinding]
    evidence: dict[str, dict[str, Any]]
    limits: dict[str, int | float]
    context: dict[str, Any]

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("Report version must be an integer")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> CitationSupportReport:
        ids, previous_end = set(), 0
        for finding in self.findings:
            if finding.id in ids or finding.start < previous_end:
                raise ValueError("Findings must have unique IDs and ordered, nonoverlapping coordinates")
            ids.add(finding.id)
            previous_end = finding.end
            if finding.status in _EVALUATED:
                for citation_id in finding.evidence_ids:
                    evidence = self.evidence.get(citation_id, {})
                    if not isinstance(evidence.get("text_sha256"), str) or not _SHA256.fullmatch(evidence["text_sha256"]):
                        raise ValueError("Evaluated findings require hashed cited evidence")
        for evidence in self.evidence.values():
            for key in ("text_sha256", "attribution_context_sha256"):
                if key in evidence and (not isinstance(evidence[key], str) or not _SHA256.fullmatch(evidence[key])):
                    raise ValueError("Evidence hashes must be SHA-256 hex digests")
        digest = (self.model_extra or {}).get("evidence_sha256")
        if "evidence_sha256" in (self.model_extra or {}) and (not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
            raise ValueError("Evidence digest must be a SHA-256 hex digest")
        total = len(self.findings)
        evaluated = sum(row.status in _EVALUATED for row in self.findings)
        supported = sum(row.status == "supported" for row in self.findings)
        complete = total > 0 and evaluated == total
        expected = (total, evaluated, supported, total - evaluated, complete)
        actual = tuple(getattr(self.coverage, key) for key in ("total", "evaluated", "supported", "uncovered", "complete"))
        if actual != expected:
            raise ValueError("Coverage must match recorded findings")
        verdict = "supported" if complete and supported == total else "unverified"
        if any(row.status in {"unsupported", "contradicted", "uncited"} for row in self.findings):
            verdict = "needs_revision"
        if self.verdict != verdict:
            raise ValueError("Verdict must match recorded findings")
        if (self.provider is None or self.model is None) and (evaluated or verdict != "unverified"):
            raise ValueError("Evaluated reports require provider and model identity")
        return self


def validate_claim_verification(raw: Any) -> dict[str, Any]:
    """Return the original bounded JSON report, without asserting it is current."""
    if type(raw) is not dict:
        raise ValueError("Citation verification must be a JSON object")
    try:
        encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Citation verification must contain finite JSON values") from exc
    if len(encoded) > MAX_CLAIM_VERIFICATION_BYTES:
        raise ValueError(f"Citation verification exceeds {MAX_CLAIM_VERIFICATION_BYTES} UTF-8 bytes")
    pending = [raw]
    while pending:
        value = pending.pop()
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("Citation verification JSON keys must be strings")
            pending.extend(value.values())
        elif type(value) is list:
            pending.extend(value)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("Citation verification must contain only JSON values")
    CitationSupportReport.model_validate(raw)
    return raw
