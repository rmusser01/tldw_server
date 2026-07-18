"""
claims_engine.py - Claim extraction and verification (moved from RAG module)

This module centralizes claim extraction & verification logic under the
Claims_Extraction module so it can be used at ingestion time and by RAG
pipelines. It mirrors the functionality that existed in
app/core/RAG/rag_service/claims.py and is designed to be extended.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.Chat.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,
    await_bounded_daemon_with_timeout,
)
from tldw_Server_API.app.core.Claims_Extraction.alignment import align_claim, align_claim_span
from tldw_Server_API.app.core.Claims_Extraction.analyze_types import ClaimsAnalyzeCallable
from tldw_Server_API.app.core.Claims_Extraction.budget_guard import (
    ClaimsJobBudget,
    ClaimsJobContext,
    estimate_claims_tokens,
    resolve_claims_job_budget,
)
from tldw_Server_API.app.core.Claims_Extraction.extractor_registry import (
    extract_heuristic_claims_texts,
    extract_ner_claims_texts,
    run_async_claims_strategy,
)
from tldw_Server_API.app.core.Claims_Extraction.monitoring import (
    estimate_claims_cost,
    record_claims_alignment_event,
    record_claims_budget_exhausted,
    record_claims_fallback,
    record_claims_output_parse_event,
    record_claims_provider_request,
    record_claims_response_format_selection,
    record_claims_throttle,
    should_throttle_claims_provider,
    suggest_claims_concurrency,
)
from tldw_Server_API.app.core.Claims_Extraction.output_parser import (
    ClaimsOutputParseError,
    ClaimsOutputSchemaError,
    coerce_llm_response_text,
    extract_claim_texts,
    parse_claims_llm_output,
    resolve_claims_response_format,
)
from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_alignment_config as resolve_runtime_alignment_config,
)
from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_context_window_chars as resolve_runtime_context_window_chars,
)
from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_extraction_passes as resolve_runtime_extraction_passes,
)
from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_json_parse_mode as resolve_runtime_parse_mode,
)
from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_llm_config as resolve_runtime_llm_config,
)
from tldw_Server_API.app.core.RAG.rag_service.types import (
    ClaimType,
    Document,
    MatchLevel,
    SourceAuthority,
    VerificationStatus,
)
from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt

_CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS = (
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)

CLAIMS_PROVIDER_CALL_TIMEOUT_SECONDS = 60.0


async def _run_bounded_claims_analyze(
    analyze_fn: ClaimsAnalyzeCallable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run one remote claims adapter with shared capacity and one deadline."""
    return await await_bounded_daemon_with_timeout(
        partial(analyze_fn, *args, **kwargs),
        pool=SYNC_ADAPTER_CALL_POOL,
        name="claims-provider-analyze",
        timeout_seconds=CLAIMS_PROVIDER_CALL_TIMEOUT_SECONDS,
        timeout_message="Claims provider call timed out",
        drain_after_timeout=True,
    )


# --------------------------- Data Models ---------------------------

@dataclass
class Claim:
    """A factual claim extracted from generated text."""
    id: str
    text: str
    span: tuple[int, int] | None = None
    claim_type: ClaimType = ClaimType.GENERAL
    extracted_values: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Evidence snippet from a source document."""
    doc_id: str
    snippet: str
    score: float = 0.0
    authority: SourceAuthority = SourceAuthority.SECONDARY


@dataclass
class ClaimVerification:
    """Result of verifying a claim against evidence."""
    claim: Claim
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    rationale: str | None = None
    match_level: MatchLevel = MatchLevel.INTERPRETATION
    source_authority: SourceAuthority = SourceAuthority.SECONDARY
    requires_external_knowledge: bool = False

    @property
    def label(self) -> str:
        """Backward-compatible label property mapping status to supported/refuted/nei."""
        contested_status = getattr(VerificationStatus, "CONTESTED", None)
        status_to_label = {
            VerificationStatus.VERIFIED: "supported",
            VerificationStatus.REFUTED: "refuted",
            VerificationStatus.CITATION_NOT_FOUND: "nei",
            VerificationStatus.MISQUOTED: "refuted",
            VerificationStatus.MISLEADING: "refuted",
            VerificationStatus.HALLUCINATION: "refuted",
            VerificationStatus.UNVERIFIED: "nei",
            VerificationStatus.NUMERICAL_ERROR: "refuted",
        }
        if contested_status is not None:
            status_to_label[contested_status] = "contested"
        return status_to_label.get(self.status, "nei")


@dataclass
class ExtractionResult:
    """Result from claim extraction phase."""
    claims: list[Claim]
    extractor_mode: str
    extraction_time_s: float = 0.0
    total_input_chars: int = 0


@dataclass
class VerificationResult:
    """Result from claim verification phase."""
    verifications: list[ClaimVerification]
    verification_time_s: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)


# --------------------------- Interfaces ---------------------------

class ClaimExtractor(Protocol):
    async def extract(
        self,
        answer: str,
        max_claims: int = 25,
        *,
        budget: ClaimsJobBudget | None = None,
        job_context: ClaimsJobContext | None = None,
    ) -> list[Claim]:
        ...


class ClaimVerifier(Protocol):
    async def verify(
        self,
        claim: Claim,
        query: str,
        base_documents: list[Document],
        retrieve_fn: Any | None = None,
        top_k: int = 5,
        conf_threshold: float = 0.7,
        mode: str = "hybrid",
        budget: ClaimsJobBudget | None = None,
        job_context: ClaimsJobContext | None = None,
    ) -> ClaimVerification:
        ...


# --------------------------- Utilities ---------------------------

_NUMERIC_RE = re.compile(r"\b(\d{1,3}(?:[\,\._]\d{3})*|\d+)(?:\s*(%|k|m|b))?\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|[A-Za-z]{3,9}\s+\d{4})\b")

# Claim type classification patterns
_STAT_PATTERNS = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(%|percent|percentage|million|billion|thousand)|"
    r"\b(grew|increased|decreased|fell|rose|dropped)\s+(?:by\s+)?\d+",
    re.IGNORECASE
)
_COMPARATIVE_PATTERNS = re.compile(
    r"\b(more|less|greater|fewer|larger|smaller|higher|lower|better|worse)\s+than\b|"
    r"\b(compared\s+to|relative\s+to|versus|vs\.?)\b",
    re.IGNORECASE
)
_TEMPORAL_PATTERNS = re.compile(
    r"\b(in\s+\d{4}|since\s+\d{4}|before\s+\d{4}|after\s+\d{4}|"
    r"during\s+\d{4}|as\s+of\s+\d{4}|by\s+\d{4})\b|"
    r"\b(yesterday|today|last\s+(?:week|month|year)|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})\b",
    re.IGNORECASE
)
_ATTRIBUTION_PATTERNS = re.compile(
    r"\b(according\s+to|said|stated|reported|claimed|wrote|noted|argued)\b|"
    r"\"[^\"]+\"\s*,?\s*(?:said|wrote|stated)",
    re.IGNORECASE
)
_CAUSAL_PATTERNS = re.compile(
    r"\b(caused|led\s+to|resulted\s+in|because\s+of|due\s+to|"
    r"as\s+a\s+result|consequently|therefore|thus)\b",
    re.IGNORECASE
)
_EXISTENCE_PATTERNS = re.compile(
    r"\b(there\s+(?:is|are|exists?|was|were)|"
    r"exists?\b|contains?\b|includes?\b)\b",
    re.IGNORECASE
)
_RANKING_PATTERNS = re.compile(
    r"\b(largest|smallest|biggest|highest|lowest|best|worst|"
    r"most|least|first|last|top|bottom|leading|primary)\b",
    re.IGNORECASE
)
_QUOTE_PATTERNS = re.compile(r'"[^"]{10,}"|\u201c[^\u201d]{10,}\u201d')

# Source authority classification patterns
_GOV_PATTERNS = re.compile(r"\.gov\b|government|official\s+(?:data|statistics|report)", re.IGNORECASE)
_PEER_REVIEWED_PATTERNS = re.compile(r"\bdoi\b|journal|peer[\-\s]?review|published\s+in", re.IGNORECASE)
_PRIMARY_PATTERNS = re.compile(r"original\s+(?:research|study|data)|primary\s+source|first[\-\s]?hand", re.IGNORECASE)
_INDUSTRY_PATTERNS = re.compile(r"whitepaper|industry\s+report|market\s+(?:research|analysis)", re.IGNORECASE)

_CLAIMS_EXTRACT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": True,
}

_CLAIMS_VERIFY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["label"],
    "additionalProperties": True,
}


def _extract_numbers_and_dates(text: str) -> dict[str, list[str]]:
    nums = [m.group(0) for m in _NUMERIC_RE.finditer(text or "")]
    dates = [m.group(0) for m in _DATE_RE.finditer(text or "")]
    return {"numbers": nums, "dates": dates}


def classify_claim_type(claim_text: str) -> tuple[ClaimType, dict[str, Any]]:
    """
    Classify a claim into its type based on pattern matching.

    Returns:
        Tuple of (ClaimType, extracted_values dict)
    """
    text = (claim_text or "").strip()
    extracted: dict[str, Any] = {}

    # Check for quotes first (highest priority)
    if _QUOTE_PATTERNS.search(text):
        quotes = _QUOTE_PATTERNS.findall(text)
        extracted["quotes"] = [q.strip('""\u201c\u201d') for q in quotes]  # noqa: B005
        return ClaimType.QUOTE, extracted

    # Check for statistics (numbers with context)
    if _STAT_PATTERNS.search(text):
        nums = _NUMERIC_RE.findall(text)
        extracted["numbers"] = [n[0] if isinstance(n, tuple) else n for n in nums]
        return ClaimType.STATISTIC, extracted

    # Check for rankings
    if _RANKING_PATTERNS.search(text):
        return ClaimType.RANKING, extracted

    # Check for comparatives
    if _COMPARATIVE_PATTERNS.search(text):
        return ClaimType.COMPARATIVE, extracted

    # Check for temporal claims
    if _TEMPORAL_PATTERNS.search(text):
        dates = _DATE_RE.findall(text)
        extracted["dates"] = dates
        return ClaimType.TEMPORAL, extracted

    # Check for attribution
    if _ATTRIBUTION_PATTERNS.search(text):
        return ClaimType.ATTRIBUTION, extracted

    # Check for causal claims
    if _CAUSAL_PATTERNS.search(text):
        return ClaimType.CAUSAL, extracted

    # Check for existence claims
    if _EXISTENCE_PATTERNS.search(text):
        return ClaimType.EXISTENCE, extracted

    return ClaimType.GENERAL, extracted


def classify_source_authority(doc: Document) -> SourceAuthority:
    """
    Classify the authority level of a source document.

    Uses heuristics based on metadata (URL, DOI, source type).
    """
    metadata = getattr(doc, "metadata", {}) or {}
    content = getattr(doc, "content", "") or ""

    # Check URL patterns
    url = str(metadata.get("url", "") or metadata.get("source_url", "") or "")

    # Primary source indicators
    if _PRIMARY_PATTERNS.search(content) or metadata.get("source_type") == "primary":
        return SourceAuthority.PRIMARY

    # Government sources
    if _GOV_PATTERNS.search(url) or _GOV_PATTERNS.search(content):
        return SourceAuthority.GOVERNMENT

    # Peer-reviewed sources
    if metadata.get("doi") or _PEER_REVIEWED_PATTERNS.search(content):
        return SourceAuthority.PEER_REVIEWED

    # Industry sources
    if _INDUSTRY_PATTERNS.search(content) or metadata.get("source_type") == "whitepaper":
        return SourceAuthority.INDUSTRY

    return SourceAuthority.SECONDARY


def classify_match_confidence(
    claim_text: str,
    evidence_text: str,
    semantic_score: float = 0.0
) -> tuple[MatchLevel, float]:
    """
    Classify the match level between a claim and evidence.

    Uses lexical overlap and semantic similarity to determine match confidence.

    Returns:
        Tuple of (MatchLevel, confidence_score)
    """
    claim = (claim_text or "").strip().lower()
    evidence = (evidence_text or "").strip().lower()

    if not claim or not evidence:
        return MatchLevel.INTERPRETATION, 0.0

    # Exact match check
    if claim in evidence or evidence in claim:
        return MatchLevel.EXACT, 1.0

    # Calculate word overlap (Jaccard similarity)
    claim_words = set(re.findall(r'\w+', claim))
    evidence_words = set(re.findall(r'\w+', evidence))

    if not claim_words or not evidence_words:
        return MatchLevel.INTERPRETATION, semantic_score

    intersection = claim_words & evidence_words
    union = claim_words | evidence_words
    jaccard = len(intersection) / len(union) if union else 0.0

    # High lexical overlap suggests paraphrase
    if jaccard >= 0.6:
        return MatchLevel.EXACT, max(jaccard, semantic_score)
    elif jaccard >= 0.3:
        return MatchLevel.PARAPHRASE, max(jaccard, semantic_score)
    else:
        # Rely on semantic score for interpretation
        if semantic_score >= 0.7:
            return MatchLevel.PARAPHRASE, semantic_score
        return MatchLevel.INTERPRETATION, max(jaccard, semantic_score)


def determine_verification_status(
    claim: Claim,
    evidence_snippets: list[Evidence],
    nli_label: str | None = None,
    nli_confidence: float = 0.0,
    numeric_match: bool | None = None,
    quote_match: bool | None = None,
    doc_only_mode: bool = False,
) -> tuple[VerificationStatus, float, str]:
    """
    Decision tree for determining verification status.

    Args:
        claim: The claim being verified
        evidence_snippets: Evidence found for the claim
        nli_label: NLI model label if available (entailment/contradiction/neutral)
        nli_confidence: NLI model confidence score
        numeric_match: Whether numeric values match (True/False/None if not checked)
        quote_match: Whether quoted text matches source (True/False/None if not a quote)
        doc_only_mode: If True, require evidence from documents

    Returns:
        Tuple of (VerificationStatus, confidence, rationale)
    """
    # No evidence at all
    if not evidence_snippets:
        if doc_only_mode:
            return VerificationStatus.CITATION_NOT_FOUND, 0.0, "No supporting documents found"
        return VerificationStatus.UNVERIFIED, 0.0, "No evidence available"

    # Quote verification
    if claim.claim_type == ClaimType.QUOTE:
        if quote_match is True:
            return VerificationStatus.VERIFIED, 0.95, "Quote matches source"
        elif quote_match is False:
            return VerificationStatus.MISQUOTED, 0.9, "Quote does not match source text"

    # Numeric verification
    if claim.claim_type == ClaimType.STATISTIC:
        if numeric_match is False:
            return VerificationStatus.NUMERICAL_ERROR, 0.85, "Numeric values do not match source"
        elif numeric_match is True:
            # Continue to check NLI for context
            pass

    # NLI-based verification
    if nli_label:
        nli_label_lower = nli_label.lower()
        if nli_label_lower == "entailment" or nli_label_lower == "supported":
            if nli_confidence >= 0.8:
                return VerificationStatus.VERIFIED, nli_confidence, f"NLI entailment ({nli_confidence:.2f})"
            elif nli_confidence >= 0.6:
                return VerificationStatus.VERIFIED, nli_confidence, f"NLI entailment (moderate: {nli_confidence:.2f})"
        elif nli_label_lower == "contradiction" or nli_label_lower == "refuted":
            if nli_confidence >= 0.7:
                return VerificationStatus.REFUTED, nli_confidence, f"NLI contradiction ({nli_confidence:.2f})"
        # Check for potentially misleading: high retrieval score but NLI neutral
        # This indicates evidence exists but doesn't clearly support the claim,
        # which may suggest the claim is technically related but deceptive in context
        elif nli_label_lower == "neutral" and nli_confidence >= 0.6:
            max_retrieval_score = max((e.score for e in evidence_snippets), default=0.0)
            if max_retrieval_score >= 0.7:
                return VerificationStatus.MISLEADING, 0.6, "Evidence found but context may be misleading"
        # Neutral or low confidence
        if nli_confidence < 0.5:
            return VerificationStatus.UNVERIFIED, nli_confidence, "Insufficient NLI confidence"

    # Fallback: check evidence quality
    max_score = max((e.score for e in evidence_snippets), default=0.0)
    max_authority = max((e.authority.value for e in evidence_snippets), default=1)

    if max_score >= 0.8 and max_authority >= 3:
        return VerificationStatus.VERIFIED, max_score, "High-quality evidence from authoritative source"
    elif max_score >= 0.6:
        return VerificationStatus.VERIFIED, max_score * 0.9, "Moderate evidence support"
    elif max_score >= 0.3:
        return VerificationStatus.UNVERIFIED, max_score, "Weak evidence - needs manual review"

    if doc_only_mode:
        return VerificationStatus.HALLUCINATION, 0.7, "No supporting evidence in provided documents"

    return VerificationStatus.UNVERIFIED, max_score, "Insufficient evidence"


def _contains_any(hay: str, needles: list[str]) -> bool:
    hay_l = (hay or "").lower()
    return any(n.lower() in hay_l for n in needles if n)


def _first_n_snippets(docs: list[Document], n: int = 3, snippet_len: int = 480) -> list[Evidence]:
    out: list[Evidence] = []
    for d in docs[:n]:
        txt = d.content or ""
        snip = txt[:snippet_len] + ("..." if len(txt) > snippet_len else "")
        authority = classify_source_authority(d)
        out.append(Evidence(
            doc_id=getattr(d, "id", ""),
            snippet=snip,
            score=getattr(d, "score", 0.0),
            authority=authority
        ))
    return out


def _find_offsets(doc_text: str, claim_text: str, snippet: str) -> tuple[int, int]:
    """Best-effort exact offsets into the full document.

    Strategy:
    1) Exact match on claim_text/snippet
    2) Normalized match (case/whitespace)
    3) Anchored window match for long text
    4) Fallback to (0, min(len(doc_text), len(snippet)))
    """
    if not isinstance(doc_text, str) or not doc_text:
        return (0, 0)
    ct = (claim_text or "").strip()
    snip = (snippet or "").strip()
    if snip.endswith("..."):
        snip = snip[:-3]
    if snip.endswith("…"):
        snip = snip[:-1]

    span = align_claim_span(doc_text, ct, mode="fuzzy", threshold=0.6)
    if span is None and snip:
        span = align_claim_span(doc_text, snip, mode="fuzzy", threshold=0.6)
    if span is not None:
        return span

    return (0, min(len(doc_text), max(len(ct), len(snip))))


def _resolve_claims_llm_config() -> tuple[str, str | None, float]:
    return resolve_runtime_llm_config(default_temperature=0.1)


def _resolve_claims_json_parse_mode() -> str:
    return resolve_runtime_parse_mode(default_mode="lenient")


def _resolve_claims_alignment_config() -> tuple[str, float]:
    return resolve_runtime_alignment_config(default_mode="fuzzy", default_threshold=0.75)


def _resolve_claims_context_window_chars() -> int:
    """Resolve the context window size used around extracted claim spans.

    Args:
        None.

    Returns:
        int: Context window size in characters.

    Notes:
        Delegates to `resolve_runtime_context_window_chars(default=0)`. When no
        valid runtime override is available, the resolver falls back to `0`.
    """
    return resolve_runtime_context_window_chars(default=0)


def _resolve_claims_extraction_passes() -> int:
    """Resolve how many claim-extraction passes should be executed.

    Args:
        None.

    Returns:
        int: Number of extraction passes to run.

    Notes:
        Delegates to `resolve_runtime_extraction_passes(default=1)`. When no
        valid runtime override is available, the resolver falls back to `1`.
    """
    return resolve_runtime_extraction_passes(default=1)


def _normalize_claim_text(value: str) -> str:
    """Trim, lowercase, and collapse whitespace; return empty string for None/empty input."""
    if not value:
        return ""
    return " ".join(str(value).strip().lower().split())


def _spans_overlap(
    first: tuple[int, int] | None,
    second: tuple[int, int] | None,
) -> bool:
    """Check whether two character spans overlap under claims-engine semantics.

    Args:
        first: Optional `(start, end)` span for the first claim.
        second: Optional `(start, end)` span for the second claim.

    Returns:
        bool: `True` if spans overlap, or if either span is `None`; otherwise
        `False`.

    Notes:
        Concrete spans use half-open interval overlap rules:
        `first[0] < second[1] and second[0] < first[1]`.
    """
    if first is None or second is None:
        return True
    return first[0] < second[1] and second[0] < first[1]


def _claims_local_nli_enabled() -> bool:
    try:
        from tldw_Server_API.app.core.config import settings as _settings  # type: ignore
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        return False
    value = _settings.get("CLAIMS_ENABLE_LOCAL_NLI", False)
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on", "enabled"}


async def _log_claims_llm_usage(
    *,
    job_context: ClaimsJobContext | None,
    operation: str,
    provider: str,
    model: str,
    prompt_text: str,
    response_text: str,
    latency_ms: int,
    status: int,
    estimated: bool = True,
) -> None:
    try:
        from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        return
    try:
        user_id = job_context.user_id if job_context else None
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        user_id = None
    try:
        api_key_id = job_context.api_key_id if job_context else None
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        api_key_id = None
    try:
        request_id = job_context.request_id if job_context else None
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        request_id = None
    endpoint = None
    try:
        endpoint = job_context.endpoint if job_context else None
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        endpoint = None
    remote_ip = None
    try:
        remote_ip = getattr(job_context, "remote_ip", None) if job_context else None
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        remote_ip = None
    user_agent = None
    try:
        user_agent = getattr(job_context, "user_agent", None) if job_context else None
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        user_agent = None
    token_name = None
    try:
        token_name = getattr(job_context, "token_name", None) if job_context else None
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        token_name = None
    conversation_id = None
    try:
        conversation_id = getattr(job_context, "conversation_id", None) if job_context else None
    except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
        conversation_id = None
    await log_llm_usage(
        user_id=user_id,
        key_id=api_key_id,
        endpoint=endpoint or "claims_engine",
        operation=operation,
        provider=provider,
        model=model or "",
        status=int(status),
        latency_ms=int(latency_ms),
        prompt_tokens=estimate_claims_tokens(prompt_text),
        completion_tokens=estimate_claims_tokens(response_text),
        total_tokens=None,
        request_id=request_id,
        remote_ip=remote_ip,
        user_agent=user_agent,
        token_name=token_name,
        conversation_id=conversation_id,
        estimated=estimated,
    )


# --------------------------- Extractors ---------------------------

class HeuristicSentenceExtractor:
    async def extract(self, answer: str, max_claims: int = 25) -> list[Claim]:
        if not answer or not isinstance(answer, str):
            return []
        parts = extract_heuristic_claims_texts(answer, max_claims)
        claims: list[Claim] = []
        for i, p in enumerate(parts[:max_claims]):
            claims.append(Claim(id=f"c{i+1}", text=p.strip(), span=None))
        return claims


class LLMBasedClaimExtractor:
    """Prompt an LLM to extract decontextualized atomic propositions as JSON."""

    def __init__(self, analyze_fn: ClaimsAnalyzeCallable):
        self._analyze = analyze_fn

    async def extract(
        self,
        answer: str,
        max_claims: int = 25,
        *,
        budget: ClaimsJobBudget | None = None,
        job_context: ClaimsJobContext | None = None,
    ) -> list[Claim]:
        if not answer:
            return []
        system = load_prompt("ingestion", "claims_extractor_system") or (
            "You extract specific, verifiable, decontextualized factual propositions. Output strict JSON."
        )
        base = load_prompt("ingestion", "claims_extractor_prompt") or (
            "Extract up to {max_claims} atomic factual propositions from the ANSWER. "
            "Each proposition should stand alone without the surrounding context, be specific and checkable. "
            "Return JSON: {{\"claims\":[{{\"text\": str}}]}}. Do not include explanations.\n\nANSWER:\n{answer}"
        )
        # Safely format template that may contain JSON braces
        try:
            prompt = base.format(max_claims=max_claims, answer=answer)
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
            # Escape all braces then restore placeholders
            _tmpl = base.replace('{', '{{').replace('}', '}}')
            _tmpl = _tmpl.replace('{{max_claims}}', '{max_claims}').replace('{{answer}}', '{answer}')
            prompt = _tmpl.format(max_claims=max_claims, answer=answer)

        provider, model_override, temperature = _resolve_claims_llm_config()
        parse_mode = _resolve_claims_json_parse_mode()
        response_format = resolve_claims_response_format(
            provider or "openai",
            schema_name="claims_extraction",
            json_schema=_CLAIMS_EXTRACT_RESPONSE_SCHEMA,
        )
        record_claims_response_format_selection(
            provider=provider or "openai",
            model=model_override or "",
            mode="extract",
            response_format=response_format,
        )
        cost_estimate = estimate_claims_cost(
            provider=provider or "openai",
            model=model_override or "",
            text=prompt,
        )
        budget_ratio = budget.remaining_ratio() if budget is not None else None
        throttle, reason = should_throttle_claims_provider(
            provider=provider or "openai",
            model=model_override or "",
            budget_ratio=budget_ratio,
        )
        if throttle:
            record_claims_throttle(
                provider=provider or "openai",
                model=model_override or "",
                mode="extract",
                reason=reason or "throttle",
            )
            record_claims_fallback(
                provider=provider or "openai",
                model=model_override or "",
                mode="extract",
                reason=reason or "throttle",
            )
            return await HeuristicSentenceExtractor().extract(answer, max_claims)
        if budget is not None:
            prompt_tokens = estimate_claims_tokens(prompt)
            if not budget.reserve(cost_usd=cost_estimate, tokens=prompt_tokens):
                record_claims_budget_exhausted(
                    provider=provider or "openai",
                    model=model_override or "",
                    mode="extract",
                    reason=budget.exhausted_reason or "budget",
                )
                if budget.strict:
                    return []
                record_claims_fallback(
                    provider=provider or "openai",
                    model=model_override or "",
                    mode="extract",
                    reason=budget.exhausted_reason or "budget",
                )
                return await HeuristicSentenceExtractor().extract(answer, max_claims)
        try:
            start_time = time.time()
            raw = await _run_bounded_claims_analyze(
                self._analyze,
                provider or "openai",
                answer,
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
            )
            latency_s = time.time() - start_time
            record_claims_provider_request(
                provider=provider or "openai",
                model=model_override or "",
                mode="extract",
                latency_s=latency_s,
                estimated_cost=cost_estimate,
            )
            text = coerce_llm_response_text(raw)
            if budget is not None:
                budget.add_usage(tokens=estimate_claims_tokens(text))
            with contextlib.suppress(_CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS):
                await _log_claims_llm_usage(
                    job_context=job_context,
                    operation="claims_extract",
                    provider=provider or "openai",
                    model=model_override or "",
                    prompt_text=prompt,
                    response_text=text,
                    latency_ms=int(latency_s * 1000),
                    status=200,
                    estimated=True,
                )
            parsed = parse_claims_llm_output(
                text,
                parse_mode=parse_mode,
                strip_think_tags=True,
            )
            claim_texts = extract_claim_texts(
                parsed,
                wrapper_key="claims",
                parse_mode=parse_mode,
                max_claims=max_claims,
            )
            out: list[Claim] = [Claim(id=f"c{i+1}", text=ct) for i, ct in enumerate(claim_texts[:max_claims])]
            if not out:
                record_claims_output_parse_event(
                    provider=provider or "openai",
                    model=model_override or "",
                    mode="extract",
                    parse_mode=parse_mode,
                    outcome="empty",
                    reason="no_claims",
                )
                record_claims_fallback(
                    provider=provider or "openai",
                    model=model_override or "",
                    mode="extract",
                    reason="empty_claims",
                )
                logger.debug("LLM extractor returned no claims; falling back to heuristics")
                return await HeuristicSentenceExtractor().extract(answer, max_claims)
            record_claims_output_parse_event(
                provider=provider or "openai",
                model=model_override or "",
                mode="extract",
                parse_mode=parse_mode,
                outcome="success",
            )
            return out
        except ClaimsOutputParseError as e:
            record_claims_output_parse_event(
                provider=provider or "openai",
                model=model_override or "",
                mode="extract",
                parse_mode=parse_mode,
                outcome="error",
                reason=e.__class__.__name__,
            )
            record_claims_fallback(
                provider=provider or "openai",
                model=model_override or "",
                mode="extract",
                reason="parse_error",
            )
            logger.warning(f"Claim extraction JSON parse failed; falling back to heuristics: {e}")
            return await HeuristicSentenceExtractor().extract(answer, max_claims)
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS as e:
            record_claims_provider_request(
                provider=provider or "openai",
                model=model_override or "",
                mode="extract",
                latency_s=None,
                error=str(e),
                estimated_cost=cost_estimate,
            )
            with contextlib.suppress(_CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS):
                await _log_claims_llm_usage(
                    job_context=job_context,
                    operation="claims_extract",
                    provider=provider or "openai",
                    model=model_override or "",
                    prompt_text=prompt,
                    response_text="",
                    latency_ms=0,
                    status=500,
                    estimated=True,
                )
            logger.warning(f"Claim extraction via LLM failed: {e}")
            record_claims_fallback(
                provider=provider or "openai",
                model=model_override or "",
                mode="extract",
                reason="provider_error",
            )
            return await HeuristicSentenceExtractor().extract(answer, max_claims)


# --------------------------- Verifier ---------------------------

class HybridClaimVerifier:
    """Verify with numeric/date checks, retrieve evidence, then LLM-judge entailment."""

    def __init__(self, analyze_fn: ClaimsAnalyzeCallable, nli_model: str | None = None):
        self._analyze = analyze_fn
        self._nli = None
        self._nli_lock = asyncio.Lock()
        self._nli_load_attempted = False
        self._nli_model_name = nli_model

    async def _get_nli(self):
        async with self._nli_lock:
            if self._nli is not None:
                return self._nli
            if not _claims_local_nli_enabled():
                return None
            if self._nli_load_attempted:
                return None
            self._nli_load_attempted = True

            def _load():
                try:
                    import os

                    from transformers import pipeline  # type: ignore

                    model_name = (
                        self._nli_model_name
                        or os.environ.get("RAG_NLI_MODEL")
                        or os.environ.get("RAG_NLI_MODEL_PATH")
                        or "roberta-large-mnli"
                    )
                    return pipeline("text-classification", model=model_name, return_all_scores=True)
                except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"NLI model load failed: {e}.")
                    return None

            try:
                loop = asyncio.get_running_loop()
                self._nli = await loop.run_in_executor(None, _load)
            except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
                self._nli = None
            return self._nli

    @staticmethod
    def _nli_best_label(nli_scores: list[dict[str, Any]]) -> tuple[str, float]:
        label_map = {"entailment": "supported", "contradiction": "refuted", "neutral": "nei"}
        best = max(nli_scores, key=lambda x: x.get("score", 0.0))
        lab = label_map.get(str(best.get("label", "")).lower(), "nei")
        return lab, float(best.get("score", 0.0))

    async def verify(
        self,
        claim: Claim,
        query: str,
        base_documents: list[Document],
        retrieve_fn: Any | None = None,
        top_k: int = 5,
        conf_threshold: float = 0.7,
        mode: str = "hybrid",
        budget: ClaimsJobBudget | None = None,
        job_context: ClaimsJobContext | None = None,
        doc_only_mode: bool = False,
        numeric_precision_mode: str = "standard",
    ) -> ClaimVerification:
        claim_text = claim.text.strip()
        nums_dates = _extract_numbers_and_dates(claim_text)

        # Classify claim type if not already done
        if claim.claim_type == ClaimType.GENERAL:
            claim.claim_type, claim.extracted_values = classify_claim_type(claim_text)

        candidate_docs: list[Document] = []
        try:
            if retrieve_fn is not None:
                candidate_docs = await retrieve_fn(claim_text, top_k=top_k)
            else:
                candidate_docs = base_documents[:top_k] if base_documents else []
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Claim-level retrieval failed, using base docs: {e}")
            candidate_docs = base_documents[:top_k] if base_documents else []

        def _score_doc(d: Document) -> float:
            score = float(getattr(d, "score", 0.0) or 0.0)
            bonus = 0.0
            if nums_dates["numbers"] and _contains_any(d.content, nums_dates["numbers"]):
                bonus += 0.2
            if nums_dates["dates"] and _contains_any(d.content, nums_dates["dates"]):
                bonus += 0.2
            return score + bonus

        candidate_docs = sorted(candidate_docs, key=_score_doc, reverse=True)
        evidence_snips = _first_n_snippets(candidate_docs, n=min(3, top_k))
        evidence_text = "\n\n".join([f"[doc:{e.doc_id}] {e.snippet}" for e in evidence_snips])
        # Map doc_id -> full text for citation offsets
        _doc_map: dict[str, str] = {}
        try:
            for d in candidate_docs:
                _doc_map[str(getattr(d, "id", ""))] = getattr(d, "content", "") or ""
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
            _doc_map = {}

        # NLI verification path (optional depending on mode)
        nli = None
        if mode in ("hybrid", "nli"):
            nli = await self._get_nli()
            nli_decision: tuple[str, float, int] | None = None
            if nli is not None:
                try:
                    best_tuple = ("nei", 0.0, -1)
                    for i, ev in enumerate(evidence_snips):
                        scores = nli({"text": ev.snippet, "text_pair": claim_text})
                        if isinstance(scores, list) and scores:
                            lab, sc = self._nli_best_label(scores[0])
                            if sc > best_tuple[1]:
                                best_tuple = (lab, sc, i)
                    if best_tuple[1] >= conf_threshold:
                        # Populate citations only for supported/refuted with best-effort exact offsets
                        cit: list[dict[str, Any]] = []
                        nli_status = VerificationStatus.VERIFIED if best_tuple[0] == "supported" else VerificationStatus.REFUTED
                        if best_tuple[0] in {"supported", "refuted"}:
                            for ev in evidence_snips:
                                full = _doc_map.get(ev.doc_id, "")
                                s, e = _find_offsets(full, claim_text, ev.snippet)
                                cit.append({"doc_id": ev.doc_id, "start": int(s), "end": int(e)})
                        # Classify match level
                        best_ev_text = evidence_snips[best_tuple[2]].snippet if best_tuple[2] >= 0 else ""
                        match_lvl, _ = classify_match_confidence(claim_text, best_ev_text, best_tuple[1])
                        max_auth = max((ev.authority for ev in evidence_snips), default=SourceAuthority.SECONDARY)
                        return ClaimVerification(
                            claim=claim,
                            status=nli_status,
                            confidence=best_tuple[1],
                            evidence=evidence_snips,
                            citations=cit,
                            rationale=f"NLI-{best_tuple[0]} {best_tuple[1]:.2f}",
                            match_level=match_lvl,
                            source_authority=max_auth,
                        )
                    else:
                        nli_decision = best_tuple
                except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS as e:
                    if mode == "hybrid":
                        logger.warning(f"NLI verification failed; falling back to LLM judge: {e}")
                    else:
                        logger.warning(f"NLI verification failed under nli mode: {e}")

            # If mode is strict NLI and we didn't return, finish with UNVERIFIED (no LLM fallback)
            if mode == "nli":
                conf = 0.0
                if nli_decision is not None:
                    conf = float(nli_decision[1])
                return ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.UNVERIFIED,
                    confidence=conf,
                    evidence=evidence_snips,
                    citations=[],
                    rationale="NLI unavailable/low confidence",
                )

        system = "You are a precise fact-checking judge. Output strict JSON only."
        judge_prompt = (
            "Given the EVIDENCE snippets and a CLAIM, decide if the claim is Supported, Refuted, or NotEnoughInfo. "
            "Return JSON as {\"label\": \"supported|refuted|nei\", \"confidence\": float, \"rationale\": str}.\n\n"
            f"CLAIM: {claim_text}\n\nEVIDENCE:\n{evidence_text}"
        )
        label = "nei"
        confidence = 0.5
        rationale = None
        provider, model_override, temperature = _resolve_claims_llm_config()
        parse_mode = _resolve_claims_json_parse_mode()
        response_format = resolve_claims_response_format(
            provider or "openai",
            schema_name="claims_verification",
            json_schema=_CLAIMS_VERIFY_RESPONSE_SCHEMA,
        )
        record_claims_response_format_selection(
            provider=provider or "openai",
            model=model_override or "",
            mode="verify",
            response_format=response_format,
        )
        cost_estimate = estimate_claims_cost(
            provider=provider or "openai",
            model=model_override or "",
            text=f"{system}\n{judge_prompt}",
        )
        budget_ratio = budget.remaining_ratio() if budget is not None else None
        throttle, reason = should_throttle_claims_provider(
            provider=provider or "openai",
            model=model_override or "",
            budget_ratio=budget_ratio,
        )
        if throttle:
            record_claims_throttle(
                provider=provider or "openai",
                model=model_override or "",
                mode="verify",
                reason=reason or "throttle",
            )
            record_claims_fallback(
                provider=provider or "openai",
                model=model_override or "",
                mode="verify",
                reason=reason or "throttle",
            )
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.UNVERIFIED,
                confidence=0.0,
                evidence=evidence_snips,
                citations=[],
                rationale="Throttled by provider health",
            )
        if budget is not None:
            prompt_tokens = estimate_claims_tokens(judge_prompt)
            if not budget.reserve(cost_usd=cost_estimate, tokens=prompt_tokens):
                record_claims_budget_exhausted(
                    provider=provider or "openai",
                    model=model_override or "",
                    mode="verify",
                    reason=budget.exhausted_reason or "budget",
                )
                record_claims_fallback(
                    provider=provider or "openai",
                    model=model_override or "",
                    mode="verify",
                    reason=budget.exhausted_reason or "budget",
                )
                return ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.UNVERIFIED,
                    confidence=0.0,
                    evidence=evidence_snips,
                    citations=[],
                    rationale="Budget exhausted",
                )
        try:
            start_time = time.time()
            raw = await _run_bounded_claims_analyze(
                self._analyze,
                provider or "openai",
                claim_text,
                judge_prompt,
                None,
                system,
                temperature,
                model_override=model_override,
                response_format=response_format,
            )
            latency_s = time.time() - start_time
            record_claims_provider_request(
                provider=provider or "openai",
                model=model_override or "",
                mode="verify",
                latency_s=latency_s,
                estimated_cost=cost_estimate,
            )
            text = coerce_llm_response_text(raw)
            if budget is not None:
                budget.add_usage(tokens=estimate_claims_tokens(text))
            with contextlib.suppress(_CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS):
                await _log_claims_llm_usage(
                    job_context=job_context,
                    operation="claims_verify",
                    provider=provider or "openai",
                    model=model_override or "",
                    prompt_text=judge_prompt,
                    response_text=text,
                    latency_ms=int(latency_s * 1000),
                    status=200,
                    estimated=True,
                )
            data = parse_claims_llm_output(
                text,
                parse_mode=parse_mode,
                strip_think_tags=True,
            )
            if not isinstance(data, dict):
                raise ClaimsOutputSchemaError("Verifier response must be a JSON object.")
            record_claims_output_parse_event(
                provider=provider or "openai",
                model=model_override or "",
                mode="verify",
                parse_mode=parse_mode,
                outcome="success",
            )
            lab = str(data.get("label", "nei")).lower().strip()
            if lab in {"supported", "refuted", "nei"}:
                label = lab
            confidence = float(data.get("confidence", confidence))
            rationale = data.get("rationale")
        except ClaimsOutputParseError as e:
            record_claims_output_parse_event(
                provider=provider or "openai",
                model=model_override or "",
                mode="verify",
                parse_mode=parse_mode,
                outcome="error",
                reason=e.__class__.__name__,
            )
            record_claims_fallback(
                provider=provider or "openai",
                model=model_override or "",
                mode="verify",
                reason="parse_error",
            )
            logger.warning(f"LLM judge parse failed; defaulting to NEI: {e}")
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS as e:
            record_claims_provider_request(
                provider=provider or "openai",
                model=model_override or "",
                mode="verify",
                latency_s=None,
                error=str(e),
                estimated_cost=cost_estimate,
            )
            with contextlib.suppress(_CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS):
                await _log_claims_llm_usage(
                    job_context=job_context,
                    operation="claims_verify",
                    provider=provider or "openai",
                    model=model_override or "",
                    prompt_text=judge_prompt,
                    response_text="",
                    latency_ms=0,
                    status=500,
                    estimated=True,
                )
            record_claims_fallback(
                provider=provider or "openai",
                model=model_override or "",
                mode="verify",
                reason="provider_error",
            )
            logger.warning(f"LLM judge failed; defaulting to NEI: {e}")
        # Check numeric precision for statistic claims
        numeric_match: bool | None = None
        if claim.claim_type == ClaimType.STATISTIC and claim.extracted_values.get("numbers"):
            try:
                from tldw_Server_API.app.core.RAG.rag_service.guardrails import check_numeric_precision
                num_result = check_numeric_precision(
                    claim_text,
                    candidate_docs,
                    mode=numeric_precision_mode,
                )
                numeric_match = num_result.all_within_tolerance
            except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
                pass

        # Check quote match for quote claims
        quote_match: bool | None = None
        if claim.claim_type == ClaimType.QUOTE:
            quotes = claim.extracted_values.get("quotes", [])
            if quotes:
                quote_match = False
                combined_evidence = " ".join(ev.snippet for ev in evidence_snips)
                for q in quotes:
                    if q.lower() in combined_evidence.lower():
                        quote_match = True
                        break

        # Use decision tree to determine final status
        status, decision_conf, decision_rationale = determine_verification_status(
            claim=claim,
            evidence_snippets=evidence_snips,
            nli_label=label,
            nli_confidence=confidence,
            numeric_match=numeric_match,
            quote_match=quote_match,
            doc_only_mode=doc_only_mode,
        )

        # Override with LLM-based result if it's more confident
        if label in {"supported", "refuted"} and confidence > decision_conf:
            status = VerificationStatus.VERIFIED if label == "supported" else VerificationStatus.REFUTED
            decision_rationale = rationale or decision_rationale
            decision_conf = confidence

        # Construct citations for traceability (doc IDs with snippet offsets)
        citations: list[dict[str, Any]] = []
        if status in {VerificationStatus.VERIFIED, VerificationStatus.REFUTED, VerificationStatus.MISQUOTED, VerificationStatus.NUMERICAL_ERROR}:
            for ev in evidence_snips:
                full = _doc_map.get(ev.doc_id, "")
                s, e = _find_offsets(full, claim_text, ev.snippet)
                citations.append({"doc_id": ev.doc_id, "start": int(s), "end": int(e)})

        # Classify match level
        best_ev_text = evidence_snips[0].snippet if evidence_snips else ""
        match_lvl, _ = classify_match_confidence(claim_text, best_ev_text, decision_conf)
        max_auth = max((ev.authority for ev in evidence_snips), default=SourceAuthority.SECONDARY)

        # Check if external knowledge is required
        requires_external = (
            status == VerificationStatus.UNVERIFIED and
            not evidence_snips and
            doc_only_mode
        )

        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=max(0.0, min(1.0, decision_conf)),
            evidence=evidence_snips,
            citations=citations,
            rationale=decision_rationale,
            match_level=match_lvl,
            source_authority=max_auth,
            requires_external_knowledge=requires_external,
        )


class ClaimsEngine:
    """High-level entry: extracts and verifies claims for a generated answer."""

    def __init__(self, analyze_fn: ClaimsAnalyzeCallable):
        self.extractor_llm = LLMBasedClaimExtractor(analyze_fn)
        self.extractor_heur = HeuristicSentenceExtractor()
        self._analyze = analyze_fn
        self.verifier = HybridClaimVerifier(analyze_fn)

    @staticmethod
    def _build_claims_from_texts(
        *,
        source_text: str,
        claim_texts: list[str],
        max_claims: int,
        alignment_context: str = "engine_extract",
    ) -> list[Claim]:
        alignment_mode, alignment_threshold = _resolve_claims_alignment_config()
        claims: list[Claim] = []
        for idx, claim_text in enumerate(claim_texts[:max_claims]):
            cleaned = (claim_text or "").strip()
            if not cleaned:
                continue
            alignment_result = align_claim(
                source_text,
                cleaned,
                mode=alignment_mode,
                threshold=alignment_threshold,
            )
            record_claims_alignment_event(
                context=alignment_context,
                mode=alignment_mode,
                result=alignment_result,
            )
            claims.append(
                Claim(
                    id=f"c{idx+1}",
                    text=cleaned,
                    span=alignment_result.span if alignment_result is not None else None,
                )
            )
        return claims

    @staticmethod
    def _dedupe_claims(
        claims: list[Claim],
        *,
        max_claims: int,
    ) -> list[Claim]:
        """Deduplicate claims by normalized text and overlapping span.

        Args:
            claims: Ordered claims to deduplicate.
            max_claims: Maximum number of claims to keep in the output list.

        Returns:
            list[Claim]: Deduplicated claims with sequential ``cN`` IDs.

        Notes:
            If an existing duplicate has ``span=None`` and an incoming duplicate
            has a non-``None`` span, the incoming claim replaces the existing
            entry in place to preserve ordering while keeping richer span data.
        """
        deduped: list[Claim] = []
        for claim in claims:
            normalized = _normalize_claim_text(claim.text)
            replacement_index: int | None = None
            is_duplicate = False
            for idx, existing in enumerate(deduped):
                if _normalize_claim_text(existing.text) != normalized:
                    continue
                if _spans_overlap(existing.span, claim.span):
                    if existing.span is None and claim.span is not None:
                        replacement_index = idx
                    else:
                        is_duplicate = True
                    break
            if replacement_index is not None:
                deduped[replacement_index] = claim
                continue
            if is_duplicate:
                continue
            deduped.append(claim)
            if len(deduped) >= max_claims:
                break

        for idx, claim in enumerate(deduped, start=1):
            claim.id = f"c{idx}"
        return deduped

    async def _extract_aps_claim_texts(self, answer: str, max_claims: int) -> list[str]:
        try:
            from tldw_Server_API.app.core.Chunking.strategies.propositions import (
                PropositionChunkingStrategy,
            )
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"APS extractor unavailable: {exc}")
            return []

        provider, model_override, temperature = _resolve_claims_llm_config()
        try:
            strategy = PropositionChunkingStrategy(
                language="en",
                llm_call_func=self._analyze,
                llm_config={
                    "window_chars": 1200,
                    "api_name": provider or "openai",
                    "temp": temperature,
                    "model_override": model_override,
                },
            )
            prop_chunks = await _run_bounded_claims_analyze(
                strategy.chunk,
                text=answer,
                max_size=1,
                overlap=0,
                engine="llm",
                proposition_prompt_profile="gemma_aps",
            )
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"APS extractor failed: {exc}")
            return []

        out: list[str] = []
        for ptxt in (prop_chunks or [])[:max_claims]:
            if isinstance(ptxt, str) and ptxt.strip():
                out.append(ptxt.strip())
        return out

    async def _extract_claims_by_mode(
        self,
        *,
        answer: str,
        claim_extractor: str,
        claims_max: int,
        budget: ClaimsJobBudget | None,
        job_context: ClaimsJobContext | None,
    ) -> tuple[list[Claim], str]:
        async def _llm_strategy(text: str, max_items: int, _language: str | None) -> list[str]:
            claim_objs = await self.extractor_llm.extract(
                text,
                max_claims=max_items,
                budget=budget,
                job_context=job_context,
            )
            return [c.text for c in claim_objs if isinstance(c.text, str) and c.text.strip()]

        async def _aps_strategy(text: str, max_items: int, _language: str | None) -> list[str]:
            return await self._extract_aps_claim_texts(text, max_items)

        strategy_map = {
            "heuristic": extract_heuristic_claims_texts,
            "ner": extract_ner_claims_texts,
            "aps": _aps_strategy,
            "llm": _llm_strategy,
        }
        requested = (claim_extractor or "auto").strip().lower()
        fallback_mode = "llm" if requested in {"auto", "detect", "ner", "aps", "llm"} else "heuristic"
        context_window_chars = _resolve_claims_context_window_chars()
        extraction_passes = _resolve_claims_extraction_passes()
        llm_multi_pass_enabled = requested in {"auto", "detect", "aps", "llm"}
        run_passes = extraction_passes if llm_multi_pass_enabled else 1
        run_passes = max(1, run_passes)

        collected_claims: list[Claim] = []
        mode = requested
        for pass_index in range(run_passes):
            pass_input = answer
            if llm_multi_pass_enabled and context_window_chars > 0 and pass_index > 0:
                pass_input = f"{answer[-context_window_chars:]}\n\n{answer}"

            dispatch = await run_async_claims_strategy(
                requested_mode=requested,
                text=pass_input,
                max_claims=claims_max,
                strategy_map=strategy_map,
                fallback_mode=fallback_mode,
                catch_exceptions=_CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS,
            )
            mode = dispatch.mode
            pass_claims = self._build_claims_from_texts(
                source_text=answer,
                claim_texts=dispatch.claim_texts,
                max_claims=claims_max,
                alignment_context="engine_extract",
            )
            if pass_claims:
                collected_claims.extend(pass_claims)

        if run_passes > 1 or context_window_chars > 0:
            claims = self._dedupe_claims(collected_claims, max_claims=claims_max)
        else:
            claims = collected_claims[:claims_max]
        if claims:
            return claims, mode

        heur_claims = await self.extractor_heur.extract(answer, claims_max)
        claims = self._build_claims_from_texts(
            source_text=answer,
            claim_texts=[c.text for c in heur_claims],
            max_claims=claims_max,
            alignment_context="engine_extract_fallback",
        )
        return claims, "heuristic"

    async def run(
        self,
        answer: str,
        query: str,
        documents: list[Document],
        claim_extractor: str = "auto",
        claim_verifier: str = "hybrid",
        claims_top_k: int = 5,
        claims_conf_threshold: float = 0.7,
        claims_max: int = 25,
        retrieve_fn: Any | None = None,
        nli_model: str | None = None,
        claims_concurrency: int = 8,
        job_budget: ClaimsJobBudget | None = None,
        job_context: ClaimsJobContext | None = None,
    ) -> dict[str, Any]:
        if not answer or not isinstance(answer, str):
            return {"claims": [], "summary": {}}

        budget = job_budget
        if budget is None:
            try:
                from tldw_Server_API.app.core.config import settings as _settings  # type: ignore
            except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
                _settings = {}
            budget = resolve_claims_job_budget(settings=_settings)

        claims, extractor_mode = await self._extract_claims_by_mode(
            answer=answer,
            claim_extractor=claim_extractor,
            claims_max=claims_max,
            budget=budget,
            job_context=job_context,
        )
        verifications: list[ClaimVerification] = []

        # Initialize verifier once if a specific NLI model is requested
        if nli_model:
            self.verifier = HybridClaimVerifier(self._analyze, nli_model=nli_model)

        async def _verify_one(c: Claim) -> ClaimVerification:
            return await self.verifier.verify(
                claim=c,
                query=query,
                base_documents=documents,
                retrieve_fn=retrieve_fn,
                top_k=claims_top_k,
                conf_threshold=claims_conf_threshold,
                mode=(claim_verifier or "hybrid").strip().lower(),
                budget=budget,
                job_context=job_context,
            )

        # Concurrency cap to avoid over-parallelization of verifications
        try:
            max_conc = int(claims_concurrency)
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
            max_conc = 8
        max_conc = max(1, min(32, max_conc))
        try:
            provider, model_override, _ = _resolve_claims_llm_config()
            budget_ratio = budget.remaining_ratio() if budget is not None else None
            max_conc = suggest_claims_concurrency(
                provider=provider,
                model=model_override or "",
                requested=max_conc,
                budget_ratio=budget_ratio,
            )
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
            pass
        sem = asyncio.Semaphore(max_conc)

        async def _bounded_verify(c: Claim) -> ClaimVerification:
            async with sem:
                return await _verify_one(c)

        tasks = [_bounded_verify(c) for c in claims]
        if tasks:
            verifications = await asyncio.gather(*tasks)

        supported = sum(1 for v in verifications if v.label == "supported")
        refuted = sum(1 for v in verifications if v.label == "refuted")
        nei = sum(1 for v in verifications if v.label == "nei")
        total = max(1, len(verifications))
        precision = supported / total
        coverage = (supported + refuted) / total

        # Count by new status values
        verified_count = sum(1 for v in verifications if v.status == VerificationStatus.VERIFIED)
        refuted_count = sum(1 for v in verifications if v.status == VerificationStatus.REFUTED)
        hallucination_count = sum(1 for v in verifications if v.status == VerificationStatus.HALLUCINATION)
        numerical_error_count = sum(1 for v in verifications if v.status == VerificationStatus.NUMERICAL_ERROR)
        misquoted_count = sum(1 for v in verifications if v.status == VerificationStatus.MISQUOTED)
        unverified_count = sum(1 for v in verifications if v.status == VerificationStatus.UNVERIFIED)

        claims_out: list[dict[str, Any]] = []
        for v in verifications:
            claims_out.append(
                {
                    "id": v.claim.id,
                    "text": v.claim.text,
                    "span": list(v.claim.span) if v.claim.span else None,
                    "claim_type": v.claim.claim_type.value,
                    "status": v.status.value,
                    "label": v.label,  # Backward compatibility
                    "confidence": v.confidence,
                    "match_level": v.match_level.value,
                    "source_authority": v.source_authority.value,
                    "requires_external_knowledge": v.requires_external_knowledge,
                    "evidence": [
                        {
                            "doc_id": e.doc_id,
                            "snippet": e.snippet,
                            "score": e.score,
                            "authority": e.authority.value,
                        } for e in v.evidence
                    ],
                    "citations": v.citations,
                    "rationale": v.rationale,
                }
            )

        return {
            "claims": claims_out,
            "summary": {
                "supported": supported,
                "refuted": refuted,
                "nei": nei,
                "precision": precision,
                "coverage": coverage,
                "claim_faithfulness": (supported / total) if total else 0.0,
                # Enhanced summary with new status breakdown
                "verified": verified_count,
                "refuted_status": refuted_count,
                "hallucination": hallucination_count,
                "numerical_error": numerical_error_count,
                "misquoted": misquoted_count,
                "unverified": unverified_count,
                "verification_rate": verified_count / total if total else 0.0,
                "budget": (budget.snapshot() if budget is not None else None),
            },
            # Raw ClaimVerification objects for full report generation
            "verifications": verifications,
        }

    async def extract_claims_only(
        self,
        answer: str,
        claim_extractor: str = "auto",
        claims_max: int = 25,
        job_budget: ClaimsJobBudget | None = None,
        job_context: ClaimsJobContext | None = None,
    ) -> ExtractionResult:
        """
        Extract claims from text without verification (first pass of two-pass architecture).

        Args:
            answer: Text to extract claims from
            claim_extractor: Extraction strategy ("auto", "ner", "aps", "llm")
            claims_max: Maximum claims to extract
            job_budget: Optional budget constraints
            job_context: Optional job context for logging

        Returns:
            ExtractionResult with claims and extraction metadata
        """
        start_time = time.time()

        if not answer or not isinstance(answer, str):
            return ExtractionResult(
                claims=[],
                extractor_mode="none",
                extraction_time_s=0.0,
                total_input_chars=0,
            )

        budget = job_budget
        if budget is None:
            try:
                from tldw_Server_API.app.core.config import settings as _settings
            except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
                _settings = {}
            budget = resolve_claims_job_budget(settings=_settings)

        claims, extractor_mode = await self._extract_claims_by_mode(
            answer=answer,
            claim_extractor=claim_extractor,
            claims_max=claims_max,
            budget=budget,
            job_context=job_context,
        )

        # Classify all claims
        for claim in claims:
            if claim.claim_type == ClaimType.GENERAL:
                claim.claim_type, claim.extracted_values = classify_claim_type(claim.text)

        return ExtractionResult(
            claims=claims,
            extractor_mode=extractor_mode,
            extraction_time_s=time.time() - start_time,
            total_input_chars=len(answer),
        )

    async def verify_claims_only(
        self,
        claims: list[Claim],
        query: str,
        documents: list[Document],
        claim_verifier: str = "hybrid",
        claims_top_k: int = 5,
        claims_conf_threshold: float = 0.7,
        claims_concurrency: int = 8,
        retrieve_fn: Any | None = None,
        nli_model: str | None = None,
        job_budget: ClaimsJobBudget | None = None,
        job_context: ClaimsJobContext | None = None,
        doc_only_mode: bool = False,
        numeric_precision_mode: str = "standard",
    ) -> VerificationResult:
        """
        Verify pre-extracted claims against documents (second pass of two-pass architecture).

        Args:
            claims: List of claims to verify
            query: Original query context
            documents: Source documents for verification
            claim_verifier: Verification strategy ("hybrid", "nli", "llm")
            claims_top_k: Top-K evidence per claim
            claims_conf_threshold: Confidence threshold
            claims_concurrency: Max parallel verifications
            retrieve_fn: Optional retrieval function
            nli_model: Optional NLI model override
            job_budget: Budget constraints
            job_context: Job context for logging
            doc_only_mode: Only accept evidence from provided documents
            numeric_precision_mode: "standard", "strict", or "academic"

        Returns:
            VerificationResult with verifications and summary
        """
        start_time = time.time()

        if not claims:
            return VerificationResult(
                verifications=[],
                verification_time_s=0.0,
                summary={"total": 0, "verified": 0},
            )

        budget = job_budget
        if budget is None:
            try:
                from tldw_Server_API.app.core.config import settings as _settings
            except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
                _settings = {}
            budget = resolve_claims_job_budget(settings=_settings)

        if nli_model:
            self.verifier = HybridClaimVerifier(self._analyze, nli_model=nli_model)

        async def _verify_one(c: Claim) -> ClaimVerification:
            return await self.verifier.verify(
                claim=c,
                query=query,
                base_documents=documents,
                retrieve_fn=retrieve_fn,
                top_k=claims_top_k,
                conf_threshold=claims_conf_threshold,
                mode=(claim_verifier or "hybrid").strip().lower(),
                budget=budget,
                job_context=job_context,
                doc_only_mode=doc_only_mode,
                numeric_precision_mode=numeric_precision_mode,
            )

        max_conc = max(1, min(32, claims_concurrency))
        try:
            provider, model_override, _ = _resolve_claims_llm_config()
            budget_ratio = budget.remaining_ratio() if budget is not None else None
            max_conc = suggest_claims_concurrency(provider=provider, model=model_override or "", requested=max_conc, budget_ratio=budget_ratio)
        except _CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS:
            pass

        sem = asyncio.Semaphore(max_conc)

        async def _bounded_verify(c: Claim) -> ClaimVerification:
            async with sem:
                return await _verify_one(c)

        verifications = await asyncio.gather(*[_bounded_verify(c) for c in claims])

        # Build summary
        total = len(verifications)
        verified = sum(1 for v in verifications if v.status == VerificationStatus.VERIFIED)
        refuted = sum(1 for v in verifications if v.status == VerificationStatus.REFUTED)
        hallucination = sum(1 for v in verifications if v.status == VerificationStatus.HALLUCINATION)

        summary = {
            "total": total,
            "verified": verified,
            "refuted": refuted,
            "hallucination": hallucination,
            "verification_rate": verified / total if total else 0.0,
            "supported": sum(1 for v in verifications if v.label == "supported"),
            "nei": sum(1 for v in verifications if v.label == "nei"),
        }

        return VerificationResult(
            verifications=list(verifications),
            verification_time_s=time.time() - start_time,
            summary=summary,
        )
