"""Deterministic Auto Chunking planner for ingestion request defaults."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

AutoChunkingGoal = Literal["balanced", "qa_search", "navigation_summary"]


@dataclass(frozen=True)
class AutoChunkingProfile:
    """Signals the planner uses to choose a chunking strategy."""

    media_type: str | None = None
    source_name: str | None = None
    title: str | None = None
    mime_type: str | None = None
    extension: str | None = None
    text_length: int = 0
    has_headings: bool = False
    has_tables: bool = False
    has_lists: bool = False
    has_timecodes: bool = False
    has_speaker_labels: bool = False
    has_chapters: bool = False
    language: str | None = None


@dataclass(frozen=True)
class AutoChunkingRequest:
    """Normalized inputs for a single Auto Chunking planning decision."""

    perform_chunking: bool
    chunking_mode: str | None
    media_type: str | None
    goal: str = "balanced"
    profile: AutoChunkingProfile | None = None
    template_name: str | None = None
    template_status: str | None = None
    template_error: str | None = None
    requested_llm: bool = False
    llm_available: bool = False
    semantic_available: bool = True


@dataclass(frozen=True)
class AutoChunkingPlan:
    """Serializable explanation of the selected Auto Chunking behavior."""

    mode: Literal["auto"]
    goal: AutoChunkingGoal
    used_llm: bool
    method: str
    max_size: int
    overlap: int
    template_name: str | None
    derived_views: tuple[str, ...]
    fallback_reason: str | None
    rationale: str
    profile: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-safe metadata representation of the plan."""
        return {
            "mode": self.mode,
            "goal": self.goal,
            "used_llm": self.used_llm,
            "method": self.method,
            "max_size": self.max_size,
            "overlap": self.overlap,
            "template_name": self.template_name,
            "derived_views": list(self.derived_views),
            "fallback_reason": self.fallback_reason,
            "rationale": self.rationale,
            "profile": self.profile,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "AutoChunkingPlan":
        """Rebuild a plan from previously stored metadata."""
        return cls(
            mode="auto",
            goal=_normalize_goal(metadata.get("goal")),
            used_llm=bool(metadata.get("used_llm")),
            method=str(metadata.get("method") or "sentences"),
            max_size=int(metadata.get("max_size") or _GOAL_SIZES["balanced"][0]),
            overlap=int(metadata.get("overlap") or _GOAL_SIZES["balanced"][1]),
            template_name=metadata.get("template_name"),
            derived_views=tuple(str(view) for view in metadata.get("derived_views") or []),
            fallback_reason=metadata.get("fallback_reason"),
            rationale=str(metadata.get("rationale") or ""),
            profile=dict(metadata.get("profile") or {}),
        )


@dataclass(frozen=True)
class AutoChunkingDecision:
    """Resolved chunking options plus optional metadata plan."""

    chunk_options: dict[str, Any] | None
    chunking_plan: dict[str, Any] | None


_GOAL_SIZES: dict[str, tuple[int, int]] = {
    "qa_search": (700, 140),
    "balanced": (900, 120),
    "navigation_summary": (1400, 100),
}

_TEXT_SCAN_CHARS = 200_000
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}\s+\S+|chapter\s+\d+|section\s+\d+)", re.IGNORECASE | re.MULTILINE)
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+\S+|\d+[.)]\s+\S+)", re.MULTILINE)
_TABLE_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_TIMECODE_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\b")
_SPEAKER_RE = re.compile(
    r"^\s*(?:speaker\s*\d+|speaker|host|guest|interviewer|interviewee)\s*:", re.IGNORECASE | re.MULTILINE
)
_CHAPTER_RE = re.compile(r"^\s*(?:chapter|book|part)\s+[\w.-]+", re.IGNORECASE | re.MULTILINE)


def profile_from_source(
    *,
    media_type: str | None = None,
    filename: str | None = None,
    url: str | None = None,
    title: str | None = None,
    mime_type: str | None = None,
    language: str | None = None,
) -> AutoChunkingProfile:
    """Build planner signals from request source metadata."""
    source_name = filename or url
    extension = None
    if filename and "." in filename:
        extension = filename.rsplit(".", 1)[-1].lower()
    return AutoChunkingProfile(
        media_type=_normalize_media_type(media_type),
        source_name=source_name,
        title=title,
        mime_type=mime_type,
        extension=extension,
        language=language,
        has_chapters=bool(extension in {"epub", "mobi", "azw3"}),
    )


def profile_from_text(text: str | None, *, max_scan_chars: int = _TEXT_SCAN_CHARS) -> AutoChunkingProfile:
    """Build planner signals by scanning extracted text."""
    sample = (text or "")[: max(0, int(max_scan_chars))]
    return AutoChunkingProfile(
        text_length=len(text or ""),
        has_headings=bool(_HEADING_RE.search(sample)),
        has_tables=bool(_TABLE_RE.search(sample)),
        has_lists=bool(_LIST_RE.search(sample)),
        has_timecodes=bool(_TIMECODE_RE.search(sample)),
        has_speaker_labels=bool(_SPEAKER_RE.search(sample)),
        has_chapters=bool(_CHAPTER_RE.search(sample)),
    )


def merge_profiles(*profiles: AutoChunkingProfile | None) -> AutoChunkingProfile:
    """Merge source and content profiles without losing positive signals."""
    merged = AutoChunkingProfile()
    for profile in profiles:
        if profile is None:
            continue
        merged = AutoChunkingProfile(
            media_type=merged.media_type or profile.media_type,
            source_name=merged.source_name or profile.source_name,
            title=merged.title or profile.title,
            mime_type=merged.mime_type or profile.mime_type,
            extension=merged.extension or profile.extension,
            text_length=max(merged.text_length, profile.text_length),
            has_headings=merged.has_headings or profile.has_headings,
            has_tables=merged.has_tables or profile.has_tables,
            has_lists=merged.has_lists or profile.has_lists,
            has_timecodes=merged.has_timecodes or profile.has_timecodes,
            has_speaker_labels=merged.has_speaker_labels or profile.has_speaker_labels,
            has_chapters=merged.has_chapters or profile.has_chapters,
            language=merged.language or profile.language,
        )
    return merged


def plan_auto_chunking(
    *,
    perform_chunking: bool,
    chunking_mode: str | None,
    media_type: str | None,
    goal: str = "balanced",
    profile: AutoChunkingProfile | None = None,
    template_name: str | None = None,
    template_status: str | None = None,
    template_error: str | None = None,
    requested_llm: bool = False,
    llm_available: bool = False,
    semantic_available: bool = True,
) -> AutoChunkingDecision:
    """Choose effective chunking options and metadata for an Auto request."""
    if not perform_chunking or chunking_mode != "auto":
        return AutoChunkingDecision(chunk_options=None, chunking_plan=None)

    normalized_goal = _normalize_goal(goal)
    base_profile = merge_profiles(
        AutoChunkingProfile(media_type=_normalize_media_type(media_type)),
        profile,
    )
    method, derived_views, rationale_bits, fallback_reasons = _choose_method(
        base_profile,
        normalized_goal,
        semantic_available=semantic_available,
    )
    fallback_reasons.extend(_template_fallback_reasons(template_status, template_error, rationale_bits))

    # Adapter availability is an explicit caller contract. Do not infer it from
    # configured chat providers until a boundary assistant adapter owns that check.
    used_llm = bool(requested_llm and llm_available)
    if requested_llm and not llm_available:
        fallback_reasons.append("ai_assist_unavailable")
        rationale_bits.append("AI assist was requested but no boundary adapter is available.")

    max_size, overlap = _size_for_goal(normalized_goal, base_profile)
    if base_profile.media_type == "email":
        max_size, overlap = 1000, 150

    chunk_options = {
        "method": method,
        "max_size": max_size,
        "overlap": overlap,
        "adaptive": False,
        "multi_level": False,
        "language": base_profile.language,
    }
    plan = AutoChunkingPlan(
        mode="auto",
        goal=normalized_goal,
        used_llm=used_llm,
        method=method,
        max_size=max_size,
        overlap=overlap,
        template_name=template_name,
        derived_views=tuple(derived_views),
        fallback_reason=";".join(fallback_reasons) if fallback_reasons else None,
        rationale=" ".join(rationale_bits),
        profile=asdict(base_profile),
    )
    return AutoChunkingDecision(
        chunk_options=chunk_options,
        chunking_plan=plan.to_metadata(),
    )


def plan_auto_chunking_request(request: AutoChunkingRequest) -> AutoChunkingDecision:
    """Plan Auto Chunking from a normalized request object."""
    return plan_auto_chunking(
        perform_chunking=request.perform_chunking,
        chunking_mode=request.chunking_mode,
        media_type=request.media_type,
        goal=request.goal,
        profile=request.profile,
        template_name=request.template_name,
        template_status=request.template_status,
        template_error=request.template_error,
        requested_llm=request.requested_llm,
        llm_available=request.llm_available,
        semantic_available=request.semantic_available,
    )


def _normalize_goal(goal: str | None) -> AutoChunkingGoal:
    if goal in _GOAL_SIZES:
        return goal  # type: ignore[return-value]
    return "balanced"


def _normalize_media_type(media_type: str | None) -> str | None:
    normalized = (media_type or "").strip().lower()
    if normalized in {"web_document", "webpage", "article", "html"}:
        return "web"
    return normalized or None


def _size_for_goal(goal: str, profile: AutoChunkingProfile) -> tuple[int, int]:
    max_size, overlap = _GOAL_SIZES[goal]
    if profile.text_length > 60_000 and goal == "navigation_summary":
        return 1800, overlap
    return max_size, overlap


def _choose_method(
    profile: AutoChunkingProfile,
    goal: str,
    *,
    semantic_available: bool,
) -> tuple[str, list[str], list[str], list[str]]:
    media_type = profile.media_type or ""
    derived_views: list[str] = []
    rationale_bits: list[str] = []
    fallback_reasons: list[str] = []

    if media_type == "ebook":
        derived_views.append("chapter_outline")
        rationale_bits.append("Detected ebook content; selected chapter-based chunking.")
        return "ebook_chapters", derived_views, rationale_bits, fallback_reasons

    if media_type == "email":
        derived_views.append("message_boundaries")
        rationale_bits.append("Detected email content; selected message-friendly sentence chunks.")
        return "sentences", derived_views, rationale_bits, fallback_reasons

    if media_type in {"audio", "video"}:
        if profile.has_timecodes:
            derived_views.append("time_ranges")
        if profile.has_speaker_labels:
            derived_views.append("speaker_segments")
        rationale_bits.append("Detected transcript-like media; selected sentence chunks.")
        return "sentences", derived_views, rationale_bits, fallback_reasons

    if media_type in {"document", "pdf", "web"}:
        if profile.has_headings or profile.has_tables or profile.has_lists:
            derived_views.extend(_structure_views(profile, goal))
            rationale_bits.append("Detected document structure; selected structure-aware chunking.")
            return "structure_aware", derived_views, rationale_bits, fallback_reasons
        if semantic_available:
            rationale_bits.append("Detected unstructured long text; selected semantic chunking.")
            return "semantic", derived_views, rationale_bits, fallback_reasons
        fallback_reasons.append("semantic_unavailable")
        rationale_bits.append("Semantic chunking is unavailable; selected sentence fallback.")
        return "sentences", derived_views, rationale_bits, fallback_reasons

    rationale_bits.append("No specialized media signals detected; selected sentence chunks.")
    return "sentences", derived_views, rationale_bits, fallback_reasons


def _structure_views(profile: AutoChunkingProfile, goal: str) -> list[str]:
    views = ["section_titles"]
    if goal == "navigation_summary" or profile.has_headings:
        views.append("outline")
    if profile.has_tables:
        views.append("table_regions")
    if profile.has_lists:
        views.append("list_blocks")
    return views


def _template_fallback_reasons(
    template_status: str | None,
    template_error: str | None,
    rationale_bits: list[str],
) -> list[str]:
    if template_status == "matched":
        rationale_bits.append("Applied matched chunking template as a planner input.")
        return []
    if template_status == "no_match":
        rationale_bits.append("No chunking template matched; used deterministic media rules.")
        return ["template_no_match"]
    if template_status == "error":
        message = template_error or "template classifier error"
        rationale_bits.append(f"Template classifier failed ({message}); used deterministic media rules.")
        return ["template_error"]
    return []
