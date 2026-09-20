"""Deterministic bounded lexical retrieval for Notes graph suggestions."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .suggestion_content import (
    MAX_CANDIDATE_WINDOWS,
    MAX_SOURCE_WINDOWS,
    MAX_WINDOW_CODE_POINTS,
    EvidenceReference,
    content_fingerprint,
    split_evidence_windows,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import (
        NoteGraphSuggestionStore,
        SuggestionNoteRecord,
    )


MAX_RETRIEVAL_TERMS = 24
RETRIEVAL_OVERFETCH = 60
MAX_CANDIDATES = 30
MAX_TAG_CATALOG = 100

_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
        "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
    }
)
_TERM_PATTERN = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    """One eligible candidate in backend rank order without exposing backend scores."""

    note_id: str
    title: str
    content: str
    fingerprint: str
    evidence_windows: tuple[EvidenceReference, ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Bounded candidate data passed to later provider-contract work."""

    source_note_id: str
    source_fingerprint: str
    source_windows: tuple[EvidenceReference, ...]
    terms: tuple[str, ...]
    candidates: tuple[RetrievedCandidate, ...]
    tag_catalog: tuple[str, ...]
    backend_overfetch_count: int
    excluded_oversized_candidate_count: int
    projection_fresh: bool
    estimated_input_tokens: int


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TERM_PATTERN.finditer(value))


def derive_retrieval_terms(title: str, content: str) -> tuple[str, ...]:
    """Select at most 24 stable lexical FTS terms, prioritizing title terms."""

    title_terms = tuple(term for term in _tokens(title) if term not in _STOP_WORDS)
    body_terms = tuple(term for term in _tokens(content) if term not in _STOP_WORDS)
    ordered_title = sorted(set(title_terms))
    body_frequency = Counter(body_terms)
    ordered_body = sorted(body_frequency, key=lambda term: (-body_frequency[term], term))
    return tuple(dict.fromkeys((*ordered_title, *ordered_body)))[:MAX_RETRIEVAL_TERMS]


class SuggestionRetriever:
    """Compose canonical content with bounded, owner-scoped backend reads."""

    def __init__(self, store: NoteGraphSuggestionStore) -> None:
        self._store = store

    def retrieve(self, *, dataset_id: str, source_note_id: str) -> RetrievalResult:
        """Load a bounded FTS shortlist without logging note-derived values."""

        source = self._store.load_source_note(dataset_id=dataset_id, note_id=source_note_id)
        source_fingerprint = content_fingerprint(source.title, source.content)
        terms = derive_retrieval_terms(source.title, source.content)
        candidate_rows, oversized_count, ranked_shortlist_count = self._store.fetch_ranked_candidates(
            dataset_id=dataset_id,
            source_note_id=source.note_id,
            terms=terms,
            source_fingerprint=source_fingerprint,
            limit=RETRIEVAL_OVERFETCH,
        )
        suppressed = self._store.list_rejected_candidate_fingerprints(
            dataset_id=dataset_id,
            source_note_id=source.note_id,
            source_fingerprint=source_fingerprint,
        )
        candidates = self._eligible_candidates(candidate_rows, suppressed)[:MAX_CANDIDATES]
        source_windows = split_evidence_windows(
            note_id=source.note_id,
            title=source.title,
            content=source.content,
            max_windows=MAX_SOURCE_WINDOWS,
            max_code_points=MAX_WINDOW_CODE_POINTS,
        )
        tag_catalog = self._store.list_tag_catalog(dataset_id=dataset_id, terms=terms, limit=MAX_TAG_CATALOG)
        projection_fresh = self._store.is_projection_fresh(dataset_id=dataset_id, note_id=source.note_id)
        evidence_code_points = sum(
            window.end_offset - window.start_offset for window in source_windows
        ) + sum(
            window.end_offset - window.start_offset
            for candidate in candidates
            for window in candidate.evidence_windows
        )
        return RetrievalResult(
            source_note_id=source.note_id,
            source_fingerprint=source_fingerprint,
            source_windows=source_windows,
            terms=terms,
            candidates=tuple(candidates),
            tag_catalog=tag_catalog,
            backend_overfetch_count=ranked_shortlist_count,
            excluded_oversized_candidate_count=oversized_count,
            projection_fresh=projection_fresh,
            estimated_input_tokens=(evidence_code_points + 3) // 4,
        )

    @staticmethod
    def _eligible_candidates(
        rows: tuple[SuggestionNoteRecord, ...],
        suppressed: frozenset[tuple[str, str]],
    ) -> list[RetrievedCandidate]:
        candidates: list[RetrievedCandidate] = []
        for row in rows:
            fingerprint = content_fingerprint(row.title, row.content)
            if (row.note_id, fingerprint) in suppressed:
                continue
            candidates.append(
                RetrievedCandidate(
                    note_id=row.note_id,
                    title=row.title,
                    content=row.content,
                    fingerprint=fingerprint,
                    evidence_windows=split_evidence_windows(
                        note_id=row.note_id,
                        title=row.title,
                        content=row.content,
                        max_windows=MAX_CANDIDATE_WINDOWS,
                        max_code_points=MAX_WINDOW_CODE_POINTS,
                    ),
                )
            )
        return candidates
