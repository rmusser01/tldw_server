"""Shared helpers for synthetic evaluation draft generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from tldw_Server_API.app.core.Evaluations.synthetic_eval_schemas import (
    SyntheticEvalProvenance,
)

_DEFAULT_SOURCES: tuple[str, ...] = ("media_db", "notes")
_DEFAULT_QUERY_INTENTS: tuple[str, ...] = (
    "lookup",
    "synthesis",
    "comparison",
    "ambiguous / underspecified",
)
_DEFAULT_DIFFICULTIES: tuple[str, ...] = (
    "straightforward",
    "distractor-heavy",
    "multi-source",
    "abstention-worthy",
)


@dataclass(frozen=True, slots=True)
class SyntheticEvalCorpusScope:
    """Normalized corpus scope for synthetic generation."""

    sources: tuple[str, ...] = _DEFAULT_SOURCES
    recipe_kind: str | None = None
    corpus_name: str | None = None
    media_ids: tuple[int, ...] = ()
    note_ids: tuple[str, ...] = ()
    indexing_fixed: bool | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return a JSON-safe corpus scope representation."""

        return {
            "sources": list(self.sources),
            "recipe_kind": self.recipe_kind,
            "corpus_name": self.corpus_name,
            "media_ids": list(self.media_ids),
            "note_ids": list(self.note_ids),
            "indexing_fixed": self.indexing_fixed,
        }


@dataclass(frozen=True, slots=True)
class SyntheticEvalStructuredTuple:
    """Structured tuple emitted before natural-language sample conversion."""

    source_kind: str
    query_intent: str
    difficulty: str
    query: str
    failure_mode: str
    target_payload: dict[str, Any] = field(default_factory=dict)
    provenance_hint: str = SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value
    corpus_notes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SyntheticEvalCoverageSummary:
    """Coverage report for the current draft corpus."""

    sources: list[str] = field(default_factory=list)
    query_intents: list[str] = field(default_factory=list)
    difficulties: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SyntheticEvalGenerationPlan:
    """Internal helper returned by the generation helpers."""

    corpus_scope: SyntheticEvalCorpusScope
    examples: list[dict[str, Any]] = field(default_factory=list)
    coverage: SyntheticEvalCoverageSummary = field(default_factory=SyntheticEvalCoverageSummary)
    missing_coverage: dict[str, list[str]] = field(default_factory=dict)


def resolve_corpus_scope(
    corpus_scope: SyntheticEvalCorpusScope | Mapping[str, Any] | Sequence[str] | None,
) -> SyntheticEvalCorpusScope:
    """Normalize a corpus scope into a stable tuple-based scope."""

    if corpus_scope is None:
        return SyntheticEvalCorpusScope()
    if isinstance(corpus_scope, SyntheticEvalCorpusScope):
        return SyntheticEvalCorpusScope(
            sources=_dedupe_tuple(corpus_scope.sources) or _DEFAULT_SOURCES,
            recipe_kind=corpus_scope.recipe_kind,
            corpus_name=corpus_scope.corpus_name,
            media_ids=_dedupe_int_tuple(corpus_scope.media_ids),
            note_ids=_dedupe_tuple(corpus_scope.note_ids),
            indexing_fixed=corpus_scope.indexing_fixed,
        )
    if isinstance(corpus_scope, Mapping):
        sources = corpus_scope.get("sources")
        if isinstance(sources, str):
            normalized_sources = _dedupe_tuple((sources.strip(),))
        elif isinstance(sources, Sequence):
            normalized_sources = _dedupe_tuple(
                tuple(str(source).strip() for source in sources if str(source).strip())
            )
        else:
            normalized_sources = _DEFAULT_SOURCES
        return SyntheticEvalCorpusScope(
            sources=normalized_sources or _DEFAULT_SOURCES,
            recipe_kind=(
                str(corpus_scope.get("recipe_kind")).strip()
                if corpus_scope.get("recipe_kind") is not None
                else None
            ),
            corpus_name=(
                str(corpus_scope.get("corpus_name")).strip()
                if corpus_scope.get("corpus_name") is not None
                else None
            ),
            media_ids=_normalize_int_tuple(corpus_scope.get("media_ids")),
            note_ids=_normalize_note_id_tuple(corpus_scope.get("note_ids")),
            indexing_fixed=_coerce_optional_bool(corpus_scope.get("indexing_fixed")),
        )
    normalized_sources = _dedupe_tuple(tuple(str(source).strip() for source in corpus_scope if str(source).strip()))
    return SyntheticEvalCorpusScope(sources=normalized_sources or _DEFAULT_SOURCES)


def ingest_examples(
    *,
    real_examples: Sequence[dict[str, Any]] | None = None,
    seed_examples: Sequence[dict[str, Any]] | None = None,
    corpus_scope: SyntheticEvalCorpusScope | Sequence[str] | None = None,
) -> SyntheticEvalGenerationPlan:
    """Normalize corpus examples and derive baseline coverage."""

    resolved_scope = resolve_corpus_scope(corpus_scope)
    examples: list[dict[str, Any]] = []
    for example in real_examples or []:
        examples.append(_normalize_existing_example(example, "real"))
    for example in seed_examples or []:
        examples.append(_normalize_existing_example(example, "seed_examples"))
    scoped_examples = [example for example in examples if example["source_kind"] in resolved_scope.sources]
    coverage = _summarize_coverage(scoped_examples)
    missing = identify_missing_coverage(coverage=coverage, corpus_scope=resolved_scope)
    return SyntheticEvalGenerationPlan(
        corpus_scope=resolved_scope,
        examples=scoped_examples,
        coverage=coverage,
        missing_coverage=missing,
    )


def identify_missing_coverage(
    *,
    coverage: SyntheticEvalCoverageSummary,
    corpus_scope: SyntheticEvalCorpusScope,
) -> dict[str, list[str]]:
    """Compute the uncovered stratification buckets for the current corpus."""

    covered_sources = set(coverage.sources)
    covered_intents = set(coverage.query_intents)
    covered_difficulties = set(coverage.difficulties)

    missing_sources = [source for source in corpus_scope.sources if source not in covered_sources]
    missing_intents = [intent for intent in _DEFAULT_QUERY_INTENTS if intent not in covered_intents]
    missing_difficulties = [difficulty for difficulty in _DEFAULT_DIFFICULTIES if difficulty not in covered_difficulties]

    return {
        "sources": missing_sources,
        "query_intents": missing_intents,
        "difficulties": missing_difficulties,
    }


def generate_structured_tuples(
    *,
    coverage_gaps: dict[str, list[str]],
    corpus_scope: SyntheticEvalCorpusScope,
    corpus_examples: Sequence[dict[str, Any]] | None = None,
    seed_examples: Sequence[dict[str, Any]] | None = None,
    target_count: int = 0,
) -> list[SyntheticEvalStructuredTuple]:
    """Generate corpus-grounded structured tuples for uncovered coverage gaps."""

    needed_count = max(0, int(target_count))
    if needed_count <= 0:
        return []

    sources = coverage_gaps.get("sources") or list(corpus_scope.sources)
    query_intents = coverage_gaps.get("query_intents") or list(_DEFAULT_QUERY_INTENTS)
    difficulties = coverage_gaps.get("difficulties") or list(_DEFAULT_DIFFICULTIES)
    topic_hint = _derive_topic_hint(corpus_examples or [], seed_examples or [])

    tuples: list[SyntheticEvalStructuredTuple] = []
    for index in range(needed_count):
        source_kind = sources[index % len(sources)]
        query_intent = query_intents[index % len(query_intents)]
        difficulty = difficulties[index % len(difficulties)]
        failure_mode = _infer_failure_mode(query_intent=query_intent, difficulty=difficulty)
        query = _build_failure_oriented_query(
            source_kind=source_kind,
            query_intent=query_intent,
            difficulty=difficulty,
            topic_hint=topic_hint,
            index=index,
        )
        target_payload = _build_target_payload(
            source_kind=source_kind,
            query_intent=query_intent,
            difficulty=difficulty,
            failure_mode=failure_mode,
            topic_hint=topic_hint,
        )
        tuples.append(
            SyntheticEvalStructuredTuple(
                source_kind=source_kind,
                query_intent=query_intent,
                difficulty=difficulty,
                query=query,
                failure_mode=failure_mode,
                target_payload=target_payload,
            )
        )
    return tuples


def convert_structured_tuples_to_draft_samples(
    *,
    recipe_kind: str,
    structured_tuples: Sequence[SyntheticEvalStructuredTuple],
    corpus_scope: SyntheticEvalCorpusScope | Mapping[str, Any] | Sequence[str] | None = None,
    generation_batch_id: str | None = None,
    generation_metadata: dict[str, Any] | None = None,
    sample_payload_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert structured tuples into repository-ready draft sample payloads."""

    resolved_scope = resolve_corpus_scope(corpus_scope)
    resolved_generation_metadata = dict(generation_metadata or {})
    resolved_payload_overrides = {
        key: value for key, value in (sample_payload_overrides or {}).items() if value is not None
    }
    draft_samples: list[dict[str, Any]] = []
    for index, structured_tuple in enumerate(structured_tuples):
        sample_payload = {
            "query": structured_tuple.query,
            "source_kind": structured_tuple.source_kind,
            "query_intent": structured_tuple.query_intent,
            "difficulty": structured_tuple.difficulty,
            "failure_mode": structured_tuple.failure_mode,
        }
        sample_payload.update(structured_tuple.target_payload)
        sample_payload.update(resolved_payload_overrides)
        draft_samples.append(
            {
                "sample_id": f"{recipe_kind}-synthetic-{index + 1}",
                "recipe_kind": recipe_kind,
                "sample_payload": sample_payload,
                "sample_metadata": {
                    "generation_stage": "structured_tuple_conversion",
                    "generation_batch_id": generation_batch_id,
                    "query_intent": structured_tuple.query_intent,
                    "difficulty": structured_tuple.difficulty,
                    "failure_mode": structured_tuple.failure_mode,
                    "source_kind": structured_tuple.source_kind,
                    "corpus_scope": resolved_scope.to_metadata_dict(),
                    "generation_metadata": resolved_generation_metadata,
                },
                "provenance": structured_tuple.provenance_hint,
                "review_state": "draft",
                "source_kind": structured_tuple.source_kind,
            }
        )
    return draft_samples


def summarize_coverage(examples: Sequence[dict[str, Any]]) -> SyntheticEvalCoverageSummary:
    """Summarize the covered sources, intents, and difficulty buckets."""

    return _summarize_coverage(examples)


def _normalize_existing_example(example: dict[str, Any], example_category: str) -> dict[str, Any]:
    source_kind = _extract_example_field(example, "source_kind", "source")
    query_intent = _extract_example_field(example, "query_intent", "intent")
    difficulty = _extract_example_field(example, "difficulty", "difficulty_label")
    payload = dict(example.get("sample_payload") or example.get("payload") or {})
    default_provenance = (
        SyntheticEvalProvenance.SYNTHETIC_FROM_SEED_EXAMPLES.value
        if example_category == "seed_examples"
        else example_category
    )
    return {
        "sample_id": example.get("sample_id"),
        "recipe_kind": example.get("recipe_kind"),
        "source_kind": source_kind,
        "query_intent": query_intent,
        "difficulty": difficulty,
        "sample_payload": payload,
        "provenance": str(example.get("provenance") or default_provenance),
        "example_category": example_category,
    }


def _summarize_coverage(examples: Sequence[dict[str, Any]]) -> SyntheticEvalCoverageSummary:
    coverage = SyntheticEvalCoverageSummary()
    coverage.sources = _stable_unique(_extract_example_field(example, "source_kind", "source") for example in examples)
    coverage.query_intents = _stable_unique(_extract_example_field(example, "query_intent", "intent") for example in examples)
    coverage.difficulties = _stable_unique(_extract_example_field(example, "difficulty", "difficulty_label") for example in examples)
    coverage.failure_modes = _stable_unique(
        _infer_failure_mode(
            query_intent=_extract_example_field(example, "query_intent", "intent"),
            difficulty=_extract_example_field(example, "difficulty", "difficulty_label"),
        )
        for example in examples
    )
    return coverage


def _infer_failure_mode(*, query_intent: str, difficulty: str) -> str:
    lowered = f"{query_intent} {difficulty}".lower()
    if "abstention" in lowered:
        return "requires_abstention"
    if "distractor" in lowered:
        return "confused_by_distractors"
    if "multi-source" in lowered or "multi source" in lowered:
        return "cross_source_resolution_gap"
    if "comparison" in lowered:
        return "comparison_confusion"
    return "missing_relevant_span"


def _build_failure_oriented_query(
    *,
    source_kind: str,
    query_intent: str,
    difficulty: str,
    topic_hint: str,
    index: int,
) -> str:
    subject = topic_hint or f"{source_kind} corpus"
    return (
        f"[{source_kind}] failure-case {index + 1}: "
        f"ask for a {query_intent} task with {difficulty} coverage around {subject}"
    )


def _build_target_payload(
    *,
    source_kind: str,
    query_intent: str,
    difficulty: str,
    failure_mode: str,
    topic_hint: str,
) -> dict[str, Any]:
    target_payload: dict[str, Any] = {
        "corpus_focus": topic_hint or source_kind,
        "stratification": {
            "source_kind": source_kind,
            "query_intent": query_intent,
            "difficulty": difficulty,
        },
        "failure_mode": failure_mode,
    }
    if source_kind == "media_db":
        target_payload["relevant_media_ids"] = []
        target_payload["relevant_note_ids"] = []
        target_payload["relevant_spans"] = []
        target_payload["distractor_metadata"] = {"source_kind": source_kind, "difficulty": difficulty}
    else:
        target_payload["relevant_note_ids"] = []
        target_payload["relevant_media_ids"] = []
        target_payload["expected_behavior"] = "abstain" if failure_mode == "requires_abstention" else "answer"
    return target_payload


def _derive_topic_hint(*examples_groups: Sequence[dict[str, Any]]) -> str:
    for examples in examples_groups:
        for example in examples:
            payload = example.get("sample_payload") or {}
            query = str(payload.get("query") or "").strip()
            if query:
                tokens = query.split()
                return " ".join(tokens[:6])
    return ""


def _extract_example_field(example: dict[str, Any], *field_names: str) -> str:
    for field_name in field_names:
        value = example.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    payload = example.get("sample_metadata") or {}
    for field_name in field_names:
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    payload = example.get("sample_payload") or example.get("payload") or {}
    for field_name in field_names:
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    return ""


def _dedupe_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_stable_unique(value for value in values if value))


def _dedupe_int_tuple(values: Sequence[int] | Sequence[Any]) -> tuple[int, ...]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _normalize_int_tuple(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        values = (values,)
    return _dedupe_int_tuple(values)


def _normalize_note_id_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        values = (values,)
    return _dedupe_tuple(tuple(str(value).strip() for value in values if str(value).strip()))


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        return None
    return bool(value)


def _stable_unique(values: Sequence[str] | Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


__all__ = [
    "SyntheticEvalCorpusScope",
    "SyntheticEvalCoverageSummary",
    "SyntheticEvalGenerationPlan",
    "SyntheticEvalStructuredTuple",
    "convert_structured_tuples_to_draft_samples",
    "generate_structured_tuples",
    "identify_missing_coverage",
    "ingest_examples",
    "resolve_corpus_scope",
    "summarize_coverage",
]
