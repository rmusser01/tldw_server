"""Deterministic topic normalization, alias resolution, and ranking."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from .topic_aliases import (
    DEFAULT_NAMESPACE,
    NORMALIZATION_VERSION,
    clean_label_text,
    dedupe_reasons,
    resolve_topic_alias,
    topic_key_for,
)
from .types import NormalizedTopicLabel, RankedTopic, ResolvedTopicLabel, TopicCandidate


_EVIDENCE_PRIORITY = {
    "grounded": 0,
    "weakly_grounded": 1,
    "derived": 2,
}
_EVIDENCE_REASON_BY_CLASS = {
    "grounded": "source_citation",
    "weakly_grounded": "tag_match",
    "derived": "derived_label",
}


def _normalize_label_text(label: object) -> str | None:
    return clean_label_text(label)


def _resolve_label(raw_label: object, evidence_class: str) -> ResolvedTopicLabel | None:
    cleaned = _normalize_label_text(raw_label)
    if cleaned is None:
        return None

    namespace, semantic_basis = resolve_topic_alias(cleaned)
    if not semantic_basis:
        return None

    topic_key = topic_key_for(namespace, semantic_basis)
    canonical_slug = topic_key.split(":", 1)[1]
    evidence_reason = _EVIDENCE_REASON_BY_CLASS.get(evidence_class, "derived_label")
    source_count = 1 if evidence_class == "grounded" else 0

    return ResolvedTopicLabel(
        namespace=namespace,
        canonical_slug=canonical_slug,
        canonical_label=semantic_basis,
        semantic_label=semantic_basis,
        topic_key=topic_key,
        normalization_version=NORMALIZATION_VERSION,
        raw_labels=[str(raw_label)],
        evidence_reasons=[evidence_reason],
        source_count=source_count,
    )


def normalize_topic_labels(labels: Iterable[object]) -> list[NormalizedTopicLabel]:
    resolved_topics = (_resolve_label(raw_label, "grounded") for raw_label in labels)
    grouped: "OrderedDict[str, list[ResolvedTopicLabel]]" = OrderedDict()
    for resolved in resolved_topics:
        if resolved is None:
            continue
        grouped.setdefault(resolved.topic_key, []).append(resolved)

    normalized_group_rows: list[tuple[str, str, list[str]]] = []
    for grouped_topics in grouped.values():
        semantic_labels = [resolved.semantic_label for resolved in grouped_topics]
        semantic_canonical = min(semantic_labels, key=_canonical_label_preference)
        namespace = grouped_topics[0].namespace
        raw_labels: list[str] = []
        for resolved in grouped_topics:
            for raw_label in resolved.raw_labels:
                if raw_label not in raw_labels:
                    raw_labels.append(raw_label)

        normalized_group_rows.append((namespace, semantic_canonical, raw_labels))

    semantic_label_counts: dict[str, int] = {}
    for _, semantic_canonical, _ in normalized_group_rows:
        semantic_label_counts[semantic_canonical] = semantic_label_counts.get(semantic_canonical, 0) + 1

    normalized_topics: list[NormalizedTopicLabel] = []
    for namespace, semantic_canonical, raw_labels in normalized_group_rows:
        canonical_label = semantic_canonical
        if semantic_label_counts.get(semantic_canonical, 0) > 1:
            canonical_label = semantic_canonical
            if namespace != DEFAULT_NAMESPACE and not semantic_canonical.startswith(f"{namespace} "):
                canonical_label = f"{namespace} {semantic_canonical}".strip()
        normalized_topics.append(
            NormalizedTopicLabel(
                canonical_label=canonical_label,
                raw_labels=raw_labels,
            )
        )

    return normalized_topics


def _canonical_label_preference(label: str) -> tuple[int, str]:
    return (len(label), label)


def _resolved_evidence_class(resolved: ResolvedTopicLabel) -> str:
    if "source_citation" in resolved.evidence_reasons:
        return "grounded"
    if "tag_match" in resolved.evidence_reasons:
        return "weakly_grounded"
    return "derived"


def _resolved_canonical_selection_key(
    resolved: ResolvedTopicLabel,
) -> tuple[int, tuple[int, str], str]:
    evidence_class = _resolved_evidence_class(resolved)
    return (
        _EVIDENCE_PRIORITY[evidence_class],
        _canonical_label_preference(resolved.canonical_label),
        resolved.canonical_label,
    )


def _merge_resolved_group(grouped_topics: list[ResolvedTopicLabel]) -> ResolvedTopicLabel:
    representative = min(grouped_topics, key=_resolved_canonical_selection_key)
    raw_labels: list[str] = []
    evidence_reasons: list[str] = []
    source_count = 0
    for resolved in grouped_topics:
        for raw_label in resolved.raw_labels:
            if raw_label not in raw_labels:
                raw_labels.append(raw_label)
        evidence_reasons = dedupe_reasons([*evidence_reasons, *resolved.evidence_reasons])
        source_count += resolved.source_count
    if not raw_labels:
        raw_labels.append(representative.canonical_label)

    return ResolvedTopicLabel(
        namespace=representative.namespace,
        canonical_slug=representative.canonical_slug,
        canonical_label=representative.canonical_label,
        semantic_label=representative.semantic_label,
        topic_key=representative.topic_key,
        normalization_version=representative.normalization_version,
        raw_labels=raw_labels,
        evidence_reasons=evidence_reasons,
        source_count=source_count,
    )


def _merge_resolved_topics(
    resolved_topics: Iterable[ResolvedTopicLabel],
) -> list[ResolvedTopicLabel]:
    grouped: "OrderedDict[str, list[ResolvedTopicLabel]]" = OrderedDict()
    for resolved in resolved_topics:
        grouped.setdefault(resolved.topic_key, []).append(resolved)

    return [_merge_resolved_group(grouped_topics) for grouped_topics in grouped.values()]


def resolve_topic_candidates(
    *,
    source_labels: Iterable[object],
    tag_labels: Iterable[object],
    derived_labels: Iterable[object],
) -> list[TopicCandidate]:
    resolved_topics: list[ResolvedTopicLabel] = []
    evidence_classes = (
        ("grounded", source_labels),
        ("weakly_grounded", tag_labels),
        ("derived", derived_labels),
    )

    for evidence_class, labels in evidence_classes:
        for raw_label in labels:
            resolved = _resolve_label(raw_label, evidence_class)
            if resolved is not None:
                resolved_topics.append(resolved)

    grouped = _merge_resolved_topics(resolved_topics)
    semantic_label_counts: dict[str, int] = {}
    for resolved in grouped:
        semantic_label_counts[resolved.semantic_label] = semantic_label_counts.get(resolved.semantic_label, 0) + 1

    candidates: list[TopicCandidate] = []
    for resolved in grouped:
        evidence_class = _resolved_evidence_class(resolved)
        canonical_label = resolved.semantic_label
        if semantic_label_counts.get(resolved.semantic_label, 0) > 1:
            if resolved.namespace != DEFAULT_NAMESPACE and not canonical_label.startswith(f"{resolved.namespace} "):
                canonical_label = f"{resolved.namespace} {canonical_label}".strip()

        candidates.append(
            TopicCandidate(
                canonical_label=canonical_label,
                semantic_label=resolved.semantic_label,
                raw_labels=list(resolved.raw_labels),
                evidence_class=evidence_class,  # type: ignore[arg-type]
                topic_key=resolved.topic_key,
                normalization_version=resolved.normalization_version,
                source_count=resolved.source_count,
                evidence_reasons=list(resolved.evidence_reasons),
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            _EVIDENCE_PRIORITY[candidate.evidence_class],
            candidate.topic_key or "",
            _canonical_label_preference(candidate.canonical_label),
        ),
    )


def _topic_key_set(labels: Iterable[object]) -> set[str]:
    topic_keys: set[str] = set()
    for raw_label in labels:
        resolved = _resolve_label(raw_label, "grounded")
        if resolved is not None:
            topic_keys.add(resolved.topic_key)
    return topic_keys


def rank_suggestion_topics(
    candidates: Iterable[TopicCandidate],
    *,
    weakness_labels: Iterable[object],
    adjacent_labels: Iterable[object],
    exploratory_labels: Iterable[object],
) -> list[RankedTopic]:
    weakness = _topic_key_set(weakness_labels)
    adjacent = _topic_key_set(adjacent_labels)
    exploratory = _topic_key_set(exploratory_labels)

    ranked: list[tuple[tuple[int, int, str], RankedTopic]] = []
    for candidate in candidates:
        topic_key = candidate.topic_key or topic_key_for("general", candidate.canonical_label)
        normalization_version = candidate.normalization_version or NORMALIZATION_VERSION
        evidence_reasons = list(candidate.evidence_reasons)
        if topic_key in weakness:
            rank_reason = "weakness"
            source_aware = candidate.evidence_class != "derived"
            priority = 0
            evidence_reasons = dedupe_reasons([*evidence_reasons, "missed_question"])
        elif topic_key in exploratory:
            rank_reason = "exploratory"
            source_aware = False
            priority = 2
        elif topic_key in adjacent and candidate.evidence_class != "derived":
            rank_reason = "adjacent"
            source_aware = True
            priority = 1
        else:
            rank_reason = "candidate"
            source_aware = False
            priority = 3

        ranked.append(
            (
                (priority, _EVIDENCE_PRIORITY[candidate.evidence_class], topic_key),
                RankedTopic(
                    canonical_label=candidate.canonical_label,
                    semantic_label=candidate.semantic_label,
                    raw_labels=list(candidate.raw_labels),
                    evidence_class=candidate.evidence_class,
                    rank_reason=rank_reason,
                    adjacency_is_source_aware=source_aware,
                    topic_key=topic_key,
                    normalization_version=normalization_version,
                    source_count=candidate.source_count,
                    evidence_reasons=evidence_reasons,
                ),
            )
        )

    ranked.sort(key=lambda item: item[0])
    return [topic for _, topic in ranked]
