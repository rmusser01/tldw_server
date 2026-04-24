from tldw_Server_API.app.core.StudySuggestions.topic_aliases import (
    NORMALIZATION_VERSION,
    resolve_topic_alias,
)
from tldw_Server_API.app.core.StudySuggestions.topic_pipeline import (
    normalize_topic_labels,
    rank_suggestion_topics,
    resolve_topic_candidates,
)
from tldw_Server_API.app.core.StudySuggestions.types import TopicCandidate


def _candidate_by_topic_key(candidates, topic_key):
    for candidate in candidates:
        if candidate.topic_key == topic_key:
            return candidate
    raise AssertionError(f"missing candidate for {topic_key}")


def test_grounded_labels_preserve_unaliased_source_text_and_collapse_explicit_synonyms():
    candidates = resolve_topic_candidates(
        source_labels=[
            "Kidney function",
            "Kidney physiology",
            "renal physiology",
        ],
        tag_labels=[],
        derived_labels=[],
    )

    assert len(candidates) == 2  # nosec B101
    function_candidate = _candidate_by_topic_key(candidates, "renal:renal-function")
    physiology_candidate = _candidate_by_topic_key(candidates, "renal:renal-physiology")

    assert function_candidate.canonical_label == "renal function"  # nosec B101
    assert function_candidate.raw_labels[0] == "Kidney function"  # nosec B101
    assert function_candidate.semantic_label == "renal function"  # nosec B101
    assert "Kidney function" in function_candidate.raw_labels  # nosec B101
    assert physiology_candidate.canonical_label == "renal physiology"  # nosec B101
    assert physiology_candidate.raw_labels[0] == "Kidney physiology"  # nosec B101
    assert physiology_candidate.semantic_label == "renal physiology"  # nosec B101
    assert "Kidney physiology" in physiology_candidate.raw_labels  # nosec B101
    assert "renal physiology" in physiology_candidate.raw_labels  # nosec B101
    assert function_candidate.normalization_version == NORMALIZATION_VERSION  # nosec B101
    assert physiology_candidate.normalization_version == NORMALIZATION_VERSION  # nosec B101


def test_source_and_tag_labels_collapse_by_semantic_topic_key_without_rewriting_source_label():
    candidates = resolve_topic_candidates(
        source_labels=["Kidney function"],
        tag_labels=["renal function"],
        derived_labels=[],
    )

    assert len(candidates) == 1  # nosec B101
    candidate = candidates[0]
    assert candidate.topic_key == "renal:renal-function"  # nosec B101
    assert candidate.canonical_label == "renal function"  # nosec B101
    assert candidate.raw_labels[0] == "Kidney function"  # nosec B101
    assert candidate.semantic_label == "renal function"  # nosec B101
    assert "Kidney function" in candidate.raw_labels  # nosec B101
    assert "renal function" in candidate.raw_labels  # nosec B101
    assert candidate.source_count == 1  # nosec B101


def test_single_explicit_alias_source_label_uses_stable_semantic_canonical_label():
    candidate = resolve_topic_candidates(
        source_labels=["Kidney physiology"],
        tag_labels=[],
        derived_labels=[],
    )[0]

    assert candidate.topic_key == "renal:renal-physiology"  # nosec B101
    assert candidate.canonical_label == "renal physiology"  # nosec B101
    assert candidate.raw_labels == ["Kidney physiology"]  # nosec B101
    assert candidate.semantic_label == "renal physiology"  # nosec B101


def test_normalize_topic_labels_uses_semantic_canonical_label_for_explicit_aliases():
    normalized = normalize_topic_labels(["Kidney physiology"])

    assert len(normalized) == 1  # nosec B101
    assert normalized[0].canonical_label == "renal physiology"  # nosec B101
    assert normalized[0].raw_labels == ["Kidney physiology"]  # nosec B101


def test_resolve_topic_alias_normalizes_raw_input_before_lookup():
    assert resolve_topic_alias("Kidney_Physiology") == (  # nosec B101
        "renal",
        "renal physiology",
    )


def test_normalize_topic_labels_collapses_mixed_raw_forms_stably():
    normalized = normalize_topic_labels(["Renal Basics", "renal-basics", "Kidney basics"])

    assert len(normalized) == 1  # nosec B101
    assert normalized[0].canonical_label == "renal basics"  # nosec B101
    assert normalized[0].raw_labels == [  # nosec B101
        "Renal Basics",
        "renal-basics",
        "Kidney basics",
    ]


def test_normalize_topic_labels_preserves_namespace_disambiguation_for_semantic_collisions():
    normalized = normalize_topic_labels(["renal overview", "cardiac overview"])

    assert [topic.canonical_label for topic in normalized] == [  # nosec B101
        "renal overview",
        "cardiac overview",
    ]
    assert [topic.raw_labels for topic in normalized] == [  # nosec B101
        ["renal overview"],
        ["cardiac overview"],
    ]


def test_normalize_topic_labels_uses_stable_namespace_labels_for_token_alias_collisions():
    normalized = normalize_topic_labels(["renal overview", "heart overview"])

    assert [topic.canonical_label for topic in normalized] == [  # nosec B101
        "renal overview",
        "cardiac overview",
    ]
    assert [topic.raw_labels for topic in normalized] == [  # nosec B101
        ["renal overview"],
        ["heart overview"],
    ]


def test_normalize_topic_labels_keeps_general_namespace_collision_unprefixed():
    normalized = normalize_topic_labels(["overview", "heart overview"])

    assert [topic.canonical_label for topic in normalized] == [  # nosec B101
        "overview",
        "cardiac overview",
    ]
    assert [topic.raw_labels for topic in normalized] == [  # nosec B101
        ["overview"],
        ["heart overview"],
    ]


def test_canonical_label_selection_is_order_independent_for_colliding_source_labels():
    forward = resolve_topic_candidates(
        source_labels=["Kidney function", "renal function"],
        tag_labels=[],
        derived_labels=[],
    )
    reverse = resolve_topic_candidates(
        source_labels=["renal function", "Kidney function"],
        tag_labels=[],
        derived_labels=[],
    )

    assert len(forward) == 1  # nosec B101
    assert len(reverse) == 1  # nosec B101
    assert forward[0].topic_key == reverse[0].topic_key == "renal:renal-function"  # nosec B101
    assert forward[0].canonical_label == reverse[0].canonical_label  # nosec B101
    assert forward[0].raw_labels[0] in {"Kidney function", "renal function"}  # nosec B101
    assert reverse[0].raw_labels[0] in {"Kidney function", "renal function"}  # nosec B101
    assert forward[0].canonical_label == "renal function"  # nosec B101


def test_source_count_only_tracks_grounded_source_labels():
    candidates = resolve_topic_candidates(
        source_labels=["Kidney physiology"],
        tag_labels=["Cardiac overview"],
        derived_labels=["Kidney function"],
    )

    assert _candidate_by_topic_key(candidates, "renal:renal-physiology").source_count == 1  # nosec B101
    assert _candidate_by_topic_key(candidates, "cardiac:overview").source_count == 0  # nosec B101
    assert _candidate_by_topic_key(candidates, "renal:renal-function").source_count == 0  # nosec B101


def test_source_metadata_outranks_tags_and_tags_outrank_derived_labels():
    candidates = resolve_topic_candidates(
        source_labels=["Kidney function"],
        tag_labels=["renal basics"],
        derived_labels=["how kidneys work"],
    )

    assert [candidate.evidence_class for candidate in candidates] == [  # nosec B101
        "grounded",
        "weakly_grounded",
        "derived",
    ]
    assert candidates[0].canonical_label == "renal function"  # nosec B101


def test_different_namespaces_can_reuse_the_same_slug_without_collision():
    candidates = resolve_topic_candidates(
        source_labels=["Renal overview", "Cardiac overview"],
        tag_labels=[],
        derived_labels=[],
    )

    assert {candidate.topic_key for candidate in candidates} == {  # nosec B101
        "renal:overview",
        "cardiac:overview",
    }
    assert len({candidate.topic_key for candidate in candidates}) == 2  # nosec B101
    assert {candidate.normalization_version for candidate in candidates} == {  # nosec B101
        NORMALIZATION_VERSION
    }


def test_ranked_topics_default_normalization_version_when_missing():
    ranked = rank_suggestion_topics(
        [
            TopicCandidate(
                canonical_label="kidney function",
                semantic_label="renal function",
                raw_labels=["Kidney function"],
                evidence_class="grounded",
                topic_key="renal:renal-function",
            )
        ],
        weakness_labels=[],
        adjacent_labels=[],
        exploratory_labels=[],
    )

    assert ranked[0].topic_key == "renal:renal-function"  # nosec B101
    assert ranked[0].normalization_version == NORMALIZATION_VERSION  # nosec B101


def test_ranked_topics_carry_normalization_version_and_topic_identity():
    candidates = resolve_topic_candidates(
        source_labels=["Renal physiology"],
        tag_labels=["Cardiac overview"],
        derived_labels=[],
    )

    ranked = rank_suggestion_topics(
        candidates,
        weakness_labels=[],
        adjacent_labels=[],
        exploratory_labels=[],
    )

    assert [topic.topic_key for topic in ranked] == [  # nosec B101
        "renal:renal-physiology",
        "cardiac:overview",
    ]
    assert all(  # nosec B101
        topic.normalization_version == NORMALIZATION_VERSION for topic in ranked
    )
    assert all(topic.source_count >= 0 for topic in ranked)  # nosec B101


def test_display_label_and_raw_label_drift_do_not_change_semantic_identity():
    kidney_variant = resolve_topic_candidates(
        source_labels=["Kidney function"],
        tag_labels=[],
        derived_labels=[],
    )[0]
    renal_variant = resolve_topic_candidates(
        source_labels=["renal function"],
        tag_labels=[],
        derived_labels=[],
    )[0]

    assert kidney_variant.topic_key == renal_variant.topic_key  # nosec B101
    assert kidney_variant.semantic_label == renal_variant.semantic_label == "renal function"  # nosec B101
    assert kidney_variant.canonical_label == "renal function"  # nosec B101
    assert kidney_variant.raw_labels[0] == "Kidney function"  # nosec B101
    assert renal_variant.canonical_label == "renal function"  # nosec B101
    assert renal_variant.raw_labels[0] == "renal function"  # nosec B101


def test_weakness_first_ranking_preserves_weakness_before_adjacent_topics():
    candidates = resolve_topic_candidates(
        source_labels=["Kidney function", "Electrolyte balance"],
        tag_labels=[],
        derived_labels=[],
    )

    ranked = rank_suggestion_topics(
        candidates,
        weakness_labels=["Electrolyte balance"],
        adjacent_labels=["Kidney function"],
        exploratory_labels=[],
    )

    assert [topic.canonical_label for topic in ranked[:2]] == [  # nosec B101
        "electrolyte balance",
        "renal function",
    ]
    assert ranked[0].rank_reason == "weakness"  # nosec B101
    assert ranked[1].rank_reason == "adjacent"  # nosec B101


def test_weakness_ranking_adds_missed_question_provenance():
    candidates = resolve_topic_candidates(
        source_labels=["Kidney function"],
        tag_labels=[],
        derived_labels=[],
    )

    ranked = rank_suggestion_topics(
        candidates,
        weakness_labels=["Kidney function"],
        adjacent_labels=[],
        exploratory_labels=[],
    )

    assert ranked[0].rank_reason == "weakness"  # nosec B101
    assert "missed_question" in ranked[0].evidence_reasons  # nosec B101


def test_exploratory_topics_never_claim_source_aware_adjacency():
    candidates = resolve_topic_candidates(
        source_labels=[],
        tag_labels=[],
        derived_labels=["dialysis overview"],
    )

    ranked = rank_suggestion_topics(
        candidates,
        weakness_labels=[],
        adjacent_labels=["dialysis overview"],
        exploratory_labels=["dialysis overview"],
    )

    assert ranked[0].canonical_label == "dialysis overview"  # nosec B101
    assert ranked[0].rank_reason == "exploratory"  # nosec B101
    assert ranked[0].adjacency_is_source_aware is False  # nosec B101
