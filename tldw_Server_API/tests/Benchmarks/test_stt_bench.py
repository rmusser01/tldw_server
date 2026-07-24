"""Tests for the deterministic native STT benchmark scorer."""

from __future__ import annotations

import math

import pytest
from Helper_Scripts.benchmarks.stt_bench import (
    EN_PROFILE,
    SCORER_VERSION,
    STRICT_PROFILE,
    EditCounts,
    edit_counts,
    normalize_en_v1,
    normalize_exact_text,
    normalize_strict_v1,
    percentile_type7,
    score_transcript,
)
from hypothesis import given, settings
from hypothesis import strategies as st


def test_normalize_public_version_constants_are_stable():
    assert SCORER_VERSION == "stt-score-v1"
    assert STRICT_PROFILE == "strict-v1"
    assert EN_PROFILE == "en-v1"


def test_normalize_exact_changes_only_crlf_and_bare_cr():
    assert normalize_exact_text("A\r\nB\rC\n d\t") == "A\nB\nC\n d\t"


def test_normalize_strict_applies_nfc_and_collapses_unicode_whitespace():
    assert normalize_strict_v1(" \te\u0301\u00a0\nHello,\rWorld! ") == "é Hello, World!"


def test_normalize_en_applies_ordered_unicode_rules():
    assert normalize_en_v1("  ＣＡＮ’T—Stop… 你好，１２  ") == "can't stop 你好 12"


@pytest.mark.parametrize("apostrophe", ["'", "\u2018", "\u2019", "\u02bc", "\uff07"])
def test_normalize_en_preserves_only_internal_mapped_apostrophes(apostrophe):
    assert normalize_en_v1(f"we{apostrophe}re") == "we're"
    assert normalize_en_v1(f"{apostrophe}a{apostrophe} a{apostrophe}{apostrophe}b") == "a a b"


def test_normalize_en_keeps_meaningful_contraction_and_number_differences():
    assert normalize_en_v1("we're") != normalize_en_v1("were")
    assert normalize_en_v1("can't") != normalize_en_v1("cant")
    assert normalize_en_v1("１２ twelve") == "12 twelve"


def test_normalize_en_preserves_non_english_letters_and_non_punctuation_symbols():
    assert normalize_en_v1("CAFÉ Привет १२ a+b") == "café привет १२ a+b"


@pytest.mark.parametrize(
    "normalizer",
    [normalize_exact_text, normalize_strict_v1, normalize_en_v1],
)
@pytest.mark.parametrize("invalid", [None, True, 7, b"text"])
def test_normalize_rejects_non_string_inputs(normalizer, invalid):
    with pytest.raises(TypeError):
        normalizer(invalid)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        (["a", "b"], ["a", "c"], EditCounts(1, 0, 0, 2)),
        (["a", "b"], ["a"], EditCounts(0, 1, 0, 2)),
        (["a"], ["a", "b"], EditCounts(0, 0, 1, 1)),
        ([], [], EditCounts(0, 0, 0, 0)),
        ([], ["a", "b"], EditCounts(0, 0, 2, 0)),
    ],
)
def test_score_edit_counts_examples(reference, hypothesis, expected):
    assert edit_counts(reference, hypothesis) == expected


def test_score_tie_priority_prefers_substitution_to_delete_insert():
    assert edit_counts(["b", "a"], ["a", "b"]) == EditCounts(2, 0, 0, 2)


def test_score_tie_priority_prefers_match_when_it_changes_operation_totals():
    assert edit_counts(["b", "a", "a"], ["a", "b", "a"]) == EditCounts(2, 0, 0, 3)


def test_score_tie_priority_prefers_deletion_to_insertion():
    assert edit_counts(
        ["b", "a", "b", "a"],
        ["a", "b", "b", "a", "b"],
    ) == EditCounts(0, 1, 2, 4)


def test_score_empty_reference_rate_keeps_insertion_penalty():
    assert edit_counts([], []).rate == 0.0
    assert edit_counts([], ["a", "b"]).rate == 2.0


def test_score_transcript_reports_exact_strict_and_normalized_metrics():
    score = score_transcript(
        "Hello, world",
        "hello world",
        normalization_profile=EN_PROFILE,
    )

    assert score.exact_match is False
    assert score.strict_wer == EditCounts(1, 0, 0, 2)
    assert score.normalized_wer == EditCounts(0, 0, 0, 2)
    assert score.normalized_cer.errors == 0


def test_score_strict_profile_reuses_strict_text_for_normalized_metrics():
    score = score_transcript(
        "a\u00a0b",
        "a b",
        normalization_profile=STRICT_PROFILE,
    )

    assert score.exact_match is False
    assert score.strict_wer.errors == 0
    assert score.strict_cer.errors == 0
    assert score.normalized_wer == score.strict_wer
    assert score.normalized_cer == score.strict_cer


def test_score_cer_counts_unicode_code_points_including_internal_spaces():
    score = score_transcript("a b", "ab", normalization_profile=STRICT_PROFILE)

    assert score.strict_cer == EditCounts(0, 1, 0, 3)


def test_score_empty_hypothesis_uses_reference_denominators():
    score = score_transcript("one two", "", normalization_profile=STRICT_PROFILE)

    assert score.strict_wer == EditCounts(0, 2, 0, 2)
    assert score.strict_wer.rate == 1.0
    assert score.strict_cer == EditCounts(0, 7, 0, 7)
    assert score.strict_cer.rate == 1.0


def test_score_empty_preprocessed_strings_use_empty_sequences():
    score = score_transcript("\t", "\u00a0", normalization_profile=EN_PROFILE)

    assert score.strict_wer.reference_units == 0
    assert score.strict_cer.reference_units == 0
    assert score.normalized_wer.reference_units == 0
    assert score.normalized_cer.reference_units == 0


def test_score_rejects_unsupported_normalization_profile():
    with pytest.raises(ValueError, match="normalization profile"):
        score_transcript("a", "a", normalization_profile="fr-v1")


@pytest.mark.parametrize("invalid", [None, True, 7, b"text"])
def test_score_rejects_non_string_transcripts(invalid):
    with pytest.raises(TypeError):
        score_transcript(invalid, "text", normalization_profile=EN_PROFILE)
    with pytest.raises(TypeError):
        score_transcript("text", invalid, normalization_profile=EN_PROFILE)


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.50, 2.5), (0.90, 3.7), (0.95, 3.85), (0.99, 3.97)],
)
def test_percentile_type7_interpolates_documented_percentiles(p, expected):
    assert percentile_type7([4.0, 1.0, 3.0, 2.0], p) == pytest.approx(expected)


def test_percentile_type7_handles_empty_and_single_value_inputs():
    assert percentile_type7([], 0.5) is None
    assert percentile_type7([4.25], 0.0) == 4.25
    assert percentile_type7([4.25], 1.0) == 4.25


@pytest.mark.parametrize("p", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_percentile_type7_rejects_invalid_percentile(p):
    with pytest.raises(ValueError):
        percentile_type7([1.0], p)


@pytest.mark.parametrize(
    "values",
    [[math.nan], [math.inf], [-math.inf], [1.0, math.nan]],
)
def test_percentile_type7_rejects_non_finite_observations(values):
    with pytest.raises(ValueError, match="finite"):
        percentile_type7(values, 0.5)


def test_percentile_type7_rejects_out_of_range_huge_percentile():
    with pytest.raises(ValueError):
        percentile_type7([1.0], 10**400)


def test_percentile_type7_rejects_unrepresentable_huge_observation():
    with pytest.raises(ValueError, match="finite"):
        percentile_type7([10**400], 0.5)


@pytest.mark.parametrize(
    ("values", "p"),
    [([True], 0.5), (["1.0"], 0.5), ([1.0], True), ([1.0], "0.5")],
)
def test_percentile_type7_rejects_boolean_and_non_numeric_inputs(values, p):
    with pytest.raises(TypeError):
        percentile_type7(values, p)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(text=st.text(max_size=80))
def test_normalize_profiles_are_idempotent(text):
    for normalizer in (normalize_exact_text, normalize_strict_v1, normalize_en_v1):
        normalized = normalizer(text)
        assert normalizer(normalized) == normalized


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(units=st.lists(st.sampled_from(["", "a", "b", "é", "1"]), max_size=12))
def test_score_edit_counts_identity(units):
    assert edit_counts(units, units) == EditCounts(0, 0, 0, len(units))


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    reference=st.text(max_size=24),
    hypothesis=st.text(max_size=24),
)
def test_score_transcript_is_deterministic(reference, hypothesis):
    first = score_transcript(reference, hypothesis, normalization_profile=EN_PROFILE)
    second = score_transcript(reference, hypothesis, normalization_profile=EN_PROFILE)
    assert first == second


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    reference=st.lists(st.sampled_from(["", "a", "b", "é", "1"]), max_size=12),
    hypothesis=st.lists(st.sampled_from(["", "a", "b", "é", "1"]), max_size=12),
)
def test_score_edit_counts_are_non_negative_and_length_consistent(reference, hypothesis):
    counts = edit_counts(reference, hypothesis)

    assert counts.substitutions >= 0
    assert counts.deletions >= 0
    assert counts.insertions >= 0
    assert counts.reference_units == len(reference)
    assert len(hypothesis) == counts.reference_units - counts.deletions + counts.insertions
    assert counts.errors >= abs(len(reference) - len(hypothesis))


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    pairs=st.lists(
        st.tuples(
            st.lists(
                st.sampled_from(["", "a", "b", "é", "1"]),
                min_size=1,
                max_size=5,
            ),
            st.lists(st.sampled_from(["", "a", "b", "é", "1"]), max_size=5),
        ),
        min_size=1,
        max_size=4,
    )
)
def test_score_pooled_counts_reconstruct_pooled_rate(pairs):
    samples = [edit_counts(reference, hypothesis) for reference, hypothesis in pairs]
    pooled = EditCounts(
        substitutions=sum(item.substitutions for item in samples),
        deletions=sum(item.deletions for item in samples),
        insertions=sum(item.insertions for item in samples),
        reference_units=sum(item.reference_units for item in samples),
    )

    assert pooled.errors == sum(item.errors for item in samples)
    assert pooled.rate == sum(item.errors for item in samples) / sum(
        item.reference_units for item in samples
    )
