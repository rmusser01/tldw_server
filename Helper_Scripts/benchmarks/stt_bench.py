"""Deterministic normalization and scoring for native STT benchmarks."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

SCORER_VERSION = "stt-score-v1"
STRICT_PROFILE = "strict-v1"
EN_PROFILE = "en-v1"

_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\uff07": "'",
    }
)


@dataclass(frozen=True)
class EditCounts:
    """Deterministic edit-operation totals for one sequence pair."""

    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        """Return the total edit distance."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Return errors divided by the non-zero reference denominator."""
        return self.errors / max(self.reference_units, 1)


@dataclass(frozen=True)
class TranscriptScore:
    """Exact, strict, and normalized scores for one transcript pair."""

    exact_match: bool
    strict_wer: EditCounts
    strict_cer: EditCounts
    normalized_wer: EditCounts
    normalized_cer: EditCounts


def _require_text(text: str) -> str:
    """Return transcript text or reject non-string input."""
    if not isinstance(text, str):
        raise TypeError("transcript text must be a string")
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse maximal Unicode whitespace runs and trim the result."""
    collapsed: list[str] = []
    pending_space = False
    for character in text:
        if character.isspace():
            pending_space = bool(collapsed)
        else:
            if pending_space:
                collapsed.append(" ")
            collapsed.append(character)
            pending_space = False
    return "".join(collapsed)


def normalize_exact_text(text: str) -> str:
    """Normalize only CRLF and bare CR line endings to LF."""
    return _require_text(text).replace("\r\n", "\n").replace("\r", "\n")


def normalize_strict_v1(text: str) -> str:
    """Apply NFC and canonical Unicode whitespace handling."""
    return _collapse_whitespace(unicodedata.normalize("NFC", _require_text(text)))


def normalize_en_v1(text: str) -> str:
    """Apply the deterministic English benchmark normalization profile."""
    normalized = unicodedata.normalize("NFKC", _require_text(text))
    normalized = normalized.translate(_APOSTROPHE_TRANSLATION).casefold()
    characters: list[str] = []
    for index, character in enumerate(normalized):
        if character == "'":
            if (
                index > 0
                and index + 1 < len(normalized)
                and normalized[index - 1].isalnum()
                and normalized[index + 1].isalnum()
            ):
                characters.append(character)
            else:
                characters.append(" ")
        elif unicodedata.category(character).startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)
    return _collapse_whitespace("".join(characters))


def edit_counts(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> EditCounts:
    """Return deterministic Levenshtein operation totals using two DP rows."""
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for reference_index, reference_unit in enumerate(reference, start=1):
        current = [(reference_index, 0, reference_index, 0)]
        for hypothesis_index, hypothesis_unit in enumerate(hypothesis, start=1):
            deletion_base = previous[hypothesis_index]
            insertion_base = current[hypothesis_index - 1]
            deletion = (
                deletion_base[0] + 1,
                deletion_base[1],
                deletion_base[2] + 1,
                deletion_base[3],
            )
            insertion = (
                insertion_base[0] + 1,
                insertion_base[1],
                insertion_base[2],
                insertion_base[3] + 1,
            )
            if reference_unit == hypothesis_unit:
                candidates = (previous[hypothesis_index - 1], deletion, insertion)
            else:
                substitution_base = previous[hypothesis_index - 1]
                substitution = (
                    substitution_base[0] + 1,
                    substitution_base[1] + 1,
                    substitution_base[2],
                    substitution_base[3],
                )
                candidates = (substitution, deletion, insertion)
            current.append(min(candidates, key=lambda candidate: candidate[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=len(reference),
    )


def _word_units(text: str) -> list[str]:
    """Split preprocessed text on ASCII spaces only."""
    return text.split(" ") if text else []


def score_transcript(
    reference: str,
    hypothesis: str,
    *,
    normalization_profile: str,
) -> TranscriptScore:
    """Score one transcript pair under the requested normalization profile."""
    if normalization_profile not in {STRICT_PROFILE, EN_PROFILE}:
        raise ValueError(f"Unsupported normalization profile: {normalization_profile}")

    exact_match = normalize_exact_text(reference) == normalize_exact_text(hypothesis)
    strict_reference = normalize_strict_v1(reference)
    strict_hypothesis = normalize_strict_v1(hypothesis)
    if normalization_profile == STRICT_PROFILE:
        normalized_reference = strict_reference
        normalized_hypothesis = strict_hypothesis
    else:
        normalized_reference = normalize_en_v1(reference)
        normalized_hypothesis = normalize_en_v1(hypothesis)

    return TranscriptScore(
        exact_match=exact_match,
        strict_wer=edit_counts(
            _word_units(strict_reference),
            _word_units(strict_hypothesis),
        ),
        strict_cer=edit_counts(strict_reference, strict_hypothesis),
        normalized_wer=edit_counts(
            _word_units(normalized_reference),
            _word_units(normalized_hypothesis),
        ),
        normalized_cer=edit_counts(normalized_reference, normalized_hypothesis),
    )


def percentile_type7(values: Sequence[float], p: float) -> float | None:
    """Return the type-7 percentile of finite observations."""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise TypeError("percentile p must be numeric")
    if not 0.0 <= p <= 1.0 or not math.isfinite(p):
        raise ValueError("percentile p must be finite and within [0, 1]")

    finite_values: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("percentile observations must be numeric")
        try:
            numeric_value = float(value)
        except OverflowError as exc:
            raise ValueError("percentile observations must be finite") from exc
        if not math.isfinite(numeric_value):
            raise ValueError("percentile observations must be finite")
        finite_values.append(numeric_value)
    if not finite_values:
        return None

    finite_values.sort()
    index = (len(finite_values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return finite_values[lower] + (
        finite_values[upper] - finite_values[lower]
    ) * fraction
