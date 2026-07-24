"""Deterministic normalization and scoring for native STT benchmarks."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import subprocess  # nosec B404
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCORER_VERSION = "stt-score-v1"
STRICT_PROFILE = "strict-v1"
EN_PROFILE = "en-v1"
BCP47_BASIC_V1 = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
STABLE_ID_V1 = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
SERIALIZED_ID_V1 = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,255}")
SAFE_MODEL_LABEL_V1 = re.compile(
    r"(?:external:[A-Za-z0-9][A-Za-z0-9._+-]{0,255}|"
    r"[A-Za-z0-9][A-Za-z0-9._+-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._+-]*)?)"
)
MAX_TAGS_PER_SAMPLE = 32
RUN_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
RESULT_STATUSES = frozenset(
    {
        "ok",
        "empty",
        "adapter_error",
        "timeout",
        "worker_crash",
        "interrupted",
        "invalid_artifact",
    }
)
MEASUREMENT_ROLES = frozenset({"accuracy", "performance_repeat"})
TIMING_CLASSES = frozenset({"cold_first", "warmup_recovery", "warm"})

_KNOWN_SAMPLE_PROFILES = frozenset({"comparison", "regression"})
_KNOWN_NORMALIZATION_PROFILES = frozenset({STRICT_PROFILE, EN_PROFILE})
_SOURCE_REQUIRED_FIELDS = frozenset({"dataset", "version", "license", "reference_provenance", "sha256"})
_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "id",
        "audio",
        "reference",
        "language",
        "normalization_profile",
        "profiles",
        "suite",
        "suite_visibility",
        "annotation_profile",
        "diagnostic_only",
        "source",
        "tags",
    }
)
_MANIFEST_FIELDS = _MANIFEST_REQUIRED_FIELDS | {"duration_seconds"}
_SHA256_V1 = re.compile(r"[0-9a-f]{64}")
_MAX_SOURCE_FIELDS = 32
_MAX_SOURCE_VALUE_LENGTH = 4096
_RESULT_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "target_id",
        "completion_key",
        "sample_id",
        "repetition",
        "attempt_id",
        "worker_attempt_id",
        "measurement_role",
        "timing_class",
        "suite",
        "suite_visibility",
        "dataset",
        "tags",
        "diagnostic_only",
        "requested_execution",
        "actual_execution",
        "execution_mismatch_reasons",
        "eligibility_reasons",
        "status",
        "reference",
        "hypothesis",
        "scorer_version",
        "strict_profile",
        "normalization_profile",
        "exact_match",
        "strict",
        "normalized",
        "adapter_nanoseconds",
        "audio_duration_seconds",
        "rtf",
        "throughput",
        "resource_observations",
        "error",
    }
)
_ACTUAL_EXECUTION_FIELDS = frozenset(
    {
        "route_id",
        "provider",
        "model_label",
        "artifact_id",
        "backend",
        "audio_egress",
        "endpoint_id",
        "source",
        "device",
        "compute_type",
        "dtype",
        "decoding_ids",
        "transport",
    }
)
_RESOURCE_OBSERVATION_FIELDS = frozenset(
    {
        "collection_method",
        "gpu_memory_bytes",
        "peak_rss_bytes",
        "rss_after_bytes",
        "rss_before_bytes",
    }
)

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


@dataclass(frozen=True)
class ManifestSample:
    """One validated, portable benchmark sample."""

    sample_id: str
    audio_relative: str
    reference: str
    language: str
    normalization_profile: str
    measured_duration_seconds: float
    profiles: tuple[str, ...]
    suite: str
    suite_visibility: str
    annotation_profile: str
    diagnostic_only: bool
    source: tuple[tuple[str, str], ...]
    tags: tuple[str, ...]
    sha256: str


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
    return finite_values[lower] + (finite_values[upper] - finite_values[lower]) * fraction


def _positive_finite_number(
    value: object,
    *,
    allow_string: bool = False,
) -> float | None:
    """Return a positive finite float or None."""
    accepted_types = (int, float, str) if allow_string else (int, float)
    if isinstance(value, bool) or not isinstance(value, accepted_types):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if number > 0.0 and math.isfinite(number) else None


def probe_audio_duration_ffprobe(audio_path: Path) -> float:
    """Measure the first audio stream duration through the pinned ffprobe path."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration:format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    try:
        # The executable/options are fixed; only the validated local file is appended.
        result = subprocess.run(  # nosec B603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("ffprobe could not measure audio duration") from exc
    if result.returncode != 0:
        raise ValueError("ffprobe failed to measure audio duration")
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("ffprobe returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("ffprobe returned malformed JSON")

    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        raise ValueError("ffprobe returned malformed stream metadata")
    for stream in streams:
        if not isinstance(stream, dict):
            raise ValueError("ffprobe returned malformed stream metadata")
        duration = _positive_finite_number(
            stream.get("duration"),
            allow_string=True,
        )
        if duration is not None:
            return duration

    format_metadata = payload.get("format", {})
    if not isinstance(format_metadata, dict):
        raise ValueError("ffprobe returned malformed format metadata")
    duration = _positive_finite_number(
        format_metadata.get("duration"),
        allow_string=True,
    )
    if duration is not None:
        return duration
    raise ValueError("ffprobe returned no positive finite audio duration")


def _manifest_error(sample_id: str, field: str, detail: str) -> ValueError:
    """Build a reference-safe manifest error."""
    return ValueError(f"sample {sample_id} field {field}: {detail}")


def _require_utf8_scalar_text(value: str, sample_id: str, field: str) -> str:
    """Reject lone surrogates that cannot form portable UTF-8 manifest text."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _manifest_error(
            sample_id,
            field,
            "must contain valid Unicode scalar text",
        ) from exc
    return value


def _require_stable_id(value: object, sample_id: str, field: str) -> str:
    """Validate and return one stable lower-case identifier."""
    if not isinstance(value, str) or STABLE_ID_V1.fullmatch(value) is None:
        raise _manifest_error(sample_id, field, "must be a stable identifier")
    return value


def _canonical_id_list(
    value: object,
    *,
    sample_id: str,
    field: str,
    maximum: int | None = None,
    known: frozenset[str] | None = None,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    """Validate, deduplicate, and sort a manifest identifier list."""
    if not isinstance(value, list):
        raise _manifest_error(sample_id, field, "must be an array")
    if require_nonempty and not value:
        raise _manifest_error(sample_id, field, "must not be empty")
    if maximum is not None and len(value) > maximum:
        raise _manifest_error(sample_id, field, f"must contain at most {maximum} items")
    identifiers = tuple(_require_stable_id(item, sample_id, field) for item in value)
    if len(set(identifiers)) != len(identifiers):
        raise _manifest_error(sample_id, field, "must contain unique items")
    if known is not None and not set(identifiers) <= known:
        raise _manifest_error(sample_id, field, "contains an unknown identifier")
    return tuple(sorted(identifiers))


def _resolve_audio_path(root: Path, relative: object, sample_id: str) -> tuple[str, Path]:
    """Resolve one manifest-relative regular file without permitting escape."""
    if not isinstance(relative, str) or not relative:
        raise _manifest_error(sample_id, "audio", "must be a relative path")
    relative = _require_utf8_scalar_text(relative, sample_id, "audio")
    if "\\" in relative or relative.startswith("//") or re.match(r"^[A-Za-z]:", relative):
        raise _manifest_error(sample_id, "audio", "must use a relative POSIX path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise _manifest_error(sample_id, "audio", "contains an unsafe path segment")
    portable_path = PurePosixPath(relative)
    if portable_path.is_absolute():
        raise _manifest_error(sample_id, "audio", "must be a relative path")
    try:
        candidate = (root / Path(*portable_path.parts)).resolve(strict=True)
        candidate.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise _manifest_error(
            sample_id,
            "audio",
            "does not resolve to a contained file",
        ) from exc
    if not candidate.is_file():
        raise _manifest_error(sample_id, "audio", "must resolve to a regular file")
    return portable_path.as_posix(), candidate


def _sha256_file(path: Path) -> str:
    """Stream a file into a SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio_for_scheduling(
    sample: ManifestSample,
    dataset_root: Path,
) -> Path:
    """Revalidate and pin the resolved audio path immediately before scheduling.

    Coordinators must pass the returned resolved path to the worker rather than
    resolving ``sample.audio_relative`` again.
    """
    if not isinstance(sample, ManifestSample):
        raise TypeError("sample must be a ManifestSample")
    sample_id = _require_stable_id(sample.sample_id, "<sample>", "id")
    try:
        root = Path(dataset_root).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise _manifest_error(
            sample_id,
            "audio",
            "dataset root does not exist",
        ) from exc
    if not root.is_dir():
        raise _manifest_error(sample_id, "audio", "dataset root must be a directory")
    relative, audio_path = _resolve_audio_path(
        root,
        sample.audio_relative,
        sample_id,
    )
    if relative != sample.audio_relative:
        raise _manifest_error(sample_id, "audio", "relative path changed")
    try:
        audio_sha256 = _sha256_file(audio_path)
    except OSError as exc:
        raise _manifest_error(
            sample_id,
            "source.sha256",
            "audio file could not be read before scheduling",
        ) from exc
    if audio_sha256 != sample.sha256:
        raise _manifest_error(
            sample_id,
            "source.sha256",
            "audio changed since manifest validation",
        )
    return audio_path


def _validate_source(
    value: object,
    *,
    sample_id: str,
) -> tuple[tuple[tuple[str, str], ...], str, dict[str, str]]:
    """Validate provenance metadata and split its audio digest from source fields."""
    if not isinstance(value, dict):
        raise _manifest_error(sample_id, "source", "must be an object")
    if not value.keys() >= _SOURCE_REQUIRED_FIELDS:
        raise _manifest_error(sample_id, "source", "is missing a required field")
    if len(value) > _MAX_SOURCE_FIELDS:
        raise _manifest_error(sample_id, "source", "contains too many fields")
    canonical: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or STABLE_ID_V1.fullmatch(key) is None:
            raise _manifest_error(sample_id, "source", "contains an invalid field name")
        if not isinstance(item, str) or not item.strip() or len(item) > _MAX_SOURCE_VALUE_LENGTH:
            raise _manifest_error(sample_id, f"source.{key}", "must be a bounded string")
        canonical[key] = _require_utf8_scalar_text(
            item,
            sample_id,
            f"source.{key}",
        )
    audio_sha256 = canonical.pop("sha256")
    if _SHA256_V1.fullmatch(audio_sha256) is None:
        raise _manifest_error(sample_id, "source.sha256", "must be a lower-case SHA-256")
    source = tuple(sorted(canonical.items()))
    return source, audio_sha256, dict(sorted(value.items()))


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _parse_manifest_lines(manifest_path: Path) -> list[tuple[int, dict[str, object]]]:
    """Parse JSONL records without accepting blank or non-object lines."""
    records: list[tuple[int, dict[str, object]]] = []
    try:
        with manifest_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"manifest line {line_number}: blank lines are invalid")
                try:
                    record = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(f"manifest line {line_number}: malformed JSON") from exc
                except ValueError as exc:
                    raise ValueError(f"manifest line {line_number}: duplicate JSON field") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"manifest line {line_number}: record must be an object")
                records.append((line_number, record))
    except (OSError, UnicodeError) as exc:
        raise ValueError("manifest could not be read as UTF-8 JSONL") from exc
    if not records:
        raise ValueError("manifest must contain at least one sample")
    return records


def load_and_validate_manifest(
    manifest_path: Path,
    dataset_root: Path,
    *,
    duration_probe: Callable[[Path], float] = probe_audio_duration_ffprobe,
) -> tuple[tuple[ManifestSample, ...], str]:
    """Validate a benchmark JSONL manifest and return its portable identity."""
    try:
        resolved_manifest = Path(manifest_path).resolve(strict=True)
        root = Path(dataset_root).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("manifest or dataset root does not exist") from exc
    if not resolved_manifest.is_file():
        raise ValueError("manifest must be a regular file")
    if not root.is_dir():
        raise ValueError("dataset root must be a directory")

    samples: list[ManifestSample] = []
    canonical_records: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    suite_visibility: dict[str, str] = {}
    for line_number, record in _parse_manifest_lines(resolved_manifest):
        raw_id = record.get("id")
        error_id = raw_id if isinstance(raw_id, str) and STABLE_ID_V1.fullmatch(raw_id) else f"<line-{line_number}>"
        unknown = set(record) - _MANIFEST_FIELDS
        missing = _MANIFEST_REQUIRED_FIELDS - record.keys()
        if unknown:
            raise _manifest_error(error_id, "record", "contains an unknown field")
        if missing:
            raise _manifest_error(error_id, "record", "is missing a required field")

        sample_id = _require_stable_id(raw_id, error_id, "id")
        if sample_id in sample_ids:
            raise _manifest_error(sample_id, "id", "must be unique")
        sample_ids.add(sample_id)

        reference = record["reference"]
        if not isinstance(reference, str) or not reference.strip():
            raise _manifest_error(sample_id, "reference", "must not be empty")
        reference = _require_utf8_scalar_text(reference, sample_id, "reference")
        language = record["language"]
        if not isinstance(language, str) or BCP47_BASIC_V1.fullmatch(language) is None:
            raise _manifest_error(sample_id, "language", "must match bcp47-basic-v1")
        language = language.lower()
        normalization_profile = record["normalization_profile"]
        if not isinstance(normalization_profile, str) or normalization_profile not in _KNOWN_NORMALIZATION_PROFILES:
            raise _manifest_error(
                sample_id,
                "normalization_profile",
                "is not supported",
            )
        if normalization_profile == EN_PROFILE and language.split("-", 1)[0] != "en":
            raise _manifest_error(
                sample_id,
                "normalization_profile",
                "en-v1 requires an English language tag",
            )
        normalized_reference = (
            normalize_en_v1(reference) if normalization_profile == EN_PROFILE else normalize_strict_v1(reference)
        )
        if not normalized_reference:
            raise _manifest_error(
                sample_id,
                "reference",
                "is empty after normalization",
            )

        profiles = _canonical_id_list(
            record["profiles"],
            sample_id=sample_id,
            field="profiles",
            known=_KNOWN_SAMPLE_PROFILES,
            require_nonempty=True,
        )
        tags = _canonical_id_list(
            record["tags"],
            sample_id=sample_id,
            field="tags",
            maximum=MAX_TAGS_PER_SAMPLE,
        )
        suite = _require_stable_id(record["suite"], sample_id, "suite")
        annotation_profile = _require_stable_id(
            record["annotation_profile"],
            sample_id,
            "annotation_profile",
        )
        visibility = record["suite_visibility"]
        if not isinstance(visibility, str) or visibility not in {"public", "private"}:
            raise _manifest_error(
                sample_id,
                "suite_visibility",
                "must be public or private",
            )
        if suite in suite_visibility and suite_visibility[suite] != visibility:
            raise _manifest_error(
                sample_id,
                "suite_visibility",
                "must be consistent within a suite",
            )
        suite_visibility[suite] = visibility
        diagnostic_only = record["diagnostic_only"]
        if not isinstance(diagnostic_only, bool):
            raise _manifest_error(sample_id, "diagnostic_only", "must be boolean")

        source, declared_sha256, canonical_source = _validate_source(
            record["source"],
            sample_id=sample_id,
        )
        audio_relative, audio_path = _resolve_audio_path(
            root,
            record["audio"],
            sample_id,
        )
        try:
            initial_sha256 = _sha256_file(audio_path)
        except OSError as exc:
            raise _manifest_error(
                sample_id,
                "source.sha256",
                "audio file could not be read",
            ) from exc
        if initial_sha256 != declared_sha256:
            raise _manifest_error(
                sample_id,
                "source.sha256",
                "does not match the audio file",
            )
        try:
            measured_duration = duration_probe(audio_path)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            raise _manifest_error(
                sample_id,
                "audio.duration",
                "could not be measured",
            ) from exc
        measured_duration = _positive_finite_number(measured_duration)
        if measured_duration is None:
            raise _manifest_error(
                sample_id,
                "audio.duration",
                "must be positive and finite",
            )

        declared_duration: float | None = None
        if "duration_seconds" in record:
            declared_duration = _positive_finite_number(record["duration_seconds"])
            if declared_duration is None:
                raise _manifest_error(
                    sample_id,
                    "duration_seconds",
                    "must be positive and finite",
                )
            tolerance = max(0.100, measured_duration * 0.01)
            rounding_slack = math.ulp(declared_duration) + math.ulp(measured_duration) + math.ulp(tolerance)
            if abs(declared_duration - measured_duration) > tolerance + rounding_slack:
                raise _manifest_error(
                    sample_id,
                    "duration_seconds",
                    "does not agree with measured duration",
                )

        sample = ManifestSample(
            sample_id=sample_id,
            audio_relative=audio_relative,
            reference=reference,
            language=language,
            normalization_profile=normalization_profile,
            measured_duration_seconds=measured_duration,
            profiles=profiles,
            suite=suite,
            suite_visibility=visibility,
            annotation_profile=annotation_profile,
            diagnostic_only=diagnostic_only,
            source=source,
            tags=tags,
            sha256=declared_sha256,
        )
        resolve_audio_for_scheduling(sample, root)
        samples.append(sample)
        canonical_record: dict[str, object] = {
            "id": sample_id,
            "audio": audio_relative,
            "reference": reference,
            "language": language,
            "normalization_profile": normalization_profile,
            "profiles": profiles,
            "suite": suite,
            "suite_visibility": visibility,
            "annotation_profile": annotation_profile,
            "diagnostic_only": diagnostic_only,
            "source": canonical_source,
            "tags": tags,
        }
        if declared_duration is not None:
            canonical_record["duration_seconds"] = declared_duration
        canonical_records.append(canonical_record)

    samples.sort(key=lambda sample: sample.sample_id)
    canonical_records.sort(key=lambda record: str(record["id"]))
    canonical_json = json.dumps(
        canonical_records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return tuple(samples), hashlib.sha256(canonical_json).hexdigest()


def select_samples(
    samples: Sequence[ManifestSample],
    *,
    profile: str,
    seed: int,
) -> tuple[tuple[ManifestSample, ...], str]:
    """Select and order one known profile, returning its shared cold probe."""
    if not isinstance(profile, str) or STABLE_ID_V1.fullmatch(profile) is None:
        raise ValueError("profile must be a stable identifier")
    if profile not in _KNOWN_SAMPLE_PROFILES:
        raise ValueError("profile is not supported")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    selected = [sample for sample in samples if profile in sample.profiles]
    if not selected:
        raise ValueError("no samples match the requested profile")
    selected.sort(key=lambda sample: hashlib.sha256(f"{seed}\0{sample.sample_id}".encode()).digest())
    return tuple(selected), selected[0].sample_id


def completion_key(
    manifest_hash: str,
    target_id: str,
    execution_contract_hash: str,
    sample_id: str,
    repetition: int,
) -> str:
    """Return the stable identity for one target/sample/repetition result."""
    for value, field in (
        (manifest_hash, "manifest_hash"),
        (execution_contract_hash, "execution_contract_hash"),
    ):
        if not isinstance(value, str) or _SHA256_V1.fullmatch(value) is None:
            raise ValueError(f"{field} must be a lower-case SHA-256")
    _require_stable_id(target_id, "<completion>", "target_id")
    _require_stable_id(sample_id, "<completion>", "sample_id")
    if isinstance(repetition, bool) or not isinstance(repetition, int):
        raise TypeError("repetition must be an integer")
    if repetition < 0:
        raise ValueError("repetition must be non-negative")
    identity = json.dumps(
        [
            "stt-completion-v1",
            manifest_hash,
            target_id,
            execution_contract_hash,
            sample_id,
            repetition,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(identity).hexdigest()


def _ensure_owner_directory(path: Path) -> None:
    """Create an owner-only run directory or reject an unsafe existing one."""
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        created = True
    except FileExistsError:
        created = False
    if os.name != "posix":
        return
    if created:
        os.chmod(path, 0o700)
        return
    if path.stat().st_mode & 0o077:
        raise PermissionError("artifact parent directory must already be owner-only")


def _fsync_directory(path: Path) -> bool:
    """Fsync one directory when the local platform supports it."""
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | directory_flag)
    except OSError as exc:
        unsupported = {
            errno.EACCES,
            errno.EBADF,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno in unsupported:
            return False
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            unsupported = {
                errno.EBADF,
                errno.EINVAL,
                errno.EROFS,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if exc.errno in unsupported:
                return False
            raise
    finally:
        os.close(descriptor)
    return True


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize one artifact mapping to strict canonical UTF-8 JSON."""
    if not isinstance(payload, Mapping):
        raise TypeError("JSON artifact payload must be a mapping")
    try:
        text = json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (text + "\n").encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("JSON artifact payload is not safely serializable") from exc


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace one owner-only JSON artifact and sync its directory."""
    destination = Path(path)
    _ensure_owner_directory(destination.parent)
    encoded = _canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if os.name == "posix":
            os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        if os.name == "posix":
            os.chmod(destination, 0o600)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_result_integer(
    value: object,
    *,
    field: str,
    minimum: int,
) -> int:
    """Return one bounded result integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"result field {field} must be an integer")
    if value < minimum:
        raise ValueError(f"result field {field} is out of range")
    return value


def _require_result_text(
    value: object,
    *,
    field: str,
    maximum: int = 4096,
    allow_empty: bool = False,
) -> str:
    """Return one bounded, portable result string."""
    if not isinstance(value, str) or (not allow_empty and not value) or len(value) > maximum:
        raise ValueError(f"result field {field} must be a bounded string")
    return _require_utf8_scalar_text(value, "<result>", field)


def _validate_reason_list(value: object, field: str) -> None:
    """Validate a bounded ordered list of non-sensitive reason strings."""
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError(f"result field {field} must be a bounded array")
    for item in value:
        _require_result_text(item, field=field, maximum=256)


def _validate_execution_mapping(
    value: object,
    *,
    field: str,
    actual: bool,
) -> None:
    """Validate one explicitly allowlisted execution summary."""
    if not isinstance(value, dict):
        raise ValueError(f"result field {field} must be an object")
    allowed = _ACTUAL_EXECUTION_FIELDS if actual else {"provider", "model_label"}
    if set(value) != set(allowed):
        raise ValueError(f"result field {field} has invalid execution fields")
    for key in ("provider", "model_label"):
        item = _require_result_text(
            value[key],
            field=f"{field}.{key}",
            maximum=256,
        )
        pattern = SAFE_MODEL_LABEL_V1 if key == "model_label" else SERIALIZED_ID_V1
        if pattern.fullmatch(item) is None:
            raise ValueError(f"result field {field}.{key} is not a safe label")
    if not actual:
        return
    decoding_ids = value["decoding_ids"]
    if not isinstance(decoding_ids, list):
        raise ValueError(f"result field {field}.decoding_ids is invalid")
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
            SttActualExecution,
            SttAudioEgress,
        )

        canonical = SttActualExecution(
            route_id=value["route_id"],
            provider=value["provider"],
            model_label=value["model_label"],
            artifact_id=value["artifact_id"],
            backend=value["backend"],
            audio_egress=SttAudioEgress(value["audio_egress"]),
            endpoint_id=value["endpoint_id"],
            source=value["source"],
            device=value["device"],
            compute_type=value["compute_type"],
            dtype=value["dtype"],
            decoding_ids=tuple(decoding_ids),
            transport=value["transport"],
        ).as_safe_dict()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"result field {field} is not canonical") from exc
    if canonical != value:
        raise ValueError(f"result field {field} is not canonical")


def _validate_edit_payload(value: object, field: str) -> None:
    """Validate persisted edit counts and their redundant derived values."""
    expected = {
        "substitutions",
        "deletions",
        "insertions",
        "reference_units",
        "errors",
        "rate",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"result field {field} must contain edit counts")
    counts = [
        _require_result_integer(value[name], field=f"{field}.{name}", minimum=0)
        for name in (
            "substitutions",
            "deletions",
            "insertions",
            "reference_units",
        )
    ]
    errors = _require_result_integer(
        value["errors"],
        field=f"{field}.errors",
        minimum=0,
    )
    if errors != sum(counts[:3]):
        raise ValueError(f"result field {field}.errors is inconsistent")
    rate = value["rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError(f"result field {field}.rate must be numeric")
    rate_float = float(rate)
    expected_rate = errors / max(counts[3], 1)
    if not math.isfinite(rate_float) or not math.isclose(
        rate_float,
        expected_rate,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"result field {field}.rate is inconsistent")


def _validate_score_mapping(value: object, field: str) -> None:
    """Validate persisted WER and CER dictionaries."""
    if not isinstance(value, dict) or set(value) != {"wer", "cer"}:
        raise ValueError(f"result field {field} must contain wer and cer")
    _validate_edit_payload(value["wer"], f"{field}.wer")
    _validate_edit_payload(value["cer"], f"{field}.cer")


def _validate_optional_positive_number(value: object, field: str) -> None:
    """Validate a finite positive performance number or None."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"result field {field} must be numeric or null")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"result field {field} must be positive and finite")


def _score_as_result_fields(score: TranscriptScore) -> dict[str, object]:
    """Return the persisted score projection used for integrity checks."""

    def edit_payload(counts: EditCounts) -> dict[str, int | float]:
        return {
            "substitutions": counts.substitutions,
            "deletions": counts.deletions,
            "insertions": counts.insertions,
            "reference_units": counts.reference_units,
            "errors": counts.errors,
            "rate": counts.rate,
        }

    return {
        "exact_match": score.exact_match,
        "strict": {
            "wer": edit_payload(score.strict_wer),
            "cer": edit_payload(score.strict_cer),
        },
        "normalized": {
            "wer": edit_payload(score.normalized_wer),
            "cer": edit_payload(score.normalized_cer),
        },
    }


def _validate_result_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Validate and shallow-copy one complete result record."""
    if not isinstance(record, Mapping):
        raise ValueError("result record must be an object")
    result = dict(record)
    if set(result) != _RESULT_REQUIRED_FIELDS:
        raise ValueError("result record has missing or unknown fields")
    if result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported result schema version")
    for field in ("run_id", "target_id", "sample_id", "suite", "dataset"):
        _require_stable_id(result[field], "<result>", field)
    if not isinstance(result["completion_key"], str) or _SHA256_V1.fullmatch(result["completion_key"]) is None:
        raise ValueError("result field completion_key must be a SHA-256")
    _require_result_integer(result["repetition"], field="repetition", minimum=0)
    _require_result_integer(result["attempt_id"], field="attempt_id", minimum=1)
    _require_result_integer(
        result["worker_attempt_id"],
        field="worker_attempt_id",
        minimum=1,
    )
    if result["measurement_role"] not in MEASUREMENT_ROLES:
        raise ValueError("unknown result measurement role")
    if result["timing_class"] not in TIMING_CLASSES:
        raise ValueError("unknown result timing class")
    if result["suite_visibility"] not in {"public", "private"}:
        raise ValueError("unknown result suite visibility")
    if not isinstance(result["diagnostic_only"], bool):
        raise ValueError("result field diagnostic_only must be boolean")
    tags = result["tags"]
    if not isinstance(tags, list) or len(tags) > MAX_TAGS_PER_SAMPLE or len(tags) != len(set(tags)):
        raise ValueError("result field tags must be a bounded unique array")
    for tag in tags:
        _require_stable_id(tag, "<result>", "tags")
    _validate_execution_mapping(
        result["requested_execution"],
        field="requested_execution",
        actual=False,
    )
    _validate_execution_mapping(
        result["actual_execution"],
        field="actual_execution",
        actual=True,
    )
    _validate_reason_list(
        result["execution_mismatch_reasons"],
        "execution_mismatch_reasons",
    )
    _validate_reason_list(result["eligibility_reasons"], "eligibility_reasons")
    if result["status"] not in RESULT_STATUSES:
        raise ValueError("unknown result status")
    reference = result["reference"]
    hypothesis = result["hypothesis"]
    if (reference is None) != (hypothesis is None):
        raise ValueError("result reference and hypothesis retention must match")
    if reference is not None:
        _require_result_text(
            reference,
            field="reference",
            maximum=1_000_000,
        )
        _require_result_text(
            hypothesis,
            field="hypothesis",
            maximum=1_000_000,
            allow_empty=True,
        )
        hypothesis_is_empty = not normalize_strict_v1(hypothesis)
        if result["status"] == "ok" and hypothesis_is_empty:
            raise ValueError("result ok status cannot carry an empty hypothesis")
        if result["status"] == "empty" and not hypothesis_is_empty:
            raise ValueError("result empty status requires an empty hypothesis")
    if result["scorer_version"] != SCORER_VERSION:
        raise ValueError("unsupported scorer version")
    if result["strict_profile"] != STRICT_PROFILE:
        raise ValueError("unsupported strict profile")
    if result["normalization_profile"] not in _KNOWN_NORMALIZATION_PROFILES:
        raise ValueError("unsupported normalization profile")
    if not isinstance(result["exact_match"], bool):
        raise ValueError("result field exact_match must be boolean")
    _validate_score_mapping(result["strict"], "strict")
    _validate_score_mapping(result["normalized"], "normalized")
    if reference is not None and hypothesis is not None:
        expected_score = _score_as_result_fields(
            score_result_text(
                reference,
                hypothesis,
                status=str(result["status"]),
                normalization_profile=str(result["normalization_profile"]),
            )
        )
        for field in ("exact_match", "strict", "normalized"):
            if result[field] != expected_score[field]:
                raise ValueError("result score is inconsistent with retained text")
    adapter_nanoseconds = result["adapter_nanoseconds"]
    if adapter_nanoseconds is not None and (
        isinstance(adapter_nanoseconds, bool) or not isinstance(adapter_nanoseconds, int)
    ):
        raise ValueError("result field adapter_nanoseconds must be an integer or null")
    audio_duration = result["audio_duration_seconds"]
    if audio_duration is not None and (
        isinstance(audio_duration, bool)
        or not isinstance(audio_duration, (int, float))
        or not math.isfinite(float(audio_duration))
    ):
        raise ValueError("result field audio_duration_seconds must be finite numeric or null")
    _validate_optional_positive_number(result["rtf"], "rtf")
    _validate_optional_positive_number(result["throughput"], "throughput")
    base_reasons = [reason for reason in result["eligibility_reasons"] if reason != "invalid_performance_duration"]
    expected_rtf, expected_throughput, expected_reasons = performance_fields(
        adapter_nanoseconds,
        audio_duration,
        eligibility_reasons=base_reasons,
    )
    if result["eligibility_reasons"] != expected_reasons:
        raise ValueError("result performance eligibility is inconsistent")
    for field, actual, expected in (
        ("rtf", result["rtf"], expected_rtf),
        ("throughput", result["throughput"], expected_throughput),
    ):
        if (actual is None) != (expected is None):
            raise ValueError(f"result performance field {field} is inconsistent")
        if (
            actual is not None
            and expected is not None
            and not math.isclose(
                float(actual),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"result performance field {field} is inconsistent")
    observations = result["resource_observations"]
    if observations is not None:
        if (
            not isinstance(observations, dict)
            or set(observations) - _RESOURCE_OBSERVATION_FIELDS
            or "collection_method" not in observations
        ):
            raise ValueError("resource observations contain unknown fields")
        _require_result_text(
            observations["collection_method"],
            field="resource_observations.collection_method",
            maximum=128,
        )
        for field, value in observations.items():
            if field == "collection_method" or value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"resource observation {field} must be non-negative bytes")
    error = result["error"]
    if error is not None:
        if not isinstance(error, dict) or set(error) != {"type", "message"}:
            raise ValueError("result error must contain type and message")
        _require_result_text(error["type"], field="error.type", maximum=128)
        _require_result_text(error["message"], field="error.message", maximum=512)
    return result


def append_result_record(path: Path, record: Mapping[str, object]) -> None:
    """Append, flush, and fsync one validated owner-only result record."""
    validated = _validate_result_record(record)
    encoded = _canonical_json_bytes(validated)
    destination = Path(path)
    _ensure_owner_directory(destination.parent)
    descriptor = os.open(
        destination,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    with os.fdopen(descriptor, "ab") as output:
        if os.name == "posix":
            os.fchmod(output.fileno(), 0o600)
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    _fsync_directory(destination.parent)


def load_result_history(path: Path) -> tuple[list[dict[str, object]], bool]:
    """Load validated history and ignore only an unterminated final JSONL line."""
    source = Path(path)
    try:
        content = source.read_bytes()
    except FileNotFoundError:
        return [], False
    if not content:
        return [], False
    lines = content.splitlines(keepends=True)
    truncated = not content.endswith(b"\n")
    if truncated:
        lines = lines[:-1]
    records: list[dict[str, object]] = []
    previous_attempt = 0
    for line_number, raw_line in enumerate(lines, start=1):
        try:
            decoded = raw_line.decode("utf-8")
            parsed = json.loads(
                decoded,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            validated = _validate_result_record(parsed)
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"results line {line_number}: invalid record") from exc
        attempt_id = int(validated["attempt_id"])
        if attempt_id <= previous_attempt:
            raise ValueError(f"results line {line_number}: attempt IDs must increase")
        previous_attempt = attempt_id
        records.append(validated)
    return records, truncated


def reduce_attempts(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Select the highest globally monotonic attempt for each completion key."""
    active: dict[str, dict[str, object]] = {}
    previous_attempt = 0
    for index, record in enumerate(records, start=1):
        validated = _validate_result_record(record)
        attempt_id = int(validated["attempt_id"])
        if attempt_id <= previous_attempt:
            raise ValueError(f"record {index}: attempt IDs must increase")
        previous_attempt = attempt_id
        active[str(validated["completion_key"])] = validated
    return {key: active[key] for key in sorted(active)}


def validate_inflight_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Validate one transcript-free coordinator in-flight record."""
    fields = {
        "target_id",
        "operation_id",
        "operation_role",
        "worker_attempt_id",
        "sample_id",
        "completion_key",
        "repetition",
        "result_attempt_id",
    }
    if not isinstance(record, Mapping) or set(record) != fields:
        raise ValueError("in-flight record has missing or unknown fields")
    validated = dict(record)
    _require_stable_id(validated["target_id"], "<inflight>", "target_id")
    _require_stable_id(validated["sample_id"], "<inflight>", "sample_id")
    _require_result_integer(
        validated["operation_id"],
        field="operation_id",
        minimum=1,
    )
    _require_result_integer(
        validated["worker_attempt_id"],
        field="worker_attempt_id",
        minimum=1,
    )
    completion = validated["completion_key"]
    if not isinstance(completion, str) or _SHA256_V1.fullmatch(completion) is None:
        raise ValueError("in-flight completion_key must be a SHA-256")
    role = validated["operation_role"]
    if role not in {"result_call", "rewarm_probe"}:
        raise ValueError("unknown in-flight operation role")
    repetition = validated["repetition"]
    result_attempt_id = validated["result_attempt_id"]
    if role == "result_call":
        _require_result_integer(repetition, field="repetition", minimum=0)
        _require_result_integer(
            result_attempt_id,
            field="result_attempt_id",
            minimum=1,
        )
    else:
        if repetition is not None:
            _require_result_integer(repetition, field="repetition", minimum=0)
        if result_attempt_id is not None:
            raise ValueError("rewarm probe cannot allocate a result attempt")
    return validated


def allocate_inflight(
    run_metadata: Mapping[str, object],
    *,
    target_id: str,
    operation_role: str,
    worker_attempt_id: int,
    sample_id: str,
    completion_key: str,
    repetition: int | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Allocate coordinator-owned monotonic IDs for one adapter operation."""
    if not isinstance(run_metadata, Mapping):
        raise ValueError("run metadata must be an object")
    if run_metadata.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported run schema version")
    operation_id = _require_result_integer(
        run_metadata.get("next_operation_id"),
        field="next_operation_id",
        minimum=1,
    )
    attempt_id = _require_result_integer(
        run_metadata.get("next_attempt_id"),
        field="next_attempt_id",
        minimum=1,
    )
    result_attempt_id = attempt_id if operation_role == "result_call" else None
    inflight = validate_inflight_record(
        {
            "target_id": target_id,
            "operation_id": operation_id,
            "operation_role": operation_role,
            "worker_attempt_id": worker_attempt_id,
            "sample_id": sample_id,
            "completion_key": completion_key,
            "repetition": repetition,
            "result_attempt_id": result_attempt_id,
        }
    )
    updated = dict(run_metadata)
    updated["next_operation_id"] = operation_id + 1
    if operation_role == "result_call":
        updated["next_attempt_id"] = attempt_id + 1
    return updated, inflight


def resume_action(
    active_result: Mapping[str, object] | None,
    *,
    retry_errors: bool,
) -> str:
    """Return execute, skip, or retry for one completion key."""
    if not isinstance(retry_errors, bool):
        raise TypeError("retry_errors must be boolean")
    if active_result is None:
        return "execute"
    validated = _validate_result_record(active_result)
    if retry_errors and validated["status"] != "ok":
        return "retry"
    return "skip"


def recover_inflight_action(
    inflight: Mapping[str, object],
    terminal_records: Sequence[Mapping[str, object]],
    *,
    interrupted: bool = False,
    timed_out: bool = False,
) -> dict[str, str | None]:
    """Describe crash recovery without creating a replayable probe result."""
    if not isinstance(interrupted, bool) or not isinstance(timed_out, bool):
        raise TypeError("recovery flags must be boolean")
    if interrupted and timed_out:
        raise ValueError("recovery cannot be both interrupted and timed out")
    active = validate_inflight_record(inflight)
    status = "interrupted" if interrupted else "timeout" if timed_out else "worker_crash"
    if active["operation_role"] == "rewarm_probe":
        return {"action": "record_rewarm", "status": status}
    for record in terminal_records:
        validated = _validate_result_record(record)
        if (
            validated["completion_key"] == active["completion_key"]
            and validated["attempt_id"] == active["result_attempt_id"]
            and validated["target_id"] == active["target_id"]
            and validated["sample_id"] == active["sample_id"]
        ):
            return {"action": "clear", "status": None}
    return {"action": "append_result", "status": status}


def score_result_text(
    reference: str,
    hypothesis: str,
    *,
    status: str,
    normalization_profile: str,
) -> TranscriptScore:
    """Score failures as empty hypotheses so selective failure cannot help."""
    if status not in RESULT_STATUSES:
        raise ValueError("unknown result status")
    effective_hypothesis = hypothesis if status == "ok" else ""
    return score_transcript(
        reference,
        effective_hypothesis,
        normalization_profile=normalization_profile,
    )


def retain_text(
    *,
    mode: str,
    status: str,
    reference: str,
    hypothesis: str,
    score: TranscriptScore,
) -> tuple[str | None, str | None]:
    """Apply text retention after deterministic scoring."""
    if mode not in {"full", "errors-only", "none"}:
        raise ValueError("unknown text-retention mode")
    if status not in RESULT_STATUSES:
        raise ValueError("unknown result status")
    _require_text(reference)
    _require_text(hypothesis)
    if not isinstance(score, TranscriptScore):
        raise TypeError("score must be a TranscriptScore")
    has_error = any(
        counts.errors
        for counts in (
            score.strict_wer,
            score.strict_cer,
            score.normalized_wer,
            score.normalized_cer,
        )
    )
    if mode == "none" or (mode == "errors-only" and status == "ok" and not has_error):
        return None, None
    return reference, hypothesis


def performance_fields(
    adapter_nanoseconds: object,
    audio_duration_seconds: object,
    *,
    eligibility_reasons: Sequence[str],
) -> tuple[float | None, float | None, list[str]]:
    """Return RTF/throughput and disqualify invalid timing observations."""
    reasons = list(eligibility_reasons)
    for reason in reasons:
        _require_result_text(reason, field="eligibility_reasons", maximum=256)
    valid_adapter = (
        not isinstance(adapter_nanoseconds, bool) and isinstance(adapter_nanoseconds, int) and adapter_nanoseconds > 0
    )
    audio_duration = _positive_finite_number(audio_duration_seconds)
    if not valid_adapter or audio_duration is None:
        if "invalid_performance_duration" not in reasons:
            reasons.append("invalid_performance_duration")
        return None, None, reasons
    processing_seconds = adapter_nanoseconds / 1_000_000_000
    if not math.isfinite(processing_seconds) or processing_seconds <= 0.0:
        if "invalid_performance_duration" not in reasons:
            reasons.append("invalid_performance_duration")
        return None, None, reasons
    rtf = processing_seconds / audio_duration
    return rtf, audio_duration / processing_seconds, reasons


_URL_SECRET_V1 = re.compile(r"(?i)\b(?:https?|wss?)://[^\s<>'\"]+")
_AUTH_SECRET_V1 = re.compile(r"(?im)\bauthorization\b\s*[:=]\s*[^\r\n]*")
_NAMED_SECRET_V1 = re.compile(
    r"(?i)\b[a-z0-9_-]*(?:api[_-]?key|token|secret)"
    r"[a-z0-9_-]*\b\s*[:=]\s*[^\s,;]+"
)
_BEARER_SECRET_V1 = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_OPENAI_SECRET_V1 = re.compile(r"(?i)\bsk-[a-z0-9_-]+")
_QUOTED_POSIX_PATH_V1 = re.compile(r"'/(?:[^'\r\n]*)'|\"/(?:[^\"\r\n]*)\"")
_QUOTED_WINDOWS_PATH_V1 = re.compile(r"(?i)'[a-z]:\\(?:[^'\r\n]*)'|\"[a-z]:\\(?:[^\"\r\n]*)\"")
_WINDOWS_PATH_V1 = re.compile(r"(?i)(?<![a-z0-9_])[a-z]:\\[^\s]+")
_UNC_PATH_V1 = re.compile(r"\\\\[^\\\s]+\\[^\s]+")
_POSIX_PATH_V1 = re.compile(r"(?<![a-z0-9_])/(?:[^\s/]+/)*[^\s]*")
_WHITESPACE_CONTROL_V1 = re.compile(r"[\s\x00-\x1f\x7f]+")


def sanitize_error(exc: BaseException) -> dict[str, str]:
    """Return one bounded error type/message with common secrets removed."""
    message = str(exc)
    for pattern, replacement in (
        (_URL_SECRET_V1, "[REDACTED_URL]"),
        (_QUOTED_POSIX_PATH_V1, "[REDACTED_PATH]"),
        (_QUOTED_WINDOWS_PATH_V1, "[REDACTED_PATH]"),
        (_WINDOWS_PATH_V1, "[REDACTED_PATH]"),
        (_UNC_PATH_V1, "[REDACTED_PATH]"),
        (_POSIX_PATH_V1, "[REDACTED_PATH]"),
        (_AUTH_SECRET_V1, "authorization=[REDACTED]"),
        (_NAMED_SECRET_V1, "credential=[REDACTED]"),
        (_BEARER_SECRET_V1, "bearer [REDACTED]"),
        (_OPENAI_SECRET_V1, "[REDACTED_SECRET]"),
    ):
        message = pattern.sub(replacement, message)
    message = _WHITESPACE_CONTROL_V1.sub(" ", message).strip()[:512]
    error_type = re.sub(r"[^A-Za-z0-9_.-]", "_", type(exc).__name__)[:128]
    return {
        "type": error_type or "Exception",
        "message": message or "redacted error",
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    """Return deterministic mean and type-7 percentile statistics."""
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "mean": sum(values) / len(values),
        "p50": percentile_type7(values, 0.50),
        "p90": percentile_type7(values, 0.90),
        "p95": percentile_type7(values, 0.95),
        "p99": percentile_type7(values, 0.99),
    }


def _quality_edit_aggregate(
    records: Sequence[Mapping[str, object]],
    *,
    profile: str,
    unit: str,
) -> dict[str, float | None]:
    """Aggregate one stored edit-count family without reconstructing text."""
    rates: list[float] = []
    errors = 0
    reference_units = 0
    for record in records:
        profile_payload = record[profile]
        if not isinstance(profile_payload, Mapping):
            raise ValueError("validated score profile is not an object")
        edit_payload = profile_payload[unit]
        if not isinstance(edit_payload, Mapping):
            raise ValueError("validated edit payload is not an object")
        rates.append(float(edit_payload["rate"]))
        errors += int(edit_payload["errors"])
        reference_units += int(edit_payload["reference_units"])
    result = _distribution(rates)
    result["pooled"] = errors / max(reference_units, 1)
    result["errors"] = errors
    result["reference_units"] = reference_units
    return result


def _quality_aggregate(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build quality metrics for one non-empty, suite-scoped population."""
    count = len(records)
    success_count = sum(record["status"] == "ok" for record in records)
    empty_count = sum(record["status"] == "empty" for record in records)
    error_count = count - success_count - empty_count
    failure_count = count - success_count
    exact_count = sum(bool(record["exact_match"]) for record in records)
    visibilities = {str(record["suite_visibility"]) for record in records}
    if len(visibilities) != 1:
        raise ValueError("one suite cannot mix public and private visibility")
    return {
        "suite_visibility": next(iter(visibilities)),
        "sample_count": count,
        "success_count": success_count,
        "success_rate": success_count / count,
        "empty_count": empty_count,
        "empty_rate": empty_count / count,
        "failure_count": failure_count,
        "failure_rate": failure_count / count,
        "error_count": error_count,
        "error_rate": error_count / count,
        "exact_match_count": exact_count,
        "exact_match_rate": exact_count / count,
        "strict": {
            unit: _quality_edit_aggregate(
                records,
                profile="strict",
                unit=unit,
            )
            for unit in ("wer", "cer")
        },
        "normalized": {
            unit: _quality_edit_aggregate(
                records,
                profile="normalized",
                unit=unit,
            )
            for unit in ("wer", "cer")
        },
    }


def _suite_quality_aggregates(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Aggregate records independently by suite to prevent unsafe pooling."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["suite"]), []).append(record)
    return {suite: _quality_aggregate(grouped[suite]) for suite in sorted(grouped)}


def _target_quality_aggregates(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Aggregate quality by target first, then suite."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["target_id"]), []).append(record)
    return {target_id: {"suites": _suite_quality_aggregates(grouped[target_id])} for target_id in sorted(grouped)}


def _slice_quality_aggregates(
    records: Sequence[Mapping[str, object]],
    *,
    dimension: str,
) -> dict[str, object]:
    """Build target-first slices with every population suite-scoped."""
    grouped: dict[str, dict[str, list[Mapping[str, object]]]] = {}
    for record in records:
        target_id = str(record["target_id"])
        if dimension == "tag":
            values = [str(tag) for tag in record["tags"]]
        elif dimension == "actual_backend":
            execution = record["actual_execution"]
            if not isinstance(execution, Mapping):
                raise ValueError("validated actual execution is not an object")
            values = [str(execution["backend"])]
        else:
            values = [str(record[dimension])]
        for value in values:
            grouped.setdefault(target_id, {}).setdefault(value, []).append(record)
    return {
        target_id: {value: _suite_quality_aggregates(grouped[target_id][value]) for value in sorted(grouped[target_id])}
        for target_id in sorted(grouped)
    }


def _warm_performance_aggregate(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate valid warm timings and count ineligible successful calls."""
    adapter_seconds: list[float] = []
    rtfs: list[float] = []
    throughputs: list[float] = []
    ineligible_count = 0
    gate_eligible_count = 0
    for record in records:
        adapter_nanoseconds = record["adapter_nanoseconds"]
        rtf = record["rtf"]
        throughput = record["throughput"]
        if (
            not isinstance(adapter_nanoseconds, int)
            or isinstance(adapter_nanoseconds, bool)
            or rtf is None
            or throughput is None
        ):
            ineligible_count += 1
            continue
        adapter_seconds.append(adapter_nanoseconds / 1_000_000_000)
        rtfs.append(float(rtf))
        throughputs.append(float(throughput))
        if not record["eligibility_reasons"]:
            gate_eligible_count += 1
    return {
        "candidate_count": len(records),
        "observation_count": len(adapter_seconds),
        "ineligible_count": ineligible_count,
        "gate_eligible_count": gate_eligible_count,
        "adapter_seconds": _distribution(adapter_seconds),
        "rtf": _distribution(rtfs),
        "throughput": _distribution(throughputs),
    }


def _warm_performance_suites(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Aggregate successful warm calls independently by suite."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["suite"]), []).append(record)
    return {suite: _warm_performance_aggregate(grouped[suite]) for suite in sorted(grouped)}


def _target_warm_performance(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Aggregate warm performance by target first, then suite."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["target_id"]), []).append(record)
    return {target_id: {"suites": _warm_performance_suites(grouped[target_id])} for target_id in sorted(grouped)}


def _cold_first_observations(
    records: Sequence[Mapping[str, object]],
    *,
    cold_probe_sample_id: str | None,
) -> dict[str, dict[str, object]]:
    """Return the single scored cold-first observation for each target."""
    cold: dict[str, dict[str, object]] = {}
    for record in records:
        if record["timing_class"] != "cold_first":
            continue
        if cold_probe_sample_id is not None and record["sample_id"] != cold_probe_sample_id:
            raise ValueError("cold-first record does not match run probe")
        target_id = str(record["target_id"])
        if target_id in cold:
            raise ValueError("target has multiple active cold-first records")
        adapter_nanoseconds = record["adapter_nanoseconds"]
        cold[target_id] = {
            "sample_id": record["sample_id"],
            "status": record["status"],
            "adapter_seconds": (
                adapter_nanoseconds / 1_000_000_000
                if isinstance(adapter_nanoseconds, int)
                and not isinstance(adapter_nanoseconds, bool)
                and adapter_nanoseconds > 0
                else None
            ),
            "audio_duration_seconds": record["audio_duration_seconds"],
            "rtf": record["rtf"],
            "throughput": record["throughput"],
            "gate_eligible": (
                record["status"] == "ok" and record["rtf"] is not None and not record["eligibility_reasons"]
            ),
        }
    return {target_id: cold[target_id] for target_id in sorted(cold)}


def aggregate_results(
    run_metadata: Mapping[str, object],
    active_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build deterministic suite-isolated quality and performance summaries."""
    if not isinstance(run_metadata, Mapping):
        raise ValueError("run metadata must be an object")
    if run_metadata.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported run schema version")
    run_id = run_metadata.get("run_id")
    _require_stable_id(run_id, "<run>", "run_id")
    if not isinstance(active_results, Mapping):
        raise ValueError("active results must be an object")
    validated_records: list[dict[str, object]] = []
    for key, record in active_results.items():
        validated = _validate_result_record(record)
        if key != validated["completion_key"]:
            raise ValueError("active result key does not match completion key")
        if validated["run_id"] != run_id:
            raise ValueError("active result belongs to another run")
        validated_records.append(validated)
    validated_records.sort(
        key=lambda record: (
            str(record["target_id"]),
            int(record["attempt_id"]),
        )
    )
    quality_records = [record for record in validated_records if record["measurement_role"] == "accuracy"]
    primary_records = [record for record in quality_records if not record["diagnostic_only"]]
    diagnostic_records = [record for record in quality_records if record["diagnostic_only"]]
    warm_candidates = [
        record
        for record in validated_records
        if record["timing_class"] == "warm" and record["status"] == "ok" and not record["diagnostic_only"]
    ]
    cold_probe = run_metadata.get("cold_probe_sample_id")
    if cold_probe is not None:
        _require_stable_id(cold_probe, "<run>", "cold_probe_sample_id")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "active_result_count": len(validated_records),
        "primary": {"targets": _target_quality_aggregates(primary_records)},
        "diagnostic": {"targets": _target_quality_aggregates(diagnostic_records)},
        "slices": {
            "dataset": _slice_quality_aggregates(
                primary_records,
                dimension="dataset",
            ),
            "tag": _slice_quality_aggregates(
                primary_records,
                dimension="tag",
            ),
            "actual_backend": _slice_quality_aggregates(
                primary_records,
                dimension="actual_backend",
            ),
        },
        "performance": {
            "warm": {"targets": _target_warm_performance(warm_candidates)},
            "cold_first": _cold_first_observations(
                validated_records,
                cold_probe_sample_id=(str(cold_probe) if cold_probe is not None else None),
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a manifest and print only portable aggregate identity."""
    parser = argparse.ArgumentParser(prog="stt-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--dataset-root", required=True, type=Path)
    arguments = parser.parse_args(argv)

    try:
        samples, content_hash = load_and_validate_manifest(
            arguments.manifest,
            arguments.dataset_root,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("error: validation could not read a required file", file=sys.stderr)
        return 2
    profile_counts = Counter(profile for sample in samples for profile in sample.profiles)
    suite_counts = Counter(sample.suite for sample in samples)
    visibility_counts = Counter(sample.suite_visibility for sample in samples)
    summary = {
        "manifest_hash": content_hash,
        "profiles": dict(sorted(profile_counts.items())),
        "sample_count": len(samples),
        "suites": dict(sorted(suite_counts.items())),
        "visibility": dict(sorted(visibility_counts.items())),
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
