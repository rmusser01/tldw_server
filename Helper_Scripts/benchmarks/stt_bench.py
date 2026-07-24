"""Deterministic normalization and scoring for native STT benchmarks."""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import re
import stat
import subprocess  # nosec B404
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
        SttBatchExecutionPlan,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
        SttProviderAdapter,
    )

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
PRODUCTION_ADAPTER_FACTORY_PATH = "Helper_Scripts.benchmarks.stt_bench:_load_native_adapter"

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
_SAFE_TARGET_SETTING_FIELDS = frozenset(
    {
        "mode",
        "task",
        "language",
        "word_timestamps",
        "diarization",
        "prompt_present",
        "hotword_count",
        "configuration_id",
        "network_collection_profile",
        "network_client_location",
    }
)
_SAFE_TARGET_SETTING_REQUIRED_FIELDS = frozenset(
    {
        "mode",
        "task",
        "language",
        "word_timestamps",
        "diarization",
        "prompt_present",
        "hotword_count",
    }
)
_COMMON_TARGET_SETTING_FIELDS = frozenset(
    {
        "git_commit",
        "language",
        "task",
        "word_timestamps",
        "prompt",
        "hotwords",
        "diarization",
        "configuration_id",
        "network_collection_profile",
        "network_client_location",
    }
)
_FACTORY_PATH_V1 = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
    r"[A-Za-z_][A-Za-z0-9_]*"
)
_GIT_COMMIT_V1 = re.compile(r"(?:[0-9a-f]{40}|unknown)")
_UNSAFE_SAFE_SETTING_V1 = re.compile(
    r"(?:authorization|bearer|api[_-]?key|token|secret|sk-)",
    re.IGNORECASE,
)
_EXECUTION_CONTRACT_SOURCE_MODULES = frozenset(
    {
        "Helper_Scripts.benchmarks.stt_bench",
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract",
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


@dataclass(frozen=True)
class PreparedTarget:
    """One immutable, secret-safe target prepared before workers start."""

    target_id: str
    provider: str
    model_label: str
    plan: SttBatchExecutionPlan = dataclass_field(repr=False)
    adapter_factory_path: str = dataclass_field(repr=False)
    execution_contract_json: str
    execution_contract_hash: str


@dataclass(frozen=True)
class WorkerSettings:
    """Immutable run settings safe to send to a spawned target worker."""

    run_id: str
    results_path: str
    manifest_hash: str
    normalization_profile: str
    cold_probe_sample_id: str
    warm_repetitions: int
    timing_sample_ids: tuple[str, ...]
    text_retention: str
    retry_errors: bool
    worker_attempt_id: int
    audio_paths: tuple[str, ...] = dataclass_field(
        repr=False,
    )

    def __post_init__(self) -> None:
        """Reject invalid or mutable worker inputs before spawning."""
        _require_stable_id(self.run_id, "<worker>", "run_id")
        _require_stable_id(
            self.cold_probe_sample_id,
            "<worker>",
            "cold_probe_sample_id",
        )
        if _SHA256_V1.fullmatch(self.manifest_hash) is None:
            raise ValueError("worker manifest_hash must be a lower-case SHA-256")
        if self.normalization_profile not in _KNOWN_NORMALIZATION_PROFILES:
            raise ValueError("worker normalization profile is unsupported")
        if (
            isinstance(self.warm_repetitions, bool)
            or not isinstance(self.warm_repetitions, int)
            or self.warm_repetitions < 1
        ):
            raise ValueError("worker warm repetitions must be positive")
        if not isinstance(self.timing_sample_ids, tuple) or len(set(self.timing_sample_ids)) != len(
            self.timing_sample_ids
        ):
            raise ValueError("worker timing sample IDs must be a unique tuple")
        for sample_id in self.timing_sample_ids:
            _require_stable_id(sample_id, "<worker>", "timing_sample_ids")
        if self.text_retention not in {"full", "errors-only", "none"}:
            raise ValueError("worker text retention mode is unsupported")
        if not isinstance(self.retry_errors, bool):
            raise TypeError("worker retry_errors must be boolean")
        if (
            isinstance(self.worker_attempt_id, bool)
            or not isinstance(self.worker_attempt_id, int)
            or self.worker_attempt_id < 1
        ):
            raise ValueError("worker attempt ID must be positive")
        if not isinstance(self.audio_paths, tuple) or not all(
            isinstance(path, str) and Path(path).is_absolute() for path in self.audio_paths
        ):
            raise ValueError("worker audio paths must be absolute strings")


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
    _require_stable_id(canonical["dataset"], sample_id, "source.dataset")
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


def _safe_target_setting_id(value: object, field_name: str) -> str:
    """Validate one non-secret opaque identifier used in a target contract."""
    if (
        not isinstance(value, str)
        or SERIALIZED_ID_V1.fullmatch(value) is None
        or _UNSAFE_SAFE_SETTING_V1.search(value) is not None
    ):
        raise ValueError(f"safe target settings field {field_name} is invalid")
    return value


def _validate_safe_target_settings(
    settings: Mapping[str, object],
) -> dict[str, object]:
    """Return an allowlisted JSON-safe target-settings projection."""
    if not isinstance(settings, Mapping):
        raise ValueError("safe target settings must be an object")
    result = dict(settings)
    fields = set(result)
    if fields - _SAFE_TARGET_SETTING_FIELDS or not _SAFE_TARGET_SETTING_REQUIRED_FIELDS.issubset(fields):
        raise ValueError("safe target settings have missing or unknown fields")
    if result["mode"] not in {"neutral-v1", "production-v1"}:
        raise ValueError("safe target settings mode is invalid")
    if result["task"] != "transcribe":
        raise ValueError("safe target settings task is invalid")
    language = result["language"]
    if language is not None and (
        not isinstance(language, str) or BCP47_BASIC_V1.fullmatch(language) is None or language != language.lower()
    ):
        raise ValueError("safe target settings language is invalid")
    for name in (
        "word_timestamps",
        "diarization",
        "prompt_present",
    ):
        if not isinstance(result[name], bool):
            raise ValueError(f"safe target settings field {name} must be boolean")
    hotword_count = result["hotword_count"]
    if isinstance(hotword_count, bool) or not isinstance(hotword_count, int) or hotword_count < 0:
        raise ValueError("safe target settings hotword_count is invalid")
    for name in (
        "configuration_id",
        "network_collection_profile",
        "network_client_location",
    ):
        if name in result:
            _safe_target_setting_id(result[name], name)
    if result["mode"] == "neutral-v1" and "configuration_id" in result:
        raise ValueError("safe target settings configuration_id is invalid in neutral-v1")
    if result["mode"] == "production-v1" and "configuration_id" not in result:
        raise ValueError("safe target settings require configuration_id in production-v1")
    try:
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("safe target settings are not serializable") from exc
    return result


def _source_path_for_module(module_name: str) -> Path:
    """Resolve an allowlisted project module without importing it."""
    repository_root = Path(__file__).resolve().parents[2]
    relative = Path(*module_name.split("."))
    candidates = (
        repository_root / relative.with_suffix(".py"),
        repository_root / relative / "__init__.py",
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(repository_root) and resolved.is_file():
            return resolved
    raise ValueError("execution contract source module is unavailable")


def build_execution_contract(
    *,
    plan: SttBatchExecutionPlan,
    git_commit: str,
    safe_target_settings: Mapping[str, object],
) -> tuple[str, str]:
    """Build canonical safe execution-contract JSON and its SHA-256."""
    if not isinstance(git_commit, str) or _GIT_COMMIT_V1.fullmatch(git_commit) is None:
        raise ValueError("git commit must be a full lowercase hash or unknown")
    descriptor = getattr(plan, "descriptor", None)
    as_safe_dict = getattr(descriptor, "as_safe_dict", None)
    if not callable(as_safe_dict):
        raise ValueError("execution plan has no validated safe descriptor")
    descriptor_payload = as_safe_dict()
    if not isinstance(descriptor_payload, dict):
        raise ValueError("execution plan safe descriptor is invalid")
    safe_settings = _validate_safe_target_settings(safe_target_settings)
    declared_modules = getattr(descriptor, "source_modules", ())
    declared_dependencies = getattr(
        descriptor,
        "dependency_distributions",
        (),
    )
    if not isinstance(declared_modules, tuple) or not isinstance(
        declared_dependencies,
        tuple,
    ):
        raise ValueError("execution plan source identity is invalid")
    source_modules = sorted(_EXECUTION_CONTRACT_SOURCE_MODULES | set(declared_modules))
    source_hashes = {module_name: _sha256_file(_source_path_for_module(module_name)) for module_name in source_modules}
    dependency_versions: dict[str, str] = {}
    for distribution in sorted(declared_dependencies):
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = "unavailable"
    payload = {
        "descriptor": descriptor_payload,
        "dependency_versions": dependency_versions,
        "git_commit": git_commit,
        "safe_target_settings": safe_settings,
        "scorer_version": SCORER_VERSION,
        "source_hashes": source_hashes,
        "unicode_version": unicodedata.unidata_version,
    }
    try:
        contract_json = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("execution contract is not safely serializable") from exc
    return (
        contract_json,
        hashlib.sha256(contract_json.encode("utf-8")).hexdigest(),
    )


def _resolve_adapter_factory(
    path: str,
) -> Callable[[str], SttProviderAdapter]:
    """Resolve one importable top-level adapter factory."""
    if not isinstance(path, str) or _FACTORY_PATH_V1.fullmatch(path) is None:
        raise ValueError("adapter factory path must use module:top_level_name")
    module_name, attribute_name = path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        raise ValueError("adapter factory module could not be imported") from None
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ValueError("adapter factory must resolve to a callable")
    return factory


def _load_native_adapter(provider: str) -> SttProviderAdapter:
    """Load the native STT registry lazily and perform strict lookup."""
    module = importlib.import_module("tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter")
    registry = module.SttProviderRegistry()
    return registry.get_adapter_strict(provider)


def _common_preflight_settings(
    mode: str,
    common_settings: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], str]:
    """Validate planner inputs and derive their non-secret contract projection."""
    if mode not in {"neutral-v1", "production-v1"}:
        raise ValueError("benchmark mode must be neutral-v1 or production-v1")
    if not isinstance(common_settings, Mapping):
        raise ValueError("common target settings must be an object")
    source = dict(common_settings)
    if set(source) - _COMMON_TARGET_SETTING_FIELDS:
        raise ValueError("common target settings have unknown fields")
    task = source.get("task", "transcribe")
    language = source.get("language")
    word_timestamps = source.get("word_timestamps", False)
    prompt = source.get("prompt")
    hotwords = source.get("hotwords", ())
    diarization = source.get("diarization", False)
    git_commit = source.get("git_commit", "unknown")
    if task != "transcribe":
        raise ValueError("common target task must be transcribe")
    if language is not None and (not isinstance(language, str) or BCP47_BASIC_V1.fullmatch(language) is None):
        raise ValueError("common target language is invalid")
    language = language.lower() if isinstance(language, str) else None
    if not isinstance(word_timestamps, bool) or not isinstance(
        diarization,
        bool,
    ):
        raise ValueError("common target boolean settings are invalid")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("common target prompt must be text or null")
    if not isinstance(hotwords, tuple) or not all(isinstance(value, str) for value in hotwords):
        raise ValueError("common target hotwords must be a tuple of strings")
    if not isinstance(git_commit, str) or _GIT_COMMIT_V1.fullmatch(git_commit) is None:
        raise ValueError("common target git commit is invalid")
    configuration_id = source.get("configuration_id")
    if mode == "neutral-v1":
        if prompt is not None or hotwords or diarization or word_timestamps or configuration_id is not None:
            raise ValueError("neutral-v1 common target settings are not neutral")
    elif configuration_id is None:
        raise ValueError("production-v1 requires configuration_id")
    planner_settings = {
        "language": language,
        "task": task,
        "word_timestamps": word_timestamps,
        "prompt": prompt,
        "hotwords": hotwords,
        "diarization": diarization,
        "mode": mode,
    }
    safe_settings: dict[str, object] = {
        "mode": mode,
        "task": task,
        "language": language,
        "word_timestamps": word_timestamps,
        "diarization": diarization,
        "prompt_present": prompt is not None,
        "hotword_count": len(hotwords),
    }
    for name in (
        "configuration_id",
        "network_collection_profile",
        "network_client_location",
    ):
        value = source.get(name)
        if value is not None:
            safe_settings[name] = value
    return (
        planner_settings,
        _validate_safe_target_settings(safe_settings),
        git_commit,
    )


def _validate_preflight_plan(
    *,
    plan: SttBatchExecutionPlan,
    adapter: object,
    requested_model: str,
    mode: str,
    allow_network_targets: bool,
    planner_settings: Mapping[str, object],
) -> tuple[str, str]:
    """Validate one planned target without loading its model or opening audio."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
        SttAudioEgress,
        SttBatchExecutionPlan,
    )

    if not isinstance(plan, SttBatchExecutionPlan):
        raise ValueError("adapter returned an invalid execution plan")
    descriptor = plan.descriptor
    adapter_name = getattr(getattr(adapter, "name", None), "value", None)
    if not isinstance(adapter_name, str):
        adapter_name = getattr(adapter, "name", None)
    if (
        not isinstance(adapter_name, str)
        or descriptor.requested_provider != adapter_name
        or descriptor.resolved_provider != adapter_name
    ):
        raise ValueError("adapter and execution-plan provider mismatch")
    if (
        SAFE_MODEL_LABEL_V1.fullmatch(requested_model) is not None
        and descriptor.requested_model_label != requested_model
    ):
        raise ValueError("requested model and execution-plan model mismatch")
    for field_name in (
        "task",
        "language",
        "prompt",
        "diarization",
        "word_timestamps",
    ):
        if getattr(plan, field_name) != planner_settings[field_name]:
            raise ValueError(f"execution plan {field_name} mismatch")
    if plan.hotwords != planner_settings["hotwords"]:
        raise ValueError("execution plan hotwords mismatch")
    if mode == "neutral-v1":
        if len(descriptor.routes) != 1:
            raise ValueError("neutral-v1 execution plan cannot contain fallback routes")
        required_honors = (
            descriptor.honors_task,
            descriptor.honors_language,
            descriptor.honors_prompt_absence,
            descriptor.honors_hotword_absence,
            descriptor.honors_diarization,
            descriptor.honors_word_timestamps,
        )
        if not all(required_honors):
            raise ValueError("neutral-v1 execution plan cannot honor common semantics")
    for route in descriptor.routes:
        if route.would_download:
            raise ValueError("execution plan would download model artifacts")
        if route.provider != descriptor.resolved_provider:
            raise ValueError("execution route provider mismatch")
        if route.audio_egress is SttAudioEgress.NONE:
            if not route.local_model_available:
                raise ValueError("local execution artifact is unavailable")
        elif not allow_network_targets or route.endpoint_id is None:
            raise ValueError("network consent and an opaque endpoint are required")
    if descriptor.routes[0].model_label != descriptor.resolved_model_label:
        raise ValueError("primary execution route model mismatch")
    return (
        descriptor.requested_provider,
        descriptor.requested_model_label,
    )


def preflight_targets(
    target_specs: Sequence[str],
    *,
    mode: str,
    allow_network_targets: bool,
    common_settings: Mapping[str, object],
    adapter_factory_path: str = PRODUCTION_ADAPTER_FACTORY_PATH,
) -> tuple[PreparedTarget, ...]:
    """Plan and validate every target before returning any executable target."""
    if isinstance(target_specs, (str, bytes)) or not isinstance(
        target_specs,
        Sequence,
    ):
        raise ValueError("target specifications must be a sequence")
    if not target_specs:
        raise ValueError("at least one target is required")
    if not isinstance(allow_network_targets, bool):
        raise ValueError("allow_network_targets must be boolean")
    planner_settings, safe_settings, git_commit = _common_preflight_settings(
        mode,
        common_settings,
    )
    suppress_error_details = bool(planner_settings["prompt"] is not None or planner_settings["hotwords"])
    factory = _resolve_adapter_factory(adapter_factory_path)
    prepared: list[PreparedTarget] = []
    errors: list[str] = []
    normalized_targets: set[tuple[str, str]] = set()
    for ordinal, target_spec in enumerate(target_specs, start=1):
        try:
            if not isinstance(target_spec, str) or target_spec.count("=") < 1:
                raise ValueError("target must use provider=model")
            provider, model = target_spec.split("=", 1)
            provider = provider.strip()
            model = model.strip()
            if (
                not provider
                or SERIALIZED_ID_V1.fullmatch(provider) is None
                or not model
                or len(model) > 4096
                or any(ord(character) < 32 for character in model)
            ):
                raise ValueError("target provider or model is invalid")
            adapter = factory(provider)
            if adapter is None:
                raise ValueError("adapter is unavailable")
            get_capabilities = getattr(adapter, "get_capabilities", None)
            if not callable(get_capabilities):
                raise ValueError("adapter has no capabilities")
            capabilities = get_capabilities()
            if getattr(capabilities, "supports_batch", None) is not True:
                raise ValueError("adapter is unavailable for batch transcription")
            planner = getattr(adapter, "plan_batch_execution", None)
            if not callable(planner):
                raise ValueError("adapter has no execution planner")
            plan = planner(model=model, **planner_settings)
            normalized_provider, model_label = _validate_preflight_plan(
                plan=plan,
                adapter=adapter,
                requested_model=model,
                mode=mode,
                allow_network_targets=allow_network_targets,
                planner_settings=planner_settings,
            )
            normalized_identity = (normalized_provider, model_label)
            if normalized_identity in normalized_targets:
                raise ValueError("duplicate normalized target")
            normalized_targets.add(normalized_identity)
            contract_json, contract_hash = build_execution_contract(
                plan=plan,
                git_commit=git_commit,
                safe_target_settings=safe_settings,
            )
            descriptor_json = json.dumps(
                plan.descriptor.as_safe_dict(),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            target_id = "target-" + hashlib.sha256(f"{ordinal}\0{descriptor_json}".encode()).hexdigest()[:16]
            prepared.append(
                PreparedTarget(
                    target_id=target_id,
                    provider=normalized_provider,
                    model_label=model_label,
                    plan=plan,
                    adapter_factory_path=adapter_factory_path,
                    execution_contract_json=contract_json,
                    execution_contract_hash=contract_hash,
                )
            )
        except (
            AttributeError,
            ImportError,
            KeyError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            sanitized = sanitize_error(exc)
            detail = "preflight rejected target" if suppress_error_details else sanitized["message"]
            errors.append(f"target {ordinal}: {sanitized['type']}: {detail}")
    if errors:
        raise ValueError("target preflight failed: " + "; ".join(errors))
    return tuple(prepared)


_RUN_IDENTITY_FIELDS = (
    "manifest_hash",
    "selected_sample_ids",
    "profile",
    "mode",
    "seed",
    "cold_probe_sample_id",
    "warm_repetitions",
    "timing_sample_ids",
    "text_retention",
    "adapter_watchdog_seconds",
    "target_matrix",
    "environment",
)
_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "resume_identity_hash",
        *_RUN_IDENTITY_FIELDS,
        "next_operation_id",
        "next_attempt_id",
        "next_worker_attempt_id",
        "worker_attempts",
    }
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        "python_version",
        "unicode_version",
        "os_name",
        "os_release",
        "architecture",
        "logical_cores",
        "physical_cores",
        "ram_bytes",
        "cpu_model",
        "git_commit",
        "git_dirty",
        "ffprobe_version",
        "accelerator",
        "collection_methods",
    }
)
_ENVIRONMENT_METHOD_FIELDS = frozenset(
    {
        "cores",
        "ram",
        "cpu",
        "git",
        "ffprobe",
        "accelerator",
    }
)
_TARGET_MATRIX_FIELDS = frozenset(
    {
        "target_id",
        "provider",
        "model_label",
        "descriptor",
        "execution_contract_hash",
    }
)
_WORKER_ATTEMPT_FIELDS = frozenset(
    {
        "worker_attempt_id",
        "target_id",
        "status",
        "spawn_to_ready_nanoseconds",
        "setup_nanoseconds",
        "total_nanoseconds",
        "exit_code",
        "rewarm_status",
        "rewarm_nanoseconds",
        "error",
    }
)


def _validate_environment_fingerprint(
    environment: Mapping[str, object],
) -> dict[str, object]:
    """Validate the bounded environment projection persisted with one run."""
    if not isinstance(environment, Mapping) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment fingerprint has missing or unknown fields")
    result = dict(environment)
    for field in (
        "python_version",
        "unicode_version",
        "os_name",
        "os_release",
        "architecture",
        "cpu_model",
        "ffprobe_version",
        "accelerator",
    ):
        value = result[field]
        if not isinstance(value, str) or not value or len(value) > 256 or "\n" in value:
            raise ValueError(f"environment field {field} is invalid")
    git_commit = result["git_commit"]
    if not isinstance(git_commit, str) or _GIT_COMMIT_V1.fullmatch(git_commit) is None:
        raise ValueError("environment git commit is invalid")
    if result["git_dirty"] is not None and not isinstance(result["git_dirty"], bool):
        raise ValueError("environment git dirty flag is invalid")
    for field in ("logical_cores", "physical_cores", "ram_bytes"):
        value = result[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"environment field {field} is invalid")
    methods = result["collection_methods"]
    if not isinstance(methods, Mapping) or set(methods) != _ENVIRONMENT_METHOD_FIELDS:
        raise ValueError("environment collection methods are invalid")
    for value in methods.values():
        if not isinstance(value, str) or not value or len(value) > 128 or "\n" in value:
            raise ValueError("environment collection method is invalid")
    result["collection_methods"] = dict(methods)
    return result


def _prepared_target_matrix(
    prepared_targets: Sequence[PreparedTarget],
) -> list[dict[str, object]]:
    """Project prepared targets into the non-secret run metadata matrix."""
    if not prepared_targets:
        raise ValueError("run metadata requires at least one target")
    matrix: list[dict[str, object]] = []
    seen: set[str] = set()
    for target in prepared_targets:
        _verify_worker_target(target)
        if target.target_id in seen:
            raise ValueError("run metadata target IDs must be unique")
        seen.add(target.target_id)
        contract = json.loads(
            target.execution_contract_json,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        matrix.append(
            {
                "target_id": target.target_id,
                "provider": target.provider,
                "model_label": target.model_label,
                "descriptor": contract["descriptor"],
                "execution_contract_hash": target.execution_contract_hash,
            }
        )
    return matrix


def _validate_target_matrix(
    target_matrix: object,
) -> list[dict[str, object]]:
    """Validate the ordered, explicitly projected target matrix."""
    if not isinstance(target_matrix, list) or not target_matrix:
        raise ValueError("run target matrix must be a non-empty array")
    validated: list[dict[str, object]] = []
    seen: set[str] = set()
    for target in target_matrix:
        if not isinstance(target, dict) or set(target) != _TARGET_MATRIX_FIELDS:
            raise ValueError("run target matrix entry has invalid fields")
        _require_stable_id(target["target_id"], "<run>", "target_id")
        target_id = str(target["target_id"])
        if target_id in seen:
            raise ValueError("run target matrix IDs must be unique")
        seen.add(target_id)
        provider = target["provider"]
        model_label = target["model_label"]
        if not isinstance(provider, str) or SERIALIZED_ID_V1.fullmatch(provider) is None:
            raise ValueError("run target provider is invalid")
        if not isinstance(model_label, str) or SAFE_MODEL_LABEL_V1.fullmatch(model_label) is None:
            raise ValueError("run target model label is invalid")
        contract_hash = target["execution_contract_hash"]
        if not isinstance(contract_hash, str) or _SHA256_V1.fullmatch(contract_hash) is None:
            raise ValueError("run execution contract hash is invalid")
        descriptor = target["descriptor"]
        if not isinstance(descriptor, dict) or "runtime_settings" in descriptor:
            raise ValueError("run target descriptor is invalid")
        try:
            descriptor_json = json.dumps(
                descriptor,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, UnicodeEncodeError, ValueError) as exc:
            raise ValueError("run target descriptor is invalid") from exc
        if len(descriptor_json) > 131_072:
            raise ValueError("run target descriptor is invalid")
        validated.append(
            {
                "target_id": target_id,
                "provider": provider,
                "model_label": model_label,
                "descriptor": json.loads(descriptor_json),
                "execution_contract_hash": contract_hash,
            }
        )
    return validated


def _run_identity_hash(payload: Mapping[str, object]) -> str:
    """Hash exactly the immutable resume identity fields."""
    identity = {field: payload[field] for field in _RUN_IDENTITY_FIELDS}
    encoded = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    if not created and path.is_symlink():
        raise OSError("artifact parent must not be a symbolic link")
    if not path.is_dir():
        raise OSError("artifact parent must be a directory")
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


def _validate_worker_attempts(value: object) -> list[dict[str, object]]:
    """Validate bounded parent-owned worker-attempt observations."""
    if not isinstance(value, list):
        raise ValueError("worker attempts must be an array")
    attempts: list[dict[str, object]] = []
    previous = 0
    for item in value:
        if not isinstance(item, dict) or set(item) != _WORKER_ATTEMPT_FIELDS:
            raise ValueError("worker attempt has missing or unknown fields")
        attempt = dict(item)
        attempt_id = _require_result_integer(
            attempt["worker_attempt_id"],
            field="worker_attempt_id",
            minimum=1,
        )
        if attempt_id <= previous:
            raise ValueError("worker attempt IDs must increase")
        previous = attempt_id
        _require_stable_id(attempt["target_id"], "<run>", "target_id")
        if attempt["status"] not in {
            "running",
            "completed",
            "setup_error",
            "worker_crash",
            "timeout",
            "interrupted",
            "protocol_error",
        }:
            raise ValueError("worker attempt status is invalid")
        for field in (
            "spawn_to_ready_nanoseconds",
            "setup_nanoseconds",
            "total_nanoseconds",
            "rewarm_nanoseconds",
        ):
            number = attempt[field]
            if number is not None:
                _require_result_integer(number, field=field, minimum=0)
        exit_code = attempt["exit_code"]
        if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
            raise ValueError("worker attempt exit code is invalid")
        if attempt["rewarm_status"] is not None and attempt["rewarm_status"] not in RESULT_STATUSES:
            raise ValueError("worker attempt rewarm status is invalid")
        error = attempt["error"]
        if error is not None:
            if not isinstance(error, dict) or set(error) != {"type", "message"}:
                raise ValueError("worker attempt error is invalid")
            _require_result_text(
                error["type"],
                field="worker_attempt.error.type",
                maximum=128,
            )
            _require_result_text(
                error["message"],
                field="worker_attempt.error.message",
                maximum=512,
            )
        attempts.append(attempt)
    return attempts


def validate_run_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate one complete run artifact and its immutable identity hash."""
    if not isinstance(metadata, Mapping) or set(metadata) != _RUN_FIELDS:
        raise ValueError("run metadata has missing or unknown fields")
    result = dict(metadata)
    if result["schema_version"] != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported run schema version")
    _require_stable_id(result["run_id"], "<run>", "run_id")
    manifest_hash = result["manifest_hash"]
    if not isinstance(manifest_hash, str) or _SHA256_V1.fullmatch(manifest_hash) is None:
        raise ValueError("run manifest hash is invalid")
    selected = result["selected_sample_ids"]
    timing = result["timing_sample_ids"]
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or not isinstance(timing, list)
        or len(timing) != len(set(timing))
    ):
        raise ValueError("run sample IDs must be unique arrays")
    for sample_id in (*selected, *timing):
        _require_stable_id(sample_id, "<run>", "sample_id")
    _require_stable_id(
        result["cold_probe_sample_id"],
        "<run>",
        "cold_probe_sample_id",
    )
    if result["cold_probe_sample_id"] not in selected or not set(timing) <= set(selected):
        raise ValueError("run probe or timing sample selection is invalid")
    if result["profile"] not in _KNOWN_SAMPLE_PROFILES:
        raise ValueError("run profile is invalid")
    if result["mode"] not in {"neutral-v1", "production-v1"}:
        raise ValueError("run mode is invalid")
    _require_result_integer(result["seed"], field="seed", minimum=0)
    _require_result_integer(
        result["warm_repetitions"],
        field="warm_repetitions",
        minimum=1,
    )
    if result["text_retention"] not in {"full", "errors-only", "none"}:
        raise ValueError("run text retention is invalid")
    watchdog = result["adapter_watchdog_seconds"]
    if watchdog is not None and (
        isinstance(watchdog, bool)
        or not isinstance(watchdog, (int, float))
        or not math.isfinite(float(watchdog))
        or float(watchdog) <= 0.0
    ):
        raise ValueError("run adapter watchdog is invalid")
    result["target_matrix"] = _validate_target_matrix(result["target_matrix"])
    result["environment"] = _validate_environment_fingerprint(result["environment"])
    for field in (
        "next_operation_id",
        "next_attempt_id",
        "next_worker_attempt_id",
    ):
        _require_result_integer(result[field], field=field, minimum=1)
    result["worker_attempts"] = _validate_worker_attempts(result["worker_attempts"])
    resume_hash = result["resume_identity_hash"]
    if (
        not isinstance(resume_hash, str)
        or _SHA256_V1.fullmatch(resume_hash) is None
        or resume_hash != _run_identity_hash(result)
    ):
        raise ValueError("run resume identity is inconsistent")
    return result


def build_run_metadata(
    *,
    run_id: str,
    manifest_hash: str,
    selected_sample_ids: Sequence[str],
    profile: str,
    mode: str,
    seed: int,
    cold_probe_sample_id: str,
    warm_repetitions: int,
    timing_sample_ids: Sequence[str],
    text_retention: str,
    adapter_watchdog_seconds: float | None,
    prepared_targets: Sequence[PreparedTarget],
    environment: Mapping[str, object],
) -> dict[str, object]:
    """Build deterministic, secret-safe metadata for a new benchmark run."""
    payload: dict[str, object] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "selected_sample_ids": list(selected_sample_ids),
        "profile": profile,
        "mode": mode,
        "seed": seed,
        "cold_probe_sample_id": cold_probe_sample_id,
        "warm_repetitions": warm_repetitions,
        "timing_sample_ids": list(timing_sample_ids),
        "text_retention": text_retention,
        "adapter_watchdog_seconds": adapter_watchdog_seconds,
        "target_matrix": _prepared_target_matrix(prepared_targets),
        "environment": _validate_environment_fingerprint(environment),
        "next_operation_id": 1,
        "next_attempt_id": 1,
        "next_worker_attempt_id": 1,
        "worker_attempts": [],
    }
    payload["resume_identity_hash"] = _run_identity_hash(payload)
    return validate_run_metadata(payload)


def assert_resume_compatible(
    existing: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Reject an explicit resume whose immutable identity changed."""
    current = validate_run_metadata(existing)
    candidate = validate_run_metadata(expected)
    if current["run_id"] != candidate["run_id"] or current["resume_identity_hash"] != candidate["resume_identity_hash"]:
        raise ValueError("run is incompatible with requested resume settings")


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
    if actual and value is None:
        return
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
    actual_unverified = "actual_execution_unverified" in result["eligibility_reasons"]
    if result["actual_execution"] is None:
        if result["status"] in {"ok", "empty"} or not actual_unverified:
            raise ValueError("result actual execution is inconsistent with status")
    elif actual_unverified:
        raise ValueError("result actual execution is inconsistent with eligibility")
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
    try:
        descriptor = os.open(
            destination,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        if exc.errno == getattr(errno, "ELOOP", -1):
            raise OSError("results path must not be a symbolic link") from exc
        raise
    with os.fdopen(descriptor, "ab") as output:
        if not stat.S_ISREG(os.fstat(output.fileno()).st_mode):
            raise OSError("results path must be a regular file")
        if os.name == "posix":
            os.fchmod(output.fileno(), 0o600)
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    _fsync_directory(destination.parent)


def _decode_result_history(
    content: bytes,
) -> tuple[list[dict[str, object]], bool]:
    """Validate result JSONL bytes, ignoring only an incomplete final line."""
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


def load_result_history(path: Path) -> tuple[list[dict[str, object]], bool]:
    """Load validated history and ignore only an unterminated final JSONL line."""
    source = Path(path)
    try:
        content = source.read_bytes()
    except FileNotFoundError:
        return [], False
    return _decode_result_history(content)


def repair_result_history(path: Path) -> list[dict[str, object]]:
    """Validate history and durably remove only an incomplete final line."""
    source = Path(path)
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except FileNotFoundError:
        return []
    except OSError as exc:
        if exc.errno == getattr(errno, "ELOOP", -1):
            raise OSError("results path must not be a symbolic link") from exc
        raise
    with os.fdopen(descriptor, "r+b") as artifact:
        details = os.fstat(artifact.fileno())
        if not stat.S_ISREG(details.st_mode):
            raise OSError("results path must be a regular file")
        content = artifact.read()
        records, truncated = _decode_result_history(content)
        if truncated:
            artifact.seek(0)
            artifact.truncate(content.rfind(b"\n") + 1)
            artifact.flush()
            os.fsync(artifact.fileno())
        if os.name == "posix":
            os.fchmod(artifact.fileno(), 0o600)
    if truncated:
        _fsync_directory(source.parent)
    return records


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
        "measurement_role",
        "timing_class",
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
    measurement_role = validated["measurement_role"]
    timing_class = validated["timing_class"]
    if role == "result_call":
        _require_result_integer(repetition, field="repetition", minimum=0)
        _require_result_integer(
            result_attempt_id,
            field="result_attempt_id",
            minimum=1,
        )
        if measurement_role not in MEASUREMENT_ROLES:
            raise ValueError("unknown in-flight measurement role")
        if timing_class not in TIMING_CLASSES:
            raise ValueError("unknown in-flight timing class")
    else:
        if repetition is not None:
            _require_result_integer(repetition, field="repetition", minimum=0)
        if result_attempt_id is not None:
            raise ValueError("rewarm probe cannot allocate a result attempt")
        if measurement_role is not None or timing_class is not None:
            raise ValueError("rewarm probe cannot have result classifications")
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
    measurement_role: str | None,
    timing_class: str | None,
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
            "measurement_role": measurement_role,
            "timing_class": timing_class,
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
            if execution is None:
                values = ["unavailable"]
            elif not isinstance(execution, Mapping):
                raise ValueError("validated actual execution is not an object")
            else:
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


def _verify_worker_target(prepared_target: PreparedTarget) -> None:
    """Rebuild and compare the exact safe execution contract in the child."""
    if not isinstance(prepared_target, PreparedTarget):
        raise ValueError("worker target is invalid")
    try:
        payload = json.loads(
            prepared_target.execution_contract_json,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("worker execution contract is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("worker execution contract is invalid")
    required = {
        "descriptor",
        "dependency_versions",
        "git_commit",
        "safe_target_settings",
        "scorer_version",
        "source_hashes",
        "unicode_version",
    }
    if set(payload) != required:
        raise ValueError("worker execution contract fields changed")
    rebuilt_json, rebuilt_hash = build_execution_contract(
        plan=prepared_target.plan,
        git_commit=payload["git_commit"],
        safe_target_settings=payload["safe_target_settings"],
    )
    if (
        rebuilt_json != prepared_target.execution_contract_json
        or rebuilt_hash != prepared_target.execution_contract_hash
        or payload["descriptor"] != prepared_target.plan.descriptor.as_safe_dict()
        or prepared_target.provider != prepared_target.plan.descriptor.requested_provider
        or prepared_target.model_label != prepared_target.plan.descriptor.requested_model_label
    ):
        raise ValueError("worker execution contract mismatch")


def _actual_matches_worker_plan(
    actual: Mapping[str, object],
    plan: SttBatchExecutionPlan,
) -> bool:
    """Return whether one canonical actual envelope matches an approved route."""
    material_fields = (
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
    )
    for route in plan.descriptor.routes:
        matches = True
        for name in material_fields:
            expected = getattr(route, name)
            if hasattr(expected, "value"):
                expected = expected.value
            elif isinstance(expected, tuple):
                expected = list(expected)
            if expected is not None and actual[name] != expected:
                matches = False
                break
        if matches:
            return True
    return False


def _classify_worker_artifact(
    artifact: object,
    plan: SttBatchExecutionPlan,
) -> dict[str, object]:
    """Project one unrestricted adapter artifact into allowlisted result data."""
    invalid = {
        "status": "invalid_artifact",
        "hypothesis": "",
        "actual_execution": None,
        "execution_mismatch_reasons": [],
        "error": {
            "type": "InvalidArtifact",
            "message": "adapter returned an invalid normalized artifact",
        },
    }
    if not isinstance(artifact, Mapping):
        return invalid
    text = artifact.get("text")
    segments = artifact.get("segments")
    actual_value = artifact.get("actual_execution")
    if not isinstance(text, str) or not isinstance(segments, list) or not isinstance(actual_value, dict):
        return invalid
    try:
        _validate_execution_mapping(
            actual_value,
            field="actual_execution",
            actual=True,
        )
    except ValueError:
        return invalid
    actual = dict(actual_value)
    if not _actual_matches_worker_plan(actual, plan):
        return invalid
    mismatch_value = artifact.get("execution_mismatch", [])
    allowed_mismatches = {
        "diarization",
        "hotword_absence",
        "hotwords",
        "language",
        "prompt_absence",
        "task",
        "word_timestamps",
    }
    if (
        not isinstance(mismatch_value, list)
        or len(mismatch_value) > 8
        or not all(isinstance(reason, str) and reason in allowed_mismatches for reason in mismatch_value)
        or mismatch_value != sorted(set(mismatch_value))
    ):
        return invalid
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
        is_planned_stt_sentinel,
    )

    if is_planned_stt_sentinel(text):
        return {
            "status": "adapter_error",
            "hypothesis": "",
            "actual_execution": actual,
            "execution_mismatch_reasons": mismatch_value,
            "error": {
                "type": "TranscriptionSentinel",
                "message": "adapter returned a transcription error sentinel",
            },
        }
    if not normalize_strict_v1(text):
        return {
            "status": "empty",
            "hypothesis": "",
            "actual_execution": actual,
            "execution_mismatch_reasons": mismatch_value,
            "error": {
                "type": "EmptyTranscription",
                "message": "adapter returned an empty transcription",
            },
        }
    return {
        "status": "ok",
        "hypothesis": text,
        "actual_execution": actual,
        "execution_mismatch_reasons": mismatch_value,
        "error": None,
    }


def _receive_worker_ack(
    connection: object,
    *,
    message_type: str,
    operation_id: int | None = None,
    completion: str | None = None,
    requires_result: bool | None = None,
    target_id: str | None = None,
    worker_attempt_id: int | None = None,
) -> dict[str, object]:
    """Receive one exact coordinator acknowledgement or fail closed."""
    message = connection.recv()
    fields = {
        "ready_ack": {
            "type",
            "target_id",
            "worker_attempt_id",
        },
        "begin_ack": {
            "type",
            "operation_id",
            "result_attempt_id",
            "completion_key",
        },
        "adapter_done_ack": {"type", "operation_id"},
        "committed_ack": {"type", "operation_id"},
    }[message_type]
    if not isinstance(message, dict) or set(message) != fields:
        raise ValueError("worker received a malformed acknowledgement")
    if message["type"] != message_type:
        raise ValueError("worker received an unexpected acknowledgement")
    if message_type == "ready_ack":
        if (
            message["target_id"] != target_id
            or isinstance(message["worker_attempt_id"], bool)
            or not isinstance(message["worker_attempt_id"], int)
            or message["worker_attempt_id"] != worker_attempt_id
        ):
            raise ValueError("worker ready acknowledgement mismatch")
        return message
    acknowledged_operation = message["operation_id"]
    if (
        isinstance(acknowledged_operation, bool)
        or not isinstance(acknowledged_operation, int)
        or acknowledged_operation < 1
        or (operation_id is not None and acknowledged_operation != operation_id)
    ):
        raise ValueError("worker acknowledgement operation mismatch")
    if message_type == "begin_ack":
        if message["completion_key"] != completion:
            raise ValueError("worker acknowledgement completion mismatch")
        result_attempt_id = message["result_attempt_id"]
        if requires_result:
            if isinstance(result_attempt_id, bool) or not isinstance(result_attempt_id, int) or result_attempt_id < 1:
                raise ValueError("worker acknowledgement attempt is invalid")
        elif result_attempt_id is not None:
            raise ValueError("rewarm acknowledgement cannot allocate a result")
    return message


def _worker_result_record(
    *,
    prepared_target: PreparedTarget,
    settings: WorkerSettings,
    sample: ManifestSample,
    repetition: int,
    result_attempt_id: int,
    measurement_role: str,
    timing_class: str,
    adapter_nanoseconds: int | None,
    classified: Mapping[str, object],
) -> dict[str, object]:
    """Build one complete deterministic record after adapter acknowledgement."""
    status = str(classified["status"])
    hypothesis = str(classified["hypothesis"])
    score = score_result_text(
        sample.reference,
        hypothesis,
        status=status,
        normalization_profile=sample.normalization_profile,
    )
    retained_reference, retained_hypothesis = retain_text(
        mode=settings.text_retention,
        status=status,
        reference=sample.reference,
        hypothesis=hypothesis,
        score=score,
    )
    mismatch_reasons = list(classified["execution_mismatch_reasons"])
    eligibility_reasons = list(mismatch_reasons)
    actual_execution = classified["actual_execution"]
    if actual_execution is None:
        eligibility_reasons.append("actual_execution_unverified")
    elif isinstance(actual_execution, Mapping):
        actual_route_id = actual_execution["route_id"]
        actual_route = next(
            route for route in prepared_target.plan.descriptor.routes if route.route_id == actual_route_id
        )
        if not actual_route.identity_resolved:
            eligibility_reasons.append("identity_unresolved")
    rtf, throughput, eligibility_reasons = performance_fields(
        adapter_nanoseconds,
        sample.measured_duration_seconds,
        eligibility_reasons=eligibility_reasons,
    )
    dataset = dict(sample.source).get("dataset")
    score_fields = _score_as_result_fields(score)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": settings.run_id,
        "target_id": prepared_target.target_id,
        "completion_key": completion_key(
            settings.manifest_hash,
            prepared_target.target_id,
            prepared_target.execution_contract_hash,
            sample.sample_id,
            repetition,
        ),
        "sample_id": sample.sample_id,
        "repetition": repetition,
        "attempt_id": result_attempt_id,
        "worker_attempt_id": settings.worker_attempt_id,
        "measurement_role": measurement_role,
        "timing_class": timing_class,
        "suite": sample.suite,
        "suite_visibility": sample.suite_visibility,
        "dataset": dataset,
        "tags": list(sample.tags),
        "diagnostic_only": sample.diagnostic_only,
        "requested_execution": {
            "provider": prepared_target.provider,
            "model_label": prepared_target.model_label,
        },
        "actual_execution": classified["actual_execution"],
        "execution_mismatch_reasons": mismatch_reasons,
        "eligibility_reasons": eligibility_reasons,
        "status": status,
        "reference": retained_reference,
        "hypothesis": retained_hypothesis,
        "scorer_version": SCORER_VERSION,
        "strict_profile": STRICT_PROFILE,
        "normalization_profile": sample.normalization_profile,
        **score_fields,
        "adapter_nanoseconds": adapter_nanoseconds,
        "audio_duration_seconds": sample.measured_duration_seconds,
        "rtf": rtf,
        "throughput": throughput,
        "resource_observations": None,
        "error": classified["error"],
    }


def _worker_main(
    connection: object,
    prepared_target: PreparedTarget,
    samples: tuple[ManifestSample, ...],
    settings: WorkerSettings,
) -> None:
    """Execute one planned target through the coordinator acknowledgement pipe."""
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        os.environ[name] = "1"
    try:
        if (
            not isinstance(settings, WorkerSettings)
            or not isinstance(samples, tuple)
            or not samples
            or not all(isinstance(sample, ManifestSample) for sample in samples)
            or len(settings.audio_paths) != len(samples)
            or len({sample.sample_id for sample in samples}) != len(samples)
            or settings.cold_probe_sample_id not in {sample.sample_id for sample in samples}
            or not set(settings.timing_sample_ids) <= {sample.sample_id for sample in samples}
        ):
            raise ValueError("worker sample settings are invalid")
        _verify_worker_target(prepared_target)
        history, truncated = load_result_history(
            Path(settings.results_path),
        )
        if truncated:
            raise ValueError("worker result history is truncated")
        setup_started = time.monotonic_ns()
        factory = _resolve_adapter_factory(
            prepared_target.adapter_factory_path,
        )
        adapter = factory(prepared_target.provider)
        adapter_name = getattr(getattr(adapter, "name", None), "value", None)
        if adapter_name is None:
            adapter_name = getattr(adapter, "name", None)
        if adapter_name != prepared_target.provider or not callable(getattr(adapter, "transcribe_batch", None)):
            raise ValueError("worker adapter does not match prepared target")
        setup_nanoseconds = time.monotonic_ns() - setup_started
    except (
        AttributeError,
        ImportError,
        KeyError,
        LookupError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        error_type = sanitize_error(exc)["type"]
        connection.send(
            {
                "type": "ready",
                "target_id": getattr(
                    prepared_target,
                    "target_id",
                    "target-invalid",
                ),
                "worker_attempt_id": getattr(
                    settings,
                    "worker_attempt_id",
                    1,
                ),
                "setup_nanoseconds": 0,
                "status": "error",
                "error": {
                    "type": error_type,
                    "message": "worker setup failed",
                },
            }
        )
        connection.close()
        return

    connection.send(
        {
            "type": "ready",
            "target_id": prepared_target.target_id,
            "worker_attempt_id": settings.worker_attempt_id,
            "setup_nanoseconds": setup_nanoseconds,
            "status": "ok",
            "error": None,
        }
    )
    _receive_worker_ack(
        connection,
        message_type="ready_ack",
        target_id=prepared_target.target_id,
        worker_attempt_id=settings.worker_attempt_id,
    )
    active = reduce_attempts(history)
    ordered = list(zip(samples, settings.audio_paths, strict=True))
    probe_index = next(
        index for index, (sample, _) in enumerate(ordered) if sample.sample_id == settings.cold_probe_sample_id
    )
    ordered.insert(0, ordered.pop(probe_index))

    def run_operation(
        sample: ManifestSample,
        audio_path: str,
        *,
        repetition: int,
        operation_role: str,
        measurement_role: str | None,
        timing_class: str | None,
    ) -> str:
        key = completion_key(
            settings.manifest_hash,
            prepared_target.target_id,
            prepared_target.execution_contract_hash,
            sample.sample_id,
            repetition,
        )
        connection.send(
            {
                "type": "begin",
                "target_id": prepared_target.target_id,
                "worker_attempt_id": settings.worker_attempt_id,
                "sample_id": sample.sample_id,
                "completion_key": key,
                "repetition": repetition,
                "operation_role": operation_role,
                "measurement_role": measurement_role,
                "timing_class": timing_class,
            }
        )
        begin_ack = _receive_worker_ack(
            connection,
            message_type="begin_ack",
            completion=key,
            requires_result=operation_role == "result_call",
        )
        operation_id = int(begin_ack["operation_id"])
        started = time.perf_counter_ns()
        artifact: object | None = None
        adapter_exception: BaseException | None = None
        try:
            artifact = adapter.transcribe_batch(
                audio_path,
                model=prepared_target.plan.descriptor.requested_model_label,
                language=prepared_target.plan.language,
                task=prepared_target.plan.task,
                word_timestamps=prepared_target.plan.word_timestamps,
                prompt=prepared_target.plan.prompt,
                hotwords=prepared_target.plan.hotwords,
                execution_plan=prepared_target.plan,
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary
            stopped = time.perf_counter_ns()
            adapter_exception = exc
            adapter_outcome = "raised"
        else:
            stopped = time.perf_counter_ns()
            adapter_outcome = "returned"
        adapter_nanoseconds = stopped - started
        connection.send(
            {
                "type": "adapter_done",
                "operation_id": operation_id,
                "status": adapter_outcome,
                "adapter_nanoseconds": adapter_nanoseconds,
            }
        )
        _receive_worker_ack(
            connection,
            message_type="adapter_done_ack",
            operation_id=operation_id,
        )
        if adapter_exception is not None:
            classified: dict[str, object] = {
                "status": "adapter_error",
                "hypothesis": "",
                "actual_execution": None,
                "execution_mismatch_reasons": [],
                "error": sanitize_error(adapter_exception),
            }
        else:
            classified = _classify_worker_artifact(
                artifact,
                prepared_target.plan,
            )
        status = str(classified["status"])
        result_attempt_id = begin_ack["result_attempt_id"]
        if operation_role == "result_call":
            if measurement_role is None or timing_class is None:
                raise ValueError("result operation is missing measurement classification")
            record = _worker_result_record(
                prepared_target=prepared_target,
                settings=settings,
                sample=sample,
                repetition=repetition,
                result_attempt_id=int(result_attempt_id),
                measurement_role=measurement_role,
                timing_class=timing_class,
                adapter_nanoseconds=adapter_nanoseconds,
                classified=classified,
            )
            append_result_record(Path(settings.results_path), record)
            active[key] = record
        connection.send(
            {
                "type": "committed",
                "operation_id": operation_id,
                "completion_key": key,
                "result_attempt_id": result_attempt_id,
                "status": status,
            }
        )
        _receive_worker_ack(
            connection,
            message_type="committed_ack",
            operation_id=operation_id,
        )
        return status

    probe, probe_path = ordered[0]
    probe_key = completion_key(
        settings.manifest_hash,
        prepared_target.target_id,
        prepared_target.execution_contract_hash,
        probe.sample_id,
        0,
    )
    probe_action = resume_action(
        active.get(probe_key),
        retry_errors=settings.retry_errors,
    )
    probe_role = "result_call" if probe_action in {"execute", "retry"} else "rewarm_probe"
    probe_status = run_operation(
        probe,
        probe_path,
        repetition=0,
        operation_role=probe_role,
        measurement_role=("accuracy" if probe_role == "result_call" else None),
        timing_class=("cold_first" if probe_role == "result_call" else None),
    )
    warmed = probe_status == "ok"

    for sample, audio_path in ordered[1:]:
        repetitions = settings.warm_repetitions if sample.sample_id in settings.timing_sample_ids else 1
        for repetition in range(repetitions):
            key = completion_key(
                settings.manifest_hash,
                prepared_target.target_id,
                prepared_target.execution_contract_hash,
                sample.sample_id,
                repetition,
            )
            if (
                resume_action(
                    active.get(key),
                    retry_errors=settings.retry_errors,
                )
                == "skip"
            ):
                continue
            timing_class = "warm" if warmed else "warmup_recovery"
            status = run_operation(
                sample,
                audio_path,
                repetition=repetition,
                operation_role="result_call",
                measurement_role=("accuracy" if repetition == 0 else "performance_repeat"),
                timing_class=timing_class,
            )
            if status == "ok":
                warmed = True
    connection.close()


def _load_json_object(path: Path) -> dict[str, object]:
    """Load one strict JSON object without following a symbolic link."""
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        if exc.errno == getattr(errno, "ELOOP", -1):
            raise OSError("artifact path must not be a symbolic link") from exc
        raise
    with os.fdopen(descriptor, "rb") as artifact:
        if not stat.S_ISREG(os.fstat(artifact.fileno()).st_mode):
            raise OSError("artifact path must be a regular file")
        content = artifact.read()
    try:
        result = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("artifact contains invalid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("artifact must contain a JSON object")
    return result


def _clear_inflight(path: Path) -> None:
    """Durably remove the coordinator in-flight marker."""
    destination = Path(path)
    try:
        if destination.is_symlink():
            raise OSError("in-flight path must not be a symbolic link")
        destination.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(destination.parent)


def _coordinator_failure_record(
    *,
    prepared_target: PreparedTarget,
    settings: WorkerSettings,
    sample: ManifestSample,
    inflight: Mapping[str, object],
    status: str,
) -> dict[str, object]:
    """Build a scored empty-hypothesis result for an uncommitted call."""
    active = validate_inflight_record(inflight)
    if active["operation_role"] != "result_call" or status not in {
        "worker_crash",
        "timeout",
        "interrupted",
    }:
        raise ValueError("coordinator failure cannot create this result")
    measurement_role = active["measurement_role"]
    timing_class = active["timing_class"]
    result_attempt_id = active["result_attempt_id"]
    if not isinstance(measurement_role, str) or not isinstance(timing_class, str):
        raise ValueError("coordinator failure lacks result classifications")
    if not isinstance(result_attempt_id, int):
        raise ValueError("coordinator failure lacks a result attempt")
    error_types = {
        "worker_crash": "WorkerCrash",
        "timeout": "AdapterTimeout",
        "interrupted": "Interrupted",
    }
    return _worker_result_record(
        prepared_target=prepared_target,
        settings=settings,
        sample=sample,
        repetition=int(active["repetition"]),
        result_attempt_id=result_attempt_id,
        measurement_role=measurement_role,
        timing_class=timing_class,
        adapter_nanoseconds=None,
        classified={
            "status": status,
            "hypothesis": "",
            "actual_execution": None,
            "execution_mismatch_reasons": [],
            "error": {
                "type": error_types[status],
                "message": "worker did not durably commit an adapter result",
            },
        },
    )


def _new_worker_attempt(
    worker_attempt_id: int,
    target_id: str,
) -> dict[str, object]:
    """Return the fixed parent-owned worker-attempt envelope."""
    return {
        "worker_attempt_id": worker_attempt_id,
        "target_id": target_id,
        "status": "running",
        "spawn_to_ready_nanoseconds": None,
        "setup_nanoseconds": None,
        "total_nanoseconds": None,
        "exit_code": None,
        "rewarm_status": None,
        "rewarm_nanoseconds": None,
        "error": None,
    }


def _validate_worker_message(
    message: object,
    *,
    message_type: str,
    fields: set[str],
) -> dict[str, object]:
    """Require one exact worker protocol message."""
    if not isinstance(message, dict) or set(message) != fields:
        raise ValueError("worker sent a malformed protocol message")
    if message["type"] != message_type:
        raise ValueError("worker sent an unexpected protocol message")
    return message


def _execute_target_attempt(
    *,
    run_directory: Path,
    metadata: Mapping[str, object],
    prepared_target: PreparedTarget,
    samples: tuple[ManifestSample, ...],
    audio_paths: tuple[str, ...],
    retry_errors: bool,
) -> dict[str, object]:
    """Run one spawned target while the parent owns durable coordination."""
    run_path = run_directory / "run.json"
    results_path = run_directory / "results.jsonl"
    inflight_path = run_directory / "inflight.json"
    current = validate_run_metadata(metadata)
    worker_attempt_id = int(current["next_worker_attempt_id"])
    attempt = _new_worker_attempt(
        worker_attempt_id,
        prepared_target.target_id,
    )
    current["next_worker_attempt_id"] = worker_attempt_id + 1
    current["worker_attempts"].append(attempt)
    validate_run_metadata(current)
    atomic_write_json(run_path, current)
    settings = WorkerSettings(
        run_id=str(current["run_id"]),
        results_path=str(results_path.resolve()),
        manifest_hash=str(current["manifest_hash"]),
        normalization_profile=samples[0].normalization_profile,
        cold_probe_sample_id=str(current["cold_probe_sample_id"]),
        warm_repetitions=int(current["warm_repetitions"]),
        timing_sample_ids=tuple(str(value) for value in current["timing_sample_ids"]),
        text_retention=str(current["text_retention"]),
        retry_errors=retry_errors,
        worker_attempt_id=worker_attempt_id,
        audio_paths=audio_paths,
    )
    sample_by_id = {sample.sample_id: sample for sample in samples}
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=_worker_main,
        args=(child_connection, prepared_target, samples, settings),
    )
    started = time.perf_counter_ns()
    process.start()
    child_connection.close()
    ready_seen = False
    setup_failed = False
    active_inflight: dict[str, object] | None = None
    adapter_deadline: float | None = None
    adapter_done_seen = False
    last_adapter_nanoseconds: int | None = None
    failure_status: str | None = None
    protocol_error: BaseException | None = None
    interrupted = False
    watchdog = current["adapter_watchdog_seconds"]
    try:
        while True:
            if adapter_deadline is not None:
                remaining = adapter_deadline - time.monotonic()
                if remaining <= 0 and not parent_connection.poll(0):
                    failure_status = "timeout"
                    process.terminate()
                    break
                wait_seconds = max(0.0, min(0.05, remaining))
            else:
                wait_seconds = 0.05
            try:
                has_message = parent_connection.poll(wait_seconds)
            except (OSError, EOFError):
                has_message = False
            if not has_message:
                if not process.is_alive():
                    try:
                        if parent_connection.poll(0):
                            has_message = True
                        else:
                            break
                    except (OSError, EOFError):
                        break
                else:
                    continue
            if not has_message:
                continue
            try:
                raw_message = parent_connection.recv()
            except (EOFError, OSError):
                break
            if not isinstance(raw_message, dict):
                raise ValueError("worker sent a malformed protocol message")
            message_type = raw_message.get("type")
            if message_type == "ready":
                if ready_seen or active_inflight is not None:
                    raise ValueError("worker sent ready out of sequence")
                message = _validate_worker_message(
                    raw_message,
                    message_type="ready",
                    fields={
                        "type",
                        "target_id",
                        "worker_attempt_id",
                        "setup_nanoseconds",
                        "status",
                        "error",
                    },
                )
                setup_nanoseconds = message["setup_nanoseconds"]
                if (
                    message["target_id"] != prepared_target.target_id
                    or message["worker_attempt_id"] != worker_attempt_id
                    or isinstance(setup_nanoseconds, bool)
                    or not isinstance(setup_nanoseconds, int)
                    or setup_nanoseconds < 0
                    or message["status"] not in {"ok", "error"}
                ):
                    raise ValueError("worker ready identity is invalid")
                error = message["error"]
                if message["status"] == "ok":
                    if error is not None:
                        raise ValueError("successful worker ready cannot have an error")
                elif not isinstance(error, dict) or set(error) != {"type", "message"}:
                    raise ValueError("failed worker ready must have a bounded error")
                ready_seen = True
                attempt["spawn_to_ready_nanoseconds"] = time.perf_counter_ns() - started
                attempt["setup_nanoseconds"] = setup_nanoseconds
                if message["status"] == "error":
                    setup_failed = True
                    attempt["status"] = "setup_error"
                    attempt["error"] = error
                validate_run_metadata(current)
                atomic_write_json(run_path, current)
                if setup_failed:
                    continue
                parent_connection.send(
                    {
                        "type": "ready_ack",
                        "target_id": prepared_target.target_id,
                        "worker_attempt_id": worker_attempt_id,
                    }
                )
                continue
            if message_type == "begin":
                if not ready_seen or setup_failed or active_inflight is not None:
                    raise ValueError("worker sent begin out of sequence")
                message = _validate_worker_message(
                    raw_message,
                    message_type="begin",
                    fields={
                        "type",
                        "target_id",
                        "worker_attempt_id",
                        "sample_id",
                        "completion_key",
                        "repetition",
                        "operation_role",
                        "measurement_role",
                        "timing_class",
                    },
                )
                sample_id = message["sample_id"]
                repetition = message["repetition"]
                if (
                    message["target_id"] != prepared_target.target_id
                    or message["worker_attempt_id"] != worker_attempt_id
                    or sample_id not in sample_by_id
                    or isinstance(repetition, bool)
                    or not isinstance(repetition, int)
                    or repetition < 0
                ):
                    raise ValueError("worker begin identity is invalid")
                expected_key = completion_key(
                    str(current["manifest_hash"]),
                    prepared_target.target_id,
                    prepared_target.execution_contract_hash,
                    str(sample_id),
                    repetition,
                )
                if message["completion_key"] != expected_key:
                    raise ValueError("worker begin completion key is invalid")
                current, active_inflight = allocate_inflight(
                    current,
                    target_id=prepared_target.target_id,
                    operation_role=str(message["operation_role"]),
                    worker_attempt_id=worker_attempt_id,
                    sample_id=str(sample_id),
                    completion_key=expected_key,
                    repetition=repetition,
                    measurement_role=message["measurement_role"],
                    timing_class=message["timing_class"],
                )
                validate_run_metadata(current)
                atomic_write_json(run_path, current)
                atomic_write_json(inflight_path, active_inflight)
                adapter_done_seen = False
                if watchdog is not None:
                    adapter_deadline = time.monotonic() + float(watchdog)
                parent_connection.send(
                    {
                        "type": "begin_ack",
                        "operation_id": active_inflight["operation_id"],
                        "result_attempt_id": active_inflight["result_attempt_id"],
                        "completion_key": active_inflight["completion_key"],
                    }
                )
                continue
            if message_type == "adapter_done":
                if active_inflight is None or adapter_done_seen:
                    raise ValueError("worker sent adapter_done out of sequence")
                message = _validate_worker_message(
                    raw_message,
                    message_type="adapter_done",
                    fields={
                        "type",
                        "operation_id",
                        "status",
                        "adapter_nanoseconds",
                    },
                )
                adapter_nanoseconds = message["adapter_nanoseconds"]
                if (
                    message["operation_id"] != active_inflight["operation_id"]
                    or message["status"] not in {"returned", "raised"}
                    or isinstance(adapter_nanoseconds, bool)
                    or not isinstance(adapter_nanoseconds, int)
                    or adapter_nanoseconds < 0
                ):
                    raise ValueError("worker adapter_done is invalid")
                adapter_deadline = None
                adapter_done_seen = True
                last_adapter_nanoseconds = adapter_nanoseconds
                parent_connection.send(
                    {
                        "type": "adapter_done_ack",
                        "operation_id": active_inflight["operation_id"],
                    }
                )
                continue
            if message_type == "committed":
                if active_inflight is None or not adapter_done_seen:
                    raise ValueError("worker sent committed out of sequence")
                message = _validate_worker_message(
                    raw_message,
                    message_type="committed",
                    fields={
                        "type",
                        "operation_id",
                        "completion_key",
                        "result_attempt_id",
                        "status",
                    },
                )
                if (
                    message["operation_id"] != active_inflight["operation_id"]
                    or message["completion_key"] != active_inflight["completion_key"]
                    or message["result_attempt_id"] != active_inflight["result_attempt_id"]
                    or message["status"] not in RESULT_STATUSES
                ):
                    raise ValueError("worker committed identity is invalid")
                if active_inflight["operation_role"] == "result_call":
                    history, truncated = load_result_history(results_path)
                    if truncated:
                        raise ValueError("worker committed a truncated result")
                    exact = [
                        record
                        for record in history
                        if record["completion_key"] == active_inflight["completion_key"]
                        and record["attempt_id"] == active_inflight["result_attempt_id"]
                    ]
                    if len(exact) != 1 or exact[0]["status"] != message["status"]:
                        raise ValueError("worker committed result is missing")
                else:
                    attempt["rewarm_status"] = message["status"]
                    attempt["rewarm_nanoseconds"] = last_adapter_nanoseconds
                    validate_run_metadata(current)
                    atomic_write_json(run_path, current)
                operation_id = int(active_inflight["operation_id"])
                _clear_inflight(inflight_path)
                active_inflight = None
                adapter_done_seen = False
                last_adapter_nanoseconds = None
                parent_connection.send(
                    {
                        "type": "committed_ack",
                        "operation_id": operation_id,
                    }
                )
                continue
            raise ValueError("worker sent an unknown protocol message")
    except KeyboardInterrupt:
        interrupted = True
        failure_status = "interrupted"
        if process.is_alive():
            process.terminate()
    except (TypeError, ValueError) as exc:
        protocol_error = exc
        if process.is_alive():
            process.terminate()
    finally:
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent_connection.close()

    if active_inflight is not None:
        history = repair_result_history(results_path)
        action = recover_inflight_action(
            active_inflight,
            history,
            interrupted=interrupted,
            timed_out=failure_status == "timeout",
        )
        if action["action"] == "append_result":
            sample = sample_by_id[str(active_inflight["sample_id"])]
            record = _coordinator_failure_record(
                prepared_target=prepared_target,
                settings=settings,
                sample=sample,
                inflight=active_inflight,
                status=str(action["status"]),
            )
            append_result_record(results_path, record)
        elif action["action"] == "record_rewarm":
            attempt["rewarm_status"] = action["status"]
            attempt["rewarm_nanoseconds"] = last_adapter_nanoseconds
        _clear_inflight(inflight_path)

    attempt["total_nanoseconds"] = time.perf_counter_ns() - started
    attempt["exit_code"] = process.exitcode
    if interrupted:
        attempt["status"] = "interrupted"
        attempt["error"] = {
            "type": "Interrupted",
            "message": "benchmark worker was interrupted",
        }
    elif protocol_error is not None:
        attempt["status"] = "protocol_error"
        attempt["error"] = {
            "type": "ProtocolError",
            "message": "worker violated the benchmark protocol",
        }
    elif failure_status == "timeout":
        attempt["status"] = "timeout"
        attempt["error"] = {
            "type": "AdapterTimeout",
            "message": "adapter call exceeded the configured watchdog",
        }
    elif setup_failed:
        pass
    elif not ready_seen or process.exitcode != 0:
        attempt["status"] = "worker_crash"
        attempt["error"] = {
            "type": "WorkerCrash",
            "message": "benchmark worker exited before clean completion",
        }
    else:
        attempt["status"] = "completed"
    validate_run_metadata(current)
    atomic_write_json(run_path, current)
    if interrupted:
        raise KeyboardInterrupt
    return current


def _recover_persisted_inflight(
    *,
    run_directory: Path,
    metadata: Mapping[str, object],
    prepared_targets: Sequence[PreparedTarget],
    samples: tuple[ManifestSample, ...],
    audio_paths: tuple[str, ...],
) -> dict[str, object]:
    """Attribute a prior parent/worker crash before starting a new worker."""
    inflight_path = run_directory / "inflight.json"
    active = validate_inflight_record(_load_json_object(inflight_path))
    current = validate_run_metadata(metadata)
    target_by_id = {target.target_id: target for target in prepared_targets}
    sample_by_id = {sample.sample_id: sample for sample in samples}
    target = target_by_id.get(str(active["target_id"]))
    sample = sample_by_id.get(str(active["sample_id"]))
    if target is None or sample is None:
        raise ValueError("in-flight work does not belong to this run")
    expected_key = completion_key(
        str(current["manifest_hash"]),
        target.target_id,
        target.execution_contract_hash,
        sample.sample_id,
        int(active["repetition"] or 0),
    )
    if active["completion_key"] != expected_key:
        raise ValueError("in-flight completion key does not belong to this run")
    matching_attempts = [
        attempt
        for attempt in current["worker_attempts"]
        if attempt["worker_attempt_id"] == active["worker_attempt_id"] and attempt["target_id"] == active["target_id"]
    ]
    if len(matching_attempts) != 1 or matching_attempts[0]["status"] != "running":
        raise ValueError("in-flight worker attempt is inconsistent")
    attempt = matching_attempts[0]
    history = repair_result_history(run_directory / "results.jsonl")
    action = recover_inflight_action(active, history)
    settings = WorkerSettings(
        run_id=str(current["run_id"]),
        results_path=str((run_directory / "results.jsonl").resolve()),
        manifest_hash=str(current["manifest_hash"]),
        normalization_profile=sample.normalization_profile,
        cold_probe_sample_id=str(current["cold_probe_sample_id"]),
        warm_repetitions=int(current["warm_repetitions"]),
        timing_sample_ids=tuple(str(value) for value in current["timing_sample_ids"]),
        text_retention=str(current["text_retention"]),
        retry_errors=False,
        worker_attempt_id=int(active["worker_attempt_id"]),
        audio_paths=audio_paths,
    )
    if action["action"] == "append_result":
        append_result_record(
            run_directory / "results.jsonl",
            _coordinator_failure_record(
                prepared_target=target,
                settings=settings,
                sample=sample,
                inflight=active,
                status="worker_crash",
            ),
        )
    elif action["action"] == "record_rewarm":
        attempt["rewarm_status"] = "worker_crash"
    attempt["status"] = "worker_crash"
    attempt["error"] = {
        "type": "WorkerCrash",
        "message": "prior benchmark worker did not complete coordination",
    }
    validate_run_metadata(current)
    atomic_write_json(run_directory / "run.json", current)
    _clear_inflight(inflight_path)
    return current


def execute_prepared_targets(
    *,
    run_directory: Path,
    run_metadata: Mapping[str, object],
    prepared_targets: Sequence[PreparedTarget],
    samples: Sequence[ManifestSample],
    audio_paths: Sequence[str],
    retry_errors: bool,
) -> dict[str, object]:
    """Create or resume a run and execute targets sequentially in CLI order."""
    if not isinstance(retry_errors, bool):
        raise TypeError("retry_errors must be boolean")
    expected = validate_run_metadata(run_metadata)
    targets = tuple(prepared_targets)
    selected_samples = tuple(samples)
    pinned_paths = tuple(audio_paths)
    if (
        not targets
        or not selected_samples
        or len(selected_samples) != len(pinned_paths)
        or not all(isinstance(sample, ManifestSample) for sample in selected_samples)
        or not all(isinstance(path, str) and Path(path).is_absolute() for path in pinned_paths)
        or [sample.sample_id for sample in selected_samples] != expected["selected_sample_ids"]
        or _prepared_target_matrix(targets) != expected["target_matrix"]
    ):
        raise ValueError("run inputs do not match immutable metadata")
    run_directory = Path(run_directory)
    _ensure_owner_directory(run_directory)
    run_path = run_directory / "run.json"
    if run_path.exists():
        current = validate_run_metadata(_load_json_object(run_path))
        assert_resume_compatible(current, expected)
    else:
        if any(run_directory.iterdir()):
            raise ValueError("new run directory must be empty")
        current = expected
        atomic_write_json(run_path, current)
    repair_result_history(run_directory / "results.jsonl")
    if (run_directory / "inflight.json").exists():
        current = _recover_persisted_inflight(
            run_directory=run_directory,
            metadata=current,
            prepared_targets=targets,
            samples=selected_samples,
            audio_paths=pinned_paths,
        )
    for target in targets:
        current = _execute_target_attempt(
            run_directory=run_directory,
            metadata=current,
            prepared_target=target,
            samples=selected_samples,
            audio_paths=pinned_paths,
            retry_errors=retry_errors,
        )
    return validate_run_metadata(current)


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
