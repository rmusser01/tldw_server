"""Deterministic normalization and scoring for native STT benchmarks."""

from __future__ import annotations

import argparse
import errno
import hashlib
import html
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import re
import stat
import subprocess  # nosec B404
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
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
_KNOWN_REFERENCE_PROVENANCE = frozenset({"canonical-dataset", "human-reviewed"})
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
        "reference_provenance",
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
    if canonical["reference_provenance"] not in _KNOWN_REFERENCE_PROVENANCE:
        raise _manifest_error(
            sample_id,
            "source.reference_provenance",
            "must be canonical-dataset or human-reviewed",
        )
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
    "reference_provenance_counts",
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
        "execution_contract",
        "execution_contract_hash",
    }
)
_EXECUTION_CONTRACT_FIELDS = frozenset(
    {
        "descriptor",
        "dependency_versions",
        "git_commit",
        "safe_target_settings",
        "scorer_version",
        "source_hashes",
        "unicode_version",
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


def _bounded_environment_text(
    value: object,
    *,
    fallback: str = "unavailable",
) -> str:
    """Return one bounded, single-line environment identity value."""
    if not isinstance(value, str):
        return fallback
    first_line = value.splitlines()[0].strip() if value.splitlines() else ""
    return first_line[:256] or fallback


def _run_environment_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Run one fixed local identity command without exposing its failures."""
    try:
        return subprocess.run(  # nosec B603
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def _positive_environment_integer(value: object) -> int | None:
    """Return a positive non-boolean integer or None."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def collect_environment_fingerprint() -> dict[str, object]:
    """Collect a bounded, model-free environment identity for run metadata."""
    logical_cores = _positive_environment_integer(os.cpu_count())
    physical_cores: int | None = None
    ram_bytes: int | None = None
    cores_method = "os.cpu_count" if logical_cores is not None else "unavailable"
    ram_method = "unavailable"
    try:
        psutil = importlib.import_module("psutil")
        logical_candidate = _positive_environment_integer(
            psutil.cpu_count(logical=True),
        )
        physical_candidate = _positive_environment_integer(
            psutil.cpu_count(logical=False),
        )
        ram_candidate = _positive_environment_integer(
            getattr(psutil.virtual_memory(), "total", None),
        )
        if logical_candidate is not None:
            logical_cores = logical_candidate
        if physical_candidate is not None:
            physical_cores = physical_candidate
        if logical_candidate is not None or physical_candidate is not None:
            cores_method = "psutil"
        if ram_candidate is not None:
            ram_bytes = ram_candidate
            ram_method = "psutil"
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError):
        pass

    repository_root = Path(__file__).resolve().parents[2]
    git_commit = "unknown"
    git_dirty: bool | None = None
    git_method = "unavailable"
    commit_result = _run_environment_command(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
    )
    status_result = _run_environment_command(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=repository_root,
    )
    if (
        commit_result is not None
        and commit_result.returncode == 0
        and _GIT_COMMIT_V1.fullmatch(commit_result.stdout.strip()) is not None
    ):
        git_commit = commit_result.stdout.strip()
    if status_result is not None and status_result.returncode == 0:
        git_dirty = bool(status_result.stdout.strip())
    if git_commit != "unknown" or git_dirty is not None:
        git_method = "git-cli"

    ffprobe_version = "unavailable"
    ffprobe_method = "unavailable"
    ffprobe_result = _run_environment_command(("ffprobe", "-version"))
    if ffprobe_result is not None and ffprobe_result.returncode == 0:
        ffprobe_version = _bounded_environment_text(ffprobe_result.stdout)
        ffprobe_method = "ffprobe-cli"

    system_name = platform.system()
    architecture = _bounded_environment_text(platform.machine())
    cpu_model = _bounded_environment_text(
        platform.processor() or platform.machine(),
    )
    cpu_method = "platform" if cpu_model != "unavailable" else "unavailable"
    if system_name == "Darwin" and architecture.lower() in {
        "arm64",
        "aarch64",
    }:
        accelerator = "apple-silicon"
        accelerator_method = "platform"
        hardware_result = _run_environment_command(
            (
                "system_profiler",
                "SPHardwareDataType",
                "-json",
                "-detailLevel",
                "mini",
            )
        )
        try:
            hardware_payload = (
                json.loads(hardware_result.stdout)
                if hardware_result is not None and hardware_result.returncode == 0
                else None
            )
            hardware_items = hardware_payload["SPHardwareDataType"]
            hardware = hardware_items[0]
            chip = _bounded_environment_text(
                hardware.get("chip_type"),
                fallback="",
            )
            machine_model = _bounded_environment_text(
                hardware.get("machine_model"),
                fallback="",
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            chip = ""
            machine_model = ""
        if chip:
            cpu_model = _bounded_environment_text(
                f"{chip} ({machine_model})" if machine_model else chip,
            )
            accelerator = chip
            cpu_method = "system-profiler-mini"
            accelerator_method = "system-profiler-mini"
    else:
        accelerator = "unavailable"
        accelerator_method = "unavailable"

    fingerprint: dict[str, object] = {
        "python_version": platform.python_version(),
        "unicode_version": unicodedata.unidata_version,
        "os_name": _bounded_environment_text(system_name or os.name),
        "os_release": _bounded_environment_text(platform.release()),
        "architecture": architecture,
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "ram_bytes": ram_bytes,
        "cpu_model": cpu_model,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "ffprobe_version": ffprobe_version,
        "accelerator": accelerator,
        "collection_methods": {
            "cores": cores_method,
            "ram": ram_method,
            "cpu": cpu_method,
            "git": git_method,
            "ffprobe": ffprobe_method,
            "accelerator": accelerator_method,
        },
    }
    return _validate_environment_fingerprint(fingerprint)


def _validate_execution_contract_projection(
    value: object,
    *,
    expected_hash: str,
) -> dict[str, object]:
    """Validate one persisted, allowlisted execution-contract projection."""
    if not isinstance(value, dict) or set(value) != _EXECUTION_CONTRACT_FIELDS:
        raise ValueError("run execution contract has missing or unknown fields")
    contract = dict(value)
    descriptor = contract["descriptor"]
    if not isinstance(descriptor, dict) or "runtime_settings" in descriptor:
        raise ValueError("run execution contract descriptor is invalid")
    dependencies = contract["dependency_versions"]
    source_hashes = contract["source_hashes"]
    if (
        not isinstance(dependencies, dict)
        or len(dependencies) > 256
        or not isinstance(source_hashes, dict)
        or not source_hashes
        or len(source_hashes) > 256
    ):
        raise ValueError("run execution contract provenance is invalid")
    for field_name, mapping in (
        ("dependency_versions", dependencies),
        ("source_hashes", source_hashes),
    ):
        for key, item in mapping.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > 256
                or "\n" in key
                or not isinstance(item, str)
                or not item
                or len(item) > 256
                or "\n" in item
            ):
                raise ValueError(f"run execution contract {field_name} is invalid")
            if field_name == "source_hashes" and _SHA256_V1.fullmatch(item) is None:
                raise ValueError("run execution contract source hash is invalid")
    git_commit = contract["git_commit"]
    if not isinstance(git_commit, str) or _GIT_COMMIT_V1.fullmatch(git_commit) is None:
        raise ValueError("run execution contract git commit is invalid")
    contract["safe_target_settings"] = _validate_safe_target_settings(
        contract["safe_target_settings"],
    )
    if contract["scorer_version"] != SCORER_VERSION:
        raise ValueError("run execution contract scorer version is invalid")
    unicode_version = contract["unicode_version"]
    if (
        not isinstance(unicode_version, str)
        or not unicode_version
        or len(unicode_version) > 64
        or "\n" in unicode_version
    ):
        raise ValueError("run execution contract Unicode version is invalid")
    try:
        canonical = json.dumps(
            contract,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("run execution contract is not serializable") from exc
    if (
        len(canonical) > 262_144
        or _SHA256_V1.fullmatch(expected_hash) is None
        or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != expected_hash
    ):
        raise ValueError("run execution contract hash is inconsistent")
    return json.loads(canonical)


def _prepared_target_matrix(
    prepared_targets: Sequence[PreparedTarget],
) -> list[dict[str, object]]:
    """Project prepared targets into the non-secret run metadata matrix."""
    if not prepared_targets:
        raise ValueError("run metadata requires at least one target")
    matrix: list[dict[str, object]] = []
    seen: set[str] = set()
    for target in prepared_targets:
        _verify_worker_target(
            target,
            verify_local_artifact=False,
        )
        if target.target_id in seen:
            raise ValueError("run metadata target IDs must be unique")
        seen.add(target.target_id)
        contract = json.loads(
            target.execution_contract_json,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        validated_contract = _validate_execution_contract_projection(
            contract,
            expected_hash=target.execution_contract_hash,
        )
        matrix.append(
            {
                "target_id": target.target_id,
                "provider": target.provider,
                "model_label": target.model_label,
                "descriptor": validated_contract["descriptor"],
                "execution_contract": validated_contract,
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
        execution_contract = _validate_execution_contract_projection(
            target["execution_contract"],
            expected_hash=contract_hash,
        )
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
        if descriptor != execution_contract["descriptor"]:
            raise ValueError("run target descriptor and execution contract differ")
        if descriptor.get("requested_provider") != provider or descriptor.get("requested_model_label") != model_label:
            raise ValueError("run target identity and descriptor differ")
        validated.append(
            {
                "target_id": target_id,
                "provider": provider,
                "model_label": model_label,
                "descriptor": json.loads(descriptor_json),
                "execution_contract": execution_contract,
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


def atomic_create_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create one owner-only JSON artifact without replacement."""
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
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise ValueError("new run cannot resume an existing run") from exc
        temporary.unlink()
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


def _validate_reference_provenance_counts(
    value: object,
    *,
    sample_count: int,
) -> dict[str, dict[str, int]]:
    """Validate bounded selected-sample provenance counts grouped by suite."""
    if not isinstance(value, dict) or not value or len(value) > sample_count:
        raise ValueError("reference provenance counts must be a bounded object")
    if any(not isinstance(suite, str) for suite in value):
        raise ValueError("reference provenance suite must be a string")
    result: dict[str, dict[str, int]] = {}
    total = 0
    for suite in sorted(value):
        _require_stable_id(suite, "<run>", "suite")
        counts = value[suite]
        if not isinstance(counts, dict) or not counts:
            raise ValueError("reference provenance suite counts must be an object")
        if any(not isinstance(provenance, str) for provenance in counts):
            raise ValueError("reference provenance kind must be a string")
        if set(counts) - _KNOWN_REFERENCE_PROVENANCE:
            raise ValueError("reference provenance counts contain an unknown kind")
        canonical: dict[str, int] = {}
        for provenance in sorted(counts):
            count = _require_result_integer(
                counts[provenance],
                field="reference_provenance_counts",
                minimum=1,
            )
            canonical[provenance] = count
            total += count
        result[suite] = canonical
    if total != sample_count:
        raise ValueError("reference provenance counts do not match selected samples")
    return result


def _reference_provenance_counts_for_samples(
    samples: Sequence[ManifestSample],
) -> dict[str, dict[str, int]]:
    """Return deterministic per-suite provenance counts for selected samples."""
    counts: dict[str, Counter[str]] = {}
    for sample in samples:
        provenance = dict(sample.source)["reference_provenance"]
        counts.setdefault(sample.suite, Counter())[provenance] += 1
    return _validate_reference_provenance_counts(
        {suite: dict(sorted(suite_counts.items())) for suite, suite_counts in sorted(counts.items())},
        sample_count=len(samples),
    )


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
    result["reference_provenance_counts"] = _validate_reference_provenance_counts(
        result["reference_provenance_counts"],
        sample_count=len(selected),
    )
    _require_stable_id(
        result["cold_probe_sample_id"],
        "<run>",
        "cold_probe_sample_id",
    )
    if result["cold_probe_sample_id"] not in selected or not set(timing) <= set(selected):
        raise ValueError("run probe or timing sample selection is invalid")
    if result["cold_probe_sample_id"] in timing:
        raise ValueError("run cold probe cannot also be a timing sample")
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
    safe_settings_json: set[str] = set()
    for target in result["target_matrix"]:
        contract = target["execution_contract"]
        safe_settings = contract["safe_target_settings"]
        if (
            safe_settings["mode"] != result["mode"]
            or contract["git_commit"] != result["environment"]["git_commit"]
            or contract["unicode_version"] != result["environment"]["unicode_version"]
        ):
            raise ValueError("run execution contract does not match run identity")
        safe_settings_json.add(
            json.dumps(
                safe_settings,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if len(safe_settings_json) != 1:
        raise ValueError("run targets do not share common safe settings")
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
    reference_provenance_counts: Mapping[str, Mapping[str, int]],
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
    canonical_provenance_counts = _validate_reference_provenance_counts(
        reference_provenance_counts,
        sample_count=len(selected_sample_ids),
    )
    payload: dict[str, object] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "manifest_hash": manifest_hash,
        "selected_sample_ids": list(selected_sample_ids),
        "reference_provenance_counts": canonical_provenance_counts,
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
        reason = _require_result_text(item, field=field, maximum=64)
        if STABLE_ID_V1.fullmatch(reason) is None:
            raise ValueError(f"result field {field} must contain stable IDs")


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
    if result["reference_provenance"] not in _KNOWN_REFERENCE_PROVENANCE:
        raise ValueError("result reference provenance is invalid")
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
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"results line {line_number}: invalid record") from exc
        except ValueError as exc:
            detail = "unsupported schema" if "schema version" in str(exc) else "invalid record"
            raise ValueError(f"results line {line_number}: {detail}") from exc
        attempt_id = int(validated["attempt_id"])
        if attempt_id <= previous_attempt:
            raise ValueError(f"results line {line_number}: attempt IDs must increase")
        previous_attempt = attempt_id
        records.append(validated)
    return records, truncated


def load_result_history(path: Path) -> tuple[list[dict[str, object]], bool]:
    """Load validated history and ignore only an unterminated final JSONL line."""
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except FileNotFoundError:
        return [], False
    except OSError as exc:
        if exc.errno == getattr(errno, "ELOOP", -1):
            raise OSError("results path must not be a symbolic link") from exc
        raise
    with os.fdopen(descriptor, "rb") as artifact:
        if not stat.S_ISREG(os.fstat(artifact.fileno()).st_mode):
            raise OSError("results path must be a regular file")
        content = artifact.read()
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
            "p25": None,
            "p50": None,
            "p75": None,
            "iqr": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    p25 = percentile_type7(values, 0.25)
    p75 = percentile_type7(values, 0.75)
    return {
        "mean": sum(values) / len(values),
        "p25": p25,
        "p50": percentile_type7(values, 0.50),
        "p75": p75,
        "iqr": p75 - p25,
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
            record["status"] != "ok"
            or not isinstance(adapter_nanoseconds, int)
            or isinstance(adapter_nanoseconds, bool)
            or rtf is None
            or throughput is None
        ):
            ineligible_count += 1
            continue
        if record["eligibility_reasons"]:
            ineligible_count += 1
            continue
        adapter_seconds.append(adapter_nanoseconds / 1_000_000_000)
        rtfs.append(float(rtf))
        throughputs.append(float(throughput))
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


def _verify_worker_target(
    prepared_target: PreparedTarget,
    *,
    verify_local_artifact: bool = True,
) -> None:
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
    if verify_local_artifact:
        local_routes = tuple(
            route
            for route in prepared_target.plan.descriptor.routes
            if route.source == "local"
        )
        if local_routes:
            model_path = prepared_target.plan.runtime_values().get(
                "model_path"
            )
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("worker local artifact path is invalid")
            module = importlib.import_module(
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter"
            )
            identify = getattr(module, "_local_artifact_id", None)
            if not callable(identify):
                raise ValueError("worker local artifact verifier is unavailable")
            try:
                current_artifact_id = identify(model_path)
            except (OSError, TypeError, ValueError):
                raise ValueError(
                    "worker local artifact could not be verified"
                ) from None
            if any(
                not route.identity_resolved
                or route.artifact_id != current_artifact_id
                for route in local_routes
            ):
                raise ValueError(
                    "worker local artifact changed after preflight"
                )
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
    safe_settings = payload["safe_target_settings"]
    if not isinstance(safe_settings, dict) or (
        safe_settings.get("task") != prepared_target.plan.task
        or safe_settings.get("language") != prepared_target.plan.language
        or safe_settings.get("word_timestamps") != prepared_target.plan.word_timestamps
        or safe_settings.get("diarization") != prepared_target.plan.diarization
        or safe_settings.get("prompt_present") is not (prepared_target.plan.prompt is not None)
        or safe_settings.get("hotword_count") != len(prepared_target.plan.hotwords)
    ):
        raise ValueError("worker execution contract settings mismatch")
    rebuilt_json, rebuilt_hash = build_execution_contract(
        plan=prepared_target.plan,
        git_commit=payload["git_commit"],
        safe_target_settings=safe_settings,
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
    source = dict(sample.source)
    dataset = source["dataset"]
    reference_provenance = source["reference_provenance"]
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
        "reference_provenance": reference_provenance,
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
        _verify_worker_target(
            prepared_target,
            verify_local_artifact=False,
        )
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
    local_artifact_verified = False

    def run_operation(
        sample: ManifestSample,
        audio_path: str,
        *,
        repetition: int,
        operation_role: str,
        measurement_role: str | None,
        timing_class: str | None,
    ) -> str:
        nonlocal local_artifact_verified
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
        if not local_artifact_verified:
            _verify_worker_target(prepared_target)
            local_artifact_verified = True
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


def _reconcile_stale_worker_attempts(
    run_directory: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Close worker attempts left running without an in-flight operation."""
    current = validate_run_metadata(metadata)
    changed = False
    for attempt in current["worker_attempts"]:
        if attempt["status"] != "running":
            continue
        attempt["status"] = "worker_crash"
        attempt["error"] = {
            "type": "WorkerCrash",
            "message": "prior benchmark worker ended without in-flight coordination",
        }
        changed = True
    if changed:
        validate_run_metadata(current)
        atomic_write_json(run_directory / "run.json", current)
    return current


@contextmanager
def _exclusive_run_coordinator_lock(
    run_directory: Path,
) -> Iterator[None]:
    """Hold a non-blocking OS lock for one run coordinator."""
    lock_path = run_directory / ".coordinator.lock"
    if lock_path.is_symlink():
        raise OSError("coordinator lock must not be a symbolic link")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    acquired = False
    windows_lock = os.name == "nt"
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("coordinator lock must be a regular file")
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        try:
            if windows_lock:
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as exc:
            if exc.errno in {
                errno.EACCES,
                errno.EAGAIN,
                getattr(errno, "EDEADLK", errno.EAGAIN),
            }:
                raise RuntimeError("another benchmark coordinator is already active for this run") from exc
            raise
        acquired = True
        yield
    finally:
        if acquired:
            if windows_lock:
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def execute_prepared_targets(
    *,
    run_directory: Path,
    run_metadata: Mapping[str, object],
    prepared_targets: Sequence[PreparedTarget],
    samples: Sequence[ManifestSample],
    audio_paths: Sequence[str],
    retry_errors: bool,
    allow_resume: bool = True,
) -> dict[str, object]:
    """Create or resume a run and execute targets sequentially in CLI order."""
    if not isinstance(retry_errors, bool):
        raise TypeError("retry_errors must be boolean")
    if not isinstance(allow_resume, bool):
        raise TypeError("allow_resume must be boolean")
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
    with _exclusive_run_coordinator_lock(run_directory):
        run_path = run_directory / "run.json"
        if run_path.exists():
            if not allow_resume:
                raise ValueError("new run cannot resume an existing run")
            current = validate_run_metadata(_load_json_object(run_path))
            assert_resume_compatible(current, expected)
        else:
            unexpected = [path for path in run_directory.iterdir() if path.name != ".coordinator.lock"]
            if unexpected:
                raise ValueError("new run directory must be empty")
            current = expected
            if allow_resume:
                atomic_write_json(run_path, current)
            else:
                atomic_create_json(run_path, current)
        repair_result_history(run_directory / "results.jsonl")
        if (run_directory / "inflight.json").exists():
            current = _recover_persisted_inflight(
                run_directory=run_directory,
                metadata=current,
                prepared_targets=targets,
                samples=selected_samples,
                audio_paths=pinned_paths,
            )
        current = _reconcile_stale_worker_attempts(
            run_directory,
            current,
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
        record for record in validated_records if record["timing_class"] == "warm" and not record["diagnostic_only"]
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
            "diagnostic_dataset": _slice_quality_aggregates(
                diagnostic_records,
                dimension="dataset",
            ),
            "diagnostic_tag": _slice_quality_aggregates(
                diagnostic_records,
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


def _text_retention_required(
    record: Mapping[str, object],
    mode: str,
) -> bool:
    """Return whether one scored record must retain transcript text."""
    return mode == "full" or (
        mode == "errors-only"
        and (
            record["status"] != "ok"
            or any(
                int(record[profile][unit]["errors"]) > 0
                for profile in ("strict", "normalized")
                for unit in ("wer", "cer")
            )
        )
    )


def _actual_matches_declared_route(
    actual: Mapping[str, object],
    route: Mapping[str, object],
) -> bool:
    """Match actual execution using the route contract's nullable wildcards."""
    return set(route) >= _ACTUAL_EXECUTION_FIELDS and all(
        route[field] is None or route[field] == actual[field] for field in _ACTUAL_EXECUTION_FIELDS
    )


def _validate_execution_against_target(
    record: Mapping[str, object],
    target: Mapping[str, object],
    *,
    artifact: str,
) -> None:
    """Bind one actual execution and its eligibility to a declared target."""
    actual_execution = record["actual_execution"]
    if actual_execution is None:
        if "actual_execution_unverified" not in record["eligibility_reasons"]:
            raise ValueError(f"{artifact} actual execution eligibility is inconsistent")
    else:
        declared_routes = target["descriptor"].get("routes")
        matching_routes = (
            [
                route
                for route in declared_routes
                if isinstance(route, dict) and _actual_matches_declared_route(actual_execution, route)
            ]
            if isinstance(declared_routes, list)
            else []
        )
        if len(matching_routes) != 1:
            raise ValueError(f"{artifact} actual execution does not match a declared route")
        identity_unresolved = matching_routes[0].get("identity_resolved") is not True
        if ("identity_unresolved" in record["eligibility_reasons"]) != identity_unresolved:
            raise ValueError(f"{artifact} execution identity eligibility is inconsistent")
    if not set(record["execution_mismatch_reasons"]) <= set(record["eligibility_reasons"]):
        raise ValueError(f"{artifact} execution mismatch eligibility is inconsistent")


def _validate_report_records(
    metadata: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Validate that result identities and repetitions belong to one run."""
    targets = {str(target["target_id"]): target for target in metadata["target_matrix"]}
    selected_samples = {str(value) for value in metadata["selected_sample_ids"]}
    timing_samples = {str(value) for value in metadata["timing_sample_ids"]}
    warm_repetitions = int(metadata["warm_repetitions"])
    validated: list[dict[str, object]] = []
    for record in records:
        result = _validate_result_record(record)
        target = targets.get(str(result["target_id"]))
        sample_id = str(result["sample_id"])
        repetition = int(result["repetition"])
        if result["run_id"] != metadata["run_id"] or target is None or sample_id not in selected_samples:
            raise ValueError("result does not belong to the reported run")
        maximum_repetition = warm_repetitions - 1 if sample_id in timing_samples else 0
        if repetition > maximum_repetition:
            raise ValueError("result repetition does not belong to the reported run")
        expected_role = "accuracy" if repetition == 0 else "performance_repeat"
        if result["measurement_role"] != expected_role:
            raise ValueError("result measurement role does not match repetition")
        allowed_timing = (
            {"cold_first"} if sample_id == metadata["cold_probe_sample_id"] else {"warm", "warmup_recovery"}
        )
        if result["timing_class"] not in allowed_timing:
            raise ValueError("result timing class does not match run schedule")
        expected_key = completion_key(
            str(metadata["manifest_hash"]),
            str(target["target_id"]),
            str(target["execution_contract_hash"]),
            sample_id,
            repetition,
        )
        if result["completion_key"] != expected_key:
            raise ValueError("result completion key does not belong to the reported run")
        if result["requested_execution"] != {
            "provider": target["provider"],
            "model_label": target["model_label"],
        }:
            raise ValueError("result requested execution does not match target")
        _validate_execution_against_target(
            result,
            target,
            artifact="result",
        )
        retention_required = _text_retention_required(
            result,
            str(metadata["text_retention"]),
        )
        if (result["reference"] is not None) != retention_required:
            raise ValueError("result text retention does not match the reported run")
        validated.append(result)
    return validated


def _summary_sample(record: Mapping[str, object]) -> dict[str, object]:
    """Project one validated active result into paired-comparison data."""
    adapter_nanoseconds = record["adapter_nanoseconds"]
    return {
        "target_id": record["target_id"],
        "sample_id": record["sample_id"],
        "repetition": record["repetition"],
        "measurement_role": record["measurement_role"],
        "timing_class": record["timing_class"],
        "suite": record["suite"],
        "suite_visibility": record["suite_visibility"],
        "dataset": record["dataset"],
        "reference_provenance": record["reference_provenance"],
        "tags": list(record["tags"]),
        "diagnostic_only": record["diagnostic_only"],
        "status": record["status"],
        "exact_match": record["exact_match"],
        "strict": record["strict"],
        "normalized": record["normalized"],
        "adapter_seconds": (
            adapter_nanoseconds / 1_000_000_000
            if isinstance(adapter_nanoseconds, int) and not isinstance(adapter_nanoseconds, bool)
            else None
        ),
        "audio_duration_seconds": record["audio_duration_seconds"],
        "rtf": record["rtf"],
        "throughput": record["throughput"],
        "actual_execution": record["actual_execution"],
        "execution_mismatch_reasons": list(record["execution_mismatch_reasons"]),
        "eligibility_reasons": list(record["eligibility_reasons"]),
        "normalization_profile": record["normalization_profile"],
        "adapter_nanoseconds": adapter_nanoseconds,
        "resource_observations": record["resource_observations"],
        "error": record["error"],
        "reference": record["reference"],
        "hypothesis": record["hypothesis"],
    }


def _report_identity(metadata: Mapping[str, object]) -> dict[str, object]:
    """Project immutable run identity and safe provenance into a summary."""
    return {
        "manifest_hash": metadata["manifest_hash"],
        "selected_sample_ids": list(metadata["selected_sample_ids"]),
        "reference_provenance_counts": {
            suite: dict(counts) for suite, counts in metadata["reference_provenance_counts"].items()
        },
        "profile": metadata["profile"],
        "mode": metadata["mode"],
        "seed": metadata["seed"],
        "cold_probe_sample_id": metadata["cold_probe_sample_id"],
        "warm_repetitions": metadata["warm_repetitions"],
        "timing_sample_ids": list(metadata["timing_sample_ids"]),
        "text_retention": metadata["text_retention"],
        "adapter_watchdog_seconds": metadata["adapter_watchdog_seconds"],
        "scorer_version": SCORER_VERSION,
        "strict_profile": STRICT_PROFILE,
        "unicode_version": metadata["environment"]["unicode_version"],
        "target_order": [target["target_id"] for target in metadata["target_matrix"]],
        "targets": list(metadata["target_matrix"]),
        "environment": dict(metadata["environment"]),
    }


def _report_eligibility(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Report explicit reasons plus mixed actual-execution identities."""
    counts: dict[str, Counter[str]] = {}
    signatures: dict[str, set[str]] = {}
    for record in records:
        target_id = str(record["target_id"])
        target_counts = counts.setdefault(target_id, Counter())
        for reason in record["eligibility_reasons"]:
            target_counts[str(reason)] += 1
        execution = record["actual_execution"]
        signature = (
            "unavailable"
            if execution is None
            else json.dumps(
                execution,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        signatures.setdefault(target_id, set()).add(signature)
    target_ids = sorted(set(counts) | set(signatures))
    return {
        "reason_counts": {
            target_id: dict(sorted(counts.get(target_id, Counter()).items())) for target_id in target_ids
        },
        "targets": {
            target_id: {
                "actual_execution_signature_count": len(signatures.get(target_id, set())),
                "mixed_actual_execution": len(signatures.get(target_id, set())) > 1,
                "unverified_actual_execution": ("unavailable" in signatures.get(target_id, set())),
            }
            for target_id in target_ids
        },
    }


def _worst_retained_examples(
    records: Sequence[Mapping[str, object]],
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Return worst accuracy examples only when their text was retained."""
    eligible = [
        record
        for record in records
        if record["measurement_role"] == "accuracy"
        and record["reference"] is not None
        and record["hypothesis"] is not None
        and (
            record["status"] != "ok"
            or float(record["strict"]["wer"]["rate"]) > 0.0
            or float(record["normalized"]["wer"]["rate"]) > 0.0
        )
    ]
    eligible.sort(
        key=lambda record: (
            -float(record["normalized"]["wer"]["rate"]),
            -float(record["strict"]["wer"]["rate"]),
            str(record["target_id"]),
            str(record["sample_id"]),
        )
    )
    return [
        {
            "target_id": record["target_id"],
            "sample_id": record["sample_id"],
            "suite": record["suite"],
            "status": record["status"],
            "strict_wer": record["strict"]["wer"]["rate"],
            "normalized_wer": record["normalized"]["wer"]["rate"],
            "reference": record["reference"],
            "hypothesis": record["hypothesis"],
        }
        for record in eligible[:limit]
    ]


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace one owner-only UTF-8 text artifact."""
    if not isinstance(text, str):
        raise TypeError("text artifact must be a string")
    destination = Path(path)
    _ensure_owner_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(text)
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


def _quality_status_table_rows(
    summary: Mapping[str, object],
    *,
    section: str,
) -> list[list[str]]:
    """Return per-suite status-rate rows for one quality population."""
    rows: list[list[str]] = []
    targets = summary[section]["targets"]
    for target_id in summary["identity"]["target_order"]:
        target = targets.get(target_id, {"suites": {}})
        for suite, metrics in target["suites"].items():
            rows.append(
                [
                    str(target_id),
                    str(suite),
                    str(metrics["sample_count"]),
                    f"{float(metrics['success_rate']):.6f}",
                    f"{float(metrics['empty_rate']):.6f}",
                    f"{float(metrics['failure_rate']):.6f}",
                    f"{float(metrics['error_rate']):.6f}",
                    f"{float(metrics['exact_match_rate']):.6f}",
                ]
            )
    return rows


def _quality_distribution_table_rows(
    summary: Mapping[str, object],
    *,
    section: str,
) -> list[list[str]]:
    """Return pooled, mean, and tail error-rate rows."""
    rows: list[list[str]] = []
    targets = summary[section]["targets"]
    for target_id in summary["identity"]["target_order"]:
        target = targets.get(target_id, {"suites": {}})
        for suite, metrics in target["suites"].items():
            for profile in ("strict", "normalized"):
                for unit in ("wer", "cer"):
                    distribution = metrics[profile][unit]
                    rows.append(
                        [
                            str(target_id),
                            str(suite),
                            f"{profile.title()} {unit.upper()}",
                            _metric_text(distribution["pooled"]),
                            _metric_text(distribution["mean"]),
                            _metric_text(distribution["p50"]),
                            _metric_text(distribution["p90"]),
                            _metric_text(distribution["p95"]),
                            _metric_text(distribution["p99"]),
                        ]
                    )
    return rows


def _slice_table_rows(
    summary: Mapping[str, object],
    *,
    dimension: str,
) -> list[list[str]]:
    """Return one human-readable dataset or tag slice table."""
    rows: list[list[str]] = []
    slices = summary["slices"][dimension]
    for target_id in summary["identity"]["target_order"]:
        for value, suites in slices.get(target_id, {}).items():
            for suite, metrics in suites.items():
                rows.append(
                    [
                        str(target_id),
                        str(value),
                        str(suite),
                        str(metrics["sample_count"]),
                        _metric_text(metrics["normalized"]["wer"]["pooled"]),
                        _metric_text(metrics["failure_rate"]),
                    ]
                )
    return rows


def _bounded_display_text(value: object, *, maximum: int = 160) -> str:
    """Collapse controls and bound retained transcript text for display."""
    display_tokens: list[str] = []
    pending_space = False
    for character in str(value):
        code_point = ord(character)
        if unicodedata.category(character) in {"Cc", "Cf", "Cs"}:
            if pending_space and display_tokens:
                display_tokens.append(" ")
            pending_space = False
            display_tokens.append(f"\\u{code_point:04x}" if code_point <= 0xFFFF else f"\\U{code_point:08x}")
        elif character.isspace():
            pending_space = bool(display_tokens)
        else:
            if pending_space:
                display_tokens.append(" ")
                pending_space = False
            display_tokens.append(character)
    text = "".join(display_tokens)
    if len(text) <= maximum:
        return text
    budget = maximum - 1
    bounded: list[str] = []
    used = 0
    for token in display_tokens:
        if used + len(token) > budget:
            break
        bounded.append(token)
        used += len(token)
    return "".join(bounded).rstrip() + "…"


def _worst_example_table_rows(
    summary: Mapping[str, object],
    *,
    markdown: bool,
) -> list[list[str]]:
    """Return bounded, escaped retained examples for a human report."""
    rows: list[list[str]] = []
    for example in summary["worst_examples"]:
        reference = _bounded_display_text(example["reference"])
        hypothesis = _bounded_display_text(example["hypothesis"])
        if markdown:
            reference = (
                html.escape(
                    reference,
                    quote=True,
                )
                .replace("\\", "\\\\")
                .replace("|", "\\|")
            )
            hypothesis = (
                html.escape(
                    hypothesis,
                    quote=True,
                )
                .replace("\\", "\\\\")
                .replace("|", "\\|")
            )
        else:
            reference = reference.replace("|", "¦")
            hypothesis = hypothesis.replace("|", "¦")
        rows.append(
            [
                str(example["target_id"]),
                str(example["sample_id"]),
                str(example["suite"]),
                str(example["status"]),
                _metric_text(example["strict_wer"]),
                _metric_text(example["normalized_wer"]),
                reference,
                hypothesis,
            ]
        )
    return rows


def _target_table_rows(summary: Mapping[str, object]) -> list[list[str]]:
    """Return the provider/model legend in target execution order."""
    targets = {str(target["target_id"]): target for target in summary["identity"]["targets"]}
    return [
        [
            str(target_id),
            str(targets[target_id]["provider"]),
            str(targets[target_id]["model_label"]),
        ]
        for target_id in summary["identity"]["target_order"]
    ]


def _metric_text(value: object) -> str:
    """Format one optional report metric consistently."""
    return "n/a" if value is None else f"{float(value):.6f}"


def _warm_table_rows(summary: Mapping[str, object]) -> list[list[str]]:
    """Return shared per-suite warm performance rows."""
    rows: list[list[str]] = []
    targets = summary["performance"]["warm"]["targets"]
    for target_id in summary["identity"]["target_order"]:
        for suite, metrics in targets.get(target_id, {"suites": {}})["suites"].items():
            rows.append(
                [
                    str(target_id),
                    str(suite),
                    str(metrics["candidate_count"]),
                    str(metrics["observation_count"]),
                    _metric_text(metrics["adapter_seconds"]["p50"]),
                    _metric_text(metrics["adapter_seconds"]["iqr"]),
                    _metric_text(metrics["rtf"]["p50"]),
                    _metric_text(metrics["rtf"]["iqr"]),
                    _metric_text(metrics["throughput"]["p50"]),
                    _metric_text(metrics["throughput"]["iqr"]),
                ]
            )
    return rows


def _cold_table_rows(summary: Mapping[str, object]) -> list[list[str]]:
    """Return shared target-level cold-first rows."""
    cold = summary["performance"]["cold_first"]
    return [
        [
            str(target_id),
            str(cold[target_id]["sample_id"]),
            str(cold[target_id]["status"]),
            _metric_text(cold[target_id]["adapter_seconds"]),
            _metric_text(cold[target_id]["rtf"]),
            _metric_text(cold[target_id]["throughput"]),
            str(bool(cold[target_id]["gate_eligible"])).lower(),
        ]
        for target_id in summary["identity"]["target_order"]
        if target_id in cold
    ]


def _backend_table_rows(summary: Mapping[str, object]) -> list[list[str]]:
    """Return shared actual-backend population rows."""
    rows: list[list[str]] = []
    slices = summary["slices"]["actual_backend"]
    for target_id in summary["identity"]["target_order"]:
        for backend, suites in slices.get(target_id, {}).items():
            for suite, metrics in suites.items():
                rows.append(
                    [
                        str(target_id),
                        str(backend),
                        str(suite),
                        str(metrics["sample_count"]),
                    ]
                )
    return rows


def _eligibility_table_rows(summary: Mapping[str, object]) -> list[list[str]]:
    """Return shared target-level gate-eligibility diagnostics."""
    rows: list[list[str]] = []
    eligibility = summary["eligibility"]
    for target_id in summary["identity"]["target_order"]:
        target = eligibility["targets"].get(
            target_id,
            {
                "actual_execution_signature_count": 0,
                "mixed_actual_execution": False,
                "unverified_actual_execution": False,
            },
        )
        reason_counts = eligibility["reason_counts"].get(target_id, {})
        reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
        rows.append(
            [
                str(target_id),
                str(target["actual_execution_signature_count"]),
                str(bool(target["mixed_actual_execution"])).lower(),
                str(bool(target["unverified_actual_execution"])).lower(),
                reasons or "none",
            ]
        )
    return rows


def _reference_provenance_text(summary: Mapping[str, object]) -> str:
    """Render immutable selected-sample provenance counts compactly."""
    suites = summary["identity"]["reference_provenance_counts"]
    return "; ".join(
        (f"{suite}[" + ", ".join(f"{provenance}={count}" for provenance, count in sorted(counts.items())) + "]")
        for suite, counts in sorted(suites.items())
    )


def render_summary_markdown(summary: Mapping[str, object]) -> str:
    """Render one summary dictionary as deterministic Markdown."""
    progress = summary["progress"]
    identity = summary["identity"]
    environment = identity["environment"]
    lines = [
        "# Native STT Benchmark Report",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Profile/mode: `{identity['profile']}` / `{identity['mode']}`",
        (f"- Scorer/Unicode: `{identity['scorer_version']}` / `{identity['unicode_version']}`"),
        f"- Reference provenance: {_reference_provenance_text(summary)}",
        (
            "- Hardware: "
            f"`{environment['cpu_model']}`; `{environment['architecture']}`; "
            f"accelerator `{environment['accelerator']}`"
        ),
        (
            "- Progress: "
            f"{progress['active_result_count']}/{progress['expected_result_count']} "
            f"active; {progress['pending_result_count']} pending"
        ),
        "",
        "## Targets",
        "",
        "| Target | Provider | Model |",
        "|---|---|---|",
    ]
    for row in _target_table_rows(summary):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Quality",
            "",
            ("| Target | Suite | Samples | Success rate | Empty rate | Failure rate | Error rate | Exact match |"),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _quality_status_table_rows(summary, section="primary"):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "| Target | Suite | Metric | Pooled | Mean | p50 | p90 | p95 | p99 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _quality_distribution_table_rows(summary, section="primary"):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Diagnostic-only aggregates",
            "",
            ("| Target | Suite | Samples | Success rate | Empty rate | Failure rate | Error rate | Exact match |"),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _quality_status_table_rows(summary, section="diagnostic"):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "| Target | Suite | Metric | Pooled | Mean | p50 | p90 | p95 | p99 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _quality_distribution_table_rows(summary, section="diagnostic"):
        lines.append("| " + " | ".join(row) + " |")
    for title, dimension in (
        ("Dataset slices", "dataset"),
        ("Tag slices", "tag"),
        ("Diagnostic dataset slices", "diagnostic_dataset"),
        ("Diagnostic tag slices", "diagnostic_tag"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Target | Value | Suite | Samples | Normalized WER | Failure rate |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in _slice_table_rows(summary, dimension=dimension):
            lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Warm performance",
            "",
            (
                "| Target | Suite | Candidates | Observations | Latency p50 | "
                "Latency IQR | RTF p50 | RTF IQR | Throughput p50 | Throughput IQR |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _warm_table_rows(summary):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Cold-first observations",
            "",
            "| Target | Sample | Status | Latency | RTF | Throughput | Gate eligible |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in _cold_table_rows(summary):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Actual backend populations",
            "",
            "| Target | Backend | Suite | Samples |",
            "|---|---|---|---:|",
        ]
    )
    for row in _backend_table_rows(summary):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Gate eligibility",
            "",
            "| Target | Execution signatures | Mixed | Unverified | Reasons |",
            "|---|---:|---|---|---|",
        ]
    )
    for row in _eligibility_table_rows(summary):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Worst retained examples",
            "",
            ("| Target | Sample | Suite | Status | Strict WER | Normalized WER | Reference | Hypothesis |"),
            "|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in _worst_example_table_rows(summary, markdown=True):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_summary_terminal(summary: Mapping[str, object]) -> str:
    """Render one summary dictionary as deterministic terminal text."""
    progress = summary["progress"]
    identity = summary["identity"]
    environment = identity["environment"]
    lines = [
        f"Native STT benchmark: {summary['run_id']}",
        f"Profile/mode: {identity['profile']} / {identity['mode']}",
        (f"Scorer/Unicode: {identity['scorer_version']} / {identity['unicode_version']}"),
        f"Reference provenance: {_reference_provenance_text(summary)}",
        (
            "Hardware: "
            f"{environment['cpu_model']}; {environment['architecture']}; "
            f"accelerator {environment['accelerator']}"
        ),
        (
            "Progress: "
            f"{progress['active_result_count']}/{progress['expected_result_count']} "
            f"active, {progress['pending_result_count']} pending"
        ),
        "Targets:",
        "target | provider | model",
    ]
    lines.extend(" | ".join(row) for row in _target_table_rows(summary))
    lines.append("target | suite | samples | success rate | empty rate | failure rate | error rate | exact match")
    lines.extend(
        " | ".join(row)
        for row in _quality_status_table_rows(
            summary,
            section="primary",
        )
    )
    lines.append("target | suite | metric | pooled | mean | p50 | p90 | p95 | p99")
    lines.extend(
        " | ".join(row)
        for row in _quality_distribution_table_rows(
            summary,
            section="primary",
        )
    )
    lines.extend(
        [
            "Diagnostic-only aggregates:",
            "target | suite | samples | success rate | empty rate | failure rate | error rate | exact match",
        ]
    )
    lines.extend(
        " | ".join(row)
        for row in _quality_status_table_rows(
            summary,
            section="diagnostic",
        )
    )
    lines.append("target | suite | metric | pooled | mean | p50 | p90 | p95 | p99")
    lines.extend(
        " | ".join(row)
        for row in _quality_distribution_table_rows(
            summary,
            section="diagnostic",
        )
    )
    for title, dimension in (
        ("Dataset slices", "dataset"),
        ("Tag slices", "tag"),
        ("Diagnostic dataset slices", "diagnostic_dataset"),
        ("Diagnostic tag slices", "diagnostic_tag"),
    ):
        lines.extend(
            [
                f"{title}:",
                "target | value | suite | samples | normalized WER | failure rate",
            ]
        )
        lines.extend(
            " | ".join(row)
            for row in _slice_table_rows(
                summary,
                dimension=dimension,
            )
        )
    lines.extend(
        [
            "Warm performance:",
            (
                "target | suite | candidates | observations | latency p50 | "
                "latency IQR | RTF p50 | RTF IQR | throughput p50 | throughput IQR"
            ),
        ]
    )
    lines.extend(" | ".join(row) for row in _warm_table_rows(summary))
    lines.extend(
        [
            "Cold-first observations:",
            "target | sample | status | latency | RTF | throughput | gate eligible",
        ]
    )
    lines.extend(" | ".join(row) for row in _cold_table_rows(summary))
    lines.extend(
        [
            "Actual backend populations:",
            "target | backend | suite | samples",
        ]
    )
    lines.extend(" | ".join(row) for row in _backend_table_rows(summary))
    lines.extend(
        [
            "Gate eligibility:",
            "target | execution signatures | mixed | unverified | reasons",
        ]
    )
    lines.extend(" | ".join(row) for row in _eligibility_table_rows(summary))
    lines.extend(
        [
            "Worst retained examples:",
            ("target | sample | suite | status | strict WER | normalized WER | reference | hypothesis"),
        ]
    )
    lines.extend(
        " | ".join(row)
        for row in _worst_example_table_rows(
            summary,
            markdown=False,
        )
    )
    return "\n".join(lines) + "\n"


def generate_report(run_directory: Path) -> dict[str, object]:
    """Regenerate disposable report artifacts from durable run state."""
    directory = Path(run_directory)
    metadata = validate_run_metadata(_load_json_object(directory / "run.json"))
    history, truncated = load_result_history(directory / "results.jsonl")
    records = _validate_report_records(metadata, history)
    active = reduce_attempts(records)
    summary = aggregate_results(metadata, active)
    target_order = [target["target_id"] for target in metadata["target_matrix"]]
    sample_order = list(metadata["selected_sample_ids"])
    ordered_records = sorted(
        active.values(),
        key=lambda record: (
            target_order.index(record["target_id"]),
            sample_order.index(record["sample_id"]),
            int(record["repetition"]),
        ),
    )
    expected_count = len(metadata["target_matrix"]) * (
        len(metadata["selected_sample_ids"])
        + len(metadata["timing_sample_ids"]) * (int(metadata["warm_repetitions"]) - 1)
    )
    active_count = len(active)
    if active_count > expected_count:
        raise ValueError("active result count exceeds run result matrix")
    summary.update(
        {
            "identity": _report_identity(metadata),
            "progress": {
                "expected_result_count": expected_count,
                "active_result_count": active_count,
                "pending_result_count": expected_count - active_count,
                "complete": active_count == expected_count,
                "history_truncated_tail_ignored": truncated,
            },
            "eligibility": _report_eligibility(ordered_records),
            "samples": [_summary_sample(record) for record in ordered_records],
            "worst_examples": _worst_retained_examples(ordered_records),
        }
    )
    summary = validate_summary(summary)
    atomic_write_json(directory / "summary.json", summary)
    _atomic_write_text(
        directory / "summary.md",
        render_summary_markdown(summary),
    )
    return summary


_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "active_result_count",
        "primary",
        "diagnostic",
        "slices",
        "performance",
        "identity",
        "progress",
        "eligibility",
        "samples",
        "worst_examples",
    }
)
_SUMMARY_IDENTITY_FIELDS = frozenset(
    {
        "manifest_hash",
        "selected_sample_ids",
        "reference_provenance_counts",
        "profile",
        "mode",
        "seed",
        "cold_probe_sample_id",
        "warm_repetitions",
        "timing_sample_ids",
        "text_retention",
        "adapter_watchdog_seconds",
        "scorer_version",
        "strict_profile",
        "unicode_version",
        "target_order",
        "targets",
        "environment",
    }
)
_SUMMARY_SAMPLE_FIELDS = frozenset(
    {
        "target_id",
        "sample_id",
        "repetition",
        "measurement_role",
        "timing_class",
        "suite",
        "suite_visibility",
        "dataset",
        "reference_provenance",
        "tags",
        "diagnostic_only",
        "status",
        "exact_match",
        "strict",
        "normalized",
        "adapter_seconds",
        "adapter_nanoseconds",
        "audio_duration_seconds",
        "rtf",
        "throughput",
        "actual_execution",
        "execution_mismatch_reasons",
        "eligibility_reasons",
        "normalization_profile",
        "resource_observations",
        "error",
        "reference",
        "hypothesis",
    }
)


def _validate_summary_sample(value: object) -> dict[str, object]:
    """Validate one comparison-safe per-sample summary projection."""
    if not isinstance(value, dict) or set(value) != _SUMMARY_SAMPLE_FIELDS:
        raise ValueError("summary sample has missing or unknown fields")
    sample = dict(value)
    for field in ("target_id", "sample_id", "suite", "dataset"):
        _require_stable_id(sample[field], "<summary>", field)
    if sample["reference_provenance"] not in _KNOWN_REFERENCE_PROVENANCE:
        raise ValueError("summary sample reference provenance is invalid")
    repetition = _require_result_integer(
        sample["repetition"],
        field="repetition",
        minimum=0,
    )
    if sample["measurement_role"] not in MEASUREMENT_ROLES:
        raise ValueError("summary sample measurement role is invalid")
    if sample["timing_class"] not in TIMING_CLASSES:
        raise ValueError("summary sample timing class is invalid")
    if sample["suite_visibility"] not in {"public", "private"}:
        raise ValueError("summary sample suite visibility is invalid")
    if not isinstance(sample["diagnostic_only"], bool):
        raise ValueError("summary sample diagnostic flag is invalid")
    tags = sample["tags"]
    if not isinstance(tags, list) or len(tags) > MAX_TAGS_PER_SAMPLE or len(tags) != len(set(tags)):
        raise ValueError("summary sample tags are invalid")
    for tag in tags:
        _require_stable_id(tag, "<summary>", "tag")
    if sample["status"] not in RESULT_STATUSES:
        raise ValueError("summary sample status is invalid")
    if not isinstance(sample["exact_match"], bool):
        raise ValueError("summary sample exact-match flag is invalid")
    _validate_score_mapping(sample["strict"], "summary.strict")
    _validate_score_mapping(sample["normalized"], "summary.normalized")
    if sample["normalization_profile"] not in _KNOWN_NORMALIZATION_PROFILES:
        raise ValueError("summary sample normalization profile is invalid")
    _validate_execution_mapping(
        sample["actual_execution"],
        field="summary.actual_execution",
        actual=True,
    )
    _validate_reason_list(
        sample["execution_mismatch_reasons"],
        "summary.execution_mismatch_reasons",
    )
    _validate_reason_list(
        sample["eligibility_reasons"],
        "summary.eligibility_reasons",
    )
    adapter_nanoseconds = sample["adapter_nanoseconds"]
    if adapter_nanoseconds is not None and (
        isinstance(adapter_nanoseconds, bool) or not isinstance(adapter_nanoseconds, int)
    ):
        raise ValueError("summary adapter duration is invalid")
    expected_seconds = (
        adapter_nanoseconds / 1_000_000_000
        if isinstance(adapter_nanoseconds, int) and not isinstance(adapter_nanoseconds, bool)
        else None
    )
    if sample["adapter_seconds"] != expected_seconds:
        raise ValueError("summary adapter seconds are inconsistent")
    audio_duration = sample["audio_duration_seconds"]
    if audio_duration is not None and (
        isinstance(audio_duration, bool)
        or not isinstance(audio_duration, (int, float))
        or not math.isfinite(float(audio_duration))
    ):
        raise ValueError("summary audio duration is invalid")
    base_reasons = [reason for reason in sample["eligibility_reasons"] if reason != "invalid_performance_duration"]
    expected_rtf, expected_throughput, expected_reasons = performance_fields(
        adapter_nanoseconds,
        audio_duration,
        eligibility_reasons=base_reasons,
    )
    if sample["eligibility_reasons"] != expected_reasons:
        raise ValueError("summary performance eligibility is inconsistent")
    for actual, expected in (
        (sample["rtf"], expected_rtf),
        (sample["throughput"], expected_throughput),
    ):
        if (actual is None) != (expected is None) or (
            actual is not None
            and expected is not None
            and not math.isclose(
                float(actual),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("summary performance value is inconsistent")
    reference = sample["reference"]
    hypothesis = sample["hypothesis"]
    if (reference is None) != (hypothesis is None):
        raise ValueError("summary retained text is inconsistent")
    if reference is not None and hypothesis is not None:
        _require_result_text(reference, field="summary.reference", maximum=1_000_000)
        _require_result_text(
            hypothesis,
            field="summary.hypothesis",
            maximum=1_000_000,
            allow_empty=True,
        )
        expected_score = _score_as_result_fields(
            score_result_text(
                reference,
                hypothesis,
                status=str(sample["status"]),
                normalization_profile=str(sample["normalization_profile"]),
            )
        )
        for field in ("exact_match", "strict", "normalized"):
            if sample[field] != expected_score[field]:
                raise ValueError("summary score is inconsistent with retained text")
    observations = sample["resource_observations"]
    if observations is not None:
        if (
            not isinstance(observations, dict)
            or set(observations) - _RESOURCE_OBSERVATION_FIELDS
            or "collection_method" not in observations
        ):
            raise ValueError("summary resource observations are invalid")
        _require_result_text(
            observations["collection_method"],
            field="summary.resource_observations.collection_method",
            maximum=128,
        )
        for field, item in observations.items():
            if field == "collection_method" or item is None:
                continue
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError("summary resource observation is invalid")
    error = sample["error"]
    if error is not None:
        if not isinstance(error, dict) or set(error) != {"type", "message"}:
            raise ValueError("summary error is invalid")
        _require_result_text(error["type"], field="summary.error.type", maximum=128)
        _require_result_text(
            error["message"],
            field="summary.error.message",
            maximum=512,
        )
    if repetition == 0 and sample["measurement_role"] != "accuracy":
        raise ValueError("summary sample role is inconsistent")
    if repetition > 0 and sample["measurement_role"] != "performance_repeat":
        raise ValueError("summary sample role is inconsistent")
    return sample


def _aggregate_summary_samples(
    *,
    run_id: str,
    cold_probe_sample_id: str,
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute every headline aggregate from validated summary samples."""
    quality = [sample for sample in samples if sample["measurement_role"] == "accuracy"]
    primary = [sample for sample in quality if not sample["diagnostic_only"]]
    diagnostic = [sample for sample in quality if sample["diagnostic_only"]]
    warm = [sample for sample in samples if sample["timing_class"] == "warm" and not sample["diagnostic_only"]]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "active_result_count": len(samples),
        "primary": {"targets": _target_quality_aggregates(primary)},
        "diagnostic": {"targets": _target_quality_aggregates(diagnostic)},
        "slices": {
            "dataset": _slice_quality_aggregates(primary, dimension="dataset"),
            "tag": _slice_quality_aggregates(primary, dimension="tag"),
            "diagnostic_dataset": _slice_quality_aggregates(
                diagnostic,
                dimension="dataset",
            ),
            "diagnostic_tag": _slice_quality_aggregates(
                diagnostic,
                dimension="tag",
            ),
            "actual_backend": _slice_quality_aggregates(
                primary,
                dimension="actual_backend",
            ),
        },
        "performance": {
            "warm": {"targets": _target_warm_performance(warm)},
            "cold_first": _cold_first_observations(
                samples,
                cold_probe_sample_id=cold_probe_sample_id,
            ),
        },
    }


def _validate_observed_reference_provenance(
    samples: Sequence[Mapping[str, object]],
    *,
    declared: Mapping[str, Mapping[str, int]],
    complete: bool,
) -> None:
    """Reconcile per-sample result provenance with immutable selected counts."""
    layouts: dict[str, tuple[str, str]] = {}
    observed: dict[str, Counter[str]] = {}
    counted_accuracy_ids: set[str] = set()
    for sample in samples:
        sample_id = str(sample["sample_id"])
        layout = (
            str(sample["suite"]),
            str(sample["reference_provenance"]),
        )
        previous = layouts.setdefault(sample_id, layout)
        if previous != layout:
            raise ValueError("summary sample reference provenance is inconsistent")
        if sample["measurement_role"] == "accuracy" and sample_id not in counted_accuracy_ids:
            observed.setdefault(layout[0], Counter())[layout[1]] += 1
            counted_accuracy_ids.add(sample_id)
    canonical_observed = {suite: dict(sorted(counts.items())) for suite, counts in sorted(observed.items())}
    for suite, counts in canonical_observed.items():
        if suite not in declared:
            raise ValueError("summary provenance suite is not declared")
        for provenance, count in counts.items():
            if count > declared[suite].get(provenance, 0):
                raise ValueError("summary provenance count exceeds the run identity")
    if complete and canonical_observed != declared:
        raise ValueError("summary provenance counts do not match the run identity")


def validate_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Strictly validate and reconcile one disposable summary artifact."""
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_FIELDS:
        raise ValueError("summary has missing or unknown fields")
    result = dict(summary)
    if result["schema_version"] != SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported summary schema version")
    run_id = _require_stable_id(result["run_id"], "<summary>", "run_id")
    identity = result["identity"]
    if not isinstance(identity, dict) or set(identity) != _SUMMARY_IDENTITY_FIELDS:
        raise ValueError("summary identity has missing or unknown fields")
    manifest_hash = identity["manifest_hash"]
    if not isinstance(manifest_hash, str) or _SHA256_V1.fullmatch(manifest_hash) is None:
        raise ValueError("summary manifest hash is invalid")
    selected = identity["selected_sample_ids"]
    timing = identity["timing_sample_ids"]
    target_order = identity["target_order"]
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or not isinstance(timing, list)
        or len(timing) != len(set(timing))
        or not isinstance(target_order, list)
        or not target_order
        or len(target_order) != len(set(target_order))
    ):
        raise ValueError("summary ordered identities are invalid")
    for sample_id in (*selected, *timing):
        _require_stable_id(sample_id, "<summary>", "sample_id")
    provenance_counts = _validate_reference_provenance_counts(
        identity["reference_provenance_counts"],
        sample_count=len(selected),
    )
    identity["reference_provenance_counts"] = provenance_counts
    cold_probe = _require_stable_id(
        identity["cold_probe_sample_id"],
        "<summary>",
        "cold_probe_sample_id",
    )
    if cold_probe not in selected or cold_probe in timing or not set(timing) <= set(selected):
        raise ValueError("summary sample schedule is invalid")
    if identity["profile"] not in _KNOWN_SAMPLE_PROFILES:
        raise ValueError("summary profile is invalid")
    if identity["mode"] not in {"neutral-v1", "production-v1"}:
        raise ValueError("summary mode is invalid")
    _require_result_integer(identity["seed"], field="seed", minimum=0)
    warm_repetitions = _require_result_integer(
        identity["warm_repetitions"],
        field="warm_repetitions",
        minimum=1,
    )
    if identity["text_retention"] not in {"full", "errors-only", "none"}:
        raise ValueError("summary text retention is invalid")
    watchdog = identity["adapter_watchdog_seconds"]
    if watchdog is not None and (
        isinstance(watchdog, bool)
        or not isinstance(watchdog, (int, float))
        or not math.isfinite(float(watchdog))
        or float(watchdog) <= 0.0
    ):
        raise ValueError("summary watchdog is invalid")
    if identity["scorer_version"] != SCORER_VERSION or identity["strict_profile"] != STRICT_PROFILE:
        raise ValueError("summary scorer identity is invalid")
    unicode_version = identity["unicode_version"]
    if (
        not isinstance(unicode_version, str)
        or not unicode_version
        or len(unicode_version) > 64
        or "\n" in unicode_version
    ):
        raise ValueError("summary Unicode identity is invalid")
    targets = _validate_target_matrix(identity["targets"])
    if target_order != [target["target_id"] for target in targets]:
        raise ValueError("summary target order is inconsistent")
    targets_by_id = {str(target["target_id"]): target for target in targets}
    environment = _validate_environment_fingerprint(identity["environment"])
    if environment["unicode_version"] != unicode_version:
        raise ValueError("summary Unicode identity is inconsistent")
    safe_settings_json: set[str] = set()
    for target in targets:
        contract = target["execution_contract"]
        if (
            contract["scorer_version"] != identity["scorer_version"]
            or contract["unicode_version"] != unicode_version
            or contract["git_commit"] != environment["git_commit"]
            or contract["safe_target_settings"]["mode"] != identity["mode"]
        ):
            raise ValueError("summary execution contract identity is inconsistent")
        safe_settings_json.add(
            json.dumps(
                contract["safe_target_settings"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if len(safe_settings_json) != 1:
        raise ValueError("summary targets do not share common safe settings")
    samples_value = result["samples"]
    if not isinstance(samples_value, list):
        raise ValueError("summary samples must be an array")
    samples = [_validate_summary_sample(sample) for sample in samples_value]
    target_ids = set(target_order)
    sample_ids = set(selected)
    timing_ids = set(timing)
    seen: set[tuple[str, str, int]] = set()
    for sample in samples:
        target_id = str(sample["target_id"])
        sample_id = str(sample["sample_id"])
        repetition = int(sample["repetition"])
        key = (target_id, sample_id, repetition)
        if target_id not in target_ids or sample_id not in sample_ids or key in seen:
            raise ValueError("summary sample does not belong to its identity")
        seen.add(key)
        _validate_execution_against_target(
            sample,
            targets_by_id[target_id],
            artifact="summary sample",
        )
        maximum_repetition = warm_repetitions - 1 if sample_id in timing_ids else 0
        if repetition > maximum_repetition:
            raise ValueError("summary sample repetition is inconsistent")
        allowed_timing = {"cold_first"} if sample_id == cold_probe else {"warm", "warmup_recovery"}
        if sample["timing_class"] not in allowed_timing:
            raise ValueError("summary sample timing class is inconsistent")
        if (sample["reference"] is not None) != _text_retention_required(
            sample,
            str(identity["text_retention"]),
        ):
            raise ValueError("summary sample text retention is inconsistent")
    expected_order = sorted(
        samples,
        key=lambda sample: (
            target_order.index(sample["target_id"]),
            selected.index(sample["sample_id"]),
            int(sample["repetition"]),
        ),
    )
    if samples != expected_order:
        raise ValueError("summary sample order is inconsistent")
    recomputed = _aggregate_summary_samples(
        run_id=run_id,
        cold_probe_sample_id=cold_probe,
        samples=samples,
    )
    for field in (
        "schema_version",
        "run_id",
        "active_result_count",
        "primary",
        "diagnostic",
        "slices",
        "performance",
    ):
        if result[field] != recomputed[field]:
            raise ValueError(f"summary {field} is inconsistent with samples")
    progress = result["progress"]
    if not isinstance(progress, dict) or set(progress) != {
        "expected_result_count",
        "active_result_count",
        "pending_result_count",
        "complete",
        "history_truncated_tail_ignored",
    }:
        raise ValueError("summary progress is invalid")
    expected_count = len(targets) * (len(selected) + len(timing) * (warm_repetitions - 1))
    active_count = len(samples)
    expected_progress = {
        "expected_result_count": expected_count,
        "active_result_count": active_count,
        "pending_result_count": expected_count - active_count,
        "complete": active_count == expected_count,
        "history_truncated_tail_ignored": progress["history_truncated_tail_ignored"],
    }
    if not isinstance(progress["history_truncated_tail_ignored"], bool) or progress != expected_progress:
        raise ValueError("summary progress is inconsistent")
    _validate_observed_reference_provenance(
        samples,
        declared=provenance_counts,
        complete=bool(progress["complete"]),
    )
    if result["eligibility"] != _report_eligibility(samples):
        raise ValueError("summary eligibility is inconsistent")
    if result["worst_examples"] != _worst_retained_examples(samples):
        raise ValueError("summary worst examples are inconsistent")
    result["identity"] = dict(identity)
    result["samples"] = samples
    return result


_POLICY_RULES = frozenset(
    {
        "max_normalized_pooled_wer_absolute_regression",
        "max_normalized_pooled_wer_relative_regression",
        "max_normalized_pooled_cer_absolute_regression",
        "max_normalized_pooled_cer_relative_regression",
        "max_failure_rate_absolute_regression",
        "max_failure_rate_relative_regression",
        "min_exact_match_rate",
        "max_warm_rtf_absolute_regression",
        "max_warm_rtf_relative_regression",
        "max_warm_adapter_seconds_absolute_regression",
        "max_warm_adapter_seconds_relative_regression",
    }
)
_QUALITY_COMPATIBILITY_FIELDS = (
    "manifest_hash",
    "selected_sample_ids",
    "reference_provenance_counts",
    "profile",
    "mode",
    "seed",
    "cold_probe_sample_id",
    "warm_repetitions",
    "timing_sample_ids",
    "scorer_version",
    "strict_profile",
    "unicode_version",
)
_HARDWARE_FIELDS = (
    "os_name",
    "os_release",
    "architecture",
    "logical_cores",
    "physical_cores",
    "ram_bytes",
    "cpu_model",
    "accelerator",
    "collection_methods",
)


def validate_policy(policy: Mapping[str, object]) -> dict[str, object]:
    """Validate the optional versioned, per-suite regression policy."""
    if not isinstance(policy, Mapping) or set(policy) != {"schema_version", "suites"}:
        raise ValueError("policy has missing or unknown fields")
    if policy["schema_version"] != 1:
        raise ValueError("unsupported policy schema version")
    suites = policy["suites"]
    if not isinstance(suites, dict) or not suites:
        raise ValueError("policy suites must be a non-empty object")
    validated_suites: dict[str, dict[str, float]] = {}
    for suite, rules in suites.items():
        _require_stable_id(suite, "<policy>", "suite")
        if not isinstance(rules, dict) or not rules or set(rules) - _POLICY_RULES:
            raise ValueError("policy suite has missing or unknown rules")
        validated_rules: dict[str, float] = {}
        for name, value in rules.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("policy bounds must be finite and non-negative")
            number = float(value)
            if name == "min_exact_match_rate" and number > 1.0:
                raise ValueError("policy exact-match minimum must be at most one")
            validated_rules[name] = number
        validated_suites[str(suite)] = validated_rules
    return {
        "schema_version": 1,
        "suites": {suite: validated_suites[suite] for suite in sorted(validated_suites)},
    }


def _sample_layout(summary: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return target-independent sample metadata and reject internal drift."""
    layout: dict[str, dict[str, object]] = {}
    for sample in summary["samples"]:
        if sample["measurement_role"] != "accuracy":
            continue
        sample_id = str(sample["sample_id"])
        metadata = {
            "suite": sample["suite"],
            "suite_visibility": sample["suite_visibility"],
            "dataset": sample["dataset"],
            "reference_provenance": sample["reference_provenance"],
            "tags": sample["tags"],
            "diagnostic_only": sample["diagnostic_only"],
            "normalization_profile": sample["normalization_profile"],
        }
        previous = layout.setdefault(sample_id, metadata)
        if previous != metadata:
            raise ValueError("summary sample metadata is inconsistent across targets")
    return layout


def _target_sample_map(
    summary: Mapping[str, object],
    target_id: str,
) -> dict[str, Mapping[str, object]]:
    """Return one target's ordered accuracy observations by sample ID."""
    return {
        str(sample["sample_id"]): sample
        for sample in summary["samples"]
        if sample["target_id"] == target_id and sample["measurement_role"] == "accuracy"
    }


def _actual_execution_signatures(
    summary: Mapping[str, object],
    target_id: str,
) -> set[str]:
    """Return canonical actual-execution identities for one target."""
    signatures: set[str] = set()
    for sample in summary["samples"]:
        if sample["target_id"] != target_id:
            continue
        execution = sample["actual_execution"]
        if execution is None:
            signatures.add("unavailable")
        else:
            signatures.add(
                json.dumps(
                    execution,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    return signatures


def _target_has_execution_mismatch(
    summary: Mapping[str, object],
    target_id: str,
) -> bool:
    """Return whether any retained result for one target mismatched its plan."""
    return any(
        sample["target_id"] == target_id and bool(sample["execution_mismatch_reasons"]) for sample in summary["samples"]
    )


def _target_identity_resolved(target: Mapping[str, object]) -> bool:
    """Return whether every declared route has immutable resolved identity."""
    routes = target["descriptor"].get("routes")
    return bool(
        isinstance(routes, list)
        and routes
        and all(
            isinstance(route, dict)
            and route.get("identity_resolved") is True
            and isinstance(route.get("artifact_id"), str)
            for route in routes
        )
    )


def _same_target_identity(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> bool:
    """Compare material target identity while allowing implementation changes."""
    return (
        baseline["provider"] == candidate["provider"]
        and baseline["model_label"] == candidate["model_label"]
        and baseline["descriptor"] == candidate["descriptor"]
        and baseline["execution_contract"]["safe_target_settings"]
        == candidate["execution_contract"]["safe_target_settings"]
    )


def _common_semantic_settings(
    target: Mapping[str, object],
) -> dict[str, object]:
    """Return settings shared across descriptive target/config comparisons."""
    settings = target["execution_contract"]["safe_target_settings"]
    return {
        name: value
        for name, value in settings.items()
        if name
        not in {
            "configuration_id",
            "network_collection_profile",
            "network_client_location",
        }
    }


def _hardware_matches(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> bool:
    """Compare the hardware and collection-method identity used for gates."""
    return all(
        baseline["identity"]["environment"][field] == candidate["identity"]["environment"][field]
        for field in _HARDWARE_FIELDS
    )


def _paired_sample_deltas(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    baseline_target_id: str,
    candidate_target_id: str,
) -> list[dict[str, object]]:
    """Build deterministic accuracy deltas in selected-sample order."""
    baseline_samples = _target_sample_map(baseline, baseline_target_id)
    candidate_samples = _target_sample_map(candidate, candidate_target_id)
    deltas: list[dict[str, object]] = []
    for sample_id in baseline["identity"]["selected_sample_ids"]:
        before = baseline_samples[sample_id]
        after = candidate_samples[sample_id]
        deltas.append(
            {
                "sample_id": sample_id,
                "suite": before["suite"],
                "baseline_status": before["status"],
                "candidate_status": after["status"],
                "normalized_wer_delta": (
                    float(after["normalized"]["wer"]["rate"]) - float(before["normalized"]["wer"]["rate"])
                ),
                "normalized_cer_delta": (
                    float(after["normalized"]["cer"]["rate"]) - float(before["normalized"]["cer"]["rate"])
                ),
                "strict_wer_delta": (float(after["strict"]["wer"]["rate"]) - float(before["strict"]["wer"]["rate"])),
                "strict_cer_delta": (float(after["strict"]["cer"]["rate"]) - float(before["strict"]["cer"]["rate"])),
                "exact_match_delta": (int(bool(after["exact_match"])) - int(bool(before["exact_match"]))),
            }
        )
    return deltas


def _suite_metric_deltas(
    baseline_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
) -> dict[str, float]:
    """Return descriptive headline deltas for one target/suite pair."""
    return {
        "normalized_pooled_wer_delta": (
            float(candidate_metrics["normalized"]["wer"]["pooled"])
            - float(baseline_metrics["normalized"]["wer"]["pooled"])
        ),
        "normalized_pooled_cer_delta": (
            float(candidate_metrics["normalized"]["cer"]["pooled"])
            - float(baseline_metrics["normalized"]["cer"]["pooled"])
        ),
        "failure_rate_delta": (float(candidate_metrics["failure_rate"]) - float(baseline_metrics["failure_rate"])),
        "exact_match_rate_delta": (
            float(candidate_metrics["exact_match_rate"]) - float(baseline_metrics["exact_match_rate"])
        ),
    }


def _descriptive_rankings(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Rank displayed target/suite WER values without inferential claims."""
    suites: dict[str, list[dict[str, object]]] = {}
    excluded: list[dict[str, str]] = []
    for pair in pairs:
        for role, summary, target_id in (
            ("baseline", baseline, pair["baseline_target_id"]),
            ("candidate", candidate, pair["candidate_target_id"]),
        ):
            eligibility = summary["eligibility"]["targets"].get(
                target_id,
                {},
            )
            if eligibility.get("mixed_actual_execution") is True:
                excluded.append(
                    {
                        "role": role,
                        "target_id": str(target_id),
                        "reason": "mixed_actual_execution",
                    }
                )
                continue
            target = summary["primary"]["targets"].get(target_id, {"suites": {}})
            for suite, metrics in target["suites"].items():
                suites.setdefault(str(suite), []).append(
                    {
                        "role": role,
                        "target_id": target_id,
                        "normalized_pooled_wer": metrics["normalized"]["wer"]["pooled"],
                    }
                )
    return {
        "label": "descriptive",
        "suites": {
            suite: sorted(
                entries,
                key=lambda entry: (
                    float(entry["normalized_pooled_wer"]),
                    str(entry["role"]),
                    str(entry["target_id"]),
                ),
            )
            for suite, entries in sorted(suites.items())
        },
        "excluded": excluded,
    }


def _provenance_differences(
    baseline_target: Mapping[str, object],
    candidate_target: Mapping[str, object],
) -> dict[str, object]:
    """Expose allowed implementation/dependency differences."""
    before = baseline_target["execution_contract"]
    after = candidate_target["execution_contract"]
    differences: dict[str, object] = {}
    for field in ("dependency_versions", "source_hashes", "git_commit"):
        if before[field] != after[field]:
            differences[field] = {
                "baseline": before[field],
                "candidate": after[field],
            }
    return differences


def _policy_metric(
    rule: str,
    quality: Mapping[str, object],
    warm: Mapping[str, object] | None,
) -> float:
    """Read the allowlisted metric used by one policy rule."""
    if "normalized_pooled_wer" in rule:
        return float(quality["normalized"]["wer"]["pooled"])
    if "normalized_pooled_cer" in rule:
        return float(quality["normalized"]["cer"]["pooled"])
    if "failure_rate" in rule:
        return float(quality["failure_rate"])
    if rule == "min_exact_match_rate":
        return float(quality["exact_match_rate"])
    if "warm_rtf" in rule:
        if warm is None or warm["rtf"]["p50"] is None:
            raise ValueError("requested warm performance gate is ineligible")
        return float(warm["rtf"]["p50"])
    if "warm_adapter_seconds" in rule:
        if warm is None or warm["adapter_seconds"]["p50"] is None:
            raise ValueError("requested warm performance gate is ineligible")
        return float(warm["adapter_seconds"]["p50"])
    raise ValueError("unknown policy rule")


def _is_network_target(
    summary: Mapping[str, object],
    target_id: str,
) -> bool:
    """Return whether any active observation sent audio beyond the process."""
    return any(
        sample["target_id"] == target_id
        and sample["actual_execution"] is not None
        and sample["actual_execution"]["audio_egress"] != "none"
        for sample in summary["samples"]
    )


def _performance_gate_eligible(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    pair: Mapping[str, object],
    suite: str,
    *,
    allow_network_performance_gates: bool,
) -> bool:
    """Return whether one warm performance threshold is comparable."""
    baseline_target_id = str(pair["baseline_target_id"])
    candidate_target_id = str(pair["candidate_target_id"])
    if (
        not pair["gate_identity_eligible"]
        or baseline["identity"]["target_order"].index(baseline_target_id)
        != candidate["identity"]["target_order"].index(candidate_target_id)
        or int(baseline["identity"]["warm_repetitions"]) < 3
        or int(candidate["identity"]["warm_repetitions"]) < 3
    ):
        return False
    baseline_warm = (
        baseline["performance"]["warm"]["targets"].get(baseline_target_id, {"suites": {}})["suites"].get(suite)
    )
    candidate_warm = (
        candidate["performance"]["warm"]["targets"].get(candidate_target_id, {"suites": {}})["suites"].get(suite)
    )
    if (
        baseline_warm is None
        or candidate_warm is None
        or int(baseline_warm["gate_eligible_count"]) < 3
        or int(candidate_warm["gate_eligible_count"]) < 3
    ):
        return False
    network = _is_network_target(
        baseline,
        baseline_target_id,
    ) or _is_network_target(candidate, candidate_target_id)
    if not network:
        return True
    before_settings = pair["baseline_target"]["execution_contract"]["safe_target_settings"]
    after_settings = pair["candidate_target"]["execution_contract"]["safe_target_settings"]
    return bool(
        allow_network_performance_gates
        and before_settings.get("network_collection_profile")
        and before_settings.get("network_collection_profile") == after_settings.get("network_collection_profile")
        and before_settings.get("network_client_location")
        and before_settings.get("network_client_location") == after_settings.get("network_client_location")
    )


def compare_summaries(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    policy: Mapping[str, object] | None = None,
    allow_network_performance_gates: bool = False,
) -> dict[str, object]:
    """Compare compatible summaries and optionally enforce eligible gates."""
    if not isinstance(allow_network_performance_gates, bool):
        raise TypeError("network performance gate opt-in must be boolean")
    validated_policy = validate_policy(policy) if policy is not None else None
    try:
        before = validate_summary(baseline)
        after = validate_summary(candidate)
    except ValueError as exc:
        raise ValueError("summaries are not compatible or valid") from exc
    for field in _QUALITY_COMPATIBILITY_FIELDS:
        if before["identity"][field] != after["identity"][field]:
            raise ValueError("summaries are not compatible for quality comparison")
    if len(before["identity"]["targets"]) != len(after["identity"]["targets"]):
        raise ValueError("summaries are not compatible target matrices")
    if not before["progress"]["complete"] or not after["progress"]["complete"]:
        if validated_policy is not None:
            raise ValueError("requested policy gates are ineligible for partial runs")
        raise ValueError("descriptive comparison does not support partial summaries")
    if _sample_layout(before) != _sample_layout(after):
        raise ValueError("summaries are not compatible sample suites")
    if _common_semantic_settings(before["identity"]["targets"][0]) != _common_semantic_settings(
        after["identity"]["targets"][0]
    ):
        raise ValueError("summaries are not compatible common settings")
    hardware_matches = _hardware_matches(before, after)
    pairs: list[dict[str, object]] = []
    public_pairs: list[dict[str, object]] = []
    for ordinal, (before_target, after_target) in enumerate(
        zip(
            before["identity"]["targets"],
            after["identity"]["targets"],
            strict=True,
        ),
        start=1,
    ):
        baseline_target_id = str(before_target["target_id"])
        candidate_target_id = str(after_target["target_id"])
        same_target = _same_target_identity(before_target, after_target)
        baseline_signatures = _actual_execution_signatures(
            before,
            baseline_target_id,
        )
        candidate_signatures = _actual_execution_signatures(
            after,
            candidate_target_id,
        )
        gate_identity_eligible = bool(
            same_target
            and hardware_matches
            and _target_identity_resolved(before_target)
            and _target_identity_resolved(after_target)
            and len(baseline_signatures) == 1
            and len(candidate_signatures) == 1
            and baseline_signatures == candidate_signatures
            and "unavailable" not in baseline_signatures
            and not _target_has_execution_mismatch(
                before,
                baseline_target_id,
            )
            and not _target_has_execution_mismatch(
                after,
                candidate_target_id,
            )
        )
        before_suites = before["primary"]["targets"].get(
            baseline_target_id,
            {"suites": {}},
        )["suites"]
        after_suites = after["primary"]["targets"].get(
            candidate_target_id,
            {"suites": {}},
        )["suites"]
        if set(before_suites) != set(after_suites):
            raise ValueError("summaries are not compatible suite populations")
        suite_deltas = {
            suite: _suite_metric_deltas(before_suites[suite], after_suites[suite]) for suite in sorted(before_suites)
        }
        internal_pair = {
            "ordinal": ordinal,
            "baseline_target_id": baseline_target_id,
            "candidate_target_id": candidate_target_id,
            "same_target": same_target,
            "gate_identity_eligible": gate_identity_eligible,
            "baseline_target": before_target,
            "candidate_target": after_target,
            "suite_deltas": suite_deltas,
            "paired_samples": _paired_sample_deltas(
                before,
                after,
                baseline_target_id,
                candidate_target_id,
            ),
            "provenance_differences": _provenance_differences(
                before_target,
                after_target,
            ),
        }
        pairs.append(internal_pair)
        public_pairs.append(
            {key: value for key, value in internal_pair.items() if key not in {"baseline_target", "candidate_target"}}
        )
    gates: list[dict[str, object]] = []
    if validated_policy is not None:
        if not all(pair["gate_identity_eligible"] for pair in pairs):
            raise ValueError("requested policy gates are ineligible for these target identities")
        available_suites = {suite for pair in pairs for suite in pair["suite_deltas"]}
        if not set(validated_policy["suites"]) <= available_suites:
            raise ValueError("policy references a suite absent from either summary")
        for pair in pairs:
            baseline_target_id = str(pair["baseline_target_id"])
            candidate_target_id = str(pair["candidate_target_id"])
            baseline_quality = before["primary"]["targets"][baseline_target_id]["suites"]
            candidate_quality = after["primary"]["targets"][candidate_target_id]["suites"]
            baseline_warm_suites = before["performance"]["warm"]["targets"].get(baseline_target_id, {"suites": {}})[
                "suites"
            ]
            candidate_warm_suites = after["performance"]["warm"]["targets"].get(candidate_target_id, {"suites": {}})[
                "suites"
            ]
            for suite, rules in validated_policy["suites"].items():
                for rule, bound in rules.items():
                    is_performance = rule.startswith("max_warm_")
                    if is_performance and not _performance_gate_eligible(
                        before,
                        after,
                        pair,
                        suite,
                        allow_network_performance_gates=allow_network_performance_gates,
                    ):
                        raise ValueError("requested warm performance gate is ineligible")
                    baseline_value = _policy_metric(
                        rule,
                        baseline_quality[suite],
                        baseline_warm_suites.get(suite),
                    )
                    candidate_value = _policy_metric(
                        rule,
                        candidate_quality[suite],
                        candidate_warm_suites.get(suite),
                    )
                    if rule == "min_exact_match_rate":
                        observed = candidate_value
                        passed = observed >= bound
                    elif rule.endswith("_relative_regression"):
                        if baseline_value == 0.0:
                            raise ValueError("relative policy rule has a zero baseline")
                        observed = (candidate_value - baseline_value) / baseline_value
                        passed = observed <= bound
                    else:
                        observed = candidate_value - baseline_value
                        passed = observed <= bound
                    gates.append(
                        {
                            "target_ordinal": pair["ordinal"],
                            "suite": suite,
                            "rule": rule,
                            "baseline": baseline_value,
                            "candidate": candidate_value,
                            "observed": observed,
                            "bound": bound,
                            "eligible": True,
                            "passed": passed,
                        }
                    )
    failed = any(not gate["passed"] for gate in gates)
    return {
        "schema_version": 1,
        "baseline_run_id": before["run_id"],
        "candidate_run_id": after["run_id"],
        "mode": "policy" if validated_policy is not None else "descriptive",
        "compatibility": {
            "quality_identity": True,
            "hardware_match": hardware_matches,
        },
        "target_pairs": public_pairs,
        "rankings": _descriptive_rankings(before, after, pairs),
        "gates": gates,
        "exit_code": 1 if failed else 0,
    }


def _compare_command(arguments: argparse.Namespace) -> int:
    """Compare summary artifacts and return the documented gate exit code."""
    policy = _load_json_object(arguments.policy) if arguments.policy is not None else None
    comparison = compare_summaries(
        _load_json_object(arguments.baseline),
        _load_json_object(arguments.candidate),
        policy=policy,
        allow_network_performance_gates=arguments.allow_network_performance_gates,
    )
    print(
        json.dumps(
            comparison,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return int(comparison["exit_code"])


def _report_command(arguments: argparse.Namespace) -> int:
    """Regenerate and print one run report."""
    summary = generate_report(arguments.run)
    print(render_summary_terminal(summary), end="")
    return 0


def _validate_command(arguments: argparse.Namespace) -> int:
    """Validate one manifest and print only its portable aggregate identity."""
    samples, content_hash = load_and_validate_manifest(
        arguments.manifest,
        arguments.dataset_root,
    )
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


def _default_run_id(
    manifest_hash: str,
    prepared_targets: Sequence[PreparedTarget],
) -> str:
    """Return a collision-resistant, stable default run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz").lower()
    identity = "\0".join(
        (
            timestamp,
            str(time.time_ns()),
            manifest_hash,
            *(target.execution_contract_hash for target in prepared_targets),
        )
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{timestamp}-{suffix}"


def _benchmark_run_directory(run_id: str) -> Path:
    """Return the repository-owned directory for one validated run ID."""
    _require_stable_id(run_id, "<cli>", "run")
    return Path(__file__).resolve().parents[2] / ".benchmarks" / "stt" / run_id


def _selected_primary_language(
    samples: Sequence[ManifestSample],
) -> str:
    """Return the single v1 primary language or reject a mixed selection."""
    languages = {sample.language for sample in samples}
    if len(languages) != 1:
        raise ValueError("v1 runs require exactly one primary language")
    return next(iter(languages))


def _selected_timing_sample_ids(
    samples: Sequence[ManifestSample],
    *,
    cold_probe_sample_id: str,
    requested_sample_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate or derive the ordered non-probe timing subset."""
    selected_ids = tuple(sample.sample_id for sample in samples)
    if requested_sample_ids is None:
        return tuple(sample_id for sample_id in selected_ids if sample_id != cold_probe_sample_id)
    timing_ids = tuple(requested_sample_ids)
    if len(timing_ids) != len(set(timing_ids)):
        raise ValueError("timing sample IDs must be unique")
    if cold_probe_sample_id in timing_ids:
        raise ValueError("the cold probe cannot also be a timing sample")
    requested = set(timing_ids)
    if not requested <= set(selected_ids):
        raise ValueError("timing samples must belong to the selected profile")
    return tuple(sample_id for sample_id in selected_ids if sample_id in requested)


def _run_command(arguments: argparse.Namespace) -> int:
    """Preflight and execute one deterministic native STT benchmark run."""
    if arguments.mode == "production-v1" and arguments.configuration_id is None:
        raise ValueError("production-v1 requires --configuration-id")
    if arguments.mode == "neutral-v1" and arguments.configuration_id is not None:
        raise ValueError("neutral-v1 rejects --configuration-id")
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative")
    if arguments.warm_repetitions < 1:
        raise ValueError("--warm-repetitions must be positive")
    if arguments.worker_watchdog_seconds is not None and (
        not math.isfinite(arguments.worker_watchdog_seconds) or arguments.worker_watchdog_seconds <= 0.0
    ):
        raise ValueError("--worker-watchdog-seconds must be positive and finite")
    if arguments.run is not None:
        _require_stable_id(arguments.run, "<cli>", "run")

    samples, manifest_hash = load_and_validate_manifest(
        arguments.manifest,
        arguments.dataset_root,
    )
    selected_samples, cold_probe_sample_id = select_samples(
        samples,
        profile=arguments.profile,
        seed=arguments.seed,
    )
    primary_language = _selected_primary_language(
        selected_samples,
    )
    timing_sample_ids = _selected_timing_sample_ids(
        selected_samples,
        cold_probe_sample_id=cold_probe_sample_id,
        requested_sample_ids=arguments.timing_sample,
    )
    if arguments.text_retention == "full" and any(sample.suite_visibility == "private" for sample in selected_samples):
        print(
            "warning: full text retention will persist private-suite transcripts",
            file=sys.stderr,
        )

    environment = collect_environment_fingerprint()
    common_settings: dict[str, object] = {
        "task": "transcribe",
        "language": primary_language,
        "word_timestamps": False,
        "prompt": None,
        "hotwords": (),
        "diarization": False,
        "git_commit": environment["git_commit"],
    }
    for name in (
        "configuration_id",
        "network_collection_profile",
        "network_client_location",
    ):
        value = getattr(arguments, name)
        if value is not None:
            common_settings[name] = value
    prepared_targets = preflight_targets(
        tuple(arguments.target),
        mode=arguments.mode,
        allow_network_targets=arguments.allow_network_targets,
        common_settings=common_settings,
    )
    run_id = arguments.run or _default_run_id(
        manifest_hash,
        prepared_targets,
    )
    run_directory = _benchmark_run_directory(run_id)
    if arguments.run is None and run_directory.exists():
        raise ValueError("generated run identifier already exists")

    audio_paths = tuple(
        str(resolve_audio_for_scheduling(sample, arguments.dataset_root)) for sample in selected_samples
    )
    run_metadata = build_run_metadata(
        run_id=run_id,
        manifest_hash=manifest_hash,
        selected_sample_ids=tuple(sample.sample_id for sample in selected_samples),
        reference_provenance_counts=(_reference_provenance_counts_for_samples(selected_samples)),
        profile=arguments.profile,
        mode=arguments.mode,
        seed=arguments.seed,
        cold_probe_sample_id=cold_probe_sample_id,
        warm_repetitions=arguments.warm_repetitions,
        timing_sample_ids=timing_sample_ids,
        text_retention=arguments.text_retention,
        adapter_watchdog_seconds=arguments.worker_watchdog_seconds,
        prepared_targets=prepared_targets,
        environment=environment,
    )
    completed = execute_prepared_targets(
        run_directory=run_directory,
        run_metadata=run_metadata,
        prepared_targets=prepared_targets,
        samples=selected_samples,
        audio_paths=audio_paths,
        retry_errors=arguments.retry_errors,
        allow_resume=arguments.run is not None,
    )
    print(
        json.dumps(
            {
                "result": "completed",
                "run_id": run_id,
                "worker_attempt_count": len(completed["worker_attempts"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the native batch STT benchmark command-line interface."""
    parser = argparse.ArgumentParser(prog="stt-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--dataset-root", required=True, type=Path)
    validate.set_defaults(handler=_validate_command)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--dataset-root", required=True, type=Path)
    run.add_argument("--target", required=True, action="append")
    run.add_argument("--profile", choices=sorted(_KNOWN_SAMPLE_PROFILES), default="comparison")
    run.add_argument(
        "--mode",
        choices=("neutral-v1", "production-v1"),
        default="neutral-v1",
    )
    run.add_argument(
        "--text-retention",
        choices=("full", "errors-only", "none"),
        default="full",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--warm-repetitions", type=int, default=1)
    run.add_argument("--timing-sample", action="append")
    run.add_argument("--worker-watchdog-seconds", type=float)
    run.add_argument("--retry-errors", action="store_true")
    run.add_argument("--allow-network-targets", action="store_true")
    run.add_argument("--configuration-id")
    run.add_argument("--network-collection-profile")
    run.add_argument("--network-client-location")
    run.add_argument("--run")
    run.set_defaults(handler=_run_command)
    report = commands.add_parser("report")
    report.add_argument("--run", required=True, type=Path)
    report.set_defaults(handler=_report_command)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--policy", type=Path)
    compare.add_argument(
        "--allow-network-performance-gates",
        action="store_true",
    )
    compare.set_defaults(handler=_compare_command)
    arguments = parser.parse_args(argv)

    try:
        return arguments.handler(arguments)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("error: command could not access a required local file", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: benchmark interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
