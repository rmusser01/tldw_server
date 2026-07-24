"""Deterministic normalization and scoring for native STT benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess  # nosec B404
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCORER_VERSION = "stt-score-v1"
STRICT_PROFILE = "strict-v1"
EN_PROFILE = "en-v1"
BCP47_BASIC_V1 = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
STABLE_ID_V1 = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
MAX_TAGS_PER_SAMPLE = 32

_KNOWN_SAMPLE_PROFILES = frozenset({"comparison", "regression"})
_KNOWN_NORMALIZATION_PROFILES = frozenset({STRICT_PROFILE, EN_PROFILE})
_SOURCE_REQUIRED_FIELDS = frozenset(
    {"dataset", "version", "license", "reference_provenance", "sha256"}
)
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
    return finite_values[lower] + (
        finite_values[upper] - finite_values[lower]
    ) * fraction


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
    identifiers = tuple(
        _require_stable_id(item, sample_id, field) for item in value
    )
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
    if (
        "\\" in relative
        or relative.startswith("//")
        or re.match(r"^[A-Za-z]:", relative)
    ):
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
        if (
            not isinstance(item, str)
            or not item.strip()
            or len(item) > _MAX_SOURCE_VALUE_LENGTH
        ):
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
                    raise ValueError(
                        f"manifest line {line_number}: malformed JSON"
                    ) from exc
                except ValueError as exc:
                    raise ValueError(
                        f"manifest line {line_number}: duplicate JSON field"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"manifest line {line_number}: record must be an object"
                    )
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
        error_id = (
            raw_id
            if isinstance(raw_id, str) and STABLE_ID_V1.fullmatch(raw_id)
            else f"<line-{line_number}>"
        )
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
        if (
            not isinstance(normalization_profile, str)
            or normalization_profile not in _KNOWN_NORMALIZATION_PROFILES
        ):
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
            normalize_en_v1(reference)
            if normalization_profile == EN_PROFILE
            else normalize_strict_v1(reference)
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
            rounding_slack = (
                math.ulp(declared_duration)
                + math.ulp(measured_duration)
                + math.ulp(tolerance)
            )
            if (
                abs(declared_duration - measured_duration)
                > tolerance + rounding_slack
            ):
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
    selected.sort(
        key=lambda sample: hashlib.sha256(
            f"{seed}\0{sample.sample_id}".encode()
        ).digest()
    )
    return tuple(selected), selected[0].sample_id


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
    profile_counts = Counter(
        profile for sample in samples for profile in sample.profiles
    )
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
