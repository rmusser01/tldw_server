"""Tests for the deterministic native STT benchmark scorer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import stat
import subprocess
import types
import wave
from dataclasses import FrozenInstanceError
from pathlib import Path

import Helper_Scripts.benchmarks.stt_bench as stt_bench
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

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttActualExecution,
    SttAudioEgress,
    SttBatchExecutionPlan,
    SttExecutionDescriptor,
    SttExecutionRoute,
)

_PREFLIGHT_PLANS: dict[tuple[str, str], SttBatchExecutionPlan | BaseException] = {}
_PREFLIGHT_CALLS: list[tuple[str, str]] = []
_PREFLIGHT_UNAVAILABLE: set[str] = set()
_PREFLIGHT_CANONICAL: dict[str, str] = {}


class _PreflightFakeAdapter:
    def __init__(self, lookup_provider: str, canonical_provider: str) -> None:
        self.lookup_provider = lookup_provider
        self.name = types.SimpleNamespace(value=canonical_provider)

    def get_capabilities(self):
        return types.SimpleNamespace(supports_batch=self.lookup_provider not in _PREFLIGHT_UNAVAILABLE)

    def plan_batch_execution(self, *, model, **settings):
        key = (self.lookup_provider, model)
        _PREFLIGHT_CALLS.append(key)
        planned = _PREFLIGHT_PLANS[key]
        if isinstance(planned, BaseException):
            raise planned
        assert settings["mode"] in {"neutral-v1", "production-v1"}
        return planned


def _preflight_fake_factory(provider: str):
    if not any(key[0] == provider for key in _PREFLIGHT_PLANS):
        raise LookupError(f"unknown provider {provider}")
    return _PreflightFakeAdapter(
        provider,
        _PREFLIGHT_CANONICAL.get(provider, provider),
    )


_NOT_AN_ADAPTER_FACTORY = object()


def _manifest_record(
    audio_path: Path,
    *,
    sample_id: str = "sample-1",
    reference: str = "Hello world",
    **overrides,
):
    record = {
        "id": sample_id,
        "audio": audio_path.name,
        "reference": reference,
        "language": "en",
        "normalization_profile": "en-v1",
        "duration_seconds": 1.0,
        "profiles": ["regression", "comparison"],
        "suite": "public-english-v1",
        "suite_visibility": "public",
        "annotation_profile": "canonical-v1",
        "diagnostic_only": False,
        "source": {
            "dataset": "fixture",
            "version": "v1",
            "license": "CC0-1.0",
            "reference_provenance": "canonical-dataset",
            "split": "test",
            "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        },
        "tags": ["single-speaker", "read-speech"],
    }
    record.update(overrides)
    return record


def _write_manifest(path: Path, records) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _valid_manifest(tmp_path: Path, **overrides):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio fixture")
    record = _manifest_record(audio_path, **overrides)
    manifest_path = _write_manifest(tmp_path / "manifest.jsonl", [record])
    return manifest_path, audio_path, record


def test_manifest_loads_canonical_portable_sample_and_hash(tmp_path):
    manifest_path, _, _ = _valid_manifest(
        tmp_path,
        language="EN-us",
        profiles=["comparison", "regression"],
        tags=["single-speaker", "read-speech"],
    )

    samples, content_hash = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )

    assert samples == (
        stt_bench.ManifestSample(
            sample_id="sample-1",
            audio_relative="clip.wav",
            reference="Hello world",
            language="en-us",
            normalization_profile="en-v1",
            measured_duration_seconds=1.0,
            profiles=("comparison", "regression"),
            suite="public-english-v1",
            suite_visibility="public",
            annotation_profile="canonical-v1",
            diagnostic_only=False,
            source=(
                ("dataset", "fixture"),
                ("license", "CC0-1.0"),
                ("reference_provenance", "canonical-dataset"),
                ("split", "test"),
                ("version", "v1"),
            ),
            tags=("read-speech", "single-speaker"),
            sha256=hashlib.sha256(b"audio fixture").hexdigest(),
        ),
    )
    assert len(content_hash) == 64
    assert str(tmp_path) not in content_hash


@pytest.mark.parametrize(
    ("mutator", "field"),
    [
        (lambda record: record.update(id=""), "id"),
        (lambda record: record.update(id="Bad ID"), "id"),
        (lambda record: record.update(reference=""), "reference"),
        (lambda record: record.update(reference="..."), "reference"),
        (lambda record: record.update(language="e"), "language"),
        (lambda record: record.update(normalization_profile="unknown-v1"), "normalization_profile"),
        (lambda record: record.update(normalization_profile=[]), "normalization_profile"),
        (
            lambda record: record.update(language="fr", normalization_profile="en-v1"),
            "normalization_profile",
        ),
        (lambda record: record.update(profiles=[]), "profiles"),
        (lambda record: record.update(profiles=["regression", "regression"]), "profiles"),
        (lambda record: record.update(profiles=["unknown"]), "profiles"),
        (lambda record: record.update(suite="Bad Suite"), "suite"),
        (lambda record: record.update(annotation_profile="Bad Profile"), "annotation_profile"),
        (lambda record: record.update(suite_visibility="secret"), "suite_visibility"),
        (lambda record: record.update(suite_visibility={}), "suite_visibility"),
        (lambda record: record.update(diagnostic_only=0), "diagnostic_only"),
        (lambda record: record.update(tags=["same", "same"]), "tags"),
        (lambda record: record.update(tags=[f"tag-{index}" for index in range(33)]), "tags"),
        (lambda record: record.update(tags=["Bad Tag"]), "tags"),
    ],
)
def test_manifest_rejects_invalid_declared_fields_without_reference_leak(
    tmp_path,
    mutator,
    field,
):
    manifest_path, _, record = _valid_manifest(tmp_path)
    mutator(record)
    _write_manifest(manifest_path, [record])

    with pytest.raises(ValueError) as error:
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )

    assert "sample-1" in str(error.value) or "<line-1>" in str(error.value)
    assert field in str(error.value)
    assert "Hello world" not in str(error.value)


@pytest.mark.parametrize(
    "duration",
    [True, 0, -1, math.nan, math.inf, -math.inf, "1.0"],
)
def test_manifest_rejects_invalid_declared_duration(tmp_path, duration):
    manifest_path, _, _ = _valid_manifest(tmp_path, duration_seconds=duration)

    with pytest.raises(ValueError, match=r"sample-1.*duration_seconds"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"reference": "\ud800"}, "reference"),
        (
            {
                "source": {
                    "dataset": "\ud800",
                    "version": "v1",
                    "license": "CC0",
                    "reference_provenance": "canonical",
                    "sha256": hashlib.sha256(b"audio fixture").hexdigest(),
                }
            },
            "source.dataset",
        ),
    ],
)
def test_manifest_rejects_lone_surrogate_text_with_field_error(
    tmp_path,
    overrides,
    field,
):
    manifest_path, _, _ = _valid_manifest(tmp_path, **overrides)

    with pytest.raises(ValueError) as error:
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )

    assert "sample-1" in str(error.value)
    assert field in str(error.value)


def test_manifest_rejects_declared_duration_outside_tolerance(tmp_path):
    manifest_path, _, _ = _valid_manifest(tmp_path, duration_seconds=1.101)

    with pytest.raises(ValueError, match=r"sample-1.*duration_seconds"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )


def test_manifest_accepts_declared_duration_at_exact_tolerance_boundary(tmp_path):
    manifest_path, _, _ = _valid_manifest(tmp_path, duration_seconds=1.1)

    samples, _ = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )

    assert samples[0].measured_duration_seconds == 1.0


@pytest.mark.parametrize(
    "measured",
    [True, 0, -1, math.nan, math.inf, -math.inf, "1.0"],
)
def test_manifest_rejects_invalid_measured_duration(tmp_path, measured):
    manifest_path, _, _ = _valid_manifest(tmp_path)

    with pytest.raises(ValueError, match=r"sample-1.*duration"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: measured,
        )


@pytest.mark.parametrize(
    "source",
    [
        {},
        {
            "dataset": "",
            "version": "v1",
            "license": "CC0",
            "reference_provenance": "canonical",
        },
        {
            "dataset": "fixture",
            "version": "v1",
            "license": "CC0",
            "reference_provenance": "canonical",
            "sha256": "not-a-digest",
        },
        {
            "dataset": "fixture",
            "version": "v1",
            "license": "CC0",
            "reference_provenance": "canonical",
            "sha256": hashlib.sha256(b"audio fixture").hexdigest(),
            "bad key": "value",
        },
        {
            "dataset": "fixture",
            "version": "v1",
            "reference_provenance": "canonical",
            "sha256": hashlib.sha256(b"audio fixture").hexdigest(),
            "extra": "value",
        },
        {
            "dataset": "fixture",
            "version": "v1",
            "license": "CC0",
            "reference_provenance": "canonical",
            "sha256": hashlib.sha256(b"audio fixture").hexdigest(),
            "extra": 1,
        },
        {
            "dataset": "fixture",
            "version": "v1",
            "license": "CC0",
            "reference_provenance": "canonical",
            "sha256": hashlib.sha256(b"audio fixture").hexdigest(),
            **{f"extra-{index}": "value" for index in range(28)},
        },
    ],
)
def test_manifest_rejects_incomplete_or_invalid_source(tmp_path, source):
    manifest_path, _, _ = _valid_manifest(tmp_path, source=source)

    with pytest.raises(ValueError, match=r"sample-1.*source"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )


@pytest.mark.parametrize(
    "field",
    ["dataset", "version", "license", "reference_provenance"],
)
def test_manifest_rejects_whitespace_only_required_provenance(tmp_path, field):
    manifest_path, _, record = _valid_manifest(tmp_path)
    record["source"][field] = " \t "
    _write_manifest(manifest_path, [record])

    with pytest.raises(ValueError, match=rf"sample-1.*source\.{field}"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )


def test_manifest_rejects_sha256_mismatch(tmp_path):
    manifest_path, _, record = _valid_manifest(tmp_path)
    record["source"]["sha256"] = "0" * 64
    _write_manifest(manifest_path, [record])

    with pytest.raises(ValueError, match=r"sample-1.*sha256"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )


def test_manifest_wraps_audio_read_failure_without_absolute_path(
    tmp_path,
    monkeypatch,
):
    manifest_path, _, _ = _valid_manifest(tmp_path)
    monkeypatch.setattr(
        stt_bench,
        "_sha256_file",
        lambda path: (_ for _ in ()).throw(OSError(f"cannot read {path}")),
    )

    with pytest.raises(ValueError) as error:
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )

    message = str(error.value)
    assert "sample-1" in message
    assert "source.sha256" in message
    assert str(tmp_path) not in message


@pytest.mark.parametrize(
    "audio",
    [
        "/absolute.wav",
        "../escape.wav",
        "folder/../../escape.wav",
        r"C:\audio.wav",
        "C:/audio.wav",
        r"\\server\share\audio.wav",
        "missing.wav",
    ],
)
def test_manifest_rejects_unsafe_or_missing_audio_paths(tmp_path, audio):
    manifest_path, _, _ = _valid_manifest(tmp_path, audio=audio)

    with pytest.raises(ValueError, match=r"sample-1.*audio"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )


def test_manifest_allows_internal_symlink_and_rejects_escape_and_directory(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    audio_path = dataset_root / "clip.wav"
    audio_path.write_bytes(b"audio fixture")
    internal = dataset_root / "internal.wav"
    internal.symlink_to(audio_path)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio fixture")
    escaped = dataset_root / "escaped.wav"
    escaped.symlink_to(outside)
    broken = dataset_root / "broken.wav"
    broken.symlink_to(dataset_root / "absent.wav")

    internal_record = _manifest_record(audio_path, audio=internal.name)
    manifest_path = _write_manifest(tmp_path / "manifest.jsonl", [internal_record])
    samples, _ = stt_bench.load_and_validate_manifest(
        manifest_path,
        dataset_root,
        duration_probe=lambda _: 1.0,
    )
    assert samples[0].audio_relative == "internal.wav"

    directory = dataset_root / "folder"
    directory.mkdir()
    for unsafe in (escaped.name, broken.name, directory.name):
        record = _manifest_record(audio_path, audio=unsafe)
        _write_manifest(manifest_path, [record])
        with pytest.raises(ValueError, match=r"sample-1.*audio"):
            stt_bench.load_and_validate_manifest(
                manifest_path,
                dataset_root,
                duration_probe=lambda _: 1.0,
            )


def test_scheduling_revalidation_rejects_post_validation_symlink_swap(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    audio_path = dataset_root / "clip.wav"
    audio_path.write_bytes(b"audio fixture")
    link = dataset_root / "linked.wav"
    link.symlink_to(audio_path)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio fixture")
    record = _manifest_record(audio_path, audio=link.name)
    manifest_path = _write_manifest(tmp_path / "manifest.jsonl", [record])
    samples, _ = stt_bench.load_and_validate_manifest(
        manifest_path,
        dataset_root,
        duration_probe=lambda _: 1.0,
    )

    scheduled_path = stt_bench.resolve_audio_for_scheduling(
        samples[0],
        dataset_root,
    )
    link.unlink()
    link.symlink_to(outside)

    assert scheduled_path == audio_path.resolve()
    with pytest.raises(ValueError, match=r"sample-1.*audio"):
        stt_bench.resolve_audio_for_scheduling(samples[0], dataset_root)

    replacement = dataset_root / "replacement.wav"
    replacement.write_bytes(b"different audio")
    link.unlink()
    link.symlink_to(replacement)
    with pytest.raises(ValueError, match=r"sample-1.*source\.sha256"):
        stt_bench.resolve_audio_for_scheduling(samples[0], dataset_root)


def test_manifest_rejects_non_scalar_audio_path_with_field_error(tmp_path):
    manifest_path, _, _ = _valid_manifest(tmp_path, audio="\ud800.wav")

    with pytest.raises(ValueError, match=r"sample-1.*audio.*Unicode scalar"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )


def test_manifest_rejects_dataset_root_that_is_not_a_directory(tmp_path):
    manifest_path, audio_path, _ = _valid_manifest(tmp_path)

    with pytest.raises(ValueError, match="dataset root"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            audio_path,
            duration_probe=lambda _: 1.0,
        )


def test_manifest_accepts_strict_profile_for_non_english_language(tmp_path):
    manifest_path, _, _ = _valid_manifest(
        tmp_path,
        language="FR-ca",
        normalization_profile="strict-v1",
        reference="Bonjour !",
    )

    samples, _ = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )

    assert samples[0].language == "fr-ca"
    assert samples[0].normalization_profile == "strict-v1"


def test_manifest_rejects_blank_malformed_non_object_duplicate_and_unknown_fields(
    tmp_path,
):
    manifest_path, audio_path, record = _valid_manifest(tmp_path)
    cases = [
        "\n",
        "{bad json}\n",
        "[]\n",
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        json.dumps({**record, "unexpected": True}) + "\n",
    ]

    for content in cases:
        manifest_path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            stt_bench.load_and_validate_manifest(
                manifest_path,
                tmp_path,
                duration_probe=lambda _: 1.0,
            )

    assert audio_path.exists()


def test_manifest_rejects_duplicate_top_level_and_nested_json_keys(tmp_path):
    manifest_path, _, record = _valid_manifest(tmp_path)
    encoded = json.dumps(record)
    duplicate_top_level = encoded[:-1] + ',"id":"sample-2"}\n'
    encoded_source = json.dumps(record["source"])
    duplicate_source = encoded_source[:-1] + ',"license":"different"}'
    duplicate_nested = encoded.replace(encoded_source, duplicate_source) + "\n"

    for content in (duplicate_top_level, duplicate_nested):
        manifest_path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate JSON field"):
            stt_bench.load_and_validate_manifest(
                manifest_path,
                tmp_path,
                duration_probe=lambda _: 1.0,
            )


def test_manifest_requires_consistent_suite_visibility(tmp_path):
    manifest_path, audio_path, first = _valid_manifest(tmp_path)
    second = _manifest_record(
        audio_path,
        sample_id="sample-2",
        suite_visibility="private",
    )
    _write_manifest(manifest_path, [first, second])

    with pytest.raises(ValueError, match=r"sample-2.*suite_visibility"):
        stt_bench.load_and_validate_manifest(
            manifest_path,
            tmp_path,
            duration_probe=lambda _: 1.0,
        )


def test_manifest_hash_is_location_and_order_independent_but_materially_sensitive(
    tmp_path,
):
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    first_root.mkdir()
    second_root.mkdir()
    records_by_root = []
    for root in (first_root, second_root):
        audio_a = root / "a.wav"
        audio_b = root / "b.wav"
        audio_a.write_bytes(b"audio-a")
        audio_b.write_bytes(b"audio-b")
        records_by_root.append(
            [
                _manifest_record(
                    audio_a,
                    sample_id="sample-a",
                    profiles=["regression", "comparison"],
                    tags=["x", "y"],
                ),
                _manifest_record(
                    audio_b,
                    sample_id="sample-b",
                    profiles=["comparison"],
                    tags=["z"],
                ),
            ]
        )
    records_by_root[1].reverse()
    records_by_root[1][1]["profiles"].reverse()
    records_by_root[1][1]["tags"].reverse()
    records_by_root[1][1]["source"] = dict(reversed(tuple(records_by_root[1][1]["source"].items())))
    manifest_a = _write_manifest(first_root / "manifest.jsonl", records_by_root[0])
    manifest_b = _write_manifest(second_root / "renamed.jsonl", records_by_root[1])

    _, hash_a = stt_bench.load_and_validate_manifest(
        manifest_a,
        first_root,
        duration_probe=lambda _: 1.0,
    )
    _, hash_b = stt_bench.load_and_validate_manifest(
        manifest_b,
        second_root,
        duration_probe=lambda _: 1.05,
    )
    assert hash_a == hash_b

    records_by_root[1][1]["reference"] = "Changed reference"
    _write_manifest(manifest_b, records_by_root[1])
    _, changed_hash = stt_bench.load_and_validate_manifest(
        manifest_b,
        second_root,
        duration_probe=lambda _: 1.0,
    )
    assert changed_hash != hash_a


@pytest.mark.parametrize(
    "mutator",
    [
        lambda record: record.update(id="sample-2"),
        lambda record: record.update(reference="Different words"),
        lambda record: record.update(language="en-US"),
        lambda record: record.update(normalization_profile="strict-v1"),
        lambda record: record.update(duration_seconds=1.05),
        lambda record: record.update(profiles=["comparison"]),
        lambda record: record.update(suite="another-suite"),
        lambda record: record.update(suite_visibility="private"),
        lambda record: record.update(annotation_profile="manual-v1"),
        lambda record: record.update(diagnostic_only=True),
        lambda record: record["source"].update(dataset="other-dataset"),
        lambda record: record.update(tags=["other-tag"]),
    ],
)
def test_manifest_hash_changes_for_each_material_declared_field(tmp_path, mutator):
    manifest_path, _, record = _valid_manifest(tmp_path)
    _, original_hash = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )
    changed = copy.deepcopy(record)
    mutator(changed)
    _write_manifest(manifest_path, [changed])

    _, changed_hash = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )

    assert changed_hash != original_hash


def test_manifest_hash_changes_with_audio_relative_path_and_sha256(tmp_path):
    manifest_path, _, record = _valid_manifest(tmp_path)
    _, original_hash = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(b"replacement audio")
    record["audio"] = replacement.name
    record["source"]["sha256"] = hashlib.sha256(replacement.read_bytes()).hexdigest()
    _write_manifest(manifest_path, [record])

    _, changed_hash = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )

    assert changed_hash != original_hash


def test_manifest_declared_duration_is_optional_but_changes_hash(tmp_path):
    manifest_path, _, record = _valid_manifest(tmp_path)
    _, declared_hash = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )
    record.pop("duration_seconds")
    _write_manifest(manifest_path, [record])
    _, omitted_hash = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )
    assert omitted_hash != declared_hash


def test_sample_order_is_deterministic_and_returns_first_cold_probe(tmp_path):
    manifest_path, audio_path, first = _valid_manifest(tmp_path)
    records = [
        {**first, "id": sample_id, "profiles": profiles}
        for sample_id, profiles in (
            ("sample-c", ["comparison"]),
            ("sample-a", ["regression", "comparison"]),
            ("sample-b", ["regression"]),
        )
    ]
    _write_manifest(manifest_path, records)
    samples, _ = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )

    selected, cold_probe = stt_bench.select_samples(
        samples,
        profile="regression",
        seed=7,
    )
    expected = sorted(
        ["sample-a", "sample-b"],
        key=lambda sample_id: hashlib.sha256(f"7\0{sample_id}".encode()).hexdigest(),
    )
    assert [sample.sample_id for sample in selected] == expected
    assert cold_probe == expected[0]
    assert audio_path.exists()


@pytest.mark.parametrize(
    ("profile", "seed", "error"),
    [
        ("unknown", 1, ValueError),
        ("Bad Profile", 1, ValueError),
        ("regression", True, TypeError),
        ("regression", 1.0, TypeError),
    ],
)
def test_sample_order_rejects_invalid_profile_or_seed(
    tmp_path,
    profile,
    seed,
    error,
):
    manifest_path, _, _ = _valid_manifest(tmp_path, profiles=["comparison"])
    samples, _ = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )
    with pytest.raises(error):
        stt_bench.select_samples(samples, profile=profile, seed=seed)


def test_sample_order_rejects_empty_profile_selection(tmp_path):
    manifest_path, _, _ = _valid_manifest(tmp_path, profiles=["comparison"])
    samples, _ = stt_bench.load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _: 1.0,
    )
    with pytest.raises(ValueError, match="no samples"):
        stt_bench.select_samples(samples, profile="regression", seed=1)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"streams": []},
        {"streams": [{"duration": "0"}], "format": {"duration": "-1"}},
        {"streams": [{"duration": "nan"}], "format": {"duration": "inf"}},
        {"streams": [{"duration": True}]},
    ],
)
def test_ffprobe_rejects_missing_or_nonpositive_duration(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(
        stt_bench.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="duration"):
        stt_bench.probe_audio_duration_ffprobe(tmp_path / "audio.wav")


def test_ffprobe_prefers_stream_duration_and_uses_exact_arguments(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(
                {
                    "streams": [{"duration": "1.25"}],
                    "format": {"duration": "2.5"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(stt_bench.subprocess, "run", fake_run)
    audio_path = tmp_path / "audio.wav"
    assert stt_bench.probe_audio_duration_ffprobe(audio_path) == 1.25
    assert calls[0][0][0] == [
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


def test_ffprobe_falls_back_to_positive_format_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stt_bench.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(
                {
                    "streams": [{"duration": "N/A"}],
                    "format": {"duration": "2.5"},
                }
            ),
            stderr="",
        ),
    )

    assert stt_bench.probe_audio_duration_ffprobe(tmp_path / "audio.wav") == 2.5


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(),
        subprocess.CompletedProcess(
            ["ffprobe"],
            1,
            stdout="",
            stderr="failure",
        ),
        subprocess.CompletedProcess(
            ["ffprobe"],
            0,
            stdout="{bad",
            stderr="",
        ),
    ],
)
def test_ffprobe_fails_safely_when_missing_failed_or_malformed(
    tmp_path,
    monkeypatch,
    failure,
):
    def fake_run(*args, **kwargs):
        if isinstance(failure, BaseException):
            raise failure
        return failure

    monkeypatch.setattr(stt_bench.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="ffprobe"):
        stt_bench.probe_audio_duration_ffprobe(tmp_path / "audio.wav")


@pytest.mark.integration
def test_ffprobe_generated_wav_integration(tmp_path):
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")
    audio_path = tmp_path / "tone.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 8_000)

    assert stt_bench.probe_audio_duration_ffprobe(audio_path) == pytest.approx(1.0)


def test_manifest_validate_cli_prints_only_safe_deterministic_summary(
    tmp_path,
    monkeypatch,
    capsys,
):
    manifest_path, _, _ = _valid_manifest(tmp_path)
    samples = (
        stt_bench.ManifestSample(
            "sample-1",
            "clip.wav",
            "TOP SECRET REFERENCE",
            "en",
            "en-v1",
            1.0,
            ("comparison", "regression"),
            "public-english-v1",
            "public",
            "canonical-v1",
            False,
            (("dataset", "fixture"),),
            ("read-speech",),
            "a" * 64,
        ),
    )
    monkeypatch.setattr(
        stt_bench,
        "load_and_validate_manifest",
        lambda *args, **kwargs: (samples, "b" * 64),
    )

    assert (
        stt_bench.main(
            [
                "validate",
                "--manifest",
                str(manifest_path),
                "--dataset-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "manifest_hash": "b" * 64,
        "profiles": {"comparison": 1, "regression": 1},
        "sample_count": 1,
        "suites": {"public-english-v1": 1},
        "visibility": {"public": 1},
    }
    assert "TOP SECRET" not in output.out
    assert str(tmp_path) not in output.out
    assert output.err == ""


def test_manifest_validate_cli_returns_nonzero_without_leaking_reference(
    tmp_path,
    monkeypatch,
    capsys,
):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        stt_bench,
        "load_and_validate_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("sample-1 field reference is invalid")),
    )
    assert (
        stt_bench.main(
            [
                "validate",
                "--manifest",
                str(manifest_path),
                "--dataset-root",
                str(tmp_path),
            ]
        )
        != 0
    )
    output = capsys.readouterr()
    assert "sample-1" in output.err
    assert str(tmp_path) not in output.err


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
    assert pooled.rate == sum(item.errors for item in samples) / sum(item.reference_units for item in samples)


def _edit_payload(counts):
    return {
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "reference_units": counts.reference_units,
        "errors": counts.errors,
        "rate": counts.rate,
    }


def _result_record(
    *,
    run_id="run-1",
    target_id="target-1",
    sample_id="sample-1",
    repetition=0,
    attempt_id=1,
    worker_attempt_id=1,
    reference="hello world",
    hypothesis="hello world",
    status="ok",
    measurement_role="accuracy",
    timing_class="warm",
    suite="public-english-v1",
    suite_visibility="public",
    dataset="fixture",
    tags=("read-speech",),
    diagnostic_only=False,
    backend="local",
    adapter_nanoseconds=500_000_000,
    audio_duration_seconds=1.0,
):
    scored_hypothesis = hypothesis if status == "ok" else ""
    score = score_transcript(
        reference,
        scored_hypothesis,
        normalization_profile=EN_PROFILE,
    )
    rtf = (
        adapter_nanoseconds / 1_000_000_000 / audio_duration_seconds
        if (
            isinstance(adapter_nanoseconds, int)
            and adapter_nanoseconds > 0
            and isinstance(audio_duration_seconds, (int, float))
            and audio_duration_seconds > 0
        )
        else None
    )
    throughput = 1 / rtf if rtf else None
    return {
        "schema_version": 1,
        "run_id": run_id,
        "target_id": target_id,
        "completion_key": stt_bench.completion_key(
            "a" * 64,
            target_id,
            "b" * 64,
            sample_id,
            repetition,
        ),
        "sample_id": sample_id,
        "repetition": repetition,
        "attempt_id": attempt_id,
        "worker_attempt_id": worker_attempt_id,
        "measurement_role": measurement_role,
        "timing_class": timing_class,
        "suite": suite,
        "suite_visibility": suite_visibility,
        "dataset": dataset,
        "tags": list(tags),
        "diagnostic_only": diagnostic_only,
        "requested_execution": {
            "provider": "fake",
            "model_label": "fake-model",
        },
        "actual_execution": {
            "route_id": "route-1",
            "provider": "fake",
            "model_label": "fake-model",
            "artifact_id": None,
            "backend": backend,
            "audio_egress": "none",
            "endpoint_id": None,
            "source": "fake",
            "device": "cpu",
            "compute_type": None,
            "dtype": None,
            "decoding_ids": [],
            "transport": None,
        },
        "execution_mismatch_reasons": [],
        "eligibility_reasons": [],
        "status": status,
        "reference": reference,
        "hypothesis": hypothesis,
        "scorer_version": SCORER_VERSION,
        "strict_profile": STRICT_PROFILE,
        "normalization_profile": EN_PROFILE,
        "exact_match": score.exact_match,
        "strict": {
            "wer": _edit_payload(score.strict_wer),
            "cer": _edit_payload(score.strict_cer),
        },
        "normalized": {
            "wer": _edit_payload(score.normalized_wer),
            "cer": _edit_payload(score.normalized_cer),
        },
        "adapter_nanoseconds": adapter_nanoseconds,
        "audio_duration_seconds": audio_duration_seconds,
        "rtf": rtf,
        "throughput": throughput,
        "resource_observations": None,
        "error": None,
    }


def test_persistence_schema_constants_and_discriminators_are_pinned():
    assert stt_bench.RUN_SCHEMA_VERSION == 1
    assert stt_bench.RESULT_SCHEMA_VERSION == 1
    assert stt_bench.SUMMARY_SCHEMA_VERSION == 1
    assert (
        frozenset(
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
        == stt_bench.RESULT_STATUSES
    )
    assert frozenset({"accuracy", "performance_repeat"}) == (stt_bench.MEASUREMENT_ROLES)
    assert frozenset({"cold_first", "warmup_recovery", "warm"}) == (stt_bench.TIMING_CLASSES)


def test_persist_completion_key_is_stable_and_binds_every_component():
    arguments = ("a" * 64, "target-1", "b" * 64, "sample-1", 0)
    first = stt_bench.completion_key(*arguments)

    assert first == stt_bench.completion_key(*arguments)
    assert len(first) == 64
    for index, replacement in enumerate(("c" * 64, "target-2", "d" * 64, "sample-2", 1)):
        changed = list(arguments)
        changed[index] = replacement
        assert stt_bench.completion_key(*changed) != first


@pytest.mark.parametrize("repetition", [True, -1, 1.5])
def test_persist_completion_key_rejects_invalid_repetition(repetition):
    with pytest.raises((TypeError, ValueError)):
        stt_bench.completion_key(
            "a" * 64,
            "target-1",
            "b" * 64,
            "sample-1",
            repetition,
        )


def test_atomic_persist_replaces_in_same_directory_and_fsyncs_file_and_parent(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "run" / "run.json"
    destination.parent.mkdir(mode=0o700)
    destination.parent.chmod(0o700)
    destination.write_text('{"old":true}\n', encoding="utf-8")
    replacements = []
    fsync_calls = []
    real_replace = os.replace
    real_fsync = os.fsync

    def recording_replace(source, target):
        replacements.append((Path(source), Path(target)))
        return real_replace(source, target)

    def recording_fsync(file_descriptor):
        fsync_calls.append(file_descriptor)
        return real_fsync(file_descriptor)

    monkeypatch.setattr(os, "replace", recording_replace)
    monkeypatch.setattr(os, "fsync", recording_fsync)

    stt_bench.atomic_write_json(destination, {"schema_version": 1, "run_id": "r"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "run_id": "r",
    }
    assert len(replacements) == 1
    assert replacements[0][0].parent == destination.parent
    assert replacements[0][1] == destination
    assert len(fsync_calls) >= 2
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
        assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700


def test_persist_append_result_record_is_owner_only_and_fsynced(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "run" / "results.jsonl"
    fsync_calls = []
    real_fsync = os.fsync

    def recording_fsync(file_descriptor):
        fsync_calls.append(file_descriptor)
        return real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    record = _result_record()

    stt_bench.append_result_record(destination, record)

    assert json.loads(destination.read_text(encoding="utf-8")) == record
    assert fsync_calls
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
        assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "mutator",
    [
        lambda record: record.pop("strict"),
        lambda record: record.update(schema_version=2),
        lambda record: record.update(status="mystery"),
        lambda record: record.update(measurement_role="setup"),
        lambda record: record.update(timing_class="lukewarm"),
        lambda record: record.update(unexpected="secret"),
    ],
)
def test_persist_result_rejects_incomplete_or_unknown_record(mutator, tmp_path):
    record = _result_record()
    mutator(record)

    with pytest.raises(ValueError):
        stt_bench.append_result_record(tmp_path / "results.jsonl", record)


def test_persist_load_history_ignores_and_reports_only_truncated_final_line(
    tmp_path,
):
    first = _result_record(attempt_id=1)
    second = _result_record(sample_id="sample-2", attempt_id=2)
    destination = tmp_path / "results.jsonl"
    destination.write_text(
        json.dumps(first) + "\n" + json.dumps(second)[:20],
        encoding="utf-8",
    )

    records, truncated = stt_bench.load_result_history(destination)

    assert records == [first]
    assert truncated is True


def test_persist_load_history_rejects_earlier_malformed_line(tmp_path):
    destination = tmp_path / "results.jsonl"
    destination.write_text(
        "{broken}\n" + json.dumps(_result_record(attempt_id=2)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        stt_bench.load_result_history(destination)


def test_persist_load_history_accepts_missing_file(tmp_path):
    assert stt_bench.load_result_history(tmp_path / "missing.jsonl") == ([], False)


@pytest.mark.parametrize(
    "attempts",
    [
        (1, 1),
        (2, 1),
    ],
)
def test_attempt_history_rejects_duplicate_or_non_monotonic_global_ids(
    tmp_path,
    attempts,
):
    destination = tmp_path / "results.jsonl"
    destination.write_text(
        "".join(
            json.dumps(_result_record(sample_id=f"sample-{index}", attempt_id=attempt)) + "\n"
            for index, attempt in enumerate(attempts, start=1)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="attempt"):
        stt_bench.load_result_history(destination)


def test_reduce_attempts_selects_highest_attempt_by_stable_completion_key():
    first = _result_record(attempt_id=1, status="adapter_error", hypothesis="")
    retry = _result_record(attempt_id=3, status="ok")
    other = _result_record(sample_id="sample-2", attempt_id=2)

    active = stt_bench.reduce_attempts([first, other, retry])

    assert list(active) == sorted(active)
    assert active[first["completion_key"]] == retry
    assert active[other["completion_key"]] == other


def _inflight_record(**overrides):
    record = {
        "target_id": "target-1",
        "operation_id": 3,
        "operation_role": "result_call",
        "worker_attempt_id": 2,
        "sample_id": "sample-1",
        "completion_key": stt_bench.completion_key(
            "a" * 64,
            "target-1",
            "b" * 64,
            "sample-1",
            0,
        ),
        "repetition": 0,
        "result_attempt_id": 4,
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    "overrides",
    [
        {"operation_role": "unknown"},
        {"operation_id": 0},
        {"result_attempt_id": None},
        {"completion_key": None},
        {"repetition": None},
        {"extra": "not-allowlisted"},
        {
            "operation_role": "rewarm_probe",
            "result_attempt_id": 4,
        },
    ],
)
def test_inflight_validation_rejects_invalid_discriminators_and_ids(overrides):
    with pytest.raises(ValueError):
        stt_bench.validate_inflight_record(_inflight_record(**overrides))


def test_inflight_rewarm_probe_requires_prior_key_and_allocates_no_result_attempt():
    run_metadata = {
        "schema_version": stt_bench.RUN_SCHEMA_VERSION,
        "next_operation_id": 7,
        "next_attempt_id": 11,
    }

    updated, inflight = stt_bench.allocate_inflight(
        run_metadata,
        target_id="target-1",
        operation_role="rewarm_probe",
        worker_attempt_id=2,
        sample_id="sample-1",
        completion_key=_inflight_record()["completion_key"],
        repetition=None,
    )

    assert updated["next_operation_id"] == 8
    assert updated["next_attempt_id"] == 11
    assert inflight["operation_id"] == 7
    assert inflight["result_attempt_id"] is None


def test_inflight_result_allocation_advances_both_global_counters_once():
    run_metadata = {
        "schema_version": stt_bench.RUN_SCHEMA_VERSION,
        "next_operation_id": 7,
        "next_attempt_id": 11,
    }

    updated, inflight = stt_bench.allocate_inflight(
        run_metadata,
        target_id="target-1",
        operation_role="result_call",
        worker_attempt_id=2,
        sample_id="sample-1",
        completion_key=_inflight_record()["completion_key"],
        repetition=0,
    )

    assert updated["next_operation_id"] == 8
    assert updated["next_attempt_id"] == 12
    assert inflight["operation_id"] == 7
    assert inflight["result_attempt_id"] == 11


@pytest.mark.parametrize(
    ("active", "retry_errors", "expected"),
    [
        (None, False, "execute"),
        (_result_record(status="ok"), False, "skip"),
        (_result_record(status="adapter_error", hypothesis=""), False, "skip"),
        (_result_record(status="adapter_error", hypothesis=""), True, "retry"),
        (_result_record(status="empty", hypothesis=""), True, "retry"),
        (_result_record(status="ok"), True, "skip"),
    ],
)
def test_attempt_resume_action_skips_terminal_records_and_retries_only_errors(
    active,
    retry_errors,
    expected,
):
    assert stt_bench.resume_action(active, retry_errors=retry_errors) == expected


def test_inflight_recovery_clears_when_exact_terminal_result_exists():
    record = _result_record(attempt_id=4)

    action = stt_bench.recover_inflight_action(
        _inflight_record(result_attempt_id=4),
        [record],
    )

    assert action == {"action": "clear", "status": None}


@pytest.mark.parametrize(
    ("interrupted", "timed_out", "expected"),
    [
        (False, False, "worker_crash"),
        (True, False, "interrupted"),
        (False, True, "timeout"),
    ],
)
def test_inflight_recovery_attributes_uncommitted_result_call(
    interrupted,
    timed_out,
    expected,
):
    action = stt_bench.recover_inflight_action(
        _inflight_record(),
        [],
        interrupted=interrupted,
        timed_out=timed_out,
    )

    assert action == {"action": "append_result", "status": expected}


def test_inflight_recovery_never_appends_result_for_rewarm_probe():
    action = stt_bench.recover_inflight_action(
        _inflight_record(
            operation_role="rewarm_probe",
            repetition=None,
            result_attempt_id=None,
        ),
        [],
        timed_out=True,
    )

    assert action == {"action": "record_rewarm", "status": "timeout"}


@pytest.mark.parametrize("status", sorted(stt_bench.RESULT_STATUSES - {"ok"}))
def test_attempt_failure_statuses_are_scored_as_empty_hypotheses(status):
    score = stt_bench.score_result_text(
        "one two",
        "selectively perfect text",
        status=status,
        normalization_profile=EN_PROFILE,
    )

    assert score.normalized_wer.deletions == 2
    assert score.normalized_wer.insertions == 0


def test_retention_is_applied_after_scoring_and_errors_only_keeps_failures():
    score = stt_bench.score_result_text(
        "Hello",
        "hello",
        status="ok",
        normalization_profile=EN_PROFILE,
    )

    assert stt_bench.retain_text(
        mode="errors-only",
        status="ok",
        reference="Hello",
        hypothesis="hello",
        score=score,
    ) == ("Hello", "hello")
    assert stt_bench.retain_text(
        mode="none",
        status="ok",
        reference="Hello",
        hypothesis="hello",
        score=score,
    ) == (None, None)


def test_retention_errors_only_discards_exact_success_and_keeps_failure():
    success = score_transcript("hello", "hello", normalization_profile=EN_PROFILE)
    failure = score_transcript("hello", "", normalization_profile=EN_PROFILE)

    assert stt_bench.retain_text(
        mode="errors-only",
        status="ok",
        reference="hello",
        hypothesis="hello",
        score=success,
    ) == (None, None)
    assert stt_bench.retain_text(
        mode="errors-only",
        status="timeout",
        reference="hello",
        hypothesis="",
        score=failure,
    ) == ("hello", "")


def test_sanitize_error_redacts_credentials_urls_paths_and_controls():
    hostile = RuntimeError(
        "Authorization: Bearer abc123\n"
        "api_key=sk-supersecret token=qwerty secret=hunter2 "
        "https://user:pass@example.test/audio?sig=private "
        "/Users/alice/.cache/model.bin C:\\Users\\alice\\model.bin "
        "\\\\server\\share\\model.bin\tfinished"
    )

    sanitized = stt_bench.sanitize_error(hostile)

    assert sanitized["type"] == "RuntimeError"
    lowered = sanitized["message"].casefold()
    for leaked in (
        "abc123",
        "supersecret",
        "qwerty",
        "hunter2",
        "user:pass",
        "sig=private",
        "/users/alice",
        "c:\\users",
        "\\\\server\\share",
        "\n",
        "\t",
    ):
        assert leaked not in lowered
    assert len(sanitized["message"]) <= 512


@pytest.mark.parametrize(
    ("adapter_nanoseconds", "audio_duration", "eligible"),
    [
        (500_000_000, 2.0, True),
        (0, 2.0, False),
        (-1, 2.0, False),
        (500_000_000, 0.0, False),
        (500_000_000, float("nan"), False),
        (float("inf"), 2.0, False),
    ],
)
def test_attempt_performance_fields_require_positive_finite_durations(
    adapter_nanoseconds,
    audio_duration,
    eligible,
):
    rtf, throughput, reasons = stt_bench.performance_fields(
        adapter_nanoseconds,
        audio_duration,
        eligibility_reasons=[],
    )

    if eligible:
        assert rtf == 0.25
        assert throughput == 4.0
        assert reasons == []
    else:
        assert rtf is None
        assert throughput is None
        assert reasons == ["invalid_performance_duration"]


def _aggregate_fixture():
    records = [
        _result_record(
            sample_id="probe",
            attempt_id=1,
            timing_class="cold_first",
            adapter_nanoseconds=1_000_000_000,
            audio_duration_seconds=1.0,
        ),
        _result_record(
            sample_id="warm-good",
            attempt_id=2,
            dataset="dataset-a",
            tags=("read-speech", "clean"),
            adapter_nanoseconds=500_000_000,
            audio_duration_seconds=1.0,
        ),
        _result_record(
            sample_id="warm-failure",
            attempt_id=3,
            reference="one two",
            hypothesis="",
            status="adapter_error",
            dataset="dataset-a",
            tags=("read-speech",),
            adapter_nanoseconds=1_000_000_000,
            audio_duration_seconds=2.0,
        ),
        _result_record(
            sample_id="warm-good",
            repetition=1,
            attempt_id=4,
            measurement_role="performance_repeat",
            dataset="dataset-a",
            tags=("read-speech", "clean"),
            adapter_nanoseconds=250_000_000,
            audio_duration_seconds=1.0,
        ),
        _result_record(
            sample_id="recovered",
            attempt_id=5,
            measurement_role="performance_repeat",
            timing_class="warmup_recovery",
            adapter_nanoseconds=100_000_000,
            audio_duration_seconds=1.0,
        ),
        _result_record(
            sample_id="diagnostic",
            attempt_id=6,
            reference="one two",
            hypothesis="wrong",
            diagnostic_only=True,
            adapter_nanoseconds=10_000_000,
            audio_duration_seconds=1.0,
        ),
        _result_record(
            sample_id="private",
            attempt_id=7,
            suite="private-english-v1",
            suite_visibility="private",
            dataset="dataset-private",
            tags=("private",),
            backend="remote",
            adapter_nanoseconds=2_000_000_000,
            audio_duration_seconds=1.0,
        ),
    ]
    active = {record["completion_key"]: record for record in records}
    metadata = {
        "schema_version": stt_bench.RUN_SCHEMA_VERSION,
        "run_id": "run-1",
        "cold_probe_sample_id": "probe",
    }
    return metadata, active


def test_aggregate_results_reports_suite_quality_without_cross_suite_pooling():
    metadata, active = _aggregate_fixture()

    summary = stt_bench.aggregate_results(metadata, active)

    assert summary["schema_version"] == stt_bench.SUMMARY_SCHEMA_VERSION
    assert summary["run_id"] == "run-1"
    target = summary["primary"]["targets"]["target-1"]
    assert set(target["suites"]) == {
        "private-english-v1",
        "public-english-v1",
    }
    public = target["suites"]["public-english-v1"]
    assert public["sample_count"] == 3
    assert public["success_count"] == 2
    assert public["empty_count"] == 0
    assert public["failure_count"] == 1
    assert public["exact_match_rate"] == pytest.approx(2 / 3)
    assert public["normalized"]["wer"]["pooled"] == pytest.approx(1 / 3)
    assert public["normalized"]["wer"]["mean"] == pytest.approx(1 / 3)
    assert public["normalized"]["wer"]["p50"] == 0.0
    assert public["normalized"]["wer"]["p90"] == pytest.approx(0.8)
    assert public["normalized"]["wer"]["p95"] == pytest.approx(0.9)
    assert public["normalized"]["wer"]["p99"] == pytest.approx(0.98)


def test_aggregate_results_separates_diagnostics_and_suite_scoped_slices():
    metadata, active = _aggregate_fixture()

    summary = stt_bench.aggregate_results(metadata, active)

    assert summary["diagnostic"]["targets"]["target-1"]["suites"]["public-english-v1"]["sample_count"] == 1
    assert summary["slices"]["dataset"]["target-1"]["dataset-a"]["public-english-v1"]["sample_count"] == 2
    assert summary["slices"]["tag"]["target-1"]["clean"]["public-english-v1"]["sample_count"] == 1
    assert summary["slices"]["actual_backend"]["target-1"]["local"]["public-english-v1"]["sample_count"] == 3
    assert summary["slices"]["actual_backend"]["target-1"]["remote"]["private-english-v1"]["sample_count"] == 1


def test_aggregate_results_uses_only_successful_warm_calls_for_performance():
    metadata, active = _aggregate_fixture()

    summary = stt_bench.aggregate_results(metadata, active)

    warm = summary["performance"]["warm"]["targets"]["target-1"]["suites"]["public-english-v1"]
    assert warm["observation_count"] == 2
    assert warm["adapter_seconds"]["mean"] == pytest.approx(0.375)
    assert warm["rtf"]["p50"] == pytest.approx(0.375)
    assert warm["throughput"]["p50"] == pytest.approx(3.0)
    cold = summary["performance"]["cold_first"]["target-1"]
    assert cold["sample_id"] == "probe"
    assert cold["adapter_seconds"] == 1.0
    assert cold["rtf"] == 1.0
    assert cold["throughput"] == 1.0


def test_aggregate_results_excludes_invalid_warm_timing_from_percentiles():
    metadata, active = _aggregate_fixture()
    invalid = _result_record(
        sample_id="invalid-timing",
        attempt_id=8,
        adapter_nanoseconds=None,
        audio_duration_seconds=None,
    )
    invalid["eligibility_reasons"] = ["invalid_performance_duration"]
    active[invalid["completion_key"]] = invalid

    summary = stt_bench.aggregate_results(metadata, active)

    warm = summary["performance"]["warm"]["targets"]["target-1"]["suites"]["public-english-v1"]
    assert warm["observation_count"] == 2
    assert warm["ineligible_count"] == 1


@pytest.mark.parametrize(
    "mutator",
    [
        lambda metadata, active: metadata.update(schema_version=2),
        lambda metadata, active: metadata.update(run_id="other-run"),
        lambda metadata, active: active.update({"wrong-key": next(iter(active.values()))}),
    ],
)
def test_aggregate_results_rejects_incompatible_inputs(mutator):
    metadata, active = _aggregate_fixture()
    mutator(metadata, active)

    with pytest.raises(ValueError):
        stt_bench.aggregate_results(metadata, active)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda record: record.update(rtf=99.0),
        lambda record: record.update(throughput=None),
        lambda record: record.update(
            adapter_nanoseconds=0,
            rtf=None,
            throughput=None,
        ),
        lambda record: record.update(
            audio_duration_seconds=None,
            rtf=None,
            throughput=None,
        ),
    ],
)
def test_persist_result_rejects_inconsistent_performance_fields(
    mutator,
    tmp_path,
):
    record = _result_record()
    mutator(record)

    with pytest.raises(ValueError, match="performance"):
        stt_bench.append_result_record(tmp_path / "results.jsonl", record)


@pytest.mark.parametrize(
    "observations",
    [
        {"collection_method": ""},
        {"collection_method": "psutil", "peak_rss_bytes": -1},
        {"collection_method": "psutil", "rss_before_bytes": True},
        {"collection_method": "psutil", "gpu_memory_bytes": 1.5},
    ],
)
def test_persist_result_rejects_invalid_resource_observation_values(
    observations,
    tmp_path,
):
    record = _result_record()
    record["resource_observations"] = observations

    with pytest.raises(ValueError, match="resource"):
        stt_bench.append_result_record(tmp_path / "results.jsonl", record)


def test_persist_result_rejects_retained_failure_scored_as_success(tmp_path):
    record = _result_record()
    record["status"] = "timeout"

    with pytest.raises(ValueError, match="score"):
        stt_bench.append_result_record(tmp_path / "results.jsonl", record)


def test_aggregate_empty_output_counts_as_failure_and_empty_subtype():
    record = _result_record(status="empty", hypothesis="")
    summary = stt_bench.aggregate_results(
        {
            "schema_version": stt_bench.RUN_SCHEMA_VERSION,
            "run_id": "run-1",
            "cold_probe_sample_id": None,
        },
        {record["completion_key"]: record},
    )

    suite = summary["primary"]["targets"]["target-1"]["suites"]["public-english-v1"]
    assert suite["success_count"] == 0
    assert suite["failure_count"] == 1
    assert suite["empty_count"] == 1
    assert suite["error_count"] == 0


def test_inflight_recovery_does_not_accept_another_sample_terminal_record():
    record = _result_record(sample_id="sample-2", attempt_id=4)
    inflight = _inflight_record(result_attempt_id=4)

    action = stt_bench.recover_inflight_action(inflight, [record])

    assert action == {"action": "append_result", "status": "worker_crash"}


def test_sanitize_error_redacts_env_style_credentials_and_quoted_paths():
    hostile = ValueError(
        "ACCESS_TOKEN=abc CLIENT_SECRET=def "
        "'/Users/alice/Model Cache/private/model.bin' "
        '"C:\\Users\\alice\\Model Cache\\private\\model.bin"'
    )

    sanitized = stt_bench.sanitize_error(hostile)["message"].casefold()

    for leaked in ("abc", "def", "model cache", "private", "users\\alice"):
        assert leaked not in sanitized


def test_aggregate_results_never_pools_distinct_targets():
    first = _result_record(target_id="target-1", attempt_id=1)
    second = _result_record(target_id="target-2", attempt_id=2)

    summary = stt_bench.aggregate_results(
        {
            "schema_version": stt_bench.RUN_SCHEMA_VERSION,
            "run_id": "run-1",
            "cold_probe_sample_id": None,
        },
        {
            first["completion_key"]: first,
            second["completion_key"]: second,
        },
    )

    assert set(summary["primary"]["targets"]) == {"target-1", "target-2"}
    assert set(summary["slices"]["dataset"]) == {"target-1", "target-2"}
    assert set(summary["performance"]["warm"]["targets"]) == {
        "target-1",
        "target-2",
    }
    for target in ("target-1", "target-2"):
        assert summary["primary"]["targets"][target]["suites"]["public-english-v1"]["sample_count"] == 1


def test_persist_result_rejects_ok_status_with_retained_empty_hypothesis(
    tmp_path,
):
    record = _result_record(hypothesis="")

    with pytest.raises(ValueError, match="empty"):
        stt_bench.append_result_record(tmp_path / "results.jsonl", record)


def test_persist_result_rejects_empty_status_with_nonempty_hypothesis(tmp_path):
    record = _result_record()
    record["status"] = "empty"

    with pytest.raises(ValueError, match="empty"):
        stt_bench.append_result_record(tmp_path / "results.jsonl", record)


@pytest.mark.parametrize(
    "message",
    [
        "Authorization: Basic LEAKME\nsafe context",
        'authorization="Basic LEAK ME"\nsafe context',
    ],
)
def test_sanitize_error_redacts_complete_authorization_value(message):
    sanitized = stt_bench.sanitize_error(RuntimeError(message))["message"]

    assert "LEAK" not in sanitized
    assert "safe context" in sanitized


@pytest.mark.parametrize(
    ("envelope", "field", "value"),
    [
        ("requested_execution", "provider", True),
        ("requested_execution", "provider", None),
        ("requested_execution", "model_label", False),
        ("requested_execution", "model_label", None),
        ("actual_execution", "provider", True),
        ("actual_execution", "provider", None),
        ("actual_execution", "model_label", False),
        ("actual_execution", "model_label", None),
        ("actual_execution", "backend", True),
        ("actual_execution", "backend", None),
    ],
)
def test_persist_result_rejects_invalid_execution_identities(
    envelope,
    field,
    value,
    tmp_path,
):
    record = _result_record()
    record[envelope][field] = value

    with pytest.raises(ValueError, match="execution"):
        stt_bench.append_result_record(tmp_path / "results.jsonl", record)


def test_persist_result_requires_complete_actual_execution_envelope(tmp_path):
    record = _result_record()
    record["actual_execution"].pop("source")

    with pytest.raises(ValueError, match="execution"):
        stt_bench.append_result_record(tmp_path / "results.jsonl", record)


def test_atomic_persist_refuses_to_chmod_unrelated_existing_parent(tmp_path):
    if os.name != "posix":
        pytest.skip("POSIX mode bits are unavailable")
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)

    with pytest.raises(PermissionError, match="owner-only"):
        stt_bench.atomic_write_json(shared / "run.json", {"run_id": "run-1"})

    if os.name == "posix":
        assert stat.S_IMODE(shared.stat().st_mode) == 0o755


def test_persist_result_round_trips_canonical_actual_execution(tmp_path):
    actual = SttActualExecution(
        route_id="route-1",
        provider="external",
        model_label="external:MyProvider",
        artifact_id=None,
        backend="remote",
        audio_egress=SttAudioEgress.REMOTE,
        endpoint_id="sha256:" + "c" * 64,
        source="external",
        device=None,
        compute_type=None,
        dtype=None,
        decoding_ids=("configuration_id", "prompt_present"),
        transport="https",
    )
    record = _result_record()
    record["requested_execution"]["model_label"] = "external:MyProvider"
    record["actual_execution"] = actual.as_safe_dict()

    stt_bench.append_result_record(tmp_path / "run" / "results.jsonl", record)

    persisted = json.loads((tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8"))
    assert persisted["actual_execution"] == actual.as_safe_dict()


@pytest.mark.parametrize(
    "decoding_ids",
    [
        ["secret"],
        ["prompt_present", "configuration_id"],
    ],
)
def test_persist_result_rejects_noncanonical_decoding_ids(
    decoding_ids,
    tmp_path,
):
    record = _result_record()
    record["actual_execution"]["decoding_ids"] = decoding_ids

    with pytest.raises(ValueError, match="execution"):
        stt_bench.append_result_record(tmp_path / "results.jsonl", record)


def _planned_target(
    *,
    provider: str = "fake",
    resolved_provider: str | None = None,
    model_label: str = "model-a",
    egress: SttAudioEgress = SttAudioEgress.NONE,
    local_model_available: bool = True,
    would_download: bool = False,
    route_count: int = 1,
    runtime_secret: str = "/private/models/secret-model",
    honors_task: bool = True,
) -> SttBatchExecutionPlan:
    resolved_provider = resolved_provider or provider
    routes = tuple(
        SttExecutionRoute(
            route_id=f"route-{index + 1}",
            provider=resolved_provider,
            model_label=model_label,
            artifact_id=None,
            identity_resolved=False,
            backend=f"backend-{index + 1}",
            source="local" if egress is SttAudioEgress.NONE else "http",
            audio_egress=egress,
            endpoint_id=(None if egress is SttAudioEgress.NONE else f"sha256:{index + 1:064x}"),
            device="cpu" if egress is SttAudioEgress.NONE else None,
            compute_type="int8" if egress is SttAudioEgress.NONE else None,
            dtype=None,
            decoding_ids=(),
            local_model_available=local_model_available,
            would_download=would_download,
            transport=None if egress is SttAudioEgress.NONE else "httpx",
        )
        for index in range(route_count)
    )
    descriptor = SttExecutionDescriptor(
        requested_provider=provider,
        requested_model_label=model_label,
        resolved_provider=resolved_provider,
        resolved_model_label=model_label,
        routes=routes,
        honors_task=honors_task,
        honors_language=True,
        honors_prompt_absence=True,
        honors_hotword_absence=True,
        honors_diarization=True,
        honors_word_timestamps=True,
        decoding_settings=(),
        source_modules=("tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract",),
        dependency_distributions=("pytest",),
    )
    return SttBatchExecutionPlan(
        descriptor=descriptor,
        task="transcribe",
        language="en",
        runtime_settings=(("model_path", runtime_secret),),
    )


@pytest.fixture
def preflight_fakes():
    _PREFLIGHT_PLANS.clear()
    _PREFLIGHT_CALLS.clear()
    _PREFLIGHT_UNAVAILABLE.clear()
    _PREFLIGHT_CANONICAL.clear()
    yield
    _PREFLIGHT_PLANS.clear()
    _PREFLIGHT_CALLS.clear()
    _PREFLIGHT_UNAVAILABLE.clear()
    _PREFLIGHT_CANONICAL.clear()


def _preflight_settings(**overrides):
    settings = {
        "git_commit": "a" * 40,
        "language": "en",
        "task": "transcribe",
        "word_timestamps": False,
        "prompt": None,
        "hotwords": (),
        "diarization": False,
    }
    settings.update(overrides)
    return settings


def test_prepared_target_and_worker_settings_are_frozen_pickleable_and_secret_safe():
    plan = _planned_target()
    contract_json, contract_hash = stt_bench.build_execution_contract(
        plan=plan,
        git_commit="a" * 40,
        safe_target_settings={
            "mode": "neutral-v1",
            "task": "transcribe",
            "language": "en",
            "word_timestamps": False,
            "diarization": False,
            "prompt_present": False,
            "hotword_count": 0,
        },
    )
    target = stt_bench.PreparedTarget(
        target_id="target-0123456789abcdef",
        provider="fake",
        model_label="model-a",
        plan=plan,
        adapter_factory_path="tests:factory",
        execution_contract_json=contract_json,
        execution_contract_hash=contract_hash,
    )
    settings = stt_bench.WorkerSettings(
        run_id="run-1",
        results_path="results.jsonl",
        normalization_profile="en-v1",
        cold_probe_sample_id="sample-1",
        warm_repetitions=1,
        timing_sample_ids=("sample-2",),
        text_retention="full",
        retry_errors=False,
    )

    assert pickle.loads(pickle.dumps(target)) == target
    assert pickle.loads(pickle.dumps(settings)) == settings
    assert "/private/models/secret-model" not in repr(target)
    assert "tests:factory" not in repr(target)
    with pytest.raises(FrozenInstanceError):
        target.provider = "changed"


def test_build_execution_contract_is_canonical_deterministic_and_excludes_runtime_secrets():
    plan = _planned_target(runtime_secret="/Users/alice/.cache/private-model")
    safe_settings = {
        "mode": "neutral-v1",
        "task": "transcribe",
        "language": "en",
        "word_timestamps": False,
        "diarization": False,
        "prompt_present": False,
        "hotword_count": 0,
    }

    first_json, first_hash = stt_bench.build_execution_contract(
        plan=plan,
        git_commit="b" * 40,
        safe_target_settings=safe_settings,
    )
    second_json, second_hash = stt_bench.build_execution_contract(
        plan=plan,
        git_commit="b" * 40,
        safe_target_settings=dict(reversed(tuple(safe_settings.items()))),
    )
    payload = json.loads(first_json)

    assert (first_json, first_hash) == (second_json, second_hash)
    assert hashlib.sha256(first_json.encode("utf-8")).hexdigest() == first_hash
    assert first_json == json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert "/Users/alice" not in first_json
    assert payload["scorer_version"] == SCORER_VERSION
    assert payload["unicode_version"]
    assert payload["source_hashes"]
    assert payload["dependency_versions"]["pytest"]


@pytest.mark.parametrize(
    "safe_settings",
    [
        {"mode": "neutral-v1", "raw_prompt": "do not persist"},
        {"mode": "neutral-v1", "configuration_id": "secret-token"},
        {"mode": "unknown-v1"},
    ],
)
def test_build_execution_contract_rejects_unknown_or_unsafe_target_settings(
    safe_settings,
):
    with pytest.raises(ValueError, match="safe target settings"):
        stt_bench.build_execution_contract(
            plan=_planned_target(),
            git_commit="c" * 40,
            safe_target_settings=safe_settings,
        )


def test_resolve_adapter_factory_accepts_only_a_top_level_callable():
    resolved = stt_bench._resolve_adapter_factory(f"{__name__}:_preflight_fake_factory")

    assert resolved is _preflight_fake_factory
    with pytest.raises(ValueError, match="module:top_level_name"):
        stt_bench._resolve_adapter_factory(__name__)
    with pytest.raises(ValueError, match="module:top_level_name"):
        stt_bench._resolve_adapter_factory(f"{__name__}:_PreflightFakeAdapter.get_capabilities")
    with pytest.raises(ValueError, match="callable"):
        stt_bench._resolve_adapter_factory(f"{__name__}:_NOT_AN_ADAPTER_FACTORY")


def test_load_native_adapter_imports_registry_lazily_and_uses_strict_lookup(
    monkeypatch,
):
    calls = []
    expected = object()

    class FakeRegistry:
        def get_adapter_strict(self, provider):
            calls.append(("strict", provider))
            return expected

    def fake_import_module(name):
        calls.append(("import", name))
        return types.SimpleNamespace(SttProviderRegistry=FakeRegistry)

    monkeypatch.setattr(stt_bench.importlib, "import_module", fake_import_module)

    assert calls == []
    assert stt_bench._load_native_adapter("faster-whisper") is expected
    assert calls == [
        (
            "import",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
        ),
        ("strict", "faster-whisper"),
    ]


def test_preflight_targets_returns_deterministic_secret_safe_targets(
    preflight_fakes,
):
    _PREFLIGHT_PLANS.update(
        {
            ("fake", "model-a"): _planned_target(runtime_secret="/private/models/model-a"),
            ("other", "model-b"): _planned_target(
                provider="other",
                model_label="model-b",
                runtime_secret="/private/models/model-b",
            ),
        }
    )

    first = stt_bench.preflight_targets(
        ("fake=model-a", "other=model-b"),
        mode="neutral-v1",
        allow_network_targets=False,
        common_settings=_preflight_settings(),
        adapter_factory_path=f"{__name__}:_preflight_fake_factory",
    )
    second = stt_bench.preflight_targets(
        ("fake=model-a", "other=model-b"),
        mode="neutral-v1",
        allow_network_targets=False,
        common_settings=_preflight_settings(),
        adapter_factory_path=f"{__name__}:_preflight_fake_factory",
    )

    assert tuple(target.target_id for target in first) == tuple(target.target_id for target in second)
    assert tuple(target.provider for target in first) == ("fake", "other")
    assert tuple(target.model_label for target in first) == (
        "model-a",
        "model-b",
    )
    assert all(re.fullmatch(r"target-[0-9a-f]{16}", target.target_id) for target in first)
    serialized = json.dumps(
        [
            {
                "target_id": target.target_id,
                "provider": target.provider,
                "model_label": target.model_label,
                "execution_contract_json": target.execution_contract_json,
                "execution_contract_hash": target.execution_contract_hash,
            }
            for target in first
        ],
        sort_keys=True,
    )
    assert "/private/models" not in serialized


def test_preflight_targets_validates_the_whole_matrix_before_failing(
    preflight_fakes,
):
    _PREFLIGHT_PLANS.update(
        {
            ("download", "model-a"): _planned_target(
                provider="download",
                would_download=True,
            ),
            ("missing", "model-b"): _planned_target(
                provider="missing",
                model_label="model-b",
                local_model_available=False,
            ),
            ("broken", "model-c"): RuntimeError("Authorization: Bearer sk-private-token"),
        }
    )

    with pytest.raises(ValueError) as raised:
        stt_bench.preflight_targets(
            (
                "download=model-a",
                "missing=model-b",
                "broken=model-c",
                "unknown=model-d",
            ),
            mode="neutral-v1",
            allow_network_targets=False,
            common_settings=_preflight_settings(),
            adapter_factory_path=f"{__name__}:_preflight_fake_factory",
        )

    assert _PREFLIGHT_CALLS == [
        ("download", "model-a"),
        ("missing", "model-b"),
        ("broken", "model-c"),
    ]
    message = str(raised.value)
    assert all(f"target {index}" in message for index in range(1, 5))
    assert "sk-private-token" not in message


def test_preflight_targets_rejects_duplicate_normalized_targets_after_planning(
    preflight_fakes,
):
    plan = _planned_target()
    _PREFLIGHT_PLANS.update(
        {
            ("fake", "model-a"): plan,
            ("alias", "model-a"): plan,
        }
    )
    _PREFLIGHT_CANONICAL["alias"] = "fake"

    with pytest.raises(ValueError, match="duplicate normalized target"):
        stt_bench.preflight_targets(
            ("fake=model-a", "alias=model-a"),
            mode="neutral-v1",
            allow_network_targets=False,
            common_settings=_preflight_settings(),
            adapter_factory_path=f"{__name__}:_preflight_fake_factory",
        )

    assert _PREFLIGHT_CALLS == [
        ("fake", "model-a"),
        ("alias", "model-a"),
    ]


@pytest.mark.parametrize(
    "egress",
    [SttAudioEgress.LOOPBACK, SttAudioEgress.REMOTE],
)
def test_preflight_targets_requires_consent_for_every_network_route(
    preflight_fakes,
    egress,
):
    _PREFLIGHT_PLANS[("network", "model-a")] = _planned_target(
        provider="network",
        egress=egress,
        local_model_available=False,
    )

    with pytest.raises(ValueError, match="network consent"):
        stt_bench.preflight_targets(
            ("network=model-a",),
            mode="neutral-v1",
            allow_network_targets=False,
            common_settings=_preflight_settings(),
            adapter_factory_path=f"{__name__}:_preflight_fake_factory",
        )

    prepared = stt_bench.preflight_targets(
        ("network=model-a",),
        mode="neutral-v1",
        allow_network_targets=True,
        common_settings=_preflight_settings(),
        adapter_factory_path=f"{__name__}:_preflight_fake_factory",
    )
    assert prepared[0].provider == "network"


def test_preflight_targets_rejects_unavailable_and_mismatched_adapters(
    preflight_fakes,
):
    _PREFLIGHT_PLANS.update(
        {
            ("unavailable", "model-a"): _planned_target(provider="unavailable"),
            ("mismatch", "model-b"): _planned_target(
                provider="other",
                model_label="model-b",
            ),
        }
    )
    _PREFLIGHT_UNAVAILABLE.add("unavailable")

    with pytest.raises(ValueError) as raised:
        stt_bench.preflight_targets(
            ("unavailable=model-a", "mismatch=model-b"),
            mode="neutral-v1",
            allow_network_targets=False,
            common_settings=_preflight_settings(),
            adapter_factory_path=f"{__name__}:_preflight_fake_factory",
        )

    assert "target 1" in str(raised.value)
    assert "target 2" in str(raised.value)


def test_preflight_targets_rejects_mismatched_resolved_provider(
    preflight_fakes,
):
    _PREFLIGHT_PLANS[("fake", "model-a")] = _planned_target(
        resolved_provider="other",
    )

    with pytest.raises(ValueError, match="provider mismatch"):
        stt_bench.preflight_targets(
            ("fake=model-a",),
            mode="neutral-v1",
            allow_network_targets=False,
            common_settings=_preflight_settings(),
            adapter_factory_path=f"{__name__}:_preflight_fake_factory",
        )


def test_preflight_targets_redacts_prompt_and_hotwords_from_planner_errors(
    preflight_fakes,
):
    prompt = "transcribe Project Nebula exactly"
    hotword = "Asterion"
    _PREFLIGHT_PLANS[("fake", "model-a")] = RuntimeError(f"planner rejected {prompt}; hotword={hotword}")

    with pytest.raises(ValueError) as raised:
        stt_bench.preflight_targets(
            ("fake=model-a",),
            mode="production-v1",
            allow_network_targets=False,
            common_settings=_preflight_settings(
                prompt=prompt,
                hotwords=(hotword,),
                configuration_id="configuration-1",
            ),
            adapter_factory_path=f"{__name__}:_preflight_fake_factory",
        )

    assert prompt not in str(raised.value)
    assert hotword not in str(raised.value)


def test_preflight_targets_rejects_invalid_specs_and_neutral_fallbacks(
    preflight_fakes,
):
    _PREFLIGHT_PLANS[("fallback", "model-a")] = _planned_target(
        provider="fallback",
        route_count=2,
    )

    with pytest.raises(ValueError) as raised:
        stt_bench.preflight_targets(
            ("missing-equals", "=model", "provider=", "fallback=model-a"),
            mode="neutral-v1",
            allow_network_targets=False,
            common_settings=_preflight_settings(),
            adapter_factory_path=f"{__name__}:_preflight_fake_factory",
        )

    assert all(f"target {index}" in str(raised.value) for index in range(1, 5))
