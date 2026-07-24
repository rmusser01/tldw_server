"""Tests for the deterministic native STT benchmark scorer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import subprocess
import wave
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
    duplicate_source = (
        encoded_source[:-1]
        + ',"license":"different"}'
    )
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
    records_by_root[1][1]["source"] = dict(
        reversed(tuple(records_by_root[1][1]["source"].items()))
    )
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
        key=lambda sample_id: hashlib.sha256(
            f"7\0{sample_id}".encode()
        ).hexdigest(),
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

    assert stt_bench.main(
        [
            "validate",
            "--manifest",
            str(manifest_path),
            "--dataset-root",
            str(tmp_path),
        ]
    ) == 0
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
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("sample-1 field reference is invalid")
        ),
    )
    assert stt_bench.main(
        [
            "validate",
            "--manifest",
            str(manifest_path),
            "--dataset-root",
            str(tmp_path),
        ]
    ) != 0
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
    assert pooled.rate == sum(item.errors for item in samples) / sum(
        item.reference_units for item in samples
    )
