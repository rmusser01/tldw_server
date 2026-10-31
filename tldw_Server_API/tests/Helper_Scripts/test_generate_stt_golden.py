"""Tests for the STT golden helper script utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from Helper_Scripts.Audio import generate_stt_golden as script
from Helper_Scripts.benchmarks.stt_bench import load_and_validate_manifest


class DummyAdapter:
    """Stub adapter that returns a fixed transcript for tests."""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe_batch(self, audio_path: str, **kwargs) -> dict:
        self.calls += 1
        return {
            "text": "Hello world",
            "segments": [{"Text": "Hello world"}],
        }


def test_resolve_audio_path_relative(tmp_path: Path) -> None:
    """_resolve_audio_path should accept base-dir relative audio paths."""
    base_dir = tmp_path / "golden"
    audio_path = base_dir / "audio" / "clip.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"data")

    resolved, rel_audio = script._resolve_audio_path(audio_path.relative_to(base_dir), base_dir)

    assert resolved == audio_path.resolve()
    assert rel_audio == "audio/clip.wav"


def test_resolve_audio_path_outside_base(tmp_path: Path) -> None:
    """_resolve_audio_path should reject files outside the base dir."""
    base_dir = tmp_path / "golden"
    base_dir.mkdir()
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"data")

    with pytest.raises(ValueError):
        script._resolve_audio_path(audio_path, base_dir)


def test_generate_golden_payload_defaults_to_unverified_candidate(
    tmp_path: Path,
) -> None:
    """Adapter text must be labeled as a candidate rather than a reference."""
    adapter = DummyAdapter()
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"data")

    payload = script._generate_golden_payload(
        adapter,
        audio_path,
        "audio/clip.wav",
        provider="faster-whisper",
        model="demo-model",
        language="en",
        min_segments=2,
        reference=None,
        reference_provenance=None,
        sample_id=None,
    )

    assert payload["artifact_type"] == "stt-transcript-candidate"
    assert payload["reference_status"] == "unverified_candidate"
    assert payload["audio"] == "audio/clip.wav"
    assert payload["provider"] == "faster-whisper"
    assert payload["model"] == "demo-model"
    assert payload["candidate_text"] == "Hello world"
    assert payload["language"] == "en"
    assert payload["min_segments"] == 2
    assert "reference" not in payload
    assert "expected_text" not in payload
    assert adapter.calls == 1


def test_generate_golden_payload_builds_manifest_only_from_independent_reference(
    tmp_path: Path,
) -> None:
    """A provenanced external reference may become a regression manifest row."""
    adapter = DummyAdapter()
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"data")

    payload = script._generate_golden_payload(
        adapter,
        audio_path,
        "audio/clip.wav",
        provider="faster-whisper",
        model="demo-model",
        language="en",
        min_segments=None,
        reference="Independently reviewed words.",
        reference_provenance="human-reviewed",
        sample_id="golden-clip-1",
    )

    assert payload["id"] == "golden-clip-1"
    assert payload["audio"] == "audio/clip.wav"
    assert payload["reference"] == "Independently reviewed words."
    assert payload["normalization_profile"] == "en-v1"
    assert payload["profiles"] == ["regression"]
    assert payload["source"]["reference_provenance"] == "human-reviewed"
    assert payload["source"]["sha256"]
    assert "candidate_text" not in payload
    assert "reference_status" not in payload
    assert adapter.calls == 0


def test_verified_golden_payload_round_trips_through_manifest_validator(
    tmp_path: Path,
) -> None:
    """Verified output is directly usable as a one-record JSONL manifest."""
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"data")
    payload = script._generate_golden_payload(
        DummyAdapter(),
        audio_path,
        "clip.wav",
        provider="faster-whisper",
        model="demo-model",
        language="en",
        min_segments=None,
        reference="Independently reviewed words.",
        reference_provenance="canonical-dataset",
        sample_id="golden-clip-1",
    )
    manifest_path = tmp_path / "manifest.jsonl"

    script._write_golden_json(manifest_path, payload)
    samples, _ = load_and_validate_manifest(
        manifest_path,
        tmp_path,
        duration_probe=lambda _path: 1.0,
    )

    assert samples[0].sample_id == "golden-clip-1"
    assert dict(samples[0].source)["reference_provenance"] == "canonical-dataset"


@pytest.mark.parametrize(
    ("reference", "provenance", "sample_id"),
    [
        ("Reviewed words", None, "golden-clip-1"),
        (None, "human-reviewed", "golden-clip-1"),
        ("Reviewed words", "model-generated", "golden-clip-1"),
        ("Reviewed words", "canonical-dataset", None),
    ],
)
def test_generate_golden_payload_rejects_unscorable_reference_inputs(
    tmp_path: Path,
    reference: str | None,
    provenance: str | None,
    sample_id: str | None,
) -> None:
    """Incomplete, model-generated, or unidentified references fail closed."""
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"data")

    with pytest.raises(ValueError):
        script._generate_golden_payload(
            DummyAdapter(),
            audio_path,
            "audio/clip.wav",
            provider="faster-whisper",
            model="demo-model",
            language="en",
            min_segments=None,
            reference=reference,
            reference_provenance=provenance,
            sample_id=sample_id,
        )


def test_generate_golden_payload_rejects_invalid_manifest_language(
    tmp_path: Path,
) -> None:
    """Verified rows enforce the benchmark's language-tag contract."""
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"data")

    with pytest.raises(ValueError, match="bcp47"):
        script._generate_golden_payload(
            DummyAdapter(),
            audio_path,
            "audio/clip.wav",
            provider="",
            model="",
            language="not a language tag",
            min_segments=None,
            reference="Reviewed words",
            reference_provenance="human-reviewed",
            sample_id="golden-clip-1",
        )


def test_load_adapter_uses_registry_without_provider_allowlist(monkeypatch) -> None:
    """Strict registry lookup, not a local provider list, defines support."""
    expected = DummyAdapter()

    class Registry:
        def get_adapter_strict(self, provider):
            assert provider == "future-provider"
            return expected

    monkeypatch.setattr(script, "_provider_registry", Registry)

    assert script._load_adapter("future-provider") is expected
    assert not hasattr(script, "SUPPORTED_PROVIDERS")


def test_cli_reference_and_provenance_must_be_supplied_together() -> None:
    """CLI parsing exposes both independent-reference arguments."""
    common = [
        "--provider",
        "faster-whisper",
        "--audio",
        "clip.wav",
        "--model",
        "demo-model",
        "--output",
        "candidate.json",
    ]

    reference = script.parse_args([*common, "--reference", "Reviewed words"])
    provenance = script.parse_args(
        [
            *common,
            "--reference-provenance",
            "canonical-dataset",
        ]
    )

    assert reference.reference == "Reviewed words"
    assert reference.reference_provenance is None
    assert provenance.reference is None
    assert provenance.reference_provenance == "canonical-dataset"


def test_cli_verified_reference_is_target_neutral() -> None:
    """Verified manifest output does not require a provider or model."""
    parsed = script.parse_args(
        [
            "--audio",
            "clip.wav",
            "--output",
            "manifest.jsonl",
            "--language",
            "en",
            "--sample-id",
            "golden-clip-1",
            "--reference",
            "Reviewed words",
            "--reference-provenance",
            "human-reviewed",
        ]
    )

    assert parsed.provider is None
    assert parsed.model is None


def test_write_golden_json(tmp_path: Path) -> None:
    """_write_golden_json should persist the payload as JSON."""
    payload = {
        "audio": "audio/clip.wav",
        "model": "demo-model",
        "expected_text": "Hello world",
    }
    output_path = tmp_path / "whisper_clip1.golden.json"

    script._write_golden_json(output_path, payload)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == payload
