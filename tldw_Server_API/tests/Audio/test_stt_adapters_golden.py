"""Opt-in real-audio regression tests for native STT adapters.

The regression manifest supplies independently verified references. Adapter
output is scored with the native benchmark scorer and is never promoted to
ground truth. Real model execution remains explicitly opt-in.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from Helper_Scripts.benchmarks.stt_bench import (
    ManifestSample,
    PreparedTarget,
    _actual_matches_worker_plan,
    _validate_execution_mapping,
    load_and_validate_manifest,
    preflight_targets,
    resolve_audio_for_scheduling,
    score_transcript,
)

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
    SttProviderRegistry,
)

TOLERANCE_DEFAULT = 0.20


@dataclass(frozen=True)
class GoldenEnvironment:
    """Validated opt-in configuration for one golden regression run."""

    audio_dir: Path
    manifest_path: Path
    targets: tuple[str, ...]
    max_normalized_wer: float
    allow_network: bool


def _bool_env(name: str) -> bool:
    """Parse the repository's conventional truthy environment values."""
    raw = os.getenv(name, "")
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_golden_targets(raw: str) -> tuple[str, ...]:
    """Parse a bounded JSON array of unique ``provider=model`` strings."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("TLDW_STT_GOLDEN_TARGETS must be a JSON array") from exc
    if not isinstance(value, list) or not value or len(value) > 32:
        raise ValueError("TLDW_STT_GOLDEN_TARGETS must be a non-empty JSON array")
    targets: list[str] = []
    for item in value:
        if not isinstance(item, str) or item.count("=") < 1:
            raise ValueError("golden targets must use provider=model strings")
        provider, model = item.split("=", 1)
        target = item.strip()
        if (
            not provider.strip()
            or not model.strip()
            or len(target) > 4096
            or any(ord(character) < 32 for character in target)
        ):
            raise ValueError("golden target provider or model is invalid")
        targets.append(target)
    if len(targets) != len(set(targets)):
        raise ValueError("golden targets must be unique")
    return tuple(targets)


def _parse_max_normalized_wer(raw: str | None) -> float:
    """Parse the opt-in per-sample normalized WER threshold."""
    try:
        threshold = TOLERANCE_DEFAULT if raw is None else float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("TLDW_STT_GOLDEN_MAX_NORMALIZED_WER must be numeric") from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("TLDW_STT_GOLDEN_MAX_NORMALIZED_WER must be finite and non-negative")
    return threshold


def _require_golden_env() -> GoldenEnvironment:
    """Return validated opt-in configuration or skip an unconfigured run."""
    if not _bool_env("TLDW_STT_GOLDEN_ENABLE"):
        pytest.skip("TLDW_STT_GOLDEN_ENABLE not set; skipping STT golden tests")
    audio_value = os.getenv("TLDW_STT_GOLDEN_AUDIO_DIR")
    manifest_value = os.getenv("TLDW_STT_GOLDEN_MANIFEST")
    targets_value = os.getenv("TLDW_STT_GOLDEN_TARGETS")
    if not audio_value or not manifest_value or not targets_value:
        raise ValueError("golden audio directory, manifest, and JSON target array must all be configured")
    audio_dir = Path(audio_value).expanduser().resolve()
    if not audio_dir.is_dir():
        raise ValueError(f"TLDW_STT_GOLDEN_AUDIO_DIR={audio_dir} is not a directory")
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = audio_dir / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ValueError(f"TLDW_STT_GOLDEN_MANIFEST={manifest_path} is not a file")
    return GoldenEnvironment(
        audio_dir=audio_dir,
        manifest_path=manifest_path,
        targets=_parse_golden_targets(targets_value),
        max_normalized_wer=_parse_max_normalized_wer(os.getenv("TLDW_STT_GOLDEN_MAX_NORMALIZED_WER")),
        allow_network=_bool_env("TLDW_STT_GOLDEN_ALLOW_NETWORK"),
    )


def _load_regression_samples(
    environment: GoldenEnvironment,
) -> tuple[ManifestSample, ...]:
    """Load only the manifest's opt-in regression profile."""
    samples, _ = load_and_validate_manifest(
        environment.manifest_path,
        environment.audio_dir,
    )
    regression = tuple(sample for sample in samples if "regression" in sample.profiles)
    if not regression:
        raise ValueError("golden manifest has no regression-profile samples")
    if any(
        dict(sample.source).get("reference_provenance") not in {"canonical-dataset", "human-reviewed"}
        for sample in regression
    ):
        raise ValueError("golden regression references must be canonical-dataset or human-reviewed")
    return regression


def _require_network_consent(
    targets: Sequence[PreparedTarget | object],
    *,
    allow_network: bool,
) -> None:
    """Require the dedicated opt-in for loopback or remote plan routes."""
    for target in targets:
        plan = getattr(target, "plan", None)
        descriptor = getattr(plan, "descriptor", None)
        for route in getattr(descriptor, "routes", ()):
            egress = getattr(route, "audio_egress", None)
            egress_value = getattr(egress, "value", egress)
            if egress_value != "none" and not allow_network:
                raise ValueError("golden loopback/remote targets require TLDW_STT_GOLDEN_ALLOW_NETWORK=1")


def _prepare_targets(
    target_specs: Sequence[str],
    sample: ManifestSample,
    *,
    allow_network: bool,
) -> tuple[PreparedTarget, ...]:
    """Create and approve exact no-download plans for one sample language."""
    prepared = preflight_targets(
        target_specs,
        mode="neutral-v1",
        allow_network_targets=allow_network,
        common_settings={
            "task": "transcribe",
            "language": sample.language,
            "word_timestamps": False,
            "prompt": None,
            "hotwords": (),
            "diarization": False,
            "git_commit": "unknown",
        },
    )
    _require_network_consent(prepared, allow_network=allow_network)
    return prepared


def _assert_artifact_shape(
    artifact: object,
    *,
    plan: object | None = None,
) -> Mapping[str, Any]:
    """Validate normalized shape and bind execution to the approved plan."""
    assert isinstance(artifact, Mapping)
    assert isinstance(artifact.get("text"), str)
    segments = artifact.get("segments")
    assert isinstance(segments, list)
    assert segments
    assert all(isinstance(segment, Mapping) for segment in segments)
    actual_execution = artifact.get("actual_execution")
    try:
        _validate_execution_mapping(
            actual_execution,
            field="golden.actual_execution",
            actual=True,
        )
    except ValueError as exc:
        raise AssertionError("golden actual execution is not canonical") from exc
    assert isinstance(actual_execution, dict)
    if plan is not None:
        assert _actual_matches_worker_plan(actual_execution, plan)
    assert artifact.get("execution_mismatch", []) == []
    return artifact


def _transcribe_planned(
    registry: SttProviderRegistry,
    target: PreparedTarget,
    audio_path: Path,
) -> Mapping[str, Any]:
    """Execute exactly the target plan approved before opening the audio."""
    adapter = registry.get_adapter_strict(target.provider)
    artifact = adapter.transcribe_batch(
        str(audio_path),
        model=target.plan.descriptor.requested_model_label,
        language=target.plan.language,
        task=target.plan.task,
        word_timestamps=target.plan.word_timestamps,
        prompt=target.plan.prompt,
        hotwords=target.plan.hotwords,
        execution_plan=target.plan,
    )
    return _assert_artifact_shape(artifact, plan=target.plan)


@pytest.mark.stt_golden
def test_native_stt_targets_against_regression_manifest() -> None:
    """Score each configured native target against verified regression text."""
    environment = _require_golden_env()
    samples = _load_regression_samples(environment)
    registry = SttProviderRegistry()
    for sample in samples:
        audio_path = resolve_audio_for_scheduling(sample, environment.audio_dir)
        targets = _prepare_targets(
            environment.targets,
            sample,
            allow_network=environment.allow_network,
        )
        for target in targets:
            artifact = _transcribe_planned(registry, target, audio_path)
            score = score_transcript(
                sample.reference,
                str(artifact["text"]),
                normalization_profile=sample.normalization_profile,
            )
            assert score.normalized_wer.rate <= environment.max_normalized_wer, (
                f"{target.provider}={target.model_label} failed {sample.sample_id}: "
                f"normalized WER={score.normalized_wer.rate:.3f} > "
                f"{environment.max_normalized_wer:.3f}"
            )


def test_golden_flow_imports_shared_manifest_loader_and_scorer() -> None:
    """Golden regression checks use the benchmark's versioned contracts."""
    from Helper_Scripts.benchmarks import stt_bench

    assert load_and_validate_manifest is stt_bench.load_and_validate_manifest
    assert score_transcript is stt_bench.score_transcript


def test_golden_environment_skips_only_when_profile_is_disabled(monkeypatch) -> None:
    """A deliberately enabled but incomplete release run must fail."""
    monkeypatch.delenv("TLDW_STT_GOLDEN_ENABLE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _require_golden_env()

    monkeypatch.setenv("TLDW_STT_GOLDEN_ENABLE", "1")
    for name in (
        "TLDW_STT_GOLDEN_AUDIO_DIR",
        "TLDW_STT_GOLDEN_MANIFEST",
        "TLDW_STT_GOLDEN_TARGETS",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="must all be configured"):
        _require_golden_env()


def test_golden_regression_rejects_model_generated_reference_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    """Regression scoring accepts only canonical or human-reviewed text."""
    sample = ManifestSample(
        sample_id="candidate-reference",
        audio_relative="clip.wav",
        reference="model output",
        language="en",
        normalization_profile="en-v1",
        measured_duration_seconds=1.0,
        profiles=("regression",),
        suite="private-golden-v1",
        suite_visibility="private",
        annotation_profile="candidate-v1",
        diagnostic_only=False,
        source=(
            ("dataset", "local-golden"),
            ("license", "user-supplied"),
            ("reference_provenance", "model-generated"),
            ("version", "1"),
        ),
        tags=("golden",),
        sha256="a" * 64,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "load_and_validate_manifest",
        lambda *_args, **_kwargs: ((sample,), "a" * 64),
    )
    environment = GoldenEnvironment(
        audio_dir=tmp_path,
        manifest_path=tmp_path / "manifest.jsonl",
        targets=("faster-whisper=large-v3",),
        max_normalized_wer=0.2,
        allow_network=False,
    )

    with pytest.raises(ValueError, match="canonical-dataset or human-reviewed"):
        _load_regression_samples(environment)


def test_parse_golden_targets_accepts_only_json_array_target_specs() -> None:
    """JSON arrays preserve target boundaries without delimiter ambiguity."""
    assert _parse_golden_targets('["faster-whisper=large-v3","external=external:custom"]') == (
        "faster-whisper=large-v3",
        "external=external:custom",
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "faster-whisper=large-v3,parakeet=parakeet-mlx",
        '"faster-whisper=large-v3"',
        "[]",
        "[1]",
        '["missing-separator"]',
        '["=model"]',
        '["provider="]',
        '["faster-whisper=large-v3","faster-whisper=large-v3"]',
    ],
)
def test_parse_golden_targets_rejects_ambiguous_or_invalid_values(raw: str) -> None:
    """Malformed or duplicate target declarations fail before model setup."""
    with pytest.raises(ValueError):
        _parse_golden_targets(raw)


def test_golden_network_routes_require_separate_consent() -> None:
    """Loopback and remote plans require the dedicated golden opt-in."""

    class Egress:
        value = "loopback"

    class Route:
        audio_egress = Egress()

    class Descriptor:
        routes = (Route(),)

    class Plan:
        descriptor = Descriptor()

    class Target:
        plan = Plan()

    with pytest.raises(ValueError, match="ALLOW_NETWORK"):
        _require_network_consent((Target(),), allow_network=False)

    _require_network_consent((Target(),), allow_network=True)


def _golden_actual_execution() -> dict[str, object]:
    """Return one canonical local actual-execution envelope."""
    return {
        "route_id": "route-1",
        "provider": "faster-whisper",
        "model_label": "large-v3",
        "artifact_id": None,
        "backend": "faster-whisper",
        "audio_egress": "none",
        "endpoint_id": None,
        "source": "local",
        "device": "cpu",
        "compute_type": "int8",
        "dtype": None,
        "decoding_ids": [],
        "transport": None,
    }


@pytest.mark.parametrize(
    ("artifact", "valid"),
    [
        (None, False),
        ({"text": 1, "segments": []}, False),
        ({"text": "words", "segments": []}, False),
        ({"text": "words", "segments": ["not-an-object"]}, False),
        (
            {
                "text": "words",
                "segments": [{"text": "words"}],
                "actual_execution": _golden_actual_execution(),
            },
            True,
        ),
    ],
)
def test_golden_artifact_contract_keeps_segment_and_execution_assertions(
    artifact: object,
    valid: bool,
) -> None:
    """Only normalized planned artifacts satisfy the golden contract."""
    if valid:
        _assert_artifact_shape(artifact)
    else:
        with pytest.raises(AssertionError):
            _assert_artifact_shape(artifact)


def test_golden_artifact_rejects_execution_mismatch_and_undeclared_route() -> None:
    """WER cannot hide semantic mismatch or execution outside the plan."""
    actual = _golden_actual_execution()
    route_values = dict(actual)
    route_values["decoding_ids"] = ()
    route = SimpleNamespace(**route_values)
    plan = SimpleNamespace(
        descriptor=SimpleNamespace(routes=(route,)),
    )
    artifact = {
        "text": "words",
        "segments": [{"text": "words"}],
        "actual_execution": actual,
    }

    _assert_artifact_shape(artifact, plan=plan)

    mismatched = dict(artifact, execution_mismatch=["language"])
    with pytest.raises(AssertionError):
        _assert_artifact_shape(mismatched, plan=plan)

    undeclared_values = dict(route_values, backend="other-backend")
    undeclared = SimpleNamespace(
        descriptor=SimpleNamespace(routes=(SimpleNamespace(**undeclared_values),)),
    )
    with pytest.raises(AssertionError):
        _assert_artifact_shape(artifact, plan=undeclared)
