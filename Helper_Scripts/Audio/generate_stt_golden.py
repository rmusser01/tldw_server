#!/usr/bin/env python3
"""
Generate an STT candidate snapshot or a provenanced manifest record.

Example:
  export PYTHONPATH=.
  BASE=/srv/tldw_stt_golden

  python Helper_Scripts/Audio/generate_stt_golden.py \
    --provider faster-whisper \
    --audio "$BASE/audio/whisper/en/clip1.wav" \
    --model large-v3 \
    --language en \
    --base-dir "$BASE" \
    --output "$BASE/whisper_clip1.golden.json" \
    --min-segments 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
        SttProviderAdapter,
    )


_REFERENCE_PROVENANCES = frozenset({"canonical-dataset", "human-reviewed"})


def _resolve_base_dir(base_dir: Path | None) -> Path:
    """Resolve and validate the golden base directory."""
    base_value = base_dir
    if base_value is None:
        env_value = os.getenv("TLDW_STT_GOLDEN_AUDIO_DIR")
        if env_value:
            base_value = Path(env_value)
    if base_value is None:
        raise ValueError("Base directory is required. Pass --base-dir or set TLDW_STT_GOLDEN_AUDIO_DIR.")

    resolved = base_value.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Base directory does not exist or is not a directory: {resolved}")
    return resolved


def _resolve_audio_path(audio_path: Path | str, base_dir: Path) -> tuple[Path, str]:
    """Resolve an audio path and return (absolute_path, relative_posix_path)."""
    audio = Path(audio_path).expanduser()
    if not audio.is_absolute():
        audio = base_dir / audio
    audio = audio.resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    try:
        rel_audio = audio.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(f"Audio file must live under base dir: {base_dir}") from exc
    return audio, rel_audio.as_posix()


def _resolve_output_path(output_path: Path | str, base_dir: Path) -> Path:
    """Resolve output path, treating relative paths as base-dir relative."""
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = base_dir / output
    return output.resolve()


def _provider_registry():
    """Create the native registry lazily so utility imports stay lightweight."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
        SttProviderRegistry,
    )

    return SttProviderRegistry()


def _load_adapter(provider: str) -> SttProviderAdapter:
    """Resolve the requested provider strictly through the native registry."""
    return _provider_registry().get_adapter_strict(provider)


def _prepare_candidate(
    provider: str,
    model: str,
    language: str | None,
    *,
    allow_network: bool,
) -> tuple[SttProviderAdapter, Any]:
    """Approve one exact no-download plan before candidate transcription."""
    from Helper_Scripts.benchmarks.stt_bench import preflight_targets

    prepared = preflight_targets(
        (f"{provider}={model}",),
        mode="neutral-v1",
        allow_network_targets=allow_network,
        common_settings={
            "task": "transcribe",
            "language": language,
            "word_timestamps": False,
            "prompt": None,
            "hotwords": (),
            "diarization": False,
            "git_commit": "unknown",
        },
    )[0]
    return _load_adapter(prepared.provider), prepared


def _sha256_file(path: Path) -> str:
    """Return the audio file's lower-case SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_record(
    *,
    audio_path: Path,
    rel_audio_path: str,
    reference: str,
    reference_provenance: str,
    sample_id: str,
    language: str | None,
) -> dict[str, Any]:
    """Build one target-neutral regression record from independent text."""
    from Helper_Scripts.benchmarks.stt_bench import BCP47_BASIC_V1, STABLE_ID_V1

    if not reference.strip():
        raise ValueError("Reference must not be empty.")
    if reference_provenance not in _REFERENCE_PROVENANCES:
        raise ValueError("Reference provenance must be canonical-dataset or human-reviewed.")
    if STABLE_ID_V1.fullmatch(sample_id) is None:
        raise ValueError("A stable --sample-id is required for a manifest record.")
    if not language:
        raise ValueError("--language is required for a manifest record.")
    normalized_language = language.strip().lower()
    if BCP47_BASIC_V1.fullmatch(normalized_language) is None:
        raise ValueError("--language must match the benchmark's bcp47-basic-v1 profile.")
    normalization_profile = "en-v1" if normalized_language.split("-", 1)[0] == "en" else "strict-v1"
    return {
        "id": sample_id,
        "audio": rel_audio_path,
        "reference": reference,
        "language": normalized_language,
        "normalization_profile": normalization_profile,
        "profiles": ["regression"],
        "suite": "private-golden-v1",
        "suite_visibility": "private",
        "annotation_profile": f"{reference_provenance}-v1",
        "diagnostic_only": False,
        "source": {
            "dataset": "local-golden",
            "version": "1",
            "license": "user-supplied",
            "reference_provenance": reference_provenance,
            "sha256": _sha256_file(audio_path),
        },
        "tags": ["golden"],
    }


def _generate_golden_payload(
    adapter: SttProviderAdapter | None,
    audio_path: Path,
    rel_audio_path: str,
    provider: str,
    model: str,
    language: str | None,
    min_segments: int | None,
    reference: str | None,
    reference_provenance: str | None,
    sample_id: str | None,
    execution_plan: Any | None = None,
) -> dict[str, Any]:
    """Build a candidate snapshot or an independently provenanced manifest row."""
    if (reference is None) != (reference_provenance is None):
        raise ValueError("--reference and --reference-provenance must be supplied together.")
    if reference is not None:
        if sample_id is None:
            raise ValueError("--sample-id is required with --reference.")
        if min_segments is not None:
            raise ValueError("--min-segments is only valid for candidate snapshots.")
        return _manifest_record(
            audio_path=audio_path,
            rel_audio_path=rel_audio_path,
            reference=reference,
            reference_provenance=str(reference_provenance),
            sample_id=sample_id,
            language=language,
        )
    if sample_id is not None:
        raise ValueError("--sample-id is only valid with an independent reference.")
    if adapter is None:
        raise ValueError("An adapter is required to generate a candidate snapshot.")
    if not provider.strip() or not model.strip():
        raise ValueError("Provider and model must not be empty.")
    if min_segments is not None and min_segments < 1:
        raise ValueError("--min-segments must be positive.")
    if execution_plan is None:
        artifact = adapter.transcribe_batch(
            str(audio_path),
            model=model,
            language=language,
            task="transcribe",
            word_timestamps=False,
        )
    else:
        artifact = adapter.transcribe_batch(
            str(audio_path),
            model=execution_plan.descriptor.requested_model_label,
            language=execution_plan.language,
            task=execution_plan.task,
            word_timestamps=execution_plan.word_timestamps,
            prompt=execution_plan.prompt,
            hotwords=execution_plan.hotwords,
            execution_plan=execution_plan,
        )
    if not isinstance(artifact, dict):
        raise ValueError("Adapter returned a non-object artifact.")
    candidate_text = artifact.get("text")
    segments = artifact.get("segments")
    if not isinstance(candidate_text, str) or not isinstance(segments, list):
        raise ValueError("Adapter returned an invalid transcription artifact.")
    if not candidate_text:
        print(
            f"WARNING: adapter returned empty transcript for {audio_path}",
            file=sys.stderr,
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "stt-transcript-candidate",
        "reference_status": "unverified_candidate",
        "audio": rel_audio_path,
        "provider": provider,
        "model": model,
        "candidate_text": candidate_text,
        "segment_count": len(segments),
    }
    if language:
        payload["language"] = language
    if min_segments is not None:
        payload["min_segments"] = min_segments
    actual_execution = artifact.get("actual_execution")
    if isinstance(actual_execution, dict):
        payload["actual_execution"] = actual_execution
    return payload


def _write_golden_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a candidate as JSON or a manifest record as one JSONL line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "reference" in payload and "profiles" in payload:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        serialized = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    output_path.write_text(
        serialized + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate an STT candidate snapshot or provenanced manifest record.",
    )
    parser.add_argument(
        "--provider",
        help="Native registry provider; required for candidate snapshots.",
    )
    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        help="Path to the audio file (absolute or relative to --base-dir).",
    )
    parser.add_argument(
        "--model",
        help="Adapter model identifier required for candidate snapshots.",
    )
    parser.add_argument(
        "--language",
        help="Optional language hint passed to the adapter (e.g. en).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory for golden audio and JSONs (default: $TLDW_STT_GOLDEN_AUDIO_DIR).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Golden JSON output path (absolute or relative to --base-dir).",
    )
    parser.add_argument(
        "--min-segments",
        type=int,
        help="Optional candidate-snapshot segment expectation.",
    )
    parser.add_argument(
        "--reference",
        help="Independently supplied reference text; never populated from adapter output.",
    )
    parser.add_argument(
        "--reference-provenance",
        choices=sorted(_REFERENCE_PROVENANCES),
        help="Required with --reference: canonical-dataset or human-reviewed.",
    )
    parser.add_argument(
        "--sample-id",
        help="Stable benchmark sample ID required with --reference.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicitly allow a planned loopback or remote candidate target.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)

    try:
        base_dir = _resolve_base_dir(args.base_dir)
        audio_path, rel_audio = _resolve_audio_path(args.audio, base_dir)
        output_path = _resolve_output_path(args.output, base_dir)
        adapter = None
        execution_plan = None
        provider = args.provider or ""
        model = args.model or ""
        if args.reference is None:
            if not provider or not model:
                raise ValueError("--provider and --model are required for candidate snapshots.")
            adapter, prepared = _prepare_candidate(
                provider,
                model,
                args.language,
                allow_network=args.allow_network,
            )
            provider = prepared.provider
            model = prepared.model_label
            execution_plan = prepared.plan
        payload = _generate_golden_payload(
            adapter,
            audio_path,
            rel_audio,
            provider,
            model,
            args.language,
            args.min_segments,
            args.reference,
            args.reference_provenance,
            args.sample_id,
            execution_plan,
        )
        _write_golden_json(output_path, payload)
    except Exception as exc:  # noqa: BLE001 - CLI provider boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote golden file to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
