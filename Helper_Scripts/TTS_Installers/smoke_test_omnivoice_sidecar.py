#!/usr/bin/env python3
"""Run a real managed-sidecar OmniVoice TTS smoke test.

The helper intentionally goes through ``OmniVoiceSidecarSupervisor`` and
``OmniVoiceAdapter`` so operators verify the same managed sidecar path used by
the API server. It does not import OmniVoice directly in the main interpreter.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import sys
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence


def _bootstrap_repo_root() -> Path:
    probe = Path(__file__).resolve()
    for candidate in (probe,) + tuple(probe.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "tldw_Server_API").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise RuntimeError(f"Unable to resolve repository root from {probe}")


_BOOTSTRAPPED_REPO_ROOT = _bootstrap_repo_root()

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest  # noqa: E402
from tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter import OmniVoiceAdapter  # noqa: E402
from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_supervisor import (  # noqa: E402
    OmniVoiceSidecarSupervisor,
)


DEFAULT_TEXT = "Hello from the OmniVoice managed sidecar smoke test."
DEFAULT_RUNTIME_BASE = Path("models") / "omnivoice_sidecar"
DEFAULT_OUTPUT_NAME = "omnivoice_sidecar_smoke.wav"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_PORT = 8039
DEFAULT_PORT_PROBE_MAX = 20
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS = 30.0
DEFAULT_NUM_STEP = 8
DEFAULT_SPEED = 1.0


@dataclass(frozen=True)
class OmniVoiceSmokeConfig:
    """Resolved inputs for a managed OmniVoice sidecar smoke test."""

    repo_root: Path
    model_path: Path
    sidecar_python: Path
    runtime_path: Path
    scratch_dir: Path
    output_path: Path
    text: str
    port: int
    num_step: Optional[int]
    speed: float
    timeout: float


@dataclass(frozen=True)
class WavAudioSummary:
    """Small quality summary for generated WAV audio."""

    byte_count: int
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
    duration_seconds: float
    rms: float
    peak: int


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    """Resolve the repository root from a path, defaulting to this script."""

    if start is None:
        return _BOOTSTRAPPED_REPO_ROOT

    probe = start.expanduser().resolve()
    candidates = (probe,) + tuple(probe.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "tldw_Server_API").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to resolve repository root from {probe}")


def _resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _resolve_path_preserving_symlink(path_value: str | Path, repo_root: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _default_runtime_path(repo_root: Path) -> Path:
    return (repo_root / DEFAULT_RUNTIME_BASE / "runtime").resolve(strict=False)


def _default_scratch_dir(runtime_path: Path) -> Path:
    return (runtime_path / "scratch").resolve(strict=False)


def _default_output_path(runtime_path: Path) -> Path:
    return (runtime_path / DEFAULT_OUTPUT_NAME).resolve(strict=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the sidecar smoke test."""

    parser = argparse.ArgumentParser(description="Smoke test real OmniVoice audio through the managed sidecar")
    parser.add_argument("--model-path", required=True, help="Local OmniVoice model directory")
    parser.add_argument("--sidecar-python", required=True, help="Python interpreter from the OmniVoice sidecar venv")
    parser.add_argument("--repo-root", help="Repository root; defaults to the current script checkout")
    parser.add_argument("--runtime-path", help="Managed sidecar runtime directory")
    parser.add_argument("--scratch-dir", help="Managed scratch/reference directory")
    parser.add_argument("--output", help="Output WAV file path")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Starting loopback port for the sidecar")
    parser.add_argument("--num-step", type=int, default=DEFAULT_NUM_STEP, help="OmniVoice generation num_step")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="OmniVoice generation speed")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Synthesis timeout in seconds")
    return parser.parse_args(argv)


def build_smoke_config(args: argparse.Namespace) -> OmniVoiceSmokeConfig:
    """Resolve CLI arguments into a smoke-test config."""

    repo_root = resolve_repo_root(Path(args.repo_root)) if args.repo_root else resolve_repo_root()
    runtime_path = (
        _resolve_path(args.runtime_path, repo_root)
        if args.runtime_path
        else _default_runtime_path(repo_root)
    )
    scratch_dir = (
        _resolve_path(args.scratch_dir, repo_root)
        if args.scratch_dir
        else _default_scratch_dir(runtime_path)
    )
    output_path = (
        _resolve_path(args.output, repo_root)
        if args.output
        else _default_output_path(runtime_path)
    )
    return OmniVoiceSmokeConfig(
        repo_root=repo_root,
        model_path=_resolve_path(args.model_path, repo_root),
        sidecar_python=_resolve_path_preserving_symlink(args.sidecar_python, repo_root),
        runtime_path=runtime_path,
        scratch_dir=scratch_dir,
        output_path=output_path,
        text=str(args.text),
        port=int(args.port),
        num_step=args.num_step,
        speed=float(args.speed),
        timeout=float(args.timeout),
    )


def validate_smoke_config(config: OmniVoiceSmokeConfig) -> None:
    """Validate operator-provided paths before starting the sidecar."""

    if not config.model_path.is_dir():
        raise ValueError(f"OmniVoice model path is not a directory: {config.model_path}")
    if not config.sidecar_python.is_file():
        raise ValueError(f"OmniVoice sidecar Python interpreter does not exist: {config.sidecar_python}")
    if config.port <= 0 or config.port > 65535:
        raise ValueError(f"OmniVoice sidecar port must be between 1 and 65535: {config.port}")
    if config.timeout <= 0:
        raise ValueError("OmniVoice sidecar smoke timeout must be positive")
    if not config.text.strip():
        raise ValueError("OmniVoice sidecar smoke text must not be empty")


def build_sidecar_provider_config(
    *,
    model_path: Path,
    sidecar_python: Path,
    runtime_path: Path,
    scratch_dir: Path,
    port: int,
    timeout: float,
) -> dict[str, object]:
    """Build the provider config consumed by the managed sidecar path."""

    return {
        "enabled": True,
        "runtime": "sidecar",
        "model": "omnivoice",
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "timeout": float(timeout),
        "max_concurrent_generations": 1,
        "extra_params": {
            "model_path": str(model_path),
            "python_path": str(sidecar_python),
            "runtime_path": str(runtime_path),
            "scratch_dir": str(scratch_dir),
            "host": "127.0.0.1",
            "port": int(port),
            "autoselect_port": True,
            "port_probe_max": DEFAULT_PORT_PROBE_MAX,
            "healthcheck_timeout_seconds": DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS,
            "healthcheck_interval_seconds": 0.25,
            "startup_backoff_seconds": 0.0,
            "idle_shutdown_seconds": 60.0,
            "warmup_on_startup": False,
            "resident_mode": False,
        },
    }


def build_tts_request(
    *,
    text: str,
    num_step: Optional[int],
    speed: float,
) -> TTSRequest:
    """Build the adapter request for the smoke synthesis."""

    extra_params: dict[str, object] = {"language_id": "en"}
    if num_step is not None:
        extra_params["num_step"] = int(num_step)
    return TTSRequest(
        text=text,
        voice="auto",
        language="en",
        format=AudioFormat.WAV,
        speed=float(speed),
        stream=False,
        provider="omnivoice",
        model="omnivoice",
        extra_params=extra_params,
    )


def _iter_pcm_samples(frames: bytes, sample_width: int):
    if sample_width == 1:
        for value in frames:
            yield value - 128
        return

    if sample_width not in {2, 3, 4}:
        raise ValueError(f"Output WAV has unsupported sample width: {sample_width} bytes")

    for index in range(0, len(frames), sample_width):
        sample = frames[index : index + sample_width]
        if len(sample) == sample_width:
            yield int.from_bytes(sample, "little", signed=True)


def validate_wav_audio(
    audio_bytes: bytes,
    *,
    expected_sample_rate: int = DEFAULT_SAMPLE_RATE,
    expected_channels: int = 1,
    min_rms: float = 1.0,
) -> WavAudioSummary:
    """Validate parseable, mono, non-silent WAV audio."""

    if not audio_bytes:
        raise ValueError("OmniVoice sidecar returned empty audio")

    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError("OmniVoice sidecar output is not a parseable WAV file") from exc

    if sample_rate != expected_sample_rate:
        raise ValueError(f"Expected {expected_sample_rate} Hz WAV audio, got {sample_rate} Hz")
    if channels != expected_channels:
        raise ValueError(f"Expected mono WAV audio, got {channels} channels")
    if frame_count <= 0 or not frames:
        raise ValueError("OmniVoice sidecar returned a WAV file with no frames")

    square_sum = 0
    sample_count = 0
    peak = 0
    for sample in _iter_pcm_samples(frames, sample_width):
        sample_abs = abs(sample)
        peak = max(peak, sample_abs)
        square_sum += sample * sample
        sample_count += 1

    if sample_count <= 0:
        raise ValueError("OmniVoice sidecar returned a WAV file with no samples")

    rms = math.sqrt(square_sum / sample_count)
    if peak <= 0 or rms < min_rms:
        raise ValueError("OmniVoice sidecar output WAV appears silent")

    return WavAudioSummary(
        byte_count=len(audio_bytes),
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
        rms=rms,
        peak=peak,
    )


async def run_smoke(config: OmniVoiceSmokeConfig) -> WavAudioSummary:
    """Run synthesis through the managed sidecar and write the verified WAV."""

    provider_config = build_sidecar_provider_config(
        model_path=config.model_path,
        sidecar_python=config.sidecar_python,
        runtime_path=config.runtime_path,
        scratch_dir=config.scratch_dir,
        port=config.port,
        timeout=config.timeout,
    )
    supervisor = OmniVoiceSidecarSupervisor(provider_config, repo_root=config.repo_root)
    adapter = OmniVoiceAdapter(provider_config)
    adapter.set_supervisor(supervisor)

    try:
        initialized = await adapter.initialize()
        if not initialized:
            raise RuntimeError("OmniVoice adapter did not initialize with the sidecar supervisor")

        request = build_tts_request(text=config.text, num_step=config.num_step, speed=config.speed)
        response = await adapter.generate(request)
        audio_bytes = response.audio_data or response.audio_content or b""
        summary = validate_wav_audio(audio_bytes)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_bytes(audio_bytes)
        return summary
    finally:
        await supervisor.shutdown()


def format_summary(config: OmniVoiceSmokeConfig, summary: WavAudioSummary) -> str:
    """Format the operator-facing success summary."""

    return "\n".join(
        [
            "OmniVoice managed sidecar smoke succeeded.",
            f"Output: {config.output_path}",
            f"Bytes: {summary.byte_count}",
            f"Sample rate: {summary.sample_rate} Hz",
            f"Channels: {summary.channels}",
            f"Frames: {summary.frame_count}",
            f"Duration: {summary.duration_seconds:.2f}s",
            f"RMS: {summary.rms:.2f}",
            f"Peak: {summary.peak}",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""

    try:
        config = build_smoke_config(parse_args(argv))
        validate_smoke_config(config)
        summary = asyncio.run(run_smoke(config))
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"OmniVoice managed sidecar smoke failed: {exc}") from exc

    print(format_summary(config, summary))
    return 0


__all__ = [
    "OmniVoiceSmokeConfig",
    "WavAudioSummary",
    "build_sidecar_provider_config",
    "build_smoke_config",
    "build_tts_request",
    "format_summary",
    "main",
    "parse_args",
    "resolve_repo_root",
    "run_smoke",
    "validate_smoke_config",
    "validate_wav_audio",
]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
