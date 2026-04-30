from __future__ import annotations

import asyncio
import json
import hashlib
import contextlib
import secrets
import threading
import time
import tempfile
import uuid
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from tldw_Server_API.app.core.TTS.audio_converter import AudioConverter
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSModelNotFoundError


PROVIDER_KEY = "pocket_tts_cpp"
PROVIDER_MANAGED_VOICE_TOKEN_KEY = "_pocket_tts_cpp_voice_trust_token"  # nosec B105
PROVIDER_MANAGED_VOICE_LEASE_DIRNAME = ".pocket_tts_cpp_leases"
PROVIDER_MANAGED_VOICE_LEASE_SUFFIX = ".json"
VALID_PRECISIONS = {"int8", "fp32"}
STREAM_PROBE_TEXT = (
    "PocketTTS.cpp streaming probe. This text is intentionally longer than a "
    "single short sentence so the CLI has a meaningful opportunity to emit "
    "incremental stdout before completion. If the runtime buffers everything "
    "until the end, the probe should fail closed."
)
_PROVIDER_MANAGED_VOICE_TOKEN_TTL_SECONDS = 30 * 60
_PROVIDER_MANAGED_VOICE_TOKEN_LOCK = threading.RLock()
_PROVIDER_MANAGED_VOICE_TOKENS: dict[str, tuple[Path, float, Path]] = {}


def get_required_model_filenames(precision: str) -> list[str]:
    suffix = "_int8" if precision == "int8" else ""
    return [
        f"flow_lm_main{suffix}.onnx",
        f"flow_lm_flow{suffix}.onnx",
        f"mimi_decoder{suffix}.onnx",
        "mimi_encoder.onnx",
        "text_conditioner.onnx",
    ]


def validate_runtime_assets(
    *,
    binary_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    precision: str,
) -> None:
    """Validate the PocketTTS.cpp runtime files needed for CLI execution."""
    if not binary_path.exists() or not binary_path.is_file():
        raise TTSModelNotFoundError(
            f"PocketTTS.cpp binary not found at {binary_path}",
            provider=PROVIDER_KEY,
            details={"binary_path": str(binary_path)},
        )
    if not os.access(binary_path, os.X_OK):
        raise TTSModelNotFoundError(
            f"PocketTTS.cpp binary is not executable at {binary_path}",
            provider=PROVIDER_KEY,
            details={"binary_path": str(binary_path)},
        )

    if not tokenizer_path.exists() or not tokenizer_path.is_file():
        raise TTSModelNotFoundError(
            f"PocketTTS.cpp tokenizer not found at {tokenizer_path}",
            provider=PROVIDER_KEY,
            details={"tokenizer_path": str(tokenizer_path)},
        )

    if not model_path.exists() or not model_path.is_dir():
        raise TTSModelNotFoundError(
            f"PocketTTS.cpp models directory not found at {model_path}",
            provider=PROVIDER_KEY,
            details={"model_path": str(model_path)},
        )

    missing = [name for name in get_required_model_filenames(precision) if not (model_path / name).exists()]
    if missing:
        raise TTSModelNotFoundError(
            "PocketTTS.cpp exported ONNX assets missing",
            provider=PROVIDER_KEY,
            details={"model_path": str(model_path), "missing": missing},
        )


def build_cli_command(
    *,
    binary_path: Path,
    text: str,
    voice_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    output_path: Optional[Path],
    precision: str,
    prefer_stdout: bool,
    enable_voice_cache: bool,
    voices_dir: Optional[Path] = None,
    temperature: Optional[float] = None,
    lsd_steps: Optional[int] = None,
    eos_threshold: Optional[float] = None,
    eos_extra: Optional[float] = None,
    noise_clamp: Optional[float] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
    profile: bool = False,
) -> list[str]:
    """Build a PocketTTS.cpp CLI invocation for a single synthesis request."""
    command = [str(binary_path)]

    if prefer_stdout:
        command.append("--stdout")
    if precision:
        command.extend(["--precision", precision])
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if lsd_steps is not None:
        command.extend(["--lsd-steps", str(lsd_steps)])
    if eos_threshold is not None:
        command.extend(["--eos-threshold", str(eos_threshold)])
    if eos_extra is not None:
        command.extend(["--eos-extra", str(eos_extra)])
    if noise_clamp is not None:
        command.extend(["--noise-clamp", str(noise_clamp)])
    if threads is not None:
        command.extend(["--threads", str(threads)])
    if model_path:
        command.extend(["--models-dir", str(model_path)])
    if tokenizer_path:
        command.extend(["--tokenizer", str(tokenizer_path)])
    if voices_dir is not None:
        command.extend(["--voices-dir", str(voices_dir)])
    if not enable_voice_cache:
        command.append("--no-cache")
    if verbose:
        command.append("--verbose")
    if profile:
        command.append("--profile")

    command.extend([text, str(voice_path)])
    if output_path is not None:
        command.append(str(output_path))
    return command


def _normalize_provider_managed_voice_path(voice_path: Path) -> Path:
    resolved_voice_path = Path(voice_path).expanduser()
    if not resolved_voice_path.exists() or not resolved_voice_path.is_file():
        raise ValueError(f"PocketTTS.cpp provider-managed voice path does not exist: {resolved_voice_path}")
    resolved_voice_path = resolved_voice_path.resolve()

    provider_dir = resolved_voice_path.parent
    if provider_dir.name != PROVIDER_KEY or provider_dir.parent.name != "providers":
        raise ValueError(
            "PocketTTS.cpp provider-managed voice path must be materialized under voices/providers/pocket_tts_cpp/"
        )

    return resolved_voice_path


def _provider_managed_voice_lease_dir(runtime_dir: Path) -> Path:
    return runtime_dir / PROVIDER_MANAGED_VOICE_LEASE_DIRNAME


def _provider_managed_voice_lease_path(runtime_dir: Path, trust_token: str) -> Path:
    return _provider_managed_voice_lease_dir(runtime_dir) / f"{trust_token}{PROVIDER_MANAGED_VOICE_LEASE_SUFFIX}"


def _write_provider_managed_voice_lease(
    lease_path: Path,
    *,
    trust_token: str,
    voice_path: Path,
    created_at: float,
) -> None:
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token": trust_token,
        "voice_path": str(voice_path),
        "created_at": created_at,
        "pid": os.getpid(),
    }
    temp_path = lease_path.with_name(f".{lease_path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    temp_path.replace(lease_path)


def _load_provider_managed_voice_lease_record(lease_path: Path) -> tuple[str, Path, float]:
    payload = json.loads(lease_path.read_text(encoding="utf-8"))
    token = str(payload.get("token") or "").strip()
    voice_path = _normalize_provider_managed_voice_path(Path(str(payload.get("voice_path") or "")))
    created_at = float(payload.get("created_at") or 0.0)
    if not token:
        raise ValueError("PocketTTS.cpp trust token lease is missing a token")
    if created_at <= 0:
        raise ValueError("PocketTTS.cpp trust token lease is missing a creation timestamp")
    return token, voice_path, created_at


def _purge_expired_provider_managed_voice_tokens(now: Optional[float] = None) -> None:
    cutoff = time.time() if now is None else now
    expired_tokens = [
        token
        for token, (_, created_at, _) in _PROVIDER_MANAGED_VOICE_TOKENS.items()
        if (cutoff - created_at) > _PROVIDER_MANAGED_VOICE_TOKEN_TTL_SECONDS
    ]
    for token in expired_tokens:
        _, _, lease_path = _PROVIDER_MANAGED_VOICE_TOKENS.pop(token, (None, 0.0, None))
        if lease_path is not None:
            with contextlib.suppress(OSError):
                lease_path.unlink()


def register_provider_managed_voice_path(voice_path: Path) -> str:
    """Register a provider-managed voice path and return an unguessable trust token."""
    resolved_voice_path = _normalize_provider_managed_voice_path(voice_path)
    token = secrets.token_urlsafe(32)
    created_at = time.time()
    lease_path = _provider_managed_voice_lease_path(resolved_voice_path.parent, token)
    _write_provider_managed_voice_lease(
        lease_path,
        trust_token=token,
        voice_path=resolved_voice_path,
        created_at=created_at,
    )
    with _PROVIDER_MANAGED_VOICE_TOKEN_LOCK:
        _purge_expired_provider_managed_voice_tokens()
        _PROVIDER_MANAGED_VOICE_TOKENS[token] = (resolved_voice_path, created_at, lease_path)
    return token


def resolve_provider_managed_voice_path(trust_token: str, voice_path: Path) -> Path:
    """Return the registered path for a service-issued trust token after verifying the path."""
    if not trust_token:
        raise ValueError("PocketTTS.cpp trust token is required")

    requested_voice_path = _normalize_provider_managed_voice_path(voice_path)
    lease_path = _provider_managed_voice_lease_path(requested_voice_path.parent, str(trust_token))
    if not lease_path.exists():
        with _PROVIDER_MANAGED_VOICE_TOKEN_LOCK:
            _PROVIDER_MANAGED_VOICE_TOKENS.pop(str(trust_token), None)
        raise ValueError("PocketTTS.cpp trust token is not registered")

    token, registered_path, created_at = _load_provider_managed_voice_lease_record(lease_path)
    if token != str(trust_token):
        with contextlib.suppress(OSError):
            lease_path.unlink()
        raise ValueError("PocketTTS.cpp trust token is not registered")

    if (time.time() - created_at) > _PROVIDER_MANAGED_VOICE_TOKEN_TTL_SECONDS:
        with contextlib.suppress(OSError):
            lease_path.unlink()
        with _PROVIDER_MANAGED_VOICE_TOKEN_LOCK:
            _PROVIDER_MANAGED_VOICE_TOKENS.pop(str(trust_token), None)
        raise ValueError("PocketTTS.cpp trust token is expired")

    if registered_path != requested_voice_path:
        raise ValueError("PocketTTS.cpp voice path does not match the registered trust token")

    with _PROVIDER_MANAGED_VOICE_TOKEN_LOCK:
        _PROVIDER_MANAGED_VOICE_TOKENS[str(trust_token)] = (registered_path, created_at, lease_path)
    return registered_path


def revoke_provider_managed_voice_token(trust_token: Optional[str], voice_path: Optional[Path] = None) -> None:
    """Drop a provider-managed trust token from the in-process registry."""
    if not trust_token:
        return
    lease_path: Optional[Path] = None
    with _PROVIDER_MANAGED_VOICE_TOKEN_LOCK:
        registered = _PROVIDER_MANAGED_VOICE_TOKENS.pop(str(trust_token), None)
        if registered is not None:
            lease_path = registered[2]
    if lease_path is None and voice_path is not None:
        try:
            requested_voice_path = _normalize_provider_managed_voice_path(voice_path)
            lease_path = _provider_managed_voice_lease_path(requested_voice_path.parent, str(trust_token))
        except ValueError:
            lease_path = None
    if lease_path is not None:
        with contextlib.suppress(OSError):
            lease_path.unlink()


def get_active_provider_managed_voice_paths(runtime_dir: Optional[Path] = None) -> set[Path]:
    """Return the currently registered provider-managed voice paths."""
    with _PROVIDER_MANAGED_VOICE_TOKEN_LOCK:
        _purge_expired_provider_managed_voice_tokens()
        if runtime_dir is None:
            return {registered_path for registered_path, _, _ in _PROVIDER_MANAGED_VOICE_TOKENS.values()}

    active_paths: set[Path] = set()

    lease_dir = _provider_managed_voice_lease_dir(runtime_dir)
    if not lease_dir.exists():
        return active_paths

    now = time.time()
    for lease_path in lease_dir.glob(f"*{PROVIDER_MANAGED_VOICE_LEASE_SUFFIX}"):
        if not lease_path.is_file():
            continue
        try:
            token, registered_path, created_at = _load_provider_managed_voice_lease_record(lease_path)
        except (OSError, ValueError, json.JSONDecodeError):
            with contextlib.suppress(OSError):
                lease_path.unlink()
            continue
        if lease_path.stem != token:
            with contextlib.suppress(OSError):
                lease_path.unlink()
            continue
        if (now - created_at) > _PROVIDER_MANAGED_VOICE_TOKEN_TTL_SECONDS:
            with contextlib.suppress(OSError):
                lease_path.unlink()
            continue
        active_paths.add(registered_path)
    return active_paths


async def probe_cli_streaming_incremental(
    *,
    binary_path: Path,
    voice_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    precision: str,
    timeout: float,
    enable_voice_cache: bool,
    voices_dir: Optional[Path] = None,
    temperature: Optional[float] = None,
    lsd_steps: Optional[int] = None,
    eos_threshold: Optional[float] = None,
    eos_extra: Optional[float] = None,
    noise_clamp: Optional[float] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
    profile: bool = False,
) -> bool:
    """Return True when the CLI emits stdout bytes before process completion."""
    command = build_cli_command(
        binary_path=binary_path,
        text=STREAM_PROBE_TEXT,
        voice_path=voice_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        output_path=None,
        precision=precision,
        prefer_stdout=True,
        enable_voice_cache=enable_voice_cache,
        voices_dir=voices_dir,
        temperature=temperature,
        lsd_steps=lsd_steps,
        eos_threshold=eos_threshold,
        eos_extra=eos_extra,
        noise_clamp=noise_clamp,
        threads=threads,
        verbose=verbose,
        profile=profile,
    )

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    start = time.monotonic()
    try:
        if process.stdout is None:
            return False

        try:
            first_chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=timeout)
        except asyncio.TimeoutError:
            logger.info("PocketTTS.cpp CLI streaming probe timed out waiting for first chunk")
            return False

        if not first_chunk:
            logger.info("PocketTTS.cpp CLI streaming probe produced no stdout bytes")
            return False

        first_byte_elapsed = time.monotonic() - start

        drain_window = min(max(timeout * 0.05, 0.01), 0.05)
        drain_deadline = time.monotonic() + drain_window
        while True:
            remaining_drain = drain_deadline - time.monotonic()
            if remaining_drain <= 0:
                break
            try:
                drain_chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=remaining_drain)
            except asyncio.TimeoutError:
                break
            if not drain_chunk:
                logger.info("PocketTTS.cpp CLI streaming probe reached EOF before later stdout progress")
                return False

        observation_window = min(max(timeout * 0.25, 0.05), 0.5)
        try:
            later_chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=observation_window)
        except asyncio.TimeoutError:
            logger.info("PocketTTS.cpp CLI streaming probe produced no later stdout bytes within observation window")
            return False

        if not later_chunk:
            logger.info("PocketTTS.cpp CLI streaming probe reached EOF without later stdout progress")
            return False

        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.info("PocketTTS.cpp CLI streaming probe timed out waiting for process exit")
            return False

        total_elapsed = time.monotonic() - start
        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="ignore").strip()
            logger.info(
                "PocketTTS.cpp CLI streaming probe exited with returncode={} stderr={}",
                process.returncode,
                stderr_text,
            )
            return False

        incremental = True
        logger.info(
            "PocketTTS.cpp CLI streaming probe result={} ttfb_ms={} total_ms={}",
            incremental,
            round(first_byte_elapsed * 1000, 2),
            round(total_elapsed * 1000, 2),
        )
        return incremental
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.communicate()


def get_runtime_dir(*, voice_manager, user_id: int) -> Path:
    """Return the user-scoped PocketTTS.cpp runtime directory."""
    voices_root = voice_manager.get_user_voices_path(user_id)
    runtime_dir = voices_root / "providers" / PROVIDER_KEY
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _infer_audio_suffix(audio_bytes: bytes) -> str:
    """Infer a likely file suffix so the shared conversion path can decode bytes correctly."""
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return ".wav"
    if audio_bytes[:3] == b"ID3" or (len(audio_bytes) > 1 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
        return ".mp3"
    if audio_bytes[:4] == b"fLaC":
        return ".flac"
    if audio_bytes[:4] == b"OggS":
        return ".ogg"
    if len(audio_bytes) >= 8 and audio_bytes[4:8] == b"ftyp":
        return ".m4a"
    return ".wav"


async def normalize_reference_audio_to_wav(
    audio_bytes: bytes,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
) -> bytes:
    """Normalize incoming reference audio bytes into a provider-safe WAV payload."""
    suffix = _infer_audio_suffix(audio_bytes)
    input_fd, input_name = tempfile.mkstemp(prefix="pocket_tts_cpp_ref_", suffix=suffix)
    output_fd, output_name = tempfile.mkstemp(prefix="pocket_tts_cpp_ref_", suffix=".wav")
    try:
        def _write_input_file() -> None:
            with os.fdopen(input_fd, "wb") as input_file:
                input_file.write(audio_bytes)
                input_file.flush()
                os.fsync(input_file.fileno())

        def _create_output_placeholder() -> None:
            with os.fdopen(output_fd, "wb") as output_file:
                output_file.flush()
                os.fsync(output_file.fileno())

        await asyncio.to_thread(_write_input_file)
        await asyncio.to_thread(_create_output_placeholder)

        input_path = Path(input_name)
        output_path = Path(output_name)
        converted = await AudioConverter.convert_to_wav(
            input_path,
            output_path,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=16,
        )
        if not converted or not output_path.exists():
            raise RuntimeError("Failed to normalize PocketTTS.cpp reference audio to WAV")
        normalized = await asyncio.to_thread(output_path.read_bytes)
        if normalized[:4] != b"RIFF" or normalized[8:12] != b"WAVE":
            raise RuntimeError("PocketTTS.cpp reference normalization did not produce WAV bytes")
        return normalized
    finally:
        with contextlib.suppress(OSError):
            Path(input_name).unlink()
        with contextlib.suppress(OSError):
            Path(output_name).unlink()

def _write_runtime_file_sync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.stem}.",
        suffix=f"{path.suffix}.tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(temp_fd, "wb") as temp_file:
            temp_file.write(payload)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise


async def _write_runtime_file(path: Path, payload: bytes) -> None:
    await asyncio.to_thread(_write_runtime_file_sync, path, payload)


async def materialize_custom_voice_reference(
    *,
    voice_manager,
    user_id: int,
    voice_id: str,
    cache_max_bytes: Optional[int] = None,
) -> Path:
    """Materialize a stored custom voice into the provider runtime cache."""
    runtime_dir = get_runtime_dir(voice_manager=voice_manager, user_id=user_id)
    target_path = runtime_dir / f"custom_{voice_id}.wav"
    voice_bytes = await voice_manager.load_voice_reference_audio(user_id, voice_id)
    normalized_bytes = await normalize_reference_audio_to_wav(voice_bytes)
    await _write_runtime_file(target_path, normalized_bytes)
    return target_path


async def materialize_direct_voice_reference(
    *,
    voice_manager,
    user_id: int,
    voice_reference: bytes,
    persist_direct_voice_references: bool,
    cache_max_bytes: Optional[int] = None,
) -> tuple[Path, bool]:
    """Materialize a direct voice reference into the provider runtime cache."""
    runtime_dir = get_runtime_dir(voice_manager=voice_manager, user_id=user_id)
    normalized_bytes = await normalize_reference_audio_to_wav(voice_reference)
    digest = hashlib.sha256(normalized_bytes).hexdigest()
    if persist_direct_voice_references:
        target_path = runtime_dir / f"ref_{digest}.wav"
    else:
        target_path = runtime_dir / f"ref_{digest}_{uuid.uuid4().hex}.wav"
    await _write_runtime_file(target_path, normalized_bytes)
    return target_path, not persist_direct_voice_references


def prune_materialized_voice_cache(
    runtime_dir: Path,
    *,
    cache_ttl_hours: Optional[int],
    cache_max_bytes: Optional[int],
    protected_paths: Optional[set[Path]] = None,
) -> list[Path]:
    """Remove expired or oversize materialized voice files from the runtime cache."""
    if not runtime_dir.exists():
        return []

    removed: list[Path] = []
    files: list[tuple[Path, float, int]] = []
    now = time.time()
    ttl_seconds = max(0, int(cache_ttl_hours or 0)) * 3600
    protected_resolved = {
        path.resolve() for path in (protected_paths or set()) if path is not None
    }
    protected_resolved.update(get_active_provider_managed_voice_paths(runtime_dir))

    for path in runtime_dir.glob("*.wav"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            resolved = path.resolve()
        except OSError:
            continue
        mtime = float(stat.st_mtime)
        size = int(stat.st_size)
        if ttl_seconds and (now - mtime) > ttl_seconds and resolved not in protected_resolved:
            try:
                path.unlink()
                removed.append(path)
            except OSError as exc:
                logger.warning(
                    "Failed pruning expired PocketTTS.cpp cache file (exception_type={})",
                    type(exc).__name__,
                )
            continue
        files.append((path, mtime, size))

    if cache_max_bytes is None or cache_max_bytes <= 0:
        return removed

    total_bytes = sum(size for _, _, size in files)
    if total_bytes <= cache_max_bytes:
        return removed

    for path, _, size in sorted(files, key=lambda item: item[1]):
        if total_bytes <= cache_max_bytes:
            break
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in protected_resolved:
            continue
        try:
            path.unlink()
            removed.append(path)
            total_bytes -= size
        except OSError as exc:
            logger.warning(
                "Failed pruning oversized PocketTTS.cpp cache file (exception_type={})",
                type(exc).__name__,
            )

    return removed


def cleanup_transient_voice_reference(path: Optional[Path], is_transient: bool) -> None:
    """Delete a transient direct-reference materialization after request completion."""
    if not is_transient or path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "Failed cleaning transient PocketTTS.cpp cache file (exception_type={})",
            type(exc).__name__,
        )
