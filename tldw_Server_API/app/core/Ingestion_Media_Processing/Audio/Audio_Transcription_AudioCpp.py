"""Strict configuration and local contract validation for audio.cpp."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import ipaddress
import json
import math
import os
import re
import stat
import threading
import wave
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, NoReturn, cast
from urllib.parse import urlparse

from tldw_Server_API.app.core import http_client as _http_client
from tldw_Server_API.app.core.exceptions import (
    STTExecutionPlanError,
    STTExecutionUnsupportedError,
    STTTranscriptionError,
)
from tldw_Server_API.app.core.http_client import (
    RetryPolicy,
    afetch,
    create_async_client,
    opaque_stt_http_observability,
    resolve_afetch_transport,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttExecutionRoute,
    SttTranscriptionOutcome,
    _normalize_audio_endpoint,
    actual_execution_from_route,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import (
    open_safe_local_path,
)
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

AUDIO_CPP_ENABLED_ENV = "STT_AUDIO_CPP_ENABLED"
AUDIO_CPP_BASE_URL_ENV = "STT_AUDIO_CPP_BASE_URL"
AUDIO_CPP_DEFAULT_MODEL_ENV = "STT_AUDIO_CPP_DEFAULT_MODEL"
AUDIO_CPP_TIMEOUT_SECONDS_ENV = "STT_AUDIO_CPP_TIMEOUT_SECONDS"

MAX_AUDIO_CPP_RESPONSE_BYTES = 1_048_576
MAX_AUDIO_CPP_CATALOG_ENTRIES = 256
MAX_AUDIO_CPP_HEALTH_MODELS = MAX_AUDIO_CPP_CATALOG_ENTRIES
MAX_AUDIO_CPP_JSON_INTEGER_DIGITS = 128
MAX_AUDIO_CPP_TRANSCRIPT_CHARS = 262_144

_TRUE_TOKENS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "n", "off"})
_SELECTORS = ("audio-cpp", "audiocpp", "audio_cpp")
_SUPPORTED_ASR_MODES = frozenset({"offline", "streaming"})
_MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._+-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._+-]*)?$"
)
_CONTRACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_MAX_MODEL_ID_LENGTH = 256
_MAX_CONTRACT_ID_LENGTH = 64
_WAV_READ_CHUNK_BYTES = 64 * 1024
_AUDIO_CPP_PATHS = (
    "/health",
    "/v1/models",
    "/v1/audio/transcriptions",
)
_AUDIO_CPP_BUSY_STATUSES = frozenset({409, 429, 503})
_AUDIO_CPP_MODEL_UNAVAILABLE_STATUSES = frozenset({404, 422})

_Afetch = Callable[..., Awaitable[Any]]
_DiscoveryCacheKey = tuple[str, str]

_audio_cpp_discovery_cache: dict[_DiscoveryCacheKey, AudioCppDiscovery] = {}
_audio_cpp_discovery_inflight: dict[
    _DiscoveryCacheKey,
    concurrent.futures.Future[AudioCppDiscovery],
] = {}
_audio_cpp_discovery_leader_loops: dict[
    _DiscoveryCacheKey,
    asyncio.AbstractEventLoop,
] = {}
# ponytail: keep one process lock; shard only after measured contention.
_audio_cpp_discovery_lock = threading.Lock()
_audio_cpp_discovery_generation = 0


@dataclass(frozen=True)
class AudioCppConfig:
    """Validated settings for one external audio.cpp server."""

    enabled: bool
    origin: str
    default_model: str | None
    timeout_seconds: float


@dataclass(frozen=True)
class AudioCppDiscovery:
    """Validated audio.cpp model metadata selected from discovery."""

    backend: str
    model_id: str
    family: str
    mode: str


class _InvalidAudioCppJSON(ValueError):
    """Internal marker for invalid JSON values."""


def _reject_json_constant(_value: str) -> NoReturn:
    raise _InvalidAudioCppJSON


def _parse_bounded_json_int(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > MAX_AUDIO_CPP_JSON_INTEGER_DIGITS:
        raise _InvalidAudioCppJSON
    return int(value)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidAudioCppJSON
        result[key] = value
    return result


def _parse_json_object(body: bytes, *, response_name: str) -> dict[str, object]:
    message = f"audio.cpp {response_name} response is invalid"
    if not isinstance(body, bytes) or len(body) > MAX_AUDIO_CPP_RESPONSE_BYTES:
        raise STTExecutionUnsupportedError(message)
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_int=_parse_bounded_json_int,
        )
    except (RecursionError, TypeError, ValueError):
        raise STTExecutionUnsupportedError(message) from None
    if not isinstance(value, dict):
        raise STTExecutionUnsupportedError(message)
    return value


def _is_safe_contract_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= _MAX_CONTRACT_ID_LENGTH
        and _CONTRACT_ID_RE.fullmatch(value) is not None
    )


def _is_safe_model_id(value: object) -> bool:
    return isinstance(value, str) and len(value) <= _MAX_MODEL_ID_LENGTH and _MODEL_ID_RE.fullmatch(value) is not None


def parse_audio_cpp_health(body: bytes) -> str:
    """Validate one bounded ``/health`` response and return its backend."""
    payload = _parse_json_object(body, response_name="health")
    backend = payload.get("backend")
    model_count = payload.get("models")
    if (
        payload.get("status") != "ok"
        or not _is_safe_contract_id(backend)
        or isinstance(model_count, bool)
        or not isinstance(model_count, int)
        or model_count < 0
        or model_count > MAX_AUDIO_CPP_HEALTH_MODELS
    ):
        raise STTExecutionUnsupportedError("audio.cpp health response is invalid")
    return backend


def parse_audio_cpp_catalog(
    body: bytes,
    *,
    backend: str,
    model_id: str,
) -> AudioCppDiscovery:
    """Validate a bounded model catalog and select an exact ASR model."""
    requested_model = _safe_model_id(model_id)
    if not _is_safe_contract_id(backend):
        raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")
    payload = _parse_json_object(body, response_name="catalog")
    entries = payload.get("data")
    if payload.get("object") != "list" or not isinstance(entries, list) or len(entries) > MAX_AUDIO_CPP_CATALOG_ENTRIES:
        raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")

    selected: AudioCppDiscovery | None = None
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")
        entry_id = entry.get("id")
        family = entry.get("family")
        task = entry.get("task")
        mode = entry.get("mode")
        if (
            not _is_safe_model_id(entry_id)
            or entry_id in seen_ids
            or not _is_safe_contract_id(family)
            or not _is_safe_contract_id(task)
            or not _is_safe_contract_id(mode)
        ):
            raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")
        seen_ids.add(entry_id)
        if entry_id == requested_model:
            if task != "asr" or mode not in _SUPPORTED_ASR_MODES:
                raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")
            selected = AudioCppDiscovery(
                backend=backend,
                model_id=entry_id,
                family=family,
                mode=mode,
            )

    if selected is None:
        raise STTExecutionUnsupportedError("audio.cpp requested model is unavailable")
    return selected


def parse_audio_cpp_transcription(body: bytes) -> str:
    """Validate a bounded transcription response and return its exact text."""
    payload = _parse_json_object(body, response_name="transcription")
    text = payload.get("text")
    if not isinstance(text, str) or len(text) > MAX_AUDIO_CPP_TRANSCRIPT_CHARS:
        raise STTExecutionUnsupportedError("audio.cpp transcription response is invalid")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise STTExecutionUnsupportedError("audio.cpp transcription response is invalid") from None
    return text


def _raw_setting(
    settings: Mapping[str, object],
    key: str,
    *,
    environment: Mapping[str, str],
    environment_key: str,
    default: object,
) -> object:
    if environment_key in environment:
        return environment[environment_key]
    return settings.get(key, default)


def _parse_enabled(raw: object) -> bool:
    if not isinstance(raw, str):
        raise STTExecutionUnsupportedError("audio.cpp enabled setting is invalid")
    token = raw.strip().casefold()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise STTExecutionUnsupportedError("audio.cpp enabled setting is invalid")


def _parse_timeout(raw: object) -> float:
    if isinstance(raw, bool) or type(raw) not in {str, int, float}:
        raise STTExecutionUnsupportedError("audio.cpp timeout setting is invalid")
    try:
        timeout = float(raw)
    except (OverflowError, TypeError, ValueError):
        raise STTExecutionUnsupportedError("audio.cpp timeout setting is invalid") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise STTExecutionUnsupportedError("audio.cpp timeout setting is invalid")
    return timeout


def _canonical_origin(raw: object) -> str:
    if not isinstance(raw, str) or raw != raw.strip():
        raise STTExecutionUnsupportedError("audio.cpp origin is invalid")
    try:
        parsed = urlparse(raw)
        if parsed.path not in {"", "/"}:
            raise ValueError
        hostname = parsed.hostname
        if hostname is None:
            raise ValueError
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if ("[" in parsed.netloc or "]" in parsed.netloc) and not isinstance(address, ipaddress.IPv6Address):
            raise ValueError
        if address is None and len(hostname) > 253:
            raise ValueError
        normalized, _egress, _endpoint_id = _normalize_audio_endpoint(raw)
        normalized_parsed = urlparse(normalized)
    except (STTExecutionUnsupportedError, TypeError, ValueError):
        raise STTExecutionUnsupportedError("audio.cpp origin is invalid") from None
    return f"{normalized_parsed.scheme}://{normalized_parsed.netloc}"


def _safe_model_id(raw: object) -> str:
    if not isinstance(raw, str):
        raise STTExecutionUnsupportedError("audio.cpp model is invalid")
    stripped = raw.strip()
    if not stripped:
        raise STTExecutionUnsupportedError("audio.cpp model is required")
    if raw != stripped:
        raise STTExecutionUnsupportedError("audio.cpp model is invalid")
    model_id = raw
    if len(model_id) > _MAX_MODEL_ID_LENGTH or _MODEL_ID_RE.fullmatch(model_id) is None:
        raise STTExecutionUnsupportedError("audio.cpp model is invalid")
    return model_id


def load_audio_cpp_config(
    stt_settings: Mapping[str, object],
    *,
    env: Mapping[str, str] | None = None,
) -> AudioCppConfig:
    """Load and strictly validate raw audio.cpp settings with env precedence."""
    environment = os.environ if env is None else env
    enabled = _parse_enabled(
        _raw_setting(
            stt_settings,
            "audio_cpp_enabled",
            environment=environment,
            environment_key=AUDIO_CPP_ENABLED_ENV,
            default="false",
        )
    )
    origin = _canonical_origin(
        _raw_setting(
            stt_settings,
            "audio_cpp_base_url",
            environment=environment,
            environment_key=AUDIO_CPP_BASE_URL_ENV,
            default="http://127.0.0.1:8080",
        )
    )
    raw_default_model = _raw_setting(
        stt_settings,
        "audio_cpp_default_model",
        environment=environment,
        environment_key=AUDIO_CPP_DEFAULT_MODEL_ENV,
        default="",
    )
    if not isinstance(raw_default_model, str):
        raise STTExecutionUnsupportedError("audio.cpp model is invalid")
    default_model = _safe_model_id(raw_default_model) if raw_default_model.strip() else None
    timeout_seconds = _parse_timeout(
        _raw_setting(
            stt_settings,
            "audio_cpp_timeout_seconds",
            environment=environment,
            environment_key=AUDIO_CPP_TIMEOUT_SECONDS_ENV,
            default="600",
        )
    )
    return AudioCppConfig(
        enabled=enabled,
        origin=origin,
        default_model=default_model,
        timeout_seconds=timeout_seconds,
    )


def normalize_audio_cpp_model(
    model: str | None,
    *,
    default_model: str | None,
) -> str:
    """Return the exact safe server model selected for audio.cpp."""
    if model is None:
        selected: object = default_model
    elif not isinstance(model, str):
        selected = model
    else:
        selected = model
        normalized_selector = selected.lower()
        if normalized_selector in _SELECTORS:
            selected = default_model
        else:
            for selector in _SELECTORS:
                prefix = f"{selector}:"
                if normalized_selector.startswith(prefix):
                    selected = selected[len(prefix) :]
                    break
    if selected is None:
        raise STTExecutionUnsupportedError("audio.cpp model is required")
    return _safe_model_id(selected)


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError
    return value


def _validate_audio_cpp_pcm_wav(handle: BinaryIO, *, file_size: int) -> None:
    header = _read_exact(handle, 12)
    riff_size = int.from_bytes(header[4:8], byteorder="little")
    if header[:4] != b"RIFF" or header[8:] != b"WAVE" or riff_size < 4 or riff_size + 8 != file_size:
        raise ValueError
    handle.seek(0)

    reader = wave.open(handle, "rb")
    try:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        frame_rate = reader.getframerate()
        frame_count = reader.getnframes()
        frame_size = channels * sample_width
        if (
            reader.getcomptype() != "NONE"
            or channels < 1
            or sample_width < 1
            or frame_rate < 1
            or frame_count < 1
            or frame_size > _WAV_READ_CHUNK_BYTES
        ):
            raise ValueError

        expected_total = frame_count * frame_size
        if expected_total > file_size:
            raise ValueError
        frames_per_chunk = max(1, _WAV_READ_CHUNK_BYTES // frame_size)
        remaining_frames = frame_count
        total_bytes = 0
        while remaining_frames:
            requested_frames = min(remaining_frames, frames_per_chunk)
            chunk = reader.readframes(requested_frames)
            expected_chunk_bytes = requested_frames * frame_size
            if len(chunk) != expected_chunk_bytes:
                raise ValueError
            total_bytes += len(chunk)
            remaining_frames -= requested_frames
        if total_bytes != expected_total or reader.readframes(1):
            raise ValueError
    finally:
        reader.close()

    if handle.closed:
        raise ValueError
    handle.seek(0)


def open_audio_cpp_wav(
    path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str],
) -> BinaryIO:
    """Open and fully validate one regular uncompressed PCM RIFF/WAVE file."""
    message = "audio.cpp WAV input is invalid"
    handle: BinaryIO | None = None
    try:
        path_object = Path(path)
        base_path = Path(base_dir).resolve(strict=False)
        candidate = path_object if path_object.is_absolute() else base_path / path_object
        if candidate.suffix.casefold() != ".wav":
            raise ValueError
        before = os.lstat(candidate)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError

        opened = open_safe_local_path(candidate, base_path, mode="rb")
        if opened is None:
            raise ValueError
        handle = cast(BinaryIO, opened)
        after = os.fstat(handle.fileno())
        if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError

        _validate_audio_cpp_pcm_wav(handle, file_size=after.st_size)
        handle.seek(0)
        return handle
    except (
        OSError,
        OverflowError,
        TypeError,
        UnicodeError,
        ValueError,
        EOFError,
        wave.Error,
    ):
        if handle is not None:
            handle.close()
        raise STTExecutionUnsupportedError(message) from None


class _AudioCppTransportError(RuntimeError):
    """Internal marker for one failed HTTP exchange."""


class _AudioCppStatusError(RuntimeError):
    """Internal marker for one rejected HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__("audio.cpp HTTP status is invalid")
        self.status_code = status_code


def audio_cpp_routes(origin: str) -> tuple[str, str, str]:
    """Build the three fixed audio.cpp routes from one canonical origin."""
    try:
        if _canonical_origin(origin) != origin:
            raise ValueError
        routes = tuple(_normalize_audio_endpoint(f"{origin}{path}")[0] for path in _AUDIO_CPP_PATHS)
        if routes != tuple(f"{origin}{path}" for path in _AUDIO_CPP_PATHS):
            raise ValueError
    except (STTExecutionUnsupportedError, TypeError, ValueError):
        raise STTExecutionUnsupportedError("audio.cpp origin is invalid") from None
    return cast(tuple[str, str, str], routes)


def reset_audio_cpp_discovery_cache() -> None:
    """Clear process-local audio.cpp discovery state."""
    global _audio_cpp_discovery_generation
    with _audio_cpp_discovery_lock:
        _audio_cpp_discovery_generation += 1
        _audio_cpp_discovery_cache.clear()
        _audio_cpp_discovery_inflight.clear()
        _audio_cpp_discovery_leader_loops.clear()


def _invalidate_audio_cpp_discovery(
    key: _DiscoveryCacheKey,
    discovery: AudioCppDiscovery,
) -> None:
    with _audio_cpp_discovery_lock:
        if _audio_cpp_discovery_cache.get(key) is discovery:
            _audio_cpp_discovery_cache.pop(key, None)


def _validate_audio_cpp_execution(
    *,
    route: SttExecutionRoute,
    origin: str,
    model_id: str,
    timeout_seconds: float,
    transport: str,
) -> tuple[tuple[str, str, str], _DiscoveryCacheKey]:
    message = "Invalid audio.cpp execution route"
    try:
        routes = audio_cpp_routes(origin)
        if _safe_model_id(model_id) != model_id:
            raise ValueError
        if type(timeout_seconds) is not float or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError
        if (
            not isinstance(transport, str)
            or transport != transport.strip().lower()
            or resolve_afetch_transport(transport) != transport
        ):
            raise ValueError
        _normalized, egress, endpoint_id = _normalize_audio_endpoint(routes[2])
        if (
            not isinstance(route, SttExecutionRoute)
            or route.provider != "audio-cpp"
            or route.backend != "audio_cpp_http"
            or route.source != "audio_cpp_http"
            or route.model_label != model_id
            or route.audio_egress is not egress
            or route.endpoint_id != endpoint_id
            or route.transport != transport
            or route.identity_resolved
            or route.artifact_id is not None
            or route.device is not None
            or route.compute_type is not None
            or route.dtype is not None
            or route.decoding_ids != ()
            or route.local_model_available
            or route.would_download
        ):
            raise ValueError
    except (
        RuntimeError,
        STTExecutionUnsupportedError,
        TypeError,
        ValueError,
    ):
        raise STTExecutionPlanError(message) from None
    return routes, (endpoint_id, model_id)


async def _close_audio_cpp_response(response: object) -> None:
    closer = getattr(response, "aclose", None)
    if closer is None:
        closer = getattr(response, "close", None)
    if not callable(closer):
        raise TypeError
    result = closer()
    if inspect.isawaitable(result):
        await result


async def _close_audio_cpp_response_before_cancellation(
    response: object,
) -> None:
    close_task = asyncio.create_task(_close_audio_cpp_response(response))
    caller_cancellation: asyncio.CancelledError | None = None
    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancellation:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                caller_cancellation = caller_cancellation or cancellation
        except Exception:  # noqa: BLE001 - closer exception is inspected below
            break
    try:
        close_task.result()
    except asyncio.CancelledError:
        if caller_cancellation is not None:
            raise caller_cancellation from None
        raise RuntimeError("audio.cpp cleanup failed") from None
    except Exception:
        if caller_cancellation is not None:
            raise caller_cancellation from None
        raise
    if caller_cancellation is not None:
        raise caller_cancellation


def _create_audio_cpp_client(
    *,
    transport: str,
    timeout_seconds: float,
) -> object:
    """Create one non-shared async client for a single transcription."""
    if transport == "httpx":
        return create_async_client(
            timeout=timeout_seconds,
            trust_env=False,
            verify=True,
        )
    if transport == "aiohttp" and _http_client.aiohttp is not None:
        return _http_client.aiohttp.ClientSession(
            timeout=_http_client.aiohttp.ClientTimeout(
                total=timeout_seconds,
            ),
            trust_env=False,
        )
    raise RuntimeError("audio.cpp transport client is unavailable")


async def _close_audio_cpp_client(client: object) -> None:
    await _close_audio_cpp_response_before_cancellation(client)


async def _open_audio_cpp_wav_async(
    path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str],
) -> BinaryIO:
    """Validate off-loop without abandoning a handle on cancellation."""
    open_task = asyncio.create_task(
        asyncio.to_thread(
            open_audio_cpp_wav,
            path,
            base_dir=base_dir,
        )
    )
    caller_cancellation: asyncio.CancelledError | None = None
    while not open_task.done():
        try:
            await asyncio.shield(open_task)
        except asyncio.CancelledError as cancellation:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                caller_cancellation = caller_cancellation or cancellation
        except Exception:  # noqa: BLE001 - task result is inspected below
            break
    try:
        handle = open_task.result()
    except BaseException:
        if caller_cancellation is not None:
            raise caller_cancellation from None
        raise
    if caller_cancellation is not None:
        try:
            await _close_audio_cpp_response_before_cancellation(handle)
        except BaseException:  # noqa: BLE001 - preserve original caller cancellation
            raise caller_cancellation from None
        raise caller_cancellation
    return handle


async def _request_audio_cpp(
    *,
    method: str,
    url: str,
    origin: str,
    timeout_seconds: float,
    transport: str,
    response_name: str,
    afetch_fn: _Afetch,
    client: object | None = None,
    files: dict[str, tuple[str, BinaryIO, str]] | None = None,
    data: dict[str, str] | None = None,
) -> bytes:
    _normalized, _egress, endpoint_id = _normalize_audio_endpoint(url)
    response: object | None = None
    request_error: BaseException | None = None
    request_kwargs: dict[str, Any] = {
        "method": method,
        "url": url,
        "timeout": timeout_seconds,
        "retry": RetryPolicy(
            attempts=1,
            retry_on_status=(),
            retry_on_methods=(),
        ),
        "allow_redirects": False,
        "verify": True,
        "transport": transport,
        "configured_endpoint": ConfiguredEndpointScope.from_url(origin),
        "max_response_bytes": MAX_AUDIO_CPP_RESPONSE_BYTES,
    }
    if client is not None:
        request_kwargs["client"] = client
    if files is not None:
        request_kwargs["files"] = files
    if data is not None:
        request_kwargs["data"] = data
    try:
        try:
            with opaque_stt_http_observability(endpoint_id):
                response = await afetch_fn(**request_kwargs)
            status_code = int(cast(Any, response).status_code)
            if status_code != 200:
                raise _AudioCppStatusError(status_code)
            content = cast(Any, response).content
            if type(content) is not bytes or len(content) > MAX_AUDIO_CPP_RESPONSE_BYTES:
                raise STTExecutionUnsupportedError(f"audio.cpp {response_name} response is invalid")
            return content
        except (
            _AudioCppStatusError,
            STTExecutionUnsupportedError,
        ):
            raise
        except Exception:  # noqa: BLE001 - transport adapters have no shared exception base
            raise _AudioCppTransportError from None
    except BaseException as exc:
        request_error = exc
        raise
    finally:
        if response is not None:
            try:
                await _close_audio_cpp_response_before_cancellation(response)
            except asyncio.CancelledError:
                if request_error is None:
                    raise
            except Exception:  # noqa: BLE001 - response closers are adapter-defined
                if request_error is None:
                    raise _AudioCppTransportError from None


def _raise_audio_cpp_runtime_error(
    error: _AudioCppStatusError | _AudioCppTransportError,
    *,
    transcription: bool,
) -> NoReturn:
    if (
        transcription
        and isinstance(error, _AudioCppStatusError)
        and error.status_code in _AUDIO_CPP_MODEL_UNAVAILABLE_STATUSES
    ):
        raise STTTranscriptionError("audio.cpp requested model is unavailable") from None
    if isinstance(error, _AudioCppStatusError) and error.status_code in _AUDIO_CPP_BUSY_STATUSES:
        raise STTTranscriptionError("audio.cpp server is busy") from None
    raise STTTranscriptionError("audio.cpp request failed") from None


async def _fetch_audio_cpp_discovery(
    *,
    health_url: str,
    catalog_url: str,
    origin: str,
    model_id: str,
    timeout_seconds: float,
    transport: str,
    afetch_fn: _Afetch,
    client: object | None,
) -> AudioCppDiscovery:
    try:
        health_body = await _request_audio_cpp(
            method="GET",
            url=health_url,
            origin=origin,
            timeout_seconds=timeout_seconds,
            transport=transport,
            response_name="health",
            afetch_fn=afetch_fn,
            client=client,
        )
        backend = parse_audio_cpp_health(health_body)
        catalog_body = await _request_audio_cpp(
            method="GET",
            url=catalog_url,
            origin=origin,
            timeout_seconds=timeout_seconds,
            transport=transport,
            response_name="catalog",
            afetch_fn=afetch_fn,
            client=client,
        )
        return parse_audio_cpp_catalog(
            catalog_body,
            backend=backend,
            model_id=model_id,
        )
    except (_AudioCppStatusError, _AudioCppTransportError) as exc:
        _raise_audio_cpp_runtime_error(exc, transcription=False)


async def _audio_cpp_discovery(
    *,
    key: _DiscoveryCacheKey,
    health_url: str,
    catalog_url: str,
    origin: str,
    model_id: str,
    timeout_seconds: float,
    transport: str,
    afetch_fn: _Afetch,
    client: object | None,
) -> AudioCppDiscovery:
    leader = False
    loop = asyncio.get_running_loop()
    with _audio_cpp_discovery_lock:
        cached = _audio_cpp_discovery_cache.get(key)
        if cached is not None:
            return cached
        future = _audio_cpp_discovery_inflight.get(key)
        if future is None:
            future = concurrent.futures.Future()
            _audio_cpp_discovery_inflight[key] = future
            _audio_cpp_discovery_leader_loops[key] = loop
            generation = _audio_cpp_discovery_generation
            leader = True
        else:
            generation = _audio_cpp_discovery_generation

    if not leader:
        wrapped = asyncio.wrap_future(future)

        def consume_follower_exception(done: asyncio.Future[AudioCppDiscovery]) -> None:
            if not done.cancelled():
                done.exception()

        wrapped.add_done_callback(consume_follower_exception)
        return await asyncio.shield(wrapped)

    try:
        discovery = await _fetch_audio_cpp_discovery(
            health_url=health_url,
            catalog_url=catalog_url,
            origin=origin,
            model_id=model_id,
            timeout_seconds=timeout_seconds,
            transport=transport,
            afetch_fn=afetch_fn,
            client=client,
        )
    except BaseException as exc:
        with _audio_cpp_discovery_lock:
            if generation == _audio_cpp_discovery_generation:
                _audio_cpp_discovery_cache.pop(key, None)
            if _audio_cpp_discovery_inflight.get(key) is future:
                _audio_cpp_discovery_inflight.pop(key, None)
                _audio_cpp_discovery_leader_loops.pop(key, None)
        if not future.done():
            future.set_exception(exc)
        raise
    else:
        with _audio_cpp_discovery_lock:
            if generation == _audio_cpp_discovery_generation and _audio_cpp_discovery_inflight.get(key) is future:
                _audio_cpp_discovery_cache[key] = discovery
            if _audio_cpp_discovery_inflight.get(key) is future:
                _audio_cpp_discovery_inflight.pop(key, None)
                _audio_cpp_discovery_leader_loops.pop(key, None)
        if not future.done():
            future.set_result(discovery)
        return discovery


async def transcribe_audio_cpp_async(
    audio_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str],
    route: SttExecutionRoute,
    origin: str,
    model_id: str,
    timeout_seconds: float,
    transport: str,
    language: str | None = None,
    afetch_fn: _Afetch | None = None,
) -> SttTranscriptionOutcome:
    """Execute one frozen, planned audio.cpp batch transcription."""
    routes, key = _validate_audio_cpp_execution(
        route=route,
        origin=origin,
        model_id=model_id,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    if language is not None and not isinstance(language, str):
        raise STTExecutionPlanError("Invalid audio.cpp execution route")
    selected_afetch = afetch if afetch_fn is None else afetch_fn
    client: object | None = None
    execution_error: BaseException | None = None
    try:
        upload = await _open_audio_cpp_wav_async(
            audio_path,
            base_dir=base_dir,
        )
        with upload:
            if afetch_fn is None:
                try:
                    client = _create_audio_cpp_client(
                        transport=transport,
                        timeout_seconds=timeout_seconds,
                    )
                except Exception:  # noqa: BLE001 - client implementations vary by transport
                    raise STTTranscriptionError(
                        "audio.cpp request failed"
                    ) from None

            discovery = await _audio_cpp_discovery(
                key=key,
                health_url=routes[0],
                catalog_url=routes[1],
                origin=origin,
                model_id=model_id,
                timeout_seconds=timeout_seconds,
                transport=transport,
                afetch_fn=selected_afetch,
                client=client,
            )
            upload.seek(0)
            data = {"model": model_id}
            if language is not None:
                data["language"] = language
            try:
                body = await _request_audio_cpp(
                    method="POST",
                    url=routes[2],
                    origin=origin,
                    timeout_seconds=timeout_seconds,
                    transport=transport,
                    response_name="transcription",
                    afetch_fn=selected_afetch,
                    client=client,
                    files={"file": ("audio.wav", upload, "audio/wav")},
                    data=data,
                )
            except (_AudioCppStatusError, _AudioCppTransportError) as exc:
                if isinstance(exc, _AudioCppTransportError) or exc.status_code in _AUDIO_CPP_MODEL_UNAVAILABLE_STATUSES:
                    _invalidate_audio_cpp_discovery(key, discovery)
                _raise_audio_cpp_runtime_error(exc, transcription=True)

        text = parse_audio_cpp_transcription(body)
        return SttTranscriptionOutcome(
            artifact={
                "text": text,
                "segments": [],
                "language": language,
                "diarization": {"enabled": False, "speakers": None},
                "usage": {"duration_ms": None, "tokens": None},
                "metadata": {
                    "provider": "audio-cpp",
                    "contract": "audio_cpp_http_v1",
                    "model_id": model_id,
                    "model_family": discovery.family,
                    "model_mode": discovery.mode,
                    "server_backend": discovery.backend,
                },
            },
            actual_execution=actual_execution_from_route(route, device=None),
        )
    except BaseException as exc:
        execution_error = exc
        raise
    finally:
        if client is not None:
            try:
                await _close_audio_cpp_client(client)
            except asyncio.CancelledError:
                if execution_error is None:
                    raise
            except Exception:  # noqa: BLE001 - client closers vary by transport
                if execution_error is None:
                    raise STTTranscriptionError(
                        "audio.cpp request failed"
                    ) from None


def transcribe_audio_cpp(
    audio_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str],
    route: SttExecutionRoute,
    origin: str,
    model_id: str,
    timeout_seconds: float,
    transport: str,
    language: str | None = None,
    afetch_fn: _Afetch | None = None,
) -> SttTranscriptionOutcome:
    """Synchronously execute one frozen audio.cpp transcription."""

    async def run() -> SttTranscriptionOutcome:
        return await transcribe_audio_cpp_async(
            audio_path,
            base_dir=base_dir,
            route=route,
            origin=origin,
            model_id=model_id,
            timeout_seconds=timeout_seconds,
            transport=transport,
            language=language,
            afetch_fn=afetch_fn,
        )

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is None or not running_loop.is_running():
        return asyncio.run(run())

    _routes, key = _validate_audio_cpp_execution(
        route=route,
        origin=origin,
        model_id=model_id,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
    with _audio_cpp_discovery_lock:
        if _audio_cpp_discovery_leader_loops.get(key) is running_loop:
            raise STTExecutionPlanError(
                "Invalid audio.cpp execution route"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(run())).result()
