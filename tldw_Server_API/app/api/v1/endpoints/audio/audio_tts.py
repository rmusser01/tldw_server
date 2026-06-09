# audio_tts.py
# Description: Audio TTS endpoints and helpers.
import base64
import contextlib
import inspect
import json
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger
from starlette import status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.Audio.tts_service import _raise_for_tts_error
from tldw_Server_API.app.core.Audio.tts_service import _sanitize_speech_request
from tldw_Server_API.app.core.AuthNZ.exceptions import QuotaExceededError, StorageError
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSError,
    TTSAuthenticationError,
    TTSProviderNotConfiguredError,
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2, get_tts_service_v2
from tldw_Server_API.app.core.TTS.utils import (
    build_tts_segments_payload,
    compute_tts_history_text_hash,
    parse_bool,
    tts_history_text_length,
)
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat
from tldw_Server_API.app.core.Storage.generated_file_helpers import (
    save_and_register_tts_audio,
)

_AUDIO_TTS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
    TTSError,
    HTTPException,
    QuotaExceededError,
    StorageError,
)

router = APIRouter(
    tags=["Audio"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)


def get_job_manager() -> JobManager:
    """Lazy import to avoid loading audio_jobs (and heavy transcriber deps) at module import time."""
    from tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs import get_job_manager as _get_job_manager

    return _get_job_manager()


def _audio_shim_attr(name: str):
    defaults: dict[str, Any] = {
        "_sanitize_speech_request": _sanitize_speech_request,
        "save_and_register_tts_audio": save_and_register_tts_audio,
    }
    try:
        from tldw_Server_API.app.api.v1.endpoints import audio as audio_shim

        if hasattr(audio_shim, name):
            return getattr(audio_shim, name)
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
        pass
    if name in defaults:
        return defaults[name]
    raise NameError(name)


def _tts_history_config() -> dict[str, Any]:
    return {
        "enabled": parse_bool(getattr(settings, "TTS_HISTORY_ENABLED", False), default=False),
        "store_text": parse_bool(getattr(settings, "TTS_HISTORY_STORE_TEXT", True), default=True),
        "store_failed": parse_bool(getattr(settings, "TTS_HISTORY_STORE_FAILED", True), default=True),
        "hash_key": getattr(settings, "TTS_HISTORY_HASH_KEY", None),
    }


def _extract_tts_metadata(request_data: OpenAISpeechRequest) -> dict[str, Any]:
    metadata = getattr(request_data, "_tts_metadata", None)
    return metadata if isinstance(metadata, dict) else {}


_AUDIO_CONTENT_TYPE_MAP = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/L16; rate=24000; channels=1",
}


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
        return None
    return parsed if parsed > 0 else None


def _resolve_pcm_sample_rate(request_data: OpenAISpeechRequest) -> int:
    metadata = _extract_tts_metadata(request_data)
    candidates: list[Any] = [
        metadata.get("sample_rate"),
        getattr(request_data, "target_sample_rate", None),
    ]
    extra_params = getattr(request_data, "extra_params", None)
    if isinstance(extra_params, dict):
        candidates.append(extra_params.get("target_sample_rate"))
        candidates.append(extra_params.get("sample_rate"))

    for candidate in candidates:
        parsed = _coerce_positive_int(candidate)
        if parsed is not None:
            return parsed
    return 24000


def _resolve_response_content_type(request_data: OpenAISpeechRequest) -> str:
    base_type = _AUDIO_CONTENT_TYPE_MAP.get(request_data.response_format, "audio/mpeg")
    if request_data.response_format != "pcm":
        return base_type
    sample_rate = _resolve_pcm_sample_rate(request_data)
    return f"audio/L16; rate={sample_rate}; channels=1"


def _append_pcm_response_headers(request_data: OpenAISpeechRequest, headers: dict[str, str]) -> None:
    if request_data.response_format != "pcm":
        return
    headers["X-Audio-Sample-Rate"] = str(_resolve_pcm_sample_rate(request_data))


def _coerce_nonempty_catalog_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_openai_voice_catalog(voices_by_provider: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    identity_keys = {"id", "voice_id", "name", "language", "lang"}
    for provider_name, voices in voices_by_provider.items():
        provider = _coerce_nonempty_catalog_string(provider_name) or "unknown"
        if not isinstance(voices, list):
            continue
        for voice in voices:
            if not isinstance(voice, dict):
                continue
            voice_id = _coerce_nonempty_catalog_string(
                voice.get("id") or voice.get("voice_id") or voice.get("name")
            )
            if voice_id is None:
                continue
            entry: dict[str, Any] = {
                "id": voice_id,
                "object": "voice",
                "provider": provider,
            }
            name = _coerce_nonempty_catalog_string(voice.get("name"))
            if name is not None:
                entry["name"] = name
            language = _coerce_nonempty_catalog_string(voice.get("language") or voice.get("lang"))
            if language is not None:
                entry["language"] = language
            metadata = {
                key: value
                for key, value in voice.items()
                if key not in identity_keys and value is not None
            }
            if metadata:
                entry["metadata"] = metadata
            data.append(entry)
    return {"object": "list", "data": data}


_MODEL_INFO_SENSITIVE_KEYS = {
    "apikey",
    "apitoken",
    "authorization",
    "baseurl",
    "command",
    "cwd",
    "env",
    "localpath",
    "modelpath",
    "password",
    "path",
    "privatekey",
    "repopath",
    "secret",
    "stderr",
    "stdout",
    "token",
    "vctargetpath",
}


def _normalize_model_info_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _looks_like_local_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text.startswith(("~", "/Users/", "/home/", "/var/", "/private/", "/opt/")):
        return True
    return len(text) >= 3 and text[1:3] in {":\\", ":/"}


def _sanitize_model_info_value(value: Any) -> Any:
    if isinstance(value, str):
        if _looks_like_local_path(value):
            return "<redacted-path>"
        return value
    if isinstance(value, list):
        return [_sanitize_model_info_value(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _normalize_model_info_key(key) in _MODEL_INFO_SENSITIVE_KEYS:
                continue
            sanitized[key] = _sanitize_model_info_value(item)
        return sanitized
    return value


def _find_provider_entry(mapping: Any, provider: str) -> tuple[Optional[str], Optional[Any]]:
    if not isinstance(mapping, dict):
        return None, None
    requested = provider.strip().lower()
    for key, value in mapping.items():
        if str(key).strip().lower() == requested:
            return str(key), value
    return None, None


def _extract_model_ids(capabilities: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    candidates = metadata.get("supported_model_ids")
    if not isinstance(candidates, list):
        candidates = capabilities.get("models")
    if not isinstance(candidates, list):
        return []
    return [
        model_id
        for item in candidates
        if (model_id := _coerce_nonempty_catalog_string(item)) is not None
    ]


def _build_provider_model_info(
    *,
    provider: str,
    provider_status: Optional[Any],
    provider_capabilities: Optional[Any],
) -> dict[str, Any]:
    status_info = provider_status if isinstance(provider_status, dict) else {}
    capabilities = provider_capabilities if isinstance(provider_capabilities, dict) else {}
    sanitized_capabilities = _sanitize_model_info_value(capabilities)
    if not isinstance(sanitized_capabilities, dict):
        sanitized_capabilities = {}
    metadata = sanitized_capabilities.get("metadata") if isinstance(sanitized_capabilities, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    initialized = bool(status_info.get("initialized", False))
    status_value = _coerce_nonempty_catalog_string(status_info.get("status")) or "unknown"
    model_info = {
        "provider": provider,
        "status": status_value,
        "initialized": initialized,
        "loaded": initialized,
        "failed": bool(status_info.get("failed", False)),
        "model_ids": _extract_model_ids(sanitized_capabilities, metadata),
        "model_families": metadata.get("model_families", {}),
        "voice_conversion": metadata.get("voice_conversion"),
        "capabilities": sanitized_capabilities,
        "unload_endpoint": f"/api/v1/audio/tts/providers/{provider}/unload",
    }
    if model_info["voice_conversion"] is None:
        model_info.pop("voice_conversion")
    return model_info


def _tts_history_error_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, (dict, list)):
            try:
                return json.dumps(detail, separators=(",", ":"), ensure_ascii=True)
            except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
                return str(detail)
        return str(detail)
    return str(exc)


@router.post(
    "/speech/jobs",
    summary="Submit a long-form TTS job",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.speech", count_as="call")),
    ],
)
async def create_speech_job(
    request_data: OpenAISpeechRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
):
    """Submit a long-form TTS job and return job id."""
    request_id = ensure_request_id(request)
    provider_hint = _audio_shim_attr("_sanitize_speech_request")(request_data, request_id=request_id)

    user_id_int, tts_overrides, _byok = await _audio_shim_attr("_resolve_tts_byok")(
        provider_hint=provider_hint,
        current_user=current_user,
        request=request,
    )

    payload = {
        "speech_request": model_dump_compat(request_data),
        "provider_hint": provider_hint,
        "provider_overrides": tts_overrides,
        "user_id": user_id_int,
    }
    # Force non-streaming in jobs worker.
    payload["speech_request"]["stream"] = False

    job = jm.create_job(
        domain="audio",
        queue="default",
        job_type="tts_longform",
        payload=payload,
        owner_user_id=str(current_user.id),
        priority=5,
        max_retries=3,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"job_id": int(job.get("id")), "status": job.get("status", "queued")},
    )


@router.get(
    "/speech/jobs/{job_id}/artifacts",
    summary="List artifacts for a TTS job",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.speech", count_as="call")),
    ],
)
async def get_speech_job_artifacts(
    job_id: int,
    current_user: User = Depends(get_request_user),
):
    with CollectionsDatabase.for_user(user_id=str(current_user.id)) as cdb:
        rows, _total = cdb.list_output_artifacts(job_id=job_id, limit=200, offset=0)
    artifacts: list[dict[str, Any]] = []
    for row in rows:
        metadata = {}
        if row.metadata_json:
            try:
                metadata = json.loads(row.metadata_json)
            except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
                metadata = {}
        artifacts.append(
            {
                "output_id": row.id,
                "format": row.format,
                "type": row.type,
                "title": row.title,
                "download_url": f"/api/v1/outputs/{row.id}/download",
                "metadata": metadata,
            }
        )
    return {"job_id": job_id, "artifacts": artifacts}


async def get_tts_service() -> TTSServiceV2:
    """Get the V2 TTS service instance."""
    return await get_tts_service_v2()


@router.post(
    "/speech",
    summary="Generates audio from text input.",
    response_class=Response,
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.speech", count_as="call")),
    ],
    responses={
        200: {
            "description": "Generated speech audio bytes or stream",
            "content": {
                "audio/aac": {},
                "audio/flac": {},
                "audio/L16": {},
                "audio/mpeg": {},
                "audio/opus": {},
                "audio/wav": {},
            },
            "headers": {
                "X-Audio-Sample-Rate": {
                    "description": "Resolved PCM sample rate in Hz (present when response_format=pcm).",
                    "schema": {"type": "string"},
                },
                "X-TTS-Alignment": {
                    "description": "Base64url-encoded JSON alignment payload when available (non-streaming).",
                    "schema": {"type": "string"},
                },
                "X-TTS-Alignment-Format": {
                    "description": "Alignment header encoding format (currently json+base64).",
                    "schema": {"type": "string"},
                },
            }
        }
    },
)
async def create_speech(
    request_data: OpenAISpeechRequest,
    request: Request,
    tts_service: TTSServiceV2 = Depends(get_tts_service),
    current_user: User = Depends(get_request_user),
    media_db: Optional[Any] = Depends(try_get_media_db_for_user),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
):
    """
    Generates audio from the input text.

    Requires authentication via Bearer token in Authorization header.
    Rate limited to 10 requests per minute per IP address.

    Input is sanitized by `TTSInputValidator`, which normalizes Unicode,
    strips HTML, and removes or rejects potentially dangerous patterns
    (e.g., obvious SQL/command/FS injection attempts). In strict mode
    (the default), such patterns result in HTTP 400 errors; in relaxed
    mode (`strict_validation` set to false via TTS config), dangerous
    substrings are stripped and the request is processed if non-empty.

    Docs: `Docs/Code_Documentation/Ingestion_Pipeline_Audio.md`,
    `Docs/STT-TTS/TTS-SETUP-GUIDE.md` (sanitization and error semantics).
    """
    request_id = ensure_request_id(request)

    provider_hint = _audio_shim_attr("_sanitize_speech_request")(request_data, request_id=request_id)
    history_cfg = _tts_history_config()
    history_enabled = bool(media_db) and history_cfg.get("enabled", False)
    history_written = False
    start_ts = time.monotonic()

    tts_provider_hint = provider_hint
    user_id_int, tts_overrides, byok_tts_resolution = await _audio_shim_attr("_resolve_tts_byok")(
        provider_hint=tts_provider_hint,
        current_user=current_user,
        request=request,
    )
    oauth_retry_attempted = False

    def _is_openai_oauth_request() -> bool:
        return (
            tts_provider_hint == "openai"
            and byok_tts_resolution is not None
            and getattr(byok_tts_resolution, "auth_source", None) == "oauth"
        )

    def _is_tts_auth_failure(exc: BaseException) -> bool:
        if isinstance(exc, TTSAuthenticationError):
            return True
        try:
            return int(getattr(exc, "status_code", 0) or 0) == status.HTTP_401_UNAUTHORIZED
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
            return False

    def _record_oauth_401_retry(outcome: str) -> None:
        try:
            log_counter(
                "byok_oauth_401_retry_total",
                labels={
                    "provider": "openai",
                    "outcome": outcome,
                },
            )
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
            pass
    logger.info(
        'Received speech request: model={}, voice={}, format={}, request_id={}',
        request_data.model,
        request_data.voice,
        request_data.response_format,
        request_id,
    )
    voice_to_voice_start: Optional[float] = None
    try:
        raw_v2v = request.headers.get("x-voice-to-voice-start") or request.headers.get("X-Voice-To-Voice-Start")
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
        raw_v2v = None
    if raw_v2v:
        try:
            ts = float(raw_v2v)
            if ts > 0:
                voice_to_voice_start = ts
        except (TypeError, ValueError):
            logger.debug("Invalid X-Voice-To-Voice-Start header")
    try:
        state_ts = getattr(request.state, "voice_to_voice_start", None)
        if voice_to_voice_start is None and isinstance(state_ts, (int, float)):
            voice_to_voice_start = float(state_ts)
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
        logger.debug("Failed to read voice_to_voice_start from request.state")
    try:
        usage_log.log_event(
            "audio.tts",
            tags=[str(request_data.model or ""), str(request_data.voice or "")],
            metadata={"stream": bool(getattr(request_data, "stream", False)), "format": request_data.response_format},
        )
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
        logger.debug("usage_log audio.tts failed")

    # Determine Content-Type
    content_type = _AUDIO_CONTENT_TYPE_MAP.get(request_data.response_format)
    if not content_type:
        logger.warning(f"Unsupported response format: {request_data.response_format}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported response_format: "
                f"{request_data.response_format}. Supported formats are: {', '.join(_AUDIO_CONTENT_TYPE_MAP.keys())}"
            ),
        )
    if request_data.stream and request_data.return_download_link:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="return_download_link requires stream=false",
        )

    def _record_tts_history(
        status: str,
        *,
        error_message: str | None = None,
        artifact_ids: list[Any] | None = None,
        output_id: int | None = None,
    ) -> None:
        nonlocal history_written
        if history_written or not history_enabled:
            return
        if status == "failed" and not history_cfg.get("store_failed", True):
            return
        if media_db is None:
            return
        try:
            text_hash = compute_tts_history_text_hash(request_data.input, history_cfg.get("hash_key"))
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
            logger.debug("TTS history: failed to compute text hash")
            return
        text_length = tts_history_text_length(request_data.input)
        text_value = request_data.input if history_cfg.get("store_text", True) else None
        metadata = _extract_tts_metadata(request_data)
        provider = metadata.get("provider") or tts_provider_hint
        model = metadata.get("model") or request_data.model
        voice_name = metadata.get("voice") or request_data.voice
        voice_id = metadata.get("voice_id")
        fmt = metadata.get("format") or request_data.response_format

        duration_ms = None
        duration_val = metadata.get("duration_ms")
        if isinstance(duration_val, (int, float)):
            duration_ms = int(duration_val)
        else:
            duration_seconds = metadata.get("duration_seconds") or metadata.get("duration")
            if isinstance(duration_seconds, (int, float)):
                duration_ms = int(float(duration_seconds) * 1000)

        params_json: dict[str, Any] = {"speed": request_data.speed}
        if request_data.target_sample_rate is not None:
            params_json["target_sample_rate"] = request_data.target_sample_rate
        if request_data.extra_params:
            try:
                extra_params = dict(request_data.extra_params)
            except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
                extra_params = None
            if extra_params:
                extra_params.pop("voice_reference", None)
                params_json["extra_params"] = extra_params
        if request_data.lang_code:
            params_json["lang_code"] = request_data.lang_code
        if request_data.normalization_options is not None:
            with contextlib.suppress(_AUDIO_TTS_NONCRITICAL_EXCEPTIONS):
                params_json["normalization_options"] = model_dump_compat(request_data.normalization_options)

        voice_info: dict[str, Any] | None = {}
        meta_voice_info = metadata.get("voice_info")
        if isinstance(meta_voice_info, dict):
            voice_info.update(meta_voice_info)
        if voice_info is not None:
            voice_info.pop("voice_reference", None)
        if request_data.voice_reference and voice_info is not None:
            voice_info["has_voice_reference"] = True
        if request_data.reference_duration_min is not None and voice_info is not None:
            voice_info["reference_duration_min"] = request_data.reference_duration_min
        if not voice_info:
            voice_info = None

        segments_json = build_tts_segments_payload(metadata.get("segments"))
        generation_time_ms = int(max(0.0, (time.monotonic() - start_ts) * 1000))

        final_status = status
        if final_status == "success" and parse_bool(metadata.get("partial"), default=False):
            final_status = "partial"

        try:
            insert_start = time.monotonic()
            media_db.create_tts_history_entry(
                user_id=str(current_user.id),
                text_hash=text_hash,
                text=text_value,
                text_length=text_length,
                provider=str(provider) if provider is not None else None,
                model=str(model) if model is not None else None,
                voice_id=str(voice_id) if voice_id is not None else None,
                voice_name=str(voice_name) if voice_name is not None else None,
                voice_info=voice_info,
                format=str(fmt) if fmt is not None else None,
                duration_ms=duration_ms,
                generation_time_ms=generation_time_ms,
                params_json=params_json if params_json else None,
                status=final_status,
                segments_json=segments_json,
                output_id=output_id,
                artifact_ids=artifact_ids,
                error_message=error_message,
            )
            try:
                log_counter(
                    "tts_history_writes_total",
                    labels={
                        "status": str(final_status or "unknown"),
                        "provider": str(provider or "unknown"),
                    },
                )
                log_histogram(
                    "tts_history_write_latency_ms",
                    value=max(0.0, (time.monotonic() - insert_start) * 1000),
                    labels={"status": str(final_status or "unknown")},
                )
            except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
                pass
            history_written = True
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("TTS history: failed to write record (request_id={}): {}", request_id, e)

    def _build_speech_iter():
        return tts_service.generate_speech(
            request_data,
            provider=tts_provider_hint,
            fallback=True,
            provider_overrides=tts_overrides,
            voice_to_voice_start=voice_to_voice_start,
            voice_to_voice_route="audio.speech",
            user_id=user_id_int,
            request_id=request_id,
        )

    speech_iter = None

    async def _refresh_openai_oauth_and_rebuild_iter() -> None:
        nonlocal user_id_int, tts_overrides, byok_tts_resolution, speech_iter
        try:
            user_id_int, tts_overrides, byok_tts_resolution = await _audio_shim_attr("_resolve_tts_byok")(
                provider_hint=tts_provider_hint,
                current_user=current_user,
                request=request,
                force_oauth_refresh=True,
            )
        except HTTPException:
            raise
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
            raise

        if byok_tts_resolution is None:
            raise TTSAuthenticationError("OpenAI OAuth refresh did not return credentials")

        api_key = (tts_overrides or {}).get("api_key") if isinstance(tts_overrides, dict) else None
        if not api_key:
            api_key = getattr(byok_tts_resolution, "api_key", None)
        if not isinstance(api_key, str) or not api_key.strip():
            raise TTSAuthenticationError("OpenAI OAuth refresh returned no access token")

        speech_iter = _build_speech_iter()

    try:
        speech_iter = _build_speech_iter()
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_for_tts_error(exc, request_id)

    async def _pull_first_chunk() -> bytes:
        nonlocal oauth_retry_attempted
        try:
            return await speech_iter.__anext__()
        except StopAsyncIteration:
            return b""
        except HTTPException:
            raise
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as exc:
            if (
                not oauth_retry_attempted
                and _is_openai_oauth_request()
                and _is_tts_auth_failure(exc)
            ):
                oauth_retry_attempted = True
                try:
                    await _refresh_openai_oauth_and_rebuild_iter()
                except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
                    _record_oauth_401_retry("refresh_failed")
                    _raise_for_tts_error(exc, request_id)
                except Exception:
                    _record_oauth_401_retry("refresh_failed")
                    _raise_for_tts_error(exc, request_id)
                try:
                    next_chunk = await speech_iter.__anext__()
                    _record_oauth_401_retry("success")
                    return next_chunk
                except StopAsyncIteration:
                    _record_oauth_401_retry("success")
                    return b""
                except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as retry_exc:
                    if _is_tts_auth_failure(retry_exc):
                        _record_oauth_401_retry("retry_auth_failed")
                        _raise_for_tts_error(exc, request_id)
                    _record_oauth_401_retry("retry_failed")
                    _raise_for_tts_error(retry_exc, request_id)
                except Exception as retry_exc:  # pragma: no cover - defensive fallback
                    if _is_tts_auth_failure(retry_exc):
                        _record_oauth_401_retry("retry_auth_failed")
                        _raise_for_tts_error(exc, request_id)
                    _record_oauth_401_retry("retry_failed")
                    _raise_for_tts_error(retry_exc, request_id)
            _raise_for_tts_error(exc, request_id)
        except Exception as exc:  # pragma: no cover - defensive fallback
            _raise_for_tts_error(exc, request_id)

    async def _stream_chunks(initial_chunk: bytes):
        stream_failed: Optional[str] = None
        stream_partial = False
        stream_completed = False
        try:
            if initial_chunk:
                if await request.is_disconnected():
                    logger.info("Client disconnected before streaming could start.")
                    stream_partial = True
                    return
                yield initial_chunk
            async for chunk in speech_iter:
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping audio generation.")
                    stream_partial = True
                    break
                yield chunk
            if not stream_partial:
                stream_completed = True
        except HTTPException as exc:
            stream_failed = _tts_history_error_message(exc)
            raise
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as exc:
            stream_failed = _tts_history_error_message(exc)
            _raise_for_tts_error(exc, request_id)
        except Exception as exc:  # pragma: no cover - defensive fallback
            stream_failed = _tts_history_error_message(exc)
            _raise_for_tts_error(exc, request_id)
        finally:
            if byok_tts_resolution is not None:
                try:
                    await byok_tts_resolution.touch_last_used()
                except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
                    logger.debug("Failed to update BYOK last_used timestamp")
            status = "success"
            error_message = None
            if stream_failed:
                status = "failed"
                error_message = stream_failed
            elif stream_partial:
                status = "partial"
                error_message = "client_disconnected"
            elif not stream_completed:
                status = "partial"
            _record_tts_history(status, error_message=error_message)

    if request_data.stream:
        try:
            first_chunk = await _pull_first_chunk()
        except HTTPException as exc:
            _record_tts_history("failed", error_message=_tts_history_error_message(exc))
            raise
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as exc:
            _record_tts_history("failed", error_message=_tts_history_error_message(exc))
            raise
        if not first_chunk:
            logger.error("Streaming generation resulted in empty audio data.")
            _record_tts_history("failed", error_message="empty_audio")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Audio generation failed to produce data.",
            )
        response_content_type = _resolve_response_content_type(request_data)
        stream_headers = {
            "Content-Disposition": f"attachment; filename=speech.{request_data.response_format}",
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "X-Request-Id": request_id,
        }
        _append_pcm_response_headers(request_data, stream_headers)
        return StreamingResponse(
            _stream_chunks(first_chunk),
            media_type=response_content_type,
            headers=stream_headers,
        )
    # Non-streaming mode: accumulate chunks and return a single response
    try:
        first_chunk = await _pull_first_chunk()
    except HTTPException as exc:
        _record_tts_history("failed", error_message=_tts_history_error_message(exc))
        raise
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as exc:
        _record_tts_history("failed", error_message=_tts_history_error_message(exc))
        raise
    all_audio_bytes = b""
    if first_chunk:
        all_audio_bytes += first_chunk
    try:
        async for chunk in speech_iter:
            all_audio_bytes += chunk
    except HTTPException as exc:
        _record_tts_history("failed", error_message=_tts_history_error_message(exc))
        raise
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as exc:
        _record_tts_history("failed", error_message=_tts_history_error_message(exc))
        _raise_for_tts_error(exc, request_id)

    # Drop any internal boundary markers if present
    all_audio_bytes = all_audio_bytes.replace(b"--final_boundary_for_non_streamed--", b"")

    if not all_audio_bytes:
        logger.error("Non-streaming generation resulted in empty audio data.")
        _record_tts_history("failed", error_message="empty_audio")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Audio generation failed to produce data."
        )

    if byok_tts_resolution is not None:
        try:
            await byok_tts_resolution.touch_last_used()
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
            logger.debug("Failed to update BYOK last_used timestamp")

    headers = {
        "Content-Disposition": f"attachment; filename=speech.{request_data.response_format}",
        "Cache-Control": "no-cache",
        "X-Request-Id": request_id,
    }
    _append_pcm_response_headers(request_data, headers)
    try:
        metadata = getattr(request_data, "_tts_metadata", None)
        alignment_payload = metadata.get("alignment") if isinstance(metadata, dict) else None
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
        alignment_payload = None
    if alignment_payload:
        try:
            alignment_json = json.dumps(alignment_payload, separators=(",", ":"), ensure_ascii=True)
            alignment_b64 = base64.urlsafe_b64encode(alignment_json.encode("utf-8")).decode("ascii")
            headers["X-TTS-Alignment"] = alignment_b64
            headers["X-TTS-Alignment-Format"] = "json+base64"
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
            logger.debug("Failed to encode alignment metadata header")

    output_id: int | None = None
    artifact_ids: list[Any] | None = None
    if request_data.return_download_link:
        try:
            file_record = await _audio_shim_attr("save_and_register_tts_audio")(
                user_id=user_id_int,
                audio_bytes=all_audio_bytes,
                audio_format=request_data.response_format,
                original_text=request_data.input,
                voice_name=request_data.voice,
                model_name=request_data.model,
                check_quota=True,
            )
            file_id = file_record.get("id")
            if file_id:
                headers["X-Download-Path"] = f"/api/v1/storage/files/{file_id}/download"
                headers["X-Generated-File-Id"] = str(file_id)
                artifact_ids = [file_id]
        except QuotaExceededError as exc:
            _record_tts_history("failed", error_message=_tts_history_error_message(exc))
            raise HTTPException(
                status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                detail=str(exc),
            ) from exc
        except StorageError as exc:
            _record_tts_history("failed", error_message=_tts_history_error_message(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store generated speech audio",
            ) from exc
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as exc:
            _record_tts_history("failed", error_message=_tts_history_error_message(exc))
            raise

    _record_tts_history("success", artifact_ids=artifact_ids, output_id=output_id)

    return Response(
        content=all_audio_bytes,
        media_type=_resolve_response_content_type(request_data),
        headers=headers,
    )


@router.post(
    "/speech/metadata",
    summary="Returns alignment metadata for a TTS request.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.speech", count_as="call")),
    ],
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "alignment": {
                            "engine": "kokoro",
                            "sample_rate": 24000,
                            "words": [
                                {"word": "Hello", "start_ms": 0, "end_ms": 400},
                                {"word": "world", "start_ms": 450, "end_ms": 900},
                            ],
                        }
                    }
                }
            }
        },
        204: {"description": "No alignment metadata available for this request."},
    },
)
async def create_speech_metadata(
    request_data: OpenAISpeechRequest,
    request: Request,
    tts_service: TTSServiceV2 = Depends(get_tts_service),
    current_user: User = Depends(get_request_user),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
):
    request_id = ensure_request_id(request)
    provider_hint = _audio_shim_attr("_sanitize_speech_request")(request_data, request_id=request_id)
    tts_provider_hint = provider_hint
    user_id_int, tts_overrides, byok_tts_resolution = await _audio_shim_attr("_resolve_tts_byok")(
        provider_hint=tts_provider_hint,
        current_user=current_user,
        request=request,
    )
    oauth_retry_attempted = False

    def _is_openai_oauth_request() -> bool:
        return (
            tts_provider_hint == "openai"
            and byok_tts_resolution is not None
            and getattr(byok_tts_resolution, "auth_source", None) == "oauth"
        )

    def _is_tts_auth_failure(exc: BaseException) -> bool:
        if isinstance(exc, TTSAuthenticationError):
            return True
        try:
            return int(getattr(exc, "status_code", 0) or 0) == status.HTTP_401_UNAUTHORIZED
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
            return False

    def _record_oauth_401_retry(outcome: str) -> None:
        try:
            log_counter(
                "byok_oauth_401_retry_total",
                labels={
                    "provider": "openai",
                    "outcome": outcome,
                },
            )
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
            pass

    try:
        usage_log.log_event(
            "audio.tts.metadata",
            tags=[str(request_data.model or ""), str(request_data.voice or "")],
            metadata={"stream": bool(getattr(request_data, "stream", False)), "format": request_data.response_format},
        )
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
        logger.debug("usage_log audio.tts.metadata failed")

    if hasattr(request_data, "stream"):
        try:
            request_data.stream = False
        except (AttributeError, TypeError) as exc:
            logger.warning(
                "audio.speech.metadata: failed to set request_data.stream=False (model={}, request_id={}): {}",
                getattr(request_data, "model", None),
                request_id,
                exc,
            )
    else:
        logger.warning(
            "audio.speech.metadata: request_data missing stream attribute (model={}, request_id={})",
            getattr(request_data, "model", None),
            request_id,
        )

    def _build_speech_iter():
        return tts_service.generate_speech(
            request_data,
            provider=tts_provider_hint,
            fallback=True,
            provider_overrides=tts_overrides,
            voice_to_voice_route="audio.speech.metadata",
            user_id=user_id_int,
            metadata_only=True,
            request_id=request_id,
        )

    speech_iter = None

    async def _refresh_openai_oauth_and_rebuild_iter() -> None:
        nonlocal user_id_int, tts_overrides, byok_tts_resolution, speech_iter
        try:
            user_id_int, tts_overrides, byok_tts_resolution = await _audio_shim_attr("_resolve_tts_byok")(
                provider_hint=tts_provider_hint,
                current_user=current_user,
                request=request,
                force_oauth_refresh=True,
            )
        except HTTPException:
            raise
        except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
            raise

        if byok_tts_resolution is None:
            raise TTSAuthenticationError("OpenAI OAuth refresh did not return credentials")

        api_key = (tts_overrides or {}).get("api_key") if isinstance(tts_overrides, dict) else None
        if not api_key:
            api_key = getattr(byok_tts_resolution, "api_key", None)
        if not isinstance(api_key, str) or not api_key.strip():
            raise TTSAuthenticationError("OpenAI OAuth refresh returned no access token")

        speech_iter = _build_speech_iter()

    try:
        speech_iter = _build_speech_iter()
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_for_tts_error(exc, request_id)

    try:
        with contextlib.suppress(StopAsyncIteration):
            await speech_iter.__anext__()
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as exc:
        if (
            not oauth_retry_attempted
            and _is_openai_oauth_request()
            and _is_tts_auth_failure(exc)
        ):
            oauth_retry_attempted = True
            try:
                await _refresh_openai_oauth_and_rebuild_iter()
            except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
                _record_oauth_401_retry("refresh_failed")
                _raise_for_tts_error(exc, request_id)
            except Exception:
                _record_oauth_401_retry("refresh_failed")
                _raise_for_tts_error(exc, request_id)
            try:
                with contextlib.suppress(StopAsyncIteration):
                    await speech_iter.__anext__()
            except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as retry_exc:
                if _is_tts_auth_failure(retry_exc):
                    _record_oauth_401_retry("retry_auth_failed")
                    _raise_for_tts_error(exc, request_id)
                _record_oauth_401_retry("retry_failed")
                _raise_for_tts_error(retry_exc, request_id)
            _record_oauth_401_retry("success")
        else:
            _raise_for_tts_error(exc, request_id)
    finally:
        if byok_tts_resolution is not None:
            try:
                await byok_tts_resolution.touch_last_used()
            except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS:
                logger.debug("Failed to update BYOK last_used timestamp")

    metadata = getattr(request_data, "_tts_metadata", None)
    alignment_payload = None
    if isinstance(metadata, dict):
        alignment_payload = metadata.get("alignment")
    if not alignment_payload:
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"X-Request-Id": request_id})
    return JSONResponse(content={"alignment": alignment_payload}, headers={"X-Request-Id": request_id})


@router.get("/providers")
async def list_tts_providers(request: Request, tts_service: TTSServiceV2 = Depends(get_tts_service)):
    """
    List all available TTS providers and their capabilities.
    """
    from datetime import datetime

    try:
        capabilities = await tts_service.get_capabilities()
        voices = await tts_service.list_voices()

        return {"providers": capabilities, "voices": voices, "timestamp": datetime.utcnow().isoformat()}
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error listing TTS providers")
        request_id = ensure_request_id(request)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("Failed to list providers", request_id, exc=e),
        ) from e


@router.get(
    "/tts/providers/{provider}/model-info",
    summary="Get focused TTS provider model information",
)
async def get_tts_provider_model_info(
    provider: str,
    request: Request,
    tts_service: TTSServiceV2 = Depends(get_tts_service),
):
    """Return focused status and capability metadata for one TTS provider."""
    request_id = ensure_request_id(request)
    provider_key = (provider or "").strip().lower()
    try:
        status_data = tts_service.get_status()
        capabilities = await tts_service.get_capabilities()
        status_providers = status_data.get("providers", {}) if isinstance(status_data, dict) else {}
        status_key, provider_status = _find_provider_entry(status_providers, provider_key)
        caps_key, provider_caps = _find_provider_entry(capabilities, provider_key)
        resolved_provider = (status_key or caps_key or provider_key).lower()
        if provider_status is None and provider_caps is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=_http_error_detail(f"Unknown TTS provider '{provider}'", request_id),
            )
        return _build_provider_model_info(
            provider=resolved_provider,
            provider_status=provider_status,
            provider_capabilities=provider_caps,
        )
    except HTTPException:
        raise
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(
            request_id=request_id,
            provider=provider_key or "<missing>",
            error_type=type(e).__name__,
        ).opt(exception=e).error("Error getting TTS provider model info")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("Failed to get provider model info", request_id, exc=e),
        ) from e


@router.post(
    "/tts/providers/{provider}/unload",
    summary="Unload a cached TTS provider runtime",
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.speech", count_as="call")),
    ],
)
async def unload_tts_provider(
    provider: str,
    request: Request,
    tts_service: TTSServiceV2 = Depends(get_tts_service),
    _current_user: User = Depends(get_request_user),
):
    """Release one cached TTS provider runtime so it can be reloaded on demand."""
    request_id = ensure_request_id(request)
    try:
        result = await tts_service.unload_provider(provider)
        return JSONResponse(content=result, headers={"X-Request-Id": request_id})
    except TTSProviderNotConfiguredError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_http_error_detail(f"Unknown TTS provider '{provider}'", request_id, exc=e),
        ) from e
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(
            request_id=request_id,
            provider=(provider or "").strip().lower() or "<missing>",
            error_type=type(e).__name__,
        ).opt(exception=e).error("Error unloading TTS provider")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("Failed to unload TTS provider", request_id, exc=e),
        ) from e


@router.get("/voices/catalog", summary="List available TTS voices across providers")
async def list_tts_voices(
    request: Request,
    provider: Optional[str] = None,
    catalog_format: Optional[str] = Query(default=None, alias="format"),
    tts_service: TTSServiceV2 = Depends(get_tts_service),
):
    """
    List available voices from TTS providers.

    - If `provider` is specified, returns voices only for that provider.
    - Otherwise returns a mapping of provider name to voice lists.
    - If `format=openai`, returns a flattened OpenAI-style list object.
    """
    try:
        all_voices = await tts_service.list_voices()
        if provider:
            key = provider.lower()
            if key in all_voices:
                provider_voices = {key: all_voices[key]}
                if (catalog_format or "").strip().lower() == "openai":
                    return _build_openai_voice_catalog(provider_voices)
                return provider_voices
            raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found or unavailable")
        if (catalog_format or "").strip().lower() == "openai":
            return _build_openai_voice_catalog(all_voices)
        return all_voices
    except HTTPException:
        raise
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error listing TTS voices")
        request_id = ensure_request_id(request)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("Failed to list voices", request_id, exc=e),
        ) from e


@router.post("/reset-metrics")
async def reset_tts_metrics(
    request: Request,
    provider: Optional[str] = None,
    tts_service: TTSServiceV2 = Depends(get_tts_service),
):
    """
    Reset TTS metrics.

    Args:
        provider: Optional provider name to reset metrics for. If not provided, resets all metrics.
        tts_service: TTS service instance (accesses tts_service.metrics when available).

    Raises:
        HTTPException: 501 when the requested metrics reset API is not implemented.
    """
    async def _call_reset(method, *args) -> None:
        result = method(*args)
        if inspect.isawaitable(result):
            await result

    def _can_call_with_provider(method) -> bool:
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return True
        params = list(sig.parameters.values())
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
            return True
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return len(positional) >= 1

    def _can_call_without_args(method) -> bool:
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return True
        required = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]
        return len(required) == 0

    try:
        metrics = getattr(tts_service, "metrics", None)
        if provider:
            provider_methods: list[tuple[str, Any]] = []
            if hasattr(tts_service, "reset_metrics"):
                provider_methods.append(("tts_service.reset_metrics", tts_service.reset_metrics))
            if metrics is not None:
                for name in ("reset", "clear"):
                    if hasattr(metrics, name):
                        provider_methods.append((f"tts_service.metrics.{name}", getattr(metrics, name)))

            for name, method in provider_methods:
                if _can_call_with_provider(method):
                    await _call_reset(method, provider)
                    logger.info("reset_tts_metrics: reset metrics for provider={} via {}", provider, name)
                    return {"message": f"Metrics reset for provider {provider}"}

            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Provider-specific TTS metrics reset not supported",
            )

        global_methods: list[tuple[str, Any]] = []
        if hasattr(tts_service, "reset_metrics"):
            global_methods.append(("tts_service.reset_metrics", tts_service.reset_metrics))
        if metrics is not None:
            for name in ("reset_all", "clear_all", "reset", "clear"):
                if hasattr(metrics, name):
                    global_methods.append((f"tts_service.metrics.{name}", getattr(metrics, name)))

        for name, method in global_methods:
            if _can_call_without_args(method):
                await _call_reset(method)
                logger.info("reset_tts_metrics: reset all TTS metrics via {}", name)
                return {"message": "All TTS metrics reset"}

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="TTS metrics reset API is not available on this service",
        )
    except HTTPException:
        raise
    except _AUDIO_TTS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error resetting metrics")
        request_id = ensure_request_id(request)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("Failed to reset metrics", request_id, exc=e),
        ) from e
