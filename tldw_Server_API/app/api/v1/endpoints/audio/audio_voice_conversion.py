"""Chatterbox voice-conversion upload endpoint.

This module exposes the OpenAI-adjacent voice conversion route for local
Chatterbox VC runtimes. It is deliberately narrow: uploads are bounded and
validated before temporary materialization, stored custom voices are resolved
through the voice manager, and temporary files are kept alive for streaming
responses until the stream has been consumed.
"""

import contextlib
import tempfile
from pathlib import Path
from typing import AsyncIterable, AsyncIterator, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from starlette import status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, TokenScopeGuard, User
from tldw_Server_API.app.api.v1.endpoints.audio.audio_tts import _AUDIO_CONTENT_TYPE_MAP, get_tts_service
from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSError,
    TTSInvalidVoiceReferenceError,
    TTSProviderNotConfiguredError,
    TTSValidationError,
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2

router = APIRouter(
    tags=["Audio"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)

VOICE_CONVERSION_SCOPE = "audio.voice_conversion"
_ALLOWED_UPLOAD_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".pcm",
    ".wav",
    ".webm",
}
_UPLOAD_CONTENT_TYPE_SUFFIXES = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/l16": ".pcm",
    "audio/m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/pcm": ".pcm",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
}
_MAX_VOICE_CONVERSION_UPLOAD_BYTES = 50 * 1024 * 1024
_VOICE_CONVERSION_UPLOAD_CHUNK_BYTES = 1024 * 1024
_VOICE_CONVERSION_NONCRITICAL_EXCEPTIONS = (
    OSError,
    TypeError,
    ValueError,
    TTSError,
)


def _upload_label(prefix: str) -> str:
    """Return a stable human-readable label for one upload role."""
    return prefix.rstrip("_")


def _safe_upload_filename(filename: Optional[str]) -> str:
    """Return a sanitized upload filename for diagnostic logging only."""
    value = str(filename or "").strip()
    if not value:
        return "<missing>"
    with contextlib.suppress(OSError, RuntimeError, ValueError):
        name = Path(value).name
        if name:
            return name[:128]
    return "<invalid>"


def _resolve_audio_suffix(filename: Optional[str], content_type: Optional[str] = None) -> str:
    """Validate and return a temporary suffix for an uploaded audio file."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in _ALLOWED_UPLOAD_SUFFIXES:
        return suffix

    if suffix:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio upload extension: {suffix}",
        )

    normalized_content_type = str(content_type or "").split(";", 1)[0].strip().lower()
    inferred_suffix = _UPLOAD_CONTENT_TYPE_SUFFIXES.get(normalized_content_type)
    if inferred_suffix is not None:
        return inferred_suffix

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Audio upload must include a supported filename extension or audio content type",
    )


async def _materialize_upload(upload: UploadFile, *, prefix: str, request_id: str) -> str:
    """Persist one upload to a temporary file and return its path."""
    upload_role = _upload_label(prefix)
    empty_detail = f"{upload_role} audio upload is empty"
    tmp_path: Optional[str] = None
    total_bytes = 0
    wrote_payload = False
    try:
        suffix = _resolve_audio_suffix(upload.filename, upload.content_type)
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            delete=False,
        ) as tmp_file:
            tmp_path = tmp_file.name
            while True:
                chunk = await upload.read(_VOICE_CONVERSION_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > _MAX_VOICE_CONVERSION_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"{upload_role} audio upload exceeds "
                            f"{_MAX_VOICE_CONVERSION_UPLOAD_BYTES} bytes"
                        ),
                    )
                tmp_file.write(chunk)
                wrote_payload = True
    except HTTPException:
        if tmp_path:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)
        raise
    except _VOICE_CONVERSION_NONCRITICAL_EXCEPTIONS as exc:
        if tmp_path:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)
        logger.bind(
            request_id=request_id,
            upload_role=upload_role,
            filename=_safe_upload_filename(upload.filename),
            content_type=upload.content_type or "<missing>",
            bytes_read=total_bytes,
            error_type=type(exc).__name__,
        ).opt(exception=exc).error("Failed to materialize voice conversion upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("Failed to prepare voice conversion upload", request_id, exc=exc),
        ) from exc

    if not wrote_payload:
        if tmp_path:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=empty_detail,
        )
    return tmp_path


def _materialize_audio_bytes(
    payload: bytes,
    *,
    prefix: str,
    filename: Optional[str],
    empty_detail: str,
    request_id: str,
) -> str:
    """Persist audio bytes to a temporary file and return its path."""
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=empty_detail,
        )
    if len(payload) > _MAX_VOICE_CONVERSION_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{prefix.rstrip('_')} audio payload exceeds {_MAX_VOICE_CONVERSION_UPLOAD_BYTES} bytes",
        )

    tmp_path: Optional[str] = None
    try:
        suffix = _resolve_audio_suffix(filename, "audio/wav")
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            delete=False,
        ) as tmp_file:
            tmp_file.write(payload)
            tmp_path = tmp_file.name
    except _VOICE_CONVERSION_NONCRITICAL_EXCEPTIONS as exc:
        if tmp_path:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)
        logger.bind(
            request_id=request_id,
            upload_role=_upload_label(prefix),
            filename=_safe_upload_filename(filename),
            bytes_read=len(payload),
            error_type=type(exc).__name__,
        ).opt(exception=exc).error("Failed to materialize voice conversion audio bytes")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("Failed to prepare voice conversion upload", request_id, exc=exc),
        ) from exc
    return tmp_path


async def _materialize_stored_target_voice(user_id: int, voice_id: str, *, request_id: str) -> str:
    """Resolve and materialize a stored custom voice reference for Chatterbox VC."""
    resolved_voice_id = voice_id.strip()
    if not resolved_voice_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="target_voice_id must not be empty",
        )

    try:
        from tldw_Server_API.app.core.TTS.voice_manager import VoiceProcessingError, get_voice_manager
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Stored custom voices are not available in this build",
        ) from exc

    try:
        voice_manager = get_voice_manager()
        payload = await voice_manager.load_voice_reference_audio(user_id, resolved_voice_id)
    except VoiceProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_http_error_detail("Stored target voice reference is not available", request_id, exc=exc),
        ) from exc

    return _materialize_audio_bytes(
        payload,
        prefix="chatterbox_vc_target_",
        filename=f"{resolved_voice_id}.wav",
        empty_detail="Stored target voice reference audio is empty",
        request_id=request_id,
    )


def _voice_conversion_content_type(response_format: str, sample_rate: Optional[int]) -> str:
    if response_format == "pcm":
        resolved_rate = sample_rate if sample_rate and sample_rate > 0 else 24000
        return f"audio/L16; rate={resolved_rate}; channels=1"
    return _AUDIO_CONTENT_TYPE_MAP.get(response_format, "audio/wav")


def _voice_conversion_headers(response_format: str, sample_rate: Optional[int], request_id: str) -> dict[str, str]:
    headers = {
        "Content-Disposition": f"attachment; filename=voice-conversion.{response_format}",
        "Cache-Control": "no-cache",
        "X-Request-Id": request_id,
    }
    if response_format == "pcm":
        resolved_rate = sample_rate if sample_rate and sample_rate > 0 else 24000
        headers["X-Audio-Sample-Rate"] = str(resolved_rate)
    return headers


def _cleanup_upload_paths(paths: tuple[Optional[str], ...]) -> None:
    """Remove temporary upload files if they were materialized."""
    for raw_path in paths:
        if raw_path:
            with contextlib.suppress(OSError):
                Path(raw_path).unlink(missing_ok=True)


async def _stream_with_upload_cleanup(
    stream: AsyncIterable[bytes],
    paths: tuple[Optional[str], ...],
) -> AsyncIterator[bytes]:
    """Yield a stream while deferring temporary upload cleanup until consumption ends."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        _cleanup_upload_paths(paths)


@router.post(
    "/voice-conversion",
    summary="Convert source speech into a target voice with Chatterbox VC.",
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard(
                "any",
                require_if_present=True,
                endpoint_id=VOICE_CONVERSION_SCOPE,
                count_as="call",
            )
        ),
    ],
)
async def create_voice_conversion(
    request: Request,
    source_audio: UploadFile = File(..., description="Source speech audio to convert."),
    target_voice: Optional[UploadFile] = File(
        default=None,
        description="Optional target voice reference audio. If omitted, Chatterbox uses its built-in reference.",
    ),
    target_voice_id: Optional[str] = Form(
        default=None,
        description="Optional stored custom voice ID to use as the target reference.",
    ),
    response_format: str = Form(default="wav", description="Output audio format."),
    stream: bool = Form(default=False, description="Return a streaming audio response."),
    tts_service: TTSServiceV2 = Depends(get_tts_service),
    current_user: User = Depends(get_request_user),
):
    """Convert uploaded speech into a target Chatterbox voice."""
    request_id = ensure_request_id(request)

    resolved_format = str(response_format or "wav").strip().lower()
    if resolved_format not in _AUDIO_CONTENT_TYPE_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported response_format: "
                f"{response_format}. Supported formats are: {', '.join(_AUDIO_CONTENT_TYPE_MAP.keys())}"
            ),
        )

    resolved_target_voice_id = str(target_voice_id or "").strip()
    if target_voice is not None and resolved_target_voice_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either target_voice or target_voice_id, not both.",
        )

    source_path: Optional[str] = None
    target_path: Optional[str] = None
    try:
        source_path = await _materialize_upload(
            source_audio,
            prefix="chatterbox_vc_source_",
            request_id=request_id,
        )
        if target_voice is not None:
            target_path = await _materialize_upload(
                target_voice,
                prefix="chatterbox_vc_target_",
                request_id=request_id,
            )
        elif resolved_target_voice_id:
            target_path = await _materialize_stored_target_voice(
                current_user.id,
                resolved_target_voice_id,
                request_id=request_id,
            )

        response = await tts_service.convert_chatterbox_voice(
            source_audio_path=source_path,
            target_voice_path=target_path,
            response_format=resolved_format,
            stream=stream,
        )
        sample_rate = response.sample_rate or 24000
        content_type = _voice_conversion_content_type(resolved_format, sample_rate)
        headers = _voice_conversion_headers(resolved_format, sample_rate, request_id)

        if response.audio_stream is not None:
            stream_with_cleanup = _stream_with_upload_cleanup(
                response.audio_stream,
                (source_path, target_path),
            )
            source_path = None
            target_path = None
            return StreamingResponse(stream_with_cleanup, media_type=content_type, headers=headers)

        audio_bytes = response.audio_data or response.audio_content
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Voice conversion failed to produce audio.",
            )
        return Response(content=audio_bytes, media_type=content_type, headers=headers)
    except (TTSInvalidVoiceReferenceError, TTSValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_http_error_detail("Invalid voice conversion request", request_id, exc=exc),
        ) from exc
    except TTSProviderNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("Chatterbox voice conversion provider is not configured", request_id, exc=exc),
        ) from exc
    except TTSError as exc:
        logger.bind(
            request_id=request_id,
            response_format=resolved_format,
            stream=stream,
            target_voice_id_present=bool(resolved_target_voice_id),
            error_type=type(exc).__name__,
        ).opt(exception=exc).error("Voice conversion failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("Voice conversion failed", request_id, exc=exc),
        ) from exc
    finally:
        _cleanup_upload_paths((source_path, target_path))
