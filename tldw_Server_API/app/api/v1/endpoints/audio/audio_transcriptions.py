# audio_transcriptions.py
# Description: Audio transcription, translation, and segmentation endpoints.
import asyncio
import contextlib
import json
import math
import mimetypes
import os
import re
import tempfile
import time
from pathlib import Path as PathLib
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from loguru import logger
import soundfile as sf
from starlette import status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal, get_db_transaction, get_request_user, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.API_Deps.billing_deps import get_billing_org_id, require_within_limit
from tldw_Server_API.app.core.Resource_Governance import cost_units
from tldw_Server_API.app.core.Billing.enforcement import (
    LimitCategory,
    enforcement_enabled,
    get_billing_enforcer,
)
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    TranscriptSegmentationRequest,
    TranscriptSegmentationResponse,
)
from tldw_Server_API.app.core.Audio.dictation_error_taxonomy import (
    build_dictation_error_payload,
)
from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.Audio.quota_helpers import EXPECTED_DB_EXC
from tldw_Server_API.app.core.Audio.transcription_service import _map_openai_audio_model_to_whisper
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_policy import (
    apply_transcript_text_policy,
    build_audio_retention_decision,
    redact_timed_segments,
    resolve_effective_stt_policy,
    retain_stt_audio_artifact,
)
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id
from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
from tldw_Server_API.app.core.Metrics.stt_metrics import (
    emit_stt_error_total,
    emit_stt_redaction_total,
    emit_stt_request_total,
    observe_stt_latency_seconds,
)
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Usage.audio_quota import (
    add_daily_minutes,
    can_start_job,
    check_daily_minutes_allow,
    finish_job,
    get_job_heartbeat_interval_seconds,
    get_limits_for_user,
    heartbeat_jobs,
    increment_jobs_started,
)

router = APIRouter(
    tags=["Audio"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)

_AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    EOFError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)
_OPENAI_AUDIO_TRANSCRIPT_RESPONSE_CONTENT = {
    "application/json": {},
    "application/x-subrip": {},
    "text/plain": {},
    "text/vtt": {},
}
_OPENAI_AUDIO_TRANSCRIPT_RESPONSES = {
    status.HTTP_200_OK: {
        "description": "OpenAI-compatible transcript response in JSON, plain text, SRT, or VTT format.",
        "content": _OPENAI_AUDIO_TRANSCRIPT_RESPONSE_CONTENT,
    },
    status.HTTP_402_PAYMENT_REQUIRED: {"description": "Billing limit exceeded. Upgrade plan to continue."},
}

_AUDIO_UPLOAD_SUFFIX_BY_CONTENT_TYPE: dict[str, str] = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/webm": ".webm",
}

_AUDIO_UPLOAD_SUFFIX_NORMALIZATION: dict[str, str] = {
    ".jpeg": ".jpg",
    ".mpga": ".mp3",
    ".oga": ".ogg",
    ".weba": ".webm",
    ".wave": ".wav",
}


def _sniff_audio_upload_suffix(sample: bytes) -> str:
    """Best-effort audio format sniffing for uploads without a filename suffix."""
    if len(sample) >= 12 and sample.startswith(b"RIFF") and sample[8:12] == b"WAVE":
        return ".wav"
    if sample.startswith(b"ID3") or (
        len(sample) >= 2 and sample[0] == 0xFF and (sample[1] & 0xE0) == 0xE0
    ):
        return ".mp3"
    if sample.startswith(b"fLaC"):
        return ".flac"
    if sample.startswith(b"OggS"):
        if b"OpusHead" in sample[:64]:
            return ".opus"
        return ".ogg"
    if len(sample) >= 12 and sample[4:8] == b"ftyp":
        return ".m4a"
    if sample.startswith(b"\x1A\x45\xDF\xA3"):
        return ".webm"
    return ""


def _resolve_audio_upload_suffix(
    filename: str | None,
    content_type: str | None,
    sample: bytes | None = None,
) -> str:
    """Return a usable temp-file suffix for staged audio uploads."""
    suffix = PathLib(filename or "").suffix.lower()
    if suffix:
        return _AUDIO_UPLOAD_SUFFIX_NORMALIZATION.get(suffix, suffix)

    sniffed_suffix = _sniff_audio_upload_suffix(sample or b"")
    if sniffed_suffix:
        return sniffed_suffix

    normalized_content_type = str(content_type or "").split(";")[0].strip().lower()
    if normalized_content_type in _AUDIO_UPLOAD_SUFFIX_BY_CONTENT_TYPE:
        return _AUDIO_UPLOAD_SUFFIX_BY_CONTENT_TYPE[normalized_content_type]

    guessed_suffix = mimetypes.guess_extension(normalized_content_type, strict=False) or ""
    guessed_suffix = guessed_suffix.lower()
    if guessed_suffix:
        return _AUDIO_UPLOAD_SUFFIX_NORMALIZATION.get(guessed_suffix, guessed_suffix)

    return ".audio"


def _audio_shim_attr(name: str):
    defaults: dict[str, Any] = {
        "check_daily_minutes_allow": check_daily_minutes_allow,
        "add_daily_minutes": add_daily_minutes,
        "get_job_heartbeat_interval_seconds": get_job_heartbeat_interval_seconds,
        "heartbeat_jobs": heartbeat_jobs,
        "get_limits_for_user": get_limits_for_user,
        "can_start_job": can_start_job,
        "increment_jobs_started": increment_jobs_started,
        "finish_job": finish_job,
        "sf": sf,
    }
    default_value = defaults.get(name)
    package_candidate: Any = None
    module_candidate: Any = None

    def _is_test_override(value: Any) -> bool:
        try:
            module_name = str(getattr(value, "__module__", "") or "")
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            module_name = ""
        if module_name.startswith("tldw_Server_API.tests") or module_name.startswith("tests."):
            return True
        if module_name == "__main__":
            return True
        try:
            class_module = str(getattr(getattr(value, "__class__", object), "__module__", "") or "")
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            class_module = ""
        return class_module.startswith("tldw_Server_API.tests") or class_module.startswith("tests.")

    try:
        from tldw_Server_API.app.api.v1.endpoints import audio as audio_pkg_shim

        if hasattr(audio_pkg_shim, name):
            package_candidate = getattr(audio_pkg_shim, name)
    except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
        logger.debug("audio_transcriptions shim package lookup failed")
    except Exception:
        logger.debug("audio_transcriptions shim package lookup raised unexpected error")
    try:
        from tldw_Server_API.app.api.v1.endpoints.audio import audio as audio_module_shim

        if hasattr(audio_module_shim, name):
            module_candidate = getattr(audio_module_shim, name)
    except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
        logger.debug("audio_transcriptions shim module lookup failed")
    except Exception:
        logger.debug("audio_transcriptions shim module lookup raised unexpected error")

    if package_candidate is not None and module_candidate is not None:
        if package_candidate is module_candidate:
            return package_candidate
        package_is_test_override = _is_test_override(package_candidate)
        module_is_test_override = _is_test_override(module_candidate)
        if package_is_test_override != module_is_test_override:
            return package_candidate if package_is_test_override else module_candidate
        if default_value is not None:
            package_changed = package_candidate is not default_value
            module_changed = module_candidate is not default_value
            if package_changed != module_changed:
                return package_candidate if package_changed else module_candidate
            if package_changed and module_changed:
                return package_candidate
        return module_candidate

    if package_candidate is not None:
        return package_candidate
    if module_candidate is not None:
        return module_candidate
    if name in defaults:
        return defaults[name]
    raise NameError(name)


async def _check_daily_minutes_allow(user_id: int, minutes: float):
    return await _audio_shim_attr("check_daily_minutes_allow")(user_id, minutes)


async def _add_daily_minutes(user_id: int, minutes: float):
    return await _audio_shim_attr("add_daily_minutes")(user_id, minutes)


def _stt_provider_envelope(stt_registry: Any, provider_name: str) -> Optional[dict[str, Any]]:
    """
    Best-effort lookup for standardized STT provider capability envelope.
    """
    try:
        entries = stt_registry.list_capabilities(include_disabled=True)
    except TypeError:
        try:
            entries = stt_registry.list_capabilities()
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            return None
    except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
        return None

    if not isinstance(entries, list):
        return None
    normalized_name = str(provider_name or "").strip().lower()
    if not normalized_name:
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = str(entry.get("provider") or "").strip().lower()
        if candidate == normalized_name:
            return entry
    return None


def _stt_provider_availability(
    stt_registry: Any,
    provider_name: str,
    envelope: Optional[dict[str, Any]] = None,
) -> str:
    """
    Resolve provider availability from envelope first, then legacy status API.
    """
    if isinstance(envelope, dict):
        availability = str(envelope.get("availability") or "").strip().lower()
        if availability:
            return availability

    get_status = getattr(stt_registry, "get_status", None)
    if callable(get_status):
        try:
            legacy_status = str(get_status(provider_name) or "").strip().lower()
            if legacy_status:
                return legacy_status
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            pass
    return "unknown"


def _normalize_timed_segments(raw_segments: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Normalize raw segment mappings into cleaned timed segment dictionaries.

    Args:
        raw_segments: Optional ``list[dict[str, Any]]`` containing provider
            segment payloads.

    Returns:
        list[dict[str, Any]]: Normalized segments shaped as
        ``{"text": str, "start_seconds": float, "end_seconds": float}``.
        ``"words"`` is included when the source segment provides it as a list.

    Notes:
        - Filters out entries that are not dictionaries or have empty ``text``.
        - Extracts ``text``, ``start_seconds``, and ``end_seconds`` with
          defensive float parsing fallbacks (``start`` defaults to ``0.0`` and
          ``end`` defaults to ``start`` when parsing fails or fields are empty).
        - Clamps ``end_seconds`` to ``start_seconds`` when ``end < start``.
        - Non-critical coercion errors are caught via
          ``_AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS``.
    """
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_segments, list):
        return normalized
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        text_raw = segment.get("text")
        if text_raw is None:
            text_raw = segment.get("Text")
        text = str(text_raw or "").strip()
        if not text:
            continue
        start_raw = segment.get("start_seconds", segment.get("start", 0.0))
        try:
            start = float(start_raw or 0.0)
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            start = 0.0
        end_raw = segment.get("end_seconds", segment.get("end", start))
        try:
            end = float(end_raw or start)
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            end = start
        if end < start:
            end = start
        norm_seg: dict[str, Any] = {
            "text": text,
            "start_seconds": start,
            "end_seconds": end,
        }
        words = segment.get("words")
        if isinstance(words, list):
            norm_seg["words"] = words
        normalized.append(norm_seg)
    return normalized


_PROVIDER_LANGUAGE_HINT_POLICY: dict[str, str] = {
    "faster-whisper": "iso639_base",
    "parakeet": "iso639_base",
    "canary": "iso639_base",
    "qwen2audio": "iso639_base",
    "qwen3-asr": "preserve_locale",
    "vibevoice": "preserve_locale",
    "external": "preserve_locale",
}


def _normalize_language_tag(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip().replace("_", "-")
    if not normalized:
        return None
    return normalized


def _extract_base_language_code(language_tag: str) -> Optional[str]:
    primary = str(language_tag or "").strip().split("-", 1)[0].lower()
    if not primary:
        return None
    if re.fullmatch(r"[a-z]{2,3}", primary):
        return primary
    return None


def _normalize_language_for_provider(provider_name: str, language: Optional[str]) -> Optional[str]:
    """
    Normalize user language hints based on provider expectations.

    - Providers with ISO-639 style expectations receive base language codes.
    - Providers that can consume locale-style hints keep the normalized locale.
    """
    normalized = _normalize_language_tag(language)
    if not normalized:
        return None

    provider = str(provider_name or "").strip().lower()
    policy = _PROVIDER_LANGUAGE_HINT_POLICY.get(provider, "preserve_locale")
    if policy == "iso639_base":
        return _extract_base_language_code(normalized) or normalized.lower()
    return normalized


def _dictation_error_detail(
    *,
    http_status: int,
    detail_status: Optional[str] = None,
    message: Optional[str] = None,
    detail: Any = None,
    exc: Optional[Exception] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return build_dictation_error_payload(
        status_code=http_status,
        status=detail_status,
        message=message,
        detail=detail,
        exc=exc,
        extra=extra,
    )


@router.post(
    "/transcriptions",
    summary="Transcribes audio into text (OpenAI Compatible)",
    responses=_OPENAI_AUDIO_TRANSCRIPT_RESPONSES,
    dependencies=[
        Depends(check_rate_limit),
        Depends(
            TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.transcriptions", count_as="call")
        ),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
        # Pessimistic pre-check: verifies at least 1 minute of transcription
        # quota remains.  Actual duration is unknown until after processing,
        # so the real usage is recorded post-transcription via
        # apply_usage_delta(TRANSCRIPTION_MINUTES_MONTH, ceil(minutes)).
        Depends(require_within_limit(LimitCategory.TRANSCRIPTION_MINUTES_MONTH, 1)),
    ],
)
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="The audio file to transcribe"),
    model: Optional[str] = Form(
        default=None,
        description="Model to use for transcription (defaults to config when omitted)",
    ),
    language: Optional[str] = Form(default=None, description="Language of the audio in ISO-639-1 format"),
    prompt: Optional[str] = Form(default=None, description="Optional text to guide the model's style"),
    hotwords: Optional[str] = Form(
        default=None,
        description="Optional hotwords to guide transcription (CSV or JSON list). Primarily used by VibeVoice-ASR.",
    ),
    response_format: str = Form(default="json", description="Format of the transcript output"),
    temperature: float = Form(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (currently ignored by all providers).",
    ),
    task: str = Form(
        default="transcribe",
        description="Task for Whisper models: 'transcribe' (default) or 'translate'. "
        "For non-Whisper providers this is ignored.",
    ),
    timestamp_granularities: Optional[str] = Form(
        default="segment", description="Timestamp granularities: 'segment', 'word' (comma-separated or JSON array)"
    ),
    # Auto-segmentation options
    segment: bool = Form(
        default=False, description="If true and JSON response, also run transcript segmentation (TreeSeg)"
    ),
    seg_K: int = Form(default=6, description="Max segments for TreeSeg (if segment=true)"),
    seg_min_segment_size: int = Form(default=5, description="Min items per segment for TreeSeg"),
    seg_lambda_balance: float = Form(default=0.01, description="Balance penalty for TreeSeg"),
    seg_utterance_expansion_width: int = Form(default=2, description="Context width for TreeSeg blocks"),
    seg_embeddings_provider: Optional[str] = Form(default=None, description="Embeddings provider override for TreeSeg"),
    seg_embeddings_model: Optional[str] = Form(default=None, description="Embeddings model override for TreeSeg"),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
    billing_org_id: int | None = Depends(get_billing_org_id),
):
    """
    Transcribes audio into the input language.
    """
    rid = ensure_request_id(request)
    request_started_at = time.perf_counter()
    metrics_provider = "other"
    metrics_model = str(model or "").strip()

    def _emit_request_metrics(*, status_label: str) -> None:
        emit_stt_request_total(
            endpoint="audio.transcriptions",
            provider=metrics_provider,
            model=metrics_model,
            status=status_label,
        )

    def _emit_error_metrics(*, status_label: str, reason: str) -> None:
        _emit_request_metrics(status_label=status_label)
        emit_stt_error_total(
            endpoint="audio.transcriptions",
            provider=metrics_provider,
            reason=reason,
        )

    def _emit_success_metrics(*, redaction_outcome: str) -> None:
        _emit_request_metrics(status_label="ok")
        emit_stt_redaction_total(
            endpoint="audio.transcriptions",
            redaction_outcome=redaction_outcome,
        )
        observe_stt_latency_seconds(
            endpoint="audio.transcriptions",
            provider=metrics_provider,
            model=metrics_model,
            value=max(0.0, time.perf_counter() - request_started_at),
        )

    # Validate file presence
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No audio file provided")
    # Content-Type whitelist
    allowed_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/flac",
        "audio/ogg",
        "audio/opus",
        "audio/webm",
    }
    ctype = (file.content_type or "").lower()
    if ctype and ctype not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"Unsupported media type: {file.content_type}"
        )

    job_heartbeat_task: Optional[asyncio.Task] = None

    async def _maybe_start_job_heartbeat(user_id: int) -> Optional[asyncio.Task]:
        """Best-effort RG job heartbeat loop (no-op when unsupported)."""
        try:
            interval = _audio_shim_attr("get_job_heartbeat_interval_seconds")()
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            logger.debug("audio.transcriptions job heartbeat interval lookup failed; skipping job heartbeat")
            return None
        if not interval or interval <= 0:
            return None

        async def _hb_loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await _audio_shim_attr("heartbeat_jobs")(user_id)
                except asyncio.CancelledError:
                    raise
                except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                    logger.debug("audio.transcriptions job heartbeat failed; continuing")

        try:
            return asyncio.create_task(_hb_loop())
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            logger.debug("audio.transcriptions failed to start job heartbeat task; continuing without heartbeat")
            return None

    # Resolve per-tier file size limit
    try:
        limits = await _audio_shim_attr("get_limits_for_user")(current_user.id)
    except EXPECTED_DB_EXC as e:
        logger.exception(
            'Failed to get limits for user {} during upload, using defaults: {}; request_id={}',
            current_user.id,
            e,
            rid,
        )
        limits = {
            "daily_minutes": 30.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": 25,
        }
    try:
        max_file_size = int((limits.get("max_file_size_mb") or 25) * 1024 * 1024)
    except (ValueError, TypeError) as e:
        logger.warning(
            'Could not parse max_file_size_mb for user {}; defaulting to 25MB: {}; request_id={}',
            current_user.id,
            e,
            rid,
        )
        max_file_size = 25 * 1024 * 1024
    upload_chunk_size = 1024 * 1024

    # Resolve default model from config when omitted.
    if not (model or "").strip():
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
            resolve_default_transcription_model,
        )

        model = resolve_default_transcription_model("whisper-1")

    # Before any heavy work, enforce concurrent jobs cap per user
    ok_job, msg_job = await _audio_shim_attr("can_start_job")(current_user.id)
    if not ok_job:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=msg_job)
    try:
        job_heartbeat_task = await _maybe_start_job_heartbeat(current_user.id)
    except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
        job_heartbeat_task = None

    # Record job start (best-effort)
    acquired_job_slot = False
    try:
        await _audio_shim_attr("increment_jobs_started")(current_user.id)
        acquired_job_slot = True
    except EXPECTED_DB_EXC as e:
        logger.exception(
            'Failed to increment jobs started: user_id={}, error={}; request_id={}',
            current_user.id,
            e,
            rid,
        )

    # Save uploaded file to temporary location and proceed with processing
    temp_audio_path = None
    canonical_path = None
    try:
        effective_stt_policy = await resolve_effective_stt_policy(
            principal=principal,
            user_id=int(current_user.id) if getattr(current_user, "id", None) is not None else None,
            db=db,
        )
    except Exception:
        _emit_error_metrics(status_label="internal_error", reason="policy_resolution_failed")
        raise
    try:
        first_chunk = await file.read(upload_chunk_size)
        file_extension = _resolve_audio_upload_suffix(
            file.filename,
            file.content_type,
            first_chunk,
        )
        total_read = 0
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
            pending_chunk = first_chunk
            while True:
                chunk = pending_chunk
                pending_chunk = None
                if not chunk:
                    chunk = await file.read(upload_chunk_size)
                    if not chunk:
                        break
                total_read += len(chunk)
                if total_read > max_file_size:
                    temp_audio_path = tmp_file.name
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"File size exceeds maximum of {int(max_file_size/1024/1024)}MB",
                    )
                tmp_file.write(chunk)
            temp_audio_path = tmp_file.name

        # Convert to canonical 16k mono WAV for consistent processing; base_dir constrains output location.
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
                ConversionError,
            )
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
                convert_to_wav as _convert_to_wav,
            )
        except ImportError as e:
            logger.debug(
                'convert_to_wav import failed; using original temp file: path={}, error={}',
                temp_audio_path,
                e,
            )
            canonical_path = temp_audio_path
        else:
            try:
                canonical_path = _convert_to_wav(
                    temp_audio_path,
                    offset=0,
                    overwrite=False,
                    base_dir=PathLib(temp_audio_path).parent,
                )
            except (ConversionError, OSError, RuntimeError, ValueError) as e:
                logger.debug(
                    'convert_to_wav failed; using original temp file: path={}, error={}',
                    temp_audio_path,
                    e,
                )
                canonical_path = temp_audio_path

        if is_test_mode():
            logger.debug("TEST_MODE: canonical audio path resolved")

        base_dir = PathLib(canonical_path).parent

        sf_mod = _audio_shim_attr("sf")
        duration_seconds = 0.0
        try:
            info_fn = getattr(sf_mod, "info", None)
            if not callable(info_fn):
                raise AttributeError("soundfile.info not available")
            info = info_fn(canonical_path)
            frames = getattr(info, "frames", None)
            samplerate = getattr(info, "samplerate", None)
            if frames is None or not samplerate:
                raise ValueError("soundfile.info returned incomplete metadata")
            duration_seconds = float(frames) / float(samplerate)
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            logger.debug("soundfile.info failed; falling back to read for duration")
            try:
                audio_data, sample_rate = sf_mod.read(canonical_path)
                duration_seconds = float(len(audio_data)) / float(sample_rate or 16000)
            except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                logger.debug("Failed to compute audio duration; defaulting to 0")
                duration_seconds = 0.0

        granularity_tokens = set()
        try:
            if timestamp_granularities:
                s = str(timestamp_granularities).strip()
                if s.startswith("["):
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        granularity_tokens = {str(x).strip().lower() for x in arr}
                else:
                    granularity_tokens = {t.strip().lower() for t in s.split(",") if t.strip()}
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            logger.debug("Failed to parse timestamp_granularities; defaulting to 'segment'")
            granularity_tokens = {"segment"}
        if not granularity_tokens:
            granularity_tokens = {"segment"}

        task_normalized = (task or "transcribe").strip().lower()
        if task_normalized not in {"transcribe", "translate"}:
            task_normalized = "transcribe"

        hotwords_norm: Optional[list[str]] = None
        raw_hotwords = (hotwords or "").strip()
        if raw_hotwords:
            if raw_hotwords.startswith("["):
                try:
                    parsed_hotwords = json.loads(raw_hotwords)
                    if isinstance(parsed_hotwords, list):
                        hotwords_norm = [str(x).strip() for x in parsed_hotwords if str(x).strip()]
                except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                    logger.debug("Failed to parse hotwords JSON; falling back to CSV parsing")
            if hotwords_norm is None:
                hotwords_norm = [part.strip() for part in raw_hotwords.split(",") if part.strip()]
            hotwords_norm = hotwords_norm[:128] if hotwords_norm else None

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
            Audio_Files as audio_files,
        )
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            is_transcription_error_message as _is_transcription_error_message,
        )
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            validate_whisper_model_identifier,
        )
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
            get_stt_provider_registry,
        )

        stt_registry = get_stt_provider_registry()
        provider, provider_model_name, provider_variant = stt_registry.resolve_provider_for_model(model or "")
        metrics_provider = str(provider or "").strip().lower() or metrics_provider
        metrics_model = str(provider_model_name or model or "").strip() or metrics_model
        provider_envelope = _stt_provider_envelope(stt_registry, provider)
        provider_availability = _stt_provider_availability(
            stt_registry,
            provider,
            envelope=provider_envelope,
        )
        language_for_provider = _normalize_language_for_provider(provider, language)
        requested_model = (model or provider_model_name or "").strip()
        if provider_availability in {"disabled", "failed"}:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_dictation_error_detail(
                    http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail_status="provider_unavailable",
                    message=(
                        f"STT provider '{provider}' is currently {provider_availability}. "
                        "Check provider configuration/health and retry."
                    ),
                    extra={
                        "provider": provider,
                        "availability": provider_availability,
                        "model": requested_model,
                    },
                ),
            )

        def _raise_on_transcription_error(text: Any) -> None:
            if _is_transcription_error_message(text):
                logger.error("Transcription returned error sentinel")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=_dictation_error_detail(
                        http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail_status="transient_failure",
                        message="Transcription failed. Please try again or use a different model.",
                    ),
                )

        minutes_est = duration_seconds / 60.0
        try:
            allow, remaining_after = await _check_daily_minutes_allow(current_user.id, minutes_est)
        except EXPECTED_DB_EXC as e:
            logger.exception(
                'check_daily_minutes_allow failed; allowing by default: user_id={}, error={}; request_id={}',
                current_user.id,
                e,
                rid,
            )
            allow = True
        if not allow:
            try:
                await _audio_shim_attr("finish_job")(current_user.id)
            except EXPECTED_DB_EXC as e:
                logger.exception(
                    'Failed to release job slot after quota denial: user_id={}, error={}; request_id={}',
                    current_user.id,
                    e,
                    rid,
                )
            acquired_job_slot = False
            if job_heartbeat_task:
                job_heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await job_heartbeat_task
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=_dictation_error_detail(
                    http_status=status.HTTP_402_PAYMENT_REQUIRED,
                    detail_status="quota_exceeded",
                    message="Transcription quota exceeded (daily minutes)",
                ),
            )
        # Secondary billing check with the real minute estimate (the dependency
        # pre-check only gates on 1 unit to verify the org has *any* quota).
        if enforcement_enabled() and billing_org_id is not None and minutes_est > 1:
            _real_units = max(1, int(math.ceil(minutes_est)))
            try:
                _enforcer = get_billing_enforcer()
                _recheck = await _enforcer.check_limit(
                    billing_org_id,
                    LimitCategory.TRANSCRIPTION_MINUTES_MONTH,
                    requested_units=_real_units,
                )
                if _recheck.should_block:
                    raise HTTPException(
                        status_code=status.HTTP_402_PAYMENT_REQUIRED,
                        detail={
                            "error": "limit_exceeded",
                            "category": "transcription_minutes_month",
                            "current": _recheck.current,
                            "limit": _recheck.limit,
                            "message": _recheck.message or "Transcription minutes quota exceeded. Please upgrade your plan.",
                            "upgrade_url": "/billing/plans",
                        },
                    )
            except HTTPException:
                raise
            except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                logger.debug("Billing secondary minutes check failed; allowing by default")

        detected_language: Optional[str] = None
        segments_for_timing: Optional[list[dict[str, Any]]] = None
        whisper_model_name = _map_openai_audio_model_to_whisper(model)
        if provider == "faster-whisper":
            try:
                whisper_model_name = validate_whisper_model_identifier(whisper_model_name)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=_http_error_detail("Invalid transcription model identifier", rid, exc=exc),
                ) from exc
        try:
            if provider == "faster-whisper":
                try:
                    model_status = audio_files.check_transcription_model_status(whisper_model_name)
                    if not model_status.get("available", False):
                        # Keep the lightweight status probe for visibility, but
                        # do not block the lazy faster-whisper first-use
                        # initialization/download path.
                        logger.info(
                            "Whisper model {} not cached locally; proceeding with first-use load/download. status_message={}",
                            model_status.get("model", whisper_model_name),
                            model_status.get("message") or "none",
                        )
                except HTTPException:
                    raise
                except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - defensive
                    logger.debug("Whisper model preflight check failed; proceeding without it")
                try:
                    if task_normalized == "translate":
                        selected_lang_for_stt: Optional[str] = None
                    else:
                        selected_lang_for_stt = language_for_provider

                    adapter = stt_registry.get_adapter(provider)
                    if adapter is None:
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=_dictation_error_detail(
                                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail_status="provider_unavailable",
                                message=f"STT provider '{provider}' is unavailable.",
                                extra={
                                    "provider": provider,
                                    "availability": provider_availability,
                                    "model": whisper_model_name,
                                },
                            ),
                        )
                    artifact = adapter.transcribe_batch(
                        canonical_path,
                        model=whisper_model_name,
                        language=selected_lang_for_stt,
                        task=task_normalized,
                        word_timestamps=("word" in granularity_tokens),
                        prompt=prompt,
                        hotwords=hotwords_norm,
                        base_dir=base_dir,
                    )
                    detected_language = artifact.get("language")
                    segments_for_timing = artifact.get("segments") or []
                    transcribed_text = artifact.get("text", "")
                except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS as e:
                    logger.error("Whisper transcription failed")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=_dictation_error_detail(
                            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail_status="transient_failure",
                            message="Whisper transcription failed",
                            exc=e,
                            extra={
                                "provider": provider,
                                "model": whisper_model_name,
                            },
                        ),
                    ) from e
            else:
                model_for_provider = model or provider_model_name
                try:
                    adapter = stt_registry.get_adapter(provider)
                    if adapter is None:
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=_dictation_error_detail(
                                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail_status="provider_unavailable",
                                message=f"STT provider '{provider}' is unavailable.",
                                extra={
                                    "provider": provider,
                                    "availability": provider_availability,
                                    "model": model_for_provider,
                                },
                            ),
                        )
                    adapter_provider = str(getattr(getattr(adapter, "name", None), "value", "")).strip()
                    if adapter_provider and adapter_provider != provider:
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=_dictation_error_detail(
                                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail_status="provider_unavailable",
                                message=(
                                    f"STT provider '{provider}' is unavailable; "
                                    f"fallback to '{adapter_provider}' was prevented for endpoint parity."
                                ),
                                extra={
                                    "provider": provider,
                                    "availability": provider_availability,
                                    "resolved_provider": adapter_provider,
                                    "model": model_for_provider,
                                },
                            ),
                        )
                    artifact = adapter.transcribe_batch(
                        canonical_path,
                        model=model_for_provider,
                        language=language_for_provider,
                        task=task_normalized,
                        word_timestamps=("word" in granularity_tokens),
                        prompt=prompt,
                        hotwords=hotwords_norm,
                        base_dir=base_dir,
                    )
                    detected_language = artifact.get("language")
                    segments_for_timing = artifact.get("segments") or []
                    transcribed_text = artifact.get("text", "")
                except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS as e:
                    logger.error("Transcription failed for STT provider")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=_dictation_error_detail(
                            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail_status="transient_failure",
                            message=(
                                f"Transcription failed for provider '{provider}' "
                                f"and model '{model_for_provider}'"
                            ),
                            exc=e,
                            extra={
                                "provider": provider,
                                "model": model_for_provider,
                            },
                        ),
                    ) from e
        finally:
            try:
                if job_heartbeat_task:
                    job_heartbeat_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await job_heartbeat_task
                if acquired_job_slot:
                    await _audio_shim_attr("finish_job")(current_user.id)
            except EXPECTED_DB_EXC as e:
                logger.exception(
                    'Failed to release job slot in finally: user_id={}, error={}; request_id={}',
                    current_user.id,
                    e,
                    rid,
                )

        _raise_on_transcription_error(transcribed_text)

        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Custom_Vocabulary import (
                postprocess_text_if_enabled as _cv_post,
            )

            transcribed_text = _cv_post(transcribed_text)
        except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
            logger.debug("Custom vocabulary postprocessing failed; continuing without it")

        raw_transcribed_text = transcribed_text
        raw_timed_segments = _normalize_timed_segments(segments_for_timing)

        transcribed_text = apply_transcript_text_policy(
            raw_transcribed_text,
            policy=effective_stt_policy,
            is_partial=False,
        )
        timed_segments = redact_timed_segments(
            raw_timed_segments,
            policy=effective_stt_policy,
        )
        if not effective_stt_policy.redact_pii:
            redaction_outcome = "not_requested"
        elif transcribed_text != raw_transcribed_text or timed_segments != raw_timed_segments:
            redaction_outcome = "applied"
        else:
            redaction_outcome = "skipped"

        try:
            await _add_daily_minutes(current_user.id, minutes_est)
        except EXPECTED_DB_EXC as e:
            logger.exception(
                'Failed to record daily minutes: user_id={}, error={}; request_id={}',
                current_user.id,
                e,
                rid,
            )
        # Billing: record actual transcription minutes to org-level enforcement
        if enforcement_enabled() and billing_org_id is not None and minutes_est > 0:
            _billed_minutes = max(1, int(math.ceil(minutes_est)))
            try:
                get_billing_enforcer().apply_usage_delta(
                    billing_org_id,
                    LimitCategory.TRANSCRIPTION_MINUTES_MONTH,
                    _billed_minutes,
                )
            except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                pass  # fail-open
            # Durable cost-units ledger write so usage survives cache expiry/restart
            try:
                await cost_units.record_cost_units_for_entity(
                    entity_scope="org",
                    entity_value=str(billing_org_id),
                    minutes=float(_billed_minutes),
                )
            except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                pass  # fail-open

        retention_decision = build_audio_retention_decision(effective_stt_policy)
        if not retention_decision.delete_after_success and canonical_path:
            try:
                await retain_stt_audio_artifact(
                    user_id=int(current_user.id),
                    source_path=canonical_path,
                    original_filename=file.filename,
                    policy=effective_stt_policy,
                    org_id=effective_stt_policy.org_id,
                )
            except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    "Failed to retain STT audio artifact; continuing with delete-after-success fallback. error={}",
                    exc,
                )

        if response_format == "text":
            _emit_success_metrics(redaction_outcome=redaction_outcome)
            return Response(content=transcribed_text, media_type="text/plain")

        if response_format == "srt":
            _raise_on_transcription_error(transcribed_text)
            if timed_segments:
                lines: list[str] = []

                def _fmt_srt_timestamp(total_seconds: float) -> str:
                    try:
                        if total_seconds is None:
                            total_seconds = 0.0
                        total_ms = int(round(max(float(total_seconds), 0.0) * 1000))
                    except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                        total_ms = 0
                    seconds, ms = divmod(total_ms, 1000)
                    hours, seconds = divmod(seconds, 3600)
                    minutes, seconds = divmod(seconds, 60)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

                idx = 1
                for seg in timed_segments:
                    text_seg = str(seg.get("text", "")).strip()
                    start = seg.get("start_seconds", 0.0)
                    end = seg.get("end_seconds", start)
                    start_ts = _fmt_srt_timestamp(start)
                    end_ts = _fmt_srt_timestamp(end)
                    lines.append(str(idx))
                    lines.append(f"{start_ts} --> {end_ts}")
                    lines.append(text_seg)
                    lines.append("")
                    idx += 1

                if lines:
                    srt_content = "\n".join(lines).rstrip() + "\n"
                else:
                    srt_content = f"1\n00:00:00,000 --> 00:00:10,000\n{transcribed_text}\n"
            else:
                srt_content = f"1\n00:00:00,000 --> 00:00:10,000\n{transcribed_text}\n"
            _emit_success_metrics(redaction_outcome=redaction_outcome)
            return Response(content=srt_content, media_type="application/x-subrip")

        if response_format == "vtt":
            _raise_on_transcription_error(transcribed_text)
            if timed_segments:
                lines_vtt: list[str] = ["WEBVTT", ""]

                def _fmt_vtt_timestamp(total_seconds: float) -> str:
                    try:
                        if total_seconds is None:
                            total_seconds = 0.0
                        total_ms = int(round(max(float(total_seconds), 0.0) * 1000))
                    except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                        total_ms = 0
                    seconds, ms = divmod(total_ms, 1000)
                    hours, seconds = divmod(seconds, 3600)
                    minutes, seconds = divmod(seconds, 60)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

                for seg in timed_segments:
                    text_seg = str(seg.get("text", "")).strip()
                    start = seg.get("start_seconds", 0.0)
                    end = seg.get("end_seconds", start)
                    start_ts = _fmt_vtt_timestamp(start)
                    end_ts = _fmt_vtt_timestamp(end)
                    lines_vtt.append(f"{start_ts} --> {end_ts}")
                    lines_vtt.append(text_seg)
                    lines_vtt.append("")

                vtt_content = "\n".join(lines_vtt).rstrip() + "\n"
            else:
                vtt_content = f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{transcribed_text}\n"
            _emit_success_metrics(redaction_outcome=redaction_outcome)
            return Response(content=vtt_content, media_type="text/vtt")

        response_data: dict[str, Any] = {"text": transcribed_text}

        if language:
            response_data["language"] = language
        elif detected_language:
            response_data["language"] = detected_language

        duration = duration_seconds
        response_data["duration"] = duration

        if "segment" in granularity_tokens:
            if timed_segments:
                segs = []
                for i, seg in enumerate(timed_segments):
                    start = float(seg.get("start_seconds", 0.0))
                    end = float(seg.get("end_seconds", duration))
                    seg_obj: dict[str, Any] = {
                        "id": i,
                        "start": start,
                        "end": end,
                        "text": seg.get("text", ""),
                    }
                    if "word" in granularity_tokens and isinstance(seg.get("words"), list):
                        seg_obj["words"] = seg["words"]
                    segs.append(seg_obj)
                response_data["segments"] = segs
            else:
                response_data["segments"] = [
                    {
                        "id": 0,
                        "seek": 0,
                        "start": 0.0,
                        "end": duration,
                        "text": transcribed_text,
                    }
                ]

        if segment:
            try:
                import re

                from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Transcript_TreeSegmentation import (
                    TreeSegmenter,
                )

                lines = [ln.strip() for ln in transcribed_text.splitlines() if ln.strip()]
                if not lines:
                    lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcribed_text) if s.strip()]
                entries = [{"composite": ln} for ln in lines]

                if entries:
                    configs = {
                        "MIN_SEGMENT_SIZE": seg_min_segment_size,
                        "LAMBDA_BALANCE": seg_lambda_balance,
                        "UTTERANCE_EXPANSION_WIDTH": seg_utterance_expansion_width,
                    }
                    if seg_embeddings_provider:
                        configs["EMBEDDINGS_PROVIDER"] = seg_embeddings_provider
                    if seg_embeddings_model:
                        configs["EMBEDDINGS_MODEL"] = seg_embeddings_model

                    segmenter = await TreeSegmenter.create_async(configs=configs, entries=entries)
                    transitions = segmenter.segment_meeting(K=seg_K)
                    segs = segmenter.get_segments()
                    response_data["segmentation"] = {
                        "transitions": transitions,
                        "transition_indices": segmenter.get_transition_indices(),
                        "segments": segs,
                    }
            except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                logger.warning("Auto-segmentation failed; continuing without it")

        if response_format == "verbose_json":
            response_data["task"] = task_normalized
            response_data["duration"] = duration

        _emit_success_metrics(redaction_outcome=redaction_outcome)
        return JSONResponse(content=response_data)

    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        detail_status = str(
            detail.get("status")
            or detail.get("detail_status")
            or ""
        ).strip().lower()
        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            _emit_error_metrics(status_label="quota_exceeded", reason="quota")
        elif exc.status_code == status.HTTP_402_PAYMENT_REQUIRED:
            _emit_error_metrics(status_label="quota_exceeded", reason="quota")
        elif exc.status_code in {
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_413_CONTENT_TOO_LARGE,
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        }:
            _emit_error_metrics(status_label="bad_request", reason="validation_error")
        elif exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            _emit_error_metrics(status_label="model_unavailable", reason="model_unavailable")
        elif detail_status in {"provider_unavailable", "transient_failure"}:
            _emit_error_metrics(status_label="provider_error", reason="provider_error")
        else:
            _emit_error_metrics(status_label="internal_error", reason="internal")
        raise
    except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS as e:
        _emit_error_metrics(status_label="internal_error", reason="internal")
        logger.error("Error during transcription")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_dictation_error_detail(
                http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail_status="transient_failure",
                message="Transcription failed",
                detail=_http_error_detail("Transcription failed", rid, exc=e),
                exc=e,
            ),
        ) from e
    finally:
        if canonical_path and canonical_path != temp_audio_path and os.path.exists(canonical_path):
            try:
                os.remove(canonical_path)
            except OSError:
                logger.warning("Failed to remove canonical audio file")
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                logger.warning("Failed to remove temp audio file")
                try:
                    increment_counter(
                        "app_warning_events_total", labels={"component": "audio", "event": "tempfile_remove_failed"}
                    )
                except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed (audio tempfile_remove_failed)")


@router.post(
    "/translations",
    summary="Translates audio into English (OpenAI Compatible)",
    responses=_OPENAI_AUDIO_TRANSCRIPT_RESPONSES,
    dependencies=[
        Depends(check_rate_limit),
        Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="audio.translations", count_as="call")),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
        # Pessimistic pre-check: verifies at least 1 minute of transcription
        # quota remains.  Actual duration is unknown until after processing,
        # so the real usage is recorded post-transcription via
        # apply_usage_delta(TRANSCRIPTION_MINUTES_MONTH, ceil(minutes)).
        Depends(require_within_limit(LimitCategory.TRANSCRIPTION_MINUTES_MONTH, 1)),
    ],
)
async def create_translation(
    request: Request,
    file: UploadFile = File(..., description="The audio file to translate"),
    model: Optional[str] = Form(
        default=None,
        description="Model to use for translation (defaults to config when omitted)",
    ),
    prompt: Optional[str] = Form(default=None, description="Optional text to guide the model's style"),
    response_format: str = Form(default="json", description="Format of the transcript output"),
    temperature: float = Form(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (currently ignored by all providers).",
    ),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
    billing_org_id: int | None = Depends(get_billing_org_id),
):
    """
    Translates audio into English.
    """
    if not (model or "").strip():
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
            resolve_default_transcription_model,
        )

        model = resolve_default_transcription_model("whisper-1")

    try:
        usage_log.log_event(
            "audio.translations",
            tags=[str(model or "")],
            metadata={"filename": getattr(file, "filename", None), "language": "en"},
        )
    except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
        logger.debug("usage_log audio.translations failed")

    return await create_transcription(
        request=request,
        file=file,
        model=model,
        language=None,
        prompt=prompt,
        hotwords=None,
        response_format=response_format,
        temperature=temperature,
        task="translate",
        timestamp_granularities="segment",
        segment=False,
        seg_K=6,
        seg_min_segment_size=5,
        seg_lambda_balance=0.01,
        seg_utterance_expansion_width=2,
        seg_embeddings_provider=None,
        seg_embeddings_model=None,
        current_user=current_user,
        principal=principal,
        db=db,
        billing_org_id=billing_org_id,
    )


@router.post("/segment/transcript", summary="Segment a transcript into coherent blocks (TreeSeg)")
async def segment_transcript(
    req: TranscriptSegmentationRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
):
    """
    Segment a transcript into coherent segments using TreeSeg (hierarchical segmentation).
    """
    try:
        if not req.entries or len(req.entries) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No entries provided")

        configs = {
            "MIN_SEGMENT_SIZE": req.min_segment_size,
            "LAMBDA_BALANCE": req.lambda_balance,
            "UTTERANCE_EXPANSION_WIDTH": req.utterance_expansion_width,
            "MIN_IMPROVEMENT_RATIO": getattr(req, "min_improvement_ratio", 0.0),
        }
        if req.embeddings_provider:
            configs["EMBEDDINGS_PROVIDER"] = req.embeddings_provider
        if req.embeddings_model:
            configs["EMBEDDINGS_MODEL"] = req.embeddings_model

        entries = [e.model_dump() for e in req.entries]

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Transcript_TreeSegmentation import (
            TreeSegmenter,
        )

        segmenter = await TreeSegmenter.create_async(configs=configs, entries=entries)
        transitions = segmenter.segment_meeting(K=req.K)
        segs = segmenter.get_segments()

        return TranscriptSegmentationResponse(
            transitions=transitions,
            transition_indices=segmenter.get_transition_indices(),
            segments=segs,
        )

    except HTTPException:
        raise
    except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Transcript segmentation failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Transcript segmentation failed") from e
