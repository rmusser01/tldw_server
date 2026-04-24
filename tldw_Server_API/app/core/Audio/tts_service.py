from typing import Any, Optional

from fastapi import HTTPException
from loguru import logger
from starlette import status

from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    record_byok_missing_credentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.TTS.tts_config import get_tts_config
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSConfigurationError,
    TTSError,
    TTSInvalidVoiceReferenceError,
    TTSModelLoadError,
    TTSModelNotFoundError,
    TTSNetworkError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSProviderUnavailableError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSResourceError,
    TTSTimeoutError,
    TTSValidationError,
)
from tldw_Server_API.app.core.TTS.tts_validation import TTSInputValidator


def _infer_tts_provider_from_model(model: Optional[str]) -> Optional[str]:
    """Best-effort mapping from model id to provider key for sanitization."""
    if not model:
        return None
    m = str(model).strip().lower()
    if m in {"tts-1", "tts-1-hd"}:
        return "openai"
    if m.startswith("kokoro"):
        return "kokoro"
    if (
        m.startswith("kitten_tts")
        or m.startswith("kitten-tts")
        or m.startswith("kittentts")
        or m.startswith("kittenml/kitten-tts")
    ):
        return "kitten_tts"
    if m.startswith("higgs"):
        return "higgs"
    if m.startswith("dia"):
        return "dia"
    if m.startswith("chatterbox"):
        return "chatterbox"
    if m.startswith("vibevoice"):
        return "vibevoice"
    if m.startswith("neutts"):
        return "neutts"
    if m.startswith("eleven"):
        return "elevenlabs"
    if m.startswith("omnivoice") or m.startswith("omni-voice") or m.startswith("omni_voice"):
        return "omnivoice"
    if m.startswith("index_tts") or m.startswith("indextts"):
        return "index_tts"
    if m.startswith("supertonic2") or m.startswith("supertonic-2") or m.startswith("tts-supertonic2"):
        return "supertonic2"
    if m.startswith("supertonic") or m.startswith("tts-supertonic"):
        return "supertonic"
    if m.startswith("echo-tts") or m.startswith("echo_tts") or m.startswith("jordand/echo-tts"):
        return "echo_tts"
    if m.startswith("qwen3_tts") or m.startswith("qwen3-tts") or m.startswith("qwen/qwen3-tts"):
        return "qwen3_tts"
    return None


def _raise_for_tts_error(exc: Exception, request_id: Optional[str]) -> None:
    if isinstance(exc, TTSInvalidVoiceReferenceError):
        logger.warning(f"TTS voice reference error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_http_error_detail("TTS voice reference invalid", request_id, exc=exc),
        )
    if isinstance(exc, TTSValidationError):
        logger.warning(f"TTS validation error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_http_error_detail("TTS validation failed", request_id, exc=exc),
        )
    if isinstance(exc, TTSModelNotFoundError):
        logger.error(f"TTS model not found: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_http_error_detail("Requested TTS model not found", request_id, exc=exc),
        )
    if isinstance(exc, TTSProviderNotConfiguredError):
        logger.error(f"TTS provider not configured: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS service unavailable", request_id, exc=exc),
        )
    if isinstance(exc, (TTSProviderInitializationError, TTSConfigurationError)):
        logger.error(f"TTS provider initialization error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS service unavailable", request_id, exc=exc),
        )
    if isinstance(exc, TTSModelLoadError):
        logger.error(f"TTS model load error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS model unavailable", request_id, exc=exc),
        )
    if isinstance(exc, TTSResourceError):
        logger.error(f"TTS resource error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS service unavailable", request_id, exc=exc),
        )
    if isinstance(exc, TTSProviderUnavailableError):
        logger.error(f"TTS provider unavailable: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS provider unavailable", request_id, exc=exc),
        )
    if isinstance(exc, TTSAuthenticationError):
        logger.error(f"TTS authentication error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_http_error_detail("TTS provider authentication failed", request_id, exc=exc),
        )
    if isinstance(exc, TTSNetworkError):
        logger.error(f"TTS network error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_http_error_detail("TTS provider request failed", request_id, exc=exc),
        )
    if isinstance(exc, TTSTimeoutError):
        logger.error(f"TTS timeout error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=_http_error_detail("TTS provider timed out", request_id, exc=exc),
        )
    if isinstance(exc, TTSRateLimitError):
        logger.warning(f"TTS rate limit exceeded: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=_http_error_detail(
                "TTS provider rate limit exceeded. Please try again later.", request_id, exc=exc
            ),
        )
    if isinstance(exc, TTSQuotaExceededError):
        logger.warning(f"TTS quota exceeded: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=_http_error_detail("TTS quota exceeded. Please review your plan or quota.", request_id, exc=exc),
        )
    if isinstance(exc, TTSError):
        logger.error(f"TTS error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("TTS generation failed", request_id, exc=exc),
        )
    logger.error(f"Unexpected error during audio generation: {exc}", exc_info=True)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=_http_error_detail("An unexpected error occurred during audio generation", request_id, exc=exc),
    )


def _sanitize_speech_request(
    request_data: Any,
    *,
    request_id: Optional[str],
) -> Optional[str]:
    """Validate and sanitize input text, returning provider hint."""
    try:
        tts_config = get_tts_config()
        validator = TTSInputValidator({"strict_validation": tts_config.strict_validation})

        provider_hint = _infer_tts_provider_from_model(getattr(request_data, "model", None))
        fields_set = getattr(request_data, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(request_data, "__pydantic_fields_set__", None)
        if fields_set is None:
            fields_set = getattr(request_data, "__fields_set__", set())
        voice_was_supplied = "voice" in set(fields_set or ())
        if not voice_was_supplied and provider_hint == "omnivoice":
            # OmniVoice must default to the provider-specific public API voice,
            # regardless of any configured global default voice.
            request_data.voice = "auto"
        sanitized_text = validator.sanitize_text(request_data.input, provider=provider_hint)
        if not sanitized_text or len(sanitized_text.strip()) == 0:
            raise TTSValidationError(
                "Input text cannot be empty after sanitization",
                details={"original_length": len(request_data.input)},
            )
        request_data.input = sanitized_text
        return provider_hint
    except TTSValidationError as exc:
        logger.warning(f"TTS validation error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_http_error_detail("TTS validation failed", request_id, exc=exc),
        ) from exc


def _tts_fallback_resolver(name: str) -> Optional[str]:
    try:
        cfg = get_tts_config()
        provider_cfg = getattr(cfg, "providers", {}).get(name)
        api_key = getattr(provider_cfg, "api_key", None) if provider_cfg else None
        return api_key or None
    except (AttributeError, KeyError, TypeError) as exc:
        logger.debug(f"TTS fallback resolver failed for provider '{name}': {exc}")
        return None


async def _resolve_tts_byok(
    *,
    provider_hint: Optional[str],
    current_user: User,
    request: Any,
    force_oauth_refresh: bool = False,
) -> tuple[Optional[int], Optional[dict[str, Any]], Optional[Any]]:
    user_id_int: Optional[int] = None
    try:
        user_id_int = getattr(current_user, "id_int", None)
        if user_id_int is None:
            raw_id = getattr(current_user, "id", None)
            if raw_id is not None:
                user_id_int = int(raw_id)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug(f"Failed to extract user_id from current_user: {exc}")
        user_id_int = None

    tts_overrides: Optional[dict[str, Any]] = None
    byok_tts_resolution = None
    if provider_hint:
        byok_tts_resolution = await resolve_byok_credentials(
            provider_hint,
            user_id=user_id_int,
            request=request,
            fallback_resolver=_tts_fallback_resolver,
            force_oauth_refresh=force_oauth_refresh,
        )
        if byok_tts_resolution.uses_byok:
            tts_overrides = {"api_key": byok_tts_resolution.api_key}
            base_url = byok_tts_resolution.credential_fields.get("base_url")
            if isinstance(base_url, str) and base_url.strip():
                tts_overrides["base_url"] = base_url.strip()
        elif not byok_tts_resolution.api_key:
            if provider_hint in {"openai", "elevenlabs"}:
                record_byok_missing_credentials(provider_hint, operation="audio_tts")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error_code": "missing_provider_credentials",
                        "message": f"TTS provider '{provider_hint}' requires an API key.",
                    },
                )

    return user_id_int, tts_overrides, byok_tts_resolution
