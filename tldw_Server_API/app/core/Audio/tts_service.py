import copy
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import HTTPException
from loguru import logger
from starlette import status

from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.AuthNZ.byok_config import PROVIDER_APP_CONFIG_KEYS
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    derive_trusted_credential_scope,
    load_server_config_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
    record_byok_missing_credentials,
    resolve_byok_credentials,
    resolve_static_server_fallback_from_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    capture_provider_override_call_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker
from tldw_Server_API.app.core.exceptions import TTSPublicHTTPException, raise_detached_error
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterFactory, TTSAdapterRegistry
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
from tldw_Server_API.app.core.TTS.utils import contains_tts_credential_fields
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

_TTS_API_KEY_REQUIRED_PROVIDERS = {"openai", "elevenlabs", "fish_s2"}
_TTS_CREDENTIAL_RESOLVER = Callable[..., Awaitable[Any]]


def _normalize_tts_provider_hint(provider_hint: Optional[str]) -> str:
    """Normalize TTS provider hints for credential requirement checks."""
    return str(provider_hint or "").strip().lower().replace("-", "_")


def _resolved_api_key(value: object) -> Optional[str]:
    """Return a non-empty API key string, or None for blank/missing values."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _raise_missing_tts_credentials(provider_hint: str) -> None:
    record_byok_missing_credentials(provider_hint, operation="audio_tts")
    raise TTSPublicHTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error_code": "missing_provider_credentials",
            "message": f"TTS provider '{provider_hint}' requires an API key.",
        },
    )


def _infer_tts_provider_from_model(model: Optional[str]) -> Optional[str]:
    """Best-effort mapping from model id to provider key for sanitization."""
    if not model:
        return None
    m = str(model).strip().lower()
    mapped_provider = TTSAdapterFactory.MODEL_PROVIDER_MAP.get(m)
    if mapped_provider is None:
        mapped_provider = TTSAdapterRegistry.resolve_provider(m)
    if mapped_provider is not None:
        return mapped_provider.value
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
    if m.startswith("fish-s2") or m.startswith("fish_s2") or m.startswith("fishaudio/s2"):
        return "fish_s2"
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


def _tts_log_context(exc: Exception) -> Any:
    return logger.bind(error_type=type(exc).__name__)


def _raise_for_tts_error_impl(exc: Exception, request_id: Optional[str]) -> None:
    if isinstance(exc, TTSInvalidVoiceReferenceError):
        _tts_log_context(exc).warning("TTS voice reference error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_http_error_detail("TTS voice reference invalid", request_id),
        )
    if isinstance(exc, TTSValidationError):
        _tts_log_context(exc).warning("TTS validation error")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_http_error_detail("TTS validation failed", request_id),
        )
    if isinstance(exc, TTSModelNotFoundError):
        _tts_log_context(exc).error("TTS model not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_http_error_detail("Requested TTS model not found", request_id),
        )
    if isinstance(exc, TTSProviderNotConfiguredError):
        _tts_log_context(exc).error("TTS provider not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS service unavailable", request_id),
        )
    if isinstance(exc, (TTSProviderInitializationError, TTSConfigurationError)):
        _tts_log_context(exc).error("TTS provider initialization error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS service unavailable", request_id),
        )
    if isinstance(exc, TTSModelLoadError):
        _tts_log_context(exc).error("TTS model load error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS model unavailable", request_id),
        )
    if isinstance(exc, TTSResourceError):
        _tts_log_context(exc).error("TTS resource error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS service unavailable", request_id),
        )
    if isinstance(exc, TTSProviderUnavailableError):
        _tts_log_context(exc).error("TTS provider unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_http_error_detail("TTS provider unavailable", request_id),
        )
    if isinstance(exc, TTSAuthenticationError):
        _tts_log_context(exc).error("TTS authentication error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_http_error_detail("TTS provider authentication failed", request_id),
        )
    if isinstance(exc, TTSNetworkError):
        _tts_log_context(exc).error("TTS network error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_http_error_detail("TTS provider request failed", request_id),
        )
    if isinstance(exc, TTSTimeoutError):
        _tts_log_context(exc).error("TTS timeout error")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=_http_error_detail("TTS provider timed out", request_id),
        )
    if isinstance(exc, TTSRateLimitError):
        _tts_log_context(exc).warning("TTS rate limit exceeded")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=_http_error_detail(
                "TTS provider rate limit exceeded. Please try again later.", request_id
            ),
        )
    if isinstance(exc, TTSQuotaExceededError):
        _tts_log_context(exc).warning("TTS quota exceeded")
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=_http_error_detail("TTS quota exceeded. Please review your plan or quota.", request_id),
        )
    if isinstance(exc, TTSError):
        _tts_log_context(exc).error("TTS error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail("TTS generation failed", request_id),
        )
    _tts_log_context(exc).error("Unexpected error during audio generation")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=_http_error_detail("An unexpected error occurred during audio generation", request_id),
    )


def _raise_for_tts_error(exc: Exception, request_id: Optional[str]) -> None:
    """Raise a bounded public HTTP error detached from provider exceptions."""

    mapped: Optional[HTTPException] = None
    try:
        _raise_for_tts_error_impl(exc, request_id)
    except HTTPException as http_exc:
        mapped = http_exc

    if mapped is None:  # pragma: no cover - implementation always raises
        mapped = HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_http_error_detail(
                "An unexpected error occurred during audio generation",
                request_id,
            ),
        )

    public_error = TTSPublicHTTPException(
        status_code=mapped.status_code,
        detail=mapped.detail,
        headers=mapped.headers,
    )
    mapped = None
    exc = None  # type: ignore[assignment]
    raise_detached_error(public_error)


def _sanitize_speech_request(
    request_data: Any,
    *,
    request_id: Optional[str],
) -> Optional[str]:
    """Validate and sanitize input text, returning provider hint."""
    if contains_tts_credential_fields(getattr(request_data, "extra_params", None)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "credential_fields_not_allowed",
                "message": "Credential fields are not accepted in TTS requests.",
            },
        )

    provider_hint = None
    if getattr(request_data, "backend", None) is None:
        model = getattr(request_data, "model", None)
        provider_hint = _infer_tts_provider_from_model(model)
        if isinstance(model, str) and model.strip() and provider_hint is None:
            _raise_for_tts_error(
                TTSModelNotFoundError("Requested TTS model is not registered"),
                request_id,
            )

    try:
        tts_config = get_tts_config()
        validator = TTSInputValidator({"strict_validation": tts_config.strict_validation})
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
        _raise_for_tts_error(exc, request_id)


def _tts_fallback_resolver(name: str) -> Optional[str]:
    try:
        cfg = get_tts_config()
        provider_cfg = getattr(cfg, "providers", {}).get(name)
        api_key = getattr(provider_cfg, "api_key", None) if provider_cfg else None
        return api_key or None
    except (AttributeError, KeyError, TypeError) as exc:
        logger.debug(f"TTS fallback resolver failed for provider '{name}': {exc}")
        return None


def _first_tts_config_string(
    config: dict[str, Any],
    *keys: str,
) -> Optional[str]:
    """Return the first non-empty string from a frozen provider config."""
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _openai_tts_endpoint(base_url: str) -> str:
    """Normalize a frozen OpenAI-compatible base URL to the speech endpoint."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/audio/speech"):
        return normalized
    return f"{normalized}/audio/speech"


def _project_tts_provider_overrides(
    provider: str,
    resolution: Any,
    provider_config_snapshot: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Project one resolved credential snapshot into TTS adapter config keys."""
    provider_key = _normalize_tts_provider_hint(provider)
    overrides: dict[str, Any] = {"credentials_resolved": True}
    api_key = _resolved_api_key(getattr(resolution, "api_key", None))
    if api_key is not None:
        overrides["api_key"] = api_key
        if provider_key == "openai":
            overrides["openai_api_key"] = api_key
        elif provider_key == "elevenlabs":
            overrides["elevenlabs_api_key"] = api_key

    app_config = getattr(resolution, "app_config", None)
    section_name = PROVIDER_APP_CONFIG_KEYS.get(
        provider_key,
        f"{provider_key.replace('-', '_')}_api",
    )
    provider_config = (
        app_config.get(section_name)
        if isinstance(app_config, dict)
        else None
    )
    if not isinstance(provider_config, dict):
        provider_config = {}
    frozen_provider_config = provider_config_snapshot or {}
    base_url = _first_tts_config_string(
        provider_config,
        "api_base_url",
        "base_url",
        "api_url",
        "api_ip",
        "endpoint",
    )
    if base_url is None:
        base_url = _first_tts_config_string(
            frozen_provider_config,
            "api_base_url",
            "base_url",
            "api_url",
            "api_ip",
            "endpoint",
        )
    if base_url is not None:
        if provider_key == "openai":
            overrides["openai_base_url"] = _openai_tts_endpoint(base_url)
        elif provider_key == "elevenlabs":
            overrides["elevenlabs_base_url"] = base_url.rstrip("/")
        else:
            overrides["base_url"] = base_url.rstrip("/")

    if provider_key == "openai":
        organization = _first_tts_config_string(
            provider_config,
            "org_id",
            "organization_id",
            "organization",
        )
        project = _first_tts_config_string(provider_config, "project_id", "project")
        if organization is not None:
            overrides["organization"] = organization
        if project is not None:
            overrides["project"] = project

    if provider_key == "fish_s2":
        for key in (
            "backend",
            "timeout",
            "model",
            "sample_rate",
            "max_text_length",
            "extra_params",
        ):
            if key in frozen_provider_config and frozen_provider_config[key] is not None:
                overrides[key] = copy.deepcopy(frozen_provider_config[key])

    return overrides


def _capture_tts_provider_config(provider: str) -> dict[str, Any]:
    """Capture one provider's non-authoritative TTS config before async resolution."""
    providers = getattr(get_tts_config(), "providers", {})
    provider_config = providers.get(_normalize_tts_provider_hint(provider)) if isinstance(providers, dict) else None
    if provider_config is None:
        return {}
    if isinstance(provider_config, dict):
        return copy.deepcopy(provider_config)
    return copy.deepcopy(model_dump_compat(provider_config))


def _tts_provider_requires_api_key(
    provider: str,
    provider_config_snapshot: dict[str, Any],
) -> bool:
    """Return whether the frozen TTS backend mode requires an API key."""
    provider_key = _normalize_tts_provider_hint(provider)
    if provider_key == "fish_s2":
        backend = str(
            provider_config_snapshot.get("backend") or "native_http"
        ).strip().lower()
        return backend in {"commercial_api", "hosted", "fish_audio"}
    return provider_key in _TTS_API_KEY_REQUIRED_PROVIDERS


async def _resolve_tts_byok(
    *,
    provider_hint: Optional[str],
    current_user: User,
    request: Any,
    model: Optional[str] = None,
    force_oauth_refresh: bool = False,
    rejected_credentials: Optional[ResolvedByokCredentials] = None,
    credential_resolver: Optional[_TTS_CREDENTIAL_RESOLVER] = None,
) -> tuple[Optional[int], Optional[dict[str, Any]], Optional[Any]]:
    user_id_int: Optional[int] = None
    try:
        user_id_int = getattr(current_user, "id_int", None)
        if user_id_int is None:
            raw_id = getattr(current_user, "id", None)
            if raw_id is not None:
                user_id_int = int(raw_id)
    except (AttributeError, TypeError, ValueError):
        logger.debug("Failed to extract user_id from current_user")
        user_id_int = None

    tts_overrides: Optional[dict[str, Any]] = None
    byok_tts_resolution = None
    rejected_credential_generation: Optional[str] = None
    if force_oauth_refresh:
        if not isinstance(rejected_credentials, ResolvedByokCredentials):
            raise ByokResolutionError(
                "invalid_provider_credentials",
                provider_hint,
            )
        try:
            rejected_provider = rejected_credentials.provider
            rejected_credential_generation = getattr(
                rejected_credentials,
                "_credential_generation",
                None,
            )
            normalized_provider = _normalize_tts_provider_hint(provider_hint)
            provider_matches = (
                bool(normalized_provider)
                and _normalize_tts_provider_hint(rejected_provider)
                == normalized_provider
            )
        except Exception:  # noqa: BLE001 - credential metadata fails closed
            raise ByokResolutionError(
                "invalid_provider_credentials",
                provider_hint,
            ) from None
        if (
            not provider_matches
            or not isinstance(rejected_credential_generation, str)
            or not rejected_credential_generation.strip()
        ):
            raise ByokResolutionError(
                "invalid_provider_credentials",
                provider_hint,
            )

    if provider_hint:
        try:
            override_snapshot = capture_provider_override_call_snapshot(provider_hint)
            override_snapshot.enforce(model)
            tts_provider_config_snapshot = _capture_tts_provider_config(provider_hint)
            server_config_snapshot = load_server_config_snapshot()
            static_fallback = resolve_static_server_fallback_from_snapshot(
                provider_hint,
                server_config_snapshot,
            )
            fallback_override = (
                override_snapshot.server_fallback(static_fallback)
                or static_fallback
            )
        except ByokResolutionError:
            raise
        except Exception:  # noqa: BLE001 - config capture failures must fail closed
            raise ByokResolutionError(
                "invalid_provider_credentials",
                provider_hint,
            ) from None
        resolver = credential_resolver or resolve_byok_credentials
        resolver_kwargs: dict[str, Any] = {
            "user_id": user_id_int,
            "request": request,
            "fallback_override": fallback_override,
            "server_config_snapshot": server_config_snapshot,
            "force_oauth_refresh": force_oauth_refresh,
        }
        if force_oauth_refresh and rejected_credential_generation is not None:
            resolver_kwargs["rejected_credential_generation"] = (
                rejected_credential_generation
            )
        byok_tts_resolution = await resolver(provider_hint, **resolver_kwargs)
        if (
            not isinstance(byok_tts_resolution, ResolvedByokCredentials)
            or byok_tts_resolution.status
            not in {ByokResolutionStatus.RESOLVED, ByokResolutionStatus.ABSENT}
            or _normalize_tts_provider_hint(byok_tts_resolution.provider)
            != _normalize_tts_provider_hint(provider_hint)
        ):
            raise ByokResolutionError(
                "invalid_provider_credentials",
                provider_hint,
            ) from None
        override_snapshot.ensure_healthy()
        resolved_api_key = _resolved_api_key(byok_tts_resolution.api_key)
        requires_api_key = _tts_provider_requires_api_key(
            provider_hint,
            tts_provider_config_snapshot,
        )
        if not resolved_api_key and requires_api_key:
            _raise_missing_tts_credentials(provider_hint)
        tts_overrides = _project_tts_provider_overrides(
            provider_hint,
            byok_tts_resolution,
            tts_provider_config_snapshot,
        )

    return user_id_int, tts_overrides, byok_tts_resolution


@asynccontextmanager
async def tts_provider_credential_scope(
    *,
    provider: str,
    model: Optional[str],
    request: Any,
    current_user: Any,
) -> AsyncIterator[
    tuple[Optional[int], dict[str, Any], ProviderCredentialRuntime, Any]
]:
    """Own one authoritative TTS credential snapshot through adapter cleanup."""
    provider_key = _normalize_tts_provider_hint(provider)
    if not provider_key:
        raise ByokResolutionError("invalid_provider_credentials", provider)

    try:
        provider_config_snapshot = _capture_tts_provider_config(provider_key)
        server_config_snapshot = load_server_config_snapshot()
        user_id, team_ids, org_ids, trusted_base_url_override = (
            derive_trusted_credential_scope(request, current_user)
        )
        runtime = ProviderCredentialRuntime(
            user_id=user_id,
            team_ids=team_ids,
            org_ids=org_ids,
            trusted_base_url_override=trusted_base_url_override,
            server_config_snapshot=server_config_snapshot,
            override_snapshot_resolver=capture_provider_override_call_snapshot,
        )
    except ByokResolutionError:
        raise
    except Exception:  # noqa: BLE001 - credential config capture fails closed
        raise ByokResolutionError(
            "invalid_provider_credentials",
            provider_key,
        ) from None

    try:
        credentials = await await_owned_worker(
            runtime.resolve(provider_key, model=model)
        )
        resolved_provider = _normalize_tts_provider_hint(
            getattr(credentials, "provider", None)
        )
        if (
            resolved_provider != provider_key
            or getattr(credentials, "credentials_resolved", None) is not True
        ):
            raise ByokResolutionError(
                "invalid_provider_credentials",
                provider_key,
            )
        if (
            _tts_provider_requires_api_key(provider_key, provider_config_snapshot)
            and _resolved_api_key(getattr(credentials, "api_key", None)) is None
        ):
            _raise_missing_tts_credentials(provider_key)
        overrides = _project_tts_provider_overrides(
            provider_key,
            credentials,
            provider_config_snapshot,
        )
        yield user_id, overrides, runtime, credentials
    finally:
        await await_owned_worker(runtime.close())
