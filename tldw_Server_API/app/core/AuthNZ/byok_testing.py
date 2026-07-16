from __future__ import annotations

import asyncio
import math
import os
import threading
from collections.abc import Callable
from typing import Any, Literal

from tldw_Server_API.app.core.AuthNZ.byok_config import build_app_config_overrides
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,
    DaemonCapacityError,
    await_bounded_daemon_with_timeout,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call
from tldw_Server_API.app.core.Chat.streaming_utils import (
    PROVIDER_STREAM_ERROR_MESSAGES,
    normalize_provider_stream_error,
    sanitized_provider_stream_exception,
)
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.http_client import (
    RetryPolicy,
    afetch_json,
    sensitive_http_observability_context,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import (
    ChatProviderRegistry,
    get_registry,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    normalize_provider,
    resolve_provider_model,
    resolve_provider_section,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import (
    canonical_provider_name,
)
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import (
    list_registered_providers,
    provider_requires_api_key,
)
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.TTS.gateway_catalog import (
    MAX_DISCOVERY_BYTES,
    _parse_discovered_models,
)
from tldw_Server_API.app.core.TTS.gateway_config import build_gateway_url

_INVALID_TEST_KEY_PREFIXES = ("invalid-", "test-invalid-", "bad-key-", "dummy-invalid-")
PROVIDER_CREDENTIAL_VALIDATION_TIMEOUT_SECONDS = 10.0


def _provider_health_admission_capacity(shared_capacity: int) -> int:
    """Reserve most shared adapter capacity for foreground provider calls."""
    if shared_capacity <= 1:
        return 0
    return min(8, max(1, shared_capacity // 4), shared_capacity - 1)


_PROVIDER_HEALTH_ADMISSION_CAPACITY = _provider_health_admission_capacity(
    SYNC_ADAPTER_CALL_POOL.capacity
)
_PROVIDER_HEALTH_ADMISSION = (
    threading.BoundedSemaphore(_PROVIDER_HEALTH_ADMISSION_CAPACITY)
    if _PROVIDER_HEALTH_ADMISSION_CAPACITY > 0
    else None
)
_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER: dict[
    str,
    threading.BoundedSemaphore,
] = {}
_PROVIDER_HEALTH_ADMISSIONS_LOCK = threading.Lock()


def provider_credential_validation_per_provider_capacity() -> int:
    """Resolve the bounded per-provider validation limit with legacy fallback."""
    for name in (
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "ADMIN_BYOK_VALIDATION_PER_PROVIDER_CONCURRENCY",
    ):
        raw_value = os.getenv(name)
        if raw_value is None:
            continue
        try:
            value = int(raw_value.strip())
        except (AttributeError, ValueError):
            continue
        if value > 0:
            return min(value, 8)
    return 2


def _provider_health_admission(provider: str) -> threading.BoundedSemaphore:
    """Return one process-wide worker-lifetime admission bound per provider."""
    provider_key = canonical_provider_name(provider)
    with _PROVIDER_HEALTH_ADMISSIONS_LOCK:
        admission = _PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER.get(provider_key)
        if admission is None:
            admission = threading.BoundedSemaphore(
                provider_credential_validation_per_provider_capacity()
            )
            _PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER[provider_key] = admission
        return admission


GatewayVerificationStatus = Literal["verified", "stored-unverified", "rejected"]


class _GatewayCredentialRejected(Exception):
    pass


class _GatewayProbeUnavailable(Exception):
    pass


def _is_test_mode() -> bool:
    return is_test_mode() or os.getenv("PYTEST_CURRENT_TEST") is not None


def _provider_validation_runtime(
    *,
    provider: str,
    api_key: str | None,
    credential_fields: dict[str, Any] | None,
    app_config: dict[str, Any],
) -> ProviderCredentialRuntime:
    """Issue one execution-owned capability for a candidate credential test."""

    captured_fields = dict(credential_fields or {})
    captured_provider = canonical_provider_name(provider)

    async def resolve_candidate(
        normalized_provider: str,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        if canonical_provider_name(normalized_provider) != captured_provider:
            raise RuntimeError("Provider credential validation context is invalid")
        return ResolvedByokCredentials(
            provider=normalized_provider,
            api_key=api_key,
            app_config=app_config,
            credential_fields=captured_fields,
            source="none",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key" if api_key else None,
        )

    return ProviderCredentialRuntime(
        user_id=None,
        team_ids=(),
        org_ids=(),
        trusted_base_url_override=True,
        server_config_snapshot={},
        resolver=resolve_candidate,
    )


def is_obviously_invalid_key(api_key: str | None) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    lowered = key.lower()
    return any(lowered.startswith(prefix) for prefix in _INVALID_TEST_KEY_PREFIXES)


def resolve_default_model_for_provider(
    provider: str,
    *,
    include_override: bool = True,
) -> str | None:
    if include_override:
        try:
            from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
                get_llm_provider_override,
                get_override_default_model,
            )

            override_default = get_override_default_model(provider)
            if override_default:
                return override_default
            override = get_llm_provider_override(provider)
            if override and override.allowed_models:
                return override.allowed_models[0]
        except Exception as override_error:
            _ = override_error  # continue with default model fallback

    normalized = (provider or "").replace(".", "_").replace("-", "_")
    env_key = f"DEFAULT_MODEL_{normalized.upper()}"
    env_val = os.getenv(env_key)
    if isinstance(env_val, str) and env_val.strip():
        return env_val.strip()

    cfg = None
    try:
        cfg = load_comprehensive_config()
    except Exception:
        cfg = None

    if cfg is not None and getattr(cfg, "has_section", None) and cfg.has_section("Chat-Module"):
        config_key = f"default_model_{normalized.lower()}"
        try:
            cfg_val = cfg.get("Chat-Module", config_key, fallback=None)
        except Exception:
            cfg_val = None
        if isinstance(cfg_val, str) and cfg_val.strip():
            return cfg_val.strip()

    return None


def build_app_config_for_provider(provider: str, credential_fields: dict[str, Any] | None) -> dict[str, Any]:
    return build_app_config_overrides(provider, credential_fields)


async def _run_bounded_health_call(
    call: Callable[[], Any],
    *,
    provider: str,
    timeout_seconds: float,
) -> Any:
    """Run one sync provider check without releasing capacity before it exits."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("Daemon call timeout must be a positive finite number")

    global_admission = _PROVIDER_HEALTH_ADMISSION
    if global_admission is None:
        raise DaemonCapacityError("Provider health check capacity is exhausted")

    loop = asyncio.get_running_loop()
    admission_deadline = loop.time() + float(timeout_seconds)
    acquired_admissions: list[threading.BoundedSemaphore] = []

    async def acquire_admission(admission: threading.BoundedSemaphore) -> None:
        while not admission.acquire(blocking=False):
            remaining = admission_deadline - loop.time()
            if remaining <= 0:
                raise DaemonCapacityError(
                    "Provider health check capacity is exhausted"
                )
            await asyncio.sleep(min(0.01, remaining))
        acquired_admissions.append(admission)

    release_lock = threading.Lock()
    admissions_released = False

    def release_admissions() -> None:
        nonlocal admissions_released
        with release_lock:
            if admissions_released:
                return
            admissions_released = True
        for admission in reversed(acquired_admissions):
            admission.release()

    def invoke() -> Any:
        try:
            with sensitive_http_observability_context():
                return call()
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise ChatProviderError(
                    provider=provider,
                    message="Provider health check failed",
                    status_code=502,
                ) from None
            raise
        finally:
            release_admissions()

    worker_released = threading.Event()
    dispatch_attempted = False
    try:
        await acquire_admission(_provider_health_admission(provider))
        await acquire_admission(global_admission)
        dispatch_attempted = True
        return await await_bounded_daemon_with_timeout(
            invoke,
            pool=SYNC_ADAPTER_CALL_POOL,
            name=f"admin-provider-health-{provider}",
            timeout_seconds=timeout_seconds,
            timeout_message="Provider health check timed out",
            released_event=worker_released,
        )
    finally:
        if not dispatch_attempted or worker_released.is_set():
            release_admissions()


def _app_config_with_health_timeout(
    provider: str,
    app_config: dict[str, Any],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    """Overlay the local adapter timeout without mutating shared config state."""
    if (
        timeout_seconds is None
        or normalize_provider(provider) not in ChatProviderRegistry.DEFAULT_LOCAL_PROVIDERS
    ):
        return app_config

    section = resolve_provider_section(provider)
    provider_config = app_config.get(section)
    merged_config = dict(app_config)
    merged_config[section] = {
        **(provider_config if isinstance(provider_config, dict) else {}),
        "api_timeout": max(1, math.ceil(timeout_seconds)),
    }
    return merged_config


def _enforce_validation_endpoint_egress(endpoint: str) -> None:
    """Fail closed when an unauthenticated validation endpoint is not public."""
    from tldw_Server_API.app.core.Security import egress as egress_policy

    policy = egress_policy.evaluate_url_policy(
        endpoint,
        sensitive_observability=True,
    )
    if not policy.allowed:
        raise ChatConfigurationError(
            provider="provider-validation",
            message=PROVIDER_STREAM_ERROR_MESSAGES[
                "provider_configuration_invalid"
            ],
        )


def _validation_result_error(value: Any) -> Any | None:
    """Return the first bounded error in supported provider response fields."""
    seen: set[int] = set()

    def visit(item: Any) -> Any | None:
        normalized = normalize_provider_stream_error(item)
        if normalized is not None:
            return normalized
        if isinstance(item, str):
            if item.lstrip().lower().startswith("error:"):
                return sanitized_provider_stream_exception(None)
            return None
        if isinstance(item, bytes):
            return None
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return None
            seen.add(identity)
            for nested in item:
                found = visit(nested)
                if found is not None:
                    return found
            return None
        if not isinstance(item, dict):
            return None

        identity = id(item)
        if identity in seen:
            return None
        seen.add(identity)
        for field in ("choices", "message", "delta", "content"):
            if field not in item:
                continue
            found = visit(item[field])
            if found is not None:
                return found
        block_type = item.get("type")
        if block_type in {"text", "output_text"} and "text" in item:
            return visit(item["text"])
        return None

    return visit(value)


def _validation_result_is_meaningful(value: Any) -> bool:
    """Accept only non-empty text or supported OpenAI completion structures."""
    if isinstance(value, str):
        return bool(value.strip()) and not value.lstrip().lower().startswith("error:")
    if not isinstance(value, dict):
        return False

    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") not in {"text", "output_text"}:
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return True
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and any(
            isinstance(call, dict)
            and isinstance(call.get("function"), dict)
            and isinstance(call["function"].get("name"), str)
            and bool(call["function"]["name"].strip())
            for call in tool_calls
        ):
            return True
        function_call = message.get("function_call")
        if (
            isinstance(function_call, dict)
            and isinstance(function_call.get("name"), str)
            and bool(function_call["name"].strip())
        ):
            return True
    return False


def provider_validation_public_error(value: Any) -> SanitizedProviderStreamError:
    """Return the bounded public contract for a credential validation failure."""
    if isinstance(value, SanitizedProviderStreamError):
        safe_statuses = {
            "provider_authentication_failed": {401, 403, 502},
            "provider_configuration_invalid": {400, 500, 503},
            "missing_provider_credentials": {500, 503},
            "provider_unavailable": {429, 500, 502, 503, 504},
        }
        if value.status_code in safe_statuses.get(value.code, set()):
            return SanitizedProviderStreamError(
                code=value.code,
                message=PROVIDER_STREAM_ERROR_MESSAGES[value.code],
                status_code=(
                    502
                    if value.code == "provider_authentication_failed"
                    else value.status_code
                ),
            )
    if isinstance(value, ValueError):
        return SanitizedProviderStreamError(
            code="provider_configuration_invalid",
            message=PROVIDER_STREAM_ERROR_MESSAGES["provider_configuration_invalid"],
            status_code=400,
        )
    if isinstance(value, ChatAuthenticationError):
        return SanitizedProviderStreamError(
            code="provider_authentication_failed",
            message=PROVIDER_STREAM_ERROR_MESSAGES["provider_authentication_failed"],
            status_code=502,
        )
    if isinstance(value, ChatBadRequestError):
        return SanitizedProviderStreamError(
            code="provider_configuration_invalid",
            message=PROVIDER_STREAM_ERROR_MESSAGES["provider_configuration_invalid"],
            status_code=400,
        )
    if isinstance(value, ChatRateLimitError):
        return SanitizedProviderStreamError(
            code="provider_unavailable",
            message=PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"],
            status_code=429,
        )
    if isinstance(value, ChatConfigurationError):
        return SanitizedProviderStreamError(
            code=value.error_code,
            message=PROVIDER_STREAM_ERROR_MESSAGES[value.error_code],
            status_code=500,
        )
    if isinstance(value, ChatProviderError):
        status_code = value.status_code if value.status_code in {500, 502, 503, 504} else 502
        return SanitizedProviderStreamError(
            code="provider_unavailable",
            message=PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"],
            status_code=status_code,
        )
    return sanitized_provider_stream_exception(value)


async def probe_gateway_credentials(
    *,
    spec: Any,
    api_key: str,
) -> GatewayVerificationStatus:
    """Probe configured model discovery without issuing a synthesis request."""
    if (
        not bool(getattr(spec, "enabled", False))
        or not bool(getattr(getattr(spec, "discovery", None), "enabled", False))
        or not getattr(spec, "models_path", None)
    ):
        return "stored-unverified"

    async def _classify_status(status: int, _headers: Any) -> None:
        if status in {401, 403}:
            raise _GatewayCredentialRejected
        if status < 200 or status >= 300:
            raise _GatewayProbeUnavailable

    headers = dict(getattr(spec, "headers", ()) or ())
    headers["Authorization"] = f"Bearer {api_key}"
    try:
        payload = await afetch_json(
            method="GET",
            url=str(build_gateway_url(spec.base_url, spec.models_path)),
            params=dict(getattr(spec, "discovery_query", ()) or ()),
            headers=headers,
            timeout=spec.discovery.timeout_seconds,
            retry=RetryPolicy(attempts=1),
            require_json_ct=True,
            max_bytes=MAX_DISCOVERY_BYTES,
            allow_redirects=False,
            on_response=_classify_status,
        )
        _parse_discovered_models(payload)
    except _GatewayCredentialRejected:
        return "rejected"
    except Exception:
        return "stored-unverified"
    return "verified"


async def test_provider_credentials(
    *,
    provider: str,
    api_key: str | None,
    credential_fields: dict[str, Any] | None = None,
    app_config: dict[str, Any] | None = None,
    model: str | None = None,
    include_override_model: bool = True,
    timeout_seconds: float | None = None,
    enforce_egress_policy: bool = False,
    authoritative_endpoint: str | None = None,
) -> str | None:
    provider_norm = normalize_provider_name(provider)
    provider_registry_name = normalize_provider(provider_norm)
    adapter = get_registry().get_adapter(provider_registry_name)
    if adapter is None and provider_registry_name not in list_registered_providers():
        raise ValueError(f"Provider '{provider_norm}' does not support key tests yet")

    resolved_app_config = (
        app_config
        if app_config is not None
        else build_app_config_for_provider(provider_norm, credential_fields)
    )
    effective_timeout = (
        PROVIDER_CREDENTIAL_VALIDATION_TIMEOUT_SECONDS
        if timeout_seconds is None
        else timeout_seconds
    )
    resolved_app_config = _app_config_with_health_timeout(
        provider_registry_name,
        resolved_app_config,
        effective_timeout,
    )
    model_to_use = (
        model
        or resolve_provider_model(provider_registry_name, resolved_app_config)
        or resolve_default_model_for_provider(
            provider_norm,
            include_override=include_override_model,
        )
    )
    if not model_to_use and provider_requires_api_key(provider_registry_name):
        raise ValueError(
            f"Model is required for provider '{provider_norm}'. "
            f"Configure DEFAULT_MODEL_{provider_norm.replace('.', '_').replace('-', '_').upper()} or pass model."
        )

    if _is_test_mode():
        if is_obviously_invalid_key(api_key):
            raise ChatAuthenticationError(
                message=f"Provider '{provider_norm}' rejected the supplied credentials.",
                provider=provider_norm,
            )
        return model_to_use

    messages_payload = [{"role": "user", "content": "ping"}]
    result: Any
    credential_runtime = _provider_validation_runtime(
        provider=provider_registry_name,
        api_key=api_key,
        credential_fields=credential_fields,
        app_config=resolved_app_config,
    )

    def enforce_authoritative_egress() -> None:
        if not enforce_egress_policy:
            return
        if (
            not isinstance(authoritative_endpoint, str)
            or not authoritative_endpoint.strip()
        ):
            raise ChatConfigurationError(
                provider=provider_norm,
                message=PROVIDER_STREAM_ERROR_MESSAGES[
                    "provider_configuration_invalid"
                ],
            )
        _enforce_validation_endpoint_egress(authoritative_endpoint.strip())

    try:
        try:
            provider_credentials = await credential_runtime.resolve(
                provider_registry_name,
                model=model_to_use,
            )
            if adapter is not None:
                request = {
                    "messages": messages_payload,
                    "system_message": None,
                    "model": model_to_use,
                    "api_key": provider_credentials.api_key,
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "app_config": provider_credentials.app_config,
                    "credentials_resolved": True,
                    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: provider_credentials,
                }

                def call_adapter() -> Any:
                    enforce_authoritative_egress()
                    return adapter.chat(request, timeout=effective_timeout)

                result = await _run_bounded_health_call(
                    call_adapter,
                    provider=provider_norm,
                    timeout_seconds=effective_timeout,
                )
            else:
                def call_legacy_provider() -> Any:
                    enforce_authoritative_egress()
                    return chat_api_call(
                        api_endpoint=provider_norm,
                        messages_payload=messages_payload,
                        api_key=provider_credentials.api_key,
                        model=model_to_use,
                        temp=0.0,
                        max_tokens=1,
                        streaming=False,
                        app_config=provider_credentials.app_config,
                        credentials_resolved=True,
                        **{
                            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY:
                                provider_credentials,
                        },
                    )

                result = await _run_bounded_health_call(
                    call_legacy_provider,
                    provider=provider_norm,
                    timeout_seconds=effective_timeout,
                )
        except ChatAPIError as exc:
            raise_detached_error(provider_validation_public_error(exc))
        except Exception as exc:
            raise_detached_error(provider_validation_public_error(exc))

        result_error = _validation_result_error(result)
        if result_error is not None:
            del result
            raise_detached_error(sanitized_provider_stream_exception(result_error))
        result_is_meaningful = _validation_result_is_meaningful(result)
        del result
        if not result_is_meaningful:
            raise_detached_error(provider_validation_public_error(None))

        return model_to_use
    finally:
        await credential_runtime.close()
