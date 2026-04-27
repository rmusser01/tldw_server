# audio_health.py
# Description: Audio health endpoints.
import asyncio
from dataclasses import asdict, is_dataclass
import importlib.util
import os
import platform
import re
import sys
import time
from ctypes.util import find_library as _ctypes_find_library
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from starlette import status

from tldw_Server_API.app.api.v1.endpoints.audio.audio_tts import get_tts_service
from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.Audio.transcription_service import _map_openai_audio_model_to_whisper
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id
from tldw_Server_API.app.core.TTS.circuit_breaker import build_qwen_runtime_breaker_key
from tldw_Server_API.app.core.TTS.tts_config import get_tts_config_manager
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

router = APIRouter(
    tags=["Audio"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)

_AUTH_HEALTH_PROVIDERS = frozenset({"openai", "elevenlabs"})
_SUSPICIOUS_PUBLIC_HEALTH_RE = re.compile(
    r"traceback|stack(?:\s*trace)?|exception|\/Users\/|[A-Za-z]:\\|\.py:\d+",
    re.IGNORECASE,
)
_SANITIZED_PUBLIC_HEALTH_MESSAGE = "Internal health diagnostics were suppressed."
_SENSITIVE_PROVIDER_DETAIL_KEYS_RAW = frozenset(
    {
        "api_key",
        "apiKey",
        "auth_token",
        "authToken",
        "authorization",
        "base_url",
        "baseURL",
        "command",
        "cwd",
        "env",
        "exception",
        "host",
        "local_path",
        "path",
        "port",
        "repo_path",
        "repoPath",
        "stack",
        "stack_trace",
        "stackTrace",
        "stderr",
        "stdout",
        "token",
        "traceback",
        "working_dir",
    }
)
_PROVIDER_API_KEY_PLACEHOLDERS: dict[str, set[str]] = {
    "openai": {
        "",
        "none",
        "null",
        "<openai_api_key>",
        "your-openai-api-key",
        "your_openai_api_key",
        "sk-mock-key-12345",
    },
    "elevenlabs": {
        "",
        "none",
        "null",
        "<elevenlabs_api_key>",
        "<eleven_labs_api_key>",
        "your-elevenlabs-api-key",
        "your_elevenlabs_api_key",
    },
}


def _build_internal_health_request(path: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
    }
    return Request(scope)


def _sanitize_public_health_payload(value: Any) -> Any:
    if isinstance(value, str):
        return (
            _SANITIZED_PUBLIC_HEALTH_MESSAGE
            if _SUSPICIOUS_PUBLIC_HEALTH_RE.search(value)
            else value
        )
    if isinstance(value, list):
        return [_sanitize_public_health_payload(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"details", "exception", "traceback", "stack", "stack_trace"}:
                continue
            sanitized[key] = _sanitize_public_health_payload(item)
        return sanitized
    return value


def _serialize_tts_caps_for_health(tts_service: TTSServiceV2, caps: Any) -> Any:
    if caps is None:
        return None
    serializer = getattr(tts_service, "_serialize_capabilities", None)
    if callable(serializer):
        try:
            return serializer(caps)
        except Exception:
            logger.debug("TTS health capabilities serialization failed via service helper")
    if isinstance(caps, dict):
        return dict(caps)
    try:
        dumped = model_dump_compat(caps)
        if isinstance(dumped, dict):
            return dumped
    except Exception:
        logger.debug("TTS health capabilities model dump failed")
    try:
        if is_dataclass(caps):
            return asdict(caps)
    except Exception:
        logger.debug("TTS health capabilities dataclass conversion failed")
    return None


def _sanitize_public_provider_detail(value: Any) -> Any:
    """Remove sensitive keys and suspicious runtime diagnostics from public TTS health details."""

    if isinstance(value, str):
        if _SUSPICIOUS_PUBLIC_HEALTH_RE.search(value):
            return _SANITIZED_PUBLIC_HEALTH_MESSAGE
        return value
    if isinstance(value, list):
        return [_sanitize_public_provider_detail(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _normalize_public_health_key(key)
            if normalized_key in _SENSITIVE_PROVIDER_DETAIL_KEYS:
                continue
            sanitized[key] = _sanitize_public_provider_detail(item)
        return sanitized
    return value


def _normalize_public_health_key(value: Any) -> str:
    """Normalize a provider-detail key before comparing it with the sensitive-key denylist."""

    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


_SENSITIVE_PROVIDER_DETAIL_KEYS = frozenset(
    _normalize_public_health_key(value) for value in _SENSITIVE_PROVIDER_DETAIL_KEYS_RAW
)


def _derive_omnivoice_supervisor_health(
    tts_service: Any,
    current_detail: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Summarize OmniVoice sidecar supervisor state without exposing process internals."""

    current_detail = current_detail if isinstance(current_detail, dict) else {}
    availability = str(current_detail.get("availability") or "").strip().lower()

    state = "disabled" if availability == "disabled" else "idle_stopped"
    last_error_code: str | None = None
    supervisor = getattr(tts_service, "_omnivoice_supervisor", None)
    if supervisor is None:
        return {
            "runtime": "sidecar",
            "sidecar_state": state,
        }

    if bool(getattr(supervisor, "_closing", False)):
        state = "shutting_down"
    else:
        process = getattr(supervisor, "_process", None)
        process_returncode = getattr(process, "returncode", None) if process is not None else None
        base_url = getattr(supervisor, "_base_url", None)
        last_activity_at = getattr(supervisor, "_last_activity_at", None)
        last_failure_at = getattr(supervisor, "last_failure_at", None)
        startup_backoff_seconds = float(getattr(supervisor, "_startup_backoff_seconds", 0.0) or 0.0)

        if process is not None and process_returncode is None and base_url and last_activity_at is not None:
            state = "ready"
        elif process is not None and process_returncode is None:
            state = "starting"
        elif last_failure_at is not None:
            now = time.time()
            if startup_backoff_seconds > 0 and (now - float(last_failure_at)) < startup_backoff_seconds:
                state = "degraded"
                last_error_code = "startup_backoff"
            else:
                state = "degraded"
                last_error_code = "startup_failed"
        elif process is not None and process_returncode is not None:
            state = "degraded"
            last_error_code = "startup_failed"

    metadata = {
        "runtime": "sidecar",
        "sidecar_state": state,
    }
    if last_error_code is not None:
        metadata["last_error_code"] = last_error_code
    return metadata


def _normalize_provider_api_key(provider_name: str, raw_key: Any) -> Optional[str]:
    normalized_provider = str(provider_name or "").strip().lower()
    text = str(raw_key or "").strip()
    if not text:
        return None
    if text.lower() in _PROVIDER_API_KEY_PLACEHOLDERS.get(normalized_provider, {"", "none", "null"}):
        return None
    return text


def _load_auth_provider_configs() -> tuple[bool, dict[str, Any]]:
    try:
        config_manager = get_tts_config_manager()
        return True, {
            provider_name: config_manager.get_provider_config(provider_name)
            for provider_name in _AUTH_HEALTH_PROVIDERS
        }
    except Exception:
        logger.debug("TTS health auth config lookup failed")
        return False, {}


def _load_detailed_circuit_breakers(tts_service: Any) -> dict[str, Any]:
    circuit_manager = getattr(tts_service, "circuit_manager", None)
    if circuit_manager is None or not hasattr(circuit_manager, "get_all_status"):
        return {}
    try:
        detailed = circuit_manager.get_all_status(detailed=True)
        return detailed if isinstance(detailed, dict) else {}
    except Exception:
        logger.debug("TTS health detailed circuit-breaker lookup failed")
        return {}


def _provider_has_auth_failure(
    provider_name: str,
    detailed_circuit_breakers: dict[str, Any],
) -> bool:
    normalized_provider = str(provider_name or "").strip().lower()
    if not normalized_provider:
        return False

    for breaker_key, breaker_status in detailed_circuit_breakers.items():
        normalized_key = str(breaker_key or "").strip().lower()
        if normalized_key != normalized_provider and not normalized_key.startswith(
            f"{normalized_provider}:"
        ):
            continue
        if not isinstance(breaker_status, dict):
            continue
        error_analysis = breaker_status.get("error_analysis")
        if not isinstance(error_analysis, dict):
            continue
        error_categories = error_analysis.get("error_categories")
        if not isinstance(error_categories, dict):
            continue
        try:
            if int(error_categories.get("authentication") or 0) > 0:
                return True
        except Exception:
            if error_categories.get("authentication"):
                return True
    return False


def _recompute_health_rollup(
    health: dict[str, Any],
    provider_details: dict[str, Any],
    capability_envelopes: list[dict[str, Any]],
) -> None:
    available_providers = (
        sum(1 for entry in capability_envelopes if entry.get("availability") == "enabled")
        if capability_envelopes
        else sum(
            1
            for details in provider_details.values()
            if isinstance(details, dict) and details.get("availability") == "enabled"
        )
    )
    health["providers"]["available"] = available_providers
    health["status"] = "healthy" if available_providers > 0 else "unhealthy"


def _enrich_external_provider_auth_health(
    health: dict[str, Any],
    provider_details: dict[str, Any],
    capability_envelopes: list[dict[str, Any]],
    tts_service: Any,
) -> None:
    configs_loaded, provider_configs = _load_auth_provider_configs()
    detailed_circuit_breakers = _load_detailed_circuit_breakers(tts_service)
    updated = False

    for provider_name in _AUTH_HEALTH_PROVIDERS:
        detail = provider_details.get(provider_name)
        matching_entries = [
            entry
            for entry in capability_envelopes
            if str(entry.get("provider") or "").strip().lower() == provider_name
        ]

        if not isinstance(detail, dict) and not matching_entries:
            continue

        current_availability = (
            str(detail.get("availability") or "").strip().lower()
            if isinstance(detail, dict)
            else ""
        )
        if not current_availability and matching_entries:
            current_availability = str(
                matching_entries[0].get("availability") or ""
            ).strip().lower()

        auth_configured: Optional[bool] = None
        auth_reason: Optional[str] = None
        if configs_loaded:
            provider_cfg = provider_configs.get(provider_name)
            configured_key = _normalize_provider_api_key(
                provider_name,
                getattr(provider_cfg, "api_key", None) if provider_cfg is not None else None,
            )
            auth_configured = configured_key is not None
            if current_availability == "enabled" and not auth_configured:
                auth_reason = "api_key_missing"

        if _provider_has_auth_failure(provider_name, detailed_circuit_breakers):
            auth_reason = "authentication_failed"

        auth_ready: Optional[bool]
        if auth_reason is not None:
            auth_ready = False
        elif auth_configured is not None:
            auth_ready = auth_configured
        else:
            auth_ready = None

        if isinstance(detail, dict):
            if auth_configured is not None:
                detail["auth_configured"] = auth_configured
            if auth_ready is not None:
                detail["auth_ready"] = auth_ready
            if auth_reason is not None or "auth_reason" in detail:
                detail["auth_reason"] = auth_reason
            if auth_reason is not None and current_availability == "enabled":
                detail["status"] = "unhealthy"
                detail["availability"] = "unhealthy"
                detail["failed"] = True
                updated = True

        for entry in matching_entries:
            if auth_configured is not None:
                entry["auth_configured"] = auth_configured
            if auth_ready is not None:
                entry["auth_ready"] = auth_ready
            if auth_reason is not None or "auth_reason" in entry:
                entry["auth_reason"] = auth_reason
            if auth_reason is not None and str(entry.get("availability") or "").strip().lower() == "enabled":
                entry["availability"] = "unhealthy"
                updated = True

    if updated:
        _recompute_health_rollup(health, provider_details, capability_envelopes)


def _module_spec_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _discover_kokoro_espeak_library(adapter: Any) -> Optional[str]:
    try:
        config = getattr(adapter, "config", {}) or {}
    except Exception:
        config = {}

    configured_path = config.get("kokoro_espeak_lib") if isinstance(config, dict) else None
    if configured_path and os.path.exists(str(configured_path)):
        return str(configured_path)

    env_path = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
    if env_path and os.path.exists(env_path):
        return env_path

    sys_plat = sys.platform
    candidates: list[str] = []
    if sys_plat == "darwin":
        candidates = [
            "/opt/homebrew/lib/libespeak-ng.dylib",
            "/usr/local/lib/libespeak-ng.dylib",
            "/opt/local/lib/libespeak-ng.dylib",
        ]
    elif sys_plat.startswith("linux"):
        arch = platform.machine() or ""
        candidates = [
            f"/usr/lib/{arch}/libespeak-ng.so.1" if arch else "",
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
            "/usr/lib64/libespeak-ng.so.1",
            "/usr/lib/libespeak-ng.so.1",
            "/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/lib/aarch64-linux-gnu/libespeak-ng.so.1",
            "/lib/libespeak-ng.so.1",
        ]
    elif sys_plat in ("win32", "cygwin"):
        pf = os.environ.get("PROGRAMFILES", r"C:\\Program Files")
        pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\\Program Files (x86)")
        candidates = [
            os.path.join(pf, "eSpeak NG", "libespeak-ng.dll"),
            os.path.join(pf86, "eSpeak NG", "libespeak-ng.dll"),
        ]
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if directory:
                candidates.append(os.path.join(directory, "libespeak-ng.dll"))

    try:
        discovered_name = _ctypes_find_library("espeak-ng") or _ctypes_find_library("espeak")
        if discovered_name and os.path.isabs(discovered_name) and os.path.exists(discovered_name):
            candidates.insert(0, discovered_name)
    except Exception:
        logger.debug("Unable to discover eSpeak library via ctypes lookup")

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _evaluate_kokoro_runtime(adapter: Any) -> tuple[bool, Optional[str], dict[str, Any]]:
    backend = "onnx" if getattr(adapter, "use_onnx", True) else "pytorch"
    diagnostics: dict[str, Any] = {"backend": backend}

    if backend == "onnx":
        has_backend = _module_spec_available("kokoro_onnx")
        espeak_path = _discover_kokoro_espeak_library(adapter)
        diagnostics["kokoro_onnx_available"] = has_backend
        diagnostics["espeak_lib_path"] = _sanitize_health_path_value(espeak_path)
        diagnostics["espeak_lib_exists"] = bool(espeak_path and os.path.exists(espeak_path))
        if not has_backend:
            return False, "kokoro_onnx_missing", diagnostics
        if not diagnostics["espeak_lib_exists"]:
            return False, "espeak_missing", diagnostics
        return True, None, diagnostics

    has_pipeline = _module_spec_available("kokoro.pipeline")
    diagnostics["kokoro_pipeline_available"] = has_pipeline
    if not has_pipeline:
        return False, "kokoro_pipeline_missing", diagnostics
    return True, None, diagnostics


def _sanitize_health_path_value(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    expanded = os.path.expanduser(text)
    if os.path.isabs(expanded):
        name = os.path.basename(expanded)
        return name or "<redacted>"
    return text


@router.get("/health")
async def get_tts_health(request: Request, tts_service: TTSServiceV2 = Depends(get_tts_service)):
    """
    Get health status of TTS providers.
    """
    from datetime import datetime

    try:
        status_data = tts_service.get_status()
        if not isinstance(status_data, dict):
            logger.warning("TTS service returned non-mapping status; falling back to defaults")
            status_map = {}
        else:
            status_map = status_data

        capabilities = await tts_service.get_capabilities()
        if not isinstance(capabilities, dict):
            capabilities = {}
        provider_details = status_map.get("providers", {})
        if not isinstance(provider_details, dict):
            provider_details = {}
        else:
            provider_details = {
                provider_name: _sanitize_public_provider_detail(detail)
                for provider_name, detail in provider_details.items()
            }
        capability_envelopes: list[dict[str, Any]] = []

        available_providers = status_map.get("available", 0)
        total_providers = status_map.get("total_providers", 0)

        factory = None
        try:
            from tldw_Server_API.app.core.TTS.adapter_registry import get_tts_factory

            factory = await get_tts_factory()
            registry = getattr(factory, "registry", None)
            list_caps = getattr(registry, "list_capabilities", None)
            if callable(list_caps):
                entries = list_caps(include_disabled=True)
                if asyncio.iscoroutine(entries):
                    entries = await entries
                if isinstance(entries, list):
                    for raw_entry in entries:
                        if not isinstance(raw_entry, dict):
                            continue
                        provider_name = str(raw_entry.get("provider") or "").strip()
                        if not provider_name:
                            continue
                        availability = str(raw_entry.get("availability") or "unknown").strip().lower() or "unknown"
                        serialized_caps = _serialize_tts_caps_for_health(tts_service, raw_entry.get("capabilities"))
                        metadata = {}
                        if isinstance(serialized_caps, dict):
                            metadata = dict(serialized_caps.get("metadata") or {})
                        runtime_name = metadata.get("runtime")
                        if provider_name == "omnivoice" and not runtime_name:
                            runtime_name = "sidecar"
                        breaker_key = provider_name
                        if provider_name == "qwen3_tts" and runtime_name:
                            breaker_key = build_qwen_runtime_breaker_key(provider_name, str(runtime_name))
                        capability_envelopes.append(
                            {
                                "provider": provider_name,
                                "availability": availability,
                                "runtime": runtime_name,
                                "breaker_key": breaker_key,
                                "capabilities": serialized_caps,
                            }
                        )
                        if serialized_caps is not None and provider_name not in capabilities:
                            capabilities[provider_name] = serialized_caps
                        current_details = provider_details.get(provider_name)
                        if isinstance(current_details, dict):
                            current_details.setdefault("availability", availability)
                            current_details.setdefault("status", availability)
                            if runtime_name:
                                current_details.setdefault("runtime", runtime_name)
                            current_details.setdefault("breaker_key", breaker_key)
                        else:
                            provider_details[provider_name] = {
                                "status": availability,
                                "availability": availability,
                                "runtime": runtime_name,
                                "breaker_key": breaker_key,
                                "initialized": False,
                                "failed": availability == "failed",
                            }
        except Exception:
            logger.debug("TTS health envelope enrichment failed")

        if capability_envelopes:
            if not total_providers:
                total_providers = len(capability_envelopes)
            if not available_providers:
                available_providers = sum(
                    1 for entry in capability_envelopes if entry.get("availability") == "enabled"
                )

        health_status = "healthy" if available_providers > 0 else "unhealthy"

        health = {
            "status": health_status,
            "providers": {
                "total": total_providers,
                "available": available_providers,
                "details": provider_details,
            },
            "circuit_breakers": status_map.get("circuit_breakers", {}),
            "capabilities": capabilities,
            "capabilities_envelope": capability_envelopes,
            "timestamp": datetime.utcnow().isoformat(),
        }

        _enrich_external_provider_auth_health(
            health,
            provider_details,
            capability_envelopes,
            tts_service,
        )

        omnivoice_detail = provider_details.get("omnivoice")
        if not isinstance(omnivoice_detail, dict):
            omnivoice_detail = {}
            provider_details["omnivoice"] = omnivoice_detail
        omnivoice_runtime = _derive_omnivoice_supervisor_health(tts_service, omnivoice_detail)
        if omnivoice_runtime:
            omnivoice_detail.update(omnivoice_runtime)
            sidecar_state = str(omnivoice_runtime.get("sidecar_state") or "").strip().lower()
            if sidecar_state == "degraded" or omnivoice_runtime.get("last_error_code"):
                omnivoice_detail["status"] = "degraded"
                omnivoice_detail["availability"] = "degraded"
                omnivoice_detail["failed"] = True
                for entry in capability_envelopes:
                    if str(entry.get("provider") or "").strip().lower() != "omnivoice":
                        continue
                    entry["status"] = "degraded"
                    entry["availability"] = "degraded"
                    entry["failed"] = True
                _recompute_health_rollup(health, provider_details, capability_envelopes)

        try:
            from tldw_Server_API.app.core.TTS.adapter_registry import TTSProvider

            if factory is None:
                from tldw_Server_API.app.core.TTS.adapter_registry import get_tts_factory

                factory = await get_tts_factory()
            adapter = await factory.registry.get_adapter(TTSProvider.KOKORO)
            if adapter:
                backend = "onnx" if getattr(adapter, "use_onnx", True) else "pytorch"
                runtime_ready, runtime_reason, runtime_diagnostics = _evaluate_kokoro_runtime(adapter)
                kokoro_info = {
                    "backend": backend,
                    "device": str(getattr(adapter, "device", "unknown")),
                    "model_path": _sanitize_health_path_value(getattr(adapter, "model_path", None)),
                    "voices_json": _sanitize_health_path_value(getattr(adapter, "voices_json", None)),
                    "runtime_ready": runtime_ready,
                    "runtime_reason": runtime_reason,
                }
                try:
                    es_env = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
                    kokoro_info["espeak_lib_env"] = _sanitize_health_path_value(es_env)
                    kokoro_info["espeak_lib_path"] = runtime_diagnostics.get("espeak_lib_path")
                    kokoro_info["espeak_lib_exists"] = bool(runtime_diagnostics.get("espeak_lib_exists"))
                except Exception:
                    logger.debug("Kokoro health eSpeak library introspection failed")
                kokoro_detail = provider_details.get("kokoro")
                if not isinstance(kokoro_detail, dict):
                    kokoro_detail = {}
                    provider_details["kokoro"] = kokoro_detail
                kokoro_detail["runtime_ready"] = runtime_ready
                kokoro_detail["runtime_reason"] = runtime_reason
                if not runtime_ready:
                    kokoro_detail["status"] = "unhealthy"
                    kokoro_detail["availability"] = "unhealthy"
                    kokoro_detail["failed"] = True

                for entry in capability_envelopes:
                    if str(entry.get("provider") or "").strip().lower() != "kokoro":
                        continue
                    entry["runtime_ready"] = runtime_ready
                    entry["runtime_reason"] = runtime_reason
                    if not runtime_ready:
                        entry["availability"] = "unhealthy"

                _recompute_health_rollup(health, provider_details, capability_envelopes)
                health["providers"]["kokoro"] = kokoro_info
        except Exception:
            logger.debug("Kokoro health enrichment failed")

        return health
    except Exception as e:
        logger.error("Error getting TTS health")
        request_id = ensure_request_id(request)
        payload = _http_error_detail("TTS health check failed", request_id, exc=e)
        return {"status": "error", **payload, "timestamp": datetime.utcnow().isoformat()}


async def collect_setup_tts_health() -> dict[str, Any]:
    """Collect TTS health without going through the HTTP routing layer."""

    request = _build_internal_health_request("/api/v1/audio/health")
    try:
        tts_service = await get_tts_service()
        return await get_tts_health(request, tts_service)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        message = detail.get("message")
        if not isinstance(message, str) or not message.strip():
            raw_error = detail.get("error")
            message = raw_error if isinstance(raw_error, str) else None
        if not isinstance(message, str) or not message.strip():
            message = "TTS health check failed"
        return {
            "status": "error",
            "providers": {"total": 0, "available": 0, "details": {}},
            "message": message,
            "error": detail,
            "status_code": exc.status_code,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("TTS setup health collection failed")
        request_id = ensure_request_id(request)
        payload = _http_error_detail("TTS health check failed", request_id, exc=exc)
        return {
            "status": "error",
            "providers": {"total": 0, "available": 0, "details": {}},
            **payload,
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        }


@router.get("/transcriptions/health", summary="Check STT transcription model health")
async def get_stt_health(
    request: Request,
    model: Optional[str] = Query(
        default=None,
        description=(
            "Transcription model to check (OpenAI-style id or internal STT model name). "
            "Defaults to the configured STT provider when omitted."
        ),
    ),
    warm: bool = Query(
        default=False,
        description="If true and the provider is Whisper, eagerly load the model to verify it can be initialized.",
    ),
):
    """
    Lightweight health/readiness endpoint for STT models.
    """
    from datetime import datetime

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as stt_lib
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files

    request_id = ensure_request_id(request)
    raw_model = (model or "").strip()
    if not raw_model:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
            resolve_default_transcription_model,
        )

        raw_model = resolve_default_transcription_model("whisper-1")
    provider_raw, _, _ = stt_lib.parse_transcription_model(raw_model)

    if provider_raw == "whisper":
        resolved_model = _map_openai_audio_model_to_whisper(raw_model)
        try:
            resolved_model = stt_lib.validate_whisper_model_identifier(resolved_model)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_http_error_detail("Invalid transcription model identifier", request_id, exc=exc),
            ) from exc
    else:
        resolved_model = raw_model

    try:
        status_info = audio_files.check_transcription_model_status(resolved_model)
    except Exception:
        logger.exception("STT health: check_transcription_model_status failed")
        status_info = {
            "available": False,
            "message": "Failed to check model status.",
            "model": resolved_model,
        }

    health: dict[str, Any] = {
        "provider": provider_raw,
        "alias": raw_model,
        "model": status_info.get("model", resolved_model),
        "available": bool(status_info.get("available", False)),
        "usable": bool(status_info.get("usable", status_info.get("available", False))),
        "on_demand": bool(status_info.get("on_demand", False)),
        "message": status_info.get("message"),
        "estimated_size": status_info.get("estimated_size"),
        "timestamp": datetime.utcnow().isoformat(),
    }

    warm_info: dict[str, Any] = {}
    if warm and provider_raw == "whisper":
        device = getattr(stt_lib, "processing_choice", "cpu")
        try:
            stt_lib.get_whisper_model(resolved_model, device, check_download_status=False)
            warm_info = {"ok": True, "device": device}
            # Warm-up succeeded, so the model is ready to serve requests.
            health["available"] = True
            health["usable"] = True
            health["on_demand"] = False
            health["message"] = f"Model {resolved_model} is available and ready for use"
            health["estimated_size"] = None
        except Exception:
            logger.exception("STT health warm-up failed")
            warm_info = {
                "ok": False,
                "device": device,
                "error": "Model initialization failed.",
            }

    if warm_info:
        health["warm"] = warm_info

    return _sanitize_public_health_payload(health)


async def collect_setup_stt_health(
    model: Optional[str] = None,
    *,
    warm: bool = False,
) -> dict[str, Any]:
    """Collect STT health without going through the HTTP routing layer."""

    request = _build_internal_health_request("/api/v1/audio/transcriptions/health")
    try:
        return await get_stt_health(request, model=model, warm=warm)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        detail = _sanitize_public_health_payload(detail)
        message = detail.get("message")
        if not isinstance(message, str) or not message.strip():
            raw_error = detail.get("error")
            message = raw_error if isinstance(raw_error, str) else None
        if not isinstance(message, str) or not message.strip():
            message = "Failed to collect STT health."
        return {
            "provider": None,
            "alias": model,
            "model": model,
            "available": False,
            "usable": False,
            "on_demand": False,
            "message": message,
            "error": (
                detail.get("error")
                if isinstance(detail, dict) and isinstance(detail.get("error"), str)
                else None
            ),
            "status_code": exc.status_code,
        }
