# Embeddings_Create.py
#
# Description: Functions for Creating and managing Embeddings in ChromaDB with LLama.cpp/OpenAI/Transformers
#
# Imports
from __future__ import annotations

#
import asyncio
import configparser
import hashlib
import json
import os
import re
import threading
import time
import warnings
import weakref
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

#
# Third-party Libraries
import numpy as np
from loguru import logger
from prometheus_client import REGISTRY, Counter, Gauge  # Assuming these are defined elsewhere or used directly
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from transformers import AutoModel, AutoTokenizer

# NOTE: Avoid importing heavy deps (torch, transformers) at module import time.
# Import them lazily inside functions/methods when needed to keep app import light.

_EMBEDDINGS_IMPORT_EXCEPTIONS = (ImportError, OSError, RuntimeError)


def _import_torch():
    """Lazily import torch only when actually needed."""
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:
        # Defer error to call site with a clearer message
        raise ImportError("'torch' is required for this embeddings provider. Install torch to proceed.") from e


def _import_transformers():
    """Lazily import transformers AutoModel/AutoTokenizer only when needed."""
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        return AutoModel, AutoTokenizer
    except _EMBEDDINGS_IMPORT_EXCEPTIONS as e:
        raise ImportError(
            "'transformers' is required for this embeddings provider. Install transformers to proceed."
        ) from e


_OPTIMUM_IMPORT_ERROR: Exception | None = None


def _import_optimum_ort_model():
    """Lazily import optimum ORT model class only when conversion is needed."""
    global _OPTIMUM_IMPORT_ERROR
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction as _ORTModelForFeatureExtraction  # type: ignore

        _OPTIMUM_IMPORT_ERROR = None
        return _ORTModelForFeatureExtraction
    except _EMBEDDINGS_IMPORT_EXCEPTIONS as e:
        _OPTIMUM_IMPORT_ERROR = e
        raise ImportError(
            "'optimum[onnxruntime]' is required for ONNX conversion. Install optimum with onnxruntime support."
        ) from e


_ORT_IMPORT_ERROR: Exception | None = None
try:
    import onnxruntime as ort  # type: ignore
except _EMBEDDINGS_IMPORT_EXCEPTIONS as e:
    ort = None  # type: ignore[assignment]
    _ORT_IMPORT_ERROR = e


def _import_onnxruntime():
    """Lazily import onnxruntime only when actually needed."""
    global ort, _ORT_IMPORT_ERROR
    if ort is not None:
        return ort
    try:
        import onnxruntime as _ort  # type: ignore

        ort = _ort
        _ORT_IMPORT_ERROR = None
        return _ort
    except _EMBEDDINGS_IMPORT_EXCEPTIONS as e:
        _ORT_IMPORT_ERROR = e
        raise ImportError(
            "'onnxruntime' is required for the ONNX embeddings provider. Install onnxruntime to proceed."
        ) from e


#
# Local Imports
import contextlib

from tldw_Server_API.app.core.config import resolve_repo_relative_path, rg_policy_path
from tldw_Server_API.app.core.Embeddings.audit_adapter import (
    log_memory_limit_exceeded,
    log_model_evicted,
)
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError, NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.LLM_Calls.chat_calls import get_openai_embeddings_batch
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram  # Keep your existing metrics
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode
from tldw_Server_API.app.core.Utils.path_utils import safe_join
from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt

_EMBEDDINGS_NONCRITICAL_EXCEPTIONS = (
    ImportError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    json.JSONDecodeError,
    re.error,
    InvalidStoragePathError,
    NetworkError,
    RetryExhaustedError,
)

#
########################################################################################################################
#
# Stuff:
# NOTE: Do not import `optimum` at module import time.
# Some environments crash when optional torch-backed deps initialize eagerly.

COMMIT_HASHES: dict[str, str] = {
    "jinaai/jina-embeddings-v3": "4be32c2f5d65b95e4bcce473545b7883ec8d2edd",
    "Alibaba-NLP/gte-large-en-v1.5": "104333d6af6f97649377c2afbde10a7704870c7b",
    "dunzhang/setll_en_400M_v5": "2aa5579fcae1c579de199a3866b6e514bbbf5d10",
}

_CACHE_SUBDIR_PATTERN = re.compile(r"[^0-9A-Za-z_.-]+")
_ALLOWLIST_ENV_VAR = "EMBEDDINGS_STORAGE_ALLOWLIST_ROOT"
_allowlist_root_env = (os.environ.get(_ALLOWLIST_ENV_VAR) or "").strip()
_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT = Path(_allowlist_root_env or resolve_repo_relative_path("models")).resolve(
    strict=False
)


def _get_http_status_from_exception(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status = getattr(response, "status_code", None)
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _is_probable_network_error(exc: Exception) -> bool:
    if isinstance(exc, (NetworkError, TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__
    if "Timeout" in name or "Connection" in name or "Connect" in name:
        return True
    msg = str(exc).lower()
    return bool("timed out" in msg or "timeout" in msg or "connection" in msg or "dns" in msg)


def _is_request_exception(exc: Exception) -> bool:
    name = type(exc).__name__
    module = getattr(exc, "__module__", "") or ""
    if module.startswith("requests."):
        return True
    if name in {"RequestException", "HTTPError"} or name.endswith("RequestException"):
        return True
    if "HTTPError" in name:
        return True
    return _get_http_status_from_exception(exc) is not None


def _model_cache_subdir_name(model_id: str) -> str:
    """
    Return a filesystem-safe subdirectory name for caching model artifacts.

    The name retains enough of the original identifier for debugging while
    appending an 8-char hash suffix to avoid collisions. All characters are
    limited to a portable ASCII set so the path works across platforms
    (including Windows, where ':' is illegal).
    """
    sanitized = _CACHE_SUBDIR_PATTERN.sub("_", model_id).strip("._")
    if not sanitized:
        sanitized = "model"
    if len(sanitized) > 80:
        sanitized = sanitized[:80].rstrip("._-")
        if not sanitized:
            sanitized = "model"
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}-{digest}"


def _synthetic_test_embedding(text: str, dims: int = 384) -> list[float]:
    """Return a deterministic normalized embedding for in-process test flows."""
    if dims <= 0:
        return []
    tokens = re.findall(r"[0-9A-Za-z_]+", str(text or "").lower())
    if not tokens:
        return [0.0] * dims
    vec = [0.0] * dims
    for token in tokens:
        token_hash = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
        vec[int.from_bytes(token_hash[:8], byteorder="big", signed=False) % dims] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return [float(val) / norm for val in vec]


def _should_use_inprocess_test_embeddings(provider: str) -> bool:
    return provider.strip().lower() == "huggingface" and is_test_mode() and env_flag_enabled("E2E_INPROCESS")


def _log_rejected_path(
    label: str,
    value: str,
    reason: str,
    *,
    resolved: str | None = None,
    base: str | None = None,
) -> None:
    """Log and count a rejected storage path value with optional context."""
    trimmed = (value or "").strip()
    if len(trimmed) > 200:
        trimmed = trimmed[:200] + "..."
    log_counter(
        "embeddings_storage_path_rejected",
        labels={"label": label, "reason": reason},
    )
    if resolved or base:
        logger.warning(
            "Rejected {}: {} ({}) resolved={} base={}",
            label,
            trimmed,
            reason,
            resolved or "",
            base or "",
        )
        return
    logger.warning("Rejected {}: {} ({})", label, trimmed, reason)


def _normalize_model_storage_base_dir(base_dir: str) -> str:
    """Normalize and validate embedding storage base dir under the allowlist root."""
    base_dir_str = str(base_dir or "").strip()
    if not base_dir_str:
        _log_rejected_path("model_storage_base_dir", base_dir_str, "empty")
        raise InvalidStoragePathError("model_storage_base_dir must be a non-empty string.")
    if "\x00" in base_dir_str:
        _log_rejected_path("model_storage_base_dir", base_dir_str, "nul_byte")
        raise InvalidStoragePathError("model_storage_base_dir contains invalid characters.")
    resolved = Path(resolve_repo_relative_path(base_dir_str)).resolve(strict=False)
    try:
        resolved.relative_to(_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT)
    except ValueError as exc:
        _log_rejected_path(
            "model_storage_base_dir",
            base_dir_str,
            "outside_allowlist",
            resolved=str(resolved),
            base=str(_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT),
        )
        raise InvalidStoragePathError(
            f"model_storage_base_dir must be within {_EMBEDDINGS_STORAGE_ALLOWLIST_ROOT}."
        ) from exc
    return str(resolved)


def _safe_model_storage_subdir(base_dir: str, subpath: str, label: str) -> str:
    """Resolve and validate a storage subpath under base_dir."""
    if not isinstance(subpath, str) or not subpath.strip():
        _log_rejected_path(label, str(subpath), "empty", base=base_dir)
        raise InvalidStoragePathError(f"{label} must be a non-empty relative path.")
    if "\x00" in subpath:
        _log_rejected_path(label, subpath, "nul_byte", base=base_dir)
        raise InvalidStoragePathError(f"{label} contains invalid characters.")

    def _path_error(_: Exception | None) -> Exception:
        candidate = ""
        try:
            candidate = os.path.abspath(os.path.join(base_dir, subpath))
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            candidate = ""
        _log_rejected_path(
            label,
            subpath,
            "outside_base_dir",
            resolved=candidate or None,
            base=base_dir,
        )
        return InvalidStoragePathError(f"{label} must be a relative path within model_storage_base_dir.")

    return safe_join(base_dir, subpath, error_factory=_path_error)


def resolve_model_storage_base_dir(
    embedding_settings: dict[str, Any] | None = None,
    default: str | None = None,
) -> str:
    """
    Determine the base directory used to persist embedding model artifacts.

    Preference order:
        1. Explicit override on the provided embedding_settings mapping.
        2. Global settings["EMBEDDINGS_MODEL_STORAGE_DIR"].
        3. Environment variable EMBEDDINGS_MODEL_STORAGE_DIR.
        4. Supplied default argument.
        5. Project default ./models/embedding_models_data/
    """
    from tldw_Server_API.app.core.config import settings

    embedding_settings = embedding_settings or settings.get("EMBEDDING_CONFIG", {}) or {}
    candidate = embedding_settings.get("model_storage_base_dir")
    if candidate:
        return str(candidate)

    try:
        configured_dir = settings.get("EMBEDDINGS_MODEL_STORAGE_DIR")
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Failed to read EMBEDDINGS_MODEL_STORAGE_DIR from settings: {}", exc)
        configured_dir = None
    if configured_dir:
        return str(configured_dir)

    env_dir = os.getenv("EMBEDDINGS_MODEL_STORAGE_DIR")
    if env_dir:
        return env_dir

    if default:
        return str(default)

    return "./models/embedding_models_data/"


# Default resource limits
DEFAULT_MAX_MODELS = 3
DEFAULT_MAX_MEMORY_GB = 8.0
DEFAULT_LRU_TTL_SECONDS = 3600


def _get_config_value(cfg, key: str, default):
    """Safely retrieve a value from config, supporting both ConfigParser and dict."""
    if cfg is None:
        return default
    try:
        return cfg.get(key, fallback=default)
    except TypeError:
        return cfg.get(key, default)
    except (AttributeError, KeyError, ValueError, configparser.Error) as exc:
        logger.debug(f"Config read failed for {key}: {exc}")
        return default


def _coerce_int(value, default: int) -> int:
    """Coerce value to int with fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float) -> float:
    """Coerce value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Resource limits - loaded from config or use defaults
def get_resource_limits():
    """Get resource limits from config file."""
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config

        config = load_comprehensive_config()
        embeddings_config = None
        try:
            if config and hasattr(config, "has_section") and config.has_section("Embeddings"):
                embeddings_config = config["Embeddings"]
            elif isinstance(config, dict):
                embeddings_config = config.get("Embeddings")
        except (AttributeError, KeyError, TypeError, configparser.Error) as exc:
            logger.debug(f"Embeddings resource limits: failed to access Embeddings section: {exc}")
            embeddings_config = None

        return {
            "max_models": _coerce_int(
                _get_config_value(embeddings_config, "max_models_in_memory", DEFAULT_MAX_MODELS),
                DEFAULT_MAX_MODELS,
            ),
            "max_memory_gb": _coerce_float(
                _get_config_value(embeddings_config, "max_model_memory_gb", DEFAULT_MAX_MEMORY_GB),
                DEFAULT_MAX_MEMORY_GB,
            ),
            "lru_ttl_seconds": _coerce_int(
                _get_config_value(embeddings_config, "model_lru_ttl_seconds", DEFAULT_LRU_TTL_SECONDS),
                DEFAULT_LRU_TTL_SECONDS,
            ),
        }
    except (OSError, TypeError, ValueError, configparser.Error) as e:
        logger.warning(f"Could not load resource limits from config: {e}. Using defaults.")
        return {
            "max_models": DEFAULT_MAX_MODELS,
            "max_memory_gb": DEFAULT_MAX_MEMORY_GB,
            "lru_ttl_seconds": DEFAULT_LRU_TTL_SECONDS,
        }


RESOURCE_LIMITS = get_resource_limits()
MAX_MODELS_IN_MEMORY = RESOURCE_LIMITS["max_models"]
MAX_MODEL_MEMORY_GB = RESOURCE_LIMITS["max_memory_gb"]
MODEL_LRU_TTL_SECONDS = RESOURCE_LIMITS["lru_ttl_seconds"]

embedding_models: dict[str, Any] = {}
embedding_models_lock = threading.RLock()  # Global reentrant lock for the embedding_models dictionary
model_last_used: dict[str, float] = {}  # Track last usage time for LRU eviction
model_memory_usage: dict[str, float] = {}  # Track estimated memory per model
model_in_use_counts: dict[str, int] = {}  # Track active users of cached models


def _mark_model_in_use(model_id: str) -> None:
    """Increment the in-use counter for a cached model."""
    with embedding_models_lock:
        model_in_use_counts[model_id] = model_in_use_counts.get(model_id, 0) + 1


def _release_model_in_use(model_id: str) -> None:
    """Decrement the in-use counter for a cached model."""
    with embedding_models_lock:
        current = model_in_use_counts.get(model_id, 0)
        if current <= 1:
            model_in_use_counts.pop(model_id, None)
        else:
            model_in_use_counts[model_id] = current - 1


def _get_or_create_metric(metric_cls, name: str, documentation: str, labelnames: tuple[str, ...]):
    """Return an existing Prometheus collector when available, else create one.

    This makes metric registration idempotent across module re-imports in tests.
    """
    try:
        existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        if existing is not None:
            existing_labels = tuple(getattr(existing, "_labelnames", ()))
            if existing_labels == labelnames:
                return existing
            try:
                REGISTRY.unregister(existing)
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                # If unregister fails, fall through and let metric creation raise with context.
                pass
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        # Accessing registry internals is best-effort; metric creation will validate again.
        pass
    return metric_cls(name, documentation, labelnames=labelnames)


# Prometheus Metrics (Ensure these are correctly defined and registered in your application)
ACTIVE_EMBEDDERS = _get_or_create_metric(
    Gauge,
    "active_embedder_instances",
    "Number of active embedder instances",
    ("provider", "model_id"),
)
# Use a distinct metric name to avoid collisions with API-level embedding_requests_total.
EMBEDDINGS_REQUESTS = _get_or_create_metric(
    Counter,
    "embedding_backend_requests_total",
    "Total number of backend embedding requests",
    ("provider", "model_id"),
)
MODEL_CACHE_HITS = _get_or_create_metric(
    Counter,
    "embedding_model_cache_hits_total",
    "Total number of model cache hits",
    ("model_id",),
)


# Add other metrics from your previous version or as needed, e.g., for load times, creation times


class RetryCfg(BaseModel):
    max_retries: int = Field(3, ge=0)
    base_delay: int = Field(1, ge=0)


class RateLimiterCfg(BaseModel):
    max_calls: int = Field(20, ge=1)
    period: int = Field(60, ge=1)


class BaseModelCfg(BaseModel):
    provider: str
    model_name_or_path: str
    trust_remote_code: bool = False
    revision: str | None = None
    max_length: int = 512
    unload_timeout_seconds: int = 300


class HFModelCfg(BaseModelCfg):
    provider: Literal["huggingface"] = "huggingface"
    hf_cache_dir_subpath: str = "huggingface_cache"


class ONNXModelCfg(BaseModelCfg):
    provider: Literal["onnx"] = "onnx"
    onnx_storage_dir_subpath: str = "onnx_models"
    onnx_providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])


class OpenAIModelCfg(BaseModelCfg):
    provider: Literal["openai"] = "openai"
    api_key: str | None = None
    dimensions: int | None = None


class LocalAPICfg(BaseModelCfg):
    provider: Literal["local_api"] = "local_api"
    api_url: str
    api_key: str | None = None
    # Consider adding chunk_size for local_api batching
    # chunk_size: int = 100


ModelCfg = Annotated[
    HFModelCfg | ONNXModelCfg | OpenAIModelCfg | LocalAPICfg,
    Field(discriminator="provider"),
]


class EmbeddingConfigSchema(BaseModel):
    default_model_id: str
    model_storage_base_dir: str | None = Field(default="./models/embedding_models_data/")
    # These are currently NOT used by the global decorators.
    # If dynamic configuration is needed, decorators must be applied differently.
    rate_limiter: RateLimiterCfg = RateLimiterCfg()
    retry_config: RetryCfg = RetryCfg()
    models: dict[str, ModelCfg]


def _ensure_hf_revision(model_name_or_path: str, expected_sha: str | None) -> None:
    if expected_sha is None:
        logger.debug(f"No revision SHA provided for {model_name_or_path}, skipping check.")
        return
    try:
        try:
            from huggingface_hub import model_info as _model_info  # type: ignore
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as import_exc:
            logger.warning(
                "huggingface_hub not available; skipping revision verification for {}. "
                "Install huggingface_hub to enable commit hash checks. Error: {}",
                model_name_or_path,
                import_exc,
            )
            return
        info = _model_info(model_name_or_path, revision=expected_sha)  # Check against the specific revision
        actual_sha = info.sha
        if actual_sha != expected_sha:
            logger.error(
                f"SHA mismatch for model {model_name_or_path}. Expected: {expected_sha}, Got: {actual_sha}. "
                f"The model on Hugging Face Hub may have changed for this commit hash."
            )
            raise RuntimeError(
                f"SHA mismatch for model {model_name_or_path}. Expected: {expected_sha}, Got: {actual_sha}"
            )
        logger.info(f"Successfully verified revision SHA {expected_sha} for model {model_name_or_path}.")
    except OSError as os_err:
        logger.warning(
            f"Skipping Hugging Face revision verification for {model_name_or_path} due to local environment error: "
            f"{os_err}. Proceeding without remote validation."
        )
        return
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:  # Catch network errors or if model/revision not found
        if _is_probable_network_error(e):
            logger.warning(
                f"Skipping Hugging Face revision verification for {model_name_or_path} due to connectivity issue: "
                f"{e}. Proceeding with locally cached artifacts."
            )
            return
        if _is_request_exception(e):
            logger.exception(f"Failed to verify revision for {model_name_or_path} (SHA: {expected_sha}): {e}")
            raise RuntimeError(f"Failed to verify model revision for {model_name_or_path}: {e}") from e
        logger.exception(f"Failed to verify revision for {model_name_or_path} (SHA: {expected_sha}): {e}")
        # Decide if this should be a fatal error. For now, we'll raise to prevent using a potentially wrong model.
        raise RuntimeError(f"Failed to verify model revision for {model_name_or_path}: {e}") from e


_EMB_SERVER_DEPRECATION_WARNED = False


def _emit_emb_server_legacy_deprecation(context: str) -> None:
    global _EMB_SERVER_DEPRECATION_WARNED
    if _EMB_SERVER_DEPRECATION_WARNED:
        return
    _EMB_SERVER_DEPRECATION_WARNED = True
    msg = (
        "Embeddings server legacy rate limiter is deprecated (Phase 2). "
        f"Context: {context}. Enable RG_ENABLED=true for enforcement. "
        "This shim will be removed in a future release."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    logger.warning(msg)


class TokenBucketLimiter:
    """Inline token-bucket rate limiter for the embeddings server.

    **Phase 2 Deprecation Notice**:
    Primary enforcement is handled by ``RGSimpleMiddleware`` and the per-module
    RG integration. When RG is disabled, this shim fails open (no sleeps or
    counters). When RG is enabled but unavailable, it fails closed with a
    ``RuntimeError``. This shim will be removed in a future release.
    """

    def __init__(self, capacity: int, period: int):
        # Parameters preserved for API compatibility; no internal state.
        self.capacity = capacity
        self.period = period  # seconds
        logger.info(
            f"TokenBucketLimiter initialized (Phase 2 shim) with capacity {capacity} tokens per {period} seconds."
        )

    def acquire(self) -> None:
        """Acquire a token, honoring ResourceGovernor if enabled."""
        if not _rg_embeddings_server_enabled():
            _emit_emb_server_legacy_deprecation("rg_disabled")
            return

        # RG enforcement with simple backoff based on retry_after.
        while True:
            decision = _maybe_enforce_with_rg_embeddings_server_sync()
            if decision is None:
                _log_rg_emb_server_fallback("rg_decision_unavailable")
                raise RuntimeError(
                    "Embeddings server ResourceGovernor unavailable while RG_ENABLED=1. "
                    "Ensure RG policy loader and backend are configured."
                )
            if decision.get("allowed", False):
                return
            retry_after = decision.get("retry_after")
            wait_s = 1.0
            try:
                if isinstance(retry_after, (int, float)) and retry_after > 0:
                    wait_s = float(retry_after)
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                wait_s = 1.0
            time.sleep(wait_s)

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            self.acquire()
            return fn(*args, **kwargs)

        return wrapper


# --- Resource Governor plumbing (optional) for embeddings server -------------
_rg_emb_server_governor = None
_rg_emb_server_loader = None
_rg_emb_server_log_lock = threading.Lock()
_rg_emb_server_lock_guard = threading.Lock()
_rg_emb_server_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()
_rg_emb_server_init_error: str | None = None
_rg_emb_server_init_error_logged = False
_rg_emb_server_fallback_logged = False


def _rg_emb_server_context() -> dict[str, str]:
    """
    Build RG context dictionary with environment variables and resolved paths.

    Returns:
        Dict containing backend, policy paths, and configuration settings.
    """
    policy_path = os.getenv("RG_POLICY_PATH")
    if policy_path:
        try:
            policy_path_resolved = resolve_repo_relative_path(policy_path)
        except (OSError, TypeError, ValueError):
            policy_path_resolved = policy_path
    else:
        policy_path_resolved = rg_policy_path()
        policy_path = policy_path_resolved
    return {
        "backend": os.getenv("RG_BACKEND", "memory"),
        "policy_path": policy_path,
        "policy_path_resolved": policy_path_resolved,
        "policy_store": os.getenv("RG_POLICY_STORE", ""),
        "policy_reload_enabled": os.getenv("RG_POLICY_RELOAD_ENABLED", ""),
        "policy_reload_interval": os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", ""),
        "cwd": os.getcwd(),
    }


def _log_rg_emb_server_init_failure(exc: Exception) -> None:
    global _rg_emb_server_init_error, _rg_emb_server_init_error_logged
    _rg_emb_server_init_error = repr(exc)
    with _rg_emb_server_log_lock:
        if _rg_emb_server_init_error_logged:
            return
        _rg_emb_server_init_error_logged = True
    ctx = _rg_emb_server_context()
    logger.exception(
        "Embeddings server ResourceGovernor init failed; legacy token-bucket fallback is retired. "
        "RG-enabled paths may fail closed until configuration is fixed. "
        "backend={} policy_path={} policy_path_resolved={} policy_store={} "
        "reload_enabled={} reload_interval={} cwd={}",
        ctx["backend"],
        ctx["policy_path"],
        ctx["policy_path_resolved"],
        ctx["policy_store"],
        ctx["policy_reload_enabled"],
        ctx["policy_reload_interval"],
        ctx["cwd"],
    )


def _log_rg_emb_server_fallback(reason: str) -> None:
    global _rg_emb_server_fallback_logged
    with _rg_emb_server_log_lock:
        if _rg_emb_server_fallback_logged:
            return
        _rg_emb_server_fallback_logged = True
    ctx = _rg_emb_server_context()
    logger.error(
        "Embeddings server ResourceGovernor unavailable; legacy token-bucket fallback is retired. "
        "RG-enabled paths fail closed. "
        "reason={} init_error={} backend={} policy_path={} policy_path_resolved={} "
        "policy_store={} reload_enabled={} reload_interval={} cwd={}",
        reason,
        _rg_emb_server_init_error,
        ctx["backend"],
        ctx["policy_path"],
        ctx["policy_path_resolved"],
        ctx["policy_store"],
        ctx["policy_reload_enabled"],
        ctx["policy_reload_interval"],
        ctx["cwd"],
    )


try:  # pragma: no cover - RG is optional
    from tldw_Server_API.app.core.config import rg_enabled  # type: ignore
    from tldw_Server_API.app.core.Resource_Governance import (  # type: ignore
        MemoryResourceGovernor,
        RedisResourceGovernor,
        RGRequest,
    )
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import (  # type: ignore
        PolicyLoader,
        PolicyReloadConfig,
        default_policy_loader,
    )
except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - safe fallback when RG not installed
    MemoryResourceGovernor = None  # type: ignore
    RedisResourceGovernor = None  # type: ignore
    RGRequest = None  # type: ignore
    PolicyLoader = None  # type: ignore
    PolicyReloadConfig = None  # type: ignore
    default_policy_loader = None  # type: ignore
    rg_enabled = None  # type: ignore


def _rg_emb_server_async_lock() -> asyncio.Lock:
    """Return a loop-bound async lock for RG initialization."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Should not happen in async context; return a fresh lock just in case.
        return asyncio.Lock()
    with _rg_emb_server_lock_guard:
        lock = _rg_emb_server_locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            _rg_emb_server_locks[loop] = lock
        return lock


def _should_enforce_rg_in_production() -> bool:
    env = (os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or os.getenv("ENV") or "dev").lower()
    if env not in {"prod", "production"}:
        return False
    test_mode = is_test_mode()
    pytest_active = bool(os.getenv("PYTEST_CURRENT_TEST"))
    return not (test_mode or pytest_active)


def _assert_rg_enabled_in_production() -> None:
    if not _should_enforce_rg_in_production():
        return
    if rg_enabled is None:
        raise RuntimeError(
            "Resource Governor is unavailable in production; embeddings rate limiting depends on RG. "
            "Install RG dependencies and set RG_ENABLED=true."
        )
    try:
        enabled = bool(rg_enabled(True))  # type: ignore[func-returns-value]
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        raise RuntimeError(
            "Resource Governor config check failed in production; embeddings rate limiting depends on RG."
        ) from exc
    if not enabled:
        raise RuntimeError(
            "Resource Governor is disabled in production; embeddings rate limiting depends on RG. "
            "Set RG_ENABLED=true or [ResourceGovernor].enabled=true."
        )


def _rg_embeddings_server_enabled() -> bool:
    """Return True when RG should gate standalone embeddings server requests."""
    _assert_rg_enabled_in_production()
    if rg_enabled is not None:
        try:
            return bool(rg_enabled(True))  # type: ignore[func-returns-value]
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            return False
    return False


_assert_rg_enabled_in_production()


async def _get_embeddings_server_rg_governor():
    """Lazily initialize a ResourceGovernor instance for the embeddings server."""
    global _rg_emb_server_governor, _rg_emb_server_loader
    if not _rg_embeddings_server_enabled():
        return None
    if RGRequest is None or PolicyLoader is None:
        _log_rg_emb_server_fallback("rg_components_unavailable")
        return None
    if _rg_emb_server_governor is not None:
        return _rg_emb_server_governor
    async with _rg_emb_server_async_lock():
        if _rg_emb_server_governor is not None:
            return _rg_emb_server_governor
        try:
            loader = (
                default_policy_loader()
                if default_policy_loader
                else PolicyLoader(
                    rg_policy_path(),
                    PolicyReloadConfig(
                        enabled=True,
                        interval_sec=int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10"),
                    ),
                )
            )
            await loader.load_once()
            _rg_emb_server_loader = loader
            backend = os.getenv("RG_BACKEND", "memory").lower()
            if backend == "redis" and RedisResourceGovernor is not None:
                gov = RedisResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            else:
                gov = MemoryResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            _rg_emb_server_governor = gov
            return gov
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - optional path
            _log_rg_emb_server_init_failure(exc)
            return None


async def _maybe_enforce_with_rg_embeddings_server_async() -> dict[str, object] | None:
    """
    Attempt to enforce embeddings server request limits via ResourceGovernor.

    Returns a decision dict when RG is used, or None when RG is unavailable/disabled.
    """
    gov = await _get_embeddings_server_rg_governor()
    if gov is None:
        return None
    policy_id = os.getenv("RG_EMBEDDINGS_SERVER_POLICY_ID", "embeddings_server.default")
    op_id = f"emb-server-{time.time_ns()}"
    try:
        decision, handle = await gov.reserve(
            RGRequest(
                entity="service:embeddings_server",
                categories={"requests": {"units": 1}},
                tags={
                    "policy_id": policy_id,
                    "module": "embeddings_server",
                },
            ),
            op_id=op_id,
        )
        if decision.allowed:
            if handle:
                try:
                    await gov.commit(handle, None, op_id=op_id)
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    logger.opt(exception=True).debug("Embeddings server RG commit failed")
            return {"allowed": True, "retry_after": None, "policy_id": policy_id}
        return {
            "allowed": False,
            "retry_after": decision.retry_after or 1,
            "policy_id": policy_id,
        }
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Embeddings server RG reserve failed: {}", exc)
        return None


def _maybe_enforce_with_rg_embeddings_server_sync() -> dict[str, object] | None:
    """
    Synchronous helper for RG enforcement around create_embeddings_batch.

    Uses asyncio.run when no event loop is running in the current thread; if a
    loop is already running, runs RG enforcement in a worker thread to avoid
    cross-loop locks while still enforcing RG.
    """
    if not _rg_embeddings_server_enabled():
        return None
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in this thread; safe to use asyncio.run.
            return asyncio.run(_maybe_enforce_with_rg_embeddings_server_async())
        # Running inside an event loop; execute RG check in a worker thread.
        decision_holder: dict[str, object] = {}
        error_holder: dict[str, Exception] = {}
        done = threading.Event()

        def _run_in_thread() -> None:
            try:
                decision_holder["decision"] = asyncio.run(_maybe_enforce_with_rg_embeddings_server_async())
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
                error_holder["error"] = exc
            finally:
                done.set()

        t = threading.Thread(target=_run_in_thread, daemon=True)
        t.start()
        timeout_s = float(os.getenv("RG_EMBEDDINGS_SERVER_SYNC_TIMEOUT_SEC", "5") or "5")
        done.wait(timeout=timeout_s)
        if not done.is_set():
            logger.debug("Embeddings server RG sync helper timed out after {}s", timeout_s)
            return None
        if error_holder:
            logger.debug("Embeddings server RG sync helper failed: {}", error_holder.get("error"))
            return None
        decision = decision_holder.get("decision")
        if isinstance(decision, dict):
            return decision
        return None
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        # Best-effort: treat RG as unavailable on any unexpected error.
        return None


def exponential_backoff(max_retries: int = 3, base_delay: int = 1):
    """
    Decorator for exponential backoff.
    Note: This uses fixed max_retries and base_delay defined at decoration time.
    It does not use RetryCfg from EmbeddingConfigSchema for dynamic configuration per call.
    """
    logger.info(f"ExponentialBackoff decorator configured with max_retries={max_retries}, base_delay={base_delay}s.")

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):  # +1 to include the initial attempt
                try:
                    return fn(*args, **kwargs)
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                    status = _get_http_status_from_exception(e)
                    is_retryable_http = status == 429 or (isinstance(status, int) and 500 <= status < 600)
                    is_network_error = _is_probable_network_error(e) and not isinstance(e, RetryExhaustedError)

                    if not (is_retryable_http or is_network_error):
                        logger.exception(f"Non-retryable error for {fn.__name__}: {e}")
                        raise

                    if attempt == max_retries:  # Last attempt failed
                        logger.error(
                            f"Final attempt ({attempt + 1}/{max_retries + 1}) failed for {fn.__name__} "
                            f"due to transient error: {e}"
                        )
                        raise

                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} for {fn.__name__} failed with transient error. "
                        f"Retrying in {delay}s. Error: {e}"
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


def evict_lru_models(keep_model_id: str | None = None) -> None:
    """
    Evict least recently used models to maintain resource limits.

    Args:
        keep_model_id: Model ID to keep regardless of LRU status
    """
    global embedding_models, model_last_used, model_memory_usage

    with embedding_models_lock:
        current_time = time.time()

        # Remove models that haven't been used within TTL
        models_to_remove = []
        for model_id, last_used in model_last_used.items():
            if model_id != keep_model_id and (current_time - last_used) > MODEL_LRU_TTL_SECONDS:
                if model_in_use_counts.get(model_id, 0) > 0:
                    continue
                models_to_remove.append(model_id)

        for model_id in models_to_remove:
            _remove_model(model_id)

        # If still over limit, remove LRU models
        while len(embedding_models) >= MAX_MODELS_IN_MEMORY:
            if len(embedding_models) == 0:
                break

            # Find LRU model (excluding keep_model_id)
            lru_model_id = None
            oldest_time = current_time

            for model_id, last_used in model_last_used.items():
                if model_id == keep_model_id:
                    continue
                if model_in_use_counts.get(model_id, 0) > 0:
                    continue
                if last_used < oldest_time:
                    oldest_time = last_used
                    lru_model_id = model_id

            if lru_model_id:
                logger.info(f"Evicting LRU model: {lru_model_id}")
                # Unified audit (non-blocking)
                with contextlib.suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                    log_model_evicted(
                        model_id=lru_model_id,
                        memory_usage_gb=model_memory_usage.get(lru_model_id, 0),
                        reason="lru_eviction",
                    )
                removed = _remove_model(lru_model_id)
                if not removed:
                    logger.debug(f"Unable to evict model '{lru_model_id}' because it is in use.")
                    break
            else:
                break


def _remove_model(model_id: str) -> bool:
    """Remove a model from memory and clean up resources."""
    if model_id not in embedding_models:
        return False

    if model_in_use_counts.get(model_id, 0) > 0:
        return False

    model = embedding_models.get(model_id)
    provider_label = ""
    if isinstance(model, HuggingFaceEmbedder):
        provider_label = "huggingface"
    elif isinstance(model, ONNXEmbedder):
        provider_label = "onnx"
    elif model is not None and hasattr(model, "provider"):
        try:
            provider_label = str(model.provider or "")
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            provider_label = ""

    try:
        # Attempt to clean up model resources
        if hasattr(model, "unload_model"):
            model.unload_model()
        elif hasattr(model, "unload"):
            model.unload()
        elif hasattr(model, "model"):
            del model.model
        elif hasattr(model, "session"):  # ONNX
            model.session = None
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Error cleaning up model {model_id}: {e}")
    finally:
        del embedding_models[model_id]
        model_last_used.pop(model_id, None)
        model_memory_usage.pop(model_id, None)
        model_in_use_counts.pop(model_id, None)
        if provider_label:
            with contextlib.suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                ACTIVE_EMBEDDERS.labels(provider=provider_label, model_id=model_id).set(0)
        logger.info(f"Removed model {model_id} from memory")
    return True


def check_memory_limit(estimated_size_gb: float = 1.0) -> bool:
    """
    Check if loading a new model would exceed memory limits.

    Args:
        estimated_size_gb: Estimated size of the new model in GB

    Returns:
        True if within limits, False otherwise
    """
    current_usage = sum(model_memory_usage.values())
    return (current_usage + estimated_size_gb) <= MAX_MODEL_MEMORY_GB


def get_directory_size(path: str) -> float:
    """
    Calculate the size of a directory in GB.

    Args:
        path: Path to the directory

    Returns:
        Size in GB
    """
    total_size = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                with contextlib.suppress(OSError):
                    total_size += os.path.getsize(filepath)
    except OSError:
        pass

    return total_size / (1024**3)  # Convert bytes to GB


def estimate_model_size(model_name: str, model_path: str | None = None) -> float:
    """
    Estimate model size, preferring actual disk size when available.

    Args:
        model_name: Name of the model
        model_path: Optional path to the model directory

    Returns:
        Estimated or actual size in GB
    """
    # If we have a path, try to get actual size
    if model_path and os.path.exists(model_path):
        actual_size = get_directory_size(model_path)
        if actual_size > 0:
            logger.debug(f"Model {model_name} actual size: {actual_size:.2f} GB")
            return actual_size

    # Check if model is already loaded and we know its size
    if model_name in model_memory_usage:
        return model_memory_usage[model_name]

    # Fallback to name-based estimation
    if "large" in model_name.lower() or "xl" in model_name.lower():
        return 2.0
    elif "base" in model_name.lower() or "medium" in model_name.lower():
        return 1.0
    elif "small" in model_name.lower() or "mini" in model_name.lower():
        return 0.5
    elif "tiny" in model_name.lower():
        return 0.25
    else:
        return 1.0  # Default estimate


class HuggingFaceEmbedder:
    def __init__(self, model_identifier: str, config: HFModelCfg, hf_cache_dir: str):
        self._lock = threading.RLock()
        self.model_identifier = model_identifier
        self.config = config
        self.hf_cache_dir = hf_cache_dir

        self.revision = config.revision or COMMIT_HASHES.get(config.model_name_or_path)
        torch = _import_torch()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize as Optional, to be populated by load_model
        # Type-only; actual classes are imported lazily at use time
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModel | None = None  # AutoModel is a class that returns a model instance

        self.unload_timer: threading.Timer | None = None
        self.last_used_time: float = 0.0
        log_counter("huggingface_embedder_init", labels={"model_id": self.model_identifier})
        logger.info(f"HuggingFaceEmbedder initialized for {model_identifier} (model: {config.model_name_or_path})")

    def _reset_timer(self) -> None:
        if self.config.unload_timeout_seconds <= 0:
            return
        with self._lock:
            if self.unload_timer:
                self.unload_timer.cancel()
            self.unload_timer = threading.Timer(self.config.unload_timeout_seconds, self.unload_model)
            self.unload_timer.daemon = True
            self.unload_timer.start()
            logger.debug(
                f"Unload timer reset for {self.model_identifier}, timeout {self.config.unload_timeout_seconds}s"
            )

    def load_model(self) -> None:
        model_load_attempted = False
        start_time = time.time()

        with self._lock:
            if self.model is None or self.tokenizer is None:  # Ensure both are loaded
                model_load_attempted = True
                log_counter("huggingface_model_load_attempt", labels={"model_id": self.model_identifier})
                logger.info(
                    f"Loading HuggingFace model/tokenizer: {self.config.model_name_or_path} (ID: {self.model_identifier}) on device {self.device}"
                )

                _ensure_hf_revision(self.config.model_name_or_path, self.revision)

                # Ensure AutoTokenizer and AutoModel are the classes from transformers (lazy import)
                AutoModel, AutoTokenizer = _import_transformers()
                # These lines assign INSTANCES to self.tokenizer and self.model
                self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                    self.config.model_name_or_path,
                    cache_dir=self.hf_cache_dir,
                    revision=self.revision,
                    trust_remote_code=self.config.trust_remote_code,
                )
                # AutoModel.from_pretrained returns an instance of a model class (e.g., BertModel, RobertaModel)
                # which is a subclass of PreTrainedModel, which is a torch.nn.Module.
                loaded_model = AutoModel.from_pretrained(  # nosec B615
                    self.config.model_name_or_path,
                    cache_dir=self.hf_cache_dir,
                    revision=self.revision,
                    trust_remote_code=self.config.trust_remote_code,
                )
                self.model = loaded_model.to(self.device)
                self.model.eval()

                ACTIVE_EMBEDDERS.labels(provider="huggingface", model_id=self.model_identifier).inc()
                log_counter("huggingface_model_load_success", labels={"model_id": self.model_identifier})
                logger.info(
                    f"HuggingFace model {self.config.model_name_or_path} loaded. Max length: {self.config.max_length}, Timeout: {self.config.unload_timeout_seconds}s."
                )

            self.last_used_time = time.time()

        self._reset_timer()

        if model_load_attempted:
            load_time = time.time() - start_time
            log_histogram("huggingface_model_load_duration", load_time, labels={"model_id": self.model_identifier})

    def unload_model(self) -> None:
        with self._lock:
            log_counter("huggingface_model_unload", labels={"model_id": self.model_identifier})
            if self.model is not None or self.tokenizer is not None:
                logger.info(
                    f"Unloading HuggingFace model/tokenizer {self.config.model_name_or_path} (ID: {self.model_identifier})"
                )
                del self.model
                del self.tokenizer
                torch = _import_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = None
                self.tokenizer = None
                ACTIVE_EMBEDDERS.labels(provider="huggingface", model_id=self.model_identifier).dec()
                logger.info(f"HuggingFace model {self.model_identifier} unloaded.")

            if self.unload_timer:
                self.unload_timer.cancel()
                self.unload_timer = None
        # Update memory accounting outside the model lock to avoid lock ordering issues.
        try:
            with embedding_models_lock:
                if self.model_identifier in model_memory_usage:
                    model_memory_usage[self.model_identifier] = 0.0
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass

    def create_embeddings(self, texts: list[str]) -> np.ndarray:
        self.load_model()

        # --- Start of critical section for using model and tokenizer ---
        # We need to ensure model and tokenizer are not None when used.
        # The lock here protects against the model being unloaded by the timer
        # thread *during* the tokenization and inference process.
        with self._lock:
            # Explicit checks to satisfy type checkers and for runtime safety
            if self.tokenizer is None or self.model is None:
                logger.error(
                    f"Model or tokenizer not loaded for {self.model_identifier} despite load_model call. This indicates a critical issue."
                )
                # Attempt a final reload under lock, though this state should ideally not be reached.
                self.load_model()
                if self.tokenizer is None or self.model is None:
                    raise RuntimeError(
                        f"Model {self.model_identifier} failed to load even after explicit reload attempt."
                    )

            # At this point, self.tokenizer and self.model are confirmed to be loaded and not None.
            # The type checker should now understand they are instances, not Optional.

            # Re-assign to local variables for type checker to potentially infer non-Optional type better
            # although the checks above should be enough for modern type checkers.
            current_tokenizer = self.tokenizer
            current_model = self.model

            log_counter("huggingface_create_embeddings_attempt", labels={"model_id": self.model_identifier})
            start_time_embed = time.time()
            torch = _import_torch()
            embeddings_tensor: torch.Tensor | None = None

            def _mean_pool(hidden_state, attention_mask):
                if attention_mask is None:
                    return hidden_state.mean(dim=1)
                mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                summed = (hidden_state * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                return summed / denom

            try:
                # Qwen3 Embeddings: apply instruction-aware formatting and use last-token pooling
                model_l = (self.config.model_name_or_path or "").lower()
                is_qwen3_embed = "qwen3" in model_l and "embedding" in model_l

                fmt_texts = texts
                if is_qwen3_embed:
                    # Load optional instruction and mode from embeddings.prompts
                    instr = load_prompt("embeddings", "qwen3_embeddings_instruction") or (
                        "Given a web search query, retrieve relevant passages that answer the query"
                    )
                    mode = (load_prompt("embeddings", "qwen3_embeddings_mode") or "auto").strip().lower()

                    def _likely_query(s: str) -> bool:
                        t = (s or "").strip().lower()
                        if t.endswith("?"):
                            return True
                        prefixes = ("what ", "who ", "when ", "where ", "why ", "how ", "explain ", "define ")
                        return len(t) <= 160 and any(t.startswith(p) for p in prefixes)

                    def _format_query(q: str) -> str:
                        return f"<Instruct>: {instr}\n<Query>: {q}"

                    def _format_doc(d: str) -> str:
                        return f"<Instruct>: {instr}\n<Document>: {d}"

                    fmt_texts = []
                    for t in texts:
                        if isinstance(t, str) and "<Instruct>:" in t:
                            fmt_texts.append(t)
                            continue
                        if mode == "query":
                            fmt_texts.append(_format_query(t))
                        elif mode == "document":
                            fmt_texts.append(_format_doc(t))
                        else:  # auto
                            fmt_texts.append(_format_query(t) if _likely_query(t) else _format_doc(t))

                # Tokenize
                inputs = current_tokenizer(
                    fmt_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    # current_model is an instance of a PreTrainedModel, which is callable (its forward method)
                    outputs = current_model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                if is_qwen3_embed:
                    # last-token pooling
                    attn = inputs.get("attention_mask")
                    if attn is not None:
                        lengths = attn.sum(dim=1) - 1
                        bsz, dim = last_hidden_state.size(0), last_hidden_state.size(-1)
                        idx = lengths.view(bsz, 1, 1).expand(bsz, 1, dim)
                        embeddings_tensor = last_hidden_state.gather(1, idx).squeeze(1)
                    else:
                        embeddings_tensor = last_hidden_state[:, -1, :]
                else:
                    # default: mean pooling with attention mask
                    embeddings_tensor = _mean_pool(last_hidden_state, inputs.get("attention_mask"))

            except RuntimeError as e:
                # Handle BFloat16 issue
                # The hasattr check is good, add an explicit None check for self.model.dtype
                if (
                    "Got unsupported ScalarType BFloat16" in str(e)
                    and current_model is not None
                    and hasattr(current_model, "dtype")
                    and current_model.dtype == torch.bfloat16
                ):  # current_model is not None here

                    logger.warning(
                        f"BFloat16 not supported for {self.config.model_name_or_path} on {self.device}. "
                        f"Falling back to float32 for model {self.model_identifier}."
                    )

                    # current_model is a torch.nn.Module, so .float() is a valid method.
                    # Re-assign to self.model as well if the change should persist.
                    self.model = current_model.float()  # self.model is now the float version
                    current_model = self.model  # Update local var for current execution
                    log_counter("huggingface_bfloat16_fallback", labels={"model_id": self.model_identifier})

                    # Retry embedding creation with the converted model
                    logger.info(f"Retrying embedding creation for {self.model_identifier} with float32 model.")
                    # Re-tokenize with same formatting
                    inputs = current_tokenizer(  # Use current_tokenizer, it hasn't changed
                        fmt_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = current_model(**inputs)  # Use the now-float current_model
                    last_hidden_state = outputs.last_hidden_state
                    if is_qwen3_embed:
                        attn = inputs.get("attention_mask")
                        if attn is not None:
                            lengths = attn.sum(dim=1) - 1
                            bsz, dim = last_hidden_state.size(0), last_hidden_state.size(-1)
                            idx = lengths.view(bsz, 1, 1).expand(bsz, 1, dim)
                            embeddings_tensor = last_hidden_state.gather(1, idx).squeeze(1)
                        else:
                            embeddings_tensor = last_hidden_state[:, -1, :]
                    else:
                        embeddings_tensor = _mean_pool(last_hidden_state, inputs.get("attention_mask"))
                else:
                    log_counter("huggingface_create_embeddings_failure", labels={"model_id": self.model_identifier})
                    logger.exception(f"RuntimeError during HuggingFace embedding for {self.model_identifier}: {e}")
                    raise
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                log_counter("huggingface_create_embeddings_failure", labels={"model_id": self.model_identifier})
                logger.exception(f"Unexpected error during HuggingFace embedding for {self.model_identifier}: {e}")
                raise

            if embeddings_tensor is None:
                # This should not happen if the try-except block is complete
                logger.error(f"Embeddings tensor is None after processing for {self.model_identifier}")
                raise RuntimeError(f"Failed to produce embeddings tensor for {self.model_identifier}")

            embedding_time = time.time() - start_time_embed
            log_histogram(
                "huggingface_create_embeddings_duration", embedding_time, labels={"model_id": self.model_identifier}
            )
            log_counter("huggingface_create_embeddings_success", labels={"model_id": self.model_identifier})
            return embeddings_tensor.cpu().float().numpy()
        # --- End of critical section ---

    def __del__(self):
        logger_debug = getattr(logger, "debug", None)
        if callable(logger_debug):
            with contextlib.suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                model_identifier = getattr(self, "model_identifier", "<uninitialized>")
                logger_debug(f"HuggingFaceEmbedder {model_identifier} is being deleted.")
        unload_timer = getattr(self, "unload_timer", None)
        if unload_timer:
            with contextlib.suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                unload_timer.cancel()
            self.unload_timer = None


class ONNXEmbedder:
    def __init__(
        self,
        model_identifier: str,
        config: ONNXModelCfg,
        onnx_model_base_storage_dir: str,
        model_storage_dir: str | None = None,
    ):
        self._lock = threading.RLock()  # Reentrant lock for this instance
        self.model_identifier = model_identifier
        self.config = config

        self.revision = config.revision or COMMIT_HASHES.get(config.model_name_or_path)

        # Directory for this specific ONNX model's files (model.onnx, tokenizer, config)
        if model_storage_dir:
            self.model_specific_onnx_dir = model_storage_dir
        else:
            self.model_specific_onnx_dir = os.path.join(
                onnx_model_base_storage_dir,
                config.model_name_or_path.split("/")[-1],
            )
        os.makedirs(self.model_specific_onnx_dir, exist_ok=True)
        self.onnx_model_file_path = os.path.join(self.model_specific_onnx_dir, "model.onnx")  # Standard name by optimum

        # Initialize critical attributes early so __del__/finalizers are safe even if setup fails
        self.session: ort.InferenceSession | None = None
        self.unload_timer: threading.Timer | None = None
        self.last_used_time: float = 0.0
        self.device_providers = config.onnx_providers

        # Tokenizer is usually stored with the ONNX model by optimum (lazy import)
        _, AutoTokenizer = _import_transformers()
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            config.model_name_or_path,  # Original HF name for tokenizer
            cache_dir=self.model_specific_onnx_dir,  # Store/load tokenizer from the model's ONNX directory
            revision=self.revision,
            trust_remote_code=config.trust_remote_code,
        )

        log_counter("onnx_embedder_init", labels={"model_id": self.model_identifier})
        logger.info(f"ONNXEmbedder initialized for {model_identifier} (model: {config.model_name_or_path})")

    def _ensure_model_converted_and_ready(self) -> None:
        # This method is called from load_model, which holds the instance lock.
        if os.path.exists(self.onnx_model_file_path):
            logger.debug(f"ONNX model file already exists at {self.onnx_model_file_path} for {self.model_identifier}")
            return

        try:
            ORTModelForFeatureExtraction = _import_optimum_ort_model()
        except ImportError as exc:
            msg = "`optimum` library is not available. Cannot convert model to ONNX on-the-fly."
            logger.error("{} Error: {}", msg, exc)
            raise RuntimeError(msg) from exc

        logger.warning(
            f"ONNX model file not found at {self.onnx_model_file_path} for {self.model_identifier}. "
            f"Attempting to convert '{self.config.model_name_or_path}' and save to '{self.model_specific_onnx_dir}'."
        )

        _ensure_hf_revision(self.config.model_name_or_path, self.revision)

        try:
            # ORTModelForFeatureExtraction.from_pretrained with export=True downloads the PyTorch model,
            # converts it, and then save_pretrained saves it to disk.
            # The `cache_dir` for from_pretrained here is where the *original* HF PyTorch model parts are downloaded.
            # It can be the same as self.model_specific_onnx_dir or a temporary HF cache.
            # For simplicity, let's use the model_specific_onnx_dir to keep related files together.
            logger.info(f"Downloading and converting {self.config.model_name_or_path} to ONNX...")
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                self.config.model_name_or_path,
                export=True,
                trust_remote_code=self.config.trust_remote_code,
                revision=self.revision,
                cache_dir=self.model_specific_onnx_dir,  # For downloading original HF model before conversion
            )

            logger.info(f"Saving converted ONNX model to {self.model_specific_onnx_dir}...")
            ort_model.save_pretrained(self.model_specific_onnx_dir)  # Saves model.onnx, config.json, etc.

            if not os.path.exists(self.onnx_model_file_path):
                raise FileNotFoundError(
                    f"ONNX 'model.onnx' (expected at {self.onnx_model_file_path}) was not found in "
                    f"{self.model_specific_onnx_dir} after export and save attempt."
                )
            logger.info(
                f"ONNX model for {self.config.model_name_or_path} (ID: {self.model_identifier}) "
                f"successfully exported and saved to {self.model_specific_onnx_dir}"
            )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
            logger.exception(f"Failed to export/download ONNX model for {self.model_identifier}: {e}")
            # Basic cleanup: if model.onnx was partially created, remove it.
            if os.path.exists(self.onnx_model_file_path):
                with contextlib.suppress(OSError):
                    os.remove(self.onnx_model_file_path)
            raise RuntimeError(f"ONNX model conversion failed for {self.model_identifier}.") from e

    def _reset_timer(self) -> None:
        # This method must be thread-safe
        if self.config.unload_timeout_seconds <= 0:
            return
        with self._lock:  # Protect timer manipulation
            if self.unload_timer:
                self.unload_timer.cancel()
            self.unload_timer = threading.Timer(self.config.unload_timeout_seconds, self.unload_model)
            self.unload_timer.daemon = True
            self.unload_timer.start()
            logger.debug(
                f"Unload timer reset for ONNX model {self.model_identifier}, timeout {self.config.unload_timeout_seconds}s"
            )

    def load_model(self) -> None:
        # This entire method needs to be atomic per instance.
        session_load_attempted = False
        start_time = time.time()

        with self._lock:
            if self.session is None:
                ort_mod = _import_onnxruntime()
                session_load_attempted = True
                log_counter("onnx_model_load_attempt", labels={"model_id": self.model_identifier})

                self._ensure_model_converted_and_ready()  # This runs under the same lock

                logger.info(
                    f"Loading ONNX model for {self.model_identifier} from {self.onnx_model_file_path} "
                    f"with providers: {self.device_providers}"
                )
                self.session = ort_mod.InferenceSession(self.onnx_model_file_path, providers=self.device_providers)

                ACTIVE_EMBEDDERS.labels(provider="onnx", model_id=self.model_identifier).inc()
                log_counter("onnx_model_load_success", labels={"model_id": self.model_identifier})
                logger.info(
                    f"ONNX model {self.model_identifier} loaded. Max length: {self.config.max_length}, Timeout: {self.config.unload_timeout_seconds}s."
                )

            self.last_used_time = time.time()

        self._reset_timer()  # Call after releasing main lock

        if session_load_attempted:
            load_time = time.time() - start_time
            log_histogram("onnx_model_load_duration", load_time, labels={"model_id": self.model_identifier})

    def unload_model(self) -> None:
        with self._lock:  # Ensure thread-safety
            log_counter("onnx_model_unload", labels={"model_id": self.model_identifier})
            if self.session is not None:
                logger.info(f"Unloading ONNX model {self.config.model_name_or_path} (ID: {self.model_identifier})")
                del self.session  # Allow OrtInferenceSession to clean up
                self.session = None
                ACTIVE_EMBEDDERS.labels(provider="onnx", model_id=self.model_identifier).dec()
                logger.info(f"ONNX model {self.model_identifier} unloaded.")

            if self.unload_timer:
                self.unload_timer.cancel()
                self.unload_timer = None
        # Update memory accounting outside the model lock to avoid lock ordering issues.
        try:
            with embedding_models_lock:
                if self.model_identifier in model_memory_usage:
                    model_memory_usage[self.model_identifier] = 0.0
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass

    def create_embeddings(self, texts: list[str]) -> np.ndarray:
        self.load_model()  # Handles locking, model loading/conversion, and timer reset

        if self.session is None or self.tokenizer is None:
            logger.error(
                f"ONNX session or tokenizer not loaded for {self.model_identifier} before create_embeddings call."
            )
            raise RuntimeError(f"ONNX model {self.model_identifier} not loaded properly.")

        log_counter("onnx_create_embeddings_attempt", labels={"model_id": self.model_identifier})
        start_time_embed = time.time()

        try:
            # Inference needs to be under lock to prevent unload during operation
            with self._lock:
                # Re-check session status in case it was unloaded by a concurrent timer thread
                if self.session is None:  # Should be rare
                    logger.warning(f"ONNX session for {self.model_identifier} became None unexpectedly. Reloading.")
                    self.load_model()
                    if self.session is None:  # Still none, critical error
                        raise RuntimeError(f"ONNX session for {self.model_identifier} could not be reloaded.")

                inputs = self.tokenizer(
                    texts,
                    return_tensors="np",  # ONNX runtime uses NumPy arrays
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                )
                ort_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                }
                # Some models need token_type_ids, some don't. Check if tokenizer provides them.
                if "token_type_ids" in inputs and inputs["token_type_ids"] is not None:
                    model_input_names = [inp.name for inp in self.session.get_inputs()]
                    if "token_type_ids" in model_input_names:
                        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
                    elif "token_type_ids" in ort_inputs:  # remove if tokenizer provided but model doesn't want
                        del ort_inputs["token_type_ids"]

                ort_outputs = self.session.run(None, ort_inputs)

                # Pooling: Mean pooling of the last hidden state, considering attention mask
                last_hidden_state = ort_outputs[0]  # Typically the first output
                if not isinstance(last_hidden_state, np.ndarray):
                    raise TypeError(f"Expected numpy array from ONNX output, got {type(last_hidden_state)}")

                input_mask_expanded = np.expand_dims(ort_inputs["attention_mask"], -1).astype(float)
                sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
                sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)  # Avoid division by zero
                embeddings_np = sum_embeddings / sum_mask

        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
            log_counter("onnx_create_embeddings_failure", labels={"model_id": self.model_identifier})
            logger.exception(f"Error creating embeddings with ONNX model {self.model_identifier}: {e}")
            raise

        embedding_time = time.time() - start_time_embed
        log_histogram("onnx_create_embeddings_duration", embedding_time, labels={"model_id": self.model_identifier})
        log_counter("onnx_create_embeddings_success", labels={"model_id": self.model_identifier})
        return embeddings_np.astype(np.float32)  # Ensure float32 output

    def __del__(self) -> None:
        try:
            if hasattr(self, "unload_timer"):
                timer = getattr(self, "unload_timer", None)
                if timer is not None:
                    try:
                        timer.cancel()
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                        # Never raise from __del__
                        pass
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            # Guard against any unexpected attribute/state issues during GC
            pass


# Exponential backoff decorator with fixed parameters.
# To make this dynamic per model_config, apply similarly to limiter.
@exponential_backoff(max_retries=3, base_delay=1)
def create_embeddings_batch(
    texts: list[str],
    user_app_config: dict[str, Any],  # Renamed for clarity: this is the top-level app config
    model_id_override: str | None = None,
) -> list[list[float]]:
    """
    Creates embeddings for a batch of texts.

    Accepted model_id formats for lookup in embedding_config.models:
    - provider:model  (e.g., "huggingface:sentence-transformers/all-MiniLM-L6-v2")
    - model          (bare model name; resolver will attempt to infer provider or
                      match a unique "provider:model" key ending with ":model")

    `user_app_config` should contain an 'embedding_config' key with EmbeddingConfigSchema structure.
    """
    if not texts:
        logger.warning("create_embeddings_batch called with empty list of texts.")
        return []

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        logger.warning(
            "create_embeddings_batch called from a running event loop; "
            "use create_embeddings_batch_async to avoid blocking."
        )

    try:
        # Extract and validate the specific embedding configuration part
        if "embedding_config" not in user_app_config:
            logger.error("'embedding_config' key not found in user_app_config.")
            raise ValueError("'embedding_config' key missing from application configuration.")

        # Pydantic will parse and validate. If it fails, it raises a ValidationError.
        embedding_service_config = EmbeddingConfigSchema(**user_app_config["embedding_config"])
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:  # Catch Pydantic ValidationError or other parsing issues
        logger.exception(f"Failed to parse embedding_config: {str(e)}")
        raise ValueError(f"Invalid embedding_config structure: {e}") from e

    model_id_to_use = model_id_override if model_id_override else embedding_service_config.default_model_id
    if not model_id_to_use:
        logger.error("No `model_id` specified and no `default_model_id` found in embedding_config.")
        raise ValueError("Embedding model ID not specified or configured as default.")

    def _resolve_model_key(models_map: dict[str, Any], mid: str) -> tuple[str, Any]:
        """Resolve a model key from models_map supporting bare or provider-prefixed IDs.

        Tries exact match first, then:
        - If mid contains ':', try its suffix as a bare key
        - If bare, try common provider prefixes (heuristic) and any unique key ending with ":mid"
        Returns (resolved_key, model_spec) on success or raises ValueError.
        """
        # 1) Exact key
        if mid in models_map:
            return mid, models_map[mid]
        # 2) If provider-prefixed, try bare suffix
        if ":" in mid:
            suffix = mid.split(":", 1)[1]
            if suffix in models_map:
                return suffix, models_map[suffix]
        # 3) If bare, try prefixed candidates based on simple heuristics
        bare = mid.split(":", 1)[1] if ":" in mid else mid
        guessed_providers = []
        if "/" in bare:
            guessed_providers.append("huggingface")
        # Always consider openai and local_api as common options
        guessed_providers.extend(["openai", "local_api"])  # order matters for tie-breaks
        for prov in guessed_providers:
            candidate = f"{prov}:{bare}"
            if candidate in models_map:
                return candidate, models_map[candidate]
        # 4) Unique suffix match (any key that ends with ":<bare>")
        suffix_matches = [k for k in models_map if k.endswith(f":{bare}")]
        if len(suffix_matches) == 1:
            k = suffix_matches[0]
            return k, models_map[k]
        logger.error(f"Configuration for `model_id` '{mid}' not found in `embedding_config.models`.")
        raise ValueError(f"Invalid `model_id` or configuration missing: {mid}")

    resolved_key, model_spec = _resolve_model_key(embedding_service_config.models, model_id_to_use)
    model_id_to_use = resolved_key

    provider = model_spec.provider

    # Phase 2: RG middleware handles ingress rate limiting

    # Ensure model_storage_base_dir exists and stays under the allowlist root
    base_dir = _normalize_model_storage_base_dir(embedding_service_config.model_storage_base_dir)
    os.makedirs(base_dir, exist_ok=True)

    EMBEDDINGS_REQUESTS.labels(provider=provider, model_id=model_id_to_use).inc()
    start_time_batch = time.time()
    embeddings_list: list[list[float]] = []

    try:
        embedder_instance: Any = None  # To hold HFEmbedder or ONNXEmbedder

        if _should_use_inprocess_test_embeddings(provider):
            logger.info(
                "Using deterministic synthetic embeddings for {} in in-process test mode",
                model_id_to_use,
            )
            embeddings_list = [_synthetic_test_embedding(text) for text in texts]
        elif provider.lower() == "huggingface":
            if not isinstance(model_spec, HFModelCfg):
                raise ValueError(f"Model spec for {model_id_to_use} is not HFModelCfg.")

            model_id_in_use: str | None = None
            with embedding_models_lock:  # Protect access to the global embedding_models cache
                if model_id_to_use not in embedding_models:
                    logger.info(f"HuggingFace model ID {model_id_to_use} not in cache. Initializing.")

                    # Setup cache directory
                    hf_cache_dir = _safe_model_storage_subdir(
                        base_dir,
                        model_spec.hf_cache_dir_subpath,
                        "hf_cache_dir_subpath",
                    )
                    os.makedirs(hf_cache_dir, exist_ok=True)

                    cache_subdir = _model_cache_subdir_name(model_id_to_use)
                    model_cache_dir = _safe_model_storage_subdir(
                        hf_cache_dir,
                        cache_subdir,
                        "model_cache_dir",
                    )

                    # Check resource limits before loading - use actual path if available
                    estimated_size = estimate_model_size(model_id_to_use, model_cache_dir)

                    if not check_memory_limit(estimated_size):
                        logger.warning(
                            f"Memory limit would be exceeded by loading {model_id_to_use} (size: {estimated_size:.2f} GB)"
                        )
                        with contextlib.suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                            log_memory_limit_exceeded(
                                model_id=model_id_to_use,
                                memory_usage_gb=estimated_size,
                                current_usage_gb=sum(model_memory_usage.values()),
                                limit_gb=MAX_MODEL_MEMORY_GB,
                            )
                        evict_lru_models(keep_model_id=model_id_to_use)
                        if not check_memory_limit(estimated_size):
                            logger.error(
                                "Memory limit still exceeded after eviction; refusing to load model {}.",
                                model_id_to_use,
                            )
                            raise RuntimeError(
                                f"Embeddings model '{model_id_to_use}' exceeds memory limit and no evictable models are available."
                            )

                    # Evict LRU models if at capacity
                    if len(embedding_models) >= MAX_MODELS_IN_MEMORY:
                        logger.info(f"At model capacity ({MAX_MODELS_IN_MEMORY}), evicting LRU models")
                        evict_lru_models(keep_model_id=model_id_to_use)
                        if len(embedding_models) >= MAX_MODELS_IN_MEMORY:
                            logger.error(
                                "Model cache still at capacity after eviction; refusing to load model {}.",
                                model_id_to_use,
                            )
                            raise RuntimeError(
                                f"Embeddings model cache at capacity and no evictable models are available for '{model_id_to_use}'."
                            )

                    os.makedirs(model_cache_dir, exist_ok=True)
                    embedding_models[model_id_to_use] = HuggingFaceEmbedder(
                        model_id_to_use,
                        model_spec,
                        model_cache_dir,
                    )
                    model_memory_usage[model_id_to_use] = estimated_size
                    model_last_used[model_id_to_use] = time.time()
                    logger.info(f"Loaded model {model_id_to_use} (size: {estimated_size:.2f} GB)")
                else:
                    MODEL_CACHE_HITS.labels(model_id=model_id_to_use).inc()
                    model_last_used[model_id_to_use] = time.time()
                embedder_instance = embedding_models[model_id_to_use]
                _mark_model_in_use(model_id_to_use)
                model_id_in_use = model_id_to_use

            if embedder_instance:
                try:
                    embeddings_np = embedder_instance.create_embeddings(texts)
                    embeddings_list = embeddings_np.tolist()
                finally:
                    if model_id_in_use:
                        _release_model_in_use(model_id_in_use)

        elif provider.lower() == "onnx":
            if not isinstance(model_spec, ONNXModelCfg):
                raise ValueError(f"Model spec for {model_id_to_use} is not ONNXModelCfg.")

            model_id_in_use = None
            with embedding_models_lock:
                if model_id_to_use not in embedding_models:
                    logger.info(f"ONNX model ID {model_id_to_use} not in cache. Initializing.")

                    onnx_root_dir = _safe_model_storage_subdir(
                        base_dir,
                        model_spec.onnx_storage_dir_subpath,
                        "onnx_storage_dir_subpath",
                    )
                    os.makedirs(onnx_root_dir, exist_ok=True)
                    cache_subdir = _model_cache_subdir_name(model_id_to_use)
                    onnx_model_path = _safe_model_storage_subdir(
                        onnx_root_dir,
                        cache_subdir,
                        "onnx_model_cache_dir",
                    )

                    # Check resource limits before loading - use actual path if available
                    estimated_size = estimate_model_size(model_id_to_use, onnx_model_path)

                    if not check_memory_limit(estimated_size):
                        logger.warning(
                            f"Memory limit would be exceeded by loading {model_id_to_use} (size: {estimated_size:.2f} GB)"
                        )
                        with contextlib.suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                            log_memory_limit_exceeded(
                                model_id=model_id_to_use,
                                memory_usage_gb=estimated_size,
                                current_usage_gb=sum(model_memory_usage.values()),
                                limit_gb=MAX_MODEL_MEMORY_GB,
                            )
                        evict_lru_models(keep_model_id=model_id_to_use)
                        if not check_memory_limit(estimated_size):
                            logger.error(
                                "Memory limit still exceeded after eviction; refusing to load ONNX model {}.",
                                model_id_to_use,
                            )
                            raise RuntimeError(
                                f"Embeddings model '{model_id_to_use}' exceeds memory limit and no evictable models are available."
                            )

                    # Evict LRU models if at capacity
                    if len(embedding_models) >= MAX_MODELS_IN_MEMORY:
                        logger.info(f"At model capacity ({MAX_MODELS_IN_MEMORY}), evicting LRU models")
                        evict_lru_models(keep_model_id=model_id_to_use)
                        if len(embedding_models) >= MAX_MODELS_IN_MEMORY:
                            logger.error(
                                "Model cache still at capacity after eviction; refusing to load ONNX model {}.",
                                model_id_to_use,
                            )
                            raise RuntimeError(
                                f"Embeddings model cache at capacity and no evictable models are available for '{model_id_to_use}'."
                            )

                    embedding_models[model_id_to_use] = ONNXEmbedder(
                        model_id_to_use,
                        model_spec,
                        onnx_root_dir,
                        model_storage_dir=onnx_model_path,
                    )
                    model_memory_usage[model_id_to_use] = estimated_size
                    model_last_used[model_id_to_use] = time.time()
                    logger.info(f"Loaded ONNX model {model_id_to_use} (size: {estimated_size:.2f} GB)")
                else:
                    MODEL_CACHE_HITS.labels(model_id=model_id_to_use).inc()
                    model_last_used[model_id_to_use] = time.time()
                embedder_instance = embedding_models[model_id_to_use]
                _mark_model_in_use(model_id_to_use)
                model_id_in_use = model_id_to_use

            if embedder_instance:
                try:
                    embeddings_np = embedder_instance.create_embeddings(texts)
                    embeddings_list = embeddings_np.tolist()
                finally:
                    if model_id_in_use:
                        _release_model_in_use(model_id_in_use)

        elif provider.lower() == "openai":
            if not isinstance(model_spec, OpenAIModelCfg):
                raise ValueError(f"Model spec for {model_id_to_use} is not OpenAIModelCfg.")

            logger.debug(
                f"Creating embeddings for {len(texts)} texts via OpenAI API with model {model_spec.model_name_or_path}"
            )
            if not callable(get_openai_embeddings_batch):  # Basic check
                logger.error("`get_openai_embeddings_batch` is not available or not callable.")
                raise NotImplementedError("OpenAI batch embedding function is not properly set up.")

            openai_app_config = user_app_config
            if model_spec.api_key:
                openai_section = dict(user_app_config.get("openai_api", {}) or {})
                if not openai_section.get("api_key"):
                    openai_section["api_key"] = model_spec.api_key
                if openai_section != user_app_config.get("openai_api", {}):
                    openai_app_config = {**user_app_config, "openai_api": openai_section}

            # Pass the full user_app_config as it might contain API keys or other necessary settings
            # for get_openai_embeddings_batch
            embeddings_list = get_openai_embeddings_batch(
                texts,
                model=model_spec.model_name_or_path,
                app_config=openai_app_config,  # Or pass only relevant parts if get_openai_embeddings_batch is refactored
                dimensions=model_spec.dimensions,
            )

        elif provider.lower() == "local_api":
            if not isinstance(model_spec, LocalAPICfg):
                raise ValueError(f"Model spec for {model_id_to_use} is not LocalAPICfg.")

            # TODO: Implement chunking for texts if len(texts) is large, based on model_spec.chunk_size
            logger.debug(
                f"Creating {len(texts)} embeddings via local API ({model_spec.api_url}) with model {model_spec.model_name_or_path}"
            )
            headers = {"Content-Type": "application/json"}
            if model_spec.api_key:
                headers["Authorization"] = f"Bearer {model_spec.api_key}"

            payload = {"texts": texts, "model": model_spec.model_name_or_path}

            # The outbound call is already wrapped by exponential backoff and the per-config rate limiter
            from tldw_Server_API.app.core.http_client import fetch as _fetch

            resp = _fetch(method="POST", url=model_spec.api_url, headers=headers, json=payload, timeout=60)
            if resp.status_code >= 400:
                resp.raise_for_status()
            response_data = resp.json()
            if "embeddings" not in response_data or not isinstance(response_data["embeddings"], list):
                logger.error(f"Local API at {model_spec.api_url} returned unexpected data format: {response_data}")
                raise ValueError("Local API embedding response format error.")
            embeddings_list = response_data["embeddings"]

        else:
            logger.error(f"Unsupported embedding provider: {provider} for model_id '{model_id_to_use}'")
            raise ValueError(f"Unsupported embedding provider: {provider}")

        batch_time = time.time() - start_time_batch
        log_histogram(
            "create_embeddings_batch_duration", batch_time, labels={"provider": provider, "model_id": model_id_to_use}
        )
        log_counter("create_embeddings_batch_success", labels={"provider": provider, "model_id": model_id_to_use})
        return embeddings_list

    except ValueError as ve:  # Configuration or validation errors
        log_counter(
            "create_embeddings_batch_error",
            labels={
                "provider": provider if "provider" in locals() else "unknown",
                "model_id": model_id_to_use if "model_id_to_use" in locals() else "unknown",
                "error_type": type(ve).__name__,
            },
        )
        logger.exception(f"Configuration or Value error in create_embeddings_batch: {ve}")
        raise
    except RuntimeError as rte:  # Model loading, conversion, or runtime issues
        log_counter(
            "create_embeddings_batch_error",
            labels={
                "provider": provider if "provider" in locals() else "unknown",
                "model_id": model_id_to_use if "model_id_to_use" in locals() else "unknown",
                "error_type": type(rte).__name__,
            },
        )
        logger.exception(f"Runtime error in create_embeddings_batch: {rte}")
        raise
    except (NetworkError, RetryExhaustedError) as req_e:
        log_counter(
            "create_embeddings_batch_error",
            labels={
                "provider": provider if "provider" in locals() else "unknown",
                "model_id": model_id_to_use if "model_id_to_use" in locals() else "unknown",
                "error_type": type(req_e).__name__,
            },
        )
        logger.exception(f"Network error after retries in create_embeddings_batch: {req_e}")
        raise
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:  # Catch-all for unexpected errors
        log_counter(
            "create_embeddings_batch_error",
            labels={
                "provider": provider if "provider" in locals() else "unknown",
                "model_id": model_id_to_use if "model_id_to_use" in locals() else "unknown",
                "error_type": type(e).__name__,
            },
        )
        logger.exception(
            f"Unexpected error in create_embeddings_batch for model_id '{model_id_to_use if 'model_id_to_use' in locals() else 'unknown'}' "
            f"(Provider: {provider if 'provider' in locals() else 'unknown'}): {e}"
        )
        raise


async def create_embeddings_batch_async(
    texts: list[str],
    user_app_config: dict[str, Any],
    model_id_override: str | None = None,
) -> list[list[float]]:
    """
    Async wrapper for create_embeddings_batch.
    Creates embeddings for multiple texts asynchronously.

    Args:
        texts: List of texts to embed
        user_app_config: Configuration dictionary containing 'embedding_config'
        model_id_override: Optional model ID to override the default

    Returns:
        List of embedding vectors (list of floats for each text)
    """
    import asyncio

    # Run the synchronous function in a thread pool to avoid blocking
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, create_embeddings_batch, texts, user_app_config, model_id_override  # Use default executor
    )


def create_embedding(
    text: str,
    user_app_config: dict[str, Any],
    model_id_override: str | None = None,
) -> list[float]:
    """
    Creates an embedding for a single text using the batch function.
    `user_app_config` should contain an 'embedding_config' key.
    """
    if not text:
        logger.warning("`create_embedding` called with empty text. Behavior depends on model.")
        # Models might return a specific embedding for empty string, or error.
        # For now, proceed and let the batch/model handle it.

    # Determine provider and model_id for logging purposes before calling batch,
    # as batch might raise an error before these are determined internally.
    provider_to_log = "unknown_provider"
    model_id_to_log = "unknown_model_id"
    try:
        if "embedding_config" in user_app_config:
            temp_config = EmbeddingConfigSchema(**user_app_config["embedding_config"])
            model_id_to_log = model_id_override or temp_config.default_model_id
            if model_id_to_log in temp_config.models:
                provider_to_log = temp_config.models[model_id_to_log].provider
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass  # Ignore parsing errors here, batch function will handle and log properly

    log_counter("create_embedding_attempt", labels={"provider": provider_to_log, "model_id": model_id_to_log})
    start_time_single = time.time()

    # The create_embeddings_batch function is already decorated with rate limiter and backoff
    embeddings_list = create_embeddings_batch(
        texts=[text], user_app_config=user_app_config, model_id_override=model_id_override  # Pass override if provided
    )

    if not embeddings_list or not embeddings_list[0]:
        # This path should ideally be caught by errors within create_embeddings_batch
        # or the specific embedder if it's a model-specific issue.
        log_counter("create_embedding_failure", labels={"provider": provider_to_log, "model_id": model_id_to_log})
        logger.error(
            f"Failed to generate embedding for single text with model_id '{model_id_to_log}'. Batch returned empty or invalid."
        )
        raise ValueError(f"Embedding generation failed for single text using model_id '{model_id_to_log}'.")

    embedding_data = embeddings_list[0]

    single_time = time.time() - start_time_single
    log_histogram(
        "create_embedding_duration", single_time, labels={"provider": provider_to_log, "model_id": model_id_to_log}
    )
    log_counter("create_embedding_success", labels={"provider": provider_to_log, "model_id": model_id_to_log})
    return embedding_data


def get_embedding_config() -> dict[str, Any]:
    """
    Get the default embedding configuration.
    Returns a configuration dictionary for use with embedding functions.
    """
    from tldw_Server_API.app.core.config import settings

    # Get embedding settings from config
    embedding_settings = settings.get("EMBEDDING_CONFIG", {})

    # Build the configuration in the expected format
    config = {
        "embedding_config": {
            # Use provider:model convention for keys and default_model_id
            "default_model_id": None,
            "models": {},
            "model_storage_base_dir": resolve_model_storage_base_dir(embedding_settings),
        }
    }

    # Add model configurations based on provider
    provider = embedding_settings.get("embedding_provider", "huggingface")
    model = embedding_settings.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    model_id_key = f"{provider}:{model}"

    # Add default configurations for common models - create proper instances
    if provider == "openai":
        config["embedding_config"]["models"][model_id_key] = OpenAIModelCfg(
            provider="openai",
            model_name_or_path=model,
            api_key=embedding_settings.get("embedding_api_key", settings.get("OPENAI_API_KEY", "")),
        )
    elif provider == "huggingface":
        config["embedding_config"]["models"][model_id_key] = HFModelCfg(
            provider="huggingface",
            model_name_or_path=model,
            trust_remote_code=False,
            hf_cache_dir_subpath="huggingface_cache",
        )
    elif provider == "local_api":
        config["embedding_config"]["models"][model_id_key] = LocalAPICfg(
            provider="local_api",
            model_name_or_path=model,
            api_url=embedding_settings.get("embedding_api_url", "http://localhost:8080/v1/embeddings"),
            api_key=embedding_settings.get("embedding_api_key", ""),
        )

    # Add common HuggingFace models that might be requested
    common_hf_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "intfloat/multilingual-e5-large-instruct",
        "Qwen/Qwen3-Embedding-0.6B",
        # Newly added supported models
        "NovaSearch/stella_en_1.5B_v5",
        "NovaSearch/stella_en_400M_v5",
        "jinaai/jina-embeddings-v4",
        "intfloat/multilingual-e5-large",
        "mixedbread-ai/mxbai-embed-large-v1",
        "jinaai/jina-embeddings-v3",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-small-en-v1.5",
    ]

    for hf_model in common_hf_models:
        hf_key = f"huggingface:{hf_model}"
        if hf_key not in config["embedding_config"]["models"]:
            config["embedding_config"]["models"][hf_key] = HFModelCfg(
                provider="huggingface",
                model_name_or_path=hf_model,
                trust_remote_code=False,
                hf_cache_dir_subpath="huggingface_cache",
            )

    # Set default_model_id now that keys are known
    config["embedding_config"]["default_model_id"] = model_id_key

    # Optional: test override for model unload timeout
    # If TEST_EMBEDDINGS_UNLOAD_TIMEOUT_SECONDS (or EMBEDDINGS_UNLOAD_TIMEOUT_SECONDS) is set,
    # apply it to all configured models. This is helpful to shorten timers during pytest runs.
    try:
        timeout_env = os.getenv("TEST_EMBEDDINGS_UNLOAD_TIMEOUT_SECONDS") or os.getenv(
            "EMBEDDINGS_UNLOAD_TIMEOUT_SECONDS"
        )
        if timeout_env:
            timeout_val = int(timeout_env)
            for model_cfg in config["embedding_config"]["models"].values():
                # Pydantic models allow attribute mutation by default
                if hasattr(model_cfg, "unload_timeout_seconds"):
                    model_cfg.unload_timeout_seconds = timeout_val
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as _e:
        # Do not fail configuration if env var is malformed; ignore silently in production path
        pass

    return config


#
# Legacy exports for backward compatibility
# Load embedding configuration from settings
from tldw_Server_API.app.core.config import settings

embedding_config = settings.get("EMBEDDING_CONFIG", {})
embedding_provider = embedding_config.get("embedding_provider", "openai")
embedding_model = embedding_config.get("embedding_model", "text-embedding-3-small")
embedding_api_url = embedding_config.get("embedding_api_url", "http://localhost:8080/v1/embeddings")
embedding_api_key = embedding_config.get("embedding_api_key", "")

#
# End of File.
#######################################################################################################################
