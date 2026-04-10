# embeddings_v5_production_enhanced.py
# Enhanced version with circuit breaker pattern and improved error recovery
"""
Production-ready OpenAI-compatible embeddings API with circuit breaker.

Key enhancements over v5:
- Circuit breaker pattern for fault tolerance
- Improved connection cleanup on failures
- Better error recovery mechanisms
- Enhanced monitoring and observability
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import hashlib
import json
import os
import threading
import time
import uuid
from asyncio import Lock
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from enum import Enum
from fnmatch import fnmatch
from functools import lru_cache
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from typing import Any

import numpy as np
import redis.asyncio as aioredis
import tiktoken
from fastapi import APIRouter, BackgroundTasks, Body, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

# Rate limiting
# Monitoring
# ============================================================================
# Metrics and Monitoring
# ============================================================================
from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    rbac_rate_limit,
    require_permissions,
    require_roles,
)
from tldw_Server_API.app.api.v1.API_Deps.billing_deps import require_within_limit
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory

# Schemas
from tldw_Server_API.app.api.v1.schemas.embeddings_models import (
    CreateEmbeddingRequest,
    CreateEmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import AuditContext, AuditEventCategory, AuditEventType
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ResolvedByokCredentials,
    record_byok_missing_credentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.crypto_utils import derive_hmac_key
from tldw_Server_API.app.core.AuthNZ.permissions import EMBEDDINGS_ADMIN, SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal, is_single_user_principal
from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_profile_mode

# Authentication
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import (
    User,
    get_request_user,
    resolve_user_id_for_request,
)

# Configuration
from tldw_Server_API.app.core.config import settings

# Audit logging: unify later via unified audit DI; legacy import removed (unused here)
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

# Circuit Breaker
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import CircuitBreaker
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import CircuitBreakerOpenError as CircuitBreakerError
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import registry as circuit_breaker_registry
from tldw_Server_API.app.core.Embeddings.dlq_crypto import decrypt_payload_if_present
from tldw_Server_API.app.core.Embeddings.messages import validate_schema
from tldw_Server_API.app.core.Embeddings.request_batching import (
    EmbeddingsRateLimitError,
)
from tldw_Server_API.app.core.Embeddings.request_batching import (
    create_embeddings_batch_async as batching_create_embeddings_batch_async,
)
from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.http_client import (
    RetryPolicy as _RetryPolicy,
)
from tldw_Server_API.app.core.http_client import (
    afetch as _http_afetch,
)
from tldw_Server_API.app.core.http_client import (
    create_async_client as _create_async_client,
)
from tldw_Server_API.app.core.Infrastructure.redis_factory import (
    create_async_redis_client,
    ensure_async_client_closed,
)
from tldw_Server_API.app.core.LLM_Calls.embeddings_adapter_registry import get_embeddings_registry
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, ensure_traceparent, get_ps_logger
from tldw_Server_API.app.core.Resource_Governance.deps import derive_entity_key
from tldw_Server_API.app.core.Resource_Governance.governor import RGRequest
from tldw_Server_API.app.core.Streaming.streams import SSEStream
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode, is_truthy
from tldw_Server_API.app.core.Usage.usage_tracker import (
    backfill_legacy_tokens_to_ledger,
    log_llm_usage,
)

# Exception buckets to replace broad Exception catches while preserving behavior.
try:
    from redis import exceptions as _redis_exceptions

    _REDIS_ERRORS: tuple[type[BaseException], ...] = (_redis_exceptions.RedisError,)
except ImportError:
    _REDIS_ERRORS = ()

_EMBEDDINGS_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    HTTPException,
    CircuitBreakerError,
    EmbeddingsRateLimitError,
    NetworkError,
    RetryExhaustedError,
    AttributeError,
    ConnectionError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
    *_REDIS_ERRORS,
)
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})

# ============================================================================
# Embeddings Implementation Import (Safe/Lazy)
# Avoid hard-failing on import so non-embedding tests can import the app.
# ============================================================================

try:
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        HFModelCfg,
        LocalAPICfg,
        ONNXModelCfg,
        OpenAIModelCfg,
        resolve_model_storage_base_dir,
    )
    EMBEDDINGS_AVAILABLE = True
except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
    # Do not raise here; allow the API to import and mark the embeddings service as unavailable.
    logger.error(f"Embeddings implementation unavailable: {e}")
    logger.error("Embeddings endpoints will respond 503 until dependencies are installed")
    EMBEDDINGS_AVAILABLE = False

    def resolve_model_storage_base_dir(*_args, **_kwargs):
        return "./models/embedding_models_data/"


# Safely get or create metrics
def get_or_create_counter(name, description, labelnames):
    """Get existing counter or create new one"""
    try:
        # Check if metric already exists
        if name in REGISTRY._names_to_collectors:
            collector = REGISTRY._names_to_collectors[name]
            # Verify it's a Counter with matching labels
            if hasattr(collector, '_labelnames') and set(collector._labelnames) == set(labelnames):
                return collector
            # If labels don't match, unregister the old one
            REGISTRY.unregister(collector)
        return Counter(name, description, labelnames)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        # Try to create new counter, handling any registration issues
        try:
            return Counter(name, description, labelnames)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            # Return existing if we can't create new
            if name in REGISTRY._names_to_collectors:
                return REGISTRY._names_to_collectors[name]
            raise

def get_or_create_histogram(name, description, labelnames):
    """Get existing histogram or create new one"""
    try:
        if name in REGISTRY._names_to_collectors:
            collector = REGISTRY._names_to_collectors[name]
            # Verify it's a Histogram with matching labels
            if hasattr(collector, '_labelnames') and set(collector._labelnames) == set(labelnames):
                return collector
            # If labels don't match, unregister the old one
            REGISTRY.unregister(collector)
        return Histogram(name, description, labelnames)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        # Try to create new, or return existing
        try:
            return Histogram(name, description, labelnames)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            if name in REGISTRY._names_to_collectors:
                return REGISTRY._names_to_collectors[name]
            raise

def get_or_create_gauge(name, description, labelnames=None):
    """Get existing gauge or create new one"""
    try:
        if name in REGISTRY._names_to_collectors:
            collector = REGISTRY._names_to_collectors[name]
            expected_labels = set(labelnames) if labelnames else set()
            existing_labels = set(collector._labelnames) if hasattr(collector, '_labelnames') else set()
            if expected_labels == existing_labels:
                return collector
            # If labels don't match, unregister the old one
            REGISTRY.unregister(collector)
        if labelnames:
            return Gauge(name, description, labelnames)
        return Gauge(name, description)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        # Try to create new, or return existing
        try:
            if labelnames:
                return Gauge(name, description, labelnames)
            return Gauge(name, description)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            if name in REGISTRY._names_to_collectors:
                return REGISTRY._names_to_collectors[name]
            raise

# Create metrics using safe getters
embedding_requests_total = get_or_create_counter(
    'embedding_requests_total',
    'Total number of embedding requests',
    ['provider', 'model', 'status']
)

embedding_request_duration = get_or_create_histogram(
    'embedding_request_duration_seconds',
    'Duration of embedding requests',
    ['provider', 'model']
)

embedding_cache_hits = get_or_create_counter(
    'embedding_cache_hits_total',
    'Number of cache hits',
    ['provider', 'model']
)

embedding_cache_size = get_or_create_gauge(
    'embedding_cache_size',
    'Current size of embedding cache'
)

active_embedding_requests = get_or_create_gauge(
    'active_embedding_requests',
    'Number of active embedding requests'
)

# Additional observability counters
embedding_provider_failures = get_or_create_counter(
    'embedding_provider_failures_total',
    'Provider failures by reason',
    ['provider', 'model', 'reason']
)

embedding_fallbacks_total = get_or_create_counter(
    'embedding_fallbacks_total',
    'Count of provider fallbacks taken',
    ['from_provider', 'to_provider']
)

embedding_policy_denied_total = get_or_create_counter(
    'embedding_policy_denied_total',
    'Requests denied by policy',
    ['provider', 'model', 'policy_type']
)

embedding_dimension_adjustments_total = get_or_create_counter(
    'embedding_dimension_adjustments_total',
    'Count of dimension adjustments performed',
    ['provider', 'model', 'method']
)

embedding_token_inputs_total = get_or_create_counter(
    'embedding_token_inputs_total',
    'Number of requests using token array inputs',
    ['mode']  # single or batch
)

byok_oauth_401_retry_total = get_or_create_counter(
    'byok_oauth_401_retry_total',
    'OpenAI OAuth 401 retry outcomes',
    ['provider', 'outcome']
)

# DLQ/admin metrics
dlq_requeued_total = get_or_create_counter(
    'embedding_dlq_requeued_total',
    'Number of DLQ items requeued via admin API',
    ['queue_name', 'status']
)
dlq_requeue_errors_total = get_or_create_counter(
    'embedding_dlq_requeue_errors_total',
    'Errors during DLQ requeue operations',
    ['queue_name', 'error_type']
)

# Orchestrator observability metrics
orchestrator_sse_connections = get_or_create_gauge(
    'orchestrator_sse_connections',
    'Current number of active SSE connections to orchestrator'
)

orchestrator_sse_disconnects_total = get_or_create_counter(
    'orchestrator_sse_disconnects_total',
    'Total number of SSE disconnect events from orchestrator',
    []
)

orchestrator_summary_failures_total = get_or_create_counter(
    'orchestrator_summary_failures_total',
    'Total number of summary failures (fallbacks returned)',
    []
)

# Export queue age and stage flags for Prometheus scraping
embedding_queue_age_current_seconds = get_or_create_gauge(
    'embedding_queue_age_current_seconds',
    'Current age (seconds) of oldest message per queue',
    ['queue_name']
)

embedding_stage_flag = get_or_create_gauge(
    'embedding_stage_flag',
    'Per-stage control flags as gauges (1=true,0=false)',
    ['stage', 'flag']
)

## Backpressure and quotas (configured later; depends on _cfg_int defined below)

# ============================================================================
# Configuration and Constants
# ============================================================================

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    LOCAL_API = "local_api"
    COHERE = "cohere"
    VOYAGE = "voyage"
    GOOGLE = "google"
    MISTRAL = "mistral"
    MLX = "mlx"

# Production configuration
DEFAULT_MAX_BATCH_SIZE = 100
DEFAULT_MAX_CACHE_SIZE = 5000
DEFAULT_CACHE_TTL_SECONDS = 3600
DEFAULT_CACHE_CLEANUP_INTERVAL = 300
DEFAULT_CONNECTION_POOL_SIZE = 20
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3

# Allow overriding via settings/env
def _cfg_int(name: str, default_val: int) -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _settings
        val = _settings.get(name, None)
        if isinstance(val, (int, float)):
            return int(val)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        env = os.getenv(name)
        if env is not None and str(env).strip() != "":
            return int(env)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    return default_val

# Backpressure and quotas configuration
def _cfg_float(name: str, default_val: float) -> float:
    try:
        v = settings.get(name, None)
        if isinstance(v, (int, float)):
            return float(v)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        env = os.getenv(name)
        if env is not None and str(env).strip() != "":
            return float(env)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    return float(default_val)

BP_MAX_DEPTH = int(_cfg_int("EMB_BACKPRESSURE_MAX_DEPTH", 25000))
BP_MAX_AGE_S = _cfg_float("EMB_BACKPRESSURE_MAX_AGE_SECONDS", 300.0)
TENANT_RPS = int(_cfg_int("EMBEDDINGS_TENANT_RPS", 0))  # 0 disables
# Orchestrator snapshot scan cap (prevent unbounded SCAN work per build)
ORCH_SCAN_MAX_KEYS = int(_cfg_int("EMB_ORCH_MAX_SCAN_KEYS", 500))


def _tenant_rps_runtime() -> int:
    """Resolve the effective tenant RPS limit at runtime.

    Prefers the environment variable (handy for tests/overrides) and otherwise
    returns the module-level default.

    Note: The module-level `TENANT_RPS` is already initialized from config/env
    at import time and is also frequently monkeypatched in unit tests. Reading
    `core.config.settings` here would bypass those monkeypatches, so we treat it
    as an initialization-time input only.
    """
    try:
        env_val = os.getenv("EMBEDDINGS_TENANT_RPS")
        if env_val is not None and str(env_val).strip() != "":
            parsed = int(env_val)
            # In pytest, the global conftest sets EMBEDDINGS_TENANT_RPS=0 to
            # disable tenant quotas by default. Allow explicit monkeypatches
            # of TENANT_RPS to override that default within unit tests.
            if (
                parsed <= 0
                and os.getenv("PYTEST_CURRENT_TEST") is not None
                and int(globals().get("TENANT_RPS", 0) or 0) > 0
            ):
                return int(TENANT_RPS)
            return parsed
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    return TENANT_RPS



async def _orchestrator_depth_and_age(client: aioredis.Redis) -> tuple[int, float]:
    """Return (max_queue_depth, max_queue_age_seconds) for core embeddings queues."""
    queues = ["embeddings:chunking", "embeddings:embedding", "embeddings:storage", "embeddings:content"]
    depths = []
    ages = []
    now = time.time()
    for q in queues:
        try:
            d = await client.xlen(q)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            d = 0
        depths.append(int(d or 0))
        try:
            items = await client.xrange(q, "-", "+", count=1)
            if items:
                first_id = items[0][0]
                ts_ms = float(first_id.split("-", 1)[0])
                ages.append(max(0.0, now - (ts_ms / 1000.0)))
            else:
                ages.append(0.0)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            ages.append(0.0)
    return (max(depths) if depths else 0, max(ages) if ages else 0.0)


def _should_enforce_tenant_rps(request: Request) -> bool:
    """
    Decide whether to enforce per-tenant RPS quotas for this request.

    Behaviour:
    - When EMBEDDINGS_TENANT_RPS_PROFILE_AWARE is unset/false (\"0\"/\"false\"/\"off\"):
      fall back to the legacy mode/profile guard via ``_is_multi_user_runtime()``,
      preserving existing behaviour.
    - When the flag is enabled (any other truthy value):
      * Disable tenant quotas for single-user profiles (PROFILE indicating
        local single-user/desktop), regardless of AUTH_MODE.
      * Disable tenant quotas when the authenticated principal is explicitly
        tagged as the single-user profile (helper-detected).
      * Otherwise, treat the runtime as multi-tenant and enforce quotas when
        a positive RPS limit is configured.
    """
    flag = os.getenv("EMBEDDINGS_TENANT_RPS_PROFILE_AWARE", "").strip().lower()
    if flag in {"", "0", "false", "off"}:
        # Compatibility path: preserve legacy AUTH_MODE/profile heuristics.
        return _is_multi_user_runtime()

    principal: AuthPrincipal | None = None
    try:
        ctx = getattr(request.state, "auth", None)
        if isinstance(ctx, AuthContext):
            principal = ctx.principal
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        principal = None

    try:
        single_profile = _is_single_user_profile()
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        single_profile = False

    if single_profile:
        # Local single-user/desktop profiles should not see tenant-style
        # RPS quotas even when AUTH_MODE is misconfigured.
        return False

    # Principal-first: if we can see an explicit single-user principal, do not
    # enforce tenant quotas even under multi-user profiles.
    if principal is not None and is_single_user_principal(principal):  # noqa: SIM103
        return False

    # Multi-tenant runtime: enforce quotas when a positive RPS is configured.
    return True

async def _check_backpressure_and_quotas(request: Request, user: User) -> HTTPException | None:
    """Return HTTPException(429) if backpressure or tenant quota exceeded; else None."""
    # Orchestrator-based backpressure
    try:
        client = await _get_redis_client()
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        client = None
    try:
        if client is not None:
            depth, age = await _orchestrator_depth_and_age(client)
            if depth >= BP_MAX_DEPTH or age >= BP_MAX_AGE_S:
                retry_after = 5
                if age >= BP_MAX_AGE_S:
                    retry_after = min(60, int(max(5, age / 2)))
                headers = {"Retry-After": str(retry_after)}
                return HTTPException(status_code=429, detail="Backpressure: queue overload", headers=headers)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    finally:
        try:
            if client is not None:
                    await ensure_async_client_closed(client)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass

    # Per-tenant quotas in multi-user mode
    try:
        # Read tenant RPS dynamically so tests can monkeypatch env at runtime
        tenant_rps = _tenant_rps_runtime()

        if _should_enforce_tenant_rps(request) and tenant_rps > 0:
            client2 = await _get_redis_client()
            try:
                # Use a single rolling key with 1-second TTL to avoid flakiness across second boundaries
                key = f"embeddings:tenant:rps:{getattr(user, 'id', 'anon')}"
                current = await client2.incr(key)
                # Ensure expiry of 1 second for a strict RPS window
                await client2.expire(key, 1)
                remaining = max(0, tenant_rps - int(current or 0))
                if current > tenant_rps:
                    headers = {"Retry-After": "1", "X-RateLimit-Limit": str(tenant_rps), "X-RateLimit-Remaining": str(0)}
                    return HTTPException(status_code=429, detail="Tenant quota exceeded", headers=headers)
                else:
                    if hasattr(request, 'state'):
                        try:
                            request.state.rate_limit_limit = tenant_rps
                            request.state.rate_limit_remaining = remaining
                        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                            pass
            finally:
                with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                    await ensure_async_client_closed(client2)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    return None


# ============================================================================
# Redis helpers for DLQ admin endpoints
# ============================================================================

async def _get_redis_client() -> aioredis.Redis:
    return await create_async_redis_client(context="embeddings_api")

def _dlq_stream_name(stage: str) -> str:
    stage = stage.strip().lower()
    if stage not in {"chunking", "embedding", "storage", "content"}:
        raise HTTPException(status_code=400, detail="Invalid stage; must be one of chunking|embedding|storage|content")
    return f"embeddings:{stage}:dlq"

def _live_stream_name(stage: str) -> str:
    stage = stage.strip().lower()
    if stage not in {"chunking", "embedding", "storage", "content"}:
        raise HTTPException(status_code=400, detail="Invalid stage; must be one of chunking|embedding|storage|content")
    return f"embeddings:{stage}"

MAX_BATCH_SIZE = _cfg_int("EMBEDDINGS_MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE)
MAX_CACHE_SIZE = _cfg_int("EMBEDDINGS_CACHE_MAX_SIZE", DEFAULT_MAX_CACHE_SIZE)
CACHE_TTL_SECONDS = _cfg_int("EMBEDDINGS_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS)
CACHE_CLEANUP_INTERVAL = _cfg_int("EMBEDDINGS_CACHE_CLEANUP_INTERVAL", DEFAULT_CACHE_CLEANUP_INTERVAL)
CONNECTION_POOL_SIZE = _cfg_int("EMBEDDINGS_CONNECTION_POOL_SIZE", DEFAULT_CONNECTION_POOL_SIZE)
REQUEST_TIMEOUT = _cfg_int("EMBEDDINGS_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
MAX_RETRIES = _cfg_int("EMBEDDINGS_MAX_RETRIES", DEFAULT_MAX_RETRIES)

EMBEDDINGS_PROVIDERS_REQUIRE_KEY = {
    "openai",
    "cohere",
    "voyage",
    "google",
    "mistral",
}


async def _resolve_embeddings_byok(
    provider: str,
    current_user: User | None,
    request: Request | None,
    *,
    force_oauth_refresh: bool = False,
) -> ResolvedByokCredentials:
    user_id_int = getattr(current_user, "id_int", None) if current_user else None
    if user_id_int is None and current_user is not None:
        try:
            user_id_int = int(getattr(current_user, "id", None))
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            user_id_int = None
    return await resolve_byok_credentials(
        provider,
        user_id=user_id_int,
        request=request,
        force_oauth_refresh=force_oauth_refresh,
    )


def _raise_missing_embeddings_key(provider: str) -> None:
    record_byok_missing_credentials(provider, operation="embeddings")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error_code": "missing_provider_credentials",
            "message": f"Embeddings provider '{provider}' requires an API key.",
        },
    )


def _is_http_401_error(exc: BaseException) -> bool:
    try:
        return int(getattr(exc, "status_code", 0) or 0) == status.HTTP_401_UNAUTHORIZED
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        return False


def _record_oauth_401_retry(provider: str, outcome: str) -> None:
    try:
        byok_oauth_401_retry_total.labels(provider=provider, outcome=outcome).inc()
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass


def _is_test_context() -> bool:
    try:
        if os.getenv("PYTEST_CURRENT_TEST"):
            return True
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    return env_flag_enabled("TESTING") or is_test_mode()


def _should_skip_missing_key(
    provider: str,
    credentials: ResolvedByokCredentials | None = None,
) -> bool:
    if not _is_test_context():
        return False
    if credentials is None:
        return True
    source = getattr(credentials, "source", None)
    return not source or source == "none"

# Circuit breaker configuration
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2

# Provider models configuration
PROVIDER_MODELS = {
    EmbeddingProvider.OPENAI: [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large"
    ],
    EmbeddingProvider.COHERE: [
        "embed-english-v3.0",
        "embed-multilingual-v3.0"
    ],
    EmbeddingProvider.HUGGINGFACE: [
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
    ],
    EmbeddingProvider.MLX: [],
}

# Optional allowlists and per-model token limits (override via settings)
def _get_allowed_providers() -> list[str] | None:
    try:
        vals = settings.get("ALLOWED_EMBEDDING_PROVIDERS", [])
        if isinstance(vals, list) and vals:
            return [str(v).lower() for v in vals]
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    return None


def _chroma_manager_for_user(user: User) -> ChromaDBManager:
    cfg = settings.get("EMBEDDING_CONFIG", {}).copy()
    cfg["USER_DB_BASE_DIR"] = settings.get("USER_DB_BASE_DIR")
    user_id = getattr(user, "id", None) or settings.get("SINGLE_USER_FIXED_ID", "1")
    return ChromaDBManager(user_id=str(user_id), user_embedding_config=cfg)


def _split_provider_model(model: str) -> tuple[str | None, str]:
    """Split provider-qualified model IDs like 'openai:model'."""
    if not isinstance(model, str):
        return None, str(model)
    if ":" in model:
        prefix, rest = model.split(":", 1)
        prefix = prefix.strip().lower()
        rest = rest.strip()
        if prefix and rest:
            return prefix, rest
    return None, model


def _resolve_model_and_provider(model: str | None, provider: str | None) -> tuple[str, str]:
    cfg = settings.get("EMBEDDING_CONFIG", {}) or {}
    default_model = model or cfg.get("embedding_model") or cfg.get("default_model_id") or "sentence-transformers/all-MiniLM-L6-v2"
    prefix_provider, stripped_model = _split_provider_model(default_model)
    if provider:
        resolved_provider = provider.lower()
        if prefix_provider and prefix_provider != resolved_provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model provider prefix '{prefix_provider}' does not match provider '{resolved_provider}'",
            )
        return stripped_model, resolved_provider
    if prefix_provider:
        return stripped_model, prefix_provider
    resolved_provider = guess_provider_for_model(stripped_model, None)
    return stripped_model, resolved_provider


def _get_allowed_models() -> list[str] | None:
    try:
        vals = settings.get("ALLOWED_EMBEDDING_MODELS", [])
        if isinstance(vals, list) and vals:
            return [str(v) for v in vals]
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    return None

def _get_model_max_tokens(provider: str, model: str) -> int:
    # Settings-driven override map: {"provider:model": max_tokens} or {"model": max_tokens}
    try:
        mapping = settings.get("EMBEDDING_MODEL_MAX_TOKENS", {}) or {}
        key1 = f"{provider}:{model}"
        if key1 in mapping:
            return int(mapping[key1])
        if model in mapping:
            return int(mapping[model])
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    # Reasonable defaults
    if provider == "openai":
        return 8192
    # Default for HF/local_api/others if not configured
    return 8192


def _build_user_metadata(user: User | None) -> dict[str, Any] | None:
    """Create metadata dict for rate limiter propagation.

    In test contexts (TESTING=true), skip attaching user metadata so that
    the embeddings batcher does not apply rate limiting during tests.
    """
    try:
        # Bypass rate limiting propagation in tests
        if env_flag_enabled("TESTING"):
            return None
        if user is None:
            return None
        user_id = getattr(user, "id", None)
        if user_id is None:
            return None
        return {"user_id": str(user_id)}
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        return None

# ============================================================================
# Enhanced TTL Cache with Better Cleanup
# ============================================================================

class TTLCache:
    """Thread-safe cache with TTL support and automatic cleanup"""

    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.cleanup_task = None
        # Optional daemon-thread cleanup to decouple from app loop
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_stop: threading.Event | None = None
        try:
            self._use_thread = is_truthy(os.getenv("EMBEDDINGS_TTLCACHE_DAEMON", "true"))
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            self._use_thread = True
        self.hits = 0
        self.misses = 0

    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._use_thread:
            # Start daemon thread once
            if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
                self._cleanup_stop = threading.Event()

                def _runner():
                    try:
                        while self._cleanup_stop and not self._cleanup_stop.is_set():
                            with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                                self._cleanup_expired_locked()
                            if self._cleanup_stop:
                                self._cleanup_stop.wait(CACHE_CLEANUP_INTERVAL)
                            else:
                                time.sleep(CACHE_CLEANUP_INTERVAL)
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                        pass

                self._cleanup_thread = threading.Thread(
                    target=_runner,
                    name="embeddings-ttlcache",
                    daemon=True,
                )
                self._cleanup_thread.start()
        else:
            if self.cleanup_task is None:
                self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._use_thread:
            try:
                if self._cleanup_stop:
                    self._cleanup_stop.set()
                # No need to join daemon thread during interpreter teardown, but attempt a brief join
                if self._cleanup_thread and self._cleanup_thread.is_alive():
                    self._cleanup_thread.join(timeout=0.5)
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                pass
            finally:
                self._cleanup_thread = None
                self._cleanup_stop = None
        else:
            if self.cleanup_task:
                self.cleanup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.cleanup_task
                self.cleanup_task = None

    async def _cleanup_loop(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(CACHE_CLEANUP_INTERVAL)
                self._cleanup_expired_locked()
            except asyncio.CancelledError:
                break
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error in cache cleanup: {e}")

    def _cleanup_expired_locked(self):
        """Remove expired entries under the cache lock."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, value in self.cache.items()
                if current_time - value['timestamp'] > self.ttl_seconds
            ]

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                embedding_cache_size.set(len(self.cache))

    async def cleanup_expired(self):
        """Async wrapper for cache cleanup."""
        self._cleanup_expired_locked()

    async def get(self, key: str) -> Any | None:
        """Get value from cache if not expired"""
        with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                self.misses += 1
                return None

            if time.time() - entry['timestamp'] <= self.ttl_seconds:
                entry['last_access'] = time.time()
                self.hits += 1
                return entry['value']

            # Entry expired; remove and count as miss
            del self.cache[key]
            embedding_cache_size.set(len(self.cache))
            self.misses += 1
            return None

    async def set(self, key: str, value: Any):
        """Set value in cache with TTL"""
        with self._lock:
            if len(self.cache) >= self.max_size:
                lru_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k].get('last_access', 0)
                )
                with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                    logger.debug(f"Embeddings TTLCache evict LRU key={lru_key[:8]}..., size={len(self.cache)}")
                del self.cache[lru_key]

            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'last_access': time.time()
            }
            embedding_cache_size.set(len(self.cache))

    async def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            embedding_cache_size.set(0)
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': (self.hits / total_requests) if total_requests else 0.0
            }

# ============================================================================
# Enhanced Connection Pool Manager with Cleanup
# ============================================================================

class ConnectionPoolManager:
    """Manages connection pools with proper cleanup"""

    def __init__(self):
        self.pools: dict[str, Any] = {}
        self.lock = Lock()
        self._closed = False

    async def get_session(self, provider: str) -> Any:
        """Get or create session for provider"""
        async with self.lock:
            if self._closed:
                # Service is spinning back up; allow sessions to be recreated.
                self._closed = False

            existing = self.pools.get(provider)
            if existing is not None and getattr(existing, "is_closed", False):
                # Drop stale client so a fresh one can be created.
                try:
                    close = getattr(existing, "aclose", None)
                    if callable(close):
                        await close()
                    else:
                        close = getattr(existing, "close", None)
                        if callable(close):
                            close()
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    pass
                existing = None
                self.pools.pop(provider, None)

            if provider not in self.pools:
                self.pools[provider] = _create_async_client(timeout=REQUEST_TIMEOUT)
            return self.pools[provider]

    async def close_all(self):
        """Close all connection pools"""
        async with self.lock:
            self._closed = True
            for session in self.pools.values():
                try:
                    close = getattr(session, "aclose", None)
                    if callable(close):
                        await close()
                    else:
                        close = getattr(session, "close", None)
                        if callable(close):
                            close()
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    pass
            self.pools.clear()

    async def remove_provider(self, provider: str):
        """Remove and close specific provider's session"""
        async with self.lock:
            if provider in self.pools:
                try:
                    close = getattr(self.pools[provider], "aclose", None)
                    if callable(close):
                        await close()
                    else:
                        close = getattr(self.pools[provider], "close", None)
                        if callable(close):
                            close()
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    pass
                del self.pools[provider]

# ============================================================================
# Initialize Circuit Breakers for Each Provider
# ============================================================================

def get_or_create_circuit_breaker(provider: str) -> CircuitBreaker:
    """Get or create circuit breaker for provider"""
    breaker_name = f"embeddings_{provider}"
    breaker = circuit_breaker_registry.get(breaker_name)

    if not breaker:
        breaker = CircuitBreaker(
            name=breaker_name,
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            expected_exception=(ConnectionError, TimeoutError, NetworkError, RetryExhaustedError),
            success_threshold=CIRCUIT_BREAKER_SUCCESS_THRESHOLD
        )
        circuit_breaker_registry.register(breaker)

    return breaker

# ============================================================================
# Global Instances
# ============================================================================

embedding_cache = TTLCache()
connection_manager = ConnectionPoolManager()

@asynccontextmanager
async def _embeddings_router_lifespan(app):
    # Startup
    logger.info("Starting embeddings service v5 enhanced (with circuit breaker)")
    await embedding_cache.start_cleanup_task()
    if not EMBEDDINGS_AVAILABLE:
        logger.error("Embeddings implementation not available - service will not function")
    try:
        ci = os.getenv("CI", "").lower() == "true"
        auto_dl = os.getenv("AUTO_DOWNLOAD_MODELS", "true").lower() == "true"
        if ci and auto_dl:
            async def _preload_models_on_startup():
                try:
                    cfg = settings.get("EMBEDDING_CONFIG", {}) or {}
                    preload_list = []
                    env_models = os.getenv("PRELOAD_EMBEDDING_MODELS")
                    if env_models:
                        preload_list.extend([m.strip() for m in env_models.split(",") if m.strip()])
                    try:
                        cfg_preload = cfg.get("preload_models", []) or []
                        if isinstance(cfg_preload, list):
                            preload_list.extend([str(m).strip() for m in cfg_preload if str(m).strip()])
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                        pass
                    default_model = cfg.get("embedding_model") or cfg.get("default_model_id") or "sentence-transformers/all-MiniLM-L6-v2"
                    default_provider = cfg.get("embedding_provider") or "huggingface"
                    if default_model:
                        if ":" in default_model:
                            preload_list.append(default_model)
                        else:
                            preload_list.append(f"{default_provider}:{default_model}")
                    seen = set()
                    final_models = []
                    for m in preload_list:
                        if m and m not in seen:
                            seen.add(m)
                            final_models.append(m)
                    if final_models:
                        logger.info(f"CI detected; preloading {len(final_models)} embedding model(s): {final_models}")
                        for full in final_models:
                            try:
                                if ":" in full:
                                    prov, mdl = full.split(":", 1)
                                    provider = prov.strip().lower()
                                    model = mdl.strip()
                                else:
                                    model = full.strip()
                                    provider = guess_provider_for_model(model)
                                if not is_model_allowed(provider, model):
                                    logger.warning(f"Skipping preload for disallowed model {provider}:{model}")
                                    continue
                                if provider == "openai" and not settings.get("OPENAI_API_KEY"):
                                    logger.info("Skipping OpenAI preload due to missing OPENAI_API_KEY")
                                    continue
                                await create_embeddings_batch_async(texts=["ci preload"], provider=provider, model_id=model)
                                logger.info(f"Preloaded model {provider}:{model}")
                            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                                logger.warning(f"Failed to preload model {full}: {e}")
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Unexpected error during preload task: {e}")
            asyncio.create_task(_preload_models_on_startup())
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to schedule model preloads: {e}")
    logger.info("Embeddings service started successfully")

    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down embeddings service")
        await embedding_cache.stop_cleanup_task()
        await connection_manager.close_all()
        logger.info("Embeddings service shutdown complete")

router = APIRouter(
    tags=["embeddings"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    },
    lifespan=_embeddings_router_lifespan,
)


# Implemented provider set for 501 guard
IMPLEMENTED_PROVIDERS = {"openai", "huggingface", "onnx", "local_api", "cohere", "google"}


# Register cleanup on process exit
def cleanup_on_exit():
    """Synchronous cleanup for process exit"""
    # Avoid logging in atexit, as sinks may already be closed.
    # Prefer asyncio.run to avoid relying on a possibly-missing current loop in 3.11+.
    try:
        try:
            asyncio.run(embedding_cache.stop_cleanup_task())
        except RuntimeError:
            # If a running loop prevents asyncio.run, fall back to best-effort
            pass
        with suppress(RuntimeError):
            asyncio.run(connection_manager.close_all())
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        # Swallow any errors during interpreter teardown
        pass

atexit.register(cleanup_on_exit)

# ============================================================================
# Helper Functions
# ============================================================================

@lru_cache(maxsize=128)
def get_tokenizer(model_name: str):
    """Get or create a tokenizer for the model"""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"No tokenizer for model '{model_name}', using cl100k_base")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in a string"""
    try:
        encoding = get_tokenizer(model_name)
        return len(encoding.encode(text))
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Token counting failed: {e}, estimating")
        return len(text) // 4

def get_cache_key(
    text: str,
    provider: str,
    model: str,
    dimensions: int | None = None,
    backend_identity: str | None = None,
) -> str:
    """Generate cache key for embedding"""
    key_parts = [text, provider, model]
    if dimensions:
        key_parts.append(str(dimensions))
    if backend_identity:
        key_parts.append(backend_identity)
    key_string = "|".join(key_parts)
    return hashlib.pbkdf2_hmac(
        "sha256",
        key_string.encode("utf-8"),
        _embedding_cache_key_secret(),
        _EMBEDDING_CACHE_KEY_PBKDF2_ITERATIONS,
        dklen=32,
    ).hex()


_EMBEDDING_CACHE_KEY_PBKDF2_ITERATIONS = 2048


@lru_cache(maxsize=1)
def _embedding_cache_key_secret() -> bytes:
    """Return a stable keyed-hash secret for embedding cache partitioning."""
    try:
        return derive_hmac_key()
    except Exception:
        # Keep cache keys deterministic in dev/test even when AuthNZ secrets are absent.
        return b"tldw_embeddings_cache_hmac_fallback"


_SENSITIVE_QUERY_KEYS = frozenset({
    "access_token", "api_key", "apikey", "auth", "bearer",
    "credential", "key", "passwd", "password", "secret", "token",
})


def _sanitize_query(query: str) -> str:
    """Remove known sensitive query params, keep the rest sorted for determinism."""
    params = parse_qs(query, keep_blank_values=True)
    filtered = {
        k: v for k, v in params.items()
        if k.lower() not in _SENSITIVE_QUERY_KEYS
    }
    if not filtered:
        return ""
    return urlencode(sorted(filtered.items()), doseq=True)


def _normalize_cache_backend_identity(config: dict[str, Any], provider: str) -> str | None:
    """Derive stable backend identity for cache partitioning."""
    if provider != "local_api":
        return None

    api_url = str(config.get("api_url") or "").strip()
    if not api_url:
        return None

    parsed = urlsplit(api_url)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        sanitized_query = _sanitize_query(parsed.query)
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), sanitized_query, ""))
    return api_url.rstrip("/")


# ---------------------------------------------------------------------------
# On-demand vector compaction (admin only)
# ---------------------------------------------------------------------------

class CompactorRunRequest(BaseModel):
    user_id: str | None = Field(default=None, description="Target user_id; defaults to current admin in single-user mode")
    media_db_path: str | None = Field(default=None, description="Override path to the per-user media database; defaults to settings")


class CompactorRunResponse(BaseModel):
    user_id: str
    collections_touched: int
    ts: float


@router.post(
    "/embeddings/compactor/run",
    response_model=CompactorRunResponse,
    summary="Run a one-shot vector compaction for a user (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def run_compactor_once(
    req: CompactorRunRequest,
    current_user: User = Depends(get_request_user),
):
    try:
        # Lazy import to avoid heavy imports on module import
        from tldw_Server_API.app.core.Embeddings.services.vector_compactor import (
            compact_once as _compact_once,  # type: ignore
        )
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.exception("Compactor module import failed")
        raise HTTPException(status_code=503, detail="Compactor unavailable") from exc
    uid = str(req.user_id or current_user.id)
    try:
        touched = await _compact_once(uid, db_path=req.media_db_path or None)
        return CompactorRunResponse(user_id=uid, collections_touched=int(touched or 0), ts=datetime.utcnow().timestamp())
    except HTTPException:
        raise
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.exception(
            "Compactor run failed for user '{}' with db_path_override={}",
            uid,
            bool(req.media_db_path),
        )
        raise HTTPException(status_code=500, detail="Compactor run failed") from e

# ============================================================================
# Token-array handling and dimension adjustment helpers
# ============================================================================

def tokens_to_texts(
    tokens_input: list[int] | list[list[int]],
    model_name: str
) -> tuple[list[str], int, list[int]]:
    """Convert token arrays to text using model tokenizer when possible.

    Returns (texts, total_token_count, per_input_token_counts).
    Uses tiktoken encoding_for_model or cl100k_base fallback.
    """
    try:
        enc = get_tokenizer(model_name)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        enc = tiktoken.get_encoding("cl100k_base")

    texts: list[str] = []
    total_tokens = 0
    token_counts: list[int] = []
    # Single token array
    if tokens_input and isinstance(tokens_input, list) and tokens_input and isinstance(tokens_input[0], int):
        arr = tokens_input  # type: ignore[assignment]
        total_tokens += len(arr)
        token_counts.append(len(arr))
        try:
            texts.append(enc.decode(arr))
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Failed to decode token array for model '{}' (index 0, tokens={}): {}",
                model_name,
                len(arr),
                exc,
            )
            raise ValueError("Invalid token array input") from exc
        return texts, total_tokens, token_counts

    # Batch of token arrays
    if tokens_input and isinstance(tokens_input, list):
        for idx, arr in enumerate(tokens_input):  # type: ignore[assignment]
            if not isinstance(arr, list) or not all(isinstance(x, int) for x in arr):
                raise ValueError("Invalid token array format")
            total_tokens += len(arr)
            token_counts.append(len(arr))
            try:
                texts.append(enc.decode(arr))
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    "Failed to decode token array for model '{}' (index {}, tokens={}): {}",
                    model_name,
                    idx,
                    len(arr),
                    exc,
                )
                raise ValueError("Invalid token array input") from exc
        return texts, total_tokens, token_counts

    raise ValueError("Invalid token array input")

def _dimension_policy() -> str:
    # reduce (slice), pad, or ignore
    try:
        val = os.getenv("EMBEDDINGS_DIMENSION_POLICY", "reduce").lower()
        if val in ("reduce", "pad", "ignore"):
            return val
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    return "reduce"


def _supports_openai_dimensions(model: str) -> bool:
    """Return True when OpenAI model supports dimensions parameter."""
    model_key = (model or "").split(":", 1)[-1]
    return model_key.startswith("text-embedding-3")

def _validate_dimensions_request(provider: str, model: str, dimensions: int | None) -> int | None:
    """Validate requested dimensions for the provider/model pair."""
    if dimensions is None:
        return None
    try:
        dim = int(dimensions)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="dimensions must be an integer",
        ) from exc
    if dim <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"dimensions must be positive, got {dim}",
        )

    provider_key = (provider or "").lower()
    model_key = (model or "").split(":", 1)[-1]
    if provider_key == "openai":
        if not _supports_openai_dimensions(model):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="dimensions is only supported for OpenAI text-embedding-3 models",
            )
        max_dims = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
        max_dim = max_dims.get(model_key)
        if max_dim is not None and dim > max_dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"dimensions {dim} exceeds maximum {max_dim} for model {model_key}",
            )
    else:
        if dim > 4096:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="dimensions must be <= 4096 for non-OpenAI providers",
            )

    return dim

def adjust_dimensions(
    vectors: list[list[float]],
    target_dim: int | None,
    provider: str,
    model: str
) -> list[list[float]]:
    if not target_dim or target_dim <= 0:
        return vectors
    policy = _dimension_policy()
    adjusted: list[list[float]] = []
    for v in vectors:
        if not isinstance(v, (list, tuple)):
            adjusted.append(v)
            continue
        arr = np.asarray(v, dtype=np.float32)
        cur = arr.shape[0]
        if cur == target_dim or policy == "ignore":
            adjusted.append(arr.tolist())
            continue
        if cur > target_dim:
            # reduce by slicing first-N
            out = arr[:target_dim]
            adjusted.append(out.tolist())
            embedding_dimension_adjustments_total.labels(provider=provider, model=model, method="reduce").inc()
        else:
            if policy == "pad":
                # zero-pad
                pad = np.zeros((target_dim - cur,), dtype=np.float32)
                out = np.concatenate([arr, pad], axis=0)
                adjusted.append(out.tolist())
                embedding_dimension_adjustments_total.labels(provider=provider, model=model, method="pad").inc()
            else:
                # reduce policy cannot expand; return as-is
                adjusted.append(arr.tolist())
    return adjusted

def decide_and_apply_l2(
    embedding: list[float] | np.ndarray,
    encoding_format: str,
    embeddings_from_adapter: bool,
) -> tuple[np.ndarray, bool]:
    """Decide and apply L2-normalization policy.

    Policy:
    - Base64 outputs: never L2-normalize (numeric representation is not returned).
    - Numeric outputs: L2-normalize by default.
    - Adapter-supplied vectors are preserved as-is unless LLM_EMBEDDINGS_L2_NORMALIZE is truthy.

    Returns (arr, did_l2) where arr is a float32 numpy array (possibly normalized).
    If an unexpected error occurs while reading env vars or applying L2, logs with context
    and preserves default behavior (numeric outputs normalized; adapter vectors preserved
    unless normalization is explicitly requested via env flag).
    """
    # Default: normalize for numeric outputs; never for base64
    do_l2 = encoding_format != "base64"

    normalize_requested: bool | None = None
    try:
        env_val = os.getenv("LLM_EMBEDDINGS_L2_NORMALIZE", "")
        normalize_requested = is_truthy(env_val)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        # Preserve default behavior on error; log with context
        logger.warning(
            "Error reading env var LLM_EMBEDDINGS_L2_NORMALIZE in decide_and_apply_l2; "
            f"encoding_format={encoding_format}, embeddings_from_adapter={embeddings_from_adapter}: {e}"
        )
        normalize_requested = None

    # Adapter vectors: preserve as-is unless normalization explicitly requested
    if embeddings_from_adapter:
        do_l2 = encoding_format != "base64" if normalize_requested is True else False

    try:
        arr = np.asarray(embedding, dtype=np.float32)
        if do_l2:
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        return arr, do_l2
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        # Log error and return original values (converted to float32 if possible)
        logger.error(
            "Error applying L2 policy in decide_and_apply_l2 "
            f"(LLM_EMBEDDINGS_L2_NORMALIZE, encoding_format={encoding_format}, "
            f"embeddings_from_adapter={embeddings_from_adapter}): {e}"
        )
        try:
            arr = np.asarray(embedding, dtype=np.float32)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            # Last resort: best-effort array without dtype guarantee
            arr = np.array(embedding)
        return arr, False

def _resolve_auth_principal_from_request(request: Request | None) -> AuthPrincipal | None:
    """Best-effort extraction of AuthPrincipal from request state."""
    if request is None:
        return None
    try:
        ctx = getattr(request.state, "auth", None)
        if isinstance(ctx, AuthContext):
            return ctx.principal
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        return None
    return None


def _is_policy_bypass_admin(principal: AuthPrincipal | None, user: User | None) -> bool:
    """
    Determine whether policy checks should allow admin bypass.

    Claim-first behavior:
    - Trust only principal role/permission claims (`admin`, `*`, `system.configure`).
    - Absence of a principal means no bypass.
    """
    _ = user
    if principal is None:
        return False
    try:
        roles = {str(role).strip().lower() for role in (principal.roles or [])}
        permissions = {
            str(permission).strip().lower()
            for permission in (principal.permissions or [])
            if str(permission).strip()
        }
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        roles = set()
        permissions = set()
    return bool(("admin" in roles) or (permissions & _ADMIN_CLAIM_PERMISSIONS))


def _should_enforce_policy(user: User | None = None, principal: AuthPrincipal | None = None) -> bool:
    # 1) Explicit env override takes highest precedence
    env_val = os.getenv("EMBEDDINGS_ENFORCE_POLICY")
    if env_val is not None:
        return is_truthy(env_val)
    # 2) In TESTING, always enforce (even for admin) for deterministic behavior
    if env_flag_enabled("TESTING"):
        return True
    # 3) Settings-level boolean if provided
    try:
        cfg_val = settings.get("EMBEDDINGS_ENFORCE_POLICY", None)
        if isinstance(cfg_val, bool):
            return cfg_val
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    # 4) Admin bypass unless strict enforcement requested
    try:
        if (
            _is_policy_bypass_admin(principal=principal, user=user)
            and not is_truthy(os.getenv("EMBEDDINGS_ENFORCE_POLICY_STRICT", "false"))
        ):
            return False
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    # Default: do not enforce
    return False


def _should_enforce_policy_for_request(
    request: Request | None,
    user: User | None = None,
) -> bool:
    """
    Request-aware policy enforcement with claim-first principal handling.

    Includes a compatibility fallback for tests that monkeypatch
    `_should_enforce_policy` with a single-argument callable.
    """
    principal = _resolve_auth_principal_from_request(request)
    try:
        return _should_enforce_policy(user, principal)
    except TypeError:
        return _should_enforce_policy(user)

def resolve_fallback_chain(primary_provider: str) -> list[str]:
    # Configurable chain; else default
    try:
        mapping = settings.get("EMBEDDINGS_FALLBACK_CHAIN", {}) or {}
        if isinstance(mapping, dict):
            chain = mapping.get(primary_provider, None)
            if isinstance(chain, list) and chain:
                return [primary_provider] + [p for p in chain if isinstance(p, str)]
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    defaults = {
        "openai": ["openai", "huggingface", "onnx", "local_api"],
        "huggingface": ["huggingface", "onnx", "local_api"],
        "onnx": ["onnx", "huggingface", "local_api"],
        "local_api": ["local_api", "huggingface"],
    }
    return defaults.get(primary_provider, [primary_provider])

def _fallback_model_map() -> dict[str, dict[str, str]]:
    """Return mapping for provider-specific model fallbacks.

    Shape: {"<src_provider>:<src_model>": {"<dst_provider>": "<dst_model>"}}
    """
    try:
        m = settings.get("EMBEDDINGS_FALLBACK_MODEL_MAP", None)
        if isinstance(m, dict) and m:
            return m
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    # Sensible defaults for common OpenAI → HF mapping
    return {
        "openai:text-embedding-3-small": {
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            "onnx": "sentence-transformers/all-MiniLM-L6-v2",
            "local_api": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "openai:text-embedding-3-large": {
            "huggingface": "sentence-transformers/all-mpnet-base-v2",
            "onnx": "sentence-transformers/all-mpnet-base-v2",
            "local_api": "sentence-transformers/all-mpnet-base-v2",
        },
        "openai:text-embedding-ada-002": {
            "huggingface": "sentence-transformers/all-mpnet-base-v2",
            "onnx": "sentence-transformers/all-mpnet-base-v2",
            "local_api": "sentence-transformers/all-mpnet-base-v2",
        },
    }

def map_model_for_provider(src_provider: str, dst_provider: str, model_id: str) -> str:
    """Map a model id to the destination provider if a mapping exists."""
    if not src_provider or not dst_provider:
        return model_id
    if src_provider == dst_provider:
        return model_id
    key = f"{src_provider}:{model_id}"
    mapping = _fallback_model_map()
    try:
        dst_map = mapping.get(key, {})
        mapped = dst_map.get(dst_provider)
        if isinstance(mapped, str) and mapped:
            return mapped
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    return model_id

# Models that require trust_remote_code=True for HuggingFace loading
def _hf_trusts_remote_code(model_name: str) -> bool:
    try:
        patterns = settings.get("TRUSTED_HF_REMOTE_CODE_MODELS", []) or []
        for pat in patterns:
            if fnmatch(model_name, pat) or fnmatch(model_name.lower(), pat.lower()):
                logger.info(f"HF trust_remote_code enabled for model '{model_name}' (matched '{pat}')")
                return True
        return False
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Failed to evaluate TRUSTED_HF_REMOTE_CODE_MODELS for '{model_name}': {e}")
        return False

# ============================================================================
# Public Configuration Endpoint
# ============================================================================

@router.get("/embeddings/providers-config", summary="List configured embedding providers and models")
async def get_embeddings_providers_config(current_user: User = Depends(get_request_user)):
    """Return enabled providers and their models from the simplified embeddings config.

    Response:
        {
          "default_provider": str,
          "default_model": str,
          "providers": [ {"name": str, "models": [str, ...]}, ... ]
        }
    """
    try:
        from tldw_Server_API.app.core.Embeddings.simplified_config import get_config as _get_cfg
        cfg = _get_cfg()
        providers = []
        for p in cfg.get_enabled_providers():
            providers.append({
                "name": p.name,
                "models": list(p.models or [])
            })
        return {
            "default_provider": cfg.default_provider,
            "default_model": cfg.default_model,
            "providers": providers,
        }
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to read embeddings providers config: {e}")
        raise HTTPException(status_code=500, detail="Failed to load embeddings configuration") from e

# ============================================================================
# Models and Warmup/Download Utilities
# ============================================================================

def is_model_allowed(provider: str, model: str) -> bool:
    providers = _get_allowed_providers()
    models = _get_allowed_models()
    if providers is not None and provider.lower() not in providers:
        return False
    if models is not None:
        for pat in models:
            if pat.endswith("*") and model.startswith(pat[:-1]):
                return True
            if model == pat:
                return True
        return False
    return True

def guess_provider_for_model(model: str, explicit_provider: str | None = None) -> str:
    if explicit_provider:
        return explicit_provider.lower()
    if ":" in model:
        p, _ = model.split(":", 1)
        return p.lower()
    # Heuristic for HF-style ids
    if ("/" in model or model.startswith((
        "sentence-transformers/","BAAI/","thenlper/","intfloat/","hkunlp/","Qwen/","microsoft/",
        "google/","facebook/","all-MiniLM-","all-mpnet-","bert-","roberta-","xlm-","distilbert-"
    ))) and model not in ["text-embedding-3-small","text-embedding-3-large","text-embedding-ada-002"]:
        return "huggingface"
    return "openai"

# ============================================================================
# Provider Configuration Builders
# ============================================================================

def build_provider_config(
    provider: EmbeddingProvider,
    model: str | None,
    api_key: str | None = None,
    api_url: str | None = None,
    dimensions: int | None = None
) -> dict[str, Any]:
    """Build provider-specific configuration"""
    if not model:
        raise ValueError(f"model is required for provider {provider.value}")

    if provider == EmbeddingProvider.OPENAI:
        if dimensions is not None:
            if dimensions <= 0:
                raise ValueError(f"dimensions must be positive, got {dimensions}")
            max_dims = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
            model_key = (model or "").split(":", 1)[-1]
            max_dim = max_dims.get(model_key)
            if max_dim is not None and dimensions > max_dim:
                raise ValueError(f"dimensions {dimensions} exceeds maximum {max_dim} for model {model_key}")
        config = {
            "provider": "openai",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("OPENAI_API_KEY"),
        }
        if dimensions is not None:
            config["dimensions"] = dimensions
        return config
    elif provider == EmbeddingProvider.HUGGINGFACE:
        return {
            "provider": "huggingface",
            "model_name_or_path": model,
            "trust_remote_code": _hf_trusts_remote_code(model),
            "hf_cache_dir_subpath": "huggingface_cache",
        }
    elif provider == EmbeddingProvider.COHERE:
        return {
            "provider": "cohere",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("COHERE_API_KEY"),
        }
    elif provider == EmbeddingProvider.VOYAGE:
        return {
            "provider": "voyage",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("VOYAGE_API_KEY"),
        }
    elif provider == EmbeddingProvider.GOOGLE:
        return {
            "provider": "google",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("GOOGLE_API_KEY"),
        }
    elif provider == EmbeddingProvider.MISTRAL:
        return {
            "provider": "mistral",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("MISTRAL_API_KEY"),
        }
    elif provider == EmbeddingProvider.ONNX:
        return {
            "provider": "onnx",
            "model_name_or_path": model,
        }
    elif provider == EmbeddingProvider.LOCAL_API:
        return {
            "provider": "local_api",
            "model_name_or_path": model,
            "api_url": api_url or settings.get("LOCAL_API_URL"),
        }
    elif provider == EmbeddingProvider.MLX:
        return {
            "provider": "mlx",
            "model_name_or_path": model,
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ============================================================================
# Enhanced Embedding Function with Circuit Breaker
# ============================================================================

async def create_embeddings_with_circuit_breaker(
    texts: list[str],
    provider: str,
    model_id: str,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Create embeddings with circuit breaker protection"""
    breaker = get_or_create_circuit_breaker(provider)

    try:
        # Use circuit breaker to protect the call
        async def _create():
            # Build proper typed ModelCfg based on provider
            if provider == "huggingface":
                model_cfg = HFModelCfg(
                    provider="huggingface",
                    model_name_or_path=config.get("model_name_or_path", model_id),
                    trust_remote_code=config.get("trust_remote_code", False),
                    hf_cache_dir_subpath=config.get("hf_cache_dir_subpath", "huggingface_cache"),
                )
            elif provider == "openai":
                dim_override = dimensions if dimensions is not None else config.get("dimensions")
                model_cfg = OpenAIModelCfg(
                    provider="openai",
                    model_name_or_path=config.get("model_name_or_path", model_id),
                    dimensions=dim_override,
                )
            elif provider == "onnx":
                model_cfg = ONNXModelCfg(
                    provider="onnx",
                    model_name_or_path=config.get("model_name_or_path", model_id),
                    onnx_storage_dir_subpath=config.get("onnx_storage_dir_subpath", "onnx_models"),
                )
            elif provider == "local_api":
                model_cfg = LocalAPICfg(
                    provider="local_api",
                    model_name_or_path=config.get("model_name_or_path", model_id),
                    api_url=config.get("api_url"),
                    api_key=config.get("api_key"),
                )
            elif provider == "cohere":
                # Direct async call to Cohere embeddings
                api_key = config.get("api_key") or settings.get("COHERE_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Cohere API key not configured")
                mdl = config.get("model_name_or_path", model_id) or "embed-english-v3.0"
                client = await connection_manager.get_session(provider)
                url = "https://api.cohere.com/v1/embed"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {"model": mdl, "texts": texts, "input_type": "search_document"}
                resp = await _http_afetch(
                    method="POST",
                    url=url,
                    client=client,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                    retry=_RetryPolicy(attempts=1),
                )
                try:
                    status_code = int(getattr(resp, "status_code", 0))
                    if status_code >= 400:
                        try:
                            detail = resp.text
                        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                            detail = ""
                        raise HTTPException(status_code=status_code, detail=f"Cohere error: {detail}")
                    data = resp.json()
                finally:
                    close = getattr(resp, "aclose", None)
                    if callable(close):
                        await close()
                    else:
                        close = getattr(resp, "close", None)
                        if callable(close):
                            close()
                embs = None
                try:
                    if isinstance(data.get("embeddings"), list):
                        embs = data["embeddings"]
                    elif isinstance(data.get("embeddings"), dict) and "float" in data["embeddings"]:
                        embs = data["embeddings"]["float"]
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    embs = None
                if not embs:
                    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Invalid Cohere response format")
                return embs
            elif provider == "google":
                # Direct async call to Google Generative Language API (text-embedding-004)
                api_key = config.get("api_key") or settings.get("GOOGLE_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google API key not configured")
                raw_model = config.get("model_name_or_path", model_id) or "models/text-embedding-004"
                model_name = raw_model if raw_model.startswith("models/") else f"models/{raw_model}"
                client = await connection_manager.get_session(provider)
                base = "https://generativelanguage.googleapis.com/v1beta"
                url = f"{base}/{model_name}:batchEmbedContents?key={api_key}"
                reqs = [{"model": model_name, "content": {"parts": [{"text": t}]}} for t in texts]
                payload = {"requests": reqs}
                resp = await _http_afetch(
                    method="POST",
                    url=url,
                    client=client,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                    retry=_RetryPolicy(attempts=1),
                )
                try:
                    status_code = int(getattr(resp, "status_code", 0))
                    if status_code >= 400:
                        try:
                            detail = resp.text
                        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                            detail = ""
                        raise HTTPException(status_code=status_code, detail=f"Google Embeddings error: {detail}")
                    data = resp.json()
                finally:
                    close = getattr(resp, "aclose", None)
                    if callable(close):
                        await close()
                    else:
                        close = getattr(resp, "close", None)
                        if callable(close):
                            close()
                embs = []
                try:
                    items = data.get("embeddings") or []
                    for it in items:
                        vec = it.get("values") or it.get("embedding") or []
                        if isinstance(vec, dict) and "values" in vec:
                            vec = vec["values"]
                        if not isinstance(vec, list):
                            raise ValueError("invalid embedding vector")
                        embs.append(vec)
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    embs = []
                if not embs or len(embs) != len(texts):
                    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Invalid Google embeddings response format")
                return embs
            elif provider == "mlx":
                loop = asyncio.get_running_loop()
                registry = get_embeddings_registry()
                adapter = registry.get_adapter("mlx")
                if adapter is None:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="MLX embeddings adapter unavailable")

                try:
                    resp = await loop.run_in_executor(
                        None,
                        lambda: adapter.embed({"input": texts, "model": model_id}),
                    )
                    data = resp.get("data") if isinstance(resp, dict) else None
                    if not data:
                        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="MLX embeddings returned empty data")
                    return [item.get("embedding", []) for item in data]
                except HTTPException:
                    raise
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
                    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"MLX embeddings error: {exc}") from exc
            else:
                raise ValueError(f"Unknown provider: {provider}")

            # Wrap config in expected structure for embeddings service batch helper
            # Include explicit defaults so batching helper does not fall back to OpenAI
            provider_qualified_id = f"{provider}:{model_id}"
            app_config = {
                "embedding_config": {
                    "default_model_id": provider_qualified_id,
                    "default_provider": provider,
                    "default_model": model_id,
                    "model_storage_base_dir": resolve_model_storage_base_dir(),
                    "models": {provider_qualified_id: model_cfg},
                }
            }

            # Pass provider-qualified override to avoid implicit defaults inside the batcher
            return await batching_create_embeddings_batch_async(
                texts=texts,
                config=app_config,
                model_id_override=provider_qualified_id,
                metadata=metadata,
            )

        return await breaker.call_async(_create)

    except CircuitBreakerError as e:
        logger.warning(f"Circuit breaker open for {provider}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service temporarily unavailable for provider {provider}. Please try again later."
        ) from e
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to create embeddings with {provider}: {e}")
        raise

async def create_embeddings_batch_async(
    texts: list[str],
    provider: str,
    model_id: str | None = None,
    dimensions: int | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[list[float]]:
    """Async wrapper for embeddings with caching and circuit breaker"""
    provider = (provider or "").strip().lower()

    if model_id:
        prefix_provider, stripped_model = _split_provider_model(model_id)
        if prefix_provider:
            if provider and prefix_provider != provider.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Model provider prefix '{prefix_provider}' does not match provider '{provider}'"
                    ),
                )
            provider = prefix_provider
            model_id = stripped_model

    embeddings = []
    uncached_texts = []
    uncached_indices = []

    try:
        provider_enum = EmbeddingProvider(provider)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {provider}"
        ) from None

    try:
        _validate_dimensions_request(provider, model_id or "", dimensions)
        config = build_provider_config(
            provider_enum,
            model_id,
            api_key,
            api_url,
            dimensions,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    backend_identity = _normalize_cache_backend_identity(config, provider)

    # Check cache
    for i, text in enumerate(texts):
        cache_key = get_cache_key(
            text,
            provider,
            model_id or "default",
            dimensions,
            backend_identity=backend_identity,
        )
        cached = await embedding_cache.get(cache_key)

        if cached:
            embeddings.append(cached)
            # Ensure Prometheus labels are always strings
            embedding_cache_hits.labels(provider=provider, model=(model_id or "default")).inc()
        else:
            embeddings.append(None)
            uncached_texts.append(text)
            uncached_indices.append(i)

    # Process uncached texts
    if uncached_texts:
        # Process in batches with circuit breaker (or synthesize in test mode for OpenAI)
        all_new_embeddings = []
        if (
            provider == "openai"
            and env_flag_enabled("TESTING")
            and not env_flag_enabled("USE_REAL_OPENAI_IN_TESTS")
        ):
            import numpy as _np
            mdl = (model_id or "text-embedding-3-small").lower()
            dim = 1536
            if "3-large" in mdl:
                dim = 3072
            for t in uncached_texts:
                seed = int(hashlib.sha256(((model_id or "") + "|" + t).encode("utf-8")).hexdigest()[:16], 16)
                rng = _np.random.default_rng(seed)
                vec = rng.standard_normal(dim, dtype=_np.float32)
                nrm = _np.linalg.norm(vec)
                if nrm > 0:
                    vec = vec / nrm
                all_new_embeddings.append(vec.tolist())
        else:
            for batch_start in range(0, len(uncached_texts), MAX_BATCH_SIZE):
                batch_end = min(batch_start + MAX_BATCH_SIZE, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]

                try:
                    batch_embeddings = await create_embeddings_with_circuit_breaker(
                        batch_texts,
                        provider,
                        model_id,
                        config,
                        metadata=metadata,
                        dimensions=dimensions,
                    )
                    if not isinstance(batch_embeddings, list) or len(batch_embeddings) != len(batch_texts):
                        batch_count = len(batch_embeddings) if isinstance(batch_embeddings, list) else "invalid"
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=(
                                f"Embedding provider returned {batch_count} embeddings, "
                                f"expected {len(batch_texts)} for batch"
                            ),
                        )
                    all_new_embeddings.extend(batch_embeddings)
                except EmbeddingsRateLimitError as e:
                    headers = {"Retry-After": str(e.retry_after)} if e.retry_after else None
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded",
                        headers=headers,
                    ) from e
                except HTTPException:
                    raise
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Failed to create embeddings for batch: {e}")

                    # Try to close and recreate connection for this provider
                    await connection_manager.remove_provider(provider)

                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Embedding service error: {str(e)}"
                    ) from e

        if len(all_new_embeddings) != len(uncached_texts):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Embedding provider returned {len(all_new_embeddings)} total embeddings, "
                    f"expected {len(uncached_texts)}"
                ),
            )

        # Update results and cache
        for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
            embedding = all_new_embeddings[i]
            embeddings[idx] = embedding

            cache_key = get_cache_key(
                text,
                provider,
                model_id or "default",
                dimensions,
                backend_identity=backend_identity,
            )
            await embedding_cache.set(cache_key, embedding)

    return embeddings


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/embeddings",
    response_model=CreateEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Create embeddings (enhanced with circuit breaker)",
    dependencies=[
        Depends(rbac_rate_limit("embeddings.create")),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
    responses={
        status.HTTP_402_PAYMENT_REQUIRED: {
            "description": "Billing limit exceeded. Upgrade plan to continue."
        },
    },
)
async def create_embedding_endpoint(
    request: Request,
    embedding_request: CreateEmbeddingRequest = Body(...),
    current_user: User = Depends(get_request_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_provider: str | None = Header(None, alias="x-provider"),
    response: Response = None
):
    """Create embeddings with circuit breaker protection and enhanced error recovery"""

    if not EMBEDDINGS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings service unavailable; dependencies not installed"
        )

    active_embedding_requests.inc()
    start_time = time.time()

    user_metadata = _build_user_metadata(current_user)
    rg_handle_id: str | None = None
    rg_commit_op_id: str | None = None
    rg_reserved_units: int = 0
    rg_actual_units: int = 0
    rg_governor = None

    try:
        # Backpressure and tenant quotas
        exc = await _check_backpressure_and_quotas(request, current_user)
        if exc is not None:
            raise exc
        # Validate provider (defer policy checks until after input validation)
        provider = x_provider or "openai"
        model = embedding_request.model
        if not model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="model is required",
            )

        # Auto-detect provider based on model name if not specified
        if ":" in model:
            parts = model.split(":", 1)
            provider = parts[0]
            model = parts[1]
        elif not x_provider:  # Only auto-detect if provider not explicitly set
            # Common HuggingFace model prefixes/patterns
            huggingface_patterns = [
                "sentence-transformers/",
                "BAAI/",
                "thenlper/",
                "intfloat/",
                "hkunlp/",
                "Qwen/",
                "microsoft/",
                "google/",
                "facebook/",
                "bert-",
                "roberta-",
                "xlm-",
                "distilbert-",
                "all-MiniLM-",
                "all-mpnet-",
            ]

            for pattern in huggingface_patterns:
                if model.startswith(pattern) or "/" in model:
                    openai_models = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
                    if model not in openai_models:
                        provider = "huggingface"
                        break

        provider = (provider or "").lower()

        try:
            EmbeddingProvider(provider)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown provider: {provider}"
            ) from None

        _validate_dimensions_request(provider, model, embedding_request.dimensions)

        # Parse and validate input FIRST (before policy checks)
        texts_to_embed: list[str] = []
        provided_token_arrays = False
        provided_token_count = 0
        token_lengths: list[int] | None = None

        if isinstance(embedding_request.input, str):
            if not embedding_request.input.strip():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input cannot be empty")
            texts_to_embed = [embedding_request.input]
        elif isinstance(embedding_request.input, list):
            if not embedding_request.input:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input list cannot be empty")

            # Support list[str], list[int], or list[list[int]]
            if all(isinstance(item, str) for item in embedding_request.input):
                if len(embedding_request.input) > 2048:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 2048 inputs allowed")
                if any(not (item or "").strip() for item in embedding_request.input):  # type: ignore[union-attr]
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Input list cannot contain empty strings",
                    )
                texts_to_embed = embedding_request.input  # type: ignore[assignment]
            elif all(isinstance(item, int) for item in embedding_request.input):
                # Single token array
                try:
                    texts_to_embed, provided_token_count, token_lengths = tokens_to_texts(embedding_request.input, model)
                    provided_token_arrays = True
                    embedding_token_inputs_total.labels(mode="single").inc()
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token array input") from e
            elif all(isinstance(item, list) for item in embedding_request.input):
                if len(embedding_request.input) > 2048:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 2048 inputs allowed")
                # Batch of token arrays
                try:
                    texts_to_embed, provided_token_count, token_lengths = tokens_to_texts(embedding_request.input, model)
                    provided_token_arrays = True
                    embedding_token_inputs_total.labels(mode="batch").inc()
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token array input") from e
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid input type")
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid input type")

        # Enforce per-model token length limits (fail-fast)
        max_tokens = _get_model_max_tokens(provider, model)
        too_long: list[tuple[int, int]] = []  # (index, token_count)
        token_total = 0
        if provided_token_arrays and token_lengths is not None:
            for idx, tok in enumerate(token_lengths):
                if tok > max_tokens:
                    too_long.append((idx, tok))
            token_total = int(provided_token_count or 0)
        else:
            for idx, t in enumerate(texts_to_embed):
                tok = count_tokens(t, model)
                token_total += int(tok or 0)
                if tok > max_tokens:
                    too_long.append((idx, tok))
        if too_long:
            # Return top-level JSON error object to match tests (not nested under "detail")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "input_too_long",
                    "message": f"One or more inputs exceed max tokens {max_tokens} for model {model}",
                    "details": [{"index": i, "tokens": tok} for (i, tok) in too_long]
                }
            )

        # Provider/model allowlist enforcement (after input validation)
        # Enforce allowlists based on config/env; admin may bypass unless STRICT is set
        enforce_policy = _should_enforce_policy_for_request(request, current_user)
        allowed_providers = _get_allowed_providers()
        if enforce_policy and allowed_providers is not None and provider.lower() not in allowed_providers:
            embedding_policy_denied_total.labels(provider=provider, model=model, policy_type="provider").inc()
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Provider '{provider}' is not allowed")

        allowed_models = _get_allowed_models()
        if enforce_policy and allowed_models is not None:
            def _model_allowed(m: str) -> bool:
                for pat in allowed_models:
                    if pat.endswith("*") and m.startswith(pat[:-1]):
                        return True
                    if m == pat:
                        return True
                return False
            if not _model_allowed(model):
                embedding_policy_denied_total.labels(provider=provider, model=model, policy_type="model").inc()
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Model '{model}' is not allowed")

        # Guard: return 501 for unsupported/unstyled providers (prevents silent fallback)
        try:
            EmbeddingProvider(provider)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            # Unknown provider is handled as 400 elsewhere, keep behavior consistent
            pass
        else:
            if provider.lower() not in IMPLEMENTED_PROVIDERS:
                raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=f"Provider '{provider}' not implemented")

        byok_cache: dict[str, ResolvedByokCredentials] = {}

        async def _resolve_provider_credentials(
            name: str,
            *,
            force_oauth_refresh: bool = False,
        ) -> ResolvedByokCredentials:
            key = (name or "").strip().lower()
            cached = byok_cache.get(key)
            if cached and not force_oauth_refresh:
                return cached
            resolved = await _resolve_embeddings_byok(
                key,
                current_user,
                request,
                force_oauth_refresh=force_oauth_refresh,
            )
            byok_cache[key] = resolved
            return resolved

        # ResourceGovernor tokens enforcement (per-minute + durable tokens/day caps).
        # Requests are enforced at ingress via RGSimpleMiddleware when enabled; token
        # accounting needs endpoint-level units.
        try:
            rg_governor = getattr(request.app.state, "rg_governor", None)
            rg_loader = getattr(request.app.state, "rg_policy_loader", None)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            rg_governor = None
            rg_loader = None

        if rg_governor is not None and rg_loader is not None:
            try:
                policy_id = str(getattr(request.state, "rg_policy_id", None) or "embeddings.default")
                rg_commit_op_id = str(
                    getattr(request.state, "request_id", None)
                    or request.headers.get("X-Request-ID")
                    or uuid.uuid4().hex
                )

                entity = derive_entity_key(request)
                try:
                    entity_scope, entity_value = entity.split(":", 1)
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    user_id = resolve_user_id_for_request(
                        current_user,
                        error_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )
                    entity_scope, entity_value = "user", str(user_id)

                daily_cap = 0
                try:
                    pol = rg_loader.get_policy(policy_id) or {}
                    daily_cap = int((pol.get("tokens") or {}).get("daily_cap") or 0)
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    daily_cap = 0
                if daily_cap > 0:
                    await backfill_legacy_tokens_to_ledger(
                        entity_scope=str(entity_scope),
                        entity_value=str(entity_value),
                    )

                rg_reserved_units = max(1, int(token_total or 0))
                rg_actual_units = int(rg_reserved_units)
                dec, hid = await rg_governor.reserve(
                    RGRequest(
                        entity=entity,
                        categories={"tokens": {"units": int(rg_reserved_units)}},
                        tags={"policy_id": policy_id, "endpoint": request.url.path},
                    ),
                    op_id=rg_commit_op_id,
                )
                if not bool(getattr(dec, "allowed", False)):
                    retry_after = int(getattr(dec, "retry_after", None) or 1)
                    headers = {"Retry-After": str(retry_after)}
                    try:
                        pol = rg_loader.get_policy(policy_id) or {}
                        per_min = int((pol.get("tokens") or {}).get("per_min") or 0)
                        limit_val = per_min or int((pol.get("tokens") or {}).get("daily_cap") or 0)
                        if limit_val:
                            headers.update(
                                {
                                    "X-RateLimit-Limit": str(limit_val),
                                    "X-RateLimit-Remaining": "0",
                                    "X-RateLimit-Reset": str(retry_after),
                                }
                            )
                            if per_min > 0:
                                headers.update(
                                    {
                                        "X-RateLimit-PerMinute-Limit": str(per_min),
                                        "X-RateLimit-PerMinute-Remaining": "0",
                                        "X-RateLimit-Tokens-Remaining": "0",
                                    }
                                )
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                        pass
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded",
                        headers=headers,
                    )
                rg_handle_id = hid
            except HTTPException:
                raise
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as rg_exc:
                logger.debug(f"RG tokens reserve skipped: {rg_exc}")
                rg_handle_id = None

        # Create embeddings
        # Special-case for OpenAI in test mode: synthesize vectors deterministically
        use_synthetic_openai = (
            provider == "openai"
            and env_flag_enabled("TESTING")
            and not env_flag_enabled("USE_REAL_OPENAI_IN_TESTS")
        )

        embeddings: list[list[float]] = []
        embeddings_from_adapter = False

        original_provider = provider
        original_model = model
        requested_provider = provider

        # Optional adapter-backed path (Stage 4 wiring): allow routing to
        # Embeddings adapters when explicitly enabled via env flag. Adapters take
        # precedence over synthetic OpenAI vectors when enabled to honor explicit
        # configuration in tests and production.
        try:
            adapters_enabled = is_truthy(os.getenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", ""))
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            adapters_enabled = False
        if adapters_enabled:
            try:
                # Currently wire OpenAI/HF/Google adapters via registry
                from tldw_Server_API.app.core.LLM_Calls.embeddings_adapter_registry import get_embeddings_registry
                registry = get_embeddings_registry()
                adapter = registry.get_adapter(provider)
                # Prepare adapter request (provider-specific key if available)
                byok_resolution = await _resolve_provider_credentials(provider)
                if provider in EMBEDDINGS_PROVIDERS_REQUIRE_KEY and not byok_resolution.api_key:
                    if not _should_skip_missing_key(provider, byok_resolution):
                        _raise_missing_embeddings_key(provider)
                _api_key: str | None = byok_resolution.api_key

                adapter_request: dict[str, Any] = {
                    "input": texts_to_embed if len(texts_to_embed) > 1 else texts_to_embed[0],
                    "model": model,
                    "api_key": _api_key,
                }
                if (
                    embedding_request.dimensions is not None
                    and provider.lower() == "openai"
                    and _supports_openai_dimensions(model)
                ):
                    adapter_request["dimensions"] = embedding_request.dimensions
                result = adapter.embed(adapter_request) if adapter else None
                if isinstance(result, dict) and isinstance(result.get("data"), list):
                    embs: list[list[float]] = []
                    for item in result["data"]:
                        vec = item.get("embedding") if isinstance(item, dict) else None
                        if isinstance(vec, list):
                            embs.append(vec)
                    if embs and len(embs) == len(texts_to_embed):
                        # Adapter-provided vectors may already be normalized. Preserve them as-is
                        # unless LLM_EMBEDDINGS_L2_NORMALIZE explicitly requests normalization.
                        processed: list[list[float]] = []
                        for v in embs:
                            arr, did_l2 = decide_and_apply_l2(
                                v,
                                embedding_request.encoding_format,
                                embeddings_from_adapter=True,
                            )
                            processed.append(arr.tolist() if did_l2 else v)
                        embeddings = processed
                        embeddings_from_adapter = True
                # If adapter failed to produce vectors, fall through to legacy/synthetic path
            except HTTPException as he:
                if (
                    he.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
                    and isinstance(getattr(he, "detail", None), dict)
                    and he.detail.get("error_code") == "missing_provider_credentials"
                ):
                    raise
                logger.debug(f"Embeddings adapter path failed; falling back to legacy: {he}")
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as _e:
                # Log and fall back silently; adapter path is optional
                logger.debug(f"Embeddings adapter path failed; falling back to legacy: {_e}")

        if use_synthetic_openai and not embeddings:
            dim = 1536
            mid = (model or "").lower()
            if "3-large" in mid:
                dim = 3072
            import numpy as _np
            embeddings = []
            for t in texts_to_embed:
                seed = int(hashlib.sha256((model + "|" + t).encode("utf-8")).hexdigest()[:16], 16)
                rng = _np.random.default_rng(seed)
                vec = rng.standard_normal(dim, dtype=_np.float32)
                nrm = _np.linalg.norm(vec)
                if nrm > 0:
                    vec = vec / nrm
                embeddings.append(vec.tolist())
        elif not embeddings:
            # Try provider with fallback chain on failure
            last_error: Exception | None = None
            # Fallback policy when explicit provider header is present:
            # Strict by default: do NOT fallback when `x-provider` header is set.
            # To allow fallback even with header, set EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER=true
            try:
                allow_hdr = is_truthy(os.getenv("EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER"))
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                allow_hdr = False
            fallback_disabled = (x_provider is not None and not allow_hdr)
            chain = [provider] if fallback_disabled else resolve_fallback_chain(provider)
            if enforce_policy and allowed_providers is not None:
                chain = [p for p in chain if p.lower() in allowed_providers or p == provider]
            fallback_from: str | None = None
            for p in chain:
                try:
                    if p != provider:
                        embedding_fallbacks_total.labels(from_provider=provider, to_provider=p).inc()
                        fallback_from = provider
                    # Map model id to destination provider if needed
                    target_model_id = map_model_for_provider(original_provider, p, original_model)
                    credentials = await _resolve_provider_credentials(p)
                    if p in EMBEDDINGS_PROVIDERS_REQUIRE_KEY and not credentials.api_key:
                        if _should_skip_missing_key(p, credentials):
                            pass
                        elif p == requested_provider:
                            _raise_missing_embeddings_key(p)
                        else:
                            continue
                    try:
                        embeddings = await create_embeddings_batch_async(
                            texts=texts_to_embed,
                            provider=p,
                            model_id=target_model_id,
                            dimensions=embedding_request.dimensions,
                            api_key=credentials.api_key,
                            metadata=user_metadata,
                        )
                    except HTTPException as auth_exc:
                        if not (
                            p == "openai"
                            and getattr(credentials, "auth_source", None) == "oauth"
                            and _is_http_401_error(auth_exc)
                        ):
                            raise
                        try:
                            refreshed_credentials = await _resolve_provider_credentials(
                                p,
                                force_oauth_refresh=True,
                            )
                        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as refresh_exc:
                            _record_oauth_401_retry(p, "refresh_failed")
                            raise auth_exc from refresh_exc
                        if not refreshed_credentials.api_key:
                            _record_oauth_401_retry(p, "refresh_missing_api_key")
                            raise auth_exc
                        try:
                            embeddings = await create_embeddings_batch_async(
                                texts=texts_to_embed,
                                provider=p,
                                model_id=target_model_id,
                                dimensions=embedding_request.dimensions,
                                api_key=refreshed_credentials.api_key,
                                metadata=user_metadata,
                            )
                        except HTTPException as retry_exc:
                            if _is_http_401_error(retry_exc):
                                _record_oauth_401_retry(p, "retry_auth_failed")
                                raise auth_exc from retry_exc
                            _record_oauth_401_retry(p, "retry_failed")
                            raise
                        _record_oauth_401_retry(p, "success")
                    provider = p
                    if target_model_id:
                        model = target_model_id
                    # Add response headers to indicate fallback
                    try:
                        if response is not None:
                            response.headers['X-Embeddings-Provider'] = provider
                            if fallback_from and fallback_from != provider:
                                response.headers['X-Embeddings-Fallback-From'] = fallback_from
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                        pass
                    break
                except HTTPException as he:
                    if (
                        he.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
                        and isinstance(getattr(he, "detail", None), dict)
                        and he.detail.get("error_code") == "missing_provider_credentials"
                    ):
                        raise
                    if he.status_code and 400 <= he.status_code < 500 and he.status_code != 429:
                        embedding_provider_failures.labels(provider=p, model=model, reason=f"http_{he.status_code}").inc()
                        last_error = he
                        break
                    embedding_provider_failures.labels(provider=p, model=model, reason=f"http_{he.status_code or 'unknown'}").inc()
                    last_error = he
                    continue
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                    embedding_provider_failures.labels(provider=p, model=model, reason="exception").inc()
                    last_error = e
                    continue
            if not embeddings:
                logger.error(f"Embedding creation failed across providers {chain}: {last_error}")
                if isinstance(last_error, HTTPException):
                    raise last_error
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Embedding providers unavailable",
                ) from last_error

        try:
            final_credentials = byok_cache.get(provider.lower())
            if final_credentials:
                await final_credentials.touch_last_used()
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"BYOK touch_last_used failed for {provider}: {exc}")

        # Optional dimension adjustment (post-process)
        dims_policy_used = None
        if embedding_request.dimensions is not None:
            # For base64 outputs, always reduce to requested dims for deterministic length
            if embedding_request.encoding_format == "base64":
                try:
                    import numpy as _np
                    target = int(embedding_request.dimensions)
                    adjusted: list[list[float]] = []
                    for v in embeddings:
                        try:
                            arr = _np.asarray(v, dtype=_np.float32)
                            if arr.shape[0] > target:
                                arr = arr[:target]
                            adjusted.append(arr.tolist())
                        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                            adjusted.append(v)
                    embeddings = adjusted
                    dims_policy_used = "reduce"
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    # Fallback to normal policy if anything goes wrong
                    dims_policy_used = _dimension_policy()
                    embeddings = adjust_dimensions(embeddings, embedding_request.dimensions, provider, model)
            else:
                dims_policy_used = _dimension_policy()
                embeddings = adjust_dimensions(embeddings, embedding_request.dimensions, provider, model)
            # Add response header for visibility
            try:
                if response is not None and dims_policy_used:
                    response.headers['X-Embeddings-Dimensions-Policy'] = dims_policy_used
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                pass

        # Format response
        output_data = []
        for i, embedding in enumerate(embeddings):
            # Adapter-provided vectors may already be normalized. Preserve them as-is
            # unless LLM_EMBEDDINGS_L2_NORMALIZE explicitly requests normalization.
            arr, did_l2 = decide_and_apply_l2(
                embedding,
                embedding_request.encoding_format,
                embeddings_from_adapter=embeddings_from_adapter,
            )
            if embedding_request.encoding_format == "base64":
                processed_value = base64.b64encode(arr.tobytes()).decode('utf-8')
            else:
                # Preserve exact adapter-supplied values when not L2-normalizing
                processed_value = embedding if not did_l2 else arr.tolist()

            output_data.append(
                EmbeddingData(
                    embedding=processed_value,
                    index=i
                )
            )

        # Calculate token usage (reused for usage logging and RG commit).
        num_tokens = int(token_total or 0)
        rg_actual_units = int(num_tokens)

        # Track metrics
        duration = time.time() - start_time
        embedding_request_duration.labels(
            provider=provider,
            model=model
        ).observe(duration)

        embedding_requests_total.labels(
            provider=provider,
            model=model,
            status="success"
        ).inc()

        logger.info(
            f"Created {len(output_data)} embeddings",
            extra={
                "user_id": current_user.id,
                "provider": provider,
                "model": model,
                "duration": duration,
                "fallback_from": original_provider if original_provider != provider else None,
                "dimensions_policy": dims_policy_used,
            }
        )

        # Persist a usage log entry (best-effort)
        try:
            user_id = getattr(current_user, 'id', None)
            api_key_id = None
            try:
                if request is not None and hasattr(request, 'state'):
                    api_key_id = getattr(request.state, 'api_key_id', None)
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                api_key_id = None
            await log_llm_usage(
                user_id=user_id,
                key_id=api_key_id,
                request=request,
                endpoint=f"{request.method}:{request.url.path}",
                operation="embeddings",
                provider=provider,
                model=model,
                status=200,
                latency_ms=int((duration) * 1000),
                prompt_tokens=int(num_tokens or 0),
                completion_tokens=0,
                total_tokens=int(num_tokens or 0),
                request_id=getattr(getattr(request, "state", None), "request_id", None) or request.headers.get('X-Request-ID'),
            )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass

        # Attach quota headers if set
        try:
            if hasattr(request, 'state') and response is not None:
                if getattr(request.state, 'rate_limit_limit', None) is not None:
                    response.headers["X-RateLimit-Limit"] = str(request.state.rate_limit_limit)
                    response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass

        return CreateEmbeddingResponse(
            data=output_data,
            model=f"{provider}:{model}" if provider != "openai" else model,
            usage=EmbeddingUsage(
                prompt_tokens=num_tokens,
                total_tokens=num_tokens
            )
        )

    finally:
        try:
            if rg_governor is not None and rg_handle_id:
                actual = int(rg_actual_units or rg_reserved_units or 0)
                actual = max(0, actual)
                await rg_governor.commit(
                    rg_handle_id,
                    actuals={"tokens": int(actual)},
                    op_id=rg_commit_op_id,
                )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as rg_commit_exc:
            logger.debug(f"RG tokens commit skipped/failed: {rg_commit_exc}")
        active_embedding_requests.dec()

class EmbeddingsBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Texts to embed")
    model: str | None = Field(None, description="Embedding model identifier")
    provider: str | None = Field(None, description="Embedding provider override")
    dimensions: int | None = Field(None, description="Requested output dimensions if supported")
    batch_size: int | None = Field(None, description="Hint for provider batch sizing")


class EmbeddingsBatchResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    provider: str
    count: int


@router.post(
    "/embeddings/batch",
    response_model=EmbeddingsBatchResponse,
    summary="Create embeddings for a batch of texts",
    dependencies=[
        Depends(rbac_rate_limit("embeddings.create")),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
    responses={
        status.HTTP_402_PAYMENT_REQUIRED: {
            "description": "Billing limit exceeded. Upgrade plan to continue."
        },
    },
)
async def create_embeddings_batch_endpoint(
    payload: EmbeddingsBatchRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
    response: Response = None
) -> EmbeddingsBatchResponse:
    texts = payload.texts or []
    if not texts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="texts must not be empty")

    for text in texts:
        if not isinstance(text, str):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="All texts must be strings")
        if not text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="texts cannot contain empty strings")

    model, provider = _resolve_model_and_provider(payload.model, payload.provider)

    _validate_dimensions_request(provider, model, payload.dimensions)

    enforce_policy = _should_enforce_policy_for_request(request, current_user)
    allowed_providers = _get_allowed_providers()
    if enforce_policy and allowed_providers is not None and provider.lower() not in allowed_providers:
        embedding_policy_denied_total.labels(provider=provider, model=model, policy_type="provider").inc()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Provider '{provider}' is not allowed")

    if enforce_policy and not is_model_allowed(provider, model):
        embedding_policy_denied_total.labels(provider=provider, model=model, policy_type="model").inc()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Model '{model}' is not allowed")

    max_tokens = _get_model_max_tokens(provider, model)
    too_long = []
    for idx, text in enumerate(texts):
        tok = count_tokens(text, model)
        if tok > max_tokens:
            too_long.append({"index": idx, "tokens": tok})

    if too_long:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "input_too_long",
                "message": f"One or more inputs exceed max tokens {max_tokens} for model {model}",
                "details": too_long
            }
        )

    # Backpressure and tenant quotas (best-effort; request may be None in some test paths)
    try:
        if request is not None:
            exc = await _check_backpressure_and_quotas(request, current_user)
            if exc is not None:
                raise exc
    except HTTPException:
        raise
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass

    user_metadata = _build_user_metadata(current_user)

    credentials = await _resolve_embeddings_byok(provider, current_user, request)
    if provider in EMBEDDINGS_PROVIDERS_REQUIRE_KEY and not credentials.api_key:
        if not _should_skip_missing_key(provider, credentials):
            _raise_missing_embeddings_key(provider)
    active_credentials = credentials
    try:
        embeddings = await create_embeddings_batch_async(
            texts=texts,
            provider=provider,
            model_id=model,
            dimensions=payload.dimensions,
            api_key=active_credentials.api_key,
            metadata=user_metadata,
        )
    except HTTPException as auth_exc:
        if not (
            provider == "openai"
            and getattr(active_credentials, "auth_source", None) == "oauth"
            and _is_http_401_error(auth_exc)
        ):
            raise
        try:
            active_credentials = await _resolve_embeddings_byok(
                provider,
                current_user,
                request,
                force_oauth_refresh=True,
            )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as refresh_exc:
            _record_oauth_401_retry(provider, "refresh_failed")
            raise auth_exc from refresh_exc
        if not active_credentials.api_key:
            _record_oauth_401_retry(provider, "refresh_missing_api_key")
            raise auth_exc
        try:
            embeddings = await create_embeddings_batch_async(
                texts=texts,
                provider=provider,
                model_id=model,
                dimensions=payload.dimensions,
                api_key=active_credentials.api_key,
                metadata=user_metadata,
            )
        except HTTPException as retry_exc:
            if _is_http_401_error(retry_exc):
                _record_oauth_401_retry(provider, "retry_auth_failed")
                raise auth_exc from retry_exc
            _record_oauth_401_retry(provider, "retry_failed")
            raise
        _record_oauth_401_retry(provider, "success")
    await active_credentials.touch_last_used()

    if payload.dimensions is not None:
        embeddings = adjust_dimensions(embeddings, payload.dimensions, provider, model)

    # Attach quota headers if present (parity with single-item endpoint)
    try:
        if hasattr(request, 'state') and response is not None:
            if getattr(request.state, 'rate_limit_limit', None) is not None:
                response.headers["X-RateLimit-Limit"] = str(request.state.rate_limit_limit)
                response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass

    return EmbeddingsBatchResponse(
        embeddings=embeddings,
        model=model,
        provider=provider,
        count=len(embeddings)
    )


# ============================================================================
# Model Management Endpoints
# ============================================================================

@router.get("/embeddings/models", summary="List available embedding models")
async def list_embedding_models():
    """List configured/known models with allowlist status."""
    cfg = settings.get("EMBEDDING_CONFIG", {}) or {}
    default_model = cfg.get("default_model_id") or cfg.get("embedding_model") or "text-embedding-3-small"
    default_provider = cfg.get("embedding_provider", "openai")

    # Collect known models from provider table + default
    known: list[dict[str, Any]] = []
    seen = set()
    # static provider models
    for prov, lst in PROVIDER_MODELS.items():
        for m in lst:
            key = (prov.value, m)
            if key in seen:
                continue
            seen.add(key)
            allowed = is_model_allowed(prov.value, m)
            known.append({"provider": prov.value, "model": m, "allowed": allowed, "default": False})
    # add default
    default_marked = False
    for item in known:
        if item.get("provider") == default_provider and item.get("model") == default_model:
            item["default"] = True
            default_marked = True
            break
    if not default_marked:
        known.append({
            "provider": default_provider,
            "model": default_model,
            "allowed": is_model_allowed(default_provider, default_model),
            "default": True
        })

    return {"data": known, "allowed_providers": _get_allowed_providers(), "allowed_models": _get_allowed_models()}


@router.get("/embeddings/models/{model_id:path}", summary="Get embedding model metadata")
async def get_embedding_model_info(
    model_id: str,
    request: Request,
    provider: str | None = Query(None, description="Provider override"),
    current_user: User = Depends(get_request_user),
):
    model = model_id
    resolved_provider = guess_provider_for_model(model, provider)

    if not is_model_allowed(resolved_provider, model):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not available")

    user_metadata = _build_user_metadata(current_user)

    try:
        credentials = await _resolve_embeddings_byok(resolved_provider, current_user, request)
        if resolved_provider in EMBEDDINGS_PROVIDERS_REQUIRE_KEY and not credentials.api_key:
            _raise_missing_embeddings_key(resolved_provider)
        active_credentials = credentials
        try:
            vectors = await create_embeddings_batch_async(
                texts=["model probe"],
                provider=resolved_provider,
                model_id=model,
                api_key=active_credentials.api_key,
                metadata=user_metadata,
            )
        except HTTPException as auth_exc:
            if not (
                resolved_provider == "openai"
                and getattr(active_credentials, "auth_source", None) == "oauth"
                and _is_http_401_error(auth_exc)
            ):
                raise
            try:
                active_credentials = await _resolve_embeddings_byok(
                    resolved_provider,
                    current_user,
                    request,
                    force_oauth_refresh=True,
                )
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as refresh_exc:
                _record_oauth_401_retry(resolved_provider, "refresh_failed")
                raise auth_exc from refresh_exc
            if not active_credentials.api_key:
                _record_oauth_401_retry(resolved_provider, "refresh_missing_api_key")
                raise auth_exc
            try:
                vectors = await create_embeddings_batch_async(
                    texts=["model probe"],
                    provider=resolved_provider,
                    model_id=model,
                    api_key=active_credentials.api_key,
                    metadata=user_metadata,
                )
            except HTTPException as retry_exc:
                if _is_http_401_error(retry_exc):
                    _record_oauth_401_retry(resolved_provider, "retry_auth_failed")
                    raise auth_exc from retry_exc
                _record_oauth_401_retry(resolved_provider, "retry_failed")
                raise
            _record_oauth_401_retry(resolved_provider, "success")
        await active_credentials.touch_last_used()
    except HTTPException:
        raise
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Model info probe failed for {resolved_provider}:{model}: {exc}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Embedding service unavailable") from exc

    dimension = None
    if vectors and vectors[0]:
        first = vectors[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            dimension = len(first)

    max_tokens = _get_model_max_tokens(resolved_provider, model)

    return {
        "model": model,
        "provider": resolved_provider,
        "dimension": dimension,
        "max_tokens": max_tokens,
        "allowed": True
    }


class TenantQuotaResponse(BaseModel):
    limit_rps: int
    remaining: int | None = None


def _is_single_user_profile() -> bool:
    """Check if profile indicates single-user deployment."""
    try:
        return is_single_user_profile_mode()
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Failed to resolve profile for runtime mode: {}", exc)
        return False


def _is_multi_user_runtime() -> bool:
    """Detect whether the current runtime should be treated as multi-user."""
    try:
        return not _is_single_user_profile()
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        return True


@router.get("/embeddings/tenant/quotas", summary="Get current tenant quotas (if multi-tenant)")
async def get_tenant_quotas(
    request: Request,
    current_user: User = Depends(get_request_user),
) -> TenantQuotaResponse:
    tenant_rps = _tenant_rps_runtime()
    if not _should_enforce_tenant_rps(request) or tenant_rps <= 0:
        return TenantQuotaResponse(limit_rps=0, remaining=None)
    client = None
    try:
        client = await _get_redis_client()
        key = f"embeddings:tenant:rps:{getattr(current_user, 'id', 'anon')}"
        val = await client.get(key)
        used = int(val or 0)
        return TenantQuotaResponse(limit_rps=tenant_rps, remaining=max(0, tenant_rps - used))
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        return TenantQuotaResponse(limit_rps=tenant_rps, remaining=None)
    finally:
        try:
            if client is not None:
                await ensure_async_client_closed(client)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass


class PriorityBumpRequest(BaseModel):
    job_id: str
    priority: str = Field(..., description="one of: high|normal|low")
    ttl_seconds: int | None = Field(default=600, ge=1, le=86400)


@router.post(
    "/embeddings/job/priority/bump",
    summary="Override/bump job priority for routing into priority queues (best-effort)",
    dependencies=[
        Depends(require_roles("admin")),
        Depends(require_permissions(EMBEDDINGS_ADMIN)),
    ],
)
async def bump_job_priority(
    req: PriorityBumpRequest,
    current_user: User = Depends(get_request_user),
) -> dict[str, Any]:
    pr = (req.priority or "").strip().lower()
    if pr not in ("high", "normal", "low"):
        raise HTTPException(status_code=400, detail="priority must be one of: high|normal|low")

    try:
        logger.info(
            "Embeddings admin priority bump requested",
            extra={
                "admin_id": getattr(current_user, "id", None),
                "job_id": req.job_id,
                "priority": pr,
            },
        )
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        # Logging must not interfere with admin operations
        pass
    client: aioredis.Redis | None = None
    try:
        client = await _get_redis_client()
        key = f"embeddings:priority:override:{req.job_id}"
        await client.set(key, pr)
        await client.expire(key, int(req.ttl_seconds or 600))
        return {
            "status": "ok",
            "job_id": req.job_id,
            "priority": pr,
            "ttl_seconds": int(req.ttl_seconds or 600),
        }
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail=f"Failed to set priority override: {e}") from e
    finally:
        try:
            if client is not None:
                await ensure_async_client_closed(client)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            # Connection cleanup must never interfere with HTTP error semantics
            pass


class ModelActionRequest(BaseModel):
    model: str
    provider: str | None = None


class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Collection name")
    metadata: dict[str, Any] | None = Field(default=None, description="Collection metadata")
    embedding_model: str | None = Field(default=None, description="Embedding model to associate")
    provider: str | None = Field(default=None, description="Provider override for dimension detection")


class CollectionResponse(BaseModel):
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionStatsResponse(BaseModel):
    name: str
    count: int
    embedding_dimension: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/embeddings/models/warmup",
    summary="Warmup (preload) an embedding model (admin)",
    dependencies=[Depends(require_roles("admin")), Depends(require_permissions(SYSTEM_CONFIGURE))],
)
async def warmup_model(
    payload: ModelActionRequest,
    current_user: User = Depends(get_request_user),
):
    provider = guess_provider_for_model(payload.model, payload.provider)
    if not is_model_allowed(provider, payload.model):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Model/provider not allowed")
    user_metadata = _build_user_metadata(current_user)

    try:
        await create_embeddings_batch_async(
            texts=["model warmup test"],
            provider=provider,
            model_id=payload.model,
            metadata=user_metadata,
        )
        return {"status": "ok", "provider": provider, "model": payload.model, "warmed": True}
    except HTTPException:
        raise
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Warmup failed for {provider}:{payload.model}: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Warmup failed: {e}") from e


@router.post(
    "/embeddings/models/download",
    summary="Download/prepare a model (admin)",
    dependencies=[Depends(require_roles("admin")), Depends(require_permissions(SYSTEM_CONFIGURE))],
)
async def download_model(
    payload: ModelActionRequest,
    current_user: User = Depends(get_request_user),
):
    provider = guess_provider_for_model(payload.model, payload.provider)
    if not is_model_allowed(provider, payload.model):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Model/provider not allowed")
    user_metadata = _build_user_metadata(current_user)

    try:
        # Trigger a load without depending on real content by generating a small embedding
        await create_embeddings_batch_async(
            texts=["download model"],
            provider=provider,
            model_id=payload.model,
            metadata=user_metadata,
        )
        return {"status": "ok", "provider": provider, "model": payload.model, "downloaded": True}
    except HTTPException:
        raise
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Download failed for {provider}:{payload.model}: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Download failed: {e}") from e

@router.delete(
    "/embeddings/cache",
    summary="Clear embedding cache (admin only)",
    dependencies=[Depends(require_roles("admin")), Depends(require_permissions(SYSTEM_CONFIGURE))],
)
async def clear_cache(
    current_user: User = Depends(get_request_user),
):
    """Clear the embedding cache - requires admin privileges"""

    cache_stats = embedding_cache.stats()
    await embedding_cache.clear()

    logger.info(
        "Cache cleared by admin",
        extra={
            "admin_id": current_user.id,
            "entries_cleared": cache_stats['size']
        }
    )

    return {
        "message": "Cache cleared successfully",
        "entries_removed": cache_stats['size']
    }


# ============================================================================
# Chroma Collection Management
# ============================================================================

@router.post(
    "/embeddings/collections",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a ChromaDB collection"
)
async def create_collection(
    payload: CollectionCreateRequest,
    current_user: User = Depends(get_request_user),
) -> CollectionResponse:
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Collection name is required")

    manager = _chroma_manager_for_user(current_user)

    user_metadata = _build_user_metadata(current_user)

    try:
        manager.client.get_collection(name=name)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass
    else:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Collection '{name}' already exists")

    metadata = payload.metadata.copy() if isinstance(payload.metadata, dict) else {}
    model, provider = _resolve_model_and_provider(payload.embedding_model, payload.provider)

    if payload.embedding_model:
        metadata.setdefault("embedding_model", model)
    metadata.setdefault("provider", provider)

    dimension = None
    try:
        vectors = await create_embeddings_batch_async(
            texts=["collection probe"],
            provider=provider,
            model_id=model,
            metadata=user_metadata,
        )
        if vectors and vectors[0]:
            first = vectors[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                dimension = len(first)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Collection dimension probe failed for {name}: {exc}")

    if dimension:
        metadata.setdefault("embedding_dimension", dimension)

    collection = manager.client.create_collection(name=name, metadata=metadata)
    coll_metadata = getattr(collection, "metadata", None) or metadata
    return CollectionResponse(name=collection.name, metadata=coll_metadata)


@router.get(
    "/embeddings/collections",
    response_model=list[CollectionResponse],
    summary="List ChromaDB collections"
)
async def list_collections(current_user: User = Depends(get_request_user)) -> list[CollectionResponse]:
    manager = _chroma_manager_for_user(current_user)
    collections = manager.client.list_collections()
    response: list[CollectionResponse] = []
    for collection in collections:
        metadata = getattr(collection, "metadata", {}) or {}
        response.append(CollectionResponse(name=collection.name, metadata=metadata))
    return response


@router.delete(
    "/embeddings/collections/{collection_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a ChromaDB collection"
)
async def delete_collection(
    collection_name: str,
    current_user: User = Depends(get_request_user),
) -> Response:
    manager = _chroma_manager_for_user(current_user)
    try:
        manager.client.delete_collection(name=collection_name)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Failed to delete collection {collection_name}: {exc}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/embeddings/collections/{collection_name}/stats",
    response_model=CollectionStatsResponse,
    summary="Retrieve collection statistics"
)
async def get_collection_stats(
    collection_name: str,
    current_user: User = Depends(get_request_user),
) -> CollectionStatsResponse:
    manager = _chroma_manager_for_user(current_user)
    try:
        collection = manager.client.get_collection(name=collection_name)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found") from None

    try:
        count = int(collection.count())
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        try:
            data = collection.get(limit=0, include=[])
            count = len(data.get("ids") or [])
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            count = 0

    metadata = getattr(collection, "metadata", {}) or {}
    dimension = metadata.get("embedding_dimension")

    if dimension is None:
        try:
            sample = collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings") or []
            candidate = None
            if embeddings:
                bucket = embeddings[0]
                if isinstance(bucket, list) and bucket:
                    candidate = bucket[0]
                elif isinstance(bucket, (np.ndarray, tuple)):
                    candidate = bucket
            if candidate is not None and hasattr(candidate, "__len__"):
                dimension = len(candidate)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass

    return CollectionStatsResponse(
        name=collection.name,
        count=count,
        embedding_dimension=dimension,
        metadata=metadata
    )

@router.get(
    "/embeddings/health",
    summary="Health check with circuit breaker status"
)
async def health_check():
    """Enhanced health check with circuit breaker status"""

    # Get circuit breaker status for all providers
    breaker_status = {}
    for provider in EmbeddingProvider:
        breaker_name = f"embeddings_{provider.value}"
        breaker = circuit_breaker_registry.get(breaker_name)
        if breaker:
            status_info = breaker.get_status()
            breaker_status[provider.value] = {
                "state": status_info["state"],
                "failure_count": status_info["failure_count"],
                "last_failure": status_info["last_failure_time"]
            }

    try:
        hyde_enabled = bool(settings.get("HYDE_ENABLED", False))
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        hyde_enabled = False
    try:
        hyde_questions = int(settings.get("HYDE_QUESTIONS_PER_CHUNK", 0) or 0)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        hyde_questions = 0
    hyde_info = {
        "enabled": hyde_enabled,
        "questions_per_chunk": hyde_questions,
    }
    hyde_provider = settings.get("HYDE_PROVIDER")
    hyde_model = settings.get("HYDE_MODEL")
    if hyde_provider:
        hyde_info["provider"] = hyde_provider
    if hyde_model:
        hyde_info["model"] = hyde_model
    hyde_weight = settings.get("HYDE_WEIGHT_QUESTION_MATCH")
    if hyde_weight is not None:
        with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
            hyde_info["weight"] = float(hyde_weight)
    hyde_k_fraction = settings.get("HYDE_K_FRACTION")
    if hyde_k_fraction is not None:
        with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
            hyde_info["k_fraction"] = float(hyde_k_fraction)

    health_status = {
        "status": "healthy" if EMBEDDINGS_AVAILABLE else "degraded",
        "service": "embeddings_v5_production_enhanced",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_stats": embedding_cache.stats(),
        "active_requests": active_embedding_requests._value.get(),
        "circuit_breakers": breaker_status,
        "hyde": hyde_info,
    }

    if not EMBEDDINGS_AVAILABLE:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )

    return health_status

@router.get(
    "/embeddings/circuit-breakers",
    summary="Get circuit breaker status (admin only)",
    dependencies=[Depends(require_roles("admin")), Depends(require_permissions(SYSTEM_CONFIGURE))],
)
async def get_circuit_breakers(
    _current_user: User = Depends(get_request_user),
):
    """Get detailed circuit breaker status - requires admin privileges"""

    return circuit_breaker_registry.get_all_status()

@router.post(
    "/embeddings/circuit-breakers/{provider}/reset",
    summary="Reset circuit breaker (admin only)",
    dependencies=[Depends(require_roles("admin")), Depends(require_permissions(SYSTEM_CONFIGURE))],
)
async def reset_circuit_breaker(
    provider: str,
    current_user: User = Depends(get_request_user),
):
    """Reset specific circuit breaker - requires admin privileges"""

    breaker_name = f"embeddings_{provider}"
    breaker = circuit_breaker_registry.get(breaker_name)

    if not breaker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Circuit breaker for provider '{provider}' not found"
        )

    breaker.reset()

    logger.info(
        "Circuit breaker reset by admin",
        extra={
            "admin_id": current_user.id,
            "provider": provider
        }
    )

    return {
        "message": f"Circuit breaker for '{provider}' reset successfully"
    }

@router.get(
    "/embeddings/metrics",
    summary="Get service metrics (admin only)",
    dependencies=[Depends(require_roles("admin")), Depends(require_permissions(SYSTEM_CONFIGURE))],
)
async def get_metrics(
    request: Request,
    current_user: User = Depends(get_request_user),
):
    """Get detailed service metrics - requires admin privileges"""

    # Helper to sum counters across all labels
    def _sum_counter(c):
        try:
            total = 0.0
            for metric in c.collect():
                for s in metric.samples:
                    # Only sum the main counter samples (exclude created/_total duplicates if any appear)
                    if s.name.endswith('_total') or s.name == metric.name:
                        total += float(s.value)
            return int(total)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            return None

    def _safe_gauge_value(g):
        try:
            return g._value.get()
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            return None

    def _details(metric):
        try:
            samples = []
            for m in metric.collect():
                for s in m.samples:
                    entry = {"name": s.name, "value": float(s.value)}
                    with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                        entry.update(s.labels)
                    samples.append(entry)
            return samples
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            return []

    payload = {
        "cache": embedding_cache.stats(),
        "active_requests": _safe_gauge_value(active_embedding_requests),
        "circuit_breakers": circuit_breaker_registry.get_all_status(),
        "counters": {
            "requests_total": _sum_counter(embedding_requests_total),
            "provider_failures_total": _sum_counter(embedding_provider_failures),
            "fallbacks_total": _sum_counter(embedding_fallbacks_total),
            "policy_denied_total": _sum_counter(embedding_policy_denied_total),
            "dimension_adjustments_total": _sum_counter(embedding_dimension_adjustments_total),
            "token_inputs_total": _sum_counter(embedding_token_inputs_total),
        },
        "details": {
            "requests": _details(embedding_requests_total),
            "provider_failures": _details(embedding_provider_failures),
            "fallbacks": _details(embedding_fallbacks_total),
            "policy_denied": _details(embedding_policy_denied_total),
            "dimension_adjustments": _details(embedding_dimension_adjustments_total),
            "token_inputs": _details(embedding_token_inputs_total),
        },
        "config": {
            "enforce_policy": _should_enforce_policy_for_request(request, current_user),
            "dimension_policy": _dimension_policy(),
            "cache": {
                "ttl_seconds": CACHE_TTL_SECONDS,
                "max_size": MAX_CACHE_SIZE,
                "cleanup_interval": CACHE_CLEANUP_INTERVAL,
            }
        }
    }
    return payload


# ============================================================================
# DLQ Admin Endpoints
# ============================================================================

class DLQItem(BaseModel):
    entry_id: str = Field(..., description="Redis stream entry ID")
    queue: str = Field(..., description="DLQ stream name")
    job_id: str | None = None
    error: str | None = None
    failed_at: str | None = None
    payload: dict[str, Any] | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    dlq_state: str | None = None
    operator_note: str | None = None


def _redact_obj(obj: Any, depth: int = 0) -> Any:
    """Redact likely PII/secrets from nested structures for previews."""
    if depth > 5:
        return obj
    SENSITIVE_KEYS = {"api_key", "authorization", "token", "password", "secret", "access_token", "id_token"}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key_low = str(k).lower().replace("-", "_")
            if key_low in SENSITIVE_KEYS:
                out[k] = "***REDACTED***"
            else:
                out[k] = _redact_obj(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [_redact_obj(x, depth + 1) for x in obj]
    if isinstance(obj, str) and len(obj) > 12 and any(x in obj.lower() for x in ("sk-", "api_key", "bearer ")):
        return "***REDACTED***"
    return obj


@router.get(
    "/embeddings/dlq",
    summary="List DLQ items for a stage (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def list_dlq_items(
    stage: str = Query("embedding", description="Stage: chunking|embedding|storage|content"),
    count: int = Query(50, ge=1, le=500, description="Max items to return"),
    job_id: str | None = Query(None, description="Optional job_id to filter"),
    _current_user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """List DLQ items for a stage.

    _current_user is included to enforce authentication via dependencies.
    """
    stream = _dlq_stream_name(stage)
    client: aioredis.Redis | None = None
    try:
        client = await _get_redis_client()
        # Reverse range: most recent first
        entries = await client.xrevrange(stream, "+", "-", count=count)
        items: list[DLQItem] = []
        for entry_id, fields in entries:
            # fields is a dict[str,str]
            payload = None
            try:
                raw_payload = fields.get("payload")
                if raw_payload:
                    payload = json.loads(raw_payload)
                elif fields.get("payload_enc"):
                    payload = decrypt_payload_if_present(fields.get("payload_enc"))
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                payload = None
            if payload is not None:
                payload = _redact_obj(payload)
            ji = fields.get("job_id")
            if job_id and ji != job_id:
                continue
            # sidecar state (quarantine/approval)
            dlq_state = None
            operator_note = None
            try:
                state_key = f"dlqstate:{stream}:{entry_id}"
                state_map = await client.hgetall(state_key)
                if isinstance(state_map, dict):
                    dlq_state = state_map.get("state") or fields.get("dlq_state")
                    operator_note = state_map.get("operator_note")
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                # Fallback to inline DLQ state fields if available
                dlq_state = fields.get("dlq_state")

            items.append(DLQItem(
                entry_id=entry_id,
                queue=stream,
                job_id=ji,
                error=fields.get("error"),
                failed_at=fields.get("failed_at"),
                payload=payload,
                fields=fields,
                dlq_state=dlq_state,
                operator_note=operator_note,
            ))
        return {"stream": stream, "count": len(items), "items": [i.model_dump() for i in items]}
    except HTTPException:
        raise
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail=f"Failed to list DLQ items: {e}") from e
    finally:
        try:
            if client is not None:
                await ensure_async_client_closed(client)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            # Connection cleanup must never interfere with HTTP error semantics
            pass


class DLQRequeueRequest(BaseModel):
    stage: str = Field(..., description="Stage: chunking|embedding|storage|content")
    entry_id: str = Field(..., description="Redis stream entry ID")
    delete_from_dlq: bool = Field(default=True)
    override_fields: dict[str, Any] | None = Field(default=None, description="Optional field overrides before requeue")


@router.post(
    "/embeddings/dlq/requeue",
    summary="Requeue a DLQ item to its live stream (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def requeue_dlq_item(
    req: DLQRequeueRequest,
    current_user: User = Depends(get_request_user),
) -> dict[str, Any]:
    dlq_stream = _dlq_stream_name(req.stage)
    live_stream = _live_stream_name(req.stage)
    client = await _get_redis_client()
    try:
        # Fetch the specific entry
        # XCLAIM not suitable; use XRANGE and filter by ID
        entries = await client.xrange(dlq_stream, min=req.entry_id, max=req.entry_id, count=1)
        if not entries:
            raise HTTPException(status_code=404, detail="DLQ entry not found")
        entry_id, fields = entries[0]
        # Quarantine enforcement: require approved_for_requeue if any state present
        try:
            st_map = await client.hgetall(f"dlqstate:{dlq_stream}:{entry_id}")
            effective_state = st_map.get("state") or fields.get("dlq_state")
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            effective_state = fields.get("dlq_state")
        if effective_state and effective_state not in ("approved_for_requeue",):
            dlq_requeued_total.labels(queue_name=dlq_stream, status="blocked").inc()
            raise HTTPException(status_code=400, detail=f"DLQ entry in state '{effective_state}', not approved for requeue")
        # Prepare requeue payload
        requeue_fields = dict(fields)
        warning = None
        # Validate original payload JSON (if present) and surface warnings
        try:
            raw = fields.get("payload")
            if raw:
                try:
                    original = json.loads(raw)
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    original = None
                if isinstance(original, dict):
                    try:
                        validate_schema(req.stage, original)
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as ve:
                        warning = f"payload schema validation failed: {ve}"
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass
        # Remove DLQ-specific fields
        for k in ["consumer_group", "worker_id", "failed_at", "error", "payload"]:
            requeue_fields.pop(k, None)
        if req.override_fields:
            requeue_fields.update(req.override_fields)
        # Requeue to live stream
        await client.xadd(live_stream, requeue_fields)
        # Optionally delete from DLQ
        if req.delete_from_dlq:
            with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                await client.xdel(dlq_stream, entry_id)
        dlq_requeued_total.labels(queue_name=dlq_stream, status="success").inc()
        out = {"message": "requeued", "from": dlq_stream, "to": live_stream, "entry_id": entry_id}
        if warning:
            out["warning"] = warning
        # Audit: DLQ requeue single
        try:
            svc = await get_audit_service_for_user(current_user)
            ctx = AuditContext(
                user_id=str(getattr(current_user, "id", "")),
                endpoint="/api/v1/embeddings/dlq/requeue",
                method="POST",
            )
            await svc.log_event(
                event_type=AuditEventType.DATA_UPDATE,
                category=AuditEventCategory.SECURITY,
                context=ctx,
                resource_type="dlq",
                resource_id=entry_id,
                action="requeue",
                metadata={"from": dlq_stream, "to": live_stream, "stage": req.stage, "warning": bool(warning)},
            )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass
        return out
    except HTTPException:
        dlq_requeued_total.labels(queue_name=dlq_stream, status="not_found").inc()
        raise
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        dlq_requeue_errors_total.labels(queue_name=dlq_stream, error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=f"Failed to requeue DLQ item: {e}") from e
    finally:
        await ensure_async_client_closed(client)


class DLQRequeueBulkRequest(BaseModel):
    stage: str
    entry_ids: list[str]
    delete_from_dlq: bool = True
    override_fields: dict[str, Any] | None = None


@router.post(
    "/embeddings/dlq/requeue/bulk",
    summary="Bulk requeue DLQ items to live stream (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def requeue_dlq_bulk(
    req: DLQRequeueBulkRequest,
    current_user: User = Depends(get_request_user),
) -> dict[str, Any]:
    dlq_stream = _dlq_stream_name(req.stage)
    live_stream = _live_stream_name(req.stage)
    client = await _get_redis_client()
    results: list[dict[str, Any]] = []
    try:
        for eid in req.entry_ids:
            status = "success"
            warning = None
            try:
                entries = await client.xrange(dlq_stream, min=eid, max=eid, count=1)
                if not entries:
                    status = "not_found"
                else:
                    eid_found, fields = entries[0]
                    try:
                        st_map = await client.hgetall(f"dlqstate:{dlq_stream}:{eid_found}")
                        effective_state = st_map.get("state") or fields.get("dlq_state")
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                        effective_state = fields.get("dlq_state")
                    if effective_state and effective_state not in ("approved_for_requeue",):
                        status = f"blocked:{effective_state}"
                        results.append({"entry_id": eid, "status": status})
                        continue
                    requeue_fields = dict(fields)
                    # Validate original payload JSON (if present) and surface warnings
                    try:
                        raw = fields.get("payload")
                        if raw:
                            try:
                                original = json.loads(raw)
                            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                                original = None
                            if isinstance(original, dict):
                                try:
                                    validate_schema(req.stage, original)
                                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as ve:
                                    warning = f"payload schema validation failed: {ve}"
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                        pass
                    for k in ["consumer_group", "worker_id", "failed_at", "error"]:
                        requeue_fields.pop(k, None)
                    requeue_fields.pop("payload", None)
                    if req.override_fields:
                        requeue_fields.update(req.override_fields)
                    await client.xadd(live_stream, requeue_fields)
                    if req.delete_from_dlq:
                        with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                            await client.xdel(dlq_stream, eid)
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                status = f"error:{type(e).__name__}"
                dlq_requeue_errors_total.labels(queue_name=dlq_stream, error_type=type(e).__name__).inc()
            else:
                dlq_requeued_total.labels(queue_name=dlq_stream, status=status).inc()
            res = {"entry_id": eid, "status": status}
            if warning:
                res["warning"] = warning
            results.append(res)
        # Audit: DLQ bulk requeue summary
        try:
            svc = await get_audit_service_for_user(current_user)
            ctx = AuditContext(
                user_id=str(getattr(current_user, "id", "")),
                endpoint="/api/v1/embeddings/dlq/requeue/bulk",
                method="POST",
            )
            counts = {"success": 0, "not_found": 0, "blocked": 0, "error": 0}
            for r in results:
                st = str(r.get("status", "success"))
                if st.startswith("blocked"):
                    counts["blocked"] += 1
                elif st in counts:
                    counts[st] += 1
                elif st.startswith("error"):
                    counts["error"] += 1
            await svc.log_event(
                event_type=AuditEventType.DATA_UPDATE,
                category=AuditEventCategory.SECURITY,
                context=ctx,
                resource_type="dlq",
                resource_id=dlq_stream,
                action="bulk_requeue",
                metadata={"stage": req.stage, **counts, "total": len(req.entry_ids)},
            )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass
        return {"from": dlq_stream, "to": live_stream, "results": results}
    finally:
        await ensure_async_client_closed(client)


@router.get(
    "/embeddings/dlq/stats",
    summary="DLQ and queue depths (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def get_dlq_stats(
    current_user: User = Depends(get_request_user),
):
    client = await _get_redis_client()
    try:
        queues = ["embeddings:chunking", "embeddings:embedding", "embeddings:storage", "embeddings:content"]
        depths = {}
        dlq_depths = {}
        for q in queues:
            try:
                depths[q] = await client.xlen(q)
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                depths[q] = 0
            dq = f"{q}:dlq"
            try:
                dlq_depths[dq] = await client.xlen(dq)
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                dlq_depths[dq] = 0
        total_dlq = sum(dlq_depths.values())

        # Aggregate worker metrics to summarize stage processed/failed
        stages = {"chunking": {"processed": 0, "failed": 0},
                  "embedding": {"processed": 0, "failed": 0},
                  "storage": {"processed": 0, "failed": 0}}
        try:
            cursor = 0
            processed = 0
            while True:
                cursor, keys = await client.scan(cursor, match="worker:metrics:*", count=100)
                for k in keys:
                    if processed >= ORCH_SCAN_MAX_KEYS:
                        cursor = 0
                        break
                    data = await client.get(k)
                    processed += 1
                    if not data:
                        continue
                    try:
                        m = json.loads(data)
                        stage = str(m.get("worker_type", "")).lower()
                        proc = int(m.get("jobs_processed", 0) or 0)
                        fail = int(m.get("jobs_failed", 0) or 0)
                        if stage in stages:
                            stages[stage]["processed"] += proc
                            stages[stage]["failed"] += fail
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                        continue
                if cursor == 0:
                    break
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass

        return {"queues": depths, "dlq": dlq_depths, "total_dlq": total_dlq, "stages": stages}
    finally:
        await ensure_async_client_closed(client)


# ---------------------------------------------------------------------------
# DLQ Quarantine State Management (admin only)
# ---------------------------------------------------------------------------

class DLQStateSetRequest(BaseModel):
    stage: str
    entry_id: str
    state: str  # quarantined | approved_for_requeue | ignored
    operator_note: str | None = None


def _dlq_state_key(stream: str, entry_id: str) -> str:
    return f"dlqstate:{stream}:{entry_id}"


@router.post(
    "/embeddings/dlq/state",
    summary="Set DLQ quarantine state (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def set_dlq_state(req: DLQStateSetRequest, current_user: User = Depends(get_request_user)):
    client = await _get_redis_client()
    try:
        dlq_stream = _dlq_stream_name(req.stage)
        # Validate entry exists
        entries = await client.xrange(dlq_stream, min=req.entry_id, max=req.entry_id, count=1)
        if not entries:
            raise HTTPException(status_code=404, detail="DLQ entry not found")
        st = (req.state or "").strip().lower()
        if st not in ("quarantined", "approved_for_requeue", "ignored"):
            raise HTTPException(status_code=400, detail="Invalid state")
        if st == "approved_for_requeue" and not (req.operator_note and req.operator_note.strip()):
            raise HTTPException(status_code=400, detail="operator_note is required to approve requeue")
        val = {
            "state": st,
            "operator_note": req.operator_note or "",
            "updated_by": getattr(current_user, "username", "admin"),
            "updated_at": datetime.utcnow().isoformat(),
        }
        await client.hset(_dlq_state_key(dlq_stream, req.entry_id), mapping=val)
        # Audit: DLQ quarantine state change
        try:
            svc = await get_audit_service_for_user(current_user)
            ctx = AuditContext(
                user_id=str(getattr(current_user, "id", "")),
                endpoint="/api/v1/embeddings/dlq/state",
                method="POST",
            )
            await svc.log_event(
                event_type=AuditEventType.DATA_UPDATE,
                category=AuditEventCategory.SECURITY,
                context=ctx,
                resource_type="dlq",
                resource_id=req.entry_id,
                action="quarantine_state",
                metadata={"stage": req.stage, "state": st, "operator_note": req.operator_note or ""},
            )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass
        return {"ok": True, "stream": dlq_stream, "entry_id": req.entry_id, "state": st}
    finally:
        await ensure_async_client_closed(client)


# ---------------------------------------------------------------------------
# Stage Controls: pause/resume/drain per stage (admin only)
# ---------------------------------------------------------------------------

class StageControlRequest(BaseModel):
    stage: str  # chunking|embedding|storage|content|all
    action: str  # pause|resume|drain


def _stage_key(stage: str, suffix: str) -> str:
    stage = stage.strip().lower()
    if stage not in {"chunking", "embedding", "storage", "content"}:
        raise HTTPException(status_code=400, detail="Invalid stage; must be chunking|embedding|storage|content")
    return f"embeddings:stage:{stage}:{suffix}"


@router.get(
    "/embeddings/stage/status",
    summary="Get per-stage pause/drain flags (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def get_stage_status(current_user: User = Depends(get_request_user)):
    client = await _get_redis_client()
    try:
        out = {}
        for st in ("chunking", "embedding", "storage", "content"):
            paused = await client.get(_stage_key(st, "paused"))
            drain = await client.get(_stage_key(st, "drain"))
            out[st] = {
                "paused": str(paused).lower() in ("1", "true", "yes"),
                "drain": str(drain).lower() in ("1", "true", "yes"),
            }
        return out
    finally:
        await ensure_async_client_closed(client)


@router.post(
    "/embeddings/stage/control",
    summary="Pause/Resume/Drain a stage (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def control_stage(req: StageControlRequest, current_user: User = Depends(get_request_user)):
    client = await _get_redis_client()
    try:
        stages = [req.stage] if req.stage != "all" else ["chunking", "embedding", "storage", "content"]
        for st in stages:
            if req.action == "pause":
                await client.set(_stage_key(st, "paused"), "1")
            elif req.action == "resume":
                await client.delete(_stage_key(st, "paused"))
                await client.delete(_stage_key(st, "drain"))
            elif req.action == "drain":
                # Mark drain intent and pause new reads; in-flight items will finish
                await client.set(_stage_key(st, "drain"), "1")
                await client.set(_stage_key(st, "paused"), "1")
            else:
                raise HTTPException(status_code=400, detail="Invalid action; must be pause|resume|drain")
        # Audit
        try:
            svc = await get_audit_service_for_user(current_user)
            ctx = AuditContext(
                user_id=str(getattr(current_user, "id", "")),
                endpoint="/api/v1/embeddings/stage/control",
                method="POST",
            )
            await svc.log_event(
                event_type=AuditEventType.CONFIG_CHANGED,
                category=AuditEventCategory.SYSTEM,
                context=ctx,
                resource_type="embeddings_stage",
                resource_id=",".join(stages),
                action=req.action,
                metadata={"stages": stages, "action": req.action},
            )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass
        return {"ok": True, "stages": stages, "action": req.action}
    finally:
        await ensure_async_client_closed(client)


# ---------------------------------------------------------------------------
# Job skip registry (admin only)
# ---------------------------------------------------------------------------

class JobSkipRequest(BaseModel):
    job_id: str
    ttl_seconds: int | None = Field(default=7 * 24 * 3600, ge=60, description="TTL for skip registry entry")


def _skip_key(job_id: str) -> str:
    return f"embeddings:skip:job:{job_id}"


@router.post(
    "/embeddings/job/skip",
    summary="Mark a job_id as skipped (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def mark_job_skipped(req: JobSkipRequest, current_user: User = Depends(get_request_user)):
    client = await _get_redis_client()
    try:
        await client.set(_skip_key(req.job_id), "1", ex=int(req.ttl_seconds))
        # Audit
        try:
            svc = await get_audit_service_for_user(current_user)
            ctx = AuditContext(
                user_id=str(getattr(current_user, "id", "")),
                endpoint="/api/v1/embeddings/job/skip",
                method="POST",
            )
            await svc.log_event(
                event_type=AuditEventType.DATA_UPDATE,
                category=AuditEventCategory.SECURITY,
                context=ctx,
                resource_type="job",
                resource_id=req.job_id,
                action="skip",
                metadata={"ttl_seconds": int(req.ttl_seconds or 0)},
            )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass
        return {"ok": True, "job_id": req.job_id, "ttl_seconds": req.ttl_seconds}
    finally:
        await ensure_async_client_closed(client)


@router.get(
    "/embeddings/job/skip/status",
    summary="Check if a job_id is marked as skipped (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def get_job_skip_status(job_id: str = Query(..., description="Job ID to check"), current_user: User = Depends(get_request_user)):
    client = await _get_redis_client()
    try:
        val = await client.get(_skip_key(job_id))
        return {"job_id": job_id, "skipped": str(val).lower() in ("1", "true", "yes")}
    finally:
        await ensure_async_client_closed(client)


# ---------------------------------------------------------------------------
# Ledger Admin Endpoints (idempotency/dedupe)
# ---------------------------------------------------------------------------

class LedgerEntry(BaseModel):
    key: str
    status: str | None = None
    ts: int | None = None
    job_id: str | None = None
    raw: dict[str, Any] | str | None = None
    ttl_seconds: int | None = None


@router.get(
    "/embeddings/ledger/status",
    summary="Inspect ledger entries by idempotency_key/dedupe_key (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def get_ledger_status(
    idempotency_key: str | None = Query(default=None),
    dedupe_key: str | None = Query(default=None),
    _current_user: User = Depends(get_request_user),
) -> dict[str, LedgerEntry | None]:
    """Return current ledger values for provided keys.

    _current_user is included to enforce authentication and RBAC via dependencies.

    Reads:
      - embeddings:ledger:idemp:{idempotency_key}
      - embeddings:ledger:dedupe:{dedupe_key}
    Values may be plain strings or JSON objects with {status, ts, job_id}.
    """
    if not idempotency_key and not dedupe_key:
        raise HTTPException(status_code=400, detail="Provide idempotency_key and/or dedupe_key")
    client = await _get_redis_client()
    try:
        out: dict[str, LedgerEntry | None] = {"idempotency": None, "dedupe": None}
        if idempotency_key:
            k = f"embeddings:ledger:idemp:{idempotency_key}"
            raw = await client.get(k)
            ttl = await client.ttl(k)
            entry = LedgerEntry(key=k, ttl_seconds=(int(ttl) if isinstance(ttl, (int, float)) else None))
            if raw is not None:
                try:
                    obj = json.loads(raw)
                    entry.status = str(obj.get("status")) if isinstance(obj, dict) else None
                    entry.ts = int(obj.get("ts")) if isinstance(obj, dict) and obj.get("ts") is not None else None
                    entry.job_id = str(obj.get("job_id")) if isinstance(obj, dict) else None
                    entry.raw = obj if isinstance(obj, dict) else raw
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    entry.raw = raw
                    entry.status = str(raw)
            out["idempotency"] = entry
        if dedupe_key:
            k = f"embeddings:ledger:dedupe:{dedupe_key}"
            raw = await client.get(k)
            ttl = await client.ttl(k)
            entry = LedgerEntry(key=k, ttl_seconds=(int(ttl) if isinstance(ttl, (int, float)) else None))
            if raw is not None:
                try:
                    obj = json.loads(raw)
                    entry.status = str(obj.get("status")) if isinstance(obj, dict) else None
                    entry.ts = int(obj.get("ts")) if isinstance(obj, dict) and obj.get("ts") is not None else None
                    entry.job_id = str(obj.get("job_id")) if isinstance(obj, dict) else None
                    entry.raw = obj if isinstance(obj, dict) else raw
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    entry.raw = raw
                    entry.status = str(raw)
            out["dedupe"] = entry
        return out
    finally:
        await ensure_async_client_closed(client)


# ---------------------------------------------------------------------------
# Re-embed Scheduling (admin only)
# ---------------------------------------------------------------------------

class ReembedScheduleRequest(BaseModel):
    media_id: int = Field(..., description="Target media_id to re-embed")
    user_id: str | None = Field(default=None, description="Owner user id; defaults to current admin")
    idempotency_key: str | None = Field(default=None, description="Optional idempotency key to dedupe creation")
    dedupe_key: str | None = Field(default=None, description="Optional dedupe key; defaults to idempotency_key if not provided")
    operation_id: str | None = Field(default=None, description="Optional operation id for replay prevention")
    priority: int | None = Field(default=50, ge=0, le=100)
    user_tier: str | None = Field(default="free")
    embedder_name: str | None = None
    embedder_version: str | None = None


class ReembedScheduleResponse(BaseModel):
    id: int
    uuid: str | None = None
    status: str
    domain: str
    queue: str
    job_type: str


@router.post(
    "/embeddings/reembed/schedule",
    response_model=ReembedScheduleResponse,
    summary="Schedule a re-embed expansion job (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def schedule_reembed(
    req: ReembedScheduleRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
):
    """Create a media embeddings Jobs row to re-embed content.

    Domain: embeddings, Queue: EMBEDDINGS_JOBS_QUEUE (default),
    Job Types: embeddings_pipeline (root) with Redis Streams chunking stage.
    """
    # Build payload
    uid = str(req.user_id or current_user.id)
    payload = {
        "user_id": uid,
        "media_id": int(req.media_id),
        "embedding_model": req.embedder_name,
        "embedding_provider": None,
        "request_source": "reembed",
        "operation_id": req.operation_id,
        "user_tier": req.user_tier or "free",
        "embedder_version": req.embedder_version,
        "current_stage": "chunking",
        "force_regenerate": False,
    }
    # Construct default idempotency/dedupe if not provided
    idempotency_key = req.idempotency_key or f"reembed:{uid}:{int(req.media_id)}:{req.embedder_name or ''}:{req.embedder_version or ''}"

    # Create job via JobManager
    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager  # local import to avoid hard dep at import-time
        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        jm = JobManager(backend=backend, db_url=db_url)
        queue = (os.getenv("EMBEDDINGS_JOBS_QUEUE") or "default").strip() or "default"
        root_queue = (os.getenv("EMBEDDINGS_ROOT_JOBS_QUEUE") or "").strip()
        if not root_queue:
            root_queue = "low" if queue != "low" else "default"
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        priority = max(1, min(10, int((req.priority or 50) / 10)))
        root_row = jm.create_job(
            domain="embeddings",
            queue=root_queue,
            job_type="embeddings_pipeline",
            payload=payload,
            owner_user_id=uid,
            priority=priority,
            idempotency_key=idempotency_key,
            request_id=rid,
        )
        from tldw_Server_API.app.core.Embeddings import redis_pipeline

        stage_payload = dict(payload)
        stage_payload["root_job_uuid"] = root_row.get("uuid")
        stage_payload["parent_job_uuid"] = root_row.get("uuid")
        stage_payload["user_id"] = uid
        if rid:
            stage_payload["request_id"] = rid
        if tp:
            stage_payload["trace_id"] = tp
        if idempotency_key:
            stage_payload["idempotency_key"] = f"{idempotency_key}:chunking"
        try:
            stream_id = redis_pipeline.enqueue_chunking_job(
                payload=stage_payload,
                root_job_uuid=str(root_row.get("uuid") or ""),
                force_regenerate=False,
                require_redis=not redis_pipeline.allow_stub(),
            )
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                jm.fail_job(
                    int(root_row["id"]),
                    error="Failed to enqueue re-embed job to Redis",
                    retryable=False,
                    enforce=False,
                )
            raise
        get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="reembed", traceparent=tp).info(
            "Scheduled re-embed job: root_id=%s stream_id=%s media_id=%s",
            root_row.get("id"),
            stream_id,
            payload.get("media_id"),
        )
        return ReembedScheduleResponse(
            id=int(root_row.get("id")),
            uuid=root_row.get("uuid"),
            status=str(root_row.get("status")),
            domain=str(root_row.get("domain")),
            queue=str(root_row.get("queue")),
            job_type=str(root_row.get("job_type")),
        )
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule re-embed: {e}") from e


# ---------------------------------------------------------------------------
# Orchestrator snapshot + SSE (admin only)
# ---------------------------------------------------------------------------

async def _build_orchestrator_snapshot(client: aioredis.Redis, now_ts: float | None = None) -> dict[str, Any]:
    """Compute a single orchestrator snapshot.

    Returns dict with keys: queues, dlq, ages, stages, flags, ts
    """
    from time import time as _now
    if now_ts is None:
        now_ts = _now()

    # Build the same structure as get_dlq_stats and add queue ages and stage flags
    queues = ["embeddings:chunking", "embeddings:embedding", "embeddings:storage", "embeddings:content"]
    depths: dict[str, int] = {}
    dlq_depths: dict[str, int] = {}
    ages: dict[str, float] = {}
    # Optional per-priority depths when priority routing is enabled
    priority_enabled = str(os.getenv("EMBEDDINGS_PRIORITY_ENABLED", "false")).lower() in ("1", "true", "yes")
    priority_depths: dict[str, dict[str, int]] = {"chunking": {}, "embedding": {}, "storage": {}, "content": {}}
    for q in queues:
        try:
            depths[q] = await client.xlen(q)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            depths[q] = 0
        # Expose per-priority sub-queue depths (high/normal/low)
        if priority_enabled:
            stage = q.split(":", 1)[1]
            for pr in ("high", "normal", "low"):
                sub = f"{q}:{pr}"
                try:
                    dsub = await client.xlen(sub)
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    dsub = 0
                depths[sub] = dsub
                priority_depths[stage][pr] = dsub
        # queue age (oldest entry)
        try:
            rng = await client.xrange(q, min='-', max='+', count=1)
            if rng:
                first_id, _ = rng[0]
                ts_ms = int(str(first_id).split('-')[0])
                ages[q] = max(0.0, (now_ts * 1000 - ts_ms) / 1000.0)
            else:
                ages[q] = 0.0
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            ages[q] = 0.0
        dq = f"{q}:dlq"
        try:
            dlq_depths[dq] = await client.xlen(dq)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            dlq_depths[dq] = 0
        with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
            embedding_queue_age_current_seconds.labels(queue_name=q).set(float(ages.get(q, 0.0)))

    # stage counters (aggregate from worker snapshots)
    stages: dict[str, dict[str, int]] = {
        "chunking": {"processed": 0, "failed": 0},
        "embedding": {"processed": 0, "failed": 0},
        "storage": {"processed": 0, "failed": 0},
        "content": {"processed": 0, "failed": 0},
    }
    try:
        cursor = 0
        processed = 0
        while True:
            cursor, keys = await client.scan(cursor, match="worker:metrics:*", count=100)
            for k in keys:
                if processed >= ORCH_SCAN_MAX_KEYS:
                    cursor = 0
                    break
                data = await client.get(k)
                processed += 1
                if not data:
                    continue
                try:
                    m = json.loads(data)
                    st = str(m.get("worker_type", "")).lower()
                    if st in stages:
                        stages[st]["processed"] += int(m.get("jobs_processed", 0) or 0)
                        stages[st]["failed"] += int(m.get("jobs_failed", 0) or 0)
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    continue
            if cursor == 0:
                break
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        pass

    # stage flags
    flags: dict[str, dict[str, bool]] = {}
    for st in ("chunking", "embedding", "storage", "content"):
        p = await client.get(f"embeddings:stage:{st}:paused")
        d = await client.get(f"embeddings:stage:{st}:drain")
        flags[st] = {
            "paused": str(p).lower() in ("1", "true", "yes"),
            "drain": str(d).lower() in ("1", "true", "yes"),
        }
        try:
            embedding_stage_flag.labels(stage=st, flag="paused").set(1.0 if flags[st]["paused"] else 0.0)
            embedding_stage_flag.labels(stage=st, flag="drain").set(1.0 if flags[st]["drain"] else 0.0)
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            pass

    return {"queues": depths, "dlq": dlq_depths, "ages": ages, "stages": stages, "flags": flags, "priority": priority_depths if priority_enabled else {}, "ts": now_ts}


async def _sse_orchestrator_stream(client: aioredis.Redis):
    import asyncio as _asyncio
    import random as _random
    while True:
        try:
            payload = await _build_orchestrator_snapshot(client)
            data = json.dumps(payload)
            # Emit event type for clients that use it
            yield f"event: summary\ndata: {data}\n\n"
            # Optional heartbeat comment
            yield ":\n\n"
            # Jittered interval around 5s
            await _asyncio.sleep(_random.uniform(4.5, 5.5))  # nosec B311 - non-cryptographic retry jitter
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
            # Keep the stream alive; emit a sanitized error and log details server-side
            try:
                logger.exception("Orchestrator stream error")
                yield f"event: error\ndata: {json.dumps({'error': 'Temporary service error'})}\n\n"
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                pass
            await _asyncio.sleep(_random.uniform(4.5, 5.5))  # nosec B311 - non-cryptographic retry jitter


@router.get(
    "/embeddings/orchestrator/events",
    summary="SSE: embeddings orchestrator live summary (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def orchestrator_events(_current_user: User = Depends(get_request_user)):
    # Admin/embeddings-admin gate is enforced via AuthNZ permissions; _current_user is used for audit context.
    try:
        logger.info(
            "Embeddings orchestrator SSE connection initiated",
            extra={"user_id": getattr(_current_user, "id", None)},
        )
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        # Audit logging is best-effort and must not break the SSE stream
        pass
    client = await _get_redis_client()

    # Legacy path (default): keep existing SSE generator behavior
    if os.getenv("STREAMS_UNIFIED", "0") != "1":
        async def _gen():
            try:
                orchestrator_sse_connections.inc()
                async for chunk in _sse_orchestrator_stream(client):
                    yield chunk
            finally:
                try:
                    orchestrator_sse_connections.dec()
                    orchestrator_sse_disconnects_total.inc()
                except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    pass
                await ensure_async_client_closed(client)
        return StreamingResponse(_gen(), media_type="text/event-stream")

    # Unified path (flagged): use SSEStream with standardized heartbeats and error handling
    async def _gen_unified():
        import random as _random
        try:
            orchestrator_sse_connections.inc()
            stream = SSEStream(
                # Honor env defaults for interval/mode; allow overriding via env
                heartbeat_interval_s=None,
                heartbeat_mode=None,
                close_on_error=False,  # do not close the stream on transient errors
                labels={"component": "embeddings", "endpoint": "orchestrator_events"},
            )

            async def _produce():
                while True:
                    try:
                        payload = await _build_orchestrator_snapshot(client)
                        await stream.send_event("summary", payload)
                    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                        # Emit a non-fatal sanitized error frame and continue; log details server-side
                        logger.exception("Provider error during orchestrator snapshot")
                        await stream.error("provider_error", "Temporary service error", data=None, close=False)
                    # Jittered ~5s cadence
                    await asyncio.sleep(_random.uniform(4.5, 5.5))  # nosec B311 - non-cryptographic retry jitter

            producer = asyncio.create_task(_produce())
            try:
                async for line in stream.iter_sse():
                    yield line
            except asyncio.CancelledError:
                # Client cancelled: cancel producer promptly and re-raise
                if not producer.done():
                    with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                        producer.cancel()
                    with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                        await asyncio.gather(producer, return_exceptions=True)
                raise
            else:
                # Normal shutdown: ensure producer completes without forced cancel
                if not producer.done():
                    with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                        await asyncio.gather(producer, return_exceptions=True)
        finally:
            try:
                orchestrator_sse_connections.dec()
                orchestrator_sse_disconnects_total.inc()
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                pass
            await ensure_async_client_closed(client)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_gen_unified(), media_type="text/event-stream", headers=headers)


@router.get(
    "/embeddings/orchestrator/summary",
    summary="Orchestrator summary for polling (admin only)",
    dependencies=[Depends(require_permissions(EMBEDDINGS_ADMIN))],
)
async def orchestrator_summary(current_user: User = Depends(get_request_user)):
    """Return a snapshot identical to the SSE payload.

    Includes: queues, dlq, ages, stages, flags, ts
    """
    client: aioredis.Redis | None = None
    def _zero_snapshot() -> dict[str, Any]:
        return {"queues": {}, "dlq": {}, "ages": {}, "stages": {}, "flags": {}, "ts": datetime.utcnow().timestamp()}

    try:
        client = await _get_redis_client()
        if getattr(client, "_tldw_is_stub", False):
            with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                orchestrator_summary_failures_total.inc()
            snapshot = _zero_snapshot()
            await ensure_async_client_closed(client)
            client = None
            return snapshot
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
            orchestrator_summary_failures_total.inc()
        return _zero_snapshot()
    try:
        return await _build_orchestrator_snapshot(client)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        with suppress(_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
            orchestrator_summary_failures_total.inc()
        return _zero_snapshot()
    finally:
        await ensure_async_client_closed(client)
