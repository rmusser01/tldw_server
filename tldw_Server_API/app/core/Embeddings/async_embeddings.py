# async_embeddings.py
# Async implementation of embeddings creation and management

import asyncio
import atexit
import contextlib
import hashlib
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.Embeddings.connection_pool import get_pool_manager
from tldw_Server_API.app.core.Embeddings.error_recovery import get_recovery_manager
from tldw_Server_API.app.core.Embeddings.metrics_integration import get_metrics
from tldw_Server_API.app.core.Embeddings.multi_tier_cache import get_multi_tier_cache
from tldw_Server_API.app.core.Embeddings.rate_limiter import get_async_rate_limiter
from tldw_Server_API.app.core.Embeddings.request_batching import get_batcher
from tldw_Server_API.app.core.Embeddings.simplified_config import get_config
from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.Utils.tokenizer import count_tokens as _count_tokens

_ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
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
_ASYNC_EMBEDDINGS_PROVIDER_EXCEPTIONS = _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS + (
    NetworkError,
    RetryExhaustedError,
)


class EmbeddingCredentialError(ValueError):
    """Sanitized error for a missing execution-scoped embedding credential."""

    def __init__(self, provider: str) -> None:
        normalized = "".join(
            char for char in str(provider or "").strip().lower() if char.isalnum() or char in ".-_"
        )[:64]
        self.provider = normalized or "unknown"
        super().__init__(f"Embedding credential is required for provider '{self.provider}'")


class EmbeddingEndpointError(ValueError):
    """Sanitized error for a missing execution-scoped embedding endpoint."""

    def __init__(self, provider: str) -> None:
        normalized = "".join(
            char for char in str(provider or "").strip().lower() if char.isalnum() or char in ".-_"
        )[:64]
        self.provider = normalized or "unknown"
        super().__init__(f"Embedding endpoint is required for provider '{self.provider}'")


class EmbeddingProviderError(RuntimeError):
    """Sanitized failure returned by an execution-scoped embedding provider."""

    _CODES = frozenset({"authentication", "provider_failure"})

    def __init__(self, provider: str, *, code: str, status_code: int | None = None) -> None:
        normalized = "".join(
            char for char in str(provider or "").strip().lower() if char.isalnum() or char in ".-_"
        )[:64]
        self.provider = normalized or "unknown"
        self.code = code if code in self._CODES else "provider_failure"
        self.status_code = status_code if isinstance(status_code, int) and 100 <= status_code <= 599 else None
        status_suffix = f", status={self.status_code}" if self.status_code is not None else ""
        super().__init__(f"Embedding provider error: {self.code} ({self.provider}{status_suffix})")


def _validate_runtime_embedding_vector(value: Any, provider: str) -> list[float]:
    """Return one finite numeric vector or raise a sanitized provider error."""
    try:
        array = np.asarray(value)
        if (
            array.ndim != 1
            or array.size == 0
            or not np.issubdtype(array.dtype, np.number)
        ):
            raise ValueError
        if not np.isfinite(array).all():
            raise ValueError
        return value if isinstance(value, list) else array.tolist()
    except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        raise EmbeddingProviderError(provider, code="provider_failure") from None


def _embedding_error_status(value: Any) -> int | None:
    """Extract only a bounded HTTP status from an exception or provider payload."""
    if isinstance(value, dict):
        for key in ("status_code", "status"):
            try:
                status = int(value.get(key))
            except (TypeError, ValueError):
                continue
            if 100 <= status <= 599:
                return status
        nested = value.get("error")
        if isinstance(nested, dict):
            return _embedding_error_status(nested)
        return None

    status = getattr(value, "status_code", None)
    response = getattr(value, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    try:
        parsed = int(status)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, int) and 100 <= parsed <= 599:
        return parsed
    message = str(value)
    for candidate in (401, 403, 429):
        if f"HTTP {candidate}" in message:
            return candidate
    return None


def _is_auth_failure(exc: Exception) -> bool:
    return _embedding_error_status(exc) in {401, 403}


def _normalize_embedding_response(payload: Any) -> list[float]:
    """Normalize embedding payloads into a single vector."""
    if isinstance(payload, dict):
        if "error" in payload:
            raise ValueError(str(payload.get("error")))
        if "embeddings" in payload:
            payload = payload["embeddings"]

    try:
        array = np.asarray(payload)
    except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
        raise ValueError(f"Unexpected embedding response type: {type(payload)}") from exc

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Unexpected embedding response dtype: {array.dtype}")

    array = array.astype(np.float32)
    if array.ndim == 1:
        return array.tolist()
    if array.ndim == 2:
        if array.shape[0] == 1:
            return array[0].tolist()
        return array.mean(axis=0).tolist()

    raise ValueError(f"Unexpected embedding response shape: {array.shape}")



class AsyncEmbeddingProvider:
    """Base class for async embedding providers"""

    def __init__(self, provider_name: str, api_key: Optional[str] = None):
        self.provider_name = provider_name
        self.api_key = api_key
        self.metrics = get_metrics()
        self.pool_manager = get_pool_manager()
        self.rate_limiter = get_async_rate_limiter()

    async def create_embedding(
        self,
        text: str,
        model: str,
        user_id: Optional[str] = None,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        credentials_resolved: bool = False,
    ) -> list[float]:
        """Create embedding asynchronously"""
        raise NotImplementedError

    async def create_embeddings_batch(
        self,
        texts: list[str],
        model: str,
        user_id: Optional[str] = None,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        credentials_resolved: bool = False,
    ) -> list[list[float]]:
        """Create embeddings for multiple texts"""
        tasks = [
            self.create_embedding(
                text,
                model,
                user_id,
                api_key_override=api_key_override,
                base_url_override=base_url_override,
                credentials_resolved=credentials_resolved,
            )
            for text in texts
        ]
        return await asyncio.gather(*tasks)


class AsyncOpenAIProvider(AsyncEmbeddingProvider):
    """Async OpenAI embeddings provider"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__("openai", api_key)
        self.default_base_url = "https://api.openai.com/v1/embeddings"
        self.base_url = base_url or self.default_base_url

    def _resolve_url(self, base_url_override: Optional[str], credentials_resolved: bool = False) -> str:
        url = base_url_override or (self.default_base_url if credentials_resolved is True else self.base_url)
        if url.endswith("/embeddings"):
            return url
        return f"{url.rstrip('/')}/embeddings"

    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
        user_id: Optional[str] = None,
        base_url_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        credentials_resolved: bool = False,
    ) -> list[float]:
        """Create embedding using OpenAI API"""
        import time as _time
        t0 = _time.perf_counter()
        status = "success"

        # Check rate limit
        if user_id:
            try:
                tokens_units = int(_count_tokens(text))
            except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                tokens_units = 0
            allowed, retry_after = await self.rate_limiter.check_rate_limit_async(
                user_id,
                cost=1,
                tokens_units=tokens_units,
            )
            if not allowed:
                status = "rate_limited"
                retry_after_msg = f" Retry after {retry_after}s." if retry_after else ""
                self.metrics.log_request(self.provider_name, model, status=status)
                self.metrics.log_error(self.provider_name, "RateLimitExceeded")
                logger.warning(
                    "Rate limit exceeded for user '{user_id}' on provider '{provider}' model '{model}'.{extra}",
                    user_id=user_id,
                    provider=self.provider_name,
                    model=model,
                    extra=retry_after_msg,
                )
                raise Exception(f"Rate limit exceeded.{retry_after_msg}")

        # Get connection pool for this provider
        pool = self.pool_manager.get_pool(self.provider_name)
        headers = {"Content-Type": "application/json"}
        api_key = api_key_override if credentials_resolved is True else self.api_key
        if credentials_resolved is True and not (isinstance(api_key, str) and api_key.strip()):
            raise EmbeddingCredentialError(self.provider_name)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "input": text,
            "model": model
        }

        try:
            data = await pool.request(
                method="POST",
                url=self._resolve_url(base_url_override, credentials_resolved),
                headers=headers,
                json_data=payload,
                sensitive_observability=(
                    credentials_resolved is True
                    and isinstance(base_url_override, str)
                    and bool(base_url_override.strip())
                ),
                bypass_circuit_breaker=credentials_resolved is True,
            )
            if isinstance(data, dict) and "error" in data:
                status = "failure"
                self.metrics.log_error(self.provider_name, "APIError")
                if credentials_resolved is True:
                    error_status = _embedding_error_status(data)
                    error_code = "authentication" if error_status in {401, 403} else "provider_failure"
                    raise EmbeddingProviderError(
                        self.provider_name,
                        code=error_code,
                        status_code=error_status,
                    ) from None
                raise ValueError(str(data.get("error")))
            if isinstance(data, dict) and data.get("data"):
                return data["data"][0]["embedding"]
            status = "failure"
            raise ValueError("Invalid OpenAI embeddings response format")
        except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
            status = "failure"
            self.metrics.log_error(self.provider_name, str(type(e).__name__))
            raise
        finally:
            # Emit metrics with the actual requested model
            elapsed = _time.perf_counter() - t0
            self.metrics.log_request(self.provider_name, model, status=status)
            self.metrics.log_request_latency(self.provider_name, model, elapsed)


class AsyncHuggingFaceProvider(AsyncEmbeddingProvider):
    """Async HuggingFace embeddings provider"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__("huggingface", api_key)
        self.default_base_url = "https://api-inference.huggingface.co/models"
        self.base_url = base_url or self.default_base_url
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def create_embedding(
        self,
        text: str,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        user_id: Optional[str] = None,
        base_url_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        credentials_resolved: bool = False,
    ) -> list[float]:
        """Create embedding using HuggingFace API"""

        # Check rate limit
        if user_id:
            try:
                tokens_units = int(_count_tokens(text))
            except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                tokens_units = 0
            allowed, retry_after = await self.rate_limiter.check_rate_limit_async(
                user_id,
                cost=1,
                tokens_units=tokens_units,
            )
            if not allowed:
                retry_after_msg = f" Retry after {retry_after}s." if retry_after else ""
                self.metrics.log_request(self.provider_name, model, status="rate_limited")
                self.metrics.log_error(self.provider_name, "RateLimitExceeded")
                logger.warning(
                    "Rate limit exceeded for user '{user_id}' on provider '{provider}' model '{model}'.{extra}",
                    user_id=user_id,
                    provider=self.provider_name,
                    model=model,
                    extra=retry_after_msg,
                )
                raise Exception(f"Rate limit exceeded.{retry_after_msg}")

        base_url = base_url_override or (self.default_base_url if credentials_resolved is True else self.base_url)
        url = f"{base_url.rstrip('/')}/{model}"

        # Get connection pool for this provider
        pool = self.pool_manager.get_pool(self.provider_name)
        headers = {"Content-Type": "application/json"}
        api_key = api_key_override if credentials_resolved is True else self.api_key
        if credentials_resolved is True and not (isinstance(api_key, str) and api_key.strip()):
            raise EmbeddingCredentialError(self.provider_name)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }

        try:
            data = await pool.request(
                method="POST",
                url=url,
                headers=headers,
                json_data=payload,
                sensitive_observability=(
                    credentials_resolved is True
                    and isinstance(base_url_override, str)
                    and bool(base_url_override.strip())
                ),
                bypass_circuit_breaker=credentials_resolved is True,
            )
            # Usage is already recorded in check_rate_limit_async
            if credentials_resolved is True and isinstance(data, dict) and "error" in data:
                error_status = _embedding_error_status(data)
                error_code = "authentication" if error_status in {401, 403} else "provider_failure"
                raise EmbeddingProviderError(
                    self.provider_name,
                    code=error_code,
                    status_code=error_status,
                ) from None
            return _normalize_embedding_response(data)
        except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
            self.metrics.log_error(self.provider_name, str(type(e).__name__))
            raise


class AsyncLocalAPIProvider(AsyncEmbeddingProvider):
    """Async local API embeddings provider (HTTP delegation)."""

    def __init__(self, api_url: str, api_key: Optional[str] = None):
        super().__init__("local_api", api_key)
        self.api_url = api_url

    async def create_embedding(
        self,
        text: str,
        model: str,
        user_id: Optional[str] = None,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        credentials_resolved: bool = False,
    ) -> list[float]:
        import time as _time
        t0 = _time.perf_counter()
        status = "success"

        if credentials_resolved is True and not (
            isinstance(base_url_override, str) and base_url_override.strip()
        ):
            raise EmbeddingEndpointError(self.provider_name)

        if user_id:
            try:
                tokens_units = int(_count_tokens(text))
            except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                tokens_units = 0
            allowed, retry_after = await self.rate_limiter.check_rate_limit_async(
                user_id,
                cost=1,
                tokens_units=tokens_units,
            )
            if not allowed:
                status = "rate_limited"
                retry_after_msg = f" Retry after {retry_after}s." if retry_after else ""
                self.metrics.log_request(self.provider_name, model, status=status)
                self.metrics.log_error(self.provider_name, "RateLimitExceeded")
                logger.warning(
                    "Rate limit exceeded for user '{user_id}' on provider '{provider}' model '{model}'.{extra}",
                    user_id=user_id,
                    provider=self.provider_name,
                    model=model,
                    extra=retry_after_msg,
                )
                raise Exception(f"Rate limit exceeded.{retry_after_msg}")

        pool = self.pool_manager.get_pool(self.provider_name)
        headers = {"Content-Type": "application/json"}
        api_key = api_key_override if credentials_resolved is True else self.api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {"texts": [text], "model": model}

        try:
            data = await pool.request(
                method="POST",
                url=base_url_override or self.api_url,
                headers=headers,
                json_data=payload,
                sensitive_observability=(
                    credentials_resolved is True
                    and isinstance(base_url_override, str)
                    and bool(base_url_override.strip())
                ),
                bypass_circuit_breaker=credentials_resolved is True,
            )
            return _normalize_embedding_response(data)
        except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
            status = "failure"
            self.metrics.log_error(self.provider_name, str(type(e).__name__))
            raise
        finally:
            elapsed = _time.perf_counter() - t0
            self.metrics.log_request(self.provider_name, model, status=status)
            self.metrics.log_request_latency(self.provider_name, model, elapsed)


class AsyncLocalProvider(AsyncEmbeddingProvider):
    """Async local embeddings provider using sentence-transformers"""

    def __init__(self, max_models_in_memory: int = 3, model_ttl_seconds: int = 3600):
        super().__init__("local", None)
        self.models: dict[str, Any] = {}
        self.model_last_used: dict[str, float] = {}
        self.model_in_use: dict[str, int] = {}
        self.max_models_in_memory = max(1, int(max_models_in_memory))
        self.model_ttl_seconds = max(0, int(model_ttl_seconds)) if model_ttl_seconds is not None else 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._model_locks: dict[str, asyncio.Lock] = {}
        self._model_locks_guard = asyncio.Lock()
        self._models_guard = asyncio.Lock()

    async def _get_model_lock(self, model_name: str) -> asyncio.Lock:
        """Return a per-model lock to prevent duplicate loads."""
        async with self._model_locks_guard:
            lock = self._model_locks.get(model_name)
            if lock is None:
                lock = asyncio.Lock()
                self._model_locks[model_name] = lock
            return lock

    async def _mark_in_use(self, model_name: str) -> None:
        async with self._models_guard:
            self.model_in_use[model_name] = self.model_in_use.get(model_name, 0) + 1

    async def _release_in_use(self, model_name: str) -> None:
        async with self._models_guard:
            current = self.model_in_use.get(model_name, 0)
            if current <= 1:
                self.model_in_use.pop(model_name, None)
            else:
                self.model_in_use[model_name] = current - 1

    async def _touch_model(self, model_name: str) -> None:
        async with self._models_guard:
            self.model_last_used[model_name] = time.time()

    async def _drop_model(self, model_name: str) -> None:
        async with self._models_guard:
            model = self.models.pop(model_name, None)
            self.model_last_used.pop(model_name, None)
            self.model_in_use.pop(model_name, None)
        if model is not None:
            try:
                if hasattr(model, "cpu"):
                    model.cpu()
            except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                pass

    async def _evict_if_needed(self, keep: Optional[str] = None) -> None:
        to_drop: list[str] = []
        async with self._models_guard:
            now = time.time()
            if self.model_ttl_seconds > 0:
                expired = [
                    name for name, last_used in self.model_last_used.items()
                    if name != keep and (now - last_used) > self.model_ttl_seconds
                ]
                for name in expired:
                    if self.model_in_use.get(name, 0) > 0:
                        continue
                    to_drop.append(name)

            # Reserve capacity only when `keep` names a model that is not loaded yet.
            reserve_slot = 1 if keep is not None and keep not in self.models else 0
            target_loaded = max(0, self.max_models_in_memory - reserve_slot)
            while len(self.models) - len(to_drop) > target_loaded:
                candidates = [
                    (name, last_used) for name, last_used in self.model_last_used.items()
                    if name != keep and self.model_in_use.get(name, 0) <= 0 and name not in to_drop
                ]
                if not candidates:
                    break
                lru_name = min(candidates, key=lambda item: item[1])[0]
                to_drop.append(lru_name)

        for name in to_drop:
            await self._drop_model(name)

    async def _load_model(self, model_name: str):
        """Load model if not already loaded"""
        if model_name not in self.models:
            lock = await self._get_model_lock(model_name)
            async with lock:
                if model_name in self.models:
                    return
                await self._evict_if_needed(keep=model_name)
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_running_loop()

                def load():
                    from sentence_transformers import SentenceTransformer
                    return SentenceTransformer(model_name)

                self.models[model_name] = await loop.run_in_executor(
                    self.executor,
                    load
                )
                await self._touch_model(model_name)
                logger.info(f"Loaded local model: {model_name}")

    async def create_embedding(
        self,
        text: str,
        model: str = "all-MiniLM-L6-v2",
        user_id: Optional[str] = None,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        credentials_resolved: bool = False,
    ) -> list[float]:
        """Create embedding using local model"""
        del api_key_override, base_url_override, credentials_resolved

        # Load model if needed
        await self._load_model(model)

        # Run encoding in thread pool
        loop = asyncio.get_running_loop()
        await self._mark_in_use(model)
        try:
            await self._touch_model(model)
            model_instance = self.models[model]
            embedding = await loop.run_in_executor(
                self.executor,
                lambda: model_instance.encode(text, convert_to_tensor=False)
            )
        finally:
            await self._release_in_use(model)

        return embedding.tolist()


class AsyncEmbeddingService:
    """
    Main async service for creating embeddings.
    Orchestrates providers, caching, batching, and fallbacks.
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize async embedding service"""
        self.config = config or get_config()
        self.pool_manager = get_pool_manager()
        self.cache = get_multi_tier_cache()
        self.batcher = get_batcher()
        self.recovery_manager = get_recovery_manager()
        self.metrics = get_metrics()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        # Initialize providers
        self.providers = {}
        self._initialize_providers()

        logger.info("Async embedding service initialized")

    def _initialize_providers(self):
        """Initialize configured providers"""
        for provider_config in self.config.providers:
            if not provider_config.enabled:
                continue

            pool_key = provider_config.name
            if provider_config.name == "openai":
                self.providers["openai"] = AsyncOpenAIProvider(
                    api_key=provider_config.api_key,
                    base_url=provider_config.api_url,
                )
            elif provider_config.name == "huggingface":
                self.providers["huggingface"] = AsyncHuggingFaceProvider(
                    api_key=provider_config.api_key,
                    base_url=provider_config.api_url,
                )
            elif provider_config.name == "local":
                if provider_config.api_url:
                    self.providers["local_api"] = AsyncLocalAPIProvider(
                        api_url=provider_config.api_url,
                        api_key=provider_config.api_key,
                    )
                    pool_key = "local_api"
                else:
                    self.providers["local"] = AsyncLocalProvider(
                        max_models_in_memory=self.config.resources.max_models_in_memory,
                        model_ttl_seconds=self.config.resources.model_ttl_seconds,
                    )
            elif provider_config.name == "local_api" and provider_config.api_url:
                self.providers["local_api"] = AsyncLocalAPIProvider(
                    api_url=provider_config.api_url,
                    api_key=provider_config.api_key,
                )
                pool_key = "local_api"

            try:
                self.pool_manager.get_pool(
                    pool_key,
                    max_connections=provider_config.max_connections,
                    timeout_seconds=provider_config.timeout_seconds,
                )
            except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Failed to initialize connection pool for provider '{pool_key}': {exc}")

            logger.info(f"Initialized provider: {provider_config.name}")

    async def create_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
        use_batching: bool = True,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        credentials_resolved: bool = False,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
    ) -> list[float]:
        """
        Create embedding with full async pipeline.

        Args:
            text: Input text
            model: Model to use (optional)
            provider: Provider to use (optional)
            user_id: User identifier for rate limiting
            use_cache: Whether to use caching
            use_batching: Whether to use batching
            on_provider_success: Zero-argument callback awaited only after direct
                primary-provider success; skipped on cache hit, failure, or fallback.

        Returns:
            Embedding vector
        """
        # Use defaults if not specified
        explicit_provider = provider is not None
        provider = provider or self.config.default_provider
        model = model or self.config.default_model
        provider = self._resolve_provider_alias(provider)
        actual_provider = provider
        actual_model = model
        provider_config = self._get_provider_config(provider)
        explicit_credentials = credentials_resolved is True
        cache_enabled = use_cache and not explicit_credentials
        effective_base_url = base_url_override if explicit_credentials else None
        if (
            not explicit_credentials
            and provider_config
            and provider in {"openai", "huggingface", "local_api"}
            and provider_config.api_url
        ):
            effective_base_url = provider_config.api_url
            # Ensure explicit provider overrides reach the async providers directly.
            if explicit_provider and provider in {"openai", "huggingface"}:
                use_batching = False
            if provider == "local_api":
                use_batching = False
        if explicit_credentials:
            use_batching = False
            if provider in {"openai", "huggingface"} and not (
                isinstance(api_key_override, str) and api_key_override.strip()
            ):
                raise EmbeddingCredentialError(provider)
            if provider == "local_api" and not (
                isinstance(effective_base_url, str) and effective_base_url.strip()
            ):
                raise EmbeddingEndpointError(provider)
        if on_provider_success is not None:
            use_batching = False

        # Create deterministic cache key across processes
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = f"{provider}:{model}:{text_hash}"
        if effective_base_url:
            cache_key = f"{cache_key}:{hashlib.sha256(effective_base_url.encode('utf-8')).hexdigest()}"

        # Check cache
        if cache_enabled:
            cached_embedding = await self.cache.get_async(cache_key)
            if cached_embedding is not None:
                self.metrics.log_cache_hit(model)
                return cached_embedding

        # Use batching if enabled
        primary_provider_dispatched = False
        if use_batching and self.batcher.enabled:
            embedding = await self.batcher.submit_request(
                text=text,
                model=model,
                provider=provider,
                metadata={"user_id": user_id}
            )
        else:
            # Direct provider call
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not available")

            provider_instance = self.providers[provider]

            try:
                call_kwargs = {"text": text, "model": model, "user_id": user_id}
                if explicit_credentials and provider in {"openai", "huggingface", "local_api"}:
                    call_kwargs.update(
                        api_key_override=api_key_override,
                        base_url_override=effective_base_url,
                        credentials_resolved=explicit_credentials,
                    )
                elif effective_base_url and provider in {"openai", "huggingface"}:
                    call_kwargs["base_url_override"] = effective_base_url
                embedding = await provider_instance.create_embedding(**call_kwargs)
                primary_provider_dispatched = True
            except _ASYNC_EMBEDDINGS_PROVIDER_EXCEPTIONS as e:
                if isinstance(e, EmbeddingProviderError):
                    raise
                if explicit_credentials:
                    error_status = _embedding_error_status(e)
                    error_code = "authentication" if error_status in {401, 403} else "provider_failure"
                    raise EmbeddingProviderError(
                        provider,
                        code=error_code,
                        status_code=error_status,
                    ) from None
                if _is_auth_failure(e):
                    raise
                # Try fallback provider
                embedding, actual_provider, actual_model = await self._try_fallback_providers(
                    text, model, provider, user_id, e
                )

        if primary_provider_dispatched and on_provider_success is not None:
            embedding = _validate_runtime_embedding_vector(embedding, provider)
            await on_provider_success()

        # Cache the result
        if cache_enabled:
            cache_key = f"{actual_provider}:{actual_model}:{text_hash}"
            actual_base_url = effective_base_url if explicit_credentials else None
            actual_provider_config = self._get_provider_config(actual_provider)
            if (
                not explicit_credentials
                and actual_provider_config
                and actual_provider in {"openai", "huggingface", "local_api"}
            ):
                actual_base_url = actual_provider_config.api_url
            if actual_base_url:
                cache_key = f"{cache_key}:{hashlib.sha256(actual_base_url.encode('utf-8')).hexdigest()}"
            await self.cache.set_async(cache_key, embedding)

        return embedding

    def _resolve_provider_alias(self, provider: str) -> str:
        """Resolve local/local_api alias based on configured providers."""
        if provider in self.providers:
            return provider
        if provider == "local" and "local_api" in self.providers:
            return "local_api"
        if provider == "local_api" and "local" in self.providers:
            return "local"
        return provider

    def _get_provider_config(self, provider: str) -> Any:
        """Return configuration for an effective provider alias."""
        provider_config = self.config.get_provider(provider)
        if provider_config is None and provider == "local_api":
            provider_config = self.config.get_provider("local")
        return provider_config

    async def create_embeddings_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        user_id: Optional[str] = None,
        parallel: bool = True,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        credentials_resolved: bool = False,
        on_provider_success: Callable[[], Awaitable[None]] | None = None,
    ) -> list[list[float]]:
        """
        Create embeddings for multiple texts.

        Args:
            texts: List of input texts
            model: Model to use
            provider: Provider to use
            user_id: User identifier
            parallel: Whether to process in parallel
            on_provider_success: Zero-argument callback awaited once per direct
                primary-provider success; skipped on cache hit, failure, or fallback.

        Returns:
            List of embedding vectors
        """
        if parallel:
            # Process in parallel
            tasks = [
                self.create_embedding(
                    text,
                    model,
                    provider,
                    user_id,
                    api_key_override=api_key_override,
                    base_url_override=base_url_override,
                    credentials_resolved=credentials_resolved,
                    on_provider_success=on_provider_success,
                )
                for text in texts
            ]
            return await asyncio.gather(*tasks)
        else:
            # Process sequentially
            embeddings = []
            for text in texts:
                embedding = await self.create_embedding(
                    text,
                    model,
                    provider,
                    user_id,
                    api_key_override=api_key_override,
                    base_url_override=base_url_override,
                    credentials_resolved=credentials_resolved,
                    on_provider_success=on_provider_success,
                )
                embeddings.append(embedding)
            return embeddings

    async def _try_fallback_providers(
        self,
        text: str,
        model: str,
        failed_provider: str,
        user_id: Optional[str],
        original_error: Exception
    ) -> tuple[list[float], str, str]:
        """Try fallback providers when primary fails"""

        # Get provider config
        provider_config = self.config.get_provider(failed_provider)

        if provider_config and provider_config.fallback_provider:
            fallback = provider_config.fallback_provider

            if fallback in self.providers:
                logger.warning(
                    f"Provider {failed_provider} failed, trying fallback {fallback}"
                )

                try:
                    provider_instance = self.providers[fallback]
                    fallback_model = model
                    fallback_config = self.config.get_provider(fallback)
                    if fallback_config and fallback_config.fallback_model:
                        fallback_model = fallback_config.fallback_model
                    else:
                        available_models = (fallback_config.models or []) if fallback_config else []
                        if available_models and model not in available_models:
                            fallback_model = available_models[0]
                            logger.debug(
                                "Substituting fallback model '{fallback_model}' for provider '{fallback}' "
                                "(original model '{model}' not available).",
                                fallback_model=fallback_model,
                                fallback=fallback,
                                model=model,
                            )
                    call_kwargs = {"text": text, "user_id": user_id}
                    if fallback_model is not None:
                        call_kwargs["model"] = fallback_model
                    if (
                        fallback_config
                        and fallback in {"openai", "huggingface"}
                        and fallback_config.api_url
                    ):
                        call_kwargs["base_url_override"] = fallback_config.api_url
                    embedding = await provider_instance.create_embedding(**call_kwargs)
                    used_model = fallback_model or model
                    return embedding, fallback, used_model
                except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Fallback provider {fallback} also failed: {e}")

        # No fallback available or fallback failed
        raise original_error

    async def warmup_providers(self):
        """Warmup all configured providers"""
        warmup_text = "Provider warmup test"

        for provider_name, provider_instance in self.providers.items():
            try:
                start_time = time.time()

                # Try to create a test embedding
                await provider_instance.create_embedding(
                    text=warmup_text,
                    model=self.config.default_model
                )

                elapsed = time.time() - start_time
                logger.info(f"Provider {provider_name} warmed up in {elapsed:.2f}s")

            except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to warmup provider {provider_name}: {e}")

    async def get_provider_status(self) -> dict[str, Any]:
        """Get status of all providers"""
        status = {}

        for provider_name, provider_instance in self.providers.items():
            try:
                # Try a test embedding
                test_start = time.time()
                await provider_instance.create_embedding(
                    text="test",
                    model=self.config.default_model
                )
                latency = time.time() - test_start

                status[provider_name] = {
                    "status": "healthy",
                    "latency_ms": int(latency * 1000)
                }
            except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
                status[provider_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

        return status

    async def shutdown(self):
        """Gracefully shutdown the service"""
        logger.info("Shutting down async embedding service...")

        # Flush batcher queues
        if self.batcher:
            await self.batcher.shutdown()

        # Close connection pools
        pool_manager = get_pool_manager()
        await pool_manager.close_all()

        # Shutdown thread pools in local providers
        for provider in self.providers.values():
            if hasattr(provider, 'executor'):
                try:
                    provider.executor.shutdown(wait=False)
                except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    with contextlib.suppress(_ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                        provider.executor.shutdown(wait=False)

        logger.info("Async embedding service shutdown complete")


# Global async service instances scoped by event loop
_async_services: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncEmbeddingService]" = (
    weakref.WeakKeyDictionary()
)
_async_service_fallback: Optional[AsyncEmbeddingService] = None
_health_check_tasks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Task]" = (
    weakref.WeakKeyDictionary()
)
_shutdown_registered = False
_service_lock = threading.Lock()


async def _cancel_health_check_task(loop: Optional[asyncio.AbstractEventLoop] = None):
    """Cancel and await the periodic health task(s) if they exist."""
    tasks: list[tuple[asyncio.AbstractEventLoop, asyncio.Task]] = []
    if loop is not None:
        task = _health_check_tasks.get(loop)
        if task is not None:
            tasks.append((loop, task))
    else:
        tasks = list(_health_check_tasks.items())

    for task_loop, task in tasks:
        try:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None

            if current_loop and task_loop is current_loop:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            else:
                # Different loop or no running loop; request cancellation and return
                try:
                    if task_loop.is_running():
                        task_loop.call_soon_threadsafe(task.cancel)
                    else:
                        task.cancel()
                except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
                    with contextlib.suppress(_ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                        task.cancel()
        finally:
            with contextlib.suppress(_ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                _health_check_tasks.pop(task_loop, None)


async def _shutdown_service(service: AsyncEmbeddingService):
    """Shutdown helper that cancels background tasks then stops the service."""
    loop = getattr(service, "_loop", None)
    await _cancel_health_check_task(loop=loop)
    await service.shutdown()


def get_async_embedding_service() -> AsyncEmbeddingService:
    """Get or create the global async embedding service."""
    global _async_service_fallback, _shutdown_registered
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    with _service_lock:
        if loop is not None:
            service = _async_services.get(loop)
            if service is None:
                service = AsyncEmbeddingService()
                service._loop = loop
                _async_services[loop] = service
        else:
            if _async_service_fallback is None:
                _async_service_fallback = AsyncEmbeddingService()
            service = _async_service_fallback

        if not _shutdown_registered:
            try:
                atexit.register(_shutdown_async_embedding_service_sync)
            except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Failed to register async embedding service shutdown hook: {exc}")
            else:
                _shutdown_registered = True

    return service


def _shutdown_async_embedding_service_sync():
    """Best-effort shutdown for all async embedding services at interpreter exit."""
    global _async_service_fallback

    with _service_lock:
        services = list(_async_services.items())
        fallback = _async_service_fallback
        _async_services.clear()
        _async_service_fallback = None

    def _shutdown_with_loop(service: AsyncEmbeddingService, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        try:
            if loop and not loop.is_closed():
                if loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(_shutdown_service(service), loop)
                    try:
                        fut.result(timeout=15)
                    except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
                        with contextlib.suppress(_ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                            logger.warning(f"Async embedding service shutdown timed out: {exc}")
                else:
                    loop.run_until_complete(_shutdown_service(service))
            else:
                asyncio.run(_shutdown_service(service))
        except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
            with contextlib.suppress(_ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS):
                logger.warning(f"Failed during async embedding service shutdown: {exc}")

    for loop, service in services:
        _shutdown_with_loop(service, loop)

    if fallback is not None:
        _shutdown_with_loop(fallback, getattr(fallback, "_loop", None))


# Convenience functions
async def create_embedding_async(
    text: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    user_id: Optional[str] = None
) -> list[float]:
    """
    Create a single embedding asynchronously.

    Args:
        text: Input text
        model: Model to use
        provider: Provider to use
        user_id: User identifier

    Returns:
        Embedding vector
    """
    service = get_async_embedding_service()
    return await service.create_embedding(text, model, provider, user_id)


async def create_embeddings_batch_async(
    texts: list[str],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    user_id: Optional[str] = None
) -> list[list[float]]:
    """
    Create embeddings for multiple texts asynchronously.

    Args:
        texts: List of input texts
        model: Model to use
        provider: Provider to use
        user_id: User identifier

    Returns:
        List of embedding vectors
    """
    service = get_async_embedding_service()
    return await service.create_embeddings_batch(texts, model, provider, user_id)


# FastAPI integration example
async def startup_event():
    """FastAPI startup event handler"""
    service = get_async_embedding_service()

    # Warmup providers
    await service.warmup_providers()

    # Start periodic tasks
    loop = asyncio.get_running_loop()
    _health_check_tasks[loop] = loop.create_task(periodic_health_check())


async def shutdown_event():
    """FastAPI shutdown event handler"""
    service = get_async_embedding_service()
    await _shutdown_service(service)


async def periodic_health_check():
    """Periodic health check for providers"""
    service = get_async_embedding_service()

    while True:
        try:
            await asyncio.sleep(60)  # Check every minute

            status = await service.get_provider_status()

            # Log unhealthy providers
            for provider, info in status.items():
                if info["status"] == "unhealthy":
                    logger.warning(f"Provider {provider} is unhealthy: {info.get('error')}")

        except asyncio.CancelledError:
            # Propagate cancellation so shutdown can stop this task promptly.
            raise
        except _ASYNC_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error in periodic health check: {e}")
