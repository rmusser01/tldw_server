# request_batching.py
# Request batching for improved throughput

import asyncio
import contextlib
import hashlib
import json
import threading
import time
import uuid
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from tldw_Server_API.app.core.Embeddings.metrics_integration import get_metrics
from tldw_Server_API.app.core.Embeddings.rate_limiter import get_async_rate_limiter
from tldw_Server_API.app.core.Embeddings.simplified_config import ProviderConfig, get_config
from tldw_Server_API.app.core.Utils.tokenizer import count_tokens as _count_tokens

_BATCHER_TOKEN_PARSE_EXCEPTIONS = (RuntimeError, TypeError, ValueError)
_BATCHER_NONCRITICAL_EXCEPTIONS = (
    asyncio.InvalidStateError,
    AttributeError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_BATCHER_FINGERPRINT_EXCEPTIONS = (RuntimeError, TypeError, ValueError)
_BATCHER_SHUTDOWN_EXCEPTIONS = (RuntimeError, TypeError, ValueError)


class EmbeddingsRateLimitError(RuntimeError):
    """Raised when the embeddings rate limiter rejects a request."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


def _get_spec_value(spec: Any, key: str) -> Optional[Any]:
    """Fetch a value from a model spec dict or object."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec.get(key)
    return getattr(spec, key, None)


def _resolve_model_spec(
    models: Any,
    model_id: str,
    model: str,
) -> tuple[Optional[str], Optional[Any]]:
    """Resolve a model spec from a mapping, returning the key and spec."""
    if not isinstance(models, dict):
        return None, None
    if model_id in models:
        return model_id, models[model_id]
    if model in models:
        return model, models[model]
    suffix_matches = [key for key in models if key.endswith(f":{model}")]
    if len(suffix_matches) == 1:
        key = suffix_matches[0]
        return key, models[key]
    return None, None


@dataclass
class BatchRequest:
    """Represents a single request in a batch"""
    request_id: str
    text: str
    model: str
    provider: str
    metadata: dict[str, Any]
    future: asyncio.Future
    timestamp: float
    config_override: Optional[dict[str, Any]] = None


class RequestBatcher:
    """
    Batches embedding requests for improved throughput.
    Collects requests and processes them in batches.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize request batcher.

        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.metrics = get_metrics()
        self.rate_limiter = get_async_rate_limiter()

        # Batching settings
        self.enabled = self.config.batching.enabled
        self.max_batch_size = self.config.batching.max_batch_size
        self.batch_timeout_ms = self.config.batching.batch_timeout_ms
        self.adaptive_batching = self.config.batching.adaptive_batching

        # Request queues per model
        self.queues: dict[tuple[str, str, str], deque] = {}
        self.processing_tasks: dict[tuple[str, str, str], asyncio.Task] = {}

        # Statistics
        self.stats = {
            'total_batches': 0,
            'total_requests': 0,
            'average_batch_size': 0,
            'average_wait_time': 0
        }

        # Adaptive batching parameters
        self.adaptive_params = {
            'current_batch_size': self.max_batch_size,
            'current_timeout': self.batch_timeout_ms,
            'throughput_history': deque(maxlen=10)
        }
        self._shutdown_requested = False

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        logger.info(
            f"Request batcher initialized: "
            f"enabled={self.enabled}, "
            f"max_batch_size={self.max_batch_size}, "
            f"timeout={self.batch_timeout_ms}ms"
        )

    def _start_processing_task(self, queue_key: tuple[str, str, str]) -> Optional[asyncio.Task]:
        """Start a processing task for the queue if we're not shutting down."""
        if self._shutdown_requested:
            return None

        new_task = asyncio.create_task(self._process_queue(queue_key))
        self.processing_tasks[queue_key] = new_task
        new_task.add_done_callback(
            lambda completed, key=queue_key: self._handle_task_done(key, completed)
        )
        return new_task

    def _handle_task_done(self, queue_key: tuple[str, str, str], completed: asyncio.Task) -> None:
        """Cleanup task tracking and restart processing if work arrived during idle exit."""
        if self.processing_tasks.get(queue_key) is completed:
            self.processing_tasks.pop(queue_key, None)

        if self._shutdown_requested:
            return

        queue = self.queues.get(queue_key)
        if queue:
            self._start_processing_task(queue_key)

    async def submit_request(
        self,
        text: str,
        model: str,
        provider: str,
        metadata: Optional[dict[str, Any]] = None,
        config_override: Optional[dict[str, Any]] = None
    ) -> Any:
        """
        Submit a request for batching.

        Args:
            text: Input text
            model: Model name
            provider: Provider name
            metadata: Optional metadata

        Returns:
            Embedding result when ready
        """
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError(
                "RequestBatcher used across multiple event loops; "
                "call get_batcher() inside each loop."
            )

        metadata = metadata or {}
        user_id = metadata.get("user_id")

        rate_limit_applied = False
        if (
            user_id
            and getattr(self.config.security, "enable_rate_limiting", False)
        ):
            try:
                tokens_units = int(_count_tokens(text))
            except _BATCHER_TOKEN_PARSE_EXCEPTIONS:
                tokens_units = 0
            allowed, retry_after = await self.rate_limiter.check_rate_limit_async(
                user_id,
                cost=1,
                tokens_units=tokens_units,
            )
            if not allowed:
                tier = "free"
                limiter = getattr(self.rate_limiter, "rate_limiter", None)
                if limiter and getattr(limiter, "user_tiers", None):
                    tier = limiter.user_tiers.get(user_id, "free")
                with contextlib.suppress(_BATCHER_NONCRITICAL_EXCEPTIONS):
                    self.metrics.log_rate_limit_hit(user_id, tier)
                retry_after_msg = f" Retry after {retry_after}s." if retry_after else ""
                raise EmbeddingsRateLimitError(
                    f"Rate limit exceeded for user '{user_id}'.{retry_after_msg}",
                    retry_after=retry_after,
                )
            rate_limit_applied = True

        if not self.enabled:
            # Batching disabled, process immediately
            return await self._process_single(
                text,
                model,
                self._resolve_provider_name(provider, model, config_override),
                metadata,
                config_override=config_override,
                rate_limit_applied=rate_limit_applied,
            )

        provider = self._resolve_provider_name(provider, model, config_override)

        # Create request
        request = BatchRequest(
            request_id=str(uuid.uuid4()),
            text=text,
            model=model,
            provider=provider,
            metadata=metadata,
            config_override=config_override,
            future=loop.create_future(),
            timestamp=time.time()
        )

        # Get or create queue for this model/config
        queue_key = self._queue_key(provider, model, config_override)
        if queue_key not in self.queues:
            self.queues[queue_key] = deque()

        task = self.processing_tasks.get(queue_key)
        if task is None or task.done():
            if task is not None and task.done():
                try:
                    reason = "cancelled" if task.cancelled() else f"finished (error={task.exception()})"
                except _BATCHER_NONCRITICAL_EXCEPTIONS:
                    reason = "finished"
                logger.debug(f"Restarting processing task for queue {self._queue_label(queue_key)}: previous task {reason}.")
            self._start_processing_task(queue_key)

        # Add to queue
        self.queues[queue_key].append(request)

        # Wait for result
        return await request.future

    async def _process_queue(self, queue_key: str):
        """
        Process requests from a queue.

        Args:
            queue_key: Queue identifier (provider:model)
        """
        queue = self.queues[queue_key]
        provider, model, _ = queue_key
        idle_timeout_s = max(5.0, self.batch_timeout_ms / 1000.0 * 10.0)
        idle_start: Optional[float] = None

        while True:
            try:
                # Collect batch
                batch = await self._collect_batch(queue)

                if not batch:
                    # No requests, wait a bit
                    if idle_start is None:
                        idle_start = time.time()
                    elif time.time() - idle_start >= idle_timeout_s:
                        logger.debug(
                            f"Processing task for queue {self._queue_label(queue_key)} idle for {idle_timeout_s:.2f}s; exiting."
                        )
                        break
                    await asyncio.sleep(0.01)
                    continue

                # Process batch
                idle_start = None
                await self._process_batch(batch, provider, model)

            except asyncio.CancelledError:
                logger.debug(f"Processing task for queue {queue_key} cancelled; shutting down.")
                raise
            except _BATCHER_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error processing batch queue {queue_key}: {e}")
                # Don't crash the processing task
                await asyncio.sleep(0.1)

    async def _collect_batch(self, queue: deque) -> list[BatchRequest]:
        """
        Collect requests into a batch.

        Args:
            queue: Request queue

        Returns:
            List of requests to process
        """
        batch = []
        start_time = time.time()

        # Determine batch parameters
        if self.adaptive_batching:
            max_size = self.adaptive_params['current_batch_size']
            timeout_ms = self.adaptive_params['current_timeout']
        else:
            max_size = self.max_batch_size
            timeout_ms = self.batch_timeout_ms

        timeout_seconds = timeout_ms / 1000.0

        # Collect requests until batch is full or timeout
        while len(batch) < max_size:
            if queue:
                batch.append(queue.popleft())
            else:
                # Queue empty, wait for more or timeout
                elapsed = time.time() - start_time

                if batch and elapsed >= timeout_seconds:
                    # Timeout reached with partial batch
                    break
                elif not batch:
                    # No requests yet, wait longer
                    await asyncio.sleep(0.001)

                    # Check timeout even for first request
                    if time.time() - start_time > timeout_seconds * 10:
                        break
                else:
                    # Have some requests, wait a bit more
                    remaining = timeout_seconds - elapsed
                    if remaining > 0:
                        await asyncio.sleep(min(0.001, remaining))
                    else:
                        break

        return batch

    async def _process_batch(
        self,
        batch: list[BatchRequest],
        provider: str,
        model: str
    ):
        """
        Process a batch of requests.

        Args:
            batch: List of requests
            provider: Provider name
            model: Model name
        """
        if not batch:
            return

        batch_start = time.time()

        try:
            # Extract texts
            texts = [req.text for req in batch]
            override_config = next(
                (req.config_override for req in batch if req.config_override is not None),
                None,
            )

            # Log batch metrics
            self.metrics.log_batch_size(provider, len(texts))

            # Process batch
            # Import here to avoid circular dependency
            if provider == "local":
                embeddings = await self._create_local_embeddings(texts, model)
            else:
                from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
                    create_embeddings_batch_async as embeddings_create_embeddings_batch_async,
                )

                # Create config for batch
                batch_config = override_config or self._build_user_app_config(provider, model)

                # Get embeddings
                embeddings = await embeddings_create_embeddings_batch_async(
                    texts,
                    batch_config,
                    model_id_override=f"{provider}:{model}"
                )

            # Distribute results
            for i, req in enumerate(batch):
                if req.future.done():
                    continue
                if i < len(embeddings):
                    req.future.set_result(embeddings[i])
                else:
                    req.future.set_exception(
                        ValueError(f"No embedding returned for request {i}")
                    )

            # Update statistics
            batch_time = time.time() - batch_start
            wait_times = [batch_start - req.timestamp for req in batch]
            avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0

            self.stats['total_batches'] += 1
            self.stats['total_requests'] += len(batch)
            self.stats['average_batch_size'] = (
                self.stats['total_requests'] / self.stats['total_batches']
            )
            self.stats['average_wait_time'] = (
                self.stats['average_wait_time'] * 0.9 + avg_wait * 0.1
            )

            # Update adaptive parameters
            if self.adaptive_batching:
                self._update_adaptive_params(len(batch), batch_time)

            logger.debug(
                f"Processed batch: size={len(batch)}, "
                f"time={batch_time:.3f}s, "
                f"avg_wait={avg_wait:.3f}s"
            )

        except _BATCHER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error processing batch: {e}")

            # Set error for all requests
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def _process_single(
        self,
        text: str,
        model: str,
        provider: str,
        metadata: Optional[dict[str, Any]] = None,
        config_override: Optional[dict[str, Any]] = None,
        rate_limit_applied: bool = False,
    ) -> Any:
        """
        Process a single request without batching.

        Args:
            text: Input text
            model: Model name
            provider: Provider name
            metadata: Optional metadata

        Returns:
            Embedding result
        """
        metadata = metadata or {}
        provider = self._resolve_provider_name(provider, model, config_override)

        # If a config override is present we need to honour it by delegating
        # to the synchronous embedding loader, but we still do the work in an
        # executor so the event loop remains responsive.
        if config_override:
            if provider == "local":
                embeddings = await self._create_local_embeddings([text], model)
                return embeddings[0]
            loop = asyncio.get_running_loop()

            def _invoke_sync():
                from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import create_embedding

                return create_embedding(
                    text,
                    config_override,
                    model_id_override=f"{provider}:{model}",
                )

            return await loop.run_in_executor(None, _invoke_sync)

        # Try the async embedding service first so we avoid blocking.
        from tldw_Server_API.app.core.Embeddings.async_embeddings import get_async_embedding_service
        service = get_async_embedding_service()
        user_id = metadata.get("user_id")
        effective_user_id = None if rate_limit_applied else user_id

        provider_candidates = []
        normalized_provider = self._normalize_provider_name(provider)
        for candidate in (provider, normalized_provider):
            if candidate and candidate not in provider_candidates:
                provider_candidates.append(candidate)
        if normalized_provider != "local_api":
            alias = self._alias_provider_name(provider)
            if alias and alias not in provider_candidates:
                provider_candidates.append(alias)

        last_provider_error: Optional[ValueError] = None
        for candidate in provider_candidates:
            try:
                return await service.create_embedding(
                    text=text,
                    model=model,
                    provider=candidate,
                    user_id=effective_user_id,
                    use_batching=False,
                )
            except ValueError as exc:
                last_provider_error = exc
                continue

        # Fall back to executor-based call if the async service cannot satisfy
        # the request (for example when the provider/model pair is not registered
        # in the async embedding config). This mirrors the legacy behaviour
        # without tying up the event loop.
        loop = asyncio.get_running_loop()

        def _invoke_sync_default():
            from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import create_embedding

            config = self._build_user_app_config(provider, model)
            return create_embedding(
                text,
                config,
                model_id_override=f"{provider}:{model}",
            )

        if last_provider_error:
            logger.debug(
                "Async provider resolution failed for provider '{provider}' (model '{model}'): {error}; falling back to executor path.",
                provider=provider,
                model=model,
                error=last_provider_error,
            )

        return await loop.run_in_executor(None, _invoke_sync_default)

    async def _create_local_embeddings(
        self,
        texts: list[str],
        model: str,
    ) -> list[list[float]]:
        """Create local (in-process) embeddings without HTTP delegation."""
        from tldw_Server_API.app.core.Embeddings.async_embeddings import get_async_embedding_service

        service = get_async_embedding_service()
        provider = service.providers.get("local")
        if provider is None:
            raise ValueError("Local embeddings provider is not configured.")

        embeddings = await provider.create_embeddings_batch(texts, model=model)
        if len(embeddings) != len(texts):
            raise ValueError("Local embeddings provider returned unexpected batch size.")
        return embeddings

    def _resolve_provider_name(
        self,
        provider: str,
        model: str,
        config_override: Optional[dict[str, Any]],
    ) -> str:
        """Resolve provider name, mapping local to local_api when configured."""
        if not provider:
            return provider
        provider_lower = provider.lower()
        if provider_lower not in {"local", "local_api"}:
            return provider
        if provider_lower == "local_api":
            return "local_api"

        api_url = self._resolve_local_api_url(provider, model, config_override)
        if api_url:
            return "local_api"
        return "local"

    def _resolve_local_api_url(
        self,
        provider: str,
        model: str,
        config_override: Optional[dict[str, Any]],
    ) -> Optional[str]:
        """Resolve API URL for local provider overrides."""
        if isinstance(config_override, dict):
            emb_cfg = config_override.get("embedding_config", {}) or {}
            models = emb_cfg.get("models", {}) or {}
            model_id = f"{provider}:{model}"
            _, spec = _resolve_model_spec(models, model_id, model)
            api_url = _get_spec_value(spec, "api_url")
            if api_url:
                return str(api_url)
            for section_key in ("local_api", "local"):
                section = config_override.get(section_key)
                if isinstance(section, dict):
                    api_url = section.get("api_url")
                    if api_url:
                        return str(api_url)

        provider_config = self.config.get_provider("local")
        if provider_config and provider_config.api_url:
            return provider_config.api_url
        provider_config = self.config.get_provider("local_api")
        if provider_config and provider_config.api_url:
            return provider_config.api_url
        return None

    def _build_user_app_config(self, provider: str, model: str) -> dict[str, Any]:
        """Construct full user app config for embeddings executor."""
        provider_config = None
        candidates = {
            provider,
            self._normalize_provider_name(provider),
        }
        alias = self._alias_provider_name(provider)
        if alias:
            candidates.add(alias)

        for candidate in candidates:
            provider_config = self.config.get_provider(candidate)
            if provider_config is not None:
                break
        if provider_config is None:
            raise ValueError(f"Provider '{provider}' is not configured for embeddings batching.")

        model_id = f"{provider}:{model}"

        normalized_provider = self._normalize_provider_name(provider)

        model_entry: dict[str, Any] = {
            "provider": normalized_provider,
            "model_name_or_path": model
        }
        if provider_config.api_key:
            model_entry["api_key"] = provider_config.api_key
        if provider_config.api_url:
            model_entry["api_url"] = provider_config.api_url

        user_app_config: dict[str, Any] = {
            "embedding_config": {
                "default_model_id": model_id,
                "models": {
                    model_id: model_entry
                }
            }
        }

        provider_section = self._build_provider_section(provider_config)
        if provider_section:
            user_app_config.update(provider_section)

        return user_app_config

    @staticmethod
    def _normalize_provider_name(provider: str) -> str:
        """Normalize provider identifiers for embedding config compatibility."""
        mapping = {
            "local": "local_api"
        }
        return mapping.get(provider.lower(), provider)

    @staticmethod
    def _alias_provider_name(provider: str) -> Optional[str]:
        """Return alternate configuration key for provider names."""
        mapping = {
            "local_api": "local",
            "local": "local_api",
        }
        return mapping.get(provider.lower())

    @staticmethod
    def _fingerprint_config(config: Optional[dict[str, Any]]) -> str:
        """Return a stable fingerprint for a config override."""
        if config is None:
            return "__default__"

        def _normalize(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return _normalize(value.model_dump())
            if isinstance(value, dict):
                return {key: _normalize(value[key]) for key in sorted(value.keys())}
            if isinstance(value, (list, tuple)):
                return [_normalize(v) for v in value]
            if isinstance(value, set):
                return sorted(_normalize(v) for v in value)
            if callable(value):
                return getattr(value, "__qualname__", repr(value))
            return value

        try:
            normalized = _normalize(config)
            payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
            return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        except _BATCHER_FINGERPRINT_EXCEPTIONS as exc:
            with contextlib.suppress(_BATCHER_NONCRITICAL_EXCEPTIONS):
                logger.debug(f"Falling back to identity fingerprint for config override due to: {exc}")
            return f"cfg_fallback:{id(config)}"

    def _queue_key(
        self,
        provider: str,
        model: str,
        config_override: Optional[dict[str, Any]],
    ) -> tuple[str, str, str]:
        """Build queue key incorporating provider, model, and config override."""
        return (provider, model, self._fingerprint_config(config_override))

    @staticmethod
    def _queue_label(queue_key: tuple[str, str, str]) -> str:
        """Readable label for queue keys (for logging/tests)."""
        provider, model, fingerprint = queue_key
        return f"{provider}:{model}:{fingerprint}"

    @staticmethod
    def _build_provider_section(provider_config: ProviderConfig) -> Optional[dict[str, Any]]:
        """Build provider-specific top-level config payload (e.g., API keys)."""
        section_key_map = {
            "openai": "openai_api",
            "huggingface": "huggingface_api",
            "local_api": "local_api",
            "local": "local_api"
        }

        section_key = section_key_map.get(provider_config.name.lower())
        if not section_key:
            return None

        section_payload: dict[str, Any] = {}
        if provider_config.api_key:
            section_payload["api_key"] = provider_config.api_key
        if provider_config.api_url:
            section_payload["api_url"] = provider_config.api_url

        return {section_key: section_payload} if section_payload else None

    def _update_adaptive_params(self, batch_size: int, processing_time: float):
        """
        Update adaptive batching parameters based on performance.

        Args:
            batch_size: Size of processed batch
            processing_time: Time taken to process
        """
        # Calculate throughput
        throughput = batch_size / processing_time if processing_time > 0 else 0
        self.adaptive_params['throughput_history'].append(throughput)

        # Adjust parameters based on throughput trend
        if len(self.adaptive_params['throughput_history']) >= 3:
            recent = list(self.adaptive_params['throughput_history'])[-3:]
            avg_throughput = sum(recent) / len(recent)

            # If throughput is decreasing, reduce batch size
            if recent[-1] < avg_throughput * 0.9:
                self.adaptive_params['current_batch_size'] = max(
                    1,
                    int(self.adaptive_params['current_batch_size'] * 0.9)
                )
                logger.debug(
                    f"Reduced batch size to {self.adaptive_params['current_batch_size']}"
                )

            # If throughput is increasing, increase batch size
            elif recent[-1] > avg_throughput * 1.1:
                self.adaptive_params['current_batch_size'] = min(
                    self.max_batch_size,
                    int(self.adaptive_params['current_batch_size'] * 1.1)
                )
                logger.debug(
                    f"Increased batch size to {self.adaptive_params['current_batch_size']}"
                )

            # Adjust timeout based on batch filling rate
            if batch_size < self.adaptive_params['current_batch_size'] * 0.5:
                # Batches not filling, reduce timeout
                self.adaptive_params['current_timeout'] = max(
                    10,
                    int(self.adaptive_params['current_timeout'] * 0.9)
                )
            elif batch_size >= self.adaptive_params['current_batch_size']:
                # Batches filling completely, can increase timeout
                self.adaptive_params['current_timeout'] = min(
                    1000,
                    int(self.adaptive_params['current_timeout'] * 1.1)
                )

    def cleanup_empty_queues(self) -> int:
        """
        Remove empty queue entries to prevent memory accumulation.

        Returns:
            Number of empty queues removed.
        """
        empty_keys = [
            key for key, queue in self.queues.items()
            if not queue and key not in self.processing_tasks
        ]
        for key in empty_keys:
            del self.queues[key]

        if empty_keys:
            logger.debug(f"Cleaned up {len(empty_keys)} empty queue keys from batcher")

        return len(empty_keys)

    def get_statistics(self) -> dict[str, Any]:
        """Get batching statistics"""
        # Opportunistically clean up empty queues during status check
        self.cleanup_empty_queues()

        queue_sizes = {
            self._queue_label(key): len(queue) for key, queue in self.queues.items()
        }

        return {
            'enabled': self.enabled,
            'total_batches': self.stats['total_batches'],
            'total_requests': self.stats['total_requests'],
            'average_batch_size': self.stats['average_batch_size'],
            'average_wait_time': self.stats['average_wait_time'],
            'queue_sizes': queue_sizes,
            'adaptive_params': self.adaptive_params if self.adaptive_batching else None
        }

    async def flush_all_queues(self):
        """Force process all pending requests in queues"""
        for queue_key, queue in self.queues.items():
            if queue:
                provider, model, _ = queue_key
                batch = list(queue)
                queue.clear()
                await self._process_batch(batch, provider, model)

    async def shutdown(self):
        """Gracefully shutdown the batcher"""
        self._shutdown_requested = True
        # Process remaining requests
        await self.flush_all_queues()

        # Cancel processing tasks
        current_loop = asyncio.get_running_loop()
        tasks_to_await: list[asyncio.Task] = []
        for task in list(self.processing_tasks.values()):
            task_loop = task.get_loop()
            if task_loop is current_loop:
                task.cancel()
                tasks_to_await.append(task)
            else:
                try:
                    if not task.done():
                        task_loop.call_soon_threadsafe(task.cancel)
                except _BATCHER_SHUTDOWN_EXCEPTIONS:
                    # Loop may already be closed; best effort cancellation
                    pass

        # Wait for tasks to complete
        if tasks_to_await:
            await asyncio.gather(*tasks_to_await, return_exceptions=True)
        self.processing_tasks.clear()

        # Clean up empty queues
        self.cleanup_empty_queues()
        self.queues.clear()

        logger.info("Request batcher shutdown complete")


# Global batcher instance
_batcher_fallback: Optional[RequestBatcher] = None
_batchers_by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, RequestBatcher]" = (
    weakref.WeakKeyDictionary()
)
_batchers_lock = threading.Lock()


def get_batcher() -> RequestBatcher:
    """Get or create the global batcher instance."""
    global _batcher_fallback
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    with _batchers_lock:
        if loop is not None:
            batcher = _batchers_by_loop.get(loop)
            if batcher is None:
                batcher = RequestBatcher()
                batcher._loop = loop
                _batchers_by_loop[loop] = batcher
            return batcher

        if _batcher_fallback is None:
            _batcher_fallback = RequestBatcher()
        return _batcher_fallback


def _resolve_provider_model(
    config: dict[str, Any],
    model_id_override: Optional[str],
) -> tuple[str, str]:
    """Resolve provider/model from override or embedding_config defaults."""
    emb_cfg = (config or {}).get("embedding_config", {}) or {}
    models = emb_cfg.get("models", {}) or {}

    model_id = model_id_override or emb_cfg.get("default_model_id")
    if model_id:
        if ":" in model_id:
            provider, model = model_id.split(":", 1)
            if provider and model:
                return provider, model
        key, spec = _resolve_model_spec(models, model_id, model_id)
        provider = _get_spec_value(spec, "provider")
        model_name = _get_spec_value(spec, "model_name_or_path") or model_id
        if provider and model_name:
            return str(provider), str(model_name)
        if key and ":" in key:
            provider, _ = key.split(":", 1)
            return provider, model_id
        if ":" not in model_id:
            provider = emb_cfg.get("default_provider")
            if provider:
                return provider, model_id

    provider = emb_cfg.get("default_provider", "openai")
    model = emb_cfg.get("default_model", "text-embedding-3-small")
    return provider, model


# Convenience function for batched requests
async def create_embeddings_batch_async(
    texts: list[str],
    config: dict[str, Any],
    model_id_override: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None
) -> list[list[float]]:
    """
    Create embeddings with automatic batching.

    Args:
        texts: List of input texts
        config: Configuration dictionary
        model_id_override: Optional model override

    Returns:
        List of embeddings
    """
    batcher = get_batcher()

    # Parse model info
    provider, model = _resolve_provider_model(config, model_id_override)

    # Submit requests
    tasks = []
    for text in texts:
        task = batcher.submit_request(
            text,
            model,
            provider,
            metadata=metadata,
            config_override=config,
        )
        tasks.append(task)

    # Wait for all results
    results = await asyncio.gather(*tasks)

    return results
