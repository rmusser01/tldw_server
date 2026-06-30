# tts_resource_manager.py
# Description: Resource management for TTS operations including connection pooling, memory management, and cleanup
#
# Imports
import asyncio
import gc
import os
import sys
import threading
import time
import weakref
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

import psutil

#
# Third-party Imports
from loguru import logger

#
# Local Imports
from .tts_exceptions import (
    TTSInsufficientMemoryError,
    TTSNetworkError,
    TTSResourceError,
)

_TTS_RESOURCE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    EOFError,
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
    ZeroDivisionError,
)

#
# Conditional imports for type checking
if TYPE_CHECKING:
    pass
#
#######################################################################################################################
#
# Resource Management System

class ResourceType(Enum):
    """Types of resources managed by the system"""
    HTTP_CONNECTION = "http_connection"
    MODEL_INSTANCE = "model_instance"
    STREAMING_SESSION = "streaming_session"
    TEMP_FILE = "temp_file"
    MEMORY_BUFFER = "memory_buffer"


@dataclass
class ResourceMetrics:
    """Metrics for resource usage tracking"""
    created_at: float
    last_used: float
    use_count: int = 0
    memory_usage: int = 0  # bytes
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_usage(self):
        """Update usage statistics"""
        self.last_used = time.time()
        self.use_count += 1


@dataclass
class ModelCacheEntry:
    """Track cached TTS model instances for LRU eviction."""
    provider: str
    cache_key: str
    model_ref: weakref.ReferenceType
    cleanup_callback: Optional[Callable]
    created_at: float
    last_used: float
    use_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingSession:
    """Manages a streaming TTS session with proper cleanup"""

    def __init__(
        self,
        session_id: str,
        provider: str,
        cleanup_callback: Optional[Callable] = None
    ):
        """
        Initialize streaming session.

        Args:
            session_id: Unique session identifier
            provider: TTS provider name
            cleanup_callback: Optional cleanup function
        """
        self.session_id = session_id
        self.provider = provider
        self.cleanup_callback = cleanup_callback
        self.created_at = time.time()
        self.last_activity = time.time()
        self.is_active = True
        self.chunks_sent = 0
        self.bytes_sent = 0
        self.error_count = 0
        self._cleanup_tasks: set[asyncio.Task] = set()

    # Backward-compat properties used by tests
    @property
    def bytes_streamed(self) -> int:
        return self.bytes_sent

    @bytes_streamed.setter
    def bytes_streamed(self, value: int) -> None:
        self.bytes_sent = value

    @property
    def start_time(self) -> float:
        return self.created_at

    @start_time.setter
    def start_time(self, value: float) -> None:
        self.created_at = value

    async def track_activity(self, chunk_size: int = 0):
        """Track session activity"""
        self.last_activity = time.time()
        if chunk_size > 0:
            self.chunks_sent += 1
            self.bytes_sent += chunk_size

    async def add_cleanup_task(self, coro):
        """Add cleanup task to be executed when session closes"""
        task = asyncio.create_task(coro)
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def close(self):
        """Close the streaming session and cleanup resources"""
        if not self.is_active:
            return

        self.is_active = False

        try:
            # Execute cleanup callback
            if self.cleanup_callback:
                if asyncio.iscoroutinefunction(self.cleanup_callback):
                    await self.cleanup_callback()
                else:
                    self.cleanup_callback()

            # Wait for cleanup tasks
            if self._cleanup_tasks:
                await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)

            logger.debug(
                f"Streaming session {self.session_id} closed: "
                f"chunks={self.chunks_sent}, bytes={self.bytes_sent}, "
                f"duration={time.time() - self.created_at:.2f}s"
            )

        except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
            logger.error(f"Error closing streaming session {self.session_id}")

    def is_expired(self, timeout: float = 300) -> bool:
        """Check if session has expired"""
        return time.time() - self.last_activity > timeout


class HTTPConnectionPool:
    """HTTP connection pool for API-based TTS providers"""

    def __init__(
        self,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
        keepalive_expiry: float = 30.0,
        timeout: float = 60.0
    ):
        """
        Initialize HTTP connection pool.

        Args:
            max_connections: Maximum total connections
            max_keepalive_connections: Maximum keep-alive connections
            keepalive_expiry: Keep-alive timeout in seconds
            timeout: Request timeout in seconds
        """
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        self.timeout = timeout

        # Connection pools per provider
        self._pools: dict[str, Any] = {}
        self._pool_metrics: dict[str, ResourceMetrics] = {}
        self._lock = asyncio.Lock()

        # Backward-compatibility: tests reference `_clients`; alias to `_pools`.
    @property
    def _clients(self) -> dict[str, Any]:
        return self._pools

    async def get_client(self, provider: str, base_url: Optional[str] = None) -> Any:
        """
        Get or create HTTP client for provider.

        Args:
            provider: Provider name
            base_url: Optional base URL for the client

        Returns:
            HTTP client instance
        """
        async with self._lock:
            if provider not in self._pools:
                # Use centralized factory for consistent trust_env/http2/limits
                try:
                    from tldw_Server_API.app.core.http_client import (
                        build_limits,
                        create_async_client,
                    )
                    limits = build_limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive_connections,
                        keepalive_expiry=self.keepalive_expiry,
                    )
                    client = create_async_client(
                        timeout=self.timeout,
                        base_url=base_url,
                        limits=limits,
                    )
                except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS as e:
                    # If central factory is unavailable, surface an error instead of constructing directly
                    raise TTSNetworkError(f"Failed to create HTTP client via factory: {e}") from e

                self._pools[provider] = client
                self._pool_metrics[provider] = ResourceMetrics(
                    created_at=time.time(),
                    last_used=time.time(),
                    metadata={"provider": provider, "base_url": base_url}
                )

                logger.debug(f"Created HTTP connection pool for {provider}")

            # Update metrics
            metrics = self._pool_metrics[provider]
            metrics.update_usage()

            return self._pools[provider]

    async def close_pool(self, provider: str):
        """Close connection pool for specific provider"""
        async with self._lock:
            if provider in self._pools:
                await self._pools[provider].aclose()
                del self._pools[provider]
                del self._pool_metrics[provider]
                logger.debug(f"Closed HTTP connection pool for {provider}")

    async def close_client(self, provider: str):
        """Close a specific client (alias for close_pool)"""
        await self.close_pool(provider)

    async def close_all(self):
        """Close all connection pools"""
        async with self._lock:
            tasks = []
            for _provider, client in self._pools.items():
                tasks.append(client.aclose())

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            self._pools.clear()
            self._pool_metrics.clear()
            logger.info("Closed all HTTP connection pools")

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get connection pool statistics"""
        stats = {}
        for provider, metrics in self._pool_metrics.items():
            stats[provider] = {
                "created_at": metrics.created_at,
                "last_used": metrics.last_used,
                "use_count": metrics.use_count,
                "is_active": metrics.is_active,
                "metadata": metrics.metadata
            }
        return stats


class MemoryMonitor:
    """Memory monitoring and management for local TTS models"""

    def __init__(
        self,
        memory_threshold: float = 0.80,  # 80% of total memory
        check_interval: float = 30.0,    # Check every 30 seconds
        cleanup_threshold: float = 0.90,  # Force cleanup at 90%
        warning_threshold: Optional[float] = None,  # Alias for memory_threshold (for tests)
        critical_threshold: Optional[float] = None  # Alias for cleanup_threshold (for tests)
    ):
        """
        Initialize memory monitor.

        Args:
            memory_threshold: Memory usage threshold (0.0-1.0)
            check_interval: Monitoring check interval in seconds
            cleanup_threshold: Force cleanup threshold (0.0-1.0)
        """
        # Handle aliases from tests
        if warning_threshold is not None:
            memory_threshold = warning_threshold / 100.0
        if critical_threshold is not None:
            cleanup_threshold = critical_threshold / 100.0

        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.cleanup_threshold = cleanup_threshold

        # Store as percentages for compatibility
        self.warning_threshold = memory_threshold * 100
        self.critical_threshold = cleanup_threshold * 100

        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._model_references: set[weakref.ReferenceType] = set()
        self._last_check_time = 0
        self._last_memory_usage = None
        self._cleanup_callbacks: list[Callable] = []

        # Get system memory info
        self.total_memory = psutil.virtual_memory().total

    def register_model(self, model_instance: Any, cleanup_callback: Optional[Callable] = None):
        """
        Register a model instance for memory monitoring.

        Args:
            model_instance: Model instance to monitor
            cleanup_callback: Optional cleanup function for the model
        """
        # Use weak reference to avoid circular references
        weak_ref = weakref.ref(model_instance)
        self._model_references.add(weak_ref)

        if cleanup_callback:
            self._cleanup_callbacks.append(cleanup_callback)

        logger.debug(f"Registered model for memory monitoring: {type(model_instance).__name__}")

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage statistics (with simple caching and robust fallbacks)"""
        now = time.time()
        if (
            self._last_memory_usage is not None
            and (now - self._last_check_time) < self.check_interval
        ):
            return self._last_memory_usage

        memory = psutil.virtual_memory()
        process = psutil.Process()
        mb = 1024 * 1024

        def _as_int(value, default=0):
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return default

        def _as_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        total_raw = getattr(memory, "total", 0)
        available_raw = getattr(memory, "available", 0)
        used_raw = getattr(memory, "used", None)
        if used_raw is None and total_raw and available_raw is not None:
            used_raw = total_raw - available_raw
        free_raw = getattr(memory, "free", None)
        if free_raw is None:
            free_raw = available_raw
        percent_raw = getattr(memory, "percent", None)
        if percent_raw is None and total_raw:
            try:
                percent_raw = (used_raw / total_raw) * 100
            except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                percent_raw = 0

        total = _as_int(total_raw)
        available = _as_int(available_raw)
        used = _as_int(used_raw, default=max(total - available, 0))
        free = _as_int(free_raw, default=available)
        percent = _as_float(percent_raw)

        # Compute warning/critical from the same percent value to avoid extra psutil calls
        usage_ratio = (percent / 100.0) if percent is not None else 0.0
        stats = {
            "total": total,
            "available": available,
            "used": used,
            "percent": percent,
            "free": free,
            "total_mb": total // mb if total else 0,
            "available_mb": available // mb if available else 0,
            "used_mb": used // mb if used else 0,
            "free_mb": free // mb if free else 0,
            "process_mb": _as_int(process.memory_info().rss) // mb,
            "threshold": self.memory_threshold * 100,
            "cleanup_threshold": self.cleanup_threshold * 100,
            "is_warning": usage_ratio > self.memory_threshold,
            "is_critical": usage_ratio > self.cleanup_threshold,
        }

        self._last_memory_usage = stats
        self._last_check_time = now
        return stats

    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical"""
        try:
            percent = float(psutil.virtual_memory().percent)
        except (TypeError, ValueError):
            percent = 0.0
        usage = percent / 100.0
        return usage > self.cleanup_threshold

    def is_memory_high(self) -> bool:
        """Check if memory usage is high"""
        try:
            percent = float(psutil.virtual_memory().percent)
        except (TypeError, ValueError):
            percent = 0.0
        usage = percent / 100.0
        return usage > self.memory_threshold

    def is_memory_warning(self) -> bool:
        """Check if memory usage is at warning level (alias for is_memory_high)"""
        return self.is_memory_high()

    async def force_cleanup(self):
        """Force memory cleanup"""
        logger.warning("Forcing memory cleanup due to high usage")

        # Run cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                logger.error("Error in cleanup callback")

        # Clean up dead references
        self._model_references = {ref for ref in self._model_references if ref() is not None}

        # Force garbage collection
        gc.collect()

        # Check if cleanup was effective
        if self.is_memory_critical():
            raise TTSInsufficientMemoryError(
                "Memory usage remains critical after cleanup",
                details=self.get_memory_usage()
            )

    async def start_monitoring(self):
        """Start memory monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Memory monitoring started")

    async def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task
        logger.info("Memory monitoring stopped")

    async def _monitor_loop(self):
        """Memory monitoring loop"""
        while self._monitoring:
            try:
                if self.is_memory_critical():
                    await self.force_cleanup()
                elif self.is_memory_high():
                    logger.warning(f"High memory usage: {psutil.virtual_memory().percent:.1f}%")

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                logger.error("Error in memory monitoring")
                await asyncio.sleep(self.check_interval)


class StreamingSessionManager:
    """Manages streaming audio sessions"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.sessions: dict[str, StreamingSession] = {}
        self.max_sessions = self.config.get("max_streaming_sessions", 10)
        self._lock = threading.Lock()

    # Tests reference `_sessions`; expose alias to `sessions`.
    @property
    def _sessions(self) -> dict[str, StreamingSession]:
        return self.sessions

    async def create_session(self, provider: str, session_id: Optional[str] = None, **kwargs) -> str:
        """Create a new streaming session

        Args:
            provider: TTS provider name
            session_id: Optional session ID, will be generated if not provided
            **kwargs: Additional session parameters

        Returns:
            Session ID
        """
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())

        with self._lock:
            if len(self.sessions) >= self.max_sessions:
                # Clean up old sessions
                self._cleanup_old_sessions()

            if len(self.sessions) >= self.max_sessions:
                raise TTSResourceError("Maximum streaming sessions reached")

            session = StreamingSession(session_id=session_id, provider=provider, **kwargs)
            self.sessions[session_id] = session
            return session_id

    async def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get an existing session"""
        return self.sessions.get(session_id)

    async def update_session(self, session_id: str, bytes_sent: int = 0, chunks_sent: int = 0) -> bool:
        """Update session statistics

        Args:
            session_id: Session ID
            bytes_sent: Additional bytes sent
            chunks_sent: Additional chunks sent

        Returns:
            True if session was updated, False if not found
        """
        session = self.sessions.get(session_id)
        if session:
            session.bytes_sent += bytes_sent
            session.chunks_sent += chunks_sent
            session.last_activity = time.time()
            return True
        return False

    async def close_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Close a streaming session and return stats

        Args:
            session_id: Session ID

        Returns:
            Session statistics or None if not found
        """
        with self._lock:
            session = self.sessions.pop(session_id, None)
        if not session:
            return None

        await session.close()
        duration = time.time() - session.created_at
        return {
            "session_id": session_id,
            "provider": session.provider,
            "duration": duration,
            "bytes_streamed": session.bytes_sent,
            "chunks_sent": session.chunks_sent,
            "error_count": session.error_count
        }

    def end_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """End a streaming session and return stats"""
        with self._lock:
            session = self.sessions.pop(session_id, None)
        if not session:
            return None

        # Dispatch asynchronous close; if the caller awaits elsewhere it can
        # manage completion, otherwise we best-effort kick it off.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; close synchronously.
            try:
                asyncio.run(session.close())
            except RuntimeError:
                # Fallback: mark inactive without cleanup.
                logger.warning("Could not run session.close(); marking session inactive")
                session.is_active = False
        else:
            asyncio.create_task(session.close())
            session.is_active = False
        stats = {
            "session_id": session_id,
            "provider": session.provider,
            "duration": time.time() - session.created_at,
            "bytes_streamed": session.bytes_sent,
            "chunks_sent": session.chunks_sent,
            "error_count": session.error_count
        }
        return stats

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        with self._lock:
            self._cleanup_old_sessions()

    def _cleanup_old_sessions(self):
        """Internal method to clean up old sessions"""
        current_time = time.time()
        expired = []

        for sid, session in self.sessions.items():
            if current_time - session.start_time > 3600:  # 1 hour timeout
                expired.append(sid)

        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        for sid in expired:
            session = self.sessions.pop(sid)
            if loop and loop.is_running():
                asyncio.create_task(session.close())
            else:
                try:
                    asyncio.run(session.close())
                except RuntimeError:
                    logger.warning(f"Unable to close session {sid} during cleanup")
                    session.is_active = False

    async def get_active_sessions(self) -> list[StreamingSession]:
        """Get list of active session objects"""
        return [s for s in self.sessions.values() if s.is_active]

    async def cleanup_inactive(self, max_age_seconds: int = 3600) -> None:
        """Remove inactive sessions older than the given age."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            to_remove = [
                sid for sid, s in self.sessions.items()
                if (not s.is_active) and (s.start_time < cutoff)
            ]
            for sid in to_remove:
                self.sessions.pop(sid, None)


class TTSResourceManager:
    """Central resource manager for TTS operations"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize TTS resource manager.

        Args:
            config: Resource management configuration
        """
        self.config = config or {}

        # Initialize components
        self.connection_pool = HTTPConnectionPool(
            max_connections=self.config.get("max_http_connections", self.config.get("max_connections", 10)),
            max_keepalive_connections=self.config.get("max_keepalive_connections", 5),
            keepalive_expiry=self.config.get("keepalive_expiry", 30.0),
            timeout=self.config.get("http_timeout", self.config.get("connection_timeout", 60.0))
        )

        self.memory_monitor = MemoryMonitor(
            memory_threshold=self.config.get("memory_threshold", 0.80),
            check_interval=self.config.get("memory_check_interval", 30.0),
            cleanup_threshold=self.config.get("memory_cleanup_threshold", 0.90),
            warning_threshold=self.config.get("memory_warning_threshold"),
            critical_threshold=self.config.get("memory_critical_threshold")
        )

        # Streaming session management
        self.session_manager = StreamingSessionManager(self.config)
        self._streaming_sessions: dict[str, StreamingSession] = {}
        self._session_cleanup_task: Optional[asyncio.Task] = None
        self._session_timeout = self.config.get("streaming_session_timeout", 300)  # 5 minutes

        # Model instance tracking
        self._model_instances: dict[str, weakref.ReferenceType] = {}
        self._registered_models: dict[str, dict[str, Any]] = {}
        self._model_cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        self._model_cache_lock = threading.Lock()
        self._model_cache_evictions = 0
        self._model_cache_max_entries = self._coerce_cache_limit(
            self.config.get("model_cache_max_entries", self.config.get("max_cached_models"))
        )

        # Resource cleanup
        self._cleanup_handlers: dict[ResourceType, list[Callable]] = {}

        logger.info("TTS Resource Manager initialized")

    @staticmethod
    def _coerce_cache_limit(raw_value: Any) -> Optional[int]:
        """Normalize cache size limits; return None when unlimited/disabled."""
        if raw_value is None:
            return None
        try:
            limit = int(raw_value)
        except (TypeError, ValueError):
            return None
        if limit <= 0:
            return None
        return limit

    @staticmethod
    def _normalize_cache_key(provider: str, model_key: Optional[str] = None) -> str:
        """Normalize provider/model into a cache key."""
        provider_key = (provider or "").strip().lower()
        model_part = (model_key or "").strip().lower()
        if provider_key and model_part:
            return f"{provider_key}:{model_part}"
        if provider_key:
            return provider_key
        return model_part

    @staticmethod
    def _cleanup_device_cache() -> None:
        """Best-effort device cleanup after model eviction."""
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
            pass

    async def initialize(self):
        """Initialize resource management"""
        # Start memory monitoring
        await self.memory_monitor.start_monitoring()

        # Start session cleanup task
        self._session_cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

        logger.info("TTS Resource Manager started")

    async def shutdown(self):
        """Shutdown resource manager and cleanup all resources"""
        logger.info("Shutting down TTS Resource Manager")

        # Stop monitoring
        await self.memory_monitor.stop_monitoring()

        # Stop session cleanup
        if self._session_cleanup_task:
            self._session_cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._session_cleanup_task

        # Close all streaming sessions
        sessions = list(self._streaming_sessions.values())
        for session in sessions:
            await session.close()
        self._streaming_sessions.clear()

        # Close connection pools
        await self.connection_pool.close_all()

        # Run cleanup handlers
        for resource_type, handlers in self._cleanup_handlers.items():
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                    logger.error(f"Error in {resource_type} cleanup handler")

        logger.info("TTS Resource Manager shutdown complete")

    @asynccontextmanager
    async def streaming_session(self, session_id: str, provider: str, cleanup_callback: Optional[Callable] = None):
        """
        Context manager for streaming sessions.

        Args:
            session_id: Unique session identifier
            provider: TTS provider name
            cleanup_callback: Optional cleanup callback
        """
        session = StreamingSession(session_id, provider, cleanup_callback)
        self._streaming_sessions[session_id] = session

        try:
            logger.debug(f"Started streaming session {session_id} for {provider}")
            yield session
        finally:
            await session.close()
            self._streaming_sessions.pop(session_id, None)
            logger.debug(f"Closed streaming session {session_id}")

    async def get_http_client(self, provider: str, base_url: Optional[str] = None) -> Any:
        """Get HTTP client for provider"""
        return await self.connection_pool.get_client(provider, base_url)

    def register_model(
        self,
        provider: str,
        model_instance: Any,
        cleanup_callback: Optional[Callable] = None,
        model_key: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Register model instance for resource management."""
        cache_key = self._normalize_cache_key(provider, model_key)
        now = time.time()

        with self._model_cache_lock:
            entry = self._model_cache.get(cache_key)
            if entry is None:
                entry = ModelCacheEntry(
                    provider=provider,
                    cache_key=cache_key,
                    model_ref=weakref.ref(model_instance),
                    cleanup_callback=cleanup_callback,
                    created_at=now,
                    last_used=now,
                    use_count=1,
                    metadata=metadata or {},
                )
                self._model_cache[cache_key] = entry
            else:
                entry.model_ref = weakref.ref(model_instance)
                entry.cleanup_callback = cleanup_callback
                entry.last_used = now
                entry.use_count += 1
                if metadata:
                    entry.metadata.update(metadata)
                self._model_cache.move_to_end(cache_key)

            # Track explicit registered models as expected by tests
            self._registered_models[provider] = {"model": model_instance, "cleanup": cleanup_callback}
            self._model_instances[provider] = weakref.ref(model_instance)

        self.memory_monitor.register_model(model_instance, cleanup_callback)

        eviction_coros = self._evict_models_if_needed(exclude_keys={cache_key})
        if eviction_coros:
            async def _run_evictions():
                for coro in eviction_coros:
                    await coro
            return _run_evictions()
        return None

    def touch_model(self, provider: str, model_key: Optional[str] = None) -> bool:
        """Mark a cached model as recently used."""
        cache_key = self._normalize_cache_key(provider, model_key)
        if not cache_key:
            return False
        with self._model_cache_lock:
            entry = self._model_cache.get(cache_key)
            if entry is None:
                return False
            entry.last_used = time.time()
            entry.use_count += 1
            self._model_cache.move_to_end(cache_key)
        return True

    def _evict_models_if_needed(self, exclude_keys: Optional[set[str]] = None) -> list[Any]:
        """Evict least-recently-used models when limits are exceeded."""
        async_cleanups: list[Any] = []
        exclude_keys = exclude_keys or set()
        max_entries = self._model_cache_max_entries

        with self._model_cache_lock:
            while max_entries is not None and len(self._model_cache) > max_entries:
                evicted = self._evict_one_locked(exclude_keys, async_cleanups, reason="capacity")
                if not evicted:
                    break

            if self.memory_monitor.is_memory_critical():
                # Best-effort eviction if memory is critical
                self._evict_one_locked(exclude_keys, async_cleanups, reason="memory")

        return async_cleanups

    def _evict_one_locked(self, exclude_keys: set[str], async_cleanups: list[Any], reason: str) -> bool:
        """Evict a single oldest entry (lock must be held)."""
        for key, entry in list(self._model_cache.items()):
            if key in exclude_keys:
                continue
            _cleanup = self._evict_entry_locked(key, entry, reason)
            if _cleanup is not None:
                async_cleanups.append(_cleanup)
            return True
        return False

    def _evict_entry_locked(self, cache_key: str, entry: ModelCacheEntry, reason: str) -> Optional[Any]:
        """Remove cache entry and run cleanup (lock must be held)."""
        self._model_cache.pop(cache_key, None)
        self._model_cache_evictions += 1

        provider = entry.provider
        self._model_instances.pop(provider, None)
        self._registered_models.pop(provider, None)

        cleanup_cb = entry.cleanup_callback
        logger.info(f"Evicting TTS model cache entry '{cache_key}' (reason={reason})")

        if cleanup_cb is None:
            self._cleanup_device_cache()
            return None

        if asyncio.iscoroutinefunction(cleanup_cb):
            async def _wrapped():
                try:
                    await cleanup_cb()
                except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                    logger.error(f"Error cleaning model for {cache_key}")
                finally:
                    self._cleanup_device_cache()
            return _wrapped()

        try:
            cleanup_cb()
        except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
            logger.error(f"Error cleaning model for {cache_key}")
        self._cleanup_device_cache()
        return None

    def register_cleanup_handler(self, resource_type: ResourceType, handler: Callable):
        """Register cleanup handler for resource type"""
        if resource_type not in self._cleanup_handlers:
            self._cleanup_handlers[resource_type] = []
        self._cleanup_handlers[resource_type].append(handler)

    async def _cleanup_expired_sessions(self):
        """Cleanup expired streaming sessions"""
        while True:
            try:
                time.time()
                expired_sessions = []

                for session_id, session in self._streaming_sessions.items():
                    if session.is_expired(self._session_timeout):
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    session = self._streaming_sessions.pop(session_id, None)
                    if session:
                        await session.close()
                        logger.info(f"Cleaned up expired streaming session {session_id}")

                # Sleep for cleanup interval
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                logger.error("Error in session cleanup")
                await asyncio.sleep(60)

    async def unregister_model(self, provider: str):
        """Unregister a model instance and run its cleanup callback"""
        entry = self._registered_models.pop(provider, None)
        if entry is not None:
            cleanup_cb = entry.get("cleanup")
            if cleanup_cb:
                try:
                    if asyncio.iscoroutinefunction(cleanup_cb):
                        await cleanup_cb()
                    else:
                        cleanup_cb()
                except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                    logger.error(f"Error in model cleanup for {provider}")
            logger.debug(f"Unregistered model for provider: {provider}")
        # Drop any cached entries for this provider
        with self._model_cache_lock:
            keys_to_drop = [
                key for key, cache_entry in self._model_cache.items()
                if cache_entry.provider == provider
            ]
            for key in keys_to_drop:
                self._model_cache.pop(key, None)
        self._model_instances.pop(provider, None)

    async def create_streaming_session(self, provider: str) -> str:
        """Create a new streaming session

        Args:
            provider: Provider name

        Returns:
            Session ID
        """
        return await self.session_manager.create_session(provider)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive resource statistics in test-expected shape"""
        mem = self.memory_monitor.get_memory_usage()
        connections = {
            "active": len(self.connection_pool._clients),
            "providers": list(self.connection_pool._clients.keys())
        }
        models = {
            "registered": list(self._registered_models.keys()),
            "cache_size": len(self._model_cache),
            "cache_max_entries": self._model_cache_max_entries,
            "cache_evictions": self._model_cache_evictions,
        }
        sessions = {
            "active": len(self.session_manager._sessions),
            "ids": list(self.session_manager._sessions.keys())
        }
        return {
            "memory": mem,
            "connections": connections,
            "models": models,
            "sessions": sessions
        }

    async def cleanup_all(self):
        """Cleanup all resources: models, clients, and sessions"""
        # Cleanup registered models
        # Make a copy of keys to avoid mutation during iteration
        for provider in list(self._registered_models.keys()):
            try:
                await self.unregister_model(provider)
            except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                logger.error(f"Error cleaning model {provider}")

        # Close all sessions managed by the session manager
        for sid in list(self.session_manager._sessions.keys()):
            try:
                await self.session_manager.close_session(sid)
            except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                logger.error(f"Error closing session {sid}")

        # Close all HTTP clients
        await self.connection_pool.close_all()

    def get_resource_stats(self) -> dict[str, Any]:
        """Get resource usage statistics"""
        return {
            "http_connections": self.connection_pool.get_stats(),
            "memory": self.memory_monitor.get_memory_usage(),
            "streaming_sessions": {
                "active": len(self._streaming_sessions),
                "sessions": [
                    {
                        "id": session.session_id,
                        "provider": session.provider,
                        "duration": time.time() - session.created_at,
                        "chunks_sent": session.chunks_sent,
                        "bytes_sent": session.bytes_sent
                    }
                    for session in self._streaming_sessions.values()
                ]
            },
            "model_instances": {
                provider: ref() is not None
                for provider, ref in self._model_instances.items()
            }
        }


# Global resource manager instance
_resource_manager: Optional[TTSResourceManager] = None
_manager_lock = asyncio.Lock()


def get_existing_resource_manager() -> Optional[TTSResourceManager]:
    """Return the global resource manager only when it already exists."""
    return _resource_manager


async def get_resource_manager(config: Optional[dict[str, Any]] = None) -> TTSResourceManager:
    """
    Get or create the global TTS resource manager.

    Args:
        config: Configuration for resource management

    Returns:
        TTSResourceManager instance
    """
    global _resource_manager

    def _auto_model_cache_limit() -> int:
        """Best-effort default cache size for mixed hardware deployments."""
        # In test/minimal app mode, avoid probing torch to prevent heavy/fragile imports.
        if os.getenv("MINIMAL_TEST_APP", "").lower() in {"1", "true", "yes", "y", "on"}:
            return 1
        if os.getenv("TLDW_TEST_MODE", "").lower() in {"1", "true", "yes", "y", "on"}:
            return 1
        try:
            import torch
        except Exception:
            return 1
        try:
            if hasattr(torch, "backends") and torch.backends.mps.is_available():
                return 1
        except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if torch.cuda.is_available():
                try:
                    props = torch.cuda.get_device_properties(0)
                    total_gb = float(getattr(props, "total_memory", 0)) / (1024 ** 3)
                    return 2 if total_gb >= 16.0 else 1
                except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
                    return 1
        except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
            return 1
        return 1

    if config is None:
        try:
            from .tts_config import get_tts_config
            tts_cfg = get_tts_config()
            perf = tts_cfg.performance
            cache_limit = getattr(perf, "model_cache_max_entries", None)
            if cache_limit is None:
                cache_limit = _auto_model_cache_limit()
            config = {
                "memory_warning_threshold": perf.memory_warning_threshold,
                "memory_critical_threshold": perf.memory_critical_threshold,
                "max_connections": perf.max_connections_per_provider,
                "connection_timeout": perf.connection_timeout,
                "model_cache_max_entries": cache_limit,
            }
        except _TTS_RESOURCE_NONCRITICAL_EXCEPTIONS:
            config = None

    if _resource_manager is None:
        async with _manager_lock:
            if _resource_manager is None:
                _resource_manager = TTSResourceManager(config)
                await _resource_manager.initialize()
                logger.info("Global TTS Resource Manager created")

    return _resource_manager


async def close_resource_manager():
    """Close the global resource manager"""
    global _resource_manager

    if _resource_manager:
        await _resource_manager.shutdown()
        _resource_manager = None
        logger.info("Global TTS Resource Manager closed")


# Alias for compatibility
reset_resource_manager = close_resource_manager


# Context manager for resource management
@asynccontextmanager
async def managed_resources(config: Optional[dict[str, Any]] = None):
    """Context manager for TTS resource management"""
    manager = await get_resource_manager(config)
    try:
        yield manager
    finally:
        # Don't close the global manager, just ensure it's cleaned up properly
        pass

#
# End of tts_resource_manager.py
#######################################################################################################################
