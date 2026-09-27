# connection_pool.py
# Connection pooling for API providers to improve performance

import asyncio
import threading
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import CircuitBreaker


class ConnectionPool:
    """
    Manages connection pools for different API providers.
    Provides reusable connections with proper lifecycle management.
    """

    def __init__(
        self,
        provider: str,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
        keepalive_timeout: int = 30,
        timeout_seconds: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize connection pool for a provider.

        Args:
            provider: Name of the provider (e.g., 'openai', 'cohere')
            max_connections: Maximum number of connections in pool
            max_keepalive_connections: Max idle connections to keep alive
            keepalive_timeout: Seconds to keep idle connections alive
            timeout_seconds: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
        """
        self.provider = provider
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_timeout = keepalive_timeout
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts

        self._lock = threading.RLock()
        self._usage_stats = {
            'requests': 0,
            'failures': 0,
            'total_time': 0,
            'active_connections': 0
        }
        self._semaphore: Optional[asyncio.BoundedSemaphore] = None
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None
        self._breaker = CircuitBreaker(
            name=f"{self.provider}.api_request",
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception,
            category="embeddings",
            service=self.provider,
            operation="api_request",
        )

        logger.info(
            f"ConnectionPool for {provider} initialized: "
            f"max_connections={max_connections}, "
            f"keepalive_timeout={keepalive_timeout}s"
        )

    @asynccontextmanager
    async def acquire_connection(self):
        """
        Context manager to acquire a slot from the pool.

        This class relies on http_client for actual connection reuse.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if self._semaphore is None or (loop and self._semaphore_loop is not loop):
            # Bind semaphore to the current loop to avoid cross-loop issues.
            self._semaphore = asyncio.BoundedSemaphore(self.max_connections)
            self._semaphore_loop = loop

        semaphore = self._semaphore
        if semaphore is not None:
            await semaphore.acquire()

        with self._lock:
            self._usage_stats['active_connections'] += 1

        try:
            yield None
        finally:
            with self._lock:
                self._usage_stats['active_connections'] -= 1
            if semaphore is not None:
                with suppress(Exception):
                    semaphore.release()

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        json_data: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        params: Optional[dict[str, str]] = None,
        sensitive_observability: bool = False,
        bypass_circuit_breaker: bool = False,
    ) -> dict[str, Any]:
        """Make an HTTP request through the pool.

        Credential-scoped requests bypass the provider-global circuit breaker so
        one tenant's invalid or unavailable credential cannot deny service to
        other tenants. Connection limits, retry policy, and failure statistics
        still apply through ``_request_impl``.
        """
        if bypass_circuit_breaker is True:
            return await self._request_impl(
                method,
                url,
                headers=headers,
                json_data=json_data,
                data=data,
                params=params,
                sensitive_observability=sensitive_observability,
            )
        return await self._breaker.call_async(
            self._request_impl,
            method,
            url,
            headers=headers,
            json_data=json_data,
            data=data,
            params=params,
            sensitive_observability=sensitive_observability,
        )

    async def _request_impl(
        self,
        method: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        json_data: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        params: Optional[dict[str, str]] = None,
        sensitive_observability: bool = False,
    ) -> dict[str, Any]:
        """
        Make an HTTP request using the connection pool.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            headers: Request headers
            json_data: JSON payload
            data: Form data or raw data
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            NetworkError: On network errors
            RetryExhaustedError: When retries are exhausted
            ValueError: On invalid responses
        """
        start_time = time.time()

        async with self.acquire_connection():
            resp = None
            try:
                retry = RetryPolicy(attempts=max(1, int(self.retry_attempts)))
                resp = await afetch(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    data=data,
                    params=params,
                    timeout=self.timeout_seconds,
                    retry=retry,
                    sensitive_observability=sensitive_observability,
                )
                elapsed = time.time() - start_time
                with self._lock:
                    self._usage_stats['requests'] += 1
                    self._usage_stats['total_time'] += elapsed

                status = int(getattr(resp, "status_code", 0))
                if status >= 400:
                    logger.error(
                        f"{self.provider} API error: "
                        f"status={status}"
                    )
                    raise NetworkError(f"HTTP {status}")

                ctype = (resp.headers.get("content-type", "") or "").lower()
                if "application/json" in ctype:
                    return resp.json()
                try:
                    text = resp.text
                except Exception:
                    text = ""
                return {"text": text}

            except (NetworkError, RetryExhaustedError) as e:
                with self._lock:
                    self._usage_stats['failures'] += 1
                logger.error(f"{self.provider} request error: {e}")
                raise
            except Exception as e:
                with self._lock:
                    self._usage_stats['failures'] += 1
                logger.error(f"{self.provider} unexpected error: {e}")
                raise
            finally:
                if resp is not None:
                    close = getattr(resp, "aclose", None)
                    if callable(close):
                        await close()

    async def close(self):
        """Close the connection pool and cleanup resources."""
        logger.info(f"ConnectionPool for {self.provider} closed (http_client-managed)")

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            avg_time = (
                self._usage_stats['total_time'] / self._usage_stats['requests']
                if self._usage_stats['requests'] > 0 else 0
            )

            success_rate = (
                (self._usage_stats['requests'] - self._usage_stats['failures'])
                / self._usage_stats['requests'] * 100
                if self._usage_stats['requests'] > 0 else 100
            )

            return {
                'provider': self.provider,
                'total_requests': self._usage_stats['requests'],
                'failed_requests': self._usage_stats['failures'],
                'success_rate': success_rate,
                'average_response_time': avg_time,
                'active_connections': self._usage_stats['active_connections'],
                'max_connections': self.max_connections
            }

    def reset_stats(self):
        """Reset usage statistics."""
        with self._lock:
            self._usage_stats = {
                'requests': 0,
                'failures': 0,
                'total_time': 0,
                'active_connections': self._usage_stats['active_connections']
            }


class ConnectionPoolManager:
    """
    Manages multiple connection pools for different providers.
    """

    def __init__(self):
        self.pools: dict[str, ConnectionPool] = {}
        self._lock = threading.RLock()

    def get_pool(
        self,
        provider: str,
        max_connections: int = 10,
        **kwargs
    ) -> ConnectionPool:
        """
        Get or create a connection pool for a provider.

        Args:
            provider: Provider name
            max_connections: Max connections in pool
            **kwargs: Additional pool configuration

        Returns:
            ConnectionPool instance
        """
        with self._lock:
            if provider not in self.pools:
                self.pools[provider] = ConnectionPool(
                    provider=provider,
                    max_connections=max_connections,
                    **kwargs
                )
            return self.pools[provider]

    async def close_all(self):
        """Close all connection pools."""
        close_tasks = []

        with self._lock:
            for pool in self.pools.values():
                close_tasks.append(pool.close())

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self.pools.clear()
        logger.info("All connection pools closed")

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all connection pools."""
        with self._lock:
            return {
                provider: pool.get_stats()
                for provider, pool in self.pools.items()
            }


# Global connection pool manager
_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> ConnectionPoolManager:
    """Get or create the global connection pool manager."""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager


async def get_connection_pool(
    provider: str,
    max_connections: int = 10,
    **kwargs
) -> ConnectionPool:
    """
    Get a connection pool for the specified provider.

    Args:
        provider: Provider name
        max_connections: Maximum connections
        **kwargs: Additional configuration

    Returns:
        ConnectionPool instance
    """
    manager = get_pool_manager()
    return manager.get_pool(provider, max_connections, **kwargs)


# Cleanup function for graceful shutdown
async def cleanup_connection_pools():
    """Clean up all connection pools on shutdown."""
    global _pool_manager
    if _pool_manager:
        await _pool_manager.close_all()
        _pool_manager = None
