# cpu_bound_handler.py
# Description: Handler for CPU-intensive operations to prevent blocking the event loop
#
# Imports
import asyncio
import atexit
import base64
import functools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

#
# 3rd-Party imports
from loguru import logger

from tldw_Server_API.app.core.Utils.executor_registry import (
    register_executor,
    shutdown_executor_sync,
)
from tldw_Server_API.app.core.testing import env_flag_enabled

#
#######################################################################################################################
#
# Type definitions:

T = TypeVar('T')

#######################################################################################################################
#
# Constants:

# Process/Thread pools
# - Process pool is now lazily created to avoid spawning child processes at import time.
# - It can be disabled via env flags in test environments to prevent semaphore leaks.

# Global handles (created on demand)
CPU_PROCESS_POOL: Optional[ProcessPoolExecutor] = None

# Thread pool for I/O-bound but CPU-heavy operations (created lazily, can be recreated)
CPU_THREAD_POOL: Optional[ThreadPoolExecutor] = None


def _create_cpu_thread_pool() -> ThreadPoolExecutor:
    """Create and register the shared CPU thread pool."""
    pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cpu_worker")
    register_executor("cpu_thread_pool", pool)
    return pool


def get_cpu_thread_pool() -> ThreadPoolExecutor:
    """Return a reusable thread pool for CPU-bound fallback execution."""
    global CPU_THREAD_POOL
    if CPU_THREAD_POOL is None or getattr(CPU_THREAD_POOL, "_shutdown", False):
        CPU_THREAD_POOL = _create_cpu_thread_pool()
    return CPU_THREAD_POOL


def _env_flag_true(name: str) -> bool:
    """Return whether an environment flag is enabled."""
    return env_flag_enabled(name)


def _get_worker_count(default: int = 4) -> int:
    """Read the configured CPU process-pool worker count."""
    try:
        return int(os.getenv("TLDR_CPU_PROCPOOL_WORKERS", default))
    except Exception:
        return default


def _process_pool_disabled() -> bool:
    """Return whether CPU process-pool creation should be skipped."""
    # Disable if explicit flag set or generic TESTING=true
    return _env_flag_true("TLDR_DISABLE_CPU_PROCPOOL") or _env_flag_true("TESTING")


def get_cpu_process_pool() -> Optional[ProcessPoolExecutor]:
    """Lazy initializer for the global CPU process pool.

    Respects TLDR_DISABLE_CPU_PROCPOOL/TESTING env flags. If disabled, returns None.
    """
    global CPU_PROCESS_POOL
    if CPU_PROCESS_POOL is not None:
        return CPU_PROCESS_POOL
    if _process_pool_disabled():
        return None

    workers = _get_worker_count(4)
    if sys.version_info >= (3, 11):
        CPU_PROCESS_POOL = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=100)
    else:
        CPU_PROCESS_POOL = ProcessPoolExecutor(max_workers=workers)
    register_executor("cpu_process_pool", CPU_PROCESS_POOL)
    return CPU_PROCESS_POOL

#######################################################################################################################
#
# Functions:

async def run_cpu_bound(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a CPU-intensive function in a process pool.

    Args:
        func: The CPU-bound function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The function's return value
    """
    loop = asyncio.get_event_loop()

    # Use functools.partial to create a picklable callable
    partial_func = functools.partial(func, *args, **kwargs)

    # Prefer process pool unless disabled; fall back to thread pool in tests
    pool = get_cpu_process_pool()
    executor = pool if pool is not None else get_cpu_thread_pool()

    try:
        result = await loop.run_in_executor(executor, partial_func)
        return result
    except Exception as e:
        logger.error(f"Error in CPU-bound operation: {e}")
        raise


async def run_cpu_bound_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a moderately CPU-intensive function in a thread pool.
    Use this for operations that are CPU-heavy but don't require process isolation.

    Args:
        func: The function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The function's return value
    """
    loop = asyncio.get_event_loop()
    partial_func = functools.partial(func, *args, **kwargs)

    try:
        result = await loop.run_in_executor(
            get_cpu_thread_pool(),
            partial_func,
        )
        return result
    except Exception as e:
        logger.error(f"Error in CPU-bound thread operation: {e}")
        raise


# CPU-intensive operations that should be offloaded

def json_encode_heavy(data: Any) -> str:
    """
    JSON encode large or complex data structures.
    This is CPU-intensive for large payloads.

    Args:
        data: Data to encode

    Returns:
        JSON string
    """
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


def json_decode_heavy(json_str: str) -> Any:
    """
    JSON decode large strings.
    This is CPU-intensive for large payloads.

    Args:
        json_str: JSON string to decode

    Returns:
        Decoded data
    """
    return json.loads(json_str)


def base64_encode_large(data: bytes) -> str:
    """
    Base64 encode large binary data.

    Args:
        data: Binary data to encode

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(data).decode('ascii')


def base64_decode_large(encoded: str) -> bytes:
    """
    Base64 decode large strings.

    Args:
        encoded: Base64 encoded string

    Returns:
        Decoded binary data
    """
    # Remove any whitespace or newlines
    cleaned = ''.join(encoded.split())
    return base64.b64decode(cleaned)


async def process_large_json_async(data: Any) -> str:
    """
    Async wrapper for processing large JSON data.

    Args:
        data: Data to encode as JSON

    Returns:
        JSON string
    """
    def _json_dump(payload: Any) -> str:
        """Serialize JSON compactly while preserving unicode characters."""
        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))

    # For small payloads, process inline
    if isinstance(data, (str, int, float, bool, type(None))):
        return _json_dump(data)

    # For larger payloads, offload to thread pool
    try:
        # Quick size estimation
        if isinstance(data, dict):
            estimated_size = len(str(data))
        elif isinstance(data, list):
            estimated_size = len(data) * 100  # Rough estimate
        else:
            estimated_size = 1000

        if estimated_size < 10000:  # Small payload
            return _json_dump(data)
        else:  # Large payload
            return await run_cpu_bound_thread(json_encode_heavy, data)
    except Exception as e:
        logger.error(f"Error encoding JSON: {e}")
        raise


async def process_large_base64_async(data: bytes) -> str:
    """
    Async wrapper for processing large base64 encoding.

    Args:
        data: Binary data to encode

    Returns:
        Base64 encoded string
    """
    # For small payloads, process inline
    if len(data) < 10000:  # Less than 10KB
        return base64.b64encode(data).decode('ascii')

    # For larger payloads, offload to thread pool
    return await run_cpu_bound_thread(base64_encode_large, data)


async def decode_large_base64_async(encoded: str) -> bytes:
    """
    Async wrapper for decoding large base64 strings.

    Args:
        encoded: Base64 encoded string

    Returns:
        Decoded binary data
    """
    cleaned = ''.join(encoded.split())
    # For small payloads, process inline
    if len(cleaned) < 10000:  # Less than 10KB
        return base64.b64decode(cleaned)

    # For larger payloads, offload to thread pool
    return await run_cpu_bound_thread(base64_decode_large, cleaned)


class CPUBoundBatcher:
    """
    Batch CPU-intensive operations for better efficiency.
    """

    def __init__(self, batch_size: int = 10, timeout: float = 0.1):
        """
        Initialize the batcher.

        Args:
            batch_size: Maximum batch size
            timeout: Maximum time to wait for batch to fill
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_operations = []
        self.results_futures = []
        self._batch_task = None

    async def add_operation(self, func: Callable, *args, **kwargs) -> Any:
        """
        Add an operation to the batch.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            The function result
        """
        future = asyncio.Future()
        self.pending_operations.append((func, args, kwargs, future))

        # Start batch processing if not already running
        if not self._batch_task or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._process_batch())

        # If batch is full, process immediately
        if len(self.pending_operations) >= self.batch_size:
            if self._batch_task and not self._batch_task.done():
                self._batch_task.cancel()
                try:
                    await self._batch_task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._batch_task = None
            await self._process_batch(delay=False)

        return await future

    async def _process_batch(self, *, delay: bool = True):
        """Process the current batch of operations."""
        # Wait for timeout or batch to fill
        if delay:
            await asyncio.sleep(self.timeout)

        if not self.pending_operations:
            return

        # Process all pending operations
        batch = self.pending_operations[:self.batch_size]
        self.pending_operations = self.pending_operations[self.batch_size:]

        # Execute operations in parallel
        tasks = []
        for func, args, kwargs, future in batch:
            task = asyncio.create_task(
                run_cpu_bound_thread(func, *args, **kwargs)
            )
            tasks.append((task, future))

        try:
            # Wait for all to complete
            for task, future in tasks:
                try:
                    result = await task
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
        finally:
            current_task = asyncio.current_task()
            if self._batch_task is current_task or (self._batch_task is not None and self._batch_task.done()):
                self._batch_task = None
            if self.pending_operations and (self._batch_task is None or self._batch_task.done()):
                delay_next = len(self.pending_operations) < self.batch_size
                self._batch_task = asyncio.create_task(self._process_batch(delay=delay_next))


# Global batcher instance
_json_batcher = CPUBoundBatcher()

async def batch_json_encode(data: Any) -> str:
    """
    Batch JSON encoding operations for efficiency.

    Args:
        data: Data to encode

    Returns:
        JSON string
    """
    return await _json_batcher.add_operation(json_encode_heavy, data)


def cleanup_pools():
    """Cleanup process and thread pools."""
    global CPU_PROCESS_POOL, CPU_THREAD_POOL
    try:
        shutdown_executor_sync("cpu_process_pool", wait=True, cancel_futures=True)
    finally:
        CPU_PROCESS_POOL = None
    try:
        shutdown_executor_sync("cpu_thread_pool", wait=True, cancel_futures=True)
    finally:
        CPU_THREAD_POOL = None


# Ensure pools are cleaned up on interpreter shutdown
atexit.register(cleanup_pools)
