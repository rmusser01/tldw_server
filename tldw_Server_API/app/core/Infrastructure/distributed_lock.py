# distributed_lock.py
# Description: Distributed lock module for cross-process migration coordination.
#
# Provides:
#   - FileLock: cross-process file-based lock using fcntl
#   - RedisLock: Redis-based distributed lock using SET NX EX
#   - acquire_migration_lock(): context manager factory that picks the best backend
#
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from loguru import logger

try:  # pragma: no cover - platform guard
    import fcntl as _fcntl_mod  # type: ignore
except ImportError:  # pragma: no cover
    _fcntl_mod = None  # type: ignore[assignment]

try:  # pragma: no cover - platform guard
    import msvcrt as _msvcrt_mod  # type: ignore
except ImportError:  # pragma: no cover
    _msvcrt_mod = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    import redis as _redis_mod  # type: ignore
except ImportError:  # pragma: no cover
    _redis_mod = None  # type: ignore[assignment]


class LockAcquisitionError(RuntimeError):
    """Raised when a distributed lock cannot be acquired within the timeout."""


def _acquire_platform_file_lock(fd: int) -> None:
    """Acquire a non-blocking exclusive lock for *fd* on the current platform."""
    if _fcntl_mod is not None:
        _fcntl_mod.flock(fd, _fcntl_mod.LOCK_EX | _fcntl_mod.LOCK_NB)
        return

    if _msvcrt_mod is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        _msvcrt_mod.locking(fd, _msvcrt_mod.LK_NBLCK, 1)
        return

    raise OSError("No supported file locking backend is available on this platform")


def _release_platform_file_lock(fd: int) -> None:
    """Release the platform-specific lock for *fd*."""
    if _fcntl_mod is not None:
        _fcntl_mod.flock(fd, _fcntl_mod.LOCK_UN)
        return

    if _msvcrt_mod is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        _msvcrt_mod.locking(fd, _msvcrt_mod.LK_UNLCK, 1)
        return

    raise OSError("No supported file locking backend is available on this platform")


class FileLock:
    """Cross-process file-based lock using the native platform backend.

    Parameters:
        path: Path to the lock file.
        timeout: Maximum seconds to wait for lock acquisition.
        stale_timeout: Retained for API compatibility. Native file locks are
            expected to be released by the platform when the owner exits.
    """

    def __init__(
        self,
        path: str | Path,
        timeout: float = 60,
        stale_timeout: float = 300,
    ) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.stale_timeout = stale_timeout
        self._fd: Optional[int] = None
        self._owner_token: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> bool:
        """Try to acquire the lock, retrying until *timeout* expires.

        Returns ``True`` on success, ``False`` if the timeout is reached.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        attempt = 0

        while True:
            attempt += 1
            fd: Optional[int] = None
            try:
                fd = os.open(
                    str(self.path),
                    os.O_CREAT | os.O_RDWR,
                    0o644,
                )
                _acquire_platform_file_lock(fd)
                # Include the PID for stale detection and a per-acquire token so
                # same-process reacquisition cannot be mistaken for this owner.
                owner_token = f"{os.getpid()}:{uuid.uuid4().hex}"
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, f"{owner_token}\n".encode())
                os.fsync(fd)
                self._fd = fd
                self._owner_token = owner_token
                logger.debug("FileLock acquired: {}", self.path)
                return True
            except OSError:
                # Could not lock — close fd and retry.
                try:
                    if fd is not None:
                        os.close(fd)
                except OSError:
                    pass

                if time.monotonic() >= deadline:
                    return False

                time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def release(self) -> None:
        """Release the lock.

        The lock file is intentionally left in place. Removing it during normal
        release creates a race where another owner can acquire the path and then
        have its active lock file unlinked by the previous releaser.
        """
        if self._fd is not None:
            try:
                _release_platform_file_lock(self._fd)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        self._owner_token = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "FileLock":
        if not self.acquire():
            raise LockAcquisitionError(
                f"Failed to acquire file lock {self.path} "
                f"within {self.timeout}s"
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


class RedisLock:
    """Redis-based distributed lock using ``SET key NX EX``.

    Uses a unique token per instance and a Lua script for safe release
    (only deletes the key if our token still owns it).

    Parameters:
        redis_client: A synchronous ``redis.Redis``-compatible client.
        key: Redis key to use for the lock.
        timeout: Maximum seconds to wait for lock acquisition.
        ttl: Seconds until the key auto-expires (prevents deadlock on crash).
    """

    _RELEASE_LUA = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """
    _RENEW_LUA = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("expire", KEYS[1], ARGV[2])
    else
        return 0
    end
    """

    def __init__(
        self,
        redis_client: Any,
        key: str = "tldw:migration:lock",
        timeout: float = 60,
        ttl: int = 300,
    ) -> None:
        self._client = redis_client
        self.key = key
        self.timeout = timeout
        self.ttl = ttl
        self._token: str = uuid.uuid4().hex

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> bool:
        """Try ``SET key NX EX`` in a retry loop until *timeout*."""
        deadline = time.monotonic() + self.timeout

        while True:
            result = self._client.set(
                self.key,
                self._token,
                ex=self.ttl,
                nx=True,
            )
            if result:
                logger.debug("RedisLock acquired: {}", self.key)
                return True

            if time.monotonic() >= deadline:
                return False

            time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def release(self) -> None:
        """Atomically release the lock only if our token still owns it."""
        try:
            self._client.eval(
                self._RELEASE_LUA,
                1,
                self.key,
                self._token,
            )
        except Exception as exc:
            logger.debug("RedisLock release failed (non-critical): {}", exc)

    def renew(self) -> bool:
        """Extend the lock TTL only if this instance still owns the key."""
        try:
            renewed = self._client.eval(
                self._RENEW_LUA,
                1,
                self.key,
                self._token,
                int(self.ttl),
            )
            return bool(renewed)
        except Exception as exc:
            logger.debug("RedisLock renew failed: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RedisLock":
        if not self.acquire():
            raise LockAcquisitionError(
                f"Failed to acquire Redis lock '{self.key}' "
                f"within {self.timeout}s"
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

@contextmanager
def acquire_migration_lock(
    *,
    lock_dir: Optional[str | Path] = None,
    lock_name: str = "db_migration",
    redis_url: Optional[str] = None,
    timeout: float = 60,
    allow_file_fallback_on_redis_error: bool = False,
) -> Generator[FileLock | RedisLock, None, None]:
    """Context manager that picks the best lock backend.

    If *redis_url* is provided, a :class:`RedisLock` is used.  Redis
    configuration errors fail closed unless
    ``allow_file_fallback_on_redis_error`` is explicitly enabled.  Without
    *redis_url*, a local :class:`FileLock` is used.

    Parameters:
        lock_dir: Directory for file-based locks.  Defaults to ``~/.tldw/locks/``.
        lock_name: Base name for the lock (file or Redis key).
        redis_url: Optional Redis connection URL.
        timeout: Maximum seconds to wait for the lock.
        allow_file_fallback_on_redis_error: When true, fall back to a local
            file lock if an explicitly configured Redis lock cannot be used.
    """
    # Try Redis first.
    if redis_url:
        if _redis_mod is None:
            if not allow_file_fallback_on_redis_error:
                raise LockAcquisitionError(
                    "Redis unavailable for migration lock: redis package is not installed"
                )
            logger.debug(
                "Redis package unavailable for migration lock; falling back to file lock",
            )
        else:
            client = None
            lock = None
            try:
                client = _redis_mod.from_url(redis_url, decode_responses=True)
                client.ping()
                key = f"tldw:{lock_name}:lock"
                lock = RedisLock(client, key=key, timeout=timeout)
            except Exception as exc:
                if client is not None:
                    try:
                        client.close()
                    except Exception as close_exc:
                        logger.debug(
                            "Failed to close Redis migration lock client: {}",
                            type(close_exc).__name__,
                        )
                    client = None
                if not allow_file_fallback_on_redis_error:
                    raise LockAcquisitionError(
                        "Redis unavailable for migration lock; refusing local file fallback"
                    ) from exc
                logger.debug(
                    "Redis unavailable for migration lock ({}); falling back to file lock",
                    type(exc).__name__,
                )

            if client is not None and lock is not None:
                try:
                    with lock:
                        yield lock
                    return
                finally:
                    try:
                        client.close()
                    except Exception as close_exc:
                        logger.debug(
                            "Failed to close Redis migration lock client: {}",
                            type(close_exc).__name__,
                        )

    # Fall back to FileLock.
    if lock_dir is None:
        lock_dir = Path.home() / ".tldw" / "locks"
    lock_path = Path(lock_dir) / f"{lock_name}.lock"
    lock = FileLock(lock_path, timeout=timeout)
    with lock:
        yield lock


__all__ = [
    "LockAcquisitionError",
    "FileLock",
    "RedisLock",
    "acquire_migration_lock",
]
