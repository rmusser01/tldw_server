# test_distributed_lock.py
"""Tests for the distributed lock module."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tldw_Server_API.app.core.Infrastructure.distributed_lock as distributed_lock_module
from tldw_Server_API.app.core.Infrastructure.distributed_lock import (
    FileLock,
    LockAcquisitionError,
    RedisLock,
    acquire_migration_lock,
)


# ======================================================================
# FileLock tests
# ======================================================================


class TestFileLockAcquireRelease:
    """Basic acquire / release behaviour."""

    def test_acquire_and_release(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path, timeout=5)

        assert lock.acquire() is True
        assert lock_path.exists()
        assert lock._fd is not None

        lock.release()
        # Lock file is kept so release cannot unlink a new owner.
        assert lock_path.exists()

    def test_acquire_creates_parent_dirs(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "sub" / "dir" / "test.lock"
        lock = FileLock(lock_path, timeout=2)
        assert lock.acquire() is True
        lock.release()

    def test_release_idempotent(self, tmp_path: Path) -> None:
        lock = FileLock(tmp_path / "test.lock", timeout=2)
        lock.acquire()
        lock.release()
        lock.release()  # Should not raise.

    def test_release_leaves_lock_file_to_avoid_reacquire_unlink_race(
        self,
        tmp_path: Path,
    ) -> None:
        lock_path = tmp_path / "release-keeps-file.lock"
        lock = FileLock(lock_path, timeout=2)

        assert lock.acquire() is True
        lock.release()

        assert lock_path.exists()

    def test_release_does_not_unlink_same_process_reacquired_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lock_path = tmp_path / "same-process-reacquire.lock"
        first = FileLock(lock_path, timeout=2)
        second = FileLock(lock_path, timeout=2)
        assert first.acquire() is True
        first_fd = first._fd
        assert first_fd is not None
        reacquired: list[bool] = []
        real_close = distributed_lock_module.os.close

        def _close_and_reacquire(fd: int) -> None:
            real_close(fd)
            if fd == first_fd and not reacquired:
                reacquired.append(second.acquire())

        monkeypatch.setattr(distributed_lock_module.os, "close", _close_and_reacquire)
        try:
            first.release()
            assert reacquired == [True]
            assert lock_path.exists()
        finally:
            second.release()


class TestFileLockContextManager:
    """Context manager protocol."""

    def test_context_manager_acquires_and_releases(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "ctx.lock"
        with FileLock(lock_path, timeout=5) as lock:
            assert lock_path.exists()
            assert isinstance(lock, FileLock)
        assert lock_path.exists()

    def test_context_manager_raises_on_timeout(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "timeout.lock"
        holder = FileLock(lock_path, timeout=5, stale_timeout=9999)
        assert holder.acquire() is True

        try:
            with pytest.raises(LockAcquisitionError):
                # Very short timeout so the test is fast.
                with FileLock(lock_path, timeout=0.3, stale_timeout=9999):
                    pass  # Should never reach here.
        finally:
            holder.release()


class TestFileLockConcurrency:
    """Second lock holder should be blocked."""

    def test_second_lock_times_out(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "concurrent.lock"
        lock1 = FileLock(lock_path, timeout=5)
        assert lock1.acquire() is True

        # A second lock with a short timeout should fail.
        lock2 = FileLock(lock_path, timeout=0.3, stale_timeout=9999)
        assert lock2.acquire() is False

        lock1.release()

        # Now lock2 should succeed.
        assert lock2.acquire() is True
        lock2.release()


class TestFileLockResidualFiles:
    """Residual lock files should not affect native lock ownership."""

    def test_acquires_when_previous_lock_file_has_dead_pid_record(
        self,
        tmp_path: Path,
    ) -> None:
        lock_path = tmp_path / "stale.lock"
        # Write a PID that almost certainly doesn't exist.
        lock_path.write_text("999999999\n")

        lock = FileLock(lock_path, timeout=2, stale_timeout=9999)
        assert lock.acquire() is True
        lock.release()

    def test_acquires_when_previous_lock_file_is_old(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "old.lock"
        lock_path.write_text(f"{os.getpid()}\n")
        # Set mtime far in the past.
        old_time = time.time() - 1000
        os.utime(str(lock_path), (old_time, old_time))

        lock = FileLock(lock_path, timeout=2, stale_timeout=60)
        assert lock.acquire() is True
        lock.release()

    def test_does_not_break_old_lock_when_owner_pid_is_alive(
        self,
        tmp_path: Path,
    ) -> None:
        lock_path = tmp_path / "live-old.lock"
        holder = FileLock(lock_path, timeout=2, stale_timeout=9999)
        assert holder.acquire() is True
        old_time = time.time() - 1000
        os.utime(str(lock_path), (old_time, old_time))

        try:
            contender = FileLock(lock_path, timeout=0.2, stale_timeout=0.01)
            assert contender.acquire() is False
            assert lock_path.exists()
        finally:
            holder.release()

    def test_does_not_unlink_locked_file_even_when_recorded_pid_is_dead(
        self,
        tmp_path: Path,
    ) -> None:
        lock_path = tmp_path / "locked-dead-pid.lock"
        holder = FileLock(lock_path, timeout=2, stale_timeout=9999)
        assert holder.acquire() is True
        lock_path.write_text("999999999\n")

        try:
            contender = FileLock(lock_path, timeout=0.2, stale_timeout=0.01)
            assert contender.acquire() is False
            assert lock_path.exists()
        finally:
            holder.release()


class _FakeMsvcrt:
    LK_NBLCK = 1
    LK_UNLCK = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []

    def locking(self, fd: int, mode: int, length: int) -> None:
        position = os.lseek(fd, 0, os.SEEK_CUR)
        self.calls.append((fd, mode, length, position))


class TestFileLockPlatformSupport:
    """Platform adapter behaviour."""

    def test_windows_fallback_uses_msvcrt_for_acquire_and_release(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_msvcrt = _FakeMsvcrt()
        lock_path = tmp_path / "msvcrt.lock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            monkeypatch.setattr(distributed_lock_module, "_fcntl_mod", None, raising=False)
            monkeypatch.setattr(distributed_lock_module, "_msvcrt_mod", fake_msvcrt, raising=False)

            distributed_lock_module._acquire_platform_file_lock(fd)
            os.lseek(fd, 3, os.SEEK_SET)
            distributed_lock_module._release_platform_file_lock(fd)
        finally:
            os.close(fd)

        assert fake_msvcrt.calls == [
            (fd, fake_msvcrt.LK_NBLCK, 1, 0),
            (fd, fake_msvcrt.LK_UNLCK, 1, 0),
        ]


# ======================================================================
# RedisLock tests (mocked)
# ======================================================================


class _MockRedis:
    """Minimal mock that simulates ``SET ... NX EX`` and ``EVAL``."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.expire_calls: list[tuple[str, int]] = []
        self.close_calls = 0

    def set(self, key: str, value: str, ex: int = 0, nx: bool = False):
        if nx and key in self._store:
            return None
        self._store[key] = value
        return True

    def get(self, key: str):
        return self._store.get(key)

    def expire(self, key: str, ttl: int):
        self.expire_calls.append((key, ttl))
        return key in self._store

    def eval(self, script: str, num_keys: int, *args):
        key = args[0]
        token = args[1]
        if "expire" in script:
            if self._store.get(key) == token:
                self.expire(key, int(args[2]))
                return 1
            return 0
        if self._store.get(key) == token:
            del self._store[key]
            return 1
        return 0

    def ping(self):
        return True

    def close(self):
        self.close_calls += 1


class TestRedisLock:
    """RedisLock with a mock Redis client."""

    def test_acquire_and_release(self) -> None:
        client = _MockRedis()
        lock = RedisLock(client, key="test:lock", timeout=5)

        assert lock.acquire() is True
        assert "test:lock" in client._store

        lock.release()
        assert "test:lock" not in client._store

    def test_context_manager(self) -> None:
        client = _MockRedis()
        with RedisLock(client, key="test:ctx", timeout=5) as lock:
            assert "test:ctx" in client._store
        assert "test:ctx" not in client._store

    def test_second_lock_times_out(self) -> None:
        client = _MockRedis()
        lock1 = RedisLock(client, key="test:dup", timeout=5)
        assert lock1.acquire() is True

        lock2 = RedisLock(client, key="test:dup", timeout=0.2)
        assert lock2.acquire() is False

        lock1.release()

    def test_release_wrong_token_noop(self) -> None:
        client = _MockRedis()
        lock = RedisLock(client, key="test:tok", timeout=5)
        lock.acquire()
        # Simulate another owner.
        client._store["test:tok"] = "someone_else"
        lock.release()
        # Key should still be there (not deleted by our lock).
        assert client._store.get("test:tok") == "someone_else"

    def test_context_manager_raises_on_timeout(self) -> None:
        client = _MockRedis()
        # Pre-fill the key so NX always fails.
        client._store["test:fail"] = "other"

        with pytest.raises(LockAcquisitionError):
            with RedisLock(client, key="test:fail", timeout=0.2):
                pass

    def test_renew_extends_ttl_only_for_current_owner(self) -> None:
        client = _MockRedis()
        lock = RedisLock(client, key="test:renew", timeout=5, ttl=10)
        assert lock.acquire() is True

        assert lock.renew() is True
        assert client.expire_calls == [("test:renew", 10)]

        client._store["test:renew"] = "someone_else"
        assert lock.renew() is False
        assert client.expire_calls == [("test:renew", 10)]


# ======================================================================
# acquire_migration_lock factory tests
# ======================================================================


class TestAcquireMigrationLock:
    """Factory context manager selects the right backend."""

    def test_returns_file_lock_by_default(self, tmp_path: Path) -> None:
        with acquire_migration_lock(
            lock_dir=str(tmp_path), lock_name="test_migration", timeout=5
        ) as lock:
            assert isinstance(lock, FileLock)

    def test_file_lock_uses_default_dir_when_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(distributed_lock_module.Path, "home", lambda: tmp_path)
        with acquire_migration_lock(lock_name="default_dir_test", timeout=5) as lock:
            assert isinstance(lock, FileLock)
            assert ".tldw" in str(lock.path)
        # Clean up.
        try:
            lock.path.unlink(missing_ok=True)
        except OSError:
            pass

    def test_fails_closed_when_configured_redis_is_unavailable(self, tmp_path: Path) -> None:
        with patch(
            "tldw_Server_API.app.core.Infrastructure.distributed_lock._redis_mod"
        ) as mock_redis:
            mock_redis.from_url.side_effect = RuntimeError("redis down")

            with pytest.raises(LockAcquisitionError, match="Redis unavailable"):
                with acquire_migration_lock(
                    lock_dir=str(tmp_path),
                    lock_name="fallback",
                    redis_url="redis://127.0.0.1:1",
                    timeout=5,
                ):
                    pass

    def test_can_opt_in_to_file_fallback_when_redis_unavailable(self, tmp_path: Path) -> None:
        with patch(
            "tldw_Server_API.app.core.Infrastructure.distributed_lock._redis_mod"
        ) as mock_redis:
            mock_redis.from_url.side_effect = RuntimeError("redis down")

            with acquire_migration_lock(
                lock_dir=str(tmp_path),
                lock_name="fallback",
                redis_url="redis://127.0.0.1:1",
                timeout=5,
                allow_file_fallback_on_redis_error=True,
            ) as lock:
                assert isinstance(lock, FileLock)

    def test_uses_redis_when_available(self, tmp_path: Path) -> None:
        mock_client = _MockRedis()
        mock_from_url = MagicMock(return_value=mock_client)

        with patch(
            "tldw_Server_API.app.core.Infrastructure.distributed_lock._redis_mod"
        ) as mock_redis:
            mock_redis.from_url = mock_from_url

            with acquire_migration_lock(
                lock_dir=str(tmp_path),
                lock_name="redis_test",
                redis_url="redis://localhost:6379",
                timeout=5,
            ) as lock:
                assert isinstance(lock, RedisLock)

        assert mock_client.close_calls == 1

    def test_redis_lock_propagates_caller_exception(self, tmp_path: Path) -> None:
        mock_client = _MockRedis()
        sentinel = RuntimeError("migration failed")

        with patch(
            "tldw_Server_API.app.core.Infrastructure.distributed_lock._redis_mod"
        ) as mock_redis:
            mock_redis.from_url.return_value = mock_client

            with pytest.raises(RuntimeError, match="migration failed") as exc_info:
                with acquire_migration_lock(
                    lock_dir=str(tmp_path),
                    lock_name="redis_exception",
                    redis_url="redis://localhost:6379",
                    timeout=5,
                    allow_file_fallback_on_redis_error=True,
                ):
                    raise sentinel

        assert exc_info.value is sentinel
        assert mock_client.close_calls == 1
        assert not (tmp_path / "redis_exception.lock").exists()

    def test_redis_lock_does_not_rerun_caller_block_on_exception(
        self,
        tmp_path: Path,
    ) -> None:
        mock_client = _MockRedis()
        calls = 0

        with patch(
            "tldw_Server_API.app.core.Infrastructure.distributed_lock._redis_mod"
        ) as mock_redis:
            mock_redis.from_url.return_value = mock_client

            with pytest.raises(ValueError, match="stop"):
                with acquire_migration_lock(
                    lock_dir=str(tmp_path),
                    lock_name="redis_no_rerun",
                    redis_url="redis://localhost:6379",
                    timeout=5,
                    allow_file_fallback_on_redis_error=True,
                ):
                    calls += 1
                    raise ValueError("stop")

        assert calls == 1
        assert not (tmp_path / "redis_no_rerun.lock").exists()
