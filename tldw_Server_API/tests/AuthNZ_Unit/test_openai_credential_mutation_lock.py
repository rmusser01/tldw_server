from __future__ import annotations

import asyncio
import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from loguru import logger


class _AdvisoryConnection:
    async def fetchval(self, query: str, *_args):
        if "pg_try_advisory_lock" in query:
            return True
        if "pg_advisory_unlock" in query:
            return True
        raise AssertionError(f"unexpected query: {query}")


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_distinct_user_locks_do_not_consume_main_authnz_pool(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    main_capacity = asyncio.Semaphore(1)
    lock_capacity = asyncio.Semaphore(4)
    all_holders_entered = asyncio.Event()
    release_holders = asyncio.Event()
    holder_count = 0
    main_acquisitions = 0
    lock_acquisitions = 0

    class _FakePool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire(self, *, timeout=None):
            nonlocal main_acquisitions
            await asyncio.wait_for(main_capacity.acquire(), timeout=timeout)
            main_acquisitions += 1
            try:
                yield object()
            finally:
                main_capacity.release()

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            nonlocal lock_acquisitions
            await asyncio.wait_for(lock_capacity.acquire(), timeout=timeout)
            lock_acquisitions += 1
            try:
                yield _AdvisoryConnection()
            finally:
                lock_capacity.release()

    pool = _FakePool()

    async def _get_db_pool():
        return pool

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    async def _hold(user_id: int) -> None:
        nonlocal holder_count
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=user_id,
            provider="openai",
        ) as locked_repo:
            assert isinstance(locked_repo, AuthnzUserProviderSecretsRepo)
            holder_count += 1
            if holder_count == 4:
                all_holders_entered.set()
            await release_holders.wait()

    holders = [asyncio.create_task(_hold(user_id)) for user_id in range(1, 5)]
    try:
        await asyncio.wait_for(all_holders_entered.wait(), timeout=1)
        async with asyncio.timeout(0.1):
            async with pool.acquire():
                pass
    finally:
        release_holders.set()
        await asyncio.gather(*holders, return_exceptions=True)

    assert main_acquisitions == 1
    assert lock_acquisitions == 4


@pytest.mark.asyncio
async def test_public_postgres_lock_fails_closed_without_dedicated_pool_api(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    main_pool_borrowed = False

    class _MainOnlyPool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire(self, *, timeout=None):
            nonlocal main_pool_borrowed
            main_pool_borrowed = True
            yield _AdvisoryConnection()

    async def _get_db_pool():
        return _MainOnlyPool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=11,
            provider="openai",
        ):
            raise AssertionError("missing dedicated pool entered protected body")

    assert exc_info.value.code == "credential_store_unavailable"
    assert not main_pool_borrowed


@pytest.mark.asyncio
async def test_bound_postgres_repo_can_revoke_without_borrowing_main_pool(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    execute_calls: list[tuple[str, tuple[object, ...]]] = []

    class _Connection:
        @contextlib.asynccontextmanager
        async def transaction(self):
            yield

        async def fetchval(self, query: str, *_args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                return True
            raise AssertionError(f"unexpected advisory query: {query}")

        async def fetchrow(self, query: str, *_args):
            assert "FROM user_provider_secrets" in query
            return {
                "id": 1,
                "user_id": 17,
                "provider": "openai",
                "encrypted_blob": "encrypted",
                "key_hint": "hint",
                "metadata": None,
                "revoked_at": None,
            }

        async def fetch(self, query: str, *_args):
            assert "FROM user_provider_secrets" in query
            return [{"provider": "openai", "revoked_at": None}]

        async def execute(self, query: str, *args):
            execute_calls.append((query, args))
            return "SELECT 1" if "pg_advisory_xact_lock" in query else "UPDATE 1"

    class _Pool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            yield _Connection()

        @contextlib.asynccontextmanager
        async def acquire(self, *, timeout=None):
            raise AssertionError("bound mutation borrowed the main AuthNZ pool")
            yield  # pragma: no cover

    async def _get_db_pool():
        return _Pool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    async with byok_runtime.openai_credential_mutation_lock(
        user_id=17,
    ) as locked_repo:
        assert locked_repo is not None
        revoked = await locked_repo.delete_secret(
            17,
            "openai",
            revoked_by=17,
        )

    assert revoked
    identity_locks = [
        call for call in execute_calls if "pg_advisory_xact_lock" in call[0]
    ]
    mutations = [
        call for call in execute_calls if "UPDATE user_provider_secrets" in call[0]
    ]
    assert len(identity_locks) == 1
    assert len(mutations) == 1
    query, args = mutations[0]
    assert query.count("revoked_at IS NULL") == 1
    assert args[-2:] == (17, "openai")


@pytest.mark.asyncio
async def test_bound_postgres_repo_cas_query_has_one_active_blob_predicate(monkeypatch):
    from datetime import datetime, timezone

    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    cas_queries: list[str] = []

    class _Connection:
        @contextlib.asynccontextmanager
        async def transaction(self):
            yield

        async def fetchval(self, query: str, *_args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                return True
            raise AssertionError(f"unexpected advisory query: {query}")

        async def fetch(self, query: str, *_args):
            assert "FROM user_provider_secrets" in query
            return [{"provider": "openai", "revoked_at": None}]

        async def execute(self, query: str, *_args):
            if "pg_advisory_xact_lock" in query:
                return "SELECT 1"
            assert "DELETE FROM user_provider_secrets" in query
            return "DELETE 0"

        async def fetchrow(self, query: str, *_args):
            cas_queries.append(query)
            return {"id": 1}

    class _Pool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            yield _Connection()

    async def _get_db_pool():
        return _Pool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    async with byok_runtime.openai_credential_mutation_lock(
        user_id=18,
    ) as locked_repo:
        assert locked_repo is not None
        updated = await locked_repo.update_secret_if_active_and_unchanged(
            user_id=18,
            provider="openai",
            encrypted_blob="next",
            expected_encrypted_blob="prior",
            key_hint="hint",
            metadata=None,
            updated_at=datetime.now(timezone.utc),
            updated_by=18,
        )

    assert updated
    assert len(cas_queries) == 1
    assert cas_queries[0].count("revoked_at IS NULL") == 1
    assert cas_queries[0].count("encrypted_blob = $8") == 1


@pytest.mark.asyncio
async def test_private_refresh_lock_delegates_to_public_mutation_lock(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    sentinel = object()
    calls: list[dict[str, object]] = []

    @contextlib.asynccontextmanager
    async def _mutation_lock(**kwargs):
        calls.append(kwargs)
        yield sentinel

    monkeypatch.setattr(
        byok_runtime,
        "openai_credential_mutation_lock",
        _mutation_lock,
        raising=False,
    )
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "memory")

    async with byok_runtime._openai_oauth_refresh_lock(
        user_id=12,
        provider="openai",
    ) as locked_repo:
        assert locked_repo is sentinel

    assert calls == [{"user_id": 12, "provider": "openai"}]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_openai_alias_and_case_variants_contend_on_one_lock(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "memory")
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_attempted = asyncio.Event()
    second_entered = asyncio.Event()

    async def _first() -> None:
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=13,
            provider="OAI",
        ):
            first_entered.set()
            await release_first.wait()

    async def _second() -> None:
        await first_entered.wait()
        second_attempted.set()
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=13,
            provider="OpenAI",
        ):
            second_entered.set()

    first = asyncio.create_task(_first())
    second = asyncio.create_task(_second())
    await first_entered.wait()
    await asyncio.wait_for(second_attempted.wait(), timeout=1.0)
    assert not second_entered.is_set()
    release_first.set()
    await asyncio.gather(first, second)
    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_public_mutation_lock_rejects_non_openai_provider(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "memory")

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=14,
            provider="anthropic",
        ):
            raise AssertionError("non-OpenAI provider entered OpenAI lock")

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
@pytest.mark.parametrize("unlock_result", [False, None])
async def test_postgres_final_unlock_requires_positive_confirmation(
    monkeypatch,
    unlock_result,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")

    class _Connection:
        async def fetchval(self, query: str, *_args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                return unlock_result
            raise AssertionError(f"unexpected query: {query}")

    class _Pool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            yield _Connection()

    async def _get_db_pool():
        return _Pool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        async with byok_runtime.openai_credential_mutation_lock(user_id=15):
            pass

    assert exc_info.value.code == "credential_store_unavailable"


@pytest.mark.asyncio
async def test_postgres_unlock_error_does_not_mask_body_exception(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")

    class _Connection:
        async def fetchval(self, query: str, *_args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                raise RuntimeError("unlock transport canary")
            raise AssertionError(f"unexpected query: {query}")

    class _Pool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            yield _Connection()

    async def _get_db_pool():
        return _Pool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    with pytest.raises(ValueError, match="body failure canary"):
        async with byok_runtime.openai_credential_mutation_lock(user_id=16):
            raise ValueError("body failure canary")


def test_public_oauth_generation_is_opaque_access_token_only_and_log_safe():
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    secret = "access-token-log-canary"

    def _payload(access_token: str, refresh_token: str, subject: str) -> dict:
        return {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "subject": subject,
                }
            },
        }

    first = byok_runtime.openai_oauth_credential_generation(
        _payload(secret, "refresh-a", "subject-a")
    )
    metadata_only_change = byok_runtime.openai_oauth_credential_generation(
        _payload(secret, "refresh-b", "subject-b")
    )
    next_generation = byok_runtime.openai_oauth_credential_generation(
        _payload("access-token-next", "refresh-b", "subject-b")
    )

    assert first is not None
    assert len(first) == 64
    assert first == metadata_only_change
    assert first != next_generation
    assert first != secret
    assert secret not in repr(first)
    assert byok_runtime._openai_oauth_generation(
        _payload(secret, "refresh-c", "subject-c")
    ) == first

    log_messages: list[str] = []
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        logger.info("credential_generation={}", first)
    finally:
        logger.remove(sink_id)
    assert secret not in "".join(log_messages)


@pytest.mark.asyncio
async def test_sqlite_file_lock_creates_configured_directory(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    lock_dir = tmp_path / "nested" / "locks"
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_DIR", str(lock_dir))

    class _SqlitePool:
        backend_type = "sqlite"

    async def _get_db_pool():
        return _SqlitePool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    async with byok_runtime.openai_credential_mutation_lock(
        user_id=20,
        provider="openai",
    ):
        assert lock_dir.is_dir()


def test_sqlite_file_lock_serializes_independent_event_loops(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_DIR", str(tmp_path))
    state_lock = threading.Lock()
    start_barrier = threading.Barrier(2)
    active = 0
    max_active = 0

    class _SqlitePool:
        backend_type = "sqlite"

    async def _get_db_pool():
        return _SqlitePool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    async def _worker() -> None:
        nonlocal active, max_active
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=21,
            provider="openai",
        ):
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            with state_lock:
                active -= 1

    def _run_worker() -> None:
        start_barrier.wait(timeout=5)
        asyncio.run(_worker())

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run_worker) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)

    assert max_active == 1


@pytest.mark.asyncio
async def test_sqlite_file_lock_times_out_while_owner_holds(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(byok_runtime, "OPENAI_OAUTH_REFRESH_LOCK_TIMEOUT_SECONDS", 0.05)

    class _SqlitePool:
        backend_type = "sqlite"

    async def _get_db_pool():
        return _SqlitePool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)
    owner_entered = asyncio.Event()
    release_owner = asyncio.Event()

    async def _owner() -> None:
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=22,
            provider="openai",
        ):
            owner_entered.set()
            await release_owner.wait()

    owner = asyncio.create_task(_owner())
    await owner_entered.wait()
    try:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            async with byok_runtime.openai_credential_mutation_lock(
                user_id=22,
                provider="openai",
            ):
                raise AssertionError("timed-out waiter entered protected body")
        assert exc_info.value.code == "credential_store_unavailable"
    finally:
        release_owner.set()
        await owner


@pytest.mark.asyncio
async def test_sqlite_file_lock_preserves_protected_body_exception(monkeypatch, tmp_path):
    """The lock boundary must not rewrite endpoint or repository failures."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_DIR", str(tmp_path))

    class _SqlitePool:
        backend_type = "sqlite"

    async def _get_db_pool():
        return _SqlitePool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    with pytest.raises(ValueError, match="protected body canary"):
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=26,
            provider="openai",
        ):
            raise ValueError("protected body canary")

    async with byok_runtime.openai_credential_mutation_lock(
        user_id=26,
        provider="openai",
    ):
        pass


@pytest.mark.asyncio
async def test_sqlite_file_lock_owner_cancellation_releases_lock(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_DIR", str(tmp_path))

    class _SqlitePool:
        backend_type = "sqlite"

    async def _get_db_pool():
        return _SqlitePool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)
    owner_entered = asyncio.Event()

    async def _owner() -> None:
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=23,
            provider="openai",
        ):
            owner_entered.set()
            await asyncio.Event().wait()

    owner = asyncio.create_task(_owner())
    await owner_entered.wait()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner

    async with asyncio.timeout(0.2):
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=23,
            provider="openai",
        ):
            pass


@pytest.mark.asyncio
async def test_cancelled_sqlite_waiter_cannot_release_owner_lock(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(byok_runtime, "OPENAI_OAUTH_REFRESH_LOCK_TIMEOUT_SECONDS", 0.05)

    class _SqlitePool:
        backend_type = "sqlite"

    async def _get_db_pool():
        return _SqlitePool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)
    owner_entered = asyncio.Event()
    release_owner = asyncio.Event()

    async def _owner() -> None:
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=24,
            provider="openai",
        ):
            owner_entered.set()
            await release_owner.wait()

    async def _waiter() -> None:
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=24,
            provider="openai",
        ):
            raise AssertionError("cancelled waiter entered protected body")

    owner = asyncio.create_task(_owner())
    await owner_entered.wait()
    waiter = asyncio.create_task(_waiter())
    await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    try:
        with pytest.raises(byok_runtime.ByokResolutionError):
            async with byok_runtime.openai_credential_mutation_lock(
                user_id=24,
                provider="openai",
            ):
                raise AssertionError("owner lock was released by cancelled waiter")
    finally:
        release_owner.set()
        await owner

    async with byok_runtime.openai_credential_mutation_lock(
        user_id=24,
        provider="openai",
    ):
        pass


@pytest.mark.asyncio
async def test_explicit_redis_without_url_fails_closed_without_fallback(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    monkeypatch.setattr(
        byok_runtime,
        "get_settings",
        lambda: SimpleNamespace(REDIS_URL=None),
    )
    monkeypatch.setattr(
        byok_runtime,
        "_get_openai_refresh_lock",
        lambda _key: (_ for _ in ()).throw(AssertionError("memory fallback used")),
    )

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=25,
            provider="openai",
        ):
            raise AssertionError("missing Redis URL entered protected body")

    assert exc_info.value.code == "credential_store_unavailable"


@pytest.mark.asyncio
async def test_redis_mutation_lock_awaits_close_only_client_cleanup(monkeypatch):
    """redis-py clients predating ``aclose`` still close after token release."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    monkeypatch.setattr(
        byok_runtime,
        "OPENAI_OAUTH_REFRESH_LOCK_RENEW_INTERVAL_SECONDS",
        3600,
    )
    cleanup_events: list[str] = []

    class _CloseOnlyRedis:
        async def set(self, _key, _token, **_kwargs):
            return True

        async def eval(self, script, _count, _key, _token, *_args):
            if "expire" in script:
                return 1
            cleanup_events.append("release")
            return 1

        async def close(self):
            await asyncio.sleep(0)
            cleanup_events.append("close")

    monkeypatch.setattr(
        byok_runtime,
        "_openai_oauth_redis_client",
        _CloseOnlyRedis,
    )

    async with byok_runtime.openai_credential_mutation_lock(user_id=27):
        pass

    assert cleanup_events == ["release", "close"]
