from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ import database as authnz_database
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.repos.llm_provider_overrides_repo import (
    AuthnzLLMProviderOverridesRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.rate_limits_repo import AuthnzRateLimitsRepo
from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)

pytestmark = pytest.mark.unit

_SECRET = "sk-live-authnz-secret-token"
_PATH = "/tmp/authnz-secret-token"
_DSN = "postgresql://authnz:database-password@db.internal:5432/authnz"
_LEAK = f"authnz backend exploded with {_SECRET} at {_PATH} using {_DSN}"


def _capture_logs() -> tuple[list[str], int]:
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    return records, sink_id


def _assert_safe_log(rendered: str) -> None:
    assert "authnz backend exploded" not in rendered
    assert _SECRET not in rendered
    assert _PATH not in rendered
    assert _DSN not in rendered
    assert "exc_info" not in rendered


class _DictFallbackRow:
    def __init__(self) -> None:
        self._keys_calls = 0

    def keys(self):
        self._keys_calls += 1
        if self._keys_calls == 1:
            raise RuntimeError(_LEAK)
        return ["provider", "key_hint"]

    def __getitem__(self, key: str) -> Any:
        return {"provider": "openai", "key_hint": "1234"}[key]

    def __iter__(self):
        return iter((("provider", "openai"), ("key_hint", "1234")))


class _FlakyMappingRow:
    def __init__(self) -> None:
        self._keys_calls = 0

    def keys(self):
        self._keys_calls += 1
        if self._keys_calls == 1:
            raise RuntimeError(_LEAK)
        return ["provider", "is_enabled"]

    def __getitem__(self, key: str) -> Any:
        return {"provider": "openai", "is_enabled": True}[key]


def test_user_provider_secret_row_key_fallback_log_omits_raw_exception() -> None:
    records, sink_id = _capture_logs()
    try:
        row = AuthnzUserProviderSecretsRepo._row_to_dict(_DictFallbackRow())
    finally:
        logger.remove(sink_id)

    assert row == {"provider": "openai", "key_hint": "1234"}
    _assert_safe_log("\n".join(records))


def test_llm_provider_override_row_cast_fallback_log_omits_raw_exception() -> None:
    records, sink_id = _capture_logs()
    try:
        row = AuthnzLLMProviderOverridesRepo._row_to_dict(_FlakyMappingRow())
    finally:
        logger.remove(sink_id)

    assert row == {"provider": "openai", "is_enabled": True}
    _assert_safe_log("\n".join(records))


class _FailingRbacBackend:
    def execute(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(_LEAK)


class _FailingRbacDatabase:
    backend = _FailingRbacBackend()

    def get_user_permissions(self, _user_id: int) -> list[str]:
        raise RuntimeError(_LEAK)


@pytest.mark.parametrize(
    "operation",
    ["get_user_roles", "get_effective_permissions"],
)
def test_rbac_repo_failure_logs_omit_raw_exception(operation: str) -> None:
    """Worker RBAC failures log bounded type metadata and still propagate."""

    repo = AuthnzRbacRepo(client_id="test-rbac-log-sanitizer")
    repo.__dict__["_db"] = _FailingRbacDatabase()

    records, sink_id = _capture_logs()
    try:
        with pytest.raises(RuntimeError, match="authnz backend exploded"):
            getattr(repo, operation)(7)
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert f"AuthnzRbacRepo.{operation} failed" in rendered
    assert "RuntimeError" in rendered
    _assert_safe_log(rendered)


class _FailingOverridePool:
    pool = None
    db_path = "unused"
    _initialized = True

    async def fetchone(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(_LEAK)

    async def fetchall(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        raise RuntimeError(_LEAK)

    async def execute(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(_LEAK)

    @asynccontextmanager
    async def transaction(self):
        raise RuntimeError(_LEAK)
        yield


class _FailingOrgSecretPool(_FailingOverridePool):
    @asynccontextmanager
    async def transaction(self):
        raise TransactionError("org provider secret write", _LEAK)
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["ensure_tables", "list_overrides", "fetch_override", "upsert_override", "patch_override", "delete_override"],
)
async def test_llm_provider_override_repo_failure_logs_omit_raw_exception(
    operation: str,
) -> None:
    repo = AuthnzLLMProviderOverridesRepo(_FailingOverridePool())  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)

    async def invoke() -> Any:
        if operation == "ensure_tables":
            return await repo.ensure_tables()
        if operation == "list_overrides":
            return await repo.list_overrides()
        if operation == "fetch_override":
            return await repo.fetch_override("openai")
        if operation == "upsert_override":
            return await repo.upsert_override(
                provider="openai",
                is_enabled=True,
                allowed_models=None,
                config_json=None,
                secret_blob=None,
                api_key_hint=None,
                updated_at=now,
            )
        if operation == "patch_override":
            return await repo.patch_override(
                provider="openai",
                fields={"is_enabled": True},
                updated_at=now,
            )
        return await repo.delete_override("openai")

    records, sink_id = _capture_logs()
    try:
        with pytest.raises(RuntimeError, match="authnz backend exploded"):
            await invoke()
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert f"AuthnzLLMProviderOverridesRepo.{operation} failed" in rendered
    _assert_safe_log(rendered)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "ensure_tables",
        "upsert_secret",
        "fetch_secret_for_user",
        "update_secret_if_active_and_unchanged",
        "list_secrets_for_user",
        "delete_secret",
        "touch_last_used",
    ],
)
async def test_user_provider_secret_repo_failure_logs_omit_raw_exception(
    operation: str,
) -> None:
    """Secret-repository failures retain only bounded type metadata in logs."""
    repo = AuthnzUserProviderSecretsRepo(_FailingOverridePool())  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)

    async def invoke() -> Any:
        if operation == "ensure_tables":
            return await repo.ensure_tables()
        if operation == "upsert_secret":
            return await repo.upsert_secret(
                user_id=7,
                provider="openai",
                encrypted_blob="encrypted-sentinel",
                key_hint="hint",
                metadata=None,
                updated_at=now,
            )
        if operation == "fetch_secret_for_user":
            return await repo.fetch_secret_for_user(7, "openai")
        if operation == "update_secret_if_active_and_unchanged":
            return await repo.update_secret_if_active_and_unchanged(
                user_id=7,
                provider="openai",
                encrypted_blob="new-encrypted-sentinel",
                expected_encrypted_blob="old-encrypted-sentinel",
                key_hint="hint",
                metadata=None,
                updated_at=now,
            )
        if operation == "list_secrets_for_user":
            return await repo.list_secrets_for_user(7)
        if operation == "delete_secret":
            return await repo.delete_secret(7, "openai")
        return await repo.touch_last_used(7, "openai", now)

    records, sink_id = _capture_logs()
    try:
        with pytest.raises(RuntimeError, match="authnz backend exploded"):
            await invoke()
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert f"AuthnzUserProviderSecretsRepo.{operation} failed" in rendered
    _assert_safe_log(rendered)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "ensure_tables",
        "upsert_secret",
        "fetch_secret",
        "fetch_authorized_secret_for_user",
        "list_secrets",
        "delete_secret",
        "touch_last_used",
    ],
)
async def test_org_provider_secret_repo_failure_logs_omit_raw_exception(
    operation: str,
) -> None:
    """Every public org-secret operation logs only its name and error type."""
    repo = AuthnzOrgProviderSecretsRepo(_FailingOrgSecretPool())  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)

    async def invoke() -> Any:
        if operation == "ensure_tables":
            return await repo.ensure_tables()
        if operation == "upsert_secret":
            return await repo.upsert_secret(
                scope_type="org",
                scope_id=7,
                provider="openai",
                encrypted_blob="encrypted-sentinel",
                key_hint="hint",
                metadata=None,
                updated_at=now,
            )
        if operation == "fetch_secret":
            return await repo.fetch_secret("org", 7, "openai")
        if operation == "fetch_authorized_secret_for_user":
            return await repo.fetch_authorized_secret_for_user("org", 7, 9, "openai")
        if operation == "list_secrets":
            return await repo.list_secrets(scope_type="org", scope_id=7)
        if operation == "delete_secret":
            return await repo.delete_secret("org", 7, "openai")
        return await repo.touch_last_used("org", 7, "openai", now)

    expected_error_type = TransactionError if operation in {"upsert_secret", "delete_secret"} else RuntimeError
    records, sink_id = _capture_logs()
    try:
        with pytest.raises(expected_error_type) as exc_info:
            await invoke()
    finally:
        logger.remove(sink_id)

    assert _LEAK in str(exc_info.value)
    rendered = "\n".join(records)
    assert f"AuthnzOrgProviderSecretsRepo.{operation} failed" in rendered
    assert expected_error_type.__name__ in rendered
    _assert_safe_log(rendered)


class _Tx:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _FetchAllCursor:
    async def fetchall(self) -> list[Any]:
        return []


class _CommitFailConn:
    async def execute(self, query: str, *params: Any) -> _FetchAllCursor:  # noqa: ARG002
        return _FetchAllCursor()

    async def commit(self) -> None:
        raise RuntimeError(_LEAK)


class _RateLimitPool:
    pool = None

    def transaction(self) -> _Tx:
        return _Tx(_CommitFailConn())


@pytest.mark.asyncio
async def test_rate_limits_explicit_commit_fallback_log_omits_raw_exception() -> None:
    repo = AuthnzRateLimitsRepo(db_pool=_RateLimitPool())  # type: ignore[arg-type]

    records, sink_id = _capture_logs()
    try:
        await repo.ensure_schema()
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))


class _PostgresTransactionConn:
    def transaction(self) -> _Tx:
        return _Tx(self)

    async def execute(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(_LEAK)


class _PostgresTransactionPool:
    def __init__(self) -> None:
        self.conn = _PostgresTransactionConn()

    def acquire(self) -> _Tx:
        return _Tx(self.conn)


class _SQLiteTransactionConn:
    def __init__(self) -> None:
        self.row_factory: Any = None
        self.rolled_back = False
        self.closed = False

    async def execute(self, query: str, *_args: Any, **_kwargs: Any) -> None:
        if query != "BEGIN IMMEDIATE":
            raise RuntimeError(_LEAK)

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        self.closed = True


def _transaction_pool(backend: Any) -> DatabasePool:
    db_pool = DatabasePool.__new__(DatabasePool)
    db_pool._initialized = True
    db_pool.pool = backend
    db_pool.db_path = _PATH
    db_pool._sqlite_uri = False
    return db_pool


@pytest.mark.asyncio
async def test_postgres_transaction_execute_failure_log_omits_raw_exception() -> None:
    db_pool = _transaction_pool(_PostgresTransactionPool())

    records, sink_id = _capture_logs()
    try:
        with pytest.raises(TransactionError) as exc_info:
            async with db_pool.transaction() as conn:
                await conn.execute("SELECT 1")
    finally:
        logger.remove(sink_id)

    assert exc_info.value.__cause__ is None
    assert _LEAK not in "".join(traceback.format_exception(exc_info.value))
    rendered = "\n".join(records)
    assert "PostgreSQL transaction failed" in rendered
    assert "RuntimeError" in rendered
    _assert_safe_log(rendered)


@pytest.mark.asyncio
async def test_sqlite_transaction_execute_failure_log_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _SQLiteTransactionConn()

    async def connect(*_args: Any, **_kwargs: Any) -> _SQLiteTransactionConn:
        return conn

    async def configure_connection(_conn: Any) -> None:
        return None

    monkeypatch.setattr(authnz_database.aiosqlite, "connect", connect)
    monkeypatch.setattr(authnz_database, "configure_sqlite_connection_async", configure_connection)
    db_pool = _transaction_pool(None)

    records, sink_id = _capture_logs()
    try:
        with pytest.raises(TransactionError) as exc_info:
            async with db_pool.transaction() as transaction_conn:
                await transaction_conn.execute("SELECT 1")
    finally:
        logger.remove(sink_id)

    assert exc_info.value.__cause__ is None
    assert _LEAK not in "".join(traceback.format_exception(exc_info.value))
    assert conn.rolled_back is True
    assert conn.closed is True
    rendered = "\n".join(records)
    assert "SQLite transaction failed" in rendered
    assert "RuntimeError" in rendered
    _assert_safe_log(rendered)
