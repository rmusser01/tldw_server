from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql
from tldw_Server_API.app.services import registration_service as registration_module
from tldw_Server_API.app.services.registration_service import RegistrationService


class _Tx:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _PoolStub:
    def __init__(self, conn: Any, *, postgres: bool) -> None:
        self._conn = conn
        self.pool = object() if postgres else None

    def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ) -> _Tx:
        del acquire_timeout_seconds
        return _Tx(self._conn)


class _CursorStub:
    def __init__(
        self,
        *,
        row: Any = None,
        lastrowid: int | None = None,
        rowcount: int = 1,
    ) -> None:
        self._row = row
        self.lastrowid = lastrowid
        self.rowcount = rowcount

    async def fetchone(self) -> Any:
        return self._row


class _PasswordServiceStub:
    def validate_password_strength(self, password: str, username: str | None = None) -> None:  # noqa: ARG002
        return None

    def hash_password(self, password: str) -> str:
        return f"hash-{password}"


def _settings_stub() -> SimpleNamespace:
    return SimpleNamespace(
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
        REGISTRATION_CODE_DEFAULT_EXPIRY_DAYS=7,
        ENABLE_ORG_SCOPED_REGISTRATION_CODES=True,
        DEFAULT_USER_ROLE="user",
        DEFAULT_STORAGE_QUOTA_MB=1024,
        USER_DATA_BASE_PATH="/tmp",  # nosec B108
        CHROMADB_BASE_PATH=None,
    )


class _SQLiteRegisterConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []
        self.committed = False
        self.membership_inserted = False

    async def execute(self, query: object, params: Any) -> _CursorStub:
        concrete = _guard_sql(
            query,
            backend="sqlite",
            connection_identity=self,
            operation="execute",
        )
        lowered = concrete.lower()
        self.execute_calls.append((lowered, params))
        if "select username, email" in lowered:
            return _CursorStub(row=None)
        if "insert into main.users" in lowered:
            return _CursorStub(lastrowid=29)
        if "select id from roles where name" in lowered:
            return _CursorStub(row=(11,))
        if "insert into user_roles" in lowered:
            self.membership_inserted = params == (29, 11)
            return _CursorStub(lastrowid=1)
        if "insert into password_history" in lowered:
            return _CursorStub(lastrowid=1)
        if "insert into audit_log" in lowered:
            return _CursorStub(lastrowid=1)
        raise AssertionError(f"Unexpected SQLite query: {query!r}")

    async def commit(self) -> None:
        self.committed = True


class _SQLiteConnWithFetchvalTrap:
    """
    SQLite-like connection shim that intentionally exposes ``fetchval``.

    RegistrationService should still select the SQLite SQL path when the
    DatabasePool indicates SQLite (pool is None).
    """

    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []
        self.committed = False

    async def execute(self, query: str, params: Any) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        return _CursorStub(lastrowid=17)

    async def commit(self) -> None:
        self.committed = True

    async def fetchval(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchval")


class _PostgresConnWithExecuteTrap:
    def __init__(self) -> None:
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchval(self, query: str, *params: Any) -> int:
        self.fetchval_calls.append((str(query), tuple(params)))
        return 23

    async def execute(self, query: str, *params: Any):  # noqa: ARG002
        raise AssertionError(f"Postgres backend path should not call conn.execute: {query!r}")


class _PostgresRegisterConn:
    def __init__(self) -> None:
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.membership_inserted = False

    async def fetchrow(self, query: str, *params: Any) -> None:  # noqa: ARG002
        assert "select username, email" in query.lower()
        return None

    async def fetchval(self, query: object, *params: Any) -> int | None:
        concrete = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self,
            operation="fetchval",
        )
        lowered = concrete.lower()
        self.fetchval_calls.append((lowered, tuple(params)))
        if "insert into public.users" in lowered:
            return 31
        if "select id from roles where name" in lowered:
            return 13
        raise AssertionError(f"Unexpected PostgreSQL fetchval query: {query!r}")

    async def execute(self, query: str, *params: Any) -> None:
        lowered = query.lower()
        self.execute_calls.append((lowered, tuple(params)))
        if "insert into user_roles" in lowered:
            self.membership_inserted = params == (31, 13)
            return
        if "insert into password_history" in lowered or "insert into audit_log" in lowered:
            return
        raise AssertionError(f"Unexpected PostgreSQL execute query: {query!r}")


class _SQLiteConnWithFetchrowTrap:
    def __init__(self, expires_at: datetime) -> None:
        self._expires_at = expires_at
        self.update_calls = 0

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def execute(self, query: str, params: Any) -> _CursorStub:
        q = str(query).lower()
        if "from registration_codes" in q:
            return _CursorStub(
                row=(
                    1,  # id
                    "user",  # role_to_grant
                    0,  # times_used
                    2,  # max_uses
                    self._expires_at.isoformat(),
                    1,  # is_active
                    "unit test code",
                    None,  # allowed_email_domain
                    None,  # org_id
                    None,  # org_role
                    None,  # team_id
                    None,  # metadata
                )
            )
        if "update registration_codes" in q:
            self.update_calls += 1
            return _CursorStub()
        raise AssertionError(f"Unexpected SQLite query: {query!r}")


def _make_service(db_pool: _PoolStub) -> RegistrationService:
    return RegistrationService(
        db_pool=db_pool,
        password_service=_PasswordServiceStub(),
        settings=_settings_stub(),
    )


@pytest.mark.asyncio
async def test_create_registration_code_sqlite_backend_selection_ignores_conn_fetchval():
    conn = _SQLiteConnWithFetchvalTrap()
    service = _make_service(_PoolStub(conn, postgres=False))
    service.generate_registration_code = lambda length=24: "sqlite-backend-code"  # noqa: ARG005

    payload = await service.create_registration_code(created_by=7, max_uses=3, role_to_grant="user")

    assert payload["id"] == 17
    assert conn.committed is True
    assert conn.execute_calls
    assert "values (?, ?, ?, ?, ?, ?, ?)" in conn.execute_calls[0][0].lower()


@pytest.mark.asyncio
async def test_create_registration_code_postgres_backend_selection_uses_fetchval():
    conn = _PostgresConnWithExecuteTrap()
    service = _make_service(_PoolStub(conn, postgres=True))
    service.generate_registration_code = lambda length=24: "postgres-backend-code"  # noqa: ARG005

    payload = await service.create_registration_code(created_by=8, max_uses=2, role_to_grant="admin")

    assert payload["id"] == 23
    assert conn.fetchval_calls, "expected Postgres fetchval path to be used"
    assert "returning id" in conn.fetchval_calls[0][0].lower()


@pytest.mark.asyncio
async def test_validate_registration_code_sqlite_backend_selection_ignores_conn_fetchrow():
    expires_at = datetime.utcnow() + timedelta(days=1)
    conn = _SQLiteConnWithFetchrowTrap(expires_at=expires_at)
    service = _make_service(_PoolStub(conn, postgres=False))

    info = await service._validate_and_use_registration_code(
        "unit-code",
        conn,
        user_email="tester@example.com",
    )

    assert info["id"] == 1
    assert info["role_to_grant"] == "user"
    assert conn.update_calls == 1


@pytest.mark.asyncio
async def test_register_user_system_provisioning_bypasses_disabled_registration_and_required_code():
    conn = _SQLiteRegisterConn()
    settings = _settings_stub()
    settings.ENABLE_REGISTRATION = False
    settings.REQUIRE_REGISTRATION_CODE = True
    service = RegistrationService(
        db_pool=_PoolStub(conn, postgres=False),
        password_service=_PasswordServiceStub(),
        settings=settings,
    )
    service._create_user_directories = lambda user_id: True  # noqa: ARG005

    payload = await service.register_user(
        username="federated-user",
        email="federated@example.com",
        password="Strong!Pass9",
        role_override="member",
        is_verified_override=True,
        system_provisioning=True,
    )

    assert payload["user_id"] == 29
    assert payload["role"] == "member"
    assert payload["is_verified"] is True
    assert payload["registration_code_id"] is None
    assert conn.committed is False
    assert conn.membership_inserted is True
    assert not any("from registration_codes" in query for query, _ in conn.execute_calls)


@pytest.mark.asyncio
async def test_register_user_postgres_persists_role_membership_before_returning() -> None:
    conn = _PostgresRegisterConn()
    service = _make_service(_PoolStub(conn, postgres=True))
    service._create_user_directories = lambda user_id: True  # noqa: ARG005

    payload = await service.register_user(
        username="postgres-user",
        email="postgres-user@example.com",
        password="Strong!Pass9",
    )

    assert payload["user_id"] == 31
    assert payload["role"] == "user"
    assert conn.membership_inserted is True
    assert any("select id from roles where name = $1" in query for query, _ in conn.fetchval_calls)


@pytest.mark.asyncio
async def test_register_user_rejects_negative_storage_quota_override() -> None:
    conn = _SQLiteRegisterConn()
    service = _make_service(_PoolStub(conn, postgres=False))
    service._create_user_directories = lambda user_id: True  # noqa: ARG005

    with pytest.raises(ValueError, match="storage_quota_override must be non-negative"):
        await service.register_user(
            username="federated-user",
            email="federated@example.com",
            password="Strong!Pass9",
            storage_quota_override=-1,
            system_provisioning=True,
        )


@pytest.mark.asyncio
async def test_registration_rollback_failure_does_not_render_backend_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "secret SQL at /private/registration.db"

    class _FailingTransaction:
        async def __aenter__(self):
            raise RuntimeError(secret)

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
            return False

    class _FailingPool:
        pool = None

        def transaction(self) -> _FailingTransaction:
            return _FailingTransaction()

    class _Logger:
        def __init__(self) -> None:
            self.events: list[str] = []

        def bind(self, **values):
            self.events.append(repr(values))
            return self

        def opt(self, *, exception):
            self.events.append(repr(exception))
            return self

        def error(self, message: str, *args) -> None:
            self.events.append(message.format(*args))

    logger_stub = _Logger()
    service = _make_service(_FailingPool())
    monkeypatch.setattr(registration_module, "logger", logger_stub)

    rolled_back = await service.rollback_user_registration(17)

    rendered = "\n".join(logger_stub.events)
    assert rolled_back is False
    assert secret not in rendered
    assert "RuntimeError" in rendered
