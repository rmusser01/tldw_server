import ast
import inspect
import io
import textwrap
from datetime import datetime, timezone

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService


@pytest.mark.asyncio
async def test_postgres_candidate_tables_are_schema_qualified() -> None:
    class _Connection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def execute(self, statement: str) -> None:
            self.statements.append(statement)

    connection = _Connection()

    await UsersDB._ensure_profile_candidate_tables(  # noqa: SLF001
        connection,
        is_postgres=True,
    )

    assert connection.statements
    assert all(
        "CREATE TABLE IF NOT EXISTS public." in statement
        for statement in connection.statements
    )
    joined = "\n".join(connection.statements).lower()
    assert "create table if not exists public.organizations" in joined
    assert "create table if not exists public.teams" in joined
    assert "primary key (org_id, user_id)" in joined
    assert "primary key (team_id, user_id)" in joined
    assert "key text not null" in joined
    assert "value_json text" in joined
    assert "primary key (user_id, key)" in joined


@pytest.mark.asyncio
async def test_users_db_candidate_readiness_inspects_complete_contract() -> None:
    class _Connection:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def fetch(self, query: str, *args):
            self.queries.append(query)
            return []

    connection = _Connection()

    with pytest.raises(
        DatabaseError,
        match="profile candidate schema validation failed",
    ):
        await UsersDB._validate_profile_candidate_tables(  # noqa: SLF001
            connection,
            is_postgres=True,
        )

    inspected_sql = "\n".join(connection.queries).lower()
    assert "data_type" in inspected_sql
    assert "is_nullable" in inspected_sql
    assert "column_default" in inspected_sql
    assert "constraint_type = 'unique'" in inspected_sql
    assert "foreign_column_name" in inspected_sql
    assert "foreign_table_schema" in inspected_sql
    assert "delete_rule" in inspected_sql
    assert "is_identity" in inspected_sql
    assert "identity_generation" in inspected_sql


def test_users_db_validates_candidate_tables_after_both_bootstrap_paths() -> None:
    source = inspect.getsource(UsersDB._create_tables)

    assert source.count("await self._validate_profile_candidate_tables(") == 2


class _FakePgConn:
    def __init__(self):
        self.calls = []
        self._authnz_profile_user_backend = "postgres"
        self._authnz_profile_user_guard_identity = self

    # Postgres path should call fetchval in create flows and execute with $N placeholders.
    async def fetchval(self, *args, **kwargs):  # pragma: no cover - not used here
        return None

    async def fetchrow(self, query: str, *args):
        self.calls.append((query, args))
        return {
            "id": int(args[0]),
            "email": "new@example.com",
            "is_active": True,
            "is_superuser": False,
            "email_verified": False,
            "is_verified": False,
        }

    async def execute(self, query: object, *args):
        concrete = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self,
            operation="execute",
        )
        self.calls.append((concrete, args))
        return "UPDATE 1"

    async def fetch(self, query: str, *args):
        concrete = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self,
            operation="fetch",
        )
        assert "locked_user" in concrete.lower()
        return [
            {
                "source_tag": "user",
                "source_id": int(args[0]),
                "candidate_value": datetime(2026, 7, 26, 12, tzinfo=timezone.utc),
            }
        ]


class _FakeCursor:
    rowcount = 1

    def __init__(self, rows=None):
        self._rows = rows or []

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeSqliteConn:
    def __init__(self):
        self.calls = []
        self.commit_calls = 0
        self._authnz_profile_user_backend = "sqlite"
        self._authnz_profile_user_guard_identity = self

    async def execute(self, query: object, params):
        # SQLite shim path passes a single params sequence
        concrete = _guard_sql(
            query,
            backend="sqlite",
            connection_identity=self,
            operation="execute",
        )
        self.calls.append((concrete, tuple(params)))
        if concrete.lstrip().lower().startswith("with target_user as"):
            return _FakeCursor(
                [
                    (
                        "user",
                        int(params[0]),
                        "2026-07-26T12:00:00.000000Z",
                    )
                ]
            )
        if concrete == "SELECT * FROM main.users WHERE id = ?":
            return _FakeCursor(
                [
                    {
                        "id": int(params[0]),
                        "email": "new@example.com",
                        "is_active": 0,
                        "is_superuser": 0,
                        "email_verified": 0,
                        "is_verified": 0,
                    }
                ]
            )
        return _FakeCursor()

    async def commit(self):
        self.commit_calls += 1


class _FakeTx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, conn, *, postgres: bool = False):
        self._conn = conn
        self.pool = object() if postgres else None

    def transaction(self):

        return _FakeTx(self._conn)


@pytest.mark.asyncio
async def test_update_user_postgres_detects_and_uses_dollar_placeholders(monkeypatch):
    fake_conn = _FakePgConn()
    users = UsersDB(db_pool=_FakePool(fake_conn, postgres=True))
    users._initialized = True  # bypass initialize

    public_read_calls = 0

    async def _fake_get_user_by_id(self, user_id: int):
        nonlocal public_read_calls
        public_read_calls += 1
        return {
            "id": user_id,
            "email": "old@example.com",
            "is_active": True,
            "is_superuser": False,
            "email_verified": False,
        }

    monkeypatch.setattr(UsersDB, "get_user_by_id", _fake_get_user_by_id, raising=False)

    updated = await users.update_user(42, email="new@example.com", is_active=True)

    assert fake_conn.calls, "Expected at least one execute call"
    sql, args = next(
        (query, params)
        for query, params in fake_conn.calls
        if "set email = $1" in query.lower()
    )
    # Expect $1/$2 placeholders and WHERE id = $N
    assert " SET email = $1, is_active = $2, updated_at" in sql.replace("\n", " ")
    assert " WHERE id = $2" in sql or " WHERE id = $3" in sql  # position depends on updates len
    # args are varargs tuple from *values
    assert tuple(args) == ("new@example.com", True, 42) or tuple(args) == ("new@example.com", True)
    assert updated["id"] == 42
    assert public_read_calls == 1


@pytest.mark.asyncio
async def test_update_user_sqlite_detects_and_uses_qmark_placeholders(monkeypatch):
    fake_conn = _FakeSqliteConn()
    users = UsersDB(db_pool=_FakePool(fake_conn))
    users._initialized = True

    public_read_calls = 0

    async def _fake_get_user_by_id(self, user_id: int):
        nonlocal public_read_calls
        public_read_calls += 1
        return {
            "id": user_id,
            "email": "old@example.com",
            "is_active": 1,
            "is_superuser": 0,
            "email_verified": 0,
        }

    monkeypatch.setattr(UsersDB, "get_user_by_id", _fake_get_user_by_id, raising=False)

    updated = await users.update_user(7, email="new@example.com", is_active=False)

    assert fake_conn.calls, "Expected at least one execute call"
    sql, params = next(
        (query, values)
        for query, values in fake_conn.calls
        if "set email = ?" in query.lower()
    )
    flat_sql = sql.replace("\n", " ")
    assert " SET email = ?, is_active = ?, updated_at = CURRENT_TIMESTAMP" in flat_sql
    assert flat_sql.strip().endswith("WHERE id = ?")
    # Ensure booleans were coerced to ints for SQLite path and id appended last
    assert params[-1] == 7
    assert params[1] in (0, 1)
    assert updated["id"] == 7
    assert public_read_calls == 1
    assert fake_conn.commit_calls == 0


@pytest.mark.asyncio
async def test_update_user_sanitizes_storage_failure(monkeypatch):
    secret = "private@example.com constraint=users_email_key"

    class _FailingConn(_FakeSqliteConn):
        def __init__(self):
            super().__init__()
            self.candidate_reads = 0

        async def execute(self, query: object, params):
            if str(query).lstrip().lower().startswith("with target_user as"):
                self.candidate_reads += 1
            if self.candidate_reads == 2:
                raise RuntimeError(secret)
            return await super().execute(query, params)

    fake_conn = _FailingConn()
    users = UsersDB(db_pool=_FakePool(fake_conn))
    users._initialized = True

    async def _fake_get_user_by_id(self, user_id: int):
        return {"id": user_id, "email": "old@example.com"}

    monkeypatch.setattr(UsersDB, "get_user_by_id", _fake_get_user_by_id)
    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(DatabaseError) as raised:
            await users.update_user(7, email="new@example.com")
    finally:
        logger.remove(sink)

    assert str(raised.value) == "Failed to update user"
    assert raised.value.__cause__ is None
    assert secret not in output.getvalue()


@pytest.mark.asyncio
async def test_create_tables_sanitizes_bootstrap_failure():
    secret = "postgres://user:password@example.invalid/authnz"

    class _FailingTransaction:
        async def __aenter__(self):
            raise RuntimeError(secret)

        async def __aexit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            return False

    class _FailingPool:
        pool = object()

        def transaction(self):
            return _FailingTransaction()

    users = UsersDB(db_pool=_FailingPool())
    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(DatabaseError) as raised:
            await users._create_tables()
    finally:
        logger.remove(sink)

    assert str(raised.value) == "Failed to create users table"
    assert raised.value.__cause__ is None
    assert secret not in output.getvalue()


def test_postgres_uuid_default_does_not_install_extensions() -> None:
    default = UsersDB._postgres_uuid_default()  # noqa: SLF001

    assert default == "gen_random_uuid()"


@pytest.mark.parametrize(
    "method",
    [
        UsersDB.create_user,
        StorageQuotaService.calculate_user_storage,
        StorageQuotaService.update_usage,
        StorageQuotaService.set_user_quota,
    ],
)
def test_pool_owned_transaction_writers_do_not_commit_inner_connection(method) -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    commit_awaits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "commit"
    ]

    assert commit_awaits == []
