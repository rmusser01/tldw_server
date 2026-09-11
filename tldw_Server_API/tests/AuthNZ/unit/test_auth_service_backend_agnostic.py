from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    ProfileUserWriteRejected,
    _guard_sql,
)
from tldw_Server_API.app.services import auth_service


def _profile_candidate(user_id: int) -> list[dict[str, Any]]:
    return [
        {
            "source_tag": "user",
            "source_id": user_id,
            "candidate_value": datetime(2026, 2, 9, 11, 0, tzinfo=timezone.utc),
        }
    ]


class _Cursor:
    def __init__(self, row: Any) -> None:
        self._row = row

    async def fetchone(self) -> Any:
        return self._row


@pytest.mark.unit
def test_versioned_user_gateway_prefers_explicit_sqlite_backend_marker() -> None:
    class _SqliteAdapterWithFetch:
        _authnz_profile_user_backend = "sqlite"

        async def fetch(self, query: str, *args: Any) -> list[Any]:
            del query, args
            return []

    gateway = auth_service._versioned_user_gateway(_SqliteAdapterWithFetch())

    assert gateway.backend == "sqlite"


@pytest.mark.unit
@pytest.mark.parametrize("marker", ["mysql", 1])
def test_versioned_user_gateway_rejects_invalid_backend_marker(marker: Any) -> None:
    class _InvalidBackendAdapter:
        _authnz_profile_user_backend = marker

    with pytest.raises(ProfileUserWriteRejected):
        auth_service._versioned_user_gateway(_InvalidBackendAdapter())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_user_by_login_identifier_prefers_fetchrow() -> None:
    db = AsyncMock()
    db.fetchrow = AsyncMock(return_value={"id": 7, "username": "alice"})

    result = await auth_service.fetch_user_by_login_identifier(db, "Alice@Example.com")

    assert result == {"id": 7, "username": "alice"}
    db.fetchrow.assert_awaited_once_with(
        "SELECT * FROM users WHERE lower(username) = $1 OR lower(email) = $2",
        "alice@example.com",
        "alice@example.com",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_user_by_login_identifier_sqlite_fallback_uses_qmark() -> None:
    class _SqliteLikeConn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[Any, ...]]] = []

        async def execute(self, query: str, params: tuple[Any, ...]) -> _Cursor:
            self.calls.append((query, params))
            return _Cursor({"id": 3, "username": "bob"})

    db = _SqliteLikeConn()

    result = await auth_service.fetch_user_by_login_identifier(db, "BOB")

    assert result == {"id": 3, "username": "bob"}
    assert db.calls == [
        (
            "SELECT * FROM users WHERE lower(username) = ? OR lower(email) = ?",
            ("bob", "bob"),
        )
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_user_password_hash_commits_sqlite_like_connection() -> None:
    class _SqliteLikeConn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[Any, ...]]] = []
            self.commits = 0

        async def execute(self, query: str, params: tuple[Any, ...]) -> None:
            self.calls.append((query, params))

        async def commit(self) -> None:
            self.commits += 1

    db = _SqliteLikeConn()

    await auth_service.update_user_password_hash(db, 42, "new-hash")

    assert db.calls == [("UPDATE users SET password_hash = ? WHERE id = ?", ("new-hash", 42))]
    assert db.commits == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_active_user_by_id_normalizes_sqlite_tuple_boolean() -> None:
    class _SqliteLikeConn:
        async def execute(self, query: str, params: tuple[Any, ...]) -> _Cursor:
            assert query == "SELECT * FROM users WHERE id = ? AND is_active = ?"
            assert params == (9, True)
            return _Cursor(
                (
                    9,
                    "f7d8d7ac-2c08-4f50-92dd-111111111111",
                    "carol",
                    "carol@example.com",
                    "hash",
                    "user",
                    1,
                    0,
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:00",
                    1024,
                    12.5,
                )
            )

    result = await auth_service.fetch_active_user_by_id(_SqliteLikeConn(), 9)

    assert result is not None
    assert result["id"] == 9
    assert result["is_active"] is True
    assert result["username"] == "carol"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_user_last_login_uses_adapter_query_shape() -> None:
    now = datetime(2026, 2, 9, 12, 0, 0)
    expected_now = now.replace(tzinfo=timezone.utc)
    db = AsyncMock()
    identity = object()
    db._authnz_profile_user_backend = "postgres"
    db._authnz_profile_user_guard_identity = identity
    executed: list[tuple[str, tuple[Any, ...]]] = []

    async def _execute(statement: Any, *parameters: Any) -> str:
        executed.append(
            (
                _guard_sql(
                    statement,
                    backend="postgres",
                    connection_identity=identity,
                    operation="execute",
                ),
                parameters,
            )
        )
        return "UPDATE 1"

    db.execute = AsyncMock(side_effect=_execute)
    db.fetch = AsyncMock(return_value=_profile_candidate(17))
    db.commit = AsyncMock()

    await auth_service.update_user_last_login(db, 17, now)

    assert (
        "UPDATE public.users SET last_login = $1 WHERE id = $2",
        (expected_now, 17),
    ) in executed
    db.commit.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_managed_sqlite_service_does_not_commit_caller_transaction() -> None:
    class _ManagedSqliteConn:
        _authnz_profile_user_backend = "sqlite"
        _authnz_profile_user_guard_identity = object()

        def __init__(self) -> None:
            self.commits = 0

        async def execute(self, query: str, params: tuple[Any, ...]) -> None:
            assert query == "UPDATE users SET password_hash = ? WHERE id = ?"
            assert params == ("new-hash", 42)

        async def commit(self) -> None:
            self.commits += 1

    db = _ManagedSqliteConn()

    await auth_service.update_user_password_hash(db, 42, "new-hash")

    assert db.commits == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_store_password_reset_token_sqlite_fallback_normalizes_placeholders() -> None:
    class _SqliteLikeConn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[Any, ...]]] = []
            self.commits = 0

        async def execute(self, query: str, params: tuple[Any, ...]) -> None:
            self.calls.append((query, params))

        async def commit(self) -> None:
            self.commits += 1

    db = _SqliteLikeConn()
    expires = datetime(2026, 2, 9, 12, 30, 0)

    await auth_service.store_password_reset_token(
        db,
        user_id=7,
        token_hash="tok-hash",
        expires_at=expires,
        ip_address="203.0.113.9",
    )

    assert len(db.calls) == 1
    query, params = db.calls[0]
    assert "INSERT INTO password_reset_tokens" in query
    assert "VALUES (?, ?, ?, ?)" in query
    assert params == (7, "tok-hash", expires, "203.0.113.9")
    assert db.commits == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_password_reset_token_record_sqlite_fallback_dynamic_in_clause() -> None:
    class _SqliteLikeConn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[Any, ...]]] = []

        async def execute(self, query: str, params: tuple[Any, ...]) -> _Cursor:
            self.calls.append((query, params))
            return _Cursor((55, None))

    db = _SqliteLikeConn()
    token_id, used_at = await auth_service.fetch_password_reset_token_record(
        db,
        user_id=4,
        hash_candidates=["h1", "h2"],
    )

    assert token_id == 55
    assert used_at is None
    assert len(db.calls) == 1
    query, params = db.calls[0]
    assert "FROM password_reset_tokens" in query
    assert "token_hash IN (?, ?)" in query
    assert params == (4, "h1", "h2")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_verify_user_email_once_uses_postgres_update_count_when_available() -> None:
    db = AsyncMock()
    db.execute = AsyncMock(return_value="UPDATE 1")
    db.fetch = AsyncMock(return_value=_profile_candidate(9))
    db.fetchrow = AsyncMock()
    db.commit = AsyncMock()

    updated = await auth_service.verify_user_email_once(
        db,
        user_id=9,
        email="User@Example.com",
        now_utc=datetime(2026, 2, 9, 13, 0, 0),
    )

    assert updated == 1
    update_parameters = db.execute.await_args_list[0].args[1:]
    assert update_parameters[1] == datetime(
        2026,
        2,
        9,
        13,
        0,
        tzinfo=timezone.utc,
    )
    db.fetchrow.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mark_user_verified_normalizes_naive_postgres_timestamp_to_utc() -> None:
    db = AsyncMock()
    db._authnz_profile_user_backend = "postgres"
    db.execute = AsyncMock(return_value="UPDATE 1")
    db.fetch = AsyncMock(return_value=_profile_candidate(11))

    await auth_service.mark_user_verified(
        db,
        user_id=11,
        now_utc=datetime(2026, 2, 9, 14, 0, 0),
    )

    update_parameters = db.execute.await_args_list[0].args[1:]
    assert update_parameters[1] == datetime(
        2026,
        2,
        9,
        14,
        0,
        tzinfo=timezone.utc,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_password_reset_normalizes_naive_postgres_timestamps_to_utc() -> None:
    db = AsyncMock()
    db._authnz_profile_user_backend = "postgres"
    db.execute = AsyncMock(return_value="UPDATE 1")

    await auth_service.apply_password_reset(
        db,
        user_id=12,
        new_password_hash="hash",
        token_record_id=34,
        now_utc=datetime(2026, 2, 9, 15, 0, 0),
    )

    expected = datetime(2026, 2, 9, 15, 0, tzinfo=timezone.utc)
    assert db.execute.await_args_list[0].args[2] == expected
    assert db.execute.await_args_list[1].args[1] == expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_verify_user_email_once_preserves_zero_for_missing_user() -> None:
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.execute = AsyncMock()
    db.commit = AsyncMock()

    updated = await auth_service.verify_user_email_once(
        db,
        user_id=404,
        email="missing@example.com",
        now_utc=datetime(2026, 2, 9, 13, 0, 0),
    )

    assert updated == 0
    db.execute.assert_not_awaited()
    db.commit.assert_awaited_once_with()
