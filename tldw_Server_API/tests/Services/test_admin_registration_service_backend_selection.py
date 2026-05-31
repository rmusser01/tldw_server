from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.admin_schemas import RegistrationCodeRequest, RegistrationSettingsUpdateRequest
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_registration_service as svc


class _CursorStub:
    def __init__(self, *, row: Any = None, rows: list[Any] | None = None, lastrowid: int | None = None) -> None:
        self._row = row
        self._rows = list(rows or ([] if row is None else [row]))
        self.lastrowid = lastrowid

    async def fetchone(self) -> Any:
        return self._row

    async def fetchall(self) -> list[Any]:
        return list(self._rows)


class _SqliteDbWithPgTraps:
    def __init__(self) -> None:
        self._is_sqlite = True
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.commit_calls = 0
        self._last_insert_payload: tuple[Any, ...] | None = None

    async def fetchrow(self, query: str, *args: Any) -> Any:  # pragma: no cover - trap
        self.fetchrow_calls.append((str(query), tuple(args)))
        raise AssertionError("sqlite path should not use fetchrow()")

    async def execute(self, query: str, params: Any = ()) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        q = str(query).lower()
        if "insert into registration_codes" in q:
            self._last_insert_payload = tuple(params)
            return _CursorStub(lastrowid=42)
        if "from registration_codes" in q and "where id = ?" in q:
            if self._last_insert_payload is None:
                raise AssertionError("select-after-insert requires captured insert payload")
            payload = self._last_insert_payload
            return _CursorStub(
                row=(
                    42,  # id
                    payload[0],  # code
                    payload[1],  # max_uses
                    0,  # times_used
                    payload[2],  # expires_at
                    "2026-02-09T00:00:00",
                    payload[3],  # created_by
                    payload[4],  # role_to_grant
                    payload[7],  # org_id
                    payload[8],  # org_role
                    payload[9],  # team_id
                    payload[6],  # metadata
                    1,  # is_active
                    payload[5],  # allowed_email_domain
                )
            )
        if "select rc.id" in q and "from registration_codes rc" in q:
            return _CursorStub(
                rows=[
                    (
                        55,
                        "LISTCODE",
                        1,
                        0,
                        "2099-01-01T00:00:00",
                        "2026-02-09T00:00:00",
                        1,
                        "user",
                        None,
                        None,
                        None,
                        "{}",
                        1,
                        None,
                        None,
                    )
                ]
            )
        if "update registration_codes set is_active = 0 where id = ?" in q:
            return _CursorStub()
        raise AssertionError(f"Unexpected sqlite query: {query!r}")

    async def commit(self) -> None:
        self.commit_calls += 1


class _PostgresDbWithSqliteTraps:
    def __init__(self) -> None:
        self._is_sqlite = False
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrow_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("postgres path should not use sqlite placeholders")
        q = str(query).lower()
        if "insert into registration_codes" in q:
            return {
                "id": 99,
                "code": args[0],
                "max_uses": args[1],
                "times_used": 0,
                "expires_at": args[2],
                "created_at": datetime(2026, 2, 9, 0, 0, 0),
                "created_by": args[3],
                "role_to_grant": args[4],
                "allowed_email_domain": args[5],
                "metadata": args[6],
                "org_id": args[7],
                "org_role": args[8],
                "team_id": args[9],
                "is_active": True,
            }
        return None

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("postgres path should not use sqlite placeholders")
        return [
            {
                "id": 66,
                "code": "PGCODE",
                "max_uses": 2,
                "times_used": 0,
                "expires_at": datetime(2099, 1, 1, 0, 0, 0),
                "created_at": datetime(2026, 2, 9, 0, 0, 0),
                "created_by": 1,
                "role_to_grant": "user",
                "org_id": None,
                "org_role": None,
                "team_id": None,
                "metadata": "{}",
                "is_active": True,
                "allowed_email_domain": None,
                "org_name": None,
            }
        ]

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("postgres path should not use sqlite placeholders")
        return "OK"


class _ExplodingRegistrationDb:
    _is_sqlite = True

    def __init__(self, message: str) -> None:
        self.message = message

    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(self.message)


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


async def _assert_registration_500_log_sanitized(
    call: Callable[[], Awaitable[object]],
    expected_detail: str,
    expected_log: str,
    raw_marker: str,
) -> None:
    messages: list[str] = []
    sink_id = svc.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await call()
    finally:
        svc.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    assert expected_log in joined
    assert raw_marker not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
@pytest.mark.unit
async def test_update_registration_settings_sanitizes_config_validation_errors(monkeypatch) -> None:
    def fail_update_config(_payload: dict[str, Any]) -> None:
        raise ValueError("registration config token at /private/authnz-config.txt")

    monkeypatch.setattr(svc.setup_manager, "update_config", fail_update_config)

    with pytest.raises(HTTPException) as exc_info:
        await svc.update_registration_settings(RegistrationSettingsUpdateRequest(enable_registration=True))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid registration settings"
    assert "registration config token" not in exc_info.value.detail
    assert "/private/authnz-config.txt" not in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.unit
async def test_update_registration_settings_sanitizes_generic_failure_log(monkeypatch) -> None:
    def fail_update_config(_payload: dict[str, Any]) -> None:
        raise RuntimeError("registration settings write failed at /private/authnz-config.txt")

    monkeypatch.setattr(svc.setup_manager, "update_config", fail_update_config)

    await _assert_registration_500_log_sanitized(
        lambda: svc.update_registration_settings(RegistrationSettingsUpdateRequest(enable_registration=True)),
        "Failed to update registration settings",
        "Failed to update registration settings",
        "registration settings write failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_registration_code_sanitizes_generic_failure_log() -> None:
    await _assert_registration_500_log_sanitized(
        lambda: svc.create_registration_code(
            RegistrationCodeRequest(max_uses=5, expiry_days=3, role_to_grant="user", metadata={"x": 1}),
            _admin_principal(),
            _ExplodingRegistrationDb("registration create DB failed at /private/registration.db"),
        ),
        "Failed to create registration code",
        "Failed to create registration code",
        "registration create DB failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_registration_codes_sanitizes_generic_failure_log() -> None:
    await _assert_registration_500_log_sanitized(
        lambda: svc.list_registration_codes(
            include_expired=True,
            db=_ExplodingRegistrationDb("registration list DB failed at /private/registration.db"),
        ),
        "Failed to retrieve registration codes",
        "Failed to list registration codes",
        "registration list DB failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_delete_registration_code_sanitizes_generic_failure_log() -> None:
    await _assert_registration_500_log_sanitized(
        lambda: svc.delete_registration_code(
            5,
            _ExplodingRegistrationDb("registration delete DB failed at /private/registration.db"),
        ),
        "Failed to delete registration code",
        "Failed to delete registration code",
        "registration delete DB failed",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_registration_code_sqlite_backend_selection_uses_execute() -> None:
    db = _SqliteDbWithPgTraps()

    response, audit_info = await svc.create_registration_code(
        RegistrationCodeRequest(max_uses=5, expiry_days=3, role_to_grant="user", metadata={"x": 1}),
        _admin_principal(),
        db,
    )

    assert response.id == 42
    assert response.max_uses == 5
    assert response.created_by == 1
    assert response.code and len(response.code) == 24
    assert audit_info["action"] == "registration_code.create"
    assert db.commit_calls == 1
    assert db.execute_calls
    assert not db.fetchrow_calls


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_registration_code_postgres_backend_selection_uses_fetchrow() -> None:
    db = _PostgresDbWithSqliteTraps()

    response, _audit_info = await svc.create_registration_code(
        RegistrationCodeRequest(max_uses=2, expiry_days=3, role_to_grant="admin", metadata={"x": 1}),
        _admin_principal(),
        db,
    )

    assert response.id == 99
    assert response.max_uses == 2
    assert db.fetchrow_calls
    assert not db.execute_calls
    assert all("?" not in query for query, _ in db.fetchrow_calls)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_registration_codes_sqlite_backend_selection_uses_execute() -> None:
    db = _SqliteDbWithPgTraps()

    response = await svc.list_registration_codes(include_expired=True, db=db)

    assert response.codes and response.codes[0].code == "LISTCODE"
    assert db.execute_calls
    assert not db.fetchrow_calls


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_registration_codes_postgres_backend_selection_uses_fetch() -> None:
    db = _PostgresDbWithSqliteTraps()

    response = await svc.list_registration_codes(include_expired=True, db=db)

    assert response.codes and response.codes[0].code == "PGCODE"
    assert db.fetch_calls
    assert not db.execute_calls


@pytest.mark.asyncio
@pytest.mark.unit
async def test_delete_registration_code_sqlite_backend_selection_uses_execute_and_commit() -> None:
    db = _SqliteDbWithPgTraps()

    response, audit_info = await svc.delete_registration_code(5, db)

    assert "revoked" in response["message"]
    assert audit_info["action"] == "registration_code.revoke"
    assert db.commit_calls == 1
    assert any("update registration_codes set is_active = 0" in q.lower() for q, _ in db.execute_calls)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_delete_registration_code_postgres_backend_selection_uses_execute() -> None:
    db = _PostgresDbWithSqliteTraps()

    response, audit_info = await svc.delete_registration_code(7, db)

    assert "revoked" in response["message"]
    assert audit_info["action"] == "registration_code.revoke"
    assert db.execute_calls
    assert any("$1" in q for q, _ in db.execute_calls)
