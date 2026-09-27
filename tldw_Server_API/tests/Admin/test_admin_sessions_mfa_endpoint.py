from __future__ import annotations

import inspect
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_sessions_mfa
from tldw_Server_API.app.api.v1.schemas.admin_schemas import AdminPrivilegedActionRequest
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_sessions_mfa_service

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.warning_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_records.append((message, args, kwargs))


@pytest.mark.asyncio
async def test_bulk_mfa_rejects_invalid_ids() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await admin_sessions_mfa.admin_get_bulk_mfa_status(ids="1,abc,2", principal=object())

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid user IDs"


@pytest.mark.asyncio
async def test_bulk_mfa_returns_failed_ids_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    async def _fake_get_user_mfa_status(_principal: object, user_id: int) -> dict[str, bool]:
        if user_id == 1:
            return {"enabled": True}
        raise RuntimeError("mfa lookup failed at /private/mfa.db")

    monkeypatch.setattr(admin_sessions_mfa, "logger", logger_stub)
    monkeypatch.setattr(
        admin_sessions_mfa.admin_sessions_mfa_service,
        "get_user_mfa_status",
        _fake_get_user_mfa_status,
    )

    result = await admin_sessions_mfa.admin_get_bulk_mfa_status(ids="1,2", principal=object())

    assert result.mfa_status == {"1": True}
    assert result.failed_user_ids == [2]
    assert logger_stub.warning_records == [("bulk MFA status: failed for user", (), {})]


@pytest.mark.parametrize(
    "endpoint",
    [
        admin_sessions_mfa.admin_revoke_user_session,
        admin_sessions_mfa.admin_revoke_all_user_sessions,
        admin_sessions_mfa.admin_disable_user_mfa,
    ],
)
def test_admin_session_mfa_mutations_depend_on_pool_not_outer_transaction(
    endpoint,
) -> None:
    dependency = inspect.signature(endpoint).parameters["db"].default.dependency

    assert dependency is get_db_pool


@pytest.mark.asyncio
async def test_session_reauth_transaction_exits_before_mutation_and_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    connection = object()

    class _Transaction:
        async def __aenter__(self) -> object:
            events.append("reauth_enter")
            return connection

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            events.append("reauth_exit")
            return False

    class _Pool:
        def transaction(self) -> _Transaction:
            return _Transaction()

    class _SessionManager:
        async def revoke_session(self, **_kwargs) -> None:
            events.append("mutation")

    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _verify(_principal, db, _password_service, **_kwargs) -> str:
        assert db is connection
        events.append("reauth")
        return "Support case 123"

    async def _emit(**_kwargs) -> None:
        events.append("audit")

    monkeypatch.setattr(
        admin_sessions_mfa_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(
        admin_sessions_mfa_service,
        "verify_privileged_action",
        _verify,
    )
    monkeypatch.setattr(
        admin_sessions_mfa_service,
        "_emit_admin_account_audit_event",
        _emit,
    )

    await admin_sessions_mfa_service.revoke_user_session(
        AuthPrincipal(
            kind="user",
            user_id=7,
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
        ),
        42,
        84,
        _SessionManager(),
        _Pool(),
        password_service=object(),
        request=AdminPrivilegedActionRequest(
            reason="Support case 123",
            admin_password="AdminPass123!",
        ),
    )

    assert events == ["reauth_enter", "reauth", "reauth_exit", "mutation", "audit"]
