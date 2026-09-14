from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminMfaRequirementRequest,
    AdminPasswordResetRequest,
    AdminPrivilegedActionRequest,
    UserUpdateRequest,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
    MandatoryAuditWriteError,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql
from tldw_Server_API.app.core.AuthNZ.transaction_hooks import (
    begin_after_commit_scope,
    finish_after_commit_scope,
)
from tldw_Server_API.app.services import (
    admin_audit_service,
    admin_sessions_mfa_service,
    admin_users_service,
)


class _FakeCursor:
    def __init__(self, row=None, *, rows=None, rowcount: int = 1) -> None:
        self._row = row
        self._rows = rows if rows is not None else ([] if row is None else [row])
        self.rowcount = rowcount

    async def fetchone(self):
        return self._row

    async def fetchall(self):
        return self._rows


class _FakeUserDb:
    _authnz_profile_user_backend = "sqlite"

    def __init__(self, metadata: str = "{}") -> None:
        self._authnz_profile_user_guard_identity = self
        self.metadata = metadata
        self.committed = False
        self.queries: list[tuple[str, object]] = []

    async def execute(self, query: object, params=None):
        concrete = _guard_sql(
            query,
            backend="sqlite",
            connection_identity=self,
            operation="execute",
        )
        self.queries.append((concrete, params))
        if concrete.lstrip().lower().startswith("with target_user as"):
            return _FakeCursor(
                rows=[
                    (
                        "user",
                        42,
                        "2026-08-01T12:00:00.000000Z",
                    )
                ]
            )
        if "SELECT id, is_system FROM roles" in concrete:
            return _FakeCursor((2, 1))
        if "SELECT id FROM users WHERE id" in concrete:
            return _FakeCursor((42,))
        if "SELECT metadata FROM users" in concrete:
            return _FakeCursor((self.metadata,))
        return _FakeCursor()

    async def commit(self) -> None:
        self.committed = True


class _FakeSessionManager:
    def __init__(self) -> None:
        self.revoked_session_id: int | None = None
        self.revoked_by: int | None = None
        self.revoke_reason: str | None = None
        self.expected_user_id: int | None = None
        self.revoked_all_user_id: int | None = None
        self.revoked_all_by: int | None = None
        self.revoke_all_reason: str | None = None

    async def revoke_session(
        self,
        *,
        session_id: int,
        expected_user_id: int,
        revoked_by: int | None,
        reason: str | None,
    ) -> bool:
        self.revoked_session_id = session_id
        self.expected_user_id = expected_user_id
        self.revoked_by = revoked_by
        self.revoke_reason = reason
        return True

    async def revoke_all_user_sessions(
        self,
        *,
        user_id: int,
        reason: str,
        revoked_by: int | None,
    ) -> int:
        self.revoked_all_user_id = user_id
        self.revoke_all_reason = reason
        self.revoked_all_by = revoked_by
        return 1


class _FakeMfaService:
    def __init__(self, *, disable_result: bool = True) -> None:
        self.disable_result = disable_result
        self.disabled_user_id: int | None = None

    async def disable_mfa(self, user_id: int) -> bool:
        self.disabled_user_id = user_id
        return self.disable_result


class _PostgresMetadataDb:
    def __init__(self, *, update_result: str) -> None:
        self.update_result = update_result
        self.fetchrow_queries: list[str] = []

    async def fetchrow(self, query: str, *_args):
        self.fetchrow_queries.append(query)
        return {"metadata": {"preserved": True}}

    async def execute(self, *_args):
        return self.update_result


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=7,
        subject="user:7",
        roles=["admin"],
        is_admin=True,
    )


async def _allow_scope(*_args, **_kwargs) -> None:
    return None


async def _allow_reauth(*_args, **_kwargs) -> str:
    return "Support case 123"


@pytest.mark.asyncio
async def test_reset_user_password_emits_durable_audit_event(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []
    hashed_passwords: list[str] = []

    async def _fake_emit(**kwargs) -> None:
        emitted.append(kwargs)

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _allow_reauth)
    monkeypatch.setattr(admin_users_service, "_emit_admin_account_audit_event", _fake_emit, raising=False)
    monkeypatch.setattr(admin_users_service, "hash_password", lambda value: hashed_passwords.append(value) or f"hashed::{value}")

    db = _FakeUserDb()
    result = await admin_users_service.reset_user_password(
        _admin_principal(),
        42,
        AdminPasswordResetRequest(
            reason="Support case 123",
            admin_password="AdminPass123!",
            temporary_password="TempPass123!",
            force_password_change=True,
        ),
        db,
        password_service=object(),
        is_pg_fn=lambda: _false_async(),
    )

    assert result["message"] == "Password reset successfully"
    assert not db.committed
    assert hashed_passwords == ["TempPass123!"]
    assert len(emitted) == 1
    assert emitted[0]["actor_id"] == 7
    assert emitted[0]["target_user_id"] == 42
    assert emitted[0]["event_type"] == AuditEventType.USER_PASSWORD_RESET
    assert emitted[0]["category"] == AuditEventCategory.AUTHENTICATION
    assert emitted[0]["resource_type"] == "user_account"
    assert emitted[0]["resource_id"] == "42"
    assert emitted[0]["action"] == "admin.user.password_reset"
    assert emitted[0]["metadata"]["reason"] == "Support case 123"
    assert emitted[0]["metadata"]["credential_provided_by_admin"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["password", "mfa"])
async def test_postgres_admin_metadata_updates_lock_and_reject_concurrent_delete(
    monkeypatch,
    operation: str,
) -> None:
    emitted: list[dict[str, object]] = []

    async def _fake_emit(**kwargs) -> None:
        emitted.append(kwargs)

    async def _is_pg() -> bool:
        return True

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _allow_reauth)
    monkeypatch.setattr(
        admin_users_service,
        "_emit_admin_account_audit_event",
        _fake_emit,
    )
    monkeypatch.setattr(admin_users_service, "hash_password", lambda _value: "hash")
    db = _PostgresMetadataDb(update_result="UPDATE 0")

    with pytest.raises(HTTPException) as raised:
        if operation == "password":
            await admin_users_service.reset_user_password(
                _admin_principal(),
                42,
                AdminPasswordResetRequest(
                    reason="Support case 123",
                    admin_password="AdminPass123!",
                    temporary_password="TempPass123!",
                    force_password_change=True,
                ),
                db,
                password_service=object(),
                is_pg_fn=_is_pg,
            )
        else:
            await admin_users_service.set_user_mfa_requirement(
                _admin_principal(),
                42,
                AdminMfaRequirementRequest(
                    require_mfa=True,
                    reason="Support case 123",
                    admin_password="AdminPass123!",
                ),
                db,
                password_service=object(),
                is_pg_fn=_is_pg,
            )

    assert raised.value.status_code == 404
    assert db.fetchrow_queries
    assert "FOR UPDATE" in db.fetchrow_queries[0].upper()
    assert emitted == []


@pytest.mark.asyncio
async def test_delete_user_forwards_admin_reauth_token_to_guardrails(monkeypatch) -> None:
    received: dict[str, object] = {}

    async def _verify(*_args, **kwargs) -> str:
        received.update(kwargs)
        return "Support case 123"

    async def _fake_emit(**_kwargs) -> None:
        return None

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _verify)
    monkeypatch.setattr(
        admin_users_service,
        "_emit_admin_account_audit_event",
        _fake_emit,
        raising=False,
    )

    await admin_users_service.delete_user(
        _admin_principal(),
        42,
        SimpleNamespace(
            reason="Support case 123",
            admin_password=None,
            admin_reauth_token="magic-token-123",
        ),
        _FakeUserDb(),
        password_service=object(),
        is_pg_fn=lambda: _false_async(),
    )

    assert received["reason"] == "Support case 123"
    assert received["admin_password"] is None
    assert received["admin_reauth_token"] == "magic-token-123"


@pytest.mark.asyncio
async def test_delete_user_unwraps_secretstr_admin_reauth_token_before_guardrails(monkeypatch) -> None:
    received: dict[str, object] = {}

    async def _verify(*_args, **kwargs) -> str:
        received.update(kwargs)
        return "Support case 123"

    async def _fake_emit(**_kwargs) -> None:
        return None

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _verify)
    monkeypatch.setattr(
        admin_users_service,
        "_emit_admin_account_audit_event",
        _fake_emit,
        raising=False,
    )

    await admin_users_service.delete_user(
        _admin_principal(),
        42,
        AdminPrivilegedActionRequest(
            reason="Support case 123",
            admin_reauth_token="magic-token-123",
        ),
        _FakeUserDb(),
        password_service=object(),
        is_pg_fn=lambda: _false_async(),
    )

    assert received["reason"] == "Support case 123"
    assert received["admin_password"] is None
    assert received["admin_reauth_token"] == "magic-token-123"


@pytest.mark.asyncio
async def test_delete_user_emits_durable_audit_event(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []

    async def _fake_emit(**kwargs) -> None:
        emitted.append(kwargs)

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _allow_reauth)
    monkeypatch.setattr(admin_users_service, "_emit_admin_account_audit_event", _fake_emit, raising=False)

    result = await admin_users_service.delete_user(
        _admin_principal(),
        42,
        SimpleNamespace(reason="Support case 123", admin_password="AdminPass123!"),
        _FakeUserDb(),
        password_service=object(),
        is_pg_fn=lambda: _false_async(),
    )

    assert result["message"] == "User 42 has been deactivated"
    assert len(emitted) == 1
    assert emitted[0]["actor_id"] == 7
    assert emitted[0]["target_user_id"] == 42
    assert emitted[0]["event_type"] == AuditEventType.USER_DEACTIVATED
    assert emitted[0]["category"] == AuditEventCategory.AUTHORIZATION
    assert emitted[0]["resource_type"] == "user_account"
    assert emitted[0]["resource_id"] == "42"
    assert emitted[0]["action"] == "admin.user.deactivate"
    assert emitted[0]["metadata"]["reason"] == "Support case 123"


@pytest.mark.asyncio
async def test_revoke_user_session_emits_durable_audit_event(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []

    async def _fake_emit(**kwargs) -> None:
        emitted.append(kwargs)

    session_manager = _FakeSessionManager()
    monkeypatch.setattr(
        admin_sessions_mfa_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_sessions_mfa_service, "verify_privileged_action", _allow_reauth)
    monkeypatch.setattr(
        admin_sessions_mfa_service,
        "_emit_admin_account_audit_event",
        _fake_emit,
        raising=False,
    )

    result = await admin_sessions_mfa_service.revoke_user_session(
        _admin_principal(),
        42,
        84,
        session_manager,
        db=object(),
        password_service=object(),
        request=SimpleNamespace(reason="Support case 123", admin_password="AdminPass123!"),
    )

    assert result.message == "Session revoked"
    assert session_manager.revoked_session_id == 84
    assert len(emitted) == 1
    assert emitted[0]["actor_id"] == 7
    assert emitted[0]["target_user_id"] == 42
    assert emitted[0]["event_type"] == AuditEventType.AUTH_TOKEN_REVOKED
    assert emitted[0]["category"] == AuditEventCategory.AUTHENTICATION
    assert emitted[0]["resource_type"] == "user_session"
    assert emitted[0]["resource_id"] == "84"
    assert emitted[0]["action"] == "admin.user.session.revoke"
    assert emitted[0]["metadata"]["reason"] == "Support case 123"


@pytest.mark.asyncio
async def test_revoke_user_session_forwards_admin_reauth_token_to_guardrails(monkeypatch) -> None:
    received: dict[str, object] = {}

    async def _verify(*_args, **kwargs) -> str:
        received.update(kwargs)
        return "Support case 123"

    async def _fake_emit(**_kwargs) -> None:
        return None

    session_manager = _FakeSessionManager()
    monkeypatch.setattr(
        admin_sessions_mfa_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_sessions_mfa_service, "verify_privileged_action", _verify)
    monkeypatch.setattr(
        admin_sessions_mfa_service,
        "_emit_admin_account_audit_event",
        _fake_emit,
        raising=False,
    )

    await admin_sessions_mfa_service.revoke_user_session(
        _admin_principal(),
        42,
        84,
        session_manager,
        db=object(),
        password_service=object(),
        request=SimpleNamespace(
            reason="Support case 123",
            admin_password=None,
            admin_reauth_token="magic-token-456",
        ),
    )

    assert received["reason"] == "Support case 123"
    assert received["admin_password"] is None
    assert received["admin_reauth_token"] == "magic-token-456"


@pytest.mark.asyncio
async def test_revoke_user_session_unwraps_secretstr_admin_reauth_token_before_guardrails(monkeypatch) -> None:
    received: dict[str, object] = {}

    async def _verify(*_args, **kwargs) -> str:
        received.update(kwargs)
        return "Support case 123"

    async def _fake_emit(**_kwargs) -> None:
        return None

    monkeypatch.setattr(
        admin_sessions_mfa_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_sessions_mfa_service, "verify_privileged_action", _verify)
    monkeypatch.setattr(
        admin_sessions_mfa_service,
        "_emit_admin_account_audit_event",
        _fake_emit,
        raising=False,
    )

    await admin_sessions_mfa_service.revoke_user_session(
        _admin_principal(),
        42,
        99,
        _FakeSessionManager(),
        _FakeUserDb(),
        password_service=object(),
        request=AdminPrivilegedActionRequest(
            reason="Support case 123",
            admin_reauth_token="magic-token-456",
        ),
    )

    assert received["reason"] == "Support case 123"
    assert received["admin_password"] is None
    assert received["admin_reauth_token"] == "magic-token-456"


@pytest.mark.asyncio
async def test_update_user_unwraps_secretstr_admin_password_before_guardrails(monkeypatch) -> None:
    received: dict[str, object] = {}

    async def _verify(*_args, **kwargs) -> str:
        received.update(kwargs)
        return "Support case 123"

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _verify)

    await admin_users_service.update_user(
        _admin_principal(),
        42,
        UserUpdateRequest(
            role="user",
            reason="Support case 123",
            admin_password="AdminPass123!",
        ),
        _FakeUserDb(),
        password_service=object(),
        is_pg_fn=lambda: _false_async(),
    )

    assert received["reason"] == "Support case 123"
    assert received["admin_password"] == "AdminPass123!"


@pytest.mark.asyncio
async def test_revoke_all_user_sessions_emits_durable_audit_event(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []

    async def _fake_emit(**kwargs) -> None:
        emitted.append(kwargs)

    session_manager = _FakeSessionManager()
    monkeypatch.setattr(
        admin_sessions_mfa_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_sessions_mfa_service, "verify_privileged_action", _allow_reauth)
    monkeypatch.setattr(
        admin_sessions_mfa_service,
        "_emit_admin_account_audit_event",
        _fake_emit,
        raising=False,
    )

    result = await admin_sessions_mfa_service.revoke_all_user_sessions(
        _admin_principal(),
        42,
        session_manager,
        db=object(),
        password_service=object(),
        request=SimpleNamespace(reason="Support case 123", admin_password="AdminPass123!"),
    )

    assert result.message == "All sessions revoked"
    assert session_manager.revoked_all_user_id == 42
    assert len(emitted) == 1
    assert emitted[0]["actor_id"] == 7
    assert emitted[0]["target_user_id"] == 42
    assert emitted[0]["event_type"] == AuditEventType.AUTH_TOKEN_REVOKED
    assert emitted[0]["category"] == AuditEventCategory.AUTHENTICATION
    assert emitted[0]["resource_type"] == "user_session"
    assert emitted[0]["resource_id"] == "42"
    assert emitted[0]["action"] == "admin.user.sessions.revoke_all"
    assert emitted[0]["metadata"]["reason"] == "Support case 123"
    assert emitted[0]["metadata"]["scope"] == "all"


@pytest.mark.asyncio
async def test_disable_user_mfa_emits_durable_audit_event(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []

    async def _fake_emit(**kwargs) -> None:
        emitted.append(kwargs)

    mfa_service = _FakeMfaService(disable_result=True)
    monkeypatch.setattr(
        admin_sessions_mfa_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_sessions_mfa_service, "verify_privileged_action", _allow_reauth)
    monkeypatch.setattr(admin_sessions_mfa_service, "get_mfa_service", lambda: mfa_service)
    monkeypatch.setattr(
        admin_sessions_mfa_service,
        "_emit_admin_account_audit_event",
        _fake_emit,
        raising=False,
    )

    result = await admin_sessions_mfa_service.disable_user_mfa(
        _admin_principal(),
        42,
        db=object(),
        password_service=object(),
        request=SimpleNamespace(reason="Support case 123", admin_password="AdminPass123!"),
    )

    assert result.message == "MFA disabled"
    assert mfa_service.disabled_user_id == 42
    assert len(emitted) == 1
    assert emitted[0]["actor_id"] == 7
    assert emitted[0]["target_user_id"] == 42
    assert emitted[0]["event_type"] == AuditEventType.CONFIG_CHANGED
    assert emitted[0]["category"] == AuditEventCategory.SECURITY
    assert emitted[0]["resource_type"] == "user_mfa"
    assert emitted[0]["resource_id"] == "42"
    assert emitted[0]["action"] == "admin.user.mfa.disable"
    assert emitted[0]["metadata"]["reason"] == "Support case 123"


@pytest.mark.asyncio
async def test_set_user_mfa_requirement_emits_durable_audit_event(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []

    async def _fake_emit(**kwargs) -> None:
        emitted.append(kwargs)

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _allow_reauth)
    monkeypatch.setattr(admin_users_service, "_emit_admin_account_audit_event", _fake_emit, raising=False)

    db = _FakeUserDb()
    result = await admin_users_service.set_user_mfa_requirement(
        _admin_principal(),
        42,
        AdminMfaRequirementRequest(
            require_mfa=False,
            reason="Support case 123",
            admin_password="AdminPass123!",
        ),
        db,
        password_service=object(),
        is_pg_fn=lambda: _false_async(),
    )

    assert result["message"] == "MFA requirement updated successfully"
    assert not db.committed
    assert len(emitted) == 1
    assert emitted[0]["actor_id"] == 7
    assert emitted[0]["target_user_id"] == 42
    assert emitted[0]["event_type"] == AuditEventType.CONFIG_CHANGED
    assert emitted[0]["category"] == AuditEventCategory.SECURITY
    assert emitted[0]["resource_type"] == "user_mfa"
    assert emitted[0]["resource_id"] == "42"
    assert emitted[0]["action"] == "admin.user.mfa_requirement.update"
    assert emitted[0]["metadata"]["reason"] == "Support case 123"
    assert emitted[0]["metadata"]["require_mfa"] is False


@pytest.mark.asyncio
async def test_emit_admin_account_audit_event_does_not_raise_when_flush_fails(monkeypatch) -> None:
    class _FailingAuditService:
        async def log_event(self, **_kwargs) -> None:
            return None

        async def flush(self, *, raise_on_failure: bool) -> None:
            raise RuntimeError("audit unavailable")

    async def _fake_get_service(_actor_id):
        return _FailingAuditService()

    monkeypatch.setattr(
        admin_audit_service,
        "get_or_create_audit_service_for_user_id_optional",
        _fake_get_service,
    )

    output = io.StringIO()
    sink = admin_audit_service.logger.add(output, format="{message} {extra}")
    try:
        await admin_audit_service.emit_admin_account_audit_event(
            actor_id=7,
            target_user_id=42,
            event_type=AuditEventType.USER_DEACTIVATED,
            category=AuditEventCategory.AUTHORIZATION,
            resource_type="user_account",
            resource_id="42",
            action="admin.user.deactivate",
            metadata={"reason": "Support case 123"},
        )
    finally:
        admin_audit_service.logger.remove(sink)

    assert "audit unavailable" not in output.getvalue()


@pytest.mark.asyncio
async def test_emit_admin_account_audit_event_raises_when_required_flush_fails(monkeypatch) -> None:
    class _FailingAuditService:
        async def log_event(self, **_kwargs) -> None:
            return None

        async def flush(self, *, raise_on_failure: bool) -> None:
            assert raise_on_failure is True
            raise RuntimeError("audit unavailable")

    async def _fake_get_service(_actor_id):
        return _FailingAuditService()

    monkeypatch.setattr(
        admin_audit_service,
        "get_or_create_audit_service_for_user_id_optional",
        _fake_get_service,
    )

    with pytest.raises(MandatoryAuditWriteError, match="Mandatory audit persistence unavailable"):
        await admin_audit_service.emit_admin_account_audit_event(
            actor_id=7,
            target_user_id=42,
            event_type=AuditEventType.AUTH_TOKEN_CREATED,
            category=AuditEventCategory.AUTHORIZATION,
            resource_type="user_impersonation",
            resource_id="42",
            action="admin.impersonation.token.create",
            metadata={"reason": "support"},
            raise_on_failure=True,
        )


@pytest.mark.asyncio
async def test_required_admin_audit_is_persisted_before_return(monkeypatch) -> None:
    flush_calls: list[bool] = []

    class _AuditService:
        async def log_event(self, **_kwargs) -> None:
            return None

        async def flush(self, *, raise_on_failure: bool) -> None:
            flush_calls.append(raise_on_failure)

    async def _fake_get_service(_actor_id):
        return _AuditService()

    monkeypatch.setattr(
        admin_audit_service,
        "get_or_create_audit_service_for_user_id_optional",
        _fake_get_service,
    )

    token = begin_after_commit_scope()
    try:
        await admin_audit_service.emit_admin_account_audit_event(
            actor_id=7,
            target_user_id=42,
            event_type=AuditEventType.AUTH_TOKEN_CREATED,
            category=AuditEventCategory.AUTHORIZATION,
            resource_type="user_impersonation",
            resource_id="42",
            action="admin.impersonation.token.create",
            raise_on_failure=True,
        )
        assert flush_calls == [True]
    finally:
        await finish_after_commit_scope(token, committed=False)


async def _false_async() -> bool:
    return False
