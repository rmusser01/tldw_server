from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminPrivilegedActionRequest,
    AdminMfaRequirementRequest,
    AdminPasswordResetRequest,
    UserUpdateRequest,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import (
    admin_audit_service,
    admin_sessions_mfa_service,
    admin_users_service,
)


class _FakeCursor:
    def __init__(self, row=None, *, rowcount: int = 1) -> None:
        self._row = row
        self.rowcount = rowcount

    async def fetchone(self):
        return self._row


class _FakeUserDb:
    def __init__(self, metadata: str = "{}") -> None:
        self.metadata = metadata
        self.committed = False
        self.queries: list[tuple[str, object]] = []

    async def execute(self, query: str, params=None):
        self.queries.append((query, params))
        if "SELECT id, is_system FROM roles" in query:
            return _FakeCursor((2, 1))
        if "SELECT metadata FROM users" in query:
            return _FakeCursor((self.metadata,))
        return _FakeCursor()

    async def commit(self) -> None:
        self.committed = True


class _FakeSessionManager:
    def __init__(self) -> None:
        self.revoked_session_id: int | None = None
        self.revoked_by: int | None = None
        self.revoked_all_user_id: int | None = None

    async def revoke_session(self, *, session_id: int, revoked_by: int | None) -> None:
        self.revoked_session_id = session_id
        self.revoked_by = revoked_by

    async def revoke_all_user_sessions(self, *, user_id: int) -> None:
        self.revoked_all_user_id = user_id


class _FakeMfaService:
    def __init__(self, *, disable_result: bool = True) -> None:
        self.disable_result = disable_result
        self.disabled_user_id: int | None = None

    async def disable_mfa(self, user_id: int) -> bool:
        self.disabled_user_id = user_id
        return self.disable_result


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

    result = await admin_users_service.reset_user_password(
        _admin_principal(),
        42,
        AdminPasswordResetRequest(
            reason="Support case 123",
            admin_password="AdminPass123!",
            temporary_password="TempPass123!",
            force_password_change=True,
        ),
        _FakeUserDb(),
        password_service=object(),
        is_pg_fn=lambda: _false_async(),
    )

    assert result["message"] == "Password reset successfully"
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

    result = await admin_users_service.set_user_mfa_requirement(
        _admin_principal(),
        42,
        AdminMfaRequirementRequest(
            require_mfa=False,
            reason="Support case 123",
            admin_password="AdminPass123!",
        ),
        _FakeUserDb(),
        password_service=object(),
        is_pg_fn=lambda: _false_async(),
    )

    assert result["message"] == "MFA requirement updated successfully"
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


async def _false_async() -> bool:
    return False
