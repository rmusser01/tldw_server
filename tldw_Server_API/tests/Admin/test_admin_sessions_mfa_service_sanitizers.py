import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.admin_schemas import AdminPrivilegedActionRequest
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_sessions_mfa_service as service


pytestmark = pytest.mark.unit


class _ExplodingSessionManager:
    async def get_user_sessions(self, _user_id: int):
        raise RuntimeError("sessions list failed at /private/sessions.db")

    async def revoke_session(self, **_kwargs):
        raise RuntimeError("session revoke failed at /private/sessions.db")

    async def revoke_all_user_sessions(self, **_kwargs):
        raise RuntimeError("sessions revoke-all failed at /private/sessions.db")


class _ExplodingMfaService:
    async def get_user_mfa_status(self, _user_id: int):
        raise RuntimeError("mfa status failed at /private/mfa.db")

    async def disable_mfa(self, _user_id: int):
        raise RuntimeError("mfa disable failed at /private/mfa.db")


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["admin"], permissions=["*"], is_admin=True)


async def _allow_scope(*_args, **_kwargs) -> None:
    return None


async def _allow_privileged_action(*_args, **_kwargs) -> str:
    return "Support case 123"


async def _assert_session_mfa_500_log_sanitized(call, expected_detail: str, expected_log: str, raw_marker: str) -> None:
    messages: list[str] = []
    sink_id = service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await call()
    finally:
        service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    assert expected_log in joined
    assert raw_marker not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
async def test_list_user_sessions_sanitizes_generic_failure_log(monkeypatch):
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)

    await _assert_session_mfa_500_log_sanitized(
        lambda: service.list_user_sessions(_principal(), 42, _ExplodingSessionManager()),
        "Failed to list sessions",
        "Failed to list sessions",
        "sessions list failed",
    )


@pytest.mark.asyncio
async def test_revoke_user_session_sanitizes_generic_failure_log(monkeypatch):
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_privileged_action)

    await _assert_session_mfa_500_log_sanitized(
        lambda: service.revoke_user_session(
            _principal(),
            42,
            99,
            _ExplodingSessionManager(),
            db=object(),
            password_service=object(),
            request=AdminPrivilegedActionRequest(reason="Support case 123"),
        ),
        "Failed to revoke session",
        "Failed to revoke session",
        "session revoke failed",
    )


@pytest.mark.asyncio
async def test_revoke_all_user_sessions_sanitizes_generic_failure_log(monkeypatch):
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_privileged_action)

    await _assert_session_mfa_500_log_sanitized(
        lambda: service.revoke_all_user_sessions(
            _principal(),
            42,
            _ExplodingSessionManager(),
            db=object(),
            password_service=object(),
            request=AdminPrivilegedActionRequest(reason="Support case 123"),
        ),
        "Failed to revoke sessions",
        "Failed to revoke all sessions",
        "sessions revoke-all failed",
    )


@pytest.mark.asyncio
async def test_get_user_mfa_status_sanitizes_generic_failure_log(monkeypatch):
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "get_mfa_service", lambda: _ExplodingMfaService())

    await _assert_session_mfa_500_log_sanitized(
        lambda: service.get_user_mfa_status(_principal(), 42),
        "Failed to fetch MFA status",
        "Failed to fetch MFA status",
        "mfa status failed",
    )


@pytest.mark.asyncio
async def test_disable_user_mfa_sanitizes_generic_failure_log(monkeypatch):
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_privileged_action)
    monkeypatch.setattr(service, "get_mfa_service", lambda: _ExplodingMfaService())

    await _assert_session_mfa_500_log_sanitized(
        lambda: service.disable_user_mfa(
            _principal(),
            42,
            db=object(),
            password_service=object(),
            request=AdminPrivilegedActionRequest(reason="Support case 123"),
        ),
        "Failed to disable MFA",
        "Failed to disable MFA",
        "mfa disable failed",
    )
