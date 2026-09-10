import io
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.admin_schemas import AdminPrivilegedActionRequest
from tldw_Server_API.app.core.AuthNZ.mfa_service import MFAService
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.services import admin_sessions_mfa_service as service

pytestmark = pytest.mark.unit


class _ExplodingSessionManager:
    async def get_user_sessions(self, _user_id: int, *, strict: bool = False):
        assert strict is True
        raise RuntimeError("sessions list failed at /private/sessions.db")

    async def revoke_session(self, **_kwargs):
        raise RuntimeError("session revoke failed at /private/sessions.db")

    async def revoke_all_user_sessions(self, **_kwargs):
        raise RuntimeError("sessions revoke-all failed at /private/sessions.db")


class _ExplodingMfaService:
    async def get_user_mfa_status(self, _user_id: int, *, strict: bool = False):
        assert strict is True
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
async def test_admin_session_and_mfa_reads_request_strict_backend_errors(monkeypatch):
    strict_calls: list[tuple[str, bool]] = []

    class _SessionManager:
        async def get_user_sessions(self, _user_id: int, *, strict: bool = False):
            strict_calls.append(("sessions", strict))
            return []

    class _MfaService:
        async def get_user_mfa_status(self, _user_id: int, *, strict: bool = False):
            strict_calls.append(("mfa", strict))
            return {"enabled": False}

    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "get_mfa_service", lambda: _MfaService())

    await service.list_user_sessions(_principal(), 42, _SessionManager())
    await service.get_user_mfa_status(_principal(), 42)

    assert strict_calls == [("sessions", True), ("mfa", True)]


@pytest.mark.asyncio
async def test_admin_single_session_revoke_binds_authorized_user(monkeypatch):
    captured: dict[str, Any] = {}

    class _SessionManager:
        async def revoke_session(self, **kwargs):
            captured.update(kwargs)
            return True

    async def _record_audit(**_kwargs) -> None:
        return None

    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_privileged_action)
    monkeypatch.setattr(service, "_emit_admin_account_audit_event", _record_audit)

    await service.revoke_user_session(
        _principal(),
        42,
        99,
        _SessionManager(),
        db=object(),
        password_service=object(),
        request=AdminPrivilegedActionRequest(reason="Support case 123"),
    )

    assert captured["expected_user_id"] == 42
    assert captured["revoked_by"] == _principal().user_id
    assert captured["reason"] == "Support case 123"


@pytest.mark.asyncio
async def test_admin_bulk_session_revoke_propagates_actor_and_reason(monkeypatch):
    captured: dict[str, Any] = {}

    class _SessionManager:
        async def revoke_all_user_sessions(self, **kwargs):
            captured.update(kwargs)
            return 3

    async def _record_audit(**_kwargs) -> None:
        return None

    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_privileged_action)
    monkeypatch.setattr(service, "_emit_admin_account_audit_event", _record_audit)

    await service.revoke_all_user_sessions(
        _principal(),
        42,
        _SessionManager(),
        db=object(),
        password_service=object(),
        request=AdminPrivilegedActionRequest(reason="Support case 123"),
    )

    assert captured == {
        "user_id": 42,
        "reason": "Support case 123",
        "revoked_by": _principal().user_id,
    }


@pytest.mark.asyncio
async def test_admin_session_ownership_mismatch_skips_success_audit(monkeypatch):
    audit_calls: list[dict[str, Any]] = []

    class _SessionManager:
        async def revoke_session(self, **_kwargs):
            return False

    async def _record_audit(**kwargs) -> None:
        audit_calls.append(kwargs)

    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_privileged_action)
    monkeypatch.setattr(service, "_emit_admin_account_audit_event", _record_audit)

    with pytest.raises(HTTPException) as exc_info:
        await service.revoke_user_session(
            _principal(),
            42,
            99,
            _SessionManager(),
            db=object(),
            password_service=object(),
            request=AdminPrivilegedActionRequest(reason="Support case 123"),
        )

    assert exc_info.value.status_code == 404
    assert audit_calls == []


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


@pytest.mark.asyncio
async def test_real_session_manager_failure_maps_to_sanitized_500(monkeypatch):
    marker = "session backend failed at /private/sessions.db?token=secret"
    manager = SessionManager(
        settings=Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="rotation-new-secret-1234567890abcd",
        )
    )
    manager._initialized = True
    manager.redis_client = None

    class _FailingRepo:
        def __init__(self, _db_pool) -> None:
            return None

        async def revoke_session_record(self, **_kwargs):
            raise RuntimeError(marker)

    async def _fake_ensure_db_pool():
        return object()

    monkeypatch.setattr(manager, "_ensure_db_pool", _fake_ensure_db_pool)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.session_manager.AuthnzSessionsRepo",
        _FailingRepo,
    )
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_privileged_action)
    output = io.StringIO()
    sink = service.logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await service.revoke_user_session(
                _principal(),
                42,
                99,
                manager,
                db=object(),
                password_service=object(),
                request=AdminPrivilegedActionRequest(reason="Support case 123"),
            )
    finally:
        service.logger.remove(sink)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to revoke session"
    assert exc_info.value.__cause__ is None
    assert marker not in output.getvalue()
    assert "/private/" not in output.getvalue()


@pytest.mark.asyncio
async def test_real_mfa_service_backend_failure_maps_to_sanitized_500(monkeypatch):
    marker = "mfa backend failed at /private/mfa.db?token=secret"

    class _FailingRepo:
        async def clear_mfa_config(self, **_kwargs) -> None:
            raise RuntimeError(marker)

    mfa_service = MFAService(
        db_pool=object(),
        settings=SimpleNamespace(APP_NAME="TLDW", PII_REDACT_LOGS=False),
        repo=_FailingRepo(),
    )
    mfa_service._initialized = True
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(service, "verify_privileged_action", _allow_privileged_action)
    monkeypatch.setattr(service, "get_mfa_service", lambda: mfa_service)
    output = io.StringIO()
    sink = service.logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await service.disable_user_mfa(
                _principal(),
                42,
                db=object(),
                password_service=object(),
                request=AdminPrivilegedActionRequest(reason="Support case 123"),
            )
    finally:
        service.logger.remove(sink)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to disable MFA"
    assert exc_info.value.__cause__ is None
    assert marker not in output.getvalue()
    assert "/private/" not in output.getvalue()
