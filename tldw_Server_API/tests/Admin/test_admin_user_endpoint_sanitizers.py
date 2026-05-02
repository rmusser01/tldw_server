from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, Response

from tldw_Server_API.app.api.v1.endpoints.admin import admin_user

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.info_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.warning_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debug_records.append((message, args, kwargs))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.info_records.append((message, args, kwargs))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_records.append((message, args, kwargs))


class _FailingHeaders:
    def __setitem__(self, _key: str, _value: str) -> None:
        raise RuntimeError("header assignment leaked /private/users.db")


class _FailingHeaderResponse:
    headers = _FailingHeaders()


class _ExplodingInvitationEmailService:
    async def send_user_invitation_email(self, *_args: Any, **_kwargs: Any) -> bool:
        raise RuntimeError("email backend exploded at /private/mail.log")


@pytest.mark.asyncio
async def test_list_users_sanitizes_generic_failure_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    async def _raise_list_users(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        raise RuntimeError("list users endpoint exploded at /private/users.db")

    monkeypatch.setattr(admin_user, "logger", logger_stub)
    monkeypatch.setattr(admin_user, "is_test_mode", lambda: False)
    monkeypatch.setattr(admin_user.admin_users_service, "list_users", _raise_list_users)

    with pytest.raises(HTTPException) as exc_info:
        await admin_user.list_users(
            request=SimpleNamespace(headers={}),
            response=Response(),
            principal=object(),
            page=1,
            limit=20,
            role=None,
            admin_capable=False,
            is_active=None,
            mfa_enabled=None,
            search=None,
            org_id=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve users"
    assert logger_stub.error_records == [("Failed to list users", (), {})]


@pytest.mark.asyncio
async def test_list_users_sanitizes_http_exception_header_assignment_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    is_test_mode_calls = iter([False, True])

    async def _raise_http_exception(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        raise HTTPException(status_code=409, detail="conflict")

    monkeypatch.setattr(admin_user, "logger", logger_stub)
    monkeypatch.setattr(admin_user, "is_test_mode", lambda: next(is_test_mode_calls))
    monkeypatch.setattr(admin_user.admin_users_service, "list_users", _raise_http_exception)

    with pytest.raises(HTTPException) as exc_info:
        await admin_user.list_users(
            request=SimpleNamespace(headers={}),
            response=_FailingHeaderResponse(),
            principal=object(),
            page=1,
            limit=20,
            role=None,
            admin_capable=False,
            is_active=None,
            mfa_enabled=None,
            search=None,
            org_id=None,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "conflict"
    assert logger_stub.debug_records == [("TEST_MODE header assignment failed", (), {})]


@pytest.mark.asyncio
async def test_list_users_sanitizes_generic_header_assignment_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    is_test_mode_calls = iter([False, True])

    async def _raise_list_users(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        raise RuntimeError("list users endpoint exploded at /private/users.db")

    monkeypatch.setattr(admin_user, "logger", logger_stub)
    monkeypatch.setattr(admin_user, "is_test_mode", lambda: next(is_test_mode_calls))
    monkeypatch.setattr(admin_user.admin_users_service, "list_users", _raise_list_users)

    with pytest.raises(HTTPException) as exc_info:
        await admin_user.list_users(
            request=SimpleNamespace(headers={}),
            response=_FailingHeaderResponse(),
            principal=object(),
            page=1,
            limit=20,
            role=None,
            admin_capable=False,
            is_active=None,
            mfa_enabled=None,
            search=None,
            org_id=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve users"
    assert logger_stub.error_records == [("Failed to list users", (), {})]
    assert logger_stub.debug_records == [("TEST_MODE header assignment failed", (), {})]


@pytest.mark.asyncio
async def test_list_users_sanitizes_test_mode_diagnostic_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _raise_get_db_pool() -> None:
        raise RuntimeError("diagnostics leaked /private/users.db")

    async def _list_users(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        return [], 0

    monkeypatch.setattr(admin_user, "logger", logger_stub)
    monkeypatch.setattr(admin_user, "is_test_mode", lambda: True)
    monkeypatch.setattr(admin_user, "get_db_pool", _raise_get_db_pool)
    monkeypatch.setattr(admin_user.admin_users_service, "list_users", _list_users)

    response = await admin_user.list_users(
        request=SimpleNamespace(headers={}),
        response=Response(),
        principal=object(),
        page=1,
        limit=20,
        role=None,
        admin_capable=False,
        is_active=None,
        mfa_enabled=None,
        search=None,
        org_id=None,
    )

    assert response.total == 0
    assert logger_stub.debug_records == [
        ("Admin list_users TEST_MODE diagnostics failed", (), {}),
    ]


@pytest.mark.asyncio
async def test_export_users_sanitizes_generic_failure_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    async def _raise_export_users(*_args: Any, **_kwargs: Any) -> tuple[str, str, str]:
        raise RuntimeError("export users endpoint exploded at /private/users.db")

    monkeypatch.setattr(admin_user, "logger", logger_stub)
    monkeypatch.setattr(admin_user.admin_users_service, "export_users", _raise_export_users)

    with pytest.raises(HTTPException) as exc_info:
        await admin_user.export_users(
            role=None,
            is_active=None,
            search=None,
            org_id=None,
            limit=10000,
            offset=0,
            format="csv",
            filename=None,
            principal=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to export users"
    assert logger_stub.error_records == [("Failed to export users", (), {})]


@pytest.mark.asyncio
async def test_invite_user_sanitizes_email_failure_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    update_records: list[dict[str, Any]] = []

    def _create_invitation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "id": "inv-secret",
            "email": "alice.secret@example.com",
            "role": "user",
            "status": "pending",
            "token": "tok-secret",
        }

    def _update_invitation_email_status(**kwargs: Any) -> None:
        update_records.append(kwargs)

    fake_email_module = SimpleNamespace(get_email_service=lambda: _ExplodingInvitationEmailService())
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.AuthNZ.email_service",
        fake_email_module,
    )
    monkeypatch.setattr(admin_user, "logger", logger_stub)
    monkeypatch.setattr(admin_user, "svc_create_invitation", _create_invitation)
    monkeypatch.setattr(admin_user, "svc_update_invitation_email_status", _update_invitation_email_status)

    response = await admin_user.invite_user(
        payload=admin_user.InviteUserRequest(
            email="alice.secret@example.com",
            role="user",
            expiry_days=7,
        ),
        principal=SimpleNamespace(username="admin"),
    )

    assert response.email_sent is False
    assert response.email_error == "email backend exploded at /private/mail.log"
    assert update_records == [
        {
            "invitation_id": "inv-secret",
            "email_sent": False,
            "email_error": "email backend exploded at /private/mail.log",
        }
    ]
    assert logger_stub.warning_records == [("Failed to send invitation email", (), {})]


@pytest.mark.asyncio
async def test_resend_user_invitation_sanitizes_email_failure_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    update_records: list[dict[str, Any]] = []

    def _resend_invitation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "id": "inv-secret",
            "email": "alice.secret@example.com",
            "role": "user",
            "status": "pending",
            "token": "tok-secret-new",
            "resend_count": 2,
        }

    def _update_invitation_email_status(**kwargs: Any) -> None:
        update_records.append(kwargs)

    fake_email_module = SimpleNamespace(get_email_service=lambda: _ExplodingInvitationEmailService())
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.AuthNZ.email_service",
        fake_email_module,
    )
    monkeypatch.setattr(admin_user, "logger", logger_stub)
    monkeypatch.setattr(admin_user, "svc_resend_invitation", _resend_invitation)
    monkeypatch.setattr(admin_user, "svc_update_invitation_email_status", _update_invitation_email_status)

    response = await admin_user.resend_user_invitation(
        invitation_id="inv-secret",
        principal=SimpleNamespace(username="admin"),
    )

    assert response.email_sent is False
    assert response.email_error == "email backend exploded at /private/mail.log"
    assert update_records == [
        {
            "invitation_id": "inv-secret",
            "email_sent": False,
            "email_error": "email backend exploded at /private/mail.log",
        }
    ]
    assert logger_stub.warning_records == [("Failed to resend invitation email", (), {})]


@pytest.mark.asyncio
async def test_list_user_invitations_sanitizes_invalid_record_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    def _list_invitations(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "inv-valid",
                "email": "alice@example.com",
                "role": "user",
                "status": "pending",
                "token": "tok-valid",
            },
            {
                "id": "inv-bad",
                "email": "broken invitation leaked /private/invitations.json",
                "role": object(),
                "status": "pending",
            },
        ]

    monkeypatch.setattr(admin_user, "logger", logger_stub)
    monkeypatch.setattr(admin_user, "svc_list_invitations", _list_invitations)

    response = await admin_user.list_user_invitations(
        principal=object(),
        invitation_status=None,
    )

    assert response.total == 1
    assert response.items[0].id == "inv-valid"
    assert logger_stub.warning_records == [("Skipping invalid invitation record", (), {})]
