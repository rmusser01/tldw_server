from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints.admin import admin_registration
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    RegistrationCodeRequest,
    RegistrationCodeResponse,
)

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debug_records.append((message, args, kwargs))


def _registration_code_response() -> RegistrationCodeResponse:
    return RegistrationCodeResponse(
        id=42,
        code="invite-code",
        max_uses=1,
        times_used=0,
        expires_at=datetime(2026, 5, 1),
        created_at=datetime(2026, 4, 25),
        created_by=7,
        role_to_grant="user",
        is_active=True,
        is_valid=True,
    )


async def _raise_audit_error(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("audit backend exploded at /private/audit.db")


@pytest.mark.asyncio
async def test_create_registration_code_sanitizes_audit_debug_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _create_registration_code(*_args: Any, **_kwargs: Any):
        return _registration_code_response(), {"event_type": "registration_code_created"}

    monkeypatch.setattr(admin_registration, "logger", logger_stub)
    monkeypatch.setattr(
        admin_registration.admin_registration_service,
        "create_registration_code",
        _create_registration_code,
    )
    monkeypatch.setattr(admin_registration, "_get_emit_admin_audit_event", lambda: _raise_audit_error)

    response = await admin_registration.create_registration_code(
        request=RegistrationCodeRequest(),
        http_request=object(),
        principal=object(),
        db=object(),
    )

    assert response.code == "invite-code"
    assert logger_stub.debug_records == [("Audit emission failed for registration code creation", (), {})]


@pytest.mark.asyncio
async def test_delete_registration_code_sanitizes_audit_debug_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _delete_registration_code(*_args: Any, **_kwargs: Any):
        return {"message": "Registration code deleted"}, {"event_type": "registration_code_deleted"}

    monkeypatch.setattr(admin_registration, "logger", logger_stub)
    monkeypatch.setattr(
        admin_registration.admin_registration_service,
        "delete_registration_code",
        _delete_registration_code,
    )
    monkeypatch.setattr(admin_registration, "_get_emit_admin_audit_event", lambda: _raise_audit_error)

    response = await admin_registration.delete_registration_code(
        code_id=42,
        http_request=object(),
        principal=object(),
        db=object(),
    )

    assert response == {"message": "Registration code deleted"}
    assert logger_stub.debug_records == [("Audit emission failed for registration code deletion", (), {})]
