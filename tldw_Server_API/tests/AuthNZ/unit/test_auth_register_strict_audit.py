from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response

from tldw_Server_API.app.api.v1.endpoints import auth as auth_endpoints
from tldw_Server_API.app.api.v1.schemas.auth_schemas import RegisterRequest
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError


class _FakeRegistrationService:
    def __init__(self) -> None:
        self.rollback_calls: list[int] = []

    async def register_user(self, **kwargs):
        return {
            "user_id": 77,
            "username": kwargs["username"],
            "email": kwargs["email"],
            "is_verified": True,
        }

    async def rollback_user_registration(self, user_id: int) -> bool:
        self.rollback_calls.append(user_id)
        return True


class _FailingAPIKeyManager:
    async def create_api_key(self, **_kwargs):
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_register_returns_503_when_default_api_key_audit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration_service = _FakeRegistrationService()

    async def _fake_get_api_key_manager():
        return _FailingAPIKeyManager()

    monkeypatch.setattr(
        auth_endpoints,
        "get_settings",
        lambda: SimpleNamespace(DATABASE_URL="sqlite:///tmp/authnz-test.db", PII_REDACT_LOGS=False),
    )
    monkeypatch.setattr(auth_endpoints, "get_profile", lambda: "multi_user")
    monkeypatch.setattr(auth_endpoints, "get_api_key_manager", _fake_get_api_key_manager)
    monkeypatch.setattr(auth_endpoints, "_finalize_register_diag", lambda *_args, **_kwargs: None)

    request = SimpleNamespace(
        headers={},
        state=SimpleNamespace(),
        url=SimpleNamespace(path="/api/v1/auth/register"),
        method="POST",
        client=SimpleNamespace(host="127.0.0.1"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth_endpoints.register(
            payload=RegisterRequest(
                username="strictregister",
                email="strictregister@example.com",
                password="SecurePass123!",
            ),
            http_request=request,
            response=Response(),
            _diag=None,
            registration_service=registration_service,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "error": {
            "message": "Mandatory audit persistence unavailable",
            "type": "audit_persistence_failure",
            "code": "audit_persistence_failure",
        }
    }
    assert registration_service.rollback_calls == [77]
