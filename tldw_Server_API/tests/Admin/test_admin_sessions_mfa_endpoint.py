from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_sessions_mfa

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
