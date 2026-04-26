from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_budgets

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))


@pytest.mark.asyncio
async def test_admin_get_budget_forecast_preserves_http_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _raise_http_exception(**_kwargs):
        raise HTTPException(status_code=403, detail="forbidden")

    monkeypatch.setattr(admin_budgets.admin_budgets_service, "list_budgets", _raise_http_exception)

    with pytest.raises(HTTPException) as exc_info:
        await admin_budgets.admin_get_budget_forecast(org_id=1, principal=None, db=None)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "forbidden"


@pytest.mark.asyncio
async def test_admin_get_budget_forecast_sanitizes_generic_error_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    async def _raise_backend_error(**_kwargs):
        raise RuntimeError("budget backend exploded at /private/admin-budgets.db")

    monkeypatch.setattr(admin_budgets, "logger", logger_stub)
    monkeypatch.setattr(admin_budgets.admin_budgets_service, "list_budgets", _raise_backend_error)

    with pytest.raises(HTTPException) as exc_info:
        await admin_budgets.admin_get_budget_forecast(org_id=42, principal=None, db=None)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Budget forecast failed"
    assert logger_stub.error_records == [("Budget forecast failed", (), {})]
