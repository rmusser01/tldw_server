from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.endpoints.admin import admin_budgets
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_db_transaction

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


@pytest.mark.asyncio
async def test_admin_list_budgets_returns_canonical_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _db_override() -> object:
        return object()

    async def _fake_list_org_budgets(db, *, org_ids, page: int, limit: int):
        assert db is not None
        assert org_ids is None
        assert page == 2
        assert limit == 10
        return (
            [
                {
                    "org_id": 7,
                    "org_name": "Research",
                    "org_slug": "research",
                    "plan_name": "default",
                    "plan_display_name": "Default",
                    "budgets": {},
                    "custom_limits": {},
                    "effective_limits": {},
                    "updated_at": None,
                }
            ],
            21,
        )

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setattr(admin_budgets.admin_budgets_service, "list_org_budgets", _fake_list_org_budgets)
    app.dependency_overrides[get_db_transaction] = _db_override

    try:
        with TestClient(app, headers={"X-API-KEY": "unit-test-api-key"}) as client:
            response = client.get("/api/v1/admin/budgets", params={"page": 2, "limit": 10})
    finally:
        app.dependency_overrides.pop(get_db_transaction, None)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 21
    assert payload["page"] == 2
    assert payload["limit"] == 10
    assert payload["pagination"]["total"] == 21
    assert payload["pagination"]["limit"] == 10
    assert payload["pagination"]["offset"] == 10
    assert payload["pagination"]["has_more"] is True
    assert payload["pagination"]["next_offset"] == 20
    assert payload["has_more"] is True
    assert payload["next_offset"] == 20
