from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.Watchlists_DB_Deps import get_watchlists_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import watchlists as watchlists_endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


pytestmark = pytest.mark.integration


class FrozenDateTime(datetime):
    current = datetime(2026, 7, 12, 7, 59, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        value = cls.current
        return value.astimezone(tz) if tz else value.replace(tzinfo=None)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    async def override_user():
        return User(id=971, username="schedule-preview", email=None, is_active=True)

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setattr(watchlists_endpoint, "datetime", FrozenDateTime)
    app = FastAPI()
    app.include_router(watchlists_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as test_client:
        yield test_client


def test_schedule_preview_uses_apscheduler_combined_day_semantics(client: TestClient):
    response = client.post(
        "/api/v1/watchlists/schedules/preview",
        json={"schedule_expr": "0 8 1 * MON", "timezone": "UTC"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "next_run_at": "2027-02-01T08:00:00Z",
        "following_run_at": "2027-03-01T08:00:00Z",
    }


def test_schedule_preview_returns_annual_next_and_following(client: TestClient):
    response = client.post(
        "/api/v1/watchlists/schedules/preview",
        json={"schedule_expr": "0 8 1 1 *", "timezone": "UTC"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "next_run_at": "2027-01-01T08:00:00Z",
        "following_run_at": "2028-01-01T08:00:00Z",
    }


def test_schedule_preview_preserves_dst_fold_offsets(client: TestClient):
    FrozenDateTime.current = datetime(2026, 10, 31, 8, 31, tzinfo=timezone.utc)

    response = client.post(
        "/api/v1/watchlists/schedules/preview",
        json={"schedule_expr": "30 1 * * *", "timezone": "America/Los_Angeles"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "next_run_at": "2026-11-01T01:30:00-07:00",
        "following_run_at": "2026-11-01T01:30:00-08:00",
    }


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"schedule_expr": "not cron", "timezone": "UTC"}, "invalid_schedule_expr"),
        ({"schedule_expr": "0 8 * * *", "timezone": "Mars/Olympus"}, "invalid_timezone"),
    ],
)
def test_schedule_preview_rejects_invalid_input(client: TestClient, payload: dict, detail: str):
    response = client.post("/api/v1/watchlists/schedules/preview", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"] == detail


def test_schedule_preview_route_keeps_auth_ownership_and_rate_dependencies():
    route = next(
        route
        for route in watchlists_endpoint.router.routes
        if isinstance(route, APIRoute) and route.path == "/watchlists/schedules/preview"
    )
    dependencies = [dependency.call for dependency in route.dependant.dependencies]

    assert get_request_user in dependencies
    assert get_watchlists_db_for_user in dependencies
    assert "watchlists.read" in {
        getattr(dependency, "_tldw_rate_limit_resource", None)
        for dependency in dependencies
    }
