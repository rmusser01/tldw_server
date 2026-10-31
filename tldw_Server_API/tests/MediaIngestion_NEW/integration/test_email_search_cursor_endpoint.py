"""Cursor API contract against real temporary SQLite storage."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import email as email_endpoint
from tldw_Server_API.tests.DB_Management.test_email_search_cursor import add_message
from tldw_Server_API.tests.DB_Management.test_email_search_cursor import cursor_db as cursor_db

pytestmark = pytest.mark.integration


@pytest.fixture()
def cursor_client(cursor_db, monkeypatch):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)
    app = FastAPI()
    app.include_router(email_endpoint.router, prefix="/api/v1/email")
    app.dependency_overrides[get_media_db_for_user] = lambda: cursor_db
    add_message(cursor_db, "first")
    add_message(cursor_db, "second")
    with TestClient(app) as client:
        yield client


def test_email_cursor_api_traversal(cursor_client):
    first = cursor_client.get("/api/v1/email/search", params={"cursor": "", "limit": 1})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["pagination"]["mode"] == "cursor"
    assert body["has_more"] is True
    second = cursor_client.get("/api/v1/email/search", params={"cursor": body["next_cursor"], "limit": 1})
    assert second.status_code == 200, second.text
    assert [r["source_message_id"] for r in second.json()["items"]] == ["first"]
    assert second.json()["next_cursor"] is None
    assert second.json()["has_more"] is False


@pytest.mark.parametrize("params", [{"cursor": "bad"}, {"cursor": "", "offset": 1}])
def test_email_cursor_api_invalid_request_returns_400(cursor_client, params):
    response = cursor_client.get("/api/v1/email/search", params=params)
    assert response.status_code == 400, response.text


def test_email_cursor_api_keeps_offset_default(cursor_client):
    response = cursor_client.get("/api/v1/email/search", params={"limit": 1})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pagination"]["mode"] == "offset"
    assert body["next_offset"] == 1
    assert "next_cursor" not in body


def test_email_cursor_api_rejects_query_change(cursor_client):
    first = cursor_client.get("/api/v1/email/search", params={"cursor": "", "limit": 1})
    response = cursor_client.get(
        "/api/v1/email/search",
        params={"cursor": first.json()["next_cursor"], "q": "subject:Other"},
    )
    assert response.status_code == 400, response.text
