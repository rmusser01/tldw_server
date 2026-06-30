from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoint
from tldw_Server_API.app.core.config import API_V1_PREFIX
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError

pytestmark = pytest.mark.integration


class _FakeMediaDB:
    last_email_search_kwargs: dict | None = None
    last_legacy_search_kwargs: dict | None = None
    email_search_calls: int = 0
    legacy_search_calls: int = 0

    def search_media_db(self, **_kwargs):
        type(self).last_legacy_search_kwargs = dict(_kwargs)
        type(self).legacy_search_calls += 1
        return ([{"id": 1, "title": "Alpha", "type": "document"}], 1)

    def search_email_messages(self, *, query=None, limit=50, offset=0, **_kwargs):
        type(self).last_email_search_kwargs = dict(_kwargs)
        type(self).email_search_calls += 1
        if query and "(" in str(query):
            raise InputError("Parentheses are not supported in email query v1.")
        rows = [
            {
                "email_message_id": 99,
                "media_id": 42,
                "media_title": "Operator Result",
                "subject": "Operator Subject",
            }
        ]
        return rows[offset : offset + limit], 1


class _ErroringEmailSearchMediaDB(_FakeMediaDB):
    def __init__(self, search_exc: Exception):
        self._search_exc = search_exc

    def search_email_messages(self, *, query=None, limit=50, offset=0, **_kwargs):
        _ = (query, limit, offset, _kwargs)
        raise self._search_exc


@pytest.fixture()
def media_search_client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    _FakeMediaDB.last_email_search_kwargs = None
    _FakeMediaDB.last_legacy_search_kwargs = None
    _FakeMediaDB.email_search_calls = 0
    _FakeMediaDB.legacy_search_calls = 0
    app = FastAPI()
    app.include_router(listing_endpoint.router, prefix=f"{API_V1_PREFIX}/media")
    app.dependency_overrides[get_media_db_for_user] = lambda: _FakeMediaDB()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture()
def media_search_client_factory(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")

    def _build(db):
        app = FastAPI()
        app.include_router(listing_endpoint.router, prefix=f"{API_V1_PREFIX}/media")
        app.dependency_overrides[get_media_db_for_user] = lambda: db
        return TestClient(app), app

    yield _build


def test_media_search_accepts_minimal_payload(media_search_client):


    resp = media_search_client.post("/api/v1/media/search", json={"query": "Alpha"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "items" in body and isinstance(body["items"], list)
    assert "pagination" in body and isinstance(body["pagination"], dict)


def test_media_search_forbids_extra_keys(media_search_client):


    resp = media_search_client.post(
        "/api/v1/media/search",
        json={"query": "Alpha", "extra_key": True},
    )
    assert resp.status_code == 422


def test_media_search_rejects_bad_fields_type(media_search_client):


    resp = media_search_client.post(
        "/api/v1/media/search",
        json={"query": "Alpha", "fields": "title"},
    )
    assert resp.status_code == 422


def test_media_search_rejects_query_too_long(media_search_client):


    resp = media_search_client.post(
        "/api/v1/media/search",
        json={"query": "a" * 1001},
    )
    assert resp.status_code == 422


def test_media_search_forwards_boost_fields(media_search_client):
    resp = media_search_client.post(
        "/api/v1/media/search",
        json={
            "query": "Alpha",
            "boost_fields": {
                "title": 3.5,
                "content": 0.5,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    assert _FakeMediaDB.last_legacy_search_kwargs is not None
    assert _FakeMediaDB.last_legacy_search_kwargs.get("boost_fields") == {
        "title": 3.5,
        "content": 0.5,
    }


def test_media_search_email_operator_mode_delegates_to_email_search(media_search_client):
    resp = media_search_client.post(
        "/api/v1/media/search",
        json={
            "query": "from:alice@example.com",
            "media_types": ["email"],
            "email_query_mode": "operators",
        },
        params={"page": 1, "results_per_page": 10},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"]
    assert body["items"][0]["id"] == 42
    assert body["items"][0]["type"] == "email"
    assert _FakeMediaDB.last_email_search_kwargs == {"include_deleted": False}
    assert _FakeMediaDB.email_search_calls == 1
    assert _FakeMediaDB.legacy_search_calls == 0


def test_media_search_email_scope_defaults_to_legacy_planner(media_search_client):
    resp = media_search_client.post(
        "/api/v1/media/search",
        json={
            "query": "from:alice@example.com",
            "media_types": ["email"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"][0]["id"] == 1
    assert body["items"][0]["type"] == "document"
    assert _FakeMediaDB.legacy_search_calls == 1
    assert _FakeMediaDB.email_search_calls == 0


def test_media_search_auto_email_delegation_uses_email_planner(media_search_client, monkeypatch):
    monkeypatch.setitem(listing_endpoint.settings, "EMAIL_MEDIA_SEARCH_DELEGATION_MODE", "auto_email")
    resp = media_search_client.post(
        "/api/v1/media/search",
        json={
            "query": "from:alice@example.com",
            "media_types": ["email"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"][0]["id"] == 42
    assert body["items"][0]["type"] == "email"
    assert _FakeMediaDB.email_search_calls == 1
    assert _FakeMediaDB.legacy_search_calls == 0


def test_media_search_auto_email_delegation_honors_explicit_legacy_mode(
    media_search_client,
    monkeypatch,
):
    monkeypatch.setitem(listing_endpoint.settings, "EMAIL_MEDIA_SEARCH_DELEGATION_MODE", "auto_email")
    resp = media_search_client.post(
        "/api/v1/media/search",
        json={
            "query": "from:alice@example.com",
            "media_types": ["email"],
            "email_query_mode": "legacy",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"][0]["id"] == 1
    assert body["items"][0]["type"] == "document"
    assert _FakeMediaDB.legacy_search_calls == 1
    assert _FakeMediaDB.email_search_calls == 0


def test_media_search_auto_email_delegation_falls_back_when_operator_disabled(
    media_search_client,
    monkeypatch,
):
    monkeypatch.setitem(listing_endpoint.settings, "EMAIL_MEDIA_SEARCH_DELEGATION_MODE", "auto_email")
    monkeypatch.setitem(listing_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", False)
    resp = media_search_client.post(
        "/api/v1/media/search",
        json={
            "query": "from:alice@example.com",
            "media_types": ["email"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"][0]["id"] == 1
    assert body["items"][0]["type"] == "document"
    assert _FakeMediaDB.legacy_search_calls == 1
    assert _FakeMediaDB.email_search_calls == 0


def test_media_search_email_operator_mode_requires_email_media_types(media_search_client):
    resp = media_search_client.post(
        "/api/v1/media/search",
        json={
            "query": "from:alice@example.com",
            "media_types": ["document"],
            "email_query_mode": "operators",
        },
    )
    assert resp.status_code == 422


def test_media_search_email_operator_mode_returns_400_for_parse_errors(media_search_client):
    resp = media_search_client.post(
        "/api/v1/media/search",
        json={
            "query": "(budget)",
            "media_types": ["email"],
            "email_query_mode": "operators",
        },
    )
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Parentheses are not supported in email query v1."}


def test_media_search_email_operator_mode_preserves_database_error_detail(
    media_search_client_factory,
    monkeypatch,
):
    fake_logger = MagicMock()
    monkeypatch.setattr(listing_endpoint, "logger", fake_logger)
    client, app = media_search_client_factory(
        _ErroringEmailSearchMediaDB(DatabaseError("driver failed"))
    )
    try:
        resp = client.post(
            "/api/v1/media/search",
            json={
                "query": "from:alice@example.com",
                "media_types": ["email"],
                "email_query_mode": "operators",
            },
        )
    finally:
        client.close()
        app.dependency_overrides.clear()

    assert resp.status_code == 500
    assert resp.json() == {"detail": "A database error occurred during the search."}
    fake_logger.error.assert_called_once_with("Database error during media search")


def test_media_search_email_operator_mode_returns_422_when_flag_disabled(
    media_search_client,
    monkeypatch,
):
    monkeypatch.setitem(listing_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", False)
    resp = media_search_client.post(
        "/api/v1/media/search",
        json={
            "query": "from:alice@example.com",
            "media_types": ["email"],
            "email_query_mode": "operators",
        },
    )
    assert resp.status_code == 422
