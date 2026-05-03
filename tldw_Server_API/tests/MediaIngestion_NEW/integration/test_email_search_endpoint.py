import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import email as email_endpoint
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.config import API_V1_PREFIX
from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError

pytestmark = pytest.mark.integration


class _FakeEmailDB:
    last_search_kwargs: dict | None = None
    last_detail_kwargs: dict | None = None
    last_sync_state_kwargs: list[dict] | None = None
    sync_state_by_source_key: dict[str, dict] = {}

    def search_email_messages(self, *, query=None, limit=50, offset=0, **_kwargs):
        type(self).last_search_kwargs = dict(_kwargs)
        if query and "(" in str(query):
            raise InputError("Parentheses are not supported in email query v1.")
        if query == "multi":
            rows = [
                {
                    "email_message_id": 1,
                    "media_id": 10,
                    "media_title": "Budget Q1",
                    "subject": "Budget Q1",
                },
                {
                    "email_message_id": 2,
                    "media_id": 11,
                    "media_title": "Budget Q2",
                    "subject": "Budget Q2",
                },
            ]
            return rows[offset : offset + limit], 2
        rows = [
            {
                "email_message_id": 1,
                "media_id": 10,
                "media_title": "Budget Q1",
                "subject": "Budget Q1",
            }
        ]
        return rows[offset : offset + limit], 1

    def get_email_message_detail(self, *, email_message_id: int, **_kwargs):
        type(self).last_detail_kwargs = dict(_kwargs)
        if int(email_message_id) == 1:
            return {
                "email_message_id": 1,
                "message_id": "<msg-1@example.com>",
                "source_message_id": "source-msg-1",
                "subject": "Budget Q1",
                "internal_date": "2025-01-10T09:00:00+00:00",
                "body_text": "Budget planning details",
                "has_attachments": True,
                "media": {"id": 10, "uuid": "abc", "url": "email://1", "title": "Budget Q1"},
                "source": {
                    "id": 2,
                    "provider": "upload",
                    "source_key": "upload:inbox",
                    "display_name": "Inbox Upload",
                },
                "participants": {
                    "from": [{"email": "alice@example.com", "display_name": "Alice"}],
                    "to": [{"email": "bob@example.com", "display_name": "Bob"}],
                    "cc": [],
                    "bcc": [],
                },
                "labels": [{"label_key": "Inbox", "label_name": "Inbox"}],
                "attachments": [
                    {
                        "id": 5,
                        "filename": "budget.xlsx",
                        "content_type": "application/vnd.ms-excel",
                        "size_bytes": 128,
                        "content_id": "<cid-5>",
                        "disposition": "attachment",
                        "extracted_text_available": False,
                    }
                ],
                "raw_metadata": {"email": {"subject": "Budget Q1"}},
                "search_text": {
                    "from": "alice@example.com",
                    "to": "bob@example.com",
                    "cc": "",
                    "bcc": "",
                    "labels": "Inbox",
                },
            }
        return None

    def get_email_sync_state(self, *, provider: str, source_key: str, tenant_id=None):
        if type(self).last_sync_state_kwargs is None:
            type(self).last_sync_state_kwargs = []
        type(self).last_sync_state_kwargs.append(
            {
                "provider": provider,
                "source_key": str(source_key),
                "tenant_id": str(tenant_id) if tenant_id is not None else None,
            }
        )
        return type(self).sync_state_by_source_key.get(str(source_key))


@pytest.fixture()
def email_search_client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    _FakeEmailDB.last_search_kwargs = None
    _FakeEmailDB.last_detail_kwargs = None
    _FakeEmailDB.last_sync_state_kwargs = None
    _FakeEmailDB.sync_state_by_source_key = {}
    app = FastAPI()
    app.include_router(email_endpoint.router, prefix=f"{API_V1_PREFIX}/email")
    app.dependency_overrides[get_media_db_for_user] = lambda: _FakeEmailDB()
    app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(
        kind="user",
        user_id=7,
        roles=["admin"],
        permissions=[],
        is_admin=True,
    )

    async def _fake_db_transaction():
        yield object()

    app.dependency_overrides[get_db_transaction] = _fake_db_transaction
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_email_search_endpoint_returns_paginated_payload(email_search_client):
    response = email_search_client.get("/api/v1/email/search", params={"q": "budget"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "items" in body and isinstance(body["items"], list)
    assert "pagination" in body and isinstance(body["pagination"], dict)
    assert body["pagination"]["total"] == 1


def test_email_search_endpoint_includes_canonical_offset_pagination(email_search_client) -> None:
    response = email_search_client.get("/api/v1/email/search", params={"q": "multi", "limit": 1})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pagination"]["mode"] == "offset"
    assert body["pagination"]["has_more"] is True
    assert body["pagination"]["next_offset"] == 1
    assert body["has_more"] is True
    assert body["next_offset"] == 1


def test_email_search_endpoint_returns_400_on_parse_error(email_search_client):
    response = email_search_client.get("/api/v1/email/search", params={"q": "(budget)"})
    assert response.status_code == 400


def test_email_detail_endpoint_returns_message_graph(email_search_client):
    response = email_search_client.get("/api/v1/email/messages/1")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["email_message_id"] == 1
    assert body["participants"]["from"][0]["email"] == "alice@example.com"
    assert body["labels"][0]["label_name"] == "Inbox"
    assert body["attachments"][0]["filename"] == "budget.xlsx"


def test_email_detail_endpoint_returns_404_when_missing(email_search_client):
    response = email_search_client.get("/api/v1/email/messages/999")
    assert response.status_code == 404


def test_email_search_endpoint_returns_404_when_operator_search_flag_disabled(
    email_search_client,
    monkeypatch,
):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", False)
    response = email_search_client.get("/api/v1/email/search", params={"q": "budget"})
    assert response.status_code == 404


def test_email_detail_endpoint_returns_404_when_operator_search_flag_disabled(
    email_search_client,
    monkeypatch,
):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", False)
    response = email_search_client.get("/api/v1/email/messages/1")
    assert response.status_code == 404


def test_email_search_endpoint_passes_include_deleted_false(email_search_client):
    response = email_search_client.get("/api/v1/email/search", params={"q": "budget"})
    assert response.status_code == 200
    assert _FakeEmailDB.last_search_kwargs == {"include_deleted": False}


def test_email_detail_endpoint_passes_include_deleted_false(email_search_client):
    response = email_search_client.get("/api/v1/email/messages/1")
    assert response.status_code == 200
    assert _FakeEmailDB.last_detail_kwargs == {"include_deleted": False}


def test_email_sources_endpoint_returns_sync_status_states(email_search_client, monkeypatch):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", True)

    async def _fake_list_connector_sources(_db, user_id):
        assert user_id == 7
        return [
            {
                "id": 11,
                "account_id": 1,
                "provider": "gmail",
                "remote_id": "INBOX",
                "type": "label",
                "path": "INBOX",
                "options": {"query": "is:unread"},
                "enabled": True,
                "last_synced_at": "2026-02-10T10:00:00+00:00",
            },
            {
                "id": 22,
                "account_id": 1,
                "provider": "gmail",
                "remote_id": "Label_123",
                "type": "label",
                "path": "Projects",
                "options": {},
                "enabled": True,
                "last_synced_at": None,
            },
            {
                "id": 33,
                "account_id": 1,
                "provider": "gmail",
                "remote_id": "Label_456",
                "type": "label",
                "path": "Alerts",
                "options": {},
                "enabled": False,
                "last_synced_at": None,
            },
            {
                "id": 44,
                "account_id": 1,
                "provider": "drive",
                "remote_id": "root",
                "type": "folder",
                "path": "/",
                "options": {},
                "enabled": True,
                "last_synced_at": None,
            },
        ]

    monkeypatch.setattr(email_endpoint, "list_connector_sources", _fake_list_connector_sources)
    _FakeEmailDB.sync_state_by_source_key = {
        "11": {
            "cursor": "1001",
            "last_run_at": "2026-02-10T09:59:00+00:00",
            "last_success_at": "2026-02-10T09:59:30+00:00",
            "error_state": None,
            "retry_backoff_count": 0,
            "updated_at": "2026-02-10T09:59:30+00:00",
        },
        "22": {
            "cursor": "1002",
            "last_run_at": "2026-02-10T10:05:00+00:00",
            "last_success_at": "2026-02-10T09:00:00+00:00",
            "error_state": "quota_exceeded",
            "retry_backoff_count": 2,
            "updated_at": "2026-02-10T10:05:00+00:00",
        },
        "33": {
            "cursor": "1003",
            "last_run_at": "2026-02-10T10:10:00+00:00",
            "last_success_at": None,
            "error_state": "invalid_history_id",
            "retry_backoff_count": 0,
            "updated_at": "2026-02-10T10:10:00+00:00",
        },
    }

    response = email_search_client.get("/api/v1/email/sources")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 3
    items_by_id = {int(item["id"]): item for item in body["items"]}
    assert 44 not in items_by_id
    assert items_by_id[11]["sync"]["state"] == "healthy"
    assert items_by_id[22]["sync"]["state"] == "retrying"
    assert items_by_id[33]["sync"]["state"] == "failed"
    assert sorted(
        (entry["provider"], entry["source_key"], entry["tenant_id"])
        for entry in (_FakeEmailDB.last_sync_state_kwargs or [])
    ) == [
        ("gmail", "11", "7"),
        ("gmail", "22", "7"),
        ("gmail", "33", "7"),
    ]


def test_email_sources_endpoint_returns_404_when_sync_flag_disabled(
    email_search_client,
    monkeypatch,
):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", False)
    response = email_search_client.get("/api/v1/email/sources")
    assert response.status_code == 404


def test_email_source_sync_endpoint_queues_job(email_search_client, monkeypatch):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", True)
    captured: dict[str, object] = {}

    async def _fake_get_source_by_id(_db, user_id, source_id):
        captured["source_lookup"] = {"user_id": user_id, "source_id": source_id}
        return {
            "id": source_id,
            "provider": "gmail",
            "remote_id": "INBOX",
            "enabled": True,
        }

    async def _fake_create_import_job(user_id, source_id, *, request_id=None):
        captured["job_create"] = {
            "user_id": user_id,
            "source_id": source_id,
            "request_id": request_id,
        }
        return {
            "id": "job-123",
            "source_id": source_id,
            "type": "import",
            "status": "queued",
            "progress_pct": 0,
            "counts": {"processed": 0, "skipped": 0, "failed": 0},
        }

    monkeypatch.setattr(email_endpoint, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(email_endpoint, "create_import_job", _fake_create_import_job)

    response = email_search_client.post("/api/v1/email/sources/22/sync")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_id"] == 22
    assert body["provider"] == "gmail"
    assert body["status"] == "queued"
    assert body["job"]["id"] == "job-123"
    assert captured["source_lookup"] == {"user_id": 7, "source_id": 22}
    assert captured["job_create"]["user_id"] == 7
    assert captured["job_create"]["source_id"] == 22


def test_email_source_sync_endpoint_returns_404_for_missing_source(
    email_search_client,
    monkeypatch,
):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", True)

    async def _fake_get_source_by_id(_db, _user_id, _source_id):
        return None

    monkeypatch.setattr(email_endpoint, "get_source_by_id", _fake_get_source_by_id)
    response = email_search_client.post("/api/v1/email/sources/999/sync")
    assert response.status_code == 404


def test_email_source_sync_endpoint_returns_400_for_non_gmail_source(
    email_search_client,
    monkeypatch,
):
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", True)

    async def _fake_get_source_by_id(_db, _user_id, source_id):
        return {
            "id": source_id,
            "provider": "drive",
            "remote_id": "root",
            "enabled": True,
        }

    monkeypatch.setattr(email_endpoint, "get_source_by_id", _fake_get_source_by_id)
    response = email_search_client.post("/api/v1/email/sources/12/sync")
    assert response.status_code == 400
