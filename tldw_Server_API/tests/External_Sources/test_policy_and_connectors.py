import asyncio
import aiosqlite
import base64
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest


@pytest.mark.unit
def test_policy_is_file_type_allowed_cases():
    from tldw_Server_API.app.core.External_Sources.policy import is_file_type_allowed

    # Empty allowlist allows all
    assert is_file_type_allowed(name="doc.pdf", mime="application/pdf", allowed=None) is True
    # Extension
    assert is_file_type_allowed(name="notes.md", mime="text/markdown", allowed=["md"]) is True
    assert is_file_type_allowed(name="notes.txt", mime="text/plain", allowed=["md"]) is False
    # Dot extension
    assert is_file_type_allowed(name="a.txt", mime="text/plain", allowed=[".txt"]) is True
    # Full mime
    assert is_file_type_allowed(name="a.bin", mime="application/pdf", allowed=["application/pdf"]) is True
    # Mime prefix
    assert is_file_type_allowed(name="a.txt", mime="text/plain", allowed=["text/"]) is True
    assert is_file_type_allowed(name="a.bin", mime="application/octet-stream", allowed=["text/"]) is False


@pytest.mark.unit
def test_policy_fail_closed_on_error(monkeypatch):
    import tldw_Server_API.app.core.External_Sources.policy as policy_mod

    class _BadProvider:
        def __str__(self):
            raise RuntimeError("policy backend exploded /private/policy.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(policy_mod, "logger", fake_logger)

    ok, reason = policy_mod.evaluate_policy_constraints({"enabled_providers": [_BadProvider()]}, provider="drive")
    assert ok is False
    assert reason == "Policy evaluation failed"
    fake_logger.warning.assert_called_once_with("Policy evaluation error")


@pytest.mark.unit
def test_default_connector_policy_enables_zotero_by_default(monkeypatch):
    from tldw_Server_API.app.core.External_Sources.policy import get_default_policy_from_env

    monkeypatch.delenv("ORG_CONNECTORS_ENABLED_PROVIDERS", raising=False)
    monkeypatch.delenv("EMAIL_GMAIL_CONNECTOR_ENABLED", raising=False)

    policy = get_default_policy_from_env(org_id=7)

    assert policy["org_id"] == 7
    assert policy["enabled_providers"] == ["drive", "notion", "zotero"]


@pytest.mark.unit
def test_connector_policy_schema_defaults_include_zotero():
    from tldw_Server_API.app.api.v1.schemas.connectors import ConnectorPolicy

    policy = ConnectorPolicy(org_id=9)

    assert policy.enabled_providers == ["drive", "notion", "zotero"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_finish_source_sync_job_clears_failure_and_rescan_flags_on_success(tmp_path: Path):
    import tldw_Server_API.app.core.External_Sources.connectors_service as svc

    db = await aiosqlite.connect(tmp_path / "connectors-sync.sqlite3")
    db.row_factory = aiosqlite.Row
    db._is_sqlite = True
    try:
        account = await svc.create_account(
            db,
            user_id=1,
            provider="drive",
            display_name="Drive",
            email="owner@example.com",
            tokens={"access_token": "tok"},
        )
        source = await svc.create_source(
            db,
            account_id=account["id"],
            provider="drive",
            remote_id="root",
            type_="folder",
            path="/",
            options={"recursive": True},
        )
        await svc.upsert_source_sync_state(
            db,
            source_id=source["id"],
            sync_mode="hybrid",
            last_sync_failed_at="2026-03-01 00:00:00",
            last_error="cursor invalid",
            retry_backoff_count=2,
            needs_full_rescan=True,
        )
        await svc.start_source_sync_job(db, source_id=source["id"], job_id="job-1")

        state = await svc.finish_source_sync_job(
            db,
            source_id=source["id"],
            job_id="job-1",
            outcome="success",
        )

        assert state["last_sync_failed_at"] is None
        assert state["last_error"] is None
        assert state["retry_backoff_count"] == 0
        assert state["needs_full_rescan"] is False
    finally:
        await db.close()


@pytest.mark.unit
def test_gmail_body_parser_prefers_plain_text_and_skips_attachment_parts():
    import tldw_Server_API.app.services.connectors_worker as worker

    plain = base64.urlsafe_b64encode(b"Plain body text").decode().rstrip("=")
    html = base64.urlsafe_b64encode(b"<p>HTML body text</p>").decode().rstrip("=")
    attachment_text = base64.urlsafe_b64encode(b"Attachment text should not appear").decode().rstrip("=")
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": plain}},
            {"mimeType": "text/html", "body": {"data": html}},
            {
                "mimeType": "text/plain",
                "filename": "notes.txt",
                "body": {"data": attachment_text, "attachmentId": "a1", "size": 42},
                "headers": [{"name": "Content-Disposition", "value": "attachment; filename=notes.txt"}],
            },
        ],
    }

    content = worker._collect_gmail_body_text(payload)
    attachments = worker._collect_gmail_attachments(payload)

    assert content == "Plain body text"
    assert len(attachments) == 1
    assert attachments[0]["filename"] == "notes.txt"


@pytest.mark.unit
def test_gmail_body_parser_handles_html_only_and_root_data_fallback():
    import tldw_Server_API.app.services.connectors_worker as worker

    html_only = base64.urlsafe_b64encode(
        b"<html><body><h1>Title</h1><p>Body text</p></body></html>"
    ).decode().rstrip("=")
    root_plain = base64.urlsafe_b64encode(b"Root body fallback").decode().rstrip("=")

    html_payload = {"mimeType": "text/html", "body": {"data": html_only}}
    root_payload = {"mimeType": "message/rfc822", "body": {"data": root_plain}}

    assert worker._collect_gmail_body_text(html_payload) == "Title Body text"
    assert worker._collect_gmail_body_text(root_payload) == "Root body fallback"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_gmail_connector_list_history_flattens_delta_events(monkeypatch):
    from tldw_Server_API.app.core.External_Sources.gmail import GmailConnector

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

        async def aclose(self):
            return None

    payload = {
        "history": [
            {
                "id": "120",
                "messagesAdded": [
                    {"message": {"id": "m1", "threadId": "t1"}},
                    {"message": {"id": "m1", "threadId": "t1"}},
                ],
                "messagesDeleted": [
                    {"message": {"id": "deleted-1", "threadId": "td"}}
                ],
            },
            {
                "id": "125",
                "labelsAdded": [
                    {
                        "message": {"id": "m2", "threadId": "t2"},
                        "labelIds": ["INBOX", "STARRED"],
                    }
                ],
                "labelsRemoved": [
                    {
                        "message": {"id": "m2", "threadId": "t2"},
                        "labelIds": ["UNREAD"],
                    }
                ],
            },
        ],
        "nextPageToken": "next-1",
        "historyId": "130",
    }

    async def _fake_afetch(*, method, url, headers=None, params=None, timeout=None):
        assert method == "GET"
        assert "gmail/v1/users/me/history" in url
        assert params["startHistoryId"] == "100"
        return _Resp(payload)

    import tldw_Server_API.app.core.External_Sources.gmail as gmail_mod

    monkeypatch.setattr(gmail_mod, "afetch", _fake_afetch)

    conn = GmailConnector(client_id="x", client_secret="y", redirect_base="http://localhost")
    items, next_cursor, latest_history_id = await conn.list_history(
        {"tokens": {"access_token": "tok"}},
        start_history_id="100",
        label_id="INBOX",
        page_size=50,
        cursor="c0",
    )

    assert [item["id"] for item in items] == ["m1", "deleted-1", "m2"]
    assert items[0]["message_added"] is True
    assert items[0]["message_deleted"] is False
    assert items[1]["message_deleted"] is True
    assert items[2]["labels_added"] == ["INBOX", "STARRED"]
    assert items[2]["labels_removed"] == ["UNREAD"]
    assert next_cursor == "next-1"
    assert latest_history_id == "130"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_gmail_connector_list_history_handles_malformed_rows_and_pagination_params(monkeypatch):
    from tldw_Server_API.app.core.External_Sources.gmail import GmailConnector

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

        async def aclose(self):
            return None

    observed_params: dict[str, Any] = {}
    payload = {
        "history": [
            "malformed-row",
            {
                "id": "203",
                "messagesAdded": [
                    None,
                    {"message": {"threadId": "missing-id"}},
                ],
                "labelsAdded": [
                    {
                        "message": {"id": "m-net", "threadId": "t-net"},
                        "labelIds": ["INBOX", "STARRED", "STARRED"],
                    }
                ],
                "labelsRemoved": [
                    {
                        "message": {"id": "m-net", "threadId": "t-net"},
                        "labelIds": ["INBOX", "TRASH"],
                    }
                ],
            },
            {
                "historyId": "204",
                "messages": [
                    {"id": "m-fallback", "threadId": "t-fallback"},
                    {"threadId": "missing-id"},
                ],
            },
        ],
        "nextPageToken": "next-2",
        "historyId": "205",
    }

    async def _fake_afetch(*, method, url, headers=None, params=None, timeout=None):
        assert method == "GET"
        assert "gmail/v1/users/me/history" in url
        observed_params.update(dict(params or {}))
        return _Resp(payload)

    import tldw_Server_API.app.core.External_Sources.gmail as gmail_mod

    monkeypatch.setattr(gmail_mod, "afetch", _fake_afetch)

    conn = GmailConnector(client_id="x", client_secret="y", redirect_base="http://localhost")
    items, next_cursor, latest_history_id = await conn.list_history(
        {"tokens": {"access_token": "tok"}},
        start_history_id="200",
        label_id="INBOX",
        page_size=999,
        cursor="c-2",
    )

    assert observed_params["startHistoryId"] == "200"
    assert observed_params["labelId"] == "INBOX"
    assert observed_params["pageToken"] == "c-2"
    assert observed_params["maxResults"] == 500
    assert [item["id"] for item in items] == ["m-net", "m-fallback"]
    assert items[0]["labels_added"] == ["STARRED"]
    assert items[0]["labels_removed"] == ["TRASH"]
    assert items[1]["message_added"] is True
    assert items[1]["message_deleted"] is False
    assert next_cursor == "next-2"
    assert latest_history_id == "205"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_gmail_connector_list_messages_sanitizes_payload_and_clamps_pagination(monkeypatch):
    from tldw_Server_API.app.core.External_Sources.gmail import GmailConnector

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

        async def aclose(self):
            return None

    observed_params: dict[str, Any] = {}
    payload = {
        "messages": [
            {"id": "m1", "threadId": "t1"},
            {"id": "   ", "threadId": "t-whitespace"},
            {"threadId": "missing-id"},
            "bad-row",
            {"id": "m2"},
        ],
        "nextPageToken": "next-page",
    }

    async def _fake_afetch(*, method, url, headers=None, params=None, timeout=None):
        assert method == "GET"
        assert "gmail/v1/users/me/messages" in url
        observed_params.update(dict(params or {}))
        return _Resp(payload)

    import tldw_Server_API.app.core.External_Sources.gmail as gmail_mod

    monkeypatch.setattr(gmail_mod, "afetch", _fake_afetch)

    conn = GmailConnector(client_id="x", client_secret="y", redirect_base="http://localhost")
    items, next_cursor = await conn.list_messages(
        {"tokens": {"access_token": "tok"}},
        label_id="INBOX",
        page_size=0,
        cursor="c-1",
        query="from:alice@example.com",
    )

    assert observed_params["labelIds"] == "INBOX"
    assert observed_params["pageToken"] == "c-1"
    assert observed_params["q"] == "from:alice@example.com"
    assert observed_params["maxResults"] == 1
    assert items == [
        {"id": "m1", "threadId": "t1"},
        {"id": "m2", "threadId": None},
    ]
    assert next_cursor == "next-page"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_notion_download_renders_nested_blocks(monkeypatch):
    from tldw_Server_API.app.core.External_Sources.notion import NotionConnector

    # Fake afetch response for Notion blocks children
    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

        async def aclose(self):
            return None

    # Construct a minimal realistic Notion blocks payload
    page_blocks = {
        "results": [
            {"object": "block", "id": "h1", "type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Title"}]}},
            {"object": "block", "id": "li1", "type": "bulleted_list_item", "has_children": True,
                "bulleted_list_item": {"rich_text": [{"plain_text": "Item"}]}},
            {"object": "block", "id": "tbl1", "type": "table", "has_children": True, "table": {"table_width": 2}},
        ],
        "has_more": False,
        "next_cursor": None,
    }

    async def _fake_afetch(*, method, url, headers=None, params=None, json=None, timeout=None):
        # Children for list item and table rows
        if url.endswith("/blocks/li1/children"):
            # One code child
            payload = {"results": [{"object": "block", "id": "c1", "type": "code", "code": {"language": "python", "rich_text": [{"plain_text": "print('x')"}]}}], "has_more": False, "next_cursor": None}
        elif url.endswith("/blocks/tbl1/children"):
            # First row is header, second is value
            payload = {
                "results": [
                    {"object": "block", "type": "table_row", "table_row": {"cells": [[{"plain_text": "H1"}], [{"plain_text": "H2"}]]}},
                    {"object": "block", "type": "table_row", "table_row": {"cells": [[{"plain_text": "V1"}], [{"plain_text": "V2"}]]}},
                ],
                "has_more": False,
                "next_cursor": None,
            }
        else:
            payload = page_blocks
        return _Resp(payload)

    # Patch afetch in module
    import tldw_Server_API.app.core.External_Sources.notion as notion_mod
    monkeypatch.setattr(notion_mod, "afetch", _fake_afetch)

    nc = NotionConnector(client_id="x", client_secret="y", redirect_base="http://localhost")
    md_bytes = await nc.download_file({"tokens": {"access_token": "tok"}}, file_id="page123")
    out = md_bytes.decode("utf-8")

    # Heading
    assert "# Title" in out
    # Bullet + code block
    assert "- Item" in out
    assert "```python" in out and "print('x')" in out
    # Table header row and separator
    assert "| H1 | H2 |" in out
    assert "| --- | --- |" in out
    # Table body
    assert "| V1 | V2 |" in out


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_drive_recursive_traversal(monkeypatch):
    # Arrange worker and patch dependencies to avoid real DB/network
    import tldw_Server_API.app.services.connectors_worker as worker

    class FakeJM:
        def __init__(self):
            self.completed = None
            self.renewed = []

        def renew_job_lease(self, *a, **kw):
            self.renewed.append((a, kw))

        def complete_job(self, jid, result=None, worker_id=None, lease_id=None, completion_token=None):
            self.completed = {"jid": jid, "result": result}

    # Fake connector that returns a folder structure:
    # root -> [folder F, doc A]; F -> [doc B]
    class FakeDriveConn:
        name = "drive"
        def authorize_url(self, *a, **kw):
            return ""
        async def exchange_code(self, *a, **kw):
            return {}
        async def list_files(self, account, parent_remote_id, *, page_size=50, cursor=None):
            if parent_remote_id in ("root", "r1"):
                return ([{"id": "F", "name": "Folder", "mimeType": "application/vnd.google-apps.folder", "is_folder": True},
                        {"id": "A", "name": "DocA.txt", "mimeType": "text/plain", "size": 10}], None)
            if parent_remote_id == "F":
                return ([{"id": "B", "name": "DocB.txt", "mimeType": "text/plain", "size": 12}], None)
            return ([], None)
        async def download_file(self, account, file_id, *, mime_type=None, export_mime=None):
            return f"content-{file_id}".encode()

    # Patch connector lookup on the package that worker imports from
    import tldw_Server_API.app.core.External_Sources as ext_pkg
    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeDriveConn())

    # Patch DB-related functions referenced inside worker module
    async def _fake_get_source_by_id(db, user_id, source_id):
        return {"id": source_id, "provider": "drive", "account_id": 123, "remote_id": "r1", "type": "folder", "path": "/", "options": {"recursive": True}}

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_should_ingest_item(db, **kwargs):
        return True

    async def _fake_record_ingested_item(db, **kwargs):
        return None

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc
    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "should_ingest_item", _fake_should_ingest_item)
    monkeypatch.setattr(svc, "record_ingested_item", _fake_record_ingested_item)

    # Patch DB pool transaction context manager to yield a dummy object
    class _DummyTx:
        async def __aenter__(self):
            return object()
        async def __aexit__(self, exc_type, exc, tb):
            return False
    class _DummyPool:
        def transaction(self):
            return _DummyTx()
    async def _fake_get_db_pool():
        return _DummyPool()
    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    # Avoid org membership DB access by returning empty list
    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    async def _fake_list_memberships_for_user(user_id: int):
        return []
    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    # Patch Media DB write
    class _FakeMDB:
        def __init__(self, *a, **kw):
            self.records = []
        def add_media_with_keywords(self, *, url, title, media_type, content, keywords, overwrite=False):
            self.records.append((url, title, content))
            return 1, "uuid", "ok"
    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    # Policy defaults in env (allow txt)
    os.environ.setdefault("ORG_CONNECTORS_ALLOWED_EXPORT_FORMATS", "md,txt,pdf")

    jm = FakeJM()
    # Act
    await worker._process_import_job(jm, jid=1, lease_id="L", worker_id="W", source_id=99, user_id=42)

    # Assert: processed 2 documents (A, B), total includes folder + docs (3)
    assert jm.completed is not None
    res = jm.completed["result"]
    assert res["processed"] == 2
    assert res["total"] == 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_drive_non_recursive_paginates(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            pass

        def complete_job(self, jid, result=None, worker_id=None, lease_id=None, completion_token=None):
            self.completed = {"jid": jid, "result": result}

    class FakeDriveConn:
        name = "drive"
        def authorize_url(self, *a, **kw):
            return ""
        async def exchange_code(self, *a, **kw):
            return {}
        async def list_files(self, account, parent_remote_id, *, page_size=50, cursor=None):
            if cursor is None:
                return ([{"id": "A", "name": "DocA.txt", "mimeType": "text/plain", "size": 10}], "c1")
            if cursor == "c1":
                return ([{"id": "B", "name": "DocB.txt", "mimeType": "text/plain", "size": 12}], None)
            return ([], None)
        async def download_file(self, account, file_id, *, mime_type=None, export_mime=None):
            return f"content-{file_id}".encode()

    import tldw_Server_API.app.core.External_Sources as ext_pkg
    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeDriveConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {"id": source_id, "provider": "drive", "account_id": 123, "remote_id": "root", "type": "folder", "path": "/", "options": {"recursive": False}}

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_should_ingest_item(db, **kwargs):
        return True

    async def _fake_record_ingested_item(db, **kwargs):
        return None

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc
    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "should_ingest_item", _fake_should_ingest_item)
    monkeypatch.setattr(svc, "record_ingested_item", _fake_record_ingested_item)

    class _DummyTx:
        async def __aenter__(self):
            return object()
        async def __aexit__(self, exc_type, exc, tb):
            return False
    class _DummyPool:
        def transaction(self):
            return _DummyTx()
    async def _fake_get_db_pool():
        return _DummyPool()
    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    async def _fake_list_memberships_for_user(user_id: int):
        return []
    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FakeMDB:
        def __init__(self, *a, **kw):
            self.records = []
        def add_media_with_keywords(self, *, url, title, media_type, content, keywords, overwrite=False):
            self.records.append((url, title, content))
            return 1, "uuid", "ok"
    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm = FakeJM()
    await worker._process_import_job(jm, jid=1, lease_id="L", worker_id="W", source_id=99, user_id=42)

    assert jm.completed is not None
    res = jm.completed["result"]
    assert res["processed"] == 2
    assert res["total"] == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_skips_record_on_ingest_failure(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            pass

        def complete_job(self, jid, result=None, worker_id=None, lease_id=None, completion_token=None):
            self.completed = {"jid": jid, "result": result}

    class FakeDriveConn:
        name = "drive"
        def authorize_url(self, *a, **kw):
            return ""
        async def exchange_code(self, *a, **kw):
            return {}
        async def list_files(self, account, parent_remote_id, *, page_size=50, cursor=None):
            return ([{"id": "A", "name": "DocA.txt", "mimeType": "text/plain", "size": 10}], None)
        async def download_file(self, account, file_id, *, mime_type=None, export_mime=None):
            return b"content-A"

    import tldw_Server_API.app.core.External_Sources as ext_pkg
    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeDriveConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {"id": source_id, "provider": "drive", "account_id": 123, "remote_id": "root", "type": "folder", "path": "/", "options": {"recursive": False}}

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_should_ingest_item(db, **kwargs):
        return True

    recorded: list = []
    async def _fake_record_ingested_item(db, **kwargs):
        recorded.append(kwargs)

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc
    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "should_ingest_item", _fake_should_ingest_item)
    monkeypatch.setattr(svc, "record_ingested_item", _fake_record_ingested_item)

    class _DummyTx:
        async def __aenter__(self):
            return object()
        async def __aexit__(self, exc_type, exc, tb):
            return False
    class _DummyPool:
        def transaction(self):
            return _DummyTx()
    async def _fake_get_db_pool():
        return _DummyPool()
    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    async def _fake_list_memberships_for_user(user_id: int):
        return []
    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FailMDB:
        def __init__(self, *a, **kw):
            pass
        def add_media_with_keywords(self, *, url, title, media_type, content, keywords, overwrite=False):
            raise RuntimeError("ingest failed")
    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FailMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm = FakeJM()
    await worker._process_import_job(jm, jid=1, lease_id="L", worker_id="W", source_id=99, user_id=42)

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 0
    assert recorded == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_gmail_initial_backfill_upserts_email_graph(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            pass

        def complete_job(self, jid, result=None, worker_id=None, lease_id=None, completion_token=None):
            self.completed = {"jid": jid, "result": result}

    plain_body = base64.urlsafe_b64encode(b"Hello from Gmail").decode().rstrip("=")

    class FakeGmailConn:
        name = "gmail"

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_messages(self, account, *, label_id=None, page_size=100, cursor=None, query=None):
            return ([{"id": "m1", "threadId": "t1"}], None)

        async def get_message(self, account, *, message_id, format="full"):
            return {
                "id": message_id,
                "historyId": "7",
                "internalDate": "1736509200000",
                "labelIds": ["INBOX", "IMPORTANT"],
                "snippet": "Fallback snippet",
                "payload": {
                    "mimeType": "multipart/alternative",
                    "headers": [
                        {"name": "Subject", "value": "Budget Update"},
                        {"name": "From", "value": "Alice <alice@example.com>"},
                        {"name": "To", "value": "Bob <bob@example.com>"},
                        {"name": "Message-ID", "value": "<m1@example.com>"},
                        {"name": "Date", "value": "Fri, 10 Jan 2025 10:00:00 +0000"},
                    ],
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {"data": plain_body},
                        }
                    ],
                },
            }

    import tldw_Server_API.app.core.External_Sources as ext_pkg

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeGmailConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "gmail",
            "account_id": 123,
            "remote_id": "INBOX",
            "type": "folder",
            "path": "/inbox",
            "options": {"query": "in:inbox"},
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_should_ingest_item(db, **kwargs):
        return True

    recorded: list[dict] = []

    async def _fake_record_ingested_item(db, **kwargs):
        recorded.append(kwargs)

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc

    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "should_ingest_item", _fake_should_ingest_item)
    monkeypatch.setattr(svc, "record_ingested_item", _fake_record_ingested_item)

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    import tldw_Server_API.app.core.AuthNZ.database as dbmod

    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs

    async def _fake_list_memberships_for_user(user_id: int):
        return []

    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FakeMDB:
        instances: list["_FakeMDB"] = []

        def __init__(self, *a, **kw):
            self.add_calls: list[dict] = []
            self.upsert_calls: list[dict] = []
            type(self).instances.append(self)

        def add_media_with_keywords(self, **kwargs):
            self.add_calls.append(kwargs)
            return 1, "uuid", "ok"

        def upsert_email_message_graph(self, **kwargs):
            self.upsert_calls.append(kwargs)
            return {"email_message_id": 1}

    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm = FakeJM()
    await worker._process_import_job(jm, jid=1, lease_id="L", worker_id="W", source_id=99, user_id=42)

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 1
    assert jm.completed["result"]["total"] == 1
    assert _FakeMDB.instances
    assert _FakeMDB.instances[0].add_calls
    assert _FakeMDB.instances[0].add_calls[0]["media_type"] == "email"
    assert _FakeMDB.instances[0].upsert_calls
    assert _FakeMDB.instances[0].upsert_calls[0]["source_message_id"] == "m1"
    assert recorded and recorded[0]["provider"] == "gmail"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_gmail_incremental_sync_advances_cursor_and_processes_only_deltas(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            return None

        def complete_job(self, jid, result=None, worker_id=None, lease_id=None, completion_token=None):
            self.completed = {"jid": jid, "result": result}

    def _b64(value: str) -> str:
        return base64.urlsafe_b64encode(value.encode("utf-8")).decode().rstrip("=")

    class FakeGmailConn:
        name = "gmail"
        run_number = 0
        list_messages_calls = 0
        list_history_starts: list[str] = []

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_messages(self, account, *, label_id=None, page_size=100, cursor=None, query=None):
            type(self).list_messages_calls += 1
            if type(self).run_number == 0:
                return (
                    [
                        {"id": "m1", "threadId": "t1"},
                        {"id": "m2", "threadId": "t2"},
                    ],
                    None,
                )
            raise AssertionError("incremental cycle should use list_history, not list_messages")

        async def list_history(
            self,
            account,
            *,
            start_history_id,
            label_id=None,
            page_size=100,
            cursor=None,
        ):
            type(self).list_history_starts.append(str(start_history_id))
            return ([{"id": "m3", "threadId": "t3"}], None, "30")

        async def get_message(self, account, *, message_id, format="full"):
            history_by_id = {"m1": "10", "m2": "20", "m3": "30"}
            return {
                "id": message_id,
                "historyId": history_by_id[message_id],
                "internalDate": "1736509200000",
                "labelIds": ["INBOX"],
                "snippet": f"Snippet {message_id}",
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "Subject", "value": f"Subject {message_id}"},
                        {"name": "From", "value": "Alice <alice@example.com>"},
                        {"name": "To", "value": "Bob <bob@example.com>"},
                        {"name": "Message-ID", "value": f"<{message_id}@example.com>"},
                        {"name": "Date", "value": "Fri, 10 Jan 2025 10:00:00 +0000"},
                    ],
                    "body": {"data": _b64(f"Body {message_id}")},
                },
            }

    import tldw_Server_API.app.core.External_Sources as ext_pkg

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeGmailConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "gmail",
            "account_id": 123,
            "remote_id": "INBOX",
            "type": "folder",
            "path": "/inbox",
            "options": {"query": "in:inbox"},
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_should_ingest_item(db, **kwargs):
        return True

    recorded_external_ids: list[str] = []

    async def _fake_record_ingested_item(db, **kwargs):
        recorded_external_ids.append(str(kwargs.get("external_id")))

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc

    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "should_ingest_item", _fake_should_ingest_item)
    monkeypatch.setattr(svc, "record_ingested_item", _fake_record_ingested_item)

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    import tldw_Server_API.app.core.AuthNZ.database as dbmod

    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs

    async def _fake_list_memberships_for_user(user_id: int):
        return []

    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FakeMDB:
        sync_state: dict[tuple[str, str, str], dict] = {}

        def __init__(self, *a, **kw):
            pass

        def _key(self, *, provider, source_key, tenant_id):
            return (str(tenant_id), str(provider), str(source_key))

        def get_email_sync_state(self, *, provider, source_key, tenant_id=None):
            row = type(self).sync_state.get(self._key(provider=provider, source_key=source_key, tenant_id=tenant_id))
            return dict(row) if row else None

        def mark_email_sync_run_started(self, *, provider, source_key, tenant_id=None, cursor=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row.setdefault("cursor", None)
            if cursor is not None:
                row["cursor"] = cursor
            row["tenant_id"] = key[0]
            row["provider"] = key[1]
            row["source_key"] = key[2]
            row["source_id"] = 1
            row["last_run_at"] = "started"
            row.setdefault("retry_backoff_count", 0)
            type(self).sync_state[key] = row
            return dict(row)

        def mark_email_sync_run_succeeded(self, *, provider, source_key, tenant_id=None, cursor=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            if cursor is not None:
                row["cursor"] = cursor
            row["tenant_id"] = key[0]
            row["provider"] = key[1]
            row["source_key"] = key[2]
            row["source_id"] = 1
            row["last_success_at"] = "succeeded"
            row["error_state"] = None
            row["retry_backoff_count"] = 0
            type(self).sync_state[key] = row
            return dict(row)

        def mark_email_sync_run_failed(self, *, provider, source_key, tenant_id=None, error_state=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["tenant_id"] = key[0]
            row["provider"] = key[1]
            row["source_key"] = key[2]
            row["source_id"] = 1
            row["error_state"] = str(error_state or "failed")
            row["retry_backoff_count"] = int(row.get("retry_backoff_count") or 0) + 1
            type(self).sync_state[key] = row
            return dict(row)

        def add_media_with_keywords(self, **kwargs):
            return 1, "uuid", "ok"

        def upsert_email_message_graph(self, **kwargs):
            return {"email_message_id": 1}

    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm_first = FakeJM()
    FakeGmailConn.run_number = 0
    await worker._process_import_job(
        jm_first,
        jid=1,
        lease_id="L1",
        worker_id="W",
        source_id=99,
        user_id=42,
    )

    jm_second = FakeJM()
    FakeGmailConn.run_number = 1
    await worker._process_import_job(
        jm_second,
        jid=2,
        lease_id="L2",
        worker_id="W",
        source_id=99,
        user_id=42,
    )

    assert jm_first.completed is not None
    assert jm_first.completed["result"]["processed"] == 2
    assert jm_second.completed is not None
    assert jm_second.completed["result"]["processed"] == 1

    assert FakeGmailConn.list_messages_calls == 1
    assert FakeGmailConn.list_history_starts == ["20"]
    assert recorded_external_ids == ["m1", "m2", "m3"]

    final_state = _FakeMDB.sync_state[("42", "gmail", "99")]
    assert final_state["cursor"] == "30"
    assert final_state["error_state"] is None
    assert final_state["retry_backoff_count"] == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_gmail_invalid_cursor_uses_bounded_replay_and_recovers(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    monkeypatch.setenv("EMAIL_SYNC_CURSOR_RECOVERY_WINDOW_DAYS", "7")
    monkeypatch.setenv("EMAIL_SYNC_CURSOR_RECOVERY_MAX_MESSAGES", "25")

    metric_increments: list[dict[str, Any]] = []
    metric_observations: list[dict[str, Any]] = []

    def _fake_metric_increment(metric_name, *, labels, value=1):
        metric_increments.append(
            {
                "metric_name": str(metric_name),
                "labels": dict(labels or {}),
                "value": value,
            }
        )

    def _fake_metric_observe(metric_name, value, *, labels):
        metric_observations.append(
            {
                "metric_name": str(metric_name),
                "labels": dict(labels or {}),
                "value": float(value),
            }
        )

    monkeypatch.setattr(worker, "_email_sync_metrics_increment", _fake_metric_increment)
    monkeypatch.setattr(worker, "_email_sync_metrics_observe", _fake_metric_observe)

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            return None

        def complete_job(
            self,
            jid,
            result=None,
            worker_id=None,
            lease_id=None,
            completion_token=None,
        ):
            self.completed = {"jid": jid, "result": result}

    def _b64(value: str) -> str:
        return base64.urlsafe_b64encode(value.encode("utf-8")).decode().rstrip("=")

    class _InvalidHistoryCursorError(RuntimeError):
        def __init__(self):
            self.response = type("_Resp", (), {"status_code": 404})()
            super().__init__("startHistoryId too old")

    class FakeGmailConn:
        name = "gmail"
        list_history_calls = 0
        list_messages_queries: list[str | None] = []
        get_message_calls = 0

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_history(
            self,
            account,
            *,
            start_history_id,
            label_id=None,
            page_size=100,
            cursor=None,
        ):
            type(self).list_history_calls += 1
            raise _InvalidHistoryCursorError()

        async def list_messages(self, account, *, label_id=None, page_size=100, cursor=None, query=None):
            type(self).list_messages_queries.append(query)
            return ([{"id": "m-replay", "threadId": "t-replay"}], None)

        async def get_message(self, account, *, message_id, format="full"):
            type(self).get_message_calls += 1
            return {
                "id": message_id,
                "historyId": "55",
                "internalDate": "1736509200000",
                "labelIds": ["INBOX"],
                "snippet": "Recovered message",
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "Subject", "value": "Recovered subject"},
                        {"name": "From", "value": "Alice <alice@example.com>"},
                        {"name": "To", "value": "Bob <bob@example.com>"},
                        {"name": "Message-ID", "value": "<m-replay@example.com>"},
                        {"name": "Date", "value": "Fri, 10 Jan 2025 10:00:00 +0000"},
                    ],
                    "body": {"data": _b64("Recovered body")},
                },
            }

    import tldw_Server_API.app.core.External_Sources as ext_pkg

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeGmailConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "gmail",
            "account_id": 123,
            "remote_id": "INBOX",
            "type": "folder",
            "path": "/inbox",
            "options": {"query": "in:inbox"},
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_should_ingest_item(db, **kwargs):
        return True

    recorded_external_ids: list[str] = []

    async def _fake_record_ingested_item(db, **kwargs):
        recorded_external_ids.append(str(kwargs.get("external_id")))

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc

    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "should_ingest_item", _fake_should_ingest_item)
    monkeypatch.setattr(svc, "record_ingested_item", _fake_record_ingested_item)

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    import tldw_Server_API.app.core.AuthNZ.database as dbmod

    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs

    async def _fake_list_memberships_for_user(user_id: int):
        return []

    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FakeMDB:
        sync_state: dict[tuple[str, str, str], dict] = {
            ("42", "gmail", "99"): {
                "tenant_id": "42",
                "provider": "gmail",
                "source_key": "99",
                "source_id": 1,
                "cursor": "50",
                "retry_backoff_count": 0,
            }
        }

        def __init__(self, *a, **kw):
            return None

        def _key(self, *, provider, source_key, tenant_id):
            return (str(tenant_id), str(provider), str(source_key))

        def get_email_sync_state(self, *, provider, source_key, tenant_id=None):
            row = type(self).sync_state.get(
                self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            )
            return dict(row) if row else None

        def mark_email_sync_run_started(self, *, provider, source_key, tenant_id=None, cursor=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["cursor"] = cursor
            row["last_run_at"] = "started"
            row["retry_backoff_count"] = int(row.get("retry_backoff_count") or 0)
            type(self).sync_state[key] = row
            return dict(row)

        def mark_email_sync_run_succeeded(self, *, provider, source_key, tenant_id=None, cursor=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["cursor"] = cursor
            row["error_state"] = None
            row["retry_backoff_count"] = 0
            type(self).sync_state[key] = row
            return dict(row)

        def mark_email_sync_run_failed(self, *, provider, source_key, tenant_id=None, error_state=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["error_state"] = str(error_state or "failed")
            row["retry_backoff_count"] = int(row.get("retry_backoff_count") or 0) + 1
            type(self).sync_state[key] = row
            return dict(row)

        def add_media_with_keywords(self, **kwargs):
            return 1, "uuid", "ok"

        def upsert_email_message_graph(self, **kwargs):
            return {"email_message_id": 1}

    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm = FakeJM()
    await worker._process_import_job(
        jm,
        jid=10,
        lease_id="L10",
        worker_id="W",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 1
    assert jm.completed["result"]["cursor_recovery"] == "bounded_replay"
    assert jm.completed["result"]["cursor_recovery_window_days"] == 7
    assert FakeGmailConn.list_history_calls == 1
    assert FakeGmailConn.get_message_calls == 1
    assert FakeGmailConn.list_messages_queries
    assert "newer_than:7d" in str(FakeGmailConn.list_messages_queries[0] or "")
    assert "in:inbox" in str(FakeGmailConn.list_messages_queries[0] or "")
    assert recorded_external_ids == ["m-replay"]
    final_state = _FakeMDB.sync_state[("42", "gmail", "99")]
    assert final_state["cursor"] == "55"
    assert final_state["error_state"] is None
    assert final_state["retry_backoff_count"] == 0
    assert any(
        call["metric_name"] == "email_sync_runs_total"
        and call["labels"].get("status") == "success"
        for call in metric_increments
    )
    assert any(
        call["metric_name"] == "email_sync_recovery_events_total"
        and call["labels"].get("outcome") == "bounded_replay"
        for call in metric_increments
    )
    assert any(
        obs["metric_name"] == "email_sync_lag_seconds"
        for obs in metric_observations
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_gmail_invalid_cursor_escalates_full_backfill_required(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    monkeypatch.setenv("EMAIL_SYNC_CURSOR_RECOVERY_WINDOW_DAYS", "5")
    monkeypatch.setenv("EMAIL_SYNC_CURSOR_RECOVERY_MAX_MESSAGES", "10")

    metric_increments: list[dict[str, Any]] = []
    metric_observations: list[dict[str, Any]] = []

    def _fake_metric_increment(metric_name, *, labels, value=1):
        metric_increments.append(
            {
                "metric_name": str(metric_name),
                "labels": dict(labels or {}),
                "value": value,
            }
        )

    def _fake_metric_observe(metric_name, value, *, labels):
        metric_observations.append(
            {
                "metric_name": str(metric_name),
                "labels": dict(labels or {}),
                "value": float(value),
            }
        )

    monkeypatch.setattr(worker, "_email_sync_metrics_increment", _fake_metric_increment)
    monkeypatch.setattr(worker, "_email_sync_metrics_observe", _fake_metric_observe)

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            return None

        def complete_job(
            self,
            jid,
            result=None,
            worker_id=None,
            lease_id=None,
            completion_token=None,
        ):
            self.completed = {"jid": jid, "result": result}

    class _InvalidHistoryCursorError(RuntimeError):
        def __init__(self):
            self.response = type("_Resp", (), {"status_code": 404})()
            super().__init__("startHistoryId too old")

    class FakeGmailConn:
        name = "gmail"
        list_history_calls = 0
        list_messages_queries: list[str | None] = []
        get_message_calls = 0

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_history(
            self,
            account,
            *,
            start_history_id,
            label_id=None,
            page_size=100,
            cursor=None,
        ):
            type(self).list_history_calls += 1
            raise _InvalidHistoryCursorError()

        async def list_messages(self, account, *, label_id=None, page_size=100, cursor=None, query=None):
            type(self).list_messages_queries.append(query)
            return ([], None)

        async def get_message(self, account, *, message_id, format="full"):
            type(self).get_message_calls += 1
            raise AssertionError("No replay items means get_message should not be called")

    import tldw_Server_API.app.core.External_Sources as ext_pkg

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeGmailConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "gmail",
            "account_id": 123,
            "remote_id": "INBOX",
            "type": "folder",
            "path": "/inbox",
            "options": {"query": "in:inbox"},
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc

    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    import tldw_Server_API.app.core.AuthNZ.database as dbmod

    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs

    async def _fake_list_memberships_for_user(user_id: int):
        return []

    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FakeMDB:
        sync_state: dict[tuple[str, str, str], dict] = {
            ("42", "gmail", "99"): {
                "tenant_id": "42",
                "provider": "gmail",
                "source_key": "99",
                "source_id": 1,
                "cursor": "50",
                "retry_backoff_count": 0,
            }
        }

        def __init__(self, *a, **kw):
            return None

        def _key(self, *, provider, source_key, tenant_id):
            return (str(tenant_id), str(provider), str(source_key))

        def get_email_sync_state(self, *, provider, source_key, tenant_id=None):
            row = type(self).sync_state.get(
                self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            )
            return dict(row) if row else None

        def mark_email_sync_run_started(self, *, provider, source_key, tenant_id=None, cursor=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["cursor"] = cursor
            row["last_run_at"] = "started"
            row["retry_backoff_count"] = int(row.get("retry_backoff_count") or 0)
            type(self).sync_state[key] = row
            return dict(row)

        def mark_email_sync_run_succeeded(self, *, provider, source_key, tenant_id=None, cursor=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["cursor"] = cursor
            row["error_state"] = None
            row["retry_backoff_count"] = 0
            type(self).sync_state[key] = row
            return dict(row)

        def mark_email_sync_run_failed(self, *, provider, source_key, tenant_id=None, error_state=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["error_state"] = str(error_state or "failed")
            row["retry_backoff_count"] = int(row.get("retry_backoff_count") or 0) + 1
            type(self).sync_state[key] = row
            return dict(row)

        def add_media_with_keywords(self, **kwargs):
            raise AssertionError("No replay items means no ingest writes")

        def upsert_email_message_graph(self, **kwargs):
            raise AssertionError("No replay items means no upsert writes")

    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm = FakeJM()
    await worker._process_import_job(
        jm,
        jid=11,
        lease_id="L11",
        worker_id="W",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 0
    assert jm.completed["result"]["cursor_recovery"] == "full_backfill_required"
    assert jm.completed["result"]["cursor_recovery_window_days"] == 5
    assert FakeGmailConn.list_history_calls == 1
    assert FakeGmailConn.list_messages_queries
    assert "newer_than:5d" in str(FakeGmailConn.list_messages_queries[0] or "")
    assert FakeGmailConn.get_message_calls == 0
    final_state = _FakeMDB.sync_state[("42", "gmail", "99")]
    assert str(final_state.get("error_state") or "").startswith(
        "cursor_invalid_full_backfill_required"
    )
    assert final_state["retry_backoff_count"] == 1
    assert final_state["cursor"] == "50"
    assert any(
        call["metric_name"] == "email_sync_runs_total"
        and call["labels"].get("status") == "failed"
        for call in metric_increments
    )
    assert any(
        call["metric_name"] == "email_sync_failures_total"
        and call["labels"].get("reason") == "cursor_invalid_full_backfill_required"
        for call in metric_increments
    )
    assert any(
        call["metric_name"] == "email_sync_recovery_events_total"
        and call["labels"].get("outcome") == "full_backfill_required"
        for call in metric_increments
    )
    assert all(
        obs["metric_name"] != "email_sync_lag_seconds"
        for obs in metric_observations
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_gmail_incremental_applies_label_and_state_deltas_without_full_fetch(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            return None

        def complete_job(self, jid, result=None, worker_id=None, lease_id=None, completion_token=None):
            self.completed = {"jid": jid, "result": result}

    class FakeGmailConn:
        name = "gmail"
        list_messages_calls = 0
        get_message_calls = 0

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_messages(self, account, *, label_id=None, page_size=100, cursor=None, query=None):
            type(self).list_messages_calls += 1
            raise AssertionError("history mode should not call list_messages")

        async def list_history(
            self,
            account,
            *,
            start_history_id,
            label_id=None,
            page_size=100,
            cursor=None,
        ):
            assert str(start_history_id) == "30"
            return (
                [
                    {
                        "id": "m-label",
                        "threadId": "t-label",
                        "historyId": "35",
                        "message_added": False,
                        "message_deleted": False,
                        "labels_added": ["STARRED"],
                        "labels_removed": ["UNREAD"],
                    },
                    {
                        "id": "m-deleted",
                        "threadId": "t-deleted",
                        "historyId": "40",
                        "message_added": False,
                        "message_deleted": True,
                        "labels_added": [],
                        "labels_removed": [],
                    },
                ],
                None,
                "40",
            )

        async def get_message(self, account, *, message_id, format="full"):
            type(self).get_message_calls += 1
            raise AssertionError("delta-only updates should not fetch full message")

    import tldw_Server_API.app.core.External_Sources as ext_pkg

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeGmailConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "gmail",
            "account_id": 123,
            "remote_id": "INBOX",
            "type": "folder",
            "path": "/inbox",
            "options": {"query": "in:inbox"},
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_should_ingest_item(db, **kwargs):
        return True

    recorded_external_ids: list[str] = []

    async def _fake_record_ingested_item(db, **kwargs):
        recorded_external_ids.append(str(kwargs.get("external_id")))

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc

    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "should_ingest_item", _fake_should_ingest_item)
    monkeypatch.setattr(svc, "record_ingested_item", _fake_record_ingested_item)

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    import tldw_Server_API.app.core.AuthNZ.database as dbmod

    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs

    async def _fake_list_memberships_for_user(user_id: int):
        return []

    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FakeMDB:
        sync_state: dict[tuple[str, str, str], dict] = {
            ("42", "gmail", "99"): {
                "tenant_id": "42",
                "provider": "gmail",
                "source_key": "99",
                "source_id": 1,
                "cursor": "30",
                "retry_backoff_count": 0,
            }
        }
        label_delta_calls: list[dict] = []
        state_reconcile_calls: list[dict] = []

        def __init__(self, *a, **kw):
            return None

        def _key(self, *, provider, source_key, tenant_id):
            return (str(tenant_id), str(provider), str(source_key))

        def get_email_sync_state(self, *, provider, source_key, tenant_id=None):
            row = type(self).sync_state.get(self._key(provider=provider, source_key=source_key, tenant_id=tenant_id))
            return dict(row) if row else None

        def mark_email_sync_run_started(self, *, provider, source_key, tenant_id=None, cursor=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["tenant_id"] = key[0]
            row["provider"] = key[1]
            row["source_key"] = key[2]
            row["source_id"] = 1
            if cursor is not None:
                row["cursor"] = cursor
            row.setdefault("retry_backoff_count", 0)
            type(self).sync_state[key] = row
            return dict(row)

        def mark_email_sync_run_succeeded(self, *, provider, source_key, tenant_id=None, cursor=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["tenant_id"] = key[0]
            row["provider"] = key[1]
            row["source_key"] = key[2]
            row["source_id"] = 1
            if cursor is not None:
                row["cursor"] = cursor
            row["error_state"] = None
            row["retry_backoff_count"] = 0
            type(self).sync_state[key] = row
            return dict(row)

        def mark_email_sync_run_failed(self, *, provider, source_key, tenant_id=None, error_state=None):
            key = self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            row = dict(type(self).sync_state.get(key) or {})
            row["tenant_id"] = key[0]
            row["provider"] = key[1]
            row["source_key"] = key[2]
            row["source_id"] = 1
            row["error_state"] = str(error_state or "failed")
            row["retry_backoff_count"] = int(row.get("retry_backoff_count") or 0) + 1
            type(self).sync_state[key] = row
            return dict(row)

        def apply_email_label_delta(self, **kwargs):
            type(self).label_delta_calls.append(dict(kwargs))
            return {"applied": True, "reason": "ok"}

        def reconcile_email_message_state(self, **kwargs):
            type(self).state_reconcile_calls.append(dict(kwargs))
            return {"applied": True, "reason": "deleted"}

        def add_media_with_keywords(self, **kwargs):
            raise AssertionError("delta-only updates should not write full media rows")

        def upsert_email_message_graph(self, **kwargs):
            raise AssertionError("delta-only updates should not upsert full message graph")

    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm = FakeJM()
    await worker._process_import_job(
        jm,
        jid=5,
        lease_id="L5",
        worker_id="W",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 2
    assert jm.completed["result"]["total"] == 2
    assert recorded_external_ids == []
    assert FakeGmailConn.list_messages_calls == 0
    assert FakeGmailConn.get_message_calls == 0
    assert len(_FakeMDB.label_delta_calls) == 1
    assert _FakeMDB.label_delta_calls[0]["source_message_id"] == "m-label"
    assert _FakeMDB.label_delta_calls[0]["labels_added"] == ["STARRED"]
    assert _FakeMDB.label_delta_calls[0]["labels_removed"] == ["UNREAD"]
    assert len(_FakeMDB.state_reconcile_calls) == 1
    assert _FakeMDB.state_reconcile_calls[0]["source_message_id"] == "m-deleted"
    assert _FakeMDB.state_reconcile_calls[0]["deleted"] is True
    final_state = _FakeMDB.sync_state[("42", "gmail", "99")]
    assert final_state["cursor"] == "40"
    assert final_state["error_state"] is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_gmail_backoff_skips_run_when_retry_window_active(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    monkeypatch.setenv("EMAIL_SYNC_RETRY_MAX_ATTEMPTS", "6")
    monkeypatch.setenv("EMAIL_SYNC_RETRY_BASE_SECONDS", "120")
    monkeypatch.setenv("EMAIL_SYNC_RETRY_MAX_BACKOFF_SECONDS", "3600")

    metric_increments: list[dict[str, Any]] = []

    def _fake_metric_increment(metric_name, *, labels, value=1):
        metric_increments.append(
            {
                "metric_name": str(metric_name),
                "labels": dict(labels or {}),
                "value": value,
            }
        )

    monkeypatch.setattr(worker, "_email_sync_metrics_increment", _fake_metric_increment)
    monkeypatch.setattr(worker, "_email_sync_metrics_observe", lambda *a, **kw: None)

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            return None

        def complete_job(
            self,
            jid,
            result=None,
            worker_id=None,
            lease_id=None,
            completion_token=None,
        ):
            self.completed = {"jid": jid, "result": result}

    class FakeGmailConn:
        name = "gmail"
        list_messages_calls = 0
        list_history_calls = 0

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_messages(self, *a, **kw):
            type(self).list_messages_calls += 1
            raise AssertionError("backoff-gated run should not list messages")

        async def list_history(self, *a, **kw):
            type(self).list_history_calls += 1
            raise AssertionError("backoff-gated run should not list history")

    import tldw_Server_API.app.core.External_Sources as ext_pkg

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeGmailConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "gmail",
            "account_id": 123,
            "remote_id": "INBOX",
            "type": "folder",
            "path": "/inbox",
            "options": {},
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc

    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    import tldw_Server_API.app.core.AuthNZ.database as dbmod

    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs

    async def _fake_list_memberships_for_user(user_id: int):
        return []

    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FakeMDB:
        started_calls = 0
        sync_state: dict[tuple[str, str, str], dict] = {
            ("42", "gmail", "99"): {
                "tenant_id": "42",
                "provider": "gmail",
                "source_key": "99",
                "source_id": 1,
                "cursor": "30",
                "last_run_at": (
                    datetime.now(timezone.utc) - timedelta(seconds=30)
                ).isoformat(),
                "error_state": "quota_exceeded",
                "retry_backoff_count": 2,
            }
        }

        def __init__(self, *a, **kw):
            return None

        def _key(self, *, provider, source_key, tenant_id):
            return (str(tenant_id), str(provider), str(source_key))

        def get_email_sync_state(self, *, provider, source_key, tenant_id=None):
            row = type(self).sync_state.get(
                self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            )
            return dict(row) if row else None

        def mark_email_sync_run_started(self, *, provider, source_key, tenant_id=None, cursor=None):
            type(self).started_calls += 1
            raise AssertionError("backoff-gated run should not mark sync as started")

        def mark_email_sync_run_succeeded(self, *, provider, source_key, tenant_id=None, cursor=None):
            return {}

        def mark_email_sync_run_failed(self, *, provider, source_key, tenant_id=None, error_state=None):
            return {}

    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm = FakeJM()
    await worker._process_import_job(
        jm,
        jid=6,
        lease_id="L6",
        worker_id="W",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["skipped"] == "backoff_active"
    assert jm.completed["result"]["retry_backoff_count"] == 2
    assert jm.completed["result"]["retry_backoff_seconds"] == 240
    assert _FakeMDB.started_calls == 0
    assert FakeGmailConn.list_messages_calls == 0
    assert FakeGmailConn.list_history_calls == 0
    assert any(
        call["metric_name"] == "email_sync_runs_total"
        and call["labels"].get("status") == "skipped"
        for call in metric_increments
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_gmail_retry_budget_exhausted_skips_run(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    monkeypatch.setenv("EMAIL_SYNC_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("EMAIL_SYNC_RETRY_BASE_SECONDS", "60")
    monkeypatch.setenv("EMAIL_SYNC_RETRY_MAX_BACKOFF_SECONDS", "600")

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            return None

        def complete_job(
            self,
            jid,
            result=None,
            worker_id=None,
            lease_id=None,
            completion_token=None,
        ):
            self.completed = {"jid": jid, "result": result}

    class FakeGmailConn:
        name = "gmail"
        list_messages_calls = 0

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_messages(self, *a, **kw):
            type(self).list_messages_calls += 1
            raise AssertionError("retry-budget exhausted run should not enumerate messages")

    import tldw_Server_API.app.core.External_Sources as ext_pkg

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeGmailConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "gmail",
            "account_id": 123,
            "remote_id": "INBOX",
            "type": "folder",
            "path": "/inbox",
            "options": {},
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc

    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    import tldw_Server_API.app.core.AuthNZ.database as dbmod

    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs

    async def _fake_list_memberships_for_user(user_id: int):
        return []

    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FakeMDB:
        started_calls = 0
        sync_state: dict[tuple[str, str, str], dict] = {
            ("42", "gmail", "99"): {
                "tenant_id": "42",
                "provider": "gmail",
                "source_key": "99",
                "source_id": 1,
                "cursor": "30",
                "last_run_at": (
                    datetime.now(timezone.utc) - timedelta(minutes=30)
                ).isoformat(),
                "error_state": "provider_unavailable",
                "retry_backoff_count": 3,
            }
        }

        def __init__(self, *a, **kw):
            return None

        def _key(self, *, provider, source_key, tenant_id):
            return (str(tenant_id), str(provider), str(source_key))

        def get_email_sync_state(self, *, provider, source_key, tenant_id=None):
            row = type(self).sync_state.get(
                self._key(provider=provider, source_key=source_key, tenant_id=tenant_id)
            )
            return dict(row) if row else None

        def mark_email_sync_run_started(self, *, provider, source_key, tenant_id=None, cursor=None):
            type(self).started_calls += 1
            raise AssertionError("retry-budget exhausted run should not mark sync started")

        def mark_email_sync_run_succeeded(self, *, provider, source_key, tenant_id=None, cursor=None):
            return {}

        def mark_email_sync_run_failed(self, *, provider, source_key, tenant_id=None, error_state=None):
            return {}

    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm = FakeJM()
    await worker._process_import_job(
        jm,
        jid=7,
        lease_id="L7",
        worker_id="W",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["skipped"] == "retry_budget_exhausted"
    assert jm.completed["result"]["retry_backoff_count"] == 3
    assert jm.completed["result"]["retry_backoff_seconds"] == 240
    assert _FakeMDB.started_calls == 0
    assert FakeGmailConn.list_messages_calls == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_gmail_large_fixture_backfill_handles_edge_cases(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    fixture_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "gmail_backfill_large_fixture.jsonc"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    pages = fixture["pages"]
    edge_cases = fixture["edge_case_ids"]
    expected_total = sum(len(page["ids"]) for page in pages)
    pages_by_cursor = {page["cursor"]: page for page in pages}

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *a, **kw):
            return None

        def complete_job(
            self,
            jid,
            result=None,
            worker_id=None,
            lease_id=None,
            completion_token=None,
        ):
            self.completed = {"jid": jid, "result": result}

    def _b64(value: str) -> str:
        return base64.urlsafe_b64encode(value.encode("utf-8")).decode().rstrip("=")

    class FakeGmailConn:
        name = "gmail"

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_messages(
            self,
            account,
            *,
            label_id=None,
            page_size=100,
            cursor=None,
            query=None,
        ):
            page = pages_by_cursor.get(cursor)
            if page is None:
                return [], None
            rows = [{"id": message_id, "threadId": f"t-{message_id}"} for message_id in page["ids"]]
            return rows, page["next_cursor"]

        async def get_message(self, account, *, message_id, format="full"):
            idx = int(message_id.replace("m", ""))
            case = edge_cases.get(message_id, "default")
            base_headers = [
                {"name": "Subject", "value": f"Fixture Subject {message_id}"},
                {"name": "From", "value": "Alice <alice@example.com>"},
                {"name": "To", "value": "Bob <bob@example.com>"},
                {"name": "Message-ID", "value": f"<{message_id}@example.com>"},
                {"name": "Date", "value": "Fri, 10 Jan 2025 10:00:00 +0000"},
            ]
            if case == "multipart_plain_html_attachment":
                payload = {
                    "mimeType": "multipart/mixed",
                    "headers": base_headers,
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": _b64(f"Plain fixture {message_id}")}},
                        {"mimeType": "text/html", "body": {"data": _b64(f"<p>HTML fixture {message_id}</p>")}},
                        {
                            "mimeType": "application/pdf",
                            "filename": f"{message_id}.pdf",
                            "body": {"attachmentId": f"a-{message_id}", "size": 128},
                            "headers": [
                                {
                                    "name": "Content-Disposition",
                                    "value": f"attachment; filename={message_id}.pdf",
                                }
                            ],
                        },
                    ],
                }
            elif case == "html_only":
                payload = {
                    "mimeType": "text/html",
                    "headers": base_headers,
                    "body": {"data": _b64(f"<h1>HTML only {message_id}</h1><p>Body</p>")},
                }
            elif case == "root_plain":
                payload = {
                    "mimeType": "message/rfc822",
                    "headers": base_headers,
                    "body": {"data": _b64(f"Root plain {message_id}")},
                }
            elif case == "duplicate_attachment_parts":
                duplicate_attachment_part = {
                    "mimeType": "application/octet-stream",
                    "filename": f"{message_id}.bin",
                    "body": {"attachmentId": f"a-{message_id}", "size": 64},
                    "headers": [
                        {
                            "name": "Content-Disposition",
                            "value": f"attachment; filename={message_id}.bin",
                        }
                    ],
                }
                payload = {
                    "mimeType": "multipart/mixed",
                    "headers": base_headers,
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": _b64(f"Body {message_id}")}},
                        duplicate_attachment_part,
                        duplicate_attachment_part,
                    ],
                }
            elif case == "inline_attachment_only":
                payload = {
                    "mimeType": "multipart/related",
                    "headers": base_headers,
                    "parts": [
                        {
                            "mimeType": "image/png",
                            "filename": f"{message_id}.png",
                            "body": {"attachmentId": f"inline-{message_id}", "size": 256},
                            "headers": [
                                {"name": "Content-Disposition", "value": "inline"},
                                {"name": "Content-ID", "value": f"<cid-{message_id}>"},
                            ],
                        }
                    ],
                }
            else:
                payload = {
                    "mimeType": "text/plain",
                    "headers": base_headers,
                    "body": {"data": _b64(f"Default body {message_id}")},
                }
            return {
                "id": message_id,
                "historyId": str(1000 + idx),
                "internalDate": str(1736509200000 + (idx * 1000)),
                "labelIds": ["INBOX", "inbox", "IMPORTANT"],
                "snippet": f"Snippet fallback {message_id}",
                "payload": payload,
            }

    import tldw_Server_API.app.core.External_Sources as ext_pkg

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeGmailConn())

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "gmail",
            "account_id": 123,
            "remote_id": "INBOX",
            "type": "folder",
            "path": "/inbox",
            "options": {"query": "in:inbox"},
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_should_ingest_item(db, **kwargs):
        return True

    recorded: list[dict[str, Any]] = []

    async def _fake_record_ingested_item(db, **kwargs):
        recorded.append(kwargs)

    import tldw_Server_API.app.core.External_Sources.connectors_service as svc

    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "should_ingest_item", _fake_should_ingest_item)
    monkeypatch.setattr(svc, "record_ingested_item", _fake_record_ingested_item)

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    import tldw_Server_API.app.core.AuthNZ.database as dbmod

    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs

    async def _fake_list_memberships_for_user(user_id: int):
        return []

    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    class _FakeMDB:
        instances: list["_FakeMDB"] = []

        def __init__(self, *a, **kw):
            self.add_calls: list[dict[str, Any]] = []
            self.upsert_calls: list[dict[str, Any]] = []
            type(self).instances.append(self)

        def add_media_with_keywords(self, **kwargs):
            self.add_calls.append(kwargs)
            return len(self.add_calls), "uuid", "ok"

        def upsert_email_message_graph(self, **kwargs):
            self.upsert_calls.append(kwargs)
            return {"email_message_id": len(self.upsert_calls)}

    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )

    jm = FakeJM()
    await worker._process_import_job(
        jm,
        jid=1,
        lease_id="L",
        worker_id="W",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == expected_total
    assert jm.completed["result"]["total"] == expected_total
    assert _FakeMDB.instances
    mdb_instance = _FakeMDB.instances[0]
    assert len(mdb_instance.add_calls) == expected_total
    assert len(mdb_instance.upsert_calls) == expected_total
    assert len(recorded) == expected_total

    add_by_url = {str(call["url"]): call for call in mdb_instance.add_calls}
    assert add_by_url["gmail://99/m001"]["content"] == "Plain fixture m001"
    assert add_by_url["gmail://99/m002"]["content"] == "HTML only m002 Body"
    assert add_by_url["gmail://99/m003"]["content"] == "Root plain m003"
    assert add_by_url["gmail://99/m005"]["content"] == "Snippet fallback m005"

    upsert_by_source = {
        str(call["source_message_id"]): call for call in mdb_instance.upsert_calls
    }
    assert upsert_by_source["m004"]["metadata"]["email"]["attachments"]
    assert len(upsert_by_source["m004"]["metadata"]["email"]["attachments"]) == 1
    assert upsert_by_source["m001"]["metadata"]["email"]["labels"] == ["INBOX", "IMPORTANT"]
