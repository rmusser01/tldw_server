"""Mocked Gmail delivery with real SQLite Media and normalized email storage."""

import base64
from contextlib import asynccontextmanager
from secrets import token_urlsafe
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("original_body", ["Original synthetic mail", "", "[empty content for gmail:m1]"])
async def test_gmail_reimport_preserves_saved_content_but_refreshes_labels(tmp_path, monkeypatch, original_body):
    from tldw_Server_API.app.core import External_Sources
    from tldw_Server_API.app.core.AuthNZ import database, orgs_teams
    from tldw_Server_API.app.core.External_Sources import connectors_service
    from tldw_Server_API.app.services import connectors_worker as worker

    state = {"body": original_body, "subject": "Original subject", "labels": ["INBOX"]}

    class Provider:
        name = "gmail"

        async def list_messages(self, *args, **kwargs):
            return [{"id": "m1"}], None

        async def get_message(self, *args, **kwargs):
            return {
                "id": "m1",
                "historyId": "7",
                "internalDate": "1736509200000",
                "labelIds": state["labels"],
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "Subject", "value": state["subject"]},
                        {"name": "From", "value": "sender@example.test"},
                        {"name": "Message-ID", "value": "<m1@example.test>"},
                    ],
                    "body": {"data": base64.urlsafe_b64encode(state["body"].encode()).decode()},
                },
            }

    @asynccontextmanager
    async def transaction():
        yield object()

    async def pool():
        return SimpleNamespace(transaction=transaction)

    async def source(*args):
        return {"id": 99, "provider": "gmail", "account_id": 123, "remote_id": "INBOX", "options": {}}

    async def tokens(*args):
        return {"access_token": token_urlsafe(16)}

    async def yes(*args, **kwargs):
        return True

    async def no_op(*args, **kwargs):
        return []

    monkeypatch.setattr(External_Sources, "get_connector_by_name", lambda name: Provider())
    monkeypatch.setattr(connectors_service, "get_source_by_id", source)
    monkeypatch.setattr(connectors_service, "get_account_tokens", tokens)
    monkeypatch.setattr(connectors_service, "should_ingest_item", yes)
    monkeypatch.setattr(connectors_service, "record_ingested_item", no_op)
    monkeypatch.setattr(database, "get_db_pool", pool)
    monkeypatch.setattr(orgs_teams, "list_memberships_for_user", no_op)
    monkeypatch.setenv("EMAIL_NATIVE_PERSIST_ENABLED", "true")
    monkeypatch.setenv("EMAIL_GMAIL_CONNECTOR_ENABLED", "true")
    db_path = str(tmp_path / "gmail.db")
    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: MediaDatabase(
            db_path=str(tmp_path / "gmail.db"),
            client_id="42",
        ),
    )
    manager = SimpleNamespace(renew_job_lease=lambda *a, **kw: None, complete_job=lambda *a, **kw: None)
    await worker._process_import_job(manager, jid=1, lease_id="L", worker_id="W", source_id=99, user_id=42)
    # Force another full delivery of the same identity, now with changed body and labels.
    state.update(body="Incoming replacement", subject="Incoming subject", labels=["STARRED"])
    await worker._process_import_job(manager, jid=2, lease_id="L", worker_id="W", source_id=99, user_id=42)
    db = MediaDatabase(db_path=db_path, client_id="42")
    try:
        rows, total = db.search_email_messages(tenant_id="42")
        assert total == 1
        detail = db.get_email_message_detail(email_message_id=rows[0]["email_message_id"], tenant_id="42")
        assert detail["body_text"] == original_body
        assert db.get_media_by_id(rows[0]["media_id"])["content"] == (original_body or "[empty content for gmail:m1]")
        assert detail["subject"] == "Original subject"
        assert [label["label_name"] for label in detail["labels"]] == ["STARRED"]
    finally:
        db.close_connection()
