"""Document conversation routes with authenticated, native Chat storage."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_Server_API.app.api.v1.API_Deps import DB_Deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(params=["sqlite", "postgres"])
def document_api(request, credentialed_test_client, tmp_path, monkeypatch, auth_headers):
    """Use a shared Media fixture and a real ChaCha backend."""
    kwargs = {}
    if request.param == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(
        db_path=tmp_path / "document-chat.sqlite", client_id="1", owner_user_id="1", **kwargs,
    )
    media = {1: {"uuid": str(uuid4())}, 2: {"uuid": str(uuid4())}}
    media_db = SimpleNamespace(get_media_by_id=lambda media_id, **_: media.get(media_id))
    monkeypatch.setattr(DB_Deps, "_resolve_media_db_for_user", lambda _user: media_db)
    overrides = credentialed_test_client.app.dependency_overrides
    monkeypatch.setitem(overrides, get_chacha_db_for_user, lambda: db)
    yield credentialed_test_client, db, media, auth_headers
    db.close_connection()
    if request.param == "postgres":
        db.backend.get_pool().close_all()


def activate_document_sync_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the fixture owner's personal dataset through the real Sync lookup."""
    dataset = SimpleNamespace(
        scope_type="personal", metadata={"default_personal": True, "client_family": "chatbook"},
    )
    service = SimpleNamespace(store=SimpleNamespace(list_datasets_for_user=lambda _: [dataset]))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sync.v2.factory.sync_v2_storage_exists_for_user",
        lambda user_id: user_id == "1",
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sync.v2.factory.sync_v2_service_for_user",
        lambda _: service,
    )


def _list(client, headers, media_uuid, **query):
    return client.get(
        "/api/v1/media/1/chat-conversations", params={"media_uuid": media_uuid, **query}, headers=headers,
    )


def _create(client, headers, media_uuid, **body):
    return client.post(
        "/api/v1/chats/", params={"document_media_id": 1, "document_media_uuid": media_uuid},
        headers=headers, json={"title": "Private PDF chat", **body},
    )


def _attach(client, headers, chat_id, record, media_uuid, *, media_id=1):
    return client.put(
        f"/api/v1/media/{media_id}/chat-conversations/{chat_id}", headers=headers,
        json={"media_uuid": media_uuid, "document_chat_record": record},
    )


def test_document_link_replay_and_visibility(document_api):
    client, db, media, headers = document_api
    media_uuid = media[1]["uuid"]
    assert _list(client, headers, media_uuid).json() == {"items": [], "has_more": False}
    created = _create(client, headers, media_uuid)
    assert created.status_code == 201, created.text
    cid = created.json()["id"]
    record = created.json()["document_chat_record"]
    detail = client.get(f"/api/v1/chats/{cid}", headers=headers)
    assert "document_chat_record" not in detail.json()
    scoped_detail = client.get(
        f"/api/v1/chats/{cid}", headers=headers,
        params={"document_media_id": 1, "document_media_uuid": media_uuid},
    )
    assert scoped_detail.json()["document_chat_record"] == record
    first = _attach(client, headers, cid, record, media_uuid)
    replay = _attach(client, headers, cid, record, media_uuid)
    assert first.status_code == replay.status_code == 204
    assert _list(client, headers, media_uuid).json()["items"] == []
    db.add_message({"conversation_id": cid, "sender": "user", "content": "Read this"})
    assert [item["id"] for item in _list(client, headers, media_uuid).json()["items"]] == [cid]

    another = _create(client, headers, media_uuid, title="Second chat").json()
    another_id = another["id"]
    assert _attach(client, headers, another_id, another["document_chat_record"], media_uuid).status_code == 204
    db.add_message({"conversation_id": another_id, "sender": "user", "content": "A second question"})
    first_page = _list(client, headers, media_uuid, limit=1).json()
    next_page = _list(client, headers, media_uuid, limit=1, offset=1).json()
    assert first_page["has_more"] and not next_page["has_more"]
    assert {first_page["items"][0]["id"], next_page["items"][0]["id"]} == {cid, another_id}

    assert _attach(client, headers, cid, record, media[2]["uuid"], media_id=2).status_code == 409
    changed_account = {**headers, "X-TLDW-Expected-User-ID": "2"}
    assert _list(client, changed_account, media_uuid).status_code == 412
    row = db.get_conversation_by_id(cid)
    db.soft_delete_conversation(cid, row["version"])
    assert [item["id"] for item in _list(client, headers, media_uuid).json()["items"]] == [another_id]
    deleted = db.get_conversation_by_id(cid, include_deleted=True)
    db.restore_conversation(cid, deleted["version"])
    assert any(item["id"] == cid for item in _list(client, headers, media_uuid).json()["items"])
    media.pop(1)
    assert _list(client, headers, media_uuid).status_code == 404


def test_document_link_rejects_replaced_record(document_api):
    client, db, media, headers = document_api
    media_uuid = media[1]["uuid"]
    cid = str(uuid4())
    original = _create(client, headers, media_uuid, id=cid).json()["document_chat_record"]
    assert _attach(client, headers, cid, original, media_uuid).status_code == 204
    assert db.hard_delete_conversation(cid)
    replacement = _create(client, headers, media_uuid, id=cid).json()["document_chat_record"]
    assert replacement != original
    assert _attach(client, headers, cid, original, media_uuid).status_code == 404
    assert _attach(client, headers, cid, replacement, media_uuid).status_code == 204


def test_document_chat_create_replays_only_for_original_document(document_api):
    client, _, media, headers = document_api
    body = {"id": str(uuid4()), "title": "Question captured before a connection loss"}
    first = _create(client, headers, media[1]["uuid"], **body)
    replay = _create(client, headers, media[1]["uuid"], **body)
    assert first.status_code == 201, first.text
    assert replay.status_code == 200, replay.text
    assert replay.json()["document_chat_record"] == first.json()["document_chat_record"]
    assert client.post(
        "/api/v1/chats/", headers=headers, json=body,
        params={"document_media_id": 2, "document_media_uuid": media[2]["uuid"]},
    ).status_code == 409


def test_document_link_does_not_cross_owners(document_api, monkeypatch):
    client, db, media, headers = document_api
    media_uuid = media[1]["uuid"]
    created = _create(client, headers, media_uuid).json()
    cid, record = created["id"], created["document_chat_record"]
    assert _attach(client, headers, cid, record, media_uuid).status_code == 204
    db.add_message({"conversation_id": cid, "sender": "user", "content": "Private question"})
    monkeypatch.setitem(client.app.dependency_overrides, get_request_user, lambda: SimpleNamespace(id=2))
    assert _list(client, headers, media_uuid).json()["items"] == []
    assert _attach(client, headers, cid, record, media_uuid).status_code == 404
