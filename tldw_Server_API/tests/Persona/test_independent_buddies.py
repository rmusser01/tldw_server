"""Independent artwork lifetime, ownership, and attachment API regressions."""

from __future__ import annotations

import importlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_starter_catalog import PersonaVisualStarterCatalogService

pytestmark = pytest.mark.integration


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(DatabasePaths, "get_user_base_directory", staticmethod(lambda user_id: tmp_path / str(user_id)))
    monkeypatch.setattr(
        DatabasePaths,
        "get_user_persona_visuals_dir",
        staticmethod(lambda user_id: tmp_path / str(user_id) / "persona_visuals"),
    )
    database = CharactersRAGDB(tmp_path / "buddies.sqlite", "1")
    yield database
    database.close_connection()


def _client(db, user_id=1):
    endpoint = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.buddies")
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/buddies")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=user_id)
    app.dependency_overrides[check_rate_limit] = lambda: None
    return TestClient(app)


def _create(client, **updates):
    payload = {"name": "Pixel friend", "source": {"kind": "starter", "starter_id": "pixel-migu"}, **updates}
    response = client.post("/api/v1/buddies", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _persona(db, user_id="1", persona_id="source-persona"):
    character_id = db.add_character_card(
        {"name": f"Source {persona_id}", "description": "Test source", "system_prompt": "Be clear."}
    )
    db.create_persona_profile(
        {
            "id": persona_id,
            "user_id": user_id,
            "name": persona_id,
            "character_card_id": character_id,
            "system_prompt": "Be clear.",
            "mode": "session_scoped",
            "is_active": True,
        }
    )
    return persona_id


def test_v65_database_upgrade_preserves_conversation_and_adds_independent_storage(tmp_path, monkeypatch):
    path = tmp_path / "upgrade.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 65)
        database = CharactersRAGDB(path, "1")
        conversation_id = database.add_conversation({"title": "Keep this", "client_id": "1"})
        database.close_connection()
    database = CharactersRAGDB(path, "1")
    try:
        with sqlite3.connect(path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            version = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?", (CharactersRAGDB._SCHEMA_NAME,)
            ).fetchone()[0]
        assert version == 66
        assert {
            "buddy_profiles",
            "buddy_assets",
            "buddy_attachments",
            "buddy_result_acknowledgements",
            "buddy_turn_owners",
            "buddy_turns",
        } <= tables
        assert database.get_conversation_by_id(conversation_id)["title"] == "Keep this"
    finally:
        database.close_connection()


def test_migration_failure_rolls_back_all_buddy_tables_and_version(tmp_path, monkeypatch):
    path = tmp_path / "rollback.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 65)
        database = CharactersRAGDB(path, "1")
        database.close_connection()
    schema = importlib.import_module("tldw_Server_API.app.core.DB_Management.Buddy_DB")
    monkeypatch.setattr(schema, "BUDDY_SCHEMA_SQL", schema.BUDDY_SCHEMA_SQL + "INVALID MIGRATION SQL;")
    with pytest.raises(Exception, match="syntax error"):
        CharactersRAGDB(path, "1")
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?", (CharactersRAGDB._SCHEMA_NAME,)
            ).fetchone()[0]
            == 65
        )
        assert not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'buddy_%'").fetchall()


def test_schema_head_v66_has_one_sqlite_and_postgres_step():
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 66
    assert CharactersRAGDB._POSTGRES_SCHEMA_VERSION == 66
    assert (
        CharactersRAGDB.__new__(CharactersRAGDB)._sqlite_linear_migration_steps()[65].__name__
        == "_migrate_from_v65_to_v66"
    )
    assert hasattr(CharactersRAGDB, "_migrate_from_v65_to_v66_postgres")


def test_profile_deleted_schema_matches_postgres_boolean_query_contract(db):
    from tldw_Server_API.app.core.DB_Management.backends.query_utils import transform_sqlite_query_for_postgres

    columns = {row["name"]: row["type"] for row in db.execute_query("PRAGMA table_info(buddy_profiles)").fetchall()}
    prepared = transform_sqlite_query_for_postgres("SELECT id FROM buddy_profiles WHERE deleted = 0")
    assert "deleted = FALSE" in prepared
    assert columns["deleted"] == "BOOLEAN"


def test_profile_updates_bind_boolean_values_for_postgres_adapter(db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.backends.query_utils import prepare_backend_statement

    with _client(db) as client:
        buddy = _create(client)
        original = db.transaction
        deleted_bindings = []

        class PreparingConnection:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, query, params=None):
                if query.startswith("UPDATE buddy_profiles"):
                    prepared, values = prepare_backend_statement(
                        BackendType.POSTGRESQL, query, params, apply_default_transform=True
                    )
                    assert "deleted = FALSE" in prepared
                    deleted_bindings.append(values[3])
                    assert type(values[3]) is bool
                return self.connection.execute(query, params or ())

        @contextmanager
        def transaction():
            with original() as connection:
                yield PreparingConnection(connection)

        monkeypatch.setattr(db, "transaction", transaction)
        url = f"/api/v1/buddies/{buddy['id']}"
        assert client.patch(url, json={"expected_version": 1, "name": "Updated"}).status_code == 200
        assert client.delete(url, params={"expected_version": 2}).status_code == 204
        assert deleted_bindings == [False, True]


@pytest.mark.timeout(60)
def test_postgres_v65_upgrade_installs_buddy_storage_and_forced_tenant_policies(
    pg_database_config, monkeypatch, tmp_path
):
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    monkeypatch.setattr(DatabasePaths, "get_user_base_directory", staticmethod(lambda user_id: tmp_path / str(user_id)))
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 65)
        database = CharactersRAGDB(":memory:", client_id="1", backend=backend)
    try:
        conversation_id = database.add_conversation({"title": "Retained", "client_id": "1"})
        database._initialize_schema_postgres()
        assert (
            backend.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = %s", (CharactersRAGDB._SCHEMA_NAME,)
            ).scalar
            == 66
        )
        assert database.get_conversation_by_id(conversation_id)["title"] == "Retained"
        assert (
            backend.execute(
                "SELECT data_type FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'buddy_profiles' AND column_name = 'deleted'"
            ).scalar
            == "boolean"
        )
        tables = [
            "buddy_profiles",
            "buddy_assets",
            "buddy_attachments",
            "buddy_result_acknowledgements",
            "buddy_turn_owners",
            "buddy_turns",
        ]
        relations = backend.execute(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = ANY(%s)", (tables,)
        ).rows
        assert {row["relname"] for row in relations} == set(tables)
        assert all(row["relrowsecurity"] and row["relforcerowsecurity"] for row in relations)
        policies = backend.execute(
            "SELECT tablename, qual, with_check FROM pg_policies WHERE schemaname = current_schema() AND tablename = ANY(%s)",
            (tables,),
        ).rows
        assert {row["tablename"] for row in policies} == set(tables)
        assert all("user_id" in row["qual"] and "app.current_user_id" in row["with_check"] for row in policies)
        with _client(database) as client:
            buddy = _create(client)
            url = f"/api/v1/buddies/{buddy['id']}"
            assert client.get(url).status_code == 200
            assert client.get("/api/v1/buddies").json()["buddies"][0]["id"] == buddy["id"]
            assert client.get(buddy["assets"][0]["content_url"]).content.startswith(b"\x89PNG")
            updated = client.patch(url, json={"expected_version": 1, "name": "Updated"})
            assert updated.status_code == 200, updated.text
            assert updated.json()["version"] == 2
            assert client.patch(url, json={"expected_version": 1, "name": "Stale"}).status_code == 409
            assert (
                client.put(
                    "/api/v1/buddies/attachment",
                    json={
                        "expected_version": 0,
                        "buddy_id": buddy["id"],
                        "scope_type": "conversation",
                        "scope_id": conversation_id,
                    },
                ).status_code
                == 200
            )
            result_id = database.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "PostgreSQL result",
                    "client_id": "1",
                }
            )
            activity = client.get("/api/v1/buddies/attachment/activity")
            assert activity.status_code == 200, activity.text
            assert activity.json()["items"][0]["result"]["id"] == result_id
            ack = client.post(
                "/api/v1/buddies/attachment/acknowledgements",
                json={"conversation_id": conversation_id, "result_message_id": result_id},
            )
            assert ack.status_code == 200, ack.text
            assert client.get("/api/v1/buddies/attachment/activity").json()["items"][0]["acknowledged"] is True
            assert client.delete(url, params={"expected_version": 2}).status_code == 204
            assert client.get(url).status_code == 404
            assert client.get("/api/v1/buddies/attachment/activity").json()["items"] == []
    finally:
        database.close_all_connections()
        backend.get_pool().close_all()


def test_conversation_target_resolves_workspace_scope_before_attachment(db):
    db.upsert_workspace("ws", "Workspace")
    conversation = db.add_conversation(
        {"title": "Scoped chat", "client_id": "1", "scope_type": "workspace", "workspace_id": "ws"}
    )
    with _client(db) as client:
        response = client.get(f"/api/v1/buddies/conversation-targets/{conversation}")
        assert response.status_code == 200
        assert response.json() == {
            "id": conversation,
            "title": "Scoped chat",
            "scope_type": "workspace",
            "workspace_id": "ws",
            "version": 1,
            "assistant_kind": None,
            "assistant_id": None,
            "assistant_name": None,
        }
    with _client(db, 2) as client:
        assert client.get(f"/api/v1/buddies/conversation-targets/{conversation}").status_code == 404


def test_conversation_target_projects_owned_persona_identity(db):
    persona_id = _persona(db)
    conversation = db.add_conversation(
        {"title": "Persona chat", "client_id": "1", "assistant_kind": "persona", "assistant_id": persona_id}
    )
    with _client(db) as client:
        response = client.get(f"/api/v1/buddies/conversation-targets/{conversation}")
        assert response.status_code == 200
        assert response.json()["assistant_kind"] == "persona"
        assert response.json()["assistant_id"] == persona_id
        assert response.json()["assistant_name"] == persona_id
    row = db.get_conversation_by_id(conversation)
    db.soft_delete_conversation(conversation, row["version"])
    with _client(db) as client:
        assert client.get(f"/api/v1/buddies/conversation-targets/{conversation}").status_code == 404


def test_starter_creates_art_without_creating_a_persona(db):
    with _client(db) as client:
        buddy = _create(client)
        assert buddy["optional_persona_id"] is None
        assert buddy["display_mode"] == "dynamic"
        assert buddy["manifest"]["states"]["idle"]
        assert buddy["attribution"]["license_label"] == "LicenseRef-User-Supplied"
        assert db.list_persona_profiles(user_id="1") == []
        assert client.get(buddy["assets"][0]["content_url"]).content.startswith(b"\x89PNG")
        assert client.get("/api/v1/buddies").json()["buddies"][0]["id"] == buddy["id"]


def test_copied_art_survives_source_persona_deletion(db):
    persona_id = _persona(db)
    source = PersonaVisualStarterCatalogService(db).copy_starter_pack_to_persona(
        starter_pack_id="pixel-migu", persona_id=persona_id, user_id="1"
    )
    with _client(db) as client:
        buddy = _create(
            client,
            source={"kind": "persona_pack", "persona_id": persona_id, "pack_id": source["id"]},
            optional_persona_id=persona_id,
        )
        first = client.get(buddy["assets"][0]["content_url"])
        profile = db.get_persona_profile(persona_id, user_id="1")
        db.soft_delete_persona_profile(persona_id=persona_id, user_id="1", expected_version=profile["version"])
        restored = client.get(f"/api/v1/buddies/{buddy['id']}")
        assert restored.status_code == 200
        assert restored.json()["optional_persona_available"] is False
        assert restored.json()["attribution"] == buddy["attribution"]
        assert client.get(buddy["assets"][0]["content_url"]).content == first.content


def test_profile_and_asset_access_cannot_cross_user_even_in_same_database(db):
    with _client(db) as owner:
        buddy = _create(owner)
    with _client(db, 2) as stranger:
        assert stranger.get("/api/v1/buddies").json() == {"buddies": []}
        assert stranger.get(f"/api/v1/buddies/{buddy['id']}").status_code == 404
        assert stranger.get(buddy["assets"][0]["content_url"]).status_code == 404
        assert (
            stranger.patch(f"/api/v1/buddies/{buddy['id']}", json={"expected_version": 1, "name": "Stolen"}).status_code
            == 404
        )


def test_versioned_update_preserves_immutable_art_and_rejects_stale_delete(db):
    with _client(db) as client:
        buddy = _create(client)
        url = f"/api/v1/buddies/{buddy['id']}"
        updated = client.patch(url, json={"expected_version": 1, "name": "Renamed", "display_mode": "static"})
        assert updated.status_code == 200
        assert updated.json()["version"] == 2
        assert updated.json()["manifest"] == buddy["manifest"]
        assert client.patch(url, json={"expected_version": 1, "name": "Stale"}).status_code == 409
        assert client.delete(url, params={"expected_version": 1}).status_code == 409
        assert client.patch(url, json={"expected_version": 2, "manifest": {}}).status_code == 422
        assert client.delete(url, params={"expected_version": 2}).status_code == 204
        assert client.get(buddy["assets"][0]["content_url"]).status_code == 404


def test_optional_persona_and_source_pack_are_owner_checked(db):
    persona_id = _persona(db, user_id="2")
    source = PersonaVisualStarterCatalogService(db).copy_starter_pack_to_persona(
        starter_pack_id="pixel-migu", persona_id=persona_id, user_id="2"
    )
    with _client(db) as client:
        assert (
            client.post(
                "/api/v1/buddies",
                json={
                    "name": "No",
                    "source": {"kind": "persona_pack", "persona_id": persona_id, "pack_id": source["id"]},
                },
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/v1/buddies",
                json={
                    "name": "No",
                    "source": {"kind": "starter", "starter_id": "pixel-migu"},
                    "optional_persona_id": persona_id,
                },
            ).status_code
            == 404
        )


def test_attachment_rechecks_target_and_rejects_stale_slot_updates(db):
    conversation_id = db.add_conversation({"title": "Owned", "client_id": "1"})
    other_id = db.add_conversation({"title": "Other", "client_id": "2"})
    with _client(db) as client:
        buddy = _create(client)
        url = "/api/v1/buddies/attachment?client_slot=browser-a"
        assert client.get(url).json()["version"] == 0
        body = {
            "expected_version": 0,
            "buddy_id": buddy["id"],
            "scope_type": "conversation",
            "scope_id": conversation_id,
        }
        attached = client.put(url, json=body)
        assert attached.status_code == 200, attached.text
        assert attached.json()["version"] == 1
        assert client.put(url, json=body).status_code == 409
        assert client.put(url, json={**body, "expected_version": 1, "scope_id": other_id}).status_code == 404
        row = db.get_conversation_by_id(conversation_id)
        db.soft_delete_conversation(conversation_id, row["version"])
        current = client.get(url).json()
        assert current["attachment"] is None
        assert current["unavailable_reason"] == "target_unavailable"
        assert client.delete(url + "&expected_version=1").json()["version"] == 2
        assert client.get(url).json()["attachment"] is None


def test_workspace_attachment_does_not_survive_workspace_deletion(db):
    workspace = db.upsert_workspace("workspace-one", "Owned workspace")
    with _client(db) as client:
        buddy = _create(client)
        response = client.put(
            "/api/v1/buddies/attachment",
            json={
                "expected_version": 0,
                "buddy_id": buddy["id"],
                "scope_type": "workspace",
                "scope_id": workspace["id"],
            },
        )
        assert response.status_code == 200, response.text
        db.delete_workspace(workspace["id"], expected_version=workspace["version"])
        assert client.get("/api/v1/buddies/attachment").json()["unavailable_reason"] == "target_unavailable"


@pytest.mark.parametrize(
    "payload",
    [
        {"name": " ", "source": {"kind": "starter", "starter_id": "pixel-migu"}},
        {"name": "No", "source": {"kind": "starter", "starter_id": "../../secrets"}},
        {"name": "No", "source": {"kind": "starter", "starter_id": "pixel-migu"}, "user_id": "2"},
    ],
)
def test_untrusted_creation_fields_are_bounded_and_rejected(db, payload):
    with _client(db) as client:
        assert client.post("/api/v1/buddies", json=payload).status_code == 422


def test_workspace_attachment_projects_only_owned_conversations_with_scope(db):
    db.upsert_workspace("ws", "My workspace")
    owned_id = db.add_conversation(
        {"title": "Own scoped chat", "client_id": "1", "scope_type": "workspace", "workspace_id": "ws"}
    )
    db.add_conversation(
        {"title": "Other user's chat", "client_id": "2", "scope_type": "workspace", "workspace_id": "ws"}
    )
    db.add_conversation({"title": "Global chat", "client_id": "1"})
    with _client(db) as client:
        buddy = _create(client)
        attached = client.put(
            "/api/v1/buddies/attachment",
            json={"expected_version": 0, "buddy_id": buddy["id"], "scope_type": "workspace", "scope_id": "ws"},
        )
        assert attached.json()["target"] == {"title": "My workspace", "workspace_id": "ws"}
        listed = client.get("/api/v1/buddies/attachment/conversations").json()
        assert listed["conversations"] == [
            {
                "id": owned_id,
                "title": "Own scoped chat",
                "scope_type": "workspace",
                "workspace_id": "ws",
                "version": 1,
                "assistant_kind": None,
                "assistant_id": None,
                "assistant_name": None,
            }
        ]


def test_starter_preview_resolves_bundled_content_and_dimensions_without_copy(db):
    catalog = PersonaVisualStarterCatalogService(db)
    starter = catalog.get_starter_pack("pixel-migu")
    asset = starter["assets"][0]
    assert asset["width"] > 0
    assert asset["height"] > 0
    content, mime_type = catalog.get_starter_asset_content("pixel-migu", asset["asset_key"])
    assert content.startswith(b"\x89PNG")
    assert mime_type == "image/png"
    with pytest.raises(Exception, match="not found"):
        catalog.get_starter_asset_content("pixel-migu", "../../secrets")


def test_starter_preview_route_requires_auth_and_never_creates_a_profile(db, monkeypatch):
    from fastapi import HTTPException
    from tldw_Server_API.app.api.v1.endpoints import persona

    app = FastAPI()
    app.include_router(persona.router, prefix="/api/v1/persona")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[check_rate_limit] = lambda: None
    monkeypatch.setattr(persona, "is_persona_enabled", lambda: True)

    def denied():
        raise HTTPException(status_code=401, detail="Authentication required")

    app.dependency_overrides[get_request_user] = denied
    url = "/api/v1/persona/visual-starter-packs/pixel-migu/assets/idle-1/content"
    with TestClient(app) as client:
        assert client.get(url).status_code == 401
        app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")
        assert client.get(url.replace("idle-1", "missing")).status_code == 404
    assert db.list_persona_profiles(user_id="1") == []


def test_failed_snapshot_publication_cleans_files_and_leaves_no_profile(db, monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.schemas.buddies import BuddyCreate
    from tldw_Server_API.app.core.Buddy.service import BuddyService

    service = BuddyService(db, "1")

    def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(service.repository, "create", fail)
    with pytest.raises(RuntimeError, match="database unavailable"):
        service.create(BuddyCreate(name="No partial profile", source={"kind": "starter", "starter_id": "pixel-migu"}))
    assert service.list_profiles(limit=50, offset=0) == {"buddies": []}
    assert not list((tmp_path / "1" / "buddies").rglob("*.png"))


def test_workspace_projection_drops_conversation_moved_after_listing(db, monkeypatch):
    from tldw_Server_API.app.core.Buddy.service import BuddyService

    db.upsert_workspace("original", "Original")
    db.upsert_workspace("new", "New")
    conversation_id = db.add_conversation(
        {"title": "Moved", "client_id": "1", "scope_type": "workspace", "workspace_id": "original"}
    )
    with _client(db) as client:
        buddy = _create(client)
        client.put(
            "/api/v1/buddies/attachment",
            json={"expected_version": 0, "buddy_id": buddy["id"], "scope_type": "workspace", "scope_id": "original"},
        )
    original = db.get_conversations_for_user

    def move_after_listing(*args, **kwargs):
        rows = original(*args, **kwargs)
        with db.transaction() as conn:
            conn.execute("UPDATE conversations SET workspace_id = ? WHERE id = ?", ("new", conversation_id))
        return rows

    monkeypatch.setattr(db, "get_conversations_for_user", move_after_listing)
    assert BuddyService(db, "1").conversations("default", limit=50, offset=0)["conversations"] == []


def test_workspace_activity_acks_exact_result_without_clearing_newer_result(db):
    db.upsert_workspace("ws", "Workspace")
    character_id = db.add_character_card({"name": "User"})
    conversation_id = db.add_conversation(
        {
            "title": "Research",
            "client_id": "1",
            "scope_type": "workspace",
            "workspace_id": "ws",
            "character_id": character_id,
        }
    )
    result_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Result " + "x" * 2500,
            "client_id": "1",
            "timestamp": "2026-09-08T01:00:00Z",
        }
    )
    user_message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "A question",
            "client_id": "1",
            "timestamp": "2026-09-08T02:00:00Z",
        }
    )
    with _client(db) as client:
        buddy = _create(client)
        client.put(
            "/api/v1/buddies/attachment",
            json={"expected_version": 0, "buddy_id": buddy["id"], "scope_type": "workspace", "scope_id": "ws"},
        )
        response = client.get("/api/v1/buddies/attachment/activity")
        assert response.status_code == 200, response.text
        item = response.json()["items"][0]
        assert item["result"]["id"] == result_id
        assert len(item["result"]["content"]) == 2000
        assert item["acknowledged"] is False
        invalid = client.post(
            "/api/v1/buddies/attachment/acknowledgements",
            json={"conversation_id": conversation_id, "result_message_id": user_message_id},
        )
        assert invalid.status_code == 404
        ack = client.post(
            "/api/v1/buddies/attachment/acknowledgements",
            json={"conversation_id": conversation_id, "result_message_id": result_id},
        )
        assert ack.status_code == 200, ack.text
        assert client.get("/api/v1/buddies/attachment/activity").json()["items"][0]["acknowledged"] is True
        newer_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "New result",
                "client_id": "1",
                "timestamp": "2026-09-08T03:00:00Z",
            }
        )
        client.post(
            "/api/v1/buddies/attachment/acknowledgements",
            json={"conversation_id": conversation_id, "result_message_id": result_id},
        )
        newer = client.get("/api/v1/buddies/attachment/activity").json()["items"][0]
        assert newer["result"]["id"] == newer_id
        assert newer["acknowledged"] is False


def test_result_ack_rejects_other_attachment_destination(db):
    first_id = db.add_conversation({"title": "First", "client_id": "1"})
    second_id = db.add_conversation({"title": "Second", "client_id": "1"})
    second_result = db.add_message(
        {"conversation_id": second_id, "sender": "assistant", "content": "Second result", "client_id": "1"}
    )
    with _client(db) as client:
        buddy = _create(client)
        client.put(
            "/api/v1/buddies/attachment",
            json={"expected_version": 0, "buddy_id": buddy["id"], "scope_type": "conversation", "scope_id": first_id},
        )
        result = client.post(
            "/api/v1/buddies/attachment/acknowledgements",
            json={"conversation_id": second_id, "result_message_id": second_result},
        )
        assert result.status_code == 404
