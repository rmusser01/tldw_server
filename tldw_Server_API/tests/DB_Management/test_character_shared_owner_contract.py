"""Character visibility and creation agree across two actual database owners."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as chats
from tldw_Server_API.app.api.v1.endpoints import characters_endpoint as characters
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import CharacterRateLimiter
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def owners(request, tmp_path):
    """PG owners share one official fixture database; SQLite uses per-user files."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    first = CharactersRAGDB(tmp_path / "1" / "ChaChaNotes.db", client_id="1", backend=backend)
    second = CharactersRAGDB(tmp_path / "2" / "ChaChaNotes.db", client_id="2", backend=backend)
    try:
        yield SimpleNamespace(first=first, second=second, postgres=request.param == "postgres")
    finally:
        first.close_all_connections()
        second.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _seed_defaults(owners):
    """Run the actual dependency maintenance bootstrap for both owners."""
    first = deps._ensure_default_character(owners.first)
    owners.first.close_connection()
    second = deps._ensure_default_character(owners.second)
    owners.second.close_connection()
    assert first is not None and second is not None
    return first, second


def _client(db, monkeypatch):
    app = FastAPI()
    app.include_router(characters.router, prefix="/api/v1/characters")
    app.include_router(chats.router, prefix="/api/v1/chats")
    app.dependency_overrides[deps.get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[chats.get_request_user] = lambda: SimpleNamespace(id=int(db.client_id))
    app.dependency_overrides[chats.require_expected_user] = lambda: None
    # Keep real quota counting and the factory; frequency and sync transport are unrelated.
    limiter = CharacterRateLimiter(enabled=False, max_chats_per_user=100)
    monkeypatch.setattr(chats, "get_character_rate_limiter", lambda: limiter)
    monkeypatch.setattr(chats, "_active_chat_sync_service", lambda *_args: None)
    return TestClient(app, raise_server_exceptions=False)


def _create(client, character):
    return client.post(
        "/api/v1/chats/",
        json={
            "character_id": str(character),
            "state": "in-progress",
            "source": "webui-character-chat",
            "scope_type": "global",
        },
    )


def _foreign(owners):
    # Keep the foreign ID absent from the other user's SQLite file too.
    owners.first.add_character_card({"name": "First owner padding"})
    return owners.first.add_character_card(
        {"name": "Private owner-one archivist", "system_prompt": "Owner-one secret", "tags": ["private_owner"]}
    )


def test_default_bootstrap_returns_the_current_owners_character(owners):
    _first, second = _seed_defaults(owners)
    card = owners.second.get_character_card_by_id(second)
    assert card["name"] == deps.DEFAULT_CHARACTER_NAME
    assert card["client_id"] == "2"
    assert deps._ensure_default_character(owners.second) == second


def test_visible_bootstrap_default_can_create_a_character_chat(owners, monkeypatch):
    _first, second = _seed_defaults(owners)
    with _client(owners.second, monkeypatch) as client:
        detail = client.get(f"/api/v1/characters/{second}")
        assert detail.status_code == 200, detail.text
        response = _create(client, second)
    assert response.status_code == 201, response.text
    stored = owners.second.get_conversation_by_id(response.json()["id"])
    assert stored["client_id"] == "2"
    assert stored["character_id"] == second


def test_user_created_owned_character_can_create_chat(owners, monkeypatch):
    character = owners.second.add_character_card({"name": "Alice archivist", "first_message": "Hello Alice"})
    owners.second.close_connection()
    with _client(owners.second, monkeypatch) as client:
        response = _create(client, character)
    assert response.status_code == 201, response.text
    assert owners.second.get_conversation_by_id(response.json()["id"])["client_id"] == "2"


@pytest.mark.parametrize("route", ["detail", "list", "query"])
def test_character_routes_do_not_expose_another_owners_card(owners, monkeypatch, route):
    foreign = _foreign(owners)
    owners.first.close_connection()
    with _client(owners.second, monkeypatch) as client:
        if route == "detail":
            response = client.get(f"/api/v1/characters/{foreign}")
            assert response.status_code == 404, response.text
        else:
            path = "/api/v1/characters/" if route == "list" else "/api/v1/characters/query"
            response = client.get(path)
            assert response.status_code == 200, response.text
            rows = response.json() if route == "list" else response.json()["items"]
            assert "Private owner-one archivist" not in [row["name"] for row in rows]


def test_default_name_lookup_cannot_reuse_another_owners_character(owners):
    owners.first.add_character_card({"name": "Private same-name lookup"})
    owners.first.close_connection()
    assert owners.second.get_character_card_by_name("Private same-name lookup") is None


def test_two_owners_can_create_the_same_character_name(owners):
    first = owners.first.add_character_card({"name": "Shared display name"})
    second = owners.second.add_character_card({"name": "Shared display name"})
    assert owners.first.get_character_card_by_id(first)["client_id"] == "1"
    assert owners.second.get_character_card_by_id(second)["client_id"] == "2"


@pytest.mark.parametrize("kind", ["missing", "deleted", "foreign"])
def test_unavailable_character_cannot_create_a_partial_chat(owners, monkeypatch, kind):
    character = 99999
    if kind == "foreign":
        character = _foreign(owners)
        owners.first.close_connection()
    elif kind == "deleted":
        character = owners.second.add_character_card({"name": "Deleted Alice character"})
        assert owners.second.soft_delete_character_card(character, expected_version=1)
    owners.second.close_connection()
    with _client(owners.second, monkeypatch) as client:
        response = _create(client, character)
    assert response.status_code in {400, 404}, response.text
    assert owners.second.count_conversations_for_user("2") == 0


def test_foreign_character_update_cannot_transfer_ownership(owners, monkeypatch):
    foreign = _foreign(owners)
    before = owners.first.get_character_card_by_id(foreign)
    owners.first.close_connection()
    with _client(owners.second, monkeypatch) as client:
        response = client.put(
            f"/api/v1/characters/{foreign}",
            params={"expected_version": 1},
            json={"description": "Unauthorized replacement"},
        )
    after = owners.first.get_character_card_by_id(foreign)
    assert after == before
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("operation", ["batch", "setup-id", "setup-query", "fts", "tags"])
def test_other_character_read_paths_exclude_foreign_cards(owners, operation):
    foreign = _foreign(owners)
    owners.first.close_connection()
    db = owners.second
    if operation == "batch":
        assert foreign not in db.get_character_cards_by_ids([foreign])
    elif operation == "setup-id":
        assert db.get_character_setup_option_by_id(foreign) is None
    elif operation == "setup-query":
        rows, count = db.query_character_setup_options(query="Private owner-one archivist")
        assert rows == [] and count == 0
    elif operation == "fts":
        assert db.search_character_cards("archivist") == []
    else:
        assert db.search_character_cards_by_tags(["private_owner"]) == []


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_foreign_card_lifecycle_cannot_change_state_or_owner(owners, operation):
    foreign = _foreign(owners)
    version = 1
    if operation == "restore":
        assert owners.first.soft_delete_character_card(foreign, expected_version=1)
        version = 2
    before = owners.first.get_character_card_by_id(foreign, include_deleted=True)
    owners.first.close_connection()
    method = owners.second.soft_delete_character_card if operation == "delete" else owners.second.restore_character_card
    try:
        result = method(foreign, expected_version=version)
    except ConflictError:
        result = None
    assert owners.first.get_character_card_by_id(foreign, include_deleted=True) == before
    assert result is not True


@pytest.mark.parametrize("operation", ["get", "list", "search", "add", "update", "delete"])
def test_exemplar_child_paths_preserve_parent_ownership(owners, operation):
    foreign = _foreign(owners)
    exemplar = owners.first.add_character_exemplar(foreign, {"text": "Private exemplar"})
    before = owners.first.get_character_exemplar_by_id(foreign, exemplar["id"], include_deleted=True)
    owners.first.close_connection()
    db = owners.second
    if operation == "get":
        assert db.get_character_exemplar_by_id(foreign, exemplar["id"]) is None
    elif operation == "list":
        assert db.list_character_exemplars(foreign) == []
    elif operation == "search":
        assert db.search_character_exemplars(foreign, query="Private", emotion="other", scenario="other") == ([], 0)
    elif operation == "add":
        with pytest.raises(InputError):
            db.add_character_exemplar(foreign, {"text": "Injected exemplar"})
    elif operation == "update":
        assert db.update_character_exemplar(foreign, exemplar["id"], {"text": "Replaced"}) is None
    else:
        assert db.soft_delete_character_exemplar(foreign, exemplar["id"]) is False
    assert owners.first.get_character_exemplar_by_id(foreign, exemplar["id"], include_deleted=True) == before


def test_owned_card_and_exemplar_lifecycle_is_preserved(owners):
    db = owners.second
    character = db.add_character_card({"name": "Owned lifecycle"})
    exemplar = db.add_character_exemplar(character, {"text": "Original"})
    assert db.update_character_card(character, {"description": "Owned edit"}, expected_version=1)
    assert db.update_character_exemplar(character, exemplar["id"], {"text": "Updated"})["text"] == "Updated"
    assert db.get_character_exemplar_by_id(character, exemplar["id"])["text"] == "Updated"
    assert db.soft_delete_character_exemplar(character, exemplar["id"])
    assert db.get_character_exemplar_by_id(character, exemplar["id"]) is None
    assert db.soft_delete_character_card(character, expected_version=2)
    assert db.restore_character_card(character, expected_version=3)
    card = db.get_character_card_by_id(character)
    assert (card["client_id"], card["version"], card["description"]) == ("2", 4, "Owned edit")


def test_character_writes_remain_owned_by_the_callers_transaction(owners):
    db = owners.second
    character = db.add_character_card({"name": "Caller transaction"})
    before = db.get_character_card_by_id(character)
    db.close_connection()
    with pytest.raises(RuntimeError, match="caller rollback"):
        with db.transaction():
            assert db.update_character_card(character, {"description": "Rollback"}, expected_version=1)
            db.add_character_exemplar(character, {"text": "Rollback exemplar"})
            assert db.get_character_card_by_id(character)["description"] == "Rollback"
            raise RuntimeError("caller rollback")
    assert db.get_character_card_by_id(character) == before
    assert db.list_character_exemplars(character) == []


def test_sqlite_shared_file_preserves_sync_client_semantics(tmp_path):
    """A SQLite client_id identifies the last syncing device, not a separate tenant."""
    path = tmp_path / "single-owner.db"
    first = CharactersRAGDB(path, client_id="device-first")
    second = CharactersRAGDB(path, client_id="device-second")
    try:
        character = first.add_character_card({"name": "Shared per-user file"})
        assert second.get_character_card_by_id(character)["client_id"] == "device-first"
        assert second.update_character_card(character, {"description": "Device edit"}, expected_version=1)
        assert first.get_character_card_by_id(character)["client_id"] == "device-second"
    finally:
        first.close_all_connections()
        second.close_all_connections()


def test_deleted_reads_and_bulk_tag_changes_still_exclude_foreign_cards(owners):
    foreign = _foreign(owners)
    exemplar = owners.first.add_character_exemplar(foreign, {"text": "Deleted foreign child"})
    assert owners.first.soft_delete_character_exemplar(foreign, exemplar["id"])
    assert owners.first.soft_delete_character_card(foreign, expected_version=1)
    before = owners.first.get_character_card_by_id(foreign, include_deleted=True)
    owners.first.close_connection()
    db = owners.second
    assert db.get_character_card_by_id(foreign, include_deleted=True) is None
    assert db.get_character_card_by_name(before["name"], include_deleted=True) is None
    assert foreign not in db.get_character_cards_by_ids([foreign], include_deleted=True)
    assert db.get_character_setup_option_by_id(foreign, include_deleted=True) is None
    assert db.get_character_exemplar_by_id(foreign, exemplar["id"], include_deleted=True) is None
    assert db.search_character_exemplars(foreign) == ([], 0)
    result = db.manage_character_tags(operation="rename", source_tag="private_owner", target_tag="changed")
    assert result["updated_count"] == 0
    assert owners.first.get_character_card_by_id(foreign, include_deleted=True) == before


def test_owned_query_setup_tags_and_noop_keep_current_semantics(owners):
    db = owners.second
    character = db.add_character_card({"name": "Owned searchable Citrine", "tags": ["owned_tag"]})
    assert db.update_character_card(character, {}, expected_version=999)
    assert db.get_character_card_by_id(character)["version"] == 1
    assert character in db.get_character_cards_by_ids([character])
    assert db.get_character_setup_option_by_id(character)["id"] == character
    rows, count = db.query_character_setup_options(query="Citrine")
    assert count == 1 and rows[0]["id"] == character
    assert db.search_character_cards("Citrine")[0]["id"] == character
    result = db.manage_character_tags(operation="rename", source_tag="owned_tag", target_tag="renamed")
    assert result["updated_count"] == 1
    assert db.search_character_cards_by_tags(["renamed"])[0]["id"] == character


def test_created_behavior_snapshot_survives_later_owned_card_edits(owners, monkeypatch):
    db = owners.second
    character = db.add_character_card({"name": "Frozen Citrine", "system_prompt": "Original frozen instruction"})
    db.close_connection()
    with _client(db, monkeypatch) as client:
        response = _create(client, character)
    assert response.status_code == 201, response.text
    conversation = response.json()["id"]
    before = db.execute_query("SELECT canonical_json FROM conversation_behavior_snapshots WHERE conversation_id = ?", (conversation,), read_only=True).fetchone()["canonical_json"]
    assert "Original frozen instruction" in before
    assert db.update_character_card(character, {"system_prompt": "Changed instruction"}, expected_version=1)
    after = db.execute_query("SELECT canonical_json FROM conversation_behavior_snapshots WHERE conversation_id = ?", (conversation,), read_only=True).fetchone()["canonical_json"]
    assert after == before


def test_postgres_foreign_noop_and_stale_parent_preflight_do_not_authorize_writes(pg_database_config, tmp_path, monkeypatch):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    first = CharactersRAGDB(tmp_path / "first.db", client_id="1", backend=backend)
    second = CharactersRAGDB(tmp_path / "second.db", client_id="2", backend=backend)
    try:
        character = first.add_character_card({"name": "Stale parent preflight"})
        row = first.get_character_card_by_id(character)
        first.close_connection()
        with pytest.raises(ConflictError):
            second.update_character_card(character, {}, expected_version=1)
        # Simulate a parent result read before an ownership change: INSERT must still guard it.
        monkeypatch.setattr(second.character_store, "get_character_card_by_id", lambda *_args, **_kwargs: row)
        with pytest.raises(InputError):
            second.add_character_exemplar(character, {"text": "Must not be inserted"})
        assert first.list_character_exemplars(character) == []
        assert first.get_character_card_by_id(character) == row
    finally:
        first.close_all_connections()
        second.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize("use_postgres", [False, pytest.param(True, marks=pytest.mark.postgres)], ids=["sqlite", "postgres"])
@pytest.mark.asyncio
async def test_real_cold_dependency_bootstrap_list_and_factory_share_owner(request, tmp_path, monkeypatch, use_postgres):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    from cachetools import LRUCache

    from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as db_module
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ChaChaOperationMiddleware
    from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeManager
    from tldw_Server_API.app.main import app as main_app

    backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config")) if use_postgres else None
    monkeypatch.setattr(db_module, "get_content_backend", lambda _config: backend)
    monkeypatch.setattr(deps.DatabasePaths, "get_user_base_directory", staticmethod(lambda user: tmp_path / str(user)))
    monkeypatch.setattr(deps.DatabasePaths, "get_chacha_db_path", staticmethod(lambda user: tmp_path / str(user) / "ChaChaNotes.db"))
    cache = LRUCache(maxsize=4)
    monkeypatch.setattr(deps, "_chacha_db_instances", cache)
    monkeypatch.setattr(deps, "_chacha_db_init_events", {})
    monkeypatch.setattr(deps, "_chacha_db_init_errors", {})
    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", ChaChaRuntimeManager())
    monkeypatch.setattr(deps, "_CHACHA_SHUTTING_DOWN", False)
    monkeypatch.setattr(deps, "_chacha_default_char_tasks", set())
    monkeypatch.delitem(main_app.dependency_overrides, deps.get_chacha_db_for_user, raising=False)
    executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(deps, "_get_chacha_executor", lambda: executor)
    monkeypatch.setattr(chats, "get_character_rate_limiter", lambda: CharacterRateLimiter(enabled=False, max_chats_per_user=100))
    monkeypatch.setattr(chats, "_active_chat_sync_service", lambda *_args: None)
    principal = {"id": 1}
    app = FastAPI()
    app.add_middleware(ChaChaOperationMiddleware)
    app.include_router(characters.router, prefix="/api/v1/characters")
    app.include_router(chats.router, prefix="/api/v1/chats")
    app.dependency_overrides[deps.get_request_user] = lambda: SimpleNamespace(id=principal["id"])
    app.dependency_overrides[chats.get_request_user] = lambda: SimpleNamespace(id=principal["id"])
    app.dependency_overrides[chats.require_expected_user] = lambda: None
    defaults = []
    try:
        assert not cache
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
            for user in (1, 2):
                principal["id"] = user
                cold = await client.get("/api/v1/characters/")
                assert cold.status_code == 200, cold.text
                if deps._chacha_default_char_tasks:
                    await asyncio.gather(*tuple(deps._chacha_default_char_tasks))
                listed = await client.get("/api/v1/characters/")
                assert listed.status_code == 200, listed.text
                default = next(card for card in listed.json() if card["name"] == deps.DEFAULT_CHARACTER_NAME)
                defaults.append(default["id"])
                db = cache[str(tmp_path / str(user))]
                assert db.client_id == str(user)
                assert db.get_character_card_by_id(default["id"])["client_id"] == str(user)
                db.close_connection()
                response = await client.post("/api/v1/chats/", json={"character_id": str(default["id"]), "state": "in-progress", "source": "webui-character-chat", "scope_type": "global"})
                assert response.status_code == 201, response.text
            assert len(cache) == 2
            if use_postgres:
                assert defaults[0] != defaults[1]
    finally:
        if deps._chacha_default_char_tasks:
            await asyncio.gather(*tuple(deps._chacha_default_char_tasks))
        executor.shutdown(wait=True)
        for db in tuple(cache.values()):
            db.close_all_connections()
        cache.clear()
        if backend is not None:
            backend.get_pool().close_all()
