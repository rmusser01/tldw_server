"""Shared PostgreSQL Notes must honor the selected owner's persistence boundary."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def note_owners(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    alice = CharactersRAGDB(tmp_path / "2" / "ChaChaNotes.db", client_id="2", backend=backend)
    bob = CharactersRAGDB(tmp_path / "3" / "ChaChaNotes.db", client_id="3", backend=backend)
    try:
        private_note = alice.add_note("Alice private Citrine", "Private amber research fixture")
        deleted_note = alice.add_note("Alice deleted Citrine", "Private tombstone fixture")
        alice.soft_delete_note(deleted_note, expected_version=1)
        own_note = bob.add_note("Bob private Indigo", "Own indigo research fixture")
        alice.close_connection()
        bob.close_connection()
        yield SimpleNamespace(alice=alice, bob=bob, private=private_note, deleted=deleted_note, own=own_note)
    finally:
        alice.close_all_connections()
        bob.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.parametrize("operation", ["list", "count", "detail", "deleted", "batch", "search"])
def test_note_reads_exclude_foreign_owner(note_owners, operation):
    db = note_owners.bob
    if operation == "list":
        assert [row["id"] for row in db.list_notes()] == [note_owners.own]
    elif operation == "count":
        assert db.count_notes() == 1
    elif operation == "detail":
        assert db.get_note_by_id(note_owners.private) is None
    elif operation == "deleted":
        assert db.list_deleted_notes() == []
    elif operation == "batch":
        assert [row["id"] for row in db.get_notes_batch([note_owners.private, note_owners.own])] == [note_owners.own]
    else:
        assert db.search_notes("Citrine") == []


@pytest.mark.parametrize("operation", ["update", "soft-delete", "hard-delete", "restore"])
def test_foreign_note_mutation_rejected(note_owners, operation):
    f = note_owners
    note_id = f.deleted if operation == "restore" else f.private
    before = f.alice.get_note_by_id(note_id, include_deleted=True)
    try:
        if operation == "update":
            result = f.bob.update_note(note_id, {"content": "Unauthorized mutation probe"}, expected_version=1)
        elif operation == "soft-delete":
            result = f.bob.soft_delete_note(note_id, expected_version=1)
        elif operation == "hard-delete":
            result = f.bob.delete_note(note_id, hard_delete=True)
        else:
            result = f.bob.restore_note(note_id, expected_version=2)
    except ConflictError:
        result = False
    assert not result
    assert f.alice.get_note_by_id(note_id, include_deleted=True) == before


@pytest.mark.parametrize(
    "operation",
    [
        "keyword-list",
        "keyword-id",
        "keyword-text",
        "foreign-note-keyword-read",
        "foreign-note-link",
        "foreign-keyword-link",
    ],
)
def test_note_linked_owner_edges(note_owners, operation):
    f = note_owners
    f.alice.add_keyword("Alice sequence separator")
    foreign_keyword = f.alice.add_keyword("Alice private amber keyword")
    own_keyword = f.bob.add_keyword("Bob own indigo keyword")
    f.alice.link_note_to_keyword(f.private, foreign_keyword)
    f.bob.link_note_to_keyword(f.own, own_keyword)
    if operation == "keyword-list":
        assert [x["id"] for x in f.bob.list_keywords()] == [own_keyword]
    elif operation == "keyword-id":
        assert f.bob.get_keyword_by_id(foreign_keyword) is None
    elif operation == "keyword-text":
        assert f.bob.get_keyword_by_text("Alice private amber keyword") is None
    elif operation == "foreign-note-keyword-read":
        assert f.bob.get_keywords_for_note(f.private) == []
    else:
        try:
            result = (
                f.bob.link_note_to_keyword(f.private, own_keyword)
                if operation == "foreign-note-link"
                else f.bob.link_note_to_keyword(f.own, foreign_keyword)
            )
        except (ConflictError, CharactersRAGDBError):
            result = False
        assert not result


@pytest.mark.parametrize(
    "operation",
    [
        "folder-list",
        "folder-name",
        "foreign-note-folders",
        "foreign-note-folder-write",
        "collection-list",
        "keyword-same-text",
    ],
)
def test_organization_owner_edges(note_owners, operation):
    f = note_owners
    if operation.startswith("folder") or "folder" in operation:
        f.alice.create_note_folder_path("Private amber folder")
        f.alice.sync_note_folders(f.private, ["Private amber folder"])
        if operation == "folder-list":
            assert f.bob.list_note_folders() == []
        elif operation == "folder-name":
            assert f.bob.get_note_folder_by_path("Private amber folder") is None
        elif operation == "foreign-note-folders":
            assert f.bob.get_note_folders_for_note(f.private) == []
        else:
            try:
                result = f.bob.sync_note_folders(f.private, ["Foreign mutation folder"])
            except (ConflictError, CharactersRAGDBError):
                result = []
            assert result == []
    elif operation == "collection-list":
        f.alice.add_keyword_collection("Private collection")
        assert f.bob.list_keyword_collections() == []
    else:
        f.alice.add_keyword("Same ordinary keyword")
        own = f.bob.add_keyword("Same ordinary keyword")
        assert f.bob.get_keyword_by_id(own)["client_id"] == "3"


@pytest.mark.parametrize("note_owners", ["postgres"], indirect=True)
def test_restricted_role_notes_are_already_rls_filtered(note_owners):
    f = note_owners
    backend = f.bob.backend
    f.alice.add_keyword("Private role keyword")
    f.alice.create_note_folder_path("Private role folder")
    role = backend.escape_identifier(f"uat209_reader_{uuid4().hex[:12]}")
    created = False
    try:
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
            backend.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}", connection=conn
            )
            backend.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}", connection=conn)
            backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
        created = True
        with chacha_operation(independent=True), f.bob.transaction() as conn:
            conn.execute(f"SET LOCAL ROLE {role}")
            flags = conn.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
            assert not flags["rolsuper"] and not flags["rolbypassrls"]
            assert conn.execute("SELECT current_setting('app.current_user_id') AS owner").fetchone()["owner"] == "3"
            assert [x["id"] for x in f.bob.list_notes()] == [f.own]
            assert f.bob.get_note_by_id(f.private) is None
            assert f.bob.list_keywords() == []
            assert f.bob.list_note_folders() == []
            owned = f.bob.add_note("Restricted owned note", "Own persistence")
            assert f.bob.get_note_by_id(owned)["client_id"] == "3"
            assert conn.execute("SELECT id FROM notes WHERE id=?", (f.private,)).fetchone() is None
    finally:
        if created:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {role}", connection=conn)
                backend.execute(f"DROP ROLE {role}", connection=conn)


@pytest.mark.parametrize("operation", ["upsert", "tombstone"])
def test_sync_projection_cannot_overwrite_foreign_note(note_owners, operation):
    f = note_owners
    before = f.alice.get_note_by_id(f.private, include_deleted=True)
    arguments = {
        "note_id": f.private,
        "sync_client_id": f.bob.client_id,
        "object_revision": 2,
        "object_hash": "sha256:" + "a" * 64,
    }
    try:
        if operation == "upsert":
            result = f.bob.upsert_note_from_sync(
                title="Owned projection", content="Owned body", conversation_id=None, message_id=None, **arguments
            )
        else:
            result = f.bob.tombstone_note_from_sync(**arguments)
    except (ConflictError, CharactersRAGDBError):
        result = False
    if f.bob.backend is not None and f.bob.backend_type.value == "postgresql":
        assert not result
    elif operation == "upsert":
        # An independent SQLite file may legitimately receive a new sync object UUID.
        assert result
        assert f.bob.get_note_by_id(f.private)["client_id"] == "3"
    assert f.alice.get_note_by_id(f.private, include_deleted=True) == before


def test_owned_note_lifecycle_and_rollback_remain_available(note_owners):
    db = note_owners.bob
    assert db.update_note(note_owners.own, {"content": "Updated own body"}, expected_version=1)
    assert db.get_note_by_id(note_owners.own)["version"] == 2
    with pytest.raises(RuntimeError, match="caller rollback"):
        with db.transaction():
            assert db.update_note(note_owners.own, {"title": "Pending title"}, expected_version=2)
            raise RuntimeError("caller rollback")
    assert db.get_note_by_id(note_owners.own)["title"] == "Bob private Indigo"
    assert db.get_note_by_id(note_owners.own)["version"] == 2
    assert db.soft_delete_note(note_owners.own, expected_version=2)
    assert db.restore_note(note_owners.own, expected_version=3)
    assert db.get_note_by_id(note_owners.own)["version"] == 4
    assert db.delete_note(note_owners.own, hard_delete=True)
    assert db.get_note_by_id(note_owners.own, include_deleted=True) is None


def test_sqlite_reopens_device_labeled_notes_without_user_filter(tmp_path):
    path = tmp_path / "legacy-device.db"
    first = CharactersRAGDB(path, client_id="original-device")
    note = first.add_note("Device note", "Readable local note")
    first.close_all_connections()
    second = CharactersRAGDB(path, client_id="new-device")
    try:
        assert second.get_note_by_id(note)["client_id"] == "original-device"
        assert second.count_notes() == 1
        assert second.update_note(note, {"content": "New device update"}, expected_version=1)
        assert second.get_note_by_id(note)["client_id"] == "new-device"
    finally:
        second.close_all_connections()


@pytest.mark.parametrize(
    "operation", ["rename", "delete", "merge-source", "merge-target", "count", "search", "punctuation-search"]
)
def test_keyword_mutation_and_search_owner_scope(note_owners, operation):
    f = note_owners
    f.alice.add_keyword("Foreign sequence spacer")
    foreign = f.alice.add_keyword("Amber-private")
    own = f.bob.add_keyword("Indigo-owned")
    before = f.alice.get_keyword_by_id(foreign)
    if operation == "count":
        assert f.bob.count_keywords() == 1
    elif operation.endswith("search"):
        assert f.bob.search_keywords("Amber" if operation == "search" else "Amber-private") == []
    else:
        with pytest.raises((ConflictError, CharactersRAGDBError)):
            if operation == "rename":
                f.bob.rename_keyword(foreign, "Unauthorized rename", expected_version=1)
            elif operation == "delete":
                f.bob.soft_delete_keyword(foreign, expected_version=1)
            else:
                f.bob.merge_keywords(
                    source_keyword_id=foreign if operation == "merge-source" else own,
                    target_keyword_id=own if operation == "merge-source" else foreign,
                    expected_source_version=1,
                    expected_target_version=1,
                )
        assert f.alice.get_keyword_by_id(foreign) == before
        assert f.bob.get_keyword_by_id(own)["version"] == 1


@pytest.mark.parametrize(
    "operation", ["detail", "name", "count", "search", "update", "delete", "parent-create", "parent-update"]
)
def test_collection_owner_scope(note_owners, operation):
    f = note_owners
    f.alice.add_keyword_collection("Collection sequence spacer")
    foreign = f.alice.add_keyword_collection("Amber private collection")
    own = f.bob.add_keyword_collection("Indigo own collection")
    before = f.alice.get_keyword_collection_by_id(foreign)
    if operation == "detail":
        assert f.bob.get_keyword_collection_by_id(foreign) is None
    elif operation == "name":
        assert f.bob.get_keyword_collection_by_name("Amber private collection") is None
    elif operation == "count":
        assert f.bob.count_keyword_collections() == 1
    elif operation == "search":
        assert f.bob.search_keyword_collections("Amber") == []
    elif operation == "parent-create" and f.bob.backend_type.value == "sqlite":
        # Historical SQLite collection schema accepts this local integer reference.
        child = f.bob.add_keyword_collection("Legacy local child", parent_id=foreign)
        assert f.bob.get_keyword_collection_by_id(child)["parent_id"] == foreign
    else:
        with pytest.raises((ConflictError, CharactersRAGDBError)):
            if operation == "update":
                f.bob.update_keyword_collection(foreign, {"name": "Unauthorized"}, expected_version=1)
            elif operation == "delete":
                f.bob.soft_delete_keyword_collection(foreign, expected_version=1)
            elif operation == "parent-create":
                f.bob.add_keyword_collection("Wrong parent child", parent_id=foreign)
            else:
                f.bob.update_keyword_collection(own, {"parent_id": foreign}, expected_version=1)
        assert f.alice.get_keyword_collection_by_id(foreign) == before
        assert f.bob.get_keyword_collection_by_id(own)["version"] == 1


@pytest.mark.parametrize(
    "operation", ["seeds", "tag-seeds", "tag-count", "user-count", "keyword-notes", "keyword-count", "matching-count"]
)
def test_graph_and_keyword_note_queries_are_owner_scoped(note_owners, operation):
    f = note_owners
    f.alice.add_keyword("Graph spacer")
    foreign = f.alice.add_keyword("ambertag")
    own = f.bob.add_keyword("indigotag")
    f.alice.link_note_to_keyword(f.private, foreign)
    f.bob.link_note_to_keyword(f.own, own)
    if operation == "seeds":
        assert f.bob.get_all_note_ids_for_graph() == [f.own]
    elif operation == "tag-seeds":
        assert f.bob.get_note_ids_by_tag_for_graph("ambertag") == []
    elif operation == "tag-count":
        assert f.bob.count_notes_per_tag() == {own: 1}
    elif operation == "user-count":
        assert f.bob.count_user_notes() == 1
    elif operation == "keyword-notes":
        assert f.bob.get_notes_for_keyword(foreign) == []
    elif operation == "keyword-count":
        assert f.bob.get_note_counts_for_keywords([foreign, own]).get(foreign, 0) == 0
    else:
        assert f.bob.count_notes_matching("Citrine") == 0


@pytest.mark.parametrize("operation", ["same-name", "restore", "rollback"])
def test_owned_organization_reuse_and_transaction_controls(note_owners, operation):
    f = note_owners
    foreign = f.alice.add_keyword("Shared keyword label")
    own = f.bob.add_keyword("Shared keyword label")
    own_collection = f.bob.add_keyword_collection("Owned collection")
    if operation == "same-name":
        assert f.bob.get_keyword_by_id(own)["client_id"] == "3"
        assert f.alice.get_keyword_by_id(foreign)["client_id"] == "2"
    elif operation == "restore":
        f.bob.soft_delete_keyword(own, expected_version=1)
        assert f.bob.add_keyword("Shared keyword label") == own
        assert f.bob.get_keyword_by_id(own)["version"] == 3
        assert f.alice.get_keyword_by_id(foreign)["version"] == 1
    else:
        with pytest.raises(RuntimeError, match="organization rollback"):
            with f.bob.transaction():
                f.bob.rename_keyword(own, "Pending rename", expected_version=1)
                f.bob.update_keyword_collection(own_collection, {"name": "Pending collection"}, expected_version=1)
                raise RuntimeError("organization rollback")
        assert f.bob.get_keyword_by_id(own)["keyword"] == "Shared keyword label"
        assert f.bob.get_keyword_collection_by_id(own_collection)["name"] == "Owned collection"


@pytest.mark.parametrize("note_owners", ["postgres"], indirect=True)
@pytest.mark.parametrize("operation", ["single", "batch", "reverse", "counts", "search", "search-count", "merge"])
def test_malformed_note_keyword_links_do_not_cross_owners(note_owners, operation):
    f = note_owners
    foreign = f.alice.add_keyword("Foreign amber")
    own = f.bob.add_keyword("Owned indigo")
    destination = f.bob.add_keyword("Owned destination")
    f.bob.link_note_to_keyword(f.own, own)
    with f.alice.transaction() as conn:
        conn.execute(
            "INSERT INTO note_keywords(note_id,keyword_id,created_at) VALUES(?,?,?)",
            (f.own, foreign, f.alice._get_current_utc_timestamp_iso()),
        )
        conn.execute(
            "INSERT INTO note_keywords(note_id,keyword_id,created_at) VALUES(?,?,?)",
            (f.private, own, f.alice._get_current_utc_timestamp_iso()),
        )
    if operation == "single":
        assert [k["id"] for k in f.bob.get_keywords_for_note(f.own)] == [own]
    elif operation == "batch":
        rows = f.bob.get_keywords_for_notes([f.own, f.private])
        assert [k["id"] for k in rows[f.own]] == [own]
        assert rows[f.private] == []
    elif operation == "reverse":
        assert [n["id"] for n in f.bob.get_notes_for_keyword(own)] == [f.own]
    elif operation == "counts":
        assert f.bob.get_note_counts_for_keywords([own, foreign]) == {own: 1}
    elif operation == "search":
        assert f.bob.search_notes_with_keywords(None, ["amber"]) == []
    elif operation == "search-count":
        assert f.bob.count_notes_matching_keywords(None, ["amber"]) == 0
    else:
        result = f.bob.merge_keywords(
            source_keyword_id=own, target_keyword_id=destination, expected_source_version=1, expected_target_version=1
        )
        assert result["merged_note_links"] == 1
        with f.alice.transaction() as conn:
            rows = conn.execute("SELECT keyword_id FROM note_keywords WHERE note_id=?", (f.private,)).fetchall()
            assert [r["keyword_id"] for r in rows] == [own]


@pytest.mark.parametrize("kind", ["conversation", "collection"])
@pytest.mark.parametrize(
    "operation", ["foreign-parent", "foreign-keyword", "foreign-read", "reverse-read", "link-count"]
)
def test_other_keyword_link_parent_scope(note_owners, kind, operation):
    f = note_owners
    f.alice.add_keyword("Keyword integer spacer")
    foreign_kw = f.alice.add_keyword("Foreign link keyword")
    own_kw = f.bob.add_keyword("Owned link keyword")
    if kind == "conversation":
        foreign_char = f.alice.add_character_card({"name": "Foreign source character"})
        own_char = f.bob.add_character_card({"name": "Owned source character"})
        foreign = f.alice.add_conversation({"character_id": foreign_char, "title": "Foreign source"})
        own = f.bob.add_conversation({"character_id": own_char, "title": "Owned source"})
    else:
        f.alice.add_keyword_collection("Collection integer spacer")
        foreign = f.alice.add_keyword_collection("Foreign source")
        own = f.bob.add_keyword_collection("Owned source")
    getattr(f.alice, f"link_{kind}_to_keyword")(foreign, foreign_kw)
    getattr(f.bob, f"link_{kind}_to_keyword")(own, own_kw)
    if operation in {"foreign-parent", "foreign-keyword"}:
        try:
            result = getattr(f.bob, f"link_{kind}_to_keyword")(
                foreign if operation == "foreign-parent" else own,
                own_kw if operation == "foreign-parent" else foreign_kw,
            )
        except (ConflictError, CharactersRAGDBError):
            result = False
        assert not result
    elif operation == "foreign-read":
        assert getattr(f.bob, f"get_keywords_for_{kind}")(foreign) == []
    elif operation == "reverse-read":
        assert getattr(f.bob, f"get_{kind}s_for_keyword")(foreign_kw) == []
    else:
        assert getattr(f.bob, f"count_{kind}_keyword_links")() == 1


@pytest.mark.parametrize("operation", ["create", "update"])
def test_note_rejects_foreign_conversation_parent(note_owners, operation):
    f = note_owners
    character = f.alice.add_character_card({"name": "Foreign parent character"})
    conversation = f.alice.add_conversation({"character_id": character, "title": "Foreign source"})
    with pytest.raises((ConflictError, CharactersRAGDBError)):
        if operation == "create":
            f.bob.add_note("Wrong parent note", "body", conversation_id=conversation)
        else:
            f.bob.update_note(f.own, {"conversation_id": conversation}, expected_version=1)
    assert f.bob.get_note_by_id(f.own)["version"] == 1


@pytest.mark.parametrize("operation", ["same-path", "source-keys"])
def test_folder_paths_and_source_keys_preserve_other_owner(note_owners, operation):
    f = note_owners
    foreign = f.alice.create_note_folder_path("Shared/Path")
    if operation == "same-path":
        own = f.bob.create_note_folder_path("Shared/Path")
        with f.bob.transaction() as conn:
            owner = conn.execute("SELECT client_id FROM note_folders WHERE id=?", (own["id"],)).fetchone()
        assert owner["client_id"] == "3"
    else:
        f.alice.sync_note_source_folders(f.private, 1, ["Shared/Path"])
        # Source IDs are globally assigned by ingestion. A conflicting foreign
        # provenance key may reject, but must never be reassigned to this owner.
        try:
            f.bob.sync_note_source_folders(f.own, 1, ["Shared/Path"])
        except CharactersRAGDBError:
            pass
        f.bob.sync_note_source_folders(f.own, 2, ["Owned/Source"])
        assert f.bob.get_note_folders_for_note(f.own)
        with f.alice.transaction() as conn:
            rows = conn.execute("SELECT folder_id FROM note_folder_source_keys WHERE source_id=1").fetchall()
        assert foreign["id"] in [row["folder_id"] for row in rows]


@pytest.mark.parametrize("use_postgres", [False, pytest.param(True, marks=pytest.mark.postgres)])
@pytest.mark.asyncio
async def test_real_cold_notes_dependency_keeps_two_http_owners_separate(request, tmp_path, monkeypatch, use_postgres):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import AsyncMock

    import httpx
    from cachetools import LRUCache
    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
    from tldw_Server_API.app.api.v1.endpoints import notes
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as db_module
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ChaChaOperationMiddleware
    from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeManager
    from tldw_Server_API.app.main import app as main_app

    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config")) if use_postgres else None
    )
    monkeypatch.setattr(db_module, "get_content_backend", lambda _config: backend)
    monkeypatch.setattr(deps.DatabasePaths, "get_user_base_directory", staticmethod(lambda user: tmp_path / str(user)))
    monkeypatch.setattr(
        deps.DatabasePaths, "get_chacha_db_path", staticmethod(lambda user: tmp_path / str(user) / "ChaChaNotes.db")
    )
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
    monkeypatch.setattr(notes, "_active_notes_sync_service", lambda _user: None)
    principal = {"id": 2}
    app = FastAPI()
    app.add_middleware(ChaChaOperationMiddleware)
    app.include_router(notes.router, prefix="/api/v1/notes")
    app.dependency_overrides[notes.get_request_user] = lambda: User(
        id=principal["id"], username="fixture", is_active=True, is_admin=False
    )
    app.dependency_overrides[notes.require_expected_user] = lambda: None
    app.dependency_overrides[notes.get_rate_limiter_dep] = lambda: SimpleNamespace(
        check_user_rate_limit=AsyncMock(return_value=(True, {}))
    )
    created = {}
    try:
        assert not cache
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://fixture"
        ) as client:
            for user in (2, 3):
                principal["id"] = user
                cold = await client.get("/api/v1/notes/")
                assert cold.status_code == 200, cold.text
                assert cold.json()["notes"] == []
                if deps._chacha_default_char_tasks:
                    await asyncio.gather(*tuple(deps._chacha_default_char_tasks))
                response = await client.post(
                    "/api/v1/notes/", json={"title": f"Owner {user} fixture", "content": "Private note body"}
                )
                assert response.status_code == 201, response.text
                created[user] = response.json()["id"]
                db = cache[str(tmp_path / str(user))]
                assert db.client_id == str(user)
            for user, other in ((2, 3), (3, 2)):
                principal["id"] = user
                listed = await client.get("/api/v1/notes/", params={"include_keywords": True})
                assert listed.status_code == 200, listed.text
                assert [row["id"] for row in listed.json()["notes"]] == [created[user]]
                foreign = await client.get(f"/api/v1/notes/{created[other]}")
                assert foreign.status_code == 404, foreign.text
                saved = await client.patch(
                    f"/api/v1/notes/{created[user]}", json={"content": "Saved own body", "expected_version": 1}
                )
                assert saved.status_code == 200, saved.text
                foreign_write = await client.patch(
                    f"/api/v1/notes/{created[other]}", json={"content": "Unauthorized", "expected_version": 1}
                )
                assert foreign_write.status_code == 404, foreign_write.text
                reloaded = await client.get(f"/api/v1/notes/{created[user]}")
                assert reloaded.status_code == 200, reloaded.text
                assert reloaded.json()["content"] == "Saved own body"
    finally:
        if deps._chacha_default_char_tasks:
            await asyncio.gather(*tuple(deps._chacha_default_char_tasks))
        executor.shutdown(wait=True)
        for db in tuple(cache.values()):
            db.close_all_connections()
        cache.clear()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.parametrize("operation", ["tag-edges", "sync-snapshot"])
def test_graph_edge_and_sync_snapshot_reads_are_owner_scoped(note_owners, operation):
    f = note_owners
    f.alice.add_keyword("Snapshot spacer")
    foreign = f.alice.add_keyword("Foreign snapshot keyword")
    own = f.bob.add_keyword("Owned snapshot keyword")
    f.alice.link_note_to_keyword(f.private, foreign)
    f.bob.link_note_to_keyword(f.own, own)
    if operation == "tag-edges":
        edges = f.bob.get_note_tag_edges([f.private, f.own])
        assert [(row["note_id"], row["keyword_id"]) for row in edges] == [(f.own, own)]
    else:
        with pytest.raises(ConflictError):
            f.bob.keyword_store.synchronized_merge_snapshot(
                source_keyword_id=foreign, target_keyword_id=own, expected_source_version=1, expected_target_version=1
            )


@pytest.mark.parametrize("mode", ["nested", "raw-begin", "supplied-connection"])
def test_note_and_organization_changes_preserve_caller_rollback(note_owners, mode):
    f = note_owners
    kw = f.bob.add_keyword("Caller keyword")
    collection = f.bob.add_keyword_collection("Caller collection")
    before = f.bob.get_note_by_id(f.own)
    if mode == "raw-begin":
        conn = f.bob.get_connection()
        conn.execute("BEGIN")
        try:
            assert f.bob.update_note(f.own, {"content": "Pending raw write"}, expected_version=1, conn=conn)
            assert f.bob.get_note_by_id(f.own)["version"] == 2
        finally:
            conn.execute("ROLLBACK")
    else:
        with pytest.raises(RuntimeError, match="outer decision"):
            with f.bob.transaction() as conn:
                if mode == "nested":
                    with f.bob.transaction():
                        f.bob.rename_keyword(kw, "Pending caller keyword", expected_version=1)
                        f.bob.link_collection_to_keyword(collection, kw)
                        f.bob.sync_note_folders(f.own, ["Pending/Folder"])
                f.bob.update_note(f.own, {"content": "Pending caller body"}, expected_version=1, conn=conn)
                raise RuntimeError("outer decision")
    assert f.bob.get_note_by_id(f.own) == before
    assert f.bob.get_keyword_by_id(kw)["keyword"] == "Caller keyword"
    assert f.bob.get_keywords_for_collection(collection) == []
    assert f.bob.get_note_folders_for_note(f.own) == []


@pytest.mark.parametrize("note_owners", ["postgres"], indirect=True)
def test_foreign_version_change_between_lookup_and_mutation_does_not_transfer_note(note_owners, monkeypatch):
    f = note_owners
    before = f.bob.get_note_by_id(f.own)
    original = f.bob._get_current_db_version

    def reassign(conn, table, column, item, **kwargs):
        version = original(conn, table, column, item, **kwargs)
        if table == "notes" and item == f.own:
            conn.execute("UPDATE notes SET client_id=? WHERE id=?", ("2", f.own))
        return version

    monkeypatch.setattr(f.bob, "_get_current_db_version", reassign)
    with pytest.raises(ConflictError):
        f.bob.update_note(f.own, {"content": "Must not transfer back"}, expected_version=1)
    assert f.bob.get_note_by_id(f.own) == before


@pytest.mark.parametrize("case", ["owned-device-label", "foreign-canonical-label", "deleted-conversation"])
def test_note_message_parent_authority_follows_its_conversation(note_owners, case):
    f = note_owners
    source_db = f.alice if case == "foreign-canonical-label" else f.bob
    character = source_db.add_character_card({"name": "Message source character"})
    conversation = source_db.add_conversation({"character_id": character, "title": "Message source"})
    message = source_db.add_message(
        {
            "conversation_id": conversation,
            "sender": "user",
            "content": "Source message",
            "client_id": "sync-device-label" if case == "owned-device-label" else "3",
        }
    )
    if case == "deleted-conversation":
        with source_db.transaction() as conn:
            conn.execute("UPDATE conversations SET deleted=? WHERE id=?", (True, conversation))
    if case == "owned-device-label" or (case == "deleted-conversation" and f.bob.backend_type.value == "sqlite"):
        note = f.bob.add_note("Owned message note", "body", message_id=message)
        assert f.bob.get_note_by_id(note)["message_id"] == message
    else:
        with pytest.raises((ConflictError, CharactersRAGDBError)):
            f.bob.add_note("Foreign message note", "body", message_id=message)
