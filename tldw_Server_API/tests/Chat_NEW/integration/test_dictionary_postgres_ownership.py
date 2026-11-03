"""Real non-bypass PostgreSQL catalog and native capture owner boundaries."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Character_Chat import character_conversation_factory
from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.DB_Management.backends.base import AuthorizationDeniedError, DatabaseError
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import build_materialized_behavior_settings
from tldw_Server_API.app.core.DB_Management.chacha.dictionary_schema import DICTIONARY_TABLES
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import _mint
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]
MODEL = "qwen2.5:0.5b"
CONFIG = {"ollama_api": {"api_url": "http://127.0.0.1:11434", "model": MODEL}}

# Exact published v82 lazy catalog definitions, before owner columns existed.
# They are fixture data for the supported upgrade, never runtime schema patches.
LEGACY_CATALOG_DDL = (
    "\n                        CREATE TABLE IF NOT EXISTS chat_dictionaries (\n                            id SERIAL PRIMARY KEY,\n                            name TEXT NOT NULL UNIQUE,\n                            description TEXT,\n                            category TEXT,\n                            tags TEXT,\n                            included_dictionary_ids TEXT,\n                            is_active BOOLEAN DEFAULT TRUE,\n                            default_token_budget INTEGER NULL,\n                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            usage_count INTEGER DEFAULT 0,\n                            last_used_at TIMESTAMP NULL,\n                            version INTEGER DEFAULT 1,\n                            deleted BOOLEAN DEFAULT FALSE\n                        )\n                        ",
    "\n                        CREATE TABLE IF NOT EXISTS dictionary_entries (\n                            id SERIAL PRIMARY KEY,\n                            dictionary_id INTEGER NOT NULL,\n                            key TEXT NOT NULL,\n                            content TEXT NOT NULL,\n                            is_regex BOOLEAN DEFAULT FALSE,\n                            probability REAL DEFAULT 1.0,\n                            max_replacements INTEGER DEFAULT 1,\n                            group_name TEXT,\n                            timed_effects TEXT DEFAULT '{}',\n                            enabled BOOLEAN DEFAULT TRUE,\n                            case_sensitive BOOLEAN DEFAULT TRUE,\n                            sort_order INTEGER,\n                            usage_count INTEGER DEFAULT 0,\n                            last_used_at TIMESTAMP NULL,\n                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE\n                        )\n                        ",
    "\n                        CREATE TABLE IF NOT EXISTS dictionary_transform_activity (\n                            id SERIAL PRIMARY KEY,\n                            dictionary_id INTEGER NOT NULL,\n                            chat_id TEXT NULL,\n                            entries_used TEXT NOT NULL,\n                            replacements INTEGER DEFAULT 0,\n                            iterations INTEGER DEFAULT 0,\n                            token_budget_used INTEGER NULL,\n                            original_text_preview TEXT NOT NULL,\n                            processed_text_preview TEXT NOT NULL,\n                            processing_time_ms REAL NULL,\n                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE\n                        )\n                        ",
    "\n                        CREATE TABLE IF NOT EXISTS dictionary_version_history (\n                            id SERIAL PRIMARY KEY,\n                            dictionary_id INTEGER NOT NULL,\n                            revision INTEGER NOT NULL,\n                            source_dictionary_version INTEGER NULL,\n                            change_type TEXT NOT NULL,\n                            summary TEXT NULL,\n                            entry_count INTEGER DEFAULT 0,\n                            dictionary_snapshot TEXT NOT NULL,\n                            entries_snapshot TEXT NOT NULL,\n                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            UNIQUE(dictionary_id, revision),\n                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE\n                        )\n                        ",
)


@pytest.fixture(autouse=True)
def private_test_paths(monkeypatch, tmp_path):
    """Use supported private test storage without changing application guards."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(tmp_path / "monitoring.db"))
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(tmp_path / "system.log"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))


@pytest.fixture
def catalogs(pg_restricted_backend):
    """Actual shared backend; table owner cannot bypass FORCE RLS."""
    alice = CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
    bob = CharactersRAGDB(":memory:", client_id="2", backend=pg_restricted_backend)
    a_service = ChatDictionaryService(alice)
    b_service = ChatDictionaryService(bob)
    did = a_service.create_dictionary("Alice private catalog")
    eid = a_service.add_entry(did, "PRIVATE_TOKEN", "ALICE_PRIVATE_CONTENT")
    try:
        yield SimpleNamespace(alice=alice, bob=bob, a=a_service, b=b_service, did=did, eid=eid,
                              backend=pg_restricted_backend)
    finally:
        alice.close_all_connections()
        bob.close_all_connections()


@pytest.fixture
def catalog_rows(pg_restricted_backend):
    """Seed published catalog rows; exercise capture without bypassing service guards."""
    alice = CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
    bob = CharactersRAGDB(":memory:", client_id="2", backend=pg_restricted_backend)
    with alice.transaction() as conn:
        for statement in LEGACY_CATALOG_DDL:
            conn.execute(statement)
        did = conn.execute("INSERT INTO chat_dictionaries(name) VALUES (?) RETURNING id",
                           ("Alice capture source",)).fetchone()["id"]
        conn.execute("INSERT INTO dictionary_entries(dictionary_id,key,content) VALUES (?,?,?)",
                     (did, "PRIVATE_TOKEN", "ALICE_PRIVATE_CONTENT"))
    try:
        yield SimpleNamespace(alice=alice, bob=bob, did=did, backend=pg_restricted_backend)
    finally:
        alice.close_all_connections()
        bob.close_all_connections()


def test_snapshot_refuses_foreign_owner_rows(catalog_rows):
    """The actual materializer snapshot must refuse rows outside the caller catalog."""
    c = catalog_rows
    with c.alice.transaction() as conn:
        own = ChatDictionaryService.snapshot_assigned_dictionaries(conn, [c.did])
        assert own["passes"][0]["entries"][0]["content"] == "ALICE_PRIVATE_CONTENT"  # nosec B101 - owner positive
    with c.bob.transaction() as conn:
        with pytest.raises(InputError):
            ChatDictionaryService.snapshot_assigned_dictionaries(conn, [c.did])


def test_catalog_refuses_foreign_owner_read(catalogs):
    """The actual unqualified by-ID service read must be owner isolated."""
    c = catalogs
    assert c.a.get_dictionary(c.did)["name"] == "Alice private catalog"  # nosec B101 - owner positive
    assert c.b.get_dictionary(c.did) is None  # nosec B101 - cross-owner isolation
    assert c.b.list_dictionaries(include_inactive=True) == []  # nosec B101 - no catalog enumeration


def test_postgres_legacy_unowned_rows_are_quarantined(pg_restricted_backend):
    """Upgrade never assigns a preexisting catalog to the first opener."""
    class LegacyChaCha(CharactersRAGDB):
        _POSTGRES_SCHEMA_VERSION = 82

    legacy = LegacyChaCha(":memory:", client_id="1", backend=pg_restricted_backend)
    with legacy.transaction() as conn:
        for statement in LEGACY_CATALOG_DDL:
            conn.execute(statement)
        did = conn.execute("INSERT INTO chat_dictionaries(name) VALUES (?) RETURNING id",
                           ("Unowned legacy",)).fetchone()["id"]
        conn.execute("INSERT INTO dictionary_entries(dictionary_id,key,content) VALUES (?,?,?)",
                     (did, "LEGACY_TOKEN", "UNOWNED_PRIVATE_CONTENT"))
    legacy.close_connection()
    first = CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
    second = CharactersRAGDB(":memory:", client_id="2", backend=pg_restricted_backend)
    try:
        for db in (first, second):
            with db.transaction() as conn:
                with pytest.raises(InputError):
                    ChatDictionaryService.snapshot_assigned_dictionaries(conn, [did])
            service = ChatDictionaryService(db)
            assert service.get_dictionary(did) is None  # nosec B101 - legacy is not caller owned
            with db.transaction() as conn:
                assert conn.execute("SELECT * FROM dictionary_entries WHERE dictionary_id = ?",
                                    (did,)).fetchall() == []  # nosec B101 - parent quarantine
        with first.transaction() as conn:
            # Schema-owner inspection of the test upgrade only: no row adoption.
            conn.execute("ALTER TABLE chat_dictionaries NO FORCE ROW LEVEL SECURITY")
            row = conn.execute("SELECT client_id,name FROM chat_dictionaries WHERE id = ?", (did,)).fetchone()
            conn.execute("ALTER TABLE chat_dictionaries FORCE ROW LEVEL SECURITY")
            assert row == {"client_id": None, "name": "Unowned legacy"}  # nosec B101 - retained unowned data
    finally:
        first.close_all_connections()
        second.close_all_connections()


@pytest.fixture
def native_catalog_chat(catalog_rows, single_user_cookie_client, monkeypatch):
    """Real native Character creation/capture against the restricted shared DB."""
    c = catalog_rows
    client, api_key = single_user_cookie_client
    assert _mint(client, api_key).status_code == 200  # nosec B101 - real authenticated session
    auth_headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
    monkeypatch.setattr(character_conversation_factory, "ensure_app_config", lambda: deepcopy(CONFIG))
    monkeypatch.setattr(chat_service, "load_comprehensive_config", lambda: deepcopy(CONFIG))
    monkeypatch.setattr(chat_service, "list_provider_models", lambda _provider: [MODEL])
    monkeypatch.setattr(gateway, "list_provider_models", chat_service.list_provider_models)
    old = client.app.dependency_overrides.get(get_chacha_db_for_user)
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: c.alice
    # Seed owned source data without claiming unrelated automatic-ID Character
    # creation coverage; that existing restricted-role bootstrap seam is open.
    char = 900001
    with c.alice.transaction() as conn:
        conn.execute("INSERT INTO character_cards(id,name,system_prompt,client_id) VALUES (?,?,?,?)",
                     (char, "OwnerGuide", "Keep the owner instruction.", "1"))
    cid = character_conversation_factory.create_character_conversation(
        c.alice, conversation_data={"character_id": char, "title": "Owner dictionary capture"},
        provider="ollama", model=MODEL,
    )
    prior = c.alice.add_message({"conversation_id": cid, "sender": "user", "content": "Prior input"})
    tail = c.alice.add_message({"conversation_id": cid, "sender": "assistant", "content": "Prior answer",
                                "parent_message_id": prior})
    try:
        yield SimpleNamespace(**vars(c), db=c.alice, client=client, headers=auth_headers, cid=cid, tail=tail)
    finally:
        if old is None:
            client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
        else:
            client.app.dependency_overrides[get_chacha_db_for_user] = old


def capture(chat):
    return chat.client.post(f"/api/v1/chat/conversations/{chat.cid}/history/selection",
                            headers=chat.headers, json={"purpose": "send", "view": {
                                "view_session_id": "owner-dictionary-view", "conversation_id": chat.cid,
                                "interpretation": {"kind": "parent_graph_v1"},
                                "cursor": {"kind": "after_message", "message_id": chat.tail},
                                "selection_revision": 1,
                            }})


def test_native_capture_refuses_foreign_legacy_assignment_before_binding(native_catalog_chat):
    """The send-only binder must not materialize another owner's dictionary."""
    chat = native_catalog_chat
    with chat.bob.transaction() as conn:
        foreign = conn.execute("INSERT INTO chat_dictionaries(name) VALUES (?) RETURNING id",
                               ("Bob private",)).fetchone()["id"]
        conn.execute("INSERT INTO dictionary_entries(dictionary_id,key,content) VALUES (?,?,?)",
                     (foreign, "PRIVATE_TOKEN", "BOB_SECRET"))
    with chat.alice.transaction() as conn:
        state = chat.alice.get_roleplay_resume_state(chat.cid, conn=conn, lock_for_update=True,
                                                   owner_client_id="1")
        assert state["materialized_settings"] is not None  # nosec B101 - usable native baseline
        assert chat.alice.upsert_conversation_settings(
            chat.cid, {**state["settings"], "chat_dictionary_ids": [foreign]}, conn=conn,
            expected_settings_version=state["settings_version"],
        )  # nosec B101 - exact legacy assignment CAS succeeded
    before = chat.alice.get_roleplay_resume_state(chat.cid, owner_client_id="1")
    assert before["settings"]["chat_dictionary_ids"] == [foreign]  # nosec B101 - fixture admission verified
    count = before["message_count"]
    response = capture(chat)
    assert response.status_code == 409, response.text  # nosec B101 - foreign assignment refusal
    after = chat.alice.get_roleplay_resume_state(chat.cid, owner_client_id="1")
    assert after["settings"] == before["settings"]  # nosec B101 - no legacy write
    assert after["settings_version"] == before["settings_version"]  # nosec B101 - no CAS mutation
    assert after["message_count"] == count  # nosec B101 - no append


def test_owner_names_and_service_writes_remain_independent(catalogs):
    c = catalogs
    other = c.b.create_dictionary("Alice private catalog")
    assert c.b.get_dictionary(other)["client_id"] == "2"  # nosec B101 - derived tenant
    assert c.a.get_dictionary(other) is None  # nosec B101 - reverse boundary
    assert c.a.update_entry(c.eid, content="ALICE_UPDATED")  # nosec B101 - normal owner mutation
    assert not c.b.update_entry(c.eid, content="BOB_OVERWRITE")  # nosec B101 - foreign update denied
    assert not c.b.delete_entry(c.eid)  # nosec B101 - foreign delete denied
    assert not c.b.update_dictionary(c.did, name="Stolen")  # nosec B101 - foreign definition update denied
    assert not c.b.delete_dictionary(c.did)  # nosec B101 - foreign definition delete denied
    assert c.a.process_text("PRIVATE_TOKEN", dictionary_id=c.did) == "ALICE_UPDATED"  # nosec B101 - owner processing
    with pytest.raises(AuthorizationDeniedError):
        c.b.process_text("PRIVATE_TOKEN", dictionary_id=c.did)
    assert c.b.get_entries(c.did) == []  # nosec B101 - child isolation
    assert c.b.get_entry(c.eid) is None  # nosec B101 - direct child isolation
    assert c.b.list_dictionary_versions(c.did) == ([], 0)  # nosec B101 - version isolation
    assert c.b.list_transform_activity(c.did) == ([], 0)  # nosec B101 - activity isolation
    with pytest.raises((DatabaseError, CharactersRAGDBError)):
        c.b.add_entry(c.did, "ILLEGAL", "FOREIGN_CHILD")
    with pytest.raises(CharactersRAGDBError, match="Included dictionaries not found"):
        c.b.create_dictionary("Illegal include", included_dictionary_ids=[c.did])
    assert c.a.get_entry(c.eid)["replacement"] == "ALICE_UPDATED"  # nosec B101 - failed writes retained owner data


@pytest.mark.parametrize("tenant", ["2", ""])
def test_all_catalog_tables_enforce_foreign_and_unbound_rls(catalogs, tenant):
    c = catalogs
    assert c.a.process_text("PRIVATE_TOKEN", dictionary_id=c.did) == "ALICE_PRIVATE_CONTENT"  # nosec B101 - activity positive
    with c.alice.transaction() as conn:
        for table in DICTIONARY_TABLES:
            assert conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"] > 0  # nosec B101 B608 - fixed table inventory
        conn.execute("SELECT set_config('app.current_user_id', ?, true)", (tenant,))
        for table in DICTIONARY_TABLES:
            assert conn.execute(f"SELECT * FROM {table}").fetchall() == []  # nosec B101 B608 - fixed table inventory
            assert conn.execute(f"DELETE FROM {table}").rowcount == 0  # nosec B101 B608 - fixed table inventory
        assert conn.execute("UPDATE chat_dictionaries SET name='Foreign overwrite'").rowcount == 0  # nosec B101 - USING fence
        assert conn.execute("UPDATE dictionary_entries SET content='Foreign overwrite'").rowcount == 0  # nosec B101 - child USING fence
    with pytest.raises(DatabaseError):
        with c.alice.transaction() as conn:
            conn.execute("SELECT set_config('app.current_user_id', ?, true)", (tenant,))
            conn.execute("INSERT INTO chat_dictionaries(name,client_id) VALUES (?,?)", ("Forged owner", "1"))
    with pytest.raises(DatabaseError):
        with c.alice.transaction() as conn:
            conn.execute("SELECT set_config('app.current_user_id', ?, true)", (tenant,))
            conn.execute("INSERT INTO dictionary_entries(dictionary_id,key,content) VALUES (?,?,?)",
                         (c.did, "ILLEGAL", "UNBOUND_CHILD"))


@pytest.mark.parametrize("tenant", ["2", ""])
def test_snapshot_requires_construction_owner_current_session(catalog_rows, tenant):
    c = catalog_rows
    with c.alice.transaction() as conn:
        conn.execute("SELECT set_config('app.current_user_id', ?, true)", (tenant,))
        with pytest.raises(InputError, match="current catalog owner"):
            ChatDictionaryService.snapshot_assigned_dictionaries(conn, [c.did])


def assign_native(chat, ids):
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=global",
                               headers=chat.headers, json={"settings": {"chat_dictionary_ids": ids}})
    assert response.status_code == 200, response.text  # nosec B101 - actual owner writer
    return chat.alice.get_roleplay_resume_state(chat.cid, owner_client_id="1")


@pytest.mark.parametrize("marker", [None, "2"])
def test_unproven_pg_materialized_plan_refuses_without_silent_refresh(native_catalog_chat, marker):
    chat = native_catalog_chat
    state = assign_native(chat, [chat.did])
    settings = deepcopy(state["settings"])
    values = deepcopy(settings["roleplayBehaviorV1"]["values"])
    if marker is None:
        values["dictionaries"].pop("catalog_owner")
    else:
        values["dictionaries"]["catalog_owner"] = marker
    settings["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    with chat.alice.transaction() as conn:
        assert chat.alice.upsert_conversation_settings(
            chat.cid, settings, conn=conn, expected_settings_version=state["settings_version"],
        )  # nosec B101 - exact historical/corrupt persisted envelope fixture
    before = chat.alice.get_roleplay_resume_state(chat.cid, owner_client_id="1")
    response = capture(chat)
    assert response.status_code == 409, response.text  # nosec B101 - no implicit owner adoption
    after = chat.alice.get_roleplay_resume_state(chat.cid, owner_client_id="1")
    assert after["settings"] == before["settings"]  # nosec B101 - no implicit catalog refresh
    assert after["settings_version"] == before["settings_version"]  # nosec B101 - no legacy write
    assert after["message_count"] == before["message_count"]  # nosec B101 - no append


def test_native_owned_frozen_plan_preserves_context_and_authored_input(native_catalog_chat):
    from functools import partial

    from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
    from tldw_Server_API.tests.Chat_NEW.integration.test_native_assigned_dictionary import prepare

    chat = native_catalog_chat
    state = assign_native(chat, [chat.did])
    plan = state["materialized_settings"]["values"]["dictionaries"]
    assert plan["catalog_owner"] == "1"  # nosec B101 - trusted protected owner marker
    assert ChatDictionaryService(chat.alice).update_entry(
        ChatDictionaryService(chat.alice).get_entries(chat.did)[0]["id"], content="LATER_CATALOG",
    )  # nosec B101 - actual catalog edit
    captured = capture(chat)
    assert captured.status_code == 200, captured.text  # nosec B101 - unchanged owned plan still usable
    body = captured.json()
    selection = resolve_history_selection(body["snapshot"], body["view"], "send", "pg-dictionary-input")["selection"]
    _, system, messages, runtime = chat.client.portal.call(partial(prepare, chat, selection, text="PRIVATE_TOKEN"))
    assert "Keep the owner instruction." in system  # nosec B101 - saved Character instructions
    assert messages[-1]["content"] == [{"type": "text", "text": "ALICE_PRIVATE_CONTENT"}]  # nosec B101 - frozen provider copy
    assert chat.alice.get_message_by_id(runtime["user_message_id"])["content"] == "PRIVATE_TOKEN"  # nosec B101 - authored input
    refreshed = assign_native(chat, [chat.did])
    assert refreshed["materialized_settings"]["values"]["dictionaries"]["passes"][0]["entries"][0]["content"] == "LATER_CATALOG"  # nosec B101 - explicit reassignment


def test_public_settings_cannot_supply_catalog_owner_authority(native_catalog_chat):
    chat = native_catalog_chat
    state = assign_native(chat, [chat.did])
    values = deepcopy(state["materialized_settings"]["values"])
    values["dictionaries"]["catalog_owner"] = "2"
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=global",
                               headers=chat.headers, json={"settings": {
                                   "roleplayBehaviorV1": build_materialized_behavior_settings(values)}})
    assert response.status_code == 422, response.text  # nosec B101 - protected field is server owned
    after = chat.alice.get_roleplay_resume_state(chat.cid, owner_client_id="1")
    assert after["settings"] == state["settings"]  # nosec B101 - no forged plan write


@pytest.mark.parametrize("include", ["foreign", "missing"])
def test_native_capture_refuses_unowned_include_path(catalog_rows, include):
    c = catalog_rows
    with c.bob.transaction() as conn:
        foreign = conn.execute("INSERT INTO chat_dictionaries(name) VALUES (?) RETURNING id",
                               ("Foreign include",)).fetchone()["id"]
    with c.alice.transaction() as conn:
        conn.execute("UPDATE chat_dictionaries SET included_dictionary_ids=? WHERE id=?",
                     (json.dumps([foreign if include == "foreign" else 987654321]), c.did))
        with pytest.raises(InputError, match="unavailable"):
            ChatDictionaryService.snapshot_assigned_dictionaries(conn, [c.did])


@pytest.mark.parametrize("drift", [
    "ALTER TABLE dictionary_entries NO FORCE ROW LEVEL SECURITY",
    "CREATE POLICY dictionary_extra ON dictionary_entries USING (true) WITH CHECK (true)",
    "ALTER TABLE chat_dictionaries ALTER COLUMN client_id DROP DEFAULT",
    "DROP INDEX uq_chat_dictionaries_owner_name",
])
def test_runtime_and_service_init_refuse_dictionary_authority_drift(catalogs, drift):
    c = catalogs
    with c.alice.transaction() as conn:
        conn.execute(drift)
    with pytest.raises(CharactersRAGDBError):
        CharactersRAGDB(":memory:", client_id="1", backend=c.backend)
    with pytest.raises(CharactersRAGDBError):
        ChatDictionaryService(c.alice)


@pytest.mark.parametrize("layout", ["legacy86", "partial86", "unsafe86"])
def test_legacy_dictionary_v86_composes_with_semantic_migrations(pg_restricted_backend, monkeypatch, layout):
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
    from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import build_dictionary_rls_sql
    from tldw_Server_API.app.core.DB_Management.chacha.dictionary_schema import postgres_dictionary_schema_statements

    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 85)
        old = CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
    try:
        with old.transaction() as conn:
            for statement in (*postgres_dictionary_schema_statements(), *build_dictionary_rls_sql()):
                conn.execute(statement)
        service = ChatDictionaryService(old)
        dictionary_id = service.create_dictionary("Retained legacy v86")
        with old.transaction() as conn:
            conn.execute("UPDATE db_schema_version SET version=86 WHERE schema_name=?", (old._SCHEMA_NAME,))
            if layout == "partial86":
                conn.execute("CREATE TABLE note_semantic_operation_receipts (id integer)")
            elif layout == "unsafe86":
                conn.execute("ALTER TABLE chat_dictionaries NO FORCE ROW LEVEL SECURITY")
    finally:
        old.close_all_connections()
    backend = DatabaseBackendFactory.create_backend(pg_restricted_backend.config)
    try:
        if layout != "legacy86":
            with pytest.raises(CharactersRAGDBError):
                CharactersRAGDB(":memory:", client_id="1", backend=backend)
            assert int(backend.execute("SELECT version FROM db_schema_version WHERE schema_name=?", (old._SCHEMA_NAME,)).scalar) == 86  # nosec B101 - Refusal cannot stamp an upgrade.
            assert not backend.table_exists("note_semantic_index_configs")  # nosec B101 - Refuse partial or unsafe catalogs without creating semantic data.
            return
        upgraded = CharactersRAGDB(":memory:", client_id="1", backend=backend)
        try:
            assert int(backend.execute("SELECT version FROM db_schema_version WHERE schema_name=?", (old._SCHEMA_NAME,)).scalar) == 89  # nosec B101 - Both published semantic and dictionary migrations must finish.
            assert ChatDictionaryService(upgraded).get_dictionary(dictionary_id)["name"] == "Retained legacy v86"  # nosec B101 - Preserve the original owned row.
            assert backend.table_exists("note_semantic_index_configs") and backend.table_exists("note_semantic_operation_receipts")  # nosec B101 - Reach both ends of semantic86-88.
        finally:
            upgraded.close_all_connections()
    finally:
        backend.get_pool().close_all()
