"""Real shared ChaCha serials preserve allocated IDs hidden by tenant RLS."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import _mint
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture(autouse=True)
def private_test_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(tmp_path / "monitoring.db"))
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(tmp_path / "system.log"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))


@pytest.fixture
def sequence_db(pg_restricted_backend):
    """Use the official schema owner that cannot bypass forced row policies."""
    db = CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
    try:
        yield db
    finally:
        db.close_connection()


def maintain(db, conn):
    """The registered migrations pass the same caller-owned raw connection."""
    db._sync_postgres_sequences(conn._connection)


def allocate(conn):
    return conn.execute("SELECT nextval('character_cards_id_seq') AS id").fetchone()["id"]


def test_public_character_create_uses_id_above_hidden_seed_and_survives_reopen(
    sequence_db, single_user_cookie_client, pg_database_config, monkeypatch,
):
    db = sequence_db
    client, api_key = single_user_cookie_client
    assert _mint(client, api_key).status_code == 200  # nosec B101 - actual cookie authentication
    headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
    with db.transaction() as conn:
        flags = conn.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
        assert flags == {"rolsuper": False, "rolbypassrls": False}  # nosec B101 - non-bypass role
        policy = conn.execute("SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE relname='character_cards'").fetchone()
        assert policy == {"relrowsecurity": True, "relforcerowsecurity": True}  # nosec B101 - forced owner policy
        assert conn.execute("SELECT COUNT(*) AS n FROM character_cards WHERE id=1").fetchone()["n"] == 0  # nosec B101 - published system_init seed is tenant-hidden
        state = conn.execute("SELECT last_value,is_called FROM character_cards_id_seq").fetchone()
        assert state["last_value"] >= 1  # nosec B101 - inspect actual bootstrap sequence, no resets
    # The official fixture's separate admin reads seed presence only. Creation
    # below still uses the checked non-bypass role; no policy/sequence is altered.
    admin = DatabaseBackendFactory.create_backend(pg_database_config)
    try:
        seed = admin.execute("SELECT id,name,client_id FROM character_cards WHERE id=1").rows
        assert seed == [{"id": 1, "name": "Default Assistant", "client_id": "system_init"}]  # nosec B101 - actual retained bootstrap source
    finally:
        admin.get_pool().close_all()
    monkeypatch.setitem(client.app.dependency_overrides, get_chacha_db_for_user, lambda: db)
    response = client.post("/api/v1/characters/", headers=headers,
                           json={"name": "Public sequence recovery", "system_prompt": "Keep this instruction."})
    assert response.status_code == 201, (state, response.text)  # nosec B101 - actual public creation
    first = response.json()["id"]
    assert first > 1  # nosec B101 - hidden seed cannot be reused
    assert db.get_character_card_by_id(first)["system_prompt"] == "Keep this instruction."  # nosec B101 - ordinary saved projection
    db.close_connection()
    other = CharactersRAGDB(":memory:", client_id="2", backend=db.backend)
    try:
        second = other.add_character_card({"name": "Public sequence recovery"})
        assert second > first  # nosec B101 - new owner and reopen preserve allocations
        assert other.get_character_card_by_id(first) is None  # nosec B101 - owner fences retained
    finally:
        other.close_connection()


def test_rolled_back_allocation_is_never_reused_for_an_empty_tenant(sequence_db):
    db = sequence_db
    with pytest.raises(RuntimeError, match="caller abort"):
        with db.transaction() as conn:
            allocated = allocate(conn)
            raise RuntimeError("caller abort")
    with db.transaction() as conn:
        maintain(db, conn)
        assert allocate(conn) > allocated  # nosec B101 - rollback does not free an allocated ID


def test_visible_explicit_import_advances_without_committing_caller_rows(sequence_db):
    db = sequence_db
    with pytest.raises(RuntimeError, match="caller import abort"):
        with db.transaction() as conn:
            conn.execute("INSERT INTO character_cards(id,name,client_id) VALUES (701,'Explicit import','1')")
            maintain(db, conn)
            allocated = allocate(conn)
            assert allocated == 702  # nosec B101 - visible explicit high ID advances
            raise RuntimeError("caller import abort")
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM character_cards WHERE id=701").fetchone()["n"] == 0  # nosec B101 - caller rows rolled back
        maintain(db, conn)
        assert allocate(conn) > allocated  # nosec B101 - nontransactional allocation retained


def test_concurrent_allocation_cannot_be_rewound_after_visible_max_read(sequence_db, monkeypatch):
    """Observe only the real SQL boundary; no DB result or guard is replaced."""
    db = sequence_db
    with db.transaction() as conn:
        conn.execute("INSERT INTO character_cards(id,name,client_id) VALUES (7,'Pending import','1')")
    at_write, release, started, completed = Event(), Event(), Event(), Event()
    execute = db.backend.execute

    def hold_character_write(query, params=None, connection=None, **kwargs):
        if (current_thread().name.startswith("maintenance") and "setval" in query
                and params and str(params[0]).replace('"', '').endswith("character_cards_id_seq")):
            at_write.set()
            assert release.wait(10)  # nosec B101 - bounded actual interleaving
        return execute(query, params, connection=connection, **kwargs)

    monkeypatch.setattr(db.backend, "execute", hold_character_write)

    def maintain_actual():
        with db.transaction() as conn:
            maintain(db, conn)

    def allocate_actual():
        with db.transaction() as conn:
            started.set()
            values = [allocate(conn) for _ in range(21)]
            completed.set()
            return values

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="maintenance") as maintenance:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="allocator") as allocator:
            repair = maintenance.submit(maintain_actual)
            try:
                assert at_write.wait(5)  # nosec B101 - maintenance reached actual final write
                allocation = allocator.submit(allocate_actual)
                assert started.wait(5)  # nosec B101 - allocator reached real request
                completed.wait(0.5)
            finally:
                release.set()
            repair.result(timeout=10)
            values = allocation.result(timeout=10)
    with db.transaction() as conn:
        assert allocate(conn) > max(values)  # nosec B101 - no allocated range rewind
