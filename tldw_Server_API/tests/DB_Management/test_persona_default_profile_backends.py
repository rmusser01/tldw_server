"""UAT207: shared PostgreSQL default IDs and local profile insert ownership."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ExternalConnection, chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.Persona.session_manager import SessionManager
from tldw_Server_API.app.core.Persona.session_materialization import materialize_persona_session

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgresql"])
def owners(request, tmp_path, monkeypatch):
    backend = None
    if request.param == "postgresql":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    dbs = [CharactersRAGDB(tmp_path / f"{owner}.db", client_id=str(owner), backend=backend) for owner in (2, 3)]
    monkeypatch.setattr(endpoint, "is_persona_enabled", lambda: True)
    try:
        yield SimpleNamespace(kind=request.param, alice=dbs[0], bob=dbs[1], backend=backend)
    finally:
        for db in dbs:
            db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.fixture
def pg_db(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "default-profile.db", client_id="2", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@contextmanager
def client_for(db, owner):
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/persona")

    async def user_override():
        return User(id=owner, username=f"fixture-{owner}", email=None, is_active=True)

    def db_override():
        return db

    app.dependency_overrides[get_request_user] = user_override
    app.dependency_overrides[get_chacha_db_for_user] = db_override
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize("route", ["profiles", "catalog"])
def test_cold_two_owner_routes_repeat_with_owned_profiles_and_policies(owners, route):
    ids = []
    for owner, db in [(2, owners.alice), (3, owners.bob)]:
        with client_for(db, owner) as client:
            first = client.get(f"/api/v1/persona/{route}")
            again = client.get(f"/api/v1/persona/{route}")
            assert [first.status_code, again.status_code] == [200, 200]
            persona_id = first.json()[0]["id"]
            assert again.json()[0]["id"] == persona_id
            detail = client.get(f"/api/v1/persona/profiles/{persona_id}")
            assert detail.status_code == 200
            assert detail.json()["id"] == persona_id
            assert db.get_persona_profile(persona_id, user_id=str(owner))["user_id"] == str(owner)
        ids.append(persona_id)
        rules = db.list_persona_policy_rules(persona_id=persona_id, user_id=str(owner))
        assert len(rules) == len(endpoint._DEFAULT_PERSONA_POLICY_RULES)
        assert all(row["persona_id"] == persona_id and row["user_id"] == str(owner) for row in rules)
        db.get_connection().rollback()
    assert ids == (["research_assistant", "research_assistant"] if owners.kind == "sqlite" else ["research_assistant:2", "research_assistant:3"])


def test_owned_legacy_default_keeps_id_data_and_custom_rules(owners):
    db = owners.alice
    db.create_persona_profile({"id": "research_assistant", "user_id": "2", "name": "My legacy assistant", "system_prompt": "Preserve this", "version": 4})
    db.replace_persona_policy_rules(persona_id="research_assistant", user_id="2", rules=[{"rule_kind": "mcp_tool", "rule_name": "custom", "allowed": False}])
    before = db.get_persona_profile("research_assistant", user_id="2")
    rules = db.list_persona_policy_rules(persona_id="research_assistant", user_id="2")
    db.get_connection().rollback()
    assert endpoint._ensure_default_persona_profile(db, user_id="2") == before
    assert db.list_persona_policy_rules(persona_id="research_assistant", user_id="2") == rules
    db.get_connection().rollback()


def test_legacy_default_tombstone_is_not_bypassed_or_resurrected(owners):
    db = owners.alice
    db.create_persona_profile({"id": "research_assistant", "user_id": "2", "name": "Research Assistant", "deleted": True, "is_active": False})
    with pytest.raises(ConflictError):
        endpoint._ensure_default_persona_profile(db, user_id="2")
    rows = db.list_persona_profiles(user_id="2", include_deleted=True)
    assert [(row["id"], row["deleted"]) for row in rows] == [("research_assistant", True)]


def test_existing_owned_default_name_falls_back_without_overwriting(owners):
    db = owners.alice
    db.create_persona_profile({"id": "custom-default", "user_id": "2", "name": "Research Assistant", "system_prompt": "Custom"})
    result = endpoint._ensure_default_persona_profile(db, user_id="2")
    assert (result["id"], result["system_prompt"]) == ("custom-default", "Custom")
    assert len(db.list_persona_profiles(user_id="2")) == 1


def test_foreign_legacy_default_is_not_reused(pg_db):
    pg_db.create_persona_profile({"id": "research_assistant", "user_id": "3", "name": "Research Assistant", "system_prompt": "Foreign"})
    result = endpoint._ensure_default_persona_profile(pg_db, user_id="2")
    assert (result["id"], result["user_id"]) == ("research_assistant:2", "2")
    assert pg_db.get_persona_profile("research_assistant", user_id="3")["system_prompt"] == "Foreign"


def test_foreign_scoped_id_never_becomes_an_owned_default(pg_db):
    pg_db.create_persona_profile({"id": "research_assistant:2", "user_id": "3", "name": "Foreign"})
    with pytest.raises(ConflictError):
        endpoint._ensure_default_persona_profile(pg_db, user_id="2")
    assert pg_db.list_persona_profiles(user_id="2") == []
    assert pg_db.get_persona_profile("research_assistant:2", user_id="3")["name"] == "Foreign"


def test_scoped_default_tombstone_is_not_resurrected(pg_db):
    pg_db.create_persona_profile({"id": "research_assistant:2", "user_id": "2", "name": "Research Assistant", "deleted": True, "is_active": False})
    with pytest.raises(ConflictError):
        endpoint._ensure_default_persona_profile(pg_db, user_id="2")
    assert pg_db.get_persona_profile("research_assistant:2", user_id="2", include_deleted=True)["deleted"] is True


def test_same_owner_raced_default_creators_converge(pg_db, monkeypatch):
    barrier = Barrier(2)
    original = pg_db.create_persona_profile

    def raced_create(data):
        barrier.wait(timeout=10)
        return original(data)

    monkeypatch.setattr(pg_db, "create_persona_profile", raced_create)

    def ensure():
        try:
            return endpoint._ensure_default_persona_profile(pg_db, user_id="2")
        finally:
            pg_db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(ensure) for _ in range(2)]
        rows = [future.result(timeout=20) for future in futures]
    assert {row["id"] for row in rows} == {"research_assistant:2"}
    assert len(pg_db.list_persona_profiles(user_id="2")) == 1


@pytest.mark.parametrize("duplicate", ["id", "name"])
def test_duplicate_profile_is_conflict_and_connection_remains_usable(pg_db, duplicate):
    pg_db.create_persona_profile({"id": "existing", "user_id": "2", "name": "Existing"})
    data = {"id": "existing" if duplicate == "id" else "new", "user_id": "2", "name": "Existing" if duplicate == "name" else "New"}
    with pytest.raises(ConflictError):
        pg_db.create_persona_profile(data)
    assert pg_db.get_connection().info.transaction_status.name == "IDLE"
    assert pg_db.get_persona_profile("existing", user_id="2")["name"] == "Existing"
    assert pg_db.create_persona_profile({"id": "retry", "user_id": "2", "name": "Retry"}) == "retry"


@pytest.mark.parametrize("mode", ["implicit", "raw", "managed-empty", "backend-empty", "borrowed-empty"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_profile_insert_preserves_caller_transaction_and_rollback(pg_db, mode, duplicate):
    db = pg_db
    db.create_persona_profile({"id": "existing", "user_id": "2", "name": "Existing"})
    conn = db.get_connection()

    def operation():
        if mode not in {"managed-empty", "backend-empty", "borrowed-empty"}:
            db.execute_query("UPDATE persona_profiles SET system_prompt = ? WHERE id = ?", ("Pending", "existing"))
        if duplicate:
            with pytest.raises(ConflictError):
                db.create_persona_profile({"id": "existing", "user_id": "2", "name": "Duplicate"})
        else:
            db.create_persona_profile({"id": "created", "user_id": "2", "name": "Created"})
        assert conn.info.transaction_status.name == "INTRANS"
        if mode not in {"managed-empty", "backend-empty", "borrowed-empty"}:
            assert db.get_persona_profile("existing", user_id="2")["system_prompt"] == "Pending"
        raise RuntimeError("Caller rolls back")

    try:
        with pytest.raises(RuntimeError, match="Caller rolls back"):
            if mode == "managed-empty":
                with db.transaction():
                    operation()
            elif mode == "backend-empty":
                with conn._backend.transaction(connection=conn._connection):
                    operation()
            elif mode == "borrowed-empty":
                with chacha_operation(independent=True, bindings=(ExternalConnection(db, conn._connection, conn._backend),)):
                    operation()
            else:
                if mode == "raw":
                    conn.execute("BEGIN")
                operation()
    finally:
        conn.rollback()
    assert db.get_persona_profile("existing", user_id="2")["system_prompt"] is None
    assert db.get_persona_profile("created", user_id="2") is None


def test_unexpected_profile_insert_failure_is_rolled_back_and_propagated(pg_db):
    with pytest.raises(CharactersRAGDBError):
        pg_db.create_persona_profile({"id": "bad-time", "user_id": "2", "name": "Bad time", "created_at": "not-a-timestamp"})
    assert pg_db.get_connection().info.transaction_status.name == "IDLE"
    assert pg_db.list_persona_profiles(user_id="2") == []


def test_scoped_profile_id_propagates_to_actual_session_policy(pg_db):
    profile = endpoint._ensure_default_persona_profile(pg_db, user_id="2")
    assert profile["id"] == "research_assistant:2"
    pg_db.create_persona_session({"id": "scoped-session", "persona_id": profile["id"], "user_id": "2", "mode": "session_scoped", "status": "active"})
    result = endpoint._load_persona_policy_rules_for_session(pg_db, session_id="scoped-session", user_id="2")
    assert result["session_exists"] is True
    assert result["persona_id"] == profile["id"]
    assert len(result["policy_rules"]) == len(endpoint._DEFAULT_PERSONA_POLICY_RULES)


def test_cold_session_fallback_uses_an_owned_default_on_each_backend(owners):
    owners.alice.create_persona_profile({"id": "research_assistant", "user_id": "2", "name": "Research Assistant"})
    result = materialize_persona_session(owners.bob, session_manager=SessionManager(), user_id="3", persona_id="research_assistant")
    expected = "research_assistant:3" if owners.kind == "postgresql" else "research_assistant"
    assert result.persona_id == expected
    assert result.profile["user_id"] == "3"
    persisted = owners.bob.get_persona_session(result.session_id, user_id="3")
    assert persisted["persona_id"] == expected
    assert owners.bob.get_persona_session(result.session_id, user_id="2") is None


@pytest.mark.parametrize("include_deleted,active_only", [(False, False), (True, False), (False, True)])
def test_profile_list_owner_flags_and_pagination_stay_unchanged(owners, include_deleted, active_only):
    db = owners.alice
    for index, (name, active, deleted, owner) in enumerate([
        ("Active", True, False, "2"),
        ("Inactive", False, False, "2"),
        ("Deleted", True, True, "2"),
        ("Foreign", True, False, "3"),
    ]):
        db.create_persona_profile({"id": name, "user_id": owner, "name": name, "is_active": active, "deleted": deleted, "last_modified": f"2026-09-01T00:00:0{index}Z"})
    rows = db.list_persona_profiles(user_id="2", include_deleted=include_deleted, active_only=active_only)
    expected = (["Deleted"] if include_deleted else []) + ([] if active_only else ["Inactive"]) + ["Active"]
    assert [row["id"] for row in rows] == expected
    page = db.list_persona_profiles(user_id="2", include_deleted=include_deleted, active_only=active_only, limit=1, offset=1)
    assert [row["id"] for row in page] == expected[1:2]
    assert db.get_persona_profile("Foreign", user_id="2") is None
    assert set(db.get_persona_profiles_by_ids(user_id="2", persona_ids=["Active", "Foreign"])) == {"Active"}
