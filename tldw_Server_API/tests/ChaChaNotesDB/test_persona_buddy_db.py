import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

import pytest

import json

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.Persona.buddy import ensure_persona_buddy_for_profile


pytestmark = pytest.mark.unit


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "persona_buddy_test.sqlite"


@pytest.fixture
def db_instance(db_path: Path) -> Iterator[CharactersRAGDB]:
    db = CharactersRAGDB(db_path, "persona-buddy-test-client")
    yield db
    db.close_connection()


def test_migration_v39_to_latest_creates_persona_buddies_table(db_path: Path) -> None:
    seeded = CharactersRAGDB(db_path, "seed-client")
    seeded.close_connection()

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (39, CharactersRAGDB._SCHEMA_NAME),
        )
        conn.execute("DROP TABLE IF EXISTS persona_buddies")
        conn.commit()

    migrated = CharactersRAGDB(db_path, "migration-check-client")
    raw_conn = migrated.get_connection()
    tables = {
        row["name"]
        for row in raw_conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    buddy_columns = {
        row["name"] for row in raw_conn.execute("PRAGMA table_info('persona_buddies')").fetchall()
    }
    buddy_indexes = {
        row["name"] for row in raw_conn.execute("PRAGMA index_list('persona_buddies')").fetchall()
    }
    schema_version = raw_conn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()["version"]
    foreign_keys = raw_conn.execute("PRAGMA foreign_key_list('persona_buddies')").fetchall()

    fk_targets = {(row["table"], row["from"], row["to"]) for row in foreign_keys}

    assert schema_version == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert "persona_buddies" in tables
    assert "source_fingerprint" in buddy_columns
    assert "derivation_version" in buddy_columns
    assert "persona_id" in buddy_columns
    assert "user_id" in buddy_columns
    assert "derived_core_json" in buddy_columns
    assert "overlay_preferences_json" in buddy_columns
    assert "created_at" in buddy_columns
    assert "last_modified" in buddy_columns
    assert "version" in buddy_columns
    assert "idx_persona_buddies_user" in buddy_indexes
    assert ("persona_profiles", "persona_id", "id") in fk_targets
    migrated.close_connection()


def test_ensure_persona_buddy_persists_row_without_incrementing_persona_profile_version(
    db_instance: CharactersRAGDB,
) -> None:
    persona_id = db_instance.create_persona_profile({"user_id": "user-1", "name": "Buddy Persona"})
    before = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert before is not None

    buddy = ensure_persona_buddy_for_profile(db_instance, before)
    persisted = db_instance.get_persona_buddy(persona_id=persona_id, user_id="user-1")
    after = db_instance.get_persona_profile(persona_id, user_id="user-1")

    assert persisted is not None
    assert after is not None
    assert buddy["persona_id"] == persona_id
    assert persisted["persona_id"] == persona_id
    assert int(after["version"]) == int(before["version"])


def test_ensure_persona_buddy_rederives_when_source_fingerprint_changes(
    db_instance: CharactersRAGDB,
) -> None:
    persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Original Buddy Persona"}
    )
    profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert profile is not None
    original = ensure_persona_buddy_for_profile(db_instance, profile)

    assert db_instance.update_persona_profile(
        persona_id=persona_id,
        user_id="user-1",
        update_data={"name": "Renamed Buddy Persona"},
        expected_version=int(profile["version"]),
    )
    updated_profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert updated_profile is not None
    repaired = ensure_persona_buddy_for_profile(db_instance, updated_profile)

    assert repaired["source_fingerprint"] != original["source_fingerprint"]
    assert int(repaired["version"]) > int(original["version"])


def test_ensure_persona_buddy_is_idempotent_without_source_changes(
    db_instance: CharactersRAGDB,
) -> None:
    persona_id = db_instance.create_persona_profile({"user_id": "user-1", "name": "Stable Buddy Persona"})
    profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert profile is not None

    first = ensure_persona_buddy_for_profile(db_instance, profile)
    persisted_before = db_instance.get_persona_buddy(persona_id=persona_id, user_id="user-1")
    second = ensure_persona_buddy_for_profile(db_instance, profile)
    persisted_after = db_instance.get_persona_buddy(persona_id=persona_id, user_id="user-1")

    assert persisted_before is not None
    assert persisted_after is not None
    assert int(first["version"]) == int(second["version"])
    assert int(persisted_before["version"]) == int(persisted_after["version"])
    assert persisted_before["last_modified"] == persisted_after["last_modified"]


def test_upsert_persona_buddy_is_noop_when_payload_is_unchanged(
    db_instance: CharactersRAGDB,
) -> None:
    persona_id = db_instance.create_persona_profile({"user_id": "user-1", "name": "Noop Upsert Persona"})
    profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert profile is not None
    original = ensure_persona_buddy_for_profile(db_instance, profile)

    repeated = db_instance.upsert_persona_buddy(
        persona_id=persona_id,
        user_id="user-1",
        derivation_version=int(original["derivation_version"]),
        source_fingerprint=str(original["source_fingerprint"]),
        derived_core=original["derived_core"],
        overlay_preferences=original["overlay_preferences"],
    )
    persisted = db_instance.get_persona_buddy(persona_id=persona_id, user_id="user-1")

    assert persisted is not None
    assert int(repeated["version"]) == int(original["version"])
    assert int(persisted["version"]) == int(original["version"])
    assert repeated["last_modified"] == original["last_modified"]
    assert persisted["last_modified"] == original["last_modified"]


def test_ensure_persona_buddy_preserves_overlay_preferences_on_rederive(
    db_instance: CharactersRAGDB,
) -> None:
    persona_id = db_instance.create_persona_profile({"user_id": "user-1", "name": "Overlay Persona"})
    profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert profile is not None
    original = ensure_persona_buddy_for_profile(db_instance, profile)
    overlay_preferences = {"accessory_id": "scarf", "eye_style": "sleepy"}
    db_instance.upsert_persona_buddy(
        persona_id=persona_id,
        user_id="user-1",
        derivation_version=int(original["derivation_version"]),
        source_fingerprint=str(original["source_fingerprint"]),
        derived_core=original["derived_core"],
        overlay_preferences=overlay_preferences,
    )

    assert db_instance.update_persona_profile(
        persona_id=persona_id,
        user_id="user-1",
        update_data={"name": "Overlay Persona Renamed"},
        expected_version=int(profile["version"]),
    )
    updated_profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert updated_profile is not None
    repaired = ensure_persona_buddy_for_profile(db_instance, updated_profile)

    assert repaired["source_fingerprint"] != original["source_fingerprint"]
    assert repaired["overlay_preferences"] == overlay_preferences


def test_get_persona_buddy_surfaces_resolver_failures(
    db_instance: CharactersRAGDB,
    monkeypatch,
) -> None:
    persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Broken Buddy Persona"}
    )
    profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert profile is not None
    ensure_persona_buddy_for_profile(db_instance, profile)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.chacha.persona_state_store.resolve_persona_buddy_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid resolved profile")),
    )

    with pytest.raises(CharactersRAGDBError, match="Unable to resolve persona buddy profile"):
        db_instance.get_persona_buddy(persona_id=persona_id, user_id="user-1")


def test_get_persona_buddy_uses_backend_aware_deleted_filter_for_postgres(
    db_instance: CharactersRAGDB,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Cursor:
        def fetchone(self):
            return None

    def _fake_execute_query(query: str, params: tuple[object, ...]):
        captured["query"] = query
        captured["params"] = params
        return _Cursor()

    monkeypatch.setattr(type(db_instance), "backend_type", BackendType.POSTGRESQL)
    monkeypatch.setattr(db_instance, "execute_query", _fake_execute_query)

    assert (
        db_instance.get_persona_buddy(
            persona_id="persona-1",
            user_id="user-1",
            include_deleted_personas=False,
        )
        is None
    )

    assert "pp.deleted = ?" in str(captured["query"])
    assert captured["params"] == ("persona-1", "user-1", False, False)


def test_soft_delete_hides_buddy_and_restore_reuses_same_row(
    db_instance: CharactersRAGDB,
) -> None:
    persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Delete Restore Buddy Persona"}
    )
    profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert profile is not None
    buddy_before_delete = ensure_persona_buddy_for_profile(db_instance, profile)
    deleted_version = int(profile["version"])

    assert db_instance.soft_delete_persona_profile(
        persona_id=persona_id,
        user_id="user-1",
        expected_version=deleted_version,
    )
    assert db_instance.get_persona_buddy(persona_id=persona_id, user_id="user-1") is None
    deleted_profile = db_instance.get_persona_profile(
        persona_id,
        user_id="user-1",
        include_deleted=True,
    )
    assert deleted_profile is not None

    restored = db_instance.restore_persona_profile(
        persona_id=persona_id,
        user_id="user-1",
        expected_version=int(deleted_profile["version"]),
    )
    restored_profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    restored_buddy = db_instance.get_persona_buddy(persona_id=persona_id, user_id="user-1")

    assert restored is True
    assert restored_profile is not None
    assert restored_profile["is_active"] is True
    assert restored_buddy is not None
    assert int(restored_buddy["version"]) == int(buddy_before_delete["version"])
    assert restored_buddy["resolved_profile"] == buddy_before_delete["resolved_profile"]


def test_restore_persona_profile_uses_backend_aware_deleted_filter_for_postgres(
    db_instance: CharactersRAGDB,
    monkeypatch,
) -> None:
    executed: list[tuple[str, tuple[object, ...]]] = []

    class _Cursor:
        def __init__(self, row: dict[str, object] | None = None, rowcount: int = 0):
            self._row = row
            self.rowcount = rowcount

        def fetchone(self):
            return self._row

    class _Conn:
        def execute(self, query: str, params: tuple[object, ...]):
            executed.append((query, params))
            if query.startswith("SELECT version, deleted FROM persona_profiles"):
                return _Cursor({"version": 3, "deleted": True})
            if query.startswith("UPDATE persona_profiles"):
                return _Cursor(rowcount=1)
            raise AssertionError(f"Unexpected query: {query}")

    @contextmanager
    def _fake_transaction():
        yield _Conn()

    monkeypatch.setattr(type(db_instance), "backend_type", BackendType.POSTGRESQL)
    monkeypatch.setattr(db_instance, "transaction", _fake_transaction)
    monkeypatch.setattr(db_instance, "_prepare_backend_statement", lambda query, params: (query, params))

    assert db_instance.restore_persona_profile(
        persona_id="persona-1",
        user_id="user-1",
        expected_version=3,
    )

    update_query, update_params = next(
        (query, params)
        for query, params in executed
        if query.startswith("UPDATE persona_profiles")
    )
    assert "deleted = ?" in update_query
    assert update_params[-1] is True


def test_restore_persona_profile_with_stale_expected_version_raises_conflict(
    db_instance: CharactersRAGDB,
) -> None:
    persona_id = db_instance.create_persona_profile({"user_id": "user-1", "name": "Stale Restore Persona"})
    profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert profile is not None
    assert db_instance.soft_delete_persona_profile(
        persona_id=persona_id,
        user_id="user-1",
        expected_version=int(profile["version"]),
    )
    deleted_profile = db_instance.get_persona_profile(persona_id, user_id="user-1", include_deleted=True)
    assert deleted_profile is not None
    stale_version = int(deleted_profile["version"]) - 1

    with pytest.raises(ConflictError):
        db_instance.restore_persona_profile(
            persona_id=persona_id,
            user_id="user-1",
            expected_version=stale_version,
        )


def test_row_to_dict_raises_on_corrupt_derived_core(db_instance: CharactersRAGDB) -> None:
    """_persona_buddy_row_to_dict must raise CharactersRAGDBError when
    derived_core is empty/corrupt, not silently return resolved_profile=None."""
    fake_row = {
        "persona_id": "test-persona-corrupt",
        "user_id": "1",
        "derivation_version": 1,
        "source_fingerprint": "abc123",
        "derived_core_json": "{}",
        "overlay_preferences_json": "{}",
        "created_at": "2026-03-31T00:00:00",
        "last_modified": "2026-03-31T00:00:00",
        "version": 1,
    }
    with pytest.raises(CharactersRAGDBError, match="Unable to resolve persona buddy profile"):
        db_instance._persona_buddy_row_to_dict(fake_row)


def test_row_to_dict_succeeds_with_valid_data(db_instance: CharactersRAGDB) -> None:
    """Happy path: well-formed derived_core produces a resolved_profile."""
    import json

    derived_core = {
        "species_id": "owl",
        "silhouette_id": "owl_round",
        "palette_id": "moss",
        "behavior_family": "steady",
        "expression_profile": "warm",
    }
    fake_row = {
        "persona_id": "test-persona-valid",
        "user_id": "1",
        "derivation_version": 1,
        "source_fingerprint": "abc123",
        "derived_core_json": json.dumps(derived_core),
        "overlay_preferences_json": "{}",
        "created_at": "2026-03-31T00:00:00",
        "last_modified": "2026-03-31T00:00:00",
        "version": 1,
    }
    result = db_instance._persona_buddy_row_to_dict(fake_row)
    assert result is not None
    assert result["resolved_profile"] is not None
    assert result["resolved_profile"]["species_id"] == "owl"
    assert result["resolved_profile"]["compatibility_status"] == "exact"
