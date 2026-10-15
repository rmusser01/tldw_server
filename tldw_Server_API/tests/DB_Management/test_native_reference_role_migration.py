"""Explicit native reference roles preserve unknown published legacy records."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import ConstraintViolationError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.DB_Management.test_native_chat_asset_io import _reserve

pytestmark = pytest.mark.integration
TABLES = (
    "conversations",
    "messages",
    "native_chat_operations",
    "native_chat_asset_candidates",
    "native_chat_asset_claims",
    "native_chat_asset_references",
    "native_chat_quota_intents",
)
ROLES = ("message_image", "document_context", "generated_image", "reference_image")


def database_kwargs(backend_name: str, request: pytest.FixtureRequest, tmp_path: Path) -> dict[str, Any]:
    """Use the official PostgreSQL fixture without provisioning another database."""
    kwargs: dict[str, Any] = {
        "db_path": str(tmp_path / "role-upgrade.sqlite"),
        "client_id": "1",
        "owner_user_id": "1",
    }
    if backend_name == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    return kwargs


def close_pool(kwargs: dict[str, Any]) -> None:
    """Release the fixture-owned pool after its connections close."""
    if "backend" in kwargs:
        kwargs["backend"].get_pool().close_all()


def seed_legacy_references(db: CharactersRAGDB) -> None:
    """Seed schema-preservation records; this does not qualify byte publication."""
    cid = db.add_conversation({"character_id": None, "title": "Preserve unknown reference roles"})
    db.add_message({"conversation_id": cid, "sender": "user", "content": "Preserved résumé 😀"})
    _reserve(db, mime_type="image/png", representation="embedded_image_v1")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET required_projection_version = 'native-fork-v1', "
            "native_creation_operation_kind = 'native_fork_v1', native_creation_operation_id = 'fork-one', "
            "native_bundle_json = ? WHERE id = ?",
            ('{"version":1,"state":"ready","asset_manifest_revision":"original-opaque-😀"}', cid),
        )
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', child_conversation_id = ?, "
            "result_json = ? WHERE operation_id = 'fork-one'",
            (cid, '{"original_receipt":true}'),
        )
        conn.execute(
            "INSERT INTO native_chat_asset_claims "
            "(client_id,claim_id,owner_key,conversation_id,candidate_id,asset_id,asset_revision,content_hash,"
            "size_bytes,mime_type,representation,state,created_at,updated_at) "
            "SELECT client_id,'claim-one',owner_key,?,candidate_id,'asset-one',7,expected_hash,"
            "expected_size_bytes,mime_type,representation,'live',created_at,updated_at "
            "FROM native_chat_asset_candidates WHERE candidate_id = 'candidate-one'",
            (cid,),
        )
        for reference, position, enabled in (("ref-one", 0, True), ("ref-two", 1, False)):
            conn.execute(
                "INSERT INTO native_chat_asset_references "
                "(client_id,reference_id,owner_key,conversation_id,claim_id,position,revision,"
                "context_enabled,created_at,updated_at) VALUES ('1',?,'native:1',?,'claim-one',?,3,?,"
                "'2026-01-01','2026-01-01')",
                (reference, cid, position, enabled),
            )


def snapshot(db: CharactersRAGDB) -> dict[str, list[dict[str, Any]]]:
    """Compare every seeded field without reinterpreting a legacy role or receipt."""
    return {
        table: sorted(
            (dict(row) for row in db.execute_query(f"SELECT * FROM {table}").fetchall()),
            key=lambda row: repr(sorted(row.items())),
        )
        for table in TABLES
    }


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_fresh_and_reopened_schema_accepts_only_explicit_finite_roles(
    backend_name: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    kwargs = database_kwargs(backend_name, request, tmp_path)
    db = CharactersRAGDB(**kwargs)
    try:
        seed_legacy_references(db)
        for _ in range(2):
            db.close_connection()
            db = CharactersRAGDB(**kwargs)
            references = snapshot(db)["native_chat_asset_references"]
            assert all(row["role"] is None for row in references)
        for role in ROLES:
            with db.transaction() as conn:
                conn.execute("UPDATE native_chat_asset_references SET role = ? WHERE reference_id = 'ref-one'", (role,))
            assert (
                db.execute_query(
                    "SELECT role FROM native_chat_asset_references WHERE reference_id = 'ref-one'"
                ).fetchone()["role"]
                == role
            )
        before = snapshot(db)
        db.close_connection()
        db = CharactersRAGDB(**kwargs)
        assert snapshot(db) == before
        for invalid in ("image", "Message_image", "message_image ", "", "unavailable_v1"):
            error = ConstraintViolationError if backend_name == "postgres" else sqlite3.IntegrityError
            match = None if backend_name == "postgres" else "(?i)check constraint"
            with pytest.raises(error, match=match):
                with db.transaction() as conn:
                    conn.execute("UPDATE native_chat_asset_references SET role = ?", (invalid,))
            assert snapshot(db) == before
    finally:
        db.close_connection()
        close_pool(kwargs)


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_previous_schema_upgrade_preserves_all_records_and_leaves_roles_unknown(
    backend_name: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = database_kwargs(backend_name, request, tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 79)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 83)
        old = CharactersRAGDB(**kwargs)
        seed_legacy_references(old)
        before = snapshot(old)
        assert all("role" not in row for row in before["native_chat_asset_references"])
        old.close_connection()
    expected = before | {
        "native_chat_asset_references": [row | {"role": None} for row in before["native_chat_asset_references"]]
    }
    incarnations = None
    try:
        for _ in range(2):
            upgraded = CharactersRAGDB(**kwargs)
            try:
                current = snapshot(upgraded)
                observed = {row["id"]: row.pop("source_incarnation_id") for row in current["conversations"]}
                assert all(isinstance(value, str) and value for value in observed.values())
                if incarnations is not None:
                    assert observed == incarnations
                incarnations = observed
                assert current == expected
                assert upgraded.execute_query("SELECT version FROM db_schema_version").fetchone()["version"] == (
                    CharactersRAGDB._POSTGRES_SCHEMA_VERSION
                    if backend_name == "postgres"
                    else CharactersRAGDB._CURRENT_SCHEMA_VERSION
                )
            finally:
                upgraded.close_connection()
    finally:
        close_pool(kwargs)


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_interrupted_role_upgrade_rolls_back_ddl_version_and_records(
    backend_name: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = database_kwargs(backend_name, request, tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 79)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 83)
        old = CharactersRAGDB(**kwargs)
        seed_legacy_references(old)
        before = snapshot(old)
        old.close_connection()
    name = "_migrate_from_v83_to_v84_postgres" if backend_name == "postgres" else "_migrate_from_v79_to_v80"
    migrate = getattr(CharactersRAGDB, name)

    def interrupted(self: CharactersRAGDB, conn: Any) -> None:
        migrate(self, conn)
        raise RuntimeError("role migration interrupted")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(CharactersRAGDB, name, interrupted)
            with pytest.raises(Exception, match="role migration interrupted"):
                CharactersRAGDB(**kwargs)
        assert old.execute_query("SELECT version FROM db_schema_version").fetchone()["version"] == (
            83 if backend_name == "postgres" else 79
        )
        assert snapshot(old) == before
    finally:
        old.close_connection()
        close_pool(kwargs)
