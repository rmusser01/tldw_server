"""Workspace upserts must refuse invisible PostgreSQL IDs without server errors."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response

from tldw_Server_API.app.api.v1.endpoints import workspace_migrations, workspaces
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceMigrationCreateRequest, WorkspaceUpsertRequest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError

pytestmark = pytest.mark.integration


@pytest.fixture
def owners(pg_restricted_backend, tmp_path):
    """Use the official shared database under its non-bypass owner role."""
    alice = CharactersRAGDB(tmp_path / "alice.db", client_id="1", backend=pg_restricted_backend)
    bob = CharactersRAGDB(tmp_path / "bob.db", client_id="2", backend=pg_restricted_backend)
    try:
        yield alice, bob
    finally:
        alice.close_all_connections()
        bob.close_all_connections()


def test_foreign_workspace_put_returns_404_and_preserves_owner(owners):
    alice, bob = owners
    before = alice.upsert_workspace("private-workspace", "Alice private workspace")
    assert bob.get_workspace(before["id"]) is None  # nosec B101 - pytest boundary oracle.
    try:
        with pytest.raises(HTTPException) as error:
            asyncio.run(
                workspaces.upsert_workspace(
                    before["id"],
                    WorkspaceUpsertRequest(name="Forbidden Bob edit"),
                    db=bob,
                    current_user=SimpleNamespace(id=2),
                )
            )
        assert (error.value.status_code, error.value.detail) == (404, "Workspace not found")  # nosec B101 - pytest public response oracle.
    finally:
        assert alice.get_workspace(before["id"]) == before  # nosec B101 - pytest full-row preservation oracle.
    assert bob.list_workspaces() == []  # nosec B101 - pytest no-copy oracle.
    created = bob.upsert_workspace("bob-workspace", "Bob workspace")
    assert bob.get_workspace(created["id"]) == created  # nosec B101 - pytest post-error connection reuse oracle.


@pytest.mark.parametrize("via_api", [False, True])
def test_foreign_workspace_migration_cannot_claim_target(owners, via_api):
    alice, bob = owners
    before = alice.upsert_workspace("private-target", "Alice target")
    payload = {
        "id": "bob-migration",
        "idempotency_key": "bob-key",
        "target_workspace_id": before["id"],
        "target_workspace_name": "Bob target",
        "manifest_hash": "a" * 64,
    }
    try:
        if via_api:
            with pytest.raises(HTTPException) as error:
                asyncio.run(
                    workspace_migrations.create_workspace_migration(
                        WorkspaceMigrationCreateRequest(**payload),
                        Response(),
                        db=bob,
                        current_user=SimpleNamespace(id=2),
                    )
                )
            assert (error.value.status_code, error.value.detail) == (404, "Workspace not found")  # nosec B101 - pytest migration response oracle.
        else:
            with pytest.raises(NotFoundError, match="^Workspace not found$"):
                bob.upsert_workspace_migration_session(payload)
    finally:
        assert alice.get_workspace(before["id"]) == before  # nosec B101 - pytest full-row preservation oracle.
    assert bob.get_workspace_migration_session("bob-migration") is None  # nosec B101 - pytest no-session oracle.


def test_separate_sqlite_owners_can_create_the_same_local_id(tmp_path):
    alice = CharactersRAGDB(tmp_path / "alice.db", client_id="1")
    bob = CharactersRAGDB(tmp_path / "bob.db", client_id="2")
    try:
        before = alice.upsert_workspace("local-id", "Alice")
        created = bob.upsert_workspace("local-id", "Bob")
        assert created["name"] == "Bob"  # nosec B101 - pytest separate-file create oracle.
        assert alice.get_workspace("local-id") == before  # nosec B101 - pytest owner preservation oracle.
        assert bob.upsert_workspace("local-id", "Bob") == created  # nosec B101 - pytest idempotent create oracle.
    finally:
        alice.close_all_connections()
        bob.close_all_connections()


def test_visible_create_race_returns_existing_workspace(tmp_path, monkeypatch):
    db = CharactersRAGDB(tmp_path / "race.db", client_id="1")
    try:
        before = db.upsert_workspace("raced-id", "Existing")
        original = db.get_workspace
        reads = 0

        def first_read_misses(workspace_id, **kwargs):
            nonlocal reads
            reads += 1
            return None if reads == 1 else original(workspace_id, **kwargs)

        monkeypatch.setattr(db, "get_workspace", first_read_misses)
        assert db.upsert_workspace("raced-id", "Losing insert") == before  # nosec B101 - pytest real unique-insert race recovery oracle.
    finally:
        db.close_all_connections()


def test_deleted_workspace_id_remains_a_conflict(tmp_path):
    db = CharactersRAGDB(tmp_path / "deleted.db", client_id="1")
    try:
        before = db.upsert_workspace("deleted-id", "Original")
        db.delete_workspace(before["id"], expected_version=before["version"])
        with pytest.raises(ConflictError):
            db.upsert_workspace("deleted-id", "Replacement")
    finally:
        db.close_all_connections()
