"""Tests for workspace CRUD endpoints and scoped chat session isolation."""
import asyncio
import base64
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Query
from fastapi import Response as FastAPIResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from httpx import Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sandbox.store import IdempotencyConflict, InMemoryStore
from tldw_Server_API.app.core.Sandbox.workspace_volumes import SandboxWorkspaceVolumeService
from tldw_Server_API.app.core.Workspaces import root_binding_service


@pytest.fixture(params=["sqlite", "postgresql"])
def deletion_db(request, tmp_path):
    """Exercise the same deletion contract on both supported storage backends."""
    backend = None
    if request.param == "postgresql":
        from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    database = CharactersRAGDB(
        db_path=str(tmp_path / "deletion.db"), client_id="user-1", backend=backend,
    )
    database.add_character_card({"name": "Deletion Char"})
    try:
        yield database
    finally:
        database.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _seed_deletion_graph(db, message_count=2):
    db.upsert_workspace("ws-delete", "Private workspace")
    character = db.get_character_card_by_name("Deletion Char")
    conversation_id = db.add_conversation({
        "title": "Private chat", "character_id": character["id"],
        "scope_type": "workspace", "workspace_id": "ws-delete",
    })
    message_ids = [db.add_message({
        "conversation_id": conversation_id, "sender": "user", "content": "deletionneedle",
    }) for _ in range(message_count)]
    quiz_id = db.create_quiz(name="Quiz", workspace_id="ws-delete")
    deck_id = db.add_deck("Deck", workspace_id="ws-delete")
    return conversation_id, message_ids, quiz_id, deck_id


_CONTENT_WRITES = [
    "add_source", "retry_source", "update_source", "review_sources", "delete_source",
    "select_sources", "reorder_sources", "add_artifact", "export_artifact",
    "update_artifact", "delete_artifact", "add_note", "update_note", "delete_note",
    "add_conversation", "add_conversation_with_connection",
]


def _seed_workspace_content(db):
    db.add_workspace_source("ws-delete", {"id": "source", "title": "Source"})
    db.add_workspace_artifact("ws-delete", {"id": "artifact", "content": "Original"})
    return db.add_workspace_note("ws-delete", {"title": "Note", "content": "Original"})["id"]


@pytest.mark.parametrize("selected", [True, False])
def test_workspace_source_selection_roundtrip_on_both_backends(deletion_db, selected):
    db = deletion_db
    db.upsert_workspace("ws-delete", "Workspace")
    source = db.add_workspace_source("ws-delete", {"id": "source", "selected": selected})
    assert bool(source["selected"]) is selected
    updated = db.update_workspace_source(
        "ws-delete", "source", {"selected": not selected}, expected_version=1,
    )
    assert bool(updated["selected"]) is not selected
    db.update_workspace_source_selection("ws-delete", selected_ids=["source"])
    assert bool(db.list_workspace_sources("ws-delete")[0]["selected"]) is True
    db.update_workspace_source_selection("ws-delete", selected_ids=[])
    assert bool(db.list_workspace_sources("ws-delete")[0]["selected"]) is False


def _mutate_workspace_content(db, operation, note_id):
    """Exercise public DB mutation methods, not their implementation helpers."""
    if operation in {"add_source", "retry_source"}:
        return db.add_workspace_source("ws-delete", {
            "id": "source" if operation == "retry_source" else "new-source", "title": "New",
        })
    if operation == "update_source":
        return db.update_workspace_source("ws-delete", "source", {"title": "Updated"}, expected_version=1)
    if operation == "review_sources":
        return db.update_workspace_source_review_states("ws-delete", ["source"], "reviewed", "1")
    if operation == "delete_source":
        return db.delete_workspace_source("ws-delete", "source")
    if operation == "select_sources":
        return db.update_workspace_source_selection("ws-delete", selected_ids=[])
    if operation == "reorder_sources":
        return db.reorder_workspace_sources("ws-delete", ["source"])
    if operation == "add_artifact":
        return db.add_workspace_artifact("ws-delete", {"id": "new-artifact", "content": "New"})
    if operation == "export_artifact":
        return db.append_workspace_artifact_export_ref("ws-delete", "artifact", {"id": "export"})
    if operation == "update_artifact":
        return db.update_workspace_artifact("ws-delete", "artifact", {"content": "Updated"}, expected_version=1)
    if operation == "delete_artifact":
        return db.delete_workspace_artifact("ws-delete", "artifact")
    if operation == "add_note":
        return db.add_workspace_note("ws-delete", {"title": "New"})
    if operation == "update_note":
        return db.update_workspace_note("ws-delete", note_id, {"content": "Updated"}, expected_version=1)
    if operation == "delete_note":
        return db.delete_workspace_note("ws-delete", note_id)
    if operation in {"add_conversation", "add_conversation_with_connection"}:
        data = {"id": "new-chat", "title": "Chat", "scope_type": "workspace", "workspace_id": "ws-delete"}
        if operation == "add_conversation_with_connection":
            with db.transaction() as conn:
                return db.add_conversation(data, conn=conn)
        return db.add_conversation(data)
    raise AssertionError(f"Unknown content operation: {operation}")


def _workspace_content_snapshot(db):
    return {
        table: [dict(row) for row in db.execute_query(
            f"SELECT * FROM {table} ORDER BY 1",  # Fixed test table names only.
        ).fetchall()]
        for table in (
            "workspace_sources", "workspace_notes", "workspace_artifacts", "workspace_artifact_versions", "conversations",
        )
    }


@pytest.mark.parametrize("operation", _CONTENT_WRITES)
def test_deleted_workspace_rejects_content_mutation(deletion_db, operation):
    db = deletion_db
    db.upsert_workspace("ws-delete", "Workspace")
    note_id = _seed_workspace_content(db)
    assert db.delete_workspace("ws-delete", 1)
    before = _workspace_content_snapshot(db)
    with pytest.raises(ConflictError, match="Workspace.*not found|Workspace.*deleted"):
        _mutate_workspace_content(db, operation, note_id)
    assert _workspace_content_snapshot(db) == before


@pytest.mark.parametrize("operation", [item for item in _CONTENT_WRITES if item != "retry_source"])
@pytest.mark.parametrize("staged", [False, True])
def test_content_mutation_preserves_parent_metadata_and_clone_staging(deletion_db, operation, staged):
    db = deletion_db
    if staged:
        db.reserve_clone_target(
            workspace_id="ws-delete", operation_id="clone-op", request_fingerprint="fingerprint",
            name="Clone", description=None, workspace_profile="research",
        )
    else:
        db.upsert_workspace("ws-delete", "Workspace")
    before = db._get_workspace_internal("ws-delete")
    note_id = _seed_workspace_content(db)
    _mutate_workspace_content(db, operation, note_id)
    assert db._get_workspace_internal("ws-delete") == before
    if staged:
        assert db.get_workspace("ws-delete") is None


@pytest.mark.parametrize("operation", ["add_source", "add_artifact", "add_note", "add_conversation", "add_conversation_with_connection"])
def test_missing_workspace_content_write_is_domain_conflict(deletion_db, operation):
    with pytest.raises(ConflictError):
        _mutate_workspace_content(deletion_db, operation, 1)
    assert all(not rows for rows in _workspace_content_snapshot(deletion_db).values())


@pytest.mark.parametrize("operation", ["add_source", "update_source", "add_artifact", "update_artifact", "add_note", "update_note", "add_conversation", "add_conversation_with_connection"])
@pytest.mark.parametrize("commit_delete", [True, False])
def test_pre_admitted_content_writer_waits_for_deletion_outcome(deletion_db, operation, commit_delete):
    db = deletion_db
    db.upsert_workspace("ws-delete", "Workspace")
    note_id = _seed_workspace_content(db)
    before = _workspace_content_snapshot(db)
    admitted, proceed, attempting = threading.Event(), threading.Event(), threading.Event()

    def write():
        try:
            assert db.get_workspace("ws-delete") is not None
            admitted.set()
            assert proceed.wait(10)
            attempting.set()
            return _mutate_workspace_content(db, operation, note_id)
        finally:
            db.close_connection()

    class AbortDeletion(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(write)
        try:
            assert admitted.wait(10)
            try:
                with db.transaction():
                    assert db.delete_workspace("ws-delete", 1)
                    proceed.set()
                    assert attempting.wait(10)
                    with pytest.raises(FutureTimeout):
                        future.result(timeout=0.2)
                    if not commit_delete:
                        raise AbortDeletion
            except AbortDeletion:
                pass
            if commit_delete:
                with pytest.raises(ConflictError):
                    future.result(timeout=10)
                assert _workspace_content_snapshot(db) == before
            else:
                future.result(timeout=10)
                assert db.get_workspace("ws-delete") is not None
                assert _workspace_content_snapshot(db) != before
        finally:
            proceed.set()


@pytest.mark.parametrize("operation", ["add_source", "update_source", "add_artifact", "update_artifact", "add_note", "update_note", "add_conversation", "add_conversation_with_connection"])
@pytest.mark.parametrize("commit_write", [True, False])
def test_deletion_waits_for_content_writer_transaction(deletion_db, operation, commit_write):
    db = deletion_db
    db.upsert_workspace("ws-delete", "Workspace")
    note_id = _seed_workspace_content(db)
    before = _workspace_content_snapshot(db)
    attempting = threading.Event()

    def delete():
        try:
            attempting.set()
            return db.delete_workspace("ws-delete", 1)
        finally:
            db.close_connection()

    class AbortWrite(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            with db.transaction():
                _mutate_workspace_content(db, operation, note_id)
                future = executor.submit(delete)
                assert attempting.wait(10)
                with pytest.raises(FutureTimeout):
                    future.result(timeout=0.2)
                if not commit_write:
                    raise AbortWrite
        except AbortWrite:
            pass
        assert future.result(timeout=10)
    assert db.get_workspace("ws-delete") is None
    if not commit_write:
        assert _workspace_content_snapshot(db) == before
    elif operation.startswith("add_conversation"):
        conversation = db.get_conversation_by_id("new-chat", include_deleted=True)
        assert conversation["deleted"]


@pytest.mark.parametrize("with_connection", [False, True])
def test_global_chat_creation_does_not_require_a_workspace(deletion_db, with_connection):
    db = deletion_db
    data = {"title": "Global chat", "workspace_id": "missing-workspace"}
    if with_connection:
        with db.transaction() as conn:
            conversation_id = db.add_conversation(data, conn=conn)
    else:
        conversation_id = db.add_conversation(data)
    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation["scope_type"] == "global"
    assert conversation["workspace_id"] is None


@pytest.mark.parametrize("operation", ["add_artifact", "update_artifact"])
def test_artifact_version_failure_rolls_back_content_and_releases_parent_lock(deletion_db, monkeypatch, operation):
    db = deletion_db
    db.upsert_workspace("ws-delete", "Workspace")
    note_id = _seed_workspace_content(db)
    before = _workspace_content_snapshot(db)
    parent = db.get_workspace("ws-delete")

    def fail_version(*args, **kwargs):
        raise RuntimeError("injected artifact version failure")

    monkeypatch.setattr(db, "_insert_workspace_artifact_version", fail_version)
    with pytest.raises(RuntimeError, match="injected artifact version failure"):
        _mutate_workspace_content(db, operation, note_id)
    assert _workspace_content_snapshot(db) == before
    assert db.get_workspace("ws-delete") == parent

    def delete():
        try:
            return db.delete_workspace("ws-delete", 1)
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(delete)
        try:
            assert future.result(timeout=10)
        finally:
            db.close_connection()


def _deletion_snapshot(db):
    return {
        "workspace": db.get_workspace("ws-delete", include_deleted=True),
        "conversations": [dict(row) for row in db.execute_query("SELECT * FROM conversations").fetchall()],
        "messages": [dict(row) for row in db.execute_query("SELECT * FROM messages").fetchall()],
        "quizzes": [dict(row) for row in db.execute_query("SELECT * FROM quizzes").fetchall()],
        "decks": [dict(row) for row in db.execute_query("SELECT * FROM decks").fetchall()],
        "sync": db.get_sync_log_entries(since_change_id=0, limit=1000),
        "search": db.search_messages_by_content("deletionneedle", limit=200),
    }


def _seed_trashed_workspace_chat(db):
    conversation_id, *_ = _seed_deletion_graph(db)
    conversation = db.get_conversation_by_id(conversation_id)
    assert db.soft_delete_conversation(conversation_id, conversation["version"])
    return db.get_conversation_by_id(conversation_id, include_deleted=True)


def _update_workspace_chat_metadata(
    db, conversation_id, *, workspace_scoped=True, expected_version=None, empty_update=False,
):
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import update_chat_session
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionUpdate

    return asyncio.run(update_chat_session(
        update_data=ChatSessionUpdate() if empty_update else ChatSessionUpdate(title="Updated"),
        chat_id=conversation_id,
        expected_version=(
            db.get_conversation_by_id(conversation_id)["version"]
            if expected_version is None else expected_version
        ),
        scope_type="workspace" if workspace_scoped else "global",
        workspace_id="ws-delete" if workspace_scoped else None,
        db=db, current_user=SimpleNamespace(id=db.client_id),
    ))


def test_workspace_chat_metadata_rejects_active_projection_under_deleted_parent(deletion_db):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    assert db.delete_workspace("ws-delete", 1)
    assert db.upsert_conversation_from_sync(
        conversation_id=conversation_id, title="Retained chat", sync_client_id=db.client_id,
        object_revision=5, object_hash="retained", scope_type="workspace", workspace_id="ws-delete",
    )
    before = _deletion_snapshot(db)
    with pytest.raises(HTTPException) as exc:
        _update_workspace_chat_metadata(db, conversation_id)
    assert exc.value.status_code == 409
    assert _deletion_snapshot(db) == before


@pytest.mark.parametrize("workspace_scoped", [False, True])
@pytest.mark.parametrize("empty_update", [False, True])
def test_chat_metadata_update_preserves_active_parent_and_global_behavior(deletion_db, workspace_scoped, empty_update):
    db = deletion_db
    if workspace_scoped:
        conversation_id, *_ = _seed_deletion_graph(db)
    else:
        conversation_id = db.add_conversation({"title": "Global chat"})
    parent_before = db.get_workspace("ws-delete")
    version_before = db.get_conversation_by_id(conversation_id)["version"]
    response = _update_workspace_chat_metadata(
        db, conversation_id, workspace_scoped=workspace_scoped, empty_update=empty_update,
    )
    original_title = "Private chat" if workspace_scoped else "Global chat"
    assert response.title == (original_title if empty_update else "Updated")
    assert db.get_conversation_by_id(conversation_id)["version"] == version_before + 1
    assert db.get_workspace("ws-delete") == parent_before


@pytest.mark.parametrize("new_identity", ["global", "other_workspace", "other_owner"])
def test_chat_metadata_rechecks_identity_after_preflight(deletion_db, monkeypatch, new_identity):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    db.upsert_workspace("ws-other", "Other workspace")
    get_conversation = db.get_conversation_by_id
    after_move = {}

    def move_after_preflight(*args, **kwargs):
        conversation = get_conversation(*args, **kwargs)
        monkeypatch.setattr(db, "get_conversation_by_id", get_conversation)
        assert db.upsert_conversation_from_sync(
            conversation_id=conversation_id, title="Moved", object_revision=1,
            object_hash="moved", sync_client_id="other-owner" if new_identity == "other_owner" else db.client_id,
            scope_type="global" if new_identity == "global" else "workspace",
            workspace_id=(
                None if new_identity == "global"
                else "ws-other" if new_identity == "other_workspace" else "ws-delete"
            ),
        )
        after_move.update(_deletion_snapshot(db))
        return conversation

    monkeypatch.setattr(db, "get_conversation_by_id", move_after_preflight)
    with pytest.raises(HTTPException) as exc:
        _update_workspace_chat_metadata(db, conversation_id, expected_version=1)
    assert exc.value.status_code == 404
    assert _deletion_snapshot(db) == after_move


def test_chat_metadata_rejects_version_change_after_preflight(deletion_db, monkeypatch):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    get_conversation = db.get_conversation_by_id
    after_competing_write = {}

    def update_after_preflight(*args, **kwargs):
        conversation = get_conversation(*args, **kwargs)
        monkeypatch.setattr(db, "get_conversation_by_id", get_conversation)
        assert db.update_conversation(conversation_id, {"title": "Competing writer"}, 1)
        after_competing_write.update(_deletion_snapshot(db))
        return conversation

    monkeypatch.setattr(db, "get_conversation_by_id", update_after_preflight)
    with pytest.raises(HTTPException) as exc:
        _update_workspace_chat_metadata(db, conversation_id, expected_version=1)
    assert exc.value.status_code == 409
    assert _deletion_snapshot(db) == after_competing_write


def test_chat_metadata_response_failure_rolls_back_mutation(deletion_db, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions

    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    before = _deletion_snapshot(db)

    def fail_response(*args, **kwargs):
        raise ValueError("injected response failure")

    monkeypatch.setattr(character_chat_sessions, "_convert_db_conversation_to_response", fail_response)
    with pytest.raises(HTTPException) as exc:
        _update_workspace_chat_metadata(db, conversation_id)
    assert exc.value.status_code == 500
    assert _deletion_snapshot(db) == before


@pytest.mark.parametrize("commit_delete", [False, True])
def test_workspace_chat_metadata_waits_for_deletion_outcome(deletion_db, monkeypatch, commit_delete):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    preflight, proceed, attempting = threading.Event(), threading.Event(), threading.Event()
    get_conversation = db.get_conversation_by_id

    def pause_after_preflight(*args, **kwargs):
        conversation = get_conversation(*args, **kwargs)
        preflight.set()
        assert proceed.wait(10)
        attempting.set()
        return conversation

    def update():
        try:
            return _update_workspace_chat_metadata(db, conversation_id, expected_version=1)
        finally:
            db.close_connection()

    class AbortDeletion(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        with monkeypatch.context() as patch:
            patch.setattr(db, "get_conversation_by_id", pause_after_preflight)
            future = executor.submit(update)
            try:
                assert preflight.wait(10)
                patch.undo()
                try:
                    with db.transaction():
                        assert db.delete_workspace("ws-delete", 1)
                        after_delete = _deletion_snapshot(db)
                        proceed.set()
                        assert attempting.wait(10)
                        with pytest.raises(FutureTimeout):
                            future.result(timeout=0.2)
                        if not commit_delete:
                            raise AbortDeletion
                except AbortDeletion:
                    pass
            finally:
                proceed.set()
        if commit_delete:
            with pytest.raises(HTTPException) as exc:
                future.result(timeout=10)
            assert exc.value.status_code == 409
            assert _deletion_snapshot(db) == after_delete
        else:
            assert future.result(timeout=10).title == "Updated"
            assert db.get_workspace("ws-delete") is not None


@pytest.mark.parametrize("commit_update", [False, True])
def test_workspace_deletion_waits_for_chat_metadata_transaction(deletion_db, commit_update):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    attempting = threading.Event()

    def delete():
        try:
            attempting.set()
            return db.delete_workspace("ws-delete", 1)
        finally:
            db.close_connection()

    class AbortUpdate(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            with db.transaction():
                assert _update_workspace_chat_metadata(db, conversation_id).title == "Updated"
                future = executor.submit(delete)
                assert attempting.wait(10)
                with pytest.raises(FutureTimeout):
                    future.result(timeout=0.2)
                if not commit_update:
                    raise AbortUpdate
        except AbortUpdate:
            pass
        assert future.result(timeout=10)
    assert db.get_workspace("ws-delete") is None
    retained = db.get_conversation_by_id(conversation_id, include_deleted=True)
    assert retained["title"] == ("Updated" if commit_update else "Private chat")


@pytest.mark.parametrize("deletion_db", ["postgresql"], indirect=True)
def test_workspace_chat_metadata_does_not_lock_child_before_parent(deletion_db, monkeypatch):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    parent_claimed, finish_delete, update_attempting = (
        threading.Event(), threading.Event(), threading.Event()
    )
    get_messages = db.get_messages_for_conversation
    lock_parent = db._lock_workspace_for_content_write

    def pause_before_children(*args, **kwargs):
        parent_claimed.set()
        assert finish_delete.wait(10)
        return get_messages(*args, **kwargs)

    def observe_parent_admission(conn, workspace_id):
        update_attempting.set()
        return lock_parent(conn, workspace_id)

    def delete():
        try:
            return db.delete_workspace("ws-delete", 1)
        finally:
            db.close_connection()

    def update():
        try:
            return _update_workspace_chat_metadata(db, conversation_id, expected_version=1)
        finally:
            db.close_connection()

    monkeypatch.setattr(db, "get_messages_for_conversation", pause_before_children)
    monkeypatch.setattr(db, "_lock_workspace_for_content_write", observe_parent_admission)
    with ThreadPoolExecutor(max_workers=2) as executor:
        deleting = executor.submit(delete)
        try:
            assert parent_claimed.wait(10)
            updating = executor.submit(update)
            assert update_attempting.wait(10)
            with pytest.raises(FutureTimeout):
                updating.result(timeout=0.2)
        finally:
            finish_delete.set()
        assert deleting.result(timeout=10)
        with pytest.raises(HTTPException) as exc:
            updating.result(timeout=10)
        assert exc.value.status_code == 409
    assert db.get_workspace("ws-delete") is None
    assert db.get_conversation_by_id(conversation_id) is None


def _edit_workspace_message(db, message_id, *, pinned=None, content="Edited", workspace_scoped=True):
    from tldw_Server_API.app.api.v1.endpoints.character_messages import edit_message
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import MessageUpdate

    return asyncio.run(edit_message(
        update_data=MessageUpdate(content=content, pinned=pinned), message_id=message_id,
        expected_version=1, scope_type="workspace" if workspace_scoped else "global",
        workspace_id="ws-delete" if workspace_scoped else None,
        db=db, current_user=SimpleNamespace(id=db.client_id),
    ))


def _message_edit_snapshot(db):
    return {
        "graph": _deletion_snapshot(db),
        "settings": [dict(row) for row in db.execute_query("SELECT * FROM conversation_settings").fetchall()],
        "metadata": [dict(row) for row in db.execute_query("SELECT * FROM message_metadata").fetchall()],
    }


@pytest.fixture
def message_edit_limiter(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    async def allow(*args, **kwargs):
        return None

    monkeypatch.setattr(character_messages, "get_character_rate_limiter", lambda: SimpleNamespace(check_rate_limit=allow))


@pytest.mark.parametrize("pinned", [None, True])
def test_message_edit_rejects_deleted_workspace_parent(deletion_db, message_edit_limiter, pinned):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    assert db.delete_workspace("ws-delete", 1)
    assert db.upsert_conversation_from_sync(
        conversation_id=conversation_id, title="Retained", sync_client_id=db.client_id,
        object_revision=5, object_hash="retained", scope_type="workspace", workspace_id="ws-delete",
    )
    message_id = db.add_message({"conversation_id": conversation_id, "sender": "user", "content": "Retained"})
    before = _message_edit_snapshot(db)
    with pytest.raises(HTTPException) as exc:
        _edit_workspace_message(db, message_id, pinned=pinned)
    assert exc.value.status_code == 409
    assert _message_edit_snapshot(db) == before


@pytest.mark.parametrize("new_identity", ["global", "other_workspace", "other_owner", "message_reparent", "global_to_workspace"])
def test_message_edit_rechecks_locked_identity(deletion_db, monkeypatch, message_edit_limiter, new_identity):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    db = deletion_db
    conversation_id, message_ids, *_ = _seed_deletion_graph(db)
    if new_identity == "global_to_workspace":
        conversation_id = db.add_conversation({"title": "Global"})
        message_ids = [db.add_message({"conversation_id": conversation_id, "sender": "user", "content": "Original"})]
    db.upsert_workspace("ws-other", "Other")
    destination = db.add_conversation({"title": "Destination", "scope_type": "workspace", "workspace_id": "ws-other"})
    verify = character_messages._verify_message_access
    after_move = {}

    def move_after_preflight(*args, **kwargs):
        message = verify(*args, **kwargs)
        if new_identity == "message_reparent":
            with db.transaction() as conn:
                conn.execute("UPDATE messages SET conversation_id = ? WHERE id = ?", (destination, message_ids[0]))
        else:
            assert db.upsert_conversation_from_sync(
                conversation_id=conversation_id, title="Moved", object_revision=1, object_hash="moved",
                sync_client_id="other-owner" if new_identity == "other_owner" else db.client_id,
                scope_type="global" if new_identity == "global" else "workspace",
                workspace_id=None if new_identity == "global" else "ws-other" if new_identity == "other_workspace" else "ws-delete",
            )
        after_move.update(_message_edit_snapshot(db))
        return message

    monkeypatch.setattr(character_messages, "_verify_message_access", move_after_preflight)
    with pytest.raises(HTTPException) as exc:
        _edit_workspace_message(db, message_ids[0], pinned=True, workspace_scoped=new_identity != "global_to_workspace")
    assert exc.value.status_code == 404
    assert _message_edit_snapshot(db) == after_move


@pytest.mark.parametrize("failure", ["metadata_bump", "response", "response_metadata"])
def test_message_edit_failure_rolls_back_all_writes(deletion_db, monkeypatch, message_edit_limiter, failure):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    db = deletion_db
    _, message_ids, *_ = _seed_deletion_graph(db)
    before = _message_edit_snapshot(db)

    def fail(*args, **kwargs):
        raise CharactersRAGDBError("injected edit failure")

    if failure == "metadata_bump":
        monkeypatch.setattr(db, "update_conversation", fail)
    elif failure == "response":
        monkeypatch.setattr(character_messages, "_convert_db_message_to_response", fail)
    else:
        monkeypatch.setattr(db, "get_message_metadata", fail)
    with pytest.raises(HTTPException) as exc:
        _edit_workspace_message(db, message_ids[0], pinned=True)
    assert exc.value.status_code == 500
    assert _message_edit_snapshot(db) == before


@pytest.mark.parametrize("commit_delete", [False, True])
def test_message_edit_waits_for_workspace_deletion(deletion_db, monkeypatch, message_edit_limiter, commit_delete):
    from tldw_Server_API.app.api.v1.endpoints import character_messages

    db = deletion_db
    _, message_ids, *_ = _seed_deletion_graph(db)
    preflight, proceed, attempting = threading.Event(), threading.Event(), threading.Event()
    verify = character_messages._verify_message_access

    def pause_after_preflight(*args, **kwargs):
        message = verify(*args, **kwargs)
        preflight.set()
        assert proceed.wait(10)
        attempting.set()
        return message

    def edit():
        try:
            return _edit_workspace_message(db, message_ids[0], pinned=True)
        finally:
            db.close_connection()

    class AbortDeletion(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(character_messages, "_verify_message_access", pause_after_preflight)
        future = executor.submit(edit)
        try:
            assert preflight.wait(10)
            try:
                with db.transaction():
                    assert db.delete_workspace("ws-delete", 1)
                    after_delete = _message_edit_snapshot(db)
                    proceed.set()
                    assert attempting.wait(10)
                    with pytest.raises(FutureTimeout):
                        future.result(timeout=0.2)
                    if not commit_delete:
                        raise AbortDeletion
            except AbortDeletion:
                pass
        finally:
            proceed.set()
        if commit_delete:
            with pytest.raises(HTTPException) as exc:
                future.result(timeout=10)
            assert exc.value.status_code == 409
            assert _message_edit_snapshot(db) == after_delete
        else:
            assert future.result(timeout=10).content == "Edited"
            assert db.get_message_metadata(message_ids[0])["extra"]["pinned"] is True


@pytest.mark.parametrize("commit_edit", [False, True])
def test_workspace_deletion_waits_for_message_edit(deletion_db, message_edit_limiter, commit_edit):
    db = deletion_db
    conversation_id, message_ids, *_ = _seed_deletion_graph(db)
    attempting = threading.Event()

    def delete():
        try:
            attempting.set()
            return db.delete_workspace("ws-delete", 1)
        finally:
            db.close_connection()

    class AbortEdit(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            with db.transaction():
                assert _edit_workspace_message(db, message_ids[0], pinned=True).content == "Edited"
                future = executor.submit(delete)
                assert attempting.wait(10)
                with pytest.raises(FutureTimeout):
                    future.result(timeout=0.2)
                if not commit_edit:
                    raise AbortEdit
        except AbortEdit:
            pass
        assert future.result(timeout=10)
    retained = db.get_message_by_id(message_ids[0], include_deleted=True)
    assert retained["content"] == ("Edited" if commit_edit else "deletionneedle")
    assert bool(retained["deleted"]) is True
    metadata = db.get_message_metadata(message_ids[0])
    assert (metadata or {}).get("extra", {}).get("pinned", False) is commit_edit
    settings = db.get_conversation_settings(conversation_id)
    assert bool(settings and message_ids[0] in settings["settings"]["pinnedMessageIds"]) is commit_edit


@pytest.mark.parametrize("deletion_db", ["postgresql"], indirect=True)
def test_message_edit_claims_parent_before_child_locks(deletion_db, monkeypatch, message_edit_limiter):
    db = deletion_db
    _, message_ids, *_ = _seed_deletion_graph(db)
    parent_claimed, finish_delete, attempting = threading.Event(), threading.Event(), threading.Event()
    get_messages = db.get_messages_for_conversation
    lock_parent = db._lock_workspace_for_content_write

    def pause_before_children(*args, **kwargs):
        parent_claimed.set()
        assert finish_delete.wait(10)
        return get_messages(*args, **kwargs)

    def observe_parent(conn, workspace_id):
        attempting.set()
        return lock_parent(conn, workspace_id)

    def delete():
        try:
            return db.delete_workspace("ws-delete", 1)
        finally:
            db.close_connection()

    def edit():
        try:
            return _edit_workspace_message(db, message_ids[0], pinned=True)
        finally:
            db.close_connection()

    monkeypatch.setattr(db, "get_messages_for_conversation", pause_before_children)
    monkeypatch.setattr(db, "_lock_workspace_for_content_write", observe_parent)
    with ThreadPoolExecutor(max_workers=2) as executor:
        deleting = executor.submit(delete)
        try:
            assert parent_claimed.wait(10)
            editing = executor.submit(edit)
            assert attempting.wait(10)
            with pytest.raises(FutureTimeout):
                editing.result(timeout=0.2)
        finally:
            finish_delete.set()
        assert deleting.result(timeout=10)
        with pytest.raises(HTTPException) as exc:
            editing.result(timeout=10)
        assert exc.value.status_code == 409


@pytest.mark.parametrize("workspace_scoped", [False, True])
@pytest.mark.parametrize("content,pinned", [("Edited", None), (None, True), (None, False), ("deletionneedle", None), ("Edited", True)])
def test_message_edit_preserves_global_and_workspace_behavior(deletion_db, message_edit_limiter, workspace_scoped, content, pinned):
    db = deletion_db
    if workspace_scoped:
        conversation_id, message_ids, *_ = _seed_deletion_graph(db)
        message_id = message_ids[0]
    else:
        conversation_id = db.add_conversation({"title": "Global"})
        message_id = db.add_message({"conversation_id": conversation_id, "sender": "user", "content": "deletionneedle"})
    before_parent = db.get_workspace("ws-delete")
    before_history = db.get_conversation_by_id(conversation_id)["history_version"]
    response = _edit_workspace_message(db, message_id, content=content, pinned=pinned, workspace_scoped=workspace_scoped)
    assert response.content == (content or "deletionneedle")
    assert response.version == 2
    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation["version"] == 1 + bool(content) + (pinned is not None)
    assert conversation["history_version"] == before_history + 1
    if pinned is not None:
        assert response.metadata_extra["pinned"] is pinned
        settings = db.get_conversation_settings(conversation_id)
        assert settings["settings_version"] == 1
        assert (message_id in settings["settings"]["pinnedMessageIds"]) is pinned
    assert db.get_workspace("ws-delete") == before_parent


@pytest.mark.parametrize("read_kind", ["images", "metadata"])
def test_message_edit_response_sql_failure_is_not_reported_as_success(
    deletion_db, monkeypatch, message_edit_limiter, read_kind,
):
    db = deletion_db
    _, message_ids, *_ = _seed_deletion_graph(db)
    before = _message_edit_snapshot(db)
    execute_query = db.execute_query
    update_conversation = db.update_conversation
    ready = False

    def mark_updated(*args, **kwargs):
        nonlocal ready
        result = update_conversation(*args, **kwargs)
        ready = True
        return result

    def fail_sql(query, *args, **kwargs):
        # Execute a real bad statement on the current transaction, not a getter mock.
        if ready and (
            (read_kind == "images" and "FROM message_images" in query)
            or (read_kind == "metadata" and "SELECT tool_calls_json, extra_json, last_modified" in query)
        ):
            return execute_query("SELECT * FROM missing_edit_response_table")
        return execute_query(query, *args, **kwargs)

    monkeypatch.setattr(db, "update_conversation", mark_updated)
    monkeypatch.setattr(db, "execute_query", fail_sql)
    # Metadata's conn-owned path executes directly on the same connection.
    transaction = db.transaction

    class FaultConnection:
        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def execute(self, query, *args, **kwargs):
            if ready and read_kind == "metadata" and "SELECT tool_calls_json, extra_json, last_modified" in query:
                return self.conn.execute("SELECT * FROM missing_edit_response_table")
            return self.conn.execute(query, *args, **kwargs)

    @contextmanager
    def faulty_transaction():
        with transaction() as conn:
            yield FaultConnection(conn)

    monkeypatch.setattr(db, "transaction", faulty_transaction)
    with pytest.raises(HTTPException) as exc:
        _edit_workspace_message(db, message_ids[0], pinned=True)
    assert exc.value.status_code == 500
    monkeypatch.undo()
    assert _message_edit_snapshot(db) == before


def _update_workspace_chat_settings(db, conversation_id, *, workspace_scoped=True):
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import update_chat_settings
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSettingsUpdate

    return asyncio.run(update_chat_settings(
        payload=ChatSettingsUpdate(settings={"authorNote": "Updated"}),
        chat_id=conversation_id,
        scope_type="workspace" if workspace_scoped else "global",
        workspace_id="ws-delete" if workspace_scoped else None,
        db=db, current_user=SimpleNamespace(id=db.client_id),
    ))


@pytest.mark.parametrize("new_identity", ["other_owner", "trashed", "global", "other_workspace"])
def test_chat_settings_rechecks_identity_after_preflight(deletion_db, monkeypatch, new_identity):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    db.upsert_workspace("ws-other", "Other workspace")
    assert db.upsert_conversation_settings(conversation_id, {"authorNote": "Original"})
    get_conversation = db.get_conversation_by_id
    after_change = {}

    def snapshot():
        return {
            "graph": _deletion_snapshot(db),
            "settings": [dict(row) for row in db.execute_query("SELECT * FROM conversation_settings").fetchall()],
        }

    def change_after_preflight(*args, **kwargs):
        conversation = get_conversation(*args, **kwargs)
        monkeypatch.setattr(db, "get_conversation_by_id", get_conversation)
        if new_identity == "trashed":
            assert db.soft_delete_conversation(conversation_id, conversation["version"])
        else:
            assert db.upsert_conversation_from_sync(
                conversation_id=conversation_id, title="Moved", object_revision=1,
                object_hash="moved", sync_client_id="other-owner" if new_identity == "other_owner" else db.client_id,
                scope_type="global" if new_identity == "global" else "workspace",
                workspace_id=(
                    None if new_identity == "global"
                    else "ws-other" if new_identity == "other_workspace" else "ws-delete"
                ),
            )
        after_change.update(snapshot())
        return conversation

    monkeypatch.setattr(db, "get_conversation_by_id", change_after_preflight)
    with pytest.raises(HTTPException) as exc:
        _update_workspace_chat_settings(db, conversation_id)
    assert exc.value.status_code == 404
    assert snapshot() == after_change


def test_workspace_chat_settings_reject_active_projection_under_deleted_parent(deletion_db):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    assert db.upsert_conversation_settings(conversation_id, {"authorNote": "Original"})
    assert db.delete_workspace("ws-delete", 1)
    assert db.upsert_conversation_from_sync(
        conversation_id=conversation_id, title="Retained chat", sync_client_id=db.client_id,
        object_revision=5, object_hash="retained", scope_type="workspace", workspace_id="ws-delete",
    )
    before = _deletion_snapshot(db)
    settings_before = db.get_conversation_settings(conversation_id)
    with pytest.raises(HTTPException) as exc:
        _update_workspace_chat_settings(db, conversation_id)
    assert exc.value.status_code == 409
    assert _deletion_snapshot(db) == before
    assert db.get_conversation_settings(conversation_id) == settings_before


@pytest.mark.parametrize("workspace_scoped", [False, True])
def test_chat_settings_update_preserves_active_parent_and_global_behavior(deletion_db, workspace_scoped):
    db = deletion_db
    if workspace_scoped:
        conversation_id, *_ = _seed_deletion_graph(db)
    else:
        conversation_id = db.add_conversation({"title": "Global chat"})
    parent_before = db.get_workspace("ws-delete")
    response = _update_workspace_chat_settings(db, conversation_id, workspace_scoped=workspace_scoped)
    assert response.settings["authorNote"] == "Updated"
    assert db.get_conversation_settings(conversation_id)["settings"]["authorNote"] == "Updated"
    assert db.get_workspace("ws-delete") == parent_before


@pytest.mark.parametrize("commit_delete", [False, True])
def test_workspace_chat_settings_wait_for_deletion_outcome(deletion_db, monkeypatch, commit_delete):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    preflight, proceed, attempting = threading.Event(), threading.Event(), threading.Event()
    get_conversation = db.get_conversation_by_id

    def pause_after_preflight(*args, **kwargs):
        conversation = get_conversation(*args, **kwargs)
        preflight.set()
        assert proceed.wait(10)
        attempting.set()
        return conversation

    def update():
        try:
            return _update_workspace_chat_settings(db, conversation_id)
        finally:
            db.close_connection()

    class AbortDeletion(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        with monkeypatch.context() as patch:
            patch.setattr(db, "get_conversation_by_id", pause_after_preflight)
            future = executor.submit(update)
            try:
                assert preflight.wait(10)
                # Only the endpoint preflight should be paused, not deletion's reads.
                patch.undo()
                try:
                    with db.transaction():
                        assert db.delete_workspace("ws-delete", 1)
                        after_delete = _deletion_snapshot(db)
                        proceed.set()
                        assert attempting.wait(10)
                        with pytest.raises(FutureTimeout):
                            future.result(timeout=0.2)
                        if not commit_delete:
                            raise AbortDeletion
                except AbortDeletion:
                    pass
            finally:
                proceed.set()
        if commit_delete:
            with pytest.raises(HTTPException) as exc:
                future.result(timeout=10)
            assert exc.value.status_code == 409
            assert _deletion_snapshot(db) == after_delete
            assert db.get_conversation_settings(conversation_id) is None
        else:
            assert future.result(timeout=10).settings["authorNote"] == "Updated"
            assert db.get_workspace("ws-delete") is not None


@pytest.mark.parametrize("commit_settings", [False, True])
def test_workspace_deletion_waits_for_chat_settings_transaction(deletion_db, commit_settings):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    attempting = threading.Event()

    def delete():
        try:
            attempting.set()
            return db.delete_workspace("ws-delete", 1)
        finally:
            db.close_connection()

    class AbortSettings(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            with db.transaction():
                assert _update_workspace_chat_settings(db, conversation_id).settings["authorNote"] == "Updated"
                future = executor.submit(delete)
                assert attempting.wait(10)
                with pytest.raises(FutureTimeout):
                    future.result(timeout=0.2)
                if not commit_settings:
                    raise AbortSettings
        except AbortSettings:
            pass
        assert future.result(timeout=10)
    assert db.get_workspace("ws-delete") is None
    assert db.get_conversation_by_id(conversation_id) is None
    settings = db.get_conversation_settings(conversation_id)
    if commit_settings:
        assert settings["settings"]["authorNote"] == "Updated"
    else:
        assert settings is None


@pytest.mark.parametrize("deletion_db", ["postgresql"], indirect=True)
def test_workspace_chat_settings_do_not_lock_child_before_parent(deletion_db, monkeypatch):
    db = deletion_db
    conversation_id, *_ = _seed_deletion_graph(db)
    parent_claimed, finish_delete, settings_attempting = (
        threading.Event(), threading.Event(), threading.Event()
    )
    get_messages = db.get_messages_for_conversation
    lock_parent = db._lock_workspace_for_content_write

    def pause_before_children(*args, **kwargs):
        parent_claimed.set()
        assert finish_delete.wait(10)
        return get_messages(*args, **kwargs)

    def observe_parent_admission(conn, workspace_id):
        settings_attempting.set()
        return lock_parent(conn, workspace_id)

    def delete():
        try:
            return db.delete_workspace("ws-delete", 1)
        finally:
            db.close_connection()

    def update():
        try:
            return _update_workspace_chat_settings(db, conversation_id)
        finally:
            db.close_connection()

    monkeypatch.setattr(db, "get_messages_for_conversation", pause_before_children)
    monkeypatch.setattr(db, "_lock_workspace_for_content_write", observe_parent_admission)
    with ThreadPoolExecutor(max_workers=2) as executor:
        deleting = executor.submit(delete)
        try:
            assert parent_claimed.wait(10)
            updating = executor.submit(update)
            assert settings_attempting.wait(10)
            with pytest.raises(FutureTimeout):
                updating.result(timeout=0.2)
        finally:
            finish_delete.set()
        assert deleting.result(timeout=10)
        with pytest.raises(HTTPException) as exc:
            updating.result(timeout=10)
        assert exc.value.status_code == 409
    assert db.get_workspace("ws-delete") is None
    assert db.get_conversation_by_id(conversation_id) is None
    assert db.get_conversation_settings(conversation_id) is None


def test_workspace_chat_restore_rejects_deleted_parent_without_side_effects(deletion_db):
    db = deletion_db
    conversation = _seed_trashed_workspace_chat(db)
    assert db.delete_workspace("ws-delete", 1)
    before = _deletion_snapshot(db)
    with pytest.raises(ConflictError, match="Workspace.*deleted"):
        db.restore_conversation(conversation["id"], conversation["version"])
    assert _deletion_snapshot(db) == before


@pytest.mark.parametrize("parent_exists", [False, True])
@pytest.mark.parametrize("trashed", [False, True])
def test_workspace_chat_restore_rejects_sync_projection_without_active_parent(
    deletion_db, parent_exists, trashed,
):
    db = deletion_db
    db.upsert_workspace("ws-delete", "Workspace")
    if parent_exists:
        assert db.delete_workspace("ws-delete", 1)
    # Sync fencing is a subsequent slice; restore must reject retained projections too.
    assert db.upsert_conversation_from_sync(
        conversation_id="retained-chat", title="Retained chat", sync_client_id=db.client_id,
        object_revision=1, object_hash="retained", scope_type="workspace", workspace_id="ws-delete",
    )
    if trashed:
        assert db.soft_delete_conversation("retained-chat", 1)
    if not parent_exists:
        with db.transaction() as conn:
            conn.execute("DELETE FROM workspaces WHERE id = ?", ("ws-delete",))
        assert db.get_conversation_by_id("retained-chat", include_deleted=True)["workspace_id"] is None
    before = _deletion_snapshot(db)
    with pytest.raises(ConflictError, match="Workspace.*not found|Workspace.*deleted"):
        db.restore_conversation("retained-chat", 2 if trashed else 1)
    assert _deletion_snapshot(db) == before


@pytest.mark.parametrize("deletion_db", ["postgresql"], indirect=True)
@pytest.mark.parametrize("destination_scope", ["workspace", "global"])
def test_workspace_chat_restore_rejects_concurrent_scope_reassignment(
    deletion_db, monkeypatch, destination_scope,
):
    db = deletion_db
    conversation = _seed_trashed_workspace_chat(db)
    db.upsert_workspace("other-workspace", "Other workspace")
    lock_parent = db._lock_workspace_for_content_write
    after_reassignment = None

    def reassign():
        try:
            assert db.upsert_conversation_from_sync(
                conversation_id=conversation["id"], title="Synced chat",
                sync_client_id=db.client_id, object_revision=1, object_hash="scope-change",
                scope_type=destination_scope,
                workspace_id="other-workspace" if destination_scope == "workspace" else None,
            )
            assert db.soft_delete_conversation(conversation["id"], 1)
        finally:
            db.close_connection()

    def reassign_after_parent_lock(conn, workspace_id):
        nonlocal after_reassignment
        lock_parent(conn, workspace_id)
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(reassign).result(timeout=10)
        after_reassignment = _deletion_snapshot(db)

    monkeypatch.setattr(db, "_lock_workspace_for_content_write", reassign_after_parent_lock)
    with pytest.raises(ConflictError, match="scope changed"):
        db.restore_conversation(conversation["id"], conversation["version"])
    assert _deletion_snapshot(db) == after_reassignment


@pytest.mark.parametrize("workspace_scoped", [False, True])
def test_chat_restore_preserves_versions_and_parent_metadata(deletion_db, workspace_scoped):
    db = deletion_db
    data = {"title": "Restore me"}
    if workspace_scoped:
        db.upsert_workspace("ws-delete", "Workspace")
        data.update(scope_type="workspace", workspace_id="ws-delete")
    conversation_id = db.add_conversation(data)
    assert db.soft_delete_conversation(conversation_id, 1)
    before = _deletion_snapshot(db)
    with pytest.raises(ConflictError, match="version mismatch"):
        db.restore_conversation(conversation_id, 1)
    assert _deletion_snapshot(db) == before
    assert db.restore_conversation(conversation_id, 2)
    restored = db.get_conversation_by_id(conversation_id)
    assert restored["version"] == 3
    assert db.get_workspace("ws-delete") == before["workspace"]
    assert db.restore_conversation(conversation_id, 2)
    assert db.get_conversation_by_id(conversation_id) == restored


@pytest.mark.parametrize("deletion_db", ["postgresql"], indirect=True)
def test_global_chat_restore_rejects_move_into_deleted_workspace(deletion_db, monkeypatch):
    db = deletion_db
    db.upsert_workspace("ws-delete", "Workspace")
    assert db.delete_workspace("ws-delete", 1)
    conversation_id = db.add_conversation({"title": "Global chat"})
    assert db.soft_delete_conversation(conversation_id, 1)
    read, proceed = threading.Event(), threading.Event()
    execute = db.backend.execute
    restoring_thread = None

    def pause_after_scope_read(query, *args, **kwargs):
        result = execute(query, *args, **kwargs)
        if (
            threading.get_ident() == restoring_thread
            and "SELECT deleted, version, scope_type, workspace_id" in query
            and "FOR UPDATE" not in query
        ):
            read.set()
            assert proceed.wait(10)
        return result

    def restore():
        nonlocal restoring_thread
        restoring_thread = threading.get_ident()
        try:
            return db.restore_conversation(conversation_id, 2)
        finally:
            db.close_connection()

    # Delay real I/O only; both competing writes still use the production store.
    monkeypatch.setattr(db.backend, "execute", pause_after_scope_read)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(restore)
        try:
            assert read.wait(10)
            assert db.upsert_conversation_from_sync(
                conversation_id=conversation_id, title="Moved chat", sync_client_id=db.client_id,
                object_revision=1, object_hash="moved", scope_type="workspace", workspace_id="ws-delete",
            )
            assert db.soft_delete_conversation(conversation_id, 1)
            after_move = _deletion_snapshot(db)
        finally:
            proceed.set()
        with pytest.raises(ConflictError, match="scope changed"):
            future.result(timeout=10)
    assert _deletion_snapshot(db) == after_move


@pytest.mark.parametrize("commit_delete", [True, False])
def test_workspace_chat_restore_waits_for_deletion_outcome(deletion_db, commit_delete):
    db = deletion_db
    conversation = _seed_trashed_workspace_chat(db)
    attempting = threading.Event()

    def restore():
        try:
            attempting.set()
            return db.restore_conversation(conversation["id"], conversation["version"])
        finally:
            db.close_connection()

    class AbortDeletion(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            with db.transaction():
                assert db.delete_workspace("ws-delete", 1)
                after_delete = _deletion_snapshot(db)
                future = executor.submit(restore)
                assert attempting.wait(10)
                with pytest.raises(FutureTimeout):
                    future.result(timeout=0.2)
                if not commit_delete:
                    raise AbortDeletion
        except AbortDeletion:
            pass
        if commit_delete:
            with pytest.raises(ConflictError):
                future.result(timeout=10)
            assert _deletion_snapshot(db) == after_delete
        else:
            assert future.result(timeout=10)
            assert db.get_conversation_by_id(conversation["id"]) is not None
            assert db.get_workspace("ws-delete") is not None


@pytest.mark.parametrize("commit_restore", [True, False])
def test_workspace_deletion_waits_for_chat_restore_transaction(deletion_db, commit_restore):
    db = deletion_db
    conversation = _seed_trashed_workspace_chat(db)
    attempting = threading.Event()

    def delete():
        try:
            attempting.set()
            return db.delete_workspace("ws-delete", 1)
        finally:
            db.close_connection()

    class AbortRestore(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            with db.transaction():
                assert db.restore_conversation(conversation["id"], conversation["version"])
                future = executor.submit(delete)
                assert attempting.wait(10)
                with pytest.raises(FutureTimeout):
                    future.result(timeout=0.2)
                if not commit_restore:
                    raise AbortRestore
        except AbortRestore:
            pass
        assert future.result(timeout=10)
    assert db.get_workspace("ws-delete") is None
    assert db.get_conversation_by_id(conversation["id"]) is None
    assert db.get_conversation_by_id(conversation["id"], include_deleted=True)["deleted"]


@pytest.mark.parametrize("failure_stage", ["message", "conversation", "quiz", "deck"])
@pytest.mark.parametrize("error_type", [CharactersRAGDBError, sqlite3.OperationalError, BackendDatabaseError])
def test_workspace_deletion_rolls_back_entire_graph(deletion_db, monkeypatch, failure_stage, error_type):
    db = deletion_db
    _seed_deletion_graph(db)
    before = _deletion_snapshot(db)
    if failure_stage in {"message", "conversation"}:
        name = f"soft_delete_{failure_stage}"
        original = getattr(db, name)

        def fail_after_write(*args, **kwargs):
            original(*args, **kwargs)
            raise error_type("injected cascade failure")

        monkeypatch.setattr(db, name, fail_after_write)
    else:
        original_transaction = db.transaction

        @contextmanager
        def failing_transaction():
            with original_transaction() as conn:
                def execute(query, params=()):
                    result = conn.execute(query, params)
                    table = "quizzes" if failure_stage == "quiz" else "decks"
                    if query.startswith(f"UPDATE {table} "):
                        raise error_type("injected cascade failure")
                    return result

                yield SimpleNamespace(execute=execute)

        monkeypatch.setattr(db, "transaction", failing_transaction)

    with pytest.raises(CharactersRAGDBError):
        db.delete_workspace("ws-delete", 1)
    assert _deletion_snapshot(db) == before


@pytest.mark.parametrize("helper", ["soft_delete_message", "soft_delete_conversation"])
def test_workspace_deletion_rejects_unsuccessful_cascade(deletion_db, monkeypatch, helper):
    db = deletion_db
    _seed_deletion_graph(db)
    before = _deletion_snapshot(db)
    original = getattr(db, helper)

    def unsuccessful(*args, **kwargs):
        original(*args, **kwargs)
        return False

    monkeypatch.setattr(db, helper, unsuccessful)
    with pytest.raises(CharactersRAGDBError):
        db.delete_workspace("ws-delete", 1)
    assert _deletion_snapshot(db) == before


def test_workspace_deletion_tombstone_precedes_children_and_preserves_helpers(deletion_db, monkeypatch):
    db = deletion_db
    conversation_id, message_ids, quiz_id, deck_id = _seed_deletion_graph(db, message_count=101)
    before_history = db.get_conversation_by_id(conversation_id)["history_version"]
    before_sync = db.get_latest_sync_log_change_id()
    original = db.soft_delete_message

    def check_parent(*args, **kwargs):
        parent = db.get_workspace("ws-delete", include_deleted=True)
        assert parent["deleted"] and parent["version"] == 2
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "soft_delete_message", check_parent)
    assert db.delete_workspace("ws-delete", 1) is True
    conversation = dict(db.execute_query("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone())
    assert conversation["deleted"]
    assert conversation["history_version"] == before_history + len(message_ids)
    assert db.get_messages_for_conversation(conversation_id) == []
    assert db.search_messages_by_content("deletionneedle") == []
    assert db.get_quiz(quiz_id)["workspace_id"] is None
    assert db.get_deck(deck_id)["workspace_id"] is None
    entries = db.get_sync_log_entries(since_change_id=before_sync, limit=1000)
    assert set(message_ids) <= {entry["entity_id"] for entry in entries if entry["entity"] == "messages"}


def test_workspace_deletion_stale_version_has_no_side_effects(deletion_db):
    _seed_deletion_graph(deletion_db)
    deletion_db.update_workspace("ws-delete", {"name": "Changed"}, 1)
    before = _deletion_snapshot(deletion_db)
    with pytest.raises(ConflictError):
        deletion_db.delete_workspace("ws-delete", 1)
    assert _deletion_snapshot(deletion_db) == before


@pytest.mark.parametrize("deletion_db", ["postgresql"], indirect=True)
@pytest.mark.parametrize("entity_column", ["entity_id", "entity_uuid"])
def test_message_deletion_postgres_sync_is_idempotent(deletion_db, entity_column, monkeypatch):
    db = deletion_db
    _, message_ids, _, _ = _seed_deletion_graph(db)
    if entity_column == "entity_uuid":
        with db.transaction() as conn:
            conn.execute("ALTER TABLE sync_log RENAME COLUMN entity_id TO entity_uuid")
    before_sync = db.get_latest_sync_log_change_id()
    with pytest.raises(ConflictError):
        db.soft_delete_message(message_ids[0], 99)
    assert db.get_sync_log_entries(since_change_id=before_sync, entity_type="messages") == []

    timestamp = "2026-09-20T12:00:00.000Z"
    monkeypatch.setattr(db, "_get_current_utc_timestamp_iso", lambda: timestamp)
    assert db.soft_delete_message(message_ids[0], 1) is True
    assert db.soft_delete_message(message_ids[0], 1) is True
    entries = db.get_sync_log_entries(since_change_id=before_sync, entity_type="messages")
    assert len(entries) == 1
    entry = entries[0]
    assert entry[entity_column] == message_ids[0]
    assert entry["operation"] == "delete"
    assert entry["version"] == 2
    assert entry["client_id"] == db.client_id
    assert entry["payload"] == {
        "id": message_ids[0], "deleted": 1, "version": 2,
        "last_modified": timestamp, "client_id": db.client_id,
    }
    deleted = db.execute_query("SELECT last_modified FROM messages WHERE id = ?", (message_ids[0],)).fetchone()
    assert entry["timestamp"] == deleted["last_modified"]


@pytest.mark.parametrize("deletion_db", ["postgresql"], indirect=True)
@pytest.mark.parametrize("entity_column", ["entity_id", "entity_uuid"])
def test_message_deletion_postgres_sync_rolls_back_with_message(deletion_db, entity_column):
    db = deletion_db
    _, message_ids, _, _ = _seed_deletion_graph(db)
    if entity_column == "entity_uuid":
        with db.transaction() as conn:
            conn.execute("ALTER TABLE sync_log RENAME COLUMN entity_id TO entity_uuid")
    before = _deletion_snapshot(db)
    with pytest.raises(RuntimeError, match="abort deletion"):
        with db.transaction() as conn:
            assert db.soft_delete_message(message_ids[0], 1, conn=conn) is True
            assert len(db.get_sync_log_entries(entity_type="messages")) == 1
            raise RuntimeError("abort deletion")
    assert _deletion_snapshot(db) == before


@pytest.mark.parametrize("deletion_db", ["postgresql"], indirect=True)
@pytest.mark.parametrize("failure", ["columns", "introspection", "insert"])
def test_workspace_deletion_rolls_back_on_postgres_message_sync_failure(deletion_db, monkeypatch, failure):
    db = deletion_db
    _seed_deletion_graph(db)
    before = _deletion_snapshot(db)
    if failure in {"columns", "introspection"}:
        original = db.backend.get_table_info

        def table_info(table, **kwargs):
            if table != "sync_log":
                return original(table, **kwargs)
            if failure == "introspection":
                raise BackendDatabaseError("injected sync metadata failure")
            return [{"name": "unsupported_entity_column"}]

        monkeypatch.setattr(db.backend, "get_table_info", table_info)
    else:
        original = db.backend.execute

        def execute(query, *args, **kwargs):
            result = original(query, *args, **kwargs)
            if query.startswith("INSERT INTO sync_log"):
                raise BackendDatabaseError("injected sync insert failure")
            return result

        monkeypatch.setattr(db.backend, "execute", execute)
    with pytest.raises(CharactersRAGDBError):
        db.delete_workspace("ws-delete", 1)
    assert _deletion_snapshot(db) == before


@pytest.fixture
def deletion_client(workspace_fastapi_app, db, monkeypatch):
    async def allow():
        return None

    async def cleanup(*args):
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(kind="user", user_id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = allow
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = allow
    monkeypatch.setattr(workspaces_endpoint, "on_workspace_deleted", cleanup)
    with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
        yield client


def test_workspace_deletion_uses_supplied_version_without_refresh(deletion_client, db):
    db.upsert_workspace("ws-delete", "Private")
    db.update_workspace("ws-delete", {"name": "Changed"}, 1)
    response = deletion_client.delete("/api/v1/workspaces/ws-delete?expected_version=1")
    assert response.status_code == 409, response.text
    assert db.get_workspace("ws-delete")["version"] == 2


@pytest.mark.parametrize(("resource", "payload"), [
    ("sources", {"id": "source", "media_id": 1, "title": "Source", "source_type": "pdf"}),
    ("artifacts", {"id": "artifact", "artifact_type": "summary", "title": "Summary"}),
    ("notes", {"title": "Note", "content": "Late result"}),
])
def test_content_api_conflicts_when_deleted_after_preflight(
    deletion_client, workspace_fastapi_app, db, monkeypatch, resource, payload,
):
    db.upsert_workspace("ws-delete", "Workspace")
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = lambda: None
    workspace_fastapi_app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: None
    require_workspace = workspaces_endpoint._require_workspace
    enqueue = MagicMock()

    def delete_after_preflight(database, workspace_id):
        workspace = require_workspace(database, workspace_id)
        assert database.delete_workspace(workspace_id, workspace["version"])
        return workspace

    monkeypatch.setattr(workspaces_endpoint, "_require_workspace", delete_after_preflight)
    monkeypatch.setattr(workspaces_endpoint, "_enqueue_workspace_source_ingest_job", enqueue)
    response = deletion_client.post(f"/api/v1/workspaces/ws-delete/{resource}", json=payload)
    assert response.status_code == 409, response.text
    assert all(not rows for rows in _workspace_content_snapshot(db).values())
    enqueue.assert_not_called()


def test_content_fence_preserves_sqlite_device_client_semantics(db):
    db.upsert_workspace("ws-delete", "Workspace")
    before = db.get_workspace("ws-delete")
    original_client = db.client_id
    try:
        db.client_id = "another-device"
        _seed_workspace_content(db)
    finally:
        db.client_id = original_client
    assert db.get_workspace("ws-delete") == before


@pytest.mark.parametrize("state", ["missing", "deleted"])
@pytest.mark.parametrize("query", ["", "?expected_version=1"])
def test_workspace_deletion_missing_or_deleted_is_404(deletion_client, db, state, query):
    if state == "deleted":
        db.upsert_workspace("ws-delete", "Private")
        db.delete_workspace("ws-delete", 1)
    response = deletion_client.delete("/api/v1/workspaces/ws-delete" + query)
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("version", ["0", "-1", "invalid", "2147483647", "2147483648", "9" * 100])
def test_workspace_deletion_rejects_invalid_version(deletion_client, db, version):
    db.upsert_workspace("ws-delete", "Private")
    response = deletion_client.delete(f"/api/v1/workspaces/ws-delete?expected_version={version}")
    assert response.status_code == 422
    assert db.get_workspace("ws-delete") is not None


def test_workspace_deletion_accepts_largest_incrementable_version(deletion_client, db):
    db.upsert_workspace("ws-delete", "Private")
    response = deletion_client.delete("/api/v1/workspaces/ws-delete?expected_version=2147483646")
    assert response.status_code == 409
    assert db.get_workspace("ws-delete")["version"] == 1


def test_workspace_deletion_status_db_failure_is_not_cached(deletion_client, db, monkeypatch):
    def unavailable(*args, **kwargs):
        raise CharactersRAGDBError("private storage error")

    monkeypatch.setattr(db, "get_workspace", unavailable)
    response = deletion_client.get("/api/v1/workspaces/ws-delete/deletion-status")
    assert response.status_code == 500
    assert response.headers["cache-control"] == "no-store"
    assert "private storage error" not in response.text


@pytest.mark.parametrize("method,path", [
    ("delete", "/ws-delete?expected_version=1"),
    ("get", "/ws-delete/deletion-status"),
])
def test_workspace_deletion_account_guard_precedes_db(deletion_client, workspace_fastapi_app, method, path):
    calls = []

    def unexpected_db():
        calls.append(True)
        raise AssertionError("DB must not resolve for a different account")

    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = unexpected_db
    response = getattr(deletion_client, method)(
        "/api/v1/workspaces" + path, headers={"X-TLDW-Expected-User-ID": "2"},
    )
    assert response.status_code == 412, response.text
    assert response.headers["cache-control"] == "no-store"
    assert calls == []


@pytest.mark.parametrize("state", ["active", "deleted", "missing"])
def test_workspace_deletion_status_is_minimal_and_no_store(deletion_client, db, state):
    if state != "missing":
        db.upsert_workspace("ws-delete", "Private name")
    if state == "deleted":
        db.delete_workspace("ws-delete", 1)
    response = deletion_client.get("/api/v1/workspaces/ws-delete/deletion-status")
    assert response.status_code == (404 if state == "missing" else 200), response.text
    assert response.headers["cache-control"] == "no-store"
    if state != "missing":
        assert response.json() == {"workspace_id": "ws-delete", "deleted": state == "deleted", "version": 2 if state == "deleted" else 1}


@pytest.mark.parametrize("method,path", [
    ("delete", "/ws-delete"),
    ("delete", "/ws-delete?expected_version=1"),
    ("get", "/ws-delete/deletion-status"),
])
def test_workspace_deletion_db_operation_runs_in_worker(deletion_client, db, monkeypatch, method, path):
    db.upsert_workspace("ws-delete", "Private")
    worker_threads = []
    operation_threads = []
    original_pool = workspaces_endpoint.run_in_threadpool
    original_connection = db._get_thread_connection

    async def record_pool(func, *args, **kwargs):
        def operation():
            worker_threads.append(threading.get_ident())
            return func(*args, **kwargs)
        return await original_pool(operation)

    def record_connection():
        operation_threads.append(threading.get_ident())
        return original_connection()

    monkeypatch.setattr(workspaces_endpoint, "run_in_threadpool", record_pool)
    monkeypatch.setattr(db, "_get_thread_connection", record_connection)
    response = getattr(deletion_client, method)("/api/v1/workspaces" + path)
    assert response.status_code == (204 if method == "delete" else 200), response.text
    assert len(worker_threads) == 1
    assert operation_threads and set(operation_threads) == set(worker_threads)


@pytest.mark.parametrize("method,failure", [
    ("delete", None), ("get", None), ("delete", "read"),
    ("get", "read"), ("delete", "cascade"),
])
def test_workspace_deletion_worker_releases_connection(deletion_client, db, monkeypatch, method, failure):
    db.upsert_workspace("ws-delete", "Private")
    original_pool = workspaces_endpoint.run_in_threadpool
    original_read = db.get_workspace
    original_close = db.close_connection
    worker_threads = []
    closed_threads = []
    retained = []
    connections = []

    def read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        connections.append(db.get_connection())
        if failure == "read":
            raise CharactersRAGDBError("read failed")
        return result

    def fail_cascade(*args, **kwargs):
        raise CharactersRAGDBError("cascade failed")

    def close():
        closed_threads.append(threading.get_ident())
        original_close()

    async def pool(func, *args, **kwargs):
        def operation():
            worker_threads.append(threading.get_ident())
            try:
                return func(*args, **kwargs)
            finally:
                retained.append(getattr(db._local, "conn", None))
        return await original_pool(operation)

    monkeypatch.setattr(db, "get_workspace", read)
    monkeypatch.setattr(db, "close_connection", close)
    monkeypatch.setattr(workspaces_endpoint, "run_in_threadpool", pool)
    if failure == "cascade":
        monkeypatch.setattr(db, "delete_workspace", fail_cascade)
    suffix = "?expected_version=1" if method == "delete" else "/deletion-status"
    response = getattr(deletion_client, method)("/api/v1/workspaces/ws-delete" + suffix)
    assert response.status_code == (500 if failure else 204 if method == "delete" else 200)
    assert retained == [None]
    assert closed_threads == worker_threads
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["delete", "get"])
async def test_workspace_deletion_worker_preserves_request_owner(deletion_db, monkeypatch, method):
    db = deletion_db
    db.upsert_workspace("ws-delete", "Private")
    db.close_connection()

    async def cleanup(*args, **kwargs):
        return None

    monkeypatch.setattr(workspaces_endpoint, "on_workspace_deleted", cleanup)
    with chacha_operation(independent=True):
        connection = db.get_connection()
        if method == "delete":
            await workspaces_endpoint.delete_workspace(
                "ws-delete", expected_version=1, db=db, current_user=SimpleNamespace(id=1),
            )
        else:
            await workspaces_endpoint.get_workspace_deletion_status(
                "ws-delete", response=FastAPIResponse(), db=db, current_user=SimpleNamespace(id=1),
            )
        # Later dependency/response work still belongs to the same request owner.
        workspace = db.get_workspace("ws-delete", include_deleted=True)
        assert bool(workspace["deleted"]) is (method == "delete")
        if db.backend_type == BackendType.POSTGRESQL:
            assert db.get_connection()._connection is connection._connection


@pytest.mark.asyncio
@pytest.mark.parametrize("deletion_db", ["postgresql"], indirect=True)
@pytest.mark.parametrize("method", ["delete", "get"])
@pytest.mark.parametrize("failure", [False, True])
async def test_workspace_deletion_cancelled_owner_preserves_worker_outcome(
    deletion_db, monkeypatch, method, failure,
):
    db = deletion_db
    db.upsert_workspace("ws-delete", "Private")
    db.close_connection()
    backend = db.backend
    pool = backend.get_pool()
    original_execute = backend.execute
    original_return = pool.return_connection
    original_pool = workspaces_endpoint.run_in_threadpool
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    checkouts, returns, outcomes = [], [], []

    def execute(query, *args, **kwargs):
        target = "UPDATE workspaces" if method == "delete" else "SELECT * FROM workspaces"
        if target not in query:
            return original_execute(query, *args, **kwargs)
        checkouts.append(kwargs["connection"])
        started.set()
        assert release.wait(10), "test did not release worker"
        if failure:
            raise BackendDatabaseError("injected worker failure")
        return original_execute(query, *args, **kwargs)

    def return_connection(raw):
        returns.append(raw)
        return original_return(raw)

    async def record_worker(func, *args, **kwargs):
        def run():
            try:
                result = func(*args, **kwargs)
                outcomes.append(result)
                return result
            except BaseException as exc:
                outcomes.append(exc)
                raise
            finally:
                finished.set()
        return await original_pool(run)

    async def request():
        with chacha_operation(independent=True):
            if method == "delete":
                await workspaces_endpoint.delete_workspace(
                    "ws-delete", expected_version=1, db=db, current_user=SimpleNamespace(id=1),
                )
            else:
                await workspaces_endpoint.get_workspace_deletion_status(
                    "ws-delete", response=FastAPIResponse(), db=db, current_user=SimpleNamespace(id=1),
                )

    monkeypatch.setattr(backend, "execute", execute)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    monkeypatch.setattr(workspaces_endpoint, "run_in_threadpool", record_worker)
    task = asyncio.create_task(request())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert returns == [], "request returned a checkout still used by its worker"
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        if not task.done():
            await task
    assert returns == checkouts
    assert len(outcomes) == 1
    if failure:
        assert isinstance(outcomes[0], CharactersRAGDBError), repr(outcomes[0])
    else:
        assert not isinstance(outcomes[0], BaseException), repr(outcomes[0])
    assert checkouts[0].info.transaction_status.name == "IDLE"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["delete", "get"])
async def test_workspace_deletion_worker_cancellation_defers_release(db, monkeypatch, method):
    db.upsert_workspace("ws-delete", "Private")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    closed = threading.Event()
    events = []
    original_read = db.get_workspace
    original_close = db.close_connection
    original_pool = workspaces_endpoint.run_in_threadpool

    def blocked_read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        events.append(("acquired", threading.get_ident()))
        started.set()
        if not release.wait(10):
            raise AssertionError("test did not release worker")
        events.append(("read-finished", threading.get_ident()))
        return result

    def close():
        original_close()
        events.append(("closed", threading.get_ident()))
        closed.set()

    async def pool(func, *args, **kwargs):
        def operation():
            try:
                return func(*args, **kwargs)
            finally:
                finished.set()
        return await original_pool(operation)

    monkeypatch.setattr(db, "get_workspace", blocked_read)
    monkeypatch.setattr(db, "close_connection", close)
    monkeypatch.setattr(workspaces_endpoint, "run_in_threadpool", pool)
    if method == "delete":
        operation = workspaces_endpoint.delete_workspace(
            "ws-delete", expected_version=1, db=db, current_user=SimpleNamespace(id=1),
        )
    else:
        operation = workspaces_endpoint.get_workspace_deletion_status(
            "ws-delete", response=FastAPIResponse(), db=db, current_user=SimpleNamespace(id=1),
        )
    task = asyncio.create_task(operation)
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not closed.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        if not task.done():
            await task
    assert closed.is_set()
    assert [event for event, _ in events] == ["acquired", "read-finished", "closed"]
    assert len({thread for _, thread in events}) == 1


@pytest.mark.parametrize("dependency,code,headers", [
    (get_auth_principal, 401, {"WWW-Authenticate": "Bearer"}),
    (WORKSPACES_READ_RATE_LIMIT, 429, {"Retry-After": "17"}),
    (get_chacha_db_for_user, 503, {"Retry-After": "3"}),
])
def test_workspace_deletion_status_dependency_errors_are_no_store(
    deletion_client, workspace_fastapi_app, dependency, code, headers,
):
    def reject():
        raise StarletteHTTPException(code, detail={"reason": "unavailable"}, headers=headers)

    workspace_fastapi_app.dependency_overrides[dependency] = reject
    response = deletion_client.get("/api/v1/workspaces/ws-delete/deletion-status")
    assert response.status_code == code
    assert response.json() == {"detail": {"reason": "unavailable"}}
    assert response.headers["cache-control"] == "no-store"
    for key, value in headers.items():
        assert response.headers[key] == value
    ordinary = deletion_client.get("/api/v1/workspaces/ws-delete")
    assert ordinary.status_code == code
    assert "cache-control" not in ordinary.headers


def test_workspace_deletion_status_preserves_dependency_validation(deletion_client, workspace_fastapi_app):
    def validated(limit: int = Query(ge=1)):
        return None

    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = validated
    status_response = deletion_client.get("/api/v1/workspaces/ws-delete/deletion-status?limit=invalid")
    ordinary = deletion_client.get("/api/v1/workspaces/ws-delete?limit=invalid")
    assert status_response.status_code == ordinary.status_code == 422
    assert status_response.json() == ordinary.json()
    assert status_response.headers["cache-control"] == "no-store"
    assert "cache-control" not in ordinary.headers


@pytest.mark.parametrize("sync_handler", [False, True])
@pytest.mark.parametrize("status_override", [False, True])
def test_workspace_deletion_status_preserves_registered_http_handlers(
    workspace_fastapi_app, sync_handler, status_override,
):
    def reject():
        raise StarletteHTTPException(401, detail="Authentication required")

    def class_handler(request, exc):
        return JSONResponse({"handler": "class"}, status_code=exc.status_code, headers={"WWW-Authenticate": "Bearer"})

    def status_handler(request, exc):
        return JSONResponse({"handler": "status"}, status_code=exc.status_code, headers={"WWW-Authenticate": "Custom"})

    async def async_class_handler(request, exc):
        return class_handler(request, exc)

    async def async_status_handler(request, exc):
        return status_handler(request, exc)

    workspace_fastapi_app.dependency_overrides[get_auth_principal] = reject
    workspace_fastapi_app.add_exception_handler(
        StarletteHTTPException, class_handler if sync_handler else async_class_handler,
    )
    if status_override:
        workspace_fastapi_app.add_exception_handler(401, status_handler if sync_handler else async_status_handler)
    with TestClient(workspace_fastapi_app) as client:
        status_response = client.get("/api/v1/workspaces/ws-delete/deletion-status")
        ordinary = client.get("/api/v1/workspaces/ws-delete")
    assert status_response.status_code == ordinary.status_code == 401
    assert status_response.json() == ordinary.json() == {"handler": "status" if status_override else "class"}
    assert status_response.headers["www-authenticate"] == ordinary.headers["www-authenticate"]
    assert status_response.headers["cache-control"] == "no-store"
    assert "cache-control" not in ordinary.headers


def test_workspace_deletion_status_preserves_registered_validation_handler(workspace_fastapi_app, monkeypatch):
    from tldw_Server_API.app import main

    def validated(limit: int = Query(ge=1)):
        return None

    def cors_headers(request, response):
        response.headers["Access-Control-Allow-Origin"] = "https://workspace.example"
        return response

    monkeypatch.setattr(main, "_apply_runtime_cors_headers", cors_headers)
    workspace_fastapi_app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(kind="user", user_id=1)
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: None
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = validated
    workspace_fastapi_app.add_exception_handler(
        RequestValidationError, main._standalone_request_validation_exception_handler,
    )
    with TestClient(workspace_fastapi_app) as client:
        status_response = client.get("/api/v1/workspaces/ws-delete/deletion-status?limit=invalid")
        ordinary = client.get("/api/v1/workspaces/ws-delete?limit=invalid")
    assert status_response.status_code == ordinary.status_code == 422
    assert status_response.json() == ordinary.json()
    assert status_response.headers["access-control-allow-origin"] == ordinary.headers["access-control-allow-origin"]
    assert status_response.headers["cache-control"] == "no-store"


@pytest.fixture
def other_owner_db(tmp_path):
    database = CharactersRAGDB(str(tmp_path / "other-owner.db"), client_id="user-2")
    try:
        yield database
    finally:
        database.close_all_connections()


def test_workspace_deletion_status_cannot_read_other_owner_tombstone(
    deletion_client, workspace_fastapi_app, db, other_owner_db,
):
    db.upsert_workspace("ws-delete", "Owner A private name")
    db.delete_workspace("ws-delete", 1)
    owner_a = deletion_client.get(
        "/api/v1/workspaces/ws-delete/deletion-status", headers={"X-TLDW-Expected-User-ID": "1"},
    )
    assert owner_a.json() == {"workspace_id": "ws-delete", "deleted": True, "version": 2}
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=2)
    workspace_fastapi_app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(kind="user", user_id=2)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: other_owner_db
    owner_b = deletion_client.get(
        "/api/v1/workspaces/ws-delete/deletion-status", headers={"X-TLDW-Expected-User-ID": "2"},
    )
    assert owner_b.status_code == 404
    assert owner_b.headers["cache-control"] == "no-store"
    assert "Owner A" not in owner_b.text


class _CapturingJobManager:
    def __init__(self) -> None:
        self.created_jobs: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        self.created_jobs.append(kwargs)
        return {"id": len(self.created_jobs), **kwargs}


class _FailingJobManager:
    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        raise RuntimeError("jobs backend unavailable")


class _ConflictingSandboxVolumeService:
    def provision_workspace_volume(self, **kwargs: Any) -> None:
        _ = kwargs
        raise IdempotencyConflict("volume-previous", key="workspace-root:previous")


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.add_character_card({"name": "Test Char"})
    return d


@pytest.fixture
def workspace_fastapi_app() -> FastAPI:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    return app


def _create_workspace_test_persona(
    db: CharactersRAGDB,
    *,
    persona_id: str = "persona-1",
    user_id: str = "1",
    name: str = "Workspace Persona",
) -> str:
    """Create a test Persona profile for workspace endpoint tests.

    Args:
        db: CharactersRAGDB test database that already contains "Test Char".
        persona_id: Stable Persona id to write into the profile row.
        user_id: Owner user id for permission-scoped Persona lookups.
        name: Display name to store on the Persona profile.

    Returns:
        The Persona id returned by ``db.create_persona_profile``.

    Side Effects:
        Asserts that the "Test Char" character card exists, then creates a
        session-scoped Persona profile linked to that character.
    """
    character = db.get_character_card_by_name("Test Char")
    assert character is not None  # nosec B101
    return db.create_persona_profile(
        {
            "id": persona_id,
            "user_id": user_id,
            "name": name,
            "character_card_id": int(character["id"]),
            "mode": "session_scoped",
            "system_prompt": "You support workspace tests.",
            "is_active": True,
        }
    )


def _get_workspace_roots_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    workspace_id: str = "ws-root",
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.get(f"/api/v1/workspaces/{workspace_id}/roots")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)


def _get_workspace_capabilities_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    workspace_id: str = "ws-root",
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[try_get_media_db_for_user] = lambda: None
    workspace_fastapi_app.dependency_overrides[
        workspaces_endpoint.try_get_workspace_job_manager
    ] = lambda: None
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.get(f"/api/v1/workspaces/{workspace_id}/capabilities")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(try_get_media_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(
            workspaces_endpoint.try_get_workspace_job_manager,
            None,
        )
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)


def _get_workspace_context_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    workspace_id: str = "ws-root",
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[try_get_media_db_for_user] = lambda: None
    workspace_fastapi_app.dependency_overrides[
        workspaces_endpoint.try_get_workspace_job_manager
    ] = lambda: None
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.get(f"/api/v1/workspaces/{workspace_id}/context")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(try_get_media_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(
            workspaces_endpoint.try_get_workspace_job_manager,
            None,
        )
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)


def _put_workspace_primary_root_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    payload: dict[str, Any],
    workspace_id: str = "ws-root",
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.put(
                f"/api/v1/workspaces/{workspace_id}/roots/primary",
                json=payload,
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


def _post_workspace_sandbox_root_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    payload: dict[str, Any],
    workspace_id: str = "ws-root",
    idempotency_key: str | None = "root-key",
    sandbox_service: Any | None = None,
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    sandbox_service = sandbox_service or SandboxWorkspaceVolumeService(store=InMemoryStore())
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[
        workspaces_endpoint.get_workspace_sandbox_volume_service
    ] = lambda: sandbox_service
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key is not None else {}
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.post(
                f"/api/v1/workspaces/{workspace_id}/roots/primary/sandbox-volume",
                json=payload,
                headers=headers,
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(
            workspaces_endpoint.get_workspace_sandbox_volume_service,
            None,
        )
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


def _get_workspace_operation_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    workspace_id: str,
    operation_id: str,
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.get(f"/api/v1/workspaces/{workspace_id}/operations/{operation_id}")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)


def test_root_path_hint_redacts_relative_path_segments() -> None:
    assert workspaces_endpoint._root_path_hint({"path_hint": "client/acme/repo"}) == "repo"
    assert workspaces_endpoint._root_path_hint({"display_name": "client\\acme\\repo"}) == "repo"


class TestWorkspaceLifecycle:
    def test_upsert_then_get(self, db):
        ws = db.upsert_workspace("ws-1", "My Workspace", study_materials_policy="workspace")
        assert ws["id"] == "ws-1"
        assert ws["study_materials_policy"] == "workspace"
        fetched = db.get_workspace("ws-1")
        assert fetched["name"] == "My Workspace"
        assert fetched["study_materials_policy"] == "workspace"

    def test_upsert_workspace_updates_existing_policy(self, db):
        original = db.upsert_workspace("ws-1", "Original Name", study_materials_policy="general")
        updated = db.upsert_workspace("ws-1", "Renamed Workspace", study_materials_policy="workspace")
        assert updated["id"] == original["id"]
        assert updated["name"] == "Renamed Workspace"
        assert updated["study_materials_policy"] == "workspace"
        assert updated["version"] == original["version"] + 1

    def test_patch_workspace_name(self, db):
        db.upsert_workspace("ws-1", "Old")
        ws = db.update_workspace("ws-1", {"name": "New"}, expected_version=1)
        assert ws["name"] == "New"
        assert ws["version"] == 2

    def test_archive_workspace(self, db):
        db.upsert_workspace("ws-1", "WS")
        ws = db.update_workspace("ws-1", {"archived": True}, expected_version=1)
        assert ws["archived"] in (True, 1)

    def test_delete_workspace_cascade(self, db):
        db.upsert_workspace("ws-1", "WS")
        conv_id = db.add_conversation({
            "title": "WS chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-1",
        })
        quiz_id = db.create_quiz(name="Workspace Quiz", workspace_id="ws-1")
        deck_id = db.add_deck("Workspace Deck", workspace_id="ws-1")
        db.delete_workspace("ws-1", expected_version=1)

        # Workspace is soft-deleted
        ws = db.get_workspace("ws-1")
        assert ws is None  # get_workspace excludes deleted

        # Conversation is also soft-deleted
        conv = db.get_conversation_by_id(conv_id)
        assert conv is None

        quiz = db.get_quiz(quiz_id)
        deck = db.get_deck(deck_id)
        assert quiz is not None
        assert deck is not None
        assert quiz["workspace_id"] is None
        assert deck["workspace_id"] is None

    def test_list_workspaces(self, db):
        for i in range(5):
            db.upsert_workspace(f"ws-{i}", f"WS {i}")
        result = db.list_workspaces()
        assert len(result) == 5

    def test_version_conflict_returns_error(self, db):
        db.upsert_workspace("ws-1", "WS")
        db.update_workspace("ws-1", {"name": "V2"}, expected_version=1)
        with pytest.raises((ConflictError, Exception)):
            db.update_workspace("ws-1", {"name": "V3"}, expected_version=1)

    def test_workspace_policy_updates(self, db):
        db.upsert_workspace("ws-1", "WS")
        ws = db.update_workspace("ws-1", {"study_materials_policy": "workspace"}, expected_version=1)
        assert ws["study_materials_policy"] == "workspace"


@pytest.mark.integration
def test_workspace_api_accepts_and_returns_study_materials_policy(workspace_fastapi_app, db):
    from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
        WORKSPACES_READ_RATE_LIMIT,
        WORKSPACES_WRITE_RATE_LIMIT,
    )
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            create_response = client.put(
                "/api/v1/workspaces/ws-api",
                json={
                    "name": "API Workspace",
                    "study_materials_policy": "workspace",
                    "workspace_profile": "project",
                },
            )
            assert create_response.status_code == 200, create_response.text
            created = create_response.json()
            assert created["study_materials_policy"] == "workspace"
            assert created["workspace_profile"] == "project"

            upsert_response = client.put(
                "/api/v1/workspaces/ws-api",
                json={
                    "name": "API Workspace Renamed",
                    "study_materials_policy": "general",
                },
            )
            assert upsert_response.status_code == 200, upsert_response.text
            upserted = upsert_response.json()
            assert upserted["name"] == "API Workspace Renamed"
            assert upserted["study_materials_policy"] == "general"
            assert upserted["workspace_profile"] == "project"

            patch_response = client.patch(
                f"/api/v1/workspaces/{created['id']}",
                json={"study_materials_policy": "workspace", "version": upserted["version"]},
            )
            assert patch_response.status_code == 200, patch_response.text
            patched = patch_response.json()
            assert patched["study_materials_policy"] == "workspace"
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_delete_workspace_invokes_sharing_cleanup_hook(
    workspace_fastapi_app,
    db,
    monkeypatch,
):
    async def _allow_rate_limit() -> None:
        return None

    cleanup_calls: list[tuple[str, int]] = []

    async def _record_cleanup(workspace_id: str, owner_user_id: int) -> None:
        cleanup_calls.append((workspace_id, owner_user_id))

    db.upsert_workspace("ws-delete-hook", "Delete Hook")
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    monkeypatch.setattr(
        workspaces_endpoint,
        "on_workspace_deleted",
        _record_cleanup,
        raising=False,
    )
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-delete-hook")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 204, response.text
    assert cleanup_calls == [("ws-delete-hook", 1)]


@pytest.mark.integration
def test_delete_workspace_cleanup_failure_log_includes_context(
    workspace_fastapi_app,
    db,
    monkeypatch,
):
    async def _allow_rate_limit() -> None:
        return None

    async def _fail_cleanup(workspace_id: str, owner_user_id: int) -> None:
        raise RuntimeError("cleanup backend unavailable")

    fake_logger = MagicMock()
    db.upsert_workspace("ws-delete-log", "Delete Log")
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    monkeypatch.setattr(workspaces_endpoint, "on_workspace_deleted", _fail_cleanup)
    monkeypatch.setattr(workspaces_endpoint, "logger", fake_logger)
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-delete-log")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 204, response.text
    fake_logger.warning.assert_called_once()
    assert fake_logger.warning.call_args.args == (
        "Workspace sharing cleanup hook failed after workspace deletion; "
        "workspace_id={} owner_user_id={}",
        "ws-delete-log",
        1,
    )


@pytest.mark.integration
def test_workspace_api_patches_assistant_defaults(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    _create_workspace_test_persona(db)
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": {
                        "assistant_kind": "persona",
                        "assistant_id": "persona-1",
                        "persona_memory_mode": "read_only",
                    },
                },
            )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["assistant_defaults"]["assistant_kind"] == "persona"
        assert payload["assistant_defaults"]["assistant_id"] == "persona-1"
        assert payload["assistant_defaults"]["persona_memory_mode"] == "read_only"
        assert payload["assistant_defaults"]["voice"] is None
        assert payload["assistant_defaults"]["style"] is None
        assert payload["assistant_defaults"]["tool_policy_profile_id"] is None
        persisted = db.get_workspace("ws-assistant")
        assert persisted is not None
        assert persisted["assistant_defaults_json"] == {
            "assistant_kind": "persona",
            "assistant_id": "persona-1",
            "persona_memory_mode": "read_only",
        }
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_api_requires_confirmation_for_read_write_assistant_default(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    _create_workspace_test_persona(db)
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            missing_confirmation = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": {
                        "assistant_kind": "persona",
                        "assistant_id": "persona-1",
                        "persona_memory_mode": "read_write",
                    },
                },
            )
            assert missing_confirmation.status_code == 422, missing_confirmation.text

            confirmed = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": {
                        "assistant_kind": "persona",
                        "assistant_id": "persona-1",
                        "persona_memory_mode": "read_write",
                    },
                    "confirm_read_write_assistant_default": True,
                },
            )
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["assistant_defaults"]["persona_memory_mode"] == "read_write"
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_api_rejects_confirmation_only_patch(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "confirm_read_write_assistant_default": True,
                },
            )
        assert response.status_code == 422, response.text
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_root_endpoints_happy_path(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            upsert_response = client.put(
                "/api/v1/workspaces/ws-root-api",
                json={"name": "Root Workspace", "study_materials_policy": "workspace"},
            )
            assert upsert_response.status_code == 200, upsert_response.text
            upserted = upsert_response.json()
            assert upserted["id"] == "ws-root-api"
            assert upserted["name"] == "Root Workspace"

            get_response = client.get("/api/v1/workspaces/ws-root-api")
            assert get_response.status_code == 200, get_response.text
            fetched = get_response.json()
            assert fetched["id"] == "ws-root-api"
            assert fetched["study_materials_policy"] == "workspace"

            list_response = client.get("/api/v1/workspaces/")
            assert list_response.status_code == 200, list_response.text
            payload = list_response.json()
            assert payload["total"] == 1
            assert payload["items"][0]["id"] == "ws-root-api"
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_roots_endpoint_returns_primary_root_contract(workspace_fastapi_app, db):
    async def _allow_rate_limit() -> None:
        return None

    db.upsert_workspace("ws-root", "Rooted Workspace")
    db.upsert_workspace_primary_root(
        "ws-root",
        {
            "root_id": "root-1",
            "backend": "host_local",
            "display_name": "Local root",
            "absolute_root": "/Users/example/project",
            "root_state": "attached",
            "indexing_state": "ready",
        },
    )

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-root/roots")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-root"
    assert payload["workspace_profile"] == "project"
    assert payload["primary_root"]["root_id"] == "root-1"
    assert payload["primary_root"]["backend"] == "host_local"
    assert payload["primary_root"]["state"] == "attached"
    assert payload["primary_root"]["path_hint"] == "Local root"
    assert "absolute_root" not in payload["primary_root"]
    assert [root["root_id"] for root in payload["roots"]] == ["root-1"]


@pytest.mark.integration
def test_list_workspace_roots_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "workspace_profile": "project"}

        def list_workspace_project_roots(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-root/roots")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace roots"


@pytest.mark.integration
def test_list_workspace_roots_maps_workspace_lookup_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

        def list_workspace_project_roots(self, workspace_id: str):
            _ = workspace_id
            pytest.fail("roots should not be listed when workspace lookup fails")

    response = _get_workspace_roots_response(workspace_fastapi_app, _DatabaseErrorDB())

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace roots"


@pytest.mark.integration
def test_workspace_roots_endpoint_fails_closed_for_unknown_root_state_and_backend(workspace_fastapi_app):
    class _InvalidRootDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "workspace_profile": "project"}

        def list_workspace_project_roots(self, workspace_id: str):
            return [
                {
                    "workspace_id": workspace_id,
                    "root_id": "root-1",
                    "backend": "legacy_backend",
                    "root_state": "ready",
                    "display_name": "Legacy root",
                    "is_primary": True,
                    "version": 1,
                }
            ]

    response = _get_workspace_roots_response(workspace_fastapi_app, _InvalidRootDB())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["primary_root"]["state"] == "failed"
    assert payload["primary_root"]["backend"] is None
    assert payload["roots"][0]["state"] == "failed"
    assert payload["roots"][0]["backend"] is None


@pytest.mark.integration
@pytest.mark.parametrize(
    ("absolute_root", "expected_hint"),
    [
        ("/Users/example/project", "project"),
        (r"C:\Users\example\project", "project"),
        ("C:\\", "project_root"),
        (r"\\server\share\project", "project"),
        (r"\\server\share", "project_root"),
        (r"\Users\example\project", "project"),
        ("relative/secret/project", "project"),
    ],
)
def test_workspace_roots_endpoint_redacts_absolute_root_fallback(
    workspace_fastapi_app,
    absolute_root,
    expected_hint,
):
    class _RootPathDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "workspace_profile": "project"}

        def list_workspace_project_roots(self, workspace_id: str):
            return [
                {
                    "workspace_id": workspace_id,
                    "root_id": "root-1",
                    "backend": "host_local",
                    "root_state": "attached",
                    "absolute_root": absolute_root,
                    "is_primary": True,
                    "version": 1,
                }
            ]

    response = _get_workspace_roots_response(workspace_fastapi_app, _RootPathDB())

    assert response.status_code == 200, response.text
    root = response.json()["primary_root"]
    assert root["path_hint"] == expected_hint
    assert "absolute_root" not in root


@pytest.mark.integration
def test_attach_workspace_primary_host_local_root_returns_redacted_roots_response(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-root"
    assert payload["workspace_profile"] == "project"
    assert payload["primary_root"]["root_id"] == "primary"
    assert payload["primary_root"]["backend"] == "host_local"
    assert payload["primary_root"]["path_hint"] == "project"
    assert "absolute_root" not in payload["primary_root"]


@pytest.mark.integration
def test_attach_workspace_primary_host_local_root_outside_allowed_returns_service_error(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside" / "project"
    allowed.mkdir()
    outside.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(outside)},
    )

    assert response.status_code == 403, response.text
    assert response.json()["detail"]["code"] == "workspace_project_root_outside_allowed_roots"


@pytest.mark.integration
def test_attach_workspace_primary_host_local_root_without_configured_roots_returns_503(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "workspace_project_roots_not_configured"


@pytest.mark.integration
def test_attach_workspace_primary_different_root_without_replace_returns_409(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    original = allowed / "original"
    replacement = allowed / "replacement"
    original.mkdir(parents=True)
    replacement.mkdir()
    db.upsert_workspace("ws-root", "Rooted Workspace")
    db.upsert_workspace_primary_root(
        "ws-root",
        {
            "root_id": "primary",
            "backend": "host_local",
            "absolute_root": str(original.resolve()),
            "root_state": "attached",
            "is_primary": True,
        },
    )
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(replacement)},
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "workspace_primary_root_exists"


@pytest.mark.integration
def test_attach_workspace_primary_replacement_with_replace_existing_returns_new_primary(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    original = allowed / "original"
    replacement = allowed / "replacement"
    original.mkdir(parents=True)
    replacement.mkdir()
    db.upsert_workspace("ws-root", "Rooted Workspace")
    db.upsert_workspace_primary_root(
        "ws-root",
        {
            "root_id": "primary",
            "backend": "host_local",
            "absolute_root": str(original.resolve()),
            "root_state": "attached",
            "is_primary": True,
        },
    )
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {
            "backend": "host_local",
            "absolute_root": str(replacement),
            "replace_existing": True,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["primary_root"]["root_id"] == "primary"
    assert payload["primary_root"]["backend"] == "host_local"
    assert payload["primary_root"]["path_hint"] == "replacement"
    assert "absolute_root" not in payload["primary_root"]


@pytest.mark.integration
def test_attach_workspace_primary_sandbox_volume_returns_not_configured_mount_state(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Rooted Workspace")

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-123",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_profile"] == "project"
    assert payload["primary_root"]["root_id"] == "primary"
    assert payload["primary_root"]["backend"] == "sandbox_volume"
    assert payload["primary_root"]["path_hint"] == "volume-123"
    assert payload["primary_root"]["sandbox_mount_state"] == "not_configured"


@pytest.mark.integration
def test_provision_workspace_sandbox_root_requires_idempotency_key(workspace_fastapi_app, db):
    db.upsert_workspace("ws-root", "Rooted Workspace")

    response = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key=None,
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "workspace_idempotency_key_required"


@pytest.mark.integration
def test_provision_workspace_sandbox_root_returns_active_operation_and_pollable_status(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Rooted Workspace")
    sandbox_service = SandboxWorkspaceVolumeService(store=InMemoryStore())

    response = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key="root-key",
        sandbox_service=sandbox_service,
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["workspace_profile"] == "project"
    assert payload["primary_root"]["backend"] == "sandbox_volume"
    assert payload["primary_root"]["sandbox_mount_state"] == "not_configured"
    assert payload["operation"]["status"] == "running"
    assert payload["operation"]["retryable"] is True
    operation_id = payload["operation"]["operation_id"]

    retry = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key="root-key",
        sandbox_service=sandbox_service,
    )
    assert retry.status_code == 202, retry.text
    assert retry.json()["operation"]["operation_id"] == operation_id

    status_response = _get_workspace_operation_response(
        workspace_fastapi_app,
        db,
        "ws-root",
        operation_id,
    )
    assert status_response.status_code == 200, status_response.text
    assert status_response.json()["operation_id"] == operation_id

    context_response = _get_workspace_context_response(workspace_fastapi_app, db, "ws-root")
    assert context_response.status_code == 200, context_response.text
    active_operations = context_response.json()["active_operations"]
    assert [operation["operation_id"] for operation in active_operations] == [operation_id]


@pytest.mark.integration
def test_provision_workspace_sandbox_root_conflicts_for_changed_idempotent_request(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Rooted Workspace")
    sandbox_service = SandboxWorkspaceVolumeService(store=InMemoryStore())
    first = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key="root-key",
        sandbox_service=sandbox_service,
    )
    assert first.status_code == 202, first.text

    changed = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "vz_linux"},
        idempotency_key="root-key",
        sandbox_service=sandbox_service,
    )

    assert changed.status_code == 409, changed.text


@pytest.mark.integration
def test_provision_workspace_sandbox_root_maps_volume_idempotency_conflict_to_409(
    workspace_fastapi_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    db.upsert_workspace("ws-root", "Rooted Workspace")

    response = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key="root-key",
        sandbox_service=_ConflictingSandboxVolumeService(),
    )

    assert response.status_code == 409, response.text
    operation = db.get_workspace_operation_by_idempotency(
        workspace_id="ws-root",
        user_id="1",
        command="provision_sandbox_root",
        idempotency_key="root-key",
    )
    assert operation is not None
    assert operation["status"] == "conflicted"
    assert operation["diagnostics"]["code"] == "workspace_sandbox_volume_idempotency_conflict"


@pytest.mark.integration
def test_attached_host_local_primary_root_is_redacted_across_read_contracts(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    secret_parent = tmp_path / "tenant-secret-123" / "user-token-456"
    allowed = secret_parent / "allowed-project-roots"
    project = allowed / "public-project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    attach_response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )
    assert attach_response.status_code == 200, attach_response.text

    roots_response = _get_workspace_roots_response(workspace_fastapi_app, db)
    capabilities_response = _get_workspace_capabilities_response(workspace_fastapi_app, db)
    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert roots_response.status_code == 200, roots_response.text
    assert capabilities_response.status_code == 200, capabilities_response.text
    assert context_response.status_code == 200, context_response.text

    roots = roots_response.json()
    capabilities = capabilities_response.json()
    context = context_response.json()
    assert roots["workspace_profile"] == "project"
    assert capabilities["workspace_profile"] == "project"
    assert context["workspace_profile"] == "project"
    assert capabilities["workspace_kind"] == "project_workspace"
    assert context["workspace_kind"] == "project_workspace"

    roots_primary = roots["primary_root"]
    capabilities_root = capabilities["project_root"]
    context_root = context["project_root"]
    for root in (roots_primary, capabilities_root, context_root):
        assert root["root_id"] == "primary"
        assert root["backend"] == "host_local"
        assert root["path_hint"] == project.name
        assert root["file_inventory"]["available"] is True
        assert "absolute_root" not in root

    context_capability_root = context["capabilities"]["project_root"]
    assert context_capability_root["root_id"] == "primary"
    assert context_capability_root["backend"] == "host_local"
    assert context_capability_root["path_hint"] == project.name
    assert context_capability_root["file_inventory"]["available"] is True

    serialized_payloads = [
        json.dumps(payload, sort_keys=True)
        for payload in (roots, capabilities, context)
    ]
    forbidden_values = [
        str(project),
        str(allowed),
        str(secret_parent),
        "tenant-secret-123",
        "user-token-456",
        allowed.name,
    ]
    for serialized in serialized_payloads:
        assert project.name in serialized
        for forbidden_value in forbidden_values:
            assert forbidden_value not in serialized


@pytest.mark.integration
def test_workspace_capabilities_and_context_include_file_inventory_summary(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed-project-roots"
    project = allowed / "public-project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    attach_response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )
    assert attach_response.status_code == 200, attach_response.text
    root = db.get_workspace_primary_root("ws-root")
    assert root is not None
    scan = db.begin_workspace_file_inventory_scan(
        "ws-root",
        str(root["root_id"]),
        int(root["version"]),
        "policy-fingerprint",
        requested_by="test-user",
    )
    db.complete_workspace_file_inventory_scan(
        str(scan["scan_id"]),
        "current",
        {
            "files": 7,
            "directories": 2,
            "symlinks": 1,
            "ignored": 3,
            "indexing_candidates": 5,
            "diagnostics": 0,
            "total_entries": 10,
        },
        [],
        root_snapshot_token="snapshot-1",
    )

    capabilities_response = _get_workspace_capabilities_response(workspace_fastapi_app, db)
    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert capabilities_response.status_code == 200, capabilities_response.text
    assert context_response.status_code == 200, context_response.text
    capabilities = capabilities_response.json()
    context = context_response.json()
    for payload in (capabilities, context, context["capabilities"]):
        inventory = payload["project_root"]["file_inventory"]
        assert inventory["state"] == "current"
        assert inventory["total_file_count"] == 7
        assert inventory["indexed_file_count"] == 0
        assert isinstance(inventory["updated_at"], str)
        assert inventory["available"] is True
        assert payload["allowed_actions"]["scan_files"] == {
            "allowed": True,
            "reason_code": None,
        }
        assert payload["allowed_actions"]["view_file_inventory"] == {
            "allowed": True,
            "reason_code": None,
        }
        assert payload["allowed_actions"]["index_file_content"] == {
            "allowed": False,
            "reason_code": "file_indexing_disabled",
        }


@pytest.mark.integration
def test_sandbox_volume_primary_root_fails_closed_across_read_contracts(
    workspace_fastapi_app,
    db,
):
    volume_id = "volume-123"
    db.upsert_workspace("ws-root", "Rooted Workspace")

    attach_response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {
            "backend": "sandbox_volume",
            "sandbox_volume_id": volume_id,
        },
    )
    assert attach_response.status_code == 200, attach_response.text

    roots_response = _get_workspace_roots_response(workspace_fastapi_app, db)
    capabilities_response = _get_workspace_capabilities_response(workspace_fastapi_app, db)
    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert roots_response.status_code == 200, roots_response.text
    assert capabilities_response.status_code == 200, capabilities_response.text
    assert context_response.status_code == 200, context_response.text

    roots = roots_response.json()
    capabilities = capabilities_response.json()
    context = context_response.json()
    assert roots["primary_root"]["backend"] == "sandbox_volume"
    assert roots["primary_root"]["path_hint"] == volume_id
    assert roots["primary_root"]["sandbox_mount_state"] == "not_configured"
    assert capabilities["project_root"]["sandbox_mount_state"] == "not_configured"
    assert context["project_root"]["sandbox_mount_state"] == "not_configured"
    assert context["capabilities"]["project_root"]["sandbox_mount_state"] == "not_configured"
    assert context["project_root"]["file_inventory"]["available"] is False
    assert context["attention_state"] == "needs_attention"
    assert context["active_operations"] == []

    for payload in (capabilities, context):
        assert payload["workspace_kind"] == "project_workspace"
        assert payload["project_root"]["backend"] == "sandbox_volume"
        assert payload["project_root"]["path_hint"] == volume_id
        for action_name in (
            "write_files",
            "run_sandbox",
            "use_sandbox",
            "use_acp_agents",
        ):
            action = payload["allowed_actions"][action_name]
            assert action["allowed"] is False
            assert action["reason_code"] == "sandbox_mount_not_configured"


@pytest.mark.integration
def test_workspace_context_manager_defaults_for_research_workspace(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Research Workspace", workspace_profile="research")

    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert context_response.status_code == 200, context_response.text
    context = context_response.json()
    assert context["workspace_profile"] == "research"
    assert context["attention_state"] == "ready"
    assert context["project_root"]["state"] == "not_configured"
    assert context["project_root"]["file_inventory"]["available"] is False
    assert context["active_operations"] == []


@pytest.mark.integration
def test_workspace_context_project_shell_without_root_is_setup_pending(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Project Workspace", workspace_profile="project")

    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert context_response.status_code == 200, context_response.text
    context = context_response.json()
    assert context["workspace_profile"] == "project"
    assert context["attention_state"] == "setup_pending"
    assert context["project_root"]["state"] == "not_configured"
    assert context["project_root"]["file_inventory"]["available"] is False
    assert context["active_operations"] == []


@pytest.mark.integration
def test_workspace_context_project_inventory_scan_is_working(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed-project-roots"
    project = allowed / "active-inventory"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Project Workspace", workspace_profile="project")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )
    attach_response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )
    assert attach_response.status_code == 200, attach_response.text
    root = db.get_workspace_primary_root("ws-root")
    assert root is not None
    db.begin_workspace_file_inventory_scan(
        "ws-root",
        str(root["root_id"]),
        int(root["version"]),
        "policy-fingerprint",
        requested_by="test-user",
    )

    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert context_response.status_code == 200, context_response.text
    context = context_response.json()
    assert context["project_root"]["file_inventory"]["state"] == "queued"
    assert context["project_root"]["file_inventory"]["available"] is True
    assert context["attention_state"] == "working"
    assert context["active_operations"] == []


@pytest.mark.integration
def test_workspace_context_archived_workspace_attention_state(
    workspace_fastapi_app,
    db,
):
    workspace = db.upsert_workspace("ws-root", "Archived Project", workspace_profile="project")
    db.update_workspace("ws-root", {"archived": True}, expected_version=int(workspace["version"]))

    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert context_response.status_code == 200, context_response.text
    context = context_response.json()
    assert context["workspace"]["archived"] is True
    assert context["attention_state"] == "archived"
    assert context["active_operations"] == []


@pytest.mark.integration
def test_attach_workspace_primary_stale_expected_workspace_version_returns_409(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    workspace = db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {
            "backend": "host_local",
            "absolute_root": str(project),
            "expected_workspace_version": workspace["version"] + 1,
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "workspace_version_mismatch"


@pytest.mark.integration
def test_list_workspaces_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def list_workspaces(self):
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspaces"


@pytest.mark.integration
def test_get_workspace_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace"


@pytest.mark.integration
def test_upsert_workspace_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def upsert_workspace(
            self,
            workspace_id: str,
            name: str,
            *,
            study_materials_policy: str,
            workspace_profile: str,
        ):
            _ = (workspace_id, name, study_materials_policy, workspace_profile)
            raise InputError("invalid workspace create")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1",
                json={"name": "Workspace", "study_materials_policy": "workspace"},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace create"


@pytest.mark.integration
def test_upsert_workspace_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def upsert_workspace(
            self,
            workspace_id: str,
            name: str,
            *,
            study_materials_policy: str,
            workspace_profile: str,
        ):
            _ = (workspace_id, name, study_materials_policy, workspace_profile)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1",
                json={"name": "Workspace", "study_materials_policy": "workspace"},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to create or update workspace"


class TestScopedChatSessions:
    def test_workspace_chat_not_visible_in_global_list(self, db):
        db.upsert_workspace("ws-1", "WS")
        db.add_conversation({"title": "Global", "character_id": 1})
        db.add_conversation({
            "title": "WS Chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-1",
        })
        global_results = db.search_conversations(None, scope_type="global")
        assert all(r["scope_type"] == "global" for r in global_results)

    def test_global_chat_not_visible_in_workspace_list(self, db):
        db.upsert_workspace("ws-1", "WS")
        db.add_conversation({"title": "Global", "character_id": 1})
        ws_results = db.search_conversations(None, scope_type="workspace", workspace_id="ws-1")
        assert len(ws_results) == 0


@pytest.mark.integration
def test_delete_workspace_maps_conflict_to_409(workspace_fastapi_app):
    class _ConflictDB:
        backend_type = BackendType.SQLITE

        def close_connection(self) -> None:
            """This stateless fake has no checkout to release."""

        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace(self, workspace_id: str, expected_version: int) -> None:
            _ = (workspace_id, expected_version)
            raise ConflictError("Workspace 'ws-1' concurrent delete detected.")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _ConflictDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 409, response.text


@pytest.mark.integration
def test_patch_workspace_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def update_workspace(self, workspace_id: str, updates: dict, expected_version: int):
            _ = (workspace_id, updates, expected_version)
            raise InputError("invalid workspace patch")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-1",
                json={"name": "Renamed", "version": 1},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace patch"


@pytest.mark.integration
def test_delete_workspace_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        backend_type = BackendType.SQLITE

        def close_connection(self) -> None:
            """This stateless fake has no checkout to release."""

        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace(self, workspace_id: str, expected_version: int) -> None:
            _ = (workspace_id, expected_version)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete workspace"


@pytest.mark.integration
def test_update_workspace_source_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def update_workspace_source(
            self,
            workspace_id: str,
            source_id: str,
            updates: dict,
            *,
            expected_version: int,
        ):
            _ = (workspace_id, source_id, updates, expected_version)
            raise InputError("invalid workspace source patch")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/sources/src-1",
                json={"title": "Renamed", "version": 1},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace source patch"


@pytest.mark.integration
def test_workspace_artifact_endpoints_happy_path(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-art-api", "Workspace Artifacts")

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            add_response = client.post(
                "/api/v1/workspaces/ws-art-api/artifacts",
                json={
                    "id": "art-1",
                    "artifact_type": "summary",
                    "title": "Draft Summary",
                    "content": "Initial summary",
                },
            )
            assert add_response.status_code == 201, add_response.text
            added = add_response.json()
            assert added["id"] == "art-1"
            assert added["status"] == "pending"
            assert added["version"] == 1

            list_response = client.get("/api/v1/workspaces/ws-art-api/artifacts")
            assert list_response.status_code == 200, list_response.text
            listed = list_response.json()
            assert len(listed) == 1
            assert listed[0]["id"] == "art-1"
            assert listed[0]["title"] == "Draft Summary"

            update_response = client.put(
                "/api/v1/workspaces/ws-art-api/artifacts/art-1",
                json={
                    "title": "Final Summary",
                    "status": "completed",
                    "content": "Completed summary",
                    "version": added["version"],
                },
            )
            assert update_response.status_code == 200, update_response.text
            updated = update_response.json()
            assert updated["title"] == "Final Summary"
            assert updated["status"] == "completed"
            assert updated["content"] == "Completed summary"
            assert updated["version"] == 2

            delete_response = client.delete("/api/v1/workspaces/ws-art-api/artifacts/art-1")
            assert delete_response.status_code == 204, delete_response.text

            final_list_response = client.get("/api/v1/workspaces/ws-art-api/artifacts")
            assert final_list_response.status_code == 200, final_list_response.text
            assert final_list_response.json() == []
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)


def test_workspace_artifact_response_defaults_null_version_for_version_id():
    from tldw_Server_API.app.api.v1.endpoints.workspaces import _art_to_response

    response = _art_to_response({
        "id": "art-null-version",
        "workspace_id": "ws-1",
        "artifact_type": "summary",
        "title": "Summary",
        "version": None,
        "created_at": "2026-05-15T00:00:00Z",
    })

    assert response.version == 1
    assert response.artifact_version_id == "art-null-version:v1"


def test_workspace_artifact_redaction_schema_requires_typed_posture():
    from pydantic import ValidationError

    from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
        WorkspaceArtifactCreateRequest,
        WorkspaceArtifactResponse,
    )

    with pytest.raises(ValidationError):
        WorkspaceArtifactCreateRequest(
            id="brief-1",
            artifact_type="workspace_brief",
            title="Brief",
            redaction={"support_safe": "yes", "redacted": False},
        )

    schema = WorkspaceArtifactResponse.model_json_schema()
    redaction_ref = schema["properties"]["redaction"]["$ref"]
    redaction_schema = schema["$defs"][redaction_ref.rsplit("/", 1)[-1]]
    assert redaction_schema["properties"]["support_safe"]["type"] == "boolean"
    assert redaction_schema["properties"]["redacted"]["type"] == "boolean"


@pytest.mark.integration
def test_workspace_artifact_export_accepted_version_preserves_identity_and_refs(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-export-api", "Workspace Exports")
        db.add_workspace_artifact(
            "ws-export-api",
            {
                "id": "brief-1",
                "artifact_type": "workspace_brief",
                "title": "ACP Research Brief --> Review",
                "status": "completed",
                "content": "# Brief\nGrounded <answer>.",
                "content_type": "text/markdown",
                "review_state": "accepted",
                "owner_scope": "workspace",
                "owner_id": "ws-export-api",
                "producer_metadata": {
                    "producer_type": "acp",
                    "producer_id": "task-42",
                    "marker": "json --> comment boundary",
                    "run_id": "run-7",
                    "session_id": "session-abc",
                },
                "source_lineage": {
                    "sources": [
                        {"source_id": "src-1", "source_type": "media", "label": "Transcript"}
                    ]
                },
                "review_metadata": {"decision": "accepted"},
                "version_metadata": {"revision_reason": "initial"},
                "export_refs": [{"format": "legacy", "artifact_version_id": "brief-1:v1"}],
                "redaction": {"support_safe": True, "redacted": False},
            },
        )

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            by_format = {}
            for export_format in ("md", "html", "json"):
                request_json = {"format": export_format}
                if export_format == "json":
                    request_json["artifact_version_id"] = "brief-1:v1"
                response = client.post(
                    "/api/v1/workspaces/ws-export-api/artifacts/brief-1/exports",
                    json=request_json,
                )
                assert response.status_code == 200, response.text
                payload = response.json()
                by_format[export_format] = payload
                assert payload["workspace_id"] == "ws-export-api"
                assert payload["artifact_id"] == "brief-1"
                assert payload["artifact_version_id"] == "brief-1:v1"
                assert payload["review_state"] == "accepted"
                assert payload["format"] == export_format
                assert payload["bytes"] == len(payload["content"].encode("utf-8"))
                assert payload["metadata"]["source_lineage"]["sources"][0]["source_id"] == "src-1"
                assert payload["metadata"]["producer_metadata"]["run_id"] == "run-7"
                assert payload["export_ref"]["artifact_version_id"] == "brief-1:v1"

            md_content = by_format["md"]["content"]
            assert "artifact_id: brief-1" in md_content
            assert "tldw-artifact-metadata-base64:" in md_content
            marker = md_content.split("<!-- tldw-artifact-metadata-base64: ", 1)[1].split(" -->", 1)[0]
            decoded_metadata = json.loads(base64.b64decode(marker).decode("utf-8"))
            assert decoded_metadata["artifact"]["title"] == "ACP Research Brief --> Review"
            assert decoded_metadata["producer_metadata"]["marker"] == "json --> comment boundary"
            assert 'data-artifact-id="brief-1"' in by_format["html"]["content"]
            assert "<h1>Brief</h1>" in by_format["html"]["content"]
            assert "&lt;answer&gt;" in by_format["html"]["content"]
            exported_json = json.loads(by_format["json"]["content"])
            assert exported_json["artifact"]["id"] == "brief-1"
            assert exported_json["metadata"]["source_lineage"]["sources"][0]["source_id"] == "src-1"

            fetch_response = client.get("/api/v1/workspaces/ws-export-api/artifacts")
            assert fetch_response.status_code == 200, fetch_response.text
            exported_artifact = fetch_response.json()[0]
            export_refs = exported_artifact["export_refs"]
            assert export_refs[0]["format"] == "legacy"
            assert [ref["format"] for ref in export_refs[-3:]] == ["md", "html", "json"]
            assert {ref["artifact_version_id"] for ref in export_refs[-3:]} == {"brief-1:v1"}
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_artifact_export_missing_version_snapshot_fails_loudly(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-export-api", "Workspace Exports")
        db.add_workspace_artifact(
            "ws-export-api",
            {
                "id": "brief-1",
                "artifact_type": "workspace_brief",
                "title": "ACP Research Brief",
                "content": "# Brief\nGrounded answer.",
                "review_state": "accepted",
                "source_lineage": {"sources": [{"source_id": "src-1"}]},
            },
        )
        with db.transaction() as conn:
            conn.execute(
                "DELETE FROM workspace_artifact_versions "
                "WHERE workspace_id = ? AND artifact_id = ? AND artifact_version_id = ?",
                ("ws-export-api", "brief-1", "brief-1:v1"),
            )

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-export-api/artifacts/brief-1/exports",
                json={"format": "md"},
            )
            assert response.status_code == 409, response.text
            assert "missing" in response.json()["detail"].lower()

            fetch_response = client.get("/api/v1/workspaces/ws-export-api/artifacts")
            assert fetch_response.status_code == 200, fetch_response.text
            assert fetch_response.json()[0]["export_refs"] == []
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_artifact_export_rejects_non_accepted_state(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-export-api", "Workspace Exports")
        db.add_workspace_artifact(
            "ws-export-api",
            {
                "id": "brief-1",
                "artifact_type": "workspace_brief",
                "title": "ACP Research Brief",
                "content": "# Draft\nNeeds more evidence.",
                "review_state": "needs_revision",
                "source_lineage": {"sources": [{"source_id": "src-1"}]},
            },
        )

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-export-api/artifacts/brief-1/exports",
                json={"format": "md"},
            )
            assert response.status_code == 409, response.text
            assert response.json()["detail"] == "workspace_artifact_not_accepted"

            fetch_response = client.get("/api/v1/workspaces/ws-export-api/artifacts")
            assert fetch_response.status_code == 200, fetch_response.text
            assert fetch_response.json()[0]["export_refs"] == []
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_artifact_export_rejects_placeholder_generated_artifact(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-export-api", "Workspace Exports")
        db.add_workspace_artifact(
            "ws-export-api",
            {
                "id": "slides-1",
                "artifact_type": "slides",
                "title": "Generated Slides",
                "content": "slides go here",
                "review_state": "accepted",
                "source_lineage": {"sources": [{"source_id": "src-1"}]},
            },
        )

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-export-api/artifacts/slides-1/exports",
                json={"format": "md"},
            )
            assert response.status_code == 409, response.text
            assert response.json()["detail"] == "workspace_artifact_placeholder_content"

            fetch_response = client.get("/api/v1/workspaces/ws-export-api/artifacts")
            assert fetch_response.status_code == 200, fetch_response.text
            assert fetch_response.json()[0]["export_refs"] == []
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_artifact_api_exposes_traceable_contract_fields(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-art-api", "Workspace Artifacts")

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            add_response = client.post(
                "/api/v1/workspaces/ws-art-api/artifacts",
                json={
                    "id": "brief-1",
                    "artifact_type": "workspace_brief",
                    "title": "ACP Research Brief",
                    "status": "completed",
                    "content": "# Brief\nGrounded answer.",
                    "content_type": "text/markdown",
                    "preview_text": "Grounded answer.",
                    "summary": "Executive summary",
                    "review_state": "accepted",
                    "owner_scope": "workspace",
                    "owner_id": "ws-art-api",
                    "producer_metadata": {
                        "producer_type": "acp",
                        "producer_id": "task-42",
                        "run_id": "run-7",
                        "session_id": "session-abc",
                    },
                    "source_lineage": {
                        "sources": [
                            {"source_id": "src-1", "source_type": "media", "label": "Transcript"}
                        ]
                    },
                    "root_artifact_id": "forged-root",
                    "artifact_version_id": "forged:v99",
                    "previous_version_id": "forged:v98",
                    "review_metadata": {"reviewer_id": "reviewer-1", "decision": "accepted"},
                    "version_metadata": {"revision_reason": "initial"},
                    "export_refs": [{"format": "md", "file_id": 101}],
                    "redaction": {"support_safe": True, "redacted": False, "retention_class": "standard"},
                },
            )

            assert add_response.status_code == 201, add_response.text
            added = add_response.json()
            assert added["review_state"] == "accepted"
            assert added["root_artifact_id"] == "brief-1"
            assert added["artifact_version_id"] == "brief-1:v1"
            assert added["producer_metadata"]["producer_type"] == "acp"
            assert added["source_lineage"]["sources"][0]["source_id"] == "src-1"
            assert added["redaction"]["support_safe"] is True

            fetch_response = client.get("/api/v1/workspaces/ws-art-api/artifacts")
            assert fetch_response.status_code == 200, fetch_response.text
            fetched = fetch_response.json()[0]
            assert fetched["artifact_version_id"] == "brief-1:v1"
            assert fetched["review_metadata"]["decision"] == "accepted"
            assert fetched["export_refs"][0]["file_id"] == 101
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)


@pytest.mark.integration
def test_list_workspace_artifacts_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def list_workspace_artifacts(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-1/artifacts")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace artifacts"


@pytest.mark.integration
def test_add_workspace_artifact_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def add_workspace_artifact(self, workspace_id: str, data: dict):
            _ = (workspace_id, data)
            raise InputError("invalid workspace artifact create")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-1/artifacts",
                json={
                    "id": "art-1",
                    "artifact_type": "summary",
                    "title": "Draft Summary",
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace artifact create"


@pytest.mark.integration
def test_delete_workspace_artifact_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace_artifact(self, workspace_id: str, artifact_id: str) -> None:
            _ = (workspace_id, artifact_id)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1/artifacts/art-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete workspace artifact"


@pytest.mark.integration
def test_update_workspace_artifact_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict,
            *,
            expected_version: int,
        ):
            _ = (workspace_id, artifact_id, updates, expected_version)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/artifacts/art-1",
                json={"title": "Updated", "version": 1},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to update workspace artifact"


@pytest.mark.integration
def test_update_workspace_note_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def update_workspace_note(
            self,
            workspace_id: str,
            note_id: int,
            updates: dict,
            *,
            expected_version: int,
        ):
            _ = (workspace_id, note_id, updates, expected_version)
            raise InputError("invalid workspace note patch")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/notes/42",
                json={"title": "Updated", "version": 1},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace note patch"


@pytest.mark.integration
def test_workspace_note_endpoints_happy_path(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-note-api", "Workspace Notes")

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            add_response = client.post(
                "/api/v1/workspaces/ws-note-api/notes",
                json={
                    "title": "Draft Note",
                    "content": "Initial note body",
                    "keywords": ["alpha", "beta"],
                },
            )
            assert add_response.status_code == 201, add_response.text
            added = add_response.json()
            assert added["title"] == "Draft Note"
            assert added["content"] == "Initial note body"
            assert json.loads(added["keywords_json"]) == ["alpha", "beta"]
            assert added["version"] == 1

            list_response = client.get("/api/v1/workspaces/ws-note-api/notes")
            assert list_response.status_code == 200, list_response.text
            listed = list_response.json()
            assert len(listed) == 1
            assert listed[0]["id"] == added["id"]

            update_response = client.put(
                f"/api/v1/workspaces/ws-note-api/notes/{added['id']}",
                json={
                    "title": "Final Note",
                    "content": "Updated note body",
                    "keywords_json": json.dumps(["gamma"]),
                    "version": added["version"],
                },
            )
            assert update_response.status_code == 200, update_response.text
            updated = update_response.json()
            assert updated["title"] == "Final Note"
            assert updated["content"] == "Updated note body"
            assert json.loads(updated["keywords_json"]) == ["gamma"]
            assert updated["version"] == 2

            delete_response = client.delete(f"/api/v1/workspaces/ws-note-api/notes/{added['id']}")
            assert delete_response.status_code == 204, delete_response.text

            final_list_response = client.get("/api/v1/workspaces/ws-note-api/notes")
            assert final_list_response.status_code == 200, final_list_response.text
            assert final_list_response.json() == []
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)


@pytest.mark.integration
def test_list_workspace_notes_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def list_workspace_notes(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-1/notes")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace notes"


@pytest.mark.integration
def test_add_workspace_note_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def add_workspace_note(self, workspace_id: str, data: dict):
            _ = (workspace_id, data)
            raise InputError("invalid workspace note create")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-1/notes",
                json={
                    "title": "Draft Note",
                    "content": "Initial note body",
                    "keywords": ["alpha"],
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace note create"


@pytest.mark.integration
def test_delete_workspace_note_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace_note(self, workspace_id: str, note_id: int) -> None:
            _ = (workspace_id, note_id)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1/notes/42")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete workspace note"


@pytest.mark.integration
def test_workspace_source_endpoints_happy_path(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    job_manager = _CapturingJobManager()
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: job_manager
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-src-api", "Workspace Sources")

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            add_response = client.post(
                "/api/v1/workspaces/ws-src-api/sources",
                json={
                    "id": "src-1",
                    "media_id": 1,
                    "title": "Video Source",
                    "source_type": "video",
                },
            )
            assert add_response.status_code == 201, add_response.text
            added = add_response.json()
            assert added["id"] == "src-1"
            assert added["selected"] is True

            duplicate_add_response = client.post(
                "/api/v1/workspaces/ws-src-api/sources",
                json={
                    "id": "src-1",
                    "media_id": 1,
                    "title": "Video Source",
                    "source_type": "video",
                },
            )
            assert duplicate_add_response.status_code == 201, duplicate_add_response.text
            duplicate_added = duplicate_add_response.json()
            assert duplicate_added["id"] == "src-1"
            assert duplicate_added["media_id"] == 1

            list_response = client.get("/api/v1/workspaces/ws-src-api/sources")
            assert list_response.status_code == 200, list_response.text
            assert [item["id"] for item in list_response.json()] == ["src-1"]

            second_add_response = client.post(
                "/api/v1/workspaces/ws-src-api/sources",
                json={
                    "id": "src-2",
                    "media_id": 2,
                    "title": "Article Source",
                    "source_type": "article",
                },
            )
            assert second_add_response.status_code == 201, second_add_response.text

            selection_response = client.put(
                "/api/v1/workspaces/ws-src-api/sources/selection",
                json={"selected_ids": ["src-2"]},
            )
            assert selection_response.status_code == 200, selection_response.text

            reorder_response = client.put(
                "/api/v1/workspaces/ws-src-api/sources/reorder",
                json={"ordered_ids": ["src-2", "src-1"]},
            )
            assert reorder_response.status_code == 200, reorder_response.text

            reordered_list_response = client.get("/api/v1/workspaces/ws-src-api/sources")
            assert reordered_list_response.status_code == 200, reordered_list_response.text
            reordered_sources = reordered_list_response.json()
            assert [item["id"] for item in reordered_sources] == ["src-2", "src-1"]
            assert reordered_sources[0]["selected"] is True
            assert reordered_sources[1]["selected"] is False

            delete_response = client.delete("/api/v1/workspaces/ws-src-api/sources/src-1")
            assert delete_response.status_code == 204, delete_response.text

            final_list_response = client.get("/api/v1/workspaces/ws-src-api/sources")
            assert final_list_response.status_code == 200, final_list_response.text
            assert [item["id"] for item in final_list_response.json()] == ["src-2"]
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(workspaces_endpoint.try_get_workspace_job_manager, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)


@pytest.mark.integration
def test_add_workspace_source_enqueues_workspace_ingest_job(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=7,
            username="researcher",
            email="researcher@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    job_manager = _CapturingJobManager()
    db.upsert_workspace("ws-job-api", "Workspace Source Jobs")
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: job_manager
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-job-api/sources",
                json={
                    "id": "src-job-1",
                    "media_id": 55,
                    "title": "NotebookLM Migration PDF",
                    "source_type": "pdf",
                    "url": "file:///imports/notebooklm.pdf",
                },
            )
            duplicate_response = client.post(
                "/api/v1/workspaces/ws-job-api/sources",
                json={
                    "id": "src-job-1",
                    "media_id": 55,
                    "title": "NotebookLM Migration PDF",
                    "source_type": "pdf",
                    "url": "file:///imports/notebooklm.pdf",
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(workspaces_endpoint.try_get_workspace_job_manager, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 201, response.text
    assert duplicate_response.status_code == 201, duplicate_response.text
    assert len(job_manager.created_jobs) == 2
    first_job = job_manager.created_jobs[0]
    assert first_job["domain"] == "media_ingest"
    assert first_job["queue"] == "default"
    assert first_job["job_type"] == "workspace_source_ingest"
    assert first_job["owner_user_id"] == "7"
    assert first_job["idempotency_key"] == "workspace-source:ws-job-api:src-job-1:55"
    assert first_job["max_retries"] == 3
    assert first_job["payload"] == {
        "workspace_id": "ws-job-api",
        "workspace_source_id": "src-job-1",
        "source_id": "src-job-1",
        "media_id": 55,
        "source_type": "pdf",
        "title": "NotebookLM Migration PDF",
        "url": "file:///imports/notebooklm.pdf",
        "requested_stages": ["ingestion", "extraction", "chunking", "indexing"],
    }
    assert job_manager.created_jobs[1]["idempotency_key"] == first_job["idempotency_key"]


@pytest.mark.integration
def test_add_workspace_source_ignores_unused_job_manager_override(workspace_fastapi_app, db):
    async def _allow_rate_limit() -> None:
        return None

    db.upsert_workspace("ws-job-fail-api", "Workspace Source Job Failure")
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=8)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = (
        lambda: _FailingJobManager()
    )
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-job-fail-api/sources",
                json={
                    "id": "src-job-fail",
                    "media_id": 56,
                    "title": "Resilient Source",
                    "source_type": "web",
                    "url": "https://example.test/resilient-source",
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(workspaces_endpoint.try_get_workspace_job_manager, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 201, response.text
    assert response.json()["id"] == "src-job-fail"
    assert [src["id"] for src in db.list_workspace_sources("ws-job-fail-api")] == ["src-job-fail"]


@pytest.mark.integration
def test_add_workspace_source_does_not_construct_job_manager(workspace_fastapi_app, db):
    async def _allow_rate_limit() -> None:
        return None

    def _raise_job_manager() -> None:
        raise RuntimeError("jobs manager construction failed")

    db.upsert_workspace("ws-job-dep-fail-api", "Workspace Source Job Dependency Failure")
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=9)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[get_job_manager] = _raise_job_manager
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-job-dep-fail-api/sources",
                json={
                    "id": "src-job-dep-fail",
                    "media_id": 57,
                    "title": "Dependency Resilient Source",
                    "source_type": "pdf",
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_job_manager, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 201, response.text
    assert response.json()["id"] == "src-job-dep-fail"
    assert [src["id"] for src in db.list_workspace_sources("ws-job-dep-fail-api")] == ["src-job-dep-fail"]


@pytest.mark.integration
def test_list_workspace_sources_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def list_workspace_sources(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-1/sources")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace sources"


@pytest.mark.integration
def test_add_workspace_source_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def add_workspace_source(self, workspace_id: str, data: dict):
            _ = (workspace_id, data)
            raise InputError("invalid workspace source create")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-1/sources",
                json={"id": "src-1", "media_id": 1, "title": "Video", "source_type": "video"},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace source create"


@pytest.mark.integration
def test_delete_workspace_source_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace_source(self, workspace_id: str, source_id: str):
            _ = (workspace_id, source_id)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1/sources/src-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete workspace source"


@pytest.mark.integration
def test_update_workspace_source_selection_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def update_workspace_source_selection(self, workspace_id: str, *, selected_ids: list[str]):
            _ = (workspace_id, selected_ids)
            raise InputError("invalid workspace source selection")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/sources/selection",
                json={"selected_ids": ["src-1"]},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace source selection"


@pytest.mark.integration
def test_reorder_workspace_sources_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def reorder_workspace_sources(self, workspace_id: str, ordered_ids: list[str]):
            _ = (workspace_id, ordered_ids)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/sources/reorder",
                json={"ordered_ids": ["src-2", "src-1"]},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to reorder workspace sources"
