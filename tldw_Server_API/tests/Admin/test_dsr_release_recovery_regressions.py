"""Bounded release probes for normalized storage and interrupted DSR execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_erasure_rejects_unavailable_chroma_at_normalized_tilde_path(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    configured_base = "~/" + os.path.relpath(tmp_path, Path.home())
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", configured_base)
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    real_storage = (
        service.DatabasePaths.resolve_user_base_directory(7, base_dir_override=configured_base) / "chroma_storage"
    )
    real_storage.mkdir(parents=True)
    sentinel = real_storage / "retained-synthetic-content"
    sentinel.write_text("synthetic embedding content")

    def unavailable_manager(_user_id):
        raise RuntimeError("injected manager failure")

    monkeypatch.setattr(service, "_get_chroma_manager_for_user", unavailable_manager)
    with pytest.raises(service.DataSubjectRequestCoverageUnavailableError):
        await service._count_embeddings(7)
    with pytest.raises(RuntimeError, match="cannot confirm|unavailable|availability"):
        await service._erase_embeddings(7)
    assert sentinel.read_text() == "synthetic embedding content"


@pytest.mark.parametrize("storage_state", ["absent", "existing", "uninspectable", "unresolved"])
async def test_erasure_requires_confirmed_storage_absence(tmp_path, monkeypatch, storage_state):
    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    storage = tmp_path / "7" / "chroma_storage"
    if storage_state == "existing":
        storage.mkdir(parents=True)

    def unavailable_manager(_user_id):
        raise RuntimeError("injected manager failure")

    monkeypatch.setattr(service, "_get_chroma_manager_for_user", unavailable_manager)
    if storage_state == "unresolved":

        def unresolved(*_args, **_kwargs):
            raise ValueError("injected path resolution failure")

        monkeypatch.setattr(service.DatabasePaths, "resolve_user_base_directory", unresolved)
    elif storage_state == "uninspectable":
        original_stat = Path.stat

        def unavailable_stat(path, *args, **kwargs):
            if path == storage:
                raise PermissionError("injected storage inspection failure")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", unavailable_stat)

    if storage_state == "absent":
        assert await service._erase_embeddings(7) == 0
        assert not storage.exists()
    else:
        with pytest.raises(RuntimeError, match="cannot confirm|unavailable|availability"):
            await service._erase_embeddings(7)


@pytest.mark.parametrize("after_delete", [False, True])
async def test_ended_erasure_recovers_through_linked_replacement_request(tmp_path, monkeypatch, after_delete):
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import DataSubjectRequestCreateRequest
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import AuthnzDataSubjectRequestsRepo
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path / "backups"))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    await reset_db_pool()
    reset_settings()
    pool = await get_db_pool()
    db = None
    try:
        subject = await ensure_test_user(pool, "recovery_subject")
        admin_id = await ensure_test_user(pool, "recovery_operator", role="admin")
        principal = AuthPrincipal(kind="user", user_id=admin_id, roles=["admin"], is_admin=True)
        repo = AuthnzDataSubjectRequestsRepo(pool)
        await repo.ensure_schema()
        record = await repo.create_or_get_request(
            client_request_id="interrupted-release-probe",
            requester_identifier=str(subject),
            resolved_user_id=subject,
            request_type="erasure",
            status="recorded",
            selected_categories=["notes"],
            preview_summary=[],
            coverage_metadata={},
            requested_by_user_id=None,
            notes=None,
        )
        db = CharactersRAGDB(tmp_path / "notes.db", client_id=str(subject))
        db.add_note("Synthetic note", "Synthetic body")
        monkeypatch.setattr(service.DatabasePaths, "get_chacha_db_path", lambda _owner: db.db_path)

        async def erase_then_interrupt(owner):
            if after_delete:
                await service._erase_notes(owner)
            raise asyncio.CancelledError("injected shutdown with no handler work remaining")

        monkeypatch.setitem(service._ERASURE_HANDLERS, "notes", erase_then_interrupt)
        with pytest.raises(asyncio.CancelledError):
            await service.execute_dsr_erasure(
                request_id=record["id"],
                user_id=subject,
                selected_categories=["notes"],
                dsr_repo=repo,
            )
        assert db.execute_query("SELECT COUNT(*) FROM notes").fetchone()[0] == (0 if after_delete else 1)
        await reset_db_pool()
        pool = await get_db_pool()
        restarted_repo = AuthnzDataSubjectRequestsRepo(pool)
        persisted = await restarted_repo.get_request_by_id(record["id"])
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops

        monkeypatch.setattr(
            admin_data_ops, "_build_dsr_repos", AsyncMock(return_value=(AuthnzUsersRepo(pool), restarted_repo))
        )
        # The durable linkage is in the real DSR record. This test does not certify
        # the independent unified audit delivery service.
        monkeypatch.setattr(admin_data_ops, "_emit_admin_audit_event", AsyncMock())
        with pytest.raises(HTTPException) as rejected:
            await admin_data_ops.execute_data_subject_request(record["id"], request=None, principal=principal)
        assert rejected.value.status_code == 409
        assert rejected.value.detail == "request_already_executing"
        assert persisted["status"] == "executing"

        # The injected execution has conclusively ended: its only thread-backed
        # handler was awaited to completion before cancellation, or never started.
        monkeypatch.setitem(service._ERASURE_HANDLERS, "notes", service._erase_notes)
        linkage = f"Recovery of DSR {record['id']}; original execution ended; operator {admin_id} confirmed."
        payload = DataSubjectRequestCreateRequest(
            client_request_id="replacement-release-probe",
            requester_identifier=str(subject),
            request_type="erasure",
            categories=["notes"],
            notes=linkage,
        )
        replacement = await admin_data_ops.create_data_subject_request(payload, request=None, principal=principal)
        replacement_id = replacement.item.id
        assert replacement_id != record["id"]
        same_replacement = await admin_data_ops.create_data_subject_request(payload, request=None, principal=principal)
        assert same_replacement.item.id == replacement_id
        result = await admin_data_ops.execute_data_subject_request(replacement_id, request=None, principal=principal)
        assert result["status"] == "completed"
        assert result["categories"]["notes"]["deleted_count"] == (0 if after_delete else 1)
        assert db.execute_query("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
        await reset_db_pool()
        pool = await get_db_pool()
        recovered_repo = AuthnzDataSubjectRequestsRepo(pool)
        receipt = await recovered_repo.get_request_by_id(replacement_id)
        assert receipt["notes"] == linkage
        assert receipt["requested_by_user_id"] == admin_id
        assert receipt["resolved_user_id"] == subject
        assert receipt["selected_categories"] == ["notes"]
        assert receipt["status"] == "completed"
        assert (await recovered_repo.get_request_by_id(record["id"]))["status"] == "executing"
    finally:
        if db is not None:
            db.close_all_connections()
        await reset_db_pool()
        reset_settings()
