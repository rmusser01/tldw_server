from __future__ import annotations

import os
import uuid

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway


def _setup_env(tmp_path) -> None:
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-api-key"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_dsr_repo.db'}"
    os.environ["TLDW_DB_ALLOWED_BASE_DIRS"] = str(tmp_path)
    os.environ["TLDW_DB_BACKUP_PATH"] = str(tmp_path / "backups")
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")


@pytest.mark.asyncio
async def test_repo_create_request_is_idempotent(tmp_path):
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
        AuthnzDataSubjectRequestsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()

    pool = await get_db_pool()
    async with pool.transaction() as conn:
        gateway = VersionedUserWriteGateway("sqlite")
        for user_id, username, role in (
            (1, "admin_requester", "admin"),
            (7, "subject_user", "user"),
        ):
            await gateway.insert_user(
                conn,
                values={
                    "id": user_id,
                    "username": username,
                    "email": f"{username}@example.com",
                    "password_hash": "hash",  # nosec B105 # Inert fixture hash, never used for login
                    "role": role,
                    "is_active": 1,
                    "uuid": str(uuid.uuid4()),
                },
            )
    repo = AuthnzDataSubjectRequestsRepo(db_pool=pool)
    await repo.ensure_schema()

    first = await repo.create_or_get_request(
        client_request_id="dsr-1",
        requester_identifier="user@example.com",
        resolved_user_id=7,
        request_type="export",
        status="recorded",
        selected_categories=["media_records"],
        preview_summary=[{"key": "media_records", "count": 3}],
        coverage_metadata={"supported": ["media_records"]},
        requested_by_user_id=1,
        notes=None,
    )
    second = await repo.create_or_get_request(
        client_request_id="dsr-1",
        requester_identifier="user@example.com",
        resolved_user_id=7,
        request_type="export",
        status="recorded",
        selected_categories=["media_records"],
        preview_summary=[{"key": "media_records", "count": 3}],
        coverage_metadata={"supported": ["media_records"]},
        requested_by_user_id=1,
        notes=None,
    )

    assert first["id"] == second["id"]
    assert first["client_request_id"] == "dsr-1"
    assert first["request_type"] == "export"
    assert first["status"] == "recorded"
    assert first["selected_categories"] == ["media_records"]
    assert first["preview_summary"] == [{"key": "media_records", "count": 3}]

    rows, total = await repo.list_requests(limit=10, offset=0)
    assert total == 1
    assert len(rows) == 1
    assert rows[0]["id"] == first["id"]


@pytest.mark.asyncio
async def test_repo_ensure_schema_surfaces_sqlite_failures(tmp_path, monkeypatch):
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
        AuthnzDataSubjectRequestsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()

    pool = await get_db_pool()
    repo = AuthnzDataSubjectRequestsRepo(db_pool=pool)

    from tldw_Server_API.app.core.AuthNZ import migrations

    def _boom(*args, **kwargs):
        raise RuntimeError("schema failed")

    monkeypatch.setattr(migrations, "ensure_authnz_tables", _boom)

    with pytest.raises(RuntimeError, match="schema failed"):
        await repo.ensure_schema()


@pytest.mark.asyncio
async def test_repo_list_requests_scopes_by_org_ids(tmp_path):
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
        AuthnzDataSubjectRequestsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()

    pool = await get_db_pool()
    async with pool.transaction() as conn:
        gateway = VersionedUserWriteGateway("sqlite")
        for user_id, username, role in (
            (1, "admin_requester", "admin"),
            (7, "subject_user_one", "user"),
            (8, "subject_user_two", "user"),
        ):
            await gateway.insert_user(
                conn,
                values={
                    "id": user_id,
                    "username": username,
                    "email": f"{username}@example.com",
                    "password_hash": "hash",  # nosec B105 # Inert fixture hash, never used for login
                    "role": role,
                    "is_active": 1,
                    "uuid": str(uuid.uuid4()),
                },
            )

    repo = AuthnzDataSubjectRequestsRepo(db_pool=pool)
    await repo.ensure_schema()

    await pool.execute(
        "INSERT INTO organizations (name, slug, owner_user_id) VALUES (?, ?, ?)",
        "Scoped Org",
        "scoped-org",
        1,
    )
    org_row = await pool.fetchone("SELECT id FROM organizations WHERE slug = ?", "scoped-org")
    org_id = int(org_row["id"])
    await pool.execute(
        "INSERT INTO org_members (org_id, user_id, role, status) VALUES (?, ?, ?, ?)",
        org_id,
        7,
        "member",
        "active",
    )

    await repo.create_or_get_request(
        client_request_id="dsr-org-1",
        requester_identifier="subject_user_one@example.com",
        resolved_user_id=7,
        request_type="access",
        status="recorded",
        selected_categories=["media_records"],
        preview_summary=[{"key": "media_records", "count": 1}],
        coverage_metadata={"supported": ["media_records"]},
        requested_by_user_id=1,
        notes=None,
    )
    await repo.create_or_get_request(
        client_request_id="dsr-org-2",
        requester_identifier="subject_user_two@example.com",
        resolved_user_id=8,
        request_type="access",
        status="recorded",
        selected_categories=["media_records"],
        preview_summary=[{"key": "media_records", "count": 2}],
        coverage_metadata={"supported": ["media_records"]},
        requested_by_user_id=1,
        notes=None,
    )

    rows, total = await repo.list_requests(limit=10, offset=0, org_ids=[org_id])
    assert total == 1
    assert [row["client_request_id"] for row in rows] == ["dsr-org-1"]
