"""Real-backend rotation rejection, rollback, and audit contracts."""

import sqlite3
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ import api_key_manager as manager_module
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager, reset_api_key_manager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def rotation_context(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[APIKeyManager, int]]:
    backend = request.param
    db_url = (
        request.getfixturevalue("pg_temp_db")["dsn"]
        if backend == "postgres"
        else f"sqlite:///{tmp_path / 'rotation.db'}"
    )
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("JWT_SECRET_KEY", uuid.uuid4().hex + uuid.uuid4().hex)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    await reset_api_key_manager()
    await reset_db_pool()
    await shutdown_all_audit_services()
    reset_settings()
    pool = await get_db_pool()
    try:
        assert bool(pool.pool) is (backend == "postgres")
        users = UsersDB(pool)
        await users.initialize(ensure_schema=False)
        user = await users.create_user(  # nosec B106 # Inert fixture hash, never used for login
            username="rotation-user",
            email="rotation@example.com",
            password_hash="unused-fixture-hash",
        )
        manager = APIKeyManager(db_pool=pool)
        await manager.initialize()
        yield manager, int(user["id"])
    finally:
        await shutdown_all_audit_services()
        await reset_api_key_manager()
        await reset_db_pool()
        reset_settings()


def _audit_rows(user_id: int) -> list[tuple[str, str]]:
    """Read the real per-user audit receipt independently of manager state."""
    with sqlite3.connect(DatabasePaths.get_audit_db_path(user_id)) as conn:
        return conn.execute(
            "SELECT action, resource_id FROM audit_events ORDER BY timestamp, event_id"
        ).fetchall()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_state", ["missing", "foreign", "revoked", "late_inactive"])
async def test_rotation_rejection_preserves_value_error_and_state(
    rotation_context: tuple[APIKeyManager, int],
    monkeypatch: pytest.MonkeyPatch,
    source_state: str,
) -> None:
    manager, user_id = rotation_context
    pool = manager.db_pool
    created = await manager.create_api_key(user_id=user_id, name="source", scope="read")
    key_id = int(created["id"])
    repo = manager._get_repo()
    active_snapshot = await repo.fetch_key_for_user(key_id, user_id)
    if source_state in {"revoked", "late_inactive"}:
        await manager.revoke_api_key(key_id, user_id=user_id, reason="test rejection")
    if source_state == "late_inactive":
        # Model a stale active read; the real conditional update must still reject
        # the persisted revocation and roll back its provisional replacement key.
        monkeypatch.setattr(repo, "fetch_key_for_user", AsyncMock(return_value=active_snapshot))

    before = [dict(row) for row in await pool.fetchall("SELECT * FROM api_keys ORDER BY id")]
    audit_before = _audit_rows(user_id)
    with pytest.raises(ValueError, match="^API key not found or unauthorized$"):
        await manager.rotate_api_key(
            key_id=key_id + 1 if source_state == "missing" else key_id,
            user_id=user_id + 1 if source_state == "foreign" else user_id,
        )

    assert [dict(row) for row in await pool.fetchall("SELECT * FROM api_keys ORDER BY id")] == before
    assert _audit_rows(user_id) == audit_before


@pytest.mark.asyncio
async def test_rotation_commits_links_and_mandatory_audit(
    rotation_context: tuple[APIKeyManager, int],
) -> None:
    manager, user_id = rotation_context
    original = await manager.create_api_key(user_id=user_id, name="source", scope="read")

    rotated = await manager.rotate_api_key(key_id=int(original["id"]), user_id=user_id)

    rows = [dict(row) for row in await manager.db_pool.fetchall("SELECT * FROM api_keys ORDER BY id")]
    assert len(rows) == 2
    assert (rows[0]["id"], rows[0]["status"], rows[0]["rotated_to"]) == (
        original["id"], "rotated", rotated["id"]
    )
    assert (rows[1]["id"], rows[1]["status"], rows[1]["rotated_from"]) == (
        rotated["id"], "active", original["id"]
    )
    assert ("api_key.rotate", str(rotated["id"])) in _audit_rows(user_id)


@pytest.mark.asyncio
async def test_rotation_audit_failure_rolls_back_both_keys(
    rotation_context: tuple[APIKeyManager, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, user_id = rotation_context
    original = await manager.create_api_key(user_id=user_id, name="source", scope="read")
    before = [dict(row) for row in await manager.db_pool.fetchall("SELECT * FROM api_keys ORDER BY id")]
    audit_before = _audit_rows(user_id)
    monkeypatch.setattr(
        manager_module,
        "emit_mandatory_api_key_management_audit",
        AsyncMock(side_effect=MandatoryAuditWriteError("Mandatory audit persistence unavailable")),
    )

    with pytest.raises(MandatoryAuditWriteError, match="Mandatory audit persistence unavailable"):
        await manager.rotate_api_key(key_id=int(original["id"]), user_id=user_id)

    assert [dict(row) for row in await manager.db_pool.fetchall("SELECT * FROM api_keys ORDER BY id")] == before
    assert _audit_rows(user_id) == audit_before


@pytest.mark.asyncio
async def test_unrelated_value_error_remains_a_storage_failure(
    rotation_context: tuple[APIKeyManager, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, user_id = rotation_context
    monkeypatch.setattr(
        manager._get_repo(), "fetch_key_for_user", AsyncMock(side_effect=ValueError("invalid storage value"))
    )

    with pytest.raises(DatabaseError, match="Failed to rotate API key"):
        await manager.rotate_api_key(key_id=1, user_id=user_id)
