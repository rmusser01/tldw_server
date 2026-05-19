import json
import sqlite3
import uuid

import pytest
import pytest_asyncio

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services
import tldw_Server_API.app.core.AuthNZ.api_key_manager as api_key_manager_module
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager, APIKeyStatus
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

pytestmark = pytest.mark.integration


async def _create_auth_user(pool, *, prefix: str) -> int:
    uname = f"{prefix}_{uuid.uuid4().hex[:8]}"
    email = f"{uname}@example.com"
    await pool.execute(
        """
        INSERT INTO users (username, email, password_hash, is_active)
        VALUES (?, ?, ?, 1)
        """,
        (uname, email, "x"),
    )
    user_row = await pool.fetchone("SELECT id FROM users WHERE username = ?", uname)
    return user_row["id"] if isinstance(user_row, dict) else user_row[0]


def _install_failing_mandatory_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail_audit(**_kwargs):
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    monkeypatch.setattr(
        api_key_manager_module,
        "emit_mandatory_api_key_management_audit",
        _fail_audit,
    )


@pytest_asyncio.fixture
async def sqlite_authnz_pool(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import reset_api_key_manager

    db_path = tmp_path / f"api_key_rotation_{uuid.uuid4().hex}.sqlite"
    user_db_base = (tmp_path / "user_databases").resolve()
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))

    await reset_api_key_manager()
    await reset_db_pool()
    await shutdown_all_audit_services()
    pool = await get_db_pool()
    try:
        yield pool
    finally:
        await shutdown_all_audit_services()
        await reset_api_key_manager()
        await reset_db_pool()


@pytest.mark.asyncio
async def test_api_key_rotation_and_audit_sqlite(sqlite_authnz_pool):
    pool = sqlite_authnz_pool

    user_id = await _create_auth_user(pool, prefix="akuser")

    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()

    # Create key
    created = await mgr.create_api_key(user_id=user_id, name="k1", description="d1", scope="read")
    key_id = created["id"]

    # Rotate key
    rotated = await mgr.rotate_api_key(key_id=key_id, user_id=user_id)
    assert rotated["id"] != key_id

    # Revoke new key
    ok = await mgr.revoke_api_key(rotated["id"], user_id=user_id, reason="cleanup")
    assert ok is True

    # Legacy compatibility audit mirror should still have entries.
    rows = await pool.fetchall("SELECT COUNT(*) AS c FROM api_key_audit_log")
    legacy_count = rows[0]["c"] if isinstance(rows[0], dict) else rows[0][0]
    assert legacy_count >= 3

    # Mandatory unified audit should persist in the real per-user audit DB.
    audit_db_path = DatabasePaths.get_audit_db_path(user_id)
    assert audit_db_path.exists(), f"Expected audit DB at {audit_db_path}"

    with sqlite3.connect(audit_db_path) as conn:
        conn.row_factory = sqlite3.Row
        unified_rows = conn.execute(
            "SELECT action, resource_id, metadata FROM audit_events ORDER BY timestamp ASC"
        ).fetchall()

    assert len(unified_rows) >= 3
    actions = [str(row["action"]) for row in unified_rows]
    assert "api_key.create" in actions
    assert "api_key.rotate" in actions
    assert "api_key.revoke" in actions

    rotate_row = next(row for row in unified_rows if row["action"] == "api_key.rotate")
    rotate_metadata = json.loads(rotate_row["metadata"] or "{}")
    assert int(rotate_metadata["old_key_id"]) == int(key_id)
    assert int(rotate_metadata["new_key_id"]) == int(rotated["id"])


@pytest.mark.asyncio
async def test_api_key_rotation_preserves_allowlists_and_metadata(sqlite_authnz_pool):
    pool = sqlite_authnz_pool

    user_id = await _create_auth_user(pool, prefix="akuser_meta")

    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()

    created = await mgr.create_api_key(
        user_id=user_id,
        name="k-meta",
        description="meta test",
        scope="read",
        allowed_ips=["127.0.0.1"],
        metadata={"purpose": "rotation-test"},
    )
    key_id = created["id"]

    rotated = await mgr.rotate_api_key(key_id=key_id, user_id=user_id)
    assert rotated["id"] != key_id

    row = await pool.fetchone(
        "SELECT allowed_ips, metadata FROM api_keys WHERE id = ?",
        rotated["id"],
    )
    if isinstance(row, dict):
        allowed_raw = row["allowed_ips"]
        metadata_raw = row["metadata"]
    else:
        allowed_raw, metadata_raw = row
    assert allowed_raw is not None
    assert metadata_raw is not None

    import json as _json

    allowed_ips = _json.loads(allowed_raw)
    metadata = _json.loads(metadata_raw)
    assert allowed_ips == ["127.0.0.1"]
    assert metadata.get("purpose") == "rotation-test"

    await pool.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
    await pool.execute("DELETE FROM users WHERE id = ?", (user_id,))


@pytest.mark.asyncio
async def test_create_api_key_rolls_back_when_mandatory_audit_fails(
    sqlite_authnz_pool,
    monkeypatch: pytest.MonkeyPatch,
):
    pool = sqlite_authnz_pool
    user_id = await _create_auth_user(pool, prefix="akuser_create_fail")
    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()
    _install_failing_mandatory_audit(monkeypatch)

    with pytest.raises(MandatoryAuditWriteError, match="Mandatory audit persistence unavailable"):
        await mgr.create_api_key(user_id=user_id, name="k-fail", description="d1", scope="read")

    rows = await pool.fetchall("SELECT id FROM api_keys WHERE user_id = ?", user_id)
    assert rows == []


@pytest.mark.asyncio
async def test_create_api_key_does_not_commit_authnz_state_before_mandatory_audit(
    sqlite_authnz_pool,
    monkeypatch: pytest.MonkeyPatch,
):
    pool = sqlite_authnz_pool
    user_id = await _create_auth_user(pool, prefix="akuser_uncommitted")
    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()
    observed: dict[str, int] = {}

    async def _fail_audit(*, resource_id: str | None = None, **_kwargs):
        assert resource_id is not None
        row = await pool.fetchone("SELECT COUNT(*) AS c FROM api_keys WHERE id = ?", int(resource_id))
        observed["visible_count"] = row["c"] if isinstance(row, dict) else row[0]
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    monkeypatch.setattr(
        api_key_manager_module,
        "emit_mandatory_api_key_management_audit",
        _fail_audit,
    )

    with pytest.raises(MandatoryAuditWriteError, match="Mandatory audit persistence unavailable"):
        await mgr.create_api_key(user_id=user_id, name="k-uncommitted", description="d1", scope="read")

    assert observed["visible_count"] == 0


@pytest.mark.asyncio
async def test_create_virtual_key_rolls_back_when_mandatory_audit_fails(
    sqlite_authnz_pool,
    monkeypatch: pytest.MonkeyPatch,
):
    pool = sqlite_authnz_pool
    user_id = await _create_auth_user(pool, prefix="akuser_virtual_fail")
    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()
    _install_failing_mandatory_audit(monkeypatch)

    with pytest.raises(MandatoryAuditWriteError, match="Mandatory audit persistence unavailable"):
        await mgr.create_virtual_key(user_id=user_id, name="vk-fail", allowed_endpoints=["chat.completions"])

    rows = await pool.fetchall(
        "SELECT id FROM api_keys WHERE user_id = ? AND COALESCE(is_virtual, 0) = 1",
        user_id,
    )
    assert rows == []


@pytest.mark.asyncio
async def test_create_virtual_key_writes_unified_audit_sqlite(sqlite_authnz_pool):
    pool = sqlite_authnz_pool
    user_id = await _create_auth_user(pool, prefix="akuser_virtual_audit")
    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()

    created = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-success",
        allowed_endpoints=["chat.completions"],
    )

    audit_db_path = DatabasePaths.get_audit_db_path(user_id)
    assert audit_db_path.exists(), f"Expected audit DB at {audit_db_path}"

    with sqlite3.connect(audit_db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT action, resource_id
            FROM audit_events
            WHERE action = 'api_key.create_virtual'
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row["action"] == "api_key.create_virtual"
    assert int(row["resource_id"]) == int(created["id"])


@pytest.mark.asyncio
async def test_legacy_api_key_audit_mirror_failure_is_non_blocking(
    sqlite_authnz_pool,
    monkeypatch: pytest.MonkeyPatch,
):
    pool = sqlite_authnz_pool
    user_id = await _create_auth_user(pool, prefix="akuser_legacy_fail")
    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()
    repo = mgr._get_repo()

    async def _failing_insert_audit_log(**_kwargs):
        raise RuntimeError("legacy mirror unavailable")

    monkeypatch.setattr(repo, "insert_audit_log", _failing_insert_audit_log)

    created = await mgr.create_api_key(user_id=user_id, name="k-legacy", description="d1", scope="read")
    assert int(created["id"]) > 0

    audit_db_path = DatabasePaths.get_audit_db_path(user_id)
    with sqlite3.connect(audit_db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT action, resource_id
            FROM audit_events
            WHERE action = 'api_key.create'
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row["action"] == "api_key.create"
    assert int(row["resource_id"]) == int(created["id"])


@pytest.mark.asyncio
async def test_rotate_api_key_rolls_back_when_mandatory_audit_fails(
    sqlite_authnz_pool,
    monkeypatch: pytest.MonkeyPatch,
):
    pool = sqlite_authnz_pool
    user_id = await _create_auth_user(pool, prefix="akuser_rotate_fail")
    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()

    created = await mgr.create_api_key(user_id=user_id, name="k1", description="d1", scope="read")
    original_key_id = int(created["id"])
    _install_failing_mandatory_audit(monkeypatch)

    with pytest.raises(MandatoryAuditWriteError, match="Mandatory audit persistence unavailable"):
        await mgr.rotate_api_key(key_id=original_key_id, user_id=user_id)

    rows = await pool.fetchall(
        """
        SELECT id, status, rotated_to, rotated_from, revoked_at
        FROM api_keys
        WHERE user_id = ?
        ORDER BY id ASC
        """,
        user_id,
    )
    assert len(rows) == 1
    row = rows[0]
    assert int(row["id"] if isinstance(row, dict) else row[0]) == original_key_id
    status_value = row["status"] if isinstance(row, dict) else row[1]
    rotated_to = row["rotated_to"] if isinstance(row, dict) else row[2]
    rotated_from = row["rotated_from"] if isinstance(row, dict) else row[3]
    revoked_at = row["revoked_at"] if isinstance(row, dict) else row[4]
    assert status_value == APIKeyStatus.ACTIVE.value
    assert rotated_to is None
    assert rotated_from is None
    assert revoked_at is None


@pytest.mark.asyncio
async def test_revoke_api_key_rolls_back_when_mandatory_audit_fails(
    sqlite_authnz_pool,
    monkeypatch: pytest.MonkeyPatch,
):
    pool = sqlite_authnz_pool
    user_id = await _create_auth_user(pool, prefix="akuser_revoke_fail")
    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()

    created = await mgr.create_api_key(user_id=user_id, name="k1", description="d1", scope="read")
    key_id = int(created["id"])
    _install_failing_mandatory_audit(monkeypatch)

    with pytest.raises(MandatoryAuditWriteError, match="Mandatory audit persistence unavailable"):
        await mgr.revoke_api_key(key_id, user_id=user_id, reason="cleanup")

    row = await pool.fetchone(
        "SELECT status, revoked_at, revoked_by, revoke_reason FROM api_keys WHERE id = ?",
        key_id,
    )
    if isinstance(row, dict):
        status_value = row["status"]
        revoked_at = row["revoked_at"]
        revoked_by = row["revoked_by"]
        revoke_reason = row["revoke_reason"]
    else:
        status_value, revoked_at, revoked_by, revoke_reason = row

    assert status_value == APIKeyStatus.ACTIVE.value
    assert revoked_at is None
    assert revoked_by is None
    assert revoke_reason is None


@pytest.mark.asyncio
async def test_rotate_api_key_rejects_revoked_source_key(sqlite_authnz_pool):
    pool = sqlite_authnz_pool
    user_id = await _create_auth_user(pool, prefix="akuser_rotate_revoked")
    mgr = APIKeyManager(db_pool=pool)
    await mgr.initialize()

    created = await mgr.create_api_key(user_id=user_id, name="k1", description="d1", scope="read")
    key_id = int(created["id"])
    revoked = await mgr.revoke_api_key(key_id, user_id=user_id, reason="cleanup")
    assert revoked is True

    with pytest.raises(ValueError, match="not found or unauthorized"):
        await mgr.rotate_api_key(key_id=key_id, user_id=user_id)
