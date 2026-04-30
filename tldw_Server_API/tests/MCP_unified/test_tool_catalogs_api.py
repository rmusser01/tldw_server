import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest


_ALLOWED_USERS_INSERT_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "uuid",
        "username",
        "email",
        "password_hash",
        "metadata",
        "is_active",
        "is_superuser",
        "is_verified",
        "role",
        "created_at",
        "updated_at",
        "last_login",
        "email_verified",
        "storage_quota_mb",
        "storage_used_mb",
        "failed_login_attempts",
        "locked_until",
    }
)


def _ensure_single_user_row() -> None:
    """Ensure the fixed single-user admin row exists in the local test users DB.

    Side effects:
    - Creates `Databases/users.db` (and parent directory) if missing.
    - Ensures a minimal `users` table exists.
    - Inserts the configured single-user row when absent.
    """
    db_path: Path = Path("Databases/users.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        uid: int = int(os.getenv("SINGLE_USER_FIXED_ID", "1"))
        # Ensure users table exists minimally
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                is_superuser INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0,
                role TEXT DEFAULT 'admin',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                email_verified INTEGER DEFAULT 0,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb INTEGER DEFAULT 0
            )
            """
        )
        columns: dict[str, dict[str, Any]] = {
            row[1]: {"type": row[2], "notnull": row[3], "default": row[4]}
            for row in conn.execute("PRAGMA table_info(users)")
        }
        cur: sqlite3.Cursor = conn.execute("SELECT id FROM users WHERE id = ?", (uid,))
        row: tuple[Any, ...] | None = cur.fetchone()
        if row is None:
            insert_cols: list[str] = []
            insert_vals: list[Any] = []

            def _add(col: str, val: Any) -> None:
                """Append column/value only when the column exists in current schema."""
                if col in columns:
                    insert_cols.append(col)
                    insert_vals.append(val)

            _add("id", uid)
            _add("username", "single_user")
            _add("email", "single@example.com")
            _add("password_hash", "x")
            _add("role", "admin")
            _add("is_active", 1)
            _add("is_superuser", 1)
            _add("is_verified", 1)
            _add("email_verified", 1)

            def _default_for_type(col_type: str) -> Any:
                """Map SQLite type names to deterministic placeholder defaults."""
                t = (col_type or "").upper()
                if "INT" in t:
                    return 0
                if "REAL" in t or "FLOA" in t or "DOUB" in t:
                    return 0.0
                return "x"

            for name, meta in columns.items():
                if meta["notnull"] and meta["default"] is None and name not in insert_cols:
                    insert_cols.append(name)
                    insert_vals.append(_default_for_type(meta["type"]))

            unexpected_insert_cols: list[str] = [
                name for name in insert_cols if name not in _ALLOWED_USERS_INSERT_COLUMNS
            ]
            if unexpected_insert_cols:
                raise ValueError(
                    f"Unexpected users insert columns in test bootstrap: {unexpected_insert_cols!r}"
                )
            placeholders: str = ", ".join("?" for _ in insert_cols)
            # Dynamic identifiers are safe here because insert_cols are validated
            # against _ALLOWED_USERS_INSERT_COLUMNS immediately above.
            conn.execute(
                f"INSERT INTO users ({', '.join(insert_cols)}) VALUES ({placeholders})",  # nosec B608
                insert_vals,
            )


@pytest.mark.integration
def test_tool_catalogs_flow():
    # Configure test env
    os.environ["TEST_MODE"] = "true"
    os.environ["MCP_ALLOWED_IPS"] = "[]"
    # Single-user mode API key; settings will normalize for testing
    os.environ.setdefault("SINGLE_USER_API_KEY", "CHANGE_ME_TO_SECURE_API_KEY")
    # Ensure Media module autoloads for MCP
    os.environ["MCP_ENABLE_MEDIA_MODULE"] = "true"

    # Clear cached MCP config and IP allowlist controller to pick up env
    try:
        from tldw_Server_API.app.core.MCP_unified.config import get_config
        get_config.cache_clear()
        from tldw_Server_API.app.core.MCP_unified.security.ip_filter import get_ip_access_controller
        get_ip_access_controller.cache_clear()
    except Exception:
        _ = None

    # Start app
    from fastapi.testclient import TestClient
    from tldw_Server_API.app.main import app

    # Disable HTTP security guard for this test (IP allowlist/mTLS)
    try:
        from tldw_Server_API.app.core.MCP_unified.security.request_guards import enforce_http_security as _ehs
        app.dependency_overrides[_ehs] = lambda: None
    except Exception:
        _ = None
    client = TestClient(app)

    # Prepare single-user row so role assignments don't fail silently
    _ensure_single_user_row()

    # Use configured test API key
    api_key = os.getenv("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
    headers = {"X-API-KEY": api_key}

    # Assign admin role to single user to satisfy module read checks where applicable
    # Find admin role id
    r_roles = client.get("/api/v1/admin/roles", headers=headers)
    assert r_roles.status_code == 200, r_roles.text
    roles = r_roles.json()
    admin_role_id = None
    for r in roles:
        if r.get("name") == "admin":
            admin_role_id = r.get("id")
            break
    assert admin_role_id is not None
    uid = int(os.getenv("SINGLE_USER_FIXED_ID", "1"))
    client.post(f"/api/v1/admin/users/{uid}/roles/{admin_role_id}", headers=headers)

    # Create catalog
    cat_name = "integration-cat-A"
    r_create = client.post(
        "/api/v1/admin/mcp/tool_catalogs",
        headers=headers,
        json={"name": cat_name, "description": "demo", "is_active": True},
    )
    assert r_create.status_code in (200, 201, 409), r_create.text
    if r_create.status_code == 201:
        catalog_id = r_create.json()["id"]
    else:
        # Fetch existing
        r_list = client.get(
            "/api/v1/admin/mcp/tool_catalogs", headers=headers, params={"limit": 100}
        )
        assert r_list.status_code == 200, r_list.text
        catalog_id = next(c["id"] for c in r_list.json() if c["name"] == cat_name)

    # Add only media.search to catalog
    r_add = client.post(
        f"/api/v1/admin/mcp/tool_catalogs/{catalog_id}/entries",
        headers=headers,
        json={"tool_name": "media.search"},
    )
    assert r_add.status_code in (200, 201), r_add.text

    # List tools filtered by catalog name; expect media.search present
    r_tools = client.get(
        "/api/v1/mcp/tools",
        headers={**headers, "X-Real-IP": "127.0.0.1"},
        params={"catalog": cat_name},
    )
    assert r_tools.status_code == 200, r_tools.text
    data = r_tools.json()
    assert "tools" in data and isinstance(data["tools"], list)
    names = {t.get("name") for t in data["tools"]}
    assert "media.search" in names

    # Ensure a write tool not added (e.g., ingest_media) is not present when filtering by catalog
    assert "ingest_media" not in names


def _create_user(conn: sqlite3.Connection, username: str, email: str, role: str = "user") -> int:
    try:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, role, is_active, is_verified) VALUES (?, ?, 'x', ?, 1, 1)",
            (username, email, role),
        )
        return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        # Fall back to a unique email by suffixing a counter
        base, at, domain = email.partition("@")
        suffix = 1
        while True:
            new_email = f"{base}+{suffix}@{domain}" if at else f"{email}.{suffix}"
            new_username = f"{username}{suffix}"
            try:
                cur = conn.execute(
                    "INSERT INTO users (username, email, password_hash, role, is_active, is_verified) VALUES (?, ?, 'x', ?, 1, 1)",
                    (new_username, new_email, role),
                )
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                suffix += 1


def _get_db_path_from_env() -> Path:


    url = os.getenv("DATABASE_URL", "sqlite:///./Databases/users.db")
    # crude parse for sqlite file paths
    if url.startswith("sqlite:///"):
        return Path(url.replace("sqlite:///", ""))
    return Path("Databases/users.db")


def _ensure_tables_for_users():


    db_path = _get_db_path_from_env()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                is_active INTEGER DEFAULT 1,
                is_superuser INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                email_verified INTEGER DEFAULT 0,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb INTEGER DEFAULT 0
            )
            """
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_org_team_scoped_catalog_management(tmp_path, monkeypatch):
    # Configure test env
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-mcp-catalogs-multi-user-secret-1234567890")
    monkeypatch.setenv("MCP_ALLOWED_IPS", "[]")
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "true")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users_mcp_catalogs_test.db'}")
    monkeypatch.setenv("VIRTUAL_KEYS_ENABLED", "true")

    # Reset settings and DB pool to honor DATABASE_URL for this test
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Clear cached MCP config and IP allowlist controller to pick up env
    try:
        from tldw_Server_API.app.core.MCP_unified.config import get_config
        get_config.cache_clear()
        from tldw_Server_API.app.core.MCP_unified.security.ip_filter import get_ip_access_controller
        get_ip_access_controller.cache_clear()
    except Exception:
        _ = None

    # Start app
    from fastapi.testclient import TestClient
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.services.app_lifecycle import reset_lifecycle_state

    reset_lifecycle_state(app)
    client = TestClient(app)

    # Ensure base tables and concrete multi-user principals.
    _ensure_tables_for_users()
    db_path = _get_db_path_from_env()
    with sqlite3.connect(str(db_path)) as conn:
        admin_id = _create_user(conn, "admin_tcat", "admin@example.com", role="admin")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        updates = []
        values: list[Any] = []
        for column in ("is_superuser", "is_verified", "email_verified"):
            if column in columns:
                updates.append(f"{column} = ?")
                values.append(1)
        if updates:
            values.append(admin_id)
            conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", values)  # nosec B608
        robert_id = _create_user(conn, "robert_tcat", "robert@example.com")
        sally_id = _create_user(conn, "sally_tcat", "sally@example.com")

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    api_key_manager = APIKeyManager()
    await api_key_manager.initialize()
    admin_virtual_key = await api_key_manager.create_virtual_key(
        user_id=admin_id,
        name="vk-admin-catalog-fixture",
        scope="admin",
    )
    admin_headers = {"X-API-KEY": admin_virtual_key["key"]}

    # Provision API keys for both users via admin endpoints
    r_vk_r = client.post(f"/api/v1/admin/users/{robert_id}/virtual-keys", headers=admin_headers, json={"name": "vk-robert"})
    assert r_vk_r.status_code == 200, r_vk_r.text
    robert_key = r_vk_r.json().get("key")
    assert isinstance(robert_key, str) and len(robert_key) > 8

    r_vk_s = client.post(f"/api/v1/admin/users/{sally_id}/virtual-keys", headers=admin_headers, json={"name": "vk-sally"})
    assert r_vk_s.status_code == 200, r_vk_s.text
    sally_key = r_vk_s.json().get("key")
    assert isinstance(sally_key, str) and len(sally_key) > 8

    robert_headers = {"X-API-KEY": robert_key}
    sally_headers = {"X-API-KEY": sally_key}

    # Create an org and a team (admin)
    org_name = f"Org-Cats-{robert_id}"
    r_org = client.post("/api/v1/admin/orgs", headers=admin_headers, json={"name": org_name})
    assert r_org.status_code == 200, r_org.text
    org_id = int(r_org.json()["id"])

    team_name = f"Team-Cats-{robert_id}"
    r_team = client.post(f"/api/v1/admin/orgs/{org_id}/teams", headers=admin_headers, json={"name": team_name})
    assert r_team.status_code == 200, r_team.text
    team_id = int(r_team.json()["id"])

    # Add org/team memberships: Robert as lead, Sally as member
    r_add_org_robert = client.post(
        f"/api/v1/admin/orgs/{org_id}/members", headers=admin_headers, json={"user_id": robert_id, "role": "lead"}
    )
    assert r_add_org_robert.status_code == 200, r_add_org_robert.text

    r_add_org_sally = client.post(
        f"/api/v1/admin/orgs/{org_id}/members", headers=admin_headers, json={"user_id": sally_id, "role": "member"}
    )
    assert r_add_org_sally.status_code == 200, r_add_org_sally.text

    r_add_team_robert = client.post(
        f"/api/v1/admin/teams/{team_id}/members", headers=admin_headers, json={"user_id": robert_id, "role": "lead"}
    )
    assert r_add_team_robert.status_code == 200, r_add_team_robert.text

    r_add_team_sally = client.post(
        f"/api/v1/admin/teams/{team_id}/members", headers=admin_headers, json={"user_id": sally_id, "role": "member"}
    )
    assert r_add_team_sally.status_code == 200, r_add_team_sally.text

    # Robert (manager) creates an org catalog
    org_cat_name = "org-scoped-cat"
    r_create_org_cat = client.post(
        f"/api/v1/orgs/{org_id}/mcp/tool_catalogs", headers=robert_headers, json={"name": org_cat_name, "description": "demo"}
    )
    assert r_create_org_cat.status_code in (201, 409), r_create_org_cat.text
    if r_create_org_cat.status_code == 201:
        org_cat_id = int(r_create_org_cat.json()["id"])
    else:
        # list to find id
        r_list_org = client.get(f"/api/v1/orgs/{org_id}/mcp/tool_catalogs", headers=robert_headers)
        assert r_list_org.status_code == 200
        org_cat_id = next(c["id"] for c in r_list_org.json() if c["name"] == org_cat_name)

    # Robert adds entry to org catalog
    r_add_entry = client.post(
        f"/api/v1/orgs/{org_id}/mcp/tool_catalogs/{org_cat_id}/entries", headers=robert_headers, json={"tool_name": "media.search"}
    )
    assert r_add_entry.status_code in (200, 201), r_add_entry.text

    # Sally (non-manager) cannot create org catalog
    r_create_org_cat_forbidden = client.post(
        f"/api/v1/orgs/{org_id}/mcp/tool_catalogs", headers=sally_headers, json={"name": "forbidden", "description": "x"}
    )
    assert r_create_org_cat_forbidden.status_code == 403

    # Robert can delete entry and catalog
    r_del_entry = client.delete(
        f"/api/v1/orgs/{org_id}/mcp/tool_catalogs/{org_cat_id}/entries/media.search", headers=robert_headers
    )
    assert r_del_entry.status_code == 200, r_del_entry.text

    r_del_cat = client.delete(
        f"/api/v1/orgs/{org_id}/mcp/tool_catalogs/{org_cat_id}", headers=robert_headers
    )
    assert r_del_cat.status_code == 200, r_del_cat.text

    # Team-scoped: Robert creates, Sally forbidden
    team_cat_name = "team-scoped-cat"
    r_create_team_cat = client.post(
        f"/api/v1/teams/{team_id}/mcp/tool_catalogs", headers=robert_headers, json={"name": team_cat_name, "description": "demo"}
    )
    assert r_create_team_cat.status_code in (201, 409), r_create_team_cat.text
    if r_create_team_cat.status_code == 201:
        team_cat_id = int(r_create_team_cat.json()["id"])
    else:
        r_list_team = client.get(f"/api/v1/teams/{team_id}/mcp/tool_catalogs", headers=robert_headers)
        assert r_list_team.status_code == 200
        team_cat_id = next(c["id"] for c in r_list_team.json() if c["name"] == team_cat_name)

    r_add_team_entry = client.post(
        f"/api/v1/teams/{team_id}/mcp/tool_catalogs/{team_cat_id}/entries", headers=robert_headers, json={"tool_name": "ingest_media"}
    )
    assert r_add_team_entry.status_code in (200, 201)

    r_sally_add_team_entry = client.post(
        f"/api/v1/teams/{team_id}/mcp/tool_catalogs/{team_cat_id}/entries", headers=sally_headers, json={"tool_name": "media.search"}
    )
    assert r_sally_add_team_entry.status_code == 403

    # Catalog not found scope errors: try deleting team catalog as org scoped
    r_del_wrong_scope = client.delete(
        f"/api/v1/orgs/{org_id}/mcp/tool_catalogs/{team_cat_id}", headers=robert_headers
    )
    assert r_del_wrong_scope.status_code == 404

    r_del_wrong_entry = client.delete(
        f"/api/v1/orgs/{org_id}/mcp/tool_catalogs/{team_cat_id}/entries/ingest_media", headers=robert_headers
    )
    assert r_del_wrong_entry.status_code == 404

    # Clean up: delete team entry and catalog with manager
    r_del_team_entry = client.delete(
        f"/api/v1/teams/{team_id}/mcp/tool_catalogs/{team_cat_id}/entries/ingest_media", headers=robert_headers
    )
    assert r_del_team_entry.status_code == 200

    r_del_team_cat = client.delete(
        f"/api/v1/teams/{team_id}/mcp/tool_catalogs/{team_cat_id}", headers=robert_headers
    )
    assert r_del_team_cat.status_code == 200
