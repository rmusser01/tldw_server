import contextlib
import os
import asyncio
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables


client = TestClient(app)


def _write_run_command_test_config(tmp_path: Path) -> Path:
    """Write an explicit opt-in MCP module config for tests that execute `run`."""
    config_path = tmp_path / "mcp_modules_with_run.yaml"
    config_path.write_text(
        """
modules:
  - id: template
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.template_module:TemplateModule
    enabled: true
    name: Template
    version: "1.0.0"
    department: demo
    settings: {}
  - id: filesystem
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module:FilesystemModule
    enabled: true
    name: Filesystem
    version: "1.0.0"
    department: system
    settings: {}
  - id: knowledge
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.knowledge_module:KnowledgeModule
    enabled: true
    name: Knowledge
    version: "1.0.0"
    department: knowledge
    settings: {}
  - id: run_command
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module:RunCommandModule
    enabled: true
    name: Run Command
    version: "0.1.0"
    department: system
    settings:
      spill_dir: .mcp-test-spills
      spill_threshold_bytes: 65536
      preview_line_limit: 200
      preview_byte_limit: 51200
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def _run(coro):


    return asyncio.run(coro)


def test_tools_execute_unauth_401():


    payload = {"tool_name": "echo", "arguments": {"message": "hi"}}
    r = client.post("/api/v1/mcp/tools/execute", json=payload)
    assert r.status_code == 401, r.text


def test_tools_execute_with_bearer_token_no_permission_403():


    # Use MCP JWT (auto-seeded secret) to authenticate
    mgr = get_jwt_manager()
    token = mgr.create_access_token(subject="42", username="tester")

    payload = {"tool_name": "echo", "arguments": {"message": "hi"}}
    r = client.post(
        "/api/v1/mcp/tools/execute",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403, r.text
    data = r.json()
    assert "detail" in data and isinstance(data["detail"], dict)
    hint = data["detail"].get("hint", "")
    # Should recommend assigning tools.execute:<tool> or wildcard
    assert "tools.execute:echo" in hint or "tools.execute:*" in hint


def test_tools_execute_with_api_key_and_role_permission_allows_200(tmp_path, monkeypatch):


    from tldw_Server_API.app.core.MCP_unified.server import reset_mcp_server

    # Point AuthNZ DB to a fresh SQLite file
    db_file = tmp_path / "mcp_allow.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(_write_run_command_test_config(tmp_path)))
    monkeypatch.setenv("MCP_MODULES", "")
    # Reset settings and DB pool to pick up new config
    _run(reset_db_pool())
    reset_settings()
    _run(reset_mcp_server())

    # Run AuthNZ migrations (creates RBAC tables and expands api_keys schema)
    ensure_authnz_tables(Path(db_file))
    # Insert a user directly (compatible with base SQLite schema)
    pool = _run(get_db_pool())
    async def _insert_user():
        async with pool.transaction() as conn:
            if hasattr(conn, 'fetchval'):
                uid = await conn.fetchval(
                    "INSERT INTO users (username, email, password_hash, is_active, role, is_verified) VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                    "permit_user", "permit@test.local", "dummyhash", True, "user", True
                )
                return uid
            else:
                cur = await conn.execute(
                    "INSERT INTO users (username, email, password_hash, is_active, role, is_verified) VALUES (?,?,?,?,?,?)",
                    ("permit_user", "permit@test.local", "dummyhash", 1, "user", 1)
                )
                uid = cur.lastrowid
                await conn.commit()
                return uid
    user_id = _run(_insert_user())
    api_mgr = _run(get_api_key_manager())
    key_data = _run(api_mgr.create_api_key(user_id=user_id, name="permit-key"))
    api_key = key_data["key"]

    # Insert wildcard tools permission, role, and assign to user
    pool = _run(get_db_pool())

    async def _seed():
        async with pool.transaction() as conn:
            # Create RBAC core tables if they don't exist
            if not hasattr(conn, 'fetchval'):
                await conn.execute("CREATE TABLE IF NOT EXISTS roles (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, description TEXT, is_system INTEGER DEFAULT 0)")
                await conn.execute("CREATE TABLE IF NOT EXISTS permissions (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, description TEXT, category TEXT)")
                await conn.execute("CREATE TABLE IF NOT EXISTS role_permissions (role_id INTEGER NOT NULL, permission_id INTEGER NOT NULL, PRIMARY KEY(role_id, permission_id))")
                await conn.execute("CREATE TABLE IF NOT EXISTS user_roles (user_id INTEGER NOT NULL, role_id INTEGER NOT NULL, PRIMARY KEY(user_id, role_id))")
            # Create permission if missing
            if hasattr(conn, 'fetchval'):
                pid = await conn.fetchval(
                    "INSERT INTO permissions (name, description, category) VALUES ($1,$2,$3) ON CONFLICT (name) DO NOTHING RETURNING id",
                    "tools.execute:*", "Wildcard tool execution", "tools"
                )
                if not pid:
                    pid = await conn.fetchval("SELECT id FROM permissions WHERE name = $1", "tools.execute:*")
                modules_read_pid = await conn.fetchval(
                    "INSERT INTO permissions (name, description, category) VALUES ($1,$2,$3) ON CONFLICT (name) DO NOTHING RETURNING id",
                    "modules.read", "Read MCP modules", "modules"
                )
                if not modules_read_pid:
                    modules_read_pid = await conn.fetchval("SELECT id FROM permissions WHERE name = $1", "modules.read")
                rid = await conn.fetchval(
                    "INSERT INTO roles (name, description, is_system) VALUES ($1,$2,$3) RETURNING id",
                    "tool_role", "Role for tool exec", False
                )
                await conn.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES ($1,$2) ON CONFLICT DO NOTHING", rid, pid)
                await conn.execute(
                    "INSERT INTO role_permissions (role_id, permission_id) VALUES ($1,$2) ON CONFLICT DO NOTHING",
                    rid,
                    modules_read_pid,
                )
                await conn.execute("INSERT INTO user_roles (user_id, role_id) VALUES ($1,$2) ON CONFLICT DO NOTHING", user_id, rid)
            else:
                # SQLite
                cur = await conn.execute("SELECT id FROM permissions WHERE name = ?", ("tools.execute:*",))
                row = await cur.fetchone()
                if row:
                    pid = row[0]
                else:
                    cur = await conn.execute(
                        "INSERT INTO permissions (name, description, category) VALUES (?,?,?)",
                        ("tools.execute:*", "Wildcard tool execution", "tools")
                    )
                    pid = cur.lastrowid
                cur = await conn.execute("SELECT id FROM permissions WHERE name = ?", ("modules.read",))
                row = await cur.fetchone()
                if row:
                    modules_read_pid = row[0]
                else:
                    cur = await conn.execute(
                        "INSERT INTO permissions (name, description, category) VALUES (?,?,?)",
                        ("modules.read", "Read MCP modules", "modules")
                    )
                    modules_read_pid = cur.lastrowid
                cur = await conn.execute(
                    "INSERT INTO roles (name, description, is_system) VALUES (?,?,?)",
                    ("tool_role", "Role for tool exec", 0)
                )
                rid = cur.lastrowid
                await conn.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (?,?)", (rid, pid))
                await conn.execute(
                    "INSERT INTO role_permissions (role_id, permission_id) VALUES (?,?)",
                    (rid, modules_read_pid),
                )
                await conn.execute("INSERT INTO user_roles (user_id, role_id) VALUES (?,?)", (user_id, rid))
                await conn.commit()

    _run(_seed())

    # Call tools/execute with API key
    payload = {"tool_name": "echo", "arguments": {"message": "hello"}}
    r = client.post(
        "/api/v1/mcp/tools/execute",
        json=payload,
        headers={"X-API-KEY": api_key},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "hello"

    async def _fake_workspace_root(self, **kwargs):  # noqa: ARG001
        return {
            "workspace_root": str(tmp_path),
            "workspace_id": "test-workspace",
            "source": "test",
            "reason": None,
        }

    from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
        McpHubWorkspaceRootResolver,
    )

    monkeypatch.setattr(
        McpHubWorkspaceRootResolver,
        "resolve_for_context",
        _fake_workspace_root,
    )

    run_payload = {"tool_name": "run", "arguments": {"command": "ls"}}
    run_response = client.post(
        "/api/v1/mcp/tools/execute",
        json=run_payload,
        headers={"X-API-KEY": api_key},
    )
    assert run_response.status_code == 200, run_response.text
    run_body = run_response.json()
    assert "[exit:0 |" in run_body["result"]


def test_tools_execute_with_api_key_can_run_virtual_cli_help(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.MCP_unified.config import get_config
    from tldw_Server_API.app.core.MCP_unified.server import get_mcp_server, reset_mcp_server

    db_file = tmp_path / "mcp_run_help.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(_write_run_command_test_config(tmp_path)))
    monkeypatch.setenv("MCP_MODULES", "")

    _run(reset_db_pool())
    reset_settings()
    with contextlib.suppress(AttributeError):
        get_config.cache_clear()  # type: ignore[attr-defined]
    _run(reset_mcp_server())

    ensure_authnz_tables(Path(db_file))
    pool = _run(get_db_pool())

    async def _insert_user():
        async with pool.transaction() as conn:
            if hasattr(conn, 'fetchval'):
                uid = await conn.fetchval(
                    "INSERT INTO users (username, email, password_hash, is_active, role, is_verified) VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                    "run_user", "run@test.local", "dummyhash", True, "user", True
                )
                return uid
            else:
                cur = await conn.execute(
                    "INSERT INTO users (username, email, password_hash, is_active, role, is_verified) VALUES (?,?,?,?,?,?)",
                    ("run_user", "run@test.local", "dummyhash", 1, "user", 1)
                )
                uid = cur.lastrowid
                await conn.commit()
                return uid

    user_id = _run(_insert_user())
    api_mgr = _run(get_api_key_manager())
    key_data = _run(api_mgr.create_api_key(user_id=user_id, name="run-key"))
    api_key = key_data["key"]

    async def _seed():
        async with pool.transaction() as conn:
            if not hasattr(conn, 'fetchval'):
                await conn.execute("CREATE TABLE IF NOT EXISTS roles (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, description TEXT, is_system INTEGER DEFAULT 0)")
                await conn.execute("CREATE TABLE IF NOT EXISTS permissions (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, description TEXT, category TEXT)")
                await conn.execute("CREATE TABLE IF NOT EXISTS role_permissions (role_id INTEGER NOT NULL, permission_id INTEGER NOT NULL, PRIMARY KEY(role_id, permission_id))")
                await conn.execute("CREATE TABLE IF NOT EXISTS user_roles (user_id INTEGER NOT NULL, role_id INTEGER NOT NULL, PRIMARY KEY(user_id, role_id))")
            if hasattr(conn, 'fetchval'):
                pid = await conn.fetchval(
                    "INSERT INTO permissions (name, description, category) VALUES ($1,$2,$3) ON CONFLICT (name) DO NOTHING RETURNING id",
                    "tools.execute:*", "Wildcard tool execution", "tools"
                )
                if not pid:
                    pid = await conn.fetchval("SELECT id FROM permissions WHERE name = $1", "tools.execute:*")
                rid = await conn.fetchval(
                    "INSERT INTO roles (name, description, is_system) VALUES ($1,$2,$3) RETURNING id",
                    "run_tool_role", "Role for run tool exec", False
                )
                await conn.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES ($1,$2) ON CONFLICT DO NOTHING", rid, pid)
                await conn.execute("INSERT INTO user_roles (user_id, role_id) VALUES ($1,$2) ON CONFLICT DO NOTHING", user_id, rid)
            else:
                cur = await conn.execute("SELECT id FROM permissions WHERE name = ?", ("tools.execute:*",))
                row = await cur.fetchone()
                if row:
                    pid = row[0]
                else:
                    cur = await conn.execute(
                        "INSERT INTO permissions (name, description, category) VALUES (?,?,?)",
                        ("tools.execute:*", "Wildcard tool execution", "tools")
                    )
                    pid = cur.lastrowid
                cur = await conn.execute(
                    "INSERT INTO roles (name, description, is_system) VALUES (?,?,?)",
                    ("run_tool_role", "Role for run tool exec", 0)
                )
                rid = cur.lastrowid
                await conn.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (?,?)", (rid, pid))
                await conn.execute("INSERT INTO user_roles (user_id, role_id) VALUES (?,?)", (user_id, rid))
                await conn.commit()

    _run(_seed())

    payload = {"tool_name": "run", "arguments": {"command": "help"}}
    r = client.post(
        "/api/v1/mcp/tools/execute",
        json=payload,
        headers={"X-API-KEY": api_key},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["module"] == "run_command"
    assert isinstance(body["result"], str)
    assert "Virtual CLI commands available in this context" in body["result"]
    assert "[exit:0 |" in body["result"]

    module_ids = set(_run(get_mcp_server().module_registry.get_all_modules()).keys())
    assert {"filesystem", "knowledge", "run_command"}.issubset(module_ids)
