import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _base_env(tmp_path: Path):
    os.environ["AUTH_MODE"] = "multi_user"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-rag-media-claims-12345678901234567890"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users.db'}"
    os.environ["VIRTUAL_KEYS_ENABLED"] = "false"
    os.environ["LLM_BUDGET_ENFORCE"] = "false"


async def _seed_user(pool, username: str):
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            (username, f"{username}@example.com", "x"),
        )
    return await pool.fetchval("SELECT id FROM users WHERE username = ?", username)


async def _ensure_permission(pool, code: str) -> int:
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            (code, code, code.split(".")[0] if "." in code else "general"),
        )
    row = await pool.fetchrow("SELECT id FROM permissions WHERE name = ?", code)
    return int(row["id"] if isinstance(row, dict) else row[0])


async def _grant_user_permission(pool, user_id: int, perm_code: str):
    perm_id = await _ensure_permission(pool, perm_code)
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT OR REPLACE INTO user_permissions (user_id, permission_id, granted)
            VALUES (?, ?, 1)
            """,
            (user_id, perm_id),
        )


async def _create_api_key(user_id: int):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    mgr = APIKeyManager()
    await mgr.initialize()
    return await mgr.create_api_key(
        user_id=user_id,
        name="test-key",
        description="claims sqlite",
        scope="write",
        expires_in_days=30,
    )


def _override_rag(app, monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.endpoints import rag_unified as rag_mod

    app.dependency_overrides[get_media_db_for_user] = lambda: type("DB", (), {"db_path": ":memory:"})()
    app.dependency_overrides[get_chacha_db_for_user] = lambda: type("DB", (), {"db_path": ":memory:"})()

    async def _fake_pipeline(**kwargs):
        return rag_mod.UnifiedRAGResponse(documents=[], query=kwargs.get("query"), expanded_queries=[], metadata={})

    monkeypatch.setattr(rag_mod, "unified_rag_pipeline", _fake_pipeline)


@pytest.mark.asyncio
async def test_rag_search_claims_with_api_key_sqlite(tmp_path, monkeypatch):
    _base_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_mode, get_settings
    assert get_settings().AUTH_MODE == "multi_user"
    assert is_single_user_mode() is False

    user_id = await _seed_user(pool, "rag_user_sqlite")
    key_info = await _create_api_key(user_id)
    api_key = key_info["key"]

    from tldw_Server_API.app.main import app

    _override_rag(app, monkeypatch)
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal as _dep_principal
    from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import get_auth_principal as _resolve_principal

    captured: list = []
    force_no_perms = {"value": True}

    from fastapi import Request

    async def _capture_principal(request: Request):
        p = await _resolve_principal(request)
        if force_no_perms["value"]:
            p.permissions = []
            p.is_admin = False
        captured.append(p)
        return p

    app.dependency_overrides[_dep_principal] = _capture_principal
    from tldw_Server_API.app.core.config import settings as app_settings

    app_settings["CSRF_ENABLED"] = False

    try:
        with TestClient(app) as client:
            body = {"query": "hello world"}
            r_forbidden = client.post(
                "/api/v1/rag/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json=body,
            )
            assert r_forbidden.status_code == 403
            assert captured and captured[0].permissions == []

            await _grant_user_permission(pool, user_id, "media.read")
            force_no_perms["value"] = False
            captured.clear()

            r_ok = client.post(
                "/api/v1/rag/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json=body,
            )
            assert r_ok.status_code == 200, r_ok.text
    finally:
        app.dependency_overrides.clear()
        try:
            await pool.close()
        except Exception:
            _ = None
        await reset_db_pool()
        reset_settings()


@pytest.mark.asyncio
async def test_media_process_videos_requires_permission_sqlite(tmp_path):
    _base_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    user_id = await _seed_user(pool, "media_user_sqlite")
    key_info = await _create_api_key(user_id)
    api_key = key_info["key"]

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal as _dep_principal
    from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import get_auth_principal as _resolve_principal
    from fastapi import Request

    app_settings["CSRF_ENABLED"] = False

    captured: list = []
    force_no_perms = {"value": True}

    async def _capture_principal(request: Request):
        p = await _resolve_principal(request)
        if force_no_perms["value"]:
            p.permissions = []
            p.is_admin = False
        captured.append(p)
        return p

    app.dependency_overrides[_dep_principal] = _capture_principal

    try:
        with TestClient(app) as client:
            r_forbidden = client.post(
                "/api/v1/media/process-videos",
                headers={"X-API-KEY": api_key},
                data={"urls": ""},
            )
            assert r_forbidden.status_code == 403

            await _grant_user_permission(pool, user_id, "media.create")
            force_no_perms["value"] = False

            r_after = client.post(
                "/api/v1/media/process-videos",
                headers={"X-API-KEY": api_key},
                data={"urls": ""},
            )
            # Payload is intentionally minimal; allow auth to pass and surface validation status
            assert r_after.status_code in (400, 207, 200)
    finally:
        try:
            await pool.close()
        except Exception:
            _ = None
        await reset_db_pool()
        reset_settings()
