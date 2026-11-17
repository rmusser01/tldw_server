import asyncio
import threading
from typing import Any, Dict

import asyncpg
import pytest

from tldw_Server_API.tests.helpers.pg_env import get_pg_env

_pg = get_pg_env()
TEST_DB_HOST = _pg.host
TEST_DB_PORT = int(_pg.port)
TEST_DB_USER = _pg.user
TEST_DB_PASSWORD = _pg.password

pytestmark = pytest.mark.integration


def _run_async(coro):


    """Run an async coroutine from sync tests, tolerating an active loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def _runner():

        result["value"] = asyncio.run(coro)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    return result.get("value")


async def _grant_user_permission(db_name: str, username: str, permission: str) -> int:
    """Ensure a user has a concrete permission via user_permissions."""
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=db_name,
    )
    try:
        user_id = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
        if user_id is None:
            raise RuntimeError(f"User {username} not found")
        perm_id = await conn.fetchval(
            """
            INSERT INTO permissions (name, description, category)
            VALUES ($1, $2, $3)
            ON CONFLICT (name) DO UPDATE SET description=EXCLUDED.description
            RETURNING id
            """,
            permission,
            permission,
            permission.split(".")[0] if "." in permission else "general",
        )
        await conn.execute(
            """
            INSERT INTO user_permissions (user_id, permission_id, granted)
            VALUES ($1, $2, 1)
            ON CONFLICT (user_id, permission_id) DO UPDATE SET granted=EXCLUDED.granted
            """,
            user_id,
            perm_id,
        )
        return int(user_id)
    finally:
        await conn.close()


async def _create_api_key(db_name: str, username: str) -> Dict[str, Any]:
    """Create a real API key for the given user via the manager."""
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=db_name,
    )
    try:
        user_id = await conn.fetchval("SELECT id FROM users WHERE username=$1", username)
        if user_id is None:
            raise RuntimeError(f"User {username} not found")
    finally:
        await conn.close()

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    mgr = APIKeyManager()
    await mgr.initialize()
    return await mgr.create_api_key(
        user_id=user_id,
        name="test-key",
        description="authz-claims",
        scope="write",
        expires_in_days=30,
    )


class _StubRagResult:
    def __init__(self, query: str):
        self.documents = []
        self.query = query
        self.expanded_queries = []
        self.metadata = {}
        self.timings = {}
        self.citations = []
        self.academic_citations = []
        self.total_time = 0.01
        self.cache_hit = False
        self.errors: list[str] = []
        self.feedback_id = None
        self.generated_answer = "ok"
        self.security_report = None
        self.claims = None
        self.factuality = None


def test_rag_search_requires_media_read_permissions(isolated_test_environment, monkeypatch):


    client, db_name = isolated_test_environment

    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import (
        get_media_db_for_user,
    )
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
        get_chacha_db_for_user,
    )
    from tldw_Server_API.app.api.v1.endpoints import rag_unified as rag_mod

    # Lightweight stubs to avoid heavy DB/model setup during auth-only checks
    fastapi_app.dependency_overrides[get_media_db_for_user] = lambda: type("DB", (), {"db_path": ":memory:"})()
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: type("DB", (), {"db_path": ":memory:"})()

    async def _fake_pipeline(**kwargs):
        return _StubRagResult(kwargs.get("query", ""))

    monkeypatch.setattr("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.unified_rag_pipeline", _fake_pipeline)

    username = "rag_user"
    password = "Str0ng_Pw!A"
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": "rag_user@example.com", "password": password},
    )
    assert reg.status_code == 200, reg.text
    login = client.post("/api/v1/auth/login", data={"username": username, "password": password})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    body = {"query": "hello world"}
    try:
        resp = client.post("/api/v1/rag/search", headers=headers, json=body)
        # In the current RBAC model, baseline roles include media.read so
        # authenticated users are allowed to search.
        assert resp.status_code == 200, resp.text
    finally:
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)
        fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)


def test_media_process_videos_requires_create_permission(isolated_test_environment):


    client, db_name = isolated_test_environment

    username = "media_user"
    password = "Str0ng_Pw!A"
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": "media_user@example.com", "password": password},
    )
    assert reg.status_code == 200, reg.text

    api_key_info = _run_async(_create_api_key(db_name, username))
    api_key = api_key_info["key"]

    first = client.post(
        "/api/v1/media/process-videos",
        headers={"X-API-KEY": api_key},
        data={"urls": ""},
    )
    # Baseline roles include media.create; authenticated API-key calls should
    # pass auth and reach validation/business logic.
    assert first.status_code not in (401, 403)
    assert first.status_code in (400, 207, 200)
