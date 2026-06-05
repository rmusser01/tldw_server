import os
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient


def _jwt_service():
    from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
    return get_jwt_service()


def _app():
    from tldw_Server_API.app.main import app
    return app


def _chat_stub_response():
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _set_env_vars(values: dict[str, str]) -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        os.environ[key] = value
    return previous


def _restore_env_vars(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _file_backed_chacha_override(
    db_path: Path,
    *,
    client_id: str,
) -> tuple[Callable[..., Any], Callable[[], None]]:
    """Return a file-backed ChaChaNotes override and cleanup callback."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    opened_dbs: list[CharactersRAGDB] = []

    async def _override_chacha_db_for_user(current_user=None):
        db = CharactersRAGDB(db_path=str(db_path), client_id=client_id)
        opened_dbs.append(db)
        return db

    def _cleanup() -> None:
        for db in opened_dbs:
            db.close_all_connections()

    return _override_chacha_db_for_user, _cleanup


@pytest.mark.asyncio
async def test_jwt_quota_enforced_for_chat_and_rag_sqlite(monkeypatch, tmp_path):
    # Multi-user mode to ensure JWTs are treated as Bearer tokens.
    old_env = _set_env_vars(
        {
            "AUTH_MODE": "multi_user",
            "DATABASE_URL": f"sqlite:///{tmp_path/'users.db'}",
            "OPENAI_API_KEY": "test",
        }
    )

    # Reset AuthNZ singletons
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import reset_api_key_manager
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()
    await reset_api_key_manager()
    fastapi_app = None
    chacha_deps = None
    cleanup_chacha = None
    try:
        pool = await get_db_pool()
        ensure_authnz_tables(Path(pool.db_path))

        # Create a user for JWT auth
        async with pool.transaction() as conn:
            await conn.execute(
                "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
                ("su", "su@example.com", "x"),
            )
        user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "su")

        # Monkeypatch chat API call at endpoint layer to avoid external calls
        import tldw_Server_API.app.api.v1.endpoints.chat as chat_endpoint
        monkeypatch.setattr(chat_endpoint, 'perform_chat_api_call', lambda **kwargs: _chat_stub_response())
        # Provide dummy provider keys; patch both module-level and schema-level maps
        import tldw_Server_API.app.api.v1.schemas.chat_request_schemas as schema_chat
        chat_endpoint.API_KEYS = {**(chat_endpoint.API_KEYS or {}), 'openai': 'test'}
        schema_chat.API_KEYS = {**(schema_chat.API_KEYS or {}), 'openai': 'test'}

        # Mint a virtual JWT with max_calls=1 for chat.completions
        svc = _jwt_service()
        token_chat = svc.create_virtual_access_token(
            user_id=int(user_id),
            username="su",
            role="user",
            scope="any",
            ttl_minutes=10,
            additional_claims={
                "allowed_endpoints": ["chat.completions"],
                "allowed_paths": ["/api/v1/chat/completions"],
                "max_calls": 1,
            },
        )

        headers = {"Authorization": f"Bearer {token_chat}"}
        body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}

        # Avoid exercising the real ChaChaNotes executor in this SQLite-focused
        # test by overriding the ChaChaNotes DB dependency with a lightweight
        # temp-file CharactersRAGDB. A file-backed DB is required because the
        # chat endpoint persists through executor threads, and SQLite :memory:
        # databases are scoped to each connection/thread.
        from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
        from tldw_Server_API.app.main import app as fastapi_app

        _override_chacha_db_for_user, cleanup_chacha = _file_backed_chacha_override(
            tmp_path / "jwt_quota_chacha.db",
            client_id="test-jwt-quota",
        )
        fastapi_app.dependency_overrides[chacha_deps.get_chacha_db_for_user] = _override_chacha_db_for_user

        with TestClient(_app()) as client:
            r1 = client.post("/api/v1/chat/completions", headers=headers, json=body)
            assert r1.status_code == 200, r1.text
            r2 = client.post("/api/v1/chat/completions", headers=headers, json=body)
            assert r2.status_code == 403
            # Mint a virtual JWT for RAG search with max_calls=1
            token_rag = svc.create_virtual_access_token(
                user_id=int(user_id),
                username="su",
                role="user",
                scope="any",
                ttl_minutes=10,
                additional_claims={
                    "allowed_endpoints": ["rag.search"],
                    "allowed_paths": ["/api/v1/rag/search"],
                    "max_calls": 1,
                },
            )
            headers_rag = {"Authorization": f"Bearer {token_rag}"}
            rag_body = {
                "query": "hello",
                "top_k": 1,
                # Force FTS-only retrieval so the quota test avoids pulling vector models.
                "search_mode": "fts",
                "enable_generation": False,
                "enable_reranking": False,
                "enable_cache": False,
            }
            r1 = client.post("/api/v1/rag/search", headers=headers_rag, json=rag_body)
            assert r1.status_code == 200, r1.text
            r2 = client.post("/api/v1/rag/search", headers=headers_rag, json=rag_body)
            assert r2.status_code == 403
    finally:
        if fastapi_app is not None and chacha_deps is not None:
            fastapi_app.dependency_overrides.pop(chacha_deps.get_chacha_db_for_user, None)
        if cleanup_chacha is not None:
            cleanup_chacha()
        _restore_env_vars(old_env)
        reset_settings()
        await reset_db_pool()
        await reset_api_key_manager()


@pytest.mark.asyncio
async def test_api_key_quota_enforced_for_rag_and_chat_sqlite(monkeypatch, tmp_path):
    # Multi-user mode to validate API key flow
    old_env = _set_env_vars(
        {
            "AUTH_MODE": "multi_user",
            "DATABASE_URL": f"sqlite:///{tmp_path/'users.db'}",
            "OPENAI_API_KEY": "test",
        }
    )

    # Reset and ensure tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import reset_api_key_manager
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()
    await reset_api_key_manager()
    fastapi_app = None
    chacha_deps = None
    cleanup_chacha = None
    try:
        pool = await get_db_pool()
        ensure_authnz_tables(Path(pool.db_path))

        # Create a user
        async with pool.transaction() as conn:
            await conn.execute(
                "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
                ("alice", "alice@example.com", "x"),
            )
        user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "alice")

        # Create virtual API key with max_calls=1 for rag.search
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
        mgr = APIKeyManager()
        await mgr.initialize()
        vk = await mgr.create_virtual_key(
            user_id=int(user_id),
            name="vk-rag",
            allowed_endpoints=["rag.search"],
            allowed_paths=["/api/v1/rag/search"],
            allowed_methods=["POST"],
            max_calls=1,
        )
        key = vk['key']
        rag_body = {
            "query": "hello",
            "top_k": 1,
            # Match single-user quota test: avoid vector retrieval to keep this lightweight.
            "search_mode": "fts",
            "enable_generation": False,
            "enable_reranking": False,
            "enable_cache": False,
        }
        headers = {"X-API-KEY": key}

        # Avoid exercising the real ChaChaNotes executor (which may have been shut
        # down in other tests) by overriding the ChaChaNotes DB dependency with a
        # lightweight temp-file CharactersRAGDB. A file-backed DB is required
        # because the chat endpoint can persist through executor threads.
        from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
        from tldw_Server_API.app.main import app as fastapi_app

        _override_chacha_db_for_user, cleanup_chacha = _file_backed_chacha_override(
            tmp_path / "api_key_quota_chacha.db",
            client_id="test-quota",
        )
        fastapi_app.dependency_overrides[chacha_deps.get_chacha_db_for_user] = _override_chacha_db_for_user

        with TestClient(_app()) as client:
            r1 = client.post("/api/v1/rag/search", headers=headers, json=rag_body)
            assert r1.status_code == 200, r1.text
            r2 = client.post("/api/v1/rag/search", headers=headers, json=rag_body)
            assert r2.status_code == 403

            # Chat with API key: monkeypatch orchestrator and enforce max_calls=1
            import tldw_Server_API.app.api.v1.endpoints.chat as chat_endpoint
            monkeypatch.setattr(chat_endpoint, 'perform_chat_api_call', lambda **kwargs: _chat_stub_response())
            import tldw_Server_API.app.api.v1.schemas.chat_request_schemas as schema_chat
            chat_endpoint.API_KEYS = {**(chat_endpoint.API_KEYS or {}), 'openai': 'test'}
            schema_chat.API_KEYS = {**(schema_chat.API_KEYS or {}), 'openai': 'test'}
            vk2 = await mgr.create_virtual_key(
                user_id=int(user_id),
                name="vk-chat",
                allowed_endpoints=["chat.completions"],
                allowed_paths=["/api/v1/chat/completions"],
                allowed_methods=["POST"],
                max_calls=1,
            )
            key2 = vk2['key']
            body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
            headers2 = {"X-API-KEY": key2}
            r1 = client.post("/api/v1/chat/completions", headers=headers2, json=body)
            assert r1.status_code == 200, r1.text
            r2 = client.post("/api/v1/chat/completions", headers=headers2, json=body)
            assert r2.status_code == 403
    finally:
        if fastapi_app is not None and chacha_deps is not None:
            fastapi_app.dependency_overrides.pop(chacha_deps.get_chacha_db_for_user, None)
        if cleanup_chacha is not None:
            cleanup_chacha()
        _restore_env_vars(old_env)
        reset_settings()
        await reset_db_pool()
        await reset_api_key_manager()
