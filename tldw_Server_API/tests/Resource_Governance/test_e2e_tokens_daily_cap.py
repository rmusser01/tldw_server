import contextlib
import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.rate_limit


@pytest.fixture(params=["memory", "redis"], ids=["rg-memory", "rg-redis"])
def rg_backend(request) -> str:
    """Exercise token daily-cap behavior under both RG backends."""
    return str(request.param)


def _reset_rg_state(app):


    for attr in ("rg_governor", "rg_policy_loader", "rg_policy_store", "rg_policy_version", "rg_policy_count"):
        try:
            if hasattr(app.state, attr):
                setattr(app.state, attr, None)
        except Exception:
            continue


@contextlib.contextmanager
def _with_rg_middleware(app):
    """Temporarily install RGSimpleMiddleware for tests that set RG_ENABLED after app import."""
    try:
        from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
        from starlette.middleware import Middleware
    except Exception:
        yield
        return

    original_user_middleware = getattr(app, "user_middleware", [])[:]
    changed = False
    try:
        already = any(getattr(m, "cls", None) is RGSimpleMiddleware for m in original_user_middleware)
        if not already:
            app.user_middleware = [Middleware(RGSimpleMiddleware), *original_user_middleware]
            changed = True
            try:
                app.middleware_stack = app.build_middleware_stack()
            except Exception:
                _ = None
        yield
    finally:
        if changed:
            try:
                app.user_middleware = original_user_middleware
                app.middleware_stack = app.build_middleware_stack()
            except Exception:
                _ = None


async def _init_authnz_sqlite(db_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    try:
        from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        await reset_db_pool()
        reset_settings()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once

        await ensure_authnz_schema_ready_once()
    except Exception:
        _ = None

    # Reset cached RG daily ledger between tests when DATABASE_URL changes.
    try:
        import tldw_Server_API.app.core.Resource_Governance.daily_caps as _dc

        _dc._daily_ledger = None  # type: ignore[attr-defined]
    except Exception:
        _ = None

    # Reset cached tokens ledger/backfill flags between tests when DATABASE_URL changes.
    try:
        import tldw_Server_API.app.core.Usage.usage_tracker as _ut

        _ut._tokens_daily_ledger = None  # type: ignore[attr-defined]
        _ut._tokens_legacy_backfill_done = set()  # type: ignore[attr-defined]
    except Exception:
        _ = None


async def _create_user_and_key(*, username: str, email: str) -> str:
    from uuid import uuid4

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username=username,
        email=email,
        password_hash="x",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid4(),
    )
    user_id = int(created_user["id"])
    mgr = APIKeyManager(pool)
    await mgr.initialize()
    # POST endpoints require write scope in the API-key fallback path.
    key_rec = await mgr.create_api_key(user_id=user_id, name=f"{username}-key", scope="write")
    return str(key_rec["key"])


@pytest.mark.asyncio
async def test_e2e_chat_tokens_daily_cap_denies(monkeypatch, tmp_path, rg_backend):
    db_path = tmp_path / "authnz_chat_tokens.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    api_key = await _create_user_and_key(username="chat-cap-user", email="chat-cap-user@example.com")

    # Minimal app + RG middleware.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    # Enable stable mock chat calls.
    monkeypatch.setenv("TEST_MODE", "true")

    body = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }

    # Choose a daily cap that allows exactly one request based on the same token
    # reserve heuristic used by the chat endpoint.
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
    from tldw_Server_API.app.core.Chat.chat_service import estimate_tokens_from_json
    from tldw_Server_API.app.api.v1.endpoints.chat import _sanitize_json_for_rate_limit

    req_model = ChatCompletionRequest(**body)
    request_json = json.dumps(req_model.model_dump())
    est = int(estimate_tokens_from_json(_sanitize_json_for_rate_limit(request_json)) or 1)
    completion_budget = int(getattr(req_model, "max_tokens", 0) or 0)
    reserve_units = max(1, int(est) + max(0, int(completion_budget)))

    policy_id = f"chat.small.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 100000, burst: 1.0 }\n"
        f"    tokens:   {{ per_min: 1000000, burst: 1.0, daily_cap: {reserve_units} }}\n"
        "    scopes: [user, api_key]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    \"/api/v1/chat/*\": {policy_id}\n"
    )
    p = tmp_path / "rg_chat_tokens.yaml"
    p.write_text(policy, encoding="utf-8")
    monkeypatch.setenv("RG_POLICY_PATH", str(p))

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    # Avoid cross-test flakiness from ChaChaNotes executor lifecycle by
    # overriding the chat DB dependency and bypassing DB-backed context building.
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    class _StubChaChaDB:
        client_id = "1"

    async def _stub_get_db(current_user=None):  # noqa: ARG001
        return _StubChaChaDB()

    app.dependency_overrides[get_chacha_db_for_user] = _stub_get_db

    import tldw_Server_API.app.api.v1.endpoints.chat as chat_ep

    async def _stub_build_context_and_messages(*, chat_db, request_data, loop, metrics, default_save_to_db, final_conversation_id, save_message_fn):  # noqa: ARG001
        llm_payload = [m.model_dump(exclude_none=True) for m in (request_data.messages or []) if getattr(m, "role", "") != "system"]
        return (
            {"name": "Test", "system_prompt": "You are a helpful AI assistant."},
            None,
            final_conversation_id or "rg-test-conv",
            False,
            llm_payload,
            False,
        )

    monkeypatch.setattr(chat_ep, "build_context_and_messages", _stub_build_context_and_messages, raising=False)

    try:
        with _with_rg_middleware(app):
            with TestClient(app) as c:
                r1 = c.post(
                    "/api/v1/chat/completions",
                    headers={"X-API-KEY": api_key},
                    json=body,
                )
                assert r1.status_code == 200

                r2 = c.post(
                    "/api/v1/chat/completions",
                    headers={"X-API-KEY": api_key},
                    json=body,
                )
                assert r2.status_code == 429
                assert r2.headers.get("Retry-After") is not None
                assert r2.headers.get("X-RateLimit-Limit") is not None
    finally:
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.asyncio
async def test_e2e_embeddings_tokens_daily_cap_denies(monkeypatch, tmp_path, rg_backend):
    db_path = tmp_path / "authnz_embeddings_tokens.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    api_key = await _create_user_and_key(username="emb-cap-user", email="emb-cap-user@example.com")

    # Minimal app + RG middleware.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    policy_id = f"embeddings.small.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 100000, burst: 1.0 }\n"
        "    tokens:   { per_min: 1000000, burst: 1.0, daily_cap: 1 }\n"
        "    scopes: [user, api_key]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    \"/api/v1/embeddings*\": {policy_id}\n"
    )
    p = tmp_path / "rg_embeddings_tokens.yaml"
    p.write_text(policy, encoding="utf-8")
    monkeypatch.setenv("RG_POLICY_PATH", str(p))

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    # Patch embeddings execution to avoid external dependencies.
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as _emb_ep

    async def _fake_create_embeddings_batch_async(
        *,
        texts,
        provider,
        model_id=None,
        dimensions=None,
        api_key=None,
        api_url=None,
        metadata=None,
    ):
        _ = (provider, model_id, dimensions, api_key, api_url, metadata)
        return [[0.0, 0.0, 0.0] for _t in (texts or [])]

    monkeypatch.setattr(_emb_ep, "EMBEDDINGS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(_emb_ep, "create_embeddings_batch_async", _fake_create_embeddings_batch_async, raising=False)

    body = {
        "model": "text-embedding-3-small",
        # Provide a single token-array input so token_total is deterministically 1.
        "input": [1],
    }

    with _with_rg_middleware(app):
        with TestClient(app) as c:
            r1 = c.post(
                "/api/v1/embeddings",
                headers={"X-API-KEY": api_key},
                json=body,
            )
            assert r1.status_code == 200, r1.text

            r2 = c.post(
                "/api/v1/embeddings",
                headers={"X-API-KEY": api_key},
                json=body,
            )
            assert r2.status_code == 429, r2.text
            assert r2.headers.get("Retry-After") is not None
            assert r2.headers.get("X-RateLimit-Limit") is not None
