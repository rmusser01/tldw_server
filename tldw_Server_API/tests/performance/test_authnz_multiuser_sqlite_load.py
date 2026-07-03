import asyncio
import importlib
import json
import os
import time
from typing import Any, Awaitable, Callable
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient


def _perf_enabled() -> bool:
    return os.getenv("PERF", "0").lower() in {"1", "true", "yes", "y", "on"}


pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        not _perf_enabled(),
        reason="set PERF=1 to run performance checks",
    ),
]


async def _build_app(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    user_db_base = tmp_path / "user_databases"
    user_db_base.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-multiuser-sqlite-0123456789abcdef")
    monkeypatch.setenv("SESSION_ENCRYPTION_KEY", Fernet.generate_key().decode("utf-8"))
    monkeypatch.setenv("ENABLE_REGISTRATION", "true")
    monkeypatch.setenv("REQUIRE_REGISTRATION_CODE", "false")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
    monkeypatch.setenv("TLDW_SQLITE_WAL_MODE", "true")
    monkeypatch.setenv("AUTHNZ_FORCE_REAL_SESSION_MANAGER", "true")
    monkeypatch.setenv("CHAT_FORCE_MOCK", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-12345")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("TESTING", "true")

    try:
        from tldw_Server_API.app.core.config import settings as core_settings
        core_settings["CSRF_ENABLED"] = False
        # The lazy core-settings mapping materializes USER_DB_BASE_DIR from env
        # once; DatabasePaths prefers the settings value over env when it is
        # non-default, so without this sync the SECOND _build_app in a session
        # would keep routing per-user DBs into the FIRST test's tmp dir.
        core_settings["USER_DB_BASE_DIR"] = user_db_base
    except Exception:
        _ = None

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once
    # Service-layer singletons cache the db pool at first use; without these
    # resets the SECOND _build_app in a session registers users (and writes
    # media/audit rows) through the previous app's pool into the previous
    # tmp database — logins then 401 and FK constraints fail (issue #2580).
    # Mirrors the canonical reset set in tests/AuthNZ/conftest.py.
    from tldw_Server_API.app.core.Audit.unified_audit_service import shutdown_audit_service
    from tldw_Server_API.app.core.AuthNZ.alerting import reset_security_alert_dispatcher
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import reset_api_key_manager
    from tldw_Server_API.app.core.AuthNZ.lockout_tracker import reset_lockout_tracker
    from tldw_Server_API.app.core.AuthNZ.rate_limiter import reset_rate_limiter
    from tldw_Server_API.app.core.AuthNZ.scheduler import reset_authnz_scheduler
    from tldw_Server_API.app.core.AuthNZ.token_blacklist import reset_token_blacklist
    from tldw_Server_API.app.core.Billing.enforcement import reset_billing_enforcer
    from tldw_Server_API.app.core.Billing.subscription_service import reset_subscription_service
    from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db
    from tldw_Server_API.app.services.org_invite_service import reset_invite_service
    from tldw_Server_API.app.services.registration_service import reset_registration_service
    from tldw_Server_API.app.services.storage_cleanup_service import reset_cleanup_service
    from tldw_Server_API.app.services.storage_quota_service import reset_storage_service
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import close_all_chacha_db_instances

    close_all_chacha_db_instances()
    await reset_db_pool()
    await reset_session_manager()
    await reset_token_blacklist()
    await reset_security_alert_dispatcher()
    await reset_authnz_scheduler()
    await reset_rate_limiter()
    await reset_lockout_tracker()
    reset_settings()
    reset_jwt_service()
    await reset_registration_service()
    await reset_invite_service()
    await reset_subscription_service()
    reset_billing_enforcer()
    await reset_api_key_manager()
    await reset_storage_service()
    await reset_cleanup_service()
    await reset_users_db()
    await shutdown_audit_service()

    await ensure_authnz_schema_ready_once()

    import tldw_Server_API.app.main as main_module
    return importlib.reload(main_module).app


@pytest.mark.asyncio
async def test_multi_user_sqlite_register_login_load(tmp_path, monkeypatch):
    app = await _build_app(tmp_path, monkeypatch)

    user_count = int(os.getenv("PERF_MULTIUSER_USERS", "50"))
    concurrency = int(os.getenv("PERF_MULTIUSER_CONCURRENCY", "10"))
    chat_per_user = int(os.getenv("PERF_MULTIUSER_CHAT_PER_USER", "2"))
    media_per_user = int(os.getenv("PERF_MULTIUSER_MEDIA_PER_USER", "1"))
    password = "T7!vQx9pLk"

    sem = asyncio.Semaphore(concurrency)

    async def _register_user(i: int, client: AsyncClient):
        payload = {
            "username": f"perfuser{i}",
            "email": f"perfuser{i}@example.com",
            "password": password,
        }
        resp = await client.post("/api/v1/auth/register", json=payload)
        if resp.status_code != 200:
            return i, resp.status_code, resp.text
        return None

    async def _login_user(i: int, client: AsyncClient):
        payload = {
            "username": f"perfuser{i}",
            "password": password,
        }
        resp = await client.post("/api/v1/auth/login", data=payload)
        if resp.status_code != 200:
            return i, resp.status_code, resp.text
        token = resp.json().get("access_token")
        if not token:
            return i, resp.status_code, "missing access_token"
        return i, token

    async def _chat_request(i: int, j: int, token: str, client: AsyncClient):
        payload = {
            "api_provider": "openai",
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"hello {i}-{j}"}],
        }
        headers = {"Authorization": f"Bearer {token}"}
        resp = await client.post("/api/v1/chat/completions", json=payload, headers=headers)
        if resp.status_code != 200:
            return i, j, resp.status_code, resp.text
        return None

    async def _media_add_request(i: int, j: int, token: str, client: AsyncClient):
        content = f"Perf media content {i}-{j}".encode("utf-8")
        data = {
            "title": f"Perf Doc {i}-{j}",
            "media_type": "document",
            "perform_analysis": "false",
            "perform_chunking": "false",
        }
        files = {
            "files": (f"perf_{i}_{j}.txt", content, "text/plain"),
        }
        headers = {"Authorization": f"Bearer {token}"}
        resp = await client.post("/api/v1/media/add", data=data, files=files, headers=headers)
        if resp.status_code not in (200, 207):
            return i, j, resp.status_code, resp.text
        return None

    async def _run_with_sem(func: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        async with sem:
            return await func(*args)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.time()
        register_results = await asyncio.gather(
            *(_run_with_sem(_register_user, i, client) for i in range(user_count))
        )
        register_errors = [r for r in register_results if r is not None]
        assert not register_errors, f"registration errors: {register_errors[:3]}"
        t1 = time.time()

        login_results = await asyncio.gather(
            *(_run_with_sem(_login_user, i, client) for i in range(user_count))
        )
        login_errors = [r for r in login_results if not (isinstance(r, tuple) and len(r) == 2)]
        assert not login_errors, f"login errors: {login_errors[:3]}"
        tokens = {i: token for (i, token) in login_results}  # type: ignore[misc]
        t2 = time.time()

        mock_response = {
            "id": "chatcmpl-perf",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }

        def _mock_chat_call(**kwargs):
            if kwargs.get("streaming"):
                def _stream_gen():
                    data_chunk = {
                        "choices": [
                            {
                                "delta": {"role": "assistant", "content": "ok"},
                                "finish_reason": None,
                                "index": 0,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(data_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                return _stream_gen()
            return mock_response

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-openai-key-12345"}),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", side_effect=_mock_chat_call),
            patch("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call", side_effect=_mock_chat_call),
        ):
            if chat_per_user > 0:
                t3 = time.time()
                chat_tasks = [
                    _run_with_sem(_chat_request, i, j, tokens[i], client)
                    for i in range(user_count)
                    for j in range(chat_per_user)
                ]
                chat_results = await asyncio.gather(*chat_tasks)
                chat_errors = [r for r in chat_results if r is not None]
                assert not chat_errors, f"chat errors: {chat_errors[:3]}"
                t4 = time.time()
            else:
                t3 = t4 = time.time()

            if media_per_user > 0:
                t5 = time.time()
                media_tasks = [
                    _run_with_sem(_media_add_request, i, j, tokens[i], client)
                    for i in range(user_count)
                    for j in range(media_per_user)
                ]
                media_results = await asyncio.gather(*media_tasks)
                media_errors = [r for r in media_results if r is not None]
                assert not media_errors, f"media add errors: {media_errors[:3]}"
                t6 = time.time()
            else:
                t5 = t6 = time.time()

    reg_dt = t1 - t0
    login_dt = t2 - t1
    reg_qps = user_count / reg_dt if reg_dt > 0 else float("inf")
    login_qps = user_count / login_dt if login_dt > 0 else float("inf")
    chat_ops = user_count * chat_per_user
    media_ops = user_count * media_per_user
    chat_dt = t4 - t3
    media_dt = t6 - t5
    chat_qps = chat_ops / chat_dt if chat_dt > 0 else float("inf")
    media_qps = media_ops / media_dt if media_dt > 0 else float("inf")
    print(
        "multi_user_sqlite_load "
        f"users={user_count} concurrency={concurrency} "
        f"register_s={reg_dt:.3f} reg_qps={reg_qps:.1f} "
        f"login_s={login_dt:.3f} login_qps={login_qps:.1f} "
        f"chat_ops={chat_ops} chat_s={chat_dt:.3f} chat_qps={chat_qps:.1f} "
        f"media_ops={media_ops} media_s={media_dt:.3f} media_qps={media_qps:.1f}"
    )


@pytest.mark.asyncio
async def test_multi_user_sqlite_streaming_chat_worst_case(tmp_path, monkeypatch):
    app = await _build_app(tmp_path, monkeypatch)

    user_count = int(os.getenv("PERF_MULTIUSER_STREAM_USERS", "6"))
    password = "T7!vQx9pLk"

    async def _register_user(i: int, client: AsyncClient):
        payload = {
            "username": f"streamuser{i}",
            "email": f"streamuser{i}@example.com",
            "password": password,
        }
        resp = await client.post("/api/v1/auth/register", json=payload)
        if resp.status_code != 200:
            diag = {
                "detail": resp.text,
                "x_tldw_register_error": resp.headers.get("X-TLDW-Register-Error"),
                "x_tldw_db": resp.headers.get("X-TLDW-DB"),
            }
            return i, resp.status_code, diag
        return None

    async def _login_user(i: int, client: AsyncClient):
        payload = {
            "username": f"streamuser{i}",
            "password": password,
        }
        resp = await client.post("/api/v1/auth/login", data=payload)
        if resp.status_code != 200:
            return i, resp.status_code, resp.text
        token = resp.json().get("access_token")
        if not token:
            return i, resp.status_code, "missing access_token"
        return i, token

    async def _stream_chat(i: int, token: str, client: AsyncClient):
        payload = {
            "api_provider": "openai",
            "model": "gpt-4o-mini",
            "stream": True,
            "messages": [{"role": "user", "content": f"stream {i}"}],
        }
        headers = {"Authorization": f"Bearer {token}"}
        async with client.stream("POST", "/api/v1/chat/completions", json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                return i, resp.status_code, await resp.aread()
            saw_chunk = False
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        break
                    saw_chunk = True
            if not saw_chunk:
                return i, 500, "no stream chunks received"
        return None

    mock_response = {
        "id": "chatcmpl-perf",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }

    def _mock_chat_call(**kwargs):
        if kwargs.get("streaming"):
            def _stream_gen():
                data_chunk = {
                    "choices": [
                        {
                            "delta": {"role": "assistant", "content": "ok"},
                            "finish_reason": None,
                            "index": 0,
                        }
                    ]
                }
                yield f"data: {json.dumps(data_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return _stream_gen()
        return mock_response

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg_errors = []
        for i in range(user_count):
            err = await _register_user(i, client)
            if err is not None:
                reg_errors.append(err)
        assert not reg_errors, f"registration errors: {reg_errors[:3]}"

        login_results = []
        for i in range(user_count):
            login_results.append(await _login_user(i, client))
        login_errors = [r for r in login_results if not (isinstance(r, tuple) and len(r) == 2)]
        assert not login_errors, f"login errors: {login_errors[:3]}"
        tokens = {i: token for (i, token) in login_results}  # type: ignore[misc]

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-openai-key-12345"}),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", side_effect=_mock_chat_call),
            patch("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call", side_effect=_mock_chat_call),
        ):
            t0 = time.time()
            stream_results = await asyncio.gather(
                *(_stream_chat(i, tokens[i], client) for i in range(user_count))
            )
            t1 = time.time()

        stream_errors = [r for r in stream_results if r is not None]
        assert not stream_errors, f"stream errors: {stream_errors[:3]}"

    stream_dt = t1 - t0
    stream_qps = user_count / stream_dt if stream_dt > 0 else float("inf")
    print(
        "multi_user_sqlite_streaming "
        f"users={user_count} stream_s={stream_dt:.3f} stream_qps={stream_qps:.1f}"
    )
