"""
Specific rate-limit tests for Character Chat:
- max_messages_per_chat
- max_chats_per_user

Skip behavior in TEST_MODE:
- These tests validate actual rate-limit enforcement. In TEST_MODE the
  Character Chat limiter is intentionally permissive to reduce flakiness,
  so expectations like 403/429 would not hold. CI runs a dedicated job
  with TEST_MODE=0 for these checks.
"""

import os
import shutil
import tempfile
import httpx
import pytest
import asyncio

from tldw_Server_API.app.core.config import clear_config_cache

# Skip this module when TEST_MODE is enabled; limiter is permissive in TEST_MODE
if str(os.getenv("TEST_MODE", "")).lower() in {"1", "true", "yes", "y", "on"}:
    pytest.skip("Character_Chat rate-limit tests require TEST_MODE=0", allow_module_level=True)


@pytest.mark.asyncio
async def test_max_messages_per_chat_limit():
    tmpdir = tempfile.mkdtemp(prefix="chacha_limit_msgs_")
    env_overrides = {
        "USER_DB_BASE_DIR": tmpdir,
        "MAX_MESSAGES_PER_CHAT": "1",  # Set very low per-chat message cap
    }
    original_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    clear_config_cache()
    crl = None
    cl = None
    try:
        import tldw_Server_API.app.core.Character_Chat.character_rate_limiter as crl  # noqa: WPS433
        from tldw_Server_API.app.core.Character_Chat import character_limits as cl  # noqa: WPS433

        crl._rate_limiter = None
        cl._limits = None

        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Create character chat
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            # First message allowed
            r = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "first"},
            )
            assert r.status_code == 201

            # Second message should be blocked (limit reached)
            r = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "second"},
            )
            assert r.status_code == 403
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_config_cache()
        try:
            if crl is not None:
                crl._rate_limiter = None
        except Exception:
            _ = None
        try:
            if cl is not None:
                cl._limits = None
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_max_chats_per_user_limit():
    tmpdir = tempfile.mkdtemp(prefix="chacha_limit_chats_")
    env_overrides = {"USER_DB_BASE_DIR": tmpdir}
    original_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    clear_config_cache()
    original_max_chats = os.environ.get("MAX_CHATS_PER_USER")
    crl = None
    cl = None
    try:
        import tldw_Server_API.app.core.Character_Chat.character_rate_limiter as crl  # noqa: WPS433
        from tldw_Server_API.app.core.Character_Chat import character_limits as cl  # noqa: WPS433

        crl._rate_limiter = None
        cl._limits = None

        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Determine baseline chat count for this fresh DB/user
            r = await client.get("/api/v1/chats/", headers=headers)
            assert r.status_code == 200
            baseline = r.json().get("total", 0)

            # Set chat limit to baseline + 1 so first create is allowed, second blocked
            os.environ["MAX_CHATS_PER_USER"] = str(baseline + 1)
            clear_config_cache()
            # Reset limiter to pick up new env var
            crl._rate_limiter = None

            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            # First chat allowed
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201

            # Second chat should be blocked
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 403
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if original_max_chats is None:
            os.environ.pop("MAX_CHATS_PER_USER", None)
        else:
            os.environ["MAX_CHATS_PER_USER"] = original_max_chats
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_config_cache()
        try:
            if crl is not None:
                crl._rate_limiter = None
        except Exception:
            _ = None
        try:
            if cl is not None:
                cl._limits = None
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_soft_message_limit_blocks_non_persisted_completion():
    tmpdir = tempfile.mkdtemp(prefix="chacha_soft_limit_")
    env_overrides = {
        "USER_DB_BASE_DIR": tmpdir,
        "MAX_MESSAGES_PER_CHAT_SOFT": "1",
    }
    original_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    clear_config_cache()
    crl = None
    cl = None
    try:
        import tldw_Server_API.app.core.Character_Chat.character_rate_limiter as crl  # noqa: WPS433
        from tldw_Server_API.app.core.Character_Chat import character_limits as cl  # noqa: WPS433

        crl._rate_limiter = None
        cl._limits = None

        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            # Add a single message to hit the soft cap
            r = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "first"},
            )
            assert r.status_code == 201

            # Non-persisted completion should be blocked by soft cap
            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "local-llm",
                    "model": "local-test",
                    "save_to_db": False,
                    "include_character_context": False,
                },
            )
            assert r.status_code == 403
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_config_cache()
        try:
            if crl is not None:
                crl._rate_limiter = None
        except Exception:
            _ = None
        try:
            if cl is not None:
                cl._limits = None
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_chat_completion_per_minute_rate_limit():
    if os.getenv("RG_ENABLED", "").lower() not in {"1", "true", "yes", "y", "on"}:
        pytest.skip("Character chat rate limits are enforced by Resource Governor when enabled.")
    tmpdir = tempfile.mkdtemp(prefix="chacha_limit_complete_")
    env_overrides = {
        "USER_DB_BASE_DIR": tmpdir,
        "MAX_CHAT_COMPLETIONS_PER_MINUTE": "3",
    }
    original_env = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    clear_config_cache()
    original_max_chats = os.environ.get("MAX_CHATS_PER_USER")
    try:
        import tldw_Server_API.app.core.Character_Chat.character_rate_limiter as crl  # noqa: WPS433

        crl._rate_limiter = None

        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Ensure chat cap allows at least one new chat
            r = await client.get("/api/v1/chats/", headers=headers)
            assert r.status_code == 200
            baseline = r.json().get("total", 0)
            os.environ["MAX_CHATS_PER_USER"] = str(baseline + 1)
            clear_config_cache()
            crl._rate_limiter = None

            # Create a chat
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            # Call legacy completion 4 times within < 60s but keep per-second burst under the 5/sec window
            statuses = []
            for i in range(4):
                resp = await client.post(f"/api/v1/chats/{chat_id}/complete", headers=headers, json={})
                statuses.append(resp.status_code)
                if i < 3:
                    await asyncio.sleep(0.2)  # keep under the 5/sec local throttle
            # Expect a 429 due to per-minute completion limiter (limit=3)
            assert any(s == 429 for s in statuses)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if original_max_chats is None:
            os.environ.pop("MAX_CHATS_PER_USER", None)
        else:
            os.environ["MAX_CHATS_PER_USER"] = original_max_chats
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_config_cache()
        try:
            crl._rate_limiter = None
        except Exception:
            _ = None
