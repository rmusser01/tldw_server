import contextlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.rate_limit


@pytest.fixture(params=["memory", "redis"], ids=["rg-memory", "rg-redis"])
def rg_backend(request) -> str:
    """Exercise header behavior under both RG backends."""
    return str(request.param)


def _repo_policy_path() -> str:


     # tldw_Server_API/tests/Resource_Governance → tldw_Server_API
    return str(Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml")


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


def _reset_rg_state(app) -> None:


    """
    Ensure each test starts with a fresh ResourceGovernor / policy loader.

    Tests in this module mutate RG_POLICY_PATH and related envs; reusing the
    same FastAPI app instance without resetting RG state can cause cross-test
    rate-limit bleed (unexpected 429s). This helper clears governor-related
    attributes so middleware will lazily reinitialize from the current env.
    """
    for attr in ("rg_governor", "rg_policy_loader", "rg_policy_store", "rg_policy_version", "rg_policy_count"):
        try:
            if hasattr(app.state, attr):
                setattr(app.state, attr, None)
        except Exception:
            continue


def _ensure_audio_transcriptions_route(app) -> None:
    if any(getattr(route, "path", None) == "/api/v1/audio/transcriptions" for route in app.routes):
        return

    from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router

    app.include_router(audio_router, prefix="/api/v1/audio", tags=["audio"])


def _install_stub_chacha_db(app, *, user_id: int):
    """
    Avoid cross-test flakiness from ChaChaNotes threadpool lifecycle by
    overriding the DB dependency for chat endpoints in these RG tests.
    """
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    class _StubChaChaDB:
        client_id = str(user_id)

    async def _stub_get_db(current_user=None):  # noqa: ARG001
        return _StubChaChaDB()

    app.dependency_overrides[get_chacha_db_for_user] = _stub_get_db
    return get_chacha_db_for_user


async def _init_authnz_sqlite(db_path, monkeypatch) -> None:
    """
    Seed a temporary AuthNZ SQLite DB and reset any cached pools/settings so
    request authentication works (single-user env API keys are deprecated).
    """
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

    # Reset cached audio RG governor/handles so RG_POLICY_PATH changes take effect.
    try:
        import tldw_Server_API.app.core.Usage.audio_quota as _aq

        _aq._rg_audio_governor = None  # type: ignore[attr-defined]
        _aq._rg_audio_loader = None  # type: ignore[attr-defined]
        _aq._reset_in_process_counters_for_tests()
    except Exception:
        _ = None


async def _create_user_and_key(*, username: str, email: str, role: str = "user") -> tuple[int, str]:
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
        role=role,
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
    return user_id, str(key_rec["key"])


@pytest.mark.asyncio
async def test_e2e_chat_headers_tokens_and_requests(monkeypatch, tmp_path, rg_backend):
    db_path = tmp_path / "authnz_chat_headers.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    user_id, api_key = await _create_user_and_key(username="chat-headers-user", email="chat-headers-user@example.com")

    # Minimal app mode with RG middleware (requests headers)
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.setenv("RG_POLICY_PATH", _repo_policy_path())
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    # Trigger mock provider path for stability
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)
    chacha_dep = _install_stub_chacha_db(app, user_id=user_id)

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
                body = {
                    "model": "openai/gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                }
                r = c.post("/api/v1/chat/completions", headers={"X-API-KEY": api_key}, json=body)
                assert r.status_code == 200, r.text
                assert r.headers.get("X-RateLimit-Limit") is not None
                assert r.headers.get("X-RateLimit-Remaining") is not None
    finally:
        app.dependency_overrides.pop(chacha_dep, None)


@pytest.mark.asyncio
async def test_e2e_chat_deny_headers_retry_after(monkeypatch, tmp_path, rg_backend):
    """
    Verify that when RGSimpleMiddleware denies /api/v1/chat/completions
    via a low-RPM policy, the response includes Retry-After and
    X-RateLimit-* headers consistent with the policy.
    """
    db_path = tmp_path / "authnz_chat_deny.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    _user_id, api_key = await _create_user_and_key(username="chat-deny-user", email="chat-deny-user@example.com")

    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")

    policy_id = f"chat.small.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 1 }\n"
        "    scopes: [user, api_key]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    /api/v1/chat/completions: {policy_id}\n"
    )
    p = tmp_path / "rg_chat.yaml"
    p.write_text(policy, encoding="utf-8")
    monkeypatch.setenv("RG_POLICY_PATH", str(p))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    monkeypatch.setenv("TEST_MODE", "true")

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    body = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    with _with_rg_middleware(app):
        with TestClient(app) as c:
            r1 = c.post("/api/v1/chat/completions", headers={"X-API-KEY": api_key}, json=body)
            assert r1.status_code != 429

            r2 = c.post("/api/v1/chat/completions", headers={"X-API-KEY": api_key}, json=body)
            assert r2.status_code == 429
            assert r2.headers.get("Retry-After") is not None
            assert r2.headers.get("X-RateLimit-Limit") == "1"
            assert r2.headers.get("X-RateLimit-Remaining") == "0"
            reset = r2.headers.get("X-RateLimit-Reset")
            assert reset is not None and int(reset) >= 1


@pytest.mark.asyncio
async def test_e2e_embeddings_headers_route_map(monkeypatch, tmp_path, rg_backend):
    db_path = tmp_path / "authnz_embeddings_headers.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    _user_id, api_key = await _create_user_and_key(
        username="embeddings-headers-user",
        email="embeddings-headers-user@example.com",
    )

    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.setenv("RG_POLICY_PATH", _repo_policy_path())
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    with _with_rg_middleware(app):
        with TestClient(app) as c:
            resp = c.get("/api/v1/embeddings/providers-config", headers={"X-API-KEY": api_key})

    assert resp.status_code in (200, 429), resp.text
    assert resp.headers.get("X-RateLimit-Limit") is not None
    assert resp.headers.get("X-RateLimit-Remaining") is not None


@pytest.mark.asyncio
async def test_e2e_embeddings_deny_headers_retry_after(monkeypatch, tmp_path, rg_backend):
    db_path = tmp_path / "authnz_embeddings_deny.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    _user_id, api_key = await _create_user_and_key(
        username="embeddings-deny-user",
        email="embeddings-deny-user@example.com",
    )

    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")

    policy_id = f"embeddings.small.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 1 }\n"
        "    scopes: [user, api_key]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    /api/v1/embeddings/providers-config: {policy_id}\n"
    )
    p = tmp_path / "rg_embeddings.yaml"
    p.write_text(policy, encoding="utf-8")
    monkeypatch.setenv("RG_POLICY_PATH", str(p))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    with _with_rg_middleware(app):
        with TestClient(app) as c:
            r1 = c.get("/api/v1/embeddings/providers-config", headers={"X-API-KEY": api_key})
            assert r1.status_code == 200, r1.text

            r2 = c.get("/api/v1/embeddings/providers-config", headers={"X-API-KEY": api_key})
            assert r2.status_code in (429, 503)
            if r2.status_code == 429:
                assert r2.headers.get("Retry-After") is not None
                assert r2.headers.get("X-RateLimit-Limit") == "1"
                assert r2.headers.get("X-RateLimit-Remaining") == "0"
                reset = r2.headers.get("X-RateLimit-Reset")
                assert reset is not None and int(reset) >= 1


@pytest.mark.asyncio
async def test_e2e_mcp_headers_route_map(monkeypatch, rg_backend):
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.setenv("RG_POLICY_PATH", _repo_policy_path())
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    import tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint as mcp_ep

    class _StubMCP:
        initialized = True

        async def get_status(self):
            return {"status": "healthy", "version": "test", "uptime_seconds": 1.0, "connections": {}, "modules": {}}

    monkeypatch.setattr(mcp_ep, "get_mcp_server", lambda: _StubMCP())

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    with _with_rg_middleware(app):
        with TestClient(app) as c:
            resp = c.get("/api/v1/mcp/status", headers={"X-API-KEY": "test-api-key"})

    assert resp.status_code == 200, resp.text
    assert resp.headers.get("X-RateLimit-Limit") is not None
    assert resp.headers.get("X-RateLimit-Remaining") is not None


@pytest.mark.asyncio
async def test_e2e_audio_websocket_streams_limit(monkeypatch, tmp_path, rg_backend):
    db_path = tmp_path / "authnz_audio_ws.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    _user_id, api_key = await _create_user_and_key(username="audio-ws-user", email="audio-ws-user@example.com")

    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.setenv("RG_POLICY_PATH", _repo_policy_path())
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    # Allow streaming quotas at the module level to avoid DB/Redis dependencies
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep

    async def _noop(*args, **kwargs):
        return None

    async def _allow_minutes(user_id: int, minutes: float):
        _ = (user_id, minutes)
        return True, 0

    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_minutes)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop)

    from tldw_Server_API.app.main import app

    _reset_rg_state(app)

    with _with_rg_middleware(app):
        with TestClient(app) as c:
            ws1 = c.websocket_connect(f"/api/v1/audio/stream/transcribe?token={api_key}")
            ws2 = c.websocket_connect(f"/api/v1/audio/stream/transcribe?token={api_key}")
            ws3 = None
            denied = False
            try:
                ws3 = c.websocket_connect(f"/api/v1/audio/stream/transcribe?token={api_key}")
                data = ws3.receive_json()
                denied = (data or {}).get("error_type") in {"rate_limited", "quota_exceeded"}
            except Exception:
                denied = True
            finally:
                try:
                    if ws3:
                        ws3.close()
                except Exception:
                    _ = None
                try:
                    ws2.close()
                except Exception:
                    _ = None
                try:
                    ws1.close()
                except Exception:
                    _ = None
            assert denied


@pytest.mark.asyncio
async def test_e2e_audio_transcriptions_headers_and_mocked_stt(monkeypatch, tmp_path, rg_backend):
    db_path = tmp_path / "authnz_audio_transcriptions.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    _user_id, api_key = await _create_user_and_key(
        username="audio-transcriptions-user",
        email="audio-transcriptions-user@example.com",
    )

    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_BACKEND", rg_backend)
    monkeypatch.setenv("RG_POLICY_STORE", "file")

    policy_id = f"audio.transcribe.{rg_backend}.{tmp_path.name.replace('-', '_')}"
    policy = (
        "schema_version: 1\n"
        "policies:\n"
        f"  {policy_id}:\n"
        "    requests: { rpm: 2 }\n"
        "    tokens: { per_min: 1000 }\n"
        "    scopes: [user, api_key]\n"
        "route_map:\n"
        "  by_path:\n"
        f"    /api/v1/audio/transcriptions: {policy_id}\n"
    )
    p = tmp_path / "rg_audio.yaml"
    p.write_text(policy, encoding="utf-8")
    monkeypatch.setenv("RG_POLICY_PATH", str(p))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")

    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files

    def _fake_check_status(model_name: str):
        return {"available": True, "model": model_name}

    monkeypatch.setattr(audio_files, "check_transcription_model_status", _fake_check_status)

    async def _ok_job(user_id: int):
        _ = user_id
        return True, ""

    async def _noop(*args, **kwargs):
        return None

    async def _allow_minutes(user_id: int, minutes: float):
        _ = (user_id, minutes)
        return True, 0

    monkeypatch.setattr(audio_ep, "can_start_job", _ok_job)
    monkeypatch.setattr(audio_ep, "finish_job", _noop)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_minutes)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop)

    import numpy as np

    class _FakeSoundFile:
        @staticmethod
        def info(path):
            _ = path
            return type("Info", (), {"frames": 1600, "samplerate": 16000})()

        @staticmethod
        def read(fd, dtype="float32"):
            _ = (fd, dtype)
            data = np.zeros((1600,), dtype="float32")
            sr = 16000
            return data, sr

    monkeypatch.setattr(audio_ep, "sf", _FakeSoundFile(), raising=False)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as tl

    def fake_speech_to_text(
        path,
        whisper_model,
        selected_source_lang=None,
        vad_filter=False,
        diarize=False,
        word_timestamps=False,
        return_language=False,
        **kwargs,
    ):

        _ = (
            path,
            whisper_model,
            selected_source_lang,
            vad_filter,
            diarize,
            word_timestamps,
            kwargs,
        )
        segs = [{"Text": "hello world"}]
        if return_language:
            return segs, "en"
        return segs

    monkeypatch.setattr(tl, "speech_to_text", fake_speech_to_text)

    from tldw_Server_API.app.main import app

    _ensure_audio_transcriptions_route(app)
    _reset_rg_state(app)

    with _with_rg_middleware(app):
        with TestClient(app) as c:
            payload = b"RIFF\x00\x00\x00\x00WAVEfmt "  # not parsed due to monkeypatched sf.read
            files = {"file": ("test.wav", payload, "audio/wav")}
            r = c.post(
                "/api/v1/audio/transcriptions",
                headers={"X-API-KEY": api_key},
                data={"model": "whisper-1", "response_format": "json"},
                files=files,
            )
            assert r.status_code == 200, r.text
            assert r.headers.get("X-RateLimit-Limit") == "2"
            assert r.headers.get("X-RateLimit-Remaining") is not None
