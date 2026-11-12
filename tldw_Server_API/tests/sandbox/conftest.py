from __future__ import annotations

import importlib.machinery
import asyncio as _asyncio
import os
import sys
import types
import warnings
import pytest

# Stub heavyweight audio deps before app import to keep sandbox endpoint tests
# deterministic in environments where torch-backed imports can abort.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


@pytest.fixture(autouse=True)
def _sandbox_clear_config_cache():
    """Reset config cache before each sandbox test to prevent env var leaks."""
    try:
        from tldw_Server_API.app.core.config import clear_config_cache
        clear_config_cache()
    except Exception:
        pass
    yield
    try:
        from tldw_Server_API.app.core.config import clear_config_cache
        clear_config_cache()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _sandbox_reset_stream_hub():
    """Clear stale event loop references from the stream hub between tests."""
    yield
    try:
        from tldw_Server_API.app.core.Sandbox.streams import get_hub
        get_hub().reset_loop()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def sandbox_auth_defaults(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Provide default auth settings and headers for sandbox tests.

    Sandbox endpoints require auth even in TEST_MODE. We default to single-user
    mode and inject a test API key unless a test explicitly opts out.
    """
    # Allow opt-out for tests that want to exercise unauthenticated behavior.
    if request.node.get_closest_marker("sandbox_no_auth"):
        return
    # Ensure single-user mode unless a test sets AUTH_MODE explicitly.
    if os.getenv("AUTH_MODE") is None:
        monkeypatch.setenv("AUTH_MODE", "single_user")
    # Avoid implicit DATABASE_URL defaults leaking into sandbox store mode tests.
    if os.getenv("DATABASE_URL") is None:
        monkeypatch.setenv("DATABASE_URL", "")
    # Use a deterministic, sufficiently long test key.
    test_key = os.getenv("SINGLE_USER_TEST_API_KEY") or "test-sandbox-api-key-123456"
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", test_key)
    # Refresh cached AuthNZ settings so env changes are picked up.
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
        reset_settings()
    except Exception:
        _ = None


@pytest.fixture(autouse=True)
def sandbox_testclient_auth_headers(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Inject X-API-KEY into TestClient HTTP/WS requests by default for sandbox tests."""
    if request.node.get_closest_marker("sandbox_no_auth"):
        return
    test_key = os.getenv("SINGLE_USER_TEST_API_KEY") or "test-sandbox-api-key-123456"
    try:
        from fastapi.testclient import TestClient as FastAPITestClient
        from starlette.testclient import TestClient as StarletteTestClient
    except Exception:
        return
    original_init = StarletteTestClient.__init__
    original_ws = StarletteTestClient.websocket_connect
    original_request = StarletteTestClient.request

    def _init(self, *args, **kwargs):  # type: ignore[no-redef]
        original_init(self, *args, **kwargs)
        try:
            if "X-API-KEY" not in self.headers and "Authorization" not in self.headers:
                self.headers["X-API-KEY"] = test_key
        except Exception:
            _ = None

    def _ws(self, url, *args, **kwargs):  # type: ignore[no-redef]
        headers = dict(kwargs.get("headers") or {})
        if "X-API-KEY" not in headers and "Authorization" not in headers:
            headers["X-API-KEY"] = test_key
        kwargs["headers"] = headers
        return original_ws(self, url, *args, **kwargs)

    def _request(self, method, url, **kwargs):  # type: ignore[no-redef]
        headers = dict(kwargs.get("headers") or {})
        if "X-API-KEY" not in headers and "Authorization" not in headers:
            headers["X-API-KEY"] = test_key
        kwargs["headers"] = headers
        return original_request(self, method, url, **kwargs)

    # Patch the concrete Starlette TestClient class (FastAPI re-exports it).
    monkeypatch.setattr(StarletteTestClient, "__init__", _init, raising=True)
    monkeypatch.setattr(StarletteTestClient, "websocket_connect", _ws, raising=True)
    monkeypatch.setattr(StarletteTestClient, "request", _request, raising=True)
    # Also patch the FastAPI alias to be safe.
    try:
        monkeypatch.setattr(FastAPITestClient, "__init__", _init, raising=True)
        monkeypatch.setattr(FastAPITestClient, "websocket_connect", _ws, raising=True)
        monkeypatch.setattr(FastAPITestClient, "request", _request, raising=True)
    except Exception:
        _ = None


@pytest.fixture(autouse=True)
def sandbox_default_fake_exec(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Default Docker runner to fake exec for sandbox tests unless opted out."""
    if request.node.get_closest_marker("sandbox_real_docker"):
        monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    else:
        monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", os.getenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC") or "1")


@pytest.fixture(autouse=True)
def sandbox_redis_factory_shim(monkeypatch: pytest.MonkeyPatch):
    """Make redis_factory compatible with sandbox tests that inject fake redis modules."""
    try:
        from tldw_Server_API.app.core.Infrastructure import redis_factory
    except Exception:
        return
    orig_redis = getattr(redis_factory, "redis", None)

    class _RedisShim:
        @staticmethod
        def from_url(url: str, **kwargs):
            import sys as _sys
            mod = _sys.modules.get("redis")
            if mod is not None:
                cls = getattr(mod, "Redis", None)
                if cls is not None:
                    if hasattr(cls, "from_url"):
                        return cls.from_url(url)
                    try:
                        return cls()
                    except Exception:
                        _ = None
            if orig_redis is not None and hasattr(orig_redis, "from_url"):
                return orig_redis.from_url(url, **kwargs)
            # Fallback to in-memory stub for tests
            try:
                return redis_factory.InMemorySyncRedis(
                    decode_responses=kwargs.get("decode_responses", True)
                )
            except Exception:
                raise

    # Ensure redis is non-None so create_sync_redis_client doesn't early abort.
    monkeypatch.setattr(redis_factory, "redis", _RedisShim, raising=False)


@pytest.fixture(autouse=True)
def sandbox_no_reload(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Optionally short-circuit importlib.reload for app.main to avoid loguru sink churn."""
    if not request.node.get_closest_marker("sandbox_no_reload"):
        return
    import importlib as _importlib
    original_reload = _importlib.reload

    def _reload(module):  # type: ignore[no-redef]
        if getattr(module, "__name__", "") == "tldw_Server_API.app.main":
            return module
        return original_reload(module)

    monkeypatch.setattr(_importlib, "reload", _reload, raising=True)


@pytest.fixture(autouse=True)
def sandbox_ws_signed_defaults(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Default sandbox WS signed URL settings to disabled unless a test opts in."""
    if request.node.get_closest_marker("sandbox_ws_signed"):
        return
    monkeypatch.setenv("SANDBOX_WS_SIGNED_URLS", "false")
    monkeypatch.delenv("SANDBOX_WS_SIGNING_SECRET", raising=False)


@pytest.fixture
def patch_sandbox_heartbeat_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speed up endpoint heartbeats without changing other asyncio clocks."""
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    native_sleep = _asyncio.sleep

    async def heartbeat_sleep(delay: float, result: object = None) -> object:
        return await native_sleep(0.01 if delay == 10 else delay, result)

    # sb.asyncio normally aliases the shared stdlib module. Replace only the
    # endpoint's reference; Jobs backoff, MCP loops and stdin clocks stay native.
    facade = types.SimpleNamespace(**{**vars(_asyncio), "sleep": heartbeat_sleep})
    monkeypatch.setattr(sb, "asyncio", facade)


@pytest.fixture(autouse=True, scope="session")
def set_ws_poll_timeout_for_tests():
    """
    Configure environment variables to make sandbox WebSocket behavior test-friendly.

    Sets sensible defaults only if not already present:
    - Sets SANDBOX_WS_POLL_TIMEOUT_SEC to "1" so the WebSocket loop notices disconnects quickly.
    - Enables SANDBOX_ENABLE_EXECUTION and SANDBOX_BACKGROUND_EXECUTION to allow execution and background mode during tests.
    - Enables SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS to avoid CI hangs by using synthetic frames.
    - Ensures "sandbox" is present in ROUTES_ENABLE so the sandbox router is active for tests.
    """
    os = __import__("os")
    os.environ.setdefault("SANDBOX_WS_POLL_TIMEOUT_SEC", "1")
    # Default to enabling execution and background mode in WS tests unless a test overrides
    os.environ.setdefault("SANDBOX_ENABLE_EXECUTION", "true")
    os.environ.setdefault("SANDBOX_BACKGROUND_EXECUTION", "true")
    # Enable synthetic WS frames to avoid hangs in CI for sandbox tests only
    os.environ.setdefault("SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS", "true")
    # Ensure the experimental sandbox router is enabled for these tests
    existing_enable = os.environ.get("ROUTES_ENABLE", "")
    parts = [p.strip().lower() for p in existing_enable.split(",") if p.strip()]
    if "sandbox" not in parts:
        parts.append("sandbox")
    os.environ["ROUTES_ENABLE"] = ",".join(parts)


@pytest.fixture(autouse=True)
def bypass_sandbox_ws_auth(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Bypass sandbox WS auth in tests unless explicitly opted out.

    Use @pytest.mark.sandbox_ws_auth on tests that should exercise real WS auth.
    """
    if request.node.get_closest_marker("sandbox_ws_auth"):
        return
    try:
        from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
        async def _fake_resolver(*_args, **_kwargs) -> int:
            return int(os.getenv("SINGLE_USER_FIXED_ID", "1"))
        monkeypatch.setattr(sb, "_resolve_sandbox_ws_user_id", _fake_resolver, raising=True)
    except Exception:
        _ = None


@pytest.fixture()
def ws_flush():
    """Publish a final frame (heartbeat) for a run to flush WS server loop.

    Usage: call ws_flush(run_id) right before closing the client WebSocket.
    """
    def _flush(run_id: str) -> None:
        try:
            from tldw_Server_API.app.core.Sandbox.streams import get_hub
            hub = get_hub()
            hub.publish_heartbeat(run_id)
        except Exception:
            # Best-effort helper; ignore if hub not available
            _ = None
    return _flush

@pytest.fixture(autouse=True, scope="session")
def reduce_warnings_noise():
    """Globally silence warnings for sandbox tests to ensure fast teardown.

    The main app and its dependencies can emit many deprecations during import.
    For focused sandbox unit tests, silence them to avoid slow exits.
    """
    warnings.filterwarnings("ignore")
