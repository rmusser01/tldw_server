import asyncio
import inspect
import os
from collections.abc import Iterable
from typing import Final

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    set_llm_provider_overrides_cache_for_tests,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.main import app


@pytest.fixture(autouse=True)
def healthy_provider_override_snapshot():
    """Keep non-lifespan embedding tests on a deterministic healthy snapshot."""
    set_llm_provider_overrides_cache_for_tests({})
    try:
        yield
    finally:
        set_llm_provider_overrides_cache_for_tests({})


@pytest.fixture
def disable_heavy_startup():
    """Deprecated no-op fixture retained for backward compatibility."""
    yield


@pytest.fixture
def admin_user():
    async def _admin():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=42, username="admin", email="a@x", is_active=True, is_admin=True)

    def _iter_dependency_calls(dependant) -> Iterable[object]:
        for child in getattr(dependant, "dependencies", []) or []:
            call = getattr(child, "call", None)
            if call is not None:
                yield call
            yield from _iter_dependency_calls(child)

    def _route_auth_dependency_keys(name: str) -> set[object]:
        keys: set[object] = set()
        for route in getattr(app, "routes", []):
            path = str(getattr(route, "path", ""))
            if not (path.startswith("/api/v1/embeddings") or path.startswith("/api/v1/vector_stores")):
                continue
            dependant = getattr(route, "dependant", None)
            if dependant is None:
                continue
            for call in _iter_dependency_calls(dependant):
                if getattr(call, "__name__", None) == name:
                    keys.add(call)
        return keys

    request_user_keys = {
        get_request_user,
        auth_deps.get_request_user,
        *_route_auth_dependency_keys("get_request_user"),
    }
    for key in request_user_keys:
        app.dependency_overrides[key] = _admin

    async def _principal_override(request: Request):
        """Override get_auth_principal with an admin AuthPrincipal for tests."""
        from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

        token_type: Final[str] = "access"
        principal = AuthPrincipal(
            kind="user",
            user_id=42,
            api_key_id=None,
            subject=None,
            token_type=token_type,
            jti=None,
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        ip = request.client.host if getattr(request, "client", None) else None
        ua = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
        request_id = request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
        request.state.auth = AuthContext(
            principal=principal,
            ip=ip,
            user_agent=ua,
            request_id=request_id,
        )
        return principal

    principal_keys = {
        auth_deps.get_auth_principal,
        *_route_auth_dependency_keys("get_auth_principal"),
    }
    for key in principal_keys:
        app.dependency_overrides[key] = _principal_override
    try:
        yield
    finally:
        for key in request_user_keys:
            app.dependency_overrides.pop(key, None)
        for key in principal_keys:
            app.dependency_overrides.pop(key, None)


class _RedisHarness:
    def __init__(self, loop: asyncio.AbstractEventLoop, async_client, sync_client, url: str):
        self.loop = loop
        self.client = async_client
        self._sync_client = sync_client
        self.url = url

    def run(self, awaitable):
        """Execute coroutine using the dedicated loop."""
        return self.loop.run_until_complete(awaitable)

    def flush(self):
        """Flush database via synchronous client."""
        return self._sync_client.flushdb()

    def close_sync(self):
        try:
            self._sync_client.close()
        except Exception:
            _ = None

    def __getattr__(self, item):
        return getattr(self.client, item)


@pytest.fixture
def redis_client():
    """Provide a real Redis client when available; skip otherwise."""
    try:
        import redis  # type: ignore
        import redis.asyncio as aioredis  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        pytest.skip(f"redis library not available: {exc}")

    url = (
        os.getenv("TEST_REDIS_URL")
        or os.getenv("EMBEDDINGS_REDIS_URL")
        or os.getenv("REDIS_URL")
        or "redis://localhost:6379/0"
    )

    sync_client = redis.Redis.from_url(url, decode_responses=True)
    try:
        sync_client.ping()
    except Exception as exc:
        sync_client.close()
        pytest.skip(f"Redis not reachable at {url}: {exc}")

    # Clean slate before tests
    try:
        sync_client.flushdb()
    except Exception:
        _ = None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        async_client = aioredis.from_url(url, decode_responses=True)
        if inspect.isawaitable(async_client):
            async_client = loop.run_until_complete(async_client)
        loop.run_until_complete(async_client.ping())
    except Exception as exc:
        loop.close()
        sync_client.close()
        pytest.skip(f"Failed to initialize async Redis client at {url}: {exc}")
    finally:
        asyncio.set_event_loop(None)

    previous_url = os.environ.get("EMBEDDINGS_REDIS_URL")
    os.environ["EMBEDDINGS_REDIS_URL"] = url

    harness = _RedisHarness(loop, async_client, sync_client, url)

    try:
        yield harness
    finally:
        if previous_url is None:
            os.environ.pop("EMBEDDINGS_REDIS_URL", None)
        else:
            os.environ["EMBEDDINGS_REDIS_URL"] = previous_url

        try:
            harness.flush()
        except Exception:
            _ = None
        try:
            harness.run(harness.client.close())
        except Exception:
            _ = None
        try:
            harness.loop.run_until_complete(harness.loop.shutdown_asyncgens())
        except Exception:
            _ = None
        try:
            harness.loop.close()
        except Exception:
            _ = None
        harness.close_sync()


# Lightweight app client + auth fixtures for property/unit tests in this package
@pytest.fixture
def test_client(disable_heavy_startup):
    """Minimal TestClient with CSRF and auth header set.

    Scope: function - keeps isolation across property-based runs.
    """
    try:
        csrf = "test-csrf"
        with TestClient(app) as client:
            # Double-submit CSRF: cookie + header
            client.cookies.set("csrf_token", csrf)
            client.headers["X-CSRF-Token"] = csrf
            # Accept Authorization in single-user mode
            client.headers["Authorization"] = f"Bearer {get_settings().SINGLE_USER_API_KEY}"
            yield client
    finally:
        # Ensure dependency overrides do not leak across tests
        try:
            app.dependency_overrides.clear()
        except Exception:
            _ = None


@pytest.fixture
def auth_headers():
    csrf = "test-csrf"
    api_key = get_settings().SINGLE_USER_API_KEY
    return {
        "Authorization": f"Bearer {api_key}",
        "X-API-KEY": api_key,
        "X-CSRF-Token": csrf,
        "Content-Type": "application/json",
    }


@pytest.fixture
def regular_user():
    return User(id=1, username="testuser", email="t@example.com", is_active=True, is_admin=False)


@pytest.fixture(autouse=True)
def _reset_app_lifecycle_state():
    """Clear stale lifecycle state on this conftest's pinned ``app``.

    Kept, but no longer for the reason it was written. It was added for #2581
    (46F -> 0F) because ``reload_app_main()`` permanently swapped
    ``sys.modules["tldw_Server_API.app.main"]``, so the root conftest's reset
    landed on the new app while the drained original -- the one every pinned
    test still routed through -- stayed drained and 503'd every request.

    Both halves of that are fixed now. The root reset covers every app that has
    lifecycle state, not just the current module's, and reloads no longer outlive
    the test that performed them (#2585). Removing this fixture leaves zero
    ``shutdown_in_progress`` responses in a full Embeddings run.

    It stays because removing it still perturbs four orchestrator-parity tests,
    which fail with a provider-authentication error that has nothing to do with
    lifecycle state and pass in isolation either way. That is an ordering
    dependency somewhere else in this suite; until it is understood, dropping
    this fixture would trade one known-good state for an unknown one.
    """
    from tldw_Server_API.app.services.app_lifecycle import reset_lifecycle_state

    reset_lifecycle_state(app)
    yield


@pytest.fixture(autouse=True)
def _sanitize_jsonschema_module(monkeypatch):
    """Ensure sys.modules['jsonschema'] is a proper ModuleType when present.

    Some tests stub 'jsonschema' with a SimpleNamespace for targeted assertions.
    Hypothesis inspects sys.modules and expects hashable module objects; wrapping
    the stub in a ModuleType avoids TypeError from unhashable SimpleNamespace.
    """
    import sys as _sys
    import types as _types
    mod = _sys.modules.get("jsonschema")
    if mod is not None and not isinstance(mod, _types.ModuleType):
        wrapper = _types.ModuleType("jsonschema")
        # Carry over commonly used attributes if present
        for attr in ("validate",):
            try:
                setattr(wrapper, attr, getattr(mod, attr))
            except Exception:
                _ = None
        monkeypatch.setitem(_sys.modules, "jsonschema", wrapper)


@pytest.fixture(autouse=True)
def _patch_hypothesis_local_constants(monkeypatch):
    """Patch Hypothesis provider constants discovery to tolerate unhashable stubs.

    Some tests insert non-module stubs (e.g., SimpleNamespace) into sys.modules.
    Hypothesis scans sys.modules and assumes hashable values; guard this by
    attempting to sanitize and retry when a TypeError arises.
    """
    try:
        from hypothesis.internal.conjecture import providers as _providers  # type: ignore
    except Exception:
        return

    orig = getattr(_providers, "_get_local_constants", None)
    if not callable(orig):
        return

    def _safe_get_local_constants():  # type: ignore[return-type]
        try:
            return orig()
        except TypeError:
            # Sanitize sys.modules: wrap unhashable stubs with ModuleType
            import sys as _sys
            import types as _types
            for name, mod in list(_sys.modules.items()):
                try:
                    hash(mod)
                    continue
                except Exception:
                    _ = None
                if isinstance(mod, _types.SimpleNamespace):
                    wrapper = _types.ModuleType(name)
                    for attr in dir(mod):
                        if attr.startswith("__") and attr.endswith("__"):
                            continue
                        try:
                            setattr(wrapper, attr, getattr(mod, attr))
                        except Exception:
                            _ = None
                    _sys.modules[name] = wrapper
            try:
                return orig()
            except Exception:
                # Fallback to existing cached constants if available
                return getattr(_providers, "_local_constants", None)

    monkeypatch.setattr(_providers, "_get_local_constants", _safe_get_local_constants, raising=False)


# ---------------------------------------------------------------------------
# Optional PGVector fixtures - skip when not available in this environment
# ---------------------------------------------------------------------------

@pytest.fixture
def pgvector_dsn():  # pragma: no cover - test helper for environments without PG
    pytest.skip("pgvector DSN not available in this test run")


@pytest.fixture
def pgvector_temp_table(pgvector_dsn):  # pragma: no cover - test helper for environments without PG
    pytest.skip("pgvector temporary table not available in this test run")
