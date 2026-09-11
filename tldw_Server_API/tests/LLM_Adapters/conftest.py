"""Local conftest for LLM_Adapters tests.

Provides:
- Lightweight stub for app.main when real app import fails (unit-only cases)
- Access to shared chat fixtures (client, authenticated_client, auth headers)
  via project-level plugin registration in pyproject.toml
- Backward-compat fixture alias client_user_only used by some tests

Note on plugin discovery and subtree runs
- These tests rely on the Chat fixtures plugin declared in pyproject.toml.
- When running tests from this subtree (or individual files) and the runner
  doesn’t load project-level plugins (e.g., PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  or non-root working directory), fixtures like `authenticated_client` may be
  missing. In those cases, explicitly importing the plugin at module level
  (as some tests already do) or adding a local pytest_plugins declaration will
  restore deterministic fixture availability.
"""

import copy
import os
import sys
import types

import pytest

from tldw_Server_API.tests._plugins.authnz_fixtures import authnz_schema_ready_sync  # noqa: F401
from tldw_Server_API.tests.helpers.app_main_state import restore_app_main, set_app_main, snapshot_app_main

_DISABLED_ROUTE_KEYS_FOR_ADAPTER_TESTS = [
    "media",
    "media-embeddings",
    "audio",
    "audio-websocket",
    "audio-jobs",
]


def _merged_csv_env_value(name: str, values: list[str]) -> str:
    existing_raw = str(os.getenv(name, "") or "")
    parts = [p.strip() for p in existing_raw.replace(" ", ",").split(",") if p.strip()]
    lowered = {p.lower() for p in parts}
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized.lower() not in lowered:
            parts.append(normalized)
            lowered.add(normalized.lower())
    return ",".join(parts)


# Keep LLM adapter integration tests isolated from heavy media/audio route imports
# that pull native STT/ML stacks during app startup in constrained environments.
# Scope this to adapter tests only; setting it during collection leaks into
# later packages that need media routes from the shared FastAPI app.

# Adapter integration tests exercise request/response behavior and provider
# shims, not provider inventory enforcement. Keep strict model selection off by
# default for this package to avoid false-negative 400s when fixtures use
# deterministic model aliases; callers can still override explicitly.
os.environ.setdefault("CHAT_ENFORCE_STRICT_MODEL_SELECTION", "0")


def _install_app_main_stub() -> None:
    """Install a minimal tldw_Server_API.app.main stub when needed.

    Some unit-oriented adapter tests do not need the full FastAPI app import,
    and importing app.main can trigger heavy runtime initialization in constrained
    environments. Keep a lightweight stub available for fallback fixtures.
    """
    if "tldw_Server_API.app.main" in sys.modules:
        return
    m = types.ModuleType("tldw_Server_API.app.main")

    class _StubApp:  # pragma: no cover - simple container
        def __init__(self):
            self.dependency_overrides = {}
            self.state = types.SimpleNamespace()

    m.app = _StubApp()
    set_app_main(m)


def _resolve_app_for_fallback_client():
    """Resolve app.main for fallback TestClient construction.

    Defaults to the lightweight stub to avoid importing the full app in unit
    adapter runs. Set LLM_ADAPTERS_TEST_IMPORT_REAL_APP_MAIN=1 to force a real
    import in environments where that's desired.
    """
    import_real = str(os.getenv("LLM_ADAPTERS_TEST_IMPORT_REAL_APP_MAIN", "0")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    if import_real:
        try:
            from tldw_Server_API.app.main import app as fastapi_app
            return fastapi_app
        except Exception:
            _install_app_main_stub()
            from tldw_Server_API.app.main import app as fastapi_app
            return fastapi_app

    _install_app_main_stub()
    from tldw_Server_API.app.main import app as fastapi_app
    return fastapi_app


@pytest.fixture(autouse=True)
def _preserve_app_main_state(monkeypatch):
    monkeypatch.setenv(
        "ROUTES_DISABLE",
        _merged_csv_env_value("ROUTES_DISABLE", _DISABLED_ROUTE_KEYS_FOR_ADAPTER_TESTS),
    )
    previous_main = snapshot_app_main()
    yield
    restore_app_main(previous_main)


@pytest.fixture(autouse=True)
def _isolate_provider_override_cache():
    """Prevent unit-test refresh workers from poisoning later adapter tests."""
    from tldw_Server_API.app.core.AuthNZ import (
        llm_provider_overrides as overrides_module,
    )

    with overrides_module._OVERRIDE_LOCK:
        original = copy.deepcopy(overrides_module._OVERRIDE_CACHE)
        original_healthy = overrides_module._OVERRIDE_CACHE_HEALTHY
        original_ttl_enabled = (
            not overrides_module._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
        )
    overrides_module.set_llm_provider_overrides_cache_for_tests({})
    try:
        yield
    finally:
        overrides_module.set_llm_provider_overrides_cache_for_tests(
            original,
            healthy=original_healthy,
            ttl_enabled=original_ttl_enabled,
        )

# Shared chat fixtures are registered at the repository root conftest.py
# However, when running this subtree in isolation or with plugin autoloading
# disabled, those fixtures may be unavailable. Provide lightweight fallbacks
# here to keep this package's tests runnable in isolation without requiring
# project-level plugin loading.


@pytest.fixture
def client_user_only(request):  # noqa: D401 - compatibility alias with fallback
    """Return an authenticated TestClient.

    Prefer the project-level `client_with_single_user` fixture, which returns
    a FastAPI TestClient with auth dependencies overridden for tests.

    Important:
    Do not resolve `authenticated_client` dynamically here. A globally-loaded
    e2e fixture uses the same name and can resolve to `tests.e2e.fixtures.APIClient`,
    which does not expose `.stream()` and breaks streaming adapter tests.
    """
    try:
        request.getfixturevalue("authnz_schema_ready_sync")
        client, _logger = request.getfixturevalue("client_with_single_user")
        return client
    except Exception:
        # Last resort: construct a minimal TestClient against the app
        from fastapi.testclient import TestClient
        fastapi_app = _resolve_app_for_fallback_client()
        return TestClient(fastapi_app)


@pytest.fixture
def client(client_user_only):
    """Provide a TestClient with a helper 'post_with_auth' used by some tests.

    This mirrors the helper from the chat_fixtures plugin but works with the
    local fallback client when plugins are not loaded.
    """
    test_client = client_user_only

    # Best-effort CSRF token discovery (the health endpoint sets it)
    try:
        resp = test_client.get("/api/v1/health")
        csrf_token = getattr(test_client, "csrf_token", None) or resp.cookies.get("csrf_token", "")
        setattr(test_client, "csrf_token", csrf_token)
    except Exception:
        csrf_token = ""

    def post_with_auth(url, auth_token, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        if csrf_token and "X-CSRF-Token" not in headers:
            headers["X-CSRF-Token"] = csrf_token
        # Attach an auth header if provided; most tests run with request-user
        # dependency overridden, so this is optional but harmless.
        try:
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings
            settings = get_settings()
            if settings.AUTH_MODE == "multi_user":
                headers.setdefault("Authorization", auth_token)
            else:
                headers.setdefault("X-API-KEY", auth_token)
        except Exception:
            _ = None
        return test_client.post(url, headers=headers, **kwargs)

    setattr(test_client, "post_with_auth", post_with_auth)
    return test_client


@pytest.fixture
def authenticated_client(client_user_only, auth_token):
    """Return a client that automatically applies auth/CSRF headers.

    Streaming adapter tests call `client.stream(...)` directly, so this fixture
    wraps request methods to ensure they remain authenticated without each test
    manually constructing headers.
    """
    test_client = client_user_only

    # Best-effort CSRF token discovery.
    try:
        resp = test_client.get("/api/v1/health")
        csrf_token = getattr(test_client, "csrf_token", None) or resp.cookies.get("csrf_token", "")
        setattr(test_client, "csrf_token", csrf_token)
    except Exception:
        csrf_token = getattr(test_client, "csrf_token", "")

    original_post = test_client.post
    original_get = test_client.get
    original_stream = getattr(test_client, "stream", None)

    def _auth_headers(existing=None):
        headers = dict(existing or {})
        if csrf_token and "X-CSRF-Token" not in headers:
            headers["X-CSRF-Token"] = csrf_token
        try:
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings
            settings = get_settings()
            if settings.AUTH_MODE == "multi_user":
                token = auth_token if str(auth_token).startswith("Bearer ") else f"Bearer {auth_token}"
                headers.setdefault("Authorization", token)
            else:
                headers.setdefault("X-API-KEY", auth_token)
        except Exception:
            headers.setdefault("X-API-KEY", auth_token)
        return headers

    def authenticated_post(url, **kwargs):
        headers = _auth_headers(kwargs.pop("headers", None))
        return original_post(url, headers=headers, **kwargs)

    def authenticated_get(url, **kwargs):
        headers = _auth_headers(kwargs.pop("headers", None))
        return original_get(url, headers=headers, **kwargs)

    def authenticated_stream(method, url, **kwargs):
        headers = _auth_headers(kwargs.pop("headers", None))
        if callable(original_stream):
            return original_stream(method, url, headers=headers, **kwargs)
        raise RuntimeError("Test client does not support streaming in this environment")

    test_client.post = authenticated_post
    test_client.get = authenticated_get
    if callable(original_stream):
        test_client.stream = authenticated_stream
    return test_client


@pytest.fixture
def auth_token():  # pragma: no cover - simple token helper
    """Provide a plausible auth token for tests that include it in requests."""
    try:
        import os as _os

        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        settings = get_settings()
        if settings.AUTH_MODE == "multi_user":
            # Not generating a real JWT here; tests using this fixture don't validate it.
            return "Bearer test-token"
        return _os.environ.get("SINGLE_USER_API_KEY", "test-api-key")
    except Exception:
        return "test-api-key"
