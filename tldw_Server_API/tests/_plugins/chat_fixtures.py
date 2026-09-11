"""Shared Chat fixtures extracted from the Chat test suite conftest.

This module is a pytest plugin (not a conftest), so it can be imported from
multiple test packages via `pytest_plugins` without triggering duplicate
plugin registrations.
"""

import asyncio
import pytest
import tempfile
import os
import json
import threading
import time
import atexit
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import Request
import datetime
from httpx import AsyncClient

try:
    # httpx >=0.23
    from httpx import ASGITransport
except Exception:  # pragma: no cover
    ASGITransport = None

# Set environment variables BEFORE any tldw imports
_ORIG_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
_ORIG_OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

if _ORIG_OPENAI_API_KEY is None:
    os.environ["OPENAI_API_KEY"] = ""
    _SET_MOCK_OPENAI_KEY = True
else:
    _SET_MOCK_OPENAI_KEY = False

_SET_MOCK_OPENAI_BASE = False
if not _ORIG_OPENAI_API_BASE and _SET_MOCK_OPENAI_KEY:
    os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"
    _SET_MOCK_OPENAI_BASE = True


# IMPORTANT: Ensure API_BEARER is not set - it causes wrong authentication path in single-user mode
if "API_BEARER" in os.environ:
    del os.environ["API_BEARER"]

def _get_app():
    # Lazy import to avoid heavy app imports during collection for tests that don't need Chat fixtures
    from tldw_Server_API.app.main import app as _app
    return _app
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
    DEFAULT_CHARACTER_NAME,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user


# Global state to track mock server
_mock_server_state = {"server": None, "thread": None, "base_url": None}

# Global variable to store original dependency overrides
_original_dependency_overrides = None


def _run_coro_safely(coro):
    """Run the given coroutine, even if an event loop is already running."""
    try:
        asyncio.run(coro)
        return
    except RuntimeError:
        # Fall back to a dedicated loop when we're inside an active asyncio loop.
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _stop_request_queue():
    """Stop the global chat request queue to avoid cross-test leakage."""
    try:
        from tldw_Server_API.app.core.Chat import request_queue as rq_mod
    except Exception:
        return

    queue = rq_mod.get_request_queue()
    if queue is None:
        return

    async def _stop():
        try:
            await queue.stop()
        except Exception:
            _ = None

    _run_coro_safely(_stop())

    # Clear the module-level singleton so future startups get a clean queue.
    try:
        rq_mod._request_queue = None  # type: ignore[attr-defined]
    except Exception:
        _ = None


def _reset_rate_limiter():
    """Reinitialise the chat rate limiter with the current TEST_* env values."""
    try:
        from tldw_Server_API.app.core.Chat.rate_limiter import initialize_rate_limiter

        initialize_rate_limiter()
    except Exception:
        _ = None


def cleanup_mock_server():
    """Cleanup function to ensure mock server is stopped."""
    server = _mock_server_state.get("server")
    thread = _mock_server_state.get("thread")

    if server:
        try:
            server.shutdown()
        except Exception:
            _ = None
        try:
            server.server_close()
        except Exception:
            _ = None

    if thread and thread.is_alive():
        thread.join(timeout=5)

    _mock_server_state["server"] = None
    _mock_server_state["thread"] = None
    _mock_server_state["base_url"] = None


# Register cleanup function
atexit.register(cleanup_mock_server)


@pytest.fixture(scope="session", autouse=True)
def preserve_app_state():
    """Preserve the original app dependency overrides across all tests."""
    global _original_dependency_overrides

    # Store the original state at the beginning of the test session
    app = _get_app()
    _original_dependency_overrides = app.dependency_overrides.copy()

    yield

    # Restore the original state at the end of the test session
    app = _get_app()
    app.dependency_overrides = _original_dependency_overrides.copy()

    # Restore OpenAI environment variables if we set test defaults
    if _SET_MOCK_OPENAI_KEY:
        if _ORIG_OPENAI_API_KEY is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = _ORIG_OPENAI_API_KEY
    if _SET_MOCK_OPENAI_BASE:
        if _ORIG_OPENAI_API_BASE is None:
            os.environ.pop("OPENAI_API_BASE", None)
        else:
            os.environ["OPENAI_API_BASE"] = _ORIG_OPENAI_API_BASE


@pytest.fixture(autouse=True)
def reset_app_overrides():
    """Reset app dependency overrides before each test."""
    global _original_dependency_overrides

    # Reset to original state before each test
    app = _get_app()
    if _original_dependency_overrides is not None:
        app.dependency_overrides = _original_dependency_overrides.copy()
    else:
        app.dependency_overrides.clear()

    _stop_request_queue()
    _reset_rate_limiter()

    yield

    # Clean up after each test
    app = _get_app()
    if _original_dependency_overrides is not None:
        app.dependency_overrides = _original_dependency_overrides.copy()
    else:
        app.dependency_overrides.clear()

    _stop_request_queue()
    _reset_rate_limiter()


@pytest.fixture(scope="session")
def mock_openai_server():
    """Start the mock OpenAI server for testing."""
    if _mock_server_state["server"]:
        yield _mock_server_state["base_url"]
        return

    class _MockOpenAIHandler(BaseHTTPRequestHandler):
        mock_completion = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-gpt",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "This is a test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

        def _send_json(self, payload, status_code=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server naming
            if self.path == "/v1/models":
                self._send_json({"data": [{"id": "mock-gpt"}]})
            else:
                self._send_json({"error": "Not found"}, status_code=404)

        def do_POST(self):  # noqa: N802 - http.server naming
            if self.path == "/v1/chat/completions":
                content_length = int(self.headers.get("Content-Length") or 0)
                if content_length:
                    # Consume request body to keep the socket healthy
                    self.rfile.read(content_length)
                response = dict(self.mock_completion)
                response["created"] = int(time.time())
                self._send_json(response)
            else:
                self._send_json({"error": "Not found"}, status_code=404)

        def log_message(self, format, *args):  # noqa: D401 - silence default logging
            return

    server = HTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, name="mock-openai-server", daemon=True)
    thread.start()

    # Store state for reuse
    _mock_server_state["server"] = server
    _mock_server_state["thread"] = thread
    _mock_server_state["base_url"] = f"http://127.0.0.1:{server.server_port}"

    # Give server moment to start
    time.sleep(0.1)

    yield _mock_server_state["base_url"]

    cleanup_mock_server()


@pytest.fixture
def configure_for_mock_server(mock_openai_server, monkeypatch):
    """Configure the application to use the mock OpenAI server."""
    # Ensure environment variables are set
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key-12345")
    monkeypatch.setenv("OPENAI_API_BASE", f"{mock_openai_server}/v1")

    # Also set custom endpoint variables
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", f"{mock_openai_server}/v1/chat/completions")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "sk-mock-key-12345")

    # Patch the OpenAI API URL in the config
    from tldw_Server_API.app.core.config import load_and_log_configs
    config = load_and_log_configs()
    if 'openai_api' not in config:
        config['openai_api'] = {}
    config['openai_api']['api_key'] = 'sk-mock-key-12345'
    config['openai_api']['api_base_url'] = f'{mock_openai_server}/v1'

    # Patch the load_and_log_configs function to return our patched config
    def mock_load_and_log_configs():
        return config

    monkeypatch.setattr('tldw_Server_API.app.core.config.load_and_log_configs', mock_load_and_log_configs)
    monkeypatch.setattr('tldw_Server_API.app.core.LLM_Calls.chat_calls.load_and_log_configs', mock_load_and_log_configs)

    yield


@pytest.fixture
def test_user():
    """Create a test user object."""
    return User(
        id=1,
        username="test_user",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture
def auth_token(test_user):
    """Generate a valid JWT token for the test user."""
    settings = get_settings()

    if settings.AUTH_MODE == "multi_user":
        jwt_service = get_jwt_service()
        # Use the correct method signature
        access_token = jwt_service.create_access_token(
            user_id=test_user.id,
            username=test_user.username,
            role="user"
        )
        return f"Bearer {access_token}"
    else:
        # In single-user mode, the value here is used in X-API-KEY header
        return os.environ.get("SINGLE_USER_API_KEY", "test-api-key")


@pytest.fixture
def mock_user_db(tmp_path):
    """Create a temporary user database and return a MagicMock adapter."""
    # Minimal mock; extend if tests need real behavior.
    mock_db = MagicMock()
    mock_db.db_path = str(tmp_path / "users.db")
    return mock_db


@pytest.fixture
def mock_chacha_db(tmp_path):
    """Create a temporary ChaChaNotes DB with a default character.

    Align the DB path with the API's dependency resolution so even without
    dependency overrides the server sees the same file:
      USER_DB_BASE_DIR = <tmp_path>
      DB path = <tmp_path>/1/ChaChaNotes.db (single-user id=1)
    """
    # Ensure API dependency resolves to our tmp_path base directory
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path)

    # In single-user mode, the request user id is 1
    user_dir = tmp_path / "1"
    user_dir.mkdir(parents=True, exist_ok=True)
    db_path = user_dir / "ChaChaNotes.db"

    db = CharactersRAGDB(str(db_path), client_id="pytest_client")

    # Ensure at least one default character exists
    char_id = db.add_character_card({
        "name": DEFAULT_CHARACTER_NAME,
        "description": "Default test character",
        "first_message": "Hello, User!",
        "personality": "Helpful",
        "scenario": "General",
        "system_prompt": "You are a helpful AI assistant."
    })
    print(f"Created default character with ID: {char_id}")

    yield db

    # Cleanup
    try:
        os.unlink(str(db_path))
    except Exception:
        _ = None


@pytest.fixture
def mock_media_db(test_user):
    """Create a mock media database."""
    mock_db = MagicMock()
    mock_db.client_id = f"user_{test_user.id}"
    return mock_db


@pytest.fixture
def setup_dependencies(test_user, mock_user_db, mock_chacha_db, mock_media_db):
    """Override all dependencies for testing."""
    settings = get_settings()
    app = _get_app()

    # Override authentication for tests in both modes
    async def mock_get_request_user(request: Request):
        # Match FastAPI dependency signature to avoid 422 from varargs
        return test_user
    app.dependency_overrides[get_request_user] = mock_get_request_user

    # Override databases
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chacha_db
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db

    yield

    # Cleanup - autouse fixture handles restore


@pytest.fixture
def client(setup_dependencies):
    """Create test client with CSRF handling."""
    # Make sure we're using the same app instance
    from tldw_Server_API.app.main import app as main_app
    # Reduce startup overhead and file handle usage during tests
    prev_validate = os.environ.get("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP")
    os.environ["PRIVILEGE_METADATA_VALIDATE_ON_STARTUP"] = "0"
    try:
        with TestClient(main_app) as test_client:
            # Get CSRF token
            response = test_client.get("/api/v1/health")
            csrf_token = response.cookies.get("csrf_token", "")
            test_client.csrf_token = csrf_token
            test_client.cookies = {"csrf_token": csrf_token}

            # Add helper method
            def post_with_auth(url, auth_token, **kwargs):
                headers = kwargs.pop("headers", {})
                headers["X-CSRF-Token"] = csrf_token

                settings = get_settings()
                if settings.AUTH_MODE == "multi_user":
                    headers["Authorization"] = auth_token
                else:
                    # Use X-API-KEY header for single-user mode
                    headers["X-API-KEY"] = auth_token

                return test_client.post(url, headers=headers, **kwargs)

            test_client.post_with_auth = post_with_auth

            yield test_client
    finally:
        # Restore env override
        if prev_validate is None:
            os.environ.pop("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", None)
        else:
            os.environ["PRIVILEGE_METADATA_VALIDATE_ON_STARTUP"] = prev_validate


@pytest.fixture
def authenticated_client(client, auth_token, setup_dependencies, mock_chacha_db):
    """Create an authenticated test client and harden dependency overrides per-request.

    - Ensures auth headers and CSRF are attached
    - Reapplies ChaCha DB override before each request to avoid cross-test resets
    """
    settings = get_settings()

    # Base methods we will wrap
    original_post = client.post
    original_get = client.get
    original_stream = getattr(client, "stream", None)

    def _apply_auth_and_overrides(headers: dict) -> dict:
        # Include CSRF token if available on the client
        csrf_token = getattr(client, "csrf_token", None)
        if csrf_token and "X-CSRF-Token" not in headers:
            headers["X-CSRF-Token"] = csrf_token

        # Attach auth header based on mode
        if settings.AUTH_MODE == "multi_user":
            headers.setdefault("Authorization", auth_token)
        else:
            headers.setdefault("X-API-KEY", auth_token)

        # Re-apply the DB override defensively to ensure the API uses the same DB
        # instance the test created data in, even if other tests reset overrides.
        try:
            _app = _get_app()
            _app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chacha_db
        except Exception:
            _ = None
        return headers

    def authenticated_post(url, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        headers = _apply_auth_and_overrides(headers)
        return original_post(url, headers=headers, **kwargs)

    def authenticated_get(url, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        headers = _apply_auth_and_overrides(headers)
        return original_get(url, headers=headers, **kwargs)

    def authenticated_stream(method, url, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        headers = _apply_auth_and_overrides(headers)
        if callable(original_stream):
            return original_stream(method, url, headers=headers, **kwargs)
        raise RuntimeError("Test client does not support streaming in this environment")

    client.post = authenticated_post
    client.get = authenticated_get
    if callable(original_stream):
        client.stream = authenticated_stream
    return client


def get_auth_headers(auth_token, csrf_token=""):
    """Helper function to get authentication headers."""
    settings = get_settings()
    headers = {"X-CSRF-Token": csrf_token}

    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
    else:
        # Use X-API-KEY header for single-user mode
        headers["X-API-KEY"] = auth_token

    return headers


@pytest.fixture
def auth_headers(auth_token):
    """Provide default auth headers for requests (works with AsyncClient too)."""
    settings = get_settings()
    if settings.AUTH_MODE == "multi_user":
        token = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
        return {"Authorization": token, "X-CSRF-Token": ""}
    else:
        return {"X-API-KEY": auth_token, "X-CSRF-Token": ""}


@pytest.fixture
async def async_client():
    """Yield an AsyncClient bound to the FastAPI app for ASGI tests."""
    # Lazily resolve the FastAPI app for this plugin
    app = _get_app()
    transport = ASGITransport(app=app) if ASGITransport else None
    kwargs = {"base_url": "http://test"}
    if transport is not None:
        kwargs["transport"] = transport
    else:  # Fallback for older httpx: pass app directly
        kwargs["app"] = app
    async with AsyncClient(**kwargs) as ac:
        yield ac


# Additional fixtures for unit tests can be added below
