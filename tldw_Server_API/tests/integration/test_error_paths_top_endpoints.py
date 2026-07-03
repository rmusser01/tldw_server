"""Error-path sweep: unauthenticated and malformed-body cases for top endpoints (audit F6)."""
import os

# Audio routers are skipped under pytest by default (see
# tldw_Server_API/app/core/testing.py::audio_imports_enabled_for_runtime and
# tldw_Server_API/app/api/v1/router_groups/content.py); opt in explicitly so
# /api/v1/audio/* routes below actually exist on the app under test. Must be
# set before the first `from tldw_Server_API.app.main import app` in this
# process (mirrors tests/Audio/conftest.py and tests/TTS_NEW/conftest.py).
os.environ.setdefault("MINIMAL_TEST_INCLUDE_AUDIO", "1")

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

# (method, path, minimal-but-malformed JSON body or None for GET)
PROTECTED_ROUTES = [
    ("POST", "/api/v1/chat/completions", {"model": 123}),          # model must be str
    ("POST", "/api/v1/embeddings", {"input": None}),               # input required
    ("POST", "/api/v1/rag/search", {"query": None}),               # query required
    ("POST", "/api/v1/media/search", None),                        # GET is 405; endpoint is POST-only
    ("POST", "/api/v1/audio/transcriptions", {}),                  # multipart file required
    ("POST", "/api/v1/audio/speech", {"input": 42}),               # input must be str
    ("GET", "/api/v1/notes/", None),
    ("GET", "/api/v1/prompts/", None),
    ("GET", "/api/v1/characters/", None),
    ("GET", "/api/v1/mcp/tools", None),                            # mcp/status is public by design (200 anon)
]


@pytest.fixture(scope="module")
def anon_client():
    """Client with NO auth override and NO API key: every request is anonymous."""
    from tldw_Server_API.app.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("method,path,body", PROTECTED_ROUTES,
                         ids=[f"{m}-{p}" for m, p, _ in PROTECTED_ROUTES])
def test_unauthenticated_request_is_rejected(anon_client, method, path, body):
    resp = anon_client.request(method, path, json=body)
    assert resp.status_code in (401, 403), (
        f"{method} {path} returned {resp.status_code}; expected auth rejection. "
        f"404 means the path is wrong - fix it from /openapi.json, do not delete the case."
    )


MALFORMED_BODY_ROUTES = [(m, p, b) for m, p, b in PROTECTED_ROUTES if b is not None]


@pytest.mark.parametrize("method,path,body", MALFORMED_BODY_ROUTES,
                         ids=[f"{m}-{p}" for m, p, _ in MALFORMED_BODY_ROUTES])
def test_malformed_body_is_422_not_500(client_user_only, method, path, body):
    resp = client_user_only.request(method, path, json=body)
    assert 400 <= resp.status_code < 500, (
        f"{method} {path} returned {resp.status_code} for malformed input; "
        f"validation errors must never be 5xx."
    )
