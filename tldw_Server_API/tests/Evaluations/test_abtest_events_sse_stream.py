import os
import json
import time
import threading
import tempfile
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    get_unified_evaluation_service_for_user,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.AuthNZ.settings import get_settings


def _auth_headers(client: TestClient):
    # Use the actual API key from settings for single_user mode
    settings = get_settings()
    csrf = getattr(client, "csrf_token", "")
    return {"X-API-KEY": settings.SINGLE_USER_API_KEY, "X-CSRF-Token": csrf}


@pytest.mark.unit
def test_embeddings_abtest_events_sse_smoke_heartbeat_and_done(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    # Use a per-test DB file for Evaluations to avoid cross-test interference
    with tempfile.NamedTemporaryFile(suffix="_evals.db", delete=False) as f:
        eval_db_path = f.name

    try:
        # Force single-user auth mode for this test and reset settings so
        # Evaluations auth uses X-API-KEY as expected even if prior tests
        # switched to multi-user.
        prev_auth_mode = os.environ.get("AUTH_MODE")
        os.environ["AUTH_MODE"] = "single_user"
        reset_settings()

        # Configure fast heartbeats in data mode for reliable detection
        os.environ.setdefault("TEST_MODE", "1")
        monkeypatch.setenv("STREAM_HEARTBEAT_MODE", "data")
        monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_S", "0.05")
        os.environ.setdefault("EVALUATIONS_TEST_DB_PATH", eval_db_path)

        with TestClient(app) as client:
            # Get CSRF token cookie (some deployments may require it)
            resp = client.get("/api/v1/health")
            client.csrf_token = resp.cookies.get("csrf_token", "")

            # 1) Create a minimal A/B test
            create_payload = {
                "name": "sse_smoke",
                "config": {
                    "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
                    "media_ids": [],
                    "retrieval": {"k": 3, "search_mode": "vector"},
                    "queries": [{"text": "hello"}],
                },
            }
            r = client.post(
                "/api/v1/evaluations/embeddings/abtest",
                json=create_payload,
                headers=_auth_headers(client),
            )
            assert r.status_code == 200, r.text
            test_id = r.json()["test_id"]

            # 2) Flip the test to completed after a short delay to allow heartbeats
            def _complete_later():
                             # Allow a couple of heartbeat intervals to pass
                time.sleep(0.2)
                try:
                    user_id = DatabasePaths.get_single_user_id()
                    svc = get_unified_evaluation_service_for_user(user_id)
                    svc.db.set_abtest_status(test_id, "completed", stats_json={"progress": {"phase": 1.0}})
                except Exception:
                    # Do not fail the test from the helper thread
                    _ = None

            t = threading.Thread(target=_complete_later, daemon=True)
            t.start()

            # 3) Start the SSE stream and collect buffered result
            sse = client.get(
                f"/api/v1/evaluations/embeddings/abtest/{test_id}/events",
                headers={**_auth_headers(client), "Accept": "text/event-stream"},
            )
            assert sse.status_code == 200
            assert "text/event-stream" in sse.headers.get("content-type", "").lower()

            lines = sse.text.splitlines()
            # Expect at least one heartbeat in data mode and a final [DONE]
            heartbeat_seen = any(
                line.startswith("data: ") and "\"heartbeat\": true" in line for line in lines
            )
            done_seen = any(line.strip() == "data: [DONE]" for line in lines)
            assert heartbeat_seen, f"No heartbeat frame found in SSE lines: {lines[:10]}..."
            assert done_seen, f"No [DONE] frame found in SSE lines: {lines[-10:]}"

    finally:
        # Restore auth mode and reset settings for subsequent tests
        try:
            if prev_auth_mode is None:
                os.environ.pop("AUTH_MODE", None)
            else:
                os.environ["AUTH_MODE"] = prev_auth_mode
            reset_settings()
        except Exception:
            _ = None

        try:
            os.unlink(eval_db_path)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_embeddings_abtest_events_treats_cancelled_as_terminal(monkeypatch):
    import asyncio

    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as unified_mod
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    from tldw_Server_API.app.core.Streaming import streams as streams_mod

    class _DBStub:
        def get_abtest(self, test_id, created_by=None):
            _ = created_by
            return {
                "test_id": test_id,
                "status": "cancelled",
                "stats_json": json.dumps({"progress": {"phase": 1.0}}),
            }

    class _SvcStub:
        def __init__(self):
            self.db = _DBStub()

    class _FakeSSEStream:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            self.frames = []
            self.done_called = False

        async def send_json(self, payload):
            self.frames.append(f"data: {json.dumps(payload)}\n\n")

        async def error(self, code, message):
            raise AssertionError(f"Unexpected SSE error: {code} {message}")

        async def done(self):
            self.done_called = True
            self.frames.append("data: [DONE]\n\n")

        async def iter_sse(self):
            sent = 0
            idle_loops = 0
            while True:
                while sent < len(self.frames):
                    frame = self.frames[sent]
                    sent += 1
                    yield frame
                if self.done_called:
                    return
                idle_loops += 1
                if idle_loops > 5:
                    raise AssertionError("SSE stream did not terminate for cancelled status")
                await asyncio.sleep(0)

    monkeypatch.setattr(unified_mod, "get_unified_evaluation_service_for_user", lambda _user_id: _SvcStub())
    monkeypatch.setattr(streams_mod, "SSEStream", _FakeSSEStream)

    response = await unified_mod.stream_embeddings_abtest_events(
        test_id="abt_cancelled",
        user_ctx="tenant-user",
        _=None,
        current_user=User(id="tenant-user", username="tester", email=None, is_active=True),
    )

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

    payload = "".join(chunks)
    assert '"status": "cancelled"' in payload
    assert "data: [DONE]" in payload
