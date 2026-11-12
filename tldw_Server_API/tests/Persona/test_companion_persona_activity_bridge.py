from pathlib import Path
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.session_manager import SessionManager
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.tests.Persona.test_persona_ws import _seed_persona_session


pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


def _recv_until(client, predicate, timeout=2.0):
    import time

    start = time.time()
    while time.time() - start < timeout:
        msg = client.receive_text()
        try:
            data = json.loads(msg)
        except Exception:
            continue
        if predicate(data):
            return data
    raise AssertionError("Expected event not received in time")


@pytest.fixture()
def persona_client_with_companion_opt_in(monkeypatch, tmp_path):
    user_id = 1
    base_dir = tmp_path / "user_db"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("TEST_MODE", "1")

    persona_db = CharactersRAGDB(
        str(DatabasePaths.get_chacha_db_path(user_id)),
        client_id="companion-persona-tests",
    )
    personalization_db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(user_id)))
    personalization_db.update_profile(str(user_id), enabled=1)
    session_manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: session_manager)

    async def override_user():
        return User(id=user_id, username="user-1", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: persona_db
    try:
        with TestClient(fastapi_app) as client:
            yield client, personalization_db
    finally:
        fastapi_app.dependency_overrides.clear()
        persona_db.close_connection()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_persona_session_creation_records_companion_activity(persona_client_with_companion_opt_in):
    client, personalization_db = persona_client_with_companion_opt_in

    created = client.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
    assert created.status_code == 200, created.text
    payload = created.json()

    events, total = personalization_db.list_companion_activity_events("1", limit=10)
    assert total == 1
    event = events[0]
    assert event["event_type"] == "persona_session_started"
    assert event["source_type"] == "persona_session"
    assert event["source_id"] == payload["session_id"]
    assert event["surface"] == "api.persona"
    assert event["provenance"]["capture_mode"] == "explicit"
    assert event["provenance"]["route"] == "/api/v1/persona/session"
    assert event["metadata"]["persona_id"] == payload["persona"]["id"]
    assert event["metadata"]["runtime_mode"] == payload["runtime_mode"]


def test_persona_session_creation_accepts_companion_surface_override(
    persona_client_with_companion_opt_in,
):
    client, personalization_db = persona_client_with_companion_opt_in

    created = client.post(
        "/api/v1/persona/session",
        json={
            "persona_id": "research_assistant",
            "surface": "companion.conversation",
        },
    )
    assert created.status_code == 200, created.text

    events, total = personalization_db.list_companion_activity_events("1", limit=10)
    assert total == 1
    event = events[0]
    assert event["event_type"] == "persona_session_started"
    assert event["surface"] == "companion.conversation"
    assert event["provenance"]["route"] == "/api/v1/persona/session"


def test_persona_stream_records_companion_summary_and_tool_activity(
    monkeypatch,
    tmp_path,
    persona_client_with_companion_opt_in,
):
    client, personalization_db = persona_client_with_companion_opt_in

    class _FakeServer:
        def __init__(self):
            self.initialized = True

        async def initialize(self):
            self.initialized = True

        async def handle_http_request(self, request, user_id=None, metadata=None):
            return SimpleNamespace(
                error=None,
                result={
                    "ok": True,
                    "saved": True,
                    "url": "https://example.com/article",
                    "secret": "persona-companion-raw-secret",
                },
            )

    async def _fake_resolve(*args, **kwargs):
        return "1", True, True

    monkeypatch.setattr(persona_ep, "get_mcp_server", lambda: _FakeServer())
    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", _fake_resolve)

    _seed_persona_session(
        tmp_path, monkeypatch, user_id="1", session_id="sess_companion_bridge", mode="session_scoped",
    )
    with client.websocket_connect("/api/v1/persona/stream") as ws:
        _ = json.loads(ws.receive_text())
        session_id = "sess_companion_bridge"
        ws.send_text(json.dumps({"type": "user_message", "session_id": session_id, "text": "https://example.com"}))
        plan = _recv_until(ws, lambda d: d.get("event") == "tool_plan")
        approved_steps = [int(step["idx"]) for step in plan.get("steps", [])]
        ws.send_text(
            json.dumps(
                {
                    "type": "confirm_plan",
                    "session_id": session_id,
                    "plan_id": plan["plan_id"],
                    "approved_steps": approved_steps,
                }
            )
        )
        _ = _recv_until(ws, lambda d: d.get("event") == "tool_call")
        tool_result = _recv_until(ws, lambda d: d.get("event") == "tool_result")
        assistant_delta = _recv_until(ws, lambda d: d.get("event") == "assistant_delta")

    events, total = personalization_db.list_companion_activity_events("1", limit=10)
    assert total == 2

    event_types = {event["event_type"] for event in events}
    assert event_types == {"persona_session_summarized", "persona_tool_executed"}

    tool_event = next(event for event in events if event["event_type"] == "persona_tool_executed")
    assert tool_event["source_id"] == f"{session_id}:{plan['plan_id']}:{tool_result['step_idx']}"
    assert tool_event["metadata"]["persona_id"] == "research_assistant"
    assert tool_event["metadata"]["tool_name"] == plan["steps"][0]["tool"]
    assert tool_event["metadata"]["step_type"] == plan["steps"][0]["step_type"]
    assert tool_event["metadata"]["ok"] is True
    assert tool_event["metadata"]["output_type"] == "dict"
    assert "persona-companion-raw-secret" not in str(tool_event["metadata"])
    assert tool_event["provenance"]["action"] == "tool_outcome"

    summary_event = next(event for event in events if event["event_type"] == "persona_session_summarized")
    assert summary_event["source_id"] == session_id
    assert summary_event["metadata"]["persona_id"] == "research_assistant"
    assert summary_event["metadata"]["plan_id"] == plan["plan_id"]
    assert summary_event["metadata"]["summary_preview"] == assistant_delta["text_delta"]
    assert summary_event["metadata"]["summary_char_count"] == len(assistant_delta["text_delta"])
    assert summary_event["provenance"]["action"] == "session_summary"


def test_companion_session_surface_propagates_to_persona_stream_activity(
    monkeypatch,
    persona_client_with_companion_opt_in,
):
    client, personalization_db = persona_client_with_companion_opt_in

    class _FakeServer:
        def __init__(self):
            self.initialized = True

        async def initialize(self):
            self.initialized = True

        async def handle_http_request(self, request, user_id=None, metadata=None):
            return SimpleNamespace(
                error=None,
                result={
                    "ok": True,
                    "saved": True,
                    "url": "https://example.com/companion",
                },
            )

    async def _fake_resolve(*args, **kwargs):
        return "1", True, True

    monkeypatch.setattr(persona_ep, "get_mcp_server", lambda: _FakeServer())
    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", _fake_resolve)

    created = client.post(
        "/api/v1/persona/session",
        json={
            "persona_id": "research_assistant",
            "surface": "companion.conversation",
        },
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]

    with client.websocket_connect("/api/v1/persona/stream") as ws:
        _ = json.loads(ws.receive_text())
        ws.send_text(
            json.dumps(
                {
                    "type": "user_message",
                    "session_id": session_id,
                    "text": "https://example.com/companion",
                }
            )
        )
        plan = _recv_until(ws, lambda d: d.get("event") == "tool_plan")
        approved_steps = [int(step["idx"]) for step in plan.get("steps", [])]
        ws.send_text(
            json.dumps(
                {
                    "type": "confirm_plan",
                    "session_id": session_id,
                    "plan_id": plan["plan_id"],
                    "approved_steps": approved_steps,
                }
            )
        )
        _ = _recv_until(ws, lambda d: d.get("event") == "tool_call")
        tool_result = _recv_until(ws, lambda d: d.get("event") == "tool_result")
        assistant_delta = _recv_until(ws, lambda d: d.get("event") == "assistant_delta")

    events, total = personalization_db.list_companion_activity_events("1", limit=10)
    assert total == 3

    started_event = next(event for event in events if event["event_type"] == "persona_session_started")
    assert started_event["surface"] == "companion.conversation"

    tool_event = next(event for event in events if event["event_type"] == "persona_tool_executed")
    assert tool_event["surface"] == "companion.conversation"
    assert tool_event["source_id"] == f"{session_id}:{plan['plan_id']}:{tool_result['step_idx']}"

    summary_event = next(
        event for event in events if event["event_type"] == "persona_session_summarized"
    )
    assert summary_event["surface"] == "companion.conversation"
    assert summary_event["metadata"]["summary_preview"] == assistant_delta["text_delta"]


def test_persona_session_resume_persists_companion_surface_override(
    persona_client_with_companion_opt_in,
):
    client, _personalization_db = persona_client_with_companion_opt_in

    created = client.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]

    resumed = client.post(
        "/api/v1/persona/session",
        json={
            "persona_id": "research_assistant",
            "resume_session_id": session_id,
            "surface": "companion.conversation",
        },
    )
    assert resumed.status_code == 200, resumed.text

    persona_db = fastapi_app.dependency_overrides[get_chacha_db_for_user]()
    session_row = persona_db.get_persona_session(session_id, user_id="1", include_deleted=False)
    assert session_row is not None
    assert session_row["activity_surface"] == "companion.conversation"


def test_companion_session_surface_survives_restart_when_persona_stream_resumes(
    monkeypatch,
    persona_client_with_companion_opt_in,
):
    client, personalization_db = persona_client_with_companion_opt_in

    class _FakeServer:
        def __init__(self):
            self.initialized = True

        async def initialize(self):
            self.initialized = True

        async def handle_http_request(self, request, user_id=None, metadata=None):
            return SimpleNamespace(
                error=None,
                result={
                    "ok": True,
                    "saved": True,
                    "url": "https://example.com/restart-proof",
                },
            )

    async def _fake_resolve(*args, **kwargs):
        return "1", True, True

    monkeypatch.setattr(persona_ep, "get_mcp_server", lambda: _FakeServer())
    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", _fake_resolve)

    created = client.post(
        "/api/v1/persona/session",
        json={
            "persona_id": "research_assistant",
            "surface": "companion.conversation",
        },
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]

    restarted_session_manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: restarted_session_manager)

    with client.websocket_connect("/api/v1/persona/stream") as ws:
        _ = json.loads(ws.receive_text())
        ws.send_text(
            json.dumps(
                {
                    "type": "user_message",
                    "session_id": session_id,
                    "text": "https://example.com/restart-proof",
                }
            )
        )
        plan = _recv_until(ws, lambda d: d.get("event") == "tool_plan")
        approved_steps = [int(step["idx"]) for step in plan.get("steps", [])]
        ws.send_text(
            json.dumps(
                {
                    "type": "confirm_plan",
                    "session_id": session_id,
                    "plan_id": plan["plan_id"],
                    "approved_steps": approved_steps,
                }
            )
        )
        _ = _recv_until(ws, lambda d: d.get("event") == "tool_call")
        tool_result = _recv_until(ws, lambda d: d.get("event") == "tool_result")
        assistant_delta = _recv_until(ws, lambda d: d.get("event") == "assistant_delta")

    events, total = personalization_db.list_companion_activity_events("1", limit=10)
    assert total == 3

    tool_event = next(event for event in events if event["event_type"] == "persona_tool_executed")
    assert tool_event["surface"] == "companion.conversation"
    assert tool_event["source_id"] == f"{session_id}:{plan['plan_id']}:{tool_result['step_idx']}"

    summary_event = next(
        event for event in events if event["event_type"] == "persona_session_summarized"
    )
    assert summary_event["surface"] == "companion.conversation"
    assert summary_event["metadata"]["summary_preview"] == assistant_delta["text_delta"]
