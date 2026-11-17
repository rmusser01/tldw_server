from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Personalization.companion_context import load_companion_context


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


@pytest.fixture(autouse=True)
def _mock_persona_auth(monkeypatch):
    async def _fake_resolve(*args, **kwargs):
        return "1", True, True

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", _fake_resolve)


def _seed_personalization_db(tmp_path, monkeypatch, *, user_id: str, enabled: bool) -> PersonalizationDB:
    base = tmp_path / "user_db"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base))
    path = DatabasePaths.get_personalization_db_path(int(user_id))
    db = PersonalizationDB(str(path))
    db.update_profile(user_id, enabled=1 if enabled else 0)
    return db


def test_persona_companion_context_ranking_prefers_query_matching_items(tmp_path, monkeypatch):
    user_id = "911"
    db = _seed_personalization_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    _ = db.insert_companion_activity_event(
        user_id=user_id,
        event_type="note_updated",
        source_type="note",
        source_id="42",
        surface="api.notes",
        dedupe_key="note.updated:42",
        tags=["backlog", "review"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/notes/42"},
        metadata={"title": "Backlog review notes"},
    )
    _ = db.insert_companion_activity_event(
        user_id=user_id,
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id="43",
        surface="api.reading",
        dedupe_key="reading.save:43",
        tags=["gardening"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/reading/save"},
        metadata={"title": "Gardening checklist"},
    )
    _ = db.upsert_companion_knowledge_card(
        user_id=user_id,
        card_type="project_focus",
        title="Backlog review",
        summary="Recent explicit activity clusters around backlog review.",
        evidence=[{"source_id": "42"}],
        score=0.8,
        status="active",
    )
    _ = db.upsert_companion_knowledge_card(
        user_id=user_id,
        card_type="topic_focus",
        title="Gardening",
        summary="Recent explicit activity clusters around gardening.",
        evidence=[{"source_id": "43"}],
        score=0.7,
        status="active",
    )

    async def _fake_resolve(*args, **kwargs):
        return user_id, True, True

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", _fake_resolve)

    with TestClient(fastapi_app) as c:
        with c.websocket_connect("/api/v1/persona/stream") as ws:
            connected = json.loads(ws.receive_text())
            ws.send_text(
                json.dumps(
                    {
                        "type": "user_message",
                        "session_id": connected["session_id"],
                        "text": "help me resume the backlog review",
                    }
                )
            )
            plan = _recv_until(ws, lambda d: d.get("event") == "tool_plan")

    companion_payload = plan.get("companion") or {}
    assert companion_payload.get("enabled") is True
    assert companion_payload.get("mode") == "ranked"
    query_value = str(plan["steps"][0]["args"]["query"])
    assert "Backlog review" in query_value
    assert "Gardening checklist" not in query_value


def test_load_companion_context_conversation_prompts_returns_ranked_candidates(
    tmp_path,
    monkeypatch,
):
    user_id = "912"
    db = _seed_personalization_db(tmp_path, monkeypatch, user_id=user_id, enabled=True)
    backlog_event_id = db.insert_companion_activity_event(
        user_id=user_id,
        event_type="note_updated",
        source_type="note",
        source_id="42",
        surface="api.notes",
        dedupe_key="note.updated:42",
        tags=["backlog", "review"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/notes/42"},
        metadata={"title": "Backlog review notes"},
    )
    _ = db.insert_companion_activity_event(
        user_id=user_id,
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id="43",
        surface="api.reading",
        dedupe_key="reading.save:43",
        tags=["gardening"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/reading/save"},
        metadata={"title": "Gardening checklist"},
    )
    backlog_card_id = db.upsert_companion_knowledge_card(
        user_id=user_id,
        card_type="project_focus",
        title="Backlog review",
        summary="Recent explicit activity clusters around backlog review.",
        evidence=[{"source_event_id": backlog_event_id}],
        score=0.8,
        status="active",
    )

    payload = load_companion_context(
        user_id=user_id,
        query="resume backlog review",
        include_candidates=True,
    )

    assert payload["mode"] == "ranked"
    assert payload["card_ids"] == [backlog_card_id]
    assert payload["cards"][0]["title"] == "Backlog review"
