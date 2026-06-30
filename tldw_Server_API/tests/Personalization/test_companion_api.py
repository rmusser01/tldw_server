from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.endpoints import companion as companion_ep
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import get_personalization_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.main import app as fastapi_app


pytestmark = pytest.mark.unit


def _build_client_with_companion_db(tmp_path, *, enabled: bool):
    db = PersonalizationDB(str(tmp_path / "personalization.db"))
    db.update_profile("1", enabled=1 if enabled else 0)

    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    def override_db_dep():
        return db

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_personalization_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client, db

    fastapi_app.dependency_overrides.clear()


@pytest.fixture()
def client_with_companion_db(tmp_path):
    yield from _build_client_with_companion_db(tmp_path, enabled=True)


@pytest.fixture()
def client_with_companion_db_opted_out(tmp_path):
    yield from _build_client_with_companion_db(tmp_path, enabled=False)


def test_companion_activity_endpoint_returns_provenance(client_with_companion_db) -> None:
    client, db = client_with_companion_db
    db.insert_companion_activity_event(
        user_id="1",
        event_type="reading.saved",
        source_type="reading_item",
        source_id="42",
        surface="reading",
        dedupe_key="reading.saved:reading_item:42",
        tags=["research", "project-alpha"],
        provenance={"source_ids": ["42"], "capture_mode": "explicit"},
        metadata={"title": "Example article"},
    )

    response = client.get("/api/v1/companion/activity?limit=10")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    assert payload["pagination"]["total"] == 1
    assert payload["pagination"]["limit"] == 10
    assert payload["pagination"]["offset"] == 0
    assert payload["pagination"]["has_more"] is False
    assert payload["pagination"]["next_offset"] is None
    assert payload["has_more"] is False
    assert payload["next_offset"] is None
    assert len(payload["items"]) == 1
    assert payload["items"][0]["event_type"] == "reading.saved"
    assert payload["items"][0]["provenance"]["capture_mode"] == "explicit"
    assert payload["items"][0]["metadata"]["title"] == "Example article"


def test_companion_knowledge_endpoint_returns_cards(client_with_companion_db) -> None:
    client, db = client_with_companion_db
    db.upsert_companion_knowledge_card(
        user_id="1",
        card_type="project_focus",
        title="Current focus",
        summary="Recent explicit activity clusters around 'project-alpha'.",
        evidence=[{"source_id": "42"}],
        score=0.9,
    )

    response = client.get("/api/v1/companion/knowledge")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["card_type"] == "project_focus"
    assert payload["items"][0]["evidence"] == [{"source_id": "42"}]


def test_companion_knowledge_detail_returns_evidence_rows(client_with_companion_db) -> None:
    client, db = client_with_companion_db
    event_id = db.insert_companion_activity_event(
        user_id="1",
        event_type="reading.saved",
        source_type="reading_item",
        source_id="42",
        surface="reading",
        dedupe_key="reading.saved:reading_item:42",
        tags=["research", "project-alpha"],
        provenance={"source_ids": ["42"], "capture_mode": "explicit"},
        metadata={"title": "Example article"},
    )
    goal_id = db.create_companion_goal(
        user_id="1",
        title="Review alpha notes",
        description="Follow up on the saved alpha reading.",
        goal_type="manual",
        config={},
        progress={},
        origin_kind="manual",
        progress_mode="computed",
        evidence=[{"event_id": event_id}],
        status="active",
    )
    card_id = db.upsert_companion_knowledge_card(
        user_id="1",
        card_type="project_focus",
        title="Current focus",
        summary="Recent explicit activity clusters around 'project-alpha'.",
        evidence=[{"event_id": event_id}, {"goal_id": goal_id}],
        score=0.9,
    )

    response = client.get(f"/api/v1/companion/knowledge/{card_id}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == card_id
    assert payload["evidence_events"][0]["id"] == event_id
    assert payload["evidence_goals"][0]["id"] == goal_id


def test_companion_activity_create_records_explicit_capture(client_with_companion_db) -> None:
    client, db = client_with_companion_db

    response = client.post(
        "/api/v1/companion/activity",
        json={
            "event_type": "extension.selection_saved",
            "source_type": "browser_selection",
            "source_id": "capture-1",
            "surface": "extension.sidepanel",
            "dedupe_key": "extension.selection_saved:capture-1",
            "tags": ["extension", "selection"],
            "provenance": {
                "capture_mode": "explicit",
                "route": "extension.context_menu",
                "action": "save_selection",
            },
            "metadata": {
                "selection": "Remember this paragraph.",
                "page_url": "https://example.com/article",
                "page_title": "Example article",
            },
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["event_type"] == "extension.selection_saved"
    assert payload["source_id"] == "capture-1"
    assert payload["provenance"]["capture_mode"] == "explicit"
    assert payload["metadata"]["selection"] == "Remember this paragraph."

    rows, total = db.list_companion_activity_events("1", limit=10, offset=0)
    assert total == 1
    assert rows[0]["source_id"] == "capture-1"
    assert rows[0]["provenance"]["action"] == "save_selection"


def test_companion_activity_create_rejects_missing_provenance(
    client_with_companion_db,
) -> None:
    client, _db = client_with_companion_db

    response = client.post(
        "/api/v1/companion/activity",
        json={
            "event_type": "extension.selection_saved",
            "source_type": "browser_selection",
            "source_id": "capture-1",
            "surface": "extension.sidepanel",
            "metadata": {
                "selection": "Remember this paragraph.",
            },
        },
    )

    assert response.status_code == 422, response.text


def test_companion_check_in_create_records_explicit_capture(client_with_companion_db) -> None:
    client, db = client_with_companion_db

    response = client.post(
        "/api/v1/companion/check-ins",
        json={
            "title": "Morning reset",
            "summary": "Re-focused on the companion capture backlog before lunch.",
            "tags": ["planning", "focus"],
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["event_type"] == "companion_check_in_recorded"
    assert payload["source_type"] == "companion_check_in"
    assert payload["surface"] == "companion.workspace"
    assert payload["metadata"]["title"] == "Morning reset"
    assert payload["metadata"]["summary"] == "Re-focused on the companion capture backlog before lunch."
    assert payload["provenance"]["route"] == "/api/v1/companion/check-ins"
    assert payload["provenance"]["action"] == "manual_check_in"

    rows, total = db.list_companion_activity_events("1", limit=10, offset=0)
    assert total == 1
    assert rows[0]["event_type"] == "companion_check_in_recorded"
    assert rows[0]["tags"] == ["planning", "focus"]


def test_companion_check_in_create_accepts_surface_override(client_with_companion_db) -> None:
    client, db = client_with_companion_db

    response = client.post(
        "/api/v1/companion/check-ins",
        json={
            "summary": "Logged a quick update from the persona sidepanel.",
            "surface": "persona.sidepanel",
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["surface"] == "persona.sidepanel"
    assert payload["metadata"]["summary"] == "Logged a quick update from the persona sidepanel."
    assert payload["provenance"]["route"] == "/api/v1/companion/check-ins"

    rows, total = db.list_companion_activity_events("1", limit=10, offset=0)
    assert total == 1
    assert rows[0]["surface"] == "persona.sidepanel"


def test_companion_reflection_detail_returns_provenance_and_evidence(client_with_companion_db) -> None:
    client, db = client_with_companion_db
    source_event_id = db.insert_companion_activity_event(
        user_id="1",
        event_type="reading.saved",
        source_type="reading_item",
        source_id="42",
        surface="reading",
        dedupe_key="reading.saved:reading_item:42",
        tags=["research", "project-alpha"],
        provenance={"source_ids": ["42"], "capture_mode": "explicit"},
        metadata={"title": "Example article"},
    )
    goal_id = db.create_companion_goal(
        user_id="1",
        title="Review alpha notes",
        description="Follow up on the saved alpha reading.",
        goal_type="manual",
        config={},
        progress={},
        origin_kind="manual",
        progress_mode="computed",
        evidence=[{"event_id": source_event_id}],
        status="active",
    )
    card_id = db.upsert_companion_knowledge_card(
        user_id="1",
        card_type="project_focus",
        title="Current focus",
        summary="Recent explicit activity clusters around 'project-alpha'.",
        evidence=[{"event_id": source_event_id}],
        score=0.9,
    )
    reflection_id = db.insert_companion_activity_event(
        user_id="1",
        event_type="companion_reflection_generated",
        source_type="companion_reflection",
        source_id="2026-03-10",
        surface="jobs.companion",
        dedupe_key="companion.reflection:daily:2026-03-10",
        provenance={
            "capture_mode": "explicit",
            "source_event_ids": [source_event_id],
            "knowledge_card_ids": [card_id],
            "goal_ids": [goal_id],
        },
        metadata={
            "title": "Daily reflection",
            "summary": "Existing reflection",
            "cadence": "daily",
            "delivery_decision": "delivered",
            "delivery_reason": "meaningful_signal",
            "theme_key": "project-alpha",
            "signal_strength": 3.0,
            "follow_up_prompts": [
                {
                    "prompt_id": "prompt-1",
                    "label": "Next concrete step",
                    "prompt_text": "What is the next concrete step for project alpha?",
                    "prompt_type": "clarify_priority",
                    "source_reflection_id": "reflection-1",
                    "source_evidence_ids": [source_event_id, card_id, goal_id],
                }
            ],
            "evidence": [
                {"kind": "knowledge_card", "card_id": card_id},
                {"kind": "goal", "goal_id": goal_id},
                {"kind": "activity_event", "source_event_id": source_event_id},
            ],
        },
    )

    response = client.get(f"/api/v1/companion/reflections/{reflection_id}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == reflection_id
    assert payload["provenance"]["source_event_ids"] == [source_event_id]
    assert payload["knowledge_cards"][0]["id"] == card_id
    assert payload["goals"][0]["id"] == goal_id
    assert payload["activity_events"][0]["id"] == source_event_id
    assert payload["delivery_decision"] == "delivered"
    assert payload["delivery_reason"] == "meaningful_signal"
    assert payload["theme_key"] == "project-alpha"
    assert payload["signal_strength"] == 3.0
    assert payload["follow_up_prompts"][0]["prompt_text"] == "What is the next concrete step for project alpha?"


def test_companion_conversation_prompts_endpoint_returns_at_most_three_prompts(
    client_with_companion_db,
) -> None:
    client, db = client_with_companion_db
    source_event_id = db.insert_companion_activity_event(
        user_id="1",
        event_type="note_updated",
        source_type="note",
        source_id="42",
        surface="api.notes",
        dedupe_key="note_updated:42",
        tags=["backlog", "review"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/notes/42"},
        metadata={"title": "Backlog review notes"},
    )
    reflection_id = db.insert_companion_activity_event(
        user_id="1",
        event_type="companion_reflection_generated",
        source_type="companion_reflection",
        source_id="2026-03-10",
        surface="jobs.companion",
        dedupe_key="companion.reflection:daily:2026-03-10",
        tags=["backlog-review"],
        provenance={"capture_mode": "explicit", "source_event_ids": [source_event_id]},
        metadata={
            "title": "Daily reflection",
            "summary": "You returned to backlog review and still need a concrete next step.",
            "cadence": "daily",
            "delivery_decision": "delivered",
            "delivery_reason": "meaningful_signal",
            "theme_key": "backlog-review",
            "signal_strength": 4.0,
            "follow_up_prompts": [
                {
                    "prompt_id": "prompt-1",
                    "label": "Choose next step",
                    "prompt_text": "What is the next concrete step for backlog review?",
                    "prompt_type": "clarify_priority",
                    "source_reflection_id": "reflection-1",
                    "source_evidence_ids": [source_event_id],
                },
                {
                    "prompt_id": "prompt-2",
                    "label": "Check blockers",
                    "prompt_text": "What is blocking backlog review right now?",
                    "prompt_type": "identify_blocker",
                    "source_reflection_id": "reflection-1",
                    "source_evidence_ids": [source_event_id],
                },
                {
                    "prompt_id": "prompt-3",
                    "label": "Define done",
                    "prompt_text": "What would make backlog review feel complete today?",
                    "prompt_type": "define_outcome",
                    "source_reflection_id": "reflection-1",
                    "source_evidence_ids": [source_event_id],
                },
                {
                    "prompt_id": "prompt-4",
                    "label": "Trim scope",
                    "prompt_text": "What can you defer to make backlog review smaller?",
                    "prompt_type": "trim_scope",
                    "source_reflection_id": "reflection-1",
                    "source_evidence_ids": [source_event_id],
                },
            ],
            "evidence": [{"kind": "activity_event", "source_event_id": source_event_id}],
        },
    )

    response = client.get(
        "/api/v1/companion/conversation-prompts",
        params={"query": "resume backlog review"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["prompt_source_kind"] == "reflection"
    assert payload["prompt_source_id"] == reflection_id
    assert len(payload["prompts"]) <= 3


def test_companion_conversation_prompts_endpoint_falls_back_to_ranked_context(
    client_with_companion_db,
) -> None:
    client, db = client_with_companion_db
    event_id = db.insert_companion_activity_event(
        user_id="1",
        event_type="note_updated",
        source_type="note",
        source_id="42",
        surface="api.notes",
        dedupe_key="note_updated:42",
        tags=["backlog", "review"],
        provenance={"capture_mode": "explicit", "route": "/api/v1/notes/42"},
        metadata={"title": "Backlog review notes"},
    )
    card_id = db.upsert_companion_knowledge_card(
        user_id="1",
        card_type="project_focus",
        title="Backlog review",
        summary="Recent explicit activity clusters around backlog review.",
        evidence=[{"source_event_id": event_id}],
        score=0.9,
        status="active",
    )

    response = client.get(
        "/api/v1/companion/conversation-prompts",
        params={"query": "resume backlog review"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["prompt_source_kind"] == "knowledge_card"
    assert payload["prompt_source_id"] == card_id
    assert payload["prompts"][0]["prompt_text"]


def test_companion_goals_create_and_list(client_with_companion_db) -> None:
    client, _db = client_with_companion_db

    create_response = client.post(
        "/api/v1/companion/goals",
        json={
            "title": "Finish reading queue",
            "description": "Read 3 saved papers this week",
            "goal_type": "reading_backlog",
            "config": {"target_count": 3},
            "progress": {"completed_count": 0},
        },
    )

    assert create_response.status_code == 201, create_response.text
    created = create_response.json()
    assert created["title"] == "Finish reading queue"
    assert created["goal_type"] == "reading_backlog"
    assert created["config"] == {"target_count": 3}

    list_response = client.get("/api/v1/companion/goals")

    assert list_response.status_code == 200, list_response.text
    payload = list_response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["id"] == created["id"]


def test_companion_goal_patch_updates_fields(client_with_companion_db) -> None:
    client, db = client_with_companion_db
    goal_id = db.create_companion_goal(
        user_id="1",
        title="Finish reading queue",
        description="Read 3 saved papers this week",
        goal_type="reading_backlog",
        config={"target_count": 3},
        progress={"completed_count": 0},
    )

    response = client.patch(
        f"/api/v1/companion/goals/{goal_id}",
        json={
            "title": "Shrink reading queue",
            "progress": {"completed_count": 1},
            "status": "paused",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == goal_id
    assert payload["title"] == "Shrink reading queue"
    assert payload["progress"] == {"completed_count": 1}
    assert payload["status"] == "paused"


def test_companion_goal_patch_rejects_null_for_non_nullable_fields(client_with_companion_db) -> None:
    client, db = client_with_companion_db
    goal_id = db.create_companion_goal(
        user_id="1",
        title="Finish reading queue",
        description="Read 3 saved papers this week",
        goal_type="reading_backlog",
        config={"target_count": 3},
        progress={"completed_count": 0},
    )

    response = client.patch(
        f"/api/v1/companion/goals/{goal_id}",
        json={"status": None},
    )

    assert response.status_code == 422, response.text


def test_companion_purge_endpoint_removes_linked_reflection_notifications(
    client_with_companion_db,
) -> None:
    client, db = client_with_companion_db
    collections_db = CollectionsDatabase.for_user(user_id=1)

    def override_collections_db():
        return collections_db

    reflection_id = db.insert_companion_activity_event(
        user_id="1",
        event_type="companion_reflection_generated",
        source_type="companion_reflection",
        source_id="2026-03-10",
        surface="jobs.companion",
        dedupe_key="companion.reflection:daily:2026-03-10",
        provenance={"capture_mode": "explicit"},
        metadata={"title": "Daily reflection", "summary": "Existing reflection"},
    )
    collections_db.create_user_notification(
        kind="companion_reflection",
        title="Daily reflection",
        message="Existing reflection",
        severity="info",
        source_job_id="501",
        source_domain="companion",
        source_job_type="companion_reflection",
        link_type="companion_reflection",
        link_id=reflection_id,
        dedupe_key="companion_reflection:daily:2026-03-10",
    )
    fastapi_app.dependency_overrides[get_collections_db_for_user] = override_collections_db
    try:
        response = client.post("/api/v1/companion/purge", json={"scope": "reflections"})
    finally:
        fastapi_app.dependency_overrides.pop(get_collections_db_for_user, None)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["deleted_counts"]["reflections"] == 1
    assert payload["deleted_counts"]["notifications"] == 1


def test_companion_rebuild_endpoint_queues_job(client_with_companion_db) -> None:
    client, _db = client_with_companion_db
    job_manager = JobManager()

    def override_job_manager():
        return job_manager

    fastapi_app.dependency_overrides[get_job_manager] = override_job_manager
    try:
        response = client.post("/api/v1/companion/rebuild", json={"scope": "knowledge"})
    finally:
        fastapi_app.dependency_overrides.pop(get_job_manager, None)

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["scope"] == "knowledge"
    assert payload["status"] == "queued"
    assert payload["job_id"] is not None
    assert payload["job_uuid"]


def test_companion_routes_include_rbac_rate_limits() -> None:
    route_resources = {
        (route.path, next(iter(sorted(route.methods or [])))): [
            getattr(dependency.call, "_tldw_rate_limit_resource", None)
            for dependency in route.dependant.dependencies
        ]
        for route in companion_ep.router.routes
        if getattr(route, "path", "").startswith("/")
    }

    assert "companion.activity.create" in route_resources[("/activity", "POST")]
    assert "companion.activity.read" in route_resources[("/activity", "GET")]
    assert "companion.checkins.create" in route_resources[("/check-ins", "POST")]
    assert "companion.knowledge.read" in route_resources[("/knowledge", "GET")]
    assert "companion.goals.read" in route_resources[("/goals", "GET")]
    assert "companion.goals.create" in route_resources[("/goals", "POST")]
    assert "companion.goals.update" in route_resources[("/goals/{goal_id}", "PATCH")]
    assert "companion.lifecycle.purge" in route_resources[("/purge", "POST")]
    assert "companion.lifecycle.rebuild" in route_resources[("/rebuild", "POST")]


def test_companion_activity_create_offloads_db_and_usage_logging(
    client_with_companion_db,
    monkeypatch,
) -> None:
    client, _db = client_with_companion_db
    offloaded_calls: list[str] = []

    async def _fake_to_thread(func, *args, **kwargs):
        offloaded_calls.append(getattr(func, "__name__", str(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(
        companion_ep,
        "asyncio",
        SimpleNamespace(to_thread=_fake_to_thread),
        raising=False,
    )

    response = client.post(
        "/api/v1/companion/activity",
        json={
            "event_type": "extension.selection_saved",
            "source_type": "browser_selection",
            "source_id": "capture-1",
            "surface": "extension.sidepanel",
            "dedupe_key": "extension.selection_saved:capture-1",
            "provenance": {
                "capture_mode": "explicit",
                "route": "extension.context_menu",
                "action": "save_selection",
            },
        },
    )

    assert response.status_code == 201, response.text
    assert any("_ensure_companion_opt_in" in name for name in offloaded_calls)
    assert any("insert_companion_activity_event" in name for name in offloaded_calls)
    assert any("log_event" in name for name in offloaded_calls)


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/api/v1/companion/activity", None),
        (
            "POST",
            "/api/v1/companion/activity",
            {
                "event_type": "extension.selection_saved",
                "source_type": "browser_selection",
                "source_id": "capture-1",
                "surface": "extension.sidepanel",
                "provenance": {
                    "capture_mode": "explicit",
                    "route": "extension.context_menu",
                    "action": "save_selection",
                },
            },
        ),
        (
            "POST",
            "/api/v1/companion/check-ins",
            {
                "summary": "Tried to save without explicit consent.",
            },
        ),
        ("GET", "/api/v1/companion/knowledge", None),
    ],
)
def test_companion_endpoints_require_personalization_opt_in(
    client_with_companion_db_opted_out,
    method: str,
    path: str,
    payload: dict | None,
) -> None:
    client, _db = client_with_companion_db_opted_out

    response = client.request(method, path, json=payload)

    assert response.status_code == 409, response.text
    assert response.json() == {
        "detail": "Enable personalization before using companion."
    }
