from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import (
    User,
    get_request_user,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.schemas.persona import PersonaVoiceAnalyticsResponse
from tldw_Server_API.app.core.VoiceAssistant import (
    IntentParser,
    VoiceCommandRegistry,
    VoiceCommandRouter,
    VoiceSessionManager,
)
from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionResult, ActionType


pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


def _client_for_user(user_id: int, db: CharactersRAGDB):
    async def override_user():
        return User(
            id=user_id,
            username=f"persona-user-{user_id}",
            email=None,
            is_active=True,
        )

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    return TestClient(fastapi_app)


@pytest.fixture()
def persona_db(tmp_path):
    db = CharactersRAGDB(
        str(tmp_path / "persona_voice_analytics_api.db"),
        client_id="persona-voice-analytics-api-tests",
    )
    yield db
    db.close_connection()


def _create_persona(client: TestClient, *, name: str) -> str:
    response = client.post(
        "/api/v1/persona/profiles",
        json={"name": name, "mode": "persistent_scoped"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _create_router(monkeypatch: pytest.MonkeyPatch) -> VoiceCommandRouter:
    registry = VoiceCommandRegistry()
    registry.load_defaults()
    session_manager = VoiceSessionManager()
    parser = IntentParser(registry=registry, llm_enabled=False)
    router = VoiceCommandRouter(
        registry=registry,
        parser=parser,
        session_manager=session_manager,
    )

    async def mock_execute_mcp_tool(intent, session):
        return ActionResult(
            success=True,
            action_type=ActionType.MCP_TOOL,
            result_data={"tool_name": intent.action_config.get("tool_name", "notes.search")},
            response_text="Executed tool",
        )

    async def mock_execute_llm_chat(intent, session):
        return ActionResult(
            success=True,
            action_type=ActionType.LLM_CHAT,
            result_data={"response": f"Planner handled: {intent.raw_text}"},
            response_text=f"Planner handled: {intent.raw_text}",
        )

    monkeypatch.setattr(router, "_execute_mcp_tool", mock_execute_mcp_tool)
    monkeypatch.setattr(router, "_execute_llm_chat", mock_execute_llm_chat)
    return router


@pytest.mark.asyncio
async def test_persona_voice_analytics_tracks_live_events_for_selected_persona(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    with _client_for_user(1, persona_db) as client:
        persona_a = _create_persona(client, name="Analytics Persona A")
        persona_b = _create_persona(client, name="Analytics Persona B")

        created_a = client.post(
            f"/api/v1/persona/profiles/{persona_a}/voice-commands",
            json={
                "name": "Search Notes",
                "phrases": ["search notes for {topic}"],
                "action_type": "mcp_tool",
                "action_config": {"tool_name": "notes.search"},
                "priority": 10,
                "enabled": True,
                "requires_confirmation": False,
            },
        )
        assert created_a.status_code == 201, created_a.text
        command_a_id = created_a.json()["id"]

        created_b = client.post(
            f"/api/v1/persona/profiles/{persona_b}/voice-commands",
            json={
                "name": "Search Tasks",
                "phrases": ["search tasks for {topic}"],
                "action_type": "mcp_tool",
                "action_config": {"tool_name": "tasks.search"},
                "priority": 10,
                "enabled": True,
                "requires_confirmation": False,
            },
        )
        assert created_b.status_code == 201, created_b.text

        router = _create_router(monkeypatch)

        await router.process_command(
            text="search notes for vector databases",
            user_id=1,
            db=persona_db,
            persona_id=persona_a,
        )
        await router.process_command(
            text="compose a limerick about rainbows",
            user_id=1,
            db=persona_db,
            persona_id=persona_a,
        )
        await router.process_command(
            text="search tasks for today's standup",
            user_id=1,
            db=persona_db,
            persona_id=persona_b,
        )
        await router.process_command(
            text="compose a generic answer",
            user_id=1,
            db=persona_db,
            persona_id=None,
        )

        rows = persona_db.execute_query(
            """
            SELECT persona_id, resolution_type, command_id
            FROM voice_command_events
            WHERE persona_id IS NOT NULL
            ORDER BY id ASC
            """
        ).fetchall()
        normalized_rows = [dict(row) for row in rows]
        assert normalized_rows[0]["persona_id"] == persona_a
        assert normalized_rows[0]["resolution_type"] == "direct_command"
        assert normalized_rows[0]["command_id"] == command_a_id
        assert normalized_rows[1]["persona_id"] == persona_a
        assert normalized_rows[1]["resolution_type"] == "planner_fallback"
        assert normalized_rows[1]["command_id"] is None
        assert normalized_rows[2]["persona_id"] == persona_b
        assert normalized_rows[2]["resolution_type"] == "direct_command"

        analytics = client.get(
            f"/api/v1/persona/profiles/{persona_a}/voice-analytics",
            params={"days": 7},
        )
        assert analytics.status_code == 200, analytics.text
        payload = analytics.json()
        assert payload["persona_id"] == persona_a
        assert payload["summary"]["total_events"] == 2
        assert payload["summary"]["direct_command_count"] == 1
        assert payload["summary"]["planner_fallback_count"] == 1
        assert payload["summary"]["success_rate"] == 1.0
        assert payload["summary"]["fallback_rate"] == 0.5
        assert payload["live_voice"] == {
            "total_committed_turns": 0,
            "vad_auto_commit_count": 0,
            "manual_commit_count": 0,
            "vad_auto_rate": 0.0,
            "manual_commit_rate": 0.0,
            "degraded_session_count": 0,
        }
        assert payload["commands"] == [
            {
                "command_id": command_a_id,
                "command_name": "Search Notes",
                "total_invocations": 1,
                "success_count": 1,
                "error_count": 0,
                "avg_response_time_ms": payload["commands"][0]["avg_response_time_ms"],
                "last_used": payload["commands"][0]["last_used"],
            }
        ]
        assert payload["fallbacks"]["total_invocations"] == 1
        assert payload["fallbacks"]["success_count"] == 1
        assert payload["fallbacks"]["error_count"] == 0

    fastapi_app.dependency_overrides.clear()


def test_persona_command_dry_run_does_not_increment_voice_analytics(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Dry Run Analytics Persona")

        created = client.post(
            f"/api/v1/persona/profiles/{persona_id}/voice-commands",
            json={
                "name": "Search Notes",
                "phrases": ["search notes for {topic}"],
                "action_type": "mcp_tool",
                "action_config": {"tool_name": "notes.search"},
                "priority": 10,
                "enabled": True,
                "requires_confirmation": False,
            },
        )
        assert created.status_code == 201, created.text

        dry_run = client.post(
            f"/api/v1/persona/profiles/{persona_id}/voice-commands/test",
            json={"heard_text": "search notes for vector databases"},
        )
        assert dry_run.status_code == 200, dry_run.text
        assert dry_run.json()["matched"] is True

        analytics = client.get(
            f"/api/v1/persona/profiles/{persona_id}/voice-analytics",
            params={"days": 7},
        )
        assert analytics.status_code == 200, analytics.text
        payload = analytics.json()
        assert payload["summary"]["total_events"] == 0
        assert payload["summary"]["direct_command_count"] == 0
        assert payload["summary"]["planner_fallback_count"] == 0
        assert payload["live_voice"] == {
            "total_committed_turns": 0,
            "vad_auto_commit_count": 0,
            "manual_commit_count": 0,
            "vad_auto_rate": 0.0,
            "manual_commit_rate": 0.0,
            "degraded_session_count": 0,
        }
        assert payload["commands"] == []
        assert payload["fallbacks"]["total_invocations"] == 0

    fastapi_app.dependency_overrides.clear()


def test_persona_voice_analytics_includes_live_voice_commit_and_degraded_metrics(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Live Voice Metrics Persona")

        persona_db.execute_query(
            """
            INSERT INTO persona_live_voice_events (
                user_id, persona_id, session_id, event_type, commit_source
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (1, persona_id, "sess-auto", "commit", "vad_auto"),
            commit=True,
        )
        persona_db.execute_query(
            """
            INSERT INTO persona_live_voice_events (
                user_id, persona_id, session_id, event_type, commit_source
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (1, persona_id, "sess-manual", "commit", "manual"),
            commit=True,
        )
        persona_db.execute_query(
            """
            INSERT INTO persona_live_voice_events (
                user_id, persona_id, session_id, event_type, commit_source
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (1, persona_id, "sess-manual", "manual_mode_required", None),
            commit=True,
        )

        analytics = client.get(
            f"/api/v1/persona/profiles/{persona_id}/voice-analytics",
            params={"days": 7},
        )
        assert analytics.status_code == 200, analytics.text
        payload = analytics.json()
        assert payload["live_voice"] == {
            "total_committed_turns": 2,
            "vad_auto_commit_count": 1,
            "manual_commit_count": 1,
            "vad_auto_rate": 0.5,
            "manual_commit_rate": 0.5,
            "degraded_session_count": 1,
        }

    fastapi_app.dependency_overrides.clear()


def test_persona_voice_analytics_returns_recent_live_sessions_from_summary_store(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Recent Session Summary Persona")
        started_at = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ended_at = (
            datetime.now(timezone.utc) - timedelta(days=1) + timedelta(minutes=5)
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        persona_db.execute_query(
            """
            INSERT INTO persona_live_voice_session_summaries (
                user_id, persona_id, session_id, created_at, updated_at, started_at, ended_at,
                auto_commit_enabled, vad_threshold, min_silence_ms, turn_stop_secs,
                min_utterance_secs, turn_detection_changed_during_session,
                total_committed_turns, vad_auto_commit_count, manual_commit_count,
                manual_mode_required_count, text_only_tts_count,
                listening_recovery_count, thinking_recovery_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                persona_id,
                "sess-summary-1",
                started_at,
                ended_at,
                started_at,
                ended_at,
                1,
                0.5,
                250,
                0.2,
                0.4,
                0,
                4,
                3,
                1,
                0,
                1,
                0,
                2,
            ),
            commit=True,
        )

        analytics = client.get(
            f"/api/v1/persona/profiles/{persona_id}/voice-analytics",
            params={"days": 7},
        )
        assert analytics.status_code == 200, analytics.text
        payload = analytics.json()
        assert payload["recent_live_sessions"] == [
            {
                "session_id": "sess-summary-1",
                "started_at": started_at,
                "ended_at": ended_at,
                "auto_commit_enabled": True,
                "vad_threshold": 0.5,
                "min_silence_ms": 250,
                "turn_stop_secs": 0.2,
                "min_utterance_secs": 0.4,
                "turn_detection_changed_during_session": False,
                "total_committed_turns": 4,
                "vad_auto_commit_count": 3,
                "manual_commit_count": 1,
                "manual_mode_required_count": 0,
                "text_only_tts_count": 1,
                "listening_recovery_count": 0,
                "thinking_recovery_count": 2,
            }
        ]

    fastapi_app.dependency_overrides.clear()


def test_persona_voice_analytics_includes_empty_recent_live_sessions_list(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Empty Recent Sessions Persona")

        analytics = client.get(
            f"/api/v1/persona/profiles/{persona_id}/voice-analytics",
            params={"days": 7},
        )
        assert analytics.status_code == 200, analytics.text
        payload = analytics.json()
        assert payload["recent_live_sessions"] == []

    fastapi_app.dependency_overrides.clear()


def test_persona_live_voice_session_update_accepts_recovery_counts(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Recovery Flush Persona")

        update = client.put(
            f"/api/v1/persona/profiles/{persona_id}/voice-analytics/live-sessions/sess-flush-1",
            json={
                "listening_recovery_count": 2,
                "thinking_recovery_count": 1,
                "finalize": True,
            },
        )
        assert update.status_code == 200, update.text
        payload = update.json()
        assert payload["session_id"] == "sess-flush-1"
        assert payload["listening_recovery_count"] == 2
        assert payload["thinking_recovery_count"] == 1
        assert payload["ended_at"]

        row = persona_db.execute_query(
            """
            SELECT session_id, listening_recovery_count, thinking_recovery_count, ended_at
            FROM persona_live_voice_session_summaries
            WHERE user_id = ? AND persona_id = ? AND session_id = ?
            """,
            (1, persona_id, "sess-flush-1"),
        ).fetchone()
        assert dict(row) == {
            "session_id": "sess-flush-1",
            "listening_recovery_count": 2,
            "thinking_recovery_count": 1,
            "ended_at": dict(row)["ended_at"],
        }
        assert dict(row)["ended_at"]

    fastapi_app.dependency_overrides.clear()


def test_persona_live_voice_session_update_is_idempotent(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Recovery Flush Idempotent Persona")

        first = client.put(
            f"/api/v1/persona/profiles/{persona_id}/voice-analytics/live-sessions/sess-flush-2",
            json={
                "listening_recovery_count": 1,
                "thinking_recovery_count": 0,
            },
        )
        assert first.status_code == 200, first.text

        second = client.put(
            f"/api/v1/persona/profiles/{persona_id}/voice-analytics/live-sessions/sess-flush-2",
            json={
                "listening_recovery_count": 3,
                "thinking_recovery_count": 4,
                "finalize": True,
            },
        )
        assert second.status_code == 200, second.text

        rows = persona_db.execute_query(
            """
            SELECT session_id, listening_recovery_count, thinking_recovery_count, ended_at
            FROM persona_live_voice_session_summaries
            WHERE user_id = ? AND persona_id = ? AND session_id = ?
            """,
            (1, persona_id, "sess-flush-2"),
        ).fetchall()
        assert len(rows) == 1
        normalized = dict(rows[0])
        assert normalized["session_id"] == "sess-flush-2"
        assert normalized["listening_recovery_count"] == 3
        assert normalized["thinking_recovery_count"] == 4
        assert normalized["ended_at"]

    fastapi_app.dependency_overrides.clear()


def test_persona_voice_analytics_response_model_accepts_recent_live_session_snapshot_fields():
    payload = PersonaVoiceAnalyticsResponse.model_validate(
        {
            "persona_id": "persona-feedback",
            "summary": {
                "total_events": 0,
                "direct_command_count": 0,
                "planner_fallback_count": 0,
                "success_rate": 0.0,
                "fallback_rate": 0.0,
                "avg_response_time_ms": 0.0,
            },
            "live_voice": {
                "total_committed_turns": 2,
                "vad_auto_commit_count": 1,
                "manual_commit_count": 1,
                "vad_auto_rate": 0.5,
                "manual_commit_rate": 0.5,
                "degraded_session_count": 1,
            },
            "recent_live_sessions": [
                {
                    "session_id": "sess-123",
                    "started_at": "2026-03-13T12:00:00Z",
                    "ended_at": "2026-03-13T12:05:00Z",
                    "auto_commit_enabled": True,
                    "vad_threshold": 0.5,
                    "min_silence_ms": 250,
                    "turn_stop_secs": 0.2,
                    "min_utterance_secs": 0.4,
                    "turn_detection_changed_during_session": False,
                    "total_committed_turns": 4,
                    "vad_auto_commit_count": 3,
                    "manual_commit_count": 1,
                    "manual_mode_required_count": 0,
                    "text_only_tts_count": 1,
                    "listening_recovery_count": 0,
                    "thinking_recovery_count": 1,
                }
            ],
            "commands": [],
            "fallbacks": {
                "total_invocations": 0,
                "success_count": 0,
                "error_count": 0,
                "avg_response_time_ms": 0.0,
                "last_used": None,
            },
        }
    )
    assert payload.recent_live_sessions[0].session_id == "sess-123"
    assert payload.recent_live_sessions[0].auto_commit_enabled is True
    assert payload.recent_live_sessions[0].thinking_recovery_count == 1


def test_persona_voice_analytics_routes_include_rate_limit_dependency():
    expected_routes = {
        ("/api/v1/persona/profiles/{persona_id}/voice-analytics", "GET"),
        (
            "/api/v1/persona/profiles/{persona_id}/voice-analytics/live-sessions/{session_id}",
            "PUT",
        ),
    }

    seen_routes: set[tuple[str, str]] = set()
    for route in iter_served_routes(fastapi_app.routes):
        if not isinstance(route.route, APIRoute):
            continue
        for method in route.methods:
            key = (route.path, method)
            if key not in expected_routes:
                continue
            seen_routes.add(key)
            dependencies = [dependency.call for dependency in route.dependant.dependencies]
            assert check_rate_limit in dependencies, key

    assert seen_routes == expected_routes
